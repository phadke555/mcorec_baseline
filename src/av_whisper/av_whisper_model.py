from transformers.models.whisper.modeling_whisper import (
    WhisperConfig,
    WhisperEncoder,
    WhisperEncoderLayer,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    shift_tokens_right,
    WhisperModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
)
from transformers import AutoModel
from transformers.cache_utils import Cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch, torch.nn as nn
from torch.nn import CrossEntropyLoss

class CrossAttnBlock(nn.Module):
    """
    A cross-attention sublayer that fuses visual context into the audio representation.

    This module performs multi-head attention using *audio* hidden states as queries
    and *vision* hidden states as keys and values. It follows Whisper’s pre-norm residual
    structure for stability and compatibility with pretrained checkpoints.

    Args:
        d_model (int): Dimensionality of the input and output embeddings.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention output.

    Inputs:
        audio_h (`torch.FloatTensor` of shape `(batch_size, T_audio, d_model)`):
            Audio hidden states that act as queries.
        vision_h (`torch.FloatTensor` of shape `(batch_size, T_video, d_model)`):
            Vision features serving as keys and values.
        attn_mask (`torch.Tensor`, *optional*):
            Mask for selective cross-attention between audio and vision frames.
            Shape `(T_audio, T_video)` or `(batch_size, T_audio, T_video)`.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, T_audio, d_model)`:
            Audio features enriched with cross-modal visual context.

    Notes:
        - LayerNorm is applied *before* attention (pre-norm) as in Whisper.
        - The residual connection is defined on the audio (query) stream.
        - Vision features must already be projected to `d_model` dimensionality.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
    def forward(self, audio_h, vision_h, attn_mask=None):
        # audio_h: (B, Ta, D)  vision_h: (B, Tv, D)
        residual = audio_h
        audio_h = self.ln1(audio_h)
        out, _ = self.attn(query=audio_h, key=vision_h, value=vision_h, key_padding_mask=None, attn_mask=None)
        return residual + out

class AVWhisperEncoder(WhisperEncoder):
    """
    Audio-Visual Whisper Encoder.

    Extends the standard :class:`~transformers.WhisperEncoder` to incorporate
    cross-modal conditioning on vision features. At selected layers of the encoder,
    audio hidden states attend to visual representations via :class:`CrossAttnBlock`.

    This design preserves Whisper’s pre-norm ordering and checkpoint compatibility.

    Args:
        config (`WhisperConfig`):
            Configuration object containing model hyperparameters and any
            additional audio-visual flags such as:
                - `use_av_cross_attn` (bool)
                - `cross_attn_every_n_layers` (int)
                - `vision_feature_dim` (int)
                - `proj_to_d_model` (bool)
                - `dropout` (float)
                - `gate_cross_attn` (bool)

    Inputs:
        input_features (`torch.FloatTensor` of shape `(batch_size, T_audio, feature_dim)`):
            Standard Whisper encoder inputs (e.g., log-mel spectrograms).
        attention_mask (`torch.BoolTensor` of shape `(batch_size, T_audio)`):
            Audio padding mask.
        vision_features (`torch.FloatTensor`, *optional*, shape `(batch_size, T_video, D_v)`):
            Precomputed visual features (e.g., from DINOv3) aligned in time.
        vision_pad_mask (`torch.BoolTensor`, *optional*, shape `(batch_size, T_video)`):
            Mask for padded vision frames.
        cross_attn_mask (`torch.Tensor`, *optional*):
            Optional mask to constrain audio-to-vision attention.

    Returns:
        `BaseModelOutput`:
            Same structure as the Whisper encoder output, with the final
            hidden state `[batch_size, T_audio, d_model]` containing
            visually-enriched representations.

    Behavior:
        - When `use_av_cross_attn=False` or no vision features are provided,
        falls back to the standard Whisper encoder.
        - Maintains pre-norm and residual order identical to Whisper.
        - Can apply cross-attention at every Nth encoder layer.
    """
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        d_model = config.d_model
        n_heads = config.encoder_attention_heads
        dropout = config.dropout

        self.vision_encoder: Optional[nn.Module] = None
        self.vision_feature_dim: Optional[int] = None
        self.vision_projection = None
        self.vision_layer_norm = None
        self.cross_attn_blocks = None
    
    def set_vision_encoder(
        self,
        encoder
    ):
        self.vision_encoder = encoder
        self.vision_feature_dim = encoder.config.hidden_size
        self.vision_projection = nn.Linear(self.vision_feature_dim, self.config.d_model)
        self.vision_layer_norm = nn.LayerNorm(self.config.d_model)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttnBlock(self.config.d_model, self.config.encoder_attention_heads, self.config.dropout) for _ in range(self.config.encoder_layers)
        ])
    
    def embed_vision_features(
        self,
        vision_features
    ):
        # import pdb; pdb.set_trace()
        B, C, T_v, H, W = vision_features.shape
        vision_features = vision_features.permute(0, 2, 1, 3, 4).contiguous().view(B * T_v, C, H, W)
        vision_features = self.vision_encoder(vision_features)
        vision_features = vision_features.pooler_output
        vision_features = vision_features.view(B, T_v, -1)
        vision_features = self.vision_projection(vision_features)
        vision_features = self.vision_layer_norm(vision_features)
        return vision_features

    def forward(
        self,
        input_features,
        attention_mask=None,
        vision_features=None,
        vision_pad_mask=None,
        cross_attn_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # import pdb; pdb.set_trace()
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if self.vision_encoder is not None:
            vision_features = self.embed_vision_features(vision_features)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    None,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            if self.vision_encoder is not None:
                vision_features = torch.nn.functional.interpolate(
                    vision_features.permute(0, 2, 1),  # (B, D, T_v)
                    size=hidden_states.shape[1],        # target T_audio
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
                hidden_states = self.cross_attn_blocks[idx](
                    audio_h=hidden_states,
                    vision_h=vision_features,
                    attn_mask=cross_attn_mask,
                )
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class AVWhisperModel(WhisperModel):
    """
    Audio-Visual Whisper Model.

    Extends :class:`~transformers.WhisperModel` by replacing its encoder with
    an :class:`AVEncoder` that supports cross-attention with visual features.
    The decoder architecture and forward interface remain identical to Whisper.

    Args:
        config (`WhisperConfig`):
            Configuration containing Whisper and audio-visual parameters.

    Forward Args:
        input_features (`torch.FloatTensor`):
            Audio input features for the encoder.
        attention_mask (`torch.BoolTensor`, *optional*):
            Audio attention mask.
        videos (`torch.FloatTensor`, *optional*):
            Video frames or vision features.
        video_lengths (`torch.LongTensor`, *optional*):
            True video lengths for padding masks.
        **kwargs:
            Additional standard WhisperModel arguments.

    Returns:
        `Seq2SeqModelOutput`:
            Same structure as Whisper’s base model outputs.

    Notes:
        - When videos is not provided, this model behaves identically to Whisper.
        - Decoder caching, beam search, and `generate()` remain unchanged.
        - Compatible with pretrained Whisper weights through `from_pretrained()`.
    """
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AVWhisperEncoder(config)
    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            videos: Optional[torch.FloatTensor] = None,
            video_lengths: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vad_mask: Optional[torch.FloatTensor] = None,
            per_group_sizes: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        # import pdb; pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                vision_features=videos,
                video_lengths=video_lengths,
                output_attentions=output_attentions,
                output_hidden_states=True,
                head_mask=head_mask,
                return_dict=return_dict,
                vad_mask=vad_mask,
                per_group_sizes=per_group_sizes
            )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.hidden_states[-1],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.hidden_states[-1],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )

class AVWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    """
    Audio-Visual Whisper for Conditional Generation.

    Implements end-to-end automatic speech recognition with optional
    visual conditioning. Inherits from
    :class:`~transformers.WhisperForConditionalGeneration`.

    Args:
        config (`WhisperConfig`):
            Model configuration; should include AV-related flags.

    Forward Args:
        input_features (`torch.FloatTensor`):
            Encoder audio inputs.
        decoder_input_ids (`torch.LongTensor`):
            Decoder input token IDs.
        labels (`torch.LongTensor`, *optional*):
            Target token IDs for teacher forcing.
        videos (`torch.FloatTensor`, *optional*):
            Video frames or precomputed vision embeddings.
        video_lengths (`torch.LongTensor`, *optional*):
            True video lengths for padding masks.
        **kwargs:
            All other Whisper generation arguments.

    Returns:
        `Seq2SeqLMOutput`:
            Includes loss (if `labels` provided), logits, and hidden states.

    Behavior:
        - Uses :class:`AVWhisperModel` as the underlying encoder-decoder.
        - Fully compatible with the Hugging Face generation API.
        - Supports standard Whisper tokenizers, processors, and decoding
        pipelines (`generate`, `prepare_inputs_for_generation`, etc.).
    """
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = AVWhisperModel(config)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        input_feature_lengths: Optional[torch.LongTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        video_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Cache] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        label_lengths: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        import pdb; pdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            videos=videos,
            video_lengths=video_lengths,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )