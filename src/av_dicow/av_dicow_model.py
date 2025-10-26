"""
This file includes significant portions of code adapted from the public Target Speaker ASR
 repository by Alexander Polok et al.
The original implementation can be found at:
https://github.com/BUTSpeechFIT/TS-ASR-Whisper
"""

import copy
import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import Cache
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers import WhisperTimeStampLogitsProcessor
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    shift_tokens_right,
    WhisperModel
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer, WhisperAttention
from transformers.utils import logging

from torch import nn
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

import torch
from torch import nn

from typing import Optional
from transformers import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor, )
from transformers.generation.logits_process import WhisperNoSpeechDetection
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateNonBeamOutput, \
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateBeamOutput, GenerateBeamDecoderOnlyOutput, \
    GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
)
from transformers.utils import logging
from transformers import WhisperConfig
import math

class DiCoWConfig(WhisperConfig):
    """This is a modified version of the `WhisperEncoder` model from the `transformers` library.
    The model has been modified to support CTC loss computation in the forward pass."""
    model_type = "DiCoW"

    def __init__(
            self,
            ctc_loss_reduction: str = "mean",
            final_dropout: float = 0.0,
            ctc_zero_infinity: bool = False,
            ctc_weight: float = 0.0,
            blank_token_id: Optional[int] = None,
            additional_layer: bool = False,
            additional_self_attention_layer: bool = False,
            pre_ctc_sub_sample: bool = False,
            use_fddt: bool = True,
            fddt_is_diagonal: bool = True,
            fddt_bias_only: bool = False,
            fddt_use_silence: bool = True,
            fddt_use_target: bool = True,
            fddt_use_overlap: bool = True,
            fddt_use_non_target: bool = True,
            remove_timestamps_from_ctc: bool = False,
            apply_fddt_to_n_layers: int = -1,
            fddt_init: str = 'suppressive',  # random, non-disturbing
            non_target_fddt_value: float = 0.0,
            use_initial_fddt: bool = False,
            use_enrollments: bool = False,
            scb_layers: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.final_dropout = final_dropout
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_weight = ctc_weight
        self.blank_token_id = blank_token_id
        self.additional_layer = additional_layer
        self.additional_self_attention_layer = additional_self_attention_layer
        self.pre_ctc_sub_sample = pre_ctc_sub_sample
        self.use_fddt = use_fddt
        self.fddt_is_diagonal = fddt_is_diagonal
        self.fddt_bias_only = fddt_bias_only
        self.fddt_use_silence = fddt_use_silence
        self.fddt_use_target = fddt_use_target
        self.fddt_use_overlap = fddt_use_overlap
        self.fddt_use_non_target = fddt_use_non_target
        self.remove_timestamps_from_ctc = remove_timestamps_from_ctc
        self.apply_fddt_to_n_layers = apply_fddt_to_n_layers
        self.fddt_init = fddt_init
        self.non_target_fddt_value = non_target_fddt_value
        self.use_initial_fddt = use_initial_fddt
        self.use_enrollments = use_enrollments
        self.scb_layers = scb_layers

class WhisperTimeStampLogitsProcessorCustom(WhisperTimeStampLogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = super().__call__(input_ids, scores)

        # Enable to early exit from silence via eos token
        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, self.eos_token_id] = scores[:, self.eos_token_id]

        return scores_processed

_HIDDEN_STATES_START_POSITION = 2

class FDDT(nn.Module):
    def __init__(self, d_model, non_target_rate=0.01, fddt_init=None, is_diagonal=False,
                 bias_only=False, use_silence=True, use_target=True, use_overlap=True, use_non_target=True):
        super().__init__()
        if use_target:
            self.target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, fddt_init=fddt_init,
                                     init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                        d_model,
                                                                                        bias=True, fddt_init=fddt_init,
                                                                                        init_eye_val=1.0))
        if use_non_target:
            self.non_target_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, fddt_init=fddt_init,
                                     init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, fddt_init=fddt_init, init_eye_val=non_target_rate))
        if use_overlap:
            self.overlap_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, fddt_init=fddt_init,
                                     init_eye_val=1.0) if is_diagonal else CustomLinear(d_model,
                                                                                        d_model,
                                                                                        bias=True, fddt_init=fddt_init,
                                                                                        init_eye_val=1.0))
        if use_silence:
            self.silence_linear = nn.Parameter(torch.zeros(d_model)) if bias_only else (
                CustomDiagonalLinear(d_model, bias=True, fddt_init=fddt_init,
                                     init_eye_val=non_target_rate) if is_diagonal else CustomLinear(
                    d_model, d_model, bias=True, fddt_init=fddt_init, init_eye_val=non_target_rate))

        self.use_silence = use_silence
        self.use_target = use_target
        self.use_overlap = use_overlap
        self.use_non_target = use_non_target
        self.bias_only = bias_only

    def forward(self, hidden_states, stno_mask):
        stno_mask = stno_mask.to(hidden_states.device)[..., None]
        if self.bias_only:
            if self.use_silence:
                hidden_states += stno_mask[:, 0, ...] * self.silence_linear
            if self.use_target:
                hidden_states += stno_mask[:, 1, ...] * self.target_linear
            if self.use_non_target:
                hidden_states += stno_mask[:, 2, ...] * self.non_target_linear
            if self.use_overlap:
                hidden_states += stno_mask[:, 3, ...] * self.overlap_linear
        else:
            orig_hidden_states = hidden_states
            hidden_states = (self.silence_linear(
                orig_hidden_states) if self.use_silence else orig_hidden_states) * stno_mask[:, 0, :] + \
                            (self.target_linear(
                                orig_hidden_states) if self.use_target else orig_hidden_states) * stno_mask[:, 1, :] + \
                            (self.non_target_linear(
                                orig_hidden_states) if self.use_non_target else orig_hidden_states) * stno_mask[:, 2,
                                                                                                      :] + \
                            (self.overlap_linear(
                                orig_hidden_states) if self.use_overlap else orig_hidden_states) * stno_mask[:, 3, :]
        return hidden_states


class CustomLinear(nn.Linear):
    def __init__(self, *args, init_eye_val=0.0, fddt_init=None, init_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_eye_val = init_eye_val
        self.fddt_init = fddt_init
        self.init_fun = init_fun
        self.reset_parameters()  # Ensure consistent init on creation

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Apply custom init function if provided
            if hasattr(self,"init_fun") and self.init_fun is not None:
                self.init_fun(self)
                return

            # Default initialization
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

            if hasattr(self, "fddt_init"):
                # FDDT-specific inits
                if self.fddt_init == 'non-disturbing':
                    # Make weight an identity matrix (if possible)
                    if self.weight.shape[0] == self.weight.shape[1]:
                        self.weight.copy_(torch.eye(self.weight.shape[0], device=self.weight.device))
                    else:
                        # Not square — fill first min(n, m) diagonals
                        eye = torch.zeros_like(self.weight)
                        n = min(self.weight.shape)
                        eye[:n, :n] = torch.eye(n, device=self.weight.device)
                        self.weight.copy_(eye)

                elif self.fddt_init == 'suppressive':
                    if self.weight.shape[0] == self.weight.shape[1]:
                        self.weight.copy_(self.init_eye_val * torch.eye(self.weight.shape[0], device=self.weight.device))
                    else:
                        eye = torch.zeros_like(self.weight)
                        n = min(self.weight.shape)
                        eye[:n, :n] = self.init_eye_val * torch.eye(n, device=self.weight.device)
                        self.weight.copy_(eye)

class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0, fddt_init=None):
        super().__init__()
        self.init_eye_val = init_eye_val
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.fddt_init = fddt_init
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # random init
            fan = self.weight.size(0)
            bound = math.sqrt(3.0 / fan)
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.zero_()

            # custom modes
            if self.fddt_init == 'non-disturbing':
                self.weight.fill_(1.0)
            elif self.fddt_init == 'suppressive':
                self.weight.fill_(self.init_eye_val)

    def forward(self, input):
        out = input * self.weight
        if self.bias is not None:
            out += self.bias
        return out

class InterpolationGate(nn.Module):
    def __init__(self, items, init_val=0.0):
        super().__init__()
        self.init_val = init_val
        self.gate = nn.Parameter(torch.full((items,), init_val))
        self.reset_parameters()

    def forward(self, orig_seq, new_seq):
        gate_act = torch.nn.functional.sigmoid(self.gate)
        output = (1 - gate_act) * orig_seq + gate_act * new_seq
        return output

    def reset_parameters(self):
        with torch.no_grad():
            self.gate.fill_(self.init_val)

def propagate_first_half_embeds_init(module):
    # Zero out all weights initially
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=1e-3)

    # Create identity mapping for first half of input (q_orig)
    # Input: [q_orig, cross_attn_output] -> map q_orig to first embed_dim outputs
    module.weight.data[:module.weight.shape[1] // 2, :module.weight.shape[1] // 2] += torch.eye(
        module.weight.shape[1] // 2)

    # Zero bias
    module.bias.data.zero_()


def propage_first_embeds_to_match_output_dim_init(module):
    # module.weight.data.zero_()
    torch.nn.init.xavier_uniform_(module.weight, gain=1e-3)

    # Create identity mapping from first embed_dim inputs to output
    module.weight.data[:, :module.weight.shape[0]] += torch.eye(module.weight.shape[0])

    # Zero bias for second linear
    module.bias.data.zero_()


# Cross attention block that can easily learn to ignore cross attention initially
class CrossAttentionEnrollBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.ffn_dim = config.encoder_ffn_dim

        self.cross_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

        # Layer normalization (pre-norm style)
        # self.norm_attn = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.cross_gate = InterpolationGate(1,init_val=-1.0)
        # Feed-forward network that maps concat space back to single channel
        self.ffn = nn.Sequential(
            CustomLinear(self.embed_dim * 2, self.ffn_dim, init_fun=propagate_first_half_embeds_init),
            ACT2FN[config.activation_function],
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
            CustomLinear(self.ffn_dim, self.embed_dim, init_fun=propage_first_embeds_to_match_output_dim_init),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, 2, T, F) - batch, channels, time, features
        Returns:
            Updated hidden states of same shape
        """
        q = hidden_states[:, 0]  # (B, T, F)
        kv = hidden_states[:, 1]  # (B, T, F)

        # Cross-attention
        attn_output = self.cross_attn(
            hidden_states=q,
            key_value_states=kv,
            output_attentions=False
        )[0]

        # Concatenate attention output with original normalized query
        q_concat = torch.cat([q, attn_output], dim=-1)  # (B, T, 2*F)

        # Feed-forward processing (no normalization to preserve initialization)
        updated_q = self.ffn(q_concat)  # (B, T, F)

        q_out = self.cross_gate(q, updated_q)
        # Return stacked result (only query channel is updated)
        return torch.stack([q_out, kv], dim=1)

class SpeakerCommunicationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_speakers = getattr(config, "mt_num_speakers", 2)
        self.config = config

        self.cae = CrossAttentionEnrollBlock(config)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        S = self.num_speakers

        # Reshape to (B//S, S, T, F)
        x_reshaped = x.view(B//S, S, T, F)

        # Call the selected method
        out = self.cae(x_reshaped)

        # Reshape back (B, T, F)
        out_merged = out.view(B, T, F)
        return out_merged

class CrossAttentionBlock(nn.Module):
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

class DiCoWEncoder(WhisperEncoder):
    config_class = DiCoWConfig

    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.ctc_weight = config.ctc_weight
        if config.additional_layer and self.ctc_weight > 0.0:
            self.additional_layer = WhisperEncoderLayer(config)
        if config.additional_self_attention_layer and self.ctc_weight > 0.0:
            self.additional_self_attention_layer = WhisperAttention(
                embed_dim=config.d_model,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
            )
        if config.pre_ctc_sub_sample and self.ctc_weight > 0.0:
            self.subsample_conv1 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.subsample_conv2 = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        if self.ctc_weight > 0.0:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size + 1, bias=False)
        self.final_dropout = nn.Dropout(config.final_dropout)
        if config.use_fddt:
            num_fddts = self.config.apply_fddt_to_n_layers if self.config.apply_fddt_to_n_layers != -1 else len(
                self.layers)
            self.initial_fddt = FDDT(
                d_model=config.d_model,
                non_target_rate=config.non_target_fddt_value,
                fddt_init=config.fddt_init,
                is_diagonal=config.fddt_is_diagonal,
                bias_only=config.fddt_bias_only,
                use_silence=config.fddt_use_silence,
                use_target=config.fddt_use_target,
                use_overlap=config.fddt_use_overlap,
                use_non_target=config.fddt_use_non_target,
            )
            self.fddts = nn.ModuleList([
                FDDT(
                    d_model=config.d_model,
                    non_target_rate=1.0,
                    fddt_init=config.fddt_init,
                    is_diagonal=config.fddt_is_diagonal,
                    bias_only=config.fddt_bias_only,
                    use_silence=config.fddt_use_silence,
                    use_target=config.fddt_use_target,
                    use_overlap=config.fddt_use_overlap,
                    use_non_target=config.fddt_use_non_target,
                )
                for _ in range(num_fddts)
            ])
            if config.use_enrollments and config.scb_layers is not None:
                self.ca_enrolls = nn.ModuleList([SpeakerCommunicationBlock(config) for _ in range(config.scb_layers)])
        self.first_task_token = self.config.vocab_size - 30 * 50 - 1 - 6  # 30 seconds of 50 Hz timestamps -1 to get to 0.0 and -6 number of tasks
        
        self.vision_encoder: Optional[nn.Module] = None
        self.vision_feature_dim: Optional[int] = None
        self.vision_projection = None
        self.vision_layer_norm = None
        self.cross_attn_blocks = None

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CustomLinear) or isinstance(module, CustomDiagonalLinear) or isinstance(module, InterpolationGate):
            module.reset_parameters()

    def set_vision_encoder(
        self,
        encoder
    ):
        self.vision_encoder = encoder
        self.vision_feature_dim = encoder.config.hidden_size
        self.vision_projection = nn.Linear(self.vision_feature_dim, self.config.d_model)
        self.vision_layer_norm = nn.LayerNorm(self.config.d_model)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(self.config.d_model, self.config.encoder_attention_heads, self.config.dropout) for _ in range(self.config.encoder_layers)
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

    def get_output_embeddings(self):
        return None

    def possibly_update_last_hidden_states(self, hidden_states):
        if hasattr(self, "additional_layer"):
            hidden_states, = self.additional_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                layer_head_mask=None,
            )
        elif hasattr(self, "additional_self_attention_layer"):
            hidden_states, _ = self.additional_self_attention_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                layer_head_mask=None,
            )

        hidden_states = self.final_dropout(hidden_states)
        if hasattr(self, "subsample_conv2"):
            hidden_states = self.subsample_conv2(self.subsample_conv1(hidden_states.transpose(1, 2))).transpose(1, 2)
        return hidden_states

    def get_loss(self, logits, labels):
        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        if self.config.remove_timestamps_from_ctc:
            labels = torch.nn.utils.rnn.pad_sequence([label[label < self.first_task_token] for label in labels],
                                                     padding_value=-100).T
        input_lengths = torch.full((logits.shape[0],), fill_value=logits.shape[1],
                                   device=logits.device)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=True):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=logits.shape[-1] - 1,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=True,
            )
        return ctc_loss

    def get_max_len(self):
        return self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]

    def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            stno_mask=None,
            return_logits=False,
            enrollments=None,
            vision_features=None,
            vision_pad_mask=None,
            cross_attn_mask=None,
    ):
        # import pdb; pdb.set_trace()
        if enrollments is not None:
            input_features = torch.stack((input_features, enrollments['input_features']), dim=1).flatten(0,1)
            stno_mask = torch.stack((stno_mask, enrollments['stno_mask']),dim=1).flatten(0,1)

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

        """<DiCoW CODE>"""
        if self.config.use_fddt:
            inputs_embeds = self.initial_fddt(inputs_embeds, stno_mask)
        """</DiCoW CODE>"""

        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        """<AV DiCoW CODE>"""
        if self.vision_encoder is not None:
            vision_features = self.embed_vision_features(vision_features)
        """<AV DiCoW CODE>"""

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
                """<DiCoW CODE>"""
                if self.config.use_fddt and idx < len(self.fddts):
                    hidden_states = self.fddts[idx](hidden_states, stno_mask)

                if self.config.use_enrollments and idx < self.config.scb_layers:
                    hidden_states = self.ca_enrolls[idx](hidden_states)
                """</DiCoW CODE>"""

                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            """<AV DiCoW CODE>"""
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
            """<AV DiCoW CODE>"""

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if enrollments is not None:
            hidden_states = hidden_states[::2]

        if return_logits:
            hidden_states = hidden_states
            hidden_states = self.possibly_update_last_hidden_states(hidden_states)
            logits = self.lm_head(hidden_states)

            return CausalLMOutput(
                loss=None, logits=logits, hidden_states=hidden_states,
            )

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DiCoW(WhisperModel):
    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.encoder = DiCoWEncoder(config)
        self.post_init()

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Cache] = None,
            decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            enrollments = None,
            vision_features=None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
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
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                head_mask=head_mask,
                return_dict=return_dict,
                stno_mask=stno_mask,
                enrollments=enrollments,
                vision_features=vision_features
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     raise ValueError("encoder_outputs should be of type BaseModelOutput when return_dict=True.")

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

# pylint: skip-file
# Copied from: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py
import pandas as pd
import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probabilities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, blank, eos):
        self.logzero = -1e10
        self.blank = blank
        self.eos = eos
        self.input_length = x.shape[1]
        self.batch_size = x.shape[0]
        self.x = x
        self.device = x.device

        # Preallocate `r` and `xs` tensors
        # `num_labels` will be set dynamically in __call__ but preallocated with maximum capacity
        self.max_num_labels = x.shape[2]  # Set to a max value that can be dynamically resized
        self.r = torch.full((self.batch_size, self.input_length, 2, self.max_num_labels), self.logzero,
                            device=self.device)
        self.xs = torch.full((self.batch_size, self.input_length, self.max_num_labels), self.logzero,
                             device=self.device)

    def initial_state(self):
        """Obtain an initial CTC state."""
        # Create initial CTC state tensor and use in-place operations to fill
        r = torch.full((self.batch_size, self.input_length, 2), self.logzero, device=self.device)
        r[..., 1] = torch.cumsum(self.x[..., self.blank], dim=1)
        s = torch.zeros((self.batch_size, 1), device=self.device)

        return r, s

    def _resize_tensors(self, number_of_current_samples, num_labels):
        if self.r.shape[0] != number_of_current_samples:
            self.r = self.r[:number_of_current_samples, ...]
            self.xs = self.xs[:number_of_current_samples, ...]

        if self.r.shape[3] != num_labels:
            self.r = self.r[:, :, :, :num_labels].fill_(self.logzero)
            self.xs = self.xs[:, :, :num_labels].fill_(self.logzero)
        else:
            self.r.fill_(self.logzero)
            self.xs.fill_(self.logzero)

    def _initialize_r(self, decoded_len):
        mask = (decoded_len == 0)
        self.r[mask, 0, 0, :] = self.xs[mask, 0]

    def _compute_log_phi(self, r_sum, cs, last, decoded_len, r_prev):
        # Expand r_sum for num_labels and initialize log_phi
        log_phi = r_sum[..., None].expand(-1, -1, cs.shape[1])

        # Create mask for cases where `decoded_len > 0` and to identify where `c == last[i]` for all `i`
        non_zero_mask = (decoded_len > 0)
        label_match_mask = (cs == last.unsqueeze(1))

        # Update log_phi where both `decoded_len > 0` and `c == last[i]`
        log_phi = torch.where((non_zero_mask.unsqueeze(1) & label_match_mask)[:, None, :], r_prev[..., 1:2], log_phi)
        return log_phi

    def _compute_log_psi(self, decoded_len, log_phi, x_current):
        """This function computes forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        and log prefix probabilities log(psi) for all labels in the batch.

        :param decoded_len: tensor of shape (batch_size,) containing the length of the decoded sequence
        :param log_phi: tensor of shape (batch_size, input_length, num_labels) containing the forward probabilities
        :param x_current: tensor of shape (batch_size, input_length, num_labels) containing the input frame

        :return log_psi: tensor of shape (batch_size,num_labels) containing the log prefix probabilities
        """
        B, T, V = log_phi.shape
        start = torch.clamp(decoded_len, min=1)  # Ensure start is at least 1 to avoid out-of-bounds

        # Initialize log_psi with the start position of r[:, start - 1, 0, :]
        log_psi = self.r[torch.arange(B), start - 1, 0, :]

        # Mask for handling sequence lengths based on decoded_len
        mask_t = torch.arange(1, T, device=decoded_len.device).expand(B, T - 1) >= decoded_len.unsqueeze(1)

        # Accumulate log_psi only up to the last valid time step for each sequence
        log_psi = torch.logaddexp(log_psi, torch.logsumexp(
            torch.where(mask_t.unsqueeze(-1), log_phi[:, :-1] + self.xs[:, 1:], self.logzero), dim=1))

        start = torch.clamp(decoded_len, 1)


        for t in range(start.min(), self.input_length):
            should_decode = decoded_len <= t
            self.r[:, t, 0] = torch.logaddexp(self.r[:, t - 1, 0],
                                              log_phi[:, t - 1]) + self.xs[:, t]
            self.r[:, t, 1] = (
                    torch.logaddexp(self.r[:, t - 1, 0], self.r[:, t - 1, 1]) + x_current[:, t, self.blank][:, None]
            )
            if ~should_decode.any():
                self.r[:, t] = torch.where(should_decode.unsqueeze(-1).unsqueeze(-1), self.r[:, t], self.logzero)

        return log_psi

    def _update_log_psi_with_eos(self, log_psi, cs, r_sum):
        # Update log_psi for eos positions
        eos_mask = (cs == self.eos)
        log_psi[eos_mask] = r_sum[:, -1].unsqueeze(1).expand_as(log_psi)[eos_mask]

        # Exclude blank probabilities if eos is not the blank
        if self.eos != self.blank:
            blank_mask = (cs == self.blank)
            log_psi[blank_mask] = self.logzero
        return log_psi

    def __call__(self, y, cs, decoded_len, samples_to_be_decoded, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        # output_length = y.shape[1] - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).

        # Dynamically resize r and xs to match num_labels if necessary
        num_labels = cs.shape[1]
        number_of_current_samples = cs.shape[0]
        self._resize_tensors(number_of_current_samples, num_labels)

        # Create a view of the current input frame
        x_current = self.x[samples_to_be_decoded]
        self.xs = torch.gather(x_current, 2, cs.unsqueeze(1).expand(-1, self.input_length, -1))

        # Initialize r for the first frame
        self._initialize_r(decoded_len)

        # prepare forward probabilities for the last label
        r_sum = torch.logaddexp(r_prev[:, :, 0], r_prev[:, :, 1])  # log(r_t^n(g) + r_t^b(g))
        last = y[:, -1]

        # precompute log_phi
        log_phi = self._compute_log_phi(r_sum, cs, last, decoded_len, r_prev)

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        log_psi = self._compute_log_psi(decoded_len, log_phi, x_current)

        # get P(...eos|X) that ends with the prefix itself
        log_psi = self._update_log_psi_with_eos(log_psi, cs, r_sum)

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.r


class CTCRescorerLogitsProcessor(LogitsProcessor):
    def __init__(
            self,
            encoder_logits: torch.FloatTensor,
            encoder_output_lens: torch.Tensor,
            blank_token_id: int,
            pad_token_id: int,
            eos_token_id: int,
            bos_token_id: int,
            tokenizer: PreTrainedTokenizer,
            ctc_margin: int,
            ctc_weight: float,
            num_beams: int,
            debug: bool = False,
            ctc_tokens_to_score: int = 500
    ):
        super().__init__()
        same_logits = torch.tensor(list((tokenizer.upper_cased_tokens.items())))

        logits = torch.nn.functional.log_softmax(encoder_logits, dim=-1)
        logits[..., same_logits[:, 1]] = logits[..., same_logits[:, 0]]

        self.logits = logits

        self.ctc_prefix_scorer = CTCPrefixScore(
            self.logits,
            blank_token_id,
            eos_token_id,
        )
        self.batch_size = logits.shape[0]
        self.input_length = logits.shape[1]
        self.num_tokens = logits.shape[2]
        self.device = logits.device
        self.ctc_weight = ctc_weight
        self.num_beams = num_beams
        self.ctc_state_prev, self.ctc_score_prev = self.ctc_prefix_scorer.initial_state()
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.debug = False
        self.first_timestamp_token_id = tokenizer.get_vocab()["<|0.00|>"]
        self.tmp_ctc_scores = torch.empty((self.batch_size, self.num_tokens - 1), device=self.device)
        self.tmp_ctc_states = torch.empty((self.batch_size, self.num_tokens - 1, self.input_length, 2),
                                          device=self.device)
        self.ctc_tokens_to_score = ctc_tokens_to_score

    def analyze_predictions(self,
                            scores, ctc_scores, next_token_scores, input_ids, k=10):
        print("\n" + "#" * 100)

        batch_size = input_ids.shape[0]

        best_att_ids = scores.topk(k=k, dim=1)
        ctc_scores[:, self.first_timestamp_token_id:] = self.ctc_prefix_scorer.logzero
        best_ctc_ids = ctc_scores.topk(k=k, dim=1)
        best_ids = next_token_scores.topk(k=k, dim=1)

        decoded_prefixes = self.tokenizer.batch_decode(
            input_ids, decode_with_timestamps=True, skip_special_tokens=False
        )

        def prepare_and_decode(best_ids_tensor):
            new_tensor = torch.zeros((batch_size, k * 2), dtype=torch.long)
            new_tensor[:, 0::2] = best_ids_tensor.indices
            new_tensor[:, 1::2] = self.tokenizer.vocab['#']

            # Flatten to (batch_size * k, 2)
            flat_tensor = new_tensor.view(-1, 2)
            decoded = self.tokenizer.batch_decode(
                flat_tensor, decode_with_timestamps=True, skip_special_tokens=False
            )
            # Reshape back to (batch_size, k)
            decoded = [(decoded[i * k:(i + 1) * k]) for i in range(batch_size)]
            return decoded

        decoded_att = prepare_and_decode(best_att_ids)
        decoded_ctc = prepare_and_decode(best_ctc_ids)
        decoded_next = prepare_and_decode(best_ids)

        for idx in range(batch_size):
            print("-" * 80)
            print(f"HYPOTHESIS {idx}")
            print("\nPREFIX:")
            print(decoded_prefixes[idx])

            def print_with_pandas(tokens, scores, title):
                df = pd.DataFrame([tokens, [f"{s.item():.2f}" for s in scores]])
                df.index = [f"{title}", "Score"]
                print(f"\n{title}:")
                print(df.to_string(index=True, header=False))

            print_with_pandas(decoded_att[idx], best_att_ids.values[idx], "ATT_TOKENS")
            print_with_pandas(decoded_ctc[idx], best_ctc_ids.values[idx], "CTC_TOKENS")
            print_with_pandas(decoded_next[idx], best_ids.values[idx], "NEXT_TOKENS")

            print(f"\nCTC_EOS: {ctc_scores[idx, self.tokenizer.eos_token_id].item():.2f}")
            print()

        print("#" * 100)

    def update_state(self, best_ids, beam_idx):
        mask = best_ids < self.first_timestamp_token_id
        self.ctc_state_prev = torch.where(mask.unsqueeze(-1).unsqueeze(-1),
                                          self.tmp_ctc_states[beam_idx, best_ids],
                                          self.ctc_state_prev[beam_idx])
        self.ctc_score_prev = torch.where(mask.unsqueeze(-1),
                                          self.tmp_ctc_scores[beam_idx, best_ids].unsqueeze(-1),
                                          self.ctc_score_prev[beam_idx])

    def __call__(self, input_ids_orig: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids_orig.clone()

        # Remove prefix from CTC scoring
        if (input_ids[:, 0] != self.bos_token_id).any():
            input_ids = torch.stack(
                [row[(row == self.bos_token_id).nonzero(as_tuple=True)[0].item():] for row in input_ids])

        # Remove task/lang/timestamp tokens from input_ids
        input_prefix_len = len(self.tokenizer.prefix_tokens)
        if input_prefix_len > 1:
            input_ids = input_ids[:, input_prefix_len - 1:]

        # Setup the first token to be the blank token(sos)
        input_ids[:, 0] = self.blank_token_id

        # If there is last token in input_ids timestamp replicate last non-timestamp token which could be potentially even the first token
        decoded_len = torch.logical_and(input_ids <= self.first_timestamp_token_id,
                                        input_ids != self.blank_token_id).sum(dim=1)
        mask = torch.logical_and(input_ids[:, -1] >= self.first_timestamp_token_id,
                                 input_ids[:, -1] != self.blank_token_id)
        last_non_timestamp_token = torch.gather(input_ids, 1,
                                                torch.logical_or(input_ids < self.first_timestamp_token_id,
                                                                 input_ids == self.blank_token_id).sum(dim=1,
                                                                                                       keepdim=True) - 1)
        input_ids[mask, -1] = last_non_timestamp_token[mask, 0]

        # If there is no eos token in the last position, we need to continue decoding
        to_be_decoded = input_ids[:, -1] != self.eos_token_id
        self.tmp_ctc_scores[:] = self.ctc_prefix_scorer.logzero

        input_ids_local = input_ids[to_be_decoded]
        ids_to_score = torch.topk(scores[:, :self.first_timestamp_token_id], k=self.ctc_tokens_to_score).indices

        # always score EOS token if not present put on position of last id
        is_eos_present = (ids_to_score == self.eos_token_id).any(dim=1)
        ids_to_score[~is_eos_present, self.ctc_tokens_to_score - 1] = self.eos_token_id

        decoded_len_local = decoded_len[to_be_decoded]

        ctc_scores_local, ctc_states_local = self.ctc_prefix_scorer(input_ids_local, ids_to_score[to_be_decoded],
                                                                    decoded_len_local, to_be_decoded,
                                                                    self.ctc_state_prev[to_be_decoded])

        # As the CTC scorer might run on subset of samples, we need to scatter the results back to the original batch
        self.tmp_ctc_scores[to_be_decoded] = (self.tmp_ctc_scores[to_be_decoded]
                                              .scatter(1, ids_to_score[to_be_decoded], ctc_scores_local))
        self.tmp_ctc_states[to_be_decoded] = (self.tmp_ctc_states[to_be_decoded].permute(0, 2, 3, 1)
                                              .scatter(3, ids_to_score[to_be_decoded].unsqueeze(1).unsqueeze(1)
                                                       .repeat(1, *ctc_states_local.shape[1:3], 1), ctc_states_local)
                                              .permute(0, 3, 1, 2))

        # Set the CTC score for the timestamp tokens to the maximum to prefer them over the rest
        self.tmp_ctc_scores[:, self.first_timestamp_token_id:] = self.tmp_ctc_scores.max(dim=1).values[:, None]
        ctc_scores = self.tmp_ctc_scores - self.ctc_score_prev

        next_token_scores = (1 - self.ctc_weight) * scores + self.ctc_weight * ctc_scores

        if self.debug:
            self.analyze_predictions(scores, ctc_scores, next_token_scores, input_ids_orig)

        return next_token_scores


class LogSoftmaxProcessor(LogitsProcessor):
    def __init__(
            self,
    ):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        return scores

class DiCoWGenerationMixin(WhisperForConditionalGeneration):

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, generation_config,
    ) -> Dict[str, Any]:
        # pylint: disable=no-memberva
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            self.encoder_logits = self.get_enc_logits(model_kwargs["encoder_outputs"].last_hidden_state)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
            self,
            batch_size: int,
            model_input_name: str,
            model_kwargs: Dict[str, torch.Tensor],
            decoder_start_token_id: torch.Tensor,
            device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        batch_size = model_kwargs['decoder_input_ids'].shape[0]
        out = super()._prepare_decoder_input_ids_for_generation(
            batch_size,
            model_input_name,
            model_kwargs,
            decoder_start_token_id,
            device,
        )
        return out

    def prepare_kwargs_for_generate(self,
                                    max_frames,
                                    cur_bsz,
                                    batch_idx_map,
                                    seek,
                                    kwargs):
        """This method also prepares STNO masks and other kwargs for generation."""

        seek_vad = seek // 2
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        num_frames_vad = num_segment_frames // 2
        max_frames_vad = max_frames // 2
        seek_num_frames = (max_frames_vad - seek_vad).clamp(max=num_frames_vad)

        stno_masks = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            segment_input_slice = kwargs["stno_mask"][prev_i: prev_i + 1, :,
                                  seek_vad[prev_i]: seek_vad[prev_i] + seek_num_frames[prev_i]]

            if segment_input_slice.shape[-1] < num_frames_vad:
                orig_len = segment_input_slice.shape[-1]
                # pad to 1500 if necessary
                segment_input_slice = torch.nn.functional.pad(
                    segment_input_slice, pad=(0, num_frames_vad - orig_len)
                )
                # set corresponding padding tokens to 1 in vad mask representing silence
                segment_input_slice[0, 0, orig_len:] = 1.0

            stno_masks.append(segment_input_slice)
        kwargs["stno_mask"] = torch.cat(stno_masks, dim=0)
        self.stno_mask_seek = kwargs["stno_mask"]

        if self.config.use_enrollments and "enrollments" in kwargs:
            for key in kwargs["enrollments"]:
                kwargs["enrollments"][key] = kwargs["enrollments"][key][batch_idx_map]

        if "labels" in kwargs:
            kwargs['labels'] = kwargs["labels"][batch_idx_map]
            kwargs['upp_labels'] = kwargs["upp_labels"][batch_idx_map]
        return kwargs


    def _retrieve_init_tokens(self, input_features, batch_size, generation_config, config, num_segment_frames, kwargs):
        if not hasattr(generation_config, "forced_decoder_ids"):
            generation_config.forced_decoder_ids = None
        if not hasattr(generation_config, "language"):
            generation_config.language = None
        if not hasattr(generation_config, "task"):
            generation_config.task = None

        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        forced_decoder_ids = generation_config.forced_decoder_ids
        if forced_decoder_ids is not None:
            if language is None and task is None and forced_decoder_ids[0][1] is None:
                logger.warning_once(
                    "Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English."
                    "This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`."
                )
        elif hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids is not None:
            forced_decoder_ids = config.forced_decoder_ids

        elif forced_decoder_ids is not None and language is not None:
            logger.info(
                f"You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}."
            )
            forced_decoder_ids = None

        if forced_decoder_ids is not None:
            return forced_decoder_ids

        init_tokens = super()._retrieve_init_tokens(input_features, batch_size, generation_config, config, num_segment_frames, kwargs)
        return init_tokens

    def detect_language(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
            generation_config: Optional[GenerationConfig] = None,
            num_segment_frames: int = 3000,
    ) -> torch.Tensor:
        """
        Detects language from log-mel input features or encoder_outputs

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
                `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
                hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            num_segment_frames (`int`, *optional*, defaults to 3000):
                The number of log-mel frames the model expects

        Return:
            A `torch.LongTensor` representing the detected language ids.
        """
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specify only one of `input_features` or `encoder_outputs` - not both!")
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
            batch_size = input_features.shape[0]
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}
            batch_size = (
                encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
            )

        generation_config = generation_config or self.generation_config
        decoder_input_ids = (
                torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                * generation_config.decoder_start_token_id
        )

        with torch.no_grad():

            """<DiCoW CODE>"""
            logits = self(**inputs, decoder_input_ids=decoder_input_ids, use_cache=False,
                          stno_mask=self.stno_mask[:, :, :num_segment_frames // 2]).logits[:, -1]
            """</DiCoW CODE>"""

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        lang_ids = logits.argmax(-1)

        return lang_ids

    def _get_logits_processor(
            self,
            generation_config: GenerationConfig,
            input_ids_seq_length: Optional[int] = None,
            encoder_input_ids: Optional[torch.LongTensor] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            device: Optional[str] = None,
            model_kwargs: Optional[dict[str, Any]] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # pylint: disable=no-member
        gen_config_copy = copy.deepcopy(generation_config)
        gen_config_copy.forced_decoder_ids = None
        processors = super()._get_logits_processor(
            gen_config_copy,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            device,
            model_kwargs,
            negative_prompt_ids,
            negative_prompt_attention_mask,
        )
        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            enc_logits = self.encoder_logits
            if generation_config.num_beams <= 1:
                processors.append(LogSoftmaxProcessor())
            else:
                enc_logits = enc_logits.repeat_interleave(generation_config.num_beams, dim=0)
            self.ctc_rescorer = CTCRescorerLogitsProcessor(
                enc_logits,
                torch.full((enc_logits.shape[0],), fill_value=enc_logits.shape[1],
                           device=enc_logits.device),
                enc_logits.shape[-1] - 1,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                generation_config.decoder_start_token_id,
                self.tokenizer,
                generation_config.ctc_margin,
                generation_config.ctc_weight,
                generation_config.num_beams,
                False,
            )
            processors.append(self.ctc_rescorer)
        return processors

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, num_beams, device):
        if generation_config.return_timestamps is True:
            """<DiCoW CODE>"""
            timestamp_processor = WhisperTimeStampLogitsProcessorCustom(generation_config, begin_index=begin_index)
            """</DiCoW CODE>"""
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens, device=device)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index, device=device
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None:
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor

    @staticmethod
    def round_to_nearest_0_02(x):
        d = Decimal(str(x))  # Use str(x) to preserve input precision
        step = Decimal('0.02')
        # Divide, round, multiply back
        rounded = (d / step).to_integral_value(rounding=ROUND_HALF_UP) * step
        return rounded

    def _fix_timestamps_from_segmentation(self, sequences):
        """
        Adjusts token sequences with global timestamps to fit within Whisper's 0–30s timestamp token range.

        This function modifies the input sequences by inserting appropriate timestamp tokens and
        offset corrections to ensure the decoded token order is correct, without splitting any segment.
        It aligns all timestamps to 0.02-second precision, inserts placeholder segments to bridge
        time gaps between 30-second windows, and maintains segment continuity during encoding.

        Args:
            sequences (dict): A dictionary containing:
                - 'segments': A list of segment lists, each segment being a dict with 'start', 'end', and 'tokens'.
                - 'sequences': A tensor used to determine device for padding.

        Returns:
            torch.Tensor: A batch of padded token sequences with corrected timestamp alignment.
        """
        # Get the token ID for the "<|0.00|>" timestamp used to detect dummy segments
        first_timestamp_token = self.tokenizer.get_vocab()["<|0.00|>"]
        empty_text_token = self.tokenizer.get_vocab()["Ġ"]
        results = []

        # Filter out segments that are either empty or consist only of the "<|0.00|>" token
        for idx, sequence_segs in enumerate(sequences['segments']):
            sequences['segments'][idx] = [
                seg for seg in sequence_segs
                if len(seg['tokens']) > 0 and (len(seg['tokens']) != 1 or seg['tokens'][0] != first_timestamp_token)
            ]

        # Iterate over each group of segments (e.g., one per utterance)
        for idx, sequence_segs in enumerate(sequences['segments']):
            result = []
            prev_segment_end_time = None
            correction = Decimal(0.0)

            for i, seg in enumerate(sequence_segs):
                # Round start and end times to nearest 0.02 seconds
                start_time = self.round_to_nearest_0_02(seg['start'].item())
                end_time = self.round_to_nearest_0_02(seg['end'].item())
                tokens = seg['tokens']

                # Determine which 30s window this segment falls into
                current_block = (start_time + correction) // 30

                if prev_segment_end_time is not None:
                    # If not the first segment, calculate difference in 30s windows
                    prev_block = prev_segment_end_time // 30
                    num_dummies = current_block - prev_block - 1

                    # Insert (30, [], 30) marker if we're moving to a new block
                    if current_block > prev_block:
                        result.append((30, [empty_text_token], 30))

                    # Insert dummy segments to bridge skipped 30s blocks
                    for _ in range(int(num_dummies)):
                        result.append((0, [empty_text_token], 30))
                else:
                    # For the first segment, add dummy blocks if it starts after 30s
                    for _ in range(int(start_time // 30)):
                        result.append((0, [empty_text_token], 30))

                # Determine whether segment fits in one block or wraps to the next
                if ((start_time + correction) // 30 == (end_time + correction) // 30) or (end_time + correction) % 30 == 0:
                    # Segment fits within a single 30s window
                    result.append(((start_time + correction) % 30, tokens, (end_time + correction) % 30))
                else:
                    # Segment would wrap across a 30s boundary
                    new_seg_start = (correction + start_time) % 30
                    seg_duration = end_time - start_time
                    new_end_time = (end_time + correction) % 30
                    # if segment duration is exactly 30s we have to use a correction trick, elsewise tokenizer will automatically adjust
                    if seg_duration == 30.0:
                        if float(new_seg_start) % 30.0 == 0.0:
                            new_end_time = Decimal(30.0)
                            correction = Decimal(0.0)
                        else:
                            correction = Decimal(-0.02)
                            new_end_time += Decimal(correction)
                    else:
                        correction = Decimal(0.0)
                    result.append((new_seg_start, tokens, new_end_time))
                # print(f'Processed segment {i}, result: {self.tokenizer.decode(self.tokenizer("".join([f"<|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}|>" for seg in result]))["input_ids"], decode_with_timestamps=True)[-250:]}')
                # Update the previous segment's end time for next iteration
                prev_segment_end_time = end_time + correction

            # Convert result segments into a token sequence with proper timestamp formatting
            encoded = self.tokenizer(
                "".join([f"<|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}|>" for seg in result])
            )['input_ids']
            results.append(encoded)

        # Pad all sequences to the same length for batching
        sequences = pad_sequence(
            [torch.tensor(res, device=sequences['sequences'].device) for res in results],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        return sequences

    @staticmethod
    def _retrieve_segment(
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            time_precision_features,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
            decoder_input_ids,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []
        idx_offset = decoder_input_ids.shape[-1]
        device = seek_sequence.device

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))
            else:
                # we want to include the last timestamp token in the last segment to know it was no single ending
                slices[-1] += 1

            last_slice = 0
            # Add each segment to list of all segments
            for i, current_slice in enumerate(slices):
                is_last_slice = i == len(slices) - 1
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0] - timestamp_begin
                idx_sliced_tokens = -1 if not is_last_slice or single_timestamp_ending else -2
                end_timestamp_pos = sliced_tokens[idx_sliced_tokens] - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx]
                                 + start_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                                 * time_precision,
                        "end": time_offset[prev_idx]
                               + end_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                               * time_precision,
                        "tokens": sliced_tokens,
                        "idxs": (idx_offset + last_slice, idx_offset + current_slice),
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[idx_offset + last_slice: idx_offset + current_slice] + time_offset[
                        prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 2].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            start_timestamp_pos = 0.0
            last_timestamp_pos = seek_num_frames[prev_idx] // 2
            skip = False
            segment_offset = seek_num_frames[prev_idx]

            if timestamps.numel() > 1:
                start_timestamp_pos = timestamps[-2].item() - timestamp_begin
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin
            elif timestamps.numel() == 1:
                # no consecutive timestamps but it has a timestamp; use the last one.
                start_timestamp_pos = timestamps[-1].item() - timestamp_begin
                if start_timestamp_pos > 200:
                    # segment does not fit into decoding window, so we need to rollback
                    segment_offset = start_timestamp_pos * input_stride - 100  # timestamp might be inaccurate
                    skip = True
            else:
                # empty sequence, or sequence w/o timestamps
                skip = True

            if skip:
                segments = []
            else:
                segments = [
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                        "tokens": seek_sequence,
                        "result": seek_outputs[idx],
                    }
                ]
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = token_timestamps + time_offset[prev_idx]
                segment_offset = seek_num_frames[prev_idx]

        if segment_offset <= 0:
            msg = f"Timestamps: {timestamps}, Segments: {segments}"
            raise ValueError(f"Segment offset: {segment_offset} <= 0. This should not happen!\n{msg}")

        return segments, segment_offset

    def generate(
            self,
            generation_config: Optional[GenerationConfig] = None,
            condition_on_prev_tokens: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            **kwargs,
    ):
        if condition_on_prev_tokens:
            raise NotImplementedError("Current version does not support conditioning")

        gen_c, _ = self._prepare_generation_config(generation_config, **kwargs)
        gen_mode = gen_c.get_generation_mode(assistant_model)

        if gen_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.BEAM_SEARCH]:
            raise ValueError(
                f"Provided generation mode {gen_mode} is not supported"
                f" for WhisperForConditionalGeneration with joint CTC decoding")

        if "stno_mask" in kwargs:
            self.stno_mask = kwargs["stno_mask"]

        output = super().generate(**kwargs, return_segments=True)

        self.encoder_logits = None

        if isinstance(output, dict):
            output = self._fix_timestamps_from_segmentation(output)

        return output


    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        seek,
        batch_idx_map,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        kwargs_local = copy.deepcopy(kwargs)
        max_frames = attention_mask.sum(-1).cpu().to(torch.long)
        kwargs_local = self.prepare_kwargs_for_generate(max_frames, cur_bsz, batch_idx_map, seek, kwargs_local)
        seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type = super().generate_with_fallback(
            segment_input,
            decoder_input_ids,
            cur_bsz,
            seek,
            batch_idx_map,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            is_shortform,
            batch_size,
            attention_mask,
            kwargs_local,
        )
        self.stno_mask_seek = None

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type


    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            """<DiCoW CODE>"""
            # Based on the next tokens select the ctc prev states and scores
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(next_tokens, torch.arange(next_tokens.shape[0]))
            """</DiCoW CODE>"""

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids




    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        If it's the first time you're diving into Beam Search, we recommend you read the following blog post:
        https://huggingface.co/blog/how-to-generate (especially the beam search section).

        You can recompute the sequence scores from the individual scores using the `compute_transition_scores` function
        (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores)

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size*num_beams, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        # 1. init beam_search values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        early_stopping = generation_config.early_stopping
        length_penalty = generation_config.length_penalty
        max_length = generation_config.max_length
        num_beams = generation_config.num_beams
        num_return_sequences = generation_config.num_return_sequences

        batch_size_unflattened, cur_len = input_ids.shape[:2]
        batch_size = batch_size_unflattened // num_beams
        # TODO (joao): standardize special cases
        if self.__class__.__name__ == "MoshiDepthDecoder":
            vocab_size = self.config.audio_vocab_size
        elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
            vocab_size = self.get_output_embeddings().out_features
        else:
            vocab_size = self.config.get_text_config().vocab_size
        decoder_prompt_len = cur_len
        this_peer_finished = False

        # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
        # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
        # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
        # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0,
        ).to(input_ids.device)

        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        # (joao) feature lost in the refactor. Probably won't implement, hurts readability with minimal gains (there
        # are newer low-memory alternatives like the offloaded cache)
        sequential = generation_config.low_memory
        if sequential:
            raise ValueError(
                "`low_memory=True` is not supported after the beam search refactor. Please check the discussion in "
                "#35802 *after the PR got merged*, and add a comment there if your questions are not yet answered."
            )

        # 2. init output tuples
        all_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # 3. init running tensors and static-shaped placeholders

        # per batch, beam-item holding current token in loop and completed sequences
        output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
        running_sequences = torch.full(
            (batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=torch.int64,
            device=input_ids.device,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(input_ids, batch_size, num_beams)
        sequences = running_sequences.detach().clone()

        # per batch, beam-item score, logprobs
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

        # per batch, beam-item state bit indicating if sentence has finished.
        is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)

        # per batch state bit indicating if there is a possibility to improve the best finished sentence.
        is_early_stop_heuristic_unsatisfied = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)

        # per batch, beam-item state bit indicating if there are valid continuations.
        next_token_hits_stopping_criteria = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
        )

        # per batch selected beam indices
        running_beam_indices = torch.full(
            (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
        )
        beam_indices = running_beam_indices.detach().clone()

        # 4. run the generation loop
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            model_outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref
            logits = model_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = logits_processor(flat_running_sequences, log_probs)

            # Store logits, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logits.clone(),)
                if return_dict_in_generate and output_scores:
                    all_scores += (log_probs.clone(),)

                if output_attentions:
                    decoder_attentions += (
                        (model_outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (model_outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (model_outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.hidden_states,)
                    )

            # This is needed to properly delete logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del model_outputs

            log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
            # continuations among all beams based on the accumulated scores.
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # d. Check which running sequences have finished
            next_token_hits_stopping_criteria = stopping_criteria(
                self._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),  # remove unfilled token indexes
                all_scores,
            )
            next_token_hits_stopping_criteria = self._unflatten_beam_dim(
                next_token_hits_stopping_criteria, batch_size, beams_to_keep
            )

            # e. Get the non-finished running `num_beams` sequences for the next generation step
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # f. Update the completed beams if a new high score in a finished sequence is found
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )


            # g. Prepare remaining data for the next iteration, including computing the stopping condition for
            # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)

            beam_idx = None
            # pluck the cache from the beam indices that will be used in the next iteration
            # NOTE: we need to check if `self._reorder_cache` exists for special models like RAG, RecurrentGemma etc.
            if model_kwargs.get("past_key_values", None) is not None:
                beam_idx = self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
                if hasattr(self, "_reorder_cache"):
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
                else:
                    model_kwargs["past_key_values"].reorder_cache(beam_idx)

            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(running_sequences.flatten(0,1)[:, cur_len], beam_idx)

            cur_len = cur_len + 1
            is_early_stop_heuristic_unsatisfied = self._check_early_stop_heuristic(
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                running_beam_scores=running_beam_scores,
                beam_scores=beam_scores,
                is_sent_finished=is_sent_finished,
                cur_len=cur_len,
                max_length=max_length,
                decoder_prompt_len=decoder_prompt_len,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )
            this_peer_finished = not self._beam_search_has_unfinished_sequences(
                is_early_stop_heuristic_unsatisfied,
                is_sent_finished,
                next_token_hits_stopping_criteria,
                early_stopping,
            )

        # 5. prepare outputs
        # Take best beams for each batch (the score is sorted in descending order)
        sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
        beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        # Crop the static-shaped tensors to the actual size.
        # `beam_indices` is initialized with -1s, and is updated with the beam index of the generated token at each
        # step. We can use it to detect the generated length, which may be != `cur_len`  (e.g. selected beam is from a
        # previous decoding iteration)
        max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
        output_length = decoder_prompt_len + max_generated_length
        sequences = sequences[:, :output_length]
        beam_indices = beam_indices[:, :max_generated_length]

        if return_dict_in_generate:
            if not output_scores:
                beam_scores = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequences

class DiCoWForConditionalGeneration(DiCoWGenerationMixin, WhisperForConditionalGeneration):
    config_class = DiCoWConfig

    def __init__(self, config: DiCoWConfig):
        super().__init__(config)
        self.model = DiCoW(config)
        self.encoder_logits = None
        self.tokenizer = None
        self.stno_mask = None
        self.stno_mask_seek = None
        self.post_init()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_enc_logits(self, hidden_states):
        encoder = self.model.encoder
        hidden_states = encoder.possibly_update_last_hidden_states(hidden_states)
        logits = encoder.lm_head(hidden_states)
        return logits

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            stno_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Cache] = None,
            decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            upp_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            forced_decoder_ids: Optional[torch.LongTensor] = None,
            enrollments= None,
            vision_features=None,
            vision_feature_lengths=None,
            input_feature_lengths=None,
            **kwargs,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        import pdb; pdb.set_trace()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            stno_mask=stno_mask,
            enrollments=enrollments,
            vision_features=vision_features
        )

        dec_lm_logits = self.proj_out(outputs.last_hidden_state)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(dec_lm_logits.device)
            dec_loss1 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            # Only compute second loss if upp_labels exist
            if upp_labels is not None:
                dec_loss2 = loss_fct(dec_lm_logits.view(-1, self.config.vocab_size), upp_labels.reshape(-1))
                dec_loss = torch.hstack((dec_loss1[..., None], dec_loss2[..., None])).min(dim=-1).values.mean()
            else:
                dec_loss = dec_loss1.mean()

            if self.config.ctc_weight > 0.0:
                enc_lm_logits = self.get_enc_logits(outputs.encoder_last_hidden_state)
                enc_labels = labels.clone()
                for token in self.tokenizer.prefix_tokens:
                    if (enc_labels[:, 0] == token).all():
                        enc_labels = enc_labels[:, 1:]
                enc_labels[enc_labels == self.config.eos_token_id] = -100

                ctc_loss = self.get_encoder().get_loss(enc_lm_logits, enc_labels)
                loss = (1 - self.config.ctc_weight) * dec_loss + self.config.ctc_weight * ctc_loss
            else:
                loss = dec_loss

        if not return_dict:
            output = (dec_lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=dec_lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _get_feat_extract_output_lengths(self, attention_mask: torch.LongTensor) -> torch.LongTensor:
        return (self.model.encoder._get_feat_extract_output_lengths(attention_mask) / 4).ceil()