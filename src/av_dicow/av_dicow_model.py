
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# Add the TS-ASR-Whisper path to import WhisperEncoderForCTC
sys.path.append('/home/rphadke1/chime/TS-ASR-Whisper/src')
from models.whisper_ctc import WhisperEncoderForCTC, WhisperForCTCConfig

# Add the DINOv3 path to import the model
sys.path.append('/home/rphadke1/chime/dinov3')
import dinov3.hub.backbones as dinov3_backbones

# Import Hugging Face components
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizerFast


class AVEncoderForCTC(nn.Module):
    """
    Audio-Visual Encoder for CTC + Attention Loss
    
    This class combines:
    1. DINOv3 visual encoder for processing lip crop images
    2. WhisperEncoderForCTC for audio processing with cross-attention to visual features
    3. Projection layer to align visual and audio feature dimensions
    """
    
    def __init__(self, config):
        super(AVEncoderForCTC, self).__init__()
        
        # Store config
        self.config = config
        
        # Initialize DINOv3 visual encoder
        # DINOv3 ViT-S16 has embed_dim=384
        self.dinov3_model_name = getattr(config, 'dinov3_model_name', 'dinov3_vits16')
        self.visual_encoder = self._load_dinov3_model()
        
        # Visual feature projection to match audio encoder dimension
        # DINOv3 ViT-S16 output: 384, Whisper typically uses 768 or 1024
        self.visual_projection = nn.Linear(
            in_features=384,  # DINOv3 ViT-S16 embed_dim
            out_features=config.d_model,  # Whisper d_model
            bias=True
        )
        
        # Initialize audio encoder (WhisperEncoderForCTC)
        self.audio_encoder = WhisperEncoderForCTC(config)
        
        # Freezing control
        self.visual_encoder_frozen = False
        self.audio_encoder_frozen = False
        
    def _load_dinov3_model(self):
        """Load pretrained DINOv3 model"""
        try:
            # Load DINOv3 model using torch.hub
            model = torch.hub.load(
                '/export/fs06/rphadke1/data/mcorec/model-bin/avwhisper/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 
                self.dinov3_model_name, 
                source='local',
                pretrained=True
            )
            return model
        except Exception as e:
            print(f"Error loading DINOv3 model: {e}")
            # Fallback: try loading from dinov3_backbones
            if self.dinov3_model_name == 'dinov3_vits16':
                return dinov3_backbones.dinov3_vits16(pretrained=True)
            else:
                raise ValueError(f"Unsupported DINOv3 model: {self.dinov3_model_name}")
    
    def freeze_visual_encoder(self):
        """Freeze visual encoder parameters"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder_frozen = True
        print("Visual encoder frozen")
    
    def freeze_audio_encoder(self):
        """Freeze audio encoder parameters"""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder_frozen = True
        print("Audio encoder frozen")
    
    def train_visual_encoder(self):
        """Unfreeze visual encoder parameters"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = True
        self.visual_encoder_frozen = False
        print("Visual encoder unfrozen")
    
    def train_audio_encoder(self):
        """Unfreeze audio encoder parameters"""
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
        self.audio_encoder_frozen = False
        print("Audio encoder unfrozen")
    
    def _process_video_frames(self, video, video_lengths):
        """
        Process video frames through DINOv3 visual encoder
        
        Args:
            video: Tensor of shape [batch, num_frames, channels, height, width]
            video_lengths: List of actual frame lengths for each sequence
            
        Returns:
            visual_features: Tensor of shape [batch, num_frames, 384]
        """
        batch_size, channels, num_frames, height, width = video.shape
        
        # Reshape to process all frames at once: [batch*num_frames, channels, height, width]
        video_reshaped = video.view(batch_size * num_frames, channels, height, width)
        
        # Process through DINOv3
        # DINOv3 expects [batch, channels, height, width] and returns [batch, embed_dim]
        visual_features = self.visual_encoder(video_reshaped)["x_norm_clstoken"]
        
        # Reshape back to [batch, num_frames, 384]
        visual_features = visual_features.view(batch_size, num_frames, -1)
        
        # Project to match audio encoder dimension
        visual_features = self.visual_projection(visual_features)
        
        return visual_features
    
    def _align_visual_audio_temporal(self, visual_features, audio_length):
        """
        Align visual features temporal dimension with audio features
        
        Whisper operates at ~50Hz internally (after conv layers), 
        while video is typically at 25fps. We need to upsample visual features 2x.
        
        Args:
            visual_features: [batch, num_frames, d_model]
            audio_length: Target audio sequence length
            
        Returns:
            aligned_visual_features: [batch, audio_length, d_model]
        """
        batch_size, num_frames, d_model = visual_features.shape
        
        # Upsample visual features to match audio temporal resolution
        # Use linear interpolation to upsample from num_frames to audio_length
        visual_features = visual_features.transpose(1, 2)  # [batch, d_model, num_frames]
        
        # Interpolate to match audio length
        visual_features = F.interpolate(
            visual_features, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        )
        
        visual_features = visual_features.transpose(1, 2)  # [batch, audio_length, d_model]
        
        return visual_features

    def forward(self, video, audio, video_lengths, audio_lengths, vad_mask=None, **kwargs):
        """
        Forward pass through audio-visual encoder
        
        Args:
            video: Tensor of shape [batch, num_frames, channels, height, width]
            audio: Tensor of shape [batch, audio_features, audio_length] (Whisper mel features)
            video_lengths: List of actual video frame lengths
            audio_lengths: List of actual audio lengths  
            vad_mask: Optional VAD mask for target speaker amplification
            **kwargs: Additional arguments for WhisperEncoderForCTC
            
        Returns:
            encoder_outputs: Output from WhisperEncoderForCTC
        """
        
        # Process video frames through DINOv3
        visual_features = self._process_video_frames(video, video_lengths)
        
        # Get audio sequence length for temporal alignment
        # Whisper conv layers reduce temporal dimension, so we need to account for that
        # For now, we'll use the audio input length and let Whisper handle the reduction
        audio_seq_length = audio.shape[-1]
        
        # Align visual features temporally with audio
        visual_features = self._align_visual_audio_temporal(visual_features, audio_seq_length)
        
        # Pass audio through WhisperEncoderForCTC with cross-attention to visual features
        # The WhisperEncoderForCTC already supports cross_attention_features parameter
        encoder_outputs = self.audio_encoder(
            input_features=audio,
            cross_attention_features=visual_features,
            vad_mask=vad_mask,
            **kwargs
        )
        
        return encoder_outputs


# Configuration class for AV-DiCOW model
class AVDiCOWConfig(WhisperForCTCConfig):
    """Configuration class for Audio-Visual Diarization-Conditioned Whisper model"""
    
    def __init__(
        self,
        dinov3_model_name: str = "dinov3_vits16",
        dinov3_freeze: bool = True,
        visual_projection_dim: Optional[int] = None,
        cross_attn_dim: Optional[int] = None,
        add_cross_attn_to_encoder: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dinov3_model_name = dinov3_model_name
        self.dinov3_freeze = dinov3_freeze
        self.visual_projection_dim = visual_projection_dim or self.d_model
        self.cross_attn_dim = cross_attn_dim or self.d_model
        self.add_cross_attn_to_encoder = add_cross_attn_to_encoder


# Output dataclass for AV-DiCOW model
@dataclass
class AVDiCOWOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ctc: Optional[torch.FloatTensor] = None
    loss_att: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    encoder_logits: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AVWhisperModelForCTC(WhisperModel):
    """
    Audio-Visual Whisper Model for CTC + Attention Loss
    
    This model combines:
    1. DINOv3 visual encoder for processing lip crop images
    2. WhisperEncoderForCTC for audio processing with cross-attention to visual features
    3. Whisper decoder for autoregressive text generation
    """
    
    def __init__(self, config: AVDiCOWConfig):
        super(AVWhisperModelForCTC, self).__init__()
        
        self.config = config
        
        # Initialize the AV encoder (combines DINOv3 + WhisperEncoderForCTC)
        self.encoder = AVEncoderForCTC(config)
        
        # Initialize Whisper decoder
        from transformers.models.whisper.modeling_whisper import WhisperDecoder
        self.decoder = WhisperDecoder(config)
        
        # Initialize language model head for decoder
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def post_init(self):
        """Initialize weights after model creation"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for different module types"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()
    
    def freeze_visual_encoder(self):
        """Freeze visual encoder parameters"""
        self.encoder.freeze_visual_encoder()
    
    def freeze_audio_encoder(self):
        """Freeze audio encoder parameters"""
        self.encoder.freeze_audio_encoder()
    
    def train_visual_encoder(self):
        """Unfreeze visual encoder parameters"""
        self.encoder.train_visual_encoder()
    
    def train_audio_encoder(self):
        """Unfreeze audio encoder parameters"""
        self.encoder.train_audio_encoder()
    
    def forward(
        self,
        videos: torch.Tensor,
        audios: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        vad_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], AVDiCOWOutput]:
        """
        Forward pass through the audio-visual Whisper model
        
        Args:
            videos: Tensor of shape [batch, num_frames, channels, height, width]
            audios: Tensor of shape [batch, audio_features, audio_length] (Whisper mel features)
            labels: Optional tensor of shape [batch, max_label_length] for training
            video_lengths: Optional tensor of actual video frame lengths
            audio_lengths: Optional tensor of actual audio lengths
            label_lengths: Optional tensor of actual label lengths
            vad_mask: Optional VAD mask for target speaker amplification
            decoder_input_ids: Optional decoder input IDs for generation
            decoder_attention_mask: Optional decoder attention mask
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary output
            **kwargs: Additional arguments
            
        Returns:
            AVDiCOWOutput or tuple of tensors
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # Pass through AV encoder (DINOv3 + WhisperEncoderForCTC with cross-attention)
        encoder_outputs = self.encoder(
            video=videos,
            audio=audios,
            video_lengths=video_lengths,
            audio_lengths=audio_lengths,
            vad_mask=vad_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None:
            # For training, shift labels to create decoder input
            from transformers.models.whisper.modeling_whisper import shift_tokens_right
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        # Pass through decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Apply language model head
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # Compute losses if labels are provided
        loss = None
        loss_ctc = None
        loss_att = None
        
        if labels is not None:
            # CTC loss from encoder
            if hasattr(encoder_outputs, 'logits') and encoder_outputs.logits is not None:
                loss_ctc = self.encoder.audio_encoder.get_loss(encoder_outputs.logits, labels)
            
            # Attention loss from decoder
            if logits is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss_att = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            # Combined loss
            if loss_ctc is not None and loss_att is not None:
                loss = (1 - self.config.ctc_weight) * loss_att + self.config.ctc_weight * loss_ctc
            elif loss_att is not None:
                loss = loss_att
            elif loss_ctc is not None:
                loss = loss_ctc
        
        if not return_dict:
            outputs = (logits,)
            if output_hidden_states:
                outputs += (decoder_outputs.hidden_states, encoder_outputs.hidden_states)
            if output_attentions:
                outputs += (decoder_outputs.attentions, encoder_outputs.attentions)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        
        return AVDiCOWOutput(
            loss=loss,
            loss_ctc=loss_ctc,
            loss_att=loss_att,
            logits=logits,
            encoder_logits=encoder_outputs.logits if hasattr(encoder_outputs, 'logits') else None,
            decoder_hidden_states=decoder_outputs.hidden_states if output_hidden_states else None,
            encoder_hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            decoder_attentions=decoder_outputs.attentions if output_attentions else None,
            encoder_attentions=encoder_outputs.attentions if output_attentions else None,
            cross_attentions=decoder_outputs.cross_attentions if output_attentions else None,
        )


class AVDiCOWContainer:
    def __init__(self, model_type='openai/whisper-small.en', 
                 pretrained_encoder=None, 
                 dinov3_model_name='dinov3_vits16',
                 ctc_weight=0.1, **kwargs):
        
        # Load your AV-DiCOW model
        self.model = AVDiCOWForConditionalGeneration.from_pretrained(
            model_type,  # This will load base Whisper weights
            # Your custom parameters
            dinov3_model_name=dinov3_model_name,
            ctc_weight=ctc_weight,
            **kwargs
        )
        
        # Load DINOv3 weights separately
        self._load_dinov3_weights()
        
        # Load feature extractor and tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(model_type)