import os
import torch
import torchaudio
import torchvision
from torchcodec.decoders import VideoDecoder
# check if AudioDecoder is available
try:
    from torchcodec.decoders import AudioDecoder
except ImportError:
    AudioDecoder = None

import random
from dataclasses import dataclass
from src.tokenizer.spm_tokenizer import TextTransform
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from transformers import WhisperProcessor
import torch.nn.functional as F

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video_rgb(path, start_time=0, end_time=None):
    """
    Load video and return RGB frames (not grayscale).
    rtype: torch.Tensor, T x C x H x W (C=3 for RGB)
    """
    video_decoder = VideoDecoder(path, dimension_order="NHWC")
    if end_time is None:
        end_time = video_decoder.metadata.duration_seconds
    vid_rgb = video_decoder.get_frames_played_in_range(start_time, end_time).data
    
    # Convert to RGB tensor: (T, H, W, C) -> (T, C, H, W)
    vid_tensor = torch.from_numpy(vid_rgb.numpy()).permute(0, 3, 1, 2)
    return vid_tensor


def load_audio(path, start_time=0, end_time=None):
    """
    Load audio from video file.
    rtype: torch.Tensor, T x 1 (mono audio at 16kHz)
    """
    if AudioDecoder is not None:
        audio_decoder = AudioDecoder(path)
        if end_time is None:
            end_time = audio_decoder.metadata.duration_seconds
        audio = audio_decoder.get_audio_played_in_range(start_time, end_time).data
        # Ensure mono audio
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)
        return audio
    else:
        # Fallback to torchaudio
        waveform, sample_rate = torchaudio.load(path, start=start_time, end=end_time)
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).unsqueeze(1)  # (T, 1)


class AddMultiSpk:
    """Add multiple speaker interference to audio."""
    def __init__(self, speech_dataset=None, interferer_spk=None, snr_levels=[-5, 0, 5, 10, 15]):
        self.snr_levels = snr_levels
        self.interferer_spk = [interferer_spk] if interferer_spk else [0, 0, 1, 2]
        self.speech_dataset = speech_dataset

    def __call__(self, speech):
        """
        Add interference to speech.
        Args:
            speech: torch.Tensor of shape (T,) - mono audio waveform
        Returns:
            torch.Tensor of shape (T,) - audio with interference
        """
        if self.speech_dataset is None:
            return speech
        
        speech_length = speech.size(0) / 16000
        if speech_length < 2:
            return speech
        
        num_interferer = random.choice(self.interferer_spk)
        interferer_signal = None
        
        for _ in range(num_interferer):
            # Get random interferer from dataset
            interferer_sample = random.choice(self.speech_dataset)
            interferer = load_audio(interferer_sample['video'])
            interferer_length = interferer.size(0) / 16000
            
            if 2 <= interferer_length <= 10:
                # Pad or trim interferer to match speech length
                interferer = cut_or_pad(interferer, len(speech))
                if interferer_signal is None:
                    interferer_signal = interferer
                else:
                    # Mix multiple interferers
                    snr_level = torch.tensor([random.choice([-5, 0, 5, 10, 15])])
                    interferer_signal = torchaudio.functional.add_noise(
                        interferer_signal.t(), interferer.t(), snr_level
                    ).t()
        
        if interferer_signal is None:
            return speech
        
        # Add interference to original speech
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        speech_with_interference = torchaudio.functional.add_noise(
            speech.t(), interferer_signal.t(), snr_level
        ).t()
        
        return speech_with_interference


class AddNoise:
    """Add background noise to audio."""
    def __init__(self, snr_target=None):
        self.snr_target = snr_target

    def __call__(self, speech):
        """
        Add noise to speech.
        Args:
            speech: torch.Tensor of shape (T,) - mono audio waveform
        Returns:
            torch.Tensor of shape (T,) - audio with noise
        """
        if self.snr_target is None:
            return speech
        
        # Generate white noise
        noise = torch.randn_like(speech)
        snr_level = torch.tensor([self.snr_target])
        
        # Add noise
        speech_with_noise = torchaudio.functional.add_noise(
            speech.t(), noise.t(), snr_level
        ).t()
        
        return speech_with_noise


class VideoTransform:
    """Video transform for RGB video processing."""
    def __init__(self, subset, target_size=224):
        self.target_size = target_size
        if subset == "train":
            self.video_pipeline = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(target_size),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
            ])
        elif subset == "val" or subset == "test":
            self.video_pipeline = torchvision.transforms.Compose([
                torchvision.transforms.Resize(target_size),
                torchvision.transforms.CenterCrop(target_size),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, video_frames):
        """
        Apply transforms to video frames.
        Args:
            video_frames: torch.Tensor of shape (T, C, H, W) where C=3 for RGB
        Returns:
            torch.Tensor of shape (T, C, H, W) with normalized RGB frames
        """
        # Normalize to [0, 1] first if needed
        if video_frames.max() > 1.0:
            video_frames = video_frames / 255.0
        
        # Apply transforms frame by frame
        transformed_frames = []
        for frame in video_frames:
            # frame: (C, H, W)
            transformed_frame = self.video_pipeline(frame)
            transformed_frames.append(transformed_frame)
        
        return torch.stack(transformed_frames, dim=0)  # (T, C, H, W)


def pad_video_batch(videos, video_lengths, max_frames=None):
    """
    Pad video batch to the same temporal length.
    Args:
        videos: List of video tensors, each of shape (T_i, C, H, W)
        video_lengths: List of actual frame counts
        max_frames: Optional maximum frames to pad to
    Returns:
        padded_videos: torch.Tensor of shape (B, C, T, H, W)
        video_lengths: torch.Tensor of shape (B,)
    """
    if max_frames is None:
        max_frames = max(video_lengths)
    
    B = len(videos)
    C, H, W = videos[0].shape[1], videos[0].shape[2], videos[0].shape[3]
    
    padded_videos = torch.zeros(B, C, max_frames, H, W, dtype=videos[0].dtype)
    
    for i, (video, length) in enumerate(zip(videos, video_lengths)):
        if length > max_frames:
            # Truncate if too long
            video = video[:max_frames]
            length = max_frames
        
        padded_videos[i, :, :length] = video.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
    
    return padded_videos, torch.tensor(video_lengths, dtype=torch.long)


@dataclass
class AVWhisperDataCollator:
    """
    Data collator for AVWhisper model that produces Whisper-compatible features.
    """
    processor: WhisperProcessor
    text_transform: TextTransform
    video_transform: VideoTransform
    max_video_frames: Optional[int] = None
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into the format expected by AVWhisper.
        
        Expected input format:
        {
            "video": str,  # path to video file
            "label": str,  # transcript text
            "start_time": float,  # optional
            "end_time": float,  # optional
        }
        
        Returns:
        {
            "input_features": torch.Tensor,  # (B, 80, 3000) - Whisper log-mel features
            "labels": torch.Tensor,          # (B, seq_len) - token IDs
            "video": torch.Tensor,           # (B, C, T, H, W) - RGB video frames
            "video_lengths": torch.Tensor,   # (B,) - actual video lengths
        }
        """
        audios = []
        videos = []
        video_lengths = []
        texts = []
        
        for feature in features:
            if "start_time" in feature and "end_time" in feature:
                video = load_video_rgb(feature["video"], feature["start_time"], feature["end_time"])
            else:
                video = load_video_rgb(feature["video"])
            if "start_time" in feature and "end_time" in feature:
                audio = load_audio(feature["video"], feature["start_time"], feature["end_time"])
            else:
                audio = load_audio(feature["video"])
            
            # Apply video transforms
            video = self.video_transform(video)  # (T, C, H, W)
            
            # Store video info
            videos.append(video)
            video_lengths.append(video.shape[0])
            
            # Store audio (will be processed by WhisperProcessor)
            audios.append(audio.squeeze(1).numpy())  # Convert to numpy for processor
            
            # Store text
            texts.append(feature["label"])
        
        # Process audio with WhisperProcessor
        # This will give us the exact input_features format Whisper expects
        audio_features = self.processor.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        input_features = audio_features.input_features  # (B, 80, 3000)
        
        # Process text with tokenizer
        text_features = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448  # Whisper's max target length
        )
        labels = text_features.input_ids  # (B, seq_len)
        
        # Pad video batch
        video_batch, video_lengths_tensor = pad_video_batch(
            videos, video_lengths, self.max_video_frames
        )
        
        return {
            "input_features": input_features,
            "labels": labels,
            "video": video_batch,
            "video_lengths": video_lengths_tensor,
        }


# Utility function to create the collator
def create_avwhisper_collator(
    processor: WhisperProcessor,
    text_transform: TextTransform,
    subset: str = "train",
    max_video_frames: Optional[int] = None
) -> AVWhisperDataCollator:
    """
    Create an AVWhisperDataCollator with appropriate transforms.
    
    Args:
        processor: WhisperProcessor instance
        text_transform: TextTransform for text processing
        subset: "train", "val", or "test"
        max_video_frames: Maximum number of video frames to pad to
    
    Returns:
        AVWhisperDataCollator instance
    """
    video_transform = VideoTransform(subset=subset)
    
    return AVWhisperDataCollator(
        processor=processor,
        text_transform=text_transform,
        video_transform=video_transform,
        max_video_frames=max_video_frames,
        sampling_rate=16000
    )
