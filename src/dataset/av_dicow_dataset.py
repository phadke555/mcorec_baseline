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
from python_speech_features import logfbank
import torch.nn.functional as F
from transformers import WhisperProcessor
from transformers import AutoImageProcessor

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


def load_video(path, start_time=0, end_time=None):
    """
    rtype: torch, T x C x H x W
    """
    video_decoder = VideoDecoder(path, dimension_order="NHWC")
    vid_end = video_decoder.metadata.duration_seconds
    if end_time is None or end_time > vid_end:
        end_time = video_decoder.metadata.duration_seconds
    if start_time > end_time:
        start_time = end_time - 1.0
    vid_rgb = video_decoder.get_frames_played_in_range(start_time, end_time).data
    vid = vid_rgb.permute(0, 3, 1, 2)
    return vid


def load_audio(path, start_time=0, end_time=None):
    if AudioDecoder is not None:
        audio_decoder = AudioDecoder(path)
        if end_time is None:
            end_time = audio_decoder.metadata.duration_seconds_from_header
        waveform = audio_decoder.get_samples_played_in_range(start_time, end_time).data
    else:
        if start_time == 0 and end_time is None:
            frame_offset = 0
            num_frames = -1
        else:
            frame_offset = int(start_time * 16000)
            num_frames = int((end_time - start_time) * 16000)
        waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames, normalize=True)
        assert sample_rate == 16000
    return waveform.transpose(1, 0)  # T x 1


class WhisperFeatureModule(torch.nn.Module):
    """
    Wraps WhisperProcessor.feature_extractor to produce [T, 80] log-Mel features
    from a mono 16k waveform shaped [T, 1].
    """
    def __init__(self, processor: WhisperProcessor):
        super().__init__()
        self.processor = processor  # holds feature_extractor under the hood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, 1] mono @ 16k
        # Convert to 1D float32 CPU numpy for Whisper feature extractor
        wav = x.squeeze(-1) if x.ndim == 2 else x  # [T]
        wav = wav.detach().cpu().numpy().astype(np.float32)

        # Whisper feature extractor returns [1, 80, n_frames]
        feats = self.processor.feature_extractor(
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features  # shape: [1, 80, n_frames]

        # Convert to [T, 80] to match your existing collator expectations
        feats = feats[0].transpose(0, 1).contiguous()  # [n_frames, 80] == [T, 80]
        return feats

def normalize_audio(waveform):
    max_val = torch.abs(waveform).max()
    return waveform / max_val if max_val > 0 else waveform

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=None,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if noise_filename is None:
            # self.noise = torch.randn(1, 16000)
            self.noise = None
        else:
            self.noise, sample_rate = torchaudio.load(noise_filename)
            assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()
    
class AddMultiSpk(torch.nn.Module):
    def __init__(
        self,
        speech_dataset=None,
        snr_target=None,
        interferer_spk=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20]
        self.interferer_spk = [interferer_spk] if interferer_spk else [0, 0, 1, 2]
        self.speech_dataset = speech_dataset

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.speech_dataset is None:
            return speech
        speech_length = speech.size(0) / 16000
        if speech_length < 2:
            return speech
        
        num_interferer = random.choice(self.interferer_spk)
        interferer_signal = None
        for _ in range(num_interferer):
            interferer = load_audio(random.choice(self.speech_dataset)['video'])
            interferer_length = interferer.size(0) / 16000
            # print(interferer, interferer_length)
            if 2 <= interferer_length <= 10:
                interferer = cut_or_pad(interferer, len(speech))
                if interferer_signal is None:
                    interferer_signal = interferer
                else:
                    snr_level = torch.tensor([random.choice([-5, 0, 5, 10, 15])])
                    interferer_signal = torchaudio.functional.add_noise(interferer_signal.t(), interferer.t(), snr_level).t()        
        
        if interferer_signal is None:
            return speech
        # print(f"Adding {num_interferer} interferer(s) to speech with length {speech_length:.2f}s")
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        speech = torchaudio.functional.add_noise(speech.t(), interferer_signal.t(), snr_level).t()
        
        return speech

class DinoFastVideoTransform(torch.nn.Module):
    """
    Apply the DINOv3 Fast ImageProcessor to a sequence of frames.
    Input:  frames  [T, C, H, W]  (uint8 0..255 or float 0..255/0..1 okay)
    Output: frames  [T, C, 224, 224] normalized with ImageNet stats
    """
    def __init__(self, pretrained_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"):
        super().__init__()
        # Fast processor is selected automatically for this repo
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        assert frames.ndim == 4 and frames.size(1) in (1, 3), "Expect [T,C,H,W] with C=1 or 3"
        # Convert to a list of CHW tensors (what the processor expects)
        # NOTE: Keep them on CPU for the processor; we'll move back to device after.
        device = frames.device
        frames_cpu = frames.detach().to("cpu")
        images = [frames_cpu[t] for t in range(frames_cpu.size(0))]  # list[CHW]

        # Fast path: rescale -> resize(224) -> normalize
        batch = self.processor(
            images=images,
            return_tensors="pt",   # gives pixel_values: [T, C, 224, 224]
        )
        pixel_values = batch["pixel_values"]  # float32, CPU
        return pixel_values.to(device)        # back to original device

class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = DinoFastVideoTransform()
        elif subset == "val" or subset == "test":
            self.video_pipeline = DinoFastVideoTransform()

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, speech_dataset=None, snr_target=None, whisper_processor: WhisperProcessor = None):
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddMultiSpk(speech_dataset=speech_dataset),
                AddNoise(),
                WhisperFeatureModule(whisper_processor),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                WhisperFeatureModule(whisper_processor),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)



# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -100 if data_type == "label" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s" if data_type != "stno_mask" else "stno_mask"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out
    
@dataclass
class DataCollator:
    text_transform: TextTransform = None
    video_transform: VideoTransform = None
    audio_transform: AudioTransform = None
    rate_ratio: int = 640
    whisper_processor: WhisperProcessor = None
    model_features_subsample_factor: int = 2
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # {"video": video, "audio": audio, "target": token_id}
        samples = []
        for feature in features:
            if "start_time" in feature and "end_time" in feature:
                if feature["start_time"] > feature["end_time"]:
                    continue
                video = load_video(feature["video"], feature["start_time"], feature["end_time"])
                audio = load_audio(feature["audio"], feature["start_time"], feature["end_time"])
            else:
                video = load_video(feature["video"])
                audio = load_audio(feature["audio"])
                
                
            # audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            # whisper_audio_len = 30 * 16000  # 480000 samples
            # video_target_len = whisper_audio_len // self.rate_ratio  # 480000 / 640 = 750 frames
            # video = cut_or_pad(video, video_target_len)
            
            video = self.video_transform(video)
            audio = self.audio_transform(audio)

            # --- VAD Mask Handling ---
            if "vad_mask" in feature:
                vad_mask = feature["vad_mask"]
                # Pad to match features
                pad_len = (self.whisper_processor.feature_extractor.n_samples - vad_mask.shape[-1]) % self.whisper_processor.feature_extractor.n_samples
                vad_mask = np.pad(vad_mask, ((0, 0), (0, pad_len)), mode='constant')

                # Downsample to meet model features sampling rate
                vad_mask = vad_mask.astype(np.float32).reshape(vad_mask.shape[0], -1,
                    self.model_features_subsample_factor * self.whisper_processor.feature_extractor.hop_length).mean(axis=-1)
                vad_mask = torch.from_numpy(vad_mask)
            else:
                vad_mask = None
            # --- ---

            if "label" in feature:
                label_ids = self.whisper_processor.tokenizer(
                    feature["label"],
                    add_special_tokens=True
                ).input_ids
                label = torch.tensor(label_ids, dtype=torch.long)
                if "vad_mask" in feature:
                    samples.append({"vision_feature": video, "input_feature": audio, "stno_mask": vad_mask, "label": label})
                else:
                    samples.append({"vision_feature": video, "input_feature": audio, "label": label})
            else:
                if "vad_mask" in feature:
                    samples.append({"vision_feature": video, "input_feature": audio, "stno_mask": vad_mask})
                else:
                    samples.append({"vision_feature": video, "input_feature": audio})
        
        batch = collate_pad(samples)
        
        batch['vision_features'] = batch['vision_features'].permute(0, 2, 1, 3, 4)
        # target_len = 1500
        # batch['vision_features'] = F.interpolate(
        #     batch['vision_features'], size=(target_len, 224, 224),
        #     mode='trilinear', align_corners=False
        # )
        batch['input_features'] = batch['input_features'].permute(0, 2, 1)
        return batch