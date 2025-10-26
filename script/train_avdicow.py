import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torch
from datasets import load_from_disk
from src.dataset.av_dicow_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.av_dicow.av_dicow_model import DiCoWForConditionalGeneration
from transformers import TrainingArguments, WhisperProcessor, WhisperConfig, AutoModel
from src.custom_trainer import AVSRTrainer
from transformers.trainer_utils import IntervalStrategy
from torchsummary import summary
import safetensors.torch
import datasets
import time
import argparse
import json
from pathlib import Path
from datasets import IterableDataset
import lhotse
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers.utils import logging
logging.set_verbosity_info()

os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"  # seconds
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ['WANDB_PROJECT'] = 'mcorec'

# NCCL_DEBUG=WARN OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 script/train_avdicow.py \
# --streaming_dataset \
# --include_mcorec \
# --batch_size 6 \
# --max_steps 400000 \
# --gradient_accumulation_steps 2 \
# --save_steps 2000 \
# --eval_steps 2000 \
# --log_interval 25 \
# --learning_rate 1e-8 \
# --warmup_steps 4000 \
# --checkpoint_name mcorec_finetuning \
# --output_dir ./model-bin

def stno_from_cut(cut, target_speaker=None):
    speakers = sorted({sup.speaker for sup in cut.supervisions})
    spk2idx = {spk: i for i, spk in enumerate(speakers)}

    # Lhotse built-in: binary (n_speakers, n_frames) mask
    vad = cut.speakers_audio_mask(speaker_to_idx_map=spk2idx)
    s_index = spk2idx[target_speaker]

    target = vad[s_index] == 1
    sil = vad.sum(axis=0) == 0
    non_target_mask = np.ones(vad.shape[0], dtype=bool)
    non_target_mask[s_index] = False
    diff = vad[non_target_mask].sum(axis=0) > 0
    overlap = np.logical_and(diff, target)
    non_target = diff & ~target
    target = target & ~overlap

    return np.stack([sil, target, non_target, overlap], axis=0)


def load_lhotse_iterable_dataset(cutset_path="/export/fs06/rphadke1/data/mcorec/data-bin/manifests/mcorec_cuts_train.jsonl.gz", num_shards=1):
    """
    Creates a streaming IterableDataset from a Lhotse CutSet.
    Each yielded sample corresponds to one supervision (speech segment)
    and includes both audio and the correctly referenced video crop
    under 'central_crops' as specified in metadata.json.
    """
    cuts = lhotse.CutSet.from_file(cutset_path)

    def gen():
        for cut in cuts:
            if cut.duration > 30.0:
                continue
            audio_path = Path(cut.recording.sources[0].source)
            session_dir = audio_path.parent  # e.g., .../train/session_00
            session_id = session_dir.name

            metadata_path = session_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                metadata = json.load(open(metadata_path))
            except Exception:
                continue
            
            start_time = cut.start
            end_time = cut.start + cut.duration
            if start_time > end_time:
                continue
            for sup in cut.supervisions:
                speaker = sup.speaker
                video_path = Path(metadata[speaker]["central"]["video"])

                # Fallback: if missing, try track_00
                if not video_path.exists():
                    fallback = (
                        session_dir
                        / "speakers"
                        / speaker
                        / "central_crops"
                        / "track_00.mp4"
                    )
                    video_path = fallback if fallback.exists() else None

                if not video_path:
                    continue

                # try:
                #     # --- duration check ---
                #     video_decoder = VideoDecoder(str(video_path))
                #     vid_dur = video_decoder.get_duration_seconds()
                #     if end_time > vid_dur or start_time >= vid_dur:
                #         continue  # skip invalid video ranges
                #     video_decoder.close()
                # except Exception:
                #     continue

                vad_mask = stno_from_cut(cut, target_speaker=speaker)

                yield { 
                    "audio": str(audio_path),
                    "video": str(video_path),
                    "start_time": start_time,
                    "end_time": end_time,
                    "vad_mask": vad_mask,
                    "label": sup.text,
                    "speaker": speaker,
                    "session_id": session_id,
                    
                }

    return IterableDataset.from_generator(gen, features=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--streaming_dataset", action="store_true", default=False)
    parser.add_argument("--include_mcorec", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_name", type=str, default="mcorec_finetuning")
    parser.add_argument("--model_name_or_path", type=str) # Or None to train from scratch
    parser.add_argument("--report_to", type=str, default="none") # wandb or none
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model-bin"))
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-large-v3-turbo", help="Whisper model to use as base")
    parser.add_argument("--vision_model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m", help="DinoV3 Vision model to use as base")

    args = parser.parse_args()

    streaming_dataset = True if args.streaming_dataset else False
    include_mcorec = True if args.include_mcorec else False
    batch_size = args.batch_size
    max_steps = args.max_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    log_interval = args.log_interval
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    resume_from_checkpoint = True if args.resume_from_checkpoint else False
    checkpoint_name = args.checkpoint_name
    model_name_or_path = args.model_name_or_path # Or None to train from scratch
    output_dir = os.path.join(args.output_dir, checkpoint_name)
    report_to = args.report_to
    whisper_model = args.whisper_model
    vision_model = args.vision_model
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load Whisper Processor
    processor = WhisperProcessor.from_pretrained(whisper_model)
    if hasattr(processor, "tokenizer"):
        try:
            processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
        except Exception:
            pass

    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    if model_name_or_path is not None and os.path.exists(model_name_or_path):
        print("Loading pretrained DiCoW checkpoint from", model_name_or_path)
        avsr_model = DiCoWForConditionalGeneration.from_pretrained(model_name_or_path)
    else:
        print("Initializing DiCoW from DiCoW 3.2 and Whisper from:", args.whisper_model)
        # Prefer loading weights directly from HF base
        try:
            avsr_model = DiCoWForConditionalGeneration.from_pretrained("BUT-FIT/DiCoW_v3_2")
            avsr_model.tokenizer = processor.tokenizer
            vision_model = AutoModel.from_pretrained(vision_model)
            avsr_model.model.encoder.set_vision_encoder(vision_model)

        except:
            print("Error no model loaded because of error")
            exit()

    # Load dataset
    train_dataset = load_lhotse_iterable_dataset()
    valid_dataset = load_lhotse_iterable_dataset("/export/fs06/rphadke1/data/mcorec/data-bin/manifests/mcorec_cuts_dev.jsonl.gz")
    from itertools import islice
    valid_dataset_subset = datasets.IterableDataset.from_generator(
        lambda: islice(valid_dataset.__iter__(), 128)
    )
    train_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="train", whisper_processor=processor,),
        video_transform=VideoTransform(subset="train"),
        whisper_processor=processor,
    )
    valid_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="test", whisper_processor=processor,),
        video_transform=VideoTransform(subset="test"),
        whisper_processor=processor,
    )
    
    
    print("train_dataset\n", train_dataset)
    print("valid_dataset\n", valid_dataset)
    summary(avsr_model)
    
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "log"),
        # group_by_length=True,
        # length_column_name='length',
        label_names = ["labels"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # auto_find_batch_size = True,
        # max_grad_norm=0.1,
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        max_steps = max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        metric_for_best_model='loss',
        greater_is_better=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False, 
        remove_unused_columns=False,
        dataloader_num_workers=1,
        # save_only_model=True, # WARNING: this will save only model and not optimizer, scheduler, etc.
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=log_interval,
        learning_rate=learning_rate,
        weight_decay=0.005,
        warmup_steps=warmup_steps,
        save_total_limit=500,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        # save_safetensors=False,
        report_to=report_to,  # enable logging to W&B,
        # report_to="none",
        run_name=checkpoint_name,  # name of the W&B run (optional)
        accelerator_config={
            "dispatch_batches": False
        }
        # dispatch_batches=False
        # ddp_find_unused_parameters=True
    )
    
    trainer = AVSRTrainer(
        model=avsr_model,
        data_collator=train_av_data_collator,
        valid_data_collator=valid_av_data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset_subset,
    )

    if not resume_from_checkpoint:
        trainer.train()
    else:
        print("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)