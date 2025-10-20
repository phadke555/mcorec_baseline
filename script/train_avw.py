import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torch
from datasets import load_from_disk
from src.dataset.avwhisper_dataset import create_avwhisper_collator
from src.tokenizer.spm_tokenizer import TextTransform
from src.av_whisper.av_whisper_model import AVWhisperForConditionalGeneration
from transformers import TrainingArguments, WhisperProcessor, WhisperConfig
from transformers import Trainer
from transformers.trainer_utils import IntervalStrategy
from torchsummary import summary
import safetensors.torch
import datasets
import time
import argparse

os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"  # seconds
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ['WANDB_PROJECT'] = 'mcorec'

# NCCL_DEBUG=WARN OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 script/train.py \
# --streaming_dataset \
# --include_mcorec \
# --batch_size 6 \
# --max_steps 400000 \
# --gradient_accumulation_steps 2 \
# --save_steps 2000 \
# --eval_steps 2000 \
# --log_interval 25 \
# --learning_rate 1e-4 \
# --warmup_steps 4000 \
# --checkpoint_name mcorec_finetuning \
# --model_name_or_path ./model-bin/avsr_cocktail \
# --output_dir ./model-bin



def load_avsr_dataset(cache_dir='data-bin/cache', include_mcorec=True, streaming=False):
    # streaming=True to avoid downloading all dataset at once, but it can be crash if network is unstable
    # streaming=False to download all dataset at once, it take time and around 1.5TB disk space. More stable.

    def format_sample(sample):
        sample['label'] = str(sample['label'], encoding='utf-8')
        sample['length'] = int(sample['length'])
        sample['sample_id'] = str(sample['sample_id'], encoding='utf-8')
        return sample
    
    # Load dataset
    finished_loading = False
    try_times = 0
    max_try_times = 5

    while not finished_loading:
        try:
            # Load dataset. It's quite bigdataset and sometime downloading can break. You can simple retry.
            lrs2 = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=streaming, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            vox2 = datasets.load_dataset("nguyenvulebinh/AVYT", "vox2", streaming=streaming, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            avyt = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt", streaming=streaming, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            avyt_mix = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt-mix", streaming=streaming, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            # Load mcorec dataset. Ensure you have permission to use this dataset.
            if include_mcorec:
                print("Loading MCoRec dataset")
                mcorec_dataset = datasets.load_dataset("MCoRecChallenge/MCoRec", streaming=streaming, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            finished_loading = True
        except Exception as e:
            try_times += 1
            if try_times >= max_try_times:
                raise e
            time.sleep(10)
    
    if not streaming:
        # That mean above datasets are already downloaded and cached
        list_datasets = [lrs2, vox2, avyt, avyt_mix]
        if include_mcorec:
            list_datasets.append(mcorec_dataset)
        for ds in list_datasets:
            for split in ds.keys():
                split_size = len(ds[split])
                if split_size > 10000:
                    num_shards = max(20, split_size // 10000)
                else:
                    num_shards = 1
                ds[split] = ds[split].to_iterable_dataset(num_shards=num_shards)
                print(f"Split {split} has {split_size} samples and {ds[split].num_shards} shards")

    if include_mcorec:
        map_dataset_probabilities = {
            "lrs2": 0.25,
            "vox2": 0.10,
            "avyt": 0.20,
            "avyt-mix": 0.25,
            "mcorec": 0.2,
        }
    else:
        map_dataset_probabilities = {
            "lrs2": 0.3,
            "vox2": 0.2,
            "avyt": 0.25,
            "avyt-mix": 0.25,
        }
    
    map_datasets = {
        "lrs2": {
            "probabilities": map_dataset_probabilities["lrs2"],
            "dataset": {
                "train": datasets.concatenate_datasets([
                    lrs2["train"], 
                    lrs2["pretrain"]
                ]),
                "valid": datasets.concatenate_datasets([
                    lrs2["valid"], 
                    lrs2["test_snr_0_interferer_2"]
                ]) if not include_mcorec else None
            },
        },
        "vox2": {
            "probabilities": map_dataset_probabilities["vox2"],
            "dataset": {
                "train": vox2["dev"],
                "valid": None,
            },
        },
        "avyt": {
            "probabilities": map_dataset_probabilities["avyt"],
            "dataset": {
                "train": datasets.concatenate_datasets([
                    avyt['talking'], 
                    avyt['silent']
                ]),
                "valid": None,
            },
        },
        "avyt-mix": {
            "probabilities": map_dataset_probabilities["avyt-mix"],
            "dataset": {
                "train": avyt_mix["train"],
                "valid": avyt_mix["test"] if not include_mcorec else None,
            },
        },
        "mcorec": {
            "probabilities": map_dataset_probabilities["mcorec"] if include_mcorec else 0,
            "dataset": {
                "train": mcorec_dataset["train"] if include_mcorec else None,
                "valid": mcorec_dataset["valid"] if include_mcorec else None,
            },
        }
    }
    print("map_datasets\n", map_datasets)
    
    train_dataset = datasets.interleave_datasets([item['dataset']['train'] for item in map_datasets.values() if item['dataset']['train'] is not None], 
                                                 seed=11,
                                                 probabilities=[item['probabilities'] for item in map_datasets.values() if item['dataset']['train'] is not None], 
                                                 stopping_strategy='all_exhausted')
    valid_dataset = datasets.interleave_datasets([item['dataset']['valid'] for item in map_datasets.values() if item['dataset']['valid'] is not None],
                                                 stopping_strategy='first_exhausted')
    
    train_dataset = train_dataset.map(format_sample)
    valid_dataset = valid_dataset.map(format_sample)
    
    # load lrs2 for interference speech
    # interference_speech = None
    print("Loading interference speech dataset. Actual file around 10GB need to download. This may take a while...")
    interference_speech = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", cache_dir=cache_dir, data_files='lrs2/lrs2-train-*.tar').remove_columns(['__key__', '__url__'])['train']
    return train_dataset, valid_dataset, interference_speech


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--streaming_dataset", action="store_true", default=False)
    parser.add_argument("--include_mcorec", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_name", type=str, default="mcorec_finetuning")
    parser.add_argument("--model_name_or_path", type=str, default="./model-bin/avwhisper") # Or None to train from scratch
    parser.add_argument("--report_to", type=str, default="none") # wandb or none
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model-bin"))
    parser.add_argument("--max_video_frames", type=int, default=300, help="Maximum number of video frames to process")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small", help="Whisper model to use as base")

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
    max_video_frames = args.max_video_frames
    whisper_model = args.whisper_model
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    # Load WhisperProcessor
    print(f"Loading WhisperProcessor from {whisper_model}...")
    processor = WhisperProcessor.from_pretrained(whisper_model)
    
    # Load AVWhisper model
    if model_name_or_path is not None and os.path.exists(model_name_or_path):
        print("Loading pretrained AVWhisper model from", model_name_or_path)
        # Note: You'll need to implement from_pretrained for AVWhisperForConditionalGeneration
        # For now, we'll load from scratch and you can add checkpoint loading later
        config = WhisperConfig.from_pretrained(whisper_model)
        avwhisper_model = AVWhisperForConditionalGeneration(config)
    else:
        # Load from scratch
        print("Loading AVWhisper model from scratch")
        config = WhisperConfig.from_pretrained(whisper_model)
        avwhisper_model = AVWhisperForConditionalGeneration(config)
        
        # Load pretrained Whisper weights (optional - you can skip this for now)
        print("Loading pretrained Whisper weights...")
        try:
            from transformers import WhisperForConditionalGeneration
            pretrained_whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model)
            # Copy decoder weights (but not the encoder since we have a custom AVEncoder)
            avwhisper_model.model.decoder.load_state_dict(pretrained_whisper.model.decoder.state_dict())
            # Copy the lm_head weights
            avwhisper_model.proj_out.load_state_dict(pretrained_whisper.proj_out.state_dict())
            print("Loaded pretrained Whisper decoder and projection weights")
        except Exception as e:
            print(f"Could not load pretrained Whisper weights: {e}")
            print("Training from scratch...")
    
    # Load dataset
    train_dataset, valid_dataset, interference_dataset = load_avsr_dataset(streaming=streaming_dataset, include_mcorec=include_mcorec)
        
    # Create AVWhisper data collators
    print("Creating AVWhisper data collators...")
    train_av_data_collator = create_avwhisper_collator(
        processor=processor,
        text_transform=text_transform,
        subset="train",
        max_video_frames=max_video_frames
    )
    valid_av_data_collator = create_avwhisper_collator(
        processor=processor,
        text_transform=text_transform,
        subset="test",
        max_video_frames=max_video_frames
    )
    
    
    print("train_dataset\n", train_dataset)
    print("valid_dataset\n", valid_dataset)
    summary(avwhisper_model)
    
    
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
        fp16=True,
        gradient_checkpointing=False, 
        remove_unused_columns=False,
        dataloader_num_workers=10,
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
    
    # Custom compute metrics function for evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # For now, just return a simple metric
        # You can implement WER/CER calculation here later
        return {"eval_loss": 0.0}
    
    trainer = Trainer(
        model=avwhisper_model,
        data_collator=train_av_data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    if not resume_from_checkpoint:
        trainer.train()
    else:
        print("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)