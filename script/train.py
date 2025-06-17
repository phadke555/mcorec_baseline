import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torch
from datasets import load_from_disk
from src.dataset.avhubert_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from transformers import TrainingArguments
from src.custom_trainer import AVSRTrainer
from transformers.trainer_utils import IntervalStrategy
from torchsummary import summary
import safetensors.torch
import datasets

# NCCL_DEBUG=WARN OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3,5 torchrun --nproc_per_node 2 train.py
# os.environ['WANDB_PROJECT'] = 'mcorec'


def load_avsr_dataset(cache_dir='data-bin/cache', streaming=False):
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
            finished_loading = True
        except Exception as e:
            try_times += 1
            if try_times >= max_try_times:
                raise e
            time.sleep(10)
    
    if not streaming:
        # That mean above datasets are already downloaded and cached
        # Now init again with streaming=True
        # lrs2 = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
        # vox2 = datasets.load_dataset("nguyenvulebinh/AVYT", "vox2", streaming=True, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
        # avyt = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt", streaming=True, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
        # avyt_mix = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt-mix", streaming=True, cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
        for ds in [lrs2, vox2, avyt, avyt_mix]:
            for split in ds.keys():
                split_size = len(ds[split])
                if split_size > 10000:
                    num_shards = max(20, split_size // 10000)
                else:
                    num_shards = 1
                ds[split] = ds[split].to_iterable_dataset(num_shards=num_shards)
                print(f"Split {split} has {split_size} samples and {ds[split].num_shards} shards")

    map_datasets = {
        "lrs2": {
            "probabilities": 0.3,
            "dataset": {
                "train": datasets.concatenate_datasets([
                    lrs2["train"], 
                    lrs2["pretrain"]
                ]),
                "valid": datasets.concatenate_datasets([
                    lrs2["valid"], 
                    lrs2["test_snr_0_interferer_2"]
                ])
            },
        },
        "vox2": {
            "probabilities": 0.2,
            "dataset": {
                "train": vox2["dev"],
                "valid": None,
            },
        },
        "avyt": {
            "probabilities": 0.25,
            "dataset": {
                "train": datasets.concatenate_datasets([
                    avyt['talking'], 
                    avyt['silent']
                ]),
                "valid": None,
            },
        },
        "avyt-mix": {
            "probabilities": 0.25,
            "dataset": {
                "train": avyt_mix["train"],
                "valid": avyt_mix["test"],
            },
        },
    }
    
    train_dataset = datasets.interleave_datasets([item['dataset']['train'] for item in map_datasets.values()], 
                                                 seed=101,
                                                 probabilities=[item['probabilities'] for item in map_datasets.values()], stopping_strategy='all_exhausted')
    valid_dataset = datasets.interleave_datasets([item['dataset']['valid'] for item in map_datasets.values()  if item['dataset']['valid'] is not None],
                                                 stopping_strategy='first_exhausted')
    
    train_dataset = train_dataset.map(format_sample)
    valid_dataset = valid_dataset.map(format_sample)
    
    # load lrs2 for interference speech
    # interference_speech = None
    print("Loading interference speech dataset. Actual file around 10GB need to download. This may take a while...")
    interference_speech = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", cache_dir=cache_dir, data_files='lrs2/lrs2-train-*.tar').remove_columns(['__key__', '__url__'])['train']
    return train_dataset, valid_dataset, interference_speech

if __name__ == "__main__":
    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    avsr_config = AVHubertAVSRConfig(odim=len(text_transform.token_list))
    avsr_model = AVHubertAVSR(avsr_config)
    
    
    # Load pretrained checkpoint
    encoder_pretrained_checkpoint = "nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h" # AVHubert encoder original (https://facebookresearch.github.io/av_hubert/)
        
    encoder_pretrained = avsr_model.avsr.encoder.from_pretrained(
        encoder_pretrained_checkpoint, 
        cache_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model-bin")
    )
    avsr_model.avsr.encoder.load_state_dict(encoder_pretrained.state_dict())
    
    
    # Load dataset
    train_dataset, valid_dataset, interference_dataset = load_avsr_dataset(streaming=True)
    
    
    train_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="train", speech_dataset=interference_dataset),
        video_transform=VideoTransform(subset="train"),
    )
    valid_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="test"),
        video_transform=VideoTransform(subset="test"),
    )
    
    
    print("train_dataset\n", train_dataset)
    print("valid_dataset\n", valid_dataset)
    summary(avsr_model)
    
    ############ Debugging ############
    # # model_name = "./model-bin/avsr_cocktail"
    # # avsr_model = AVHubertAVSR.from_pretrained(model_name)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # avsr_model.eval().cuda()
    # batch_size = 6
    # batch_samples = []
    # for sample in train_dataset:
    #     batch_samples.append(sample)
    #     if len(batch_samples) == batch_size:
    #         break
    # features = train_av_data_collator(batch_samples)
    # for key in features:
    #     if isinstance(features[key], torch.Tensor):
    #         if key in ["videos", "audios"]:
    #             features[key] = features[key]
    #         features[key] = features[key].cuda()
    # output = avsr_model(**features)
    # print(output)
    # exit()
    ##################################

    
    batch_size = 6
    max_steps = 200000
    gradient_accumulation_steps = 2
    save_steps = 2000
    eval_steps = 2000
    log_interval = 25
    learning_rate = 1e-4
    warmup_steps = 4000
    checkpoint_name = "avhubert_avsr_cocktail"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model-bin/{checkpoint_name}")
    
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
        # report_to="wandb",  # enable logging to W&B,
        report_to="none",
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
        eval_dataset=valid_dataset,
    )
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    
    