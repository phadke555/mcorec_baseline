---
layout: page
title: Rules
parent: CHiME-9 Task 1 - MCoRec
nav_order: 3
---

Summary of the rules for systems participating in the challenge:

- For building the system, it is allowed to use the training subset of MCoRec dataset and external data listed in the subsection Data and pre-trained models. If you believe there is some public dataset missing, you can propose it to be added until the deadline as specified in the schedule.
- The development subset of MCoRec can be used for evaluating the system throughout the challenge period, but not for training or automatic tuning of the systems.
- Pre-trained models are listed in the “Data and pre-trained models” subsection. Only those pre-trained models are allowed to be used. If you believe there is some model missing, you can propose it to be added until the deadline as specified in the schedule.
If your system does not comply with these rules (e.g. by using a private dataset), you may still submit your system, but we will not include it in the final rankings.

## Evaluation

### Given:

- A video containing group conversations.
- Bounding boxes (bbox) for n interested speakers, indicating their locations throughout the video.

### System's output:

- A .vtt file for each speaker, containing their speech content, time-aligned to the video.
- A clustering output that indicates how many distinct conversations are present and which speakers belong to which conversation cluster.

### Metrics:

The system output will be evaluated on two key aspects:

1. Word Error Rate (WER) for Individual Speakers

    - Each .vtt file will be evaluated against ground-truth transcriptions for the corresponding speaker.
    - Standard WER will be computed per speaker.
    - The average WER across all speakers will be used.

2. Conversation Clustering Performance

    - The goal is to identify how many independent conversations took place and correctly group the involved speakers.
    - For evaluation, each speaker is assigned a cluster ID, indicating the conversation they belong to.
    - The clustering performance will be evaluated using pairwise F1 score, defined as follows:

        - For all pairs of speakers, consider whether the system and the ground truth agree on whether the pair is in the same conversation.
        - Compute precision and recall:
            - Precision = (# correctly predicted same-cluster pairs) / (# predicted same-cluster pairs)
            - Recall = (# correctly predicted same-cluster pairs) / (# actual same-cluster pairs)
            - F1 score = 2 * (Precision * Recall) / (Precision + Recall)



## External data and pre-trained models

Besides the MCoRec dataset published with this challenge, the participants are allowed to use public datasets and pre-trained models listed below. In case you want to propose additional dataset or pre-trained model to be added to these lists, do so by contacting us at [Slack]() until #TBU. If you want to use a private dataset or model, you may still submit your system to the challenge, but we will not include it in the final rankings.

Participants may use these publicly available datasets for building the systems:

- [AMI](https://groups.inf.ed.ac.uk/ami/corpus/)
- [LibriSpeech](https://www.openslr.org/12/)
- [TEDLIUM](https://www.openslr.org/51/)
- [MUSAN](https://www.openslr.org/17/)
- [RWCP Sound Scene Database](https://www.openslr.org/13/)
- [REVERB Challenge RIRs.](http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz)
- [Aachen AIR dataset.](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
- [BUT Reverb database.](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database)
- [SLR28 RIR and Noise Database (contains Aachen AIR, MUSAN noise, RWCP sound scene database and REVERB challenge RIRs, plus simulated ones).](https://www.openslr.org/28/)
- [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [FSD50k](https://zenodo.org/record/4060432)
- WSJ0-2mix, WHAM, [WHAMR](http://wham.whisper.ai/), WSJ
- [SINS](https://github.com/fgnt/sins)
- [LibriCSS acoustic transfer functions (ATF)](https://github.com/jsalt2020-asrdiar/jsalt2020_simulate/tree/master)
- [NOTSOFAR1 simulated CSS dataset](https://github.com/microsoft/NOTSOFAR1-Challenge#2-training-on-the-full-simulated-training-dataset)
- [Ego4D](https://ego4d-data.org/)
- [Project Aria Datasets](https://www.projectaria.com/datasets/)
- [DNS challenge noises](https://github.com/microsoft/DNS-Challenge)

In addition, following pre-trained models may be used:

- Wav2vec:
  - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wav2vec-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec-large)
- Wav2vec 2.0:
  - [Fairseq:](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20)
    - All models including [Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH)](https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv_ftsb300_updated.pt) and the multi-lingual [XLSR-53 56k](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)
  - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html)
    - [WAV2VEC2_BASE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-base)
    - [WAV2VEC2_LARGE](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large)
    - [WAV2VEC2_LARGE_LV60K](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-large-lv60k)
    - [WAV2VEC2_XLSR53](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-xlsr53)
    - [WAV2VEC2_ASR_LARGE_LV60K_960H](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-large-lv60k-960h)
    - [WAV2VEC2_ASR_BASE_960](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec2-asr-base-960h)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
    - [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
    - [facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)
    - [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
    - [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
    - [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60)
    - [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
    - [wav2vec2-large lv60 + speaker verification](https://huggingface.co/anton-l/wav2vec2-base-superb-sv)
    - Other models on Huggingface using the same weights as the [Fairseq ones](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20).
  - [S3PRL](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2)
    - [wav2vec2_base_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_base_960)
    - [wav2vec2_large_960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_960)
    - [wav2vec2_large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_ll60k)
    - [wav2vec2_large_lv60_cv_swbd_fsh](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_large_lv60_cv_swbd_fsh)
    - [wav2vec2_conformer_relpos](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_relpos)
    - [wav2vec2_conformer_rope](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2_conformer_rope)
    - [Xlsr_53](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#xlsr_53)
- [HuBERT](https://arxiv.org/abs/2106.07447)
  - [Torchaudio](https://pytorch.org/audio/0.10.0/pipelines.html#wav2vec-2-0-hubert-representation-learning)
    - [HuBERT base](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-base)
    - [HuBERT large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-large)
    - [HuBERT xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-xlarge)
    - [HuBERT ASR large](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-large)
    - [HuBERT ASR xlarge](https://pytorch.org/audio/0.10.0/pipelines.html#hubert-asr-xlarge)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/hubert)
    - [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
    - [hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k)
    - [hubert-xlarge-ll60k](https://huggingface.co/facebook/hubert-xlarge-ll60k)
    - [hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)
    - [hubert-xlarge-ls960-ft](https://huggingface.co/facebook/hubert-xlarge-ls960-ft)
  - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [hubert-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-base)
    - [hubert-large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#hubert-large_ll60k)
- [WavLM](https://arxiv.org/abs/2110.13900)
  - [Huggingface:](https://huggingface.co/docs/transformers/model_doc/wavlm)
    - [wavlm-base](https://www.chimechallenge.org/current/task1/rules#:~:text=Huggingface%3A-,wavlm%2Dbase,-wavlm%2Dbase%2Dsv)
    - [wavlm-base-sv](https://huggingface.co/microsoft/wavlm-base-sv)
    - [wavlm-base-sd](https://huggingface.co/microsoft/wavlm-base-sd)
    - [wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
    - [wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)
    - [wavlm-base-plus-sd](https://huggingface.co/microsoft/wavlm-base-plus-sd)
    - [wavlm-large](https://huggingface.co/microsoft/wavlm-large)
  - [S3PRL:](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
    - [wavlm-base](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base)
    - [wavlm-base-plus](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-base-plus)
    - [wavlm-large](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wavlm-large)
- [Tacotron2](https://github.com/NVIDIA/tacotron2)
  - [Torchaudio:](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-text-to-speech)
    - [tacotron2-wavernn-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-phone-ljspeech)
    - [tacotron2-griffinlim-phone-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-phone-ljspeech)
    - [tacotron2-wavernn-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-wavernn-char-ljspeech)
    - [tacotron2-griffinlim-char-ljspeech](https://pytorch.org/audio/0.10.0/pipelines.html#tacotron2-griffinlim-char-ljspeech)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
  - [Speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
  - [NeMo toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [X-vector extractor](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
  - [VBx repo ResNet101 16kHz](https://github.com/BUTSpeechFIT/VBx/tree/master/VBx/models/ResNet101_16kHz)
  - [Kaldi](https://kaldi-asr.org/models/m7)
  - [Pyannote Audio](https://huggingface.co/pyannote/embedding)
- [Pyannote Segmentation](https://huggingface.co/pyannote/segmentation)
- [Pyannote Diarization (Pyannote Segmentation+ECAPA-TDNN from SpeechBrain)](https://huggingface.co/pyannote/speaker-diarization)
- [NeMo toolkit ASR pre-trained models:](https://github.com/NVIDIA/NeMo)
  - [Citrinet](https://huggingface.co/nvidia/stt_en_citrinet_1024_gamma_0_25)
  - [Conformer-CTC](https://huggingface.co/nvidia/stt_en_conformer_ctc_large)
  - [Conformer-Transducer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-transducer)
  - [FastConformer](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [NeMo toolkit speaker ID embeddings models:](https://github.com/NVIDIA/NeMo)
  - [TitaNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#titanet)
  - [SpeakerNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet)
  - [ECAPA-TDNN](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_recognition/models.html#ecapa-tdnn)
- [NeMo toolkit VAD models:](https://github.com/NVIDIA/NeMo)
  - [MarbleNet VAD](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad)
- [NeMo toolkit diarization models:](https://github.com/NVIDIA/NeMo)
  - [Multi-Scale Diarization Decoder](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speaker_diarization/models.html#multi-scale-diarization-decoder)
- [Whisper](https://arxiv.org/abs/2212.04356)
  - [Whisper official repo (all versions: small, medium, large v1/v2/v3)](https://github.com/openai/whisper)
- [OWSM: Open Whisper-style Speech Model](https://www.wavlab.org/activities/2024/owsm/)
  - [OWSM v3 E-Branchformer](https://huggingface.co/espnet/owsm_v3.1_ebf)
  - [OWSM v3 with E-Branchformer (base, smaller version)](https://huggingface.co/espnet/owsm_v3.1_ebf_base)
  - [OWSM v3](https://huggingface.co/espnet/owsm_v3)
  - [OWSM v2](https://huggingface.co/espnet/owsm_v2)
  - [OWSM v2 E-Branchformer (note: this is probably the best OWSM for English ASR)](https://huggingface.co/espnet/owsm_v2_ebranchformer)
  - [OWSM v1](https://huggingface.co/espnet/owsm_v1)
- [Icefall Zipformer](https://huggingface.co/yfyeung/icefall-asr-gigaspeech-zipformer-2023-10-17)
- [RWKV Transducer](https://www.modelscope.cn/models/iic/speech_rwkv_transducer_asr-en-16k-gigaspeech-vocab5001-pytorch-online/summary)