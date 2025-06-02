import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from src.dataset.avhubert_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from src.talking_detector.segmentation import segment_by_asd
from datasets import load_from_disk
import torch, torchvision, torchaudio
from src.cluster.conv_spks import (
    get_speaker_activity_segments, 
    calculate_conversation_scores, 
    cluster_speakers, 
    get_clustering_f1_score
)
import json
import math
from tqdm import tqdm
import glob

def load_model():
    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    # Load data collator
    audio_transform = AudioTransform(subset="test")
    video_transform = VideoTransform(subset="test")
    
    av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=audio_transform,
        video_transform=video_transform,
    )
    
    # Load model
    model_name = "./model-bin/avsr_cocktail"
    avsr_model = AVHubertAVSR.from_pretrained(model_name)
    avsr_model.eval().cuda()
    beam_search = get_beam_search_decoder(avsr_model.avsr, text_transform.token_list, beam_size=3)
    
    return avsr_model.avsr, text_transform, av_data_collator, beam_search
    

def inference(model, video, audio, text_transform, beam_search):
    avhubert_features = model.encoder(
        input_features = audio, 
        video = video,
    )
    audiovisual_feat = avhubert_features.last_hidden_state

    audiovisual_feat = audiovisual_feat.squeeze(0)

    nbest_hyps = beam_search(audiovisual_feat)
    nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
    predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
    predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
    return predicted

def chunk_video(video_path, asd_path=None, max_length=15):
    # load video and split into chunks for inference
    if asd_path is not None:
        with open(asd_path, "r") as f:
            asd = json.load(f)

        # Convert frame numbers to integers and sort them
        frames = sorted([int(f) for f in asd.keys()])
        # Find the minimum frame number to normalize frame indices
        min_frame = min(frames)

        segments_by_frames = segment_by_asd(asd)
        # Normalize frame indices, for inference, don't care about the actual frame indices
        segments = [((seg[0] - min_frame) / 25, (seg[-1] - min_frame) / 25) for seg in segments_by_frames]

    else:
        # Get video duration
        audio, rate = torchaudio.load(video_path)
        video_duration = audio.shape[1] / rate
        # num chunks
        num_chunks = math.ceil(video_duration / max_length)
        chunk_size = math.ceil(video_duration / num_chunks)
        segments = []
        # Convert to integer steps for range
        steps = int(video_duration * 100)  # Convert to centiseconds for precision
        step_size = int(chunk_size * 100)
        for i in range(0, steps, step_size):
            start_time = i / 100
            end_time = min((i + step_size) / 100, video_duration)
            segments.append((start_time, end_time))
        
    # if len(segments) > 0:
    #     print(f"Total segments: {len(segments)}")
    #     for idx, seg in enumerate(segments):
    #         print(f"Segment {idx}: {seg[0]:.2f}s - {seg[-1]:.2f}s, len: {seg[-1] - seg[0]:.2f}s")
    #     print(f"Max segment length: {max([seg[-1] - seg[0] for seg in segments]):.2f}s")
    #     print(f"Min segment length: {min([seg[-1] - seg[0] for seg in segments]):.2f}s")
    return segments


def format_vtt_timestamp(timestamp):
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp - int(timestamp)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def infer_video(
    model, 
    text_transform, 
    av_data_collator, 
    beam_search, 
    video_path, asd_path=None, offset=0.
):
    
    segments = chunk_video(video_path, asd_path)
    segment_output = []
    for seg in tqdm(segments, desc="Processing segments", total=len(segments)):
        # Inference
        sample = {
            "video": video_path,
            "start_time": seg[0],
            "end_time": seg[1],
        }
        sample_features = av_data_collator([sample])
        audios = sample_features["audios"].cuda()
        videos = sample_features["videos"].cuda()
        audio_lengths = sample_features["audio_lengths"].cuda()
        video_lengths = sample_features["video_lengths"].cuda()        
        output = inference(model, videos, audios, text_transform, beam_search)
        segment_output.append(output)

    return [
        {
            "start_time": seg[0] + offset,
            "end_time": seg[1] + offset,
            "text": output
        } for seg, output in zip(segments, segment_output)
    ]

def mcorec_session_infer(model, text_transform, av_data_collator, beam_search, session_dir, output_dir):
    # Infer session
    with open(os.path.join(session_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    # Process speaker clustering
    speaker_segments = {}
    for speaker_name, speaker_data in metadata.items():
        list_tracks_asd = []
        for track in speaker_data['central']['crops']:
            list_tracks_asd.append(os.path.join(session_dir, track['asd']))
        uem_start = speaker_data['central']['uem']['start']
        uem_end = speaker_data['central']['uem']['end']
        speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)
        speaker_segments[speaker_name] = speaker_activity_segments
    scores = calculate_conversation_scores(speaker_segments)
    clusters = cluster_speakers(scores, list(speaker_segments.keys()))   
    output_clusters_file = os.path.join(output_dir, "speaker_to_cluster.json")
    with open(output_clusters_file, "w") as f:
        json.dump(clusters, f, indent=4)    
    
    # Process speaker transcripts
    for speaker_name, speaker_data in tqdm(metadata.items(), desc="Processing speakers", total=len(metadata)):
        speaker_track_hypotheses = []
        for track in speaker_data['central']['crops']:
            video_path = os.path.join(session_dir, track['lip'])
            asd_path = os.path.join(session_dir, track['asd']) if 'asd' in track else None
            with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
                crop_metadata = json.load(f)
            track_start_time = crop_metadata['start_time']
            hypotheses = infer_video(model, text_transform, av_data_collator, beam_search, video_path, asd_path, offset=track_start_time)
            speaker_track_hypotheses.extend(hypotheses)

        output_file = os.path.join(output_dir, f"{speaker_name}.vtt")
        with open(output_file, "w") as f:
            f.write("WEBVTT\n\n")
            for hyp in speaker_track_hypotheses:
                text = hyp["text"].strip().replace("<unk>", "").strip()
                start_time = format_vtt_timestamp(hyp["start_time"])
                end_time = format_vtt_timestamp(hyp["end_time"])
                if len(text) == 0:
                    continue
                f.write(f"{start_time} --> {end_time}\n{text}\n\n")
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Infering speaker clustering and transcripts from video")
    parser.add_argument('--session_dir', type=str, required=True, help='Path to folder containing session data')
    # parser.add_argument('--session_dir', type=str, default="/home/tbnguyen/workspaces/mcorec_baseline/data-bin/mcorec_release/dev/session_40", help='Path to folder containing session data')
    opt = parser.parse_args()

    # Load model    
    model, text_transform, av_data_collator, beam_search = load_model()
    
    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Infering {len(all_session_dirs)} sessions")

    for session_dir in all_session_dirs:
        output_dir = os.path.join(session_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Infering session {session_dir.split('/')[-1]}")
        mcorec_session_infer(model, text_transform, av_data_collator, beam_search, session_dir, output_dir)

if __name__ == "__main__":
    main()
        
 