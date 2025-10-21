import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import jiwer
import webvtt
import json
from src.cluster.conv_spks import (
    get_clustering_f1_score,
    get_speaker_clustering_f1_score
)
import meeteval
from meeteval.viz.visualize import AlignmentVisualization
from src.tokenizer.norm_text import remove_disfluencies
import glob
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
text_normalizer = EnglishTextNormalizer({})
from datetime import timedelta

def convert_webvtt_to_seglst(ref_vtt, hypo_vtt, speaker="spk_0", session_id="session_40"):
    ref_segments = []
    for caption in webvtt.read(ref_vtt):
        start = (
            timedelta(hours=int(caption.start[:2]), minutes=int(caption.start[3:5]),
                    seconds=float(caption.start[6:]))
            .total_seconds()
        )
        end = (
            timedelta(hours=int(caption.end[:2]), minutes=int(caption.end[3:5]),
                    seconds=float(caption.end[6:]))
            .total_seconds()
        )
        ref_segments.append({
            "session_id": session_id,
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "words": remove_disfluencies(text_normalizer(caption.text))
        })
    # with open(f"/home/rphadke1/chime/misc_dump/{session_id}_{speaker}_ref.seglst.json", "w") as f:
    #     json.dump(ref_segments, f, indent=2)
    hypo_segments = []
    for caption in webvtt.read(hypo_vtt):
        start = (
            timedelta(hours=int(caption.start[:2]), minutes=int(caption.start[3:5]),
                    seconds=float(caption.start[6:]))
            .total_seconds()
        )
        end = (
            timedelta(hours=int(caption.end[:2]), minutes=int(caption.end[3:5]),
                    seconds=float(caption.end[6:]))
            .total_seconds()
        )
        hypo_segments.append({
            "session_id": session_id,
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "words": remove_disfluencies(text_normalizer(caption.text))
        })
    # with open(f"/home/rphadke1/chime/misc_dump/{session_id}_{speaker}_hypo.seglst.json", "w") as f:
    #     json.dump(hypo_segments, f, indent=2)
    return ref_segments, hypo_segments

def evaluate_conversation_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_clustering_f1_score(label_data, output_data)

def evaluate_speaker_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_speaker_clustering_f1_score(label_data, output_data)


def benchmark_vtt_wer(ref_vtt, hypo_vtt, ref_uem_start, ref_uem_end, hypo_uem_start, hypo_uem_end, show_diff=False):
    ref_strings = []
    hypo_strings = []
    for caption in webvtt.read(ref_vtt):
        if caption.start_in_seconds + caption.start_time.milliseconds/1000 < ref_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds/1000 > ref_uem_end:
            continue
        ref_strings.append(remove_disfluencies(text_normalizer(caption.text)))
    for caption in webvtt.read(hypo_vtt):
        if caption.start_in_seconds + caption.start_time.milliseconds/1000 < hypo_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds/1000 > hypo_uem_end:
            continue
        hypo_strings.append(remove_disfluencies(text_normalizer(caption.text)))
    
    if show_diff:
        # Show the WER error type (insertion, deletion, substitution) using wer library
        out = jiwer.process_words(
            [" ".join(ref_strings)],
            [" ".join(hypo_strings)],
        )
        print(jiwer.visualize_alignment(out))
        measures = jiwer.compute_measures(" ".join(ref_strings), " ".join(hypo_strings))
        print(f"Hits: {measures['hits']} | Substitutions: {measures['substitutions']} | Deletions: {measures['deletions']} | Insertions: {measures['insertions']}")
    return jiwer.wer(" ".join(ref_strings), " ".join(hypo_strings))

def evaluate_speaker_transcripts(label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end, session_id="session_40"):
    ref = []
    hypo = []
    speaker_to_wer = {}
    for speaker, uem_start, uem_end in zip(speaker_list, speaker_uem_start, speaker_uem_end):
        ref_vtt = os.path.join(label_path, f"{speaker}.vtt")
        hypo_vtt = os.path.join(output_path, f"{speaker}.vtt")
        wer_score = benchmark_vtt_wer(ref_vtt, hypo_vtt, uem_start, uem_end, uem_start, uem_end)
        speaker_to_wer[speaker] = round(wer_score, 4)
        ref_seglst, hypo_seglst = convert_webvtt_to_seglst(ref_vtt, hypo_vtt, speaker, session_id=session_id)
        ref.extend(ref_seglst)
        hypo.extend(hypo_seglst)
    return speaker_to_wer, ref, hypo

def compute_cperrors():
    wers = meeteval.wer.cpwer(meeteval.io.load(f'/home/rphadke1/chime/misc_dump/ref.seglst.json').sorted("start_time"), meeteval.io.load(f'/home/rphadke1/chime/misc_dump/hypo.seglst.json').sorted("start_time"))
    # wers = meeteval.wer.tcpwer(meeteval.io.load(f'/home/rphadke1/chime/misc_dump/ref.seglst.json').sorted("start_time"), meeteval.io.load(f'/home/rphadke1/chime/misc_dump/hypo.seglst.json').sorted("start_time"), collar=7)
    wers = sum(wers.values())
    cperror, t, i, d, s = wers.error_rate, wers.length, wers.insertions, wers.deletions, wers.substitutions
    print(f"CPErrorRate={cperror} | Substitutions={s/t} | Insertions={i/t} | Deletions={d/t}")

# def visualize_wer():
    # av = AlignmentVisualization(
    #     meeteval.io.load(f'/home/rphadke1/chime/misc_dump/{session_dir}_ref.seglst.json').groupby("session_id")[session_dir],
    #     meeteval.io.load(f'/home/rphadke1/chime/misc_dump/{session_dir}_hypo.seglst.json').groupby("session_id")[session_dir]
    # )
    # av.dump(f"/home/rphadke1/chime/misc_dump/{session_dir}_viz.html")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate speaker clustering and transcripts from video")
    parser.add_argument('--session_dir', type=str, required=True, help='Path to folder containing session data')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Name of the output directory within each session (default: output)')
    parser.add_argument('--label_dir_name', type=str, default='labels', help='Name of the label directory within each session (default: labels)')
    opt = parser.parse_args()

    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Evaluating {len(all_session_dirs)} sessions")

    all_conversation_clustering_f1_score = []
    all_speaker_wer = []
    all_cluster_speaker_wer = []

    refs, hypos = [], []
    for session_dir in all_session_dirs:
        session_id = session_dir.split('/')[-1]
        print(f"Evaluating session {session_dir.split('/')[-1]}")
        label_path = os.path.join(session_dir, opt.label_dir_name)
        output_path = os.path.join(session_dir, opt.output_dir_name)
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"
        assert os.path.exists(output_path), f"Output path {output_path} does not exist"
        
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        speaker_list = list(metadata.keys())
        speaker_uem_start = [metadata[spk]['central']["uem"]["start"] for spk in speaker_list]
        speaker_uem_end = [metadata[spk]['central']["uem"]["end"] for spk in speaker_list]

        conversation_clustering_f1_score = evaluate_conversation_clustering(label_path, output_path)
        print(f"Conversation clustering F1 score: {conversation_clustering_f1_score}")
        all_conversation_clustering_f1_score.append(conversation_clustering_f1_score)

        speaker_to_wer, ref, hypo = evaluate_speaker_transcripts(label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end, session_id=session_id)
        print(f"Speaker to WER: {speaker_to_wer}")
        all_speaker_wer.extend(list(speaker_to_wer.values()))

        refs.extend(ref)
        hypos.extend(hypo)

        speaker_clustering_f1_score = evaluate_speaker_clustering(label_path, output_path)
        print(f"Speaker clustering F1 score: {speaker_clustering_f1_score}")

        cluster_speaker_to_wer = {}
        for speaker, wer in speaker_to_wer.items():
            cluster_speaker_wer = 0.5 * wer + 0.5 * (1 - speaker_clustering_f1_score[speaker])
            cluster_speaker_to_wer[speaker] = cluster_speaker_wer
        print(f"Joint ASR-Clustering Error Rate: {cluster_speaker_to_wer}")
        all_cluster_speaker_wer.extend(list(cluster_speaker_to_wer.values()))
    with open("/home/rphadke1/chime/misc_dump/ref.seglst.json", "w") as f:
        json.dump(refs, f, indent=2)
    with open("/home/rphadke1/chime/misc_dump/hypo.seglst.json", "w") as f:
        json.dump(hypos, f, indent=2)
    compute_cperrors()
    print(f"Average Conversation Clustering F1 score: {sum(all_conversation_clustering_f1_score) / len(all_conversation_clustering_f1_score)}")
    print(f"Average Speaker WER: {sum(all_speaker_wer) / len(all_speaker_wer)}")
    print(f"Average Joint ASR-Clustering Error Rate: {sum(all_cluster_speaker_wer) / len(all_cluster_speaker_wer)}")

if __name__ == "__main__":
    main()