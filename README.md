# CHiME-9 Task 1 - MCoRec baseline
### ASR for multimodal conversations in smart glasses
This repository contains the baseline system for CHiME-9 challenge, Task 1 MCoRec. For information on how to participate in the challenge and how to get the development and evaluation datasets, please refer to the [challenge website](https://www.chimechallenge.org/current/task1/index). 

This README is a guide on how to install and run the baseline system for the challenge. This baseline system uses a pre-trained model.

## Sections
1. <a href="#install">Installation</a>
2. <a href="#dataset">Download MCoRec dataset</a>
3. <a href="#running">Running the baseline system</a>
4. <a href="#evaluation">Evaluation</a>
5. <a href="#results">Results</a>

## <a id="install">1. Installation </a>

Clone and install enviroment and download pre-trained model

```sh
# Clone the baseline code repo
git clone https://github.com/nguyenvulebinh/mcorec_baseline.git
cd mcorec_baseline

# Create Conda environment
conda create --name mcorec python=3.11
conda activate mcorec

# Install dependencies
pip install -r requirements.txt

# Download and unzip the pre-trained model
wget https://huggingface.co/nguyenvulebinh/mcorec_baseline/resolve/main/model-bin.zip
unzip model-bin.zip
```

## <a id="dataset">2. Download MCoRec dataset</a>

To use the MCoRec dataset, you first need to request access and then download it using your Hugging Face token.

-  **Request Access:**
    - Go to the [MCoRec dataset repository](https://huggingface.co/datasets/nguyenvulebinh/mcorec) on Hugging Face.
    - Request access to the dataset. You will need a Hugging Face account.

-  **Get Your Hugging Face Token:**

    If you don't have one, create a Hugging Face User Access Token. Follow the instructions [here](https://huggingface.co/docs/hub/security-tokens) to obtain your `HF_TOKEN`. Make sure it has read permissions.

-  **Download and Unzip Dataset Files:**

    It's recommended to download and unzip the data into a `data-bin` directory within your `mcorec_baseline` project folder.

```sh
# Navigate to your baseline project directory if you aren't already there
# cd mcorec_baseline 

# Create a data directory (if it doesn't exist)
mkdir -p data-bin
cd data-bin

# Export your Hugging Face token (replace 'your_actual_token_here' with your actual token)
export HF_TOKEN=your_actual_token_here

# Download the development set
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/nguyenvulebinh/mcorec/resolve/main/dev_without_central_videos.zip

# Unzip the downloaded files
unzip dev_without_central_videos.zip

# Download the train set TBU
# Download the evaluation set TBU
```


## <a id="running">3. Running the baseline system</a>

- The baseline system assumes the following directory layout:

    ```sh
    ├── data-bin
    │   ├── dev
    │   │   ├── session_132
    │   │   │    ├── central_video.mp4
    │   │   │    ├── labels
    │   │   │    │   ├── speaker_to_cluster.json
    │   │   │    │   ├── spk_0.vtt
    │   │   │    │   ├── spk_1.vtt
    │   │   │    │   ├── spk_2.vtt
    │   │   │    │   ├── spk_3.vtt
    │   │   │    │   ├── spk_4.vtt
    │   │   │    │   └── spk_5.vtt
    │   │   │    ├── metadata.json
    │   │   │    └── speakers
    │   │   │        ├── spk_0
    │   │   │        ├── spk_1
    │   │   │        ├── spk_2
    │   │   │        ├── spk_3
    │   │   │        ├── spk_4
    │   │   │        │   └── central_crops
    │   │   │        │       ├── track_00_asd.json
    │   │   │        │       ├── track_00_bbox.json
    │   │   │        │       ├── track_00.json
    │   │   │        │       ├── track_00_lip.av.mp4
    │   │   │        │       ├── track_00.mp4
    │   │   │        │       ├── track_01_asd.json
    │   │   │        │       ├── track_01_bbox.json
    │   │   │        │       ├── track_01.json
    │   │   │        │       ├── track_01_lip.av.mp4
    │   │   │        │       └── track_01.mp4
    │   │   │        └── spk_5
    │   │   └── ...
    │   ├── train
    │   └── eval
    ├── model-bin
    ├── script
    │   ├── inference.py
    │   ├── evaluate.py
    │   ├── asd.py
    │   ├── lip_crop.py
    │   └── train.py
    └── src
    ```

    `data-bin/`: This is where the MCoRec dataset (downloaded in Section 2) should be placed. It's organized into dev, train, and eval subsets. Each session folder (e.g., session_132) contains the raw and preprocessed data required for inference and evaluation.

    `model-bin/`: Contains the pre-trained model components, which were downloaded and unzipped during the [Section 1](#1-installation).
    
    `script/`: Contains the main Python scripts for performing tasks such as inference (inference.py), evaluation (evaluate.py), and data preprocessing (asd.py, lip_crop.py).

    `src/`: Contains the underlying Python codebase and modules for the MCoRec baseline system.

- Running Inference

    The `inference.py` script processes a session (or multiple sessions) and generates ASR output.

    * **Input**: The path to a session folder (e.g., `data-bin/dev/session_132/`) or a pattern matching multiple session folders.
    * **Output**: An `output` folder will be created inside each processed session directory (e.g., `data-bin/dev/session_132/output/`). This folder will contain the system's hypotheses, structured similarly to the ground truth `labels` folder.

    * **Commands:**

        ```sh

        # Infer single session
        python script/inference.py --session_dir path_to_session_folder 
        # Example: python script/inference.py --session_dir data-bin/dev/session_132

        # Infer all sessions
        python script/inference.py --session_dir "path_to_set_folder/*"
        # Example: python script/inference.py --session_dir "data-bin/dev/*"
        ```

- Data Preprocessing Details (Optional)

    The MCoRec dataset provided for the challenge typically comes with pre-generated files necessary for the baseline system. Specifically, within each speaker's `central_crops` folder (e.g., `data-bin/dev/session_132/speakers/spk_4/central_crops/`), you will find multiple video tracks (e.g., `track_00.mp4`, `track_01.mp4`). These are 96x96 face crop videos of the target speaker (e.g., `spk_4`).

    For each of these tracks (e.g., `track_00.mp4`), two important processed files are expected by the inference script:

    * `track_00_asd.json`: Contains active speaker detection scores for each frame of the corresponding track video. These scores are used to determine when a speaker is talking, allowing the audio-visual speech recognition system to run only during the speaking segments.
    * `track_00_lip.av.mp4`: A video similar to the face track, but specifically cropped around the lip region.

    **The provided dataset should already include these `_asd.json` and `_lip.av.mp4` files.** The scripts below are available if you wish to understand how they are generated, regenerate them, or process custom data.

    * To generate `_asd.json` (Active Speaker Detection scores): This script takes a track video as input and outputs a JSON file with frame-wise speaking scores.

        ```sh
        python script/asd.py --video path_to_the_track_video

        # Example: !python script/asd.py --video data-bin/dev/session_132/speakers/spk_4/central_crops/track_00.mp4
        ```

    * To generate `_lip.av.mp4` (Lip Crop video):
    This script takes a track video as input and outputs a new video cropped to the lip region.

        ```sh
        python script/lip_crop.py --video path_to_the_track_video

        # Example: !python script/lip_crop.py --video data-bin/dev/session_132/speakers/spk_4/central_crops/track_00.mp4
        ```

    You can use the pre-packaged processed data by default or choose to modify/regenerate these files if needed for your experiments.

- **[Run Example in Google Colab](https://colab.research.google.com/drive/1LmFdN-gEfC6Xt-4xRSz_VJyZiiMsVLGY?usp=sharing)**: Interactive notebook for a step-by-step run-through.


## <a id="#evaluation">4. Evaluation</a>

The `evaluate.py` script calculates performance metrics based on the system's output and the ground truth labels.

* **Input**: The path to a session folder (which should contain both the `labels` and the system-generated `output` subdirectories) or a pattern matching multiple session folders.
* **Output**: Evaluation metrics.

    * Conversation clustering F1 score
    * Speaker's WER
    * Speaker's clustering f1 score 
    * Speaker's WER based clustering: 0.5 * WER + 0.5 * (1 - f1_score) 

* **Commands:**

    ```sh

    # Evaluating single session
    python script/evaluate.py --session_dir path_to_session_folder 
    # Example: python script/evaluate.py --session_dir data-bin/dev/session_132

    # Output:
    # Evaluating 1 sessions
    # Evaluating session session_132
    # Conversation clustering F1 score: 1.0
    # Speaker to WER: {'spk_0': 0.4405, 'spk_1': 0.6137, 'spk_2': 0.5257, 'spk_3': 0.5556, 'spk_4': 0.7073, 'spk_5': 0.7102}
    # Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0, 'spk_5': 1.0}
    # Cluster speaker to WER: {'spk_0': 0.22025, 'spk_1': 0.30685, 'spk_2': 0.26285, 'spk_3': 0.2778, 'spk_4': 0.35365, 'spk_5': 0.3551}
    # Average conversation clustering F1 score: 1.0
    # Average speaker WER: 0.5922
    # Average Speaker's WER based clustering: 0.2961


    # Evaluating all sessions
    python script/evaluate.py --session_dir "path_to_set_folder/*"
    # Example: python script/evaluate.py --session_dir "data-bin/dev/*"
    ```

## <a id="#results">5. Results</a>

The results for the baseline model on dev subset are the following:

- Average Conversation clustering F1 score: 0.8553
- Average Speaker WER: 0.5674
- Average Speaker's WER based clustering: 0.3651

## Acknowledgement

This repository is built using the [auto_avsr](https://github.com/mpc001/auto_avsr), [espnet](https://github.com/espnet/espnet), and [avhubert](https://github.com/facebookresearch/av_hubert) repositories.

## License
MCoRec is CC-BY-NC licensed, as found in the LICENSE file.