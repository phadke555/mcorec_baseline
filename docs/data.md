---
layout: page
title: Data
parent: CHiME-9 Task 1 - MCoRec
nav_order: 1
---

The main [MCoRec dataset](#) for this task consists of recording sessions between multiple conversation partners using phones and 360 camera. Phone will be used to record egocentric view of each participant while 360 camera will be used to record every participants at once. The collected data is split into training, development, and evaluation subsets. The evaluation subset will remain hidden until shortly before the final submission of the systems. In the baseline system, a training subset is used to build the system, while the results are reported on the development subset. The participants can use the training subsets to build their systems and the development set to evaluate and compare to the baseline. No training or automatic tuning is allowed on the development set.

|             | number of sessions | number of conversations | duration | avg duration per recording | number of speakers |
| ----------- | :--------------------:| :--------------------: | :--------: | :--------------------------: | :------------------: |
| train       | 56              |  120  | 5.6 h    | 6 min                      | 20                 |
| dev         | 25               | 60  | 2.5 h   | 6 min                      | 12                 |
| eval        | 69                | 157 | 6.9 h    | 6 min                      | 24                 |

Besides the MCoRec dataset, public datasets as listed in the [Rules](rules) page can be used.

In addition to the MCoRec dataset, we also publish [AVYT dataset](#), which can help the participants to build their systems. The AVYT dataset contains 1500 hours of audio-visual (face crops) speech recognition dataset.

## Description of the MCoRec data

The MCoRec dataset consists of 56 recordings for training (released on July 1th 2025), 25 recordings for development (released on July 1th 2025) and 69 recordings for evaluation (will be released on TBC). One recording session features a fews conversations, each have two or more participants. One 360 camera place in the middle of the room to capture the central view which contants every participants. Each participant have a phone to capture the egocentric view. The phone placed close to and in front of the speaker, offering a clear view of the speaker’s face. For system development and evaluation only the recordings of central view are provided. Note that the number of people show in the central can be bigger than the number of paticipants which need to be processed. The target participants will be provided by the bbox.

## Detailed desciption of data structure and formats

The directory structure is as follows:
```
Example of session with 8 speakers separated into 2 conversations.

session_id
├── central_video.mp4
└── conversations
    ├── 00
    │   ├── spk_1
    │   │   ├── central
    │   │   │   ├── bbox.json
    │   │   │   └── uem.json
    │   │   ├── ego
    │   │   │   ├── bbox.json
    │   │   │   ├── uem.json
    │   │   │   └── video.mp4
    │   │   └── transcript.vtt
    │   ├── spk_2
    │   ├── spk_3
    │   └── spk_4
    └── 01
        ├── spk_5
        ├── spk_6
        ├── spk_7
        └── spk_8
```

Each `spk_*` folder holds three sub-components:

- **central**: Annotations for the target speaker in the central view
- **ego**: Egocentric video and annotations
- **transcript.vtt**: Time-aligned transcription of the speaker’s utterances

Below is a detailed description of each file type and its schema:

### 1. `bbox.json`

Stores bounding-box trajectories for a single speaker in a given video stream. Each entry uses the frame number (as a string) to map to a bounding box specification:

```json
{
  "0": { "x": 93, "y": 308, "w": 206, "h": 309 },
  "1": { "x": 93, "y": 309, "w": 205, "h": 309 },
  "...": { /* additional frames */ }
}
```

* Each **frame key** (string) maps to an object with:
  * **x**, **y**: top-left corner coordinates in pixels.
  * **w**, **h**: width and height of the bounding box in pixels.
* `bbox.json` in central folder mapping to `central_video.mp4` video.
* `bbox.json` in ego folder mapping to speaker's `video.mp4`.

### 2. `uem.json`

Defines Valid Sub-Intervals for processing (Unit Enable Mask).

```
{
    "start": 3.5,               // in seconds when the conversation start
    "stop": 363.9                 // in seconds when the conversation stop
}
```
* **start** / **stop**: floats representing seconds.

### 3. `video.mp4` (Ego Only)

Standard H.264-encoded MP4 video captured by the participant’s device.

* **Frame rate**: 25 fps
* **Resolution**: 720p.

### 4. `central_video.mp4`

Full-session central view video (360° perspective). All speakers appear.

* **Frame rate**: 25 fps.
* **Resolution**: 4K.

### 5. `transcript.vtt`

WebVTT format file for the speaker’s speech transcript.

```
WEBVTT

00:00:03.500 --> 00:00:05.200
Hello everyone,

00:00:06.000 --> 00:00:08.300
Welcome to the meeting.
```

* Each cue consists of:

  * **Time stamp**: `hh:mm:ss.mmm --> hh:mm:ss.mmm`
  * **Text lines**: the utterance content. Multiple lines may appear per cue.


## Getting the data

For obtaining the data, please refer to the download link at [this website](#). Note that a registration is needed to obtain the data and they should not be further distributed to not-registered individuals.