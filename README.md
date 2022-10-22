# NeurIPS 2022: MineRL BASALT Behavioural Cloning Baseline

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

This repository contains a behavioural cloning baseline solution for the MineRL BASALT 2022 Competition ("basalt" track)! This solution fine-tunes the "width-x1" models of OpenAI VPT for more sample-efficient training.

You can find the "intro" track baseline solution [here](https://github.com/minerllabs/basalt-2022-intro-track-baseline).

MineRL BASALT is a competition on solving human-judged tasks. The tasks in this competition do not have a pre-defined reward function: the goal is to produce trajectories that are judged by real humans to be effective at solving a given task.

See [the AICrowd competition page](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition) for further details on the competition.


## Downloading the BASALT dataset

You can find the index files containing all the download URLs for the full BASALT dataset in the [OpenAI VPT repository at the bottom](https://github.com/openai/Video-Pre-Training#basalt-2022-dataset).

We have included a download utility (`utils/download_dataset.py`) to help you download the dataset. You can use it with the index files from the OpenAI VPT repository. For example, if you download the FindCave dataset index file, named `find-cave-Jul-28.json`, you can download first 100 demonstrations to `MineRLBasaltFindCave-v0` directory with:

```
python download_dataset.py --json-file find-cave-Jul-28.json --output-dir MineRLBasaltFindCave-v0 --num-demos 100
```

Basic dataset statistics (note: one trajectory/demonstration may be split up into multiple videos):
```
Size  #Videos  Name
--------------------------------------------------
146G  1399     MineRLBasaltBuildVillageHouse-v0
165G  2833     MineRLBasaltCreateVillageAnimalPen-v0
165G  5466     MineRLBasaltFindCave-v0
175G  4230     MineRLBasaltMakeWaterfall-v0
```



## Setting up

Install [MineRL v1.0.0](https://github.com/minerllabs/minerl) (or newer) and the requirements for [OpenAI VPT](https://github.com/openai/Video-Pre-Training).

Download the dataset following above instructions. Also download the 1x width foundational model [.weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights) and [.model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) files for the OpenAI VPT model.

Place these data files under `data` to match the following structure:

```
├── data
│   ├── MineRLBasaltBuildVillageHouse-v0
│   │   ├── Player70-f153ac423f61-20220707-111912.jsonl
│   │   ├── Player70-f153ac423f61-20220707-111912.mp4
│   │   └── ... rest of the files
│   ├── MineRLBasaltCreateVillageAnimalPen-v0
│   │   └── ... files as above
│   ├── MineRLBasaltFindCave-v0
│   │   └── ... files as above
│   ├── MineRLBasaltMakeWaterfall-v0
│   │   └── ... files as above
│   └── VPT-models
│       ├── foundation-model-1x.model
│       └── foundation-model-1x.weights
```


## Training models

Running following code will save a fine-tuned network for each task under `train` directory. This has been tested to fit into a 8GB GPU.

```
python train.py
```

## Visualizing/enjoying/evaluating models

To run the trained model for `MineRLBasaltFindCave-v0`, run the following:

```
python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights train/MineRLBasaltFindCave.weights --env MineRLBasaltFindCave-v0 --show
```

Change `FindCave` to other tasks to run for different tasks.

## How to Submit a Model on AICrowd.

**Note:** This repository is *not* submittable as-is. You first need to train the models, add them to the git repository and then submit to AICrowd.

To submit this baseline agent follow the [submission instructions](https://github.com/minerllabs/basalt_2022_competition_submission_template/), but use this repo instead of the starter kit repo.
