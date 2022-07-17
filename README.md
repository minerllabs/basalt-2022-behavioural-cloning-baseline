# NeurIPS 2022: MineRL BASALT Behavioural Cloning Baseline

## NOTE: this code will be updated with the final submission kit once submissions open.


[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

This repository contains a behavioural cloning baseline solution for the MineRL BASALT 2022 Competition ("basalt" track)!

There will be a separate baseline solution for the "intro" track of the competition.

MineRL BASALT is a competition on solving human-judged tasks. The tasks in this competition do not have a pre-defined reward function: the goal is to produce trajectories that are judged by real humans to be effective at solving a given task.

See [the AICrowd competition page](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition) for further details on the competition.


## Setting up

Install [MineRL v1.0.0](https://github.com/minerllabs/minerl) (or newer) and the requirements for [OpenAI VPT](https://github.com/openai/Video-Pre-Training).

Download the dummy BASALT dataset from [here](https://microsofteur-my.sharepoint.com/:f:/g/personal/t-anssik_microsoft_com/Ej9R17fChVVLtPZQmnA233ABmhtzPBnS-v0BOv6na8_IZA?e=izua7z) (password: `findcave2022`). Also download the 1x width foundational model [.weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights) and [.model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) files for the OpenAI VPT model.

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

To run the trained model for `MineRLBasaltFindCave-v0`, run the following (after loading up it should show a screen of the agent going around):

```
python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights train/MineRLBasaltFindCave.weights --env MineRLBasaltFindCave-v0
```

Change `FindCave` to other tasks to run for different tasks.

## How to Submit a Model on AICrowd.

To submit this baseline agent follow the [submission instructions](https://github.com/minerllabs/basalt_2022_competition_submission_template/), but use this repo instead of the starter kit repo.
