from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        weights="train/MineRLBasaltCreateVillageAnimalPen.weights",
        env="MineRLBasaltCreateVillageAnimalPen-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS
    )
