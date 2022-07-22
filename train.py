# Train one model for each task
from behavioural_cloning import behavioural_cloning_train

def main():
    print("===Training FindCave model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindCave.weights"
    )

    print("===Training MakeWaterfall model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    print("===Training CreateVillageAnimalPen model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    print("===Training BuildVillageHouse model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )



if __name__ == "__main__":
    main()
