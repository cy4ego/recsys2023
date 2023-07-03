#!/bin/bash

N_TRIALS=100
EPOCHS=1000
FE=21
OUTPUT_DIR="/data/recsys/2023/output/tabsurvey/v24/"

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="pytorch_tabular"
KERAS_ENV="tensorflow-1.15.0"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

declare -A MODELS
MODELS=( # ["CatBoost"]=$GBDT_ENV
        #  ["XGBoost"]=$GBDT_ENV
        #  ["LightGBM"]=$GBDT_ENV          # 0.258779
        #  ["RandomForest"]=$SKLEARN_ENV   # 0.300401 
        #  ["LinearModel"]=$SKLEARN_ENV    # 0.441873
        #  ["DecisionTree"]=$SKLEARN_ENV   # 0.308201
        #  ["KNN"]=$SKLEARN_ENV            # 0.458835
        #  ["RLN"]=$KERAS_ENV              #
        #  ["DNFNet"]=$KERAS_ENV
        #  ["VIME"]=$TORCH_ENV
        #  ["TabNet"]=$TORCH_ENV
        #  ["TabTransformer"]=$TORCH_ENV
         ["SAINT"]=$TORCH_ENV
        #  ["NODE"]=$TORCH_ENV
        )

CONFIGS=( "config/recsys2023.yml"
          # "config/recsys2023_meta.yml"
          )

# conda init bash
eval "$(conda shell.bash hook)"

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS[$model]}"

    conda activate "${MODELS[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --fe $FE --output_dir $OUTPUT_DIR

    conda deactivate

  done

done
