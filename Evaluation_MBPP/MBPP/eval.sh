MODEL_NAME_OR_PATH="/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/f02cb64b091c07f4e96b79960fd89caf434578e0"
DATASET_ROOT="data/"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=1,2,3 python -m accelerate.commands.launch --config_file test_config.yaml eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --dataroot ${DATASET_ROOT}