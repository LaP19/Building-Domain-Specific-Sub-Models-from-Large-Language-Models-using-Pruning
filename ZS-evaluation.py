#!/usr/bin/env python3

import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


import os
from pathlib import Path
from typing import Optional


# Model weights
def get_weight_dir_big(
        model_ref: str,
        hf_cache_dir: Optional[os.PathLike] = os.environ.get("HF_HOME", None),
        revision: str = 'main'):
    """
    Convenience function for retrieving locally stored HF weights.
    """
    if hf_cache_dir is None:
        hf_cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    model_path = "--".join(['models'] + model_ref.split('/'))
    snapshot = (hf_cache_dir / f'{model_path}/refs/{revision}').read_text()
    model_weights_dir = hf_cache_dir / f"{model_path}/snapshots/{snapshot}"
    model_name = f"{model_path}/snapshots/{snapshot}"
    return model_weights_dir, model_name


def get_weight_dir(
        dataset: str,
        hf_cache_dir: Optional[os.PathLike] = os.environ.get("HF_HOME", None)):
    if hf_cache_dir is None:
        hf_cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    model_path = 'Gemma-Instruct-7b-pruned_' + dataset
    model_weights_dir = hf_cache_dir / f"{model_path}"
    return model_weights_dir, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = "", type=str, help='LLaMA model')
    parser.add_argument('--dataset', default="", type=str, help='pruning dataset')
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--eval_zero_shot", default=True, action="store_true")
    parser.add_argument("--csr", default=False, action="store_true")
    parser.add_argument("--math", default=False, action="store_true")
    args = parser.parse_args()

    if args.model == "":
        weights_dir, model_name = get_weight_dir(args.dataset)

    else:
        weights_dir, model_name = get_weight_dir_big(args.model)

    model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.float16, device_map="auto")
    print("DONE!")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    model.seqlen = model.config.max_position_embeddings
    print(model.seqlen)
    print("DONE!")

    print(model)

    device = torch.device("cuda:0")

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        if args.csr:
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        elif args.math:
            task_list = ["math_algebra", "math_counting_and_prob", "math_geometry", "math_intermediate_algebra", "math_num_theory", "math_prealgebra", "math_precalc", "math_asdiv"]
        num_shot = 0
        results = eval_zero_shot(weights_dir, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

if __name__ == '__main__':
    main()
