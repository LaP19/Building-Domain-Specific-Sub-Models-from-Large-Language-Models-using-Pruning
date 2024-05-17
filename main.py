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

def get_llm(model_name, cache_dir="$HF_HOME"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

import os
from pathlib import Path
from typing import Optional

# Model weights
def get_weight_dir(
        model_ref: str,
        hf_cache_dir: Optional[os.PathLike]=os.environ.get("HF_HOME", None),
        revision: str='main'
        ) -> Path:
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
    return model_weights_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--pruning_dataset', type=str, help='Calibration set')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--seqlen', type=int, default=4096, help='Sequence Length')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    torch.cuda.empty_cache()
    model_name = args.model.split("/")[-1]
    print("Questo è il nuovo branch")
    print(f"loading llm model {args.model}")
    weights_dir = get_weight_dir(args.model)
    model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.float16, device_map = "auto")
    print("DONE!")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    model.seqlen = args.seqlen
    print(model.seqlen)
    print("DONE!")

    print(model)
    print(model.hf_device_map)
    print(model.device)


    device = torch.device("cuda:0")

    if "30b" in args.model or "70b" in args.model or "Mistral" in args.model or "Mixtral" in args.model or "CodeLlama" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
        print("use device: ", device)
    #else:
        #model.to(device)

    # print(model.hf_device_map)
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
