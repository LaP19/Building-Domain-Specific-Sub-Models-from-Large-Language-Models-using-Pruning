import argparse
from collections import defaultdict
import torch.cuda.sparse
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = True
import os
from pathlib import Path
from typing import Optional
from lib.prune import prepare_calibration_input

from transformers import AutoModelForCausalLM, AutoTokenizer

#batch_sizes = [4, 8 , 16, 32, 64, 128]
batch_sizes = [1]
#sizes = [384, 768, 1024, 1280]
dense_results = defaultdict(list)
sparse_results = defaultdict(list)

def get_weight_dir_pruned(
        dataset: str,
        hf_cache_dir: Optional[os.PathLike] = os.environ.get("/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface", None),
        revision: str = 'main'
) -> Path:
    # Convenience function for retrieving locally stored HF weights.

    if hf_cache_dir is None:
        hf_cache_dir = Path("/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface").expanduser()
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    model_path = 'CodeLlama-Instruct-7b-pruned_' + dataset
    model_weights_dir = hf_cache_dir / f"{model_path}"
    return model_weights_dir

def get_weight_dir(
        model_ref: str,
        hf_cache_dir: Optional[os.PathLike]=os.environ.get("HF_HOME", None),
        revision: str='main'
        ) -> Path:

    #Convenience function for retrieving locally stored HF weights.

    if hf_cache_dir is None:
        hf_cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    model_path = "--".join(['models'] + model_ref.split('/'))
    snapshot = (hf_cache_dir / f'{model_path}/refs/{revision}').read_text()
    model_weights_dir = hf_cache_dir / f"{model_path}/snapshots/{snapshot}"
    return model_weights_dir

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="", type=str, help="model")
parser.add_argument("--pruning_dataset", default="", type=str, help="dataset")
args = parser.parse_args()

weights_dir = get_weight_dir_pruned(args.pruning_dataset)
# Perform inference with sparse model
model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.float16, device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(weights_dir)

weights_dir = get_weight_dir(args.model)
# Perform inference with sparse model
model_dense = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.float16, device_map = "auto")
tokenizer_dense = AutoTokenizer.from_pretrained(weights_dir)

dense_times = {}
sparse_times = {}
for batch_size in batch_sizes:

    x = torch.rand(2048, 4096).half().cuda()
    y = torch.rand(2048, 11008).half().cuda()
    z = torch.rand(2048, 4096).half().cuda()

    for name, m in model_dense.named_modules():
        if ".0." in name:
            if isinstance(m, torch.nn.Linear):
                print(name)
                linear = torch.nn.Linear(m.in_features, m.out_features).half().cuda().eval()
                linear.weight = torch.nn.Parameter(m.weight)
                if 'head' in name:
                    continue

                with torch.inference_mode():
                    if m.out_features == 4096 and m.in_features == 4096:
                        dense_output = linear(x)
                        dense_t = Timer(stmt="linear(x)",
                                        globals={"linear": linear,
                                                "x": x}).blocked_autorange().median * 1e3
                    elif m.in_features == 11008 and m.out_features == 4096:
                        dense_output = linear(y)
                        dense_t = Timer(stmt="linear(y)",
                                        globals={"linear": linear,
                                                 "y": y}).blocked_autorange().median * 1e3
                    elif m.in_features == 4096 and m.out_features == 11008:
                        dense_output = linear(z)
                        dense_t = Timer(stmt="linear(z)",
                                        globals={"linear": linear,
                                                 "z": z}).blocked_autorange().median * 1e3
                    print(dense_t)
                    dense_times[name] = dense_t
    print(dense_times)

    for name, m in model.named_modules():
        if ".0." in name:
            if isinstance(m, torch.nn.Linear):
                if 'head' in name:
                    continue
                linear = torch.nn.Linear(m.in_features, m.out_features).half().cuda().eval()
                linear.weight = torch.nn.Parameter(m.weight)

                    # accelerate via SparseSemiStructuredTensor
                linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))
                #print(linear.weight)

                with torch.inference_mode():
                    if m.out_features == 4096 and m.in_features == 4096:
                        sparse_output = linear(x)
                        sparse_t = Timer(stmt="linear(x)",
                                        globals=locals(),).blocked_autorange().median * 1e3
                    elif m.out_features == 4096 and m.in_features == 11008:
                        sparse_output = linear(y)
                        sparse_t = Timer(stmt="linear(y)",
                                         globals=locals(), ).blocked_autorange().median * 1e3
                    else:
                        sparse_output = linear(z)
                        sparse_t = Timer(stmt="linear(z)",
                                         globals=locals(), ).blocked_autorange().median * 1e3
                print(sparse_t)
                sparse_times[name] = sparse_t
    print(sparse_times)


final_speedups = {}

for elem in dense_times:
    print(elem)
    final_speedups[elem] = dense_times[elem]/sparse_times[elem]
    print(f"Dense: {dense_times[elem]:.3f}ms Sparse: {sparse_times[elem]:.3f}ms | Speedup: {(dense_times[elem] / sparse_times[elem]):.3f}x")

print(final_speedups)
