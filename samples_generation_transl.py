#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random
from typing import Optional
import numpy as np

import torch
from datasets import Dataset
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from nltk import word_tokenize
from nltk.corpus.reader import WordNetCorpusReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import numpy as np
from nltk.translate import meteor_score
from packaging import version
import nltk
import ssl

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

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
    model_path = 'Gemma-Instruct-7b-pruned_' + dataset
    model_weights_dir = hf_cache_dir / f"{model_path}"
    return model_weights_dir


def generate_completions(model, tokenizer, device, prompt):
    tokenizer.pad_token = tokenizer.eos_token
    # Tokenize all prompts in a batch
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate
    generate_ids = model.generate(**input, max_length=1500, temperature=0.95, top_p=0.95, top_k=250, do_sample=True)

    # Decode generated completions
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    prompt = prompt.replace("<s>", "")
    output = output.replace(prompt, "")

    return output


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


def main():
    # Ensure that your GPU is available and CUDA is properly installed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", type=str, help="model")
    parser.add_argument("--dataset", default="", type=str, help="dataset")
    parser.add_argument("--eval_dataset", default="", type=str, help="dataset for evaluation")
    args = parser.parse_args()

    if args.model != "":
        print(f"loading model {args.model}")
        weights_dir = get_weight_dir(args.model)
        print("Dir: ", weights_dir)
        model = AutoModelForCausalLM.from_pretrained(weights_dir).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"loading model pruned with dataset {args.dataset}")
        weights_dir = get_weight_dir_pruned(args.dataset)
        print("Dir: ", weights_dir)
        model = AutoModelForCausalLM.from_pretrained(weights_dir).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset")

    if "wmt" in args.eval_dataset:
        dataset = Dataset.from_file(
            "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/wmt14/fr-en/1.0.0/d5cfc45c32d826941d8678bf74c810c2aaa057cdc5544f1e23a5dab8c0407a9f.incomplete/wmt14-train-00000-00003-of-NNNNN.arrow")
        dataset = dataset['translation']
    elif "opus_books" in args.eval_dataset:
        dataset = Dataset.from_file(
            "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/Helsinki-NLP___opus_books/en-fr/0.0.0/1f9f6191d0e91a3c539c2595e2fe48fc1420de9b/opus_books-train.arrow")
        dataset = dataset['translation']
        # I want to use half of the dataset for pruning and half for evaluation
        dataset = dataset[int(len(dataset) / 2):]

    # Instantiate the OpenMultilingualWordnet class
    wordnet = WordNetCorpusReader("/leonardo/home/userexternal/lpuccion/nltk_data/corpora/wordnet", None)
    print(len(dataset))
    num_samples_per_task = 5
    sum = 0

    for j in range(400):
        print(j+1)
        i = random.randint(0, len(dataset) - 1)
        translation = dataset[i]
        print(translation["en"])
        print(translation["fr"])
        en_sentence = translation["en"]
        system = "WRITE ONLY THE FRENCH SENTENCE IN OUTPUT"
        user = f"Please translate the following sentence from english to french: \"{en_sentence}\""
        #print(user)
        references = {}
        #prompt = f"<start_of_turn>user\n{system}\n{user}\n<end_of_turn>\n<start_of_turn>model\n"
        prompt = f"[INST]\n{system}\n{user}\n[/INST]\n"
        for k in range(num_samples_per_task):
            output = generate_completions(model, tokenizer, device, prompt)
            #idx = output.find("model")
            #output = output[idx + len("model"):]
            references[k] = word_tokenize(output)

        print(references)
        print(translation["fr"])
        verify = translation["fr"]
        score = round(meteor_score.meteor_score(references.values(), word_tokenize(verify), wordnet=wordnet,alpha = 0.9, beta = 3, gamma = 0.5),4)
        print(score)
        sum += score
        if j+1 == 0:
            print("The average score is: ", 0)
        else:
            print("The average score is: ", sum/(j+1))


    print("Batch generation finished.")


if __name__ == '__main__':
    main()
