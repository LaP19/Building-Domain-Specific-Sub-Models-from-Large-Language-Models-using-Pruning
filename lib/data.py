#!/usr/bin/env python3
# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from typing import Optional

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_data_dir(
        hf_cache_dir: Optional[os.PathLike]=os.environ.get("HG_DATASETS_CACHE", None),
        ) -> Path:
    """
    Convenience function for retrieving locally stored HF weights.
    """
    if hf_cache_dir is None:
        hf_cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    if not isinstance(hf_cache_dir, Path):
        hf_cache_dir = Path(hf_cache_dir)
    data_dir = hf_cache_dir / "allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
    return data_dir

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets from cache
    dataset1 = Dataset.from_file("/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train-00000-of-00002.arrow")
    dataset2 = Dataset.from_file("/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train-00001-of-00002.arrow")
    traindata = concatenate_datasets([dataset1, dataset2])
    print("trandata:")
    print(traindata)
    #valdata = Dataset.from_file("/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/allenai___c4/default-b1a492c8b4369786/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train.arrow")
    #print("valdata:")
    #print(valdata)
    # Concatenate the shards into a single dataset
    print("ARRIVED")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(nsamples):
        print(count)
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        count+=1
        inp = trainenc.input_ids[:, i:j]
        attn_mask = trainenc.attention_mask
        trainloader.append((inp, attn_mask))
        print(_)

    print("Finished getting the train samples")
    print(trainloader)

    return trainloader, _

def get_human_eval(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = load_dataset("openai_humaneval", split = "test")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['prompt'][i] + dataset['canonical_solution'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_math_qa(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/meta-math___meta_math_qa/default/0.0.0/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18/meta_math_qa-train.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['original_question'][i] + "\n" + dataset['response'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_math_instruct(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/TIGER-Lab___math_instruct/default/0.0.0/360fd8fe1b090553caf0e6271b8e32fc093a5d83/math_instruct-train.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['instruction'][i] + "\n" + dataset['output'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_math_orca(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/microsoft___orca-math-word-problems-200k/default/0.0.0/29255d1770cc4eac66e5e7fa378cba542c026350/orca-math-word-problems-200k-train.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['question'][i] + "\n" + dataset['answer'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_java(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/semeru___code-text-java/default/0.0.0/f73aa95d9ea6bfd39fd016e493a5591b5050acff/code-text-java-train.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['code'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_cpp(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/nguyentruong-ins___nhlcoding_cleaned_cpp_dataset/default/0.0.0/037bf868128470e210dba3c4984faa8a055acff9/nhlcoding_cleaned_cpp_dataset-train-00000-of-00004.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['solution'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_javascript(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/semeru___code-text-javascript/default/0.0.0/e63b00e6cedd664bb28f057f9518cd26604b105f/code-text-javascript-train.arrow")
    print(dataset)
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        sample = dataset['code'][i]
        print(sample)
        trainenc = tokenizer(sample, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _
def get_apps(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/codeparrot___apps/all/0.0.0/21e74ddf8de1a21436da12e3e653065c5213e9d1/apps-test-00000-of-00002.arrow")

    print(len(dataset))
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(dataset['solutions'][0])
    count = 0
    for _ in range(min(nsamples, len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        trainenc = tokenizer(dataset['solutions'][i], return_tensors='pt')
        while(trainenc.input_ids.shape[1] > 10000):
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset['solutions'][i], return_tensors='pt')
        #trainenc = tokenizer(dataset['solutions'][i], return_tensors='pt', padding='max_length', max_length=seqlen, truncation = True)
        count += 1
        inp = trainenc.input_ids
        print(inp)
        print(inp.shape)
        attn_mask = trainenc.attention_mask
        print(attn_mask.shape)
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _
# Load and process python_code_instructions_18k_alpaca dataset
def get_python_alpaca(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/iamtarun___python_code_instructions_18k_alpaca/default/0.0.0/7cae181e29701a8663a07a3ea43c8e105b663ba1/python_code_instructions_18k_alpaca-train.arrow")

    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(dataset['instruction'][0])
    for _ in range(min(nsamples,len(dataset))):
        print(_)
        i = random.randint(0, len(dataset) - 1)
        prompt = dataset['prompt'][i]
        trainenc = tokenizer(prompt, return_tensors='pt')
        print("LENGHT IS: ", trainenc.input_ids.shape[1])
        #trainenc = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=seqlen)
        inp = trainenc.input_ids
        attn_mask = trainenc.attention_mask
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_github_python(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset1 = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/thomwolf___github-python/default/0.0.0/e9120c87cf9ab145f6fa35529b94ad5e39f27564/github-python-train-00000-of-00005.arrow")
    dataset2 = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/thomwolf___github-python/default/0.0.0/e9120c87cf9ab145f6fa35529b94ad5e39f27564/github-python-train-00001-of-00005.arrow")
    dataset_temp = concatenate_datasets([dataset1, dataset2])
    dataset3 = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/thomwolf___github-python/default/0.0.0/e9120c87cf9ab145f6fa35529b94ad5e39f27564/github-python-train-00002-of-00005.arrow")
    dataset = concatenate_datasets([dataset_temp, dataset3])

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count=0
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    for _ in range(min(nsamples,len(dataset))):
        print(count)
        i = random.randint(0, len(dataset) - 1)
        print(dataset[i]['content'])
        trainenc = tokenizer(dataset[i]['content'], return_tensors='pt')
        while (trainenc.input_ids.shape[1] > 10000 or trainenc.input_ids.shape[1] == 1):
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['content'], return_tensors='pt')
        #trainenc = tokenizer(dataset[i]['content'], return_tensors='pt', padding='max_length', max_length=seqlen, truncation=True)
        count += 1
        inp = trainenc.input_ids
        print(inp)
        print(inp.shape)
        attn_mask = trainenc.attention_mask
        print(attn_mask.shape)
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_openbookqa(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/openbookqa/main/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-train.arrow")

    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(dataset['question_stem'][0])
    data = dataset['choices'][0]
    formatted_texts = [f"{label}: {text}" for label, text in zip(data["label"], data["text"])]
    result = ", ".join(formatted_texts)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    count = 0
    print (result)
    for _ in range(min(nsamples, len(dataset))):
        i = random.randint(0, len(dataset) - 1)
        print(count)
        data = dataset['choices'][i]
        formatted_texts = [f"{label}: {text}" for label, text in zip(data["label"], data["text"])]
        result = ", ".join(formatted_texts)
        prompt = "[INST]\nChoose the right option: " + dataset['question_stem'][i] + '#Choices: ' + result + "\n[/INST]\n"
        trainenc = tokenizer(prompt, return_tensors='pt')
        count+=1
        inp = trainenc.input_ids
        # print(inp)
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_hellaswag(nsamples, seed, seqlen, tokenizer):
    # Load train dataset
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/Rowan___hellaswag/default/0.1.0/6002345709e0801764318f06bf06ce1e7d1a1fe3/hellaswag-train.arrow")

    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(dataset['ctx'][0])
    formatted_endings = [f"{i + 1}: {ending}" for i, ending in enumerate(dataset['endings'][0])]
    result = " ".join(formatted_endings)
    print(result)

    for _ in range(min(nsamples, len(dataset))):
        count = 0
        i = random.randint(0, len(dataset) - 1)
        print(count)
        formatted_endings = [f"{j + 1}: {ending}" for j, ending in enumerate(dataset['endings'][i])]
        result = " ".join(formatted_endings)
        prompt = "[INST]\nChoose the most appropriate ending: " + dataset['ctx'][i] + '###Options: ' + result + "\n[/INST]n"
        print(prompt)
        trainenc = tokenizer(prompt, return_tensors='pt')
        print(trainenc.input_ids.shape[1])
        count+=1
        inp = trainenc.input_ids
        # print(inp)
        attn_mask = trainenc.attention_mask
        # print(attn_mask)
        trainloader.append((inp, attn_mask))

    return trainloader, _

def get_commonsenseqa(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/tau___commonsense_qa/default/0.0.0/94630fe30dad47192a8546eb75f094926d47e155/commonsense_qa-train.arrow")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    print(dataset['question'][0])
    for _ in range(min(nsamples,len(dataset))):
        print(_)
        i = random.randint(0, len(dataset) - 1)
        data = dataset['choices'][i]
        formatted_texts = [f"{label}: {text}" for label, text in zip(data["label"], data["text"])]
        result = ", ".join(formatted_texts)
        prompt = '[INST]\nChoose the right option related to the following sentence. ' + dataset['question'][i] + ' ' + result + "\n[/INST]\n"
        print(prompt)
        trainenc = tokenizer(prompt, return_tensors='pt')
        inp = trainenc.input_ids
        attn_mask = trainenc.attention_mask
        trainloader.append((inp, attn_mask))

    return trainloader, _


def get_wmt14(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/wmt14/fr-en/1.0.0/d5cfc45c32d826941d8678bf74c810c2aaa057cdc5544f1e23a5dab8c0407a9f.incomplete/wmt14-train-00000-00002-of-NNNNN.arrow")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    translation = dataset['translation'][0]
    prompt = "#Text: " + translation["en"] + " #Translation: " + translation["fr"]
    print(prompt)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"
    count = 0
    max = 0
    for _ in range(min(nsamples,len(dataset))):
        i = random.randint(0, len(dataset) - 1)
        prompt = dataset['translation'][i]
        translation = "#Text: " + prompt["en"] + " #Translation: " + prompt["fr"]
        trainenc = tokenizer(translation, return_tensors='pt')
        leng = trainenc.input_ids.shape[1]
        if leng > max:
            max = leng
        print("LENGTH IS: ", leng)
        print(count)
        trainenc = tokenizer(translation, return_tensors='pt', padding='max_length', max_length=seqlen)
        count +=1
        inp = trainenc.input_ids
        attn_mask = trainenc.attention_mask
        print(attn_mask)
        trainloader.append((inp, attn_mask))
    print(max)

    return trainloader, _

def get_song_translation(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/Nicolas-BZRD___english_french_songs_lyrics_translation_original/default/0.0.0/f2b64a358132305095e602ce11819d834344d90f/english_french_songs_lyrics_translation_original-train.arrow")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    translation = dataset['original_version'][0] + " " + dataset['french_version'][0]
    print(translation)
    count = 0
    for _ in range(min(nsamples,len(dataset))):
        print(count)
        while True:
            i = random.randint(0, len(dataset) - 1)
            print(type(dataset['original_version'][i]))
            print(type(dataset['french_version'][i]))
            if isinstance(dataset['original_version'][i], str) and isinstance(dataset['french_version'][i], str):
                translation = "#Text: " + dataset['original_version'][i] + " #Translation: " + dataset['french_version'][i]
                trainenc = tokenizer(translation, return_tensors='pt')
                count +=1
                inp = trainenc.input_ids
                attn_mask = trainenc.attention_mask
                print(attn_mask)
                trainloader.append((inp, attn_mask))
                break

    return trainloader, _

def get_opus_books(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    dataset = Dataset.from_file(
        "/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/datasets/Helsinki-NLP___opus_books/en-fr/0.0.0/1f9f6191d0e91a3c539c2595e2fe48fc1420de9b/opus_books-train.arrow")
    dataset = dataset['translation']
    #I want to use half of the dataset for pruning and half for evaluation
    dataset = dataset[:int(len(dataset)/2)]
    print(len(dataset))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    count = 0
    for _ in range(min(nsamples,len(dataset))):
        print(count)
        while True:
            i = random.randint(0, len(dataset) - 1)
            prompt = dataset[i]
            if isinstance(prompt["en"], str) and isinstance(prompt["fr"], str):
                translation = "#Text: " + prompt["en"] + " #Translation: " + prompt["fr"]
                print(translation)
                trainenc = tokenizer(translation, return_tensors='pt')
                count +=1
                inp = trainenc.input_ids
                attn_mask = trainenc.attention_mask
                print(attn_mask)
                trainloader.append((inp, attn_mask))
                break

    return trainloader, _


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    match name:
        case 'wikitext2':
            return get_wikitext2(nsamples, seed, seqlen, tokenizer)
        case "c4":
            return get_c4(nsamples, seed, seqlen, tokenizer)

        #CODE GENERATION
        case "humaneval": #humaneval instruction with padding right
            return get_human_eval(nsamples, seed, seqlen, tokenizer)
        case "python_alpaca": #python_alpaca instruction with padding right
            return get_python_alpaca(nsamples, seed, seqlen, tokenizer)
        case "apps": #apps code without padding (taking code snippets that are longer than 4096 tokens)
            return get_apps(nsamples, seed, seqlen, tokenizer)
        case "github_python":  # code snippets from github, without instruction
            return get_github_python(nsamples, seed, seqlen, tokenizer)
        case "java":
            return get_java(nsamples, seed, seqlen, tokenizer)
        case "javascript":
            return get_javascript(nsamples, seed, seqlen, tokenizer)
        case "cpp":
            return get_cpp(nsamples, seed, seqlen, tokenizer)
        #COMMONSENSE REASONING
        case "openbookqa":
            return get_openbookqa(nsamples, seed, seqlen, tokenizer)
        case "hellaswag":
            return get_hellaswag(nsamples, seed, seqlen, tokenizer)
        case "commonsenseqa":
            return get_commonsenseqa(nsamples, seed, seqlen, tokenizer)
        case "wmt14":
            return get_wmt14(nsamples, seed, seqlen, tokenizer)
        case "song_translation" :
            return get_song_translation(nsamples, seed, seqlen, tokenizer)
        case "opus_books":
            return get_opus_books(nsamples, seed, seqlen, tokenizer)

        case "math_qa":
            return get_math_qa(nsamples, seed, seqlen, tokenizer)
        case "math_instruct":
            return get_math_instruct(nsamples, seed, seqlen, tokenizer)
        case "math_orca":
            return get_math_orca(nsamples, seed, seqlen, tokenizer)




