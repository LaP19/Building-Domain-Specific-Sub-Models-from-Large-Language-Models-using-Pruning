#! /usr/bin/python3

# HuggingFace Model Downloader Script
# Author: Dr Musashi Jacobs-Harukawa, DDSS


"""
Convenience command-line script to download model assets to cache for offline usage on compute nodes.
Usage:
`python hf_model_downloader.py --repo_id='HF_MODEL_REF' --revision='main' --cache_dir=''`
"""

import argparse
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

def download_save_huggingface_model(repo_id: str, revision: str, cache_dir: str):
    if cache_dir=='':
        cache_dir = os.environ.get("HF_HOME")
    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, resume_download=True)

def download_save_huggingface_dataset(dataset_id: str, subset: str, cache_dir: str, split: str):
    if cache_dir == '':
        cache_dir = os.environ.get("HG_DATASETS_CACHE")

    # Use snapshot_download for dataset
    dataset = load_dataset(dataset_id, subset, split=split, cache_dir=cache_dir)
    return dataset


def main():
    # Read in arguments
    parser = argparse.ArgumentParser()

    # Common arguments for model and dataset downloaders
    parser.add_argument('--repo_id', type=str, help="HF Model Hub Repo ID or Dataset ID")
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--cache_dir', type=str, default='', help='Location to save assets, defaults to None')

    # Additional arguments for dataset downloader
    parser.add_argument('--data_files', type=str, default='', help='JSON dictionary specifying data files for dataset')
    parser.add_argument('--split', type=str, default='', help='Dataset split to load')
    parser.add_argument('--subset', type=str, default='', help='Subset')

    args = parser.parse_args()

    # Decide whether to download a model or a dataset based on the provided arguments
    if args.split != "":
        # Download and load dataset
        dataset = download_save_huggingface_dataset(args.repo_id, args.subset, args.cache_dir, args.split)
        print(dataset)
    else:
        # Download model
        download_save_huggingface_model(args.repo_id, args.revision, args.cache_dir)


if __name__ == '__main__':
    main()

