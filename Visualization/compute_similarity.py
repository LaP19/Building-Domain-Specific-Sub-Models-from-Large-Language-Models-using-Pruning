import argparse
import pickle
import torch
import io
import matplotlib.pyplot as plt
import numpy as np
import re

# Metrics definitions
def jaccard_similarity(tensor1, tensor2):
    intersection = (tensor1 & tensor2).float().sum()
    union = (tensor1 | tensor2).float().sum()
    return (intersection / union).item()

# Function to sort layer data based on layer number
def sort_layer_data(results_dict):
    for metric, data in results_dict.items():
        results_dict[metric]['layer'] = sorted(data['layer'], key=lambda x: int(re.search(r"Layer (\d+)", x['name']).group(1)))

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model')
parser.add_argument('--dataset1', type=str, help='First dataset')
parser.add_argument('--dataset2', type=str, help='Second dataset')
args = parser.parse_args()

# Load data from both files
mask_path1 = f"/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/masks/{args.model}_wanda_mask_{args.dataset1}.pkl"
mask_path2 = f"/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/masks/{args.model}_wanda_mask_{args.dataset2}.pkl"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open(mask_path1, "rb") as f:
    data1 = CPU_Unpickler(f).load()

with open(mask_path2, "rb") as f:
    data2 = CPU_Unpickler(f).load()

print(type(data1))
print(type(data2))

if isinstance(data1, bytes):
    data1 = torch.load(io.BytesIO(data1), map_location=torch.device('cpu'))

if isinstance(data2, bytes):
    data2 = torch.load(io.BytesIO(data2), map_location=torch.device('cpu'))

# Extracting mask names for saved file names
mask_name1 = mask_path1.split('/')[-1].split('.')[0]
mask_name2 = mask_path2.split('/')[-1].split('.')[0]
combined_mask_name = f"{mask_name1}_vs_{mask_name2}"

# Metrics accumulators
results = {
    'jaccard': {'layer': [], 'sublayer': [], 'whole': None},
}

whole_data1 = []
whole_data2 = []

print("here")

count = 0

for layer, sublayers in data1.items():
    layer_data1 = []
    layer_data2 = []

    print(count)
    count += 1

    for sublayer_name, mask1 in sublayers.items():
        mask2 = data2[layer][sublayer_name]

        # Move masks to CPU
        mask1 = mask1.to('cpu')
        mask2 = mask2.to('cpu')

        # Sublayer metrics
        results['jaccard']['sublayer'].append({
            'name': f"Layer {layer} Sublayer {sublayer_name} (Shape: {mask1.shape})",
            'value': 1 - jaccard_similarity(mask1, mask2)
        })

        # Accumulate for layer
        layer_data1.append(mask1.view(-1))
        layer_data2.append(mask2.view(-1))

    # Combine sublayer tensors for layer
    layer_data1 = torch.cat(layer_data1)
    layer_data2 = torch.cat(layer_data2)

    # Layer metrics
    results['jaccard']['layer'].append({
        'name': f"Layer {layer}",
        'value': 1 - jaccard_similarity(layer_data1, layer_data2)
    })

    # Accumulate for whole masks
    whole_data1.append(layer_data1)
    whole_data2.append(layer_data2)

# Combine layer tensors for whole masks
whole_data1 = torch.cat(whole_data1)
whole_data2 = torch.cat(whole_data2)

# Whole masks metrics
results['jaccard']['whole'] = 1 - jaccard_similarity(whole_data1, whole_data2)

print(results)

# Apply the sorting function to your results
sort_layer_data(results)


# Visualization and Saving
for metric, data in results.items():
    # Sort sublayer data based on the metric values
    sorted_sublayer_data = sorted(data['sublayer'], key=lambda x: x['value'], reverse=False)

    # 1. A figure comparing the metrics for different layers
    plt.figure(figsize=(10, 6))
    layer_values = [layer['value'] for layer in data['layer']]
    #layer_names = [layer['name'] for layer in data['layer']]
    layer_names = []
    if "13" in args.model:
        for i in range(40):
            layer_names.append(i)
    elif "CodeLlama-Instruct-7b" in args.model or "Mistral" in args.model:
        for i in range(32):
            layer_names.append(i)
    elif "Gemma" in args.model:
        for i in range(28):
            layer_names.append(i)
    plt.bar(layer_names, layer_values, edgecolor='k', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.title(f'Jaccard Distance Distribution - Layer', fontsize=20)
    plt.xlabel('Layer Number', fontsize=20)
    plt.ylabel(f'Jaccard Distance', fontsize=20)
    plt.tight_layout()
    if metric in ['jaccard']:
        plt.ylim(0, 0.3)
    plt.savefig(f'/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/distance_graphs/{combined_mask_name}_{metric}_layers_comparison.png')
    plt.close()

    """
    # 3. A figure per layer comparing sublayers
    # First, group the sublayers by layer
    sublayer_groups = {}
    for sublayer in sorted_sublayer_data:
        layer_match = re.search(r'Layer (\d+)', sublayer['name'])
        layer_number = layer_match.group(1) if layer_match else 'unknown'

        if layer_number not in sublayer_groups:
            sublayer_groups[layer_number] = []
        sublayer_groups[layer_number].append(sublayer)

    for layer_number, sublayers in sublayer_groups.items():
        plt.figure(figsize=(12, 7))
        for sublayer in sublayers:
            plt.bar(sublayer['name'], sublayer['value'], edgecolor='k', alpha=0.7, label=sublayer['name'])
        plt.title(f'{metric.capitalize()} Similarity Distribution - Layer {layer_number} Sublayers')
        plt.xlabel('Sublayer')
        plt.ylabel(f'{metric.capitalize()} Similarity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if metric in ['jaccard', 'hamming']:
            plt.ylim(0, 1)
        elif metric == 'cosine':
            plt.ylim(-1, 1)
        plt.savefig(f"/leonardo_scratch/large/userexternal/lpuccion/.cache/huggingface/similarity_graphs/{args.model}_{combined_mask_name}_layer_{layer_number}_{metric}_sublayer_comparison.png")
        plt.close()
    """
