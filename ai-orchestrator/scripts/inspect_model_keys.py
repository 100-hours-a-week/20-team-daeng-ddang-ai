
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sys

SEARCH_MODEL_ID = "20-team-daeng-ddang-ai/dog-emotion-classification"

def inspect_keys():
    print(f"Downloading/Loading {SEARCH_MODEL_ID}...")
    try:
        # Try safetensors first
        path = hf_hub_download(repo_id=SEARCH_MODEL_ID, filename="model.safetensors")
        state_dict = load_file(path)
    except Exception as e:
        print(f"Safetensors not found or failed: {e}")
        try:
            path = hf_hub_download(repo_id=SEARCH_MODEL_ID, filename="pytorch_model.bin")
            state_dict = torch.load(path, map_location="cpu")
        except Exception as e2:
            print(f"Pytorch bin not found or failed: {e2}")
            return

    print(f"\nLoaded state dict from {path}")
    print(f"Total keys: {len(state_dict)}")
    print("Sample keys (all):")
    for i, key in enumerate(list(state_dict.keys())):
        if "classifier" in key:
            print(f"  {key}")

if __name__ == "__main__":
    inspect_keys()