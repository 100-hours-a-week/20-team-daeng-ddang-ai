
from huggingface_hub import hf_hub_download
import json

MODEL_ID = "20-team-daeng-ddang-ai/dog-emotion-classification"

def inspect_config():
    try:
        print(f"Downloading config.json for {MODEL_ID}...")
        config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        print("Config content:")
        # Print keys regarding labels
        if 'id2label' in config:
            print(f"id2label: {config['id2label']}")
        else:
            print("id2label NOT found in config.json")
            
        if 'label2id' in config:
            print(f"label2id: {config['label2id']}")
            
        if 'architectures' in config:
            print(f"architectures: {config['architectures']}")

    except Exception as e:
        print(f"Failed to inspect config: {e}")

if __name__ == "__main__":
    inspect_config()
