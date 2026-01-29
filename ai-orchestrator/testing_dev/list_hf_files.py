
from huggingface_hub import HfApi

repo_id = "wuhp/dog-yolo"
api = HfApi()

try:
    files = api.list_repo_files(repo_id)
    print(f"Files in {repo_id}:")
    for f in files:
        print(f" - {f}")
except Exception as e:
    print(f"Error listing files: {e}")
