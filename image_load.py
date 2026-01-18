from datasets import load_dataset
from huggingface_hub import hf_hub_download
import os

repo_id = "imageomics/sentinel-beetles"  # Replace with your repo ID
local_dir = "BeetleImages"
os.makedirs(local_dir, exist_ok=True)

# 1. Stream the dataset to get the file paths
ds = load_dataset(repo_id, streaming=True, split="train")

print("Starting downloads...")

for row in ds:
    file_path = row["file_path"] # The column from your screenshot
    
    # 2. Download each specific file
    hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        local_dir=local_dir
    )
    print(f"Downloaded: {file_path}")