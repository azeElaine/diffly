import os
import subprocess
from huggingface_hub import snapshot_download

print("设置 Hugging Face 镜像站点...")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("下载 ControlNet 模型...")
controlnet_dir = "control_v1p_sd15_qrcode_monster"
if not os.path.exists(controlnet_dir):
    os.makedirs(controlnet_dir, exist_ok=True)
    snapshot_download(
        repo_id="monster-labs/control_v1p_sd15_qrcode_monster",
        local_dir=controlnet_dir,
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"]
    )

print("下载 Cetus-Mix 模型...")
cetus_dir = "cetus-mix"
if not os.path.exists(cetus_dir):
    os.makedirs(cetus_dir, exist_ok=True)
    snapshot_download(
        repo_id="fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned",
        local_dir=cetus_dir,
        allow_patterns=["cetusMix_Whalefall2_fp16.safetensors", "model_index.json"]
    )

print("模型下载完成！") 