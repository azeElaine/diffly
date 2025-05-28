import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """检查文件是否存在，并打印状态信息"""
    path = Path(filepath)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ {description} 已找到: {path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ {description} 未找到: {path}")
        return False

def main():
    """检查所需的模型文件"""
    print("检查模型文件...")
    
    # ControlNet 模型文件
    controlnet_dir = "control_v1p_sd15_qrcode_monster"
    controlnet_config = os.path.join(controlnet_dir, "config.json")
    controlnet_model = os.path.join(controlnet_dir, "diffusion_pytorch_model.safetensors")
    
    # Cetus-Mix 模型文件
    cetus_dir = "cetus-mix"
    cetus_model = os.path.join(cetus_dir, "cetusMix_Whalefall2_fp16.safetensors")
    cetus_config = os.path.join(cetus_dir, "model_index.json")
    
    # 检查文件是否存在
    files_ok = True
    files_ok &= check_file(controlnet_config, "ControlNet配置文件")
    files_ok &= check_file(controlnet_model, "ControlNet模型文件")
    files_ok &= check_file(cetus_model, "Cetus-Mix模型文件")
    files_ok &= check_file(cetus_config, "Cetus-Mix配置文件")
    
    # 输出结果
    if files_ok:
        print("\n✓ 所有模型文件已找到！")
    else:
        print("\n✗ 缺少一些模型文件。请下载所需的模型文件。")
        print("\n下载说明:")
        print("1. 从 Hugging Face 下载 ControlNet 模型:")
        print("   - 模型: monster-labs/control_v1p_sd15_qrcode_monster")
        print("   - 文件: config.json, diffusion_pytorch_model.safetensors")
        print("   - 保存到: ./control_v1p_sd15_qrcode_monster/")
        print("\n2. 从 Hugging Face 下载 Cetus-Mix 模型:")
        print("   - 模型: fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned")
        print("   - 文件: cetusMix_Whalefall2_fp16.safetensors, model_index.json")
        print("   - 保存到: ./cetus-mix/")
    
    return 0 if files_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 