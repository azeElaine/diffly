import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image, ImageOps

from diffqrcoder import DiffQRCoderPipeline


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default="control_v1p_sd15_qrcode_monster"
    )
    parser.add_argument(
        "--pipe_ckpt",
        type=str,
        default="cetus-mix/cetusMix_Whalefall2_fp16.safetensors"
    )
    parser.add_argument(
        "--qrcode_path",
        type=str,
        default="qrcode/lywx.png"
    )
    parser.add_argument(
        "--qrcode_module_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--qrcode_padding",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Winter wonderland, fresh snowfall, evergreen trees, cozy log cabin, smoke rising from chimney, aurora borealis in night sky.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="easynegative"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "-srg",
        "--scanning_robust_guidance_scale",
        type=float,
        default=300,
    )
    parser.add_argument(
        "-pg",
        "--perceptual_guidance_scale",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--srmpgd_num_iteration",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--srmpgd_lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output"
    )
    # logo相关参数  
    parser.add_argument(  
        "--logo_path",  
        type=str,  
        default=None,  
        help="Path to logo image file"  
    )  
    parser.add_argument(  
        "--logo_guidance_scale",   
        type=int,  
        default=100,  
        help="Scale for logo guidance loss"  
    )
    parser.add_argument(
        "--logo_size_ratio",
        type=float,
        default=0.25,
        help="Logo size as a ratio of the smallest dimension (0.1-0.5)"
    )
    parser.add_argument(
        "--logo_position",
        type=str,
        default="center",
        choices=["center", "top_left", "top_right", "bottom_left", "bottom_right"],
        help="Position of the logo in the QR code"
    )
    parser.add_argument(
        "--edge_smoothness_weight",
        type=float,
        default=0.1,
        help="Weight for edge smoothness between logo and QR code (0.0-1.0)"
    )
    parser.add_argument(
        "--extract_logo",
        action="store_true",
        help="Extract logo from background (removes background)"
    )
    parser.add_argument(
        "--bg_threshold",
        type=int,
        default=240,
        help="Threshold for background removal (0-255), higher values preserve more of the logo"
    )
    return parser.parse_args()


def remove_background(image, threshold=240):
    """移除图像背景，保留logo"""
    # 转换为RGBA模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 获取图像数据
    data = np.array(image)
    
    # 创建alpha通道
    r, g, b, a = data.T
    
    # 简单阈值法移除白色/浅色背景
    white_areas = (r > threshold) & (g > threshold) & (b > threshold)
    data[..., 3][white_areas.T] = 0  # 设置透明
    
    # 创建新图像
    return Image.fromarray(data)


def resize_and_position_logo(logo_img, target_size, position, size_ratio):
    """调整logo大小并根据指定位置放置"""
    # 计算logo尺寸
    min_dim = min(target_size, target_size)
    logo_size = int(min_dim * size_ratio)
    
    # 调整logo尺寸
    logo_img = logo_img.resize((logo_size, logo_size), Image.LANCZOS)
    
    # 创建空白图像
    result = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    
    # 计算位置
    if position == "center":
        pos_x = (target_size - logo_size) // 2
        pos_y = (target_size - logo_size) // 2
    elif position == "top_left":
        pos_x = target_size // 8
        pos_y = target_size // 8
    elif position == "top_right":
        pos_x = target_size - logo_size - target_size // 8
        pos_y = target_size // 8
    elif position == "bottom_left":
        pos_x = target_size // 8
        pos_y = target_size - logo_size - target_size // 8
    elif position == "bottom_right":
        pos_x = target_size - logo_size - target_size // 8
        pos_y = target_size - logo_size - target_size // 8
    
    # 贴上logo
    result.paste(logo_img, (pos_x, pos_y), logo_img if logo_img.mode == 'RGBA' else None)
    
    return result


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    qrcode = load_image(args.qrcode_path)
    logo_img = None
    
    if args.logo_path:  
        # 加载logo图像
        original_logo = load_image(args.logo_path)
        
        # 如果需要提取logo（移除背景）
        if args.extract_logo:
            print("提取logo，移除背景...")
            original_logo = remove_background(original_logo, args.bg_threshold)
        
        # 调整logo大小和位置
        target_size = 512  # 标准输出尺寸
        logo_img = resize_and_position_logo(
            original_logo, 
            target_size, 
            args.logo_position, 
            args.logo_size_ratio
        )
        
        # 保存处理后的logo用于检查
        logo_path = Path(args.output_folder, "processed_logo.png")
        logo_img.save(logo_path)
        print(f"已保存处理后的logo到: {logo_path}")
        
        # 转换回PIL图像格式用于diffuser
        if logo_img.mode == "RGBA":
            # 创建白色背景
            background = Image.new("RGB", logo_img.size, (255, 255, 255))
            # 合成图像
            background.paste(logo_img, mask=logo_img.split()[3])  # 使用alpha通道作为mask
            logo_img = background
    
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_ckpt,
        torch_dtype=torch.float16,
        local_files_only=True 
    )
    pipe = DiffQRCoderPipeline.from_single_file(
        args.pipe_ckpt,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=True 
    )
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # Memory optimizations  
    try:  
        pipe.enable_attention_slicing()  
        pipe.enable_xformers_memory_efficient_attention()  
    except AttributeError:  
    # Fallback if xformers is not available  
        pipe.enable_attention_slicing()  
      
    try:  
        pipe.enable_sequential_cpu_offload()  
    except AttributeError:  
        pass  
  
    # Enable gradient checkpointing if available  
    if hasattr(pipe.unet, 'enable_gradient_checkpointing'):  
        pipe.unet.enable_gradient_checkpointing()

    # 设置EdgeSmoothness权重（通过环境变量传递给模型）
    if args.edge_smoothness_weight is not None:
        os.environ["LOGO_EDGE_SMOOTHNESS_WEIGHT"] = str(args.edge_smoothness_weight)

    print("开始生成带logo的二维码...")
    # 增加扫描鲁棒性引导比例，确保二维码部分可扫描
    scanning_robust_scale = args.scanning_robust_guidance_scale
    if args.logo_path:
        # 当有logo时，增加扫描鲁棒性权重以确保非logo区域可扫描
        scanning_robust_scale *= 1.2

    result = pipe(
        prompt=args.prompt,
        qrcode=qrcode,
        logo_image=logo_img,
        logo_guidance_scale=args.logo_guidance_scale,
        qrcode_module_size=args.qrcode_module_size,
        qrcode_padding=args.qrcode_padding,
        negative_prompt=args.neg_prompt,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=args.device).manual_seed(1),
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        scanning_robust_guidance_scale=scanning_robust_scale,
        perceptual_guidance_scale=args.perceptual_guidance_scale,
        srmpgd_num_iteration=args.srmpgd_num_iteration,
        srmpgd_lr=args.srmpgd_lr,
    )
    
    # 保存结果
    suffix = ""
    if args.logo_path:
        suffix = f"_with_logo_{args.logo_position}_{args.logo_size_ratio}"
        if args.extract_logo:
            suffix += "_extracted"
    
    output_filename = f"qrcode{suffix}.png"
    output_path = Path(args.output_folder, output_filename)
    result.images[0].save(output_path)
    print(f"已保存结果到: {output_path}")
