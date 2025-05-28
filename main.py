import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image, ImageOps, ImageEnhance

from diffqrcoder import DiffQRCoderPipeline


def check_cuda_and_memory():
    """检查CUDA是否可用以及GPU内存情况"""
    print("\n===== CUDA 和 GPU 内存检查 =====")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"当前设备索引: {torch.cuda.current_device()}")
        print(f"设备数量: {torch.cuda.device_count()}")
        
        # 显示内存信息
        print("\n----- GPU 内存信息 -----")
        print(f"分配的内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存的内存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"最大内存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        # 尝试清理内存
        torch.cuda.empty_cache()
        print("\n已清理 CUDA 缓存")
    print("==============================\n")


def custom_postprocess_image(tensor_image):
    """自定义图像后处理函数，确保图像不会变黑或变灰，同时保持二维码的可扫描性"""
    # 确保张量在CPU上并分离梯度
    tensor_image = tensor_image.cpu().detach()
    
    # 打印原始值范围
    print(f"后处理前图像值范围: min={tensor_image.min().item() if not torch.isnan(tensor_image.min()) else 'nan'}, max={tensor_image.max().item() if not torch.isnan(tensor_image.max()) else 'nan'}")
    
    # 检查并修复NaN值
    if torch.isnan(tensor_image).any():
        print("检测到NaN值，尝试修复...")
        # 将NaN值替换为0
        tensor_image = torch.nan_to_num(tensor_image, nan=0.0)
    
    # 如果图像值范围异常，尝试修复
    if tensor_image.min() < -1.0 or tensor_image.max() > 1.0:
        print("检测到异常值范围，尝试修复...")
        tensor_image = torch.clamp(tensor_image, -1.0, 1.0)
    
    # 转换到0-1范围
    if tensor_image.min() < 0:
        # 如果有负值，假设范围是[-1, 1]
        tensor_image = (tensor_image + 1.0) / 2.0
    
    # 确保值在[0, 1]范围内
    tensor_image = torch.clamp(tensor_image, 0.0, 1.0)
    
    # 检查图像是否是灰色（所有通道值接近）
    if tensor_image.shape[0] == 3:
        channel_mean = tensor_image.mean(dim=(1, 2))
        channel_std = torch.std(channel_mean)
        
        if channel_std < 0.02:  # 如果通道间标准差很小，说明是灰色图像
            print("检测到图像接近灰色，轻微增加彩色对比度...")
            # 增加对比度，但保持适度
            tensor_image = (tensor_image - 0.5) * 1.2 + 0.5
            tensor_image = torch.clamp(tensor_image, 0.0, 1.0)
            
            # 为不同通道添加轻微的色调变化
            tensor_image[0] = tensor_image[0] * 1.05  # 轻微增加红色
            tensor_image[2] = tensor_image[2] * 1.05  # 轻微增加蓝色
            tensor_image = torch.clamp(tensor_image, 0.0, 1.0)
    
    # 如果图像全黑或接近全黑/全灰，生成彩色噪声图像
    if tensor_image.mean() < 0.2 or (tensor_image.max() - tensor_image.min()) < 0.1:
        print("检测到图像接近全黑或全灰，生成彩色噪声图像...")
        # 生成彩色随机噪声
        noise = torch.rand_like(tensor_image) * 0.5 + 0.25  # 值在[0.25, 0.75]范围内
        
        # 为噪声添加一些结构
        if tensor_image.shape[0] == 3:
            # 红色通道噪声
            noise[0] = noise[0] * 1.2
            # 绿色通道噪声
            noise[1] = noise[1] * 0.9
            # 蓝色通道噪声
            noise[2] = noise[2] * 1.1
            
        # 混合原图和噪声
        tensor_image = tensor_image * 0.2 + noise * 0.8
        tensor_image = torch.clamp(tensor_image, 0.0, 1.0)
    
    # 转换为PIL图像
    tensor_image = tensor_image * 255
    tensor_image = tensor_image.to(torch.uint8)
    
    # 转换为PIL图像
    if tensor_image.shape[0] == 3:
        # 如果是[3, H, W]格式，转置为[H, W, 3]
        tensor_image = tensor_image.permute(1, 2, 0)
    
    # 转换为numpy数组
    numpy_image = tensor_image.numpy()
    
    # 创建PIL图像
    pil_image = Image.fromarray(numpy_image)
    
    # 轻微增强图像
    try:
        # 轻微增加对比度
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 轻微增加饱和度
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)
    except Exception as e:
        print(f"增强图像时出错: {e}")
    
    return pil_image


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
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate results during generation process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--random_seed",
        action="store_true",
        help="Use random seed instead of fixed seed"
    )
    parser.add_argument(
        "--use_original_qrcode",
        action="store_true",
        help="Use original QR code as base image with minimal modifications"
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
    # 检查 CUDA 和 GPU 内存
    check_cuda_and_memory()
    
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 创建保存中间结果的目录
    intermediate_dir = os.path.join(args.output_folder, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # 设置随机种子
    if args.random_seed:
        import random
        args.seed = random.randint(1, 1000000)
        print(f"使用随机种子: {args.seed}")
    else:
        print(f"使用固定种子: {args.seed}")
    
    # 设置 Hugging Face 镜像站点
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
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
    
    # 使用本地文件加载模型
    try:
        pipe = DiffQRCoderPipeline.from_single_file(
            args.pipe_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=True 
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保已下载所需的模型文件，并放置在正确的位置。")
        print(f"ControlNet模型路径: {args.controlnet_ckpt}")
        print(f"Pipeline模型路径: {args.pipe_ckpt}")
        exit(1)
    
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
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        scanning_robust_guidance_scale=scanning_robust_scale,
        perceptual_guidance_scale=args.perceptual_guidance_scale,
        srmpgd_num_iteration=args.srmpgd_num_iteration,
        srmpgd_lr=args.srmpgd_lr,
        save_intermediate_steps=args.save_intermediate,
        intermediate_dir=intermediate_dir,
        use_custom_postprocess=True,  # 使用自定义后处理
        use_original_qrcode=args.use_original_qrcode,  # 使用原始二维码
    )
    
    # 保存结果
    suffix = ""
    if args.logo_path:
        suffix = f"_with_logo_{args.logo_position}_{args.logo_size_ratio}"
        if args.extract_logo:
            suffix += "_extracted"
    
    output_filename = f"qrcode{suffix}.png"
    output_path = Path(args.output_folder, output_filename)
    
    # 检查图像是否为黑色
    result_img_array = np.array(result.images[0])
    print(f"最终图像像素值统计: min={result_img_array.min()}, max={result_img_array.max()}, mean={result_img_array.mean():.2f}")
    
    if result_img_array.mean() < 10:  # 如果平均值小于10，可能是黑色图像
        print("警告: 生成的图像可能是全黑的!")
        
        # 尝试调整图像亮度并保存一个副本
        try:
            enhanced_img = ImageEnhance.Brightness(result.images[0]).enhance(10.0)  # 增强亮度10倍
            enhanced_path = Path(args.output_folder, f"enhanced_{output_filename}")
            enhanced_img.save(enhanced_path)
            print(f"已保存增强亮度的图像到: {enhanced_path}")
        except Exception as e:
            print(f"增强图像亮度时出错: {e}")
    
    result.images[0].save(output_path)
    print(f"已保存结果到: {output_path}")
