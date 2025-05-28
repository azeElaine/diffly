import os
import argparse
import random
import subprocess
from PIL import Image

# 预设风格提示词
STYLE_PRESETS = {
    "水彩": "Beautiful watercolor painting, soft colors, artistic, flowing, gentle brush strokes",
    "油画": "Oil painting with rich textures, vibrant colors, detailed brush strokes, artistic masterpiece",
    "赛博朋克": "Cyberpunk style, neon lights, futuristic, digital, high tech, urban, night city",
    "日式浮世绘": "Japanese ukiyo-e style, traditional, woodblock print, elegant, artistic",
    "中国水墨": "Chinese ink painting, traditional, elegant, flowing, minimal, black and white",
    "像素艺术": "Pixel art style, retro gaming, 8-bit, colorful blocks, nostalgic",
    "霓虹": "Neon sign, glowing, vibrant colors, night scene, urban, modern",
    "复古": "Vintage style, retro, nostalgic, old-fashioned, classic design",
    "未来主义": "Futuristic design, sleek, modern, high-tech, clean lines, minimalist",
    "自然": "Nature inspired, organic shapes, plants, flowers, green, peaceful",
    "几何": "Geometric patterns, abstract shapes, clean lines, modern design",
    "卡通": "Cartoon style, playful, colorful, fun, animated characters",
    "极简": "Minimalist design, clean, simple, elegant, uncluttered",
    "蒸汽朋克": "Steampunk style, vintage machinery, brass, copper, Victorian era, mechanical",
    "海洋": "Ocean themed, blue colors, waves, underwater scene, marine life",
    "太空": "Space themed, stars, galaxies, planets, cosmic, nebula, dark background"
}

def print_styles():
    """打印所有可用的风格预设"""
    print("\n可用的风格预设:")
    for i, (style, _) in enumerate(STYLE_PRESETS.items(), 1):
        print(f"{i}. {style}")

def generate_qrcode(args):
    """生成艺术二维码"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 设置随机种子
    if args.seed is None:
        args.seed = random.randint(1, 1000000)
    
    print(f"使用随机种子: {args.seed}")
    
    # 如果使用预设风格，获取对应的提示词
    prompt = args.prompt
    if args.style and args.style in STYLE_PRESETS:
        if prompt:
            prompt = f"{prompt}, {STYLE_PRESETS[args.style]}"
        else:
            prompt = STYLE_PRESETS[args.style]
    
    # 构建命令行参数
    cmd_args = [
        "python", "main.py",
        "--qrcode_path", args.qrcode,
        "--prompt", prompt,
        "--controlnet_ckpt", "control_v1p_sd15_qrcode_monster",
        "--pipe_ckpt", "cetus-mix/cetusMix_Whalefall2_fp16.safetensors",
        "--qrcode_module_size", "20",
        "--qrcode_padding", "70",
        "--num_inference_steps", "30",
        "--neg_prompt", "easynegative, blurry, distorted, low quality",
        "--controlnet_conditioning_scale", "0.9",
        "--scanning_robust_guidance_scale", "500",
        "--perceptual_guidance_scale", "2",
        "--srmpgd_num_iteration", "10",
        "--srmpgd_lr", "0.05",
        "--seed", str(args.seed),
        "--save_intermediate",
        "--output_folder", os.path.dirname(args.output)
    ]
    
    # 添加logo相关参数
    if args.logo:
        logo_args = [
            "--logo_path", args.logo,
            "--logo_guidance_scale", "200",
            "--logo_size_ratio", str(args.logo_size),
            "--logo_position", args.logo_position,
            "--edge_smoothness_weight", "0.2",
            "--extract_logo",
            "--bg_threshold", "240"
        ]
        cmd_args.extend(logo_args)
    
    # 如果需要使用原始二维码
    if args.use_original:
        cmd_args.append("--use_original_qrcode")
    
    # 运行命令
    print("开始生成艺术二维码...")
    print(f"二维码: {args.qrcode}")
    if args.logo:
        print(f"Logo: {args.logo}")
    print(f"提示词: {prompt}")
    
    result = subprocess.run(cmd_args)
    
    if result.returncode == 0:
        # 重命名输出文件
        import glob
        output_files = glob.glob(os.path.join(os.path.dirname(args.output), "qrcode*.png"))
        if output_files:
            latest_file = max(output_files, key=os.path.getctime)
            os.rename(latest_file, args.output)
            print(f"生成成功！输出文件: {args.output}")
        else:
            print("警告：找不到生成的输出文件")
    else:
        print("生成失败，请检查错误信息")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成艺术二维码")
    parser.add_argument("--qrcode", type=str, required=True, help="二维码图片路径")
    parser.add_argument("--logo", type=str, help="Logo图片路径")
    parser.add_argument("--prompt", type=str, help="文本提示词，描述想要的风格")
    parser.add_argument("--style", type=str, help="使用预设风格")
    parser.add_argument("--output", type=str, default="output/art_qrcode.png", help="输出图片路径")
    parser.add_argument("--logo_size", type=float, default=0.3, help="Logo尺寸比例(0.1-0.5)")
    parser.add_argument("--logo_position", type=str, default="center", 
                       choices=["center", "top_left", "top_right", "bottom_left", "bottom_right"],
                       help="Logo位置")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--use_original", action="store_true", help="使用原始二维码作为基础")
    parser.add_argument("--list_styles", action="store_true", help="列出所有可用的风格预设")
    args = parser.parse_args()
    
    # 如果请求列出风格预设，则打印并退出
    if args.list_styles:
        print_styles()
        return
    
    # 检查输入文件是否存在
    if not os.path.exists(args.qrcode):
        print(f"错误：二维码图片不存在 - {args.qrcode}")
        return
    
    if args.logo and not os.path.exists(args.logo):
        print(f"错误：Logo图片不存在 - {args.logo}")
        return
    
    # 检查是否提供了提示词或风格
    if not args.prompt and not args.style:
        print("错误：必须提供提示词(--prompt)或选择预设风格(--style)")
        print_styles()
        return
    
    # 如果提供了风格名称但不在预设中，提示用户
    if args.style and args.style not in STYLE_PRESETS:
        print(f"错误：未知的风格预设 '{args.style}'")
        print_styles()
        return
    
    # 生成二维码
    generate_qrcode(args)

if __name__ == "__main__":
    main() 