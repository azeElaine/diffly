# 艺术二维码生成器

这是一个基于扩散模型的艺术二维码生成器，可以将普通二维码与logo和艺术风格完美融合，生成既美观又可扫描的艺术二维码。

## 功能特点

- 支持自定义二维码输入
- 支持添加logo（可自动提取logo，去除背景）
- 多种预设艺术风格可选
- 自定义文本提示词描述想要的风格
- 可调整logo大小和位置
- 保持二维码的可扫描性

## 安装依赖

确保已安装所需的Python库：

```bash
pip install torch diffusers transformers kornia pillow
```

## 使用方法

### 基本用法

```bash
python art_qrcode.py --qrcode 你的二维码.png --style 水彩
```

### 添加logo

```bash
python art_qrcode.py --qrcode 你的二维码.png --logo 你的logo.png --style 油画
```

### 使用自定义提示词

```bash
python art_qrcode.py --qrcode 你的二维码.png --prompt "夜晚的城市，霓虹灯，下雨的街道"
```

### 查看所有可用的风格预设

```bash
python art_qrcode.py --list_styles
```

### 完整参数说明

```
--qrcode        二维码图片路径（必需）
--logo          Logo图片路径（可选）
--prompt        文本提示词，描述想要的风格（与--style二选一）
--style         使用预设风格（与--prompt二选一）
--output        输出图片路径（默认：output/art_qrcode.png）
--logo_size     Logo尺寸比例，范围0.1-0.5（默认：0.3）
--logo_position Logo位置，可选：center, top_left, top_right, bottom_left, bottom_right（默认：center）
--seed          随机种子（可选）
--use_original  使用原始二维码作为基础（可选）
--list_styles   列出所有可用的风格预设
```

## 可用的风格预设

- 水彩
- 油画
- 赛博朋克
- 日式浮世绘
- 中国水墨
- 像素艺术
- 霓虹
- 复古
- 未来主义
- 自然
- 几何
- 卡通
- 极简
- 蒸汽朋克
- 海洋
- 太空

## 示例

生成水彩风格的二维码：
```bash
python art_qrcode.py --qrcode qrcode/example.png --style 水彩 --output output/watercolor_qrcode.png
```

生成带logo的赛博朋克风格二维码：
```bash
python art_qrcode.py --qrcode qrcode/example.png --logo logo.png --style 赛博朋克 --logo_position center --logo_size 0.3 --output output/cyberpunk_qrcode.png
```

## 故障排除

如果生成的二维码无法扫描，可以尝试以下方法：

1. 增加 `--use_original` 参数，使用原始二维码作为基础
2. 尝试不同的随机种子
3. 调整logo大小和位置
4. 使用更简单的艺术风格，如"极简"或"水彩"

## 注意事项

- 生成的二维码质量可能会因为不同的提示词和随机种子而有所不同
- 较复杂的艺术风格可能会降低二维码的可扫描性
- 添加logo会占用二维码的一部分区域，可能会影响可扫描性，建议使用较小的logo或将logo放在角落位置 