import torch
from torch import nn

from diffqrcoder.losses import PerceptualLoss, ScanningRobustLoss
from .logo_loss import LogoLoss

# 梯度缩放因子，用于避免梯度爆炸
GRADIENT_SCALE = 1.0

class ScanningRobustPerceptualGuidance(nn.Module):
    def __init__(
        self,
        module_size: int = 20,
        scanning_robust_guidance_scale: int = 500,
        perceptual_guidance_scale: int = 2,
        logo_guidance_scale: int = 100,
        feature_layer: int = 34,
        use_normalize: bool = True
    ):
        super().__init__()
        self.module_size = module_size
        self.scanning_robust_guidance_scale = scanning_robust_guidance_scale
        self.perceptual_guidance_scale = perceptual_guidance_scale
        self.logo_guidance_scale = logo_guidance_scale
        self.scanning_robust_loss_fn = ScanningRobustLoss(module_size=module_size)
        self.perceptual_loss_fn = PerceptualLoss()
        self.logo_loss_fn = LogoLoss(logo_guidance_scale)
        self.device = None
        self.dtype = None

    def to(self, device):
        self.device = device
        self.scanning_robust_loss_fn = self.scanning_robust_loss_fn.to(device)
        self.perceptual_loss_fn = self.perceptual_loss_fn.to(device)
        return self

    def to_dtype(self, dtype):
        self.dtype = dtype
        return self

    def compute_loss(
        self,
        image: torch.Tensor,
        qrcode: torch.Tensor,
        ref_image: torch.Tensor = None,
        logo_image: torch.Tensor = None,
        logo_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 检查并修复输入中的NaN值
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.0)
        if torch.isnan(qrcode).any():
            qrcode = torch.nan_to_num(qrcode, nan=0.0)
        if ref_image is not None and torch.isnan(ref_image).any():
            ref_image = torch.nan_to_num(ref_image, nan=0.0)
        
        # 计算扫描鲁棒性损失
        scanning_robust_loss = self.scanning_robust_loss_fn(image, qrcode)
        
        # 计算感知损失
        perceptual_loss = torch.tensor(0.0, device=image.device)
        if ref_image is not None:
            perceptual_loss = self.perceptual_loss_fn(image, ref_image)
        
        # 计算Logo损失
        logo_loss = torch.tensor(0.0, device=image.device)
        if logo_image is not None and logo_mask is not None:
            # 确保mask格式正确
            if len(logo_mask.shape) == 3:
                logo_mask = logo_mask.unsqueeze(1)
            
            # 确保数据类型一致
            logo_mask = logo_mask.to(device=image.device, dtype=image.dtype)
            logo_image = logo_image.to(device=image.device, dtype=image.dtype)
            
            # 为logo区域和非logo区域创建反向掩码
            non_logo_mask = 1.0 - logo_mask
            
            # 计算logo区域的损失
            logo_loss = self.logo_loss_fn.compute_loss(image, logo_image, logo_mask)
            
            # 对非logo区域应用更高权重的扫描鲁棒性损失
            scanning_robust_loss = self.scanning_robust_loss_fn(
                image * non_logo_mask, 
                qrcode * non_logo_mask
            )
        
        # 组合损失
        total_loss = (
            self.scanning_robust_guidance_scale * scanning_robust_loss +
            self.perceptual_guidance_scale * perceptual_loss +
            logo_loss  # logo_loss已经在内部应用了缩放因子
        )
        
        return total_loss * GRADIENT_SCALE

    def compute_score(
        self,
        latents: torch.Tensor,
        image: torch.Tensor,
        qrcode: torch.Tensor,
        ref_image: torch.Tensor = None,
        logo_image: torch.Tensor = None,
        logo_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """计算梯度得分"""
        # 检查并修复输入中的NaN值
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.0)
        if torch.isnan(qrcode).any():
            qrcode = torch.nan_to_num(qrcode, nan=0.0)
        if ref_image is not None and torch.isnan(ref_image).any():
            ref_image = torch.nan_to_num(ref_image, nan=0.0)
        
        # 计算损失
        loss = self.compute_loss(image, qrcode, ref_image, logo_image, logo_mask)
        
        # 如果损失是NaN，返回零梯度
        if torch.isnan(loss):
            print("警告：损失为NaN，返回零梯度")
            return torch.zeros_like(latents)
        
        try:
            # 计算梯度
            grad = torch.autograd.grad(loss, latents)[0]
            
            # 检查并修复梯度中的NaN值
            if torch.isnan(grad).any():
                print("警告：梯度包含NaN值，使用零梯度")
                grad = torch.zeros_like(grad)
            
            return -grad  # 注意负号，与原始实现保持一致
        except Exception as e:
            print(f"计算梯度时出错: {e}")
            return torch.zeros_like(latents)
