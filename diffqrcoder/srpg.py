import torch
from torch import nn

from diffqrcoder.losses import PerceptualLoss, ScanningRobustLoss
from diffqrcoder.losses import LogoLoss


GRADIENT_SCALE = 100


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
        self.logo_loss_fn = LogoLoss(feature_layer=feature_layer, use_input_norm=use_normalize) 

    def compute_loss(self,
        image: torch.Tensor,
        qrcode: torch.Tensor,
        ref_image: torch.Tensor,
        logo_image: torch.Tensor = None,
        logo_mask: torch.Tensor = None) -> torch.Tensor:
        
        # 如果没有提供logo相关参数，使用原来的损失计算
        if logo_image is None or logo_mask is None:
            loss = (
                self.scanning_robust_guidance_scale * self.scanning_robust_loss_fn(image, qrcode) +
                self.perceptual_guidance_scale * self.perceptual_loss_fn(image, ref_image)
            )
            return loss * GRADIENT_SCALE
        
        # 确保mask格式正确
        if len(logo_mask.shape) == 3:
            logo_mask = logo_mask.unsqueeze(1)
        logo_mask = logo_mask.to(image.dtype)
        
        # 为logo区域和非logo区域创建反向掩码
        non_logo_mask = 1.0 - logo_mask
        
        # 计算logo区域的损失 - 只使用logo损失
        logo_loss = self.logo_loss_fn(image, logo_image, logo_mask)
        
        # 计算非logo区域的损失 - 增强扫描鲁棒性损失
        # 对非logo区域应用更高权重的扫描鲁棒性损失，确保它能被正确识别为二维码
        # 使用 * 运算来限制损失计算范围在非logo区域
        scanning_robust_loss = self.scanning_robust_loss_fn(
            image * non_logo_mask, 
            qrcode * non_logo_mask
        )
        
        # 对非logo区域应用极低权重的感知损失，主要目的是使其在视觉上与整体协调
        # 但扫描鲁棒性仍然是最重要的
        perceptual_loss = self.perceptual_loss_fn(
            image * non_logo_mask, 
            ref_image * non_logo_mask
        )
        
        # 合并损失 - 为非logo区域的扫描鲁棒性损失提供更高权重
        # 这样可以确保二维码部分能被正确识别
        total_loss = (
            self.logo_guidance_scale * logo_loss +  # logo区域应用高权重的logo损失
            (self.scanning_robust_guidance_scale * 1.5) * scanning_robust_loss +  # 非logo区域应用增强的扫描鲁棒性损失
            (self.perceptual_guidance_scale * 0.3) * perceptual_loss  # 对非logo区域应用极低权重的感知损失
        )
            
        return total_loss * GRADIENT_SCALE

    def compute_score(self, latents: torch.Tensor, image: torch.Tensor, qrcode: torch.Tensor,
                     ref_image: torch.Tensor, logo_image: torch.Tensor = None,
                     logo_mask: torch.Tensor = None) -> torch.Tensor:
        loss = self.compute_loss(image, qrcode, ref_image, logo_image, logo_mask)
        return -torch.autograd.grad(loss, latents)[0]
