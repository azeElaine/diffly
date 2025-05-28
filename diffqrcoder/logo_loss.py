import torch
import torch.nn.functional as F

class LogoLoss:
    def __init__(self, logo_guidance_scale=100):
        self.logo_guidance_scale = logo_guidance_scale
        # 获取环境变量中的边缘平滑权重，如果不存在则使用默认值
        import os
        self.edge_smoothness_weight = float(os.environ.get("LOGO_EDGE_SMOOTHNESS_WEIGHT", "0.1"))
        print(f"Logo边缘平滑权重: {self.edge_smoothness_weight}")

    def compute_loss(self, image, logo_image, logo_mask):
        """计算logo损失，确保logo区域保持原样"""
        if logo_image is None or logo_mask is None:
            return torch.tensor(0.0, device=image.device)
        
        # 确保所有输入都在同一设备上并且类型相同
        logo_image = logo_image.to(device=image.device, dtype=image.dtype)
        logo_mask = logo_mask.to(device=image.device, dtype=image.dtype)
        
        # 检查并修复NaN值
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.0)
        if torch.isnan(logo_image).any():
            logo_image = torch.nan_to_num(logo_image, nan=0.0)
        if torch.isnan(logo_mask).any():
            logo_mask = torch.nan_to_num(logo_mask, nan=0.0)
        
        # 计算logo区域的差异
        logo_diff = (image - logo_image) ** 2
        logo_loss = (logo_diff * logo_mask).mean()
        
        # 计算logo边缘的平滑度
        if self.edge_smoothness_weight > 0:
            # 使用sobel算子检测边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            
            # 计算logo边缘
            logo_mask_expanded = logo_mask.expand(-1, 3, -1, -1)
            edge_mask = F.conv2d(logo_mask_expanded[:, 0:1], sobel_x, padding=1) ** 2 + F.conv2d(logo_mask_expanded[:, 0:1], sobel_y, padding=1) ** 2
            edge_mask = edge_mask.sqrt()
            
            # 归一化边缘掩码
            edge_mask = edge_mask / (edge_mask.max() + 1e-6)
            
            # 计算边缘区域的平滑度
            edge_loss = (logo_diff * edge_mask).mean()
            
            # 组合损失
            total_loss = logo_loss + self.edge_smoothness_weight * edge_loss
        else:
            total_loss = logo_loss
        
        # 应用logo引导比例
        return total_loss * self.logo_guidance_scale
    
    def compute_score(self, image, logo_image, logo_mask):
        """计算logo得分，用于梯度更新"""
        if logo_image is None or logo_mask is None:
            return torch.zeros_like(image)
        
        # 确保所有输入都在同一设备上并且类型相同
        logo_image = logo_image.to(device=image.device, dtype=image.dtype)
        logo_mask = logo_mask.to(device=image.device, dtype=image.dtype)
        
        # 检查并修复NaN值
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.0)
        if torch.isnan(logo_image).any():
            logo_image = torch.nan_to_num(logo_image, nan=0.0)
        if torch.isnan(logo_mask).any():
            logo_mask = torch.nan_to_num(logo_mask, nan=0.0)
        
        # 计算logo区域的得分
        logo_diff = (image - logo_image)
        logo_score = logo_diff * logo_mask
        
        # 计算logo边缘的平滑度得分
        if self.edge_smoothness_weight > 0:
            # 使用sobel算子检测边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            
            # 计算logo边缘
            logo_mask_expanded = logo_mask.expand(-1, 3, -1, -1)
            edge_mask = F.conv2d(logo_mask_expanded[:, 0:1], sobel_x, padding=1) ** 2 + F.conv2d(logo_mask_expanded[:, 0:1], sobel_y, padding=1) ** 2
            edge_mask = edge_mask.sqrt()
            
            # 归一化边缘掩码
            edge_mask = edge_mask / (edge_mask.max() + 1e-6)
            
            # 计算边缘区域的平滑度得分
            edge_score = logo_diff * edge_mask
            
            # 组合得分
            total_score = logo_score + self.edge_smoothness_weight * edge_score
        else:
            total_score = logo_score
        
        # 应用logo引导比例
        return total_score * self.logo_guidance_scale 