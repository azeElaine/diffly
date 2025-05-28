import torch
import torch.nn.functional as F
from torch import nn
import os
from .perceptual_loss import VGGFeatureExtractor

class LogoLoss(nn.Module):
    def __init__(self,  
        feature_layer: int = 34,        # 这个参数现在不会传递给VGGFeatureExtractor  
        use_bn: bool = False,           # 这个参数现在不会传递给VGGFeatureExtractor  
        use_input_norm: bool = True,    # 这个参数现在不会传递给VGGFeatureExtractor  
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    ):  
        super().__init__()  
          
        # 使用现有VGGFeatureExtractor的正确参数  
        self.feature_extractor = VGGFeatureExtractor(  
            requires_grad=False,  
            pretrained_weights="DEFAULT"  
        ).to(device)
        
        # 从环境变量读取边缘平滑权重，如果不存在则使用默认值
        self.edge_smoothness_weight = 0.1
        if "LOGO_EDGE_SMOOTHNESS_WEIGHT" in os.environ:
            try:
                self.edge_smoothness_weight = float(os.environ["LOGO_EDGE_SMOOTHNESS_WEIGHT"])
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值
                pass

    def forward(self,
        generated_image: torch.Tensor,
        logo_image: torch.Tensor,
        logo_mask: torch.Tensor
    ) -> torch.Tensor:
        # 打印数据类型信息，用于调试
        print(f"Logo Loss - Input types: generated_image={generated_image.dtype}, logo_image={logo_image.dtype}, logo_mask={logo_mask.dtype}")
        
        # 提取logo区域特征
        logo_region = generated_image * logo_mask
        target_region = logo_image * logo_mask

        # 计算像素级MSE损失 - 保持logo的精确细节
        pixel_loss = F.mse_loss(logo_region, target_region)
        
        # 计算特征相似性损失 - 保持高级特征
        generated_features = self.feature_extractor(logo_region)
        target_features = self.feature_extractor(target_region)

        feature_loss = 0
        for gen_feat, tar_feat in zip(generated_features, target_features):
            feature_loss += F.mse_loss(gen_feat, tar_feat)
        feature_loss = feature_loss / len(generated_features)
        
        # 计算边缘平滑损失 - 确保与周围二维码区域的自然过渡
        # 创建略微扩大的mask以获取边缘区域
        # 显式检查并匹配数据类型
        dtype = logo_mask.dtype
        edge_kernel = torch.ones(1, 1, 3, 3, dtype=dtype, device=logo_mask.device)
        print(f"Logo Loss - Filter types: edge_kernel={edge_kernel.dtype}, logo_mask={logo_mask.dtype}")
        dilated_mask = F.conv2d(logo_mask, edge_kernel, padding=1)
        dilated_mask = torch.clamp(dilated_mask, 0, 1)
        edge_mask = dilated_mask - logo_mask
        
        # 提取边缘区域的图像
        edge_region_generated = generated_image * edge_mask
        
        # 使用索贝尔滤波器检测边缘上的梯度
        # 确保与输入图像相同的数据类型
        dtype = generated_image.dtype
        device = generated_image.device
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        
        # 对每个通道单独应用
        edge_smoothness_loss = 0
        for c in range(generated_image.shape[1]):
            channel = edge_region_generated[:, c:c+1]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            gradients = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            edge_smoothness_loss += torch.mean(gradients)
        
        # 组合三种损失，使用从环境变量读取的权重
        total_loss = pixel_loss + feature_loss + self.edge_smoothness_weight * edge_smoothness_loss
        
        return total_loss
