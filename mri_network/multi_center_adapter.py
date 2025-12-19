import torch
import torch.nn as nn
from mri_network.promptmrplusV2 import PromptMR
import re
import fastmri
import math
import torch.nn.functional as F
from typing import Tuple, List


class FeatureAdapter(nn.Module):
    """特征级别的适配器 - 更稳定的版本"""

    def __init__(self, center_type=None, contrast_type=None):
        super().__init__()
        self.center_type = center_type
        self.contrast_type = contrast_type

        # 轻量级的残差适配器，添加BatchNorm提高稳定性
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        # 自适应权重，初始化为较小值
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # x: [B, H, W]
        if len(x.shape) == 3:
            x_input = x.unsqueeze(1)  # [B, 1, H, W]
        else:
            x_input = x

        # 生成适配残差
        residual = self.adapter(x_input)  # [B, 1, H, W]

        # 限制alpha范围，防止过大的调整
        alpha_clamped = torch.clamp(self.alpha, -0.1, 0.1)

        # 应用适配
        adapted = x_input + alpha_clamped * residual

        if len(x.shape) == 3:
            return adapted.squeeze(1)  # [B, H, W]
        else:
            return adapted


class MultiCenterAdaptivePromptMR(nn.Module):
    """多中心自适应PromptMR - 带UNet风格归一化"""

    def __init__(self, base_promptmr_config):
        super().__init__()
        # 基础PromptMR模型
        self.base_model = PromptMR(**base_promptmr_config)

        # 使用更稳定的特征适配器
        self.center_adapters = nn.ModuleDict({
            'Center001': FeatureAdapter(center_type='UIH_30T_umr780'),
            'Center002': FeatureAdapter(center_type='Siemens_30T_CIMA'),
            'Center003': FeatureAdapter(center_type='UIH_30T_umr880'),
            'Center005': FeatureAdapter(center_type='Mixed_Scanners'),
            'Center006': FeatureAdapter(center_type='Siemens_30T_Prisma'),
            'Center007': FeatureAdapter(center_type='Siemens_Mixed')
        })

        # 对比度特定适配器
        self.contrast_adapters = nn.ModuleDict({
            'Cine': FeatureAdapter(contrast_type='cardiac_motion'),
            'LGE': FeatureAdapter(contrast_type='scar_enhancement'),
            'Mapping': FeatureAdapter(contrast_type='quantitative'),
            'T1w': FeatureAdapter(contrast_type='t1_weighted'),
            'T2w': FeatureAdapter(contrast_type='t2_weighted'),
            'Perfusion': FeatureAdapter(contrast_type='dynamic'),
            'T1rho': FeatureAdapter(contrast_type='t1rho_specific'),
            'Flow2d': FeatureAdapter(contrast_type='flow_encoding'),
            'BlackBlood': FeatureAdapter(contrast_type='vessel_suppression')
        })

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """参考NormPromptUnet的归一化方法"""
        b, h, w = x.shape
        x_flat = x.reshape(b, h * w)

        mean = x_flat.mean(dim=1).view(b, 1, 1)
        std = x_flat.std(dim=1).view(b, 1, 1)

        # 防止除零
        std = torch.clamp(std, min=1e-8)

        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """参考NormPromptUnet的反归一化方法"""
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """参考NormPromptUnet的padding方法"""
        _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int],
              h_mult: int, w_mult: int) -> torch.Tensor:
        """参考NormPromptUnet的unpadding方法"""
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def extract_metadata_from_filename(self, filename):
        """从文件名提取元数据"""
        if isinstance(filename, (list, tuple)):
            filename = filename[0]

        # 提取中心ID
        center_match = re.search(r'(Center\d+)', filename)
        center_id = center_match.group(1) if center_match else 'unknown'

        # 提取对比度类型
        contrast_types = ['BlackBlood', 'Cine', 'LGE', 'Perfusion', 'Mapping', 'T1rho', 'T1w', 'T2w', 'Flow2d']
        contrast_type = 'unknown'
        for ct in contrast_types:
            if ct in filename:
                contrast_type = ct
                break

        return center_id, contrast_type

    def apply_adaptations(self, features, center_id, contrast_type):
        """应用适配器 - 使用UNet风格的归一化"""

        # 输入检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("WARNING: Input features contain NaN/Inf, skipping adaptation")
            return features

        # 保存原始形状
        original_shape = features.shape

        # Step 1: 归一化 (参考NormPromptUnet)
        try:
            features_norm, mean, std = self.norm(features)
        except Exception as e:
            print(f"ERROR in normalization: {e}")
            return features

        # 检查归一化结果
        if torch.isnan(features_norm).any() or torch.isinf(features_norm).any():
            print("WARNING: Normalization produced NaN/Inf")
            return features

        # Step 2: Padding (参考NormPromptUnet)
        try:
            features_padded, pad_sizes = self.pad(features_norm)
        except Exception as e:
            print(f"ERROR in padding: {e}")
            features_padded, pad_sizes = features_norm, None

        adapted = features_padded

        # Step 3: 应用适配器
        try:
            # 中心适配
            if center_id in self.center_adapters:
                center_adapted = self.center_adapters[center_id](adapted)
                if torch.isnan(center_adapted).any() or torch.isinf(center_adapted).any():
                    print(f"WARNING: Center adapter {center_id} produced NaN/Inf")
                else:
                    adapted = center_adapted

            # 对比度适配
            if contrast_type in self.contrast_adapters:
                contrast_adapted = self.contrast_adapters[contrast_type](adapted)
                if torch.isnan(contrast_adapted).any() or torch.isinf(contrast_adapted).any():
                    print(f"WARNING: Contrast adapter {contrast_type} produced NaN/Inf")
                else:
                    adapted = contrast_adapted

        except Exception as e:
            print(f"ERROR in adaptation: {e}")
            adapted = features_padded  # 回退到padding后的归一化特征

        # Step 4: Unpadding (参考NormPromptUnet)
        if pad_sizes is not None:
            try:
                adapted = self.unpad(adapted, *pad_sizes)
            except Exception as e:
                print(f"ERROR in unpadding: {e}")
                # 如果unpadding失败，尝试简单的裁剪
                target_h, target_w = original_shape[1], original_shape[2]
                current_h, current_w = adapted.shape[1], adapted.shape[2]
                if current_h >= target_h and current_w >= target_w:
                    h_start = (current_h - target_h) // 2
                    w_start = (current_w - target_w) // 2
                    adapted = adapted[:, h_start:h_start + target_h, w_start:w_start + target_w]

        # Step 5: 反归一化 (参考NormPromptUnet)
        try:
            adapted = self.unnorm(adapted, mean, std)
        except Exception as e:
            print(f"ERROR in unnormalization: {e}")
            return features

        # 最终检查
        if torch.isnan(adapted).any() or torch.isinf(adapted).any():
            print("WARNING: Final adapted result contains NaN/Inf, returning original features")
            return features

        # 确保输出形状正确
        if adapted.shape != original_shape:
            print(f"WARNING: Shape mismatch after adaptation: {adapted.shape} vs {original_shape}")
            try:
                adapted = F.interpolate(adapted.unsqueeze(1), size=original_shape[1:],
                                        mode='bilinear', align_corners=False).squeeze(1)
            except:
                return features

        return adapted

    def forward(self, masked_kspace, mask, filenames=None):
        # 基础重建
        recon_image = self.base_model(masked_kspace, mask)

        if torch.isnan(recon_image).any():
            print(3+'a')

        # 如果没有文件名信息，直接返回基础结果
        if filenames is None:
            return recon_image

        # 提取元数据
        center_id, contrast_type = self.extract_metadata_from_filename(filenames)

        # 应用适配（使用UNet风格的归一化）
        adapted_image = self.apply_adaptations(recon_image, center_id, contrast_type)

        return adapted_image