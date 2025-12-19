import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as cal_ssim, ms_ssim
import fastmri


class AdaptiveMultiScaleSSIMLoss(nn.Module):
    """自适应多尺度SSIM损失 - 针对测试SSIM优化"""

    def __init__(self, scales=[1, 2, 4, 8], adaptive_weights=True):
        super().__init__()
        self.scales = scales
        self.adaptive_weights = adaptive_weights

        # 可学习的尺度权重
        if adaptive_weights:
            self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        else:
            self.register_buffer('scale_weights', torch.ones(len(scales)))

    def forward(self, pred, target):
        total_loss = 0
        weights = F.softmax(self.scale_weights, dim=0) if self.adaptive_weights else self.scale_weights

        for i, scale in enumerate(self.scales):
            if scale > 1:
                # 使用更好的下采样方法
                kernel_size = scale
                pred_scaled = F.avg_pool2d(pred.unsqueeze(1), kernel_size, stride=scale).squeeze(1)
                target_scaled = F.avg_pool2d(target.unsqueeze(1), kernel_size, stride=scale).squeeze(1)
            else:
                pred_scaled = pred
                target_scaled = target

            # 更稳定的归一化
            pred_norm = self.robust_normalize(pred_scaled)
            target_norm = self.robust_normalize(target_scaled)

            # 使用更大的窗口尺寸提高SSIM稳定性
            win_size = min(7, min(pred_norm.shape[-2:]) // 4 * 2 + 1)  # 自适应窗口大小
            scale_loss = 1 - cal_ssim(pred_norm.unsqueeze(1), target_norm.unsqueeze(1),
                                      data_range=1.0, win_size=win_size)

            total_loss += weights[i] * scale_loss

        return total_loss

    def robust_normalize(self, x):
        """更鲁棒的归一化方法"""
        # 使用percentile归一化，避免异常值影响
        x_flat = x.view(x.shape[0], -1)
        p1 = torch.quantile(x_flat, 0.01, dim=1, keepdim=True).unsqueeze(-1)
        p99 = torch.quantile(x_flat, 0.99, dim=1, keepdim=True).unsqueeze(-1)

        x_norm = (x - p1) / (p99 - p1 + 1e-8)
        return torch.clamp(x_norm, 0, 1)


class EnhancedSSIMOptimizedLoss(nn.Module):
    """增强版SSIM优化损失函数 - 专门针对高SSIM优化"""

    def __init__(self):
        super().__init__()
        self.ms_ssim_loss = AdaptiveMultiScaleSSIMLoss(scales=[1, 2, 4, 8])
        self.epoch = 0

        # 中心特定的权重 - 根据你的测试结果调整
        self.center_weights = {
            'Center001': {'boost': 1.0},  # UIH_30T_umr780 表现中等
            'Center002': {'boost': 1.3},  # Siemens_30T_CIMA 表现最差，需要更多关注
            'Center003': {'boost': 1.1},  # UIH_30T_umr880 表现中等
            'Center004': {'boost': 1.2},  # Siemens_15T_Aera 表现较差
            'Center005': {'boost': 1.1},  # GE_15T_voyager 表现中等
            'Center006': {'boost': 0.9},  # UIH_30T_umr790 表现最好，可以降低权重
            'Center008': {'boost': 1.1},  # GE_15T_voyager 表现中等
            'default': {'boost': 1.0}
        }

        # 对比度特定权重 - 更精细化
        self.contrast_weights = {
            'Cine': {'ssim': 1.2, 'ms_ssim': 0.4, 'freq': 0.3, 'edge': 0.2},  # 心脏运动，需要更好的时间一致性
            'LGE': {'ssim': 1.3, 'ms_ssim': 0.5, 'freq': 0.4, 'edge': 0.3},  # 疤痕增强，需要高对比度
            'Mapping': {'ssim': 1.1, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.1},  # 定量成像，需要精确性
            'T1w': {'ssim': 1.0, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.2},
            'T2w': {'ssim': 1.0, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.2},
            'Perfusion': {'ssim': 1.2, 'ms_ssim': 0.4, 'freq': 0.3, 'edge': 0.2},  # 动态成像
            'T1rho': {'ssim': 1.1, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.1},
            'Flow2d': {'ssim': 1.1, 'ms_ssim': 0.3, 'freq': 0.3, 'edge': 0.2},  # 流量编码
            'BlackBlood': {'ssim': 1.0, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.3},  # 血管抑制
            'default': {'ssim': 1.0, 'ms_ssim': 0.3, 'freq': 0.2, 'edge': 0.2}
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, pred, target, contrast_type=None, center_id=None):
        # 动态权重调整 - 随着训练进行，更加关注SSIM
        epoch_factor = min(1.0, self.epoch / 15)
        ssim_focus = 1.0 + 0.5 * epoch_factor  # SSIM权重随epoch增加

        # 基础SSIM损失 - 使用更大的窗口
        pred_norm = self.robust_normalize(pred)
        target_norm = self.robust_normalize(target)

        # 自适应窗口大小
        min_dim = min(pred.shape[-2:])
        win_size = min(11, min_dim // 6 * 2 + 1)  # 更大的窗口，更稳定的SSIM

        ssim_loss = 1 - cal_ssim(pred_norm.unsqueeze(1), target_norm.unsqueeze(1),
                                 data_range=1.0, win_size=win_size)

        # MS-SSIM损失 - 使用pytorch内置的更稳定版本
        ms_ssim_loss = 1 - ms_ssim(pred_norm.unsqueeze(1), target_norm.unsqueeze(1),
                                   data_range=1.0, win_size=win_size)

        # 频域SSIM损失
        freq_ssim_loss = self.enhanced_frequency_domain_ssim(pred, target)

        # 边缘保持损失
        edge_loss = self.edge_preserving_loss(pred, target)

        # 获取权重
        contrast_weights = self.contrast_weights.get(contrast_type, self.contrast_weights['default'])
        center_boost = self.center_weights.get(center_id, self.center_weights['default'])['boost']

        # 组合损失
        total_loss = (
                contrast_weights['ssim'] * ssim_focus * ssim_loss +
                contrast_weights['ms_ssim'] * ms_ssim_loss +
                contrast_weights['freq'] * freq_ssim_loss +
                contrast_weights['edge'] * edge_loss
        )

        # 应用中心特定的增强
        total_loss = total_loss * center_boost

        return total_loss

    def robust_normalize(self, x):
        """鲁棒归一化 - 避免异常值影响SSIM计算"""
        x_flat = x.view(x.shape[0], -1)

        # 使用更稳定的归一化方法
        mean = x_flat.mean(dim=1, keepdim=True).unsqueeze(-1)
        std = x_flat.std(dim=1, keepdim=True).unsqueeze(-1)

        # Z-score归一化后映射到[0,1]
        x_normalized = (x - mean) / (std + 1e-8)
        x_normalized = torch.sigmoid(x_normalized)  # 映射到[0,1]

        return x_normalized

    def enhanced_frequency_domain_ssim(self, pred, target):
        """增强的频域SSIM损失"""
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # 幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # 相位谱
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        # 幅度SSIM
        pred_mag_norm = self.robust_normalize(pred_mag)
        target_mag_norm = self.robust_normalize(target_mag)
        mag_ssim_loss = 1 - cal_ssim(pred_mag_norm.unsqueeze(1), target_mag_norm.unsqueeze(1),
                                     data_range=1.0, win_size=3)

        # 相位一致性损失
        phase_diff = torch.cos(pred_phase - target_phase)  # 相位差的余弦
        phase_loss = 1 - phase_diff.mean()

        return 0.7 * mag_ssim_loss + 0.3 * phase_loss

    def edge_preserving_loss(self, pred, target):
        """边缘保持损失 - 提高细节重建质量"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

        # 计算梯度
        pred_grad_x = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred.unsqueeze(1), sobel_y, padding=1)
        target_grad_x = F.conv2d(target.unsqueeze(1), sobel_x, padding=1)
        target_grad_y = F.conv2d(target.unsqueeze(1), sobel_y, padding=1)

        # 梯度幅度
        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

        # 归一化梯度
        pred_grad_norm = self.robust_normalize(pred_grad_mag.squeeze(1))
        target_grad_norm = self.robust_normalize(target_grad_mag.squeeze(1))

        # 梯度SSIM
        edge_ssim_loss = 1 - cal_ssim(pred_grad_norm.unsqueeze(1), target_grad_norm.unsqueeze(1),
                                      data_range=1.0, win_size=3)

        return edge_ssim_loss


# 使用方式
class SSIMFocusedLoss(nn.Module):
    """专门针对测试SSIM优化的损失函数"""

    def __init__(self):
        super().__init__()
        self.enhanced_loss = EnhancedSSIMOptimizedLoss()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.enhanced_loss.set_epoch(epoch)

    def forward(self, pred, target, contrast_type=None, filenames=None):
        # 提取中心ID
        center_id = None
        if filenames is not None:
            filename = str(filenames[0]) if isinstance(filenames, (list, tuple)) else str(filenames)
            import re
            center_match = re.search(r'(Center\d+)', filename)
            center_id = center_match.group(1) if center_match else None

        return self.enhanced_loss(pred, target, contrast_type, center_id)