import os
import glob
from scipy.io import loadmat
import h5py
import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

root = '/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/preprocess1/center002_map'


def loadmat(filename):
    """
    Load .mat file using appropriate method based on file format.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    try:
        data = scipy.io.loadmat(filename)
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        return data
    except (NotImplementedError, ValueError) as e:
        try:
            with h5py.File(filename, 'r') as f:
                data = {}
                for k, v in f.items():
                    if isinstance(v, h5py.Dataset):
                        data[k] = v[()]
                    elif isinstance(v, h5py.Group):
                        data[k] = loadmat_group(v)
            return data
        except Exception as h5_error:
            raise RuntimeError(f"Could not load {filename} with either method. "
                               f"scipy error: {e}, h5py error: {h5_error}")


def loadmat_group(group):
    """Load a group in Matlab v7.3 format .mat file using h5py."""
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


def analyze_kspace(kspace_data):
    """分析k-space数据的统计特征"""
    # 如果是复数数据，计算幅度
    if np.iscomplexobj(kspace_data):
        magnitude = np.abs(kspace_data)
        real_part = np.real(kspace_data)
        imag_part = np.imag(kspace_data)
    else:
        # 假设最后一维是 [real, imag]
        if kspace_data.shape[-1] == 2:
            real_part = kspace_data[..., 0]
            imag_part = kspace_data[..., 1]
            magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        else:
            magnitude = np.abs(kspace_data)
            real_part = kspace_data
            imag_part = None

    stats = {
        'shape': kspace_data.shape,
        'dtype': kspace_data.dtype,
        'magnitude_min': np.min(magnitude),
        'magnitude_max': np.max(magnitude),
        'magnitude_mean': np.mean(magnitude),
        'magnitude_std': np.std(magnitude),
        'magnitude_median': np.median(magnitude),
        'has_nan': np.isnan(kspace_data).any(),
        'has_inf': np.isinf(kspace_data).any(),
        'has_zero': np.any(magnitude == 0),
        'zero_percentage': np.sum(magnitude == 0) / magnitude.size * 100,
    }

    # 检查异常值
    stats['very_large_values'] = np.sum(magnitude > 1e10)
    stats['very_small_values'] = np.sum((magnitude > 0) & (magnitude < 1e-10))

    # 检查动态范围
    non_zero_magnitude = magnitude[magnitude > 0]
    if len(non_zero_magnitude) > 0:
        stats['dynamic_range'] = np.max(non_zero_magnitude) / np.min(non_zero_magnitude)
    else:
        stats['dynamic_range'] = 0

    # 检查实部和虚部
    if imag_part is not None:
        stats['real_min'] = np.min(real_part)
        stats['real_max'] = np.max(real_part)
        stats['imag_min'] = np.min(imag_part)
        stats['imag_max'] = np.max(imag_part)
        stats['real_has_nan'] = np.isnan(real_part).any()
        stats['imag_has_nan'] = np.isnan(imag_part).any()

    return stats


def main():
    files = sorted(glob.glob(os.path.join(root, "**/*.h5"), recursive=True))

    print("K-space数据分析报告")
    print("=" * 80)

    # 存储统计信息
    all_stats = []
    problem_files = []
    size_stats = defaultdict(list)

    for i, file_path in enumerate(files):
        try:
            hf = loadmat(file_path)
            kspace = hf["kspace"]
            f_name = file_path.split('/')[-1]

            # 分析k-space
            stats = analyze_kspace(kspace)
            stats['filename'] = f_name
            all_stats.append(stats)

            # 按尺寸分组
            size_key = f"{stats['shape'][-3]}x{stats['shape'][-2]}"
            size_stats[size_key].append(stats)

            # 检查问题文件
            is_problem = False
            problems = []

            if stats['has_nan']:
                problems.append("HAS_NaN")
                is_problem = True
            if stats['has_inf']:
                problems.append("HAS_INF")
                is_problem = True
            if stats['magnitude_max'] > 1e8:
                problems.append(f"VERY_LARGE_MAX({stats['magnitude_max']:.2e})")
                is_problem = True
            if stats['dynamic_range'] > 1e12:
                problems.append(f"HIGH_DYNAMIC_RANGE({stats['dynamic_range']:.2e})")
                is_problem = True
            if stats['very_large_values'] > 0:
                problems.append(f"LARGE_VALUES_COUNT({stats['very_large_values']})")
                is_problem = True

            if is_problem:
                problem_files.append({
                    'filename': f_name,
                    'problems': problems,
                    'stats': stats
                })

            # 打印基本信息
            status = "PROBLEM" if is_problem else "OK"
            print(f"{status} {f_name:60} {stats['shape']} "
                  f"Max: {stats['magnitude_max']:.2e} "
                  f"Range: {stats['dynamic_range']:.2e}")

            # 如果有问题，打印详细信息
            if is_problem:
                print(f"    Problems: {', '.join(problems)}")
                if stats['has_nan']:
                    print(f"Contains NaN values!")
                if stats['has_inf']:
                    print(f"Contains Inf values!")

        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")
            continue

    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)

    # 总体统计
    total_files = len(all_stats)
    problem_count = len(problem_files)
    print(f"总文件数: {total_files}")
    print(f"问题文件数: {problem_count} ({problem_count / total_files * 100:.1f}%)")

    # 幅度统计
    all_max_values = [s['magnitude_max'] for s in all_stats]
    all_min_values = [s['magnitude_min'] for s in all_stats]
    all_mean_values = [s['magnitude_mean'] for s in all_stats]

    print(f"\n K-space幅度统计:")
    print(f"  最大值范围: {np.min(all_max_values):.2e} ~ {np.max(all_max_values):.2e}")
    print(f"  最小值范围: {np.min(all_min_values):.2e} ~ {np.max(all_min_values):.2e}")
    print(f"  平均值范围: {np.min(all_mean_values):.2e} ~ {np.max(all_mean_values):.2e}")

    # 按尺寸统计
    print(f"\n 按图像尺寸统计:")
    for size, stats_list in sorted(size_stats.items()):
        max_vals = [s['magnitude_max'] for s in stats_list]
        problem_in_size = sum(1 for s in stats_list if s['magnitude_max'] > 1e8 or s['has_nan'] or s['has_inf'])
        print(f"  {size:15} 文件数: {len(stats_list):3d}, "
              f"问题文件: {problem_in_size:2d}, "
              f"最大值范围: {np.min(max_vals):.2e}~{np.max(max_vals):.2e}")

    # 详细问题报告
    if problem_files:
        print(f"\n  问题文件详细报告:")
        print("-" * 80)
        for pf in problem_files[:10]:  # 只显示前10个
            stats = pf['stats']
            print(f"文件: {pf['filename']}")
            print(f"  形状: {stats['shape']}")
            print(f"  问题: {', '.join(pf['problems'])}")
            print(f"  幅度: min={stats['magnitude_min']:.2e}, max={stats['magnitude_max']:.2e}")
            print(f"  动态范围: {stats['dynamic_range']:.2e}")
            if 'real_max' in stats:
                print(f"  实部范围: [{stats['real_min']:.2e}, {stats['real_max']:.2e}]")
                print(f"  虚部范围: [{stats['imag_min']:.2e}, {stats['imag_max']:.2e}]")
            print()

        if len(problem_files) > 10:
            print(f"... 还有 {len(problem_files) - 10} 个问题文件")

    # 生成修复建议
    print(f"\n 修复建议:")
    print("-" * 40)

    extreme_max = max(all_max_values)
    if extreme_max > 1e10:
        print(f"1. 发现极大值 ({extreme_max:.2e})，建议设置k-space clipping阈值为 1e8")

    if problem_count > 0:
        print(f"2. {problem_count} 个文件有数值问题，建议在训练前添加数据清理")

    # 建议的clipping阈值
    percentile_99 = np.percentile(all_max_values, 99)
    percentile_95 = np.percentile(all_max_values, 95)
    print(f"3. 建议的clipping阈值: {percentile_99:.2e} (99%分位数)")
    print(f"4. 保守的clipping阈值: {percentile_95:.2e} (95%分位数)")

    print(f"\n 推荐的数据预处理代码:")
    print(f"```python")
    print(f"def preprocess_kspace(kspace_data, clip_threshold={percentile_95:.1e}):")
    print(f"    # 检查并处理NaN和Inf")
    print(f"    if np.isnan(kspace_data).any() or np.isinf(kspace_data).any():")
    print(f"        print('Warning: Found NaN or Inf values, replacing with zeros')")
    print(f"        kspace_data = np.nan_to_num(kspace_data, nan=0.0, posinf=0.0, neginf=0.0)")
    print(f"    ")
    print(f"    # 计算幅度并进行clipping")
    print(f"    if np.iscomplexobj(kspace_data):")
    print(f"        magnitude = np.abs(kspace_data)")
    print(f"        if magnitude.max() > clip_threshold:")
    print(f"            scale_factor = clip_threshold / magnitude.max()")
    print(f"            kspace_data = kspace_data * scale_factor")
    print(f"    else:")
    print(f"        # 假设最后一维是[real, imag]")
    print(f"        if kspace_data.shape[-1] == 2:")
    print(f"            magnitude = np.sqrt(kspace_data[..., 0]**2 + kspace_data[..., 1]**2)")
    print(f"            if magnitude.max() > clip_threshold:")
    print(f"                scale_factor = clip_threshold / magnitude.max()")
    print(f"                kspace_data = kspace_data * scale_factor")
    print(f"    ")
    print(f"    return kspace_data")
    print(f"```")

    # 训练时的梯度处理建议
    print(f"\n 训练时的梯度处理建议:")
    print(f"```python")
    print(f"# 在训练循环中添加梯度裁剪")
    print(f"torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
    print(f"")
    print(f"# 或者使用梯度裁剪值")
    print(f"torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)")
    print(f"")
    print(f"# 检查梯度是否正常")
    print(f"for name, param in model.named_parameters():")
    print(f"    if param.grad is not None:")
    print(f"        grad_norm = param.grad.norm()")
    print(f"        if torch.isnan(grad_norm) or torch.isinf(grad_norm):")
    print(f"            print(f'Warning: abnormal gradient in {{name}}: {{grad_norm}}')")
    print(f"```")


if __name__ == "__main__":
    main()
