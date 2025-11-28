import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
import random

# 导入你的模型
from vgg16 import CIFAR10_VGG16, select_device

# ================= 配置 =================
CONFIG = {
    'batch_size': 100,  # 测试用的批次大小
    'noise_levels': [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],  # 模拟不同的噪声强度 (误码率)
    'model_path': './checkpoints/vgg16_cifar10_best.pth'
}

# ================= 定义层级策略 =================
# 基于之前的 L2 分析和论文 "Are All Layers Created Equal?" 的结论

# 1. 关键层 (Critical Layers) -> 走 TCP (不加噪声)
# 包括：输入层(features.0)，所有全连接层(dense)，输出层(classifier)，以及深层高范数卷积(features.31+)
CRITICAL_LAYERS = [
    'features.0.weight',  # 第一层：基础特征 (L2虽小但位置关键)
    'features.31.weight', 'features.35.weight', 'features.38.weight', 'features.41.weight',  # 深层卷积
    'dense.0.weight', 'dense.3.weight', 'dense.6.weight',  # 全连接层 (高能量)
    'classifier.weight'  # 输出层 (决策关键)
]

# 2. 鲁棒层 (Robust Layers) -> 走 UDP (加噪声)
# 包括：中间的卷积层，它们通常有冗余且L2范数适中
ROBUST_LAYERS = [
    'features.4.weight', 'features.7.weight', 'features.8.weight',
    'features.10.weight', 'features.11.weight', 'features.14.weight', 'features.15.weight',
    'features.17.weight', 'features.18.weight', 'features.20.weight', 'features.21.weight',
    'features.24.weight', 'features.25.weight', 'features.27.weight', 'features.28.weight',
    'features.30.weight', 'features.34.weight', 'features.37.weight', 'features.40.weight'
]


# 辅助函数：获取所有带权重的层名
def get_all_weight_layers(model):
    return [name for name, _ in model.named_parameters() if 'weight' in name]


def get_test_loader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # 为了速度，只取测试集的一部分(例如1000张)进行快速验证，正式跑可以去掉indices
    indices = range(1000)
    subset = torch.utils.data.Subset(testset, indices)
    return DataLoader(subset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def inject_noise(model, target_layers, std):
    """
    向指定层(target_layers)注入高斯噪声，模拟 UDP 传输错误
    """
    noisy_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in noisy_model.named_parameters():
            if name in target_layers:
                # 生成噪声：均值为0，标准差为std
                noise = torch.randn_like(param) * std
                param.add_(noise)
    return noisy_model


def main():
    device = select_device('cuda')
    print(f"Using device: {device}")

    # 1. 加载基准模型
    print("加载模型...")
    original_model = CIFAR10_VGG16(num_classes=10).to(device)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    if 'model' in checkpoint:
        original_model.load_state_dict(checkpoint['model'])
    else:
        original_model.load_state_dict(checkpoint)

    testloader = get_test_loader()
    all_layers = get_all_weight_layers(original_model)

    # 结果容器
    results = {
        'noise': CONFIG['noise_levels'],
        'smart_split': [],  # 你的策略
        'random_split': []  # 对照组
    }

    print("\n开始对比实验：Smart Split (你的策略) vs Random Split (随机策略)")
    print(f"{'Noise Std':<10} | {'Smart Acc':<10} | {'Random Acc':<10} | {'Drop (Smart)':<12}")
    print("-" * 50)

    for std in CONFIG['noise_levels']:
        # --- 策略 A: Smart Split ---
        # 只干扰 Robust 层 (模拟它们走 UDP)，保护 Critical 层 (模拟走 TCP)
        model_smart = inject_noise(original_model, ROBUST_LAYERS, std)
        acc_smart = evaluate(model_smart, testloader, device)
        results['smart_split'].append(acc_smart)

        # --- 策略 B: Random Split (对照组) ---
        # 随机选择一半层作为 UDP 层进行干扰
        random.seed(42)  # 固定种子保证公平
        random_layers = random.sample(all_layers, len(all_layers) // 2)
        model_random = inject_noise(original_model, random_layers, std)
        acc_random = evaluate(model_random, testloader, device)
        results['random_split'].append(acc_random)

        # 打印进度
        base_acc = results['smart_split'][0]  # 0噪声时的准确率
        drop = base_acc - acc_smart
        print(f"{std:<10.2f} | {acc_smart:<10.2f} | {acc_random:<10.2f} | -{drop:.2f}%")

    # 2. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(results['noise'], results['smart_split'], 'r-o', label='Smart Split (Your Method)', linewidth=2)
    plt.plot(results['noise'], results['random_split'], 'b--s', label='Random Split (Baseline)', linewidth=2)

    plt.title('Impact of Transmission Noise: Smart vs Random Layer Split', fontsize=14)
    plt.xlabel('Noise Standard Deviation (Simulating UDP Error Rate)', fontsize=12)
    plt.ylabel('Model Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    save_path = './analysis_results/validation_comparison.png'
    plt.savefig(save_path)
    print(f"\n实验完成！对比图已保存至: {save_path}")
    print("结论：如果红色线在蓝色线之上，说明你的分层策略有效提升了鲁棒性。")


if __name__ == '__main__':
    main()