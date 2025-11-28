import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime

from vgg16 import CIFAR100_VGG16, select_device

# ================= 配置参数 =================
CONFIG = {
    'device': 'cuda',
    'epochs': 50,
    'batch_size': 64,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'save_path': './checkpoints_cifar100',
    'analysis_path': './analysis_results_cifar100',
    'log_file': 'layer_split_log_refined.txt',  # 新日志文件
    'num_workers': 2
}


def get_data_loaders(batch_size, num_workers):
    print("正在准备 CIFAR-100 数据...")
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1} Train', leave=False)
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total


def log_and_print(message, log_path):
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# ================= 改进的核心算法：3-Class Clustering & Filter BN =================

def kmeans_split_3_levels(values):
    """
    K-Means with k=3 (Low, Mid, High)
    返回两个阈值: low_mid_thresh, mid_high_thresh
    """
    data = np.array(values).reshape(-1, 1)
    if len(data) < 3: return data[0][0]  # 数据太少不聚类

    # 初始化三个中心: Min, Median, Max
    c1 = np.min(data)
    c2 = np.median(data)
    c3 = np.max(data)

    for _ in range(15):
        dist1 = np.abs(data - c1)
        dist2 = np.abs(data - c2)
        dist3 = np.abs(data - c3)

        # 归类
        labels = np.argmin(np.vstack((dist1.T, dist2.T, dist3.T)), axis=0)

        # 更新中心
        new_c1 = data[labels == 0].mean() if np.any(labels == 0) else c1
        new_c2 = data[labels == 1].mean() if np.any(labels == 1) else c2
        new_c3 = data[labels == 2].mean() if np.any(labels == 2) else c3

        if c1 == new_c1 and c2 == new_c2 and c3 == new_c3:
            break
        c1, c2, c3 = new_c1, new_c2, new_c3

    # 确保 c1 < c2 < c3
    centers = sorted([c1, c2, c3])

    # 我们只关心 High 组的分界线 (High Threshold)
    # 取 Mid 和 High 的中间点作为关键层的门槛
    critical_threshold = (centers[1] + centers[2]) / 2
    return critical_threshold


def classify_layers_refined(model):
    """
    改进版分类逻辑：
    1. 排除 1D 参数 (Batch Norm)，只分析 Conv 和 Dense
    2. 使用 k=3 聚类，只有 Top Cluster 被判定为 Critical
    """
    layer_scores = {}
    score_values = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 【关键修改】 剔除 BN 层 (ndim=1)
            if len(param.shape) <= 1:
                continue

            l2_val = param.norm(p=2).item()
            num_params = param.numel()
            # 计算能量密度 RMS
            rms_val = (l2_val / np.sqrt(num_params)) * 100

            layer_scores[name] = rms_val
            score_values.append(rms_val)

    # 计算高阶阈值 (筛选真正的 Top Tier)
    threshold = kmeans_split_3_levels(score_values)

    critical = []
    robust = []

    for name, val in layer_scores.items():
        if val >= threshold:
            critical.append(f"{name} ({val:.2f})")
        else:
            robust.append(f"{name} ({val:.2f})")

    return critical, robust, threshold


def record_layer_metrics(model, epoch, history_list):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # 同样只记录主要层
            if len(param.shape) <= 1: continue

            l2_val = param.norm(p=2).item()
            num_params = param.numel()
            rms_val = (l2_val / np.sqrt(num_params)) * 100

            history_list.append({
                'epoch': epoch + 1,
                'layer': name,
                'l2_norm': l2_val,
                'rms_score': rms_val
            })


def save_and_plot_analysis(history_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    csv_path = os.path.join(save_dir, 'training_metrics_refined.csv')
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 8))
    layers = df['layer'].unique()
    for layer_name in layers:
        layer_data = df[df['layer'] == layer_name]
        short_name = layer_name.replace('features.', 'F').replace('dense.', 'D').replace('classifier.', 'C')
        plt.plot(layer_data['epoch'], layer_data['rms_score'], label=short_name, marker='o', markersize=3)
    plt.title('Layer RMS Score Evolution (Conv & Dense Only)')
    plt.xlabel('Epoch')
    plt.ylabel('RMS Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rms_refined_plot.png'), dpi=300)


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log (Refined) - {datetime.datetime.now()}\n")
        f.write("Strategy: RMS Score + BN Filtering + K-Means(k=3)\n")
        f.write("Target: Filter out small BN layers, select only Top-Tier Conv/Dense.\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR100_VGG16(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = []

    log_and_print(f"开始训练 (Refined 策略)...", log_path)

    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        record_layer_metrics(model, epoch, history)
        critical_layers, robust_layers, thresh = classify_layers_refined(model)

        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"=" * 20 + " 动态分层 (Refined) " + "=" * 20)
        msg.append(f"Top-Tier 阈值: {thresh:.4f}")
        msg.append(f"🔴 关键层 (Critical/TCP, Count={len(critical_layers)}):")
        msg.append(", ".join([x.split(' ')[0] for x in critical_layers]))
        msg.append(f"🟢 鲁棒层 (Robust/UDP, Count={len(robust_layers)}):")
        # 显示一部分鲁棒层
        if len(robust_layers) > 0:
            msg.append(", ".join([x.split(' ')[0] for x in robust_layers]))
        else:
            msg.append("None")
        msg.append("=" * 60)

        log_and_print("\n".join(msg), log_path)
        scheduler.step()

    total_time = time.time() - start_time
    log_and_print(f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。", log_path)
    save_and_plot_analysis(history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()