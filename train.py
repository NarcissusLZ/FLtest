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
    'log_file': 'layer_split_log_final.txt',
    'num_workers': 2
}


# ... (数据加载和训练函数保持不变，省略以节省篇幅，直接使用之前的即可) ...
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


def record_layer_metrics(model, epoch, history_list):
    # 此函数仅用于记录原始数据，不做加权，保证 CSV 数据的纯净性
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
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
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    df.to_csv(os.path.join(save_dir, 'training_metrics_final.csv'), index=False)
    # (绘图代码略，与之前相同)


# ================= 【核心修改】 混合加权评分策略 =================

def kmeans_split_auto(values):
    """自动 K-Means (k=2)"""
    data = np.array(values).reshape(-1, 1)
    if len(data) < 2: return data[0][0]

    # 简单的 k=2 聚类
    c1, c2 = np.min(data), np.max(data)
    for _ in range(10):
        dist1 = np.abs(data - c1)
        dist2 = np.abs(data - c2)
        group1 = data[dist1 <= dist2]
        group2 = data[dist1 > dist2]
        new_c1 = group1.mean() if len(group1) > 0 else c1
        new_c2 = group2.mean() if len(group2) > 0 else c2
        if c1 == new_c1 and c2 == new_c2: break
        c1, c2 = new_c1, new_c2

    return (c1 + c2) / 2


def classify_layers_final(model):
    layer_scores = {}
    score_values = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 1. 过滤掉 BN 层 (一维参数)
            if len(param.shape) <= 1: continue

            # 2. 计算基础 RMS 分数 (能量密度)
            l2_val = param.norm(p=2).item()
            num_params = param.numel()
            base_score = (l2_val / np.sqrt(num_params)) * 100

            # 3. 应用位置加权 (Positional Weighting)
            # 这是为了弥补纯统计指标的不足，符合 "First & Last layers matter most" 原则
            weighted_score = base_score

            if "dense" in name or "classifier" in name:
                weighted_score *= 2.0  # 全连接层大幅加分
            elif "features.3" in name or "features.4" in name:
                # 深层卷积 (根据VGG结构，30层往后算深层)
                # 这里简单通过序号判断，VGG16 features结构比较长
                # 我们可以解析名字中的数字，数字大的加分
                try:
                    layer_idx = int(name.split('.')[1])
                    if layer_idx > 20:  # 假设20层以后的卷积更重要
                        weighted_score *= 1.3
                except:
                    pass
            elif "features.0" in name:
                weighted_score *= 1.5  # 第一层极其重要，加分

            layer_scores[name] = weighted_score
            score_values.append(weighted_score)

    # 4. 动态阈值划分
    threshold = kmeans_split_auto(score_values)

    critical = []
    robust = []

    for name, val in layer_scores.items():
        if val >= threshold:
            critical.append(f"{name}")  # 简洁输出
        else:
            robust.append(f"{name}")

    return critical, robust, threshold


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log (Final) - {datetime.datetime.now()}\n")
        f.write("Strategy: RMS Score * Positional Weights (Heuristic)\n")
        f.write("Weights: Dense(x2.0), DeepConv(x1.3), FirstConv(x1.5)\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR100_VGG16(num_classes=100).to(device)  # 使用 CIFAR100 模型

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    history = []

    log_and_print(f"开始训练 (Final 策略)...", log_path)
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        record_layer_metrics(model, epoch, history)
        critical_layers, robust_layers, thresh = classify_layers_final(model)

        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"Threshold: {thresh:.2f}")
        msg.append(f"🔴 Critical (TCP): {', '.join(critical_layers)}")
        msg.append(f"🟢 Robust (UDP):  {', '.join(robust_layers)}")
        msg.append("-" * 60)

        log_and_print("\n".join(msg), log_path)
        scheduler.step()

    total_time = time.time() - start_time
    log_and_print(f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。", log_path)
    save_and_plot_analysis(history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()