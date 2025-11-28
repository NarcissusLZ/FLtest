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

# 导入你的模型文件
from vgg16 import CIFAR10_VGG16, select_device

# ================= 配置参数 =================
CONFIG = {
    'device': 'cuda',  # 'cuda', 'mps' (Mac), or 'cpu'
    'epochs': 50,  # 训练轮数
    'batch_size': 64,  # 批次大小
    'lr': 0.01,  # 初始学习率
    'momentum': 0.9,  # SGD 动量
    'weight_decay': 5e-4,  # 权重衰减 (L2正则化)
    'save_path': './checkpoints',  # 模型保存路径
    'analysis_path': './analysis_results',  # 分析结果保存路径
    'log_file': 'layer_split_log.txt',  # 日志文件
    'num_workers': 2  # 数据加载线程数
}


# ================= 1. 数据准备 =================
def get_data_loaders(batch_size, num_workers):
    print("正在准备 CIFAR-10 数据...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


# ================= 2. 训练与评估函数 =================
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
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


# ================= 3. 核心功能：L2范数分层逻辑 =================

def log_and_print(message, log_path):
    """同时输出到控制台和文件"""
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def kmeans_split_log_space(values):
    """
    对数空间 K-Means 聚类。
    解决 VGG 中全连接层 L2 (20+) 与卷积层 L2 (5~12) 数量级差异大的问题。
    """
    data = np.array(values)
    # 转到对数空间: log(x)
    data_log = np.log(data + 1e-6).reshape(-1, 1)

    if len(data) < 2: return np.exp(data_log[0][0])

    # K-Means (k=2)
    c1, c2 = np.min(data_log), np.max(data_log)
    for _ in range(10):
        dist1 = np.abs(data_log - c1)
        dist2 = np.abs(data_log - c2)
        group1 = data_log[dist1 <= dist2]
        group2 = data_log[dist1 > dist2]
        new_c1 = group1.mean() if len(group1) > 0 else c1
        new_c2 = group2.mean() if len(group2) > 0 else c2
        if c1 == new_c1 and c2 == new_c2: break
        c1, c2 = new_c1, new_c2

    thresh_log = (c1 + c2) / 2
    return np.exp(thresh_log)  # 还原回线性空间


def classify_layers(model):
    """
    计算每一层的 L2 范数，并根据分布动态划分为 Critical 和 Robust。
    包含特定层的结构性加权（第一层和分类头）。
    """
    layer_scores = {}
    score_values = []

    for name, param in model.named_parameters():
        # 只分析权重，且忽略 BatchNorm 层 (shape 为 1 的是一维参数)
        if 'weight' in name and len(param.shape) > 1:

            # 1. 基础指标：L2 范数
            l2_val = param.norm(p=2).item()

            # 2. 结构性加权 (修正纯统计的偏差)
            weighted_l2 = l2_val

            if "features.0" in name:
                # 第一层卷积极其重要，但参数少L2小，给予高权重
                weighted_l2 *= 4.0
            elif "classifier" in name or "dense.6" in name:
                # 输出层直接决定结果，给予加权
                weighted_l2 *= 2.0

            layer_scores[name] = weighted_l2
            score_values.append(weighted_l2)

    # 3. 计算动态阈值
    threshold = kmeans_split_log_space(score_values)

    critical = []
    robust = []

    for name, val in layer_scores.items():
        if val >= threshold:
            critical.append(name)
        else:
            robust.append(name)

    return critical, robust, threshold


def record_metrics(model, epoch, history_list):
    """记录原始数据用于绘图"""
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            l2_val = param.norm(p=2).item()
            history_list.append({
                'epoch': epoch + 1,
                'layer': name,
                'l2_norm': l2_val
            })


def save_analysis(history_list, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    df.to_csv(os.path.join(save_dir, 'l2_history.csv'), index=False)
    print(f"L2范数历史数据已保存至: {save_dir}")


# ================= 4. 主流程 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    # 初始化日志
    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log - {datetime.datetime.now()}\n")
        f.write("Dataset: CIFAR-10\n")
        f.write("Strategy: L2 Norm with Log-Space Clustering & Structural Weights\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR10_VGG16(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = []
    best_acc = 0.0

    print(f"开始训练，日志将写入 {log_path} ...")
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        # 训练与验证
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # 记录数据
        record_metrics(model, epoch, history)

        # 【核心调用】计算分层
        critical_layers, robust_layers, thresh = classify_layers(model)

        # 构建日志信息
        msg = []
        msg.append(f"\n[Epoch {epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"分层阈值 (Weighted L2): {thresh:.2f}")
        msg.append(f"🔴 关键层 (TCP): {', '.join(critical_layers)}")
        msg.append(f"🟢 鲁棒层 (UDP): {', '.join(robust_layers)}")
        msg.append("-" * 60)

        # 输出与保存
        log_and_print("\n".join(msg), log_path)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['save_path'], 'best_model.pth'))

        scheduler.step()

    total_time = time.time() - start_time
    log_and_print(f"\n训练结束，总耗时 {total_time / 60:.2f} 分钟。最佳准确率: {best_acc:.2f}%", log_path)
    save_analysis(history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()