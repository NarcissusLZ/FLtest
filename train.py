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
    'log_file': 'layer_split_log_fix.txt',
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


def record_layer_metrics(model, epoch, history_list):
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            l2_val = param.norm(p=2).item()
            history_list.append({
                'epoch': epoch + 1,
                'layer': name,
                'l2_norm': l2_val
            })


def save_and_plot_analysis(history_list, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    df.to_csv(os.path.join(save_dir, 'training_metrics_fix.csv'), index=False)
    # 简单的 L2 趋势图
    plt.figure(figsize=(12, 8))
    for layer_name in df['layer'].unique():
        layer_data = df[df['layer'] == layer_name]
        short_name = layer_name.replace('features.', 'F').replace('dense.', 'D')
        plt.plot(layer_data['epoch'], layer_data['l2_norm'], label=short_name, marker='o', markersize=3)
    plt.title('Layer L2 Norm Evolution')
    plt.savefig(os.path.join(save_dir, 'l2_fix_plot.png'), dpi=300)


# ================= 【核心修改】 L2 + 结构权重 + 对数聚类 =================

def kmeans_split_log_space(values):
    """
    在对数空间进行 K-Means (k=2)，能更好地处理数量级差异
    比如: [6, 12, 20] -> Log: [1.8, 2.5, 3.0]
    Threshold ~ 2.4 (Log) -> ~11 (Linear)
    这样 12 和 20 都会被划分为 High，而 6 是 Low。
    """
    data = np.array(values)
    # 避免 log(0)
    data_log = np.log(data + 1e-6).reshape(-1, 1)

    if len(data) < 2: return np.exp(data_log[0][0])

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


def classify_layers_fix(model):
    layer_scores = {}
    score_values = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 1. 过滤 BN 层
            if len(param.shape) <= 1: continue

            # 2. 回归单纯的 L2 范数 (Dense 层会天然很高)
            l2_val = param.norm(p=2).item()

            # 3. 结构性加权 (Structural Weighting)
            # 这里的目的是让 First Layer 也能达到 Dense Layer 的数量级
            weighted_l2 = l2_val

            if "features.0" in name:
                weighted_l2 *= 4.0  # 第一层 L2通常~6, x4后~24 (媲美Dense)
            elif "classifier" in name:
                weighted_l2 *= 2.0  # 分类头
            # 深层卷积 (Deep Conv) 和 全连接 (Dense) 不需要加权
            # 因为 Deep Conv L2 通常 ~12，Dense L2 通常 ~20
            # 它们自然比 Shallow Conv (~6) 高，会被对数聚类自动分到 High 组

            layer_scores[name] = weighted_l2
            score_values.append(weighted_l2)

    # 4. 对数空间动态划分
    threshold = kmeans_split_log_space(score_values)

    critical = []
    robust = []

    for name, val in layer_scores.items():
        if val >= threshold:
            critical.append(f"{name}")
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
        f.write(f"Training Log (Fix) - {datetime.datetime.now()}\n")
        f.write("Strategy: L2 Norm (Base) + Log-Space Clustering + First Layer Boost\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR100_VGG16(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    history = []

    log_and_print(f"开始训练 (Fix 策略)...", log_path)
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        record_layer_metrics(model, epoch, history)
        critical_layers, robust_layers, thresh = classify_layers_fix(model)

        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"Weighted L2 Threshold: {thresh:.2f}")
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