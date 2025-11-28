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

# 导入 CIFAR-100 模型
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
    'log_file': 'layer_split_log_rms.txt',  # 修改日志名
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


# ================= 核心逻辑：基于RMS的动态分层 =================
def simple_kmeans_split(values):
    """简单的1D K-Means"""
    data = np.array(values).reshape(-1, 1)
    # 防止只有1个值报错
    if len(data) < 2: return data[0][0]

    c1 = np.min(data)
    c2 = np.max(data)
    for _ in range(10):
        dist1 = np.abs(data - c1)
        dist2 = np.abs(data - c2)
        group1 = data[dist1 <= dist2]
        group2 = data[dist1 > dist2]

        new_c1 = group1.mean() if len(group1) > 0 else c1
        new_c2 = group2.mean() if len(group2) > 0 else c2

        if c1 == new_c1 and c2 == new_c2: break
        c1, c2 = new_c1, new_c2

    threshold = (c1 + c2) / 2
    return threshold


def classify_layers_realtime(model):
    """
    修改点：使用 RMS (L2 / sqrt(N)) 代替纯 L2 sum
    这样可以消除全连接层参数过多带来的数值偏差
    """
    layer_scores = {}
    score_values = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            # 1. 获取 L2 范数
            l2_val = param.norm(p=2).item()
            # 2. 获取参数数量
            num_params = param.numel()
            # 3. 计算 RMS (能量密度)
            rms_val = l2_val / np.sqrt(num_params)

            # 放大一点数值方便阅读 (可选)
            score = rms_val * 100

            layer_scores[name] = score
            score_values.append(score)

    # 计算阈值
    threshold = simple_kmeans_split(score_values)

    critical = []
    robust = []

    for name, val in layer_scores.items():
        if val >= threshold:
            critical.append(f"{name} (Score:{val:.2f})")
        else:
            robust.append(f"{name} (Score:{val:.2f})")

    return critical, robust, threshold


def record_layer_metrics(model, epoch, history_list):
    """同时记录L2和RMS供分析"""
    for name, param in model.named_parameters():
        if 'weight' in name:
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
    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    df.to_csv(csv_path, index=False)

    # 绘制 RMS 趋势图
    plt.figure(figsize=(12, 8))
    layers = df['layer'].unique()
    for layer_name in layers:
        layer_data = df[df['layer'] == layer_name]
        short_name = layer_name.replace('features.', 'F').replace('dense.', 'D').replace('classifier.', 'C')
        plt.plot(layer_data['epoch'], layer_data['rms_score'], label=short_name, marker='o', markersize=3)
    plt.title('Layer Energy Density (RMS) Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('RMS Score (L2 / sqrt(N))')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rms_evolution_plot.png'), dpi=300)


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log (CIFAR-100) - RMS Strategy - {datetime.datetime.now()}\n")
        f.write("Strategy: Real-time Dynamic Split based on Weight RMS (Energy Density)\n")
        f.write("Reason: To prevent Dense layers from dominating L2 norms due to size.\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR100_VGG16(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = []

    log_and_print(f"开始训练 (RMS 策略)...", log_path)

    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # 记录数据
        record_layer_metrics(model, epoch, history)

        # 实时分类 (使用RMS)
        critical_layers, robust_layers, thresh = classify_layers_realtime(model)

        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"=" * 20 + " 动态分层 (RMS Score) " + "=" * 20)
        msg.append(f"当前阈值 (RMS): {thresh:.4f}")
        msg.append(f"🔴 关键层 (Critical/TCP, Count={len(critical_layers)}):")
        # 此时你应该能看到卷积层回归了
        msg.append(", ".join([x.split(' ')[0] for x in critical_layers]))
        msg.append(f"🟢 鲁棒层 (Robust/UDP, Count={len(robust_layers)}):")
        msg.append(f"Includes {len(robust_layers)} layers with Score < {thresh:.4f}")
        msg.append("=" * 60)

        log_and_print("\n".join(msg), log_path)
        scheduler.step()

    total_time = time.time() - start_time
    log_and_print(f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。", log_path)
    save_and_plot_analysis(history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()