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
    'device': 'cuda',
    'epochs': 50,
    'batch_size': 64,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'save_path': './checkpoints',
    'analysis_path': './analysis_results',
    'log_file': 'layer_split_log.txt',  # 新增：日志文件名
    'num_workers': 2
}


def get_data_loaders(batch_size, num_workers):
    """准备 CIFAR-10 数据集"""
    print("正在准备数据...")
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


# ================= 辅助功能：双重日志记录 =================
def log_and_print(message, log_path):
    """
    既打印到控制台，也追加写入文件
    """
    print(message)  # 控制台输出
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')  # 文件写入


# ================= 核心逻辑：动态分层算法 =================
def simple_kmeans_split(values):
    """简单的1D K-Means (k=2) 实现"""
    data = np.array(values).reshape(-1, 1)
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
    """获取当前L2并分类"""
    layer_l2 = {}
    l2_values = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            val = param.norm(p=2).item()
            layer_l2[name] = val
            l2_values.append(val)

    threshold = simple_kmeans_split(l2_values)

    critical = []
    robust = []

    for name, val in layer_l2.items():
        if val >= threshold:
            critical.append(f"{name} ({val:.2f})")
        else:
            robust.append(f"{name} ({val:.2f})")

    return critical, robust, threshold


def record_layer_l2_norms(model, epoch, history_list):
    """记录数据用于事后绘图"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_val = param.norm(p=2).item()
            history_list.append({
                'epoch': epoch + 1,
                'layer': name,
                'l2_norm': l2_val
            })


def save_and_plot_analysis(history_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    csv_path = os.path.join(save_dir, 'training_l2_history.csv')
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 8))
    layers = df['layer'].unique()
    for layer_name in layers:
        layer_data = df[df['layer'] == layer_name]
        short_name = layer_name.replace('features.', 'F').replace('dense.', 'D').replace('classifier.', 'C')
        plt.plot(layer_data['epoch'], layer_data['l2_norm'], label=short_name, marker='o', markersize=3)
    plt.title('Layer L2 Norm Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'l2_evolution_plot.png'), dpi=300)


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    # 初始化日志文件路径
    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])
    # 清空之前的日志（如果需要保留追加，去掉这行 'w' 模式的写入）
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log - Started at {datetime.datetime.now()}\n")
        f.write("Strategy: Real-time Dynamic Split based on Pure L2 Norm\n")
        f.write("=" * 60 + "\n")

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR10_VGG16(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    l2_history = []

    log_and_print(f"开始训练 {CONFIG['epochs']} 轮...", log_path)
    log_and_print(f"日志文件位置: {log_path}", log_path)

    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        # 1. 训练
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)

        # 2. 验证
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # 3. 记录历史数据
        record_layer_l2_norms(model, epoch, l2_history)

        # 4. 【实时输出】 计算并打印本轮的分层结果
        critical_layers, robust_layers, thresh = classify_layers_realtime(model)

        # 构建要打印和保存的日志信息
        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"=" * 20 + " 动态分层 (Only L2) " + "=" * 20)
        msg.append(f"当前轮次 L2 阈值 (Threshold): {thresh:.4f}")

        msg.append(f"🔴 关键层 (Critical/TCP, Count={len(critical_layers)}):")
        # 记录所有关键层名字
        msg.append(", ".join([x.split(' ')[0] for x in critical_layers]))

        msg.append(f"🟢 鲁棒层 (Robust/UDP, Count={len(robust_layers)}):")
        # 鲁棒层通常较多，如果不希望日志太长，可以只记名字
        msg.append(", ".join([x.split(' ')[0] for x in robust_layers]))

        msg.append("=" * 60)

        # 将上面构建的所有信息一次性输出到控制台和文件
        log_and_print("\n".join(msg), log_path)

        scheduler.step()

    total_time = time.time() - start_time
    final_msg = f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。"
    log_and_print(final_msg, log_path)

    save_and_plot_analysis(l2_history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()