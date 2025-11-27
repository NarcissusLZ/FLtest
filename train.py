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
    'analysis_path': './analysis_results',  # 新增：分析结果保存路径
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


# ================= 新增：记录L2范数的函数 =================
def record_layer_l2_norms(model, epoch, history_list):
    """
    计算当前模型所有层的L2范数，并追加到 history_list 中
    """
    for name, param in model.named_parameters():
        # 只记录权重(weight)，不记录偏置(bias)，因为权重更能代表层的重要性
        if 'weight' in name:
            l2_val = param.norm(p=2).item()
            history_list.append({
                'epoch': epoch + 1,
                'layer': name,
                'l2_norm': l2_val
            })


def save_and_plot_analysis(history_list, save_dir):
    """
    保存CSV并绘制折线图
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 保存 CSV
    df = pd.DataFrame(history_list)
    csv_path = os.path.join(save_dir, 'training_l2_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[Analysis] 详细数据已保存至: {csv_path}")

    # 2. 绘制折线图
    plt.figure(figsize=(12, 8))

    # 获取所有唯一的层名称
    layers = df['layer'].unique()

    # 为每一层画一条线
    for layer_name in layers:
        layer_data = df[df['layer'] == layer_name]
        # 简化图例名称，去掉 'features.' 等前缀让图更清晰
        short_name = layer_name.replace('features.', 'F').replace('dense.', 'D').replace('classifier.', 'C')
        plt.plot(layer_data['epoch'], layer_data['l2_norm'], label=short_name, marker='o', markersize=3)

    plt.title('Layer L2 Norm Evolution During Training')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # 图例放在外侧防止遮挡
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'l2_evolution_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"[Analysis] 趋势图已保存至: {plot_path}")


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])

    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR10_VGG16(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 用于存储每一轮的L2数据
    l2_history = []

    print(f"开始训练 {CONFIG['epochs']} 轮，并将记录每一轮的参数变化...")
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        # 1. 训练
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)

        # 2. 验证
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # 3. 【核心步骤】记录本轮的 L2 范数
        record_layer_l2_norms(model, epoch, l2_history)

        scheduler.step()

        print(f"Epoch {epoch + 1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

        # 可选：保存最佳模型 (省略以保持代码简洁，可复用之前的逻辑)

    total_time = time.time() - start_time
    print(f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。正在生成分析报告...")

    # 4. 训练结束后，生成文件和图片
    save_and_plot_analysis(l2_history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()