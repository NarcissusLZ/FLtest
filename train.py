import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm  # 用于显示进度条

# 导入你的模型文件
from vgg16 import CIFAR10_VGG16, select_device

# ================= 配置参数 =================
CONFIG = {
    'device': 'cuda',  # 'cuda', 'mps' (Mac), or 'cpu'
    'epochs': 50,  # 训练轮数
    'batch_size': 64,  # 批次大小 (如果显存不够可以调小，比如 32)
    'lr': 0.01,  # 初始学习率
    'momentum': 0.9,  # SGD 动量
    'weight_decay': 5e-4,  # 权重衰减 (L2正则化)
    'save_path': './checkpoints',  # 模型保存路径
    'num_workers': 2  # 数据加载线程数
}


def get_data_loaders(batch_size, num_workers):
    """准备 CIFAR-10 数据集，包含数据增强"""
    print("正在准备数据...")

    # 训练集的数据增强：随机裁剪、翻转、标准化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 测试集仅做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载并加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 Epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1} Train')

    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条信息
        pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """在测试集上评估"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return running_loss / len(dataloader), acc


def main():
    # 1. 设置设备
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    # 2. 创建保存目录
    if not os.path.exists(CONFIG['save_path']):
        os.makedirs(CONFIG['save_path'])

    # 3. 加载数据
    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])

    # 4. 初始化模型
    print("正在初始化 VGG16 模型...")
    model = CIFAR10_VGG16(num_classes=10)
    model = model.to(device)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # VGG 通常使用 SGD + Momentum 训练效果较好
    optimizer = optim.SGD(model.parameters(),
                          lr=CONFIG['lr'],
                          momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])

    # 学习率调度器：每 15 个 epoch 学习率乘以 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 6. 开始训练循环
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        # 训练
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)

        # 验证
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | LR: {current_lr:.5f}")

        # 保存最佳模型
        if val_acc > best_acc:
            print(f"检测到性能提升 ({best_acc:.2f}% -> {val_acc:.2f}%)，正在保存模型...")
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(CONFIG['save_path'], 'vgg16_cifar10_best.pth'))
            best_acc = val_acc

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"训练完成！总耗时: {total_time / 60:.2f} 分钟。最佳准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()