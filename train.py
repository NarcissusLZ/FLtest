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
import copy

# 导入你的模型文件
# 确保 vgg16.py 在同目录下
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
    'log_file': 'dual_factor_log.txt',  # 日志文件名
    'num_workers': 2,

    # === 核心算法参数 (Innovation Points) ===
    'critical_ratio': 0.5,  # 关键层比例 (Top 50%)

    # 梯度权重 (Gradient Importance Beta)
    # Score = Norm(Movement_Step) + beta * Norm(Gradient)
    # Movement_Step = W_t - W_{t-1} (当前轮的变化幅度)
    'grad_beta': 1.0
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
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    # 注意：函数结束时，模型参数 param.grad 中保留了最后一个 Batch 的梯度
    # 这正是我们用来计算“敏感度”的最佳时机
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 验证阶段不计算梯度
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total


# ================= 辅助类：双因子指标计算器 =================
class LayerMetricCalculator:
    """
    修改后逻辑：
    1. Movement: 权重相对于上一轮的变化 (W_t - W_{t-1})
    2. Gradient: 当前梯度的范数 (Loss Sensitivity)
    """

    def __init__(self, model):
        # 初始化时，prev_weights 就是初始权重 W_0
        print("初始化 MetricCalculator: 正在备份上一轮权重(W_t-1)...")
        self.prev_weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # 存到CPU节省显存
                self.prev_weights[name] = param.data.clone().detach().cpu()

    def update_prev_weights(self, model):
        """
        每轮结束后调用，将当前权重更新为“上一轮权重”，供下一轮计算差值使用
        """
        for name, param in model.named_parameters():
            if name in self.prev_weights:
                self.prev_weights[name] = param.data.clone().detach().cpu()

    def get_dual_metrics(self, model):
        """
        获取双因子原始数据
        """
        metrics_data = []

        for name, param in model.named_parameters():
            # 只处理卷积层和全连接层的权重
            if 'weight' not in name or param.dim() <= 1:
                continue

            # --- 因子1: Movement (W_t - W_{t-1}) ---
            # 修改：计算当前权重与 prev_weights 的差值
            movement = 0.0
            if name in self.prev_weights:
                prev_w = self.prev_weights[name].to(param.device)
                movement = torch.norm(param.data - prev_w, p=2).item()

            # --- 因子2: Gradient Norm (Sensitivity) ---
            grad_val = 0.0
            if param.grad is not None:
                grad_val = param.grad.norm(p=2).item()

            metrics_data.append({
                'name': name,
                'movement': movement,
                'grad': grad_val
            })

        return metrics_data


# ================= 核心逻辑：双因子融合分层算法 =================
def classify_layers_dual_factor(model, metric_calculator, ratio, grad_beta):
    """
    基于 [Weight Increment] 和 [Gradient Sensitivity] 的融合分层。
    Score = Norm(W_t - W_{t-1}) + beta * Norm(Gradient)
    """
    # 1. 获取原始数据
    raw_data = metric_calculator.get_dual_metrics(model)
    if not raw_data: return [], [], 0.0

    # 2. 准备归一化
    movements = [x['movement'] for x in raw_data]
    grads = [x['grad'] for x in raw_data]

    max_mov = max(movements) if movements and max(movements) > 0 else 1.0
    max_grad = max(grads) if grads and max(grads) > 0 else 1.0

    final_scores = []

    for item in raw_data:
        # Min-Max Normalization (Min假设为0，简化计算)
        norm_mov = item['movement'] / max_mov
        norm_grad = item['grad'] / max_grad

        # === 核心公式 ===
        combined_score = norm_mov + (grad_beta * norm_grad)

        final_scores.append({
            'name': item['name'],
            'score': combined_score,
            'raw_mov': item['movement'],
            'raw_grad': item['grad']
        })

    # 3. 排序 (Score 越大越 Critical)
    final_scores.sort(key=lambda x: x['score'], reverse=True)

    # 4. 切分 Top-K
    num_critical = int(len(final_scores) * ratio)
    if num_critical == 0 and ratio > 0: num_critical = 1  # 至少保留一层

    critical_list = final_scores[:num_critical]
    robust_list = final_scores[num_critical:]

    # 5. 格式化输出 (Name | Score | Movement | Gradient)
    critical_desc = [f"{x['name']} (S:{x['score']:.2f}|Mov:{x['raw_mov']:.4f}|G:{x['raw_grad']:.2f})" for x in
                     critical_list]
    robust_desc = [f"{x['name']} (S:{x['score']:.2f}|Mov:{x['raw_mov']:.4f}|G:{x['raw_grad']:.2f})" for x in
                   robust_list]

    threshold = critical_list[-1]['score'] if critical_list else 0.0

    return critical_desc, robust_desc, threshold


def log_and_print(message, log_path):
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def save_and_plot_analysis(history_list, save_dir):
    """简单的绘图函数，记录综合得分的变化"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame(history_list)
    csv_path = os.path.join(save_dir, 'layer_scores_history.csv')
    df.to_csv(csv_path, index=False)


# ================= 主函数 =================
def main():
    device = select_device(CONFIG['device'])
    print(f"Using device: {device}")

    if not os.path.exists(CONFIG['save_path']): os.makedirs(CONFIG['save_path'])
    if not os.path.exists(CONFIG['analysis_path']): os.makedirs(CONFIG['analysis_path'])
    log_path = os.path.join(CONFIG['analysis_path'], CONFIG['log_file'])

    # 1. 模型初始化
    trainloader, testloader = get_data_loaders(CONFIG['batch_size'], CONFIG['num_workers'])
    model = CIFAR10_VGG16(num_classes=10).to(device)

    # 2. 【关键步骤】 初始化指标计算器 (保存 W_0 作为第一轮的 W_{t-1})
    metric_calc = LayerMetricCalculator(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 记录日志头
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log - {datetime.datetime.now()}\n")
        f.write(f"Strategy: Dual-Factor Metric (Incremental Movement + Beta*Gradient)\n")
        f.write(f"Params: Ratio={CONFIG['critical_ratio']}, Beta={CONFIG['grad_beta']}\n")
        f.write("=" * 60 + "\n")

    score_history = []
    start_time = time.time()

    log_and_print("开始训练...", log_path)

    for epoch in range(CONFIG['epochs']):
        # --- 训练 ---
        # 这里的 train_one_epoch 会保留最后一个 batch 的梯度
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch)

        # --- 验证 ---
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        # --- 【核心】执行双因子分层算法 ---
        # 这里的计算基于：W_t (当前) - W_{t-1} (MetricCalc中保存的)
        critical_layers, robust_layers, thresh = classify_layers_dual_factor(
            model,
            metric_calc,
            CONFIG['critical_ratio'],
            CONFIG['grad_beta']
        )

        # --- 【重要修改】更新 W_{t-1} ---
        # 计算完本轮的分层后，立即将当前权重保存为 "上一轮权重"，供下一轮使用
        metric_calc.update_prev_weights(model)

        # --- 构建日志 ---
        msg = []
        msg.append(f"\n[{epoch + 1}/{CONFIG['epochs']}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        msg.append(f"=" * 10 + f" 双因子分层 (W_t-W_{{t-1}} + {CONFIG['grad_beta']}*Grad) " + "=" * 10)
        msg.append(f"当前分界 Score: {thresh:.4f}")

        msg.append(f"🔴 关键层 (Critical/TCP, Count={len(critical_layers)}):")
        # 打印详细信息: Name (Score|Mov|Grad)
        msg.append("\n".join(critical_layers))

        msg.append(f"\n🟢 鲁棒层 (Robust/UDP, Count={len(robust_layers)}):")
        # 鲁棒层只打印名字简化显示
        msg.append(", ".join([x.split(' ')[0] for x in robust_layers]))

        msg.append("=" * 60)
        log_and_print("\n".join(msg), log_path)

        scheduler.step()

    total_time = time.time() - start_time
    log_and_print(f"\n训练结束，耗时 {total_time / 60:.2f} 分钟。", log_path)

    # 保存简单数据
    save_and_plot_analysis(score_history, CONFIG['analysis_path'])


if __name__ == '__main__':
    main()