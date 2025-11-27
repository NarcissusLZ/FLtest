import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os

# 导入你的模型定义
from vgg16 import CIFAR10_VGG16, select_device


def analyze_model_l2(model_path='./checkpoints/vgg16_cifar10_best.pth', save_dir='./analysis_results'):
    # 1. 设置环境
    device = select_device('cpu')  # 分析不需要GPU，CPU即可
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 加载模型
    print(f"正在加载模型: {model_path}")
    model = CIFAR10_VGG16(num_classes=10)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容处理 state_dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("错误：找不到模型文件，将使用随机初始化的模型演示。")

    model.eval()

    # 3. 提取每一层的 L2 范数
    print("正在计算每层的 L2 范数...")
    layer_data = []

    for name, param in model.named_parameters():
        # 我们通常只关心 'weight'，不关心 'bias'，因为权重矩阵决定了层的主要特征
        if 'weight' in name:
            # 对于 BatchNorm 层，weight 对应 gamma，也可以看作一种缩放系数
            # 如果只想看卷积和全连接，可以加判断 if isinstance(...)

            l2_val = param.norm(p=2).item()  # 计算 L2 范数

            # 记录数据
            layer_data.append({
                'Layer Name': name,
                'L2 Norm': l2_val,
                'Shape': str(list(param.shape))  # 顺便记录一下形状
            })

    # 4. 保存数据到 CSV
    df = pd.DataFrame(layer_data)
    csv_path = os.path.join(save_dir, 'layer_l2_norms.csv')
    df.to_csv(csv_path, index=False)
    print(f"数据已保存到: {csv_path}")

    # 5. 可视化绘图
    plt.figure(figsize=(15, 8))  # 设置画布大小

    # 绘制柱状图
    bars = plt.bar(df['Layer Name'], df['L2 Norm'], color='skyblue', edgecolor='navy')

    # 美化图表
    plt.title('L2 Norm of Weights per Layer', fontsize=16)
    plt.xlabel('Layer Name', fontsize=12)
    plt.ylabel('L2 Norm Value', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)  # x轴标签旋转90度以免重叠
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=6, rotation=90)

    # 保存图片
    plot_path = os.path.join(save_dir, 'l2_norm_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"图表已保存到: {plot_path}")

    # 显示一下
    # plt.show()


if __name__ == '__main__':
    analyze_model_l2()