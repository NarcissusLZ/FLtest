import torch, yaml
import torch.nn as nn


# 根据配置和硬件支持自动选择设备
def select_device(device_config):
    if device_config == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_config == 'mps' and torch.backends.mps.is_available():  # macOS Metal
        return torch.device('mps')
    else:
        return torch.device('cpu')


# 适配CIFAR-10的VGG配置：第一层96通道适合32x32输入
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class CIFAR10_VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_VGG16, self).__init__()

        self.features = self._make_layers(vgg)
        # 添加自适应池化确保输出为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Sequential(
            nn.Linear(vgg[-2], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # 使用传入的num_classes参数
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)  # 确保1x1输出
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

class CIFAR100_VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100_VGG16, self).__init__()

        self.features = self._make_layers(vgg)
        # 添加自适应池化确保输出为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # CIFAR-100优化的分类器：更强的正则化
        self.dense = nn.Sequential(
            nn.Linear(vgg[-2], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 增加dropout防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # 添加额外的层提升表征能力
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        # 100个类别的分类器
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = CIFAR10_VGG16()
    # 获取配置参数
    with open("../config.yaml", 'r') as f:
        conf = yaml.safe_load(f)

    device_config = conf.get('device').lower()
    device = select_device(device_config)

    model = model.to(device)  # 统一迁移到目标设备
    print(model)
    print(next(model.parameters()).device)

    # 计算并打印每层的参数量及百分比
    total_params = 0
    layer_params_dict = {}

    # 首先计算总参数量
    for name, param in model.named_parameters():
        layer_params = param.numel()
        layer_params_dict[name] = layer_params
        total_params += layer_params

    # 按层分组的参数量统计
    grouped_params = {}
    for name, param_count in layer_params_dict.items():
        # 提取主层名称（如features, dense, classifier）
        main_layer = name.split('.')[0]
        if main_layer not in grouped_params:
            grouped_params[main_layer] = 0
        grouped_params[main_layer] += param_count

    # 打印每层及其参数量和百分比
    print("\n==== 详细参数统计 ====")
    for name, param_count in layer_params_dict.items():
        percentage = (param_count / total_params) * 100
        print(f"{name}: {param_count:,} 参数 ({percentage:.2f}%)")

    # 打印按主要层分组的统计
    print("\n==== 主要层参数统计 ====")
    for layer_name, param_count in grouped_params.items():
        percentage = (param_count / total_params) * 100
        print(f"{layer_name}: {param_count:,} 参数 ({percentage:.2f}%)")

    # 总计
    total_params_mb = (total_params * 4) / 1024 / 1024
    print(f"\n总模型参数量: {total_params:,} ({total_params_mb:.2f} MB)")