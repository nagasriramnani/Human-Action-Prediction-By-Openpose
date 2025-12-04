import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3d(nn.Module):
    """Squeeze-and-Excitation Block for 3D"""
    def __init__(self, channel, reduction=16):
        super(SEBlock3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResBlock3d(nn.Module):
    """Basic Residual Block 3D"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        self.use_se = use_se
        if use_se:
            self.se = SEBlock3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ActionResNet3D(nn.Module):
    def __init__(self, num_classes=6, in_channels=2):
        super(ActionResNet3D, self).__init__()
        
        # Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, use_se=False)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, use_se=True)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, use_se=True)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, use_se=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride, use_se):
        layers = []
        layers.append(ResBlock3d(in_channels, out_channels, stride, use_se))
        for _ in range(1, blocks):
            layers.append(ResBlock3d(out_channels, out_channels, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, C, T, J) -> (B, C, T, J, 1) for Conv3d spatial
        x = x.unsqueeze(-1) 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ActionResNet3D()
    # Test with dummy input: (Batch, Channels, Time, Joints)
    dummy = torch.randn(2, 2, 32, 25)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
