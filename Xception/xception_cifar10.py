import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Xception 논문 기반 수동 설계 (CIFAR-10용 단순화)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        layers = []
        if grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_channels, out_channels, 3, 1, 1))
            in_channels = out_channels
        for i in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_channels, in_channels, 3, 1, 1))
        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))
        self.block = nn.Sequential(*layers)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
    def forward(self, x):
        out = self.block(x)
        skip = x
        if self.skip is not None:
            skip = self.skip(x)
        return out + skip

class Xception(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = XceptionBlock(64, 128, 2, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, stride=2, start_with_relu=True, grow_first=True)
        self.middle = nn.Sequential(*[XceptionBlock(728, 728, 3, stride=1, start_with_relu=True, grow_first=True) for _ in range(2)])
        self.exit = nn.Sequential(
            XceptionBlock(728, 1024, 2, stride=2, start_with_relu=True, grow_first=False),
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle(x)
        x = self.exit(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Xception(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy (trained 10 epochs): {100 * correct / total:.2f}%")
