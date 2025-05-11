import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time
import json
from pathlib import Path
from torchsummary import summary

# 데이터 전처리 (CIFAR-10을 32x32로 유지, 정규화)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# LeNet 직접 구현 (사전훈련 가정: 여기서는 예시로 마지막 2개 FC 레이어만 학습)
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # 결과 저장할 디렉토리 생성
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet(num_classes=10).to(device)
    
    # 모델 구조 출력
    print("\nModel Architecture:")
    print(model)
    print("\nDetailed Model Summary:")
    summary(model, (3, 32, 32))

    # feature extractor 및 classifier 앞부분 동결 (파라미터 업데이트 X)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in list(model.classifier.parameters())[:2]:  # classifier[0] (Linear), classifier[1] (Tanh)
        param.requires_grad = False

    # 마지막 2개 FC 레이어만 optimizer에 등록 (파인튜닝)
    params_to_update = list(model.classifier.parameters())[2:]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=0.001)

    # 10 에폭 학습 (마지막 2개 FC 레이어만 파인튜닝)
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

    # 테스트셋에서 파인튜닝된 모델의 정확도 평가
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

    print(f"Test Accuracy (fine-tuned last 2 FC): {100 * correct / total:.2f}%")
