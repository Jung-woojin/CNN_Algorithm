import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time
import json
from pathlib import Path
from torchsummary import summary

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

if __name__ == "__main__":
    # 결과 저장할 디렉토리 생성
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet201(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model = model.to(device)
    
    # 모델 구조 출력
    print("\nModel Architecture:")
    print(model)
    print("\nDetailed Model Summary:")
    summary(model, (3, 224, 224))

    # feature extractor 동결, 마지막 FC 레이어만 파인튜닝
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False
    params_to_update = model.classifier.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=0.001)

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
    print(f"Test Accuracy (fine-tuned last FC): {100 * correct / total:.2f}%")
