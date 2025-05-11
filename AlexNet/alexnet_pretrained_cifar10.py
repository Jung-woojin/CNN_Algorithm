import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time
import json
from pathlib import Path
from torchsummary import summary
import time
import json
from pathlib import Path

# 데이터 전처리 (ImageNet 사전학습 모델용)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss

if __name__ == "__main__":
    # 결과 저장할 디렉토리 생성
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 사전훈련된 AlexNet 불러오기 (ImageNet에서 학습된 가중치 사용)
    model = models.alexnet(pretrained=True)
    # 2. 마지막 분류 레이어를 CIFAR-10(10클래스)용으로 교체 (랜덤 초기화)
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(device)
    
    # 모델 구조 출력
    print("\nModel Architecture:")
    print(model)
    print("\nDetailed Model Summary:")
    summary(model, (3, 224, 224))

    # 3. feature extractor 및 classifier 앞부분 동결 (파라미터 업데이트 X)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier[:4]:
        param.requires_grad = False

    # 4. 마지막 2개 FC 레이어만 optimizer에 등록 (파인튜닝)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=0.001)
    
    # 학습 결과를 저장할 딕셔너리
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    
    print("Starting training...")
    # 학습 루프
    num_epochs = 30
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 에폭 종료 후 훈련셋과 검증셋 성능 평가
        train_acc, train_loss = evaluate_model(model, trainloader, criterion, device)
        val_acc, val_loss = evaluate_model(model, testloader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        # 결과 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s')
        print('-' * 60)
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), results_dir / 'alexnet_pretrained_best_model.pth')
    
    print("Training finished!")
    
    # 최종 테스트 셋 성능 평가
    model.load_state_dict(torch.load(results_dir / 'alexnet_pretrained_best_model.pth'))
    test_acc, test_loss = evaluate_model(model, testloader, criterion, device)
    print(f'\nFinal Test Set Performance:')
    print(f'Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    # 학습 결과 저장
    history['test_acc'] = test_acc
    history['test_loss'] = test_loss
    
    with open(results_dir / 'alexnet_pretrained_history.json', 'w') as f:
        json.dump(history, f)
