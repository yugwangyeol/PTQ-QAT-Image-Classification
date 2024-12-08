# main_quantization.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_cifar10
from model_utils import evaluate_model, measure_inference_time
from quantization_methods import apply_ptq, apply_qat, measure_model_size
from models import QuantizableCNN  # 추가된 import
from config import Config

def main():
    # 재현성을 위한 시드 설정
    torch.manual_seed(42)
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로드
    trainloader, testloader = load_cifar10(Config.BATCH_SIZE, Config.NUM_WORKERS)
    
    # Calibration용 작은 데이터셋 생성
    calibration_dataset = torch.utils.data.Subset(
        trainloader.dataset,
        indices=torch.randperm(len(trainloader.dataset))[:1000].tolist()
    )
    
    calibration_loader = torch.utils.data.DataLoader(
        calibration_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    # 기본 모델 설정
    base_model = QuantizableCNN()
    base_model = base_model.to(device)
    
    # 초기 학습
    print("Training base model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        base_model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 기본 모델 학습
    for epoch in range(Config.EPOCHS):
        base_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
    
    # 양자화 적용
    print("\nApplying quantization methods...")
    try:
        ptq_model = apply_ptq(base_model, calibration_loader)
        print("PTQ applied successfully")
    except Exception as e:
        print(f"Error applying PTQ: {e}")
        ptq_model = None
    
    try:
        qat_model = apply_qat(base_model, trainloader, criterion, optimizer, device)
        print("QAT applied successfully")
    except Exception as e:
        print(f"Error applying QAT: {e}")
        qat_model = None
    
    # 모델 딕셔너리 생성
    models = {'Base': base_model}
    if ptq_model is not None:
        models['PTQ'] = ptq_model
    if qat_model is not None:
        models['QAT'] = qat_model
    
    # 결과 평가 및 출력
    results = {name: {} for name in models.keys()}
    
    print("\n=== Results ===")
    
    # 정확도 평가
    print("\nAccuracy:")
    for name, model in models.items():
        try:
            model_device = 'cpu' if name in ['PTQ', 'QAT'] else device
            model.to(model_device)
            acc = evaluate_model(model, testloader, model_device)
            results[name]['accuracy'] = acc
            print(f"{name} Model: {acc:.2f}%")
        except Exception as e:
            print(f"Error evaluating {name} model: {e}")
    
    # 모델 크기 비교
    print("\nModel Size (MB):")
    for name, model in models.items():
        try:
            size = measure_model_size(model)
            results[name]['size'] = size
            print(f"{name} Model: {size:.2f} MB")
            if name != 'Base':
                reduction = 100 * (results['Base']['size'] - size) / results['Base']['size']
                print(f"  Size reduction: {reduction:.1f}%")
        except Exception as e:
            print(f"Error measuring size for {name} model: {e}")
    
    # 추론 시간 비교
    # main_quantization.py의 inference time 측정 부분
    print("\nInference Time (ms):")
    # 모든 모델을 CPU에서 측정
    for name, model in models.items():
        try:
            model.to('cpu')  # 모든 모델을 CPU로
            mean_time, std_time = measure_inference_time(model, testloader, 'cpu', Config.INFERENCE_RUNS)
            results[name]['inference'] = {
                'mean': mean_time,
                'std': std_time
            }
            print(f"{name} Model: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
            if name != 'Base':
                speedup = 100 * (results['Base']['inference']['mean'] - mean_time) / results['Base']['inference']['mean']
                print(f"  Speed improvement: {speedup:.1f}%")
        except Exception as e:
            print(f"Error measuring inference time for {name} model: {e}")

if __name__ == "__main__":
    main()