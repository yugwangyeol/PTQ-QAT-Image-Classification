# quantization_methods.py
import torch
import torch.nn as nn
import os
from copy import deepcopy
from models import QuantizableWrapper  # 추가된 import

def apply_ptq(model, calibration_loader, device='cpu'):
    """Post Training Quantization 적용"""
    model_ptq = deepcopy(model)
    model_ptq.to('cpu')
    model_ptq.eval()
    
    # 양자화를 위한 모델 준비
    model_ptq = QuantizableWrapper(model_ptq)
    
    # 퓨전을 위한 설정
    model_ptq.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model_ptq, inplace=True)
    
    # Calibration
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model_ptq(inputs.cpu())
    
    torch.quantization.convert(model_ptq, inplace=True)
    
    return model_ptq

def apply_qat(model, train_loader, criterion, optimizer, device, epochs=5):
    """Quantization Aware Training 적용"""
    model_qat = deepcopy(model)
    model_qat = QuantizableWrapper(model_qat)
    model_qat.to(device)
    
    # QAT 설정
    model_qat.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model_qat, inplace=True)
    
    # QAT 수행
    for epoch in range(epochs):
        model_qat.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model_qat(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'QAT Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.3f}')
    
    model_qat.eval()
    model_qat.to('cpu')
    torch.quantization.convert(model_qat, inplace=True)
    
    return model_qat

def measure_model_size(model):
    """모델 크기 측정 (MB 단위)"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)  # MB로 변환
    os.remove('temp.p')
    return size