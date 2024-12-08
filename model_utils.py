# model_utils.py
import torch
import torch.nn as nn
from torch.nn.utils import prune
import numpy as np
import time

def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def count_parameters(model):
    total_params = 0
    zero_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
            zero_params += torch.sum(p == 0).item()
    return total_params, zero_params

def measure_inference_time(model, testloader, device, num_runs=100):
    model.eval()
    times = []
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            
            # Warm-up
            for _ in range(10):
                _ = model(inputs)
            
            # 시간 측정
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            break
    
    return np.mean(times), np.std(times)