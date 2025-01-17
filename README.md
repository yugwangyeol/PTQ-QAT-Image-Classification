# PTQ-QAT-Image-Classification

## Project Overview
이 프로젝트는 딥러닝 모델의 경량화 방법인 Post-Training Quantization(PTQ)와 Quantization-Aware Training(QAT)의 성능을 비교 분석합니다. CIFAR10 데이터셋을 사용하여 정확도, 모델 크기, 추론 시간 측면에서 각 방법의 효과를 검증합니다.

### Quantization Methods
- **Post-Training Quantization (PTQ)**
  - 학습이 완료된 모델에 직접 양자화 적용
  - Calibration 데이터셋을 사용하여 양자화 범위 조정
  
- **Quantization-Aware Training (QAT)**
  - 학습 과정에서 양자화를 고려한 파인튜닝 수행
  - 양자화로 인한 정확도 손실 최소화

## Key Features
- CNN 모델을 사용한 CIFAR10 이미지 분류
- PTQ 및 QAT 구현 및 적용
- 모델 성능 평가 (정확도, 크기, 추론 시간)
- 양자화 방법 간 성능 비교 분석

## Getting Started

### Prerequisites
```bash
torch >= 1.7.0
torchvision >= 0.8.0
numpy
```

### Installation
1. Repository Clone
```bash
git clone https://github.com/username/ptq-qat-image-classification.git
cd ptq-qat-image-classification
```

2. Package Installation
```bash
pip install torch torchvision numpy
```

### Usage
```bash
python main_quantization.py
```

## Project Structure
```
├── config.py           # Configuration
├── data_loader.py      # Data loading and preprocessing
├── main_quantization.py # Main execution file
├── model_utils.py      # Model training/evaluation utilities
├── models.py          # Model architecture
├── quantization_methods.py # Quantization implementation
└── README.md
```

## Configuration
config.py에서 다음 설정들을 조정할 수 있습니다:
```python
# 데이터 관련 설정
BATCH_SIZE = 128
NUM_WORKERS = 4

# 학습 관련 설정
EPOCHS = 100
LEARNING_RATE = 0.1

# Quantization 관련 설정
QAT_EPOCHS = 5
CALIBRATION_SIZE = 1000
```

## Results

<div align="center">
  <img src="https://github.com/user-attachments/assets/553f1cc5-603a-4f45-a3f3-3a025d647df9" alt="PTQ-QAT Comparison">
</div>

### Accuracy Comparison
- Base Model: 76.63%
- PTQ Model: 69.34% (Base 대비 -7.29%p)
- QAT Model: 74.93% (Base 대비 -1.70%p)

### Model Size Comparison
- Base Model: 5.96MB
- PTQ/QAT Model: 1.51MB (74.6% 감소)

### Inference Time Comparison
- Base Model: 385.82 ± 72.12ms
- PTQ Model: 182.55 ± 39.81ms (52.7% 속도 향상)
- QAT Model: 185.02 ± 44.61ms (52.1% 속도 향상)

## Analysis

### PTQ vs QAT 비교
1. **정확도 측면**
   - QAT가 PTQ보다 더 나은 정확도 유지
   - PTQ는 상대적으로 큰 정확도 손실 발생

2. **리소스 효율성**
   - 두 방법 모두 동일한 수준의 모델 크기 감소
   - 추론 속도 향상도 비슷한 수준

3. **구현 복잡도**
   - PTQ: 구현이 단순, 빠른 적용 가능
   - QAT: 추가 학습 필요, 구현이 더 복잡

## Conclusion
- QAT는 정확도를 유지하면서 모델 경량화가 필요한 경우에 적합
- PTQ는 빠른 구현이 필요하고 정확도 손실을 감수할 수 있는 경우에 적합
- 두 방법 모두 효과적인 모델 경량화와 추론 속도 개선 달성