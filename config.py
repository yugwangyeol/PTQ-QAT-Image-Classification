# config.py
class Config:
    # 데이터 관련 설정
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # 학습 관련 설정
    EPOCHS = 100
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Pruning 관련 설정
    PRUNING_AMOUNT = 0.5
    
    # 평가 관련 설정
    INFERENCE_RUNS = 100
    # config.py에 추가할 설정들
    QAT_EPOCHS = 5  # QAT 학습 에포크 수
    CALIBRATION_SIZE = 1000  # PTQ Calibration 데이터셋 크기