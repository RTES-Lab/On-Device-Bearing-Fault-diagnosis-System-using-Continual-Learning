# DeepFault with UOS Dataset

## 소개
 UOS 데이터셋을 활용하여 베어링 결함 진단을 위한 연속 학습(Continual Learning) 프레임워크를 제공합니다. 

## 목차
- [설치 방법](#설치-방법)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [상세 사용법](#상세-사용법)

## 설치 방법

### 환경 설정

1. 저장소 클론
```bash
git clone https://github.com/On-Device-Bearing-Fault-diagnosis-System-using-Continual-Learning.git
cd On-Device-Bearing-Fault-diagnosis-System-using-Continual-Learning
```

2. Conda 환경 생성 및 활성화
```bash
conda create -n myvenv python=3.8
conda activate myvenv
```

3. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 데이터셋 준비

1. UOS 데이터셋 구조:
UOS 데이터셋은 연속 학습(Continual Learning) 시나리오에 맞게 세 가지 phase로 구성되어 있다.

    Phase 1  ([Deep groove ball bearing](https://data.mendeley.com/datasets/53vtnjy6c6/1))

    Phase 2 ([Cylindrical roller bearing](https://data.mendeley.com/datasets/7trwzz77xh/1))

    Phase 3 ([Tapered roller bearing](https://data.mendeley.com/datasets/2cygy6y4rk/1))


 <br/> 

## 빠른 시작

### 기본 학습 실행
```bash
# ER 알고리즘을 사용한 기본 학습
python main.py --strategy replay

# 전체 학습 데이터 사용
python main.py --strategy joint
```


## 프로젝트 구조
```
DeepFault_UOS/
├── dfb/                    # 핵심 모듈
│   ├── model/             # 모델 구현
│   └── databuilder/       # 데이터 처리
├── src/                    # 소스 코드
│   ├── data/              # 데이터 관리
│   ├── experiments/       # 실험 구현
│   └── utils/             # 유틸리티
├── results/                # 실험 결과
└── configs/                # 설정 파일
```

## 상세 사용법

### 주요 학습 매개변수
```bash
python main.py --strategy [replay/joint] [매개변수...]
```

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| --strategy | 학습 전략 (replay / joint) | replay |
| --epoch | 학습 에폭 수 | 200 |
| --batch-size | 배치 크기 | 64 |
| --lr | 학습률 | 0.01 |
| --memory-size | replay memory buffer 크기 | 2000 |
| --device | 학습 장치 (cuda/cpu) | cuda |


