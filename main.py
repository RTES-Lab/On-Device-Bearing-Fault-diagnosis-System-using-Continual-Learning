"""
베어링 결함 진단을 위한 연속 학습(Continual Learning) 실험의 메인 스크립트
이 스크립트는 실험의 전체적인 흐름을 제어하고 실행합니다.

주요 기능:
- 커맨드 라인 인자 파싱
- 실험 환경 초기화
- 모델 생성 및 학습 실행
- 결과 저장
"""

import os
import torch
from src.utils.config import get_args
from src.experiments.cl_experiment import CLExperiment
from dfb.model.wdcnn import WDCNN

def main():
    # 커맨드 라인에서 실험 설정 인자들을 가져오기
    # (learning rate, batch size, 학습 전략 등)
    args = get_args()
    
    # CLExperiment 클래스를 통해 연속 학습 실험 초기화
    # CLExperiment는 데이터 로딩, 모델 학습, 평가를 관리
    cl_experiment = CLExperiment(args)
    
    # WDCNN(Wide Deep Convolutional Neural Network) 모델 생성
    # n_classes=4는 베어링의 상태를 4가지로 분류 (정상, 내륜 결함, 외륜 결함, 볼 결함)
    model = WDCNN(n_classes=4)
    
    # initialize: 데이터 준비, 옵티마이저 설정, 학습 전략 선택
    cl_experiment.initialize(model)
    # execute: 선택된 전략(joint/replay)에 따라 학습 수행
    cl_experiment.execute()
    # save_results: 학습된 모델과 실험 결과를 저장
    cl_experiment.save_results()

if __name__ == "__main__":
    main()