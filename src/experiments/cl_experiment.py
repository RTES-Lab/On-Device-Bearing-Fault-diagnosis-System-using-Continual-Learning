"""
연속 학습(Continual Learning) 실험을 위한 핵심 클래스
이 클래스는 데이터 처리, 모델 학습, 평가, 결과 저장 등 실험의 전체 과정을 관리한다.

주요 기능:
- 데이터셋 준비 및 벤치마크 생성
- 학습 전략(Joint Training/Replay) 설정
- 모델 학습 및 평가 실행
- 실험 결과 저장 및 로깅
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised import JointTraining, Replay
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
from torch.utils.data import TensorDataset

class CLExperiment:
    def __init__(self, opt):
        """
        CLExperiment 클래스 초기화
        
        Args:
            opt: 실험 설정이 담긴 인자 객체
                (학습률, 배치 크기, 메모리 크기 등의 설정 포함)
        """
        self.opt = opt
        self.avg_results = [] 
        self.num_tasks = 0
        self._setup_data()  

        current_dir = os.getcwd()
        self.save_dir = os.path.join(current_dir, 'results')
        os.makedirs(self.save_dir, exist_ok=True)

    def _setup_data(self):
        """데이터 핸들러를 통해 학습/검증/테스트 데이터 로드"""
        from src.data.data_handler import DataHandler
        data_handler = DataHandler()
        self.train_data, self.val_data, self.test_data = data_handler.prepare_data()
        
    def initialize(self, model, scenario=None):
        """
        실험 초기화: 모델과 시나리오 설정
        
        Args:
            model: 학습할 딥러닝 모델
            scenario: 사용할 학습 시나리오 
        """
        self.model = model
        
        if not scenario:
            self._make_benchmark()
            
        self._setup_training()
        self._setup_cl_strategy()
    
    def _make_benchmark(self):
        """
        Avalanche 벤치마크 생성
        - 각 phase(A, B, C)별 데이터를 개별 task로 변환
        - 데이터 형식을 Avalanche 프레임워크에 맞게 변환
        """
        # Create benchmark from data
        train_datasets = []
        test_datasets = []
        
        # 각 phase(A, B, C)별로 데이터셋 처리
        for i, phase in enumerate(['A', 'B', 'C']):
            # 현재 phase의 학습/테스트 데이터 가져오기
            train_dataset = self.train_data[phase]
            test_dataset = self.test_data[phase]
            
            # 데이터가 tuple 형태로 반환된 경우 처리
            if isinstance(train_dataset, tuple):
                train_data, train_labels = train_dataset
                test_data, test_labels = test_dataset
                
                # numpy array를 pytorch tensor로 변환
                if not isinstance(train_data, torch.Tensor):
                    train_data = torch.tensor(train_data, dtype=torch.float32)
                if not isinstance(train_labels, torch.Tensor):
                    train_labels = torch.tensor(train_labels, dtype=torch.long)
                if not isinstance(test_data, torch.Tensor):
                    test_data = torch.tensor(test_data, dtype=torch.float32)
                if not isinstance(test_labels, torch.Tensor):
                    test_labels = torch.tensor(test_labels, dtype=torch.long)
                
                # 채널 차원 추가 (필요한 경우)
                if len(train_data.shape) == 2:  
                    train_data = train_data.unsqueeze(1)  
                if len(test_data.shape) == 2: 
                    test_data = test_data.unsqueeze(1)   
                
                # TensorDataset 생성
                train_dataset = TensorDataset(train_data, train_labels)
                test_dataset = TensorDataset(test_data, test_labels)
                
                # targets 속성 추가 (Avalanche 요구사항)
                train_dataset.targets = train_labels
                test_dataset.targets = test_labels
                
                print(f"Phase {phase} data shapes:")
                print(f"Train data: {train_data.shape}, Train labels: {train_labels.shape}")
                print(f"Test data: {test_data.shape}, Test labels: {test_labels.shape}")
            
            # Avalanche 데이터셋으로 변환
            train_avalanche = make_classification_dataset(train_dataset, task_labels=i)
            test_avalanche = make_classification_dataset(test_dataset, task_labels=i)
            
            train_datasets.append(train_avalanche)
            test_datasets.append(test_avalanche)
        
        # Avalanche 벤치마크 생성
        self.scenario = dataset_benchmark(
            train_datasets,
            test_datasets
        )

    def _setup_training(self):
        """
        - SGD 옵티마이저 초기화
        - Cross Entropy Loss 함수 설정
        """
        self.optimizer = SGD(
            self.model.parameters(),
            lr=getattr(self.opt, 'lr', 0.01),
            momentum=getattr(self.opt, 'momentum', 0),
            weight_decay=getattr(self.opt, 'weight_decay', 0.01)
        )
        self.criterion = CrossEntropyLoss()

    def _setup_cl_strategy(self):
        """
        연속 학습 전략 설정
        - 평가 메트릭 설정 (accuracy, loss, forgetting)
        - 로깅 설정
        - 학습 전략(Joint/Replay) 초기화
        """
        # 평가 플러그인 설정
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), TextLogger(open(self._get_log_path(), 'a'))]
        )

        # 전략 공통 인자
        strategy_args = dict(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_epochs=getattr(self.opt, 'epoch', 200),
            train_mb_size=getattr(self.opt, 'batch_size', 64),
            eval_mb_size=getattr(self.opt, 'batch_size', 64),
            device=getattr(self.opt, 'device', 'cuda'),
            evaluator=eval_plugin
        )

        # 설정에 따라 학습 전략 초기화
        if self.opt.strategy.lower() == 'replay':
            # Replay 전략: 이전 데이터의 일부를 메모리에 저장하며 학습
            self.strategy = Replay(
                **strategy_args,
                mem_size=getattr(self.opt, 'memory_size', 2000)
            )
        else:  # joint training
            # Joint 전략: 모든 데이터를 동시에 학습
            self.strategy = JointTraining(**strategy_args)

    def execute(self):
        """설정된 전략에 따라 학습 실행"""
        if self.opt.strategy.lower() == 'joint':
            self._execute_joint()
        else:
            self._execute_sequential()
        
    def _execute_sequential(self):
        """
        순차적 학습 실행 (Replay 전략)
        - 각 phase별로 순차적으로 학습하고 평가
        """
        results = []
        for exp in self.scenario.train_stream:
            self.strategy.train(exp)
            results.append(self.strategy.eval(self.scenario.test_stream))

    def _execute_joint(self):
        """
        통합 학습 실행 (Joint 전략)
        - 모든 phase의 데이터를 한번에 학습
        """
        results = []
        self.strategy.train(self.scenario.train_stream)
        results.append(self.strategy.eval(self.scenario.test_stream))

    def _get_log_path(self):
        """로그 파일 경로 생성"""
        base_name = f"{self.opt.strategy.lower()}_experiment"
        return os.path.join(self.save_dir, f"{base_name}.log")

    def _get_save_path(self, suffix):
        """결과 저장 파일 경로 생성"""
        base_name = f"{self.opt.strategy.lower()}_experiment"
        return os.path.join(self.save_dir, f"{base_name}_{suffix}")

    def save_results(self):
        """학습된 모델의 가중치를 파일로 저장"""
        model_path = self._get_save_path("model.pth")
        torch.save(self.model.state_dict(), model_path)