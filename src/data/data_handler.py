"""
데이터 처리를 위한 DataHandler 클래스
이 클래스는 베어링 결함 진단을 위한 데이터를 로드하고 전처리하는 역할을 담당한다.

주요 기능:
- Parquet 파일에서 데이터 로드
- Phase별 데이터 분할
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dfb.databuilder import build_from_dataframe

class DataHandler:
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        학습, 검증, 테스트 데이터를 parquet 파일에서 로드

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                학습(train), 검증(validation), 테스트(test) 데이터프레임
        """
        train_df = pd.read_parquet("uos_data/uos_cl_train.parquet")
        val_df = pd.read_parquet("uos_data/uos_cl_val.parquet")
        test_df = pd.read_parquet("uos_data/uos_cl_test.parquet")
        return train_df, val_df, test_df

    @staticmethod
    def split_by_phase(df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        데이터프레임을 phase별로 분할
        
        Args:
            df (pd.DataFrame): 분할할 데이터프레임
            
        Returns:
            List[pd.DataFrame]: phase 1, 2, 3에 해당하는 데이터프레임 리스트
            
        Note:
            - phase는 베어링의 동작 상태를 나타내는 지표
            - 총 3개의 phase로 구분 (1, 2, 3)
        """
        return [df[df['phase'] == i].reset_index(drop=True) for i in range(1, 4)]

    @staticmethod
    def generate_uos(phase1: pd.DataFrame, phase2: pd.DataFrame, 
                    phase3: pd.DataFrame, sample_length: int) -> Dict:
        """
        각 phase의 데이터프레임으로부터 UOS 데이터 생성
        
        Args:
            phase1 (pd.DataFrame): Phase 1 데이터
            phase2 (pd.DataFrame): Phase 2 데이터
            phase3 (pd.DataFrame): Phase 3 데이터
            sample_length (int): 샘플링할 데이터 길이
            
        Returns:
            Dict: 각 phase('A', 'B', 'C')별 처리된 데이터
            
        Note:
            - 베어링 결함 라벨을 숫자로 매핑:
                - H (Health): 0 - 정상 상태
                - IR (Inner Race): 1 - 내륜 결함
                - OR (Outer Race): 2 - 외륜 결함
                - B (Ball): 3 - 볼 결함
        """
        data = {}
        # 베어링 결함 타입을 숫자 라벨로 매핑
        label_map = {'H': 0, 'IR': 1, 'OR': 2, 'B': 3}
        
        # 각 phase별로 데이터 처리
        for phase_df, key in zip([phase1, phase2, phase3], ['A', 'B', 'C']):
            # 결함 타입을 숫자 라벨로 변환
            filtered_df = phase_df.assign(
                label=phase_df["bearing_fault"].map(label_map).astype(int)
            )
            # build_from_dataframe 함수를 사용하여 UOS 데이터 생성
            data[key] = build_from_dataframe(filtered_df, sample_length=sample_length)
            
        return data

    def prepare_data(self, sample_length: int = 2048):
        """
        전체 데이터 준비 과정을 실행
        
        Args:
            sample_length (int, optional): 샘플링할 데이터 길이. 기본값 2048
            
        Returns:
            Tuple: (train_data, val_data, test_data)
                각각의 데이터는 phase별로 구분된 딕셔너리 형태
                
        Process:
            1. parquet 파일에서 데이터 로드
            2. 각 데이터셋을 phase별로 분할
            3. 각 phase별로 UOS 데이터 생성
        """
        # 데이터 로드
        train_df, val_df, test_df = self.load_data()
        
        # phase별로 분할
        train_phases = self.split_by_phase(train_df)
        val_phases = self.split_by_phase(val_df)
        test_phases = self.split_by_phase(test_df)
        
        # UOS 데이터 생성
        train_data = self.generate_uos(*train_phases, sample_length)
        val_data = self.generate_uos(*val_phases, sample_length)
        test_data = self.generate_uos(*test_phases, sample_length)
        
        return train_data, val_data, test_data