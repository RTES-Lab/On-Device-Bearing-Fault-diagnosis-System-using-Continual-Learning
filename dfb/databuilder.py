import os
import numpy as np
import pandas as pd
import scipy
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split


def build(
    sample_length: int, shift: int, one_hot: bool = False, type: str = "df", **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """데이터 생성 wrapper 함수, 추후 확장성을 위해 사용

    Parameters
    ----------
    sample_length: int
        각 데이터의 sample 길이
    shift: int
        overlapping으로 데이터를 샘플링할 때 샘플 간 간격
    one_hot: bool
        데이터를 one-hot encoding으로 생성할지 여부
    type: str
        데이터 소스, 현재는 df만 지원
    **kwargs: dict, optional
        type이 df일 경우 "df"으로 데이터프레임을 줌
    """
    if type not in ["df"]:
        raise ValueError("type argument must be in [df]")

    if type == "df":
        return build_from_dataframe(
            sample_length=sample_length, shift=shift, one_hot=one_hot, df=kwargs["df"]
        )


# def split_dataframe(
#     df: pd.DataFrame, train_ratio, val_ratio
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     cum_train_ratio = train_ratio
#     cum_val_ratio = cum_train_ratio + val_ratio

#     cols = df.columns

#     train_df = {}

#     val_df = {}

#     test_df = {}

#     for c in cols:
#         train_df[c] = []
#         val_df[c] = []
#         test_df[c] = []

#     for _, row in df.iterrows():
#         segment_length = row.data.size
#         train_idx = (int)(segment_length * cum_train_ratio)
#         val_idx = (int)(segment_length * cum_val_ratio)
#         for c in cols:
#             if c == "data":
#                 train_df[c].append(row[c][:train_idx])
#                 val_df[c].append(row[c][train_idx:val_idx])
#                 test_df[c].append(row[c][val_idx:])
#             else:
#                 train_df[c].append(row[c])
#                 val_df[c].append(row[c])
#                 test_df[c].append(row[c])
    
#     train_df = pd.DataFrame(train_df)
#     val_df = pd.DataFrame(val_df)
#     test_df = pd.DataFrame(test_df)

#     return train_df, val_df, test_df

def split_dataframe(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data segments of dataframe to the training, validation, and test segments.
    Author: Seongjae Lee

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe. The dataframe must contain the columns "data" and "label".
    train_ratio: float
        The ratio of the train data segment.
    val_ratio: float
        The ratio of the validation data segment.
        The test data segment ratio is automatically selected to 1 - train_ratio - val_ratio.
        train_ratio + val_ratio cannot be exceed 1.0.

    Returns
    ----------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the train, validation, and test dataframes.
    """
    cum_train_ratio = train_ratio
    cum_val_ratio = cum_train_ratio + val_ratio

    train_df = df.copy().reset_index(drop=True)
    val_df = df.copy().reset_index(drop=True)
    test_df = df.copy().reset_index(drop=True)

    for i in range(len(df)):
        segment_length = df.iloc[i]["data"].size
        train_idx = (int)(segment_length * cum_train_ratio)
        val_idx = (int)(segment_length * cum_val_ratio)

        train_df.at[i, "data"] = train_df.at[i, "data"][:train_idx]
        val_df.at[i, "data"] = val_df.at[i, "data"][train_idx:val_idx]
        test_df.at[i, "data"] = test_df.at[i, "data"][val_idx:]

    return train_df, val_df, test_df

def slice_dataframe(df: pd.DataFrame, window_map: List, shift_map: List) -> pd.DataFrame:
    result_rows = []
    
    for _, row in df.iterrows():
        data_array = row['data']
        other_columns = row.drop('data')
        window_size = window_map[other_columns["label"]]
        shift_size = shift_map[other_columns["label"]]
        
        for i in range(0, len(data_array) - window_size + 1, shift_size):
            window = data_array[i:i+window_size]
            new_row = other_columns.to_dict()
            new_row['data'] = window
            result_rows.append(new_row)

    result_df = pd.DataFrame(result_rows)
    return result_df

def load_uos(rootdir, sampling_rate=16000):
    bearing_types = ["DeepGrooveBall", "CylindricalRoller", "TaperedRoller"]
    sampling_rate = sampling_rate
    rotating_speeds = [600, 800, 1000, 1200, 1400, 1600]

    data_df = {
        "data": [],
        "bearing_type": [],
        "sampling_rate": [],
        "rotating_speed": [],
        "machine_fault": [],
        "bearing_fault": [],
        "label": []
    }

    for bearing_type in bearing_types:
        for rotating_speed in rotating_speeds:
            dir = os.path.join(rootdir, f"BearingType_{bearing_type}/SamplingRate_{sampling_rate}/RotatingSpeed_{rotating_speed}")

            file_list = [file for file in os.listdir(dir) if file.endswith(".mat") and os.path.isfile(os.path.join(dir, file))]
            file_list.sort()

            for file in file_list:
                file_dir = os.path.join(dir, file)
                file_key = os.path.splitext(file)[0]
                data_properties = file_key.split(sep="_")
                mat_data = scipy.io.loadmat(file_dir)
                vib_data = mat_data["Data"].ravel()
                data_df["data"].append(vib_data)
                data_df["bearing_type"].append(bearing_type)
                data_df["sampling_rate"].append(sampling_rate)
                data_df["rotating_speed"].append(rotating_speed)
                data_df["machine_fault"].append(data_properties[0])
                data_df["bearing_fault"].append(data_properties[1])
                status = f"{data_properties[0]}_{data_properties[1]}"
                if status == "H_H":
                    label = 0
                else:
                    label = 1
                data_df["label"].append(label)
    
    data_df = pd.DataFrame(data_df)

    return data_df

def build_from_dataframe(
    df: pd.DataFrame, sample_length: int, shift: int = 2048, one_hot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """데이터프레임으로부터 np.ndarray 형태의 데이터 쌍 생성

    Parameters
    ----------
    df: pd.DataFrame
        데이터프레임. 데이터프레임은 np.ndarray타입의 "data"컬럼와, int타입의 "label"컬럼을 가지고 있어야 함
    sample_length: int
        각 데이터의 sample 길이
    shift: int
        overlapping으로 데이터를 샘플링할 때 샘플 간 간격
    one_hot: bool
        데이터를 one-hot encoding으로 생성할지 여부

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        (데이터, 레이블) 튜플 반환
    """
    n_class = df["label"].max() - df["label"].min() + 1 # 4
    n_data = df.shape[0]  # 72000
    data = []
    label = []

    for i in range(n_data):
        d = df.iloc[i]["data"]
        td, tl = sample_data(
            d, sample_length, df.iloc[i]["label"], n_class, shift, one_hot
        )
        data.append(td)
        label.append(tl)

    data_array = np.concatenate(tuple(data), axis=0)
    label_array = np.concatenate(tuple(label), axis=0)

    return data_array, label_array


# bootstrap_from_dataframe(filter_cwru, sample_length=sample_length, n_sample=100, sample_ratio=sample_ratio, dataset_name="cwru", one_hot=False, n_map=sample_map["cwru"])

def bootstrap_from_dataframe(
    df: pd.DataFrame, sample_length: int, n_sample: int, sample_ratio , dataset_name, one_hot: bool = False, n_map: Dict = None
) -> Tuple[np.ndarray, np.ndarray]:    
#     df: pd.DataFrame, sample_length: int, n_sample: int, one_hot: bool = False, n_map: Dict = None
# ) -> Tuple[np.ndarray, np.ndarray]:
    """데이터프레임으로부터 np.ndarray 형태의 데이터 쌍 생성

    Parameters
    ----------
    df: pd.DataFrame
        데이터프레임. 데이터프레임은 np.ndarray타입의 "data"컬럼와, int타입의 "label"컬럼을 가지고 있어야 함
    sample_length: int
        각 데이터의 sample 길이
    shift: int
        overlapping으로 데이터를 샘플링할 때 샘플 간 간격
    one_hot: bool
        데이터를 one-hot encoding으로 생성할지 여부

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        (데이터, 레이블) 튜플 반환
    """
    n_class = df["label"].max() - df["label"].min() + 1
    n_data = df.shape[0]
    data = []
    label = []
    indiv_sample = n_sample // n_data

    for i in range(n_data):
        d = df.iloc[i]["data"]
        if n_map == None:
            n_samples = indiv_sample
        else:
            n_samples = n_map[str(df.iloc[i]["label"])]
        if(dataset_name == "mfpt"):
            if(sample_ratio > 0.5):
                if(i > 2 and i < 5):
                    n_samples += 1
            else:
                if(i > 2 and i < 7):
                    n_samples += 1
        td, tl = bootstrap_data(
            d, sample_length, n_samples, df.iloc[i]["label"], n_class, one_hot
        )
        data.append(td)
        label.append(tl)

    data_array = np.concatenate(tuple(data), axis=0)
    label_array = np.concatenate(tuple(label), axis=0)

    return data_array, label_array

def bootstrap_data(
    data: np.ndarray,
    sample_length: int,
    n_samples: int,
    cls_id: int,
    num_class: int,
    one_hot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """(N,) 크기 np array로부터 데이터를 자름

    Parameters
    ----------
    data: np.ndarray
        자를 대상이 되는 데이터
    sample_length: int
        각 샘플의 길이
    shift: int
        overlapping으로 데이터를 샘플링할 때 샘플 간 간격
    cls_id: int
        data의 클래스 id
    num_class: int
        전체 데이터셋의 클래스 수 (one_hot encoding을 만들 때 사용)
    one_hot: bool
        one_hot encoding으로 데이터를 반환할 경우 True

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        (데이터, 레이블) 튜플 반환

    Raises
    ----------
    ValueError
        class id가 전체 클래스 수를 넘어가는 경우

    Notes
    ----------
    원핫 인코딩은 빼는게 좋을듯..
    """
    if cls_id >= num_class:
        raise ValueError("class id is out of bound")
    
    bootstrap_index = np.random.randint(0, len(data) - sample_length, size=n_samples)
    sampled_data = np.array(
        [
            data[i : i + sample_length]
            for i in bootstrap_index
        ]
    )
    if one_hot:
        label = np.zeros((sampled_data.shape[0], num_class))
        label[:, cls_id] = 1
    else:
        label = np.zeros((sampled_data.shape[0]))
        label = label + cls_id
    return sampled_data, label

def sample_data(
    data: np.ndarray,
    sample_length: int,
    cls_id: int,
    num_class: int,
    shift: int,
    one_hot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """(N,) 크기 np array로부터 데이터를 자름

    Parameters
    ----------
    data: np.ndarray
        자를 대상이 되는 데이터
    sample_length: int
        각 샘플의 길이
    shift: int
        overlapping으로 데이터를 샘플링할 때 샘플 간 간격
    cls_id: int
        data의 클래스 id
    num_class: int
        전체 데이터셋의 클래스 수 (one_hot encoding을 만들 때 사용)
    one_hot: bool
        one_hot encoding으로 데이터를 반환할 경우 True

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        (데이터, 레이블) 튜플 반환

    Raises
    ----------
    ValueError
        class id가 전체 클래스 수를 넘어가는 경우

    Notes
    ----------
    원핫 인코딩은 빼는게 좋을듯..
    """
    if cls_id >= num_class:
        raise ValueError("class id is out of bound")

    if len(data) == sample_length:
        sampled_data = np.array([data])
    else:
        sampled_data = np.array(
            [
                data[i : i + sample_length] for i in range(0, len(data) - sample_length + 1, shift)
            ]
        )
    
    if one_hot:
        label = np.zeros((sampled_data.shape[0], num_class))
        label[:, cls_id] = 1
    else:
        label = np.zeros((sampled_data.shape[0]))
        label = label + cls_id
    return sampled_data, label


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    train_size: float,
    random_state: int = None,
    shuffle: bool = True,
    stratify: np.ndarray = False,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split numpy array-style data pair (X, y) to train, validation, and test dataset.

    Parameters
    ----------
    X: np.ndarray
        Data
    y: np.ndarray
        Lable
    test_size: float
        Ratio of the test dataset (0~1)
    val_size: float
        Ratio of the validation dataset (0~1)
    train_size: float
        Ratio of the train dataset (0~1)
    random_state: int
        Random state used for data split
    shuffle: bool
        Whether or not to shuffle the data before splitting.
    stratify: bool
        Option for the stratified split. If true, data is splited based on the label's distribution

    Returns
    ----------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        Return ((X_train, y_train), (X_val, y_val), (X_test, y_test)) pairs.

    Raises
    ----------
        train_size + val_size + test size must be 1.0.

    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("data split ratio error")

    if stratify:
        stratify_y = y

    X_nt, X_test, y_nt, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_y,
    )

    if stratify:
        stratify_y = y_nt

    X_train, X_val, y_train, y_val = train_test_split(
        X_nt,
        y_nt,
        test_size=(val_size / (train_size + val_size)),
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_y,
    )

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test))
