from dfb.databuilder import *
from dfb.dataset import *
from dfb.trainmodule import *
from dfb.processing import *
from dfb.model.ticnn import *
from dfb.model.wdcnn import *
from dfb.model.stimcnn import *
from dfb.model.stftcnn import *
from dfb.model.wdcnnrnn import *
from dfb.model.aannet import *
from dfb.model.nrf_model import *
from dfb.paramsampler import *

import numpy as np

import tensorflow as tf

import time


def tflite_forward(
    interpreter: tf.lite.Interpreter,
    input: np.ndarray,
    input_detail: Dict,
    output_detail: Dict,
) -> Tuple[np.ndarray, float]:
    input_dtype = input_detail["dtype"]

    if list(input.shape) != list(input_detail["shape"]):
        print(input.shape)
        print(input_detail["shape"])
        raise ValueError("Model and input's shapes are not matched.")

    if input_dtype == np.uint8 or input_dtype == np.int8:
        scale, zero_point = input_detail["quantization"]
        input_data = input / scale + zero_point
        input_data = input_data.astype(input_dtype)
    else:
        input_data = input.astype(input_dtype)

    input_index = input_detail["index"]
    output_index = output_detail["index"]

    start = time.time()
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    end = time.time()

    return output_data, (end - start)


def tflite_evaluate(
    interpreter: tf.lite.Interpreter, dataloader: DataLoader
) -> Tuple[float, List]:
    input_detail = interpreter.get_input_details()[0]

    output_detail = interpreter.get_output_details()[0]

    total_corrects = 0
    total_samples = 0
    elapsed_times = []

    for data, label in dataloader:
        data = data.cpu().numpy()
        label = label.cpu().numpy()

        logits, inference_time = tflite_forward(
            interpreter, data, input_detail, output_detail
        )
        pred = np.argmax(logits, axis=-1)

        corrects = (pred == label).sum()
        total_corrects += corrects
        samples = label.size
        total_samples += samples
        elapsed_times.append(inference_time)

    accuracy = total_corrects / total_samples

    return accuracy, elapsed_times
