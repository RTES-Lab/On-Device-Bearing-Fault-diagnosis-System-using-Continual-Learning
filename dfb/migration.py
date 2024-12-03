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

from torch.utils.data import DataLoader

import numpy as np

import onnx
import onnxruntime

import tensorflow as tf
from onnx_tf.backend import prepare

import time


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def torch2onnx(
    model: torch.nn.Module,
    input_shape: Tuple,
    output_path: str, opset_version: int=13, 
    do_constant_folding: bool=True
) -> None:
    x = torch.randn(input_shape, requires_grad=True)
    model = model.cpu()
    src_model = model.eval()

    torch.onnx.export(
        src_model,
        x,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    target_model = onnx.load(output_path)
    onnx.checker.check_model(target_model)

    ort_session = onnxruntime.InferenceSession(output_path)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    target_out = ort_session.run(None, ort_inputs)

    src_out = src_model(x)

    np.testing.assert_allclose(to_numpy(src_out), target_out[0], rtol=1e-03, atol=1e-05)


def onnx2tf(onnx_path: str, input_shape: Tuple, output_path: str) -> None:
    x = np.random.randn(*input_shape).astype(np.float32)
    with tf.device("/device:cpu:0"):
        tf_rep = prepare(onnx.load(onnx_path))
        tf_rep.export_graph(output_path)

        target_model = tf.saved_model.load(output_path)

        target_out = target_model(input=x)["output"].numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    src_out = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(src_out[0], target_out, rtol=1e-05, atol=1e-05)


def tf2tflite(tf_path: str, tflite_path: str, quant: str="fp32",
              ref_dataloader: DataLoader=None) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    if quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        if ref_dataloader is None:
            raise ValueError("INT8 and Dynamic quantization needs reference dataset")

        def ref_data_gen():
            for data, _ in ref_dataloader:
                yield [data]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = ref_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quant == "dynamic":
        if ref_dataloader is None:
            raise ValueError("INT8 and Dynamic quantization needs reference dataset")

        def ref_data_gen():
            for data, _ in ref_dataloader:
                yield [data]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = ref_data_gen
    else:
        pass

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
