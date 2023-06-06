# ONNX

## Exporting to ONNX

[Exporting to ONNX](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/mxnet_onnx_export/mxnet_onnx_export.html)의 샘플은 아래와 같습니다.

```python
import os

from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

def save(model_dir, model):
    symbol_file = os.path.join(model_dir, 'model-symbol.json')
    params_file = os.path.join(model_dir, 'model-0000.params')

    model.symbol.save(symbol_file)
    model.save_params(params_file)

    data_shapes = [[dim for dim in data_desc.shape] for data_desc in model.data_shapes]
    output_path = os.path.join(model_dir, 'model.onnx')

    onnx_mxnet.export_model(symbol_file, params_file, data_shapes, np.float32, output_path)
```    


## Reference

[Exporting to ONNX](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/mxnet_onnx_export/mxnet_onnx_export.html)
