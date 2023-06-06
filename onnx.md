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

## edge_application.py

[edge_application.py](https://github.com/aws-samples/ml-edge-getting-started/blob/main/samples/onnx_accelerator_sample1/onnxacceleratorsampleone/with_ggv2/components/aws.samples.windturbine.detector/edge_application.py)

```python
import onnxruntime as ort

# Initialize the OTA Model Manager
model_name = args.model_name
model_version = args.model_version

sess = None
try:
    sess = ort.InferenceSession(args.model_path)
except Exception as e:
    logging.error(e)
    exit()
    
x = turbine.create_dataset(data, TIME_STEPS, STEP)
x = np.transpose(x, (0, 2, 1)).reshape(x.shape[0], NUM_FEATURES, 10, 10).astype(np.float32)
ptemp = sess.run(None, {"input": x})    

p = np.asarray(ptemp[0])

a = x.reshape(x.shape[0], NUM_FEATURES, 100).transpose((0,2,1))
b = p.reshape(p.shape[0], NUM_FEATURES, 100).transpose((0,2,1))
            
# check the anomalies
pred_mae_loss = np.mean(np.abs(b - a), axis=1).transpose((1,0))
values = np.mean(pred_mae_loss, axis=1)
anomalies = (values > thresholds)
            
if anomalies.any():
    logging.info("Anomaly detected: %s" % anomalies)
else:
    logging.info("Ok")

time.sleep(PREDICTIONS_INTERVAL)
```

## Reference

[Exporting to ONNX](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/mxnet_onnx_export/mxnet_onnx_export.html)

[Hosting ONNX models with Amazon Elastic Inference](https://sagemaker-examples.readthedocs.io/en/latest/aws_sagemaker_studio/frameworks/mxnet_onnx_ei/mxnet_onnx_ei.html)
