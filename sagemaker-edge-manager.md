# SageMaker Edge manager

[SageMaker Edge Manager end of life](https://docs.aws.amazon.com/sagemaker/latest/dg/edge-eol.html)에 따르면 April 26, 2024이후로 SageMaker Edge Manger를 사용할 수 없습니다.

### Q: How can I continue deploying models on the edge?
A: We suggest you try one the following machine learning tools. For a cross-platform edge runtime, use ONNX. ONNX is a popular, well-maintained open-source solution that translates your models into instructions that many types of hardware can run, and is compatible with the latest ML frameworks. ONNX can be integrated into your SageMaker workflows as an automated step for your edge deployments.

For edge deployments and monitoring use AWS IoT Greengrass V2. AWS IoT Greengrass V2 has an extensible packaging and deployment mechanism that can fit models and applications at the edge. You can use the built-in MQTT channels to send model telemetry back for Amazon SageMaker Model Monitor or use the built-in permissions system to send data captured from the model back to Amazon Simple Storage Service (Amazon S3). If you don't or can't use AWS IoT Greengrass V2, we suggest using MQTT and IoT Jobs (C/C++ library) to create a lightweight OTA mechanism to deliver models.

We have prepared [sample code available at this GitHub repository](https://github.com/aws-samples/ml-edge-getting-started) to help you transition to these suggested tools.
