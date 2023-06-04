# Anomaly Detection

## Anomaly 란?

[이상 (Anomaly)](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/anomaly.md)에서는 Anomaly의 정의에 대해 설명합니다. 이상은 시계열 데이터의 갑작스러운 급변과 같이 정형화된 데이터 또는 특정 패턴을 보이는 데이터와 다른 형태로 나타나는 데이터를 말합니다. 


### Anomaly Detection 용도

- 유지보수를 예측: default/defect 예측
- 네트워크 보안
- Health Care
- financial fraud 검출 

### 뉴욕 택시 정보

[nyc_taxi.csv](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/nyc-taxi/nyc_taxi.csv)에서는 뉴욕 택시의 호출 정보를 표시하고 있습니다.

### Cloud Logs

[cloudwatchlogs.ipynb](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/cloudwatchlogs/cloudwatchlogs.ipynb)에서는 [realAWSCloudwatch](https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch)를 이용하여 실제 CloudWatch로 수집된 데이터에서 Anomaly를 아래와 같이 확인할 수 있습니다.

![image](https://github.com/kyopark2014/ML-anomaly-detection/assets/52392004/3324aa46-838d-4d22-86b7-5bc73698772c)


## RCF 란?

[Random Cut Forest (RCF)](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/rcf.md)에서는 RCF 이론에 대해 설명합니다. RCF 알고리즘은 각 데이터에 대해 이상치 스코어(anomaly score)를 부여할 수 있습니다. 이상치 스코어가 높을수록 이상(Anomaly)일 가능성이 높아집니다. 이상의 판단 기준은 데이터의 형태나 시스템의 허용범위에 따라 다를 수 있으나 통상 평균값에서 3 표준편차 범위를 넘을 경우 이상치로 간주합니다 3표준편차 범위(μ±3σ)는 전체 데이터셋의 99.73%에 해당하며, 따라서 0.27%에 해당하는 데이터를 이상치로 볼 수 있다는 의미입니다. 

RCF 알고리즘은 학습 데이터에서 임의로 샘플을 추출하여 작업을 수행하며 reservoir sampling 기법을 이용하여 데이터 스트림으로부터 샘플 데이터를 효율적으로 추출합니다. 데이터를 랜덤하게 분할하는 이진트리를 구성하는데 트리의 리프노드가 하나의 데이터를 가질때까지 반복합니다. 이상치 스코어는 Forest를 구성하는 트리의 평균 깊이(depth)와 반비례합니다. 

## SageMaker를 이용한 Anomaly Detection

[SageMaker의 RCF](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/SageMaker/README.md)에서는 SageMaker의 Built-in Algorithm을 이용한 RCF 구현 방법에 대해 설명합니다.



## RRCF (Robust Random Cut Forest)를 이용한 Anomaly Detection

[RRCF](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/rrcf/README.md)에서는 RRCF를 이용한 Anomaly Detection에 대해 설명합니다.

## Case Stuides

### Network Anomaly Detection

[case-study-network-anomaly](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/case-study-network-anomaly.md)에서는 네트워크에서 발생하는 비정상 트래픽의 검출에 대해 분석합니다.

## Reference 

[이상 탐지를 위한 Amazon SageMaker 의 Random Cut Forest 빌트인 알고리즘](https://aws.amazon.com/ko/blogs/korea/use-the-built-in-amazon-sagemaker-random-cut-forest-algorithm-for-anomaly-detection/)

[Outlier Detection Algorithm: Isolation Forest](https://datanetworkanalysis.github.io/2020/04/01/isolation_forest)

[Random Cut Forest (RCF) Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)

[Using Random Cut Forests for real-time anomaly detection in Amazon OpenSearch Service](https://aws.amazon.com/ko/blogs/big-data/using-random-cut-forests-for-real-time-anomaly-detection-in-amazon-opensearch-service/)

[Real Time Anomaly Detection in Open Distro for Elasticsearch](https://opensearch.org/blog/real-time-anomaly-detection-in-open-distro-for-elasticsearch/)

[Unsupervised Anomaly Detection with Isolation Forest - Elena Sharova](https://www.youtube.com/watch?v=5p8B2Ikcw-k)

[Robust Anomaly Detection + Seasonal-Trend Decomposition : Time Series Talk](https://www.youtube.com/watch?v=1NXryMoU7Ho)

[STL Decomposition.ipynb - Time-Series-Analysis](https://github.com/ritvikmath/Time-Series-Analysis/blob/master/STL%20Decomposition.ipynb)

[Anomaly Detection using Isolation Forest - Time Series](https://www.youtube.com/watch?v=hkXPdkPfgoo)

