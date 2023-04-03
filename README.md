# Anomaly Detection

## Anomaly 란?

[Anomaly](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/anomaly.md)에서는 Anomaly의 정의에 대해 설명합니다.

## RCF 란?

[Random Cut Forest (RCF)](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/rcf.md)에서는 RCF 이론에 대해 설명합니다.

## Sagemaker의 Built-in Algorithm


### Random Cut Forest

Amazon SageMaker의 Random Cut Forest (RCF) 알고리즘은 데이터셋에 포함되어 있는 이상치들을 탐지하는 비지도 학습(unsupervised learning) 알고리즘입니다. 특히, Amazon SageMaker의 RCF 알고리즘은 각 데이터에 대해 이상치 스코어(anomaly score)를 부여합니다. 이상치 스코어가 낮을 경우 해당 데이터는 정상(normal)일 가능성이 높은 반면, 높은 스코어를 보일 경우 이 데이터는 이상(anomaly)일 가능성이 높다는 것을 나타냅니다.

Amazon SageMaker의 RCF 알고리즘은 우선 학습 데이터에서 임의로 샘플을 추출하여 작업을 수행합니다. 학습 데이터가 지나치게 커서 머신 한 대에 올리기 어려울 경우, [reservoir sampling 기법](https://en.wikipedia.org/wiki/Reservoir_sampling)을 이용하여 데이터 스트림으로부터 샘플 데이터를 효율적으로 추출합니다 이를 통해서 서브 샘플 데이터들이  Random Cut Forest를 구성하는 트리 각각에 분포하게 됩니다. 각각의 서브 샘플 데이터는 바운딩 박스를 랜덤하게 분할하는 식으로 만들어진 이진 트리를 구성하게 되는데, 이러한 트리 구성은 각 리프 노드(즉, 트리 맨 끝의 터미널 노드)가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질 때까지 계속 반복됩니다. 입력 데이터에 할당된 이상치 스코어는 포레스트를 구성하는 트리의 평균 깊이(depth)와 반비례합니다.

- Anomaly를 비지도 알고리즘을 통해 detecton 하기 위해 anomaly score를 부여: 점수는 case by case
- Random Cut Forest Algorithm
. 학습데이터에서 임의의 샘풀을 추출. 이때 reservior sampling 사용하여 데이터 스트림에서 데이터를 효율적으로 추출
. 서브 셈플 데이터들을 Random Cut Forest를 구성하는 이진트리 구성하고 이때 모든 leaf가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질때까지 계속 반복
. 입력 데이터에 할당된 anomaly score는 forest를 구성하는 tree의 평균 depth와 반비례

### IP Insight*s


## Reference 

[이상 탐지를 위한 Amazon SageMaker 의 Random Cut Forest 빌트인 알고리즘](https://aws.amazon.com/ko/blogs/korea/use-the-built-in-amazon-sagemaker-random-cut-forest-algorithm-for-anomaly-detection/)

[Random Cut Forest (RCF) Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)

[Using Random Cut Forests for real-time anomaly detection in Amazon OpenSearch Service](https://aws.amazon.com/ko/blogs/big-data/using-random-cut-forests-for-real-time-anomaly-detection-in-amazon-opensearch-service/)

[Real Time Anomaly Detection in Open Distro for Elasticsearch](https://opensearch.org/blog/real-time-anomaly-detection-in-open-distro-for-elasticsearch/)

[Amazon SageMaker 모델 배포 방법 소개](https://www.youtube.com/watch?v=UigWJPfClcI&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr&index=2)
