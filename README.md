# Anomaly Detection

이상(Anomaly)라는 것은 정형화된 데이터 또는 특정 패턴을 보이는 데이터와 다른 형태로 나타나는 데이터를 말합니다. 예를 들면, 시계열 데이터에서 예측 불가한 순간 급변 형태의 값을 보이는 것을 생각해볼 수 있습니다. 또 예측치와 맞지 않는 결과, 분류가 안되는 데이터 등도 ‘이상’의 예에 해당됩니다. 데이터넷에 이러한 이상치들이 포함되어 있을 경우 머신 러닝 작업의 복잡도가 높아집니다. 왜냐하면 “정상적인 (regular)” 데이터의 경우 단순한 모델로 표현될 수 있기 때문입니다.


- 사람은 어떤것들의 순서가 바뀐다거나 하는것에 놀라운 직관을 가지고 있습니다. 
- anomaly 정의는 “a data point lying in a low-density region" 입니다. 오른쪽 끝의 점과 같이 낮은 밀도로 발생하는곳을 anomaly로 얘기할 수 있습니다. anomaly 자체가 나쁘것이 아니라, 낮은 빈도로 발생하거나 정상 범위를 벗어난것 정도로 생각해야 합니다.

![image](https://user-images.githubusercontent.com/52392004/228092087-fe43cfa8-d6b6-4f46-bb1c-a52bf587dcec.png)

## RCF 



lightweight density estimation에 해당합니다. 


#### Isolation Forest 

The [Isolation Forest (2008)](https://dl.acm.org/doi/10.1109/ICDM.2008.17) method [1] uses random forests to isolate anomalous data.

Isolation Forest recursively partitions the hyperspace of features to construct trees (the leaf nodes are feature samples), and assigns an anomaly score to each data point based on the sample tree heights. It is a batch processing method.
  
![image](https://user-images.githubusercontent.com/52392004/228095136-e95a1976-b4f7-4552-affa-83723dc2b40e.png)

  
#### Random Cut Forest

[Robust random cut forest based anomaly detection on streams - 2016](https://dl.acm.org/doi/10.5555/3045390.3045676)

[Robust Random Cut Forest Based Anomaly Detection on Streams](https://www.semanticscholar.org/paper/Robust-Random-Cut-Forest-Based-Anomaly-Detection-on-Guha-Mishra/ecb365ef9b67cd5540cc4c53035a6a7bd88678f9?p2df)

![image](https://user-images.githubusercontent.com/52392004/228092706-48d72e27-5db5-4214-9a70-6fcf68f1865e.png)

여기서, Reservoir Sampling은 순차적으로 한 번에 하나의 샘플만 볼 수 있고 전체 샘플 개수를 모르는 상황에서 무작위 추출을 하는 방법입니다. 

shingle (조약돌)

[랜덤 포레스트 (Random Forest)](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)



- RCF는 제한된 메모리와 컴퓨팅을 가지고 Isolation Forest로 데이터 스트링을 처리합니다. (adapted Isolation Forest to work on data streams with bounded memory and lightweight compute.)

- RCF incrementally updates the forest on each feature sample and interleaves training and inference. RCF also emits an anomaly score for each feature sample. The RCF estimator has been proven as it have been used in production settings


- Putting RCFs to practice for real time anomaly detection in a “set it and forget it” environment requires additional work; we list them down here. First, RCFs emit an anomaly score that is hard to reason about for the user, its magnitude is a function of the data timeseries on which the RCF is trained. We need an additional learning primitive that continuously learns the baseline anomaly score distribution to detect large score values it is a classifier function that maps the anomaly score to a boolean outcome (anomaly or not). Note that this classifier is different from RCF itself. RCF isolates anomalies (i.e., not the baseline) and gives a score timeseries that captures and quantifies anomalous events; the classifier can also be simple, since it operates on one-dimensional positive data. The classifier needs to work with small amounts of data, so it does not block anomaly detection. The classifier emits two values to aid the user: (1) anomaly grade, quantifying severity of the anomaly, and (2) confidence, quantifying the amount of data seen and RCF size.

Second, RCFs require training time to learn the initial distribution (i.e., the forest). An RCF requires hundreds of samples, which takes time to arrive at the Elasticsearch cluster. This prevents both interactive exploration of anomalies on current and past data (e.g., using queries), and delays the time-to-detection. In practice, users would define and turn on a detector after ingesting some data - we can leverage this to train an initial model. Further, to support interactive ad-hoc exploration, we need to a fast RCF construction primitive on data at rest.  



## Anomaly detection의 구성 

1) A RCF model for estimating the density of an input data stream
2) A thresholding model for determining if a point should be labeled as anomalous


## Anomaly grade and a confidence score

The anomaly grade is a measurement of the severity of an anomaly on a scale from zero to one. A zero anomaly grade indicates that the corresponding data point is normal.
Any non-zero grade means that the anomaly score output by RCF exceeds the calculated score threshold, and therefore indicates the presence of an anomaly. 

The confidence score is a measurement of the probability that the anomaly detection model correctly reports an anomaly within the algorithm’s inherent error bounds. 



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
