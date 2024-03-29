# Random Cut Forest (RCF) 란? 

- RCF(Random Cut Forest)은 데이터셋에서 이상치(outlier)를 탐지하는 비지도 학습 알고리즘입니다.

- RCF는 lightweight density estimation에 해당합니다. 

## Isolation Forest

[Isolation Forest](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/isolation-forest.md)에 대해 설명합니다.
  
## Random Cut Forest

[Robust random cut forest based anomaly detection on streams - 2016](https://dl.acm.org/doi/10.5555/3045390.3045676)

[Robust Random Cut Forest Based Anomaly Detection on Streams](https://www.semanticscholar.org/paper/Robust-Random-Cut-Forest-Based-Anomaly-Detection-on-Guha-Mishra/ecb365ef9b67cd5540cc4c53035a6a7bd88678f9?p2df)

![image](https://user-images.githubusercontent.com/52392004/228092706-48d72e27-5db5-4214-9a70-6fcf68f1865e.png)


### Reservoir Sampling

Reservoir Sampling은 순차적으로 한 번에 하나의 샘플만 볼 수 있고 전체 샘플 개수를 모르는 상황에서 무작위 추출을 하는 방법입니다. 


### Shingling

Shingling을 이용하여 전처리를 함으로써 정확도를 향상시킵니다.

이상 탐지 알고리즘에서 shingling은 연속성을 지닌 데이터에서 s길이 만큼의 시퀀스를 s차원의 백터로 변환함으로써 1차원 데이터를 s차원 데이터로 바꾸는 기술입니다. 이렇게 하면 정기적으로 나타나는 변화를 탐지하기 좋고, 이상치 스코어의 원래 값에서 작은 규모의 노이즈를 필터링할 수 있습니다. shingling 사이즈가 너무 작을 경우에 Random Cut Forest 알고리즘이 데이터의 작은 변화에도 심하게 영향을 받을 수 있고, 너무 크면 이상치를 탐지하지 못할 수 있습니다






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

## Reference


