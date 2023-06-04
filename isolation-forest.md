# Isolation Forest

Isolation Forest는 [Random Forest](https://github.com/kyopark2014/ML-Algorithms/blob/main/random-forest.md)처럼 Tree 구조에 기반한 비정상 검출 알고리즘(Anomaly detection algorithm)으로 비정상 적인 데이터를 검출하기 위하여 비지도학습(unsupervised learning)을 이용합니다. 

[Isolation Forest (2008)](https://dl.acm.org/doi/10.1109/ICDM.2008.17)는 비정상 데이터를 분류하기 위하여 를 이용합니다.

Isolation Forest recursively partitions the hyperspace of features to construct trees (the leaf nodes are feature samples), and assigns an anomaly score to each data point based on the sample tree heights. It is a batch processing method.
  
![image](https://user-images.githubusercontent.com/52392004/228095136-e95a1976-b4f7-4552-affa-83723dc2b40e.png)


### [sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)


Isolation Forest는 하나의 feature를 랜덤하게 선택함으로써 observation들을 isolate 합니다. 이후 선택된 feature들의 최대, 최소값 사이에서 랜덤하게 split value를 선택합니다. 

recursive한 partitioning은 트리구조로 표현될수 있기 때문에, 하나의 나누어진 수는 루트 노드부터 terminating 노드의 path 길이랑 같다. 

Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.



This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

## Reference 

[How to use the Isolation Forest model for outlier detection](https://practicaldatascience.co.uk/machine-learning/how-to-use-the-isolation-forest-model-for-outlier-detection)
