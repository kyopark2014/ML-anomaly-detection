# SageMaker를 이용한 Anomaly Detection

## Sagemaker의 Random Cut Forest

Amazon SageMaker의 Built-in Algorithm중에 Random Cut Forest (RCF) 알고리즘은 데이터셋에 포함되어 있는 이상치들을 탐지하는 비지도 학습(unsupervised learning) 알고리즘입니다. 특히, Amazon SageMaker의 RCF 알고리즘은 각 데이터에 대해 이상치 스코어(anomaly score)를 부여합니다. 이상치 스코어가 낮을 경우 해당 데이터는 정상(normal)일 가능성이 높은 반면, 높은 스코어를 보일 경우 이 데이터는 이상(anomaly)일 가능성이 높다는 것을 나타냅니다.

Amazon SageMaker의 RCF 알고리즘은 우선 학습 데이터에서 임의로 샘플을 추출하여 작업을 수행합니다. 학습 데이터가 지나치게 커서 머신 한 대에 올리기 어려울 경우, [reservoir sampling 기법](https://en.wikipedia.org/wiki/Reservoir_sampling)을 이용하여 데이터 스트림으로부터 샘플 데이터를 효율적으로 추출합니다 이를 통해서 서브 샘플 데이터들이  Random Cut Forest를 구성하는 트리 각각에 분포하게 됩니다. 각각의 서브 샘플 데이터는 바운딩 박스를 랜덤하게 분할하는 식으로 만들어진 이진 트리를 구성하게 되는데, 이러한 트리 구성은 각 리프 노드(즉, 트리 맨 끝의 터미널 노드)가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질 때까지 계속 반복됩니다. 입력 데이터에 할당된 이상치 스코어는 포레스트를 구성하는 트리의 평균 깊이(depth)와 반비례합니다.

- Anomaly를 비지도 알고리즘을 통해 detecton 하기 위해 anomaly score를 부여: 점수는 case by case
- Random Cut Forest Algorithm
. 학습데이터에서 임의의 샘풀을 추출. 이때 reservior sampling 사용하여 데이터 스트림에서 데이터를 효율적으로 추출
. 서브 셈플 데이터들을 Random Cut Forest를 구성하는 이진트리 구성하고 이때 모든 leaf가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질때까지 계속 반복
. 입력 데이터에 할당된 anomaly score는 forest를 구성하는 tree의 평균 depth와 반비례

## 구현 방법

[SageMaker-RCF.ipynb](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/SageMaker/SageMaker-RCF.ipynb)는 SageMaker의 RCF 라이브러리를 이용하여 Anomaly Detection을 수행합니다. 


### 모델 학습

아래와 같이 RandomCutForest를 선언하고 모델을 학습합니다.

```java
from sagemaker import RandomCutForest

session = sagemaker.Session()

rcf = RandomCutForest(
    role=execution_role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    data_location=f"s3://{bucket}/{prefix}/",
    output_path=f"s3://{bucket}/{prefix}/output",
    num_samples_per_tree=512,
    num_trees=50,
)

rcf.fit(rcf.record_set(taxi_data.value.to_numpy().reshape(-1, 1)))
```

### 추론 (Inference)

아래와 같이 추론을 수행합니다. 

```java
rcf_inference = rcf.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

results = rcf_inference.predict(taxi_data_numpy)
scores = [datum["score"] for datum in results["scores"]]

# add scores to taxi data frame and print first few values
taxi_data["score"] = pd.Series(scores, index=taxi_data.index)
taxi_data.head()
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/229640338-795fc0a4-1a9f-4b8c-b6c3-b53e6cac72a9.png)





## Reference

[An Introduction to SageMaker Random Cut Forests](https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest.html#An-Introduction-to-SageMaker-Random-Cut-Forests)

[An Introduction to SageMaker Random Cut Forests - Notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest.ipynb)
