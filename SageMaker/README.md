# SageMaker를 이용한 Anomaly Detection

## Sagemaker의 Random Cut Forest

Amazon SageMaker의 Built-in Algorithm중에 Random Cut Forest (RCF) 알고리즘은 데이터셋에 포함되어 있는 이상치들을 탐지하는 비지도 학습(unsupervised learning) 알고리즘입니다. 특히, Amazon SageMaker의 RCF 알고리즘은 각 데이터에 대해 이상치 스코어(anomaly score)를 부여합니다. 이상치 스코어가 낮을 경우 해당 데이터는 정상(normal)일 가능성이 높은 반면, 높은 스코어를 보일 경우 이 데이터는 이상(anomaly)일 가능성이 높다는 것을 나타냅니다.

Amazon SageMaker의 RCF 알고리즘은 우선 학습 데이터에서 임의로 샘플을 추출하여 작업을 수행합니다. 학습 데이터가 지나치게 커서 머신 한 대에 올리기 어려울 경우, [reservoir sampling 기법](https://en.wikipedia.org/wiki/Reservoir_sampling)을 이용하여 데이터 스트림으로부터 샘플 데이터를 효율적으로 추출합니다 이를 통해서 서브 샘플 데이터들이  Random Cut Forest를 구성하는 트리 각각에 분포하게 됩니다. 각각의 서브 샘플 데이터는 바운딩 박스를 랜덤하게 분할하는 식으로 만들어진 이진 트리를 구성하게 되는데, 이러한 트리 구성은 각 리프 노드(즉, 트리 맨 끝의 터미널 노드)가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질 때까지 계속 반복됩니다. 입력 데이터에 할당된 이상치 스코어는 포레스트를 구성하는 트리의 평균 깊이(depth)와 반비례합니다.

- Anomaly를 비지도 알고리즘을 통해 detecton 하기 위해 anomaly score를 부여: 점수는 case by case
- Random Cut Forest Algorithm
. 학습데이터에서 임의의 샘풀을 추출. 이때 reservior sampling 사용하여 데이터 스트림에서 데이터를 효율적으로 추출
. 서브 셈플 데이터들을 Random Cut Forest를 구성하는 이진트리 구성하고 이때 모든 leaf가 데이터를 하나만 담고 있는 바운딩 박스로 만들어질때까지 계속 반복
. 입력 데이터에 할당된 anomaly score는 forest를 구성하는 tree의 평균 depth와 반비례

## SageMaker Sample을 이용한 구현

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


## SageMaker RCF의 제한 및 조금 다른 구현

### Built-in 라이브러리의 배포방법

[rcf-serverless.ipynb](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/SageMaker/rcf-serverless.ipynb)는 AWS Sample을 조금 수정해서 테스트를 한 노트북입니다.

아래와 같이 RandomCutForest는 predict 함수를 제공하지 않습니다. 

```java
results = rcf.predict(
    taxi_data_numpy[:6]
)
AttributeError: 'RandomCutForest' object has no attribute 'predict'
```

Instacne type을 local로 지정할 수 없습니다. Built-in 알고리즘은 비슷한 정책을 가지고 있으며, XGBoost와 같이 Open 소스인 경우만 제공한다고 합니다.

```java
rcf_inference2 = rcf.deploy(initial_instance_count=1, instance_type="local")

ClientError: An error occurred (ValidationException) when calling the CreateEndpointConfig operation: 1 validation error detected: Value 'local' at 'productionVariants.1.member.instanceType' failed to satisfy constraint
```

따라서, SageMaker의 RandomCutForest는 오로지 SageMaker를 통해서만 제공 가능합니다. [Use the built-in Amazon SageMaker Random Cut Forest algorithm for anomaly detection](https://aws.amazon.com/ko/blogs/machine-learning/use-the-built-in-amazon-sagemaker-random-cut-forest-algorithm-for-anomaly-detection/)를 보면 2018년에는 ECR을 이용해 배포가 가능하였던것으로 보여지나, 현재는 ECR이나 Lambda의 내용을 확인할 수 없으므로 IoT등으로 확대는 어려운것으로 보여집니다.


### Serverless 구현

[rcf-serverless.ipynb](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/SageMaker/rcf-serverless.ipynb)에서는 [SageMaker Endpoint (Single Model Endpoint)](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/sagemaker/sm-special-webinar/lab_2_serving/2.1.Deploy.ipynb)를 참조하여 serverless(Lambda)로 설정을 바꿉니다.


모델이름은 SageMaker에서 확인하였습니다. 만약 코드로 확인하려면 [SageMaker - Random Cut Forest - 상세 라이브러리 설명](https://sagemaker.readthedocs.io/en/v2.20.0/algorithms/randomcutforest.html)을 통해 구현합니다.

아래와 같이 endpoint configureation을 생성합니다.

```java
endpoint_config_name = rcf_inference.endpoint

model_name = 'randomcutforest-2023-04-03-17-35-44-922'

endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 4096,
                "MaxConcurrency": 20,
            },            
        },
    ],
)
```

아래의 "sm_client.create_endpoint"로 endpoint 생성이 가능하지만 SageMaker Console에서 직접 생성하였습니다.

```java
endpoint_name = f"randomcutforest-2023-04-03-17-35-44-922"
endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name, 
    EndpointConfigName=endpoint_config_name
)
print(f"Creating Endpoint: {endpoint_response['EndpointArn']}")

```

아래와 같이 inference 결과를 확인합니다. 처음에는 cold start로 느리지만 두번째부터는 빠르게 결과를 리턴합니다.

```java
results = rcf_inference.predict(taxi_data_numpy)
scores = [datum["score"] for datum in results["scores"]]

# add scores to taxi data frame and print first few values
taxi_data["score"] = pd.Series(scores, index=taxi_data.index)
taxi_data.head()
```

거의 동일한 패턴의 결과를 얻습니다.

![image](https://user-images.githubusercontent.com/52392004/229644878-be9e6816-b489-4672-b3ac-86c7bb3424fd.png)

### SageMaker의 Anomaly Detection 결과 

[nyc_taxi.csv](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv)는 아래와 같습니다.

```java
timestamp,value
2014-07-01 00:00:00,10844
2014-07-01 00:30:00,8127
2014-07-01 01:00:00,6210
2014-07-01 01:30:00,4656
2014-07-01 02:00:00,3820
2014-07-01 02:30:00,2873
2014-07-01 03:00:00,2369
2014-07-01 03:30:00,2064
2014-07-01 04:00:00,2221
2014-07-01 04:30:00,2158
2014-07-01 05:00:00,2515
2014-07-01 05:30:00,4364
2014-07-01 06:00:00,6526
2014-07-01 06:30:00,11039
2014-07-01 07:00:00,13857
2014-07-01 07:30:00,15865
2014-07-01 08:00:00,17920
2014-07-01 08:30:00,20346
2014-07-01 09:00:00,19539
2014-07-01 09:30:00,20107
2014-07-01 10:00:00,18984
```

이때의 패턴은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/229644978-ed276ca7-e8fb-485e-8af6-3a78667a83cb.png)

Anomaly Score를 구하면 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/229645341-c42235d7-ed14-49d3-b642-d3fabc985da2.png)


여기서 Anaomaly Score에 대한 cutoff를 정의합니다.

```java
score_mean = taxi_data["score"].mean()
score_std = taxi_data["score"].std()
score_cutoff = score_mean + 3 * score_std

anomalies = taxi_data_subset[taxi_data_subset["score"] > score_cutoff]
anomalies
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/229645476-67636e47-021a-4f00-a640-7ae418371ac2.png)

이것을 plot으로 표시하면 아래와 같이 anomaly를 구분할 수 있습니다.

![image](https://user-images.githubusercontent.com/52392004/229645556-4520e78c-637a-45e7-83b2-b4b4a0ab78e8.png)


## Reference

[An Introduction to SageMaker Random Cut Forests](https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest.html#An-Introduction-to-SageMaker-Random-Cut-Forests)

[An Introduction to SageMaker Random Cut Forests - Notebook](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest.ipynb)

[Amazon SageMaker 모델 배포 실습 - 김대근](https://www.youtube.com/watch?v=1rr9GgJelBU&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr&index=6)

[SageMaker Endpoint (Single Model Endpoint)](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/sagemaker/sm-special-webinar/lab_2_serving/2.1.Deploy.ipynb)

[Deploy SageMaker Serverless Endpoint](https://github.com/aws-samples/sm-model-serving-patterns/blob/main/key_features/ptn_4.2_serverless-inference/serverless_endpoint_kornlp_korsts.ipynb)

[SageMaker - Random Cut Forest - 상세 라이브러리 설명](https://sagemaker.readthedocs.io/en/v2.20.0/algorithms/randomcutforest.html)

[Use the built-in Amazon SageMaker Random Cut Forest algorithm for anomaly detection](https://aws.amazon.com/ko/blogs/machine-learning/use-the-built-in-amazon-sagemaker-random-cut-forest-algorithm-for-anomaly-detection/)
