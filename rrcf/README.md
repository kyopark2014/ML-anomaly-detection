# RRCF를 이용한 Anomaly Detection

## RRCF Basic

[rrcf-basic](https://github.com/kyopark2014/ML-anomaly-detection/blob/main/rrcf/rrcf-basic.md)은 기본적인 RRCF에 대해 설명합니다.

## RRCF를 이용한 NYC Taxi 분석

[rrcf-anomaly-detection.ipynb](https://github.com/kyopark2014/ML-anomaly-detection/tree/main/rrcf)에서는 RRCF로 NYC의 텍시정보에 대한 anomaly를 분석합니다. 

데이터를 로딩합니다.

```python
import numpy as np
import pandas as pd
import rrcf
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt
import seaborn as sns

# Read data
taxi = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv',
                   index_col=0)
taxi.index = pd.to_datetime(taxi.index)
data = taxi['value'].astype(float).values
```

RRCF를 수행합니다.

```python
# Set tree parameters
num_trees = 200
shingle_size = 48
tree_size = 1000

# Use the "shingle" generator to create rolling window
points = rrcf.shingle(data, size=shingle_size)
points = np.vstack([point for point in points])
n = points.shape[0]
sample_size_range = (n // tree_size, tree_size)

forest = []
while len(forest) < num_trees:
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = [rrcf.RCTree(points[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)
    
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)

for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
    
avg_codisp /= index
avg_codisp.index = taxi.iloc[(shingle_size - 1):].index
```

Isolation Forest를 수행합니다.

```python
contamination = taxi['event'].sum()/len(taxi)

IF = IsolationForest(n_estimators=num_trees,
                     contamination=contamination,
                     random_state=0)

IF.fit(points)
if_scores = IF.score_samples(points)
if_scores = pd.Series(-if_scores,
                      index=(taxi
                             .iloc[(shingle_size - 1):]
                             .index))
```

결과를 표시합니다.

```python
# Normalize anomaly scores to (0, 1)
avg_codisp = ((avg_codisp - avg_codisp.min())
              / (avg_codisp.max() - avg_codisp.min()))
if_scores = ((if_scores - if_scores.min())
              / (if_scores.max() - if_scores.min()))
              
fig, ax = plt.subplots(2, figsize=(10, 6))
(taxi['value'] / 1000).plot(ax=ax[0], color='0.5',
                            alpha=0.8)
if_scores.plot(ax=ax[1], color='#7EBDE6', alpha=0.8,
               label='IF')
avg_codisp.plot(ax=ax[1], color='#E8685D', alpha=0.8,
                label='RRCF')
ax[1].legend(frameon=True, loc=2, fontsize=12)

for event, duration in events.items():
    start, end = duration
    ax[0].axvspan(start, end, alpha=0.3,
                  color='springgreen')

ax[0].set_xlabel('')
ax[1].set_xlabel('')

ax[0].set_ylabel('Taxi passengers (thousands)', size=13)
ax[1].set_ylabel('Normalized Anomaly Score', size=13)
ax[0].set_title('Anomaly detection on NYC Taxi data',
                size=14)

ax[0].xaxis.set_ticklabels([])

ax[0].set_xlim(taxi.index[0], taxi.index[-1])
ax[1].set_xlim(taxi.index[0], taxi.index[-1])
plt.tight_layout()
```

이때의 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/229655838-8699fe0d-3dc0-4beb-a7f3-cdf400c49946.png)

## Reference

[rrcf](https://klabum.github.io/rrcf/)

[rrcf-github](https://github.com/kLabUM/rrcf)
