Read Stream Data from S3 in JSON format
https://towardsdatascience.com/how-to-read-data-files-on-s3-from-amazon-sagemaker-f288850bfe8f


```python
import boto3
import json
import pandas as pd
from sagemaker import get_execution_role

bucket = 'raz-eu-central-1-tutorial'
prefix = 'kinesis-analytics/'


role = get_execution_role()

```


```python
%%time
#list the content of the bucket / prefix
conn = boto3.client('s3')
contents = conn.list_objects(Bucket=bucket, Prefix=prefix)['Contents']

data = pd.DataFrame()

for f in contents:
    response = conn.get_object(Bucket=bucket, Key=f['Key'])
    if response['ContentType'] == 'application/octet-stream':
        #print(f['Key'])
        body = response['Body']
        #jsonObject = body.read()
        #print(jsonObject)
        temp = pd.read_json(body,lines=True)
        data = data.append(temp)
data.head()
```

    CPU times: user 12.9 s, sys: 62.3 ms, total: 12.9 s
    Wall time: 32.9 s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRESSURE</th>
      <th>EVENT_TIME</th>
      <th>TEMPERATURE</th>
      <th>HUMIDITY</th>
      <th>ANOMALY_SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>989.938041</td>
      <td>2020-11-22 22:00:31.760894</td>
      <td>20.38</td>
      <td>42.347507</td>
      <td>1.307758</td>
    </tr>
    <tr>
      <th>1</th>
      <td>989.916192</td>
      <td>2020-11-22 22:00:32.872372</td>
      <td>20.39</td>
      <td>42.359085</td>
      <td>1.289527</td>
    </tr>
    <tr>
      <th>2</th>
      <td>989.885988</td>
      <td>2020-11-22 22:00:33.990491</td>
      <td>20.38</td>
      <td>42.370639</td>
      <td>1.196880</td>
    </tr>
    <tr>
      <th>3</th>
      <td>989.864618</td>
      <td>2020-11-22 22:00:35.105371</td>
      <td>20.38</td>
      <td>42.370614</td>
      <td>1.304555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>989.942219</td>
      <td>2020-11-22 22:00:36.217284</td>
      <td>20.39</td>
      <td>42.370651</td>
      <td>1.298816</td>
    </tr>
  </tbody>
</table>
</div>




```python
#show number of rows and columns
%%time
data.shape
```

    UsageError: Line magic function `%%time` not found.


Scatter plot of temperature and Anomaly score
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html


```python
ax1 = data.plot.scatter(x='TEMPERATURE',
                      y='ANOMALY_SCORE',
                      c='DarkBlue')
```


![png](output_5_0.png)


Binarize the anomaly score column with the threshold of 2 and move the column to the front of the table to sever as label column
https://stackoverflow.com/questions/40717156/binarize-integer-in-a-pandas-dataframe


```python
data['ANOMALY_SCORE'] = (data['ANOMALY_SCORE']>2.0).astype(int)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRESSURE</th>
      <th>EVENT_TIME</th>
      <th>TEMPERATURE</th>
      <th>HUMIDITY</th>
      <th>ANOMALY_SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>989.938041</td>
      <td>2020-11-22 22:00:31.760894</td>
      <td>20.38</td>
      <td>42.347507</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>989.916192</td>
      <td>2020-11-22 22:00:32.872372</td>
      <td>20.39</td>
      <td>42.359085</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>989.885988</td>
      <td>2020-11-22 22:00:33.990491</td>
      <td>20.38</td>
      <td>42.370639</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>989.864618</td>
      <td>2020-11-22 22:00:35.105371</td>
      <td>20.38</td>
      <td>42.370614</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>989.942219</td>
      <td>2020-11-22 22:00:36.217284</td>
      <td>20.39</td>
      <td>42.370651</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Count positive and negative rows in data frame
https://stackoverflow.com/questions/17322109/get-dataframe-row-count-based-on-conditions

```python
#postivie labels
data[(data['ANOMALY_SCORE']==1)].count()
```




    PRESSURE         79
    EVENT_TIME       79
    TEMPERATURE      79
    HUMIDITY         79
    ANOMALY_SCORE    79
    dtype: int64




```python
#negative labels
data[(data['ANOMALY_SCORE']==0)].count()
```




    PRESSURE         30731
    EVENT_TIME       30731
    TEMPERATURE      30731
    HUMIDITY         30731
    ANOMALY_SCORE    30731
    dtype: int64



Over sample postive lables with SMOTE: Synthetic minority over-sampling technique
https://medium.com/@peterkoman_68710/smotenc-smote-for-pandas-dataframe-deb2c2128da5,https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTENC.html


```python
#import sys
#!{sys.executable} -m pip install imblearn
from imblearn.over_sampling import SMOTENC


y = data['ANOMALY_SCORE']
X = data.drop(['ANOMALY_SCORE','EVENT_TIME'], axis=1)

smote = SMOTENC(random_state=42,categorical_features=[1])

X_res, y_res = smote.fit_sample(X, y)

X_res = pd.DataFrame(X_res, columns=X.columns)
y_res = pd.DataFrame(y_res, columns=['ANOMALY_SCORE'])
df_res = y_res.merge(X_res, left_index=True, right_index=True)

df_res.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ANOMALY_SCORE</th>
      <th>PRESSURE</th>
      <th>TEMPERATURE</th>
      <th>HUMIDITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>989.938041</td>
      <td>20.38</td>
      <td>42.347507</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>989.916192</td>
      <td>20.39</td>
      <td>42.359085</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>989.885988</td>
      <td>20.38</td>
      <td>42.370639</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>989.864618</td>
      <td>20.38</td>
      <td>42.370614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>989.942219</td>
      <td>20.39</td>
      <td>42.370651</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_res.count()
```




    ANOMALY_SCORE    61462
    PRESSURE         61462
    TEMPERATURE      61462
    HUMIDITY         61462
    dtype: int64




```python
df_res[(df_res['ANOMALY_SCORE']==1)].count()
```




    ANOMALY_SCORE    30731
    PRESSURE         30731
    TEMPERATURE      30731
    HUMIDITY         30731
    dtype: int64



Plot histogram to show distribution 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html


```python
df_res['TEMPERATURE'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f6eff0e29b0>




![png](output_16_1.png)



```python
Preapare data for training the linear learner algorithm -

https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
https://aws.amazon.com/blogs/machine-learning/build-multiclass-classifiers-with-amazon-sagemaker-linear-learner/

Spilt data 80 / 20 for training and Validation
```


      File "<ipython-input-15-d2f160daeb39>", line 1
        Preapare data for training the linear learner algorithm - https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
                    ^
    SyntaxError: invalid syntax




```python
from sklearn.model_selection import train_test_split
import numpy as np

arr = df_res.to_numpy().astype('float32')

covtype_labels = arr[:,0]
covtype_features = arr[:,1:]

# shuffle and split into train and test sets
np.random.seed(0)

train_features, test_features, train_labels, test_labels = train_test_split(covtype_features, covtype_labels, test_size=0.2)

# further split the test set into validation and test sets
val_features, test_features, val_labels, test_labels = train_test_split(test_features, test_labels, test_size=0.5)
```


```python
#plot distrubition to check it is consistent
pd.DataFrame(data=train_features)[1].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f6efe97e0b8>




![png](output_19_1.png)



```python

```

Training a linear learner classifier using the Amazon SageMaker 
https://aws.amazon.com/blogs/machine-learning/build-multiclass-classifiers-with-amazon-sagemaker-linear-learner/


```python
import sagemaker
from sagemaker.amazon.amazon_estimator import RecordSet


# instantiate the LinearLearner estimator object
#https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
    
binary_estimator = sagemaker.LinearLearner(role=role,
                                               train_instance_count=1,
                                               train_instance_type='ml.m4.xlarge',
                                               predictor_type='binary_classifier')

# wrap data in RecordSet objects
train_records = binary_estimator.record_set(train_features, train_labels, channel='train')
val_records = binary_estimator.record_set(val_features, val_labels, channel='validation')
test_records = binary_estimator.record_set(test_features, test_labels, channel='test')


```


```python
# start a training job
binary_estimator.fit([train_records, val_records, test_records])
```

    'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.
    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.
    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.
    'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.


    2020-11-24 18:50:45 Starting - Starting the training job...
    2020-11-24 18:50:48 Starting - Launching requested ML instances......
    2020-11-24 18:51:55 Starting - Preparing the instances for training......
    2020-11-24 18:52:50 Downloading - Downloading input data...
    2020-11-24 18:53:37 Training - Downloading the training image..[34mDocker entrypoint called with argument(s): train[0m
    [34mRunning default environment configuration script[0m
    [34m[11/24/2020 18:53:54 INFO 140513285547840] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'auto', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [34m[11/24/2020 18:53:54 INFO 140513285547840] Merging with provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'3', u'mini_batch_size': u'1000', u'predictor_type': u'binary_classifier'}[0m
    [34m[11/24/2020 18:53:54 INFO 140513285547840] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'3', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'binary_classifier', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
    [34m[11/24/2020 18:53:54 WARNING 140513285547840] Loggers have already been setup.[0m
    [34mProcess 1 is a worker.[0m
    [34m[11/24/2020 18:53:54 INFO 140513285547840] Using default worker.[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] Checkpoint loading and saving are disabled.[0m
    [34m[2020-11-24 18:53:55.173] [tensorio] [warning] TensorIO is already initialized; ignoring the initialization routine.[0m
    [34m[2020-11-24 18:53:55.178] [tensorio] [warning] TensorIO is already initialized; ignoring the initialization routine.[0m
    [34m[2020-11-24 18:53:55.211] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 41, "num_examples": 1, "num_bytes": 56000}[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] Create Store: local[0m
    [34m[2020-11-24 18:53:55.286] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 73, "num_examples": 11, "num_bytes": 616000}[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] Scaler algorithm parameters
     <algorithm.scaler.ScalerAlgorithmStable object at 0x7fcb6d4361d0>[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] Scaling model computed with parameters:
     {'stdev_weight': [0m
    [34m[1.693133 2.071373 1.881285][0m
    [34m<NDArray 3 @cpu(0)>, 'stdev_label': None, 'mean_label': None, 'mean_weight': [0m
    [34m[990.113     21.414627  40.87438 ][0m
    [34m<NDArray 3 @cpu(0)>}[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] nvidia-smi took: 0.0251948833466 secs to identify 0 gpus[0m
    [34m[11/24/2020 18:53:55 INFO 140513285547840] Number of GPUs being used: 0[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 12, "sum": 12.0, "min": 12}, "Total Records Seen": {"count": 1, "max": 12000, "sum": 12000.0, "min": 12000}, "Max Records Seen Between Resets": {"count": 1, "max": 11000, "sum": 11000.0, "min": 11000}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1606244035.392138, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1606244035.392093}
    [0m
    [34m[2020-11-24 18:53:56.971] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 3, "duration": 1579, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5956824210030692, "sum": 0.5956824210030692, "min": 0.5956824210030692}}, "EndTime": 1606244036.971848, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.971769}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5953751656668527, "sum": 0.5953751656668527, "min": 0.5953751656668527}}, "EndTime": 1606244036.971978, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.97196}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5878506855867347, "sum": 0.5878506855867347, "min": 0.5878506855867347}}, "EndTime": 1606244036.972026, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972015}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5863856331961496, "sum": 0.5863856331961496, "min": 0.5863856331961496}}, "EndTime": 1606244036.972076, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972059}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24566620433573821, "sum": 0.24566620433573821, "min": 0.24566620433573821}}, "EndTime": 1606244036.972139, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972122}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24868228024852518, "sum": 0.24868228024852518, "min": 0.24868228024852518}}, "EndTime": 1606244036.972194, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972182}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24839271794533244, "sum": 0.24839271794533244, "min": 0.24839271794533244}}, "EndTime": 1606244036.972231, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972222}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24411407159299267, "sum": 0.24411407159299267, "min": 0.24411407159299267}}, "EndTime": 1606244036.972267, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972257}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5936469321737484, "sum": 0.5936469321737484, "min": 0.5936469321737484}}, "EndTime": 1606244036.972302, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972293}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6254722887934471, "sum": 0.6254722887934471, "min": 0.6254722887934471}}, "EndTime": 1606244036.972351, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972334}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5983840269750478, "sum": 0.5983840269750478, "min": 0.5983840269750478}}, "EndTime": 1606244036.972416, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972398}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5674726886360013, "sum": 0.5674726886360013, "min": 0.5674726886360013}}, "EndTime": 1606244036.972462, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972451}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24344759166483976, "sum": 0.24344759166483976, "min": 0.24344759166483976}}, "EndTime": 1606244036.972498, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972488}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25063257240762515, "sum": 0.25063257240762515, "min": 0.25063257240762515}}, "EndTime": 1606244036.972532, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972523}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23965391509386957, "sum": 0.23965391509386957, "min": 0.23965391509386957}}, "EndTime": 1606244036.972566, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972557}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24782748973612884, "sum": 0.24782748973612884, "min": 0.24782748973612884}}, "EndTime": 1606244036.972601, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972591}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5935600835060587, "sum": 0.5935600835060587, "min": 0.5935600835060587}}, "EndTime": 1606244036.972641, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972631}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6225448135064573, "sum": 0.6225448135064573, "min": 0.6225448135064573}}, "EndTime": 1606244036.972676, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972665}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6118164598114636, "sum": 0.6118164598114636, "min": 0.6118164598114636}}, "EndTime": 1606244036.972738, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.97272}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5725872914839764, "sum": 0.5725872914839764, "min": 0.5725872914839764}}, "EndTime": 1606244036.9728, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972784}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48741709868761957, "sum": 0.48741709868761957, "min": 0.48741709868761957}}, "EndTime": 1606244036.972854, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972842}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48498278310347576, "sum": 0.48498278310347576, "min": 0.48498278310347576}}, "EndTime": 1606244036.972891, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972881}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4849574528908243, "sum": 0.4849574528908243, "min": 0.4849574528908243}}, "EndTime": 1606244036.972925, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972916}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48525076792191485, "sum": 0.48525076792191485, "min": 0.48525076792191485}}, "EndTime": 1606244036.972959, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972949}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6901275323361766, "sum": 0.6901275323361766, "min": 0.6901275323361766}}, "EndTime": 1606244036.972993, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.972983}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6915164172114158, "sum": 0.6915164172114158, "min": 0.6915164172114158}}, "EndTime": 1606244036.973027, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973017}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6885690805863361, "sum": 0.6885690805863361, "min": 0.6885690805863361}}, "EndTime": 1606244036.97306, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973051}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902749720982143, "sum": 0.6902749720982143, "min": 0.6902749720982143}}, "EndTime": 1606244036.973121, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973104}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6979549548090721, "sum": 0.6979549548090721, "min": 0.6979549548090721}}, "EndTime": 1606244036.973183, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973167}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6978170278121014, "sum": 0.6978170278121014, "min": 0.6978170278121014}}, "EndTime": 1606244036.97324, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973226}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6977945170499841, "sum": 0.6977945170499841, "min": 0.6977945170499841}}, "EndTime": 1606244036.973278, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973268}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6978982419383769, "sum": 0.6978982419383769, "min": 0.6978982419383769}}, "EndTime": 1606244036.973312, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244036.973303}
    [0m
    [34m[11/24/2020 18:53:56 INFO 140513285547840] #quality_metric: host=algo-1, epoch=0, train binary_classification_cross_entropy_objective <loss>=0.595682421003[0m
    [34m[2020-11-24 18:53:56.999] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 0, "duration": 1825, "num_examples": 1, "num_bytes": 56000}[0m
    [34m[2020-11-24 18:53:57.293] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 2, "duration": 293, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5082786043653107, "sum": 0.5082786043653107, "min": 0.5082786043653107}}, "EndTime": 1606244037.581311, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581218}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5048641399727402, "sum": 0.5048641399727402, "min": 0.5048641399727402}}, "EndTime": 1606244037.581407, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581391}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5026949141726136, "sum": 0.5026949141726136, "min": 0.5026949141726136}}, "EndTime": 1606244037.581474, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581455}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4974651938729204, "sum": 0.4974651938729204, "min": 0.4974651938729204}}, "EndTime": 1606244037.581542, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581523}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17791455210205634, "sum": 0.17791455210205634, "min": 0.17791455210205634}}, "EndTime": 1606244037.581616, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581597}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17714810270538478, "sum": 0.17714810270538478, "min": 0.17714810270538478}}, "EndTime": 1606244037.581685, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581668}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.18006382056800155, "sum": 0.18006382056800155, "min": 0.18006382056800155}}, "EndTime": 1606244037.581744, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581728}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17566178592144377, "sum": 0.17566178592144377, "min": 0.17566178592144377}}, "EndTime": 1606244037.58181, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581791}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.507887804581915, "sum": 0.507887804581915, "min": 0.507887804581915}}, "EndTime": 1606244037.581875, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581858}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5305947832280714, "sum": 0.5305947832280714, "min": 0.5305947832280714}}, "EndTime": 1606244037.58195, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.58193}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.511765764002454, "sum": 0.511765764002454, "min": 0.511765764002454}}, "EndTime": 1606244037.582017, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.581998}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48307357624226505, "sum": 0.48307357624226505, "min": 0.48307357624226505}}, "EndTime": 1606244037.582083, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582065}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1773518519800337, "sum": 0.1773518519800337, "min": 0.1773518519800337}}, "EndTime": 1606244037.582151, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582131}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17840260401091745, "sum": 0.17840260401091745, "min": 0.17840260401091745}}, "EndTime": 1606244037.582218, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582199}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17606066664383227, "sum": 0.17606066664383227, "min": 0.17606066664383227}}, "EndTime": 1606244037.582284, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582265}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.17758404545236248, "sum": 0.17758404545236248, "min": 0.17758404545236248}}, "EndTime": 1606244037.582351, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582332}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5268539940150987, "sum": 0.5268539940150987, "min": 0.5268539940150987}}, "EndTime": 1606244037.582421, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582402}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5456268466958153, "sum": 0.5456268466958153, "min": 0.5456268466958153}}, "EndTime": 1606244037.58248, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582461}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5420105931350008, "sum": 0.5420105931350008, "min": 0.5420105931350008}}, "EndTime": 1606244037.582554, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582535}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.50565076083808, "sum": 0.50565076083808, "min": 0.50565076083808}}, "EndTime": 1606244037.582623, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582604}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4682328720415527, "sum": 0.4682328720415527, "min": 0.4682328720415527}}, "EndTime": 1606244037.582695, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582676}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48523801163668123, "sum": 0.48523801163668123, "min": 0.48523801163668123}}, "EndTime": 1606244037.582768, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.58275}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.48452776292229255, "sum": 0.48452776292229255, "min": 0.48452776292229255}}, "EndTime": 1606244037.582836, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582818}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4826424422110481, "sum": 0.4826424422110481, "min": 0.4826424422110481}}, "EndTime": 1606244037.582901, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582883}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895903374137842, "sum": 0.6895903374137842, "min": 0.6895903374137842}}, "EndTime": 1606244037.582958, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.582939}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6901829623648448, "sum": 0.6901829623648448, "min": 0.6901829623648448}}, "EndTime": 1606244037.583024, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583006}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892435537500254, "sum": 0.6892435537500254, "min": 0.6892435537500254}}, "EndTime": 1606244037.583092, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583074}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895000061770041, "sum": 0.6895000061770041, "min": 0.6895000061770041}}, "EndTime": 1606244037.583158, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583139}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6929185658140521, "sum": 0.6929185658140521, "min": 0.6929185658140521}}, "EndTime": 1606244037.583211, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583194}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6862645867667677, "sum": 0.6862645867667677, "min": 0.6862645867667677}}, "EndTime": 1606244037.583274, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583256}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.688371925838134, "sum": 0.688371925838134, "min": 0.688371925838134}}, "EndTime": 1606244037.583338, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.58332}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894469033069655, "sum": 0.6894469033069655, "min": 0.6894469033069655}}, "EndTime": 1606244037.58341, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244037.583392}
    [0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] #quality_metric: host=algo-1, epoch=0, validation binary_classification_cross_entropy_objective <loss>=0.508278604365[0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=sampled_accuracy, value=0.943377806704[0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] Epoch 0: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] Saving model for epoch: 0[0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] Saved checkpoint to "/tmp/tmpsB6Tvb/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] #progress_metric: host=algo-1, completed 6 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 62, "sum": 62.0, "min": 62}, "Total Records Seen": {"count": 1, "max": 61169, "sum": 61169.0, "min": 61169}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1606244037.591622, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1606244035.392403}
    [0m
    [34m[11/24/2020 18:53:57 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=22356.2318742 records/second[0m
    [34m[2020-11-24 18:53:59.760] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 5, "duration": 2168, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4501761648995536, "sum": 0.4501761648995536, "min": 0.4501761648995536}}, "EndTime": 1606244039.760911, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.760801}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4439723224250638, "sum": 0.4439723224250638, "min": 0.4439723224250638}}, "EndTime": 1606244039.76102, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.760997}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4460249085718272, "sum": 0.4460249085718272, "min": 0.4460249085718272}}, "EndTime": 1606244039.761103, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761084}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.43772595775370693, "sum": 0.43772595775370693, "min": 0.43772595775370693}}, "EndTime": 1606244039.761173, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761154}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.15046576908656528, "sum": 0.15046576908656528, "min": 0.15046576908656528}}, "EndTime": 1606244039.761248, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761228}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1495884754414461, "sum": 0.1495884754414461, "min": 0.1495884754414461}}, "EndTime": 1606244039.761318, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761299}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.15228331865583147, "sum": 0.15228331865583147, "min": 0.15228331865583147}}, "EndTime": 1606244039.761385, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761367}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.14753376411905095, "sum": 0.14753376411905095, "min": 0.14753376411905095}}, "EndTime": 1606244039.761451, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761433}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4507424266581633, "sum": 0.4507424266581633, "min": 0.4507424266581633}}, "EndTime": 1606244039.76152, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761501}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.46645258253447863, "sum": 0.46645258253447863, "min": 0.46645258253447863}}, "EndTime": 1606244039.761587, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761569}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4537874531648597, "sum": 0.4537874531648597, "min": 0.4537874531648597}}, "EndTime": 1606244039.761662, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761642}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4264061783771126, "sum": 0.4264061783771126, "min": 0.4264061783771126}}, "EndTime": 1606244039.761731, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761713}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.160155799087213, "sum": 0.160155799087213, "min": 0.160155799087213}}, "EndTime": 1606244039.7618, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761781}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.16104603389817843, "sum": 0.16104603389817843, "min": 0.16104603389817843}}, "EndTime": 1606244039.761868, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761849}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1589711531035754, "sum": 0.1589711531035754, "min": 0.1589711531035754}}, "EndTime": 1606244039.761939, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.76192}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.16025042724609376, "sum": 0.16025042724609376, "min": 0.16025042724609376}}, "EndTime": 1606244039.762007, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.761988}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.49733634294782364, "sum": 0.49733634294782364, "min": 0.49733634294782364}}, "EndTime": 1606244039.762078, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762059}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5090554604043765, "sum": 0.5090554604043765, "min": 0.5090554604043765}}, "EndTime": 1606244039.762144, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762126}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.5088660508759167, "sum": 0.5088660508759167, "min": 0.5088660508759167}}, "EndTime": 1606244039.762211, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762192}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4858610235720265, "sum": 0.4858610235720265, "min": 0.4858610235720265}}, "EndTime": 1606244039.762279, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762261}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47667461473114636, "sum": 0.47667461473114636, "min": 0.47667461473114636}}, "EndTime": 1606244039.762344, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762326}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4770319350884885, "sum": 0.4770319350884885, "min": 0.4770319350884885}}, "EndTime": 1606244039.76241, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762393}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4771143836196588, "sum": 0.4771143836196588, "min": 0.4771143836196588}}, "EndTime": 1606244039.76249, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762471}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4774246259416853, "sum": 0.4774246259416853, "min": 0.4774246259416853}}, "EndTime": 1606244039.762559, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762541}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893462823361767, "sum": 0.6893462823361767, "min": 0.6893462823361767}}, "EndTime": 1606244039.762626, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762608}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893075025908801, "sum": 0.6893075025908801, "min": 0.6893075025908801}}, "EndTime": 1606244039.762699, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.76268}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893364980269452, "sum": 0.6893364980269452, "min": 0.6893364980269452}}, "EndTime": 1606244039.762769, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762749}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894003594846142, "sum": 0.6894003594846142, "min": 0.6894003594846142}}, "EndTime": 1606244039.762837, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762818}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908187181122449, "sum": 0.6908187181122449, "min": 0.6908187181122449}}, "EndTime": 1606244039.762907, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762887}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908437512456155, "sum": 0.6908437512456155, "min": 0.6908437512456155}}, "EndTime": 1606244039.762977, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.762959}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908959637077488, "sum": 0.6908959637077488, "min": 0.6908959637077488}}, "EndTime": 1606244039.763051, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.763032}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6909769349390147, "sum": 0.6909769349390147, "min": 0.6909769349390147}}, "EndTime": 1606244039.763117, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244039.7631}
    [0m
    [34m[11/24/2020 18:53:59 INFO 140513285547840] #quality_metric: host=algo-1, epoch=1, train binary_classification_cross_entropy_objective <loss>=0.4501761649[0m
    [34m[2020-11-24 18:54:00.060] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 5, "duration": 280, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.40360338836084, "sum": 0.40360338836084, "min": 0.40360338836084}}, "EndTime": 1606244040.33459, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.334495}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3947714077235432, "sum": 0.3947714077235432, "min": 0.3947714077235432}}, "EndTime": 1606244040.334686, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.334671}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.40087894969757704, "sum": 0.40087894969757704, "min": 0.40087894969757704}}, "EndTime": 1606244040.334753, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.334734}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3895992548427712, "sum": 0.3895992548427712, "min": 0.3895992548427712}}, "EndTime": 1606244040.334822, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.334809}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13245815738885286, "sum": 0.13245815738885286, "min": 0.13245815738885286}}, "EndTime": 1606244040.334965, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.334865}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13117440608825584, "sum": 0.13117440608825584, "min": 0.13117440608825584}}, "EndTime": 1606244040.33504, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335023}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.133951827364871, "sum": 0.133951827364871, "min": 0.133951827364871}}, "EndTime": 1606244040.335104, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335086}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.12864915951431827, "sum": 0.12864915951431827, "min": 0.12864915951431827}}, "EndTime": 1606244040.335171, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335151}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4051495200447954, "sum": 0.4051495200447954, "min": 0.4051495200447954}}, "EndTime": 1606244040.335247, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335228}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4142981857265047, "sum": 0.4142981857265047, "min": 0.4142981857265047}}, "EndTime": 1606244040.335319, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.3353}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4073382004285689, "sum": 0.4073382004285689, "min": 0.4073382004285689}}, "EndTime": 1606244040.335398, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335378}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.381212163118908, "sum": 0.381212163118908, "min": 0.381212163118908}}, "EndTime": 1606244040.335475, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335455}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.15060796871117027, "sum": 0.15060796871117027, "min": 0.15060796871117027}}, "EndTime": 1606244040.335554, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335534}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1513050267411644, "sum": 0.1513050267411644, "min": 0.1513050267411644}}, "EndTime": 1606244040.335627, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.33561}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.14949927033642546, "sum": 0.14949927033642546, "min": 0.14949927033642546}}, "EndTime": 1606244040.335699, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335682}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.15047491267881136, "sum": 0.15047491267881136, "min": 0.15047491267881136}}, "EndTime": 1606244040.335765, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335746}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4815700840104336, "sum": 0.4815700840104336, "min": 0.4815700840104336}}, "EndTime": 1606244040.335832, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335815}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4861970846248767, "sum": 0.4861970846248767, "min": 0.4861970846248767}}, "EndTime": 1606244040.335891, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335878}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4892973125543417, "sum": 0.4892973125543417, "min": 0.4892973125543417}}, "EndTime": 1606244040.335974, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.335958}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47825413761952007, "sum": 0.47825413761952007, "min": 0.47825413761952007}}, "EndTime": 1606244040.336035, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.33602}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47567295554868955, "sum": 0.47567295554868955, "min": 0.47567295554868955}}, "EndTime": 1606244040.336083, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336063}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.474562652716992, "sum": 0.474562652716992, "min": 0.474562652716992}}, "EndTime": 1606244040.336149, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336131}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47715083481316606, "sum": 0.47715083481316606, "min": 0.47715083481316606}}, "EndTime": 1606244040.336223, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336203}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47751629100418463, "sum": 0.47751629100418463, "min": 0.47751629100418463}}, "EndTime": 1606244040.336292, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336274}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893354764094851, "sum": 0.6893354764094851, "min": 0.6893354764094851}}, "EndTime": 1606244040.336359, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.33634}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893747654327382, "sum": 0.6893747654327382, "min": 0.6893747654327382}}, "EndTime": 1606244040.336424, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336406}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895299477643932, "sum": 0.6895299477643932, "min": 0.6895299477643932}}, "EndTime": 1606244040.336498, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336481}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893842469354114, "sum": 0.6893842469354114, "min": 0.6893842469354114}}, "EndTime": 1606244040.336562, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336543}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6891867069402038, "sum": 0.6891867069402038, "min": 0.6891867069402038}}, "EndTime": 1606244040.336624, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336612}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896055713756597, "sum": 0.6896055713756597, "min": 0.6896055713756597}}, "EndTime": 1606244040.336694, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336674}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6891914911391174, "sum": 0.6891914911391174, "min": 0.6891914911391174}}, "EndTime": 1606244040.336769, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.33675}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892524269866198, "sum": 0.6892524269866198, "min": 0.6892524269866198}}, "EndTime": 1606244040.336841, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244040.336823}
    [0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] #quality_metric: host=algo-1, epoch=1, validation binary_classification_cross_entropy_objective <loss>=0.403603388361[0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=sampled_accuracy, value=0.970387243736[0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] Epoch 1: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] Saving model for epoch: 1[0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] Saved checkpoint to "/tmp/tmp26AY77/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] #progress_metric: host=algo-1, completed 13 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 112, "sum": 112.0, "min": 112}, "Total Records Seen": {"count": 1, "max": 110338, "sum": 110338.0, "min": 110338}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1606244040.343521, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1606244037.592132}
    [0m
    [34m[11/24/2020 18:54:00 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=17869.7133378 records/second[0m
    [34m[2020-11-24 18:54:02.066] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 7, "duration": 1722, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.36833852152921714, "sum": 0.36833852152921714, "min": 0.36833852152921714}}, "EndTime": 1606244042.066455, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066344}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3579849785006776, "sum": 0.3579849785006776, "min": 0.3579849785006776}}, "EndTime": 1606244042.066562, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066541}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3664571427325813, "sum": 0.3664571427325813, "min": 0.3664571427325813}}, "EndTime": 1606244042.06663, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06661}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.35353284687898595, "sum": 0.35353284687898595, "min": 0.35353284687898595}}, "EndTime": 1606244042.0667, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066682}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11286931454405492, "sum": 0.11286931454405492, "min": 0.11286931454405492}}, "EndTime": 1606244042.066772, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066752}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11120873478480747, "sum": 0.11120873478480747, "min": 0.11120873478480747}}, "EndTime": 1606244042.066833, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066816}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.11407252315599091, "sum": 0.11407252315599091, "min": 0.11407252315599091}}, "EndTime": 1606244042.066892, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066876}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.10869429779052735, "sum": 0.10869429779052735, "min": 0.10869429779052735}}, "EndTime": 1606244042.066956, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.066937}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.37039173671177456, "sum": 0.37039173671177456, "min": 0.37039173671177456}}, "EndTime": 1606244042.067022, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067003}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3754835460429289, "sum": 0.3754835460429289, "min": 0.3754835460429289}}, "EndTime": 1606244042.067091, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067072}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3718283903160874, "sum": 0.3718283903160874, "min": 0.3718283903160874}}, "EndTime": 1606244042.067159, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06714}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.34699400921257173, "sum": 0.34699400921257173, "min": 0.34699400921257173}}, "EndTime": 1606244042.067224, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067206}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.14007102343500877, "sum": 0.14007102343500877, "min": 0.14007102343500877}}, "EndTime": 1606244042.067287, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06727}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1403840061109893, "sum": 0.1403840061109893, "min": 0.1403840061109893}}, "EndTime": 1606244042.067352, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067333}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13926023864746093, "sum": 0.13926023864746093, "min": 0.13926023864746093}}, "EndTime": 1606244042.067421, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067401}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1397728287054568, "sum": 0.1397728287054568, "min": 0.1397728287054568}}, "EndTime": 1606244042.067489, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06747}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4772465895049426, "sum": 0.4772465895049426, "min": 0.4772465895049426}}, "EndTime": 1606244042.067556, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067538}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.478850896095743, "sum": 0.478850896095743, "min": 0.478850896095743}}, "EndTime": 1606244042.067623, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067604}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4820378380600287, "sum": 0.4820378380600287, "min": 0.4820378380600287}}, "EndTime": 1606244042.067693, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067674}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763581493144133, "sum": 0.4763581493144133, "min": 0.4763581493144133}}, "EndTime": 1606244042.067768, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067747}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4766805114746094, "sum": 0.4766805114746094, "min": 0.4766805114746094}}, "EndTime": 1606244042.067845, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067826}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47697173760861766, "sum": 0.47697173760861766, "min": 0.47697173760861766}}, "EndTime": 1606244042.067933, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067893}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47673475304428414, "sum": 0.47673475304428414, "min": 0.47673475304428414}}, "EndTime": 1606244042.068001, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.067984}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4770590334522481, "sum": 0.4770590334522481, "min": 0.4770590334522481}}, "EndTime": 1606244042.068039, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068029}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894266581632653, "sum": 0.6894266581632653, "min": 0.6894266581632653}}, "EndTime": 1606244042.068071, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068063}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894414797413105, "sum": 0.6894414797413105, "min": 0.6894414797413105}}, "EndTime": 1606244042.068104, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068095}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894283982880262, "sum": 0.6894283982880262, "min": 0.6894283982880262}}, "EndTime": 1606244042.068136, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068127}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894328264508929, "sum": 0.6894328264508929, "min": 0.6894328264508929}}, "EndTime": 1606244042.068175, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06816}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908819231305804, "sum": 0.6908819231305804, "min": 0.6908819231305804}}, "EndTime": 1606244042.068243, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068225}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6914233435805963, "sum": 0.6914233435805963, "min": 0.6914233435805963}}, "EndTime": 1606244042.06832, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.0683}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908781252491231, "sum": 0.6908781252491231, "min": 0.6908781252491231}}, "EndTime": 1606244042.068389, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.06837}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6913988722197864, "sum": 0.6913988722197864, "min": 0.6913988722197864}}, "EndTime": 1606244042.068458, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.068441}
    [0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] #quality_metric: host=algo-1, epoch=2, train binary_classification_cross_entropy_objective <loss>=0.368338521529[0m
    [34m[2020-11-24 18:54:02.362] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 8, "duration": 276, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.34151306425442807, "sum": 0.34151306425442807, "min": 0.34151306425442807}}, "EndTime": 1606244042.641951, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.641859}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3298643294041761, "sum": 0.3298643294041761, "min": 0.3298643294041761}}, "EndTime": 1606244042.64205, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642034}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3405464115260751, "sum": 0.3405464115260751, "min": 0.3405464115260751}}, "EndTime": 1606244042.642109, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642091}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3261015470276971, "sum": 0.3261015470276971, "min": 0.3261015470276971}}, "EndTime": 1606244042.642173, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642155}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.10173662113049424, "sum": 0.10173662113049424, "min": 0.10173662113049424}}, "EndTime": 1606244042.642243, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642226}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0995773780318762, "sum": 0.0995773780318762, "min": 0.0995773780318762}}, "EndTime": 1606244042.642315, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642295}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.10268513899535338, "sum": 0.10268513899535338, "min": 0.10268513899535338}}, "EndTime": 1606244042.642389, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642369}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.09715650572747313, "sum": 0.09715650572747313, "min": 0.09715650572747313}}, "EndTime": 1606244042.642459, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642439}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.34411628790177934, "sum": 0.34411628790177934, "min": 0.34411628790177934}}, "EndTime": 1606244042.642531, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642512}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3454625892825364, "sum": 0.3454625892825364, "min": 0.3454625892825364}}, "EndTime": 1606244042.642602, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642581}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3447747632419873, "sum": 0.3447747632419873, "min": 0.3447747632419873}}, "EndTime": 1606244042.642674, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642655}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3213807933694409, "sum": 0.3213807933694409, "min": 0.3213807933694409}}, "EndTime": 1606244042.642745, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642726}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13821672345852565, "sum": 0.13821672345852565, "min": 0.13821672345852565}}, "EndTime": 1606244042.642819, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642798}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1383036297848742, "sum": 0.1383036297848742, "min": 0.1383036297848742}}, "EndTime": 1606244042.642892, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642872}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13769341011680455, "sum": 0.13769341011680455, "min": 0.13769341011680455}}, "EndTime": 1606244042.642964, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.642944}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.137894679465823, "sum": 0.137894679465823, "min": 0.137894679465823}}, "EndTime": 1606244042.643037, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643016}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47677501327162103, "sum": 0.47677501327162103, "min": 0.47677501327162103}}, "EndTime": 1606244042.643116, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643096}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4768266625098673, "sum": 0.4768266625098673, "min": 0.4768266625098673}}, "EndTime": 1606244042.643187, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643169}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47915710116534443, "sum": 0.47915710116534443, "min": 0.47915710116534443}}, "EndTime": 1606244042.643267, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643246}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4767041726179089, "sum": 0.4767041726179089, "min": 0.4767041726179089}}, "EndTime": 1606244042.643347, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643327}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47733741732944, "sum": 0.47733741732944, "min": 0.47733741732944}}, "EndTime": 1606244042.643422, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643403}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4738975634141966, "sum": 0.4738975634141966, "min": 0.4738975634141966}}, "EndTime": 1606244042.643499, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643479}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4746812124652658, "sum": 0.4746812124652658, "min": 0.4746812124652658}}, "EndTime": 1606244042.643567, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643548}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4742644384789459, "sum": 0.4742644384789459, "min": 0.4742644384789459}}, "EndTime": 1606244042.643646, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643626}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894093943930122, "sum": 0.6894093943930122, "min": 0.6894093943930122}}, "EndTime": 1606244042.643711, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643695}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895432985841433, "sum": 0.6895432985841433, "min": 0.6895432985841433}}, "EndTime": 1606244042.643752, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643739}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894514777161499, "sum": 0.6894514777161499, "min": 0.6894514777161499}}, "EndTime": 1606244042.643814, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643796}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894251720702808, "sum": 0.6894251720702808, "min": 0.6894251720702808}}, "EndTime": 1606244042.643872, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.64386}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895134227887993, "sum": 0.6895134227887993, "min": 0.6895134227887993}}, "EndTime": 1606244042.643966, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.643945}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6900058377654166, "sum": 0.6900058377654166, "min": 0.6900058377654166}}, "EndTime": 1606244042.644037, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.64402}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689410514099141, "sum": 0.689410514099141, "min": 0.689410514099141}}, "EndTime": 1606244042.644103, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.644084}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.690010423346835, "sum": 0.690010423346835, "min": 0.690010423346835}}, "EndTime": 1606244042.644181, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244042.644161}
    [0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] #quality_metric: host=algo-1, epoch=2, validation binary_classification_cross_entropy_objective <loss>=0.341513064254[0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=sampled_accuracy, value=0.975919297104[0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] Epoch 2: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] Saving model for epoch: 2[0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] Saved checkpoint to "/tmp/tmpOpejyM/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] #progress_metric: host=algo-1, completed 20 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 162, "sum": 162.0, "min": 162}, "Total Records Seen": {"count": 1, "max": 159507, "sum": 159507.0, "min": 159507}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1606244042.651096, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1606244040.344063}
    [0m
    [34m[11/24/2020 18:54:02 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=21311.4834329 records/second[0m
    [34m[2020-11-24 18:54:04.299] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 9, "duration": 1647, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3179140332280373, "sum": 0.3179140332280373, "min": 0.3179140332280373}}, "EndTime": 1606244044.299547, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299427}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.30531499496771364, "sum": 0.30531499496771364, "min": 0.30531499496771364}}, "EndTime": 1606244044.29965, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299628}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.31748714400310907, "sum": 0.31748714400310907, "min": 0.31748714400310907}}, "EndTime": 1606244044.299725, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299706}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3020340937400351, "sum": 0.3020340937400351, "min": 0.3020340937400351}}, "EndTime": 1606244044.299797, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299776}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08836229425547075, "sum": 0.08836229425547075, "min": 0.08836229425547075}}, "EndTime": 1606244044.299868, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299848}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08585160329390545, "sum": 0.08585160329390545, "min": 0.08585160329390545}}, "EndTime": 1606244044.299976, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.299954}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08912546041060466, "sum": 0.08912546041060466, "min": 0.08912546041060466}}, "EndTime": 1606244044.300053, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300032}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08366902876873406, "sum": 0.08366902876873406, "min": 0.08366902876873406}}, "EndTime": 1606244044.300125, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300105}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.32090837875677614, "sum": 0.32090837875677614, "min": 0.32090837875677614}}, "EndTime": 1606244044.300194, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300176}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.31963423437001753, "sum": 0.31963423437001753, "min": 0.31963423437001753}}, "EndTime": 1606244044.300263, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300243}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3210331514319595, "sum": 0.3210331514319595, "min": 0.3210331514319595}}, "EndTime": 1606244044.300331, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300312}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2986295645577567, "sum": 0.2986295645577567, "min": 0.2986295645577567}}, "EndTime": 1606244044.300403, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300383}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13267040564089405, "sum": 0.13267040564089405, "min": 0.13267040564089405}}, "EndTime": 1606244044.300474, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300453}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1326322170958227, "sum": 0.1326322170958227, "min": 0.1326322170958227}}, "EndTime": 1606244044.300541, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300522}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13232780269700653, "sum": 0.13232780269700653, "min": 0.13232780269700653}}, "EndTime": 1606244044.300614, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300595}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13237368805554448, "sum": 0.13237368805554448, "min": 0.13237368805554448}}, "EndTime": 1606244044.300692, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300674}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47583868345922353, "sum": 0.47583868345922353, "min": 0.47583868345922353}}, "EndTime": 1606244044.300759, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.30074}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757229676538584, "sum": 0.4757229676538584, "min": 0.4757229676538584}}, "EndTime": 1606244044.300828, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300808}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4770711750886878, "sum": 0.4770711750886878, "min": 0.4770711750886878}}, "EndTime": 1606244044.300898, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300878}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579437006736286, "sum": 0.47579437006736286, "min": 0.47579437006736286}}, "EndTime": 1606244044.300967, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.300948}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47699390956333704, "sum": 0.47699390956333704, "min": 0.47699390956333704}}, "EndTime": 1606244044.301038, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301018}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4776956731056681, "sum": 0.4776956731056681, "min": 0.4776956731056681}}, "EndTime": 1606244044.301107, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301088}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4768522507025271, "sum": 0.4768522507025271, "min": 0.4768522507025271}}, "EndTime": 1606244044.301174, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301155}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47764508305763714, "sum": 0.47764508305763714, "min": 0.47764508305763714}}, "EndTime": 1606244044.30125, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301231}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894204039281728, "sum": 0.6894204039281728, "min": 0.6894204039281728}}, "EndTime": 1606244044.30132, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.3013}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894360077527104, "sum": 0.6894360077527104, "min": 0.6894360077527104}}, "EndTime": 1606244044.301388, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301369}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894223433514031, "sum": 0.6894223433514031, "min": 0.6894223433514031}}, "EndTime": 1606244044.301466, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301446}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894329809072066, "sum": 0.6894329809072066, "min": 0.6894329809072066}}, "EndTime": 1606244044.301533, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301515}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6910232157804528, "sum": 0.6910232157804528, "min": 0.6910232157804528}}, "EndTime": 1606244044.301601, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301581}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6921379544005102, "sum": 0.6921379544005102, "min": 0.6921379544005102}}, "EndTime": 1606244044.30168, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301659}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.691044895717076, "sum": 0.691044895717076, "min": 0.691044895717076}}, "EndTime": 1606244044.30176, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.30174}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6921218959263393, "sum": 0.6921218959263393, "min": 0.6921218959263393}}, "EndTime": 1606244044.30184, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.301819}
    [0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] #quality_metric: host=algo-1, epoch=3, train binary_classification_cross_entropy_objective <loss>=0.317914033228[0m
    [34m[2020-11-24 18:54:04.583] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 11, "duration": 261, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3017089724579503, "sum": 0.3017089724579503, "min": 0.3017089724579503}}, "EndTime": 1606244044.85686, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.856763}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2884364551726669, "sum": 0.2884364551726669, "min": 0.2884364551726669}}, "EndTime": 1606244044.856965, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.856942}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3019329881652574, "sum": 0.3019329881652574, "min": 0.3019329881652574}}, "EndTime": 1606244044.857014, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857002}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2856334856345527, "sum": 0.2856334856345527, "min": 0.2856334856345527}}, "EndTime": 1606244044.857076, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857064}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08264164485490715, "sum": 0.08264164485490715, "min": 0.08264164485490715}}, "EndTime": 1606244044.85712, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.85711}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07976961725509947, "sum": 0.07976961725509947, "min": 0.07976961725509947}}, "EndTime": 1606244044.857166, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857151}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.08325921926675307, "sum": 0.08325921926675307, "min": 0.08325921926675307}}, "EndTime": 1606244044.857229, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857211}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07779686589835329, "sum": 0.07779686589835329, "min": 0.07779686589835329}}, "EndTime": 1606244044.85729, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857272}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3051148571954113, "sum": 0.3051148571954113, "min": 0.3051148571954113}}, "EndTime": 1606244044.857364, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857345}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3014858170126689, "sum": 0.3014858170126689, "min": 0.3014858170126689}}, "EndTime": 1606244044.857428, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857411}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.3046464110056973, "sum": 0.3046464110056973, "min": 0.3046464110056973}}, "EndTime": 1606244044.857502, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857483}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2835317922654852, "sum": 0.2835317922654852, "min": 0.2835317922654852}}, "EndTime": 1606244044.85757, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857551}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13447139909435785, "sum": 0.13447139909435785, "min": 0.13447139909435785}}, "EndTime": 1606244044.857632, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857613}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13419588931872253, "sum": 0.13419588931872253, "min": 0.13419588931872253}}, "EndTime": 1606244044.857699, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857682}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1342977169395618, "sum": 0.1342977169395618, "min": 0.1342977169395618}}, "EndTime": 1606244044.857737, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857727}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13405046034743723, "sum": 0.13405046034743723, "min": 0.13405046034743723}}, "EndTime": 1606244044.857771, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857762}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47663651853363403, "sum": 0.47663651853363403, "min": 0.47663651853363403}}, "EndTime": 1606244044.857805, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857796}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47665068865055144, "sum": 0.47665068865055144, "min": 0.47665068865055144}}, "EndTime": 1606244044.857845, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.85783}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4771033317151304, "sum": 0.4771033317151304, "min": 0.4771033317151304}}, "EndTime": 1606244044.857906, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.85789}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47655814034598215, "sum": 0.47655814034598215, "min": 0.47655814034598215}}, "EndTime": 1606244044.857944, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857934}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47698528092418785, "sum": 0.47698528092418785, "min": 0.47698528092418785}}, "EndTime": 1606244044.858005, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.857987}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.477964640206791, "sum": 0.477964640206791, "min": 0.477964640206791}}, "EndTime": 1606244044.858063, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.85805}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4739893246969415, "sum": 0.4739893246969415, "min": 0.4739893246969415}}, "EndTime": 1606244044.858117, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858099}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47543432035238237, "sum": 0.47543432035238237, "min": 0.47543432035238237}}, "EndTime": 1606244044.858179, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858165}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893692413836553, "sum": 0.6893692413836553, "min": 0.6893692413836553}}, "EndTime": 1606244044.858234, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858216}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895199299944832, "sum": 0.6895199299944832, "min": 0.6895199299944832}}, "EndTime": 1606244044.85831, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.85829}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894074020112643, "sum": 0.6894074020112643, "min": 0.6894074020112643}}, "EndTime": 1606244044.858386, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858366}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894558150256995, "sum": 0.6894558150256995, "min": 0.6894558150256995}}, "EndTime": 1606244044.858463, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858441}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6897767026921727, "sum": 0.6897767026921727, "min": 0.6897767026921727}}, "EndTime": 1606244044.858536, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858518}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902286145960979, "sum": 0.6902286145960979, "min": 0.6902286145960979}}, "EndTime": 1606244044.858602, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858583}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896376654801523, "sum": 0.6896376654801523, "min": 0.6896376654801523}}, "EndTime": 1606244044.858683, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858665}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902650460101399, "sum": 0.6902650460101399, "min": 0.6902650460101399}}, "EndTime": 1606244044.85876, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244044.858741}
    [0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] #quality_metric: host=algo-1, epoch=3, validation binary_classification_cross_entropy_objective <loss>=0.301708972458[0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=sampled_accuracy, value=0.983891962252[0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] Epoch 3: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] Saving model for epoch: 3[0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] Saved checkpoint to "/tmp/tmpNZ43om/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] #progress_metric: host=algo-1, completed 26 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 212, "sum": 212.0, "min": 212}, "Total Records Seen": {"count": 1, "max": 208676, "sum": 208676.0, "min": 208676}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1606244044.865182, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1606244042.651576}
    [0m
    [34m[11/24/2020 18:54:04 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=22211.0189948 records/second[0m
    
    2020-11-24 18:53:51 Training - Training image download completed. Training in progress.[34m[2020-11-24 18:54:06.346] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 11, "duration": 1480, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.28486068351901306, "sum": 0.28486068351901306, "min": 0.28486068351901306}}, "EndTime": 1606244046.346612, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.346499}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.271174142642897, "sum": 0.271174142642897, "min": 0.271174142642897}}, "EndTime": 1606244046.346723, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.3467}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.285439070409658, "sum": 0.285439070409658, "min": 0.285439070409658}}, "EndTime": 1606244046.346796, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.346775}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26869843961754625, "sum": 0.26869843961754625, "min": 0.26869843961754625}}, "EndTime": 1606244046.34687, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.346849}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07319793459833884, "sum": 0.07319793459833884, "min": 0.07319793459833884}}, "EndTime": 1606244046.346933, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.346913}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07017592145958726, "sum": 0.07017592145958726, "min": 0.07017592145958726}}, "EndTime": 1606244046.347003, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.346984}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07370908892884546, "sum": 0.07370908892884546, "min": 0.07370908892884546}}, "EndTime": 1606244046.347072, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347052}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06843023557079081, "sum": 0.06843023557079081, "min": 0.06843023557079081}}, "EndTime": 1606244046.347139, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347121}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.28861461841816805, "sum": 0.28861461841816805, "min": 0.28861461841816805}}, "EndTime": 1606244046.347207, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347187}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2834415055878308, "sum": 0.2834415055878308, "min": 0.2834415055878308}}, "EndTime": 1606244046.347283, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347263}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.28781124379683515, "sum": 0.28781124379683515, "min": 0.28781124379683515}}, "EndTime": 1606244046.347352, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347333}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2675837959756657, "sum": 0.2675837959756657, "min": 0.2675837959756657}}, "EndTime": 1606244046.347422, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347403}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13063780445955236, "sum": 0.13063780445955236, "min": 0.13063780445955236}}, "EndTime": 1606244046.347493, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347471}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13054645429338727, "sum": 0.13054645429338727, "min": 0.13054645429338727}}, "EndTime": 1606244046.347559, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347539}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13053489233523, "sum": 0.13053489233523, "min": 0.13053489233523}}, "EndTime": 1606244046.347636, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347616}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13047246099978077, "sum": 0.13047246099978077, "min": 0.13047246099978077}}, "EndTime": 1606244046.347711, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347691}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.475794950524155, "sum": 0.475794950524155, "min": 0.475794950524155}}, "EndTime": 1606244046.347785, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347765}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757908605458785, "sum": 0.4757908605458785, "min": 0.4757908605458785}}, "EndTime": 1606244046.347852, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347833}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4760057260941486, "sum": 0.4760057260941486, "min": 0.4760057260941486}}, "EndTime": 1606244046.347948, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.347901}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757895364566725, "sum": 0.4757895364566725, "min": 0.4757895364566725}}, "EndTime": 1606244046.348008, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.34799}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4770156542719627, "sum": 0.4770156542719627, "min": 0.4770156542719627}}, "EndTime": 1606244046.348071, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348052}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47807986637037625, "sum": 0.47807986637037625, "min": 0.47807986637037625}}, "EndTime": 1606244046.348141, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348122}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47690580780652103, "sum": 0.47690580780652103, "min": 0.47690580780652103}}, "EndTime": 1606244046.348214, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348193}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780010351064254, "sum": 0.4780010351064254, "min": 0.4780010351064254}}, "EndTime": 1606244046.348286, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348266}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894147463428731, "sum": 0.6894147463428731, "min": 0.6894147463428731}}, "EndTime": 1606244046.348353, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348334}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894370565609056, "sum": 0.6894370565609056, "min": 0.6894370565609056}}, "EndTime": 1606244046.348417, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348399}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689416119011081, "sum": 0.689416119011081, "min": 0.689416119011081}}, "EndTime": 1606244046.348482, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348463}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894340409259407, "sum": 0.6894340409259407, "min": 0.6894340409259407}}, "EndTime": 1606244046.348549, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348529}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6910490834761639, "sum": 0.6910490834761639, "min": 0.6910490834761639}}, "EndTime": 1606244046.348622, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348603}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923300046336894, "sum": 0.6923300046336894, "min": 0.6923300046336894}}, "EndTime": 1606244046.348702, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.348682}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6910605867346938, "sum": 0.6910605867346938, "min": 0.6910605867346938}}, "EndTime": 1606244046.348781, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.34876}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923243956273916, "sum": 0.6923243956273916, "min": 0.6923243956273916}}, "EndTime": 1606244046.348861, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.34884}
    [0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] #quality_metric: host=algo-1, epoch=4, train binary_classification_cross_entropy_objective <loss>=0.284860683519[0m
    [34m[2020-11-24 18:54:06.681] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 14, "duration": 314, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2750455792393236, "sum": 0.2750455792393236, "min": 0.2750455792393236}}, "EndTime": 1606244046.952915, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.952823}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26123867972138415, "sum": 0.26123867972138415, "min": 0.26123867972138415}}, "EndTime": 1606244046.953025, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953004}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.27611103175790463, "sum": 0.27611103175790463, "min": 0.27611103175790463}}, "EndTime": 1606244046.953096, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953076}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25909934523359324, "sum": 0.25909934523359324, "min": 0.25909934523359324}}, "EndTime": 1606244046.953174, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953153}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.07061782451161647, "sum": 0.07061782451161647, "min": 0.07061782451161647}}, "EndTime": 1606244046.953242, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953224}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06738664667109034, "sum": 0.06738664667109034, "min": 0.06738664667109034}}, "EndTime": 1606244046.953308, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953289}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.071046817508304, "sum": 0.071046817508304, "min": 0.071046817508304}}, "EndTime": 1606244046.953378, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953359}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06581378968422241, "sum": 0.06581378968422241, "min": 0.06581378968422241}}, "EndTime": 1606244046.953447, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953426}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.27912839398568745, "sum": 0.27912839398568745, "min": 0.27912839398568745}}, "EndTime": 1606244046.953515, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953496}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.27264647106762785, "sum": 0.27264647106762785, "min": 0.27264647106762785}}, "EndTime": 1606244046.953585, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953564}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.277892216075099, "sum": 0.277892216075099, "min": 0.277892216075099}}, "EndTime": 1606244046.953658, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953637}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25893334073117297, "sum": 0.25893334073117297, "min": 0.25893334073117297}}, "EndTime": 1606244046.953736, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953715}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1335937585033115, "sum": 0.1335937585033115, "min": 0.1335937585033115}}, "EndTime": 1606244046.953807, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953787}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13322693867905303, "sum": 0.13322693867905303, "min": 0.13322693867905303}}, "EndTime": 1606244046.953879, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953857}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13355523493326413, "sum": 0.13355523493326413, "min": 0.13355523493326413}}, "EndTime": 1606244046.953947, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.953927}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13320339723010424, "sum": 0.13320339723010424, "min": 0.13320339723010424}}, "EndTime": 1606244046.954021, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954001}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47658777407625386, "sum": 0.47658777407625386, "min": 0.47658777407625386}}, "EndTime": 1606244046.954094, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954074}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47665633062877216, "sum": 0.47665633062877216, "min": 0.47665633062877216}}, "EndTime": 1606244046.954166, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954146}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47665059430724127, "sum": 0.47665059430724127, "min": 0.47665059430724127}}, "EndTime": 1606244046.954237, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954218}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47654954765659957, "sum": 0.47654954765659957, "min": 0.47654954765659957}}, "EndTime": 1606244046.954299, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954286}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47590464409810496, "sum": 0.47590464409810496, "min": 0.47590464409810496}}, "EndTime": 1606244046.954337, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954328}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4783166226834844, "sum": 0.4783166226834844, "min": 0.4783166226834844}}, "EndTime": 1606244046.954372, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954363}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4749356017530139, "sum": 0.4749356017530139, "min": 0.4749356017530139}}, "EndTime": 1606244046.954414, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954404}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4784431283729223, "sum": 0.4784431283729223, "min": 0.4784431283729223}}, "EndTime": 1606244046.954469, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954453}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689343803447968, "sum": 0.689343803447968, "min": 0.689343803447968}}, "EndTime": 1606244046.954512, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954501}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894529983813469, "sum": 0.6894529983813469, "min": 0.6894529983813469}}, "EndTime": 1606244046.954549, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954539}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893530888158648, "sum": 0.6893530888158648, "min": 0.6893530888158648}}, "EndTime": 1606244046.954589, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954579}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894907580498876, "sum": 0.6894907580498876, "min": 0.6894907580498876}}, "EndTime": 1606244046.954646, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.95463}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896837323254263, "sum": 0.6896837323254263, "min": 0.6896837323254263}}, "EndTime": 1606244046.954689, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954679}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904213654269545, "sum": 0.6904213654269545, "min": 0.6904213654269545}}, "EndTime": 1606244046.95473, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.95472}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895338406672976, "sum": 0.6895338406672976, "min": 0.6895338406672976}}, "EndTime": 1606244046.95477, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.95476}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903847987047438, "sum": 0.6903847987047438, "min": 0.6903847987047438}}, "EndTime": 1606244046.954832, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244046.954816}
    [0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] #quality_metric: host=algo-1, epoch=4, validation binary_classification_cross_entropy_objective <loss>=0.275045579239[0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=sampled_accuracy, value=0.988122356004[0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] Epoch 4: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] Saving model for epoch: 4[0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] Saved checkpoint to "/tmp/tmp4KgX2E/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] #progress_metric: host=algo-1, completed 33 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 262, "sum": 262.0, "min": 262}, "Total Records Seen": {"count": 1, "max": 257845, "sum": 257845.0, "min": 257845}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1606244046.961793, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1606244044.865607}
    [0m
    [34m[11/24/2020 18:54:06 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=23455.0564775 records/second[0m
    [34m[2020-11-24 18:54:08.584] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 13, "duration": 1622, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26230252199756854, "sum": 0.26230252199756854, "min": 0.26230252199756854}}, "EndTime": 1606244048.585146, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.584838}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24830142803581393, "sum": 0.24830142803581393, "min": 0.24830142803581393}}, "EndTime": 1606244048.585248, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585226}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2635944951894332, "sum": 0.2635944951894332, "min": 0.2635944951894332}}, "EndTime": 1606244048.58533, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585309}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24638544977927695, "sum": 0.24638544977927695, "min": 0.24638544977927695}}, "EndTime": 1606244048.58542, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585397}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06344646749691088, "sum": 0.06344646749691088, "min": 0.06344646749691088}}, "EndTime": 1606244048.585506, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585485}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06014860059777084, "sum": 0.06014860059777084, "min": 0.06014860059777084}}, "EndTime": 1606244048.585571, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585555}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06381411120356345, "sum": 0.06381411120356345, "min": 0.06381411120356345}}, "EndTime": 1606244048.58564, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585625}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.058752282201027385, "sum": 0.058752282201027385, "min": 0.058752282201027385}}, "EndTime": 1606244048.585691, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585681}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26668314252580916, "sum": 0.26668314252580916, "min": 0.26668314252580916}}, "EndTime": 1606244048.585733, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585717}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25924094375298945, "sum": 0.25924094375298945, "min": 0.25924094375298945}}, "EndTime": 1606244048.585793, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585779}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26524632574587453, "sum": 0.26524632574587453, "min": 0.26524632574587453}}, "EndTime": 1606244048.58585, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585832}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24695939231405453, "sum": 0.24695939231405453, "min": 0.24695939231405453}}, "EndTime": 1606244048.585916, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585899}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1302250534369021, "sum": 0.1302250534369021, "min": 0.1302250534369021}}, "EndTime": 1606244048.585982, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.585965}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13008296530587332, "sum": 0.13008296530587332, "min": 0.13008296530587332}}, "EndTime": 1606244048.586034, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586022}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13020395800532128, "sum": 0.13020395800532128, "min": 0.13020395800532128}}, "EndTime": 1606244048.58607, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586061}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13006813158307756, "sum": 0.13006813158307756, "min": 0.13006813158307756}}, "EndTime": 1606244048.586112, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586096}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757818204918686, "sum": 0.4757818204918686, "min": 0.4757818204918686}}, "EndTime": 1606244048.586178, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586158}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579954809072067, "sum": 0.47579954809072067, "min": 0.47579954809072067}}, "EndTime": 1606244048.586251, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586232}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47580574814154175, "sum": 0.47580574814154175, "min": 0.47580574814154175}}, "EndTime": 1606244048.586333, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586311}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579548676159916, "sum": 0.47579548676159916, "min": 0.47579548676159916}}, "EndTime": 1606244048.586414, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586393}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476961702930684, "sum": 0.476961702930684, "min": 0.476961702930684}}, "EndTime": 1606244048.586495, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586474}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4781697661730708, "sum": 0.4781697661730708, "min": 0.4781697661730708}}, "EndTime": 1606244048.586575, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586554}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4769053319814254, "sum": 0.4769053319814254, "min": 0.4769053319814254}}, "EndTime": 1606244048.586654, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586635}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4782062726702009, "sum": 0.4782062726702009, "min": 0.4782062726702009}}, "EndTime": 1606244048.586733, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586713}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894076724928253, "sum": 0.6894076724928253, "min": 0.6894076724928253}}, "EndTime": 1606244048.586813, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586793}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689434334891183, "sum": 0.689434334891183, "min": 0.689434334891183}}, "EndTime": 1606244048.586881, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586863}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894082703882334, "sum": 0.6894082703882334, "min": 0.6894082703882334}}, "EndTime": 1606244048.586951, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586933}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894333757672991, "sum": 0.6894333757672991, "min": 0.6894333757672991}}, "EndTime": 1606244048.587018, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.586998}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6909728244080835, "sum": 0.6909728244080835, "min": 0.6909728244080835}}, "EndTime": 1606244048.587098, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.587078}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924753604811065, "sum": 0.6924753604811065, "min": 0.6924753604811065}}, "EndTime": 1606244048.587175, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.587154}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6909806717853156, "sum": 0.6909806717853156, "min": 0.6909806717853156}}, "EndTime": 1606244048.587255, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.587234}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924687338069994, "sum": 0.6924687338069994, "min": 0.6924687338069994}}, "EndTime": 1606244048.58733, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244048.58731}
    [0m
    [34m[11/24/2020 18:54:08 INFO 140513285547840] #quality_metric: host=algo-1, epoch=5, train binary_classification_cross_entropy_objective <loss>=0.262302521998[0m
    [34m[2020-11-24 18:54:08.883] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 17, "duration": 271, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25653235314142964, "sum": 0.25653235314142964, "min": 0.25653235314142964}}, "EndTime": 1606244049.171759, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.171661}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24263048032344145, "sum": 0.24263048032344145, "min": 0.24263048032344145}}, "EndTime": 1606244049.17187, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.171854}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.25819773663149337, "sum": 0.25819773663149337, "min": 0.25819773663149337}}, "EndTime": 1606244049.171963, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17194}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24095603801356127, "sum": 0.24095603801356127, "min": 0.24095603801356127}}, "EndTime": 1606244049.172041, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172022}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0626726173788183, "sum": 0.0626726173788183, "min": 0.0626726173788183}}, "EndTime": 1606244049.172115, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172095}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05916666418187905, "sum": 0.05916666418187905, "min": 0.05916666418187905}}, "EndTime": 1606244049.172186, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172166}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.06298870928621897, "sum": 0.06298870928621897, "min": 0.06298870928621897}}, "EndTime": 1606244049.172257, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172238}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.057895718899449736, "sum": 0.057895718899449736, "min": 0.057895718899449736}}, "EndTime": 1606244049.172327, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172309}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.26116343534539743, "sum": 0.26116343534539743, "min": 0.26116343534539743}}, "EndTime": 1606244049.1724, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17238}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2529599431513033, "sum": 0.2529599431513033, "min": 0.2529599431513033}}, "EndTime": 1606244049.17248, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17246}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2594062105049111, "sum": 0.2594062105049111, "min": 0.2594062105049111}}, "EndTime": 1606244049.172552, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172532}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24221202243317078, "sum": 0.24221202243317078, "min": 0.24221202243317078}}, "EndTime": 1606244049.172634, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172619}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1334060103506131, "sum": 0.1334060103506131, "min": 0.1334060103506131}}, "EndTime": 1606244049.17271, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172691}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13321442220795243, "sum": 0.13321442220795243, "min": 0.13321442220795243}}, "EndTime": 1606244049.172789, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.172769}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13340138551120528, "sum": 0.13340138551120528, "min": 0.13340138551120528}}, "EndTime": 1606244049.17286, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17284}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13321756734702161, "sum": 0.13321756734702161, "min": 0.13321756734702161}}, "EndTime": 1606244049.172941, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17292}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765680414281138, "sum": 0.4765680414281138, "min": 0.4765680414281138}}, "EndTime": 1606244049.173021, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47661009992406456, "sum": 0.47661009992406456, "min": 0.47661009992406456}}, "EndTime": 1606244049.1731, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17308}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765737131989588, "sum": 0.4765737131989588, "min": 0.4765737131989588}}, "EndTime": 1606244049.173176, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173157}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765129784516702, "sum": 0.4765129784516702, "min": 0.4765129784516702}}, "EndTime": 1606244049.173257, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173236}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47504758477948056, "sum": 0.47504758477948056, "min": 0.47504758477948056}}, "EndTime": 1606244049.173328, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173308}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4770583833346257, "sum": 0.4770583833346257, "min": 0.4770583833346257}}, "EndTime": 1606244049.1734, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17338}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47623618014503527, "sum": 0.47623618014503527, "min": 0.47623618014503527}}, "EndTime": 1606244049.173463, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173444}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4783800896626903, "sum": 0.4783800896626903, "min": 0.4783800896626903}}, "EndTime": 1606244049.17354, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17352}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893250403014759, "sum": 0.6893250403014759, "min": 0.6893250403014759}}, "EndTime": 1606244049.173613, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173597}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893926608690485, "sum": 0.6893926608690485, "min": 0.6893926608690485}}, "EndTime": 1606244049.173676, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173657}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893058985403838, "sum": 0.6893058985403838, "min": 0.6893058985403838}}, "EndTime": 1606244049.173723, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173712}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895544199225105, "sum": 0.6895544199225105, "min": 0.6895544199225105}}, "EndTime": 1606244049.173758, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173748}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895620356620891, "sum": 0.6895620356620891, "min": 0.6895620356620891}}, "EndTime": 1606244049.173796, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173782}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6905741879344957, "sum": 0.6905741879344957, "min": 0.6905741879344957}}, "EndTime": 1606244049.17386, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173841}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894652232881714, "sum": 0.6894652232881714, "min": 0.6894652232881714}}, "EndTime": 1606244049.173929, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.17391}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903311781878585, "sum": 0.6903311781878585, "min": 0.6903311781878585}}, "EndTime": 1606244049.174, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244049.173981}
    [0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] #quality_metric: host=algo-1, epoch=5, validation binary_classification_cross_entropy_objective <loss>=0.256532353141[0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=sampled_accuracy, value=0.992027334852[0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] Epoch 5: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] Saving model for epoch: 5[0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] Saved checkpoint to "/tmp/tmpNy3Op2/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] #progress_metric: host=algo-1, completed 40 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 312, "sum": 312.0, "min": 312}, "Total Records Seen": {"count": 1, "max": 307014, "sum": 307014.0, "min": 307014}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1606244049.181343, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1606244046.962105}
    [0m
    [34m[11/24/2020 18:54:09 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=22154.2960006 records/second[0m
    [34m[2020-11-24 18:54:10.659] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 15, "duration": 1478, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24635190971530213, "sum": 0.24635190971530213, "min": 0.24635190971530213}}, "EndTime": 1606244050.660076, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.659961}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23240220922353316, "sum": 0.23240220922353316, "min": 0.23240220922353316}}, "EndTime": 1606244050.660189, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660165}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24815606066645407, "sum": 0.24815606066645407, "min": 0.24815606066645407}}, "EndTime": 1606244050.660268, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660247}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23087830056949538, "sum": 0.23087830056949538, "min": 0.23087830056949538}}, "EndTime": 1606244050.660338, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660319}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0568072067572146, "sum": 0.0568072067572146, "min": 0.0568072067572146}}, "EndTime": 1606244050.660401, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660383}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05336268008485132, "sum": 0.05336268008485132, "min": 0.05336268008485132}}, "EndTime": 1606244050.660468, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660449}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05708854395029496, "sum": 0.05708854395029496, "min": 0.05708854395029496}}, "EndTime": 1606244050.660539, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.66052}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05222574576553033, "sum": 0.05222574576553033, "min": 0.05222574576553033}}, "EndTime": 1606244050.660608, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660589}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2512319516551738, "sum": 0.2512319516551738, "min": 0.2512319516551738}}, "EndTime": 1606244050.660677, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660659}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24246673210299746, "sum": 0.24246673210299746, "min": 0.24246673210299746}}, "EndTime": 1606244050.660744, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660725}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24936065206722338, "sum": 0.24936065206722338, "min": 0.24936065206722338}}, "EndTime": 1606244050.660813, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660795}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23268700144242266, "sum": 0.23268700144242266, "min": 0.23268700144242266}}, "EndTime": 1606244050.660885, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660866}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13016558183942523, "sum": 0.13016558183942523, "min": 0.13016558183942523}}, "EndTime": 1606244050.660952, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660934}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13000390407017298, "sum": 0.13000390407017298, "min": 0.13000390407017298}}, "EndTime": 1606244050.661016, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.660999}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301634591550243, "sum": 0.1301634591550243, "min": 0.1301634591550243}}, "EndTime": 1606244050.661082, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661064}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1300047468847158, "sum": 0.1300047468847158, "min": 0.1300047468847158}}, "EndTime": 1606244050.661149, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661131}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47577825927734374, "sum": 0.47577825927734374, "min": 0.47577825927734374}}, "EndTime": 1606244050.661218, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661199}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757980969487404, "sum": 0.4757980969487404, "min": 0.4757980969487404}}, "EndTime": 1606244050.661289, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661271}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757790863659917, "sum": 0.4757790863659917, "min": 0.4757790863659917}}, "EndTime": 1606244050.66136, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661341}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579584363042093, "sum": 0.47579584363042093, "min": 0.47579584363042093}}, "EndTime": 1606244050.661426, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661407}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47687307116450095, "sum": 0.47687307116450095, "min": 0.47687307116450095}}, "EndTime": 1606244050.661494, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661474}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47809559039680327, "sum": 0.47809559039680327, "min": 0.47809559039680327}}, "EndTime": 1606244050.661562, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661543}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47684904137436224, "sum": 0.47684904137436224, "min": 0.47684904137436224}}, "EndTime": 1606244050.661629, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661609}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4781278200733418, "sum": 0.4781278200733418, "min": 0.4781278200733418}}, "EndTime": 1606244050.661704, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661685}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894001240632972, "sum": 0.6894001240632972, "min": 0.6894001240632972}}, "EndTime": 1606244050.661766, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661748}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894348281548948, "sum": 0.6894348281548948, "min": 0.6894348281548948}}, "EndTime": 1606244050.661836, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661818}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894000630281409, "sum": 0.6894000630281409, "min": 0.6894000630281409}}, "EndTime": 1606244050.661903, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661886}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894347721022003, "sum": 0.6894347721022003, "min": 0.6894347721022003}}, "EndTime": 1606244050.661971, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.661953}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908616594587054, "sum": 0.6908616594587054, "min": 0.6908616594587054}}, "EndTime": 1606244050.662037, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.66202}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924193451550542, "sum": 0.6924193451550542, "min": 0.6924193451550542}}, "EndTime": 1606244050.662096, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.662078}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6908665186045121, "sum": 0.6908665186045121, "min": 0.6908665186045121}}, "EndTime": 1606244050.662162, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.662143}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924117630939094, "sum": 0.6924117630939094, "min": 0.6924117630939094}}, "EndTime": 1606244050.662217, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244050.6622}
    [0m
    [34m[11/24/2020 18:54:10 INFO 140513285547840] #quality_metric: host=algo-1, epoch=6, train binary_classification_cross_entropy_objective <loss>=0.246351909715[0m
    [34m[2020-11-24 18:54:10.923] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 20, "duration": 241, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24323215509446328, "sum": 0.24323215509446328, "min": 0.24323215509446328}}, "EndTime": 1606244051.205458, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205363}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22955483456135883, "sum": 0.22955483456135883, "min": 0.22955483456135883}}, "EndTime": 1606244051.205559, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205543}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24532974980531513, "sum": 0.24532974980531513, "min": 0.24532974980531513}}, "EndTime": 1606244051.205617, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205598}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2282046253813853, "sum": 0.2282046253813853, "min": 0.2282046253813853}}, "EndTime": 1606244051.205691, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205672}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05715592831538083, "sum": 0.05715592831538083, "min": 0.05715592831538083}}, "EndTime": 1606244051.20576, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205743}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.053470405506614437, "sum": 0.053470405506614437, "min": 0.053470405506614437}}, "EndTime": 1606244051.205836, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205815}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05740283825790847, "sum": 0.05740283825790847, "min": 0.05740283825790847}}, "EndTime": 1606244051.20592, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205898}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.052424106929964014, "sum": 0.052424106929964014, "min": 0.052424106929964014}}, "EndTime": 1606244051.205993, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.205974}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2482913687917335, "sum": 0.2482913687917335, "min": 0.2482913687917335}}, "EndTime": 1606244051.206064, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206044}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23916650183069718, "sum": 0.23916650183069718, "min": 0.23916650183069718}}, "EndTime": 1606244051.206132, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206114}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.24617633577515005, "sum": 0.24617633577515005, "min": 0.24617633577515005}}, "EndTime": 1606244051.206175, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206165}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2304836779667972, "sum": 0.2304836779667972, "min": 0.2304836779667972}}, "EndTime": 1606244051.206211, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206201}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13335853751071144, "sum": 0.13335853751071144, "min": 0.13335853751071144}}, "EndTime": 1606244051.206275, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206256}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1332438545276705, "sum": 0.1332438545276705, "min": 0.1332438545276705}}, "EndTime": 1606244051.206348, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206329}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13336110603836512, "sum": 0.13336110603836512, "min": 0.13336110603836512}}, "EndTime": 1606244051.206414, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206396}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13324801789174204, "sum": 0.13324801789174204, "min": 0.13324801789174204}}, "EndTime": 1606244051.206481, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206463}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47655292911945396, "sum": 0.47655292911945396, "min": 0.47655292911945396}}, "EndTime": 1606244051.206548, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.20653}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47657392919548475, "sum": 0.47657392919548475, "min": 0.47657392919548475}}, "EndTime": 1606244051.206617, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206596}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765539817921781, "sum": 0.4765539817921781, "min": 0.4765539817921781}}, "EndTime": 1606244051.206689, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.20667}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476478842311056, "sum": 0.476478842311056, "min": 0.476478842311056}}, "EndTime": 1606244051.206755, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206736}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4744788808229883, "sum": 0.4744788808229883, "min": 0.4744788808229883}}, "EndTime": 1606244051.206823, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206805}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47638051796145675, "sum": 0.47638051796145675, "min": 0.47638051796145675}}, "EndTime": 1606244051.206897, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206872}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4771147882205207, "sum": 0.4771147882205207, "min": 0.4771147882205207}}, "EndTime": 1606244051.206968, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.206947}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4775395975258775, "sum": 0.4775395975258775, "min": 0.4775395975258775}}, "EndTime": 1606244051.207034, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207015}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893091869012874, "sum": 0.6893091869012874, "min": 0.6893091869012874}}, "EndTime": 1606244051.207106, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207087}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893644522193042, "sum": 0.6893644522193042, "min": 0.6893644522193042}}, "EndTime": 1606244051.207179, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207158}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892752742053553, "sum": 0.6892752742053553, "min": 0.6892752742053553}}, "EndTime": 1606244051.207251, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207231}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896272877160323, "sum": 0.6896272877160323, "min": 0.6896272877160323}}, "EndTime": 1606244051.207323, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207303}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895198728919534, "sum": 0.6895198728919534, "min": 0.6895198728919534}}, "EndTime": 1606244051.207396, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207376}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904815341143821, "sum": 0.6904815341143821, "min": 0.6904815341143821}}, "EndTime": 1606244051.207469, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207449}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894630968396143, "sum": 0.6894630968396143, "min": 0.6894630968396143}}, "EndTime": 1606244051.207542, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207521}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904271898849988, "sum": 0.6904271898849988, "min": 0.6904271898849988}}, "EndTime": 1606244051.207616, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244051.207598}
    [0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] #quality_metric: host=algo-1, epoch=6, validation binary_classification_cross_entropy_objective <loss>=0.243232155094[0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=sampled_accuracy, value=0.993003579564[0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] Epoch 6: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] Saving model for epoch: 6[0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] Saved checkpoint to "/tmp/tmpHVdqgB/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] #progress_metric: host=algo-1, completed 46 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 362, "sum": 362.0, "min": 362}, "Total Records Seen": {"count": 1, "max": 356183, "sum": 356183.0, "min": 356183}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1606244051.215361, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1606244049.181664}
    [0m
    [34m[11/24/2020 18:54:11 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=24175.2961956 records/second[0m
    [34m[2020-11-24 18:54:12.952] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 17, "duration": 1736, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2346909718416175, "sum": 0.2346909718416175, "min": 0.2346909718416175}}, "EndTime": 1606244052.952989, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.952886}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22097825155452805, "sum": 0.22097825155452805, "min": 0.22097825155452805}}, "EndTime": 1606244052.953084, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953063}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2368662274419045, "sum": 0.2368662274419045, "min": 0.2368662274419045}}, "EndTime": 1606244052.953156, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953137}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21972996707838408, "sum": 0.21972996707838408, "min": 0.21972996707838408}}, "EndTime": 1606244052.953219, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953203}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.052043523983079557, "sum": 0.052043523983079557, "min": 0.052043523983079557}}, "EndTime": 1606244052.953287, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953268}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04855898432828942, "sum": 0.04855898432828942, "min": 0.04855898432828942}}, "EndTime": 1606244052.953355, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953336}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05226955382677974, "sum": 0.05226955382677974, "min": 0.05226955382677974}}, "EndTime": 1606244052.953423, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953404}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04761459226024394, "sum": 0.04761459226024394, "min": 0.04761459226024394}}, "EndTime": 1606244052.953488, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953472}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.239960345832669, "sum": 0.239960345832669, "min": 0.239960345832669}}, "EndTime": 1606244052.953553, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953534}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23045772599200814, "sum": 0.23045772599200814, "min": 0.23045772599200814}}, "EndTime": 1606244052.953629, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953603}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23778845619668765, "sum": 0.23778845619668765, "min": 0.23778845619668765}}, "EndTime": 1606244052.953698, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953679}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22243555839694276, "sum": 0.22243555839694276, "min": 0.22243555839694276}}, "EndTime": 1606244052.953777, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953758}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301610938870177, "sum": 0.1301610938870177, "min": 0.1301610938870177}}, "EndTime": 1606244052.953845, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953826}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13004203952088647, "sum": 0.13004203952088647, "min": 0.13004203952088647}}, "EndTime": 1606244052.953912, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953894}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13016171949736927, "sum": 0.13016171949736927, "min": 0.13016171949736927}}, "EndTime": 1606244052.953977, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.953959}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13004733105095065, "sum": 0.13004733105095065, "min": 0.13004733105095065}}, "EndTime": 1606244052.954045, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954026}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47577385664959343, "sum": 0.47577385664959343, "min": 0.47577385664959343}}, "EndTime": 1606244052.95411, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954091}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757984095982143, "sum": 0.4757984095982143, "min": 0.4757984095982143}}, "EndTime": 1606244052.954178, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.95416}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47577308063117824, "sum": 0.47577308063117824, "min": 0.47577308063117824}}, "EndTime": 1606244052.954246, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954227}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579602735869736, "sum": 0.47579602735869736, "min": 0.47579602735869736}}, "EndTime": 1606244052.954312, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954294}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47677793604013874, "sum": 0.47677793604013874, "min": 0.47677793604013874}}, "EndTime": 1606244052.954378, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.95436}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47813498609893174, "sum": 0.47813498609893174, "min": 0.47813498609893174}}, "EndTime": 1606244052.954445, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954426}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4767619579081633, "sum": 0.4767619579081633, "min": 0.4767619579081633}}, "EndTime": 1606244052.954513, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954496}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.478142292256258, "sum": 0.478142292256258, "min": 0.478142292256258}}, "EndTime": 1606244052.95458, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954561}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893927500199298, "sum": 0.6893927500199298, "min": 0.6893927500199298}}, "EndTime": 1606244052.954646, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954627}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894334866270727, "sum": 0.6894334866270727, "min": 0.6894334866270727}}, "EndTime": 1606244052.954721, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954701}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893924473353794, "sum": 0.6893924473353794, "min": 0.6893924473353794}}, "EndTime": 1606244052.95479, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954771}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894340471540179, "sum": 0.6894340471540179, "min": 0.6894340471540179}}, "EndTime": 1606244052.954864, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954845}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6907433720802775, "sum": 0.6907433720802775, "min": 0.6907433720802775}}, "EndTime": 1606244052.954931, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.954914}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924664082429847, "sum": 0.6924664082429847, "min": 0.6924664082429847}}, "EndTime": 1606244052.954997, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.95498}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6907466405751754, "sum": 0.6907466405751754, "min": 0.6907466405751754}}, "EndTime": 1606244052.955036, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.955026}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924656845404177, "sum": 0.6924656845404177, "min": 0.6924656845404177}}, "EndTime": 1606244052.9551, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244052.955082}
    [0m
    [34m[11/24/2020 18:54:12 INFO 140513285547840] #quality_metric: host=algo-1, epoch=7, train binary_classification_cross_entropy_objective <loss>=0.234690971842[0m
    [34m[2020-11-24 18:54:13.226] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 23, "duration": 252, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23336397022678065, "sum": 0.23336397022678065, "min": 0.23336397022678065}}, "EndTime": 1606244053.496152, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496034}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2199469270591351, "sum": 0.2199469270591351, "min": 0.2199469270591351}}, "EndTime": 1606244053.496251, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496235}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23577498148289092, "sum": 0.23577498148289092, "min": 0.23577498148289092}}, "EndTime": 1606244053.496295, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496283}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21882585543512406, "sum": 0.21882585543512406, "min": 0.21882585543512406}}, "EndTime": 1606244053.49635, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496333}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05310292566710949, "sum": 0.05310292566710949, "min": 0.05310292566710949}}, "EndTime": 1606244053.496512, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496399}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04931672683105071, "sum": 0.04931672683105071, "min": 0.04931672683105071}}, "EndTime": 1606244053.496663, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496638}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05330717854572747, "sum": 0.05330717854572747, "min": 0.05330717854572747}}, "EndTime": 1606244053.496743, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496724}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04844047881243402, "sum": 0.04844047881243402, "min": 0.04844047881243402}}, "EndTime": 1606244053.496813, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496793}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23875156505620715, "sum": 0.23875156505620715, "min": 0.23875156505620715}}, "EndTime": 1606244053.496881, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496864}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22906837748891556, "sum": 0.22906837748891556, "min": 0.22906837748891556}}, "EndTime": 1606244053.496927, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496916}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23638989193758358, "sum": 0.23638989193758358, "min": 0.23638989193758358}}, "EndTime": 1606244053.496981, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.496964}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22184493166982178, "sum": 0.22184493166982178, "min": 0.22184493166982178}}, "EndTime": 1606244053.497045, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497028}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1333495342417257, "sum": 0.1333495342417257, "min": 0.1333495342417257}}, "EndTime": 1606244053.497113, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497094}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1333025958070218, "sum": 0.1333025958070218, "min": 0.1333025958070218}}, "EndTime": 1606244053.497179, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497161}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13335358681447568, "sum": 0.13335358681447568, "min": 0.13335358681447568}}, "EndTime": 1606244053.497243, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497225}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13331043142237417, "sum": 0.13331043142237417, "min": 0.13331043142237417}}, "EndTime": 1606244053.49729, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497279}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47654131992686427, "sum": 0.47654131992686427, "min": 0.47654131992686427}}, "EndTime": 1606244053.49735, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497332}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765353303680271, "sum": 0.4765353303680271, "min": 0.4765353303680271}}, "EndTime": 1606244053.497414, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497395}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47654256625164615, "sum": 0.47654256625164615, "min": 0.47654256625164615}}, "EndTime": 1606244053.497476, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497459}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4764372915310771, "sum": 0.4764372915310771, "min": 0.4764372915310771}}, "EndTime": 1606244053.497544, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497525}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4741827185519386, "sum": 0.4741827185519386, "min": 0.4741827185519386}}, "EndTime": 1606244053.497611, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497593}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4762215854832221, "sum": 0.4762215854832221, "min": 0.4762215854832221}}, "EndTime": 1606244053.497677, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497658}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47761011294344136, "sum": 0.47761011294344136, "min": 0.47761011294344136}}, "EndTime": 1606244053.497743, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497724}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47662640517905447, "sum": 0.47662640517905447, "min": 0.47662640517905447}}, "EndTime": 1606244053.497814, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497795}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892953097451754, "sum": 0.6892953097451754, "min": 0.6892953097451754}}, "EndTime": 1606244053.497879, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497861}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893660001944069, "sum": 0.6893660001944069, "min": 0.6893660001944069}}, "EndTime": 1606244053.497939, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.497922}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892588460557902, "sum": 0.6892588460557902, "min": 0.6892588460557902}}, "EndTime": 1606244053.497986, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.49797}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896970285253652, "sum": 0.6896970285253652, "min": 0.6896970285253652}}, "EndTime": 1606244053.49805, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.498031}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895340963873225, "sum": 0.6895340963873225, "min": 0.6895340963873225}}, "EndTime": 1606244053.498112, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.498094}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903829528033986, "sum": 0.6903829528033986, "min": 0.6903829528033986}}, "EndTime": 1606244053.498177, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.498159}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689489732687069, "sum": 0.689489732687069, "min": 0.689489732687069}}, "EndTime": 1606244053.498244, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.498226}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904487870548743, "sum": 0.6904487870548743, "min": 0.6904487870548743}}, "EndTime": 1606244053.49831, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244053.498291}
    [0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] #quality_metric: host=algo-1, epoch=7, validation binary_classification_cross_entropy_objective <loss>=0.233363970227[0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=sampled_accuracy, value=0.993817116824[0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] Epoch 7: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] Saving model for epoch: 7[0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] Saved checkpoint to "/tmp/tmpwVT4P3/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] #progress_metric: host=algo-1, completed 53 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 412, "sum": 412.0, "min": 412}, "Total Records Seen": {"count": 1, "max": 405352, "sum": 405352.0, "min": 405352}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1606244053.504934, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1606244051.215708}
    [0m
    [34m[11/24/2020 18:54:13 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=21477.0042645 records/second[0m
    [34m[2020-11-24 18:54:15.245] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 19, "duration": 1740, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2258969150465362, "sum": 0.2258969150465362, "min": 0.2258969150465362}}, "EndTime": 1606244055.24565, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245537}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2124336258245974, "sum": 0.2124336258245974, "min": 0.2124336258245974}}, "EndTime": 1606244055.245762, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245739}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22834320722307477, "sum": 0.22834320722307477, "min": 0.22834320722307477}}, "EndTime": 1606244055.245824, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245809}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21138077809859296, "sum": 0.21138077809859296, "min": 0.21138077809859296}}, "EndTime": 1606244055.245878, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245865}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.048474902795285595, "sum": 0.048474902795285595, "min": 0.048474902795285595}}, "EndTime": 1606244055.245929, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245916}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.045005735903370134, "sum": 0.045005735903370134, "min": 0.045005735903370134}}, "EndTime": 1606244055.24598, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.245967}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04866288406994878, "sum": 0.04866288406994878, "min": 0.04866288406994878}}, "EndTime": 1606244055.24603, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246017}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04420795137055066, "sum": 0.04420795137055066, "min": 0.04420795137055066}}, "EndTime": 1606244055.246079, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246066}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23146703011648995, "sum": 0.23146703011648995, "min": 0.23146703011648995}}, "EndTime": 1606244055.246129, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246116}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22151645613689813, "sum": 0.22151645613689813, "min": 0.22151645613689813}}, "EndTime": 1606244055.246179, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246166}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22908673967633927, "sum": 0.22908673967633927, "min": 0.22908673967633927}}, "EndTime": 1606244055.246228, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246215}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21474226223692602, "sum": 0.21474226223692602, "min": 0.21474226223692602}}, "EndTime": 1606244055.246278, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246265}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301617458109953, "sum": 0.1301617458109953, "min": 0.1301617458109953}}, "EndTime": 1606244055.246327, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246314}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13009669915024116, "sum": 0.13009669915024116, "min": 0.13009669915024116}}, "EndTime": 1606244055.246378, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246363}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301626642966757, "sum": 0.1301626642966757, "min": 0.1301626642966757}}, "EndTime": 1606244055.246439, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246424}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13010287958261918, "sum": 0.13010287958261918, "min": 0.13010287958261918}}, "EndTime": 1606244055.246489, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246476}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47576914074956156, "sum": 0.47576914074956156, "min": 0.47576914074956156}}, "EndTime": 1606244055.246538, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246525}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757983491858658, "sum": 0.4757983491858658, "min": 0.4757983491858658}}, "EndTime": 1606244055.246587, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246574}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757684400908801, "sum": 0.4757684400908801, "min": 0.4757684400908801}}, "EndTime": 1606244055.246636, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246623}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757962932975925, "sum": 0.4757962932975925, "min": 0.4757962932975925}}, "EndTime": 1606244055.246723, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246705}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47669138258330673, "sum": 0.47669138258330673, "min": 0.47669138258330673}}, "EndTime": 1606244055.246781, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246768}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47809342302594865, "sum": 0.47809342302594865, "min": 0.47809342302594865}}, "EndTime": 1606244055.246855, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246818}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4766733834402902, "sum": 0.4766733834402902, "min": 0.4766733834402902}}, "EndTime": 1606244055.246912, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246898}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780692157356107, "sum": 0.4780692157356107, "min": 0.4780692157356107}}, "EndTime": 1606244055.246962, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246949}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.68938590038066, "sum": 0.68938590038066, "min": 0.68938590038066}}, "EndTime": 1606244055.24701, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.246997}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894343971719548, "sum": 0.6894343971719548, "min": 0.6894343971719548}}, "EndTime": 1606244055.24706, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247047}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893856400470344, "sum": 0.6893856400470344, "min": 0.6893856400470344}}, "EndTime": 1606244055.247109, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247096}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689434880470743, "sum": 0.689434880470743, "min": 0.689434880470743}}, "EndTime": 1606244055.247158, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247145}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6906391800860969, "sum": 0.6906391800860969, "min": 0.6906391800860969}}, "EndTime": 1606244055.247206, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247193}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923831562898597, "sum": 0.6923831562898597, "min": 0.6923831562898597}}, "EndTime": 1606244055.247255, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247242}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6906411357023278, "sum": 0.6906411357023278, "min": 0.6906411357023278}}, "EndTime": 1606244055.247299, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247288}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923842736069037, "sum": 0.6923842736069037, "min": 0.6923842736069037}}, "EndTime": 1606244055.247332, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.247323}
    [0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] #quality_metric: host=algo-1, epoch=8, train binary_classification_cross_entropy_objective <loss>=0.225896915047[0m
    [34m[2020-11-24 18:54:15.559] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 26, "duration": 289, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22581909581887927, "sum": 0.22581909581887927, "min": 0.22581909581887927}}, "EndTime": 1606244055.835996, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.835869}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21266803002846268, "sum": 0.21266803002846268, "min": 0.21266803002846268}}, "EndTime": 1606244055.836106, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836083}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22845951866102732, "sum": 0.22845951866102732, "min": 0.22845951866102732}}, "EndTime": 1606244055.83618, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836159}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21170807505445297, "sum": 0.21170807505445297, "min": 0.21170807505445297}}, "EndTime": 1606244055.836255, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836234}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.049968313868256065, "sum": 0.049968313868256065, "min": 0.049968313868256065}}, "EndTime": 1606244055.836324, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836306}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.046216469419301236, "sum": 0.046216469419301236, "min": 0.046216469419301236}}, "EndTime": 1606244055.836394, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836375}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.05014345128412864, "sum": 0.05014345128412864, "min": 0.05014345128412864}}, "EndTime": 1606244055.836456, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.83644}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04547296329154434, "sum": 0.04547296329154434, "min": 0.04547296329154434}}, "EndTime": 1606244055.836518, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836501}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.23145622715203645, "sum": 0.23145622715203645, "min": 0.23145622715203645}}, "EndTime": 1606244055.836591, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836571}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22145461818894926, "sum": 0.22145461818894926, "min": 0.22145461818894926}}, "EndTime": 1606244055.836661, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836642}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22892547406321662, "sum": 0.22892547406321662, "min": 0.22892547406321662}}, "EndTime": 1606244055.83673, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836712}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2152610121508821, "sum": 0.2152610121508821, "min": 0.2152610121508821}}, "EndTime": 1606244055.836783, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836765}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13335835534122764, "sum": 0.13335835534122764, "min": 0.13335835534122764}}, "EndTime": 1606244055.836848, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836831}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1333889773487075, "sum": 0.1333889773487075, "min": 0.1333889773487075}}, "EndTime": 1606244055.83691, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836897}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13336357230904278, "sum": 0.13336357230904278, "min": 0.13336357230904278}}, "EndTime": 1606244055.836955, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.836939}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13340309734574135, "sum": 0.13340309734574135, "min": 0.13340309734574135}}, "EndTime": 1606244055.837037, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837018}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765330425427552, "sum": 0.4765330425427552, "min": 0.4765330425427552}}, "EndTime": 1606244055.837116, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837095}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765050424413808, "sum": 0.4765050424413808, "min": 0.4765050424413808}}, "EndTime": 1606244055.837199, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837179}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47653471341243286, "sum": 0.47653471341243286, "min": 0.47653471341243286}}, "EndTime": 1606244055.83727, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.83725}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47639848787932765, "sum": 0.47639848787932765, "min": 0.47639848787932765}}, "EndTime": 1606244055.83735, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.83733}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4740894564657463, "sum": 0.4740894564657463, "min": 0.4740894564657463}}, "EndTime": 1606244055.837397, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837385}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4759517437543537, "sum": 0.4759517437543537, "min": 0.4759517437543537}}, "EndTime": 1606244055.837451, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837434}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47790172314977475, "sum": 0.47790172314977475, "min": 0.47790172314977475}}, "EndTime": 1606244055.837516, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837497}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47641246682690597, "sum": 0.47641246682690597, "min": 0.47641246682690597}}, "EndTime": 1606244055.837593, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837573}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892835702098544, "sum": 0.6892835702098544, "min": 0.6892835702098544}}, "EndTime": 1606244055.837667, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837647}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893650108310092, "sum": 0.6893650108310092, "min": 0.6893650108310092}}, "EndTime": 1606244055.837738, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837719}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892495743428462, "sum": 0.6892495743428462, "min": 0.6892495743428462}}, "EndTime": 1606244055.83782, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837799}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6897178423974943, "sum": 0.6897178423974943, "min": 0.6897178423974943}}, "EndTime": 1606244055.837893, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837873}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895555359045613, "sum": 0.6895555359045613, "min": 0.6895555359045613}}, "EndTime": 1606244055.837965, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.837944}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.690157040299013, "sum": 0.690157040299013, "min": 0.690157040299013}}, "EndTime": 1606244055.838036, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.838016}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895060292525436, "sum": 0.6895060292525436, "min": 0.6895060292525436}}, "EndTime": 1606244055.838109, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.838088}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902700822050006, "sum": 0.6902700822050006, "min": 0.6902700822050006}}, "EndTime": 1606244055.838179, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244055.83816}
    [0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] #quality_metric: host=algo-1, epoch=8, validation binary_classification_cross_entropy_objective <loss>=0.225819095819[0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=sampled_accuracy, value=0.994142531728[0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] Epoch 8: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] Saving model for epoch: 8[0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] Saved checkpoint to "/tmp/tmpB8X7Iv/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] #progress_metric: host=algo-1, completed 60 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 462, "sum": 462.0, "min": 462}, "Total Records Seen": {"count": 1, "max": 454521, "sum": 454521.0, "min": 454521}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1606244055.845361, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1606244053.505269}
    [0m
    [34m[11/24/2020 18:54:15 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=21010.2553749 records/second[0m
    [34m[2020-11-24 18:54:17.392] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 21, "duration": 1546, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21907198069046954, "sum": 0.21907198069046954, "min": 0.21907198069046954}}, "EndTime": 1606244057.392177, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.39207}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20582018824986048, "sum": 0.20582018824986048, "min": 0.20582018824986048}}, "EndTime": 1606244057.392284, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392262}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22171710641043527, "sum": 0.22171710641043527, "min": 0.22171710641043527}}, "EndTime": 1606244057.392356, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392337}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20490554653868384, "sum": 0.20490554653868384, "min": 0.20490554653868384}}, "EndTime": 1606244057.39242, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392402}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04571539345566107, "sum": 0.04571539345566107, "min": 0.04571539345566107}}, "EndTime": 1606244057.392497, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392477}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04230730228034817, "sum": 0.04230730228034817, "min": 0.04230730228034817}}, "EndTime": 1606244057.392564, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392546}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04587532370431083, "sum": 0.04587532370431083, "min": 0.04587532370431083}}, "EndTime": 1606244057.392633, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392614}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04162342032607721, "sum": 0.04162342032607721, "min": 0.04162342032607721}}, "EndTime": 1606244057.392696, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392678}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22487395430584342, "sum": 0.22487395430584342, "min": 0.22487395430584342}}, "EndTime": 1606244057.392758, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392739}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.214637349965621, "sum": 0.214637349965621, "min": 0.214637349965621}}, "EndTime": 1606244057.392817, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392798}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22235042151626275, "sum": 0.22235042151626275, "min": 0.22235042151626275}}, "EndTime": 1606244057.392875, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392857}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20875079283422354, "sum": 0.20875079283422354, "min": 0.20875079283422354}}, "EndTime": 1606244057.392942, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392924}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13016119026651188, "sum": 0.13016119026651188, "min": 0.13016119026651188}}, "EndTime": 1606244057.393008, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.392989}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301514598301479, "sum": 0.1301514598301479, "min": 0.1301514598301479}}, "EndTime": 1606244057.393076, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393057}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13016255748515226, "sum": 0.13016255748515226, "min": 0.13016255748515226}}, "EndTime": 1606244057.393143, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393123}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13015661325260083, "sum": 0.13015661325260083, "min": 0.13015661325260083}}, "EndTime": 1606244057.393208, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.39319}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47576419877032844, "sum": 0.47576419877032844, "min": 0.47576419877032844}}, "EndTime": 1606244057.393281, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393262}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757986637037628, "sum": 0.4757986637037628, "min": 0.4757986637037628}}, "EndTime": 1606244057.393348, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393329}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757636046117666, "sum": 0.4757636046117666, "min": 0.4757636046117666}}, "EndTime": 1606244057.393418, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393399}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579602673588967, "sum": 0.47579602673588967, "min": 0.47579602673588967}}, "EndTime": 1606244057.393487, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393468}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476607872787787, "sum": 0.476607872787787, "min": 0.476607872787787}}, "EndTime": 1606244057.393559, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393541}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47811240745077327, "sum": 0.47811240745077327, "min": 0.47811240745077327}}, "EndTime": 1606244057.393626, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393606}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765857898945711, "sum": 0.4765857898945711, "min": 0.4765857898945711}}, "EndTime": 1606244057.393692, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393672}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780979134695871, "sum": 0.4780979134695871, "min": 0.4780979134695871}}, "EndTime": 1606244057.393758, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.39374}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893793746014031, "sum": 0.6893793746014031, "min": 0.6893793746014031}}, "EndTime": 1606244057.39383, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393811}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894320566605548, "sum": 0.6894320566605548, "min": 0.6894320566605548}}, "EndTime": 1606244057.3939, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393881}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893792064433195, "sum": 0.6893792064433195, "min": 0.6893792064433195}}, "EndTime": 1606244057.393966, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.393948}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894339749083227, "sum": 0.6894339749083227, "min": 0.6894339749083227}}, "EndTime": 1606244057.394034, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.394016}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6905408013791454, "sum": 0.6905408013791454, "min": 0.6905408013791454}}, "EndTime": 1606244057.394096, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.39408}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6924017944335937, "sum": 0.6924017944335937, "min": 0.6924017944335937}}, "EndTime": 1606244057.394141, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.394131}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6905417206433354, "sum": 0.6905417206433354, "min": 0.6905417206433354}}, "EndTime": 1606244057.394191, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.394175}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.692407855598294, "sum": 0.692407855598294, "min": 0.692407855598294}}, "EndTime": 1606244057.394254, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.394237}
    [0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] #quality_metric: host=algo-1, epoch=9, train binary_classification_cross_entropy_objective <loss>=0.21907198069[0m
    [34m[2020-11-24 18:54:17.667] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 29, "duration": 255, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21988844010327951, "sum": 0.21988844010327951, "min": 0.21988844010327951}}, "EndTime": 1606244057.958072, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.957978}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2068988478536758, "sum": 0.2068988478536758, "min": 0.2068988478536758}}, "EndTime": 1606244057.958174, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958158}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22269690110479393, "sum": 0.22269690110479393, "min": 0.22269690110479393}}, "EndTime": 1606244057.958242, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958223}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2060529331489064, "sum": 0.2060529331489064, "min": 0.2060529331489064}}, "EndTime": 1606244057.958317, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958299}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04748050622663861, "sum": 0.04748050622663861, "min": 0.04748050622663861}}, "EndTime": 1606244057.958384, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958365}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0438130552047943, "sum": 0.0438130552047943, "min": 0.0438130552047943}}, "EndTime": 1606244057.958454, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958434}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04763215006348212, "sum": 0.04763215006348212, "min": 0.04763215006348212}}, "EndTime": 1606244057.958527, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958508}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04317397850947779, "sum": 0.04317397850947779, "min": 0.04317397850947779}}, "EndTime": 1606244057.958591, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958574}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2257157234815057, "sum": 0.2257157234815057, "min": 0.2257157234815057}}, "EndTime": 1606244057.958655, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958638}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21545903782793338, "sum": 0.21545903782793338, "min": 0.21545903782793338}}, "EndTime": 1606244057.958716, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958699}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.22306995162194224, "sum": 0.22306995162194224, "min": 0.22306995162194224}}, "EndTime": 1606244057.958783, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958766}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2099990326459036, "sum": 0.2099990326459036, "min": 0.2099990326459036}}, "EndTime": 1606244057.958849, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958831}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.133377233313459, "sum": 0.133377233313459, "min": 0.133377233313459}}, "EndTime": 1606244057.958923, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.95891}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13357497456405049, "sum": 0.13357497456405049, "min": 0.13357497456405049}}, "EndTime": 1606244057.95898, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.958963}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1333851192038648, "sum": 0.1333851192038648, "min": 0.1333851192038648}}, "EndTime": 1606244057.959048, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959029}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13359715610383738, "sum": 0.13359715610383738, "min": 0.13359715610383738}}, "EndTime": 1606244057.959106, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959088}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47652804483003425, "sum": 0.47652804483003425, "min": 0.47652804483003425}}, "EndTime": 1606244057.959185, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959166}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476470536375682, "sum": 0.476470536375682, "min": 0.476470536375682}}, "EndTime": 1606244057.959256, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959236}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765301750026694, "sum": 0.4765301750026694, "min": 0.4765301750026694}}, "EndTime": 1606244057.959326, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959307}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763522633814168, "sum": 0.4763522633814168, "min": 0.4763522633814168}}, "EndTime": 1606244057.959395, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959375}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47412300296067184, "sum": 0.47412300296067184, "min": 0.47412300296067184}}, "EndTime": 1606244057.959465, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959446}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47514136575387894, "sum": 0.47514136575387894, "min": 0.47514136575387894}}, "EndTime": 1606244057.959529, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.95951}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47808274810026335, "sum": 0.47808274810026335, "min": 0.47808274810026335}}, "EndTime": 1606244057.959603, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959584}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47625307876979184, "sum": 0.47625307876979184, "min": 0.47625307876979184}}, "EndTime": 1606244057.959666, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959647}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892743444272063, "sum": 0.6892743444272063, "min": 0.6892743444272063}}, "EndTime": 1606244057.959734, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959716}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893685487051411, "sum": 0.6893685487051411, "min": 0.6893685487051411}}, "EndTime": 1606244057.9598, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959782}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892435661636188, "sum": 0.6892435661636188, "min": 0.6892435661636188}}, "EndTime": 1606244057.959901, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959881}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6897142064559743, "sum": 0.6897142064559743, "min": 0.6897142064559743}}, "EndTime": 1606244057.959997, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.959977}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895663146277495, "sum": 0.6895663146277495, "min": 0.6895663146277495}}, "EndTime": 1606244057.960065, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.960052}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6900977864934225, "sum": 0.6900977864934225, "min": 0.6900977864934225}}, "EndTime": 1606244057.960119, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.960101}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895211130099385, "sum": 0.6895211130099385, "min": 0.6895211130099385}}, "EndTime": 1606244057.960186, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.960168}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902821221492829, "sum": 0.6902821221492829, "min": 0.6902821221492829}}, "EndTime": 1606244057.960255, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244057.960236}
    [0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] #quality_metric: host=algo-1, epoch=9, validation binary_classification_cross_entropy_objective <loss>=0.219888440103[0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=sampled_accuracy, value=0.99430523918[0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] Epoch 9: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] Saving model for epoch: 9[0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] Saved checkpoint to "/tmp/tmpXU5fJ0/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] #progress_metric: host=algo-1, completed 66 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 512, "sum": 512.0, "min": 512}, "Total Records Seen": {"count": 1, "max": 503690, "sum": 503690.0, "min": 503690}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1606244057.967503, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1606244055.845665}
    [0m
    [34m[11/24/2020 18:54:17 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=23171.2943635 records/second[0m
    [34m[2020-11-24 18:54:19.404] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 23, "duration": 1436, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21363399832589286, "sum": 0.21363399832589286, "min": 0.21363399832589286}}, "EndTime": 1606244059.404632, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.404525}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20048853691256777, "sum": 0.20048853691256777, "min": 0.20048853691256777}}, "EndTime": 1606244059.404751, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.404729}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21642485077527104, "sum": 0.21642485077527104, "min": 0.21642485077527104}}, "EndTime": 1606244059.40483, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.404809}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1996714091398278, "sum": 0.1996714091398278, "min": 0.1996714091398278}}, "EndTime": 1606244059.404902, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.404883}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.043533548627580916, "sum": 0.043533548627580916, "min": 0.043533548627580916}}, "EndTime": 1606244059.404972, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.404952}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04019319798995037, "sum": 0.04019319798995037, "min": 0.04019319798995037}}, "EndTime": 1606244059.405051, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.40503}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04367190824236188, "sum": 0.04367190824236188, "min": 0.04367190824236188}}, "EndTime": 1606244059.40512, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405101}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03960031326449647, "sum": 0.03960031326449647, "min": 0.03960031326449647}}, "EndTime": 1606244059.405187, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405169}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2196152508794045, "sum": 0.2196152508794045, "min": 0.2196152508794045}}, "EndTime": 1606244059.405255, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405236}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2091351446034957, "sum": 0.2091351446034957, "min": 0.2091351446034957}}, "EndTime": 1606244059.405324, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405305}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2169955297976124, "sum": 0.2169955297976124, "min": 0.2169955297976124}}, "EndTime": 1606244059.40539, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405372}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20388339607083067, "sum": 0.20388339607083067, "min": 0.20388339607083067}}, "EndTime": 1606244059.405458, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405439}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13015853508151307, "sum": 0.13015853508151307, "min": 0.13015853508151307}}, "EndTime": 1606244059.405527, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405508}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1301737561907087, "sum": 0.1301737561907087, "min": 0.1301737561907087}}, "EndTime": 1606244059.405593, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405575}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.130160624757105, "sum": 0.130160624757105, "min": 0.130160624757105}}, "EndTime": 1606244059.405667, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405647}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13017626937554808, "sum": 0.13017626937554808, "min": 0.13017626937554808}}, "EndTime": 1606244059.405735, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405716}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47575912413305166, "sum": 0.47575912413305166, "min": 0.47575912413305166}}, "EndTime": 1606244059.405801, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405783}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757984650281011, "sum": 0.4757984650281011, "min": 0.4757984650281011}}, "EndTime": 1606244059.405868, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.40585}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757586327377631, "sum": 0.4757586327377631, "min": 0.4757586327377631}}, "EndTime": 1606244059.405935, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405917}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579590092873086, "sum": 0.47579590092873086, "min": 0.47579590092873086}}, "EndTime": 1606244059.406003, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.405986}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765359652772242, "sum": 0.4765359652772242, "min": 0.4765359652772242}}, "EndTime": 1606244059.40607, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406052}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780510882942044, "sum": 0.4780510882942044, "min": 0.4780510882942044}}, "EndTime": 1606244059.406137, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406118}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47651136561802454, "sum": 0.47651136561802454, "min": 0.47651136561802454}}, "EndTime": 1606244059.406204, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406185}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47805164259307237, "sum": 0.47805164259307237, "min": 0.47805164259307237}}, "EndTime": 1606244059.406272, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406253}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893735999282525, "sum": 0.6893735999282525, "min": 0.6893735999282525}}, "EndTime": 1606244059.406338, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.40632}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894318785375478, "sum": 0.6894318785375478, "min": 0.6894318785375478}}, "EndTime": 1606244059.406406, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406386}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893735052614796, "sum": 0.6893735052614796, "min": 0.6893735052614796}}, "EndTime": 1606244059.406471, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406453}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894341306102519, "sum": 0.6894341306102519, "min": 0.6894341306102519}}, "EndTime": 1606244059.40655, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.40653}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904595897441007, "sum": 0.6904595897441007, "min": 0.6904595897441007}}, "EndTime": 1606244059.406616, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406598}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923248353296396, "sum": 0.6923248353296396, "min": 0.6923248353296396}}, "EndTime": 1606244059.40668, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406662}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6904599721480389, "sum": 0.6904599721480389, "min": 0.6904599721480389}}, "EndTime": 1606244059.406717, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406708}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923312751614318, "sum": 0.6923312751614318, "min": 0.6923312751614318}}, "EndTime": 1606244059.406757, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.406742}
    [0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] #quality_metric: host=algo-1, epoch=10, train binary_classification_cross_entropy_objective <loss>=0.213633998326[0m
    [34m[2020-11-24 18:54:19.694] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 32, "duration": 270, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21510754568818102, "sum": 0.21510754568818102, "min": 0.21510754568818102}}, "EndTime": 1606244059.978442, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978353}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2021823855746735, "sum": 0.2021823855746735, "min": 0.2021823855746735}}, "EndTime": 1606244059.978548, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978526}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21803762607841354, "sum": 0.21803762607841354, "min": 0.21803762607841354}}, "EndTime": 1606244059.978619, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.9786}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20141503060636015, "sum": 0.20141503060636015, "min": 0.20141503060636015}}, "EndTime": 1606244059.978694, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978678}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.045483254858774816, "sum": 0.045483254858774816, "min": 0.045483254858774816}}, "EndTime": 1606244059.978761, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978742}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.041935101340579706, "sum": 0.041935101340579706, "min": 0.041935101340579706}}, "EndTime": 1606244059.978819, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978802}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04561574940878523, "sum": 0.04561574940878523, "min": 0.04561574940878523}}, "EndTime": 1606244059.978885, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.978867}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04138126227431293, "sum": 0.04138126227431293, "min": 0.04138126227431293}}, "EndTime": 1606244059.978959, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.97894}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2210785964782099, "sum": 0.2210785964782099, "min": 0.2210785964782099}}, "EndTime": 1606244059.979039, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979017}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21059888323936873, "sum": 0.21059888323936873, "min": 0.21059888323936873}}, "EndTime": 1606244059.97911, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.97909}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2183587998782157, "sum": 0.2183587998782157, "min": 0.2183587998782157}}, "EndTime": 1606244059.979181, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979162}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20565726847823962, "sum": 0.20565726847823962, "min": 0.20565726847823962}}, "EndTime": 1606244059.979251, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979231}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13340078717600123, "sum": 0.13340078717600123, "min": 0.13340078717600123}}, "EndTime": 1606244059.979324, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979304}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1337455981645916, "sum": 0.1337455981645916, "min": 0.1337455981645916}}, "EndTime": 1606244059.979405, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979385}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13341148691770116, "sum": 0.13341148691770116, "min": 0.13341148691770116}}, "EndTime": 1606244059.979475, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979456}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13377086386166992, "sum": 0.13377086386166992, "min": 0.13377086386166992}}, "EndTime": 1606244059.979542, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979524}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47652641492521497, "sum": 0.47652641492521497, "min": 0.47652641492521497}}, "EndTime": 1606244059.979622, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979602}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47644495568367, "sum": 0.47644495568367, "min": 0.47644495568367}}, "EndTime": 1606244059.979695, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979675}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765287300603924, "sum": 0.4765287300603924, "min": 0.4765287300603924}}, "EndTime": 1606244059.979762, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979743}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47631148969238757, "sum": 0.47631148969238757, "min": 0.47631148969238757}}, "EndTime": 1606244059.979829, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.97981}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4742262368864785, "sum": 0.4742262368864785, "min": 0.4742262368864785}}, "EndTime": 1606244059.979902, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979884}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4744133879453322, "sum": 0.4744133879453322, "min": 0.4744133879453322}}, "EndTime": 1606244059.980005, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.979985}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4781778387087702, "sum": 0.4781778387087702, "min": 0.4781778387087702}}, "EndTime": 1606244059.980075, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980056}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47592824978739984, "sum": 0.47592824978739984, "min": 0.47592824978739984}}, "EndTime": 1606244059.980147, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980128}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892676510176207, "sum": 0.6892676510176207, "min": 0.6892676510176207}}, "EndTime": 1606244059.980217, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980198}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893808108527459, "sum": 0.6893808108527459, "min": 0.6893808108527459}}, "EndTime": 1606244059.980283, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980263}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892384629353536, "sum": 0.6892384629353536, "min": 0.6892384629353536}}, "EndTime": 1606244059.980338, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980321}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896894636815198, "sum": 0.6896894636815198, "min": 0.6896894636815198}}, "EndTime": 1606244059.980411, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980392}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895712701342527, "sum": 0.6895712701342527, "min": 0.6895712701342527}}, "EndTime": 1606244059.980478, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980459}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6900980980746179, "sum": 0.6900980980746179, "min": 0.6900980980746179}}, "EndTime": 1606244059.98055, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980531}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895350249241122, "sum": 0.6895350249241122, "min": 0.6895350249241122}}, "EndTime": 1606244059.98062, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.980601}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902217647752349, "sum": 0.6902217647752349, "min": 0.6902217647752349}}, "EndTime": 1606244059.98069, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244059.98067}
    [0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] #quality_metric: host=algo-1, epoch=10, validation binary_classification_cross_entropy_objective <loss>=0.215107545688[0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=sampled_accuracy, value=0.994630654084[0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] Epoch 10: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] Saving model for epoch: 10[0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] Saved checkpoint to "/tmp/tmpici69F/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] #progress_metric: host=algo-1, completed 73 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 562, "sum": 562.0, "min": 562}, "Total Records Seen": {"count": 1, "max": 552859, "sum": 552859.0, "min": 552859}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1606244059.987773, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1606244057.967863}
    [0m
    [34m[11/24/2020 18:54:19 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=24340.474542 records/second[0m
    [34m[2020-11-24 18:54:21.664] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 25, "duration": 1675, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20919547692123724, "sum": 0.20919547692123724, "min": 0.20919547692123724}}, "EndTime": 1606244061.664572, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.664305}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.196048441050004, "sum": 0.196048441050004, "min": 0.196048441050004}}, "EndTime": 1606244061.664675, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.664651}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21209235428790657, "sum": 0.21209235428790657, "min": 0.21209235428790657}}, "EndTime": 1606244061.664754, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.664734}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19529871944505342, "sum": 0.19529871944505342, "min": 0.19529871944505342}}, "EndTime": 1606244061.664829, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.664809}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04177687586570273, "sum": 0.04177687586570273, "min": 0.04177687586570273}}, "EndTime": 1606244061.664912, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.66489}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03851253587372449, "sum": 0.03851253587372449, "min": 0.03851253587372449}}, "EndTime": 1606244061.664986, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.664966}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.041898337617212414, "sum": 0.041898337617212414, "min": 0.041898337617212414}}, "EndTime": 1606244061.66511, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665085}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.037993528366088866, "sum": 0.037993528366088866, "min": 0.037993528366088866}}, "EndTime": 1606244061.665184, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665163}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21531652022381217, "sum": 0.21531652022381217, "min": 0.21531652022381217}}, "EndTime": 1606244061.665255, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665234}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20460003195003587, "sum": 0.20460003195003587, "min": 0.20460003195003587}}, "EndTime": 1606244061.665318, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665299}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21263508792799346, "sum": 0.21263508792799346, "min": 0.21263508792799346}}, "EndTime": 1606244061.665389, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665369}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19979950387137277, "sum": 0.19979950387137277, "min": 0.19979950387137277}}, "EndTime": 1606244061.665459, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.66544}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13014895256198183, "sum": 0.13014895256198183, "min": 0.13014895256198183}}, "EndTime": 1606244061.665531, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665511}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13018071030597297, "sum": 0.13018071030597297, "min": 0.13018071030597297}}, "EndTime": 1606244061.665601, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665581}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13015182853231624, "sum": 0.13015182853231624, "min": 0.13015182853231624}}, "EndTime": 1606244061.665684, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665654}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13018060006900709, "sum": 0.13018060006900709, "min": 0.13018060006900709}}, "EndTime": 1606244061.665751, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665732}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47575429675043845, "sum": 0.47575429675043845, "min": 0.47575429675043845}}, "EndTime": 1606244061.665825, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665806}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757989501953125, "sum": 0.4757989501953125, "min": 0.4757989501953125}}, "EndTime": 1606244061.665895, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665875}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47575387510961414, "sum": 0.47575387510961414, "min": 0.47575387510961414}}, "EndTime": 1606244061.665963, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.665943}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757956437091438, "sum": 0.4757956437091438, "min": 0.4757956437091438}}, "EndTime": 1606244061.66603, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666011}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47647450349768816, "sum": 0.47647450349768816, "min": 0.47647450349768816}}, "EndTime": 1606244061.666095, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666077}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47804244621432557, "sum": 0.47804244621432557, "min": 0.47804244621432557}}, "EndTime": 1606244061.666165, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666145}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47644853926678093, "sum": 0.47644853926678093, "min": 0.47644853926678093}}, "EndTime": 1606244061.666241, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666221}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780560551857462, "sum": 0.4780560551857462, "min": 0.4780560551857462}}, "EndTime": 1606244061.6663, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666282}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893684517996652, "sum": 0.6893684517996652, "min": 0.6893684517996652}}, "EndTime": 1606244061.666356, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666338}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894299901845504, "sum": 0.6894299901845504, "min": 0.6894299901845504}}, "EndTime": 1606244061.666412, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666395}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893684318698182, "sum": 0.6893684318698182, "min": 0.6893684318698182}}, "EndTime": 1606244061.666469, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.66645}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894329858896684, "sum": 0.6894329858896684, "min": 0.6894329858896684}}, "EndTime": 1606244061.666527, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.66651}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.690394146354831, "sum": 0.690394146354831, "min": 0.690394146354831}}, "EndTime": 1606244061.666588, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666569}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923392545738999, "sum": 0.6923392545738999, "min": 0.6923392545738999}}, "EndTime": 1606244061.666641, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666623}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903942385303731, "sum": 0.6903942385303731, "min": 0.6903942385303731}}, "EndTime": 1606244061.666696, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666678}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6923437462631536, "sum": 0.6923437462631536, "min": 0.6923437462631536}}, "EndTime": 1606244061.666749, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244061.666732}
    [0m
    [34m[11/24/2020 18:54:21 INFO 140513285547840] #quality_metric: host=algo-1, epoch=11, train binary_classification_cross_entropy_objective <loss>=0.209195476921[0m
    [34m[2020-11-24 18:54:21.988] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 35, "duration": 300, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21116130343485917, "sum": 0.21116130343485917, "min": 0.21116130343485917}}, "EndTime": 1606244062.256787, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.256696}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1981591219394633, "sum": 0.1981591219394633, "min": 0.1981591219394633}}, "EndTime": 1606244062.256891, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.256868}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21417870911021905, "sum": 0.21417870911021905, "min": 0.21417870911021905}}, "EndTime": 1606244062.25697, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.256949}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19744550665853236, "sum": 0.19744550665853236, "min": 0.19744550665853236}}, "EndTime": 1606244062.257036, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257023}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04385948266495799, "sum": 0.04385948266495799, "min": 0.04385948266495799}}, "EndTime": 1606244062.2571, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257082}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.040415917744746704, "sum": 0.040415917744746704, "min": 0.040415917744746704}}, "EndTime": 1606244062.257168, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.25715}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04397577266525532, "sum": 0.04397577266525532, "min": 0.04397577266525532}}, "EndTime": 1606244062.257237, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257217}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0399313914128014, "sum": 0.0399313914128014, "min": 0.0399313914128014}}, "EndTime": 1606244062.257308, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257289}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21724280479324395, "sum": 0.21724280479324395, "min": 0.21724280479324395}}, "EndTime": 1606244062.257394, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257372}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2065014912102225, "sum": 0.2065014912102225, "min": 0.2065014912102225}}, "EndTime": 1606244062.257473, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257453}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21447795088346253, "sum": 0.21447795088346253, "min": 0.21447795088346253}}, "EndTime": 1606244062.257554, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257533}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2019262248671408, "sum": 0.2019262248671408, "min": 0.2019262248671408}}, "EndTime": 1606244062.257636, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257615}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13341454252372764, "sum": 0.13341454252372764, "min": 0.13341454252372764}}, "EndTime": 1606244062.257717, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257697}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13393494425710623, "sum": 0.13393494425710623, "min": 0.13393494425710623}}, "EndTime": 1606244062.257782, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257765}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13342751379785808, "sum": 0.13342751379785808, "min": 0.13342751379785808}}, "EndTime": 1606244062.257848, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257829}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13395611207181843, "sum": 0.13395611207181843, "min": 0.13395611207181843}}, "EndTime": 1606244062.257922, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257905}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47652749118376664, "sum": 0.47652749118376664, "min": 0.47652749118376664}}, "EndTime": 1606244062.25799, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.257971}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476412957163847, "sum": 0.476412957163847, "min": 0.476412957163847}}, "EndTime": 1606244062.258063, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258043}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47652981500845953, "sum": 0.47652981500845953, "min": 0.47652981500845953}}, "EndTime": 1606244062.25813, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258112}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47626308660882716, "sum": 0.47626308660882716, "min": 0.47626308660882716}}, "EndTime": 1606244062.2582, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.25818}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4743596929468862, "sum": 0.4743596929468862, "min": 0.4743596929468862}}, "EndTime": 1606244062.258271, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258251}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47380318286323486, "sum": 0.47380318286323486, "min": 0.47380318286323486}}, "EndTime": 1606244062.258338, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258319}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4782289951273567, "sum": 0.4782289951273567, "min": 0.4782289951273567}}, "EndTime": 1606244062.258409, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.25839}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4752103592648865, "sum": 0.4752103592648865, "min": 0.4752103592648865}}, "EndTime": 1606244062.258476, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258457}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892635669453774, "sum": 0.6892635669453774, "min": 0.6892635669453774}}, "EndTime": 1606244062.258543, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258524}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894053326652371, "sum": 0.6894053326652371, "min": 0.6894053326652371}}, "EndTime": 1606244062.25861, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258591}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689234756236351, "sum": 0.689234756236351, "min": 0.689234756236351}}, "EndTime": 1606244062.258679, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.25866}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896488724723143, "sum": 0.6896488724723143, "min": 0.6896488724723143}}, "EndTime": 1606244062.258745, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258726}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895787766342089, "sum": 0.6895787766342089, "min": 0.6895787766342089}}, "EndTime": 1606244062.258811, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258792}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6900435577104582, "sum": 0.6900435577104582, "min": 0.6900435577104582}}, "EndTime": 1606244062.258879, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258859}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895501831630678, "sum": 0.6895501831630678, "min": 0.6895501831630678}}, "EndTime": 1606244062.258946, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.258927}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6900414101587923, "sum": 0.6900414101587923, "min": 0.6900414101587923}}, "EndTime": 1606244062.259022, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244062.259003}
    [0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] #quality_metric: host=algo-1, epoch=11, validation binary_classification_cross_entropy_objective <loss>=0.211161303435[0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=sampled_accuracy, value=0.994793361536[0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] Epoch 11: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] Saving model for epoch: 11[0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] Saved checkpoint to "/tmp/tmpfnL7H1/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] #progress_metric: host=algo-1, completed 80 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 612, "sum": 612.0, "min": 612}, "Total Records Seen": {"count": 1, "max": 602028, "sum": 602028.0, "min": 602028}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1606244062.266244, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1606244059.988111}
    [0m
    [34m[11/24/2020 18:54:22 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=21581.7173929 records/second[0m
    [34m[2020-11-24 18:54:23.685] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 27, "duration": 1418, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20549237683354593, "sum": 0.20549237683354593, "min": 0.20549237683354593}}, "EndTime": 1606244063.685545, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685443}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19220833603216678, "sum": 0.19220833603216678, "min": 0.19220833603216678}}, "EndTime": 1606244063.68565, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685627}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20846520124162946, "sum": 0.20846520124162946, "min": 0.20846520124162946}}, "EndTime": 1606244063.685731, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685711}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19150396043427137, "sum": 0.19150396043427137, "min": 0.19150396043427137}}, "EndTime": 1606244063.685798, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685779}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04033929801473812, "sum": 0.04033929801473812, "min": 0.04033929801473812}}, "EndTime": 1606244063.685868, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685849}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.037146418629860394, "sum": 0.037146418629860394, "min": 0.037146418629860394}}, "EndTime": 1606244063.685931, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685914}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04044730198140047, "sum": 0.04044730198140047, "min": 0.04044730198140047}}, "EndTime": 1606244063.685998, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.685979}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03668876788080955, "sum": 0.03668876788080955, "min": 0.03668876788080955}}, "EndTime": 1606244063.686065, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686046}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21172400369449537, "sum": 0.21172400369449537, "min": 0.21172400369449537}}, "EndTime": 1606244063.686128, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.68611}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20072852029605787, "sum": 0.20072852029605787, "min": 0.20072852029605787}}, "EndTime": 1606244063.686197, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686179}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.209006558788066, "sum": 0.209006558788066, "min": 0.209006558788066}}, "EndTime": 1606244063.686264, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686246}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1962479210678412, "sum": 0.1962479210678412, "min": 0.1962479210678412}}, "EndTime": 1606244063.686328, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686311}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13013122652014908, "sum": 0.13013122652014908, "min": 0.13013122652014908}}, "EndTime": 1606244063.686392, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686375}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13015702477279975, "sum": 0.13015702477279975, "min": 0.13015702477279975}}, "EndTime": 1606244063.686459, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686441}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13013465912488043, "sum": 0.13013465912488043, "min": 0.13013465912488043}}, "EndTime": 1606244063.686526, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686508}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13015472941495934, "sum": 0.13015472941495934, "min": 0.13015472941495934}}, "EndTime": 1606244063.686596, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686578}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757498748156489, "sum": 0.4757498748156489, "min": 0.4757498748156489}}, "EndTime": 1606244063.686663, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686643}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757984544403699, "sum": 0.4757984544403699, "min": 0.4757984544403699}}, "EndTime": 1606244063.686727, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686709}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47574953289421235, "sum": 0.47574953289421235, "min": 0.47574953289421235}}, "EndTime": 1606244063.686792, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686774}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579523514728156, "sum": 0.47579523514728156, "min": 0.47579523514728156}}, "EndTime": 1606244063.686859, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686841}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4764240716428173, "sum": 0.4764240716428173, "min": 0.4764240716428173}}, "EndTime": 1606244063.686927, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686908}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47799048661212534, "sum": 0.47799048661212534, "min": 0.47799048661212534}}, "EndTime": 1606244063.68699, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.686971}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763971208844866, "sum": 0.4763971208844866, "min": 0.4763971208844866}}, "EndTime": 1606244063.687056, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687038}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4780038695043447, "sum": 0.4780038695043447, "min": 0.4780038695043447}}, "EndTime": 1606244063.687123, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687104}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689364125777264, "sum": 0.689364125777264, "min": 0.689364125777264}}, "EndTime": 1606244063.687191, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687172}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.68943044857103, "sum": 0.68943044857103, "min": 0.68943044857103}}, "EndTime": 1606244063.687254, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687236}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893641394790339, "sum": 0.6893641394790339, "min": 0.6893641394790339}}, "EndTime": 1606244063.687328, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687308}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894328775211256, "sum": 0.6894328775211256, "min": 0.6894328775211256}}, "EndTime": 1606244063.687396, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687377}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903433489118304, "sum": 0.6903433489118304, "min": 0.6903433489118304}}, "EndTime": 1606244063.687472, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687451}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6922700419523278, "sum": 0.6922700419523278, "min": 0.6922700419523278}}, "EndTime": 1606244063.687539, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.68752}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6903432629643654, "sum": 0.6903432629643654, "min": 0.6903432629643654}}, "EndTime": 1606244063.687605, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.687587}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6922711430763712, "sum": 0.6922711430763712, "min": 0.6922711430763712}}, "EndTime": 1606244063.687667, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244063.68765}
    [0m
    [34m[11/24/2020 18:54:23 INFO 140513285547840] #quality_metric: host=algo-1, epoch=12, train binary_classification_cross_entropy_objective <loss>=0.205492376834[0m
    [34m[2020-11-24 18:54:23.964] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 38, "duration": 257, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.207834607492702, "sum": 0.207834607492702, "min": 0.207834607492702}}, "EndTime": 1606244064.234877, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.234779}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1946335211274991, "sum": 0.1946335211274991, "min": 0.1946335211274991}}, "EndTime": 1606244064.234987, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.234964}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21091320721830795, "sum": 0.21091320721830795, "min": 0.21091320721830795}}, "EndTime": 1606244064.235061, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235041}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19395429646584797, "sum": 0.19395429646584797, "min": 0.19395429646584797}}, "EndTime": 1606244064.235137, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235117}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.042520823363872993, "sum": 0.042520823363872993, "min": 0.042520823363872993}}, "EndTime": 1606244064.235212, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235194}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.039191052186694705, "sum": 0.039191052186694705, "min": 0.039191052186694705}}, "EndTime": 1606244064.235277, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235257}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04262497639989682, "sum": 0.04262497639989682, "min": 0.04262497639989682}}, "EndTime": 1606244064.235345, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235327}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03876489429958007, "sum": 0.03876489429958007, "min": 0.03876489429958007}}, "EndTime": 1606244064.235414, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235395}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21400178995576233, "sum": 0.21400178995576233, "min": 0.21400178995576233}}, "EndTime": 1606244064.235485, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235465}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20296078833081302, "sum": 0.20296078833081302, "min": 0.20296078833081302}}, "EndTime": 1606244064.235554, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235535}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2112139845530606, "sum": 0.2112139845530606, "min": 0.2112139845530606}}, "EndTime": 1606244064.235622, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235603}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1986391599427362, "sum": 0.1986391599427362, "min": 0.1986391599427362}}, "EndTime": 1606244064.235693, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235673}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1334053024654469, "sum": 0.1334053024654469, "min": 0.1334053024654469}}, "EndTime": 1606244064.23576, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.23574}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13402566574623015, "sum": 0.13402566574623015, "min": 0.13402566574623015}}, "EndTime": 1606244064.23583, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235811}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13342066894243254, "sum": 0.13342066894243254, "min": 0.13342066894243254}}, "EndTime": 1606244064.235901, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235881}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13404097108626808, "sum": 0.13404097108626808, "min": 0.13404097108626808}}, "EndTime": 1606244064.236016, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.235995}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765302892077291, "sum": 0.4765302892077291, "min": 0.4765302892077291}}, "EndTime": 1606244064.236087, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236066}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763908373816875, "sum": 0.4763908373816875, "min": 0.4763908373816875}}, "EndTime": 1606244064.236167, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236147}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765326751003892, "sum": 0.4765326751003892, "min": 0.4765326751003892}}, "EndTime": 1606244064.23624, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236221}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47622262574235275, "sum": 0.47622262574235275, "min": 0.47622262574235275}}, "EndTime": 1606244064.236313, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236294}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47448316847816413, "sum": 0.47448316847816413, "min": 0.47448316847816413}}, "EndTime": 1606244064.236385, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236366}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47378651513131637, "sum": 0.47378651513131637, "min": 0.47378651513131637}}, "EndTime": 1606244064.236433, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236422}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4782376275402383, "sum": 0.4782376275402383, "min": 0.4782376275402383}}, "EndTime": 1606244064.236475, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236459}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4745832518339545, "sum": 0.4745832518339545, "min": 0.4745832518339545}}, "EndTime": 1606244064.236552, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236531}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892616912514078, "sum": 0.6892616912514078, "min": 0.6892616912514078}}, "EndTime": 1606244064.236622, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236603}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894195338161381, "sum": 0.6894195338161381, "min": 0.6894195338161381}}, "EndTime": 1606244064.236702, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236682}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892325168240936, "sum": 0.6892325168240936, "min": 0.6892325168240936}}, "EndTime": 1606244064.236771, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236752}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896176299403322, "sum": 0.6896176299403322, "min": 0.6896176299403322}}, "EndTime": 1606244064.236835, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236815}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895916631855641, "sum": 0.6895916631855641, "min": 0.6895916631855641}}, "EndTime": 1606244064.236915, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236894}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6899108365347826, "sum": 0.6899108365347826, "min": 0.6899108365347826}}, "EndTime": 1606244064.236984, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.236966}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895672878534755, "sum": 0.6895672878534755, "min": 0.6895672878534755}}, "EndTime": 1606244064.237061, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.237041}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6898304523104299, "sum": 0.6898304523104299, "min": 0.6898304523104299}}, "EndTime": 1606244064.237138, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244064.237118}
    [0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] #quality_metric: host=algo-1, epoch=12, validation binary_classification_cross_entropy_objective <loss>=0.207834607493[0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=sampled_accuracy, value=0.994956068988[0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] Epoch 12: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] Saving model for epoch: 12[0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] Saved checkpoint to "/tmp/tmpSLa81D/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] #progress_metric: host=algo-1, completed 86 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 662, "sum": 662.0, "min": 662}, "Total Records Seen": {"count": 1, "max": 651197, "sum": 651197.0, "min": 651197}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1606244064.243848, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1606244062.266561}
    [0m
    [34m[11/24/2020 18:54:24 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=24865.0384033 records/second[0m
    [34m[2020-11-24 18:54:25.888] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 29, "duration": 1643, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2023409542161591, "sum": 0.2023409542161591, "min": 0.2023409542161591}}, "EndTime": 1606244065.888456, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888356}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.18879760711047114, "sum": 0.18879760711047114, "min": 0.18879760711047114}}, "EndTime": 1606244065.888549, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888528}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20536657278878348, "sum": 0.20536657278878348, "min": 0.20536657278878348}}, "EndTime": 1606244065.888625, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888605}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.18812184547891422, "sum": 0.18812184547891422, "min": 0.18812184547891422}}, "EndTime": 1606244065.888672, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888661}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03914547110577019, "sum": 0.03914547110577019, "min": 0.03914547110577019}}, "EndTime": 1606244065.888709, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888699}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.036025762207654057, "sum": 0.036025762207654057, "min": 0.036025762207654057}}, "EndTime": 1606244065.88876, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888743}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03924256978716169, "sum": 0.03924256978716169, "min": 0.03924256978716169}}, "EndTime": 1606244065.888829, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.88881}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.035619620186941967, "sum": 0.035619620186941967, "min": 0.035619620186941967}}, "EndTime": 1606244065.888898, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888879}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2086620284099968, "sum": 0.2086620284099968, "min": 0.2086620284099968}}, "EndTime": 1606244065.888968, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.888948}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1973438412413305, "sum": 0.1973438412413305, "min": 0.1973438412413305}}, "EndTime": 1606244065.889033, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889016}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20592783744967713, "sum": 0.20592783744967713, "min": 0.20592783744967713}}, "EndTime": 1606244065.8891, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889082}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19308644135144293, "sum": 0.19308644135144293, "min": 0.19308644135144293}}, "EndTime": 1606244065.889167, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889149}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13010378203100087, "sum": 0.13010378203100087, "min": 0.13010378203100087}}, "EndTime": 1606244065.889239, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889219}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13013241763990752, "sum": 0.13013241763990752, "min": 0.13013241763990752}}, "EndTime": 1606244065.889318, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.8893}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13010738746487366, "sum": 0.13010738746487366, "min": 0.13010738746487366}}, "EndTime": 1606244065.889384, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889366}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13012899624571508, "sum": 0.13012899624571508, "min": 0.13012899624571508}}, "EndTime": 1606244065.88945, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889432}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47574606510084505, "sum": 0.47574606510084505, "min": 0.47574606510084505}}, "EndTime": 1606244065.889521, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889501}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579914824816644, "sum": 0.47579914824816644, "min": 0.47579914824816644}}, "EndTime": 1606244065.889587, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889569}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47574576988998724, "sum": 0.47574576988998724, "min": 0.47574576988998724}}, "EndTime": 1606244065.889654, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889636}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579509626116073, "sum": 0.47579509626116073, "min": 0.47579509626116073}}, "EndTime": 1606244065.889721, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889703}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763800117336974, "sum": 0.4763800117336974, "min": 0.4763800117336974}}, "EndTime": 1606244065.889788, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889769}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47798619920380264, "sum": 0.47798619920380264, "min": 0.47798619920380264}}, "EndTime": 1606244065.889862, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889843}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763524020448023, "sum": 0.4763524020448023, "min": 0.4763524020448023}}, "EndTime": 1606244065.889934, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889915}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4779932543696189, "sum": 0.4779932543696189, "min": 0.4779932543696189}}, "EndTime": 1606244065.89, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.889981}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893606679488201, "sum": 0.6893606679488201, "min": 0.6893606679488201}}, "EndTime": 1606244065.890075, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890055}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894288342534279, "sum": 0.6894288342534279, "min": 0.6894288342534279}}, "EndTime": 1606244065.890143, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890124}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893607102997449, "sum": 0.6893607102997449, "min": 0.6893607102997449}}, "EndTime": 1606244065.890215, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890195}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894316393793846, "sum": 0.6894316393793846, "min": 0.6894316393793846}}, "EndTime": 1606244065.890291, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890271}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902975750358737, "sum": 0.6902975750358737, "min": 0.6902975750358737}}, "EndTime": 1606244065.890361, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890341}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6922752598353795, "sum": 0.6922752598353795, "min": 0.6922752598353795}}, "EndTime": 1606244065.890435, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890417}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.690297391930405, "sum": 0.690297391930405, "min": 0.690297391930405}}, "EndTime": 1606244065.890509, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890489}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.692271869270169, "sum": 0.692271869270169, "min": 0.692271869270169}}, "EndTime": 1606244065.890585, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244065.890567}
    [0m
    [34m[11/24/2020 18:54:25 INFO 140513285547840] #quality_metric: host=algo-1, epoch=13, train binary_classification_cross_entropy_objective <loss>=0.202340954216[0m
    [34m[2020-11-24 18:54:26.172] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 41, "duration": 259, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2049757054059233, "sum": 0.2049757054059233, "min": 0.2049757054059233}}, "EndTime": 1606244066.438313, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438223}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19143458247533884, "sum": 0.19143458247533884, "min": 0.19143458247533884}}, "EndTime": 1606244066.438417, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438396}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20809550646885264, "sum": 0.20809550646885264, "min": 0.20809550646885264}}, "EndTime": 1606244066.438487, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438468}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19077540731880047, "sum": 0.19077540731880047, "min": 0.19077540731880047}}, "EndTime": 1606244066.438556, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438537}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04140395184041156, "sum": 0.04140395184041156, "min": 0.04140395184041156}}, "EndTime": 1606244066.438628, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438611}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03816698934292506, "sum": 0.03816698934292506, "min": 0.03816698934292506}}, "EndTime": 1606244066.438689, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438672}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04149817888953187, "sum": 0.04149817888953187, "min": 0.04149817888953187}}, "EndTime": 1606244066.438758, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.43874}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03778865990637181, "sum": 0.03778865990637181, "min": 0.03778865990637181}}, "EndTime": 1606244066.438824, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438804}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.21121136807790222, "sum": 0.21121136807790222, "min": 0.21121136807790222}}, "EndTime": 1606244066.43889, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.438871}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19980276426196758, "sum": 0.19980276426196758, "min": 0.19980276426196758}}, "EndTime": 1606244066.438958, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.43894}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20841730684478393, "sum": 0.20841730684478393, "min": 0.20841730684478393}}, "EndTime": 1606244066.439027, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439009}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19565535414075372, "sum": 0.19565535414075372, "min": 0.19565535414075372}}, "EndTime": 1606244066.439102, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439083}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13337286481475583, "sum": 0.13337286481475583, "min": 0.13337286481475583}}, "EndTime": 1606244066.439179, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439161}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13406737759258397, "sum": 0.13406737759258397, "min": 0.13406737759258397}}, "EndTime": 1606244066.439255, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439237}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13338834363476212, "sum": 0.13338834363476212, "min": 0.13338834363476212}}, "EndTime": 1606244066.439326, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439307}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13407328460102472, "sum": 0.13407328460102472, "min": 0.13407328460102472}}, "EndTime": 1606244066.439395, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439376}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.476534353418223, "sum": 0.476534353418223, "min": 0.476534353418223}}, "EndTime": 1606244066.439463, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439444}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47636036200978027, "sum": 0.47636036200978027, "min": 0.47636036200978027}}, "EndTime": 1606244066.439532, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439513}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765368088270064, "sum": 0.4765368088270064, "min": 0.4765368088270064}}, "EndTime": 1606244066.4396, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.43958}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4761745578258153, "sum": 0.4761745578258153, "min": 0.4761745578258153}}, "EndTime": 1606244066.439668, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439649}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47455897084517623, "sum": 0.47455897084517623, "min": 0.47455897084517623}}, "EndTime": 1606244066.439734, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439715}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47416457360240016, "sum": 0.47416457360240016, "min": 0.47416457360240016}}, "EndTime": 1606244066.439806, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439787}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47817578425905505, "sum": 0.47817578425905505, "min": 0.47817578425905505}}, "EndTime": 1606244066.439878, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.439859}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47395621391914583, "sum": 0.47395621391914583, "min": 0.47395621391914583}}, "EndTime": 1606244066.439991, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.43997}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892615782877074, "sum": 0.6892615782877074, "min": 0.6892615782877074}}, "EndTime": 1606244066.440059, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.44004}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894470001329943, "sum": 0.6894470001329943, "min": 0.6894470001329943}}, "EndTime": 1606244066.440129, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.44011}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892310520200671, "sum": 0.6892310520200671, "min": 0.6892310520200671}}, "EndTime": 1606244066.440195, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440176}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895875592515711, "sum": 0.6895875592515711, "min": 0.6895875592515711}}, "EndTime": 1606244066.44026, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440241}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896178608331702, "sum": 0.6896178608331702, "min": 0.6896178608331702}}, "EndTime": 1606244066.440323, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440304}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6898368353801793, "sum": 0.6898368353801793, "min": 0.6898368353801793}}, "EndTime": 1606244066.440376, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440358}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.689596325731262, "sum": 0.689596325731262, "min": 0.689596325731262}}, "EndTime": 1606244066.440434, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440417}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6897344666818358, "sum": 0.6897344666818358, "min": 0.6897344666818358}}, "EndTime": 1606244066.440503, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244066.440485}
    [0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] #quality_metric: host=algo-1, epoch=13, validation binary_classification_cross_entropy_objective <loss>=0.204975705406[0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=sampled_accuracy, value=0.994956068988[0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] Epoch 13: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] Saving model for epoch: 13[0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] Saved checkpoint to "/tmp/tmpkqQV7G/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] #progress_metric: host=algo-1, completed 93 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 712, "sum": 712.0, "min": 712}, "Total Records Seen": {"count": 1, "max": 700366, "sum": 700366.0, "min": 700366}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1606244066.44692, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1606244064.244164}
    [0m
    [34m[11/24/2020 18:54:26 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=22320.2038721 records/second[0m
    [34m[2020-11-24 18:54:28.048] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 31, "duration": 1600, "num_examples": 50, "num_bytes": 2753464}[0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19961075763313138, "sum": 0.19961075763313138, "min": 0.19961075763313138}}, "EndTime": 1606244068.04818, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048078}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.185673128711934, "sum": 0.185673128711934, "min": 0.185673128711934}}, "EndTime": 1606244068.048276, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048253}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20267121513522401, "sum": 0.20267121513522401, "min": 0.20267121513522401}}, "EndTime": 1606244068.048344, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048326}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.18501344455018334, "sum": 0.18501344455018334, "min": 0.18501344455018334}}, "EndTime": 1606244068.048408, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048391}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03814158786073023, "sum": 0.03814158786073023, "min": 0.03814158786073023}}, "EndTime": 1606244068.048477, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048457}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.035091987765565213, "sum": 0.035091987765565213, "min": 0.035091987765565213}}, "EndTime": 1606244068.048545, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048526}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.038229680236505005, "sum": 0.038229680236505005, "min": 0.038229680236505005}}, "EndTime": 1606244068.048612, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048593}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.034729879106794086, "sum": 0.034729879106794086, "min": 0.034729879106794086}}, "EndTime": 1606244068.048681, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048663}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20600641087123325, "sum": 0.20600641087123325, "min": 0.20600641087123325}}, "EndTime": 1606244068.048751, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048732}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1942999192841199, "sum": 0.1942999192841199, "min": 0.1942999192841199}}, "EndTime": 1606244068.04882, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.0488}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20327008087781012, "sum": 0.20327008087781012, "min": 0.20327008087781012}}, "EndTime": 1606244068.048897, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.04888}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1901961825623804, "sum": 0.1901961825623804, "min": 0.1901961825623804}}, "EndTime": 1606244068.048965, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.048946}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13006945084552377, "sum": 0.13006945084552377, "min": 0.13006945084552377}}, "EndTime": 1606244068.049035, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049016}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13009586692342953, "sum": 0.13009586692342953, "min": 0.13009586692342953}}, "EndTime": 1606244068.049103, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049084}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.130072826463349, "sum": 0.130072826463349, "min": 0.130072826463349}}, "EndTime": 1606244068.04917, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049152}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13009217927893812, "sum": 0.13009217927893812, "min": 0.13009217927893812}}, "EndTime": 1606244068.049236, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049217}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757426720444037, "sum": 0.4757426720444037, "min": 0.4757426720444037}}, "EndTime": 1606244068.049298, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.04928}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757985951949139, "sum": 0.4757985951949139, "min": 0.4757985951949139}}, "EndTime": 1606244068.049363, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049345}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4757424615353954, "sum": 0.4757424615353954, "min": 0.4757424615353954}}, "EndTime": 1606244068.049429, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049411}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47579482596261163, "sum": 0.47579482596261163, "min": 0.47579482596261163}}, "EndTime": 1606244068.049491, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049473}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763433732013313, "sum": 0.4763433732013313, "min": 0.4763433732013313}}, "EndTime": 1606244068.049558, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.04954}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47795757994359855, "sum": 0.47795757994359855, "min": 0.47795757994359855}}, "EndTime": 1606244068.049633, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049613}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763146754673549, "sum": 0.4763146754673549, "min": 0.4763146754673549}}, "EndTime": 1606244068.049709, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.04969}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47795201017418687, "sum": 0.47795201017418687, "min": 0.47795201017418687}}, "EndTime": 1606244068.049785, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049766}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893577880859375, "sum": 0.6893577880859375, "min": 0.6893577880859375}}, "EndTime": 1606244068.049853, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049835}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894294744997609, "sum": 0.6894294744997609, "min": 0.6894294744997609}}, "EndTime": 1606244068.049922, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049902}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6893578665597099, "sum": 0.6893578665597099, "min": 0.6893578665597099}}, "EndTime": 1606244068.04999, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.049971}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894317166075414, "sum": 0.6894317166075414, "min": 0.6894317166075414}}, "EndTime": 1606244068.050056, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.050038}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902487294722577, "sum": 0.6902487294722577, "min": 0.6902487294722577}}, "EndTime": 1606244068.050115, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.050103}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6922138646962691, "sum": 0.6922138646962691, "min": 0.6922138646962691}}, "EndTime": 1606244068.050151, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.050142}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6902485949457908, "sum": 0.6902485949457908, "min": 0.6902485949457908}}, "EndTime": 1606244068.050206, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.050188}
    [0m
    [34m#metrics {"Metrics": {"train_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6922082669005102, "sum": 0.6922082669005102, "min": 0.6922082669005102}}, "EndTime": 1606244068.050264, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.050247}
    [0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] #quality_metric: host=algo-1, epoch=14, train binary_classification_cross_entropy_objective <loss>=0.199610757633[0m
    [34m[2020-11-24 18:54:28.334] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 44, "duration": 261, "num_examples": 7, "num_bytes": 344176}[0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20247774468622726, "sum": 0.20247774468622726, "min": 0.20247774468622726}}, "EndTime": 1606244068.600411, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600321}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.18847397798984478, "sum": 0.18847397798984478, "min": 0.18847397798984478}}, "EndTime": 1606244068.60052, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600499}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20562270499967109, "sum": 0.20562270499967109, "min": 0.20562270499967109}}, "EndTime": 1606244068.600592, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600573}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1878232996276796, "sum": 0.1878232996276796, "min": 0.1878232996276796}}, "EndTime": 1606244068.600664, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600646}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.040462391600715894, "sum": 0.040462391600715894, "min": 0.040462391600715894}}, "EndTime": 1606244068.600731, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600715}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.03732493755447333, "sum": 0.03732493755447333, "min": 0.03732493755447333}}, "EndTime": 1606244068.600805, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600785}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.04054764319505825, "sum": 0.04054764319505825, "min": 0.04054764319505825}}, "EndTime": 1606244068.600884, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600864}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.0369879260266444, "sum": 0.0369879260266444, "min": 0.0369879260266444}}, "EndTime": 1606244068.600963, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.600944}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.2087688712937181, "sum": 0.2087688712937181, "min": 0.2087688712937181}}, "EndTime": 1606244068.601039, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.60102}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19693712345598732, "sum": 0.19693712345598732, "min": 0.19693712345598732}}, "EndTime": 1606244068.601117, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601099}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.20598133495565427, "sum": 0.20598133495565427, "min": 0.20598133495565427}}, "EndTime": 1606244068.60118, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601164}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.19290446831665634, "sum": 0.19290446831665634, "min": 0.19290446831665634}}, "EndTime": 1606244068.601282, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601259}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13332415837400557, "sum": 0.13332415837400557, "min": 0.13332415837400557}}, "EndTime": 1606244068.601352, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601335}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13404940969192203, "sum": 0.13404940969192203, "min": 0.13404940969192203}}, "EndTime": 1606244068.601416, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601381}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.13333999610203234, "sum": 0.13333999610203234, "min": 0.13333999610203234}}, "EndTime": 1606244068.601481, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601463}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.1340498587536649, "sum": 0.1340498587536649, "min": 0.1340498587536649}}, "EndTime": 1606244068.601557, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601537}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47653957581698525, "sum": 0.47653957581698525, "min": 0.47653957581698525}}, "EndTime": 1606244068.601636, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601616}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4763404928121123, "sum": 0.4763404928121123, "min": 0.4763404928121123}}, "EndTime": 1606244068.601714, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601695}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4765419703991608, "sum": 0.4765419703991608, "min": 0.4765419703991608}}, "EndTime": 1606244068.601793, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601773}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4761363624401446, "sum": 0.4761363624401446, "min": 0.4761363624401446}}, "EndTime": 1606244068.601857, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601841}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.4745502813297648, "sum": 0.4745502813297648, "min": 0.4745502813297648}}, "EndTime": 1606244068.601894, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601885}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47461095525044555, "sum": 0.47461095525044555, "min": 0.47461095525044555}}, "EndTime": 1606244068.601929, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601919}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47800636353571874, "sum": 0.47800636353571874, "min": 0.47800636353571874}}, "EndTime": 1606244068.601963, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601954}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.47376723309661833, "sum": 0.47376723309661833, "min": 0.47376723309661833}}, "EndTime": 1606244068.602003, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.601989}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892629984027976, "sum": 0.6892629984027976, "min": 0.6892629984027976}}, "EndTime": 1606244068.602065, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602047}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6894680411738834, "sum": 0.6894680411738834, "min": 0.6894680411738834}}, "EndTime": 1606244068.602141, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602122}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6892300601739508, "sum": 0.6892300601739508, "min": 0.6892300601739508}}, "EndTime": 1606244068.60221, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602192}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895711745495832, "sum": 0.6895711745495832, "min": 0.6895711745495832}}, "EndTime": 1606244068.602278, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.60226}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896505421006327, "sum": 0.6896505421006327, "min": 0.6896505421006327}}, "EndTime": 1606244068.602345, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602326}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6895820997531741, "sum": 0.6895820997531741, "min": 0.6895820997531741}}, "EndTime": 1606244068.602411, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602393}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6896282448040869, "sum": 0.6896282448040869, "min": 0.6896282448040869}}, "EndTime": 1606244068.602454, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602443}
    [0m
    [34m#metrics {"Metrics": {"validation_binary_classification_cross_entropy_objective": {"count": 1, "max": 0.6898461170239981, "sum": 0.6898461170239981, "min": 0.6898461170239981}}, "EndTime": 1606244068.602488, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244068.602478}
    [0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] #quality_metric: host=algo-1, epoch=14, validation binary_classification_cross_entropy_objective <loss>=0.202477744686[0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=sampled_accuracy, value=0.994956068988[0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] Epoch 14: Loss improved. Updating best model[0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] Saving model for epoch: 14[0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] Saved checkpoint to "/tmp/tmpeHhL0w/mx-mod-0000.params"[0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] #progress_metric: host=algo-1, completed 100 % of epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Batches Since Last Reset": {"count": 1, "max": 50, "sum": 50.0, "min": 50}, "Number of Records Since Last Reset": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Total Batches Seen": {"count": 1, "max": 762, "sum": 762.0, "min": 762}, "Total Records Seen": {"count": 1, "max": 749535, "sum": 749535.0, "min": 749535}, "Max Records Seen Between Resets": {"count": 1, "max": 49169, "sum": 49169.0, "min": 49169}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1606244068.608946, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1606244066.447221}
    [0m
    [34m[11/24/2020 18:54:28 INFO 140513285547840] #throughput_metric: host=algo-1, train throughput=22743.7031111 records/second[0m
    [34m[11/24/2020 18:54:28 WARNING 140513285547840] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [34m[11/24/2020 18:54:28 WARNING 140513285547840] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
    [34m[2020-11-24 18:54:28.892] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 47, "duration": 265, "num_examples": 7, "num_bytes": 344176}[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=sampled_accuracy, value=0.994956068988[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] Epoch 14: Loss improved. Updating best model[0m
    [34m[2020-11-24 18:54:29.194] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/validation", "epoch": 49, "duration": 27, "num_examples": 7, "num_bytes": 344176}[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('binary_classification_cross_entropy_objective', 0.03732493755447333)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('binary_classification_accuracy', 0.9947933615359583)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('binary_f_1.000', 0.9948320413436692)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('precision', 0.9897172236503856)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('recall', 1.0)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #validation_score (algo-1) : ('roc_auc_score', 0.9989881693648817)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation binary_classification_cross_entropy_objective <loss>=0.0373249375545[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation binary_classification_accuracy <score>=0.994793361536[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation binary_f_1.000 <score>=0.994832041344[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation precision <score>=0.98971722365[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation recall <score>=1.0[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, validation roc_auc_score <score>=0.998988169365[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] Best model found for hyperparameters: {"lr_scheduler_step": 100, "wd": 0.0001, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 1e-05}[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] Saved checkpoint to "/tmp/tmpT3faGg/mx-mod-0000.params"[0m
    [34m[2020-11-24 18:54:29.204] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/test", "epoch": 0, "duration": 34025, "num_examples": 1, "num_bytes": 56000}[0m
    [34m[2020-11-24 18:54:29.233] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/test", "epoch": 1, "duration": 28, "num_examples": 7, "num_bytes": 344232}[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Batches Since Last Reset": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Number of Records Since Last Reset": {"count": 1, "max": 6147, "sum": 6147.0, "min": 6147}, "Total Batches Seen": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Total Records Seen": {"count": 1, "max": 6147, "sum": 6147.0, "min": 6147}, "Max Records Seen Between Resets": {"count": 1, "max": 6147, "sum": 6147.0, "min": 6147}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1606244069.236004, "Dimensions": {"Host": "algo-1", "Meta": "test_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1606244069.204112}
    [0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('binary_classification_cross_entropy_objective', 0.03407410239521732)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('binary_classification_accuracy', 0.9956076134699854)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('binary_f_1.000', 0.9956416464891041)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('precision', 0.991321118611379)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('recall', 1.0)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #test_score (algo-1) : ('roc_auc_score', 0.9992747948083756)[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test binary_classification_cross_entropy_objective <loss>=0.0340741023952[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test binary_classification_accuracy <score>=0.99560761347[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test binary_f_1.000 <score>=0.995641646489[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test precision <score>=0.991321118611[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test recall <score>=1.0[0m
    [34m[11/24/2020 18:54:29 INFO 140513285547840] #quality_metric: host=algo-1, test roc_auc_score <score>=0.999274794808[0m
    [34m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 34290.366888046265, "sum": 34290.366888046265, "min": 34290.366888046265}, "finalize.time": {"count": 1, "max": 592.2269821166992, "sum": 592.2269821166992, "min": 592.2269821166992}, "initialize.time": {"count": 1, "max": 209.92398262023926, "sum": 209.92398262023926, "min": 209.92398262023926}, "check_early_stopping.time": {"count": 16, "max": 1.2650489807128906, "sum": 16.918420791625977, "min": 0.9469985961914062}, "setuptime": {"count": 1, "max": 28.358936309814453, "sum": 28.358936309814453, "min": 28.358936309814453}, "update.time": {"count": 15, "max": 2748.759984970093, "sum": 33169.42238807678, "min": 1974.5841026306152}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1606244069.2408, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1606244035.169554}
    [0m
    
    2020-11-24 18:54:39 Uploading - Uploading generated training model
    2020-11-24 18:54:39 Completed - Training job completed
    Training seconds: 109
    Billable seconds: 109



```python
# deploy a model hosting endpoint
binary_predictor = binary_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    Parameter image will be renamed to image_uri in SageMaker Python SDK v2.


    ---------------!


```python
import seaborn as sns
def evaluate_metrics(predictor, test_features, test_labels):
    """
    Evaluate a model on a test set using the given prediction endpoint. Display classification metrics.
    """
    # split the test dataset into 100 batches and evaluate using prediction endpoint
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(test_features, 100)]

    # parse protobuf responses to extract predicted labels
    extract_label = lambda x: x.label['predicted_label'].float32_tensor.values
    test_preds = np.concatenate([np.array([extract_label(x) for x in batch]) for batch in prediction_batches])
    test_preds = test_preds.reshape((-1,))
    
    # calculate accuracy
    accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]
    
    # calculate recall for each class
    recall_per_class, classes = [], []
    for target_label in np.unique(test_labels):
        recall_numerator = np.logical_and(test_preds == target_label, test_labels == target_label).sum()
        recall_denominator = (test_labels == target_label).sum()
        recall_per_class.append(recall_numerator / recall_denominator)
        classes.append(label_map[target_label])
    recall = pd.DataFrame({'recall': recall_per_class, 'class_label': classes})
    recall.sort_values('class_label', ascending=False, inplace=True)

    # calculate confusion matrix
    label_mapper = np.vectorize(lambda x: label_map[x])
    confusion_matrix = pd.crosstab(label_mapper(test_labels), label_mapper(test_preds), 
                                   rownames=['Actuals'], colnames=['Predictions'], normalize='index')

    # display results
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap="YlGnBu").set_title('Confusion Matrix')  
    ax = recall.plot(kind='barh', x='class_label', y='recall', color='steelblue', title='Recall', legend=False)
    ax.set_ylabel('')
    print('Accuracy: {:.3f}'.format(accuracy))
```
# evaluate metrics of the model trained with default hyperparameters
evaluate_metrics(binary_predictor, test_features, test_labels)

```python
print(test_features[12])
result = binary_predictor.predict(test_features[12])
print(result)
```

    [991.94183  24.07     39.01388]
    [label {
      key: "predicted_label"
      value {
        float32_tensor {
          values: 1.0
        }
      }
    }
    label {
      key: "score"
      value {
        float32_tensor {
          values: 0.9999949932098389
        }
      }
    }
    ]

