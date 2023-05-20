```python
import pandas as pd
import numpy as np
import pickle
import time
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns
```

#### Data cleaning functions


```python
def column_rename(df):
    df.rename(columns={'srch_ci':'check_in', 'srch_co':'check_out', 'srch_adults_cnt':'adult_count', \
                         'srch_children_cnt':'child_count', 'srch_rm_cnt':'room_count',\
                         'srch_destination_id':'destination_id', 'srch_destination_type_id':'destination_type_id', \
                        'cnt':'similar_events'}, inplace=True)
```


```python
def feature_engine(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['check_in'] = pd.to_datetime(df['check_in'], errors='coerce')
    df['check_out'] = pd.to_datetime(df['check_out'], errors='coerce')
    df['stay_duration'] = (df['check_out'] - df['check_in']).astype('timedelta64[D]')
    df['plan_time'] = (df['check_in'] - df['date_time']).astype('timedelta64[D]')
#     df['day_of_week'] = df['date_time'].dt.day_name()
```


```python
def fillna(df):
    df['orig_destination_distance'] = df['orig_destination_distance'].astype(np.float64)
    df['orig_destination_distance'].fillna((df['orig_destination_distance'].mean()), inplace=True)
    df['stay_duration'].fillna((df['stay_duration'].mean()), inplace=True)
    df['plan_time'].fillna((df['plan_time'].mean()), inplace=True)
```

### Train 2m dataset


```python
train = pd.read_csv('train_2m.csv')
```


```python
train.head()
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.1932</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.6259</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1999999 entries, 0 to 1999998
    Data columns (total 24 columns):
    date_time                    object
    site_name                    int64
    posa_continent               int64
    user_location_country        int64
    user_location_region         int64
    user_location_city           int64
    orig_destination_distance    float64
    user_id                      int64
    is_mobile                    int64
    is_package                   int64
    channel                      int64
    srch_ci                      object
    srch_co                      object
    srch_adults_cnt              int64
    srch_children_cnt            int64
    srch_rm_cnt                  int64
    srch_destination_id          int64
    srch_destination_type_id     int64
    is_booking                   int64
    cnt                          int64
    hotel_continent              int64
    hotel_country                int64
    hotel_market                 int64
    hotel_cluster                int64
    dtypes: float64(1), int64(20), object(3)
    memory usage: 366.2+ MB



```python
#clean up
column_rename(train)
feature_engine(train)
fillna(train)
```


```python
most_common_clusters = list(train.hotel_cluster.value_counts().head(10).index)
most_common_clusters
```




    [91, 41, 48, 64, 5, 65, 98, 59, 70, 42]




```python
train = train.loc[train['hotel_cluster'].isin(most_common_clusters)]
```


```python
# remove datetime columns
train.drop(columns=['date_time', 'check_in','check_out'], inplace=True)
```


```python
# with open('train_cleaned.pickle', 'wb') as to_write:
#     pickle.dump(train, to_write)
```


```python
train.to_csv('train_cleaned.csv')
```

# **Model 1: Keras**


```python
import pandas as pd
import numpy as np
import pickle
import time
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
```


```python
with open('train_cleaned.pickle','rb') as read_file:
    train = pickle.load(read_file)
```


```python
train.head()
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
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>...</th>
      <th>destination_id</th>
      <th>destination_type_id</th>
      <th>is_booking</th>
      <th>similar_events</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
      <th>stay_duration</th>
      <th>plan_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>70</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>98</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>30</td>
      <td>4</td>
      <td>195</td>
      <td>991</td>
      <td>47725</td>
      <td>2014.665587</td>
      <td>1048</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>8803</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>151</td>
      <td>1197</td>
      <td>5</td>
      <td>2.0</td>
      <td>215.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# xx = pd.Series([1,1,2,2,5])
# np_utils.to_categorical(xx, num_classes=5)
```

### Train, test, split


```python
train['hotel_cluster'].value_counts()
```




    91    55754
    41    41797
    48    39949
    64    39626
    5     32914
    65    32783
    98    32019
    59    30603
    70    30271
    42    29740
    Name: hotel_cluster, dtype: int64




```python
train['hotel_cluster'].replace([91, 41, 48, 64, 5, 65, 98, 59, 70, 42], \
                               ['ninety-one','forty-one','forty-eight','sixty-four', 'five',\
                               'sixty-five','ninety-eight','fifty-nine','seventy','forty-two'], inplace=True)
train['hotel_cluster'].value_counts()
```




    ninety-one      55754
    forty-one       41797
    forty-eight     39949
    sixty-four      39626
    five            32914
    sixty-five      32783
    ninety-eight    32019
    fifty-nine      30603
    seventy         30271
    forty-two       29740
    Name: hotel_cluster, dtype: int64




```python
X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']
```


```python
from sklearn.preprocessing import LabelEncoder

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
# scales data between 0 and 1
X_train = keras.utils.normalize(X_train, axis=1)  
X_test = keras.utils.normalize(X_test, axis=1)
```


```python
# y_test = np_utils.to_categorical(y_test, num_classes=10)
# y_train = np_utils.to_categorical(y_train, num_classes=10)
```


```python
X_train.shape
```




    (292364, 22)



### Keras

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


```python
def c_model(optimizer):

  model = Sequential()
  model.add(Dense(2048, activation='relu', input_dim=22))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model

model = KerasClassifier(build_fn=c_model, epochs=30, batch_size=128)
parameters = {'optimizer':['RMSprop','Adagrad','Adam','Adamax']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)
```

    /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)
    WARNING: Logging before flag parsing goes to stderr.
    W0918 14:57:21.156636 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0918 14:57:21.339053 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0918 14:57:21.390038 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0918 14:57:21.530771 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    W0918 14:57:21.584022 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3295: The name tf.log is deprecated. Please use tf.math.log instead.
    
    W0918 14:57:21.751450 4559062464 deprecation.py:323] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0918 14:57:21.906458 4559062464 deprecation_wrapper.py:119] From /Users/kelseyheng/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    


    Epoch 1/30
    109824/194909 [===============>..............] - ETA: 33s - loss: 2.2566 - acc: 0.1673


```python
print(clf.best_score_, clf.best_params_)
means = clf.cv.results_['mean_tset_score']
parameters = clf.cv_results['params']
for mean, parameter in zip(means, parameters):
  print(mean, parameter)
```

# Model 1 Keras: Hyperparameter Tuning


```python
import pandas as pd
import numpy as np
import pickle
import time
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import zipfile
import io
from google.colab import files
```


```python
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
```

    Using TensorFlow backend.



```python
uploaded = files.upload()
```



<input type="file" id="files-d08a7386-20b9-4376-973c-56a55108eea6" name="files[]" multiple disabled />
<output id="result-d08a7386-20b9-4376-973c-56a55108eea6">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving train_cleaned.csv.zip to train_cleaned.csv.zip



```python
! unzip train_cleaned.csv.zip
```

    Archive:  train_cleaned.csv.zip
      inflating: train_cleaned.csv       



```python
train = pd.read_csv('train_cleaned.csv')
```


```python
train.head()
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
      <th>Unnamed: 0</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>adult_count</th>
      <th>child_count</th>
      <th>room_count</th>
      <th>destination_id</th>
      <th>destination_type_id</th>
      <th>is_booking</th>
      <th>similar_events</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
      <th>stay_duration</th>
      <th>plan_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>70</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>98</td>
      <td>1.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43</td>
      <td>30</td>
      <td>4</td>
      <td>195</td>
      <td>991</td>
      <td>47725</td>
      <td>2014.665587</td>
      <td>1048</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8803</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>151</td>
      <td>1197</td>
      <td>5</td>
      <td>2.0</td>
      <td>215.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.drop(columns=['Unnamed: 0'], inplace=True)
```

#### Train, test, split


```python
train['hotel_cluster'].value_counts()
```




    91    55754
    41    41797
    48    39949
    64    39626
    5     32914
    65    32783
    98    32019
    59    30603
    70    30271
    42    29740
    Name: hotel_cluster, dtype: int64




```python
train['hotel_cluster'].replace([91, 41, 48, 64, 5, 65, 98, 59, 70, 42], \
                               ['ninety-one','forty-one','forty-eight','sixty-four', 'five',\
                               'sixty-five','ninety-eight','fifty-nine','seventy','forty-two'], inplace=True)
train['hotel_cluster'].value_counts()
```




    ninety-one      55754
    forty-one       41797
    forty-eight     39949
    sixty-four      39626
    five            32914
    sixty-five      32783
    ninety-eight    32019
    fifty-nine      30603
    seventy         30271
    forty-two       29740
    Name: hotel_cluster, dtype: int64




```python
X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']
```


```python
from sklearn.preprocessing import LabelEncoder

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_y)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
# scales data between 0 and 1
X_train = keras.utils.normalize(X_train, axis=1)  
X_test = keras.utils.normalize(X_test, axis=1)
```


```python
X_train.shape
```




    (292364, 22)



### KERAS
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/


```python
def c_model():

  model = Sequential()
  model.add(Dense(2048, activation='relu', input_dim=22))
  # model.add(Dense(1024, activation='relu'))
  # model.add(Dropout(0.25))
  model.add(Dense(512, activation='relu'))
  # model.add(Dropout(0.25))
  model.add(Dense(256, activation='relu'))
  # model.add(Dropout(0.25))
  # model.add(Dense(128, activation='relu'))
  # model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  # model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))
  
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model
```


```python
model = KerasClassifier(build_fn=c_model, epochs=100, batch_size=128)
```


```python
model.fit(X_train, y_train)
```

    Epoch 1/100
    292364/292364 [==============================] - 23s 78us/step - loss: 2.1926 - acc: 0.2014
    Epoch 2/100
    292364/292364 [==============================] - 21s 71us/step - loss: 2.0742 - acc: 0.2481
    Epoch 3/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.9533 - acc: 0.2886
    Epoch 4/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.9276 - acc: 0.2977
    Epoch 5/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.9110 - acc: 0.3024
    Epoch 6/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8951 - acc: 0.3089
    Epoch 7/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8841 - acc: 0.3119
    Epoch 8/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8691 - acc: 0.3163
    Epoch 9/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8626 - acc: 0.3179
    Epoch 10/100
    292364/292364 [==============================] - 20s 68us/step - loss: 1.8487 - acc: 0.3220
    Epoch 11/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8382 - acc: 0.3248
    Epoch 12/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8264 - acc: 0.3280
    Epoch 13/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8202 - acc: 0.3299
    Epoch 14/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.8056 - acc: 0.3343
    Epoch 15/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7946 - acc: 0.3378
    Epoch 16/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7885 - acc: 0.3391
    Epoch 17/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7785 - acc: 0.3429
    Epoch 18/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7744 - acc: 0.3438
    Epoch 19/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7632 - acc: 0.3463
    Epoch 20/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7627 - acc: 0.3468
    Epoch 21/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7531 - acc: 0.3500
    Epoch 22/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7463 - acc: 0.3522
    Epoch 23/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7442 - acc: 0.3523
    Epoch 24/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7380 - acc: 0.3543
    Epoch 25/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7346 - acc: 0.3546
    Epoch 26/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7320 - acc: 0.3559
    Epoch 27/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7298 - acc: 0.3565
    Epoch 28/100
    292364/292364 [==============================] - 21s 70us/step - loss: 1.7260 - acc: 0.3572
    Epoch 29/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7214 - acc: 0.3593
    Epoch 30/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7236 - acc: 0.3584
    Epoch 31/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7151 - acc: 0.3601
    Epoch 32/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7152 - acc: 0.3613
    Epoch 33/100
    292364/292364 [==============================] - 21s 70us/step - loss: 1.7106 - acc: 0.3623
    Epoch 34/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7143 - acc: 0.3618
    Epoch 35/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7058 - acc: 0.3630
    Epoch 36/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7049 - acc: 0.3635
    Epoch 37/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.7009 - acc: 0.3648
    Epoch 38/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.7007 - acc: 0.3651
    Epoch 39/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6948 - acc: 0.3668
    Epoch 40/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6951 - acc: 0.3668
    Epoch 41/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6943 - acc: 0.3671
    Epoch 42/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6926 - acc: 0.3678
    Epoch 43/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6940 - acc: 0.3675
    Epoch 44/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6882 - acc: 0.3683
    Epoch 45/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6878 - acc: 0.3696
    Epoch 46/100
    292364/292364 [==============================] - 20s 68us/step - loss: 1.6829 - acc: 0.3704
    Epoch 47/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6821 - acc: 0.3708
    Epoch 48/100
    292364/292364 [==============================] - 21s 71us/step - loss: 1.6794 - acc: 0.3717
    Epoch 49/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6775 - acc: 0.3722
    Epoch 50/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6784 - acc: 0.3721
    Epoch 51/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6745 - acc: 0.3735
    Epoch 52/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6739 - acc: 0.3728
    Epoch 53/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6693 - acc: 0.3750
    Epoch 54/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6716 - acc: 0.3749
    Epoch 55/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6661 - acc: 0.3762
    Epoch 56/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6670 - acc: 0.3752
    Epoch 57/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6639 - acc: 0.3766
    Epoch 58/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6656 - acc: 0.3761
    Epoch 59/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6606 - acc: 0.3778
    Epoch 60/100
    292364/292364 [==============================] - 21s 71us/step - loss: 1.6597 - acc: 0.3777
    Epoch 61/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6584 - acc: 0.3780
    Epoch 62/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6585 - acc: 0.3782
    Epoch 63/100
    292364/292364 [==============================] - 21s 70us/step - loss: 1.6583 - acc: 0.3784
    Epoch 64/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6543 - acc: 0.3797
    Epoch 65/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6570 - acc: 0.3788
    Epoch 66/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6534 - acc: 0.3796
    Epoch 67/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6483 - acc: 0.3812
    Epoch 68/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6525 - acc: 0.3805
    Epoch 69/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6508 - acc: 0.3807
    Epoch 70/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6495 - acc: 0.3816
    Epoch 71/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6458 - acc: 0.3830
    Epoch 72/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6471 - acc: 0.3817
    Epoch 73/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6468 - acc: 0.3821
    Epoch 74/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6437 - acc: 0.3834
    Epoch 75/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6390 - acc: 0.3846
    Epoch 76/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6415 - acc: 0.3838
    Epoch 77/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6399 - acc: 0.3854
    Epoch 78/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6429 - acc: 0.3838
    Epoch 79/100
    292364/292364 [==============================] - 21s 71us/step - loss: 1.6387 - acc: 0.3844
    Epoch 80/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6346 - acc: 0.3857
    Epoch 81/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6372 - acc: 0.3853
    Epoch 82/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6331 - acc: 0.3877
    Epoch 83/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6351 - acc: 0.3860
    Epoch 84/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6354 - acc: 0.3854
    Epoch 85/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6317 - acc: 0.3878
    Epoch 86/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6336 - acc: 0.3866
    Epoch 87/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6298 - acc: 0.3885
    Epoch 88/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6292 - acc: 0.3888
    Epoch 89/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6238 - acc: 0.3904
    Epoch 90/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6263 - acc: 0.3892
    Epoch 91/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6279 - acc: 0.3887
    Epoch 92/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6218 - acc: 0.3912
    Epoch 93/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6244 - acc: 0.3891
    Epoch 94/100
    292364/292364 [==============================] - 21s 70us/step - loss: 1.6238 - acc: 0.3890
    Epoch 95/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6218 - acc: 0.3906
    Epoch 96/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6274 - acc: 0.3887
    Epoch 97/100
    292364/292364 [==============================] - 20s 70us/step - loss: 1.6192 - acc: 0.3913
    Epoch 98/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6201 - acc: 0.3906
    Epoch 99/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6189 - acc: 0.3911
    Epoch 100/100
    292364/292364 [==============================] - 20s 69us/step - loss: 1.6168 - acc: 0.3909





    <keras.callbacks.History at 0x7eff0e7bbe80>



## GridSearchCV
#### Tune batch size and epochs


```python
model = KerasClassifier(build_fn=c_model)

batch_sizes = [128] #batch_size = 128
epochs = [30] #, 50, 100, 200]
parameters = {'epochs': epochs}
clf = GridSearchCV(model, parameters, cv=2)
clf.fit(X_train, y_train)
```


```python
print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parameter in zip(means, parameters):
  print(mean, parameter)
```

#### Tune optimizer


```python
def c_model(optimizer):

  model = Sequential()
  model.add(Dense(2048, activation='relu', input_dim=22))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model

model = KerasClassifier(build_fn=c_model, epochs=10, batch_size=128)
parameters = {'optimizer':['RMSprop','Adagrad','Adam','Adamax']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)
```


```python
print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parameter in zip(means, parameters):
  print(mean, parameter)
```

    0.31756303785630646 {'optimizer': 'Adam'}
    0.28852731526514486 {'optimizer': 'RMSprop'}
    0.2824013900458307 {'optimizer': 'Adagrad'}
    0.31756303785630646 {'optimizer': 'Adam'}
    0.30522225718563567 {'optimizer': 'Adamax'}


#### Tune activation function


```python
def c_model(activation):

  model = Sequential()
  model.add(Dense(2048, activation=activation, input_dim=22))
  model.add(Dense(512, activation=activation))
  model.add(Dense(256, activation=activation))
  model.add(Dense(64, activation=activation))
  model.add(Dense(10, activation='softmax'))
  
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model

model = KerasClassifier(build_fn=c_model, epochs=10 , batch_size=128)
parameters = {'activation':['softmax', 'softplus','relu','tanh','sigmoid','hard_sigmoid','linear']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)
```


```python
print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parameter in zip(means, parameters):
  print(mean, parameter)
```

    0.32606955712986296 {'activation': 'tanh'}
    0.15242300693543726 {'activation': 'softmax'}
    0.169446306658109 {'activation': 'softplus'}
    0.3091009837076906 {'activation': 'relu'}
    0.32606955712986296 {'activation': 'tanh'}
    0.15242300693543726 {'activation': 'sigmoid'}
    0.15242300693543726 {'activation': 'hard_sigmoid'}
    0.2097932714021173 {'activation': 'linear'}



```python
from keras.models import load_model
```


```python
model.save('model.h5')
files.download('keras_model.h5')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-41-adb3cdf34e00> in <module>()
    ----> 1 model.save('model.h5')
          2 files.download('keras_model.h5')


    AttributeError: 'KerasClassifier' object has no attribute 'save'



```python
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
```




    ['accuracy',
     'adjusted_mutual_info_score',
     'adjusted_rand_score',
     'average_precision',
     'balanced_accuracy',
     'brier_score_loss',
     'completeness_score',
     'explained_variance',
     'f1',
     'f1_macro',
     'f1_micro',
     'f1_samples',
     'f1_weighted',
     'fowlkes_mallows_score',
     'homogeneity_score',
     'jaccard',
     'jaccard_macro',
     'jaccard_micro',
     'jaccard_samples',
     'jaccard_weighted',
     'max_error',
     'mutual_info_score',
     'neg_log_loss',
     'neg_mean_absolute_error',
     'neg_mean_squared_error',
     'neg_mean_squared_log_error',
     'neg_median_absolute_error',
     'normalized_mutual_info_score',
     'precision',
     'precision_macro',
     'precision_micro',
     'precision_samples',
     'precision_weighted',
     'r2',
     'recall',
     'recall_macro',
     'recall_micro',
     'recall_samples',
     'recall_weighted',
     'roc_auc',
     'v_measure_score']



# **Model 2: Decision Tree**






```python
import pandas as pd
import numpy as np
import pickle
import time
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import zipfile
import io
from google.colab import files
```


```python
uploaded = files.upload()
```



<input type="file" id="files-5f8d2dd1-5357-429f-a768-2bbabc4d3759" name="files[]" multiple disabled />
<output id="result-5f8d2dd1-5357-429f-a768-2bbabc4d3759">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving train_round5.csv.zip to train_round5.csv.zip



```python
! unzip train_round5.csv.zip
```

    Archive:  train_round5.csv.zip
      inflating: train_round5.csv        



```python
train = pd.read_csv('train_round5.csv')
```


```python
train.head()
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
      <th>Unnamed: 0</th>
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>check_in</th>
      <th>check_out</th>
      <th>adult_count</th>
      <th>child_count</th>
      <th>room_count</th>
      <th>destination_id</th>
      <th>destination_type_id</th>
      <th>is_booking</th>
      <th>similar_events</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
      <th>stay_duration</th>
      <th>plan_time</th>
      <th>solo_travel</th>
      <th>short_trip</th>
      <th>weekend_trip</th>
      <th>booking_rate</th>
      <th>price_compare</th>
      <th>biz_trip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2014-07-16 09:42:23</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2014-08-01 00:00:00</td>
      <td>2014-08-02 00:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2014-07-16 09:45:48</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2014-08-01 00:00:00</td>
      <td>2014-08-02 00:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>41</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2014-07-16 09:55:24</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2014-08-01 00:00:00</td>
      <td>2014-08-02 00:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>70</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2014-07-16 10:00:06</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>189</td>
      <td>10067</td>
      <td>2014.665587</td>
      <td>501</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2014-08-01 00:00:00</td>
      <td>2014-08-02 00:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8267</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>675</td>
      <td>98</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2014-11-22 20:55:38</td>
      <td>30</td>
      <td>4</td>
      <td>195</td>
      <td>991</td>
      <td>47725</td>
      <td>2014.665587</td>
      <td>1048</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>2015-06-26 00:00:00</td>
      <td>2015-06-28 00:00:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>8803</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>151</td>
      <td>1197</td>
      <td>5</td>
      <td>2.0</td>
      <td>215.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove datetime columns for model
train.drop(columns=['Unnamed: 0','date_time', 'check_in','check_out',], inplace=True)
```


```python
#drop according to feature importance 
train.drop(columns=['is_booking','short_trip','room_count','solo_travel','is_mobile','weekend_trip','posa_continent','similar_events',\
                    'is_package','adult_count','child_count','channel','user_location_country','price_compare','biz_trip'], inplace=True)
```


```python
train['hotel_cluster'].replace([91, 41, 48, 64, 5, 65, 98, 59, 70, 42], \
                               ['ninety-one','forty-one','forty-eight','sixty-four', 'five',\
                               'sixty-five','ninety-eight','fifty-nine','seventy','forty-two'], inplace=True)
train['hotel_cluster'].value_counts()
```




    ninety-one      55752
    forty-one       41797
    forty-eight     39948
    sixty-four      39625
    five            32914
    sixty-five      32783
    ninety-eight    32017
    fifty-nine      30602
    seventy         30269
    forty-two       29740
    Name: hotel_cluster, dtype: int64




```python
train.shape
```




    (365447, 14)



### Decision Tree Classifier


```python
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, confusion_matrix, make_scorer, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, log_loss
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
```


```python
X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']

# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(y)

# y = LabelEncoder().fit_transform(y)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
```


```python
tree = DecisionTreeClassifier(max_depth=12, criterion='entropy', max_features=12)

tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_val)
```


```python
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("The score is")
print("Training: {:6.2f}%".format(100*tree.score(X_train, y_train)))
print("Validate: {:6.2f}%".format(100*tree.score(X_val, y_val)))
print("Testing: {:6.2f}%".format(100*tree.score(X_test, y_test)))
```

    Accuracy: 0.46273087973731014
    The score is
    Training:  49.21%
    Validate:  46.27%
    Testing:  46.20%



```python
tree_pred = tree.predict(X_val)
print('Tree Results:')
print(confusion_matrix(y_val, tree_pred))
print(classification_report(y_val, tree_pred))
```

    Tree Results:
    [[2278  480  173   59  178  217 1796    7   17  801]
     [ 535 2239  240  522  105  254 1049  141  195 1312]
     [ 253  310 1145  515  285  190 5008  106   22  189]
     [ 122  365  236 2030   57  784 2643 1754  108  409]
     [ 305  187  476  126  562   81 3965    5   29  184]
     [ 235  110   55  579   24 2378  958 1615   44  318]
     [ 296  159  649  463  366  260 8514  266    9   77]
     [  22  109   79  985    8 1140  984 2543    5  197]
     [   2   14    4   10    2    0   10    0 6468   37]
     [ 441  838   50  189   19  469   88   36  253 5664]]
                  precision    recall  f1-score   support
    
      fifty-nine       0.51      0.38      0.43      6006
            five       0.47      0.34      0.39      6592
     forty-eight       0.37      0.14      0.21      8023
       forty-one       0.37      0.24      0.29      8508
       forty-two       0.35      0.09      0.15      5920
    ninety-eight       0.41      0.38      0.39      6316
      ninety-one       0.34      0.77      0.47     11059
         seventy       0.39      0.42      0.41      6072
      sixty-five       0.90      0.99      0.94      6547
      sixty-four       0.62      0.70      0.66      8047
    
        accuracy                           0.46     73090
       macro avg       0.47      0.45      0.43     73090
    weighted avg       0.46      0.46      0.44     73090
    



```python
# feat_imp = pd.DataFrame({'importance':tree.feature_importances_})    
# feat_imp['feature'] = X_train.columns
# feat_imp.sort_values(by='importance', ascending=False, inplace=True)

# title = 'Feature Importance'
# figsize = (10,10)
# # featax = featfig.add_subplot(1, 1, 1)

# feat_imp.sort_values(by='importance', inplace=True)
# feat_imp = feat_imp.set_index('feature', drop=True)
# feat_imp.plot.barh(title=title, figsize=figsize)
# plt.xlabel('Feature Importance Score')

# plt.show()

feature_importance = abs(tree.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure(1,figsize=(17,10))
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center', color='#eca02c')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=14)

featax.set_xlabel('Relative Feature Importance', fontsize=20, weight='bold')
featax.tick_params(axis="x", labelsize=16)


featax.spines['right'].set_visible(False)
featax.spines['top'].set_visible(False)
featax.spines['bottom'].set_visible(True)
featax.spines['left'].set_visible(True)

featax.patch.set_visible(False)

plt.savefig('feature_importance.jpg', transparent=True);
# plt.show()

```


    
![png](output_83_0.png)
    



```python
files.download('all_feature_importance.jpg')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-19-22cd69b2ec8f> in <module>()
    ----> 1 files.download('all_feature_importance.jpg')
    

    /usr/local/lib/python3.6/dist-packages/google/colab/files.py in download(filename)
        142       raise OSError(msg)
        143     else:
    --> 144       raise FileNotFoundError(msg)  # pylint: disable=undefined-variable
        145 
        146   started = _threading.Event()


    FileNotFoundError: Cannot find file: all_feature_importance.jpg



```python
est = DecisionTreeClassifier()

rf_p_dist={
           'criterion':['gini','entropy'],
           'max_depth':[3,6,9,12],
           'max_features':[3,6,9,12],
          }

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=5)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
hypertuning_rscv(est, rf_p_dist, 40, X, y)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_search.py:266: UserWarning: The total space of parameters 32 is smaller than n_iter=40. Running 32 iterations. For exhaustive searches, use GridSearchCV.
      % (grid_size, self.n_iter, grid_size), UserWarning)





    ({'criterion': 'entropy', 'max_depth': 12, 'max_features': 12},
     0.4421407208158775)



### XGBoost


```python
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
```


```python
X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']

# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
```


```python
model = XGBClassifier(max_depth=12)

model = model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
```


```python
print("Validate accuracy: %.2f" % (accuracy_score(y_val, y_pred_val) * 100))
print("Test accuracy: %.2f" % (accuracy_score(y_test, y_pred_test) * 100))
```

    Validate accuracy: 56.16
    Test accuracy: 56.33



```python
feature_importance = abs(model.feature_importances_)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure(1,figsize=(17,10))
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center', color='#eca02c')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=14)

featax.set_xlabel('Relative Feature Importance', fontsize=20, weight='bold')
featax.tick_params(axis="x", labelsize=16)


featax.spines['right'].set_visible(False)
featax.spines['top'].set_visible(False)
featax.spines['bottom'].set_visible(True)
featax.spines['left'].set_visible(True)

featax.patch.set_visible(False)

plt.savefig('feature_importance.jpg', transparent=True);
# plt.show()
```


    
![png](output_92_0.png)
    


### GridSearchCV


```python
from sklearn.model_selection import GridSearchCV
```


```python
est = xgb.XGBClassifier()

rf_p_dist = {
     "eta"    : [0.01, 0.1, 0.3] ,
     "max_depth"        : [ 6, 9, 12],
     "min_child_weight" : [ 3, 5, 7 ],
     "gamma"            : [ 0.1, 0.3, 0.4 ],
     }

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=5)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score
```


```python
hypertuning_rscv(est, rf_p_dist, 1, X, y)
```

### K-fold evaluation


```python
# k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
```


```python
X = train.drop(['hotel_cluster'], axis=1)
y = train['hotel_cluster']
```


```python
# CV model
model = XGBClassifier(max_depth=12)
kfold = KFold(n_splits=5, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

    [0.46795731 0.4705158  0.46681443 0.46996128 0.46602088]
    Accuracy: 46.83% (0.17%)



```python
results = [0.46795731, 0.4705158,  0.46681443, 0.46996128, 0.46602088]
# Accuracy: 46.83% (0.17%)
```


```python
x = ['1','2','3','4','5']
plt.figure(figsize=(8,5))
sns.despine()
ax = sns.barplot(x=x, y=results,color='#eca02c')

# plt.title('Accuracy Score', fontsize=20, fontweight='bold')

# plt.xlabel('k-fold', fontsize=20, fontweight='bold')
plt.xticks(fontsize=16, weight='bold')

plt.ylabel('Accuracy', fontsize=18, fontweight='bold')
plt.yticks(fontsize=18)
plt.ylim([0,1.0])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.savefig('kfold_score.jpg', transparent=True)
```


    
![png](output_102_0.png)
    



```python
files.download('kfold_score.jpg')
```


```python

```
