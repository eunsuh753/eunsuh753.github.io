---
layout: post
title:  "입원 기간 예측"
toc: true
toc_sticky: true
---


**이화여자대학교  
윤하영 이선민 이진솔 조은서**


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from lightgbm import plot_importance

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
from sklearn.preprocessing import (StandardScaler, 
                                   PolynomialFeatures) 
import warnings
warnings.filterwarnings("ignore")
```


```python
from sklearn.model_selection import KFold
```


```python
df = pd.read_csv("train_data.csv")
```

## 1. 데이터분석

### 상관관계


```python
# Stay와 VIsitors with Patient 간의 높은 상관관계 확인
plt.figure(figsize=(12,12))
sns.heatmap(data=df.corr().abs(), annot=True, fmt='.2f', linewidths=.5)
```




    <AxesSubplot:>



![output_7_1](https://user-images.githubusercontent.com/62747570/139969438-fee504bd-2fe3-4bd0-849d-fa882986ac76.png)




## 2. 전처리


```python
df2 = df.copy()
```


```python
# 범주형 변수 원핫인코딩

illness_mapping = {'Extreme':2, 'Minor':0, 'Moderate':1}
df2['Severity of Illness'] = df['Severity of Illness'].map(illness_mapping)

type_mapping = {'Emergency': 2, 'Trauma':0, 'Urgent':1 }
df2['Type of Admission'] = df['Type of Admission'].map(type_mapping)

department_mapping = {'radiotherapy': 0, 'anesthesia':1, 'gynecology':2, 'TB & Chest disease': 3, 'surgery':4}
df2['Department'] = df['Department'].map(department_mapping)


# Stay 중간값 처리 - 'Stay More than 100 Days'는 105로 처리
stay_mapping = {'0-10':5, '11-20':15, '21-30':25, '31-40':35, '41-50':45, '51-60':55, '61-70':65, '71-80':75, '81-90':85, '91-100':95, 'More than 100 Days':105}
df2['Stay'] = df['Stay'].map(stay_mapping)

# Age 처리
age_mapping = {'0-10':5, '11-20':15, '21-30':25, '31-40':35, '41-50':45, '51-60':55, '61-70':65, '71-80':75, '81-90':85, '91-100':95}
df2['Age'] = df['Age'].map(age_mapping)


df2 = df2.loc[:, ['Department','Type of Admission','Visitors with Patient', 'Severity of Illness','Available Extra Rooms in Hospital','Age','Admission_Deposit','Stay']]

```


```python
df2.head(5)
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
      <th>Department</th>
      <th>Type of Admission</th>
      <th>Visitors with Patient</th>
      <th>Severity of Illness</th>
      <th>Available Extra Rooms in Hospital</th>
      <th>Age</th>
      <th>Admission_Deposit</th>
      <th>Stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>55</td>
      <td>4911.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>5954.0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>4745.0</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>7272.0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>5558.0</td>
      <td>45</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 결측치 처리
print("Data set의 결측치: ")
print(df2.isna().sum().sum())
```

    Data set의 결측치: 
    0



```python
# 31만개의 data set에 비해 결측치가 매우 적다고 판단하여 결측치 모두 삭제
df3 = df2.dropna()
```

## 3. 모델링


```python
X_data = df3.drop(['Stay'], axis=1)
y_data = df3['Stay']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.1, random_state=777)
```


```python
# 정답값에 +- 15일 오차를 허용한 새로운 정확도 산출 함수
def new_accuracy(y_test, pred):
    accuracy_count=0
    for label,y in zip(y_test, pred):
        if ((y >= label-15) & (y <= label+15)):
            accuracy_count += 1
    return accuracy_count/len(pred)
```

### 1) 방문자수 예측 


```python
V_X_data =  df3.drop(['Stay', 'Visitors with Patient'], axis=1)
V_y_data = df3['Visitors with Patient']
V_X_train, V_X_test, V_y_train, V_y_test = train_test_split(V_X_data, V_y_data, test_size = 0.1, random_state=777)
```


```python
V_X_train.head()
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
      <th>Department</th>
      <th>Type of Admission</th>
      <th>Severity of Illness</th>
      <th>Available Extra Rooms in Hospital</th>
      <th>Age</th>
      <th>Admission_Deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>314293</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>15</td>
      <td>7731.0</td>
    </tr>
    <tr>
      <th>86058</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>45</td>
      <td>4401.0</td>
    </tr>
    <tr>
      <th>170972</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>65</td>
      <td>4355.0</td>
    </tr>
    <tr>
      <th>161774</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>55</td>
      <td>6155.0</td>
    </tr>
    <tr>
      <th>14123</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>35</td>
      <td>7404.0</td>
    </tr>
  </tbody>
</table>

</div>




```python
from sklearn.preprocessing import StandardScaler
V_sc = StandardScaler()
scaled_V_X_train = V_sc.fit_transform(V_X_train)
scaled_V_X_test = V_sc.transform(V_X_test)
```


```python
hyper_params = {
    'objective': 'regression',
    'metric':{'rmse', 'mean_absolute_error'},
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 6,
    "num_leaves": 64,
    "num_iterations":2000
}

V_lgb_train = lgb.Dataset(scaled_V_X_train, V_y_train)
V_lgb_eval = lgb.Dataset(scaled_V_X_test, V_y_test, reference = V_lgb_train)
V_lgbm = lgb.train(hyper_params, V_lgb_train, num_boost_round=1000, valid_sets = V_lgb_eval, early_stopping_rounds =200)
```


```python
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import metrics

mae = metrics.mean_absolute_error(V_y_test, V_lgbm_pred)
V_lgbm_pred = V_lgbm.predict(scaled_V_X_test)
print("Lightgbm Regressor Model")
print("RMSE: ", np.round(np.sqrt(mean_squared_error(V_y_test, V_lgbm_pred)),3))
print("r2 score: ",np.round(r2_score(V_y_test, V_lgbm_pred),3))
print("MAE: ", mae)
```

    Lightgbm Regressor Model
    RMSE:  1.677
    r2 score:  0.125



```python
print("new accuracy: ", np.round(new_accuracy(V_y_test, V_lgbm_pred),5))
```

    new accuracy:  0.99934



```python
scaled_V_X_test
```




    array([[-1.15220374, -0.97504549,  0.13720764, -0.16890599, -0.5963106 ,
             0.20119   ],
           [ 0.35277395, -0.97504549,  1.64521831, -1.02423671,  0.99172938,
             1.15758432],
           [ 0.35277395,  1.21220623,  1.64521831, -0.16890599, -0.06696394,
            -1.47871341],
           ...,
           [ 0.35277395,  1.21220623, -1.37080303, -0.16890599,  0.99172938,
            -0.75244284],
           [ 0.35277395, -0.97504549,  0.13720764,  1.54175546, -0.5963106 ,
            -0.54901344],
           [ 0.35277395,  1.21220623,  0.13720764,  0.68642473,  1.52107604,
             0.78018136]])



#### 방문자수 잘 예측하는지 샘플 테스트 


```python
ex = [[1, 1, 1, 1, 13, 6000]]
ex_scaled = V_sc.transform(ex)
ex_scaled
```




    array([[-1.15220374,  0.11858037,  0.13720764, -1.87956743, -1.76087325,
             1.03055601]])




```python
V_lgbm.predict(ex_scaled)
```




    array([2.41062702])



#### 학습된 방문자수 시각화 


```python
width = 12
height = 10
plt.figure(figsize=(width, height))

ax1 = sns.distplot(V_y_train, hist=False, color="r", label="Actual Value")
sns.distplot(V_lgbm_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')


plt.show()
plt.close()
```

![output_29_0](https://user-images.githubusercontent.com/62747570/139969547-0327d9fc-4bec-40c7-a6c9-46364a926f78.png)




### 2) 입원기간 예측


```python
X_train.head()
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
      <th>Department</th>
      <th>Type of Admission</th>
      <th>Visitors with Patient</th>
      <th>Severity of Illness</th>
      <th>Available Extra Rooms in Hospital</th>
      <th>Age</th>
      <th>Admission_Deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>314293</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>15</td>
      <td>7731.0</td>
    </tr>
    <tr>
      <th>86058</th>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>45</td>
      <td>4401.0</td>
    </tr>
    <tr>
      <th>170972</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>65</td>
      <td>4355.0</td>
    </tr>
    <tr>
      <th>161774</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>55</td>
      <td>6155.0</td>
    </tr>
    <tr>
      <th>14123</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>35</td>
      <td>7404.0</td>
    </tr>
  </tbody>
</table>

</div>




```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)
```


```python
hyper_params = {
    'objective': 'regression',
    'metric':{'rmse', 'mean_absolute_error'},
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 6,
    "num_leaves": 64,
    "num_iterations":2000
}

lgb_train = lgb.Dataset(scaled_X_train, y_train)
lgb_eval = lgb.Dataset(scaled_X_test, y_test, reference = lgb_train)
lgbm = lgb.train(hyper_params, lgb_train, num_boost_round=1000, valid_sets = lgb_eval, early_stopping_rounds =200)
```


```python
lgbm_pred = lgbm.predict(scaled_X_test)
mae = metrics.mean_absolute_error(y_test, lgbm_pred)
print("Lightgbm Regressor Model")
print("RMSE: ", np.round(np.sqrt(mean_squared_error(y_test, lgbm_pred)),3))
print("r2 score: ",np.round(r2_score(y_test, lgbm_pred),3))
print("MAE: ", mae)
```

    Lightgbm Regressor Model
    RMSE:  16.796
    r2 score:  0.401



```python
print("new accuracy: ", np.round(new_accuracy(y_test, lgbm_pred),5))
```

    new accuracy:  0.70987



```python
ex2 = [[1, 1, 2.4937412 , 1, 1, 13, 6000]]
ex2_scaled = sc.transform(ex2)
lgbm.predict(ex2_scaled)
```




    array([27.2710498])



#### 변수 중요도


```python
# Feature Importance
#column_4: Admission_deposit column_2: visitors with patients
from lightgbm import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

f, ax = plt.subplots(figsize=(6,6))
plot_importance(lgbm, max_num_features=15, ax=ax)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='Feature importance', ylabel='Features'>



![output_38_1](https://user-images.githubusercontent.com/62747570/139969558-0a8e8ae0-f4eb-4f8d-afdf-151c06b1c1be.png)




#### 학습된 입원기간 시각화


```python
# DISTRIBUTION PLOT Training

plt.figure(figsize=(width, height))

ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")
sns.distplot(lgbm_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')


plt.show()
plt.close()


```

![output_40_0](https://user-images.githubusercontent.com/62747570/139969585-ce07b9b5-8275-492f-82be-1a865e0e5606.png)





```python

```

## 4. 성능 개선하기


```python
kf = KFold(n_splits = 10)
df = pd.DataFrame(columns = ['Department', 'Type of Admission','Visitors with Patient', 'Severity of Illness', 
                                 'Available Extra Rooms in Hospital', 'Age','Admission_Deposit','mae','Stay'])
for train_index, test_index in kf.split(X_data):
    X_train, X_test = X_data.loc[train_index,:],X_data.loc[test_index,:]
    y_train, y_test = y_data.loc[train_index], y_data.loc[test_index]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train)
    lgbm = lgb.train(hyper_params, lgb_train, num_boost_round=1000, valid_sets = lgb_eval, early_stopping_rounds =200)
    lgb_val_pred = lgbm.predict(X_test)
   
    
     #for 돌면서 mae 추가하기 
    difference = y_test - lgb_val_pred 
    abs_diff = abs(difference)
    
    mae_test = pd.concat([X_test, abs_diff], axis = 1)
    mae_test.rename(columns = {'Stay':'mae'}, inplace = True)
    
    mae_test = pd.concat([mae_test, y_test], axis=1)
    small_mae = mae_test[mae_test['mae'] < 40]
    df = pd.concat([df, small_mae], axis=0)

small_mae_final = df
```


```python
small_mae_final
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
      <th>Department</th>
      <th>Type of Admission</th>
      <th>Visitors with Patient</th>
      <th>Severity of Illness</th>
      <th>Available Extra Rooms in Hospital</th>
      <th>Age</th>
      <th>Admission_Deposit</th>
      <th>mae</th>
      <th>Stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>55</td>
      <td>4911.0</td>
      <td>18.944358</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>5954.0</td>
      <td>17.165030</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>4745.0</td>
      <td>10.136774</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>7272.0</td>
      <td>15.137026</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>55</td>
      <td>5558.0</td>
      <td>17.325648</td>
      <td>45</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>318433</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>45</td>
      <td>4144.0</td>
      <td>8.603907</td>
      <td>15</td>
    </tr>
    <tr>
      <th>318434</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>85</td>
      <td>6699.0</td>
      <td>13.513199</td>
      <td>35</td>
    </tr>
    <tr>
      <th>318435</th>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>75</td>
      <td>4235.0</td>
      <td>6.391259</td>
      <td>15</td>
    </tr>
    <tr>
      <th>318436</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>15</td>
      <td>3761.0</td>
      <td>29.594069</td>
      <td>15</td>
    </tr>
    <tr>
      <th>318437</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>15</td>
      <td>4752.0</td>
      <td>13.301626</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>308266 rows × 9 columns</p>

</div>




```python
#약 3%삭제
small_mae_final.shape
```




    (308266, 9)




```python
data_final = small_mae_final.astype('float')
data_final.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 308263 entries, 0 to 318437
    Data columns (total 9 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   Department                         308263 non-null  float64
     1   Type of Admission                  308263 non-null  float64
     2   Visitors with Patient              308263 non-null  float64
     3   Severity of Illness                308263 non-null  float64
     4   Available Extra Rooms in Hospital  308263 non-null  float64
     5   Age                                308263 non-null  float64
     6   Admission_Deposit                  308263 non-null  float64
     7   mae                                308263 non-null  float64
     8   Stay                               308263 non-null  float64
    dtypes: float64(9)
    memory usage: 23.5 MB


### 1) 최종 방문자수 예측 모델 


```python
V_X_data =  data_final.drop(['Stay', 'mae','Visitors with Patient'], axis=1)
V_y_data = data_final['Visitors with Patient']
V_X_train, V_X_test, V_y_train, V_y_test = train_test_split(V_X_data, V_y_data, test_size = 0.1, random_state=777)

from sklearn.preprocessing import StandardScaler
V_sc = StandardScaler()
scaled_V_X_train = V_sc.fit_transform(V_X_train)
scaled_V_X_test = V_sc.transform(V_X_test)

hyper_params = {
    'objective': 'regression',
    'metric':{'rmse', 'mean_absolute_error'},
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 6,
    "num_leaves": 64,
    "num_iterations":2000
}

V_lgb_train = lgb.Dataset(scaled_V_X_train, V_y_train)
V_lgb_eval = lgb.Dataset(scaled_V_X_test, V_y_test, reference = V_lgb_train)
V_lgbm = lgb.train(hyper_params, V_lgb_train, num_boost_round=1000, valid_sets = V_lgb_eval, early_stopping_rounds =200)
```


```python
V_lgbm_pred_f = V_lgbm.predict(scaled_V_X_test)
print("Lightgbm Regressor Model")
print("RMSE: ", np.round(np.sqrt(mean_squared_error(V_y_test, V_lgbm_pred_f)),3))
print("r2 score: ",np.round(r2_score(V_y_test, V_lgbm_pred_f),3))

mae = metrics.mean_absolute_error(V_y_test, V_lgbm_pred_f)
print("MAE: ", mae)

print("new accuracy: ", np.round(new_accuracy(V_y_test, V_lgbm_pred_f),5))
```

    Lightgbm Regressor Model
    RMSE:  1.633
    r2 score:  0.128
    MAE:  1.0837676468166921
    new accuracy:  0.99929


### 2) 최종 입원기간 예측 모델


```python
X_data =  data_final.drop(['Stay', 'mae'], axis=1)
y_data = data_final['Stay']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.1, random_state=767)
```


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)
```


```python
hyper_params = {
    'objective': 'regression',
    'metric':{'rmse', 'mean_absolute_error'},
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 6,
    "num_leaves": 64,
    "num_iterations":2000
}

lgb_train = lgb.Dataset(scaled_X_train, y_train)
lgb_eval = lgb.Dataset(scaled_X_test, y_test, reference = lgb_train)
lgbm = lgb.train(hyper_params, lgb_train, num_boost_round=1000, valid_sets = lgb_eval, early_stopping_rounds =200)
```


```python
lgbm_pred_f = lgbm.predict(scaled_X_test)
print("Lightgbm Regressor Model")
print("RMSE: ", np.round(np.sqrt(mean_squared_error(y_test, lgbm_pred_f)),3))
print("r2 score: ",np.round(r2_score(y_test, lgbm_pred_f),3))

mae = metrics.mean_absolute_error(y_test, lgbm_pred_f)
print("MAE: ", mae)

print("new accuracy: ", np.round(new_accuracy(y_test, lgbm_pred_f),5))
```

    Lightgbm Regressor Model
    RMSE:  13.778
    r2 score:  0.487
    MAE:  10.61113587349634
    new accuracy:  0.75359


### 최종 모델 변환하기 


```python
import pickle
pickle.dump(V_lgbm,open("V_code_final.sav", "wb"))
pickle.dump(V_sc, open("V_scaler_final.sav", "wb"))
```


```python
pickle.dump(lgbm,open("code_final.sav", "wb"))
pickle.dump(sc, open("scaler_final.sav", "wb"))
```


```python

```
