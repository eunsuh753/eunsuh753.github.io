# 1. 데이터 파악 및 EDA


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
cd /content/drive/MyDrive/신용카드 연체 예측
```

    /content/drive/MyDrive/신용카드 연체 예측
    


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import seaborn as sns
```


```python
df = pd.read_csv("data/train.csv")
df.head()
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>202500.0</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>Municipal apartment</td>
      <td>-13899</td>
      <td>-4709</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>-6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>1</td>
      <td>247500.0</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>-11380</td>
      <td>-1540</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Laborers</td>
      <td>3.0</td>
      <td>-5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>450000.0</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-19087</td>
      <td>-4434</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>-22.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-15088</td>
      <td>-2092</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>2.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>157500.0</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-15037</td>
      <td>-2105</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>-26.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 26457 entries, 0 to 26456
    Data columns (total 20 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   index          26457 non-null  int64  
     1   gender         26457 non-null  object 
     2   car            26457 non-null  object 
     3   reality        26457 non-null  object 
     4   child_num      26457 non-null  int64  
     5   income_total   26457 non-null  float64
     6   income_type    26457 non-null  object 
     7   edu_type       26457 non-null  object 
     8   family_type    26457 non-null  object 
     9   house_type     26457 non-null  object 
     10  DAYS_BIRTH     26457 non-null  int64  
     11  DAYS_EMPLOYED  26457 non-null  int64  
     12  FLAG_MOBIL     26457 non-null  int64  
     13  work_phone     26457 non-null  int64  
     14  phone          26457 non-null  int64  
     15  email          26457 non-null  int64  
     16  occyp_type     18286 non-null  object 
     17  family_size    26457 non-null  float64
     18  begin_month    26457 non-null  float64
     19  credit         26457 non-null  float64
    dtypes: float64(4), int64(8), object(8)
    memory usage: 4.0+ MB
    


```python
df.describe()
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
      <th>index</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>2.645700e+04</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>26457.0</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
      <td>26457.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13228.000000</td>
      <td>0.428658</td>
      <td>1.873065e+05</td>
      <td>-15958.053899</td>
      <td>59068.750728</td>
      <td>1.0</td>
      <td>0.224742</td>
      <td>0.294251</td>
      <td>0.091280</td>
      <td>2.196848</td>
      <td>-26.123294</td>
      <td>1.519560</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7637.622372</td>
      <td>0.747326</td>
      <td>1.018784e+05</td>
      <td>4201.589022</td>
      <td>137475.427503</td>
      <td>0.0</td>
      <td>0.417420</td>
      <td>0.455714</td>
      <td>0.288013</td>
      <td>0.916717</td>
      <td>16.559550</td>
      <td>0.702283</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.700000e+04</td>
      <td>-25152.000000</td>
      <td>-15713.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-60.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6614.000000</td>
      <td>0.000000</td>
      <td>1.215000e+05</td>
      <td>-19431.000000</td>
      <td>-3153.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>-39.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13228.000000</td>
      <td>0.000000</td>
      <td>1.575000e+05</td>
      <td>-15547.000000</td>
      <td>-1539.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>-24.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>19842.000000</td>
      <td>1.000000</td>
      <td>2.250000e+05</td>
      <td>-12446.000000</td>
      <td>-407.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>-12.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>26456.000000</td>
      <td>19.000000</td>
      <td>1.575000e+06</td>
      <td>-7705.000000</td>
      <td>365243.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (26457, 20)



### 중복 데이터 확인


- 'begin_month' 변수를 제외한 중복 데이터 10,000개 이상  
=> 동일 인물이 다른 시기에 카드 개설했다고 가정  
=> 데이터의 절반이 중복 데이터?  
=> 한 행으로 줄이는 대신 '개설 카드 개수' 열 추가


```python
df.columns
```




    Index(['index', 'gender', 'car', 'reality', 'child_num', 'income_total',
           'income_type', 'edu_type', 'family_type', 'house_type', 'DAYS_BIRTH',
           'DAYS_EMPLOYED', 'FLAG_MOBIL', 'work_phone', 'phone', 'email',
           'occyp_type', 'family_size', 'begin_month', 'credit'],
          dtype='object')




```python
cols = ['gender', 'car', 'reality', 'child_num', 'income_total',
       'income_type', 'edu_type', 'family_type', 'house_type', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'FLAG_MOBIL', 'work_phone', 'phone', 'email',
       'occyp_type', 'family_size', 'credit']
df[df.duplicated(cols)]
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>319</th>
      <td>319</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>1</td>
      <td>225000.0</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>-12640</td>
      <td>-399</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Accountants</td>
      <td>3.0</td>
      <td>-21.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>330</th>
      <td>330</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>State servant</td>
      <td>Secondary / secondary special</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>-19363</td>
      <td>-12332</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Medicine staff</td>
      <td>1.0</td>
      <td>-18.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>419</th>
      <td>419</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-18820</td>
      <td>-3185</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Low-skill Laborers</td>
      <td>2.0</td>
      <td>-7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>421</td>
      <td>F</td>
      <td>Y</td>
      <td>N</td>
      <td>0</td>
      <td>180000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10351</td>
      <td>-1322</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>-16.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>457</th>
      <td>457</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>1</td>
      <td>112500.0</td>
      <td>Commercial associate</td>
      <td>Higher education</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>-10551</td>
      <td>-3000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Core staff</td>
      <td>3.0</td>
      <td>-17.0</td>
      <td>2.0</td>
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
      <td>...</td>
      <td>...</td>
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
      <th>26445</th>
      <td>26445</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.0</td>
      <td>Commercial associate</td>
      <td>Incomplete higher</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-13687</td>
      <td>-4144</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>-5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26446</th>
      <td>26446</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>-16300</td>
      <td>-9698</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Managers</td>
      <td>2.0</td>
      <td>-41.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>26447</th>
      <td>26447</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>99000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-14226</td>
      <td>-1026</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>4.0</td>
      <td>-43.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>26449</th>
      <td>26449</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>90000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10498</td>
      <td>-2418</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26451</th>
      <td>26451</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-12831</td>
      <td>-803</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Accountants</td>
      <td>2.0</td>
      <td>-44.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>14358 rows × 20 columns</p>
</div>




```python
df['ALL'] = \
    df['child_num'].astype(str) + '_' + df['income_total'].astype(str) + '_' +\
    df['DAYS_BIRTH'].astype(str) + '_' + df['DAYS_EMPLOYED'].astype(str) + '_' +\
    df['work_phone'].astype(str) + '_' + df['phone'].astype(str) + '_' +\
    df['email'].astype(str) + '_' + df['family_size'].astype(str) + '_' +\
    df['gender'].astype(str) + '_' + df['car'].astype(str) + '_' +\
    df['reality'].astype(str) + '_' + df['income_type'].astype(str) + '_' +\
    df['edu_type'].astype(str) + '_' + df['family_type'].astype(str) + '_' +\
    df['house_type'].astype(str) + '_' + df['occyp_type'].astype(str) + '_' +\
    df['credit'].astype(str)
```


```python
df = df.sort_values(by='ALL')
num_card = df.groupby('ALL').size()
type(num_card)
```




    pandas.core.series.Series




```python
df.drop_duplicates(cols, inplace=True)
df['num_card'] = num_card.values
df
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
      <th>ALL</th>
      <th>num_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>8076</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>-10179</td>
      <td>-1813</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>1.0</td>
      <td>-7.0</td>
      <td>1.0</td>
      <td>0_101250.0_-10179_-1813_0_1_0_1.0_F_N_Y_Workin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>4803</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>-22059</td>
      <td>365243</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-30.0</td>
      <td>0.0</td>
      <td>0_101250.0_-22059_365243_0_1_0_1.0_F_N_Y_Pensi...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>12317</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>-22059</td>
      <td>365243</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-2.0</td>
      <td>1.0</td>
      <td>0_101250.0_-22059_365243_0_1_0_1.0_F_N_Y_Pensi...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>14668</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10396</td>
      <td>-564</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0_103500.0_-10396_-564_0_0_1_2.0_F_N_Y_Working...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>8190</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10396</td>
      <td>-564</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0_103500.0_-10396_-564_0_0_1_2.0_F_N_Y_Working...</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>2671</th>
      <td>2671</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>5</td>
      <td>157500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-13039</td>
      <td>-3375</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>7.0</td>
      <td>-11.0</td>
      <td>2.0</td>
      <td>5_157500.0_-13039_-3375_0_0_0_7.0_F_N_Y_Workin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4094</th>
      <td>4094</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>5</td>
      <td>189000.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Separated</td>
      <td>House / apartment</td>
      <td>-15450</td>
      <td>-428</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>-50.0</td>
      <td>2.0</td>
      <td>5_189000.0_-15450_-428_0_1_0_6.0_F_N_Y_Working...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10321</th>
      <td>10321</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>5</td>
      <td>202500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>Co-op apartment</td>
      <td>-11384</td>
      <td>-2727</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Managers</td>
      <td>7.0</td>
      <td>-46.0</td>
      <td>0.0</td>
      <td>5_202500.0_-11384_-2727_0_0_0_7.0_M_Y_Y_Workin...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17978</th>
      <td>17978</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>5</td>
      <td>202500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>Co-op apartment</td>
      <td>-11384</td>
      <td>-2727</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Managers</td>
      <td>7.0</td>
      <td>-21.0</td>
      <td>1.0</td>
      <td>5_202500.0_-11384_-2727_0_0_0_7.0_M_Y_Y_Workin...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25638</th>
      <td>25638</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>7</td>
      <td>157500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-13827</td>
      <td>-1649</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Cleaning staff</td>
      <td>9.0</td>
      <td>-31.0</td>
      <td>2.0</td>
      <td>7_157500.0_-13827_-1649_1_1_0_9.0_F_N_N_Workin...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>12099 rows × 22 columns</p>
</div>




```python
df.drop('ALL', axis=1, inplace=True)
df.shape
```




    (12099, 21)



### 범주형 변수 파악


```python
print(df.income_type.value_counts())
plt.hist(df.income_type)
plt.xticks(rotation=45)
plt.show()
```

    Working                 6250
    Commercial associate    2854
    Pensioner               2064
    State servant            927
    Student                    4
    Name: income_type, dtype: int64
    


![png](output_17_1.png)



```python
print(df.edu_type.value_counts())
plt.hist(df.edu_type)
plt.xticks(rotation=45)
plt.show()
	['Higher education' ,'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree']


```

    Secondary / secondary special    8295
    Higher education                 3190
    Incomplete higher                 472
    Lower secondary                   132
    Academic degree                    10
    Name: edu_type, dtype: int64
    


![png](output_18_1.png)



```python
df.groupby(['edu_type'])['income_total'].mean().reset_index()
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
      <th>edu_type</th>
      <th>income_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Academic degree</td>
      <td>249750.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Higher education</td>
      <td>220317.937618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Incomplete higher</td>
      <td>204108.368644</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lower secondary</td>
      <td>142329.545455</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Secondary / secondary special</td>
      <td>168527.268716</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.family_type.value_counts())
plt.hist(df.family_type)
plt.xticks(rotation=45)
plt.show()
```

    Married                 8219
    Single / not married    1660
    Civil marriage           996
    Separated                716
    Widow                    508
    Name: family_type, dtype: int64
    


![png](output_20_1.png)



```python
print(df.house_type.value_counts())
plt.hist(df.house_type)
plt.xticks(rotation=45)
plt.show()
```

    House / apartment      10798
    With parents             578
    Municipal apartment      390
    Rented apartment         188
    Office apartment          97
    Co-op apartment           48
    Name: house_type, dtype: int64
    


![png](output_21_1.png)



```python
print(df.occyp_type.value_counts(dropna=False))
```

    NaN                      3666
    Laborers                 2158
    Sales staff              1182
    Core staff               1134
    Managers                  989
    Drivers                   759
    High skill tech staff     457
    Accountants               406
    Medicine staff            371
    Cooking staff             227
    Security staff            209
    Cleaning staff            191
    Private service staff     105
    Low-skill Laborers         64
    Waiters/barmen staff       54
    Secretaries                54
    HR staff                   27
    IT staff                   25
    Realty agents              21
    Name: occyp_type, dtype: int64
    


```python
df.FLAG_MOBIL.value_counts()
```




    1    12099
    Name: FLAG_MOBIL, dtype: int64



- FLAG_MOBIL 모든 값이 1이므로 제거


```python
df.drop(['FLAG_MOBIL'], axis=1, inplace=True)
df.head()
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
      <th>num_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>8076</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>-10179</td>
      <td>-1813</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>1.0</td>
      <td>-7.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>4803</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>-22059</td>
      <td>365243</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-30.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>12317</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>-22059</td>
      <td>365243</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>-2.0</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>14668</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10396</td>
      <td>-564</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>8190</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-10396</td>
      <td>-564</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 2. 전처리

### 결측치


```python
df.isnull().sum()
```




    index               0
    gender              0
    car                 0
    reality             0
    child_num           0
    income_total        0
    income_type         0
    edu_type            0
    family_type         0
    house_type          0
    DAYS_BIRTH          0
    DAYS_EMPLOYED       0
    work_phone          0
    phone               0
    email               0
    occyp_type       3666
    family_size         0
    begin_month         0
    credit              0
    num_card            0
    dtype: int64




```python
df[df['income_type']=='Pensioner']['occyp_type'].value_counts(dropna=False)
```




    NaN               2055
    Laborers             3
    Core staff           2
    Accountants          1
    Managers             1
    Drivers              1
    Medicine staff       1
    Name: occyp_type, dtype: int64



- income_type이 Pensioner인 행의 occyp_type은 대부분 결측치  
=> 결측 중 income_type이 Pensioner인 사람은 'Pensioner'라는 새로운 occyp_type 지정


```python
df['occyp_type'] = np.where(((pd.notnull(df['occyp_type'])==False) & (df['income_type']=='Pensioner')), 'Pensioner', df['occyp_type'])
```


```python
df.occyp_type.isnull().sum()
```




    1611



- 남은 3731개의 결측치 처리는 추후 진행


```python
df[df['income_type']=='Pensioner']['DAYS_EMPLOYED'].value_counts()
```




     365243    2053
    -3680         1
    -5521         1
    -1325         1
    -673          1
    -443          1
    -2208         1
    -2745         1
    -672          1
    -620          1
    -586          1
    -198          1
    Name: DAYS_EMPLOYED, dtype: int64



## 이상치 처리

### 파생 변수


```python
# 음수값 절댓값 취해서 양수로 변환

feats = ['DAYS_BIRTH', 'begin_month', 'DAYS_EMPLOYED']
for feat in feats:
    df[feat]=np.abs(df[feat])
```


```python
# 이해하기 쉽게 나이와 근속년차 변수 생성

df['Age'] = df['DAYS_BIRTH']//365
df['career_year'] = np.ceil(df['DAYS_EMPLOYED']/365)
df['career_start_age'] = df['Age'] - df['career_year']
df.head()
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
      <th>num_card</th>
      <th>Age</th>
      <th>career_year</th>
      <th>career_start_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>8076</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>10179</td>
      <td>1813</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>27</td>
      <td>5.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>4803</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>22059</td>
      <td>365243</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Pensioner</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>60</td>
      <td>1001.0</td>
      <td>-941.0</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>12317</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>22059</td>
      <td>365243</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Pensioner</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>60</td>
      <td>1001.0</td>
      <td>-941.0</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>14668</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>10396</td>
      <td>564</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>8190</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>10396</td>
      <td>564</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ability: 소득 / (살아온 일 수 + 근무일 수) or 소득/근무일 수
df['ability'] = df['income_total'] / (df['DAYS_BIRTH'] + df['DAYS_EMPLOYED'])
df['ability2'] = df['income_total'] / df['DAYS_EMPLOYED']

# income_unit: 소득 / 가족 수
df['income_unit'] = df['income_total']/df['family_size'] # 가족 수 이상치 추후 처리
```


```python
df.head()
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>occyp_type</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
      <th>num_card</th>
      <th>Age</th>
      <th>career_year</th>
      <th>career_start_age</th>
      <th>ability</th>
      <th>ability2</th>
      <th>income_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>8076</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>10179</td>
      <td>1813</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Sales staff</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>27</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>8.443129</td>
      <td>55.846663</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>4803</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>22059</td>
      <td>365243</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Pensioner</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>60</td>
      <td>1001.0</td>
      <td>-941.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>12317</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>101250.0</td>
      <td>Pensioner</td>
      <td>Secondary / secondary special</td>
      <td>Widow</td>
      <td>House / apartment</td>
      <td>22059</td>
      <td>365243</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Pensioner</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>60</td>
      <td>1001.0</td>
      <td>-941.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>14668</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>10396</td>
      <td>564</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>8190</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>103500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>10396</td>
      <td>564</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>IT staff</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
  </tbody>
</table>
</div>



- income_type이 Pensioner인 행의 DAYS_EMPLOYED 값이 대부분 이상치(365243)로 설정되어 있는 문제  
=> 다른 사람들의 평균으로 대체


```python
# 연금받는 사람들 나이의 중앙값
a = df[df['income_type']=='Pensioner']['Age'].median()
```


```python
# 연금받는 사람들이 아닌 사람들 업무 시작 나이의 중앙값
b = df[df['income_type']!='Pensioner']['career_start_age'].median()
```

=> 연금받는 사람들의 근속년수를 (a-b)로 대체


```python
df['career_year'] = np.where(((df['income_type']=='Pensioner') & (df['DAYS_EMPLOYED']==365243)), (a-b), df['career_year'])
```


```python
# 이에 맞게 'DAYS_EMPLOYED'도 수정
df['DAYS_EMPLOYED'] = np.where(((df['income_type']=='Pensioner') & (df['DAYS_EMPLOYED']==365243)), (a-b)*365, df['DAYS_EMPLOYED'])
```


```python
df['career_start_age'] = df['Age'] - df['career_year']
df[df['income_type']=='Pensioner'][['DAYS_EMPLOYED', 'career_year', 'career_start_age']]
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
      <th>DAYS_EMPLOYED</th>
      <th>career_year</th>
      <th>career_start_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4803</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>22083</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>20605</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>8544</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24147</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>15878</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>20752</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>22437</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>11690</th>
      <td>10585.0</td>
      <td>29.0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
<p>2064 rows × 3 columns</p>
</div>



- occyp_type 변수를 income_total변수를 기준으로 categorizing하려하였으나 income_total변수와의 다중공선성 문제로 occyp_type 변수를 아예 삭제?  


```python
df.drop('occyp_type', axis=1, inplace=True)
```


```python
# df.to_csv("train_ppc.csv")
```

- widow의 gender가 M인 경우 -> 이상치로 고려하여 F로 바꿈


```python
df.loc[(df['family_type']=='Widow') & (df['gender']=='M'),'gender'] = 'F'
```


```python
### widow의 income_total 매우 낮은 인사이트 찾음 => 나중에 넣자
```

- 범주형 변수 -> 숫자 변환하기


```python
df['gender'] = df['gender'].apply(lambda x: 0 if x=='M' else 1)
df['car'] = df['car'].apply(lambda x: 1 if x=='Y' else 0)
df['reality'] = df['reality'].apply(lambda x: 1 if x=='Y' else 0)
```


```python
df2 = df.copy()
```


```python
# LabelEncoder
features = ['income_type', 'edu_type', 'family_type', 'house_type']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df2[features] = df2[features].apply(le.fit_transform)
```


```python
df2.head()
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
      <th>index</th>
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>credit</th>
      <th>num_card</th>
      <th>Age</th>
      <th>career_year</th>
      <th>career_start_age</th>
      <th>ability</th>
      <th>ability2</th>
      <th>income_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>8076</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>10179</td>
      <td>1813.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>27</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>8.443129</td>
      <td>55.846663</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>4803</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>22059</td>
      <td>10585.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>60</td>
      <td>29.0</td>
      <td>31.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>12317</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>22059</td>
      <td>10585.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>60</td>
      <td>29.0</td>
      <td>31.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>14668</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>103500.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>10396</td>
      <td>564.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>8190</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>103500.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>10396</td>
      <td>564.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.to_csv("data/train_ppc3.csv")
```

- 상관관계 파악하기


```python
features = ['gender', 'car', 'reality', 'child_num', 'income_total', 'income_type', 'income_unit', 'edu_type', 'family_type', 'house_type', 
            'work_phone', 'phone', 'email', 'family_size', 'Age', 'career_year','ability','ability2', 'num_card','credit']
# plt.pcolor(df2[features])
```


```python
plt.pcolor(df2)
plt.xticks(np.arange(0.5, len(df2.columns), 1), df.columns)
plt.yticks(np.arange(0.5, len(df2.index), 1), df.index)
plt.show()
```


![png](output_62_0.png)



```python
df2[features].corr(method='pearson')['income_total'].sort_values(ascending=False)
```




    income_total    1.000000
    ability         0.746919
    income_unit     0.714088
    ability2        0.250949
    car             0.220338
    email           0.087179
    num_card        0.049676
    reality         0.038093
    child_num       0.026223
    family_size     0.024119
    phone           0.013752
    house_type     -0.004490
    credit         -0.012535
    family_type    -0.016296
    work_phone     -0.034946
    Age            -0.059389
    income_type    -0.073091
    career_year    -0.119632
    gender         -0.199097
    edu_type       -0.232049
    Name: income_total, dtype: float64




```python
df2[features].corr(method='pearson')['credit'].sort_values(ascending=False)
```




    credit          1.000000
    num_card        0.161282
    edu_type        0.023987
    Age             0.013226
    reality         0.006177
    email           0.001986
    career_year     0.001442
    gender          0.000184
    child_num       0.000163
    family_type    -0.000334
    family_size    -0.000436
    car            -0.001155
    income_type    -0.003076
    ability2       -0.003379
    income_unit    -0.005509
    phone          -0.006537
    house_type     -0.010261
    ability        -0.011250
    work_phone     -0.011967
    income_total   -0.012535
    Name: credit, dtype: float64




```python
df2[features].corr(method='pearson')['num_card'].sort_values(ascending=False)
```




    num_card        1.000000
    credit          0.161282
    income_total    0.049676
    gender          0.035643
    ability         0.028756
    income_unit     0.027118
    career_year     0.018927
    car             0.018001
    family_size     0.014973
    reality         0.010463
    child_num       0.008199
    work_phone      0.006967
    phone           0.005917
    email           0.004092
    Age             0.003338
    income_type     0.002610
    house_type     -0.002891
    family_type    -0.012871
    edu_type       -0.017949
    ability2       -0.021975
    Name: num_card, dtype: float64




```python
df2[features].corr(method='pearson')['ability'].sort_values(ascending=False)
```




    ability         1.000000
    income_total    0.746919
    income_unit     0.477359
    ability2        0.427906
    car             0.246699
    child_num       0.168998
    family_size     0.141357
    email           0.128064
    house_type      0.124077
    income_type     0.097318
    work_phone      0.088646
    num_card        0.028756
    credit         -0.010079
    phone          -0.015331
    family_type    -0.042939
    reality        -0.054464
    edu_type       -0.260246
    gender         -0.283376
    Age            -0.581895
    career_year    -0.591428
    Name: ability, dtype: float64



### (불균형 데이터 -> 오버샘플링)

# 우리의 목표는 신용도가 안 좋은 사람을 골라내는 작업  
(더군다나 0,1인 그룹이 2인 그룹보다 상대적으로 매우 적음)  
=> credit이 1인 사람을 모두 0으로 바꿔서 합쳐버려  
=> 다중분류를 이진분류 작업으로 바꿔버려


```python
df2['credit'] = df2['credit'].apply(lambda x: 0 if x==1 else x)
df2['credit'].value_counts()
```




    2.0    6974
    0.0    5125
    Name: credit, dtype: int64




```python
# 나이대별 신용도 분포
df2['grp_age'] = df2['Age']//10
age_credit = df2.groupby(['grp_age','credit'])['index'].count().reset_index()
age_credit.rename(columns={'index':'cnt'}, inplace=True)
age_credit
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
      <th>grp_age</th>
      <th>credit</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.0</td>
      <td>759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2.0</td>
      <td>941</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.0</td>
      <td>1470</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.0</td>
      <td>1937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>1257</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>2.0</td>
      <td>1804</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>0.0</td>
      <td>1124</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>2.0</td>
      <td>1556</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>0.0</td>
      <td>515</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>2.0</td>
      <td>736</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_sum = age_credit.groupby('grp_age')['cnt'].sum()
age_sum
```




    grp_age
    2    1700
    3    3407
    4    3061
    5    2680
    6    1251
    Name: cnt, dtype: int64




```python
merged_df = age_credit.merge(age_sum, on='grp_age')
merged_df['pct'] = merged_df['cnt_x'] / merged_df['cnt_y']
# merged_df.drop(columns = ['cnt_x', 'cnt_y'], inplace=True)
merged_df
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
      <th>grp_age</th>
      <th>credit</th>
      <th>cnt_x</th>
      <th>cnt_y</th>
      <th>pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.0</td>
      <td>759</td>
      <td>1700</td>
      <td>0.446471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2.0</td>
      <td>941</td>
      <td>1700</td>
      <td>0.553529</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.0</td>
      <td>1470</td>
      <td>3407</td>
      <td>0.431465</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.0</td>
      <td>1937</td>
      <td>3407</td>
      <td>0.568535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>1257</td>
      <td>3061</td>
      <td>0.410650</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>2.0</td>
      <td>1804</td>
      <td>3061</td>
      <td>0.589350</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>0.0</td>
      <td>1124</td>
      <td>2680</td>
      <td>0.419403</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>2.0</td>
      <td>1556</td>
      <td>2680</td>
      <td>0.580597</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>0.0</td>
      <td>515</td>
      <td>1251</td>
      <td>0.411671</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>2.0</td>
      <td>736</td>
      <td>1251</td>
      <td>0.588329</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 신용도별 나이 분포 -> 나이대 분포와 거의 비슷해서 인사이트가 없어 보임, 신용카드 연체 요소에는 나이가 고려대상이 아님.
credit_age = merged_df.groupby(['credit', 'grp_age'])['cnt_x'].sum().reset_index()
gp = credit_age.groupby('credit')['cnt_x'].sum()
merged_df2 = credit_age.merge(gp, on='credit')
merged_df2['cr_pct'] = merged_df2['cnt_x_x'] / merged_df2['cnt_x_y']
merged_df2.drop(columns = ['cnt_x_x', 'cnt_x_y'], inplace=True)
merged_df2
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
      <th>credit</th>
      <th>grp_age</th>
      <th>cr_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2</td>
      <td>0.148098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3</td>
      <td>0.286829</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>4</td>
      <td>0.245268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>5</td>
      <td>0.219317</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>6</td>
      <td>0.100488</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>2</td>
      <td>0.134930</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.0</td>
      <td>3</td>
      <td>0.277746</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.0</td>
      <td>4</td>
      <td>0.258675</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>5</td>
      <td>0.223114</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0</td>
      <td>6</td>
      <td>0.105535</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 나이대 분포
grp_age = df2['grp_age'].value_counts().reset_index()
grp_age.rename(columns={'grp_age':'cnt', 'index': 'grp_age'}, inplace=True)
total = grp_age.sum(axis=0)[1]
grp_age['pct'] = grp_age['cnt'] / total
grp_age.sort_values(by='grp_age')
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
      <th>grp_age</th>
      <th>cnt</th>
      <th>pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1700</td>
      <td>0.140507</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>3407</td>
      <td>0.281594</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3061</td>
      <td>0.252996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2680</td>
      <td>0.221506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>1251</td>
      <td>0.103397</td>
    </tr>
  </tbody>
</table>
</div>



<!-- - 신용도가 높은 사람들의 가입 기간 고려하여 비교하기 -->

## 카드 개수에 따른 신용도


```python
card_credit = df[df['num_card']>=2][['num_card', 'credit']]
plt.scatter(x='credit',y='num_card', data=card_credit, alpha=.3, s=2)
plt.show()
```


![png](output_77_0.png)



```python
X = card_credit.groupby(['num_card'])['credit'].mean().reset_index()
Y = card_credit.groupby(['num_card'])['credit'].count().reset_index()

XY = pd.merge(X, Y, on=['num_card'], suffixes=('_mean', '_count'))
XY
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
      <th>num_card</th>
      <th>credit_mean</th>
      <th>credit_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1.455934</td>
      <td>2553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1.537010</td>
      <td>1378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1.623620</td>
      <td>906</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1.644401</td>
      <td>509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>1.698925</td>
      <td>279</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>1.745223</td>
      <td>157</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>1.678899</td>
      <td>109</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>1.571429</td>
      <td>49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>1.833333</td>
      <td>36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>1.500000</td>
      <td>22</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>1.625000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>2.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>1.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19</td>
      <td>1.666667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20</td>
      <td>2.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(XY.head(7)['num_card'], XY.head(7)['credit_mean'],marker='o')
plt.xlabel("The Number of Cards")
plt.ylabel("Mean Credit")
plt.show()
```


![png](output_79_0.png)



```python
# 카드 개수에 따른 수입
card_inc = df[df['num_card']>=2][['num_card', 'income_total']]
X_ = card_inc.groupby(['num_card'])['income_total'].mean().reset_index()
Y_ = card_inc.groupby(['num_card'])['income_total'].count().reset_index()

XY_ = pd.merge(X_, Y_, on=['num_card'], suffixes=('_mean', '_count'))
XY_
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
      <th>num_card</th>
      <th>income_total_mean</th>
      <th>income_total_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>181017.724442</td>
      <td>2553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>183405.726052</td>
      <td>1378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>190476.655629</td>
      <td>906</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>204933.925344</td>
      <td>509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>189545.161290</td>
      <td>279</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>182622.611465</td>
      <td>157</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>205720.183486</td>
      <td>109</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>208974.489796</td>
      <td>49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>173750.000000</td>
      <td>36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>214159.090909</td>
      <td>22</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>185062.500000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>253500.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>288300.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19</td>
      <td>210000.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20</td>
      <td>225000.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>157500.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(XY_[XY_['income_total_mean']>=200000]['num_card'], XY_[XY_['income_total_mean']>=200000]['income_total_mean'],marker='o')
plt.xlabel("The Number of Cards")
plt.ylabel("Mean Income")
plt.show()
```


![png](output_81_0.png)



```python
# 카드 개수에 따른 이메일 소유 여부
card_email = df[df['num_card']>=2][['num_card', 'email']]
X_1 = card_email.groupby(['num_card'])['email'].mean().reset_index()
Y_1 = card_email.groupby(['num_card'])['email'].count().reset_index()

XY_1 = pd.merge(X_1, Y_1, on=['num_card'], suffixes=('_mean', '_count'))
XY_1
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
      <th>num_card</th>
      <th>email_mean</th>
      <th>email_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.093615</td>
      <td>2553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.089260</td>
      <td>1378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.090508</td>
      <td>906</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.110020</td>
      <td>509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.103943</td>
      <td>279</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.101911</td>
      <td>157</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.045872</td>
      <td>109</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.081633</td>
      <td>49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>0.083333</td>
      <td>36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>0.045455</td>
      <td>22</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>0.250000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>0.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>0.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19</td>
      <td>0.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>20</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(XY_1.head(9)['num_card'], XY_1.head(9)['email_mean'],marker='o')
plt.xlabel("The Number of Cards")
plt.ylabel("Mean Num of Email")
plt.show()
```


![png](output_83_0.png)



```python
# 가족 수에 따른 근무일 수
mean = df.groupby('family_size')['DAYS_EMPLOYED'].mean().reset_index()
count = df.groupby('family_size')['DAYS_EMPLOYED'].count().reset_index()
mean_count = pd.merge(mean, count, on=['family_size'], suffixes=('_mean', '_count'))
mean_count
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
      <th>family_size</th>
      <th>DAYS_EMPLOYED_mean</th>
      <th>DAYS_EMPLOYED_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4504.253527</td>
      <td>2410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>4419.630556</td>
      <td>6434</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>2484.872126</td>
      <td>2088</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>2484.197382</td>
      <td>993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>2780.965035</td>
      <td>143</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>2785.500000</td>
      <td>24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>3051.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.0</td>
      <td>1649.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.0</td>
      <td>1689.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20.0</td>
      <td>1853.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(mean_count.head(6)['family_size'], mean_count.head(6)['DAYS_EMPLOYED_mean'],marker='o')
plt.xlabel("Family size")
plt.ylabel("Mean Days of Employed")
plt.show()
```


![png](output_85_0.png)



```python
# 가족 수에 따른 신용도
mean = df.groupby('family_size')['credit'].mean().reset_index()
count = df.groupby('family_size')['credit'].count().reset_index()
mean_count = pd.merge(mean, count, on=['family_size'], suffixes=('_mean', '_count'))
mean_count
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
      <th>family_size</th>
      <th>credit_mean</th>
      <th>credit_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.419917</td>
      <td>2410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.430992</td>
      <td>6434</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.398467</td>
      <td>2088</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.434038</td>
      <td>993</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>1.447552</td>
      <td>143</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>1.375000</td>
      <td>24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>1.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.0</td>
      <td>2.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.0</td>
      <td>2.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20.0</td>
      <td>2.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(mean_count.head(5)['family_size'], mean_count.head(5)['credit_mean'],marker='o')
plt.xlabel("Family size")
plt.ylabel("Mean Credit")
plt.ylim(1.375,1.5)
plt.show()
```


![png](output_87_0.png)



```python
inc_credit = df.groupby('income_type')['credit'].mean().reset_index()
sns.barplot(inc_credit['income_type'], inc_credit['credit'])
plt.xticks(rotation=35)
plt.xlabel('Type of Income')
plt.ylabel("Mean Credit")
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](output_88_1.png)



```python
hs_credit = df.groupby('house_type')['credit'].mean().reset_index()
hs_credit = hs_credit.sort_values(by='credit',ascending=True)
sns.barplot(hs_credit['house_type'], hs_credit['credit'])
plt.xticks(rotation=35)
plt.ylim(1.0,)
plt.xlabel('Type of House')
plt.ylabel("Mean Credit")
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](output_89_1.png)



```python
edu_credit = df.groupby('edu_type')['credit'].mean().reset_index()
edu_credit = edu_credit.sort_values(by='credit',ascending=True)
sns.barplot(edu_credit['edu_type'], edu_credit['credit'])
plt.xticks(rotation=35)
plt.ylim(1.0,)
plt.xlabel('Type of Education')
plt.ylabel("Mean Credit")
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    


![png](output_90_1.png)



```python
df.groupby(['house_type', 'edu_type'])['credit'].mean().reset_index()
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
      <th>house_type</th>
      <th>edu_type</th>
      <th>credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Co-op apartment</td>
      <td>Higher education</td>
      <td>1.238095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Co-op apartment</td>
      <td>Incomplete higher</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Co-op apartment</td>
      <td>Lower secondary</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Co-op apartment</td>
      <td>Secondary / secondary special</td>
      <td>1.476190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>House / apartment</td>
      <td>Academic degree</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>House / apartment</td>
      <td>Higher education</td>
      <td>1.399930</td>
    </tr>
    <tr>
      <th>6</th>
      <td>House / apartment</td>
      <td>Incomplete higher</td>
      <td>1.418667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>House / apartment</td>
      <td>Lower secondary</td>
      <td>1.434783</td>
    </tr>
    <tr>
      <th>8</th>
      <td>House / apartment</td>
      <td>Secondary / secondary special</td>
      <td>1.435255</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Municipal apartment</td>
      <td>Higher education</td>
      <td>1.366197</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Municipal apartment</td>
      <td>Incomplete higher</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Municipal apartment</td>
      <td>Lower secondary</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Municipal apartment</td>
      <td>Secondary / secondary special</td>
      <td>1.471947</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Office apartment</td>
      <td>Higher education</td>
      <td>1.344828</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Office apartment</td>
      <td>Incomplete higher</td>
      <td>1.750000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Office apartment</td>
      <td>Lower secondary</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Office apartment</td>
      <td>Secondary / secondary special</td>
      <td>1.412698</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Rented apartment</td>
      <td>Higher education</td>
      <td>1.230769</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Rented apartment</td>
      <td>Incomplete higher</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Rented apartment</td>
      <td>Lower secondary</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Rented apartment</td>
      <td>Secondary / secondary special</td>
      <td>1.396825</td>
    </tr>
    <tr>
      <th>21</th>
      <td>With parents</td>
      <td>Higher education</td>
      <td>1.380208</td>
    </tr>
    <tr>
      <th>22</th>
      <td>With parents</td>
      <td>Incomplete higher</td>
      <td>1.350000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>With parents</td>
      <td>Lower secondary</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>With parents</td>
      <td>Secondary / secondary special</td>
      <td>1.419255</td>
    </tr>
  </tbody>
</table>
</div>




```python
# House type에 따른 신용도
plt.figure(figsize=(15,7.5))
sns.violinplot(x = 'house_type', y = 'credit', data = df)

plt.xticks(rotation=35)
plt.xlabel('Type of House')
plt.ylabel("Credit")
plt.show()
```


![png](output_92_0.png)



```python
# Edu type에 따른 신용도
plt.figure(figsize=(15,7.5))
sns.violinplot(x = 'edu_type', y = 'credit', data = df)

plt.xticks(rotation=35)
plt.xlabel('Type of Education')
plt.ylabel("Credit")
plt.show()
```


![png](output_93_0.png)


# 모델링
## ANN - 이진 분류


```python
df2.shape
```




    (12099, 25)




```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import StandardScaler

X = df2.drop(columns=['index', 'credit'], axis=1)
y = df2['credit']
y = y.apply(lambda x: 1 if x==2 else x)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```

    (9074, 24) (3025, 24)
    (9074,) (3025,)
    


```python
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=X.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 256)               6400      
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 39,425
    Trainable params: 39,425
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=False)
mc = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
```


```python
history = model.fit(X_train, y_train, 
                    batch_size=10, 
                    validation_split=0.1, 
                    epochs=500, 
                    callbacks=[es, mc])
```

    Epoch 1/500
    817/817 [==============================] - 6s 4ms/step - loss: 0.6610 - accuracy: 0.6133 - val_loss: 0.6523 - val_accuracy: 0.6300
    Epoch 2/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.6358 - accuracy: 0.6424 - val_loss: 0.6423 - val_accuracy: 0.6465
    Epoch 3/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.6226 - accuracy: 0.6542 - val_loss: 0.6341 - val_accuracy: 0.6487
    Epoch 4/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.6156 - accuracy: 0.6594 - val_loss: 0.6279 - val_accuracy: 0.6410
    Epoch 5/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.6065 - accuracy: 0.6667 - val_loss: 0.6242 - val_accuracy: 0.6432
    Epoch 6/500
    817/817 [==============================] - 3s 3ms/step - loss: 0.5993 - accuracy: 0.6678 - val_loss: 0.6382 - val_accuracy: 0.6465
    Epoch 7/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5939 - accuracy: 0.6744 - val_loss: 0.6289 - val_accuracy: 0.6575
    Epoch 8/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5845 - accuracy: 0.6782 - val_loss: 0.6222 - val_accuracy: 0.6476
    Epoch 9/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5770 - accuracy: 0.6842 - val_loss: 0.6387 - val_accuracy: 0.6344
    Epoch 10/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5720 - accuracy: 0.6860 - val_loss: 0.6541 - val_accuracy: 0.6465
    Epoch 11/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5623 - accuracy: 0.7000 - val_loss: 0.6750 - val_accuracy: 0.6388
    Epoch 12/500
    817/817 [==============================] - 3s 4ms/step - loss: 0.5536 - accuracy: 0.7008 - val_loss: 0.6644 - val_accuracy: 0.6289
    


```python
df_history = pd.DataFrame(history.history)
df_h = df_history.loc[:, ['accuracy', 'val_accuracy']]
df_h.plot()
plt.ylim(0,)
```




    (0.0, 0.7052106320858001)




![png](output_101_1.png)



```python
df_h2 = df_history.loc[:, ['loss', 'val_loss']]
df_h2.plot()
print(f"min_val_loss: {df_h2['loss'].min()}")
```

    min_val_loss: 0.4702399671077728
    


![png](output_102_1.png)



```python
score = model.evaluate(X_test, y_test)
print(f'test loss: {score[0]}')
print(f'test accuracy: {score[1]}')
```

    95/95 [==============================] - 0s 2ms/step - loss: 0.6568 - accuracy: 0.6284
    test loss: 0.6568126678466797
    test accuracy: 0.6284297704696655
    

- 쓰레기


```python
from lightgbm import LGBMClassifier
```


```python
lgbm_wrapper = LGBMClassifier(n_estimators=400)
```


```python
evals = [(X_test, y_test)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))
```


```python
get_clf_eval(y_test, preds)
```

    오차행렬:
     [[ 386  900]
     [ 109 1630]]
    
    정확도: 0.6664
    정밀도: 0.6443
    재현율: 0.9373
    F1: 0.7636
    AUC: 0.6187
    


```python
from lightgbm import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm_wrapper, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4152e0d690>




![png](output_110_1.png)



```python
X.head()
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
      <th>gender</th>
      <th>car</th>
      <th>reality</th>
      <th>child_num</th>
      <th>income_total</th>
      <th>income_type</th>
      <th>edu_type</th>
      <th>family_type</th>
      <th>house_type</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>work_phone</th>
      <th>phone</th>
      <th>email</th>
      <th>family_size</th>
      <th>begin_month</th>
      <th>num_card</th>
      <th>Age</th>
      <th>career_year</th>
      <th>career_start_age</th>
      <th>ability</th>
      <th>ability2</th>
      <th>income_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8076</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>10179</td>
      <td>1813.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>27</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>8.443129</td>
      <td>55.846663</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>4803</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>22059</td>
      <td>10585.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>4</td>
      <td>60</td>
      <td>29.0</td>
      <td>31.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>12317</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>101250.0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>22059</td>
      <td>10585.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>60</td>
      <td>29.0</td>
      <td>31.0</td>
      <td>0.261424</td>
      <td>0.277213</td>
      <td>101250.0</td>
    </tr>
    <tr>
      <th>14668</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>103500.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>10396</td>
      <td>564.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
    <tr>
      <th>8190</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>103500.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>10396</td>
      <td>564.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>28</td>
      <td>2.0</td>
      <td>26.0</td>
      <td>9.443431</td>
      <td>183.510638</td>
      <td>51750.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#제일 높은 feature importance :begin month
```



# Label 바꾸지 않은 데이터 (0,1,2)


```python
df.shape
```




    (12099, 25)




```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['index', 'credit'], axis=1)
y = df['credit']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_15 (Dense)             (None, 128)               3200      
    _________________________________________________________________
    dense_16 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    dense_17 (Dense)             (None, 1)                 65        
    =================================================================
    Total params: 11,521
    Trainable params: 11,521
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=False)
mc = ModelCheckpoint(filepath='best_model2.h5', monitor='val_accuracy', save_best_only=True)
```


```python
history = model.fit(X_train, y_train, 
                    batch_size=10, 
                    validation_split=0.1, 
                    epochs=300, 
                    callbacks=[es, mc])
```


```python
df_history = pd.DataFrame(history.history)
df_h = df_history.loc[:, ['accuracy', 'val_accuracy']]
df_h.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f7d8bb09590>




![png](output_120_1.png)


## Catboost


```python
import catboost as cb
from sklearn.metrics import classification_report
```


```python
pip install catboost
```


```python
import catboost as cb
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df2.drop(columns=['index', 'credit'], axis=1)
y = df2['credit']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
train_dataset = cb.Pool(X_train,y_train)
test_dataset = cb.Pool(X_test,y_test)
```


```python
model = cb.CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy')
```


```python
grid = {'learning_rate': [0.03, 0.1],
'depth': [4, 6, 10],
'l2_leaf_reg': [1, 3, 5,],
'iterations': [50, 100, 150]}
```


```python
model.grid_search(grid,train_dataset)
```


```python
from sklearn.metrics import classification_report

pred = model.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.00      0.00      0.00       432
             1.0       0.70      0.29      0.41       854
             2.0       0.64      0.98      0.77      1739
    
        accuracy                           0.65      3025
       macro avg       0.45      0.42      0.39      3025
    weighted avg       0.57      0.65      0.56      3025
    
    

    /usr/local/lib/python3.7/dist-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
def plot_feature_importance(importance,names,model_type):

  #Create arrays from feature importance and feature names
  feature_importance = np.array(importance)
  feature_names = np.array(names)

  #Create a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

  #Define size of bar plot
  plt.figure(figsize=(10,8))
  #Plot Searborn bar chart
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  #Add chart labels
  plt.title(model_type + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')

plot_feature_importance(model.get_feature_importance(),X.columns,'CATBOOST')
```


![png](output_131_0.png)


## LGBM


```python
from lightgbm import LGBMClassifier
```


```python
lgbm_wrapper = LGBMClassifier(n_estimators=400)
```


```python
evals = [(X_test, y_test)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, 
                 eval_metric="multi_logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
```


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, preds))
```

                  precision    recall  f1-score   support
    
             0.0       0.38      0.01      0.01       432
             1.0       0.72      0.28      0.40       854
             2.0       0.64      0.99      0.77      1739
    
        accuracy                           0.65      3025
       macro avg       0.58      0.42      0.40      3025
    weighted avg       0.62      0.65      0.56      3025
    
    


```python
# LightGbm Classifier 모델 정확도
lgbm_wrapper.score(X_test, y_test)
```




    0.6456198347107438




```python
# Catboost 모델 정확도
model.score(X_test, y_test)
```




    0.6466115702479339


