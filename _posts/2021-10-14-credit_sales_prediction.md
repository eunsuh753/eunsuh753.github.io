---
layout: post
title: "신용카드 매출 예측"

toc: true
sticky-toc: true
---

```python
import pandas as pd

train = pd.read_csv("data/funda_train.csv")
```


```python
train.describe()
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
      <th>store_id</th>
      <th>card_id</th>
      <th>installment_term</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.556613e+06</td>
      <td>6.556613e+06</td>
      <td>6.556613e+06</td>
      <td>6.556613e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.084930e+03</td>
      <td>2.268127e+06</td>
      <td>1.382017e-01</td>
      <td>1.043511e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.152183e+02</td>
      <td>1.351058e+06</td>
      <td>1.188152e+00</td>
      <td>3.104031e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.771429e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.860000e+02</td>
      <td>1.088828e+06</td>
      <td>0.000000e+00</td>
      <td>2.142857e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.074000e+03</td>
      <td>2.239304e+06</td>
      <td>0.000000e+00</td>
      <td>4.285714e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.615000e+03</td>
      <td>3.438488e+06</td>
      <td>0.000000e+00</td>
      <td>8.571429e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.136000e+03</td>
      <td>4.663856e+06</td>
      <td>9.300000e+01</td>
      <td>5.571429e+06</td>
    </tr>
  </tbody>
</table>

</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6556613 entries, 0 to 6556612
    Data columns (total 9 columns):
     #   Column            Dtype  
    ---  ------            -----  
     0   store_id          int64  
     1   card_id           int64  
     2   card_company      object 
     3   transacted_date   object 
     4   transacted_time   object 
     5   installment_term  int64  
     6   region            object 
     7   type_of_business  object 
     8   amount            float64
    dtypes: float64(1), int64(3), object(5)
    memory usage: 450.2+ MB


전처리

* 매출액에 음수 값이 보이고, 환불은 금액으로 예상됨
* 환불은 log 정규화를 했을때 무한대가 나오기 때문에 제거 하기로 결정
* 환불발생 이전 데이터 중 카드아이디가 같고 환불액의 절대값이 같은 후보 리스트를 찾음
* 환불 후보리스트 중 가장 최근시간(max)을 제거
* 시계열 모델링을 위해 month 단위로 resampling 진행 (daily sampling결과 상당히 난잡한 형태로 나오는 결과를 얻었다)
* 상점 매출이 발생하지 월은 log 정규화시 0이 아닌 최솟값으로 대치하기 위해 2로 대치(log1=0 ,log2=0.693)

+ 상점 매출이 최근 3개월간 거래 내역 (결제 내역) 이 존재하지 않다면 삭제(폐점으로 간주)
+ 폐점 비율을 시각화하는것도 유의미 할 것 같다.

* 3가지 정규화를 진행
  log / box - cox / 제곱근
  그 중, log 정규화가 비교적 정규성을 잘 표현한다고 결론지었다.

2개의 모델링 추가(prophet / arima)
최근 3개월 값의 mse 값 차이를 반영하여 두 모델간 성능 비교

* resampling시 영업 시작전 데이터는 제거하고 시작일 부터 데이터를 유지시킴


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pmdarima as pm
import pmdarima
import statsmodels
import math

from tqdm import tqdm

from pmdarima.arima.stationarity import ADFTest #단위근검정 ADF_TEST
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #AR값을 정하기위한 acf,pcaf
import warnings
warnings.filterwarnings("ignore")

print('Numpy: %s'%(np.__version__))
print('Pandas: %s'%(pd.__version__))

print('seaborn: %s'%(sns.__version__))
print('statsmodel: %s'%(statsmodels.__version__))
print('matplotlib: %s'%(matplotlib.__version__))

```

    Numpy: 1.19.5
    Pandas: 1.2.0
    seaborn: 0.10.0
    statsmodel: 0.12.1
    matplotlib: 3.1.3



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
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>region</th>
      <th>type_of_business</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>b</td>
      <td>2016-06-01</td>
      <td>13:13</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>1857.142857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>h</td>
      <td>2016-06-01</td>
      <td>18:12</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>857.142857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>18:52</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>20:22</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>7857.142857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>c</td>
      <td>2016-06-02</td>
      <td>11:06</td>
      <td>0</td>
      <td>NaN</td>
      <td>기타 미용업</td>
      <td>2000.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
train.isna().sum()
```




    store_id                  0
    card_id                   0
    card_company              0
    transacted_date           0
    transacted_time           0
    installment_term          0
    region              2042766
    type_of_business    3952609
    amount                    0
    dtype: int64




```python
len(train)
```




    6556613



region 변수의 경우 31%가, type of business 의 경우 60% 이상이 missing 되었다. 따라서 두 변수는 아예 drop 하도록 결정하였다.


```python
#변수 2개 드랍
train = train.drop(['region','type_of_business'], axis = 1)
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
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>b</td>
      <td>2016-06-01</td>
      <td>13:13</td>
      <td>0</td>
      <td>1857.142857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>h</td>
      <td>2016-06-01</td>
      <td>18:12</td>
      <td>0</td>
      <td>857.142857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>18:52</td>
      <td>0</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>20:22</td>
      <td>0</td>
      <td>7857.142857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>c</td>
      <td>2016-06-02</td>
      <td>11:06</td>
      <td>0</td>
      <td>2000.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 데이터 잠시 잘라서 사용
#train_raw = train
train = train.iloc[0:100000]
```


```python
plt.figure(figsize=(8, 4))
sns.boxplot(train['amount'])
```




    <AxesSubplot:xlabel='amount'>



![output_12_1](https://user-images.githubusercontent.com/62747570/138007422-f16c8275-96f9-4f55-a178-38fe777f4e4a.png)





```python
# amount값이 음수인 데이터들이 존재 (환불값)
train[train['amount']<0].head()
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
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>40</td>
      <td>a</td>
      <td>2016-06-10</td>
      <td>17:26</td>
      <td>2</td>
      <td>-8571.428571</td>
    </tr>
    <tr>
      <th>347</th>
      <td>0</td>
      <td>285</td>
      <td>a</td>
      <td>2016-08-04</td>
      <td>17:52</td>
      <td>0</td>
      <td>-1857.142857</td>
    </tr>
    <tr>
      <th>731</th>
      <td>0</td>
      <td>473</td>
      <td>g</td>
      <td>2016-10-17</td>
      <td>10:32</td>
      <td>0</td>
      <td>-2000.000000</td>
    </tr>
    <tr>
      <th>831</th>
      <td>0</td>
      <td>230</td>
      <td>b</td>
      <td>2016-11-03</td>
      <td>15:36</td>
      <td>0</td>
      <td>-85.714286</td>
    </tr>
    <tr>
      <th>944</th>
      <td>0</td>
      <td>138</td>
      <td>a</td>
      <td>2016-11-28</td>
      <td>13:21</td>
      <td>0</td>
      <td>-57.142857</td>
    </tr>
  </tbody>
</table>

</div>




```python
def refund_remove(df):
    refund=df[df['amount']<0]
    non_refund=df[df['amount']>0]
    remove_data=pd.DataFrame()
    
    for i in tqdm(df.store_id.unique()):
        divided_data=non_refund[non_refund['store_id']==i] ##non_refund 스토어 데이터를 스토어별로 나눔
        divided_data2=refund[refund['store_id']==i] ##refund 스토어 데이터를 나눔 스토어별로 나눔
        
        for neg in divided_data2.to_records()[:]: ##환불데이터를 차례대로 검사
            refund_store=neg['store_id']
            refund_id=neg['card_id'] ## 환불 카드 아이디
            refund_datetime=neg['datetime'] ## 환불 시간
            refund_amount=abs(neg['amount']) ## 환불액 절대값을 씌움
                
            ##환불시간 이전의 데이터중 카드이이디와 환불액이 같은 후보 리스트를 뽑는다.
            refund_pay_list=divided_data[divided_data['datetime']<=refund_datetime]
            refund_pay_list=refund_pay_list[refund_pay_list['card_id']==refund_id]
            refund_pay_list=refund_pay_list[refund_pay_list['amount']==refund_amount]
                
                
            #후보리스트가 있으면,카드아이디, 환불액이 같으면서 가장 최근시간을 제거
            if(len(refund_pay_list)!=0):
                refund_datetime=max(refund_pay_list['datetime']) ##가장 최근 시간을 구한다
                remove=divided_data[divided_data['datetime']==refund_datetime] ##가장 최근시간
                remove=remove[remove['card_id']==refund_id] ##환불 카드 아이디
                remove=remove[remove['amount']==refund_amount] ##환불액
                divided_data=divided_data.drop(index=remove.index) #인덱스를 통해 제거
                    
        ##제거한데이터를 데이터프레임에 추가한다.
        remove_data=pd.concat([remove_data,divided_data],axis=0)
    
    return remove_data

##월별로 다운 샘플링해주는 함수
def month_resampling(df):
    new_data=pd.DataFrame() 
    df['year_month']=df['transacted_date'].str.slice(stop=7)
    year_month=df['year_month'].drop_duplicates()
    
    downsampling_data=df.groupby(['store_id', 'year_month']).amount.sum()
    downsampling_data=pd.DataFrame(downsampling_data)
    downsampling_data=downsampling_data.reset_index(drop=False,inplace=False)
    
    for i in tqdm(df.store_id.unique()):
        store=downsampling_data[downsampling_data['store_id']==i]
        start_time=min(store['year_month'])
        store=store.merge(year_month,how='outer')
        store=store.sort_values(by=['year_month'], axis=0, ascending=True) ##데이터를 시간순으로 정렬
        
        store['amount']=store['amount'].fillna(2)   #매출이 발생하지 않는 월은 2로 채움
        store['store_id']=store['store_id'].fillna(i) #store_id 결측치 채운다.
        store=store[store['year_month']>=start_time]  #매출 시작일 이후만 뽑는다.
        
        new_data=pd.concat([new_data,store],axis=0)
        
    return new_data

        
    return new_data
##상점 매출 시계열 그래프
def store_plot(data,start_id,end_id):
    plt.figure(figsize=(15, 6))
    for i in data['store_id'].unique()[start_id:end_id]:
        plt.plot(data[data['store_id']== i].index, data[data['store_id'] == i].amount, label='store_{}'.format(i))
    plt.legend()  

##상점 매출 분포
def store_displot(data,start_id,end_id):
    plt.figure(figsize=(15, 6))
    for i in data.store_id.unique()[start_id:end_id]:
        sns.distplot(data[data.store_id == i].amount)
    plt.grid()
    plt.show()
    
##Series 데이터로 변환 함수
def time_series(df,i):
    store=df[df['store_id']==i]
    index=pd.date_range(min(store['year_month']),'2019-03',freq='BM') ##영업 시작일부터 2019년 2월까지 데이터가 존제
    ts=pd.Series(store['amount'].values,index=index)
    return ts
def acf_pacf_plot(data=None,store_id=None):
    ts=time_series(data,store_id)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    plot_acf(ts,lags=20,ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(ts, lags=20, ax=ax2)

##매출 변동계수를 구하는 함수
def coefficient_variation(df,i):
    cv_data=df.groupby(['store_id']).amount.std()/df.groupby(['store_id']).amount.mean()
    cv=cv_data[i]
    return cv
```


```python
train['datetime'] = pd.to_datetime(train.transacted_date + " " + 
                                train.transacted_time, format='%Y-%m-%d %H:%M:%S')
```


```python
train_remove=refund_remove(train)
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 27/27 [00:09<00:00,  2.77it/s]



```python
train_remove.head()
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
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>amount</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>b</td>
      <td>2016-06-01</td>
      <td>13:13</td>
      <td>0</td>
      <td>1857.142857</td>
      <td>2016-06-01 13:13:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>h</td>
      <td>2016-06-01</td>
      <td>18:12</td>
      <td>0</td>
      <td>857.142857</td>
      <td>2016-06-01 18:12:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>18:52</td>
      <td>0</td>
      <td>2000.000000</td>
      <td>2016-06-01 18:52:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>20:22</td>
      <td>0</td>
      <td>7857.142857</td>
      <td>2016-06-01 20:22:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>c</td>
      <td>2016-06-02</td>
      <td>11:06</td>
      <td>0</td>
      <td>2000.000000</td>
      <td>2016-06-02 11:06:00</td>
    </tr>
  </tbody>
</table>

</div>



# 3개월간 기록 없을 경우 제거하는 과정


```python
# 날짜와 시간 합친 변수 date 생성
#train_remove['date'] = train_remove['transacted_date']+' '+train_remove['transacted_time']
#train_remove['date'] = pd.to_datetime(train_remove['date'])

# 날짜 관련 피처 추출과 전처리
from workalendar.asia import SouthKorea
cal = SouthKorea()
train_remove['weekday'] = train_remove['datetime'].dt.weekday
train_remove['holiday'] = train_remove['datetime'].apply(lambda x: cal.is_holiday(x)).astype(int)
train_remove['workday'] = train_remove['datetime'].apply(lambda x: cal.is_working_day(x)).astype(int)

train_remove.set_index('datetime', drop=False, inplace=True) 
train_remove.sort_index(inplace=True) # date 순으로 재배열
train_remove.head()
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
      <th>store_id</th>
      <th>card_id</th>
      <th>card_company</th>
      <th>transacted_date</th>
      <th>transacted_time</th>
      <th>installment_term</th>
      <th>amount</th>
      <th>datetime</th>
      <th>weekday</th>
      <th>holiday</th>
      <th>workday</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-06-01 09:34:00</th>
      <td>20</td>
      <td>56020</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>09:34</td>
      <td>0</td>
      <td>188571.428571</td>
      <td>2016-06-01 09:34:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-06-01 09:41:00</th>
      <td>27</td>
      <td>66099</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>09:41</td>
      <td>0</td>
      <td>1114.285714</td>
      <td>2016-06-01 09:41:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-06-01 09:49:00</th>
      <td>27</td>
      <td>66100</td>
      <td>a</td>
      <td>2016-06-01</td>
      <td>09:49</td>
      <td>0</td>
      <td>1314.285714</td>
      <td>2016-06-01 09:49:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-06-01 10:21:00</th>
      <td>27</td>
      <td>66101</td>
      <td>g</td>
      <td>2016-06-01</td>
      <td>10:21</td>
      <td>0</td>
      <td>1157.142857</td>
      <td>2016-06-01 10:21:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-06-01 10:41:00</th>
      <td>17</td>
      <td>48134</td>
      <td>c</td>
      <td>2016-06-01</td>
      <td>10:41</td>
      <td>0</td>
      <td>1485.714286</td>
      <td>2016-06-01 10:41:00</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>




```python
active_stores = train_remove.last('3M')['store_id'].unique()
# last('3M'): 마지막 3개월에 데이터가 존재하는가
train_remove['is_active'] = train_remove['store_id'].isin(active_stores).astype(int)
train_remove['is_active'].value_counts()

```




    1    93957
    0     4446
    Name: is_active, dtype: int64




```python
# 3개월간 거래 데이터가 없는 상점을 drop 시켰다
train_remove = train_remove[train_remove['is_active'] == 1]
train_remove['is_active'].value_counts()
```




    1    93957
    Name: is_active, dtype: int64



## EDA

### 요일별 매출 비교


```python
# 0: 월 ~ 6: 일
week_amount = pd.DataFrame(train_remove[['store_id', 'transacted_date', 'weekday','amount']].groupby(['store_id', 'weekday']).mean())
week_amount.reset_index(inplace=True)
week_amount.head(10)
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
      <th>store_id</th>
      <th>weekday</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>5199.281488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>5861.904762</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>5157.750343</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>5224.659158</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>5316.340782</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>5</td>
      <td>5953.328710</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>6</td>
      <td>5583.867320</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>938.944873</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1</td>
      <td>987.021727</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>1001.012518</td>
    </tr>
  </tbody>
</table>

</div>




```python
from matplotlib import font_manager, rc
font_path = "C:\\Windows\\Fonts\\malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

x = [0,1,2,3,4,5,6]
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
week_amount[week_amount['store_id']==0].plot.bar('weekday', 'amount')
plt.xlabel("요일")
plt.title("요일별 매출 비교")
plt.xticks(x,weekdays)
plt.show()
```

![output_25_0](https://user-images.githubusercontent.com/62747570/138007465-f50e797d-985e-48ee-85e5-eeef8688fd68.png)





```python
targets=[0,1,2,3,4,5,6,7,8,9]
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
g = sns.barplot(x='store_id', y='amount', hue='weekday', data=week_amount[week_amount['store_id'].isin(targets)],
           dodge=True)
plt.title("요일별 매출 비교")
plt.show()
```

![output_26_0](https://user-images.githubusercontent.com/62747570/138007469-8ab2939c-5d91-4110-a12c-4dc054b37a72.png)




### 근무일/휴일 매출 비교


```python
work_amount = pd.DataFrame(train_remove[['store_id', 'transacted_date', 'workday','amount']].groupby(['store_id', 'workday']).mean())
work_amount.reset_index(inplace=True)
work_amount.head(10)
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
      <th>store_id</th>
      <th>workday</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>5763.705867</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>5222.904199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1109.126984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>973.890039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>16235.647530</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>1</td>
      <td>15338.775510</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>0</td>
      <td>4922.960230</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>1</td>
      <td>5387.078724</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>0</td>
      <td>5555.884409</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>1</td>
      <td>5661.264779</td>
    </tr>
  </tbody>
</table>

</div>




```python
targets=[0,1,2,3,4,5,6,7,8,9]
g = sns.barplot(x='store_id', y='amount', hue='workday', data=work_amount[work_amount['store_id'].isin(targets)],
           dodge=True)
```

![output_29_0](https://user-images.githubusercontent.com/62747570/138007472-d2743afa-8aa4-4c6d-b448-03b8523fdbb1.png)





```python
##월별로 데이터 다운샘플링 시행 
resampling_data=month_resampling(train_remove)
resampling_data['store_id']=resampling_data['store_id'].astype(int)
pd.set_option('display.float_format', '{:.2f}'.format)
resampling_data
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 26/26 [00:00<00:00, 89.94it/s]





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
      <th>store_id</th>
      <th>year_month</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>2016-06</td>
      <td>442467.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>2016-07</td>
      <td>417571.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>2016-08</td>
      <td>303021.43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>2016-09</td>
      <td>242408.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>2016-10</td>
      <td>148896.43</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>13</td>
      <td>2018-10</td>
      <td>897900.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>13</td>
      <td>2018-11</td>
      <td>770485.71</td>
    </tr>
    <tr>
      <th>19</th>
      <td>13</td>
      <td>2018-12</td>
      <td>877942.86</td>
    </tr>
    <tr>
      <th>20</th>
      <td>13</td>
      <td>2019-01</td>
      <td>714242.86</td>
    </tr>
    <tr>
      <th>21</th>
      <td>13</td>
      <td>2019-02</td>
      <td>787157.14</td>
    </tr>
  </tbody>
</table>
<p>777 rows × 3 columns</p>

</div>




```python
store_2=time_series(resampling_data,2)
store_2.plot()
```




    <AxesSubplot:>



![output_31_1](https://user-images.githubusercontent.com/62747570/138007474-b610391c-ba6c-4918-bea5-079a395d8f95.png)





```python
store_1=time_series(resampling_data,1)
sns.boxplot(store_1)
```




    <AxesSubplot:>



![output_32_1](https://user-images.githubusercontent.com/62747570/138007475-8140fec0-2b56-4302-9f96-9bc448e8e292.png)




# 변환 과정

### 전처리를 진행했다. 이후 데이터를 관찰했을 때, 변동폭이 크다는 점, 


```python
from prophet import Prophet
```


```python
#test / train 분리한 코드
outcome = pd.DataFrame()
pred = []
for i in tqdm(resampling_data.store_id.unique()):
    
    store=resampling_data[resampling_data['store_id']==i]
## 변동계수가 0.3 이하일 경우에만 변환을 적용
    cv=coefficient_variation(resampling_data,i)
    if cv<0.3:
        store.amount=np.log(store['amount'])
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        k = (len(store) - 3)
        # 최근 3개월 데이터 분리
        train = store[:k]
        test = store[k:]
        #Prophet 모델
        model = Prophet(seasonality_mode = 'additive')
        model.fit(train)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        #exp 를 활용하여 원래 값으로 치환
#         forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: np.exp(x))
#         model.history['y'] = model.history['y'].apply(lambda x : np.exp(x))
#         store['y'] = store.y.apply(lambda x : np.exp(x))
        model.plot(forecast)
    else :
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        k = (len(store) - 3)
        train = store[:k]
        test = store[k:]
        model = Prophet(seasonality_mode = 'additive')
        model.fit(train)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)     
        model.plot(forecast)
    pred = pd.concat([test, forecast.yhat.tail(3)],axis = 1)
    outcome = pd.concat([outcome, pred], ignore_index = True)


    
```

      0%|                                                                                           | 0/26 [00:00<?, ?it/s]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
      4%|███▏                                                                               | 1/26 [00:07<02:56,  7.07s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
      8%|██████▍                                                                            | 2/26 [00:12<02:20,  5.85s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     12%|█████████▌                                                                         | 3/26 [00:16<02:03,  5.38s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     15%|████████████▊                                                                      | 4/26 [00:20<01:43,  4.71s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     19%|███████████████▉                                                                   | 5/26 [00:23<01:27,  4.18s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     23%|███████████████████▏                                                               | 6/26 [00:28<01:24,  4.23s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     27%|██████████████████████▎                                                            | 7/26 [00:31<01:17,  4.08s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     31%|█████████████████████████▌                                                         | 8/26 [00:36<01:15,  4.22s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     35%|████████████████████████████▋                                                      | 9/26 [00:42<01:21,  4.80s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     38%|███████████████████████████████▌                                                  | 10/26 [00:47<01:18,  4.90s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     42%|██████████████████████████████████▋                                               | 11/26 [00:51<01:10,  4.70s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 23.
     46%|█████████████████████████████████████▊                                            | 12/26 [00:56<01:03,  4.55s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 21.
     50%|█████████████████████████████████████████                                         | 13/26 [00:59<00:55,  4.25s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 21.
     54%|████████████████████████████████████████████▏                                     | 14/26 [01:03<00:50,  4.18s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 20.
     58%|███████████████████████████████████████████████▎                                  | 15/26 [01:07<00:46,  4.20s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 20.
     62%|██████████████████████████████████████████████████▍                               | 16/26 [01:11<00:39,  4.00s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 19.
     65%|█████████████████████████████████████████████████████▌                            | 17/26 [01:16<00:37,  4.22s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 19.
     69%|████████████████████████████████████████████████████████▊                         | 18/26 [01:19<00:32,  4.02s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 19.
     73%|███████████████████████████████████████████████████████████▉                      | 19/26 [01:23<00:26,  3.81s/it]INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 19.
     77%|███████████████████████████████████████████████████████████████                   | 20/26 [03:32<04:09, 41.56s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 17.
     81%|██████████████████████████████████████████████████████████████████▏               | 21/26 [03:38<02:34, 30.93s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 16.
     85%|█████████████████████████████████████████████████████████████████████▍            | 22/26 [03:44<01:33, 23.41s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 16.
     88%|████████████████████████████████████████████████████████████████████████▌         | 23/26 [03:50<00:54, 18.04s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
     92%|███████████████████████████████████████████████████████████████████████████▋      | 24/26 [03:58<00:30, 15.19s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 15.
     96%|██████████████████████████████████████████████████████████████████████████████▊   | 25/26 [04:06<00:12, 12.86s/it]INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:prophet:n_changepoints greater than number of observations. Using 14.
    100%|██████████████████████████████████████████████████████████████████████████████████| 26/26 [04:13<00:00,  9.77s/it]



![output_35_1](https://user-images.githubusercontent.com/62747570/138008153-957b4ed6-a6a7-45bd-b8f4-6d4c93916c19.png)





![output_35_2](https://user-images.githubusercontent.com/62747570/138008156-38f66297-b163-4d86-b3e3-af024aea710b.png)





![output_35_3](https://user-images.githubusercontent.com/62747570/138008158-7ca2552f-fbf7-4eb5-961a-91de34e63e00.png)





![output_35_4](https://user-images.githubusercontent.com/62747570/138008160-be17e800-b084-45af-927e-937c6c730aff.png)





![output_35_5](https://user-images.githubusercontent.com/62747570/138008161-71955c81-cbb3-4a99-8768-1e59be5c9317.png)



![png](output_35_6.png)



![png](output_35_7.png)



![png](output_35_8.png)



![png](output_35_9.png)



![png](output_35_10.png)



![png](output_35_11.png)



![png](output_35_12.png)



![png](output_35_13.png)



![png](output_35_14.png)



![png](output_35_15.png)



![png](output_35_16.png)



![png](output_35_17.png)



![png](output_35_18.png)



![png](output_35_19.png)



![png](output_35_20.png)



![output_35_21](https://user-images.githubusercontent.com/62747570/138008329-63b7c715-7397-463b-96b8-b874b77fb773.png)





![output_35_22](https://user-images.githubusercontent.com/62747570/138008331-b472d2e3-59bc-4ebc-83e3-8b9ab4b772fa.png)





![output_35_23](https://user-images.githubusercontent.com/62747570/138008333-6d979e0a-cac4-437e-a363-337050fb6720.png)





![output_35_24](https://user-images.githubusercontent.com/62747570/138008334-6f20f07c-3871-4858-a0d0-fa1912f24767.png)





![output_35_25](https://user-images.githubusercontent.com/62747570/138008335-f674b9db-267b-4bd2-be4f-e4693b6940d1.png)





![png](output_35_26.png)



```python
#RMSE 값 계산을 위한 처리
from sklearn.metrics import mean_squared_error, r2_score

```


```python
# outcome[['y','yhat']] = outcome[['y','yhat']].apply(lambda x : np.log(x))
print(outcome)
print(outcome.isna().sum())
outcome = outcome.dropna()
outcome = outcome.sort_values(by='store_id' ,ascending=True)
```

        store_id       ds     y  yhat
    17      0.00  2019-02 13.15 13.23
    16      0.00  2019-01 13.43 13.41
    15      0.00  2018-12 13.68 13.66
    20      1.00  2019-02 11.26 11.08
    19      1.00  2019-01 11.91 11.42
    ..       ...      ...   ...   ...
    52     26.00  2019-01 12.74   NaN
    53     26.00  2019-02 12.83 12.61
    5      27.00  2019-02 14.26 14.48
    4      27.00  2019-01 14.32 15.01
    3      27.00  2018-12 14.28 13.89
    
    [75 rows x 4 columns]
    store_id    0
    ds          0
    y           0
    yhat        2
    dtype: int64



```python
adding = 0
for i in outcome.store_id.unique():
    store = outcome[outcome.store_id == i]
    rmse = mean_squared_error(store.yhat, store.y)**0.5
    adding += rmse
    print("%d번째 상점의 MSE의 값은 %.04f입니다" %(i, rmse))
adding = adding / len(outcome.store_id.unique())
print("전체 MSE 값의 평균은 %.04f입니다" %(adding))
```

    0번째 상점의 MSE의 값은 0.0025입니다
    1번째 상점의 MSE의 값은 0.0936입니다
    2번째 상점의 MSE의 값은 0.0521입니다
    4번째 상점의 MSE의 값은 0.2767입니다
    5번째 상점의 MSE의 값은 0.1501입니다
    6번째 상점의 MSE의 값은 0.1099입니다
    7번째 상점의 MSE의 값은 3.9109입니다
    8번째 상점의 MSE의 값은 0.1105입니다
    9번째 상점의 MSE의 값은 0.7510입니다
    10번째 상점의 MSE의 값은 0.0111입니다
    11번째 상점의 MSE의 값은 0.3116입니다
    12번째 상점의 MSE의 값은 0.2751입니다
    13번째 상점의 MSE의 값은 0.0144입니다
    14번째 상점의 MSE의 값은 0.5937입니다
    15번째 상점의 MSE의 값은 0.8803입니다
    16번째 상점의 MSE의 값은 0.2167입니다
    17번째 상점의 MSE의 값은 0.1293입니다
    18번째 상점의 MSE의 값은 0.7811입니다
    19번째 상점의 MSE의 값은 0.0178입니다
    20번째 상점의 MSE의 값은 0.6644입니다
    22번째 상점의 MSE의 값은 0.0841입니다
    23번째 상점의 MSE의 값은 3.5296입니다
    24번째 상점의 MSE의 값은 1.7269입니다
    25번째 상점의 MSE의 값은 2.3238입니다
    26번째 상점의 MSE의 값은 0.2155입니다
    27번째 상점의 MSE의 값은 0.2232입니다
    전체 MSE 값의 평균은 0.6714입니다



```python
#예측 모델 (이후 3개월)
#test / train 분리한 코드
prediction_prophet = pd.DataFrame()

    
for i in tqdm(resampling_data.store_id.unique()):
    
    store=resampling_data[resampling_data['store_id']==i]

## 변동계수가 0.3 이하일 경우에만 변환을 적용
    cv=coefficient_variation(resampling_data,i)
    if cv<0.3:
        store.amount=np.log(store['amount'])
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        #exp 를 활용하여 원래 값으로 치환
        forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: np.exp(x))
        model.history['y'] = model.history['y'].apply(lambda x : np.exp(x))
        model.plot(forecast)
    else :
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        model.plot(forecast)
    predition_prophet = pd.concat([prediction_prophet, forecast['ds','yhat']], ignore_index = True)
```

단순 제곱근 변환을 한 부분 (예측 모델은 일단 프로펫


```python
for i in tqdm(resampling_data.store_id.unique()):
    pred=[]
    store=resampling_data[resampling_data['store_id']==i]

    if cv<0.3:
        store.amount=np.sqrt(store['amount'])
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: np.square(x))
        model.history['y'] = model.history['y'].apply(lambda x : np.square(x))
        model.plot(forecast)
    else :
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        model.plot(forecast)

```

박스 콕스 변환 추가한 것


```python
from scipy import stats #박스 콕스 라이브러리
from scipy.special import inv_boxcox # 역함수

for i in tqdm(resampling_data.store_id.unique()):
    pred=[]
    store=resampling_data[resampling_data['store_id']==i]

    if cv<0.3:
        store.amount, fitted_lambda =stats.boxcox(store['amount'])
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, fitted_lambda))
        model.history['y'] = model.history['y'].apply(lambda x : inv_boxcox(x, fitted_lambda))
        model.plot(forecast)
        print(f"Lambda value used for Transformation: {fitted_lambda}") #if 람다가 0이면 단순 로그를 취하게 된다 2로 가까워지면 제곱근 함수
    else :
        store.rename(columns = {'amount' : 'y', 'year_month' : 'ds'}, inplace = True)
        model = Prophet(seasonality_mode = 'additive')
        model.fit(store)
        future = model.make_future_dataframe(periods = 3, freq = 'MS')
        forecast = model.predict(future)
        model.plot(forecast)
```

qq plot 에 근거, 보면은 생각 대로 cv값이 0.3 이상일 경우, log값을 취한 그래프들의 경향성이 정규성에서 더 벗어나고 있다고 생각된다.   
따라서, cv값에 근거하여 변환의 여부를 정하는 것이 옳다고 판단할 근거가 된다.  
반면에 박스 콕스의 경우, 오히려 더 낮은 cv값에서 유의미한 변환 정도를 보여 주었다.  
람다가 1일 경우 단순 수직 나열의 형태로 변환된다. (근데 여기서는 0.5만 넘어도 수직이 되는 경우가 많다.)
그리고 람다가 반대로 0이면, 단순 log 변환이랑 동일하고, 0보다 작으면(마이너스 값이면) 변환이 수평이 된다.(사용 불가능)  
박스 콕스의 특징에 따르면, 람다 값에 큰 영향을 안받기도 하고 (변환이 되면 그 이상의 유의미한 차이를 보이지 않음) + 변환이 필요 없는 경우도 많다고 한다.  
다만 이 변환은 예측 구간에 큰 영향을 준다고 한다. 편향 조정이라고 중간값과 평균 사이의 어쩌고 저쩌고 가 있는데 아무튼 이걸 역변환할때 반영할 수 있다고 한다. 아무튼 그렇다고 한다.(https://otexts.com/fppkr/transformations.html)


```python
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller as adf_test
from pmdarima.arima import auto_arima
```

# 아리마 예측 모델


```python
outcome_arima = pd.DataFrame()
for i in tqdm(resampling_data['store_id'].unique()):
    store = time_series(resampling_data[resampling_data['store_id']==i], i)
    store = pd.DataFrame(store, columns = ['amount'])
    store.reset_index(inplace = True)
    store.rename(columns = {'index': 'date'}, inplace = True)
    store  = store.set_index('date')
    store = store.fillna(store.mean())
    cv =  coefficient_variation(resampling_data, i)
    try:
        p_val = adf_test(store)
        if p_val[1]<0.05:
            d = 0
        else : 
            d =1
   
        if cv <0.3 :
            store.amount =np.log(store['amount'])
            k = (len(store) - 3)
            store_data_train = store[:k]
            store_data_test = store[k:]
        else :
            k = (len(store) - 3)
            store_data_train = store[:k]
            store_data_test = store[k:]
            
    except ValueError:
        d = 2
        if cv <0.3 :
            store.amount =np.log(store['amount'])
            k = (len(store) - 3)
            store_data_train = store[:k]
            store_data_test = store[k:]

        else :
            k = (len(store) - 3)
            store_data_train = store[:k]
            store_data_test = store[k:]

    auto_arima_model = auto_arima(store_data_train,start_p = 1, start_q = 1,
                              max_p = 2,seasonal = False,
                              d=d,trace =True, error_action = 'ignore',
                             suppress_warnings = True,
                             stepwise =False,
                             )
    
#     forcast_data = pd.DataFrame(auto_arima_model.predict(n_periods = 3), index = store_data_test.index)
#     forcast_data.columns = ['predicted']
    model = ARIMA(store_data_train.amount.values, order= auto_arima_model.order)
    model_fit = model.fit()
    forcast_data =  pd.DataFrame(model_fit.forecast(steps = 3), index = store_data_test.index)
    forcast_data.columns = ['predicted']
    print(forcast_data)
    plt.title('forcast')
    plt.plot(store, label = 'original')
    plt.plot(forcast_data, label = 'predicted')
    plt.show()
    pred_arima = pd.concat([forcast_data, store_data_test],axis = 1)
    outcome_arima = pd.concat([outcome_arima, pred_arima])
    
```

      0%|                                                                                           | 0/26 [00:00<?, ?it/s]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=812.522, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=811.105, Time=0.03 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=812.963, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=817.705, Time=0.06 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=820.634, Time=0.06 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=808.295, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=810.442, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=813.194, Time=0.06 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=816.034, Time=0.10 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.20 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=810.203, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=812.214, Time=0.09 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=814.139, Time=0.11 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=815.086, Time=0.15 sec
    
    Best model:  ARIMA(1,1,0)(0,0,0)[0] intercept
    Total fit time: 1.307 seconds
                predicted
    date                 
    2018-12-31 1852169.54
    2019-01-31 1780909.50
    2019-02-28 1808081.43



![output_47_2](https://user-images.githubusercontent.com/62747570/138008624-65e12552-fe26-42fb-898b-6cdbe1d62dd1.png)




      4%|███▏                                                                               | 1/26 [00:01<00:46,  1.86s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=247.184, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.07 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.29 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.29 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.03 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=17.568, Time=0.13 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=17.805, Time=0.19 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=17.858, Time=0.18 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=19.612, Time=0.27 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.10 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=20.207, Time=0.15 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=20.804, Time=0.18 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=20.390, Time=0.23 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 2.373 seconds
                predicted
    date                 
    2018-12-31      14.42
    2019-01-31      14.41
    2019-02-28      14.41



![output_47_5](https://user-images.githubusercontent.com/62747570/138008625-a2f2aaa9-4a89-4e6b-bac3-daaa8e3d9dea.png)




      8%|██████▍                                                                            | 2/26 [00:04<00:58,  2.45s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=238.945, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.06 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.10 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.21 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.21 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.24 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.04 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=3.872, Time=0.06 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=4.951, Time=0.16 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=4.408, Time=0.18 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.21 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=3.330, Time=0.18 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=7.836, Time=0.07 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.23 sec
    
    Best model:  ARIMA(2,0,1)(0,0,0)[0]          
    Total fit time: 2.026 seconds
                predicted
    date                 
    2018-12-31      12.44
    2019-01-31      12.49
    2019-02-28      12.52



![output_47_8](https://user-images.githubusercontent.com/62747570/138008627-3e093cc9-e35f-4de3-b205-2b0b7942f5cc.png)




     12%|█████████▌                                                                         | 3/26 [00:07<00:57,  2.49s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=794.797, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=791.979, Time=0.03 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=793.487, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=796.473, Time=0.09 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=799.039, Time=0.04 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=799.214, Time=0.05 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=791.436, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=793.503, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=794.304, Time=0.08 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.18 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=799.141, Time=0.09 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=793.562, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=795.454, Time=0.08 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=796.296, Time=0.13 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
    
    Best model:  ARIMA(1,1,0)(0,0,0)[0] intercept
    Total fit time: 1.212 seconds
                predicted
    date                 
    2018-12-31  629138.87
    2019-01-31  700305.50
    2019-02-28  674782.02



![output_47_11](https://user-images.githubusercontent.com/62747570/138008628-740af9aa-e337-4c9c-b841-e5696c843bf4.png)




     15%|████████████▊                                                                      | 4/26 [00:09<00:48,  2.19s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=758.152, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=760.172, Time=0.03 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=758.925, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=775.789, Time=0.04 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=771.194, Time=0.04 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=774.014, Time=0.06 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=760.166, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=756.100, Time=0.08 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=762.313, Time=0.04 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.25 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.23 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=759.966, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=754.968, Time=0.08 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=764.244, Time=0.09 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
    
    Best model:  ARIMA(2,1,1)(0,0,0)[0] intercept
    Total fit time: 1.346 seconds
                predicted
    date                 
    2018-12-31  404112.36
    2019-01-31  400455.46
    2019-02-28  392820.81



![output_47_14](https://user-images.githubusercontent.com/62747570/138008632-9b66b19e-aece-4de2-a063-00aa5bda7aed.png)




     19%|███████████████▉                                                                   | 5/26 [00:10<00:43,  2.06s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=243.310, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.06 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.12 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.24 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.22 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=-24.516, Time=0.10 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=-23.160, Time=0.15 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.27 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.11 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.19 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.24 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 2.368 seconds
                predicted
    date                 
    2018-12-31      13.48
    2019-01-31      13.49
    2019-02-28      13.49



![png](output_47_17.png)


     23%|███████████████████▏                                                               | 6/26 [00:13<00:47,  2.35s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-0.135, Time=0.04 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.08 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.20 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.36 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-3.664, Time=0.03 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.14 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=-4.194, Time=0.04 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.19 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.29 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.36 sec
    
    Best model:  ARIMA(2,1,0)(0,0,0)[0] intercept
    Total fit time: 3.049 seconds
                predicted
    date                 
    2018-12-31      11.27
    2019-01-31      11.33
    2019-02-28      11.31



![png](output_47_20.png)


     27%|██████████████████████▎                                                            | 7/26 [00:17<00:52,  2.75s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=244.193, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.10 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.11 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.16 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.24 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.26 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.07 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=-8.333, Time=0.10 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=-6.373, Time=0.13 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=-4.694, Time=0.19 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.22 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.11 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.15 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.27 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 2.344 seconds
                predicted
    date                 
    2018-12-31      13.67
    2019-01-31      13.67
    2019-02-28      13.68



![png](output_47_23.png)


     31%|█████████████████████████▌                                                         | 8/26 [00:20<00:50,  2.80s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=249.847, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.09 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.19 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.23 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.30 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.03 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=18.209, Time=0.11 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=19.995, Time=0.14 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=21.995, Time=0.15 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=23.742, Time=0.20 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.08 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=21.727, Time=0.17 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.13 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.09 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 1.991 seconds
                predicted
    date                 
    2018-12-31      15.05
    2019-01-31      15.05
    2019-02-28      15.05



![png](output_47_26.png)


     35%|████████████████████████████▋                                                      | 9/26 [00:22<00:46,  2.72s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=888.819, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=890.685, Time=0.02 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=887.303, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=894.810, Time=0.04 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=909.726, Time=0.06 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=890.270, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=892.017, Time=0.04 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=889.186, Time=0.05 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=889.994, Time=0.02 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=892.024, Time=0.04 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=890.484, Time=0.07 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
    
    Best model:  ARIMA(0,1,2)(0,0,0)[0] intercept
    Total fit time: 1.582 seconds
                predicted
    date                 
    2018-12-31 1072096.70
    2019-01-31  853602.42
    2019-02-28  853602.42



![png](output_47_29.png)


     38%|███████████████████████████████▌                                                  | 10/26 [00:24<00:40,  2.52s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=784.772, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=780.628, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=783.517, Time=0.04 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=784.810, Time=0.05 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=793.876, Time=0.08 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=793.044, Time=0.08 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=782.963, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=782.001, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=779.598, Time=0.10 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=782.998, Time=0.14 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=788.585, Time=0.18 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=781.842, Time=0.03 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=778.438, Time=0.06 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=778.884, Time=0.12 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=780.353, Time=0.20 secBest model:  ARIMA(2,1,1)(0,0,0)[0] interceptTotal fit time: 1.200 seconds            predicteddate                 2018-12-31  614942.412019-01-31  373678.972019-02-28  577071.56



![png](output_47_32.png)


     42%|██████████████████████████████████▋                                               | 11/26 [00:26<00:34,  2.27s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=766.708, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=760.110, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=761.641, Time=0.04 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=763.104, Time=0.04 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=770.507, Time=0.08 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=760.654, Time=0.08 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=758.337, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=759.369, Time=0.03 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=761.793, Time=0.06 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=760.576, Time=0.18 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.13 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=757.633, Time=0.02 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=756.745, Time=0.07 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=760.083, Time=0.13 secBest model:  ARIMA(2,1,2)(0,0,0)[0] interceptTotal fit time: 1.184 seconds            predicteddate                 2018-12-31  441359.322019-01-31  293184.062019-02-28  292308.04



![png](output_47_35.png)


     46%|█████████████████████████████████████▊                                            | 12/26 [00:28<00:29,  2.11s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=762.114, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=764.082, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=764.850, Time=0.05 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=765.349, Time=0.08 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=768.650, Time=0.05 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.38 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=764.658, Time=0.03 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=765.864, Time=0.07 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=766.805, Time=0.08 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=766.517, Time=0.14 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=767.015, Time=0.23 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=765.976, Time=0.04 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=767.664, Time=0.12 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=767.870, Time=0.20 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 1.905 seconds            predicteddate                 2018-12-31 2384857.142019-01-31 2384857.142019-02-28 2384857.14



![png](output_47_38.png)


     50%|█████████████████████████████████████████                                         | 13/26 [00:31<00:30,  2.33s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=757.276, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=747.657, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=746.164, Time=0.04 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=749.444, Time=0.08 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=751.465, Time=0.12 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=750.757, Time=0.13 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=745.516, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=746.776, Time=0.05 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=747.317, Time=0.14 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=749.497, Time=0.13 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=750.452, Time=0.14 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=746.727, Time=0.04 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=748.735, Time=0.06 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=749.191, Time=0.44 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=751.294, Time=0.38 secBest model:  ARIMA(1,1,0)(0,0,0)[0] interceptTotal fit time: 1.842 seconds            predicteddate                 2018-12-31  381072.852019-01-31  338976.542019-02-28  365549.87



![png](output_47_41.png)


     54%|████████████████████████████████████████████▏                                     | 14/26 [00:33<00:28,  2.41s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=781.220, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=794.079, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=791.589, Time=0.06 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=792.370, Time=0.03 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=796.919, Time=0.06 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.18 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.47 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=793.292, Time=0.05 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=793.609, Time=0.09 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=796.720, Time=0.12 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.45 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 2.601 seconds            predicteddate                 2018-12-31  858214.292019-01-31  858214.292019-02-28  858214.29



![png](output_47_44.png)


     58%|███████████████████████████████████████████████▎                                  | 15/26 [00:37<00:29,  2.68s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=666.261, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=668.403, Time=0.04 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=670.422, Time=0.05 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=671.763, Time=0.06 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=670.039, Time=0.30 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.35 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=668.234, Time=0.03 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=670.251, Time=0.07 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=672.197, Time=0.07 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=673.662, Time=0.08 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=671.793, Time=0.45 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=670.263, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=672.274, Time=0.08 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=672.241, Time=0.14 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=674.725, Time=0.12 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 1.899 seconds
                predicted
    date                 
    2018-12-31  747214.29
    2019-01-31  747214.29
    2019-02-28  747214.29



![png](output_47_47.png)


     62%|██████████████████████████████████████████████████▍                               | 16/26 [00:39<00:27,  2.71s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=761.979, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=744.213, Time=0.03 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=743.504, Time=0.04 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=742.267, Time=0.06 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=740.085, Time=0.08 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=741.848, Time=0.06 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=723.016, Time=0.02 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=722.914, Time=0.06 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=inf, Time=0.30 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.31 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.35 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=720.279, Time=0.04 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=724.742, Time=0.08 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.31 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.22 secBest model:  ARIMA(2,0,0)(0,0,0)[0]          Total fit time: 1.980 seconds            predicteddate                 2018-12-31  409443.242019-01-31  403828.862019-02-28  437745.61



![png](output_47_50.png)


     65%|█████████████████████████████████████████████████████▌                            | 17/26 [00:42<00:24,  2.69s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=763.408, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=752.891, Time=0.20 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=754.403, Time=0.04 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=756.811, Time=0.05 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=760.190, Time=0.06 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=760.944, Time=0.05 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=734.590, Time=0.02 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=730.570, Time=0.04 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=732.082, Time=0.08 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=733.498, Time=0.28 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=736.731, Time=0.13 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=734.911, Time=0.02 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=731.954, Time=0.10 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=731.991, Time=0.08 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=733.964, Time=0.15 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 1.328 seconds            predicteddate                 2018-12-31  561979.072019-01-31  450373.322019-02-28  523340.79



![png](output_47_53.png)


     69%|████████████████████████████████████████████████████████▊                         | 18/26 [00:44<00:19,  2.49s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=633.170, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=635.137, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=638.215, Time=0.03 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=640.972, Time=0.03 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.34 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.39 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=635.125, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=636.572, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=640.325, Time=0.04 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.20 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.25 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=635.344, Time=0.02 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=637.319, Time=0.04 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 1.968 seconds            predicteddate                 2018-12-31  290085.712019-01-31  290085.712019-02-28  290085.71



![png](output_47_56.png)


     73%|███████████████████████████████████████████████████████████▉                      | 19/26 [00:47<00:17,  2.50s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=679.664, Time=0.02 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=680.748, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=683.213, Time=0.02 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=681.798, Time=0.06 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=687.346, Time=0.05 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=681.071, Time=0.03 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=682.656, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=682.946, Time=0.16 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=683.720, Time=0.12 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=682.904, Time=0.05 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=684.709, Time=0.05 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=684.480, Time=0.23 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=684.777, Time=0.24 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 1.743 seconds            predicteddate                 2018-12-31 1260557.142019-01-31 1260557.142019-02-28 1260557.14



![png](output_47_59.png)


     77%|███████████████████████████████████████████████████████████████                   | 20/26 [00:49<00:14,  2.48s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=16.305, Time=0.05 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=14.597, Time=0.06 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=16.360, Time=0.07 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=18.261, Time=0.08 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.44 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=15.129, Time=0.03 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=16.561, Time=0.08 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=18.209, Time=0.10 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=19.189, Time=0.14 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=15.869, Time=0.05 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=16.909, Time=0.17 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.31 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 secBest model:  ARIMA(0,1,1)(0,0,0)[0] interceptTotal fit time: 2.656 seconds            predicteddate                 2018-12-31      14.122019-01-31      14.122019-02-28      14.12



![png](output_47_62.png)


     81%|██████████████████████████████████████████████████████████████████▏               | 21/26 [00:52<00:13,  2.73s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=558.819, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=559.242, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=561.908, Time=0.02 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=572.902, Time=0.04 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=570.125, Time=0.12 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=559.836, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=560.694, Time=0.07 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=562.253, Time=0.13 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=560.595, Time=0.04 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=562.045, Time=0.10 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=564.956, Time=0.08 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.25 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 1.765 seconds            predicteddate                 2018-12-31  281714.292019-01-31  281714.292019-02-28  281714.29



![png](output_47_65.png)


     85%|█████████████████████████████████████████████████████████████████████▍            | 22/26 [00:54<00:10,  2.56s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=580.150, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=578.693, Time=0.01 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=580.602, Time=0.02 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=584.555, Time=0.03 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=588.234, Time=0.03 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=582.429, Time=0.06 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=574.914, Time=0.01 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=569.415, Time=0.06 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=571.657, Time=0.06 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=576.926, Time=0.07 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=582.218, Time=0.08 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=573.874, Time=0.02 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=570.922, Time=0.06 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=573.368, Time=0.09 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=575.563, Time=0.28 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 0.908 seconds            predicteddate                 2018-12-31   90185.472019-01-31   93525.762019-02-28   94467.71



![png](output_47_68.png)


     88%|████████████████████████████████████████████████████████████████████████▌         | 23/26 [00:56<00:06,  2.24s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=12.982, Time=0.02 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=14.337, Time=0.06 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.41 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=nan, Time=0.21 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=14.179, Time=0.06 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=16.179, Time=0.08 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.25 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.39 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.42 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=16.178, Time=0.08 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=17.841, Time=0.21 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.31 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.56 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 3.552 seconds            predicteddate                 2018-12-31      12.472019-01-31      12.472019-02-28      12.47



![png](output_47_71.png)


     92%|███████████████████████████████████████████████████████████████████████████▋      | 24/26 [01:00<00:05,  2.82s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=32.776, Time=0.04 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.14 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.51 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=nan, Time=0.11 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=26.943, Time=0.03 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.22 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.36 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.35 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=27.134, Time=0.40 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=28.815, Time=0.04 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.45 secBest model:  ARIMA(1,1,0)(0,0,0)[0] interceptTotal fit time: 3.818 seconds            predicteddate                 2018-12-31      13.192019-01-31      13.252019-02-28      13.22



![png](output_47_74.png)


     96%|██████████████████████████████████████████████████████████████████████████████▊   | 25/26 [01:04<00:03,  3.28s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=155.326, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.10 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.09 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.25 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.21 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=nan, Time=0.09 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.05 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=9.015, Time=0.11 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=inf, Time=0.20 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.19 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.20 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.08 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.16 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.24 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.26 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 2.241 seconds            predicteddate                 2018-12-31      13.602019-01-31      13.692019-02-28      13.67



![png](output_47_77.png)


    100%|██████████████████████████████████████████████████████████████████████████████████| 26/26 [01:07<00:00,  2.61s/it]



```python
print(outcome_arima)print(outcome_arima.isna().sum())outcome_arima = outcome_arima.dropna()
```

                predicted     amountdate                            2018-12-31 1961539.59  673285.712019-01-31 1900760.68 2635400.002019-02-28 2002084.48 1877428.572018-12-31      14.49      14.282019-01-31      14.49      14.32...               ...        ...2019-01-31      13.26      13.172019-02-28      13.24      13.392018-12-31      13.64      13.692019-01-31      13.64      13.482019-02-28      13.64      13.58[78 rows x 2 columns]predicted    0amount       0dtype: int64



```python
adding = 0for i in range(3,len(outcome_arima),3):    store = outcome_arima.iloc[i-3:i]    rmse = mean_squared_error(store.predicted, store.amount)**0.5    adding += rmse    print("%d번째 상점의 MSE의 값은 %.04f입니다" %((i-1)/3+1, rmse))adding = adding / len(outcome_arima)*3print("전체 MSE 값의 평균은 %.04f입니다" %(adding))
```

    1번째 상점의 MSE의 값은 738277354903.9401입니다2번째 상점의 MSE의 값은 0.0417입니다3번째 상점의 MSE의 값은 0.0062입니다4번째 상점의 MSE의 값은 55436932232.2871입니다5번째 상점의 MSE의 값은 39392817286.2170입니다6번째 상점의 MSE의 값은 0.0478입니다7번째 상점의 MSE의 값은 0.1374입니다8번째 상점의 MSE의 값은 0.0080입니다9번째 상점의 MSE의 값은 0.0403입니다10번째 상점의 MSE의 값은 134309936384.0209입니다11번째 상점의 MSE의 값은 56039445100.3552입니다12번째 상점의 MSE의 값은 20613031067.4096입니다13번째 상점의 MSE의 값은 107755496778.2711입니다14번째 상점의 MSE의 값은 51039979485.3982입니다15번째 상점의 MSE의 값은 11560760574.0048입니다16번째 상점의 MSE의 값은 160973362934.2268입니다17번째 상점의 MSE의 값은 14766857277.9139입니다18번째 상점의 MSE의 값은 26888749297.6553입니다19번째 상점의 MSE의 값은 9915965899.2347입니다20번째 상점의 MSE의 값은 25727880041.3361입니다21번째 상점의 MSE의 값은 0.0050입니다22번째 상점의 MSE의 값은 3094244897.9592입니다23번째 상점의 MSE의 값은 12225167018.7456입니다24번째 상점의 MSE의 값은 0.2106입니다25번째 상점의 MSE의 값은 0.0112입니다전체 MSE 값의 평균은 56462230045.3648입니다


# 아리마 모델로 3개월 추가 예측


```python
prediction_arima = pd.DataFrame()for i in tqdm(resampling_data['store_id'].unique()):    store = time_series(resampling_data[resampling_data['store_id']==i], i)    store = pd.DataFrame(store, columns = ['amount'])    store.reset_index(inplace = True)    store.rename(columns = {'index': 'date'}, inplace = True)    store  = store.set_index('date')    store = store.fillna(store.mean())    cv =  coefficient_variation(resampling_data, i)    try:        p_val = adf_test(store)        if p_val[1]<0.05:            d = 0        else :             d =1           if cv <0.3 :            store.amount =np.log(store['amount'])            store_data_train = store        else :            store_data_train = store                except ValueError:        d = 2        if cv <0.3 :            store.amount =np.log(store['amount'])            store_data_train = store        else :            store_data_train = store    auto_arima_model = auto_arima(store_data_train,start_p = 1, start_q = 1,                              max_p = 2,seasonal = False,                              d=d,trace =True, error_action = 'ignore',                             suppress_warnings = True,                             stepwise =False,                             )    #     forcast_data = pd.DataFrame(auto_arima_model.predict(n_periods = 3), index = store_data_test.index)#     forcast_data.columns = ['predicted']    model = ARIMA(store_data_train.amount.values, order= auto_arima_model.order)    model_fit = model.fit()    time = ['3개월 짜리 인덱스 만들어 주세요']    forcast_data =  pd.DataFrame(model_fit.forecast(steps = 3), index = store_data_test.index#여기다 새인덱스 넣어주세요)    forcast_data.columns = ['predicted']    print(forcast_data)    plt.title('forcast')    plt.plot(store, label = 'original')    plt.plot(forcast_data, label = 'predicted')    plt.show()    prediction_arima = pd.concat([prediction_arima, forcast_data])    
```

      0%|                                                                                           | 0/26 [00:00<?, ?it/s] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=932.892, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=927.676, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=930.522, Time=0.04 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=924.339, Time=0.11 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=928.915, Time=0.18 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=929.883, Time=0.41 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=924.172, Time=0.06 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=923.044, Time=0.08 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.17 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=922.608, Time=0.11 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=927.433, Time=0.30 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=912.604, Time=0.05 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=912.877, Time=0.12 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=912.506, Time=0.15 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=915.981, Time=0.22 secBest model:  ARIMA(2,1,2)(0,0,0)[0] interceptTotal fit time: 2.057 seconds            predicteddate                 2018-12-31  995827.182019-01-31 2578300.422019-02-28 1519432.86



![png](output_51_2.png)


      4%|███▏                                                                               | 1/26 [00:02<01:12,  2.92s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=271.655, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.07 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.09 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.22 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.30 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.31 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.04 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=15.586, Time=0.06 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=15.005, Time=0.24 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.27 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=16.891, Time=0.23 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.13 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=12.980, Time=0.22 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=14.999, Time=0.26 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.38 secBest model:  ARIMA(2,0,1)(0,0,0)[0]          Total fit time: 2.860 seconds            predicteddate                 2018-12-31      14.282019-01-31      14.312019-02-28      14.35



![png](output_51_5.png)


      8%|██████▍                                                                            | 2/26 [00:06<01:22,  3.45s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=262.579, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.05 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.11 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.21 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.36 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.32 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.05 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=-0.283, Time=0.07 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=0.761, Time=0.20 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=0.297, Time=0.20 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=2.126, Time=0.35 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.08 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=-0.863, Time=0.23 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=1.880, Time=0.10 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.30 secBest model:  ARIMA(2,0,1)(0,0,0)[0]          Total fit time: 2.646 seconds            predicteddate                 2018-12-31      12.472019-01-31      12.502019-02-28      12.52



![png](output_51_8.png)


     12%|█████████▌                                                                         | 3/26 [00:09<01:17,  3.35s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=877.639, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=876.087, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=877.646, Time=0.03 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=880.452, Time=0.04 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=883.716, Time=0.05 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=884.426, Time=0.12 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=876.942, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=877.716, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=879.646, Time=0.08 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=879.579, Time=0.17 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=882.279, Time=0.12 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=878.035, Time=0.03 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=879.541, Time=0.05 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.21 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 secBest model:  ARIMA(0,1,1)(0,0,0)[0] interceptTotal fit time: 1.307 seconds            predicteddate                 2018-12-31  586247.312019-01-31  586247.312019-02-28  586247.31



![png](output_51_11.png)


     15%|████████████▊                                                                      | 4/26 [00:11<01:00,  2.77s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=834.627, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=836.667, Time=0.04 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=834.371, Time=0.05 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=837.797, Time=0.12 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=835.296, Time=0.07 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=841.198, Time=0.07 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=836.655, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=834.871, Time=0.12 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=835.333, Time=0.11 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=837.795, Time=0.14 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=836.383, Time=0.02 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=834.086, Time=0.06 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=837.084, Time=0.07 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=835.707, Time=0.22 secBest model:  ARIMA(2,1,1)(0,0,0)[0] interceptTotal fit time: 1.484 seconds            predicteddate                 2018-12-31  260709.422019-01-31  317102.582019-02-28  336773.94



![png](output_51_14.png)


     19%|███████████████▉                                                                   | 5/26 [00:13<00:52,  2.50s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=267.407, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.07 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.07 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.17 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.23 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.25 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.04 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=-21.326, Time=0.11 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=inf, Time=0.24 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.15 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.22 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.09 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.21 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.16 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.27 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 2.304 seconds            predicteddate                 2018-12-31      13.452019-01-31      13.522019-02-28      13.47



![png](output_51_17.png)


     23%|███████████████████▏                                                               | 6/26 [00:16<00:52,  2.64s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=9.181, Time=0.03 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.11 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.17 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.07 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.22 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=1.625, Time=0.32 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=3.209, Time=0.04 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.18 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.29 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.23 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=1.423, Time=0.04 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.34 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.22 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 secBest model:  ARIMA(2,1,0)(0,0,0)[0] interceptTotal fit time: 2.951 seconds            predicteddate                 2018-12-31      11.462019-01-31      11.572019-02-28      11.43



![png](output_51_20.png)


     27%|██████████████████████▎                                                            | 7/26 [00:20<00:54,  2.89s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=268.422, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.06 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.09 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.21 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.23 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.28 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.08 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=-13.190, Time=0.15 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=-11.194, Time=0.14 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=-9.412, Time=0.18 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.31 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.16 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.17 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.23 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.23 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 2.536 seconds            predicteddate                 2018-12-31      13.702019-01-31      13.702019-02-28      13.70



![png](output_51_23.png)


     31%|█████████████████████████▌                                                         | 8/26 [00:23<00:53,  2.98s/it] ARIMA(0,0,0)(0,0,0)[0]             : AIC=274.729, Time=0.01 sec ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.10 sec ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.11 sec ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.23 sec ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.23 sec ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.31 sec ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.06 sec ARIMA(1,0,1)(0,0,0)[0]             : AIC=17.219, Time=0.19 sec ARIMA(1,0,2)(0,0,0)[0]             : AIC=19.124, Time=0.16 sec ARIMA(1,0,3)(0,0,0)[0]             : AIC=21.045, Time=0.15 sec ARIMA(1,0,4)(0,0,0)[0]             : AIC=22.903, Time=0.31 sec ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.11 sec ARIMA(2,0,1)(0,0,0)[0]             : AIC=21.521, Time=0.21 sec ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.14 sec ARIMA(2,0,3)(0,0,0)[0]             : AIC=23.117, Time=0.27 secBest model:  ARIMA(1,0,1)(0,0,0)[0]          Total fit time: 2.608 seconds            predicteddate                 2018-12-31      15.112019-01-31      15.102019-02-28      15.09



![png](output_51_26.png)


     35%|████████████████████████████▋                                                      | 9/26 [00:26<00:51,  3.03s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=977.231, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=979.022, Time=0.02 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=975.073, Time=0.03 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=984.036, Time=0.04 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=1003.730, Time=0.06 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=978.617, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=980.297, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=976.960, Time=0.08 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.42 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=978.229, Time=0.03 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=980.240, Time=0.06 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=977.970, Time=0.11 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 secBest model:  ARIMA(0,1,2)(0,0,0)[0] interceptTotal fit time: 1.856 seconds            predicteddate                 2018-12-31 1319071.862019-01-31 1301903.812019-02-28 1301903.81



![png](output_51_29.png)


     38%|███████████████████████████████▌                                                  | 10/26 [00:28<00:45,  2.85s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=870.932, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=869.057, Time=0.04 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=864.167, Time=0.04 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=866.803, Time=0.16 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=877.688, Time=0.22 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=871.523, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=869.291, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=862.269, Time=0.05 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=866.898, Time=0.07 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=872.299, Time=0.17 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=866.667, Time=0.02 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=863.144, Time=0.07 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=863.144, Time=0.14 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=864.822, Time=0.11 secBest model:  ARIMA(1,1,2)(0,0,0)[0] interceptTotal fit time: 1.542 seconds            predicteddate                 2018-12-31  346588.392019-01-31  456844.502019-02-28  427548.04



![png](output_51_32.png)


     42%|██████████████████████████████████▋                                               | 11/26 [00:30<00:38,  2.60s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=844.861, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=836.188, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=840.609, Time=0.05 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=845.732, Time=0.06 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=842.848, Time=0.13 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=836.271, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=836.989, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=836.743, Time=0.07 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=841.856, Time=0.13 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=836.052, Time=0.02 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=835.400, Time=0.07 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=838.221, Time=0.12 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=839.855, Time=0.32 secBest model:  ARIMA(2,1,1)(0,0,0)[0] interceptTotal fit time: 1.583 seconds            predicteddate                 2018-12-31  468384.932019-01-31  453706.302019-02-28  488788.40



![png](output_51_35.png)


     46%|█████████████████████████████████████▊                                            | 12/26 [00:33<00:35,  2.53s/it] ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=846.285, Time=0.01 sec ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=848.671, Time=0.03 sec ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=848.478, Time=0.08 sec ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=849.184, Time=0.05 sec ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=852.510, Time=0.04 sec ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=849.961, Time=0.17 sec ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=849.222, Time=0.02 sec ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=850.092, Time=0.04 sec ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=850.345, Time=0.08 sec ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=850.684, Time=0.15 sec ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=850.933, Time=0.13 sec ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=850.011, Time=0.03 sec ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=851.620, Time=0.07 sec ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=849.653, Time=0.12 sec ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=851.628, Time=0.24 secBest model:  ARIMA(0,1,0)(0,0,0)[0] interceptTotal fit time: 1.300 seconds            predicteddate                 2018-12-31 2459000.002019-01-31 2459000.002019-02-28 2459000.00



![png](output_51_38.png)


     50%|█████████████████████████████████████████                                         | 13/26 [00:35<00:30,  2.32s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=838.321, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=829.561, Time=0.04 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=827.459, Time=0.04 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=829.879, Time=0.04 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=831.110, Time=0.07 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.27 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=826.349, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=828.023, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=828.019, Time=0.10 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=830.229, Time=0.10 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=830.846, Time=0.12 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=827.871, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=827.988, Time=0.13 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=829.870, Time=0.10 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=831.699, Time=0.11 sec
    
    Best model:  ARIMA(1,1,0)(0,0,0)[0] intercept
    Total fit time: 1.219 seconds
                predicted
    date                 
    2018-12-31  106005.25
    2019-01-31   76613.56
    2019-02-28   94598.21



![png](output_51_41.png)


     54%|████████████████████████████████████████████▏                                     | 14/26 [00:36<00:25,  2.14s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=867.817, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=881.803, Time=0.03 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=889.343, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=878.756, Time=0.05 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=961.212, Time=0.06 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=879.922, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=884.702, Time=0.04 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.12 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=880.712, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=880.870, Time=0.06 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=883.615, Time=0.07 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.33 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 1.640 seconds
                predicted
    date                 
    2018-12-31  749485.71
    2019-01-31  749485.71
    2019-02-28  749485.71



![png](output_51_44.png)


     58%|███████████████████████████████████████████████▎                                  | 15/26 [00:39<00:23,  2.16s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=764.790, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=763.121, Time=0.02 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=768.638, Time=0.08 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=763.206, Time=0.05 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=763.971, Time=0.04 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=776.468, Time=0.09 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=765.044, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=766.699, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=763.057, Time=0.05 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=763.238, Time=0.08 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=764.775, Time=0.11 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=759.285, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=761.400, Time=0.04 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=763.282, Time=0.05 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=764.155, Time=0.22 sec
    
    Best model:  ARIMA(2,1,0)(0,0,0)[0] intercept
    Total fit time: 0.937 seconds
                predicted
    date                 
    2018-12-31  487373.30
    2019-01-31  406824.80
    2019-02-28  478416.09



![png](output_51_47.png)


     62%|██████████████████████████████████████████████████▍                               | 16/26 [00:40<00:19,  1.97s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=847.582, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=827.476, Time=0.03 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=826.833, Time=0.03 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=824.293, Time=0.03 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=822.797, Time=0.05 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=822.409, Time=0.07 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=803.398, Time=0.02 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=802.469, Time=0.05 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=805.565, Time=0.06 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=inf, Time=0.34 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.20 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=800.852, Time=0.05 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=804.797, Time=0.07 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.16 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.23 sec
    
    Best model:  ARIMA(2,0,0)(0,0,0)[0]          
    Total fit time: 1.420 seconds
                predicted
    date                 
    2018-12-31  404688.02
    2019-01-31  423959.79
    2019-02-28  437972.83



![png](output_51_50.png)


     65%|█████████████████████████████████████████████████████▌                            | 17/26 [00:42<00:17,  1.98s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=850.678, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=838.112, Time=0.03 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=837.929, Time=0.03 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=839.734, Time=0.04 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=845.864, Time=0.04 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=843.695, Time=0.05 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=816.201, Time=0.02 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=811.910, Time=0.05 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=813.419, Time=0.06 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=814.315, Time=0.14 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=817.319, Time=0.21 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=816.351, Time=0.03 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=813.024, Time=0.10 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=812.687, Time=0.13 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=814.496, Time=0.15 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 1.102 seconds
                predicted
    date                 
    2018-12-31  490517.73
    2019-01-31  494491.23
    2019-02-28  491962.19



![png](output_51_53.png)


     69%|████████████████████████████████████████████████████████▊                         | 18/26 [00:44<00:15,  1.91s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=713.632, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=715.743, Time=0.03 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=718.220, Time=0.04 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=719.780, Time=0.05 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=720.702, Time=0.08 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=715.646, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=717.702, Time=0.05 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=720.198, Time=0.05 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=717.408, Time=0.35 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=716.569, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=718.543, Time=0.06 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=724.584, Time=0.12 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 1.748 seconds
                predicted
    date                 
    2018-12-31  361028.57
    2019-01-31  361028.57
    2019-02-28  361028.57



![png](output_51_56.png)


     73%|███████████████████████████████████████████████████████████▉                      | 19/26 [00:46<00:14,  2.03s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=761.397, Time=0.01 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=762.444, Time=0.02 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=764.866, Time=0.03 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.09 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=770.661, Time=0.07 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.36 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=762.795, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=764.335, Time=0.05 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=764.142, Time=0.17 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=764.656, Time=0.12 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=764.635, Time=0.03 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=766.431, Time=0.06 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=765.647, Time=0.18 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=765.611, Time=0.38 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 1.953 seconds
                predicted
    date                 
    2018-12-31 1298328.57
    2019-01-31 1298328.57
    2019-02-28 1298328.57



![png](output_51_59.png)


     77%|███████████████████████████████████████████████████████████████                   | 20/26 [00:49<00:13,  2.17s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=14.935, Time=0.04 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=12.808, Time=0.05 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=14.663, Time=0.08 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=16.551, Time=0.08 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.35 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=13.424, Time=0.04 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=14.786, Time=0.06 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=16.339, Time=0.13 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=17.359, Time=0.13 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=14.075, Time=0.05 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=15.030, Time=0.13 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.23 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.34 sec
    
    Best model:  ARIMA(0,1,1)(0,0,0)[0] intercept
    Total fit time: 2.413 seconds
                predicted
    date                 
    2018-12-31      14.17
    2019-01-31      14.17
    2019-02-28      14.17



![png](output_51_62.png)


     81%|██████████████████████████████████████████████████████████████████▏               | 21/26 [00:52<00:12,  2.45s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=636.178, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=636.144, Time=0.04 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=639.293, Time=0.06 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=656.552, Time=0.07 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.34 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=639.496, Time=0.08 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=636.816, Time=0.02 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=637.806, Time=0.07 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=640.022, Time=0.14 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.27 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=678.095, Time=0.15 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=637.869, Time=0.06 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=639.307, Time=0.11 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=642.003, Time=0.24 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.63 sec
    
    Best model:  ARIMA(0,1,1)(0,0,0)[0] intercept
    Total fit time: 2.348 seconds
                predicted
    date                 
    2018-12-31  227881.66
    2019-01-31  227881.66
    2019-02-28  227881.66



![png](output_51_65.png)


     85%|█████████████████████████████████████████████████████████████████████▍            | 22/26 [00:55<00:10,  2.66s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=662.132, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=659.691, Time=0.25 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=661.172, Time=0.04 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=665.541, Time=0.03 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=666.900, Time=0.07 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=660.951, Time=0.06 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=654.708, Time=0.02 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=649.251, Time=0.05 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=650.328, Time=0.08 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=650.576, Time=0.14 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=650.795, Time=0.12 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=654.161, Time=0.03 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=651.172, Time=0.06 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=651.861, Time=0.13 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=651.125, Time=0.28 sec
    
    Best model:  ARIMA(1,0,1)(0,0,0)[0]          
    Total fit time: 1.398 seconds
                predicted
    date                 
    2018-12-31   78388.38
    2019-01-31   93441.34
    2019-02-28   98610.65



![png](output_51_68.png)


     88%|████████████████████████████████████████████████████████████████████████▌         | 23/26 [00:57<00:07,  2.48s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=15.121, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=16.791, Time=0.06 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=17.919, Time=0.12 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.40 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.43 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=16.754, Time=0.04 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=18.752, Time=0.09 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.31 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=18.741, Time=0.06 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=20.447, Time=0.21 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.60 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.57 sec
    
    Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
    Total fit time: 3.815 seconds
                predicted
    date                 
    2018-12-31      11.93
    2019-01-31      11.93
    2019-02-28      11.93



![png](output_51_71.png)


     92%|███████████████████████████████████████████████████████████████████████████▋      | 24/26 [01:01<00:06,  3.07s/it]
    
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=34.370, Time=0.02 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.16 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.26 sec
     ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec
     ARIMA(0,1,4)(0,0,0)[0] intercept   : AIC=inf, Time=0.43 sec
     ARIMA(0,1,5)(0,0,0)[0] intercept   : AIC=inf, Time=0.37 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=27.323, Time=0.06 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.17 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.31 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.35 sec
     ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=25.929, Time=0.45 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=29.161, Time=0.06 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.30 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.35 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.39 sec
    
    Best model:  ARIMA(1,1,4)(0,0,0)[0] intercept
    Total fit time: 4.083 seconds
                predicted
    date                 
    2018-12-31      13.24
    2019-01-31      13.35
    2019-02-28      13.20



![png](output_51_74.png)


     96%|██████████████████████████████████████████████████████████████████████████████▊   | 25/26 [01:07<00:03,  3.71s/it]
    
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=179.492, Time=0.01 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(0,0,2)(0,0,0)[0]             : AIC=inf, Time=0.10 sec
     ARIMA(0,0,3)(0,0,0)[0]             : AIC=inf, Time=0.21 sec
     ARIMA(0,0,4)(0,0,0)[0]             : AIC=inf, Time=0.21 sec
     ARIMA(0,0,5)(0,0,0)[0]             : AIC=inf, Time=0.30 sec
     ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.05 sec
     ARIMA(1,0,1)(0,0,0)[0]             : AIC=5.821, Time=0.08 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=inf, Time=0.22 sec
     ARIMA(1,0,3)(0,0,0)[0]             : AIC=5.672, Time=0.23 sec
     ARIMA(1,0,4)(0,0,0)[0]             : AIC=inf, Time=0.25 sec
     ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.09 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.30 sec
     ARIMA(2,0,2)(0,0,0)[0]             : AIC=inf, Time=0.31 sec
     ARIMA(2,0,3)(0,0,0)[0]             : AIC=inf, Time=0.34 sec
    
    Best model:  ARIMA(1,0,3)(0,0,0)[0]          
    Total fit time: 2.792 seconds
                predicted
    date                 
    2018-12-31      13.77
    2019-01-31      13.61
    2019-02-28      13.68



![png](output_51_77.png)


    100%|██████████████████████████████████████████████████████████████████████████████████| 26/26 [01:10<00:00,  2.72s/it]



```python
#QQ plot 그래프 그리는 함수
from scipy.stats import probplot
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import stats 
for i in tqdm(resampling_data.store_id.unique()):
    k = resampling_data[resampling_data.store_id == i]
    cv=coefficient_variation(resampling_data,i)

    x1 = k.amount
    x2 = np.log(k.amount)
    x3, lambda_cal = stats.boxcox(k.amount)

    f, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes[0][0].boxplot(x1)
    probplot(x1, plot=axes[1][0]) #scipy.stats.probplot
    axes[0][1].boxplot(x2)
    probplot(x2, plot=axes[1][1]) #scipy.stats.probplot
    axes[0][2].boxplot(x3)
    probplot(x3, plot=axes[1][2]) #scipy.stats.probplot
    print(f"cv 값 : {cv}")
    print(f"람다 값 : {lambda_cal}")
    plt.axis("equal")
    plt.xlim(-2,2)
    plt.show()
```

      0%|                                                                                         | 0/1948 [00:00<?, ?it/s]
    
    cv 값 : 0.617347437908329
    람다 값 : 0.564454679867715



![output_52_2](https://user-images.githubusercontent.com/62747570/138008767-501c633e-af77-43f7-b308-c311ab703685.png)




      0%|                                                                                 | 1/1948 [00:01<35:30,  1.09s/it]
    
    cv 값 : 0.27591441069934197
    람다 값 : 0.3681286392572889



![output_52_5](https://user-images.githubusercontent.com/62747570/138008770-c54e3fe1-7c5b-4a14-9dc5-2fc3b4abafe9.png)




      0%|                                                                                 | 2/1948 [00:02<41:16,  1.27s/it]
    
    cv 값 : 0.42445611551526397
    람다 값 : 0.6789882111632558



![output_52_8](https://user-images.githubusercontent.com/62747570/138008782-b96e8076-627a-416d-b73a-b4056977651c.png)




      0%|                                                                                 | 3/1948 [00:03<38:32,  1.19s/it]
    
    cv 값 : 0.22202541203722728
    람다 값 : 0.2860733718883887



![output_52_11](https://user-images.githubusercontent.com/62747570/138008784-6e2fb0e6-fa54-4eaf-9495-11ce2cf2af29.png)




      0%|▏                                                                                | 4/1948 [00:04<36:57,  1.14s/it]
    
    cv 값 : 0.16771570440987096
    람다 값 : 0.09097150987213742



![output_52_14](https://user-images.githubusercontent.com/62747570/138008785-100357fd-00b8-4425-ae33-65213eab337f.png)




      0%|▏                                                                                | 5/1948 [00:05<37:32,  1.16s/it]
    
    cv 값 : 0.5578141056265072
    람다 값 : -0.631967772039019



![png](output_52_17.png)


      0%|▏                                                                                | 6/1948 [00:06<36:22,  1.12s/it]cv 값 : 0.45802517617212796람다 값 : 0.532918019569187



![png](output_52_20.png)


      0%|▎                                                                                | 7/1948 [00:07<35:14,  1.09s/it]cv 값 : 0.166152483908934람다 값 : 1.2467978448442512



![png](output_52_23.png)


      0%|▎                                                                                | 8/1948 [00:08<34:20,  1.06s/it]cv 값 : 0.2965631155495316람다 값 : 1.1992361578864452



![png](output_52_26.png)


      0%|▎                                                                                | 9/1948 [00:09<33:57,  1.05s/it]cv 값 : 0.22153466821086654람다 값 : 1.0887067563802286



![png](output_52_29.png)


      1%|▍                                                                               | 10/1948 [00:10<33:58,  1.05s/it]cv 값 : 0.18177797311749058람다 값 : 0.4523829839082143



![png](output_52_32.png)


      1%|▍                                                                               | 11/1948 [00:12<35:43,  1.11s/it]cv 값 : 0.44334901612879357람다 값 : 0.641973234434573



![png](output_52_35.png)


      1%|▍                                                                               | 12/1948 [00:13<35:07,  1.09s/it]cv 값 : 0.1810292820987311람다 값 : 1.6295900651976658



![png](output_52_38.png)


      1%|▌                                                                               | 13/1948 [00:14<34:52,  1.08s/it]cv 값 : 0.3687922129997283람다 값 : -0.1354001507292183



![png](output_52_41.png)


      1%|▌                                                                               | 14/1948 [00:15<35:10,  1.09s/it]cv 값 : 0.38598175366952775람다 값 : 0.735545150956471



![png](output_52_44.png)


      1%|▌                                                                               | 15/1948 [00:16<35:22,  1.10s/it]cv 값 : 0.20842514741945511람다 값 : -0.42246649929344404



![png](output_52_47.png)


      1%|▋                                                                               | 16/1948 [00:17<36:13,  1.12s/it]cv 값 : 0.30859136290462214람다 값 : 0.5428777180160058



![png](output_52_50.png)


      1%|▋                                                                               | 17/1948 [00:18<36:49,  1.14s/it]cv 값 : 0.18951621120743142람다 값 : 0.14218046459251726



![png](output_52_53.png)


      1%|▋                                                                               | 18/1948 [00:20<36:01,  1.12s/it]
    
    cv 값 : 0.3624663362291062
    람다 값 : 1.4471464937917469



![png](output_52_56.png)


      1%|▊                                                                               | 19/1948 [00:21<35:00,  1.09s/it]cv 값 : 0.12105491363362339람다 값 : 1.4354496402935992



![png](output_52_59.png)


      1%|▊                                                                               | 20/1948 [00:22<36:40,  1.14s/it]cv 값 : 0.2160823415210115람다 값 : 0.45119322678053075



![png](output_52_62.png)


      1%|▊                                                                               | 21/1948 [00:23<35:25,  1.10s/it]cv 값 : 0.2608951588481173람다 값 : 1.8081409454939155



![png](output_52_65.png)


      1%|▉                                                                               | 22/1948 [00:24<35:51,  1.12s/it]cv 값 : 0.4070526877391136람다 값 : 1.119069471316562



![png](output_52_68.png)


      1%|▉                                                                               | 23/1948 [00:25<36:14,  1.13s/it]
    
    cv 값 : 0.2378608442622346
    람다 값 : 0.3290847844262547



![png](output_52_71.png)


      1%|▉                                                                               | 24/1948 [00:26<35:55,  1.12s/it]cv 값 : 0.4822322927724152람다 값 : -0.25488941770346607



![png](output_52_74.png)


      1%|█                                                                               | 25/1948 [00:27<35:57,  1.12s/it]cv 값 : 0.24899494483962537람다 값 : -0.2186398313161886



![png](output_52_77.png)


      1%|█                                                                               | 26/1948 [00:28<35:34,  1.11s/it]cv 값 : 0.20110528522396634람다 값 : 0.1407153198788213



![png](output_52_80.png)


      1%|█                                                                               | 27/1948 [00:29<34:37,  1.08s/it]
    
    cv 값 : 0.3894371777033834
    람다 값 : 1.4465696987051047



![png](output_52_83.png)


      1%|█▏                                                                              | 28/1948 [00:31<36:10,  1.13s/it]
    
    cv 값 : 0.3552343424399798
    람다 값 : 0.17818001129377112



![png](output_52_86.png)


      1%|█▏                                                                              | 29/1948 [00:32<37:39,  1.18s/it]
    
    cv 값 : 0.9489073424739632
    람다 값 : 0.1766909407824638



![png](output_52_89.png)


      2%|█▏                                                                              | 30/1948 [00:33<36:28,  1.14s/it]
    
    cv 값 : 0.3401372469478217
    람다 값 : 1.1641652479910956



![png](output_52_92.png)


      2%|█▎                                                                              | 31/1948 [00:34<36:12,  1.13s/it]
    
    cv 값 : 0.20730031479968092
    람다 값 : 1.997184211099463



![png](output_52_95.png)


      2%|█▎                                                                              | 32/1948 [00:35<36:07,  1.13s/it]
    
    cv 값 : 0.30157835268740524
    람다 값 : -0.006888987572507405



![png](output_52_98.png)


      2%|█▎                                                                              | 33/1948 [00:37<37:11,  1.17s/it]cv 값 : 0.45105053925933913람다 값 : 0.33379092253947884



![png](output_52_101.png)


      2%|█▍                                                                              | 34/1948 [00:38<36:11,  1.13s/it]
    
    cv 값 : 0.19616241131547918
    람다 값 : 0.5987304461503334



![png](output_52_104.png)


      2%|█▍                                                                              | 35/1948 [00:39<34:36,  1.09s/it]
    
    cv 값 : 0.5605689630026796
    람다 값 : -0.38890713671142657



![png](output_52_107.png)


      2%|█▍                                                                              | 36/1948 [00:40<34:02,  1.07s/it]
    
    cv 값 : 0.21947399093059386
    람다 값 : -0.6975174426008258



![png](output_52_110.png)


      2%|█▌                                                                              | 37/1948 [00:41<33:16,  1.04s/it]
    
    cv 값 : 0.11445925257885475
    람다 값 : 1.0125288629921458



![png](output_52_113.png)


      2%|█▌                                                                              | 38/1948 [00:42<34:17,  1.08s/it]
    
    cv 값 : 0.5183607523647678
    람다 값 : 0.5495816140074805



![png](output_52_116.png)


      2%|█▌                                                                              | 39/1948 [00:43<33:44,  1.06s/it]
    
    cv 값 : 0.1361160290969987
    람다 값 : -0.9203271555670565



![png](output_52_119.png)


      2%|█▋                                                                              | 40/1948 [00:44<33:41,  1.06s/it]
    
    cv 값 : 0.14553508997833803
    람다 값 : -0.3790999070019723



![png](output_52_122.png)


      2%|█▋                                                                              | 41/1948 [00:45<34:55,  1.10s/it]
    
    cv 값 : 0.32672867918665666
    람다 값 : 0.5661386925457498



![png](output_52_125.png)


      2%|█▋                                                                              | 42/1948 [00:46<34:20,  1.08s/it]
    
    cv 값 : 0.24518049586613563
    람다 값 : -0.22091803308478897



![png](output_52_128.png)


      2%|█▊                                                                              | 43/1948 [00:47<35:36,  1.12s/it]
    
    cv 값 : 0.28401551002623193
    람다 값 : 0.8580048963837191



![png](output_52_131.png)


      2%|█▊                                                                              | 44/1948 [00:48<34:56,  1.10s/it]
    
    cv 값 : 0.34053475758692625
    람다 값 : 1.280602025762187



![png](output_52_134.png)


      2%|█▊                                                                              | 45/1948 [00:49<34:20,  1.08s/it]
    
    cv 값 : 0.37800238603385855
    람다 값 : 0.8428732841015317



![png](output_52_137.png)


      2%|█▉                                                                              | 46/1948 [00:50<34:09,  1.08s/it]
    
    cv 값 : 0.4929857219696969
    람다 값 : 0.1682021945581777



![png](output_52_140.png)


      2%|█▉                                                                              | 47/1948 [00:52<36:19,  1.15s/it]cv 값 : 0.2657795344888713람다 값 : -0.20421627140886828



![png](output_52_143.png)


      2%|█▉                                                                              | 48/1948 [00:53<35:34,  1.12s/it]cv 값 : 0.16509155342394222람다 값 : 1.5349644266295632



![png](output_52_146.png)


      3%|██                                                                              | 49/1948 [00:54<35:07,  1.11s/it]cv 값 : 0.3015017439732714람다 값 : 1.132510387889486



![png](output_52_149.png)


      3%|██                                                                              | 50/1948 [00:55<36:39,  1.16s/it]cv 값 : 0.19767423511514653람다 값 : -0.010024475921902505



![png](output_52_152.png)


      3%|██                                                                              | 51/1948 [00:56<38:12,  1.21s/it]cv 값 : 0.25520925385177307람다 값 : 0.970942464616921



![png](output_52_155.png)


      3%|██▏                                                                             | 52/1948 [00:58<37:02,  1.17s/it]cv 값 : 0.33611086444795724람다 값 : -0.2996612835532012



![png](output_52_158.png)


      3%|██▏                                                                             | 53/1948 [00:59<36:34,  1.16s/it]cv 값 : 0.22537177086739402람다 값 : -0.6543172282204229



![png](output_52_161.png)


      3%|██▏                                                                             | 54/1948 [01:00<36:06,  1.14s/it]cv 값 : 0.15961240160892298람다 값 : 0.869010704771371



![png](output_52_164.png)


      3%|██▎                                                                             | 55/1948 [01:01<35:27,  1.12s/it]cv 값 : 0.26563064452850554람다 값 : 0.01952527344461887



![png](output_52_167.png)


      3%|██▎                                                                             | 56/1948 [01:02<37:45,  1.20s/it]cv 값 : 0.46807548920852304람다 값 : 0.6579566246301017



![png](output_52_170.png)


      3%|██▎                                                                             | 57/1948 [01:03<38:06,  1.21s/it]cv 값 : 0.21447575544303774람다 값 : 0.9832706467689969



![png](output_52_173.png)


      3%|██▍                                                                             | 58/1948 [01:05<38:36,  1.23s/it]
    
    cv 값 : 0.704887367499074
    람다 값 : 0.3996989504286678



![png](output_52_176.png)


      3%|██▍                                                                             | 59/1948 [01:06<39:57,  1.27s/it]cv 값 : 0.25856747950300607람다 값 : 0.35896294989739086



![png](output_52_179.png)


      3%|██▍                                                                             | 60/1948 [01:07<40:42,  1.29s/it]cv 값 : 0.19001140439005082람다 값 : 1.621750237896497



![png](output_52_182.png)


      3%|██▌                                                                             | 61/1948 [01:09<40:07,  1.28s/it]cv 값 : 0.5213096295106417람다 값 : -0.03414634226072599



![png](output_52_185.png)


      3%|██▌                                                                             | 62/1948 [01:10<38:34,  1.23s/it]
    
    cv 값 : 0.23516121484773
    람다 값 : 2.0145102977938256



![png](output_52_188.png)


      3%|██▌                                                                             | 63/1948 [01:11<38:35,  1.23s/it]
    
    cv 값 : 0.35658347753075625
    람다 값 : -1.061666768260232



![png](output_52_191.png)


      3%|██▋                                                                             | 64/1948 [01:12<37:12,  1.18s/it]cv 값 : 0.2241894070860468람다 값 : 0.7122553153722785



![png](output_52_194.png)


      3%|██▋                                                                             | 65/1948 [01:13<36:42,  1.17s/it]cv 값 : 0.4022767489879665람다 값 : 0.7444280847698478



![png](output_52_197.png)


      3%|██▋                                                                             | 66/1948 [01:14<36:52,  1.18s/it]cv 값 : 0.2670944131139075람다 값 : 1.8334421672457268



![png](output_52_200.png)


      3%|██▊                                                                             | 67/1948 [01:16<36:25,  1.16s/it]cv 값 : 0.1742317870678618람다 값 : 0.0389814311788649



![png](output_52_203.png)


      3%|██▊                                                                             | 68/1948 [01:17<36:31,  1.17s/it]cv 값 : 0.15532929883954333람다 값 : -0.15393552859243095



![png](output_52_206.png)


      4%|██▊                                                                             | 69/1948 [01:18<35:43,  1.14s/it]cv 값 : 0.1864038017231435람다 값 : -0.589523452718305



![png](output_52_209.png)


      4%|██▊                                                                             | 70/1948 [01:19<37:26,  1.20s/it]cv 값 : 1.5813570803339434람다 값 : -0.863269705353409



![png](output_52_212.png)


      4%|██▉                                                                             | 71/1948 [01:20<37:32,  1.20s/it]cv 값 : 0.24952570214682215람다 값 : -0.49632287857614543



![png](output_52_215.png)


      4%|██▉                                                                             | 72/1948 [01:21<35:52,  1.15s/it]cv 값 : 0.5267038846636638람다 값 : 0.47537870198387905



![png](output_52_218.png)


      4%|██▉                                                                             | 73/1948 [01:23<35:51,  1.15s/it]
    
    cv 값 : 0.2933133645351817
    람다 값 : 1.2346886278139546



![png](output_52_221.png)


      4%|███                                                                             | 74/1948 [01:24<38:23,  1.23s/it]cv 값 : 0.32341964495300013람다 값 : 1.1464740388842498



![png](output_52_224.png)


      4%|███                                                                             | 75/1948 [01:25<37:09,  1.19s/it]cv 값 : 0.16072591222604077람다 값 : 1.2838757055490206



![png](output_52_227.png)


      4%|███                                                                             | 76/1948 [01:26<37:59,  1.22s/it]cv 값 : 0.2629938329644189람다 값 : 0.6735526519150805



![png](output_52_230.png)


      4%|███▏                                                                            | 77/1948 [01:27<37:32,  1.20s/it]cv 값 : 0.5764025678626077람다 값 : 0.29609569370921135



![png](output_52_233.png)


      4%|███▏                                                                            | 78/1948 [01:29<38:31,  1.24s/it]cv 값 : 0.44838914077733477람다 값 : 0.06588113171694676



![png](output_52_236.png)


      4%|███▏                                                                            | 79/1948 [01:30<36:47,  1.18s/it]cv 값 : 0.255174070158105람다 값 : 0.6030004090514938



![png](output_52_239.png)


      4%|███▎                                                                            | 80/1948 [01:31<35:59,  1.16s/it]cv 값 : 0.39934259351087376람다 값 : 0.7445910540734114



![png](output_52_242.png)


      4%|███▎                                                                            | 81/1948 [01:32<34:40,  1.11s/it]cv 값 : 0.2050510399089547람다 값 : -0.14302171452744042



![png](output_52_245.png)


      4%|███▎                                                                            | 82/1948 [01:33<36:24,  1.17s/it]cv 값 : 0.6011888056469271람다 값 : 0.509950983291062



![png](output_52_248.png)


      4%|███▍                                                                            | 83/1948 [01:34<36:41,  1.18s/it]cv 값 : 0.19471052764126204람다 값 : 0.7867520270681186



![png](output_52_251.png)


      4%|███▍                                                                            | 84/1948 [01:36<36:33,  1.18s/it]cv 값 : 0.33674491720174893람다 값 : 1.0949787223137248



![png](output_52_254.png)


      4%|███▍                                                                            | 85/1948 [01:37<36:00,  1.16s/it]
    
    cv 값 : 0.2620206085619278
    람다 값 : 0.3146513978752726



![png](output_52_257.png)


      4%|███▌                                                                            | 86/1948 [01:38<34:59,  1.13s/it]cv 값 : 0.3712120542012271람다 값 : 0.16016458003884798



![png](output_52_260.png)


      4%|███▌                                                                            | 87/1948 [01:40<46:33,  1.50s/it]cv 값 : 0.23292104981868775람다 값 : 1.2751095136547446



![png](output_52_263.png)


      5%|███▌                                                                            | 88/1948 [01:41<42:56,  1.39s/it]cv 값 : 0.33386275249518715람다 값 : 0.6671685328224579



![png](output_52_266.png)


      5%|███▋                                                                            | 89/1948 [01:42<39:28,  1.27s/it]cv 값 : 0.38722414381921166람다 값 : 0.8258000540029893



![png](output_52_269.png)


      5%|███▋                                                                            | 90/1948 [01:43<37:22,  1.21s/it]cv 값 : 0.5452952681712284람다 값 : -0.49449101073015195



![png](output_52_272.png)


      5%|███▋                                                                            | 91/1948 [01:45<36:50,  1.19s/it]cv 값 : 0.20985733776156637람다 값 : 0.7718382850055175



![png](output_52_275.png)


      5%|███▊                                                                            | 92/1948 [01:46<36:49,  1.19s/it]cv 값 : 0.18086595349619222람다 값 : 1.2795068236516325



![png](output_52_278.png)


      5%|███▊                                                                            | 93/1948 [01:47<35:13,  1.14s/it]cv 값 : 0.3128335074047463람다 값 : 0.20354370823797782



![png](output_52_281.png)


      5%|███▊                                                                            | 94/1948 [01:48<34:40,  1.12s/it]cv 값 : 0.4264550173176483람다 값 : 0.7441130748474921



![png](output_52_284.png)


      5%|███▉                                                                            | 95/1948 [01:49<34:01,  1.10s/it]cv 값 : 0.1768936986993455람다 값 : 0.42137983615151725



![png](output_52_287.png)


      5%|███▉                                                                            | 96/1948 [01:50<35:31,  1.15s/it]cv 값 : 0.16624155189096418람다 값 : 2.3224175276352255



![png](output_52_290.png)


      5%|███▉                                                                            | 97/1948 [01:51<34:55,  1.13s/it]cv 값 : 0.53083277288682람다 값 : -0.16130477151109981



![png](output_52_293.png)


      5%|████                                                                            | 98/1948 [01:52<36:11,  1.17s/it]cv 값 : 0.26469959558528894람다 값 : 0.669055942010987



![png](output_52_296.png)


      5%|████                                                                            | 99/1948 [01:54<35:27,  1.15s/it]cv 값 : 0.4079848381042146람다 값 : 1.4125354776143877



![png](output_52_299.png)


      5%|████                                                                           | 100/1948 [01:55<35:09,  1.14s/it]cv 값 : 0.20576830779834668람다 값 : 1.5139184215677977



![png](output_52_302.png)


      5%|████                                                                           | 101/1948 [01:56<35:11,  1.14s/it]cv 값 : 0.44212246600595617람다 값 : -0.39233013801265354



![png](output_52_305.png)


      5%|████▏                                                                          | 102/1948 [01:57<37:11,  1.21s/it]cv 값 : 0.1672902092258036람다 값 : 1.2047239664070664



![png](output_52_308.png)


      5%|████▏                                                                          | 103/1948 [01:58<36:53,  1.20s/it]cv 값 : 0.35823871628525467람다 값 : 0.10997558019106596



![png](output_52_311.png)


      5%|████▏                                                                          | 104/1948 [02:00<36:43,  1.20s/it]cv 값 : 0.30732119175304534람다 값 : 0.5654707305103649



![png](output_52_314.png)


      5%|████▎                                                                          | 105/1948 [02:01<36:55,  1.20s/it]cv 값 : 0.18793505351330722람다 값 : 1.3369446493424384



![png](output_52_317.png)


      5%|████▎                                                                          | 106/1948 [02:02<37:06,  1.21s/it]cv 값 : 0.2699706563083961람다 값 : 1.3237680064674986



![png](output_52_320.png)


      5%|████▎                                                                          | 107/1948 [02:03<37:08,  1.21s/it]cv 값 : 0.20517600957508067람다 값 : -0.3336845367141278



![png](output_52_323.png)


      6%|████▍                                                                          | 108/1948 [02:04<35:14,  1.15s/it]cv 값 : 0.19320026618800426람다 값 : 0.670404073018684



![png](output_52_326.png)


      6%|████▍                                                                          | 109/1948 [02:06<36:48,  1.20s/it]cv 값 : 0.6458814351183465람다 값 : 0.5345081440815521



![png](output_52_329.png)


      6%|████▍                                                                          | 110/1948 [02:07<36:54,  1.21s/it]cv 값 : 0.12725560699389316람다 값 : 1.342264525622712



![png](output_52_332.png)


      6%|████▌                                                                          | 111/1948 [02:08<35:22,  1.16s/it]cv 값 : 0.31866221425741026람다 값 : 0.7411012714027428



![png](output_52_335.png)


      6%|████▌                                                                          | 112/1948 [02:09<34:26,  1.13s/it]
    
    cv 값 : 0.310712660752888
    람다 값 : 1.9083319089712065



![png](output_52_338.png)


      6%|████▌                                                                          | 113/1948 [02:10<34:51,  1.14s/it]
    
    cv 값 : 0.27699252666685087
    람다 값 : 0.9730994987726241



![png](output_52_341.png)


      6%|████▌                                                                          | 114/1948 [02:11<37:11,  1.22s/it]cv 값 : 0.45656534766484563람다 값 : 0.5803353289361147



![png](output_52_344.png)


      6%|████▋                                                                          | 115/1948 [02:13<37:15,  1.22s/it]
    
    cv 값 : 0.6295869349267522
    람다 값 : 0.5446302566921629



![png](output_52_347.png)


      6%|████▋                                                                          | 116/1948 [02:14<35:59,  1.18s/it]
    
    cv 값 : 0.29564580623661585
    람다 값 : 1.5127408945248384



![png](output_52_350.png)


      6%|████▋                                                                          | 117/1948 [02:15<35:06,  1.15s/it]
    
    cv 값 : 0.8696332759587851
    람다 값 : 0.236016454171884



![png](output_52_353.png)


      6%|████▊                                                                          | 118/1948 [02:16<33:34,  1.10s/it]
    
    cv 값 : 0.37362598131159025
    람다 값 : 0.5632938797430495



![png](output_52_356.png)


      6%|████▊                                                                          | 119/1948 [02:17<34:11,  1.12s/it]
    
    cv 값 : 0.7325537144759513
    람다 값 : 0.3637548622038612



![png](output_52_359.png)


      6%|████▊                                                                          | 120/1948 [02:18<35:22,  1.16s/it]
    
    cv 값 : 0.7784036517496438
    람다 값 : 0.4313096775249441



![png](output_52_362.png)


      6%|████▉                                                                          | 121/1948 [02:19<34:41,  1.14s/it]
    
    cv 값 : 0.15123098289099884
    람다 값 : 0.8802586038154674



![png](output_52_365.png)


      6%|████▉                                                                          | 122/1948 [02:20<33:25,  1.10s/it]
    
    cv 값 : 0.1077299791093073
    람다 값 : 1.5533191151532688



![png](output_52_368.png)


      6%|████▉                                                                          | 123/1948 [02:22<34:39,  1.14s/it]
    
    cv 값 : 0.13798216316352105
    람다 값 : 0.09287552643172146



![png](output_52_371.png)


      6%|█████                                                                          | 124/1948 [02:23<34:23,  1.13s/it]cv 값 : 0.22092104952584643람다 값 : -0.8211019779245082



![png](output_52_374.png)


      6%|█████                                                                          | 125/1948 [02:24<34:42,  1.14s/it]
    
    cv 값 : 0.2222930821753316
    람다 값 : 2.415118971357763



![png](output_52_377.png)


      6%|█████                                                                          | 126/1948 [02:25<34:09,  1.12s/it]cv 값 : 0.31427105857186266람다 값 : 0.9031867880301178



![png](output_52_380.png)


      7%|█████▏                                                                         | 127/1948 [02:26<37:36,  1.24s/it]cv 값 : 0.2597194621495532람다 값 : -0.018393501420675144



![png](output_52_383.png)


      7%|█████▏                                                                         | 128/1948 [02:28<37:14,  1.23s/it]cv 값 : 0.7426267063650068람다 값 : 0.42651713423374243



![png](output_52_386.png)


      7%|█████▏                                                                         | 129/1948 [02:29<35:57,  1.19s/it]cv 값 : 0.26131238180968863람다 값 : -0.30954048216579744



![png](output_52_389.png)


      7%|█████▎                                                                         | 130/1948 [02:30<35:09,  1.16s/it]cv 값 : 0.21521301831641457람다 값 : -0.22847998063725689



![png](output_52_392.png)


      7%|█████▎                                                                         | 131/1948 [02:31<35:35,  1.18s/it]cv 값 : 0.20374546375002694람다 값 : 0.5164751699078459



![png](output_52_395.png)


      7%|█████▎                                                                         | 132/1948 [02:32<36:04,  1.19s/it]cv 값 : 0.2767662489566363람다 값 : 0.6026771867770206



![png](output_52_398.png)


      7%|█████▍                                                                         | 133/1948 [02:33<35:32,  1.18s/it]cv 값 : 0.36731192581567174람다 값 : -0.7149201464505129



![png](output_52_401.png)


      7%|█████▍                                                                         | 134/1948 [02:35<35:06,  1.16s/it]cv 값 : 0.5155959538021363람다 값 : -0.2770928822498353



![png](output_52_404.png)


      7%|█████▍                                                                         | 135/1948 [02:36<38:06,  1.26s/it]cv 값 : 0.2614109563500471람다 값 : -0.7785362064880373



![png](output_52_407.png)


      7%|█████▌                                                                         | 136/1948 [02:38<39:59,  1.32s/it]cv 값 : 0.30841077701229475람다 값 : 0.5122928734070521



![png](output_52_410.png)


      7%|█████▌                                                                         | 137/1948 [02:39<39:14,  1.30s/it]cv 값 : 0.4822389858816193람다 값 : 0.6204349779188361



![png](output_52_413.png)


      7%|█████▌                                                                         | 138/1948 [02:40<38:28,  1.28s/it]cv 값 : 0.35618471574777094람다 값 : 0.7870672476250218



![png](output_52_416.png)


      7%|█████▋                                                                         | 139/1948 [02:41<35:55,  1.19s/it]cv 값 : 0.3566405987883651람다 값 : -0.05331217792368204



![png](output_52_419.png)


      7%|█████▋                                                                         | 140/1948 [02:42<36:06,  1.20s/it]cv 값 : 0.29650067274581915람다 값 : 1.0968599973043616



![png](output_52_422.png)


      7%|█████▋                                                                         | 141/1948 [02:44<38:12,  1.27s/it]cv 값 : 0.30777227587863226람다 값 : 0.27684442879236654



![png](output_52_425.png)


      7%|█████▊                                                                         | 142/1948 [02:45<37:00,  1.23s/it]cv 값 : 0.17903183186346652람다 값 : 0.7822436390939863



![png](output_52_428.png)


      7%|█████▊                                                                         | 143/1948 [02:46<35:33,  1.18s/it]cv 값 : 0.2458517906619314람다 값 : 0.7434537752038621



![png](output_52_431.png)


      7%|█████▊                                                                         | 144/1948 [02:47<39:30,  1.31s/it]cv 값 : 0.3665632409215946람다 값 : -0.0061043533946943005



![png](output_52_434.png)


      7%|█████▉                                                                         | 145/1948 [02:49<39:33,  1.32s/it]cv 값 : 0.57257785034526람다 값 : 0.6108695975958368



![png](output_52_437.png)


      7%|█████▉                                                                         | 146/1948 [02:50<38:45,  1.29s/it]cv 값 : 0.38081696477728294람다 값 : -0.3836452611647775



![png](output_52_440.png)


      8%|█████▉                                                                         | 147/1948 [02:51<38:28,  1.28s/it]cv 값 : 0.265082737462391람다 값 : 0.11828794068807444



![png](output_52_443.png)


      8%|██████                                                                         | 148/1948 [02:52<37:16,  1.24s/it]cv 값 : 0.6999246584097987람다 값 : 0.25704254328477244



![png](output_52_446.png)


      8%|██████                                                                         | 149/1948 [02:53<35:46,  1.19s/it]cv 값 : 0.36925408310822916람다 값 : 0.9689168581025593



![png](output_52_449.png)


      8%|██████                                                                         | 150/1948 [02:55<35:38,  1.19s/it]cv 값 : 0.6984291739899235람다 값 : -0.26596253089062155



![png](output_52_452.png)


      8%|██████                                                                         | 151/1948 [02:56<34:57,  1.17s/it]cv 값 : 0.5873101739604119람다 값 : 0.18841991583229584



![png](output_52_455.png)


      8%|██████▏                                                                        | 152/1948 [02:57<33:54,  1.13s/it]cv 값 : 0.387110187113778람다 값 : 0.660667903394147



![png](output_52_458.png)


      8%|██████▏                                                                        | 153/1948 [02:58<34:07,  1.14s/it]cv 값 : 0.2953927601598908람다 값 : -0.4836762561525202



![png](output_52_461.png)


      8%|██████▏                                                                        | 154/1948 [02:59<35:45,  1.20s/it]cv 값 : 0.26178717805447876람다 값 : -0.6184432959547533



![png](output_52_464.png)


      8%|██████▎                                                                        | 155/1948 [03:01<35:42,  1.19s/it]cv 값 : 0.2912988097269136람다 값 : 0.2448957451935105



![png](output_52_467.png)


      8%|██████▎                                                                        | 156/1948 [03:02<35:41,  1.20s/it]cv 값 : 0.3590183078120557람다 값 : 0.6007260696079844



![png](output_52_470.png)


      8%|██████▎                                                                        | 157/1948 [03:03<34:42,  1.16s/it]cv 값 : 0.18070271167100768람다 값 : -0.5959740926154226



![png](output_52_473.png)


      8%|██████▍                                                                        | 158/1948 [03:04<33:44,  1.13s/it]cv 값 : 0.18560544451724778람다 값 : 1.2715811587444819



![png](output_52_476.png)


      8%|██████▍                                                                        | 159/1948 [03:05<35:02,  1.18s/it]cv 값 : 0.5007369973807684람다 값 : 0.101819748900383



![png](output_52_479.png)


      8%|██████▍                                                                        | 160/1948 [03:06<33:32,  1.13s/it]cv 값 : 0.44460478171521983람다 값 : -0.49881204994861267



![png](output_52_482.png)


      8%|██████▌                                                                        | 161/1948 [03:07<33:21,  1.12s/it]cv 값 : 0.6027885452309106람다 값 : -0.26171052072746614



![png](output_52_485.png)


      8%|██████▌                                                                        | 162/1948 [03:08<33:39,  1.13s/it]cv 값 : 0.8994136807863088람다 값 : 0.3737891853467547



![png](output_52_488.png)


      8%|██████▌                                                                        | 163/1948 [03:10<34:39,  1.16s/it]cv 값 : 0.26722218027279554람다 값 : 0.9794054281477272



![png](output_52_491.png)


      8%|██████▋                                                                        | 164/1948 [03:11<33:41,  1.13s/it]cv 값 : 0.23002416454722804람다 값 : 0.2805250558611558



![png](output_52_494.png)


      8%|██████▋                                                                        | 165/1948 [03:12<32:20,  1.09s/it]cv 값 : 0.369739509803231람다 값 : -0.1426042406165636



![png](output_52_497.png)


      9%|██████▋                                                                        | 166/1948 [03:13<32:51,  1.11s/it]cv 값 : 0.18291625879568613람다 값 : 0.876076467900788



![png](output_52_500.png)


      9%|██████▊                                                                        | 167/1948 [03:14<33:11,  1.12s/it]cv 값 : 0.3532280214203286람다 값 : 1.31636750325708



![png](output_52_503.png)


      9%|██████▊                                                                        | 168/1948 [03:15<34:46,  1.17s/it]cv 값 : 1.6474257884614592람다 값 : 0.10285139926248697



![png](output_52_506.png)


      9%|██████▊                                                                        | 169/1948 [03:16<33:51,  1.14s/it]cv 값 : 0.48736786991073167람다 값 : 0.16846784300169193



![png](output_52_509.png)


      9%|██████▉                                                                        | 170/1948 [03:17<33:39,  1.14s/it]cv 값 : 0.17913691515965346람다 값 : 1.8013451368662716



![png](output_52_512.png)


      9%|██████▉                                                                        | 171/1948 [03:19<38:01,  1.28s/it]cv 값 : 0.8515050163020191람다 값 : -0.04678925911137554



![png](output_52_515.png)


      9%|██████▉                                                                        | 172/1948 [03:20<38:21,  1.30s/it]cv 값 : 0.8670365428278853람다 값 : 0.16848832914346773



![png](output_52_518.png)


      9%|███████                                                                        | 173/1948 [03:22<36:52,  1.25s/it]cv 값 : 0.27373830457785225람다 값 : 0.8331186074841448



![png](output_52_521.png)


      9%|███████                                                                        | 174/1948 [03:23<40:24,  1.37s/it]cv 값 : 0.2316198996121725람다 값 : -0.25067431797904943



![png](output_52_524.png)


      9%|███████                                                                        | 175/1948 [03:25<46:35,  1.58s/it]cv 값 : 0.40819535844353183람다 값 : -0.019192307240531194



![png](output_52_527.png)


      9%|███████▏                                                                       | 176/1948 [03:27<46:03,  1.56s/it]cv 값 : 0.20877892440572582람다 값 : -0.7853511031987314



![png](output_52_530.png)


      9%|███████▏                                                                       | 177/1948 [03:28<47:04,  1.59s/it]cv 값 : 0.42164744954602557람다 값 : 0.6804122569476224



![png](output_52_533.png)


      9%|███████▏                                                                       | 178/1948 [03:30<46:28,  1.58s/it]cv 값 : 0.17862826971442516람다 값 : 1.8907544154301161



![png](output_52_536.png)


      9%|███████▎                                                                       | 179/1948 [03:31<44:19,  1.50s/it]cv 값 : 0.3418474358452204람다 값 : 0.8532647318050368



![png](output_52_539.png)


      9%|███████▎                                                                       | 180/1948 [03:33<42:25,  1.44s/it]cv 값 : 0.23317675225517187람다 값 : 1.0527514102568558



![png](output_52_542.png)


      9%|███████▎                                                                       | 181/1948 [03:34<40:51,  1.39s/it]cv 값 : 0.39674112835456243람다 값 : 0.11708638569768945



![png](output_52_545.png)


      9%|███████▍                                                                       | 182/1948 [03:35<42:18,  1.44s/it]cv 값 : 0.33402221716376645람다 값 : 0.4110643436258494



![png](output_52_548.png)


      9%|███████▍                                                                       | 183/1948 [03:37<39:24,  1.34s/it]cv 값 : 0.3772468648177799람다 값 : 0.22280280458772175



![png](output_52_551.png)


      9%|███████▍                                                                       | 184/1948 [03:38<37:26,  1.27s/it]cv 값 : 0.1677910092702148람다 값 : 1.7326692877580732



![png](output_52_554.png)


      9%|███████▌                                                                       | 185/1948 [03:39<37:41,  1.28s/it]cv 값 : 0.6112834576497435람다 값 : -0.37156905096254456



![png](output_52_557.png)


     10%|███████▌                                                                       | 186/1948 [03:40<36:52,  1.26s/it]cv 값 : 0.4139555685600997람다 값 : -0.011518686839813446



![png](output_52_560.png)


     10%|███████▌                                                                       | 187/1948 [03:42<38:58,  1.33s/it]cv 값 : 0.4486955651331335람다 값 : 0.8497001511112379



![png](output_52_563.png)


     10%|███████▌                                                                       | 188/1948 [03:43<38:29,  1.31s/it]cv 값 : 0.23324374779941606람다 값 : -0.9952295676646782



![png](output_52_566.png)


     10%|███████▋                                                                       | 189/1948 [03:44<37:23,  1.28s/it]cv 값 : 0.44340223249554145람다 값 : 0.6615290696267532



![png](output_52_569.png)


     10%|███████▋                                                                       | 190/1948 [03:46<38:56,  1.33s/it]cv 값 : 0.20112406092794186람다 값 : -1.280236508442312



![png](output_52_572.png)


     10%|███████▋                                                                       | 191/1948 [03:47<36:41,  1.25s/it]cv 값 : 0.4969222618070296람다 값 : -0.4650839640419881



![png](output_52_575.png)


     10%|███████▊                                                                       | 192/1948 [03:48<35:23,  1.21s/it]cv 값 : 0.20983578386362067람다 값 : 0.7446709009627941



![png](output_52_578.png)


     10%|███████▊                                                                       | 193/1948 [03:49<35:24,  1.21s/it]cv 값 : 0.4792186994923619람다 값 : -0.41924180265773725



![png](output_52_581.png)


     10%|███████▊                                                                       | 194/1948 [03:51<40:07,  1.37s/it]cv 값 : 0.19452467080740984람다 값 : 0.5667044209821677



![png](output_52_584.png)


     10%|███████▉                                                                       | 195/1948 [03:52<39:15,  1.34s/it]cv 값 : 0.23640445338104013람다 값 : 1.4721276444806735



![png](output_52_587.png)


     10%|███████▉                                                                       | 196/1948 [03:53<38:47,  1.33s/it]cv 값 : 0.17619041428787927람다 값 : 1.185148392488958



![png](output_52_590.png)


     10%|███████▉                                                                       | 197/1948 [03:55<41:49,  1.43s/it]cv 값 : 0.49621293970729746람다 값 : 0.23823288263879058



![png](output_52_593.png)


     10%|████████                                                                       | 198/1948 [03:56<38:41,  1.33s/it]cv 값 : 0.720468254968489람다 값 : -0.10653165670889592



![png](output_52_596.png)


     10%|████████                                                                       | 199/1948 [03:57<38:01,  1.30s/it]cv 값 : 0.26931318477040717람다 값 : -0.056285939923355344



![png](output_52_599.png)


     10%|████████                                                                       | 200/1948 [03:58<36:21,  1.25s/it]cv 값 : 0.6956214827226974람다 값 : -0.11398749374975803



![png](output_52_602.png)


     10%|████████▏                                                                      | 201/1948 [04:00<35:10,  1.21s/it]cv 값 : 0.3945726159239665람다 값 : 0.19080226145200124



![png](output_52_605.png)


     10%|████████▏                                                                      | 202/1948 [04:01<34:22,  1.18s/it]cv 값 : 0.5878801885021775람다 값 : 0.523140008175757



![png](output_52_608.png)


     10%|████████▏                                                                      | 203/1948 [04:02<35:05,  1.21s/it]cv 값 : 0.3530405440076926람다 값 : 0.07563352572204315



![png](output_52_611.png)


     10%|████████▎                                                                      | 204/1948 [04:03<33:08,  1.14s/it]cv 값 : 0.14356804767435843람다 값 : 0.938318546562802



![png](output_52_614.png)


     11%|████████▎                                                                      | 205/1948 [04:04<31:58,  1.10s/it]cv 값 : 0.6508911822583033람다 값 : 0.17045010313152528



![png](output_52_617.png)


     11%|████████▎                                                                      | 206/1948 [04:05<36:01,  1.24s/it]cv 값 : 0.19885751452764966람다 값 : 0.08761764227736



![png](output_52_620.png)


     11%|████████▍                                                                      | 207/1948 [04:07<36:33,  1.26s/it]cv 값 : 0.44490592813756674람다 값 : 0.6242758016863792



![png](output_52_623.png)


     11%|████████▍                                                                      | 208/1948 [04:08<36:28,  1.26s/it]cv 값 : 0.378758614003077람다 값 : 0.21876896135267593



![png](output_52_626.png)


     11%|████████▍                                                                      | 209/1948 [04:09<35:31,  1.23s/it]cv 값 : 0.18927313214370042람다 값 : 0.8184142057283509



![png](output_52_629.png)


     11%|████████▌                                                                      | 210/1948 [04:10<34:59,  1.21s/it]cv 값 : 0.5763864711856888람다 값 : 0.39474865251344665



![png](output_52_632.png)


     11%|████████▌                                                                      | 211/1948 [04:12<35:09,  1.21s/it]cv 값 : 0.20386345243276835람다 값 : -0.29969565605201215



![png](output_52_635.png)


     11%|████████▌                                                                      | 212/1948 [04:13<36:52,  1.27s/it]cv 값 : 0.3382196017312225람다 값 : 0.34635028090294606



![png](output_52_638.png)


     11%|████████▋                                                                      | 213/1948 [04:14<36:57,  1.28s/it]cv 값 : 0.1934518397610946람다 값 : 0.7890580235275692



![png](output_52_641.png)


     11%|████████▋                                                                      | 214/1948 [04:16<41:34,  1.44s/it]cv 값 : 0.6156927846009286람다 값 : 0.4267565268212458



![png](output_52_644.png)


     11%|████████▋                                                                      | 215/1948 [04:17<39:04,  1.35s/it]cv 값 : 0.40481710187117675람다 값 : 0.745161657155108



![png](output_52_647.png)


     11%|████████▊                                                                      | 216/1948 [04:19<38:54,  1.35s/it]cv 값 : 0.345438675044881람다 값 : 0.291924816847104



![png](output_52_650.png)


     11%|████████▊                                                                      | 217/1948 [04:20<37:07,  1.29s/it]cv 값 : 0.3752909506370742람다 값 : 0.4039162779902934



![png](output_52_653.png)


     11%|████████▊                                                                      | 218/1948 [04:21<35:01,  1.21s/it]cv 값 : 0.6492501618809203람다 값 : -0.8058971883991558



![png](output_52_656.png)


     11%|████████▉                                                                      | 219/1948 [04:22<33:48,  1.17s/it]cv 값 : 0.4339233824153307람다 값 : -0.6293520520098969



![png](output_52_659.png)


     11%|████████▉                                                                      | 220/1948 [04:23<35:42,  1.24s/it]cv 값 : 0.36537427892363983람다 값 : -0.3030601205724369



![png](output_52_662.png)


     11%|████████▉                                                                      | 221/1948 [04:24<35:00,  1.22s/it]cv 값 : 0.4933693630569673람다 값 : -0.8984341293082961



![png](output_52_665.png)


     11%|█████████                                                                      | 222/1948 [04:26<35:54,  1.25s/it]cv 값 : 0.2065731945717218람다 값 : 1.080618241253696



![png](output_52_668.png)


     11%|█████████                                                                      | 223/1948 [04:27<33:41,  1.17s/it]cv 값 : 0.27435911685812614람다 값 : 0.7837653157408416



![png](output_52_671.png)


     11%|█████████                                                                      | 224/1948 [04:28<34:33,  1.20s/it]cv 값 : 0.4008604845464597람다 값 : 0.011573857022370466



![png](output_52_674.png)


     12%|█████████                                                                      | 225/1948 [04:29<36:19,  1.26s/it]cv 값 : 0.2353288340261705람다 값 : 1.0904491902582059



![png](output_52_677.png)


     12%|█████████▏                                                                     | 226/1948 [04:31<35:44,  1.25s/it]cv 값 : 0.32025912025289505람다 값 : -0.587659297043401



![png](output_52_680.png)


     12%|█████████▏                                                                     | 227/1948 [04:32<34:35,  1.21s/it]cv 값 : 0.2983270933319518람다 값 : 0.523101752208329



![png](output_52_683.png)


     12%|█████████▏                                                                     | 228/1948 [04:33<33:23,  1.16s/it]cv 값 : 0.1415810348362221람다 값 : 0.8942445158377498



![png](output_52_686.png)


     12%|█████████▎                                                                     | 229/1948 [04:34<33:54,  1.18s/it]cv 값 : 0.716806961597322람다 값 : -0.5020079892485101



![png](output_52_689.png)


     12%|█████████▎                                                                     | 230/1948 [04:35<35:18,  1.23s/it]cv 값 : 0.6005052816896254람다 값 : 0.21321685056079606



![png](output_52_692.png)


     12%|█████████▎                                                                     | 231/1948 [04:37<34:53,  1.22s/it]cv 값 : 0.1906339853090387람다 값 : 2.3122253649614173



![png](output_52_695.png)


     12%|█████████▍                                                                     | 232/1948 [04:38<34:40,  1.21s/it]cv 값 : 0.5722435200782505람다 값 : 0.21066898656527144



![png](output_52_698.png)


     12%|█████████▍                                                                     | 233/1948 [04:39<34:34,  1.21s/it]cv 값 : 0.5707620096281618람다 값 : 0.18162932261636508



![png](output_52_701.png)


     12%|█████████▍                                                                     | 234/1948 [04:40<34:28,  1.21s/it]cv 값 : 0.40519291392119056람다 값 : -0.005571073472390931



![png](output_52_704.png)


     12%|█████████▌                                                                     | 235/1948 [04:41<32:45,  1.15s/it]cv 값 : 0.5197745953294355람다 값 : -0.16990589189995103



![png](output_52_707.png)


     12%|█████████▌                                                                     | 236/1948 [04:42<32:53,  1.15s/it]cv 값 : 0.17286477713121876람다 값 : -0.6512626750608712



![png](output_52_710.png)


     12%|█████████▌                                                                     | 237/1948 [04:44<36:22,  1.28s/it]cv 값 : 0.5232993699811981람다 값 : 0.10268382348518908



![png](output_52_713.png)


     12%|█████████▋                                                                     | 238/1948 [04:45<37:24,  1.31s/it]cv 값 : 0.6247985265925501람다 값 : 0.45944456386529703



![png](output_52_716.png)


     12%|█████████▋                                                                     | 239/1948 [04:46<35:05,  1.23s/it]cv 값 : 0.3135557839564267람다 값 : -0.14898521716365834



![png](output_52_719.png)


     12%|█████████▋                                                                     | 240/1948 [04:48<34:32,  1.21s/it]cv 값 : 0.7258244375836159람다 값 : 0.13372679494305956



![png](output_52_722.png)


     12%|█████████▊                                                                     | 241/1948 [04:49<33:25,  1.17s/it]cv 값 : 0.44911654805957857람다 값 : -0.4026033475414832



![png](output_52_725.png)


     12%|█████████▊                                                                     | 242/1948 [04:50<33:15,  1.17s/it]cv 값 : 0.3214953521383201람다 값 : 0.529788503131833



![png](output_52_728.png)


     12%|█████████▊                                                                     | 243/1948 [04:51<34:21,  1.21s/it]cv 값 : 0.30581963986280497람다 값 : 1.7960300467325734



![png](output_52_731.png)


     13%|█████████▉                                                                     | 244/1948 [04:52<34:05,  1.20s/it]cv 값 : 0.4787038319547979람다 값 : -1.1244459753279152



![png](output_52_734.png)


     13%|█████████▉                                                                     | 245/1948 [04:53<34:27,  1.21s/it]cv 값 : 0.2487497287307002람다 값 : 1.573848773210797



![png](output_52_737.png)


     13%|█████████▉                                                                     | 246/1948 [04:55<34:46,  1.23s/it]cv 값 : 0.7466625703585758람다 값 : 0.1636935594569936



![png](output_52_740.png)


     13%|██████████                                                                     | 247/1948 [04:56<38:07,  1.34s/it]cv 값 : 0.35176765932393905람다 값 : 0.7934593681858004



![png](output_52_743.png)


     13%|██████████                                                                     | 248/1948 [04:58<37:00,  1.31s/it]cv 값 : 0.33329893459630694람다 값 : 0.9615854806523958



![png](output_52_746.png)


     13%|██████████                                                                     | 249/1948 [04:59<34:57,  1.23s/it]cv 값 : 0.25986969538211147람다 값 : 0.45953772147895533



![png](output_52_749.png)


     13%|██████████▏                                                                    | 250/1948 [05:00<33:43,  1.19s/it]cv 값 : 0.531074466363763람다 값 : 0.689704856401239



![png](output_52_752.png)


     13%|██████████▏                                                                    | 251/1948 [05:01<33:18,  1.18s/it]cv 값 : 0.17673438480953207람다 값 : 0.45467102600671516



![png](output_52_755.png)


     13%|██████████▏                                                                    | 252/1948 [05:02<30:52,  1.09s/it]cv 값 : 0.34927788071589705람다 값 : -0.15992239766537353



![png](output_52_758.png)


     13%|██████████▎                                                                    | 253/1948 [05:03<33:48,  1.20s/it]cv 값 : 0.20610026865150124람다 값 : 0.35840742260243



![png](output_52_761.png)


     13%|██████████▎                                                                    | 254/1948 [05:04<32:26,  1.15s/it]cv 값 : 0.38116816312633756람다 값 : 0.8068734906445333



![png](output_52_764.png)


     13%|██████████▎                                                                    | 255/1948 [05:05<32:36,  1.16s/it]cv 값 : 0.6425756275364464람다 값 : -0.920921895101601



![png](output_52_767.png)


     13%|██████████▍                                                                    | 256/1948 [05:07<33:14,  1.18s/it]cv 값 : 0.3687081182238614람다 값 : 0.6794268827021953



![png](output_52_770.png)


     13%|██████████▍                                                                    | 257/1948 [05:08<32:45,  1.16s/it]cv 값 : 0.11623900953734691람다 값 : -1.305044054861732



![png](output_52_773.png)


     13%|██████████▍                                                                    | 258/1948 [05:09<32:43,  1.16s/it]cv 값 : 0.6224340011639722람다 값 : 0.15490985313866423



![png](output_52_776.png)


     13%|██████████▌                                                                    | 259/1948 [05:10<33:02,  1.17s/it]cv 값 : 0.36277050478342726람다 값 : 0.7731374262238311



![png](output_52_779.png)


     13%|██████████▌                                                                    | 260/1948 [05:11<34:09,  1.21s/it]cv 값 : 0.37748713185373484람다 값 : 0.4377746342104944



![png](output_52_782.png)


     13%|██████████▌                                                                    | 261/1948 [05:13<34:10,  1.22s/it]cv 값 : 0.3022373349267476람다 값 : 0.9623006888155664



![png](output_52_785.png)


     13%|██████████▋                                                                    | 262/1948 [05:14<33:47,  1.20s/it]cv 값 : 0.3904148876149271람다 값 : -1.1291852418382116



![png](output_52_788.png)


     14%|██████████▋                                                                    | 263/1948 [05:15<33:33,  1.19s/it]cv 값 : 0.13580469971497355람다 값 : -0.6681518068790436



![png](output_52_791.png)


     14%|██████████▋                                                                    | 264/1948 [05:16<33:22,  1.19s/it]cv 값 : 0.7795074414390067람다 값 : 0.35921800816425703



![png](output_52_794.png)


     14%|██████████▋                                                                    | 265/1948 [05:17<34:10,  1.22s/it]cv 값 : 0.2156941094006407람다 값 : 0.9821158346941816



![png](output_52_797.png)


     14%|██████████▊                                                                    | 266/1948 [05:19<38:12,  1.36s/it]cv 값 : 0.3446028578813677람다 값 : 0.6007143818355087



![png](output_52_800.png)


     14%|██████████▊                                                                    | 267/1948 [05:21<37:52,  1.35s/it]cv 값 : 0.43266250160774067람다 값 : 0.08890434157100509



![png](output_52_803.png)


     14%|██████████▊                                                                    | 268/1948 [05:22<35:07,  1.25s/it]cv 값 : 0.2641095389092671람다 값 : 1.2462228051239441



![png](output_52_806.png)


     14%|██████████▉                                                                    | 269/1948 [05:23<35:23,  1.26s/it]cv 값 : 0.5203951131830329람다 값 : -0.42868783284813833



![png](output_52_809.png)


     14%|██████████▉                                                                    | 270/1948 [05:24<38:29,  1.38s/it]cv 값 : 0.7053910985116706람다 값 : -0.6511969931130187



![png](output_52_812.png)


     14%|██████████▉                                                                    | 271/1948 [05:26<38:18,  1.37s/it]cv 값 : 0.33888718849185157람다 값 : 0.15222495795168



![png](output_52_815.png)


     14%|███████████                                                                    | 272/1948 [05:27<37:44,  1.35s/it]cv 값 : 0.3371387151762066람다 값 : 0.5000789982503632



![png](output_52_818.png)


     14%|███████████                                                                    | 273/1948 [05:28<35:21,  1.27s/it]cv 값 : 0.3390657116925783람다 값 : 0.09555771138962206



![png](output_52_821.png)


     14%|███████████                                                                    | 274/1948 [05:30<40:46,  1.46s/it]cv 값 : 0.7901517942288787람다 값 : 0.10783343039353664



![png](output_52_824.png)


     14%|███████████▏                                                                   | 275/1948 [05:31<38:07,  1.37s/it]cv 값 : 0.2754426101711715람다 값 : 0.8711030736924783



![png](output_52_827.png)


     14%|███████████▏                                                                   | 276/1948 [05:32<35:44,  1.28s/it]cv 값 : 0.3865464409123169람다 값 : -1.7266599017413562



![png](output_52_830.png)


     14%|███████████▏                                                                   | 277/1948 [05:34<35:13,  1.26s/it]cv 값 : 0.34342813593244503람다 값 : 0.8811046990697193



![png](output_52_833.png)


     14%|███████████▎                                                                   | 278/1948 [05:35<35:38,  1.28s/it]cv 값 : 0.15874045575949386람다 값 : 1.3524634925435208



![png](output_52_836.png)


     14%|███████████▎                                                                   | 279/1948 [05:36<33:17,  1.20s/it]cv 값 : 0.8512026678206657람다 값 : 0.18383780675325462



![png](output_52_839.png)


     14%|███████████▎                                                                   | 280/1948 [05:37<31:42,  1.14s/it]cv 값 : 0.19064588456518305람다 값 : 1.6950135877890806



![png](output_52_842.png)


     14%|███████████▍                                                                   | 281/1948 [05:38<30:43,  1.11s/it]cv 값 : 0.3921404839957755람다 값 : -0.44263755269046323



![png](output_52_845.png)


     14%|███████████▍                                                                   | 282/1948 [05:39<32:09,  1.16s/it]cv 값 : 0.49894501059464247람다 값 : 0.5983554513516055



![png](output_52_848.png)


     15%|███████████▍                                                                   | 283/1948 [05:40<31:23,  1.13s/it]cv 값 : 0.36344223151944627람다 값 : 0.8654853684709068



![png](output_52_851.png)


     15%|███████████▌                                                                   | 284/1948 [05:41<31:09,  1.12s/it]cv 값 : 0.509462279786992람다 값 : 0.4568848185232845



![png](output_52_854.png)


     15%|███████████▌                                                                   | 285/1948 [05:43<31:28,  1.14s/it]cv 값 : 0.26330717675124704람다 값 : -0.1345700822864596



![png](output_52_857.png)


     15%|███████████▌                                                                   | 286/1948 [05:44<32:27,  1.17s/it]cv 값 : 0.33695959587758056람다 값 : 1.2137215658837335



![png](output_52_860.png)


     15%|███████████▋                                                                   | 287/1948 [05:45<34:16,  1.24s/it]cv 값 : 0.3071384741046445람다 값 : 0.5249036751412747



![png](output_52_863.png)


     15%|███████████▋                                                                   | 288/1948 [05:46<34:40,  1.25s/it]cv 값 : 0.31746317647450867람다 값 : 1.0554548936947044



![png](output_52_866.png)


     15%|███████████▋                                                                   | 289/1948 [05:48<32:51,  1.19s/it]cv 값 : 0.4145669427443062람다 값 : 0.6756733639519514



![png](output_52_869.png)


     15%|███████████▊                                                                   | 290/1948 [05:49<32:42,  1.18s/it]cv 값 : 0.3030914959879093람다 값 : 0.2334230525635767



![png](output_52_872.png)


     15%|███████████▊                                                                   | 291/1948 [05:50<32:42,  1.18s/it]cv 값 : 0.5734742259366884람다 값 : -0.1892472487201671



![png](output_52_875.png)


     15%|███████████▊                                                                   | 292/1948 [05:51<31:27,  1.14s/it]cv 값 : 0.39572242577322436람다 값 : 0.6635776029573359



![png](output_52_878.png)


     15%|███████████▉                                                                   | 293/1948 [05:52<30:07,  1.09s/it]cv 값 : 0.2638275669594699람다 값 : -0.325663670667047



![png](output_52_881.png)


     15%|███████████▉                                                                   | 294/1948 [05:53<31:01,  1.13s/it]cv 값 : 0.4684402703137191람다 값 : 0.500890052217977



![png](output_52_884.png)


     15%|███████████▉                                                                   | 295/1948 [05:54<30:26,  1.11s/it]cv 값 : 0.43768988307791296람다 값 : 0.37895013891945956



![png](output_52_887.png)


     15%|████████████                                                                   | 296/1948 [05:55<32:28,  1.18s/it]cv 값 : 0.7715258425604292람다 값 : 0.012507418358767085



![png](output_52_890.png)


     15%|████████████                                                                   | 297/1948 [05:57<31:44,  1.15s/it]cv 값 : 0.2894203730188542람다 값 : 0.8867588907212924



![png](output_52_893.png)


     15%|████████████                                                                   | 298/1948 [05:58<31:29,  1.15s/it]cv 값 : 0.2706544370541784람다 값 : 0.9334477512772623



![png](output_52_896.png)


     15%|████████████▏                                                                  | 299/1948 [05:59<31:05,  1.13s/it]cv 값 : 0.3816952957886776람다 값 : 0.5913654837721031



![png](output_52_899.png)


     15%|████████████▏                                                                  | 300/1948 [06:01<35:41,  1.30s/it]cv 값 : 0.9375349485762455람다 값 : 0.19310327902551377



![png](output_52_902.png)


     15%|████████████▏                                                                  | 301/1948 [06:02<34:18,  1.25s/it]cv 값 : 0.27884082993368714람다 값 : 0.01411638846004155



![png](output_52_905.png)


     16%|████████████▏                                                                  | 302/1948 [06:03<34:04,  1.24s/it]cv 값 : 0.4111733184442842람다 값 : 1.0875389336638932



![png](output_52_908.png)


     16%|████████████▎                                                                  | 303/1948 [06:04<32:56,  1.20s/it]cv 값 : 0.19787460535154577람다 값 : 1.7979830458005648



![png](output_52_911.png)


     16%|████████████▎                                                                  | 304/1948 [06:05<33:47,  1.23s/it]cv 값 : 0.4502474898862205람다 값 : 0.5956024997054635



![png](output_52_914.png)


     16%|████████████▎                                                                  | 305/1948 [06:06<32:24,  1.18s/it]cv 값 : 0.28094177913103람다 값 : -0.3069805757004179



![png](output_52_917.png)


     16%|████████████▍                                                                  | 306/1948 [06:08<32:51,  1.20s/it]cv 값 : 0.32316534020493476람다 값 : 0.03147635910013246



![png](output_52_920.png)


     16%|████████████▍                                                                  | 307/1948 [06:09<33:06,  1.21s/it]cv 값 : 0.4760537137736111람다 값 : 0.14944294052551874



![png](output_52_923.png)


     16%|████████████▍                                                                  | 308/1948 [06:10<33:12,  1.22s/it]cv 값 : 0.18396276317861956람다 값 : 1.0646117448832155



![png](output_52_926.png)


     16%|████████████▌                                                                  | 309/1948 [06:11<32:56,  1.21s/it]cv 값 : 0.2957588653488992람다 값 : 0.912433248160687



![png](output_52_929.png)


     16%|████████████▌                                                                  | 310/1948 [06:12<32:03,  1.17s/it]cv 값 : 0.6973884160262925람다 값 : 0.46942966412596493



![png](output_52_932.png)


     16%|████████████▌                                                                  | 311/1948 [06:14<32:46,  1.20s/it]cv 값 : 0.16325448171506884람다 값 : 0.6786664113903249



![png](output_52_935.png)


     16%|████████████▋                                                                  | 312/1948 [06:15<32:16,  1.18s/it]cv 값 : 0.36665560453703727람다 값 : 0.4956916316485693



![png](output_52_938.png)


     16%|████████████▋                                                                  | 313/1948 [06:16<32:29,  1.19s/it]cv 값 : 0.30008527670583335람다 값 : 0.2501454777917439



![png](output_52_941.png)


     16%|████████████▋                                                                  | 314/1948 [06:17<31:34,  1.16s/it]cv 값 : 0.2500782397467789람다 값 : 0.42479201433869374



![png](output_52_944.png)


     16%|████████████▊                                                                  | 315/1948 [06:18<32:14,  1.18s/it]cv 값 : 0.45497459289806225람다 값 : 0.3733502317849812



![png](output_52_947.png)


     16%|████████████▊                                                                  | 316/1948 [06:19<31:44,  1.17s/it]cv 값 : 0.610318009415629람다 값 : 0.5804323018987593



![png](output_52_950.png)


     16%|████████████▊                                                                  | 317/1948 [06:21<31:28,  1.16s/it]cv 값 : 0.5821072994709886람다 값 : 0.47636446386947445



![png](output_52_953.png)


     16%|████████████▉                                                                  | 318/1948 [06:22<32:32,  1.20s/it]cv 값 : 0.2634571444939174람다 값 : 0.7507443565868361



![png](output_52_956.png)


     16%|████████████▉                                                                  | 319/1948 [06:23<31:34,  1.16s/it]cv 값 : 0.32743579500768977람다 값 : -0.48390365442995026



![png](output_52_959.png)


     16%|████████████▉                                                                  | 320/1948 [06:24<31:46,  1.17s/it]cv 값 : 0.4852775808047713람다 값 : 0.2615912828522204



![png](output_52_962.png)


     16%|█████████████                                                                  | 321/1948 [06:25<31:38,  1.17s/it]cv 값 : 0.6106194916483408람다 값 : 0.3734384053679052



![png](output_52_965.png)


     17%|█████████████                                                                  | 322/1948 [06:27<32:48,  1.21s/it]cv 값 : 0.2377028883118382람다 값 : 0.5306614916801419



![png](output_52_968.png)


     17%|█████████████                                                                  | 323/1948 [06:28<32:00,  1.18s/it]cv 값 : 0.2548040758215667람다 값 : 0.5429287141413665



![png](output_52_971.png)


     17%|█████████████▏                                                                 | 324/1948 [06:29<31:59,  1.18s/it]cv 값 : 0.24154577161549493람다 값 : 0.49859738515063906



![png](output_52_974.png)


     17%|█████████████▏                                                                 | 325/1948 [06:30<30:37,  1.13s/it]cv 값 : 0.33800912194477306람다 값 : 0.06749553011304445



![png](output_52_977.png)


     17%|█████████████▏                                                                 | 326/1948 [06:31<31:45,  1.17s/it]cv 값 : 0.5126296292465772람다 값 : 0.5596630614890871



![png](output_52_980.png)


     17%|█████████████▎                                                                 | 327/1948 [06:32<30:46,  1.14s/it]cv 값 : 0.5054903665883099람다 값 : -0.16020864430315226



![png](output_52_983.png)


     17%|█████████████▎                                                                 | 328/1948 [06:33<30:04,  1.11s/it]cv 값 : 0.482483930125204람다 값 : 0.47048654033925413



![png](output_52_986.png)


     17%|█████████████▎                                                                 | 329/1948 [06:34<30:19,  1.12s/it]cv 값 : 0.3253807791688544람다 값 : -0.13893938957960705



![png](output_52_989.png)


     17%|█████████████▍                                                                 | 330/1948 [06:36<31:04,  1.15s/it]cv 값 : 0.17354023193860874람다 값 : 0.8193711201736067



![png](output_52_992.png)


     17%|█████████████▍                                                                 | 331/1948 [06:37<30:55,  1.15s/it]cv 값 : 0.44788701540477355람다 값 : 0.9644939413914646



![png](output_52_995.png)


     17%|█████████████▍                                                                 | 332/1948 [06:38<30:23,  1.13s/it]cv 값 : 0.36605728801985776람다 값 : 0.6656554843216518



![png](output_52_998.png)


     17%|█████████████▌                                                                 | 333/1948 [06:39<30:49,  1.15s/it]cv 값 : 0.3937571796666262람다 값 : 0.3383613497742235



![png](output_52_1001.png)


     17%|█████████████▌                                                                 | 334/1948 [06:40<30:40,  1.14s/it]cv 값 : 0.33650963971971415람다 값 : 0.5686147808366694



![png](output_52_1004.png)


     17%|█████████████▌                                                                 | 335/1948 [06:42<34:43,  1.29s/it]cv 값 : 0.1892890731715983람다 값 : 1.0470467069615421



![png](output_52_1007.png)


     17%|█████████████▋                                                                 | 336/1948 [06:43<37:08,  1.38s/it]cv 값 : 0.18833577144601954람다 값 : 1.106761376058177



![png](output_52_1010.png)


     17%|█████████████▋                                                                 | 337/1948 [06:45<35:03,  1.31s/it]cv 값 : 0.22190415916861256람다 값 : 0.5369514694032754



![png](output_52_1013.png)


     17%|█████████████▋                                                                 | 338/1948 [06:46<34:22,  1.28s/it]cv 값 : 0.6565650128310173람다 값 : 0.4297040663053209



![png](output_52_1016.png)


     17%|█████████████▋                                                                 | 339/1948 [06:47<34:30,  1.29s/it]cv 값 : 0.3646247503669486람다 값 : 0.615632110487315



![png](output_52_1019.png)


     17%|█████████████▊                                                                 | 340/1948 [06:48<33:12,  1.24s/it]cv 값 : 0.27598193336726384람다 값 : 0.3859434723535377



![png](output_52_1022.png)


     18%|█████████████▊                                                                 | 341/1948 [06:50<37:09,  1.39s/it]cv 값 : 0.23122876060197645람다 값 : -0.7059211901124646



![png](output_52_1025.png)


     18%|█████████████▊                                                                 | 342/1948 [06:52<41:22,  1.55s/it]cv 값 : 0.5551900479172172람다 값 : -0.34579052154847034



![png](output_52_1028.png)


     18%|█████████████▉                                                                 | 343/1948 [06:53<40:51,  1.53s/it]cv 값 : 0.25185707927306733람다 값 : 1.2472282623190725



![png](output_52_1031.png)


     18%|█████████████▉                                                                 | 344/1948 [06:55<42:53,  1.60s/it]cv 값 : 0.23995487928977444람다 값 : 1.205567866273549



![png](output_52_1034.png)


     18%|█████████████▉                                                                 | 345/1948 [06:56<39:52,  1.49s/it]cv 값 : 0.17868750394125713람다 값 : 0.6531696644350459



![png](output_52_1037.png)


     18%|██████████████                                                                 | 346/1948 [06:58<37:47,  1.42s/it]cv 값 : 0.24513007991886668람다 값 : -0.27283216024310214



![png](output_52_1040.png)


     18%|██████████████                                                                 | 347/1948 [06:59<37:20,  1.40s/it]cv 값 : 0.5860176401429393람다 값 : -1.1626688520658575



![png](output_52_1043.png)


     18%|██████████████                                                                 | 348/1948 [07:00<38:05,  1.43s/it]cv 값 : 0.19567247224970777람다 값 : 1.6114186145816471



![png](output_52_1046.png)


     18%|██████████████▏                                                                | 349/1948 [07:03<45:01,  1.69s/it]cv 값 : 0.5340378979793038람다 값 : 0.00963279920334463



![png](output_52_1049.png)


     18%|██████████████▏                                                                | 350/1948 [07:04<43:21,  1.63s/it]cv 값 : 0.18945508642158038람다 값 : 0.9256061550371026



![png](output_52_1052.png)


     18%|██████████████▏                                                                | 351/1948 [07:06<40:46,  1.53s/it]cv 값 : 0.45022128904945985람다 값 : 0.6217565244106873



![png](output_52_1055.png)


     18%|██████████████▎                                                                | 352/1948 [07:07<37:24,  1.41s/it]cv 값 : 0.46257274675921306람다 값 : 0.1256696855536241



![png](output_52_1058.png)


     18%|██████████████▎                                                                | 353/1948 [07:08<37:27,  1.41s/it]cv 값 : 0.6451876588165761람다 값 : 0.01091002479321813



![png](output_52_1061.png)


     18%|██████████████▎                                                                | 354/1948 [07:10<39:01,  1.47s/it]cv 값 : 0.32960233640731973람다 값 : 0.5375073586327558



![png](output_52_1064.png)


     18%|██████████████▍                                                                | 355/1948 [07:11<39:19,  1.48s/it]cv 값 : 0.4060306116691816람다 값 : 0.6787286489205524



![png](output_52_1067.png)


     18%|██████████████▍                                                                | 356/1948 [07:13<39:40,  1.50s/it]cv 값 : 0.4166484089902192람다 값 : 0.512810327418668



![png](output_52_1070.png)


     18%|██████████████▍                                                                | 357/1948 [07:14<40:27,  1.53s/it]cv 값 : 0.158408580635158람다 값 : 0.34957320426147354



![png](output_52_1073.png)


     18%|██████████████▌                                                                | 358/1948 [07:15<37:34,  1.42s/it]cv 값 : 0.3849387176735611람다 값 : 0.5095876534151618



![png](output_52_1076.png)


     18%|██████████████▌                                                                | 359/1948 [07:17<35:54,  1.36s/it]cv 값 : 0.2615306243427606람다 값 : 1.4587573226162716



![png](output_52_1079.png)


     18%|██████████████▌                                                                | 360/1948 [07:18<37:11,  1.41s/it]cv 값 : 0.5756764969791736람다 값 : -0.30490566501711636



![png](output_52_1082.png)


     19%|██████████████▋                                                                | 361/1948 [07:20<37:10,  1.41s/it]cv 값 : 0.327844949063746람다 값 : -0.37516787646844807



![png](output_52_1085.png)


     19%|██████████████▋                                                                | 362/1948 [07:21<37:04,  1.40s/it]cv 값 : 0.2960799472416867람다 값 : -0.504093530213954



![png](output_52_1088.png)


     19%|██████████████▋                                                                | 363/1948 [07:22<36:38,  1.39s/it]cv 값 : 0.8018786309345822람다 값 : 0.40269882752328334



![png](output_52_1091.png)


     19%|██████████████▊                                                                | 364/1948 [07:24<35:07,  1.33s/it]cv 값 : 0.2458353713123596람다 값 : 1.553015376488827



![png](output_52_1094.png)


     19%|██████████████▊                                                                | 365/1948 [07:25<35:07,  1.33s/it]cv 값 : 0.19852385934950775람다 값 : -0.4673944917088762



![png](output_52_1097.png)


     19%|██████████████▊                                                                | 366/1948 [07:26<35:25,  1.34s/it]cv 값 : 1.1782404895314116람다 값 : 0.12969458547054272



![png](output_52_1100.png)


     19%|██████████████▉                                                                | 367/1948 [07:28<36:10,  1.37s/it]cv 값 : 0.7920845432784315람다 값 : 0.3422153285756985



![png](output_52_1103.png)


     19%|██████████████▉                                                                | 368/1948 [07:29<35:41,  1.36s/it]cv 값 : 0.6219905823099992람다 값 : -1.1554822842164598



![png](output_52_1106.png)


     19%|██████████████▉                                                                | 369/1948 [07:30<34:14,  1.30s/it]cv 값 : 0.32859719917450536람다 값 : -0.27412334510964126



![png](output_52_1109.png)


     19%|███████████████                                                                | 370/1948 [07:32<34:26,  1.31s/it]cv 값 : 0.3173532870213449람다 값 : 0.7264086013782235



![png](output_52_1112.png)


     19%|███████████████                                                                | 371/1948 [07:33<34:16,  1.30s/it]cv 값 : 0.27661040301659434람다 값 : 0.8728846702151642



![png](output_52_1115.png)


     19%|███████████████                                                                | 372/1948 [07:34<33:42,  1.28s/it]cv 값 : 0.23643948527689992람다 값 : 0.21415999580349201



![png](output_52_1118.png)


     19%|███████████████▏                                                               | 373/1948 [07:35<31:48,  1.21s/it]cv 값 : 0.1836757988389676람다 값 : 0.874913293395937



![png](output_52_1121.png)


     19%|███████████████▏                                                               | 374/1948 [07:36<30:31,  1.16s/it]cv 값 : 0.24184937895961578람다 값 : 0.3261290035055414



![png](output_52_1124.png)


     19%|███████████████▏                                                               | 375/1948 [07:37<30:33,  1.17s/it]cv 값 : 0.35585022933519705람다 값 : 0.89329381557178



![png](output_52_1127.png)


     19%|███████████████▏                                                               | 376/1948 [07:38<28:52,  1.10s/it]cv 값 : 0.42491421357150433람다 값 : 0.7493974718147328



![png](output_52_1130.png)


     19%|███████████████▎                                                               | 377/1948 [07:39<28:45,  1.10s/it]cv 값 : 0.2209169559253726람다 값 : -0.29969133057963904



![png](output_52_1133.png)


     19%|███████████████▎                                                               | 378/1948 [07:41<30:05,  1.15s/it]cv 값 : 0.2619624500165879람다 값 : 0.5583313175556823



![png](output_52_1136.png)


     19%|███████████████▎                                                               | 379/1948 [07:42<30:18,  1.16s/it]cv 값 : 0.4083200311616929람다 값 : 0.09465015198583449



![png](output_52_1139.png)


     20%|███████████████▍                                                               | 380/1948 [07:43<31:01,  1.19s/it]cv 값 : 0.33585818322542227람다 값 : 1.1360673643296



![png](output_52_1142.png)


     20%|███████████████▍                                                               | 381/1948 [07:44<29:51,  1.14s/it]cv 값 : 0.28563392419011613람다 값 : 0.12791968984360536



![png](output_52_1145.png)


     20%|███████████████▍                                                               | 382/1948 [07:45<29:09,  1.12s/it]cv 값 : 0.16561596567013676람다 값 : 0.3974113638640883



![png](output_52_1148.png)


     20%|███████████████▌                                                               | 383/1948 [07:46<28:33,  1.09s/it]cv 값 : 0.28450286305570227람다 값 : 0.8130554160094186



![png](output_52_1151.png)


     20%|███████████████▌                                                               | 384/1948 [07:48<34:11,  1.31s/it]cv 값 : 0.4796572264171914람다 값 : -0.2738054922426663



![png](output_52_1154.png)


     20%|███████████████▌                                                               | 385/1948 [07:49<33:36,  1.29s/it]cv 값 : 0.384108675708595람다 값 : 0.5642766183333066



![png](output_52_1157.png)


     20%|███████████████▋                                                               | 386/1948 [07:51<33:42,  1.29s/it]cv 값 : 0.48826117596980784람다 값 : 0.2754137861555665



![png](output_52_1160.png)


     20%|███████████████▋                                                               | 387/1948 [07:52<32:11,  1.24s/it]cv 값 : 0.2797281294108032람다 값 : 0.24454837803933338



![png](output_52_1163.png)


     20%|███████████████▋                                                               | 388/1948 [07:53<32:49,  1.26s/it]cv 값 : 0.3087684470763985람다 값 : 0.5039189884839793



![png](output_52_1166.png)


     20%|███████████████▊                                                               | 389/1948 [07:54<32:20,  1.24s/it]cv 값 : 0.29876318398541113람다 값 : -0.6010633037487785



![png](output_52_1169.png)


     20%|███████████████▊                                                               | 390/1948 [07:56<33:00,  1.27s/it]cv 값 : 0.5502238009255438람다 값 : -0.33817634670085195



![png](output_52_1172.png)


     20%|███████████████▊                                                               | 391/1948 [07:57<32:04,  1.24s/it]cv 값 : 0.1978821240481665람다 값 : -0.3382294292881996



![png](output_52_1175.png)


     20%|███████████████▉                                                               | 392/1948 [07:58<31:14,  1.20s/it]cv 값 : 0.1636070754909854람다 값 : 2.3697525698387674



![png](output_52_1178.png)


     20%|███████████████▉                                                               | 393/1948 [07:59<31:55,  1.23s/it]cv 값 : 0.2660203198071597람다 값 : -0.2588716050705818



![png](output_52_1181.png)


     20%|███████████████▉                                                               | 394/1948 [08:00<31:13,  1.21s/it]cv 값 : 0.30337478812504776람다 값 : 0.9059582414809295



![png](output_52_1184.png)


     20%|████████████████                                                               | 395/1948 [08:02<34:20,  1.33s/it]cv 값 : 0.472283270356851람다 값 : 0.18701661356017163



![png](output_52_1187.png)


     20%|████████████████                                                               | 396/1948 [08:04<44:04,  1.70s/it]cv 값 : 0.49922499619102906람다 값 : -0.11368794472207781



![png](output_52_1190.png)


     20%|████████████████                                                               | 397/1948 [08:08<56:47,  2.20s/it]cv 값 : 0.23427789021406506람다 값 : 0.7087807907842092



![png](output_52_1193.png)


     20%|███████████████▋                                                             | 398/1948 [08:11<1:01:47,  2.39s/it]cv 값 : 0.7447271252727931람다 값 : 0.3422941429592868



![png](output_52_1196.png)


     20%|████████████████▏                                                              | 399/1948 [08:12<54:21,  2.11s/it]cv 값 : 0.35222067675603475람다 값 : 0.4730485131049639



![png](output_52_1199.png)


     21%|███████████████▊                                                             | 400/1948 [08:16<1:08:07,  2.64s/it]cv 값 : 0.39870289756173366람다 값 : 0.8941054648500876



![png](output_52_1202.png)


     21%|███████████████▊                                                             | 401/1948 [08:18<1:07:12,  2.61s/it]cv 값 : 0.2929600432551334람다 값 : 0.6905734023877159



![png](output_52_1205.png)


     21%|████████████████▎                                                              | 402/1948 [08:20<57:06,  2.22s/it]cv 값 : 0.6214805088813288람다 값 : -0.23657033382117185



![png](output_52_1208.png)


     21%|████████████████▎                                                              | 403/1948 [08:21<50:38,  1.97s/it]cv 값 : 0.2971479634176055람다 값 : 0.5694677590258316



![png](output_52_1211.png)


     21%|████████████████▍                                                              | 404/1948 [08:23<48:08,  1.87s/it]cv 값 : 0.24736575887466544람다 값 : 1.4436498829847288



![png](output_52_1214.png)


     21%|████████████████▍                                                              | 405/1948 [08:24<46:28,  1.81s/it]cv 값 : 0.38819579981312724람다 값 : 0.40410378676516934



![png](output_52_1217.png)


     21%|████████████████▍                                                              | 406/1948 [08:26<44:10,  1.72s/it]cv 값 : 0.17318274096965244람다 값 : 0.16267954446697835



![png](output_52_1220.png)


     21%|████████████████▌                                                              | 407/1948 [08:27<39:59,  1.56s/it]cv 값 : 0.42584328388438225람다 값 : -0.016487327533490713



![png](output_52_1223.png)


     21%|████████████████▌                                                              | 408/1948 [08:29<38:28,  1.50s/it]cv 값 : 0.33047176209765006람다 값 : 0.39207511136194223



![png](output_52_1226.png)


     21%|████████████████▌                                                              | 409/1948 [08:30<36:45,  1.43s/it]cv 값 : 0.23979863714979097람다 값 : -0.3189816361876477



![png](output_52_1229.png)


     21%|████████████████▋                                                              | 410/1948 [08:31<35:31,  1.39s/it]cv 값 : 0.2748986739382349람다 값 : -0.8473526257170887



![png](output_52_1232.png)


     21%|████████████████▋                                                              | 411/1948 [08:33<40:58,  1.60s/it]cv 값 : 0.37754278412996745람다 값 : 0.44966988907811695



![png](output_52_1235.png)


     21%|████████████████▋                                                              | 412/1948 [08:37<54:20,  2.12s/it]cv 값 : 0.2543608865841283람다 값 : 1.1395129955495023



![png](output_52_1238.png)


     21%|████████████████▎                                                            | 413/1948 [08:40<1:03:27,  2.48s/it]cv 값 : 0.22227878478680102람다 값 : 0.39016499713279873



![png](output_52_1241.png)


     21%|████████████████▎                                                            | 414/1948 [08:43<1:07:37,  2.65s/it]cv 값 : 0.14125045979674308람다 값 : 1.2738344722724302



![png](output_52_1244.png)


     21%|████████████████▍                                                            | 415/1948 [08:47<1:16:54,  3.01s/it]cv 값 : 0.6156024970288808람다 값 : 0.6334482177345032



![png](output_52_1247.png)


     21%|████████████████▍                                                            | 416/1948 [08:50<1:21:57,  3.21s/it]cv 값 : 0.5935576430405625람다 값 : 0.4489154081905361



![png](output_52_1250.png)


     21%|████████████████▍                                                            | 417/1948 [08:55<1:34:30,  3.70s/it]cv 값 : 0.32177831267709284람다 값 : -0.8271073671737352



![png](output_52_1253.png)


     21%|████████████████▌                                                            | 418/1948 [08:57<1:20:35,  3.16s/it]cv 값 : 0.30766571377748453람다 값 : 0.5190293301606403



![png](output_52_1256.png)


     22%|████████████████▌                                                            | 419/1948 [08:59<1:09:31,  2.73s/it]cv 값 : 0.24007201945426787람다 값 : 1.3349515272157388



![png](output_52_1259.png)


     22%|████████████████▌                                                            | 420/1948 [09:01<1:01:52,  2.43s/it]cv 값 : 0.37181340007945574람다 값 : -0.059398561744332765



![png](output_52_1262.png)


     22%|█████████████████                                                              | 421/1948 [09:02<54:47,  2.15s/it]cv 값 : 0.2390787997599226람다 값 : 0.9480496787262943



![png](output_52_1265.png)


     22%|█████████████████                                                              | 422/1948 [09:04<50:59,  2.01s/it]cv 값 : 0.36785715789061296람다 값 : 1.1464886106895995



![png](output_52_1268.png)


     22%|█████████████████▏                                                             | 423/1948 [09:06<51:19,  2.02s/it]cv 값 : 0.49202092325530256람다 값 : 0.596653794759422



![png](output_52_1271.png)


     22%|█████████████████▏                                                             | 424/1948 [09:08<55:09,  2.17s/it]cv 값 : 0.27256418078086025람다 값 : -0.7378358552736919



![png](output_52_1274.png)


     22%|█████████████████▏                                                             | 425/1948 [09:11<57:03,  2.25s/it]cv 값 : 0.46049861169918815람다 값 : 0.8553967390183239



![png](output_52_1277.png)


     22%|█████████████████▎                                                             | 426/1948 [09:13<53:12,  2.10s/it]cv 값 : 0.22934057663037427람다 값 : -0.4305466086464703



![png](output_52_1280.png)


     22%|█████████████████▎                                                             | 427/1948 [09:14<47:25,  1.87s/it]cv 값 : 0.8569511254444833람다 값 : 0.3642909916338478



![png](output_52_1283.png)


     22%|█████████████████▎                                                             | 428/1948 [09:15<44:34,  1.76s/it]cv 값 : 0.23494740450028434람다 값 : -0.2842491054080461



![png](output_52_1286.png)


     22%|█████████████████▍                                                             | 429/1948 [09:17<40:21,  1.59s/it]cv 값 : 0.5432011603053857람다 값 : 0.5737275766157016



![png](output_52_1289.png)


     22%|█████████████████▍                                                             | 430/1948 [09:18<40:28,  1.60s/it]cv 값 : 0.5004299887063417람다 값 : 0.5919404307056856



![png](output_52_1292.png)


     22%|█████████████████▍                                                             | 431/1948 [09:22<58:31,  2.31s/it]cv 값 : 0.357531533531594람다 값 : 0.4397553656627553



![png](output_52_1295.png)


     22%|█████████████████▌                                                             | 432/1948 [09:24<55:50,  2.21s/it]cv 값 : 0.2317998927594879람다 값 : -0.8653756991796807



![png](output_52_1298.png)


     22%|█████████████████▌                                                             | 433/1948 [09:26<54:04,  2.14s/it]cv 값 : 0.3969536471414912람다 값 : -0.4741697884541348



![png](output_52_1301.png)


     22%|█████████████████▌                                                             | 434/1948 [09:28<52:06,  2.06s/it]cv 값 : 0.1484329466237372람다 값 : 1.9701615783274968



![png](output_52_1304.png)


     22%|█████████████████▋                                                             | 435/1948 [09:29<45:10,  1.79s/it]cv 값 : 0.5235575932335659람다 값 : 0.7311184585447468



![png](output_52_1307.png)


     22%|█████████████████▋                                                             | 436/1948 [09:30<40:50,  1.62s/it]cv 값 : 0.3740707779705521람다 값 : 0.5589989469522517



![png](output_52_1310.png)


     22%|█████████████████▋                                                             | 437/1948 [09:31<36:58,  1.47s/it]cv 값 : 0.1777471820494683람다 값 : 0.09945231093464213



![png](output_52_1313.png)


     22%|█████████████████▊                                                             | 438/1948 [09:33<36:21,  1.44s/it]cv 값 : 0.9196535263292483람다 값 : 0.3635686159626158



![png](output_52_1316.png)


     23%|█████████████████▊                                                             | 439/1948 [09:34<34:21,  1.37s/it]cv 값 : 0.3634283012701267람다 값 : 0.411992402471957



![png](output_52_1319.png)


     23%|█████████████████▊                                                             | 440/1948 [09:35<33:30,  1.33s/it]cv 값 : 0.19216883710682583람다 값 : 0.8322783822710613



![png](output_52_1322.png)


     23%|█████████████████▉                                                             | 441/1948 [09:37<32:46,  1.31s/it]cv 값 : 0.25859917235006474람다 값 : 0.5692901065513166



![png](output_52_1325.png)


     23%|█████████████████▉                                                             | 442/1948 [09:38<33:39,  1.34s/it]cv 값 : 0.3780663376850014람다 값 : 0.435203545525419



![png](output_52_1328.png)


     23%|█████████████████▉                                                             | 443/1948 [09:39<32:32,  1.30s/it]cv 값 : 0.26914789879060586람다 값 : 0.857293420804035



![png](output_52_1331.png)


     23%|██████████████████                                                             | 444/1948 [09:40<31:08,  1.24s/it]cv 값 : 0.5556399080785822람다 값 : 0.057182820521817286



![png](output_52_1334.png)


     23%|██████████████████                                                             | 445/1948 [09:41<29:33,  1.18s/it]cv 값 : 0.3751597864494626람다 값 : 0.6714107648317674



![png](output_52_1337.png)


     23%|██████████████████                                                             | 446/1948 [09:43<29:41,  1.19s/it]cv 값 : 0.17326710260670572람다 값 : -0.15579691397368856



![png](output_52_1340.png)


     23%|██████████████████▏                                                            | 447/1948 [09:44<28:46,  1.15s/it]cv 값 : 0.36397033727840894람다 값 : 0.21684092691618811



![png](output_52_1343.png)


     23%|██████████████████▏                                                            | 448/1948 [09:45<27:39,  1.11s/it]cv 값 : 0.5524699074308745람다 값 : 0.4210276338206611



![png](output_52_1346.png)


     23%|██████████████████▏                                                            | 449/1948 [09:46<27:23,  1.10s/it]cv 값 : 0.3198112179145923람다 값 : 1.0162329300541868



![png](output_52_1349.png)


     23%|██████████████████▏                                                            | 450/1948 [09:47<27:50,  1.11s/it]cv 값 : 0.9851130673396934람다 값 : 0.053080677005733566



![png](output_52_1352.png)


     23%|██████████████████▎                                                            | 451/1948 [09:48<26:42,  1.07s/it]cv 값 : 0.30643729055778646람다 값 : 0.6887621992182225



![png](output_52_1355.png)


     23%|██████████████████▎                                                            | 452/1948 [09:49<26:21,  1.06s/it]cv 값 : 1.0376380358138784람다 값 : -0.42404396021081603



![png](output_52_1358.png)


     23%|██████████████████▎                                                            | 453/1948 [09:50<26:38,  1.07s/it]cv 값 : 0.30787981817509746람다 값 : 0.529209019422105



![png](output_52_1361.png)


     23%|██████████████████▍                                                            | 454/1948 [09:51<26:34,  1.07s/it]cv 값 : 0.33559882954125453람다 값 : 0.03311686694023771



![png](output_52_1364.png)


     23%|██████████████████▍                                                            | 455/1948 [09:52<27:25,  1.10s/it]cv 값 : 0.26019945615290474람다 값 : 0.9914928738434101



![png](output_52_1367.png)


     23%|██████████████████▍                                                            | 456/1948 [09:53<27:00,  1.09s/it]cv 값 : 0.5295984521425261람다 값 : 0.6023337298695477



![png](output_52_1370.png)


     23%|██████████████████▌                                                            | 457/1948 [09:54<26:49,  1.08s/it]cv 값 : 0.32340000921689416람다 값 : 0.8500337935863026



![png](output_52_1373.png)


     24%|██████████████████▌                                                            | 458/1948 [09:55<26:46,  1.08s/it]cv 값 : 0.16353756040880166람다 값 : 0.47011737240404683



![png](output_52_1376.png)


     24%|██████████████████▌                                                            | 459/1948 [09:56<26:58,  1.09s/it]cv 값 : 0.32375320701762694람다 값 : 0.16153892212567988



![png](output_52_1379.png)


     24%|██████████████████▋                                                            | 460/1948 [09:57<26:06,  1.05s/it]cv 값 : 0.2856140082523316람다 값 : 0.478562791685813



![png](output_52_1382.png)


     24%|██████████████████▋                                                            | 461/1948 [09:58<25:38,  1.03s/it]cv 값 : 0.16417576288760072람다 값 : -0.7498339864384352



![png](output_52_1385.png)


     24%|██████████████████▋                                                            | 462/1948 [09:59<25:33,  1.03s/it]cv 값 : 0.35559326387184176람다 값 : 0.8183672551141142



![png](output_52_1388.png)


     24%|██████████████████▊                                                            | 463/1948 [10:00<25:03,  1.01s/it]cv 값 : 0.3810079231749921람다 값 : 0.8244398627948176



![png](output_52_1391.png)


     24%|██████████████████▊                                                            | 464/1948 [10:02<26:08,  1.06s/it]cv 값 : 0.3532156081982833람다 값 : 0.28208116468896755



![png](output_52_1394.png)


     24%|██████████████████▊                                                            | 465/1948 [10:03<26:24,  1.07s/it]cv 값 : 0.568625743068814람다 값 : 0.43106270907322036



![png](output_52_1397.png)


     24%|██████████████████▉                                                            | 466/1948 [10:04<26:10,  1.06s/it]cv 값 : 0.5054475487698865람다 값 : -0.10278808545058502



![png](output_52_1400.png)


     24%|██████████████████▉                                                            | 467/1948 [10:05<25:55,  1.05s/it]cv 값 : 0.5140436632756646람다 값 : 0.7749718399601747



![png](output_52_1403.png)


     24%|██████████████████▉                                                            | 468/1948 [10:06<26:14,  1.06s/it]cv 값 : 0.22121486483939626람다 값 : -0.18141152645900072



![png](output_52_1406.png)


     24%|███████████████████                                                            | 469/1948 [10:07<25:38,  1.04s/it]cv 값 : 0.386723879592842람다 값 : 0.5506094976337008



![png](output_52_1409.png)


     24%|███████████████████                                                            | 470/1948 [10:08<24:52,  1.01s/it]cv 값 : 1.430950068058488람다 값 : 0.03541648607328731



![png](output_52_1412.png)


     24%|███████████████████                                                            | 471/1948 [10:09<24:44,  1.00s/it]cv 값 : 0.34666297906496857람다 값 : 0.8479027528679074



![png](output_52_1415.png)


     24%|███████████████████▏                                                           | 472/1948 [10:10<24:17,  1.01it/s]cv 값 : 0.36486164320886405람다 값 : 0.5528838595216164



![png](output_52_1418.png)


     24%|███████████████████▏                                                           | 473/1948 [10:11<24:16,  1.01it/s]cv 값 : 0.58664771410773람다 값 : 0.1793987038592766



![png](output_52_1421.png)


     24%|███████████████████▏                                                           | 474/1948 [10:12<24:37,  1.00s/it]cv 값 : 0.33434377107981944람다 값 : 0.18574431626664442



![png](output_52_1424.png)


     24%|███████████████████▎                                                           | 475/1948 [10:13<24:38,  1.00s/it]cv 값 : 0.26811273507265243람다 값 : 1.0517700274095503



![png](output_52_1427.png)


     24%|███████████████████▎                                                           | 476/1948 [10:14<24:42,  1.01s/it]cv 값 : 0.4252651811957129람다 값 : 0.6135709242146696



![png](output_52_1430.png)


     24%|███████████████████▎                                                           | 477/1948 [10:15<25:06,  1.02s/it]cv 값 : 0.1505061171518471람다 값 : 0.9264071286849628



![png](output_52_1433.png)


     25%|███████████████████▍                                                           | 478/1948 [10:16<25:04,  1.02s/it]cv 값 : 0.5631179156942424람다 값 : 0.5195768916286682



![png](output_52_1436.png)


     25%|███████████████████▍                                                           | 479/1948 [10:17<25:15,  1.03s/it]cv 값 : 0.41991695146766517람다 값 : -0.5646652654868044



![png](output_52_1439.png)


     25%|███████████████████▍                                                           | 480/1948 [10:18<25:43,  1.05s/it]cv 값 : 0.26232706590570026람다 값 : -0.9923975327241291



![png](output_52_1442.png)


     25%|███████████████████▌                                                           | 481/1948 [10:19<25:20,  1.04s/it]cv 값 : 0.3376194411228351람다 값 : -0.036713983025109635



![png](output_52_1445.png)


     25%|███████████████████▌                                                           | 482/1948 [10:20<26:27,  1.08s/it]cv 값 : 0.3307752496119294람다 값 : 0.2157276356691524



![png](output_52_1448.png)


     25%|███████████████████▌                                                           | 483/1948 [10:21<28:03,  1.15s/it]cv 값 : 0.34836764911300994람다 값 : -0.8733655038785575



![png](output_52_1451.png)


     25%|███████████████████▋                                                           | 484/1948 [10:23<29:05,  1.19s/it]cv 값 : 1.1329522318783665람다 값 : -0.4970757805288805



![png](output_52_1454.png)


     25%|███████████████████▋                                                           | 485/1948 [10:24<29:20,  1.20s/it]cv 값 : 0.6213151615011386람다 값 : 0.7447606962781277



![png](output_52_1457.png)


     25%|███████████████████▋                                                           | 486/1948 [10:25<29:55,  1.23s/it]cv 값 : 0.5475947026253325람다 값 : 0.09943847979279091



![png](output_52_1460.png)


     25%|███████████████████▊                                                           | 487/1948 [10:26<28:23,  1.17s/it]cv 값 : 0.43436397645088803람다 값 : 0.36284308803723486



![png](output_52_1463.png)


     25%|███████████████████▊                                                           | 488/1948 [10:27<27:40,  1.14s/it]cv 값 : 0.22287824228570163람다 값 : -1.0424360676677447



![png](output_52_1466.png)


     25%|███████████████████▊                                                           | 489/1948 [10:29<28:31,  1.17s/it]cv 값 : 0.35578056912346806람다 값 : 0.25162296439047105



![png](output_52_1469.png)


     25%|███████████████████▊                                                           | 490/1948 [10:30<29:21,  1.21s/it]cv 값 : 0.2916215763678723람다 값 : 0.33875943694417704



![png](output_52_1472.png)


     25%|███████████████████▉                                                           | 491/1948 [10:31<28:38,  1.18s/it]cv 값 : 0.13836144934553854람다 값 : -0.055779425447750657



![png](output_52_1475.png)


     25%|███████████████████▉                                                           | 492/1948 [10:32<28:28,  1.17s/it]cv 값 : 0.2619891058636556람다 값 : 0.4457072984974666



![png](output_52_1478.png)


     25%|███████████████████▉                                                           | 493/1948 [10:33<28:37,  1.18s/it]cv 값 : 0.406010228396768람다 값 : 0.23942489602218975



![png](output_52_1481.png)


     25%|████████████████████                                                           | 494/1948 [10:35<28:47,  1.19s/it]cv 값 : 1.0310973401746195람다 값 : 0.031937176558948545



![png](output_52_1484.png)


     25%|████████████████████                                                           | 495/1948 [10:36<29:28,  1.22s/it]cv 값 : 0.3681001497343538람다 값 : 0.32897554882081576



![png](output_52_1487.png)


     25%|████████████████████                                                           | 496/1948 [10:37<28:11,  1.16s/it]cv 값 : 0.24915071300253133람다 값 : -0.585459805485134



![png](output_52_1490.png)


     26%|████████████████████▏                                                          | 497/1948 [10:38<28:31,  1.18s/it]cv 값 : 0.17734070638941268람다 값 : 0.4269158837174803



![png](output_52_1493.png)


     26%|████████████████████▏                                                          | 498/1948 [10:39<29:34,  1.22s/it]cv 값 : 0.3212183115514529람다 값 : -0.32695154493075596



![png](output_52_1496.png)


     26%|████████████████████▏                                                          | 499/1948 [10:41<30:38,  1.27s/it]cv 값 : 0.23080153245542614람다 값 : 0.6460893478374897



![png](output_52_1499.png)


     26%|████████████████████▎                                                          | 500/1948 [10:42<30:06,  1.25s/it]cv 값 : 1.50399885495842람다 값 : 0.19525542098981408



![png](output_52_1502.png)


     26%|████████████████████▎                                                          | 501/1948 [10:43<30:41,  1.27s/it]cv 값 : 0.1801733886278429람다 값 : -2.0902259934159684



![png](output_52_1505.png)


     26%|████████████████████▎                                                          | 502/1948 [10:45<30:15,  1.26s/it]cv 값 : 0.26324764951883683람다 값 : 0.2952768958282295



![png](output_52_1508.png)


     26%|████████████████████▍                                                          | 503/1948 [10:46<29:49,  1.24s/it]cv 값 : 0.6840767213958365람다 값 : 0.3839258083576604



![png](output_52_1511.png)


     26%|████████████████████▍                                                          | 504/1948 [10:47<29:30,  1.23s/it]cv 값 : 0.19779778050339544람다 값 : 1.3575371408265908



![png](output_52_1514.png)


     26%|████████████████████▍                                                          | 505/1948 [10:48<29:16,  1.22s/it]cv 값 : 0.5625939429104861람다 값 : 0.5616094855733176



![png](output_52_1517.png)


     26%|████████████████████▌                                                          | 506/1948 [10:49<29:27,  1.23s/it]cv 값 : 0.6234259567583974람다 값 : 0.009040798718113889



![png](output_52_1520.png)


     26%|████████████████████▌                                                          | 507/1948 [10:51<29:38,  1.23s/it]cv 값 : 0.3466508279647387람다 값 : 0.8181121516494554



![png](output_52_1523.png)


     26%|████████████████████▌                                                          | 508/1948 [10:52<31:34,  1.32s/it]cv 값 : 0.36730391107217963람다 값 : 0.24955496423523593



![png](output_52_1526.png)


     26%|████████████████████▋                                                          | 509/1948 [10:54<32:17,  1.35s/it]cv 값 : 0.3211060292628289람다 값 : -0.018303215631412934



![png](output_52_1529.png)


     26%|████████████████████▋                                                          | 510/1948 [10:55<33:30,  1.40s/it]cv 값 : 0.6203175537653498람다 값 : 0.49955352689171534



![png](output_52_1532.png)


     26%|████████████████████▋                                                          | 511/1948 [10:56<32:47,  1.37s/it]cv 값 : 0.47128425664502616람다 값 : 0.5796231093375805



![png](output_52_1535.png)


     26%|████████████████████▊                                                          | 512/1948 [10:58<33:47,  1.41s/it]cv 값 : 0.4101947341143955람다 값 : 0.2835055828150852



![png](output_52_1538.png)


     26%|████████████████████▊                                                          | 513/1948 [10:59<34:13,  1.43s/it]cv 값 : 0.40180940964658846람다 값 : 0.6490706563937613



![png](output_52_1541.png)


     26%|████████████████████▊                                                          | 514/1948 [11:01<33:21,  1.40s/it]cv 값 : 0.29306180925770414람다 값 : 0.025097309458296888



![png](output_52_1544.png)


     26%|████████████████████▉                                                          | 515/1948 [11:02<33:02,  1.38s/it]cv 값 : 0.3530378147112477람다 값 : 1.5512018520850719



![png](output_52_1547.png)


     26%|████████████████████▉                                                          | 516/1948 [11:03<33:04,  1.39s/it]cv 값 : 0.3555776070030091람다 값 : 0.5570589017079299



![png](output_52_1550.png)


     27%|████████████████████▉                                                          | 517/1948 [11:05<32:47,  1.38s/it]cv 값 : 0.6219272434089801람다 값 : 0.5465269681836291



![png](output_52_1553.png)


     27%|█████████████████████                                                          | 518/1948 [11:06<32:49,  1.38s/it]cv 값 : 0.310086106071451람다 값 : -0.35034630179703946



![png](output_52_1556.png)


     27%|█████████████████████                                                          | 519/1948 [11:08<32:30,  1.36s/it]cv 값 : 0.26009248205381236람다 값 : -0.22496550141886879



![png](output_52_1559.png)


     27%|█████████████████████                                                          | 520/1948 [11:09<30:56,  1.30s/it]cv 값 : 0.21759698636263675람다 값 : 0.466493555482529



![png](output_52_1562.png)


     27%|█████████████████████▏                                                         | 521/1948 [11:10<31:21,  1.32s/it]cv 값 : 0.39494519101473785람다 값 : 0.8597623069818898



![png](output_52_1565.png)


     27%|█████████████████████▏                                                         | 522/1948 [11:11<30:05,  1.27s/it]cv 값 : 0.24140138040157463람다 값 : -0.002458973271135502



![png](output_52_1568.png)


     27%|█████████████████████▏                                                         | 523/1948 [11:12<29:02,  1.22s/it]cv 값 : 0.4344279190120289람다 값 : -0.58033338151389



![png](output_52_1571.png)


     27%|█████████████████████▎                                                         | 524/1948 [11:14<29:18,  1.23s/it]cv 값 : 0.2829091816037967람다 값 : 0.11902507563836567



![png](output_52_1574.png)


     27%|█████████████████████▎                                                         | 525/1948 [11:15<29:06,  1.23s/it]cv 값 : 0.3843554022983119람다 값 : 0.23308737874596414



![png](output_52_1577.png)


     27%|█████████████████████▎                                                         | 526/1948 [11:16<29:38,  1.25s/it]cv 값 : 0.352578995510943람다 값 : 0.20054489810679849



![png](output_52_1580.png)


     27%|█████████████████████▎                                                         | 527/1948 [11:17<29:01,  1.23s/it]cv 값 : 0.5948754154264648람다 값 : 0.2831767640479662



![png](output_52_1583.png)


     27%|█████████████████████▍                                                         | 528/1948 [11:18<28:47,  1.22s/it]cv 값 : 0.2551098506048002람다 값 : 0.004060168601274217



![png](output_52_1586.png)


     27%|█████████████████████▍                                                         | 529/1948 [11:20<28:35,  1.21s/it]cv 값 : 0.11845737237726776람다 값 : -0.2527609198083406



![png](output_52_1589.png)


     27%|█████████████████████▍                                                         | 530/1948 [11:21<29:16,  1.24s/it]cv 값 : 0.4952911939080628람다 값 : 0.3356347225731545



![png](output_52_1592.png)


     27%|█████████████████████▌                                                         | 531/1948 [11:22<28:04,  1.19s/it]cv 값 : 0.5491943205992676람다 값 : -0.03499509942088769



![png](output_52_1595.png)


     27%|█████████████████████▌                                                         | 532/1948 [11:23<28:20,  1.20s/it]cv 값 : 0.1844542839594727람다 값 : 1.1513053063385321



![png](output_52_1598.png)


     27%|█████████████████████▌                                                         | 533/1948 [11:24<27:45,  1.18s/it]cv 값 : 0.24241614099118106람다 값 : -0.33984748786261826



![png](output_52_1601.png)


     27%|█████████████████████▋                                                         | 534/1948 [11:26<27:51,  1.18s/it]cv 값 : 0.6227096351341234람다 값 : -1.1133816018526226



![png](output_52_1604.png)


     27%|█████████████████████▋                                                         | 535/1948 [11:27<29:46,  1.26s/it]cv 값 : 0.6749436572162206람다 값 : -0.17419885563051007



![png](output_52_1607.png)


     28%|█████████████████████▋                                                         | 536/1948 [11:29<31:42,  1.35s/it]cv 값 : 0.4173524909785516람다 값 : 0.7720067798210218



![png](output_52_1610.png)


     28%|█████████████████████▊                                                         | 537/1948 [11:30<32:01,  1.36s/it]cv 값 : 0.25854981863424986람다 값 : -0.6739068462798907



![png](output_52_1613.png)


     28%|█████████████████████▊                                                         | 538/1948 [11:31<30:57,  1.32s/it]cv 값 : 0.5361118513069296람다 값 : 0.5776440406497397



![png](output_52_1616.png)


     28%|█████████████████████▊                                                         | 539/1948 [11:33<31:24,  1.34s/it]cv 값 : 0.36720903140568534람다 값 : 0.20171427871564085



![png](output_52_1619.png)


     28%|█████████████████████▉                                                         | 540/1948 [11:34<30:08,  1.28s/it]cv 값 : 0.2584368743533557람다 값 : -0.20791693896382576



![png](output_52_1622.png)


     28%|█████████████████████▉                                                         | 541/1948 [11:35<30:41,  1.31s/it]cv 값 : 0.4704633105387674람다 값 : -0.42507352038356017



![png](output_52_1625.png)


     28%|█████████████████████▉                                                         | 542/1948 [11:36<29:31,  1.26s/it]cv 값 : 0.19069126262083372람다 값 : 1.228213793989711



![png](output_52_1628.png)


     28%|██████████████████████                                                         | 543/1948 [11:38<30:12,  1.29s/it]cv 값 : 0.5858655047463464람다 값 : -1.1624030669207333



![png](output_52_1631.png)


     28%|██████████████████████                                                         | 544/1948 [11:39<29:31,  1.26s/it]cv 값 : 0.6203343765046353람다 값 : 0.32196773913326554



![png](output_52_1634.png)


     28%|██████████████████████                                                         | 545/1948 [11:40<28:51,  1.23s/it]cv 값 : 0.15640811983439357람다 값 : 0.5512980018543183



![png](output_52_1637.png)


     28%|██████████████████████▏                                                        | 546/1948 [11:41<27:51,  1.19s/it]cv 값 : 0.25562701189907505람다 값 : 1.6705772524388665



![png](output_52_1640.png)


     28%|██████████████████████▏                                                        | 547/1948 [11:42<28:18,  1.21s/it]cv 값 : 0.35471091314313974람다 값 : -0.4943978881605685



![png](output_52_1643.png)


     28%|██████████████████████▏                                                        | 548/1948 [11:44<29:18,  1.26s/it]cv 값 : 0.2772129255613454람다 값 : 2.037427100034738



![png](output_52_1646.png)


     28%|██████████████████████▎                                                        | 549/1948 [11:45<29:20,  1.26s/it]cv 값 : 0.37654530293405925람다 값 : 0.6724199692930687



![png](output_52_1649.png)


     28%|██████████████████████▎                                                        | 550/1948 [11:46<27:38,  1.19s/it]cv 값 : 0.789226212767047람다 값 : 0.20202008999443083



![png](output_52_1652.png)


     28%|██████████████████████▎                                                        | 551/1948 [11:47<27:17,  1.17s/it]cv 값 : 0.21561766517051964람다 값 : 1.5117938944139842



![png](output_52_1655.png)


     28%|██████████████████████▍                                                        | 552/1948 [11:48<27:11,  1.17s/it]cv 값 : 0.38678279265997956람다 값 : -0.3251290334362096



![png](output_52_1658.png)


     28%|██████████████████████▍                                                        | 553/1948 [11:50<28:18,  1.22s/it]cv 값 : 0.3739839155651857람다 값 : 0.3725459433691947



![png](output_52_1661.png)


     28%|██████████████████████▍                                                        | 554/1948 [11:51<30:33,  1.32s/it]cv 값 : 0.26632823108735176람다 값 : 0.4303114101804503



![png](output_52_1664.png)


     28%|██████████████████████▌                                                        | 555/1948 [11:52<29:32,  1.27s/it]cv 값 : 0.2581810436176426람다 값 : 0.8122930243162532



![png](output_52_1667.png)


     29%|██████████████████████▌                                                        | 556/1948 [11:54<29:34,  1.27s/it]cv 값 : 0.27703724678340547람다 값 : -0.046266700676586316



![png](output_52_1670.png)


     29%|██████████████████████▌                                                        | 557/1948 [11:55<30:24,  1.31s/it]cv 값 : 0.17803874169337594람다 값 : -0.44596091860998427



![png](output_52_1673.png)


     29%|██████████████████████▋                                                        | 558/1948 [11:56<31:02,  1.34s/it]cv 값 : 0.2518366046447163람다 값 : 1.5298538280967109



![png](output_52_1676.png)


     29%|██████████████████████▋                                                        | 559/1948 [11:58<35:18,  1.53s/it]cv 값 : 0.5011367625061142람다 값 : 0.5903897344009076



![png](output_52_1679.png)


     29%|██████████████████████▋                                                        | 560/1948 [11:59<32:33,  1.41s/it]cv 값 : 0.4891742717787669람다 값 : 0.5295019168374672



![png](output_52_1682.png)


     29%|██████████████████████▊                                                        | 561/1948 [12:01<30:07,  1.30s/it]cv 값 : 0.3970064757408384람다 값 : 1.1821806538357063



![png](output_52_1685.png)


     29%|██████████████████████▊                                                        | 562/1948 [12:02<29:16,  1.27s/it]cv 값 : 0.5904226689796032람다 값 : -0.18055545978749235



![png](output_52_1688.png)


     29%|██████████████████████▊                                                        | 563/1948 [12:03<30:42,  1.33s/it]cv 값 : 0.14322953760441903람다 값 : 1.2989557138254337



![png](output_52_1691.png)


     29%|██████████████████████▊                                                        | 564/1948 [12:05<34:05,  1.48s/it]cv 값 : 0.7164032037636768람다 값 : 0.2502031881389788



![png](output_52_1694.png)


     29%|██████████████████████▉                                                        | 565/1948 [12:07<35:32,  1.54s/it]cv 값 : 0.26935814277628684람다 값 : 0.4596626559186777



![png](output_52_1697.png)


     29%|██████████████████████▉                                                        | 566/1948 [12:09<37:28,  1.63s/it]cv 값 : 0.39918861491422986람다 값 : -0.2997953930523689



![png](output_52_1700.png)


     29%|██████████████████████▉                                                        | 567/1948 [12:10<33:56,  1.47s/it]cv 값 : 0.3580249331893803람다 값 : 0.6195798670096921



![png](output_52_1703.png)


     29%|███████████████████████                                                        | 568/1948 [12:12<38:22,  1.67s/it]cv 값 : 0.12827129392292305람다 값 : 1.7797079451980502



![png](output_52_1706.png)


     29%|███████████████████████                                                        | 569/1948 [12:14<38:50,  1.69s/it]cv 값 : 0.2669889840437501람다 값 : 0.9140367902391342



![png](output_52_1709.png)


     29%|███████████████████████                                                        | 570/1948 [12:15<40:01,  1.74s/it]cv 값 : 0.565725578451785람다 값 : -0.40028172801467365



![png](output_52_1712.png)


     29%|███████████████████████▏                                                       | 571/1948 [12:19<51:46,  2.26s/it]cv 값 : 0.281817389270701람다 값 : 0.2437628638266125



![png](output_52_1715.png)


     29%|███████████████████████▏                                                       | 572/1948 [12:21<49:38,  2.16s/it]cv 값 : 0.2571555836723513람다 값 : 1.4944411777178064



![png](output_52_1718.png)


     29%|███████████████████████▏                                                       | 573/1948 [12:24<56:43,  2.48s/it]cv 값 : 0.47374546434627474람다 값 : -0.1367963807460596



![png](output_52_1721.png)


     29%|███████████████████████▎                                                       | 574/1948 [12:26<53:08,  2.32s/it]cv 값 : 0.3258452282583372람다 값 : -0.42374499681341116



![png](output_52_1724.png)


     30%|███████████████████████▎                                                       | 575/1948 [12:28<50:18,  2.20s/it]cv 값 : 0.18322715111114732람다 값 : -0.12160037895507456



![png](output_52_1727.png)


     30%|███████████████████████▎                                                       | 576/1948 [12:30<46:44,  2.04s/it]cv 값 : 0.38657235728603473람다 값 : 0.6186782617953204



![png](output_52_1730.png)


     30%|███████████████████████▍                                                       | 577/1948 [12:31<44:31,  1.95s/it]cv 값 : 0.26691792454513713람다 값 : 0.4597725396403855



![png](output_52_1733.png)


     30%|███████████████████████▍                                                       | 578/1948 [12:33<41:43,  1.83s/it]cv 값 : 0.37732525955258145람다 값 : 0.7491927008241165



![png](output_52_1736.png)


     30%|███████████████████████▍                                                       | 579/1948 [12:35<42:13,  1.85s/it]cv 값 : 0.2267607155300731람다 값 : 0.5425517777340294



![png](output_52_1739.png)


     30%|███████████████████████▌                                                       | 580/1948 [12:37<43:36,  1.91s/it]cv 값 : 0.8020191113501075람다 값 : 0.15163647549888112



![png](output_52_1742.png)


     30%|███████████████████████▌                                                       | 581/1948 [12:39<44:22,  1.95s/it]cv 값 : 0.3283703596729805람다 값 : 0.5537523368622072



![png](output_52_1745.png)


     30%|███████████████████████▌                                                       | 582/1948 [12:41<48:02,  2.11s/it]cv 값 : 0.32190910647337384람다 값 : 0.19745453049267858



![png](output_52_1748.png)


     30%|███████████████████████▋                                                       | 583/1948 [12:43<46:52,  2.06s/it]cv 값 : 0.3716303254490566람다 값 : 0.691804802063924



![png](output_52_1751.png)


     30%|███████████████████████▋                                                       | 584/1948 [12:45<44:28,  1.96s/it]cv 값 : 0.355720476455521람다 값 : 0.27480729384362684



![png](output_52_1754.png)


     30%|███████████████████████▋                                                       | 585/1948 [12:46<41:20,  1.82s/it]cv 값 : 0.17008188628945603람다 값 : -0.21095294176770682



![png](output_52_1757.png)


     30%|███████████████████████▊                                                       | 586/1948 [12:48<37:22,  1.65s/it]cv 값 : 0.14509864395872824람다 값 : 0.5976453065379589



![png](output_52_1760.png)


     30%|███████████████████████▊                                                       | 587/1948 [12:49<34:48,  1.53s/it]cv 값 : 0.2351214143635106람다 값 : 0.7126284368828074



![png](output_52_1763.png)


     30%|███████████████████████▊                                                       | 588/1948 [12:50<33:42,  1.49s/it]cv 값 : 0.2697222118686892람다 값 : 1.1480617093382195



![png](output_52_1766.png)


     30%|███████████████████████▉                                                       | 589/1948 [12:52<31:56,  1.41s/it]cv 값 : 0.34751757394436106람다 값 : 0.7879470584629484



![png](output_52_1769.png)


     30%|███████████████████████▉                                                       | 590/1948 [12:53<30:27,  1.35s/it]cv 값 : 0.6882964025217614람다 값 : -0.012529512030713494



![png](output_52_1772.png)


     30%|███████████████████████▉                                                       | 591/1948 [12:54<29:53,  1.32s/it]cv 값 : 0.07880686183239523람다 값 : -0.03029770208116198



![png](output_52_1775.png)


     30%|████████████████████████                                                       | 592/1948 [12:55<28:41,  1.27s/it]cv 값 : 0.8010660543579109람다 값 : 0.07761771538852302



![png](output_52_1778.png)


     30%|████████████████████████                                                       | 593/1948 [12:56<27:54,  1.24s/it]cv 값 : 0.629901381599517람다 값 : 0.3427565547745928



![png](output_52_1781.png)


     30%|████████████████████████                                                       | 594/1948 [12:58<27:21,  1.21s/it]cv 값 : 0.2447079939477597람다 값 : -0.0026847819393561014



![png](output_52_1784.png)


     31%|████████████████████████▏                                                      | 595/1948 [12:59<27:31,  1.22s/it]cv 값 : 0.23694342687556202람다 값 : 0.340033370585762



![png](output_52_1787.png)


     31%|████████████████████████▏                                                      | 596/1948 [13:00<26:52,  1.19s/it]cv 값 : 0.1596313859639889람다 값 : 1.7878741868475634



![png](output_52_1790.png)


     31%|████████████████████████▏                                                      | 597/1948 [13:01<27:57,  1.24s/it]cv 값 : 0.30245341100898676람다 값 : 1.024020516430255



![png](output_52_1793.png)


     31%|████████████████████████▎                                                      | 598/1948 [13:02<26:24,  1.17s/it]cv 값 : 0.2109109713161496람다 값 : -0.03724927287445257



![png](output_52_1796.png)


     31%|████████████████████████▎                                                      | 599/1948 [13:03<26:13,  1.17s/it]cv 값 : 1.0542869781390647람다 값 : 0.1578879553400236



![png](output_52_1799.png)


     31%|████████████████████████▎                                                      | 600/1948 [13:04<25:31,  1.14s/it]cv 값 : 0.3876844534184309람다 값 : 0.1496628005111855



![png](output_52_1802.png)


     31%|████████████████████████▎                                                      | 601/1948 [13:06<26:37,  1.19s/it]cv 값 : 0.5564485626558627람다 값 : 0.47025279217903465



![png](output_52_1805.png)


     31%|████████████████████████▍                                                      | 602/1948 [13:07<27:08,  1.21s/it]cv 값 : 0.24323404700532755람다 값 : 0.6258565966581634



![png](output_52_1808.png)


     31%|████████████████████████▍                                                      | 603/1948 [13:08<26:27,  1.18s/it]cv 값 : 0.1314320071678213람다 값 : 0.7390233891690685



![png](output_52_1811.png)


     31%|████████████████████████▍                                                      | 604/1948 [13:09<26:15,  1.17s/it]cv 값 : 0.6161167749087402람다 값 : -0.12515633200930332



![png](output_52_1814.png)


     31%|████████████████████████▌                                                      | 605/1948 [13:10<25:53,  1.16s/it]cv 값 : 0.24084826523888916람다 값 : 1.161610765881475



![png](output_52_1817.png)


     31%|████████████████████████▌                                                      | 606/1948 [13:12<29:16,  1.31s/it]cv 값 : 0.1867630756108613람다 값 : 0.6911428409242591



![png](output_52_1820.png)


     31%|████████████████████████▌                                                      | 607/1948 [13:13<28:46,  1.29s/it]cv 값 : 0.40023188530761483람다 값 : 0.13812911806913686



![png](output_52_1823.png)


     31%|████████████████████████▋                                                      | 608/1948 [13:14<27:29,  1.23s/it]cv 값 : 0.22994394982101682람다 값 : 0.9992073542179938



![png](output_52_1826.png)


     31%|████████████████████████▋                                                      | 609/1948 [13:16<26:44,  1.20s/it]cv 값 : 0.2397515972402697람다 값 : 0.5194777979247717



![png](output_52_1829.png)


     31%|████████████████████████▋                                                      | 610/1948 [13:17<26:28,  1.19s/it]cv 값 : 0.2645983516055972람다 값 : 0.45047368934272586



![png](output_52_1832.png)


     31%|████████████████████████▊                                                      | 611/1948 [13:18<26:06,  1.17s/it]cv 값 : 0.7459705514893842람다 값 : 0.3273383771590448



![png](output_52_1835.png)


     31%|████████████████████████▊                                                      | 612/1948 [13:19<25:08,  1.13s/it]cv 값 : 0.5227202944271809람다 값 : -0.3299655919047378



![png](output_52_1838.png)


     31%|████████████████████████▊                                                      | 613/1948 [13:20<24:55,  1.12s/it]cv 값 : 0.25356476389104166람다 값 : 0.24378215583991727



![png](output_52_1841.png)


     32%|████████████████████████▉                                                      | 614/1948 [13:21<23:55,  1.08s/it]cv 값 : 0.2870377124042553람다 값 : 0.4642541942635227



![png](output_52_1844.png)


     32%|████████████████████████▉                                                      | 615/1948 [13:22<24:44,  1.11s/it]cv 값 : 0.2625291801920073람다 값 : -0.9140779572644978



![png](output_52_1847.png)


     32%|████████████████████████▉                                                      | 616/1948 [13:23<25:03,  1.13s/it]cv 값 : 0.3768581522290979람다 값 : 0.6040389053535771



![png](output_52_1850.png)


     32%|█████████████████████████                                                      | 617/1948 [13:24<24:12,  1.09s/it]cv 값 : 0.41151479645369576람다 값 : 0.5331791701824762



![png](output_52_1853.png)


     32%|█████████████████████████                                                      | 618/1948 [13:25<24:13,  1.09s/it]cv 값 : 0.2528769817044022람다 값 : -1.0522862811433433



![png](output_52_1856.png)


     32%|█████████████████████████                                                      | 619/1948 [13:27<25:12,  1.14s/it]cv 값 : 0.7011335676357745람다 값 : 0.4310029970585845



![png](output_52_1859.png)


     32%|█████████████████████████▏                                                     | 620/1948 [13:28<25:26,  1.15s/it]cv 값 : 0.14607334983839773람다 값 : 1.5507075394318346



![png](output_52_1862.png)


     32%|█████████████████████████▏                                                     | 621/1948 [13:29<25:21,  1.15s/it]cv 값 : 0.16931359520841238람다 값 : -1.2488562700452617



![png](output_52_1865.png)


     32%|█████████████████████████▏                                                     | 622/1948 [13:30<25:13,  1.14s/it]cv 값 : 0.446677743272401람다 값 : 0.5168566620726655



![png](output_52_1868.png)


     32%|█████████████████████████▎                                                     | 623/1948 [13:31<24:31,  1.11s/it]cv 값 : 0.3045988223826333람다 값 : -0.8303763383255478



![png](output_52_1871.png)


     32%|█████████████████████████▎                                                     | 624/1948 [13:32<25:28,  1.15s/it]cv 값 : 0.3946653824525559람다 값 : -0.4725203811170227



![png](output_52_1874.png)


     32%|█████████████████████████▎                                                     | 625/1948 [13:34<26:02,  1.18s/it]cv 값 : 0.45434905864946146람다 값 : 0.5361041786231663



![png](output_52_1877.png)


     32%|█████████████████████████▍                                                     | 626/1948 [13:35<25:30,  1.16s/it]cv 값 : 0.4833937263483038람다 값 : 0.4784381251628708



![png](output_52_1880.png)


     32%|█████████████████████████▍                                                     | 627/1948 [13:36<25:14,  1.15s/it]cv 값 : 0.5492153296672222람다 값 : -0.12815112792124622



![png](output_52_1883.png)


     32%|█████████████████████████▍                                                     | 628/1948 [13:37<24:56,  1.13s/it]cv 값 : 0.3946208905584085람다 값 : 0.4195761722448717



![png](output_52_1886.png)


     32%|█████████████████████████▌                                                     | 629/1948 [13:38<23:41,  1.08s/it]cv 값 : 0.2583142578354493람다 값 : 0.04349216311485504



![png](output_52_1889.png)


     32%|█████████████████████████▌                                                     | 630/1948 [13:39<23:52,  1.09s/it]cv 값 : 1.4514216957443042람다 값 : -0.26193459093463584



![png](output_52_1892.png)


     32%|█████████████████████████▌                                                     | 631/1948 [13:40<23:46,  1.08s/it]cv 값 : 0.4103397718968405람다 값 : 0.33764180886400946



![png](output_52_1895.png)


     32%|█████████████████████████▋                                                     | 632/1948 [13:41<23:09,  1.06s/it]cv 값 : 0.5184518214229146람다 값 : -0.242099126527785



![png](output_52_1898.png)


     32%|█████████████████████████▋                                                     | 633/1948 [13:42<24:34,  1.12s/it]cv 값 : 0.7809603620400952람다 값 : -0.08827551905447044



![png](output_52_1901.png)


     33%|█████████████████████████▋                                                     | 634/1948 [13:44<26:44,  1.22s/it]cv 값 : 0.46416043824199726람다 값 : -0.4065964194593113



![png](output_52_1904.png)


     33%|█████████████████████████▊                                                     | 635/1948 [13:45<29:23,  1.34s/it]cv 값 : 0.32569537414120936람다 값 : 0.16091308201240626



![png](output_52_1907.png)


     33%|█████████████████████████▊                                                     | 636/1948 [13:47<30:04,  1.38s/it]cv 값 : 0.37399656560418165람다 값 : 0.7083951458681906



![png](output_52_1910.png)


     33%|█████████████████████████▊                                                     | 637/1948 [13:49<33:55,  1.55s/it]cv 값 : 0.7984028957057523람다 값 : -0.4453553051017927



![png](output_52_1913.png)


     33%|█████████████████████████▊                                                     | 638/1948 [13:50<33:33,  1.54s/it]cv 값 : 0.44451832074107994람다 값 : -0.4274313609206605



![png](output_52_1916.png)


     33%|█████████████████████████▉                                                     | 639/1948 [13:52<31:03,  1.42s/it]cv 값 : 0.585667205525349람다 값 : 0.4977656929918574



![png](output_52_1919.png)


     33%|█████████████████████████▉                                                     | 640/1948 [13:53<31:24,  1.44s/it]cv 값 : 0.23224481208840675람다 값 : 0.54312431521782



![png](output_52_1922.png)


     33%|█████████████████████████▉                                                     | 641/1948 [13:54<30:45,  1.41s/it]cv 값 : 0.16211226318710736람다 값 : 0.053411601377442865



![png](output_52_1925.png)


     33%|██████████████████████████                                                     | 642/1948 [13:56<30:47,  1.41s/it]cv 값 : 0.15047792071223134람다 값 : -0.4314536786462471



![png](output_52_1928.png)


     33%|██████████████████████████                                                     | 643/1948 [13:57<29:37,  1.36s/it]cv 값 : 0.6158970086425207람다 값 : 0.7015249076685677



![png](output_52_1931.png)


     33%|██████████████████████████                                                     | 644/1948 [13:58<27:32,  1.27s/it]cv 값 : 0.5622067205865586람다 값 : 0.30240199114389754



![png](output_52_1934.png)


     33%|██████████████████████████▏                                                    | 645/1948 [13:59<26:25,  1.22s/it]cv 값 : 0.5839461408143924람다 값 : 0.44121007087988



![png](output_52_1937.png)


     33%|██████████████████████████▏                                                    | 646/1948 [14:00<26:26,  1.22s/it]cv 값 : 0.5147082712927169람다 값 : -0.05563852614189999



![png](output_52_1940.png)


     33%|██████████████████████████▏                                                    | 647/1948 [14:01<25:47,  1.19s/it]cv 값 : 0.21670790121490932람다 값 : -0.0807820923357579



![png](output_52_1943.png)


     33%|██████████████████████████▎                                                    | 648/1948 [14:04<31:26,  1.45s/it]cv 값 : 0.6326191401588073람다 값 : 0.021642084719673077



![png](output_52_1946.png)


     33%|██████████████████████████▎                                                    | 649/1948 [14:06<35:51,  1.66s/it]cv 값 : 0.3349773215975881람다 값 : 0.3992096822533215



![png](output_52_1949.png)


     33%|██████████████████████████▎                                                    | 650/1948 [14:07<33:30,  1.55s/it]cv 값 : 0.14305719948027254람다 값 : -0.07263531981792269



![png](output_52_1952.png)


     33%|██████████████████████████▍                                                    | 651/1948 [14:08<31:19,  1.45s/it]cv 값 : 0.8814149465599959람다 값 : -0.2703034771991946



![png](output_52_1955.png)


     33%|██████████████████████████▍                                                    | 652/1948 [14:09<29:40,  1.37s/it]cv 값 : 0.5977109965895855람다 값 : 0.6384643844556381



![png](output_52_1958.png)


     34%|██████████████████████████▍                                                    | 653/1948 [14:11<29:11,  1.35s/it]cv 값 : 0.2648014048267243람다 값 : 0.8087282825919944



![png](output_52_1961.png)


     34%|██████████████████████████▌                                                    | 654/1948 [14:12<27:34,  1.28s/it]cv 값 : 0.22487661085289717람다 값 : 0.16598666964943173



![png](output_52_1964.png)


     34%|██████████████████████████▌                                                    | 655/1948 [14:13<27:02,  1.25s/it]cv 값 : 0.2973270789424724람다 값 : 0.1494750001569499



![png](output_52_1967.png)


     34%|██████████████████████████▌                                                    | 656/1948 [14:14<27:36,  1.28s/it]cv 값 : 0.335468619309133람다 값 : 1.2702559390289123



![png](output_52_1970.png)


     34%|██████████████████████████▋                                                    | 657/1948 [14:16<27:32,  1.28s/it]cv 값 : 0.5360917224420723람다 값 : 0.2744558882927403



![png](output_52_1973.png)


     34%|██████████████████████████▋                                                    | 658/1948 [14:17<26:31,  1.23s/it]cv 값 : 0.25906725656772583람다 값 : 0.06272958281886414



![png](output_52_1976.png)


     34%|██████████████████████████▋                                                    | 659/1948 [14:18<27:09,  1.26s/it]cv 값 : 0.4633219809511843람다 값 : 0.48572197931947464



![png](output_52_1979.png)


     34%|██████████████████████████▊                                                    | 660/1948 [14:19<25:41,  1.20s/it]cv 값 : 0.21061797618786726람다 값 : -1.3792472099720385



![png](output_52_1982.png)


     34%|██████████████████████████▊                                                    | 661/1948 [14:20<25:32,  1.19s/it]cv 값 : 0.2319028440792827람다 값 : 1.1481166510190752



![png](output_52_1985.png)


     34%|██████████████████████████▊                                                    | 662/1948 [14:21<25:04,  1.17s/it]cv 값 : 0.18713836293370495람다 값 : -0.1379887378105187



![png](output_52_1988.png)


     34%|██████████████████████████▉                                                    | 663/1948 [14:23<24:40,  1.15s/it]cv 값 : 0.4065409790103896람다 값 : 1.0878140697475482



![png](output_52_1991.png)


     34%|██████████████████████████▉                                                    | 664/1948 [14:24<26:09,  1.22s/it]cv 값 : 0.311637934421696람다 값 : 0.6046559450185942



![png](output_52_1994.png)


     34%|██████████████████████████▉                                                    | 665/1948 [14:25<25:30,  1.19s/it]cv 값 : 0.41284276946362247람다 값 : -0.019802041400564258



![png](output_52_1997.png)


     34%|███████████████████████████                                                    | 666/1948 [14:26<24:18,  1.14s/it]cv 값 : 0.6704344178073603람다 값 : -0.14338985525696757



![png](output_52_2000.png)


     34%|███████████████████████████                                                    | 667/1948 [14:27<23:49,  1.12s/it]cv 값 : 0.3394204183732308람다 값 : 1.318617483611258



![png](output_52_2003.png)


     34%|███████████████████████████                                                    | 668/1948 [14:28<24:15,  1.14s/it]cv 값 : 0.19229630899947145람다 값 : 0.06269985018937357



![png](output_52_2006.png)


     34%|███████████████████████████▏                                                   | 669/1948 [14:29<23:52,  1.12s/it]cv 값 : 0.2756610248774358람다 값 : 0.7135595773768777



![png](output_52_2009.png)


     34%|███████████████████████████▏                                                   | 670/1948 [14:31<24:09,  1.13s/it]cv 값 : 0.218474003005008람다 값 : 0.8817612386761521



![png](output_52_2012.png)


     34%|███████████████████████████▏                                                   | 671/1948 [14:32<22:59,  1.08s/it]cv 값 : 0.8692022497216423람다 값 : 0.32752296226113675



![png](output_52_2015.png)


     34%|███████████████████████████▎                                                   | 672/1948 [14:33<23:43,  1.12s/it]cv 값 : 0.1489653520807027람다 값 : 0.18042302868019305



![png](output_52_2018.png)


     35%|███████████████████████████▎                                                   | 673/1948 [14:34<23:35,  1.11s/it]cv 값 : 0.1950717717381603람다 값 : 1.1520774701339729



![png](output_52_2021.png)


     35%|███████████████████████████▎                                                   | 674/1948 [14:35<23:31,  1.11s/it]cv 값 : 0.2807524883250216람다 값 : 1.5552676172178594



![png](output_52_2024.png)


     35%|███████████████████████████▎                                                   | 675/1948 [14:36<23:30,  1.11s/it]cv 값 : 0.16063951493099968람다 값 : 0.5088561501147244



![png](output_52_2027.png)


     35%|███████████████████████████▍                                                   | 676/1948 [14:37<22:45,  1.07s/it]cv 값 : 0.3303639305396477람다 값 : -0.0025898000604151466



![png](output_52_2030.png)


     35%|███████████████████████████▍                                                   | 677/1948 [14:38<24:41,  1.17s/it]cv 값 : 0.8044354404040218람다 값 : 0.41282630571184353



![png](output_52_2033.png)


     35%|███████████████████████████▍                                                   | 678/1948 [14:40<24:58,  1.18s/it]cv 값 : 0.8342345382311664람다 값 : 0.2941622466925435



![png](output_52_2036.png)


     35%|███████████████████████████▌                                                   | 679/1948 [14:41<23:55,  1.13s/it]cv 값 : 0.07241986795659532람다 값 : 0.9802578472006598



![png](output_52_2039.png)


     35%|███████████████████████████▌                                                   | 680/1948 [14:42<22:51,  1.08s/it]cv 값 : 0.2915379415234593람다 값 : -0.3170010808606621



![png](output_52_2042.png)


     35%|███████████████████████████▌                                                   | 681/1948 [14:43<24:48,  1.17s/it]cv 값 : 0.2548499806640116람다 값 : 1.1035443678596073



![png](output_52_2045.png)


     35%|███████████████████████████▋                                                   | 682/1948 [14:44<25:01,  1.19s/it]cv 값 : 0.20591290733060275람다 값 : 1.3358617247640852



![png](output_52_2048.png)


     35%|███████████████████████████▋                                                   | 683/1948 [14:45<24:44,  1.17s/it]cv 값 : 0.36072915574096015람다 값 : 0.49572403098470486



![png](output_52_2051.png)


     35%|███████████████████████████▋                                                   | 684/1948 [14:46<24:22,  1.16s/it]cv 값 : 0.2439194326065904람다 값 : -0.9344810370084208



![png](output_52_2054.png)


     35%|███████████████████████████▊                                                   | 685/1948 [14:48<24:29,  1.16s/it]cv 값 : 0.5015693811732577람다 값 : 0.14845999074120153



![png](output_52_2057.png)


     35%|███████████████████████████▊                                                   | 686/1948 [14:49<25:08,  1.20s/it]cv 값 : 0.5942073559360279람다 값 : 0.5412692269120565



![png](output_52_2060.png)


     35%|███████████████████████████▊                                                   | 687/1948 [14:50<23:58,  1.14s/it]cv 값 : 0.4328002096219061람다 값 : 0.7929422205438953



![png](output_52_2063.png)


     35%|███████████████████████████▉                                                   | 688/1948 [14:51<23:43,  1.13s/it]cv 값 : 0.14050888783005358람다 값 : 0.12881887233152794



![png](output_52_2066.png)


     35%|███████████████████████████▉                                                   | 689/1948 [14:52<22:59,  1.10s/it]cv 값 : 0.2776148519431643람다 값 : -0.07599827427500726



![png](output_52_2069.png)


     35%|███████████████████████████▉                                                   | 690/1948 [14:53<23:50,  1.14s/it]cv 값 : 0.17719553480167333람다 값 : -0.1680143772206349



![png](output_52_2072.png)


     35%|████████████████████████████                                                   | 691/1948 [14:54<23:03,  1.10s/it]cv 값 : 0.3853972635778876람다 값 : -1.3384176503866692



![png](output_52_2075.png)


     36%|████████████████████████████                                                   | 692/1948 [14:55<23:34,  1.13s/it]cv 값 : 0.8514240049644368람다 값 : -0.20217977358999042



![png](output_52_2078.png)


     36%|████████████████████████████                                                   | 693/1948 [14:57<23:51,  1.14s/it]cv 값 : 0.2355403804044387람다 값 : -1.1799748751066386



![png](output_52_2081.png)


     36%|████████████████████████████▏                                                  | 694/1948 [14:58<24:23,  1.17s/it]cv 값 : 0.38721402638155383람다 값 : -0.1711991829105606



![png](output_52_2084.png)


     36%|████████████████████████████▏                                                  | 695/1948 [14:59<25:23,  1.22s/it]cv 값 : 0.23191342564464976람다 값 : -0.17579275826884078



![png](output_52_2087.png)


     36%|████████████████████████████▏                                                  | 696/1948 [15:00<24:44,  1.19s/it]cv 값 : 0.25875101637727305람다 값 : 0.543681653636517



![png](output_52_2090.png)


     36%|████████████████████████████▎                                                  | 697/1948 [15:01<24:10,  1.16s/it]cv 값 : 0.4741188936174738람다 값 : 0.6839405677351528



![png](output_52_2093.png)


     36%|████████████████████████████▎                                                  | 698/1948 [15:03<23:59,  1.15s/it]cv 값 : 0.3261081003341632람다 값 : 0.7694507556849866



![png](output_52_2096.png)


     36%|████████████████████████████▎                                                  | 699/1948 [15:04<24:16,  1.17s/it]cv 값 : 0.2565407285289855람다 값 : -1.0441319094728476



![png](output_52_2099.png)


     36%|████████████████████████████▍                                                  | 700/1948 [15:05<24:46,  1.19s/it]cv 값 : 0.40109323295421634람다 값 : 0.9920324713812351



![png](output_52_2102.png)


     36%|████████████████████████████▍                                                  | 701/1948 [15:06<23:48,  1.15s/it]cv 값 : 0.4603254016349255람다 값 : 0.14358780633199103



![png](output_52_2105.png)


     36%|████████████████████████████▍                                                  | 702/1948 [15:07<23:22,  1.13s/it]cv 값 : 0.5452197135227845람다 값 : 0.4328889847894157



![png](output_52_2108.png)


     36%|████████████████████████████▌                                                  | 703/1948 [15:08<23:40,  1.14s/it]cv 값 : 0.3394672031803186람다 값 : 0.22963688070828858



![png](output_52_2111.png)


     36%|████████████████████████████▌                                                  | 704/1948 [15:10<24:50,  1.20s/it]cv 값 : 0.325549147397174람다 값 : -0.37965837358050364



![png](output_52_2114.png)


     36%|████████████████████████████▌                                                  | 705/1948 [15:11<24:29,  1.18s/it]cv 값 : 0.1814558536003326람다 값 : 0.799324571150472



![png](output_52_2117.png)


     36%|████████████████████████████▋                                                  | 706/1948 [15:12<23:26,  1.13s/it]cv 값 : 0.5456434246428148람다 값 : -0.7218031277053105



![png](output_52_2120.png)


     36%|████████████████████████████▋                                                  | 707/1948 [15:13<23:17,  1.13s/it]cv 값 : 0.15689610935647344람다 값 : 1.288399333839414



![png](output_52_2123.png)


     36%|████████████████████████████▋                                                  | 708/1948 [15:14<23:34,  1.14s/it]cv 값 : 0.37196343154935624람다 값 : 0.7852039075095565



![png](output_52_2126.png)


     36%|████████████████████████████▊                                                  | 709/1948 [15:15<23:17,  1.13s/it]cv 값 : 0.3013580420040107람다 값 : -0.5453280141006738



![png](output_52_2129.png)


     36%|████████████████████████████▊                                                  | 710/1948 [15:16<23:55,  1.16s/it]cv 값 : 0.3223659175278994람다 값 : -0.14113833258630326



![png](output_52_2132.png)


     36%|████████████████████████████▊                                                  | 711/1948 [15:17<23:15,  1.13s/it]cv 값 : 0.12834539499655495람다 값 : 0.40292278613382804



![png](output_52_2135.png)


     37%|████████████████████████████▊                                                  | 712/1948 [15:19<24:09,  1.17s/it]cv 값 : 0.6229554898936432람다 값 : 0.21473304773380647



![png](output_52_2138.png)


     37%|████████████████████████████▉                                                  | 713/1948 [15:20<23:24,  1.14s/it]cv 값 : 0.5848282239449913람다 값 : 0.18696620432540717



![png](output_52_2141.png)


     37%|████████████████████████████▉                                                  | 714/1948 [15:21<23:04,  1.12s/it]cv 값 : 0.2960180993850971람다 값 : -0.2131717511703409



![png](output_52_2144.png)


     37%|████████████████████████████▉                                                  | 715/1948 [15:22<23:03,  1.12s/it]cv 값 : 0.3603849834046114람다 값 : -0.21841918795043347



![png](output_52_2147.png)


     37%|█████████████████████████████                                                  | 716/1948 [15:23<23:16,  1.13s/it]cv 값 : 0.30833386772441423람다 값 : -0.4116742635818371



![png](output_52_2150.png)


     37%|█████████████████████████████                                                  | 717/1948 [15:24<24:12,  1.18s/it]cv 값 : 0.31377128499990387람다 값 : 1.1556529627868815



![png](output_52_2153.png)


     37%|█████████████████████████████                                                  | 718/1948 [15:26<23:26,  1.14s/it]cv 값 : 0.41716999614360367람다 값 : 0.7504965769343567



![png](output_52_2156.png)


     37%|█████████████████████████████▏                                                 | 719/1948 [15:27<22:59,  1.12s/it]cv 값 : 0.5137778424512943람다 값 : 0.207296477783893



![png](output_52_2159.png)


     37%|█████████████████████████████▏                                                 | 720/1948 [15:28<22:41,  1.11s/it]cv 값 : 0.32557850050245124람다 값 : -0.07163471239498831



![png](output_52_2162.png)


     37%|█████████████████████████████▏                                                 | 721/1948 [15:29<23:24,  1.15s/it]cv 값 : 0.5813700039578955람다 값 : 0.5275234371387965



![png](output_52_2165.png)


     37%|█████████████████████████████▎                                                 | 722/1948 [15:30<23:33,  1.15s/it]cv 값 : 0.19548240290517438람다 값 : 0.9246976741451879



![png](output_52_2168.png)


     37%|█████████████████████████████▎                                                 | 723/1948 [15:31<22:44,  1.11s/it]cv 값 : 0.4004321105554414람다 값 : 0.770787181558885



![png](output_52_2171.png)


     37%|█████████████████████████████▎                                                 | 724/1948 [15:32<21:50,  1.07s/it]cv 값 : 0.46131491705778926람다 값 : 0.8869002904759841



![png](output_52_2174.png)


     37%|█████████████████████████████▍                                                 | 725/1948 [15:33<22:44,  1.12s/it]cv 값 : 0.7538515036628693람다 값 : -0.08456086745054561



![png](output_52_2177.png)


     37%|█████████████████████████████▍                                                 | 726/1948 [15:34<23:13,  1.14s/it]cv 값 : 0.3143710577653559람다 값 : 1.7770683250118002



![png](output_52_2180.png)


     37%|█████████████████████████████▍                                                 | 727/1948 [15:36<23:00,  1.13s/it]cv 값 : 0.2517587154837992람다 값 : -0.3908138577304824



![png](output_52_2183.png)


     37%|█████████████████████████████▌                                                 | 728/1948 [15:37<26:46,  1.32s/it]cv 값 : 0.3977437396356967람다 값 : 0.7173460224813428



![png](output_52_2186.png)


     37%|█████████████████████████████▌                                                 | 729/1948 [15:39<28:16,  1.39s/it]cv 값 : 0.8628770975867124람다 값 : 0.2538955908214204



![png](output_52_2189.png)


     37%|█████████████████████████████▌                                                 | 730/1948 [15:41<29:36,  1.46s/it]cv 값 : 0.3957400323676417람다 값 : 0.2479179928540097



![png](output_52_2192.png)


     38%|█████████████████████████████▋                                                 | 731/1948 [15:42<29:11,  1.44s/it]cv 값 : 0.30053434153342534람다 값 : -0.7231961885075389



![png](output_52_2195.png)


     38%|█████████████████████████████▋                                                 | 732/1948 [15:43<28:23,  1.40s/it]cv 값 : 0.2507775150959441람다 값 : -0.09882032172864838



![png](output_52_2198.png)


     38%|█████████████████████████████▋                                                 | 733/1948 [15:45<31:02,  1.53s/it]cv 값 : 0.15303721417713761람다 값 : 0.5606838224382638



![png](output_52_2201.png)


     38%|█████████████████████████████▊                                                 | 734/1948 [15:47<33:11,  1.64s/it]cv 값 : 0.2225329753714864람다 값 : 1.659611879041025



![png](output_52_2204.png)


     38%|█████████████████████████████▊                                                 | 735/1948 [15:50<43:39,  2.16s/it]cv 값 : 0.883425116861619람다 값 : 0.25004951750294563



![png](output_52_2207.png)


     38%|█████████████████████████████▊                                                 | 736/1948 [15:55<57:19,  2.84s/it]cv 값 : 0.20029371627823353람다 값 : 0.7429323503063757



![png](output_52_2210.png)


     38%|█████████████████████████████▏                                               | 737/1948 [15:58<1:01:42,  3.06s/it]cv 값 : 0.5106428259427128람다 값 : 0.018939719611353437



![png](output_52_2213.png)


     38%|█████████████████████████████▉                                                 | 738/1948 [16:01<58:29,  2.90s/it]cv 값 : 0.4887870235464763람다 값 : 0.1972728392401385



![png](output_52_2216.png)


     38%|█████████████████████████████▉                                                 | 739/1948 [16:02<50:08,  2.49s/it]cv 값 : 0.33061168084492876람다 값 : 1.3715571042789558



![png](output_52_2219.png)


     38%|██████████████████████████████                                                 | 740/1948 [16:04<45:52,  2.28s/it]cv 값 : 0.40637269685909977람다 값 : 0.1340743020296954



![png](output_52_2222.png)


     38%|██████████████████████████████                                                 | 741/1948 [16:06<41:35,  2.07s/it]cv 값 : 0.9551160271406605람다 값 : -0.5221358871406355



![png](output_52_2225.png)


     38%|██████████████████████████████                                                 | 742/1948 [16:07<37:18,  1.86s/it]cv 값 : 0.6909457417769392람다 값 : 0.415791929219797



![png](output_52_2228.png)


     38%|██████████████████████████████▏                                                | 743/1948 [16:08<34:16,  1.71s/it]cv 값 : 0.4037306539509557람다 값 : 0.6081638575369812



![png](output_52_2231.png)


     38%|██████████████████████████████▏                                                | 744/1948 [16:10<35:36,  1.77s/it]cv 값 : 0.6021088079903189람다 값 : 0.3968370370878742



![png](output_52_2234.png)


     38%|██████████████████████████████▏                                                | 745/1948 [16:12<31:50,  1.59s/it]cv 값 : 0.41202391049346615람다 값 : -0.2803430261200296



![png](output_52_2237.png)


     38%|██████████████████████████████▎                                                | 746/1948 [16:13<31:33,  1.58s/it]cv 값 : 1.5755778500200646람다 값 : -0.28729826519314694



![png](output_52_2240.png)


     38%|██████████████████████████████▎                                                | 747/1948 [16:14<30:17,  1.51s/it]cv 값 : 0.2947225568221116람다 값 : -2.0787840811597906



![png](output_52_2243.png)


     38%|██████████████████████████████▎                                                | 748/1948 [16:16<30:08,  1.51s/it]cv 값 : 0.1774972487412406람다 값 : -1.78643308669645



![png](output_52_2246.png)


     38%|██████████████████████████████▍                                                | 749/1948 [16:17<28:23,  1.42s/it]cv 값 : 0.1542846757079098람다 값 : 1.8555313146864667



![png](output_52_2249.png)


     39%|██████████████████████████████▍                                                | 750/1948 [16:18<27:52,  1.40s/it]cv 값 : 0.44542240052302084람다 값 : 0.81972960497424



![png](output_52_2252.png)


     39%|██████████████████████████████▍                                                | 751/1948 [16:20<26:57,  1.35s/it]cv 값 : 0.38077847176309215람다 값 : 0.6226868282501907



![png](output_52_2255.png)


     39%|██████████████████████████████▍                                                | 752/1948 [16:21<26:07,  1.31s/it]cv 값 : 0.38414061926401766람다 값 : 0.06110809374454299



![png](output_52_2258.png)


     39%|██████████████████████████████▌                                                | 753/1948 [16:22<27:05,  1.36s/it]cv 값 : 0.26021102489636416람다 값 : 0.4278290596655427



![png](output_52_2261.png)


     39%|██████████████████████████████▌                                                | 754/1948 [16:24<27:51,  1.40s/it]cv 값 : 0.5677192768714698람다 값 : -0.021560944839942413



![png](output_52_2264.png)


     39%|██████████████████████████████▌                                                | 755/1948 [16:25<28:04,  1.41s/it]cv 값 : 0.2841579097168668람다 값 : 0.8207641718675752



![png](output_52_2267.png)


     39%|██████████████████████████████▋                                                | 756/1948 [16:27<26:25,  1.33s/it]cv 값 : 0.3033864710048051람다 값 : 0.14396670482993967



![png](output_52_2270.png)


     39%|██████████████████████████████▋                                                | 757/1948 [16:28<25:50,  1.30s/it]cv 값 : 0.6170913973317125람다 값 : 0.31106308110890224



![png](output_52_2273.png)


     39%|██████████████████████████████▋                                                | 758/1948 [16:29<25:42,  1.30s/it]cv 값 : 0.13812309805553663람다 값 : 0.5961662683678236



![png](output_52_2276.png)


     39%|██████████████████████████████▊                                                | 759/1948 [16:30<24:45,  1.25s/it]cv 값 : 0.297582792400222람다 값 : 1.5250958944422346



![png](output_52_2279.png)


     39%|██████████████████████████████▊                                                | 760/1948 [16:31<24:34,  1.24s/it]cv 값 : 0.4745366217178158람다 값 : 0.7744303007000466



![png](output_52_2282.png)


     39%|██████████████████████████████▊                                                | 761/1948 [16:33<24:42,  1.25s/it]cv 값 : 0.3303505608199866람다 값 : 0.2742777366616906



![png](output_52_2285.png)


     39%|██████████████████████████████▉                                                | 762/1948 [16:34<24:40,  1.25s/it]cv 값 : 0.5531541554952888람다 값 : 0.38414944990015665



![png](output_52_2288.png)


     39%|██████████████████████████████▉                                                | 763/1948 [16:35<24:05,  1.22s/it]cv 값 : 0.37870084504140683람다 값 : 0.9784891785358011



![png](output_52_2291.png)


     39%|██████████████████████████████▉                                                | 764/1948 [16:36<22:56,  1.16s/it]cv 값 : 0.5722983537913975람다 값 : 0.49658549289068227



![png](output_52_2294.png)


     39%|███████████████████████████████                                                | 765/1948 [16:37<22:15,  1.13s/it]cv 값 : 0.29629033984557257람다 값 : 0.17696533384073446



![png](output_52_2297.png)


     39%|███████████████████████████████                                                | 766/1948 [16:38<23:19,  1.18s/it]cv 값 : 0.4438390541554502람다 값 : 1.1227374498988036



![png](output_52_2300.png)


     39%|███████████████████████████████                                                | 767/1948 [16:40<23:40,  1.20s/it]cv 값 : 0.11633947173578185람다 값 : -0.877759889060625



![png](output_52_2303.png)


     39%|███████████████████████████████▏                                               | 768/1948 [16:41<22:47,  1.16s/it]cv 값 : 0.2731441787648394람다 값 : -0.5521100102762136



![png](output_52_2306.png)


     39%|███████████████████████████████▏                                               | 769/1948 [16:42<22:47,  1.16s/it]cv 값 : 0.2764931295870785람다 값 : 1.4179033288159288



![png](output_52_2309.png)


     40%|███████████████████████████████▏                                               | 770/1948 [16:43<23:09,  1.18s/it]cv 값 : 0.3348285995242391람다 값 : 0.5978101301069394



![png](output_52_2312.png)


     40%|███████████████████████████████▎                                               | 771/1948 [16:44<23:22,  1.19s/it]cv 값 : 0.9045670202552765람다 값 : 0.2666498736807328



![png](output_52_2315.png)


     40%|███████████████████████████████▎                                               | 772/1948 [16:45<22:59,  1.17s/it]cv 값 : 0.2523175834469783람다 값 : 0.9733799411225708



![png](output_52_2318.png)


     40%|███████████████████████████████▎                                               | 773/1948 [16:47<22:39,  1.16s/it]cv 값 : 0.452000816907673람다 값 : -0.11516901345901731



![png](output_52_2321.png)


     40%|███████████████████████████████▍                                               | 774/1948 [16:48<21:50,  1.12s/it]cv 값 : 0.18144713280670746람다 값 : -1.0601298412506204



![png](output_52_2324.png)


     40%|███████████████████████████████▍                                               | 775/1948 [16:49<22:44,  1.16s/it]cv 값 : 0.672683730231155람다 값 : 0.4035492992630343



![png](output_52_2327.png)


     40%|███████████████████████████████▍                                               | 776/1948 [16:50<22:54,  1.17s/it]cv 값 : 0.22171728224642678람다 값 : 0.3297635337175331



![png](output_52_2330.png)


     40%|███████████████████████████████▌                                               | 777/1948 [16:51<22:00,  1.13s/it]cv 값 : 0.43505621147595996람다 값 : 0.7907529798641668



![png](output_52_2333.png)


     40%|███████████████████████████████▌                                               | 778/1948 [16:52<21:09,  1.08s/it]cv 값 : 0.3490978614093469람다 값 : 0.32010834195671356



![png](output_52_2336.png)


     40%|███████████████████████████████▌                                               | 779/1948 [16:53<22:02,  1.13s/it]cv 값 : 0.6060586029360703람다 값 : 0.5371048966648813



![png](output_52_2339.png)


     40%|███████████████████████████████▋                                               | 780/1948 [16:54<21:54,  1.13s/it]cv 값 : 0.1245222632048863람다 값 : 0.053922114540521385



![png](output_52_2342.png)


     40%|███████████████████████████████▋                                               | 781/1948 [16:56<21:53,  1.13s/it]cv 값 : 0.20947548656072473람다 값 : 1.1454377440809356



![png](output_52_2345.png)


     40%|███████████████████████████████▋                                               | 782/1948 [16:57<21:52,  1.13s/it]cv 값 : 0.6486277897764186람다 값 : 0.5925934352593024



![png](output_52_2348.png)


     40%|███████████████████████████████▊                                               | 783/1948 [16:58<21:21,  1.10s/it]cv 값 : 0.22726058631274856람다 값 : 0.63449283882213



![png](output_52_2351.png)


     40%|███████████████████████████████▊                                               | 784/1948 [16:59<22:18,  1.15s/it]cv 값 : 0.7327596442867639람다 값 : 0.4345728973685454



![png](output_52_2354.png)


     40%|███████████████████████████████▊                                               | 785/1948 [17:00<22:25,  1.16s/it]cv 값 : 0.5291404580034403람다 값 : -0.2993088448617592



![png](output_52_2357.png)


     40%|███████████████████████████████▉                                               | 786/1948 [17:01<21:39,  1.12s/it]cv 값 : 0.3749310050870762람다 값 : -0.6073709372287823



![png](output_52_2360.png)


     40%|███████████████████████████████▉                                               | 787/1948 [17:02<22:24,  1.16s/it]cv 값 : 0.35448722823737344람다 값 : 0.7042073530694789



![png](output_52_2363.png)


     40%|███████████████████████████████▉                                               | 788/1948 [17:04<25:03,  1.30s/it]cv 값 : 0.19511366362561391람다 값 : 0.4187711617715243



![png](output_52_2366.png)


     41%|███████████████████████████████▉                                               | 789/1948 [17:07<33:51,  1.75s/it]cv 값 : 0.6829672557602857람다 값 : 0.48320025588125537



![png](output_52_2369.png)


     41%|████████████████████████████████                                               | 790/1948 [17:10<39:01,  2.02s/it]cv 값 : 0.2072875023605947람다 값 : 1.1476982061220877



![png](output_52_2372.png)


     41%|████████████████████████████████                                               | 791/1948 [17:11<34:43,  1.80s/it]cv 값 : 0.44267969432668586람다 값 : -0.08528760966313918



![png](output_52_2375.png)


     41%|████████████████████████████████                                               | 792/1948 [17:12<31:43,  1.65s/it]cv 값 : 0.7538501680711086람다 값 : 0.3181122669079233



![png](output_52_2378.png)


     41%|████████████████████████████████▏                                              | 793/1948 [17:13<30:01,  1.56s/it]cv 값 : 0.4835770997862911람다 값 : 0.31551893364508427



![png](output_52_2381.png)


     41%|████████████████████████████████▏                                              | 794/1948 [17:15<28:08,  1.46s/it]cv 값 : 0.18581887071036943람다 값 : -0.3415506848015866



![png](output_52_2384.png)


     41%|████████████████████████████████▏                                              | 795/1948 [17:16<26:07,  1.36s/it]cv 값 : 0.22210537903560085람다 값 : 0.9279816036457708



![png](output_52_2387.png)


     41%|████████████████████████████████▎                                              | 796/1948 [17:17<24:17,  1.26s/it]cv 값 : 0.15150200462754917람다 값 : 1.4196249179160805



![png](output_52_2390.png)


     41%|████████████████████████████████▎                                              | 797/1948 [17:18<24:00,  1.25s/it]cv 값 : 0.6259155928375845람다 값 : 0.48244353766268666



![png](output_52_2393.png)


     41%|████████████████████████████████▎                                              | 798/1948 [17:19<23:14,  1.21s/it]cv 값 : 0.23735434035100214람다 값 : 1.1037220258050764



![png](output_52_2396.png)


     41%|████████████████████████████████▍                                              | 799/1948 [17:20<22:41,  1.19s/it]cv 값 : 0.2239721594080131람다 값 : 1.1394759698679005



![png](output_52_2399.png)


     41%|████████████████████████████████▍                                              | 800/1948 [17:22<22:35,  1.18s/it]cv 값 : 0.3732699738147712람다 값 : 0.410989352977078



![png](output_52_2402.png)


     41%|████████████████████████████████▍                                              | 801/1948 [17:23<23:18,  1.22s/it]cv 값 : 0.29086403813341566람다 값 : 0.6326962274746127



![png](output_52_2405.png)


     41%|████████████████████████████████▌                                              | 802/1948 [17:25<26:34,  1.39s/it]cv 값 : 0.5933547943174752람다 값 : 0.42482622159979655



![png](output_52_2408.png)


     41%|████████████████████████████████▌                                              | 803/1948 [17:26<26:47,  1.40s/it]cv 값 : 0.3540877109815016람다 값 : 0.4014204954164027



![png](output_52_2411.png)


     41%|████████████████████████████████▌                                              | 804/1948 [17:27<25:17,  1.33s/it]cv 값 : 0.41468049093878134람다 값 : -0.49762995765107604



![png](output_52_2414.png)


     41%|████████████████████████████████▋                                              | 805/1948 [17:28<24:27,  1.28s/it]cv 값 : 1.0347869814529354람다 값 : 0.053125833838344634



![png](output_52_2417.png)


     41%|████████████████████████████████▋                                              | 806/1948 [17:30<24:31,  1.29s/it]cv 값 : 0.36963074865631734람다 값 : 0.7694769173158743



![png](output_52_2420.png)


     41%|████████████████████████████████▋                                              | 807/1948 [17:31<24:41,  1.30s/it]cv 값 : 0.4218340070115072람다 값 : 0.2464028732805128



![png](output_52_2423.png)


     41%|████████████████████████████████▊                                              | 808/1948 [17:32<24:21,  1.28s/it]cv 값 : 0.6963448098393485람다 값 : 0.4084959139575241



![png](output_52_2426.png)


     42%|████████████████████████████████▊                                              | 809/1948 [17:35<33:35,  1.77s/it]cv 값 : 0.25176313004287165람다 값 : -1.1482607321340617



![png](output_52_2429.png)


     42%|████████████████████████████████▊                                              | 810/1948 [17:39<48:09,  2.54s/it]cv 값 : 0.4083800229680141람다 값 : 0.5069035763282237



![png](output_52_2432.png)


     42%|████████████████████████████████▉                                              | 811/1948 [17:42<49:34,  2.62s/it]cv 값 : 0.2696618936975986람다 값 : -1.161846008909856



![png](output_52_2435.png)


     42%|████████████████████████████████▉                                              | 812/1948 [17:46<56:34,  2.99s/it]cv 값 : 0.2218121576444391람다 값 : -0.6179822769980109



![png](output_52_2438.png)


     42%|████████████████████████████████▏                                            | 813/1948 [17:52<1:11:15,  3.77s/it]cv 값 : 0.19088235444016277람다 값 : -1.4059703853598151



![png](output_52_2441.png)


     42%|████████████████████████████████▏                                            | 814/1948 [17:55<1:09:10,  3.66s/it]cv 값 : 0.4499133259339262람다 값 : -0.12938954642390185



![png](output_52_2444.png)


     42%|████████████████████████████████▏                                            | 815/1948 [17:58<1:06:26,  3.52s/it]cv 값 : 0.28955731704659676람다 값 : 0.3224362320668659



![png](output_52_2447.png)


     42%|████████████████████████████████▎                                            | 816/1948 [18:03<1:13:41,  3.91s/it]cv 값 : 0.26041102334083194람다 값 : 2.003691132720639



![png](output_52_2450.png)


     42%|████████████████████████████████▎                                            | 817/1948 [18:05<1:01:26,  3.26s/it]cv 값 : 0.2534107855005316람다 값 : 0.5608599139310461



![png](output_52_2453.png)


     42%|█████████████████████████████████▏                                             | 818/1948 [18:07<52:07,  2.77s/it]cv 값 : 0.21695605560673054람다 값 : 0.6291867374530634



![png](output_52_2456.png)


     42%|█████████████████████████████████▏                                             | 819/1948 [18:10<55:16,  2.94s/it]cv 값 : 0.30414220032313505람다 값 : 0.5406368251158816



![png](output_52_2459.png)


     42%|█████████████████████████████████▎                                             | 820/1948 [18:13<59:16,  3.15s/it]cv 값 : 0.362981205425971람다 값 : 0.7669261212801108



![png](output_52_2462.png)


     42%|████████████████████████████████▍                                            | 821/1948 [18:18<1:04:30,  3.43s/it]cv 값 : 0.3719890920934932람다 값 : 0.8316549903007824



![png](output_52_2465.png)


     42%|████████████████████████████████▍                                            | 822/1948 [18:22<1:07:37,  3.60s/it]cv 값 : 0.8720412933740079람다 값 : 0.3509519621541303



![png](output_52_2468.png)


     42%|████████████████████████████████▌                                            | 823/1948 [18:24<1:01:40,  3.29s/it]cv 값 : 0.44383498848029람다 값 : -0.41855039142878764



![png](output_52_2471.png)


     42%|█████████████████████████████████▍                                             | 824/1948 [18:26<55:53,  2.98s/it]cv 값 : 0.47842390912030364람다 값 : -0.14443046949110103



![png](output_52_2474.png)


     42%|█████████████████████████████████▍                                             | 825/1948 [18:28<49:57,  2.67s/it]cv 값 : 0.4995109625018933람다 값 : 0.6131714740829638



![png](output_52_2477.png)


     42%|█████████████████████████████████▍                                             | 826/1948 [18:31<47:13,  2.53s/it]cv 값 : 0.2608111169656427람다 값 : 1.2795691939585279



![png](output_52_2480.png)


     42%|█████████████████████████████████▌                                             | 827/1948 [18:33<44:11,  2.37s/it]cv 값 : 0.6003827973246382람다 값 : 0.33677897977425536



![png](output_52_2483.png)


     43%|█████████████████████████████████▌                                             | 828/1948 [18:34<41:19,  2.21s/it]cv 값 : 0.2758568424746583람다 값 : -0.6099181864352002



![png](output_52_2486.png)


     43%|█████████████████████████████████▌                                             | 829/1948 [18:36<39:39,  2.13s/it]cv 값 : 0.14382047735470244람다 값 : -0.19780725154479673



![png](output_52_2489.png)


     43%|█████████████████████████████████▋                                             | 830/1948 [18:38<37:18,  2.00s/it]cv 값 : 0.1156467990608659람다 값 : 3.100113038384836



![png](output_52_2492.png)


     43%|█████████████████████████████████▋                                             | 831/1948 [18:40<38:32,  2.07s/it]cv 값 : 0.5029448490107623람다 값 : 0.5542847810770463



![png](output_52_2495.png)


     43%|█████████████████████████████████▋                                             | 832/1948 [18:42<35:05,  1.89s/it]cv 값 : 0.23226072876333212람다 값 : 0.5478886700218726



![png](output_52_2498.png)


     43%|█████████████████████████████████▊                                             | 833/1948 [18:44<35:26,  1.91s/it]cv 값 : 0.41707070238456584람다 값 : 0.21345810934986767



![png](output_52_2501.png)


     43%|█████████████████████████████████▊                                             | 834/1948 [18:45<34:34,  1.86s/it]cv 값 : 0.2781569151824074람다 값 : -1.1992247918946763



![png](output_52_2504.png)


     43%|█████████████████████████████████▊                                             | 835/1948 [18:47<34:56,  1.88s/it]cv 값 : 0.4120551942317412람다 값 : 0.41641858767439527



![png](output_52_2507.png)


     43%|█████████████████████████████████▉                                             | 836/1948 [18:49<34:42,  1.87s/it]cv 값 : 0.38453600801579907람다 값 : 0.9241352384826486



![png](output_52_2510.png)


     43%|█████████████████████████████████▉                                             | 837/1948 [18:51<36:34,  1.98s/it]cv 값 : 0.15780656043056163람다 값 : 0.5138197858801161



![png](output_52_2513.png)


     43%|█████████████████████████████████▉                                             | 838/1948 [18:53<35:26,  1.92s/it]cv 값 : 0.8092494529305422람다 값 : -0.052253107921993454



![png](output_52_2516.png)


     43%|██████████████████████████████████                                             | 839/1948 [18:55<34:59,  1.89s/it]cv 값 : 0.46456000893887517람다 값 : -0.032170011654810766



![png](output_52_2519.png)


     43%|██████████████████████████████████                                             | 840/1948 [18:57<34:05,  1.85s/it]cv 값 : 0.3149447754711782람다 값 : -0.6397608900638703



![png](output_52_2522.png)


     43%|██████████████████████████████████                                             | 841/1948 [18:59<34:40,  1.88s/it]cv 값 : 0.5692520446038999람다 값 : -0.14779088423078784



![png](output_52_2525.png)


     43%|██████████████████████████████████▏                                            | 842/1948 [19:01<34:43,  1.88s/it]cv 값 : 0.2499070441518403람다 값 : 0.2014467579601223



![png](output_52_2528.png)


     43%|██████████████████████████████████▏                                            | 843/1948 [19:03<34:54,  1.90s/it]cv 값 : 0.12543814454776797람다 값 : -1.306463956063783



![png](output_52_2531.png)


     43%|██████████████████████████████████▏                                            | 844/1948 [19:04<33:55,  1.84s/it]cv 값 : 0.4118368758839214람다 값 : 0.33866902227486223



![png](output_52_2534.png)


     43%|██████████████████████████████████▎                                            | 845/1948 [19:06<32:39,  1.78s/it]cv 값 : 0.20963143804496284람다 값 : -0.7264255990217016



![png](output_52_2537.png)


     43%|██████████████████████████████████▎                                            | 846/1948 [19:08<33:02,  1.80s/it]cv 값 : 0.3336853029145802람다 값 : 0.43695715744981295



![png](output_52_2540.png)


     43%|██████████████████████████████████▎                                            | 847/1948 [19:10<33:59,  1.85s/it]cv 값 : 0.2710839549954443람다 값 : -0.8789600843086868



![png](output_52_2543.png)


     44%|██████████████████████████████████▍                                            | 848/1948 [19:12<34:48,  1.90s/it]cv 값 : 0.2626162207595293람다 값 : 0.8906641355624753



![png](output_52_2546.png)


     44%|██████████████████████████████████▍                                            | 849/1948 [19:14<34:20,  1.88s/it]cv 값 : 0.14067580691998272람다 값 : 0.5827064216446631



![png](output_52_2549.png)


     44%|██████████████████████████████████▍                                            | 850/1948 [19:15<33:15,  1.82s/it]cv 값 : 0.29437888418348573람다 값 : 0.3395113342028942



![png](output_52_2552.png)


     44%|██████████████████████████████████▌                                            | 851/1948 [19:17<34:00,  1.86s/it]cv 값 : 0.3205415935352667람다 값 : 0.11601674781481938



![png](output_52_2555.png)


     44%|██████████████████████████████████▌                                            | 852/1948 [19:19<33:19,  1.82s/it]cv 값 : 0.24906456515094183람다 값 : -0.4881179910947531



![png](output_52_2558.png)


     44%|██████████████████████████████████▌                                            | 853/1948 [19:21<32:50,  1.80s/it]cv 값 : 1.9504461273486269람다 값 : -0.5872198817200865



![png](output_52_2561.png)


     44%|██████████████████████████████████▋                                            | 854/1948 [19:22<31:43,  1.74s/it]cv 값 : 0.16141958818063581람다 값 : -0.1271585208992832



![png](output_52_2564.png)


     44%|██████████████████████████████████▋                                            | 855/1948 [19:24<33:51,  1.86s/it]cv 값 : 0.21178921854152769람다 값 : 0.7746488547805452



![png](output_52_2567.png)


     44%|██████████████████████████████████▋                                            | 856/1948 [19:26<31:58,  1.76s/it]cv 값 : 0.39680235803559577람다 값 : 0.4778851730365701



![png](output_52_2570.png)


     44%|██████████████████████████████████▊                                            | 857/1948 [19:27<30:53,  1.70s/it]cv 값 : 0.261251739444365람다 값 : 0.7803903314033691



![png](output_52_2573.png)


     44%|██████████████████████████████████▊                                            | 858/1948 [19:29<30:32,  1.68s/it]cv 값 : 0.8187016942707029람다 값 : -0.5686461989979296



![png](output_52_2576.png)


     44%|██████████████████████████████████▊                                            | 859/1948 [19:31<32:01,  1.76s/it]cv 값 : 0.3682504848124524람다 값 : 0.9320088577789004



![png](output_52_2579.png)


     44%|██████████████████████████████████▉                                            | 860/1948 [19:33<30:55,  1.71s/it]cv 값 : 0.5132022487394186람다 값 : -0.38209274137661464



![png](output_52_2582.png)


     44%|██████████████████████████████████▉                                            | 861/1948 [19:35<31:46,  1.75s/it]cv 값 : 0.412762465144183람다 값 : 0.5469314476556664



![png](output_52_2585.png)


     44%|██████████████████████████████████▉                                            | 862/1948 [19:36<31:10,  1.72s/it]cv 값 : 0.25282954643201483람다 값 : 1.6938705159556258



![png](output_52_2588.png)


     44%|██████████████████████████████████▉                                            | 863/1948 [19:38<31:28,  1.74s/it]cv 값 : 0.19693104148488416람다 값 : -0.6933785507143655



![png](output_52_2591.png)


     44%|███████████████████████████████████                                            | 864/1948 [19:40<33:14,  1.84s/it]cv 값 : 0.2403740740857135람다 값 : -0.5450977768822295



![png](output_52_2594.png)


     44%|███████████████████████████████████                                            | 865/1948 [19:42<33:26,  1.85s/it]cv 값 : 0.37896594023277086람다 값 : -0.34884963530904634



![png](output_52_2597.png)


     44%|███████████████████████████████████                                            | 866/1948 [19:44<33:45,  1.87s/it]cv 값 : 0.2612215787940728람다 값 : 0.24052355037186254



![png](output_52_2600.png)


     45%|███████████████████████████████████▏                                           | 867/1948 [19:46<33:25,  1.86s/it]cv 값 : 0.17175047564889356람다 값 : 1.6138696995849522



![png](output_52_2603.png)


     45%|███████████████████████████████████▏                                           | 868/1948 [19:47<33:23,  1.86s/it]cv 값 : 0.25045379444164745람다 값 : -0.945405695650662



![png](output_52_2606.png)


     45%|███████████████████████████████████▏                                           | 869/1948 [19:49<33:34,  1.87s/it]cv 값 : 0.24179631151882708람다 값 : 0.724239992154147



![png](output_52_2609.png)


     45%|███████████████████████████████████▎                                           | 870/1948 [19:51<33:48,  1.88s/it]cv 값 : 0.5769660412146436람다 값 : 0.5463331782447897



![png](output_52_2612.png)


     45%|███████████████████████████████████▎                                           | 871/1948 [19:53<32:52,  1.83s/it]cv 값 : 0.23604668797187847람다 값 : -1.584329026755537



![png](output_52_2615.png)


     45%|███████████████████████████████████▎                                           | 872/1948 [19:55<33:11,  1.85s/it]cv 값 : 0.29226258971016467람다 값 : 0.3627166429549076



![png](output_52_2618.png)


     45%|███████████████████████████████████▍                                           | 873/1948 [19:57<34:37,  1.93s/it]cv 값 : 0.5938691539842148람다 값 : -0.520060373325103



![png](output_52_2621.png)


     45%|███████████████████████████████████▍                                           | 874/1948 [19:59<33:13,  1.86s/it]cv 값 : 0.3095501533294028람다 값 : 1.147704733895645



![png](output_52_2624.png)


     45%|███████████████████████████████████▍                                           | 875/1948 [20:00<32:34,  1.82s/it]cv 값 : 0.3253929320368409람다 값 : 0.39240627958054647



![png](output_52_2627.png)


     45%|███████████████████████████████████▌                                           | 876/1948 [20:02<31:58,  1.79s/it]cv 값 : 0.362909497761706람다 값 : 0.5920081551454758



![png](output_52_2630.png)


     45%|███████████████████████████████████▌                                           | 877/1948 [20:04<34:07,  1.91s/it]cv 값 : 0.3122857741647677람다 값 : 0.20334483859715324



![png](output_52_2633.png)


     45%|███████████████████████████████████▌                                           | 878/1948 [20:06<32:59,  1.85s/it]cv 값 : 0.22824265136631383람다 값 : -0.4847473943270556



![png](output_52_2636.png)


     45%|███████████████████████████████████▋                                           | 879/1948 [20:08<32:31,  1.83s/it]cv 값 : 0.4340473151757452람다 값 : 0.8476448456113943



![png](output_52_2639.png)


     45%|███████████████████████████████████▋                                           | 880/1948 [20:10<32:10,  1.81s/it]cv 값 : 0.3703986254721045람다 값 : -0.5239454557013643



![png](output_52_2642.png)


     45%|███████████████████████████████████▋                                           | 881/1948 [20:11<32:11,  1.81s/it]cv 값 : 0.2600128430236898람다 값 : 0.6003815452830412



![png](output_52_2645.png)


     45%|███████████████████████████████████▊                                           | 882/1948 [20:13<32:47,  1.85s/it]cv 값 : 0.3261023235253102람다 값 : 0.2937047334854707



![png](output_52_2648.png)


     45%|███████████████████████████████████▊                                           | 883/1948 [20:15<32:56,  1.86s/it]cv 값 : 0.44059708247763785람다 값 : 1.2732941005174019



![png](output_52_2651.png)


     45%|███████████████████████████████████▊                                           | 884/1948 [20:17<31:38,  1.78s/it]cv 값 : 0.19849652565313353람다 값 : -1.299826764474812



![png](output_52_2654.png)


     45%|███████████████████████████████████▉                                           | 885/1948 [20:19<31:28,  1.78s/it]cv 값 : 0.2451957804577577람다 값 : -0.20092154293210993



![png](output_52_2657.png)


     45%|███████████████████████████████████▉                                           | 886/1948 [20:21<32:20,  1.83s/it]cv 값 : 0.5254048649139016람다 값 : 0.6188342116569733



![png](output_52_2660.png)


     46%|███████████████████████████████████▉                                           | 887/1948 [20:22<32:11,  1.82s/it]cv 값 : 0.2287255094476083람다 값 : -1.0109848294091235



![png](output_52_2663.png)


     46%|████████████████████████████████████                                           | 888/1948 [20:24<33:04,  1.87s/it]cv 값 : 0.20336445065589803람다 값 : 0.048303953883097626



![png](output_52_2666.png)


     46%|████████████████████████████████████                                           | 889/1948 [20:26<32:56,  1.87s/it]cv 값 : 0.3553611164696771람다 값 : 1.1245601278027997



![png](output_52_2669.png)


     46%|████████████████████████████████████                                           | 890/1948 [20:28<33:31,  1.90s/it]cv 값 : 0.13785090889827534람다 값 : -0.5786351552898396



![png](output_52_2672.png)


     46%|████████████████████████████████████▏                                          | 891/1948 [20:30<33:34,  1.91s/it]cv 값 : 0.4725531738350121람다 값 : 0.008195846119255477



![png](output_52_2675.png)


     46%|████████████████████████████████████▏                                          | 892/1948 [20:32<31:30,  1.79s/it]cv 값 : 0.24171371570066605람다 값 : 1.8584618015503647



![png](output_52_2678.png)


     46%|████████████████████████████████████▏                                          | 893/1948 [20:34<31:57,  1.82s/it]cv 값 : 0.15516365371391175람다 값 : 2.1101976089094174



![png](output_52_2681.png)


     46%|████████████████████████████████████▎                                          | 894/1948 [20:35<32:20,  1.84s/it]cv 값 : 0.19986296580710314람다 값 : 1.6569887444252716



![png](output_52_2684.png)


     46%|████████████████████████████████████▎                                          | 895/1948 [20:37<32:21,  1.84s/it]cv 값 : 0.2902125734784584람다 값 : 1.145361646501091



![png](output_52_2687.png)


     46%|████████████████████████████████████▎                                          | 896/1948 [20:39<32:33,  1.86s/it]cv 값 : 0.5310597926695648람다 값 : 0.3821470123557649



![png](output_52_2690.png)


     46%|████████████████████████████████████▍                                          | 897/1948 [20:41<31:58,  1.83s/it]cv 값 : 0.37113248466685855람다 값 : -0.1608803508144031



![png](output_52_2693.png)


     46%|████████████████████████████████████▍                                          | 898/1948 [20:43<31:49,  1.82s/it]cv 값 : 0.3028628315231033람다 값 : -0.9422320723074109



![png](output_52_2696.png)


     46%|████████████████████████████████████▍                                          | 899/1948 [20:45<33:30,  1.92s/it]cv 값 : 0.21407170864604375람다 값 : 1.079824557285741



![png](output_52_2699.png)


     46%|████████████████████████████████████▍                                          | 900/1948 [20:46<31:47,  1.82s/it]cv 값 : 0.11185442019532485람다 값 : 3.846428812856938



![png](output_52_2702.png)


     46%|████████████████████████████████████▌                                          | 901/1948 [20:48<31:03,  1.78s/it]cv 값 : 0.15249281286556896람다 값 : 1.121763443623201



![png](output_52_2705.png)


     46%|████████████████████████████████████▌                                          | 902/1948 [20:50<31:30,  1.81s/it]cv 값 : 0.22002006181037181람다 값 : 1.0919263953750808



![png](output_52_2708.png)


     46%|████████████████████████████████████▌                                          | 903/1948 [20:52<32:02,  1.84s/it]cv 값 : 0.4221691980765625람다 값 : -0.452451933569214



![png](output_52_2711.png)


     46%|████████████████████████████████████▋                                          | 904/1948 [20:54<32:22,  1.86s/it]cv 값 : 0.2640814997374504람다 값 : 0.2315894868072029



![png](output_52_2714.png)


     46%|████████████████████████████████████▋                                          | 905/1948 [20:56<33:12,  1.91s/it]cv 값 : 0.3658320973772818람다 값 : 0.7680662119453942



![png](output_52_2717.png)


     47%|████████████████████████████████████▋                                          | 906/1948 [20:58<34:22,  1.98s/it]cv 값 : 0.26381950168916146람다 값 : -1.2662631070817367



![png](output_52_2720.png)


     47%|████████████████████████████████████▊                                          | 907/1948 [21:00<34:08,  1.97s/it]cv 값 : 0.5340080735821855람다 값 : 0.14025736865346755



![png](output_52_2723.png)


     47%|████████████████████████████████████▊                                          | 908/1948 [21:02<34:10,  1.97s/it]cv 값 : 0.39452647999902596람다 값 : 0.7864043451967893



![png](output_52_2726.png)


     47%|████████████████████████████████████▊                                          | 909/1948 [21:04<35:19,  2.04s/it]cv 값 : 0.7937652291504892람다 값 : 0.4051542159961485



![png](output_52_2729.png)


     47%|████████████████████████████████████▉                                          | 910/1948 [21:06<34:14,  1.98s/it]cv 값 : 0.14751790077931562람다 값 : 1.3623905261515146



![png](output_52_2732.png)


     47%|████████████████████████████████████▉                                          | 911/1948 [21:07<32:01,  1.85s/it]cv 값 : 0.18236304417699908람다 값 : -0.33769379365676055



![png](output_52_2735.png)


     47%|████████████████████████████████████▉                                          | 912/1948 [21:09<31:40,  1.83s/it]cv 값 : 0.343611968672639람다 값 : 0.7667346229032791



![png](output_52_2738.png)


     47%|█████████████████████████████████████                                          | 913/1948 [21:11<31:38,  1.83s/it]cv 값 : 0.5058563062510161람다 값 : 0.6144385120672231



![png](output_52_2741.png)


     47%|█████████████████████████████████████                                          | 914/1948 [21:13<31:35,  1.83s/it]cv 값 : 0.3314611515720394람다 값 : -0.4000255116719106



![png](output_52_2744.png)


     47%|█████████████████████████████████████                                          | 915/1948 [21:16<35:35,  2.07s/it]cv 값 : 0.5956190467962496람다 값 : 0.3061820769415153



![png](output_52_2747.png)


     47%|█████████████████████████████████████▏                                         | 916/1948 [21:19<42:54,  2.49s/it]cv 값 : 0.3219988894048216람다 값 : 0.7659255130638112



![png](output_52_2750.png)


     47%|█████████████████████████████████████▏                                         | 917/1948 [21:21<40:51,  2.38s/it]cv 값 : 0.8870384107234275람다 값 : 0.17776690411216592



![png](output_52_2753.png)


     47%|█████████████████████████████████████▏                                         | 918/1948 [21:23<36:35,  2.13s/it]cv 값 : 0.35063073985311144람다 값 : -0.05387579894428648



![png](output_52_2756.png)


     47%|█████████████████████████████████████▎                                         | 919/1948 [21:25<36:58,  2.16s/it]cv 값 : 0.3796787784819217람다 값 : -0.23429647848055785



![png](output_52_2759.png)


     47%|█████████████████████████████████████▎                                         | 920/1948 [21:27<34:56,  2.04s/it]cv 값 : 0.33242371484644173람다 값 : -0.4333346367760865



![png](output_52_2762.png)


     47%|█████████████████████████████████████▎                                         | 921/1948 [21:29<35:13,  2.06s/it]cv 값 : 0.6338111992461577람다 값 : 0.020794443065062916



![png](output_52_2765.png)


     47%|█████████████████████████████████████▍                                         | 922/1948 [21:31<34:24,  2.01s/it]cv 값 : 0.9250192287546475람다 값 : 0.1807508614753469



![png](output_52_2768.png)


     47%|█████████████████████████████████████▍                                         | 923/1948 [21:32<32:41,  1.91s/it]cv 값 : 0.4629425955715513람다 값 : 0.27284024614222385



![png](output_52_2771.png)


     47%|█████████████████████████████████████▍                                         | 924/1948 [21:34<32:33,  1.91s/it]cv 값 : 0.29664811805588737람다 값 : 0.8202026946511766



![png](output_52_2774.png)


     47%|█████████████████████████████████████▌                                         | 925/1948 [21:36<32:10,  1.89s/it]cv 값 : 0.4707185723200525람다 값 : 0.07377694514459548



![png](output_52_2777.png)


     48%|█████████████████████████████████████▌                                         | 926/1948 [21:38<31:49,  1.87s/it]cv 값 : 0.3601696804264233람다 값 : 0.8044555574870116



![png](output_52_2780.png)


     48%|█████████████████████████████████████▌                                         | 927/1948 [21:40<32:27,  1.91s/it]cv 값 : 0.18334739497629773람다 값 : -1.0904333940208926



![png](output_52_2783.png)


     48%|█████████████████████████████████████▋                                         | 928/1948 [21:42<31:22,  1.85s/it]cv 값 : 0.24593092812285602람다 값 : 0.6418811991066452



![png](output_52_2786.png)


     48%|█████████████████████████████████████▋                                         | 929/1948 [21:43<30:06,  1.77s/it]cv 값 : 0.43575224310838756람다 값 : 0.45289954029300206



![png](output_52_2789.png)


     48%|█████████████████████████████████████▋                                         | 930/1948 [21:45<30:23,  1.79s/it]cv 값 : 0.5451690141142366람다 값 : 0.0958442855439697



![png](output_52_2792.png)


     48%|█████████████████████████████████████▊                                         | 931/1948 [21:47<30:14,  1.78s/it]cv 값 : 0.27972437021534374람다 값 : 1.3257441791709632



![png](output_52_2795.png)


     48%|█████████████████████████████████████▊                                         | 932/1948 [21:49<31:24,  1.86s/it]cv 값 : 0.17846269774361706람다 값 : -0.24438010508359706



![png](output_52_2798.png)


     48%|█████████████████████████████████████▊                                         | 933/1948 [21:51<30:54,  1.83s/it]cv 값 : 0.21021439606670753람다 값 : 0.7463060714606324



![png](output_52_2801.png)


     48%|█████████████████████████████████████▉                                         | 934/1948 [21:52<29:34,  1.75s/it]cv 값 : 0.38292224203865477람다 값 : 0.5482516386935292



![png](output_52_2804.png)


     48%|█████████████████████████████████████▉                                         | 935/1948 [21:54<30:39,  1.82s/it]cv 값 : 0.236609418501577람다 값 : 1.1633315989880186



![png](output_52_2807.png)


     48%|█████████████████████████████████████▉                                         | 936/1948 [21:56<31:30,  1.87s/it]cv 값 : 0.3149274789644386람다 값 : 0.3946121346690648



![png](output_52_2810.png)


     48%|█████████████████████████████████████▉                                         | 937/1948 [21:58<30:43,  1.82s/it]cv 값 : 0.24161949088734636람다 값 : -0.5146449676837341



![png](output_52_2813.png)


     48%|██████████████████████████████████████                                         | 938/1948 [22:00<31:05,  1.85s/it]cv 값 : 0.4237814904565143람다 값 : 0.7811195360980273



![png](output_52_2816.png)


     48%|██████████████████████████████████████                                         | 939/1948 [22:02<32:05,  1.91s/it]cv 값 : 0.24944922404170086람다 값 : 0.03374071741289491



![png](output_52_2819.png)


     48%|██████████████████████████████████████                                         | 940/1948 [22:04<31:15,  1.86s/it]cv 값 : 0.3525050517771164람다 값 : 0.6859493186065047



![png](output_52_2822.png)


     48%|██████████████████████████████████████▏                                        | 941/1948 [22:06<31:46,  1.89s/it]cv 값 : 0.9844443143187042람다 값 : 0.2556551577756876



![png](output_52_2825.png)


     48%|██████████████████████████████████████▏                                        | 942/1948 [22:08<35:06,  2.09s/it]cv 값 : 0.2981899294007965람다 값 : 0.9884259068999051



![png](output_52_2828.png)


     48%|██████████████████████████████████████▏                                        | 943/1948 [22:11<40:24,  2.41s/it]cv 값 : 0.38272949721425836람다 값 : -0.007175623108753963



![png](output_52_2831.png)


     48%|██████████████████████████████████████▎                                        | 944/1948 [22:13<39:11,  2.34s/it]cv 값 : 0.8971342503413224람다 값 : 0.33427720156251306



![png](output_52_2834.png)


     49%|██████████████████████████████████████▎                                        | 945/1948 [22:16<39:42,  2.38s/it]cv 값 : 0.4886593675768734람다 값 : -0.07669406109930753



![png](output_52_2837.png)


     49%|██████████████████████████████████████▎                                        | 946/1948 [22:18<39:11,  2.35s/it]cv 값 : 0.23883990824279663람다 값 : 0.5846494446975203



![png](output_52_2840.png)


     49%|██████████████████████████████████████▍                                        | 947/1948 [22:20<38:37,  2.31s/it]cv 값 : 0.3550434450485976람다 값 : 0.4518731991940614



![png](output_52_2843.png)


     49%|██████████████████████████████████████▍                                        | 948/1948 [22:23<37:34,  2.25s/it]cv 값 : 0.16040371802302572람다 값 : 1.6122179087257609



![png](output_52_2846.png)


     49%|██████████████████████████████████████▍                                        | 949/1948 [22:25<37:45,  2.27s/it]cv 값 : 0.7078819406435269람다 값 : 0.345309434193344



![png](output_52_2849.png)


     49%|██████████████████████████████████████▌                                        | 950/1948 [22:27<37:16,  2.24s/it]cv 값 : 0.17162337985322046람다 값 : 0.5315778380843492



![png](output_52_2852.png)


     49%|██████████████████████████████████████▌                                        | 951/1948 [22:30<38:46,  2.33s/it]cv 값 : 0.5004962189454107람다 값 : 0.6472274259387916



![png](output_52_2855.png)


     49%|██████████████████████████████████████▌                                        | 952/1948 [22:32<37:26,  2.26s/it]cv 값 : 0.3305272460659559람다 값 : 0.15200650276551883



![png](output_52_2858.png)


     49%|██████████████████████████████████████▋                                        | 953/1948 [22:35<44:29,  2.68s/it]cv 값 : 0.19349923340114503람다 값 : 0.551141652982188



![png](output_52_2861.png)


     49%|██████████████████████████████████████▋                                        | 954/1948 [22:37<41:49,  2.53s/it]cv 값 : 0.3889057156671778람다 값 : 0.5164495455166378



![png](output_52_2864.png)


     49%|██████████████████████████████████████▋                                        | 955/1948 [22:39<38:50,  2.35s/it]cv 값 : 0.45858655790037567람다 값 : -0.27253378853204185



![png](output_52_2867.png)


     49%|██████████████████████████████████████▊                                        | 956/1948 [22:41<36:19,  2.20s/it]cv 값 : 0.5517682000302165람다 값 : 0.5994950407848476



![png](output_52_2870.png)


     49%|██████████████████████████████████████▊                                        | 957/1948 [22:43<35:14,  2.13s/it]cv 값 : 0.3726656673765182람다 값 : 0.7634899759191976



![png](output_52_2873.png)


     49%|██████████████████████████████████████▊                                        | 958/1948 [22:45<34:06,  2.07s/it]cv 값 : 0.3554060927447713람다 값 : -0.011361039502125681



![png](output_52_2876.png)


     49%|██████████████████████████████████████▉                                        | 959/1948 [22:47<34:24,  2.09s/it]cv 값 : 0.5406059362862466람다 값 : 0.36003161461704164



![png](output_52_2879.png)


     49%|██████████████████████████████████████▉                                        | 960/1948 [22:49<34:03,  2.07s/it]cv 값 : 0.29412484100259373람다 값 : -0.13231632268780089



![png](output_52_2882.png)


     49%|██████████████████████████████████████▉                                        | 961/1948 [22:51<34:03,  2.07s/it]cv 값 : 0.43850899095427115람다 값 : 0.44303675794434366



![png](output_52_2885.png)


     49%|███████████████████████████████████████                                        | 962/1948 [22:53<32:06,  1.95s/it]cv 값 : 0.46340594005407876람다 값 : -0.43683117378564434



![png](output_52_2888.png)


     49%|███████████████████████████████████████                                        | 963/1948 [22:55<32:00,  1.95s/it]cv 값 : 0.18438689044945147람다 값 : 0.24250278554691304



![png](output_52_2891.png)


     49%|███████████████████████████████████████                                        | 964/1948 [22:57<30:17,  1.85s/it]cv 값 : 0.44816973718874775람다 값 : 0.6646220700304963



![png](output_52_2894.png)


     50%|███████████████████████████████████████▏                                       | 965/1948 [22:58<29:18,  1.79s/it]cv 값 : 0.21508687583605854람다 값 : 0.5596668070702788



![png](output_52_2897.png)


     50%|███████████████████████████████████████▏                                       | 966/1948 [23:00<30:34,  1.87s/it]cv 값 : 0.24044005356094325람다 값 : 0.2569548392946156



![png](output_52_2900.png)


     50%|███████████████████████████████████████▏                                       | 967/1948 [23:02<29:42,  1.82s/it]cv 값 : 0.3083058891204803람다 값 : 0.5706783490362344



![png](output_52_2903.png)


     50%|███████████████████████████████████████▎                                       | 968/1948 [23:04<30:11,  1.85s/it]cv 값 : 0.29321581438935196람다 값 : 1.0860131263166632



![png](output_52_2906.png)


     50%|███████████████████████████████████████▎                                       | 969/1948 [23:06<33:24,  2.05s/it]cv 값 : 0.34666775200064537람다 값 : 1.3448612037279368



![png](output_52_2909.png)


     50%|███████████████████████████████████████▎                                       | 970/1948 [23:08<32:15,  1.98s/it]cv 값 : 0.21546227742775256람다 값 : 1.0894538447851718



![png](output_52_2912.png)


     50%|███████████████████████████████████████▍                                       | 971/1948 [23:10<32:48,  2.01s/it]cv 값 : 0.2165648253092654람다 값 : 0.9923057210012434



![png](output_52_2915.png)


     50%|███████████████████████████████████████▍                                       | 972/1948 [23:12<31:15,  1.92s/it]cv 값 : 0.2992743952940569람다 값 : 0.9210356249631229



![png](output_52_2918.png)


     50%|███████████████████████████████████████▍                                       | 973/1948 [23:14<30:12,  1.86s/it]cv 값 : 0.16471561236754623람다 값 : -0.6421644642547744



![png](output_52_2921.png)


     50%|███████████████████████████████████████▌                                       | 974/1948 [23:16<29:44,  1.83s/it]cv 값 : 0.21761591868149507람다 값 : 0.1854234567316884



![png](output_52_2924.png)


     50%|███████████████████████████████████████▌                                       | 975/1948 [23:17<30:01,  1.85s/it]cv 값 : 0.16239432225540204람다 값 : 0.39708620079880663



![png](output_52_2927.png)


     50%|███████████████████████████████████████▌                                       | 976/1948 [23:19<29:31,  1.82s/it]cv 값 : 0.20614637636220526람다 값 : 0.05087579337279545



![png](output_52_2930.png)


     50%|███████████████████████████████████████▌                                       | 977/1948 [23:21<29:17,  1.81s/it]cv 값 : 0.393689791964269람다 값 : 0.6229140427531537



![png](output_52_2933.png)


     50%|███████████████████████████████████████▋                                       | 978/1948 [23:23<28:44,  1.78s/it]cv 값 : 0.2990420268762997람다 값 : 0.1804489841031833



![png](output_52_2936.png)


     50%|███████████████████████████████████████▋                                       | 979/1948 [23:25<30:20,  1.88s/it]cv 값 : 0.9590786567705023람다 값 : 0.32778760024970754



![png](output_52_2939.png)


     50%|███████████████████████████████████████▋                                       | 980/1948 [23:27<29:56,  1.86s/it]cv 값 : 0.5824788411440263람다 값 : -0.013385429791740287



![png](output_52_2942.png)


     50%|███████████████████████████████████████▊                                       | 981/1948 [23:28<28:51,  1.79s/it]cv 값 : 0.6340385496874249람다 값 : -0.17493484346308041



![png](output_52_2945.png)


     50%|███████████████████████████████████████▊                                       | 982/1948 [23:30<29:04,  1.81s/it]cv 값 : 0.2547137223500536람다 값 : -0.18672643528933547



![png](output_52_2948.png)


     50%|███████████████████████████████████████▊                                       | 983/1948 [23:32<29:00,  1.80s/it]cv 값 : 0.22586146489926298람다 값 : 0.9852787883886148



![png](output_52_2951.png)


     51%|███████████████████████████████████████▉                                       | 984/1948 [23:34<29:21,  1.83s/it]cv 값 : 0.3774380016683003람다 값 : -0.3384512699983851



![png](output_52_2954.png)


     51%|███████████████████████████████████████▉                                       | 985/1948 [23:36<30:19,  1.89s/it]cv 값 : 0.4319965944453191람다 값 : 0.6605462971867384



![png](output_52_2957.png)


     51%|███████████████████████████████████████▉                                       | 986/1948 [23:37<29:11,  1.82s/it]cv 값 : 0.28941038259394436람다 값 : 0.0043798722971413505



![png](output_52_2960.png)


     51%|████████████████████████████████████████                                       | 987/1948 [23:39<29:35,  1.85s/it]cv 값 : 0.28960437353196367람다 값 : 0.9907887489184845



![png](output_52_2963.png)


     51%|████████████████████████████████████████                                       | 988/1948 [23:41<29:42,  1.86s/it]cv 값 : 0.3912697648354739람다 값 : 0.6996395347707351



![png](output_52_2966.png)


     51%|████████████████████████████████████████                                       | 989/1948 [23:43<28:37,  1.79s/it]cv 값 : 0.2302536904544485람다 값 : 0.6833941972649761



![png](output_52_2969.png)


     51%|████████████████████████████████████████▏                                      | 990/1948 [23:45<32:04,  2.01s/it]cv 값 : 0.338236674678481람다 값 : 0.6891212100104385



![png](output_52_2972.png)


     51%|████████████████████████████████████████▏                                      | 991/1948 [23:47<31:34,  1.98s/it]cv 값 : 0.622187953007072람다 값 : 0.3664759680365889



![png](output_52_2975.png)


     51%|████████████████████████████████████████▏                                      | 992/1948 [23:49<32:08,  2.02s/it]cv 값 : 0.3624726274284434람다 값 : 0.6501376717422934



![png](output_52_2978.png)


     51%|████████████████████████████████████████▎                                      | 993/1948 [23:51<30:56,  1.94s/it]cv 값 : 1.2360508039439908람다 값 : 0.2540503371384295



![png](output_52_2981.png)


     51%|████████████████████████████████████████▎                                      | 994/1948 [23:53<30:32,  1.92s/it]cv 값 : 0.570779965685481람다 값 : 0.01821112731289881



![png](output_52_2984.png)


     51%|████████████████████████████████████████▎                                      | 995/1948 [23:55<31:19,  1.97s/it]cv 값 : 0.5183223503741917람다 값 : 0.8600922558875377



![png](output_52_2987.png)


     51%|████████████████████████████████████████▍                                      | 996/1948 [23:57<30:33,  1.93s/it]cv 값 : 0.34315135477015457람다 값 : 0.07903622110381886



![png](output_52_2990.png)


     51%|████████████████████████████████████████▍                                      | 997/1948 [23:59<30:05,  1.90s/it]cv 값 : 0.34456994505676714람다 값 : -0.4339864655061497



![png](output_52_2993.png)


     51%|████████████████████████████████████████▍                                      | 998/1948 [24:01<29:38,  1.87s/it]cv 값 : 0.30274743983288493람다 값 : 1.0057125306244636



![png](output_52_2996.png)


     51%|████████████████████████████████████████▌                                      | 999/1948 [24:02<28:13,  1.78s/it]cv 값 : 0.28825546409294717람다 값 : 0.550436533116048



![png](output_52_2999.png)


     51%|████████████████████████████████████████                                      | 1000/1948 [24:06<36:57,  2.34s/it]cv 값 : 0.34706859396574724람다 값 : 0.833748691832286



![png](output_52_3002.png)


     51%|████████████████████████████████████████                                      | 1001/1948 [24:08<34:37,  2.19s/it]cv 값 : 0.3114141200989155람다 값 : 0.8504023436757358



![png](output_52_3005.png)


     51%|████████████████████████████████████████                                      | 1002/1948 [24:10<33:07,  2.10s/it]cv 값 : 0.4900395935761561람다 값 : 0.6374876796996533



![png](output_52_3008.png)


     51%|████████████████████████████████████████▏                                     | 1003/1948 [24:12<32:18,  2.05s/it]cv 값 : 0.3421345860561566람다 값 : 0.8446015132301685



![png](output_52_3011.png)


     52%|████████████████████████████████████████▏                                     | 1004/1948 [24:13<30:35,  1.94s/it]cv 값 : 0.3904584422785775람다 값 : 0.4143664306598697



![png](output_52_3014.png)


     52%|████████████████████████████████████████▏                                     | 1005/1948 [24:15<29:15,  1.86s/it]cv 값 : 0.42501239244255024람다 값 : 0.4207451695329149



![png](output_52_3017.png)


     52%|████████████████████████████████████████▎                                     | 1006/1948 [24:17<30:00,  1.91s/it]cv 값 : 0.35096261751904095람다 값 : 0.2926651687356047



![png](output_52_3020.png)


     52%|████████████████████████████████████████▎                                     | 1007/1948 [24:19<29:02,  1.85s/it]cv 값 : 0.2581783192667498람다 값 : -0.4575293551434562



![png](output_52_3023.png)


     52%|████████████████████████████████████████▎                                     | 1008/1948 [24:20<28:49,  1.84s/it]cv 값 : 0.4422720276983544람다 값 : 0.4956417930611897



![png](output_52_3026.png)


     52%|████████████████████████████████████████▍                                     | 1009/1948 [24:22<28:17,  1.81s/it]cv 값 : 1.318011238689691람다 값 : -0.6122584006041409



![png](output_52_3029.png)


     52%|████████████████████████████████████████▍                                     | 1010/1948 [24:24<29:33,  1.89s/it]cv 값 : 0.543485972361078람다 값 : -0.3482448603772231



![png](output_52_3032.png)


     52%|████████████████████████████████████████▍                                     | 1011/1948 [24:26<29:32,  1.89s/it]cv 값 : 0.48370048057728027람다 값 : -0.1096916747810187



![png](output_52_3035.png)


     52%|████████████████████████████████████████▌                                     | 1012/1948 [24:28<28:58,  1.86s/it]cv 값 : 0.3456319549284382람다 값 : 0.7522854940754543



![png](output_52_3038.png)


     52%|████████████████████████████████████████▌                                     | 1013/1948 [24:30<30:08,  1.93s/it]cv 값 : 0.3775829427907139람다 값 : 0.07323597308625035



![png](output_52_3041.png)


     52%|████████████████████████████████████████▌                                     | 1014/1948 [24:32<28:59,  1.86s/it]cv 값 : 0.4588999930498942람다 값 : 0.35524975823174926



![png](output_52_3044.png)


     52%|████████████████████████████████████████▋                                     | 1015/1948 [24:34<29:49,  1.92s/it]cv 값 : 0.6862470515509593람다 값 : 0.04350363723489848



![png](output_52_3047.png)


     52%|████████████████████████████████████████▋                                     | 1016/1948 [24:35<28:17,  1.82s/it]cv 값 : 0.24030710551389436람다 값 : 0.8000986166772873



![png](output_52_3050.png)


     52%|████████████████████████████████████████▋                                     | 1017/1948 [24:37<27:40,  1.78s/it]cv 값 : 0.5206582439540401람다 값 : -1.3869394851989656



![png](output_52_3053.png)


     52%|████████████████████████████████████████▊                                     | 1018/1948 [24:39<27:57,  1.80s/it]cv 값 : 0.4657836278803421람다 값 : 0.9174595002424648



![png](output_52_3056.png)


     52%|████████████████████████████████████████▊                                     | 1019/1948 [24:41<29:16,  1.89s/it]cv 값 : 0.5187124564620575람다 값 : -0.3394756705656778



![png](output_52_3059.png)


     52%|████████████████████████████████████████▊                                     | 1020/1948 [24:43<28:03,  1.81s/it]cv 값 : 0.2899907389871794람다 값 : -0.12884758831712229



![png](output_52_3062.png)


     52%|████████████████████████████████████████▉                                     | 1021/1948 [24:44<28:02,  1.82s/it]cv 값 : 0.47048788709666817람다 값 : -0.09253077326432005



![png](output_52_3065.png)


     52%|████████████████████████████████████████▉                                     | 1022/1948 [24:46<28:09,  1.82s/it]cv 값 : 0.30912163412470545람다 값 : -0.622525835154594



![png](output_52_3068.png)


     53%|████████████████████████████████████████▉                                     | 1023/1948 [24:48<27:42,  1.80s/it]cv 값 : 0.5441466814042856람다 값 : 0.7294873191940457



![png](output_52_3071.png)


     53%|█████████████████████████████████████████                                     | 1024/1948 [24:50<28:05,  1.82s/it]cv 값 : 0.39911691095266005람다 값 : 0.7158531382914936



![png](output_52_3074.png)


     53%|█████████████████████████████████████████                                     | 1025/1948 [24:51<26:39,  1.73s/it]cv 값 : 0.2242225926820125람다 값 : 1.1785859447640454



![png](output_52_3077.png)


     53%|█████████████████████████████████████████                                     | 1026/1948 [24:53<26:03,  1.70s/it]cv 값 : 0.4759132829385942람다 값 : 0.7594236563260531



![png](output_52_3080.png)


     53%|█████████████████████████████████████████                                     | 1027/1948 [24:55<26:40,  1.74s/it]cv 값 : 0.4487403630517061람다 값 : 0.565784770134746



![png](output_52_3083.png)


     53%|█████████████████████████████████████████▏                                    | 1028/1948 [24:57<26:56,  1.76s/it]cv 값 : 0.4392882406222224람다 값 : 0.27470727876645545



![png](output_52_3086.png)


     53%|█████████████████████████████████████████▏                                    | 1029/1948 [24:58<27:02,  1.77s/it]cv 값 : 0.272614983893272람다 값 : 0.8507617160075702



![png](output_52_3089.png)


     53%|█████████████████████████████████████████▏                                    | 1030/1948 [25:00<27:13,  1.78s/it]cv 값 : 0.3108404455218665람다 값 : 0.8790361180370407



![png](output_52_3092.png)


     53%|█████████████████████████████████████████▎                                    | 1031/1948 [25:02<26:38,  1.74s/it]cv 값 : 0.6772211747616305람다 값 : -0.02104716170901993



![png](output_52_3095.png)


     53%|█████████████████████████████████████████▎                                    | 1032/1948 [25:04<29:07,  1.91s/it]cv 값 : 0.5254489473183382람다 값 : 0.03174610987182066



![png](output_52_3098.png)


     53%|█████████████████████████████████████████▎                                    | 1033/1948 [25:06<30:18,  1.99s/it]cv 값 : 0.22817274118236255람다 값 : 0.23636771738004578



![png](output_52_3101.png)


     53%|█████████████████████████████████████████▍                                    | 1034/1948 [25:08<29:10,  1.92s/it]cv 값 : 0.3362780559327284람다 값 : 1.2878865443248562



![png](output_52_3104.png)


     53%|█████████████████████████████████████████▍                                    | 1035/1948 [25:10<29:54,  1.97s/it]cv 값 : 0.487840179170061람다 값 : 0.13167944503680457



![png](output_52_3107.png)


     53%|█████████████████████████████████████████▍                                    | 1036/1948 [25:12<28:40,  1.89s/it]cv 값 : 0.32603218729276384람다 값 : 0.3816607225065841



![png](output_52_3110.png)


     53%|█████████████████████████████████████████▌                                    | 1037/1948 [25:14<28:54,  1.90s/it]cv 값 : 0.30025763642118575람다 값 : 0.6260471197978986



![png](output_52_3113.png)


     53%|█████████████████████████████████████████▌                                    | 1038/1948 [25:16<28:55,  1.91s/it]cv 값 : 0.2776670100551258람다 값 : 0.7790527644842581



![png](output_52_3116.png)


     53%|█████████████████████████████████████████▌                                    | 1039/1948 [25:18<28:23,  1.87s/it]cv 값 : 0.9000535809689509람다 값 : 0.10927569394758407



![png](output_52_3119.png)


     53%|█████████████████████████████████████████▋                                    | 1040/1948 [25:19<28:02,  1.85s/it]cv 값 : 0.3060426199571554람다 값 : 0.16329601201247998



![png](output_52_3122.png)


     53%|█████████████████████████████████████████▋                                    | 1041/1948 [25:21<28:19,  1.87s/it]cv 값 : 0.4108944812842967람다 값 : 0.08729471426896238



![png](output_52_3125.png)


     53%|█████████████████████████████████████████▋                                    | 1042/1948 [25:23<27:29,  1.82s/it]cv 값 : 0.48678496806410854람다 값 : 0.17608497773468604



![png](output_52_3128.png)


     54%|█████████████████████████████████████████▊                                    | 1043/1948 [25:25<28:04,  1.86s/it]cv 값 : 0.9423805328091348람다 값 : 0.00974257145617814



![png](output_52_3131.png)


     54%|█████████████████████████████████████████▊                                    | 1044/1948 [25:27<27:08,  1.80s/it]cv 값 : 0.6981643965300716람다 값 : 0.4520838013950287



![png](output_52_3134.png)


     54%|█████████████████████████████████████████▊                                    | 1045/1948 [25:28<26:52,  1.79s/it]cv 값 : 1.1193100640221825람다 값 : 0.22539029746487724



![png](output_52_3137.png)


     54%|█████████████████████████████████████████▉                                    | 1046/1948 [25:31<28:57,  1.93s/it]cv 값 : 0.374849634868135람다 값 : 0.31528675175664056



![png](output_52_3140.png)


     54%|█████████████████████████████████████████▉                                    | 1047/1948 [25:32<27:43,  1.85s/it]cv 값 : 0.36725410198114167람다 값 : 0.819354627151429



![png](output_52_3143.png)


     54%|█████████████████████████████████████████▉                                    | 1048/1948 [25:34<27:39,  1.84s/it]cv 값 : 0.6215876565406089람다 값 : 0.4898015728043911



![png](output_52_3146.png)


     54%|██████████████████████████████████████████                                    | 1049/1948 [25:36<26:12,  1.75s/it]cv 값 : 0.3258274820403802람다 값 : 0.2975009819109752



![png](output_52_3149.png)


     54%|██████████████████████████████████████████                                    | 1050/1948 [25:38<26:34,  1.78s/it]cv 값 : 0.13723971259510281람다 값 : 2.860310426984756



![png](output_52_3152.png)


     54%|██████████████████████████████████████████                                    | 1051/1948 [25:39<27:18,  1.83s/it]cv 값 : 0.7042916621636967람다 값 : 0.2394160662315486



![png](output_52_3155.png)


     54%|██████████████████████████████████████████                                    | 1052/1948 [25:41<27:00,  1.81s/it]cv 값 : 0.569788701127413람다 값 : 0.2570196016266074



![png](output_52_3158.png)


     54%|██████████████████████████████████████████▏                                   | 1053/1948 [25:43<26:35,  1.78s/it]cv 값 : 0.1696550323582809람다 값 : 0.6452539733034695



![png](output_52_3161.png)


     54%|██████████████████████████████████████████▏                                   | 1054/1948 [25:45<25:52,  1.74s/it]cv 값 : 0.19586858087133815람다 값 : 0.2389883546588699



![png](output_52_3164.png)


     54%|██████████████████████████████████████████▏                                   | 1055/1948 [25:46<26:13,  1.76s/it]cv 값 : 0.23237010474269718람다 값 : -1.5697078039033514



![png](output_52_3167.png)


     54%|██████████████████████████████████████████▎                                   | 1056/1948 [25:48<26:13,  1.76s/it]cv 값 : 0.13611570572476028람다 값 : 1.106835667132867



![png](output_52_3170.png)


     54%|██████████████████████████████████████████▎                                   | 1057/1948 [25:50<28:35,  1.93s/it]cv 값 : 1.1571664165189366람다 값 : 0.1766673514024945



![png](output_52_3173.png)


     54%|██████████████████████████████████████████▎                                   | 1058/1948 [25:53<30:35,  2.06s/it]cv 값 : 0.16527571453016815람다 값 : 1.6490115954541262



![png](output_52_3176.png)


     54%|██████████████████████████████████████████▍                                   | 1059/1948 [25:55<29:55,  2.02s/it]cv 값 : 0.6649336177881452람다 값 : -0.5518682001662998



![png](output_52_3179.png)


     54%|██████████████████████████████████████████▍                                   | 1060/1948 [25:57<29:18,  1.98s/it]cv 값 : 0.4196939829294755람다 값 : 0.5520092754189094



![png](output_52_3182.png)


     54%|██████████████████████████████████████████▍                                   | 1061/1948 [25:58<28:09,  1.90s/it]cv 값 : 0.5140741347820237람다 값 : -0.0978252318339657



![png](output_52_3185.png)


     55%|██████████████████████████████████████████▌                                   | 1062/1948 [26:00<28:10,  1.91s/it]cv 값 : 0.5539316255440654람다 값 : 0.5474865695329361



![png](output_52_3188.png)


     55%|██████████████████████████████████████████▌                                   | 1063/1948 [26:02<29:02,  1.97s/it]cv 값 : 0.5506306332949836람다 값 : -0.2497491618295909



![png](output_52_3191.png)


     55%|██████████████████████████████████████████▌                                   | 1064/1948 [26:04<28:12,  1.91s/it]cv 값 : 0.4140409328721152람다 값 : 1.3679536069177112



![png](output_52_3194.png)


     55%|██████████████████████████████████████████▋                                   | 1065/1948 [26:06<28:22,  1.93s/it]cv 값 : 0.4858129317890502람다 값 : 0.1315236767545873



![png](output_52_3197.png)


     55%|██████████████████████████████████████████▋                                   | 1066/1948 [26:08<27:47,  1.89s/it]cv 값 : 0.5331399190244207람다 값 : -0.19721434092970144



![png](output_52_3200.png)


     55%|██████████████████████████████████████████▋                                   | 1067/1948 [26:10<27:08,  1.85s/it]cv 값 : 0.3688799250494083람다 값 : 0.3537985467780007



![png](output_52_3203.png)


     55%|██████████████████████████████████████████▊                                   | 1068/1948 [26:12<27:45,  1.89s/it]cv 값 : 0.6802285818085835람다 값 : 0.3327917205045269



![png](output_52_3206.png)


     55%|██████████████████████████████████████████▊                                   | 1069/1948 [26:13<26:53,  1.84s/it]cv 값 : 0.42974007787069873람다 값 : 0.471227426860022



![png](output_52_3209.png)


     55%|██████████████████████████████████████████▊                                   | 1070/1948 [26:15<26:21,  1.80s/it]cv 값 : 1.2267438278384977람다 값 : 0.23901655846182387



![png](output_52_3212.png)


     55%|██████████████████████████████████████████▉                                   | 1071/1948 [26:17<25:23,  1.74s/it]cv 값 : 0.5022888444559852람다 값 : 0.6304122415611946



![png](output_52_3215.png)


     55%|██████████████████████████████████████████▉                                   | 1072/1948 [26:19<26:39,  1.83s/it]cv 값 : 0.5668347449611534람다 값 : -0.43293119198525576



![png](output_52_3218.png)


     55%|██████████████████████████████████████████▉                                   | 1073/1948 [26:21<26:55,  1.85s/it]cv 값 : 0.4123719117030296람다 값 : -0.2473507977569736



![png](output_52_3221.png)


     55%|███████████████████████████████████████████                                   | 1074/1948 [26:23<27:16,  1.87s/it]cv 값 : 0.29518496358674867람다 값 : -0.12154278238171973



![png](output_52_3224.png)


     55%|███████████████████████████████████████████                                   | 1075/1948 [26:25<27:39,  1.90s/it]cv 값 : 0.18582829617276778람다 값 : 2.103805257904889



![png](output_52_3227.png)


     55%|███████████████████████████████████████████                                   | 1076/1948 [26:26<27:45,  1.91s/it]cv 값 : 0.5747849624223321람다 값 : 0.3775497837757839



![png](output_52_3230.png)


     55%|███████████████████████████████████████████                                   | 1077/1948 [26:28<27:51,  1.92s/it]cv 값 : 0.36924941253960814람다 값 : 0.7323559613965067



![png](output_52_3233.png)


     55%|███████████████████████████████████████████▏                                  | 1078/1948 [26:30<27:41,  1.91s/it]cv 값 : 0.5755606274384716람다 값 : -0.5899738835048158



![png](output_52_3236.png)


     55%|███████████████████████████████████████████▏                                  | 1079/1948 [26:32<27:11,  1.88s/it]cv 값 : 0.33338968856894347람다 값 : 0.7489506976681125



![png](output_52_3239.png)


     55%|███████████████████████████████████████████▏                                  | 1080/1948 [26:34<27:03,  1.87s/it]cv 값 : 0.5083379362361198람다 값 : 0.21925398246754704



![png](output_52_3242.png)


     55%|███████████████████████████████████████████▎                                  | 1081/1948 [26:36<27:09,  1.88s/it]cv 값 : 0.6338011091887863람다 값 : -0.9198351439257084



![png](output_52_3245.png)


     56%|███████████████████████████████████████████▎                                  | 1082/1948 [26:38<26:47,  1.86s/it]cv 값 : 0.7259482503625495람다 값 : -0.01773206847250674



![png](output_52_3248.png)


     56%|███████████████████████████████████████████▎                                  | 1083/1948 [26:40<26:46,  1.86s/it]cv 값 : 0.1911983659992189람다 값 : 0.8296004035311099



![png](output_52_3251.png)


     56%|███████████████████████████████████████████▍                                  | 1084/1948 [26:41<25:51,  1.80s/it]cv 값 : 0.724985801710467람다 값 : 0.45240551706109233



![png](output_52_3254.png)


     56%|███████████████████████████████████████████▍                                  | 1085/1948 [26:43<27:05,  1.88s/it]cv 값 : 0.3182124359590879람다 값 : -0.8828877661464293



![png](output_52_3257.png)


     56%|███████████████████████████████████████████▍                                  | 1086/1948 [26:45<27:23,  1.91s/it]cv 값 : 0.28374399914637516람다 값 : 0.8306845290311451



![png](output_52_3260.png)


     56%|███████████████████████████████████████████▌                                  | 1087/1948 [26:47<27:11,  1.89s/it]cv 값 : 0.3319511943299072람다 값 : 0.19505625129612056



![png](output_52_3263.png)


     56%|███████████████████████████████████████████▌                                  | 1088/1948 [26:49<27:11,  1.90s/it]cv 값 : 0.14338482757323004람다 값 : 0.900027617753798



![png](output_52_3266.png)


     56%|███████████████████████████████████████████▌                                  | 1089/1948 [26:51<26:27,  1.85s/it]cv 값 : 0.946968479611488람다 값 : -0.024502030387832788



![png](output_52_3269.png)


     56%|███████████████████████████████████████████▋                                  | 1090/1948 [26:53<27:05,  1.89s/it]cv 값 : 0.4930324878611531람다 값 : 0.49798490625184616



![png](output_52_3272.png)


     56%|███████████████████████████████████████████▋                                  | 1091/1948 [26:55<27:04,  1.90s/it]cv 값 : 0.520771544756177람다 값 : -0.7297956309602022



![png](output_52_3275.png)


     56%|███████████████████████████████████████████▋                                  | 1092/1948 [26:56<26:49,  1.88s/it]cv 값 : 0.4417699830874326람다 값 : 0.6388875007250638



![png](output_52_3278.png)


     56%|███████████████████████████████████████████▊                                  | 1093/1948 [26:58<25:01,  1.76s/it]cv 값 : 0.9844471124190686람다 값 : -0.8251796282223387



![png](output_52_3281.png)


     56%|███████████████████████████████████████████▊                                  | 1094/1948 [27:01<29:46,  2.09s/it]cv 값 : 0.34477523137538274람다 값 : 0.3266225852886901



![png](output_52_3284.png)


     56%|███████████████████████████████████████████▊                                  | 1095/1948 [27:02<27:29,  1.93s/it]cv 값 : 0.2575981970257264람다 값 : 1.3594651948959515



![png](output_52_3287.png)


     56%|███████████████████████████████████████████▉                                  | 1096/1948 [27:04<26:48,  1.89s/it]cv 값 : 0.37688180356887657람다 값 : 0.04512471057235302



![png](output_52_3290.png)


     56%|███████████████████████████████████████████▉                                  | 1097/1948 [27:06<26:14,  1.85s/it]cv 값 : 0.3451037954379036람다 값 : 0.31961045740875277



![png](output_52_3293.png)


     56%|███████████████████████████████████████████▉                                  | 1098/1948 [27:08<26:20,  1.86s/it]cv 값 : 0.6903051557808157람다 값 : 0.4384135724286044



![png](output_52_3296.png)


     56%|████████████████████████████████████████████                                  | 1099/1948 [27:10<26:08,  1.85s/it]cv 값 : 0.3815256629871634람다 값 : 0.9796410114220975



![png](output_52_3299.png)


     56%|████████████████████████████████████████████                                  | 1100/1948 [27:11<24:57,  1.77s/it]cv 값 : 0.504441704564402람다 값 : 0.5066747641063302



![png](output_52_3302.png)


     57%|████████████████████████████████████████████                                  | 1101/1948 [27:13<24:58,  1.77s/it]cv 값 : 0.33221876955540214람다 값 : 0.7369389916752472



![png](output_52_3305.png)


     57%|████████████████████████████████████████████▏                                 | 1102/1948 [27:15<25:21,  1.80s/it]cv 값 : 0.20971748018520742람다 값 : 0.7124228795004732



![png](output_52_3308.png)


     57%|████████████████████████████████████████████▏                                 | 1103/1948 [27:17<25:50,  1.83s/it]cv 값 : 0.5318742523804796람다 값 : 0.12322476267604675



![png](output_52_3311.png)


     57%|████████████████████████████████████████████▏                                 | 1104/1948 [27:18<25:15,  1.80s/it]cv 값 : 0.6145237979444728람다 값 : 0.5302006249338193



![png](output_52_3314.png)


     57%|████████████████████████████████████████████▏                                 | 1105/1948 [27:20<25:17,  1.80s/it]cv 값 : 0.3819171622775225람다 값 : 0.444275706117879



![png](output_52_3317.png)


     57%|████████████████████████████████████████████▎                                 | 1106/1948 [27:22<24:28,  1.74s/it]cv 값 : 0.44732367155007613람다 값 : 0.45779850066925615



![png](output_52_3320.png)


     57%|████████████████████████████████████████████▎                                 | 1107/1948 [27:24<24:41,  1.76s/it]cv 값 : 0.13020249969687345람다 값 : 1.17408548516206



![png](output_52_3323.png)


     57%|████████████████████████████████████████████▎                                 | 1108/1948 [27:26<25:49,  1.84s/it]cv 값 : 0.8984089952644481람다 값 : -0.42752276858109195



![png](output_52_3326.png)


     57%|████████████████████████████████████████████▍                                 | 1109/1948 [27:28<26:16,  1.88s/it]cv 값 : 0.11819501566319159람다 값 : 1.984888026361389



![png](output_52_3329.png)


     57%|████████████████████████████████████████████▍                                 | 1110/1948 [27:30<26:16,  1.88s/it]cv 값 : 0.8435548097374205람다 값 : 0.2762002309570688



![png](output_52_3332.png)


     57%|████████████████████████████████████████████▍                                 | 1111/1948 [27:31<26:10,  1.88s/it]cv 값 : 0.2292719866487813람다 값 : 1.3738622806479237



![png](output_52_3335.png)


     57%|████████████████████████████████████████████▌                                 | 1112/1948 [27:33<26:31,  1.90s/it]cv 값 : 0.5556603000975197람다 값 : 0.3044924927526977



![png](output_52_3338.png)


     57%|████████████████████████████████████████████▌                                 | 1113/1948 [27:35<26:04,  1.87s/it]cv 값 : 0.5982217917563655람다 값 : -0.5111566308082943



![png](output_52_3341.png)


     57%|████████████████████████████████████████████▌                                 | 1114/1948 [27:37<25:42,  1.85s/it]cv 값 : 0.5683660951424517람다 값 : 0.015578798235432487



![png](output_52_3344.png)


     57%|████████████████████████████████████████████▋                                 | 1115/1948 [27:39<24:56,  1.80s/it]cv 값 : 0.16242147816199762람다 값 : 1.9670170253814607



![png](output_52_3347.png)


     57%|████████████████████████████████████████████▋                                 | 1116/1948 [27:41<26:14,  1.89s/it]cv 값 : 0.5109932657499804람다 값 : 0.2770168261244499



![png](output_52_3350.png)


     57%|████████████████████████████████████████████▋                                 | 1117/1948 [27:42<25:07,  1.81s/it]cv 값 : 0.3410469955251602람다 값 : 0.8287516858882318



![png](output_52_3353.png)


     57%|████████████████████████████████████████████▊                                 | 1118/1948 [27:44<24:46,  1.79s/it]cv 값 : 0.41971243879501735람다 값 : -0.19912921691499552



![png](output_52_3356.png)


     57%|████████████████████████████████████████████▊                                 | 1119/1948 [27:47<27:24,  1.98s/it]cv 값 : 0.7879847398337768람다 값 : 0.33715832173148946



![png](output_52_3359.png)


     57%|████████████████████████████████████████████▊                                 | 1120/1948 [27:49<28:31,  2.07s/it]cv 값 : 0.4718024731044489람다 값 : 0.6371897175454222



![png](output_52_3362.png)


     58%|████████████████████████████████████████████▉                                 | 1121/1948 [27:52<31:23,  2.28s/it]cv 값 : 0.22727058989230062람다 값 : -0.4625855440185745



![png](output_52_3365.png)


     58%|████████████████████████████████████████████▉                                 | 1122/1948 [27:55<33:55,  2.46s/it]cv 값 : 0.1489760456316523람다 값 : 0.9051945685584208



![png](output_52_3368.png)


     58%|████████████████████████████████████████████▉                                 | 1123/1948 [27:57<31:53,  2.32s/it]cv 값 : 0.25608608097684854람다 값 : 1.661206968657412



![png](output_52_3371.png)


     58%|█████████████████████████████████████████████                                 | 1124/1948 [27:58<30:28,  2.22s/it]cv 값 : 0.5174387535105253람다 값 : -0.13385903963098805



![png](output_52_3374.png)


     58%|█████████████████████████████████████████████                                 | 1125/1948 [28:02<36:43,  2.68s/it]cv 값 : 0.2018421005471104람다 값 : -0.10341961409880396



![png](output_52_3377.png)


     58%|█████████████████████████████████████████████                                 | 1126/1948 [28:06<39:10,  2.86s/it]cv 값 : 0.2172940715475709람다 값 : 0.014620298378653614



![png](output_52_3380.png)


     58%|█████████████████████████████████████████████▏                                | 1127/1948 [28:07<35:11,  2.57s/it]cv 값 : 0.26243074846584047람다 값 : 0.9287810516817567



![png](output_52_3383.png)


     58%|█████████████████████████████████████████████▏                                | 1128/1948 [28:10<35:07,  2.57s/it]cv 값 : 0.38347114338869553람다 값 : 0.16401290887341688



![png](output_52_3386.png)


     58%|█████████████████████████████████████████████▏                                | 1129/1948 [28:14<39:12,  2.87s/it]cv 값 : 0.46596340329853114람다 값 : 0.32636868208506276



![png](output_52_3389.png)


     58%|█████████████████████████████████████████████▏                                | 1130/1948 [28:17<40:11,  2.95s/it]cv 값 : 0.3597279360213751람다 값 : 0.8277452161712882



![png](output_52_3392.png)


     58%|█████████████████████████████████████████████▎                                | 1131/1948 [28:18<35:22,  2.60s/it]cv 값 : 0.2632265151657433람다 값 : 0.6480988308839178



![png](output_52_3395.png)


     58%|█████████████████████████████████████████████▎                                | 1132/1948 [28:20<31:39,  2.33s/it]cv 값 : 0.29207004255447466람다 값 : 0.3326013201665177



![png](output_52_3398.png)


     58%|█████████████████████████████████████████████▎                                | 1133/1948 [28:25<43:16,  3.19s/it]cv 값 : 0.4029881801497815람다 값 : 0.21002342863844076



![png](output_52_3401.png)


     58%|█████████████████████████████████████████████▍                                | 1134/1948 [28:28<42:45,  3.15s/it]cv 값 : 0.7955320970790264람다 값 : 0.23999264506483894



![png](output_52_3404.png)


     58%|█████████████████████████████████████████████▍                                | 1135/1948 [28:30<38:15,  2.82s/it]cv 값 : 0.3312199037527366람다 값 : -0.19594797744389778



![png](output_52_3407.png)


     58%|█████████████████████████████████████████████▍                                | 1136/1948 [28:32<34:23,  2.54s/it]cv 값 : 0.5163480295900946람다 값 : -0.18014635249514788



![png](output_52_3410.png)


     58%|█████████████████████████████████████████████▌                                | 1137/1948 [28:34<31:51,  2.36s/it]cv 값 : 0.5465253544615061람다 값 : -0.638789158368812



![png](output_52_3413.png)


     58%|█████████████████████████████████████████████▌                                | 1138/1948 [28:37<34:39,  2.57s/it]cv 값 : 0.29818139134753835람다 값 : 1.0221198814999441



![png](output_52_3416.png)


     58%|█████████████████████████████████████████████▌                                | 1139/1948 [28:41<38:44,  2.87s/it]cv 값 : 0.2979198954302721람다 값 : 1.1526276827323416



![png](output_52_3419.png)


     59%|█████████████████████████████████████████████▋                                | 1140/1948 [28:44<37:59,  2.82s/it]cv 값 : 0.2987191490305736람다 값 : 0.9253091019411157



![png](output_52_3422.png)


     59%|█████████████████████████████████████████████▋                                | 1141/1948 [28:46<37:54,  2.82s/it]cv 값 : 0.30812942517170355람다 값 : -0.2947677445599309



![png](output_52_3425.png)


     59%|█████████████████████████████████████████████▋                                | 1142/1948 [28:51<44:02,  3.28s/it]cv 값 : 0.8675861334530471람다 값 : -0.6052349239377982



![png](output_52_3428.png)


     59%|█████████████████████████████████████████████▊                                | 1143/1948 [28:54<44:24,  3.31s/it]cv 값 : 0.35803051500355537람다 값 : 1.3320529510852106



![png](output_52_3431.png)


     59%|█████████████████████████████████████████████▊                                | 1144/1948 [28:57<41:18,  3.08s/it]cv 값 : 0.6061149721348784람다 값 : -0.8228596056530011



![png](output_52_3434.png)


     59%|█████████████████████████████████████████████▊                                | 1145/1948 [28:59<36:55,  2.76s/it]cv 값 : 0.1997123801391814람다 값 : 1.9049229126333476



![png](output_52_3437.png)


     59%|█████████████████████████████████████████████▉                                | 1146/1948 [29:01<36:10,  2.71s/it]cv 값 : 1.1416451190165635람다 값 : 0.02931653340191783



![png](output_52_3440.png)


     59%|█████████████████████████████████████████████▉                                | 1147/1948 [29:04<34:14,  2.57s/it]cv 값 : 0.38197046311473964람다 값 : 0.4738522948604607



![png](output_52_3443.png)


     59%|█████████████████████████████████████████████▉                                | 1148/1948 [29:07<38:31,  2.89s/it]cv 값 : 1.1374351340454452람다 값 : 0.3244706937212878



![png](output_52_3446.png)


     59%|██████████████████████████████████████████████                                | 1149/1948 [29:09<35:59,  2.70s/it]cv 값 : 1.0628120246780866람다 값 : 0.16757853004816398



![png](output_52_3449.png)


     59%|██████████████████████████████████████████████                                | 1150/1948 [29:11<32:16,  2.43s/it]cv 값 : 0.6044164437544625람다 값 : -0.147623575123185



![png](output_52_3452.png)


     59%|██████████████████████████████████████████████                                | 1151/1948 [29:13<29:56,  2.25s/it]cv 값 : 0.38703136752955924람다 값 : 0.45370043812650185



![png](output_52_3455.png)


     59%|██████████████████████████████████████████████▏                               | 1152/1948 [29:15<28:20,  2.14s/it]cv 값 : 0.30812592455214804람다 값 : 0.5210337033652163



![png](output_52_3458.png)


     59%|██████████████████████████████████████████████▏                               | 1153/1948 [29:17<26:47,  2.02s/it]cv 값 : 0.2832240395959137람다 값 : 0.3486572881478112



![png](output_52_3461.png)


     59%|██████████████████████████████████████████████▏                               | 1154/1948 [29:19<27:32,  2.08s/it]cv 값 : 0.5243019106614051람다 값 : 0.3487783194326536



![png](output_52_3464.png)


     59%|██████████████████████████████████████████████▏                               | 1155/1948 [29:21<27:18,  2.07s/it]cv 값 : 0.3750123066571402람다 값 : 0.36154961043643596



![png](output_52_3467.png)


     59%|██████████████████████████████████████████████▎                               | 1156/1948 [29:23<25:57,  1.97s/it]cv 값 : 0.494312810379836람다 값 : 0.6086206011944604



![png](output_52_3470.png)


     59%|██████████████████████████████████████████████▎                               | 1157/1948 [29:25<26:03,  1.98s/it]cv 값 : 0.5257123345393844람다 값 : 0.45146313056459414



![png](output_52_3473.png)


     59%|██████████████████████████████████████████████▎                               | 1158/1948 [29:27<25:38,  1.95s/it]cv 값 : 0.5477217936538358람다 값 : 0.343658948891033



![png](output_52_3476.png)


     59%|██████████████████████████████████████████████▍                               | 1159/1948 [29:29<26:07,  1.99s/it]cv 값 : 1.1238134674928람다 값 : 0.2964988012865157



![png](output_52_3479.png)


     60%|██████████████████████████████████████████████▍                               | 1160/1948 [29:31<26:13,  2.00s/it]cv 값 : 0.20389088513127135람다 값 : 1.3253804178311206



![png](output_52_3482.png)


     60%|██████████████████████████████████████████████▍                               | 1161/1948 [29:32<25:21,  1.93s/it]cv 값 : 0.3224255049600667람다 값 : 1.0977957668322675



![png](output_52_3485.png)


     60%|██████████████████████████████████████████████▌                               | 1162/1948 [29:34<24:59,  1.91s/it]cv 값 : 0.4427997992689018람다 값 : 0.6721381916393918



![png](output_52_3488.png)


     60%|██████████████████████████████████████████████▌                               | 1163/1948 [29:36<24:00,  1.84s/it]cv 값 : 0.2977120808745037람다 값 : 0.6402354312684823



![png](output_52_3491.png)


     60%|██████████████████████████████████████████████▌                               | 1164/1948 [29:38<24:30,  1.88s/it]cv 값 : 0.4570520058984372람다 값 : 0.6453939323162892



![png](output_52_3494.png)


     60%|██████████████████████████████████████████████▋                               | 1165/1948 [29:40<23:52,  1.83s/it]cv 값 : 0.5306393202194755람다 값 : 0.6872010104432659



![png](output_52_3497.png)


     60%|██████████████████████████████████████████████▋                               | 1166/1948 [29:41<23:35,  1.81s/it]cv 값 : 0.8916600506573858람다 값 : 0.29751314144899965



![png](output_52_3500.png)


     60%|██████████████████████████████████████████████▋                               | 1167/1948 [29:43<22:57,  1.76s/it]cv 값 : 0.23757646294155274람다 값 : 2.0560510129306726



![png](output_52_3503.png)


     60%|██████████████████████████████████████████████▊                               | 1168/1948 [29:45<23:38,  1.82s/it]cv 값 : 0.42967408056550904람다 값 : 1.4875048052635402



![png](output_52_3506.png)


     60%|██████████████████████████████████████████████▊                               | 1169/1948 [29:47<23:41,  1.82s/it]cv 값 : 0.48927871199440254람다 값 : 0.6368513089250569



![png](output_52_3509.png)


     60%|██████████████████████████████████████████████▊                               | 1170/1948 [29:49<22:52,  1.76s/it]cv 값 : 0.32635446187482864람다 값 : -0.41951635312038926



![png](output_52_3512.png)


     60%|██████████████████████████████████████████████▉                               | 1171/1948 [29:51<23:57,  1.85s/it]cv 값 : 0.5202564876995518람다 값 : 0.014670214847578862



![png](output_52_3515.png)


     60%|██████████████████████████████████████████████▉                               | 1172/1948 [29:52<23:07,  1.79s/it]cv 값 : 0.4027215161970386람다 값 : 0.7418992378021114



![png](output_52_3518.png)


     60%|██████████████████████████████████████████████▉                               | 1173/1948 [29:54<23:15,  1.80s/it]cv 값 : 0.8276000750531369람다 값 : -0.20005012134530478



![png](output_52_3521.png)


     60%|███████████████████████████████████████████████                               | 1174/1948 [29:56<22:49,  1.77s/it]cv 값 : 0.6577249048193223람다 값 : 0.1653242020479509



![png](output_52_3524.png)


     60%|███████████████████████████████████████████████                               | 1175/1948 [29:57<22:23,  1.74s/it]cv 값 : 0.3532096463478022람다 값 : 0.9629779468543593



![png](output_52_3527.png)


     60%|███████████████████████████████████████████████                               | 1176/1948 [29:59<21:26,  1.67s/it]cv 값 : 1.087715263317818람다 값 : 0.06306752116827141



![png](output_52_3530.png)


     60%|███████████████████████████████████████████████▏                              | 1177/1948 [30:01<21:36,  1.68s/it]cv 값 : 0.44813439704134594람다 값 : -0.03052848440156949



![png](output_52_3533.png)


     60%|███████████████████████████████████████████████▏                              | 1178/1948 [30:02<22:01,  1.72s/it]cv 값 : 0.4833053908841044람다 값 : 0.424767433712855



![png](output_52_3536.png)


     61%|███████████████████████████████████████████████▏                              | 1179/1948 [30:04<21:11,  1.65s/it]cv 값 : 0.9547765393077782람다 값 : 0.23885105728853417



![png](output_52_3539.png)


     61%|███████████████████████████████████████████████▏                              | 1180/1948 [30:06<21:40,  1.69s/it]cv 값 : 0.23629472099924428람다 값 : -1.0121586987596967



![png](output_52_3542.png)


     61%|███████████████████████████████████████████████▎                              | 1181/1948 [30:07<21:27,  1.68s/it]cv 값 : 0.4330052103874321람다 값 : 0.4050738783180707



![png](output_52_3545.png)


     61%|███████████████████████████████████████████████▎                              | 1182/1948 [30:09<21:27,  1.68s/it]cv 값 : 0.6780503365524577람다 값 : 0.0021193984479690045



![png](output_52_3548.png)


     61%|███████████████████████████████████████████████▎                              | 1183/1948 [30:11<21:01,  1.65s/it]cv 값 : 0.6535632160524025람다 값 : -0.18762458999238618



![png](output_52_3551.png)


     61%|███████████████████████████████████████████████▍                              | 1184/1948 [30:12<21:30,  1.69s/it]cv 값 : 0.39508172349994186람다 값 : -0.25184510948798844



![png](output_52_3554.png)


     61%|███████████████████████████████████████████████▍                              | 1185/1948 [30:14<21:40,  1.70s/it]cv 값 : 0.4058099338847003람다 값 : 0.7303539275830545



![png](output_52_3557.png)


     61%|███████████████████████████████████████████████▍                              | 1186/1948 [30:16<21:17,  1.68s/it]cv 값 : 0.3755871073704357람다 값 : 0.43279578601315144



![png](output_52_3560.png)


     61%|███████████████████████████████████████████████▌                              | 1187/1948 [30:18<22:14,  1.75s/it]cv 값 : 0.3654168805493519람다 값 : 0.0025707293193320086



![png](output_52_3563.png)


     61%|███████████████████████████████████████████████▌                              | 1188/1948 [30:19<22:18,  1.76s/it]cv 값 : 0.49972825763509804람다 값 : 0.5402016952906803



![png](output_52_3566.png)


     61%|███████████████████████████████████████████████▌                              | 1189/1948 [30:21<22:35,  1.79s/it]cv 값 : 1.276100127276327람다 값 : -0.3670891429230131



![png](output_52_3569.png)


     61%|███████████████████████████████████████████████▋                              | 1190/1948 [30:23<21:54,  1.73s/it]cv 값 : 0.245688858608585람다 값 : 1.1033015535896924



![png](output_52_3572.png)


     61%|███████████████████████████████████████████████▋                              | 1191/1948 [30:25<23:08,  1.83s/it]cv 값 : 1.107033527520754람다 값 : 0.19462497197337272



![png](output_52_3575.png)


     61%|███████████████████████████████████████████████▋                              | 1192/1948 [30:27<23:10,  1.84s/it]cv 값 : 0.29488546051576153람다 값 : 0.8939773350220066



![png](output_52_3578.png)


     61%|███████████████████████████████████████████████▊                              | 1193/1948 [30:29<22:47,  1.81s/it]cv 값 : 0.6807078680023732람다 값 : -0.05614533581137813



![png](output_52_3581.png)


     61%|███████████████████████████████████████████████▊                              | 1194/1948 [30:30<23:09,  1.84s/it]cv 값 : 0.2734793797605356람다 값 : 0.24808705645843276



![png](output_52_3584.png)


     61%|███████████████████████████████████████████████▊                              | 1195/1948 [30:32<23:40,  1.89s/it]cv 값 : 0.2867075217036057람다 값 : 0.007901572132843522



![png](output_52_3587.png)


     61%|███████████████████████████████████████████████▉                              | 1196/1948 [30:34<23:23,  1.87s/it]cv 값 : 0.7593257860264152람다 값 : 0.3514876657650338



![png](output_52_3590.png)


     61%|███████████████████████████████████████████████▉                              | 1197/1948 [30:36<22:58,  1.84s/it]cv 값 : 0.3712763967284287람다 값 : -0.27885412256022635



![png](output_52_3593.png)


     61%|███████████████████████████████████████████████▉                              | 1198/1948 [30:38<23:36,  1.89s/it]cv 값 : 0.37813069631695295람다 값 : 1.1988676238489033



![png](output_52_3596.png)


     62%|████████████████████████████████████████████████                              | 1199/1948 [30:40<23:36,  1.89s/it]cv 값 : 0.9323412494258718람다 값 : -0.4067829719531897



![png](output_52_3599.png)


     62%|████████████████████████████████████████████████                              | 1200/1948 [30:42<23:56,  1.92s/it]cv 값 : 0.8967773728638233람다 값 : 0.36595836374907115



![png](output_52_3602.png)


     62%|████████████████████████████████████████████████                              | 1201/1948 [30:44<23:09,  1.86s/it]cv 값 : 0.2726547609525011람다 값 : 1.2333078764499295



![png](output_52_3605.png)


     62%|████████████████████████████████████████████████▏                             | 1202/1948 [30:46<23:05,  1.86s/it]cv 값 : 0.37474313247186797람다 값 : 0.39270947371857823



![png](output_52_3608.png)


     62%|████████████████████████████████████████████████▏                             | 1203/1948 [30:47<21:39,  1.74s/it]cv 값 : 0.6126117377007617람다 값 : 0.5513915034299107



![png](output_52_3611.png)


     62%|████████████████████████████████████████████████▏                             | 1204/1948 [30:49<21:51,  1.76s/it]cv 값 : 0.29127392372085753람다 값 : 1.1129862614094996



![png](output_52_3614.png)


     62%|████████████████████████████████████████████████▏                             | 1205/1948 [30:51<21:39,  1.75s/it]cv 값 : 0.414815283425351람다 값 : 0.8279895172176036



![png](output_52_3617.png)


     62%|████████████████████████████████████████████████▎                             | 1206/1948 [30:52<21:34,  1.74s/it]cv 값 : 0.2333042626088181람다 값 : 0.35165274632205723



![png](output_52_3620.png)


     62%|████████████████████████████████████████████████▎                             | 1207/1948 [30:54<22:58,  1.86s/it]cv 값 : 0.3987457888706744람다 값 : 0.152072719391388



![png](output_52_3623.png)


     62%|████████████████████████████████████████████████▎                             | 1208/1948 [30:56<22:39,  1.84s/it]cv 값 : 2.0173943933146163람다 값 : -0.13080314476561866



![png](output_52_3626.png)


     62%|████████████████████████████████████████████████▍                             | 1209/1948 [30:58<22:43,  1.85s/it]cv 값 : 0.6693176872202259람다 값 : 0.49762864936810136



![png](output_52_3629.png)


     62%|████████████████████████████████████████████████▍                             | 1210/1948 [31:00<22:45,  1.85s/it]cv 값 : 1.282886598388899람다 값 : 0.2336316831251164



![png](output_52_3632.png)


     62%|████████████████████████████████████████████████▍                             | 1211/1948 [31:02<22:26,  1.83s/it]cv 값 : 0.31070852240232416람다 값 : 0.03182039808245449



![png](output_52_3635.png)


     62%|████████████████████████████████████████████████▌                             | 1212/1948 [31:03<22:04,  1.80s/it]cv 값 : 0.2861095270627557람다 값 : 0.1420010247673153



![png](output_52_3638.png)


     62%|████████████████████████████████████████████████▌                             | 1213/1948 [31:05<22:41,  1.85s/it]cv 값 : 0.223870800330969람다 값 : 1.1628456509516667



![png](output_52_3641.png)


     62%|████████████████████████████████████████████████▌                             | 1214/1948 [31:07<22:30,  1.84s/it]cv 값 : 0.1822672398977645람다 값 : 0.06383695840829585



![png](output_52_3644.png)


     62%|████████████████████████████████████████████████▋                             | 1215/1948 [31:09<21:43,  1.78s/it]cv 값 : 0.31721420057660793람다 값 : 0.5409300573151087



![png](output_52_3647.png)


     62%|████████████████████████████████████████████████▋                             | 1216/1948 [31:11<22:09,  1.82s/it]cv 값 : 1.1134890361436787람다 값 : -0.15125231018279517



![png](output_52_3650.png)


     62%|████████████████████████████████████████████████▋                             | 1217/1948 [31:13<23:11,  1.90s/it]cv 값 : 0.7168028260898268람다 값 : -0.19400646561864687



![png](output_52_3653.png)


     63%|████████████████████████████████████████████████▊                             | 1218/1948 [31:15<22:32,  1.85s/it]cv 값 : 0.24215848642223275람다 값 : -0.04214862493677849



![png](output_52_3656.png)


     63%|████████████████████████████████████████████████▊                             | 1219/1948 [31:16<22:33,  1.86s/it]cv 값 : 0.2369151595137265람다 값 : 0.09926578952369465



![png](output_52_3659.png)


     63%|████████████████████████████████████████████████▊                             | 1220/1948 [31:18<21:31,  1.77s/it]cv 값 : 0.5646329177603872람다 값 : 0.2517272546165063



![png](output_52_3662.png)


     63%|████████████████████████████████████████████████▉                             | 1221/1948 [31:20<21:39,  1.79s/it]cv 값 : 0.48954670929461813람다 값 : 0.29066871822354734



![png](output_52_3665.png)


     63%|████████████████████████████████████████████████▉                             | 1222/1948 [31:22<21:07,  1.75s/it]cv 값 : 0.6314335455379998람다 값 : 0.5190916766512929



![png](output_52_3668.png)


     63%|████████████████████████████████████████████████▉                             | 1223/1948 [31:23<20:25,  1.69s/it]cv 값 : 0.617148679691999람다 값 : -0.10427470743771423



![png](output_52_3671.png)


     63%|█████████████████████████████████████████████████                             | 1224/1948 [31:25<20:11,  1.67s/it]cv 값 : 0.6430880246393101람다 값 : 0.025936332782531882



![png](output_52_3674.png)


     63%|█████████████████████████████████████████████████                             | 1225/1948 [31:26<19:32,  1.62s/it]cv 값 : 0.5089726111816271람다 값 : 0.8408997496131039



![png](output_52_3677.png)


     63%|█████████████████████████████████████████████████                             | 1226/1948 [31:28<19:01,  1.58s/it]cv 값 : 1.0235234938718694람다 값 : 0.3184389877929391



![png](output_52_3680.png)


     63%|█████████████████████████████████████████████████▏                            | 1227/1948 [31:30<20:35,  1.71s/it]cv 값 : 0.838560259337465람다 값 : 0.3928079478811763



![png](output_52_3683.png)


     63%|█████████████████████████████████████████████████▏                            | 1228/1948 [31:31<19:44,  1.65s/it]cv 값 : 0.5940728290830863람다 값 : 0.5538084167503554



![png](output_52_3686.png)


     63%|█████████████████████████████████████████████████▏                            | 1229/1948 [31:33<19:34,  1.63s/it]cv 값 : 0.24474924113039817람다 값 : 1.2509340812757404



![png](output_52_3689.png)


     63%|█████████████████████████████████████████████████▎                            | 1230/1948 [31:35<21:29,  1.80s/it]cv 값 : 0.31562039683315835람다 값 : 0.2230305019750308



![png](output_52_3692.png)


     63%|█████████████████████████████████████████████████▎                            | 1231/1948 [31:37<21:23,  1.79s/it]cv 값 : 1.5463195965297378람다 값 : 0.0973255104272326



![png](output_52_3695.png)


     63%|█████████████████████████████████████████████████▎                            | 1232/1948 [31:38<20:32,  1.72s/it]cv 값 : 1.2838640148192229람다 값 : 0.07535887609288955



![png](output_52_3698.png)


     63%|█████████████████████████████████████████████████▎                            | 1233/1948 [31:40<20:49,  1.75s/it]cv 값 : 0.45113926565464246람다 값 : 0.4672700443744014



![png](output_52_3701.png)


     63%|█████████████████████████████████████████████████▍                            | 1234/1948 [31:42<20:37,  1.73s/it]cv 값 : 0.39903091462771084람다 값 : 0.15648492801716374



![png](output_52_3704.png)


     63%|█████████████████████████████████████████████████▍                            | 1235/1948 [31:43<19:30,  1.64s/it]cv 값 : 0.6043408474919819람다 값 : 0.07600795854221376



![png](output_52_3707.png)


     63%|█████████████████████████████████████████████████▍                            | 1236/1948 [31:45<20:00,  1.69s/it]cv 값 : 0.5449005040607707람다 값 : 0.6350596473906797



![png](output_52_3710.png)


     64%|█████████████████████████████████████████████████▌                            | 1237/1948 [31:47<20:26,  1.73s/it]cv 값 : 0.405356860750241람다 값 : 0.3433387092808879



![png](output_52_3713.png)


     64%|█████████████████████████████████████████████████▌                            | 1238/1948 [31:49<21:53,  1.85s/it]cv 값 : 0.4733866213437766람다 값 : 0.08649514718750505



![png](output_52_3716.png)


     64%|█████████████████████████████████████████████████▌                            | 1239/1948 [31:50<20:24,  1.73s/it]cv 값 : 0.9571180385576673람다 값 : 0.14725619298083975



![png](output_52_3719.png)


     64%|█████████████████████████████████████████████████▋                            | 1240/1948 [31:52<21:03,  1.78s/it]cv 값 : 1.0167260773224114람다 값 : 0.09738043025056993



![png](output_52_3722.png)


     64%|█████████████████████████████████████████████████▋                            | 1241/1948 [31:54<21:30,  1.83s/it]cv 값 : 0.8513590898728804람다 값 : 0.36889125382045773



![png](output_52_3725.png)


     64%|█████████████████████████████████████████████████▋                            | 1242/1948 [31:57<22:58,  1.95s/it]cv 값 : 0.30460129757568843람다 값 : -0.15786575791033713



![png](output_52_3728.png)


     64%|█████████████████████████████████████████████████▊                            | 1243/1948 [31:59<23:51,  2.03s/it]cv 값 : 0.3368477647265278람다 값 : 0.9917388432303886



![png](output_52_3731.png)


     64%|█████████████████████████████████████████████████▊                            | 1244/1948 [32:01<26:13,  2.24s/it]cv 값 : 0.2979066728378908람다 값 : -0.19196298458300984



![png](output_52_3734.png)


     64%|█████████████████████████████████████████████████▊                            | 1245/1948 [32:04<25:46,  2.20s/it]cv 값 : 0.7957108659957937람다 값 : 0.2970605767478373



![png](output_52_3737.png)


     64%|█████████████████████████████████████████████████▉                            | 1246/1948 [32:08<32:56,  2.82s/it]cv 값 : 0.7265507101730811람다 값 : 0.2381886238498033



![png](output_52_3740.png)


     64%|█████████████████████████████████████████████████▉                            | 1247/1948 [32:11<33:31,  2.87s/it]cv 값 : 1.1492570065754308람다 값 : -0.10195249002118134



![png](output_52_3743.png)


     64%|█████████████████████████████████████████████████▉                            | 1248/1948 [32:13<30:59,  2.66s/it]cv 값 : 1.019903967921513람다 값 : 0.09871375589235665



![png](output_52_3746.png)


     64%|██████████████████████████████████████████████████                            | 1249/1948 [32:15<29:42,  2.55s/it]cv 값 : 0.19562155457045022람다 값 : -2.148831477395526



![png](output_52_3749.png)


     64%|██████████████████████████████████████████████████                            | 1250/1948 [32:17<27:39,  2.38s/it]cv 값 : 0.33393931600440174람다 값 : 0.4042274675881152



![png](output_52_3752.png)


     64%|██████████████████████████████████████████████████                            | 1251/1948 [32:21<31:04,  2.67s/it]cv 값 : 0.31730862256161174람다 값 : 0.049861088845151266



![png](output_52_3755.png)


     64%|██████████████████████████████████████████████████▏                           | 1252/1948 [32:23<29:00,  2.50s/it]cv 값 : 0.9047281943524904람다 값 : -0.7353661072799373



![png](output_52_3758.png)


     64%|██████████████████████████████████████████████████▏                           | 1253/1948 [32:25<27:49,  2.40s/it]cv 값 : 0.14212823860169282람다 값 : 0.6648060861488507



![png](output_52_3761.png)


     64%|██████████████████████████████████████████████████▏                           | 1254/1948 [32:27<26:08,  2.26s/it]cv 값 : 0.26308551468268754람다 값 : -0.6868189040273017



![png](output_52_3764.png)


     64%|██████████████████████████████████████████████████▎                           | 1255/1948 [32:30<28:19,  2.45s/it]cv 값 : 0.31151846525370896람다 값 : 0.44999658338780874



![png](output_52_3767.png)


     64%|██████████████████████████████████████████████████▎                           | 1256/1948 [32:31<25:41,  2.23s/it]cv 값 : 0.518433463623659람다 값 : 0.09756717589903781



![png](output_52_3770.png)


     65%|██████████████████████████████████████████████████▎                           | 1257/1948 [32:33<23:33,  2.05s/it]cv 값 : 0.44981668728307483람다 값 : 0.13201892335223442



![png](output_52_3773.png)


     65%|██████████████████████████████████████████████████▎                           | 1258/1948 [32:35<23:24,  2.04s/it]cv 값 : 0.5471339201677617람다 값 : 0.634424049809524



![png](output_52_3776.png)


     65%|██████████████████████████████████████████████████▍                           | 1259/1948 [32:37<22:05,  1.92s/it]cv 값 : 0.18107584028580095람다 값 : 2.266301594325647



![png](output_52_3779.png)


     65%|██████████████████████████████████████████████████▍                           | 1260/1948 [32:38<20:50,  1.82s/it]cv 값 : 0.44696897041882006람다 값 : 0.6187983753875365



![png](output_52_3782.png)


     65%|██████████████████████████████████████████████████▍                           | 1261/1948 [32:40<20:47,  1.82s/it]cv 값 : 0.36039049112648763람다 값 : 0.08679586669874217



![png](output_52_3785.png)


     65%|██████████████████████████████████████████████████▌                           | 1262/1948 [32:42<21:35,  1.89s/it]cv 값 : 0.2872627315733912람다 값 : 0.5252691959342942



![png](output_52_3788.png)


     65%|██████████████████████████████████████████████████▌                           | 1263/1948 [32:45<24:50,  2.18s/it]cv 값 : 0.35624399441805527람다 값 : 0.41840236439324147



![png](output_52_3791.png)


     65%|██████████████████████████████████████████████████▌                           | 1264/1948 [32:47<24:35,  2.16s/it]cv 값 : 0.28182664393752777람다 값 : -1.016400084069674



![png](output_52_3794.png)


     65%|██████████████████████████████████████████████████▋                           | 1265/1948 [32:49<23:50,  2.09s/it]cv 값 : 0.5455754317921104람다 값 : -0.6587856896861443



![png](output_52_3797.png)


     65%|██████████████████████████████████████████████████▋                           | 1266/1948 [32:51<23:40,  2.08s/it]cv 값 : 0.7410742313294352람다 값 : 0.42513318755240953



![png](output_52_3800.png)


     65%|██████████████████████████████████████████████████▋                           | 1267/1948 [32:53<23:59,  2.11s/it]cv 값 : 0.8646133538804238람다 값 : 0.27237383763677586



![png](output_52_3803.png)


     65%|██████████████████████████████████████████████████▊                           | 1268/1948 [32:55<23:39,  2.09s/it]cv 값 : 0.37741469298919134람다 값 : 1.2993058799192958



![png](output_52_3806.png)


     65%|██████████████████████████████████████████████████▊                           | 1269/1948 [32:57<22:35,  2.00s/it]cv 값 : 0.7333079865205752람다 값 : 0.4509808514412889



![png](output_52_3809.png)


     65%|██████████████████████████████████████████████████▊                           | 1270/1948 [32:59<22:10,  1.96s/it]cv 값 : 0.09428839841630268람다 값 : -1.8665583454591739



![png](output_52_3812.png)


     65%|██████████████████████████████████████████████████▉                           | 1271/1948 [33:03<28:46,  2.55s/it]cv 값 : 0.6815800793943958람다 값 : 0.2539773478978428



![png](output_52_3815.png)


     65%|██████████████████████████████████████████████████▉                           | 1272/1948 [33:06<29:44,  2.64s/it]cv 값 : 0.7183142252836039람다 값 : 0.03060920142265496



![png](output_52_3818.png)


     65%|██████████████████████████████████████████████████▉                           | 1273/1948 [33:08<27:22,  2.43s/it]cv 값 : 0.5994963117073135람다 값 : 0.5580364393076886



![png](output_52_3821.png)


     65%|███████████████████████████████████████████████████                           | 1274/1948 [33:09<24:50,  2.21s/it]cv 값 : 0.46427885837969607람다 값 : 0.7593273138905569



![png](output_52_3824.png)


     65%|███████████████████████████████████████████████████                           | 1275/1948 [33:11<23:15,  2.07s/it]cv 값 : 0.24068893395839935람다 값 : 0.23966313976918835



![png](output_52_3827.png)


     66%|███████████████████████████████████████████████████                           | 1276/1948 [33:14<26:01,  2.32s/it]cv 값 : 0.57960498581782람다 값 : 0.5326074280393095



![png](output_52_3830.png)


     66%|███████████████████████████████████████████████████▏                          | 1277/1948 [33:17<27:55,  2.50s/it]cv 값 : 0.40931901168192386람다 값 : 0.8096851194858096



![png](output_52_3833.png)


     66%|███████████████████████████████████████████████████▏                          | 1278/1948 [33:19<24:43,  2.21s/it]cv 값 : 0.546476299162611람다 값 : 0.27689994976627136



![png](output_52_3836.png)


     66%|███████████████████████████████████████████████████▏                          | 1279/1948 [33:21<25:19,  2.27s/it]cv 값 : 0.7099046342423622람다 값 : 0.4576610124008345



![png](output_52_3839.png)


     66%|███████████████████████████████████████████████████▎                          | 1280/1948 [33:24<27:45,  2.49s/it]cv 값 : 0.3465428005195406람다 값 : 1.0277016986052054



![png](output_52_3842.png)


     66%|███████████████████████████████████████████████████▎                          | 1281/1948 [33:27<30:31,  2.75s/it]cv 값 : 1.096628994198948람다 값 : 0.1153667527711016



![png](output_52_3845.png)


     66%|███████████████████████████████████████████████████▎                          | 1282/1948 [33:29<27:28,  2.48s/it]cv 값 : 1.1790321784952145람다 값 : 0.12174438685025434



![png](output_52_3848.png)


     66%|███████████████████████████████████████████████████▎                          | 1283/1948 [33:31<24:11,  2.18s/it]cv 값 : 1.6481842477487383람다 값 : 0.18041114071204228



![png](output_52_3851.png)


     66%|███████████████████████████████████████████████████▍                          | 1284/1948 [33:32<23:03,  2.08s/it]cv 값 : 0.32219723858679195람다 값 : 0.6633789502187469



![png](output_52_3854.png)


     66%|███████████████████████████████████████████████████▍                          | 1285/1948 [33:34<21:59,  1.99s/it]cv 값 : 0.33123225389393657람다 값 : 0.5792817759219164



![png](output_52_3857.png)


     66%|███████████████████████████████████████████████████▍                          | 1286/1948 [33:36<20:10,  1.83s/it]cv 값 : 0.13271157888041174람다 값 : 1.3938171118483191



![png](output_52_3860.png)


     66%|███████████████████████████████████████████████████▌                          | 1287/1948 [33:37<19:55,  1.81s/it]cv 값 : 0.0869347453477202람다 값 : 1.9808740150806075



![png](output_52_3863.png)


     66%|███████████████████████████████████████████████████▌                          | 1288/1948 [33:41<27:03,  2.46s/it]cv 값 : 0.06527673775779724람다 값 : 1.513697269650896



![png](output_52_3866.png)


     66%|███████████████████████████████████████████████████▌                          | 1289/1948 [33:47<38:06,  3.47s/it]cv 값 : 0.15319265365224743람다 값 : 3.2992148410613895



![png](output_52_3869.png)


     66%|███████████████████████████████████████████████████▋                          | 1290/1948 [33:50<34:38,  3.16s/it]cv 값 : 0.22474410760328029람다 값 : -0.5740752712695022



![png](output_52_3872.png)


     66%|███████████████████████████████████████████████████▋                          | 1291/1948 [33:51<30:05,  2.75s/it]cv 값 : 0.3489717456258033람다 값 : 1.8102734787595995



![png](output_52_3875.png)


     66%|███████████████████████████████████████████████████▋                          | 1292/1948 [33:55<32:03,  2.93s/it]cv 값 : 0.11016345278295481람다 값 : 1.19913502231242



![png](output_52_3878.png)


     66%|███████████████████████████████████████████████████▊                          | 1293/1948 [33:59<36:19,  3.33s/it]cv 값 : 0.16091045364151813람다 값 : 2.4636460207288007



![png](output_52_3881.png)


     66%|███████████████████████████████████████████████████▊                          | 1294/1948 [34:01<32:55,  3.02s/it]cv 값 : 0.8874575159854994람다 값 : 0.28673403648329115



![png](output_52_3884.png)


     66%|███████████████████████████████████████████████████▊                          | 1295/1948 [34:03<29:27,  2.71s/it]cv 값 : 0.29071353634789676람다 값 : 1.514812311058478



![png](output_52_3887.png)


     67%|███████████████████████████████████████████████████▉                          | 1296/1948 [34:06<27:55,  2.57s/it]cv 값 : 0.36198523464324805람다 값 : 0.59787308204351



![png](output_52_3890.png)


     67%|███████████████████████████████████████████████████▉                          | 1297/1948 [34:08<25:52,  2.38s/it]cv 값 : 0.9200438185943383람다 값 : 0.34172154614574807



![png](output_52_3893.png)


     67%|███████████████████████████████████████████████████▉                          | 1298/1948 [34:10<26:11,  2.42s/it]cv 값 : 0.9584140716980719람다 값 : 0.3098227339306907



![png](output_52_3896.png)


     67%|████████████████████████████████████████████████████                          | 1299/1948 [34:12<24:54,  2.30s/it]cv 값 : 2.219764873682853람다 값 : -0.09361239914660306



![png](output_52_3899.png)


     67%|████████████████████████████████████████████████████                          | 1300/1948 [34:14<22:56,  2.12s/it]cv 값 : 0.6996261879028417람다 값 : 0.21742905926950423



![png](output_52_3902.png)


     67%|████████████████████████████████████████████████████                          | 1301/1948 [34:16<22:16,  2.07s/it]cv 값 : 0.5910131465980427람다 값 : -0.006794879482443801



![png](output_52_3905.png)


     67%|████████████████████████████████████████████████████▏                         | 1302/1948 [34:18<21:41,  2.01s/it]cv 값 : 0.6340835171886136람다 값 : 0.24664313271794785



![png](output_52_3908.png)


     67%|████████████████████████████████████████████████████▏                         | 1303/1948 [34:20<21:53,  2.04s/it]cv 값 : 0.45921412873361916람다 값 : 0.7113120653984981



![png](output_52_3911.png)


     67%|████████████████████████████████████████████████████▏                         | 1304/1948 [34:21<20:40,  1.93s/it]cv 값 : 0.48106206604140267람다 값 : -0.847639607725484



![png](output_52_3914.png)


     67%|████████████████████████████████████████████████████▎                         | 1305/1948 [34:23<20:10,  1.88s/it]cv 값 : 0.39738454624777225람다 값 : 1.4391463838279808



![png](output_52_3917.png)


     67%|████████████████████████████████████████████████████▎                         | 1306/1948 [34:26<21:38,  2.02s/it]cv 값 : 0.7072059963584065람다 값 : 0.13943431292048353



![png](output_52_3920.png)


     67%|████████████████████████████████████████████████████▎                         | 1307/1948 [34:28<21:36,  2.02s/it]cv 값 : 1.3946194333030153람다 값 : -0.07283536532661858



![png](output_52_3923.png)


     67%|████████████████████████████████████████████████████▎                         | 1308/1948 [34:30<21:25,  2.01s/it]cv 값 : 0.1951368628755422람다 값 : 1.8510572731890813



![png](output_52_3926.png)


     67%|████████████████████████████████████████████████████▍                         | 1309/1948 [34:31<20:57,  1.97s/it]cv 값 : 0.7318267054109658람다 값 : 0.2416415057529003



![png](output_52_3929.png)


     67%|████████████████████████████████████████████████████▍                         | 1310/1948 [34:33<20:38,  1.94s/it]cv 값 : 0.14673519851481584람다 값 : 5.006667398068151



![png](output_52_3932.png)


     67%|████████████████████████████████████████████████████▍                         | 1311/1948 [34:36<22:20,  2.10s/it]cv 값 : 1.17161496497344람다 값 : 0.22038015632645636



![png](output_52_3935.png)


     67%|████████████████████████████████████████████████████▌                         | 1312/1948 [34:38<21:46,  2.05s/it]cv 값 : 0.6830633974159814람다 값 : 0.33926739419753366



![png](output_52_3938.png)


     67%|████████████████████████████████████████████████████▌                         | 1313/1948 [34:40<21:27,  2.03s/it]cv 값 : 0.6946780111394149람다 값 : 0.5183927201968852



![png](output_52_3941.png)


     67%|████████████████████████████████████████████████████▌                         | 1314/1948 [34:42<21:19,  2.02s/it]cv 값 : 0.8370588496887503람다 값 : 0.3336286080648786



![png](output_52_3944.png)


     68%|████████████████████████████████████████████████████▋                         | 1315/1948 [34:43<20:21,  1.93s/it]cv 값 : 1.2473527913246505람다 값 : -0.15577400841646238



![png](output_52_3947.png)


     68%|████████████████████████████████████████████████████▋                         | 1316/1948 [34:45<20:31,  1.95s/it]cv 값 : 0.23208810246016964람다 값 : 0.5151377367456157



![png](output_52_3950.png)


     68%|████████████████████████████████████████████████████▋                         | 1317/1948 [34:47<19:19,  1.84s/it]cv 값 : 0.6283127592425863람다 값 : 0.4376157950577412



![png](output_52_3953.png)


     68%|████████████████████████████████████████████████████▊                         | 1318/1948 [34:49<18:58,  1.81s/it]cv 값 : 0.5928256513934469람다 값 : 0.6325423318589446



![png](output_52_3956.png)


     68%|████████████████████████████████████████████████████▊                         | 1319/1948 [34:51<19:08,  1.83s/it]cv 값 : 0.34813893527702133람다 값 : 1.0552312860783946



![png](output_52_3959.png)


     68%|████████████████████████████████████████████████████▊                         | 1320/1948 [34:52<19:11,  1.83s/it]cv 값 : 0.4589946200704444람다 값 : 1.0068032212124165



![png](output_52_3962.png)


     68%|████████████████████████████████████████████████████▉                         | 1321/1948 [34:54<18:50,  1.80s/it]cv 값 : 0.40287561481139406람다 값 : 0.8652905605546299



![png](output_52_3965.png)


     68%|████████████████████████████████████████████████████▉                         | 1322/1948 [34:56<18:44,  1.80s/it]cv 값 : 0.4519179067229557람다 값 : 0.6691166550744059



![png](output_52_3968.png)


     68%|████████████████████████████████████████████████████▉                         | 1323/1948 [34:58<18:28,  1.77s/it]cv 값 : 0.9169762248710469람다 값 : 0.24980208685273064



![png](output_52_3971.png)


     68%|█████████████████████████████████████████████████████                         | 1324/1948 [34:59<18:31,  1.78s/it]cv 값 : 0.50771147771336람다 값 : -0.16535129269659887



![png](output_52_3974.png)


     68%|█████████████████████████████████████████████████████                         | 1325/1948 [35:01<19:01,  1.83s/it]cv 값 : 0.6197019752295972람다 값 : 0.33646943078203717



![png](output_52_3977.png)


     68%|█████████████████████████████████████████████████████                         | 1326/1948 [35:03<18:09,  1.75s/it]cv 값 : 1.3750436450985952람다 값 : 0.16997387805034597



![png](output_52_3980.png)


     68%|█████████████████████████████████████████████████████▏                        | 1327/1948 [35:05<18:54,  1.83s/it]cv 값 : 0.8211011908298482람다 값 : 0.37453911722878774



![png](output_52_3983.png)


     68%|█████████████████████████████████████████████████████▏                        | 1328/1948 [35:08<21:38,  2.09s/it]cv 값 : 0.9150132751733178람다 값 : 0.2928569439211851



![png](output_52_3986.png)


     68%|█████████████████████████████████████████████████████▏                        | 1329/1948 [35:11<25:41,  2.49s/it]cv 값 : 0.346169661237116람다 값 : 0.9364917123101781



![png](output_52_3989.png)


     68%|█████████████████████████████████████████████████████▎                        | 1330/1948 [35:13<25:12,  2.45s/it]cv 값 : 0.2858781251470286람다 값 : 0.6895743284745444



![png](output_52_3992.png)


     68%|█████████████████████████████████████████████████████▎                        | 1331/1948 [35:16<25:16,  2.46s/it]cv 값 : 0.32283000754597835람다 값 : 1.3730818560643594



![png](output_52_3995.png)


     68%|█████████████████████████████████████████████████████▎                        | 1332/1948 [35:19<26:12,  2.55s/it]cv 값 : 0.4664072877999676람다 값 : -0.006414144598230285



![png](output_52_3998.png)


     68%|█████████████████████████████████████████████████████▎                        | 1333/1948 [35:23<30:28,  2.97s/it]cv 값 : 1.0381400585788083람다 값 : 0.26202559514003654



![png](output_52_4001.png)


     68%|█████████████████████████████████████████████████████▍                        | 1334/1948 [35:29<40:07,  3.92s/it]cv 값 : 0.37198945007330664람다 값 : 1.383417090582477



![png](output_52_4004.png)


     69%|█████████████████████████████████████████████████████▍                        | 1335/1948 [35:33<39:39,  3.88s/it]cv 값 : 0.2638963915791025람다 값 : 1.4599414642689068



![png](output_52_4007.png)


     69%|█████████████████████████████████████████████████████▍                        | 1336/1948 [35:37<41:48,  4.10s/it]cv 값 : 0.9849928949122395람다 값 : 0.2108424866308086



![png](output_52_4010.png)


     69%|█████████████████████████████████████████████████████▌                        | 1337/1948 [35:40<37:09,  3.65s/it]cv 값 : 0.27046051939769106람다 값 : 1.352775134818799



![png](output_52_4013.png)


     69%|█████████████████████████████████████████████████████▌                        | 1338/1948 [35:42<33:45,  3.32s/it]cv 값 : 1.0553098649437909람다 값 : 0.313573726192485



![png](output_52_4016.png)


     69%|█████████████████████████████████████████████████████▌                        | 1339/1948 [35:45<30:11,  2.97s/it]cv 값 : 1.654153627999315람다 값 : -0.12151627475222343



![png](output_52_4019.png)


     69%|█████████████████████████████████████████████████████▋                        | 1340/1948 [35:46<26:33,  2.62s/it]cv 값 : 0.2632897820691667람다 값 : -0.096591096239652



![png](output_52_4022.png)


     69%|█████████████████████████████████████████████████████▋                        | 1341/1948 [35:48<24:43,  2.44s/it]cv 값 : 0.21163501111042857람다 값 : 1.238651050156749



![png](output_52_4025.png)


     69%|█████████████████████████████████████████████████████▋                        | 1342/1948 [35:50<23:30,  2.33s/it]cv 값 : 1.0572013311966137람다 값 : 0.043142329169813066



![png](output_52_4028.png)


     69%|█████████████████████████████████████████████████████▊                        | 1343/1948 [35:52<22:08,  2.20s/it]cv 값 : 0.29054158547705233람다 값 : 0.7921674293147791



![png](output_52_4031.png)


     69%|█████████████████████████████████████████████████████▊                        | 1344/1948 [35:54<21:20,  2.12s/it]cv 값 : 0.974792383748063람다 값 : -0.46643915944641196



![png](output_52_4034.png)


     69%|█████████████████████████████████████████████████████▊                        | 1345/1948 [35:56<20:35,  2.05s/it]cv 값 : 0.4830853656161255람다 값 : 0.42110938642888635



![png](output_52_4037.png)


     69%|█████████████████████████████████████████████████████▉                        | 1346/1948 [35:58<19:39,  1.96s/it]cv 값 : 0.19680228935261032람다 값 : 0.46174679624237674



![png](output_52_4040.png)


     69%|█████████████████████████████████████████████████████▉                        | 1347/1948 [36:00<19:31,  1.95s/it]cv 값 : 0.6367842722371382람다 값 : 0.5284067729282578



![png](output_52_4043.png)


     69%|█████████████████████████████████████████████████████▉                        | 1348/1948 [36:02<19:12,  1.92s/it]cv 값 : 1.0840177073742963람다 값 : 0.04836512480215962



![png](output_52_4046.png)


     69%|██████████████████████████████████████████████████████                        | 1349/1948 [36:03<18:30,  1.85s/it]cv 값 : 0.513441718079506람다 값 : 0.509073556852897



![png](output_52_4049.png)


     69%|██████████████████████████████████████████████████████                        | 1350/1948 [36:05<18:51,  1.89s/it]cv 값 : 0.1895143596216344람다 값 : -0.9094063014791833



![png](output_52_4052.png)


     69%|██████████████████████████████████████████████████████                        | 1351/1948 [36:07<18:10,  1.83s/it]cv 값 : 0.47772775107248505람다 값 : -0.25155790029381714



![png](output_52_4055.png)


     69%|██████████████████████████████████████████████████████▏                       | 1352/1948 [36:09<18:45,  1.89s/it]cv 값 : 0.5080776365596507람다 값 : -0.5205579865027966



![png](output_52_4058.png)


     69%|██████████████████████████████████████████████████████▏                       | 1353/1948 [36:11<18:22,  1.85s/it]cv 값 : 1.812762442968189람다 값 : 0.1809292000567434



![png](output_52_4061.png)


     70%|██████████████████████████████████████████████████████▏                       | 1354/1948 [36:12<17:43,  1.79s/it]cv 값 : 0.7713018919564835람다 값 : 0.398098088450645



![png](output_52_4064.png)


     70%|██████████████████████████████████████████████████████▎                       | 1355/1948 [36:15<18:53,  1.91s/it]cv 값 : 0.32179735105629226람다 값 : 0.9069741838085515



![png](output_52_4067.png)


     70%|██████████████████████████████████████████████████████▎                       | 1356/1948 [36:17<19:00,  1.93s/it]cv 값 : 0.4987714487349207람다 값 : 0.5929813282466091



![png](output_52_4070.png)


     70%|██████████████████████████████████████████████████████▎                       | 1357/1948 [36:18<18:23,  1.87s/it]cv 값 : 0.4009081953653638람다 값 : -0.1847815482162369



![png](output_52_4073.png)


     70%|██████████████████████████████████████████████████████▍                       | 1358/1948 [36:21<19:31,  1.99s/it]cv 값 : 0.22411076160976134람다 값 : 0.22330840409782343



![png](output_52_4076.png)


     70%|██████████████████████████████████████████████████████▍                       | 1359/1948 [36:22<19:05,  1.94s/it]cv 값 : 0.7054347429591187람다 값 : 0.4913076949462151



![png](output_52_4079.png)


     70%|██████████████████████████████████████████████████████▍                       | 1360/1948 [36:24<18:50,  1.92s/it]cv 값 : 0.8624832341342527람다 값 : 0.30475922428474495



![png](output_52_4082.png)


     70%|██████████████████████████████████████████████████████▍                       | 1361/1948 [36:26<19:20,  1.98s/it]cv 값 : 0.44299445204739746람다 값 : 1.2786848523728767



![png](output_52_4085.png)


     70%|██████████████████████████████████████████████████████▌                       | 1362/1948 [36:28<18:58,  1.94s/it]cv 값 : 0.36448349969328253람다 값 : 1.0136325708875364



![png](output_52_4088.png)


     70%|██████████████████████████████████████████████████████▌                       | 1363/1948 [36:30<18:50,  1.93s/it]cv 값 : 0.6129879679883238람다 값 : 0.3854434472327749



![png](output_52_4091.png)


     70%|██████████████████████████████████████████████████████▌                       | 1364/1948 [36:32<17:57,  1.84s/it]cv 값 : 1.1931344593923918람다 값 : 0.2891121906219184



![png](output_52_4094.png)


     70%|██████████████████████████████████████████████████████▋                       | 1365/1948 [36:34<18:40,  1.92s/it]cv 값 : 0.6287005869701012람다 값 : 0.4344869350266625



![png](output_52_4097.png)


     70%|██████████████████████████████████████████████████████▋                       | 1366/1948 [36:36<19:30,  2.01s/it]cv 값 : 0.26098093994723187람다 값 : 0.7530401239288064



![png](output_52_4100.png)


     70%|██████████████████████████████████████████████████████▋                       | 1367/1948 [36:38<19:11,  1.98s/it]cv 값 : 0.38746378702369827람다 값 : 0.6751807132741894



![png](output_52_4103.png)


     70%|██████████████████████████████████████████████████████▊                       | 1368/1948 [36:40<18:48,  1.95s/it]cv 값 : 0.23527326071023374람다 값 : 1.9944421841384277



![png](output_52_4106.png)


     70%|██████████████████████████████████████████████████████▊                       | 1369/1948 [36:42<19:28,  2.02s/it]cv 값 : 0.5520017882464198람다 값 : 0.4973409269750118



![png](output_52_4109.png)


     70%|██████████████████████████████████████████████████████▊                       | 1370/1948 [36:44<18:35,  1.93s/it]cv 값 : 0.8171701147287319람다 값 : 0.08840617747159749



![png](output_52_4112.png)


     70%|██████████████████████████████████████████████████████▉                       | 1371/1948 [36:46<18:45,  1.95s/it]cv 값 : 0.53258592489345람다 값 : 0.7678358974825077



![png](output_52_4115.png)


     70%|██████████████████████████████████████████████████████▉                       | 1372/1948 [36:47<17:50,  1.86s/it]cv 값 : 0.2803378711519288람다 값 : 0.5944536723952037



![png](output_52_4118.png)


     70%|██████████████████████████████████████████████████████▉                       | 1373/1948 [36:49<17:40,  1.85s/it]cv 값 : 0.5957769386223994람다 값 : 0.5399265896625818



![png](output_52_4121.png)


     71%|███████████████████████████████████████████████████████                       | 1374/1948 [36:51<18:34,  1.94s/it]cv 값 : 0.5795060279755699람다 값 : 0.5274050685086591



![png](output_52_4124.png)


     71%|███████████████████████████████████████████████████████                       | 1375/1948 [36:53<18:48,  1.97s/it]cv 값 : 0.7304694537636323람다 값 : 0.3120244167110808



![png](output_52_4127.png)


     71%|███████████████████████████████████████████████████████                       | 1376/1948 [36:55<18:21,  1.93s/it]cv 값 : 0.3595040428010071람다 값 : 0.5068978802560323



![png](output_52_4130.png)


     71%|███████████████████████████████████████████████████████▏                      | 1377/1948 [36:57<17:43,  1.86s/it]cv 값 : 0.2905431948744895람다 값 : -1.228469468636596



![png](output_52_4133.png)


     71%|███████████████████████████████████████████████████████▏                      | 1378/1948 [36:59<19:12,  2.02s/it]cv 값 : 0.45552859606251367람다 값 : 0.7526059303259723



![png](output_52_4136.png)


     71%|███████████████████████████████████████████████████████▏                      | 1379/1948 [37:01<19:04,  2.01s/it]cv 값 : 0.641136951751709람다 값 : 0.5406449422598211



![png](output_52_4139.png)


     71%|███████████████████████████████████████████████████████▎                      | 1380/1948 [37:03<18:43,  1.98s/it]cv 값 : 0.2265882605379048람다 값 : 2.0255701307349376



![png](output_52_4142.png)


     71%|███████████████████████████████████████████████████████▎                      | 1381/1948 [37:05<18:45,  1.98s/it]cv 값 : 0.6108246417007728람다 값 : 0.4326154809315267



![png](output_52_4145.png)


     71%|███████████████████████████████████████████████████████▎                      | 1382/1948 [37:07<18:35,  1.97s/it]cv 값 : 0.423189193335653람다 값 : 1.1531087798113464



![png](output_52_4148.png)


     71%|███████████████████████████████████████████████████████▍                      | 1383/1948 [37:10<19:56,  2.12s/it]cv 값 : 0.5182487734708775람다 값 : 0.15949330613406615



![png](output_52_4151.png)


     71%|███████████████████████████████████████████████████████▍                      | 1384/1948 [37:12<19:07,  2.04s/it]cv 값 : 0.34087193231089563람다 값 : 1.0688500834626702



![png](output_52_4154.png)


     71%|███████████████████████████████████████████████████████▍                      | 1385/1948 [37:13<18:26,  1.97s/it]cv 값 : 0.670388467734995람다 값 : 0.17413401205369422



![png](output_52_4157.png)


     71%|███████████████████████████████████████████████████████▍                      | 1386/1948 [37:15<18:12,  1.94s/it]cv 값 : 0.3108222424800956람다 값 : 1.003226917532686



![png](output_52_4160.png)


     71%|███████████████████████████████████████████████████████▌                      | 1387/1948 [37:17<18:23,  1.97s/it]cv 값 : 1.122505289712284람다 값 : 0.21084913168745414



![png](output_52_4163.png)


     71%|███████████████████████████████████████████████████████▌                      | 1388/1948 [37:20<19:06,  2.05s/it]cv 값 : 0.46828938397308256람다 값 : 0.7686651657810468



![png](output_52_4166.png)


     71%|███████████████████████████████████████████████████████▌                      | 1389/1948 [37:21<18:55,  2.03s/it]cv 값 : 0.6469886912064662람다 값 : 0.480157807619451



![png](output_52_4169.png)


     71%|███████████████████████████████████████████████████████▋                      | 1390/1948 [37:23<18:47,  2.02s/it]cv 값 : 0.9111277485189292람다 값 : -0.21501723322724386



![png](output_52_4172.png)


     71%|███████████████████████████████████████████████████████▋                      | 1391/1948 [37:26<19:31,  2.10s/it]cv 값 : 0.18963128927439865람다 값 : -0.6738329280948485



![png](output_52_4175.png)


     71%|███████████████████████████████████████████████████████▋                      | 1392/1948 [37:27<18:16,  1.97s/it]cv 값 : 0.4585295888228344람다 값 : 0.8340232868243007



![png](output_52_4178.png)


     72%|███████████████████████████████████████████████████████▊                      | 1393/1948 [37:29<18:17,  1.98s/it]cv 값 : 0.16667628499132822람다 값 : -1.5539711544072026



![png](output_52_4181.png)


     72%|███████████████████████████████████████████████████████▊                      | 1394/1948 [37:31<18:21,  1.99s/it]cv 값 : 0.18149542708246177람다 값 : -0.5030941388594302



![png](output_52_4184.png)


     72%|███████████████████████████████████████████████████████▊                      | 1395/1948 [37:33<17:54,  1.94s/it]cv 값 : 0.4819093294385957람다 값 : 0.6784199769931308



![png](output_52_4187.png)


     72%|███████████████████████████████████████████████████████▉                      | 1396/1948 [37:35<18:10,  1.98s/it]cv 값 : 0.7329844493887454람다 값 : 0.34409371837802905



![png](output_52_4190.png)


     72%|███████████████████████████████████████████████████████▉                      | 1397/1948 [37:37<17:30,  1.91s/it]cv 값 : 0.4580951666574467람다 값 : 0.7743884234264534



![png](output_52_4193.png)


     72%|███████████████████████████████████████████████████████▉                      | 1398/1948 [37:39<17:33,  1.92s/it]cv 값 : 0.2513951143585654람다 값 : 1.657442667720938



![png](output_52_4196.png)


     72%|████████████████████████████████████████████████████████                      | 1399/1948 [37:41<17:50,  1.95s/it]cv 값 : 0.6648036106379788람다 값 : 0.480659525403855



![png](output_52_4199.png)


     72%|████████████████████████████████████████████████████████                      | 1400/1948 [37:43<17:48,  1.95s/it]cv 값 : 0.4985939781607275람다 값 : 0.947306755260778



![png](output_52_4202.png)


     72%|████████████████████████████████████████████████████████                      | 1401/1948 [37:45<17:22,  1.91s/it]cv 값 : 0.4479086079713767람다 값 : -0.51421874841243



![png](output_52_4205.png)


     72%|████████████████████████████████████████████████████████▏                     | 1402/1948 [37:47<17:46,  1.95s/it]cv 값 : 0.315064181922956람다 값 : 1.064356064225078



![png](output_52_4208.png)


     72%|████████████████████████████████████████████████████████▏                     | 1403/1948 [37:49<19:11,  2.11s/it]cv 값 : 0.8391781664291555람다 값 : 0.34143551408993844



![png](output_52_4211.png)


     72%|████████████████████████████████████████████████████████▏                     | 1404/1948 [37:52<19:23,  2.14s/it]cv 값 : 0.313448715909235람다 값 : 0.02475839209859783



![png](output_52_4214.png)


     72%|████████████████████████████████████████████████████████▎                     | 1405/1948 [37:54<19:42,  2.18s/it]cv 값 : 0.5315616968091454람다 값 : 0.6172525486646534



![png](output_52_4217.png)


     72%|████████████████████████████████████████████████████████▎                     | 1406/1948 [37:56<19:16,  2.13s/it]cv 값 : 0.5124003391049183람다 값 : -0.10252659436031802



![png](output_52_4220.png)


     72%|████████████████████████████████████████████████████████▎                     | 1407/1948 [37:58<18:27,  2.05s/it]cv 값 : 0.7902235818261787람다 값 : 0.0101896419248831



![png](output_52_4223.png)


     72%|████████████████████████████████████████████████████████▍                     | 1408/1948 [37:59<17:40,  1.96s/it]cv 값 : 0.30317673877973594람다 값 : 1.25490790118575



![png](output_52_4226.png)


     72%|████████████████████████████████████████████████████████▍                     | 1409/1948 [38:02<18:08,  2.02s/it]cv 값 : 0.28347247410004905람다 값 : 0.11041748700255052



![png](output_52_4229.png)


     72%|████████████████████████████████████████████████████████▍                     | 1410/1948 [38:03<17:03,  1.90s/it]cv 값 : 0.23034406115156414람다 값 : 1.7752477938366644



![png](output_52_4232.png)


     72%|████████████████████████████████████████████████████████▍                     | 1411/1948 [38:05<17:41,  1.98s/it]cv 값 : 0.4431815433513807람다 값 : 0.565983858850388



![png](output_52_4235.png)


     72%|████████████████████████████████████████████████████████▌                     | 1412/1948 [38:08<18:51,  2.11s/it]cv 값 : 1.5268759741619093람다 값 : 0.18579265933980538



![png](output_52_4238.png)


     73%|████████████████████████████████████████████████████████▌                     | 1413/1948 [38:10<19:27,  2.18s/it]cv 값 : 0.48018917856171506람다 값 : 0.508577803003389



![png](output_52_4241.png)


     73%|████████████████████████████████████████████████████████▌                     | 1414/1948 [38:12<19:14,  2.16s/it]cv 값 : 0.42775255972631004람다 값 : 0.040890791883422546



![png](output_52_4244.png)


     73%|████████████████████████████████████████████████████████▋                     | 1415/1948 [38:14<18:41,  2.10s/it]cv 값 : 0.5682450291755334람다 값 : 0.5995840150840043



![png](output_52_4247.png)


     73%|████████████████████████████████████████████████████████▋                     | 1416/1948 [38:16<18:14,  2.06s/it]cv 값 : 0.698599090186437람다 값 : 0.57514037573084



![png](output_52_4250.png)


     73%|████████████████████████████████████████████████████████▋                     | 1417/1948 [38:18<17:50,  2.02s/it]cv 값 : 1.1473682136370877람다 값 : 0.2094946713636221



![png](output_52_4253.png)


     73%|████████████████████████████████████████████████████████▊                     | 1418/1948 [38:21<18:49,  2.13s/it]cv 값 : 0.3242072750385139람다 값 : 1.2274392837571837



![png](output_52_4256.png)


     73%|████████████████████████████████████████████████████████▊                     | 1419/1948 [38:22<18:09,  2.06s/it]cv 값 : 0.6091839216867491람다 값 : 0.31791843482059695



![png](output_52_4259.png)


     73%|████████████████████████████████████████████████████████▊                     | 1420/1948 [38:25<18:25,  2.09s/it]cv 값 : 0.21255384648083445람다 값 : 0.2598489669739609



![png](output_52_4262.png)


     73%|████████████████████████████████████████████████████████▉                     | 1421/1948 [38:26<17:25,  1.98s/it]cv 값 : 0.5083861906238236람다 값 : 0.07582957122417323



![png](output_52_4265.png)


     73%|████████████████████████████████████████████████████████▉                     | 1422/1948 [38:28<16:42,  1.91s/it]cv 값 : 0.4561169066396378람다 값 : 1.097921143143207



![png](output_52_4268.png)


     73%|████████████████████████████████████████████████████████▉                     | 1423/1948 [38:30<17:01,  1.95s/it]cv 값 : 0.4504260614801461람다 값 : 0.5983465484280254



![png](output_52_4271.png)


     73%|█████████████████████████████████████████████████████████                     | 1424/1948 [38:32<16:01,  1.83s/it]cv 값 : 0.3888291284538432람다 값 : -1.0753497332910575



![png](output_52_4274.png)


     73%|█████████████████████████████████████████████████████████                     | 1425/1948 [38:33<15:52,  1.82s/it]cv 값 : 0.6905875419549912람다 값 : 0.3353747559011394



![png](output_52_4277.png)


     73%|█████████████████████████████████████████████████████████                     | 1426/1948 [38:35<15:33,  1.79s/it]cv 값 : 0.35385996773715345람다 값 : 0.7862013882073522



![png](output_52_4280.png)


     73%|█████████████████████████████████████████████████████████▏                    | 1427/1948 [38:37<15:28,  1.78s/it]cv 값 : 0.3771744843458967람다 값 : 0.4816516929009207



![png](output_52_4283.png)


     73%|█████████████████████████████████████████████████████████▏                    | 1428/1948 [38:39<16:03,  1.85s/it]cv 값 : 0.39239555322560116람다 값 : -0.20462394533958714



![png](output_52_4286.png)


     73%|█████████████████████████████████████████████████████████▏                    | 1429/1948 [38:41<16:14,  1.88s/it]cv 값 : 0.4643774200454108람다 값 : 0.36702892947823273



![png](output_52_4289.png)


     73%|█████████████████████████████████████████████████████████▎                    | 1430/1948 [38:43<15:44,  1.82s/it]cv 값 : 0.38992900107793543람다 값 : 0.2673561719058833



![png](output_52_4292.png)


     73%|█████████████████████████████████████████████████████████▎                    | 1431/1948 [38:44<15:16,  1.77s/it]cv 값 : 0.6413783078043551람다 값 : 0.3063571249777256



![png](output_52_4295.png)


     74%|█████████████████████████████████████████████████████████▎                    | 1432/1948 [38:46<15:51,  1.84s/it]cv 값 : 0.4016747369832147람다 값 : 0.8994310257340814



![png](output_52_4298.png)


     74%|█████████████████████████████████████████████████████████▍                    | 1433/1948 [38:48<15:38,  1.82s/it]cv 값 : 0.4012274798534459람다 값 : -0.0679808888468976



![png](output_52_4301.png)


     74%|█████████████████████████████████████████████████████████▍                    | 1434/1948 [38:50<15:53,  1.85s/it]cv 값 : 0.23719935275096793람다 값 : 1.2108939459603303



![png](output_52_4304.png)


     74%|█████████████████████████████████████████████████████████▍                    | 1435/1948 [38:52<15:42,  1.84s/it]cv 값 : 0.30099103104534847람다 값 : 1.5555117636011588



![png](output_52_4307.png)


     74%|█████████████████████████████████████████████████████████▍                    | 1436/1948 [38:54<15:47,  1.85s/it]cv 값 : 0.22601596274818628람다 값 : 2.1410567346493607



![png](output_52_4310.png)


     74%|█████████████████████████████████████████████████████████▌                    | 1437/1948 [38:55<15:35,  1.83s/it]cv 값 : 0.3077062487605162람다 값 : 1.3941955977297045



![png](output_52_4313.png)


     74%|█████████████████████████████████████████████████████████▌                    | 1438/1948 [38:57<16:08,  1.90s/it]cv 값 : 0.6133530838042677람다 값 : 0.31063306797233137



![png](output_52_4316.png)


     74%|█████████████████████████████████████████████████████████▌                    | 1439/1948 [38:59<16:00,  1.89s/it]cv 값 : 0.37417954233183137람다 값 : 1.1731443163915936



![png](output_52_4319.png)


     74%|█████████████████████████████████████████████████████████▋                    | 1440/1948 [39:01<16:27,  1.94s/it]cv 값 : 0.41592246709367064람다 값 : 0.955060160640341



![png](output_52_4322.png)


     74%|█████████████████████████████████████████████████████████▋                    | 1441/1948 [39:03<16:02,  1.90s/it]cv 값 : 1.1370839393752579람다 값 : 0.2927859249714457



![png](output_52_4325.png)


     74%|█████████████████████████████████████████████████████████▋                    | 1442/1948 [39:05<16:03,  1.90s/it]cv 값 : 0.7404649702399748람다 값 : 0.26463637459476097



![png](output_52_4328.png)


     74%|█████████████████████████████████████████████████████████▊                    | 1443/1948 [39:07<15:45,  1.87s/it]cv 값 : 0.30187092750618005람다 값 : 0.7800456158365525



![png](output_52_4331.png)


     74%|█████████████████████████████████████████████████████████▊                    | 1444/1948 [39:09<15:10,  1.81s/it]cv 값 : 0.38536187701028496람다 값 : 0.9487039219777612



![png](output_52_4334.png)


     74%|█████████████████████████████████████████████████████████▊                    | 1445/1948 [39:10<15:21,  1.83s/it]cv 값 : 0.7169564804426174람다 값 : 0.5378627868477163



![png](output_52_4337.png)


     74%|█████████████████████████████████████████████████████████▉                    | 1446/1948 [39:12<14:59,  1.79s/it]cv 값 : 0.6121674891286367람다 값 : 0.1335130318059613



![png](output_52_4340.png)


     74%|█████████████████████████████████████████████████████████▉                    | 1447/1948 [39:14<14:56,  1.79s/it]cv 값 : 0.44092218127479965람다 값 : 1.0501208962593154



![png](output_52_4343.png)


     74%|█████████████████████████████████████████████████████████▉                    | 1448/1948 [39:16<15:05,  1.81s/it]cv 값 : 0.3661650536153143람다 값 : 1.126301808435386



![png](output_52_4346.png)


     74%|██████████████████████████████████████████████████████████                    | 1449/1948 [39:18<15:17,  1.84s/it]cv 값 : 0.6588422593241686람다 값 : 0.5362765630263343



![png](output_52_4349.png)


     74%|██████████████████████████████████████████████████████████                    | 1450/1948 [39:19<14:44,  1.78s/it]cv 값 : 0.33249336670001395람다 값 : 1.4075007089504368



![png](output_52_4352.png)


     74%|██████████████████████████████████████████████████████████                    | 1451/1948 [39:21<15:15,  1.84s/it]cv 값 : 0.5021498275365935람다 값 : 1.2023105347627236



![png](output_52_4355.png)


     75%|██████████████████████████████████████████████████████████▏                   | 1452/1948 [39:23<15:09,  1.83s/it]cv 값 : 0.11300693838091164람다 값 : -1.3386382937170513



![png](output_52_4358.png)


     75%|██████████████████████████████████████████████████████████▏                   | 1453/1948 [39:25<15:51,  1.92s/it]cv 값 : 0.3240145022127973람다 값 : 0.2813891584007534



![png](output_52_4361.png)


     75%|██████████████████████████████████████████████████████████▏                   | 1454/1948 [39:27<15:12,  1.85s/it]cv 값 : 0.2630298550545558람다 값 : 1.1241980831470828



![png](output_52_4364.png)


     75%|██████████████████████████████████████████████████████████▎                   | 1455/1948 [39:29<15:01,  1.83s/it]cv 값 : 0.44352273122039365람다 값 : -0.6165722878734031



![png](output_52_4367.png)


     75%|██████████████████████████████████████████████████████████▎                   | 1456/1948 [39:31<15:28,  1.89s/it]cv 값 : 0.5689107968585335람다 값 : -0.062392909687594214



![png](output_52_4370.png)


     75%|██████████████████████████████████████████████████████████▎                   | 1457/1948 [39:32<14:56,  1.82s/it]cv 값 : 0.22648106976499474람다 값 : 1.2864924197468823



![png](output_52_4373.png)


     75%|██████████████████████████████████████████████████████████▍                   | 1458/1948 [39:34<15:12,  1.86s/it]cv 값 : 0.38373819493324507람다 값 : -0.014069410908102863



![png](output_52_4376.png)


     75%|██████████████████████████████████████████████████████████▍                   | 1459/1948 [39:37<15:48,  1.94s/it]cv 값 : 0.39930363996656737람다 값 : 0.31787853128957655



![png](output_52_4379.png)


     75%|██████████████████████████████████████████████████████████▍                   | 1460/1948 [39:38<15:35,  1.92s/it]cv 값 : 0.5478721860433223람다 값 : 0.31483071758023135



![png](output_52_4382.png)


     75%|██████████████████████████████████████████████████████████▌                   | 1461/1948 [39:41<18:29,  2.28s/it]cv 값 : 0.754290262880034람다 값 : 0.36726676695319316



![png](output_52_4385.png)


     75%|██████████████████████████████████████████████████████████▌                   | 1462/1948 [39:43<16:42,  2.06s/it]cv 값 : 0.15931122609859658람다 값 : 1.2485292694159367



![png](output_52_4388.png)


     75%|██████████████████████████████████████████████████████████▌                   | 1463/1948 [39:47<20:55,  2.59s/it]cv 값 : 0.3740374807268638람다 값 : 0.1676264279884084



![png](output_52_4391.png)


     75%|██████████████████████████████████████████████████████████▌                   | 1464/1948 [39:51<25:13,  3.13s/it]cv 값 : 0.4078561429597432람다 값 : -1.1507117841261716



![png](output_52_4394.png)


     75%|██████████████████████████████████████████████████████████▋                   | 1465/1948 [39:54<23:39,  2.94s/it]cv 값 : 0.5352807105273277람다 값 : 0.6104883054710468



![png](output_52_4397.png)


     75%|██████████████████████████████████████████████████████████▋                   | 1466/1948 [39:56<21:28,  2.67s/it]cv 값 : 0.6683911756264512람다 값 : 0.4278957247472131



![png](output_52_4400.png)


     75%|██████████████████████████████████████████████████████████▋                   | 1467/1948 [39:58<20:20,  2.54s/it]cv 값 : 0.2439130738580723람다 값 : 0.2416205412354015



![png](output_52_4403.png)


     75%|██████████████████████████████████████████████████████████▊                   | 1468/1948 [40:00<19:21,  2.42s/it]cv 값 : 0.5769825762642544람다 값 : 0.37630372771601034



![png](output_52_4406.png)


     75%|██████████████████████████████████████████████████████████▊                   | 1469/1948 [40:04<21:50,  2.74s/it]cv 값 : 0.5050248863524148람다 값 : -0.4354015930488753



![png](output_52_4409.png)


     75%|██████████████████████████████████████████████████████████▊                   | 1470/1948 [40:08<25:25,  3.19s/it]cv 값 : 0.7289728606169706람다 값 : -0.0945629897411152



![png](output_52_4412.png)


     76%|██████████████████████████████████████████████████████████▉                   | 1471/1948 [40:10<22:51,  2.88s/it]cv 값 : 0.7095959595770062람다 값 : 0.26372799533177055



![png](output_52_4415.png)


     76%|██████████████████████████████████████████████████████████▉                   | 1472/1948 [40:11<19:22,  2.44s/it]cv 값 : 0.728973115446003람다 값 : 0.4384663929661739



![png](output_52_4418.png)


     76%|██████████████████████████████████████████████████████████▉                   | 1473/1948 [40:13<16:44,  2.11s/it]cv 값 : 0.3890057512038601람다 값 : 0.591004131538376



![png](output_52_4421.png)


     76%|███████████████████████████████████████████████████████████                   | 1474/1948 [40:14<14:11,  1.80s/it]cv 값 : 0.47680884050330014람다 값 : 0.18478118321336945



![png](output_52_4424.png)


     76%|███████████████████████████████████████████████████████████                   | 1475/1948 [40:15<12:59,  1.65s/it]cv 값 : 0.27243995575339347람다 값 : 0.5063730446642329



![png](output_52_4427.png)


     76%|███████████████████████████████████████████████████████████                   | 1476/1948 [40:17<12:23,  1.58s/it]cv 값 : 0.6972697322309859람다 값 : 0.27303790259749333



![png](output_52_4430.png)


     76%|███████████████████████████████████████████████████████████▏                  | 1477/1948 [40:18<11:28,  1.46s/it]cv 값 : 0.3365202709103794람다 값 : 0.9986804281159475



![png](output_52_4433.png)


     76%|███████████████████████████████████████████████████████████▏                  | 1478/1948 [40:19<12:02,  1.54s/it]cv 값 : 0.48908008036859024람다 값 : 0.6955149100898836



![png](output_52_4436.png)


     76%|███████████████████████████████████████████████████████████▏                  | 1479/1948 [40:21<11:26,  1.46s/it]cv 값 : 0.6201184041530187람다 값 : 0.5145816423235552



![png](output_52_4439.png)


     76%|███████████████████████████████████████████████████████████▎                  | 1480/1948 [40:22<10:43,  1.38s/it]cv 값 : 0.5692152011350631람다 값 : 0.17517686292295903



![png](output_52_4442.png)


     76%|███████████████████████████████████████████████████████████▎                  | 1481/1948 [40:23<10:25,  1.34s/it]cv 값 : 0.3549727634604314람다 값 : 1.0421521368587094



![png](output_52_4445.png)


     76%|███████████████████████████████████████████████████████████▎                  | 1482/1948 [40:24<10:06,  1.30s/it]cv 값 : 0.519974369006508람다 값 : 0.35298459810050903



![png](output_52_4448.png)


     76%|███████████████████████████████████████████████████████████▍                  | 1483/1948 [40:26<09:59,  1.29s/it]cv 값 : 0.2738734754497907람다 값 : 2.133698447672704



![png](output_52_4451.png)


     76%|███████████████████████████████████████████████████████████▍                  | 1484/1948 [40:27<09:52,  1.28s/it]cv 값 : 0.6212675102004751람다 값 : 0.46627590825661475



![png](output_52_4454.png)


     76%|███████████████████████████████████████████████████████████▍                  | 1485/1948 [40:28<10:00,  1.30s/it]cv 값 : 0.6424946177958134람다 값 : 0.5541576259215214



![png](output_52_4457.png)


     76%|███████████████████████████████████████████████████████████▌                  | 1486/1948 [40:29<09:22,  1.22s/it]cv 값 : 1.0051230843265995람다 값 : 0.09009124306229283



![png](output_52_4460.png)


     76%|███████████████████████████████████████████████████████████▌                  | 1487/1948 [40:31<09:27,  1.23s/it]cv 값 : 0.23750092057281635람다 값 : 1.8086452194487073



![png](output_52_4463.png)


     76%|███████████████████████████████████████████████████████████▌                  | 1488/1948 [40:32<09:12,  1.20s/it]cv 값 : 0.3991781281164027람다 값 : 0.9212996631987892



![png](output_52_4466.png)


     76%|███████████████████████████████████████████████████████████▌                  | 1489/1948 [40:33<09:14,  1.21s/it]cv 값 : 0.8366304309446009람다 값 : 0.35856328912203117



![png](output_52_4469.png)


     76%|███████████████████████████████████████████████████████████▋                  | 1490/1948 [40:34<09:00,  1.18s/it]cv 값 : 0.30361915016476104람다 값 : -1.4068089837789162



![png](output_52_4472.png)


     77%|███████████████████████████████████████████████████████████▋                  | 1491/1948 [40:35<09:04,  1.19s/it]cv 값 : 0.21823145780015832람다 값 : -1.036593022054091



![png](output_52_4475.png)


     77%|███████████████████████████████████████████████████████████▋                  | 1492/1948 [40:36<09:04,  1.19s/it]cv 값 : 1.3974484886638967람다 값 : 0.050780082716340734



![png](output_52_4478.png)


     77%|███████████████████████████████████████████████████████████▊                  | 1493/1948 [40:38<08:45,  1.16s/it]cv 값 : 0.49529021354476954람다 값 : 0.8108701885032348



![png](output_52_4481.png)


     77%|███████████████████████████████████████████████████████████▊                  | 1494/1948 [40:39<08:49,  1.17s/it]cv 값 : 0.1737417683549913람다 값 : 1.9245409539169038



![png](output_52_4484.png)


     77%|███████████████████████████████████████████████████████████▊                  | 1495/1948 [40:40<08:50,  1.17s/it]cv 값 : 2.129905044377825람다 값 : 0.044503125132913594



![png](output_52_4487.png)


     77%|███████████████████████████████████████████████████████████▉                  | 1496/1948 [40:41<08:41,  1.15s/it]cv 값 : 0.9875646640615154람다 값 : -0.1612684185763854



![png](output_52_4490.png)


     77%|███████████████████████████████████████████████████████████▉                  | 1497/1948 [40:42<08:32,  1.14s/it]cv 값 : 0.19558477386996315람다 값 : 0.5073023407024188



![png](output_52_4493.png)


     77%|███████████████████████████████████████████████████████████▉                  | 1498/1948 [40:43<08:41,  1.16s/it]cv 값 : 0.3184081911600211람다 값 : 0.9399946141157919



![png](output_52_4496.png)


     77%|████████████████████████████████████████████████████████████                  | 1499/1948 [40:44<08:28,  1.13s/it]cv 값 : 0.21701790544862373람다 값 : 0.5611929233370855



![png](output_52_4499.png)


     77%|████████████████████████████████████████████████████████████                  | 1500/1948 [40:46<08:29,  1.14s/it]cv 값 : 0.45990352260610606람다 값 : 0.7704013735719665



![png](output_52_4502.png)


     77%|████████████████████████████████████████████████████████████                  | 1501/1948 [40:47<08:28,  1.14s/it]cv 값 : 0.5969328182260197람다 값 : 0.5613791963001179



![png](output_52_4505.png)


     77%|████████████████████████████████████████████████████████████▏                 | 1502/1948 [40:48<08:06,  1.09s/it]cv 값 : 0.22043107128138584람다 값 : 1.7545124702706107



![png](output_52_4508.png)


     77%|████████████████████████████████████████████████████████████▏                 | 1503/1948 [40:49<08:29,  1.14s/it]cv 값 : 0.3158785182911436람다 값 : 0.007335154919522203



![png](output_52_4511.png)


     77%|████████████████████████████████████████████████████████████▏                 | 1504/1948 [40:50<08:39,  1.17s/it]cv 값 : 0.5203911692756302람다 값 : 0.5730122848322746



![png](output_52_4514.png)


     77%|████████████████████████████████████████████████████████████▎                 | 1505/1948 [40:51<08:17,  1.12s/it]cv 값 : 0.16926437281287768람다 값 : -0.24531044935555113



![png](output_52_4517.png)


     77%|████████████████████████████████████████████████████████████▎                 | 1506/1948 [40:52<08:14,  1.12s/it]cv 값 : 1.1244415152093319람다 값 : 0.15880143122081822



![png](output_52_4520.png)


     77%|████████████████████████████████████████████████████████████▎                 | 1507/1948 [40:54<08:33,  1.16s/it]cv 값 : 0.4766074139040829람다 값 : 0.09806999322449228



![png](output_52_4523.png)


     77%|████████████████████████████████████████████████████████████▍                 | 1508/1948 [40:55<08:21,  1.14s/it]cv 값 : 0.2051534962640678람다 값 : 0.21455372433282136



![png](output_52_4526.png)


     77%|████████████████████████████████████████████████████████████▍                 | 1509/1948 [40:56<08:38,  1.18s/it]cv 값 : 0.25433306959347035람다 값 : 1.277358390723186



![png](output_52_4529.png)


     78%|████████████████████████████████████████████████████████████▍                 | 1510/1948 [40:57<08:35,  1.18s/it]cv 값 : 0.2109534453320573람다 값 : 0.5731830052274414



![png](output_52_4532.png)


     78%|████████████████████████████████████████████████████████████▌                 | 1511/1948 [40:58<08:23,  1.15s/it]cv 값 : 0.43711207544266445람다 값 : 0.8628973275373009



![png](output_52_4535.png)


     78%|████████████████████████████████████████████████████████████▌                 | 1512/1948 [40:59<08:33,  1.18s/it]cv 값 : 0.32076914504946547람다 값 : 0.5662201498916258



![png](output_52_4538.png)


     78%|████████████████████████████████████████████████████████████▌                 | 1513/1948 [41:01<08:35,  1.19s/it]cv 값 : 1.1841997767790682람다 값 : 0.19356183955792847



![png](output_52_4541.png)


     78%|████████████████████████████████████████████████████████████▌                 | 1514/1948 [41:02<08:22,  1.16s/it]cv 값 : 0.3101331522143766람다 값 : 1.0061287744278447



![png](output_52_4544.png)


     78%|████████████████████████████████████████████████████████████▋                 | 1515/1948 [41:03<07:52,  1.09s/it]cv 값 : 0.1764414166113332람다 값 : 1.5963496249484777



![png](output_52_4547.png)


     78%|████████████████████████████████████████████████████████████▋                 | 1516/1948 [41:04<08:25,  1.17s/it]cv 값 : 0.2766705398993251람다 값 : 1.810450340933979



![png](output_52_4550.png)


     78%|████████████████████████████████████████████████████████████▋                 | 1517/1948 [41:05<08:37,  1.20s/it]cv 값 : 1.3172494605477048람다 값 : 0.24287122348151274



![png](output_52_4553.png)


     78%|████████████████████████████████████████████████████████████▊                 | 1518/1948 [41:06<08:26,  1.18s/it]cv 값 : 0.43885419283275795람다 값 : 0.8887975579991615



![png](output_52_4556.png)


     78%|████████████████████████████████████████████████████████████▊                 | 1519/1948 [41:07<07:59,  1.12s/it]cv 값 : 0.45204218109045907람다 값 : 0.5038100264926689



![png](output_52_4559.png)


     78%|████████████████████████████████████████████████████████████▊                 | 1520/1948 [41:08<07:41,  1.08s/it]cv 값 : 0.4971891380598233람다 값 : 0.5801284921860449



![png](output_52_4562.png)


     78%|████████████████████████████████████████████████████████████▉                 | 1521/1948 [41:10<08:15,  1.16s/it]cv 값 : 0.3968195993430472람다 값 : 1.8245051383142459



![png](output_52_4565.png)


     78%|████████████████████████████████████████████████████████████▉                 | 1522/1948 [41:11<08:31,  1.20s/it]cv 값 : 0.3121377522008033람다 값 : 1.4625924260635002



![png](output_52_4568.png)


     78%|████████████████████████████████████████████████████████████▉                 | 1523/1948 [41:12<08:22,  1.18s/it]cv 값 : 0.28994925591525983람다 값 : 0.5616303815414948



![png](output_52_4571.png)


     78%|█████████████████████████████████████████████████████████████                 | 1524/1948 [41:13<08:11,  1.16s/it]cv 값 : 0.3248369178629786람다 값 : 1.3023484279042143



![png](output_52_4574.png)


     78%|█████████████████████████████████████████████████████████████                 | 1525/1948 [41:14<08:20,  1.18s/it]cv 값 : 0.49270344251023296람다 값 : 0.4118133159053879



![png](output_52_4577.png)


     78%|█████████████████████████████████████████████████████████████                 | 1526/1948 [41:16<08:12,  1.17s/it]cv 값 : 1.297529742454523람다 값 : -0.05277973659416808



![png](output_52_4580.png)


     78%|█████████████████████████████████████████████████████████████▏                | 1527/1948 [41:17<08:14,  1.18s/it]cv 값 : 0.3326293506894562람다 값 : 1.3175420048154292



![png](output_52_4583.png)


     78%|█████████████████████████████████████████████████████████████▏                | 1528/1948 [41:18<08:07,  1.16s/it]cv 값 : 0.2827327199547339람다 값 : 1.3134661672263575



![png](output_52_4586.png)


     78%|█████████████████████████████████████████████████████████████▏                | 1529/1948 [41:19<08:21,  1.20s/it]cv 값 : 1.1774830084603318람다 값 : 0.037046357291758224



![png](output_52_4589.png)


     79%|█████████████████████████████████████████████████████████████▎                | 1530/1948 [41:20<08:06,  1.16s/it]cv 값 : 0.4263858327379007람다 값 : 0.5126591918624221



![png](output_52_4592.png)


     79%|█████████████████████████████████████████████████████████████▎                | 1531/1948 [41:21<07:51,  1.13s/it]cv 값 : 0.3388981584308577람다 값 : 1.0876099378534185



![png](output_52_4595.png)


     79%|█████████████████████████████████████████████████████████████▎                | 1532/1948 [41:22<07:40,  1.11s/it]cv 값 : 0.5610586204670196람다 값 : -0.07601935784221879



![png](output_52_4598.png)


     79%|█████████████████████████████████████████████████████████████▍                | 1533/1948 [41:24<08:16,  1.20s/it]cv 값 : 0.37703984605507046람다 값 : -0.08930246143576165



![png](output_52_4601.png)


     79%|█████████████████████████████████████████████████████████████▍                | 1534/1948 [41:25<08:34,  1.24s/it]cv 값 : 0.2229605417038273람다 값 : -0.36417285871281285



![png](output_52_4604.png)


     79%|█████████████████████████████████████████████████████████████▍                | 1535/1948 [41:26<08:15,  1.20s/it]cv 값 : 0.4043565343449401람다 값 : 1.0781880573801823



![png](output_52_4607.png)


     79%|█████████████████████████████████████████████████████████████▌                | 1536/1948 [41:27<07:59,  1.16s/it]cv 값 : 0.20309220050416735람다 값 : 1.7733240028217832



![png](output_52_4610.png)


     79%|█████████████████████████████████████████████████████████████▌                | 1537/1948 [41:28<07:45,  1.13s/it]cv 값 : 0.6951869852981518람다 값 : 0.2874896438543178



![png](output_52_4613.png)


     79%|█████████████████████████████████████████████████████████████▌                | 1538/1948 [41:30<08:07,  1.19s/it]cv 값 : 0.37520897542014686람다 값 : 0.14140253830762253



![png](output_52_4616.png)


     79%|█████████████████████████████████████████████████████████████▌                | 1539/1948 [41:31<08:02,  1.18s/it]cv 값 : 0.6070358415002634람다 값 : 0.4968897953003326



![png](output_52_4619.png)


     79%|█████████████████████████████████████████████████████████████▋                | 1540/1948 [41:32<07:49,  1.15s/it]cv 값 : 0.3840789422247708람다 값 : -0.5185197264584743



![png](output_52_4622.png)


     79%|█████████████████████████████████████████████████████████████▋                | 1541/1948 [41:33<08:32,  1.26s/it]cv 값 : 0.27363341814674386람다 값 : 1.4611315052806346



![png](output_52_4625.png)


     79%|█████████████████████████████████████████████████████████████▋                | 1542/1948 [41:35<09:35,  1.42s/it]cv 값 : 1.0472921010362257람다 값 : 0.32183457424584233



![png](output_52_4628.png)


     79%|█████████████████████████████████████████████████████████████▊                | 1543/1948 [41:37<10:04,  1.49s/it]cv 값 : 0.37508323211683786람다 값 : 0.8518181074548467



![png](output_52_4631.png)


     79%|█████████████████████████████████████████████████████████████▊                | 1544/1948 [41:39<11:08,  1.65s/it]cv 값 : 0.300443652664468람다 값 : 0.901133392686544



![png](output_52_4634.png)


     79%|█████████████████████████████████████████████████████████████▊                | 1545/1948 [41:41<11:31,  1.72s/it]cv 값 : 0.9961877549303784람다 값 : 0.03316438851841872



![png](output_52_4637.png)


     79%|█████████████████████████████████████████████████████████████▉                | 1546/1948 [41:43<11:38,  1.74s/it]cv 값 : 0.5446213041612462람다 값 : -0.10426294973077986



![png](output_52_4640.png)


     79%|█████████████████████████████████████████████████████████████▉                | 1547/1948 [41:45<12:03,  1.80s/it]cv 값 : 0.20738188037917954람다 값 : 1.665783430219685



![png](output_52_4643.png)


     79%|█████████████████████████████████████████████████████████████▉                | 1548/1948 [41:47<12:28,  1.87s/it]cv 값 : 0.49188916121702675람다 값 : 0.3545692497547035



![png](output_52_4646.png)


     80%|██████████████████████████████████████████████████████████████                | 1549/1948 [41:48<12:08,  1.82s/it]cv 값 : 0.21143864870438706람다 값 : -0.09517482061686616



![png](output_52_4649.png)


     80%|██████████████████████████████████████████████████████████████                | 1550/1948 [41:50<12:13,  1.84s/it]cv 값 : 0.6437359237470501람다 값 : 0.4847504466232066



![png](output_52_4652.png)


     80%|██████████████████████████████████████████████████████████████                | 1551/1948 [41:52<12:31,  1.89s/it]cv 값 : 0.7797790106945722람다 값 : 0.22262595712396382



![png](output_52_4655.png)


     80%|██████████████████████████████████████████████████████████████▏               | 1552/1948 [41:54<12:16,  1.86s/it]cv 값 : 0.6254862179552834람다 값 : 0.5362535073580296



![png](output_52_4658.png)


     80%|██████████████████████████████████████████████████████████████▏               | 1553/1948 [41:56<12:00,  1.82s/it]cv 값 : 0.18623340349252543람다 값 : 1.7914242984689455



![png](output_52_4661.png)


     80%|██████████████████████████████████████████████████████████████▏               | 1554/1948 [41:58<12:01,  1.83s/it]cv 값 : 0.9656649777035693람다 값 : 0.28156911290154846



![png](output_52_4664.png)


     80%|██████████████████████████████████████████████████████████████▎               | 1555/1948 [41:59<11:11,  1.71s/it]cv 값 : 0.3700680814083549람다 값 : 0.40067204742969487



![png](output_52_4667.png)


     80%|██████████████████████████████████████████████████████████████▎               | 1556/1948 [42:01<11:29,  1.76s/it]cv 값 : 0.1537731932170605람다 값 : 0.01485102694588962



![png](output_52_4670.png)


     80%|██████████████████████████████████████████████████████████████▎               | 1557/1948 [42:02<10:26,  1.60s/it]cv 값 : 0.29058424123980503람다 값 : 0.34076777165692446



![png](output_52_4673.png)


     80%|██████████████████████████████████████████████████████████████▍               | 1558/1948 [42:03<09:45,  1.50s/it]cv 값 : 0.9583455242239629람다 값 : 0.2073433994603035



![png](output_52_4676.png)


     80%|██████████████████████████████████████████████████████████████▍               | 1559/1948 [42:05<09:11,  1.42s/it]cv 값 : 0.8226375678129734람다 값 : 0.24166139685963417



![png](output_52_4679.png)


     80%|██████████████████████████████████████████████████████████████▍               | 1560/1948 [42:07<10:19,  1.60s/it]cv 값 : 0.2274288455779744람다 값 : 0.8626879451728087



![png](output_52_4682.png)


     80%|██████████████████████████████████████████████████████████████▌               | 1561/1948 [42:09<11:08,  1.73s/it]cv 값 : 0.562539245074457람다 값 : 0.47970355057895997



![png](output_52_4685.png)


     80%|██████████████████████████████████████████████████████████████▌               | 1562/1948 [42:10<10:42,  1.67s/it]cv 값 : 0.46992337645662047람다 값 : 0.21475787898425922



![png](output_52_4688.png)


     80%|██████████████████████████████████████████████████████████████▌               | 1563/1948 [42:11<10:02,  1.57s/it]cv 값 : 0.41988006044620885람다 값 : 0.4888871118940049



![png](output_52_4691.png)


     80%|██████████████████████████████████████████████████████████████▌               | 1564/1948 [42:13<09:13,  1.44s/it]cv 값 : 0.29322125726932013람다 값 : 1.6648594468267122



![png](output_52_4694.png)


     80%|██████████████████████████████████████████████████████████████▋               | 1565/1948 [42:14<08:58,  1.41s/it]cv 값 : 0.5083548786409362람다 값 : 0.33735367816416406



![png](output_52_4697.png)


     80%|██████████████████████████████████████████████████████████████▋               | 1566/1948 [42:15<08:48,  1.38s/it]cv 값 : 0.37432877660045843람다 값 : 0.583619581586373



![png](output_52_4700.png)


     80%|██████████████████████████████████████████████████████████████▋               | 1567/1948 [42:17<08:27,  1.33s/it]cv 값 : 0.2883602625250222람다 값 : 1.253898453825706



![png](output_52_4703.png)


     80%|██████████████████████████████████████████████████████████████▊               | 1568/1948 [42:18<08:27,  1.34s/it]cv 값 : 0.5786882258594562람다 값 : 0.6561994211722102



![png](output_52_4706.png)


     81%|██████████████████████████████████████████████████████████████▊               | 1569/1948 [42:19<08:13,  1.30s/it]cv 값 : 0.5940887832124565람다 값 : 0.34327868011862617



![png](output_52_4709.png)


     81%|██████████████████████████████████████████████████████████████▊               | 1570/1948 [42:20<08:13,  1.31s/it]cv 값 : 0.6210395107439934람다 값 : 0.48845344163733084



![png](output_52_4712.png)


     81%|██████████████████████████████████████████████████████████████▉               | 1571/1948 [42:22<07:57,  1.27s/it]cv 값 : 1.4932652906990784람다 값 : -0.3069429402340417



![png](output_52_4715.png)


     81%|██████████████████████████████████████████████████████████████▉               | 1572/1948 [42:23<07:38,  1.22s/it]cv 값 : 0.3158219574671278람다 값 : 0.9953821276622616



![png](output_52_4718.png)


     81%|██████████████████████████████████████████████████████████████▉               | 1573/1948 [42:24<07:19,  1.17s/it]cv 값 : 0.36690726035702226람다 값 : 0.6126647889450135



![png](output_52_4721.png)


     81%|███████████████████████████████████████████████████████████████               | 1574/1948 [42:25<07:28,  1.20s/it]cv 값 : 0.4653995375908262람다 값 : 1.0351668767955473



![png](output_52_4724.png)


     81%|███████████████████████████████████████████████████████████████               | 1575/1948 [42:26<07:26,  1.20s/it]cv 값 : 0.5173851488841231람다 값 : 0.6799557593561532



![png](output_52_4727.png)


     81%|███████████████████████████████████████████████████████████████               | 1576/1948 [42:27<07:10,  1.16s/it]cv 값 : 0.6551073377878912람다 값 : 0.2571847701425429



![png](output_52_4730.png)


     81%|███████████████████████████████████████████████████████████████▏              | 1577/1948 [42:28<06:58,  1.13s/it]cv 값 : 0.43167271947989877람다 값 : 1.0427750346160303



![png](output_52_4733.png)


     81%|███████████████████████████████████████████████████████████████▏              | 1578/1948 [42:30<07:14,  1.18s/it]cv 값 : 0.33593902214097227람다 값 : 0.9677180304451101



![png](output_52_4736.png)


     81%|███████████████████████████████████████████████████████████████▏              | 1579/1948 [42:31<07:11,  1.17s/it]cv 값 : 0.4552016928047441람다 값 : 0.6666125492645266



![png](output_52_4739.png)


     81%|███████████████████████████████████████████████████████████████▎              | 1580/1948 [42:32<07:01,  1.15s/it]cv 값 : 0.39287893575967936람다 값 : 0.9156787920559578



![png](output_52_4742.png)


     81%|███████████████████████████████████████████████████████████████▎              | 1581/1948 [42:33<06:48,  1.11s/it]cv 값 : 0.345984744604745람다 값 : 1.3037690961635666



![png](output_52_4745.png)


     81%|███████████████████████████████████████████████████████████████▎              | 1582/1948 [42:34<06:56,  1.14s/it]cv 값 : 0.3153903806451462람다 값 : 1.4355493879415304



![png](output_52_4748.png)


     81%|███████████████████████████████████████████████████████████████▍              | 1583/1948 [42:36<07:33,  1.24s/it]cv 값 : 0.3281686099339116람다 값 : 1.2146177897595107



![png](output_52_4751.png)


     81%|███████████████████████████████████████████████████████████████▍              | 1584/1948 [42:37<07:14,  1.19s/it]cv 값 : 0.378810710035061람다 값 : 1.2783741019655748



![png](output_52_4754.png)


     81%|███████████████████████████████████████████████████████████████▍              | 1585/1948 [42:38<07:07,  1.18s/it]cv 값 : 0.2373670427866507람다 값 : 2.1124467774455544



![png](output_52_4757.png)


     81%|███████████████████████████████████████████████████████████████▌              | 1586/1948 [42:39<06:58,  1.16s/it]cv 값 : 0.8606625765876015람다 값 : 0.31064727004015424



![png](output_52_4760.png)


     81%|███████████████████████████████████████████████████████████████▌              | 1587/1948 [42:40<07:12,  1.20s/it]cv 값 : 0.4672044311878127람다 값 : -1.0531220007227065



![png](output_52_4763.png)


     82%|███████████████████████████████████████████████████████████████▌              | 1588/1948 [42:41<07:10,  1.19s/it]cv 값 : 0.306264563823109람다 값 : 0.7297314628807311



![png](output_52_4766.png)


     82%|███████████████████████████████████████████████████████████████▋              | 1589/1948 [42:42<06:40,  1.12s/it]cv 값 : 0.1790816059405791람다 값 : 0.8218933625657457



![png](output_52_4769.png)


     82%|███████████████████████████████████████████████████████████████▋              | 1590/1948 [42:43<06:33,  1.10s/it]cv 값 : 0.29556930491950795람다 값 : 1.0951303043938505



![png](output_52_4772.png)


     82%|███████████████████████████████████████████████████████████████▋              | 1591/1948 [42:44<06:31,  1.10s/it]cv 값 : 0.15272776119423365람다 값 : -0.9573470000910188



![png](output_52_4775.png)


     82%|███████████████████████████████████████████████████████████████▋              | 1592/1948 [42:46<07:00,  1.18s/it]cv 값 : 0.24434116438975834람다 값 : 0.48050590295350204



![png](output_52_4778.png)


     82%|███████████████████████████████████████████████████████████████▊              | 1593/1948 [42:47<07:22,  1.25s/it]cv 값 : 0.5877385032145688람다 값 : 0.45364279767384086



![png](output_52_4781.png)


     82%|███████████████████████████████████████████████████████████████▊              | 1594/1948 [42:48<06:58,  1.18s/it]cv 값 : 0.4041430695457178람다 값 : 0.60812885278866



![png](output_52_4784.png)


     82%|███████████████████████████████████████████████████████████████▊              | 1595/1948 [42:49<06:43,  1.14s/it]cv 값 : 0.1851386356607079람다 값 : 0.0811213915276739



![png](output_52_4787.png)


     82%|███████████████████████████████████████████████████████████████▉              | 1596/1948 [42:51<07:00,  1.20s/it]cv 값 : 0.9391807734281481람다 값 : 0.3478306795362123



![png](output_52_4790.png)


     82%|███████████████████████████████████████████████████████████████▉              | 1597/1948 [42:52<06:59,  1.20s/it]cv 값 : 0.47618314627415187람다 값 : 0.6590456810042037



![png](output_52_4793.png)


     82%|███████████████████████████████████████████████████████████████▉              | 1598/1948 [42:53<06:53,  1.18s/it]cv 값 : 0.18389038677184738람다 값 : -0.6211297815637092



![png](output_52_4796.png)


     82%|████████████████████████████████████████████████████████████████              | 1599/1948 [42:54<06:33,  1.13s/it]cv 값 : 0.30592267111703064람다 값 : -0.24487627443249665



![png](output_52_4799.png)


     82%|████████████████████████████████████████████████████████████████              | 1600/1948 [42:55<06:42,  1.16s/it]cv 값 : 0.6757281948596787람다 값 : 0.5394182536616258



![png](output_52_4802.png)


     82%|████████████████████████████████████████████████████████████████              | 1601/1948 [42:56<06:35,  1.14s/it]cv 값 : 0.35554694355128186람다 값 : 1.250859573815358



![png](output_52_4805.png)


     82%|████████████████████████████████████████████████████████████████▏             | 1602/1948 [42:57<06:24,  1.11s/it]cv 값 : 0.7175074173122925람다 값 : 0.3394120439679146



![png](output_52_4808.png)


     82%|████████████████████████████████████████████████████████████████▏             | 1603/1948 [42:58<06:04,  1.06s/it]cv 값 : 0.25863506267048014람다 값 : 0.5401031889066681



![png](output_52_4811.png)


     82%|████████████████████████████████████████████████████████████████▏             | 1604/1948 [42:59<06:07,  1.07s/it]cv 값 : 0.3708535813567908람다 값 : -0.1324891139708882



![png](output_52_4814.png)


     82%|████████████████████████████████████████████████████████████████▎             | 1605/1948 [43:01<06:26,  1.13s/it]cv 값 : 1.7960605403166792람다 값 : 0.16499949355565657



![png](output_52_4817.png)


     82%|████████████████████████████████████████████████████████████████▎             | 1606/1948 [43:02<06:14,  1.10s/it]cv 값 : 0.15174565253226288람다 값 : 0.5476132491415252



![png](output_52_4820.png)


     82%|████████████████████████████████████████████████████████████████▎             | 1607/1948 [43:03<05:57,  1.05s/it]cv 값 : 0.32608164396412026람다 값 : 0.42673722343756987



![png](output_52_4823.png)


     83%|████████████████████████████████████████████████████████████████▍             | 1608/1948 [43:04<05:56,  1.05s/it]cv 값 : 0.4593524811763052람다 값 : 0.038032231869062565



![png](output_52_4826.png)


     83%|████████████████████████████████████████████████████████████████▍             | 1609/1948 [43:05<05:38,  1.00it/s]cv 값 : 0.24108580722828266람다 값 : 1.5697498912748944



![png](output_52_4829.png)


     83%|████████████████████████████████████████████████████████████████▍             | 1610/1948 [43:06<05:55,  1.05s/it]cv 값 : 0.5261344473638336람다 값 : 0.2554850838605644



![png](output_52_4832.png)


     83%|████████████████████████████████████████████████████████████████▌             | 1611/1948 [43:07<05:52,  1.05s/it]cv 값 : 0.34078542472232304람다 값 : 0.318910631656777



![png](output_52_4835.png)


     83%|████████████████████████████████████████████████████████████████▌             | 1612/1948 [43:08<05:44,  1.02s/it]cv 값 : 0.30700188388398303람다 값 : 0.9861688741047184



![png](output_52_4838.png)


     83%|████████████████████████████████████████████████████████████████▌             | 1613/1948 [43:09<05:29,  1.02it/s]cv 값 : 0.252560093710856람다 값 : 1.6557275844092572



![png](output_52_4841.png)


     83%|████████████████████████████████████████████████████████████████▋             | 1614/1948 [43:10<06:21,  1.14s/it]cv 값 : 0.4972410055251865람다 값 : 0.5431446906156351



![png](output_52_4844.png)


     83%|████████████████████████████████████████████████████████████████▋             | 1615/1948 [43:11<06:12,  1.12s/it]cv 값 : 0.49471309753617865람다 값 : 0.9428360839436285



![png](output_52_4847.png)


     83%|████████████████████████████████████████████████████████████████▋             | 1616/1948 [43:12<06:08,  1.11s/it]cv 값 : 0.6849823125367469람다 값 : 0.5802711797076994



![png](output_52_4850.png)


     83%|████████████████████████████████████████████████████████████████▋             | 1617/1948 [43:13<05:52,  1.06s/it]cv 값 : 0.3594220441280174람다 값 : 0.9607837346409813



![png](output_52_4853.png)


     83%|████████████████████████████████████████████████████████████████▊             | 1618/1948 [43:14<05:42,  1.04s/it]cv 값 : 0.2326301203119343람다 값 : 0.9782106805452904



![png](output_52_4856.png)


     83%|████████████████████████████████████████████████████████████████▊             | 1619/1948 [43:16<06:17,  1.15s/it]cv 값 : 0.1436370768834374람다 값 : 2.3989030135101923



![png](output_52_4859.png)


     83%|████████████████████████████████████████████████████████████████▊             | 1620/1948 [43:17<06:15,  1.14s/it]cv 값 : 0.4782659294368965람다 값 : 0.1279109634786372



![png](output_52_4862.png)


     83%|████████████████████████████████████████████████████████████████▉             | 1621/1948 [43:18<06:03,  1.11s/it]cv 값 : 0.3046674695757197람다 값 : 0.9588705849156182



![png](output_52_4865.png)


     83%|████████████████████████████████████████████████████████████████▉             | 1622/1948 [43:19<05:47,  1.06s/it]cv 값 : 0.2311087835306449람다 값 : 1.8714034945734657



![png](output_52_4868.png)


     83%|████████████████████████████████████████████████████████████████▉             | 1623/1948 [43:20<06:09,  1.14s/it]cv 값 : 0.18011686608068336람다 값 : 2.2077212676856797



![png](output_52_4871.png)


     83%|█████████████████████████████████████████████████████████████████             | 1624/1948 [43:21<05:59,  1.11s/it]cv 값 : 0.6378086106126489람다 값 : 0.16977146614770272



![png](output_52_4874.png)


     83%|█████████████████████████████████████████████████████████████████             | 1625/1948 [43:22<05:53,  1.09s/it]cv 값 : 0.22033773068390086람다 값 : 1.5560924954169242



![png](output_52_4877.png)


     83%|█████████████████████████████████████████████████████████████████             | 1626/1948 [43:23<05:49,  1.09s/it]cv 값 : 0.2887515608441334람다 값 : 1.5785220646571754



![png](output_52_4880.png)


     84%|█████████████████████████████████████████████████████████████████▏            | 1627/1948 [43:25<06:26,  1.20s/it]cv 값 : 0.37883821597541084람다 값 : 0.5776210882361433



![png](output_52_4883.png)


     84%|█████████████████████████████████████████████████████████████████▏            | 1628/1948 [43:26<06:09,  1.16s/it]cv 값 : 0.4319547370476945람다 값 : 0.6964139831467627



![png](output_52_4886.png)


     84%|█████████████████████████████████████████████████████████████████▏            | 1629/1948 [43:27<05:55,  1.12s/it]cv 값 : 0.21746669025726637람다 값 : 2.880537982057863



![png](output_52_4889.png)


     84%|█████████████████████████████████████████████████████████████████▎            | 1630/1948 [43:28<06:00,  1.13s/it]cv 값 : 0.33440687656825546람다 값 : 1.1448298918508908



![png](output_52_4892.png)


     84%|█████████████████████████████████████████████████████████████████▎            | 1631/1948 [43:29<06:08,  1.16s/it]cv 값 : 0.8571064057179172람다 값 : 0.36444506017191564



![png](output_52_4895.png)


     84%|█████████████████████████████████████████████████████████████████▎            | 1632/1948 [43:30<06:22,  1.21s/it]cv 값 : 0.4034222863374427람다 값 : 0.9508454191199073



![png](output_52_4898.png)


     84%|█████████████████████████████████████████████████████████████████▍            | 1633/1948 [43:32<06:02,  1.15s/it]cv 값 : 0.46555869214966805람다 값 : 1.0411794534340633



![png](output_52_4901.png)


     84%|█████████████████████████████████████████████████████████████████▍            | 1634/1948 [43:33<05:59,  1.15s/it]cv 값 : 0.36633685153173196람다 값 : 1.207838467906563



![png](output_52_4904.png)


     84%|█████████████████████████████████████████████████████████████████▍            | 1635/1948 [43:34<05:48,  1.11s/it]cv 값 : 0.36970029428703977람다 값 : 0.9929122899195187



![png](output_52_4907.png)


     84%|█████████████████████████████████████████████████████████████████▌            | 1636/1948 [43:35<05:48,  1.12s/it]cv 값 : 1.042368409116366람다 값 : 0.2879102538083953



![png](output_52_4910.png)


     84%|█████████████████████████████████████████████████████████████████▌            | 1637/1948 [43:36<06:04,  1.17s/it]cv 값 : 0.28464376609415437람다 값 : 1.4965542332238662



![png](output_52_4913.png)


     84%|█████████████████████████████████████████████████████████████████▌            | 1638/1948 [43:37<05:57,  1.15s/it]cv 값 : 0.611356347269223람다 값 : 0.6647050591537392



![png](output_52_4916.png)


     84%|█████████████████████████████████████████████████████████████████▋            | 1639/1948 [43:38<05:48,  1.13s/it]cv 값 : 0.7342427275917817람다 값 : -0.21705669819177448



![png](output_52_4919.png)


     84%|█████████████████████████████████████████████████████████████████▋            | 1640/1948 [43:40<05:58,  1.16s/it]cv 값 : 0.5460565329172681람다 값 : 0.33540076044061556



![png](output_52_4922.png)


     84%|█████████████████████████████████████████████████████████████████▋            | 1641/1948 [43:41<05:51,  1.15s/it]cv 값 : 0.5659455274128331람다 값 : 0.3427917728799501



![png](output_52_4925.png)


     84%|█████████████████████████████████████████████████████████████████▋            | 1642/1948 [43:42<05:40,  1.11s/it]cv 값 : 0.5732846650566752람다 값 : -0.634613084367873



![png](output_52_4928.png)


     84%|█████████████████████████████████████████████████████████████████▊            | 1643/1948 [43:43<05:37,  1.11s/it]cv 값 : 0.18963875329264812람다 값 : 0.449331801628371



![png](output_52_4931.png)


     84%|█████████████████████████████████████████████████████████████████▊            | 1644/1948 [43:44<05:24,  1.07s/it]cv 값 : 1.4238898761359116람다 값 : 0.26334208486373345



![png](output_52_4934.png)


     84%|█████████████████████████████████████████████████████████████████▊            | 1645/1948 [43:45<05:38,  1.12s/it]cv 값 : 0.38953505454032433람다 값 : 0.27585213518238516



![png](output_52_4937.png)


     84%|█████████████████████████████████████████████████████████████████▉            | 1646/1948 [43:46<05:34,  1.11s/it]cv 값 : 0.5473150151056899람다 값 : -0.0032308938505955774



![png](output_52_4940.png)


     85%|█████████████████████████████████████████████████████████████████▉            | 1647/1948 [43:47<05:31,  1.10s/it]cv 값 : 0.2584137981898864람다 값 : -1.666377249627809



![png](output_52_4943.png)


     85%|█████████████████████████████████████████████████████████████████▉            | 1648/1948 [43:48<05:38,  1.13s/it]cv 값 : 0.737984568614939람다 값 : 0.28710946103549495



![png](output_52_4946.png)


     85%|██████████████████████████████████████████████████████████████████            | 1649/1948 [43:49<05:36,  1.13s/it]cv 값 : 0.28089177843505136람다 값 : 0.07631385613960258



![png](output_52_4949.png)


     85%|██████████████████████████████████████████████████████████████████            | 1650/1948 [43:51<05:52,  1.18s/it]cv 값 : 0.2731448859153282람다 값 : -0.05846787794768422



![png](output_52_4952.png)


     85%|██████████████████████████████████████████████████████████████████            | 1651/1948 [43:52<05:48,  1.17s/it]cv 값 : 0.15919385813294581람다 값 : -0.9823570100055142



![png](output_52_4955.png)


     85%|██████████████████████████████████████████████████████████████████▏           | 1652/1948 [43:53<05:41,  1.15s/it]cv 값 : 0.8493313664416546람다 값 : -0.799646640360531



![png](output_52_4958.png)


     85%|██████████████████████████████████████████████████████████████████▏           | 1653/1948 [43:54<05:47,  1.18s/it]cv 값 : 0.21490111415815266람다 값 : -0.3278265070296575



![png](output_52_4961.png)


     85%|██████████████████████████████████████████████████████████████████▏           | 1654/1948 [43:56<06:00,  1.23s/it]cv 값 : 0.427637209834212람다 값 : 0.6510383678996066



![png](output_52_4964.png)


     85%|██████████████████████████████████████████████████████████████████▎           | 1655/1948 [43:57<05:48,  1.19s/it]cv 값 : 0.22795489046996362람다 값 : -0.686550827089378



![png](output_52_4967.png)


     85%|██████████████████████████████████████████████████████████████████▎           | 1656/1948 [43:58<05:38,  1.16s/it]cv 값 : 0.33855071445547674람다 값 : 0.9099977283331655



![png](output_52_4970.png)


     85%|██████████████████████████████████████████████████████████████████▎           | 1657/1948 [43:59<05:47,  1.19s/it]cv 값 : 0.49635179313234107람다 값 : 0.714673214732832



![png](output_52_4973.png)


     85%|██████████████████████████████████████████████████████████████████▍           | 1658/1948 [44:00<05:45,  1.19s/it]cv 값 : 0.8800253387944944람다 값 : 0.3990101693858922



![png](output_52_4976.png)


     85%|██████████████████████████████████████████████████████████████████▍           | 1659/1948 [44:01<05:46,  1.20s/it]cv 값 : 0.44065700439440136람다 값 : 0.5398132519562087



![png](output_52_4979.png)


     85%|██████████████████████████████████████████████████████████████████▍           | 1660/1948 [44:03<05:32,  1.15s/it]cv 값 : 0.25354341856097296람다 값 : 1.1379758852623982



![png](output_52_4982.png)


     85%|██████████████████████████████████████████████████████████████████▌           | 1661/1948 [44:04<05:38,  1.18s/it]cv 값 : 0.3821654333473837람다 값 : 0.32616768928416845



![png](output_52_4985.png)


     85%|██████████████████████████████████████████████████████████████████▌           | 1662/1948 [44:05<05:24,  1.13s/it]cv 값 : 0.8356639276064692람다 값 : 0.38671531897376904



![png](output_52_4988.png)


     85%|██████████████████████████████████████████████████████████████████▌           | 1663/1948 [44:06<05:38,  1.19s/it]cv 값 : 0.5286878010875824람다 값 : -0.9646080129877178



![png](output_52_4991.png)


     85%|██████████████████████████████████████████████████████████████████▋           | 1664/1948 [44:07<05:37,  1.19s/it]cv 값 : 0.3263211544338142람다 값 : 1.0436494015674471



![png](output_52_4994.png)


     85%|██████████████████████████████████████████████████████████████████▋           | 1665/1948 [44:08<05:28,  1.16s/it]cv 값 : 0.6896369528830787람다 값 : 0.339773912447447



![png](output_52_4997.png)


     86%|██████████████████████████████████████████████████████████████████▋           | 1666/1948 [44:09<05:22,  1.14s/it]cv 값 : 0.3229208478467914람다 값 : 1.2175169378795467



![png](output_52_5000.png)


     86%|██████████████████████████████████████████████████████████████████▋           | 1667/1948 [44:11<05:24,  1.15s/it]cv 값 : 0.37953682196534844람다 값 : 1.0127663596791308



![png](output_52_5003.png)


     86%|██████████████████████████████████████████████████████████████████▊           | 1668/1948 [44:12<05:26,  1.16s/it]cv 값 : 0.19101076085869556람다 값 : 1.3962097843859764



![png](output_52_5006.png)


     86%|██████████████████████████████████████████████████████████████████▊           | 1669/1948 [44:13<05:24,  1.16s/it]cv 값 : 0.508086358422876람다 값 : 0.24375379228528654



![png](output_52_5009.png)


     86%|██████████████████████████████████████████████████████████████████▊           | 1670/1948 [44:14<05:14,  1.13s/it]cv 값 : 0.38564635146398435람다 값 : 0.3844382750735666



![png](output_52_5012.png)


     86%|██████████████████████████████████████████████████████████████████▉           | 1671/1948 [44:15<05:06,  1.11s/it]cv 값 : 0.27692409312281074람다 값 : -0.25756758545906777



![png](output_52_5015.png)


     86%|██████████████████████████████████████████████████████████████████▉           | 1672/1948 [44:16<05:27,  1.19s/it]cv 값 : 0.5454798805183432람다 값 : 0.17362123789681394



![png](output_52_5018.png)


     86%|██████████████████████████████████████████████████████████████████▉           | 1673/1948 [44:17<05:09,  1.13s/it]cv 값 : 0.5000041001571756람다 값 : -0.18808887074900607



![png](output_52_5021.png)


     86%|███████████████████████████████████████████████████████████████████           | 1674/1948 [44:18<04:57,  1.09s/it]cv 값 : 0.39523429260036474람다 값 : 0.0699261228940811



![png](output_52_5024.png)


     86%|███████████████████████████████████████████████████████████████████           | 1675/1948 [44:20<04:53,  1.07s/it]cv 값 : 0.27253291428692406람다 값 : 1.0508260313729036



![png](output_52_5027.png)


     86%|███████████████████████████████████████████████████████████████████           | 1676/1948 [44:21<04:51,  1.07s/it]cv 값 : 0.2616259944898449람다 값 : 0.7372131831780069



![png](output_52_5030.png)


     86%|███████████████████████████████████████████████████████████████████▏          | 1677/1948 [44:22<04:52,  1.08s/it]cv 값 : 0.6689476536106557람다 값 : 0.4251148579065988



![png](output_52_5033.png)


     86%|███████████████████████████████████████████████████████████████████▏          | 1678/1948 [44:23<04:53,  1.09s/it]cv 값 : 0.47110291584206176람다 값 : 0.6757843524926294



![png](output_52_5036.png)


     86%|███████████████████████████████████████████████████████████████████▏          | 1679/1948 [44:24<04:58,  1.11s/it]cv 값 : 0.2670889220100174람다 값 : 1.1433520903501553



![png](output_52_5039.png)


     86%|███████████████████████████████████████████████████████████████████▎          | 1680/1948 [44:25<05:10,  1.16s/it]cv 값 : 0.32205177928712037람다 값 : -0.3496950843038841



![png](output_52_5042.png)


     86%|███████████████████████████████████████████████████████████████████▎          | 1681/1948 [44:27<05:28,  1.23s/it]cv 값 : 0.6265002141573163람다 값 : 0.2173340479549894



![png](output_52_5045.png)


     86%|███████████████████████████████████████████████████████████████████▎          | 1682/1948 [44:28<05:20,  1.20s/it]cv 값 : 0.41278752811529484람다 값 : 1.053765351952674



![png](output_52_5048.png)


     86%|███████████████████████████████████████████████████████████████████▍          | 1683/1948 [44:29<05:13,  1.18s/it]cv 값 : 0.2717038986499829람다 값 : 0.9341636325028146



![png](output_52_5051.png)


     86%|███████████████████████████████████████████████████████████████████▍          | 1684/1948 [44:31<05:51,  1.33s/it]cv 값 : 0.580238866720491람다 값 : 0.38602870649679977



![png](output_52_5054.png)


     86%|███████████████████████████████████████████████████████████████████▍          | 1685/1948 [44:33<06:44,  1.54s/it]cv 값 : 0.36312242252351157람다 값 : 0.04103063612149721



![png](output_52_5057.png)


     87%|███████████████████████████████████████████████████████████████████▌          | 1686/1948 [44:34<06:23,  1.46s/it]cv 값 : 0.38666053953540214람다 값 : 1.1711664004892965



![png](output_52_5060.png)


     87%|███████████████████████████████████████████████████████████████████▌          | 1687/1948 [44:35<05:55,  1.36s/it]cv 값 : 0.5344370988389388람다 값 : 0.5554642458562034



![png](output_52_5063.png)


     87%|███████████████████████████████████████████████████████████████████▌          | 1688/1948 [44:36<05:51,  1.35s/it]cv 값 : 0.455239697376773람다 값 : 0.940949175791564



![png](output_52_5066.png)


     87%|███████████████████████████████████████████████████████████████████▋          | 1689/1948 [44:37<05:26,  1.26s/it]cv 값 : 0.5138135766373432람다 값 : 0.661676212062217



![png](output_52_5069.png)


     87%|███████████████████████████████████████████████████████████████████▋          | 1690/1948 [44:39<05:24,  1.26s/it]cv 값 : 1.3060449707956805람다 값 : 0.051686405961618546



![png](output_52_5072.png)


     87%|███████████████████████████████████████████████████████████████████▋          | 1691/1948 [44:40<05:14,  1.22s/it]cv 값 : 0.2513172047546612람다 값 : 1.9055480685220259



![png](output_52_5075.png)


     87%|███████████████████████████████████████████████████████████████████▋          | 1692/1948 [44:41<05:33,  1.30s/it]cv 값 : 0.32493643740921957람다 값 : -0.014513586998869229



![png](output_52_5078.png)


     87%|███████████████████████████████████████████████████████████████████▊          | 1693/1948 [44:43<05:32,  1.30s/it]cv 값 : 0.3346724645481088람다 값 : 0.9008376685393147



![png](output_52_5081.png)


     87%|███████████████████████████████████████████████████████████████████▊          | 1694/1948 [44:44<05:31,  1.31s/it]cv 값 : 0.21232453841010912람다 값 : 0.5947828890464137



![png](output_52_5084.png)


     87%|███████████████████████████████████████████████████████████████████▊          | 1695/1948 [44:45<05:15,  1.25s/it]cv 값 : 0.4038279115183716람다 값 : -0.44938798528170054



![png](output_52_5087.png)


     87%|███████████████████████████████████████████████████████████████████▉          | 1696/1948 [44:46<05:18,  1.26s/it]cv 값 : 0.106206973075563람다 값 : 2.0878153327803566



![png](output_52_5090.png)


     87%|███████████████████████████████████████████████████████████████████▉          | 1697/1948 [44:47<05:00,  1.20s/it]cv 값 : 0.4296454685202064람다 값 : 0.8879534390016558



![png](output_52_5093.png)


     87%|███████████████████████████████████████████████████████████████████▉          | 1698/1948 [44:48<04:49,  1.16s/it]cv 값 : 0.37833180566536667람다 값 : -1.3397650437620625



![png](output_52_5096.png)


     87%|████████████████████████████████████████████████████████████████████          | 1699/1948 [44:50<05:01,  1.21s/it]cv 값 : 1.0373037478043563람다 값 : 0.30968683423235766



![png](output_52_5099.png)


     87%|████████████████████████████████████████████████████████████████████          | 1700/1948 [44:51<05:02,  1.22s/it]cv 값 : 0.3286457382957691람다 값 : 0.5211375371602874



![png](output_52_5102.png)


     87%|████████████████████████████████████████████████████████████████████          | 1701/1948 [44:52<04:58,  1.21s/it]cv 값 : 0.2997749493975892람다 값 : 0.6525520942649433



![png](output_52_5105.png)


     87%|████████████████████████████████████████████████████████████████████▏         | 1702/1948 [44:53<04:48,  1.17s/it]cv 값 : 0.40641134957356106람다 값 : 0.6971096120488768



![png](output_52_5108.png)


     87%|████████████████████████████████████████████████████████████████████▏         | 1703/1948 [44:54<04:42,  1.15s/it]cv 값 : 0.28375931716386726람다 값 : 0.5989272528138295



![png](output_52_5111.png)


     87%|████████████████████████████████████████████████████████████████████▏         | 1704/1948 [44:56<04:50,  1.19s/it]cv 값 : 0.6183610251007471람다 값 : -0.18431073144529173



![png](output_52_5114.png)


     88%|████████████████████████████████████████████████████████████████████▎         | 1705/1948 [44:57<04:41,  1.16s/it]cv 값 : 0.21824869026244598람다 값 : 0.872203965183743



![png](output_52_5117.png)


     88%|████████████████████████████████████████████████████████████████████▎         | 1706/1948 [44:58<04:33,  1.13s/it]cv 값 : 0.48712200029203534람다 값 : 0.02510834654582561



![png](output_52_5120.png)


     88%|████████████████████████████████████████████████████████████████████▎         | 1707/1948 [44:59<04:30,  1.12s/it]cv 값 : 0.7244319263988415람다 값 : 0.36883905777040826



![png](output_52_5123.png)


     88%|████████████████████████████████████████████████████████████████████▍         | 1708/1948 [45:00<04:33,  1.14s/it]cv 값 : 0.45575410850556747람다 값 : -0.35346225634178335



![png](output_52_5126.png)


     88%|████████████████████████████████████████████████████████████████████▍         | 1709/1948 [45:01<04:39,  1.17s/it]cv 값 : 0.30905016378354355람다 값 : 1.6268413577374745



![png](output_52_5129.png)


     88%|████████████████████████████████████████████████████████████████████▍         | 1710/1948 [45:02<04:39,  1.17s/it]cv 값 : 0.3801059278138284람다 값 : 0.36701852607358937



![png](output_52_5132.png)


     88%|████████████████████████████████████████████████████████████████████▌         | 1711/1948 [45:04<04:36,  1.17s/it]cv 값 : 0.3970057455456638람다 값 : -0.1243836699203752



![png](output_52_5135.png)


     88%|████████████████████████████████████████████████████████████████████▌         | 1712/1948 [45:05<04:32,  1.15s/it]cv 값 : 0.2582039555488991람다 값 : 1.5081011325642757



![png](output_52_5138.png)


     88%|████████████████████████████████████████████████████████████████████▌         | 1713/1948 [45:06<04:41,  1.20s/it]cv 값 : 0.32221486159830764람다 값 : 0.9412048375889537



![png](output_52_5141.png)


     88%|████████████████████████████████████████████████████████████████████▋         | 1714/1948 [45:07<04:27,  1.14s/it]cv 값 : 0.2873952594174903람다 값 : 1.4013853829181981



![png](output_52_5144.png)


     88%|████████████████████████████████████████████████████████████████████▋         | 1715/1948 [45:08<04:18,  1.11s/it]cv 값 : 0.24976885603599802람다 값 : -0.470062124191185



![png](output_52_5147.png)


     88%|████████████████████████████████████████████████████████████████████▋         | 1716/1948 [45:09<04:26,  1.15s/it]cv 값 : 0.2374167813534762람다 값 : 2.224119434359244



![png](output_52_5150.png)


     88%|████████████████████████████████████████████████████████████████████▊         | 1717/1948 [45:11<04:29,  1.17s/it]cv 값 : 0.2366334327739184람다 값 : 1.5419281158834857



![png](output_52_5153.png)


     88%|████████████████████████████████████████████████████████████████████▊         | 1718/1948 [45:12<04:29,  1.17s/it]cv 값 : 0.30631389862488434람다 값 : -0.25701385446209646



![png](output_52_5156.png)


     88%|████████████████████████████████████████████████████████████████████▊         | 1719/1948 [45:13<04:26,  1.16s/it]cv 값 : 0.6252061159097756람다 값 : -0.002065276806333693



![png](output_52_5159.png)


     88%|████████████████████████████████████████████████████████████████████▊         | 1720/1948 [45:14<04:14,  1.12s/it]cv 값 : 0.29891708386361276람다 값 : 1.4789680053506002



![png](output_52_5162.png)


     88%|████████████████████████████████████████████████████████████████████▉         | 1721/1948 [45:15<04:24,  1.17s/it]cv 값 : 0.25331873131596877람다 값 : 0.9261516400593223



![png](output_52_5165.png)


     88%|████████████████████████████████████████████████████████████████████▉         | 1722/1948 [45:16<04:19,  1.15s/it]cv 값 : 0.4759286458023573람다 값 : 0.6940100924586117



![png](output_52_5168.png)


     88%|████████████████████████████████████████████████████████████████████▉         | 1723/1948 [45:17<04:18,  1.15s/it]cv 값 : 0.425840616056631람다 값 : 0.8829025161264934



![png](output_52_5171.png)


     89%|█████████████████████████████████████████████████████████████████████         | 1724/1948 [45:18<04:04,  1.09s/it]cv 값 : 0.7169389696613021람다 값 : 0.48937121978908055



![png](output_52_5174.png)


     89%|█████████████████████████████████████████████████████████████████████         | 1725/1948 [45:20<04:06,  1.11s/it]cv 값 : 0.40320665444605575람다 값 : 0.7292679654568188



![png](output_52_5177.png)


     89%|█████████████████████████████████████████████████████████████████████         | 1726/1948 [45:21<04:22,  1.18s/it]cv 값 : 0.26700567865117697람다 값 : 1.526557670090512



![png](output_52_5180.png)


     89%|█████████████████████████████████████████████████████████████████████▏        | 1727/1948 [45:22<04:18,  1.17s/it]cv 값 : 0.8381378142787859람다 값 : 0.2610435482896753



![png](output_52_5183.png)


     89%|█████████████████████████████████████████████████████████████████████▏        | 1728/1948 [45:23<04:05,  1.11s/it]cv 값 : 0.2674887939758872람다 값 : 0.03879999152179427



![png](output_52_5186.png)


     89%|█████████████████████████████████████████████████████████████████████▏        | 1729/1948 [45:24<04:05,  1.12s/it]cv 값 : 0.40969732095709743람다 값 : 0.3330938018416802



![png](output_52_5189.png)


     89%|█████████████████████████████████████████████████████████████████████▎        | 1730/1948 [45:25<04:00,  1.11s/it]cv 값 : 0.4542780398120972람다 값 : 1.0206924195316949



![png](output_52_5192.png)


     89%|█████████████████████████████████████████████████████████████████████▎        | 1731/1948 [45:26<03:54,  1.08s/it]cv 값 : 0.26752136904405555람다 값 : 1.9793352538329756



![png](output_52_5195.png)


     89%|█████████████████████████████████████████████████████████████████████▎        | 1732/1948 [45:27<03:54,  1.08s/it]cv 값 : 0.2984114545048106람다 값 : 1.3228443882090202



![png](output_52_5198.png)


     89%|█████████████████████████████████████████████████████████████████████▍        | 1733/1948 [45:28<03:55,  1.09s/it]cv 값 : 0.25746177459635117람다 값 : 2.2240175495530505



![png](output_52_5201.png)


     89%|█████████████████████████████████████████████████████████████████████▍        | 1734/1948 [45:30<03:51,  1.08s/it]cv 값 : 0.9693115980738193람다 값 : 0.18030351517089513



![png](output_52_5204.png)


     89%|█████████████████████████████████████████████████████████████████████▍        | 1735/1948 [45:31<04:04,  1.15s/it]cv 값 : 0.4626098662230005람다 값 : -0.1680950116222926



![png](output_52_5207.png)


     89%|█████████████████████████████████████████████████████████████████████▌        | 1736/1948 [45:32<03:59,  1.13s/it]cv 값 : 0.27539688806844576람다 값 : 1.6303615791556254



![png](output_52_5210.png)


     89%|█████████████████████████████████████████████████████████████████████▌        | 1737/1948 [45:33<03:50,  1.09s/it]cv 값 : 0.4226775275764536람다 값 : 0.8515056012823273



![png](output_52_5213.png)


     89%|█████████████████████████████████████████████████████████████████████▌        | 1738/1948 [45:34<03:48,  1.09s/it]cv 값 : 0.3070961968205244람다 값 : 1.2502366470155473



![png](output_52_5216.png)


     89%|█████████████████████████████████████████████████████████████████████▋        | 1739/1948 [45:35<03:46,  1.08s/it]cv 값 : 0.18282149592573851람다 값 : -0.039362485564841704



![png](output_52_5219.png)


     89%|█████████████████████████████████████████████████████████████████████▋        | 1740/1948 [45:36<03:55,  1.13s/it]cv 값 : 0.18955579864447597람다 값 : 1.1926254963431306



![png](output_52_5222.png)


     89%|█████████████████████████████████████████████████████████████████████▋        | 1741/1948 [45:38<03:58,  1.15s/it]cv 값 : 0.5749221351330212람다 값 : 0.3876071459219212



![png](output_52_5225.png)


     89%|█████████████████████████████████████████████████████████████████████▊        | 1742/1948 [45:38<03:46,  1.10s/it]cv 값 : 0.3935570349963883람다 값 : 0.5864118043969241



![png](output_52_5228.png)


     89%|█████████████████████████████████████████████████████████████████████▊        | 1743/1948 [45:40<03:39,  1.07s/it]cv 값 : 0.285687661126308람다 값 : 1.0234361400606904



![png](output_52_5231.png)


     90%|█████████████████████████████████████████████████████████████████████▊        | 1744/1948 [45:41<03:45,  1.11s/it]cv 값 : 0.22643343336337138람다 값 : 0.7601262134304612



![png](output_52_5234.png)


     90%|█████████████████████████████████████████████████████████████████████▊        | 1745/1948 [45:42<03:47,  1.12s/it]cv 값 : 0.3377536794005343람다 값 : 0.4823712957633292



![png](output_52_5237.png)


     90%|█████████████████████████████████████████████████████████████████████▉        | 1746/1948 [45:43<03:45,  1.12s/it]cv 값 : 0.48689284298253127람다 값 : 0.3185106031844566



![png](output_52_5240.png)


     90%|█████████████████████████████████████████████████████████████████████▉        | 1747/1948 [45:44<03:42,  1.10s/it]cv 값 : 0.19635848204856346람다 값 : 0.7568090470335783



![png](output_52_5243.png)


     90%|█████████████████████████████████████████████████████████████████████▉        | 1748/1948 [45:45<03:42,  1.11s/it]cv 값 : 0.44332855019606565람다 값 : 0.6250913205691918



![png](output_52_5246.png)


     90%|██████████████████████████████████████████████████████████████████████        | 1749/1948 [45:46<03:44,  1.13s/it]cv 값 : 0.531570746152731람다 값 : -0.17122889600775582



![png](output_52_5249.png)


     90%|██████████████████████████████████████████████████████████████████████        | 1750/1948 [45:47<03:41,  1.12s/it]cv 값 : 0.3500568188322325람다 값 : 0.3405496289537643



![png](output_52_5252.png)


     90%|██████████████████████████████████████████████████████████████████████        | 1751/1948 [45:48<03:36,  1.10s/it]cv 값 : 0.1304443143582718람다 값 : 0.255442445547055



![png](output_52_5255.png)


     90%|██████████████████████████████████████████████████████████████████████▏       | 1752/1948 [45:50<03:35,  1.10s/it]cv 값 : 0.41390257179262047람다 값 : 0.4786048637329258



![png](output_52_5258.png)


     90%|██████████████████████████████████████████████████████████████████████▏       | 1753/1948 [45:51<03:39,  1.13s/it]cv 값 : 0.32129364644257624람다 값 : 0.06791969413404815



![png](output_52_5261.png)


     90%|██████████████████████████████████████████████████████████████████████▏       | 1754/1948 [45:52<03:37,  1.12s/it]cv 값 : 0.24548552849002753람다 값 : -0.7395197307073513



![png](output_52_5264.png)


     90%|██████████████████████████████████████████████████████████████████████▎       | 1755/1948 [45:53<03:37,  1.13s/it]cv 값 : 1.2176887871026956람다 값 : 0.18750431979594975



![png](output_52_5267.png)


     90%|██████████████████████████████████████████████████████████████████████▎       | 1756/1948 [45:54<03:35,  1.12s/it]cv 값 : 0.3002195192482078람다 값 : 0.49301511399588105



![png](output_52_5270.png)


     90%|██████████████████████████████████████████████████████████████████████▎       | 1757/1948 [45:55<03:29,  1.10s/it]cv 값 : 0.5790680717956732람다 값 : 0.6708389181353918



![png](output_52_5273.png)


     90%|██████████████████████████████████████████████████████████████████████▍       | 1758/1948 [45:56<03:36,  1.14s/it]cv 값 : 0.2227066908190018람다 값 : 1.3597552604737593



![png](output_52_5276.png)


     90%|██████████████████████████████████████████████████████████████████████▍       | 1759/1948 [45:58<03:35,  1.14s/it]cv 값 : 0.4412973960717562람다 값 : 0.7858259218516639



![png](output_52_5279.png)


     90%|██████████████████████████████████████████████████████████████████████▍       | 1760/1948 [45:59<03:33,  1.14s/it]cv 값 : 0.3986197839321228람다 값 : 0.2159615231334887



![png](output_52_5282.png)


     90%|██████████████████████████████████████████████████████████████████████▌       | 1761/1948 [46:00<03:27,  1.11s/it]cv 값 : 0.5777871751798707람다 값 : 0.618018357420206



![png](output_52_5285.png)


     90%|██████████████████████████████████████████████████████████████████████▌       | 1762/1948 [46:01<03:31,  1.14s/it]cv 값 : 0.600493913655866람다 값 : 0.5488065110714434



![png](output_52_5288.png)


     91%|██████████████████████████████████████████████████████████████████████▌       | 1763/1948 [46:02<03:32,  1.15s/it]cv 값 : 0.28279840194118216람다 값 : 0.19422009386239236



![png](output_52_5291.png)


     91%|██████████████████████████████████████████████████████████████████████▋       | 1764/1948 [46:03<03:22,  1.10s/it]cv 값 : 0.2562566555591543람다 값 : 1.4602018584335081



![png](output_52_5294.png)


     91%|██████████████████████████████████████████████████████████████████████▋       | 1765/1948 [46:04<03:25,  1.12s/it]cv 값 : 0.2784241334928952람다 값 : 0.6431383382714445



![png](output_52_5297.png)


     91%|██████████████████████████████████████████████████████████████████████▋       | 1766/1948 [46:05<03:26,  1.13s/it]cv 값 : 0.5087159620143954람다 값 : 0.21464469516391263



![png](output_52_5300.png)


     91%|██████████████████████████████████████████████████████████████████████▊       | 1767/1948 [46:07<03:27,  1.15s/it]cv 값 : 0.2784067052697582람다 값 : 1.0139475635484714



![png](output_52_5303.png)


     91%|██████████████████████████████████████████████████████████████████████▊       | 1768/1948 [46:08<03:22,  1.12s/it]cv 값 : 0.27058189415211026람다 값 : 1.4523164624734655



![png](output_52_5306.png)


     91%|██████████████████████████████████████████████████████████████████████▊       | 1769/1948 [46:09<03:25,  1.15s/it]cv 값 : 0.3982307605030147람다 값 : 0.2924338954806717



![png](output_52_5309.png)


     91%|██████████████████████████████████████████████████████████████████████▊       | 1770/1948 [46:10<03:16,  1.10s/it]cv 값 : 0.3332675568759521람다 값 : 0.5456751724187007



![png](output_52_5312.png)


     91%|██████████████████████████████████████████████████████████████████████▉       | 1771/1948 [46:11<03:25,  1.16s/it]cv 값 : 0.3007325270843129람다 값 : 1.0020085893476538



![png](output_52_5315.png)


     91%|██████████████████████████████████████████████████████████████████████▉       | 1772/1948 [46:12<03:31,  1.20s/it]cv 값 : 0.45004772113477304람다 값 : 0.44033774541084564



![png](output_52_5318.png)


     91%|██████████████████████████████████████████████████████████████████████▉       | 1773/1948 [46:13<03:21,  1.15s/it]cv 값 : 0.34379442227800255람다 값 : 0.8129781285917206



![png](output_52_5321.png)


     91%|███████████████████████████████████████████████████████████████████████       | 1774/1948 [46:15<03:20,  1.15s/it]cv 값 : 0.7197599817928041람다 값 : -0.462999435840273



![png](output_52_5324.png)


     91%|███████████████████████████████████████████████████████████████████████       | 1775/1948 [46:16<03:23,  1.18s/it]cv 값 : 0.2135881568096715람다 값 : 2.364682487663021



![png](output_52_5327.png)


     91%|███████████████████████████████████████████████████████████████████████       | 1776/1948 [46:17<03:37,  1.27s/it]cv 값 : 0.46199433727291767람다 값 : 0.7009302102901093



![png](output_52_5330.png)


     91%|███████████████████████████████████████████████████████████████████████▏      | 1777/1948 [46:18<03:26,  1.21s/it]cv 값 : 0.4354166077789583람다 값 : -0.7016001992263383



![png](output_52_5333.png)


     91%|███████████████████████████████████████████████████████████████████████▏      | 1778/1948 [46:20<03:26,  1.21s/it]cv 값 : 0.7164276322417209람다 값 : -0.2033108938700132



![png](output_52_5336.png)


     91%|███████████████████████████████████████████████████████████████████████▏      | 1779/1948 [46:21<03:20,  1.19s/it]cv 값 : 0.23279483932594786람다 값 : 0.8939169936799829



![png](output_52_5339.png)


     91%|███████████████████████████████████████████████████████████████████████▎      | 1780/1948 [46:22<03:18,  1.18s/it]cv 값 : 0.337970846022665람다 값 : 0.028107467014728056



![png](output_52_5342.png)


     91%|███████████████████████████████████████████████████████████████████████▎      | 1781/1948 [46:23<03:12,  1.15s/it]cv 값 : 0.2300636820769933람다 값 : 0.43231260332795424



![png](output_52_5345.png)


     91%|███████████████████████████████████████████████████████████████████████▎      | 1782/1948 [46:24<03:13,  1.17s/it]cv 값 : 0.2609442838941735람다 값 : 1.8348060034083404



![png](output_52_5348.png)


     92%|███████████████████████████████████████████████████████████████████████▍      | 1783/1948 [46:25<03:11,  1.16s/it]cv 값 : 0.37426013663638524람다 값 : 0.7289924926946765



![png](output_52_5351.png)


     92%|███████████████████████████████████████████████████████████████████████▍      | 1784/1948 [46:26<03:05,  1.13s/it]cv 값 : 0.22921011376289람다 값 : 1.871327574843763



![png](output_52_5354.png)


     92%|███████████████████████████████████████████████████████████████████████▍      | 1785/1948 [46:28<03:12,  1.18s/it]cv 값 : 0.15826565333505885람다 값 : 0.17977035543070116



![png](output_52_5357.png)


     92%|███████████████████████████████████████████████████████████████████████▌      | 1786/1948 [46:29<03:04,  1.14s/it]cv 값 : 0.2564214095468948람다 값 : 0.771260379938319



![png](output_52_5360.png)


     92%|███████████████████████████████████████████████████████████████████████▌      | 1787/1948 [46:30<02:59,  1.11s/it]cv 값 : 0.3036333076329333람다 값 : -0.4522433249633096



![png](output_52_5363.png)


     92%|███████████████████████████████████████████████████████████████████████▌      | 1788/1948 [46:31<03:02,  1.14s/it]cv 값 : 0.609564404654327람다 값 : 0.6746246943268875



![png](output_52_5366.png)


     92%|███████████████████████████████████████████████████████████████████████▋      | 1789/1948 [46:32<03:07,  1.18s/it]cv 값 : 0.2776931331938362람다 값 : 1.8330276485329449



![png](output_52_5369.png)


     92%|███████████████████████████████████████████████████████████████████████▋      | 1790/1948 [46:33<03:02,  1.15s/it]cv 값 : 0.2135460722638946람다 값 : 1.4770993016464302



![png](output_52_5372.png)


     92%|███████████████████████████████████████████████████████████████████████▋      | 1791/1948 [46:35<03:00,  1.15s/it]cv 값 : 0.5421343713397473람다 값 : 0.6587391584080194



![png](output_52_5375.png)


     92%|███████████████████████████████████████████████████████████████████████▊      | 1792/1948 [46:36<02:59,  1.15s/it]cv 값 : 0.3722111427357366람다 값 : 1.3173457466927556



![png](output_52_5378.png)


     92%|███████████████████████████████████████████████████████████████████████▊      | 1793/1948 [46:37<03:04,  1.19s/it]cv 값 : 0.7744190079265979람다 값 : 0.33387864045885307



![png](output_52_5381.png)


     92%|███████████████████████████████████████████████████████████████████████▊      | 1794/1948 [46:38<02:58,  1.16s/it]cv 값 : 0.31698756683284496람다 값 : 1.6829338038746784



![png](output_52_5384.png)


     92%|███████████████████████████████████████████████████████████████████████▊      | 1795/1948 [46:39<02:58,  1.17s/it]cv 값 : 0.5626854962912314람다 값 : 0.7980584034541137



![png](output_52_5387.png)


     92%|███████████████████████████████████████████████████████████████████████▉      | 1796/1948 [46:40<02:54,  1.15s/it]cv 값 : 0.210842777259232람다 값 : 2.1994335838626324



![png](output_52_5390.png)


     92%|███████████████████████████████████████████████████████████████████████▉      | 1797/1948 [46:41<02:50,  1.13s/it]cv 값 : 0.24503178719351312람다 값 : 2.153212208455772



![png](output_52_5393.png)


     92%|███████████████████████████████████████████████████████████████████████▉      | 1798/1948 [46:43<02:55,  1.17s/it]cv 값 : 0.7865468630877278람다 값 : 0.18821142846404967



![png](output_52_5396.png)


     92%|████████████████████████████████████████████████████████████████████████      | 1799/1948 [46:44<02:53,  1.16s/it]cv 값 : 0.32925350594824754람다 값 : 1.0592703167852566



![png](output_52_5399.png)


     92%|████████████████████████████████████████████████████████████████████████      | 1800/1948 [46:45<02:46,  1.13s/it]cv 값 : 0.3409282500164106람다 값 : 0.9086488069273524



![png](output_52_5402.png)


     92%|████████████████████████████████████████████████████████████████████████      | 1801/1948 [46:46<02:45,  1.12s/it]cv 값 : 0.794614688558228람다 값 : 0.3775911393055526



![png](output_52_5405.png)


     93%|████████████████████████████████████████████████████████████████████████▏     | 1802/1948 [46:47<02:52,  1.18s/it]cv 값 : 0.24285651137376685람다 값 : 2.037134216635983



![png](output_52_5408.png)


     93%|████████████████████████████████████████████████████████████████████████▏     | 1803/1948 [46:48<02:49,  1.17s/it]cv 값 : 0.7052669783846303람다 값 : 0.47172530820676334



![png](output_52_5411.png)


     93%|████████████████████████████████████████████████████████████████████████▏     | 1804/1948 [46:50<02:43,  1.13s/it]cv 값 : 0.2929108989504792람다 값 : 1.0587767261085497



![png](output_52_5414.png)


     93%|████████████████████████████████████████████████████████████████████████▎     | 1805/1948 [46:51<02:44,  1.15s/it]cv 값 : 0.3278250440927353람다 값 : 0.996784142370555



![png](output_52_5417.png)


     93%|████████████████████████████████████████████████████████████████████████▎     | 1806/1948 [46:52<02:42,  1.14s/it]cv 값 : 0.6166783181819219람다 값 : 0.5449549414815716



![png](output_52_5420.png)


     93%|████████████████████████████████████████████████████████████████████████▎     | 1807/1948 [46:53<02:37,  1.12s/it]cv 값 : 1.5004854582058076람다 값 : -0.015844112349858316



![png](output_52_5423.png)


     93%|████████████████████████████████████████████████████████████████████████▍     | 1808/1948 [46:54<02:32,  1.09s/it]cv 값 : 0.733664510734876람다 값 : 0.4575663507056905



![png](output_52_5426.png)


     93%|████████████████████████████████████████████████████████████████████████▍     | 1809/1948 [46:55<02:30,  1.09s/it]cv 값 : 0.19828857883353987람다 값 : 0.18651669147333516



![png](output_52_5429.png)


     93%|████████████████████████████████████████████████████████████████████████▍     | 1810/1948 [46:56<02:30,  1.09s/it]cv 값 : 0.22155407482084913람다 값 : 1.5375142971783713



![png](output_52_5432.png)


     93%|████████████████████████████████████████████████████████████████████████▌     | 1811/1948 [46:57<02:38,  1.15s/it]cv 값 : 0.5558588956085131람다 값 : 0.9493588096738758



![png](output_52_5435.png)


     93%|████████████████████████████████████████████████████████████████████████▌     | 1812/1948 [46:59<02:36,  1.15s/it]cv 값 : 0.38091422486998083람다 값 : 1.3403055486727888



![png](output_52_5438.png)


     93%|████████████████████████████████████████████████████████████████████████▌     | 1813/1948 [47:00<02:35,  1.15s/it]cv 값 : 0.2030092702013517람다 값 : 1.1696182241086155



![png](output_52_5441.png)


     93%|████████████████████████████████████████████████████████████████████████▋     | 1814/1948 [47:01<02:33,  1.15s/it]cv 값 : 0.3882952168442574람다 값 : 0.040749999767062166



![png](output_52_5444.png)


     93%|████████████████████████████████████████████████████████████████████████▋     | 1815/1948 [47:02<02:37,  1.18s/it]cv 값 : 0.16019704106157676람다 값 : 0.10577489934796608



![png](output_52_5447.png)


     93%|████████████████████████████████████████████████████████████████████████▋     | 1816/1948 [47:03<02:27,  1.12s/it]cv 값 : 0.33711320928607524람다 값 : 0.21600368683711543



![png](output_52_5450.png)


     93%|████████████████████████████████████████████████████████████████████████▊     | 1817/1948 [47:04<02:28,  1.13s/it]cv 값 : 0.4489590007150494람다 값 : 0.7178326083850934



![png](output_52_5453.png)


     93%|████████████████████████████████████████████████████████████████████████▊     | 1818/1948 [47:05<02:22,  1.10s/it]cv 값 : 0.5457479805846689람다 값 : 0.5805192564287975



![png](output_52_5456.png)


     93%|████████████████████████████████████████████████████████████████████████▊     | 1819/1948 [47:06<02:20,  1.09s/it]cv 값 : 0.33035977889005275람다 값 : 0.22658241537658336



![png](output_52_5459.png)


     93%|████████████████████████████████████████████████████████████████████████▊     | 1820/1948 [47:07<02:22,  1.12s/it]cv 값 : 0.3918850830344171람다 값 : 1.3020795519060426



![png](output_52_5462.png)


     93%|████████████████████████████████████████████████████████████████████████▉     | 1821/1948 [47:09<02:22,  1.12s/it]cv 값 : 1.4606727077187람다 값 : 0.2505991859352215



![png](output_52_5465.png)


     94%|████████████████████████████████████████████████████████████████████████▉     | 1822/1948 [47:10<02:23,  1.14s/it]cv 값 : 0.34126891106375357람다 값 : 0.0664569239523734



![png](output_52_5468.png)


     94%|████████████████████████████████████████████████████████████████████████▉     | 1823/1948 [47:11<02:23,  1.15s/it]cv 값 : 0.4781525160946541람다 값 : 0.31203300338418755



![png](output_52_5471.png)


     94%|█████████████████████████████████████████████████████████████████████████     | 1824/1948 [47:12<02:24,  1.16s/it]cv 값 : 0.28046085701105045람다 값 : -0.11587819696755178



![png](output_52_5474.png)


     94%|█████████████████████████████████████████████████████████████████████████     | 1825/1948 [47:13<02:20,  1.14s/it]cv 값 : 0.21276696134730624람다 값 : 0.780159787344627



![png](output_52_5477.png)


     94%|█████████████████████████████████████████████████████████████████████████     | 1826/1948 [47:14<02:16,  1.12s/it]cv 값 : 0.3630388529970225람다 값 : 0.7445395577182853



![png](output_52_5480.png)


     94%|█████████████████████████████████████████████████████████████████████████▏    | 1827/1948 [47:16<02:17,  1.14s/it]cv 값 : 0.29355639078303586람다 값 : 2.0525254192292945



![png](output_52_5483.png)


     94%|█████████████████████████████████████████████████████████████████████████▏    | 1828/1948 [47:17<02:12,  1.11s/it]cv 값 : 0.30462983307944685람다 값 : -0.15907822030402294



![png](output_52_5486.png)


     94%|█████████████████████████████████████████████████████████████████████████▏    | 1829/1948 [47:18<02:15,  1.14s/it]cv 값 : 0.4940100320639777람다 값 : 0.48873052243924775



![png](output_52_5489.png)


     94%|█████████████████████████████████████████████████████████████████████████▎    | 1830/1948 [47:19<02:15,  1.15s/it]cv 값 : 0.3176213505088758람다 값 : 0.8766707215834774



![png](output_52_5492.png)


     94%|█████████████████████████████████████████████████████████████████████████▎    | 1831/1948 [47:20<02:19,  1.19s/it]cv 값 : 0.2595032646001931람다 값 : 0.7040369714883393



![png](output_52_5495.png)


     94%|█████████████████████████████████████████████████████████████████████████▎    | 1832/1948 [47:21<02:16,  1.18s/it]cv 값 : 0.4649941887264505람다 값 : 0.6217878230967641



![png](output_52_5498.png)


     94%|█████████████████████████████████████████████████████████████████████████▍    | 1833/1948 [47:23<02:18,  1.21s/it]cv 값 : 0.5530830099121655람다 값 : 0.003393191142800417



![png](output_52_5501.png)


     94%|█████████████████████████████████████████████████████████████████████████▍    | 1834/1948 [47:24<02:15,  1.19s/it]cv 값 : 0.20387101426078827람다 값 : 0.8905864365178222



![png](output_52_5504.png)


     94%|█████████████████████████████████████████████████████████████████████████▍    | 1835/1948 [47:25<02:08,  1.14s/it]cv 값 : 0.288586947415259람다 값 : 1.1111665709100307



![png](output_52_5507.png)


     94%|█████████████████████████████████████████████████████████████████████████▌    | 1836/1948 [47:26<02:03,  1.10s/it]cv 값 : 0.23380286324502964람다 값 : -0.8869372672434951



![png](output_52_5510.png)


     94%|█████████████████████████████████████████████████████████████████████████▌    | 1837/1948 [47:27<02:00,  1.08s/it]cv 값 : 0.4927771154949874람다 값 : -0.4607873477805973



![png](output_52_5513.png)


     94%|█████████████████████████████████████████████████████████████████████████▌    | 1838/1948 [47:28<02:04,  1.13s/it]cv 값 : 0.38498293864771993람다 값 : 0.2195913940589762



![png](output_52_5516.png)


     94%|█████████████████████████████████████████████████████████████████████████▋    | 1839/1948 [47:29<02:03,  1.13s/it]cv 값 : 0.49633326068645045람다 값 : 0.6035423844259077



![png](output_52_5519.png)


     94%|█████████████████████████████████████████████████████████████████████████▋    | 1840/1948 [47:30<01:58,  1.10s/it]cv 값 : 0.5125213326981586람다 값 : 0.6695130272949097



![png](output_52_5522.png)


     95%|█████████████████████████████████████████████████████████████████████████▋    | 1841/1948 [47:31<01:56,  1.09s/it]cv 값 : 0.39112188571882384람다 값 : 0.7895557834341185



![png](output_52_5525.png)


     95%|█████████████████████████████████████████████████████████████████████████▊    | 1842/1948 [47:32<01:58,  1.11s/it]cv 값 : 0.3170566790874581람다 값 : 1.392187398893896



![png](output_52_5528.png)


     95%|█████████████████████████████████████████████████████████████████████████▊    | 1843/1948 [47:34<01:54,  1.09s/it]cv 값 : 0.3043266913666781람다 값 : 1.154969352438568



![png](output_52_5531.png)


     95%|█████████████████████████████████████████████████████████████████████████▊    | 1844/1948 [47:35<01:56,  1.12s/it]cv 값 : 0.20711918759830367람다 값 : 1.6227277162303224



![png](output_52_5534.png)


     95%|█████████████████████████████████████████████████████████████████████████▉    | 1845/1948 [47:36<01:54,  1.11s/it]cv 값 : 0.3817704697025428람다 값 : 0.7256983018760302



![png](output_52_5537.png)


     95%|█████████████████████████████████████████████████████████████████████████▉    | 1846/1948 [47:37<01:58,  1.16s/it]cv 값 : 0.5252401065479484람다 값 : 0.6664661860615468



![png](output_52_5540.png)


     95%|█████████████████████████████████████████████████████████████████████████▉    | 1847/1948 [47:38<01:55,  1.14s/it]cv 값 : 0.3204978391479251람다 값 : 0.30924820172200684



![png](output_52_5543.png)


     95%|█████████████████████████████████████████████████████████████████████████▉    | 1848/1948 [47:39<01:54,  1.15s/it]cv 값 : 0.28439313377968917람다 값 : 0.5564442539696888



![png](output_52_5546.png)


     95%|██████████████████████████████████████████████████████████████████████████    | 1849/1948 [47:40<01:51,  1.13s/it]cv 값 : 0.2586501065209342람다 값 : 1.805390967531878



![png](output_52_5549.png)


     95%|██████████████████████████████████████████████████████████████████████████    | 1850/1948 [47:42<01:49,  1.12s/it]cv 값 : 0.1757084541928598람다 값 : 2.383793802316462



![png](output_52_5552.png)


     95%|██████████████████████████████████████████████████████████████████████████    | 1851/1948 [47:43<01:51,  1.15s/it]cv 값 : 0.2117488693858332람다 값 : 0.7914895029492053



![png](output_52_5555.png)


     95%|██████████████████████████████████████████████████████████████████████████▏   | 1852/1948 [47:44<01:48,  1.13s/it]cv 값 : 0.31490494241148265람다 값 : 0.6990051298332811



![png](output_52_5558.png)


     95%|██████████████████████████████████████████████████████████████████████████▏   | 1853/1948 [47:45<01:46,  1.12s/it]cv 값 : 0.36841132102884117람다 값 : 0.10195872658096637



![png](output_52_5561.png)


     95%|██████████████████████████████████████████████████████████████████████████▏   | 1854/1948 [47:46<01:44,  1.11s/it]cv 값 : 0.518154314872529람다 값 : -0.21702156712856807



![png](output_52_5564.png)


     95%|██████████████████████████████████████████████████████████████████████████▎   | 1855/1948 [47:48<02:00,  1.30s/it]cv 값 : 0.35462469953509484람다 값 : 0.4035321993336125



![png](output_52_5567.png)


     95%|██████████████████████████████████████████████████████████████████████████▎   | 1856/1948 [47:49<02:00,  1.31s/it]cv 값 : 0.3458937126325987람다 값 : 1.0766704613605715



![png](output_52_5570.png)


     95%|██████████████████████████████████████████████████████████████████████████▎   | 1857/1948 [47:50<01:55,  1.27s/it]cv 값 : 0.6930852874417753람다 값 : 0.2963459876764238



![png](output_52_5573.png)


     95%|██████████████████████████████████████████████████████████████████████████▍   | 1858/1948 [47:51<01:52,  1.25s/it]cv 값 : 0.18140303160059368람다 값 : 2.7226361879089973



![png](output_52_5576.png)


     95%|██████████████████████████████████████████████████████████████████████████▍   | 1859/1948 [47:53<01:53,  1.28s/it]cv 값 : 0.18789358982615628람다 값 : 2.169796242693519



![png](output_52_5579.png)


     95%|██████████████████████████████████████████████████████████████████████████▍   | 1860/1948 [47:54<01:46,  1.21s/it]cv 값 : 0.21721205198951693람다 값 : 2.392405193430094



![png](output_52_5582.png)


     96%|██████████████████████████████████████████████████████████████████████████▌   | 1861/1948 [47:55<01:46,  1.22s/it]cv 값 : 1.1853883854499572람다 값 : 0.1717569078838278



![png](output_52_5585.png)


     96%|██████████████████████████████████████████████████████████████████████████▌   | 1862/1948 [47:56<01:41,  1.18s/it]cv 값 : 0.3718559892465535람다 값 : 1.1819711240450401



![png](output_52_5588.png)


     96%|██████████████████████████████████████████████████████████████████████████▌   | 1863/1948 [47:57<01:39,  1.17s/it]cv 값 : 0.3671037079389639람다 값 : 0.6814001810090401



![png](output_52_5591.png)


     96%|██████████████████████████████████████████████████████████████████████████▋   | 1864/1948 [47:59<01:38,  1.17s/it]cv 값 : 0.34863479478771525람다 값 : 1.3617786789551733



![png](output_52_5594.png)


     96%|██████████████████████████████████████████████████████████████████████████▋   | 1865/1948 [48:00<01:39,  1.20s/it]cv 값 : 0.27242662165266285람다 값 : 1.6649386468731302



![png](output_52_5597.png)


     96%|██████████████████████████████████████████████████████████████████████████▋   | 1866/1948 [48:01<01:37,  1.19s/it]cv 값 : 0.24023372930809034람다 값 : -0.1223114668256565



![png](output_52_5600.png)


     96%|██████████████████████████████████████████████████████████████████████████▊   | 1867/1948 [48:02<01:33,  1.15s/it]cv 값 : 0.39823580063172237람다 값 : 1.0618293670589485



![png](output_52_5603.png)


     96%|██████████████████████████████████████████████████████████████████████████▊   | 1868/1948 [48:03<01:34,  1.18s/it]cv 값 : 0.8433687446621685람다 값 : 0.3506221620620493



![png](output_52_5606.png)


     96%|██████████████████████████████████████████████████████████████████████████▊   | 1869/1948 [48:04<01:33,  1.18s/it]cv 값 : 0.2673389122014831람다 값 : 1.6397148772751764



![png](output_52_5609.png)


     96%|██████████████████████████████████████████████████████████████████████████▉   | 1870/1948 [48:06<01:31,  1.18s/it]cv 값 : 0.4580215595161187람다 값 : 0.9520218904609977



![png](output_52_5612.png)


     96%|██████████████████████████████████████████████████████████████████████████▉   | 1871/1948 [48:07<01:29,  1.16s/it]cv 값 : 0.2891112738332375람다 값 : 1.5033569063563719



![png](output_52_5615.png)


     96%|██████████████████████████████████████████████████████████████████████████▉   | 1872/1948 [48:08<01:26,  1.14s/it]cv 값 : 0.7722572740494811람다 값 : 0.3136491297397491



![png](output_52_5618.png)


     96%|██████████████████████████████████████████████████████████████████████████▉   | 1873/1948 [48:09<01:30,  1.21s/it]cv 값 : 0.25291732113875753람다 값 : 2.4587637638986704



![png](output_52_5621.png)


     96%|███████████████████████████████████████████████████████████████████████████   | 1874/1948 [48:11<01:43,  1.40s/it]cv 값 : 0.3287679919608012람다 값 : -0.5083434823292162



![png](output_52_5624.png)


     96%|███████████████████████████████████████████████████████████████████████████   | 1875/1948 [48:13<01:50,  1.52s/it]cv 값 : 0.2880884121680097람다 값 : 0.4605282081228072



![png](output_52_5627.png)


     96%|███████████████████████████████████████████████████████████████████████████   | 1876/1948 [48:15<01:56,  1.62s/it]cv 값 : 0.2615638648640418람다 값 : 0.22522811861688594



![png](output_52_5630.png)


     96%|███████████████████████████████████████████████████████████████████████████▏  | 1877/1948 [48:17<02:03,  1.73s/it]cv 값 : 0.10392296963863397람다 값 : 1.4570241376138613



![png](output_52_5633.png)


     96%|███████████████████████████████████████████████████████████████████████████▏  | 1878/1948 [48:18<02:01,  1.74s/it]cv 값 : 0.26355709876907524람다 값 : -0.46290479239810645



![png](output_52_5636.png)


     96%|███████████████████████████████████████████████████████████████████████████▏  | 1879/1948 [48:20<01:57,  1.71s/it]cv 값 : 0.29438653883527527람다 값 : 1.4759703305415364



![png](output_52_5639.png)


     97%|███████████████████████████████████████████████████████████████████████████▎  | 1880/1948 [48:22<01:51,  1.65s/it]cv 값 : 0.28594615970022463람다 값 : -0.9146028681126874



![png](output_52_5642.png)


     97%|███████████████████████████████████████████████████████████████████████████▎  | 1881/1948 [48:24<02:02,  1.82s/it]cv 값 : 0.40804137868645557람다 값 : 0.6753133877156948



![png](output_52_5645.png)


     97%|███████████████████████████████████████████████████████████████████████████▎  | 1882/1948 [48:26<02:16,  2.06s/it]cv 값 : 0.8016411069897827람다 값 : 0.46205419216868465



![png](output_52_5648.png)


     97%|███████████████████████████████████████████████████████████████████████████▍  | 1883/1948 [48:29<02:16,  2.10s/it]cv 값 : 0.45722501422997835람다 값 : 0.9604612744762747



![png](output_52_5651.png)


     97%|███████████████████████████████████████████████████████████████████████████▍  | 1884/1948 [48:30<02:03,  1.92s/it]cv 값 : 0.5560668109029243람다 값 : -0.08852063700179338



![png](output_52_5654.png)


     97%|███████████████████████████████████████████████████████████████████████████▍  | 1885/1948 [48:32<02:02,  1.95s/it]cv 값 : 0.6791818833068479람다 값 : 0.6488873156130806



![png](output_52_5657.png)


     97%|███████████████████████████████████████████████████████████████████████████▌  | 1886/1948 [48:34<02:07,  2.06s/it]cv 값 : 0.1823666221530741람다 값 : 1.1251298025165801



![png](output_52_5660.png)


     97%|███████████████████████████████████████████████████████████████████████████▌  | 1887/1948 [48:36<01:56,  1.91s/it]cv 값 : 0.33967369742679676람다 값 : -0.7902481799181842



![png](output_52_5663.png)


     97%|███████████████████████████████████████████████████████████████████████████▌  | 1888/1948 [48:37<01:45,  1.75s/it]cv 값 : 0.23390376963859863람다 값 : 0.9442277807204179



![png](output_52_5666.png)


     97%|███████████████████████████████████████████████████████████████████████████▋  | 1889/1948 [48:39<01:32,  1.57s/it]cv 값 : 0.37822516173861054람다 값 : -0.5584770318496003



![png](output_52_5669.png)


     97%|███████████████████████████████████████████████████████████████████████████▋  | 1890/1948 [48:40<01:26,  1.49s/it]cv 값 : 0.20687789527517697람다 값 : 0.06405301236156698



![png](output_52_5672.png)


     97%|███████████████████████████████████████████████████████████████████████████▋  | 1891/1948 [48:41<01:19,  1.40s/it]cv 값 : 0.23459937618295063람다 값 : 0.12689613693567192



![png](output_52_5675.png)


     97%|███████████████████████████████████████████████████████████████████████████▊  | 1892/1948 [48:42<01:12,  1.30s/it]cv 값 : 0.42394244153378574람다 값 : 0.8445191171323722



![png](output_52_5678.png)


     97%|███████████████████████████████████████████████████████████████████████████▊  | 1893/1948 [48:43<01:08,  1.25s/it]cv 값 : 0.2176077168750772람다 값 : 0.40452069370397153



![png](output_52_5681.png)


     97%|███████████████████████████████████████████████████████████████████████████▊  | 1894/1948 [48:44<01:07,  1.26s/it]cv 값 : 0.7963171517847573람다 값 : 0.4093408969082534



![png](output_52_5684.png)


     97%|███████████████████████████████████████████████████████████████████████████▉  | 1895/1948 [48:46<01:06,  1.25s/it]cv 값 : 0.9680669498461083람다 값 : 0.12781634240442827



![png](output_52_5687.png)


     97%|███████████████████████████████████████████████████████████████████████████▉  | 1896/1948 [48:47<01:03,  1.21s/it]cv 값 : 0.2732226937965958람다 값 : 0.18000205626152072



![png](output_52_5690.png)


     97%|███████████████████████████████████████████████████████████████████████████▉  | 1897/1948 [48:48<01:00,  1.19s/it]cv 값 : 0.5195119390331948람다 값 : 0.6334041536343561



![png](output_52_5693.png)


     97%|███████████████████████████████████████████████████████████████████████████▉  | 1898/1948 [48:49<00:58,  1.18s/it]cv 값 : 0.8948722863564058람다 값 : -0.3165952612016289



![png](output_52_5696.png)


     97%|████████████████████████████████████████████████████████████████████████████  | 1899/1948 [48:51<01:01,  1.25s/it]cv 값 : 0.26810780079948354람다 값 : -0.19934863433992708



![png](output_52_5699.png)


     98%|████████████████████████████████████████████████████████████████████████████  | 1900/1948 [48:52<00:59,  1.23s/it]cv 값 : 0.5006728985380505람다 값 : 0.6322744793777062



![png](output_52_5702.png)


     98%|████████████████████████████████████████████████████████████████████████████  | 1901/1948 [48:53<00:57,  1.21s/it]cv 값 : 0.4546898618279108람다 값 : -0.5854648762117547



![png](output_52_5705.png)


     98%|████████████████████████████████████████████████████████████████████████████▏ | 1902/1948 [48:54<00:55,  1.22s/it]cv 값 : 0.3080740001632996람다 값 : 1.682660178446638



![png](output_52_5708.png)


     98%|████████████████████████████████████████████████████████████████████████████▏ | 1903/1948 [48:55<00:54,  1.21s/it]cv 값 : 0.27513675459261044람다 값 : -1.3301031429197643



![png](output_52_5711.png)


     98%|████████████████████████████████████████████████████████████████████████████▏ | 1904/1948 [48:57<00:53,  1.21s/it]cv 값 : 0.19942316515406425람다 값 : 0.9554989445998138



![png](output_52_5714.png)


     98%|████████████████████████████████████████████████████████████████████████████▎ | 1905/1948 [48:58<00:49,  1.16s/it]cv 값 : 0.26016504614508434람다 값 : 0.2864462199052539



![png](output_52_5717.png)


     98%|████████████████████████████████████████████████████████████████████████████▎ | 1906/1948 [48:59<00:49,  1.18s/it]cv 값 : 0.17937522867805317람다 값 : 1.686878116801868



![png](output_52_5720.png)


     98%|████████████████████████████████████████████████████████████████████████████▎ | 1907/1948 [49:00<00:51,  1.26s/it]cv 값 : 0.1801566784519027람다 값 : 2.1105397389298606



![png](output_52_5723.png)


     98%|████████████████████████████████████████████████████████████████████████████▍ | 1908/1948 [49:02<00:50,  1.27s/it]cv 값 : 1.2376706841173368람다 값 : 0.326741736762832



![png](output_52_5726.png)


     98%|████████████████████████████████████████████████████████████████████████████▍ | 1909/1948 [49:03<00:48,  1.23s/it]cv 값 : 0.38860530078942085람다 값 : 1.0775249755359573



![png](output_52_5729.png)


     98%|████████████████████████████████████████████████████████████████████████████▍ | 1910/1948 [49:04<00:44,  1.17s/it]cv 값 : 0.47730562299855944람다 값 : 0.7029867162715887



![png](output_52_5732.png)


     98%|████████████████████████████████████████████████████████████████████████████▌ | 1911/1948 [49:05<00:42,  1.16s/it]cv 값 : 0.31584040605819846람다 값 : -0.047280557265082956



![png](output_52_5735.png)


     98%|████████████████████████████████████████████████████████████████████████████▌ | 1912/1948 [49:06<00:42,  1.18s/it]cv 값 : 0.5335078277011549람다 값 : 0.8909929631410062



![png](output_52_5738.png)


     98%|████████████████████████████████████████████████████████████████████████████▌ | 1913/1948 [49:07<00:39,  1.13s/it]cv 값 : 0.24144763725254725람다 값 : 1.2904068675434706



![png](output_52_5741.png)


     98%|████████████████████████████████████████████████████████████████████████████▋ | 1914/1948 [49:08<00:38,  1.13s/it]cv 값 : 0.14418244885547568람다 값 : 3.121384751747884



![png](output_52_5744.png)


     98%|████████████████████████████████████████████████████████████████████████████▋ | 1915/1948 [49:09<00:37,  1.15s/it]cv 값 : 0.5414493237555131람다 값 : 0.8340608559544879



![png](output_52_5747.png)


     98%|████████████████████████████████████████████████████████████████████████████▋ | 1916/1948 [49:11<00:38,  1.22s/it]cv 값 : 0.36923466128316473람다 값 : -0.3373016297590172



![png](output_52_5750.png)


     98%|████████████████████████████████████████████████████████████████████████████▊ | 1917/1948 [49:12<00:37,  1.20s/it]cv 값 : 0.13714518740896905람다 값 : 1.4799509518189184



![png](output_52_5753.png)


     98%|████████████████████████████████████████████████████████████████████████████▊ | 1918/1948 [49:13<00:35,  1.18s/it]cv 값 : 0.5364117846367215람다 값 : 0.5273132921706537



![png](output_52_5756.png)


     99%|████████████████████████████████████████████████████████████████████████████▊ | 1919/1948 [49:14<00:34,  1.20s/it]cv 값 : 0.4996228098713809람다 값 : 0.21572526829370683



![png](output_52_5759.png)


     99%|████████████████████████████████████████████████████████████████████████████▉ | 1920/1948 [49:16<00:33,  1.19s/it]cv 값 : 0.36239107059921305람다 값 : 0.653189432247367



![png](output_52_5762.png)


     99%|████████████████████████████████████████████████████████████████████████████▉ | 1921/1948 [49:17<00:31,  1.18s/it]cv 값 : 0.3004709703047692람다 값 : -0.09016338572350759



![png](output_52_5765.png)


     99%|████████████████████████████████████████████████████████████████████████████▉ | 1922/1948 [49:18<00:30,  1.17s/it]cv 값 : 0.18760148924876546람다 값 : 0.6002884003796034



![png](output_52_5768.png)


     99%|████████████████████████████████████████████████████████████████████████████▉ | 1923/1948 [49:19<00:28,  1.13s/it]cv 값 : 0.3999590265253531람다 값 : 0.8140165711153421



![png](output_52_5771.png)


     99%|█████████████████████████████████████████████████████████████████████████████ | 1924/1948 [49:20<00:27,  1.13s/it]cv 값 : 1.0422239412602288람다 값 : 0.3171677599246791



![png](output_52_5774.png)


     99%|█████████████████████████████████████████████████████████████████████████████ | 1925/1948 [49:21<00:27,  1.19s/it]cv 값 : 0.3656400374614958람다 값 : 0.10981707255333867



![png](output_52_5777.png)


     99%|█████████████████████████████████████████████████████████████████████████████ | 1926/1948 [49:22<00:24,  1.13s/it]cv 값 : 0.29922972411412485람다 값 : -1.3018087663420412



![png](output_52_5780.png)


     99%|█████████████████████████████████████████████████████████████████████████████▏| 1927/1948 [49:24<00:24,  1.16s/it]cv 값 : 0.3080388154237728람다 값 : 1.6634651758567551



![png](output_52_5783.png)


     99%|█████████████████████████████████████████████████████████████████████████████▏| 1928/1948 [49:25<00:23,  1.17s/it]cv 값 : 0.18150237549501877람다 값 : 3.2609783945631503



![png](output_52_5786.png)


     99%|█████████████████████████████████████████████████████████████████████████████▏| 1929/1948 [49:26<00:24,  1.27s/it]cv 값 : 0.3419579584669729람다 값 : 1.0816576531471058



![png](output_52_5789.png)


     99%|█████████████████████████████████████████████████████████████████████████████▎| 1930/1948 [49:27<00:21,  1.20s/it]cv 값 : 0.5018925036399039람다 값 : -0.1762192298129059



![png](output_52_5792.png)


     99%|█████████████████████████████████████████████████████████████████████████████▎| 1931/1948 [49:28<00:19,  1.13s/it]cv 값 : 0.2983618667211535람다 값 : 1.5165554879759204



![png](output_52_5795.png)


     99%|█████████████████████████████████████████████████████████████████████████████▎| 1932/1948 [49:29<00:18,  1.15s/it]cv 값 : 0.39700166708084417람다 값 : 0.46527248164994334



![png](output_52_5798.png)


     99%|█████████████████████████████████████████████████████████████████████████████▍| 1933/1948 [49:30<00:16,  1.09s/it]cv 값 : 0.3255628477755965람다 값 : 1.7802104150213474



![png](output_52_5801.png)


     99%|█████████████████████████████████████████████████████████████████████████████▍| 1934/1948 [49:32<00:15,  1.12s/it]cv 값 : 0.603159770080702람다 값 : 0.25042904261459753



![png](output_52_5804.png)


     99%|█████████████████████████████████████████████████████████████████████████████▍| 1935/1948 [49:33<00:14,  1.09s/it]cv 값 : 0.4587277424820505람다 값 : 0.716484117077425



![png](output_52_5807.png)


     99%|█████████████████████████████████████████████████████████████████████████████▌| 1936/1948 [49:34<00:12,  1.05s/it]cv 값 : 0.25517917423116354람다 값 : 1.8773698226274471



![png](output_52_5810.png)


     99%|█████████████████████████████████████████████████████████████████████████████▌| 1937/1948 [49:35<00:11,  1.03s/it]cv 값 : 0.30336477987570837람다 값 : 0.940275604053601



![png](output_52_5813.png)


     99%|█████████████████████████████████████████████████████████████████████████████▌| 1938/1948 [49:36<00:10,  1.07s/it]cv 값 : 0.2566815948764441람다 값 : 2.4173887631027853



![png](output_52_5816.png)


    100%|█████████████████████████████████████████████████████████████████████████████▋| 1939/1948 [49:37<00:09,  1.09s/it]cv 값 : 0.3381431750480781람다 값 : 1.6117001205556358



![png](output_52_5819.png)


    100%|█████████████████████████████████████████████████████████████████████████████▋| 1940/1948 [49:38<00:08,  1.08s/it]cv 값 : 0.7077020222189052람다 값 : 0.4325208070425582



![png](output_52_5822.png)


    100%|█████████████████████████████████████████████████████████████████████████████▋| 1941/1948 [49:39<00:07,  1.04s/it]cv 값 : 0.32187908448497676람다 값 : 1.4679002505670344



![png](output_52_5825.png)


    100%|█████████████████████████████████████████████████████████████████████████████▊| 1942/1948 [49:40<00:06,  1.05s/it]cv 값 : 0.42567105149960016람다 값 : 0.7895711925610938



![png](output_52_5828.png)


    100%|█████████████████████████████████████████████████████████████████████████████▊| 1943/1948 [49:41<00:05,  1.07s/it]cv 값 : 0.27316569268065505람다 값 : 1.728422949931239



![png](output_52_5831.png)


    100%|█████████████████████████████████████████████████████████████████████████████▊| 1944/1948 [49:42<00:04,  1.07s/it]cv 값 : 0.2510928101217064람다 값 : 2.4391470602841574



![png](output_52_5834.png)


    100%|█████████████████████████████████████████████████████████████████████████████▉| 1945/1948 [49:43<00:03,  1.07s/it]cv 값 : 0.34410079203316374람다 값 : 1.3894381811821208



![png](output_52_5837.png)


    100%|█████████████████████████████████████████████████████████████████████████████▉| 1946/1948 [49:44<00:02,  1.04s/it]cv 값 : 0.34781444040382326람다 값 : -0.0426646486766427



![png](output_52_5840.png)


    100%|█████████████████████████████████████████████████████████████████████████████▉| 1947/1948 [49:45<00:01,  1.09s/it]cv 값 : 0.35900575840695115람다 값 : -0.024265119338806353



![png](output_52_5843.png)


    100%|██████████████████████████████████████████████████████████████████████████████| 1948/1948 [49:46<00:00,  1.53s/it]

