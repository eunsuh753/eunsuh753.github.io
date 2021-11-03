---
layout: post
title:  "시계열 모델링: FB's Prophet"

toc: true
toc_sticky: true

---

```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
cd /content/drive/MyDrive/금융빅데이터분석가/핀테크_데이터분석
```

    /content/drive/MyDrive/금융빅데이터분석가/핀테크_데이터분석


## 1. Airline Passenger 예측

### Facebook의 Prophet을 사용하여 월간 시계열 모델링 기법 실습

- Air passengers는 1949~1960년까지의 고전적인 시계열 데이터 세트


```python
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv('data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.columns = ['ds', 'y']
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-01</td>
      <td>112</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-01</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-01</td>
      <td>132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-01</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-01</td>
      <td>121</td>
    </tr>
  </tbody>
</table>

</div>



- 단위: 천명
- 데이터는 월 단위로 보고되며 매월 1회 측정됨.


```python
# 모델 인스턴스화, 피팅
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365 * 5) # 5년 예측
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
```

    INFO:numexpr.utils:NumExpr defaulting to 2 threads.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



![output_6_1](https://user-images.githubusercontent.com/62747570/140023588-ea5fdb15-bef7-402d-bfc3-d400c54636b5.png)




- 매월 1일에 계절성 계산을 적절히 적용할 수 있지만, 나머지 날에는 그래프에서 보듯이 무엇을 해야 하는지 잘 모르고 예측할 수 없는 방식으로 계절성 곡선을 과적합하였음.  



```python
# Prophet에게 훈련된 월별 데이터와 일치하도록 월별 규모로만 예측하도록 지시하여 문제 해결
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=12 * 5, freq='MS') # 연간 12개의 예측만, 월별 규모로만
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



![output_8_1](https://user-images.githubusercontent.com/62747570/140023593-681cdca6-846f-4381-ba53-32b4fca2d2f0.png)




- 월간 빈도로 미래 예측
- 기본적으로 빈도는 매일의 경우 'D'로 설정되며 period는 예측하려는 일 수임.
- freq를 다른 설정으로 변경할 때마다 기간을 동일한 눈금으로 설정해야 함.

---

## 2. Divvy bike share program

### Facebook의 Prophet을 사용하여 서브 일간(sub-daily) 시계열 모델링 기법 실습

### (sub-daily: 하루 내에도 패턴이 존재함)

- 미국 일리노이주 시카고에 있는 Divvy 자전거 공유 프로그램 데이터
  - 2014년 초부터 2018년 말까지 시간당 자전거 이용 횟수 포함
  - 매년 매우 강한 계절성과 함께 일반적으로 증가하는 추세


```python
data = pd.read_csv('data/divvy_hourly.csv')
df = pd.DataFrame({'ds': pd.to_datetime(data['date']),'y': data['rides']})
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-01-01 01:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-01-01 02:00:00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-01-01 03:00:00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-01-01 04:00:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-01-01 07:00:00</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

</div>




```python
plt.scatter(df['ds'], df['y'], alpha=.5, s=1)
plt.tight_layout()
plt.show()
```

![output_13_0](https://user-images.githubusercontent.com/62747570/140023596-fa61333a-1fbb-4fd3-9c16-2046a33bc623.png)




- 시간별 데이터이고 밤새 타는 횟수가 매우 적기 때문에 데이터는 최저점에서 측정 밀도를 보여줌
- Prophet이 최소 2일의 데이터를 보고 데이터 사이의 간격이 1일 미만인 경우, 이는 일간 계절성에 맞을 것임 (뭔말?)


```python
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365 * 24, freq='h') # 매시간, 시간별이므로 기간 조정
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
```

![output_15_0](https://user-images.githubusercontent.com/62747570/140023598-addb57a9-c176-40ba-b88c-9bc91ea5db0c.png)




- 다소 큰 불확실성이 보임


```python
# 구성 요소 플롯
fig2 = model.plot_components(forecast)
plt.show()
```

![output_17_0](https://user-images.githubusercontent.com/62747570/140023600-5bc9a165-12d9-48b7-a4a7-85c7f5ce1ba7.png)




- 복잡한 계절성
  - 계절성 자체는 연중 계절성  
  - 계절성 속의 계절성  
  - 일별 계절성은 낮에 상승하고 밤에 감소하지만 증가량은 연중 시기에 따라 달라짐  
  - Prophet은 이러한 유형의 계절성을 포착하지는 못하기 때문에 예측의 불확실성이 매우 큰 것임. (이를 제어하는 몇 가지 기술 배울 것임)  

- 주간 계절성
  - 단일 곡선
  - 일요일에서 일요일로 이동 -> 시간별 데이터의 보다 지속적인 특성을 반영하기 위한 것

- 연간 계절성
  - 꽤 물결 모양 (계절성과 관련된 푸리에 차수(Fourier Order)에서 다루게 됨

- 일별 계절성
  - 오전 8시경에 많이 타고 출근하는 것으로 보임
  - 오후 5시경에 많이 타고 퇴근하는 것으로 보임
  - 자정 직후에 올빼미족

** 음의 승차 횟수를 갖는 것은 불가능하지만 모델은 일부 음수 값을 예측하는 문제가 있음


## 규칙적인 간격이 있는 데이터 사용 (data with regular gaps)

ex) 근무 시간, 개인 시간 및 수면 시간이 있는 사람이 데이터를 수집한 경우  
하지만 완벽한 주기로 측정값을 수집하는 것은 불가능할 수 있음.

- 이상치 관점에서 Prophet은 결측치를 처리하는 데 강력하지만 누락된 데이터가 일정한 간격으로 발생하면 Prophet은 그 간격동안 추정할 훈련 데이터가 전혀 없을 것임.
- 계절성은 데이터가 존재하는 기간 동안 제약을 받지만 갭 기간 동안 제약이 없으며, Prophet의 예측은 표시된 실제 데이터보다 훨씬 더 큰 변동을 볼 수 있음.


```python
# Divvy의 데이터가 매일 오전 8시에서 오후 6시 사이세만 수집되었다고 가정.
# 다음 시간 이외의 데이터 제거하여 시뮬레이션 가능
df = df[(df['ds'].dt.hour >= 8) & (df['ds'].dt.hour < 18)]

plt.scatter(df['ds'], df['y'], alpha=.5, s=1)
plt.tight_layout()
plt.show()
```

![output_20_0](https://user-images.githubusercontent.com/62747570/140023603-ca2421fc-0d3c-4428-99ea-5ded734595aa.png)




- 원데이터에 비해 y축 값이 낮으면서(?) 훨씬 더 희소함
- 야간 데이터 모두 손실


```python
# 이제 매일 오전 8시에서ㅓ 오후 6시 사이의 시간당 하나씩 10개의 데이터 포인트만 있음
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)
future = model.make_future_dataframe(periods=365 * 24, freq='h')
forecast = model.predict(future)
fig = model.plot(forecast)
plt.show()
```

![output_22_0](https://user-images.githubusercontent.com/62747570/140023606-be6181c9-e80e-42ce-9ff4-f8dbfe93e7de.png)




- 과거 훈련 데이터보다 미래 기간의 훨씬 더 넓은 일일 변동을 보여줌


```python
# 2018년 8월의 단 3일만 확대하여 재플로팅
fig = model.plot(forecast)
plt.xlim(pd.to_datetime(['2018-08-01', '2018-08-04']))
plt.ylim(-2000, 4000)
plt.show()
```

![output_24_0](https://user-images.githubusercontent.com/62747570/140023607-d9653904-4c93-47fc-ad18-3c8da93dd1b3.png)




- 오전 8시 이전에 승객이 몰리고 오전 8시에 국지적으로 정점을 찍는다.
- 정오에 슬럼프가 있었고 오후 6시 이후에 큰 피크가 나타난다.
- Prophet이 훈련 데이터가 없는 오전 8시 이전과 오후 6시 이후에 거친 예측을 한다는 점을 제외하면 원데이터의 예측과 거의 동일함
- 이 영역은 제약이 없으며 데이터가 존재하는 정오 기간 동안 방정식이 작동하는 거의 모든 패턴을 따를 수 있음.


```python
# future 훈련 데이터에 규칙적인 간격이 있었던 시간을 제외하도록 DataFrame 수정
from fbprophet.plot import add_changepoints_to_plot

future2 = future[(future['ds'].dt.hour >= 8) & (future['ds'].dt.hour < 18)]
forecast2 = model.predict(future2)
fig = model.plot(forecast2)
add_changepoints_to_plot(fig.gca(), model, forecast2)
plt.show()
```

![output_26_0](https://user-images.githubusercontent.com/62747570/140023609-6d8051ff-3fc3-4d64-998f-201228d13a56.png)




- 더 좁은 범위의 예측을 보여줌


```python
# 8월의 동일한 3일 다시 표시
fig = model.plot(forecast2, figsize=(10, 4))
plt.xlim(pd.to_datetime(['2018-08-01', '2018-08-04']))
plt.ylim(-2000, 4000)
plt.show()
```

![output_28_0](https://user-images.githubusercontent.com/62747570/140023610-e2ffaa26-0d5c-4ece-8a42-2a3a9e18c270.png)




- 오전 8시에서 오후 6시 사이는 같은 곡선으로 표시되지만 그 사이를 단순 직선으로 연결함


```python
# 일일 계절성 플로팅
from fbprophet.plot import plot_seasonality
plot_seasonality(model, 'daily', figsize=(10, 3))
plt.show()
```

![output_30_0](https://user-images.githubusercontent.com/62747570/140023611-96572462-6779-4780-90f9-0580526caa1e.png)




- 일일 계절성은 두 버전 모두 동일함

---

# 자동 추세 변화점(Automatic Trend Changepoint) 감지

- 추세 변경점: 모델의 추세 구성 요소가 갑자기 나타나는 시계열의 위치
- 변경점이 발생하는 여러 이유 존재 ex) Facebook은 일일 활성 사용자 수를 모델링하고 새로운 기능이 출시되면 추세의 급격환 변화를 확인할 수 있음, 항공사 승객 수는 규모의 경제로 인해 갑자기 변경될 수 있으므로 훨씬 저렴한 항공편 가능
- Divvy 데이터 세트에서 2년 후 성장이 둔화되는 것을 확인하였음. 이를 자세히 보자

### 기본 변경점 감지(Default Changepoint Detection)

- Prophet 세트 변경점이 발생할 수 있는 잠재적 날짜의 수를 먼저 지정하여 변경점 지정
- Prophet은 이러한 각 지점에서 변화의 크기를 계산하는 작업을 수행하여 해당 크기를 가능한 한 낮게 유지하면서 추세 곡선에 맞추려고 함.
- changepoint_prior_scale을 조정하여 유연성 조정 가능
- 계절성과 공휴일 모두 정규화를 위한 사전 척도가 있다는 이전의 주장 인식(?)
- Prophet의 기본 설정에서 이러한 잠재적인 변화점의 대부분의 크기는 거의 0에 가까우므로 추세 곡선에 미미한 영향을 미침



```python
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

df = pd.read_csv('data/divvy_daily.csv')
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
      <th>date</th>
      <th>rides</th>
      <th>temperature</th>
      <th>weather</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2014</td>
      <td>95</td>
      <td>19.483158</td>
      <td>rain or snow</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2014</td>
      <td>111</td>
      <td>16.833333</td>
      <td>rain or snow</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2014</td>
      <td>6</td>
      <td>-5.633333</td>
      <td>clear</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2014</td>
      <td>181</td>
      <td>30.007735</td>
      <td>rain or snow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2014</td>
      <td>32</td>
      <td>16.756250</td>
      <td>rain or snow</td>
    </tr>
  </tbody>
</table>

</div>




```python
df = df[['date', 'rides']]
df['date'] = pd.to_datetime(df['date'])
df.columns = ['ds', 'y']
```


```python
import numpy as np 

def set_changepoints(df, n_changepoints, changepoint_range):
  df = df.sort_values('ds').reset_index(drop=True) 
  hist_size = int(np.floor(df.shape[0] * changepoint_range)) 
  if n_changepoints + 1 > hist_size: 
    n_changepoints = hist_size - 1 
    print('n_changepoints greater than number of observations. Using {}.'.format(n_changepoints)) 
  if n_changepoints > 0: 
    cp_indexes = np.linspace(0, hist_size - 1, n_changepoints + 1).round().astype(np.int) 
    changepoints = df.iloc[cp_indexes]['ds'].tail(-1) 
  else: 
    changepoints = pd.Series(pd.to_datetime([]), name='ds') 
  return changepoints

changepoints = set_changepoints(df, 25, .8)

plt.figure(figsize=(10, 6))
plt.scatter(df['ds'], df['y'], c='#0072B2')
for cp in changepoints:
    plt.axvline(x=cp, c='gray', ls='--')
plt.xlabel('Date')
plt.ylabel('Rides per day')
plt.show()
```

![output_36_0](https://user-images.githubusercontent.com/62747570/140023613-c13a94d3-32be-4dc9-a51b-98cebfad4947.png)




- default로 Prophet은 25개의 잠재적인 변경점을 배치함
- 데이터의 크기를 결정하기 전에 25개의 위치가 이 이미지에서 수직 파선으로 표시됨


```python
# 곱셈 계절성 사용, 연간 계절성의 푸리에 차수 약간 유지(?)
model = Prophet(seasonality_mode='multiplicative',
                yearly_seasonality=4)
model.fit(df)
# futureDataFrame 지정없이 predict 호출 -> 과거값을 예측하지만 미래값은 예측하지 못하게 됨
forecast = model.predict()
```

    INFO:numexpr.utils:NumExpr defaulting to 2 threads.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.



```python
fig = model.plot(forecast)
# 중요한 변경점의위치 확인
# 첫 번째 인수: 변경점을 추가할 축 (get current axes), 두 번째 인수: 모델, 세 번째 인수: 예측
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()
```

![output_39_0](https://user-images.githubusercontent.com/62747570/140023615-58f33a3c-399f-4ca3-8e41-d9dfe306cc7a.png)




- 25개의 잠재적 변경점 중 5개가 실제로 중요하다고 결정했음
- 이 5개는 이 플롯에서 수직 파선으로 표시됨
- 첫 번째 변경점에서는 추세가 구부러지고 있다고 말하기 어렵지만. 다음 4개에서는 명확해짐.
- 각 변화점에서 추세의 기울기는 더 작아지도록 허용됨.

25개의 잠재적 변화점 각각의 크기는 model.params에 저장되지만 이러한 값들은 정규화되어 절대값은 의미가 없고 상대적 크기만 의미가 있다. 모델 parameter는 변경점 크기에 대한 키가 되는 'delta'와 함께 사전에 저장된다.


```python
print(model.params['delta'])
```

    [[-6.80598139e-08 -4.07802987e-08 -3.87616359e-08 -6.14254863e-03
      -1.49851989e-08 -7.77562608e-09  3.32923340e-07  3.48169660e-07
       2.77856376e-02  1.17257391e-07  1.53990866e-07 -4.33911330e-08
      -5.18705711e-03 -9.52778668e-02 -8.68189406e-02 -5.60180478e-02
      -3.39913071e-02 -4.62595330e-09 -1.11355235e-06 -4.83772637e-08
      -4.29852468e-08 -4.55749633e-08  5.32563788e-09 -2.60731418e-08
      -2.28944904e-09]]



```python
deltas = model.params['delta']
deltas = deltas.reshape(1,-1)
```


```python
fig, ax = plt.subplots(1,1, figsize=(20,6))
add_changepoints_to_plot(fig.gca(), model, forecast)
ax2 = ax.twinx()
ax2.bar(df['ds'], deltas, facecolor='blue', edgecolor='red', width=15, alpha=0.5)
plt.show()
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-18-cfe016add7ff> in <module>()
          2 add_changepoints_to_plot(fig.gca(), model, forecast)
          3 ax2 = ax.twinx()
    ----> 4 ax2.bar(df['ds'], deltas, facecolor='blue', edgecolor='red', width=15, alpha=0.5)
          5 plt.show()


    /usr/local/lib/python3.7/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
       1563     def inner(ax, *args, data=None, **kwargs):
       1564         if data is None:
    -> 1565             return func(ax, *map(sanitize_sequence, args), **kwargs)
       1566 
       1567         bound = new_sig.bind(ax, *args, **kwargs)


    /usr/local/lib/python3.7/dist-packages/matplotlib/axes/_axes.py in bar(self, x, height, width, bottom, align, **kwargs)
       2340         x, height, width, y, linewidth = np.broadcast_arrays(
       2341             # Make args iterable too.
    -> 2342             np.atleast_1d(x), height, width, y, linewidth)
       2343 
       2344         # Now that units have been converted, set the tick locations.


    <__array_function__ internals> in broadcast_arrays(*args, **kwargs)


    /usr/local/lib/python3.7/dist-packages/numpy/lib/stride_tricks.py in broadcast_arrays(subok, *args)
        256     args = [np.array(_m, copy=False, subok=subok) for _m in args]
        257 
    --> 258     shape = _broadcast_shape(*args)
        259 
        260     if all(array.shape == shape for array in args):


    /usr/local/lib/python3.7/dist-packages/numpy/lib/stride_tricks.py in _broadcast_shape(*args)
        187     # use the old-iterator because np.nditer does not handle size 0 arrays
        188     # consistently
    --> 189     b = np.broadcast(*args[:32])
        190     # unfortunately, it cannot handle 32 or more arguments directly
        191     for pos in range(32, len(args), 31):


    ValueError: shape mismatch: objects cannot be broadcast to a single shape



![output_44_1](https://user-images.githubusercontent.com/62747570/140023616-6470b11a-471c-4d44-96c6-e08997f8beb3.png)





```python

```
