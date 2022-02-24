# pytorch_utils (WIP)

<b>Installation</b>

    $ python3 -m pip install git+https://github.com/pkmklong/pytorch_utils.git
 
<i>Synthetic Data Generation</i>

```python
from pytorch.utils import MockDataset

train_dataset = MockDataset(
    features=5,
    pos_n=100,
    neg_n=100,
    pos_mean=150,
    pos_std=80,
    neg_mean=200,
    neg_std=100
)
    
train_dataset.data
>> tensor([[120.8692,  96.8306,  77.1077,  ..., 288.5028, 139.9560,  97.9168],
>>         [ 41.1911, -82.3300, 133.0649,  ..., 295.8810, 375.1372, 178.4848],
>>         [133.8616,  10.1892, 116.5640,  ..., 117.4343, 165.8598, 385.8340],
>>         ...,
>>         [226.8078,  56.6908,  68.1813,  ..., 185.2966, 166.0162, 224.8710],
>>         [243.6762,  99.0662, 165.8202,  ..., 169.0571, 160.4391, 141.3412],
>>         [  7.1814, 268.7216, 274.3542,  ..., 308.0595, 421.0810, 193.6014]],
>>        dtype=torch.float64)

train_dataset.data.shape
>> torch.Size([200, 5])

train_dataset.label
>> tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
>>        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
>>        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
>>        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
>>        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
>>        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
>>        0., 0.], dtype=torch.float64)

df = train_dataset.to_df(row_ids=True, label=True)
print(df.head(5))
>>         col_0       col_1       col_2       col_3       col_4  row_ids  label
>> 0  148.404175   26.659887  182.252655   88.354713   80.416580        0    1.0
>> 1  177.925339   49.712185  192.029495  195.813110   88.991829        1    1.0
>> 2   30.146475  207.450897  193.121506  156.405838   92.489151        2    1.0
>> 3  335.195282  222.256516  133.536621   34.579933  215.421402        3    1.0
>> 4  351.968506  375.252045  112.089806  200.617950  175.515839        4    1.0  


visualize_data(train_dataset.data, train_dataset.label, x1=1, x2=2)
plt.show()
```
<img src="https://github.com/pkmklong/pytorch_utils/blob/main/images/demo_data.png" height="200" class="center" title="Synthetic Data Plotting">

