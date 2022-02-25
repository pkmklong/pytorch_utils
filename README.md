# pytorch_utils (WIP)

<b>Installation</b>

    $ python3 -m pip install git+https://github.com/pkmklong/pytorch_utils.git
 
<i>Synthetic Data Generation</i>

```python
from pytorch.utils import MockDataset

train_dataset = MockDataset(
    features=5,
    pos_n=500,
    neg_n=1000,
    pos_mean=150,
    pos_std=50,
    neg_mean=200,
    neg_std=80
)

train_dataset.data.shape
>> torch.Size([2000, 5])

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
<img src="https://github.com/pkmklong/pytorch_utils/blob/main/images/demo_data.png" height="400" class="center" title="Synthetic Data Plotting">

```python
test_dataset = MockDataset(
    features=5,
    pos_n=1000,
    neg_n=1000,
    pos_mean=150,
    pos_std=50,
    neg_mean=200,
    neg_std=80
)
test_data_loader = data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False) 
eval_model(model, test_data_loader)
>> Accuracy of the model: 90.50%
```

<i>Data to seq conversion</i>
```python
train_dataset = MockDataset(
    features=10,
    pos_n=100,
    neg_n=100,
    pos_mean=150,
    pos_std=50,
    neg_mean=200,
    neg_std=80
)
df = train_dataset.to_df(row_ids=True, label=True)
exclude_cols = ["row_ids", "label"]

df_seq = sort_to_sequence(df, key_col="row_ids", exclude_cols=exclude_cols)

df_seq
>> array([['col_0', 'col_3', 'col_7', ..., 'col_6', 'col_8', 'col_1'],
>>        ['col_0', 'col_8', 'col_5', ..., 'col_7', 'col_3', 'col_6'],
>>        ['col_0', 'col_5', 'col_3', ..., 'col_4', 'col_8', 'col_7'],
>>        ...,
>>        ['col_0', 'col_1', 'col_8', ..., 'col_3', 'col_4', 'col_5'],
>>        ['col_7', 'col_1', 'col_6', ..., 'col_3', 'col_4', 'col_8'],
>>        ['col_8', 'col_2', 'col_9', ..., 'col_1', 'col_6', 'col_5']],
>>       dtype=object)
```

<i>Model Testing</i>
```python
model = MyModule(n_input=5)
loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_data_loader = data.DataLoader(train_dataset, batch_size=20, shuffle=True)
train_model(model, optimizer, train_data_loader, loss_module)
```
<img src="https://github.com/pkmklong/pytorch_utils/blob/main/images/progress_bar.png" height="50" class="center" title="Model Training Progress Bar">

