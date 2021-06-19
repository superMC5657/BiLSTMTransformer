import pandas as pd
import random


#读取排取的数据
data = pd.DataFrame()
for i in range(1,5):
    tmp = pd.read_excel('./weibo/data/{i}.xlsx'.format(i=i))
    if i ==0 :
        data = tmp
    else:
        data = pd.concat([data,tmp])

data = data.reset_index(drop=True)


#只保留评论内容和标签
data = data[['评论内容','标签']]
data = data.rename(columns = {'评论内容':'text','标签':'label'})

#对标签进行减一，因为torch的标签是从0开始的
data['label'] = data['label']  - 1


positive_data = data[data['label']==0]
negative_data = data[data['label']==1]

#对正样本下采样，保持与负样本数量一致
positive_data = positive_data.sample(n = negative_data.shape[0],random_state= 2021)
data = pd.concat([positive_data,negative_data])
data = data.reset_index(drop=True)


#随机 划分训练集（0.8），测试集（0.1），验证集（0.1）
train_index = int(data.shape[0] * 0.8)
text_index = int(data.shape[0] * 0.9)
random_index = list(range(0,data.shape[0]))
random.shuffle(random_index)
train_data = data.iloc[random_index[0:train_index]]
val_data = data.iloc[random_index[train_index:text_index]]
test_data = data.iloc[random_index[text_index:]]


#保存至train.txt，dev.txt，test.txt
train_data.to_csv('./weibo/data/train.txt', sep='\t',index=False, header=None)
val_data.to_csv('./weibo/data/dev.txt', sep='\t',index=False, header=None)
test_data.to_csv('./weibo/data/test.txt', sep='\t',index=False, header=None)

