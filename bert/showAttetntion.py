'''
Author: LawsonAbs
Date: 2020-12-17 10:58:34
LastEditTime: 2020-12-17 11:03:14
FilePath: /bert/showAttetntion.py
'''

''' 生成tokens并用 bert处理'''
def process(text,toknizer,bert):
    inputs = tokenizer(text,return_tensors='pt')
    #print(inputs)
    output = bert(**inputs,output_attentions=True) # 因为需要输出attention score 的值，所以这里用了 output_attentions 参数
    return output


'''得到某层某个头的attention。【因为attentions 是一个tuple】
01.参数：
output:bert处理后的结果
layer:代表取哪一层的attention
head:代表取哪一head
'''
def getAttention(output,layer,head):
    res = output[2]
    # print(type(res))# <class tuple>
    # print(len(res)) # 12 。解释一下是12的原因：Number of hidden layers in the Transformer encoder 是12
    # print(res) #(tensor()...,tensor()...)

    layer_attention_score = res[layer] # 得到 attention 中的最后一个tuple，也就是最后一层的attention 值
    #print(type(res)) # 去掉一维之后仍然是tuple <class 'tuple'>
    #print(attention_score.size()) # torch.Size([1, 12, 28, 28])。 这个size = [batch_size,num_head,seq_len,seq_len]

    layer_attention_score.squeeze_(0) # 去掉第一维的1
    #print(attention_score.size()) # 因为有12个head，这里只取第一个
    layer_head_attention_score = layer_attention_score[head,:,:]
    #print(layer_head_attention_score.size())
    #print(layer_head_attention_score)
    return layer_head_attention_score 

''' 找出和 making 最相关的几个单词
01.得到前几个相关word
02.参数：
first_attention_score 是一个tensor的二维矩阵
'''
def getTopWord(first_attention_score):
    #making => 对应的下标是18。所以拿到18这行的行向量
    score = first_attention_score[18].tolist()  
    #print(score)

    # 将上面的list转为dict，方便后面根据dict排序，然后找出相似度最高的index
    dic = {}
    for i in range(len(score)):
        dic[i] = score[i]
    #print(dic)

    # 将得到的dic 排序，值高的放在最前面
    res = sorted(dic.items(),key = lambda dic:dic[1],reverse=True)
    res1 = dict(res) # 排序后是list，再转为dict
    keys = res1.keys() 
    #print(keys)

    # 得到tokens的id，并转为list
    tokens = inputs['input_ids'].squeeze_(0).tolist()
    i= 0
    for _ in keys:# 将token转换成相应的word
        if i < 5:
            print(tokenizer.convert_ids_to_tokens( tokens[_]))
            i+=1
        else:
            break



''' 可视化attention 的结果'''
def drawAttention(inputs,first_attention_score):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as ticker
    import torch

    #将first_attention_score 的grad属性去除 => 使用detach()，并转为numpy
    res2 = first_attention_score.detach().numpy() 
    temp = inputs["input_ids"][0] # 得到input_ids，为了将其转换成tokens
    print(temp)
    tokens= tokenizer.convert_ids_to_tokens(temp)
    #print(type(tokens)) # <class 'list'>
    #print(tokens) # ['[CLS]', 'it', 'is', ...]

    # tokens就是我们的横纵坐标(标签)
    df = pd.DataFrame(res2, columns=tokens, index=tokens)
    fig = plt.figure(figsize = (40,40))# 画图，这里调整了图片的大小，否则会因为word太长导致文字重叠
    ax = fig.add_subplot(111) 
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

    plt.show()


import torch as t
from transformers import BertModel,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-uncased")
bert = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-uncased")
#text ="It is in this spirit that a majority of American governments have passed new laws since 2008 making the registration or voting process more difficult."
text = "I used to be a banker, but I lose interest."
layer = 4
head=3

output = process(text,tokenizer,bert)

# print(text)
# print(output)
layer_head_attention_score = getAttention(output,layer,head)
inputs = tokenizer(text,return_tensors='pt')
drawAttention(inputs,layer_head_attention_score)