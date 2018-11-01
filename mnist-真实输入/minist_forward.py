# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 23:13:15 2018

@author: 李欣芮
"""
#当如相应模块
import tensorflow as tf


#输入的图片训练集为784维
INPUT_NODE=784
#训练集的标签为10维，10分类，10个数索引号的概率
OUTPUT_NODE=10
#定义了隐藏的节点个数
LAYER1_NODE=500

#定义生成权重函数
def get_weight(shape,regularizer,name):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1),name=name)
    #加入正则化，参数惩罚
    if regularizer !=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

#定义生成偏置函数        
def get_bias(shape,name):
    b=tf.Variable(tf.zeros(shape),name=name)
    return b

def forward(x,regularizer):
    #第一层隐藏层
    w1=get_weight([INPUT_NODE,LAYER1_NODE],regularizer,'w1')
    b1=get_bias([LAYER1_NODE],'b1')
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)
    #第二层隐藏层
    w2=get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer,'w2')
    b2=get_bias([OUTPUT_NODE],'b2')
    y=tf.matmul(y1,w2)+b2
    #返回结果
    return y




    
    
    
    