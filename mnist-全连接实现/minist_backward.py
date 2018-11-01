# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:26:26 2018

@author: 李欣芮
"""
#导入相应的模块
import tensorflow as tf
#导入并下载mnist数据集
#from tensorflow.examples.tutorials.mnist import input_data
#导入前向传播模块
import minist_forward
#导入系统模块，系统调用（例如打开关闭写入读取文件等操作）
import os


#一次喂如多少张图片
BATCH_SIZE=200
#初始学习率
LEARNING_RATE_BASE=0.1
#学习衰减率
LEARNING_RATE_DECAY=0.99
#正则化系数
REGULARIZER=0.0001
#训练的轮数
STEPS=50000
#滑动平均衰减率
MOVING_AVERAGE_DECAY=0.99
#模型存储路径
MODEL_SAVE_PATH="./model/"
#模型名称
MODEL_NAME="mnist_model"

def backward(mnist):
    
    #生成输入占位，标准输出占位
    x=tf.placeholder(tf.float32,shape=[None,minist_forward.INPUT_NODE])
    y_=tf.placeholder(tf.float32,shape=[None,minist_forward.OUTPUT_NODE])
    #前向计算
    y=minist_forward.forward(x,REGULARIZER)
    #轮数计数器
    global_step=tf.Variable(0,trainable=False)
    
    #损失函数(这里求交叉熵，对标准值)
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    #求标平均值
    cem=tf.reduce_mean(ce)
    #正则化后的losse值
    loss=cem+tf.add_n(tf.get_collection("losses"))
    
    #学习率的指数衰减
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    
    #定义训练过程
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    #定义滑动平均
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')
    
    #实例化Saver    
    saver=tf.train.Saver()
    
    #建立计算会话
    with tf.Session() as sess:
        #初始化所有变量
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
      
        
        #开始训练
        for i in range(STEPS):
            #文件中读取的mnist从它的下一组训练样本开始
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            #训练结果赋值
            _, loss_value, step, b=sess.run([train_op ,loss ,global_step,'b2:0'], feed_dict={x:xs,y_:ys})
            #每训练1000轮打印一次训练的轮数和误差值，并且保存当前训练的情况
            if i % 1000 == 0:
                
                print("After %d training step(s), loss on training batch is %g." % (step,loss_value))
                #存储器，进行存储，将当前的计算模型
                saver.save(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    