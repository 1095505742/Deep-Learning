# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:44:19 2018

@author: 李欣芮
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import minist_forward
import minist_backward
TEST_INTERVAL_SECS=10

def test(mnist):
    #使用到Graph.as_default() 的上下文管理器（ context manager），它能够在这个上下文里面覆盖默认的图
    with tf.Graph().as_default() as g:
        
        #生成测试集的占位
        x=tf.placeholder(tf.float32,shape=[None,minist_forward.INPUT_NODE])
        y_=tf.placeholder(tf.float32,shape=[None,minist_forward.OUTPUT_NODE])
        y=minist_forward.forward(x,None)
        
        #实例化影子容器，容器中存储着影子值得地址
        ema=tf.train.ExponentialMovingAverage(minist_backward.MOVING_AVERAGE_DECAY)
        #将影子值和本身变量进行绑定，本身变量名指向影子值的地址
        ema_restore=ema.variables_to_restore()
        #saver加载器加载的是变量的内存
        saver=tf.train.Saver(ema_restore)
        
        #准确预测值当相
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #tf.cast转换数据类型
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        while True:
            with tf.Session() as sess:
                #加载训练好的模型
                ckpt=tf.train.get_checkpoint_state(minist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    
                    print(ckpt.model_checkpoint_path)
                    #恢复会话
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    
                    #恢复轮数
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    #计算准确率
                    accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                    #打印提示
                    
                    print("after %s training step(s), test accuracy = %g" % (global_step,accuracy_score))
                   # print("The current training steps' w2 is:", sess.run(g.get_tensor_by_name("w2:0")))
                   # print("The current training steps' b2 is:", sess.run(g.get_tensor_by_name("b2:0")))
                 
                else:
                    print("No checkpoint file found!")
                    return
            #引入时间线程，睡眠5秒
            time.sleep(TEST_INTERVAL_SECS)
            
#定义主函数            
def main():
    #对于数据集采用one_hot格式
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)
    
    
#定义函数入口
if __name__=='__main__':
    main()