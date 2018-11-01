# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:11:20 2018

@author: 李欣芮
"""


import tensorflow as tf 
from PIL import Image
import numpy as np
import minist_backward
import minist_forward
import matplotlib.pyplot as plt


#恢复模型，导入处理后的图片
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        
        x = tf.placeholder(tf.float32, [None,minist_forward.INPUT_NODE])
        y = minist_forward.forward(x,None)
        preValue = tf.arg_max(y,1)
        
        variable_averages = tf.train.ExponentialMovingAverage(minist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(minist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
            
                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
               print("No checkpoint file found")
               return -1


#对需要进行预处理的图片进行处理
def pre_pic(preName):
    #打开图片
    img = Image.open(preName)
    plt.imshow(img)
    plt.show()
#    print(img)
    #用消除锯齿的方法重新定义图片大小，为28*28(满足尺寸要求)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    plt.imshow(reIm)
    plt.show()
#    print(reIm)
    #将图片使用convert('L')变成灰度图，并用np.array转换成矩阵的形式(满足颜色要求)
    im_arr = np.array(reIm.convert('L'))
#    print(im_arr)
    #设定阈值
    threshold = 50
    #对颜色进行反色处理，调节0为白，1为黑
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            #过滤点，设定的阈值，滤掉噪声
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255
#    print(im_arr)
    #将图片矩阵，改变形状为1行784列
    num_arr = im_arr.reshape([1,784])
#    print(num_arr)
    #对图片矩阵中的值转换为浮点数类型
    nm_arr = num_arr.astype(np.float32)
    #将图片中的元素转换成0到1的浮点数
    img_ready = np.multiply(nm_arr , 1.0/255.0)
   
#    print(img_ready)
    return img_ready



def application():
    #输入测试图片的数量
    testNum = int(input("input the number of test pictures："))
    for i in range(testNum):
        #输入图片的路径
        testPic = input("the path of test picture:")
        #进行图片的预处理
        testPicArr = pre_pic(testPic)
        #预测结果
        preValue = restore_model(testPicArr)
        print("The prediction number is:",preValue)
        
def main():
    application()
    
if __name__=='__main__':
    main()