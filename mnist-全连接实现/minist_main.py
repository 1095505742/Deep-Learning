# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:31:26 2018

@author: 李欣芮
"""

import minist_backward
import tensorflow.examples.tutorials.mnist.input_data as input_data

#定义主函数            
def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    minist_backward.backward(mnist)
    
#定义函数入口
if __name__=='__main__':
    main()