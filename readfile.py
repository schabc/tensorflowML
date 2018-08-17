# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:35:57 2018

@author: schabc
""" 
import os  

data_path = "data/cifar-10-python/cifar-10-batches-py/"  
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.html':  
               L.append(os.path.join(root, file))  
    return L  

print(file_name(data_path))
#其中os.path.splitext()函数将路径拆分为文件名+扩展名