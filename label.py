#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:29:41 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
import csv
import os

# import label data
table = csv.reader(open('/Users/zt/Desktop/face/label.csv', encoding='utf-8'))

# labels
labels = []
for row in table:
    row = row[0]  # get str
    label = row.rfind('\t')  # get label
    label = row[label + 1:]
    labels.append(label)
# no header
labels = labels[1:]


path = '/Users/zt/Desktop/face/data/face_ex'
for i in range(1,501):
    label = str(round(float(labels[i - 1])))
    os.rename(path + '/SCUT-FBP-' + str(i) + '_1.jpg', path + '/' + str(i) + '-' + label + '.jpg')






































