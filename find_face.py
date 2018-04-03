#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:55:39 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import face_recognition as fr
import os
import time
 
# how to get face   
def find_and_save(face_folder):  # we can import image folder and output faces
    path = face_folder + '/face_ex'
    if not os.path.exists(path):  # create a new folder
        os.makedirs(path)
    folder = os.listdir(face_folder)  # all file in folder
    start = time.time()
    for jpg in folder:  # get jpg
        find1 = jpg.find('.jpg')
        find2 = jpg.find('.JPG')
        if find1 == -1 and find2 == -1:  # no image
            pass
        else:
            find = max(find1, find2)  #jpg or JPG
            jpg = jpg[:find]  #image name
            name = face_folder + '/' + jpg + '.jpg'
            # laod image
            image = fr.load_image_file(name)  # dtype = 'uint8'
            # find all face
            face_locations = fr.face_locations(image)  # [(), (),...., ()]
            # cut and save
            i = 0
            for face_location in face_locations:
                i = i + 1
                up, right, down, left = face_location
                # cut
                face = image[up:down, left:right]
                # paint
                face = Image.fromarray(face)
                # resize
                face = face.resize((128, 128))
                # save
                face.save(face_folder + '/face_ex/' + jpg + '_' + str(i) + '.jpg')
    end = time.time()
    return print('Total {0:.4f} s.'.format(end - start))
              
































































