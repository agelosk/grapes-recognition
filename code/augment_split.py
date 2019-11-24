import numpy as np

from PIL import Image,ImageEnhance

import os,math,shutil,random

from Automold_Library import Automold as am
from Automold_Library import Helpers as hp
from functions import printProgressBar


dir = './dataset/data'
path = "/images_masks"
path2 = "/augmented"

path_train = './dataset/train'
path_val = './dataset/val'
#path_test = './dataset/test'

#Create necessary folders
try:
    os.mkdir(dir+path)
    os.mkdir(dir+path+path2)
    os.mkdir(path_train)
    os.mkdir(path_val)
    #os.mkdir(dir+path_test)
except OSError:
    print ("Creation of a directory failed!")
else:
    print ("Successfully created the directories")

# Keep images with annotated masks
for f in os.listdir(dir):
    if ('.jpg' in f):
        image = f[:-4]
        for f1 in os.listdir(dir):
            if (image+'.npz' == f1):
                shutil.copy2(dir+'/'+f, dir+path+'/'+f)
                shutil.copy2(dir+'/'+f1, dir+path+'/'+f1)
                break


#Augment dataset
# Each image is augmented with 11 more images:
# 4 with shifted Brightness: 0.25,0.5 (darker) and 2,3 (brighter)
# 4 with shifted Contrast: 0.25,0.5 (lower) and 3,6 (higher)
# 2 with rain added: 1 with heavy rain and one with terrential
# 1 with fog added: 0.3

print("Augmenting Dataset... ")
printProgressBar(0, len(os.listdir(dir+path))//2, prefix = 'Progress:', suffix = 'Complete', length = 50)
t = 1
for f in os.listdir(dir+path):
    if ('.jpg' in f):
        image = f[:-4]
        for f1 in os.listdir(dir+path):
            if (image+'.npz' == f1):
                mask = f1[:-4]
                break

        img1 = Image.open(dir+path+'/'+f)
        img2 = hp.load_images(dir+path+'/'+f)

        img1.save(dir+path+path2+'/'+image+'_1.jpg')

        foggy = am.add_fog(img2[0],fog_coeff=0.3)
        rainy1 = am.add_rain(img2[0],rain_type='heavy')
        rainy2 = am.add_rain(img2[0],rain_type='torrential')

        im = Image.fromarray(foggy)
        im.save(dir+path+path2+'/'+image+'_2.jpg')

        im = Image.fromarray(rainy1)
        im.save(dir+path+path2+'/'+image+'_3.jpg')

        im = Image.fromarray(rainy2)
        im.save(dir+path+path2+'/'+image+'_4.jpg')

        enhancer1 = ImageEnhance.Brightness(img1)
        enhancer2 = ImageEnhance.Contrast(img1)

        i = 5
        for b in [0.25,0.5,2,3]:
            imgi = enhancer1.enhance(b)
            imgi.save(dir+path+path2+'/'+image+'_'+str(i)+'.jpg')
            i = i + 1

        for c in [0.25,0.5,3,6]:
            imgi = enhancer2.enhance(c)
            imgi.save(dir+path+path2+'/'+image+'_'+str(i)+'.jpg')
            i = i + 1

        for j in range(1,13):
            shutil.copy2(dir+path+'/'+f1,dir+path+path2+'/'+mask+'_'+str(j)+'.npz')

        printProgressBar(t, len(os.listdir(dir+path))//2, prefix = 'Progress:', suffix = 'Complete', length = 50)
        t = t + 1

#Split dataset into training and validation set with 90%-10% ratio

x_list = []
for f in os.listdir(dir+path+path2):
    if ('.jpg' in f):
        x_list.append(f)

random.shuffle(x_list)

split_index = math.floor(len(x_list)*0.90)
x_train = x_list[:split_index]
x_val  = x_list[split_index:]

for f in x_train:
    shutil.move(dir+path+path2+'/'+f, path_train+'/'+f)
    shutil.move(dir+path+path2+'/'+f[:-4]+'.npz', path_train+'/'+f[:-4]+'.npz')

for f in x_val:
    shutil.move(dir+path+path2+'/'+f, path_val+'/'+f)
    shutil.move(dir+path+path2+'/'+f[:-4]+'.npz', path_val+'/'+f[:-4]+'.npz')

shutil.rmtree(dir+path)
