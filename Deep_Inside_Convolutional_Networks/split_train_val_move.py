import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle
import shutil

list_path = './'
train_list = 'newFood724_train.txt'
val_list = 'newFood724_val.txt'
# val_ratio = 0.8

src_dir = 'images'
des_dir = 'val'
os.mkdir(des_dir)

class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names

lst.sort()
print(lst)

with open(os.path.join(list_path,'food_id.txt'), 'w') as f:
    for ind in range(0,len(lst)):
        f.writelines(str(ind) + ":" + lst[ind] + '\n')

f_train = open(os.path.join(list_path, train_list), 'w')
f_val = open(os.path.join(list_path, val_list), 'w')

for ind in range(0,len(lst)):
    os.mkdir(des_dir + '/' + lst[ind])
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    shuffle(sblst)
    print(len(sblst))
    cnt=0
    for pic_name in sblst:
        filepath_src= src_dir + '/' + lst[ind] + '/' + pic_name
        val_path= des_dir + '/' + lst[ind] + '/' + pic_name
        if pic_name.endswith('.db') == True:
            continue
        else:
            cnt+=1
            if cnt <= 10:
                f_val.writelines(val_path + " " + str(ind) + '\n')
                # move files
                shutil.move(filepath_src, val_path)
            else:
                f_train.writelines(filepath_src + " " + str(ind) + '\n')

f_train.close()
f_val.close()
