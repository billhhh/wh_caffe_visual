import os
import os.path
from shutil import copyfile
from math import floor
from random import shuffle

list_path = './'
src_dir = 'images'
cnt_list = 'cnt_list.txt'

class_names = []
for filename in os.listdir(src_dir):
    path = os.path.join(src_dir, filename)
    if os.path.isdir(path):
        class_names.append(filename)

lst = class_names

lst.sort()
# print(lst)

cnt_txt = open(os.path.join(list_path, cnt_list), 'w')

total_cnt=0
for ind in range(0,len(lst)):
    sblst=os.listdir(os.path.join(src_dir,lst[ind]))
    shuffle(sblst)
    print(lst[ind]+": %20s"%str(len(sblst))+"\n")
    cnt_txt.writelines(lst[ind]+": %20s"%str(len(sblst))+"\n")
    total_cnt=total_cnt+len(sblst)
	
print("average: "+str(total_cnt/724))
cnt_txt.writelines("average: "+str(total_cnt/724))

cnt_txt.close()
