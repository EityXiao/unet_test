import os
import numpy as np
dir1 = sorted(os.listdir('./label1'))
dir2 = sorted(os.listdir('./label2'))
dir3 = sorted(os.listdir('./label3'))
print(len(dir1), len(dir2), len(dir3))
x = []
y = []
for ind in range(len(dir1)):
    i, j, k = dir1[ind], dir2[ind], dir3[ind]
    tmp_x = []
    tmp_y = []
    if i==j and i==k and j==k and '.txt' in i:
        d1 = open(os.path.join('./label1', dir1[0])).readlines()
        d2 = open(os.path.join('./label2', dir1[0])).readlines()
        d3 = open(os.path.join('./label1', dir3[0])).readlines()
        for ind_ in range(19):
            x1, y1 = map(int, d1[ind_].strip('\n').split(','))
            x2, y2 = map(int, d2[ind_].strip('\n').split(','))
            x3, y3 = map(int, d3[ind_].strip('\n').split(','))
            if x3 == 1 and y3 == 1:
                tmp_x.append(x1)
                tmp_y.append(y1)
            else:
                dis1 = (x1-x3)**2+(y1-y3)**2
                dis2 = (x2-x3)**2+(y2-y3)**2
                dis3 = (x2-x1)**2+(y2-y1)**2
                if dis1 <= dis2 and dis1 <= dis3:
                    tmp_x.append((x1+x3)//2)
                    tmp_y.append((y1+y3)//2)
                elif dis2 < dis1 and dis2 <= dis3:
                    tmp_x.append((x2+x3)//2)
                    tmp_y.append((y2+y3)//2)
                else:
                    tmp_x.append((x2+x1)//2)
                    tmp_y.append((y2+y1)//2)
    x.append(tmp_x)
    y.append(tmp_y)  
import json 
with open('target.json', 'w') as f:
    json.dump({'x':x,'y':y}, f)
         




        
