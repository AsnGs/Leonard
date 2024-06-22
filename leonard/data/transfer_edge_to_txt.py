import numpy as np

# 读入 edges200m.npy，然后转成 txt 存储
a=np.load('./leonard/data/edges200m.npy',allow_pickle=True)
for i in range(len(a)):
    a[i]=str(a[i])
f=open('edges200m.txt','w')
f.write('\n'.join(a))
