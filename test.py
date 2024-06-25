import numpy as np

if __name__ == '__main__':
    data=np.load('./leonard/data/vertex200m.npy',allow_pickle=True)
    print(data.shape)