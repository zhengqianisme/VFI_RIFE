import numpy as np
if __name__ == '__main__':
    # a=np.load('../arXiv-2020-RIFE/0.npz/ft0ft1.npy',encoding="bytes")
    a = np.load('../arXiv-2020-RIFE/dataset_train/0/ft0ft1.npy', encoding="bytes")
    # a = np.load('../arXiv-2020-RIFE/dataset_train/0/i0i1gt.npy', encoding="bytes")
    print(a.shape)