from scipy import signal
import numpy as np
from .remove_noise import remove_noise



# FFTデータからピークを自動検出
def find_peaks(data,N):
    maximal_idx = signal.argrelmax(data, order=15) #一次元配列なので[0]は省略
    #print(maximal_idx.to_lis)
    maximal_idx = np.array(maximal_idx)
    maximal_idx = np.squeeze(maximal_idx)
    #print(type(maximal_idx))
    print(maximal_idx)

    return maximal_idx
