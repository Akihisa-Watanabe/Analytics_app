import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_noise(original_data,dt,N):
    data = np.array(original_data)#データ読み込み
    t = np.arange(0, N*dt, dt) #時間軸
    fq = np.linspace(0, 1.0/dt, N) #周波数軸

    # フーリエ変換
    F = np.fft.fft(data)
    F_abs = np.abs(F) # 複素数を絶対値に変換
    F_abs_amp = F_abs / N * 2 # 振幅をもとの信号に揃える

    #----------フィルタリング-------------------
    F2 = np.copy(F) # FFT結果コピー
    fc = np.argmax(F_abs_amp)+10# カットオフ（周波数）
    F2[(fq > fc)] = 0 # カットオフを超える周波数のデータをゼロにする（ノイズ除去）

    #逆変換
    # 周波数でフィルタリング（ノイズ除去済み）-> IFFT
    F2_ifft = np.fft.ifft(F2) # IFFT
    filtered_data = F2_ifft.real * 2 # 実数部の取得、振幅を元スケールに戻す

    return filtered_data #ノイズ除去済みデータ、

if __name__ == '__main__':
    remove_noise(data, dt,N)
