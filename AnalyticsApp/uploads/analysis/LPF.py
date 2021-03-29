from scipy import signal 
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_path = "/Users/watanabeakihisa/Documents/mitou_jr/data/2019_8:22/neo_to_koyo/running_data.csv"#input("File Path =>")
data = pd.read_csv(data_path, header=None,dtype=float) #CSVファイルの読み込み
data = np.array(data)
data = data[:,3] 

class IIR_LPF:
    def __init__(self):
        self.cf_L = 5#カットオフ周波数[Hz]
        self.fsmp = 50 #サンプリング周波数 [Hz]
        self.fnyq = self.fsmp/2. #ナイキスト周波数 [Hz]
        self.order =2
        self.dt = 0.02

    def filtering(self, data):
        b, a = signal.iirfilter(N=self.order, Wn=self.cf_L/self.fnyq, btype='lowpass', analog=False, ftype="butter")
        w, h = signal.freqz(b, a, 50)
        gain = 20*np.log(abs(h))
        filtered_data = signal.filtfilt(b, a, data)
        return w,gain, filtered_data

    def fft(self,data):
        N = len(data)
        t = np.arange(0, N*self.dt, self.dt) #時間軸
        fq = np.linspace(0, 1.0/self.dt, N) #周波数軸
        freq = np.fft.fftfreq(N, d=self.dt) # 周波数
        # フーリエ変換
        F = np.fft.fft(data)
        F_abs = np.abs(F) # 複素数を絶対値に変換

        return freq, F_abs
    
