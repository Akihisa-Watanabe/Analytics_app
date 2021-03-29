import numpy as np
from scipy.signal import argrelmax
from scipy import signal 
import pandas as pd
from itertools import islice
from .LPF import IIR_LPF as LPF
import statsmodels.api as sm
import io
import matplotlib
matplotlib.use('Agg')
import japanize_matplotlib
import matplotlib.pyplot as plt
import base64


class graph_plot():
    def __init__(self,data_path):
        self.data = pd.read_csv(data_path, header=None,dtype=float) #CSVファイルの読み込み
        self.data = np.array(self.data)
        self.data = self.data[:,3] 
        self.dt = 0.02
        self.t = np.arange(0, len(self.data)*self.dt, self.dt) #時間軸
        #--------------ノイズ除去------------------
        self.filter =LPF()
        self.w,self.gain, self.signal = self.filter.filtering(self.data)

    def window(self,seq, n):

        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result


    def SSA_anom(self, test, traject, w, ncol_t, ncol_h, ns_t, ns_h,
                normalize=False):

        H_test = np.array(
            tuple(x[:ncol_t] for x in self.window(test, w))[:w])  # test matrix
        H_hist = np.array(
            tuple(x[:ncol_h] for x in self.window(traject, w))[:w])  # trajectory matrix
        if normalize:
            H_test = (H_test - H_test.mean(axis=0,
                                        keepdims=True)) / H_test.std(axis=0)
            H_hist = (H_hist - H_hist.mean(axis=0,
                                        keepdims=True)) / H_hist.std(axis=0)
        Q, s1 = np.linalg.svd(H_test)[0:2]
        Q = Q[:, 0:ns_t]
        ratio_t = sum(s1[0:ns_t]) / sum(s1)
        U, s2 = np.linalg.svd(H_hist)[0:2]
        U = U[:, 0:ns_h]
        ratio_h = sum(s2[0:ns_t]) /sum(s2)
        anom = 1 - np.linalg.svd(np.matmul(U.T, Q),
                                        compute_uv=False
                                        )[0]
        return (anom, ratio_t, ratio_h)

    def SSA_CD(series, w, lag,
            ncol_h=None, ncol_t=None,
            ns_h=None, ns_t=None,
            standardize=False, normalize=False, fill=True):
        """
        Change Detection by Singular Spectrum Analysis
        SSA を使った変化点検知
        -------------------------------------------------
        w   : window width (= row width of matrices) 短いほうが感度高くなる
        lag : default=round(w / 4)  Lag among 2 matrices 長いほうが感度高くなる
        ncol_h: 履歴行列の列数 
        ncol_t: テスト行列の列数
        ns_h: 履歴行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        ns_t: テスト行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        standardize: 変換後の異常度の時系列を積分面積1で規格化するか
        fill: 戻り値の要素数を NaN 埋めで series と揃えるかどうか
        -------------------------------------------------
        Returns
        list: 3要素のリスト
            要素1: 2つの部分時系列を比較して求めた異常度のリスト
            要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率のリスト
        """
        if ncol_h is None:
            ncol_h = round(w / 2)
        if ncol_t is None:
            ncol_t = round(w / 2)
        if ns_h is None:
            ns_h = np.min([1, ncol_h])
        if ns_t is None:
            ns_t = np.min([1, ncol_t])
        if min(ncol_h, ncol_t) > w:
            print('ncol and ncol must be <= w')
        if ns_h > ncol_h or ns_t > ncol_t:
            print('I recommend to set ns_h >= ncol_h and ns_t >= ncol_t')
        start_at = lag + w + ncol_h
        end_at = len(series) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.SSA_anom(series[t - w - ncol_t:t],
                                series[t - lag - w - ncol_h:t - lag],
                                w=w, ncol_t=ncol_t, ncol_h=ncol_h,
                                ns_t=ns_t, ns_h=ns_h,
                                normalize=normalize)]
        anom = [round(x, 14) for x, r1, r2 in res]
        ratio_t = [r1 for x, r1, r2 in res]
        ratio_h = [r2 for x, r1, r2 in res]
        if fill == True:
            anom = [np.nan] * (start_at - 1) + anom
        if standardize:
            c = np.nansum(anom)
            if c != 0:
                anom = [x / c for x in anom]
        return [anom, ratio_t, ratio_h]


 



    def SSA_CD(self,series, w, lag,
            ncol_h=None, ncol_t=None,
            ns_h=None, ns_t=None,
            standardize=False, normalize=False, fill=True):
        """
        Change Detection by Singular Spectrum Analysis
        SSA を使った変化点検知
        -------------------------------------------------
        w   : window width (= row width of matrices) 短いほうが感度高くなる
        lag : default=round(w / 4)  Lag among 2 matrices 長いほうが感度高くなる
        ncol_h: 履歴行列の列数 
        ncol_t: テスト行列の列数
        ns_h: 履歴行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        ns_t: テスト行列から取り出す特異ベクトルの数. default=1 少ないほうが感度高くなる
        standardize: 変換後の異常度の時系列を積分面積1で規格化するか
        fill: 戻り値の要素数を NaN 埋めで series と揃えるかどうか
        -------------------------------------------------
        Returns
        list: 3要素のリスト
            要素1: 2つの部分時系列を比較して求めた異常度のリスト
            要素2, 3: テスト行列・履歴行列をそれぞれの特異値の累積寄与率のリスト
        """
        if ncol_h is None:
            ncol_h = round(w / 2)
        if ncol_t is None:
            ncol_t = round(w / 2)
        if ns_h is None:
            ns_h = np.min([1, ncol_h])
        if ns_t is None:
            ns_t = np.min([1, ncol_t])
        if min(ncol_h, ncol_t) > w:
            print('ncol and ncol must be <= w')
        if ns_h > ncol_h or ns_t > ncol_t:
            print('I recommend to set ns_h >= ncol_h and ns_t >= ncol_t')
        start_at = lag + w + ncol_h
        end_at = len(series) + 1
        res = []
        for t in range(start_at, end_at):
            res = res + [self.SSA_anom(series[t - w - ncol_t:t],
                                series[t - lag - w - ncol_h:t - lag],
                                w=w, ncol_t=ncol_t, ncol_h=ncol_h,
                                ns_t=ns_t, ns_h=ns_h,
                                normalize=normalize)]
        anom = [round(x, 14) for x, r1, r2 in res]
        ratio_t = [r1 for x, r1, r2 in res]
        ratio_h = [r2 for x, r1, r2 in res]
        if fill == True:
            anom = [np.nan] * (start_at - 1) + anom
        if standardize:
            c = np.nansum(anom)
            if c != 0:
                anom = [x / c for x in anom]
        return [anom, ratio_t, ratio_h]



    def create_fig(self, option):

    #-----------グラフ描写1(周波数領域での比較)-------------------
        if option==1:
            freq1,fft1 = self.filter.fft(self.data)
            freq2,fft2 = self.filter.fft(self.signal)

    #-----------グラフ描写2(時間領域での比較)-------------------
            plt.figure(figsize=(15,8))
            plt.xlabel('時間[s]', fontsize=14)
            plt.ylabel('角速度[rad/s]', fontsize=14)
            plt.plot(self.t, self.data, label="raw data")
            plt.plot(self.t, self.signal,c="r", label="filtered data")
            


    #----------自己相関----------------------
        elif option==2:
            N = round(len(self.signal)/2)
            acf = sm.tsa.stattools.acf(self.data, nlags=200) #y軸データ、ラグ134、グラフaxes
            idx = argrelmax(acf)
            sm.graphics.tsa.plot_acf(self.data, lags=200,alpha=None,title=None)
            plt.xlabel('Lag', fontsize=20, labelpad=10)
            


    #---------------------------グラフ描写(SSA)------------------------
        elif option==3:
            score = self.SSA_CD(series=self.signal, standardize=True, w=7, lag=1, ns_h=2, ns_t=2, normalize=True)
            score = np.array(score[0])
            score = np.nan_to_num(score,nan = 0)
            peak_idx=argrelmax(score, order=len(score))
            peak_time = peak_idx[0] * self.dt

            fig = plt.figure(figsize=(15,8))
            #fig = plt.figure(figsize=(3,9))

            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            color_code ="#0036bccf"
            #ax1.set_title("バトンパスのタイミング：{}秒後".format(peak_time[0]), pad=30,fontsize="large")
            #ax1.set_title("バトンデータ")

            #ax1.set_title("Anomaly Detection by Singular Spectrum Analysis",pad=30,fontsize='xx-large')
    
            ax1.plot(self.t,self.signal, color="#2c2c2cff", label='バトンのデータ')
            ax2.plot(self.t,score, color=color_code, label='特異スペクトル解析結果')

            #plt.plot(Z, color='red', linewidth=3, alpha=0.7)
            ax1.set_ylabel("角速度[rad/s]",color="#2c2c2cff")
            ax1.set_xlabel("時間[s]")
            ax2.set_ylabel("変化度", color=color_code)
            ax2.tick_params(axis = 'y', colors =color_code)
            handler1, label1 = ax1.get_legend_handles_labels()
            handler2, label2 = ax2.get_legend_handles_labels()
            ax1.legend(handler1 + handler2, label1 + label2)

            return peak_time[0]

            #plt.plot(score)
            

    # SVG化
    def plt2svg(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        s = buf.getvalue()
        graph = base64.b64encode(s)
        graph = graph.decode('utf-8')
        buf.close()
        return graph
