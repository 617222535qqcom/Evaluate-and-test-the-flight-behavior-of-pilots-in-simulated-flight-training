import pandas as pd
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

# 读取CSV文件数据
def read_csv_data(file_path, column_index=0):
    # 使用pandas读取CSV文件中的数据
    df = pd.read_csv(file_path)
    # 假设我们感兴趣的数据在指定列
    data = df.iloc[:, column_index].values
    return data

# 进行短时傅里叶变换
def perform_stft(data, fs=10, window='hann', nperseg=32):
    # 执行STFT
    f, t, Zxx = stft(data, fs=fs, window=window, nperseg=nperseg)
    return f, t, Zxx

# 绘制STFT结果
def plot_stft(f, t, Zxx):
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.show()

# 主程序
def main():
    file_path = 'C:/Users/张和铭/pythonProject55/combined/sub-cp004_ses-20210330_level-01B-run-001_combined_data.csv'  # CSV文件路径
    column_index = 0  # 数据所在列的索引

    # 读取数据
    data = read_csv_data(file_path, column_index)

    # 执行STFT
    f, t, Zxx = perform_stft(data, fs=10)  # 假设采样频率为1000Hz

    # 绘制结果
    plot_stft(f, t, Zxx)

if __name__ == "__main__":
    main()