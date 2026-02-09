# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
import pywt
import matplotlib.pyplot as plt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from multiprocessing import Pool
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
from PyQt6.QtWidgets import QFileDialog


# 定义一个函数用于读取单个文件并提取 y_data
def read_file(file_path: str, skiprows: int = 10, start_row: int = 5000) -> np.ndarray:
    data = pd.read_csv(file_path, skiprows=skiprows, sep='\t', header=None)
    y_data = data.iloc[start_row:, 0].values  # 提取 y 数据
    return y_data

# 使用 ThreadPoolExecutor 并发读取文件
def read_files_concurrently(folder_path: str, skiprows: int = 10, start_row: int = 5000) -> np.ndarray:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
    y_sum = None
    with ThreadPoolExecutor() as executor:
        file_names = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))
        if not file_names:
            raise ValueError(f"在 {folder_path} 中没有找到.txt文件")
            
        futures = [
            executor.submit(read_file, os.path.join(folder_path, file_name), skiprows, start_row)
            for file_name in file_names
        ]
        
        for future in as_completed(futures):
            try:
                y_data = future.result()
                if y_sum is None:
                    y_sum = np.zeros_like(y_data)
                y_sum += y_data
            except Exception as e:
                logging.error(f"处理文件时出错: {e}")
                
    return y_sum

# 定义一个函数用于平滑数据
def smooth_data(data, window_size=3, method='savgol'):
    """平滑数据"""
    if method == 'savgol':
        return savgol_filter(data, window_size, 2)
    else:
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 定义一个函数用于估计噪声水平
def estimate_noise(intensities):
    """估计噪声水平"""
    return np.median(np.abs(np.diff(intensities)))

# 定义一个函数用于背景校正
def correct_baseline(intensities, method='min'):
    """
    背景校正，确保强度为非负。
    """
    if method == 'min':
        baseline = np.min(intensities)
    elif method == 'median':
        baseline = np.median(intensities)
    else:
        raise ValueError("Unsupported baseline correction method.")
    
    corrected = intensities - baseline
    corrected[corrected < 0] = 0  # 确保强度非负
    return corrected

# 定义高斯函数
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

# 使用高斯拟合精确定位峰中心
def refine_peak_center(x, y, peak_index):
    peak_range = slice(max(0, peak_index-5), min(len(x), peak_index+6))
    x_peak = x[peak_range]
    y_peak = y[peak_range]
    initial_guess = [y[peak_index], x[peak_index], 1.0]
    try:
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=initial_guess)
        refined_center = popt[1]
    except RuntimeError:
        refined_center = x[peak_index]
    return refined_center

def calculate_cwt(intensities, widths, wavelet='mexh'):
    cwt_matrix, _ = pywt.cwt(intensities, widths, wavelet)
    return cwt_matrix

def find_peaks_cwt(cwt_matrix, ms_time, intensities, snr_threshold=0.05, prominence_ratio=0.012):
    max_cwt = np.max(cwt_matrix, axis=0)
    noise_level = estimate_noise(intensities)
    height_threshold = snr_threshold * noise_level
    prominence_threshold = prominence_ratio * np.max(max_cwt)
    peaks, _ = find_peaks(max_cwt, height=height_threshold, prominence=prominence_threshold)
    results_full = peak_widths(max_cwt, peaks, rel_height=0.5)
    left_bases = results_full[2].astype(int)
    right_bases = results_full[3].astype(int)
    valid_indices = (left_bases >= 0) & (right_bases < len(ms_time))
    peaks = peaks[valid_indices]
    peak_info = []

    for i, peak in enumerate(peaks):
        refined_center = refine_peak_center(ms_time, intensities, peak)
        peak_info.append((
            refined_center, 
            time_to_mz(refined_center), 
            intensities[peak], 
            ms_time[left_bases[i]], 
            ms_time[right_bases[i]]
        ))

    return np.array(peak_info)

def find_mass_spec_peaks_cwt(ms_time, intensities, widths=np.arange(1, 30), snr_threshold=0.02, prominence_ratio=0.005):
    ms_time = np.array(ms_time)
    intensities = np.array(intensities)
    if len(ms_time) != len(intensities):
        raise ValueError("mz_values 和 intensities 的长度必须一致。")
    corrected_intensities = correct_baseline(intensities)
    cwt_matrix = calculate_cwt(corrected_intensities, widths)
    peak_info = find_peaks_cwt(cwt_matrix, ms_time, corrected_intensities, snr_threshold, prominence_ratio)
    return peak_info

# 定义一个函数用于将飞行时间转换为 m/z 值
def time_to_mz(time: np.ndarray) -> np.ndarray:
    a, b, c = Config.TIME_COEFFICIENTS.values()
    return a + b * time + c * time * time

# 定义一个函数用于绘制质谱数据及其峰值
def plot_peaks(ms_time, intensities, peaks):
    """绘制质谱数据并标注峰值"""
    plt.figure(figsize=(12, 6))
    plt.plot(ms_time, intensities, label="原始强度", color='blue', alpha=0.5)
    
    # 绘制平滑后的数据
    smoothed_intensities = smooth_data(intensities)
    plt.plot(ms_time, smoothed_intensities, label="平滑后强度", color='green', alpha=0.7)
    
    # 标注峰值
    peak_ms_time = [peak[0] for peak in peaks]
    peak_intensities = [peak[2] for peak in peaks]
    plt.scatter(peak_ms_time, peak_intensities, color='red', label="检测到的峰", zorder=5)
    
    for peak in peaks:
        plt.annotate(f'{peak[0]:.2f}', (peak[0], peak[2]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'  # 使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置 sans-serif 字体
    plt.xlabel("m/z")
    plt.ylabel("强度")
    plt.title("质谱图与检测到的峰")
    plt.legend()
    plt.show()

# 定义一个函数用于绘制 CWT 结果 
def plot_cwt(cwt_matrix, ms_time, intensities, peaks):
    """绘制 CWT 结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(ms_time, intensities, label="原始强度", color='blue', alpha=0.5)
    
    # 绘制平滑后的数据
    smoothed_intensities = smooth_data(intensities)
    plt.plot(ms_time, smoothed_intensities, label="平滑后强度", color='green', alpha=0.7)
    
    # 绘制 CWT 结果
    plt.imshow(cwt_matrix, cmap='coolwarm', aspect='auto', origin='lower',
               extent=[ms_time[0], ms_time[-1], 1, len(cwt_matrix)])
    
    # 标注峰值
    peak_ms_time = [peak[0] for peak in peaks]
    peak_intensities = [peak[2] for peak in peaks]
    plt.scatter(peak_ms_time, peak_intensities, color='red', label="检测到的峰", zorder=5)
    
    for peak in peaks:
        plt.annotate(f'{peak[0]:.2f}', (peak[0], peak[2]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'  # 使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置 sans-serif 字体
    plt.xlabel("m/z")
    plt.ylabel("强度")
    plt.title("CWT 结果与检测到的峰")
    plt.legend()
    plt.show()

def plot_peaks_mz(ms_time, intensities, peaks, time_to_mz):
    """绘制以 m/z 为横轴的质谱数据并标注峰值"""
    ms_mz = [time_to_mz(i) for i in ms_time]
    
    plt.figure(figsize=(12, 6))
    plt.plot(ms_mz, intensities, label="原始强度", color='blue', alpha=0.5)
    
    # 绘制平滑后的数据
    smoothed_intensities = smooth_data(intensities)
    plt.plot(ms_mz, smoothed_intensities, label="平滑后强度", color='green', alpha=0.7)
    
    # 标注峰值
    peak_ms_mz = [time_to_mz(peak[0]) for peak in peaks]
    peak_intensities = [peak[2] for peak in peaks]
    plt.scatter(peak_ms_mz, peak_intensities, color='red', label="检测到的峰", zorder=5)
    
    for peak, mz in zip(peaks, peak_ms_mz):
        plt.annotate(f'{mz:.2f}', (mz, peak[2]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'  # 使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置 sans-serif 字体
    plt.xlabel("m/z")
    plt.ylabel("强度")
    plt.title("质谱图与检测到的峰")
    plt.legend()
    plt.show()

def plot_each_peak(intensities, peak_info, output_dir='peaks_output'):
    """
    绘制并保存每个峰的图像为单独的 PNG 文件。
    
    参数：
    - intensities: 已经过背景校正的强度序列数据
    - peak_info: 包含每个峰信息的列表 (refined_center, mz, intensity, left_base, right_base)
    - output_dir: 输出图像保存的目录
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for peak in peak_info:
        # 获取峰信息
        mz = peak[1]
        left_base = int(peak[3]) - 5000
        right_base = int(peak[4]) - 5000
        
        # 创建新图
        plt.figure(figsize=(10, 5))
        
        # 生成峰的时间和强度数据
        peak_time = list(range(left_base, right_base))
        peak_intensities = [intensities[j] for j in peak_time]
        
        # 绘制峰
        plt.plot(peak_time, peak_intensities, label='Intensity', color='lightgray')
        
        # 设置标题和标签
        plt.xlabel('Time (ms)')
        plt.ylabel('Intensity')
        plt.title(f'Peak {mz}')
        plt.legend()
        
        # 保存为 PNG 文件
        filename = os.path.join(output_dir, f'peak_{mz}.png')
        plt.savefig(filename)
        plt.close()  # 关闭图像，释放内存
        print(f"已保存峰 {mz} 的图像至文件：{filename}")

    print(f"所有峰的图像已保存至目录：{output_dir}")

# 建议将常量提取为配置
class Config:
    WINDOW_SIZE = 3
    SNR_THRESHOLD = 0.02
    PROMINENCE_RATIO = 0.005
    TIME_COEFFICIENTS = {
        'a': 0.284406,
        'b': 0.000640911,
        'c': 3.6683E-07
    }

# 使用多进程处理大量数据
def process_data_parallel(data_chunks):
    with Pool() as pool:
        results = pool.map(process_chunk, data_chunks)
    return np.concatenate(results)

@dataclass
class PeakConfig:
    """峰值检测配置"""
    window_size: int = 3
    snr_threshold: float = 0.02
    prominence_ratio: float = 0.005
    min_peak_distance: int = 10
    min_peak_width: int = 3
    max_peak_width: int = 50
    wavelet_widths: np.ndarray = field(default_factory=lambda: np.arange(1, 30))
    time_coefficients: Dict[str, float] = None
    
    def __post_init__(self):
        if self.time_coefficients is None:
            self.time_coefficients = {
                'a': 0.284406,
                'b': 0.000640911,
                'c': 3.6683E-07
            }

class PeakDetector:
    """峰值检测器"""
    def __init__(self, config: PeakConfig = None):
        self.config = config or PeakConfig()
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def gaussian(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
        """高斯函数"""
        return amp * np.exp(-(x-cen)**2 / (2*wid**2))
    
    def smooth_data(self, data: np.ndarray) -> np.ndarray:
        """使用Savitzky-Golay滤波平滑数据"""
        try:
            return savgol_filter(data, self.config.window_size, 2)
        except Exception as e:
            self.logger.warning(f"平滑数据失败: {e}")
            return data
    
    def estimate_noise(self, intensities: np.ndarray) -> float:
        """使用中位绝对偏差(MAD)估计噪声水平"""
        diff = np.abs(np.diff(intensities))
        mad = np.median(diff)
        return mad * 1.4826  # 转换为标准差估计
    
    def correct_baseline(self, intensities: np.ndarray) -> np.ndarray:
        """使用滑动窗口最小值进行背景校正"""
        window_size = self.config.window_size * 10
        pad_width = window_size // 2
        padded = np.pad(intensities, pad_width, mode='edge')
        baseline = np.zeros_like(intensities)
        
        for i in range(len(intensities)):
            window = padded[i:i + window_size]
            baseline[i] = np.percentile(window, 5)
        
        corrected = intensities - baseline
        return np.maximum(corrected, 0)
    
    def refine_peak_center(self, x: np.ndarray, y: np.ndarray, peak_index: int) -> float:
        """使用高斯拟合精确定位峰中心"""
        try:
            peak_range = slice(
                max(0, peak_index-self.config.min_peak_width),
                min(len(x), peak_index+self.config.min_peak_width+1)
            )
            x_peak = x[peak_range]
            y_peak = y[peak_range]
            
            # 初始参数估计
            amp_init = y[peak_index]
            cen_init = x[peak_index]
            wid_init = self.config.min_peak_width / 2
            
            popt, _ = curve_fit(
                self.gaussian, x_peak, y_peak,
                p0=[amp_init, cen_init, wid_init],
                bounds=([0, x_peak[0], 0], 
                       [np.inf, x_peak[-1], self.config.max_peak_width])
            )
            return popt[1]
        except Exception as e:
            self.logger.warning(f"峰中心精确定位失败: {e}")
            return x[peak_index]
    
    def calculate_cwt(self, intensities: np.ndarray) -> np.ndarray:
        """计算连续小波变换"""
        return pywt.cwt(intensities, self.config.wavelet_widths, 'mexh')[0]
    
    def find_peaks_cwt(self, cwt_matrix: np.ndarray, ms_time: np.ndarray, 
                      intensities: np.ndarray) -> np.ndarray:
        """使用CWT寻找峰值"""
        max_cwt = np.max(cwt_matrix, axis=0)
        noise_level = self.estimate_noise(intensities)
        height_threshold = self.config.snr_threshold * noise_level
        prominence_threshold = self.config.prominence_ratio * np.max(max_cwt)
        
        peaks, properties = find_peaks(
            max_cwt,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=self.config.min_peak_distance,
            width=(self.config.min_peak_width, self.config.max_peak_width)
        )
        
        results_full = peak_widths(max_cwt, peaks, rel_height=0.5)
        left_bases = results_full[2].astype(int)
        right_bases = results_full[3].astype(int)
        
        # 验证峰的有效性
        valid_indices = (
            (left_bases >= 0) & 
            (right_bases < len(ms_time)) &
            (right_bases - left_bases >= self.config.min_peak_width) &
            (right_bases - left_bases <= self.config.max_peak_width)
        )
        
        peaks = peaks[valid_indices]
        left_bases = left_bases[valid_indices]
        right_bases = right_bases[valid_indices]
        
        peak_info = []
        for i, peak in enumerate(peaks):
            refined_center = self.refine_peak_center(ms_time, intensities, peak)
            peak_info.append((
                refined_center,
                self.time_to_mz(refined_center),
                intensities[peak],
                ms_time[left_bases[i]],
                ms_time[right_bases[i]]
            ))
        
        return np.array(peak_info)
    
    def time_to_mz(self, time: float) -> float:
        """将飞行时间转换为质荷比"""
        a = self.config.time_coefficients['a']
        b = self.config.time_coefficients['b']
        c = self.config.time_coefficients['c']
        return a + b * time + c * time * time
    
    def detect_peaks(self, ms_time: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """主要的峰值检测函数"""
        # 数据预处理
        intensities = self.smooth_data(intensities)
        intensities = self.correct_baseline(intensities)
        
        # CWT分析
        cwt_matrix = self.calculate_cwt(intensities)
        
        # 峰值检测
        peak_info = self.find_peaks_cwt(cwt_matrix, ms_time, intensities)
        
        return peak_info

    def plot_results(self, ms_time: np.ndarray, intensities: np.ndarray, 
                    peaks: np.ndarray, show_mz: bool = True):
        """绘制结果"""
        plt.figure(figsize=(12, 6))
        
        if show_mz:
            x_values = [self.time_to_mz(t) for t in ms_time]
            peak_x = [peak[1] for peak in peaks]  # 使用m/z值
            x_label = "m/z"
        else:
            x_values = ms_time
            peak_x = [peak[0] for peak in peaks]  # 使用时间值
            x_label = "Time (ms)"
            
        # 绘制原始数据和平滑后的数据
        plt.plot(x_values, intensities, 'gray', alpha=0.5, label='Raw Data')
        smoothed = self.smooth_data(intensities)
        plt.plot(x_values, smoothed, 'b', alpha=0.7, label='Smoothed Data')
        
        # 标注峰值
        peak_intensities = [peak[2] for peak in peaks]
        plt.scatter(peak_x, peak_intensities, color='red', s=50, 
                   label='Detected Peaks', zorder=5)
        
        # 添加峰值标注
        for x, y, mz in zip(peak_x, peak_intensities, [p[1] for p in peaks]):
            plt.annotate(f'{mz:.2f}', (x, y), xytext=(0, 10),
                        textcoords='offset points', ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.xlabel(x_label)
        plt.ylabel('Intensity')
        plt.title('Mass Spectrometry Peak Detection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # 读取数据
    strat = time.time()
    default_folder = os.path.join(os.getcwd(), "data", "raw", "C6F11O2H", "11.5eV")
    folder_path = sys.argv[1] if len(sys.argv) > 1 else default_folder
    print(f"使用数据目录: {folder_path}")
    try:
        # df = pd.read_csv(file_path, skiprows=10, sep='\t', header=None)
        y_sum = read_files_concurrently(folder_path)

        # 假设第一列是强度值，第二列是 m/z 值
        # intensities = df.iloc[2500:, 0].values
        intensities = y_sum
        ms_time = [i for i in range(5000,len(intensities)+5000)]
        ms_value = [time_to_mz(i) for i in ms_time]
        # intensities = df.iloc[12000:13000, 0].values
        # mz_values = [i for i in range(12000,13000)]
        
        # 寻找峰值
        peaks = find_mass_spec_peaks_cwt(ms_time, intensities)

        # 将峰值信息存储到csv文件中
        peaks_df = pd.DataFrame(peaks, columns=['Peak_Time', 'm/z', 'Intensity', 'Left', 'Right'])
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "保存峰值数据",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
        )
        if file_path:
            if file_path.endswith('.csv'):
                peaks_df.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                peaks_df.to_excel(file_path, index=False)
            else:
                peaks_df.to_csv(file_path, index=False)
            print(f"峰值数据已保存至：{file_path}")

        print("找到的峰值:")
        for peak in peaks:
            print(f"Peak: {peak[0]}, m/z: {peak[1]:.4f}, Intensity: {peak[2]:.4f}, Left: {peak[3]:.4f}, Right: {peak[4]:.4f}")
        
        end = time.time()
        print(f"运行时间: {end-strat} s")
        # 绘图
        plot_peaks_mz(ms_time, intensities, peaks, time_to_mz)
        # cwt_matrix = calculate_cwt_concurrently(intensities, np.arange(1, 31))
        # plot_cwt(cwt_matrix, ms_time, intensities, peaks)
        # plot_each_peak(intensities, peaks)



    except FileNotFoundError:
        print(f"文件未找到: {folder_path}")
    except Exception as e:
        print(f"发生错误: {e}")
