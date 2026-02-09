# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from numpy import trapz
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Dict, Tuple, Optional, Union

NUMBER_REGEX = re.compile(r'[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?')


class DataProcessor:
    """
    数据处理类，封装原代码处理原始的txt数据的函数，
    包括提取头信息，和强度信息，数据读取函数等
    """
    
    def __init__(self, log_level=logging.INFO):
        """
        初始化数据处理类
        
        Args:
            log_level: 日志级别，默认为INFO
        """
        # 配置日志
        logging.basicConfig(level=log_level, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def read_file(self, file_path: str, skiprows: int = 10, start_row: int = 5000) -> np.ndarray:
        """
        读取单个txt文件并提取数据
        
        Args:
            file_path: 文件路径
            skiprows: 跳过的行数，默认为10（头文件行数）
            start_row: 开始提取的行索引，默认为5000
            
        Returns:
            提取的y数据数组
        """
        try:
            data = pd.read_csv(file_path, skiprows=skiprows, sep='\t', header=None)
            y_data = data.iloc[start_row:, 0].values  # 提取 y 数据
            return y_data
        except Exception as e:
            self.logger.error(f"读取文件 {file_path} 时出错: {e}")
            raise
    
    def read_files_concurrently(self, folder_path: str, file_extension: str = ".txt") -> np.ndarray:
        """
        并发读取文件夹中的所有文件并累加数据
        
        Args:
            folder_path: 文件夹路径
            file_extension: 文件扩展名，默认为".txt"
            
        Returns:
            累加后的y数据数组
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
            
        y_sum = None
        with ThreadPoolExecutor() as executor:
            file_names = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
            if not file_names:
                raise ValueError(f"在 {folder_path} 中没有找到{file_extension}文件")
                
            futures = []
            for file_name in file_names:
                file_path = os.path.join(folder_path, file_name)
                futures.append(executor.submit(self.read_file, file_path))
            
            for future in as_completed(futures):
                try:
                    y_data = future.result()
                    if y_sum is None:
                        y_sum = np.zeros_like(y_data)
                    y_sum += y_data
                except Exception as e:
                    self.logger.error(f"处理文件时出错: {e}")
                    
        return y_sum
    
    def get_signal(self, file_path: str, choice: str = 'intensity') -> List[str]:
        """
        获取txt中信号信息
        
        Args:
            file_path: 文件路径
            choice: 选择返回的数据类型，'intensity'表示返回强度数据，'head'表示返回头文件数据
            
        Returns:
            根据choice返回相应的数据列表
        """
        try:
            with open(file_path) as f:
                lines = f.readlines()
                lines = [x.strip() for x in lines]

                # 头文件矫正模式,如果是催化质谱，有10行头文件，如果是TOF1,有7行头文件。
                intensity_info = lines[10:]
                head_info = lines[:10]
                
            if choice == 'intensity':
                return intensity_info
            elif choice == 'head':
                return head_info
            else:
                raise ValueError(f"无效的choice参数: {choice}，应为'intensity'或'head'")
        except Exception as e:
            self.logger.error(f"获取信号数据时出错: {e}")
            raise
    
    def get_head_info(self, head_list: List[str]) -> List[str]:
        """
        识别头文件中的数据
        
        Args:
            head_list: 头文件数据列表
            
        Returns:
            提取的头文件信息列表
        """
        head_list_info = []
        try:
            for head in head_list:
                mo = NUMBER_REGEX.search(head)
                if mo:
                    head_info = mo.group()
                    head_list_info.append(head_info)
                else:
                    head_list_info.append("1")  # 如果没有匹配到数字，默认为1
            return head_list_info
        except Exception as e:
            self.logger.error(f"处理头文件信息时出错: {e}")
            raise
    
    def integrate(self, info: Dict, intensity_list: List[str], mode: str = 'integrate') -> Dict:
        """
        定义积分函数，计算峰面积或峰高
        
        Args:
            info: 包含物种的m/z以及峰的peak,start,end值的配置信息
            intensity_list: 强度数据列表
            mode: 'integrate'表示计算峰面积，其他值表示计算峰高
            
        Returns:
            包含各物种峰面积或峰高的字典
        """
        list_peak_area = {}
        list_peak_height = {}
        try:
            for species in info:
                start = info[species]['Start']
                end = info[species]['End']
                list_tof = [x for x in range(start, end + 1)]
                list_intensity_slice = intensity_list[start: end + 1]
                list_intensity_slice = list(map(int, list_intensity_slice))
                integrate_peak = trapz(list_intensity_slice, list_tof, dx=0.01)
                peak_height = max(list_intensity_slice)
                list_peak_height.setdefault(species, peak_height)
                list_peak_area.setdefault(species, integrate_peak)
                
            if mode == 'integrate':
                return list_peak_area
            else:
                return list_peak_height
        except Exception as e:
            self.logger.error(f"积分计算时出错: {e}")
            raise
    
    def calculate(self, path: str, configuration: Dict) -> List[Dict]:
        """
        计算文件夹中所有文件的积分结果
        
        Args:
            path: 文件夹路径
            configuration: 配置信息字典
            
        Returns:
            包含各文件积分结果的列表
        """
        try:
            file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
            list_integrate = []
            
            for file_path in file_paths:
                intensity = self.get_signal(file_path, 'intensity')
                head_list = self.get_head_info(self.get_signal(file_path, 'head'))
                integrate_result = self.integrate(configuration, intensity)
                
                # 用流强矫正，如用TOF-1，需要注释掉
                for result in integrate_result:
                    integrate_result[result] = integrate_result[result] / float(head_list[1])
                    
                list_integrate.append(integrate_result)
                
            return list_integrate
        except Exception as e:
            self.logger.error(f"计算积分结果时出错: {e}")
            raise
    
    def get_result(self, path: str, configuration: Dict) -> Dict:
        """
        获取最终结果
        
        Args:
            path: 文件夹路径
            configuration: 配置信息字典
            
        Returns:
            包含各物种积分结果的字典
        """
        try:
            list_integrate = self.calculate(path, configuration)
            dic_area = {}

            for i, integrate_result in enumerate(list_integrate):
                for species in integrate_result:
                    dic_area.setdefault(species, []).append(integrate_result[species])
                    
            return dic_area
        except Exception as e:
            self.logger.error(f"获取最终结果时出错: {e}")
            raise
    
    def load_data_for_plot(self, file_path: str, skiprows: int = 10, start_index: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据用于绘图
        
        Args:
            file_path: 文件路径
            skiprows: 跳过的行数
            start_index: 开始的索引
            
        Returns:
            x_data, y_data元组
        """
        try:
            data = np.loadtxt(file_path, skiprows=skiprows)
            y_data = data[start_index:]
            x_data = np.arange(start_index + 1, len(data) + 1)
            return x_data, y_data
        except Exception as e:
            self.logger.error(f"加载绘图数据时出错: {e}")
            raise
    
    def load_folder_data_for_plot(self, folder_path: str, skiprows: int = 10, start_index: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载文件夹中的数据用于绘图
        
        Args:
            folder_path: 文件夹路径
            skiprows: 跳过的行数
            start_index: 开始的索引
            
        Returns:
            x_data, y_data元组
        """
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError("文件夹不存在")

            y_cumulative = None
            for file in os.listdir(folder_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder_path, file)
                    data = np.loadtxt(file_path, skiprows=skiprows)
                    y_data = data[start_index:]
                    if y_cumulative is None:
                        y_cumulative = np.zeros_like(y_data)
                    y_cumulative += y_data

            if y_cumulative is not None:
                x_data = np.arange(start_index + 1, len(y_cumulative) + start_index + 1)
                return x_data, y_cumulative
            else:
                raise ValueError("没有找到任何有效的文本文件进行绘图")
        except Exception as e:
            self.logger.error(f"加载文件夹数据时出错: {e}")
            raise
