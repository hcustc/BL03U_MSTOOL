# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from .data_processor import DataProcessor
except ImportError:  # Allow running as a direct script from repository root.
    from src.bl03u_massspec.data_processor import DataProcessor


def get_configuration_example():
    """
    示例配置信息，实际应用中应从配置文件读取
    """
    return {
        '18': {'Start': 16000, 'End': 16100, 'Peak': 16050},
        '28': {'Start': 17000, 'End': 17100, 'Peak': 17050},
        '32': {'Start': 18000, 'End': 18100, 'Peak': 18050},
    }


def main():
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 示例1：读取单个文件并绘制
    print("\n示例1：读取单个文件并绘制")
    try:
        # 假设src目录下有txt数据文件
        example_file_path = os.path.join(current_dir, 'src', 'test.txt')
        
        # 检查文件是否存在，如果不存在则提示用户
        if not os.path.exists(example_file_path):
            print(f"示例文件 {example_file_path} 不存在，请修改为实际存在的文件路径")
        else:
            # 读取数据并绘制
            x_data, y_data = processor.load_data_for_plot(example_file_path)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_data, y_data, 'g-', linewidth=1)
            plt.title('单个文件质谱图')
            plt.xlabel('TOF')
            plt.ylabel('COUNTS')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('单个文件质谱图.png')
            plt.show()
            
            print(f"成功读取并绘制文件: {example_file_path}")
    except Exception as e:
        print(f"示例1执行出错: {e}")
    
    # 示例2：读取文件夹中的所有文件并累加绘制
    print("\n示例2：读取文件夹中的所有文件并累加绘制")
    try:
        # 使用仓库保留的最小示例数据目录
        example_folder_path = os.path.join(current_dir, 'data', 'raw', 'C6F11O2H', '11.5eV')
        
        # 检查文件夹是否存在，如果不存在则提示用户
        if not os.path.exists(example_folder_path):
            print(f"示例文件夹 {example_folder_path} 不存在，请修改为实际存在的文件夹路径")
        else:
            # 读取文件夹数据并绘制
            x_data, y_data = processor.load_folder_data_for_plot(example_folder_path)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_data, y_data, 'b-', linewidth=1)
            plt.title('文件夹累加质谱图')
            plt.xlabel('TOF')
            plt.ylabel('COUNTS')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('文件夹累加质谱图.png')
            plt.show()
            
            print(f"成功读取并绘制文件夹: {example_folder_path}")
    except Exception as e:
        print(f"示例2执行出错: {e}")
    
    # 示例3：提取头信息和强度信息
    print("\n示例3：提取头信息和强度信息")
    try:
        # 使用示例1中的文件
        example_file_path = os.path.join(current_dir, 'src', 'test.txt')
        
        # 检查文件是否存在，如果不存在则提示用户
        if not os.path.exists(example_file_path):
            print(f"示例文件 {example_file_path} 不存在，请修改为实际存在的文件路径")
        else:
            # 提取头信息
            head_info = processor.get_signal(example_file_path, 'head')
            print("头信息前3行:")
            for i, line in enumerate(head_info[:3]):
                print(f"  {i+1}: {line}")
            
            # 提取强度信息
            intensity_info = processor.get_signal(example_file_path, 'intensity')
            print(f"强度信息共 {len(intensity_info)} 行，前3行:")
            for i, line in enumerate(intensity_info[:3]):
                print(f"  {i+1}: {line}")
            
            # 提取头文件中的数值信息
            head_values = processor.get_head_info(head_info)
            print("头文件中的数值信息:")
            for i, value in enumerate(head_values):
                print(f"  {i+1}: {value}")
    except Exception as e:
        print(f"示例3执行出错: {e}")
    
    # 示例4：计算峰面积
    print("\n示例4：计算峰面积")
    try:
        # 使用示例1中的文件
        example_file_path = os.path.join(current_dir, 'src', 'test.txt')
        
        # 检查文件是否存在，如果不存在则提示用户
        if not os.path.exists(example_file_path):
            print(f"示例文件 {example_file_path} 不存在，请修改为实际存在的文件路径")
        else:
            # 获取配置信息
            config = get_configuration_example()
            
            # 获取强度信息
            intensity_info = processor.get_signal(example_file_path, 'intensity')
            
            # 计算峰面积
            peak_areas = processor.integrate(config, intensity_info)
            print("峰面积计算结果:")
            for species, area in peak_areas.items():
                print(f"  物种 {species}: 面积 = {area:.2f}")
            
            # 计算峰高
            peak_heights = processor.integrate(config, intensity_info, mode='height')
            print("峰高计算结果:")
            for species, height in peak_heights.items():
                print(f"  物种 {species}: 高度 = {height}")
    except Exception as e:
        print(f"示例4执行出错: {e}")


if __name__ == "__main__":
    main()
