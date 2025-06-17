import os

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QTableWidgetItem, QTableWidget, QLabel, QLineEdit, QGraphicsPixmapItem, \
    QGraphicsScene, QGraphicsView, QMessageBox, QMenu, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QDialogButtonBox, QFileDialog
import pyqtgraph as pg
from pyqtgraph import mkPen, LinearRegionItem, GraphicsLayoutWidget
import numpy as np
from PyQt6 import QtCore
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap

from massspec import Ui_MainWindow
from FindPeak import PeakDetector, find_mass_spec_peaks_cwt, PeakConfig


class PeakDialog(QDialog):
    """峰值编辑对话框"""
    def __init__(self, parent=None, peak_data=None):
        super().__init__(parent)
        self.setWindowTitle("编辑峰值")
        self.setup_ui()
        if peak_data:
            self.set_peak_data(peak_data)

    def setup_ui(self):
        layout = QFormLayout()
        
        # 创建输入框
        self.species_edit = QLineEdit()
        self.time_edit = QLineEdit()
        self.mz_edit = QLineEdit()
        self.intensity_edit = QLineEdit()
        self.left_edit = QLineEdit()
        self.right_edit = QLineEdit()
        
        # 添加到布局
        layout.addRow("Species:", self.species_edit)
        layout.addRow("飞行时间:", self.time_edit)
        layout.addRow("质量数 (m/z):", self.mz_edit)
        layout.addRow("强度:", self.intensity_edit)
        layout.addRow("左边界:", self.left_edit)
        layout.addRow("右边界:", self.right_edit)
        
        # 添加确定和取消按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Orientation.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(buttons)
        self.setLayout(main_layout)

    def set_peak_data(self, peak_data):
        """设置对话框中的峰值数据"""
        self.species_edit.setText(str(peak_data[0]))
        self.time_edit.setText(str(peak_data[1]))
        self.mz_edit.setText(str(peak_data[2]))
        self.intensity_edit.setText(str(peak_data[3]))
        self.left_edit.setText(str(peak_data[4]))
        self.right_edit.setText(str(peak_data[5]))

    def get_peak_data(self):
        """获取对话框中的峰值数据"""
        return {
            'species': self.species_edit.text(),
            'time': float(self.time_edit.text()),
            'mz': float(self.mz_edit.text()),
            'intensity': float(self.intensity_edit.text()),
            'left': float(self.left_edit.text()),
            'right': float(self.right_edit.text())
        }

class massspec_func(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(massspec_func, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.plot_graph)
        self.pushButton_plot_graph_sum.clicked.connect(self.plot_graph_sum)
        self.setWindowIcon(QIcon('icons/icon.png'))
        self.setWindowTitle('BL03U_MassSpectrumTool')
        self.pushButton_3.clicked.connect(self.calculate)
        self.toolButton.clicked.connect(self.transfer)
        self.savePeakdata.clicked.connect(self.save)
        self.addPeak.clicked.connect(self.add_peak)
        self.pushButton_4.clicked.connect(self.auto_find_peaks)  # 自动寻峰按钮

        
        # 加载并显示图片
        self.original_pixmap = QPixmap('icons/bjt.png')  # 替换为您的图像文件路径
        if not self.original_pixmap.isNull():
            self.update_label_picture()

        # 连接 resize 事件以动态调整 label_picture 的大小
        self.widget_2.resizeEvent = self.on_widget_2_resize

        # # 设置 label 组件的图片
        # image_path = 'icons/bjt.png'  # 替换为您的图像文件路径
        # pixmap = QPixmap(image_path)
        # self.label_picture.setPixmap(pixmap)

        # 初始化图表相关的成员变量
        self.p1 = None
        self.p2 = None
        self.region = None
        self.spectrum_plot = None

        # 添加一个定时器用于延迟更新
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # 设置延迟时间，单位为毫秒
        self.update_timer.timeout.connect(self.on_update_timeout)
        self.pending_update = False

        # 初始化文本框的可见性
        self.label_4.setVisible(False)  # 假设 label_4 是用于显示ABC结果的文本框
        self.pushButton_3.setVisible(False)

        # 连接选项卡切换事件
        self.peakResult.currentChanged.connect(self.on_tab_changed)

        # 初始化复选框状态
        self.checkBox_show_gaussian.setChecked(True)  # 默认显示高斯拟合图

        # 设置表格的上下文菜单
        self.peakData.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.peakData.customContextMenuRequested.connect(self.show_peak_context_menu)

        # 添加更新按钮
        self.updatePeaks = QPushButton("更新峰标注")
        self.horizontalLayout_4.addWidget(self.updatePeaks)
        self.updatePeaks.clicked.connect(self.update_all_peaks)

        # 添加清除按钮
        self.clearPeaksButton = QPushButton("清除峰值数据")
        self.horizontalLayout_4.addWidget(self.clearPeaksButton)
        self.clearPeaksButton.clicked.connect(self.clear_peak_data)

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print(f"无法加载图片: {image_path}")
        else:
            # 根据 label_picture 的大小调整图片大小并保持宽高比
            scaled_pixmap = pixmap.scaled(
                self.widget_2.size(),
                aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.label_picture.setPixmap(scaled_pixmap)

    
    def update_label_picture(self):
        """更新 label_picture 的图片"""
        scaled_pixmap = self.original_pixmap.scaled(
            self.label_picture.size(),
            aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.label_picture.setPixmap(scaled_pixmap)

    def on_widget_2_resize(self, event):
        """当 widget_2 大小改变时调整 label_picture 的大小"""
        # 使用 setScaledContents 来自动调整图片大小，而不是手动设置几何形状
        self.update_label_picture()

    def clear_widgets(self):
        """清除旧的小部件"""
        if self.label_picture is not None:
            self.layout().removeWidget(self.label_picture)
            self.label_picture.setParent(None)
            self.label_picture.deleteLater()
            self.label_picture = None

        while self.graph_layout.count():
            item = self.graph_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def setup_plots(self, x_data, y_data, title="质谱图"):
        """设置并绘制图表"""
        pg.setConfigOptions(antialias=True)

        win = GraphicsLayoutWidget()
        self.graph_layout.addWidget(win)

        # 检查复选框状态，决定是否显示高斯拟合图
        if self.checkBox_show_gaussian.isChecked():
            self.p1 = win.addPlot(title="高斯拟合图")
            self.p1.setLabel('left', 'COUNTS')
            self.p1.showGrid(x=True, y=True)
            self.p1.setLabel('bottom', 'TOF')

        win.nextRow()

        self.p2 = win.addPlot(title=title)
        self.p2.setLabel('left', 'COUNTS')
        self.p2.showGrid(x=True, y=True)
        self.p2.setLabel('bottom', 'TOF')
        self.spectrum_plot = self.p2.plot(x_data, y_data, pen=mkPen('g', width=2), name='质谱图')

        # 添加 LinearRegionItem 到 p2，设置移动模式和边界约束
        self.region = LinearRegionItem(
            values=[16612, 16620],
            bounds=None,  # 移除边界约束
            movable=True,  # 允许整体移动
            brush=pg.mkBrush(color=(128, 128, 128, 50))  # 半透明灰色
        )
        self.region.setZValue(20)
        self.p2.addItem(self.region, ignoreBounds=True)
        
        # 连接信号
        self.region.sigRegionChanged.connect(self.update_gaussian_fit)
        if self.checkBox_show_gaussian.isChecked():
            self.p1.sigRangeChanged.connect(lambda window, viewRange: self.update_region(viewRange))

    def plot_graph(self):
        """从单个文件绘制图表"""
        try:
            self.clear_widgets()
            file_path = self.lineEdit.text().strip('"')
            data = np.loadtxt(file_path, skiprows=10)
            x_data = np.arange(4001, len(data) + 1)
            y_data = data[4000:]
            self.setup_plots(x_data, y_data)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'无法加载文件: {str(e)}')

    def plot_graph_sum(self):
        """从文件夹中的多个文件绘制累积图表"""
        try:
            self.clear_widgets()
            folder_path = self.folder_path.text().strip('"')
            if not os.path.exists(folder_path):
                raise FileNotFoundError("文件夹不存在")

            y_cumulative = None
            for file in os.listdir(folder_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder_path, file)
                    data = np.loadtxt(file_path, skiprows=10)
                    y_data = data[4000:]
                    if y_cumulative is None:
                        y_cumulative = np.zeros_like(y_data)
                    y_cumulative += y_data

            if y_cumulative is not None:
                x_data = np.arange(4001, len(y_cumulative) + 4001)
                self.setup_plots(x_data, y_cumulative, title="累加质谱图")
            else:
                raise ValueError("没有找到任何有效的文本文件进行绘图")
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e))

    def update_gaussian_fit(self):
        """准备更新高斯拟合图"""
        if not self.pending_update:
            self.pending_update = True
            self.update_timer.start()

    def on_update_timeout(self):
        """实际执行更新高斯拟合图"""
        self.update_timer.stop()
        self.pending_update = False

        if self.region and self.p1 and self.p2:
            try:
                # 暂时断开信号连接以防止递归更新
                self.region.sigRegionChanged.disconnect(self.update_gaussian_fit)

                minX, maxX = self.region.getRegion()

                # 确保 x_slice 不会超出原始数据的边界
                x_start = max(0, int(minX) - 4000)
                x_end = min(int(maxX) - 4000 + 1, len(self.spectrum_plot.yData))

                x_data = np.arange(4001, 4001 + len(self.spectrum_plot.yData))[x_start:x_end]
                y_data = self.spectrum_plot.yData[x_start:x_end]

                if len(x_data) != len(y_data):
                    print("警告：X 和 Y 数据长度不匹配")
                    return

                if len(x_data) < 3 or len(y_data) < 3:
                    print("警告：选择区域内的数据点不足，无法进行拟合。")
                    return

                # 数据预处理：移除基线
                baseline = np.min(y_data)
                y_data = y_data - baseline

                # 提供更合理的初始猜测值
                max_y = np.max(y_data)
                mean_x = np.sum(x_data * y_data) / np.sum(y_data)  # 加权平均作为中心点
                sigma_guess = (maxX - minX) / 4  # 使用区域宽度的1/4作为初始sigma
                initial_guess = [mean_x, sigma_guess, max_y]

                # 设置更宽松的参数边界
                bounds = (
                    [min(x_data), 0.1, 0],  # 下界
                    [max(x_data), (maxX - minX) / 2, max_y * 2]  # 上界
                )

                # 使用更稳健的拟合方法
                popt, pcov = curve_fit(
                    self.func_gaosi, 
                    x_data, 
                    y_data, 
                    p0=initial_guess, 
                    bounds=bounds, 
                    maxfev=10000,  # 增加最大迭代次数
                    method='trf'  # 使用更稳健的拟合方法
                )

                time_max = self.findChild(QLabel, "label_8")
                mass = self.findChild(QLabel, "label_9")
                peak_left = self.findChild(QLabel, "label_10")
                peak_right = self.findChild(QLabel, "label_11")

                time_max.setText(str(round(popt[0], 2)))  # 使用拟合得到的中心点
                mass.setText(str(round(self.transfer_2(), 2)))
                peak_left.setText(str(round(minX, 2)))
                peak_right.setText(str(round(maxX, 2)))

                self.p1.clear()
                x_interp = np.linspace(x_data[0], x_data[-1], 1000)
                y_fit = self.func_gaosi(x_interp, *popt) + baseline  # 添加回基线
                self.p1.plot(x_interp, y_fit, pen=mkPen('r', width=2), name='拟合图')

            except Exception as e:
                print(f"高斯拟合错误：{e}")
            finally:
                # 无论是否发生异常，都重新连接信号
                self.region.sigRegionChanged.connect(self.update_gaussian_fit)

    @staticmethod
    def func_gaosi(x, miu, sigma, amplitude):
        """高斯函数"""
        return amplitude * np.exp(-(x - miu)**2 / (2 * sigma**2))

    def transfer_2(self):
        """基于高斯拟合结果估算质荷比 (m/z)"""
        time_max_label = self.findChild(QLabel, "label_8")
        try:
            time_of_flight_peak = float(time_max_label.text())
        except ValueError:
            return None  # 或者你可以选择抛出异常

        calibration_constant = 1.0  # 这里只是一个示例值
        mz_ratio = calibration_constant * (time_of_flight_peak ** 2)
        return mz_ratio

    # 读取表格2数据
    def read_table_data(self):
        table = self.findChild(QTableWidget, "peakData")
        data = {
            'Species': [],
            '飞行时间': [],
            '质量数 (m/z)': [],
            '强度': [],
            '左边界': [],
            '右边界': []
        }

        for row in range(table.rowCount()):
            # 检查该行是否有任何数据
            has_data = False
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item and item.text().strip():
                    has_data = True
                    break
            
            if not has_data:
                continue

            # 读取每一列的数据
            for col in range(table.columnCount()):
                item = table.item(row, col)
                value = item.text().strip() if item and item.text().strip() else ""
                
                # 根据列名进行数据类型转换
                if col in [1, 2, 3, 4, 5]:  # 数值类型的列
                    try:
                        value = float(value) if value else 0.0
                    except ValueError:
                        value = 0.0
                
                data[table.horizontalHeaderItem(col).text()].append(value)

        df = pd.DataFrame(data)
        return df

    # 读取表格2数据
    def read_table_data_1(self):
        table = self.findChild(QTableWidget, "region")
        column1_data = []
        column2_data = []

        for row in range(table.rowCount()):
            item1 = table.item(row, 0)  # 第一列数据
            item2 = table.item(row, 1)  # 第二列数据

            if item1 is not None and item1.text():
                column1_data.append(item1.text())

            if item2 is not None and item2.text():
                column2_data.append(item2.text())

        df = pd.DataFrame({
            'time': column1_data,
            'mass': column2_data
        })
        return df

        # print(df)

    def calculate(self):
        df = self.read_table_data_1()
        poly = PolynomialFeatures(degree=2)  # 二次多项式
        x = poly.fit_transform(df[['time']])
        y = df['mass'].astype(float)  # 将y转换为浮点类型
        # print(y)
        model = LinearRegression()
        model.fit(x, y)

        # 获取系数
        coefficients = model.coef_
        intercept = model.intercept_
        y_pred = model.predict(x)
        # print(y_pred)
        # 计算残差
        # residuals = y - y_pred
        # print(residuals)
        # 计算R平方（COD）
        r2 = model.score(x, y)
        # print(r2)
        #
        # 提取回归系数
        a, b, c = coefficients[1], coefficients[2], intercept
        # print(a, b, c)
        # # 格式化输出结果
        # print("y =", f"{c}" + " + " f"{a}" "x" + " + " f"{b}" "x²")
        # print(f"R²(COD) = {r2}")

        # 在标签上设置文本内容
        result_text = f"y = {c} + {a}x + {b}x²\n, R²(COD) = {r2}"
        label = self.findChild(QLabel, "label_4")
        label.setText(result_text)
        # print("Column 1 data:", column1_data)
        # print("Column 2 data:", column2_data)

    def transfer(self):
        a = self.findChild(QLineEdit, "lineEdit_4")
        b = self.findChild(QLineEdit, "lineEdit_5")
        c = self.findChild(QLineEdit, "lineEdit_6")
        # print(a.text(), b.text(), c.text())
        # print(float(a.text()))
        time = self.findChild(QLineEdit, "lineEdit_2")
        t = time.text()
        m = float(c.text()) + float(t) * float(b.text()) + float(t) * float(t) * float(a.text())
        lineEdit = self.findChild(QLineEdit, "lineEdit_3")  # 找到名为lineEdit_7的QLineEdit对象
        lineEdit.setText(str(m))  # 将值设置为QLineEdit的文本内容

    def transfer_2(self):
        a = self.findChild(QLineEdit, "lineEdit_4")
        b = self.findChild(QLineEdit, "lineEdit_5")
        c = self.findChild(QLineEdit, "lineEdit_6")
        # print(a.text(), b.text(), c.text())
        # print(float(a.text()))
        time = self.findChild(QLabel, "label_8")
        t = time.text()
        m = float(c.text()) + float(t) * float(b.text()) + float(t) * float(t) * float(a.text())
        return m

    def save(self):
        try:
            df = self.read_table_data()
            if df.empty:
                QMessageBox.warning(self, "警告", "没有数据可以保存！")
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存文件",
                "",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", "文件保存成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件时发生错误：{str(e)}")

    def add_peak(self):
        """添加峰值到表格"""
        try:
            # 获取当前选择区域的范围
            minX, maxX = self.region.getRegion()
            
            # 获取该范围内的数据
            x_data = self.spectrum_plot.xData
            y_data = self.spectrum_plot.yData
            
            # 找到范围内的最大值点
            mask = (x_data >= minX) & (x_data <= maxX)
            x_range = x_data[mask]
            y_range = y_data[mask]
            max_index = np.argmax(y_range)
            
            # 获取峰值信息
            time = x_range[max_index]  # 飞行时间
            intensity = y_range[max_index]  # 强度
            
            # 计算质量数
            a = float(self.findChild(QLineEdit, "lineEdit_4").text())
            b = float(self.findChild(QLineEdit, "lineEdit_5").text())
            c = float(self.findChild(QLineEdit, "lineEdit_6").text())
            
            # 确保时间是有效的
            if time < 0:
                raise ValueError("飞行时间无效，不能为负数。")
            
            mz = c + b * time + a * time * time
            
            # 创建峰值数据列表，格式与自动寻峰返回的数据一致
            peak_data = [time, mz, intensity, minX, maxX]
            
            # 在表格中显示
            row = self.peakData.rowCount()
            self.peakData.insertRow(row)
            
            # 按列顺序设置数据
            self.peakData.setItem(row, 0, QTableWidgetItem("Unknown"))  # Species 列
            self.peakData.setItem(row, 1, QTableWidgetItem(f"{time:.2f}"))  # 飞行时间列
            self.peakData.setItem(row, 2, QTableWidgetItem(f"{mz:.2f}"))    # 质量数列
            self.peakData.setItem(row, 3, QTableWidgetItem(f"{intensity:.2f}"))  # 强度列
            self.peakData.setItem(row, 4, QTableWidgetItem(f"{minX:.2f}"))  # 左边界列
            self.peakData.setItem(row, 5, QTableWidgetItem(f"{maxX:.2f}"))  # 右边界列
            
            # 添加标注
            text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
            text_item.setPos(time, intensity)
            self.p2.addItem(text_item)
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'添加峰值失败: {str(e)}')

    def auto_find_peaks(self):
        """自动寻峰功能"""
        try:
            # 获取当前图表数据
            x_data = self.spectrum_plot.xData
            y_data = self.spectrum_plot.yData
            
            
            a = self.findChild(QLineEdit, "lineEdit_4")
            b = self.findChild(QLineEdit, "lineEdit_5")
            c = self.findChild(QLineEdit, "lineEdit_6")

            custom_coeffs = {
                            'a': float(c.text()),
                            'b': float(b.text()),
                            'c': float(a.text())
                        }


            config = PeakConfig(time_coefficients=custom_coeffs)
            # 创建峰值检测器实例
            peak_detector = PeakDetector(config)

            # 检测峰值
            peaks = peak_detector.detect_peaks(x_data, y_data)

            # 在表格中显示检测到的峰值
            self.display_peaks(peaks)

            # 在图表上标注检测到的峰值
            self.annotate_peaks_on_plot(peaks)

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'自动寻峰失败: {str(e)}')

    def display_peaks(self, peaks):
        """在表格中显示检测到的峰值"""
        table = self.findChild(QTableWidget, "peakData")
        table.setRowCount(len(peaks))

        # 设置表格的列标题
        headers = ["Species", "飞行时间", "质量数 (m/z)", "强度", "左边界", "右边界"]
        table.setHorizontalHeaderLabels(headers)

        for i, peak in enumerate(peaks):
            # 获取峰值数据
            time, mz, intensity, left, right = peak[0], peak[1], peak[2], peak[3], peak[4]
            
            # 按列顺序设置数据
            table.setItem(i, 0, QTableWidgetItem("Unknown"))  # Species 列
            table.setItem(i, 1, QTableWidgetItem(f"{time:.2f}"))  # 飞行时间列
            table.setItem(i, 2, QTableWidgetItem(f"{mz:.2f}"))    # 质量数列
            table.setItem(i, 3, QTableWidgetItem(f"{intensity:.2f}"))  # 强度列
            table.setItem(i, 4, QTableWidgetItem(f"{left:.2f}"))  # 左边界列
            table.setItem(i, 5, QTableWidgetItem(f"{right:.2f}"))  # 右边界列

    @staticmethod
    def mz_to_time(mz: float, a: float, b: float, c: float) -> float:
        """将质量数转换为飞行时间"""
        # 这里假设 a, b, c 是已知的时间系数
        # 需要求解二次方程 a + b * time + c * time^2 = mz
        # 这可以通过求解二次方程来实现
        discriminant = b**2 - 4*c*(a - mz)
        if discriminant < 0:
            raise ValueError("无解")
        time1 = (-b + np.sqrt(discriminant)) / (2*c)
        time2 = (-b - np.sqrt(discriminant)) / (2*c)
        # 返回正值的解
        return time1 if time1 > 0 else time2

    def annotate_peaks_on_plot(self, peaks):
        """在质谱图上标注检测到的峰值"""
        for peak in peaks:
            time, mz, intensity = peak[0], peak[1], peak[2]  # 直接使用返回的飞行时间
            try:
                text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
                text_item.setPos(time, intensity)  # 使用飞行时间作为x坐标
                self.p2.addItem(text_item)
            except Exception as e:
                print(f"无法标注峰值 {mz}: {e}")

    def update_region(self, viewRange):
        """更新选择区域"""
        minX, maxX = viewRange[0]
        self.region.setRegion([minX, maxX])

    def on_tab_changed(self):
        """选项卡切换事件处理"""
        # 检查当前选项卡是否为"质谱定标"
        if self.peakResult.tabText(self.peakResult.currentIndex()) == "质谱定标":
            self.label_4.setVisible(True)  # 显示文本框
            self.pushButton_3.setVisible(True)
        else:
            self.label_4.setVisible(False)  # 隐藏文本框
            self.pushButton_3.setVisible(False)

    def calculate_abc(self):
        """计算ABC参数"""
        df = self.read_table_data_1()
        poly = PolynomialFeatures(degree=2)  # 二次多项式
        x = poly.fit_transform(df[['time']])
        y = df['mass'].astype(float)  # 将y转换为浮点类型

        model = LinearRegression()
        model.fit(x, y)

        # 获取系数
        coefficients = model.coef_
        intercept = model.intercept_
        r2 = model.score(x, y)

        # 提取回归系数
        a, b, c = coefficients[1], coefficients[2], intercept

        # 在标签上设置文本内容
        result_text = f"y = {c} + {a}x + {b}x²\n, R²(COD) = {r2}"
        label = self.findChild(QLabel, "label_4")
        label.setText(result_text)

    def show_peak_context_menu(self, position):
        """显示峰值表格的上下文菜单"""
        menu = QMenu()
        add_action = menu.addAction("添加峰值")
        delete_action = menu.addAction("删除峰值")
        edit_action = menu.addAction("修改峰值")
        
        # 获取点击的行
        row = self.peakData.rowAt(position.y())
        
        # 根据是否选中行来启用/禁用菜单项
        delete_action.setEnabled(row >= 0)
        edit_action.setEnabled(row >= 0)
        
        action = menu.exec_(self.peakData.mapToGlobal(position))
        
        if action == add_action:
            self.add_peak_manually()
        elif action == delete_action and row >= 0:
            self.delete_peak(row)
        elif action == edit_action and row >= 0:
            self.edit_peak(row)

    def add_peak_manually(self):
        """手动添加峰值"""
        dialog = PeakDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            peak_data = dialog.get_peak_data()
            self.add_peak_to_table(peak_data)
            self.annotate_peak(peak_data)

    def delete_peak(self, row):
        """删除峰值"""
        reply = QMessageBox.question(self, '确认删除', 
                                   '确定要删除这个峰值吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.peakData.removeRow(row)
            self.update_plot()  # 更新图表

    def edit_peak(self, row):
        """编辑峰值"""
        current_data = []
        for col in range(self.peakData.columnCount()):
            item = self.peakData.item(row, col)
            current_data.append(item.text() if item else "")
            
        dialog = PeakDialog(self, current_data)
        if dialog.exec_() == QDialog.Accepted:
            peak_data = dialog.get_peak_data()
            self.update_peak_in_table(row, peak_data)
            self.update_plot()  # 更新图表

    def add_peak_to_table(self, peak_data):
        """将峰值添加到表格"""
        row = self.peakData.rowCount()
        self.peakData.insertRow(row)
        
        self.peakData.setItem(row, 0, QTableWidgetItem(peak_data['species']))
        self.peakData.setItem(row, 1, QTableWidgetItem(f"{peak_data['time']:.2f}"))
        self.peakData.setItem(row, 2, QTableWidgetItem(f"{peak_data['mz']:.2f}"))
        self.peakData.setItem(row, 3, QTableWidgetItem(f"{peak_data['intensity']:.2f}"))
        self.peakData.setItem(row, 4, QTableWidgetItem(f"{peak_data['left']:.2f}"))
        self.peakData.setItem(row, 5, QTableWidgetItem(f"{peak_data['right']:.2f}"))

    def update_peak_in_table(self, row, peak_data):
        """更新表格中的峰值"""
        self.peakData.setItem(row, 0, QTableWidgetItem(peak_data['species']))
        self.peakData.setItem(row, 1, QTableWidgetItem(f"{peak_data['time']:.2f}"))
        self.peakData.setItem(row, 2, QTableWidgetItem(f"{peak_data['mz']:.2f}"))
        self.peakData.setItem(row, 3, QTableWidgetItem(f"{peak_data['intensity']:.2f}"))
        self.peakData.setItem(row, 4, QTableWidgetItem(f"{peak_data['left']:.2f}"))
        self.peakData.setItem(row, 5, QTableWidgetItem(f"{peak_data['right']:.2f}"))

    def update_plot(self):
        """更新图表显示"""
        # 清除现有标注
        for item in self.p2.items[:]:
            if isinstance(item, pg.TextItem):
                self.p2.removeItem(item)
        
        # 重新添加所有峰值标注
        for row in range(self.peakData.rowCount()):
            try:
                # 使用飞行时间作为 x 坐标
                time = float(self.peakData.item(row, 1).text())  # 第2列是飞行时间
                mz = float(self.peakData.item(row, 2).text())    # 第3列是质量数
                intensity = float(self.peakData.item(row, 3).text())  # 第4列是强度
                
                # 使用飞行时间作为 x 坐标进行标注
                text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
                text_item.setPos(time, intensity)  # 直接使用飞行时间作为 x 坐标
                self.p2.addItem(text_item)
            except (ValueError, AttributeError) as e:
                print(f"更新峰值标注时出错: {e}")

    def annotate_peak(self, peak_data):
        """在图表上标注单个峰值"""
        try:
            text_item = pg.TextItem(text=f"{peak_data['mz']:.2f}", color='red', anchor=(0.5, 1.5))
            text_item.setPos(peak_data['time'], peak_data['intensity'])
            self.p2.addItem(text_item)
        except Exception as e:
            print(f"无法标注峰值: {e}")

    def update_all_peaks(self):
        """更新所有峰值标注"""
        try:
            # 清除现有标注
            for item in self.p2.items[:]:
                if isinstance(item, pg.TextItem):
                    self.p2.removeItem(item)
            
            # 从表格中读取所有峰值数据并重新标注
            for row in range(self.peakData.rowCount()):
                try:
                    # 获取峰值数据
                    time = float(self.peakData.item(row, 1).text())
                    mz = float(self.peakData.item(row, 2).text())
                    intensity = float(self.peakData.item(row, 3).text())
                    
                    # 添加标注
                    text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
                    text_item.setPos(time, intensity)
                    self.p2.addItem(text_item)
                    
                except (ValueError, AttributeError) as e:
                    print(f"更新第 {row+1} 行峰值标注时出错: {e}")
                
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'更新峰值标注失败: {str(e)}')

    def clear_peak_data(self):
        """清除峰值数据表格中的所有数据"""
        try:
            # 清除表格数据
            self.peakData.setRowCount(0)
            
            # 清除图表上的峰值标注
            if self.p2:
                for item in self.p2.items[:]:
                    if isinstance(item, pg.TextItem):
                        self.p2.removeItem(item)
            
            # 清除高斯拟合图
            if self.p1:
                self.p1.clear()
            
            # 清除标签显示
            time_max = self.findChild(QLabel, "label_8")
            mass = self.findChild(QLabel, "label_9")
            peak_left = self.findChild(QLabel, "label_10")
            peak_right = self.findChild(QLabel, "label_11")
            
            if time_max: time_max.setText("")
            if mass: mass.setText("")
            if peak_left: peak_left.setText("")
            if peak_right: peak_right.setText("")
            
            QMessageBox.information(self, "成功", "峰值数据已清除")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清除数据时发生错误：{str(e)}")



