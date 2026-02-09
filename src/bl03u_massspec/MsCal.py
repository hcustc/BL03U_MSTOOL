import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 将数据导入excel文件
# data = {'time': [6127.755, 8457.441, 10068.66, 14058.12, 14149.07, 14238.99, 14418.0],
#         'mass': [18.01002, 31.98983, 43.98983, 81.91294, 82.91359, 83.91096, 85.91006]}
# df = pd.DataFrame(data)
#
# file_name = 'data_exp.xlsx'  # Excel文件的名称
# sheet_name = 'Sheet1'  # Excel工作表的名称
# df.to_excel(file_name, sheet_name=sheet_name, index=False)  # index=False防止将行索引也写入Excel

# 导入数据
file_name = 'E:/BL03U_MassSpectrumTool/data/data_exp.xlsx'  # Excel文件的名称
sheet_name = 'Sheet1'  # Excel工作表的名称
df = pd.read_excel(file_name, sheet_name=sheet_name)
print(df)

# 使用线性回归模型拟合数据
poly = PolynomialFeatures(degree=2)  # 二次多项式
x = poly.fit_transform(df[['time']])
y = df['mass']
print(y)
model = LinearRegression()
model.fit(x, y)

# 获取系数
coefficients = model.coef_
intercept = model.intercept_
y_pred = model.predict(x)
# 计算残差
residuals = y - y_pred
# 计算R平方（COD）
r2 = model.score(x, y)

# 提取回归系数
a, b, c = coefficients[1], coefficients[2], intercept
# 格式化输出结果
# print("y =", f"{c}" + " + " f"{a}" "x" + " + " f"{b}" "x²")
# print(f"R²(COD) = {r2}")

# 存储abc的值
with open('data/abc.txt', 'w') as f:
    f.write(f"{a} {b} {c} ")
    f.write('\n')
    f.write("y = {}x + {}x**2 + {}".format(a, b, c))
    f.write('\n')
    f.write("R平方(COD) = {}".format(r2))

# 绘图
fig = plt.figure(figsize=(12, 10))
spec = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.5, 1])

# 设置
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False  # 解决负号无法显示的问题
}
rcParams.update(config)


# 拟合图
ax1 = fig.add_subplot(spec[0, :2])
ax1.scatter(df['time'], df['mass'], color='red')
title1 = 'y = {}$x^2$ + {}x + {}, $R^2$ = {}'.format(a, b, c, r2)
x = np.linspace(5000, 15000, 2000)
y = b * x**2 + a * x + c
ax1.plot(x, y, color='blue')
ax1.set_xlabel('time')
ax1.set_ylabel('m/z')
ax1.set_title(title1, fontsize=15)

# 残差图
ax2 = fig.add_subplot(spec[1, 0])
x1 = df['time']
y1 = x1 * 0
ax2.plot(x1, y1, color='blue')
ax2.scatter(df['time'], residuals, color='blue')
ax2.set_xlabel('time')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Time')

# 残差图（质量）
ax3 = fig.add_subplot(spec[1, 1])
x2 = df['mass']
y2 = x2 * 0
ax3.plot(x2, y2, color='blue')
ax3.scatter(df['mass'], residuals, color='blue')
ax3.set_xlabel('mass')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs m/z')

# 储存图片
fig.savefig('data/拟合图.png', dpi=300, bbox_inches='tight')
