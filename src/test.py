import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

x = np.linspace(0, 10, 100)
y = gaussian(x, 2.5, 5, 1) + np.random.normal(size=len(x))

p0 = [max(y), x[np.argmax(y)], (max(x) - min(x)) / 10]
coeff, _ = curve_fit(gaussian, x, y, p0=p0)

plt.plot(x, y, 'b+:', label='data')
plt.plot(x, gaussian(x, *coeff), 'ro:', label='fit')
plt.legend()
plt.show()
