import matplotlib.pyplot as plt
import numpy as np


def plotTest():
    x = np.arange(10)
    y = x**2

    plt.subplot(1,2,1) # 1행2열 중 첫번째
    plt.plot(x,y, 'r')

    plt.subplot(1,2,2) # 1행2열 중 두번째
    plt.plot(x, np.sin(x))
    plt.xticks([]) # x좌표 눈금 제거
    
    plt.show()

if __name__ == '__main__':
    plotTest()
