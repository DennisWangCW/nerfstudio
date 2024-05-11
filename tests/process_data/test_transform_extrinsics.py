import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





if __name__ == "__main__":
    # 读取 JSON 文件
    with open('data.json', 'r') as f:
        data = json.load(f)

    # 创建三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取每个点的位置和方向
    for frame in data['frames']:
        file_path = frame['file_path']
        transform_matrix = np.array(frame['transform_matrix'])
        position = transform_matrix[:3, 3]
        direction = transform_matrix[:3, :3] @ np.array([0, 0, 1])  # 取 Z 轴方向作为箭头方向

        # 绘制点
        ax.scatter(*position, c='r')

        # 绘制箭头
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=0.1)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()
