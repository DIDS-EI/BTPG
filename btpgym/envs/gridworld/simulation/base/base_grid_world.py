import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer


class GridWidget(QWidget):
    def __init__(self, grid_size=(10, 10), agent_pos=[0, 0], object_pos=[9, 9]):
        super().__init__()
        self.grid_size = grid_size
        self.agent_pos = agent_pos
        self.object_pos = object_pos
        self.cell_size = 50  # 每个格子的像素大小
        self.initUI()

    def initUI(self):
        self.setMinimumSize(self.grid_size[0] * self.cell_size, self.grid_size[1] * self.cell_size)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.moveAgent)
        self.timer.start(500)  # 设置移动的时间间隔为500毫秒

    def moveAgent(self):
        # 简单的移动逻辑：每次朝目标物体的方向移动一格
        if self.agent_pos[0] < self.object_pos[0]:
            self.agent_pos[0] += 1
        elif self.agent_pos[0] > self.object_pos[0]:
            self.agent_pos[0] -= 1

        if self.agent_pos[1] < self.object_pos[1]:
            self.agent_pos[1] += 1
        elif self.agent_pos[1] > self.object_pos[1]:
            self.agent_pos[1] -= 1

        self.update()  # 更新网格的显示

        if self.agent_pos == self.object_pos:
            self.timer.stop()  # 如果到达目标位置，则停止定时器

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawGrid(qp)
        qp.end()

    def drawGrid(self, qp):
        color_agent = QColor(255, 0, 0)  # 智能体颜色
        color_object = QColor(0, 0, 255)  # 物体颜色
        color_grid = QColor(200, 200, 200)  # 网格颜色

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if [x, y] == self.agent_pos:
                    qp.fillRect(*rect, QBrush(color_agent))
                elif [x, y] == self.object_pos:
                    qp.fillRect(*rect, QBrush(color_object))
                qp.setPen(color_grid)
                qp.drawRect(*rect)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Grid Environment Visualization with Movement')
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        self.grid_widget = GridWidget()
        self.setCentralWidget(self.grid_widget)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


class GridEnvironment:
    def __init__(self, width, height, visualize=False):
        self.width = width
        self.height = height
        self.visualize = visualize
        self.agent_position = [0, 0]  # 假定智能体初始位置
        self.object_position = [width - 1, height - 1]  # 假定物体初始位置
        if visualize:
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication([])
            self.window = MainWindow(self)

    def move_agent(self, direction):
        """
        根据给定的方向移动智能体。
        direction: 'up', 'down', 'left', 'right'
        """
        if direction == 'up' and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif direction == 'down' and self.agent_position[1] < self.height - 1:
            self.agent_position[1] += 1
        elif direction == 'left' and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif direction == 'right' and self.agent_position[0] < self.width - 1:
            self.agent_position[0] += 1

        self.print_positions()
        if self.visualize:
            self.window.update_positions(self.agent_position, self.object_position)

    def print_positions(self):
        print(f"智能体位置: {self.agent_position}, 物体位置: {self.object_position}")


# 可视化部分的主窗口和网格小部件类将类似于之前的实现，但这里需要修改以接收GridEnvironment对象并从中读取位置信息。
# 为了简化，我们这里不重复那部分代码。你可以根据之前提供的代码示例来整合。



if __name__ == '__main__':
    # main()

    # env = GridEnvironment(10, 10, visualize=False)
    # env.move_agent('right')
    # env.move_agent('down')

    env = GridEnvironment(10, 10, visualize=True)
    # 启动Qt事件循环
    env.app.exec_()