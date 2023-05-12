import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QTextCursor
from mainwindow import  Ui_MainWindow
import game as gm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
## 线程类
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QWaitCondition, QMutex
from PyQt5.QtCore import QAbstractTableModel, Qt
import time

def ppp(data,tabel):

    # 获取dataframe的行列
    model = QStandardItemModel(data.shape[0], data.shape[1])
    # 设置水平方向的标头内容
    model.setHorizontalHeaderLabels(data.columns.values)
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            sss = data[data.columns.values[column]][data.index.values[row]]
            if isinstance(sss,float):
                sss = round(sss,7)
            sss = str(sss)
            item = QStandardItem(sss)
            # 设置每个位置的文本值
            model.setItem(row, column, item)

    # 实例化表格视图，设置模型为自定义的模型
    tabel.setModel(model)
    #  设置它不能被编辑
    tabel.setEditTriggers(QAbstractItemView.NoEditTriggers)

    # # 水平方向标签拓展剩下的窗口部分，填满表格
    # tabel.horizontalHeader().setStretchLastSection(True)
    # # 水平方向，表格大小拓展到适当的尺寸
    # tabel.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    tabel.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)





class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):  #这里面增加响应函数
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.random.clicked.connect(self.random_game)
        self.identical.clicked.connect(self.identical_game)
        self.zerosum.clicked.connect(self.zerosum_game)
        self.learn.clicked.connect(self.learn_plot)
        self.game = -1


    def random_game(self): ## 随机实例
        if self.get_para():
            self.game = gm.SGame.random_game(num_states=self.Num_states, num_players=self.Num_players,
                                             num_actions=self.Num_actions,delta=self.Delta, seed=self.Seed)
            self.print_game()

    def identical_game(self): ## 随机实例
        if self.get_para():
            self.game = gm.SGame.random_identical_interest_game(num_states=self.Num_states, num_players=self.Num_players,
                                             num_actions=self.Num_actions,delta=self.Delta, seed=self.Seed)
            self.print_game()

    def zerosum_game(self): ## 随机实例
        if self.get_para():
            self.game = gm.SGame.random_zero_sum_game(num_states=self.Num_states, num_players=self.Num_players,
                                             num_actions=self.Num_actions,delta=self.Delta, seed=self.Seed)
            self.print_game()

    def get_para(self):  ## 获取参数
        list = [self.num_states.text(),self.num_players.text(),self.num_actions.text(),self.delta.text(),self.seed.text()]
        if '' in list:
            return False
        self.Num_states = int(list[0])
        self.Num_players = int(list[1])
        self.Num_actions = int(list[2])
        self.Delta = float(list[3])
        self.Seed = int(list[4])
        return True

    def get_trainpara(self):
        list = [self.episode.text(),self.epsilon.text()]
        if '' in list:
            return False
        self.Episode = int(list[0])
        self.Epsilon = float(list[1])
        self.Policy = self.policy.currentText()
        return True


    def print_game(self): ## 打印信息
        df = gm.game_to_table(self.game)
        ppp(df,self.tabel)


    def learn_plot(self):  ## 学习函数
        if self.game == -1:
            return
        if self.get_trainpara():
            if self.Policy == 'Eps Greedy':
                self.Policy = gm.EpsGreedy()
            elif self.Policy == 'Logit Best Response':
                self.Policy = gm.LogitBestResponse()
            elif self.Policy == 'Smooth Time Response':
                self.Policy = gm.SmoothTimeResponse()
            elif self.Policy == 'Smooth Time Root Response':
                self.Policy = gm.SmoothTimeRootResponse()

            gm.learn_and_plot(self.game, episode=self.Episode, start_state=0, policy=self.Policy, eps=self.Epsilon)
            plt.show()


app = QApplication(sys.argv)
main = MyMainForm()
main.show()
sys.exit(app.exec())

# game1 = gm.SGame.random_game(num_states=3, num_players=2, num_actions=3, delta=0.95, seed=123)
# gm.print_game(game1)
# gm.learn_and_plot(game1, episode=1000, start_state=0, policy=gm.EpsGreedy(), eps=0.1)
# plt.show()

