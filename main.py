import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import scipy.io as sio
# 确保模型模块正确导入
from modell import Generator2

loss_data_matrix = np.zeros((500, 1))  #创建一个空矩阵来存储损失值
i = 1
# 设置固定的最大值和最小值
MIN_VAL = -7100
MAX_VAL = 2600
# 读取MATLAB处理好的数据
data1 = loadmat(r'D:\Data——surfer\MATLAB\163\MATLAB_code_11\extend_data\sc1up_data.mat')
data2 = loadmat(r'D:\Data——surfer\MATLAB\163\MATLAB_code_11\extend_data\sc1down_data.mat')
# # 将数据转换为矩阵列表
matrix_down_list = [np.array(matrix) for matrix in data1['chains2']]  # 这是一个for循环
target_data = [np.array(matrix) for matrix in data2['chains1']]


def load_data(input_directory, target_directory):
    input_data = []
    target_data = []
    input_files = os.listdir(input_directory)

    for input_file in input_files:
        if input_file.endswith('.txt'):
            prefix, suffix = input_file.split('_')
            target_file = 'down_' + suffix  # 构建对应的目标文件名
            input_path = os.path.join(input_directory, input_file)
            target_path = os.path.join(target_directory, target_file)

            if os.path.exists(target_path):
                input_matrix = np.loadtxt(input_path)
                target_matrix = np.loadtxt(target_path)

                # 使用固定的最大值和最小值进行归一化处理
                input_normalized = 2 * ((input_matrix - MIN_VAL) / (MAX_VAL - MIN_VAL)) - 1
                target_normalized = 2 * ((target_matrix - MIN_VAL) / (MAX_VAL - MIN_VAL)) - 1

                input_data.append(input_normalized)
                target_data.append(target_normalized)

    return np.array(input_data), np.array(target_data)


# 训练网络的函数
def train_network(input_train, target_train, input_test, target_test, epochs, batch_size, learning_rate):
    # 将数据转换为张 量并添加通道维度
    input_train_tensors = torch.Tensor(input_train).unsqueeze(1)
    target_train_tensors = torch.Tensor(target_train).unsqueeze(1)
    input_test_tensors = torch.Tensor(input_test).unsqueeze(1)
    target_test_tensors = torch.Tensor(target_test).unsqueeze(1)

    # 创建训练和测试的数据加载器
    train_dataset = TensorDataset(input_train_tensors, target_train_tensors)
    test_dataset = TensorDataset(input_test_tensors, target_test_tensors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    #model = Generator1()
    input_nc = 1
    output_nc = 1
    num_downs = 7
    model = Generator2(input_nc, output_nc, num_downs)
    if torch.cuda.is_available():
        model.cuda()  # 如果可用，将模型转移到GPU

    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, targets)  # 计算损失
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            running_loss += loss.item() * inputs.size(0)

        # 计算并打印每个epoch的平均训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}')

        # 验证模型性能
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for inputs, targets in test_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item() * inputs.size(0)

            # 计算并打印验证损失
            validation_loss = validation_loss / len(test_loader.dataset)
            print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {validation_loss:.4f}')
            loss_data_matrix[epoch] = validation_loss

    # 保存训练好的模型
    torch.save(model.state_dict(), 'model1.pth')
    print('Training complete. Model saved.')
    sio.savemat('loss_data_matrix_500_16_0.002_modl1.mat', {'down_data': loss_data_matrix})#存储的未归一化的数据，应该将其归一化后才是真实损失函数曲线

# 执行代码，确保使用正确的数据路径
# input_train, target_train = load_data(r'D:\Data——surfer\py\cnn\code\data\train_up',
#                                       r'D:\Data——surfer\py\cnn\code\data\targt_down')
# input_test, target_test = load_data(r'D:\Data——surfer\py\cnn\code\data\val_up',
#                                     r'D:\Data——surfer\py\cnn\code\data\val_down')


input_train = matrix_down_list
target_train = target_data
input_test = matrix_down_list
target_test = target_data
# 训练网络
train_network(input_train, target_train, input_test, target_test, epochs=500, batch_size=16, learning_rate=0.002)
