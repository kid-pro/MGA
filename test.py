import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from modell import Generator1  # 确保 Generator 类正确导入
import scipy.io as sio
from scipy.io import loadmat
MIN_VAL = -7100  # 设置固定的最小值
MAX_VAL = 2600   # 设置固定的最大值
# 读取MATLAB处理好的数据

data1 = loadmat(r'D:\Data——surfer\MATLAB\163\MATLAB_code_11\extend_data\sc1up_data.mat')
data2 = loadmat(r'D:\Data——surfer\MATLAB\163\MATLAB_code_11\extend_data\sc1Turedown_data.mat')
# # 将数据转换为矩阵列表
matrix_down_list = [np.array(matrix) for matrix in data1['chains2']]  # 这是一个for循环
target_data = [np.array(matrix) for matrix in data2['chains3']]
def load_model(model_path):

    #model = Generator1()
    input_nc = 1
    output_nc = 1
    num_downs = 7
    model = Generator2(input_nc, output_nc, num_downs)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_data(directory):
    data = []
    files = sorted(os.listdir(directory))
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            matrix = np.loadtxt(file_path)
            normalized = 2 * ((matrix - MIN_VAL) / (MAX_VAL - MIN_VAL)) - 1  # 使用固定的最大最小值进行归一化
            data.append(normalized)
    return np.array(data), files

def predict_and_evaluate(model, input_data):
    input_tensors = torch.Tensor(input_data).unsqueeze(1)
    dataset = TensorDataset(input_tensors)
    loader = DataLoader(dataset, batch_size=1)

    mse = torch.nn.MSELoss()
    total_mse = 0
    count = 0
    i = 0
    predicted_list = []
    for batch, (inputs,) in enumerate(loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        outputs = model(inputs)
        predicted = outputs.squeeze().cpu().detach().numpy()

        predicted_list.append(predicted)
    return predicted_list

model_path = 'model2.pth'

test_input_directory = r'D:\pcyang\ccn-cl\test\up'



model = load_model(model_path)
test_data= matrix_down_list
predicted_list = predict_and_evaluate(model, test_data)

sio.savemat('CNNscdown_data2.mat', {'down_data': predicted_list})






