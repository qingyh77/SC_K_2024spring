from matplotlib import pyplot
from pandas import Index
from torch import nn
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import io
import time
import math


##
#E:\hqy\Sigma_cpd\采样位置固定
#E:\hqy\Mat_R2
rho = 0.16 #0.1:0.02:0.2
sigma_index = 3
R_num = 6
input_index = 6
v=0.1


experiment = 'Mat_R'+str(R_num) + '_v' + str(v) + '_SNR20'
#file_path = 'E:\\hqy\\Sigma_net\\采样位置固定\\'
#file_path = 'F:\\hqy\\Mat_R8_SNR20_sinc2C\\'
file_path = 'E:\\hqy\\'+experiment + '\\Mat_rho'+str(rho)+'_R6_SNR20_sinc2C\\'
#file_path = 'F:\\黄清扬\\Sigma_cpd\\NET\\'
file_name = 'NMFCPD_S_C_rho'+str(rho)+'_' + 'R%d'%R_num +'_%d.mat'%input_index
input_filename = file_path + file_name
Rcp = 5
batch_size = 10
input_window = 20
output_window = 1

## 准备数据
data = scio.loadmat(input_filename)  #C:\Users\94528\Desktop\semester7\SPL_2022_CBTD\SPL_2022_CBTD\c3_LL1\matDA
type(data)
C_data = np.array(data['C_icpd'])
Time_len, R = C_data.shape
test_data_size = int(0.1 * Time_len)
train_data = C_data[:-test_data_size,:]
test_data = C_data[-test_data_size:,:]



# 数据预处理
from sklearn.preprocessing import MinMaxScaler
#归标准化，变成[-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
# 格式处理
#train_data_normalized = scaler.fit_transform(train_data)
train_data_normalized = train_data
train_data_normalized = torch.FloatTensor(train_data_normalized)

# 数据分类，分成训练数据和测试数据

def create_inout_sequences(input_data, input_window, output_window):
    inout_seq = []
    L, _ = input_data.shape
    block_num = L - input_window - output_window + 1
    for i in range(block_num):
        train_seq = input_data[ i:i + input_window]
        train_label = input_data[ i + output_window:i + input_window + output_window]
        if i ==0:
            temp_seq = torch.unsqueeze(train_seq, 0)
            temp_label = torch.unsqueeze(train_label, 0)
        else:
            temp_seq = torch.cat((temp_seq, torch.unsqueeze(train_seq, 0)),0)
            temp_label = torch.cat((temp_label, torch.unsqueeze(train_label, 0) ), 0)

    inout_seq = torch.stack((temp_seq, temp_label), 1)
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, input_window,output_window)

#test_data_normalized = scaler.fit_transform(test_data)
test_data_normalized = test_data
test_data_normalized = torch.FloatTensor(test_data_normalized)
test_inout_seq = create_inout_sequences(test_data_normalized, input_window, output_window)

## 分batch
def get_batch(input_data, i, batch_size):
    R = input_data.size(-1)
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[i:i+batch_len,:,:]
    # test1 = torch.stack([item[0] for item in data])
    # test2 = torch.stack([item[1] for item in data])
    test1 = data[:,0,:,:]
    test2 = data[:,1,:,:]
    # ( seq_len, batch, 1 ) , 1 is feature size

    input = torch.stack([item[0] for item in data])
    target = torch.stack([item[1] for item in data])
    return input, target


def train(train_data, batch_size):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()
    save_output = []
    for batch, i in enumerate(range(0, train_data.size(0), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        data, targets = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, data.size(0), model.hidden_layer_size),
                             torch.zeros(1, data.size(0), model.hidden_layer_size))
        output = model(data)
        loss = criterion(output, targets)

        ## 过往数据可以参与训练，但想要观察的还是预测数据的效果，因此只需要记录最后一个预测结果
        save_output = output[-1:,-1,:]
        if train_data.size(0)-i<=batch_size:
            data_tt = targets[-1:,-1:,:]
            model.hidden_cell = (torch.zeros(1, data_tt.size(0), model.hidden_layer_size),
                                     torch.zeros(1, data_tt.size(0), model.hidden_layer_size))
            output2 = model(data_tt)
            save_output2 = output2[-1,-1:,:]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()

        # if output.size(0)<batch_size:
        #     output = torch.cat( (output,torch.zeros(batch_size-output.size(0),input_window,Rcp)) ,0 )
        # if batch == 0:
        #     save_output = torch.unsqueeze(output,0)
        # else:
        #     save_output = torch.cat((save_output,torch.unsqueeze(output,0)),0)

        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        # optimizer.step()

        # total_loss += loss.item()
        log_interval = max(int(len(train_data) / batch_size / 5), 1)
        if epoch%10 == 0 and batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return save_output,save_output2




## 搭建网络

class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        #test = input_seq.view(-1, 1, 5)
        #print(test)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        output = predictions
        return output


# 实例化
model = LSTM(input_size=R, hidden_layer_size=100, output_size=R)
# model.add_module('linear', nn.Linear(100, 1))
criterion = nn.MSELoss()
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
print(model)



# 训练

# print(test_inputs)




# for epoch in range(1, epochs+1):
#     print("----第{}轮训练开始----".format(epoch))
#     C_predict_epoch = train(train_inout_seq)
#
#     # if epoch == 1:
#     #     C_save = torch.unsqueeze(C_predict_epoch,0)
#     # else:
#     #     C_save = torch.cat((C_save,torch.unsqueeze(C_predict_epoch,0)),0)
#     if epoch%5 == 0:
#         if epoch == 5:
#             C_save = torch.unsqueeze(C_predict_epoch,0)
#         else:
#             C_save = torch.cat((C_save,torch.unsqueeze(C_predict_epoch,0)),0)
epochs = 10
ts_start = input_window+batch_size
for tt in range(ts_start,Time_len):
    print("----第{}时刻开始----".format(tt))
    window_len = ts_start+150
    window_left = max(0,tt-window_len)
    C_data_now = C_data[window_left:(tt),:]
    C_data_now = torch.FloatTensor(C_data_now)
    input_window_now = min(tt, input_window)
    C_seq = create_inout_sequences(C_data_now, input_window_now,output_window)
    batch_size_now = min(tt, batch_size)
    for epoch in range(1, epochs+1):
        # print("----第{}时刻，第{}轮训练开始----".format(tt, epoch))
        _, C_predict_epoch2 = train(C_seq, batch_size_now)
        if epoch %5 ==0:
            if epoch == 5:
                #C_save_t1 = C_predict_epoch
                C_save_t2 = C_predict_epoch2
            else:
                #C_save_t1 = torch.cat((C_save_t1, C_predict_epoch),0)
                C_save_t2 = torch.cat((C_save_t2, C_predict_epoch2), 0)

    if tt == ts_start:
        #C_save1 = torch.unsqueeze(C_save_t1, 0)
        C_save2 = torch.unsqueeze(C_save_t2, 0)
    else:
        #C_save1 = torch.cat((C_save1, torch.unsqueeze(C_save_t1, 0)),0 )
        C_save2 = torch.cat((C_save2, torch.unsqueeze(C_save_t2, 0)), 0)


# C_save1 = C_save1.detach().numpy()
# filename2 = 'ICPD_LSTM_sigma%d_lr=1e-4_online1_Timelen.mat' % input_index
# save_filename = file_path + filename2
# io.savemat(save_filename, {'C_pred':C_save1})

C_save2 = C_save2.detach().numpy()
filename3 = 'NMFCPD_LSTM_rho'+str(rho) + '_lr=1e-4_online_%d.mat'%input_index
save_filename = file_path +'NEToutput\\' + filename3
io.savemat(save_filename, {'C_pred':C_save2})


