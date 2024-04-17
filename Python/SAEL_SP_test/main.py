from models import AutoEncoderLayer
from models import *
import torch
from torch import nn
import numpy as np
import scipy.io as scio
from scipy import io
import time
import math

""" 
导入文件：
    file_path:文件路径
    file_method:张量补全/分解方法
    file_name:导入数据文件名
    input_filename：最终地址
    C_data：最终导出数据
"""



# #E:\hqy\Sigma_cpd\采样位置固定
# sigma_index = 2
input_index = 2  #标准差
sourceNum = 2
snr = 10
#file_path = 'E:\\hqy\\Sigma_net\\采样位置固定\\'
file_path_1 = 'E:\\hqy\\Mat_R%d_' %sourceNum
file_path_2 = 'SNR%d_sinc2C\\' %snr
file_path = file_path_1+file_path_2
#file_path = 'F:\\黄清扬\\Sigma_cpd\\NET\\'
file_method = 'IDW'  #ICPD_C/DWCPD_C/IDW/TPS
file_name_1 = file_method+'_sigma%d' % input_index
file_name_2 = '_R%d.mat' %sourceNum
file_name = file_name_1+file_name_2
input_filename = file_path + file_name

# 正式导入
data = scio.loadmat(input_filename)
type(data)
data_name = 'Xhat_idw' #C_icpd/C_dwcpd/Xhat_idw/Xhat_tps
C_data = np.array(data[data_name])
if data_name[0] == 'C':
    Time_len, R = C_data.shape
else:
    I, J, Time_len = C_data.shape

data_tensor = torch.FloatTensor(C_data)  # L*F*T的张量，第一个维度为位置
data_tensor = data_tensor.permute([0,2,1])

## SAEL-SP的输入无位置信息



"""
数据处理函数
     create_inout_sequences：
     get_batch：
"""
def create_inout_sequences(input_data, input_window, output_window):
    inout_seq = []
    L = input_data.size(0)
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


## get_batch略不同于online_LSTM
def get_batch(input_data, i, batch_size):
    R = input_data.size(-1)
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[i:i+batch_len]
    # test1 = torch.stack([item[0] for item in data])
    # test2 = torch.stack([item[1] for item in data])
    test1 = data[:,0,:,:]
    test2 = data[:,1,:,:]
    # ( seq_len, batch, 1 ) , 1 is feature size

    input = torch.stack([item[0] for item in data])
    target = torch.stack([item[1] for item in data])
    return input, target


"""
搭建网络
     StackedAE
     SAEL
"""
class SAEL(nn.Module):
    def __init__(self, layers_list=None,input_size=1, hidden_layer_size=100,output_size=1,num_layers = 3 ):
        super(SAEL, self).__init__()
        self.layers_list = layers_list
        self.initialize()
        self.init_hidden()
        self.encoder_1 = self.layers_list[0]
        self.encoder_2 = self.layers_list[1]

        self.lstm0 = nn.LSTMCell(input_size=input_size,hidden_size=128)
        self.lstm1 = nn.LSTMCell(input_size=128,hidden_size=64)
        self.lstm2 = nn.LSTMCell(input_size=64,hidden_size=32)

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size= hidden_layer_size,num_layers=num_layers, bidirectional= True)
        self.fc = nn.Linear(2* hidden_layer_size, output_size)

    def initialize(self):
        for layer in self.layers_list:
            # assert isinstance(layer, AutoEncoderLayer)
            layer.is_training_layer = False
            # for param in layer.parameters():
            #     param.requires_grad = True

    def init_hidden(self):
        return ()


    def forward(self,x):
        out = x
        out = self.encoder_1(out)
        out = self.encoder_2(out)

        lstm_out, self.hidden_self = self.lstm(out,self.hidden_cell)
        predictions = self.fc(lstm_out)
        output = predictions
        return output

"""
逐层训练AE
    
"""
def train_layers(layers_list=None, layer=None, epoch=None, validate=True, weight_decay_list=None,Train_Data=None):
    """
    逐层贪婪预训练 --当训练第i层时, 将i-1层冻结
    :param layers_list:
    :param layer:
    :param epoch:
    :return:
    """
    if torch.cuda.is_available():
        for model in layers_list:
            model.cuda()

    ## 生成数据
    #train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.Adam(layers_list[layer].parameters(), lr=0.005,weight_decay=weight_decay_list[layer])
    criterion_1 = nn.MSELoss()

    # train
    for epoch_index in range(epoch):
        sum_loss = 0.

        # 冻结当前层之前的所有层的参数  --第0层没有前置层
        if layer != 0:
            for index in range(layer):
                layers_list[index].lock_grad()
                layers_list[index].is_training_layer = False  # 除了冻结参数,也要设置冻结层的输出返回方式

        for batch_index, i in enumerate( range( 0, Train_Data.size(0), batch_size)):
            # 生成输入数据
            train_data,_ = get_batch(Train_Data,i,batch_size)
            if torch.cuda.is_available():
                train_data = train_data.cuda()  # 注意Tensor放到GPU上的操作方式,和model不同
            out = train_data
            #out = train_data.view(-1,train_data.size(2))

            # 对前(layer-1)冻结了的层进行前向计算
            if layer != 0:
                for l in range(layer):
                    out = layers_list[l](out)

            # 训练第layer层
            pred, encoded = layers_list[layer](out)

            optimizer.zero_grad()
            MSE_loss = criterion_1(pred, out)
            SR_loss = layers_list[layer].sparse_loss(encoded)
            loss = MSE_loss + SR_loss
            sum_loss += loss
            loss.backward()
            optimizer.step()

            ## 结果打印
            if  (batch_index + 1) % math.ceil(len(Train_Data)/10/2) == 0 and ((epoch_index+1)==epoch/epoch or (epoch_index+1)==epoch):  #(batch_index+1) == 1 or
                print("Train Layer: {}, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    layer, (epoch_index + 1), epoch, (batch_index + 1), math.ceil(len(Train_Data)/10), loss
                ))

        if validate:
            pass
    print('Finish Training a layer')
    # layers_list[layer].decoder = nn.Sequential()
    # print(layers_list)

###############  全局训练
def train_whole(model=None, epoch=50,Train_Data=None):
    print(">>start training whole model")
    if torch.cuda.is_available():
        model.cuda()

    for param in model.parameters():
        param.require_grad = True

    ## 数据

    optimizer = optim.Adam(model.parameters(),lr=0.005,weight_decay=0.005)
    criterion = torch.nn.MSELoss()

    ## 正式训练
    for epoch_index in range(epoch):
        sum_loss = 0.
        for batch_index, i in enumerate(range(0, Train_Data.size(0), batch_size)):
            # 生成输入数据
            train_data, targets = get_batch(Train_Data, i, batch_size)
            if torch.cuda.is_available():
                train_data = train_data.cuda()  # 注意Tensor放到GPU上的操作方式,和model不同
                targets = targets.cuda()
                hidden_cell = (torch.zeros(6, 2*train_data.size(0), model.hidden_layer_size),
                                 torch.zeros(6, 2*train_data.size(0), model.hidden_layer_size))
            x = train_data



            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(6, train_data.size(1), model.hidden_layer_size).to('cuda:0'),
                                  torch.zeros(6, train_data.size(1), model.hidden_layer_size).to('cuda:0'))
            model.cuda()
            out = model(x)
            loss = criterion(out,targets)

            save_output = out[-1:,-1,:]
            if Train_Data.size(0)-i<=batch_size:
                data_tt = targets[-1:,-1:,:]
                model.hidden_cell = (torch.zeros(6, data_tt.size(1), model.hidden_layer_size).to('cuda:0'),
                                     torch.zeros(6, data_tt.size(1), model.hidden_layer_size).to('cuda:0'))
                model.cuda()
                out2 = model(data_tt)
                save_output2 = out2[-1,-1:,:]

            sum_loss += loss
            optimizer.step()

            if (batch_index + 1) % math.ceil(len(Train_Data)/10/2) == 0 and ((epoch_index+1)==epoch/epoch or (epoch_index+1)==epoch):
                print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    (epoch_index + 1), epoch, (batch_index + 1), math.ceil(len(Train_Data)/10), loss
                ))

            if (epoch_index+1)%10 ==0 and Train_Data.size(0)-i<=batch_size:
                if (epoch_index+1) == 10:
                    C_save_t2 = save_output2
                else:
                    C_save_t2 = torch.cat((C_save_t2,save_output2),0)

    return save_output, C_save_t2

AE_train_epochs = 10
num_layers = 2
encoder_1 = AutoEncoderLayer(J,50,SelfTraining=True,sparsity_ratio=0.20)
decoder_2 = AutoEncoderLayer(50,32,SelfTraining=True,sparsity_ratio=0.15)
layers_list = [encoder_1,decoder_2]
L2_weight = [0.006,0.004]
SP_list = [0.20,0.15]
input_window = 20
output_window = 1
batch_size = 10
## online/offline?
data_train = data_tensor
t_start = 914
t_end = 924
for tt in  range(t_start,t_end):
    for ii in range(I):
        print("================================ Time  {}/{} , Loc : {}/{} ===============================".format(tt,t_end,(ii+1),I))
        data_SAEinput = data_train[ii,614:tt,:]
        data_seq = create_inout_sequences(data_SAEinput, input_window, output_window)
        layers_list[0].is_training_layer = True
        layers_list[0].acquire_grad()

        layers_list[1].is_training_layer = True
        layers_list[1].acquire_grad()

        for level in range(num_layers):
            train_layers(layers_list=layers_list, layer=level,epoch=AE_train_epochs,validate=True,weight_decay_list=L2_weight,Train_Data=data_seq)

        SAEL_model = SAEL(layers_list=layers_list, input_size=32,hidden_layer_size=100,output_size=64)
        _,C_pred = train_whole(SAEL_model,20, data_seq)
        if ii%1000==0 or ii==0:
            X_pred = torch.unsqueeze(C_pred,0)
        else:
            X_pred = torch.cat((X_pred,torch.unsqueeze(C_pred,0)),0)

        if (ii+1)%1000==0 or ii==I-1 :
            #save_index = math.ceil((ii+1)/1000)
            filename3 = 'IDW_SAEL_sigma%d' % input_index + '_tt%d' % tt + '_ii%d' % (ii+1) + '.mat'
            save_filename = file_path + 'NEToutput\\' + filename3
            X_pred_save = X_pred.cpu()
            X_pred_save = X_pred_save.detach().numpy()
            io.savemat(save_filename, {'X_pred': X_pred_save})


    # if tt == 914:
    #     Xt_pred = torch.unsqueeze(X_pred, 0)
    # else:
    #     Xt_pred = torch.cat((Xt_pred, torch.unsqueeze(X_pred, 0)), 0)

# Xt_pred = Xt_pred.detach().numpy()
# filename3 = 'IDW_SAEL_sigma%d' % input_index + '.mat'
# save_filename = file_path + 'NEToutput\\' + filename3
# io.savemat(save_filename, {'X_pred': X_pred})