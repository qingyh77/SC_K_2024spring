from ConvLSTM import ConvLSTM, ConvLSTM_baseline
import torch
import math
from torch import nn
import numpy as np
import scipy.io as scio
from scipy import io
import time


""" 
参数配置：
    Rcp
    batch_size
    input_window
    output_window
    lr
"""
class ModelConfig(object):
    def __init__(self):
        self.batch_size = 10
        self.lr = 0.0001
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (3,3)
        self.num_layers = 1
        self.height = 2601
        self.width = 64
        self.time_step = 20
        self.batch_first = True
        self.bias = True
        self.device = torch.device('cpu')

class PredictConfig(object):
    def __init__(self):
        self.batch_size = 1
        self.lr = 0.0001
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (3,3)
        self.num_layers = 1
        self.height = 2601
        self.width = 64
        self.time_step = 20
        self.batch_first = True
        self.bias = True


R_num = 8
rho_set = [0.1]
v=0.01
Rcp = 5
# batch_size = 50
input_window = 20
output_window = 1
lr = 0.0001
for rr in range(0,len(rho_set)):
    rho = rho_set[rr]
    """ 
    导入文件：
        file_path:文件路径
        file_method:张量补全/分解方法
        file_name:导入数据文件名
        input_filename：最终地址
        C_data：最终导出数据
    """
    #E:\hqy\Sigma_cpd\采样位置固定
    input_index = 2  #标准差
    sourceNum = R_num
    snr = 20
    #file_path = 'E:\\hqy\\Sigma_net\\采样位置固定\\'
    file_path_1 = 'F:\\黄清扬\\SpectrumPrediction_2024\\MATLABoutput\\Mat_R%d_' %sourceNum
    #file_path_2 = 'v'+str(v) + '_SNR%d' % snr + '_张量补全失败\\Mat_rho' + str(rho) + '_R'+str(sourceNum) + '_SNR20_sinc2C\\'
    file_path_2 = 'v' + str(v) + '_SNR%d' % snr + '\\Mat_rho' + str(rho) + '_R' + str(sourceNum) + '_SNR20_sinc2C\\'
    file_path = file_path_1+file_path_2
    #file_path = 'F:\\黄清扬\\Sigma_cpd\\NET\\'
    file_method = 'IDW'  #ICPD_C/DWCPD_C/IDW/TPS
    file_name_1 = file_method+'_rho'+str(rho)
    file_name_2 = '_R%d.mat' %sourceNum
    file_name = file_name_1+file_name_2
    input_filename = file_path + file_name

    # C_pred_save = 1
    # C_pred_save = C_pred_save.detach().numpy()
    # filename3 = '_convLSTM_sigma%d_offline.mat' % input_index
    # save_filename = file_path +'NEToutput\\' + file_method + filename3
    # io.savemat(save_filename, {'X_pred':C_pred_save})

    # 正式导入
    data = scio.loadmat(input_filename)
    type(data)
    data_name = 'Xhat_idw' #C_icpd/C_dwcpd/Xhat_idw/Xhat_tps
    C_data = np.array(data[data_name])
    if data_name[0] == 'C':
        Time_len, R = C_data.shape
    else:
        I, J, Time_len = C_data.shape


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
    训练函数
    """
    def train(train_data):
        model.train() # Turn on the train mode \o/
        total_loss = 0.
        start_time = time.time()
        save_output = []
        for batch, i in enumerate(range(0, train_data.size(0), batch_size)):  # Now len-1 is not necessary
            # data and target are the same shape with (input_window,batch_len,1)
            data, targets = get_batch(train_data, i , batch_size)
            optimizer.zero_grad()
            # model.hidden_cell = (torch.zeros(1, data.size(0), model.hidden_layer_size),
            #                      torch.zeros(1, data.size(0), model.hidden_layer_size))
            output = model(data,config)
            loss = criterion(output, targets)

            ## 过往数据可以参与训练，但想要观察的还是预测数据的效果，因此只需要记录最后一个预测结果
            # save_output = output[-1,-1,:]
            # if train_data.size(0)-i <= batch_size:
            #     pred_config = PredictConfig()
            #     data_tt = targets[-1:,:]
            #     # model.hidden_cell = (torch.zeros(1, data_tt.size(0), model.hidden_layer_size),
            #     #                          torch.zeros(1, data_tt.size(0), model.hidden_layer_size))
            #     output2 = model(data_tt,pred_config)
            #     save_output2 = output2[-1,-1,:]

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
            if epoch%2 == 0 and batch % log_interval == 0 and batch > 0:
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
        # return save_output,save_output2

    """
    预测函数
    """
    def predict_future(eval_model, data_source, steps, epoch):
        eval_model.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        data, _ = get_batch(data_source, 0, 1)
        with torch.no_grad():
            for i in range(0, steps):
                output = eval_model(data[:,-input_window:],preconfig)
                # (seq-len , batch-size , features-num)
                # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
                data = torch.cat((data, output[:,-1:,:]),1)  # [m,m+1,..., m+n+1]

        return data

        # I used this plot to visualize if the model pics up any long therm structure within the data.
        # pyplot.plot(data, color="red")
        # pyplot.plot(data[:input_window], color="blue")
        # pyplot.grid(True, which='both')
        # pyplot.axhline(y=0, color='k')
        # pyplot.savefig('graph/transformer-future%d.png' % steps)
        # pyplot.show()
        # pyplot.close()


    """
    搭建网络
    """
    in_channels = 1
    out_channels = 5
    kernel_size = (3,3)
    num_layer = 2
    # model = ConvLSTM(in_channels=in_channels,
    #                  out_channels=out_channels,
    #                  kernel_size=kernel_size,
    #                  num_layers=num_layer,
    #                  batch_first=True,
    #                  bias=True,
    #                  return_all_layers=False)
    config = ModelConfig()
    batch_size = config.batch_size
    model = ConvLSTM_baseline(config)
    model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(),lr=lr,eps=1e-8,weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    print(model)

    """
    正式训练
        离线训练：
                训练：测试=9：1
                训练轮数epochs:
                batch_size:
                time_step:
    """
    epochs = 10
    start = 1*math.ceil(1.5*2/rho)
    test_data_size = 100
    C_data = C_data[:,:,28:-1]
    #C_data = C_data[:,:,28:228]
    C_data = torch.FloatTensor(C_data)
    C_data_now = C_data.permute(2,0,1)
    C_data_now = torch.unsqueeze(C_data_now, 1)
    train_data = C_data_now[:-test_data_size,:]
    test_data = C_data_now[-test_data_size:,:]

    train_in_seq = create_inout_sequences(train_data.to(config.device), input_window, output_window)
    test_in_seq = create_inout_sequences(test_data.to(config.device), input_window,output_window)

    for epoch in range(1,epochs+1):
        print("----第{}轮训练开始----".format(epoch))
        train(train_in_seq.to(config.device))

        if epoch % 5 ==0:
            preconfig = PredictConfig()
            C_pred_epoch = predict_future(model,test_in_seq.to(config.device),test_data_size-config.time_step,epoch)
            if epoch == 5:
                C_pred_save = C_pred_epoch
            else:
                C_pred_save = torch.cat((C_pred_save,C_pred_epoch),0)




    C_pred_save = C_pred_save.detach().numpy()


    # for tt in range(ts_start,Time_len):
    #     print("----第{}时刻开始----".format(tt))
    #     window_len = 220
    #     window_left = max(0,tt-window_len)
    #     C_data_now = C_data[:,:,window_left:(tt)]
    #     C_data_now = torch.FloatTensor(C_data_now)
    #
    #     C_data_now = torch.unsqueeze(C_data_now, 1)
    #     input_window_now = min(tt, input_window)
    #     C_seq = create_inout_sequences(C_data_now, input_window_now,output_window)
    #     batch_size_now = min(tt, batch_size)
    #     for epoch in range(1, epochs+1):
    #         # print("----第{}时刻，第{}轮训练开始----".format(tt, epoch))
    #         C_predict_epoch, C_predict_epoch2 = train(C_seq, batch_size_now)
    #         if epoch %5 ==0:
    #             if epoch == 5:
    #                 C_save_t1 = C_predict_epoch
    #                 C_save_t2 = C_predict_epoch2
    #             else:
    #                 C_save_t1 = torch.cat((C_save_t1, C_predict_epoch),0)
    #                 C_save_t2 = torch.cat((C_save_t2, C_predict_epoch2), 0)
    #
    #     if tt == ts_start:
    #         C_save1 = torch.unsqueeze(C_save_t1, 0)
    #         C_save2 = torch.unsqueeze(C_save_t2, 0)
    #     else:
    #         C_save1 = torch.cat((C_save1, torch.unsqueeze(C_save_t1, 0)),0 )
    #         C_save2 = torch.cat((C_save2, torc

    filename3 = '_convLSTM_sigma%d' % input_index
    filename4 = '_offline_epochs%d.mat' % epochs
    save_filename = file_path +'NEToutput\\' + file_method + filename3 + filename4
    io.savemat(save_filename, {'X_pred':C_pred_save})
    train_in_seq = []
    test_in_seq = []