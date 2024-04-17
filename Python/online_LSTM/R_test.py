R_num = 8
rho_set = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]  # 0.1:0.02:0.2
for rr in range(0, len(rho_set)):
    rho = rho_set[rr]
    for ii in range(1, R_num + 1):
        sigma_index = 3

        input_index = ii  # 1:1:6
        v = 0.1

        experiment = 'Mat_R' + str(R_num) + '_v' + str(v) + '_SNR30'
        # file_path = 'E:\\hqy\\Sigma_net\\采样位置固定\\'
        # file_path = 'F:\\hqy\\Mat_R8_SNR20_sinc2C\\'
        # file_path = 'E:\\hqy\\'+experiment + '\\Mat_rho'+str(rho)+'_R6_SNR30_sinc2C\\'
        file_path = 'E:\\黄清扬\\' + experiment + '\\Mat_rho' + str(rho) + '_R'+str(R_num)+'_SNR30_sinc2C\\'
        # file_path = 'F:\\黄清扬\\Sigma_cpd\\NET\\'
        file_name = 'NMFCPD_S_C_rho' + str(rho) + '_' + 'R%d' % R_num + '_%d.mat' % input_index
        input_filename = file_path + file_name
        print(input_filename)