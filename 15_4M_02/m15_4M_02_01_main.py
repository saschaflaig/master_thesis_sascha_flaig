# MODEL m15_4M_02 - MAIN

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv


# IMPORT FUNCTIONS/CLASSES
import m15_4M_02_02_data_loading
import m15_4M_02_03_nn_class
import m15_4M_02_04_training
import m15_4M_02_05_testing
import m15_4M_02_06_plotting_saving_data
import m15_4M_02_07_statistics


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 2
learning_rate = 0.01
weight_decay = 1e-7
epochs = 1200
hidden_size = 9


# IMPORTING DATA
data = pd.read_csv("df_final.csv", sep = '\t')


# NETWORK PARAMETERS
percentage_train = 64
percentage_vali = 18
percentage_test = 18
count_train= round((len(data)) * (percentage_train / 100)) - 1 
count_vali= round((len(data))* (percentage_vali / 100)) - 1
count_test = round((len(data))* (percentage_test / 100)) - 1
in_swl = 1
in_p = 1
in_t = 1
in_sd = 1
in_rh = 1
in_wv = 1
in_w5 = 1
in_w6 = 1
in_p_forecast = 1 #houer
in_t_forecast = 1 #houer
in_sd_forecast = 1 #houer
in_rh_forecast = 1 #houer
in_wv_forecast = 1 #houer
in_w_forcecast = 1 #houer
forecast_horizon = 24 * 7 #houers
input_size_swl = in_swl + in_p + in_p_forecast + in_t + in_t_forecast + in_sd + in_sd_forecast + in_rh + in_rh_forecast + in_wv + in_wv_forecast + in_w5 + in_w_forcecast + in_w6 + in_w_forcecast
input_size = in_p + in_p_forecast + in_t + in_t_forecast + in_sd + in_sd_forecast + in_rh + in_rh_forecast + in_wv + in_wv_forecast + in_w5 + in_w_forcecast + in_w6 + in_w_forcecast
output_size = forecast_horizon
batch_size_train = round(count_train - max(in_p, in_t, in_swl) - forecast_horizon - 1)
batch_size_vali = round(count_vali - max(in_p, in_t, in_swl) - forecast_horizon  - 1)
batch_size_test = round(count_test - max(in_p, in_t, in_swl) - forecast_horizon - 1)


# DATA LOADING
# data preperation
inp_train_swl, inp_train_p, inp_train_t, inp_train_sd, inp_train_rh, inp_train_wv, inp_train_w_5, inp_train_w_6, tar_train_swl = m15_4M_02_02_data_loading.Data_Preperation(data, 0, count_train, count_train)
inp_vali_swl, inp_vali_p, inp_vali_t, inp_vali_sd, inp_vali_rh, inp_vali_wv, inp_vali_w_5, inp_vali_w_6, tar_vali_swl = m15_4M_02_02_data_loading.Data_Preperation(data, count_train, count_train + count_vali, count_vali)
inp_test_swl, inp_test_p, inp_test_t, inp_test_sd, inp_test_rh, inp_test_wv, inp_test_w_5, inp_test_w_6, tar_test_swl = m15_4M_02_02_data_loading.Data_Preperation(data, count_train + count_vali, count_train + count_vali + count_test, count_test)

# data loading
train_dataset = m15_4M_02_02_data_loading.Load_Data(inp_train_swl, inp_train_p, inp_train_t, inp_train_sd, inp_train_rh, inp_train_wv, inp_train_w_5, inp_train_w_6, tar_train_swl, in_p, in_t, in_sd, in_rh, in_wv, in_swl, in_w5, in_w6, in_p_forecast, in_t_forecast, in_sd_forecast, in_rh_forecast, in_wv_forecast, in_w_forcecast, forecast_horizon, batch_size_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, drop_last=True)

vali_dataset = m15_4M_02_02_data_loading.Load_Data(inp_vali_swl, inp_vali_p, inp_vali_t, inp_vali_sd, inp_vali_rh, inp_vali_wv, inp_vali_w_5, inp_vali_w_6, tar_vali_swl, in_p, in_t, in_sd, in_rh, in_wv, in_swl, in_w5, in_w6, in_p_forecast, in_t_forecast, in_sd_forecast, in_rh_forecast, in_wv_forecast, in_w_forcecast, forecast_horizon, batch_size_vali)
vali_loader = DataLoader(vali_dataset, batch_size=batch_size_vali, shuffle=False, drop_last=True)

test_dataset = m15_4M_02_02_data_loading.Load_Data(inp_test_swl, inp_test_p, inp_test_t, inp_test_sd, inp_test_rh, inp_test_wv, inp_test_w_5, inp_test_w_6, tar_test_swl, in_p, in_t, in_sd, in_rh, in_wv, in_swl, in_w5, in_w6, in_p_forecast, in_t_forecast, in_sd_forecast, in_rh_forecast, in_wv_forecast, in_w_forcecast, forecast_horizon, batch_size_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=True)


# NN CLASS
lstm = m15_4M_02_03_nn_class.ANN(input_size_swl, input_size, hidden_size, output_size, num_lstm_layers)
#lstm.load_state_dict(torch.load('Trained_G01_M01.pt'))
print(lstm)


# TRAINING
# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate, weight_decay = weight_decay)

hold_loss_train = []
hold_loss_vali = []
hold_lstm_output_train = []
hold_inp_tensor_train = []
hold_tar_tensor_train = []

hold_lstm_output_vali, hold_inp_tensor_vali, hold_tar_tensor_vali = m15_4M_02_04_training.Training(lstm, criterion, optimizer, epochs, train_loader, vali_loader, hold_loss_train, hold_loss_vali, hold_lstm_output_train, hold_inp_tensor_train, hold_tar_tensor_train, train_dataset, batch_size_train, batch_size_vali, forecast_horizon, input_size_swl, input_size, hidden_size, num_lstm_layers)

hold_lstm_output_train = torch.cat(hold_lstm_output_train, dim = 0)
hold_inp_tensor_train = torch.cat(hold_inp_tensor_train, dim = 0)
hold_tar_tensor_train = torch.cat(hold_tar_tensor_train, dim = 0)


#params = list(lstm.parameters())


# TESTING
hold_lstm_output_test = [] 
hold_inp_tensor_test = []
hold_tar_tensor_test = []

m15_4M_02_05_testing.Testing(lstm, criterion, optimizer, test_loader, batch_size_test, forecast_horizon, input_size_swl, input_size, hidden_size, hold_lstm_output_test, hold_inp_tensor_test, hold_tar_tensor_test, num_lstm_layers)

hold_lstm_output_test = torch.cat(hold_lstm_output_test, dim = 0)
hold_inp_tensor_test = torch.cat(hold_inp_tensor_test, dim = 0)
hold_tar_tensor_test = torch.cat(hold_tar_tensor_test, dim = 0)


#PLOTTING
# plot loss vs. epoch - training and validation set
m15_4M_02_06_plotting_saving_data.Loss_Vs_Epoch(hold_loss_train, hold_loss_vali)

# get max / min data of swl to denormalize input data
data_not_norm = pd.read_csv("df_not_norm.csv", sep = '\t')
swl_max = data_not_norm['Grundwasserstand  [m ü. NN]'].max() + 0.15
swl_min = data_not_norm['Grundwasserstand  [m ü. NN]'].min() - 0.15

# plot target against prediction of the training
prediction_train_fh, prediction_train_5_24, prediction_train_3_24, prediction_train_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_lstm_output_train, forecast_horizon, swl_max, swl_min)
target_train_fh, target_train_5_24, target_train_3_24, target_train_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_tar_tensor_train, forecast_horizon, swl_max, swl_min)
m15_4M_02_06_plotting_saving_data.Safe_Data(hold_inp_tensor_train, hold_tar_tensor_train, hold_lstm_output_train, prediction_train_fh, target_train_fh, 'train', forecast_horizon)

date_range_train = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train_fh),'h'),dtype='datetime64[h]')
date_range_train = date_range_train.flatten()
date_range_train_5_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24 + len(prediction_train_5_24),'h'),dtype='datetime64[h]')
date_range_train_5_24 = date_range_train_5_24.flatten()
date_range_train_3_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24 + len(prediction_train_3_24),'h'),dtype='datetime64[h]')
date_range_train_3_24 = date_range_train_3_24.flatten()
date_range_train_1_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24 + len(prediction_train_1_24),'h'),dtype='datetime64[h]')
date_range_train_1_24 = date_range_train_1_24.flatten()
print(date_range_train)

m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_train, target_train_fh, prediction_train_fh, 'training_set_performance')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_train_5_24, target_train_5_24, prediction_train_5_24, 'training_set_performance_5_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_train_3_24, target_train_3_24, prediction_train_3_24, 'training_set_performance_3_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_train_1_24, target_train_1_24, prediction_train_1_24, 'training_set_performance_1_24')

# plot target against prediction of the validation
prediction_vali_fh, prediction_vali_5_24, prediction_vali_3_24, prediction_vali_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_lstm_output_vali, forecast_horizon, swl_max, swl_min)
target_vali_fh,target_vali_5_24, target_vali_3_24, target_vali_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_tar_tensor_vali, forecast_horizon, swl_max, swl_min)
m15_4M_02_06_plotting_saving_data.Safe_Data(hold_inp_tensor_vali, hold_tar_tensor_vali, hold_lstm_output_vali, prediction_vali_fh, target_vali_fh, 'vali', forecast_horizon)

date_range_vali = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train_fh) + forecast_horizon + 2,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train_fh) + forecast_horizon + 2 + len(prediction_vali_fh),'h'),dtype='datetime64[h]')
date_range_vali = date_range_vali.flatten()
date_range_vali_5_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24 + len(prediction_train_5_24) + 5*24 + 2,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24 + len(prediction_train_5_24) + 5*24 + 2 + len(prediction_vali_5_24),'h'),dtype='datetime64[h]')
date_range_vali_5_24 = date_range_vali_5_24.flatten()
date_range_vali_3_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24 + len(prediction_train_3_24) + 3*24 + 2,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24 + len(prediction_train_3_24) + 3*24 + 2 + len(prediction_vali_3_24),'h'),dtype='datetime64[h]')
date_range_vali_3_24 = date_range_vali_3_24.flatten()
date_range_vali_1_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24 + len(prediction_train_1_24) + 1*24 + 2,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24 + len(prediction_train_1_24) + 1*24 + 2 + len(prediction_vali_1_24),'h'),dtype='datetime64[h]')
date_range_vali_1_24 = date_range_vali_1_24.flatten()
print(date_range_vali)

m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_vali, target_vali_fh, prediction_vali_fh, 'validation_set_performance')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_vali_5_24, target_vali_5_24, prediction_vali_5_24, 'validation_set_performance_5_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_vali_3_24, target_vali_3_24, prediction_vali_3_24, 'validation_set_performance_3_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_vali_1_24, target_vali_1_24, prediction_vali_1_24, 'validation_set_performance_1_24')

# plot target against prediction of the testing
prediction_test_fh, prediction_test_5_24, prediction_test_3_24, prediction_test_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_lstm_output_test, forecast_horizon, swl_max, swl_min)
target_test_fh, target_test_5_24, target_test_3_24, target_test_1_24 = m15_4M_02_06_plotting_saving_data.Denormalization(hold_tar_tensor_test, forecast_horizon, swl_max, swl_min)

m15_4M_02_06_plotting_saving_data.Safe_Data(hold_inp_tensor_test, hold_tar_tensor_test, hold_lstm_output_test, prediction_test_fh, target_test_fh, 'test', forecast_horizon)

date_range_test = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train_fh) + forecast_horizon + 2 + len(prediction_vali_fh) + 2 + forecast_horizon,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train_fh) + forecast_horizon + 2 + len(prediction_vali_fh) + 2 + forecast_horizon + len(prediction_test_fh),'h'),dtype='datetime64[h]')
date_range_test = date_range_test.flatten()
date_range_test_5_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24 + len(prediction_train_5_24) + 5*24 + 2 + len(prediction_vali_5_24) + 2 + 5*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(5*24 + len(prediction_train_5_24) + 5*24 + 2 + len(prediction_vali_5_24) + 2 + 5*24 + len(prediction_test_5_24),'h'),dtype='datetime64[h]')
date_range_test_5_24 = date_range_test_5_24.flatten()
date_range_test_3_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24 + len(prediction_train_3_24) + 3*24 + 2 + len(prediction_vali_3_24) + 2 + 3*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(3*24 + len(prediction_train_3_24) + 3*24 + 2 + len(prediction_vali_3_24) + 2 + 3*24 + len(prediction_test_3_24),'h'),dtype='datetime64[h]')
date_range_test_3_24 = date_range_test_3_24.flatten()
date_range_test_1_24 = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24 + len(prediction_train_1_24) + 1*24 + 2 + len(prediction_vali_1_24) + 2 + 1*24,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(1*24 + len(prediction_train_1_24) + 1*24 + 2 + len(prediction_vali_1_24) + 2 + 1*24 + len(prediction_test_1_24),'h'),dtype='datetime64[h]')
date_range_test_1_24 = date_range_test_1_24.flatten()
print(date_range_test)

m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_test, target_test_fh, prediction_test_fh, 'testing_set_performance')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_test_5_24, target_test_5_24, prediction_test_5_24, 'testing_set_performance_5_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_test_3_24, target_test_3_24, prediction_test_3_24, 'testing_set_performance_3_24')
m15_4M_02_06_plotting_saving_data.Target_Vs_Prediction(date_range_test_1_24, target_test_1_24, prediction_test_1_24, 'testing_set_performance_1_24')


# STATISTICS
max_diff_train_fh, sse_train_fh, mse_train_fh, mse_train_5_24, mse_train_3_24, mse_train_1_24 = m15_4M_02_07_statistics.Compute_Statistics(prediction_train_fh, target_train_fh, prediction_train_5_24, target_train_5_24, prediction_train_3_24, target_train_3_24, prediction_train_1_24, target_train_1_24, 'train')
max_diff_vali_fh, sse_vali_fh, mse_vali_fh, mse_vali_5_24, mse_vali_3_24, mse_vali_1_24 = m15_4M_02_07_statistics.Compute_Statistics(prediction_vali_fh, target_vali_fh, prediction_vali_5_24, target_vali_5_24, prediction_vali_3_24, target_vali_3_24, prediction_vali_1_24, target_vali_1_24, 'vali')
max_diff_test_fh, see_test_fh, mse_test_fh, mse_test_5_24, mse_test_3_24, mse_test_1_24 = m15_4M_02_07_statistics.Compute_Statistics(prediction_test_fh, target_test_fh, prediction_test_5_24, target_test_5_24, prediction_test_3_24, target_test_3_24, prediction_test_1_24, target_test_1_24, 'test')

with open('statistics/statistics.csv', 'a+', newline = '') as f:
	file = csv.writer(f, delimiter = '\t')
	file.writerow(['max_diff_train_fh', 'sse_train_fh', 'mse_train_fh', 'max_diff_vali_fh', 'sse_vali_fh', 'mse_vali_fh', 'max_diff_test_fh', 'see_test_fh', 'mse_test_fh', 'mse_test_5_24','mse_test_3_24','mse_test_1_24'])
	file.writerow([max_diff_train_fh, sse_train_fh, mse_train_fh, max_diff_vali_fh, sse_vali_fh, mse_vali_fh, max_diff_test_fh, see_test_fh, mse_test_fh, mse_test_5_24, mse_test_3_24, mse_test_1_24])
	
# save the model parameters
torch.save(lstm.state_dict(), 'trained/trained_lstm.py')