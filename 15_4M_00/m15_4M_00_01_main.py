# MODEL m15_4M_00 - MAIN

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv


# IMPORT FUNCTIONS/CLASSES
import m15_4M_00_02_data_loading
import m15_4M_00_03_nn_class
import m15_4M_00_04_training
import m15_4M_00_05_testing
import m15_4M_00_06_plotting_saving_data
import m15_4M_00_07_statistics


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 2
learning_rate = 0.001
weight_decay = 1e-7
epochs = 1900
hidden_size = 400


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
in_et = 1
in_w5 = 1
in_w6 = 1
in_p_forecast = 24 * 7 #houers
in_et_forecast = 24 * 7 #houers
in_w_forcecast = 24 * 7 #houers
forecast_horizon = 24 * 7 #houers
input_size = in_swl + in_p + in_et + in_p_forecast + in_et_forecast + in_w5 + in_w_forcecast + in_w6 + in_w_forcecast
output_size = forecast_horizon
batch_size_train = count_train - max(in_p, in_et, in_swl) - forecast_horizon - 1
batch_size_vali = count_vali - max(in_p, in_et, in_swl) - forecast_horizon  - 1
batch_size_test = count_test - max(in_p, in_et, in_swl) - forecast_horizon - 1


# DATA LOADING
# data preperation
inp_train_swl, inp_train_p, inp_train_et, inp_train_w_5, inp_train_w_6, tar_train_swl = m15_4M_00_02_data_loading.Data_Preperation(data, 0, count_train, count_train)
inp_vali_swl, inp_vali_p, inp_vali_et, inp_vali_w_5, inp_vali_w_6, tar_vali_swl = m15_4M_00_02_data_loading.Data_Preperation(data, count_train, count_train + count_vali, count_vali)
inp_test_swl, inp_test_p, inp_test_et, inp_test_w_5, inp_test_w_6, tar_test_swl = m15_4M_00_02_data_loading.Data_Preperation(data, count_train + count_vali, count_train + count_vali + count_test, count_test)

# data loading
train_dataset = m15_4M_00_02_data_loading.Load_Data(inp_train_swl, inp_train_p, inp_train_et, inp_train_w_5, inp_train_w_6, tar_train_swl, in_p, in_et, in_swl,in_w5, in_w6, in_p_forecast, in_et_forecast, in_w_forcecast, forecast_horizon)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, drop_last=True)

vali_dataset = m15_4M_00_02_data_loading.Load_Data(inp_vali_swl, inp_vali_p, inp_vali_et, inp_vali_w_5, inp_vali_w_6, tar_vali_swl, in_p, in_et, in_swl, in_w5, in_w6, in_p_forecast, in_et_forecast, in_w_forcecast, forecast_horizon)
vali_loader = DataLoader(vali_dataset, batch_size=batch_size_vali, shuffle=False, drop_last=True)

test_dataset = m15_4M_00_02_data_loading.Load_Data(inp_test_swl, inp_test_p, inp_test_et,inp_test_w_5, inp_test_w_6, tar_test_swl, in_p, in_et, in_swl, in_w5, in_w6, in_p_forecast, in_et_forecast, in_w_forcecast, forecast_horizon)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=True)


# NN CLASS
lstm = m15_4M_00_03_nn_class.ANN(input_size, hidden_size, output_size, num_lstm_layers)
#lstm.load_state_dict(torch.load('Trained_G01_M01.pt'))
print(lstm)


# TRAINING
# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate, weight_decay = weight_decay)

hold_loss_train = []
hold_loss_vali = []

lstm_output_train, inp_tensor_train, tar_tensor_train, lstm_output_vali, inp_tensor_vali, tar_tensor_vali = m15_4M_00_04_training.Training(lstm, criterion, optimizer, epochs, train_loader, vali_loader, hold_loss_train, hold_loss_vali, train_dataset, batch_size_train)

#params = list(lstm.parameters())


# TESTING 
lstm_output_test, inp_tensor_test, tar_tensor_test = m15_4M_00_05_testing.Testing(lstm, criterion, optimizer, test_loader)


#PLOTTING
# plot loss vs. epoch - training and validation set
m15_4M_00_06_plotting_saving_data.Loss_Vs_Epoch(hold_loss_train, hold_loss_vali)

# get max / min data of swl to denormalize input data
data_not_norm = pd.read_csv("df_not_norm.csv", sep = '\t')
swl_max = data_not_norm['Grundwasserstand  [m ü. NN]'].max() + 0.15
swl_min = data_not_norm['Grundwasserstand  [m ü. NN]'].min() - 0.15

# plot target against prediction of the training
prediction_train = m15_4M_00_06_plotting_saving_data.Denormalization(lstm_output_train, forecast_horizon, swl_max, swl_min)
target_train = m15_4M_00_06_plotting_saving_data.Denormalization(tar_tensor_train, forecast_horizon, swl_max, swl_min)
#m15_4M_00_06_plotting_saving_data.Safe_Data(inp_tensor_train, tar_tensor_train, lstm_output_train, prediction_train, target_train, 'train', forecast_horizon)

date_range_train = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train),'h'),dtype='datetime64[h]')
date_range_train = date_range_train.flatten()
print(date_range_train)
m15_4M_00_06_plotting_saving_data.Target_Vs_Prediction(date_range_train, target_train, prediction_train, 'training_set_performance')

# plot target against prediction of the validation
prediction_vali = m15_4M_00_06_plotting_saving_data.Denormalization(lstm_output_vali, forecast_horizon, swl_max, swl_min)
target_vali = m15_4M_00_06_plotting_saving_data.Denormalization(tar_tensor_vali, forecast_horizon, swl_max, swl_min)
#m15_4M_00_06_plotting_saving_data.Safe_Data(inp_tensor_vali, tar_tensor_vali, lstm_output_vali, prediction_vali, target_vali, 'vali', forecast_horizon)

date_range_vali = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train) + forecast_horizon + 2,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train) + forecast_horizon + 2 + len(prediction_vali),'h'),dtype='datetime64[h]')
date_range_vali = date_range_vali.flatten()
print(date_range_vali)
m15_4M_00_06_plotting_saving_data.Target_Vs_Prediction(date_range_vali, target_vali, prediction_vali, 'validation_set_performance')

# plot target against prediction of the testing
prediction_test = m15_4M_00_06_plotting_saving_data.Denormalization(lstm_output_test, forecast_horizon, swl_max, swl_min)
target_test = m15_4M_00_06_plotting_saving_data.Denormalization(tar_tensor_test, forecast_horizon, swl_max, swl_min)
#m15_4M_00_06_plotting_saving_data.Safe_Data(inp_tensor_test, tar_tensor_test, lstm_output_test, prediction_test, target_test, 'test', forecast_horizon)

date_range_test = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train) + forecast_horizon + 2 + len(prediction_vali) + 2 + forecast_horizon,'h'), np.datetime64(data.iloc[0,0]) + np.timedelta64(forecast_horizon + len(prediction_train) + forecast_horizon + 2 + len(prediction_vali) + 2 + forecast_horizon + len(prediction_test),'h'),dtype='datetime64[h]')
date_range_test = date_range_test.flatten()
print(date_range_test)
m15_4M_00_06_plotting_saving_data.Target_Vs_Prediction(date_range_test, target_test, prediction_test, 'testing_set_performance')


# STATISTICS
max_diff_train, sse_train, mse_train = m15_4M_00_07_statistics.Compute_Statistics(prediction_train, target_train, 'train')
max_diff_vali, sse_vali, mse_vali = m15_4M_00_07_statistics.Compute_Statistics(prediction_vali, target_vali, 'vali')
max_diff_test, see_test, mse_test = m15_4M_00_07_statistics.Compute_Statistics(prediction_test, target_test, 'test')

with open('statistics/statistics.csv', 'a+', newline = '') as f:
	file = csv.writer(f, delimiter = '\t')
	file.writerow(['max_diff_train', 'sse_train', 'mse_train', 'max_diff_vali', 'sse_vali', 'mse_vali', 'max_diff_test', 'see_test', 'mse_test'])
	file.writerow([max_diff_train, sse_train, mse_train, max_diff_vali, sse_vali, mse_vali, max_diff_test, see_test, mse_test])


# save the model parameters
torch.save(lstm.state_dict(), 'trained/trained_lstm.py')