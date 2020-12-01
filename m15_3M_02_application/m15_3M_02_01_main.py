# MODEL m15_3M_02 application - MAIN

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
from torch.autograd import Variable

# IMPORT FUNCTIONS/CLASSES
import m15_3M_02_02_data_loading
import m15_3M_02_03_nn_class
import m15_3M_02_04_plotting_saving_data


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 2
hidden_size = 4


# IMPORTING DATA
data = pd.read_csv("df_final", sep = '\t')


# NETWORK PARAMETERS
count_data = round(len(data))
in_swl = 1
in_p = 1
in_et = 1
in_p_forecast = 1 #houer
in_et_forecast = 1 #houer
forecast_horizon = 24 * 5 #houers
input_size_swl = in_swl + in_p + in_et + in_p_forecast + in_et_forecast
input_size = in_p + in_et + in_p_forecast + in_et_forecast
output_size = forecast_horizon
batch_size = round(count_data - forecast_horizon) 
print(batch_size)


# DATA LOADING
# data preperation
inp_swl, inp_p, inp_et = m15_3M_02_02_data_loading.Data_Preperation(data, count_data)

# data loading
dataset = m15_3M_02_02_data_loading.Load_Data(inp_swl, inp_p, inp_et, in_p, in_et, in_swl, in_p_forecast, in_et_forecast, forecast_horizon, batch_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# NN CLASS
lstm = m15_3M_02_03_nn_class.ANN(input_size_swl, input_size, hidden_size, output_size, num_lstm_layers)
print(lstm)


# LSTM APPLICATION
lstm.load_state_dict(torch.load('trained_lstm.py'))
lstm.eval()

hold_lstm_output = [] 
hold_inp_tensor = []

for inp_tensor_swl, inp_tensor in data_loader:

	inp_tensor_swl = Variable(inp_tensor_swl)
	inp_tensor_swl = inp_tensor_swl.view(batch_size, 1, input_size_swl)
	inp_tensor = Variable(inp_tensor)
	inp_tensor = inp_tensor.view(batch_size, forecast_horizon - 1, input_size)

	lstm_output = lstm(inp_tensor_swl, inp_tensor)
	
	hold_lstm_output.append(lstm_output)
	hold_inp_tensor.append(inp_tensor)



hold_lstm_output = torch.cat(hold_lstm_output, dim = 0)
hold_inp_tensor = torch.cat(hold_inp_tensor, dim = 0)


#PLOTTING
# get max / min data of swl to denormalize input data
data_not_norm = pd.read_csv("df_not_norm.csv", sep = '\t')
swl_max = data_not_norm['Grundwasserstand  [m ü. NN]'].max() + 0.15
swl_min = data_not_norm['Grundwasserstand  [m ü. NN]'].min() - 0.15


# plot target against prediction of the testing
forecast = m15_3M_02_04_plotting_saving_data.Denormalization(hold_lstm_output, swl_max, swl_min)

m15_3M_02_04_plotting_saving_data.Safe_Data(hold_inp_tensor, hold_lstm_output, forecast)
print(forecast)
m15_3M_02_04_plotting_saving_data.Prediction(forecast, 'forecast')
