# MODEL m15_3M_02 - DATA LOADING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# data_preperation: training, validation, testing
def Data_Preperation(data,count_start,count_end,count_data):
	# get data 
	inp_swl = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	inp_p = data['Niederschlag  [mm/h]'].to_numpy()[count_start:count_end]
	inp_et = data['Verdunstung berechnet [mm/h]'].to_numpy()[count_start:count_end]
	tar_swl = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	
	# convert dataframe to pytorch tensor
	inp_swl = torch.FloatTensor(inp_swl)
	inp_p = torch.FloatTensor(inp_p)
	inp_et = torch.FloatTensor(inp_et)
	tar_swl = torch.FloatTensor(tar_swl)
	
	# change the shape of the pytorch tensor to fit to the network
	inp_swl = inp_swl.view(count_data,1)
	inp_p = inp_p.view(count_data,1)
	inp_et = inp_et.view(count_data,1)
	tar_swl = tar_swl.view(count_data,1)
	
	return inp_swl, inp_p, inp_et, tar_swl

# data loading
class Load_Data(Dataset):
	def __init__(self, inp_swl, inp_p, inp_et, tar_swl, in_p, in_et, in_swl, in_p_forecast, in_et_forecast, in_w_forcecast, forecast_horizon, batch_size):
		assert len(inp_swl) == len(tar_swl)
		assert len(inp_p) == len(tar_swl)
		assert len(inp_et) == len(tar_swl)
		self.inp_swl = inp_swl
		self.inp_p = inp_p
		self.inp_et = inp_et
		self.tar_swl = tar_swl
		self.in_swl = in_swl
		self.in_p = in_p
		self.in_et = in_et
		self.in_p_forecast = in_p_forecast
		self.in_et_forecast = in_et_forecast
		self.in_w_forcecast = in_w_forcecast
		self.forecast_horizon = forecast_horizon
		self.batch_size = batch_size

	def __len__(self):
		samples = len(self.tar_swl) - self.in_p - self.forecast_horizon + 1
		return samples
	
	# determine the input	
	def __getitem__(self, idx):
		
		inp_swl = self.inp_swl[idx : idx + self.in_swl]
		inp_p_swl = self.inp_p[idx : idx + self.in_p + self.in_p_forecast]
		inp_et_swl = self.inp_et[idx : idx + self.in_et + self.in_et_forecast]
		
		inputs_swl = np.concatenate([inp_swl, inp_p_swl, inp_et_swl])
		inputs_swl = inputs_swl.reshape(1,len(inputs_swl))
		
		
		inputs_final = []
		for i in range(1, self.forecast_horizon):
			inp_p = self.inp_p[idx + i : idx + self.in_p + self.in_p_forecast + i]
			inp_et = self.inp_et[idx + i : idx + self.in_et + self.in_et_forecast + i]
			inputs = np.concatenate([inp_p, inp_et])
			inputs_final.append(inputs)

		inputs = np.reshape(inputs_final, (self.forecast_horizon - 1, len(inputs)))
		
		tar_swl = self.tar_swl[idx + 1 : idx + 1 + self.forecast_horizon]
		tar_swl = np.array(tar_swl)
		tar_swl = tar_swl.reshape(1,len(tar_swl))

		return inputs_swl, inputs, tar_swl
