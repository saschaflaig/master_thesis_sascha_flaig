# MODEL m15_3M_03 - DATA LOADING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# data_preperation: training, validation, testing
def Data_Preperation(data,count_start,count_end,count_data):
	# get data 
	inp_swl = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	inp_p = data['Niederschlag  [mm/h]'].to_numpy()[count_start:count_end]
	inp_t = data['Lufttemperatur  [°C]'].to_numpy()[count_start:count_end]
	inp_sd = data['Sonnenscheindauer  [min/h]'].to_numpy()[count_start:count_end]
	inp_rh = data['relative Luftfeuchte  [%]'].to_numpy()[count_start:count_end]
	inp_wv = data['Windgeschwindigkeit  [m/s]'].to_numpy()[count_start:count_end]
	tar_swl = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	
	# convert dataframe to pytorch tensor
	inp_swl = torch.FloatTensor(inp_swl)
	inp_p = torch.FloatTensor(inp_p)
	inp_t = torch.FloatTensor(inp_t)
	inp_sd = torch.FloatTensor(inp_sd)
	inp_rh = torch.FloatTensor(inp_rh)
	inp_wv = torch.FloatTensor(inp_wv)
	tar_swl = torch.FloatTensor(tar_swl)
	
	# change the shape of the pytorch tensor to fit to the network
	inp_swl = inp_swl.view(count_data,1)
	inp_p = inp_p.view(count_data,1)
	inp_t = inp_t.view(count_data,1)
	inp_sd = inp_sd.view(count_data,1)
	inp_rh = inp_rh.view(count_data,1)
	inp_wv = inp_wv.view(count_data,1)
	tar_swl = tar_swl.view(count_data,1)
	
	return inp_swl, inp_p, inp_t, inp_sd, inp_rh, inp_wv, tar_swl

# data loading
class Load_Data(Dataset):
	def __init__(self, inp_swl, inp_p, inp_t, inp_sd, inp_rh, inp_wv, tar_swl, in_p, in_t, in_sd, in_rh, in_wv, in_swl, in_p_forecast, in_t_forecast, in_sd_forecast, in_rh_forecast, in_wv_forecast, in_w_forcecast, forecast_horizon, batch_size):
		assert len(inp_swl) == len(tar_swl)
		assert len(inp_p) == len(tar_swl)
		assert len(inp_t) == len(tar_swl)
		assert len(inp_sd) == len(tar_swl)
		assert len(inp_rh) == len(tar_swl)
		assert len(inp_wv) == len(tar_swl)
		self.inp_swl = inp_swl
		self.inp_p = inp_p
		self.inp_t = inp_t
		self.inp_sd = inp_sd
		self.inp_rh = inp_rh
		self.inp_wv = inp_wv
		self.tar_swl = tar_swl
		self.in_swl = in_swl
		self.in_p = in_p
		self.in_t = in_t
		self.in_sd = in_sd
		self.in_rh = in_rh
		self.in_wv = in_wv
		self.in_p_forecast = in_p_forecast
		self.in_t_forecast = in_t_forecast
		self.in_sd_forecast = in_sd_forecast
		self.in_rh_forecast = in_rh_forecast
		self.in_wv_forecast = in_wv_forecast
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
		inp_t_swl = self.inp_t[idx : idx + self.in_t + self.in_t_forecast]
		inp_sd_swl = self.inp_sd[idx : idx + self.in_sd + self.in_sd_forecast]
		inp_rh_swl = self.inp_rh[idx : idx + self.in_rh + self.in_rh_forecast]
		inp_wv_swl = self.inp_wv[idx : idx + self.in_wv + self.in_wv_forecast]
		
		inputs_swl = np.concatenate([inp_swl, inp_p_swl, inp_t_swl, inp_sd_swl, inp_rh_swl, inp_wv_swl])
		inputs_swl = inputs_swl.reshape(1,len(inputs_swl))
		
		
		inputs_final = []
		for i in range(1, self.forecast_horizon):
			inp_p = self.inp_p[idx + i : idx + self.in_p + self.in_p_forecast + i]
			inp_t = self.inp_t[idx + i : idx + self.in_t + self.in_t_forecast + i]
			inp_sd = self.inp_sd[idx + i : idx + self.in_sd + self.in_sd_forecast + i]
			inp_rh = self.inp_rh[idx + i : idx + self.in_rh + self.in_rh_forecast + i]
			inp_wv = self.inp_wv[idx + i : idx + self.in_wv + self.in_wv_forecast + i]
			
			inputs = np.concatenate([inp_p, inp_t, inp_sd, inp_rh, inp_wv])
			inputs_final.append(inputs)

		inputs = np.reshape(inputs_final, (self.forecast_horizon - 1, len(inputs)))
		
		tar_swl = self.tar_swl[idx + 1 : idx + 1 + self.forecast_horizon]
		tar_swl = np.array(tar_swl)
		tar_swl = tar_swl.reshape(1,len(tar_swl))

		return inputs_swl, inputs, tar_swl
