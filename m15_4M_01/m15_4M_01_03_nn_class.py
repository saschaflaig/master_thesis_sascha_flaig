# MODEL m15_4M_01 - NN CLASS

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# artificial neural network 
class ANN(nn.Module):
	def __init__(self, input_size_swl, input_size, hidden_size, output_size, num_lstm_layers, bias = True):
		super(ANN, self).__init__()
		self.num_lstm_layers = num_lstm_layers
		self.input_size_swl = input_size_swl
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		
		# lstm layer with the swl in the inout --> input_size = 15
		self.lstm_swl = nn.LSTM(input_size_swl, hidden_size, num_lstm_layers, batch_first = True)
		
		# lstm layer without the swl in the inout --> input_size = 14
		self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first = True)
		
		# readout layer
		self.fc1 = nn.Linear(hidden_size * output_size, output_size)
		nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, input_swl, input):
		# set initial hidden and cell state
		h_0 = torch.zeros(self.num_lstm_layers, input.size(0), self.hidden_size)
		c_0 = torch.zeros(self.num_lstm_layers, input.size(0), self.hidden_size)

		# forward propagate
		out_swl, (hn_swl, cn_swl) = self.lstm_swl(input_swl, (h_0, c_0))
		out_lstm, (hn, cn) = self.lstm(input, (hn_swl, cn_swl))
		out = torch.cat((out_swl, out_lstm), 1)
		out = out.reshape(input.size(0), 1, self.hidden_size * self.output_size)
		out = self.fc1(out)
		out = self.Sigmoid(out)
		
		return out