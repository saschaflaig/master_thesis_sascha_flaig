# MODEL m15_4M_00  - NN CLASS

import torch
import torch.nn as nn

# artificial neural network 
class ANN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_lstm_layers, bias=True):
		super(ANN, self).__init__()
		self.num_lstm_layers = num_lstm_layers
		self.hidden_size = hidden_size
		
		# lstm layer
		self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first = True)
		
		# readout layer
		self.fc1 = nn.Linear(hidden_size, output_size)
		nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, x):
		# set initial hidden and cell states
		h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size)
		
		# forward propagate
		out, hidden = self.lstm(x, (h0, c0))
		out = self.fc1(out)
		out = self.Sigmoid(out)
		
		return out