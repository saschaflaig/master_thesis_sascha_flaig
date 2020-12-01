#Model m15_3M_00

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 1
learning_rate = 0.0055
weight_decay = 1e-6
epochs = 1900


# IMPORTING DATA
data = pd.read_csv("df_final.csv", sep = '\t')


# NETWORK PARAMETERS
percentage_train = 64
percentage_validation = 18
percentage_test = 18
count_train= round((len(data)) * (percentage_train / 100)) - 1 
count_validation= round((len(data))*(percentage_validation / 100)) - 1 
count_test = round((len(data))* (percentage_test / 100)) - 1 
days_elapsed_inP = 1
days_elapsed_inET = 1
days_elapsed_inSWL = 1
days_elapsed_trSWL = 7
Prediction_Horizon = 7
input_size = days_elapsed_inP + days_elapsed_inET + days_elapsed_inSWL + Prediction_Horizon + Prediction_Horizon
output_size = days_elapsed_trSWL
hidden_size = 40
batch_size_train = count_train - max(days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL) - Prediction_Horizon - 1
batch_size_validation = count_validation - max(days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL) - Prediction_Horizon  - 1
batch_size_test = count_test - max(days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL) - Prediction_Horizon - 1
SWLmax, SWLmin = 644.116443, 643.326389


# DATA PREPARATION: TRAINING, VALIDATION, TEST
def extracting_data(count_start,count_end,count_data):
	# get data 
	I_P = data['Niederschlag  [mm/h]'].to_numpy()[count_start:count_end]
	I_ET = data['Verdunstung berechnet [mm/h]'].to_numpy()[count_start:count_end]
	I_SWL = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	T_SWL = data['Grundwasserstand  [m ü. NN]'].to_numpy()[count_start:count_end]
	# convert pandas dataframe to PyTorch tensor
	I_P = torch.FloatTensor(I_P)
	I_ET = torch.FloatTensor(I_ET)
	I_SWL = torch.FloatTensor(I_SWL)
	T_SWL = torch.FloatTensor(T_SWL)
	# change the shape of the PyTorch tensor to fit to the network
	I_P = I_P.view(count_data,1)
	I_ET = I_ET.view(count_data,1)
	I_SWL = I_SWL.view(count_data,1)
	T_SWL = T_SWL.view(count_data,1)
	return (I_P, I_ET, I_SWL, T_SWL)

I_train_P, I_train_ET, I_train_SWL, T_train_SWL = extracting_data(0,count_train,count_train)
I_validation_P, I_validation_ET, I_validation_SWL, T_validation_SWL = extracting_data(count_train,count_train+count_validation,count_validation)
I_test_P, I_test_ET, I_test_SWL, T_test_SWL = extracting_data(count_train+count_validation,count_train+count_validation+count_test,count_test)


# DATA LOADING
class LoadData(Dataset):
	def __init__(self, input1, input2, input3, target, days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL, days_elapsed_trSWL, Prediction_Horizon):
		assert len(input1) == len(target)
		assert len(input2) == len(target)
		assert len(input3) == len(target)
		self.input1 = input1
		self.input2 = input2
		self.input3 = input3
		self.target = target
		self.days_elapsed_inP = days_elapsed_inP
		self.days_elapsed_inET = days_elapsed_inET
		self.days_elapsed_inSWL = days_elapsed_inSWL
		self.days_elapsed_trSWL = days_elapsed_trSWL
		self.Prediction_Horizon = Prediction_Horizon

	def __len__(self):
		samples = len(self.target) - self.days_elapsed_inP - Prediction_Horizon + 1
		return samples

	def __getitem__(self, idx):
		input1 = self.input1[idx:idx+self.days_elapsed_inP+self.Prediction_Horizon]
		input2 = self.input2[idx:idx+self.days_elapsed_inET+self.Prediction_Horizon]
		input3 = self.input3[idx:idx+self.days_elapsed_inSWL]
		inputs = np.concatenate([input1, input2, input3])
		inputs = np.array(inputs)
		inputs = inputs.reshape((1,len(inputs)))
		
		target = self.target[idx+self.days_elapsed_inP:idx+self.days_elapsed_inP+self.Prediction_Horizon]
		target = np.array(target)
		target = target.reshape((1,len(target)))
		
		return (inputs, target)

train_dataset = LoadData(I_train_P, I_train_ET, I_train_SWL, T_train_SWL, days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL, days_elapsed_trSWL, Prediction_Horizon)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, drop_last=True)
validation_dataset = LoadData(I_validation_P, I_validation_ET, I_validation_SWL, T_validation_SWL, days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL, days_elapsed_trSWL, Prediction_Horizon)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size_validation, shuffle=False, drop_last=True)
test_dataset = LoadData(I_test_P, I_test_ET, I_test_SWL, T_test_SWL, days_elapsed_inP, days_elapsed_inET, days_elapsed_inSWL, days_elapsed_trSWL, Prediction_Horizon)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=True)


# THE NEURAL NETWORK
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_lstm_layers, bias=True):
		super(Net, self).__init__()
		# LSTM layer
		self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first = True)
		# Readout layer - Fully connected layer
		self.fc1 = nn.Linear(hidden_size, output_size)
		nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
		self.Sigmoid = nn.Sigmoid()

	def forward(self, x):
		#Set initial hidden and cell states
		h0 = torch.zeros(num_lstm_layers, x.size(0), hidden_size)
		c0 = torch.zeros(num_lstm_layers, x.size(0), hidden_size)
		
		#Forward propagate LSTM
		out, hidden = self.lstm(x, (h0, c0))
		out = self.fc1(out)
		out = self.Sigmoid(out)
		return out


# THE NETWORK TRAINING
LSTM = Net(input_size, hidden_size, output_size, num_lstm_layers)
#LSTM.load_state_dict(torch.load('Trained_G01_M01.pt'))
print(LSTM)

# Choose the Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(LSTM.parameters(), lr=learning_rate, weight_decay=weight_decay)

hold_loss_train=[]
hold_loss_validation=[]

# the training loop
for epoch in range(epochs):
	for batch_idx, (inputtensor, targettensor) in enumerate(train_loader):
		LSTM.train()
		
		inputtensor = Variable(inputtensor)
		targettensor = Variable(targettensor)

		optimizer.zero_grad()
		
		LSTM_output_train= LSTM(inputtensor)
		
		loss = criterion(LSTM_output_train, targettensor)
		
		loss.backward()

		optimizer.step()
		
		# the validation loop	
		with torch.no_grad():
			for dataval, targetval in validation_loader:
				LSTM.eval()
				dataval = Variable(dataval)
				targetval = Variable(targetval)
	
				LSTM_output_validation = LSTM(dataval)
				LSTM_validation_loss = criterion(LSTM_output_validation, targetval)

		# print out the loss versus epoch - training phase
		print('Training: Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, epochs, batch_idx+1, len(train_dataset)//batch_size_train, loss.item()))
		
		hold_loss_train.append(loss)
		hold_loss_validation.append(LSTM_validation_loss)

params = list(LSTM.parameters())


# PLOTTING + TESTING
# plotting style
font = {'family' : 'Times New Roman',
		'weight' : 'bold',
		'size'   : 18}

plt.rc('font', **font)

# plot the loss vs. Epoch - training and validation sets
f, (p1, p2) = plt.subplots(2, 1, sharey=False)
plt.subplots_adjust(hspace=0.3)
f.set_figheight(10)
f.set_figwidth(10)
p1.plot(np.array(hold_loss_train))
p1.set_title('Loss vs. Epoch - Training Set')
p1.set_xlabel('Epoch')
p1.set_ylabel('Loss')
p2.plot(np.array(hold_loss_validation))
p2.set_title('Loss vs. Epoch - Validation Set')
p2.set_xlabel('Epoch')
p2.set_ylabel('Loss')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('plots/Loss_Curve', dpi=100)

# preparing output/target data
def prepare_data (tensor):
	tensor = tensor.data.cpu().numpy()
	tensor = tensor[:,:,Prediction_Horizon - 1]
	tensor = (tensor*SWLmax) - (tensor*SWLmin) + SWLmin
	tensor = tensor.flatten()
	return tensor

# plotting target against prediction set
def plotting(date,target_data,prediction_data,name):
	plt.close('all')
	fig, ax1 = plt.subplots(figsize=(16, 8))
	color = 'tab:blue'
	ax1.set_xlabel(None)
	ax1.set_ylabel('Moorwasserstand [m ü. NN]', size=18)
	ax1.plot(date, target_data, color = color, label='Gemessen')
	ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
	ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
	color = 'tab:red'
	ax1.plot(date, prediction_data, color=color, label='Vorhersage')
	plt.title(None)
	leg = ax1.legend()
	ax1.legend(loc='upper right', frameon=False, fontsize=18)
	fig.tight_layout()
	fig.savefig('plots/' + name)
	
# plot the target against prediction for the training phase
Prediction_Train = prepare_data(LSTM_output_train)
Target_Train = prepare_data(targettensor)

Date_Train = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon,'D'), np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon+len(Prediction_Train),'D'))
Date_Train = Date_Train.flatten()

plotting(Date_Train, Target_Train, Prediction_Train, 'Training_Set_Performance')

# plot the target against prediction for the validation phase
Prediction_Validation = prepare_data(LSTM_output_validation)
Target_Validation = prepare_data(targetval)

Date_Validation = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon+len(Prediction_Train)+count_train-len(Prediction_Train),'D'), np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon+len(Prediction_Train)+count_train-len(Prediction_Train)+len(Prediction_Validation),'D'))
Date_Validation = Date_Validation.flatten()

plotting(Date_Validation, Target_Validation, Prediction_Validation, 'Validation_Set_Performance')


# THE NETWORK TESTING
LSTM.eval()
#LSTM.load_state_dict(torch.load('Trained_G01_M01.pt'))
for datatest, targettest in test_loader:
	datatest = Variable(datatest)
	targettest = Variable(targettest)

	LSTM_output_test = LSTM(datatest)
	LSTM_test_loss = criterion(LSTM_output_test, targettest)

# plot the target against prediction for the testing phase
Prediction_Test = prepare_data(LSTM_output_test)
Target_Test = prepare_data(targettest)

Date_Test = np.arange(np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon+len(Prediction_Train)+count_train-len(Prediction_Train)+len(Prediction_Validation)+count_validation-len(Prediction_Validation),'D'), np.datetime64(data.iloc[0,0]) + np.timedelta64(Prediction_Horizon+len(Prediction_Train)+count_train-len(Prediction_Train)+len(Prediction_Validation)+count_validation-len(Prediction_Validation)+len(Prediction_Test),'D'))
Date_Test = Date_Test.flatten()

plotting(Date_Test, Target_Test, Prediction_Test, 'Testing_Set_Performance')

# the network performance measures
print(loss.item())
print(LSTM_validation_loss.item())
print(LSTM_test_loss.item())

def compute_statistics (prediction_data,target_data,set_name):
	if len(prediction_data) == len(target_data):
		print('Data_lenght_' + set_name + ': ', len(prediction_data))
		max_diff = np.max(np.subtract(prediction_data,target_data))
		print('Max_diff_'+ set_name +': ', max_diff)
		see = np.sum(np.square(np.subtract(prediction_data,target_data)))
		print('SSE_' + set_name +': ',see)
		mse =  see / len(prediction_data)
		print('MSE_' + set_name + ': ',mse)
	else:
		print('Lengh of Prediction and Target data are not the same')
	return max_diff,see,mse
	
Max_train, SSE_train, MSE_train = compute_statistics(Prediction_Train,Target_Train, 'train')
Max_validation, SSE_validation, MSE_validation = compute_statistics(Prediction_Validation,Target_Validation,'validation')
Max_test, SSE_test, MSE_test = compute_statistics(Prediction_Test,Target_Test,'test')

# write the statistics to a file
with open('statistics.csv', 'a+', newline = '') as f:
	file = csv.writer(f, delimiter = '\t')
	file.writerow(['Max_train','SSE_train','MSE_train','Max_validation','SSE_Validation','MSE_validation','Max_test','SSE_test','MSE_test'])
	file.writerow([Max_train,SSE_train,MSE_train,Max_validation,SSE_validation,MSE_validation,Max_test,SSE_test,MSE_test])

# SAVE THE MODEL PARATAMETERS
#torch.save(LSTM.state_dict(), 'Trained_G01_M01.pt')
