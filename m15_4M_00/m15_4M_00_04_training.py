# MODEL m15_4M_00 - TRAINING

import torch
from torch.autograd import Variable

# training the artificial neural network 
def Training(lstm, criterion, optimizer, epochs, train_loader, vali_loader, hold_loss_train, hold_loss_vali, train_dataset, batch_size_train):
	for epoch in range(epochs):
		for batch_idx, (inp_tensor_train, tar_tensor_train) in enumerate(train_loader):
			lstm.train()
			
			inp_tensor_train = Variable(inp_tensor_train)
			tar_tensor_train = Variable(tar_tensor_train)
			
			optimizer.zero_grad()
			
			lstm_output_train= lstm(inp_tensor_train)
			
			lstm_train_loss = criterion(lstm_output_train, tar_tensor_train)
			
			lstm_train_loss.backward()
	
			optimizer.step()
			
			# validating the artificial neural network 
			with torch.no_grad():
				for inp_tensor_vali, tar_tensor_vali in vali_loader:
					lstm.eval()
					
					inp_tensor_vali = Variable(inp_tensor_vali)
					tar_tensor_vali = Variable(tar_tensor_vali)
		
					lstm_output_vali = lstm(inp_tensor_vali)
					lstm_vali_loss = criterion(lstm_output_vali, tar_tensor_vali)
	
			# print out the loss versus epoch - training phase
			print('Training: Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch + 1, epochs, batch_idx + 1, len(train_dataset) // batch_size_train, lstm_train_loss.item()))
			
			hold_loss_train.append(lstm_train_loss.item())
			hold_loss_vali.append(lstm_vali_loss.item())
			
	return lstm_output_train, inp_tensor_train, tar_tensor_train, lstm_output_vali, inp_tensor_vali, tar_tensor_vali
