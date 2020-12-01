# MODEL m15_4M_02 - TESTING

from torch.autograd import Variable

# testing the artificial neural network 
def Testing(lstm, criterion, optimizer, test_loader, batch_size_test, forecast_horizon, input_size_swl, input_size, hidden_size, hold_lstm_output_test, hold_inp_tensor_test, hold_tar_tensor_test, num_lstm_layers):
	lstm.eval()
	#lstm.load_state_dict(torch.load('Trained_G01_M01.pt'))
	for inp_tensor_swl_train, inp_tensor_test, tar_tensor_test in test_loader:
		
		inp_tensor_swl_train = Variable(inp_tensor_swl_train)
		inp_tensor_swl_train = inp_tensor_swl_train.view(batch_size_test, 1, input_size_swl)
		inp_tensor_test = Variable(inp_tensor_test)
		inp_tensor_test = inp_tensor_test.view(batch_size_test, forecast_horizon - 1, input_size)
		tar_tensor_test = Variable(tar_tensor_test)
		tar_tensor_test = tar_tensor_test.view(batch_size_test, 1, forecast_horizon)
	
		lstm_output_test = lstm(inp_tensor_swl_train, inp_tensor_test)
		lstm_test_loss = criterion(lstm_output_test, tar_tensor_test)
		
		hold_lstm_output_test.append(lstm_output_test)
		hold_inp_tensor_test.append(inp_tensor_test)
		hold_tar_tensor_test.append(tar_tensor_test)