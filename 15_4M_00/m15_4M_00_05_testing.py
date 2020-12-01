# MODEL m15_4M_00  - TESTING

from torch.autograd import Variable

# testing the artificial neural network 
def Testing(lstm, criterion, optimizer, test_loader):
	lstm.eval()
	#lstm.load_state_dict(torch.load('Trained_G01_M01.pt'))
	for inp_tensor_test, tar_tensor_test in test_loader:
	
		inp_tensor_test = Variable(inp_tensor_test)
		tar_tensor_test = Variable(tar_tensor_test)
	
		lstm_output_test = lstm(inp_tensor_test)
		lstm_test_loss = criterion(lstm_output_test, tar_tensor_test)
		
	return lstm_output_test, inp_tensor_test, tar_tensor_test