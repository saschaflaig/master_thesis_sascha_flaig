# MODEL m15_4M_01- PLOTTING + SAVING DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# plot the loss vs. epoch - training and validation set
def Loss_Vs_Epoch(hold_loss_train, hold_loss_vali):
	fig, (p1, p2) = plt.subplots(2, 1, sharey=False, figsize=(16, 8))
	plt.subplots_adjust(hspace=0.5)
	p1.plot(np.array(hold_loss_train))
	p1.set_title('Fehler vs. Epoche - Trainingsdatensatz', size=18)
	p1.tick_params(axis='x', labelsize=16)
	p1.tick_params(axis='y', labelsize=16)
	p1.set_xlabel('Epoche', size=18)
	p1.set_ylabel('Fehler', size=18)
	p2.plot(np.array(hold_loss_vali))
	p2.set_title('Fehler vs. Epoche - Validierungsdatensatz', size=18)
	p2.tick_params(axis='x', labelsize=16)
	p2.tick_params(axis='y', labelsize=16)
	p2.set_xlabel('Epoch', size=18)
	p2.set_ylabel('Fehler', size=18)
	fig.savefig('plots/loss_curve')

# denormalization of output/target data
def Denormalization(tensor, forecast_horizon, swl_max, swl_min):
	tensor = tensor.data.cpu().numpy()
	tensor_fh = tensor[:,:,forecast_horizon - 1]
	tensor_fh = (tensor_fh*swl_max) - (tensor_fh*swl_min) + swl_min
	tensor_fh = tensor_fh.flatten()
	tensor_5_24 = tensor[:,:, 5*24 - 1]
	tensor_5_24 = (tensor_5_24*swl_max) - (tensor_5_24*swl_min) + swl_min
	tensor_5_24 = tensor_5_24.flatten()
	tensor_3_24 = tensor[:,:, 3*24 - 1]
	tensor_3_24 = (tensor_3_24*swl_max) - (tensor_3_24*swl_min) + swl_min
	tensor_3_24 = tensor_3_24.flatten()
	tensor_1_24 = tensor[:,:, 1*24 - 1]
	tensor_1_24 = (tensor_1_24*swl_max) - (tensor_1_24*swl_min) + swl_min
	tensor_1_24 = tensor_1_24.flatten()
	return tensor_fh, tensor_5_24, tensor_3_24, tensor_1_24

# plotting target against prediction set
def Target_Vs_Prediction(date_range, target_data, prediction_data, plot_name):
	plt.close('all')
	fig, ax1 = plt.subplots(figsize=(16, 8))
	color = 'tab:blue'
	ax1.set_xlabel(None)
	ax1.set_ylabel('Moorwasserstand [m ü. NN]', size=18)
	ax1.plot(date_range, target_data, color = color, label='Gemessen')
	ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
	ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
	color = 'tab:red'
	ax1.plot(date_range, prediction_data, color=color, label='Vorhersage')
	plt.title(None)
	leg = ax1.legend()
	ax1.legend(loc='upper right', frameon=False, fontsize=18)
	fig.tight_layout()
	fig.savefig('plots/' + plot_name)

# safe the processed data in files
def Safe_Data(inp_tensor, tar_tensor, lstm_out, lstm_out_fh_ren, target_fh_ren, lstm_out_5_24_ren, target_5_24_ren, lstm_out_3_24_ren, target_3_24_ren, lstm_out_1_24_ren, target_1_24_ren, set_name, forecast_horizon):
	pd.DataFrame(np.squeeze(inp_tensor.view(-1).numpy())).to_csv('data/' + set_name + '/01_input.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(tar_tensor.numpy())).to_csv('data/' + set_name + '/02_target.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(tar_tensor.numpy()[:,:,forecast_horizon - 1])).to_csv('data/' + set_name + '/03_target_end.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out.detach().numpy())).to_csv('data/' + set_name + '/04_lstm_out.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out_fh_ren)).to_csv('data/' + set_name + '/05_lstm_out_fh_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(target_fh_ren)).to_csv('data/' + set_name + '/06_target_fh_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out_5_24_ren)).to_csv('data/' + set_name + '/07_lstm_out_5_25_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(target_5_24_ren)).to_csv('data/' + set_name + '/08_target_5_25_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out_3_24_ren)).to_csv('data/' + set_name + '/09_lstm_out_3_24_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(target_3_24_ren)).to_csv('data/' + set_name + '/10_target_3_24_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out_1_24_ren)).to_csv('data/' + set_name + '/11_lstm_out_1_24_ren.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(target_1_24_ren)).to_csv('data/' + set_name + '/12_target_1_24_ren.csv', sep = "\t", float_format = '%.4f')