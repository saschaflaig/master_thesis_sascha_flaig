# MODEL m15_4M_01 application - PLOTTING + SAVING DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# plotting style

# denormalization
def Denormalization(tensor, swl_max, swl_min):
	tensor = tensor.data.cpu().numpy()
	tensor = tensor[0,:,:]
	tensor = (tensor*swl_max) - (tensor*swl_min) + swl_min
	tensor = tensor.flatten()
	return tensor
	
	
# plotting target against prediction set
def Prediction(prediction_data, plot_name):
	plt.close('all')
	fig, ax1 = plt.subplots(figsize=(16, 8))
	color = 'tab:blue'
	ax1.set_xlabel(None)
	ax1.set_ylabel('Moorwasserstand [m ü. NN]', size=18)
	ax1.plot(prediction_data, color = color, label='Vorhersage')
	ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
	ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
	color = 'tab:red'
	plt.title(None)
	leg = ax1.legend()
	ax1.legend(loc='upper right', frameon=False, fontsize=18)
	fig.tight_layout()
	fig.savefig('plots/' + plot_name)


# safe the processed data in files
def Safe_Data(inp_tensor, lstm_out, lstm_out_ren):
	pd.DataFrame(np.squeeze(inp_tensor.view(-1).numpy())).to_csv('data/01_input.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out.detach().numpy())).to_csv('data/02_lstm_out.csv', sep = "\t", float_format = '%.4f')
	pd.DataFrame(np.squeeze(lstm_out_ren)).to_csv('data/03_lstm_out_ren.csv', sep = "\t", float_format = '%.4f')