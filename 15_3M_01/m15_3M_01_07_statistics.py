# MODEL m15_3M_01 - STATISTICS

import numpy as np

# compute the statistics
def Compute_Statistics (prediction_data,target_data,set_name):
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
		
	return max_diff, see, mse
