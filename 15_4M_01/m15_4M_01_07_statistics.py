# MODEL m15_4M_01 - STATISTICS

import numpy as np

# compute the statistics
def Compute_Statistics (prediction_data_fh, target_data_fh, prediction_data_5_24, target_data_5_24, prediction_data_3_24, target_data_3_24, prediction_data_1_24, target_data_1_24, set_name):
	if len(prediction_data_fh) == len(target_data_fh):
		print('Data_lenght_' + set_name + ': ', len(prediction_data_fh))
		max_diff_fh = np.max(np.subtract(prediction_data_fh,target_data_fh))
		print('Max_diff_fh_'+ set_name +': ', max_diff_fh)
		see_fh = np.sum(np.square(np.subtract(prediction_data_fh,target_data_fh)))
		print('SSE_fh_' + set_name +': ',see_fh)
		mse_fh =  see_fh / len(prediction_data_fh)
		print('MSE_fh_' + set_name + ': ',mse_fh)
		
		print('Data_lenght_' + set_name + ': ', len(prediction_data_5_24))
		max_diff_5_24 = np.max(np.subtract(prediction_data_5_24,target_data_5_24))
		print('Max_diff_5_24_'+ set_name +': ', max_diff_5_24)
		see_5_24 = np.sum(np.square(np.subtract(prediction_data_5_24,target_data_5_24)))
		print('SSE_5_24_' + set_name +': ',see_5_24)
		mse_5_24 =  see_5_24 / len(prediction_data_5_24)
		print('MSE_5_24_' + set_name + ': ',mse_5_24)

		print('Data_lenght_' + set_name + ': ', len(prediction_data_3_24))
		max_diff_3_24 = np.max(np.subtract(prediction_data_3_24,target_data_3_24))
		print('Max_diff_3_24_'+ set_name +': ', max_diff_3_24)
		see_3_24 = np.sum(np.square(np.subtract(prediction_data_3_24,target_data_3_24)))
		print('SSE_3_24_' + set_name +': ',see_3_24)
		mse_3_24 =  see_3_24 / len(prediction_data_3_24)
		print('MSE_3_24_' + set_name + ': ',mse_3_24)

		print('Data_lenght_' + set_name + ': ', len(prediction_data_1_24))
		max_diff_1_24 = np.max(np.subtract(prediction_data_1_24,target_data_1_24))
		print('Max_diff_1_24_'+ set_name +': ', max_diff_1_24)
		see_1_24 = np.sum(np.square(np.subtract(prediction_data_1_24,target_data_1_24)))
		print('SSE_1_24_' + set_name +': ',see_1_24)
		mse_1_24 =  see_1_24 / len(prediction_data_1_24)
		print('MSE_1_24_' + set_name + ': ',mse_1_24)
		

	else:
		print('Lengh of Prediction and Target data are not the same')
		
	return max_diff_fh, see_fh, mse_fh, mse_5_24, mse_3_24, mse_1_24
