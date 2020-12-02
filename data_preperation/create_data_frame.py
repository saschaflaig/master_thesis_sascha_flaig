# editing of the kup_maps data for the neural network - master thesis of sascha flaig 2020
#
# what is done? --> creates a data frame that includes the swamp water level data + precipitaion data + evapotranspiration data + the pumping rates 
# how to use/edit
#	- 'define the paramters' (earliest swl_data 2016-12-12)) --> run it --> produces a combined data frame of the normalized data (--> for the neural network)
#																			and a combined data frame with 'real' (not normalized) values
#	- if new data is added:
#		- add parameter and its path to the 'input_file.input'
#		- add the new paramter to the '#input file check'-section
#		- create a new edit section for the new paramter --> data frame
#		- add parameter to the 'create combined/final data frame' 


######################################################################################################################################
# define the parameters

strt_date = '2016-12-12'
end_date = '2020-05-06'


######################################################################################################################################
# imports

import numpy as np
import sys, os
from os.path import basename
import configparser
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# instructions 
if len(sys.argv) != 2:
	sys.stderr.write("Anwendung: {:} input_file\n".format(basename(sys.argv[0])))
	sys.exit()
	
	
######################################################################################################################################
# input file check

input_file_FN = os.path.abspath(sys.argv[1])
if not os.path.exists(input_file_FN):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_file_FN)
import_file = configparser.ConfigParser()
import_file.read(input_file_FN)

if 'input' not in import_file:
	raise ConfigError('Necessary section \'input\' in import_file \'{}\' not found.'.format(input_file_FN))
input = import_file['input']

if 'swl' not in input:
	raise ConfigError('Necessary key \'swl\' in import_file \'{}\' not found.'.format(input_file_FN))
swl = input['swl']

if 'p_obe' not in input:
	raise ConfigError('Necessary key \'p_obe\' in import_file \'{}\' not found.'.format(input_file_FN))
p_obe = input['p_obe']

if 'p_gar' not in input:
	raise ConfigError('Necessary key \'p_gar\' in import_file \'{}\' not found.'.format(input_file_FN))
p_gar = input['p_gar']

if 'et_obe' not in input:
	raise ConfigError('Necessary key \'et_obe\' in import_file \'{}\' not found.'.format(input_file_FN))
et_obe = input['et_obe']

if 'et_gar' not in input:
	raise ConfigError('Necessary key \'et_gar\' in import_file \'{}\' not found.'.format(input_file_FN))
et_gar = input['et_gar']

if 'w_2' not in input:
	raise ConfigError('Necessary key \'w_2\' in import_file \'{}\' not found.'.format(input_file_FN))
w_2 = input['w_2']

if 'w_3' not in input:
	raise ConfigError('Necessary key \'w_3\' in import_file \'{}\' not found.'.format(input_file_FN))
w_3 = input['w_3']

if 'w_4' not in input:
	raise ConfigError('Necessary key \'w_4\' in import_file \'{}\' not found.'.format(input_file_FN))
w_4 = input['w_4']

if 'w_5' not in input:
	raise ConfigError('Necessary key \'w_5\' in import_file \'{}\' not found.'.format(input_file_FN))
w_5 = input['w_5']

if 'w_6' not in input:
	raise ConfigError('Necessary key \'w_6\' in import_file \'{}\' not found.'.format(input_file_FN))
w_6 = input['w_6']


######################################################################################################################################	
# create plotting functions 

# function: plot data
def plot_data(y_label,data,name):
	plt.close('all')
	fig, ax1 = plt.subplots(figsize=(16, 8))
	ax1.set_xlabel(None)
	ax1.set_ylabel(y_label, size=18)
	ax1.plot(data)
	ax1.tick_params(axis='y', labelsize=18)
	ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
	#ax1.set_xlim('2016-12-5','2020-05-13')
	fig.tight_layout()
	plt.title(None)
	plt.savefig('graphs/'+name)
	
#function: plot data against swl
def plt_data_vs_swl(df_input_hourly, str_date, end_date, input_y_axis, name):
	df_swl_hourly_final_plt = swl_hourly.loc[str_date:end_date]
	df_input_hourly_final_plt = df_input_hourly.loc[str_date:end_date]
	
	plt.close('all')
	fig, ax1 = plt.subplots(figsize=(16, 8))
	color = 'tab:blue'
	ax1.set_xlabel(None)
	ax1.set_ylabel('Moorwasserstand [m ü. NN]', color=color, size=18)
	ax1.plot(df_swl_hourly_final_plt, color=color)
	ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
	ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
	#ax1.set_xlim('2016-12-5','2020-05-13')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:red'
	ax2.set_ylabel(input_y_axis, color=color, size=18)  # we already handled the x-label with ax1
	ax2.plot(df_input_hourly_final_plt, color=color, alpha=0.6)
	ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.title(None)
	plt.savefig('graphs/'+name)
	
	
######################################################################################################################################	
# create data frame for swamp water level
swl_data = pd.read_csv(swl,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
swl_data = swl_data.loc[strt_date:end_date]
swl_hourly = swl_data.resample('H').mean()
print(swl_hourly[swl_hourly.isnull().any(axis=1)]) # prints NAN
swl_hourly[swl_hourly.isnull().any(axis=1)].to_csv('data_edited/swl_hourly_NaN.csv', sep = "\t", float_format = '%.4f') #create csv with NaN
plot_data('Moorwasserstand [m ü. NN]', swl_hourly, 'swl_hourly_with_NaN') #plotting swl with NaN

# interpolate NaN
swl_hourly = swl_hourly.interpolate(method='time')
print(swl_hourly[swl_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now!
swl_hourly.to_csv('data_edited/swl_hourly_final.csv', sep = "\t", float_format = '%.4f')
print('swl_hourly_max: ',swl_hourly.max())
print('swl_hourly_min: ',swl_hourly.min())
plot_data('Moorwasserstand [m ü. NN]', swl_hourly, 'swl_hourly_final') # plotting swl

# calculate normalized data (with buffers)
swl_hourly_max = swl_hourly.max() + 0.15
print('swl_hourly_max_buffer: ',swl_hourly_max)
swl_hourly_min = swl_hourly.min() - 0.15
print('swl_hourly_min_buffer: ',swl_hourly_min)
swl_hourly_norm = (swl_hourly - swl_hourly_min)/(swl_hourly_max - swl_hourly_min)


######################################################################################################################################	
# create data frame for precipitaion - oberau
p_obe_data = pd.read_csv(p_obe,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
p_obe_data = p_obe_data.loc[strt_date:end_date]
p_obe_hourly = p_obe_data.resample('H').mean()
print(p_obe_hourly[p_obe_hourly.isnull().any(axis=1)]) # prints NaN
p_obe_hourly[p_obe_hourly.isnull().any(axis=1)].to_csv('data_edited/p_obe_hourly_NaN.csv', sep = "\t", float_format = '%.4f') # create csv with NaN
plot_data('Niederschlag [mm/h]', p_obe_hourly, 'p_obe_hourly_with_NaN') # plotting p_obe with NaN

# create data frame for precipitaion - garmisch
p_gar_data = pd.read_csv(p_gar,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
p_gar_data = p_gar_data.loc[strt_date:end_date]
p_gar_hourly = p_gar_data.resample('H').mean()
print(p_gar_hourly[p_gar_hourly.isnull().any(axis=1)]) # prints NaN
p_gar_hourly[p_gar_hourly.isnull().any(axis=1)].to_csv('data_edited/p_gar_hourly_NaN.csv', sep = "\t", float_format = '%.4f') # create csv with NaN
plot_data('Niederschlag [mm/h]', p_gar_hourly, 'p_gar_hourly_with_NaN') # plotting p_gar with NaN
	
# replace NaN of oberau precipitaion data set with data of garmisch precipitaion data set + interpolate the remaining NaN after
p_hourly = p_obe_hourly.fillna(p_gar_hourly)
print(p_hourly[p_hourly.isnull().any(axis=1)]) # prints NaN
p_hourly = p_hourly.interpolate(method='time') # interpolate NaN
print(p_hourly[p_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
p_hourly.to_csv('data_edited/p_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Niederschlag [mm/h]', p_hourly, 'p_hourly_final') # plotting p
print('p_hourly_max: ',p_hourly.max())
print('p_hourly_min: ',p_hourly.min())

# calculate normalized data (with buffers)
p_hourly_max = p_hourly.max() * 1.3
print('p_hourly_max_buffer: ', p_hourly_max)
p_hourly_min = p_hourly.min()
print('p_hourly_min_buffer: ', p_hourly_min)
p_hourly_norm = (p_hourly - p_hourly_min)/(p_hourly_max - p_hourly_min)

plt_data_vs_swl(p_hourly, strt_date, end_date, 'Niederschlag [mm/h]', 'p_vs_swl') # plotting p vs swl


######################################################################################################################################	
# create data frame evapotranspiration - oberau
et_obe_data = pd.read_csv(et_obe,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
et_obe_data = et_obe_data.loc[strt_date:end_date]
et_obe_hourly = et_obe_data.resample('H').mean()
print(et_obe_hourly[et_obe_hourly.isnull().any(axis=1)]) # prints NaN
et_obe_hourly[et_obe_hourly.isnull().any(axis=1)].to_csv('data_edited/et_obe_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Evapotranspiration [mm/h]', et_obe_hourly, 'et_obe_hourly_with_NaN') # plotting et_obe with NaN

# create data frame evapotranspiration - garmisch 
et_gar_data = pd.read_csv(et_gar,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
et_gar_data = et_gar_data.loc[strt_date:end_date]
et_gar_hourly = et_gar_data.resample('H').mean()
print(et_gar_hourly[et_gar_hourly.isnull().any(axis=1)]) # prints NaN
et_gar_hourly[et_gar_hourly.isnull().any(axis=1)].to_csv('data_edited/et_obe_hourly_NaN_2.csv', sep = "\t", float_format = '%.4f')
plot_data('Evapotranspiration [mm/h]', et_gar_hourly, 'et_gar_hourly_with_NaN') # plotting et_gar with NaN

# replace NaN of oberau evapotranspiration data set with data of garmisch evapotranspiration data set + interpolate the remaining NaN after
et_hourly = et_obe_hourly.fillna(et_gar_hourly)
print(et_hourly[et_hourly.isnull().any(axis=1)]) # prints NaN
et_hourly = et_hourly.interpolate(method='time') # interpolate NaN
print(et_hourly[et_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
et_hourly.to_csv('data_edited/et_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Evapotranspiration [mm/h]', et_hourly, 'et_hourly_final') # plotting et
print('et_hourly_max: ',et_hourly.max())
print('et_hourly_min: ',et_hourly.min())

# calculate normalized data 
et_hourly_max = et_hourly.max() * 1.3
print('et_hourly_max_buffer: ',et_hourly_max)
et_hourly_min = et_hourly.min()
print('et_hourly_min_buffer: ',et_hourly_min)
et_hourly_norm = (et_hourly - et_hourly_min)/(et_hourly_max - et_hourly_min)

plt_data_vs_swl(et_hourly, strt_date, end_date, 'Evapotranspiration [mm/h]', 'et_vs_swl') # plotting et vs swl


######################################################################################################################################	
# create data frame for all wells
w_2_data = pd.read_csv(w_2,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
w_2_hourly = w_2_data.loc[strt_date:end_date]
w_2_hourly = w_2_hourly.resample('H').mean()
print(w_2_hourly[w_2_hourly.isnull().any(axis=1)])# prints NaN
w_2_hourly[w_2_hourly.isnull().any(axis=1)].to_csv('data_edited/w_2_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_2_hourly, 'w_2_hourly_with_NaN') # plotting w_2 with NaN
w_2_hourly = w_2_hourly.interpolate(method='time') # interpolate NaN
w_2_hourly = w_2_hourly.rename(columns={'Entnahmerate  [l/s]' : 'Entnahmerate_w_2  [l/s]'})
print(w_2_hourly[w_2_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
w_2_hourly.to_csv('data_edited/w_2_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_2_hourly, 'w_2_hourly_final') # plotting w_2
print('w_2_hourly_max: ',w_2_hourly.max())
print('w_2_hourly_min: ',w_2_hourly.min())

# calculate normalized data
w_2_hourly_max = w_2_hourly.max()
print('w_2_hourly_max_buffer: ',w_2_hourly_max)
w_2_hourly_min = w_2_hourly.min()
print('w_2_hourly_min_buffer: ',w_2_hourly_min)
w_2_hourly_norm = (w_2_hourly - w_2_hourly_min)/(w_2_hourly_max - w_2_hourly_min)

plt_data_vs_swl(w_2_hourly, strt_date, end_date, 'Pumprate Brunnen 2 [l/s]', 'w_2_vs_swl') # plotting w_2 vs swl


w_3_data = pd.read_csv(w_3,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
w_3_hourly = w_3_data.loc[strt_date:end_date]
w_3_hourly = w_3_hourly.resample('H').mean()
print(w_3_hourly[w_3_hourly.isnull().any(axis=1)])# prints NaN
w_3_hourly[w_3_hourly.isnull().any(axis=1)].to_csv('data_edited/w_3_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_3_hourly, 'w_3_hourly_with_NaN') # plotting w_3 with NaN
w_3_hourly = w_3_hourly.interpolate(method='time') # interpolate NaN
w_3_hourly = w_3_hourly.rename(columns={'Entnahmerate  [l/s]' : 'Entnahmerate_w_3  [l/s]'})
print(w_3_hourly[w_3_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
w_3_hourly.to_csv('data_edited/w_3_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_3_hourly, 'w_3_hourly_final') # plotting w_3
print('w_3_hourly_max: ',w_3_hourly.max())
print('w_3_hourly_min: ',w_3_hourly.min())

# calculate normalized data
w_3_hourly_max = w_3_hourly.max()
print('w_3_hourly_max_buffer: ',w_3_hourly_max)
w_3_hourly_min = w_3_hourly.min()
print('w_3_hourly_min_buffer: ',w_3_hourly_min)
w_3_hourly_norm = (w_3_hourly - w_3_hourly_min)/(w_3_hourly_max - w_3_hourly_min)

plt_data_vs_swl(w_3_hourly, strt_date, end_date, 'Pumprate Brunnen 3 [l/s]', 'w_3_vs_swl') # plotting w_3 vs swl


w_4_data = pd.read_csv(w_4,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
w_4_hourly = w_4_data.loc[strt_date:end_date]
w_4_hourly = w_4_hourly.resample('H').mean()
print(w_4_hourly[w_4_hourly.isnull().any(axis=1)])# prints NaN
w_4_hourly[w_4_hourly.isnull().any(axis=1)].to_csv('data_edited/w_4_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_4_hourly, 'w_4_hourly_with_NaN') # plotting w_4 with NaN
w_4_hourly = w_4_hourly.interpolate(method='time') # interpolate NaN
w_4_hourly = w_4_hourly.rename(columns={'Entnahmerate  [l/s]' : 'Entnahmerate_w_4  [l/s]'})
print(w_4_hourly[w_4_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
w_4_hourly.to_csv('data_edited/w_4_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_4_hourly, 'w_4_hourly_final') # plotting w_4
print('w_4_hourly_max: ',w_4_hourly.max())
print('w_4_hourly_min: ',w_4_hourly.min())

# calculate normalized data
w_4_hourly_max = w_4_hourly.max()
print('w_4_hourly_max_buffer: ',w_4_hourly_max)
w_4_hourly_min = w_4_hourly.min()
print('w_4_hourly_min_buffer: ',w_4_hourly_min)
w_4_hourly_norm = (w_4_hourly - w_4_hourly_min)/(w_4_hourly_max - w_4_hourly_min)

plt_data_vs_swl(w_4_hourly, strt_date, end_date, 'Pumprate Brunnen 4 [l/s]', 'w_4_vs_swl') # plotting w_4 vs swl


w_5_data = pd.read_csv(w_5,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
w_5_hourly = w_5_data.loc[strt_date:end_date]
w_5_hourly = w_5_hourly.resample('H').mean()
print(w_5_hourly[w_5_hourly.isnull().any(axis=1)])# prints NaN
w_5_hourly[w_5_hourly.isnull().any(axis=1)].to_csv('data_edited/w_5_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_5_hourly, 'w_5_hourly_with_NaN') # plotting w_5 with NaN
w_5_hourly = w_5_hourly.interpolate(method='time') # interpolate NaN
w_5_hourly = w_5_hourly.rename(columns={'Entnahmerate  [l/s]' : 'Entnahmerate_w_5  [l/s]'})
print(w_5_hourly[w_5_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
w_5_hourly.to_csv('data_edited/w_5_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_5_hourly, 'w_5_hourly_final') # plotting w_5
print('w_5_hourly_max: ',w_5_hourly.max())
print('w_5_hourly_min: ',w_5_hourly.min())

# calculate normalized data
w_5_hourly_max = w_5_hourly.max()
print('w_5_hourly_max_buffer: ',w_5_hourly_max)
w_5_hourly_min = w_5_hourly.min()
print('w_5_hourly_min_buffer: ',w_5_hourly_min)
w_5_hourly_norm = (w_5_hourly - w_5_hourly_min)/(w_5_hourly_max - w_5_hourly_min)

plt_data_vs_swl(w_5_hourly, strt_date, end_date, 'Pumprate Brunnen 5 [l/s]', 'w_5_vs_swl') # plotting w_5 vs swl


w_6_data = pd.read_csv(w_6,sep="\t", index_col=['Timestamp'],parse_dates={'Timestamp':['# Zeitpunkt']},date_parser=dateparse)
w_6_hourly = w_6_data.loc[strt_date:end_date]
w_6_hourly = w_6_hourly.resample('H').mean()
print(w_6_hourly[w_6_hourly.isnull().any(axis=1)])# prints NaN
w_6_hourly[w_6_hourly.isnull().any(axis=1)].to_csv('data_edited/w_6_hourly_NaN.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_6_hourly, 'w_6_hourly_with_NaN') # plotting w_6 with NaN
w_6_hourly = w_6_hourly.interpolate(method='time') # interpolate NaN
w_6_hourly = w_6_hourly.rename(columns={'Entnahmerate  [l/s]' : 'Entnahmerate_w_6  [l/s]'})
print(w_6_hourly[w_6_hourly.isnull().any(axis=1)]) # prints NaN --> should be empty now
w_6_hourly.to_csv('data_edited/w_6_hourly_final.csv', sep = "\t", float_format = '%.4f')
plot_data('Entnahmerate [l/s]', w_6_hourly, 'w_6_hourly_final') # plotting w_6
print('w_6_hourly_max: ',w_6_hourly.max())
print('w_6_hourly_min: ',w_6_hourly.min())

# calculate normalized data
w_6_hourly_max = w_6_hourly.max()
print('w_6_hourly_max_buffer: ',w_6_hourly_max)
w_6_hourly_min = w_6_hourly.min()
print('w_6_hourly_min_buffer: ',w_6_hourly_min)
w_6_hourly_norm = (w_6_hourly - w_6_hourly_min)/(w_6_hourly_max - w_6_hourly_min)

plt_data_vs_swl(w_6_hourly, strt_date, end_date, 'Pumprate Brunnen 6 [l/s]', 'w_6_vs_swl') # plotting w_6 vs swl


######################################################################################################################################	
# create combined/final data frame
df_swl_hourly_final = swl_hourly.loc[strt_date:end_date] # hourly swamp water level data frame
df_swl_hourly_norm_final = swl_hourly_norm.loc[strt_date:end_date] # normalized hourly swamp water level data frame

df_p_hourly_final = p_hourly.loc[strt_date:end_date] # hourly precipitaion data frame
df_p_hourly_norm_final = p_hourly_norm.loc[strt_date:end_date] # normalized hourly precipitaion data frame

df_et_hourly = et_hourly.loc[strt_date:end_date] # hourly evapotranspiration data frame
df_et_hourly_norm_final = et_hourly_norm.loc[strt_date:end_date] # normalized hourly evapotranspiration data frame

df_w_2_hourly_final = w_2_hourly.loc[strt_date:end_date] # hourly well 2 pumping data frame
df_w_2_hourly_norm_final = w_2_hourly_norm.loc[strt_date:end_date] # normalized hourly well 2 pumping data frame

df_w_3_hourly_final = w_3_hourly.loc[strt_date:end_date] # hourly well 3 pumping data frame
df_w_3_hourly_norm_final = w_3_hourly_norm.loc[strt_date:end_date] # normalized hourly well 3 pumping data frame

df_w_4_hourly_final = w_4_hourly.loc[strt_date:end_date] # hourly well 4 pumping data frame
df_w_4_hourly_norm_final = w_4_hourly_norm.loc[strt_date:end_date] # normalized hourly well 4 pumping data frame

df_w_5_hourly_final = w_5_hourly.loc[strt_date:end_date] # hourly well 5 pumping data frame
df_w_5_hourly_norm_final = w_5_hourly_norm.loc[strt_date:end_date] # normalized hourly well 5 pumping data frame

df_w_6_hourly_final = w_6_hourly.loc[strt_date:end_date] # hourly well 6 pumping data frame
df_w_6_hourly_norm_final = w_6_hourly_norm.loc[strt_date:end_date] # normalized hourly well 6 pumping data frame

# combined data frame - normalized
df_final = pd.merge(df_swl_hourly_norm_final, df_p_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_et_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_w_2_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_w_3_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_w_4_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_w_5_hourly_norm_final, on = 'Timestamp')
df_final = pd.merge(df_final, df_w_6_hourly_norm_final, on = 'Timestamp')
df_final.to_csv('df_final.csv', sep = "\t", float_format = '%.4f')
print (df_final.head())

# combined data frame
df_not_norm = pd.merge(df_swl_hourly_final, df_p_hourly_final, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_et_hourly, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_w_2_hourly_final, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_w_3_hourly_final, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_w_4_hourly_final, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_w_5_hourly_final, on = 'Timestamp')
df_not_norm = pd.merge(df_not_norm, df_w_6_hourly_final, on = 'Timestamp')
df_not_norm.to_csv('df_not_norm.csv', sep = "\t", float_format = '%.4f')
print (df_not_norm.head())