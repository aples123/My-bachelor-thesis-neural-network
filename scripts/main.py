import tensorflow as tf
import numpy as np
import sys
import data_preparation as d_prep
import data_visualisation as d_vis
import neural_network as n_n


"""Running the script with a configuration"""
tf.random.set_seed(1)
arguments = sys.argv[1:]
arguments = [argument.split('=') for argument in arguments]
arguments = {name: arg for name, arg in arguments}
if 'filepath' not in arguments:
    print('\n\n To read your configuration, run the script main.py from the terminal with a filepath argument containing a file path to a JSON file.')
    print('No argument was given, using default config7b.json', end='\n\n')
    arguments['filepath'] = '../resources/config7b.json'

"""Reading the configuration parameters"""
config_ = d_prep.read_config(arguments['filepath'])
filename = config_['filename']
qval_threshold = config_['qval_threshold']
learning_score_type = config_['learning_score_type']
all_decoy = config_['all_decoy']
initial_lr = config_['initial_lr']
epochs_number = config_['epochs_number']
data = d_prep.read_df(filename)
data['MMT_score'] = data['Mascot_score'] - data[['MIT', 'MHT']].min(axis=1)
d_prep.get_basic_data_info(data, f'Full dataset {filename}')

"""Computing q-values"""
score_types = ['Mascot_score', 'Mascot_delta_score', 'MMT_score']
data_sorted_list = []
for i in range(len(score_types)):
    data_sorted_list.append(d_prep.compute_qvalues(data, score_types[i]))
d_vis.qval_plot(data_sorted_list, score_types)
score_idx = score_types.index(learning_score_type)

"""Creating training and validation data"""
train_df, val_df = n_n.create_train_and_val_df(data_sorted_list, score_idx, qval_threshold, all_decoy)

"""Normalizing the data, creating a neural network, training and validating the model"""
x, _ = n_n.get_x_y(train_df)
x_min = np.min(x, axis=0).values
x_max = np.max(x, axis=0).values

hidden_neurons_number = 8
model, history = n_n.nn_training(hidden_neurons_number, train_df, val_df, x_min, x_max, all_decoy, initial_lr, epochs_number)

model.save('../resources/my_model.h5')
np.save('../resources/my_history.npy', history.history)

model = tf.keras.models.load_model('../resources/my_model.h5',
                                   custom_objects={"sensitivity": n_n.sensitivity, "specificity": n_n.specificity})
history = np.load('../resources/my_history.npy', allow_pickle=True).item()
model.summary()

d_vis.history_plot(history)

"""Using the model for a prediction on a whole dataset (unusual behaviour caused by the specifics of proteonomics)"""
x, y = n_n.get_x_y(data)
data = n_n.nn_predicting(x, model, data, x_min, x_max)

"""Comparing the model performance to commercial systems"""
score_types = ['Mascot_score', 'Mascot_delta_score', 'MMT_score', 'ANN_mscore']
d_vis.histogram_plot(score_types, data)

data_sorted_list = []
for i in range(len(score_types)):
    data_sorted_list.append(d_prep.compute_qvalues(data, score_types[i]))
d_vis.qval_plot(data_sorted_list, score_types)


