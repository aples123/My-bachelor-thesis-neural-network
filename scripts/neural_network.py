import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter
from tensorflow import keras
import tensorflow.keras.backend as K
from tqdm.keras import TqdmCallback


def get_x_y(data):
    """
    Splits the data into two sets, one containing data features and a vector of its labels.
    """
    y = data.iloc[:, 0]
    x = data.drop('ANN_mscore', axis=1, errors='ignore').iloc[:, 1:]
    return x, y


def create_train_and_val_df(data_sorted_list, score_idx, qval_threshold, all_decoy):
    """
    Creates training and validation data taking into account the configuration parameters.
    Uses data_sorted_list - a list of datasets sorted according to different score types.
    Chooses the dataset sorted according to the given score type index.
    Extracts only the biological (target samples) of q-values under the given threshold.
    Adds the same number of decoy samples or all available decoy samples, depending on all_decoy being True or False.
    Splits the dataset so that 90% of chosen samples is used for training and 10% for validation.
    Returns the training and validation datasets
    """
    target_df = data_sorted_list[score_idx].loc[
        (data_sorted_list[score_idx]['pos_neg'] == 1) & (data_sorted_list[score_idx]['qvalue'] <= qval_threshold)].sample(
        frac=1, random_state=1).reset_index(drop=True)
    decoy_df = data_sorted_list[score_idx].loc[(data_sorted_list[score_idx]['pos_neg'] == 0)].sample(frac=1, random_state=1).reset_index(drop=True)

    if all_decoy:
        sample_fraction = 1
    else:
        sample_fraction = min(target_df.shape[0] / decoy_df.shape[0], 1)
    decoy_df = decoy_df.sample(frac=sample_fraction, random_state=1).reset_index(drop=True)

    val_target_idx = int(target_df.shape[0] * 0.1)
    val_target = target_df.iloc[:val_target_idx]
    target_df = target_df.iloc[val_target_idx:]

    val_decoy_idx = int(decoy_df.shape[0] * 0.1)
    val_decoy = decoy_df.iloc[:val_decoy_idx]
    decoy_df = decoy_df.iloc[val_decoy_idx:]

    train_df = pd.concat([target_df, decoy_df]).sample(frac=1, random_state=1).reset_index(drop=True).drop(['qvalue', 'FDR'], axis=1)
    val_df = pd.concat([val_target, val_decoy]).sample(frac=1, random_state=1).reset_index(drop=True).drop(['qvalue', 'FDR'], axis=1)
    return train_df, val_df


def data_normalization(x, x_min, x_max):
    """
    Normalizes the data using two vectors - with minimum and maximum values of every feature from the training data.
    Returns the normalized data.
    """
    x = x.values
    return (x - x_min) / (x_max - x_min)


def compute_sample_weights(train_df):
    """
    Computes the sample weights using class counts.
    The weight value is a fraction of N_min/N, where N_min is a number of samples of the least numerous class
    and N is the number of class samples that weight is computed.
    Equals one for N = N_min, less than one for N > N_min.
    Returns a dict of class labels and corresponding weights and a label of the most numerous class.
    """
    class_count_dict = dict(Counter(train_df['pos_neg']))
    min_class_count = min(class_count_dict.values())
    max_class_key = max(class_count_dict, key=class_count_dict.get)
    sample_weights_dict = {}
    for key in class_count_dict:
        sample_weights_dict[key] = min_class_count / class_count_dict[key]
    return sample_weights_dict, max_class_key


def sensitivity(y_true, y_pred):
    """
    Calculates the sensitivity
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    """
    Calculates the specificity
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positives = K.sum(neg_y_true * y_pred)
    true_negatives = K.sum(neg_y_true * neg_y_pred)
    return true_negatives / (true_negatives + false_positives + K.epsilon())


def nn_training(hidden_neurons_number, train_df, val_df, x_min, x_max, all_decoy, initial_lr, epochs_number):
    """
    Main function of the model.
    Uses all the other functions from this module to build, train and validate the neural network model.
    Returns a model and its history.
    """
    x, y = get_x_y(train_df)
    sample_weights = np.ones(shape=(len(y),))

    x_val, y_val = get_x_y(val_df)
    val_sample_weights = np.ones(shape=(len(y_val),))

    if all_decoy:
        sample_weights_dict, max_class_key = compute_sample_weights(train_df)
        sample_weights[y == max_class_key] = sample_weights_dict[max_class_key]
        val_sample_weights[y_val == max_class_key] = sample_weights_dict[max_class_key]

    x = data_normalization(x, x_min, x_max)

    x_val = data_normalization(x_val, x_min, x_max)
    validation_data = (x_val, y_val, val_sample_weights)

    n_features = x.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_features,)),
        tf.keras.layers.Dense(hidden_neurons_number, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=100000,
        decay_rate=0.95)
    opt = keras.optimizers.SGD(learning_rate=lr_schedule)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', sensitivity, specificity])

    history = model.fit(x, y, sample_weight=sample_weights, epochs=epochs_number,
                        verbose=0, callbacks=[TqdmCallback(verbose=0), callback], validation_data=validation_data)
    return model, history


def nn_predicting(x, model, data, x_min, x_max):
    """
    Normalizes the data and makes a prediction using a trained model.
    Final values are in the range of 0 to 100, similarly to scores usually used in that domain.
    Returns the dataframe with a column containing the final values of the new score.
    """
    x = data_normalization(x, x_min, x_max)
    mscore = model.predict(x)
    data['ANN_mscore'] = mscore * 100
    return data
