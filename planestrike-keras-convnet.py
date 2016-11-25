import numpy as np
import pylab, random, keras, math, h5py, json, gzip

from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import LearningRateScheduler

layer_name_dict = {
    'Dense': 'denseLayer',
    'Dropout': 'dropoutLayer',
    'Flatten': 'flattenLayer',
    'Embedding': 'embeddingLayer',
    'BatchNormalization': 'batchNormalizationLayer',
    'LeakyReLU': 'leakyReLULayer',
    'PReLU': 'parametricReLULayer',
    'ParametricSoftplus': 'parametricSoftplusLayer',
    'ThresholdedLinear': 'thresholdedLinearLayer',
    'ThresholdedReLu': 'thresholdedReLuLayer',
    'LSTM': 'rLSTMLayer',
    'GRU': 'rGRULayer',
    'JZS1': 'rJZS1Layer',
    'JZS2': 'rJZS2Layer',
    'JZS3': 'rJZS3Layer',
    'Convolution2D': 'convolution2DLayer',
    'MaxPooling2D': 'maxPooling2DLayer'
}

layer_params_dict = {
    'Dense': ['weights', 'activation'],
    'Dropout': ['p'],
    'Flatten': [],
    'Embedding': ['weights'],
    'BatchNormalization': ['weights', 'epsilon'],
    'LeakyReLU': ['alpha'],
    'PReLU': ['weights'],
    'ParametricSoftplus': ['weights'],
    'ThresholdedLinear': ['theta'],
    'ThresholdedReLu': ['theta'],
    'LSTM': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'GRU': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS1': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS2': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'JZS3': ['weights', 'activation', 'inner_activation', 'return_sequences'],
    'Convolution2D': ['weights', 'nb_filter', 'nb_row', 'nb_col', 'border_mode', 'subsample', 'activation'],
    'MaxPooling2D': ['pool_size', 'stride', 'ignore_border']
}

layer_weights_dict = {
    'Dense': ['W', 'b'],
    'Embedding': ['E'],
    'BatchNormalization': ['gamma', 'beta', 'mean', 'std'],
    'PReLU': ['alphas'],
    'ParametricSoftplus': ['alphas', 'betas'],
    'LSTM': ['W_xi', 'W_hi', 'b_i', 'W_xc', 'W_hc', 'b_c', 'W_xf', 'W_hf', 'b_f', 'W_xo', 'W_ho', 'b_o'],
    'GRU': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h'],
    'JZS1': ['W_xz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_hh', 'b_h', 'Pmat'],
    'JZS2': ['W_xz', 'W_hz', 'b_z', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h', 'Pmat'],
    'JZS3': ['W_xz', 'W_hz', 'b_z', 'W_xr', 'W_hr', 'b_r', 'W_xh', 'W_hh', 'b_h'],
    'Convolution2D': ['W', 'b']
}

def serialize(model_json_file, weights_hdf5_file, save_filepath, compress):
    with open(model_json_file, 'r') as f:
        model_metadata = json.load(f)
    weights_file = h5py.File(weights_hdf5_file, 'r')

    layers = []

    num_activation_layers = 0
    for k, layer in enumerate(model_metadata['config']):
        if layer['class_name'] == 'Activation':
            num_activation_layers += 1
            prev_layer_name = model_metadata['config'][k-1]['class_name']
            idx_activation = layer_params_dict[prev_layer_name].index('activation')
            layers[k-num_activation_layers]['parameters'][idx_activation] = layer['config']['activation']
            continue

        layer_params = []

        for param in layer_params_dict[layer['class_name']]:
            if param == 'weights':
                layer_weights = list(weights_file.keys())
                weights = {}
                weight_names = layer_weights_dict[layer['class_name']]
                for name in weight_names:
                    weights[name] = weights_file.get('{}/{}_{}'.format(layer['config']['name'], layer['config']['name'], name)).value.tolist()
                layer_params.append(weights)
            else:
                layer_params.append(layer['config'][param])

        layers.append({
            'layerName': layer_name_dict[layer['class_name']],
            'parameters': layer_params
        })


    if compress:
        with gzip.open(save_filepath, 'wb') as f:
            f.write(json.dumps(layers).encode('utf8'))
    else:
        with open(save_filepath, 'w') as f:
            json.dump(layers, f)

BOARD_HEIGHT = 6
BOARD_WIDTH = 6
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH
PLANE_SIZE = 8
TRAINING = True
ITERATIONS = 20000
HIDDEN_UNITS = BOARD_SIZE
OUTPUT_UNITS = BOARD_SIZE
ALPHA = 0.005      # step size
WINDOW_SIZE = 50

def init_plane():

    hidden_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))

    # Populate the plane's position
    # First figure out the plane's orientation
    #   0: heading right
    #   1: heading up
    #   2: heading left
    #   3: heading down

    plane_orientation = random.randint(0, 3)

    # Figrue out plane core's position as the '*' below
    #        |         | |
    #       -*-        |-*-
    #        |         | |
    #       ---
    if plane_orientation == 0:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(2, BOARD_WIDTH - 2)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column - 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column - 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column - 2] = 1
    elif plane_orientation == 1:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 3)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        hidden_board[plane_core_row + 2][plane_core_column] = 1
        hidden_board[plane_core_row + 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row + 2][plane_core_column - 1] = 1
    elif plane_orientation == 2:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column + 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column + 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column + 2] = 1
    elif plane_orientation == 3:
        plane_core_row = random.randint(2, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 2)
        # Populate the tail
        hidden_board[plane_core_row - 2][plane_core_column] = 1
        hidden_board[plane_core_row - 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row - 2][plane_core_column - 1] = 1

    # Populate the cross
    hidden_board[plane_core_row][plane_core_column] = 1
    hidden_board[plane_core_row + 1][plane_core_column] = 1
    hidden_board[plane_core_row - 1][plane_core_column] = 1
    hidden_board[plane_core_row][plane_core_column + 1] = 1
    hidden_board[plane_core_row][plane_core_column - 1] = 1
    return hidden_board

def play_game(training=TRAINING):
    hidden_board = init_plane()
    if training == False:
        print hidden_board
    game_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    board_pos_log = []
    action_log = []
    hit_log = []
    hits = 0
    while (hits < PLANE_SIZE and len(action_log) < BOARD_SIZE):
        board_pos_log.append(np.copy(game_board))
        probs = model.predict_proba(game_board.reshape(1,1,BOARD_HEIGHT, BOARD_WIDTH), verbose=0)[0]
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]
        if training:
            strike_pos = np.random.choice(BOARD_SIZE, p=probs)
        else:
            strike_pos = np.argmax(probs)
        x = strike_pos / BOARD_WIDTH
        y = strike_pos % BOARD_WIDTH
        if hidden_board[x][y] == 1:
            hits = hits + 1
            game_board[x][y] = 1
            hit_log.append(1)
        else:
            game_board[x][y] = -1
            hit_log.append(0)
        action_log.append(strike_pos)
        if training == False:
            print str(x) + ', ' + str(y) + ' *** ' + str(hit_log[-1])
    return board_pos_log, action_log, hit_log

def rewards_calculator(hit_log, gamma=0.5):
    """ Discounted sum of future hits over trajectory"""
    hit_log_weighted = [(item -
                         float(PLANE_SIZE - sum(hit_log[:index])) / float(BOARD_SIZE - index)) * (
            gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]


model = Sequential([
    Convolution2D(10, 3, 3, input_shape=(1, 6, 6)),
    Activation('relu'),
    Dropout(0.5),
    Flatten(),
    Dense(BOARD_SIZE),
    Activation('softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

game_lengths = []
reward = 0

def scheduler(epoch):
    return ALPHA * reward

change_lr = LearningRateScheduler(scheduler)

if TRAINING == True:
    for game in range(ITERATIONS):
        if game % 100 == 0:
            print game
            model.save('planestrike.h5')
        board_position_log, action_log, hit_log = play_game(training=TRAINING)
        game_lengths.append(len(action_log))
        rewards_log = rewards_calculator(hit_log)
        j = 0
        for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
            # Take step along gradient
            model.fit(current_board.reshape(1,1,BOARD_HEIGHT, BOARD_WIDTH), np.array([action]), verbose=0, nb_epoch=1, callbacks=[change_lr])


    running_average_length = [np.mean(game_lengths[i:i+WINDOW_SIZE]) for i in range(len(game_lengths)-WINDOW_SIZE)]
    pylab.plot(running_average_length)
    pylab.show()

    model_metadata = json.loads(model.to_json())
    with open('planestrike_keras_model.json', 'w') as f:
        json.dump(model_metadata, f)
    model.save_weights('planestrike_keras_weights.hdf5')
    serialize('planestrike_keras_model.json','planestrike_keras_weights.hdf5','planestrike_model.json',False)
    print 'finishing...'
else:
    model = load_model('planestrike.h5')
    play_game(TRAINING)
