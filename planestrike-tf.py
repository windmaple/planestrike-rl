import tensorflow as tf
import numpy as np

import pylab, random

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
    while (hits < 8 and len(action_log) < BOARD_SIZE):
        board_pos_log.append([game_board.flatten()])
        tmp = sess.run([probabilities], feed_dict={input_positions:[game_board.flatten()]})
        probs = tmp[0][0]
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




input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))
labels = tf.placeholder(tf.int64)
learning_rate = tf.placeholder(tf.float32, shape=[])

# hidden layer
W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE, HIDDEN_UNITS], stddev=0.1 / np.sqrt(float(BOARD_SIZE))))
b1 = tf.Variable(tf.zeros([1, HIDDEN_UNITS]))
h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)

# 2nd layer
W2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUTPUT_UNITS], stddev=0.1 / np.sqrt(float(HIDDEN_UNITS))))
b2 = tf.Variable(tf.zeros([1, OUTPUT_UNITS]))
logits = tf.matmul(h1, W2) + b2
probabilities = tf.nn.softmax(logits)

init = tf.initialize_all_variables()
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)
# Start TF session
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


game_lengths = []


if TRAINING == True:
    for game in range(ITERATIONS):
        if game % 1000 == 0:
            print game
            saver.save(sess, 'model.ckpt')
        board_position_log, action_log, hit_log = play_game(training=TRAINING)
        game_lengths.append(len(action_log))
        rewards_log = rewards_calculator(hit_log)
        for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
            # Take step along gradient
            if TRAINING:
                sess.run([train_step],
                    feed_dict={input_positions:current_board, labels:[action], learning_rate:ALPHA * reward})

    running_average_length = [np.mean(game_lengths[i:i + WINDOW_SIZE]) for i in range(len(game_lengths) - WINDOW_SIZE)]
    pylab.plot(running_average_length)
    pylab.show()
else:
    saver.restore(sess, 'model.ckpt')
    play_game(False)
