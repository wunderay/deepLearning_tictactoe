# State setting of the board and players

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
num_actions = 9


class State:
    def __init__(self, player1, player2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.player1 = player1
        self.player2 = player2
        self.isEnd = False
        self.boardHash = None
        # player player 1 goes first by initializing it first
        self.playerSymbol = 1

    # getting the unique hash of the current state of the board
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    # logic to get the winner
    def winner(self):

        # checking row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        # checking column
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        # checking diagonal
        diag_sum1_l_to_r = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2_r_to_l = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1_l_to_r), abs(diag_sum2_r_to_l))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1_l_to_r == 3 or diag_sum2_r_to_l == 3:
                return 1
            else:
                return -1

        # checking for tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    # checking for all the redundant positions
    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                # 0 indicates not occupied and is available
                # thus, if 0 the we append that board position into positions array
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # The game end !!
    def giveReward(self):

        result = self.winner()

        # backpropagate reward
        # player 1 is winner
        if result == 1:
            self.player1.feed_Reward(1)
            self.player2.feed_Reward(0)

        # player 2 is winner
        elif result == -1:
            self.player1.feed_Reward(0)
            self.player2.feed_Reward(1)

        # game is a tie
        else:
            self.player1.feed_Reward(0.1)
            self.player2.feed_Reward(0.5)

    # Resetting the board
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def train(self):
        epoch = 100
        for episode in range(epoch):
            print("Epoch {} / {}".format(episode, epoch))
            actions = []
            while not self.isEnd:
                if self.playerSymbol > 0:
                    action = self.player1.make_Move(self.availablePositions(), self.board, self.playerSymbol)
                    actions.append(action)
                    self.player1.add_State(self.board)
                    current_state = self.board
                    try:
                        self.availablePositions().index(action)
                    except ValueError:
                        self.player1.feed_Reward(-2)
                    else:
                        self.updateState(action)
                    next_state = self.board
                else:
                    action = self.player2.make_Move(self.availablePositions(), self.board, self.playerSymbol)
                    actions.append(action)
                    self.player2.add_State(self.board)
                    current_state = self.board
                    try:
                        self.availablePositions().index(action)
                    except ValueError:
                        self.player2.feed_Reward(-2)
                    else:
                        self.updateState(action)
                    next_state = self.board
                if len(self.availablePositions()) ==0:
                    self.isEnd = True
            self.giveReward()
            self.reset()
            self.player1.train_model()
            self.player2.train_model()
            self.player1.reset()
            self.player2.reset()
        return self.player1, self.player2

    # playing with human player
    def playWithHuman(self):
        while not self.isEnd:
            # Player_1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            # self.showBoard() --------------------------------Fernando needs to do GUI for this portion
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "It's a win!")
                else:
                    print("It's a tie!")
                self.reset()
                break

            else:
                # Player_2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                # self.showBoard() --------------------------------Fernando needs to do GUI for this portion
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "It's a win!")
                    else:
                        print("It's a tie!")
                    self.reset()
                    break


from tensorflow.keras import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # array of every position that was taken
        self.lr = 0.2
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.decay_gamma = 0.9
        self.states_value = {}  # states should correspond to the value
        self.actions = []
        self.model = self.create_model()
        # "hack" implemented by DeepMind to improve convergence
        # self.target_model = self.create_model()
        self.rewards = []

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def load_model(self, filename):
        """
        :param filename: name of the model
        sets the self.model and self.target_model to the model passed in
        :return: none
        """
        self.model = load_model(filename)
        # self.target_model = load_model(filename)

    # Saves model to specified path
    def save_model(self, filename):
        self.model.save(filename)

    def create_model(self):
        """
        MODIFY THIS
        Defines and creates the neural network
        :return: returns the neural network model
        """
        # FIX MODEL
        self.model = Sequential()
        self.model.add(layers.InputLayer(input_shape=(3, 3, 1)))
        self.model.add(Conv2D(12, (2, 2), activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(Dense(num_actions))
        self.model.compile(loss="MeanSquaredError",
                           optimizer=Adam(learning_rate=self.lr))
        self.model.summary()
        return self.model

    def make_Move(self, positions, current_board, symbol):
        """
        :param positions:
        :param current_board:
        :param symbol:
        :return: index position for where the player will place their piece next
        """
        # action = 0
        next_board = current_board.copy()
        if np.random.uniform(0, 1) < self.epsilon:
            #  make a random move on the board
            index = np.random.choice(len(positions))
            action = positions[index]
        else:
            next_board = np.expand_dims(current_board, -1)
            next_board = np.expand_dims(next_board, 0)
            action = np.amax(self.model.predict(next_board), axis=1)
        # Reduce the number of random actions as the model learns more
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon >= self.epsilon_min else self.epsilon_min
        self.add_State(current_board)
        self.actions.append(action)
        return action

    def train_model(self):
        self.states = [np.expand_dims(state, -1) for state in self.states]
        self.states = [np.expand_dims(state, 0) for state in self.states]
        state_list = np.asarray(self.states)

        self.rewards = np.array(self.rewards)
        self.rewards = [np.asarray(reward) for reward in self.rewards]
        targets = []
        index = 0
        for state in state_list:
            targets.append(self.rewards[index] + self.decay_gamma * np.amax(self.model.predict(state), axis=1))
            index += 1
        targets = np.asarray(targets)
        index = 0
        for target, state in zip(targets, state_list):
            self.model.fit(state, target)
            if index > 9:
                break
            index +=1
        targets = None
        state_list = None

    # additional hashstate
    def add_State(self, state):
        self.states.append(state)
        self.rewards.append(0)

    def get_state(self, index=-1):
        return self.states[index]

    # at the end of game, backpropagate and update states value
    def feed_Reward(self, reward):
        for st in range(len(self.states)):
            self.rewards[-st] += self.lr * (self.decay_gamma * reward - self.rewards[-st])
            reward = self.rewards[-st]
            # self.rewards.append(reward)

    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def save_Policy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_Policy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def make_Move(self, positions):  # Have the player insert a row and column
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def add_State(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feed_Reward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":

    # train the NN

    # P1 = Player("P1")
    #
    # P2 = Player("P2")
    #
    # cs = State(P1, P2)
    #
    # print("Training the Neural Network...")
    #
    # play1, play2 = cs.train()
    # play1.save_model("player1")
    # play2.save_model("player2")

    # Play against a human
    P1 = Player("Computer Player", exp_rate=0)

    P1.load_model("player1")

    P2 = HumanPlayer("Human")

    cs = State(P1, P2)
    cs.playWithHuman()