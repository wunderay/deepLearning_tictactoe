# State setting of the board and players

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras import *
import numpy as np
import pickle
from tkinter import *
from tkinter import messagebox

root = Tk()
root.title('Tic-Tac-Toe - Deep Learning')
root.resizable(False, False)

BOARD_ROWS = 3
BOARD_COLS = 3
num_actions = 9

hasTakenTurn = False
humanAction = None


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
        diag_sum2_r_to_l = sum([self.board[i, BOARD_COLS - i - 1]
                                for i in range(BOARD_COLS)])
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
        epoch = 50
        for episode in range(epoch):
            print("Epoch {} / {}".format(episode + 1, epoch))
            actions = []
            while not self.isEnd:
                if self.playerSymbol > 0:
                    action = self.player1.make_Move(
                        self.availablePositions(), self.board, self.playerSymbol)
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
                    action = self.player2.make_Move(
                        self.availablePositions(), self.board, self.playerSymbol)
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
                if len(self.availablePositions()) == 0:
                    self.isEnd = True
            self.giveReward()
            self.reset()
            self.player1.train_model()
            self.player2.train_model()
            self.player1.reset()
            self.player2.reset()
        return self.player1, self.player2

    def AIUpdateGUI(self, action):
        action = str(action)

        if(action == "(0, 0)"):
            s1["text"] = "X"
        elif(action == "(0, 1)"):
            s2["text"] = "X"
        elif(action == "(0, 2)"):
            s3["text"] = "X"
        elif(action == "(1, 0)"):
            s4["text"] = "X"
        elif(action == "(1, 1)"):
            s5["text"] = "X"
        elif(action == "(1, 2)"):
            s6["text"] = "X"
        elif(action == "(2, 0)"):
            s7["text"] = "X"
        elif(action == "(2, 1)"):
            s8["text"] = "X"
        elif(action == "(2, 2)"):
            s9["text"] = "X"

    def humanTakeTurnGUI(self, square):
        global hasTakenTurn, humanAction

        if(square["text"] == " "):
            square["text"] = "O"
            if(str(square) == ".!button"):
                hasTakenTurn = True
                humanAction = (0, 0)
            elif(str(square) == ".!button2"):
                hasTakenTurn = True
                humanAction = (0, 1)
            elif(str(square) == ".!button3"):
                hasTakenTurn = True
                humanAction = (0, 2)
            elif(str(square) == ".!button4"):
                hasTakenTurn = True
                humanAction = (1, 0)
            elif(str(square) == ".!button5"):
                hasTakenTurn = True
                humanAction = (1, 1)
            elif(str(square) == ".!button6"):
                hasTakenTurn = True
                humanAction = (1, 2)
            elif(str(square) == ".!button7"):
                hasTakenTurn = True
                humanAction = (2, 0)
            elif(str(square) == ".!button8"):
                hasTakenTurn = True
                humanAction = (2, 1)
            elif(str(square) == ".!button9"):
                hasTakenTurn = True
                humanAction = (2, 2)
        else:
            messagebox.showerror("Tic-Tac-Toe", "Square unavailable")

    def highlightBoard(self):
        # Check to see if player "X" won
        if(s1["text"] == "X" and s2["text"] == "X" and s3["text"] == "X"):
            s1.config(bg="gold")
            s2.config(bg="gold")
            s3.config(bg="gold")
        elif(s4["text"] == "X" and s5["text"] == "X" and s6["text"] == "X"):
            s4.config(bg="gold")
            s5.config(bg="gold")
            s6.config(bg="gold")
        elif(s7["text"] == "X" and s8["text"] == "X" and s9["text"] == "X"):
            s7.config(bg="gold")
            s8.config(bg="gold")
            s9.config(bg="gold")
        elif(s1["text"] == "X" and s4["text"] == "X" and s7["text"] == "X"):
            s1.config(bg="gold")
            s4.config(bg="gold")
            s7.config(bg="gold")
        elif(s2["text"] == "X" and s5["text"] == "X" and s8["text"] == "X"):
            s2.config(bg="gold")
            s5.config(bg="gold")
            s8.config(bg="gold")
        elif(s3["text"] == "X" and s6["text"] == "X" and s9["text"] == "X"):
            s3.config(bg="gold")
            s6.config(bg="gold")
            s9.config(bg="gold")
        elif(s1["text"] == "X" and s5["text"] == "X" and s9["text"] == "X"):
            s1.config(bg="gold")
            s5.config(bg="gold")
            s9.config(bg="gold")
        elif(s3["text"] == "X" and s5["text"] == "X" and s7["text"] == "X"):
            s3.config(bg="gold")
            s5.config(bg="gold")
            s7.config(bg="gold")

        # Check to see if player "O" won
        elif(s1["text"] == "O" and s2["text"] == "O" and s3["text"] == "O"):
            s1.config(bg="gold")
            s2.config(bg="gold")
            s3.config(bg="gold")
        elif(s4["text"] == "O" and s5["text"] == "O" and s6["text"] == "O"):
            s4.config(bg="gold")
            s5.config(bg="gold")
            s6.config(bg="gold")
        elif(s7["text"] == "O" and s8["text"] == "O" and s9["text"] == "O"):
            s7.config(bg="gold")
            s8.config(bg="gold")
            s9.config(bg="gold")
        elif(s1["text"] == "O" and s4["text"] == "O" and s7["text"] == "O"):
            s1.config(bg="gold")
            s4.config(bg="gold")
            s7.config(bg="gold")
        elif(s2["text"] == "O" and s5["text"] == "O" and s8["text"] == "O"):
            s2.config(bg="gold")
            s5.config(bg="gold")
            s8.config(bg="gold")
        elif(s3["text"] == "O" and s6["text"] == "O" and s9["text"] == "O"):
            s3.config(bg="gold")
            s6.config(bg="gold")
            s9.config(bg="gold")
        elif(s1["text"] == "O" and s5["text"] == "O" and s9["text"] == "O"):
            s1.config(bg="gold")
            s5.config(bg="gold")
            s9.config(bg="gold")
        elif(s3["text"] == "O" and s5["text"] == "O" and s7["text"] == "O"):
            s3.config(bg="gold")
            s5.config(bg="gold")
            s7.config(bg="gold")

    def showBoard(self):
        global s1, s2, s3, s4, s5, s6, s7, s8, s9
        global turnToggle, count
        turnToggle = True
        count = 0
        s1 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s1))
        s2 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s2))
        s3 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s3))
        s4 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s4))
        s5 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s5))
        s6 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s6))
        s7 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s7))
        s8 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s8))
        s9 = Button(root, text=" ", height=10, width=20,
                    bg="SystemButtonFace", command=lambda: cs.humanTakeTurnGUI(s9))

        # Plot squares to screen
        s1.grid(row=0, column=0)
        s2.grid(row=0, column=1)
        s3.grid(row=0, column=2)
        s4.grid(row=1, column=0)
        s5.grid(row=1, column=1)
        s6.grid(row=1, column=2)
        s7.grid(row=2, column=0)
        s8.grid(row=2, column=1)
        s9.grid(row=2, column=2)

    # playing with human player
    def playWithHuman(self):
        global hasTakenTurn, humanAction
        cs.showBoard()
        while not self.isEnd:
            # Player_1
            positions = self.availablePositions()
            p1_action = self.player1.make_Move(
                positions, self.board, self.playerSymbol)
            # take action and update board state
            self.updateState(p1_action)
            # check board status if it is end
            print(self.board)
            win = self.winner()
            if win is not None:
                if win == 1:
                    cs.highlightBoard()
                    print(self.player1.name, "It's a win!")
                    messagebox.showinfo(
                        "Tic-Tac-Toe", "AI wins!")
                else:
                    print("It's a tie!")
                    messagebox.showinfo("Tic-Tac-Toe", "Tie!")
                self.reset()
                break

            else:
                # Player_2
                print("Waiting for human...")
                while(True):
                    root.update()
                    if(hasTakenTurn == True):
                        self.updateState(humanAction)
                        hasTakenTurn = False
                        humanAction = None
                        break

                # USED ONLY FOR COMMAND LINE VERSION OF GAME
                # positions = self.availablePositions()
                # p2_action = self.player2.make_Move(positions)

                # self.updateState(p2_action)

                win = self.winner()
                if win is not None:
                    if win == -1:
                        cs.highlightBoard()
                        print(self.player2.name, "It's a win!")
                        messagebox.showinfo(
                            "Tic-Tac-Toe", "{} wins!".format(self.player2.name,))
                    else:
                        print("It's a tie!")
                        messagebox.showinfo("Tic-Tac-Toe", "Tie!")
                    self.reset()
                    break
            print()
            print(self.board)


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
        self.model.add(Conv2D(64, (2, 2), activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(Dense(num_actions))
        self.model.compile(loss="MeanSquaredError",
                           optimizer=Adam(learning_rate=self.lr))
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
        self.epsilon = self.epsilon * \
            self.epsilon_decay if self.epsilon >= self.epsilon_min else self.epsilon_min
        self.add_State(current_board)
        self.actions.append(action)

        cs.AIUpdateGUI(action)

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
            targets.append(
                self.rewards[index] + self.decay_gamma * np.amax(self.model.predict(state), axis=1))
            index += 1
        targets = np.asarray(targets)
        index = 0
        for target, state in zip(targets, state_list):
            self.model.fit(state, target)
            if index > 9:
                break
            index += 1
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
            self.rewards[-st] += self.lr * \
                (self.decay_gamma * reward - self.rewards[-st])
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
            else:
                print("Invalid move\n")

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
    #
    # P1 = Player("P1")
    #
    # P2 = Player("P2")
    #
    # cs = State(P1, P2)
    #
    # print("Training the Neural Network...")
    #
    # play1, play2 = cs.train()
    # play1.save_model("player1_v4")
    # play2.save_model("player2_v4")

    # Play against a human
    P1 = Player("Computer Player", exp_rate=0)

    P1.load_model("player1_v4")

    P2 = HumanPlayer("Human")

    cs = State(P1, P2)
    cs.playWithHuman()
