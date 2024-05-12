from tictactoe import Board
from IPython.display import clear_output

from tqdm import tqdm

import pickle, math, logging, random
import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 3, 3

name = 'qlvsran'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{name}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

class TicTacToe:
    def __init__(self, p1, p2):
        self.board = Board(dimensions=(ROWS, COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None

    def getBoard(self):
        self.boardHash = str(self.board.board.flatten())
        return self.boardHash

    def getWinner(self):
        return self.board.result()

    def getPositions(self):
        return self.board.possible_moves()

    def setReward(self):
        result = self.getWinner()
        if result == 1:
            self.p1.setReward(1)  # Player 1 wins
            self.p2.setReward(0)  # Player 2 loses
        elif result == 2:
            self.p1.setReward(0)  # Player 1 loses
            self.p2.setReward(1)  # Player 2 wins
        else:
            # In case of a draw or the game is still ongoing
            self.p1.setReward(0.2)  # Draw reward for Player 1
            self.p2.setReward(0.2)

    def reset(self):
        self.board = Board()
        self.boardHash = None
        self.isEnd = False

    def setMove(self, action):
        self.board.push(action)

    def train(self, rounds=100, epsilon_decay=0.99995, min_epsilon=0.01):
        initial_epsilon = self.p1.epsilon if hasattr(self.p1, 'epsilon') else 1.0
        for i in tqdm(range(rounds)):
            if hasattr(self.p1, 'epsilon') and hasattr(self.p2, 'epsilon'):
                # Adjust epsilon for both players before the round starts
                self.p1.epsilon = max(min_epsilon, self.p1.epsilon * epsilon_decay)
                self.p2.epsilon = max(min_epsilon, self.p2.epsilon * epsilon_decay)

            while not self.isEnd:
                positions = self.getPositions()
                p1_action = self.p1.getMove(positions, self.board)
                self.setMove(p1_action)
                board_hash = self.getBoard()
                self.p1.setState(board_hash)

                win = self.getWinner()
                if win is not None:
                    self.setReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    positions = self.getPositions()
                    p2_action = self.p2.getMove(positions, self.board)
                    self.setMove(p2_action)
                    board_hash = self.getBoard()
                    self.p2.setState(board_hash)

                    win = self.getWinner()
                    if win is not None:
                        self.setReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    def play(self,check):
        while not self.isEnd:
            positions = self.getPositions()
            p1_action = self.p1.getMove(positions, self.board)
            self.setMove(p1_action)
            clear_output()
            if check:
                self.showBoard(self.p1.name)

            win = self.getWinner()
            if win is not None:
                if win == 1:
                    if check:
                        fig, ax = plt.subplots(figsize=(7, 7))
                        ax.text(0.5, 0.5, f"{self.p1.name} wins!", fontsize=30, ha='center', va='center')
                        plt.axis('off')
                        plt.show()
                    logger.warning(f"{self.p1.name}")
                    self.reset()
                    return self.p1.name  # Return player 1's name as the winner

                else:  # In case of a tie
                    if check:
                        fig, ax = plt.subplots(figsize=(7, 7))
                        ax.text(0.5, 0.5, "Tie Game", fontsize=30, ha='center', va='center')
                        plt.axis('off')
                        plt.show()
                    logger.warning("tie")
                    self.reset()
                    return "tie"  # Return "tie" to indicate the game ended in a tie

            else:
                positions = self.getPositions()
                p2_action = self.p2.getMove(positions, self.board)
                self.setMove(p2_action)
                clear_output()
                if check:
                    self.showBoard(self.p2.name)

                win = self.getWinner()
                if win is not None:
                    if win == 2:
                        if check:
                            fig, ax = plt.subplots(figsize=(7, 7))
                            ax.text(0.5, 0.5, f"{self.p2.name} wins!", fontsize=30, ha='center', va='center')
                            plt.axis('off')
                            plt.show()
                        logger.warning(f"{self.p2.name}")
                        self.reset()
                        return self.p2.name  # Return player 2's name as the winner

                    else:  # In case of a tie
                        if check:
                            fig, ax = plt.subplots(figsize=(7, 7))
                            ax.text(0.5, 0.5, "Tie Game", fontsize=30, ha='center', va='center')
                            plt.axis('off')
                            plt.show()
                        logger.warning("tie")
                        self.reset()
                        return "tie" 

    def plotBoard(self,name):

        # Define the symbols for players
        symbols = {1: 'X', 2: 'O'}

        # Create a new figure with defined figure size and square subplots
        fig, ax = plt.subplots(figsize=(7, 7))

        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')

        # Plot the board with player symbols
        for i in range(3):
            for j in range(3):
                # Add player symbol to the corresponding cell
                # Note: the origin is at the top-left corner for the Tic-Tac-Toe board
                player = self.board.get_mark_at_position((i, j))
                symbol = symbols.get(player, ' ')
                color = 'red' if symbol == 'O' else 'black'
                ax.text(j + 0.5, 2.5 - i, symbol, ha='center', va='center', fontsize=50, color=color, family='monospace')
                # ax.text(j + 0.5, 2.5 - i, symbol, ha='center', va='center', fontsize=50, family='monospace')

        # Draw grid lines
        for x in range(1, 3):
            ax.axvline(x=x, color='black', linestyle='-', linewidth=2)
        for y in range(1, 3):
            ax.axhline(y=y, color='black', linestyle='-', linewidth=2)

        # Set limits to just beyond the grid
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)

        # Remove axis ticks and labels
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.title(f"{name} played the Turn")
        # plt.show()
        plt.show(block=False)  # Set block=False to ensure non-blocking behavior
        plt.pause(0.8)  # Display the plot for 0.5 seconds
        plt.close() 

    
    def showBoard(self,name):
        self.plotBoard(name)

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def getMove(positions, board):
        # This loop will repeatedly ask for valid row and column numbers until a valid move is entered
        while True:
            row = int(input('Enter row: '))
            col = int(input('Enter col: '))

            move = (col, row)
            if move in positions:
                return move
            else:
                print("Invalid move, please try again.")


class RandomPlayer:
    def __init__(self, name='random'):
        self.name = name

    def getMove(self, positions, board):
        # Selects a random move from the list of possible moves
        return random.choice(positions)


class DefaultPlayer:
    def __init__(self, name='default'):
        self.name = name

    def getWinningMove(self, board, turn):
        for move in board.possible_moves():
            temp_board = board.copy()
            temp_board.push(tuple(move))
            if temp_board.result() == turn:
                return move
        return None

    def blockWinningMove(self, board, turn):
        opponent_turn = turn % 2 + 1
        for move in board.possible_moves():
            temp_board = board.copy()
            temp_board.set_mark(move.tolist(), opponent_turn)
            if temp_board.result() == opponent_turn:
                return move
        return None

    def getMove(self, positions, board):
        winMove = self.getWinningMove(board, board.turn)
        if winMove is not None:
            return winMove

        blockMove = self.blockWinningMove(board, board.turn)
        if blockMove is not None:
            return blockMove
        randomMove = random.choice(board.possible_moves())
        return randomMove


class MinimaxPlayer:
    def __init__(self, name='minimax', use_alpha_beta=True):
        self.name = name
        self.use_alpha_beta = use_alpha_beta
        self.moves_explored = 0

    def minimax(self, gameBoard, alpha, beta, maximizingPlayer):
        gameStatus = gameBoard.result()

        # Simplified scoring check
        if gameStatus in [1, 2, 0]:
            return {1: (1, None), 2: (-1, None), 0: (0, None)}[gameStatus]

        if maximizingPlayer:
            return self._maximize(gameBoard, alpha, beta)
        else:
            return self._minimize(gameBoard, alpha, beta)

    def _maximize(self, gameBoard, alpha, beta):
        bestScore = float('-inf')
        bestAction = None
        for action in gameBoard.possible_moves():
            self.moves_explored += 1
            cloneBoard = gameBoard.copy()
            cloneBoard.push(action)
            score, _ = self.minimax(cloneBoard, alpha, beta, False)
            if score > bestScore:
                bestScore, bestAction = score, action
            if self.use_alpha_beta:
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        return bestScore, bestAction

    def _minimize(self, gameBoard, alpha, beta):
        worstScore = float('inf')
        worstAction = None
        for action in gameBoard.possible_moves():
            self.moves_explored += 1
            cloneBoard = gameBoard.copy()
            cloneBoard.push(action)
            score, _ = self.minimax(cloneBoard, alpha, beta, True)
            if score < worstScore:
                worstScore, worstAction = score, action
            if self.use_alpha_beta:
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return worstScore, worstAction

    def getMove(self, positions, gameBoard):
        self.moves_explored = 0
        _, move = self.minimax(gameBoard, -math.inf, math.inf, gameBoard.turn == 1)
        # Optionally, print the number of moves explored
        # print(f"Moves Explored: {self.moves_explored}")
        return move

class QLearningPlayer:
    def __init__(self, name='q-agent', alpha=0.6, epsilon=0.3, gamma=0.95):
        self.name = name
        self.states = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q_table = {}

    def getHash(self, board):
        # Simplify the hash generation process
        return str(board.board.flatten())

    def getMove(self, positions, current_board):
        # Decide between exploration and exploitation
        if np.random.uniform(0, 1) <= self.epsilon:
            # Exploration: Random move
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            # Exploitation: Choose the best move based on Q values
            action = self._choose_best_action(positions, current_board)
        return action

    def _choose_best_action(self, positions, current_board):
        maxValue = -999
        for p in positions:
            nextBoard = current_board.copy()
            nextBoard.push(tuple(p))
            nextBoardState = self.getHash(nextBoard)
            value = self.Q_table.get(nextBoardState, 0)  # Use a default value of 0
            if value > maxValue:
                maxValue = value
                action = p
        return action

    def setState(self, state):
        # Append state to states list
        self.states.append(state)

    def setReward(self, reward):
        # Traverse the states in reverse to update Q values
        for st in reversed(self.states):
            if st not in self.Q_table:
                self.Q_table[st] = 0
            self.Q_table[st] += self.alpha * (reward + self.gamma * self.Q_table[st] - self.Q_table[st])
            reward = self.Q_table[st]

    def reset(self):
        # Reset the states list
        self.states = []

    def savePolicy(self):
        # Save the Q-table to a file
        with open('policy_' + str(self.name), 'wb') as fw:
            pickle.dump(self.Q_table, fw)

    def loadPolicy(self, file):
        # Load a Q-table from a file
        with open(file, 'rb') as fr:
            self.Q_table = pickle.load(fr)