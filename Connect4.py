from tic_tac_toe import Board
from IPython.display import clear_output

from tqdm import tqdm

import pickle, math, logging, random
import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 6, 7

name = 'qlvsran'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{name}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


class Connect4(Board):
    def __init__(self, dimensions):
        super().__init__(dimensions, x_in_a_row=4)
        self.rows, self.cols = dimensions
        # Initialize a 2D list to represent the board state. 0 = empty, 1 = player 1, 2 = player 2
        self.board_state = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.state_dict = dict()

    def possible_moves(self):
        possible_moves_list = []
        for move in super().possible_moves():
            if move[0] in possible_moves_list:
                continue
            else:
                possible_moves_list.append(move[0])
        return possible_moves_list

    def push(self, col):
        self.state_dict = dict()
        for move in super().possible_moves():
            self.state_dict[move[0]] = []
        for move in super().possible_moves():
            self.state_dict[move[0]].append(move[1])
        super().push((col, max(self.state_dict[col])))
        for row in reversed(range(self.rows)):
            if self.board_state[row][col] == 0:
                self.board_state[row][col] = self.turn
                break 

    def copy(self):
        board = Connect4(self.dimensions)
        board.turn = self.turn
        board.board = self.board.copy()
        board.board_state = [row[:] for row in self.board_state]
        return board
    
    def is_full(self):
        return all(self.board_state[0][col] != 0 for col in range(self.cols))



class ConnectFourBoard:
    def __init__(self, p1, p2):
        self.board = Connect4(dimensions=(ROWS, COLS))
        self.board.x_in_a_row = 4
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None

    def displayBoard(self,name):
        # print('-' * 23)
        # print(self.board)
        self.plot_connect4(name)
        # print('-' * 23)

    def plot_connect4(self,name):
        # Access the board_state directly from the Connect4 instance
        board_state = self.board.board_state
        rows, cols = len(board_state), len(board_state[0])
        fig, ax = plt.subplots()
        ax.set_facecolor('blue')
        ax.set_aspect('equal', adjustable='box')

        for x in range(cols):
            for y in range(rows):
                color = 'white' if board_state[y][x] == 0 else ('red' if board_state[y][x] == 1 else 'yellow')
                circle = plt.Circle((x, rows - y - 1), 0.45, color=color, ec='black')
                ax.add_artist(circle)

        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        ax.set_xticks(np.arange(-0.5, cols, 1))
        ax.set_yticks(np.arange(-0.5, rows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which='both', color='black', linestyle='-', linewidth=2)
        if(name==""):
            plt.title("Connect Four")
        else :
            plt.title(f"{name} played the Turn")
        plt.show(block=False)  # Set block=False to ensure non-blocking behavior
        plt.pause(0.5)  # Display the plot for 0.5 seconds
        plt.close() 
    def getBoard(self):
        self.boardHash = str(self.board.board.flatten())
        return self.boardHash

    def getWinner(self):
        return self.board.result()

    def getPositions(self):
        return self.board.possible_moves()

    def getGameOver(self):
        return self.board.result() is not None


    def setReward(self):
        result = self.getWinner()
        if result == 1:
            if hasattr(self.p1, 'setReward'):
                self.p1.setReward(1)
            if hasattr(self.p2, 'setReward'):
                self.p2.setReward(0)
        elif result == 2:
            if hasattr(self.p1, 'setReward'):
                self.p1.setReward(0)
            if hasattr(self.p2, 'setReward'):
                self.p2.setReward(1)
        else:
            # Assuming a draw is the only other option
            if hasattr(self.p1, 'setReward'):
                self.p1.setReward(0.2)
            if hasattr(self.p2, 'setReward'):
                self.p2.setReward(0.2)


    def setMove(self, move):
        self.board.push(move)

    def reset(self):
        self.board = Connect4(dimensions=(ROWS, COLS))
        self.board.x_in_a_row = 4
        self.boardHash = None
        self.isEnd = False

    def evaluate_agents(self,agent1, agent2, games=100):
        win1, win2, draws = 0, 0, 0
        for _ in range(games):
            self.reset()
            while not self.getGameOver():
                positions = self.getPositions()
                if self.board.turn == 1:
                    action = agent1.getMove(positions, self.board)
                else:
                    action = agent2.getMove(positions, self.board)
                self.setMove(action)
            
            result = self.getWinner()
            if result == 1:
                win1 += 1
            elif result == 2:
                win2 += 1
            else:
                draws += 1
        
        return win1/games, win2/games, draws/games
    def train(self, rounds=100, epsilon_decay=0.999995, min_epsilon=0.1,evaluation_interval=1000):
        # Initialize epsilon to the player's current epsilon or 1.0 if not set
        initial_epsilon = getattr(self.p1, 'epsilon', 1.0)
        
        for i in tqdm(range(rounds)):
            # Reset epsilon to its initial value at the start of each round
            if hasattr(self.p1, 'epsilon') and hasattr(self.p2, 'epsilon'):
                self.p1.epsilon = max(min_epsilon, self.p1.epsilon * epsilon_decay)            
            while not self.getGameOver():
                # Player 1's turn
                positions = self.getPositions()
                p1_action = self.p1.getMove(positions, self.board)
                self.setMove(p1_action)
                board_hash = self.getBoard()
                self.p1.setState(board_hash)
                
                win = self.getWinner()
                # print(win)
                if win is not None:
                    # Set rewards and reset for the next game
                    self.setReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                
                # Player 2's turn
                positions = self.getPositions()
                p2_action = self.p2.getMove(positions, self.board)
                self.setMove(p2_action)
                board_hash = self.getBoard()
                self.p2.setState(board_hash)

                win = self.getWinner()
                if win is not None:
                    # Set rewards and reset for the next game
                    self.setReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
            if (i+1) % evaluation_interval == 0:
                win_rate_p1, win_rate_p2, draw_rate = self.evaluate_agents(self.p1, self.p2, games=1000)
                print(f"Round {i+1}: Win rate P1: {win_rate_p1:.2f}, Win rate P2: {win_rate_p2:.2f}, Draw rate: {draw_rate:.2f}")
                if(win_rate_p1>=.90):
                    break
                if(win_rate_p1>=.7):
                    min_epsilon=0.01


            


    def play(self,vis):
        if vis:
            self.displayBoard("")

        while not self.getGameOver():
            # PLAYER 1
            positions = self.getPositions()
            move = self.p1.getMove(positions, self.board)
            self.setMove(move)
            clear_output()
            if vis:
                self.displayBoard(self.p1.name)

            winStatus = self.getWinner()
            if winStatus is not None:
                if winStatus == 1:
                    if vis:
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, f"{self.p1.name} wins!", fontsize=30, ha='center', va='center')
                        plt.axis('off')
                        plt.show()
                    logger.warning(f"{self.p1.name}")
                    self.reset()
                    return self.p1.name  # Return the name of player 1 as the winner

                else:
                    if vis:
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, "Tie Game", fontsize=30, ha='center', va='center')
                        plt.axis('off')
                        plt.show()
                    logger.warning("tie")
                    self.reset()
                    return "tie"  # Return "tie" to indicate the game ended in a tie
            else:
                positions = self.getPositions()
                move = self.p2.getMove(positions, self.board)
                self.setMove(move)
                clear_output()
                if vis:
                    self.displayBoard(self.p2.name)

                winStatus = self.getWinner()
                if winStatus is not None:
                    if winStatus == 2:
                        if vis:
                            fig, ax = plt.subplots()
                            ax.text(0.5, 0.5, f"{self.p2.name} wins!", fontsize=30, ha='center', va='center')
                            plt.axis('off')
                            plt.show()
                        logger.warning(f"{self.p2.name}")
                        self.reset()
                        return self.p2.name  # Return the name of player 2 as the winner

                    else:
                        if vis:
                            fig, ax = plt.subplots()
                            ax.text(0.5, 0.5, "Tie Game", fontsize=30, ha='center', va='center')
                            plt.axis('off')
                            plt.show()
                        logger.warning("tie")
                        self.reset()
                        return "tie"

class RandomPlayer:
    def __init__(self, name='random'):
        self.name = name

    def getMove(self, positions, board):
        randomMove = random.choice(board.possible_moves())
        return randomMove

class DefaultPlayer:
    def __init__(self, name='default'):
        self.name = name

    def getMove(self, positions, board):
        # First, try to find a winning move for the player
        for move in positions:
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.result() == board.turn:  # Check if this move wins the game
                return move

        # If no winning move found, try to block opponent's winning move
        opponent = 1 if board.turn == 2 else 2  # Determine the opponent's id
        for move in positions:
            temp_board = board.copy()
            temp_board.turn = opponent  # Temporarily set the board's turn to the opponent's
            temp_board.push(move)
            if temp_board.result() == opponent:  # Check if the opponent would win with this move
                return move

        # As a fallback, choose a random move from the available positions
        return random.choice(positions)
    
class MinimaxPlayer:
    def __init__(self, name='minimax', use_pruning=True):
        self.name = name
        self.use_pruning = use_pruning  # Flag for alpha-beta pruning
        self.move_count = 0  # Tracks the number of moves explored

    def minimax(self, board, depth, alpha, beta, maximizing):
        self.move_count += 1  # Increment move exploration counter

        # Base case: Check for terminal state or depth limit
        if depth == 0 or board.has_won(1) or board.has_won(2):
            return self.evaluate(board), None

        if maximizing:
            return self.maximize(board, depth, alpha, beta)
        else:
            return self.minimize(board, depth, alpha, beta)

    def maximize(self, board, depth, alpha, beta):
        maxEval = float('-inf')
        bestMove = None
        for move in board.possible_moves():
            tempBoard = board.copy()
            tempBoard.push(move)
            eval, _ = self.minimax(tempBoard, depth - 1, alpha, beta, False)
            if eval > maxEval:
                maxEval, bestMove = eval, move
            alpha = max(alpha, eval) if self.use_pruning else alpha
            if self.use_pruning and beta <= alpha:
                break
        return maxEval, bestMove

    def minimize(self, board, depth, alpha, beta):
        minEval = float('inf')
        bestMove = None
        for move in board.possible_moves():
            tempBoard = board.copy()
            tempBoard.push(move)
            eval, _ = self.minimax(tempBoard, depth - 1, alpha, beta, True)
            if eval < minEval:
                minEval, bestMove = eval, move
            beta = min(beta, eval) if self.use_pruning else beta
            if self.use_pruning and alpha >= beta:
                break
        return minEval, bestMove

    def evaluate(self, board):
        if board.has_won(1):
            return 1
        elif board.has_won(2):
            return -1
        return 0

    def getMove(self, positions, board):
        self.move_count = 0  # Reset move counter
        _, move = self.minimax(board, 5, float('-inf'), float('inf'), board.turn == 1)
        # Optionally print the move count for debugging
        # print(f"Moves explored: {self.move_count}")
        return move

class QLearningPlayer:
    def __init__(self, name='q-agent', alpha=0.6, epsilon=0.3, gamma=0.9):
        self.name = name
        self.states = []  # List of states visited
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.Q_table = {}  # Initialize Q-table as an empty dictionary

    def getBoard(self, board):
        # Convert board state to a hashable type for use as a Q-table key
        return str(board.board.flatten())

    def getMove(self, positions, current_board):
        # Decide on action based on epsilon-greedy strategy
        if np.random.uniform(0, 1) <= self.epsilon:
            # Exploration: choose a random action
            return random.choice(positions)
        else:
            # Exploitation: choose the best action from Q-table
            return self.chooseBestAction(positions, current_board)

    def chooseBestAction(self, positions, current_board):
        maxValue = -float('inf')
        bestAction = None
        for move in positions:
            _nextBoard = current_board.copy()
            _nextBoard.x_in_a_row = 4  # This seems like game-specific logic that might be better encapsulated elsewhere
            _nextBoard.push(move)
            _nextBoardState = self.getBoard(_nextBoard)
            value = self.Q_table.get(_nextBoardState, 0)
            if value > maxValue:
                maxValue, bestAction = value, move

        # Ensure there is always an action to return
        return bestAction if bestAction is not None else random.choice(positions)

    def setState(self, state):
        # Append state to the history of visited states
        self.states.append(state)

    def setReward(self, reward):
        # Update Q-values for all visited states
        for st in reversed(self.states):
            self.Q_table[st] = self.Q_table.get(st, 0) + self.alpha * (reward + self.gamma * self.Q_table.get(st, 0) - self.Q_table[st])
            reward = self.Q_table[st]

    def reset(self):
        # Clear the history of visited states
        self.states = []

    def savePolicy(self):
        # Save the Q-table to a file
        with open('policy_' + self.name, 'wb') as fw:
            pickle.dump(self.Q_table, fw)

    def loadPolicy(self, file):
        # Load the Q-table from a file
        with open(file, 'rb') as fr:
            self.Q_table = pickle.load(fr)

class MinimaxPlayerWithoutABP:
    def __init__(self, name='minimax_withoutABP'):
        self.name = name
        self.evaluated_moves_count = 0 

    def minimax(self, board, maximizing):
        # Increment move count here if you're implementing move counting
        self.evaluated_moves_count += 1
        if self.evaluated_moves_count%10000 ==0:
            print(self.evaluated_moves_count)
        # Check for terminal states: win, lose, draw (assuming result() properly checks these conditions)
        game_result = board.result()
        if game_result is not None:
            if game_result == 1:  # Maximizing player (X) wins
                return 1, None
            elif game_result == 2:  # Minimizing player (O) wins
                return -1, None
            elif game_result == 0:  # Draw
                return 0, None

        if maximizing:
            maxEval = float('-inf')
            bestMove = None
            for move in board.possible_moves():
                tempBoard = board.copy()
                tempBoard.x_in_a_row = 4
                tempBoard.push(move)
                eval = self.minimax(tempBoard, not maximizing)[0]
                if eval > maxEval:
                    maxEval = eval
                    bestMove = move
            return maxEval, bestMove
        else:
            minEval = float('inf')
            bestMove = None
            for move in board.possible_moves():
                tempBoard = board.copy()
                tempBoard.x_in_a_row = 4
                tempBoard.push(move)
                eval = self.minimax(tempBoard, not maximizing)[0]
                if eval < minEval:
                    minEval = eval
                    bestMove = move
            return minEval, bestMove

    def getMove(self, positions, board):
        _, move = self.minimax(board, True if board.turn == 1 else False)
        print(f"Evaluated {self.evaluated_moves_count} moves for this decision.")
        return move
    
class HumanPlayer:
    def __init__(self, name='human'):
        self.name = name

    def getMove(self, positions, board):
        while True:
            col = int(input('Choose column: '))
            return col