import argparse
import tic_tac_toe
from tic_tac_toe import *

# Set up command line arguments
parser = argparse.ArgumentParser(description='Train or Play TicTacToe.')
parser.add_argument('--train', action='store_true', help='If set, train the model. Otherwise, play the game.')

# Parse arguments
args = parser.parse_args()

if args.train:
    # TRAINING BLOCK
    p1 = QLearningPlayer('Q1_final_tictactoe', epsilon=1)
    p2 = QLearningPlayer('demo-Q2', epsilon=1)

    game = TicTacToe(p1, p2)
    game.train(200000)

    p1.savePolicy()
    p2.savePolicy()

else:
    # PLAYING BLOCK
    p1 = QLearningPlayer('Q1', epsilon=0)
    p1.loadPolicy('policy_Q1_tictactoe_v2')
    # p1 = DefaultPlayer()                         # Uncomment Qlearning and choose agent of your liking 
    # p1= RandomPlayer()
    # p1 = MinimaxPlayer(use_alpha_beta=True)
    # p1 = HumanPlayer('Player 1')

    p2 = DefaultPlayer()
    # p2 = MinimaxPlayer(use_alpha_beta=True)      # Uncomment Qlearning and choose agent of your liking
    # p2 = RandomPlayer()
    # p2 = HumanPlayer('Player 1')

    game = TicTacToe(p1, p2)
    # winner = game.play(True)
    results = {
        p1.name: 0,
        p2.name: 0,
        'tie': 0
    }

    # Run the game 10 times (it seems like it was supposed to be 10 but only set for 1 loop)
    for _ in range(10):  # Adjusted to run 10 times
        winner = game.play(False)
        
        if winner == game.p1.name:
            results[p1.name] += 1
        elif winner == game.p2.name:
            results[p2.name] += 1
        else:
            results['tie'] += 1

    print("Results after 10 games:")  # Adjusted to reflect actual number of games
    for key, value in results.items():
        print(f"{key}: {value} wins")

