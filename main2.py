import argparse
from Connect4 import QLearningPlayer, ConnectFourBoard, DefaultPlayer  # Assuming these are the correct imports
import Connect4
from Connect4 import *


Connect4.ROWS = 6
Connect4.COLS = 6

parser = argparse.ArgumentParser(description="Train or play a Connect Four AI model.")
parser.add_argument("--train", action="store_true", help="Enable training mode. Without this flag, the game will run in playing mode.")

args = parser.parse_args()

if args.train:
    # TRAINING MODE
    p1 = QLearningPlayer('Q1_Connect4_v1-6x6', epsilon=1)
    p2 = QLearningPlayer('Q2_Connect4_v1', epsilon=1)

    game = ConnectFourBoard(p1, p2)
    game.train(1000000)

    p1.savePolicy()
    # p2.savePolicy()
else:
    # PLAYING MODE
    p1 = QLearningPlayer('Q1', epsilon=0)
    p1.loadPolicy('policy_Q1-6x6_v2')
    # p1= RandomPlayer()
    # p1 = DefaultPlayer()
    # p1 = MinimaxPlayer(use_pruning=True)   # Uncomment Qlearning and choose agent of your liking 
    # p1 = HumanPlayer()

    p2= RandomPlayer()
    # p2 = DefaultPlayer()
    # p2 = MinimaxPlayer(use_pruning=True)   # Uncomment  and choose agent of your liking
    # p2 = HumanPlayer()

    game = ConnectFourBoard(p1, p2)
    # game.play(True)
    results = {
        p1.name: 0,  
        p2.name: 0,
        'tie': 0
    }

    # Run the game 10 times
    for _ in range(100):
          # Adjusted to run 10 times
        winner = game.play(False)
        
        if winner == game.p1.name:
            results[p1.name] += 1
        elif winner == game.p2.name:
            results[p2.name] += 1
        else:
            results['tie'] += 1


    # Print the results
    print("Results after 10 games:")
    for key, value in results.items():
        print(f"{key}: {value} wins")