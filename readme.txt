
# Tic Tac Toe and Connect 4 AI Agent Play and Training Guide

This guide provides instructions for training AI agents and playing against them in the games of Tic Tac Toe and Connect 4.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Tic Tac Toe (main.py)

### To Train the Agents:
Run the script with the `--train` flag to train the Q-learning agents. Training will involve the agents playing against each other for a number of games specified in the code.


python main.py --train



The training progress will be automatically saved in policy files.

### To Play Against an Agent:
Run the script without any flags. By default, you will play against the pre-trained Q-learning agent.



python main.py



To change the agent you play against, uncomment the desired player lines in the PLAYING BLOCK of `main.py`.



---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Connect 4 (main2.py)

### To Train the Agents:
Similarly, to train agents for Connect 4, run the script with the `--train` flag:



python main2.py --train



Training progress and policies are saved similarly to Tic Tac Toe.

### To Play Against an Agent:
To play the game, run without any flags:



python main2.py



You can choose which agent to play against by uncommenting the appropriate line in the PLAYING MODE section of `main2.py`.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Agent Selection
You can choose between several agents:

- `QLearningPlayer`: An AI trained via Q-learning.
- `DefaultPlayer`: A simple AI with default behavior.
- `MinimaxPlayer`: An AI that uses the Minimax algorithm.
- `RandomPlayer`: An AI that makes random moves.
- `HumanPlayer`: Allows a human player to play (input required).

To switch agents, comment out the current player and uncomment the agent you wish to play against in the respective Python file.

## Results
After playing, the results will be printed out, displaying the number of wins for each agent and ties.

Note: For a fair comparison, ensure that the agents are appropriately trained by running the training mode first before playing against them.

Enjoy testing your skills against these AI agents!

