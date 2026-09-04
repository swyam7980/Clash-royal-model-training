# ClashRoyaleRL
This project contains code that trains several neural networks, with the end goal of defeating the training opponent in the real-time strategy game Clash Royale. The main reinforcement learning technique implemented in this project is [Proximal-Policy-Optimization](https://arxiv.org/abs/1707.06347) combined with [multi-agent reinforcement learning](https://arxiv.org/abs/1911.10635) and also makes use of both [state abstraction](https://ala2021.vub.ac.be/papers/ALA2021_paper_50.pdf) and [action-space factoring](https://arxiv.org/abs/1705.07269).


# Methodology

## Overview
In this project, there are 3 agents that work together to interact with the environment. One chooses the cards, another chooses 1 of 48 tiles, and the last one chooses 1 of 9 tiles. Each of these 3 agents are comprised of an actor and critic and is trained using PPO.

## Inputs
At each timestep t, the current state of the game will be broken into 9 parts:
- An image of the top part of the field downscaled to 51x33
- An image of the bottom part of the field downscaled to 51x32
- 2 images, the left and right bridges respectively, both downscaled to 4x6
- An estimate of the current amount of elixir the player has (provided by another neural network).
- 4 one-hot encoded vectors, each corresponding to a card currently in the players hand.

these 9 parts are encoded by an autoencoder, which returns a single vector, which will then be given to each of the 3 agents.

## Actions
At every timestep t, each of the 3 agent's actors will choose an action, these individual actions will be combined and a single action will be excecuted in the environment.

### Cards
The agent which chooses the cards will return a softmax distribution, each probability cooresponding to a card currently available to the player.

### Tiles
In Clash Royale there are hundreds of locations in which a card can be placed, in order to reduce the amount of training time needed, this project splits the location in which the chosen card will be placed, into 2 parts, origin and shell.

#### Origin
In order for a tile to be considered an origin tile it must meet the following requirements:

1. The tile is not on the edge of the grid
2. The tile is not next to another origin tile

#### Shell 
In order for a tile to be considered a shell tile it must be directly touching the side or corner of an origin tile

2 examples of the tile types are shown below, the origin tiles are marked in red, shell tiles are marked in white, other tiles are black and are not included in the action space.

![3x3im](https://user-images.githubusercontent.com/107654508/189499330-8d94b262-8a3e-4c7d-a4df-eb41675d40da.png)
![imageedit_24_3998630696_320x320](https://user-images.githubusercontent.com/107654508/189499669-e552f3ec-446e-4f0c-afa7-da29b7b30272.png)

## Reward function
The agent gets the following rewards:
- Step reward: `+1`
- Win reward: `+300`
- Loss reward: `-200`
### Crowns
The rewards for crowns lost and crowns won follow the function below:

`(4.9*log(4.8*(number of crowns player has won) + 0.75) + 1.4) - (4.9*log(4.8*(number of crowns player has lost) + 0.75) + 1.4)`


This function gives the agents a larger reward the bigger the difference between the number of crowns the player has won and lost is, and results in a maximum reward of `+15` and a minimum reward of `-15`.
 
## Model architecture
Overview:

![Clash Royale Bot diagram 2 drawio](https://user-images.githubusercontent.com/107654508/189508061-ea59d39d-d6f3-45d8-a7a7-10de33bc8e9e.png)

State encoder:

![State autoencoder diagram drawio (2)](https://user-images.githubusercontent.com/107654508/189507834-3ab31ae1-5173-40e3-8e6b-2b0abd24fe07.png)



## Installation

This project runs on Windows with Python 3.9. Python 3.12 is not compatible with
the pinned TensorFlow and Keras versions used by this project. Python 3.9 can be
installed alongside another Python version.

From PowerShell, clone the repository and create a virtual environment:

```powershell
cd D:\
git clone https://github.com/Jaso1024/Real-Time-Strategy-RL-Clash-Royale.git
cd Real-Time-Strategy-RL-Clash-Royale
git checkout main
py -3.9 -m venv venv
.\venv\Scripts\Activate.ps1
python --version   # confirm it says Python 3.9.13
```

If PowerShell blocks activation, allow scripts for the current PowerShell
process and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Install all required packages together while the virtual environment is active:

```powershell
pip install numpy==1.23.2 pandas==1.4.3 tensorflow==2.9.1 keras==2.9.0 opencv-python==4.6.0.66 Pillow==9.2.0 keyboard==0.13.5 pywin32==304 pyautogui==0.9.53 absl-py==1.2.0 pyscreeze==0.1.28 pynput
```

The package names correspond to the imports used by the project: `opencv-python`
provides `cv2`, and `Pillow` provides `PIL`.

# Setup
Before running the program, certain modifications must be made to the configuration of the BlueStacks App Player and game

## BlueStacks

1. Install BlueStacks 5 from [bluestacks.com](https://www.bluestacks.com/).
   BlueStacks 10 is not supported by this program.
2. Open the Play Store in BlueStacks and install Clash Royale.
3. Collapse the right sidebar by clicking the two arrows in the upper-right
   corner.
4. Equip the deck shown below for the initial run.
5. Keep the BlueStacks window fully visible, large, and in a consistent
   position. Do not place other windows over it because the bot reads screen
   captures.
6. In `CRHandler.py`, update the window title in this line to exactly match
   the title of your BlueStacks window if it is not `default`:

   ```python
   window = win32gui.FindWindow(None, "default")
   ```

The bot selects cards with the BlueStacks keyboard bindings `1`, `2`, `3`, and
`4`. Make sure those keys select the four card slots in Clash Royale.

Deck: Make sure that the deck being used is the same as the one below

![image](https://user-images.githubusercontent.com/107654508/189466078-d9dd5956-696c-4fc8-8bd7-32d270113b9d.png)


## Running the agent

### Record a deck

With the virtual environment active and BlueStacks open, start in the project
directory:

```powershell
cd D:\Real-Time-Strategy-RL-Clash-Royale
python -m tools.deck_manager capture --name my_deck
```

Cycle your eight cards through hand slot 1 while prompted. Activate the saved
deck before recording matches:

```powershell
python -m tools.deck_manager activate --name my_deck
```

Record several real matches manually. Press `Ctrl+C` after each match so the
recorder saves it:

```powershell
python -m tools.recorder
```

### Train and run a model

Train a behavioral-cloning model from the recorded matches:

```powershell
python -m tools.behavioral_clone --name my_deck_v1
```

Watch the model play or continue training it:

```powershell
python run.py --model my_deck_v1
```

The complete workflow is:

```powershell
python -m tools.deck_manager capture --name my_deck
python -m tools.deck_manager activate --name my_deck
python -m tools.recorder
python -m tools.behavioral_clone --name my_deck_v1
python run.py --model my_deck_v1
```

Keep BlueStacks open and unobstructed while recording or running the model.







