# SquidGame-RL
This Repository attempts to construct the game environment, states, agents, and reward to train and deploy a reinforcement learning model, based on the notion of Squid Game's Red Light Green Light version. The game is practically the same the statues game , except that the curator, as seen in the movie, eliminates the individual who moves after the red light signal
Green Light, Red Light (RLGL), often known as the statues game, is a well-known game played
all over the globe with rules ranging from basic to complex. Green Light , Red Light is a statue
variant. The game's title alludes to the colors of a traffic light. The often called curator stands at
one end of the field, while the rest of the players stand at the other. The curator turns away
from the others and shouts, "Green Light!" The others then flee as quickly as they can towards
the curator. The curator then screams out "Red Light!" at any time and turn to face the others,
forcing the others to freeze in place. If someone fails to stop, they are out and must restart from
the beginning. The rules, even though with simple variations are:
1. Begin with everyone on the starting line.
2. When the curator shouts 'Green Light,' everyone rushes to the finish line.
3. When the curator says 'Red Light,' everyone must come to a halt.
  ●  If players are still advancing when the 'Red Light' signal is given, they must return
to the starting line.
4. Begin a new round when everyone has crossed the finish line or when the majority of
players have crossed the finish line.
The project attempted to construct the game environment, states, agents, and reward to train
and deploy a reinforcement learning model, based on the notion of Squid Game's Red Light
Green Light version. The game is practically the same as the regulation stated above, except that
the curator, as seen in the movie, eliminates the individual who moves after the red light signal.
# Game Scene
The game's scenario operates in accordance with the guidelines outlined above. The game
scene has a simple curator, and it currently has three states: red, green, and white rectangles.
The states substitute a basic state of white portrayed as a fuzzy logic for the Big Dolls feature of
rotating around. Despite the claim that the movie offers multiplayer, the game only has a single
player, who is represented by a yellowish-green rectangle. The finish line is drawn on the top of
the glass, just in front of the doll. The game will terminate when you cross the finish line.
# Design
The Curator, finish line, and player all have minor features that aid in the game's logic. This are
the features included in each one of the classes that are important
1. The Big Doll's Curator: It is divided into three states, which are represented by the colors
red, green, and white. The Red indicates a halt, the Green indicates a continuation, and
the White indicates the time to turn.
2. The Finish Line: A halt line is a straight line that represents or displays the finish of the
game or the player's or agent's reaching point.
3. The Player: The player, who will also serve as the agent in the reinforcement learning
environment, is a large yellowish-green rectangle with a set velocity, accelerator, and
beginning location that will travel progressively towards the finish line.
# Agent
Creating the game environment, accompanying the states, agents, and rewards to train and
deploy a reinforcement learning model based on the Red Light Green Light version of Squid
Game has been very difficult. Designing the game environment using pygame and then relating
a certain model to attain a certain learning curve has been ambiguous, if not difficult. The
pygame environment was created in accordance with the design subsection mentioned above.
The player, also known as the agent for the RL scheme, has a move function that only includes
the moving forward or up capability. The forward key has been included into the pressed key
function of pygame. The agent's states are move and do nothing (stop/pause). In this scenario,
the game states will be active, loss, or win.
The gaming environment makes use of game states rather than a score and reward system.
Despite the fact that we included a reward system, there is no continuous scoring that climbs as
a player approaches the finish line; instead, a player is awarded simply for being alive.
# Model Architecture
So we've got the input and a mechanism to use the model's output to play the game, so let's
have a look at the model architecture. We employ three Convolution layers before flattening
them into dense layers and an output layer. The model we used to train the agent with a
q-learning model of a sequential feedforward convolutional neural network with dilutions of
32,64,64. At each gate, we employed a RELU activation function. Finally, to optimize the
learning rate, we employed an Adam optimizer.The output layers are made up of two neurons, one for each highest anticipated reward for
each action. Then we select the action with the greatest potential payoff (Q-value). The build
model is listed in the annex.

# Future Work
There are features left over from the game's design for the goal of developing a better output.
For example, the gaming concept must be multiplayer , to begin with, therefore future
development areas will revolve around designing a game with numerous players and thus
agents. The sophistication of multiplayer should be included in the gaming. In terms of the
overall design of the game experience, among other things, there are a few gaps here and
there; The curator's face can be replaced with a huge doll's face, and the player can be built as a
player with gif motion capacity but limited motion grants. Regarding the models utilized, there
will be a separate feed forward network for multiplayers, so developing a multiple agent will be
a difficult but necessary future road.

