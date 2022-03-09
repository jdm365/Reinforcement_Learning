Base attempt to create general MuZero algorithm.

** Initial Sketch of Plan **

1) Take current (and possibly past) environment state(s) and downsample them to some "hidden state size" to create root node state.
   This downsampling is done by a network referred to as a "representation function".
2) Get estimates for move probabilities using policy network and estimate of future rewards from value network.
3) Using MCTS algorithm, expand root node and select child using UCB. If child is already expanded continue UCB path down tree until leaf node is found.
   Note on expansion: Because the environment is assumed to be unknown, expanded states and predicted rewards must be estimated by some function. The output
                      of this function will be a state of size "hidden state size". This function will be approximated by another neural network and is termed 
                      the "dynamics function". 
                      Additionally because the future states are not known with certainty, we assume that all possible actions will be available at future states.
                      Actions are only masked at the root nodes. Because simulated "illegal" actions never actually get taken, they will never be in the replay 
                      memory and the loss function will be updated without these "illegal" actions. So long as a policy is found that is greater than random
                      actions, the agent will learn not to select these illegal actions in its simulations.
4) At a leaf node, evaluate state using policy and value networks and expand using dynamics network.
5) After expansion backpropogate the visit counts and values predicted by the value network.
6) Repeat MCTS (Roughly 3-5) for N simulations, then choose action with the highest visit count (plus some noise for exploration controlled by temperature 
   parameter).
7) Store state, probs, rewards in replay buffer.
8) Repeat 1-7 until end of episode.
9) Learn using loss function from paper and sampling from replay buffer (buffer is of fixed length to eventually deprecate transitions taken by worse policy).
10) Repeat 1-9 until convergence.
