Basic Alphazero implementation.

I made some changes in this version of Monte Carlo Tree Search code that others have not made.

I decided to make the state of the world and which player's turn it is an attribute of node and board 
respectively. I figured this would make things simpler in the agent class, though in retrospect I think
both methods are fairly equivalent.

NOTE^: I think Deepmind did not have states as attributes of the nodes. I believe this choice actually
stems from their MuZero pseudocode which they linked to in their paper on the arXiV. For MuZero, the
states are represented as hidden states which are outputs of the network. This forward pass is computationally
expensive and take up memory, so it doesn't make sense to calculate these values for every child node 
(even the ones you don't visit).

I included an implementation here for Connect 4 using a resnet architecture or a vision-esque transformer
architecture. Also included in a folder is an implementation for Chess.