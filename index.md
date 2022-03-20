## Work in this Repository

Welocme! The work in this repository represents a combination of udemy course work authored by Dr. Phil Tabor, and
to a larger extent independent work I have embarked on for fun.

I decided to create a separate repository not long ago to host all of my RL work as I spend the greatest amount
of time on it. 

Some of the projects I have enjoyed the most thus far are creating an AlphaZero program and teaching in to play
connect 4 and chess, and doing the same with DeepMind's newer model MuZero.

![This is an image] (AlphaZero.png)

I made some adaptations for the AlphaZero implementation to attempt to train faster (and hopefully better)
by using a vision-esque transformer network structure rather than the standard ResNet architecture employed
by DeepMind and others.

[Vision Transformer Paper] (https://arxiv.org/abs/2010.11929)

I plan to continue this work in the future and keep learning and experimenting with cool new ideas.

### Future Areas of Study

I did notice that in the MuZero paper, the DeepMind team mentioned that they were not able to acheive great 
performance in the game of Go when they increased the Monte Carlo Tree Search simulation number much past
100 (was 1600 in AlphaGo for reference). They attributed this to the model's lack of ability to project
future states in the game. 

Ordinarily in a stochastic environment this makes sense, however in a deterministic environment with
a state that can be represented by less than 1000 bits, this shouldn't make any sense. Learning a perfect
model in a relatively small deterministic environment should be extremely easy, and clearly (to my mind 
at least) does not neccessitate a giant ResNet architecture.

I would love to do some digging soon and try to figure out which models to use in 1) deterministic cases
and 2) stochastic cases.

I also would like to study and try to implement [model-based meta-policy optimization] at some point. (https://arxiv.org/abs/1809.05214)
