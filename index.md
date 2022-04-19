## Work in this Repository

Welocme! The work in this repository represents a combination of udemy course work authored by Dr. Phil Tabor, and
to a larger extent independent work I have embarked on for fun.

I decided to create a separate repository not long ago to host all of my RL work as I spend the greatest amount
of time on it. 

The largest projects I have worked on and I think the ones I have enjoyed the most are building an 
[AlphaZero](https://github.com/jdm365/Reinforcement_Learning/tree/main/AlphaZero) program and teaching in to play
connect 4 and chess, and doing the same with DeepMind's newer model 
[MuZero](https://github.com/jdm365/Reinforcement_Learning/tree/main/MuZero).

![AlphaZero](AlphaZero.png)

I made some adaptations for the AlphaZero implementation to attempt to train faster (and hopefully better)
by using a vision-esque transformer network structure rather than the standard ResNet architecture employed
by DeepMind and others. 

[Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

Even with these models, successfully learning games like chess and go proved relatively intractible to someone
like me. For reference, Deep Mind used 5000 first generation TPU's to generate self play games, and 64 second
generation TPU's to train their networks. I do have a RTX 3080 ti, which while powerful for me would take over 50 
years to compute what Deepmind did in 9 hours. 

Regardless it was a fun learning experience and the model was able to defeat me in Connect 4!

### Meta-Learning Models

I also have been working on meta-learning and working with ensemble models.
[model-based meta-policy optimization](https://arxiv.org/abs/1809.05214) 

Incidentally I believe that models such as A3C (Asynchronous advantage actor-critc) should not be viewed as a 
reinforcement algorithm itself. It effectively sidesteps the explore-exploit dilemna to a degree by collecting
trajectories from multiple actors. This can be used to drastically improve the performance of any RL algorithm and
I believe shouldn't be used to judge its robustness.

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

I also would like to explore the concept of [intrinsic curiosity](https://arxiv.org/pdf/1705.05363.pdf).
I have read the paper and think that having an agent which at any time at any space in the environment
know how and when to explore is crucial and that this method of exploration is fascinating. I think this will
prove to be the ripest area of RL in the future and that measuring performance on games like Mario will
become the leading benchmarks.
