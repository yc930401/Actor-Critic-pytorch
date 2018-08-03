# DQN to play Cartpole game with pytorch

DQN to play Cartpole game with pytorch

## Introduction

Humans excel at solving a wide variety of challenging problems, from low-level motor control through to high-level cognitive tasks. 
Like a human, our agents learn for themselves to achieve successful strategies that lead to the greatest long-term rewards. This paradigm of 
learning by trial-and-error, solely from rewards or punishments, is known as reinforcement learning (RL). Also like a human, our agents 
construct and learn their own knowledge directly from raw inputs, such as vision, without any hand-engineered features or domain heuristics. 
This is achieved by deep learning of neural networks.
The agents must continually make value judgements so as to select good actions over bad. This knowledge is represented by a Q-network that 
estimates the total reward that an agent can expect to receive after taking a particular action. The key idea was to use deep neural networks 
to represent the Q-network, and to train this Q-network to predict total reward. Previous attempts to combine RL with neural networks had 
largely failed due to unstable learning. To address these instabilities, our Deep Q-Networks (DQN) algorithm stores all of the agent's experiences
 and then randomly samples and replays these experiences to provide diverse and decorrelated training data. </br>
Reinforcement learning: </br>
![reinforcement learning](reinforcement_learning.png) </br>
In this post, I implement a DQN to Cartpole game: </br>
![Cartpole](Cartpole.png) </br>


## Methodology

1. Get data from gym, and preprocess the state using raw images
2. Define a DQN network, experience-replay memory, and a function to choose action based on exploration threshold
3. Play Cartpole game, save the experiences and train the network with data from memory
4. Save the model



## References:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html </br>
http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf </br>
