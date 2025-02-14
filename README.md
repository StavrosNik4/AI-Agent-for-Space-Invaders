# AI Agent that learns to play Space Invaders
This was a Deep Learning project for the "NDM-08-02 - Computetional Intelligence - Deep Reinforcement Learning" class of School of Informatics of AUTh.

In this project you will find a Deep Q-Learning Algorithm Implementation and all the code for the experiments I ran.

## Requirements
The requirements.txt file can be used to install all the important Python packages for this project if you want to run the code. Use this command:

`pip install requirements.txt`

## Code
For each experiment there is a .py file with already set parameters. 

### Enviroment

I use the Farama Foundation **Gymnasium** Python Package to manage the enviroment of the game.

### Game State Preprocess
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_1.png" width="500px">
</p>

The Game State is the 210x160 screenshot of the game's current frame. I grayscale, crop, normalize and downscale it to a 84x84 image.

### Neural Network Architectures Used
I use PyTorch to define and manage the architectures of the Neural Networks. I used 2 kinds of architectures for this project. I wanted to check if the second one was indeed better for this kind of problem as stated in the scientific papers I read. For both kinds of networks the input layer is the 4 last preprocessed images of the game state while the output is a list of 6 numbers (as many as the option the agent has for the game).

#### Convolutional Neural Network (CNN)
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_2.png" width="500px">
</p>

As you can see in the image, I use 2 Convolutional Layers and 1 Fully Connected one before the output layer.

#### Dueling Network
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_3.png" width="500px">
</p>

The only difference from the CNN is that I use 2 parallel Fully Connected layers (streams) instead of one. One is a Value Layer and the other is an Advantage Layer. I aggregate those 2 layers on the output layer. This architecture tends to have better results and that's why I choose to use this one after the 2 first expirements where I compared the 2 architectures.

### Deep Q-Learning Algorithm Implementation
I implemented the Deep Q-Learning Algorithm using 2 Neural Networks, a main and a target one. I'm also using epsilon-greedy policy.

### Memory

I implemented 2 different kinds of memory to save experiences that will help the agent during the training.

#### Simple Memory
This is just a simple queue where I save recent experiences and I sample a mini-batch when I need some of them. It is used on Expirements 1 and 2.

#### Prioritized Experience Replay
This is a better memory implementation where I use a smarter way to save and sample experiences using their TD-error. Experiences with higher TD-error tend to be more usefull for the training process, but we also need to use those ones with lower TD-error as well. That's why I use a Sum Tree Data Structure to implement this kind of memory. It is used on Expirements 3 and 4.

### Training
If you want to train one of the models I implemented on your own (or maybe you want to edit one of those and try it out), you can edit the code of the .py file to change some parameters. 

### Testing
If you want to test one of the Saved Models simply change the TRAINING boolean to False and change the `num_episodes_model` variable to choose the saved model you want.

### Saliency Maps
In `saliency_maps.py` file I load a model, I let it play and I plot the Saliency Maps to see which parts of the input are more important for the decision making process of the agent.

<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_11.png" width="500px">
</p>

## Saved Models
Models from all experiments can be found in the Saved Models folder in .pth format. Each model name is as follows:

`architecture_algorithm_model_numberOfTrainingEpisodes.pth`

For example `dueling_dqn_model_200.pth` refers to the saved model of a dueling network architecture that was trained for 200 episodes with the Deep Q-Learning Algorithm.

## Presentation & Full Report
The English presentation of this project is available in this repo in PDF form. There is a full report in PDF format but currently is only available in Greek.

## Video Showcase
In the links below you can find 2 videos that showcase how Agents from Experiment 3 and Experiment 4 play one match (3 lives) of the game on Hard Mode.

<li> Experiment_3: https://youtu.be/3huXFd_Eogw </li>
<li> Experiment_4: https://youtu.be/A-5bzvdXvkU </li>

## Medium Article
You can also read [my article on Deep Learning on Medium](https://medium.com/@stavrosnik4k/τι-είναι-η-βαθιά-ενισχυτική-μάθηση-792389f81bbf) where I use this project to explain it better to non Computer Science people. Currently, it's only available in Greek, but you can use an auto-translation tool to read it. 

## Bibliografic References
[1] Mark Towers et al. “Gymnasium”. In: (Mar. 2023).

[2] M. G. Bellemare et al. “The Arcade Learning Environment: An Evaluation Platform for General Agents”. In: Journal of Artificial Intelligence Research 47 (June 2013), pp. 253–279.

[3] Ziyun Wang et al. “Dueling Network Architectures for Deep Reinforcement Learning”. In: International Conference on Machine Learning. 2015. Url: https://api.semanticscholar.org/CorpusID:5389801 

[4] H. V. Hasselt, Arthur Guez, and David Silver. “Deep Reinforcement Learning with Double Q-Learning”. In: AAAI Conference on Artificial Intelligence. 2015. url: https://api.semanticscholar.org/CorpusID:5389801 