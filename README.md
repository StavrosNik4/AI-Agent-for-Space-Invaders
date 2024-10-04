# AI Agent that learns to play Space Invaders
This was a Deep Learning project for the "NDM-08-02 - Computetional Intelligence - Deep Reinforcement Learning" class of School of Informatics of AUTh.

In this project you will find a Deep Q-Learning Algorithm Implementation and all the code for the experiments I ran.

## Requirements
The requirements.txt file can be used to install all the important Python packages for this project if you want to run the code. Use this command:

`pip install requirements.txt`

## Code
For each experiment there is a .py file with already set parameters. 

### Game State Preprocess
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_1.png" width="500px">
</p>

The Game State is the 210x160 screenshot of the game's current frame. I grayscale, crop, normalize and downscale it to a 84x84 image.

### Neural Network Architectures Used
I use PyTorch to define and manage the architectures of the Neural Networks. I used 2 kinds of architectures for this project. I wanted to check if the second one was indeed better for this kind of problem as stated in the scientific papers I read. For both kinds of networks the input layer is the 4 last preprocessed images of the game state while the output is a list of 6 numbers (as many as the option the agent have).

#### Convolutional Neural Network (CNN)
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_2.png" width="500px">
</p>

As you can see in the image, I use 2 Convolutional Layers and 1 Fully Connected one before the output layer.

#### Dueling Network
<p align="center">
<img src="https://github.com/StavrosNik4/AI-Agent-for-Space-Invaders/blob/6f868e6783945cfbc9a3c15ee836c900481148a4/images/Figure_3.png" width="500px">
</p>

The only difference from the CNN is that I use 2 parallel Fully Connected layers (streams) instead of one. One is a Value Layer and the other is an Advantage Layer. I aggregate those 2 layers on the output layer. This architecture tends to have better results and that's why I choose to choose this one after the 2 first expirements where I compare the 2 architectures.

### Deep Q-Learning Algorithm Implementation
I implemented the Deep Q-Learning Algorithm using 2 Neural Networks, a main and a target one with epsilon-greedy policy.

### Memory

#### Simple Memory

#### Prioritized Experience Replay

### Training
If you want to train one of the models I implemented on your own (or maybe you want to edit one of those and try it out), you can edit the code of the .py file to change some parameters. 

### Testing


### Saliency Maps
In `saliency_maps.py` file you can find...

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
You can also read [my article on Deep Learning on Medium](https://medium.com/@stavrosnik4k/τι-είναι-η-βαθιά-ενισχυτική-μάθηση-792389f81bbf) where I use this project to explain it better to non Computer Science people.

## Bibliografic References
