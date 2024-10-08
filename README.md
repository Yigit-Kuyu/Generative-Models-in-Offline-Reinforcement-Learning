# INFO

This repository is dedicated to the implementation and exploration of various generative models, specifically Variational Autoencoders (VAE), Diffusion Model (DM), Decision Transformer (DT), and Generative Adversarial Network (GAN) in the context of Offline Reinforcement Learning (RL) using the D4RL environment. 


## Repository Structure

- **GAN_RL_d4rl**: Implements GANs for offline RL tasks using the D4RL environment. This script includes the setup for training GANs to generate realistic state-action pairs that can be used to augment the offline dataset.

- **CGDT_Transformer**: Contains the training and evaluation of the DT model (GPT2-based) and also includes a SequenceTrainer class for managing the training iterations. The Critic network is implemented as an ImitationTransformer, which shares a similar architecture to the DT but focuses on evaluating the predicted actions and returns.

- **QT_Transformer**: Focuses on DT which is the main model that learns to predict actions based on past states, actions, and rewards. It is implemented as a GPT-like transformer model. The critic network evaluates the quality of actions predicted by the DT. 

- **DiffusionQL**: Leverages DM to learn policies, that is used to generate a diverse set of trajectories by simulating the diffusion process, which helps in exploring the state-action space more effectively.

- **BEAR_VAE**: Helps to constrain the learned policy to remain close to the behavior policy to reduce errors in value estimation and prevent distributional shift.

  
## Installation

To get started, clone the repository and run each code independently.
