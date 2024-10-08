import d4rl
from mjrl.utils.gym_env import GymEnv
import gym
import numpy as np
import torch
import argparse
import random
import os
import time
import collections
import transformers
import torch.nn.functional as F
import torch.nn as nn


from ql_trainer import Trainer
from trajectory_gpt2 import GPT2Model


# https://github.com/charleshsc/QT/tree/8c83dc305dc0515e1bb8e1ecb2266efa82bbf128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(env_name):
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    
    use_timeouts = 'timeouts' in dataset
    
    episode_step = 0
    paths = []
    # Loop is to create episode trajectories
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            # If timeouts are not used, consider episode done after 1000 steps
            final_timestep = (episode_step == 1000-1)
        
        # Collect data for current timestep
        for k in ['observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
        
        
        # When an episode is complete, it packages all the collected data for that episode into a dictionary 
        # and adds it to the "paths" list.
        if done_bool or final_timestep: # If episode is done (either by terminal state or timeout)
            episode_step = 0
            episode_data = {}
            # Convert collected data to numpy arrays
            for k in data_:
                episode_data[k] = np.array(data_[k])
            
            #  Each "path" is essentially one complete episode from the dataset, 
            #  containing all the information (observations, actions, rewards, terminals) f
            paths.append(episode_data)
            # Reset data collection for next episode
            data_ = collections.defaultdict(list)
        else:
            episode_step += 1
    
    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    return paths




def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    with torch.no_grad():
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
        rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=0).to(dtype=torch.float32),
                None,
                timesteps=timesteps.to(dtype=torch.long),
            )            
            
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
#             rewards[-1] = reward

            timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        critic,
        max_ep_len=2000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None, # [12.0, 9.0, 6.0], the return-to-go that we try to reach
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    if len(ep_return) > 1:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).unsqueeze(-1)
    else:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                critic,
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=1).to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

            action = action.detach().cpu().numpy().flatten()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[:,-1:] - (reward/scale)
            else:
                pred_return = target_return[:,-1:]
            target_return = torch.cat(
                [target_return, pred_return], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length



class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])




class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            sar=False,
            scale=1.,
            rtg_no_q=False,
            infer_no_q=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.config = config
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Create attention mask if not provided
        # The attention_mask ensures 
        # that the model only attends to valid elements in the sequence and ignores the padding. 
        # The attention_mask is a binary mask that indicates 
        # which elements in the input sequence are valid and 
        # should be attended to (1), and which elements are padding and should be ignored (0).
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            # This creates a tensor of ones with the same batch size and sequence length as the input
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device) # creates a tensor of ones with the shape (batch_size, seq_length).

        # Embed each modality (state, action, return, reward) 
        # This transforms each input into the hidden size dimension
        state_embeddings = self.embed_state(states) #  vector of size hidden_size.
        action_embeddings = self.embed_action(actions) #  vector of size hidden_size.
        returns_embeddings = self.embed_return(returns_to_go) #  vector of size hidden_size.
        reward_embeddings = self.embed_rewards(rewards / self.scale)  # Note: rewards are scaled

        # Embed timesteps
        # This creates positional embeddings for each timestep,
        # that represent the position of each element in the sequence. 
        time_embeddings = self.embed_timestep(timesteps) #  vector of size hidden_size.

        # Add time embeddings to each modality embedding
        # Helps the model distinguish between events at different time steps
        # Implementation: Element-wise addition of time embeddings to each modality
        state_embeddings += time_embeddings    # Enrich state information with temporal context
        action_embeddings += time_embeddings   # Add time awareness to action embeddings
        returns_embeddings += time_embeddings  # Incorporate time into return-to-go embeddings
        reward_embeddings += time_embeddings   # Enhance reward embeddings with temporal information
        
        if self.sar: # False
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, reward_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        else: # True
            # The transformation is necessary to prepare the input data in a format 
            # that the transformer model can process. 
            # Stacking: Combines the returns_embeddings, state_embeddings, and action_embeddings
            # Permutation: Changes the order of the dimensions to (batch_size, 3*seq_length, hidden_size)
            # Reshape: Flattens the modality dimension (3) and sequence length dimension (seq_length) into a single sequence dimension (3*seq_length)
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # inputs_embeds: The input embeddings for the transformer model.
        stacked_inputs = self.embed_ln(stacked_inputs) # Layer normalization

        
        #  Preparing the attention_mask to match the format of the stacked_inputs
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        # Go GPT2Model forward function 
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # the last_hidden_state is used to generate predictions for the next actions, states, and rewards. 
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        if self.sar: # False, sar(state-action-reward)
            action_preds = self.predict_action(x[:, 0])
            rewards_preds = self.predict_rewards(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
        else:
            action_preds = self.predict_action(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
            rewards_preds = None


        return state_preds, action_preds, rewards_preds

    def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=50, dim=0)
        actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=50, dim=0)
        rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=50, dim=0)
        timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=50, dim=0)

        bs = returns_to_go.shape[0] # 3
        returns_to_go = returns_to_go.reshape(bs, -1, 1).repeat_interleave(repeats=50 // bs, dim=0) 
        returns_to_go = torch.cat([returns_to_go, torch.randn((50-returns_to_go.shape[0], returns_to_go.shape[1], 1), device=returns_to_go.device)], dim=0) # 50
            

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # padding
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1).repeat_interleave(repeats=50, dim=0)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1
            ).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
        else:
            attention_mask = None

        returns_to_go[bs:, -1] = returns_to_go[bs:, -1] + torch.randn_like(returns_to_go[bs:, -1]) * 0.1
        if not self.rtg_no_q:
            returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale
        _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
    
        
        state_rpt = states[:, -1, :]
        action_preds = action_preds[:, -1, :]

        q_value = critic.q_min(state_rpt, action_preds).flatten()
        idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)

        if not self.infer_no_q:
            return action_preds[idx]
        else:
            return action_preds[0]








class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 8 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def save_checkpoint(state,name):
  filename =name
  torch.save(state, filename)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def experiment(exp_prefix,
        seed, env, dataset, mode, K, pct_traj,
        batch_size, embed_dim, n_layer, n_head,
        activation_function, dropout, learning_rate,
        lr_min, weight_decay, warmup_steps,
        num_eval_episodes, max_iters,
        num_steps_per_iter, save_path,
        discount, tau, eta, eta2, lambda_val,
        max_q_backup, lr_decay, grad_norm,
        early_stop, early_epoch, k_rewards,
        use_discount, sar, reward_tune, scale,
        test_scale, rtg_no_q, infer_no_q
):
    device = 'cuda'

    env_name, dataset = env, dataset
    #group_name = f'{exp_prefix}-{env_name}-{dataset}'
    #timestr = time.strftime("%y%m%d-%H%M%S")
    #exp_prefix = f'{group_name}-{seed}-{timestr}'

    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 9000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [5000, 4000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        # from decision_transformer.envs.reacher_2d import Reacher2dEnv
        # env = Reacher2dEnv()
        env = gym.make('Reacher-v4')
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
        dversion = 2
    elif env_name == 'pen':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'hammer':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.
    elif env_name == 'door':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 100.
    elif env_name == 'relocate':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 1000]
        scale = 1000.
        dversion = 1
    elif env_name == 'kitchen':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.
    elif env_name == 'maze2d':
        if 'open' in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [300, 200, 150,  100, 50, 20]
        scale = 10.
    elif env_name == 'antmaze':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
        scale = 1.
    else:
        raise NotImplementedError
    
    if scale is not None:
        scale = scale
    
    max_ep_len = max_ep_len
    env_targets = env_targets
    scale = scale
    if test_scale is None:
        test_scale = scale

    env.seed(seed)
    set_seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    
    # Use the function directly
    env_name = 'halfcheetah-medium-v2'  # You can change this to match your specific environment
    completed_trajectories = create_dataset(env_name) # dict_keys(['observations', 'actions', 'rewards', 'terminals'])

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in completed_trajectories: # path: refers the whole trajectory or episode: (observations, actions, rewards, terminals) 
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    obs_lens, returns = np.array(traj_lens), np.array(returns) # obs_lens: length of each completed trajectory or episode
                                                               # returns: sum of rewards for each completed trajectory or episode

    # used for input normalization
    states = np.concatenate(states, axis=0) # input(list): states (1000,), output(np.array): states (1000000, 17)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(obs_lens) # represents the total size of the dataset in terms of individual state-action pairs.
    
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(obs_lens)} episodes, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    

    # Use the previously defined variables directly
    # K, batch_size, num_eval_episodes are already defined
    pct_traj = pct_traj  # This was defined earlier with a default value of 1.

    # only train on top pct_traj trajectories (for %BC experiment), BC: behavior cloning
    # Calculate the number of timesteps to consider based on the percentage of trajectories (pct_traj)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    
    # Sort the indices of returns from lowest to highest
    sorted_inds = np.argsort(returns)  # lowest to highest
    
    # Initialize variables for selecting top trajectories
    num_episode = 1  # Start with the top trajectory/episode (highest return)
    timesteps = obs_lens[sorted_inds[-1]]  # Get the length of the top trajectory (trajectory with highest return)
    ind = len(completed_trajectories) - 2  # Start from the second-best trajectory (second highest return)
    
    # Note: The top trajectory refers to the trajectory with the highest cumulative reward (return).
    # We start by including this best-performing trajectory and then iteratively add more high-performing
    # trajectories until we reach the desired number of timesteps for training.
    
    # This setup prepares for the following while loop, which will:
    # 1. Add more trajectories until we reach the desired number of timesteps
    # 2. Select only the top-performing trajectories for training
    while ind >= 0 and timesteps + obs_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += obs_lens[sorted_inds[ind]]
        num_episode += 1
        ind -= 1
    
    # Select only the indices of the top-performing trajectories.
    # We use negative indexing to get the last 'num_episode' elements.
    # This effectively selects the indices of trajectories with the highest returns,
    # that means sorted_inds[0] is the index of the trajectory with the highest return.
    sorted_inds = sorted_inds[-num_episode:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = obs_lens[sorted_inds] / sum(obs_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        """
        Generates a batch of data for training the model.

        This function samples trajectories from the dataset, processes them,
        and returns batched data ready for model input.

        Args:
            batch_size (int): The number of trajectories to sample. Default is 256.
            max_len (int): The maximum sequence length to consider. Default is K.

        Returns:
            tuple: A tuple containing batched and processed data:
                - s (torch.Tensor): States, shape (batch_size, max_len, state_dim)
                - a (torch.Tensor): Actions, shape (batch_size, max_len, act_dim)
                - r (torch.Tensor): Rewards, shape (batch_size, max_len, 1)
                - d (torch.Tensor): Done flags, shape (batch_size, max_len, 1)
                - rtg (torch.Tensor): Return-to-go, shape (batch_size, max_len, 1)
                - timesteps (torch.Tensor): Timesteps, shape (batch_size, max_len)
                - mask (torch.Tensor): Attention mask, shape (batch_size, max_len)
                - target_a (torch.Tensor): Target actions, shape (batch_size, max_len, act_dim)

        The function samples trajectories, extracts sequences, applies necessary
        preprocessing (e.g., reward scaling, return-to-go calculation), pads sequences,
        and converts data to PyTorch tensors on the specified device.
        """
        
        # Randomly select batch_size number of trajectories, weighted by their lengths
        batch_inds = np.random.choice(np.arange(num_episode), size=batch_size, replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        #   rtg (return-to-go): the sum of rewards from the current timestep to the end of the episode
        s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
        
        # max_len: maximum length of the trajectory or episode, it is 20
        # the loop below ensures that all sequences in a batch have the same length.
        for i in range(batch_size):
       
            # Get the index of the trajectory from the sorted indices
            # batch_inds[i] gives us the index within the top-performing trajectories
            # sorted_inds[batch_inds[i]] gives us the actual index in the original list
            rand_indx = int(sorted_inds[batch_inds[i]])
            
            # Retrieve the trajectory from the completed_trajectories list
            # This trajectory is one of the top-performing ones, as determined earlier
            traj = completed_trajectories[rand_indx]
            # Choose a random starting point within the trajectory
            si = random.randint(0, traj['rewards'].shape[0] - 1) 

            # Selecting random segments of each episode,
            # this helps the model generalize better by seeing different parts of trajectories during training.
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            # Apply reward scaling for specific environments
            if reward_tune == 'cql_antmaze':
                traj_rewards = (traj['rewards']-0.5) * 4.0
            else:
                traj_rewards = traj['rewards']
            r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
            
            # Calculate return-to-go
            # it stores the cumulative future rewards for each segment of the trajectory. 
            rtg.append(discount_cumsum(traj_rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            #  To have the RTG sequence be exactly one step longer than the state sequence
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            
            # Pad sequences to max_len and apply state normalization
            # Padding: Ensures that all sequences in a batch have the same length (max_len).
            # Normalization: Adjusts the states to have a mean of 0 and a standard deviation of 1.
            # The -1 index is used to refer to the most recently added sequence in the list
            tlen = s[-1].shape[1] # length of the current trajectory or episode
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1) # padding with zeros to make it 20
            s[-1] = (s[-1] - state_mean) / state_std # normalizing the states
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        # Convert numpy arrays to PyTorch tensors and move to specified device
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, target_a, d, rtg, timesteps, mask

    def eval_episodes(target_rew): 
        def fn(model, critic):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        critic,
                        max_ep_len=max_ep_len,
                        scale=test_scale,
                        target_return=[t/test_scale for t in target_rew], # target_return: the target return-to-go that we try to reach
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_normalized_score': env.get_normalized_score(np.mean(returns)),
                }
        return fn

    model = DecisionTransformer(
        state_dim=state_dim,        # Dimension of the state space
        act_dim=act_dim,            # Dimension of the action space
        max_length=K,               # Maximum sequence length the model can handle
        max_ep_len=max_ep_len,      # Maximum episode length
        hidden_size=embed_dim,      # Size of the hidden layers in the transformer
        n_layer=n_layer,            # Number of transformer layers
        n_head=n_head,              # Number of attention heads in each transformer layer
        n_inner=4*embed_dim,        # Size of the inner feedforward layer in transformer blocks
        activation_function=activation_function,  # Activation function used in the model
        n_positions=1024,           # Maximum number of positions for positional encoding
        resid_pdrop=dropout,        # Dropout rate for residual connections
        attn_pdrop=dropout,         # Dropout rate for attention probabilities
        scale=scale,                # Scaling factor for rewards
        sar=sar,                    # Flag for using state-action-reward ordering
        rtg_no_q=rtg_no_q,          # Flag for not using Q-values in return-to-go calculations
        infer_no_q=infer_no_q       # Flag for not using Q-values during inference
    )
    
    
    critic = Critic(state_dim, act_dim, hidden_dim=embed_dim)


    model = model.to(device=device)
    critic = critic.to(device=device)

    trainer = Trainer(
        model=model,                # The main model (Decision Transformer)
        critic=critic,              # The critic model for value estimation
        batch_size=batch_size,      # Number of samples in each training batch
        tau=tau,                    # Soft update coefficient for target networks
        discount=discount,          # Discount factor for future rewards
        get_batch=get_batch,        # Function to retrieve a batch of data
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),  # Loss function for training
        eval_fns=[eval_episodes(env_targets)],  # Functions for evaluation
        max_q_backup=max_q_backup,  # Flag for using max Q-value in TD backup
        eta=eta,                    # Learning rate for actor
        eta2=eta2,                  # Learning rate for critic
        ema_decay=0.995,            # Exponential moving average decay rate
        step_start_ema=1000,        # Step to start using EMA
        update_ema_every=5,         # Frequency of EMA updates
        lr=learning_rate,           # Initial learning rate
        weight_decay=weight_decay,  # L2 regularization factor
        lr_decay=lr_decay,          # Learning rate decay flag
        lr_maxt=max_iters,          # Maximum number of iterations for lr scheduling
        lr_min=lr_min,              # Minimum learning rate
        grad_norm=grad_norm,        # Gradient clipping norm
        scale=scale,                # Scaling factor for rewards
        k_rewards=k_rewards,        # Number of rewards to consider in TD error
        use_discount=use_discount   # Flag to use discounted rewards
    )


    # Initialize variables to track the best performance
    best_ret = -10000  # Best return (total reward) achieved so far
    best_nor_ret = -1000  # Best normalized return achieved so far
    best_iter = -1  # Iteration number where the best performance was achieved

 
    for iter in range(max_iters):
        outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter+1, log_writer=None)
        trainer.scale_up_eta(lambda_val)
        ret = outputs['Best_return_mean']
        nor_ret = outputs['Best_normalized_score']
        if ret > best_ret:
            state = {
                'epoch': iter+1,
                'actor': trainer.actor.state_dict(),
                'critic': trainer.critic_target.state_dict(),
            }
            #save_checkpoint(state, os.path.join(save_path, exp_prefix, 'epoch_{}.pth'.format(iter + 1)))
            best_ret = ret
            best_nor_ret = nor_ret
            best_iter = iter + 1
        print(f'Current best return mean is {best_ret}, normalized score is {best_nor_ret*100}, Iteration {best_iter}')
        
        if early_stop and iter >= early_epoch:
            break
    print(f'The final best return mean is {best_ret}')
    print(f'The final best normalized return is {best_nor_ret * 100}')

if __name__ == '__main__':
    exp_name = 'gym-experiment' 
    seed = 123
    env = 'halfcheetah'
    dataset = 'medium'  # medium, medium-replay, medium-expert, expert
    mode = 'normal'  # normal for standard setting, delayed for sparse
    K = 20 # maximum length of the trajectory or episode
    pct_traj = 1. # percentage of trajectories to consider for training
    batch_size = 256 # number of trajectories to sample for training
    embed_dim = 256 # dimension of the embedding space
    n_layer = 4 # number of transformer layers
    n_head = 4 # number of attention heads
    activation_function = 'relu' # activation function
    dropout = 0.1
    learning_rate = 3e-4
    lr_min = 0.
    weight_decay = 1e-4
    warmup_steps = 10 #10000
    num_eval_episodes = 10
    max_iters = 5 #500
    num_steps_per_iter = 10#1000
    save_path = './save/'

    discount = 0.99
    tau = 0.005
    eta = 1.0
    eta2 = 1.0
    lambda_val = 1.0
    max_q_backup = False
    lr_decay = False
    grad_norm = 2.0
    early_stop = False
    early_epoch = 100
    k_rewards = False
    use_discount = False
    sar = False
    reward_tune = 'no'
    scale = None
    test_scale = None
    rtg_no_q = False
    infer_no_q = False

    # Call experiment function with these variables
    experiment(exp_prefix=exp_name, seed=seed, env=env, dataset=dataset, mode=mode, K=K, pct_traj=pct_traj,
               batch_size=batch_size, embed_dim=embed_dim, n_layer=n_layer, n_head=n_head,
               activation_function=activation_function, dropout=dropout, learning_rate=learning_rate,
               lr_min=lr_min, weight_decay=weight_decay, warmup_steps=warmup_steps,
               num_eval_episodes=num_eval_episodes, max_iters=max_iters,
               num_steps_per_iter=num_steps_per_iter, save_path=save_path,
               discount=discount, tau=tau, eta=eta, eta2=eta2, lambda_val=lambda_val,
               max_q_backup=max_q_backup, lr_decay=lr_decay, grad_norm=grad_norm,
               early_stop=early_stop, early_epoch=early_epoch, k_rewards=k_rewards,
               use_discount=use_discount, sar=sar, reward_tune=reward_tune, scale=scale,
               test_scale=test_scale, rtg_no_q=rtg_no_q, infer_no_q=infer_no_q)
