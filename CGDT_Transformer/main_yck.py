
import os
import sys
import argparse
import pickle
import collections
import random
import time
import gym
import d4rl
import torch
import numpy as np
import wandb
from copy import deepcopy
from tqdm import tqdm
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from decision_transformer import DecisionTransformer, ImitationTransformer
from d4rl import get_normalized_score as get_d4rl_normalized_score

MAX_EPISODE_LEN = 10#1000
NUM_PASSED_STEPS = 2 #20


# https://github.com/sharkwyf/cgdt/tree/7eab0186a82c60ddf4671178fa354bf0a512fe79


######################### Trainer #########################



class CriticSequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        critic,
        critic_optimizer,
        log_temperature_optimizer,
        value_coef=0.,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.value_coef = value_coef
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        critic_loss_fn,
        value_loss_fn,
        dataloader,
    ):

        losses, nlls, entropies, mses = [], [], [], []
        critic_losses, value_losses = [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        self.critic.train()
        for _, trajs in enumerate(dataloader):
            (critic_loss) = self.train_step_critic(critic_loss_fn, trajs)
            critic_losses.append(critic_loss)

        self.critic.eval()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, mse, value_loss = self.train_step_stochastic(loss_fn, value_loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            mses.append(mse)
            value_losses.append(value_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/mse"] = mses[-1]
        logs["training/train_critic_loss_mean"] = np.mean(critic_losses)
        logs["training/train_critic_loss_std"] = np.std(critic_losses)
        logs["training/train_value_loss_mean"] = np.mean(value_losses)
        logs["training/train_value_loss_std"] = np.std(value_losses)
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_critic(self, critic_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        rtg_discounted_target = torch.clone(rtg_discounted)

        state_preds, action_preds, return_preds = self.critic.forward(
            states,
            actions,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        critic_loss, critic_mse, critic_nll, rtg, rtg_hat_mean, rtg_hat_std = critic_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.25)
        self.critic_optimizer.step()

        return (
            critic_loss.detach().cpu().item(),
        )

    def train_step_stochastic(self, loss_fn, value_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)
        rtg_discounted_target = torch.clone(rtg_discounted)
        # Diffusion Transformer:
        state_preds, action_preds, return_preds= self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, mse = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )

        _, _, return_preds = self.critic.forward(
            states,
            action_preds.mean,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        value_loss, value_mse, value_nll, rtg, rtg_hat_mean, rtg_hat_std = value_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
        )

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        ((loss + value_loss * self.value_coef) / np.abs(1 + self.value_coef)).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            mse.detach().cpu().item(),
            value_loss.detach().cpu().item(),
        )


################ Replay Buffer #################

class ReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            self.trajectories[
                self.start_idx : self.start_idx + len(new_trajs)
            ] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity


######### Create Dataset #########



def create_dataset():
    datasets = []

    for env_name in ["halfcheetah"]:
        for dataset_type in ["medium", "medium-expert", "medium-replay", "expert"]:
            name = f"{env_name}-{dataset_type}-v2"
            env = gym.make(name)
            dataset = env.get_dataset()

            # Process raw data into episodes
            N = dataset["rewards"].shape[0]
            data_ = collections.defaultdict(list)
            use_timeouts = "timeouts" in dataset
            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset["terminals"][i])
                final_timestep = dataset["timeouts"][i] if use_timeouts else episode_step == 1000 - 1
                for k in ["observations", "next_observations", "actions", "rewards", "terminals"]:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {k: np.array(v) for k, v in data_.items()}
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            # Calculate statistics on processed data
            returns = np.array([np.sum(p["rewards"]) for p in paths])
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            print(f"Number of samples collected: {num_samples}")
            print(
                f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
            )

            with open(f"{name}.pkl", "wb") as f:
                pickle.dump(paths, f)
            
            datasets.append((name, paths))
    
    return datasets






################# DATA #################

class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
        gamma,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale
        self.gamma = gamma

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood. We clamp them to be within
        # the user defined action range.
        self.action_range = action_range

    def __call__(self, traj):
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        ss = traj["observations"][si : si + self.max_len].reshape(-1, self.state_dim)
        aa = traj["actions"][si : si + self.max_len].reshape(-1, self.act_dim)
        rr = traj["rewards"][si : si + self.max_len].reshape(-1, 1)
        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        rtg = traj["rtg"][si : si + tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        ss = (ss - self.state_mean) / self.state_std

        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])

        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)

        if self.gamma != 1.0:
            rtg_discounted = traj["rtg_discounted"][si : si + tlen + 1].reshape(
                -1, 1
            )
            if rtg_discounted.shape[0] <= tlen:
                rtg_discounted = np.concatenate([rtg_discounted, np.zeros((1, 1))])
            rtg_discounted = (
                np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg_discounted])
                * self.reward_scale
            )
            rtg_discounted = torch.from_numpy(rtg_discounted).to(dtype=torch.float32)
        else:
            rtg_discounted = torch.clone(rtg)

        return ss, aa, rr, dd, rtg, rtg_discounted, timesteps, ordering, padding_mask


def create_dataloader(trajectories, num_iters, batch_size, max_len, state_dim, act_dim, state_mean, state_std, reward_scale, gamma, action_range, num_workers):
    """
    This function creates a PyTorch DataLoader for training or evaluation purposes.
    It performs the following steps:

    1. Calculates the total number of sub-trajectories to sample based on batch size and number of iterations.
    2. Samples indices of trajectories using the sample_trajs function, which favors longer trajectories.
    3. Creates a transform object (TransformSamplingSubTraj) to preprocess the sampled trajectories.
    4. Creates a SubTrajectory dataset using the sampled trajectories and the transform.
    5. Returns a DataLoader that efficiently loads and batches the data for training or evaluation.

    The DataLoader provides batched data with the following properties:
    - It samples sub-trajectories from the input trajectories.
    - The data is transformed using the specified parameters (e.g., state normalization, reward scaling).
    - It uses multiple workers for efficient data loading.
    - The data is not shuffled to maintain the original order of the trajectories.
    - It uses pinned memory for faster data transfer to GPU (if available).

    This function is crucial for preparing the data in a format suitable for training
    decision transformer models on reinforcement learning tasks.
    """

    # Calculate total number of sub-trajectories to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size) # Randomly sample trajectory indices based on their lengths

    # Create a transform object for preprocessing the trajectories
    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
        gamma=gamma,
    )

    
    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    # Return a DataLoader with the specified properties
    dataloader=torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,)
    
    # Example usage of the dataloader
     
    # Create an iterator from the dataloader
    dataloader_iter = iter(dataloader)

    # Get the first batch
    first_batch = next(dataloader_iter)

    # Unpack all elements of the batch
    states, actions, rewards, dones, rtg, rtg_discounted, timesteps, ordering, padding_mask = first_batch

    # Now you can print information about each element
    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    print("Rewards shape:", rewards.shape)
    print("Dones shape:", dones.shape)
    print("RTG shape:", rtg.shape)
    print("Discounted RTG shape:", rtg_discounted.shape)
    print("Timesteps shape:", timesteps.shape)
    print("Ordering shape:", ordering.shape)
    print("Padding mask shape:", padding_mask.shape)
    
    
    return dataloader


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):
    # Calculate the length of each trajectory
    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    
    # Calculate the probability of sampling each trajectory
    # Longer trajectories have a higher probability of being sampled
    p_sample = traj_lens / np.sum(traj_lens)

    # Randomly sample trajectory indices based on their lengths
    # - np.arange(len(trajectories)): Creates an array of indices for all trajectories
    # - size=sample_size: Number of samples to draw
    # - replace=True: Allow sampling the same index multiple times
    # - p=p_sample: Probability distribution for sampling (longer trajectories have higher probability)
    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds






################# EVALUATION #################


def get_normalized_score(env_name, score):
    if "bernoulli-bandit" in env_name:
        return score
    else:
        return get_d4rl_normalized_score(env_name, score)


def create_vec_eval_episodes_fn(env_name, vec_env, eval_rtg, state_dim, act_dim, state_mean, state_std, device, use_mean=False, reward_scale=0.001, no_reward=False, delayed_reward=False):
    
    def eval_episodes_fn(model):
        
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        returns, lengths, _ = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            no_reward=no_reward,
            delayed_reward=delayed_reward,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/score_mean{suffix}": get_normalized_score(env_name, np.mean(returns)) * 100,
            f"evaluation/score_std{suffix}": get_normalized_score(env_name, np.std(returns)) * 100,
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
    no_reward=False,
    delayed_reward=False,
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

        # the return action is a SquashNormal distribution
        if model.discrete_action:
            action = action_dist[:, -1]
        else:
            action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
            if use_mean:
                action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
            action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        if delayed_reward:
            reward.fill(0)

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            if delayed_reward:
                rewards[ind, -1] = torch.tensor(episode_return[ind], dtype=torch.float32, device=device)
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )



################# UTILS #################


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


################# Main Part #################

class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range, self.discrete_action = self._get_env_spec(variant)
        self.gamma = variant["gamma"]
        #self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(master_work_dir, variant["env"], variant["no_reward"], variant["delayed_reward"])
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(os.getcwd(), variant["env"], variant["no_reward"], variant["delayed_reward"])
        top_data = self.offline_trajs[-int(len(self.offline_trajs) * variant["critic_top_percent"]):]
        train_size = int(0.9 * len(top_data))
        test_size = len(top_data) - train_size
        self.train_offline_trajs, self.test_offline_trajs = torch.utils.data.random_split(top_data, [train_size, test_size])

        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            discrete_action=self.discrete_action,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.critic = ImitationTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_critic_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            discrete_action=self.discrete_action,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.critic_optimizer = Lamb(
            self.critic.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.pretrain_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations

        self.best_pretrain_iter = None
        self.best_critic_state_dict = None
        self.min_pretrain_loss_mean = 1e9
        self.n_ascending_cnt = 0

        self.best_train_iter = None
        self.best_model_state_dict = None
        self.max_d4rl_score = -1e9

        self.pretrain_iter = 0
        self.train_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] or "kitchen" in variant["env"] or "bandit" in variant["env"] else 0.001
        '''
        if not variant["no_wandb"]:
            name = "{}{}_{}_rtg_{}_{}_iters_{}_{}_{}".format(
                variant["env"], "_delayed" if variant["delayed_reward"] else "", variant["tag"], variant["eval_rtg"], variant["online_rtg"], variant["max_pretrain_iters"], variant["max_train_iters"], variant["max_online_iters"]
            )
            variant["exp_name"] = name
            wandb.init(
                name=name,
                project="qdt-v4",
                config=variant,
                tags=[],
                reinit=True,
            )
            print(f"wandb initialized")
        '''
       

    def _get_env_spec(self, variant):
        if "bernoulli-bandit" in variant["env"]:
            from decision_transformer.envs.bernoulli_bandit import BernoulliBanditEnv
            env = BernoulliBanditEnv(
                num_arms=2,
                reward_power=3.0,
                reward_scale=0.9,
                generation_seed=0,
                bernoulli_prob=0.9,
                loop=False,
            )
            state_dim = 1
            act_dim = 2
            action_range = [0, 1]
            discrete_action = True
            env.close()
        else:
            env = gym.make(variant["env"])
            state_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            action_range = [
                float(env.action_space.low.min()) + 1e-6,
                float(env.action_space.high.max()) - 1e-6,
            ]
            discrete_action = False
            env.close()
        return state_dim, act_dim, action_range, discrete_action

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "train_iter": self.train_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
            "best_train_iter": self.best_train_iter,
            "best_model_state_dict": self.best_model_state_dict,
            "max_d4rl_score": self.max_d4rl_score,
        }
        if self.critic is not None:
            to_save.update({
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "pretrain_scheduler_state_dict": self.pretrain_scheduler.state_dict(),
                "best_critic_state_dict": self.best_critic_state_dict,
                "min_pretrain_loss_mean": self.min_pretrain_loss_mean,
                "n_ascending_cnt": self.n_ascending_cnt,
                "best_pretrain_iter": self.best_pretrain_iter,
            })

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "critic_state_dict" in checkpoint:
                self.critic.load_state_dict(checkpoint["critic_state_dict"])
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
                self.pretrain_scheduler.load_state_dict(checkpoint["pretrain_scheduler_state_dict"])
                self.best_pretrain_iter = checkpoint["best_pretrain_iter"]
                self.best_critic_state_dict = checkpoint["best_critic_state_dict"]
                self.min_pretrain_loss_mean = checkpoint["min_pretrain_loss_mean"]
                self.n_ascending_cnt = checkpoint["n_ascending_cnt"]
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )

            self.best_train_iter = checkpoint["best_train_iter"]
            self.best_model_state_dict = checkpoint["best_model_state_dict"]
            self.max_d4rl_score = checkpoint["max_d4rl_score"]

            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.train_iter = checkpoint["train_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, work_dir, env_name, no_reward=False, delayed_reward=False):

        dataset_path = f"{work_dir}/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        if no_reward: # False
            for traj in trajectories:
                traj["rewards"].fill(0)
        elif delayed_reward: # False
            for traj in trajectories:
                traj["rewards"][-1] = traj["rewards"].sum()
                traj["rewards"][:-1] = 0

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        # Select the top-performing trajectories based on returns
        sorted_inds = sorted_inds[-num_trajectories:]
        # Process each selected trajectory
        trajectories = [self._process_trajectory(trajectories[ii]) for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _process_trajectory(self, traj):
        traj["rtg"] = discount_cumsum(traj["rewards"], gamma=1.0)
        if self.gamma == 1.0:
            traj["rtg_discounted"] = traj["rtg"].copy()
        else:
            traj["rtg_discounted"] = discount_cumsum(traj["rewards"], gamma=self.gamma)
        return traj

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                delayed_reward=self.variant["delayed_reward"],
            )

            trajs = [self._process_trajectory(traj) for traj in trajs]

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ] # return--> eval_episodes_fn

        if self.variant["trainer"] == "PretrainCriticSequenceTrainer": # False
            trainer = PretrainCriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                linear_value_coef_start=self.variant["linear_value_coef_start"],
                linear_value_coef_steps=self.variant["linear_value_coef_steps"],
                pretrain_scheduler=self.pretrain_scheduler,
                scheduler=self.scheduler,
                device=self.device,
                action_space=eval_envs.action_space,
            )
        elif self.variant["trainer"] == "CriticSequenceTrainer": # False
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else: # True
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Pretraining Trainer: {type(trainer)}")

        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            train_critic_dataloader, test_critic_dataloader, train_dataloader = [None] * 3
            if self.n_ascending_cnt < NUM_PASSED_STEPS: # True
                train_critic_dataloader = create_dataloader(
                    trajectories=self.train_offline_trajs,
                    num_iters=self.variant["num_updates_per_critic_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

                test_critic_dataloader = create_dataloader(
                    trajectories=self.test_offline_trajs,
                    num_iters=self.variant["num_tests_per_critic_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

            if self.variant["num_updates_per_pretrain_iter"]: # False
                train_dataloader = create_dataloader(
                    trajectories=self.offline_trajs,
                    num_iters=self.variant["num_updates_per_pretrain_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

            # Critic is ImitationTransformer
            # train_outputs:
            # training/train_critic_loss_mean: mean of critic loss during training
            # training/train_critic_loss_std: standard deviation of critic loss during training
            # training/train_critic_mse_mean: mean of critic mean squared error during training
            # training/train_critic_mse_std: standard deviation of critic mean squared error during training
            # training/train_critic_nll_mean: mean of critic negative log likelihood during training
            # training/train_critic_nll_std: standard deviation of critic negative log likelihood during training
            # training/train_rtg_mean: mean of return-to-go during training
            # training/train_rtg_hat_mean: mean of predicted return-to-go during training
            # training/train_rtg_hat_std: standard deviation of predicted return-to-go during training
            # training/train_rtg_delta_std: standard deviation of the difference between predicted and actual return-to-go during training
            # training/eval_critic_loss_mean: mean of critic loss during evaluation
            # training/eval_critic_loss_std: standard deviation of critic loss during evaluation
            # training/eval_critic_mse_mean: mean of critic mean squared error during evaluation
            # training/eval_critic_mse_std: standard deviation of critic mean squared error during evaluation
            # training/eval_critic_nll_mean: mean of critic negative log likelihood during evaluation
            # training/eval_critic_nll_std: standard deviation of critic negative log likelihood during evaluation
            # training/eval_rtg_mean: mean of return-to-go during evaluation
            # training/eval_rtg_hat_mean: mean of predicted return-to-go during evaluation
            # training/eval_rtg_hat_std: standard deviation of predicted return-to-go during evaluation
            # training/eval_rtg_delta_std: standard deviation of the difference between predicted and actual return-to-go during evaluation
            # training/rtg_mean_histogram: histogram of return-to-go means
            # training/rtg_hat_delta_histogram: histogram of differences between predicted and actual return-to-go
            # training/rtg_hat_mean_histogram: histogram of predicted return-to-go means
            # training/rtg_hat_std_histogram: histogram of predicted return-to-go standard deviations
            # training/eval_rtg_mean_histogram: histogram of return-to-go means during evaluation
            # training/eval_rtg_hat_delta_histogram: histogram of differences between predicted and actual return-to-go during evaluation
            # training/eval_rtg_hat_mean_histogram: histogram of predicted return-to-go means during evaluation
            # training/eval_rtg_hat_std_histogram: histogram of predicted return-to-go standard deviations during evaluation
            # time/training: time spent on training
            
            # For PretrainCriticSequenceTrainer 
            #train_outputs = trainer.pretrain_iteration(loss_fn=loss_fn, critic_loss_fn=critic_loss_fn, train_critic_dataloader=train_critic_dataloader, test_critic_dataloader=test_critic_dataloader, train_dataloader=train_dataloader)
            # For SequenceTrainer
            train_outputs = trainer.train_iteration(loss_fn=loss_fn, train_dataloader=train_dataloader)
            if self.variant["num_updates_per_pretrain_iter"]: # 0, False
                eval_outputs, eval_reward = self.evaluate(eval_fns)
            else:
                eval_outputs = {}
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            print(f"Iteration: {self.pretrain_iter}, Total transitions sampled: {self.total_transitions_sampled}")
            for key, value in outputs.items():
                print(f"{key}: {value}")

            self._save_model(
                path_prefix=os.getcwd(),
                is_pretrain_model=True,
            )

            if self.n_ascending_cnt < NUM_PASSED_STEPS: # True
                if outputs["training/eval_critic_loss_mean"] <= self.min_pretrain_loss_mean:
                    self.best_critic_state_dict = deepcopy(self.critic.state_dict())
                    self.best_pretrain_iter = self.pretrain_iter
                if outputs["training/eval_critic_loss_mean"] > self.min_pretrain_loss_mean:
                    self.n_ascending_cnt += 1
                else:
                    self.n_ascending_cnt = 0
                self.min_pretrain_loss_mean = min(outputs["training/eval_critic_loss_mean"], self.min_pretrain_loss_mean)

            self.pretrain_iter += 1

        if self.best_pretrain_iter is not None: # True
            print(f"\n\n\n*** Loading best critic from iter {self.best_pretrain_iter} ***")
            self.critic.load_state_dict(self.best_critic_state_dict)

        return outputs


    def train(self, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):
        print("\n\n\n*** Train ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ] # return--> eval_episodes_fn

        if self.variant["trainer"] == "PretrainCriticSequenceTrainer": # False
            trainer = PretrainCriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                linear_value_coef_start=self.variant["linear_value_coef_start"],
                linear_value_coef_steps=self.variant["linear_value_coef_steps"],
                pretrain_scheduler=self.pretrain_scheduler,
                scheduler=self.scheduler,
                device=self.device,
                action_space=eval_envs.action_space,
            )
        elif self.variant["trainer"] == "CriticSequenceTrainer": # True
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else: # False
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Training Trainer: {type(trainer)}")

        while self.train_iter < self.variant["max_train_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_train_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                gamma=self.variant["gamma"],
                action_range=self.action_range,
                num_workers=self.variant["num_workers"],
            )

            # train_outputs:
            # training/train_loss_mean: mean of training loss
            # training/train_loss_std: standard deviation of training loss
            # training/nll: negative log-likelihood
            # training/entropy: entropy
            # training/mse: mean squared error
            # training/temp_value: temperature value
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                critic_loss_fn=critic_loss_fn,
                value_loss_fn=value_loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns) # eval_fns--> eval_episodes_fn
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            
            print(f"Iteration: {self.pretrain_iter + self.train_iter}")
            print(f"Total transitions sampled: {self.total_transitions_sampled}")
            for key, value in outputs.items():
                print(f"{key}: {value}")

            self._save_model(
                path_prefix=os.getcwd(),
                is_pretrain_model=True,
            )

            current_score = outputs["evaluation/score_mean_gm"]
            if current_score > self.max_d4rl_score:
                self.best_train_iter = self.train_iter
                self.best_model_state_dict = deepcopy(self.model.state_dict())
                self.max_d4rl_score = current_score
                print(f"New best model at iter {self.best_train_iter} with score {self.max_d4rl_score:.4f}")
            elif current_score == self.max_d4rl_score:
                print(f"Matched best score at iter {self.train_iter} with score {current_score:.4f}")

            self.train_iter += 1

        if self.best_train_iter is not None:
            print(f"\n\n\n*** Loading best model from iter {self.best_train_iter} ({self.max_d4rl_score}) ***")
            self.model.load_state_dict(self.best_model_state_dict)

        return outputs

    def online_tuning(self, online_envs, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):

        print("\n\n\n*** Online Finetuning ***")

        if self.variant["use_critic"]:
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Online Finetuning Trainer: {type(trainer)}")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ]

        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                gamma=self.variant["gamma"],
                action_range=self.action_range,
                num_workers=self.variant["num_workers"],
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                critic_loss_fn=critic_loss_fn,
                value_loss_fn=value_loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            
            print("Metrics:")
            for key, value in outputs.items():
                print(f"{key}: {value}")
            print(f"Iteration: {self.pretrain_iter + self.train_iter + self.online_iter}")
            print(f"Total transitions sampled: {self.total_transitions_sampled}")

            self._save_model(
                path_prefix=os.getcwd(),
                is_pretrain_model=False,
            )

            self.online_iter += 1

        return outputs

    def final_evaluate(self, eval_envs):
        eval_start = time.time()
        self.model.eval()
        outputs = {}

        # evaluation on eval_rtg
        for eval_rtg_coef in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            eval_rtg = self.variant["eval_rtg"] * eval_rtg_coef
            print(f"Evaluating on eval_rtg_coef: {eval_rtg_coef}, ({eval_rtg})")
            input_eval_fns =  create_vec_eval_episodes_fn(
                    env_name=self.variant["env"],
                    vec_env=eval_envs,
                    eval_rtg=eval_rtg,
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                    use_mean=True,
                    reward_scale=self.reward_scale,
                    no_reward=self.variant["no_reward"],
                    delayed_reward=self.variant["delayed_reward"],
                ) # return--> eval_episodes_fn
            
            eval_fns = [input_eval_fns] * int(100 / self.variant["num_eval_rollouts"])
            score_mean_gms = []
            for eval_fn in eval_fns:
                o = eval_fn(self.model)
                score_mean_gms.append(o["evaluation/score_mean_gm"])
            outputs[f"final_evaluation/score_mean_gm_rtg{eval_rtg_coef}"] = np.mean(score_mean_gms)

        # evaluation on eval_context_len
        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ] * int(100 / self.variant["num_eval_rollouts"])
        for eval_context_length in [1, 5, 10, 15, 20]:
            print(f"Evaluating on eval_context_length: {eval_context_length}")
            if eval_context_length > self.variant["K"]:
                break
            self.model.eval_context_length = eval_context_length
            score_mean_gms = []
            for eval_fn in eval_fns:
                o = eval_fn(self.model)
                score_mean_gms.append(o["evaluation/score_mean_gm"])
            outputs[f"final_evaluation/score_mean_gm_ctx{eval_context_length}"] = np.mean(score_mean_gms)

        outputs["time/final_evaluation"] = time.time() - eval_start

        print("Metrics:")
        for key, value in outputs.items():
            print(f"{key}: {value}")
        print(f"Iteration: {self.pretrain_iter + self.train_iter + self.online_iter}")
        print(f"Total transitions sampled: {self.total_transitions_sampled}")

                   

        return outputs

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            for _ in range(int(self.variant["num_eval_episodes"] / self.variant["num_eval_rollouts"])):
                o = eval_fn(self.model)
                for k, v in o.items():
                    if k not in outputs:
                        outputs[k] = [v]
                    else:
                        outputs[k].append(v)
        for k, v in outputs.items():
            outputs[k] = np.mean(v)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def run(self):

        set_seed_everywhere(seed)

        import d4rl

        def critic_loss_fn(rtg_hat,rtg,attention_mask,**kwargs): 

            # inputs: (return_preds, rtg_discounted_target[:, :-1], padding_mask, shuffled_return_preds=shuffled_return_preds)
            mse, nll, asymmetric_l2_loss = [torch.tensor(0, device=rtg.device) for _ in range(3)]
            if self.variant["critic_loss"] == "nll_infonce":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')

                shuffled_rtg_hat = kwargs["shuffled_return_preds"]
                # pos_nll = (rtg_hat.nll(rtg)[attention_mask > 0] * (-1)).exp().sum().log()
                # neg_nll = (shuffled_rtg_hat.nll(rtg)[:, -1][attention_mask[:, -1] > 0] * (-1)).exp().sum().log()
                # loss = - pos_nll + neg_nll / rtg.shape[1] * 0

                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
                shuffled_nll = shuffled_rtg_hat.nll(0)[:, -1][attention_mask[:, -1] > 0].mean()
                loss = nll + shuffled_nll / rtg.shape[1] * 1
                
                # Add noise for neg_nll
                # TODO: improve for all positions

            elif self.variant["critic_loss"] == "expectile":
                tau = self.variant["tau1"]
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]    # TODO: test with mean() or not
                nll = torch.mean((torch.abs(tau - (u < 0).float()).squeeze(-1) * rtg_hat.nll(rtg))[attention_mask > 0]) * (1 / max(tau, 1 - tau))
                loss = nll
            elif self.variant["critic_loss"] == "nll":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
                loss = nll
            elif self.variant["critic_loss"] == "mse":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                mse = mse / rtg[attention_mask > 0].mean().detach() ** 2    # scale critic loss 
                nll = torch.tensor(0, device=rtg.device)
                loss = mse
            else:
                raise NotImplementedError()

            return (
                loss,
                mse,
                nll,
                rtg[attention_mask > 0],
                rtg_hat.mu[attention_mask > 0],
                rtg_hat.sigma[attention_mask > 0],
            )

        def value_loss_fn(
            rtg_hat,
            rtg,
            attention_mask,
        ):
            mse, nll, asymmetric_l2_loss = [torch.tensor(0, device=rtg.device) for _ in range(3)]
            if self.variant["value_loss"] == "expectile":
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]
                asymmetric_l2_loss = torch.mean(torch.abs(self.variant["tau"] - (u < 0).float()) * u**2) \
                    * (1 / max(self.variant["tau"], 1 - self.variant["tau"]))
            elif self.variant["value_loss"] == "gumbel":
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]
                asymmetric_l2_loss = torch.mean((-u).exp() + u)
            elif self.variant["value_loss"] == "nll":
                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
            elif self.variant["value_loss"] == "mse":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                mse = mse / rtg[attention_mask > 0].mean().detach() ** 2
            else:
                raise NotImplementedError()
            
            loss = mse + nll + asymmetric_l2_loss

            return (
                loss,
                mse,
                nll,
                rtg[attention_mask > 0],
                rtg_hat.mu[attention_mask > 0],
                rtg_hat.sigma[attention_mask > 0],
            )

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            if self.variant["action_loss"] == "mse":
                if self.discrete_action:
                    a_hat = a_hat_dist
                else:
                    a_hat = a_hat_dist.mean
                mse_loss = torch.nn.functional.mse_loss(a_hat[attention_mask > 0], a[attention_mask > 0], reduction='mean')
                log_likelihood = torch.tensor(0, device=a.device)
            elif self.variant["action_loss"] == "nll":
                mse_loss = torch.tensor(0, device=a.device)
                log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
            else:
                raise NotImplementedError()

            if self.variant["action_loss"] == "nll_entropy":
                entropy = a_hat_dist.entropy().mean()
            else:
                entropy = torch.tensor(self.target_entropy, device=a.device).detach()

            loss = -(log_likelihood + entropy_reg * entropy) + mse_loss

            return (
                loss,
                -log_likelihood,
                entropy,
                mse_loss,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            if "bernoulli-bandit" in env_name:
                def make_env_fn():
                    from decision_transformer.envs.bernoulli_bandit import BernoulliBanditEnv, BernoulliBanditWrapper
                    bernoulli_prob = 1 - float(env_name[-3:])
                    env = BernoulliBanditEnv(
                        num_arms=2,
                        reward_power=3.0,
                        reward_scale=0.9,
                        generation_seed=0,
                        bernoulli_prob=bernoulli_prob,
                        loop=False,
                    )
                    env = BernoulliBanditWrapper(env)
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                    else:
                        pass
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    
                    print(f"Make env {env_name} with bernoulli_prob: {bernoulli_prob}")
                    return env

                return make_env_fn
            else:
                def make_env_fn():
                    import d4rl

                    env = gym.make(env_name)
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                    else:
                        pass
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    return env

                return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        
        eval_envs = SubprocVecEnv( # stable baseline paralel processing--> DUsUN KALDIRABILIR MIsIN?
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_rollouts"])
            ]
        )

        outputs = {}
        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]: # 0, False
            
            # outputs:
            # time/total: total time elapsed
            # training/train_critic_loss_mean: mean of critic loss during training
            # training/train_critic_loss_std: standard deviation of critic loss during training
            # training/train_critic_mse_mean: mean of critic mean squared error during training
            # training/train_critic_mse_std: standard deviation of critic mean squared error during training
            # training/train_critic_nll_mean: mean of critic negative log likelihood during training
            # training/train_critic_nll_std: standard deviation of critic negative log likelihood during training
            # training/train_rtg_mean: mean of return-to-go during training
            # training/train_rtg_hat_mean: mean of predicted return-to-go during training
            # training/train_rtg_hat_std: standard deviation of predicted return-to-go during training
            # training/train_rtg_delta_std: standard deviation of the difference between predicted and actual return-to-go during training
            # training/eval_critic_loss_mean: mean of critic loss during evaluation
            # training/eval_critic_loss_std: standard deviation of critic loss during evaluation
            # training/eval_critic_mse_mean: mean of critic mean squared error during evaluation
            # training/eval_critic_mse_std: standard deviation of critic mean squared error during evaluation
            # training/eval_critic_nll_mean: mean of critic negative log likelihood during evaluation
            # training/eval_critic_nll_std: standard deviation of critic negative log likelihood during evaluation
            # training/eval_rtg_mean: mean of return-to-go during evaluation
            # training/eval_rtg_hat_mean: mean of predicted return-to-go during evaluation
            # training/eval_rtg_hat_std: standard deviation of predicted return-to-go during evaluation
            # training/eval_rtg_delta_std: standard deviation of the difference between predicted and actual return-to-go during evaluation
            # training/rtg_mean_histogram: histogram of return-to-go means
            # training/rtg_hat_delta_histogram: histogram of differences between predicted and actual return-to-go
            # training/rtg_hat_mean_histogram: histogram of predicted return-to-go means
            # training/rtg_hat_std_histogram: histogram of predicted return-to-go standard deviations
            # training/eval_rtg_mean_histogram: histogram of return-to-go means during evaluation
            # training/eval_rtg_hat_delta_histogram: histogram of differences between predicted and actual return-to-go during evaluation
            # training/eval_rtg_hat_mean_histogram: histogram of predicted return-to-go means during evaluation
            # training/eval_rtg_hat_std_histogram: histogram of predicted return-to-go standard deviations during evaluation
            # time/training: time spent on training
            
            outputs = self.pretrain(eval_envs, loss_fn, critic_loss_fn, value_loss_fn) # Train critic network which is ImitationTransformer

        if self.variant["max_train_iters"]: # True
            outputs = self.train(eval_envs, loss_fn, critic_loss_fn, value_loss_fn) # Same as above "pretrain", Train critic network again which is ImitationTransformer

        if self.variant["max_online_iters"]: # 0, False
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            outputs = self.online_tuning(online_envs, eval_envs, loss_fn, critic_loss_fn, value_loss_fn)
            online_envs.close()

        if True:
            outputs = self.final_evaluate(eval_envs) # eval_envs is stable baseline parallel processing

        eval_envs.close()

        if not self.variant["no_wandb"]:
            wandb.finish()
            print(f"wandb finialized")

        return outputs


def train(param):
    args = param["args"]
    for k in param.keys():
        if k not in ["args", "tasks"]:
            args.__setattr__(k, param[k])
    if "tasks" in param:
        for k in param["tasks"].keys():
            assert k in args, f"no {k} in args"
            args.__setattr__(k, param["tasks"][k])
            
    set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    outputs = experiment.run()
    
    return {
        "max_d4rl_score": max([v for k, v in outputs.items() if "score_mean_gm" in k]),
    }


if __name__ == "__main__":
    # Default values for the original variables
    seed = 10
    env = "halfcheetah-medium-v2"
    no_reward = False
    delayed_reward = False
    K = 20
    embed_dim = 512
    n_layer = 4
    n_head = 4
    activation_function = "relu"
    dropout = 0.1
    eval_context_length = 5
    ordering = 0
    critic_top_percent = 1.0
    gamma = 1.0
    n_critic_layer = 4
    trainer = "CriticSequenceTrainer" # "PretrainCriticSequenceTrainer" , "SequenceTrainer" 
    action_loss = "mse" 
    critic_loss = "mse"  
    value_loss = "mse"  
    value_coef = 0.0
    linear_value_coef_start = 0
    linear_value_coef_steps = 0
    beta = 1.0
    tau = 0.5
    tau1 = 0.5
    eval_rtg = 3600
    num_eval_rollouts = 2 #10
    num_eval_episodes = 10
    init_temperature = 0.1
    batch_size =16 #256
    learning_rate = 1e-4
    weight_decay = 5e-4
    warmup_steps = 10000
    max_pretrain_iters = 0 #100
    num_updates_per_critic_iter = 2 #1200
    num_tests_per_critic_iter = 200
    num_updates_per_pretrain_iter = 0
    max_train_iters = 2 # 100
    num_updates_per_train_iter = 800
    max_online_iters = 0 # 1500
    online_rtg = 7200
    num_online_rollouts = 1
    replay_size = 1000
    num_updates_per_online_iter = 300
    eval_interval = 10
    device = "cuda"
    log_to_tb = True
    save_dir = "./exp"
    exp_name = "default"
    no_wandb = False
    num_workers=1

    


    

    variant = {
            "args": None,
            "seed": seed,
            "env": env,
            "no_reward": no_reward,
            "delayed_reward": delayed_reward,
            "K": K,
            "embed_dim": embed_dim,
            "n_layer": n_layer,
            "n_head": n_head,
            "activation_function": activation_function,
            "dropout": dropout,
            "eval_context_length": eval_context_length,
            "ordering": ordering,
            "critic_top_percent": critic_top_percent,
            "gamma": gamma,
            "n_critic_layer": n_critic_layer,
            "trainer": trainer,
            "action_loss": action_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            "value_coef": value_coef,
            "linear_value_coef_start": linear_value_coef_start,
            "linear_value_coef_steps": linear_value_coef_steps,
            "beta": beta,
            "tau": tau,
            "tau1": tau1,
            "eval_rtg": eval_rtg,
            "num_eval_rollouts": num_eval_rollouts,
            "num_eval_episodes": num_eval_episodes,
            "init_temperature": init_temperature,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "max_pretrain_iters": max_pretrain_iters,
            "num_updates_per_critic_iter": num_updates_per_critic_iter,
            "num_tests_per_critic_iter": num_tests_per_critic_iter,
            "num_updates_per_pretrain_iter": num_updates_per_pretrain_iter,
            "max_train_iters": max_train_iters,
            "num_updates_per_train_iter": num_updates_per_train_iter,
            "max_online_iters": max_online_iters,
            "online_rtg": online_rtg,
            "num_online_rollouts": num_online_rollouts,
            "replay_size": replay_size,
            "num_updates_per_online_iter": num_updates_per_online_iter,
            "eval_interval": eval_interval,
            "device": device,
            "log_to_tb": log_to_tb,
            "save_dir": save_dir,
            "exp_name": exp_name,
            "no_wandb": no_wandb,
            "num_workers": num_workers
    }



    set_seed_everywhere(seed)
   
   # Only for the first run
   # _=create_dataset()

    experiment = Experiment(variant)

    print("=" * 50)
    outputs = experiment.run()
    
    print(f"Max D4RL score: {max([v for k, v in outputs.items() if 'score_mean_gm' in k])}")
