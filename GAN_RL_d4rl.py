import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import d4rl
import torch.nn.functional as F
import math
import numpy.linalg as np_linalg
import scipy.linalg


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReplayBuffer(object):
	def __init__(self, state_dim=10, action_dim=4):
		self.storage = dict()
		self.storage['observations'] = np.zeros((1000000, state_dim), np.float32)
		self.storage['next_observations'] = np.zeros((1000000, state_dim), np.float32)
		self.storage['actions'] = np.zeros((1000000, action_dim), np.float32)
		self.storage['rewards'] = np.zeros((1000000, 1), np.float32)
		self.storage['terminals'] = np.zeros((1000000, 1), np.float32)
		self.storage['bootstrap_mask'] = np.zeros((10000000, 4), np.float32)
		self.buffer_size = 1000000
		self.ctr = 0

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage['observations'][self.ctr] = data[0]
		self.storage['next_observations'][self.ctr] = data[1]
		self.storage['actions'][self.ctr] = data[2]
		self.storage['rewards'][self.ctr] = data[3]
		self.storage['terminals'][self.ctr] = data[4]
		self.ctr += 1
		self.ctr = self.ctr % self.buffer_size

	def sample(self, batch_size, with_data_policy=False):
		ind = np.random.randint(0, self.storage['observations'].shape[0], size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		s = self.storage['observations'][ind]
		a = self.storage['actions'][ind]
		r = self.storage['rewards'][ind]
		s2 = self.storage['next_observations'][ind]
		d = self.storage['terminals'][ind]
		mask = self.storage['bootstrap_mask'][ind]

		if with_data_policy:
				data_mean = self.storage['data_policy_mean'][ind]
				data_cov = self.storage['data_policy_logvar'][ind]

				return (np.array(s), 
						np.array(s2), 
						np.array(a), 
						np.array(r).reshape(-1, 1), 
						np.array(d).reshape(-1, 1),
						np.array(mask),
						np.array(data_mean),
						np.array(data_cov))

		return (np.array(s), 
				np.array(s2), 
				np.array(a), 
				np.array(r).reshape(-1, 1), 
				np.array(d).reshape(-1, 1),
				np.array(mask))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename, bootstrap_dim=None):
		"""Deprecated, use load_hdf5 in main.py with the D4RL environments""" 
		with gzip.open(filename, 'rb') as f:
				self.storage = pickle.load(f)
		
		sum_returns = self.storage['rewards'].sum()
		num_traj = self.storage['terminals'].sum()
		if num_traj == 0:
				num_traj = 1000
		average_per_traj_return = sum_returns/num_traj
		print ("Average Return: ", average_per_traj_return)
		# import ipdb; ipdb.set_trace()
		
		num_samples = self.storage['observations'].shape[0]
		if bootstrap_dim is not None:
				self.bootstrap_dim = bootstrap_dim
				bootstrap_mask = np.random.binomial(n=1, size=(1, num_samples, bootstrap_dim,), p=0.8)
				bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
				self.storage['bootstrap_mask'] = bootstrap_mask[:num_samples]



def normalized_evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        all_rewards.append(avg_reward)
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}, D4RL Score: {d4rl_score}")
    print("---------------------------------------")
    return avg_reward, std_rewards, median_reward, d4rl_score

class RegularActor(nn.Module):
    """A probabilistic actor which does regular stochastic mapping of actions from states"""
    def __init__(self, state_dim, action_dim, max_action,):
        super(RegularActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        
        std_a = torch.exp(log_std_a)
        z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device) 
        return self.max_action * torch.tanh(z)

    def sample_multiple(self, state, num_sample=10):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        
        std_a = torch.exp(log_std_a)
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        z = mean_a.unsqueeze(1) +\
             std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
        return self.max_action * torch.tanh(z), z 

    def log_pis(self, state, action=None, raw_action=None):
        """Get log pis for the model."""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean_a = self.mean(a)
        log_std_a = self.log_std(a)
        std_a = torch.exp(log_std_a)
        normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
        if raw_action is None:
            raw_action = atanh(action)
        else:
            action = torch.tanh(raw_action)
        log_normal = normal_dist.log_prob(raw_action)
        log_pis = log_normal.sum(-1)
        log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
        return log_pis



class EnsembleCritic(nn.Module):
    """ Critic which does have a network of 4 Q-functions"""
    def __init__(self, num_qs, state_dim, action_dim):
        super(EnsembleCritic, self).__init__()
        
        self.num_qs = num_qs

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        # self.l7 = nn.Linear(state_dim + action_dim, 400)
        # self.l8 = nn.Linear(400, 300)
        # self.l9 = nn.Linear(300, 1)

        # self.l10 = nn.Linear(state_dim + action_dim, 400)
        # self.l11 = nn.Linear(400, 300)
        # self.l12 = nn.Linear(300, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def q_all(self, state, action, with_var=False):
        all_qs = []
        
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        # q3 = F.relu(self.l7(torch.cat([state, action], 1)))
        # q3 = F.relu(self.l8(q3))
        # q3 = self.l9(q3)

        # q4 = F.relu(self.l10(torch.cat([state, action], 1)))
        # q4 = F.relu(self.l11(q4))
        # q4 = self.l12(q4)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module): # To handle out of distribution samples
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
        
        u = self.decode(state, z)

        return u, mean, std
    
    def decode_softplus(self, state, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
        
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        
    def decode(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def decode_bc(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_bc_test(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def decode_multiple(self, state, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)), self.d3(a)


class BEAR(object):
    def __init__(self, num_qs, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False, version=0, lambda_=0.4,
                 threshold=0.05, mode='auto', num_samples_match=10, mmd_sigma=10.0,
                 lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='gaussian'):# laplacian 
        latent_dim = action_dim * 2
        self.actor = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = RegularActor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.delta_conf = delta_conf
        self.use_bootstrap = use_bootstrap
        self.version = version
        self._lambda = lambda_
        self.threshold = threshold
        self.mode = mode
        self.num_qs = num_qs
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_kl = use_kl
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type
        
        if self.mode == 'auto':
            # Use lagrange multipliers on the constraint if set to auto mode 
            # for the purpose of maintaing support matching at all times
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2,], lr=1e-3)

        self.epoch = 0

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss
    
    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def kl_loss(self, samples1, state, sigma=0.2):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)
    
    def entropy_loss(self, samples1, state, sigma=0.2):
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        # print (samples1_log_prob.min(), samples1_log_prob.max())
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return (samples1_prob).mean(1)
    
    def select_action(self, state):      
        """When running the actor, we just select action based on the max of the Q-function computed over
            samples from the policy -- which biases things to support."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            state_np, next_state_np, action, reward, done, mask = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)
            mask            = torch.FloatTensor(mask).to(device)
            
            # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training: In this step, we explicitly compute the actions 
            with torch.no_grad():
                # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
                
                # Compute value of perturbed actions sampled from the VAE
                target_Qs = self.critic_target(state_rep, self.actor_target(state_rep))

                # Soft Clipped Double Q-learning 
                target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0]
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
                target_Q = reward + done * discount * target_Q

            current_Qs = self.critic(state, action, with_var=False)
            if self.use_bootstrap: 
                critic_loss = (F.mse_loss(current_Qs[0], target_Q, reduction='none') * mask[:, 0:1]).mean() +(F.mse_loss(current_Qs[1], target_Q, reduction='none') * mask[:, 1:2]).mean() 
            else:
                critic_loss = F.mse_loss(current_Qs[0], target_Q) + F.mse_loss(current_Qs[1], target_Q) #+ F.mse_loss(current_Qs[2], target_Q) + F.mse_loss(current_Qs[3], target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Action Training
            # If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
            num_samples = self.num_samples_match
            sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
            actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_samples)#  num)

            # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
            if self.use_kl:
                mmd_loss = self.kl_loss(raw_sampled_actions, state)
            else:
                if self.kernel_type == 'gaussian':
                    mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
                else:
                    mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

            action_divergence = ((sampled_actions - actor_actions)**2).sum(-1)
            raw_action_divergence = ((raw_sampled_actions - raw_actor_actions)**2).sum(-1)

            ## Update through TD3 style
            # Compute the Q-values and uncertainty (std_q) for the actor's actions
            critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
            critic_qs = self.critic.q_all(state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples*state.size(0), state.size(1)), actor_actions.permute(1, 0, 2).contiguous().view(num_samples*actor_actions.size(0), actor_actions.size(2)))
            critic_qs = critic_qs.view(self.num_qs, num_samples, actor_actions.size(0), 1)
            critic_qs = critic_qs.mean(1)
            # Compute the standard deviation across critics (uncertainty)
            std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)

            if not self.use_ensemble: # Determine whether to include uncertainty penalty
                std_q = torch.zeros_like(std_q).to(device)
                
            # Select the appropriate aggregation of Q-values
            if self.version == '0':
                critic_qs = critic_qs.min(0)[0]
            elif self.version == '1':
                critic_qs = critic_qs.max(0)[0]
            elif self.version == '2':
                critic_qs = critic_qs.mean(0)

            # We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
            # Compute the actor loss, including the uncertainty penalty
            if self.epoch >= 20: # Only start adding the uncertainty penalty after 20 epochs
                if self.mode == 'auto':
                    #actor_loss = (-critic_qs +self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +self.log_lagrange2.exp() * mmd_loss).mean()
                    actor_loss = (-critic_qs + self._lambda * (math.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q + self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = (-critic_qs +\
                        self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
                        100.0*mmd_loss).mean()      # This coefficient is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
            else:
                # If we are still in the initial epochs, only do support matching
                # Warm-up period without uncertainty penalty
                if self.mode == 'auto':
                    actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
                else:
                    actor_loss = 100.0*mmd_loss.mean()

            std_loss = self._lambda*(np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q.detach() 

            self.actor_optimizer.zero_grad()
            if self.mode =='auto':
                actor_loss.backward(retain_graph=True)
            else:
                actor_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Threshold for the lagrange multiplier
            thresh = 0.05
            if self.use_kl:
                thresh = -2.0

            if self.mode == 'auto':
                #lagrange_loss = (-critic_qs +self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * (std_q) +self.log_lagrange2.exp() * (mmd_loss - thresh)).mean()
                lagrange_loss = (-critic_qs + self._lambda * (math.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q + self.log_lagrange2.exp() * (mmd_loss - self.threshold)).mean()
                self.lagrange2_opt.zero_grad()
                #(-lagrange_loss).backward() # orj
                # Detach the tensor to avoid in-place operations and create a new computation graph
                lagrange_loss_detached = lagrange_loss.detach()
                # Create a new tensor with requires_grad=True
                lagrange_loss_new = -lagrange_loss_detached.requires_grad_()
                # Compute the gradients
                lagrange_loss_new.backward(retain_graph=True)
                # (-lagrange_loss).backward()
                # self.lagrange1_opt.step()
                self.lagrange2_opt.step() 
                self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)   
            
            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
      
            
            if it % 100 == 0:
                print("Policy Performance:")
                print(f"Training epoch: {self.epoch} Iteration: {it}")             
                print(f"VAE Loss: {vae_loss.item():.4f}")
                print(f"Critic Loss: {critic_loss.item():.4f}")
                print(f"Actor Loss: {actor_loss.item():.4f}")
                print(f"MMD Loss: {mmd_loss.mean().item():.4f}")
                print(f"Std Q: {std_q.mean().item():.4f}")
                if self.mode == 'auto':
                    print(f"Lagrange Loss: {lagrange_loss.item():.4f}")
                print("--------------------")

             # Add checks for nan values
            if math.isnan(critic_loss.item()) or math.isnan(actor_loss.item()):
                print(f"NaN detected at iteration {it}. Stopping training.")
                #break
        
        self.epoch = self.epoch + 1

      

def weighted_mse_loss(inputs, target, weights):
    return torch.mean(weights * (inputs - target)**2)

def load_and_preprocess_d4rl_data(env_name):
    # Load the environment
    env = gym.make(env_name)
    
    # Get the dataset
    dataset = env.get_dataset()
    
    # Extract states and actions
    states = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_observations']
    dones = dataset['terminals']
    
    # Combine states and actions for GAN training
    combined_data = np.concatenate((states, actions), axis=1)
    
    # Normalize the data to [-1, 1] range due to the tanh activation function in the generator
    data_min = combined_data.min(axis=0)
    data_max = combined_data.max(axis=0)
    normalized_data = 2 * (combined_data - data_min) / (data_max - data_min) - 1
    
    return env, normalized_data, dataset, (data_min, data_max)



def create_combined_dataset(combined_data, normalization_params, state_dim, original_dataset):
        # Denormalize the combined data
        data_min, data_max = normalization_params
        denormalized_data = (combined_data + 1) / 2 * (data_max - data_min) + data_min

        # Split combined data back into states and actions
        combined_states = denormalized_data[:, :state_dim]
        combined_actions = denormalized_data[:, state_dim:]

        # Create a new dataset with combined data
        combined_dataset = {
            'observations': combined_states,
            'actions': combined_actions,
            'rewards': np.concatenate((original_dataset['rewards'], original_dataset['rewards'])),
            'next_observations': np.concatenate((original_dataset['next_observations'], original_dataset['next_observations'])),
            'terminals': np.concatenate((original_dataset['terminals'], original_dataset['terminals']))
        }
        return combined_dataset


# Generator
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh() # range [-1, 1]
        )

    def forward(self, x):
        return self.model(x.to(self.model[0].weight.device))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # range [0, 1]
        )

    def forward(self, x):
        return self.model(x)



# GAN training function
def train_gan(generator, discriminator, data, epochs=1000, batch_size=32):
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    g_optimizer = optim.Adam(generator.parameters())
    d_optimizer = optim.Adam(discriminator.parameters())

    for epoch in range(epochs):
        #### Train Discriminator ####
        # Generate random noise from gaussian distribution as input for the generator
        noise1 = torch.randn(batch_size, generator.model[0].in_features, device=device)
        # Generate fake data
        generated_data = generator(noise1).detach()  # detach to avoid training generator here
        '''
        We use the .detach() method to prevent the generated data from being used for training the generator.
        This is because we only want to train the discriminator on real and fake data, not on the generated data.
        '''
        
        
        # Sample real data from the dataset
        real_data = torch.FloatTensor(data[np.random.randint(0, data.shape[0], batch_size)]).to(device)

        # Train Discriminator on real data
        d_optimizer.zero_grad()  # Reset gradients
        real_output = discriminator(real_data)  # Forward pass with real data
        # Calculate loss for real data (target: 1 = real)
        d_loss_real = criterion(real_output, torch.ones_like(real_output))
        
        # Train Discriminator on fake data
        fake_output = discriminator(generated_data)  # Forward pass with fake data
        # Calculate loss for fake data (target: 0 = fake)
        d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
        
        # Calculate total loss for discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()   # Backward pass for discriminator
        d_optimizer.step()  # Update discriminator weights


        #### Train Generator ####
        g_optimizer.zero_grad()  # Reset gradients
        noise2 = torch.randn(batch_size, generator.model[0].in_features)  # New noise for generator
        generated_data = generator(noise2)  # Generate fake data
        
        g_output = discriminator(generated_data)  # Discriminator's opinion on fake data
        '''
        We don't use detach() here because we want the gradients to flow through 
        both the discriminator and the generator. 
        The goal is to update the generator's weights based on how well it fools the discriminator.
        '''
        # Calculate generator loss (target: 1 = fool discriminator)
        g_loss = criterion(g_output, torch.ones_like(g_output))
        g_loss.backward()  # Backward pass for generator
        '''
        When we call g_loss.backward(), 
        the gradients are computed for both the generator and the discriminator parameters.
        '''
        g_optimizer.step()  # Update generator weights
        '''
        The key point is that we only update the generator's parameters using g_optimizer.step(). 
        We don't call d_optimizer.step() during the generator training phase.
        '''

        
        print(f"GAN Training Progress: Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    
    return generator, discriminator


def fill_replay_buffer_with_combined_data(replay_buffer, combined_data, state_dim, action_dim):
    """
    Fill the replay buffer with combined real and synthetic data.
    The combined_data is a dictionary containing 'observations', 'actions', 'rewards', 'next_observations', and 'terminals'.
    """
    num_samples = len(combined_data['observations'])
    print("Replay Buffering...")
    print(f"Number of samples: {num_samples}")
    for i in range(num_samples - 1):
        state = combined_data['observations'][i]
        action = combined_data['actions'][i]
        next_state = combined_data['next_observations'][i]
        reward = combined_data['rewards'][i]
        done = combined_data['terminals'][i]

        # Store this transition in the replay buffer
        replay_buffer.add((state, next_state, action, reward, done))
    print("Replay Buffer filled")
    return replay_buffer

if __name__ == "__main__":
    
    env_name = "halfcheetah-random-v2"
    env, normalized_data, original_dataset, normalization_params = load_and_preprocess_d4rl_data(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # GAN Model Training
    input_dim = normalized_data.shape[1]  # Noise dimension
    output_dim = state_dim + action_dim
    # Build the generator and discriminator models
    generator = Generator(input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # Train the GAN using the normalized data
    generator, discriminator = train_gan(generator, discriminator, normalized_data)

    
    # Generate synthetic data using the trained GAN
    noise = torch.randn(len(normalized_data), input_dim, device=device)
    synthetic_data = generator(noise).detach().cpu().numpy()
    synthetic_data = np.clip(synthetic_data, -1, 1)
    
    # Combine real and synthetic data
    combined_data = np.vstack((normalized_data, synthetic_data))

    # Denormalize the combined data
    combined_dataset = create_combined_dataset(combined_data, normalization_params, state_dim, original_dataset)


    ######### Training RL Model with GAN-generated data + original data #########
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
    replay_buffer_with_gan_denormalized = fill_replay_buffer_with_combined_data(replay_buffer, combined_dataset, state_dim, action_dim)
    
    # Initialize BEAR policy
    max_action = 1.0  # Since the data is normalized, max_action should be 1.0
    policy = BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
                version='0', lambda_=0.5, threshold=0.05, mode='auto', num_samples_match=10,
                mmd_sigma=20.0, lagrange_thresh=10.0, use_kl=False, use_ensemble=False, kernel_type='gaussian')

    # Training loop
    max_timesteps =10 #1e6
    eval_freq = 10 #5e3
    training_iters = 0

    average_returns = []
    grad_steps = 0
    max_grad_steps = int(max_timesteps*eval_freq)
    while grad_steps  < max_grad_steps:  
        policy.train(replay_buffer_with_gan_denormalized, iterations=int(eval_freq))
        grad_steps += int(eval_freq)
        ret_eval, std_ret, median_ret, d4rl_score = normalized_evaluate_policy(policy, env)
        

        training_iters += eval_freq
        average_returns.append(ret_eval)

        print(f"Average Return: {ret_eval}, std: {std_ret}, median: {median_ret}, D4RL Score: {d4rl_score}")

    # Calculate average, std and median of the training loop
    avg_return = np.mean(average_returns)
    std_return = np.std(average_returns)
    median_return = np.median(average_returns)

    # Write results to a text file
    with open('BEAR_training_results_with_gan.txt', 'w') as f:
        f.write(f"Training Results for {env_name}:\n")
        f.write(f"Average Return: {avg_return:.2f}\n")
        f.write(f"Standard Deviation: {std_return:.2f}\n")
        f.write(f"Median Return: {median_return:.2f}\n")
        f.write(f"D4RL Score: {d4rl_score:.2f}\n")

    print("BEAR Training results with GAN have been written to 'BEAR_training_results_with_gan.txt'")
    


    
    ######### Evaluate GAN Model #########
    
    data_scaled = normalized_data
    num_samples = 1000
    noise = torch.randn(num_samples, generator.model[0].in_features, device=device)
    generated_data = generator(noise).detach().cpu().numpy()

    # Calculate mean and standard deviation for real and generated data
    real_mean, real_std = np.mean(data_scaled, axis=0), np.std(data_scaled, axis=0)
    gen_mean, gen_std = np.mean(generated_data, axis=0), np.std(generated_data, axis=0)
    gan_figure_evaluation = False # Time consuming
    
    if gan_figure_evaluation:
    
        # Plot histograms for comparison
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('GAN Evaluation: Real vs Generated Data Distribution')

        axs[0, 0].hist(data_scaled[:, 0], bins=50, alpha=0.5, label='Real')
        axs[0, 0].hist(generated_data[:, 0], bins=50, alpha=0.5, label='Generated')
        axs[0, 0].set_title('Distribution of First Feature')
        axs[0, 0].legend()

        axs[0, 1].hist(data_scaled[:, -1], bins=50, alpha=0.5, label='Real')
        axs[0, 1].hist(generated_data[:, -1], bins=50, alpha=0.5, label='Generated')
        axs[0, 1].set_title('Distribution of Last Feature')
        axs[0, 1].legend()

        # Plot mean and standard deviation comparison
        feature_indices = range(data_scaled.shape[1])
        axs[1, 0].errorbar(feature_indices, real_mean, yerr=real_std, fmt='o', label='Real')
        axs[1, 0].errorbar(feature_indices, gen_mean, yerr=gen_std, fmt='o', label='Generated')
        axs[1, 0].set_title('Mean and Std Dev Comparison')
        axs[1, 0].set_xlabel('Feature Index')
        axs[1, 0].set_ylabel('Value')
        axs[1, 0].legend()

        # Calculate and plot autocorrelation
        real_autocorr = np.array([np.correlate(data_scaled[:, i], data_scaled[:, i], mode='full') for i in range(data_scaled.shape[1])])
        gen_autocorr = np.array([np.correlate(generated_data[:, i], generated_data[:, i], mode='full') for i in range(generated_data.shape[1])])
    
        real_autocorr = real_autocorr[:, real_autocorr.shape[1]//2:] / real_autocorr[:, real_autocorr.shape[1]//2:].max()
        gen_autocorr = gen_autocorr[:, gen_autocorr.shape[1]//2:] / gen_autocorr[:, gen_autocorr.shape[1]//2:].max()

        axs[1, 1].plot(real_autocorr.mean(axis=0)[:50], label='Real')
        axs[1, 1].plot(gen_autocorr.mean(axis=0)[:50], label='Generated')
        axs[1, 1].set_title('Average Autocorrelation')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('Autocorrelation')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    # Calculate Frechet Inception Distance (FID)
    import numpy.linalg as linalg
    def calculate_fid(real_features, generated_features):
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        
        diff = mu1 - mu2
        #covmean = linalg.inv(linalg.sqrtm(linalg.inv(sigma1.dot(sigma2))))
        covmean = np_linalg.inv(scipy.linalg.sqrtm(np_linalg.inv(sigma1.dot(sigma2))))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        return fid

    fid_score = calculate_fid(data_scaled, generated_data)
    print(f"Frechet Inception Distance (FID): {fid_score}")

    
    
    
    ######### Training RL Model without GAN training #########
    
    def load_hdf5_mujoco(dataset, replay_buffer):
        """
        Use this loader for the gym mujoco environments
        """
        all_obs = dataset['observations']
        all_act = dataset['actions']
        N = min(all_obs.shape[0], 2000000)
        _obs = all_obs[:N]
        _actions = all_act[:N]
        _next_obs = np.concatenate([all_obs[1:N,:], np.zeros_like(_obs[0])[np.newaxis,:]], axis=0)
        _rew = dataset['rewards'][:N]
        _done = dataset['terminals'][:N]

        replay_buffer.storage['observations'] = _obs
        replay_buffer.storage['next_observations'] = _next_obs
        replay_buffer.storage['actions'] = _actions
        replay_buffer.storage['rewards'] = _rew 
        replay_buffer.storage['terminals'] = _done
        replay_buffer.buffer_size = N-1


    # Initialize BEAR policy
    max_action = 1.0  # Since the data is normalized, max_action should be 1.0
    policy = BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
                version='0', lambda_=0.5, threshold=0.05, mode='auto', num_samples_match=10,
                mmd_sigma=20.0, lagrange_thresh=10.0, use_kl=False, use_ensemble=False, kernel_type='gaussian')
    
    # Load dataset
    rb=ReplayBuffer()
    dataset = d4rl.qlearning_dataset(env)
    load_hdf5_mujoco(dataset, rb)
    
    max_timesteps =10 #1e6
    eval_freq = 10 #5e3
    training_iters = 0

    average_returns = []
    grad_steps = 0
    max_grad_steps = int(max_timesteps*eval_freq)
    while grad_steps  < max_grad_steps:  
        policy.train(rb, iterations=int(eval_freq))
        grad_steps += int(eval_freq)
        ret_eval, std_ret, median_ret, d4rl_score = normalized_evaluate_policy(policy, env)
        

        training_iters += eval_freq
        average_returns.append(ret_eval)

        print(f"Training without GAN: Average Return: {ret_eval}, std: {std_ret}, median: {median_ret}, D4RL Score: {d4rl_score}")

    # Calculate average, std and median of the training loop
    avg_return = np.mean(average_returns)
    std_return = np.std(average_returns)
    median_return = np.median(average_returns)

    # Write results to a text file
    with open('BEAR_training_results_without_gan.txt', 'w') as f:
        f.write(f"Training Results for {env_name}:\n")
        f.write(f"Average Return: {avg_return:.2f}\n")
        f.write(f"Standard Deviation: {std_return:.2f}\n")
        f.write(f"Median Return: {median_return:.2f}\n")
        f.write(f"D4RL Score: {d4rl_score:.2f}\n")

    print("BEAR Training results without GAN have been written to 'BEAR_training_results_without_gan.txt'")


print("Done")