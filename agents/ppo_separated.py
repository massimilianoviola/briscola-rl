import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from typing import List


class actor(nn.Module):
    """Given a state outputs actions probabilities"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int],
    ) -> None:
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], output_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.MLP(x)


class critic(nn.Module):
    """Given a state outputs its value"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int],
    ) -> None:
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], output_dim),
        )

    def forward(self, x):
        return self.MLP(x)


class PPOAgent:
    """PPO with separated networks to represent policy and value function"""
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        discount: float,
        gae_lambda: float,
        critic_loss_fn,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_layers: List[int] = [256, 256],
        critic_layers: List[int] = [256, 256],
        ppo_steps: int = 5,
        ppo_clip: float = 0.2,
        ent_coeff: float = 0.0,
        num_episodes: int = 32,
        device=None,
        log=True,
    ) -> None:
        self.name = "PPOAgent"

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.n_features = n_features
        self.n_actions = n_actions

        self.last_state = None
        self.state = None
        self.action = None
        self.log_prob = None
        self.reward = None

        self.reset_batch()

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.deck = np.zeros((10, 4))

        self.loss_fn = critic_loss_fn
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate

        self.actor_net = actor(n_features, n_actions, actor_layers).to(self.device)
        self.critic_net = critic(n_features, 1, critic_layers).to(self.device)

        self.actor_opt = optim.Adam(
            self.actor_net.parameters(),
            lr=actor_learning_rate,
        )

        self.critic_opt = optim.Adam(
            self.critic_net.parameters(),
            lr=critic_learning_rate,
        )

        self.ppo_steps = ppo_steps
        self.ppo_clip = ppo_clip
        self.ent_coeff = ent_coeff
        self.num_episodes = num_episodes

        self.log = log
        self.ep = 0
        self.max_takes = 10

    def reset_batch(self):
        self.states_batch = []
        self.actions_batch = []
        self.log_probs_batch = []
        self.episode_rewards = []
        self.rewards_batch = []
        self.dones = []

    def observe(self, env, player):
        """Observes the environment and updates the state"""
        self.get_state(env, player)

    def get_state(self, env, player):
        """"""
        state = np.zeros(26)
        value_offset = 2
        seed_offset = 2
        features_per_card = 6
        state[0] = player.points
        state[1] = env.counter

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * features_per_card + value_offset
            seed_index = (i + 3) * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

            self.deck[card.number][card.seed] = 1

        deck = self.deck.flatten()
        #state = np.concatenate((state, deck))

        self.last_state = self.state
        self.state = state
        self.done = env.check_end_game()

    def select_action(self, available_actions):
        state = torch.from_numpy(self.state).float().to(self.device)
        actions_probabilities = self.actor_net(state)
        #print(actions_probabilities)
        dist = distributions.Categorical(actions_probabilities)
        takes = 0
        while True:
            takes += 1
            if takes >= self.max_takes:
                action = np.random.choice(available_actions)
                self.action = torch.as_tensor(action)
                self.log_prob = dist.log_prob(self.action).detach()
                break
            self.action = dist.sample().detach()
            self.log_prob = dist.log_prob(self.action).detach()
            if self.action in available_actions:
                break
        
        #print(f'state: {state}')
        #print(f'probs {actions_probabilities.detach()} ---> {self.action}')
        return self.action.item()

    def get_returns(self, rewards_batch, normalize=True):
        returns = []
        for rewards in reversed(rewards_batch):
            G = 0
            for r in reversed(rewards):
                G = r + self.discount * G
                returns.append(G)
        returns = returns[::-1]
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        return returns

    def get_values(self, states):
        values = self.critic_net(states).squeeze()
        return values

    def get_advantages(self, returns, values, normalize=True):
        adv = returns - values.detach()
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        return adv

    def gae_advantages(self, rewards_batch, values, dones, norm_adv=True, norm_ret=False):
        v = values.detach()
        rewards = rewards_batch.flatten()
        adv = np.zeros_like(rewards)
        gae_cum = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_v = 0.0
                d = 0
            else:
                next_v = v[i+1]
                d = 1
            delta = rewards[i] + self.discount * next_v * d - v[i]
            gae_cum = delta + self.discount * self.gae_lambda * gae_cum * d
            adv[i] = gae_cum

        adv = torch.Tensor(adv).to(self.device)
        ret = adv + v

        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        if norm_ret:
            ret = (ret - ret.mean()) / (ret.std() + 1e-10)

        return adv, ret

    def evaluate(self, states, actions):
        action_probabilities = self.actor_net(states)
        state_values = self.critic_net(states)
        dist = distributions.Categorical(action_probabilities)
        current_log_probs = dist.log_prob(actions)
        return current_log_probs, state_values.squeeze(), dist.entropy()

    def update(self, reward):
        self.reward = reward
        #print(f'reward: {reward}')
        #print('-'*100)
        self.states_batch.append(self.last_state)
        self.actions_batch.append(self.action)
        self.log_probs_batch.append(self.log_prob)
        self.episode_rewards.append(self.reward)
        self.dones.append(self.done)
        '''
        print('*'*140)
        print(f'UPDATING')
        print(f'last_state {self.last_state}')
        print(f'action {self.action}')
        print(f'reward {self.reward}')
        print(f'state {self.state}')
        print('*'*140)
        '''
        if self.done:
            self.ep += 1
            self.rewards_batch.append(self.episode_rewards)
            #print(self.episode_rewards)
            self.episode_rewards = []
            if len(self.rewards_batch) >= self.num_episodes:
                self.learn()
                self.reset_batch()

    def learn(self):
        self.states_batch = torch.Tensor(np.array(self.states_batch))
        self.actions_batch = torch.Tensor(np.array(self.actions_batch))
        self.log_probs_batch = torch.Tensor(np.array(self.log_probs_batch))
        self.rewards_batch = torch.Tensor(np.array(self.rewards_batch))

        returns = self.get_returns(self.rewards_batch, True)
        #print(returns)
        values = self.get_values(self.states_batch)
        adv = self.get_advantages(returns, values)
        #adv, returns = self.gae_advantages(self.rewards_batch, values, self.dones, True, True)
        for _ in range(self.ppo_steps):
            #breakpoint()
            # values = self.get_values(self.states_batch)
            curr_log_probs, values, entropy = self.evaluate(
                self.states_batch, self.actions_batch
            )

            ratios = torch.exp(curr_log_probs - self.log_probs_batch)
            loss_1 = ratios * adv
            loss_2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
            policy_loss = -torch.min(loss_1, loss_2).mean()
            entropy_loss = -entropy.mean()

            actor_loss = policy_loss + self.ent_coeff * entropy_loss
            critic_loss = self.loss_fn(values, returns)

            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            #print()

        if self.log:
            print(f"EPISODE #{self.ep}", end=" ")
            print(f"actor_loss: {actor_loss.detach():.4f}", end=" ")
            print(f"critic_loss: {critic_loss.detach():.4f}", end=" ")
            print(f"entropy_loss: {entropy_loss:.4f}", end="\n")

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass

    def update_epsilon(self):
        pass

    def reset(self):
        self.deck = np.zeros((10, 4))

    def save(self, path):
        """Saves policy network's state dictionary to path"""
        torch.save(self.actor_net.state_dict(), path)
        torch.save(self.critic_net.state_dict(), path)

    def load(self, path):
        """Loads policy network's state dictionary from path"""
        self.actor_net.load_state_dict(torch.load(path))
        self.critic_net.load_state_dict(torch.load(path))
