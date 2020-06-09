import torch
import torch.nn as nn
import torch.optim as optim


class A2C_2AM():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 long_horizon=False,
                 cognition_cost=0.01,
                 cognitive_coef=0.5
                 ):

        assert not acktr
        self.cognitive_coef = cognitive_coef
        self.long_horizon = long_horizon
        self.cognition_cost = cognition_cost
        self.gamma_cog = 0.99

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        actions = rollouts.actions.view(-1, action_shape), rollouts.actions_cog.view(-1, 1)
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, *self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            actions)

        values, value_cog = values
        dist_entropy, dist_entropy_cog = dist_entropy
        action_log_probs, action_cog_log_probs = action_log_probs

        values = values.view(num_steps, num_processes, 1)
        value_cog = value_cog.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        action_cog_log_probs = action_cog_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values

        # apply only where actions where taken
        env_actions_idx = rollouts.actions_cog == 1
        # action_loss = -(advantages[env_actions_idx].detach() * action_log_probs[env_actions_idx]).mean()
        # value_loss = advantages.pow(2).mean()
        value_loss = advantages[env_actions_idx].pow(2).mean()
        action_loss = -(advantages[env_actions_idx].detach() * action_log_probs[env_actions_idx]).mean()
        # extract loss while entropy is a series
        # dist_entropy = dist_entropy[env_actions_idx]
        env_loss = (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef)

        returns_cog = self.compute_cognitive_returns(
            rollouts.value_cog_preds[-1],
            advantages, rollouts.actions_cog,
            rollouts.masks[:-1].view(num_steps, num_processes, 1)
        )
        advantages_cog = returns_cog[:-1] - value_cog[-1]
        value_cog_loss = advantages_cog.pow(2).mean()

        action_cog_loss = -(advantages_cog.detach() * action_cog_log_probs[:-1]).mean()
        cog_loss = (value_cog_loss * self.value_loss_coef + action_cog_loss -
                    dist_entropy_cog * self.entropy_coef)

        self.optimizer.zero_grad()

        (env_loss + self.cognitive_coef*cog_loss).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def compute_cognitive_returns(
            self,
            next_value,
            advantages,
            a_cog,
            masks
    ):
        with torch.no_grad():
            rewards = -a_cog * advantages.pow(2) + (1 - a_cog) * (-self.cognition_cost)
            returns = torch.zeros_like(rewards)

            returns[-1] = next_value
            for step in reversed(range(rewards.size(0)-1)):
                # if no env action taken (a_c = 0)
                if self.long_horizon:
                    returns[step] = masks[step] * self.gamma_cog * next_value + rewards[step]
                else:
                    returns[step] = masks[step] * (1 - a_cog[step]) * self.gamma_cog * next_value + rewards[step]
                next_value = returns[step]

        return returns
