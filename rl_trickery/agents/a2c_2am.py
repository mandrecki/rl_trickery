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
                 cognition_cost=0.1,
                 cognitive_coef=0.5,
                 only_action_values=True,
                 ):

        assert not acktr
        self.cognitive_coef = cognitive_coef
        self.long_horizon = long_horizon
        self.cognition_cost = cognition_cost
        self.only_action_values = only_action_values
        self.gamma_cog = 0.9

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
        if self.only_action_values:
            value_loss = advantages[rollouts.actions_cog == 1].pow(2).mean()
        else:
            value_loss = advantages.pow(2).mean()
        action_loss = -(advantages[rollouts.actions_cog == 1].detach() * action_log_probs[rollouts.actions_cog == 1]).mean()
        # extract loss while entropy is a series
        # dist_entropy = dist_entropy[env_actions_idx]
        if not torch.isnan(action_loss):
            env_loss = (value_loss * self.value_loss_coef + action_loss -
                        dist_entropy * self.entropy_coef)
        else:
            env_loss = (value_loss * self.value_loss_coef +
                        dist_entropy * self.entropy_coef)

        returns_cog = self.compute_cognitive_returns(
            rollouts.value_cog_preds[-1],
            advantages, rollouts.actions_cog,
            rollouts.masks[:-1].view(num_steps, num_processes, 1)
        )
        advantages_cog = returns_cog[:-1] - value_cog[:-1]
        value_cog_loss = advantages_cog.pow(2).mean()

        action_cog_loss = -(advantages_cog[1:].detach() * action_cog_log_probs[1:-1]).mean()
        cog_loss = (value_cog_loss + action_cog_loss -
                    dist_entropy_cog * self.entropy_coef)

        self.optimizer.zero_grad()

        (env_loss + self.cognitive_coef*cog_loss).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        value_loss = value_loss.item(), value_cog_loss.item()
        action_loss = action_loss.item(), action_cog_loss.item()
        dist_entropy = dist_entropy.item(), dist_entropy_cog.item()

        return value_loss, action_loss, dist_entropy

    def compute_cognitive_returns(
            self,
            next_value,
            advantages,
            a_cog,
            masks
    ):
        with torch.no_grad():
            # rewards = -a_cog * advantages.pow(2) + (1 - a_cog) * (-self.cognition_cost)
            advantages2 = advantages.pow(2)
            returns = torch.zeros_like(advantages)
            returns[-1] = next_value
            for step in reversed(range(advantages2.size(0)-1)):
                value_accuracy = (torch.log2(advantages2[step] + 1e-5) - torch.log2(advantages2[step+1] + 1e-5)) - self.cognition_cost

                # if no env action taken (a_c = 0)
                if self.long_horizon:
                    returns[step] = masks[step+1] * self.gamma_cog * next_value + value_accuracy * (1 - a_cog[step])
                else:
                    returns[step] = masks[step+1] * (1 - a_cog[step]) * self.gamma_cog * next_value + value_accuracy * (1 - a_cog[step])
                next_value = returns[step]

        return returns
