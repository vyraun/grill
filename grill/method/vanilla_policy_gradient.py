from policy_gradient_method import PolicyGradientMethod
from grill.core.rl import estimate_advantages
from grill.util.misc import safelog_sym, attrwise_cat, sanity_check_params
import theano.tensor as T
import numpy as np

# Partially based on the RLLab implementation
class VanillaPolicyGradient(PolicyGradientMethod):
    def __init__(self, policy, baseline,
            learning_rate=0.01,
            batchsize=1,    # Note: this refers to the number of *episodes* to use
            lam=0.95        # Lambda parameter for GAE
    ):
        super(VanillaPolicyGradient, self).__init__('VPG', policy, baseline, learning_rate, batchsize, lam)

        observation_var = policy.get_input_var()
        pdist_var = policy.get_output_var()
        action_var = T.ivector('action') # TODO: consider continuous action spaces
        advantage_var = T.fvector('advantage')
        log_likelihood_var = safelog_sym(pdist_var[T.arange(action_var.shape[0]),action_var])
        loss_var = -T.mean(log_likelihood_var * advantage_var)
        self._build_update(
                [observation_var, action_var, advantage_var],
                policy.get_param_vars(),
                loss_var
        )

    def post_episode(self, engine, episode):
        if engine.num_episodes % self.batchsize == 0:
            recent_episodes = engine.recent_episodes(self.batchsize)
            all_observations, all_actions = attrwise_cat(recent_episodes, ['observations', 'actions'])
            estimate_advantages(self.baseline, recent_episodes, engine.discount, self.lam)
            all_advantages = attrwise_cat(recent_episodes, ['advantages'])[0]
            self._update(all_observations, all_actions, all_advantages)
            sanity_check_params(self.policy.get_params())
