from gradient_method import GradientMethod
from grill.util.misc import attrwise_cat

# Base class for policy gradient methods. Not intended to be used directly.
# Use a subclass (vanilla, natural)
class PolicyGradientMethod(GradientMethod):
    def __init__(self,
            name,
            policy,
            baseline,
            learning_rate,
            batchsize,  # Note: this refers to the number of *episodes* to use
            lam         # Lambda parameter for GAE
    ):
        super(PolicyGradientMethod, self).__init__(name, learning_rate, batchsize)
        self.policy = policy
        self.baseline = baseline
        self.lam = lam
