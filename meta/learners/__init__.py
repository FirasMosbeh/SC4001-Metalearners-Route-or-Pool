"""Meta-learners, one per file. LEARNERS maps a name to its class, so notebooks can pick one by name."""
from meta.learners.max import MaxLearner
from meta.learners.mean import MeanLearner
from meta.learners.sorted_lr import SortedLRLearner
from meta.learners.stacking_lr import StackingLRLearner
from meta.learners.mlp import MLPLearner
from meta.learners.shuffled_mlp import ShuffledMLPLearner
from meta.learners.deep_sets import DeepSetsLearner
from meta.learners.gating import GatingLearner
from meta.learners.best_expert import BestExpertLearner
from meta.learners.pseudo_unseen import PseudoUnseenLearner


def pseudo_unseen(base):
    """Constructor of base wrapped in pseudo-unseen meta-training, taking base's arguments."""
    return lambda **kwargs: PseudoUnseenLearner(base(**kwargs))


LEARNERS = {
    # 0. Baseline: one expert, chosen on seen validation images
    "best_expert": BestExpertLearner,

    # 1. No training (Pooling)
    "mean": MeanLearner,
    "avg": MeanLearner,
    "max": MaxLearner,

    # 2. Simple models: Pair A (Identity control)
    "sorted_lr": SortedLRLearner,
    "sorted-lr": SortedLRLearner,
    "sorted_score_lr": SortedLRLearner,
    "stacking_lr": StackingLRLearner,
    "stacking-lr": StackingLRLearner,
    "lr": StackingLRLearner,

    # 3. Flexible models: Pair B (Identity control)
    "shuffled_mlp": ShuffledMLPLearner,
    "shuffled-mlp": ShuffledMLPLearner,
    "mlp": MLPLearner,

    # 4. Set-based pooling & routing gating
    "deep_sets": DeepSetsLearner,
    "deep-sets": DeepSetsLearner,
    "gating": GatingLearner,
    "gating_network": GatingLearner,

    # 5. Trained as if each image's generator were unseen
    "pu_stacking_lr": pseudo_unseen(StackingLRLearner),
    "pu_mlp": pseudo_unseen(MLPLearner),
    "pu_deep_sets": pseudo_unseen(DeepSetsLearner),
    "pu_gating": pseudo_unseen(GatingLearner),
}
