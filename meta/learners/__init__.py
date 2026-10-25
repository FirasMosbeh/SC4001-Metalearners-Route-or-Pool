"""Meta-learners, one per file. LEARNERS maps a name to its class, so notebooks can pick one by name."""
from meta.learners.max import MaxLearner

LEARNERS = {
    "max": MaxLearner,
}
