from .target_tracker import BayesianTargetTracker
from .associate import PruneAssociate


class NNTracker(BayesianTargetTracker, PruneAssociate):
    """Nearest neighbor tracker."""
    pass
