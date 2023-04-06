from .target_tracker import BayesianTargetTracker
from .associate import MergeAssociate


class PDATracker(BayesianTargetTracker, MergeAssociate):
    """PDA tracker."""
    pass
