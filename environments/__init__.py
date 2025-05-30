"""
Exploration environments package.

This package contains environments for robot exploration tasks.
"""

# from environments.env_Backup import ExplorationEnv
from environments.env import ExplorationEnv
from environments.exploration import ExplorationStrategy, RandomExplorationStrategy, RLExplorationStrategy
from environments.map_generator import Map_Generator
# from environments.nonlearning import NonLearningExploration

__all__ = ['ExplorationEnv', 'Map_Generator', 'NonLearningExploration']