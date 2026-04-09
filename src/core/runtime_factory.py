"""Centralized runtime dependency factory for CLI and app entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from config import ProjectConfig
from src.agent import QLearningAgent
from src.core.interfaces import AgentProtocol, EnvironmentProtocol
from src.environment import TsunamiAlertEnvironment


@dataclass(frozen=True)
class RuntimeBundle:
    """Immutable runtime bundle used by training/evaluation flows."""

    config: ProjectConfig
    environment: EnvironmentProtocol
    agent: AgentProtocol


class RuntimeFactory:
    """Builds consistent environment/agent bundles for all entrypoints."""

    @staticmethod
    def build_training_bundle(training_episodes: int, seed: int) -> RuntimeBundle:
        """Creates a training-ready runtime bundle."""
        config = ProjectConfig(training_episodes=training_episodes, random_seed=seed)
        environment = TsunamiAlertEnvironment(config=config, seed=seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            min_epsilon=config.min_epsilon,
            seed=seed,
        )
        return RuntimeBundle(config=config, environment=environment, agent=agent)

    @staticmethod
    def build_evaluation_bundle(evaluation_episodes: int, seed: int) -> RuntimeBundle:
        """Creates an evaluation-ready runtime bundle (greedy by default)."""
        config = ProjectConfig(evaluation_episodes=evaluation_episodes, random_seed=seed)
        environment = TsunamiAlertEnvironment(config=config, seed=seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            seed=seed,
        )
        return RuntimeBundle(config=config, environment=environment, agent=agent)

