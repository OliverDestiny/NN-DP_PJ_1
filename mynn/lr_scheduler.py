from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        Args:
            optimizer: The optimizer whose learning rate will be adjusted
            milestones: List of step indices at which to decay the learning rate
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.milestones = np.sort(milestones)
        self.gamma = gamma
        self.last_milestone = 0  # Track the last milestone reached

    def step(self) -> None:
        self.step_count += 1
        # Check if current step is in milestones and hasn't been processed yet
        if self.step_count in self.milestones and self.step_count > self.last_milestone:
            self.optimizer.init_lr *= self.gamma
            self.last_milestone = self.step_count

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        """
        Args:
            optimizer: The optimizer whose learning rate will be adjusted
            gamma: Multiplicative factor of learning rate decay each step
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.init_lr *= (self.gamma ** self.step_count)