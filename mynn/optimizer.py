from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        if init_lr <= 0:
            raise ValueError("Initial learning rate must be positive.")
        self.init_lr = init_lr
        self.model = model

    def apply_weight_decay(self, param, decay, lr):
        """
        Apply weight decay to the parameter if decay is enabled.
        """
        return param * (1 - lr * decay)

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] = self.apply_weight_decay(
                            layer.params[key],
                            getattr(layer, 'weight_decay_lambda', 0.0),
                            self.init_lr
                        )
                    # Update parameters using gradient descent
                    layer.params[key] -= self.init_lr * layer.grads[key]

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        """
        Momentum Gradient Descent optimizer
        
        Args:
            init_lr (float): Initial learning rate
            model: The model containing parameters to optimize
            mu (float): Momentum factor (default: 0.9)
        """
        super().__init__(init_lr, model)
        self.mu = mu  # Momentum coefficient
        self.velocities = {}  # Stores velocity for each parameter
        
        # Initialize velocities for all optimizable parameters
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    self.velocities[(id(layer), key)] = np.zeros_like(layer.params[key])
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # Get velocity buffer for this parameter
                    vel_key = (id(layer), key)
                    
                    # Update velocity: v = mu*v - lr*grad
                    self.velocities[vel_key] = (
                        self.mu * self.velocities[vel_key]
                        - self.init_lr * layer.grads[key]
                    )
                    
                    # Apply weight decay if enabled
                    if getattr(layer, 'weight_decay', False):
                        layer.params[key] = self.apply_weight_decay(
                            layer.params[key],
                            getattr(layer, 'weight_decay_lambda', 0.0),
                            self.init_lr
                        )
                    
                    # Update parameters: param += velocity
                    layer.params[key] += self.velocities[vel_key]