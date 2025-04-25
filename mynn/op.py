from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=lambda size: np.random.normal(scale=np.sqrt(2/sum(size)), size=size), weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}
        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # Save input for backward pass
        return np.dot(X, self.params['W']) + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        
        # Gradient for weights: dL/dW = X.T @ grad
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size
        
        # Gradient for bias: dL/db = sum(grad, axis=0)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        
        # Gradient for input: dL/dX = grad @ W.T
        input_grad = np.dot(grad, self.params['W'].T)
        
        return input_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels, 1))
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        batch_size, in_channels, in_h, in_w = X.shape
        self.input = X
        
        # Calculate output dimensions
        out_h = (in_h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)))
        else:
            X_padded = X
            
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = X_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, h, w] = np.sum(receptive_field * self.W[oc]) + self.b[oc]
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_h, out_w = grads.shape
        X = self.input
        
        # Initialize gradients
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        input_grad = np.zeros_like(X)
        
        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)))
            input_grad_padded = np.pad(input_grad, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)))
        else:
            X_padded = X
            input_grad_padded = input_grad
            
        # Compute gradients
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = X_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Gradient for weights
                        self.grads['W'][oc] += grads[b, oc, h, w] * receptive_field
                        
                        # Gradient for input
                        input_grad_padded[b, :, h_start:h_end, w_start:w_end] += grads[b, oc, h, w] * self.W[oc]
                
                # Gradient for bias
                self.grads['b'][oc] += np.sum(grads[b, oc])
        
        # Remove padding from input gradient if needed
        if self.padding > 0:
            input_grad = input_grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        # Average gradients over batch
        self.grads['W'] /= batch_size
        self.grads['b'] /= batch_size
        
        return input_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.where(X<0, 0, X)
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        return np.where(self.input < 0, 0, grads)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        
        # Numerical stability
        probs = np.clip(probs, 1e-15, 1.0)
        
        # Get probabilities of correct classes
        batch_size = predicts.shape[0]
        correct_probs = probs[np.arange(batch_size), labels]
        
        # Compute loss
        loss = -np.mean(np.log(correct_probs))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        
        if self.has_softmax:
            # Compute softmax gradients
            probs = softmax(self.predicts)
            probs[np.arange(batch_size), self.labels] -= 1
            self.grads = probs / batch_size
        else:
            # For raw logits
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(batch_size), self.labels] = -1.0 / self.predicts[np.arange(batch_size), self.labels]
            self.grads /= batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_=1e-4) -> None:
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.optimizable = False

    def forward(self, X):
        """Compute L2 regularization term"""
        l2_term = 0
        for layer in self.model.layers:
            if layer.optimizable and hasattr(layer, 'W'):
                l2_term += np.sum(layer.W ** 2)
        return self.lambda_ * 0.5 * l2_term

    def backward(self):
        """Compute gradients for L2 regularization"""
        for layer in self.model.layers:
            if layer.optimizable and hasattr(layer, 'W'):
                if layer.grads['W'] is None:
                    layer.grads['W'] = self.lambda_ * layer.W
                else:
                    layer.grads['W'] += self.lambda_ * layer.W
        return None
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition