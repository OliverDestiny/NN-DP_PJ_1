from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implemented using the operators in op.py.
    """
    def __init__(self, conv_params, fc_params, act_func='ReLU'):
        super().__init__()
        self.conv_layers = []
        self.fc_layers = []
        self.act_func = act_func

        # Initialize convolutional layers
        for params in conv_params:
            self.conv_layers.append(conv2D(**params))
            if act_func == 'ReLU':
                self.conv_layers.append(ReLU())

        # Initialize fully connected layers
        for i in range(len(fc_params) - 1):
            self.fc_layers.append(Linear(fc_params[i], fc_params[i + 1]))
            if i < len(fc_params) - 2 and act_func == 'ReLU':
                self.fc_layers.append(ReLU())

        # Combine all layers into a single list for optimizer compatibility
        self.layers = self.conv_layers + self.fc_layers

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Forward pass through convolutional layers
        for layer in self.conv_layers:
            X = layer(X)

        # Flatten the output for fully connected layers
        X = X.reshape(X.shape[0], -1)

        # Forward pass through fully connected layers
        for layer in self.fc_layers:
            X = layer(X)

        return X

    def backward(self, loss_grad):
        # Backward pass through fully connected layers
        grads = loss_grad
        for layer in reversed(self.fc_layers):
            grads = layer.backward(grads)

        # Find the last conv2D layer (ignoring activation layers like ReLU)
        last_conv_layer = next(layer for layer in reversed(self.conv_layers) if isinstance(layer, conv2D))

        # Reshape gradients to match the output shape of the last conv layer
        grads = grads.reshape(-1, last_conv_layer.out_channels, 
                              last_conv_layer.input.shape[2], 
                              last_conv_layer.input.shape[3])

        # Backward pass through convolutional layers
        for layer in reversed(self.conv_layers):
            grads = layer.backward(grads)

        return grads

    def save_model(self, save_path):
        """Save the model parameters to a file."""
        param_list = {
            "conv_layers": [
                {"W": layer.W, "b": layer.b} for layer in self.conv_layers if isinstance(layer, conv2D)
            ],
            "fc_layers": [
                {"W": layer.W, "b": layer.b} for layer in self.fc_layers if isinstance(layer, Linear)
            ]
        }
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, param_path):
        """Load the model parameters from a file."""
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)

        # Load parameters into convolutional layers
        conv_idx = 0
        for layer in self.conv_layers:
            if isinstance(layer, conv2D):
                layer.W = param_list["conv_layers"][conv_idx]["W"]
                layer.b = param_list["conv_layers"][conv_idx]["b"]
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                conv_idx += 1

        # Load parameters into fully connected layers
        fc_idx = 0
        for layer in self.fc_layers:
            if isinstance(layer, Linear):
                layer.W = param_list["fc_layers"][fc_idx]["W"]
                layer.b = param_list["fc_layers"][fc_idx]["b"]
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                fc_idx += 1