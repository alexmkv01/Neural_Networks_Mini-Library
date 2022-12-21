import numpy as np
import pickle


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # First thing is to have an element-wise transformation of the inputs, where the input here refers to the output
        # of the LinearLayer (Z), it's just called the input to the SigmoidLayer. Sigmoid is given in the slides.
        sigmoid = 1/(1 + np.exp(-x))

        # For now, we just store x in the self._cache_current, as this is what we will pass in the backwards
        self._cache_current = x

        return sigmoid

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # The gradient of the activation will be dLoss/dZ = dLoss/dA * dA/dZ, where dLoss/dA is grad_z
        # The value of the derivative of sigmoid is the actual sigmoid value going forward multiplied
        # by 1 - sigma. This is then multiplied by grad_z
        sigmaValue = self.forward(self._cache_current)
        dLossdZ = grad_z * sigmaValue * (1 - sigmaValue)

        return dLossdZ

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # ReLU as a function can be represented as 
        # reLU = max(0, x), since it's set to 0 for any x value less than 0, otherwise linear relationship with x.
        
        # The value of self._cache_current will be x here as well. This will be used in the backward pass
        self._cache_current = x
        # The value of reLU will be the maximum between 0 and x, as a matrix
        relu = np.maximum(0, x)

        return relu

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # ReLU derivative is just a 0 in the index where x is less than 0, where we stored this x in the
        # self._current_cache variable. Thus, we just index using it
        dLossdZ = np.copy(grad_z)
        dLossdZ[self._cache_current <= 0] = 0

        return dLossdZ

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # The number of weights for each layer will be the number of inputs times number of outputs
        # If we have 30 inputs and 5 outputs, the number of weights needed between those will be 30 * 5
        self._W = xavier_init(size = (self.n_in, self.n_out), gain = 1)
        # For every layer, eahc neuron in the layer has one bias, and so if we have N inputs, then we will
        # have N bias terms
        self._b = np.zeros(self.n_out)

        self._cache_current = []
        self._grad_W_current = []
        self._grad_b_current = []

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Since grad_z is the gradient of the scalar loss with respect to the output, so dLoss/dZ, we
        # need to store, for every single layer, the partial derivatives of the output with respect to
        # each of our parameters - dZ/dW, dZ/dX, and dZ/dB
        dZdW = x
        dZdX = self._W
        # Notice that the derivative of the output with respect to the bias is one column vector of 1s, which
        # will be the same size as the number of batches. This is because we BROADCAST each b vector, where
        # each b vector is of dimension (n_out, 1). This is done because if we do not broadcast, then we end 
        # up repeating the columns in our derivative calculation when we multiply by dLoss/dZ, which makes no
        # sense. We want dZdB to just result in a COLUMN vector. This also makes more sense, as the bias vector
        # itself is just a column anyway
        dZdB = np.ones((x.shape[0]))
        
        # Save these in the _cache_current list
        self._cache_current = [dZdW, dZdX, dZdB]

              
        Z = x @ self._W + self._b
       
        return Z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # grad_z is dLoss/dZ, which is then used in the chain rule to multiply by the gradient we found in 
        # self._cache_current to find the partial derivatives of the Loss with respect to each parameter, such
        # that we get dLoss/dW = dLoss/dZ * dZ/dW and dLoss/dB = dLoss/dZ * dZ/dB
        
        self._cache_current = np.array(self._cache_current, dtype="object")
        self._grad_W_current = grad_z.T @ self._cache_current[0]
        self._grad_b_current = grad_z.T @ self._cache_current[2]
        
        # The same is done to find dLoss/dX, which will be dLoss/dX = dLoss/dZ * dZ/dX. This is what gets passed to
        # the lower layer
        dLossdX = grad_z @ self._cache_current[1].T

       
        return dLossdX

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Essentially, what we do here is that we want to update the parameters from the gradients that we just
        # found. We know that in gradient descent, in every layer, the weights and biases are updated as according to
        # the equation b = b - aplha*gradient (we're minimizing by going in the direction opposite the gradient), and 
        # the same is done for W such that W = W - alpha*gradient
        self._W = self._W - learning_rate * self._grad_W_current.T
        self._b = self._b - learning_rate * self._grad_b_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # The length of the neurons list will be the number of linear layers which we have, where the number of neurons in 
        # each layer is given by the actual values in each entry of the list. input_dims will be the number of neurons in the
        # first layer (input layer), neurons will be a list of all subsequent output layers, with the number of neurons in each,
        # and activations will be the activations to apply to each output of a layer
            
        # Initialize layers to be empty list
        self._layers = []        

        # We iterate over the length of neurons
        for i in range(len(neurons)):
            # For every single index, we need to
            # 1- Create a layer with appropriate input and output dimensions
            if (i == 0):
                # Append the correct activation
                if (self.activations[i] == "relu"):
                    # Append the first layer, which is the input layer, as well as its associated activation layer
                    self._layers.append([LinearLayer(input_dim, neurons[0]), ReluLayer()])
                elif (self.activations[i] == "sigmoid"):
                    self._layers.append([LinearLayer(input_dim, neurons[0]), SigmoidLayer()])
                elif (self.activations[i] == "identity"):
                    # Create the ones matrices for when we multiply by identity
                    self._layers.append([LinearLayer(input_dim, neurons[0]),])
            else:
                # Append the correct activation
                if (self.activations[i] == "relu"):
                    # Append the correct rest of layers, as well as its associated activation layer
                    self._layers.append([LinearLayer(neurons[i - 1], neurons[i]), ReluLayer()])
                elif (self.activations[i] == "sigmoid"):
                    self._layers.append([LinearLayer(neurons[i - 1], neurons[i]), SigmoidLayer()])
                elif (self.activations[i] == "identity"):
                    self._layers.append([LinearLayer(neurons[i - 1], neurons[i]),])
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Essentially, we want to compute a forward pass over the entire NN, meaning that the output of one layer
        # of the neural network will be the input of the next layer, until we reach the final layer, which we will
        # then return as the final output of the forward pass. Thus, the output of self._layers[i] will be the input
        # to self._layers[i+1]
        # Define our initial z to be the input
        a = x.copy()
        # Loop over all the layers in our network
        for i in range(len(self._layers)):
            
            # Find the output of the linear layer to give as input to the next one
            z = self._layers[i][0].forward(a)
           
            # Calculate z after passing it through the activation function in self.activations, where we assume that each
            # element of the activations list is a class, defined as either SigmoidLayer or ReLULayer, which can call the
            # .forward() function on the z found by the forward pass
            if (self.activations[i] == "relu"):
                a = self._layers[i][1].forward(z)
            elif (self.activations[i] == "sigmoid"):
                a = self._layers[i][1].forward(z)
            elif (self.activations[i] == "identity"):
                a = z

        return z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Again, we want to iterate for every layer within our NN, where for every layer, the derivative that gets
        # propogated back will be the derivative with respect to the output of that layer, multiplied by the derivative
        # of the output with respect to the activation, multiplied by the derivative of the previous input to the activation 
        # with respect to the input 
        # output = dLoss/dZ * dA/dZ[3] * dZ[3]/dA[2], where dA[2] is the output of the previous
        
        # Define the dZdANext, which we change every iteration
        dZdA = grad_z.copy()
        # Iterating over all the different layers
        for i in range(len(self._layers)):


            # Find the derivative of the loss with respect Z
            if (self.activations[-i-1] == "relu"):
                dAdZ = self._layers[-i-1][1].backward(dZdA)
               
            elif (self.activations[-i-1] == "sigmoid"):
                dAdZ = self._layers[-i-1][1].backward(dZdA)
                
            elif (self.activations[-i-1] == "identity"):
                dAdZ = dZdA
                
           
            # Find the derivative of Z with respect to A
            dZdA = self._layers[-i-1][0].backward(dAdZ)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Now that we have the overall gradient computed, when we call .backward() for the whole network,
        # then for EACH LAYER, the corresponding dLoss/dW and dLoss/dX and all the information in cache_current
        # will be stored WITH RESPECT TO THAT LAYER. Meaning, all we need to do to ensure that the correct gradients
        # are being used, is just to loop over the layers, as they store the appropriate gradients for themselves
        # as we are backpropogating and forward passing
        
        # Iterating over all layers
        for i in range(len(self._layers)):
            self._layers[i][0].update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Based on the string given in loss_fun, we then need to decode the string (compare it to either mse or
        # cross_entropy), and then based on which one matches, we define our loss layer to be an instance of that
        # class
        if (self.loss_fun == "mse"):
            self._loss_layer = MSELossLayer()
        elif (self.loss_fun == "cross_entropy"):
            self._loss_layer = CrossEntropyLossLayer()
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # We want to shuffle the input and target datasets. One way to do this is to use the permutations that
        # we previously have seen, which goes as follows
        # Define the random generator
        np.random.seed(0)
        randomGen = np.random.default_rng()
        # Define the indices of the shuffle, where we create a permutation of the length of the inpiut dataset
        # (same as taregt dataset)
        shuffledIndices = randomGen.permutation(input_dataset.shape[0])
        # Shuffle the input and target datasets
        shuffledInputs = input_dataset[shuffledIndices]
        shuffledTargets = target_dataset[shuffledIndices]

        # Returns shuffled datasets
        return shuffledInputs, shuffledTargets

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # THIS IS THE BIG BOY!!! TRAINING LOOP!!!!
        # We want to implement the main training loop, in which we loop nb_epoch times. Before we do the
        # loop, we want to make sure we shuffle the data if the self.shuffle_flag is True. We then want to
        # loop nb_epoch times, where every single time we will
        # (1) Perform forward pass with batch
        # (2) Get Loss using the self._loss_layer
        # (3) Do backprop of loss
        # (4) Do one step of update_params()

        # Shuffle if true, else just keep the same
        if (self.shuffle_flag == True):
            inputs, targets = self.shuffle(input_dataset=input_dataset, target_dataset=target_dataset)
        else:
            inputs = input_dataset.copy()
            targets = target_dataset.copy()

        # Loop for nb_epoch times
        for i in range(self.nb_epoch):
            # Define the indices used for batching
            startBatchIndex = 0
            endBatchIndex = self.batch_size

            # We continue looping for every single batch, dividing the overall dataset into different batches,
            # until our enfBatchIndex exceeds the length of the overall input dataset. At that point, we know that
            # we've iterated over the whole thing
            while (endBatchIndex <= inputs.shape[0]):
                # (0) Take only the iteration's appropriate batch (which block of the whole dataset)
                currentBatchInputs = inputs[startBatchIndex:endBatchIndex]
                currentBatchTargets = targets[startBatchIndex:endBatchIndex]

                # (1) Perform forward pass with batch
                networkOutput = self.network(currentBatchInputs)
                # Make float
                networkOutputFloat = networkOutput.astype(float)
               
                # (2) Get loss using the self._loss_layer
                networkLoss = self._loss_layer.forward(networkOutputFloat, currentBatchTargets)
                # Find the gradient of the loss using the backward() function
                networkLossGradient = self._loss_layer.backward()

                # (3) Do backpropogation of loss using the gradient of the loss
                self.network.backward(networkLossGradient)

                # (4) Do one step of updating the network parameters
                self.network.update_params(self.learning_rate)
                
                # (5) UPDATE END AND START BATCH
                startBatchIndex = endBatchIndex
                endBatchIndex = endBatchIndex + self.batch_size
                

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # In the evaluation, what we want to do is essentially just find the loss of the network given that
        # it's already been trained earlier. Thus, what we do is just call the forward function of the loss
        # function on the output from the network
        # (1) Get network output
        networkOutput = self.network(input_dataset)
        networkOutputFloat = networkOutput.astype(float)

        # (2) Get loss using the self._loss_layer
        networkLoss = self._loss_layer.forward(networkOutputFloat, target_dataset)
                
        # Return the final networkLoss
        return networkLoss
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """
    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # We define the normalization parameters, in which we will scale the smallest value
        # to a and the largest to b, where a = 0 and b = 1
        self.a = 0
        self.b = 1
        # Find the minimum and maximum values for every feature across the batches, making sure to eliminate NaNs
        self.xMin = np.nanmin(data, axis=0)
        self.xMax = np.nanmax(data, axis=0)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # here we are applying the min-max normalisation
        # but first we need to make sure we need to account for the NaN values
        # for any NaN value, we will replace it with the median of our dataset
        # since the median is unbiased compared to the mean
                
        # Define the median of the data, which we will replace the NaNs with
        median = np.nanmedian(data, axis = 0)
        # Get locations of NaN values in data
        nanIndex = np.where(np.isnan(data))
        # Repalce nan values with the median, where np.take() takes elements from an array along
        # a specific axis. Here, we take elements for the NaNs, and insert the median 
        data[nanIndex] = np.take(median, nanIndex[1])

        # Here, we apply the min-max normalization of the data, such that we scale the smallest value to
        # be 0, and the largest to be 1. This is done because weight updates are proportional to the
        # input, where not min-max scaling would mean that larger-valued inputs would skew with the
        # weight update
        # Apply normalization
        normalized = self.a + (((data - self.xMin)*(self.b - self.a))/(self.xMax - self.xMin))

        return normalized
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Here, we want to reverse our pre-processing, in which we just solve for X given
        # X', or the min-max normalized dataset. This was done by hand
        reverted = ((data)*(self.xMax - self.xMin) - (self.a*(self.xMax - self.xMin)))/(self.b - self.a) \
            + self.xMin
        
        # Ensure there are no NaNs
        # Define the median of the data, which we will replace the NaNs with
        median = np.nanmedian(reverted, axis = 0) # remember we use median because it is unbiased unlike the mean
        # Get locations of NaN values in reverted
        nanIndex = np.where(np.isnan(reverted))
        # Repalce nan values with the median
        reverted[nanIndex] = np.take(median, nanIndex[1]) 

        return reverted

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

if __name__ == "__main__":
    example_main()
