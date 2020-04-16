import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        output = nn.DotProduct(x, self.w)
        return output

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        output = self.run(x)
        output = nn.as_scalar(output)
        if output < 0:
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            new_changes = 0
            for x, y in dataset.iterate_once(1):
                predictions = self.get_prediction(x)
                if abs(predictions -  nn.as_scalar(y)) >= 1e-5:
                    self.w.update(direction = x, multiplier = nn.as_scalar(y))
                    new_changes += 1

            if new_changes == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_dims = [10, 20]
        self.weights = []
        self.biases = []
        self.num_input_features = 1
        self.learning_rate = -1e-2

        self.weights.append(nn.Parameter(self.num_input_features, self.hidden_dims[0]))
        self.biases.append(nn.Parameter(1, self.hidden_dims[0]))
        for index in range(1, len(self.hidden_dims)):
            self.weights.append(nn.Parameter(self.hidden_dims[index-1], self.hidden_dims[index]))
            self.biases.append(nn.Parameter(1, self.hidden_dims[index]))

        self.output_weight = nn.Parameter(self.hidden_dims[-1], 1)
        self.output_bias = nn.Parameter(1, 1)

        self.activation_fn = nn.ReLU


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        for index in range(len(self.hidden_dims)):
            x = nn.Linear(x, self.weights[index])
            x = nn.AddBias(x, self.biases[index])
            x = nn.ReLU(x)

        x = nn.Linear(x, self.output_weight)
        x = nn.AddBias(x, self.output_bias)

        return x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            sum_loss = []
            for x, y in dataset.iterate_once(20):
                loss = self.get_loss(x, y)
                sum_loss.append(nn.as_scalar(loss))
                gradidents = nn.gradients(loss, [*self.weights, *self.biases, self.output_weight, self.output_bias])

                for index in range(len(self.hidden_dims)):
                    self.weights[index].update(gradidents[index], self.learning_rate)
                    self.biases[index].update(gradidents[index+len(self.hidden_dims)], self.learning_rate)
                self.output_weight.update(gradidents[-2], self.learning_rate)
                self.output_bias.update(gradidents[-1], self.learning_rate)
            print(sum(sum_loss) / 10)
            if (sum(sum_loss) / 10) <= 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_dims = [512, 32]
        self.weights = []
        self.biases = []
        self.num_input_features = 784
        self.learning_rate = -1e-1
        self.batch_size = 50

        self.weights.append(nn.Parameter(self.num_input_features, self.hidden_dims[0]))
        self.biases.append(nn.Parameter(1, self.hidden_dims[0]))
        for index in range(1, len(self.hidden_dims)):
            self.weights.append(nn.Parameter(self.hidden_dims[index - 1], self.hidden_dims[index]))
            self.biases.append(nn.Parameter(1, self.hidden_dims[index]))

        self.output_weight = nn.Parameter(self.hidden_dims[-1], 10)
        self.output_bias = nn.Parameter(1, 10)

        self.activation_fn = nn.ReLU

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for index in range(len(self.hidden_dims)):
            x = nn.Linear(x, self.weights[index])
            x = nn.AddBias(x, self.biases[index])
            x = nn.ReLU(x)

        x = nn.Linear(x, self.output_weight)
        x = nn.AddBias(x, self.output_bias)

        return x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        predictions = self.run(x)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        for i in range(100000):
            sum_loss = []
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                sum_loss.append(nn.as_scalar(loss))
                gradidents = nn.gradients(loss, [*self.weights, *self.biases, self.output_weight, self.output_bias])

                for index in range(len(self.hidden_dims)):
                    self.weights[index].update(gradidents[index], self.learning_rate)
                    self.biases[index].update(gradidents[index+len(self.hidden_dims)], self.learning_rate)
                self.output_weight.update(gradidents[-2], self.learning_rate)
                self.output_bias.update(gradidents[-1], self.learning_rate)
            val_accuracy = dataset.get_validation_accuracy()
            print("Epoch {} with loss : {}, val accuracy: {}".format(i+1, sum(sum_loss) / 10, val_accuracy))
            if val_accuracy >= 0.98:
                break



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_dims = [16, 32]
        self.weights = []
        self.biases = []
        self.num_input_features = 1
        self.learning_rate = -1e-2
        self.batch_size = 20

        self.weights.append(nn.Parameter(self.num_input_features, self.hidden_dims[0]))
        self.biases.append(nn.Parameter(1, self.hidden_dims[0]))
        for index in range(1, len(self.hidden_dims)):
            self.weights.append(nn.Parameter(self.hidden_dims[index - 1], self.hidden_dims[index]))
            self.biases.append(nn.Parameter(1, self.hidden_dims[index]))

        self.output_weight = nn.Parameter(self.hidden_dims[-1], 10)
        self.output_bias = nn.Parameter(1, 10)

        self.activation_fn = nn.ReLU

        self.weight_1 = nn.Parameter(self.num_chars, 128)
        self.hidden_1 = nn.Parameter(128, 128)
        self.output_1 = nn.Parameter(128, 64)

        self.weight_2 = nn.Parameter(128, 32)
        self.hidden_2 = nn.Parameter(32, 32)
        self.output_2 = nn.Parameter(32, 5)



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        hiddens = []

        for index, x in enumerate(xs):
            if index == 0:
                hidden_states = nn.Linear(x, self.weight_1) # (batch_size, num_chars) * (num_chars, 32)
                hidden_states = nn.ReLU(hidden_states)
            else:
                out1 = nn.Linear(x, self.weight_1) # (batch_size, num_chars) * (num_chars, 32)
                out2 = nn.Linear(hidden_states, self.hidden_1) # (batch_size, 32) * (32, 32)
                hidden_states = nn.Add(out1, out2)
                hidden_states = nn.ReLU(hidden_states)
            hiddens.append(hidden_states)

        for index, x in enumerate(hiddens):
            if index == 0:
                hidden_states = nn.Linear(x, self.weight_2) # (batch_size, num_chars) * (num_chars, 32)
                hidden_states = nn.ReLU(hidden_states)
            else:
                out1 = nn.Linear(x, self.weight_2) # (batch_size, num_chars) * (num_chars, 32)
                out2 = nn.Linear(hidden_states, self.hidden_2) # (batch_size, 32) * (32, 32)
                hidden_states = nn.Add(out1, out2)
                hidden_states = nn.ReLU(hidden_states)

        output = nn.Linear(hidden_states, self.output_2)

        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(xs)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for i in range(10000):
            sum_loss = []
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                sum_loss.append(nn.as_scalar(loss))
                grad_wrt_w1, grad_wrt_h1, grad_wrt_o1, grad_wrt_w2, grad_wrt_h2, grad_wrt_o2 \
                        = nn.gradients(loss, [self.weight_1, self.hidden_1, self.output_1,
                                            self.weight_2, self.hidden_2, self.output_2])

                self.weight_1.update(grad_wrt_w1, self.learning_rate)
                self.hidden_1.update(grad_wrt_h1, self.learning_rate)
                self.output_1.update(grad_wrt_o1, self.learning_rate)

                self.weight_2.update(grad_wrt_w2, self.learning_rate)
                self.hidden_2.update(grad_wrt_h2, self.learning_rate)
                self.output_2.update(grad_wrt_o2, self.learning_rate)

            val_accuracy = dataset.get_validation_accuracy()
            print("Epoch {} with loss : {}, val accuracy: {}".format(i+1, sum(sum_loss) / 10, val_accuracy))
            if val_accuracy >= 0.85:
                break
