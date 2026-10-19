import numpy as np

from simon.activation_functions import sigmoid


class NeuralNetwork:
    input_feature_count: int
    hidden_layer_counts: list[int]
    layers: list[int]
    total_feature_count: int

    def __init__(self, hidden_layers: list[int], input_features: int) -> None:
        self.hidden_layer_counts = hidden_layers
        self.input_feature_count = input_features
        self.layers = [self.input_feature_count]
        self.layers.extend(self.hidden_layer_counts)
        self.layers.append(1)
        self.total_feature_count = 0
        for i in range(len(self.layers) - 1):
            self.total_feature_count += self.layers[i] * self.layers[i + 1]
            self.total_feature_count += self.layers[i + 1]

        # instance attributes now, not class attributes
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []

    def setup_weights_biases(self, features: np.ndarray) -> None:
        if len(features) != self.total_feature_count:
            raise ValueError(
                "Features are of length " + str(len(features)) +
                " but should be " + str(self.total_feature_count)
            )

        weights: list[np.ndarray] = []
        weight_offset = 0
        for layer in range(1, len(self.layers)):
            neurons: list[list[float]] = []
            for _ in range(self.layers[layer]):
                neuron_weights: list[float] = []
                for weight in range(self.layers[layer - 1]):
                    neuron_weights.append(features[weight + weight_offset])
                weight_offset += self.layers[layer - 1]
                neurons.append(neuron_weights)
            # layer's weights are rectangular now -> convert here
            weights.append(np.array(neurons, dtype=np.float64))

        arr_biases: np.ndarray = features[weight_offset:]
        biases: list[np.ndarray] = []
        bias_offset: int = 0
        for layer in range(1, len(self.layers)):
            n = self.layers[layer]
            biases.append(np.array(arr_biases[bias_offset:bias_offset + n], dtype=np.float64))
            bias_offset += n

        # assign only once both are fully built
        self._weights = weights
        self._biases = biases

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        # inputs: shape (n_samples, n_in)
        a = np.asarray(inputs, dtype=np.float64)
        for layer_weights, layer_biases in zip(self._weights, self._biases):
            a = sigmoid(a @ layer_weights.T + layer_biases)
        # a: shape (n_samples, 1) after the final layer -> flatten to (n_samples,)
        return a[:, 0]

    def unpack(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """features: (n_particles, total_feature_count)"""
        features = np.ascontiguousarray(features, dtype=np.float64)
        n_particles = features.shape[0]
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        offset = 0

        for layer in range(1, len(self.layers)):
            n_in, n_out = self.layers[layer - 1], self.layers[layer]
            block = features[:, offset:offset + n_in * n_out]
            weights.append(np.ascontiguousarray(
                block.reshape(n_particles, n_out, n_in).transpose(0, 2, 1)))
            offset += n_in * n_out

        for layer in range(1, len(self.layers)):
            n_out = self.layers[layer]
            biases.append(np.ascontiguousarray(features[:, offset:offset + n_out]))
            offset += n_out

        if offset != self.total_feature_count:
            raise ValueError(f"consumed {offset} genes, expected {self.total_feature_count}")
        return weights, biases

    def evaluate_batch(self, inputs: np.ndarray,
                       weights: list[np.ndarray],
                       biases: list[np.ndarray]) -> np.ndarray:
        """inputs: (n_samples, n_in) shared across particles -> (n_particles, n_samples)"""
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        n_samples = inputs.shape[0]
        n_particles, n_in, n_out = weights[0].shape

        # first layer: one wide GEMM across all particles
        stacked = weights[0].transpose(1, 0, 2).reshape(n_in, n_particles * n_out)
        a = (inputs @ stacked).reshape(n_samples, n_particles, n_out).transpose(1, 0, 2)
        a = sigmoid(a + biases[0][:, None, :])

        # later layers: activations differ per particle
        for layer_weights, layer_biases in zip(weights[1:], biases[1:]):
            a = sigmoid(np.matmul(a, layer_weights) + layer_biases[:, None, :])

        return a[:, :, 0]
