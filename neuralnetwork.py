from engine import Value
import random
from typing import Any


class Neuron:
    def __init__(self, input_size):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, x) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, input_size: int, layers: list[int]) -> None:
        all_layers = [input_size] + layers
        self.layers = [
            Layer(all_layers[i], all_layers[i + 1]) for i in range(len(layers))
        ]

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
