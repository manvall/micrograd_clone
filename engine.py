import math


class Value:
    def __init__(self, data, _children=tuple(), _op=str(), label=str()):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad: float = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label}, grad={self.grad})"

    def __add__(self, other) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward

        return out

    def __radd__(self, other) -> "Value":
        return self + other

    def __sub__(self, other) -> "Value":
        return -self + other

    def __rsub__(self, other) -> "Value":
        return self - other

    def __neg__(self) -> "Value":
        return self * -1

    def __mul__(self, other) -> "Value":
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __rmul__(self, other) -> "Value":
        return self * other

    def __truediv__(self, other) -> "Value":
        return self * other**-1

    def __rtruediv__(self, other) -> "Value":
        return self**-1 * other

    def __pow__(self, other: int | float) -> "Value":
        assert isinstance(other, (int, float)), "Only supporting int/float for now"
        out = Value(self.data**other, _children=(self,), label=f"**{other}")

        def backward():
            self.grad += (
                other * ((self.data) ** (other - 1)) * out.grad
            )  # d(X**m) is m * x**(m-1)

        out._backward = backward

        return out

    def tanh(self) -> "Value":
        n = self.data
        tanh_value = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)

        out = Value(tanh_value, (self,), _op="tanh")

        def backward():
            self.grad += (1 - (tanh_value) ** 2) * out.grad

        out._backward = backward

        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), _children=(self,), label="exp")

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward

        return out

    def backward(self):
        # Build a topological sort
        topo = []
        visited = set()

        def build_topo(root: Value) -> list[Value]:
            if root not in visited:
                visited.add(root)
                for child_node in root._prev:
                    build_topo(child_node)
                topo.append(root)
            return topo

        topo = build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
