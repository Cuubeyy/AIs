class NeuralNetwork:
    def __init__(self, layers):
        if len(layers) < 2:
            print("Not enough layers!")
            return

        self.layers_size = layers
        self.layers = []
        self.create_graph()

    def create_graph(self):
        for i, layer_size in enumerate(self.layers_size):
            layer = Layer(layer_size)
            self.layers.append(layer)

            # Connect nodes from the current layer to nodes in the next layer
            if i > 0 and layer:
                previous_layer = self.layers[-2]
                for node in layer.nodes:
                    for next_node in previous_layer.nodes:
                        connection = Connection(node, next_node, weight=0.1)  # Set an initial weight here
                        node.connections.append(connection)

    def calculate_result(self, inputs):
        self.layers[0].set_input(inputs)
        output = [inputs]
        for layer in self.layers:
            output.append(layer.get_node_values(output[-1]))
        output.pop(0)
        return output


class Layer:
    def __init__(self, size):
        self.size = size
        self.nodes = []
        self.create_nodes()

    def create_nodes(self):
        for node_index in range(self.size):
            self.nodes.append(Node(1, 0.3))

    def set_input(self, inputs):
        for inp_index, inp in enumerate(inputs):
            self.nodes[inp_index].value = inp
            self.nodes[inp_index].input_node = True

    def get_node_values(self, inputs) -> list[int]:
        values = []
        for node in self.nodes:
            values.append(node.calculate_output(inputs))
        return values


class Node:
    def __init__(self, bias, learning_rate):
        self.value = 0
        self.bias = bias
        self.learning_rate = learning_rate
        self.input_node = False
        self.connections = []

    def calculate_output(self, input_values):
        if self.input_node:
            return self.value

        self.value = self.bias

        for connection in self.connections:
            self.value += connection.weight * connection.start.value

        return self.value


class Connection:
    def __init__(self, start, target, weight):
        self.start = start
        self.target = target
        self.weight = weight
        start.connections.append(self)


if __name__ == "__main__":
    nw = NeuralNetwork([2, 3, 2])

    print(nw.calculate_result([1, 2]))
