from .image_classification import (
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
    Cifar10Net,
    Cifar10NeuralNetwork,
    FashionMnistNeuralNetwork,
    FashionMnistConvolutionalNeuralNetwork,
)
from .diagonal_network import DiagonalNeuralNetwork, DiagonalNeuralNetworkDataset
from .ffn import FFNNeuralNetwork, IrisFFNNeuralNetwork
from .dnnODEModel import DNNClassificationODEModel
from .longLinear import LongLinearModel
from .decoder_tf import greedy_decode, CausalLanguageModel, Tokenizer, get_collate_fn
from .gpt2 import GPT2, GPT2Tokenizer


MODELMAPPING = {
    "NeuralNetwork": NeuralNetwork,
    "ConvolutionalNeuralNetwork": ConvolutionalNeuralNetwork,
    "Cifar10Net": Cifar10Net,
    "Cifar10NeuralNetwork": Cifar10NeuralNetwork,
    "FashionMnistNeuralNetwork": FashionMnistNeuralNetwork,
    "FashionMnistConvolutionalNeuralNetwork": FashionMnistConvolutionalNeuralNetwork,
    "DiagonalNeuralNetwork": DiagonalNeuralNetwork,
    "FFNNeuralNetwork": FFNNeuralNetwork,
    "IrisFFNNeuralNetwork": IrisFFNNeuralNetwork,
    "DNNClassificationODEModel": DNNClassificationODEModel,
    "LongLinearModel": LongLinearModel,
    "GPT2": GPT2,
    "CausalLanguageModel": CausalLanguageModel,
}
