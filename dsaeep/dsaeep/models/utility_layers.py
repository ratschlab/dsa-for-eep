# ========================================
#
# Utility Layers
#
# ========================================
import math

import gin
import torch
import torch.nn as nn

gin.config.external_configurable(torch.nn.Identity, module="torch.nn")
gin.config.external_configurable(torch.nn.Linear, module="torch.nn")


def Activation(activation: str = None, dim=-1):
    """
    Returns an initialized layer for the given activation function.
    Based on: https://github.com/HazyResearch/safari

    Parameter
    ---------
    activation: str
        Name of the activation function. If None, returns an identity layer.
    dim: int
        Dimension along which to apply the activation function, if applicable.
        Default: -1
    """
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    else:
        raise NotImplementedError(f"hidden activation '{activation}' is not implemented")


@gin.configurable("MLP")
class MLP(nn.Module):
    """
    A basic multi-layer perceptron (MLP) module with
    configurable hidden layers and activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [],
        hidden_activation: str = "linear",
    ):
        super().__init__()

        assert isinstance(
            hidden_dims, list
        ), f"[{self.__class__.__name__}] hidden_dims must be a list of integers"
        assert len(hidden_dims) == 0 or all(
            [isinstance(dim, int) for dim in hidden_dims]
        ), f"[{self.__class__.__name__}] hidden_dims must be a list of integers"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation

        # single linear layer case
        if len(self.hidden_dims) == 0:
            self.mlp = nn.Linear(self.input_dim, self.output_dim)

        else:
            layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
            layers = []

            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if i < len(layer_dims) - 2:
                    layers.append(Activation(self.hidden_activation))

            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    "Positional Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"

    def __init__(self, emb, max_len=3000):
        super().__init__()
        emb_tensor = emb if emb % 2 == 0 else emb + 1
        pe = torch.zeros(max_len, emb_tensor)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_tensor, 2).float() * (-math.log(10000.0) / emb_tensor)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :emb]
