import warnings

from xformers.components.reversible import ReversibleBlock, ReversibleSequence
import torch
import torch.nn as nn


class ReversibleWrapper(ReversibleBlock):
    """
        A wrapper class for the ReversibleBlock that incorporates layer normalization.

        This class extends the functionality of ReversibleBlock by optionally adding
        a layer normalization step in the forward pass. This can help stabilize
        the learning process in deep neural networks.
    """
    def __init__(self, f: nn.Module, g: nn.Module, split_dim: int = -1):
        """
        Initialization of ReversibleWrapper.

        :param f: A neural network module to be used as the 'f' function in the reversible block.
        :param g: A neural network module to be used as the 'g' function in the reversible block.
        :param split_dim: The dimension along which the input tensor should be split. Default is -1.
        """
        super().__init__(f, g, split_dim)
        self.layer_norm = nn.Identity()

    def apply_layer_norm(self, model_dim: int):
        """
        Applies layer normalization to the reversible block.

        This method replaces the identity layer with a layer normalization layer.
        It should be called before the forward pass if layer normalization is desired.

        :param model_dim: The dimension of the model over which normalization is to be applied.
        """
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-12)

    def forward(self, x: torch.Tensor, f_args={}, g_args={}):
        """
        Defines the forward pass with optional layer normalization.

        Splits the input tensor into two parts, processes them with the 'f' and 'g' functions,
        applies layer normalization if it's not set to identity, and then concatenates the outputs.

        :param x: The input tensor to the reversible block.
        :param f_args: Optional arguments for the 'f' function.
        :param g_args: Optional arguments for the 'g' function.
        :return: The output tensor after processing and recombination.
        """
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = None, None

        if self.layer_norm == nn.Identity:
            warnings.warn("No layer norm applied, if not desired then call apply_layer_norm() before forward pass")

        with torch.no_grad():
            y1 = self.layer_norm(x1 + self.f(x2, record_rng=self.training, **f_args))
            y2 = self.layer_norm(x2 + self.g(y1, record_rng=self.training, **g_args))

        return torch.cat([y1, y2], dim=self.split_dim)


class ReversibleSequenceWrapper(ReversibleSequence):
    """
    A wrapper for a sequence of reversible blocks.

    This class manages a sequence of reversible blocks, facilitating the construction of complex reversible architectures.
    It optionally supports layer normalization for each block in the sequence.

    """
    def __init__(self, blocks: nn.ModuleList, model_dim: int = None, layer_norm: bool = False):
        """
        Initialization of ReversibleSequenceWrapper.

        :param blocks:  A list of tuples, where each tuple contains two nn.Module instances (f, g) for each reversible block.
        :param model_dim: The dimension of the model for layer normalization. Required if layer_norm is True.
        :param layer_norm: Flag to indicate whether layer normalization should be applied to each block.
        """
        super().__init__(blocks)
        if model_dim is None and layer_norm:
            raise ValueError("When layer norm should be applied, then provide also model_dim argument")
        self.blocks = nn.ModuleList([ReversibleWrapper(f, g) for f, g in blocks])

        if layer_norm:
            for block in self.blocks:
                block.apply_layer_norm(model_dim)


def reversible_layer_constructor(f: nn.Module, g: nn.Module, model_dim: int = None, layer_norm: bool = False) -> ReversibleSequenceWrapper:
    """
    Constructs a reversible layer sequence wrapper.

    This function creates a `ReversibleSequenceWrapper` given the functions `f` and `g`, along with model dimensions
    and a layer normalization flag. It is a utility for easy construction of a reversible sequence with one block.

    :param f: The first function (module) to be used in the reversible block.
    :param g: The second function (module) to be used in the reversible block.
    :param model_dim: The dimension of the model for layer normalization. Required if layer_norm is True.
    :param layer_norm: Flag to indicate whether layer normalization should be applied..
    :return: An instance of ReversibleSequenceWrapper containing the constructed reversible block.
    """
    assert isinstance(f, nn.Module)
    assert isinstance(g, nn.Module)

    return ReversibleSequenceWrapper(torch.nn.ModuleList([torch.nn.Sequential(f, g)]), model_dim, layer_norm)


class ReversibleResidualBlock(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module, dim_model: int, layer_norm: bool = False):
        """
        Reversible Layer which avoids storing activations. Activations are recomputed during backward pass.
        Refer to equations (31) to (43) and algorithm 1 for an understanding of the process.
        :param f: Function F which should ideally be some kind of attention.
        :param g: Function F which should ideally be a feedforward layer.
        :param layer_norm: Whether to apply layer norm after attention and feedforward layer
        """
        super().__init__()
        assert isinstance(f, nn.Module)
        assert isinstance(g, nn.Module)

        self.rev_layer = reversible_layer_constructor(f, g, dim_model, layer_norm)
    #    self.rev_layer.apply_layer_norm(dim_model)

    def forward(self, x: torch.Tensor):
        return self.rev_layer(x)






