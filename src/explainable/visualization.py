from src.model.model import ReversibleLongBert
import torch
from torchviz import make_dot

b, n, d = 128, 3072, 512
x = torch.randn(b, n, d, requires_grad=True).to("cuda")
h = 8
blocks = 2
dilation_rates = [[1, 3, 5], [1,3,5]]
segment_lengths = [[128, 512, 1024], [128, 512, 1024]]

att = ReversibleLongBert(blocks, h, d, dilation_rates=dilation_rates, segment_lengths=segment_lengths,
                         reversible=True).to("cuda")
y = att(x)

dot = (make_dot(y.mean(), params=dict(att.named_parameters()), show_attrs=True, show_saved=True))

dot.render("model_graph")