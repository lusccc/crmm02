from torch import nn
from typing import Tuple, List, Union, Optional, T


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


prefix_size = 512
prefix_length = 10
gpt_embedding_size = 768
clip_project = MLP(
    (
        prefix_size,
        (prefix_size * prefix_length) // 2,
        gpt_embedding_size * prefix_length,
    )
)
print(clip_project)