import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torchviz import make_dot
import torch.nn.functional as F

class AdaptiveDropout(nn.Module):
    def __init__(self, initial_p=0.5, max_p=0.5, min_p=0.1):
        super(AdaptiveDropout, self).__init__()
        self.initial_p = initial_p
        self.max_p = max_p
        self.min_p = min_p
        self.p = initial_p  # 当前dropout概率

    def forward(self, x):
        # 检查模型是否处于训练模式
        if not self.training:
            return x

        # 根据输入x自适应调整dropout概率
        self.p = self.calculate_dropout_rate(x)

        # 应用dropout
        return F.dropout(x, self.p, self.training)

    def calculate_dropout_rate(self, x):
        # 实现自适应调整逻辑
        variance = x.var().item()
        threshold = 1.0  # 示例阈值，你需要根据自己的数据集来调整
        delta = 0.05  # 调整幅度，根据实验结果来选择适当的值

        # 示例：如果输入的方差大于阈值，则降低dropout概率
        if variance > threshold:
            new_p = max(self.min_p, self.p - delta)
        else:
            new_p = min(self.max_p, self.p + delta)
        return new_p


def create_classifier(input_dim, n_class):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.GELU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, n_class),
    )


class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(self, input_dim, output_dim, act='relu', num_hidden_lyr=2,
                 dropout_prob=0.5, return_layer_outs=False,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.return_layer_outs = return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = act
        self.activation = create_act(act)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 2)])))
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer, activation='linear')
        self.layers.append(final_layer)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]])

    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        # model.store_layer_output(self, layer_inputs[-1])
        if self.return_layer_outs:
            return layer_inputs[-1], layer_inputs
        else:
            return layer_inputs[-1]


def calc_mlp_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    dims = dims[:-1]
    return dims


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'linear':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def hf_loss_func(inputs, classifier, labels, num_labels, class_weights=None):
    logits = classifier(inputs)
    if type(logits) is tuple:
        logits, layer_outputs = logits[0], logits[1]
    else:  # simple classifier
        layer_outputs = [inputs, logits]
    if labels is not None:
        if num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            labels = labels.long()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        # make_dot(loss, show_attrs=True, ).render("mlp", directory='./', format="pdf")
    else:
        return None, logits, layer_outputs

    return loss, logits, layer_outputs


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError("initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`")


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


if __name__ == '__main__':
    mlp = MLP(input_dim=256, output_dim=64, act='relu', num_hidden_lyr=2,
              dropout_prob=0.5, return_layer_outs=False,
              hidden_channels=[128, 96], bn=False)
    print(mlp)
    x = torch.randn(2, 256)
    make_dot(mlp(x), show_attrs=True, ).render("mlp", directory='./', format="pdf")
