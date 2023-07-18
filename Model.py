from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import math


class SVM(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(SVM, self).__init__()
        self.args = args

        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.model(x)
        return y

class DNN(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(DNN, self).__init__()
        self.args = args

        self.layers = []
        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus
        }[args.nonlin]

        self.layers.append(self.fullConnectedLayer(input_dim, 1024, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(1024, 2048, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(2048, 256, args.batchnorm))
        self.layers.append(nn.Linear(256, output_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.model(x)
        return y

    def fullConnectedLayer(self, input_dim, output_dim, batchnorm=False):
        if batchnorm:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                self.activate()
            )
        else:
            return nn.Sequential(
                nn.Dropout(p=self.args.dropout_rate, inplace=False),
                nn.Linear(input_dim, output_dim),
                self.activate()
            )


class ATTDNN(nn.Module):
    def __init__(self, args, input_dim, output_dim) -> None:
        super(ATTDNN, self).__init__()
        self.args = args

        self.layers = []
        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus
        }[args.nonlin]

        self.layers.append(AttentionLayer(args, input_dim, args.nClasses*input_dim, args.nClasses))
        self.layers.append(self.fullConnectedLayer(input_dim, 1024, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(1024, 2048, args.batchnorm))
        self.layers.append(self.fullConnectedLayer(2048, 256, args.batchnorm))
        self.layers.append(nn.Linear(256, output_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.model(x)
        return y

    def fullConnectedLayer(self, input_dim, output_dim, batchnorm=False):
        if batchnorm:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                self.activate()
            )
        else:
            return nn.Sequential(
                nn.Dropout(p=self.args.dropout_rate, inplace=False),
                nn.Linear(input_dim, output_dim),
                self.activate()
            )


class AttentionLayer(nn.Module):
    def __init__(self, args, in_size, hidden_size, num_attention_heads):
        """
        Softmax(Q@K.T)@V
        """
        super(AttentionLayer, self).__init__()

        self.activate = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'softplus': nn.Softplus,
            'sigmoid': nn.Sigmoid
        }[args.nonlin]

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.key_layer = nn.Linear(in_size, hidden_size)
        self.query_layer = nn.Linear(in_size, hidden_size)
        self.value_layer = nn.Linear(in_size, hidden_size)

    def forward(self, x):
        key = self.activate()(self.key_layer(x))  # (batch, hidden_size)
        query = self.activate()(self.query_layer(x))
        value = self.activate()(self.value_layer(x))

        key_heads = self.trans_to_multiple_heads(key)  # (batch, heads_num, head_size)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(key_heads, query_heads.permute(0, 2, 1))  # (batch, heads_num, heads_num)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_normalized = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_scores_normalized, value_heads)  # (batch, heads_num, head_size)

        return out.mean(dim=-2)


    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x


class UDNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, hidden_dim=2048) -> None:
        super(UDNN, self).__init__()
        self.args = args
        self.activate = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'softplus': nn.Softplus
        }[args.nonlin]

        self.initial_layer = self.fullConnectedLayer(input_dim, args.initial_dim) # expand dim
        
        self.down_1 = self.fullConnectedLayer(hidden_dim, hidden_dim/2)
        self.down_2 = self.fullConnectedLayer(hidden_dim/2, hidden_dim/4)
        self.down_3 = self.fullConnectedLayer(hidden_dim/4, hidden_dim/8)

        self.up_1 = self.fullConnectedLayer(hidden_dim/8, hidden_dim/4)
        self.up_2 = self.fullConnectedLayer(hidden_dim/4, hidden_dim/2)
        self.up_3 = self.fullConnectedLayer(hidden_dim/2, hidden_dim)
        
        self.output_layer = self.fullConnectedLayer(args.initial_dim, output_dim)

    def forward(self, x):

        # initial_out = self.initial_layer(x)  # 2048
        # down_1_out = self.down_1(initial_out)  # 1024
        # down_2_out = self.down_2(down_1_out)  # 512

        # down_3_out = self.down_3(down_2_out)  # 256
        
        # up_1_out = self.up_1(down_3_out)  # 512
        # up_2_out = self.up_2((up_1_out+down_2_out)/2)  # 1024
        # up_3_out = self.up_3((up_2_out+down_1_out)/2)  # 2048
        # out = self.output_layer((up_3_out+initial_out)/2)

        initial_out = self.initial_layer(x)  # 256
        up_1_out = self.up_1(initial_out)  # 512
        up_2_out = self.up_2(up_1_out)  # 1024
        up_3_out = self.up_3(up_2_out)  # 2048

        down_1_out = self.down_1(up_3_out)  # 1024
        down_2_out = self.down_2(down_1_out)  # 512
        down_3_out = self.down_3(down_2_out)  # 256
        out = self.output_layer(down_3_out)

        return out

    def fullConnectedLayer(self, input_dim, output_dim):

        return nn.Sequential(
            nn.Dropout(p=self.args.dropout_rate, inplace=False),
            nn.Linear(int(input_dim), int(output_dim)),
            self.activate()
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN'], default='DNN')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=10240)  # 486400
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu', choices=["cpu", "cuda"])
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
    parser.add_argument('--save_dir', default='./Results', type=str)
    parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "softplus"])
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
    parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                       help='fix random seeds and set cuda deterministic')
    parser.add_argument('--warmup_steps', default=10, type=int)
    parser.add_argument('--warmup_start_lr', default=1e-5, type=float)
    parser.add_argument('--power', default=0.5, type=float)
    parser.add_argument('-batchnorm', '--batchnorm', action='store_true')
    parser.add_argument('--dropout_rate', default=0., type=float)
    parser.add_argument('--nClasses', default=79, type=int)
    parser.add_argument('--initial_dim', default=256, type=int)

    args = parser.parse_args()
    args.device = torch.device('cuda')

    model = UDNN(args, 61, 79).to(args.device)
    x = torch.rand(10, 61).to(args.device)

    x.requires_grad_()
    y = model(x)
    # print(model)
    print(y.size())
    # target = torch.ones_like(y)
    # loss = torch.nn.MSELoss()(y, target)
    # print(loss)
    # loss.backward()