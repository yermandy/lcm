import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiBiasedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_datasets):
        super(MultiBiasedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.biases = nn.ParameterList([])

        for i in range(n_datasets):
            self.biases.append(nn.Parameter(torch.Tensor(out_features)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        for bias in self.biases:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, input, datasets_numbers):
        outputs = input.matmul(self.weight.t())
        for output, dataset_number in zip(outputs, datasets_numbers):
            output += self.biases[dataset_number]
        return outputs


class LCM(nn.Module):
    def __init__(self, n_datasets, datasets_ages=None, ages=np.arange(1, 91)):
        super(LCM, self).__init__()

        dataset_ages = np.arange(1, 91)
        self.poly_deg_1 = nn.Parameter(torch.Tensor([dataset_ages ** d for d in range(2)]), requires_grad=False)
        self.poly_deg_2 = nn.Parameter(torch.Tensor([dataset_ages ** d for d in range(3)]), requires_grad=False)

        # self.datasets_ages = nn.ParameterList([])
        # for d_ages in datasets_ages:
        #     self.datasets_ages.append(nn.Parameter(torch.Tensor(d_ages), requires_grad=False))

        self.a_hat = nn.Parameter(torch.Tensor(ages), requires_grad=False)
        self.g_hat = nn.Parameter(torch.Tensor([-1, 1]), requires_grad=False)

        self.us = nn.ParameterList([])
        self.vs = nn.ParameterList([])
        self.zs = nn.ParameterList([])

        for i in range(n_datasets):
            self.us.append(nn.Parameter(torch.Tensor([[0, 1, 0], [0, 1, 0]])))            
            self.vs.append(nn.Parameter(torch.Tensor([[1, 0], [1, 0]])))
            # self.zs.append(nn.Parameter(torch.Tensor([[5, 0], [-5, 0]])))
            self.zs.append(nn.Parameter(torch.Tensor([[-1, -0.1, 0.0012], [1, 0.1, -0.0012]])))


    def forward(self, input, datasets_numbers, return_PagIx=False):

        ## p(a,g|x)
        PagIx = F.softmax(input, dim=1)

        PaIag_PgIag_d = {}

        for d in datasets_numbers:
            d = d.item()
            if d not in PaIag_PgIag_d:
                mu = (self.us[d] @ self.poly_deg_2).view(-1, 1)
                sigma = (self.vs[d] @ self.poly_deg_1).view(-1, 1)
                gamma = (self.zs[d] @ self.poly_deg_2).view(-1, 1)

                ## p(ĝ|a,g)
                PgIag = (self.g_hat * gamma).sigmoid()
                
                ## p(â|a,g)
                PaIag = F.softmax((-0.5 * ((mu - self.a_hat) ** 2)) / (sigma ** 2 + 1e-8), dim=1)

                PaIag_PgIag_d[d] = torch.cat((PaIag * PgIag[:, 0].unsqueeze(1), PaIag * PgIag[:, 1].unsqueeze(1)), dim=1)

        # outputs = torch.empty_like(input)
        outputs = torch.empty((len(input), 180), device=input.device)

        for i, (p, d) in enumerate(zip(PagIx, datasets_numbers)):
            d = d.item()

            ## p(â|a,g) * p(ĝ|a,g)
            PaIag_PgIag = PaIag_PgIag_d[d]

            ## p(â,ĝ|x,d) = p(â|a,g) * p(ĝ|a,g) * p(a,g|x)            
            output = (PaIag_PgIag * p.unsqueeze(1)).sum(0)
            output = output / output.sum()
            outputs[i] = output

        if return_PagIx:
            return outputs, PagIx

        return outputs