import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

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
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        for bias in self.biases:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, input, datasets_numbers):
        outputs = input.matmul(self.weight.t())
        for output, dataset_number in zip(outputs, datasets_numbers):
            output += self.biases[dataset_number]
        return outputs


class LCM(nn.Module):
    def __init__(self, n_datasets, datasets_ages, ages=[], poly_degree=3):
        super(LCM, self).__init__()

        self.poly = nn.Parameter(torch.Tensor([ages ** d for d in range(poly_degree)]), requires_grad=False)

        self.datasets_ages = nn.ParameterList([])
        for d_ages in datasets_ages:
            self.datasets_ages.append(nn.Parameter(torch.Tensor([d_ages]).T, requires_grad=False))

        self.a_hat = nn.Parameter(torch.Tensor(ages), requires_grad=False)
        self.g_hat = nn.Parameter(torch.Tensor([-1, 1]), requires_grad=False)

        self.zs = nn.ParameterList([])
        self.us = nn.ParameterList([])
        self.vs = nn.ParameterList([])
        for i in range(n_datasets):
            for dataset_poly_coeffs in [self.zs, self.us, self.vs]:
                # TODO: initialize wisely
                poly_coeffs = nn.Parameter(torch.Tensor(2, poly_degree))
                nn.init.kaiming_uniform_(poly_coeffs, a=sqrt(5))
                dataset_poly_coeffs.append(poly_coeffs)

    def forward(self, input, datasets_numbers):

        ## p(a,g|x)
        PagIx = F.softmax(input, dim=1)

        PaIag_PgIag_d = {}
        
        for d in datasets_numbers:
            d = d.item()
            if d not in PaIag_PgIag_d:
                zeta = (self.zs[d] @ self.poly).flatten()
                mu = (self.us[d] @ self.poly).flatten()
                sigma = (self.vs[d] @ self.poly).flatten()
                norm = torch.exp(-0.5 * (mu - self.datasets_ages[d]) ** 2 / (sigma ** 2 + 1e-4)).sum(dim=0) + 1e-4
                
                ## p(ĝ|a,g)
                PgIag = (self.g_hat * zeta.unsqueeze(1)).sigmoid()

                ## p(â|a,g)
                PaIag = torch.exp(-0.5 * (mu.unsqueeze(1) - self.a_hat) ** 2 / (sigma.unsqueeze(1) ** 2 + 1e-4)) / norm.unsqueeze(1)

                PaIag_PgIag_d[d] = torch.cat((PaIag * PgIag[:, 0].unsqueeze(1), PaIag * PgIag[:, 1].unsqueeze(1)), dim=1)

        outputs = torch.empty_like(input)

        for i, p, d in zip(range(len(input)), PagIx, datasets_numbers):
            d = d.item()

            ## p(â|a,g) * p(ĝ|a,g)
            PaIag_PgIag = PaIag_PgIag_d[d]

            ## p(â,ĝ|x,d) = p(â|a,g) * p(ĝ|a,g) * p(a,g|x)            
            outputs[i] = (PaIag_PgIag * p.unsqueeze(1)).sum(1)

        return outputs