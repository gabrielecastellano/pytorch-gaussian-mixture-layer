import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from early_classifier.sgdm.torch_ard import LinearARD
from torch.distributions.multivariate_normal import MultivariateNormal


class GMML(nn.Module):
    def __init__(self, input_dim, d, n_class, n_component=1, cov_type="full", **kwargs):
        """An GMM layer, which can be used as a last layer for a classification neural network.
        Attributes:
            input_dim (int): Input dimension
            n_class (int): The number of classes
            n_component (int): The number of Gaussian components
            cov_type: (str): The type of covariance matrices. If "diag", diagonal matrices are used, which is computationally advantageous. If "full", the model uses full rank matrices that have high expression capability at the cost of increased computational complexity.
        """
        super(GMML, self).__init__(**kwargs)
        assert input_dim > 0
        assert d > 0
        assert n_class > 1
        assert n_component > 0
        assert cov_type in ["diag", "full", "tril"]
        self.input_dim = input_dim
        self.d = input_dim
        self.s = n_class
        self.g = n_component
        self.cov_type = cov_type
        self.n_total_component = n_component*n_class
        # self.ones_mask = (torch.triu(torch.ones(input_dim, input_dim)) == 1)
        # Bias term will be set in the linear layer so we omitted "+1"
        #if cov_type == "diag":
        #    self.H = int(2 * self.input_dim)
        #else:
        #    self.H = int(self.input_dim * (self.input_dim + 3) / 2)
        # Parameters
        self.bottleneck = nn.Identity() #nn.Linear(self.input_dim, self.d, bias=False)
        self.mu_p = Parameter(torch.zeros(self.s, self.g, self.d), requires_grad=True)
        self.omega_p = Parameter(torch.randn(self.s, self.g), requires_grad=True)
        self._last_log_likelihood = None
        if True: # self.cov_type == "full":
            sigma_p = torch.eye(self.d).reshape(1, 1, self.d, self.d).repeat(self.s, self.g, 1, 1)
            self.sigma_p = Parameter(sigma_p, requires_grad=True)
            # self.sigma_p = Parameter(torch.randn(self.s, self.g, self.d, self.d), requires_grad=True)
            # self.sigma_p.register_hook(zero_grad_hook(self.cov_type))
        '''
        elif self.cov_type == "diag":
            self.sigma_p = Parameter(torch.ones(self.s, self.g, self.d), requires_grad=True)
        elif self.cov_type == "tril":
            self.sigma_p_l = Parameter(torch.zeros(self.s, self.g, round(self.d*(self.d - 1)/2)), requires_grad=True)
            self.sigma_p_d = Parameter(torch.ones(self.s, self.g, self.d), requires_grad=True)
            # with torch.no_grad():
            #     # self.sigma_p[:, :, torch.arange(self.d)] = 1
            #     self.sigma_p[:, :, [round(2*n + n*(n-1)/2) for n in range(self.d)]] = 1
        '''
        # Enforced parameters
        self.distributions = {node: {0: None} for node in range(self.s)}
        with torch.no_grad():
            self.parameter_enforcing()

    def forward(self, x):
        output = torch.zeros(x.shape[0], self.s).to(self.sigma_p.device)
        x = self.bottleneck(x)
        for node in range(self.s):
            distribution = self.distributions[node][0]
            for i in range(x.shape[0]):
                output[i][node] = distribution.log_prob(x[i])
        return output

    def parameter_enforcing(self):
        # SIGMA - symmetric positive definite
        # with torch.no_grad():
        #     self.sigma_p[self.sigma_p < 2e-04] = 0
        device = self.sigma_p.device
        tli = torch.tril_indices(row=self.sigma_p.size(-2), col=self.sigma_p.size(-1), offset=-1).to(device)
        tui = torch.triu_indices(row=self.sigma_p.size(-2), col=self.sigma_p.size(-1), offset=1).to(device)
        if True: #self.cov_type == "full":
            sigma_p = self.sigma_p
            m = torch.matmul(sigma_p.transpose(-2, -1), sigma_p).to(device)
            sigma = m + 0.01*torch.mean(torch.linalg.eig(m).eigenvalues.real, dim=-1, keepdim=True)*torch.eye(self.d).to(device)
        if self.cov_type == "diag":
            sigma[:, :, tli[0], tli[1]] = 0
            sigma[:, :, tui[0], tui[1]] = 0
        elif self.cov_type == "tril":
            sigma[:, :, tui[0], tui[1]] = 0
        '''
        if self.cov_type == "diag":
            sigma_p = torch.diag_embed(self.sigma_p)
            m = torch.matmul(sigma_p.transpose(-2, -1), sigma_p)
            self.sigma = m + 0.01 * torch.diag_embed(torch.linalg.eig(m).eigenvalues.real)
        elif self.cov_type == "tril":
            d = self.sigma_p_d * self.sigma_p_d
            d += torch.mean(d) * 0.01
            self.sigma = torch.diag_embed(d)
            ti = torch.tril_indices(row=self.d, col=self.d, offset=-1)
            self.sigma[:, :, ti[0], ti[1]] = self.sigma_p_l
        '''
        # MU - no transformation
        mu = self.mu_p
        # OMEGA - should sum up to 1
        omega = torch.softmax(self.omega_p, -1)
        for node in range(self.s):
            if self.cov_type == "tril":
                self.distributions[node][0] = MultivariateNormal(mu[node][0], scale_tril=sigma[node][0])
            else:
                self.distributions[node][0] = MultivariateNormal(mu[node][0], covariance_matrix=sigma[node][0])


def zero_grad_hook(cov_type):
    def hook(grad):
        grad = grad.clone() # NEVER change the given grad inplace
        tli = torch.tril_indices(row=grad.size(-2), col=grad.size(-1), offset=-1)
        tui = torch.triu_indices(row=grad.size(-2), col=grad.size(-1), offset=1)
        if cov_type == "diag":
            grad[:, :, tli[0], tli[1]] = 0
            grad[:, :, tui[0], tui[1]] = 0
        elif cov_type == "tril":
            grad[:, :, tui[0], tui[1]] = 0
        return grad
    return hook
