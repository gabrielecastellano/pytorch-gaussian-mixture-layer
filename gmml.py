import math
import torch

import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal


class GMML(nn.Module):
    def __init__(self, input_dim, d, n_class, n_component=1, cov_type="full", log_stretch_trick=False,
                 **kwargs):
        """A GMM layer, which can be used as a last layer for a classification neural network.
        Attributes:
            input_dim (int): Input dimension
            d (int): Reduced number of dimensions after random projection
            n_class (int): The number of classes
            n_component (int): The number of Gaussian components per class
            cov_type: (str): The type of covariance matrices. If "diag", diagonal matrices are used, which is
                computationally advantageous. If "full", the model uses full rank matrices that have high expression
                capability at the cost of increased computational complexity. If "tril", use lower triangular matrices.
            log_stretch_trick (bool): If True, computes the weighted sum over the logarithms of log probabilities, i.e.,
            log(log(p) instead of log(p). This can help in situations where a big portion of data is classified with
            maximum confidence (log_prob = 0, prob = 1) after the weighted sum.
        """
        super(GMML, self).__init__(**kwargs)
        assert input_dim > 0
        assert d > 0
        assert n_class > 1
        assert n_component > 0
        assert cov_type in ["diag", "full", "tril"]
        self.input_dim = input_dim
        self.d = d
        self.s = n_class
        self.g = n_component
        self.cov_type = cov_type
        self.n_total_component = n_component*n_class
        self.log_stretch_trick = log_stretch_trick

        # Dimensionality reduction
        self.bottleneck = nn.Linear(self.input_dim, self.d)
        self.bottleneck.requires_grad_(False)
        self.bottleneck.weight.data = get_achlioptas(self.input_dim, self.d).transpose(0, 1)

        # Free parameters
        self.mu_p = Parameter(torch.randn(self.s, self.g, self.d), requires_grad=True)
        self.omega_p = Parameter(torch.ones(self.s, self.g), requires_grad=True)
        sigma_data = torch.eye(self.d).reshape(1, 1, self.d, self.d).repeat(self.s, self.g, 1, 1)
        self.sigma_p = Parameter(sigma_data, requires_grad=True)

        # Sampled parameters
        self.omega = None
        self.distribution = None
        with torch.no_grad():
            self.parameter_enforcing()

    def init_mu(self, mu):
        with torch.no_grad():
            self.mu_p.data = mu

    def init_omega(self, omega):
        with torch.no_grad():
            self.omega_p.data = omega

    def forward(self, x):
        b = x.shape[0]
        x = self.bottleneck(x)
        x = x.reshape(b, 1, x.shape[1])
        log_wp = self.distribution.log_prob(x) + self.omega.reshape(self.s*self.g).log()
        if self.log_stretch_trick:
            log_wp = -torch.log(-log_wp)
        log_mixture_p = log_wp - log_wp.logsumexp(dim=-1, keepdim=True)
        return log_mixture_p.reshape(b, self.s, self.g).logsumexp(dim=-1)

    def parameter_enforcing(self):
        # OMEGA - should sum up to 1
        # omega = torch.softmax(self.omega_p, -1) / self.omega_p.shape[-2]
        omega = self.omega_p.data.clone()
        omega -= omega.min()
        omega = omega / omega.sum(dim=-1, keepdim=True)
        omega /= omega.shape[-2]
        self.omega = omega
        # SIGMA - symmetric positive definite
        device = self.sigma_p.device
        tli = torch.tril_indices(row=self.sigma_p.size(-2), col=self.sigma_p.size(-1), offset=-1).to(device)
        tui = torch.triu_indices(row=self.sigma_p.size(-2), col=self.sigma_p.size(-1), offset=1).to(device)
        sigma_p = self.sigma_p
        m = torch.matmul(sigma_p.transpose(-2, -1), sigma_p).to(device)
        sigma = m + torch.diag_embed(0.01*torch.mean(torch.linalg.eig(m).eigenvalues.real, dim=-1, keepdim=True)
                                     .repeat(1, 1, self.d)).to(device)
        if self.cov_type == "diag":
            sigma[:, :, tli[0], tli[1]] = 0
            sigma[:, :, tui[0], tui[1]] = 0
        elif self.cov_type == "tril":
            sigma[:, :, tui[0], tui[1]] = 0
        # MU - no transformation
        mu = self.mu_p
        # Initialize normal distribution using sampled MU and SIGMA
        if self.cov_type == "full":
            self.distribution = MultivariateNormal(mu.flatten(0, 1), covariance_matrix=sigma.flatten(0, 1))
        else:
            self.distribution = MultivariateNormal(mu.flatten(0, 1), scale_tril=sigma.flatten(0, 1))


def get_achlioptas(n, m, s=3):
    """
    Random Projection algorithm for Dimensionality Reduction from Achlioptas 2001 (Microsoft)
    https://dl.acm.org/doi/pdf/10.1145/375551.375608
    Args:
        n: input data dimension
        m: output desired dimension
        s: 1 / density. Should be greater or equal than 1 (density range is (0, 1]).

    Returns: n*m matrix that can be used for random-projection dimensionality reduction.

    """
    t = torch.rand(n, m)
    t = t.masked_fill(t.greater(1 - 1 / (2 * s)), -1)
    t = t.masked_fill(t.greater(1 / (2 * s)), 0)
    t = t.masked_fill(t.greater(0), 1)
    t = t*math.sqrt(s)/math.sqrt(m)
    return t
