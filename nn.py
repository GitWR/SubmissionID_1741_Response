import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional

dtype = th.double
device = th.device('cpu')


class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho, hi, ni, no, bType):
        super(BiMap, self).__init__()
        self._W = functional.StiefelParameter(th.empty(ho, hi, ni, no, dtype=dtype, device=device))
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no
        self.bType = bType
        functional.init_bimap_parameter(self._W, self.bType)

    def forward(self, X):
        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.LogEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.ReEig.apply(P)


class CholeskyDe(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, P):
        return functional.CholeskyDecomposition.apply(P)
        # return functional.CholeskyDe(P) # using the cholesky decomposition of Pytorch, facilitating backprop


class CholeskyMu(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, P):
        return functional.CholeskyMu(P)



class CholeskyBatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.9
        self.running_mean = th.eye(n, dtype=dtype, requires_grad=False)  ################################
        self.running_var = th.tensor(1.0)
        self.s = nn.Parameter(th.tensor(1.0), requires_grad=True)
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        # self.weight = functional.CholeskyParameter(th.eye(n, dtype=dtype))

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        if self.training:
            # mean = functional.BaryGeom(X_batched)
            mean = functional.choleskyBaryGeom(X_batched)
            var = functional.choleskyBaryGeom_var(X_batched, mean)
            with th.no_grad():
                self.running_mean.data = functional.cholesky_geodesic(self.running_mean, mean, self.momentum)
                self.running_var.data = functional.cholesky_geodesic_var(self.running_var, var, self.momentum)  # update running var
            X_centered = functional.cholesky_pt(N, h, X_batched, mean)
            X_scalling = functional.cholesky_pt_scal(N, X_centered, var, self.s)
        else:
            X_centered = functional.cholesky_pt(N, h, X_batched, self.running_mean)
            X_scalling = functional.cholesky_pt_scal(N, X_centered, self.running_var, self.s)  # batch scalling in the test stage
        # X_normalized = functional.CongrG(X_centered, self.weight, 'pos')
        # return X_centered
        # X_N = functional.cholesky_pt(N, h, X_centered, th.cholesky(self.weight, upper=False))
        return X_scalling.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CholeskyPt(nn.Module):

    def __init__(self, n):
        super(__class__, self).__init__()
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        # self.weight = functional.CholeskyParameter(th.eye(n,dtype=dtype, device=device))
        self.weight = functional.CholeskyParameter(th.empty(n, n, dtype=dtype, device=device))
        functional.init_pt_parameter(self.weight) # for opti of bias on L_+
        # self.weight = functional.SPDParameter(th.eye(n, dtype=dtype, device=device))  # for opti of bias on SPD


    def forward(self, X):
        N, h, n, n = X.shape
        # X_N = functional.cholesky_pt_xx(N, h, X, th.cholesky(self.weight, upper=False))
        X_N = functional.cholesky_pt_xx(N, h, X, self.weight)  # for opti of bias on L_+
        return X_N.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()
