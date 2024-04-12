import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional
from .functional import batchLogG, batchExpG

dtype = th.double
device = th.device('cpu')


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, X_1, X_2):
        return th.cat([X_1, X_2], 1)


class Avg(nn.Module):
    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, x):
        for i in range(x.shape[1]-1):
            x[:][0] = x[:][0] + x[:][i+1]
        x = x[:, 0]/x.shape[1]
        return x.unsqueeze(1)


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


class ExpEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of Exp eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.ExpEig.apply(P)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.SqmEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.ReEig.apply(P)


class BaryGeom(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, x):
        return functional.BaryGeom(x)


class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = th.eye(n, dtype=dtype)  ################################
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight = functional.SPDParameter(th.eye(n, dtype=dtype))

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        if self.training:
            mean = functional.BaryGeom(X_batched)
            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)
            X_centered = functional.CongrG(X_batched, mean, 'neg')
        else:
            X_centered = functional.CongrG(X_batched, self.running_mean, 'neg')
        X_normalized = functional.CongrG(X_centered, self.weight, 'pos')
        return X_normalized.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


class CholeskyDe(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, P):
        return functional.CholeskyDecomposition.apply(P)
        # return functional.CholeskyDe(P) # using the cholesky decomposition of Pytorch, facilitating backprop


class retr(nn.Module):

    def forward(self, P):
        return functional.retr(P)


class CholeskyMu(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, P):
        return functional.CholeskyMu(P)


class CholeskyBatchMean(nn.Module):
    """
    Input X: (N,h) LT matrices of size (n,n) with h channels and batch size N
    Output P:  mean matrices
    SPD parameter of size (n,n)
    """
    def forward(self, P):
        return functional.CholeskyBaryG.apply(P)


class CholeskyBatchCenter(nn.Module):
    """
    Input X: (N,h) LT matrices of size (n,n) with h channels and batch size N
    Output P:  mean matrices
    SPD parameter of size (n,n)
    """
    def __init__(self, n):
        super(__class__, self).__init__()
        self.momentum = 0.2
        self.running_mean = th.eye(n, dtype=dtype)

    def forward(self, x, mean):
        N, h, n, n = x.shape
        X_batched = x.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        if self.training:
            with th.no_grad():
                self.running_mean.data = functional.cholesky_geodesic(self.running_mean, mean, self.momentum)
            X_centered = functional.CholeskyPt.apply(X_batched, mean)
        else:
            X_centered = functional.cholesky_pt(N, h, X_batched, self.running_mean)
            #  X_normalized = functional.CongrG(X_centered, self.weight, 'pos') 暂时不bias
        return X_centered.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()


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
        # for opti of bias on SPD


class CholeskyPtSPD(nn.Module):

    def __init__(self, n):
        super(__class__, self).__init__()
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        # self.weight = functional.CholeskyParameter(th.eye(n,dtype=dtype, device=device))
        # self.weight = functional.CholeskyParameter(th.empty(n, n, dtype=dtype, device=device))
        # functional.init_pt_parameter(self.weight)
        self.weight = functional.SPDParameter(th.eye(n, dtype=dtype, device=device))


    def forward(self, X):
        N, h, n, n = X.shape
        X_N = functional.cholesky_pt_xx(N, h, X, th.cholesky(self.weight, upper=False))
        # X_N = functional.cholesky_pt_xx(N, h, X, self.weight)
        return X_N.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()



class CovPool(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode='add_id'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        f = th.squeeze(f, 1)
        return functional.cov_pool(f, self._reg_mode)


class CovPoolBlock(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,L,n,T)
    Output X: L covariance matrices, shape (batch_size,L,1,n,n)
    """

    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        ff = [functional.cov_pool(f[:, i, :, :], self._reg_mode)[:, None, :, :, :] for i in range(f.shape[1])]
        return th.cat(ff, 1)


class CovPoolMean(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """

    def __init__(self, reg_mode='mle'):
        super(__class__, self).__init__()
        self._reg_mode = reg_mode

    def forward(self, f):
        return functional.cov_pool_mu(f, self._reg_mode)


class BatchFrechetMean(nn.Module):
    """
    Input f: (batch_size,1,n,n)
    Output x: FrechetMean of size (batch_size,1,n,n)
    """

    def __init__(self, batch_size):
        super(BatchFrechetMean, self).__init__()
        self.batch_size = batch_size
        # self.weights = nn.Parameter(th.rand(self.batch_size, requires_grad=True))
        self.weights = nn.Parameter(th.rand(self.batch_size, 2, requires_grad=True))

    def forward(self, f):
        x = functional.batchwFM(f, self.weights)
        return x


class ADDMean(nn.Module):
    """
    Input f: (batch_size,1,n,n)
    Output x: FrechetMean of size (batch_size,1,n,n)
    """

    def __init__(self, batch_size):
        super(ADDMean, self).__init__()
        self.batch_size = batch_size
        self.weights = nn.Parameter(th.rand(self.batch_size, 2, requires_grad=True))

    def forward(self, f):
        return functional.Frechet_mean(f, self.weights)

class ADDMeanM(nn.Module):
    """
    Input f: (batch_size,1,n,n)
    Output x: FrechetMean of size (batch_size,1,n,n)
    """

    def __init__(self, batch_size, n):
        super(ADDMeanM, self).__init__()
        self.batch_size = batch_size
        self.weights = nn.Parameter(th.rand(self.batch_size, n, requires_grad=True))

    def forward(self, f):
        return functional.Frechet_mean_m(f, self.weights)


class BatchConcat(nn.Module):
    """
    Input f: (batch_size,1,n,n)
    Output x: FrechetMean of size (batch_size,1,n,n)
    """

    def __init__(self, batch_size):
        super(BatchConcat, self).__init__()
        self.batch_size = batch_size

    def forward(self, f):
        return functional.BatchDiag(f)


class SPDCov2D(nn.Module):
    """
    Input f
    output X
    """

    def __init__(self, batch_size):
        super(SPDCov2D, self).__init__()
        # self.kern_size = kern_size
        # self.stride = stride
        self.batch_size = batch_size
        self.weight_matrix = nn.Parameter(th.randn(self.batch_size,
                                                   requires_grad=True))

    # x: [batches, 1, 30, 30]
    def forward(self, x):
        # x = functional.unfold(x, self.kern_size, self.stride)
        # print(self.weight_matrix)
        # weighted Frechet expectation
        """
        M1 = X1 
        Mn = TXn Mn-1(Wn/Σn,j=1 wj))
        """
        # out = []
        # for i in range(1, x.shape[0]):
        # out.append(functional.Frechet_mean(x, functional.weightNormalize(self.weight_matrix)))

        # out = th.stack(out)
        # return out.double()
        out = functional.Frechet_mean(x, functional.weightNormalize(self.weight_matrix))
        out = out.view(out.shape[0], 1, out.shape[1], out.shape[2])
        return out


## NEW ReLu tangent function that can be used instead of ReEIG
## It's simple : SPD -> projection into tangent map -> ReLu -> Projection into manifold
## TO BE ADDED INTO nn.py
class Relut(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def __init__(self):
        super(Relut, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, P):
        reluproj = self.relu(batchLogG(P, P))
        exp = batchExpG(reluproj, P)
        return exp

