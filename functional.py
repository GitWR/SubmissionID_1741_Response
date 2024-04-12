from typing import Any

import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function as F


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


def init_bimap_parameter(W, type):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            if type == 0:
                xx = ni
            else:
                xx = no
            v = th.empty(xx, xx, dtype=W.dtype, device=W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:ni, :no]
            W.data[i, j] = vv


def init_bimap_parameter_identity(W):
    """ initializes to identity a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i, j] = th.eye(ni, no)


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


class CholeskyParameter(nn.Parameter):
    """ Parameter constrained to Cholesky """
    pass


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    # print(X)
    # print('-----')
    # print(W)
    return W.t().matmul(X).matmul(W)


def bimap_channels(X, W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
    return P


def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = th.linalg.eig(P[i, j])
                # S[i, j] = s[:, 0]
                S[i, j] = s[:]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = th.svd(P[i, j])
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='svd')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


class InvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Inv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Inv_op)


def geodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    '''
    M = CongrG(PowerEig.apply(CongrG(B, A, 'neg'), t), A, 'pos')[0, 0]
    return M


def cov_pool(f, reg_mode='add_id'):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    [batches, channels, features(64*64)]
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    f = f.view(f.shape[0], f.shape[1], -1)
    bs, n, T = f.shape
    X = f.matmul(f.transpose(-1, -2)) / (T - 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(X, 0.75)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def cov_pool_mu(f, reg_mode):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    alpha = 1
    bs, n, T = f.shape
    mu = f.mean(-1, True);
    f = f - mu
    X = f.matmul(f.transpose(-1, -2)) / (T - 1) + alpha * mu.matmul(mu.transpose(-1, -2))
    aug1 = th.cat((X, alpha * mu), 2)
    aug2 = th.cat((alpha * mu.transpose(1, 2), th.ones(mu.shape[0], 1, 1, dtype=mu.dtype, device=f.device)), 2)
    X = th.cat((aug1, aug2), 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(0.75)(X)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def add_id(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i] = P[i] + alpha * P[i].trace() * th.eye(P[i].shape[-1], dtype=P.dtype, device=P.device)
    return P


def dist_lc(x, y):
    lx = th.cholesky(x, upper=False)
    ly = th.cholesky(y, upper=False)
    dx = th.diag(lx)
    dy = th.diag(ly)
    a = lx + ly + th.matmul(dx, dy)
    return th.matmul(a, a.t())


def dist_riemann(x, y):
    '''
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n)
    :return:
    '''
    return LogEig.apply(CongrG(x, y, 'neg')).view(x.shape[0], x.shape[1], -1).norm(p=2, dim=-1)


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P).matmul(GG)
    return PP


def LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def ExpG(x, X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def karcher_step(x, G, alpha):
    '''
    One step in the Karcher flow
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def BaryGeom(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    with th.no_grad():
        G = th.mean(x, dim=0)[0, :, :]
        for _ in range(k):
            G = karcher_step(x, G, alpha)
        return G



def Cholesky_LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    # lx = th.cholesky(x, upper=False)
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        dx = th.diag(x[i][0])
        dL = th.diag(X)
        L = X - th.diag(dL)
        K = x[i][0] - th.diag(dx)
        Q[i][0] = K - L + th.diag(dL * th.log(1 / dL * dx))
    return Q
    # return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def Cholesky_ExpG(x, X):
    """ Exp mapping of x on the SPD manifold at X """
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    # lx = th.cholesky(x, upper=False)
    for i in range(x.shape[0]):
        dx = th.diag(x[i][0])
        dL = th.diag(X)
        L = th.tril(X, -1)
        K = th.tril(x[i][0], -1)
        Q[i][0] = K + L + th.diag(dL * th.exp((1 / dL) * dx))
    return Q


def Cholesky_ExpG_Ins(x, X):
    """ Exp mapping of x on the SPD manifold at X """
    n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(n, n, dtype=x.dtype, device=x.device)
    # lx = th.cholesky(x, upper=False)
    dx = th.diag(x)
    dL = th.diag(X)
    K = th.tril(x, -1)
    L = th.tril(X, -1)
    Q = K + L + th.diag(dL * th.exp((1 / dL) * dx))
    return Q


def cholesky_karcher_step(x, G, alpha):
    '''
    One step in the Karcher flow
    '''
    x_log = Cholesky_LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = Cholesky_ExpG(alpha * G_tan, G)[0, 0]
    return G


def cholesky_karcher_step_xx(x):
    '''
    One step in the Karcher flow
    '''
    # x_log = Cholesky_LogG(x, G)
    Q = th.zeros(x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
    P = th.zeros(x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        a = th.tril(x[i][0], -1)
        b = th.diag(th.log(th.diag(x[i][0])))
        Q = Q + a
        P = P + b
    G = Q/x.shape[0] + th.diag(th.exp(th.diag(P)/x.shape[0]))
    # G = x.mean(dim=0)[None, ...]
    # G = Cholesky_ExpG(alpha * G_tan, G)[0, 0]
    return G


def cholesky_dia(A):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    P = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(batch_size):
        a = A[i][0]
        P[i][0] = th.diag(th.diag(a))
        Q[i][0] = a - P[i][0]
    return Q, P


def cholesky_geodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: running mean
    :param B: mean
    :param t: momentum
    note1: th.diag(X) return diagonal matrix (nxn matrix return a 1xn vector)
    note2: X-th.diag(th.diag(X)) = th.tril(x,-1) return low triangular matrix
    :return:
    '''
    M = (1 - t) * (A - th.diag(th.diag(A))) + t * (B - th.diag(th.diag(B))) + th.diag(
        th.exp((1 - t) * th.log(th.diag(A)) + t * th.log(th.diag(B))))
    return M

def cholesky_geodesic_var(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: running mean
    :param B: mean
    :param t: momentum
    note1: th.diag(X) return diagonal matrix (nxn matrix return a 1xn vector)
    note2: X-th.diag(th.diag(X)) = th.tril(x,-1) return low triangular matrix
    :return:
    '''
    V = (1 - t) * A + t * B
    return V


def cholesky_pt(N, h, A, B):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(N):
        a = A[i][0]
        Q[i][0] = th.tril(a, -1) - th.tril(B, -1) + th.diag((1 / th.diag(B)) * th.diag(a))
    return Q


def cholesky_pt_scal(N, A, B, c):    #  here A is the centered samples, B is the variance
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    S = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    # B = torch.tensor(B)   # because this part using th.sqrt(), the float value B needs to be converted to tensor type
    for i in range(N):
        a = A[i][0]
        # a_bar = th.tril(a, -1) + th.diag(th.log(th.diag(a)))  # log map a at mean point (I)
        a_bar = (a / th.sqrt(B)) * c   # divide the variance and scaling in the tangent space
        # a_bar_exp = th.tril(a_bar, -1) + th.diag(th.exp(th.diag(a_bar)))
        S[i][0] = a_bar    # th.tril(a, -1) - th.tril(B, -1) + th.diag((1 / th.diag(B)) * th.diag(a))
    return S

def cholesky_pt_xx(N, h, A, B):
    batch_size, channels, n, n = A.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=A.dtype, device=A.device)
    for i in range(N):
        a = A[i][0]
        Q[i][0] = th.tril(a, -1) + th.tril(B, -1) + th.diag(th.diag(B) * th.diag(a))
    return Q


def cholesky_backward_Pt(dx):
    pass


class CholeskyPt(F):
    @staticmethod
    def forward(ctx, x, mean):
        N, h, n, n = x.shape
        Q = cholesky_pt(N, h, x, mean)  # X_5
        ctx.save_for_backward(x)  # save X_3,X_2
        ctx.save_for_backward(mean)  # save X_3,X_2
        return Q

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        X_3 = ctx.saved_variables[0]
        X_2 = ctx.saved_variables[1]
        return cholesky_backward_Pt(dx, )


class CholeskyBaryG(F):

    @staticmethod
    def forward(ctx, P):
        Q = choleskyBaryGeom(P)  # X_3
        ctx.save_for_backward(P)  # save X_3
        return Q

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        P = ctx.saved_variables[0]
        return cholesky_backward_mean(dx, P)


def cholesky_backward_mean(dx, X):
    downX, diaX = cholesky_dia(X)
    N, h, n, n = downX.shape
    logDiaX = th.mean(th.log(diaX))
    return dx/N + th.diag(1/downX.transpose(2, 3).matmul(th.exp(logDiaX/N).transpose(2, 3)).matmul(dx))


def choleskyBaryGeom(x):
    '''
    x after cholesky decomposition
    '''
    k = 1
    for _ in range(k):
        G = cholesky_karcher_step_xx(x)
    return G

def choleskyBaryGeom_var(x, mean):
    '''
    x after cholesky decomposition
    compute the batch variance
    '''
    dist = 0.0
    for i in range(x.shape[0]):
        a1 = th.tril(x[i][0], -1)  # the part of Euclidean
        b1 = th.diag(th.log(th.diag(x[i][0])))  # the part of SPD (diagonal part)
        a2 = th.tril(mean, -1)
        b2 = th.diag(th.log(th.diag(mean)))
        d1 = a1 - a2
        d2 = b1 - b2
        dist = dist + torch.norm(d1, p=2) + torch.norm(d2, p=2)
    var = dist / x.shape[0]
    return var


def CholeskyDe(x):
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        Q[i][0] = th.cholesky(x[i][0], upper=False)
    return Q


class CholeskyDecomposition(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        Q = CholeskyDe(P)
        ctx.save_for_backward(P)
        return Q

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        P = ctx.saved_variables[0]
        return cholesky_backward_de(dx, P)


def cholesky_backward_de(dx, Q, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input Q: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    Q_inverse = th.inverse(Q)
    dp = Q_inverse.transpose(2, 3).matmul(th.tril(Q.transpose(2, 3).matmul(dx))).matmul(Q_inverse)
    return dp


class CholeskyMulti(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        Q = CholeskyMu(P)
        ctx.save_for_backward(Q)
        return Q

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        Q = ctx.saved_variables[0]
        return cholesky_backward_mu(dx, Q)


def cholesky_backward_mu(dx, Q, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input Q: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    dp = 2 * dx * Q
    return dp


def CholeskyMu(x):
    batch_size, channels, n, n = x.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        Q[i][0] = x[i][0].matmul(x[i][0].t())
    return Q


def karcher_step_weighted(x, G, alpha, weights):
    '''
    One step in the Karcher flow
    Weights is a weight vector of shape (batch_size,)
    Output is mean of shape (n,n)
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mul(weights[:, None, None, None]).sum(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def bary_geom_weighted(x, weights):
    '''
    Function which computes the weighted Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Weights is a weight vector of shape (batch_size,)
    Output is (1,1,n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    # with th.no_grad():
    G = x.mul(weights[:, None, None, None]).sum(dim=0)[0, :, :]
    for _ in range(k):
        G = karcher_step_weighted(x, G, alpha, weights)
    return G[None, None, :, :]


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Log function and its derivative """
    _threshold = 1e-3

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S) ** 3


class Power_op():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return S ** cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S ** (cls._power - 1)


class Inv_op():
    """ Inverse function and its derivative """

    @classmethod
    def fn(cls, S, param=None):
        return 1 / S

    @classmethod
    def fn_deriv(cls, S, param=None):
        return log(S)


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)


def batchwFM(x, weights):
    '''
        Function which computes the Riemannian barycenter for a batch of data using the geodesic iterative mean estimator
        Input x : (batch_size, # of SPDs compute FM,n,n) to average along dim = 1
        Weights is a weight vector of shape (batch_size,# of SPDs compute FM)
        Output is (batch,1,n,n) Riemannian mean
     '''

    # with th.no_grad():
    G = x[:, 0].unsqueeze(dim=1)
    # weights = th.cat((weights, th.tensor([0]).float()))
    for i in range(1, x.shape[1]):
        sum = weights[:, :i + 1].sum(dim=1)
        w = weights[:, i] / sum
        G = batchgeodesic(G, x[:, i].unsqueeze(dim=1), w)

    return G


def batchConcat(P):
    P = (P)


def wFM(x, weights):
    '''
        Function which computes the Riemannian barycenter for a batch of data using the geodesic iterative mean estimator
        Input x is a batch of SPD matrices (batch_size,1,n,n) to average
        Weights is a weight vector of shape (batch_size,)
        Output is (n,n) Riemannian mean
     '''

    x = th.squeeze(x)
    # with th.no_grad():
    G = x[0]
    # weights = th.cat((weights,th.tensor([0]).float()))
    for i in range(1, x.shape[0]):
        sum = weights[:i + 1].sum()
        w = weights[i] / sum
        G = geodesic(G, x[i], w)
    return G


def Frechet_mean(x, weights):
    # x [64, 2, 64 ,64]
    # print(x.shape)
    # sum_weight = th.sum(weights)
    out = []
    # m1 = x[0][0]*weights[0]/sum_weight
    # out.append(m1)
    # print(weights.shape)

    for i in range(0, x.shape[0]):
        m = (weights[i][0] ** 2) * x[i][0] + (weights[i][1] ** 2) * x[i][1]  # count mean in Euclidean space
        m = m / ((weights[i][0] ** 2) + (weights[i][1] ** 2))
        # print(weights[i][0])
        # m = x[i][0]
        # for j in range(1, x.shape[1]):
        #     m += x[i][j]
        out.append(m)
    return th.stack(out).unsqueeze(1)


def Frechet_mean_m(x, weights, ):
    # x [64, 2, 64 ,64]
    # print(x.shape)
    out = []
    # m1 = x[0][0]*weights[0]/sum_weight
    # out.append(m1)
    # print(weights.shape)
    for i in range(0, x.shape[0]):
        sum = 0
        m = (weights[i][0] ** 2) * x[i][0]
        for j in range(0, x.shape[1] - 1):
            m = m + (weights[i][j + 1] ** 2) * x[i][j]
            sum += weights[i][j] ** 2
        m = m / sum
        out.append(m)
    return th.stack(out).unsqueeze(1)


def Frechet_mean_airm(x, weights):
    out = []
    for i in range(0, x.shape[0]):
        m = (weights[i] * weights[i] * dist_riemann(x[i][0], x[i][1]))  # count mean in airm space
        out.append(m)
    return th.stack(out).unsqueeze(1)


def Frechet_mean_transport(x, weights):
    out = []
    GG = SqminvEig.apply(th.unsqueeze(x[:, 1], 1))
    PP = th.unsqueeze(x[:, 0], 1)
    m = GG.matmul(PP).matmul(GG)
    return m


def Frechet_mean_log(x, weights):
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)

    @staticmethod
    def forward(ctx, P):
        out = []
        x = LogEig.apply(x)
        for i in range(0, x.shape[0]):
            # m = (weights[i] * weights[i] * x[i][0] + (1 - weights[i] * weights[i]) * x[i][1])/2  # count mean in Euclidean space
            m = x[i][0] + x[i][1]
            out.append(m)

    return ExpEig.apply(th.stack(out).unsqueeze(1))


def Frechet_mean_lc(x, weights):
    out = []
    for i in range(0, x.shape[0]):
        # m = (weights[i] * weights[i] * dist_lc(x[i][0], x[i][1]))   # count mean in lc space
        weight = weights[i] ** 2
        m = weight * dist_lc(x[i][0], x[i][1])
        out.append(m)
    return th.stack(out).unsqueeze(1)


def unfold(x, kernel_size, stride):
    out = []
    # x [64, 1, 64 ,64]
    s = x.view(-1, x.shape[2], x.shape[3])
    for i in range(1, x.shape[0] - kernel_size, stride):
        samples = []
        for j in range(1, kernel_size):
            samples.append(s[i + j])
        samples = th.stack(samples)
        out.append(samples)
    return th.stack(out)


class PowerEigbatch(F):
    """
     Input P: (batch_size,h) SPD matrices of size (n,n)
     Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
     """

    @staticmethod
    def forward(ctx, P, power):
        Power_opbatch._power = power
        X, U, S, S_fn = modeig_forward(P, Power_opbatch)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_opbatch), None


class Power_opbatch():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return th.pow(S, cls._power[:S.shape[0], None, None].repeat(1, 1, S.shape[2]))

    @classmethod
    def fn_deriv(cls, S, param=None):
        return cls._power[:S.shape[0], None, None].repeat(1, 1, S.shape[2]) * th.pow(
            S, cls._power[:S.shape[0], None, None].repeat(1, 1, S.shape[2]) - 1)


def bary_geom_weightedbatch(x, weights):
    '''
        Function which computes the Riemannian barycenter for a batch of data using the geodesic iterative mean estimator
        Input x : (batch_size, # of SPDs compute FM,n,n) to average along dim = 1
        Weights is a weight vector of shape (batch_size,# of SPDs compute FM)
        Output is (batch,1,n,n) Riemannian mean
     '''
    k = 1
    alpha = 1
    # with th.no_grad():
    G = x.mul(weights[:x.shape[0], :, None, None]).sum(dim=1)[:, None]
    for _ in range(k):
        G = karcher_step_weightedbatch(x, G, alpha, weights)
    return G


def karcher_step_weightedbatch(x, G, alpha, weights):
    '''
    One step in the Karcher flow
    x : [batch, #SPDs/channels, n, n]
    G : [batch, 1, n ,n]
    Weights is a weight vector of shape (batch_size,#SPDs/channels)
    Output is mean of shape (n,n)
    '''
    x_log = batchLogG(x, G)
    G_tan = x_log.mul(weights[:x.shape[0], :, None, None]).sum(dim=1)[:, None]
    G = batchExpG(alpha * G_tan, G)
    return G


def weightNormalize(weights):
    out = []
    for row in weights.view(weights.shape[0], -1):
        out.append(th.clamp(row, min=0.001, max=0.999))
    return th.stack(out).view(*weights.shape)


def batchCongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (batch_size,channels) SPD matrices of size (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G)
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G)
    PP = GG.matmul(P).matmul(GG)
    return PP


def batchgeodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (batch,1,n,n) to start from
    :param B: SPD matrix (batch,1,n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (batch,1,n,n) along the geodesic
    '''
    M = batchCongrG(PowerEigbatch.apply(batchCongrG(B, A, 'neg'), t), A, 'pos')
    return M


def batchLogG(x, X):
    """ Logarithmc mapping of x [batch, channels(#SPDs),n,n] on the SPD manifold at X [batch, 1, n, n]
        Output : [batch, channels(#SPDs), n, n]"""
    return batchCongrG(LogEig.apply(batchCongrG(x, X, 'neg')), X, 'pos')


def batchExpG(x, X):
    """ Exponential mapping of x [batch,channels,n,n] on the SPD manifold at X [batch,1,n,n]
        Output : [batch,channels(#SPDs),n,n]"""
    return batchCongrG(ExpEig.apply(batchCongrG(x, X, 'neg')), X, 'pos')


def init_pt_parameter(W):
    xx, xx = W.shape
    # v = th.empty(xx, xx, dtype=W.dtype, device=W.device).uniform_(0., 1.)
    # W.data = th.tril(v)
    th.eye(xx)


def retr(P):
    adjust_v = 1e-2
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    for k in range(batch_size):
        delta = P[k][0]
        thl = th.tril(delta, -1)
        Dg = th.diag(delta)
        for i in range(Dg.shape[0]):
            if Dg[i] < adjust_v:
                Dg[i] = adjust_v
        P[k][0] = thl + th.diag(Dg)

    return P

def clip(W):
    g = th.norm(W)
    thr = 10 # 10-->hdm05  4-->radar 5-->radar-10%
    W = W * min(thr/g, 1)
    return W