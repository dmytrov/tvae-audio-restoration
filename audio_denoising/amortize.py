import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import numpy as np
from tqdm import tqdm

EPS = np.finfo("float32").eps  # The difference between 1.0 and the next smallest representable float larger than 1.0


def compute_probabilities(log_f_joint):
    """ Returns vectors probabilities (normalized)
        :param log_f        : [N, K]    log-joint probability of (X, Kset) 
    """
    f_joint = (log_f_joint - log_f_joint.max(dim=-1, keepdim=True)[0]).exp()
    Ps = f_joint / (f_joint.sum(-1 , keepdim=True))
    return Ps


def compute_marginal(Kset, log_f_joint):
    """ Returns bits marginal probability
        :param Kset         : [N, K, H] truncated binary posterior sets
        :param log_p_joint  : [N, K]    log-joint probability of (X, Kset) 
    """
    Ps = compute_probabilities(log_f_joint)
    marginal = Kset * Ps[..., None]
    marginal = marginal.sum(-2)
    return marginal


def stable_logist(x):
    # This is much more numerically stable than simple
    # return 1 / (1 + torch.exp(-x))
    return torch.clamp(torch.sigmoid(x), min=EPS, max=1-EPS)


def stable_entropy(log_f):
    """ Computes entropy from unnormalized log-probability (energy)
        :param log_f    : [N, K]    K energy values (batch of N distributions)
    """
    log_f = log_f - log_f.max(dim=-1, keepdim=True)[0]
    Z = torch.exp(log_f).sum(-1)
    H = torch.log(Z) - 1/Z * (torch.exp(log_f) * log_f).sum(-1) 
    return H


def stable_cross_entropy(p, logit_q):
    """ Computes cross-entropy between p(.) and logistic(logit_q(.))
        :param  p           : tensor of p-probabilities
        :param  logit_q     : tensor of logits of q-probabilities
    """
    # Simple reference implementation:
    q_s = stable_logist(logit_q)
    crossH_pq = - (p * torch.log(q_s) + (1-p) * torch.log(1-q_s))
    
    #log_q = torch.where(logit_q > 0, 
    #                    -torch.log(torch.exp(-logit_q) + 1),
    #                    logit_q - torch.log(torch.exp(logit_q) + 1))
    #log_not_q = torch.where(logit_q < 0, 
    #                    -torch.log(torch.exp(logit_q) + 1),
    #                    -logit_q - torch.log(torch.exp(-logit_q) + 1))
    #crossH_pq = - (p * log_q + (1-p) * log_not_q)
    return crossH_pq



class SamplerModule(Module):

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, X, Kset, log_f, marginal_p=None, indexes=None):
        raise NotImplementedError()

    def train_batch(self, indexes, X, Kset, log_f):
        # Can be skipped
        pass

    def train_dataset(self, dataloader, datatransformer, Kset, log_f):
        raise NotImplementedError()

    def sample_q(self, X, indexes=None, nsamples=1000):
        raise NotImplementedError()
    

class SimpleTrainableSampler(SamplerModule):
    def _train_epoch(self, dataloader, datatransformer, Kset, log_f, optimizer, on_finish=None):
        self.train()
        model_device = next(self.parameters()).device
        losses = []
        for batch_idx, (indexes, X) in enumerate(tqdm(dataloader)):
            indexes, X = indexes.to(model_device), X.to(model_device)
            X = datatransformer(X)
            optimizer.zero_grad()
            res = self(X, Kset[indexes], log_f[indexes], indexes=indexes)
            loss = res["objective"]
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            #print("Batch {:4d} | loss: {:9.4f}".format(batch_idx, loss))
        
        if on_finish is not None:
            on_finish(X, Kset, log_f, np.array(losses).mean(), res)

        return losses


    def train_dataset(self, dataloader, datatransformer, Kset, log_f):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        epoch_loss = []
        
        #with torch.autograd.detect_anomaly():
        for epoch in range(10):
            losses = self._train_epoch(dataloader, datatransformer, Kset, log_f, optimizer, on_finish=None)
            epoch_loss.append(np.array(losses).mean())
            print("Optimizing {} | Epoch: {:4d} | <loss>: {:9.4f}".format(self.__class__.__name__, epoch+1, epoch_loss[-1]))
            if epoch > 1 and epoch_loss[-1] > epoch_loss[-2]:
                break


class SequenceBernoulli(SimpleTrainableSampler):
    """ Multivariate Bernoulli based on full chain conditional factorization
    """

    def __init__(self, variationalparams=None) -> None:
        super().__init__()
        self.variationalparams = variationalparams


    def forward(self, X, Kset, log_f, marginal_p=None, indexes=None):
        """ Returns relaxed cross-entropy H(p_K | q(X))
            :param X            : [N, D]    N data points
            :param Kset         : [N, K, H] truncated binary posterior sets
            :param log_f        : [N, K]    log-joint of (X, Kset) 
            :param marginal_p   : [N, H]    marginal probability of bits
            :param indexes      : [N]       data points indexes
        """
        N, K, H = Kset.shape
        device = X.device

        # Mean params M_params is [N, H]
        # Conditional params C_params is [N, (H*H-H)/2]
        M_params, C_params = self.variationalparams(X, indexes)

        C = torch.zeros(size=(N, H, H), device=device)
        tril_ind = torch.tril_indices(H, H, -1, device=device)
        C[:, tril_ind[0], tril_ind[1]] = C_params

        #print(Kset.shape, C.shape)
        cond_params = (Kset.unsqueeze(-2) * C.unsqueeze(-3)).sum(-1)  # [N, K, H]
        #print(M_params.shape, cond_params.shape)
        S_prob_params = M_params.unsqueeze(-2) + cond_params  # [N, K, H]
        
        q_s = stable_logist(S_prob_params)  # [N, K, H]
        p_s = compute_probabilities(log_f)  # [N, K]
        crossH_pq = - (
            p_s.unsqueeze(-1) * (
                    Kset * torch.log(q_s) + 
                    (1-Kset) * torch.log(1-q_s)
                )
            ).sum(dim=(-1, -2))
        
        H_p = stable_entropy(log_f)
        KL_pq = crossH_pq - H_p

        res = {}
        res["objective"] = KL_pq.mean()  # average number of mismatched bits per sample
        return res


    def sample_q(self, X, indexes=None, nsamples=1000):
        """ Sample from the fitted density
            :param X            : [N, D]    N data points
            :param indexes      : [N]       data points indexes
            :param nsamples     : int       number of samples
            :returns samples    : [nsamples, N, H]
        """
        self.eval()

        with torch.no_grad():
            M_params, C_params = self.variationalparams(X, indexes)
            N, H = M_params.shape
            device = X.device

            C = torch.zeros(size=(N, H, H), device=device)
            tril_ind = torch.tril_indices(H, H, -1, device=device)
            C[:, tril_ind[0], tril_ind[1]] = C_params

            res = []
            L = torch.zeros(size=(N, H, H), device=device)  # intermediate sampling steps memory 
            for i in range(H + nsamples):
                cond_params = (L*C).sum(-1) 
                S_prob_params = M_params + cond_params
                S_new = torch.bernoulli(stable_logist(S_prob_params))
                L = L.diagonal_scatter(S_new, dim1=-1, dim2=-2)
                res.append(L[:, -1, :].clone().detach())
                L[:, 1:, :] = L.clone()[:, :-1, :]

            return torch.stack(res[H:], dim=0)

    
class MarginalBernoulli(SimpleTrainableSampler):
    def __init__(self, variationalparams=None) -> None:
        super().__init__()
        self.variationalparams = variationalparams


    def forward(self, X, Kset, log_f, marginal_p=None, indexes=None):
        """ Returns relaxed cross-entropy H(p_K | q(X))
            :param X            : [N, D]    N data points
            :param Kset         : [N, K, H] truncated binary posterior sets
            :param log_f        : [N, K]    log-joint of (X, Kset) 
            :param marginal_p   : [N, H]    marginal probability of bits
            :param indexes      : [N]       data points indexes
        """
        # Mean params M_params is [N, H]
        M_params = self.variationalparams(X, indexes)

        marginal_p = compute_marginal(Kset, log_f)
        marginal_p = torch.clamp(marginal_p, min=0.01, max=0.99)
        
        crossH_pq = stable_cross_entropy(p=marginal_p, logit_q=M_params).sum(-1)
        if torch.any(torch.isnan(crossH_pq)):
            breakpoint()
        H_marginal_p = - (marginal_p * torch.log(marginal_p) + (1-marginal_p)*torch.log(1-marginal_p)).sum(-1)
        if torch.any(torch.isnan(H_marginal_p)):
            breakpoint()
        KL_pq = crossH_pq - H_marginal_p

        res = {}
        res["objective"] = KL_pq.mean()  
        return res


    def sample_q(self, X, indexes=None, nsamples=1000):
        """ Sample from the fitted density
            :param X            : [N, D]    N data points
            :param indexes      : [N]       data points indexes
            :param nsamples     : int       number of samples
            :returns samples    : [nsamples, N, H]
        """
        self.eval()

        with torch.no_grad():
            M_params = self.variationalparams(X, indexes)
            res = torch.bernoulli(stable_logist(M_params).repeat(nsamples, 1, 1))
            return res


