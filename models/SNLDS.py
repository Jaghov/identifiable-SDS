import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical, Normal
from abc import ABC, abstractmethod

from models.modules import MLP, CNNFastEncoder, CNNFastDecoder

class NeuralSNLDS(nn.Module):
    ## Class could be combined with Abstract class MSM for code efficiency
    ## The model allows different settings where annealing=True implements schedule from
    ## Dong et al. (2020) https://proceedings.mlr.press/v119/dong20e.html;
    ## and inference=='alpha' implements Ansari et al. (2023) loss function
    ## https://arxiv.org/abs/2110.13878
    ## We recommend the setting with annealing=False and inference='alpha' and recurent encoder
    ## which is the best that worked for estimation.
    def __init__(self, obs_dim, latent_dim, hidden_dim, num_states, device='cpu', annealing=False):
        super(NeuralSNLDS, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_states = num_states
        self.device = device
        self.temperature = 1
        if annealing:
            self.temperature = 1e6
        ## Neural net params
        # Transitions p(z_t|z_t-1,s_t)
        self.transitions = nn.ModuleList([MLP(latent_dim, latent_dim, hidden_dim, 'softplus') for _ in range(self.num_states)]).to(device).float()
        #self.decoder = nn.Linear(latent_dim, obs_dim).to(device)
        ## MSM params
        # logits of p(s_t|s_t-1)
        self.Q = nn.Parameter(torch.zeros(self.num_states, self.num_states).to(device).float())
        # logits of p(s_1)
        self.pi = nn.Parameter(torch.zeros(num_states).to(device).float())
        #self.pi = torch.zeros(num_states).to(device)
        # Init mean and covariances: Phi params
        self.init_mean = nn.Parameter(torch.randn(self.num_states, self.latent_dim).to(device).float())
        self.init_cov = nn.Parameter(((torch.rand(self.num_states,1,1)*torch.eye(self.latent_dim)[None,:,:])*5).to(device).float())
        self.covs = nn.Parameter((torch.eye(self.latent_dim)[None,:,:]).repeat(self.num_states,1,1).to(device).float())

    @abstractmethod
    def _encode_obs(self, x):
        """Encodes a sequence of observations as a pair of latent
        representation
        
        Parameters
        ----------
        x : Tensor(B,T,D) or Tensor(B,T,C,W,H)
            Input observation. Either a series of observed states (B,T,D) or
            and image of shape (B,T,C,W,H)
        """
        raise NotImplementedError

    def _compute_local_evidence(self, z):
        """Computes the p(z_{1:T} | s_{1:T} ) for all states shape B, T, num_states
        =log p(z_1 | s_1) + sum log p(z_t | z_t-1, s_t)
        """
        T = z.size(1)
        init_distrib_ = torch.distributions.MultivariateNormal(self.init_mean, torch.matmul(self.init_cov,self.init_cov.transpose(1,2)) + 1e-6*torch.eye(self.latent_dim)[None,:,:].to(self.device))
        log_local_evidence_1 = init_distrib_.log_prob(z[:,0:1,None,:].repeat(1,1,self.num_states,1))
        if T==1:
            return log_local_evidence_1
        means_ = torch.cat([self.transitions[i](z[:,:-1, None,:]) for i in range(self.num_states)], dim=2)
        covs = torch.matmul(self.covs,self.covs.transpose(1,2)) + 1e-6*torch.eye(self.latent_dim)[None,:,:].to(self.device)
        distribs = [torch.distributions.MultivariateNormal(means_[:,:,i,:], covs[i,:,:]) for i in range(self.num_states)]
        log_local_evidence_T = torch.cat([distribs[i].log_prob(z[:,1:,:])[:,:,None] for i in range(self.num_states)], dim=2)
        return torch.cat([log_local_evidence_1, log_local_evidence_T], dim=1)

    def _alpha(self, local_evidence):
        """ p( s_t |z_1, ..., z_t) using dynamic programming
        """
        N, T, _ = local_evidence.shape
        log_Z = torch.zeros((N,T)).to(self.device)
        log_alpha = torch.zeros((N, T, self.num_states)).to(self.device)
        log_prob = local_evidence[:,0,:] + torch.log((self.pi/self.temperature).softmax(-1))
        log_Z[:,0] = torch.logsumexp(log_prob, dim=-1)
        log_alpha[:,0,:] = log_prob - log_Z[:,0,None]
        Q = (self.Q[None,None,:,:].expand(N,T,-1,-1)/self.temperature).softmax(-1).transpose(2,3).log()
        for t in range(1, T):
            #log_prob = local_evidence[:,t,:] + torch.log(torch.matmul((Q.transpose(2,3))[:,t,:,:],alpha[:,t-1,:,None]))[:,:,0]
            log_prob = torch.logsumexp(local_evidence[:,t,:, None] + Q[:,t,:,:] + log_alpha[:,t-1,None,:], dim=-1) 
            
            log_Z[:,t] = torch.logsumexp(log_prob, dim=-1)
            log_alpha[:,t,:] = log_prob - log_Z[:,t,None] # log_p_z normalises the log_prob to get alpha
        return log_alpha, log_Z

    def _beta(self, local_evidence, log_Z):
        """ p( s_t |z_t+1, ..., z_T) using dynamic programming
        """
        N, T, _ = local_evidence.shape
        log_beta = torch.zeros((N, T, self.num_states)).to(self.device)
        Q = (self.Q[None,None,:,:].expand(N,T,-1,-1)/self.temperature).softmax(-1).log()
        for t in reversed(range(1, T)):
            #beta_ = torch.matmul(Q[:,t,:,:], (torch.exp(local_evidence[:,t,:])*beta[:,t,:])[:,:,None])[:,:,0]
            beta_ = torch.logsumexp(Q[:,t,:,:] + local_evidence[:,t,None,:] + log_beta[:,t,None,:], axis=-1)
            log_beta[:,t-1,:] = beta_ - log_Z[:,t,None]
        return log_beta

    def _compute_posteriors(self, log_evidence):
        """Computes the p(s_n | z_{1:T} ) and p(s_n-1, s_n | z_{1:T}) for all states
        """
        log_alpha, log_Z = self._alpha(log_evidence)
        log_beta = self._beta(log_evidence, log_Z)
        log_gamma = log_alpha + log_beta # log(p_z)
        B, T, _ = log_evidence.shape
        #alpha_beta_evidence = torch.matmul(alpha[:,:T-1,:,None], (beta*torch.exp(log_evidence))[:,1:,None,:])
        log_alpha_beta_evidence = log_alpha[:,:T-1,:,None] + log_beta[:,1:,None,:] + log_evidence[:,1:,None,:]
        Q = (self.Q[None,None,:,:].expand(B,T,-1,-1)/self.temperature).softmax(-1).log()
        #paired_marginals = Q[:,1:,:,:]*(alpha_beta_evidence/torch.exp(log_Z[:,1:,None,None])).float()
        log_paired_marginals = Q[:,1:,:,:] + log_alpha_beta_evidence - log_Z[:,1:,None,None]
        
        return log_gamma.exp().detach(), log_paired_marginals.exp().detach()
    
    @abstractmethod
    def _decode(self, z):
        raise NotImplementedError

    @abstractmethod
    def _compute_loss(self, x, x_hat, z_mu, z_log_var, z_sample, log_Z=None, gamma=None, paired_marginals=None, log_evidence=None):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def predict_sequence(self, input, seq_len=None):
        (B, T, *_) = input.size()
        if seq_len is None:
            seq_len = T
        z_sample, _, _ = self._encode_obs(input)
        z_sample = z_sample.reshape(B,T,-1)
        log_evidence = self._compute_local_evidence(z_sample)
        gamma, _ = self._compute_posteriors(log_evidence)
        #last_discrete = Categorical(gamma[:,-1,:]).sample()
        last_discrete = gamma[:,-1,:].argmax(-1)
        last_continous = z_sample[:,-1,:]
        latent_seq = torch.zeros(B,seq_len,self.latent_dim).to(input.device)
        Q = self.Q
        for t in range(seq_len):
            # next discrete state
            last_discrete_distrib = torch.cat([Q[last_discrete[b].long(),:][None,:] for b in range(B)], dim=0)
            last_discrete = Categorical(logits=last_discrete_distrib).sample()
            # next observation mean
            last_continous = torch.cat([self.transitions[last_discrete[b]](last_continous[b,None,:]) for b in range(B)], dim=0)
            latent_seq[:,t,:] = last_continous
        # decode
        return self._decode(latent_seq.reshape(B*seq_len,-1)).reshape(B,seq_len,-1)