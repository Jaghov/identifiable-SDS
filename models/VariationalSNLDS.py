import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical, Normal

from models.modules import MLP, CNNFastEncoder, CNNFastDecoder
from models.SNLDS import NeuralSNLDS

class VariationalSNLDS(NeuralSNLDS):
    ## Class could be combined with Abstract class MSM for code efficiency
    ## The model allows different settings where annealing=True implements schedule from
    ## Dong et al. (2020) https://proceedings.mlr.press/v119/dong20e.html;
    ## and inference=='alpha' implements Ansari et al. (2023) loss function
    ## https://arxiv.org/abs/2110.13878
    ## We recommend the setting with annealing=False and inference='alpha' and recurent encoder
    ## which is the best that worked for estimation.
    def __init__(self, obs_dim, latent_dim, hidden_dim, num_states, beta=1, encoder_type='recurent', device='cpu', annealing=False, inference='alpha'):
        super(VariationalSNLDS, self).__init__(obs_dim, latent_dim, hidden_dim, num_states, device, annealing)
        self.beta = beta
        self.var = torch.tensor(5e-4).to(device)
        self.scaling = 0
        self.temperature = 1
        self.annealing = annealing
        self.inference = inference
        if annealing:
            self.scaling = 1e6
            self.temperature = 1e6
        self.encoder_type = encoder_type
        n_feat = 64
        n_layers = 2
        ## Neural net params
        # Transitions p(z_t|z_t-1,s_t)
        self.transitions = nn.ModuleList([MLP(latent_dim, latent_dim, hidden_dim, 'softplus') for _ in range(self.num_states)]).to(device).float()
        # Encoder q(z|x)
        if self.encoder_type=='factored':
            #self.encoder = nn.Linear(obs_dim, 2*latent_dim).to(device).float()
            self.encoder = MLP(obs_dim, 2*latent_dim, hidden_dim, 'leakyrelu').to(device).float()
        elif self.encoder_type=='video':
            self.img_embedding = CNNFastEncoder(3, hidden_dim, n_feat, n_layers=n_layers).to(device).float()
            self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True).to(device).float()
            self.encoder_causal = nn.LSTM(hidden_dim*2, hidden_dim, num_layers=2, batch_first=True, bidirectional=False).to(device).float() # ASK Why causal after bidirectional?
            self.encoder_mean_var = nn.Linear(hidden_dim, 2*latent_dim).to(device).float()
        else:
            self.encoder = nn.LSTM(obs_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True).to(device).float()
            self.encoder_causal = nn.LSTM(hidden_dim*2, hidden_dim, num_layers=2, batch_first=True, bidirectional=False).to(device).float()
            self.encoder_mean_var = nn.Linear(hidden_dim, 2*latent_dim).to(device).float()
        # Decoder p(x|z)
        if self.encoder_type=='video':
            self.decoder = CNNFastDecoder(latent_dim, 3, n_feat, n_layers=n_layers).to(device).float()
        else:
            self.decoder = MLP(latent_dim, obs_dim, hidden_dim, 'leakyrelu').to(device).float()
    
    def _encode_obs(self, x):
        """Encodes a sequence of observations as a pair of latent
        variables zmu and z_log_var as well as it's encoded representation
        as z which is reparemtrised as zmu + eps*z_std
        
        Parameters
        ----------
        x : Tensor(B,T,D) or Tensor(B,T,C,W,H)
            Input observation. Either a series of observed states (B,T,D) or
            and image of shape (B,T,C,W,H)
        Returns
        -------
        sample : Tensor(B,T,latent_dim)
            reparemetarised sampled encoding of the input
        z_log_var : Tensor(B,T,latent_dim)
            mean for encoding for each latent dim
        z_mu : Tensor(B,T,latent_dim)
            mean for encoding for each latent dim
        
        """
        if self.encoder_type=='factored':
            (B, T, D) = x.shape
            (z_mu, z_log_var) = self.encoder(x.reshape(B*T,-1)).split(self.latent_dim, dim=-1)
        elif self.encoder_type=='video':
            (B, T, C, W, H) = x.shape
            x = self.img_embedding(x.reshape(B*T,C,W,H)).reshape(B,T,-1) # Stacks the batch and time dimensions into one, performs the convolution, then unstacks them.
            output, _ = self.encoder(x)
            # output contains h^x_{1:T}
            output, _ = self.encoder_causal(output)
            (z_mu, z_log_var) = self.encoder_mean_var(output).split(self.latent_dim, dim=-1)
        else:
            output, _ = self.encoder(x)
            # output contains h^x_{1:T}
            output, _ = self.encoder_causal(output)
            (z_mu, z_log_var) = self.encoder_mean_var(output).split(self.latent_dim, dim=-1)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = (z_log_var*0.5).exp()
        sample = z_mu + z_std*eps
        return sample, z_mu, z_log_var

    
    def _decode(self, z):
        return self.decoder(z)

    def _compute_elbo(self, x, x_hat, z_mu, z_log_var, z_sample, log_Z=None, gamma=None, paired_marginals=None, log_evidence=None):

        (B, T, D) = x.size()
        # max: ELBO = log p(x_t|z_t) - (log q(z) + log q(s) - log p(z_t | s_t) - log p(s_t| s_t-1))
        # min: -ELBO =  - log p(x_t|z_t) + log q(z) + log q(s) - log p(z_t | s_t) - log p(s_t| s_t-1)
        # Fixed variance
        # Reconstruction Loss p(x_t | z_t)
        #decoder_x_1 = MultivariateNormal(x_hat, covariance_matrix=torch.eye(D).to(self.device)*self.var)
        #p_x_1 = (decoder_x_1.log_prob(x)).sum(-1)
        #recon_loss = (p_x_1).sum()/B


        decoder_x_2 = Normal(x_hat, torch.sqrt(self.var))
        p_x_2 = (decoder_x_2.log_prob(x)).sum(-1)
        recon_loss = (p_x_2).sum()/(B)
        #print(recon_loss)
        #print(recon_loss_2)
        ## KL terms
        q_z = MultivariateNormal(z_mu, torch.diag_embed(torch.exp(z_log_var)))
        entropy_q = -(q_z.log_prob(z_sample)).sum()/B
        if self.beta==0:
            msm_loss = 0
            CE_term = 0
        else:
            if self.inference=='alpha':
                msm_loss = log_Z.sum()/B
            else:
                # pi
                msm_loss = (gamma[:,0,:]*torch.log((self.pi/self.temperature).softmax(-1)[None,:])).sum()/B
                # Q
                Q = (self.Q[None,None,:,:].expand(B,T,-1,-1)/self.temperature).softmax(-1)
                msm_loss += (paired_marginals*torch.log(Q[:,1:,:,:])).sum()/B
                # p(z_t|z_t-1)
                msm_loss += (gamma[:,:]*log_evidence[:,:]).sum()/B
            CE_term = 0
            if self.annealing:
                CE_term = self.scaling*self.kl_categorical_uniform(gamma)# +  self.scaling*self.kl_categorical_uniform((self.pi).softmax(-1))
        elbo = recon_loss + entropy_q + self.beta*msm_loss
        losses = {
            'kld': entropy_q,
            'elbo': elbo,
            'loss': -elbo + CE_term,
            'recon_loss': recon_loss,
            'msm_loss': msm_loss,
            'CE': CE_term
        }
        return losses
    
    def kl_categorical_uniform(self, gamma, eps=1e-16):
        """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
        prob = (1/self.num_states)
        kl_div = prob * (torch.log(torch.tensor(prob)) - torch.log(gamma + eps))
        return kl_div.sum() / (gamma.size(0))

    def forward(self, x):
        # input is [B, T, D]
        (B, T, *_) = x.shape
        z_sample, z_mu, z_log_var = self._encode_obs(x)
        z_sample = z_sample.reshape(B,T,-1)
        z_mu = z_mu.reshape(B,T,-1)
        z_log_var = z_log_var.reshape(B,T,-1)
        if self.beta==0:
            log_evidence, gamma, paired_marginals, log_Z = None, None, None, None
        else:
            log_evidence = self._compute_local_evidence(z_sample)
            if self.inference=='alpha':
                if self.annealing:
                    log_alpha, log_Z = self._alpha(log_evidence)
                    log_beta = self._beta(log_evidence, log_Z)
                    gamma = (log_alpha + log_beta).exp()
                else:
                    log_Z = self._alpha(log_evidence)[1].sum(-1)
                    gamma = None
                paired_marginals = None
            else:
                with torch.no_grad():
                    gamma, paired_marginals = self._compute_posteriors(log_evidence)
                    log_Z = None
        x_hat = self._decode(z_sample.reshape(B*T,-1)).reshape(B,T,-1)
        # ELBO
        losses = self._compute_elbo(x.reshape(B,T,-1), x_hat, z_mu, z_log_var, z_sample, log_Z, gamma, paired_marginals, log_evidence)
        return x_hat, z_sample, gamma, losses
    