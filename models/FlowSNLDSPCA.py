import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical, Normal

from models.glow import Glow, BatchNorm, PCABlock
from models.SNLDS import NeuralSNLDS

class FlowSNLDS(NeuralSNLDS):
    ## Class could be combined with Abstract class MSM for code efficiency
    ## The model allows different settings where annealing=True implements schedule from
    ## Dong et al. (2020) https://proceedings.mlr.press/v119/dong20e.html;
    ## and inference=='alpha' implements Ansari et al. (2023) loss function
    ## https://arxiv.org/abs/2110.13878
    ## We recommend the setting with annealing=False and inference='alpha' and recurent encoder
    ## which is the best that worked for estimation.
    def __init__(self, obs_dim, latent_dim, hidden_dim, num_states, encoder_type='factored', device='cpu', annealing=False, inference='alpha', n_bits=8):
        super(FlowSNLDS, self).__init__(obs_dim, latent_dim, hidden_dim, num_states, device, annealing)

        self.temperature = 1
        self.annealing = annealing
        self.inference = inference
        self.n_bits = torch.tensor(n_bits)
        if annealing:
            self.temperature = 1e6
        self.encoder_type = encoder_type


        if encoder_type == 'factored':
            self.encoder = Glow(n_channels=obs_dim, n_steps=32, n_flow_blocks=1, dequantize=False, input_type=encoder_type).to(device).float()
        elif encoder_type == 'video':
            self.encoder = Glow(n_channels=3, n_steps=16, n_flow_blocks=3, dequantize=True, input_type='image', n_bits=self.n_bits).to(device).float()
            self.norm = BatchNorm()
            self.pca = PCABlock()

        self.decoder = self.encoder.inverse
    
    def _encode_obs(self, x, train=True):
        """
        Returns encoded observations and jacobian
        
        Parameters
        ----------
        x : Tensor(B,T,D) or Tensor(B,T,C,W,H)
            Input observation. Either a series of observed states (B,T,D) or
            and image of shape (B,T,C,W,H)
        train: (bool) decides if the input data should be dequantized in the image case
        Returns
        -------
        z: Tensor(x.shape)
            latent variable
        log_det_jacobian : Tensor(B,T,latent_dim)
            volume changed

        
        """
        (B, T, *_) = x.size()
        if self.encoder_type == "factored":
            (B, T, D) = x.size()
            (z_list, log_det_jacobian) = self.encoder(x.reshape(B*T,-1))
            z, log_det_jacobian = self.encoder.list_to_z(z_list).reshape(B, T, D), log_det_jacobian.reshape(B,T)
        elif self.encoder_type == "video":
            (B, T, C, H, W)  = x.size()
            (z_list, log_det_jacobian) = self.encoder(x.reshape(B*T, C, H, W), train)
            z, log_det_jacobian = self.encoder.list_to_z(z_list).reshape(B, T, C, H, W), log_det_jacobian.reshape(B,T)
        
        
        return z, log_det_jacobian


    def _decode(self, z, V=None, mean=None, std=None):
        '''
        z: Tensor(B,T,D) or Tensor(B,T,C,W,H)
            Input observation. Either a series of observed states (B,T,D) or
            and image of shape (B,T,C,W,H)
        V: PCA rotation matrix
        mean: Batchnorm mean for Neural PCA block
        std: Batcnorm mean for Neural PCA Block

        x: Tensor(z.shape); Tensor(B,T,D) or Tensor(B,T,C,W,H)
            decoded output
        '''
        shape = z.shape
        if self.encoder_type == 'video':
            B, T, C, H, W = shape
            z = self.pca.inverse(z, V)
            z.reshape(B*T, C, H, W )
            z = self.norm.inverse(z, mean, std)            
        else:
            B, T, D = z.shape
            z.reshape(B*T, D)
        z = self.encoder.z_to_list(z)
        return self.decoder(z).reshape(shape)

    
    
    def _compute_loss(self, log_det_jacobian, log_p_z, gamma, paired_marginals, log_evidence, z_image=None):

        # max: ELBO = log p(x_t|z_t) - (log q(z) + log q(s) - log p(z_t | s_t) - log p(s_t| s_t-1))
        # min: -ELBO =  - log p(x_t|z_t) + log q(z) + log q(s) - log p(z_t | s_t) - log p(s_t| s_t-1)
        # Fixed variance
        # Reconstruction Loss p(x_t | z_t)
        #decoder_x_1 = MultivariateNormal(x_hat, covariance_matrix=torch.eye(D).to(self.device)*self.var)
        #p_x_1 = (decoder_x_1.log_prob(x)).sum(-1)
        #recon_loss = (p_x_1).sum()/B
        
        (B, T, *_) = log_p_z.size()
        
        
        
        # log_Z == log_p_z
        msm_loss = 0
        if self.inference=='alpha':
            msm_loss = log_p_z.sum()/B
        else:
            # pi
            msm_loss = (gamma[:,0,:]*torch.log((self.pi/self.temperature).softmax(-1)[None,:])).sum()/B
            # Q
            Q = (self.Q[None,None,:,:].expand(B,T,-1,-1)/self.temperature).softmax(-1)
            msm_loss += (paired_marginals*torch.log(Q[:,1:,:,:])).sum()/B
            # p(z_t|z_t-1)
            msm_loss += (gamma[:,:]*log_evidence[:,:]).sum()/B
            
        log_likelihood =  log_det_jacobian + log_p_z
        jacobian = log_det_jacobian.sum()/B
        
        if z_image is not None:
            n_pixels = z_image.shape[-1] + self.latent_dim
            c = n_pixels * torch.pow(2, self.n_bits)
            p_z_image = Normal(0,1).log_prob(z_image.view(B*T,-1)).sum(-1).reshape(B,T)
            #p_z_image = (-0.5 * (z_image ** 2 + torch.log(2 * pi ))).view(z_image.shape[0],-1).sum(-1)
            # print(p_z_image.shape)
            log_likelihood += c + p_z_image
            
            
            
        
        loss = -log_likelihood.sum()/(B)
        
        if self.encoder_type == 'video': # Convert to bits
            loss = (loss / (torch.log(torch.tensor(2.))  * n_pixels))
            p_z_image = (p_z_image / (torch.log(torch.tensor(2.))  * n_pixels)).mean()
            jacobian = (jacobian / (torch.log(torch.tensor(2.))  * n_pixels))        
         
              
        losses = {
            'loss': loss,
            'jacobian': jacobian,
            'temp_p_z': log_p_z.mean(),
            'msm_loss': msm_loss,
            'img_p_z': p_z_image if z_image is not None else 0
        }
        return losses

    def forward(self, x, train=True):
        # input is [B, T, D]
        
        z_sample, log_det_jacobian = self._encode_obs(x, train=train)
        z_image = None
        if self.encoder_type == 'video':
            (B, T, C, H, W) = x.shape
            z, norm_jacobian, mean, std = self.norm(z_sample)
            log_det_jacobian += norm_jacobian

            z_sample, V = self.pca(z_sample)
            z = z_sample.reshape(B, T, -1)
            z_temporal, z_image = z[:,:,:self.latent_dim], z[:,:,self.latent_dim:]
        else:
            (B, T, _) = x.shape
            z_temporal = z_sample.reshape(B, T, -1)
        gamma = None
        log_evidence = self._compute_local_evidence(z_temporal)
        if self.inference=='alpha':
            if self.annealing:
                log_alpha, log_Z = self._alpha(log_evidence)
                log_beta = self._beta(log_evidence, log_Z)
                gamma = (log_alpha + log_beta).exp()
            else:
                log_Z = self._alpha(log_evidence)[0].sum(-1) # marginalise out the states
                gamma = None
            paired_marginals = None
        else:
            with torch.no_grad():
                gamma, paired_marginals = self._compute_posteriors(log_evidence)
                log_Z = None
        
        if self.encoder_type == 'video':
            x_hat = self._decode(z_sample)
        else:
            x_hat = self._decode(z_sample)
        
        
        
        losses = self._compute_loss( log_det_jacobian, log_Z,  gamma, paired_marginals, log_evidence, z_image)
        return x_hat , z_sample, gamma, losses, V, mean, std
    

    def compute_training_statistics(self, dataloader, ):
        """
        x: pytorch Dataset
        Iterates over the entire dataset to compute mean mu, variance and 
        """
        self.requires_grad_(False)

        mean = torch.zeros()
        std = torch.zeros()
        V_tilde = torch.zeros()

        for i, (sample,) in enumerate(dataloader, 1):
            B, T, C, H, W = sample.size()



        pass
    
