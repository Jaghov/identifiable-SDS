from pathlib import Path
import tqdm

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from sklearn.datasets import make_moons
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.glow import Glow

import gc


gc.collect()
torch.cuda.empty_cache()

def show(img):
    npimg = img#.cpu().numpy()
    plt.imshow(npimg)


# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')


n_train = 3000
train_set, _ = make_moons(n_samples=n_train, noise=0.1)
train_set = torch.tensor(train_set, dtype=torch.float32)

# Necessary Hyperparameters
num_epochs = 200
learning_rate = 5e-4
batch_size = 16
n_bits = 5
n_steps=2


trainloader = DataLoader(train_set, batch_size=batch_size, num_workers=2)



# Plot and save the training set
plt.figure(figsize=(10, 8))
plt.scatter(train_set[:, 0], train_set[:, 1], s=10)
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.xlim(-2, 3)
plt.ylim(-1.5, 2)
plt.savefig('temp_results/train_set.png')
plt.close()




def mse(x, x_hat):
  '''
  x, x_hat (tensor)
  '''
  return ((x_hat-x)**2).mean()






model = Glow(n_channels=2, n_steps=n_steps, n_flow_blocks=1, dequantize=False, input_type='factored').to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
# print(model)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def rand_sample(model, tag, z):
    z = z.to(device)
    with torch.no_grad():
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        samples = model.inverse([z]).to('cpu').numpy()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        plt.figure(figsize=(10, 8))
        plt.scatter(samples[:, 0], samples[:, 1], s=10)
        plt.title('Random samples')
        plt.xlim(-2, 3)
        plt.ylim(-1.5, 2)
        plt.savefig(f'temp_results/sample-{tag}.png')
        plt.close()



pi = torch.tensor(np.pi)
torch.cuda.empty_cache()
def loss_function_Glow(z_list, log_det_jacobian):
  # fit z to normal distribution


  log_p_z = 0
  for z_i in z_list:
    log_p_z += (-0.5 * (z_i ** 2 + torch.log(2 * pi ))).view(z_i.shape[0],-1).sum(-1)


  log_likelihood = log_p_z + log_det_jacobian 
  loss = -log_likelihood


  return (
    loss.mean(),
    log_p_z.mean(), 
    log_det_jacobian.mean(),
  )



model.train()
n_samples = 3000
sample = torch.randn(n_samples, 2)
for epoch in range(num_epochs):
    total_loss = 0 # <- You may wish to add logging info here
    err = 0
    #max_grad = 0

    with tqdm.tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, data in enumerate(tepoch):
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################
            data = data.to(device) # Need at least one batch/random data with right shape - .view(-1,28*28)
                        # This is required to initialize to model properly below
                        # when we save the computational graph for testing (jit.save)
            with torch.no_grad():
                _,_, = model.forward(data)


            # forward pass
            z_list, log_det_jacobian = model.forward(data)

            log_det_jacobian = log_det_jacobian.mean()

            # compute loss
            loss, prior, jacobian = loss_function_Glow( z_list, log_det_jacobian)


            # backwards
            optimizer.zero_grad()
            loss.backward()


            # update params
            optimizer.step()

            # Logging
            total_loss += loss.item()
            





            # torch.cuda.empty_cache()
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    err = mse(data, model.inverse(z_list)).mean()
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=loss.item()/len(data), log_prior=prior.item(), log_jacobian=jacobian.item(), recon_err = err.item(), lr=optimizer.param_groups[0]['lr']  )#, avg_weights=avg_grad.item(), max_grad=max_grad.item() )

    if epoch % 1 == 0:
        rand_sample(model, epoch, sample)

    # save the model
    if epoch == num_epochs - 1:
        with torch.no_grad():
            torch.jit.save(torch.jit.trace(model, (data), check_trace=False),
                'weights/Glow_model.pth')

