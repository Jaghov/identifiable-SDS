import unittest
import torch
from models.FlowSNLDS import *


seed = torch.manual_seed(0)


class TestSNLDS(unittest.TestCase):
    def setUp(self):
        # self.seed = torch.cuda.manual_seed(0).to('cuda')
        self.snlds = FlowSNLDS(8, 2, hidden_dim=8, num_states=3, encoder_type='video', device='cuda', annealing=True, inference='alpha', n_bits=8)

    def test_is_invertible(self):
        x = torch.rand((1, 10, 3,8,8), device='cuda')

        with torch.no_grad():
            x_hat , z_sample, *_ = self.snlds(x)

    
        self.assertTrue(torch.allclose(x, x_hat, atol=1e-1))

        

if __name__ == "__main__":
    unittest.main()