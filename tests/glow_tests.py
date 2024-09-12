import unittest
import torch
from models.glow import *



seed = torch.manual_seed(0)

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.seed = torch.manual_seed(0)
        self.preprocess = Preprocess()
    
    def test_preprocess_jitters_data(self):
        input = torch.rand( (6,6), generator=self.seed)
        output = None
        
        with torch.no_grad():
            output= self.preprocess(input)
        
        self.assertFalse(torch.eq(input, output).all())
    
    def test_all_outputs_less_than_05(self):
        input = torch.ones((6,6))
        
        output = None
        with torch.no_grad():
            output = self.preprocess(input)
        
        self.assertTrue((output < .5).all())
        
    def test_all_outputs_greater_than_05(self):
        input = torch.zeros((6,6))
        
        output = None
        with torch.no_grad():
            output= self.preprocess(input)
        
        self.assertTrue((output > -0.5).all())

    def test_no_jitter_on_validation(self):
        input = torch.rand((6,6), generator=self.seed)
        output = None
        
        with torch.no_grad():
            output = self.preprocess(input, train=False)
        
        self.assertTrue(torch.allclose(input, self.preprocess.inverse(output)))
    
    def test_does_not_jitter_continuous_data(self):
        input = torch.rand( (6,6), generator=self.seed)
        output = None
        self.preprocess.dequantize=False
        
        with torch.no_grad():
            output = self.preprocess(input)
        
        self.assertTrue(torch.allclose(input, self.preprocess.inverse(output)))
        
    def test_jitters_2D_data(self):
        input = torch.rand((2,1), generator=self.seed)
        
        output = None
        self.preprocess.dequantize=True
        
        with torch.no_grad():
            output = self.preprocess(input)
        
        self.assertFalse(torch.eq(input, output).all())
        

class TestActnorm(unittest.TestCase):
    def setUp(self):
        self.seed = torch.manual_seed(0)
        self.actnorm = Actnorm()
    
    def test_first_input_has_zero_mean_and_unit_variance_per_channel(self):
        input = torch.tensor([[[-2., -0.], [2., 4.]],
                          [[0., 2.], [4., 6.]],
                          [[2., 3.], [4., 7.]]]).unsqueeze(0)
        B,C,W,_ = input.shape
        expected_mean = torch.zeros(B,C,1)
        expected_std = torch.ones(B,C,1)

        
        output = None
        
        with torch.no_grad():
            output, _ = self.actnorm(input)
        output_mean = output.mean(dim=(0,2,3), keepdim=True)
        output_std = output.std(dim=(0,2,3), keepdim=True)
        self.assertTrue(torch.eq(output_mean,expected_mean).all() , f"mean of {expected_mean} expected, \n {output_mean} recieved")
        self.assertTrue(torch.eq(output_std,expected_std).all() , f"std of {expected_std} expected, \n {output_std} recieved")
        
    def test_scale_and_bias_params_should_not_change(self):
        input1 = torch.tensor([[[-2., -0.], [2., 4.]],
                          [[0., 2.], [4., 6.]],
                          [[2., 3.], [4., 7.]]]).unsqueeze(0)
        
        input2 = torch.tensor([[[-2., -0.], [3., 4.]],
                          [[0., 1.], [4., 6.]],
                          [[2., 3.], [4., 6.]]]).unsqueeze(0)
        
        B,C,W,_ = input2.shape
        expected_mean = torch.zeros(B,C,1)
        expected_std = torch.ones(B,C,1)

        
        output = None
        
        with torch.no_grad():
            _, _ = self.actnorm(input1)
            output, _ = self.actnorm(input2)
        output_mean = output.mean(dim=(2,3), keepdim=True)
        output_std = output.std(dim=(2,3), keepdim=True)
        self.assertFalse(torch.eq(output_mean,expected_mean).all() , f"mean of {expected_mean} expected, \n {output_mean} recieved")
        self.assertFalse(torch.eq(output_std,expected_std).all() , f"std of {expected_std} expected, \n {output_std} recieved")
        
    def test_should_be_invertible(self):
        input = torch.tensor([[[-2., -0.], [2., 4.]],
                          [[0., 2.], [4., 6.]],
                          [[2., 3.], [4., 7.]]]).unsqueeze(0)
        
        output = None
        
        with torch.no_grad():
            output, _ = self.actnorm(input)
            output = self.actnorm.inverse(output)
        
        self.assertTrue(torch.allclose(output, input) , f"input {input} expected, \n {output} recieved")

    def test_returns_jacobian(self):
        input = torch.randn((2,3,12,12), generator=self.seed)
        print()
        B,_,_,_ = input.shape
        expected_output = input.std(dim=(0,2,3)).abs().sum().unsqueeze(0)
        
        output = None
        
        with torch.no_grad():
            _ , jacobian = self.actnorm(input)
        
        self.assertEqual(jacobian.shape, expected_output.shape, f"Expected jacobian of shape [{B},1] got {jacobian.shape} instead")

    def test_inverts_2D_data(self):
        input = torch.tensor([[-2., -1],
                             [2., 4.],
                             [4., 6.]])
        self.actnorm.input_type= 'factored'
        
        output = None
        
        with torch.no_grad():
            output, _ = self.actnorm(input)
            output = self.actnorm.inverse(output)
        
        self.assertTrue(torch.allclose(output, input) , f"input {input} expected, \n {output} recieved")
        

class TestInv1x1Conv(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.inv = Inv1x1Conv(self.channels)
        
        
    def test_is_invertible(self):
        input = torch.randn(3, self.channels, 4, 4)
        
        z, jacobian = None, None
        
        with torch.no_grad():
            z, jacobian = self.inv(input)
        
        self.assertTrue(torch.allclose(input, self.inv.inverse(z),atol=1e-6), f"input: {input} expected, \n {self.inv.inverse(z)} recieved")
        self.assertFalse(torch.isnan(jacobian).any())
    
    def test_inverts_2D_data(self):
        input = torch.tensor([[-2., -1],
                             [2., 4.],
                             [4., 6.]])
        self.inv = Inv1x1Conv(input.shape[1], input_type='factored' )
        
        output = None
        
        with torch.no_grad():
            output, _ = self.inv(input)
            output = self.inv.inverse(output)
        
        self.assertTrue(torch.allclose(output, input) , f"input {input} expected, \n {output} recieved")

class TestAffineCoupling(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.affine = AffineCoupling(self.channels)
        
        
    def test_is_invertible(self):
        input = torch.randn(4, self.channels, 4, 4)
        
        z, jacobian = None, None

        with torch.no_grad():
            z, jacobian = self.affine(input)
        
        self.assertTrue(torch.allclose(input, self.affine.inverse(z)), f"input: {input} expected, \n {self.affine.inverse(z)} recieved")
        self.assertFalse(torch.isnan(jacobian).any())
        
    def test_2D_is_invertible(self):
        input = torch.randn(4, 2)
        self.affine = AffineCoupling(input.shape[1], input_type='factored')
        
        z, jacobian = None, None

        with torch.no_grad():
            z, jacobian = self.affine(input)
        
        self.assertTrue(torch.allclose(input, self.affine.inverse(z)), f"input: {input} expected, \n {self.affine.inverse(z)} recieved")
        self.assertFalse(torch.isnan(jacobian).any())

class TestFlowStep(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.flow_step = FlowStep(self.channels, input_type='image')
        
        
    def test_is_invertible(self):
        input = torch.randn(4, self.channels, 12, 12)
        
        
        z, jacobian = None, None
        
        with torch.no_grad():
            z, jacobian = self.flow_step(input)
        
        self.assertTrue(torch.allclose(input, self.flow_step.inverse(z),atol=1e-6), f"input: {input} expected, \n {self.flow_step.inverse(z)} recieved")
        self.assertFalse(torch.isnan(jacobian).any())
        
    def test_2D_is_invertible(self):
        input = torch.randn(4, 2)
        self.flow_step = FlowStep(input.shape[1], input_type='factored')
        
        
        z, jacobian = None, None
        
        with torch.no_grad():
            z, jacobian = self.flow_step(input)
        
        self.assertTrue(torch.allclose(input, self.flow_step.inverse(z),atol=1e-6), f"input: {input} expected, \n {self.flow_step.inverse(z)} recieved")
        self.assertFalse(torch.isnan(jacobian).any())

class TestFlowBlock(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.flow_block = FlowBlock(self.channels,3)
        
        
    def test_is_invertible(self):
        input = torch.randn(4, self.channels, 12, 12)
        
        z, jacobian = None, None
        inverse = None
        
        with torch.no_grad():
            z, h, jacobian = self.flow_block(input)
            inverse = self.flow_block.inverse(h, z)
            
        self.assertTrue(torch.allclose(input, inverse, atol=1e-5), f"input: {input} expected, \n {inverse} recieved")
        self.assertFalse(torch.isnan(jacobian).any())
        
    def test_2d_is_invertible(self):
        input = torch.randn(4, 2)
        self.flow_block = FlowBlock(input.shape[1],3, split=False, input_type='factored' )
        
        z, jacobian = None, None
        inverse = None
        
        with torch.no_grad():
            z, _, jacobian = self.flow_block(input)
            inverse = self.flow_block.inverse(z)
            
        self.assertTrue(torch.allclose(input, inverse, atol=1e-5), f"input: {input} expected, \n {inverse} recieved")
        self.assertFalse(torch.isnan(jacobian).any())

class TestGlow(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.n_blocks = 3
        self.glow = Glow(self.channels, n_steps=3, n_flow_blocks=self.n_blocks, n_bits=8)
        
        
    def test_is_invertible(self):
        input = torch.rand( (4, self.channels, 32, 32))
        
        z, jacobian = None, None
        inverse = None
        
        with torch.no_grad():
            z, jacobian = self.glow(input, train=False)
            inverse = self.glow.inverse(z)
        self.assertTrue(torch.allclose(input, inverse, atol=1e-6), f"input: {input} expected, \n {inverse} recieved")
        self.assertFalse(torch.isnan(jacobian).any())
        
    def test_Glow_has_the_right_number_of_layers(self):
        self.assertEqual(self.n_blocks, len(self.glow.flow_layers))
    
    def test_z_to_list_image_has_same_shape_as_glow(self):
        input = torch.randn(4, self.channels, 32, 32)
        expected = None
        
        with torch.no_grad():
            expected, _ = self.glow(input)
        squished = self.glow.z_to_list(input)

        self.assertEqual(len(expected), len(squished))
        # for expected_shape, squished_shape in zip(expected, squished):
        #     print(f"{expected_shape.shape}, expected {squished_shape.shape} recieved")
        
        for z, z_s in zip(expected, squished):
            self.assertEqual(z.shape, z_s.shape)
        

        
        self.assertEqual(input.shape, self.glow.list_to_z(squished).shape )
    
        
    def test_list_to_z_image_has_the_right_shape(self):
        input = torch.randn(4, self.channels, 32, 32)
        
        expected = input.shape

        
        with torch.no_grad():
            input, _ = self.glow(input)
        unsquished = self.glow.list_to_z(input)
        self.assertEqual(expected,unsquished.shape )
    
    def test_2d_is_invertible(self):
        input = torch.randn(4, 2)
        self.glow = Glow(n_channels=input.shape[1], n_steps=3, n_flow_blocks=1, dequantize=False, input_type='factored', n_bits=8 )
        
        z, jacobian = None, None
        inverse = None
        
        with torch.no_grad():
            z, jacobian = self.glow(input)
            inverse = self.glow.inverse(z)
            
        self.assertTrue(torch.allclose(input, inverse, atol=1e-5), f"input: {input} expected, \n {inverse} recieved")
        self.assertFalse(torch.isnan(jacobian).any())

    def test_z_to_list_2D_has_same_shape_as_glow(self):
        input = torch.randn(4, 2)
        self.glow = Glow(n_channels=input.shape[1], n_steps=3, n_flow_blocks=1, dequantize=False, input_type='factored', n_bits=8 )
        expected = None
        
        with torch.no_grad():
            expected, _ = self.glow(input)
        squished = self.glow.z_to_list(input)

        self.assertEqual(len(expected), len(squished))
        # for expected_shape, squished_shape in zip(expected, squished):
        #     print(f"{expected_shape.shape}, expected {squished_shape.shape} recieved")
        
        for z, z_s in zip(expected, squished):
            self.assertEqual(z.shape, z_s.shape)
        

        
        self.assertEqual(input.shape, self.glow.list_to_z(squished).shape )
    
        
    def test_list_to_z_2D_has_the_right_shape(self):
        input = torch.randn(4, 2)
        self.glow = Glow(n_channels=input.shape[1], n_steps=3, n_flow_blocks=1, dequantize=False, input_type='factored', n_bits=8 )
        
        expected = input.shape

        
        with torch.no_grad():
            input, _ = self.glow(input)
        unsquished = self.glow.list_to_z(input)
        self.assertEqual(expected,unsquished.shape )

class TestNeuralPCA(unittest.TestCase):
    def setUp(self):
        self.channels = 3
        self.pca = PCABlock()

    def test_is_invertible(self):
        input = torch.randn(3,2, self.channels, 16, 16)
        
        z = None, None
        
        with torch.no_grad():
            z, V = self.pca(input)
        
        self.assertTrue(torch.allclose(input, self.pca.inverse(z, V),atol=1e-5), f"input: {input} expected, \n {self.pca.inverse(z, V)} recieved")

class TestBatchNorm(unittest.TestCase):
    def setUp(self):
        self.batch_norm = BatchNorm(latent_dim=2, input_type='image', dimensions=16)



    def test_inverse(self):
        x = torch.randn(2, 3, 3, 16, 16)
        z, _, scale, bias = self.batch_norm(x)
        
        x_reconstructed = self.batch_norm.inverse(z, scale, bias)
        self.assertTrue(torch.allclose(x, x_reconstructed, atol=1e-5))



    def test_outputs_statistics(self):
        x = torch.randn(100, 3, 3, 16, 16)
        z, _, _, _ = self.batch_norm(x)
        
        z_flat = z.view(-1, z.size(-1))
        mean = z_flat.mean(dim=0)
        std = z_flat.std(dim=0)
        
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-1))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-1))

    

if __name__ == "__main__":
    unittest.main()