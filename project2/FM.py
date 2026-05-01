import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalFM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels,
            self.modelconfig.num_feat,
            self.modelconfig.num_classes,
            self.modelconfig.input_dim,
        )

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  
        device = images.device
        B = images.shape[0]

        x0 = torch.randn_like(images)
        x1 = images

        c = F.one_hot(
            conditions,
            num_classes=self.modelconfig.num_classes
        ).float().to(device)

        mask = (torch.rand(B, device=device) < self.modelconfig.mask_p).view(B, 1)
        c = torch.where(
            mask,
            torch.full_like(c, float(self.modelconfig.condition_mask_value)),
            c
        )

        t = torch.rand(B, 1, device=device)
        t_img = t.view(B, 1, 1, 1)

        x_t = x0 * (1 - t_img) + x1 * t_img

        u_t = x1 - x0
        v_pred = self.network(x_t, t, c)
        loss = self.loss_fn(v_pred, u_t)

        # ==================================================== #
        return loss

    @torch.no_grad()
    def sample(self, conditions, omega):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        B = conditions.shape[0]
        T = self.modelconfig.T

        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float().to(device)
        c_uncond = torch.full_like(c, float(self.modelconfig.condition_mask_value))

        x = torch.randn(
            B, 
            self.modelconfig.num_channels, 
            self.modelconfig.input_dim, 
            self.modelconfig.input_dim, 
            device=device
        )

        dt = 1.0 / T

        for k in range(T):
            t_k = torch.full((B, 1), k * dt, device=device)
            v_cond = self.network(x, t_k, c)
            v_uncond = self.network(x, t_k, c_uncond)
            v_guided = (1.0 + omega) * v_cond - omega * v_uncond
            x = x + v_guided * dt

        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images