import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        t = (t_s - 1) / (T - 1)
        beta_t = beta_1 + t * (beta_T - beta_1)
        sqrt_beta_t = torch.sqrt(beta_t)

        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)

        t_all = (torch.arange(1, T + 1, device=device).float() - 1) / (T - 1)
        beta_t_all = beta_1 + t_all * (beta_T - beta_1)
        alpha_t_all = 1 - beta_t_all

        alpha_bar_all = torch.cumprod(alpha_t_all, dim=0)
        t_idx = t_s.long().squeeze() - 1
        alpha_t_bar = alpha_bar_all[t_idx].view(-1, 1, 1, 1)

        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)      
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

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
        B = images.shape[0]

        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float().to(images.device)

        t = torch.randint(1, self.modelconfig.T+1, (B,1), device=images.device)
        eps = torch.randn_like(images)

        sched = self.scheduler(t)
        sqrt_alpha_bar = sched['sqrt_alpha_bar']
        sqrt_oneminus_alpha_bar = sched['sqrt_oneminus_alpha_bar']

        x_t = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * eps

        t_norm = t.float() / self.modelconfig.T
        eps_pred = self.network(x_t, t_norm, c)

        noise_loss = self.loss_fn(eps_pred, eps)

        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
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

        device = next(self.parameters()).device
        B = conditions.shape[0]

        c = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float().to(device)

        X_t = torch.randn(
            B,
            self.modelconfig.num_channels,
            self.modelconfig.input_dim,
            self.modelconfig.input_dim
        ).to(device)

        for t in reversed(range(1, T + 1)):
            t_tensor = torch.full((B, 1), t, device=device)

            sched = self.scheduler(t_tensor)

            beta_t = sched['beta_t'].view(-1, 1, 1, 1)
            oneover_sqrt_alpha = sched['oneover_sqrt_alpha'].view(-1, 1, 1, 1)
            sqrt_oneminus_alpha_bar = sched['sqrt_oneminus_alpha_bar']

            t_norm = t_tensor.float() / T

            eps = self.network(X_t, t_norm, c)

            if t > 1:
                z = torch.randn_like(X_t)
            else:
                z = torch.zeros_like(X_t)

            X_t = oneover_sqrt_alpha * (
                X_t - (beta_t / sqrt_oneminus_alpha_bar) * eps
            ) + torch.sqrt(beta_t) * z


        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images