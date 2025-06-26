import torch
import torchvision
import lightning as L
from torch import nn
from aim import Image


# Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


# Generator model with self-attention
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Critic Model (Discriminator)
class Critic(nn.Module):
    def __init__(self, img_channels):
        super(Critic, self).__init__()
        def critic_block(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *critic_block(img_channels, 64),
            *critic_block(64, 128),
            *critic_block(128, 256),
            
            *critic_block(256, 512),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Output Wasserstein distance
        )

    def forward(self, img):
        return self.model(img)


class WGAN(L.LightningModule):
    def __init__(self, latent_dim=128, img_channels=3, lr_g=1e-4, lr_c=5e-5, b1=0.5, b2=0.999, lambda_gp=10):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim)
        self.critic = Critic(img_channels)
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def compute_gradient_penalty(self, real_imgs, fake_imgs):
        """Compute the gradient penalty for WGAN-GP."""
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=real_imgs.device)
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
        fake = torch.ones(real_imgs.size(0), 1, device=real_imgs.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Get optimizers
        opt_g, opt_c = self.optimizers()

        c_loss = torch.nan
        if batch_idx % 1 == 0:
            # Train Critic
            opt_c.zero_grad()
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)  # стандартное нормальное распределение
            z = normal_dist.sample((batch_size, self.hparams.latent_dim))
            z = z.view(batch_size, self.hparams.latent_dim, 1, 1).to(real_imgs.device)
            fake_imgs = self(z)
            # real_imgs = real_imgs + torch.randn_like(real_imgs) * 0.1
            # fake_imgs = fake_imgs + torch.randn_like(fake_imgs) * 0.1
            real_loss = torch.mean(self.critic(real_imgs))
            fake_loss = torch.mean(self.critic(fake_imgs))
            gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
            c_loss = fake_loss - real_loss + self.hparams.lambda_gp * gradient_penalty
            c_loss.backward()
            opt_c.step()
        self.log("c_loss", c_loss, prog_bar=True)

        g_loss = torch.nan
        if batch_idx % 8 == 0:
            # --- Train Generator ---
            opt_g.zero_grad()
            normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)  # стандартное нормальное распределение
            z = normal_dist.sample((batch_size, self.hparams.latent_dim))
            z = z.view(batch_size, self.hparams.latent_dim, 1, 1).to(real_imgs.device)
            gen_imgs = self(z)
            g_loss = -torch.mean(self.critic(gen_imgs))
            g_loss.backward()
            opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)

        
        if batch_idx % (8 * 20) == 0:
            grid = torchvision.utils.make_grid(gen_imgs[:8], normalize=True, value_range=(-1, 1))
            self.logger.experiment.track(Image(grid), "generated_images", self.global_step)

            grid = torchvision.utils.make_grid(real_imgs[:8], normalize=True, value_range=(-1, 1))
            self.logger.experiment.track(Image(grid), "real_images", self.global_step)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g,
                                 betas=(self.hparams.b1, self.hparams.b2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.lr_c,
                                 betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_c], []
