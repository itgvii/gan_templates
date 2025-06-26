import torch
import torchvision
import lightning as L
from torch import nn
from aim import Image


# Generator Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels and out_channels differ, use a 1x1 conv for shortcut
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.init_size = 4  # Initial image size (4x4)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 8x8
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),  # 16x16
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),  # 32x32
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

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

        self.generator = Generator(latent_dim, img_channels)
        self.critic = Critic(img_channels)
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False  # <-- Add this line

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

    def training_step(self, batch, batch_idx):  # <-- Remove optimizer_idx
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Get optimizers
        opt_g, opt_c = self.optimizers()

        c_loss = torch.nan
        if batch_idx % 1 == 0:
            # Train Critic
            opt_c.zero_grad()
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_imgs = self(z)
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
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            gen_imgs = self(z)
            g_loss = -torch.mean(self.critic(gen_imgs))
            g_loss.backward()
            opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)

        
        if batch_idx % (8 * 6) == 0:
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
