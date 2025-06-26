import torch
import torchvision
import lightning as L
from torch import nn
from aim import Image


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



# Discriminator model with binary output for each image in the batch
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),  # Add dropout layer
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5),  # Add dropout layer
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(512 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)  # Flatten the tensor
        output = self.linear(output)
        output = self.sigmoid(output)
        output = output.view(-1, 1)  # Reshape to have shape (batch_size, 1)
        return output


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


class GAN(L.LightningModule):
    def __init__(self, latent_dim=100, lr=2e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
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

    def training_step(self, batch, batch_idx):  # <-- Remove optimizer_idx
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        z = torch.randn(batch_size, self.hparams.latent_dim, 1, 1, device=self.device)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Get optimizers
        opt_g, opt_d = self.optimizers()

        d_loss = torch.nan
        if batch_idx % 1 == 0:
            # --- Train Discriminator ---
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
            fake_imgs = self(z).detach()
            fake_loss = self.adversarial_loss(self.discriminator(fake_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            self.log("d_loss", d_loss, prog_bar=True)

        g_loss_log = torch.nan
        if batch_idx % 1 == 0:
            # --- Train Generator ---
            generated_imgs = self(z)
            validity = self.discriminator(generated_imgs)
            with torch.no_grad():
                g_loss_log = self.adversarial_loss(validity, valid)
            g_loss = torch.log(validity).mean()
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
        self.log("g_loss", g_loss_log, prog_bar=True)

        
        if batch_idx % 50 == 0:
            grid = torchvision.utils.make_grid(generated_imgs[:8], normalize=True, value_range=(-1, 1))
            self.logger.experiment.track(Image(grid), "generated_images", self.global_step)

            grid = torchvision.utils.make_grid(real_imgs[:8], normalize=True, value_range=(-1, 1))
            self.logger.experiment.track(Image(grid), "real_images", self.global_step)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr,
                                 betas=(self.hparams.b1, self.hparams.b2), maximize=True)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr,
                                 betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d], []

