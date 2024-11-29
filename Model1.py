"Beta-Variational Autoencoder (model)"
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'cuda' else {}

class Encoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1).to(device)  # 250x250
        self.relu1 = torch.nn.ReLU().to(device)
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1).to(device)  # 125x125
        self.relu2 = torch.nn.ReLU().to(device)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1).to(device)  # 63x63
        self.relu3 = torch.nn.ReLU().to(device)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1).to(device)  # 32x32
        self.relu4 = torch.nn.ReLU().to(device)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1).to(device)  # 16x16
        self.relu5 = torch.nn.ReLU().to(device)
        self.fc1 = torch.nn.Linear(512 * 16 * 16, 4096).to(device)
        self.fc2 = torch.nn.Linear(4096, 2 * latent_dimension).to(device)
        self.latent_dimension = latent_dimension

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        z = self.fc2(x)
        z_mean = z[:, :self.latent_dimension]
        z_logvar = z[:, self.latent_dimension:]
        return z_mean, z_logvar


class Decoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dimension, 4096).to(device)
        self.fc2 = torch.nn.Linear(4096, 512 * 16 * 16).to(device)
        self.conv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1).to(device)  # 32x32
        self.relu1 = torch.nn.ReLU().to(device)
        self.conv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1).to(device)  # 64x64
        self.relu2 = torch.nn.ReLU().to(device)
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1).to(device)  # 125x125
        self.relu3 = torch.nn.ReLU().to(device)
        self.conv4 = torch.nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1).to(device)  # 250x250
        self.relu4 = torch.nn.ReLU().to(device)
        self.conv5 = torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1).to(device)  # 500x500
        self.sigmoid = torch.nn.Sigmoid().to(device)

    def forward(self, z):
        x = torch.nn.functional.relu(self.fc1(z))
        x = torch.nn.functional.relu(self.fc2(x))
        x = x.view(-1, 512, 16, 16)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.sigmoid(self.conv5(x))
        return x


class VAE(torch.nn.Module):
    def __init__(self, latent_dimension, beta):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dimension)
        self.decoder = Decoder(latent_dimension)
        self.beta = beta

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_logvar

    def loss_function(self, x, x_reconstructed, z_mean, z_logvar):
        reconstruction_loss = torch.nn.MSELoss(reduction='sum')(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return reconstruction_loss + self.beta * kl_divergence, reconstruction_loss, kl_divergence

    def decode(self, z):
        return self.decoder(z)


