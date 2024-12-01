"Beta-Variational Autoencoder (model)"
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'cuda' else {}

class Encoder(torch.nn.Module):
    def __init__(self, latent_dimension, input_channels=3, input_size=500):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1)  # 250x250
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1)  # 125x125
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 63x63
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 32x32
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 16x16

        # Dynamically calculate the flattened size
        test_input = torch.zeros(1, input_channels, input_size, input_size)
        conv_out = self._get_conv_output_size(test_input)
        self.fc1 = torch.nn.Linear(conv_out, 4096)
        self.fc2 = torch.nn.Linear(4096, 2 * latent_dimension)

        self.latent_dimension = latent_dimension

    def _get_conv_output_size(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)
        z_mean = z[:, :self.latent_dimension]
        z_logvar = z[:, self.latent_dimension:]
        return z_mean, z_logvar


class Decoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dimension, 4096)
        self.fc2 = torch.nn.Linear(4096, 512 * 16 * 16)
        self.conv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 32x32
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 64x64
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 128x128
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)  # 256x256
        self.relu4 = torch.nn.ReLU()
        # Adjust the final layer to output exactly 500x500
        self.conv5 = torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0)  # 500x500
        self.sigmoid = torch.nn.Sigmoid()

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
        # Resize reconstructed output to match the input size if needed
        if x_reconstructed.size(2) != x.size(2) or x_reconstructed.size(3) != x.size(3):
            x_reconstructed = torch.nn.functional.interpolate(x_reconstructed, size=(x.size(2), x.size(3)), mode='bilinear')
        
        reconstruction_loss = torch.nn.MSELoss(reduction='sum')(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return reconstruction_loss + self.beta * kl_divergence, reconstruction_loss, kl_divergence

    def decode(self, z):
        return self.decoder(z)





