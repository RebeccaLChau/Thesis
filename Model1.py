"Beta-Variational Autoencoder (model)"
import torch

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Encoder Class
class Encoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()

        # Adjust convolution layers to ensure output size matches 512 x 16 x 16
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu3 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.relu4 = torch.nn.ReLU()
        
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Changed kernel to 3 to match 16x16 output
        self.relu5 = torch.nn.ReLU()
        
        # Update fully connected layer to match output dimensions of 512 * 16 * 16
        self.fc1 = torch.nn.Linear(512 * 16 * 16, 4048)
        self.fc2 = torch.nn.Linear(4048, 2 * latent_dimension)
        self.latent_dimension = latent_dimension

    def forward(self, x):
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.relu1(x)
        
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.relu2(x)
        
        x = self.conv3(x)
        print("After conv3:", x.shape)
        x = self.relu3(x)
        
        x = self.conv4(x)
        print("After conv4:", x.shape)
        x = self.relu4(x)
        
        x = self.conv5(x)
        print("After conv5:", x.shape)  # Should print (batch_size, 512, 16, 16)
        x = self.relu5(x)
        
        x = x.view(x.size(0), -1)
        print("After flatten:", x.shape)  # Should print (batch_size, 512*16*16 = 131072)
        
        x = torch.nn.functional.relu(self.fc1(x))
        z = self.fc2(x)
        
        z_mean = z[:, :self.latent_dimension]
        z_logvar = z[:, self.latent_dimension:]
        return z_mean, z_logvar

# Decoder Class
class Decoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        super(Decoder, self).__init__()

        self.fc1 = torch.nn.Linear(latent_dimension, 4048)
        self.fc2 = torch.nn.Linear(4048, 512 * 16 * 16)  # Match dimensions of encoder's flatten output
        
        self.conv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        self.upsample = torch.nn.Upsample(size=(500, 500), mode='bilinear', align_corners=False)  # Final output to match input size

    def forward(self, z):
        x = torch.nn.functional.relu(self.fc1(z))
        x = torch.nn.functional.relu(self.fc2(x))
        print("Before reshaping:", x.shape)
        
        x = x.view(-1, 512, 16, 16)
        print("After reshaping:", x.shape)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = torch.sigmoid(self.conv5(x))
        x = self.upsample(x)  # Upsample to 500x500
        return x

# VAE Class
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

# Instantiate the VAE model and move to device
latent_dimension = 64
beta = 1.0
model = VAE(latent_dimension, beta).to(device)

# Test with a sample input
x = torch.randn(1, 3, 500, 500).to(device)  # Example input: batch size 1, 3 channels, 500x500 resolution
x_reconstructed, z_mean, z_logvar = model(x)

