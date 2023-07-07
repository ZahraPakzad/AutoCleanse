import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self, layer_sizes, dropout):
        """
         @brief Initialize Autoencoder with given layer sizes and dropout. This is the base constructor for Autoencoder. You can override it in your subclass if you want to customize the layers.
         @param layer_sizes List of size of layers to use
         @param dropout List of ( drop_layer, drop_chance )
        """
        super(Autoencoder, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Encoder layers
        encoder_layers = []
        # Add linear and reLU layers to the encoder.
        for i in range(self.num_layers - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.ReLU())
            # Add dropout to encoder layer.
            for drop_layer, drop_chance in dropout:
                if i == drop_layer:
                    encoder_layers.append(nn.Dropout(drop_chance))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(self.num_layers - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(layer_sizes[0], layer_sizes[0]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def build_autoencoder(layers,dropout):
    """
     @brief Build autoencoder encoder decoder and optimizer.
     @param layers Number of layers to use
     @param dropout Dropout specification for the architecture
     @return A tuple of : py : class : ` torch. autoencoder. Autoencoder ` : py : class : ` torch. encoder. Encoder
    """
    autoencoder = Autoencoder(layers,dropout)
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    return autoencoder, encoder, decoder, optimizer
