import io
import torch
import torch.nn as nn
import torch.nn.functional as F 
from AutoEncoder.bucketfs_client import BucketFS_client
from AutoEncoder.utils import generate_autoencoder_name

class Autoencoder(nn.Module):
    
    def __init__(self, layers, dropout, continous_columns, categorical_columns, l1_strength=0.0, l2_strength=0.0):
        """
         @brief Initialize Autoencoder with given layer sizes and dropout. This is the base constructor for Autoencoder. You can override it in your subclass if you want to customize the layers.
         @param layers: List of size of layers to use
         @param dropout: List of ( drop_layer, drop_chance )
        """
        super(Autoencoder, self).__init__()
        self.layer_sizes = layers
        self.num_layers = len(layers)
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.continous_columns = continous_columns
        self.categorical_columns = categorical_columns

        # Encoder layers
        encoder_layers = []
        for i in range(self.num_layers - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers[-1].register_forward_hook(self.add_regularization_hook) 
            # for drop_layer, drop_chance in dropout:
            #     if i == drop_layer:
            #         encoder_layers.append(nn.Dropout(drop_chance))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(self.num_layers - 1, 0, -1):
            decoder_layers.append(nn.Linear(layers[i], layers[i - 1]))
            decoder_layers.append(nn.ReLU())
            for drop_layer, drop_chance in dropout:
                if (i == self.num_layers - 1 - drop_layer):
                    decoder_layers.append(nn.Dropout(drop_chance))
            
        decoder_layers.append(nn.Linear(layers[0], layers[0]))
        self.decoder = nn.Sequential(*decoder_layers)

    def add_regularization_hook(self, module, input, output):
        l1_reg = self.l1_strength * F.l1_loss(output, torch.zeros_like(output))
        l2_reg = self.l2_strength * F.mse_loss(output, torch.zeros_like(output))
        module.register_forward_hook(None)  
        module._forward_hooks.clear()
        return output + l1_reg + l2_reg 

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
def build_autoencoder(layers,dropout,continous_columns,categorical_columns,learning_rate=1e-3,weight_decay=0,l1_strength=0,l2_strength=0,load_method=None,weight_path=None):
    """
     @brief Build autoencoder encoder decoder and optimizer.
     @param layers: A list specifying the number of layers and their respective size
     @param dropout: A list of tupple specifying dropout layers position and their respective dropout chance
     @param learning_rate:
     @param weight_decay:  
     @param load_method: Weight loading method. Can be "BucketFS" or "local". Disabled by default
    """
    autoencoder = Autoencoder(layers,dropout,continous_columns,categorical_columns)
    
    if (weight_path is None):
        weight_path = generate_autoencoder_name(layers,load_method)

    if (load_method is not None):    
        if (load_method=="BucketFS"):
            # Load weight from BuckeFS
            client = BucketFS_client()
            weight = client.download(weight_path)
        elif(load_method=="local"):
            # Load weight by local file
            with open(weight_path, 'rb') as file:
                weight = io.BytesIO(file.read())
        autoencoder.load_state_dict(torch.load(weight))

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return autoencoder, encoder, decoder, optimizer
