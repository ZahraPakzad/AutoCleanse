import io
import torch
from AutoCleanse.bucketfs_client import *

# client.delete("/autoencoder/autoencoder_110_1000_500_35.pth")
# client.delete("/autoencoder/test_scaler.pkl")
# client.delete("/autoencoder/test_encoder.pkl")
# client.delete("container/gpt-release-FZUN5MRZBFJPZWNNOIXWN5B34ZNRZUW2PR7VA6FUT4SYTAX2URQA.tar.gz")

# with open("/home/tung/development/AutoEncoder/BU/autoencoder_1000_500_35.pth", "rb") as f:
#     buffer = io.BytesIO(f.read())
#     buffer.seek(0)
#     client.upload('/autoencoder/autoencoder_110_1000_500_35.pth', buffer)

# with open("/home/tung/development/AutoEncoder/test_scaler.pkl", "rb") as f:
#     buffer = io.BytesIO(f.read())
#     buffer.seek(0)
#     client.upload("/autoencoder/test_scaler.pkl", buffer)

# with open("/home/tung/development/AutoEncoder/test_encoder.pkl", "rb") as f:
#     buffer = io.BytesIO(f.read())
#     buffer.seek(0)
#     client.upload("/autoencoder/test_encoder.pkl", buffer)
print(bucketfs_client.check("autoencoder/autoencoder_test.pth"))

# bucketfs_client.view()
# print(client.check("autoencoder/autoencoder_test_scaler.pkl"))
