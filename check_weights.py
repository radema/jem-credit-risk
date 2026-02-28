import torch

state = torch.load(
    "models/artifacts/autoencoder.pth", map_location="cpu", weights_only=True
)
print(f"Number of keys: {len(state.keys())}")
print("First 10 keys:")
for k in list(state.keys())[:10]:
    print(k)
