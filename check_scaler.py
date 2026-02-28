import torch

state = torch.load("models/artifacts/scaler.pth", map_location="cpu", weights_only=True)
print(type(state))
print(state.keys())
