import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

path = "/work1/yfeng/yfeng/sonic-models/models/particlenet_AK4_PT/1/model.pt"

model = torch.jit.load(path)
model = model.to(device)
model.eval()

nevts = 1
pf_points   = torch.randn(nevts, 2,  100).to(device)
pf_features = torch.randn(nevts, 20, 100).to(device)
pf_mask     = torch.randn(nevts, 1,  100).to(device)
sv_points   = torch.randn(nevts, 2,  10).to(device)
sv_features = torch.randn(nevts, 11, 10).to(device)
sv_mask     = torch.randn(nevts, 1,  10).to(device)

output = model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)
output = output.to("cpu")
print(output)