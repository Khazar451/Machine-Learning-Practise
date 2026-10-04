import torch
from torch.utils.benchmark import Timer

# lets create a Pytorch tensor

X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
# print(X.shape)

# lets check Pytorch accessing the gpu

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
# print(device)

# let's create tensor on gpu

M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M = M.to(device)
# print(M.device)

R = M @ M.T
# print(R)

M = torch.rand((1000, 1000)) # on the cpu
print(Timer(stmt="M @ M.T", globals={"M": M}).timeit(100))
M = torch.rand((1000, 1000), device="cuda") # on the gpu
print(Timer(stmt="M @ M.T", globals={"M": M}).timeit(100))
