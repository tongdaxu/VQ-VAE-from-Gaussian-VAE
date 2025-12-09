import torch
import gq_cuda
import torch.nn.functional as F

def reference(mu,std,noise,result,beta):
    result[:,:] = -torch.sum(((noise[None,:,:] - mu[:,None,:]) / std[:,None,:])**2 - noise[None,:,:] ** 2 * beta, dim=2)

if __name__ == "__main__":
    from tqdm import tqdm
    b = 1024 * 4 * 8 * 8
    dim = 16
    n_samples = 65536
    beta = 1.0
    result = torch.zeros([b, n_samples]).cuda().contiguous()
    noise = torch.randn([n_samples, dim]).cuda()

    for _ in tqdm(range(10000)):
        mu = torch.randn([b, dim]).cuda()
        std = torch.abs(torch.randn([b, dim])).cuda()
        gq_cuda.ops.gq_cuda(
            mu,std,noise,result,dim,b,n_samples,beta
        )
        # idx = torch.argmax(result, dim=1)


