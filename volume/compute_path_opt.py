import torch

def compute_path(latent_A: torch.Tensor,
                 latent_B: torch.Tensor,
                 model,
                 steps=50,
                 lr=1e-2,
                 smooth_term_w=1e-3,
                 iters_per_step=100,
                 ):
    """
    latent_A, latent_B: latent endpoints
    model: f: R^d -> R (scalar output we want to keep constant)
    steps: number of segments along path
    """
    
    # initialize path as linear interpolation
    zs = torch.stack([latent_A * (1 - t/steps) + latent_B * (t/steps) for t in range(steps+1)], dim=0)
    zs = zs.detach().clone().requires_grad_(True)  # (steps+1, d)

    optimizer = torch.optim.Adam([zs], lr=lr)

    target = model(latent_A.unsqueeze(0)).detach()
    
    for _ in range(steps * iters_per_step):
        optimizer.zero_grad()
        
        diff = zs[2:] - 2 * zs[1:-1] + zs[:-2]
        smooth_loss = (diff.pow(2).sum(dim=1)).mean()
        
        outs = model(zs)
        consistency_loss = ((outs - target).pow(2)).mean()
        
        loss = consistency_loss + smooth_term_w * smooth_loss
        loss.backward()
        
        optimizer.step()

        # force endpoints consistensy, else it will blow out at each step
        with torch.no_grad():
            zs[0] = latent_A
            zs[-1] = latent_B
    
    return zs.detach()