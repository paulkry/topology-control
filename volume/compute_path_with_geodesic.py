def compute_path(latent_A, latent_B, model, n_steps=100):
    # Original pathfinding logic here
    pass

def compute_path2(latent_A, latent_B, model, n_steps=100):
    # Alternative pathfinding logic here
    pass

# --------------------------------------
# Geodesic-based pathfinding in latent space
# --------------------------------------

import torch

def compute_geodesic_path(latent_A, latent_B, decoder, n_steps=100, n_iter=10, lr=0.05):
    """
    Approximate a geodesic between latent_A and latent_B in latent space
    using the decoder to estimate the latent manifold.

    decoder: neural network decoder(z) representing the manifold
    """
    # Create initial linear path
    z_path = torch.stack([
        latent_A + (latent_B - latent_A) * t
        for t in torch.linspace(0, 1, n_steps)
    ], dim=0)

    # Only optimize interior points
    z_path = z_path.detach()
    z_path.requires_grad = False
    z_path_param = torch.nn.Parameter(z_path[1:-1].clone(), requires_grad=True)
    optimizer = torch.optim.Adam([z_path_param], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        energy = 0.0
        # Reconstruct full path with fixed endpoints
        full_path = torch.cat([z_path[:1], z_path_param, z_path[-1:]], dim=0)
        for i in range(n_steps - 1):
            z1 = full_path[i]
            z2 = full_path[i + 1]
            dz = z2 - z1

            mid_z = (z1 + z2) / 2
            mid_z = mid_z.detach().requires_grad_(True)
            x = decoder(mid_z)
            # Compute Jacobian of decoder at mid_z
            J = torch.autograd.functional.jacobian(decoder, mid_z, create_graph=True)
            if J.ndim > 2:
                J = J.squeeze(0)
            pushforward = torch.matmul(J, dz)
            energy += torch.sum(pushforward**2)

        energy.backward()
        optimizer.step()

    # Return the optimized path with endpoints
    return torch.cat([z_path[:1], z_path_param.detach(), z_path[-1:]], dim=0)
