import torch
import numpy as np

from volume.config import DEV

def sdf_torus(p, t = torch.tensor([0.35, 0.175])):
    """
    Signed distance function for a torus in PyTorch.
    
    p: (N, 3) tensor of 3D points
    t: (2,) tensor with major and minor radius
    """
    pxz = torch.norm(p[..., [0, 2]], dim=-1) - t[0]
    q = torch.stack([pxz, p[..., 1]], dim=-1)
    return torch.norm(q, dim=-1) - t[1]

def sdf_2_torus(p, t = torch.tensor([0.35, 0.175])):
    """
    Signed distance function for a genus-2-torus.
    
    p: (N, 3) tensor of 3D points
    t: (2,) tensor with major and minor radius
    """
    x1 = p[..., 0:1] + 0.3
    yz = p[..., 1:]
    x2 = x1 - t[0] - t[1]                       
    p1 = torch.cat([x1, yz], dim=-1)
    p2 = torch.cat([x2, yz], dim=-1)
    return torch.minimum(sdf_torus(p1, t), sdf_torus(p2, t))


def sdf_solid_angle(p, c=torch.tensor([np.sin(np.pi/4), np.cos(np.pi/4)]), radius=0.7):
    """
    p: (..., 3) tensor of 3D points
    c: (2,) tensor [sin(angle), cos(angle)]
    ra: float, radius
    """
    # q = [length(xz), y]
    c = c.to(DEV)
    q = torch.stack([
        torch.norm(p[..., [0, 2]], dim=-1),
        p[..., 1]
    ], dim=-1).to(DEV)  # (..., 2)

    l = torch.norm(q, dim=-1) - radius

    dot_qc = (q * c).sum(dim=-1)
    clamp_dot = torch.clamp(dot_qc, 0.0, radius)
    qc = c * clamp_dot.unsqueeze(-1)

    m = torch.norm(q - qc, dim=-1)
    s = torch.sign(c[1] * q[..., 0] - c[0] * q[..., 1])

    return torch.max(l, m * s)

def sdf_sphere(query_points, radius = 0.37):
    dist = torch.linalg.norm(query_points, axis=-1)
    return dist - radius

def sdf_box(p: torch.Tensor, b = torch.tensor([0.16, 0.33, 0.50])) -> torch.Tensor:
    """
    SDF for axis-aligned box centered at origin.
    
    p: (..., 3) tensor of query points
    b: (3,) tensor of box half-sizes along x, y, z (box extends from -b to +b)
    
    Returns: (...) tensor of signed distances
    """
    b = b.to(DEV)
    q = torch.abs(p) - b
    q_max = torch.clamp(q, min=0.0)  # same as max(q, 0)
    outside_dist = torch.norm(q_max, dim=-1)
    inside_dist = torch.min(torch.max(q[..., 0], torch.max(q[..., 1], q[..., 2])), torch.tensor(0.0, device=p.device))
    return outside_dist + inside_dist

class SDF_interpolator:
    """
    SDF interpolator for 3 sdfs.
    Will be used to generate pseudo 2D
    latent vectors.
    """
    def __init__(self, sdf1=None, sdf2=None, sdf3=None):
        self.sdf1 = sdf1 or sdf_solid_angle
        self.sdf2 = sdf2 or sdf_sphere
        self.sdf3 = sdf3 or sdf_box

    def __call__(self, parameters, coords):
        alpha, beta = parameters.flatten().tolist()
        gamma = 1 - alpha - beta
        return alpha * self.sdf1(coords) + beta * self.sdf2(coords) + gamma * self.sdf3(coords)

    def eval(self):
        """
        Just to make it compatible with predict_sdf in compute_volume.py
        """
        pass

