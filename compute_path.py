
def compute_path(latent_A, latent_B, model, n_steps=100):
    """
    Compute a path in latent space from latent_A to latent_B,
    while trying to keepmthe model output quantity constant along the path.

    Assumes that model(latent_A) = model(latent_B)
    """
    current_latent = latent_A.clone()
    path = [current_latent]
    target_quantity = model(latent_B).detach().requires_grad_(False)

    for steps_remaining in range(n_steps-1, 1, -1):
        current_quantity = model(current_latent)
        current_quantity.backward()

        gradient = current_latent.grad
        gradient_norm_squared = gradient.norm()**2
        remaining_path = latent_B - current_latent

        path_projection = remaining_path - (remaining_path@gradient / gradient_norm_squared) * gradient
        gradient_correction = (target_quantity - current_quantity) / gradient_norm_squared * gradient
        
        current_latent = path_projection / (steps_remaining+1) + gradient_correction

        path.append(current_latent.clone())

    path.append(latent_B)
    return path