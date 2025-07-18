
def compute_path(latent_A, latent_B, model, n_steps=100):
    """
    Compute a path in latent space from latent_A to latent_B,
    while trying to keepmthe model output quantity constant along the path.

    model here refers to classifier or the regressor not DeepSDF!

    Assumes that model(latent_A) = model(latent_B)
    """
    current_latent = latent_A.clone().detach().requires_grad_(True)
    path = [current_latent]
    target_quantity = model(latent_B).detach().requires_grad_(False)

    for steps_remaining in range(n_steps-1, 1, -1):
        current_quantity = model(current_latent)
        current_quantity.backward()

        gradient = current_latent.grad.clone()
        current_latent.grad.zero_()

        gradient_norm_squared = gradient.norm()**2
        remaining_path = latent_B - current_latent

        path_projection = remaining_path - (remaining_path@gradient / gradient_norm_squared) * gradient
        gradient_correction = (target_quantity - current_quantity) / gradient_norm_squared * gradient
        
        current_latent = path_projection / (steps_remaining+1) + gradient_correction

        current_latent = current_latent.detach().requires_grad_(True)

        path.append(current_latent.clone())

    path.append(latent_B)
    return path




def compute_path2(latent_A, latent_B, model, n_steps=100):
    """
    Compute a path in latent space from latent_A to latent_B,
    while trying to keepmthe model output quantity constant along the path.

    model here refers to classifier or the regressor not DeepSDF!

    Assumes that model(latent_A) = model(latent_B)
    """
    current_latent = latent_A.clone().detach().requires_grad_(True)
    path = [current_latent]
    target_quantity = model(latent_B).detach().requires_grad_(False)

    epsilon = 0.03

    for steps_remaining in range(n_steps-1, 1, -1):
        current_quantity = model(current_latent)
        current_quantity.backward()

        gradient = current_latent.grad.clone()
        current_latent.grad.zero_()

        gradient_norm_squared = gradient.norm()**2
        remaining_path = latent_B - current_latent

        if remaining_path.norm() < epsilon:
            break

        path_projection = remaining_path - (remaining_path@gradient / gradient_norm_squared) * gradient
        # gradient_correction = (target_quantity - current_quantity) / gradient_norm_squared * gradient

        path_projection /= path_projection.norm()
        
        # current_latent = path_projection / (steps_remaining+1) + gradient_correction
        new_latent = current_latent + path_projection * 0.02

        current_latent = new_latent.detach().requires_grad_(True)

        path.append(current_latent.detach().clone())

    path.append(latent_B)
    return path