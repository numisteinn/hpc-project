import torch
import os

from plot import prep_figure, plot_state, plot_finalize


def get_acc(pos, mass, G, softening):
    """
    Compute gravitational acceleration on each particle.
    pos: [N, 3] tensor
    mass: [N, 1] tensor
    Returns: [N, 3] tensor of accelerations.
    """
    # Split coordinates
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # Compute pairwise separations (broadcasting: r_j - r_i)
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # Compute inverse r^3 with softening
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3 = torch.where(
        inv_r3 > 0, inv_r3 ** (-1.5), torch.zeros_like(inv_r3, dtype=torch.float32)
    )

    # Compute acceleration components
    r_ = G * (dx * inv_r3)
    ax = r_ @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # Concatenate acceleration components into [N, 3] tensor
    return torch.cat((ax, ay, az), dim=1)


def get_energy(pos, vel, mass, G):
    """
    Compute kinetic (KE) and potential (PE) energy.
    pos, vel: [N, 3] tensors
    mass: [N, 1] tensor
    """
    KE = 0.5 * torch.sum(mass * vel**2)

    # print(f"Torch {KE, pos[:3], vel[:3]}")
    # Coordinates
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # Compute inverse distances (avoid singularity)
    inv_r = torch.sqrt(dx**2 + dy**2 + dz**2)
    inv_r = torch.where(
        inv_r > 0, 1.0 / inv_r, torch.zeros_like(inv_r, dtype=torch.float32)
    )

    # Use upper triangle to sum each interaction once
    mask = torch.triu(torch.ones_like(inv_r, dtype=torch.float32), diagonal=1)
    PE = G * torch.sum((-(mass @ mass.T) * inv_r) * mask)

    return KE, PE


def main(
    N=100,
    t=0,
    t_end=10.0,
    dt=0.01,
    softening=0.1,
    G=1.0,
    save_plot=True,
    plot_real_time=False,
    pos_init=None,
    vel_init=None,
):
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    print(device)

    # Initialize masses as a [N,1] tensor on device.
    mass = (20.0 * torch.ones((N, 1), device=device)) / N

    # Generate initial conditions as GPU tensors.
    torch.manual_seed(17)
    if pos_init is None:
        pos = torch.randn((N, 3), device=device, dtype=torch.float32)
    else:
        pos = pos_init.to(device)
    if vel_init is None:
        vel = torch.randn((N, 3), device=device, dtype=torch.float32)
    else:
        vel = vel_init.to(device)

    # Convert to Center-of-Mass frame.
    vel -= torch.mean(mass * vel, dim=0) / torch.mean(mass)

    # Initial acceleration and energy.
    acc = get_acc(pos, mass, G, softening)
    KE, PE = get_energy(pos, vel, mass, G)

    Nt = int((t_end / dt) + 0.5)

    pos_save = torch.zeros((N, 3, Nt + 1), device=device)
    pos_save[:, :, 0] = pos
    KE_save = torch.zeros(Nt + 1, device=device)
    KE_save[0] = KE
    PE_save = torch.zeros(Nt + 1, device=device)
    PE_save[0] = PE
    t_all = torch.linspace(0, Nt * dt, Nt + 1, device=device)

    if save_plot:
        prep_figure()

    # Main simulation loop.
    for i in range(1, Nt + 1):
        # First half kick
        vel += acc * (dt / 2.0)
        # Drift: update positions
        pos += vel * dt
        # Recompute acceleration
        acc = get_acc(pos, mass, G, softening)
        # Second half kick
        vel += acc * (dt / 2.0)
        t += dt

        KE, PE = get_energy(pos, vel, mass, G)
        pos_save[:, :, i] = pos
        KE_save[i] = KE
        PE_save[i] = PE

        if plot_real_time:
            # For plotting, data may need to be moved to CPU.
            plot_state(
                i,
                t_all.cpu().numpy(),
                pos_save.cpu().numpy(),
                KE_save.cpu().numpy(),
                PE_save.cpu().numpy(),
            )

    if save_plot:
        plot_state(
            i,
            t_all.cpu().numpy(),
            pos_save.cpu().numpy(),
            KE_save.cpu().numpy(),
            PE_save.cpu().numpy(),
        )
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"nbody_pytorch_{N}_{t_end}_{dt}_{softening}_{G}.png",
    )
    plot_finalize(output_path)

    return pos, vel, KE_save, PE_save


if __name__ == "__main__":
    main()
