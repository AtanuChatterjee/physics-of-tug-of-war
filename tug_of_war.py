import numpy as np
from numba import njit

# ---------------------- Simulation Parameters ----------------------
Nagents = 7                 # Number of agents per team
N = 100                     # Number of rope nodes
M = 10                      # Mass of each agent
m = 1                       # Mass of each rope node
gamma = 0.5                 # Damping coefficient for rope nodes
F0 = 5.0                    # Pulling force amplitude
EA = 100.0                  # Elastic modulus times cross-sectional area
g = 1.0                     # Gravitational acceleration
mu_static = 0.3             # Static friction coefficient
mu_kinetic = 0.2            # Kinetic friction coefficient
total_time = 500.0          # Total simulation time
dt_initial = 1e-3           # Initial time step
dt_min = 1e-6               # Minimum time step
dt_max = 1e-2               # Maximum time step
v_threshold = 1e-5          # Velocity threshold for static vs kinetic friction

@njit
def create_rope_nodes(N):
    q = np.linspace(-1, 1, N)
    refLen = np.diff(q).mean()
    midpoint_index = N // 2
    # Set midpoint(s) to zero displacement
    if N % 2 == 0:
        q[midpoint_index - 1] = 0.0
        q[midpoint_index] = 0.0
    else:
        q[midpoint_index] = 0.0
    return q, refLen

def place_agents(q, Nagents, omega_left, omega_right, noise_std):
    N = len(q)
    left_index = np.argmin(np.abs(q + 0.2))
    right_index = np.argmin(np.abs(q - 0.2))

    # Distribute agents towards the left side
    left_team_indices = np.linspace(left_index, 0, Nagents, endpoint=False).astype(np.int32)
    left_team_indices = np.unique(left_team_indices)
    if len(left_team_indices) < Nagents:
        additional_indices = np.setdiff1d(np.arange(0, left_index), left_team_indices)
        left_team_indices = np.concatenate(
            [left_team_indices, additional_indices[:Nagents - len(left_team_indices)]]
        ).astype(np.int32)

    # Distribute agents towards the right side
    right_team_indices = np.linspace(right_index, N - 1, Nagents, endpoint=False).astype(np.int32)
    right_team_indices = np.unique(right_team_indices)
    if len(right_team_indices) < Nagents:
        additional_indices = np.setdiff1d(np.arange(right_index + 1, N), right_team_indices)
        right_team_indices = np.concatenate(
            [right_team_indices, additional_indices[:Nagents - len(right_team_indices)]]
        ).astype(np.int32)

    # Assign frequencies and phases
    # np.random.seed(0)
    omega_left_agents = np.random.normal(omega_left, noise_std, len(left_team_indices))
    omega_right_agents = np.random.normal(omega_right, noise_std, len(right_team_indices))
    omega_agents = np.concatenate([omega_left_agents, omega_right_agents])
    phi_agents = np.random.uniform(0, 2 * np.pi, len(omega_agents))

    return left_team_indices, right_team_indices, omega_agents, phi_agents

@njit
def agent_interaction(omega, phi, dt, K_left, K_right, team_labels, sigma_phi):
    Nagents = len(phi)
    phi_interaction = np.zeros(Nagents)
    for i in range(Nagents):
        interaction_sum = 0.0
        team_size = 0
        for j in range(Nagents):
            if i != j and team_labels[i] == team_labels[j]:
                K = K_left if team_labels[i] == -1 else K_right
                interaction_sum += K * np.sin(phi[j] - phi[i])
                team_size += 1
        if team_size > 0:
            phi_interaction[i] = interaction_sum / team_size
        else:
            phi_interaction[i] = 0.0
    noise = sigma_phi * np.random.normal(0, 1, Nagents) * np.sqrt(dt)
    updated_phi = phi + (omega + phi_interaction) * dt + noise
    return updated_phi

@njit
def getFp(phi, agent_indices, team_signs, F0, N):
    # Half-wave rectified sinusoidal pulling
    # Agents pull in a periodic manner: F = F0 * max(0, sin(phi))
    Fp = np.zeros(N)
    sinusoid = np.maximum(0.0, np.sin(phi))
    Fp_agents = F0 * sinusoid * team_signs
    for idx in range(len(agent_indices)):
        Fp[agent_indices[idx]] = Fp_agents[idx]
    return Fp

@njit
def gradEs(q, EA, refLen):
    dx = q[1:] - q[:-1]
    Fx = EA * (dx - refLen) / refLen
    F = np.zeros_like(q)
    F[:-1] -= Fx
    F[1:] += Fx
    return F

@njit
def getFriction(M, F_net_agents, qDot_agents, mu_static, mu_kinetic, g, v_threshold):
    fr_agents = np.zeros_like(F_net_agents)
    for i in range(len(F_net_agents)):
        F_max_static = mu_static * M * g
        F_k = mu_kinetic * M * g
        if np.abs(qDot_agents[i]) < v_threshold:
            # Agent is approximately stationary
            if np.abs(F_net_agents[i]) <= F_max_static:
                # Static friction balances it out
                fr_agents[i] = -F_net_agents[i]
            else:
                # Exceeds static friction, slips
                fr_agents[i] = -F_max_static * np.sign(F_net_agents[i])
        else:
            # Agent is moving, kinetic friction
            fr_agents[i] = -F_k * np.sign(qDot_agents[i])
    return fr_agents

def run_simulation(omega_left=0.1, omega_right=0.1, noise_std=0.1, K_left=1.0, K_right=1.0):
    q, refLen = create_rope_nodes(N)
    qDot = np.zeros_like(q)

    left_team_indices, right_team_indices, omega_agents, phi_agents = place_agents(
        q, Nagents, omega_left, omega_right, noise_std
    )

    agent_indices = np.concatenate([left_team_indices, right_team_indices])
    team_labels = np.concatenate([
        np.full(len(left_team_indices), -1),
        np.full(len(right_team_indices), 1)
    ])
    team_signs = team_labels

    phi = phi_agents.copy()
    omega = omega_agents.copy()

    m_array = np.full(N, m)
    m_array[agent_indices] = M

    gamma_array = np.full(N, gamma)

    t = 0.0
    t_end = total_time
    dt = dt_initial

    agent_forces_list = []
    agent_phases_list = []
    times_list = []
    q_history_list = []

    # Velocity Verlet scheme:
    # a_n = F_n/m; q_{n+1} = q_n + v_n*dt + 0.5*a_n*dt^2
    # After updating q, compute F_{n+1}, a_{n+1}, then v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt

    # Initial forces and accelerations
    Fs = gradEs(q, EA, refLen)
    Fp = getFp(phi, agent_indices, team_signs, F0, N)
    F_damping = -gamma_array * qDot
    F_net = -Fs + Fp + F_damping

    # Friction for initial step
    F_net_agents = F_net[agent_indices]
    qDot_agents = qDot[agent_indices]
    fr_agents = getFriction(M, F_net_agents, qDot_agents, mu_static, mu_kinetic, g, v_threshold)
    F_net[agent_indices] += fr_agents

    a = F_net / m_array

    save_interval = 1.0  # Save every 1 second
    next_save_time = 0.0

    while t < t_end:
        # Check termination condition: if one side crosses a certain point
        if (q[left_team_indices] > -0.05).any():
            print(f"Simulation terminated: Left team has lost at time {t:.2f}s.")
            break
        if (q[right_team_indices] < 0.05).any():
            print(f"Simulation terminated: Right team has lost at time {t:.2f}s.")
            break

        # Update phases (Kuramoto)
        phi = agent_interaction(omega, phi, dt, K_left, K_right, team_labels, sigma_phi=noise_std)
        phi = phi % (2 * np.pi)

        # Position update
        q_new = q + qDot * dt + 0.5 * a * dt**2

        # Compute new forces
        Fs_new = gradEs(q_new, EA, refLen)
        Fp_new = getFp(phi, agent_indices, team_signs, F0, N)
        F_damping_new = -gamma_array * (qDot)  # damping depends on velocity at old step

        F_net_new = -Fs_new + Fp_new + F_damping_new

        # Compute friction at new step
        # Apply friction based on old velocity first
        # Then do velocity verlet correction
        F_net_new_agents = F_net_new[agent_indices]
        fr_agents_new = getFriction(M, F_net_new_agents, qDot[agent_indices], mu_static, mu_kinetic, g, v_threshold)
        F_net_new[agent_indices] += fr_agents_new

        a_new = F_net_new / m_array

        # Velocity update
        qDot_new = qDot + 0.5 * (a + a_new) * dt

        q = q_new
        qDot = qDot_new
        a = a_new

        t += dt

        # Adaptive time stepping (simple heuristic)
        max_acceleration = np.max(np.abs(a))
        if max_acceleration > 1e4:
            dt = max(dt / 1.2, dt_min)
        elif max_acceleration < 1e2:
            dt = min(dt * 1.2, dt_max)
        # dt is now updated for next iteration

        if t >= next_save_time:
            agent_forces_list.append(Fp_new[agent_indices].copy())
            agent_phases_list.append(phi.copy())
            times_list.append(t)
            q_history_list.append(q.copy())
            print(f"Time: {t:.2f}s")
            next_save_time += save_interval

        # Update old forces for next iteration
        Fs = Fs_new
        Fp = Fp_new
        # friction will be recalculated each step

    agent_forces = np.array(agent_forces_list)
    agent_phases = np.array(agent_phases_list)
    times = np.array(times_list)
    q_history = np.array(q_history_list)

    np.savez('tug_of_war.npz',
             q_history=q_history,
             agent_phases=agent_phases,
             agent_forces=agent_forces,
             times=times,
             team_labels=team_labels,
             agent_indices=agent_indices,
             left_team_indices=left_team_indices,
             right_team_indices=right_team_indices
             )

if __name__ == "__main__":
    run_simulation(omega_left=0.5, omega_right=0.5, noise_std=0.2, K_left=1, K_right=1)
