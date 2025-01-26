import itertools
import numpy as np
from numba import njit
import concurrent.futures

# ---------------------- Simulation Parameters ----------------------
Nagents = 7                 # Number of agents per team
N = 100                     # Number of rope nodes
M = 10                      # Mass of each agent
m = 1                       # Mass of each rope node
gamma = 0.5                 # Damping coefficient for rope nodes
EA = 100.0                  # Elastic modulus times cross-sectional area
g = 1.0                     # Acceleration due to gravity
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
def getFp(phi, agent_indices, team_signs, F0_left, F0_right, left_team_size, N):
    # Half-wave rectified sinusoidal pulling
    # First half of agent_indices are left team, second half are right team
    # phi -> phases, team_signs -> -1 for left, +1 for right
    # Apply F0_left to left team, F0_right to right team
    Fp = np.zeros(N)
    sinusoid = np.maximum(0.0, np.sin(phi))

    # Split agent_indices into left and right
    left_indices = agent_indices[:left_team_size]
    right_indices = agent_indices[left_team_size:]

    # Forces for left team
    Fp_left_agents = F0_left * sinusoid[:left_team_size] * team_signs[:left_team_size]
    # Forces for right team
    Fp_right_agents = F0_right * sinusoid[left_team_size:] * team_signs[left_team_size:]

    for i, idx in enumerate(left_indices):
        Fp[idx] = Fp_left_agents[i]
    for i, idx in enumerate(right_indices):
        Fp[idx] = Fp_right_agents[i]

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
    F_max_static = mu_static * M * g
    F_k = mu_kinetic * M * g
    for i in range(len(F_net_agents)):
        if np.abs(qDot_agents[i]) < v_threshold:
            # Agent nearly stationary
            if np.abs(F_net_agents[i]) <= F_max_static:
                # Static friction balances exactly
                fr_agents[i] = -F_net_agents[i]
            else:
                # Sliding occurs
                fr_agents[i] = -F_max_static * np.sign(F_net_agents[i])
        else:
            # Kinetic friction
            fr_agents[i] = -F_k * np.sign(qDot_agents[i])
    return fr_agents

def place_agents(q, Nagents, omega_left, omega_right, noise_std):
    N = len(q)
    left_index = np.argmin(np.abs(q + 0.2))
    right_index = np.argmin(np.abs(q - 0.2))

    left_team_indices = np.linspace(left_index, 0, Nagents, endpoint=False).astype(int)
    left_team_indices = np.unique(left_team_indices)
    if len(left_team_indices) < Nagents:
        additional_indices = np.setdiff1d(np.arange(0, left_index), left_team_indices)
        left_team_indices = np.concatenate(
            [left_team_indices, additional_indices[:Nagents - len(left_team_indices)]]
        ).astype(int)

    right_team_indices = np.linspace(right_index, N - 1, Nagents, endpoint=False).astype(int)
    right_team_indices = np.unique(right_team_indices)
    if len(right_team_indices) < Nagents:
        additional_indices = np.setdiff1d(np.arange(right_index + 1, N), right_team_indices)
        right_team_indices = np.concatenate(
            [right_team_indices, additional_indices[:Nagents - len(right_team_indices)]]
        ).astype(int)

    # Frequencies and phases
    # np.random.seed(0)
    omega_left_agents = np.random.normal(omega_left, noise_std, len(left_team_indices))
    omega_right_agents = np.random.normal(omega_right, noise_std, len(right_team_indices))
    omega_agents = np.concatenate([omega_left_agents, omega_right_agents])
    phi_agents = np.random.uniform(0, 2 * np.pi, len(omega_agents))

    return left_team_indices, right_team_indices, omega_agents, phi_agents

def run_simulation_outcome(omega_left, omega_right, noise_std, K_left, K_right, F0_left, F0_right):
    # Set up rope and agents
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
    left_team_size = len(left_team_indices)

    phi = phi_agents.copy()
    omega = omega_agents.copy()

    m_array = np.full(N, m)
    m_array[agent_indices] = M

    gamma_array = np.full(N, gamma)

    t = 0.0
    dt = dt_initial

    # Initial force computations
    Fs = gradEs(q, EA, refLen)
    Fp = getFp(phi, agent_indices, team_labels, F0_left, F0_right, left_team_size, N)
    F_damping = -gamma_array * qDot
    F_net = -Fs + Fp + F_damping

    qDot_agents = qDot[agent_indices]
    F_net_agents = F_net[agent_indices]
    fr_agents = getFriction(M, F_net_agents, qDot_agents, mu_static, mu_kinetic, g, v_threshold)
    F_net[agent_indices] += fr_agents

    a = F_net / m_array

    while t < total_time:
        if (q[left_team_indices] > -0.05).any():
            return "right team wins"
        if (q[right_team_indices] < 0.05).any():
            return "left team wins"

        # Update phases (Kuramoto)
        phi = agent_interaction(omega, phi, dt, K_left, K_right, team_labels, sigma_phi=noise_std)
        phi = phi % (2 * np.pi)

        # Velocity Verlet: position update
        q_new = q + qDot * dt + 0.5 * a * dt**2

        # Compute new forces based on q_new
        Fs_new = gradEs(q_new, EA, refLen)
        Fp_new = getFp(phi, agent_indices, team_labels, F0_left, F0_right, left_team_size, N)
        # Damping approximated with old velocity
        F_damping_new = -gamma_array * qDot

        F_net_new = -Fs_new + Fp_new + F_damping_new

        # Use old velocities for friction direction
        F_net_new_agents = F_net_new[agent_indices]
        fr_agents_new = getFriction(M, F_net_new_agents, qDot[agent_indices], mu_static, mu_kinetic, g, v_threshold)
        F_net_new[agent_indices] += fr_agents_new

        a_new = F_net_new / m_array

        # Velocity update
        qDot_new = qDot + 0.5 * (a + a_new) * dt

        # Adaptive time stepping (simple heuristic)
        max_acceleration = np.max(np.abs(a_new))
        if max_acceleration > 1e4:
            dt = max(dt / 1.2, dt_min)
        elif max_acceleration < 1e2:
            dt = min(dt * 1.2, dt_max)

        q = q_new
        qDot = qDot_new
        a = a_new
        Fs = Fs_new
        Fp = Fp_new

        t += dt

    return "tie"

def simulate_noise(noise, iterations, omega_left, omega_right, K_left, K_right, F0_left, F0_right):
    left_wins = 0
    right_wins = 0
    ties = 0
    for _ in range(iterations):
        outcome = run_simulation_outcome(
            omega_left=omega_left,
            omega_right=omega_right,
            noise_std=noise,
            K_left=K_left,
            K_right=K_right,
            F0_left=F0_left,
            F0_right=F0_right
        )
        if outcome == "left team wins":
            left_wins += 1
        elif outcome == "right team wins":
            right_wins += 1
        else:
            ties += 1
    return noise, left_wins, right_wins, ties

def noise_sweep(omega_left, omega_right, K_left, K_right, F0_left, F0_right, noise_values, iterations):
    results = []
    tasks = [(noise, iterations, omega_left, omega_right, K_left, K_right, F0_left, F0_right) for noise in noise_values]

    num_workers = None
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(simulate_noise, *t): t[0] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            noise, left_wins, right_wins, ties = future.result()
            results.append((noise, left_wins, right_wins, ties))
            print(f"Noise: {noise:.3f}, Left Wins: {left_wins}, Right Wins: {right_wins}, Ties: {ties}")

    results.sort(key=lambda x: x[0])  # Sort by noise
    return results

if __name__ == "__main__":
    print("Precompiling Numba functions...")
    # Precompile
    run_simulation_outcome(0.1, 0.1, 0.0, 1.0, 1.0, F0_left=5.0, F0_right=5.0)
    print("Precompilation completed.")

    # Fixed parameters
    omega_left = 0.5
    omega_right = 0.3
    K_left = 1.0
    K_right = 1.0
    F0_left = 5.0
    F0_right = 5.0

    noise_values = np.linspace(0, 1, 20)
    iterations = 100
    print("Starting noise sweep...")
    results = noise_sweep(omega_left, omega_right, K_left, K_right, F0_left, F0_right, noise_values, iterations)
    print("Noise sweep completed.")

    # Extract arrays
    noise_array = np.array([r[0] for r in results])
    left_wins_array = np.array([r[1] for r in results])
    right_wins_array = np.array([r[2] for r in results])
    ties_array = np.array([r[3] for r in results])

    total = left_wins_array + right_wins_array + ties_array
    order_parameter = (left_wins_array - right_wins_array) / total

    np.savez("tug_of_war_noise_sweep_wl_05_wr_03.npz",
             noise=noise_array,
             left_wins=left_wins_array,
             right_wins=right_wins_array,
             ties=ties_array,
             order_parameter=order_parameter,
             omega_left=omega_left,
             omega_right=omega_right,
             K_left=K_left,
             K_right=K_right,
             F0_left=F0_left,
             F0_right=F0_right,
             iterations=iterations)

