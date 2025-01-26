import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from av.deprecation import method
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import ListedColormap, BoundaryNorm

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=2)

def get_data_single_run(data):
    data = np.load('tug_of_war.npz')
    times = data['times']
    position = data['q_history']
    agent_forces = data['agent_forces']
    agent_phases = data['agent_phases']

    team_labels = data['team_labels']
    agent_indices = data['agent_indices']
    left_team_indices = data['left_team_indices']
    right_team_indices = data['right_team_indices']

    left_team_mask = (team_labels == -1)
    right_team_mask = (team_labels == 1)
    left_agent_forces = agent_forces[:, left_team_mask]
    right_agent_forces = agent_forces[:, right_team_mask]
    left_agent_phases = agent_phases[:, left_team_mask]
    right_agent_phases = agent_phases[:, right_team_mask]
    return times, position, left_agent_forces, right_agent_forces, left_agent_phases, right_agent_phases


def plot_rope_center(times, position):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.fill_between(times, 0, 0.5, color='tab:blue', alpha=0.4)
    ax.fill_between(times, 0, -0.5, color='tab:red', alpha=0.4)
    ax.axhline(0, color='tab:gray', linestyle='-', linewidth=2)
    center = np.mean(position, axis=1)
    ax.plot(times, savgol_filter(center, 101, 3), linewidth=2, color='black')
    ax.set(xlabel='Simulation time', ylabel='Rope center of mass', ylim=(-0.5, 0.5), xlim=(-1, 300))
    ax.set(xticks=[0, 100, 200, 300])
    plt.tight_layout()
    plt.show()


def plot_total_rope_length(times, position):
    rope_length = np.abs(np.max(position, axis=1) - np.min(position, axis=1))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axhline(2, color='tab:gray', linestyle='-', linewidth=2)
    ax.plot(times, savgol_filter(rope_length, 101, 3), linewidth=2, color='black')
    ax.set(xlabel='Simulation time', ylabel='Total rope length', xlim=(-1, 300), ylim=(1, 3))
    ax.set(xticks=[0, 100, 200, 300])
    plt.tight_layout()
    plt.show()

def plot_kuramoto_order_parameter(times, left_agent_phases, right_agent_phases):
    def tent_weight(phi):
        w = np.zeros_like(phi)
        mask1 = (phi > 0) & (phi <= np.pi / 2)
        w[mask1] = phi[mask1] / (np.pi / 2)
        mask2 = (phi > np.pi / 2) & (phi < np.pi)
        w[mask2] = 2 - (phi[mask2] / (np.pi / 2))
        return w

    T = len(times)
    left_op = np.zeros(T)
    right_op = np.zeros(T)

    for i in range(T):
        left_phi = left_agent_phases[i]
        right_phi = right_agent_phases[i]

        left_w = tent_weight(left_phi)
        right_w = tent_weight(right_phi)

        left_den = np.sum(left_w)
        if left_den > 0:
            left_num = np.sum(left_w * np.exp(1j * left_phi))
            left_op[i] = np.abs(left_num / left_den)
        else:
            left_op[i] = 0.0

        right_den = np.sum(right_w)
        if right_den > 0:
            right_num = np.sum(right_w * np.exp(1j * right_phi))
            right_op[i] = np.abs(right_num / right_den)
        else:
            right_op[i] = 0.0

    # left_op, right_op = compute_tent_weighted_order_parameter(times, left_agent_phases, right_agent_phases)
    plt.plot(times, left_op, label='Left (tent-weighted)')
    plt.plot(times, right_op, label='Right (tent-weighted)')
    plt.xlabel('Time')
    plt.ylabel('Weighted Order Parameter')
    # plt.ylim(0,1)
    plt.xlim(0,50)
    plt.legend()
    plt.show()
    # return left_op, right_op

def plot_total_force(times, left_agent_forces, right_agent_forces):
    total_left_force = np.abs(np.sum(left_agent_forces, axis=1))
    # total_left_force = savgol_filter(total_left_force, 11, 3)
    total_right_force = np.abs(np.sum(right_agent_forces, axis=1))
    # total_right_force = savgol_filter(total_right_force, 11, 3)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(times, total_left_force, linewidth=2, color='tab:red', label='Left Team')
    ax.plot(times, total_right_force, linewidth=2, color='tab:blue', label='Right Team')
    ax.set(xlabel=('Simulation time'), ylabel=('Force (arb.)'), xlim=(-1, 50), ylim=(-10, 50),
           xticks=[0, 10, 20, 30, 40, 50])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_contest_heatmap(contest):
    data = np.load(contest)
    left_wins = data['left_wins']
    right_wins = data['right_wins']
    ties = data['ties']
    wK_values = data['wK_values']

    total = left_wins + right_wins + ties
    score_diff = (left_wins - right_wins) / total

    plt.figure(figsize=(4, 3))
    plt.imshow(
        score_diff,
        interpolation='bicubic',
        origin='lower',
        extent=[wK_values[0], wK_values[-1], wK_values[0], wK_values[-1]],
        cmap='Spectral_r',
        vmax=1,
        vmin=-1
    )
    plt.colorbar(label='Normalized score')
    plt.xlabel(r'$\left(\omega/\kappa\right)_{left}$')
    plt.ylabel(r'$\left(\omega/\kappa\right)_{right}$')
    plt.tight_layout()
    plt.show()


def order_noise(contest_noise_files):
    colors = ['black', 'tab:red', 'tab:blue']
    labels = ['eq', 'l dominate', 'r dominate']

    plt.figure(figsize=(4, 3))

    for contest_noise, color, label in zip(contest_noise_files, colors, labels):
        data = np.load(contest_noise)
        left_wins = data['left_wins']
        right_wins = data['right_wins']
        ties = data['ties']
        noise = data['noise']
        total = left_wins + right_wins + ties
        score_diff = (left_wins - right_wins) / total

        score_diff = savgol_filter(score_diff, 11, 3)
        plt.plot(noise, score_diff, '-', color=color, linewidth=2, label=label)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlim(0, 1.0)
    plt.ylim(-1, 1)
    plt.ylabel(r'Normalized score')
    plt.xlabel(r'$\xi$')
    plt.tight_layout()
    plt.show()


def plot_diagonal(contest):
    data = np.load(contest)
    left_wins = data['left_wins']
    right_wins = data['right_wins']
    ties = data['ties']
    wK_values = data['wK_values']

    mid_index = len(wK_values) // 2
    wK_right_fixed = wK_values[mid_index]
    lw_line = left_wins[mid_index, :]
    rw_line = right_wins[mid_index, :]
    tie_line = ties[mid_index, :]

    total = lw_line + rw_line + tie_line
    score_diff_line = (lw_line - rw_line) / total

    plt.figure(figsize=(4, 3))
    plt.plot(wK_values, score_diff_line, '-')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(r'$\left(\omega/\kappa\right)_{left}$')
    plt.ylabel('Normalized score')
    plt.tight_layout()
    plt.show()


def plot_bifurcation_diagram(contest):
    data = np.load(contest)
    left_wins = data['left_wins']
    right_wins = data['right_wins']
    ties = data['ties']
    wK_values = data['wK_values']

    mid_index = len(wK_values) // 2
    wK_right_fixed = wK_values[mid_index]
    lw_line = left_wins[mid_index, :]
    rw_line = right_wins[mid_index, :]
    tie_line = ties[mid_index, :]

    total = lw_line + rw_line + tie_line
    score_diff_line = (lw_line - rw_line) / total

    plt.figure(figsize=(4, 3))
    plt.plot(wK_values, score_diff_line, '-')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(r'$\left(\omega/\kappa\right)_{left}$')
    plt.ylabel('Normalized score')
    plt.tight_layout()
    plt.show()

times, position, left_agent_forces, right_agent_forces, left_agent_phases, right_agent_phases = get_data_single_run('tug_of_war.npz')
# plot_diagonal('tug_of_war_contests_n_0_4_r_dominate.npz')
# plot_contest_heatmap('tug_of_war_contests_n_0_2_r_dominate.npz')
# order_noise(
#     ['tug_of_war_noise_sweep_eq.npz', 'tug_of_war_noise_sweep_wl_05_wr_03.npz', 'tug_of_war_noise_sweep_wl_03_wr_05.npz'])
# plot_bifurcation_diagram('tug_of_war_contests_n_0_02_.npz')
# plot_rope_center(times, position)
# plot_total_rope_length(times, position)
# plot_total_force(times, left_agent_forces, right_agent_forces)
plot_kuramoto_order_parameter(times, left_agent_phases, right_agent_phases)
