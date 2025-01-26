import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=2)

def animate_simulation(filename):
    data = np.load(filename)
    q_history = data['q_history']
    agent_phases = data['agent_phases']
    times = data['times']
    team_labels = data['team_labels']
    agent_indices = data['agent_indices']
    left_team_indices = data['left_team_indices']
    right_team_indices = data['right_team_indices']

    Nsteps, N = q_history.shape
    num_agents = len(agent_indices)
    num_left_agents = len(left_team_indices)
    num_right_agents = len(right_team_indices)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(-1.5, 1.5)
    rope_line, = ax.plot([], [], '-', linewidth=2, color='black', zorder=10)
    fixed_center = ax.vlines(0, -0.05, 0.05, linewidth=2, color='tab:gray', zorder=5)

    team_left_scatter = ax.scatter([], [], s=200, facecolors='white', edgecolors='black', zorder=10)
    team_right_scatter = ax.scatter([], [], s=200, facecolors='white', edgecolors='black', zorder=10)

    def update(timeStep):
        q = q_history[timeStep, :]
        phi = agent_phases[timeStep, :]
        current_time = times[timeStep]
        agent_positions = q[agent_indices]
        agent_positions_left = q[left_team_indices]
        agent_positions_right = q[right_team_indices]
        phases = phi % (2 * np.pi)
        pulling = (phases > 0) & (phases < np.pi)
        applying_force = pulling

        applying_force_left = applying_force[team_labels == -1]
        applying_force_right = applying_force[team_labels == 1]

        left_facecolors = np.where(applying_force_left, 'tab:red', 'white')
        right_facecolors = np.where(applying_force_right, 'tab:blue', 'white')
        team_left_scatter.set_offsets(np.c_[agent_positions_left, np.zeros(len(agent_positions_left))])
        team_right_scatter.set_offsets(np.c_[agent_positions_right, np.zeros(len(agent_positions_right))])
        team_left_scatter.set_facecolors(left_facecolors)
        team_right_scatter.set_facecolors(right_facecolors)
        rope_line.set_data(q, np.zeros_like(q))
        ax.set_title(f'Time: {current_time:.2f}s')

    anim = FuncAnimation(fig, update, frames=Nsteps, repeat=False)
    return anim

anim = animate_simulation('tug_of_war.npz')

# Save the animation as a gif
writer = PillowWriter(fps=10)
anim.save('tug_of_war.gif', writer=writer)