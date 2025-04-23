import numpy as np
import matplotlib.pyplot as plt

# Grid-world environment: 4x4 grid, goal at (3,3), obstacle at (1,1)
class GridWorld:
    def __init__(self):
        self.size = 4
        self.goal = (3, 3)
        self.obstacle = (1, 1)
        self.actions = ['up', 'down', 'left', 'right']  # 0, 1, 2, 3
        self.state = (0, 0)  # Start at (0,0)

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)
        
        next_state = (x, y)
        if next_state == self.obstacle:
            next_state = self.state  # Stay if hitting obstacle
        self.state = next_state
        
        reward = 1 if next_state == self.goal else -0.01  # Small penalty per step
        done = next_state == self.goal
        return next_state, reward, done

    def reset(self):
        self.state = (0, 0)
        return self.state

# Monte Carlo Control (Epsilon-Greedy)
def monte_carlo_control(env, episodes=1000, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size, env.size, len(env.actions)))
    returns = {(i, j, a): [] for i in range(env.size) for j in range(env.size) for a in range(len(env.actions))}
    policy = np.zeros((env.size, env.size), dtype=int)

    for _ in range(episodes):
        state = env.reset()
        episode = []  # (state, action, reward)
        done = False
        
        # Generate episode
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(len(env.actions))
            else:
                action = policy[state]
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # Update Q-values
        G = 0  # Return
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G
            if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:  # First visit
                returns[state + (action,)].append(G)
                Q[state][action] = np.mean(returns[state + (action,)])
                policy[state] = np.argmax(Q[state])

    return policy, Q

# TD(0) Learning
def td_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.size, env.size, len(env.actions)))
    policy = np.zeros((env.size, env.size), dtype=int)

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(len(env.actions))
            else:
                action = policy[state]
            next_state, reward, done = env.step(action)
            
            # TD(0) update
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) * (1 - done) - Q[state][action])
            policy[state] = np.argmax(Q[state])
            state = next_state

    return policy, Q

# Visualize policy
def plot_policy(policy, env, title):
    grid = np.zeros((env.size, env.size))
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.goal:
                grid[i, j] = 4  # Goal
            elif (i, j) == env.obstacle:
                grid[i, j] = -1  # Obstacle
            else:
                grid[i, j] = policy[i, j]

    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Action (0=Up, 1=Down, 2=Left, 3=Right)')
    plt.title(title)
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.show()

# Test the implementation
if __name__ == "__main__":
    env = GridWorld()

    # Monte Carlo Control
    policy_mc, Q_mc = monte_carlo_control(env)
    print("Monte Carlo - Q-Value (sample):")
    print(f"Start (0,0), Action 3 (Right): {Q_mc[0, 0, 3]:.2f}")
    plot_policy(policy_mc, env, "Optimal Policy (Monte Carlo Control)")

    # TD(0) Learning
    policy_td, Q_td = td_learning(env)
    print("TD(0) - Q-Value (sample):")
    print(f"Start (0,0), Action 3 (Right): {Q_td[0, 0, 3]:.2f}")
    plot_policy(policy_td, env, "Optimal Policy (TD(0) Learning)")