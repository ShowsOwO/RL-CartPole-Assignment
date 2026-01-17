import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time  # 引入 time 库用于控制演示速度


# --- 1. 定义 Q 网络 (保持不变) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- 2. 定义 DQN 智能体 (保持不变) ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory_size = 10000
        self.memory = []

        # 尝试使用 GPU，如果之前配置失败会自动回退到 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 如果是 Mac M1/M2/M3，取消下面这行的注释
        # if torch.backends.mps.is_available(): self.device = torch.device("mps")
        print(f"Agent using device: {self.device}")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-Greedy 策略
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        current_q = self.q_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# --- 3. 辅助函数 (保持不变) ---
def plot_and_save_results(scores, filename='training_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training Performance on CartPole-v1')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"图表已保存至: {filename}")


# --- 4. 主训练循环 (训练时不要开渲染，太慢) ---
def train(model_path='dqn_cartpole_model.pth'):
    # 这里 render_mode 默认为 None，速度最快
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)
    episodes = 250
    scores = []

    print("--- 开始训练 (无渲染模式) ---")
    for e in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += 1
            agent.replay()
        scores.append(score)
        if e % 10 == 0:
            agent.update_target_network()
            print(f"Episode: {e}/{episodes} | Score: {score:.1f} | Epsilon: {agent.epsilon:.2f}")
        if np.mean(scores[-10:]) > 495:
            print(f"Solved at episode {e}!")
            break

    env.close()
    plot_and_save_results(scores)
    # 保存训练好的模型权重
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"模型权重已保存至: {model_path}")


# --- 5. 【新增】可视化测试函数 ---
def test(model_path='dqn_cartpole_model.pth'):
    print("--- 开始可视化演示 ---")
    # 关键点：这里指定 render_mode="human" 会弹出窗口
    env = gym.make('CartPole-v1', render_mode="human")

    agent = DQNAgent(state_dim=4, action_dim=2)

    # 加载之前训练好保存的权重
    if os.path.exists(model_path):
        # 注意 map_location，确保在CPU机器上也能加载GPU训练出来的模型
        agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        print("成功加载模型权重！")
    else:
        print("警告：未找到模型文件，将使用未经训练的随机 Agent 演示。")

    # 关键点：测试时，把 epsilon 设为 0，让 AI 完全根据学到的经验行动，不再随机探索
    agent.epsilon = 0.0

    # 运行几个回合看看效果
    for i in range(3):  # 演示 3 轮
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            # 这里会自动渲染画面
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += 1
            # time.sleep(0.02) # 如果觉得演示速度太快，可以取消注释加一点延迟

        print(f"演示回合 {i + 1} 得分: {score}")
        time.sleep(1)  # 回合间停顿一下

    env.close()


if __name__ == "__main__":
    model_filename = 'dqn_cartpole_model.pth'

    # 1. 先运行训练
    train(model_path=model_filename)

    # 2. 训练完成后，询问是否要看演示
    user_input = input("\n训练结束。是否要加载模型并查看可视化演示？(y/n): ")
    if user_input.lower() == 'y':
        test(model_path=model_filename)
    else:
        print("程序结束。")