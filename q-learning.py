import numpy as np
import gym
import time

from gym.envs.registration import register


try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78 #reward task yang telah diselesaikan
    )
except:
    print('Sudah ada id yang teregister')

env = gym.make('FrozenLakeNotSlippery-v0')
env.reset()

for step in range(15):
    env.render() #renderscreen
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    time.sleep(0.5)
    if done:
        env.reset()
env.close()

## rows--States coloumns-Actions
action_size = env.action_space.n #4
state_size = env.observation_space.n # 16 observasi diindex dari 0 atau disebut state

q_table = np.zeros([state_size, action_size])
print(q_table)

EPOCHS = 20000 #agent belajar sendiri
ALPHA = 0.8 #learning rate
GAMMA = 0.99 #discount rate

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

#pada permainan pertama akan mengandalkan eksplorasi dikarenakan epsilon nilai =1
#epsilon greedy algorithm
def epsilon_greedy_action_selection(epsilon,q_table,discrete_state):
    #random number dengan ,1 dibelakang
    random_number = np.random.random()

    #exploitasi (memilih aksi yang memaksimalkan nilai q)
    if random_number > epsilon:
        state_row = q_table[discrete_state,:]
        action = np.argmax(state_row) #mengambil posisi index yang memiliki nilai maximum

    #explorasi  (memilih random aksi)
    else:
        action = env.action_space.sample()

    return action

def compute_next_q_value(old_q_value,reward,next_optimal_q_value):

    #rumus qlearning
    return old_q_value + ALPHA * (reward + GAMMA*next_optimal_q_value - old_q_value)

def reduce_epsilon(epsilon,epoch):

    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)


# inisialisasi rewards
rewards = []
log_interval = 1000

# Play 20k games
for episode in range(EPOCHS):
    # Reset the environment
    state = env.reset()
    done = False
    total_rewards = 0

    while not done:
        action = epsilon_greedy_action_selection(epsilon, q_table, state)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Look up current/old qtable value Q(st,at)
        old_q_value = q_table[state, action]

        # Get the next optimal Q-Value
        next_optimal_q_value = np.max(q_table[new_state, :])

        # Compute next q value
        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

        # Update Q Table
        q_table[state, action] = next_q

        total_rewards = total_rewards + reward

        # Our new state is state
        state = new_state

    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = reduce_epsilon(epsilon, episode)
    rewards.append(total_rewards)


    if episode %  log_interval == 0: #menang game dalam 1000 pertandingan
        print(np.sum(rewards))
        #setelah itu ditambah total dari kemenangan dari game tersebut

env.close()

print(q_table)

state = env.reset()

for step in range(10):
    env.render()
    action = np.argmax(q_table[state,:])
    state,reward,done,info = env.step(action)

    time.sleep(1)

    if done:
        break

env.close()