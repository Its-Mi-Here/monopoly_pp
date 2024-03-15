import numpy as np
import pandas as pd


class MonopolyEnv:
    def __init__(self, num_states, max_turns=100):
        self.actions = ["skip", "buy", "give"]
        self.x = 0
        self.done = False
        self.episode_length = 0
        self.no_operation = False
        
        self.num_states = num_states
        self.board = np.zeros(self.num_states)
        self.state_observation = [self.x, self.board]
        self.max_turns = max_turns
        
    def reset(self):
        self.done = False
        self.episode_length = 0
        self.x = 0
        self.board = np.zeros(self.num_states)
        self.state_observation = [self.x, self.board]
        return [self.x, self.board]
    
    def action_space(self):
        return self.actions

    def update_position_roll(self):
        # Update the game state based on the action
        # roll = np.random.randint(low=1, high=5)
        roll = np.random.randint(low=1, high=3)
        self.x = (self.x + roll) % self.num_states
        return [self.x, self.board]
  
    def step(self, action):
        
        if self.episode_length > self.max_turns:
            self.done = True
            self.no_operation = True
            return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length
            
        if np.all(self.board == 2):
            self.done = True
            self.no_operation = True
            return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length

        self.action = action
        self.reward = self.get_reward()
        self.state_observation = self.take_action()
        self.episode_length += 1
        self.no_operation = False
        
        if(self.episode_length >= self.max_turns):
            self.done = True
        
        return self.state_observation, self.reward, self.done, self.no_operation, self.episode_length
    
    def get_reward(self):
        '''
        Return value : rewards
        Input argument. 
        '''
        self.reward = 0
        if self.board[self.x] == 0:
            if self.action == "buy":
                self.reward += 0
            elif self.action == "give":
                # Invalid action
                self.reward += -10
            else:
                # Skipping even when it can buy
                self.reward += -2

        elif self.board[self.x] == 1:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Good action
                self.reward += 0
            else:
                # Skipping even when it can sell
                self.reward += -2

        else:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Invalid action
                self.reward += -1
            else:
                # Skipping correct action
                self.reward += 0


        if np.all(self.board == 2):
            self.reward += 3
        else:
            self.reward += -1
    
        return self.reward
    
    def take_action(self):
        if self.action == "buy":
            if int(self.board[self.x]) == 0:        
                self.board[self.x] = 1
                
        elif self.action == "give":
            if int(self.board[self.x]) == 1:
                self.board[self.x] = 2
                
        return [self.x, self.board]




'''Functions for the agent'''
def best_state_action_value(current_state):
    max_val = np.inf*-1
#     from IPython.core.debugger import Tracer; Tracer()() 
    for key in current_state.keys():
        if current_state[key] > max_val:
            max_val = current_state[key]
            best_action = key
    return best_action, max_val

# def current_state_to_string(state):
#     current_state = ''.join(str(int(e)) for e in state)
#     return current_state
# self.x self.board
def current_state_to_string(state):
    x = state[0]
    board = state[1]
    # current_state = ''.join(x).join(board[x])
    current_state = f"{int(x)}{int(board[x])}"
    
    # current_state = ''.join(str(int(e)) for e in state)
    return current_state


def get_all_states_as_strings(num_states, num_agents):
    states = []
    for i in range(num_states):
        for j in range(num_agents+1):
            tmp = [i,j]
            states.append("".join(str(a) for a in tmp))
    # print(states)
    return states

def initialize_Q(num_states, num_agents):
    Q = {}
    states = get_all_states_as_strings(num_states, num_agents)
    for state in states:
        Q[state] = {}
        for i in range(3): #Number of actions = 3
            # Q[state][i] = np.random.uniform(-2,2,1)
            Q[state][i] = -10
    return Q



if __name__ == "__main__":
    # MAX_STATES = MAX_HOR_LENGTH*MAX_HOR_LENGTH
    num_states = 10
    num_agents = 2

    TOTAL_EPISODES = 1000
    SIM_RUN = 10
    SHOW_EVERY = 10
    OBSERVATION_SPACE = 2
    # LEARNING_RATE = 0.05 # alpha in the literature
    LEARNING_RATE = 0.05 # alpha in the literature
    DISCOUNT = 0.95 # gamma IN the literature
    EPSILON = 0.1
    START_EPSILON_DECAYING = 150
    END_EPSILON_DECAYING = 600
    epsilon_decay_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    debug = False
    Summed_reward = []


    for sim in range(SIM_RUN):
        EPSILON = 0.1 
        done = False
        
        agent = MonopolyEnv(num_states, 200)
        Q_table = initialize_Q(num_states, num_agents)

        Total_running_reward = []
        action_space = agent.action_space()
        action_indexes = [i for i in range(len(action_space))]
        cnt = 0
        no_op = False
        episode_lengths = []
        for episode in range(TOTAL_EPISODES):
            done = False
            current_state = agent.reset()
            cnt += 1
            total_episode_reward = 0
            episode_length = 0

            while not done:
                current_state = agent.update_position_roll()
                if debug:
                    print(f"Agent.x: {current_state[0]}")
                
                current_state_str = current_state_to_string(current_state)
                kind_of_selection_ = 'None'

                if np.random.uniform() > EPSILON:
                    action, max_qt1 = best_state_action_value(Q_table[current_state_str])
                    kind_of_selection_ = 'Greedy'
                else:
                    action = np.random.choice(action_indexes)
                    max_qt1 = Q_table[current_state_str][action]
                    kind_of_selection_ = 'Random'

                if debug:
                    print(f"action: {action_space[action]}, current_state: {current_state_str}")
                    print(f"Before action: {current_state}")
                
                next_state, reward, done, no_op, episode_length = agent.step(action_space[action])
                total_episode_reward += reward
                
                if debug:
                    print(f"After action: {next_state}")
                    print(f"reward: {reward}")
                    print(f"---------------------------------------")
                    
                Q_table[current_state_str][action] += LEARNING_RATE*(reward + DISCOUNT*max_qt1 - Q_table[current_state_str][action])
    #             print(f'current state : {current_state}. Action : {action_space[action]}. Next state: {next_state}. Kind of Sel: {kind_of_selection_}')
                current_state = next_state
                # print(current_state)
                cnt+=1

            Total_running_reward.append(total_episode_reward)
            episode_lengths.append(episode_length)
            
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                EPSILON -= epsilon_decay_value
        
            # print(f"---------------------------------------")
        
        if sim == 0:
            Summed_reward = Total_running_reward
        else: 
            Summed_reward = np.vstack((Summed_reward,Total_running_reward))
            
        if sim % SHOW_EVERY == 0:
            print(sim)

    # Displaying average reward

    df = pd.DataFrame(Summed_reward)
    Mean_total_reward = df.mean()
    Mean_total_reward
    # print('--------------------------')
    # print(f'Numer of steps per episode : {episode_length}. Reward : {Total_running_reward}')
    import matplotlib.pyplot as plt
    plt.plot(Mean_total_reward)
    plt.grid()
    plt.title('Mean reward after 10 simulation of 1000 Episode each')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards / Costs')
    plt.show()

    # Moving average
    N = 10
    cumsum, moving_aves = [0], []

    for i, x in enumerate(Total_running_reward, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    plt.plot(moving_aves)
    plt.title('Moving Average')
    plt.show()

    plt.plot(episode_lengths)