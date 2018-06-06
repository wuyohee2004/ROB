

import numpy as np
import pandas as pd
import time, sys

np.random.seed(2)
N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )
    #print(table)
    return table

#build_q_table(N_STATES, ACTIONS)

def choose_action(state,q_table):
	state_actions = q_table.iloc[state,:]
	if (np.random.uniform() > EPSILON) or (state_actions.all()==0):
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax()
	return action_name

def get_env_feedback(S,A):
	if A == 'right': #move right
		if S == N_STATES - 2: #terminate
			S_ = 'terminal'
			R = 1
		else:
			S_ = S + 1
			R = 0
	else: #move left
		R = 0
		if S == 0:
			S_ = S #reach the wall
		else:
			S_ = S - 1
	return S_, R

def update_env(S,episode,step_counter):
	#How environment be updated
	env_list = ['-']*(N_STATES-1) + ['T']
	if S == 'terminal':
		interaction = 'Episode %s:	total_steps = %s' % (episode+1, step_counter)
		print('\r{}'.format(interaction), end='')
		time.sleep(2)
		print('\r                             ', end='')
	else:
		env_list[S] = 'O'
		interaction = ''.join(env_list)
		print('\r{}'.format(interaction), end='')
		# sys.stdout.write(format(interaction))
		# sys.stdout.flush()
		time.sleep(FRESH_TIME)

def rl():
	# main RL loop
	q_table = build_q_table(N_STATES, ACTIONS)
	for episode in range(MAX_EPISODES):
		step_counter = 0 
		S = 0 
		is_terminated = False
		update_env(S, episode, step_counter)
		while not is_terminated:
			A = choose_action(S, q_table)
			S_, R = get_env_feedback(S, A) # take action and get next state
			q_predict = q_table.ix[S,A]
			if S_ != 'terminal':
				q_target = R + GAMMA * q_table.iloc[S_,:].max() # next state is not terminal
			else:
				q_target = R #next state is terminal
				is_terminated = True #terminate this episode
			q_table.ix[S,A] += ALPHA * (q_target - q_predict) #update
			S = S_ #move to next state
			update_env(S, episode, step_counter+1)
			step_counter += 1
	return q_table

if __name__ == "__main__":
	q_table = rl()
	print('\r\nQ-table:\n')
	print(q_table)

