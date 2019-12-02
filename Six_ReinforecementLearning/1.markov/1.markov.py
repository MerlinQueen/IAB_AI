# %%
import gym
env = gym.make('CartPole-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # print('reward{}'.format(reward))
        action = env.action_space.sample()
        observation,reward,done,info = env.step(action)
        if  done:
            print("episode finished after {} timesteps".format(t+1))
            break
env.close()


# %%
