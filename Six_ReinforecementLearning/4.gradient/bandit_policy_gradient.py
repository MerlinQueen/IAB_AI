# %%
import tensorflow as tf 
import numpy as np 


# %%
#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)
def pullBandit(bandit):
    # 获取一个随机数
    result = np.random.randn(1)
    if result > bandit:
        # 返回奖励
        return 1 
    else:
        return -1


# %%
tf.reset_default_graph()
# 这两条线建立了网络的前馈部分。这做了实际的选择
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)
# 他在接下来的6个项目中建立了培训程序。我们将奖励和选择的行动馈送到网络中
# 计算损失，并使用它更新网络
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)


# %%
# 把一批事件集用来训练智能体
total_episodes =1000
# 记分牌设置为0。
total_reward = np.zeros(num_bandits)
# 设置采取随机动作的概率
e =0.1
init = tf.initialize_all_variables()
# 激活tf
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i <total_episodes:
        # 选择随机动作或者从网络中选择
        if np.random.rand(1)<e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])  #获取回报
        # 更新网络
        _,resp,ww = sess.run([update,responsible_weight,weights],feed_dict={reward_holder:[reward],action_holder:[action]}) 

        # 更新我们的分数
        total_reward[action] += reward  
        if  i%50 == 0:
            print('running reward for the'+str(num_bandits)+'bandits'+str(total_reward))    
        i+=1
print("the agent thinks bandit"+str(np.argmax(ww)+1)+"is the most promising")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("right")
else:
    print("wrong")


# %%
