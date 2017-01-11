import gym
import tensorflow as tf
import nn
layersarray=[4,2]
inputsize=4
nnlayers=nn.create(layersarray,inputsize)
sess=tf.Session()
#input=tf.placeholder('float', shape=[inputsize])
#sess.run(tf.global_variables_initializer(),feed_dict={input:[0,1,2,3]})
#outputs=nn.input(nnlayers,input)
sess.run(tf.global_variables_initializer())
env = gym.make('CartPole-v0')
observation = env.reset()
tscore=0
actionspertrain=20
while 1:
    score=[]
    for _ in xrange(20):
        obs=[]
        for o in observation:
            obs.append(o)
        print obs
        actarray=nn.input(nnlayers,obs)
        if tf.argmax(actarray,axis=0)==0:
            action=0
        else:
            action=1
        print 'action='+str(action)
        observation, reward, done, info = env.step(action)
        score.append(1/(reward+1))
        tscore=tscore+reward
        if done:
            env.reset()
            tscore=0
    if tscore>=200:
        print 'tscore over thereshold, tscore='+str(tscore)
        break
    nnlayers=nn.train(nnlayers,score,sess,actionspertrain)


print 'done'
