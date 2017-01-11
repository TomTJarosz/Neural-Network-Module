import tensorflow as tf
import gym
import numpy
#inputs: layerarray- an array where layerarray[0] is the number of nodes in the first hidden layer
#inputsize- number of input nondes
#outputs:an array where layers[0] is a list of all the first level weights and biases
def create(layersarray,inputsize):
    layers=[]
    prevlayersize=inputsize
    for layersize in layersarray:
        layers.append({'weights': tf.Variable(tf.random_normal([prevlayersize,layersize])),
                      'biases':tf.Variable(tf.random_normal([layersize]))})
        prelayersize=layersize
    return layers
#inputs:nnet- a neural net array as returned from create or train
#inputarray- an array of the inputs for the neural nets
def input(nnet,inputarray):
#    prevlayer=[inputarray.astype(numpy.float32)]
    prevlayer=[inputarray]
    for layer in nnet:
        prevlayer = tf.add(tf.matmul(prevlayer,layer['weights']),layer['biases'])
        prevlayer = tf.nn.relu(prevlayer)
    return prevlayer

def train(nnlayers,score,sess,apt):
    sessobs=[]
    sessacts=[]
    for i in xrange(apt):
        sessobs.append(sess.observation)

    
    
    
    
#    cost=tf.placeholder('float', shape=[len(score)])
#    optimizer = tf.train.AdamOptimizer().minimize(cost)
#    _, c = sess.run([optimizer, cost], feed_dict={cost:score} )
#    print 'cost='+str(c)
#
