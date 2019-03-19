#coding:utf-8
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
SEED=23455
COST=1
PROFIT=9

rdm=np.random.RandomState(SEED)
X=rdm.rand(32,2)
Y=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

#2定义损失函数及反向传播方法。
#定义损失函数为MSE，反向传播方法为梯度下降。
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=20000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=(i*BATCH_SIZE)%32+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i% 500==0:
            print("After %d training step, w1 is : "%(i))
            print(sess.run(w1))
    print("Final w1 is: \n",sess.run(w1))

