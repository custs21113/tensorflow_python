#coding:utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE=30
seed=2
#基于seed生成随机数
rdm=np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标点(x0,x1)作为输入数据集
X=rdm.randn(300,2)
#从x这个300行2列的矩阵中取出一行，判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
#作为输入数据集的标签（正确答案）
Y_=[int(x0*x0+x1*x1<2)for(x0,x1)in X]
#遍历Y中的每个元素，1赋值‘red’,其余赋值‘blue’，这样可视化显示时人可以直观区分
Y_c=[['red'if y else 'bule'] for y in Y_]
#对数据集X和标签Y进行shape整理，第一个元素为-1表示，随第二个参数计算得到，第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
X=np.vstack(X).reshape(-1,2)
Y_=np.vstack(Y_).reshape(-1,1)
print(X)
print(Y_)
print(Y_c)
#用plt.scatter画出数据集X各行中第0列元素和第1列元素的点即各行的（x0,x1),用各行Y_c对应的值表示颜色。（c是color的缩写）
plt.scatter(X[:0],X[:1],c=np.squeeze(Y_c))

#定义神经网络的输入，参数和输出，定义前向出播过程
def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b;
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32.shape(None,1))

w1-get_weight([2,11],0.01
              )