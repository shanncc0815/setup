import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument('iter', type=int, default=200, help='Number of iteration to benchmark')
args = parser.parse_args()

n = 8192
dtype = tf.float32
with tf.device("/gpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)


# avoid optimizing away redundant nodes
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
iters = args.iter

# pre-warming
for i in range(10):
  sess.run(product.op)

start = time.time()
for i in range(iters):
  sess.run(product.op)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**9
print('\n')
print(' took: %.2f sec for %d iteration' % (elapsed, iters))
print(' %d x %d matmul took: %.2f sec, %.2f G ops/sec' % (n, n, elapsed/iters, rate,))
