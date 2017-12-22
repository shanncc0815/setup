import tensorflow as tf

print('\n\n')
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

print('\n\n')
print(sess.run(hello))

print('\n\n')
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))

