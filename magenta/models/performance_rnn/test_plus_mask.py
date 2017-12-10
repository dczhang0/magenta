import tensorflow as tf

foo = tf.constant([[1.0,2], [4,5], [3, 1]])
# tf.contrib.util.constant_value(foo), show values in command lines
# tt = tf.reshape(foo, [3,2])

tt = tf.argmax(foo, axis=1)
# tt_z = tf.reshape(tt, [tt.shape.as_list()[0]]+[1])
# [tt.shape.as_list()[0]] if output none, which will result wrong use in reshape
tt_z = tf.reshape(tt, [-1]+[1])
# sh_m_r = tt.shape.as_list()  list
# tt_m = [foo[:, tt[i]] for i in range(tt.shape[0])]
#  or range(sh_m_r[0]), but not useful if the dimension is None
# aaaa = tf.reshape(tt_m, [3, tf.shape(tt)])
# the dimension of new shape must be integers, not tensor

tt_0 = [foo[tt[0], :]]
for i in range(tt.shape[0]-1):
    tt_i = [foo[tt[i+1], :]]
    tt_0 = tf.concat([tt_0, tt_i], 0)

tt_m = tf.gather_nd(foo, tt_z)
# foo[:, tt[1]]
# foo[:, 0:2]
# foo[:,[0,1]]  # not working

sess = tf.Session()

print(sess.run(tt))
print(sess.run(tt_0))
inputs = tf.placeholder(tf.float32, [3, 3])
aa_m = inputs + tt_0
# print(sess.run(aa_m))
