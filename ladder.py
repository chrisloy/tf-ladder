import tensorflow as tf
from tensorflow.python import control_flow_ops
from tensorflow.python.framework.ops import op_scope


eps = 1e-3


def flatten(l):
    return sum(map(lambda i: flatten(i) if type(i) == list else [i], l), [])


def param_norm(shape, name):
    return tf.Variable(tf.truncated_normal(flatten(shape), 0.1, seed=1234), dtype="float32", name=name)


def param_zeros(shape, name):
    return tf.Variable(tf.zeros(flatten(shape)), dtype="float32", name=name)


def param(shape, init, name=None, trainable=True):
    return tf.Variable(tf.ones(flatten(shape)) * init, dtype="float32", name=name, trainable=trainable)


def batch_norm_wrapper(z, is_training, decay=0.999):

    gamma = param([z.get_shape()[-1]], 1)
    beta = param([z.get_shape()[-1]], 0)
    pop_mean = param([z.get_shape()[-1]], 0, trainable=False)
    pop_var = param([z.get_shape()[-1]], 1, trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(z, [0])
        train_mean = tf.assign(pop_mean, tf.mul(pop_mean, decay) + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, tf.mul(pop_var, decay) + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(z, batch_mean, batch_var, beta, gamma, eps)
    else:
        return tf.nn.batch_normalization(z, pop_mean, pop_var, beta, gamma, eps)


def ladder_network(x, layers, noise, training, denoising_cost):

    def batch_norm(z, batch_mean, batch_var, gamma, beta, include_noise):
        with op_scope([z, batch_mean, batch_var, gamma, beta], None, "batchnorm"):
            z_out = (z - batch_mean) / tf.sqrt(tf.add(batch_var, eps))
            if include_noise:
                z_out = add_noise(z_out, noise)
            z_fixed = tf.mul(gamma, z_out) + beta
            return z_fixed, z_out

    def batch_norm_and_noise(z, is_training, include_noise, decay=0.99999):

        gamma = param([z.get_shape()[-1]], 1)
        beta = param([z.get_shape()[-1]], 0)
        pop_mean = param([z.get_shape()[-1]], 0, trainable=False)
        pop_var = param([z.get_shape()[-1]], 1, trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(z, [0])
            train_mean = tf.assign(pop_mean, tf.mul(pop_mean, decay) + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, tf.mul(pop_var, decay) + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return batch_norm(z, batch_mean, batch_var, gamma, beta, include_noise)
        else:
            return batch_norm(z, pop_mean, pop_var, gamma, beta, include_noise)

    h_clean = x
    h_corr = x

    print "Graph shape: %s" % " --> ".join(map(str, layers))

    z_corrs = [h_corr]
    z_cleans = [h_clean]

    h_cleans = []

    # Encoders
    for i, nodes in enumerate(layers[1:]):

        w = param_norm([layers[i], nodes], "W%d" % i)
        z_clean = tf.matmul(h_clean, w)
        z_corr = tf.matmul(h_corr, w)

        h_clean, z_clean = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_clean, True, False),
            lambda: batch_norm_and_noise(z_clean, False, False)
        )

        h_corr, z_corr = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_corr, True, True),
            lambda: batch_norm_and_noise(z_corr, False, True)
        )

        z_cleans.append(z_clean)
        z_corrs.append(z_corr)

        if i + 2 < len(layers):
            h_cleans.append(h_clean)
            h_clean = tf.nn.relu(h_clean)
            h_corr = tf.nn.relu(h_corr)

    z_dec = h_corr
    reverse_layers = layers[::-1]
    dec_cost = []

    # Decoder
    for j, nodes in enumerate(reverse_layers):

        i = len(layers) - (j + 1)

        if j != 0:
            v = param_norm([layers[i+1], nodes], "V%d" % i)
            z_dec = tf.matmul(z_dec, v)

        _, z_dec = control_flow_ops.cond(
            training,
            lambda: batch_norm_and_noise(z_dec, True, False),
            lambda: batch_norm_and_noise(z_dec, False, False)
        )

        z_corr = z_corrs[i]
        z_clean = z_cleans[i]

        z_dec = combinator(z_dec, z_corr, nodes)

        cost = tf.reduce_mean(tf.reduce_sum(tf.square(z_dec - z_clean), 1)) / nodes
        dec_cost.append((cost * denoising_cost[i]))

    y_clean = h_clean
    y_corr = h_corr
    u_cost = tf.add_n(dec_cost)

    return y_clean, y_corr, u_cost, h_cleans


def add_noise(x, noise_var):
    return x + tf.random_normal(tf.shape(x)) * noise_var


def combinator(z_est, z_corr, size):

    a1 = param([size], 0., name='a1')
    a2 = param([size], 1., name='a2')
    a3 = param([size], 0., name='a3')
    a4 = param([size], 0., name='a4')
    a5 = param([size], 0., name='a5')
    a6 = param([size], 0., name='a6')
    a7 = param([size], 1., name='a7')
    a8 = param([size], 0., name='a8')
    a9 = param([size], 0., name='a9')
    a10 = param([size], 0., name='a10')

    mu = tf.mul(a1, tf.sigmoid(tf.mul(a2, z_corr) + a3)) + tf.mul(a4, z_corr) + a5
    va = tf.mul(a6, tf.sigmoid(tf.mul(a7, z_corr) + a8)) + tf.mul(a9, z_corr) + a10

    return (z_est - mu) * va + mu


class Model:
    def __init__(self, x, y, y_gold, loss, train_step, i_state=None, pre_loss=None, pre_train=None, training=None,
                 u_train_step=None):
        self.x = x
        self.y = y
        self.y_gold = y_gold
        self.loss = loss
        self.train_step = train_step
        self.report_name = "ERROR"
        self.report_target = loss
        self.i_state = i_state
        self.pre_loss = pre_loss
        self.pre_train = pre_train
        self.training = training
        self.u_train_step = u_train_step

    def set_report(self, name, target):
        self.report_name = name
        self.report_target = target

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_gold, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def dev_labelled_feed(self, d):
        l = d.x_test.shape[0] / 2
        if self.training is not None:
            return {self.x: d.x_train[0:l, :], self.y_gold: d.y_train[0:l, :], self.training: False}
        else:
            return {self.x: d.x_train[0:l, :], self.y_gold: d.y_train[0:l, :]}

    def test_labelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_test, self.y_gold: d.y_test, self.training: False}
        else:
            return {self.x: d.x_test, self.y_gold: d.y_test}

    def train_unlabelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_train, self.training: False}
        else:
            return {self.x: d.x_train}

    def test_unlabelled_feed(self, d):
        if self.training is not None:
            return {self.x: d.x_test, self.training: False}
        else:
            return {self.x: d.x_test}

    def train_batch_feed(self, d, lower, upper):
        if self.training is not None:
            return {self.x: d.x_train[lower:upper], self.y_gold: d.y_train[lower:upper], self.training: True}
        else:
            return {self.x: d.x_train[lower:upper], self.y_gold: d.y_train[lower:upper]}


def train(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def y_and_loss(logits, y_gold, one_hot=False):
    if one_hot:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.softmax(logits, name="y")
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, y_gold))
        y = tf.nn.sigmoid(logits, name="y")
    return y, loss


def ladder_model(
        features,
        output,
        learning_rate,
        hidden_nodes,
        noise_var,
        noise_costs):

    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, shape=[None, features], name="x")
    y_gold = tf.placeholder(tf.float32, shape=[None, output], name="y_gold")
    training = tf.placeholder(tf.bool, name="training")

    layers = [features] + hidden_nodes + [output]
    y_clean, y_corr, u_cost, _ = ladder_network(x, layers, noise_var, training, denoising_cost=noise_costs)
    _, s_cost = y_and_loss(y_corr, y_gold)
    y, error = y_and_loss(y_clean, y_gold)
    loss = s_cost + u_cost
    train_step = train(loss, learning_rate)
    u_train_step = train(u_cost, learning_rate)

    m = Model(x, y, y_gold, loss, train_step, training=training, u_train_step=u_train_step)
    m.set_report("ERROR", error)
    return m
