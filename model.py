import tensorflow as tf
from util import get_data

class OCR(object):
    def __init__(self, num_epochs, batch_size, num_filters, max_time, num_units, num_classes, learning_rate, save_path):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_filters = num_filters
        self.max_time = max_time
        self.num_units = num_units
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.save_path = save_path

    def get_batch(self, iteration, flag):
        data_set = get_data(flag)[iteration * self.batch_size, (iteration + 1) * self.batch_size]
        return data_set


    def build_model(self):
        self.inputs = tf.placeholder(tf.float32)
        self.sequence_length = tf.placeholder(tf.int32)
        self.labels = tf.sparse_placeholder(tf.int32)

        with tf.name_scope("cnn"):
            convolution_filter = tf.get_variable("convolution_filter", [3, 3, 1, self.num_filters])
            convolution_biases = tf.get_variable("convolution_biases", [self.num_filters])
            self.features = tf.nn.conv2d(self.inputs, convolution_filter)
            self.features = tf.nn.relu(self.features + convolution_biases)
            self.features = tf.nn.max_pool(self.features)
            self.features = tf.concat(self.features, 2)

        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
            self.outputs, self.state = tf.nn.dynamic_rnn(cell, self.features, self.sequence_length)
            self.outputs = tf.reshape(self.outputs, [-1, self.num_units])
            softmax_weights = tf.get_variable("softmax_weights", [self.num_units, self.num_classes])
            softmax_biases = tf.get_variable("softmax_biases", [self.num_classes])
            self.logits = tf.nn.xw_plus_b(self.outputs, softmax_weights, softmax_biases)
            self.logits = tf.reshape(self.logits, [self.batch_size, -1, self.num_classes])
            self.logits = tf.transpose(self.logits, [1, 0, 2])

    def compute_loss(self):
        self.loss = tf.nn.ctc_loss(self.labels, self.logits, self.sequence_length)
        self.loss = tf.reduce_mean(self.loss)

    def train_model(self):
        initializer = tf.global_variables_initializer()
        global_step = tf.global_variables()
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(initializer)
            for epoch in range(self.num_epochs):
                num_iteration = 10000 / self.batch_size + 1
                for iteration in range(num_iteration):
                    batch_data = self.get_batch(iteration)
                    _, loss = session.run([optimizer, self.loss], batch_data)
                    print("Epoch: %d, Iteration: %d, Loss: %.2f.", epoch, iteration, loss)
                    if loss < 0.001:
                        print("The current model parameters have met the qualification !")
                        saver.save(session, self.save_path)

    def test_model(self):
        self.decoded, self.log_probabilities = tf.nn.ctc_beam_search_decoder(self.logits, self.sequence_length)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, self.save_path)
            num_iteration = 1000 / self.batch_size + 1
            for iteration in range(num_iteration):
                batch_data = self.get_batch(iteration)
                decoded, log_probabilities = session.run([self.decoded, self.log_probabilities], batch_data)


