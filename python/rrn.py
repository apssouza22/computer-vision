import numpy as np


# source https://github.com/javaidnabi31/RNN-from-scratch/blob/master/RNN_char_text%20generator.ipynb
# good example of object oriented implementation https://github.com/JasonFengGit/RNN-Language-Classifier/blob/master/nn.py
class RNN:
	def __init__(self, hidden_size, vocab_size, seq_length, learning_rate):
		# hyper parameters
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.seq_length = seq_length
		self.learning_rate = learning_rate

		# weights for input -> header layer
		self.w_layer1 = np.random.uniform(-np.sqrt(1. / vocab_size), np.sqrt(1. / vocab_size), (hidden_size, vocab_size))
		# weights for header -> header layer
		self.w_layer3 = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (vocab_size, hidden_size))
		# weights for header -> output layer
		self.w_layer2 = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
		self.b_layer2 = np.zeros((hidden_size, 1))  # bias for hidden layer
		self.b_layer3 = np.zeros((vocab_size, 1))  # bias for output

		# memory vars for Adagrad - https://machinelearningmastery.com/gradient-descent-with-adagrad-from-scratch/
		# ignore if you implement another approach
		self.mem_w_layer1 = np.zeros_like(self.w_layer1)
		self.mem_w_layer2 = np.zeros_like(self.w_layer2)
		self.mem_w_layer3 = np.zeros_like(self.w_layer3)
		self.mem_b_layer2 = np.zeros_like(self.b_layer2)
		self.mem_b_layer3 = np.zeros_like(self.b_layer3)

	def softmax(self, x):
		p = np.exp(x - np.max(x))
		return p / np.sum(p)

	def forward(self, inputs, hprev):
		hot_encoded_chars, activations, net, prediction = {}, {}, {}, {}
		activations[-1] = np.copy(hprev)
		for t in range(len(inputs)):
			hot_encoded_chars[t] = np.zeros((self.vocab_size, 1))
			hot_encoded_chars[t][inputs[t]] = 1
			activations[t] = np.tanh(np.dot(self.w_layer1, hot_encoded_chars[t]) + np.dot(self.w_layer2, activations[t - 1]) + self.b_layer2)
			# taking the dot product between the activation and the weight matrix -- this is called the "net input" to the current layer
			net[t] = np.dot(self.w_layer3, activations[t]) + self.b_layer3  # unnormalised log probs for next char
			prediction[t] = self.softmax(net[t])
		return hot_encoded_chars, activations, prediction

	def backward(self, hot_encoded_chars, activations, predictions, targets):
		# backward pass: compute gradients going backwards
		d_w_layer1, d_w_layer2, d_w_layer3 = np.zeros_like(self.w_layer1), np.zeros_like(self.w_layer2), np.zeros_like(self.w_layer3)
		d_b_layer2, d_b_layer3 = np.zeros_like(self.b_layer2), np.zeros_like(self.b_layer3)
		dhnext = np.zeros_like(activations[0])

		for t in reversed(range(self.seq_length)):
			d_prediction = np.copy(predictions[t])
			d_prediction[targets[t]] -= 1  # backprop into y

			d_b_layer3 += d_b_layer3
			# dh includes gradient from two sides, next cell and current output
			dh = np.dot(self.w_layer3.T, d_prediction) + dhnext  # backprop into h

			derivative_dir = (1 - activations[t] * activations[t]) * dh  # derivative_dir is the term used in many equations
			d_b_layer2 += derivative_dir

			d_w_layer1 += np.dot(derivative_dir, hot_encoded_chars[t].T)
			d_w_layer2 += np.dot(derivative_dir, activations[t - 1].T)
			d_w_layer3 += np.dot(d_prediction, activations[t].T)

			# pass the gradient from next cell to the next iteration.
			dhnext = np.dot(self.w_layer2.T, derivative_dir)
		# clip to mitigate exploding gradients
		for dparam in [d_w_layer1, d_w_layer2, d_w_layer3, d_b_layer2, d_b_layer3]:
			np.clip(dparam, -5, 5, out=dparam)
		return d_w_layer1, d_w_layer2, d_w_layer3, d_b_layer2, d_b_layer3

	def loss(self, ps, targets):
		"""loss for a sequence"""
		# calculate cross-entrpy loss
		return sum(-np.log(ps[t][targets[t], 0]) for t in range(self.seq_length))

	def update_model(self, dU, dW, dV, db, dc):
		# parameter update with adagrad
		for param, dparam, mem in zip([self.w_layer1, self.w_layer2, self.w_layer3, self.b_layer2, self.b_layer3],
									  [dU, dW, dV, db, dc],
									  [self.mem_w_layer1, self.mem_w_layer2, self.mem_w_layer3, self.mem_b_layer2, self.mem_b_layer3]):
			mem += dparam * dparam
			param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

	def sample(self, h, seed_ix, n):
		"""
		sample a sequence of integers from the model
		h is memory state, seed_ix is seed letter from the first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		ixes = []
		for t in range(n):
			h = np.tanh(np.dot(self.w_layer1, x) + np.dot(self.w_layer2, h) + self.b_layer2)
			y = np.dot(self.w_layer3, h) + self.b_layer3
			p = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p=p.ravel())
			x = np.zeros((self.vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		return ixes

	def train(self, data_reader):
		iter_num = 0
		threshold = 0.01
		smooth_loss = -np.log(1.0 / data_reader.vocab_size) * self.seq_length
		while (smooth_loss > threshold):
			if data_reader.just_started():
				hprev = np.zeros((self.hidden_size, 1))
			inputs, targets = data_reader.next_batch()
			hot_encoded_chars, activations, predictions = self.forward(inputs, hprev)
			dU, dW, dV, db, dc = self.backward(hot_encoded_chars, activations, predictions, targets)
			loss = self.loss(predictions, targets)
			self.update_model(dU, dW, dV, db, dc)
			smooth_loss = smooth_loss * 0.999 + loss * 0.001
			hprev = activations[self.seq_length - 1]
			if not iter_num % 500:
				sample_ix = self.sample(hprev, inputs[0], 200)
				print(''.join(data_reader.ix_to_char[ix] for ix in sample_ix))
				print("\n\niter :%d, loss:%f" % (iter_num, smooth_loss))
			iter_num += 1

	def predict(self, data_reader, start, n):
		# initialize input vector
		x = np.zeros((self.vocab_size, 1))
		chars = [ch for ch in start]
		ixes = []
		for i in range(len(chars)):
			ix = data_reader.char_to_ix[chars[i]]
			x[ix] = 1
			ixes.append(ix)

		h = np.zeros((self.hidden_size, 1))
		# predict next n chars
		for t in range(n):
			h = np.tanh(np.dot(self.w_layer1, x) + np.dot(self.w_layer2, h) + self.b_layer2)
			y = np.dot(self.w_layer3, h) + self.b_layer3
			p = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p=p.ravel())
			x = np.zeros((self.vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		txt = ''.join(data_reader.ix_to_char[i] for i in ixes)
		return txt


# To read the training data and make a vocabulary and dictiornary to index the chars
class DataReader:
	def __init__(self, path, seq_length):
		# uncomment below , if you dont want to use any file for text reading and comment next 2 lines
		# self.data = "some really long text to test this. maybe not perfect but should get you going."
		self.fp = open(path, "r")
		self.data = self.fp.read()
		# find unique chars
		chars = list(set(self.data))
		# create dictionary mapping for each char
		self.char_to_ix = {ch: i for (i, ch) in enumerate(chars)}
		self.ix_to_char = {i: ch for (i, ch) in enumerate(chars)}
		# total data
		self.data_size = len(self.data)
		# num of unique chars
		self.vocab_size = len(chars)
		self.pointer = 0
		self.seq_length = seq_length

	def next_batch(self):
		input_start = self.pointer
		input_end = self.pointer + self.seq_length
		inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
		targets = [self.char_to_ix[ch] for ch in self.data[input_start + 1:input_end + 1]]
		self.pointer += self.seq_length
		if self.pointer + self.seq_length + 1 >= self.data_size:
			# reset pointer
			self.pointer = 0
		return inputs, targets

	def just_started(self):
		return self.pointer == 0

	def close(self):
		self.fp.close()


if __name__ == "__main__":
	seq_length = 25
	data_reader = DataReader("../texts/input.txt", seq_length)
	rnn = RNN(hidden_size=100, vocab_size=data_reader.vocab_size, seq_length=seq_length, learning_rate=1e-1)
	rnn.train(data_reader)
	rnn.predict(data_reader, 'get', 50)
