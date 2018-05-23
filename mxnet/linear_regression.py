from mxnet import ndarray as nd
from mxnet import autograd as ag
import random


N = 1000
BATCH_SIZE = 40
EPOCHS = 50
RATE = 0.01


def data_iter(X, y):
    idx = range(N)
    random.shuffle(idx)
    for i in range(0, N, BATCH_SIZE):
        k = min(i + BATCH_SIZE, N)
        j = nd.array(idx[i : k])
        yield nd.take(X, j), nd.take(y, j)


def make_data():
    W_true = nd.array([1, 2])
    b_true = 3
    X = nd.random_normal(shape=(N, 2))
    noise = nd.random_normal(shape=(N,)) * 0.01
    y = nd.dot(X, W_true) + b_true + noise
    return X, y


def forward(X, W, b):
    return nd.dot(X, W) + b


def loss(y, y_pred):
    return (y - y_pred) ** 2


def train(X, y):
    W = nd.random_normal(shape=(2, ))
    b = nd.zeros((1,))
    W.attach_grad()
    b.attach_grad()
    for i in xrange(EPOCHS):
        print 'Epoch %d' % i
        for X_batch, y_batch in data_iter(X, y):
            with ag.record():
                y_pred = forward(X_batch, W, b)
                l = loss(y_batch, y_pred) / BATCH_SIZE
            l.backward()
            W -= RATE * W.grad
            b -= RATE * b.grad
        y_pred = forward(X, W, b)
        l = loss(y_pred, y)
        print 'Avg loss %f' % nd.mean(l).asscalar()
    print W, b


def main():
    X, y = make_data()
    train(X, y)


if __name__ == "__main__":
    main()
