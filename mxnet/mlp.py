import matplotlib
import mxnet as mx
import time
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon

matplotlib.use('qt5agg')

BATCH_SIZE = 256
N_INPUT = 28 * 28
N_HIDDEN = 256
N_OUTPUT = 10
N_EPOCH = 50
LR = .1
try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


def get_data():
    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
    return mnist_train, mnist_test


def relu(X):
    return nd.maximum(X, 0)


def net(X, params):
    X = X.reshape((-1, N_INPUT))
    h1 = relu(nd.dot(X, params['W1']) + params['b1'])
    output = nd.dot(h1, params['W2']) + params['b2']
    return output


def test_acc(test_data, params):
    n_total = 0
    n_correct = 0
    for data, label in test_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        pred = net(data, params).argmax(axis=1)
        compare = (pred == label)
        n_total += compare.shape[0]
        n_correct += compare.sum().asscalar()
    return float(n_correct) / n_total


def main():
    mnist_train, mnist_test = get_data()
    n_train = len(mnist_train)
    train_data = gluon.data.DataLoader(mnist_train, BATCH_SIZE, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, BATCH_SIZE, shuffle=True)

    """
    Init
    """
    W1 = nd.random_normal(shape=(N_INPUT, N_HIDDEN), scale=1e-2, ctx=ctx)
    b1 = nd.zeros(N_HIDDEN, ctx=ctx)
    W2 = nd.random_normal(shape=(N_HIDDEN, N_OUTPUT), scale=1e-2, ctx=ctx)
    b2 = nd.zeros(N_OUTPUT, ctx=ctx)

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    for param in params.itervalues():
        param.attach_grad()

    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

    """
    Train
    """
    for e in xrange(N_EPOCH):
        t1 = time.time()
        print 'Epoch %d' % e
        loss_total = 0.
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with ag.record():
                output = net(data, params)
                loss = loss_func(output, label)
            loss_total += loss.sum().asscalar()
            loss.backward()
            for param in params.itervalues():
                param -= LR / BATCH_SIZE * param.grad
        print 'Avg loss %f' % (loss_total / n_train)
        print 'Avg accuracy %f' % test_acc(test_data, params)
        t2 = time.time()
        print 'Time cost %d seconds' % (t2 - t1)


if __name__ == "__main__":
    main()
