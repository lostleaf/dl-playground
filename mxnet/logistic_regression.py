import random
from mxnet import ndarray as nd
from mxnet import autograd as ag
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder


BATCH_SIZE = 40
EPOCHS = 5000
RATE = 0.1


def make_data():
    one_hot = OneHotEncoder()
    iris_data = datasets.load_iris()
    X = nd.array(iris_data.data)
    y = nd.array(one_hot.fit_transform(iris_data.target.reshape(-1, 1)).toarray())
    return X, y, nd.array(iris_data.target)


def forward(X, W, b):
    e = nd.exp(nd.dot(X, W) + b)
    s = nd.sum(e, axis=1).reshape((-1, 1))
    return e / s


def loss(y, y_pred):
    return -nd.sum(y * nd.log(y_pred), axis=1)


def predict(X, W, b):
    prob = forward(X, W, b)
    return nd.argmax(prob, axis=1)


def train(X, y, y_origin):
    W = 0.01 * nd.random_normal(shape=(X.shape[1], y.shape[1]))
    b = 0.01 * nd.random_normal(shape=(y.shape[1],))
    W.attach_grad()
    b.attach_grad()
    for i in xrange(EPOCHS):
        print 'Epoch %d' % i
        with ag.record():
            y_pred = forward(X, W, b)
            l = loss(y, y_pred) / BATCH_SIZE
        l.backward()
        W -= RATE * W.grad
        b -= RATE * b.grad
        y_pred = predict(X, W, b)
        acc = nd.mean(y_pred == y_origin).asscalar()
        print 'Avg loss %f, Avg acc %f' % (nd.mean(l).asscalar(), acc)
    print W, b


def main():
    X, y, y_origin = make_data()
    train(X, y, y_origin)


if __name__ == "__main__":
    main()
