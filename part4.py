from pylab import *
import numpy as np

def set_creation(data, size):
    N =
    train_set = np.zeros((size))





def gd_vanilla(x, y, init_w, alpha, max_iter):
    EPS = 1e-5  # EPS = 10**(-5)
    prev_w = init_w - 10 * EPS
    w = init_w.copy()
    iter = 0
    succ = list()
    i_list = list()
    M = loadmat("mnist_all.mat")
    while norm(w - prev_w) > EPS and iter < max_iter:
        prev_w = w.copy()
        w -= alpha * gradient(x, y, w)
        if iter % 5 == 0:
            count = 0
            for i in range(10):
                for j in range(100):
                    test = hstack((1, M["test" + str(i)][j].T))  # originally M["train5"][148:149].T
                    if argmax(f(test, w)) == i:
                        count += 1
            succ.append(count / 10.0)
            i_list.append(iter)
        if iter % 500 == 0:
            print "Iter", iter
            print "Gradient: ", gradient(x, y, w), "\n"
        iter += 1
    plt.plot(i_list, succ)
    plt.show()
    return w
