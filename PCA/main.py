import os
import csv
import numpy as np
import matplotlib.pyplot as plt

folder = 'E:\\Code\\Python\\PCA\\PCA Project Datasets\\'
abalone = 'abalone.csv'
iris = 'iris.csv'
seeds = 'seeds_dataset.csv'
debug = False
fake = False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def reading_data_set(path):
    print("reading data")
    with open(folder+path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        dictio = {}
        data = []
        for row in csv_reader:
            listy = []
            for i in range(len(row)):
                if is_number(row[i]):
                    listy.append(float(row[i]))
                else:
                    if i in dictio:
                        lst = dictio[i]
                        if row[i] in lst:
                            pass
                        else:
                            dictio[i].append(row[i])
                    else:
                        dictio[i] = [row[i]]
                    lst = dictio[i]
                    listy.append(lst.index(row[i]))

            line_count += 1
            data.append(listy)
            if line_count>2 and debug and fake:
                break
        data = np.array(data)

        y_data = data[:, data.shape[1]-1]
        x_data = data[:, :data.shape[1]-1]
        print(x_data.shape[1])
        print(line_count, " lines read from file.")
        return dictio, x_data, y_data


class PCA:
    def __init__(self, n_components, whiten=False):
        self.n=n_components
        self.s=whiten

    ### this is computing eigenvalues and vector as fast as possible

    def householder_reflection(self, a, e):

        assert a.ndim == 1
        assert np.allclose(1, np.sum(e ** 2))

        u = a - np.sign(a[0]) * np.linalg.norm(a) * e
        v = u / np.linalg.norm(u)
        H = np.eye(len(a)) - 2 * np.outer(v, v)

        return H

    def qr_decomposition(self, A):

        n, m = A.shape
        assert n >= m

        Q = np.eye(n)
        R = A.copy()

        for i in range(m - int(n == m)):
            r = R[i:, i]

            if np.allclose(r[1:], 0):
                continue

            e = np.zeros(n - i)
            e[0] = 1

            H = np.eye(n)
            H[i:, i:] = self.householder_reflection(r, e)

            Q = np.dot(Q, H.T)
            R = np.dot(H, R)

        return Q, R

    def eigen_decomposition(self, covariance, max_iter=1000):
        covariance_k = covariance
        Q_k = np.eye(covariance.shape[1])

        for k in range(max_iter):
            Q, R = self.qr_decomposition(covariance_k)
            Q_k = np.dot(Q_k, Q)
            covariance_k = np.dot(R, Q)

        eigenvalues = np.diag(covariance_k)
        eigenvectors = Q_k
        return eigenvalues, eigenvectors

    def explain_variance(self):
        return self.values / np.sum(self.values)

    def all_variance(self):
        return self.eigens / np.sum(self.eigens)

    def fit(self, X):
        n, m = X.shape
        self.mu = X.mean(axis=0)
        X = X - self.mu
        if self.s:
            self.std = X.std(axis=0)
            X = X/self.std
        covariance = np.matmul(X.T, X)/(n-1)
        self.values, self.vectors = self.eigen_decomposition(covariance)

        self.eigens = self.values

        # self.values, self.vectors = np.linalg.eig(covariance)
        descending_order = np.argsort(-1*self.values)
        self.values = self.values[descending_order]
        self.vectors = self.vectors[:, descending_order]
        if self.n is not None:
            self.values = self.values[0:self.n]
            self.vectors = self.vectors[:, 0:self.n]
        # self.vectors = self.vectors.T

    def transform(self, X):
        X = X - self.mu
        if self.s:
            X = X / self.std
        return np.dot(X, self.vectors)


def test_PCA(data, Y, n, name):
    from sklearn.decomposition import PCA
    pca = PCA(n)
    pca.fit(data)
    transformed = pca.transform(data)
    print("components:", pca.components_)
    print("variances:", pca.explained_variance_)
    print("transformer", transformed)
    if not fake:
        visualisation(transformed[:, 0], transformed[:, 1], Y, name)


def PCAF(data, n):
    print("PCA")
    pca = PCA(n)
    pca.fit(data)
    X = pca.transform(data)
    print(pca.all_variance())
    print(pca.explain_variance())
    plt.plot(range(len(pca.all_variance())), pca.all_variance())
    plt.show()
    return X


def visualisation(data, data_Y, name):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if data.shape[1] is 3:
        ax = Axes3D(fig)
    classes = np.unique(data_Y)
    c = []
    for i in range(classes.shape[0]):
        ii = np.where(data_Y == classes[i])[0]
        c.append((np.random.randint(0, 255) / 255, np.random.randint(0, 255) / 255, np.random.randint(0, 255) / 255, 1))
        if data.shape[1] is 2:
            x = data[ii, 0]
            y = data[ii, 1]
            ax.scatter(x, y, color=c)
        elif data.shape[1] is 3:
            x = data[ii, 0]
            print(x.shape)
            y = data[ii, 1]
            print(y.shape)
            z = data[ii, 2]
            print(z.shape)
            print(data.shape)
            co = c[len(c)-1]
            print(co)
            print(x)
            print(y)
            print(z)
            print("====================")
            ax.scatter(x, y, z, color=co, s=50)

        else:
            print("What happened")
        plt.title(name.split(".csv")[0])

    ax.set_xlabel('principal component 1')
    ax.set_ylabel('principal component 2')
    if data.shape[1] is 3:
        ax.set_zlabel('principal component 3')
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
    plt.show()


if __name__ == '__main__':
    n_components = 3
    if debug and fake:
        data_X = np.array([[1, 2], [3, 4], [5, 6]])
        test_PCA(data_X, data_X, n_components, "fake")

    if not debug:
        non_numeric, data_X, data_Y = reading_data_set(abalone)
        if fake:
            test_PCA(data_X, data_Y, n_components, abalone)
        reduced = PCAF(data_X, n_components)
        visualisation(reduced, data_Y, abalone)

    non_numeric, data_X, data_Y = reading_data_set(iris)
    if debug and not fake:
        test_PCA(data_X, data_Y, n_components, iris)
    reduced = PCAF(data_X, n_components)
    visualisation(reduced, data_Y, iris)

    if not debug:
        non_numeric, data_X, data_Y = reading_data_set(seeds)
        if fake:
            test_PCA(data_X, data_Y, n_components, seeds)
        reduced = PCAF(data_X, n_components)
        visualisation(reduced, data_Y, seeds)



