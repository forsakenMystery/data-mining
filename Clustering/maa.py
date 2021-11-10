import numpy as np

# print LDA

A = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
print(A)
mean_A = np.mean(A, axis=0, keepdims=True)
print(mean_A)
sigma_A = np.cov(A.T, bias=True)
print(sigma_A)

print("====================")

B = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
print(B)
mean_B = np.mean(B, axis=0, keepdims=True)
print(mean_B)
sigma_B = np.cov(B.T, bias=True)
print(sigma_B)

print("====================")

S_W = sigma_A+sigma_B
print(S_W)

print("====================")
S_B = np.dot((mean_A - mean_B).T, (mean_A-mean_B))
print(S_B)

print("====================")

eig_value, eig_vector = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))
print(eig_vector.T[0])

print("====================")


# PCA


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
        print(X)
        print("====================")
        self.mu = X.mean(axis=0)
        print(self.mu)
        print("====================")
        X = X - self.mu
        if self.s:
            self.std = X.std(axis=0)
            X = X/self.std
        covariance = np.matmul(X.T, X)/(n-1)
        print(covariance)
        print("====================")
        self.values, self.vectors = self.eigen_decomposition(covariance)
        print(self.values)
        print(self.vectors)
        print("====================")

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


print("====================")
print("====================")
print("====================")

print("PCA")
pca = PCA(2)
pca.fit(A)
X = pca.transform(A)
print(X)

print("====================")
