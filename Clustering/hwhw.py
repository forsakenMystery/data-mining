import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans, DBSCAN, Birch, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


debug = False
np.random.seed(0)
n_samples = 1500

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), np.random.randint(0, 3, n_samples)

X_digits, y_digits = datasets.load_digits(return_X_y=True)
dat = scale(X_digits)
n_dat, n_features = dat.shape
n_digits = len(np.unique(y_digits))
labels = y_digits
iris = datasets.load_iris()
X_iris = iris.data
# print(X_iris[0])
y_iris = iris.target

data_set = {"noisy circles": noisy_circles, "noisy moons": noisy_moons, "blobs": blobs, "digits": (dat, y_digits), "iris": (X_iris, y_iris)}


def benchmark(estimator, name, data, st, op=None, we=None):
    print("algorithm:", name)
    X, Y = data
    estimator.fit(X)
    print("homogeneity score: %.2f" % metrics.homogeneity_score(Y, estimator.labels_))
    print("completeness score: %.2f" % metrics.completeness_score(Y, estimator.labels_))
    print("jaccard score: %.2f" % metrics.jaccard_similarity_score(Y, estimator.labels_))
    print("normalized mutual information score: %.2f" % metrics.normalized_mutual_info_score(Y, estimator.labels_))

    pallet = get_cmap(np.unique(estimator.labels_).shape[0] * 5)
    if data == "digits":
        X = PCA(n_components=2).fit_transform(we)
    if data == "iris":
        X = PCA(n_components=2).fit_transform(we)
    print("silhouette score: %.2f" % metrics.silhouette_score(X, estimator.labels_))
    for i, c in enumerate(np.unique(estimator.labels_)):
        x = X[np.where(estimator.labels_ == c)]
        plt.scatter(x[:, 0], x[:, 1], c=pallet(i * 5), label=f"class number %d" % c)
    plt.title("algorithm is " + name + " on data " + st + " " + op)
    plt.legend()
    plt.show()
    print("=====================")


for data in data_set:
    X, Y = data_set[data]
    sets = (X, Y)
    pallet = get_cmap(np.unique(Y).shape[0] * 5)
    if not data == "digits":
        for i, c in enumerate(np.unique(Y)):
            x = X[np.where(Y == c)]
            plt.scatter(x[:, 0], x[:, 1], c=pallet(i*5), label=f"class number %d" % c)
        plt.title("data is "+data)
        plt.legend()
        plt.show()
    elif data == "iris":
        # nothing to show
        print("meh")
    else:
        plt.subplot(251)
        plt.imshow(X[np.where(Y==0)][0].reshape((8, 8)), cmap="gray")
        plt.title("0")
        plt.subplot(252)
        plt.imshow(X[np.where(Y == 1)][0].reshape((8, 8)), cmap="gray")
        plt.title("1")
        plt.subplot(253)
        plt.imshow(X[np.where(Y == 2)][0].reshape((8, 8)), cmap="gray")
        plt.title("2")
        plt.subplot(254)
        plt.imshow(X[np.where(Y == 3)][0].reshape((8, 8)), cmap="gray")
        plt.title("3")
        plt.subplot(255)
        plt.imshow(X[np.where(Y == 4)][0].reshape((8, 8)), cmap="gray")
        plt.title("4")
        plt.subplot(256)
        plt.imshow(X[np.where(Y == 5)][0].reshape((8, 8)), cmap="gray")
        plt.title("5")
        plt.subplot(257)
        plt.imshow(X[np.where(Y == 6)][0].reshape((8, 8)), cmap="gray")
        plt.title("6")
        plt.subplot(258)
        plt.imshow(X[np.where(Y == 7)][0].reshape((8, 8)), cmap="gray")
        plt.title("7")
        plt.subplot(259)
        plt.imshow(X[np.where(Y == 8)][0].reshape((8, 8)), cmap="gray")
        plt.title("8")
        plt.subplot(2, 5, 10)
        plt.imshow(X[np.where(Y == 9)][0].reshape((8, 8)), cmap="gray")
        plt.title("9")
        plt.show()
    if debug:
        print("X")
        k = np.unique(Y).shape[0]
        cluster = k
        reduced_data = PCA(n_components=k).fit_transform(X)

        X_STD = reduced_data

        X_STD = X_STD - np.mean(X_STD, axis=0)
        X_STD /= np.std(X_STD)
        neuron_map = (cluster, 1)
        som = MiniSom(neuron_map[0], neuron_map[1], reduced_data.shape[1], sigma=0.5, learning_rate=0.5, neighborhood_function='gaussian',
                      random_seed=10)
        som.pca_weights_init(X_STD)
        som.train_batch(X_STD, 1000, verbose=True)

        winner_coordinates = np.array([som.winner(x) for x in X_STD]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, neuron_map)
        print("homogeneity score: %.2f" % metrics.homogeneity_score(Y, cluster_index))
        print("completeness score: %.2f" % metrics.completeness_score(Y, cluster_index))
        print("jaccard score: %.2f" % metrics.jaccard_similarity_score(Y, cluster_index))
        print("normalized mutual information score: %.2f" % metrics.normalized_mutual_info_score(Y, cluster_index))
        print("sihoute score: %.2f" % metrics.silhouette_score(X, cluster_index))
        for c in np.unique(cluster_index):
            plt.scatter(X_STD[cluster_index == c, 0],
                        X_STD[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)
        for centroid in som.get_weights():
            plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                        s=80, linewidths=35, color='k')
        plt.legend()
        plt.show()
        print("======================")
    elif not debug:
        print("F")
        X_STD = X
        if data == "digits":
            reduced_data = PCA(n_components=k).fit_transform(X)
            sets = (reduced_data, Y)
            X_STD = reduced_data
        if data == "iris":
            reduced_data = PCA(n_components=3).fit_transform(X)
            sets = (reduced_data, Y)
            X_STD = reduced_data
        k = np.unique(Y).shape[0]
        cluster = k

        met = 'euclidean'
        if data == "aniso":
            eps = .15
        elif data == "noisy circles":
            eps = .1
        elif data == "noisy moons":
            eps = .1
        else:
            eps = .5
        clf = DBSCAN(eps=eps, metric=met)
        benchmark(clf, "dbscan", sets, data, "epsilon of %.2f" % eps, X)

        clf = Birch(n_clusters=cluster)
        benchmark(clf, "birch", sets, data, "with number of cluster: %d" % cluster, X)