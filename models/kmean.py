import numpy as np

class KMeans:
    def __init__(self, init_type='k-means++', k=3, num_iterations=10):
        self.init_type = init_type
        self.k = k
        self.num_iterations = num_iterations
        self.centroids = []
        self.clusters = []

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    def calculate_centroid(self, data):
        if len(data) == 0:
            return []

        dim = data.shape[1]
        num_points = data.shape[0]
        centroids = []

        for i in range(dim):
            sum_coordinate = np.sum(data[:, i])
            centroid = sum_coordinate / num_points
            centroids.append(centroid)

        return centroids

    def kmeans_initialization(self, data):
        centroids = []

        # Initial centroid selection
        random_idx = np.random.choice(data.shape[0])
        centroids.append(data[random_idx, :])

        if self.init_type == 'k-means++':
            for _ in range(self.k - 1):
                dist = []
                for i in range(data.shape[0]):
                    point = data[i, :]
                    distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                    min_distance = np.min(distances)
                    dist.append(min_distance)

                probabilities = dist / np.sum(dist)
                random_idx = np.random.choice(data.shape[0], p=probabilities)
                centroids.append(data[random_idx, :])
        else:
            random_idxs = np.random.choice(data.shape[0], self.k - 1)
            centroids.extend(data[random_idxs, :])

        centroids = np.array(centroids)

        return centroids

    def train(self, data):
        # Initialization
        self.centroids = self.kmeans_initialization(data)

        # Assign data points to clusters
        for _ in range(self.num_iterations):
            # Create empty clusters
            self.clusters = [[] for _ in range(self.k)]

            # Assign data points to the nearest centroid
            for point in data:
                distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
                nearest_centroid_idx = np.argmin(distances)
                self.clusters[nearest_centroid_idx].append(point)

            # Update centroids
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    self.centroids[i] = np.mean(cluster, axis=0)

        return self.centroids, self.clusters
