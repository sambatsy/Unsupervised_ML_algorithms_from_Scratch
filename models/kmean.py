#kmean

class KMeans:
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

    def kmeans_initialization(self, data, init_type, k):
        centroids = []

        # Initial centroid selection
        random_idx = np.random.choice(data.shape[0])
        centroids.append(data[random_idx, :])

        if init_type == 'k-means++':
            for _ in range(k - 1):
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
            random_idxs = np.random.choice(data.shape[0], k - 1)
            centroids.extend(data[random_idxs, :])

        centroids = np.array(centroids)

        return centroids

    def train(self, data, init_type, k, num_iterations):
        # Initialization
        centroids = self.kmeans_initialization(data, init_type, k)

        # Assign data points to clusters
        for _ in range(num_iterations):
            # Create empty clusters
            clusters = [[] for _ in range(k)]

            # Assign data points to the nearest centroid
            for point in data:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                nearest_centroid_idx = np.argmin(distances)
                clusters[nearest_centroid_idx].append(point)

            # Update centroids
            for i, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    centroids[i] = np.mean(cluster, axis=0)

        return centroids, clusters
