
class DBScan:

#Initialize DBScan with min_samples and eps
    def __init__(self, min_samples: int, eps: float):
        self.min_samples = min_samples
        self.eps = eps
        self.point_labels = None

#Get neighbors for a given point in the dataset
    def get_neighbors(self, point_index: int, dataset: np.array) -> List:
        point_neighbors = []
        for y_index, point_y in enumerate(dataset):
            if point_index != y_index and np.linalg.norm(dataset[point_index] - point_y) <= self.eps:
                point_neighbors.append(y_index)
        return point_neighbors

#Check if a point is a core point based on min_samples
    def is_core(self, x_index: int, dataset: np.array) -> bool:
        return len(self.get_neighbors(x_index, dataset)) >= self.min_samples

#Assign cluster labels to neighbors of a core point
    def cluster_neighbors(self, x_index: int, dataset: np.array, cluster_index: int) -> None:
        for neighbor_index in self.get_neighbors(x_index, dataset):
            if self.point_labels[neighbor_index] == -1:
                self.point_labels[neighbor_index] = cluster_index
                if self.is_core(neighbor_index, dataset):
                    self.cluster_neighbors(neighbor_index, dataset, cluster_index)

#Perform DBScan clustering on the dataset
    def perform_clustering(self, dataset: np.array):
        cluster_index = 0
        self.point_labels = [-1] * len(dataset)
        for x_index, point_x in enumerate(dataset):
            if self.point_labels[x_index] != -1:
                continue
            if self.is_core(x_index, dataset):
                self.point_labels[x_index] = cluster_index
                self.cluster_neighbors(x_index, dataset, cluster_index)
            cluster_index += 1
        return self.point_labels
