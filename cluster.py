class Cluster:
    def __init__(self):
        self.clusters = []

    def insert(self, mc):
        self.clusters.append(mc)

    def remove(self, mc):
        self.clusters.remove(mc)

    def get_clusters(self):
        return self.clusters
