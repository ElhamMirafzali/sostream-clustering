class MicroCluster:
    def __init__(self, number_data_points, radius, centroid, current_timestamp):
        self.number_data_points = number_data_points
        self.radius = radius
        self.centroid = centroid
        self.creation_time = current_timestamp
        self.last_edited_time = current_timestamp
        self.data_points = []

    def get_radius(self):
        return self.radius

    def get_centroid(self):
        return self.centroid

    def set_radius(self, radius):
        self.radius = radius

    def update_last_edited_time(self, time):
        self.last_edited_time = time

    def insert(self, sample_id):
        self.data_points.append(sample_id)

    def merge_data_points(self, l1, l2):
        self.data_points = l1 + l2

    def calculate_fade(self, current_time, decay_rate):
        t = current_time - self.creation_time
        fade_value = 2 ** (decay_rate * t)
        return fade_value
