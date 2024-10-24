import os
import pickle

class Save_PseudoGT:
    def __init__(self, save_dir, save_name):
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_path = os.path.join(self.save_dir, self.save_name)
        self.data = dict()
        self.world_transformation_matrices = None
        self.registration_matrix = [{'matrix': None} for _ in range(1000)]

    def save(self, data):
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.save_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def check(self):
        return os.path.exists(self.save_path)

    def add_world_transformation_matrices(self, world_transformation_matrices):
        self.world_transformation_matrices = world_transformation_matrices

    def add_registration_matrix(self, instance_id, matrix):
        self.registration_matrix[instance_id]['matrix'] = matrix