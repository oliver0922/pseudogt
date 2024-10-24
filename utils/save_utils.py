import os
import pickle

class Save_PseudoGT:
    def __init__(self, save_dir, save_name, save=True):
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_path = os.path.join(self.save_dir, self.save_name)
        if save:
            self.data = dict()
            self.world_transformation_matrices = None
            self.registration_matrix = dict()
            self.bbox_obj_list = [{} for _ in range(1000)]
            self.static_bbox_obj_list = dict()
            self.id_merge_data = None

    def save(self):
        self.data = {
            'world_transformation_matrices': self.world_transformation_matrices,
            'registration_matrix': self.registration_matrix,
            'bbox_obj_list': self.bbox_obj_list,
            'static_bbox_obj_list': self.static_bbox_obj_list,
            'id_merge_data': self.id_merge_data
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self):
        with open(self.save_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def check(self):
        return os.path.exists(self.save_path)

    def add_world_transformation_matrices(self, world_transformation_matrices):
        self.world_transformation_matrices = world_transformation_matrices

    def add_registration_matrix(self, instance_id, matrix):
        self.registration_matrix[instance_id] = matrix

    def add_bbox(self, instance_id, frame, bbox_obj):
        self.bbox_obj_list[instance_id][frame] = bbox_obj

    def add_static_bbox(self, instance_id, bbox_obj):
        self.static_bbox_obj_list[instance_id] = bbox_obj

    def add_id_merge_data(self, id_merge_data):
        self.id_merge_data = id_merge_data