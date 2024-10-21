import copy
import numpy as np

def id_merging(frame_range, instance_id_list, instance_pcd_list, speed_momentum, position_diff_threshold):
    # last_frame_per_instance = dict()
    # for instance_id in instance_id_list:
    #     if len(instance_pcd_list[instance_id]) == 0:
    #         continue
    #     last_frame_per_instance[instance_id] = max(instance_pcd_list[instance_id].keys())

    instance_pcd_list = copy.deepcopy(instance_pcd_list)
    appearance = []
    corr = dict()
    tried_list = dict()

    recent_position_frame = dict()
    estimated_velocity = dict()
    estimated_position = dict()
    for frame_idx in frame_range:
        ########## Estimate current position ##########
        for instance_id in estimated_velocity.keys():
            estimated_position[instance_id] = recent_position_frame[instance_id][1] + estimated_velocity[instance_id] * (frame_idx - recent_position_frame[instance_id][0])
        ###############################################

        ########## Get current position ##########
        position_this_frame = dict()
        for instance_id in instance_id_list:
            if frame_idx in instance_pcd_list[instance_id]:
                position_this_frame[instance_id] = np.mean(instance_pcd_list[instance_id][frame_idx], axis=0)
        ##########################################

        ############### Merge IDs ################
        new_position_this_frame = copy.deepcopy(position_this_frame)
        for instance_id in position_this_frame.keys():
            if instance_id not in appearance:
                appearance.append(instance_id)
                for prev_instance_id in estimated_position.keys():
                    # if last_frame_per_instance[prev_instance_id] >= frame_idx:
                    #     continue
                    
                    tried_list[(instance_id, prev_instance_id)] = np.linalg.norm(position_this_frame[instance_id] - estimated_position[prev_instance_id])
                    if np.linalg.norm(position_this_frame[instance_id] - estimated_position[prev_instance_id]) < position_diff_threshold:
                        corr[instance_id] = prev_instance_id
                        # change instance_id to prev_instance_id in instance_pcd_list
                        for frame_idx2 in instance_pcd_list[instance_id].keys():
                            if frame_idx2 in instance_pcd_list[prev_instance_id]:
                                instance_pcd_list[prev_instance_id][frame_idx2] = np.concatenate((instance_pcd_list[prev_instance_id][frame_idx2], instance_pcd_list[instance_id][frame_idx2]))
                            else:
                                instance_pcd_list[prev_instance_id][frame_idx2] = instance_pcd_list[instance_id][frame_idx2]
                        instance_pcd_list[instance_id] = dict()
                        new_position_this_frame[prev_instance_id] = np.mean(instance_pcd_list[prev_instance_id][frame_idx], axis=0)
                        del new_position_this_frame[instance_id]
                        break
        position_this_frame = new_position_this_frame
        ##########################################

        ############# Update velocity #############
        new_estimated_velocity = dict()
        for instance_id in instance_id_list:
            if instance_id in estimated_velocity and instance_id in position_this_frame:
                recent_frame, recent_position = recent_position_frame[instance_id]
                new_estimated_velocity[instance_id] = (position_this_frame[instance_id] - recent_position) / (frame_idx - recent_frame) * speed_momentum + \
                    (1 - speed_momentum) * estimated_velocity[instance_id]
            elif instance_id in position_this_frame and instance_id in recent_position_frame:
                recent_frame, recent_position = recent_position_frame[instance_id]
                new_estimated_velocity[instance_id] = (position_this_frame[instance_id] - recent_position) / (frame_idx - recent_frame)
            elif instance_id in estimated_velocity:
                new_estimated_velocity[instance_id] = estimated_velocity[instance_id]
        estimated_velocity = new_estimated_velocity
        ##########################################

        for instance_id in position_this_frame:
            recent_position_frame[instance_id] = (frame_idx, position_this_frame[instance_id])
    return corr, tried_list



def merge_instance_ids(instance_pcd_list, instance_color_list, unique_instance_id_list, corr):
    new_unique_instance_id_list = []
    for i in range(len(unique_instance_id_list)):
        if unique_instance_id_list[i] in corr.keys():
            continue
        new_unique_instance_id_list.append(unique_instance_id_list[i])
    new_instance_pcd_list = [{} for _ in range(np.max(new_unique_instance_id_list) + 1)]
    new_instance_color_list = {}
    for instance_id in range(len(instance_pcd_list)):
        if not instance_id in unique_instance_id_list:
            continue
        src_id, target_id = instance_id, instance_id
        while target_id in corr.keys():
            target_id = corr[target_id]
        for frame_idx in instance_pcd_list[src_id].keys():
            new_instance_pcd_list[target_id][frame_idx] = instance_pcd_list[src_id][frame_idx]
        if target_id == src_id:
            new_instance_color_list[target_id] = instance_color_list[src_id]
    return new_instance_pcd_list, new_instance_color_list, new_unique_instance_id_list