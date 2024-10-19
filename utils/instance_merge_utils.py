import numpy as np

def id_merging(frame_range, instance_pcd_list, speed_momentum, position_diff_threshold):
    tried_list = dict()

    speed_list = dict()
    estimated_position_list = dict()
    previous_position = dict()
    corr = dict()
    appeared_id = list()
    for frame_idx in frame_range:
        pcd = []
        pcd_id = []
        for i in range(len(instance_pcd_list)):
            if frame_idx in instance_pcd_list[i].keys():
                pcd.extend(instance_pcd_list[i][frame_idx])
                pcd_id.extend([i] * len(instance_pcd_list[i][frame_idx]))
        pcd = np.array(pcd)
        pcd_id = np.array(pcd_id)

        for i in range(len(pcd_id)):
            while pcd_id[i] in corr.keys():
                pcd_id[i] = corr[pcd_id[i]]

        # estimate position based on speed and previous position
        new_estimated_position = dict()
        for i in speed_list.keys():
            new_estimated_position[i] = estimated_position_list[i] + speed_list[i]
        estimated_position_list = new_estimated_position

        new_previous_position = dict()
        new_speed_list = dict()
        new_estimated_position_list = dict()

        for i in range(np.unique(pcd_id).shape[0]):
            instance_id = np.unique(pcd_id)[i]
            # get center position of the instance point cloud
            mask = np.where(pcd_id == instance_id)
            position = np.mean(pcd[mask], axis=0)
            # if any estimated position is close enough to the current position, merge the id
            if appeared_id.count(instance_id) == 0:
                for j in estimated_position_list.keys():
                    #print(f"distance between {instance_id} and {j} is {np.linalg.norm(position - estimated_position_list[j])}")
                    tried_list[(instance_id, j)] = (position, estimated_position_list[j])
                    if not j in pcd_id and np.linalg.norm(position - estimated_position_list[j]) < position_diff_threshold:
                        corr[instance_id] = j
                        pcd_id[mask] = j
                        instance_id = j
                        break
            # update lists
            if previous_position.get(instance_id) is not None:
                prev_frame, prev_pos = previous_position[instance_id]
                if speed_list.get(instance_id) is not None:
                    new_speed_list[instance_id] = (position - prev_pos) / (frame_idx - prev_frame) * speed_momentum + (1 - speed_momentum) * speed_list[instance_id]
                else:
                    new_speed_list[instance_id] = (position - prev_pos) / (frame_idx - prev_frame)
            new_previous_position[instance_id] = (frame_idx, position)
            new_estimated_position_list[instance_id] = position
            if not instance_id in appeared_id:
                appeared_id.append(instance_id)

        tmp = new_estimated_position_list
        for i in estimated_position_list.keys():
            if tmp.get(i) is None:
                tmp[i] = estimated_position_list[i]
        estimated_position_list = tmp

        tmp = new_speed_list
        for i in speed_list.keys():
            if tmp.get(i) is None:
                tmp[i] = speed_list[i]
        speed_list = tmp
        
        tmp = new_previous_position
        for i in previous_position.keys():
            if tmp.get(i) is None:
                tmp[i] = previous_position[i]
        previous_position = tmp

        for i in range(len(pcd_id)):
            while pcd_id[i] in corr.keys():
                pcd_id[i] = corr[pcd_id[i]]
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