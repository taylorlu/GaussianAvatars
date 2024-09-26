import json, os, shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

# 得到相机的运动数据（位置、旋转）
def get_camera_transform():
    radius = 50
    camara_transforms = []
    start = -30
    end = 30
    gap = 15

    for y_pitch in range(start, end+1, gap):
        for z_yaw in range(start, end+1, gap):
            location_x = radius * np.cos(z_yaw / 180 * np.pi) * np.cos(y_pitch / 180 * np.pi)
            location_y = radius * np.sin(z_yaw / 180 * np.pi) * np.cos(y_pitch / 180 * np.pi)
            location_z = -radius * np.sin(y_pitch / 180 * np.pi)

            print([-location_x, -location_y, location_z, 0, y_pitch, z_yaw])
            camara_transforms.append([-location_x, -location_y, location_z, 0, y_pitch, z_yaw])

    camara_transforms_blender = []
    for x_roll in range(-start, -(end+1), -gap):
        for z_yaw in range(-start, -(end+1), -gap):
            location_x = -radius * np.sin(z_yaw / 180 * np.pi) * np.cos(x_roll / 180 * np.pi)
            location_y = radius * np.cos(z_yaw / 180 * np.pi) * np.cos(x_roll / 180 * np.pi)
            location_z = radius * np.sin(x_roll / 180 * np.pi)

            print([location_x, location_y, location_z, 90-x_roll, 0, z_yaw-180])
            camara_transforms_blender.append([location_x, location_y, location_z, 90-x_roll, 0, z_yaw-180])

    return camara_transforms, camara_transforms_blender


# 将基础数据存为一个 .json 文件
def make_json(step=1, scale=100, discard_begin_end=True):
    camara_transforms, camara_transforms_blender = get_camera_transform()
    meta_faces_path = "606/meta_faces"
    if(not os.path.exists(meta_faces_path)):
        os.makedirs(meta_faces_path)
    abs_faces_path = os.path.join(os.path.abspath('.'), meta_faces_path).replace('\\', '/')
    resolution = 800

    if(step==1):
        ## gen camera transform for UE4
        data_dict = {
            "film_fov": 53.38,
            "film_resolution": [resolution, resolution],
            "output_image_path": abs_faces_path,
            "camera_transform": []
        }

        for camara_transform in camara_transforms:
            data_dict['camera_transform'].append(camara_transform)

        data_json = json.dumps(data_dict, indent=4)
        with open("camera_data.json", 'w') as json_file:
            json_file.write(data_json)

    else:
        ## gen camera transform of blender/images for training
        tmp_dict = {
            "cx": 255.5,
            "cy": 255.5,
            "fl_x": 2048,
            "fl_y": 2048,
            "h": resolution,
            "w": resolution,
            "camera_angle_x": 0.9316567547145732,
            "camera_angle_y": 0.9316567547145732,
        }
        data_dict = tmp_dict.copy()
        data_dict.update({"frames": []})
        for i, camara_transform in enumerate(camara_transforms_blender):
            go_start = 0
            to_end = 190
            if(discard_begin_end):
                go_start = 9
                to_end = 199
                for k in range(go_start): ## discard the first 10 frames due to motion blur.
                    (lambda f: (os.remove(f) if os.path.isfile(f) else None))(os.path.join(meta_faces_path, "{:02d}_{:04d}.png".format(i, k)))
                k = 199 # this is a cycle, UE4 always start and the 2 frame and end with the first frame
                (lambda f: (os.remove(f) if os.path.isfile(f) else None))(os.path.join(meta_faces_path, "{:02d}_{:04d}.png".format(i, k)))
            for k in range(go_start, to_end):
                item = tmp_dict.copy()
                item['timestep_index'] = k - go_start
                item['timestep_index_original'] = k - go_start
                item['timestep_id'] = "frame_{:04d}".format(k - go_start)
                item['camera_index'] = i
                item['camera_id'] = "cam_{:04d}".format(i)

                r = R.from_euler('xyz', camara_transform[3:], degrees=True).as_matrix()
                t = np.array(camara_transform[:3])[:, None] / scale
                item['transform_matrix'] = np.concatenate([np.concatenate([r, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0).tolist()

                src = os.path.join(meta_faces_path, "{:02d}_{:04d}.png".format(i, k))
                dst = os.path.join(meta_faces_path, "{:02d}_{:04d}.png".format(i, k - go_start))
                shutil.move(src, dst)
                item['file_path'] = "meta_faces/{:02d}_{:04d}.png".format(i, k - go_start)
                item['flame_param_path'] = "flame_param/{:04d}.npz".format(k - go_start)
                data_dict['frames'].append(item)

        data_json = json.dumps(data_dict, indent=4)
        with open("camera_data_blender.json", 'w') as json_file:
            json_file.write(data_json)


if __name__ == '__main__':
    make_json(step=2, scale=90.6472, discard_begin_end=False)
