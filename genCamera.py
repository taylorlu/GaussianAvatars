import json
import numpy as np

# 得到相机的运动数据（位置、旋转）
def get_camera_transform():
    radius = 50
    camara_transforms = []

    for y_pitch in range(-20, 21, 10):
        for z_yaw in range(-20, 21, 10):
            location_x = radius * np.cos(z_yaw / 180 * np.pi) * np.cos(y_pitch / 180 * np.pi)
            location_y = radius * np.sin(z_yaw / 180 * np.pi) * np.cos(y_pitch / 180 * np.pi)
            location_z = -radius * np.sin(y_pitch / 180 * np.pi)

            print([-location_x, -location_y, location_z, 0, y_pitch, z_yaw])
            camara_transforms.append([-location_x, -location_y, location_z, 0, y_pitch, z_yaw])

    return camara_transforms


# 将基础数据存为一个 .json 文件
def make_json():
    camara_transforms = get_camera_transform()

    data_dict = {
        "film_fov": 53.38,
        "film_resolution": [512, 512],
        "delay_every_frame": 1.0,
        "output_image_path": r"E:\workplace\AvatarSplat\GaussianAvatars\splat\meta_faces",
        "camera_transform": []
    }

    for camara_transform in camara_transforms:
        data_dict['camera_transform'].append(camara_transform)

    data_json = json.dumps(data_dict, indent=4)
    with open("camera_data.json", 'w') as json_file:
        json_file.write(data_json)


if __name__ == '__main__':
    make_json()