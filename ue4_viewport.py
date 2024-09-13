import unreal
import os
import json
import math

# sequence asset path
sequence_asset_path = '/Game/Render_Sequence.Render_Sequence'


# read json file
def read_json_file(path):
    with open(path, "rb") as file:
        file_json = json.load(file)

    export_path = file_json['output_image_path']  # image export path
    film_fov = file_json.get('film_fov', None)  # camera film fov
    film_resolution = file_json.get("film_resolution", [512, 512])  # camera file resolution
    delay_every_frame = file_json.get("delay_every_frame", 3.0)  # sequence run delay every frame
    camera_transforms = file_json['camera_transform']  # camera data

    camera_transform_array = []
    camera_rotation_array = []
    for camera_transform in camera_transforms:
        
        camera_transform_array.append(unreal.Transform(
            location=[camera_transform[0], camera_transform[1], camera_transform[2]],
            rotation=[camera_transform[3], camera_transform[4], camera_transform[5]],
            scale=[1.0, 1.0, 1.0]
        ))
        camera_rotation_array.append(unreal.Rotator(
            roll=camera_transform[3],
            pitch=camera_transform[4],
            yaw=camera_transform[5]
        ))

    return camera_transform_array, camera_rotation_array, export_path, film_fov, film_resolution, delay_every_frame


def create_sequence(asset_name, camera_transform_array, camera_rotation_array, film_fov, film_resolution, package_path='/Game/'):
    # Create a new sequence asset
    sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(asset_name, package_path, unreal.LevelSequence,
                                                                       unreal.LevelSequenceFactoryNew())
    sequence.set_display_rate(unreal.FrameRate(numerator=25, denominator=1))
    total_duration = 0.2  # Total sequence duration depends on the number of cameras
    total_frame = 5
    sequence.set_playback_start_seconds(0)
    sequence.set_playback_end_seconds(total_duration)

    # Add a camera cut track to the sequence
    camera_cut_track = sequence.add_master_track(unreal.MovieSceneCameraCutTrack)

    camera_cut_section = camera_cut_track.add_section()
    camera_cut_section.set_start_frame_bounded(0)
    camera_cut_section.set_end_frame_bounded(total_frame)
    camera_cut_section.set_start_frame_seconds(0)  # Set start time for each camera
    camera_cut_section.set_end_frame_seconds(total_duration)  # Set end time for each camera

    # Create a cine camera actor
    camera_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CineCameraActor,
                                                                    unreal.Vector(0, 0, 0),
                                                                    unreal.Rotator(0, 0, 0))
    camera_actor.set_actor_label('my_camera')
    camera_component = camera_actor.get_cine_camera_component()

    # Set FOV and lens settings
    filmback = camera_component.get_editor_property("filmback")
    filmback.set_editor_property("sensor_height", 60*film_resolution[1]/film_resolution[0])
    filmback.set_editor_property("sensor_width", 60)

    lens_settings = camera_component.get_editor_property("lens_settings")
    lens_settings.set_editor_property("min_focal_length", 4.0)
    lens_settings.set_editor_property("max_focal_length", 1000)
    lens_settings.set_editor_property("min_f_stop", 1.2)
    lens_settings.set_editor_property("max_f_stop", 500)
    camera_component.set_editor_property("current_aperture", 100)

    # we cannot directly set field_of_view here, it can be derivative by current_focal_length and sensor_width
    # tan(fov_x / 2) = (sensor_width / 2) / current_focal_length
    # suppose sensor_width = 60, current_focal_length = 40, then fov = 73.7398 deg
    ratio = math.tan(film_fov / 360.0 * math.pi) * 2
    camera_component.set_editor_property("current_focal_length", filmback.get_editor_property("sensor_width") / ratio)

    # Add the camera to the sequence
    camera_binding = sequence.add_possessable(camera_actor)
    transform_track = camera_binding.add_track(unreal.MovieScene3DTransformTrack)
    transform_section = transform_track.add_section()
    transform_section.set_start_frame_bounded(0)
    transform_section.set_end_frame_bounded(total_frame)

    # Add keyframes for the camera's position and rotation
    channel_location_x = transform_section.get_channels()[0]
    channel_location_y = transform_section.get_channels()[1]
    channel_location_z = transform_section.get_channels()[2]
    channel_rotation_x = transform_section.get_channels()[3]
    channel_rotation_y = transform_section.get_channels()[4]
    channel_rotation_z = transform_section.get_channels()[5]

    camera_location_x = camera_transform_array[0].get_editor_property("translation").get_editor_property("x")
    camera_location_y = camera_transform_array[0].get_editor_property("translation").get_editor_property("y")
    camera_location_z = camera_transform_array[0].get_editor_property("translation").get_editor_property("z")
    camera_rotate_roll = camera_rotation_array[0].get_editor_property("roll")
    camera_rotate_pitch = camera_rotation_array[0].get_editor_property("pitch")
    camera_rotate_yaw = camera_rotation_array[0].get_editor_property("yaw")

    new_time = unreal.FrameNumber(value=0)
    channel_location_x.add_key(new_time, camera_location_x, 0.0)
    channel_location_y.add_key(new_time, camera_location_y, 0.0)
    channel_location_z.add_key(new_time, camera_location_z, 0.0)
    channel_rotation_x.add_key(new_time, camera_rotate_roll, 0.0)
    channel_rotation_y.add_key(new_time, camera_rotate_pitch, 0.0)
    channel_rotation_z.add_key(new_time, camera_rotate_yaw, 0.0)

    # Bind the camera cut section to the camera actor
    camera_binding_id = sequence.make_binding_id(camera_binding, unreal.MovieSceneObjectBindingSpace.LOCAL)
    camera_cut_section.set_camera_binding_id(camera_binding_id)

    # Save the sequence asset
    unreal.EditorAssetLibrary.save_loaded_asset(sequence, False)

    return transform_section


# Sequence movie rendering
def render_sequence_to_movie(current_camera_index, export_path, film_resolution, delay_every_frame, on_finished_callback):
    capture_settings = unreal.AutomatedLevelSequenceCapture()

    capture_settings.settings.output_directory = unreal.DirectoryPath(export_path)
    capture_settings.settings.game_mode_override = None
    capture_settings.settings.output_format = f"{current_camera_index}" + "_{frame}"
    capture_settings.settings.overwrite_existing = False
    capture_settings.settings.use_relative_frame_numbers = False
    capture_settings.settings.handle_frames = 0
    capture_settings.settings.zero_pad_frame_numbers = 4
    capture_settings.settings.use_custom_frame_rate = False
    capture_settings.settings.resolution.res_x = film_resolution[0]
    capture_settings.settings.resolution.res_y = film_resolution[1]
    capture_settings.settings.enable_texture_streaming = False
    capture_settings.settings.cinematic_engine_scalability = True
    capture_settings.settings.cinematic_mode = True
    capture_settings.settings.allow_movement = False
    capture_settings.settings.allow_turning = False
    capture_settings.settings.show_player = False
    capture_settings.settings.show_hud = False
    capture_settings.use_separate_process = False
    capture_settings.close_editor_when_capture_starts = False
    capture_settings.additional_command_line_arguments = "-NOSCREENMESSAGES"
    capture_settings.delay_every_frame = delay_every_frame

    capture_settings.level_sequence_asset = unreal.SoftObjectPath(sequence_asset_path)

    capture_settings.set_image_capture_protocol_type(
        unreal.load_class(None, "/Script/MovieSceneCapture.ImageSequenceProtocol_PNG"))

    capture_settings.get_image_capture_protocol().compression_quality = 100

    unreal.SequencerTools.render_movie(capture_settings, on_finished_callback)


def check_sequence_asset_exist(root_dir):
    sequence_path = os.path.join(root_dir, 'Content', 'Render_Sequence.uasset')
    if os.path.exists(sequence_path):
        unreal.log('---sequence is exist os path---')
        os.remove(sequence_path)
    else:
        unreal.log('---sequence is not exist os path---')

    if unreal.EditorAssetLibrary.does_asset_exist(sequence_asset_path):
        unreal.log('---sequence dose asset exist---')
        unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)
    else:
        unreal.log('---sequence does not asset exist---')

    sequence_asset_data = unreal.AssetRegistryHelpers.get_asset_registry().get_asset_by_object_path(sequence_asset_path)
    if sequence_asset_data:
        unreal.log('---sequence is exist on content---')
        sequence_asset = unreal.AssetData.get_asset(sequence_asset_data)
        unreal.EditorAssetLibrary.delete_loaded_asset(sequence_asset)
    else:
        unreal.log('---sequence is not exit on content---')


current_camera_index = 0  # Global variable to track which camera is being rendered
transform_section, export_path, film_resolution, delay_every_frame, on_finished_callback = None, None, None, None, None
camera_transform_array = []
camera_rotation_array = []

def on_render_movie_finished(success):
    global current_camera_index, transform_section, export_path, film_resolution, delay_every_frame, on_finished_callback, camera_transform_array, camera_rotation_array
    unreal.log('on_render_movie_finished is called')
    if success:
        current_camera_index += 1
        if(current_camera_index==len(camera_transform_array)):
            return

        # Add keyframes for the camera's position and rotation
        channel_location_x = transform_section.get_channels()[0]
        channel_location_y = transform_section.get_channels()[1]
        channel_location_z = transform_section.get_channels()[2]
        channel_rotation_x = transform_section.get_channels()[3]
        channel_rotation_y = transform_section.get_channels()[4]
        channel_rotation_z = transform_section.get_channels()[5]

        camera_location_x = camera_transform_array[current_camera_index].get_editor_property("translation").get_editor_property("x")
        camera_location_y = camera_transform_array[current_camera_index].get_editor_property("translation").get_editor_property("y")
        camera_location_z = camera_transform_array[current_camera_index].get_editor_property("translation").get_editor_property("z")
        camera_rotate_roll = camera_rotation_array[current_camera_index].get_editor_property("roll")
        camera_rotate_pitch = camera_rotation_array[current_camera_index].get_editor_property("pitch")
        camera_rotate_yaw = camera_rotation_array[current_camera_index].get_editor_property("yaw")

        new_time = unreal.FrameNumber(value=0)
        channel_location_x.add_key(new_time, camera_location_x)
        channel_location_y.add_key(new_time, camera_location_y)
        channel_location_z.add_key(new_time, camera_location_z)
        channel_rotation_x.add_key(new_time, camera_rotate_roll)
        channel_rotation_y.add_key(new_time, camera_rotate_pitch)
        channel_rotation_z.add_key(new_time, camera_rotate_yaw)

        unreal.log(f'Finished rendering for camera {current_camera_index - 1}')
        render_sequence_to_movie(current_camera_index, export_path, film_resolution, delay_every_frame, on_finished_callback)
    else:
        unreal.log('Render failed')

def main():
    global current_camera_index, transform_section, camera_transform_array, camera_rotation_array, transform_section, export_path, film_resolution, delay_every_frame, on_finished_callback
    root_dir = unreal.SystemLibrary.get_project_directory()
    json_path = os.path.join(root_dir, 'camera_data.json')

    camera_transform_array, camera_rotation_array, export_path, film_fov, film_resolution, delay_every_frame = read_json_file(json_path)

    check_sequence_asset_exist(root_dir)

    transform_section = create_sequence('Render_Sequence', camera_transform_array, camera_rotation_array, film_fov, film_resolution)

    # Prepare the callback
    on_finished_callback = unreal.OnRenderMovieStopped()
    on_finished_callback.bind_callable(on_render_movie_finished)

    render_sequence_to_movie(current_camera_index, export_path, film_resolution, delay_every_frame, on_finished_callback)

main()