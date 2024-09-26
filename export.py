import pytorch3d.transforms
from flame_model.flame import FlameHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
# from nobuco import pytorch_to_keras, ChannelOrder
from scene.flame_gaussian_model import FlameGaussianModel
from scene.gaussian_model import GaussianModel
from roma import quat_wxyz_to_xyzw, unitquat_to_rotmat
from utils.graphics_utils import compute_face_orientation
import os
import numpy as np
import pytorch3d

class MyGaussianModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gaussians = FlameGaussianModel(sh_degree=3)
        path = 'output/cd269be3-9/point_cloud/iteration_50000/'
        GaussianModel.load_ply(self.gaussians, path+'point_cloud.ply')
        self.flame_model = FlameHead(100, 50)
        flame_param = np.load(path+'flame_param.npz')
        self.register_buffer("static_offset", torch.from_numpy(flame_param['static_offset']))
        self.register_buffer("translation", torch.from_numpy(flame_param['translation']))

        self.register_buffer("zero_shape", torch.from_numpy(flame_param['shape'])[None, :100])
        self.register_buffer("zero_expr", torch.from_numpy(flame_param['expr'][-5, :50]))
        self.register_buffer("zero_rotation", torch.from_numpy(flame_param['rotation'][-5]))
        self.register_buffer("zero_neck_pose", torch.from_numpy(flame_param['neck_pose'][-5]))
        self.register_buffer("zero_jaw_pose", torch.from_numpy(flame_param['jaw_pose'][-5]))
        self.register_buffer("zero_eyes_pose", torch.from_numpy(flame_param['eyes_pose'][-5]))
        self.rot = unitquat_to_rotmat(quat_wxyz_to_xyzw(self.rotation_activation(self.gaussians._rotation)))

    def rotation_activation(self, x):
        return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-12)

    def forward(self, x):
        # shape = x[None, :100]
        expr = x[None, 100:150]
        rotation = x[None, 150:153]
        neck = x[None, 153:156]
        jaw = x[None, 156:159]
        eyes = x[None, 159:165]
        # print(self.zero_shape)
        # print(self.zero_rotation)
        # print(self.zero_jaw_pose)
        # print(self.translation)
        verts = self.flame_model(self.zero_shape, 
                                 self.zero_expr + expr, 
                                 self.zero_rotation + rotation, 
                                 self.zero_neck_pose + neck, 
                                 self.zero_jaw_pose + jaw, 
                                 self.zero_eyes_pose + eyes, 
                                 self.translation, 
                                 zero_centered_at_root_node=False, 
                                 return_landmarks=False, 
                                 static_offset=self.static_offset)
        print(verts)
        # verts = self.flame_model(self.zero_shape, 
        #                          expr, 
        #                          self.zero_rotation + rotation, 
        #                          neck, 
        #                          jaw, 
        #                          eyes, 
        #                          self.translation, 
        #                          zero_centered_at_root_node=False, 
        #                          return_landmarks=False, 
        #                          static_offset=self.static_offset)
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.gaussians.face_center = triangles.mean(dim=-2)[0]

        # orientation and scale
        self.gaussians.face_orien_mat, self.gaussians.face_scaling = compute_face_orientation(verts[0], faces, return_scale=True)

        xyz = torch.bmm(self.gaussians.face_orien_mat[self.gaussians.binding], self.gaussians._xyz[..., None])

        binding1 = self.gaussians.binding.unsqueeze(-1).expand(-1, self.gaussians.face_scaling.size(-1)).long()
        binding2 = self.gaussians.binding.unsqueeze(-1).expand(-1, self.gaussians.face_center.size(-1)).long()
        xyz = xyz[..., 0] * torch.gather(self.gaussians.face_scaling, 0, binding1) + torch.gather(self.gaussians.face_center, 0, binding2)

        opacity = self.gaussians.opacity_activation(self.gaussians._opacity)

        features_dc = self.gaussians._features_dc
        features_rest = self.gaussians._features_rest
        shs = torch.cat((features_dc, features_rest), dim=1)

        color = (torch.cat([0.5 + 0.282 * shs[:, 0, :], opacity], -1) * 255).clip(0, 255)

        scaling = self.gaussians.scaling_activation(self.gaussians._scaling)
        scales = scaling * self.gaussians.face_scaling[self.gaussians.binding]

        face_orien_mat = self.gaussians.face_orien_mat[self.gaussians.binding]
        rotations = torch.matmul(face_orien_mat, self.rot).transpose(2, 1)

        simga = torch.einsum('bij,bi->bij', rotations, scales)

        simga = torch.cat([torch.sum(simga[:, :, 0] ** 2, dim=1, keepdim=True), 
                           torch.sum(simga[:, :, 0] * simga[:, :, 1], dim=1, keepdim=True), 
                           torch.sum(simga[:, :, 0] * simga[:, :, 2], dim=1, keepdim=True), 
                           torch.sum(simga[:, :, 1] ** 2, dim=1, keepdim=True), 
                           torch.sum(simga[:, :, 1] * simga[:, :, 2], dim=1, keepdim=True), 
                           torch.sum(simga[:, :, 2] ** 2, dim=1, keepdim=True)], dim=1)

        output = torch.cat([xyz, color, simga], dim=-1)

        return output

device = torch.device('cuda')
torch_input = torch.zeros([168]).to(device)
gaussian_model = MyGaussianModel().to(device).eval()
output = gaussian_model(torch_input)

output_dir = 'exp'
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)

# import onnx
# onnx_model = onnx.load("myflame.onnx")
# onnx.checker.check_model(onnx_model)
#, output_names={0: 'xyz', 1: 'opacity', 2: 'scales', 3: 'rotations', 4: 'shs'}
from nobuco import pytorch_to_keras, ChannelOrder
# from pytorch2keras import pytorch_to_keras
# keras_model = pytorch_to_keras(gaussian_model, [torch_input], output_names={0: 'xyz', 1: 'scales', 2: 'rotations', 3: 'shs', 4: 'opacity'})
keras_model = pytorch_to_keras(
    gaussian_model, 
    args=[torch_input]
)

keras_model.save('exp', save_format='tf')

#tensorflowjs_converter --input_format=tf_saved_model exp outtfjs
