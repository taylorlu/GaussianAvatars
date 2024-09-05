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
        GaussianModel.load_ply(self.gaussians, 'output/point_cloud.ply')
        self.flame_model = FlameHead(300, 100)
        self.register_buffer("static_offset", torch.from_numpy(np.load('output/flame_param.npz')['static_offset']))
        self.rot = unitquat_to_rotmat(quat_wxyz_to_xyzw(self.rotation_activation(self.gaussians._rotation)))

    def rotation_activation(self, x):
        return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-12)

    def forward(self, x):
        shape = x[None, :300]
        expr = x[None, 300:400]
        rotation = x[None, 400:403]
        neck = x[None, 403:406]
        jaw = x[None, 406:409]
        eyes = x[None, 409:415]
        translation = x[None, 415:418]
        verts = self.flame_model(shape, expr, rotation, neck, jaw, eyes, translation, return_landmarks=False, static_offset=self.static_offset)

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

        scaling = self.gaussians.scaling_activation(self.gaussians._scaling)
        scales = scaling * self.gaussians.face_scaling[self.gaussians.binding]

        features_dc = self.gaussians._features_dc
        features_rest = self.gaussians._features_rest
        shs = torch.cat((features_dc, features_rest), dim=1)
        color = (torch.cat([0.5 + 0.282 * shs[:, 0, :], opacity], -1) * 255).clip(0, 255)

        face_orien_mat = self.gaussians.face_orien_mat[self.gaussians.binding]
        rotations = torch.matmul(face_orien_mat, self.rot).transpose(2, 1).reshape([-1, 9])

        output = torch.cat([xyz, scales, color, rotations], dim=-1)

        return output

device = torch.device('cuda')
torch_input = torch.zeros([418]).to(device)
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