from flame_model.flame import FlameHead
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
# from nobuco import pytorch_to_keras, ChannelOrder
from scene.flame_gaussian_model import FlameGaussianModel
from scene.gaussian_model import GaussianModel
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_product, quat_wxyz_to_xyzw
from utils.graphics_utils import compute_face_orientation
import os


class MyGaussianModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gaussians = FlameGaussianModel(sh_degree=3)
        # super(self.gaussians).load_ply('output/point_cloud.ply')
        # self.gaussians.load_ply('output/point_cloud.ply')
        GaussianModel.load_ply(self.gaussians, 'output/point_cloud.ply')
        self.flame_model = FlameHead(300, 100)

    def rotation_activation(self, x):
        return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-12)

    def forward(self, x):
        shape = x[:, :300]
        expr = x[:, 300:400]
        rotation = x[:, 400:403]
        neck = x[:, 403:406]
        jaw = x[:, 406:409]
        eyes = x[:, 409:415]
        translation = x[:, 415:418]
        verts = self.flame_model(shape, expr, rotation, neck, jaw, eyes, translation, return_landmarks=False)

        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.gaussians.face_center = triangles.mean(dim=-2)[0]
        self.gaussians.face_center = self.gaussians.face_center.to(self.gaussians._xyz.device)

        # orientation and scale
        self.gaussians.face_orien_mat, self.gaussians.face_scaling = compute_face_orientation(verts[0], faces, return_scale=True)
        self.gaussians.face_orien_mat = self.gaussians.face_orien_mat.to(self.gaussians._xyz.device)
        self.gaussians.face_scaling = self.gaussians.face_scaling.to(self.gaussians._xyz.device)
        self.gaussians.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.gaussians.face_orien_mat))  # roma

        self.gaussians._xyz.requires_grad_(False)
        xyz = torch.bmm(self.gaussians.face_orien_mat[self.gaussians.binding], self.gaussians._xyz[..., None])
        # xyz = xyz * self.gaussians.face_scaling[self.gaussians.binding] + self.gaussians.face_center[self.gaussians.binding]

        binding1 = self.gaussians.binding.unsqueeze(-1).expand(-1, self.gaussians.face_scaling.size(-1)).long()
        binding2 = self.gaussians.binding.unsqueeze(-1).expand(-1, self.gaussians.face_center.size(-1)).long()
        xyz = xyz[..., 0] * torch.gather(self.gaussians.face_scaling, 0, binding1) + torch.gather(self.gaussians.face_center, 0, binding2)

        self.gaussians._opacity.requires_grad_(False)
        opacity = self.gaussians.opacity_activation(self.gaussians._opacity)

        self.gaussians._scaling.requires_grad_(False)
        scaling = self.gaussians.scaling_activation(self.gaussians._scaling)
        scales = scaling * self.gaussians.face_scaling[self.gaussians.binding]

        self.gaussians._rotation.requires_grad_(False)
        rot = self.rotation_activation(self.gaussians._rotation)
        face_orien_quat = self.rotation_activation(self.gaussians.face_orien_quat[self.gaussians.binding])
        rotations = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))

        self.gaussians._features_dc.requires_grad_(False)
        self.gaussians._features_rest.requires_grad_(False)
        features_dc = self.gaussians._features_dc
        features_rest = self.gaussians._features_rest
        shs = torch.cat((features_dc, features_rest), dim=1)

        return xyz, opacity, scales, rotations, shs


torch_input = torch.randn([1, 418])
gaussian_model = MyGaussianModel()

# xyz, opacity, scales, rotations, shs = gaussian_model(torch_input)
# print(xyz.shape, opacity.shape, scales.shape, rotations.shape, shs.shape)
output_dir = 'exp'
if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)

# onnx_program = torch.onnx.export(gaussian_model, torch.randn([1, 418]), 'myflame.onnx', opset_version=16, input_names=["input"])
# # onnx_program.save("myflame.onnx")

# from pytorch2keras import pytorch_to_keras
# keras_model = pytorch_to_keras(gaussian_model, [torch_input], [(418, )])
# # keras_model.save(os.path.join(output_dir, 'tf_model'), save_format='tf')

# onnx_program = torch.onnx.export(gaussian_model, torch_input, 'myflame.onnx', opset_version=16, input_names=["coeff"], output_names=['xyz', 'opacity', 'scales', 'rotations', 'shs'])
# # onnx_program = torch.onnx.dynamo_export(gaussian_model, torch_input)
# # onnx_program.save("myflame.onnx")

# import onnx
# onnx_model = onnx.load("myflame.onnx")
# onnx.checker.check_model(onnx_model)

from nobuco import pytorch_to_keras, ChannelOrder
# from pytorch2keras import pytorch_to_keras
keras_model = pytorch_to_keras(gaussian_model, [torch_input], input_names={torch_input: "coeff"}, output_names={0: 'xyz', 1: 'opacity', 2: 'scales', 3: 'rotations', 4: 'shs'})
# keras_model = pytorch_to_keras(
#     gaussian_model, 
#     args=[torch_input],
#     inputs_channel_order=ChannelOrder.PYTORCH,
#     outputs_channel_order=ChannelOrder.PYTORCH, 
# )

# from onnx2keras import onnx_to_keras
# k_model = onnx_to_keras(onnx_model, ['input'], [(1, 418, )])
keras_model.save('exp', save_format='tf')