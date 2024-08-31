from flame_model.flame import FlameHead
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyFlameHeadModel(FlameHead):
    def forward(self, x):
        shape = x[:, :300]
        expr = x[:, 300:400]
        rotation = x[:, 400:403]
        neck = x[:, 403:406]
        jaw = x[:, 406:409]
        eyes = x[:, 409:415]
        translation = x[:, 415:418]
        output = super(MyFlameHeadModel, self).forward(shape, expr, rotation, neck, jaw, eyes, translation, return_landmarks=False)
        return output

# flame_model = FlameHead(shape_params=300, expr_params=100)

# shape = torch.randn([1, 300])
# expr = torch.randn([1, 100])
# rotation = torch.randn([1, 3])
# neck = torch.randn([1, 3])
# jaw = torch.randn([1, 3])
# eyes = torch.randn([1, 6])
# translation = torch.randn([1, 5561])
# zero_centered_at_root_node=False,  # otherwise, zero centered at the face
# return_landmarks=True,
# return_verts_cano=False,
# static_offset = torch.randn([1, 5143])
# static_offset = torch.randn([1, 5143, 3])

torch_input = torch.randn([1, 418])
flame_model = MyFlameHeadModel(300, 100, add_teeth=True)

# torch_input = torch.randn([1, 1, 32, 32])
# flame_model = MyModel()
output = flame_model(torch_input)

# for n, p in flame_model.named_buffers():
#     print('=====', n, p.dtype)

# for name, param_tensor in flame_model.state_dict().items():
#     if param_tensor.dtype == torch.int64:
#         new_param = param_tensor.to(torch.int32)
#         setattr(flame_model, name, new_param)

# for name, param_tensor in flame_model.state_dict().items():
#     if param_tensor.dtype == torch.int64:
#         print(name)

onnx_program = torch.onnx.export(flame_model, torch_input, 'myflame.onnx', opset_version=16, input_names=["input"])
# onnx_program = torch.onnx.dynamo_export(flame_model, torch_input)
# onnx_program.save("myflame.onnx")

import onnx
onnx_model = onnx.load("myflame.onnx")
onnx.checker.check_model(onnx_model)

from nobuco import pytorch_to_keras, ChannelOrder
keras_model = pytorch_to_keras(
    flame_model, 
    args=[torch_input],
    inputs_channel_order=ChannelOrder.PYTORCH,
    outputs_channel_order=ChannelOrder.PYTORCH, 
)
keras_model.save('exp', save_format='tf')

from tensorflowjs import converters, quantization
converters.convert_tf_saved_model(saved_model_dir='exp', 
                                  output_dir='tfjs', 
                                #   quantization_dtype_map={quantization.QUANTIZATION_DTYPE_UINT8:True}
                                 )