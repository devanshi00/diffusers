import torch
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.testing_utils import torch_device
from torch import nn
from diffusers.utils.flashpack.serialization import pack_to_file
import tempfile
import os

class DummyModel(ModelMixin, nn.Module, ConfigMixin):
    config_name = "config.json"
    _internal_dict = {}

    @register_to_config
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

def test_flashpack_loading():
    model = DummyModel()
    model.to(torch_device)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        flashpack_path = os.path.join(tmpdir, "model.flashpack")
        pack_to_file(model, flashpack_path, target_dtype=torch.float32)

        loaded_model = DummyModel.from_pretrained(tmpdir, use_flashpack=True, low_cpu_mem_usage=False)
        loaded_model.to(torch_device)

        assert torch.allclose(model.linear.weight, loaded_model.linear.weight)
