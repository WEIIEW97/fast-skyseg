import torch.nn as nn


def replace_modules(
    parent_module: nn.Module, old_module_type: nn.Module, new_module_type: nn.Module
):
    for name, child in parent_module.named_children():
        if isinstance(child, old_module_type):
            setattr(parent_module, name, new_module_type())
        else:
            replace_modules(child, old_module_type, new_module_type)
