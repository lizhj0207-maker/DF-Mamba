from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {"configuration_dff": ["DFFConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_dff"] = [
        "DFFForPreTraining",
        "DFFLayer",
        "DFFModel",
        "DFFPreTrainedModel",
        "DFFForImageClassification",
        "DFFForJointClassificationRegression",
    ]


if TYPE_CHECKING:
    from .configuration_dff import DFFConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dff import (
            DFFForPreTraining,
            DFFLayer,
            DFFModel,
            DFFPreTrainedModel,
            DFFForImageClassification,
            DFFForJointClassificationRegression,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
