from nodes.aio_generate import AIOImageGenerate
from nodes.flux2_klein_settings import AIOFlux2Klein9BSettings
from nodes.lora_configuration import AIOLoraConfiguration
from nodes.z_image_settings import AIOZImageTurboSettings


NODE_CLASSES = (
    AIOImageGenerate,
    AIOFlux2Klein9BSettings,
    AIOLoraConfiguration,
    AIOZImageTurboSettings,
)


def test_explicit_node_inputs_have_tooltips():
    missing = []

    for node_class in NODE_CLASSES:
        inputs = node_class.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, spec in inputs.get(section, {}).items():
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if not options.get("tooltip"):
                    missing.append(f"{node_class.__name__}.{section}.{name}")

    assert missing == []
