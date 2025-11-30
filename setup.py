from setuptools import setup, find_packages

setup(name="pit", version="0.0.0", packages=[
    "pit",
    "pit.dataset",
    "pit.evaluations",
    "pit.evaluations.fid",
    "pit.evaluations.fvd",
    "pit.models",
    "pit.modules",
    "pit.modules.losses",
    "pit.modules.lpips",
    "pit.modules.lpips.loss",
    "pit.modules.lpips.model",
    "pit.modules.flux",
    "pit.modules.flux.modules",
    "pit.quantization",
])
