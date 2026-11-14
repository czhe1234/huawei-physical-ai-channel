"""Check the one-carrier NeRF2 decoder against the author's CSI decoder."""
import argparse
import ast
import json
import math

import numpy as np
import scipy.constants as sc
import torch
import torch.nn.functional as F

from neural_fields import nerf2_complex_render
from study_common import AUTHORS, REPORTS, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tree = ast.parse((AUTHORS / "nerf2/renderer.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"Renderer", "Renderer_CSI"}]
    namespace = {"torch": torch, "np": np, "F": F, "sc": sc}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "author_CSI_renderer", "exec"), namespace)
    original = namespace["Renderer_CSI"](None, n_samples=32, near=.01, far=4.)
    generator = torch.Generator(device=device).manual_seed(20261017)
    raw = torch.randn(3, 7, 32, 4, generator=generator, device=device)*.2
    distance = torch.linspace(.01, 4., 32, device=device)[None, None].expand(3, 7, -1)
    direction = F.normalize(torch.randn(3, 7, 3, generator=generator, device=device), dim=-1)
    repeated = torch.cat([raw[..., i, None].expand(-1, -1, -1, 26) for i in range(4)], -1)
    reference = original.raw2outputs_signal(repeated, distance, direction)[..., 0]
    actual = nerf2_complex_render(raw, distance, direction, frequency=2.4e9)
    maximum = float((actual-reference).abs().max())
    if not torch.allclose(actual, reference, rtol=3e-5, atol=3e-6):
        raise ValueError(f"Scalar decoder differs from the published CSI decoder: {maximum}")
    result = {"status": "passed", "nerf2_scalar_decoder_matches_author_26_carrier_repetition": True,
              "comparison_frequency_hz": 2.4e9, "maximum_complex_absolute_error": maximum,
              "production_frequency_hz": 3.438e9, "measured_labels_used": False}
    atomic_json(REPORTS / "field_equation_checks.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
