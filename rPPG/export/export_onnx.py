"""Export a trained PyTorch rPPG model to ONNX format.

Usage:
    python export/export_onnx.py --model DeepPhys --weights weights/PURE_DeepPhys.pth \
        --output weights/PURE_DeepPhys.onnx --img-size 72 --chunk 180

Supported models: DeepPhys, TSCAN, PhysNet, EfficientPhys,
                  PhysFormer, PhysMamba, RhythmFormer, BigSmall,
                  iBVPNet, FactorizePhys
"""

import argparse
import json
import os
import sys

# Thêm thư mục gốc rPPG/ vào sys.path để import được models/
RPPG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RPPG_ROOT not in sys.path:
    sys.path.insert(0, RPPG_ROOT)

import torch
import torch.onnx


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, args):
    """Instantiate the requested model in eval mode."""
    name = model_name.lower()

    if name == "deepphys":
        from models.DeepPhys import DeepPhys
        model = DeepPhys(img_size=args.img_size)
        # Input: (N, 6, H, W)  — DiffNorm (3ch) + Standardized (3ch)
        dummy = torch.zeros(1, 6, args.img_size, args.img_size)
        input_names = ["appearance_motion_input"]
        dynamic = {"appearance_motion_input": {0: "batch"}, "output": {0: "batch"}}

    elif name == "tscan":
        from models.TSCAN import TSCAN
        model = TSCAN(frame_depth=args.frame_depth, img_size=args.img_size)
        dummy = torch.zeros(args.frame_depth, 6, args.img_size, args.img_size)
        input_names = ["frame_sequence"]
        dynamic = {"frame_sequence": {0: "T"}, "output": {0: "T"}}

    elif name == "physnet":
        from models.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=args.chunk)
        dummy = torch.zeros(1, 3, args.chunk, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    elif name == "efficientphys":
        from models.EfficientPhys import EfficientPhys
        model = EfficientPhys(frame_depth=args.frame_depth, img_size=args.img_size)
        dummy = torch.zeros(args.frame_depth + 1, 3, args.img_size, args.img_size)
        input_names = ["frame_sequence"]
        dynamic = {"frame_sequence": {0: "T"}, "output": {0: "T"}}

    elif name == "physformer":
        from models.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(args.chunk, args.img_size, args.img_size),
            patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4,
            num_layers=12, dropout_rate=0.2, theta=0.7,
        )
        dummy = torch.zeros(1, 3, args.chunk, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    elif name == "physmamba":
        from models.PhysMamba import PhysMamba
        model = PhysMamba(frames=args.chunk)
        dummy = torch.zeros(1, 3, args.chunk, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    elif name == "rhythmformer":
        from models.RhythmFormer import RhythmFormer
        model = RhythmFormer()
        dummy = torch.zeros(1, args.chunk, 3, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    elif name == "bigsmall":
        from models.BigSmall import BigSmall
        model = BigSmall(n_segment=args.chunk)
        dummy_big   = torch.zeros(args.chunk, 3, 144, 144)
        dummy_small = torch.zeros(args.chunk, 3,   9,   9)
        dummy = (dummy_big, dummy_small)
        input_names = ["big_frames", "small_frames"]
        dynamic = {
            "big_frames":   {0: "T"},
            "small_frames": {0: "T"},
            "output":       {0: "T"},
        }

    elif name == "ibvpnet":
        from models.iBVPNet import iBVPNet
        model = iBVPNet(frames=args.chunk, in_channels=3)
        dummy = torch.zeros(1, 3, args.chunk + 1, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    elif name == "factorizephys":
        from models.FactorizePhys.FactorizePhys import FactorizePhys
        md_config = {
            "FRAME_NUM": args.chunk, "MD_FSAM": True, "MD_TYPE": "NMF",
            "MD_R": 1, "MD_S": 1, "MD_STEPS": 3,
            "MD_RESIDUAL": True, "MD_INFERENCE": True, "MD_TRANSFORM": "T_KAB",
        }
        model = FactorizePhys(frames=args.chunk, md_config=md_config, in_channels=3)
        dummy = torch.zeros(1, 3, args.chunk + 1, args.img_size, args.img_size)
        input_names = ["video_clip"]
        dynamic = {"video_clip": {0: "batch"}, "output": {0: "batch"}}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, dummy, input_names, dynamic


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    print(f"[export] Model   : {args.model}")
    print(f"[export] Weights : {args.weights}")
    print(f"[export] Output  : {args.output}")

    model, dummy, input_names, dynamic_axes = build_model(args.model, args)

    state_dict = torch.load(args.weights, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            args.output,
            opset_version=17,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"[export] ONNX saved → {args.output}")

    # Save model metadata alongside the ONNX file
    meta = {
        "model":      args.model,
        "weights":    args.weights,
        "img_size":   args.img_size,
        "chunk":      args.chunk,
        "frame_depth": args.frame_depth,
        "opset":      17,
    }
    meta_path = args.output.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] Metadata → {meta_path}")

    # Validate: compare PyTorch vs ONNX output
    if args.validate:
        _validate(model, dummy, args.output)


def _validate(model, dummy, onnx_path: str, tol: float = 1e-4):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[validate] onnxruntime not installed — skipping validation")
        return

    if isinstance(dummy, tuple):
        pt_inputs = [x.unsqueeze(0) if x.dim() == 3 else x for x in dummy]
    else:
        pt_inputs = [dummy]

    with torch.no_grad():
        pt_out = model(*pt_inputs) if len(pt_inputs) > 1 else model(pt_inputs[0])

    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    pt_np = pt_out.cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        sess.get_inputs()[i].name: (pt_inputs[i].numpy() if len(pt_inputs) > 1 else pt_inputs[0].numpy())
        for i in range(len(sess.get_inputs()))
    }
    ort_out = sess.run(None, ort_inputs)[0]

    max_diff = np.abs(pt_np - ort_out).max()
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"[validate] Max diff PyTorch vs ONNX: {max_diff:.2e}  [{status}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Export rPPG model to ONNX")
    p.add_argument("--model",      required=True, help="Model name (e.g. DeepPhys)")
    p.add_argument("--weights",    required=True, help="Path to .pth weights file")
    p.add_argument("--output",     required=True, help="Output .onnx path")
    p.add_argument("--img-size",   type=int, default=72,  dest="img_size")
    p.add_argument("--chunk",      type=int, default=180, help="Temporal chunk length")
    p.add_argument("--frame-depth",type=int, default=10,  dest="frame_depth")
    p.add_argument("--validate",   action="store_true",   help="Validate ONNX vs PyTorch")
    return p.parse_args()


if __name__ == "__main__":
    export(get_args())
