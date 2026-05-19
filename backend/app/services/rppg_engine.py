"""
rPPGEngine  — ONNX session hoặc PyTorch fallback (global, thread-safe)
SessionState — per-WebSocket state (buffer, prev_frame)

Backend tự động chọn:
  - ONNX (onnxruntime) khi model_path kết thúc bằng .onnx và file tồn tại.
  - PyTorch fallback khi model_path kết thúc bằng .pth, hoặc khi file .onnx
    không tồn tại và tìm được file .pth tương ứng.
"""

import json
import logging
import os
import sys
import threading
from collections import deque

import numpy as np
import onnxruntime as ort

from app.services.preprocessor import build_deepphys_input, build_chunk_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path đến thư mục rPPG (để import model khi dùng PyTorch fallback)
# ---------------------------------------------------------------------------
_RPPG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../rPPG")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model_type(path: str) -> str:
    """Xác định kiến trúc model từ tên file."""
    name = os.path.basename(path).lower()
    # Thứ tự quan trọng: factorize trước ibvp (iBVP_FactorizePhys)
    if "bigsmall"     in name: return "BigSmall"
    if "deepphys"     in name: return "DeepPhys"
    if "efficientphys" in name: return "EfficientPhys"
    if "tscan"        in name: return "TSCAN"
    if "physnet"      in name: return "PhysNet"
    if "physformer"   in name: return "PhysFormer"
    if "physmamba"    in name: return "PhysMamba"
    if "rhythm"       in name: return "RhythmFormer"
    if "factorize"    in name: return "FactorizePhys"
    if "ibvp"         in name: return "iBVPNet"
    raise ValueError(f"Không xác định được kiến trúc từ tên file: {path}")


def _resolve_path(model_path: str) -> tuple[str, str]:
    """
    Trả về (đường_dẫn_thực, 'onnx'|'pytorch').

    Ưu tiên:
    1. path cho sẵn nếu file tồn tại.
    2. Nếu .onnx không có → tìm .pth cùng thư mục, sau đó rPPG/weights/.
    """
    if os.path.exists(model_path):
        ext = os.path.splitext(model_path)[1].lower()
        return model_path, "onnx" if ext == ".onnx" else "pytorch"

    if model_path.endswith(".onnx"):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        candidates = [
            os.path.join(os.path.dirname(model_path), stem + ".pth"),
            os.path.join(_RPPG_PATH, "weights", stem + ".pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                logger.warning(
                    "ONNX không tồn tại (%s), chuyển sang PyTorch fallback: %s",
                    model_path, c,
                )
                return c, "pytorch"

    raise FileNotFoundError(
        f"Không tìm thấy model: {model_path}\n"
        f"  Đã kiểm tra .pth tại: {os.path.dirname(model_path)} và {_RPPG_PATH}/weights/"
    )


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

class PyTorchBackend:
    """
    Wrapper PyTorch model, giao diện tương tự ort.InferenceSession.

    Lưu ý về input format:
    - DeepPhys / TSCAN / EfficientPhys: (1, 6, H, W) — tương thích với
      build_deepphys_input() hiện có, chạy được ngay.
    - PhysNet / iBVPNet / FactorizePhys / PhysFormer / RhythmFormer:
      cần chunk frames (1, 3, T, H, W) — cần cập nhật SessionState và
      preprocessor để tích lũy đủ T frame trước khi infer.
    - BigSmall: hai input riêng biệt — không tương thích với luồng hiện tại.
    - PhysMamba: yêu cầu selective_scan_cuda (CUDA kernel), không chạy được
      nếu thư viện chưa tương thích.
    """

    def __init__(
        self,
        pth_path: str,
        device: str = "cpu",
        img_size: int = 72,
        chunk: int = 180,
        frame_depth: int = 10,
    ):
        import torch

        if _RPPG_PATH not in sys.path:
            sys.path.insert(0, _RPPG_PATH)

        self.device = torch.device(
            "cuda" if "cuda" in device and torch.cuda.is_available() else "cpu"
        )
        model_type = _detect_model_type(pth_path)
        logger.info("PyTorch fallback: %s  →  %s", pth_path, model_type)

        self._model = self._build(model_type, img_size, chunk, frame_depth)
        self._load_weights(pth_path, model_type)
        self._model.eval()
        self._input_name = "input"

    # ── model factory ──────────────────────────────────────────────────────

    def _build(self, model_type: str, img_size: int, chunk: int, frame_depth: int):
        import torch

        if model_type == "DeepPhys":
            from models.DeepPhys import DeepPhys
            return DeepPhys(img_size=img_size)

        if model_type == "TSCAN":
            from models.TSCAN import TSCAN
            return TSCAN(frame_depth=frame_depth, img_size=img_size)

        if model_type == "PhysNet":
            from models.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
            return PhysNet_padding_Encoder_Decoder_MAX(frames=chunk)

        if model_type == "EfficientPhys":
            from models.EfficientPhys import EfficientPhys
            return EfficientPhys(frame_depth=frame_depth, img_size=img_size)

        if model_type == "PhysFormer":
            from models.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
            pf_chunk, pf_img = 160, 128
            return ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(pf_chunk, pf_img, pf_img),
                patches=(4, 4, 4), dim=96, ff_dim=144,
                num_heads=4, num_layers=12, dropout_rate=0.2, theta=0.7,
            )

        if model_type == "PhysMamba":
            try:
                from models.PhysMamba import PhysMamba
                return PhysMamba(frames=chunk)
            except (ImportError, OSError) as e:
                raise ImportError(
                    "PhysMamba yêu cầu selective_scan_cuda (CUDA kernel). "
                    "Thư viện này chưa tương thích với CUDA version hiện tại.\n"
                    f"Lỗi gốc: {e}"
                ) from e

        if model_type == "RhythmFormer":
            from models.RhythmFormer import RhythmFormer
            return RhythmFormer()

        if model_type == "BigSmall":
            from models.BigSmall import BigSmall
            return BigSmall(n_segment=chunk)

        if model_type == "iBVPNet":
            from models.iBVPNet import iBVPNet
            return iBVPNet(frames=chunk, in_channels=3)

        if model_type == "FactorizePhys":
            from models.FactorizePhys.FactorizePhys import FactorizePhys
            md_config = {
                "FRAME_NUM": chunk, "MD_FSAM": True, "MD_TYPE": "NMF",
                "MD_R": 1, "MD_S": 1, "MD_STEPS": 3,
                "MD_RESIDUAL": True, "MD_INFERENCE": True, "MD_TRANSFORM": "T_KAB",
            }
            return FactorizePhys(frames=chunk, md_config=md_config, in_channels=3)

        raise ValueError(f"Kiến trúc không được hỗ trợ: {model_type}")

    # ── weight loading ──────────────────────────────────────────────────────

    def _load_weights(self, pth_path: str, model_type: str):
        import torch

        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.debug("Missing keys (%d): %s …", len(missing), missing[:3])
        self._model.to(self.device)

    # ── InferenceSession interface ──────────────────────────────────────────

    def get_inputs(self):
        class _Info:
            def __init__(self, name):
                self.name = name
        return [_Info(self._input_name)]

    def run(self, _output_names, input_dict: dict) -> list:
        import torch

        inp_np = next(iter(input_dict.values()))
        inp = torch.from_numpy(inp_np).to(self.device)

        with torch.no_grad():
            out = self._model(inp)

        if isinstance(out, (tuple, list)):
            out = out[0]
        return [out.cpu().numpy()]


# ---------------------------------------------------------------------------
# RPPGEngine
# ---------------------------------------------------------------------------

class RPPGEngine:
    """Wraps inference session (ONNX hoặc PyTorch). Stateless — chia sẻ giữa các connection."""

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        with open(config_path) as f:
            cfg = json.load(f)

        self.img_size: int   = cfg.get("img_size", 72)
        self.buffer_size: int = cfg.get("chunk", 180)
        self.norm_type: str  = cfg.get("norm_type", "DiffNorm")
        frame_depth: int     = cfg.get("frame_depth", 10)

        resolved, backend = _resolve_path(model_path)

        if backend == "onnx":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "cuda" in device
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(resolved, providers=providers)
            logger.info(
                "ONNX backend | %s | img_size=%d | buffer=%d | norm=%s | device=%s",
                os.path.basename(resolved), self.img_size, self.buffer_size,
                self.norm_type, device,
            )
        else:
            self.session = PyTorchBackend(
                resolved, device,
                img_size=self.img_size,
                chunk=self.buffer_size,
                frame_depth=frame_depth,
            )
            logger.info(
                "PyTorch backend | %s | img_size=%d | buffer=%d | norm=%s | device=%s",
                os.path.basename(resolved), self.img_size, self.buffer_size,
                self.norm_type, device,
            )

        self.model_type: str = cfg.get("model", "")

        # Xác định số frames cần tích lũy trước khi infer
        # - Frame-by-frame models (DeepPhys, TSCAN, EfficientPhys): input_frames = 0 (dùng prev_frame)
        # - Chunk models (FactorizePhys, PhysNet, ...): đọc dim 2 từ ONNX shape
        _FRAMEWISE = {"DeepPhys", "TSCAN", "EfficientPhys"}
        if self.model_type in _FRAMEWISE:
            self.input_frames: int = 0  # framewise
        else:
            # Lấy số frames từ shape dim 2 của ONNX input (e.g. 181 với FactorizePhys)
            try:
                in_shape = self.session.get_inputs()[0].shape  # ['batch', 3, 181, H, W]
                self.input_frames = int(in_shape[2])
            except Exception:
                self.input_frames = self.buffer_size + 1  # fallback
            logger.info("Chunk model: input_frames=%d", self.input_frames)

        self._input_name: str = self.session.get_inputs()[0].name

    def infer(self, inp: np.ndarray) -> float:
        """Forward pass frame-wise. inp: (1, 6, H, W) float32. Trả về 1 scalar BVP."""
        out = self.session.run(None, {self._input_name: inp})[0]
        return float(out.flatten()[0])

    def infer_chunk(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass chunk-wise. inp: (1, 3, T+1, H, W) float32. Trả về (T,) BVP array."""
        out = self.session.run(None, {self._input_name: inp})[0]  # (1, T)
        return out.flatten().astype(np.float64)  # (T,)

# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class SessionState:
    """Per-WebSocket connection state. Not thread-safe — one per connection.

    Hỗ trợ 2 chế độ:
    - Frame-wise (DeepPhys, TSCAN, EfficientPhys): infer từng cặp frame, tích lũy BVP scalars.
    - Chunk-wise (FactorizePhys, PhysNet, ...): tích lũy input_frames raw crops,
      infer một lần → nhận buffer_size BVP values; sau đó slide window 1 frame.
    """

    def __init__(self, engine: RPPGEngine, fps: int = 30):
        self.engine = engine
        self.fps = fps
        self._is_chunk: bool = engine.input_frames > 0

        self.age: int | None = None
        if self._is_chunk:
            # Lưu raw frame crops (deque giữ tối đa input_frames)
            self._frame_buf: deque[np.ndarray] = deque(maxlen=engine.input_frames)
            self._bvp: deque[float] = deque(maxlen=engine.buffer_size)
            self._step_counter: int = 0
            self._chunk_stride: int = 15  # Chỉ infer 1 lần mỗi 15 frames (~0.5s) để tránh lag
            
            # Threading cho background inference
            self._is_inferring: bool = False
            self._lock = threading.Lock()
        else:
            self._prev_frame: np.ndarray | None = None
            self._bvp: deque[float] = deque(maxlen=engine.buffer_size)

    # ------------------------------------------------------------------

    def push_frame(self, face_crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Push một (img_size × img_size) BGR face crop.
        Trả về numpy BVP array khi đủ buffer_size, ngược lại trả None.
        """
        if self._is_chunk:
            return self._push_chunk(face_crop_bgr)
        return self._push_framewise(face_crop_bgr)

    # ------------------------------------------------------------------

    def _push_framewise(self, frame: np.ndarray) -> np.ndarray | None:
        """Frame-by-frame inference (DeepPhys, TSCAN, EfficientPhys)."""
        if self._prev_frame is None:
            self._prev_frame = frame
            return None

        inp = build_deepphys_input(frame, self._prev_frame)
        bvp_val = self.engine.infer(inp)
        self._bvp.append(bvp_val)
        self._prev_frame = frame

        if len(self._bvp) >= self.engine.buffer_size:
            return np.array(self._bvp, dtype=np.float64)
        return None

    def _do_infer_bg(self, frames_copy: list[np.ndarray]) -> None:
        """Chạy inference trong background thread để không block WebSocket loop."""
        try:
            inp = build_chunk_input(frames_copy)
            bvp_arr = self.engine.infer_chunk(inp)
            with self._lock:
                self._bvp.clear()
                for v in bvp_arr:
                    self._bvp.append(float(v))
        except Exception as e:
            logger.error("Background infer error: %s", e)
        finally:
            self._is_inferring = False

    def _push_chunk(self, frame: np.ndarray) -> np.ndarray | None:
        """Chunk-wise inference (FactorizePhys, PhysNet, ...)."""
        self._frame_buf.append(frame)

        if len(self._frame_buf) < self.engine.input_frames:
            return None  # chưa đủ frames

        self._step_counter += 1

        # Lấy copy của BVP buffer hiện tại một cách an toàn
        with self._lock:
            bvp_copy = np.array(self._bvp, dtype=np.float64) if len(self._bvp) > 0 else None

        # Tránh lag: chỉ infer 1 lần mỗi 15 frames và đảm bảo thread cũ đã xong
        if self._step_counter % self._chunk_stride == 1:
            if not self._is_inferring:
                self._is_inferring = True
                frames_copy = list(self._frame_buf)
                threading.Thread(
                    target=self._do_infer_bg, 
                    args=(frames_copy,), 
                    daemon=True
                ).start()

        return bvp_copy

    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self._is_chunk:
            with self._lock:
                self._frame_buf.clear()
                self._step_counter = 0
                self._bvp.clear()
        else:
            self._prev_frame = None
            self._bvp.clear()
