# ONNX Conversion Status

## Kết quả cuối cùng: 34/36 model đã convert

| Trạng thái | Số lượng |
|---|---|
| ✅ Đã export và copy vào `backend/weights` | 34 |
| ❌ Không thể export (xem lý do bên dưới) | 2 |

---

## 1. Những việc đã làm (session này)

- Cài `onnxscript` (thiếu dẫn đến lỗi export với PyTorch 2.12.0).
- Fix `rPPG/models/RhythmFormer.py` và `rPPG/models/PhysMamba.py`: thay `from timm.models.layers import ...` bằng inline implementation của `trunc_normal_` và `DropPath` (torchvision 0.21.0+cu124 không tương thích với torch 2.12.0+cu130 nên timm init bị lỗi).
- Fix `rPPG/export/export_onnx.py`:
  - `PhysFormerONNXWrapper.forward()`: gọi `self.model.forward(video_clip, self.gra_sharp)` trực tiếp thay vì `self.model(...)` để ONNX tracing giữ được tham số `gra_sharp` float.
  - Hardcode `pf_img=128`, `pf_chunk=160` cho PhysFormer (model yêu cầu spatial sau stem là 4×4, chỉ đạt được với img_size=128).
- Fix `rPPG/export_batch.sh`: đưa case `*factorize*` lên trước `*ibvp*` (tránh `iBVP_FactorizePhys_FSAM_Res` bị map nhầm sang iBVPNet).
- Copy 18 file `.onnx` còn thiếu từ `rPPG/weights/` sang `backend/weights/`.
- Export thành công 7 model còn lại: `PURE_iBVPNet`, `iBVP_FactorizePhys_FSAM_Res`, `PURE/SCAMPS/UBFC-rPPG_PhysFormer_DiffNormalized`, `PURE/UBFC-rPPG_RhythmFormer`.

---

## 2. Model không thể export: PhysMamba (2 file)

- `PURE_PhysMamba_DiffNormalized.pth`
- `UBFC-rPPG_PhysMamba_DiffNormalized.pth`

**Lý do kép:**
1. `mamba_ssm` phụ thuộc vào `selective_scan_cuda` — một CUDA C++ extension bị lỗi symbol (`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`) do không tương thích với CUDA 13.0.
2. Ngay cả khi `mamba_ssm` chạy được, `selective_scan_cuda.fwd` là custom CUDA kernel — **không có backend ONNX** cho phép export.

**Cách giải quyết (nếu cần):**
- Cài lại `mamba_ssm` và `selective_scan_cuda` build cho CUDA 13.0 (nếu có bản).
- Hoặc thay thế `selective_scan_fn` trong `BiMamba._ssm_forward()` bằng pure-PyTorch reference implementation của SSM (tốc độ chậm hơn nhưng ONNX-compatible).

---

## 3. Danh sách model đã có trong `backend/weights/`

```
BP4D_BigSmall_Multitask_Fold1        BP4D_BigSmall_Multitask_Fold2
BP4D_BigSmall_Multitask_Fold3        BP4D_PseudoLabel_DeepPhys
BP4D_PseudoLabel_EfficientPhys       BP4D_PseudoLabel_PhysNet_DiffNormalized
BP4D_PseudoLabel_TSCAN               iBVP_EfficientPhys
iBVP_FactorizePhys_FSAM_Res          MA-UBFC_deepphys
MA-UBFC_efficientphys                MA-UBFC_physnet
MA-UBFC_tscan                        PURE_DeepPhys
PURE_EfficientPhys                   PURE_FactorizePhys_FSAM_Res
PURE_iBVPNet                         PURE_PhysFormer_DiffNormalized
PURE_PhysNet_DiffNormalized          PURE_RhythmFormer
PURE_TSCAN                           SCAMPS_DeepPhys
SCAMPS_EfficientPhys                 SCAMPS_FactorizePhys_FSAM_Res
SCAMPS_PhysFormer_DiffNormalized     SCAMPS_PhysNet_DiffNormalized
SCAMPS_TSCAN                         UBFC-rPPG_DeepPhys
UBFC-rPPG_EfficientPhys              UBFC-rPPG_FactorizePhys_FSAM_Res
UBFC-rPPG_PhysFormer_DiffNormalized  UBFC-rPPG_PhysNet_DiffNormalized
UBFC-rPPG_RhythmFormer               UBFC-rPPG_TSCAN
```

---

## 4. Ghi chú kỹ thuật

- **PhysFormer**: cần `img_size=128` và `chunk=160` (không phải 72/180). Sau 3 lần MaxPool(1,2,2) không gian là 128/8=16, sau patch stride=(4,4,4) còn 4×4 — khớp với hardcode reshape `view(B, dim, t//4, 4, 4)` trong model.
- **iBVPNet**: lỗi `0 outputs` trước đây là do thiếu `onnxscript`, không phải lỗi model.
- **RhythmFormer**: dùng `timm.models.layers` cũ — đã thay bằng inline `DropPath`/`trunc_normal_` tương thích.
- Tất cả export dùng opset 18 (PyTorch 2.12.0 yêu cầu `onnxscript` cho mọi opset).
