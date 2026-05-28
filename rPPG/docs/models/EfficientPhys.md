# EfficientPhys

> **EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement**
> Liu et al., WACV 2023

Đơn giản hóa TS-CAN: bỏ branch appearance, tự tính frame difference bên
trong model. Mục tiêu **deploy on-device** (mobile, edge).

## Ý tưởng cốt lõi

> "Tại sao phải preprocess DiffNorm bên ngoài rồi feed vào model? Cho model
> tự tính `torch.diff` thì:
> 1. Pipeline đơn giản hơn (chỉ cần raw normalized frames)
> 2. Bỏ được branch appearance → giảm model size 2x
> 3. Vẫn giữ TSM cho temporal context"

## Kiến trúc

```mermaid
flowchart LR
    IN["Input (T+1, 3, 72, 72)<br/>Standardized only, NO DiffNorm"]
    IN --> DIFF["torch.diff() internally<br/>→ (T, 3, 72, 72)"]
    DIFF --> AM[Self-attention mask<br/>shared across spatial]
    AM --> CNN[CNN blocks<br/>+ TSM trước mỗi conv]
    CNN --> POOL[Pool → FC]
    POOL --> OUT["1-D rPPG (T, 1)"]

    style DIFF fill:#ffe5cc
```

**Tại sao phải append 1 extra frame ở input?** Vì `torch.diff` giảm độ
dài time đi 1. Nếu input là `T` frames, output sau diff sẽ chỉ còn `T-1`.
Code training/inference duplicate frame cuối:

```python
data = torch.cat([data, data[-1:].clone()], dim=0)   # T+1 frames
pred = model(data)   # output T frames
```

## Kỹ thuật cụ thể

| Kỹ thuật | Mục đích |
|---|---|
| **Internal `torch.diff`** | Bỏ DiffNorm pre-process; pipeline gọn hơn |
| **Self-attention mask** | Thay cho appearance branch (1 mask thay vì 2-branch) |
| **TSM** | Temporal context như TS-CAN |
| **Frame padding +1** | Bù đắp cho việc diff giảm length |
| **Standardized-only input** | Chỉ 3 channels, không cần 6ch như DeepPhys |

## Input / Output

| | Shape | Ghi chú |
|---|---|---|
| Input | `(T+1, 3, 72, 72)` | Standardized, **+1 frame ở temporal dim** |
| Output | `(T, 1)` | rPPG per frame |
| Constraint | `T % frame_depth == 0` | TSM alignment |

## Sử dụng trong repo

- **Notebook**: [groupB_training.ipynb](../../notebooks_training/groupB_training.ipynb), [groupB_inference.ipynb](../../notebooks_inference/groupB_inference.ipynb)
- **Class**: `EfficientPhys(frame_depth=10, img_size=72)`
- **Loss**: `nn.MSELoss()`
- **Weights pre-trained**: `PURE_EfficientPhys.pth`, `SCAMPS_EfficientPhys.pth`, `UBFC-rPPG_EfficientPhys.pth`, `iBVP_EfficientPhys.pth`, `BP4D_PseudoLabel_EfficientPhys.pth`, `MA-UBFC_efficientphys.pth`

## So sánh DeepPhys ↔ TS-CAN ↔ EfficientPhys

| | DeepPhys | TS-CAN | EfficientPhys |
|---|---|---|---|
| Branches | 2 (motion + appearance) | 2 | 1 (combined) |
| Input channels | 6 (DiffN+Std) | 6 | 3 (Std only) |
| Temporal | ✗ | TSM | TSM |
| Frame difference | Pre-process (DiffNorm) | Pre-process | Internal `torch.diff` |
| Params | ~9 M | ~9 M | ~9 M (bỏ 1 branch nhưng thêm internal ops) |
| Pipeline | DiffNorm cần riêng | DiffNorm cần riêng | Chỉ cần Standardize |
