# rPPG Notebooks — Architecture Documentation

Sơ đồ kiến trúc tổng quan của các training/inference notebooks trong [rPPG/](../).

## Mục lục

| File | Nội dung |
|---|---|
| [folder_structure.md](folder_structure.md) | Cấu trúc folder + vai trò từng thư mục |
| [training_pipeline.md](training_pipeline.md) | Pipeline training (preprocessing → split → train+val loop → save best) |
| [inference_pipeline.md](inference_pipeline.md) | Pipeline inference (preprocessing → forward → post-process → HR) |
| [notebook_groups.md](notebook_groups.md) | Bảng tổng hợp 8 nhóm notebook + model thuộc nhóm |
| [models/](models/) | Kỹ thuật + kiến trúc chi tiết của từng model (10 files) |

## Tổng quan nhanh

```
                          ┌──────────────────────┐
                          │   Raw video + PPG    │
                          │   data/<dataset>/    │
                          └──────────┬───────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │  Preprocessing (inside notebook)         │
              │  • face crop + resize                    │
              │  • normalize (DiffNorm / Standardized)   │
              │  • chunk into fixed-length clips         │
              └──────────────┬───────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │  preprocessed_data/        │
                │  *.npy (or *.pickle)       │
                └─────┬──────────────────┬───┘
                      │                  │
        ┌─────────────▼──┐         ┌─────▼──────────┐
        │  TRAINING      │         │  INFERENCE     │
        │  notebooks_    │         │  notebooks_    │
        │  training/*    │         │  inference/*   │
        └─────┬──────────┘         └─────┬──────────┘
              │                          │
              ▼                          ▼
    ┌──────────────────┐       ┌──────────────────────┐
    │ final_model_     │ ────► │ Load weights, predict│
    │ release/*.pth    │       │ → HR via FFT         │
    └──────────────────┘       └─────────┬────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ results/.../         │
                              │ metrics.json + csv   │
                              └──────────────────────┘
```

Chi tiết từng pipeline xem [training_pipeline.md](training_pipeline.md) và [inference_pipeline.md](inference_pipeline.md).

## Nguyên tắc thiết kế

- **Notebooks tự-chứa (self-contained)**: tất cả model/loss code đã inline vào notebook — không có import nội bộ.
- **Mỗi nhóm 1 pipeline preprocessing chung** — một notebook xử lý 1 nhóm model có cùng input shape/normalization.
- **Inference pre-process = Training pre-process** — bảo đảm consistency giữa train/test.
- **Best-checkpoint by val HR-MAE** — metric benchmark thật, không phải training loss.
