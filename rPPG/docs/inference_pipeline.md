# Inference pipeline

Mỗi inference notebook trong [notebooks_inference/](../notebooks_inference/) tuân
theo cùng 1 pipeline. Cell ordering:

```
1. Imports
2. ## Inlined source                  (model code đã inline)
3. Paths + hyperparameters + MODELS list (tên model + weight path)
4. Preprocessing helpers              (giống y hệt training để bảo đảm consistency)
5. Discover subjects + read ground truth
6. Data preprocessing (cache .npy)
7. Dataset class + DataLoader
8. Post-processing functions          (detrend, bandpass, FFT)
9. Main inference loop (per model, per subject → metrics)
10. Convert metrics.json → CSV
```

## Sơ đồ luồng dữ liệu

```mermaid
flowchart TD
    A["data/Normal/&lt;subject&gt;/<br/>video.mkv + ppg.csv"] --> B["Read video + resample PPG<br/>(same as training)"]
    B --> C["Face crop + resize"]
    C --> D["Normalize (same scheme as training)"]
    D --> E[Chunk into clips]
    E --> F["preprocessed_data/Normal/&lt;group&gt;/"]

    F --> G[Inference Dataset]
    G --> H[DataLoader<br/>shuffle=False]

    I["MODELS list:<br/>(name, class, weight_path)"] --> J[Loop over models]
    J --> K["Load model + weights<br/>final_model_release/X.pth"]
    K --> L[Set eval mode]

    H --> M["Forward pass per chunk<br/>(no grad)"]
    L --> M
    M --> N["Reshape output<br/>(model-specific)"]

    N --> O{Label type?}
    O -->|DiffNormalized| P["cumsum<br/>(integrate derivative)"]
    O -->|Standardized| Q[Skip cumsum]
    P --> R["Detrend<br/>(scipy.signal.detrend)"]
    Q --> R
    R --> S["Bandpass filter<br/>(0.75-2.5 Hz typical)"]
    S --> T["Periodogram → peak frequency"]
    T --> U[HR_pred = peak × 60 bpm]

    V[Ground truth PPG] --> W[Same post-processing] --> X[HR_gt]

    U --> Y[Compute MAE / RMSE]
    X --> Y
    Y --> Z["results/Normal/&lt;group&gt;/&lt;model&gt;/<br/>metrics.json + predictions.csv"]

    style F fill:#e1f5ff
    style K fill:#d4edda
    style Z fill:#fff3cd
```

## MODELS list pattern

Mỗi inference notebook có 1 cell `MODELS` liệt kê các weights cần test:

```python
MODELS = [
    ("PURE_DeepPhys",              "DeepPhys", "final_model_release/PURE_DeepPhys.pth"),
    ("UBFC-rPPG_DeepPhys",         "DeepPhys", "final_model_release/UBFC-rPPG_DeepPhys.pth"),
    ("MA-UBFC_deepphys",           "DeepPhys", "final_model_release/MA-UBFC_deepphys.pth"),
    # ... mỗi entry là 1 weight để benchmark
]
```

Inference loop sẽ lặp qua tất cả → 1 output folder cho mỗi `(model_name, weight)`.

## Post-processing chi tiết

Khác nhau giữa các nhóm — xem [notebook_groups.md](notebook_groups.md) hoặc
[../notebooks_inference/model_groups.md](../notebooks_inference/model_groups.md):

| Bước | Nhóm A/B/C/D/E | Nhóm F/G |
|---|---|---|
| Cumsum trên prediction | ✓ (label = DiffNormalized) | ✗ (label = Standardized) |
| Detrend | ✓ | ✓ |
| Bandpass filter | ✓ | ✓ |
| FFT peak → HR | ✓ | ✓ |

## Output sau inference

```
results/Normal/<group>/<model>/
├── metrics.json          {MAE, RMSE, Pearson_r, n_subjects, ...}
├── predictions.csv       {subject_id, chunk_id, pred_hr, gt_hr, error}
└── (optional) plots/
```
