# Training pipeline

Mỗi training notebook trong [notebooks_training/](../notebooks_training/) tuân
theo cùng 1 pipeline. Cell ordering trong notebook:

```
1. Imports                           (os, torch, tqdm, ...)
2. ## Inlined source                  (model + loss code đã inline)
3. ## Training utilities              (set_seed, val_split, HR-MAE, savers...)
4. Paths + hyperparameters
5. Preprocessing helpers              (read_video, normalize, face crop)
6. Dataset discovery + preprocess     (chạy 1 lần, cache .npy)
7. Dataset class + train/val split + DataLoaders
8. Training loop (train + val per epoch, save best, early stop, CSV log)
```

## Sơ đồ luồng dữ liệu

```mermaid
flowchart TD
    A["data/Headmotion/&lt;subject&gt;/<br/>video.mkv + ppg.csv"] --> B[Read video frames<br/>+ resample PPG]
    B --> C["Face crop + resize<br/>(Haar cascade)"]
    C --> D["Normalize<br/>(DiffNormalized / Standardized)"]
    D --> E["Chunk into clips<br/>(CHUNK_LENGTH per group)"]
    E --> F["preprocessed_data/.../<br/>*_input*.npy + *_label*.npy"]

    F --> G["Dataset class<br/>(GroupADataset, etc.)"]
    G --> H["train_val_split<br/>(val_ratio=0.2, seed=42)"]
    H --> I[train_loader]
    H --> J[val_loader]

    I --> K[Training loop]
    K --> L[Model forward]
    L --> M["Loss (NegPearson / MSE / Multi-task)"]
    M --> N["loss.backward()"]
    N --> O["clip_grad_norm_(max=1.0)"]
    O --> P["optimizer.step()<br/>scheduler.step (OneCycleLR)"]
    P --> K

    J --> Q["Validation (per epoch)"]
    Q --> R["Forward + compute_hr_mae_batch<br/>(periodogram, band 0.6-3.3 Hz)"]
    R --> S{val_hr_mae<br/>improved?}
    S -->|yes| T["BestCheckpointSaver<br/>→ final_model_release/Model.pth"]
    S -->|no| U[EarlyStopping counter +1]
    U --> V{counter ≥ patience?}
    V -->|yes| W[Stop training]
    R --> X["MetricLogger.log()<br/>→ results/.../train_logs/Model.csv"]

    style F fill:#e1f5ff
    style T fill:#d4edda
    style X fill:#fff3cd
    style W fill:#f8d7da
```

## Training utilities chung (cell 3)

Tất cả 8 notebooks chia sẻ utility cell định nghĩa:

| Tên | Mục đích |
|---|---|
| `set_seed(42)` | Reproducibility (random + numpy + torch + CUDA + cudnn.deterministic) |
| `train_val_split(ds, 0.2, 42)` | Random split có seed cố định |
| `compute_hr_fft(sig, fps)` | HR (bpm) từ peak của periodogram trong band 0.6-3.3 Hz |
| `compute_hr_mae_batch(pred, label)` | Mean abs HR error trên batch |
| `BestCheckpointSaver(path, mode='min')` | Chỉ save khi metric improve |
| `EarlyStopping(patience=5)` | Dừng khi metric không improve |
| `MetricLogger(csv_path)` | Append CSV mỗi epoch |

## Common training-loop pseudocode

```python
saver   = BestCheckpointSaver(SAVE_PATH, mode="min")
stopper = EarlyStopping(patience=5, mode="min")
logger  = MetricLogger(LOG_PATH)

for epoch in range(EPOCHS):
    # ----- TRAIN -----
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # ----- VAL -----
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pred = model(data)
            val_hr_mae += compute_hr_mae_batch(pred, label, fps=VIDEO_FPS)

    # ----- LOG + SAVE BEST + EARLY STOP -----
    saver.step(model, val_hr_mae)         # save if best so far
    logger.log(epoch=epoch, val_hr_mae=val_hr_mae, ...)
    if stopper.step(val_hr_mae):
        break
```

## Output sau training

- **Best weights**: `final_model_release/<Model>.pth`
- **Metrics log**: `results/<dataset>/<group>/train_logs/<Model>.csv` với schema
  `{epoch, train_loss, val_loss, val_hr_mae, lr, time_sec}`
