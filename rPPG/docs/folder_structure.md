# Cấu trúc folder

```
rPPG/
├── configs/                       Reference YAML configs (chỉ tham khảo,
│   ├── train_configs/             notebooks KHÔNG load từ đây)
│   └── infer_configs/
│
├── data/                          Raw input data (KHÔNG track vào git)
│   ├── Headmotion/                  Dataset cho training
│   │   ├── <subject>/             Mỗi subject 1 folder
│   │   │   ├── ppg.csv              Ground-truth PPG signal
│   │   │   └── frame_timestamps.csv
│   │   └── videos/<subject>.mkv
│   └── Normal/                      Dataset cho inference/benchmark
│       └── <subject>/...
│
├── preprocessed_data/             Cache .npy chunks sau preprocessing
│   ├── Headmotion/
│   │   ├── groupA/<subject>/      Mỗi group có cache riêng (khác nhau về
│   │   ├── groupB/<subject>/      normalization / shape)
│   │   ├── ...
│   │   └── bigsmall/<subject>/
│   └── Normal/...
│
├── notebooks_training/            8 self-contained training notebooks
│   ├── bigsmall_training.ipynb    Multi-task (BVP + RESP + AU)
│   ├── groupA_training.ipynb      DeepPhys + TS-CAN
│   ├── groupB_training.ipynb      EfficientPhys
│   ├── groupC_training.ipynb      PhysNet
│   ├── groupD_training.ipynb      PhysFormer
│   ├── groupE_training.ipynb      PhysMamba
│   ├── groupF_training.ipynb      iBVPNet + FactorizePhys
│   └── groupG_training.ipynb      RhythmFormer
│
├── notebooks_inference/           8 inference notebooks (mirror training)
│   ├── bigsmall_inference.ipynb
│   ├── groupA_inference.ipynb     (cùng tên model với training)
│   ├── ...
│   └── model_groups.md            Chi tiết preprocessing/forward mỗi group
│
├── final_model_release/           Trained weights (.pth) — output của training
│   ├── PURE_DeepPhys.pth
│   ├── UBFC-rPPG_PhysFormer.pth
│   └── ...
│
├── export/                        Export sang ONNX
│   └── export_onnx.py             Self-contained (model code đã inline)
│
├── results/                       Outputs từ notebooks
│   ├── Headmotion/                  Training logs
│   │   └── <group>/train_logs/<model>.csv
│   └── Normal/                      Inference results
│       └── <group>/<model>/
│           ├── metrics.json
│           └── predictions.csv
│
└── docs/                          Tài liệu (bạn đang đọc)
```

## Folders KHÔNG còn

Trước refactor, có các folder `neural_methods/`, `evaluation/`, `dataset/`,
`unsupervised/`. Code từ những folder này đã được **inline trực tiếp vào
notebooks** (xem các cell `# === inlined from ...`) và vào
[export/export_onnx.py](../export/export_onnx.py), nên các folder gốc đã bị xóa.
