#!/usr/bin/env bash
# Batch export .pth -> .onnx for rPPG weights, validate, and copy to backend on success
cd /home/iec/MinhHieu/Non-Invasive
set -e
rm -f rPPG/export_batch_log.txt
cd rPPG
for p in weights/*.pth; do
  name=$(basename "$p" .pth)
  if [ -f "weights/${name}.onnx" ]; then
    echo "SKIP ${name} - onnx exists" | tee -a export_batch_log.txt
    continue
  fi
  lname=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  model=""
  case "$lname" in
    *bigsmall*) model=BigSmall ;;
    *deepphys*) model=DeepPhys ;;
    *efficientphys*|*efficient*) model=EfficientPhys ;;
    *tscan*) model=TSCAN ;;
    *physnet*) model=PhysNet ;;
    *physformer*) model=PhysFormer ;;
    *physmamba*) model=PhysMamba ;;
    *rhythm*|*rhythmformer*) model=RhythmFormer ;;
    *factorize*) model=FactorizePhys ;;   # must precede *ibvp* (iBVP_FactorizePhys matches both)
    *ibvpnet*|*ibvp*) model=iBVPNet ;;
    *) echo "UNKNOWN ${name} - skipping" | tee -a export_batch_log.txt ; continue ;;
  esac
  echo "EXPORT ${name} as ${model}" | tee -a export_batch_log.txt
  /home/iec/miniconda3/bin/python export/export_onnx.py --model ${model} --weights "${p}" --output "weights/${name}.onnx" --img-size 72 --chunk 180 --frame-depth 10 --validate 2>&1 | tee -a export_batch_log.txt
  rc=${PIPESTATUS[0]}
  echo "RET ${rc} for ${name}" | tee -a export_batch_log.txt
  if [ ${rc} -eq 0 ] && [ -f "weights/${name}.onnx" ]; then
    cp -v "weights/${name}.onnx" ../backend/weights/ 2>&1 | tee -a export_batch_log.txt
  else
    echo "FAILED ${name}" | tee -a export_batch_log.txt
  fi
done

# summary
python - <<'PY'
import glob,os
pths=sorted(glob.glob('weights/*.pth'))
onnx=sorted(glob.glob('weights/*.onnx'))
name_pth={os.path.splitext(os.path.basename(p))[0] for p in pths}
name_onnx={os.path.splitext(os.path.basename(o))[0] for o in onnx}
print('TOTAL PTH',len(pths),'TOTAL ONNX',len(onnx))
print('UNEXPORTED:')
for n in sorted(name_pth-name_onnx):
    print(' ',n)
print('\nONNX COPIED to backend:')
for o in sorted(glob.glob('../backend/weights/*.onnx')):
    print(' ',os.path.basename(o))
PY
