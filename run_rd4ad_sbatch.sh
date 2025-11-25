#!/bin/bash
#SBATCH -p common-gpu                # 使用 GPU 分区
#SBATCH --gres=gpu:1                 # 申请 1 块 GPU
#SBATCH -c 4                         # 4 CPU cores
#SBATCH --mem=32G                    # 32G 内存
#SBATCH -t 48:00:00                  # 48 小时
#SBATCH -o slurm-%j.out              # 标准输出文件
#SBATCH -e slurm-%j.err              # 错误输出文件

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "===== SLURM JOB START ====="
echo "Running on node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "Working directory: $REPO_DIR"
echo "============================"

# -----------------------------
# 1. 加载 conda 环境
# -----------------------------
source ~/.bashrc
conda activate simplenet   # 或替换为你的 RD4AD 环境名称

# -----------------------------
# 2. 设置路径和超参数
# -----------------------------
export PYTHONUNBUFFERED=1

DATA_ROOT="/work/zc199/CV/mvtec"   # 修改为你的 MVTec 数据集路径
export DATA_ROOT
DATA_LINK="${REPO_DIR}/mvtec"
RESULTS_DIR="${REPO_DIR}/job_outputs/${SLURM_JOB_ID:-manual_run}"
RUN_EVAL="${RUN_EVAL:-1}"
CLASS_LIST=(carpet bottle hazelnut leather cable capsule grid pill transistor metal_nut screw toothbrush zipper tile wood)

# 若需要，将数据集路径链接到项目根目录（main.py 默认读取 ./mvtec）
if [ ! -e "$DATA_LINK" ]; then
  ln -s "$DATA_ROOT" "$DATA_LINK"
fi

# -----------------------------
# 3. 运行 RD4AD 主程序
# -----------------------------
echo "Running RD4AD with main.py defaults..."
python main.py

if [ "$RUN_EVAL" -eq 1 ]; then
  echo "Running evaluation for trained classes..."
  python - <<'PY'
from test import test

classes = ["carpet", "bottle", "hazelnut", "leather", "cable", "capsule", "grid", "pill",
           "transistor", "metal_nut", "screw", "toothbrush", "zipper", "tile", "wood"]

for cls in classes:
    test(cls)
PY
fi

mkdir -p "$RESULTS_DIR"
if compgen -G "./logs/*" > /dev/null; then
  echo "Archiving logs to $RESULTS_DIR"
  cp ./logs/* "$RESULTS_DIR"/
fi
if compgen -G "./checkpoints/*.pth" > /dev/null; then
  echo "Archiving checkpoints to $RESULTS_DIR"
  cp ./checkpoints/*.pth "$RESULTS_DIR"/
fi

echo "===== JOB FINISHED ====="
