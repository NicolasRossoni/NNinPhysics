#### Importando Bibliotecas
import json
from pathlib import Path

import modal

### ============= ### ###  Hiperparametros  ### ###  ============= ###

# Caso ANEUMO: paciente AN4, fator de vazao m=0.002.
# Fonte: https://arxiv.org/abs/2505.14717
# A nuvem de pontos com (xyz, u, v, w, p, sdf, xyz_inlet) ja foi extraida
# do arquivo bruto m=0.002/4.npz do dataset e empacotada localmente como
# case_AN4_m002.npz. Este script apenas envia esse npz para o volume Modal,
# em /preprocess/exp_04/, para que 2_train.py o leia direto do volume.

LOCAL_NPZ = "/tmp/case_AN4_m002.npz"   # arquivo parseado (xyz mm, u v w m/s, p Pa, sdf mm)

### ============= ### ###  Modal App  ### ###  ============= ###

app = modal.App(
    "nnphysics-exp04-preprocess",
    image=(modal.Image.debian_slim(python_version="3.11")
           .pip_install("numpy")
           .add_local_file(LOCAL_NPZ, "/root/case_AN4_m002.npz", copy=True)),
)
volume = modal.Volume.from_name("tcc", create_if_missing=True)
VOLUME_PATH = "/data"
OUT_DIR = "/data/preprocess/exp_04"


### ============= ### ###  Funcao remota  ### ###  ============= ###

@app.function(volumes={VOLUME_PATH: volume}, timeout=10 * 60)
def upload_dataset():
    import shutil
    import numpy as np

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    src = Path("/root/case_AN4_m002.npz")
    dst = out / "case_AN4_m002.npz"
    shutil.copy(src, dst)

    # Verificacao do conteudo
    d = np.load(dst)
    xyz = d["xyz"]
    u = d["u"]; v = d["v"]; w = d["w"]; p = d["p"]; sdf = d["sdf"]
    xyz_inlet = d["xyz_inlet"]
    N = len(xyz)
    bbox_min = xyz.min(0).tolist(); bbox_max = xyz.max(0).tolist()
    extent = [bbox_max[i] - bbox_min[i] for i in range(3)]
    speed = (u**2 + v**2 + w**2) ** 0.5

    metadata = {
        "case": "AN4",
        "mass_flow_factor": 0.002,
        "N_points": int(N),
        "N_inlet_points": int(len(xyz_inlet)),
        "bbox_min_mm": bbox_min,
        "bbox_max_mm": bbox_max,
        "extent_mm": extent,
        "speed_max_mps": float(speed.max()),
        "speed_mean_mps": float(speed.mean()),
        "u_range_mps": [float(u.min()), float(u.max())],
        "p_range_Pa": [float(p.min()), float(p.max())],
        "source": "https://arxiv.org/abs/2505.14717",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()

    print(f"[preprocess] case_AN4_m002.npz copiado para {OUT_DIR}")
    print(f"[preprocess] N={N} pontos; bbox extent={extent} mm")
    print(f"[preprocess] |u| max={speed.max():.4f} m/s, p range=[{p.min():.2f},{p.max():.2f}] Pa")
    return metadata


@app.local_entrypoint()
def main():
    meta = upload_dataset.remote()
    print("[preprocess] metadata:", json.dumps(meta, indent=2))
