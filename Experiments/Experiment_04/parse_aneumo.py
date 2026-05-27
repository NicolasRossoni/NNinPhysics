"""Parse ANEUMO real_data/cfd_data/m=0.002/4.npz e sobe pra Modal volume.

ANEUMO format (descoberto via Data_preprocessing/cfdPreprocess.py):
  array_internal: xyz puvw columns (xyz em metros)
  npz packed:
    X_sup: shape (1, N, 4) = [x, y, z, sdf]  (xyz em metros)
    Y_sup: shape (1, N, 4) = [p, u_shifted, v, w]  (u_shifted = u - 0.5)
    Simple_inlet: (1, 7) = [cx, cy, cz, nx, ny, nz, flow_rate]
    X_inlet: (1, M, 3) = inlet points
"""
import json
from pathlib import Path
import modal

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pyvista", "vtk", "matplotlib")
    .add_local_file("/tmp/Aneumo/real_data/cfd_data/m=0.002/4.npz",
                    "/root/aneumo_m002_case4.npz", copy=True)
    .add_local_file("/tmp/Aneumo/real_data/cfd_data/m=0.003/4.npz",
                    "/root/aneumo_m003_case4.npz", copy=True)
    .add_local_file("/tmp/Aneumo/real_data/img_data/4.npy",
                    "/root/aneumo_img_case4.npy", copy=True))
app = modal.App("tcc-aneur-real-parse", image=image)
volume = modal.Volume.from_name("tcc")
VOL = "/data"


@app.function(cpu=4, memory=8192, timeout=10*60, volumes={VOL: volume})
def parse_and_save():
    """Le os npz da Aneumo, descompacta nos componentes fisicos, salva
    no volume Modal em formato uniforme:
       tcc:final/aneurisma_real/case_AN116_m002.npz
       chaves: xyz (N,3) [mm], u v w p (N,) [m/s, Pa], sdf (N,) [m]
    """
    import time as t
    import numpy as np
    t0 = t.time()
    out_dir = Path(VOL) / "final/aneurisma_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag, src in [("m002", "/root/aneumo_m002_case4.npz"),
                     ("m003", "/root/aneumo_m003_case4.npz")]:
        d = np.load(src, allow_pickle=True)
        X_sup = d['X_sup'][0]  # (N, 4) = [x, y, z, sdf]
        Y_sup = d['Y_sup'][0]  # (N, 4) = [p, u_shifted, v, w]
        X_inlet = d['X_inlet'][0]  # (M, 3) inlet pts
        simple_inlet = d['Simple_inlet'][0]  # (7,)
        xyz = X_sup[:, 0:3] * 1000.0  # m -> mm
        xyz_inlet = X_inlet * 1000.0   # m -> mm
        sdf = X_sup[:, 3] * 1000.0      # m -> mm (signed distance to wall)
        p = Y_sup[:, 0]                 # Pa
        u = Y_sup[:, 1] + 0.5            # desfaz shift
        v = Y_sup[:, 2]
        w = Y_sup[:, 3]
        speed = np.sqrt(u**2 + v**2 + w**2)
        N = len(xyz)
        bbox_min = xyz.min(0); bbox_max = xyz.max(0)
        extent = bbox_max - bbox_min
        out_npz = out_dir / f"case_AN4_{tag}.npz"
        np.savez(out_npz, xyz=xyz, u=u, v=v, w=w, p=p, sdf=sdf,
                 xyz_inlet=xyz_inlet, simple_inlet=simple_inlet)
        # Tambem salvar a mask 3D
        img = np.load("/root/aneumo_img_case4.npy")
        out_mask = out_dir / "case_AN4_mask.npz"
        np.savez(out_mask, mask=img[0,0])  # remove batch+channel
        info = {
            "tag": tag, "N": int(N),
            "bbox_min_mm": bbox_min.tolist(),
            "bbox_max_mm": bbox_max.tolist(),
            "extent_mm": extent.tolist(),
            "speed_max": float(speed.max()),
            "speed_mean": float(speed.mean()),
            "u_range": [float(u.min()), float(u.max())],
            "p_range": [float(p.min()), float(p.max())],
            "n_inlet_pts": int(len(xyz_inlet)),
            "simple_inlet": simple_inlet.tolist(),
            "mask_shape": list(img.shape),
            "out_npz": str(out_npz),
        }
        results[tag] = info
        print(f"[{tag}] N={N} extent={extent} mm, |u|_max={speed.max():.4f} m/s")

    out_json = out_dir / "parse_info.json"
    out_json.write_text(json.dumps(results, indent=2))
    volume.commit()
    print(f"DONE wall={t.time()-t0:.1f}s")
    return results


@app.local_entrypoint()
def main():
    r = parse_and_save.remote()
    print(json.dumps(r, indent=2))
