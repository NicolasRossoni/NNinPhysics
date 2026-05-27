"""Plots do campo CFD real ANEUMO: 3D scatter + slices interpoladas +
streamlines no plano central.
"""
import json
from pathlib import Path
import modal

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "matplotlib", "scikit-learn"))
app = modal.App("tcc-aneur-real-plot", image=image)
volume = modal.Volume.from_name("tcc")
VOL = "/data"


@app.function(cpu=8, memory=24576, timeout=20*60, volumes={VOL: volume})
def plot_all(tag: str = "m002"):
    import time as t
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.interpolate import LinearNDInterpolator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    t0 = t.time()
    npz = Path(VOL) / f"final/aneurisma_real/case_AN4_{tag}.npz"
    d = np.load(npz)
    xyz = d['xyz']  # mm
    u = d['u']; v = d['v']; w = d['w']; p = d['p']
    speed = np.sqrt(u**2 + v**2 + w**2)
    N = len(xyz)
    bbox_min = xyz.min(0); bbox_max = xyz.max(0)
    center = (bbox_min + bbox_max) / 2
    extent = bbox_max - bbox_min
    print(f"[{tag}] N={N} bbox extent {extent} mm, center {center}")

    out_dir = Path(VOL) / "final/aneurisma_real"

    # =========================================================================
    # FIG 1: 3D scatter colorido por |u|
    # =========================================================================
    print(f"[{tag}] FIG 1: 3D scatter |u|")
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # Subsample para nao matar matplotlib
    sub = np.random.choice(N, min(10000, N), replace=False)
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=speed[sub], cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Anatomia real (ANEUMO case 4 {tag}) — N={N} pontos')
    plt.colorbar(sc, ax=ax1, label='|u| (m/s)', shrink=0.6)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=p[sub], cmap='RdBu_r', s=2, alpha=0.6)
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.set_title('Campo de pressão p')
    plt.colorbar(sc2, ax=ax2, label='p (Pa)', shrink=0.6)

    plt.tight_layout()
    out1 = out_dir / f"plot_3d_{tag}.png"
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved {out1}")

    # =========================================================================
    # FIG 2: 4 lines (u,v,w,p) x 5 slices ao longo do eixo principal (PCA)
    # =========================================================================
    print(f"[{tag}] FIG 2: 4x5 slices canon-style")
    # PCA
    centered = xyz - xyz.mean(0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary = eigvecs[:, -1]
    s_proj = centered @ primary
    s_min, s_max = s_proj.min(), s_proj.max()
    # 5 slice positions
    s_targets = np.linspace(s_min + 0.1*(s_max-s_min),
                             s_max - 0.1*(s_max-s_min), 5)
    # build local basis perp to primary
    e1 = primary
    if abs(e1[2]) < 0.9: e2 = np.cross(e1, [0,0,1])
    else: e2 = np.cross(e1, [1,0,0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    # Use kD-tree to find points near each slice
    tree = cKDTree(xyz)
    fields = [("u", u), ("v", v), ("w", w), ("p", p)]
    # Resolucao slice grid
    n_grid = 50
    # plano local: bbox transverse extent
    perp_extent = max(abs((centered @ e2)).max(), abs((centered @ e3)).max())
    grid2 = np.linspace(-perp_extent*1.05, perp_extent*1.05, n_grid)
    G2, G3 = np.meshgrid(grid2, grid2, indexing='ij')

    fig, axes = plt.subplots(4, 5, figsize=(20, 13))
    # Pre-compute global ranges per field for consistent colorbar
    field_ranges = {}
    for fname, field in fields:
        field_ranges[fname] = (float(np.percentile(field, 1)),
                                float(np.percentile(field, 99)))
    for row, (fname, field) in enumerate(fields):
        vmin, vmax = field_ranges[fname]
        absmax = max(abs(vmin), abs(vmax))
        for col, s_t in enumerate(s_targets):
            # Interpolar field na grid 2D do slice
            plane_center = xyz.mean(0) + s_t * primary
            pts_3d = plane_center[None, None, :] + \
                     G2[..., None]*e2[None, None, :] + \
                     G3[..., None]*e3[None, None, :]
            pts_flat = pts_3d.reshape(-1, 3)
            # Pra cada grid point, achar k=5 vizinhos no point cloud, IDW
            ds, idxs = tree.query(pts_flat, k=8)
            weights = 1.0 / (ds + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            f_at_grid = (field[idxs] * weights).sum(axis=1)
            # Mask grid points longe do cloud (>1mm)
            dist_to_cloud = ds[:, 0]
            f_at_grid[dist_to_cloud > 1.0] = np.nan
            f_2d = f_at_grid.reshape(n_grid, n_grid)
            ax = axes[row, col]
            im = ax.imshow(f_2d.T, origin='lower',
                           extent=[grid2.min(), grid2.max(),
                                   grid2.min(), grid2.max()],
                           cmap='RdBu_r', vmin=-absmax, vmax=absmax,
                           aspect='equal')
            if row == 0:
                ax.set_title(f"s = {s_t:.2f} mm")
            if col == 0:
                ax.set_ylabel(f"{fname}\n[{-absmax:.3f}, {absmax:.3f}]")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 4:
                plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"ANEUMO case 4 ({tag}): slices canon-style (perp eixo PCA)")
    plt.tight_layout()
    out2 = out_dir / f"slices_{tag}.png"
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved {out2}")

    # =========================================================================
    # FIG 3: streamlines no plano central (passing through bbox center)
    # =========================================================================
    print(f"[{tag}] FIG 3: streamlines plano central")
    # Plano XY central (z = mean_z)
    nx, ny = 80, 80
    x_grid = np.linspace(bbox_min[0], bbox_max[0], nx)
    y_grid = np.linspace(bbox_min[1], bbox_max[1], ny)
    z_center = center[2]
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing='ij')
    pts_grid = np.stack([Xg.ravel(), Yg.ravel(),
                          np.full(Xg.size, z_center)], axis=1)
    ds_g, idxs_g = tree.query(pts_grid, k=8)
    weights_g = 1.0 / (ds_g + 1e-6); weights_g /= weights_g.sum(axis=1, keepdims=True)
    u_g = (u[idxs_g] * weights_g).sum(axis=1).reshape(nx, ny)
    v_g = (v[idxs_g] * weights_g).sum(axis=1).reshape(nx, ny)
    speed_g = np.sqrt(u_g**2 + v_g**2)
    # Mask longe
    far_mask = ds_g[:, 0] > 1.0
    speed_g.flat[far_mask] = np.nan

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(speed_g.T, origin='lower',
                   extent=[bbox_min[0], bbox_max[0],
                           bbox_min[1], bbox_max[1]],
                   cmap='viridis', aspect='equal')
    try:
        ax.streamplot(Xg.T, Yg.T, u_g.T, v_g.T, color='white', density=2.5,
                      linewidth=0.6)
    except Exception as e:
        print(f"  streamline failed: {e}")
    plt.colorbar(im, ax=ax, label='|u_xy| (m/s)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f"ANEUMO case 4 ({tag}): streamlines no plano z = {z_center:.2f} mm")
    out3 = out_dir / f"streamlines_{tag}.png"
    plt.savefig(out3, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved {out3}")

    # =========================================================================
    # FIG 4: comparacao lado-a-lado: idealizada v1 (esquerda) vs real (direita)
    # Apenas se ja temos o v1 streamlines salvo
    # =========================================================================
    v1_streamlines = Path(VOL) / "final/aneurisma_cfd/ns_re300_64_streamlines.png"
    if v1_streamlines.exists():
        print(f"[{tag}] FIG 4: lado-a-lado v1 vs real")
        import matplotlib.image as mpimg
        v1_img = mpimg.imread(str(v1_streamlines))
        real_img = mpimg.imread(str(out3))
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        axes[0].imshow(v1_img); axes[0].axis('off')
        axes[0].set_title('v1 idealizada: cilindro + esfera', fontsize=14)
        axes[1].imshow(real_img); axes[1].axis('off')
        axes[1].set_title(f'v2 anatomia REAL ANEUMO ({tag})', fontsize=14)
        plt.tight_layout()
        out4 = out_dir / f"comparison_v1_v2_{tag}.png"
        plt.savefig(out4, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  saved {out4}")

    wall = t.time() - t0
    print(f"[{tag}] DONE wall={wall:.0f}s")
    return {"tag": tag, "wall": wall, "N": N,
            "bbox": [bbox_min.tolist(), bbox_max.tolist()]}


@app.local_entrypoint()
def main():
    f1 = plot_all.spawn(tag="m002")
    f2 = plot_all.spawn(tag="m003")
    print(f"m002: {f1.get()}")
    print(f"m003: {f2.get()}")
