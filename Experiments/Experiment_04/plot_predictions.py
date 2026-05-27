"""Gerar galeria de figuras para aneurisma.pdf v3.

Lê GT (case_AN4_m002.npz), PINN pred (pinn_pred.npz), Mix pred (mix_pred.npz)
e history JSONs. Produz 20+ figuras estilo gallery em PNG.
"""
import json
from pathlib import Path
import modal

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "matplotlib", "scikit-learn"))
app = modal.App("tcc-aneur-real-plot-v3", image=image)
volume = modal.Volume.from_name("tcc")
VOL = "/data"


@app.function(cpu=8, memory=24576, timeout=30*60, volumes={VOL: volume})
def gallery():
    import time as t
    import numpy as np
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    t0 = t.time()
    out_dir = Path(VOL) / "final/aneurisma_real/figs_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar dados
    gt = np.load(f"{VOL}/final/aneurisma_real/case_AN4_m002.npz")
    xyz = gt['xyz']; u = gt['u']; v = gt['v']; w = gt['w']; p = gt['p']
    speed = np.sqrt(u**2 + v**2 + w**2)
    N = len(xyz)
    bbox_min = xyz.min(0); bbox_max = xyz.max(0)
    center = (bbox_min + bbox_max) / 2

    pinn = np.load(f"{VOL}/final/aneurisma_real/pinn_pred.npz")
    u_pn = pinn['u_pred']; v_pn = pinn['v_pred']; w_pn = pinn['w_pred']; p_pn = pinn['p_pred']
    speed_pn = np.sqrt(u_pn**2 + v_pn**2 + w_pn**2)

    mix = np.load(f"{VOL}/final/aneurisma_real/mix_pred.npz")
    u_mx = mix['u_pred']; v_mx = mix['v_pred']; w_mx = mix['w_pred']; p_mx = mix['p_pred']
    speed_mx = np.sqrt(u_mx**2 + v_mx**2 + w_mx**2)

    pinn_hist = json.loads(Path(f"{VOL}/final/aneurisma_real/pinn_history.json").read_text())
    mix_hist = json.loads(Path(f"{VOL}/final/aneurisma_real/mix_history.json").read_text())

    print(f"Loaded: N={N} pts; PINN MSE={pinn_hist['test_mse']:.4e}, "
          f"Mix MSE={mix_hist['test_mse']:.4e}")

    # Subsample para 3D
    rng = np.random.default_rng(0)
    sub = rng.choice(N, min(10000, N), replace=False)

    # ======== Categoria A: Ground Truth (referência ANEUMO) ========

    # A1: 3D scatter |u| + p
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=speed[sub], cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'GT: |u| (ANEUMO case 4, N={N})')
    plt.colorbar(sc, ax=ax1, label='|u| (m/s)', shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=p[sub], cmap='RdBu_r', s=2, alpha=0.6)
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.set_title('GT: pressão p')
    plt.colorbar(sc2, ax=ax2, label='p (Pa)', shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "A1_gt_3d.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved A1")

    # A2: 3D scatter cada componente u, v, w
    fig = plt.figure(figsize=(18, 6))
    for i, (name, comp) in enumerate(zip(['u','v','w'], [u, v, w])):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        absmax = max(abs(comp.min()), abs(comp.max()))
        sc = ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                        c=comp[sub], cmap='RdBu_r', s=2, alpha=0.6,
                        vmin=-absmax, vmax=absmax)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f'GT: {name} (m/s)')
        plt.colorbar(sc, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "A2_gt_components.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved A2")

    # Plano central z = center[2]: interpolar via k-NN no point cloud
    tree = cKDTree(xyz)
    def interp_field(field, x_target, y_target, z_target, n_grid=100,
                     pad=0.5):
        x_range = np.linspace(bbox_min[0]-pad, bbox_max[0]+pad, n_grid)
        y_range = np.linspace(bbox_min[1]-pad, bbox_max[1]+pad, n_grid)
        Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')
        pts_g = np.stack([Xg.ravel(), Yg.ravel(),
                           np.full(Xg.size, z_target)], axis=1)
        ds, idxs = tree.query(pts_g, k=8)
        weights = 1.0 / (ds + 1e-6); weights /= weights.sum(axis=1, keepdims=True)
        f_g = (field[idxs] * weights).sum(axis=1).reshape(n_grid, n_grid)
        # mask longe
        f_g.flat[ds[:, 0] > 1.0] = np.nan
        return x_range, y_range, f_g

    n_grid = 100
    x_r, y_r, speed_g = interp_field(speed, None, None, center[2], n_grid=n_grid)
    _, _, p_g = interp_field(p, None, None, center[2], n_grid=n_grid)
    _, _, speed_pn_g = interp_field(speed_pn, None, None, center[2], n_grid=n_grid)
    _, _, p_pn_g = interp_field(p_pn, None, None, center[2], n_grid=n_grid)
    _, _, speed_mx_g = interp_field(speed_mx, None, None, center[2], n_grid=n_grid)
    _, _, p_mx_g = interp_field(p_mx, None, None, center[2], n_grid=n_grid)
    extent = [x_r.min(), x_r.max(), y_r.min(), y_r.max()]

    # A3: heatmap 2D |u| plano central GT
    speed_vmax = float(np.nanmax(speed_g))
    p_vmax = float(np.nanmax(np.abs(p_g)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(speed_g.T, origin='lower', extent=extent,
                   cmap='viridis', vmin=0, vmax=speed_vmax)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'GT: |u| no plano z={center[2]:.2f} mm')
    plt.colorbar(im, ax=ax, label='|u| (m/s)')
    plt.tight_layout()
    plt.savefig(out_dir / "A3_gt_plane_speed.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved A3")

    # A4: heatmap 2D p plano central GT
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(p_g.T, origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-p_vmax, vmax=p_vmax)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'GT: p no plano z={center[2]:.2f} mm')
    plt.colorbar(im, ax=ax, label='p (Pa)')
    plt.tight_layout()
    plt.savefig(out_dir / "A4_gt_plane_p.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved A4")

    # A5: Slices 4x5 PCA aligned (GT)
    centered = xyz - xyz.mean(0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary = eigvecs[:, -1]
    e1 = primary
    if abs(e1[2]) < 0.9: e2 = np.cross(e1, [0,0,1])
    else: e2 = np.cross(e1, [1,0,0])
    e2 /= np.linalg.norm(e2); e3 = np.cross(e1, e2)
    s_proj = centered @ primary
    s_targets = np.linspace(s_proj.min() + 0.1*(s_proj.max()-s_proj.min()),
                             s_proj.max() - 0.1*(s_proj.max()-s_proj.min()), 5)
    perp_ext = max(abs((centered @ e2)).max(), abs((centered @ e3)).max())
    grid_local = np.linspace(-perp_ext*1.05, perp_ext*1.05, 50)
    G2, G3 = np.meshgrid(grid_local, grid_local, indexing='ij')

    def slice_field(field, s_t):
        plane_c = xyz.mean(0) + s_t * primary
        pts_3d = plane_c[None,None,:] + G2[...,None]*e2[None,None,:] + G3[...,None]*e3[None,None,:]
        pts_flat = pts_3d.reshape(-1, 3)
        ds, idxs = tree.query(pts_flat, k=8)
        wts = 1.0/(ds+1e-6); wts /= wts.sum(axis=1, keepdims=True)
        f = (field[idxs]*wts).sum(axis=1)
        f[ds[:, 0] > 1.0] = np.nan
        return f.reshape(50, 50)

    def plot_slices_4x5(fields_list, names, suptitle, out_name):
        fig, axes = plt.subplots(4, 5, figsize=(20, 13))
        for row, (name, field) in enumerate(zip(names, fields_list)):
            vmax = float(np.nanpercentile(np.abs(field), 99))
            for col, s_t in enumerate(s_targets):
                f2 = slice_field(field, s_t)
                ax = axes[row, col]
                im = ax.imshow(f2.T, origin='lower',
                                extent=[grid_local.min(), grid_local.max(),
                                        grid_local.min(), grid_local.max()],
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                aspect='equal')
                if row == 0: ax.set_title(f"s={s_t:.2f} mm")
                if col == 0: ax.set_ylabel(f"{name}\n[±{vmax:.3f}]")
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=130, bbox_inches='tight')
        plt.close()
    plot_slices_4x5([u, v, w, p], ['u', 'v', 'w', 'p'],
                     'GT (ANEUMO): slices 4x5 perp eixo PCA',
                     "A5_gt_slices.png")
    print("saved A5")

    # A6: streamlines GT (re-uso da v2)
    nx, ny = 80, 80
    x_grid = np.linspace(bbox_min[0], bbox_max[0], nx)
    y_grid = np.linspace(bbox_min[1], bbox_max[1], ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing='ij')
    pts_grid = np.stack([Xg.ravel(), Yg.ravel(),
                          np.full(Xg.size, center[2])], axis=1)
    ds_g, idxs_g = tree.query(pts_grid, k=8)
    wts_g = 1/(ds_g+1e-6); wts_g /= wts_g.sum(axis=1, keepdims=True)
    def grid_field(field):
        return (field[idxs_g] * wts_g).sum(axis=1).reshape(nx, ny)
    u_grid = grid_field(u); v_grid = grid_field(v)
    spd_grid = np.sqrt(u_grid**2 + v_grid**2)
    spd_grid.flat[ds_g[:, 0] > 1.0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(spd_grid.T, origin='lower',
                   extent=[bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1]],
                   cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='|u_xy| (m/s)')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'GT: streamlines (heatmap |u_xy|) plano z={center[2]:.2f}mm')
    plt.tight_layout()
    plt.savefig(out_dir / "A6_gt_streamlines.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved A6")

    # A7: histogramas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(speed, bins=80, color='steelblue', alpha=0.75)
    axes[0].set_xlabel('|u| (m/s)'); axes[0].set_ylabel('count')
    axes[0].set_title('GT: distribuição |u|')
    axes[0].grid(alpha=0.3)
    axes[1].hist(p, bins=80, color='coral', alpha=0.75)
    axes[1].set_xlabel('p (Pa)'); axes[1].set_ylabel('count')
    axes[1].set_title('GT: distribuição p')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "A7_gt_histograms.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved A7")

    # ======== Categoria B: PINN predictions ========

    # B1: 3D scatter |u| PINN + p PINN
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=speed_pn[sub], cmap='viridis', s=2, alpha=0.6,
                     vmin=0, vmax=speed.max())
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'PINN: |u| (test_mse_uvw={pinn_hist["test_mse_uvw"]:.2e})')
    plt.colorbar(sc, ax=ax1, label='|u| (m/s)', shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=p_pn[sub], cmap='RdBu_r', s=2, alpha=0.6,
                      vmin=p.min(), vmax=p.max())
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.set_title('PINN: p')
    plt.colorbar(sc2, ax=ax2, label='p (Pa)', shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "B1_pinn_3d.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved B1")

    # B3: PINN heatmap plano central
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im0 = axes[0].imshow(speed_pn_g.T, origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=speed_vmax)
    axes[0].set_xlabel('X (mm)'); axes[0].set_ylabel('Y (mm)')
    axes[0].set_title(f'PINN: |u| no plano z={center[2]:.2f} mm')
    plt.colorbar(im0, ax=axes[0], label='|u| (m/s)')
    im1 = axes[1].imshow(p_pn_g.T, origin='lower', extent=extent,
                          cmap='RdBu_r', vmin=-p_vmax, vmax=p_vmax)
    axes[1].set_xlabel('X (mm)'); axes[1].set_ylabel('Y (mm)')
    axes[1].set_title(f'PINN: p no plano z={center[2]:.2f} mm')
    plt.colorbar(im1, ax=axes[1], label='p (Pa)')
    plt.tight_layout()
    plt.savefig(out_dir / "B3_pinn_plane.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved B3")

    # B4: slices 4x5 PINN
    plot_slices_4x5([u_pn, v_pn, w_pn, p_pn], ['u', 'v', 'w', 'p'],
                     'PINN: slices 4x5 perp eixo PCA',
                     "B4_pinn_slices.png")
    print("saved B4")

    # B5: erro absoluto |u_PINN - u_GT|
    err_pn_speed = np.abs(speed_pn - speed)
    err_pn_p = np.abs(p_pn - p)
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=err_pn_speed[sub], cmap='Reds', s=2, alpha=0.6)
    ax1.set_title(f'PINN: erro |Δ|u|| (max={err_pn_speed.max():.3f})')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    plt.colorbar(sc, ax=ax1, shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=err_pn_p[sub], cmap='Reds', s=2, alpha=0.6)
    ax2.set_title(f'PINN: erro |Δp| (max={err_pn_p.max():.3f})')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    plt.colorbar(sc2, ax=ax2, shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "B5_pinn_error.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved B5")

    # B6: histograma erros
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(err_pn_speed, bins=80, color='red', alpha=0.7)
    axes[0].set_xlabel('|Δ|u|| (m/s)'); axes[0].set_ylabel('count')
    axes[0].set_title('PINN: distribuição de erro em |u|')
    axes[0].grid(alpha=0.3)
    axes[1].hist(err_pn_p, bins=80, color='red', alpha=0.7)
    axes[1].set_xlabel('|Δp| (Pa)'); axes[1].set_ylabel('count')
    axes[1].set_title('PINN: distribuição de erro em p')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "B6_pinn_error_hist.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved B6")

    # ======== Categoria C: Mix predictions ========

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=speed_mx[sub], cmap='viridis', s=2, alpha=0.6,
                     vmin=0, vmax=speed.max())
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Mix: |u| (test_mse_uvw={mix_hist["test_mse_uvw"]:.2e})')
    plt.colorbar(sc, ax=ax1, label='|u| (m/s)', shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=p_mx[sub], cmap='RdBu_r', s=2, alpha=0.6,
                      vmin=p.min(), vmax=p.max())
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.set_title('Mix: p')
    plt.colorbar(sc2, ax=ax2, label='p (Pa)', shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "C1_mix_3d.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved C1")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im0 = axes[0].imshow(speed_mx_g.T, origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=speed_vmax)
    axes[0].set_title(f'Mix: |u| plano z={center[2]:.2f}')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(p_mx_g.T, origin='lower', extent=extent,
                          cmap='RdBu_r', vmin=-p_vmax, vmax=p_vmax)
    axes[1].set_title(f'Mix: p plano z={center[2]:.2f}')
    plt.colorbar(im1, ax=axes[1])
    for ax in axes: ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    plt.tight_layout()
    plt.savefig(out_dir / "C2_mix_plane.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved C2")

    plot_slices_4x5([u_mx, v_mx, w_mx, p_mx], ['u', 'v', 'w', 'p'],
                     'MixFunn: slices 4x5 perp eixo PCA',
                     "C3_mix_slices.png")
    print("saved C3")

    err_mx_speed = np.abs(speed_mx - speed)
    err_mx_p = np.abs(p_mx - p)
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                     c=err_mx_speed[sub], cmap='Reds', s=2, alpha=0.6)
    ax1.set_title(f'Mix: erro |Δ|u|| (max={err_mx_speed.max():.3f})')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    plt.colorbar(sc, ax=ax1, shrink=0.6)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                      c=err_mx_p[sub], cmap='Reds', s=2, alpha=0.6)
    ax2.set_title(f'Mix: erro |Δp| (max={err_mx_p.max():.3f})')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    plt.colorbar(sc2, ax=ax2, shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "C4_mix_error.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved C4")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(err_mx_speed, bins=80, color='orange', alpha=0.7)
    axes[0].set_xlabel('|Δ|u|| (m/s)'); axes[0].set_ylabel('count')
    axes[0].set_title('Mix: distribuição de erro em |u|')
    axes[0].grid(alpha=0.3)
    axes[1].hist(err_mx_p, bins=80, color='orange', alpha=0.7)
    axes[1].set_xlabel('|Δp| (Pa)'); axes[1].set_ylabel('count')
    axes[1].set_title('Mix: distribuição de erro em p')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "C5_mix_error_hist.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved C5")

    # ======== Categoria D: Comparações ========

    # D1: 4-painel plano central — GT vs PINN vs Mix vs erro
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    err_pn_g = np.abs(speed_pn_g - speed_g)
    err_mx_g = np.abs(speed_mx_g - speed_g)
    for col, (name, field) in enumerate(zip(
        ['GT', 'PINN', 'Mix', '|Δ_PINN|'],
        [speed_g, speed_pn_g, speed_mx_g, err_pn_g])):
        ax = axes[0, col]
        cmap = 'Reds' if 'Δ' in name else 'viridis'
        vmax = float(np.nanmax(field))
        im = ax.imshow(field.T, origin='lower', extent=extent, cmap=cmap,
                       vmin=0, vmax=vmax)
        ax.set_title(f'{name} |u|', fontsize=12)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    # Linha 2: pressão
    err_pn_p_g = np.abs(p_pn_g - p_g)
    err_mx_p_g = np.abs(p_mx_g - p_g)
    for col, (name, field) in enumerate(zip(
        ['GT', 'PINN', 'Mix', '|Δ_Mix|'],
        [p_g, p_pn_g, p_mx_g, err_mx_g])):
        ax = axes[1, col]
        if 'Δ' in name:
            cmap = 'Reds'
            vmin = 0; vmax = float(np.nanmax(field))
        else:
            cmap = 'RdBu_r'; vmin = -p_vmax; vmax = p_vmax
            if col < 3 and 'u' not in name and 'Δ' not in name:
                pass
        if col == 3:
            im = ax.imshow(field.T, origin='lower', extent=extent,
                            cmap='Reds', vmin=0)
            ax.set_title(f'{name} |u|', fontsize=12)
        else:
            im = ax.imshow(field.T, origin='lower', extent=extent,
                            cmap='RdBu_r', vmin=-p_vmax, vmax=p_vmax)
            ax.set_title(f'{name} p', fontsize=12)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    fig.suptitle('Comparação 4-painel plano central z=39.75mm')
    plt.tight_layout()
    plt.savefig(out_dir / "D1_compare_plane.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved D1")

    # D2: slice central s=0 comparação GT/PINN/Mix/erro
    def make_central_slice_compare():
        fig, axes = plt.subplots(2, 4, figsize=(22, 12))
        slc_gt_u = slice_field(speed, 0)
        slc_pn_u = slice_field(speed_pn, 0)
        slc_mx_u = slice_field(speed_mx, 0)
        slc_gt_p = slice_field(p, 0)
        slc_pn_p = slice_field(p_pn, 0)
        slc_mx_p = slice_field(p_mx, 0)
        u_vmax = max(np.nanmax(slc_gt_u), np.nanmax(slc_pn_u), np.nanmax(slc_mx_u))
        p_vmax_l = max(abs(np.nanmin(slc_gt_p)), abs(np.nanmax(slc_gt_p)))
        for col, (name, field) in enumerate(zip(
            ['GT', 'PINN', 'Mix', '|Δ|'],
            [slc_gt_u, slc_pn_u, slc_mx_u, np.abs(slc_pn_u - slc_gt_u)])):
            ax = axes[0, col]
            cmap = 'Reds' if 'Δ' in name else 'viridis'
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                           vmin=0, vmax=u_vmax if 'Δ' not in name else None)
            ax.set_title(f'{name} |u|')
            plt.colorbar(im, ax=ax)
            ax.set_xticks([]); ax.set_yticks([])
        for col, (name, field) in enumerate(zip(
            ['GT', 'PINN', 'Mix', '|Δ|_Mix'],
            [slc_gt_p, slc_pn_p, slc_mx_p, np.abs(slc_mx_p - slc_gt_p)])):
            ax = axes[1, col]
            if 'Δ' in name:
                im = ax.imshow(field.T, origin='lower', cmap='Reds', vmin=0)
            else:
                im = ax.imshow(field.T, origin='lower', cmap='RdBu_r',
                                vmin=-p_vmax_l, vmax=p_vmax_l)
            ax.set_title(f'{name} p')
            plt.colorbar(im, ax=ax)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Slice central s=0 (perp eixo PCA)')
        plt.tight_layout()
        plt.savefig(out_dir / "D2_compare_central_slice.png", dpi=130,
                     bbox_inches='tight')
        plt.close()
    make_central_slice_compare()
    print("saved D2")

    # D3: convergência
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    h_pn = pinn_hist["history"]
    steps_pn = [h["step"] for h in h_pn]
    lt_pn = [h["loss_train"] for h in h_pn]
    lv_pn = [h["loss_val"] for h in h_pn]
    axes[0].semilogy(steps_pn, lt_pn, 'b-', label='train', alpha=0.7)
    axes[0].semilogy(steps_pn, lv_pn, 'g-', label='val', linewidth=2)
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('MSE loss (normalizada)')
    axes[0].set_title(f'PINN ({pinn_hist["n_par"]} params)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    h_mx = mix_hist["history"]
    steps_mx = [h["step"] for h in h_mx]
    lt_mx = [h["loss_train"] for h in h_mx]
    lv_mx = [h["loss_val"] for h in h_mx]
    axes[1].semilogy(steps_mx, lt_mx, 'orange', label='train', alpha=0.7)
    axes[1].semilogy(steps_mx, lv_mx, 'r-', label='val', linewidth=2)
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel('MSE loss (normalizada)')
    axes[1].set_title(f'MixFunn ({mix_hist["n_par"]} params)')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "D3_convergence.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved D3")

    # D4: scatter pred vs gt 1:1
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for col, (name, field_gt, field_pn, field_mx) in enumerate(zip(
        ['u', 'v', 'w', 'p'],
        [u, v, w, p], [u_pn, v_pn, w_pn, p_pn], [u_mx, v_mx, w_mx, p_mx])):
        # subsample for speed
        ax = axes[0, col]
        ax.scatter(field_gt[sub], field_pn[sub], s=1, alpha=0.4, c='b')
        lo, hi = min(field_gt.min(), field_pn.min()), max(field_gt.max(), field_pn.max())
        ax.plot([lo, hi], [lo, hi], 'k-', linewidth=1)
        ax.set_xlabel(f'GT {name}'); ax.set_ylabel(f'PINN {name}')
        ax.set_title(f'PINN {name}')
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.scatter(field_gt[sub], field_mx[sub], s=1, alpha=0.4, c='orange')
        ax.plot([lo, hi], [lo, hi], 'k-', linewidth=1)
        ax.set_xlabel(f'GT {name}'); ax.set_ylabel(f'Mix {name}')
        ax.set_title(f'Mix {name}')
        ax.grid(alpha=0.3)
    fig.suptitle('Predição vs Ground Truth (1:1)')
    plt.tight_layout()
    plt.savefig(out_dir / "D4_pred_vs_gt.png", dpi=130, bbox_inches='tight')
    plt.close()
    print("saved D4")

    wall = t.time() - t0
    print(f"DONE wall={wall:.0f}s, figs in {out_dir}")
    return {"wall": wall, "n_figs": len(list(out_dir.glob("*.png")))}


@app.local_entrypoint()
def main():
    r = gallery.remote()
    print(r)
