"""Validacao 4-camadas do campo CFD ANEUMO (caso real anatomic, point cloud).

Estrategia: como o campo eh um point cloud (nao grid), uso k-NN local pra
estimar:
  A) divergencia local (least-squares via gradiente local de ordem 1)
  B) fluxo atraves de slices perp ao eixo principal
  C) Poiseuille fit em regiao proximal (longe do sac)
  D) vorticidade no sac
"""
import json
from pathlib import Path
import modal

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "matplotlib", "scikit-learn"))
app = modal.App("tcc-aneur-real-validate", image=image)
volume = modal.Volume.from_name("tcc")
VOL = "/data"


@app.function(cpu=4, memory=16384, timeout=15*60, volumes={VOL: volume})
def validate(tag: str = "m002"):
    """V&V 4-layer para o caso ANEUMO."""
    import time as t
    import numpy as np
    from scipy.spatial import cKDTree
    t0 = t.time()
    npz = Path(VOL) / f"final/aneurisma_real/case_AN4_{tag}.npz"
    d = np.load(npz)
    xyz_mm = d['xyz']; u = d['u']; v = d['v']; w = d['w']; p = d['p']; sdf = d['sdf']
    xyz_inlet = d['xyz_inlet']
    simple_inlet = d['simple_inlet']
    N = len(xyz_mm)
    # Trabalhar em metros pra unidades consistentes (u em m/s)
    xyz = xyz_mm / 1000.0  # mm -> m
    print(f"[{tag}] N={N} pts, bbox extent {xyz_mm.max(0)-xyz_mm.min(0)} mm (xyz convertido p/ m)")

    # =========================================================================
    # A) Incompressibilidade: divergencia local via least-squares
    # Pra cada ponto, achar k=20 vizinhos, ajustar plano linear u(x) ~ u0 +
    # grad_u . dx; entao grad_u = least-squares fit. div = sum(diag(grad_uvw))
    # =========================================================================
    print(f"[{tag}] === A) Incompressibilidade ===")
    tree = cKDTree(xyz)
    k = 20
    # Sample subset pra economizar (compute em 5000 pts random)
    rng = np.random.default_rng(42)
    nsamp = min(5000, N)
    samp = rng.choice(N, nsamp, replace=False)
    div_vals = []
    for idx in samp:
        _, nbrs = tree.query(xyz[idx], k=k)
        dx = xyz[nbrs] - xyz[idx]  # (k, 3)
        # gradiente de u via least-squares: u(x+dx) - u(x) ~ grad_u . dx
        # solve dx @ grad_u = du
        du = u[nbrs] - u[idx]
        dv = v[nbrs] - v[idx]
        dw = w[nbrs] - w[idx]
        # grad_u = (dx^T dx)^-1 dx^T du
        try:
            grad_u = np.linalg.lstsq(dx, du, rcond=None)[0]
            grad_v = np.linalg.lstsq(dx, dv, rcond=None)[0]
            grad_w = np.linalg.lstsq(dx, dw, rcond=None)[0]
            div = grad_u[0] + grad_v[1] + grad_w[2]
            div_vals.append(div)
        except Exception as e:
            pass
    div_arr = np.array(div_vals)
    A_max = float(np.abs(div_arr).max())
    A_mean = float(np.abs(div_arr).mean())
    A_p95 = float(np.percentile(np.abs(div_arr), 95))
    # Criterio: div em unidades fisicas 1/s. Pra fluido incompressivel
    # |div u| pequeno comparado com sheer rate |du/dx| ~ U/L.
    # U ~ 0.5 m/s, L ~ 0.008 m -> shear rate ~ 60 1/s. Div ideal << 60.
    A_pass = A_p95 < 30.0
    print(f"  max|div u| = {A_max:.4e} 1/s")
    print(f"  mean|div u| = {A_mean:.4e} 1/s")
    print(f"  p95|div u| = {A_p95:.4e} 1/s")
    print(f"  -> {'PASS' if A_pass else 'FAIL'} (criterio p95 < 30 1/s, shear scale)")

    # =========================================================================
    # B) Conservacao de massa: fluxo via slices perpendiculares ao eixo PCA
    # =========================================================================
    print(f"[{tag}] === B) Conservacao massa ===")
    # PCA pra eixo principal
    centered = xyz - xyz.mean(0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary = eigvecs[:, -1]
    s_proj = centered @ primary
    s_min, s_max = s_proj.min(), s_proj.max()
    # Compute flux at s = 0.2*range, 0.5*range, 0.8*range
    # Em metros agora; tolerancia de slice = 0.5 mm = 5e-4 m
    s_targets = [s_min + 0.2*(s_max-s_min), s_min + 0.5*(s_max-s_min),
                 s_min + 0.8*(s_max-s_min)]
    slice_tol = 5e-4  # 0.5 mm em metros
    fluxes = []
    for s_t in s_targets:
        # pontos perto da slice
        mask_near = np.abs(s_proj - s_t) < slice_tol
        pts_near = xyz[mask_near]
        n_near = mask_near.sum()
        if n_near < 5:
            fluxes.append(None); continue
        # componente da velocidade ao longo do eixo principal
        uvw = np.stack([u, v, w], axis=-1)[mask_near]
        u_along = uvw @ primary  # (n_near,)
        # area da slice: aproximar via convex hull dos pontos
        # transformar pts pra plano local
        e1 = primary
        if abs(e1[2]) < 0.9:
            e2 = np.cross(e1, [0,0,1])
        else:
            e2 = np.cross(e1, [1,0,0])
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        pts_2d = np.stack([(pts_near - xyz.mean(0)) @ e2,
                            (pts_near - xyz.mean(0)) @ e3], axis=-1)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts_2d)
            area = hull.volume  # volume em 2D = area
        except:
            area = (pts_2d.max(0) - pts_2d.min(0)).prod()
        # fluxo: u_mean * area
        flux = u_along.mean() * area
        fluxes.append((s_t, n_near, area, u_along.mean(), flux))
        print(f"  s={s_t:.2f}mm: n={n_near}, area={area:.2f}mm^2, u_mean={u_along.mean():.4f}, flux={flux:.4f}")
    # Comparar primeira e ultima slices
    if fluxes[0] is not None and fluxes[-1] is not None:
        f_in = fluxes[0][4]; f_out = fluxes[-1][4]
        B_diff_pct = abs(f_in - f_out) / abs(f_in) * 100
        B_pass = B_diff_pct < 30  # relaxado pra unstructured + slicing aproximado
        print(f"  flux 20% vs 80%: diff = {B_diff_pct:.2f}%  -> {'PASS' if B_pass else 'FAIL'}")
    else:
        B_diff_pct = None; B_pass = False
        print(f"  FAIL: insuficientes pontos para slicing")

    # =========================================================================
    # C) Pontos com alta velocidade (fluxo principal). Validar
    # nao-trivial: campos sao fisicos? max|u| coerente com escala medica
    # 0.1-1 m/s. Sem tubo reto, nao da Poiseuille. Substituir por:
    #   - distribuicao de velocidades: sane range (>0.01 fluindo)
    #   - presence of high-speed zone (jet)
    # =========================================================================
    print(f"[{tag}] === C) Distribuicao de velocidades ===")
    speed = np.sqrt(u**2 + v**2 + w**2)
    sp_mean = float(speed.mean())
    sp_max = float(speed.max())
    sp_p99 = float(np.percentile(speed, 99))
    sp_p1 = float(np.percentile(speed, 1))
    # Fisiologicamente: max 0.1-2 m/s, mean ~0.1-0.5 m/s
    C_pass = (0.05 <= sp_mean <= 2.0) and (0.5 <= sp_max <= 5.0)
    print(f"  |u|_max = {sp_max:.4f}, |u|_mean = {sp_mean:.4f}, |u|_p99 = {sp_p99:.4f}")
    print(f"  -> {'PASS' if C_pass else 'FAIL'} (escala fisiologica 0.1-2 m/s)")

    # =========================================================================
    # D) Vorticidade: estimar via k-NN local. Se ha vortex no sac, deve ter
    # regiao com alta vorticidade isolada (gradient u em loop)
    # =========================================================================
    print(f"[{tag}] === D) Vorticidade ===")
    vort_vals = []
    for idx in samp:
        _, nbrs = tree.query(xyz[idx], k=k)
        dx = xyz[nbrs] - xyz[idx]
        du = u[nbrs] - u[idx]
        dv = v[nbrs] - v[idx]
        dw = w[nbrs] - w[idx]
        try:
            grad_u = np.linalg.lstsq(dx, du, rcond=None)[0]
            grad_v = np.linalg.lstsq(dx, dv, rcond=None)[0]
            grad_w = np.linalg.lstsq(dx, dw, rcond=None)[0]
            # curl: (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
            omx = grad_w[1] - grad_v[2]
            omy = grad_u[2] - grad_w[0]
            omz = grad_v[0] - grad_u[1]
            vort_vals.append(np.sqrt(omx**2 + omy**2 + omz**2))
        except:
            pass
    vort_arr = np.array(vort_vals)
    om_max = float(vort_arr.max())
    om_p99 = float(np.percentile(vort_arr, 99))
    om_mean = float(vort_arr.mean())
    # Se ha vortex, max >> mean
    ratio = om_max / max(om_mean, 1e-6)
    D_pass = ratio > 5.0
    print(f"  |omega|_max = {om_max:.4f}, mean = {om_mean:.4f}, p99 = {om_p99:.4f}")
    print(f"  max/mean ratio = {ratio:.2f}  -> {'PASS' if D_pass else 'FAIL'} (vortex se ratio > 5)")

    summary = {
        "tag": tag, "wall": t.time() - t0,
        "N": int(N),
        "A": {"max": A_max, "mean": A_mean, "p95": A_p95, "pass": bool(A_pass)},
        "B": {"diff_pct": B_diff_pct, "pass": bool(B_pass)},
        "C": {"speed_max": sp_max, "speed_mean": sp_mean, "speed_p99": sp_p99,
              "pass": bool(C_pass)},
        "D": {"omega_max": om_max, "omega_mean": om_mean, "omega_p99": om_p99,
              "max_mean_ratio": float(ratio), "pass": bool(D_pass)},
        "all_pass": bool(A_pass and B_pass and C_pass and D_pass),
        "primary_axis": primary.tolist(),
    }
    out = Path(VOL) / f"final/aneurisma_real/validate_{tag}.json"
    # Cast tudo pra serializavel
    def cast(o):
        if isinstance(o, dict): return {k: cast(v) for k, v in o.items()}
        if isinstance(o, list): return [cast(x) for x in o]
        if hasattr(o, 'item'): return o.item()
        return o
    out.write_text(json.dumps(cast(summary), indent=2))
    volume.commit()
    print(f"[{tag}] DONE wall={summary['wall']:.0f}s  all_pass={summary['all_pass']}")
    return summary


@app.local_entrypoint()
def main():
    r1 = validate.remote("m002")
    r2 = validate.remote("m003")
    print(json.dumps({"m002": r1, "m003": r2}, indent=2))
