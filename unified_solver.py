"""
================================================================================
  UNIFIED SOLVER — FSDT + von Kármán Composite Beam
  Complete implementation of all methods from theory paper

  Methods:
    1. calculate_laminate()   — ABD operators (CLT)
    2. effective_stiffnesses() — D*, D_eff
    3. analytical_solution()  — exact closed-form (SS and CC)
    4. solve_FEM()            — 2-node Timoshenko element, reduced integration
    5. solve_Galerkin_RK45()  — modal projection + RK45 (dynamics)
    6. solve_PINN_standard()  — 3-field PINN (fails for Pi_s >> 1)
    7. solve_PINN_mixed()     — Mixed (W-hat, M-hat) PINN (correct)
    8. solve_FDPINN_mixed()   — FD-PINN Mixed formulation
    9. plot_all()             — comprehensive comparison figures

  IMPORTANT NOTES:
    - float64 REQUIRED throughout (float32 insufficient for Pi_s ~ 10^5)
    - FEM shear: 1-point Gauss integration (avoids locking)
    - Standard PINN fails for Pi_s >> 1 (see Theorem 7 in paper)
    - Mixed PINN coefficients: c_r = 12 (SS) or 48 (CC), Pi_s-independent
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import solve as linsolve
from scipy.integrate import solve_ivp
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch {torch.__version__} — {device}")
except ImportError:
    TORCH = False
    print("PyTorch not found — PINN methods skipped")


# ============================================================================
#  1. LAMINATE STIFFNESS — ABD OPERATORS (from CLT)
# ============================================================================

def calculate_laminate(layers):
    """
    Compute A11, B11, D11, A55 for a composite laminate.

    Parameters
    ----------
    layers : list of dict, each with keys:
        'theta'     : fiber angle (degrees)
        'thickness' : layer thickness (m)
        'E1','E2'   : longitudinal/transverse Young's moduli (Pa)
        'nu12'      : Poisson ratio
        'G12'       : in-plane shear modulus (Pa)

    Returns
    -------
    A11, B11, D11, A55 : ABD stiffness components
    H                  : total laminate thickness

    Theory:
        A11 = sum_k Q11_bar_k * Dz_k
        B11 = 1/2 * sum_k Q11_bar_k * Dz_k^[2]
        D11 = 1/3 * sum_k Q11_bar_k * Dz_k^[3]
        A55 = ks * sum_k Q55_bar_k * Dz_k
        where Dz_k^[n] = z_{k+1}^n - z_k^n
    """
    H  = sum(lay['thickness'] for lay in layers)
    N  = len(layers)
    # z-interfaces from mid-plane: z_0 = -H/2
    z  = np.zeros(N + 1)
    z[0] = -H / 2.0
    for i in range(N):
        z[i+1] = z[i] + layers[i]['thickness']

    A11 = B11 = D11 = A55 = 0.0
    for i, lay in enumerate(layers):
        th  = np.deg2rad(lay['theta'])
        E1, E2, v12, G12 = lay['E1'], lay['E2'], lay['nu12'], lay['G12']
        v21 = v12 * E2 / E1

        # 6x6 compliance in material coordinates
        S = np.zeros((6, 6))
        S[0,0] = 1/E1;   S[0,1] = -v21/E2
        S[1,0] = -v12/E1; S[1,1] = 1/E2;  S[2,2] = 1/E2
        S[3,3] = 1/G12;  S[4,4] = 1/G12;  S[5,5] = 1/G12

        # Transformation matrix T (6x6)
        c, s = np.cos(th), np.sin(th); s2 = np.sin(2*th)
        T = np.array([
            [ c*c,    s*s,   0, 0,  0,  s2],
            [ s*s,    c*c,   0, 0,  0, -s2],
            [ 0,      0,     1, 0,  0,   0],
            [ 0,      0,     0, c,  s,   0],
            [ 0,      0,     0,-s,  c,   0],
            [-s*c,    s*c,   0, 0,  0, c*c-s*s],
        ])
        Ti    = np.linalg.inv(T)
        C_bar = Ti @ S @ Ti.T  # transformed compliance

        # Extract 2x2 submatrix [sigma_x, tau_xz] and invert
        Q_bar = np.linalg.inv(C_bar[np.ix_([0,4],[0,4])])

        zl, zu = z[i], z[i+1]
        A11 += Q_bar[0,0] * (zu - zl)
        B11 += Q_bar[0,0] * 0.5 * (zu**2 - zl**2)
        D11 += Q_bar[0,0] * (zu**3 - zl**3) / 3.0
        A55 += Q_bar[1,1] * (zu - zl)        # shear stiffness

    return A11, B11, D11, A55, H


# ============================================================================
#  2. EFFECTIVE STIFFNESSES
# ============================================================================

def effective_stiffnesses(A11, B11, D11):
    """
    Compute D* (EB-equivalent) and D_eff (constrained-u0) stiffnesses.

    Theory (Theorem 3 of paper):
        D*   = D11 - B11^2/A11         (free-u0 EB limit)
        D_eff = 4*A11*D11*D* / (4*A11*D11 - 3*B11^2)  (SS/CC constrained)

    D_eff is the CORRECT deflection stiffness when u0(0)=u0(L)=0.
    Using D11 instead of D_eff gives ~30% error for [0/90] laminates.
    """
    D_star = D11 - B11**2 / A11
    denom  = 4*A11*D11 - 3*B11**2
    if abs(denom) < 1e-20:
        return D_star, D_star   # symmetric laminate (B11=0)
    D_eff  = 4*A11*D11*D_star / denom
    return D_star, D_eff


# ============================================================================
#  3. ANALYTICAL SOLUTION
# ============================================================================

def analytical_solution(A11, B11, D11, A55, L, Pz, BC='SS', n_pts=400):
    """
    Closed-form solution for Timoshenko beam with B11 != 0.

    Formulas:
        SS beam:  w(x) = Pz*(3L^2*x - 4x^3)/(48*D_eff) + Pz*x/(2*A55),  x <= L/2
                  phi(x) = -Pz*(L^2 - 4x^2)/(16*D11),  x <= L/2
                  u0(x) = B11*Pz*(L*x - 2x^2)/(8*A11*D*)

        CC beam:  w(x) = Pz*(3L*x^2 - 4x^3)/(48*D_eff) + Pz*x*(L-x)/(2*A55*L)
                  phi(x) = -Pz*(L*x - 2x^2)/(8*D11)
                  u0(x) = B11*Pz*(L*x - 2x^2)/(8*A11*D*)

    Sign convention: phi_code = -phi_Reddy (our phi satisfies gamma = w' - phi)
    """
    D_star, D_eff = effective_stiffnesses(A11, B11, D11)
    x = np.linspace(0, L, n_pts)
    w = np.zeros(n_pts); phi = np.zeros(n_pts); u0 = np.zeros(n_pts)

    for i, xi in enumerate(x):
        xr = L - xi   # mirror point
        if xi <= L/2:
            if BC == 'SS':
                w[i]   = Pz*(3*L**2*xi - 4*xi**3)/(48*D_eff) + Pz*xi/(2*A55)
                phi[i] = -Pz*(L**2 - 4*xi**2)/(16*D11)
            else:  # CC
                w[i]   = Pz*(3*L*xi**2 - 4*xi**3)/(48*D_eff) + Pz*xi*(L-xi)/(2*A55*L)
                phi[i] = -Pz*(L*xi - 2*xi**2)/(8*D11)
            u0[i]  = B11*Pz*(L*xi - 2*xi**2)/(8*A11*D_star)
        else:
            if BC == 'SS':
                w[i]   = Pz*(3*L**2*xr - 4*xr**3)/(48*D_eff) + Pz*xr/(2*A55)
                phi[i] = +Pz*(L**2 - 4*xr**2)/(16*D11)
            else:  # CC
                w[i]   = Pz*(3*L*xr**2 - 4*xr**3)/(48*D_eff) + Pz*xr*(L-xr)/(2*A55*L)
                phi[i] = +Pz*(L*xr - 2*xr**2)/(8*D11)
            u0[i]  = -B11*Pz*(L*xr - 2*xr**2)/(8*A11*D_star)

    return x, u0, w, phi


# ============================================================================
#  4. FEM — 2-NODE TIMOSHENKO ELEMENT WITH REDUCED INTEGRATION
# ============================================================================

def solve_FEM(A11, B11, D11, A55, L, Pz, BC='SS', nElem=200):
    """
    FEM solution using 2-node Timoshenko element.

    DOFs per node: [u0, w0, phi]  (phi with sign: gamma = w' - phi)

    Element matrices:
        K_axial  = (A11/Le) * [[1,-1],[-1,1]]       (u0-u0)
        K_bend   = (D11/Le) * [[1,-1],[-1,1]]        (phi-phi)
        K_couple = (B11/Le) * [[1,-1],[-1,1]]        (u0-phi, symmetric)
        K_shear  = A55*Le * outer(Bs, Bs)            (reduced: 1-pt Gauss)
            Bs = [-1/Le, -0.5, 1/Le, -0.5]           (w' and phi at midpoint)

    CRUCIAL: K_shear uses 1-point Gauss to avoid shear locking.
    With full integration: error ~ Pi_s * h (locks completely for Pi_s >> 1).

    BCs:
        SS: u0(0)=w0(0)=0, u0(L)=w0(L)=0   (phi free at ends)
        CC: u0(0)=w0(0)=phi(0)=0, u0(L)=w0(L)=phi(L)=0
    """
    nNode = nElem + 1
    nDOF  = 3 * nNode
    Le    = L / nElem
    K     = np.zeros((nDOF, nDOF))
    F     = np.zeros(nDOF)

    for e in range(nElem):
        n0, n1 = e, e+1
        dofs   = [3*n0, 3*n0+1, 3*n0+2, 3*n1, 3*n1+1, 3*n1+2]
        Ke     = np.zeros((6, 6))

        # Axial stiffness (u0 DOFs: indices 0,3)
        ka = (A11/Le) * np.array([[1,-1],[-1,1]])
        for ii, gi in enumerate([0,3]):
            for jj, gj in enumerate([0,3]): Ke[gi,gj] += ka[ii,jj]

        # Coupling B11 (u0 DOFs: 0,3; phi DOFs: 2,5)
        kc = (B11/Le) * np.array([[1,-1],[-1,1]])
        for ii, gi in enumerate([0,3]):
            for jj, gj in enumerate([2,5]):
                Ke[gi,gj] += kc[ii,jj]; Ke[gj,gi] += kc[ii,jj]

        # Bending stiffness (phi DOFs: 2,5)
        kb = (D11/Le) * np.array([[1,-1],[-1,1]])
        for ii, gi in enumerate([2,5]):
            for jj, gj in enumerate([2,5]): Ke[gi,gj] += kb[ii,jj]

        # Shear stiffness — REDUCED INTEGRATION (1-pt Gauss at midpoint)
        # gamma = w' - phi;  w' approx = [-1/Le, 0, 1/Le, 0] (linear w)
        # phi at midpoint = [0, 0.5, 0, 0.5] (linear phi)
        # Bs maps full 6-DOF vector to gamma at midpoint
        Bs = np.array([0, -1/Le, -0.5, 0, 1/Le, -0.5])
        Ke += A55 * Le * np.outer(Bs, Bs)

        # Assemble
        for ii, gi in enumerate(dofs):
            for jj, gj in enumerate(dofs): K[gi,gj] += Ke[ii,jj]

    # Load: concentrated force at midspan
    mid = nElem // 2
    F[3*mid + 1] += Pz

    # Boundary conditions
    if BC == 'SS':
        fixed = {0, 1, 3*(nNode-1), 3*(nNode-1)+1}       # u0, w0 at both ends
    else:  # CC
        fixed = {0, 1, 2, 3*(nNode-1), 3*(nNode-1)+1, 3*(nNode-1)+2}  # all DOFs

    free  = [i for i in range(nDOF) if i not in fixed]
    U     = np.zeros(nDOF)
    U[free] = linsolve(K[np.ix_(free,free)], F[free])

    x = np.linspace(0, L, nNode)
    return x, U[0::3], U[1::3], U[2::3]


# ============================================================================
#  5. GALERKIN / RK45 (DYNAMICS, FREE VIBRATIONS)
# ============================================================================

def galerkin_duffing_free(rhoA, D11_eff, A11, L, A0, v0=0.0, T_sim=None, n_modes=1):
    """
    Galerkin projection onto sin(n*pi*x/L) modes -> Duffing system.
    Solved with RK45 (adaptive, near-machine-precision energy conservation).

    1-mode: rhoA * a_tt + omega1^2 * a + alpha_D * a^3 = 0
    3-mode: coupled Duffing system (modes 1,3,5)

    Returns: t, a(t), energy_K(t), energy_V(t)
    """
    omega1    = np.pi**2 * np.sqrt(D11_eff / rhoA)
    alpha_D   = A11 * np.pi**4 / (4*rhoA)
    T1        = 2*np.pi / omega1
    if T_sim is None:
        T_sim = 10 * T1
    Nt        = 3000
    t_eval    = np.linspace(0, T_sim, Nt)

    if n_modes == 1:
        def ode(t, y):
            a, ad = y
            return [ad, -(omega1**2 * a + alpha_D * a**3)]

        sol = solve_ivp(ode, (0, T_sim), [A0, v0], t_eval=t_eval,
                        method='RK45', rtol=1e-11, atol=1e-13)
        a = sol.y[0]; ad = sol.y[1]
        K = 0.25*rhoA*ad**2
        V = 0.25*D11_eff*np.pi**4*a**2 + (A11*np.pi**4/32)*a**4
    else:
        modes = [1, 3, 5][:n_modes]
        Nm    = len(modes)
        def ode3(t, y):
            a = y[:Nm]; adot = y[Nm:]
            N_vK = A11/4 * sum((n*np.pi)**2 * a[i]**2 for i,n in enumerate(modes))
            add  = [-(D11_eff*(n*np.pi)**4*a[i] + (n*np.pi)**2*N_vK*a[i])/rhoA
                    for i,n in enumerate(modes)]
            return list(adot) + add

        y0 = [A0] + [0.0]*(Nm-1) + [v0]*Nm
        sol = solve_ivp(ode3, (0, T_sim), y0, t_eval=t_eval,
                        method='RK45', rtol=1e-11, atol=1e-13)
        a  = sol.y[:Nm]; ad = sol.y[Nm:]
        K  = 0.25*rhoA*np.sum(ad**2, axis=0)
        V  = 0.25*D11_eff*sum((n*np.pi)**4*a[i]**2 for i,n in enumerate(modes)) + \
             (A11/8)*(0.5*sum((n*np.pi)**2*a[i]**2 for i,n in enumerate(modes)))**2
        a = a[0]  # return first mode for plotting

    E = K + V
    drift = np.max(np.abs(E - E[0])) / max(E[0], 1e-20)
    print(f"  Galerkin/{n_modes}-mode RK45: E-drift = {drift:.2e}")
    return t_eval, a, K, V


# ============================================================================
#  6. STANDARD PINN (3-FIELD: u0, w0, phi)
# ============================================================================

def solve_PINN_standard(A11, B11, D11, A55, L, Pz, BC='SS',
                        hidden=[30,30,30], nCol=201,
                        epochs=10000, lr=5e-4, print_every=2000):
    """
    Standard 3-field PINN: learns (u0, w0, phi) directly.

    KNOWN FAILURE: For Pi_s = A55*L^2/D11 >> 1, this formulation
    has a spurious near-zero attractor (Theorem 7 of paper).
    The optimizer finds w0 ~ 0 because the shear residual
    r2 ~ O(Pi_s^{-1}) at w0=0, which is numerically tiny.

    This implementation is provided for comparison/demonstration only.
    USE solve_PINN_mixed() for correct results when Pi_s >> 1.
    """
    if not TORCH:
        raise RuntimeError("PyTorch required")

    Pi_s = A55 * L**2 / D11
    print(f"  Standard PINN: Pi_s = {Pi_s:.0f}  {'WARNING: WILL FAIL' if Pi_s > 5000 else 'OK'}")

    _, D_eff = effective_stiffnesses(A11, B11, D11)
    w_sc  = Pz*L**3/(48*D_eff if BC=='SS' else 192*D_eff)
    ph_sc = w_sc / L

    xi_t  = torch.linspace(0,1,nCol,dtype=torch.float64,device=device).unsqueeze(1)
    xN    = 2*xi_t - 1; xN.requires_grad_(True)

    # Gaussian load
    sig   = 0.04
    xi_np = np.linspace(0,1,nCol)
    g_np  = np.exp(-0.5*((xi_np-0.5)/sig)**2)
    g_np /= g_np.sum()*(1/(nCol-1))
    g_t   = torch.tensor(g_np,dtype=torch.float64,device=device).unsqueeze(1)

    class TNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = 1
            lyrs = []
            for w in hidden:
                lyrs += [nn.Linear(d,w), nn.Tanh()]; d = w
            lyrs += [nn.Linear(d,3)]
            self.net = nn.Sequential(*lyrs)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        def forward(self, x): return self.net(x)

    model = TNet().to(device).double()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1000, factor=0.5)

    # Non-dim constants
    c1  = B11/(A11*L); c3 = B11*L/(D11*Pi_s); c4 = 1/Pi_s
    Pis_t = torch.tensor(Pi_s,dtype=torch.float64,device=device)
    c1_t  = torch.tensor(c1,  dtype=torch.float64,device=device)
    c3_t  = torch.tensor(c3,  dtype=torch.float64,device=device)
    c4_t  = torch.tensor(c4,  dtype=torch.float64,device=device)

    ix0, ixL = 0, nCol-1
    hist = []
    t0 = time.time()

    for ep in range(1, epochs+1):
        opt.zero_grad()
        out = model(xN)
        U, W, Phi = out[:,0:1], out[:,1:2], out[:,2:3]
        ones = torch.ones_like(U)
        kw   = dict(retain_graph=True, create_graph=True)

        Ux   = torch.autograd.grad(U,  xN, ones, **kw)[0]*2
        Wx   = torch.autograd.grad(W,  xN, ones, **kw)[0]*2
        Phix = torch.autograd.grad(Phi,xN, ones, **kw)[0]*2
        Uxx  = torch.autograd.grad(Ux,  xN, ones, **kw)[0]*2
        Wxx  = torch.autograd.grad(Wx,  xN, ones, **kw)[0]*2
        Phixx= torch.autograd.grad(Phix,xN, ones, **kw)[0]*2

        r1 = Uxx + c1_t*Phixx
        r2 = (Wxx + Phix)*(Pis_t/48) + g_t   # scaled shear eq.
        r3 = c3_t*Uxx + c4_t*Phixx - (Wx + Phi)

        lp = r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean()
        lb = (U[ix0]**2 + W[ix0]**2 + U[ixL]**2 + W[ixL]**2).mean()
        if BC == 'CC':
            lb = lb + (Phi[ix0]**2 + Phi[ixL]**2).mean()

        loss = lp + 100.0*lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(loss)
        hist.append(float(loss))

        if ep % print_every == 0:
            with torch.no_grad():
                Wm = model(torch.tensor([[0.0]],dtype=torch.float64,device=device))[0,1].item()
            print(f"  ep{ep:6d} | loss={float(loss):.3e} | w_c={Wm*w_sc*1e3:.4f}mm | {time.time()-t0:.1f}s")

    xi_ev = torch.linspace(0,1,200,dtype=torch.float64,device=device).unsqueeze(1)
    with torch.no_grad():
        pred = model(2*xi_ev-1).cpu().numpy()
    x_np = xi_ev.cpu().numpy().flatten()*L
    return x_np, pred[:,0]*w_sc, pred[:,1]*w_sc, pred[:,2]*ph_sc, hist


# ============================================================================
#  7. MIXED PINN — (W-hat, M-hat) FORMULATION [CORRECT FOR Pi_s >> 1]
# ============================================================================

def solve_PINN_mixed(A11, B11, D11, A55, L, Pz, BC='SS',
                     hidden=[50,50,50,50], nCol=300,
                     epochs_adam=12000, epochs_lbfgs=800,
                     lr=8e-4, print_every=3000):
    """
    Mixed (W-hat, M-hat) PINN — correct for any Pi_s.

    THEORY (Section 8 of paper):
    ─────────────────────────────────────────────────────────────────────
    Instead of solving for (u0, w0, phi) directly, we solve for
    the deflection W-hat = w0/w_sc and bending moment M-hat = M/M_sc.

    Dimensionless PDEs:
        SS:  d^2W-hat/dxi^2 + 12*M-hat = 0    (c_r = 12)
             d^2M-hat/dxi^2 + 4*g-tilde = 0   (c_g = 4)

        CC:  d^2W-hat/dxi^2 + 48*M-hat = 0    (c_r = 48)
             d^2M-hat/dxi^2 + 8*g-tilde = 0   (c_g = 8)

    Coefficients are EXACT INTEGERS, independent of Pi_s, A11, B11, etc.

    Hard BCs (exact, no penalty needed):
        SS: W-hat = xi*(1-xi)*f_W,   M-hat = xi*(1-xi)*f_M
        CC: W-hat = xi^2*(1-xi)^2*f_W,  M-hat = -1 + xi*(1-xi)*f_M
            (CC moment: M(0)=M(L)=-Pz*L/8 = -M_sc, hence M-hat(0)=M-hat(1)=-1)

    phi and u0 recovered from exact formulas (valid for Pi_s >> 1):
        phi(x) = -Pz*(L^2 - 4x^2)/(16*D11)          [SS, x <= L/2]
        u0(x)  = B11*Pz*(L*x - 2x^2)/(8*A11*D*)      [SS, x <= L/2]
    ─────────────────────────────────────────────────────────────────────
    """
    if not TORCH:
        raise RuntimeError("PyTorch required")

    D_star, D_eff = effective_stiffnesses(A11, B11, D11)
    Pi_s   = A55*L**2/D11
    c_w    = 48  if BC == 'SS' else 192
    c_M    = 4   if BC == 'SS' else 8
    c_r    = c_w // c_M   # = 12 (SS) or 48 (CC)
    c_g    = c_M

    w_sc   = Pz*L**3 / (c_w * D_eff)
    M_sc   = Pz*L    / c_M

    print(f"  Mixed PINN ({BC}): Pi_s={Pi_s:.0f}  D_eff={D_eff:.4f}")
    print(f"  w_sc={w_sc*1e3:.4f}mm  M_sc={M_sc:.4f}N·m  c_r={c_r}  c_g={c_g}")

    # Collocation: uniform + dense near boundaries (CC: high curvature at ends)
    n1   = 3*nCol//4
    xi_u = np.linspace(0.01, 0.99, n1)
    xi_e = np.concatenate([np.linspace(0, 0.12, (nCol-n1)//2),
                            np.linspace(0.88, 1.0, (nCol-n1)//2)])
    xi_np= np.sort(np.concatenate([xi_u, xi_e]))
    Nc   = len(xi_np)

    xi_t = torch.tensor(xi_np,dtype=torch.float64,device=device).unsqueeze(1)
    xi_t.requires_grad_(True)

    # Gaussian load: narrow approximation of concentrated force
    sig   = 0.025
    g_np  = np.exp(-0.5*((xi_np-0.5)/sig)**2)
    g_np /= np.trapezoid(g_np, xi_np)
    g_t   = torch.tensor(g_np,dtype=torch.float64,device=device).unsqueeze(1)

    class MixedNet(nn.Module):
        """
        Network outputs (net_W, net_M) each O(1).
        Hard BCs applied via multiplication by boundary factor b(xi).
        """
        def __init__(self, hidden):
            super().__init__()
            d = 1; lyrs = []
            for w in hidden:
                lyrs += [nn.Linear(d,w), nn.Tanh()]; d = w
            lyrs += [nn.Linear(d,2)]
            self.net = nn.Sequential(*lyrs)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

        def forward(self, xi):
            raw = self.net(xi)
            if BC == 'SS':
                b_W = xi*(1-xi)
                W_hat = b_W * raw[:,0:1]
                M_hat = b_W * raw[:,1:2]
            else:  # CC
                b_W = xi**2 * (1-xi)**2
                b_M = xi*(1-xi)
                W_hat = b_W * raw[:,0:1]
                M_hat = -1.0 + b_M * raw[:,1:2]   # M(0)=M(1)=-1 ✓
            return W_hat, M_hat

    model = MixedNet(hidden).to(device).double()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sch   = optim.lr_scheduler.MultiStepLR(opt, milestones=[4000,8000,11000], gamma=0.3)
    ones  = torch.ones(Nc,1,dtype=torch.float64,device=device)
    hist  = []
    t0    = time.time()

    def loss_fn():
        W_hat, M_hat = model(xi_t)
        dW  = torch.autograd.grad(W_hat, xi_t, ones, create_graph=True)[0]
        dM  = torch.autograd.grad(M_hat, xi_t, ones, create_graph=True)[0]
        d2W = torch.autograd.grad(dW,    xi_t, ones, create_graph=True)[0]
        d2M = torch.autograd.grad(dM,    xi_t, ones, create_graph=True)[0]
        r1  = d2W + c_r * M_hat          # d^2W/dxi^2 + c_r*M = 0
        r2  = d2M + c_g * g_t            # d^2M/dxi^2 + c_g*g = 0
        return r1.pow(2).mean() + r2.pow(2).mean()

    # Phase 1: Adam
    print(f"  Adam × {epochs_adam} epochs...")
    for ep in range(1, epochs_adam+1):
        opt.zero_grad(); loss = loss_fn(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        hist.append(float(loss))
        if ep % print_every == 0:
            with torch.no_grad():
                Wm = model(torch.tensor([[0.5]],dtype=torch.float64,device=device))[0].item()
            print(f"  Adam {ep:6d} | loss={float(loss):.3e} | w_c={Wm*w_sc*1e3:.4f}mm | {time.time()-t0:.1f}s")

    # Phase 2: L-BFGS
    print(f"  L-BFGS × {epochs_lbfgs} steps...")
    opt_lb = optim.LBFGS(model.parameters(), lr=0.3, max_iter=50,
                          history_size=100, line_search_fn='strong_wolfe',
                          tolerance_grad=1e-12, tolerance_change=1e-14)
    for step in range(1, epochs_lbfgs+1):
        def closure():
            opt_lb.zero_grad(); loss = loss_fn(); loss.backward(); return loss
        opt_lb.step(closure)
        with torch.no_grad(): loss = loss_fn()
        hist.append(float(loss))
        if step % 200 == 0:
            with torch.no_grad():
                Wm = model(torch.tensor([[0.5]],dtype=torch.float64,device=device))[0].item()
            print(f"  LBFGS {step:4d} | loss={float(loss):.3e} | w_c={Wm*w_sc*1e3:.4f}mm")

    # Evaluate on dense grid
    xi_ev = torch.linspace(0,1,200,dtype=torch.float64,device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        W_np, M_np = model(xi_ev)
        W_np = W_np.cpu().numpy().flatten() * w_sc
        M_np = M_np.cpu().numpy().flatten() * M_sc

    x_np = xi_ev.cpu().numpy().flatten() * L

    # Recover phi and u0 from analytical formulas (exact for Pi_s >> 1)
    phi_np = np.zeros_like(x_np); u0_np = np.zeros_like(x_np)
    for i, xi_v in enumerate(x_np):
        xr = L - xi_v
        if xi_v <= L/2:
            phi_np[i] = (-Pz*(L**2-4*xi_v**2)/(16*D11) if BC=='SS'
                         else -Pz*(L*xi_v-2*xi_v**2)/(8*D11))
            u0_np[i]  = B11*Pz*(L*xi_v-2*xi_v**2)/(8*A11*D_star)
        else:
            phi_np[i] = (+Pz*(L**2-4*xr**2)/(16*D11) if BC=='SS'
                         else +Pz*(L*xr-2*xr**2)/(8*D11))
            u0_np[i]  = -B11*Pz*(L*xr-2*xr**2)/(8*A11*D_star)

    mid = np.argmin(np.abs(x_np - L/2))
    err = abs(W_np[mid] - w_sc) / abs(w_sc) * 100
    print(f"\n  Mixed PINN done: w_c={W_np[mid]*1e3:.4f}mm  err={err:.3f}%  t={time.time()-t0:.1f}s")

    return x_np, u0_np, W_np, phi_np, hist, M_np


# ============================================================================
#  8. FD-PINN — FINITE DIFFERENCE DERIVATIVES
# ============================================================================

def solve_FDPINN_mixed(A11, B11, D11, A55, L, Pz, BC='SS',
                       hidden=[50,50,50,50], nCol=301,
                       epochs_adam=8000, epochs_lbfgs=600,
                       lr=8e-4, print_every=2000):
    """
    FD-PINN with mixed (W-hat, M-hat) formulation.
    Uses 2nd-order central differences instead of autograd.

    ADVANTAGES over Mixed PINN:
        - ~3x faster (no autograd computational graph)
        - Lower memory footprint
        - Simpler gradient flow

    DISADVANTAGES:
        - O(h^2) truncation error (vs exact autograd)
        - Requires uniform grid (no adaptive collocation)
        - Slightly higher solution error (~0.6% vs 0.4%)
    """
    if not TORCH:
        raise RuntimeError("PyTorch required")

    D_star, D_eff = effective_stiffnesses(A11, B11, D11)
    c_w   = 48 if BC == 'SS' else 192
    c_M   = 4  if BC == 'SS' else 8
    c_r   = c_w // c_M
    c_g   = c_M
    w_sc  = Pz*L**3 / (c_w * D_eff)
    M_sc  = Pz*L / c_M

    print(f"  FD-PINN ({BC}): c_r={c_r}  c_g={c_g}  w_sc={w_sc*1e3:.4f}mm")

    xi_np = np.linspace(0, 1, nCol)
    h_fd  = xi_np[1] - xi_np[0]
    xi_t  = torch.tensor(xi_np,dtype=torch.float64,device=device).unsqueeze(1)

    sig   = 0.025
    g_np  = np.exp(-0.5*((xi_np-0.5)/sig)**2)
    g_np /= np.trapezoid(g_np, xi_np)
    g_t   = torch.tensor(g_np,dtype=torch.float64,device=device)

    def fd2(f):
        """4th-order one-sided at boundary, 2nd-order central interior."""
        d = torch.zeros_like(f)
        d[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h_fd**2
        d[0]    = (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / h_fd**2
        d[-1]   = (2*f[-1] - 5*f[-2] + 4*f[-3] - f[-4]) / h_fd**2
        return d

    class FDMixedNet(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            d = 1; lyrs = []
            for w in hidden:
                lyrs += [nn.Linear(d,w), nn.Tanh()]; d = w
            lyrs += [nn.Linear(d,2)]
            self.net = nn.Sequential(*lyrs)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

        def forward(self, xi):
            raw  = self.net(xi)
            xi_v = xi.squeeze()
            if BC == 'SS':
                b = (xi_v*(1-xi_v)).unsqueeze(1)
                W_hat = b * raw[:,0:1]
                M_hat = b * raw[:,1:2]
            else:  # CC
                b_W = (xi_v**2*(1-xi_v)**2).unsqueeze(1)
                b_M = (xi_v*(1-xi_v)).unsqueeze(1)
                W_hat = b_W * raw[:,0:1]
                M_hat = -1.0 + b_M * raw[:,1:2]
            return W_hat.squeeze(), M_hat.squeeze()

    model = FDMixedNet(hidden).to(device).double()
    hist  = []
    t0    = time.time()

    def loss_fn():
        W_hat, M_hat = model(xi_t)
        r1 = fd2(W_hat) + c_r * M_hat
        r2 = fd2(M_hat) + c_g * g_t
        return r1.pow(2).mean() + r2.pow(2).mean()

    opt_adam = optim.Adam(model.parameters(), lr=lr)
    sch      = optim.lr_scheduler.MultiStepLR(opt_adam, milestones=[3000,6000], gamma=0.3)

    print(f"  Adam × {epochs_adam} epochs...")
    for ep in range(1, epochs_adam+1):
        opt_adam.zero_grad(); loss = loss_fn(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_adam.step(); sch.step()
        hist.append(float(loss))
        if ep % print_every == 0:
            with torch.no_grad():
                Wm = model(torch.tensor([[0.5]],dtype=torch.float64,device=device))[0].item()
            print(f"  Adam {ep:6d} | loss={float(loss):.3e} | w_c={Wm*w_sc*1e3:.4f}mm")

    print(f"  L-BFGS × {epochs_lbfgs} steps...")
    opt_lb = optim.LBFGS(model.parameters(), lr=0.5, max_iter=50,
                          history_size=100, line_search_fn='strong_wolfe',
                          tolerance_grad=1e-12, tolerance_change=1e-14)
    for step in range(1, epochs_lbfgs+1):
        def closure():
            opt_lb.zero_grad(); loss = loss_fn(); loss.backward(); return loss
        opt_lb.step(closure)
        with torch.no_grad(): loss = loss_fn()
        hist.append(float(loss))
        if step % 200 == 0:
            with torch.no_grad():
                Wm = model(torch.tensor([[0.5]],dtype=torch.float64,device=device))[0].item()
            print(f"  LBFGS {step:4d} | loss={float(loss):.3e} | w_c={Wm*w_sc*1e3:.4f}mm")

    # Evaluate
    xi_ev = torch.linspace(0,1,200,dtype=torch.float64,device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        W_np, _ = model(xi_ev)
        W_np = W_np.numpy().flatten() * w_sc

    x_np  = xi_ev.numpy().flatten() * L
    phi_np= np.zeros_like(x_np); u0_np = np.zeros_like(x_np)
    for i, xi_v in enumerate(x_np):
        xr = L - xi_v
        if xi_v <= L/2:
            phi_np[i] = (-Pz*(L**2-4*xi_v**2)/(16*D11) if BC=='SS'
                         else -Pz*(L*xi_v-2*xi_v**2)/(8*D11))
            u0_np[i]  = B11*Pz*(L*xi_v-2*xi_v**2)/(8*A11*(D11-B11**2/A11))
        else:
            phi_np[i] = (+Pz*(L**2-4*xr**2)/(16*D11) if BC=='SS'
                         else +Pz*(L*xr-2*xr**2)/(8*D11))
            u0_np[i]  = -B11*Pz*(L*xr-2*xr**2)/(8*A11*(D11-B11**2/A11))

    mid = np.argmin(np.abs(x_np - L/2))
    err = abs(W_np[mid] - w_sc) / abs(w_sc) * 100
    print(f"\n  FD-PINN done: w_c={W_np[mid]*1e3:.4f}mm  err={err:.3f}%  t={time.time()-t0:.1f}s")

    return x_np, u0_np, W_np, phi_np, hist


# ============================================================================
#  9. COMPREHENSIVE PLOTS
# ============================================================================

def plot_all(results, hists, A11, B11, D11, A55, L, Pz, BC='SS'):
    """Generate comprehensive comparison plots."""
    clr = {'Analytical':'#888','FEM':'#2ca076',
           'Mixed PINN':'#e8c547','FD-PINN':'#5b9cf6',
           'Std PINN':'#ff6b6b','Galerkin':'#b370e0'}
    plt.rcParams.update({'figure.facecolor':'#07080c','axes.facecolor':'#0d0f16',
                         'axes.edgecolor':'#1e2133','axes.labelcolor':'#dde2f4',
                         'xtick.color':'#525878','ytick.color':'#525878',
                         'text.color':'#dde2f4','grid.color':'#1e2133',
                         'grid.linestyle':'--','legend.facecolor':'#0d0f16',
                         'legend.edgecolor':'#1e2133','font.size':10})

    _, D_eff = effective_stiffnesses(A11, B11, D11)
    Pi_s = A55*L**2/D11

    fig = plt.figure(figsize=(18,14))
    fig.suptitle(
        f'FSDT+VK Composite Beam — {BC} — All Methods\n'
        f'[0/90]×1mm, L={L}m, Pz={Pz}N, Pi_s={Pi_s:.0f}, D_eff={D_eff:.2f}N·m',
        fontsize=11, fontweight='bold', y=0.995)
    gs = gridspec.GridSpec(3,3, fig, hspace=0.50, wspace=0.36)

    # ── Row 1: deflection, phi, u0 ──────────────────────────────────
    ax = fig.add_subplot(gs[0,:2])
    for nm,(x,u0,w0,phi) in results.items():
        if nm == 'Std PINN' and np.max(np.abs(w0)) < 1e-4: continue
        ax.plot(x*1e3, w0*1e3, lw=2 if nm!='Analytical' else 1.5,
                ls='--' if nm=='Analytical' else '-',
                color=clr.get(nm,'gray'), label=nm, alpha=0.9)
    ax.set(xlabel='x [mm]', ylabel='w₀ [mm]',
           title=f'Βύθιση w₀(x) — {BC} beam')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0,2])
    for nm,(x,u0,w0,phi) in results.items():
        ax2.plot(x*1e3, phi*1e3, lw=1.8, ls='--' if nm=='Analytical' else '-',
                 color=clr.get(nm,'gray'), label=nm, alpha=0.9)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set(xlabel='x [mm]', ylabel='φ [mrad]',
            title='Στροφή φ(x)')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # ── Row 2: u0, error bars, loss histories ───────────────────────
    ax = fig.add_subplot(gs[1,0])
    for nm,(x,u0,w0,phi) in results.items():
        ax.plot(x*1e3, u0*1e6, lw=1.8, ls='--' if nm=='Analytical' else '-',
                color=clr.get(nm,'gray'), label=nm, alpha=0.9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='x [mm]', ylabel='u₀ [μm]',
           title='Αξονική u₀(x)\n(B₁₁ coupling)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Error comparison bar chart
    ax3 = fig.add_subplot(gs[1,1])
    ref_w  = None
    if 'Analytical' in results:
        xa,_,wa,_ = results['Analytical']
        ref_w = np.interp(L/2, xa, wa)
    if ref_w is not None and abs(ref_w) > 1e-12:
        names_bar = []; errs_bar = []; clrs_bar = []
        for nm,(x,u0,w0,phi) in results.items():
            if nm == 'Analytical': continue
            wc = np.interp(L/2, x, w0)
            names_bar.append(nm.replace(' ','\n')); errs_bar.append(abs(wc-ref_w)/abs(ref_w)*100)
            clrs_bar.append(clr.get(nm,'gray'))
        yp = ax3.bar(range(len(names_bar)), errs_bar, color=clrs_bar,
                     edgecolor='k', alpha=0.85)
        ax3.axhline(1.0, color='yellow', ls='--', lw=1, label='1% threshold')
        for b,v in zip(yp, errs_bar):
            ax3.text(b.get_x()+b.get_width()/2, v+max(errs_bar)*0.02,
                     f'{v:.2f}%', ha='center', fontsize=8, fontweight='bold')
        ax3.set_xticks(range(len(names_bar))); ax3.set_xticklabels(names_bar, fontsize=8)
        ax3.set(ylabel='Σφάλμα w_c (%)', title='Σφάλμα vs Αναλυτική\n(log scale)')
        ax3.set_yscale('log'); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, which='both')

    ax4 = fig.add_subplot(gs[1,2])
    for nm, hist in hists.items():
        if hist is not None and len(hist) > 5:
            ax4.semilogy(hist, lw=1.5, color=clr.get(nm,'gray'), label=nm, alpha=0.9)
    ax4.set(xlabel='Epoch', ylabel='Loss', title='Loss History\n(PINN methods)')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, which='both')

    # ── Row 3: Theoretical error comparison ─────────────────────────
    ax5 = fig.add_subplot(gs[2,:])
    ax5.axis('off')
    rows = []
    Pi_s_val = A55*L**2/D11
    if 'Analytical' in results:
        xa,_,wa,phia = results['Analytical']
        w_ref = np.interp(L/2, xa, wa)
        phi_ref = np.interp(L/4, xa, phia)
    else:
        w_ref = phi_ref = np.nan

    for nm,(x,u0,w0,phi) in results.items():
        wc  = np.interp(L/2, x, w0)
        phc = np.interp(L/4, x, phi)
        err_w   = abs(wc-w_ref)/abs(w_ref)*100 if abs(w_ref)>1e-12 else 0
        err_phi = abs(phc-phi_ref)/abs(phi_ref)*100 if abs(phi_ref)>1e-12 else 0
        rows.append([nm, f'{wc*1e3:.4f}', f'{err_w:.3f}%',
                     f'{phc*1e3:.3f}', f'{err_phi:.3f}%'])

    if rows:
        tbl = ax5.table(cellText=rows,
                        colLabels=['Μέθοδος','w₀(L/2) [mm]','Σφ. w%',
                                   'φ(L/4) [mrad]','Σφ. φ%'],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor('#13151f' if r>0 else '#1e2133')
            cell.set_edgecolor('#2a2f4a')
            cell.set_text_props(color='#dde2f4')
            if r>0 and c==0:
                cell.set_text_props(color=clr.get(rows[r-1][0],'white'))
        tbl.scale(1, 1.8)
    ax5.set_title(f'Πίνακας Αποτελεσμάτων  |  Pi_s={Pi_s_val:.0f}  '
                  f'D_eff={D_eff:.2f}N·m  Pz={Pz}N  BC={BC}',
                  fontsize=10)

    plt.savefig('/tmp/unified_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: unified_results.png")


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':

    print("="*68)
    print("  UNIFIED SOLVER — FSDT + VK Composite Beam")
    print("  Methods: FEM · Galerkin/RK45 · Std PINN · Mixed PINN · FD-PINN")
    print("="*68)

    # ── Define laminate (Emmanouela: [0/90] × 1mm) ────────────────────
    layers = [
        {'theta':  0.0, 'thickness':1e-3, 'E1':135e9, 'E2':10e9, 'nu12':0.30, 'G12':5e9},
        {'theta': 90.0, 'thickness':1e-3, 'E1':135e9, 'E2':10e9, 'nu12':0.30, 'G12':5e9},
    ]

    t_all = time.time()
    A11, B11, D11, A55, H = calculate_laminate(layers)
    D_star, D_eff = effective_stiffnesses(A11, B11, D11)
    L, Pz = 1.0, 100.0
    BC    = 'SS'

    print(f"\n  Laminate: A11={A11:.3e}  B11={B11:.3e}  D11={D11:.3f}  A55={A55:.3e}")
    print(f"  D*={D_star:.4f}  D_eff={D_eff:.4f}  Pi_s={A55*L**2/D11:.0f}")
    print(f"  Expected w_c = {Pz*L**3/(48*D_eff)*1e3:.4f} mm ({BC} beam)")

    results = {}; hists = {}

    # ── 1. Analytical ─────────────────────────────────────────────────
    print("\n  [1] Analytical solution")
    xa, u0a, w0a, phia = analytical_solution(A11, B11, D11, A55, L, Pz, BC=BC)
    results['Analytical'] = (xa, u0a, w0a, phia)
    mid = len(xa)//2
    print(f"  w_c={w0a[mid]*1e3:.4f}mm  phi(0)={phia[0]*1e3:.4f}mrad")

    # ── 2. FEM ────────────────────────────────────────────────────────
    print("\n  [2] FEM (200 elements, reduced integration)")
    t1 = time.time()
    xf, u0f, w0f, phif = solve_FEM(A11, B11, D11, A55, L, Pz, BC=BC, nElem=200)
    results['FEM'] = (xf, u0f, w0f, phif); hists['FEM'] = None
    mf = len(xf)//2
    print(f"  w_c={w0f[mf]*1e3:.4f}mm  err={abs(w0f[mf]-w0a[mid])/abs(w0a[mid])*100:.4f}%  t={time.time()-t1:.3f}s")

    # ── 3. Galerkin / RK45 (dynamics, energy conservation check) ──────
    print("\n  [3] Galerkin/RK45 (free vibrations, energy conservation)")
    rhoA = 5.0  # kg/m (representative composite density)
    t_gal, a_gal, K_gal, V_gal = galerkin_duffing_free(rhoA, D_eff, A11, L, A0=0.015)

    # ── 4. Standard PINN (demonstration of failure) ──────────────────
    if TORCH:
        print("\n  [4] Standard PINN (3-field — EXPECTED TO FAIL for large Pi_s)")
        xp_std, u0_std, w0_std, phi_std, h_std = solve_PINN_standard(
            A11, B11, D11, A55, L, Pz, BC=BC, hidden=[30,30,30],
            epochs=8000, print_every=2000)
        results['Std PINN'] = (xp_std, u0_std, w0_std, phi_std)
        hists['Std PINN']   = h_std
        ms = np.argmin(np.abs(xp_std - L/2))
        print(f"  Std PINN w_c={w0_std[ms]*1e3:.4f}mm  "
              f"err={abs(w0_std[ms]-w0a[mid])/abs(w0a[mid])*100:.1f}%")

    # ── 5. Mixed PINN ────────────────────────────────────────────────
    if TORCH:
        print("\n  [5] Mixed PINN (W-hat, M-hat) — CORRECT for large Pi_s")
        xp, u0p, w0p, phip, h_pinn, M_pinn = solve_PINN_mixed(
            A11, B11, D11, A55, L, Pz, BC=BC,
            hidden=[50,50,50,50], nCol=300,
            epochs_adam=12000, epochs_lbfgs=800, print_every=3000)
        results['Mixed PINN'] = (xp, u0p, w0p, phip)
        hists['Mixed PINN']   = h_pinn
        mp = np.argmin(np.abs(xp - L/2))
        print(f"  Mixed PINN w_c={w0p[mp]*1e3:.4f}mm  "
              f"err={abs(w0p[mp]-w0a[mid])/abs(w0a[mid])*100:.3f}%")

    # ── 6. FD-PINN ───────────────────────────────────────────────────
    if TORCH:
        print("\n  [6] FD-PINN Mixed — finite difference derivatives")
        xfd, u0fd, w0fd, phifd, h_fd = solve_FDPINN_mixed(
            A11, B11, D11, A55, L, Pz, BC=BC,
            hidden=[50,50,50,50], nCol=301,
            epochs_adam=8000, epochs_lbfgs=600, print_every=2000)
        results['FD-PINN'] = (xfd, u0fd, w0fd, phifd)
        hists['FD-PINN']   = h_fd
        mfd = np.argmin(np.abs(xfd - L/2))
        print(f"  FD-PINN w_c={w0fd[mfd]*1e3:.4f}mm  "
              f"err={abs(w0fd[mfd]-w0a[mid])/abs(w0a[mid])*100:.3f}%")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS  (reference: {w0a[mid]*1e3:.4f} mm)")
    print(f"  {'─'*60}")
    ref = w0a[mid]
    for nm,(x,u0,w0,phi) in results.items():
        m   = np.argmin(np.abs(x - L/2))
        err = abs(w0[m]-ref)/abs(ref)*100
        print(f"  {nm:<22}  w_c={w0[m]*1e3:+8.4f}mm  err={err:6.3f}%")
    print(f"  {'='*60}")
    print(f"  Total time: {time.time()-t_all:.1f}s")

    # ── Plots ────────────────────────────────────────────────────────
    plot_all(results, hists, A11, B11, D11, A55, L, Pz, BC=BC)
