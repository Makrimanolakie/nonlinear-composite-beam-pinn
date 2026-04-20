# Nonlinear Composite Beam Analysis with Mixed PINN

**From Euler–Bernoulli to Mixed Physics-Informed Neural Networks**  
FSDT · ABD Laminates · von Kármán Nonlinearity · FEM · Standard PINN · Mixed PINN · FD-PINN

**Emmanouela [Surname] · NTUA Athens · Preprint 2025**

## ✨ Τι περιλαμβάνει το repo

- Πλήρες θεωρητικό paper (PDF + LaTeX source)
- Summary of Approaches, Limits & Code Evolution
- Ενιαίο Python solver (`unified_solver.py`) με **όλες τις μεθόδους**:
  - Analytical solution (Deff)
  - FEM (reduced integration – shear locking free)
  - Galerkin + RK45 (dynamics)
  - Standard 3-field PINN (για σύγκριση – αποτυγχάνει σε Πs ≫ 1)
  - **Mixed (Ŵ, M̂) PINN** ← η σωστή λύση
  - **FD-PINN** (3× ταχύτερο)

## 🚀 Γρήγορη Εγκατάσταση

```bash
git clone https://github.com/YOURUSERNAME/nonlinear-composite-beam-pinn.git
cd nonlinear-composite-beam-pinn
pip install -r requirements.txt
python unified_solver.py
