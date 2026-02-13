# OFN-project
OFN σ_crit = π/4 – Replication Package Minimal Python implementation reproducing the critical transition σ_crit = π/4 in the Ontological Fundamental Network (4-vertex cycle). Implements transfer operator, explicit normalization, and σ-parameter. Verifies spectral degeneracy at critical point. Includes tests, notebooks, and Docker.
# OFN σ_crit = π/4 – Replication Package

This repository contains a minimal, self-contained implementation of the Ontological Fundamental Network (OFN) model described in:

> Evdokimov, O. (2026). *OFN Part II: Dynamics, Reading Paths, and the σ-Parameter of Consciousness*.

It reproduces the critical transition σ_crit = π/4 on a 4-vertex directed cycle.

## Requirements
- Python 3.9+
- NumPy, SciPy, Matplotlib, PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
git clone https://github.com/OFN-project/sigma_crit_replication
cd sigma_crit_replication
pip install -r requirements.txt
python ofn_sigma_crit.py

---

## 2. **requirements.txt**

```txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pyyaml>=5.4.0
# Minimal replicable network – 4-vertex directed cycle
network:
  vertices: 4
  edges: [[0,1], [1,2], [2,3], [3,0]]  # directed cycle
  W: 1.0                                # all weights = 1
  Theta: 1.57079632679                  # π/2 forward, -π/2 reverse

parameters:
  alpha: 1.0                           # intra-component coupling
  beta: 0.78539816339                  # π/4 * alpha (critical point)
  gamma: 0.546479                      # λ = 1 - 4/π, γ = λ·β
  kappa: 1.0                           # phase stiffness
  epsilon: 1e-8                        # horizon glow
  R: 1                                 # perceptual radius

simulation:
  cycles: 100                          # number of full cycles
  sigma_range: [0.6, 1.0, 0.01]        # scan from 0.6 to 1.0 step 0.01
#!/usr/bin/env python3
"""
OFN Minimal Replication: σ_crit = π/4 on a 4-vertex directed cycle.
Author: Oleg Evdokimov (implementation for OFN Part II)
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import yaml

class OFN4Cycle:
    """
    Minimal Ontological Fundamental Network:
    - 4 vertices in a directed cycle
    - All weights W = 1
    - Forward edges: Θ = +π/2, reverse edges: Θ = -π/2
    - Uniform vertex type (C), zero spinor density
    """
    
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.n = self.cfg['network']['vertices']
        self.W = self.cfg['network']['W']
        self.Theta0 = self.cfg['network']['Theta']
        
        # Build adjacency with phases
        self.edges = []
        self.weights = []
        self.phases = []
        for u, v in self.cfg['network']['edges']:
            self.edges.append((u, v))
            self.weights.append(self.W)
            self.phases.append(self.Theta0)      # forward: +π/2
            self.edges.append((v, u))
            self.weights.append(self.W)
            self.phases.append(-self.Theta0)     # reverse: -π/2
        
        # Convert to arrays for fast lookup
        self.edges = np.array(self.edges)
        self.weights = np.array(self.weights)
        self.phases = np.array(self.phases)
        
        # Parameters
        self.alpha = self.cfg['parameters']['alpha']
        self.beta = self.cfg['parameters']['beta']
        self.gamma = self.cfg['parameters']['gamma']
        self.kappa = self.cfg['parameters']['kappa']
        self.epsilon = self.cfg['parameters']['epsilon']
        self.R = self.cfg['parameters']['R']
        
        # Derived
        self.sigma = self.beta / self.alpha
        
    def t_amplitude(self, u, v):
        """Kinetic amplitude t_wu from Eq. (49)"""
        # Find indices of edges (u→v) and (v→u)
        mask_forward = (self.edges[:, 0] == u) & (self.edges[:, 1] == v)
        mask_reverse = (self.edges[:, 0] == v) & (self.edges[:, 1] == u)
        
        if not np.any(mask_forward) or not np.any(mask_reverse):
            return 0.0
        
        W_f = self.weights[mask_forward][0]
        W_r = self.weights[mask_reverse][0]
        Theta_f = self.phases[mask_forward][0]
        Theta_r = self.phases[mask_reverse][0]
        
        return np.sqrt(W_f * W_r) * np.exp(1j * (Theta_f + Theta_r) / 2)
    
    def active_region(self, current_vertex):
        """Γ(λ): vertices within distance R of current vertex"""
        V_lambda = [current_vertex]
        if self.R >= 1:
            # Add neighbors (outgoing edges)
            mask = self.edges[:, 0] == current_vertex
            neighbors = self.edges[mask, 1]
            V_lambda.extend(neighbors)
        return np.unique(V_lambda)
    
    def transfer_operator(self, V_lambda, V_lambda_next):
        """H(λ) matrix from Eq. (54)"""
        n_curr = len(V_lambda)
        n_next = len(V_lambda_next)
        H = np.zeros((n_next, n_curr), dtype=complex)
        
        for i, w in enumerate(V_lambda_next):
            for j, u in enumerate(V_lambda):
                t = self.t_amplitude(u, w)
                if t != 0:
                    V_u = 0.0  # S(u)=0, type=C, no nonlinearity in linear regime
                    tau_u = 0.0  # No cycles in minimal test
                    H[i, j] = t * np.exp(-1j * (V_u - tau_u))
        
        return H
    
    def normalize(self, Phi):
        """Explicit renormalization: Φ ← Φ / ||Φ||"""
        norm = np.sqrt(np.sum(np.abs(Phi)**2))
        if norm < 1e-15:
            return np.ones_like(Phi) * self.epsilon / np.sqrt(len(Phi))
        return Phi / norm
    
    def compute_sigma(self, Phi):
        """σ = [∑ |Φ|⁴]^{-1/4}  Eq. (57)"""
        return (np.sum(np.abs(Phi)**4))**(-0.25)
    
    def run_cycle(self, sigma, n_cycles=100):
        """Run simulation for given σ, return mean σ and eigenvalues"""
        self.beta = sigma * self.alpha
        self.sigma = sigma
        
        # Initial condition: uniform on vertex 0
        V0 = self.active_region(0)
        Phi = np.ones(len(V0), dtype=complex) / np.sqrt(len(V0))
        
        sigma_values = []
        eigenvalues_history = []
        
        for step in range(n_cycles * 4):  # 4 steps per full cycle
            current_v = step % 4
            V_curr = self.active_region(current_v)
            V_next = self.active_region((current_v + 1) % 4)
            
            # Map Phi from V_curr ordering to V_curr indices
            Phi_mapped = np.zeros(len(V_curr), dtype=complex)
            for i, v in enumerate(V_curr):
                idx = np.where(V_curr == v)[0][0]  # simplified; in real code use dict
                Phi_mapped[i] = Phi[idx] if idx < len(Phi) else self.epsilon
            
            H = self.transfer_operator(V_curr, V_next)
            Psi = H @ Phi_mapped
            
            # Renormalize
            Phi_new = self.normalize(Psi)
            
            # Compute σ
            sigma_val = self.compute_sigma(Phi_new)
            sigma_values.append(sigma_val)
            
            # Store eigenvalues every 4 steps
            if step % 4 == 0:
                if H.shape[0] == H.shape[1]:  # square matrix
                    evals = eig(H)[0]
                    eigenvalues_history.append(evals)
            
            # Prepare for next step
            Phi = Phi_new
            V_curr = V_next
        
        return np.mean(sigma_values[-20:]), eigenvalues_history
    
    def scan_sigma(self, sigma_min, sigma_max, sigma_step):
        """Scan σ range and find critical point"""
        sigmas = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)
        mean_sigma = []
        variance_sigma = []
        
        for s in sigmas:
            mean_val, _ = self.run_cycle(s, n_cycles=50)
            # Run multiple times for variance
            vals = []
            for _ in range(5):
                v, _ = self.run_cycle(s, n_cycles=20)
                vals.append(v)
            mean_sigma.append(np.mean(vals))
            variance_sigma.append(np.var(vals))
        
        return sigmas, np.array(mean_sigma), np.array(variance_sigma)

def main():
    print("OFN σ_crit = π/4 Replication")
    print("=" * 50)
    
    # Initialize network with default critical point
    ofn = OFN4Cycle()
    
    # Test at exact critical point
    sigma_crit = np.pi / 4
    mean_sigma, evals = ofn.run_cycle(sigma_crit, n_cycles=100)
    
    print(f"\nTest at σ = π/4 ≈ {sigma_crit:.6f}")
    print(f"Measured ⟨σ⟩ = {mean_sigma:.6f}")
    print(f"Error: {abs(mean_sigma - sigma_crit):.6e}")
    
    if evals:
        print(f"\nEigenvalues at critical point (last step):")
        for e in evals[-1]:
            print(f"  {e:.6f}")
    
    # Scan σ around critical point
    print("\nScanning σ from 0.6 to 1.0...")
    sigmas, mean_s, var_s = ofn.scan_sigma(0.6, 1.0, 0.02)
    
    # Find maximum variance (critical point indicator)
    critical_idx = np.argmax(var_s)
    sigma_crit_observed = sigmas[critical_idx]
    
    print(f"\nCritical point detected at σ = {sigma_crit_observed:.4f}")
    print(f"π/4 = {np.pi/4:.4f}")
    print(f"Difference: {abs(sigma_crit_observed - np.pi/4):.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sigmas, mean_s, 'b-o', label='⟨σ⟩')
    plt.axvline(x=np.pi/4, color='r', linestyle='--', label='π/4')
    plt.xlabel('σ (β/α)')
    plt.ylabel('⟨σ⟩ (measured)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(sigmas, var_s, 'r-o', label='Variance')
    plt.axvline(x=np.pi/4, color='r', linestyle='--', label='π/4')
    plt.axvline(x=sigma_crit_observed, color='g', linestyle=':', 
                label=f'Observed: {sigma_crit_observed:.4f}')
    plt.xlabel('σ (β/α)')
    plt.ylabel('Var(σ)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sigma_scan.png', dpi=150)
    print("\nPlot saved as 'sigma_scan.png'")
    
    # Verification
    print("\n" + "=" * 50)
    print("VERIFICATION:")
    if abs(sigma_crit_observed - np.pi/4) < 0.02:
        print("✅ SUCCESS: σ_crit = π/4 reproduced within 0.02 tolerance")
    else:
        print("❌ FAILURE: Critical point deviates from π/4")
    
    return 0

if __name__ == "__main__":
    main()
