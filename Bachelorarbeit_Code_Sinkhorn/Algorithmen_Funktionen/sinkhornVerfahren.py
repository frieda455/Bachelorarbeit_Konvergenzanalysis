# Sinkhorn-Algorithmus fuer das entropisch regularisierte Transportproblem

# ---------- Imports - Pakete ---------- 

import numpy as np
import time

# ---------- Hilfsfunktionen ----------

def k_matrix(c, eps): 
    """Berechnet die K-Matrix fuer das entropisch regularisierte Transportproblem."""
    return np.exp(-c / eps - 1.0) # Durch Broadcasting wird die Division von c durch eps elementweise durchgefuehrt

def transportplan(alpha, beta, K):
    """Berechnet den Transportplan fuer gegebene Parameter alpha, beta und K-Matrix"""
    # Es gilt: alpha[:, None].shape == (n, 1), beta[None, :].shape == (1, m), K.shape == (n, m)
    # Stern-Operator ist elementweise Multiplikation
    # Python Broadcasting sorgt dafür, dass die Dimensionen automatisch angepasst werden
    # Es gilt gerade: alpha[:, None] * K * beta[None, :] = alpha[i] * K[i, j] * beta[j]

    return alpha[:, None] * K * beta[None, :]

# ---------- Sinkhorn-Algorithmus ----------

def sinkhorn(beta0, c, mu, nu, eps, tol=1e-6, maxit=10000):
    """Fuehrt den gesamten Sinkhorn-Algorithmus aus und speichert die Ergebnisse in einer Container-Klasse.

    Args:
        beta0: Startvektor für beta
        c: Kostenmatrix der Dimension nxm
        mu: Zeilenmarginal der Laenge n
        nu: Spaltenmarginal der Laenge m
        eps: regularisierungsparameter
        tol: Toleranz fuer das Konvergenzkriterium. Defaults to 1e-6.
        maxit: Maximale Anzahl von Iterationen. Defaults to 10000.

    Returns:
        Ergebnisobjekt der Container-Klasse
    """

    n, m = c.shape

    # Berechnung
    zeit1 = time.time()

    # Initialisierung
    K = k_matrix(c, eps)
    alpha = np.ones(n)
    beta = beta0
    historie = {
        "residuum": [],
        "alpha": [],
        "beta": [],
        "transportkosten": [],
        "iterationen": []
    }

    for k in range(1, maxit+1):

        # Iterationsschritt
        alpha = mu / (K @ beta)
        beta = nu / (K.T @ alpha)

        # Berechnung des Residuums für den Transportplan        
        mu_residuum = np.linalg.norm(alpha*(K @ beta) - mu, np.inf)
        nu_residuum = np.linalg.norm(beta*(K.T @ alpha) - nu, np.inf)
        residuum = max(mu_residuum, nu_residuum)

        # Speichern der Iterationshistorie
        historie["residuum"].append(residuum)
        historie["iterationen"].append(k)

        if residuum < tol:
            break

    zeit2 = time.time()

    pi = transportplan(alpha, beta, K)

    ergebnis = {
        "param": {"eps": eps, "n": n, "m": m,},

        "alpha": alpha,
        "beta": beta,
        "pi": pi,
        "kosten": np.sum(pi * c),

        "iterationen": len(historie["residuum"]),
        "zeit": zeit2 - zeit1,
        "residuum": historie["residuum"][-1],

        "historie": historie,
    }

    return ergebnis