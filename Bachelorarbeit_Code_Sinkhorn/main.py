# Tests des Sinkhorn-Algorithmus fuer das entropische regularisierte Transportproblem

# ---------- Imports - Pakete, Algorithmen und Hilfsfunktionen ---------- 

from Tests.Plausibilitaetstest import plausibilitaetstest
from Tests.Parametertest import parameterTest, parameterTestTol, diagrammParameterTest
from Tests.Konvergenztest import konvergenzTest

from Algorithmen_Funktionen.DatenGenerieren import datenGenerierenZufall
from Algorithmen_Funktionen.simplexVerfahren import simplexLoeser

import numpy as np

# ---------- Plausibilitaetstest ----------

# maxit = 100000
# tol = 1e-7

# dimWerte = [(64,64),(64,64),(64,64),(64,64),(64,64),(64,64),(64,64),(64,64),(64,64)]
# epsWerte = [1, 0.8, 0.5, 0.1, 0.08, 0.05,0.01,0.008,0.005,0.001]
# plausibilitaetstest(dimWerte,epsWerte,"Plausibilitaetstest64",tol=tol,maxit=maxit, ausgeben=True)

# dimWerte = [(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512),(512,512)]
# epsWerte = [1,0.8,0.5,0.08,0.05,0.01,0.008,0.005,0.001]
# plausibilitaetstest(dimWerte,epsWerte,"Plausibilitaetstest512",10e-8,20000,ausgeben=False)

# -- Vorher andere Daten generiert, zufaellig! --

# dimWerte = [(512,512),(512,512),(512,512),(512,512),(512,512),(512,512)]
# epsWerte = [1,0.8,0.5,0.08,0.05,0.01]
# plausibilitaetstest(dimWerte,epsWerte,"Plausibilitaetstest512ZufaelligesProblem",tol=tol,maxit=maxit,ausgeben=False)

# ---------- Parametertest ----------
# maxit = 100000
# tol = 1e-7

# Dimension 64
# epsWerte1 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009]
# parameterTest(64,64,epsWerte1,"Parametertest64_1", tol=tol, maxit=maxit)

# Dimension 128
# epsWerte1 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
# parameterTest(128,128,epsWerte1,"Parametertest128_1", tol=tol, maxit=maxit)

# Dimension 256
# epsWerte1 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
# parameterTest(256,256,epsWerte1,"Parametertest256_1", tol=tol, maxit=maxit)

# Dimension 512
# epsWerte1 = [1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# parameterTest(512,512,epsWerte1,"Parametertest512_1", tol=tol, maxit=maxit)

# Dimension 1024
# epsWerte1 = [1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# parameterTest(1024,1024,epsWerte1,"Parametertest1024_1", tol=tol, maxit=maxit)

# Dimension 2048
# maxit = 100000
# tol = 1e-6
# epsWerte1 = [1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5]
# parameterTest(2048,2048,epsWerte1,"Parametertest2048_1", tol=tol, maxit=maxit)
# diagrammParameterTest("Parametertest2048_1", maxit)

# epsWerte1 = [1.2, 1.3, 1.4, 1.5]
# parameterTest(2048,2048,epsWerte1,"Parametertest2048_1_1", tol=tol, maxit=maxit)

# Test mit verschiedener Toleranz
# for n in [64, 128, 256, 512]:
#     for tol in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
#         epsWerte = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08]
#         maxit = 100000
#         parameterTest(n,n,epsWerte,f"Parametertest{n}_tol{tol}", tol=tol, maxit=maxit)

# n= 512
# for tol in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
#     epsWerte = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#     maxit = 100000
#     parameterTest(n,n,epsWerte,f"Parametertest{n}_tol{tol}", tol=tol, maxit=maxit)

# ----------

# n = 128
# epsWerte = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08]
# tolWerte = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# parameterTestTol(n, n, epsWerte, tolWerte, dateiname=f"ParametertestTol_{n}_Zeit", maxit=100000)

# ----------

# n = 64
# epsWerte = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009]
# maxit = 100000
# tol=1e-6
# parameterTest(n,n,epsWerte,f"Parametertest{n}_tol{tol}", tol=tol, maxit=maxit)

# ---------- Konvergenztest ----------

# for seed in [13, 39, 42, 68, 74, 97]:
#     konvergenzTest(256, 256, [0.2, 0.3, 0.4, 0.6, 0.8, 1.0], f"Konvergenztest256_1_{seed}", seed=seed)

# ---------- Simplex-Test ----------

# n_values = [64, 128, 256, 512, 1024, 2048, 4000, 6000, 8000, 10000]

# print("=" * 65)
# print(f"{'n':<6} | {'Status':<10} | {'Laufzeit (s)':<12} | {'Residuum':<12} | {'Kosten':<14}")
# print("=" * 65)

# for n in n_values:

#     mu, nu, c = datenGenerierenZufall(n, n)
#     res = simplexLoeser(mu, nu, c)

#     n_val = n
#     zeit = res["zeit"]
#     marginalFehler = res["marginalFehler"]
#     kosten = res["kosten"]
#     status_text = "GÜLTIG" if res["istGueltig"] else "UNGÜLTIG"

#     print(f"{n_val:<6} | {status_text:<10} | {zeit:<12.5f} | {marginalFehler:<12.2e} | {kosten:<14.4f}")

#     # Ausführliche Ausgabe im Fehlerfall
#     if not res["istGueltig"]:
#         print(f"       └── WARNUNG/FEHLER: {res['fehlerMeldung']}")

# print("=" * 65)