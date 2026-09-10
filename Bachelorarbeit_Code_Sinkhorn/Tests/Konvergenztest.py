# Testet die Konvergenz des Sinkhorn-Algorithmus fuer verschiedene Dimensionen und Parameter.

# ---------- Imports - Pakete, Algorithmen und Hilfsfunktionen ---------- 

# Pakete für Pfad- und Dateiverwaltung
from Tests.pfadTestEinstellung import CSV_ORDNER, TEX_ORDNER, DIAGRAMM_ORDNER

# Algorithmen und Hilfsfunktionen
from Algorithmen_Funktionen.sinkhornVerfahren import sinkhorn
from Algorithmen_Funktionen.DatenGenerieren import datenGenerierenZufall

# Pakete für die Datenanalyse
import csv
import pandas as pd
from scipy import stats
import numpy as np
import math

# Darstellung der Plots
from Algorithmen_Funktionen.plotStilEinstellung import plt, COLORS, COLORS_LIGHT

# ---------- Konvergenztest ----------

def konvergenzTest(n, m, epsWerte, dateiname= "Konvergenztest", tol=1e-7, maxit=100000, seed=13):
    """Erstellt ein Diagramm des Konvergenzverlaufs des Sinkhorn-Algorithmus für verschiedene Epsilon-Werte bei festen Dimensionen n und m."""

    # Evtl. alte CSV-Datei löschen
    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    if dateipfad.is_file():
        dateipfad.unlink()

    mu, nu, c = datenGenerierenZufall(n,m,seed)

    for eps in epsWerte:
        ergebnisSinkhorn = sinkhorn(np.ones(m), c, mu, nu, eps, tol, maxit)
        sicherungKonvergenzTest(dateiname, seed, ergebnisSinkhorn)

    diagrammKonvergenzTest(dateiname)

#  ---------- Hilfsfunktionen ---------- 

def sicherungKonvergenzTest(dateiname, seed, ergebnisSinkhorn):
    """Speichert die Ergebnisse des Konvergenztests in einer CSV-Datei."""
    
    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    dateiExistiert = dateipfad.is_file()

    with open(dateipfad, "a", newline="") as f:
    
        writer = csv.writer(f)
    
        if not dateiExistiert:
            writer.writerow([
                "Seed",
                "n",
                "m",
                "eps",
                "Iteration",
                "Residuum"
            ])

        historie = ergebnisSinkhorn["historie"]
        residuuen = historie["residuum"]

        # Zeilenweise Speicherung für jede Iteration k
        for k in range(len(residuuen)):
            writer.writerow([
                seed,
                ergebnisSinkhorn["param"]["n"],
                ergebnisSinkhorn["param"]["m"],
                ergebnisSinkhorn["param"]["eps"],
                historie["iterationen"][k],
                residuuen[k],
            ])

def diagrammKonvergenzTest(dateiname):
    """Erstellt ein Diagramm des Konvergenzverlaufs."""

    # ---- CSV-Datei einlesen ----
    df = pd.read_csv(CSV_ORDNER / f"{dateiname}.csv", sep=",", header=0)

    # ---- Plot erstellen ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # Gruppierung nach eps: Für jedes epsilon wird eine Kurve gezeichnet
    for i, (eps, gruppe) in enumerate(df.groupby("eps", sort=False)):

        color = list(COLORS.values())[i % len(COLORS)]

        ax.plot(
            gruppe["Iteration"],
            gruppe["Residuum"],
            label=rf"$\varepsilon = {eps}$",
            linewidth=1.2,
            color=color,
        )

    # Achseneinstellungen für die semilogarithmische Konvergenzdarstellung
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(
        r"Residuum $\max(\|\pi^{(k)}\mathbf{1}_m - \mu\|_\infty, \|(\pi^{(k)})^T\mathbf{1}_n - \nu\|_\infty)$"
    )

    ax.set_title(
        f"Konvergenzverlauf des Sinkhorn-Algorithmus ($n={df['n'].iloc[0]}, m={df['m'].iloc[0]}$) mit Seed={df['Seed'].iloc[0]}"
    )

    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.35, linestyle="--")

    # Speichern des Diagramms
    fig.tight_layout()
    fig.savefig(DIAGRAMM_ORDNER / f"{dateiname}.pdf", bbox_inches="tight")

    plt.close(fig)