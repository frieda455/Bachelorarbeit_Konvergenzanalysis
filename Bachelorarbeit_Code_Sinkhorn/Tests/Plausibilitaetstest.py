# Ziel des Plausibilitaetstest ist es, prüfen zu können, ob der eigens programmierte Algorithmus in einfachen Fällen richtige Lösungen produziert, um die Korrektheit abschätzen zu können. Dabei werden für verschiedene Problemgrößen einfache Probleme generiert und gelöst. Die Ergebnisse in Bezug auf die Differenz der Kosten zu den minimalen Kosten werden in einer tex-Datei als Tabelle gespeichert.

# ---------- Imports - Pakete, Algorithmen und Hilfsfunktionen ---------- 

# Pakete für Pfad- und Dateiverwaltung
from Tests.pfadTestEinstellung import CSV_ORDNER, TEX_ORDNER, DIAGRAMM_ORDNER

# Algorithmen und Hilfsfunktionen
from Algorithmen_Funktionen.sinkhornVerfahren import sinkhorn
from Algorithmen_Funktionen.DatenGenerieren import datenGenerierenEinfach, datenGenerierenZufall
from Algorithmen_Funktionen.simplexVerfahren import simplexLoeser

from Algorithmen_Funktionen.Hilfsfunktionen import dfZuLatex

# Pakete für die Datenanalyse
import csv
import pandas as pd
from scipy import stats
import numpy as np

# Darstellung der Plots
from Algorithmen_Funktionen.plotStilEinstellung import plt, COLORS, COLORS_LIGHT

# ---------- Plausibilitaetstest ----------

def plausibilitaetstest(dimWerte, epsWerte, dateiname = "Plausibilitaetstest", tol=1e-7, maxit=100000, ausgeben=False):
    """Fuehrt den Plausibilitätstest für verschiedene Dimensionen und Epsilon-Werte durch."""

    # Evtl. alte CSV-Datei löschen
    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    if dateipfad.is_file():
        dateipfad.unlink()

    for (n, m), eps in zip(dimWerte, epsWerte):

        mu, nu, c = datenGenerierenEinfach(n,m)

        ergebnisSinkhorn = sinkhorn(np.ones(m), c, mu, nu, eps, tol, maxit)
        ergebnisSimplex = simplexLoeser(mu,nu,c)

        sicherungPlausibilitaetsTest(dateiname, ergebnisSinkhorn, ergebnisSimplex)

        if ausgeben:
            # Zeige bei Matrixausgaben alle Nachkommastellen an
            np.set_printoptions(
                suppress=False,
                precision=7,
                formatter={'float_kind': lambda x: f"{x:.7f}"}
            )

            print("Sinkhorn für " + f"{n}, {m}" + ":" + "\n")
            print(ergebnisSinkhorn["pi"])
            
            print("Simplex für " + f"{n}, {m}" + ":" + "\n")
            print(ergebnisSimplex["pi"])

    latexPlausibilitaetstest(dateiname)

#  ---------- Hilfsfunktionen ---------- 

def sicherungPlausibilitaetsTest(dateiname, ergebnisSinkhorn, ergebnisSimplex):
    """Speichert die Ergebnisse des Plausibilitätstests in einer CSV-Datei."""

    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    dateiExistiert = dateipfad.is_file()
    
    with open(dateipfad, "a", newline="") as f:
    
        writer = csv.writer(f)
    
        if not dateiExistiert:
            writer.writerow([
                "n",
                "m",
                "eps",
                "Iterationen",
                "Zeit [s]",
                "Marginalfehler",
                "Kosten",
                "Kostendifferenz",
            ])
    
        writer.writerow([
            ergebnisSinkhorn["param"]["n"],
            ergebnisSinkhorn["param"]["m"],
            ergebnisSinkhorn["param"]["eps"],
            ergebnisSinkhorn["iterationen"],
            ergebnisSinkhorn["zeit"],
            ergebnisSinkhorn["residuum"],
            ergebnisSinkhorn["kosten"],
            abs(ergebnisSinkhorn["kosten"] - ergebnisSimplex["kosten"]),
        ])

def latexPlausibilitaetstest(dateiname):
    """Erstellt eine LaTeX-Tabelle der Ergebnisse des Plausibilitätstests."""

    # ---- CSV-Datei ----
    df_latex = pd.read_csv(CSV_ORDNER / f"{dateiname}.csv", sep=',', header=0)

    # ---- Sicherung der Tabelle als .tex Datei ----
    # Darstellung in Zehnerpotenzen
    df_latex["eps"] = df_latex["eps"].map(lambda x: f"{x:.2e}" if pd.notnull(x) else "nan")

    # Nachkommastellen bearbeiten
    df_latex["Zeit [s]"] = df_latex["Zeit [s]"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Marginalfehler"] = df_latex["Marginalfehler"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Kostendifferenz"] = df_latex["Kostendifferenz"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Kosten"] = df_latex["Kosten"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")

    # Spaltennamen anpassen
    df_latex = df_latex.rename(columns={"eps": r"$\varepsilon$"})

    caption="Ergebnis des Plausibilitätstests"

    dfZuLatex(df_latex, dateiname, caption)