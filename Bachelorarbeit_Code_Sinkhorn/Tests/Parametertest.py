# Ziel des Parametertests ist es, die Auswirkungen der Parameter epsilon auf die Konvergenzgeschwindigkeit und die Genauigkeit des semiglattes Newton-Verfahrens zu untersuchen. Dazu werden für verschiedene Kombinationen von epsilon zufällige Probleme generiert, geloest und die Ergebnisse in einer CSV-Datei gespeichert.

# ---------- Imports - Pakete, Algorithmen und Hilfsfunktionen ---------- 

# Pakete für Pfad- und Dateiverwaltung
from Tests.pfadTestEinstellung import CSV_ORDNER, TEX_ORDNER, DIAGRAMM_ORDNER

# Algorithmen und Hilfsfunktionen
from Algorithmen_Funktionen.sinkhornVerfahren import sinkhorn
from Algorithmen_Funktionen.simplexVerfahren import simplexLoeser
from Algorithmen_Funktionen.DatenGenerieren import datenGenerierenZufall

from Algorithmen_Funktionen.Hilfsfunktionen import dfZuLatex

# Pakete für die Datenanalyse
import csv
import pandas as pd
from scipy import stats
import numpy as np
import math

# Darstellung der Plots
from Algorithmen_Funktionen.plotStilEinstellung import plt, COLORS, COLORS_LIGHT

# ---------- Parametertest ----------

def parameterTest(n, m, epsWerte, dateiname="Parametertest", tol=1e-6, maxit=100000, numSeeds=10):
    """Fuehrt den Parametertest für verschiedene Epsilon-Werte bei festen Dimensionen n und m durch."""

    # Evtl. alte CSV-Datei löschen
    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    if dateipfad.is_file():
        dateipfad.unlink()

    for eps in epsWerte:
        for seed in range(numSeeds):

            mu, nu, c = datenGenerierenZufall(n,m,seed)

            ergebnisSinkhorn = sinkhorn(np.ones(m), c, mu, nu, eps, tol, maxit)
            ergebnisSimplex = simplexLoeser(mu, nu, c)

            sicherungParameterTest(dateiname, seed, ergebnisSinkhorn, ergebnisSimplex)
    
    diagrammParameterTest(dateiname, maxit)

#  ---------- Hilfsfunktionen ---------- 

def sicherungParameterTest(dateiname, seed, ergebnisSinkhorn, ergebnisSimplex):
    """Speichert die Ergebnisse des Parametertests in einer CSV-Datei."""

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
                "Iterationen",
                "Zeit [s]",
                "Marginalfehler",
                "Kosten",
                "Kostendifferenz",
            ])

        writer.writerow([
            seed,
            ergebnisSinkhorn["param"]["n"],
            ergebnisSinkhorn["param"]["m"],
            ergebnisSinkhorn["param"]["eps"],
            ergebnisSinkhorn["iterationen"],
            ergebnisSinkhorn["zeit"],
            ergebnisSinkhorn["residuum"],
            ergebnisSinkhorn["kosten"],
            abs(ergebnisSinkhorn["kosten"] - ergebnisSimplex["kosten"]),
        ])

def diagrammParameterTest(dateiname, maxit):
    """Erstellt ein Diagramm der Ergebnisse des Parametertests."""

    # ---- CSV-Datei ----
    df = pd.read_csv(CSV_ORDNER / f"{dateiname}.csv", sep=',', header=0)

    # Gruppierung der Daten nach eps, Berechnung des getrimmten Mittelwerts pro Spalte
    werte = df.columns[4:9].tolist()

    # Nicht-konvergente Werte Zeilen NaN setzen
    df["Konvergenz"] = df["Iterationen"] < maxit
    df[werte] = df[werte].where(df["Konvergenz"])

    # Durchschnitt für Anteil Konvergenz und Rest getrimmter Mittelwert
    df_gruppiert = df.groupby("eps", as_index=False).agg({
        "Konvergenz": "mean",
        "Iterationen": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
        "Zeit [s]": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
        "Marginalfehler": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
        "Kosten": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
        "Kostendifferenz": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
    }).sort_values("eps", ascending=False).reset_index(drop=True)

    # ---- Sicherung der Tabelle als .tex Datei ----
    df_latex = df_gruppiert.copy().sort_values("eps", ascending=False)

    # Darstellung in Zehnerpotenzen
    df_latex["eps"] = df_latex["eps"].map(lambda x: f"{x:.2e}" if pd.notnull(x) else "nan")

    # Erfolgsquote / Konvergenzrate als Prozentwert formatieren (z. B. 100% oder 10%)
    df_latex["Konvergenz"] = df_latex["Konvergenz"].map(lambda x: f"{x * 100:.0f}\\%" if pd.notnull(x) else "nan")

    # Nachkommastellen bearbeiten - Iterationen:
    df_latex["Iterationen"] = df_latex["Iterationen"].map(lambda x: int(np.ceil(x)) if pd.notnull(x) else "nan")

    # Nachkommastellen bearbeiten - Zeit, Marginalfehler, Transportkosten, Transportkosten-Differenz:  
    df_latex["Zeit [s]"] = df_latex["Zeit [s]"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Marginalfehler"] = df_latex["Marginalfehler"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Kosten"] = df_latex["Kosten"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")
    df_latex["Kostendifferenz"] = df_latex["Kostendifferenz"].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "nan")

    # Spaltennamen anpassen
    df_latex = df_latex.rename(columns={
        "eps": r"$\varepsilon$",
        "Konvergenz": "Konv.",
        "Iterationen": "Iter.",
        "Marginalfehler": "Marg.fehl.",
        "Kostendifferenz": "Kostendiff."
        })

    caption="Getrimmte Mittelwerte der Parameter in Abhängigkeit von $\\varepsilon$"

    dfZuLatex(df_latex, dateiname, caption)



# --------------------------------------------------------------
# ---------- Erweiterter Parametertest für Toleranzen ----------

def parameterTestTol(n, m, epsWerte, tolWerte, dateiname="Parametertest_Tol", maxit=100000, numSeeds=10):
    
    dateipfad = CSV_ORDNER / f"{dateiname}.csv"
    if dateipfad.is_file():
        dateipfad.unlink()

    # Schleife über Toleranzen, Epsilon-Werte und Seeds
    for tol in tolWerte:
        for eps in epsWerte:
            for seed in range(numSeeds):
                mu, nu, c = datenGenerierenZufall(n, m, seed)

                ergebnisSinkhorn = sinkhorn(np.ones(m), c, mu, nu, eps, tol, maxit)
                ergebnisSimplex = simplexLoeser(mu, nu, c)

                sicherungParameterTestTol(dateiname, seed, tol, ergebnisSinkhorn, ergebnisSimplex)
    
    diagrammParameterTestTol(dateiname, maxit)

# ---------- Hilfsfunktionen ---------- 

def sicherungParameterTestTol(dateiname, seed, tol, ergebnisSinkhorn, ergebnisSimplex):

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
                "tol",
                "Iterationen",
                "Zeit [s]",
                "Marginalfehler",
                "Kosten",
                "Kostendifferenz",
            ])

        writer.writerow([
            seed,
            ergebnisSinkhorn["param"]["n"],
            ergebnisSinkhorn["param"]["m"],
            ergebnisSinkhorn["param"]["eps"],
            tol,
            ergebnisSinkhorn["iterationen"],
            ergebnisSinkhorn["zeit"],
            ergebnisSinkhorn["residuum"],
            ergebnisSinkhorn["kosten"],
            abs(ergebnisSinkhorn["kosten"] - ergebnisSimplex["kosten"]),
        ])

def diagrammParameterTestTol(dateiname, maxit):

    df = pd.read_csv(CSV_ORDNER / f"{dateiname}.csv", sep=',', header=0)

    # Konvergenz ermitteln
    df["Konvergenz"] = df["Iterationen"] < maxit

    # Werte ausblenden, falls nicht konvergiert
    werte = df.columns[5:10].tolist()
    df[werte] = df[werte].where(df["Konvergenz"])

    # Groupby nach 'tol' und 'eps'
    df_gruppiert = df.groupby(["tol", "eps"], as_index=False).agg({
        "Konvergenz": "mean",
        "Kostendifferenz": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
        "Zeit [s]": lambda x: stats.trim_mean(x.dropna(), 0.1) if len(x.dropna()) > 0 else np.nan,
    }).sort_values(["tol", "eps"], ascending=[False, False]).reset_index(drop=True)

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx() # Zweite y-Achse für die Laufzeit

    tol_werte = sorted(df_gruppiert["tol"].unique(), reverse=True)
    colors = list(COLORS.values()) if isinstance(COLORS, dict) else COLORS

    # Für die Legende
    lines_tol = []
    labels_tol = []

    for i, tol in enumerate(tol_werte):
        df_tol = df_gruppiert[df_gruppiert["tol"] == tol].sort_values("eps", ascending=False)
        color = colors[i % len(colors)]
        label_tol = rf"tol = $10^{{{int(np.log10(tol))}}}$" if tol > 0 else f"tol = {tol}"

        # Achse links: Kostendifferenz (Durchgezogen)
        line_cost, = ax1.plot(
            df_tol["eps"],
            df_tol["Kostendifferenz"],
            linewidth=0.8,
            linestyle='-',
            color=color,
            alpha=0.85,
            label=label_tol
        )
        lines_tol.append(line_cost)
        labels_tol.append(label_tol)

        # Achse rechts: Zeit [s] (Gestrichelt)
        ax2.plot(
            df_tol["eps"],
            df_tol["Zeit [s]"],
            linewidth=0.8,
            linestyle='--',
            color=color,
            alpha=0.85
        )

        # Unterscheidung voll-konvergierter vs. teil-konvergierter Punkte
        voll_konv = df_tol[df_tol["Konvergenz"] == 1.0]
        teil_konv = df_tol[df_tol["Konvergenz"] < 1.0]

        ax1.scatter(
            voll_konv["eps"], voll_konv["Kostendifferenz"],
            color=color, marker='o', s=10, alpha=0.9, zorder=3
        )
        ax2.scatter(
            voll_konv["eps"], voll_konv["Zeit [s]"],
            color=color, marker='s', s=8, alpha=0.6, zorder=3
        )

        # Rot gefüllte Marker bei < 100% Konvergenz
        if not teil_konv.empty:
            ax1.scatter(
                teil_konv["eps"], teil_konv["Kostendifferenz"],
                color='red', marker='o', s=14, zorder=4
            )
            ax2.scatter(
                teil_konv["eps"], teil_konv["Zeit [s]"],
                color='red', marker='s', s=12, zorder=4
            )

    # Achsen-Skalierung und Formatierung
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel(r"$\varepsilon$")
    ax1.set_ylabel("Getrimmte Kostendifferenz (Durchgezogen)", color='black')
    ax2.set_ylabel("Getrimmte Laufzeit [s] (Gestrichelt)", color='black')

    ax1.set_title(r"Kostendifferenz und Laufzeit in Abhängigkeit von $\varepsilon$ und der Toleranz")
    ax1.grid(True, which="both", alpha=0.2, linewidth=0.5)

    # Legende
    dummy_cost, = ax1.plot([], [], color='gray', linestyle='-', linewidth=0.8, label='Kostendifferenz')
    dummy_time, = ax1.plot([], [], color='gray', linestyle='--', linewidth=0.8, label='Laufzeit [s]')
    dummy_red = ax1.scatter([], [], color='red', marker='o', s=14, label='Konvergenz $<100\\%$')

    leg_style = ax1.legend(
        handles=[dummy_cost, dummy_time, dummy_red], 
        loc="upper right", 
        bbox_to_anchor=(0.99, 0.99), 
        fontsize="x-small", 
        framealpha=0.85
    )

    leg_tol = ax1.legend(
        lines_tol, 
        labels_tol, 
        loc="upper right", 
        bbox_to_anchor=(0.99, 0.77), 
        fontsize="x-small", 
        framealpha=0.85, 
        title="Toleranzen",
        title_fontsize="x-small"
    )

    ax1.add_artist(leg_style)

    fig.tight_layout()
    fig.savefig(DIAGRAMM_ORDNER / f"{dateiname}.pdf", bbox_inches="tight")

    plt.close(fig)