# Quadratische Regularisierung und Penalisierung des Hitchcock-Problems — Konvergenzanalysis

Dieses Repository enthält den Quellcode sowie die Skripte zur Generierung der numerischen Testergebnisse der Bachelorarbeit von **Frieda Winning** an der Technischen Universität Dortmund (Fakultät für Mathematik, Lehrstuhl für Numerische Analysis und Optimierung, Sommersemester 2026).

---

## Übersicht

Zweck des Codes ist die numerische Untersuchung des **Sinkhorn-Algorithmus** für das entropisch regularisierte Transportproblem hinsichtlich seiner Leistung, Laufzeit und Genauigkeit. 

---

## Projektstruktur

Das Projekt ist grob in drei Bereiche aufgeteilt. Der Ordner *Algorithmen_Funktionen* enthält die Implementierungen der verschiedenen Lösungsverfahren sowie die der Datengenerierung. Der Ordner *Tests* enthält die Skripte zur Ausführung der numerischen Experimente. In *Ergebnisse* sind diese Auswertungen als CSV-, PDF- oder TEX-Dateien zu finden. Durch Ausführen der Datei main.py können alle Tests aufgerufen werden.

## Systemanforderungen & Installation

Der Code wurde unter **Python 3.14** entwickelt und getestet.

### Benötigte Bibliotheken
Die externen Abhängigkeiten und deren Verwendungszweck im Projekt:

- **`numpy`**: Datenverwaltung, Vektorarithmetik und Erzeugung stochastischer Testdaten
- **`POT`** (Python Optimal Transport, `import ot`): Vergleichsmessungen mit dem Simplex-Algorithmus
- **`pandas`**: Strukturierte Auswertung und Export der Messergebnisse
- **`matplotlib`**: Erstellung und Anpassung der Plot-Visualisierungen

*(Hinweis: Weitere genutzte Module wie `time`, `sys`, `csv`, `math` und `cycler` sind Bestandteil der Python-Standardbibliothek.)*

### Ein-Klick-Installation
Sämtliche benötigten Pakete können über den folgenden Befehl installiert werden:

```bash
pip install numpy POT pandas matplotlib
