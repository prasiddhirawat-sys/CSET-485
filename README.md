# CSET485 Assignment 1 – German Credit (roll no. ending 0113, random_state = 113)

* `CSET485_Assignment1_GermanCredit.ipynb` – full analysis (preprocessing, Part 1, Part 2.1–2.4). Run top to bottom; writes `results.json`, `fairness_sweep.csv`, `figs/`.
* `build_report.py` – builds `CSET485_Assignment1_Report.pdf` from `results.json` (every number in the PDF comes from the notebook run).
* `german_credit.csv` – copy of the Statlog German Credit data (decoded labels). Source mirror: https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv (same 1,000 rows as the UCI file).
* `analysis.py` – the notebook as a plain script.

Reproduce: `pip install -r requirements.txt && jupyter nbconvert --to notebook --execute CSET485_Assignment1_GermanCredit.ipynb && python build_report.py`
