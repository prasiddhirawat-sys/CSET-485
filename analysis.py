# %% [markdown]
# # CSET485 – AI and Society · Assignment 1
# **Dataset:** German Credit (roll number ending 0113 → last digit 3) · **random_state = 113** · 70/30 split
#
# Target: `credit_risk` (1 = approve / good credit, 0 = reject / bad credit).
# Every number quoted in the PDF report is produced by this notebook (run top to bottom). Results are saved to `results.json`, figures to `figs/`.
#
# **Data source note.** The UCI archive link in the brief was not reachable from my build environment, so the data is loaded from a GitHub mirror of the *same* Statlog German Credit data (1,000 rows, 20 attributes, 700 good / 300 bad; categories decoded from the A11/A12… codes into readable labels). Category counts (e.g. personal_status_sex = 548/310/92/50) match the UCI file. A copy is stored as `german_credit.csv`.

# %%
import json, os, re, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
SEED = 113
os.makedirs("figs", exist_ok=True)
R = {}   # every reported number goes here
URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv("german_credit.csv") if os.path.exists("german_credit.csv") else pd.read_csv(URL)
print(df.shape); df.head()

# %% [markdown]
# ## 0. Data preprocessing
# * **Missing values:** checked for NaNs and for placeholder strings; none exist → nothing to impute.
# * **Categorical encoding:** 13 categorical attributes one-hot encoded with `drop_first=True` (avoids the dummy-variable trap so OLS/Logit are full-rank). The reference category is the alphabetically first level.
# * **Scaling:** numeric attributes are z-scored; the scaler is **fit on the training split only** and applied to the test split (no leakage). Coefficients are therefore "per 1 SD" for numeric predictors.
# * **Outliers:** counted with the 1.5×IQR rule. `amount` is strongly right-skewed (max 18,424 DM vs median ≈2,320) so it is **log-transformed**; no rows are removed because the extreme values are genuine large loans and n is small.
# * **Split:** `train_test_split(test_size=0.3, random_state=113)`, not stratified (the brief does not ask for stratification).

# %%
y = df["credit_risk"].astype(int)
R["n"] = len(df); R["base_rate_approve"] = float(y.mean())
R["missing_values_total"] = int(df.isna().sum().sum())
placeholders = int(df.astype(str).isin(["?", "NA", "nan", "", " "]).sum().sum())
R["placeholder_cells"] = placeholders
num_cols = ["duration", "amount", "installment_rate", "present_residence", "age", "number_credits", "people_liable"]
cat_cols = [c for c in df.columns if c not in num_cols + ["credit_risk"]]
R["n_categorical"] = len(cat_cols)
out = {}
for c in ["duration", "amount", "age"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    out[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
R["iqr_outliers"] = out
R["amount_skew_raw"] = float(stats.skew(df.amount)); R["amount_skew_log"] = float(stats.skew(np.log(df.amount)))
print(R["missing_values_total"], placeholders, out, R["amount_skew_raw"], R["amount_skew_log"])

X = df.drop(columns="credit_risk").copy()
X["amount"] = np.log(X["amount"]); X = X.rename(columns={"amount": "log_amount"})
num_cols = [("log_amount" if c == "amount" else c) for c in num_cols]
# rare levels (<30 rows) are pooled into one level; the reference level is the MOST FREQUENT level of each attribute
# (an alphabetical reference such as purpose="business", n=9, is entirely 'approve' in the training split -> complete separation)
pooled = {}
for c in cat_cols:
    vc = X[c].value_counts(); rare = list(vc[vc < 30].index)
    if rare: pooled[c] = rare; X[c] = X[c].where(~X[c].isin(rare), "rare_levels")
R["pooled_rare_levels"] = pooled
parts = [X.drop(columns=cat_cols)]; refs = {}
for c in cat_cols:
    ref = X[c].value_counts().idxmax(); refs[c] = ref
    D = pd.get_dummies(X[c], prefix=c, dtype=float).drop(columns=f"{c}_{ref}"); parts.append(D)
X = pd.concat(parts, axis=1); R["reference_levels"] = refs
X.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in X.columns]
assert X.columns.is_unique
R["n_features"] = X.shape[1]
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
R["n_train"], R["n_test"] = len(Xtr_raw), len(Xte_raw)
sc = StandardScaler().fit(Xtr_raw[num_cols])
Xtr, Xte = Xtr_raw.copy(), Xte_raw.copy()
Xtr[num_cols] = sc.transform(Xtr_raw[num_cols]); Xte[num_cols] = sc.transform(Xte_raw[num_cols])
R["test_base_rate"] = float(yte.mean()); R["train_base_rate"] = float(ytr.mean())
print(X.shape, Xtr.shape, Xte.shape)

def add_c(M): return sm.add_constant(M, has_constant="add")
def fit_logit(yy, XX):
    try:
        m = sm.Logit(yy, add_c(XX)).fit(disp=0, maxiter=300)
        ok = m.mle_retvals["converged"]
    except Exception:
        ok = False
    if not ok:
        m = sm.Logit(yy, add_c(XX)).fit(disp=0, maxiter=5000, method="bfgs"); ok = m.mle_retvals["converged"]
    assert ok and np.abs(m.params).max() < 15, "logit did not converge / separation"
    return m

# %% [markdown]
# ## Part 1 — Build
# Model A = OLS on approval indicator (linear probability model); Model B = logistic regression. Both fit on the 70% training split, evaluated on the 30% test split.
# **Label convention:** both models predict the *positive class* (approve = 1). "High" below means "predicted score at or above the cut-off" (i.e. predicted approve) under each rule, exactly as the brief's thresholding rules are written.

# %%
ols = sm.OLS(ytr, add_c(Xtr)).fit()
logit = fit_logit(ytr, Xtr)
R["logit_converged"] = bool(logit.mle_retvals["converged"]); R["logit_max_abs_coef"] = float(np.abs(logit.params).max())
pA = ols.predict(add_c(Xte)); pB = logit.predict(add_c(Xte))
med = float(np.median(pA))
fA = (pA >= med).astype(int); fB = (pB >= 0.5).astype(int)
ct = pd.crosstab(fA.rename("A: OLS>=median"), fB.rename("B: Logit>=0.5"))
print(ct)
mism = int((fA != fB).sum())
R["part1"] = dict(
    median_ols=med, ols_min=float(pA.min()), ols_max=float(pA.max()), ols_outside_01=int(((pA < 0) | (pA > 1)).sum()),
    ols_mean=float(pA.mean()), logit_mean=float(pB.mean()),
    n_flag_A=int(fA.sum()), n_flag_B=int(fB.sum()),
    confusion=ct.values.tolist(),  # rows: A low/high, cols: B low/high
    mismatch=mism, mismatch_pct=100 * mism / len(yte),
    A_high_B_low=int(((fA == 1) & (fB == 0)).sum()), A_low_B_high=int(((fA == 0) & (fB == 1)).sum()),
    ols_share_ge_05=float((pA >= .5).mean()), mismatch_if_ols_at_05=int(((pA >= .5).astype(int) != fB).sum()),
    corr_pearson=float(np.corrcoef(pA, pB)[0, 1]), corr_spearman=float(stats.spearmanr(pA, pB)[0]),
    auc_A=float(roc_auc_score(yte, pA)), auc_B=float(roc_auc_score(yte, pB)),
    acc_A=float(accuracy_score(yte, fA)), acc_B=float(accuracy_score(yte, fB)),
    acc_A_at05=float(accuracy_score(yte, (pA >= .5).astype(int))),
    pctile_of_05_in_ols=float((pA < .5).mean() * 100),
    median_logit=float(np.median(pB)),
    mean_pB_mismatch_A_low_B_high=float(pB[(fA == 0) & (fB == 1)].mean()),
    mean_pA_mismatch_A_low_B_high=float(pA[(fA == 0) & (fB == 1)].mean()),
)
for k, v in R["part1"].items(): print(k, v)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(pA, bins=25, color="#4C72B0"); ax[0].axvline(med, color="r", ls="--", label=f"median = {med:.3f}"); ax[0].axvline(.5, color="k", ls=":", label="0.5")
ax[0].set_title("Model A (OLS) test predictions"); ax[0].legend(); ax[0].set_xlabel("predicted P(approve)")
ax[1].scatter(pA, pB, c=np.where(fA != fB, "#C44E52", "#8C8C8C"), s=14)
ax[1].axvline(med, color="r", ls="--"); ax[1].axhline(.5, color="k", ls=":")
ax[1].set_xlabel("OLS prediction"); ax[1].set_ylabel("Logit probability"); ax[1].set_title("Red = mismatched individuals")
plt.tight_layout(); plt.savefig("figs/part1.png", dpi=150); plt.close()

# %% [markdown]
# ## Part 2.1 — Simpson's-paradox hunt
# Compact specification (9 predictors) so that subgroup fits on ~300 rows are stable; fitted on the **full 1,000 rows**, predictors standardised on the full data.
# Split variable chosen: **sex** (derived from `personal_status_sex`: female vs male), because it is the sensitive attribute used again in 2.4. Key predictor: **age** (per 1 SD).
# I also scanned five other splits × all predictors (table below) to see what is out there. *This scan is exploratory – with ~50 comparisons, a few |z|>1.96 would arise by chance.*

# %%
z = lambda s: (s - s.mean()) / s.std()
C = pd.DataFrame({
    "duration": z(df.duration), "log_amount": z(np.log(df.amount)), "age": z(df.age), "installment_rate": z(df.installment_rate),
    "number_credits": z(df.number_credits),
    "no_checking_account": (df.status == "no checking account").astype(float),
    "critical_credit_history": df.credit_history.str.startswith("critical").astype(float),
    "telephone": (df.telephone == "yes").astype(float), "own_housing": (df.housing == "own").astype(float)})
female = df.personal_status_sex.str.startswith("female")
def coef_table(m, c):
    ci = m.conf_int().loc[c]; return dict(coef=float(m.params[c]), lo=float(ci[0]), hi=float(ci[1]), p=float(m.pvalues[c]), se=float(m.bse[c]))
def run_split(mask, name, preds):
    f = fit_logit(y, C[preds]); a = fit_logit(y[mask], C.loc[mask, preds]); b = fit_logit(y[~mask], C.loc[~mask, preds])
    rows = []
    for c in preds:
        A_, B_, F_ = coef_table(a, c), coef_table(b, c), coef_table(f, c)
        zz = (A_["coef"] - B_["coef"]) / np.hypot(A_["se"], B_["se"])
        rows.append(dict(split=name, predictor=c, full=F_["coef"], A=A_["coef"], B=B_["coef"], pA=A_["p"], pB=B_["p"], z_diff=float(zz),
                         p_diff=float(2 * (1 - stats.norm.cdf(abs(zz)))), sign_flip=bool(np.sign(A_["coef"]) != np.sign(B_["coef"]))))
    return f, a, b, pd.DataFrame(rows)
splits = {"sex (female=A)": (female, None), "age>33 (A)": (df.age > 33, "age"), "age<25 (A)": (df.age < 25, "age"),
          "own_housing (A)": (C.own_housing == 1, "own_housing"), "telephone (A)": (C.telephone == 1, "telephone"),
          "skilled+ job (A)": (df.job.isin(["skilled employee/official", "management/self-employed/highly qualified employee/officer"]), None)}
scan = []
for nm, (mk, drop) in splits.items():
    preds = [c for c in C.columns if c != drop]
    scan.append(run_split(mk, nm, preds)[3])
scan = pd.concat(scan, ignore_index=True)
R["scan_n_comparisons"] = int(len(scan)); R["scan_n_sig_005"] = int((scan.p_diff < .05).sum())
R["scan_top"] = scan.reindex(scan.z_diff.abs().sort_values(ascending=False).index).head(6)[["split", "predictor", "full", "A", "B", "z_diff", "p_diff"]].round(3).to_dict("records")
print(scan[scan.p_diff < .05].round(3))

preds = list(C.columns)
f21, a21, b21, tab21 = run_split(female, "sex", preds)
R["s21"] = dict(n_A=int(female.sum()), n_B=int((~female).sum()),
                age_full=coef_table(f21, "age"), age_A=coef_table(a21, "age"), age_B=coef_table(b21, "age"),
                z=float(tab21.set_index("predictor").loc["age", "z_diff"]), p_diff=float(tab21.set_index("predictor").loc["age", "p_diff"]),
                bonferroni_p=float(min(1, tab21.set_index("predictor").loc["age", "p_diff"] * len(scan))),
                approve_rate_A=float(y[female].mean()), approve_rate_B=float(y[~female].mean()),
                mean_age_A=float(df.age[female].mean()), mean_age_B=float(df.age[~female].mean()),
                age_or_A=float(np.exp(a21.params["age"])), age_or_B=float(np.exp(b21.params["age"])), age_or_full=float(np.exp(f21.params["age"])),
                age_sd=float(df.age.std()))
# per-year effect in raw units for interpretation
R["s21"]["age_or_per10y_A"] = float(np.exp(a21.params["age"] / df.age.std() * 10)); R["s21"]["age_or_per10y_B"] = float(np.exp(b21.params["age"] / df.age.std() * 10))
R["s21"]["age_or_per10y_full"] = float(np.exp(f21.params["age"] / df.age.std() * 10))
R["s21_table"] = tab21.round(4).to_dict("records")
print(json.dumps(R["s21"], indent=1))
print(tab21.round(3))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for i, (nm, m, col) in enumerate([("Full (n=1000)", f21, "k"), (f"Female (n={int(female.sum())})", a21, "#C44E52"), (f"Male (n={int((~female).sum())})", b21, "#4C72B0")]):
    t = coef_table(m, "age"); ax.errorbar(t["coef"], i, xerr=[[t["coef"] - t["lo"]], [t["hi"] - t["coef"]]], fmt="o", color=col, capsize=4)
ax.axvline(0, color="grey", ls="--"); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Full", "Female", "Male"]); ax.invert_yaxis()
ax.set_xlabel("logit coefficient of age (per 1 SD), 95% CI"); ax.set_title("Age effect on P(approve) by sex")
plt.tight_layout(); plt.savefig("figs/part21.png", dpi=150); plt.close()

# %% [markdown]
# ## Part 2.2 — Omitted-variable bias
# Model B (all features, 700 training rows) vs the same model **without `duration`** (loan duration in months). Duration is plausibly a confounder of the amount→approval relationship: bigger loans run longer, and longer loans are riskier.

# %%
full = logit
red_cols = [c for c in Xtr.columns if c != "duration"]
red = fit_logit(ytr, Xtr[red_cols])
cmp_ = pd.DataFrame({"full": full.params, "reduced": red.params}).drop("const").drop("duration")
cmp_["abs_change"] = cmp_["reduced"] - cmp_["full"]
cmp_["pct_chg"] = 100 * (cmp_["reduced"] - cmp_["full"]) / cmp_["full"].abs()
cmp_["p_full"] = full.pvalues; cmp_["p_red"] = red.pvalues
R["corr_dur_logamount_train"] = float(np.corrcoef(Xtr_raw.duration, Xtr_raw.log_amount)[0, 1])
R["s22_duration_coef"] = float(full.params["duration"]); R["s22_duration_p"] = float(full.pvalues["duration"])
key = ["log_amount", "installment_rate", "age", "number_credits", "present_residence"]
R["s22_key"] = cmp_.loc[key].round(4).reset_index().rename(columns={"index": "predictor"}).to_dict("records")
big = cmp_[(cmp_["full"].abs() > 0.1)].pct_chg.abs().sort_values(ascending=False).head(8).index
R["s22_top"] = cmp_.loc[big].round(4).reset_index().rename(columns={"index": "predictor"}).to_dict("records")
R["s22_max_abs_shift_pred"] = cmp_.abs_change.abs().idxmax(); R["s22_max_abs_shift_row"] = cmp_.loc[R["s22_max_abs_shift_pred"]].round(4).to_dict()
R["s22_sign_flips"] = [c for c in cmp_.index if np.sign(cmp_["full"][c]) != np.sign(cmp_["reduced"][c])]
R["s22_pct_median_abs"] = float(cmp_.pct_chg.abs().median())
R["s22_pct_log_amount"] = float(cmp_.pct_chg["log_amount"])
R["s22_auc_full_test"] = float(roc_auc_score(yte, full.predict(add_c(Xte)))); R["s22_auc_red_test"] = float(roc_auc_score(yte, red.predict(add_c(Xte[red_cols]))))
R["s22_logamount_p_full"] = float(full.pvalues["log_amount"]); R["s22_logamount_p_red"] = float(red.pvalues["log_amount"])
print(cmp_.loc[key].round(3)); print(cmp_.loc[big].round(3)); print({k: R[k] for k in ["corr_dur_logamount_train", "s22_max_abs_shift_pred", "s22_sign_flips", "s22_auc_full_test", "s22_auc_red_test"]})
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.bar(np.arange(len(key)) - .2, cmp_.loc[key].full, .4, label="full model", color="#4C72B0"); ax.bar(np.arange(len(key)) + .2, cmp_.loc[key].reduced, .4, label="without duration", color="#DD8452")
ax.set_xticks(range(len(key))); ax.set_xticklabels(key, rotation=20, fontsize=8); ax.axhline(0, color="k", lw=.6); ax.legend(); ax.set_title("Coefficients before/after dropping duration")
plt.tight_layout(); plt.savefig("figs/part22.png", dpi=150); plt.close()

# %% [markdown]
# ## Part 2.3 — Adversarial subset
# Test-set cases where Model B is confident (P ≥ 0.75 or P ≤ 0.25) and wrong. If more than 50 exist, 50 are sampled with `random_state=113`.

# %%
pt = pd.Series(pB.values, index=yte.index)
fp = (pt >= .75) & (yte == 0); fn = (pt <= .25) & (yte == 1)
wrong = fp | fn
R["s23"] = dict(n_fp=int(fp.sum()), n_fn=int(fn.sum()), n_wrong=int(wrong.sum()), pct_wrong=float(100 * wrong.mean()),
                n_conf=int(((pt >= .75) | (pt <= .25)).sum()), n_test=len(yte))
R["s23"]["pct_wrong_of_confident"] = 100 * R["s23"]["n_wrong"] / R["s23"]["n_conf"]
idx = yte.index[wrong]
samp = pd.Series(idx).sample(n=min(50, len(idx)), random_state=SEED).values if len(idx) > 50 else idx.values
R["s23"]["n_subset"] = int(len(samp)); sub_fp = int(fp.loc[samp].sum()); R["s23"]["subset_fp"] = sub_fp; R["s23"]["subset_fn"] = int(len(samp) - sub_fp)
age_raw = df.age; fem = female
R["s23"]["age_subset_mean"] = float(age_raw.loc[samp].mean()); R["s23"]["age_test_mean"] = float(age_raw.loc[yte.index].mean())
R["s23"]["age_subset_median"] = float(age_raw.loc[samp].median()); R["s23"]["age_test_median"] = float(age_raw.loc[yte.index].median())
R["s23"]["pct_under25_subset"] = float(100 * (age_raw.loc[samp] < 25).mean()); R["s23"]["pct_under25_test"] = float(100 * (age_raw.loc[yte.index] < 25).mean())
R["s23"]["pct_female_subset"] = float(100 * fem.loc[samp].mean()); R["s23"]["pct_female_test"] = float(100 * fem.loc[yte.index].mean())
R["s23"]["mean_duration_subset"] = float(df.duration.loc[samp].mean()); R["s23"]["mean_duration_test"] = float(df.duration.loc[yte.index].mean())
R["s23"]["mean_amount_subset"] = float(df.amount.loc[samp].mean()); R["s23"]["mean_amount_test"] = float(df.amount.loc[yte.index].mean())
# which binary traits differ most?
rows = []
bin_cols = [c for c in Xte_raw.columns if c not in num_cols]
for c in bin_cols:
    a_ = Xte_raw.loc[samp, c].mean(); b_ = Xte_raw[c].mean()
    tab = [[Xte_raw.loc[samp, c].sum(), len(samp) - Xte_raw.loc[samp, c].sum()], [Xte_raw[c].sum() - Xte_raw.loc[samp, c].sum(), len(Xte_raw) - len(samp) - (Xte_raw[c].sum() - Xte_raw.loc[samp, c].sum())]]
    rows.append(dict(feature=c, pct_subset=100 * a_, pct_test=100 * b_, diff=100 * (a_ - b_), p=float(stats.fisher_exact(tab)[1]), coef=float(logit.params[c])))
tr = pd.DataFrame(rows).sort_values("diff", key=abs, ascending=False)
R["s23_traits"] = tr.head(8).round(3).to_dict("records")
print(R["s23"]); print(tr.head(10).round(3))
# FP vs FN breakdown of top trait
top = tr.iloc[0].feature
R["s23_top_trait"] = top
R["s23_top_by_type"] = dict(fp_pct=float(100 * Xte_raw.loc[[i for i in samp if fp.loc[i]], top].mean()) if sub_fp else None,
                            fn_pct=float(100 * Xte_raw.loc[[i for i in samp if fn.loc[i]], top].mean()) if (len(samp) - sub_fp) else None)
R["s23_top_in_conf_correct"] = float(100 * Xte_raw.loc[((pt >= .75) | (pt <= .25)) & ~wrong, top].mean())
# numeric traits
R["s23_num"] = {c: dict(subset=float(Xte_raw.loc[samp, c].mean()), test=float(Xte_raw[c].mean())) for c in ["duration", "log_amount", "age", "installment_rate"]}
# 2nd view: error rate among confident predictions by top-trait
mask_t = Xte_raw[top] == 1; conf = (pt >= .75) | (pt <= .25)
R["s23_err_rate_top_trait"] = dict(n_conf_with=int((conf & mask_t).sum()), err_with=float(100 * (wrong & mask_t).sum() / max(1, (conf & mask_t).sum())),
                                   n_conf_without=int((conf & ~mask_t).sum()), err_without=float(100 * (wrong & ~mask_t).sum() / max(1, (conf & ~mask_t).sum())))
# calibration check: how many confident errors would a perfectly calibrated model make?
confm = (pt >= .75) | (pt <= .25)
exp_wrong = float(np.where(pt[confm] >= .5, 1 - pt[confm], pt[confm]).sum())
R["s23_calib"] = dict(expected_wrong_if_calibrated=exp_wrong, observed_wrong=int(wrong.sum()),
                      n_hi=int((pt >= .75).sum()), mean_pred_hi=float(pt[pt >= .75].mean()), actual_rate_hi=float(yte[pt >= .75].mean()),
                      n_lo=int((pt <= .25).sum()), mean_pred_lo=float(pt[pt <= .25].mean()), actual_rate_lo=float(yte[pt <= .25].mean()))
crit = "credit_history_critical_account_other_credits_existing"
tabx = [[int((conf & (Xte_raw[crit] == 0) & wrong).sum()), int((conf & (Xte_raw[crit] == 0) & ~wrong).sum())],
        [int((conf & (Xte_raw[crit] == 1) & wrong).sum()), int((conf & (Xte_raw[crit] == 1) & ~wrong).sum())]]
R["s23_crit"] = dict(subset_pct_no_crit=float(100 * (1 - Xte_raw.loc[samp, crit].mean())), test_pct_no_crit=float(100 * (1 - Xte_raw[crit].mean())),
                     n_no_crit_in_subset=int((Xte_raw.loc[samp, crit] == 0).sum()), fisher_p_error_rate=float(stats.fisher_exact(tabx)[1]),
                     approve_rate_crit_train=float(ytr[Xtr_raw[crit] == 1].mean()), approve_rate_nocrit_train=float(ytr[Xtr_raw[crit] == 0].mean()), coef_crit=float(logit.params[crit]),
                     coef_status_lt100=float(logit.params["status_100_DM"]), coef_female_status=float(logit.params["personal_status_sex_female_divorced_separated_married"]), coef_unknown_savings=float(logit.params["savings_unknown_no_savings_account"]), coef_status_0_200=float(logit.params["status_0_200_DM"]))
R["s23_female_fisher_p"] = float(stats.fisher_exact([[int(fem.loc[samp].sum()), len(samp) - int(fem.loc[samp].sum())], [int(fem.loc[yte.index].sum()) - int(fem.loc[samp].sum()), len(yte) - len(samp) - (int(fem.loc[yte.index].sum()) - int(fem.loc[samp].sum()))]])[1])
bins_ = [0, 24, 34, 49, 120]; labs_ = ["<25", "25-34", "35-49", "50+"]
ab_s = pd.cut(age_raw.loc[samp], bins_, labels=labs_).value_counts(normalize=True).reindex(labs_) * 100
ab_t = pd.cut(age_raw.loc[yte.index], bins_, labels=labs_).value_counts(normalize=True).reindex(labs_) * 100
R["s23_age_buckets"] = {l: dict(subset=float(ab_s[l]), test=float(ab_t[l])) for l in labs_}
nochk = (df.status == "no checking account"); unksav = (df.savings == "unknown/no savings account")
fps_ = [i for i in samp if fp.loc[i]]; fns_ = [i for i in samp if fn.loc[i]]
R["s23_mech"] = dict(mean_P_fp=float(pt.loc[fps_].mean()), mean_P_fn=float(pt.loc[fns_].mean()),
                     pct_nochk_fp=float(100 * nochk.loc[fps_].mean()), pct_nochk_test=float(100 * nochk.loc[yte.index].mean()),
                     pct_unksav_fp=float(100 * unksav.loc[fps_].mean()), pct_unksav_test=float(100 * unksav.loc[yte.index].mean()),
                     pct_nochk_or_unksav_fp=float(100 * (nochk | unksav).loc[fps_].mean()),
                     pct_fn_lowstatus=float(100 * (df.status.isin(["... < 100 DM", "0 <= ... < 200 DM"])).loc[fns_].mean()),
                     approve_rate_nochk=float(y[nochk].mean()), approve_rate_unksav=float(y[unksav].mean()),
                     fp_err_rate_hi_conf=float(100 * fp.sum() / max(1, (pt >= .75).sum())), fn_err_rate_lo_conf=float(100 * fn.sum() / max(1, (pt <= .25).sum())))
print(R["s23_calib"], R["s23_crit"], R["s23_mech"], R["s23_age_buckets"])
print(R["s23_top_by_type"], R["s23_err_rate_top_trait"], R["s23_top_in_conf_correct"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(pt, bins=20, color="#8C8C8C", label="test set"); ax[0].hist(pt.loc[samp], bins=20, color="#C44E52", alpha=.8, label="adversarial subset")
ax[0].set_xlabel("P(approve)"); ax[0].legend(); ax[0].set_title("Confident-wrong cases")
t8 = tr.head(8).iloc[::-1]; ax[1].barh(t8.feature.str[:32], t8["diff"], color="#C44E52"); ax[1].set_title("% points: subset minus test prevalence"); ax[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig("figs/part23.png", dpi=150); plt.close()

# %% [markdown]
# ## Part 2.4 — Fairness trade-off
# Sensitive variable: **sex** (Group A = female, Group B = male), test set (n=300). Thresholds t ∈ {0.05, 0.15, …, 0.95}; Ŷ=1 ("approve") if P ≥ t.
# DP = P(Ŷ=1|A)/P(Ŷ=1|B); EO = min(TPR_A/TPR_B, FPR_A/FPR_B); PP = PPV_A/PPV_B. Positive class = approve. 95% bootstrap intervals (1,000 resamples of the test set, seed 113) show how noisy ratios are with ~90 women in the test set.

# %%
ts = np.round(np.arange(.05, .96, .1), 2)
gA = female.loc[yte.index].values; yv = yte.values; pv = pB.values
def metrics(yv, pv, gA, t):
    yh = (pv >= t).astype(int); out = {}
    def rates(m):
        P = (yv[m] == 1).sum(); N = (yv[m] == 0).sum()
        tp = ((yh[m] == 1) & (yv[m] == 1)).sum(); fpp = ((yh[m] == 1) & (yv[m] == 0)).sum()
        sel = yh[m].mean(); tpr = tp / P if P else np.nan; fpr = fpp / N if N else np.nan; ppv = tp / (tp + fpp) if (tp + fpp) else np.nan
        return sel, tpr, fpr, ppv
    sA, tA, fA_, pA_ = rates(gA); sB, tB, fB_, pB_ = rates(~gA)
    dv = lambda a, b: a / b if (b and not np.isnan(b) and not np.isnan(a)) else np.nan
    DP = dv(sA, sB); r1 = dv(tA, tB); r2 = dv(fA_, fB_)
    EO = np.nanmin([r1, r2]) if not (np.isnan(r1) and np.isnan(r2)) else np.nan
    PP = dv(pA_, pB_)
    return dict(t=float(t), DP=DP, EO=EO, PP=PP, acc=float((yh == yv).mean()), accA=float((yh[gA] == yv[gA]).mean()), accB=float((yh[~gA] == yv[~gA]).mean()),
                sel=float(yh.mean()), selA=sA, selB=sB, tprA=tA, tprB=tB, fprA=fA_, fprB=fB_, ppvA=pA_, ppvB=pB_, TPR_ratio=r1, FPR_ratio=r2)
rng = np.random.default_rng(SEED)
sweep = pd.DataFrame([metrics(yv, pv, gA, t) for t in ts])
boot = {k: [] for k in ["DP", "EO", "PP"]}
n = len(yv); B_ = 1000; bi = [rng.integers(0, n, n) for _ in range(B_)]
for t in ts:
    vals = [metrics(yv[i], pv[i], gA[i], t) for i in bi]
    for k in boot: boot[k].append(np.nanpercentile([v[k] for v in vals], [2.5, 97.5]) if not np.all(np.isnan([v[k] for v in vals])) else [np.nan, np.nan])
for k in boot: sweep[k + "_lo"] = [b[0] for b in boot[k]]; sweep[k + "_hi"] = [b[1] for b in boot[k]]
R["s24_group_sizes"] = dict(A_female=int(gA.sum()), B_male=int((~gA).sum()), pos_rate_A=float(yv[gA].mean()), pos_rate_B=float(yv[~gA].mean()))
R["s24_sweep"] = sweep.round(4).replace({np.nan: None}).to_dict("records")
pd.set_option("display.width", 250); print(sweep[["t", "DP", "EO", "PP", "acc", "accA", "accB", "selA", "selB", "tprA", "tprB", "fprA", "fprB", "ppvA", "ppvB"]].round(3))

# trade-off quantification. t=0.05 approves everybody, so DP=EO=1 trivially -> excluded as degenerate ("fairness by approving all")
R["s24_degenerate_t005"] = dict(overall_approve_rate=float(sweep.sel[0]), acc=float(sweep.acc[0]))
ok = sweep[sweep.t > 0.05].dropna(subset=["DP", "EO"]).reset_index(drop=True)
R["s24_spearman_DP_EO"] = float(stats.spearmanr(ok.DP, ok.EO)[0]); R["s24_EO_below_DP_count"] = int((ok.EO < ok.DP).sum()); R["s24_n_thresholds"] = int(len(ok))
pairs = []
for i_ in range(len(ok) - 1):
    a, b = ok.iloc[i_], ok.iloc[i_ + 1]
    for frm, to in [(a, b), (b, a)]:
        if to.DP > frm.DP and to.EO < frm.EO:
            pairs.append(dict(t_from=float(frm.t), t_to=float(to.t), DP_from=float(frm.DP), DP_to=float(to.DP), EO_from=float(frm.EO), EO_to=float(to.EO),
                              DP_gain_pct=float(100 * (to.DP - frm.DP) / frm.DP), EO_loss_pct=float(100 * (frm.EO - to.EO) / frm.EO)))
for p_ in pairs: p_["cost"] = p_["EO_loss_pct"] / p_["DP_gain_pct"]
R["s24_pairs_opposite"] = pairs
R["s24_best"] = dict(t_bestDP=float(ok.loc[ok.DP.sub(1).abs().idxmin()].t), DP_best=float(ok.DP.sub(1).abs().sub(1).abs().min()) if False else float(ok.loc[ok.DP.sub(1).abs().idxmin()].DP),
                     t_bestEO=float(ok.loc[ok.EO.sub(1).abs().idxmin()].t), EO_best=float(ok.loc[ok.EO.sub(1).abs().idxmin()].EO),
                     t_bestPP=float(ok.loc[ok.PP.sub(1).abs().idxmin()].t), PP_best=float(ok.loc[ok.PP.sub(1).abs().idxmin()].PP),
                     t_bestacc=float(ok.loc[ok.acc.idxmax()].t), acc_best=float(ok.acc.max()))
# threshold where PP is (almost) perfect vs where DP/EO are best
r35 = sweep[sweep.t == 0.35].iloc[0]
R["s24_t035"] = {k: float(r35[k]) for k in ["DP", "EO", "PP", "acc", "accA", "accB", "sel", "selA", "selB", "tprA", "tprB", "fprA", "fprB", "ppvA", "ppvB", "DP_lo", "DP_hi", "EO_lo", "EO_hi", "PP_lo", "PP_hi", "TPR_ratio", "FPR_ratio"]}
print(R["s24_best"], pairs, R["s24_spearman_DP_EO"], R["s24_EO_below_DP_count"])

# two real, illustrative test-set individuals for the Part-3 scenario (both repaid: y=1)
cand = pd.DataFrame({"idx": yte.index, "P": pt.values, "female": fem.loc[yte.index].values, "y": yte.values})
cand = cand.join(df[["age", "duration", "amount", "credit_history", "status", "savings"]], on="idx")
w_ = cand[(cand.y == 1) & cand.female].iloc[(cand[(cand.y == 1) & cand.female].P - 0.50).abs().argsort()[:1]]
m_ = cand[(cand.y == 1) & ~cand.female].iloc[(cand[(cand.y == 1) & ~cand.female].P - 0.62).abs().argsort()[:1]]
ex = []
for r_ in pd.concat([w_, m_]).to_dict("records"):
    r_["outcome"] = {f"{t:.2f}": bool(r_["P"] >= t) for t in ts}; ex.append(r_)
R["s24_examples"] = ex; print(ex)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
pl = sweep.dropna(subset=["DP", "EO"])
ax.plot(pl.DP, pl.EO, "-o", color="#4C72B0")
for k_, (_, r) in enumerate(pl.iterrows()): ax.annotate(f"{r.t:.2f}", (r.DP, r.EO), textcoords="offset points", xytext=((7, 6) if k_ % 2 == 0 else (-26, -13)), fontsize=8)
ax.axvline(1, color="grey", ls="--", lw=.8); ax.axhline(1, color="grey", ls="--", lw=.8); ax.plot([1], [1], "r*", ms=12, label="ideal (1,1)")
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_xlabel("Demographic Parity ratio (female/male)"); ax.set_ylabel("Equalized Odds ratio"); ax.set_title("Fairness trade-off across thresholds (labels = t)"); ax.legend()
plt.tight_layout(); plt.savefig("figs/part24.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k, c in [("DP", "#4C72B0"), ("EO", "#DD8452"), ("PP", "#55A868")]: ax.plot(sweep.t, sweep[k], "-o", label=k, color=c)
ax.axhline(1, color="grey", ls="--", lw=.8); ax.set_xlabel("threshold t"); ax.set_ylabel("ratio (ideal = 1)"); ax.legend(); ax.set_title("DP, EO, PP vs threshold")
plt.tight_layout(); plt.savefig("figs/part24_lines.png", dpi=150); plt.close()

# %% [markdown]
# ## Save results

# %%
sweep.round(4).to_csv("fairness_sweep.csv", index=False)
json.dump(R, open("results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved")
