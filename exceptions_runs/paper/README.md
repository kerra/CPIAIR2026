# paper/ — what each file is for

Everything here is regenerated from the run tree by cheap commands (`grok/aggpaper.py`, `grok/lesions_report.py`,
`grok/item_order.py`, `python3 -m grok.cli.manifold_summary`, `python3 -m grok.cli.robustness`,
`python3 -m grok.cli.model_validation` (~7 min)); only `lesions/tables/` (raw GPU lesion sweeps) is impossible to
recreate. The per-run manifold grids behind `manifolds/` live in `../../exceptions_runs_pos/paper/manifolds*/` (source,
not indexed). Claims C1–C4 are defined in report. The paper's appendix tables are written from these files by
`python3 -m grok.cli.paper_tables` (→ `paper_latex/tables/*.tex`), which also copies the figures listed below into
`paper_latex/figures/`. Every file in this tree is used by the paper.

## Main text (top level)
| file | claim | what it shows |
|---|---|---|
| `regime_shares.png` (+ `.csv`) | C1 | share of runs best described by uniform / exemplar / rule family vs step/grok, per architecture |
| `hidden_progress.png` | C1 | mechanistic half-rise vs behavioural half-rise: representation forms 2–3× before behaviour |
| `hidden_progress_stats.txt` | C1 | pooled Spearman, sign tests, lead ratios, plan's time-lock criterion |
| `order_vs_overreg.png` (+ `_stats.txt`) | C2 | peak overregularization vs route order; use the RIGHT panel; logistic fit + arch test in the txt |
| `grok_order.png` | C4 | Track B′: order of behavioural grok and cognitive switch per task in joint training |
| `runs_table.csv` | all | one row per run × task: grok, switch, half-rises, t_mem/t_gen/W, overreg, regimes — the source table |
| `lesions/lesion_headline.png` | C3 | rule / exceptions / overreg vs low-energy spectral energy removed |
| `lesions/lesion_cognitive.png` (+ `_table.csv`) | C3 | best-model family shares and GCM λ vs modes kept |
| `lesions/lesion_development.png` (+ `.csv`) | C3 | the same lesions at 0.25 / 0.5 / 1.0 × grok |
| `lesions/lesion_stats.txt`, `lesions/lesion_summary.csv` | C3 | dose-response, SI, specificity, units; per-run D50 / SI |
| `manifolds/manifold_heatmaps.png` | illustration / C4 | per architecture: tasks × comparable EQ locations, seed-mean structure metric (joint matrix runs, final checkpoint) — where each algorithm lives |
| `manifolds/manifold_table.md` (+ `.csv`) | C4 | best location per arch × task, metric mean ± sd, dominant Fourier mode per seed, circle corr |
| `manifolds/pos_ablation.md` (+ `.csv`) | C4 | layer-0 structure without vs with learned positional embeddings (matched_2M_pos), grok order of both trees |

## Supplementary (`supp/`)
| file | note |
|---|---|
| `cognitive_regimes.png` | per-probe ΔBIC trajectories (exemplar vs uniform, rule family vs exemplar, rulex vs rule) behind `regime_shares.png` |
| `ucurve_rel_add_rho0.02_wd0.1.png` | the representative U-curve (test acc / acc_exc / overreg vs step/grok) |
| `grok_delay.png` | cost of exceptions: grok step vs rho per architecture and WD |
| `grok_order_clean.csv` | Track B′ numbers behind `grok_order.png` |
| `item_order_add_rho0_wd0.1.png`, `item_order_stats_add_rho0_wd0.1.txt` | per-item learning order across architectures (C4 boundary; negative) |
| `robustness_stats.txt`, `robustness_sensitivity.csv` | C2 robustness: threshold sensitivity, cluster bootstrap, leave-one-arch-out, between/within cells, random-intercept model (needs statsmodels); C1 threshold sensitivity and lead CIs |
| `model_validation.csv`, `model_validation_stats.txt` | cognitive-model comparison validated on a checkpoint subset: held-out likelihood, permutation null, ALCOVE-lite / ATRIUM-lite / prototype |

Regenerate: `python3 grok/aggpaper.py --base-out ./exceptions_runs`, `python3 grok/lesions_report.py --base-out ./exceptions_runs`,
`python3 grok/item_order.py --base-out ./exceptions_runs --task add`, `python3 -m grok.cli.manifold_summary`,
`python3 -m grok.cli.robustness --base-out ./exceptions_runs`, `python3 -m grok.cli.model_validation --base-out ./exceptions_runs`,
then `python3 -m grok.cli.paper_tables`. div / max lesion variants: `lesions_report.py --task div|max`.
