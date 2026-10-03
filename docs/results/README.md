# Results

Every file here is written by a script and carries its protocol in the header. Nothing is edited by hand.
A number that is absent is absent on purpose: the run did not happen, or did not converge.

| file | produced by | content |
|---|---|---|
| `native_candidates_build.log` | `scripts/train/build_native.py` | candidate ceiling, mean candidates, timing over the whole manifest |
| `ranker_native.md` / `.json` | `scripts/train/train_ranker.py` | LambdaRank cross-validation by cluster, seeds, feature-group ablations, paired bootstrap |
| `ranker_peptide.md` / `.json` | the same, `--tag peptide` | the peptide-site benchmark, with and without the peptide feature group |
| `methods_comparison.md` / `.json` | `scripts/eval/compare_methods.py` | native, fpocket and P2Rank candidates on the same structures, each with and without the learned ranker |
| `eval_<set>.md` / `.json` | `scripts/eval/evaluate.py` | one benchmark under the shared protocol; `train-similar` structures are a separate row |
| `ablations.md` / `.json` | `scripts/train/collect_ablations.py` | the network ablation grid, mean ± seed sd, paired against `full` |

Conventions in every table: success is DCA ≤ 4 Å (DCC reported where available); N is the structure's own number of
ligand sites; intervals are 95 % cluster bootstraps; gains carry a paired cluster bootstrap; "not run" means not run.
