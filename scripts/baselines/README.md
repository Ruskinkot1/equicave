# Optional baselines (not part of EquiCave)

fpocket (MIT) and P2Rank (MIT) are run here only to compare against them on the same structures and splits.
Nothing in `src/` or `training/` imports or calls them. Install them yourself (see their repositories) and set
`FPOCKET` / `PRANK` to the executables; P2Rank needs a Java runtime.

    python scripts/baselines/run_external.py --tool fpocket --jobs 4
    python scripts/baselines/run_external.py --tool p2rank --jobs 4

Each run writes `data/processed/candidates_<tool>.csv` in the same layout as the native table (centre, label, dca,
n_sites, cluster30, fold, tool score and rank), so `scripts/eval/evaluate.py` and `train_ranker.py --tag` treat them alike.
