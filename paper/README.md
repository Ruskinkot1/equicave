# Manuscript

`main.md` is the draft. Rules for it:

1. Every number must be traceable to a file in `docs/results/`. A number that is not there is written `TBD`.
2. No value from an unconverged run, ever, not even "preliminary".
3. Published values of other methods are quoted only as context, marked as such, and never put in the same table as
   our measurements without a note that the protocols differ.
4. Baselines in our tables are re-run by `scripts/baselines/run_external.py` on our structures and our labels.
5. When a claim of `docs/PAPER_PLAN.md` fails, it goes into the paper as a negative result, not out of it.
