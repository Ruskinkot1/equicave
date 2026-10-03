# Wrappers for competing predictors

`run_competitors.py` calls each learned competitor as

    <wrapper> <protein.pdb> <output.json>

and the wrapper must write, best first:

```json
[{"center": [12.3, 4.5, -7.8], "score": 0.91, "rank": 1},
 {"center": [30.1, 2.2, 11.0], "score": 0.42, "rank": 2}]
```

Centres must be in the coordinate frame of the input PDB (no re-centring, no re-numbering). Anything else the method
needs — its own conda environment, CUDA version, Java, weights — stays inside the wrapper, so our environment never
takes a dependency on it. Point the runner at a wrapper with the matching variable:

    DEEPPOCKET=scripts/baselines/wrappers/deeppocket.sh
    DEEPSURF=scripts/baselines/wrappers/deepsurf.sh
    GRASP=scripts/baselines/wrappers/grasp.sh
    VNEGNN=scripts/baselines/wrappers/vnegnn.sh

Before using a method, check its licence and the licence of its weights at the primary source and record the result in
`docs/PROVENANCE.md`. A method whose weights have no licence is not run.
