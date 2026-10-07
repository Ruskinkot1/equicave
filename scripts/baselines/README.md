# Optional baselines (not part of EquiCave)

fpocket (MIT) and P2Rank (MIT) are run here only to compare against them on the same structures and splits.
Nothing in `src/` or `training/` imports or calls them. Install them yourself (see their repositories) and set
`FPOCKET` / `PRANK` to the executables; P2Rank needs a Java runtime.

    python scripts/baselines/run_external.py --tool fpocket --jobs 4
    python scripts/baselines/run_external.py --tool p2rank --jobs 4

Each run writes `data/processed/candidates_<tool>.csv` in the same layout as the native table (centre, label, dca,
n_sites, cluster30, fold, tool score and rank), so `scripts/eval/evaluate.py` and `train_ranker.py --tag` treat them alike.

## Modern deep-learning predictors

Three of the seven deep-learning pocket predictors surveyed in `docs/LITERATURE_SURVEY.md` can be run from
published weights, and all three are driven end to end here. Each driver runs the method's **own** code and writes
only the conversion into our candidate table, so the labels, ligand rule and non-redundancy are the ones every
other method in `scripts/eval/compare_on_set.py` gets. Each writes resumable chunks and takes `--no-resume`.
Every deviation from the authors' environment is listed in the driver's module docstring.

### GrASP (MIT, weights in its repository)

    git clone https://github.com/tiwarylab/GrASP
    GRASP_REPO=/path/to/GrASP python scripts/baselines/run_grasp.py --set coach420

### DeepPocket (MIT; weights mirrored on Zenodo)

Needs fpocket, libmolgrid and **numpy 1.x**: the `molgrid` wheel's boost::python numpy bridge is compiled against
numpy 1, and under numpy 2 every `.tonumpy()` raises. A venv with `--system-site-packages` over numpy 1.26 costs
100 MB instead of a second torch.

    git clone https://github.com/devalab/DeepPocket
    git clone https://github.com/Discngine/fpocket && cd fpocket && make     # serial: -j breaks its vendored qhull
    python -m venv --system-site-packages venv_dp && venv_dp/bin/pip install "numpy==1.26.4" molgrid biopython scikit-image
    # weights: the authors' SharePoint link answers 403 from here; Zenodo record 13833813 (CC-BY-4.0) mirrors them
    curl -o first_model.pth.tar "https://zenodo.org/api/records/13833813/files/first_model_fold1_best_test_auc_85001.pth.tar/content"
    DEEPPOCKET_REPO=/path/to/DeepPocket FPOCKET=/path/to/fpocket/bin/fpocket \
      venv_dp/bin/python scripts/baselines/run_deeppocket.py --set coach420 --checkpoint first_model.pth.tar

### DeepSurf (AGPL-3.0 code; no licence stated for the weights)

Needs its own interpreter: `tensorflow.contrib` is TF 1.x only and its `import pybel` is openbabel 2's layout, so
neither exists for Python 3.11. It also needs DMS and about **4 GB of peak memory** per large structure, because
their `simplify_dms` runs KMeans with a few thousand clusters over tens of thousands of surface points -- run it
when nothing else large is resident or the kernel will kill it.

    micromamba create -p env_ds -c conda-forge python=3.7 "openbabel=2.4.1"
    env_ds/bin/pip install "numpy==1.18.5" "tensorflow==1.15.5" "scipy==1.4.1" "scikit-learn==0.22.2"
    gdown 1nIBoD3_5nuMqgRGx4G1OHZwLsiUjb7JG && 7z x models.7z          # their published models
    wget www.cgl.ucsf.edu/Overview/ftp/dms.zip && unzip dms.zip && cd dms && make install
    DEEPSURF_REPO=/path/to/DeepSurf DEEPSURF_PYTHON=$PWD/env_ds/bin/python DEEPSURF_MODELS=$PWD/models \
      python scripts/baselines/run_deepsurf.py --set coach420

`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set by the driver: with the C++ protobuf implementation,
importing TensorFlow after openbabel segfaults in this environment.

**Their own cap on receptor size.** `readSurfPoints` returns `None` above 100000 DMS surface points and
`simplify_dms` unpacks it unguarded, so a large receptor arrives as `TypeError: cannot unpack non-iterable
NoneType object`. On COACH420 it fires on the three structures above 20000 atoms (2WVA, 1E5Q, 1Q51) and no others.
Nothing in their arguments changes it, so those structures are excluded with the reason recorded -- they are a
property of the published method, not of this machine, and scoring them as refusals would be as wrong as hiding
them.

`failed.txt` therefore records `id<TAB>ours|theirs<TAB>reason`, and `--retry-ours` lets back in only the failures
of ours. Two shards on a 15 GB machine do provoke OOM kills on ordinary structures (9500-16100 atoms here), which
a serial pass afterwards recovers:

    python scripts/baselines/run_deepsurf.py --set coach420 --shard 0/2 &
    python scripts/baselines/run_deepsurf.py --set coach420 --shard 1/2 &
    wait && python scripts/baselines/run_deepsurf.py --set coach420 --retry-ours

### The four that cannot be compared

VN-EGNN, EquiPocket and GDEGAN publish no weights, so a comparison means retraining them on their data.
PointSite commits its weights and its bundled SparseConvNet builds once `setup.py`'s `-std=c++11` becomes
`-std=c++20` for current torch headers, but the compile exhausts memory here; it also segments binding *atoms*
rather than ranking sites, so any top-N for it would depend on a clustering we would have to choose on its behalf.


## CryptoBench needs the apo/holo pairing, so the baselines are not run on it

`eval_set_to_manifest.py` writes each CryptoBench row's relevant ligand from the entry's **holo** partner, because
that is where the ligand is: the apo structure is the one without it. `run_competitors.py` then looks for that
ligand code inside the apo file, finds nothing, and drops the structure. A first run produced 3 rows out of 228 --
the three where the holo ligand's code happens to occur in the apo file too, which is coincidence rather than
signal. That table was deleted rather than kept.

Doing it properly means superposing the holo structure onto the apo one and transferring the ligand, which is what
`evaluate.py` already does through the `holo_pdb_id` note and `equicave.superpose`. The baseline path does not,
so no competitor number exists for CryptoBench and none should be quoted. Our own CryptoBench numbers are
unaffected: they come through `evaluate.py`, which performs the transfer.
