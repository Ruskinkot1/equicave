# EquiCave: one command per stage. CPU stages need only `pip install -e .[dev]`; the network needs a GPU.
PY ?= python3
export PYTHONPATH := src:.
JOBS ?= 4
DS   ?= data/processed
TAG  ?= max

.PHONY: help setup test data data-max candidates candidates-max esm ranker ranker-max peptide-data peptide-ranker labels net net-oof ablations eval baselines all all-max clean
help:            ## list the targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/' | expand -t24

setup:           ## install the package and the training extras
	$(PY) -m pip install -e ".[dev]" && $(PY) -m pip install -e ".[train]"

test:            ## CPU tests on synthetic data (must stay green)
	$(PY) -m pytest tests -q

data:            ## manifest from RCSB + structure download (about 1500 entries, 20 min)
	$(PY) scripts/data/build_manifest.py --n 3200 --seed 0
	$(PY) scripts/data/fetch_structures.py --jobs 8

candidates:      ## native candidate table with features and labels (about 1.5 h on 4 cores)
	$(PY) scripts/train/build_native.py --jobs $(JOBS)

data-max:        ## the maximum manifest: every matching RCSB entry, <= 3 per 30 % cluster (30-60 min of metadata queries)
	$(PY) scripts/data/build_manifest.py --n 130000 --per-cluster 3 --seed 0 --out $(DS)/max
	cp $(DS)/max/manifest.csv $(DS)/manifest_max.csv

candidates-max:  ## candidates for the maximum manifest, streaming (downloads and deletes each structure; hours)
	$(PY) scripts/train/build_native.py --manifest $(DS)/manifest_max.csv --tag max --stream --keep-chunks --jobs $(JOBS)

esm:             ## ESM-2 features per candidate for the ranker (no GPU needed; ~0.5 s per structure)
	$(PY) scripts/train/build_esm_features.py --tag $(TAG) --dims 16

ranker-max:      ## the ranker on the maximum table, with the language-model features
	$(PY) scripts/train/train_ranker.py --tag max --features-extra esm --seeds 5 --ablate --model models/ranker_max.txt

ranker:          ## LightGBM LambdaRank, cluster 5-fold CV, 5 seeds, ablations
	$(PY) scripts/train/train_ranker.py --seeds 5 --ablate --model models/ranker_native.txt

peptide-data:    ## peptide-site benchmark: manifest, structures, merged cavity+groove candidates
	$(PY) scripts/data/build_peptide_manifest.py --n 4000 --seed 0 --disjoint-from $(DS)/manifest.csv
	$(PY) scripts/data/fetch_structures.py --manifest $(DS)/manifest_peptide.csv --jobs 8
	$(PY) scripts/train/build_peptide.py --jobs $(JOBS)

peptide-ranker:  ## ranker for peptide sites (with and without the peptide feature group)
	$(PY) scripts/train/train_ranker.py --tag peptide --seeds 5 --ablate --model models/ranker_peptide.txt

labels:          ## property and hotspot label statistics
	$(PY) -m training pockets-labels

net:             ## EquiCave-Net on one fold (GPU; --device cuda is the default when available)
	$(PY) -m training pockets-net --config training/configs/pockets_net.yaml --out runs/training/pockets-net --device auto

net-oof:         ## out-of-fold network features for the hybrid ranker (GPU, 5 models)
	$(PY) -m training pockets-net --config training/configs/pockets_net.yaml --out runs/training/pockets-net-oof --set mode=oof

ablations:       ## the full ablation grid, 3 seeds each (GPU, long)
	bash scripts/train/run_ablations.sh

baselines:       ## optional: fpocket and P2Rank on the same structures (needs FPOCKET / PRANK, Java)
	$(PY) scripts/baselines/run_external.py --tool fpocket --jobs $(JOBS)
	$(PY) scripts/baselines/run_external.py --tool p2rank --jobs $(JOBS)

eval:            ## every benchmark with one protocol
	$(PY) scripts/data/fetch_eval_sets.py
	for s in heldout coach420 holo4k; do $(PY) scripts/eval/evaluate.py --set $$s --ranker models/ranker_native.txt --jobs $(JOBS); done

all: data candidates ranker peptide-data peptide-ranker labels
all-max: data-max candidates-max esm ranker-max   ## the large-scale pipeline, start to finish
clean:
	rm -rf runs __pycache__ .pytest_cache
