"""Held-out benchmark targets (external control). Their 30 % clusters and UniProt accessions are removed from training.

Twelve drug targets with a crystal ligand; chosen by the authors before any model was trained. Never used for training
or model selection. The manifest builder writes their cluster and UniProt ids to data/processed/heldout_targets.json.

`lig` is the CCD component id as it appears in the deposited PDB entry, checked against the files on 2026-10-03
(`scripts/data/check_targets.py`), not from memory: five of the ids first written here were wrong and the evaluation
silently skipped those entries.
"""
TARGETS = {
    "EGFR_TM_6LUD": dict(pdb="6LUD", lig="YY3", chain="A"),
    "EGFR_TM_6LUB": dict(pdb="6LUB", lig="EUX", chain="A"),
    "EGFR_WT_1M17": dict(pdb="1M17", lig="AQ4", chain="A"),
    "ER_3ERT": dict(pdb="3ERT", lig="OHT", chain="A"),
    "BACE1_2OHU": dict(pdb="2OHU", lig="IP7", chain="A"),
    "HIV_PR_1HSG": dict(pdb="1HSG", lig="MK1", chain="A"),
    "CDK2_1DI8": dict(pdb="1DI8", lig="DTQ", chain="A"),
    "HSP90_2VCI": dict(pdb="2VCI", lig="2GJ", chain="A"),
    "BRD4_3MXF": dict(pdb="3MXF", lig="JQ1", chain="A"),
    "A2A_4EIY": dict(pdb="4EIY", lig="ZMA", chain="A"),
    "FXA_2P16": dict(pdb="2P16", lig="GG2", chain="A"),
    "PPARG_2PRG": dict(pdb="2PRG", lig="BRL", chain="A"),
}
