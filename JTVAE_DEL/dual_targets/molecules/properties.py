
import pandas as pd
from joblib import Parallel, delayed

from rdkit.Chem import Crippen, QED
from rdkit.Chem import Descriptors

from .conversion import mols_from_smiles
from .sascorer.sascorer import calculateScore
from .advina import calculateDockingScore
from .advina_gpx4 import calculateGPX4DockingScore
from .advina_LPA1 import calculateLPA1DockingScore


def logp(mol):
    return Crippen.MolLogP(mol) if mol else None


def mr(mol):
    return Crippen.MolMR(mol) if mol else None


def qed(mol):
    return QED.qed(mol) if mol else None


def sas(mol):
    return calculateScore(mol) if mol else None


def docking(mol):
    score = calculateDockingScore(mol) if mol else None
    return score


def docking_gpx4(mol):
    score = calculateGPX4DockingScore(mol) if mol else None
    return score


def docking_lpa1(mol):
    score = calculateLPA1DockingScore(mol) if mol else None
    return score


def tpsa(mol):
    return Descriptors.TPSA(mol) if mol else None


def add_property(dataset, name, n_jobs):
    # single
    # fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr, "score": docking, "tpsa": tpsa}[name]
    fn = {"qed": qed, "SAS": sas, "logP": logp, "mr": mr, "GPX4": docking_gpx4, "LPA1": docking_lpa1, "tpsa": tpsa}[name]
    smiles = dataset.smiles.tolist()
    mols = mols_from_smiles(smiles)
    pjob = Parallel(n_jobs=n_jobs, verbose=0)
    prop = pjob(delayed(fn)(mol) for mol in mols)
    new_data = pd.DataFrame(prop, columns=[name])
    return pd.concat([dataset, new_data], axis=1, sort=False)

