"""Download C2DB and export the exciton-binding-energy task snapshot."""

import argparse
import pickle
import urllib.request
from pathlib import Path

from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor


DEFAULT_URL = 'https://cmr.fysik.dtu.dk/_downloads/c2db.db'


def download_database(url, destination):
    """Download the ASE database unless the destination already exists."""
    destination = Path(destination)
    if destination.is_file():
        print(f'Using existing database: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {destination}')
    urllib.request.urlretrieve(url, destination)
    return destination


def extract_exciton_dataset(database, threshold=0.01):
    """Return C2DB records with a finite E_B greater than the threshold."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(database)

    adaptor = AseAtomsAdaptor()
    records = []
    for row in connect(database).select():
        binding_energy = row.get('E_B', None)
        if binding_energy is None or binding_energy <= threshold:
            continue
        atoms = row.toatoms()
        records.append({
            'uid': row.uid,
            'formula': row.formula,
            'exciton_binding_energy': float(binding_energy),
            'pymatgen_structure': adaptor.get_structure(atoms),
            'ase_atoms': atoms,
            'spacegroup': row.get('spacegroup', 'Unknown'),
            'gap_pbe': row.get('gap', None),
            'gap_gw': row.get('gap_gw', None),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('c2db.db'))
    parser.add_argument('--output', type=Path,
                        default=Path('c2db_exciton_dataset.pkl'))
    parser.add_argument('--download-url', default=DEFAULT_URL)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='retain records with E_B above this value in eV')
    args = parser.parse_args()

    if args.skip_download:
        database = args.database
    else:
        database = download_database(args.download_url, args.database)
    records = extract_exciton_dataset(database, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        pickle.dump(records, stream)
    print(f'Exported {len(records)} records to {args.output}')


if __name__ == '__main__':
    main()
