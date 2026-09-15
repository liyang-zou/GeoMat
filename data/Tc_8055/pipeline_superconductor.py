"""Build the superconductivity snapshot from Materials Cloud batch archives."""

import argparse
import os
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


RECORD_ID = '3kbt5-r3n56'
BATCH_NAMES = [f'batch-{letter}' for letter in 'abcdefghijklmnopq']


def download_file(batch_name, destination, retries=10, retry_delay=5):
    """Download one batch archive with retry and HTTP range support."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    filename = f'{batch_name}.tar.bz2'
    url = (f'https://archive.materialscloud.org/api/records/{RECORD_ID}/'
           f'files/{filename}/content')

    for attempt in range(1, retries + 1):
        existing_size = destination.stat().st_size if destination.exists() else 0
        headers = {'Range': f'bytes={existing_size}-'} if existing_size else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                if response.status_code == 416:
                    remote_size = response.headers.get('Content-Range', '').split('/')[-1]
                    if remote_size.isdigit() and existing_size == int(remote_size):
                        print(f'Using complete archive: {destination}')
                        return destination
                    destination.unlink(missing_ok=True)
                    raise IOError('Server rejected the partial-file byte range')
                response.raise_for_status()

                append = response.status_code == 206 and existing_size > 0
                mode = 'ab' if append else 'wb'
                initial = existing_size if append else 0
                content_length = int(response.headers.get('Content-Length', 0))
                total = initial + content_length if content_length else None
                with destination.open(mode) as stream, tqdm(
                        total=total, initial=initial, unit='B', unit_scale=True,
                        desc=filename) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            progress.update(len(chunk))
            if total is not None and destination.stat().st_size != total:
                raise IOError(f'Incomplete download: {destination}')
            return destination
        except (requests.RequestException, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f'Failed to download {filename}') from error
            print(f'Download attempt {attempt}/{retries} failed: {error}')
            time.sleep(retry_delay)


def parse_mcmillan_dat(text):
    """Extract McMillan and Allen-Dynes Tc values indexed by mu*."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            mu, tc_mcmillan, tc_ad = map(float, fields[:3])
        except ValueError:
            continue
        values[f'Tc_McMillan_mu_{mu}'] = tc_mcmillan
        values[f'Tc_AD_mu_{mu}'] = tc_ad
    return values


def process_tar_archive(path):
    """Retain calculations containing both a CIF and Tc_AD at mu*=0.10."""
    files_by_calculation = {}
    with tarfile.open(path, 'r:bz2') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            filename = os.path.basename(member.name)
            if filename not in {'geo_opt.cif', 'McMillan.dat'}:
                continue
            calculation_id = os.path.dirname(member.name)
            files = files_by_calculation.setdefault(
                calculation_id, {'cif': None, 'mcmillan': None})
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode('utf-8', errors='ignore')
            files['cif' if filename == 'geo_opt.cif' else 'mcmillan'] = text

    records = []
    skipped = 0
    for calculation_id, files in files_by_calculation.items():
        if not files['cif'] or not files['mcmillan']:
            skipped += 1
            continue
        labels = parse_mcmillan_dat(files['mcmillan'])
        if 'Tc_AD_mu_0.1' not in labels:
            skipped += 1
            continue
        record = {
            'compound_id': calculation_id,
            'cif_string': files['cif'],
            'target_Tc_AD': labels['Tc_AD_mu_0.1'],
        }
        record.update(labels)
        records.append(record)
    print(f'{Path(path).name}: retained {len(records)}, skipped {skipped}')
    return records


def build_snapshot(output, download_dir, keep_archives=False):
    """Download all unprocessed batches and update the output snapshot."""
    output = Path(output)
    download_dir = Path(download_dir)
    if output.is_file():
        dataframe = pd.read_pickle(output)
        records = dataframe.to_dict(orient='records')
        processed = {str(value).split('/', 1)[0]
                     for value in dataframe['compound_id']}
        print(f'Resuming from {len(records)} records in {output}')
    else:
        records, processed = [], set()

    for batch_name in BATCH_NAMES:
        if batch_name in processed:
            continue
        archive_path = download_dir / f'{batch_name}.tar.bz2'
        try:
            download_file(batch_name, archive_path)
            records.extend(process_tar_archive(archive_path))
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_pickle(output)
            print(f'Saved {len(records)} aligned records to {output}')
        except Exception:
            if records:
                output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_pickle(output)
            raise
        finally:
            if not keep_archives:
                archive_path.unlink(missing_ok=True)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('superconductors_clean_dataset.pkl'))
    parser.add_argument('--download-dir', type=Path, default=Path('temp_batches'))
    parser.add_argument('--keep-archives', action='store_true')
    args = parser.parse_args()
    dataframe = build_snapshot(args.output, args.download_dir, args.keep_archives)
    print(f'Completed snapshot with {len(dataframe)} records: {args.output}')


if __name__ == '__main__':
    main()
