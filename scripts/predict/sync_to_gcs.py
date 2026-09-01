"""Upload the contents of a local directory to a GCS prefix.

    python sync_to_gcs.py <local_dir> gs://bucket/prefix/
"""
import os
import sys

from google.cloud import storage
from google.cloud.storage import transfer_manager

MAX_WORKERS = 16


def sync(local_dir, gcs_uri):
    if not gcs_uri.startswith('gs://'):
        raise SystemExit(f'not a gs:// URI: {gcs_uri}')
    bucket_name, _, prefix = gcs_uri[len('gs://'):].partition('/')
    if not bucket_name:
        raise SystemExit(f'no bucket in URI: {gcs_uri}')

    names = sorted(f for f in os.listdir(local_dir)
                   if os.path.isfile(os.path.join(local_dir, f)))
    if not names:
        print(f'[sync] nothing to upload from {local_dir}', flush=True)
        return 0

    bucket = storage.Client().bucket(bucket_name)
    results = transfer_manager.upload_many_from_filenames(
        bucket, names,
        source_directory=local_dir,
        blob_name_prefix=(prefix.rstrip('/') + '/') if prefix else '',
        max_workers=MAX_WORKERS,
        worker_type=transfer_manager.THREAD,
    )
    failures = [(n, r) for n, r in zip(names, results) if isinstance(r, Exception)]
    print(f'[sync] uploaded {len(names) - len(failures)}/{len(names)} -> {gcs_uri}',
          flush=True)
    for name, err in failures[:10]:
        print(f'[sync] FAILED {name}: {err}', file=sys.stderr, flush=True)
    if len(failures) > 10:
        print(f'[sync] ... and {len(failures) - 10} more failures',
              file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(sync(sys.argv[1], sys.argv[2]))
