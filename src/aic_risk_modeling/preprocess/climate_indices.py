"""Download monthly climate indices (AMO, SOI, ONI, MEI, TNA) from NOAA."""

import io
import urllib.request

import pandas as pd

# NOAA headers declare the missing value inconsistently, so detect it by magnitude.
NODATA_ABS = 60.0

MAX_FILL_GAP = 2

# CPC sources, more quickly update
CPC_SOI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/soi'
CPC_SSTOI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices'
# Anomaly columns of sstoi.indices (monthly OISST; the file repeats the 'ANOM' header)
SSTOI_COLUMNS = ['YR', 'MON', 'nino12', 'nino12_anom', 'nino3', 'nino3_anom',
                 'nino4', 'nino4_anom', 'nino34', 'nino34_anom']
SSTOI_INDICES = {'nino34': 'nino34_anom', 'nino4': 'nino4_anom'}


def _fetch_text(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode('ascii', errors='replace')


def _parse_cpc_soi(text):
    """Standardized SOI from CPC's fixed-width year x month table.

    The file holds two tables (anomaly, then standardized); the one after the
    'STANDARDIZED' header is used. Fields are 6 characters wide and can run
    together ('-999.9-999.9'), so they are sliced, not split.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if 'STANDARDIZED' in l)
    except StopIteration:
        raise ValueError('CPC SOI: no STANDARDIZED table found') from None
    rows = []
    for line in lines[start:]:
        head = line[:4]
        if not head.isdigit():
            if rows:
                break  # end of the table
            continue
        year = int(head)
        for m in range(12):
            field = line[4 + 6 * m: 10 + 6 * m].strip()
            if field:
                rows.append((pd.Timestamp(year, m + 1, 1), float(field)))
    return pd.DataFrame(rows, columns=['Date', 'metric'])


def _parse_cpc_sstoi(text, index_name):
    """One Nino-region anomaly column from CPC's monthly OISST sstoi.indices."""
    df = pd.read_csv(io.StringIO(text), sep=r'\s+', skiprows=1, header=None,
                     names=SSTOI_COLUMNS)
    dates = pd.to_datetime(dict(year=df['YR'], month=df['MON'], day=1))
    return pd.DataFrame({'Date': dates, 'metric': df[SSTOI_INDICES[index_name]].astype(float)})


def download_clim_indices(
        index_name: str,
        year_start: int,
        year_end: int,
        last_month: int = 12
    ) -> pd.DataFrame:
    """Monthly index values for Jan year_start .. Dec year_end, short gaps interpolated.

    Raises:
        ValueError: if the index name is unknown, if a requested month is absent
            from the source entirely, or if missing values remain after
            interpolation.

    Args:
    index_name: one of 'amo', 'soi', 'oni', 'mei', 'tna' (PSL/NCEI), or the
        CPC sources 'soi_cpc', 'nino34', 'nino4' (see CPC_SOI_URL, CPC_SSTOI_URL).
    year_start: First year to download (but samples are monthly)
    year_end: Last year for download (but samples are monthly)
    last_month: Last month of year_end to return (default: the full year)
    """
    clim_registry = {
        'amo':'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat',
        'soi':'https://psl.noaa.gov/data/timeseries/month/data/soi.long.csv',
        'oni':'https://psl.noaa.gov/data/correlation/oni.csv',
        'mei': 'https://psl.noaa.gov/data/correlation/meiv2.csv',
        'tna': 'https://psl.noaa.gov/data/correlation/tna.csv',
        'soi_cpc': CPC_SOI_URL,
        'nino34': CPC_SSTOI_URL,
        'nino4': CPC_SSTOI_URL,
    }

    try:
        download_url = clim_registry[index_name]
    except KeyError:
        raise ValueError(f'{index_name} not found. Current options are {list(clim_registry.keys())}')

    if index_name == 'soi_cpc':
        df = _parse_cpc_soi(_fetch_text(download_url))
    elif index_name in SSTOI_INDICES:
        df = _parse_cpc_sstoi(_fetch_text(download_url), index_name)
    elif index_name == 'amo':
        df = pd.read_csv(download_url, skiprows=1, sep=r'\s+')
        df['Date'] = df['Year'].astype(str) + '-' + df['month'].astype(str) + '-01'
        df = df.drop(columns=['Year','month'])[['Date','SSTA']]
    else:
        df = pd.read_csv(download_url)

    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Date', 'metric']

    df = df.set_index('Date')
    df = df[~df.index.duplicated(keep='last')].sort_index()

    wanted = pd.date_range(f'{year_start}-01-01', f'{year_end}-{last_month:02d}-01', freq='MS')
    absent = wanted.difference(df.index)
    if len(absent):
        raise ValueError(
            f'{index_name}: {len(absent)} month(s) of [{year_start}, {year_end}] '
            f'are not in the source, first {absent[0].date()}. The record '
            f'ends at {df.index.max().date()}.')
    df = df.loc[wanted]

    df['metric'] = df['metric'].where(df['metric'].abs() <= NODATA_ABS)
    missing = df.index[df['metric'].isna()]
    if len(missing):
        df['metric'] = df['metric'].interpolate(method='time', limit=MAX_FILL_GAP,
                                                limit_area='inside')
        still = df.index[df['metric'].isna()]
        if len(still):
            raise ValueError(
                f'{index_name}: {len(still)} month(s) of [{year_start}, {year_end}] '
                f'are missing and could not be interpolated '
                f'({still[0].date()} .. {still[-1].date()}); the last usable '
                f'observation is {df["metric"].last_valid_index().date()}. Either '
                f'the run exceeds MAX_FILL_GAP={MAX_FILL_GAP} months or it is at '
                f'the edge of the window, where filling it would be extrapolation.')

    n_expected = 12 * (year_end - year_start) + last_month
    if len(df) != n_expected:
        raise ValueError(f'{index_name}: got {len(df)} months, expected {n_expected}. '
                         f'Callers index this positionally; a length change '
                         f'silently shifts the calendar.')
    return df
