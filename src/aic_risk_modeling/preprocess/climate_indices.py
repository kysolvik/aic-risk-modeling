"""Helpers to download non-spatial climate indices"""

import pandas as pd

# NOAA's headers declare their missing value inconsistently.
# So detecting missingness by magnitude rather than
# trusting either the header or a single hard-coded constant.
NODATA_ABS = 60.0

# Set max no data gap to fill with linear interpolation
MAX_FILL_GAP = 2


def download_clim_indices(
        index_name: str,
        year_start: int,
        year_end: int
    ) -> pd.DataFrame:
    """Download non-spatial climate indices from NOAA.

    Raises:
        ValueError: if the index name is unknown, if a requested month is absent
            from the source entirely, or if missing values remain after
            interpolation.

    Args:
    index_name: one of 'amo', 'soi', 'oni', 'mei', 'tna'.
    year_start: First year to download (but samples are monthly)
    year_end: Last year for download (but samples are monthly)
    """
    clim_registry = {
        'amo':'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat',
        'soi':'https://psl.noaa.gov/data/timeseries/month/data/soi.long.csv',
        'oni':'https://psl.noaa.gov/data/correlation/oni.csv',
        'mei': 'https://psl.noaa.gov/data/correlation/meiv2.csv',
        'tna': 'https://psl.noaa.gov/data/correlation/tna.csv'
    }

    try:
        download_url = clim_registry[index_name]
    except KeyError:
        raise ValueError(f'{index_name} not found. Current options are {list(clim_registry.keys())}')

    if index_name == 'amo':
        df = pd.read_csv(download_url, skiprows=1, sep='\s+')
        df['Date'] = df['Year'].astype(str) + '-' + df['month'].astype(str) + '-01'
        df = df.drop(columns=['Year','month'])[['Date','SSTA']]
    else:
        df = pd.read_csv(download_url)

    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Date', 'metric']

    df = df.set_index('Date')
    df = df[~df.index.duplicated(keep='last')].sort_index()

    wanted = pd.date_range(f'{year_start}-01-01', f'{year_end}-12-01', freq='MS')
    absent = wanted.difference(df.index)
    if len(absent):
        raise ValueError(
            f'{index_name}: {len(absent)} month(s) of [{year_start}, {year_end}] '
            f'are not in the source, first {absent[0].date()}. The record '
            f'ends at {df.index.max().date()}.')
    df = df.loc[wanted]

    # Filter out nodata
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

    n_expected = 12 * (year_end - year_start + 1)
    if len(df) != n_expected:
        raise ValueError(f'{index_name}: got {len(df)} months, expected {n_expected}. '
                         f'Callers index this positionally; a length change '
                         f'silently shifts the calendar.')
    return df
