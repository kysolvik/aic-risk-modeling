"""Per-source information cutoffs for the operational forecast.

The forecast for target year Y is issued on ISSUE_MONTH/ISSUE_DAY of Y-1 and may only
use data published by then. Each source's cutoff below is the last month (or year)
that was already available at an audit run before the issue date
(scripts/preprocessing/audit_source_latency.py), so it is safe at issue time.

The SAME table is applied to every training year, so the model is trained on exactly
the information it will have when forecasting. Change a value only together with a
re-export of every year and a retrain.

Measured: out/latency_audit_2026-10-05.json (run 2026-10-05 for the Oct 31 issue).
"""

import calendar
import datetime

ISSUE_MONTH, ISSUE_DAY = 10, 31
AUDIT_FILE = 'out/latency_audit_2026-10-05.json'

# Monthly / sub-monthly sources: last month of Y-1 that is published by the issue date.
# None = not yet determined (using it raises).
LAST_MONTH = {
    # Weather / vegetation
    'agera5': 9,          # daily; through 2026-09-27 on 10/5
    'era5_land': 8,       # monthly aggregates; Aug on 10/5
    'mod13': 9,           # MOD13A2/MYD13A2 16-day; only composites complete by the cutoff
    'chirps_cwd': 9,      # Macedo assets, updated by hand -- confirm at every issue (stale at 2025-12 on 10/5)
    # Burned area (monthly products)
    'mcd64a1': 7,         # Jul on 10/5
    'vnp64a1': 7,         # Jul on 10/5
    # Active-fire DOY archives (one earliest-DOY image per year, rasterized from FIRMS by
    # preprocess_fire_detections.py). Through 2026-06-30 on 10/5; refreshed from FIRMS
    # (NRT for the latest months) before every issue -- confirm with the audit
    'viirs_snpp': 9,
    'viirs_noaa20': 9,
    'mod14': 9,
    'mod14_aqua': 9,
    'mod14_terra': 9,
    # Climate indices (aic_risk_modeling.preprocess.climate_indices names)
    'soi_cpc': 9,
    'nino34': 9,
    'nino4': 9,
    'amo': 9,
    'mei': 8,
    'tna': 7,
}

# Annual products: the newest usable year is Y - lag
ANNUAL_LAG = {
    'alphaearth': 2,
    'nightlights': 2,
    'hansen': 2,          # UMD/hansen/global_forest_change_2025_v1_13 holds loss through 2025
    'landscan': 3,
    # MapBiomas Amazonia col6 (1985-2023) only, while MapBiomas updates its country
    # products: 2023 is Y-4 for the 2027 forecast, so every year uses Y-4
    'mapbiomas_amazonia': 4,
}
# Last year in Amazonia col6; a forecast needing a later year must switch collections
MAPBIOMAS_AMAZONIA_LAST_YEAR = 2023


def issue_date(target_year):
    return datetime.date(target_year - 1, ISSUE_MONTH, ISSUE_DAY)


def last_month(source):
    month = LAST_MONTH[source]
    if month is None:
        raise ValueError(f'{source}: cutoff not set yet -- run audit_source_latency.py '
                         f'and fill LAST_MONTH in {__name__}')
    return month


def cutoff_date(source, target_year):
    """Last day of data from `source` usable for forecasting `target_year`."""
    y, m = target_year - 1, last_month(source)
    return datetime.date(y, m, calendar.monthrange(y, m)[1])


def cutoff_doy(source, target_year):
    """Day-of-year of cutoff_date in Y-1 (for DOY-valued burn rasters)."""
    return cutoff_date(source, target_year).timetuple().tm_yday


def month_window(source, target_year, n_months=12):
    """(year, month) pairs of the n months ending at the source's cutoff, oldest first."""
    last = (target_year - 1) * 12 + last_month(source) - 1
    return [(i // 12, i % 12 + 1) for i in range(last - n_months + 1, last + 1)]


def annual_year(source, target_year):
    """Newest usable year of an annual product for forecasting `target_year`."""
    return target_year - ANNUAL_LAG[source]
