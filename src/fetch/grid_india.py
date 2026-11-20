"""Downloader for Grid-India (NLDC) public reports.

How the site works (verified 2026-11-02):
  * grid-india.in is a JS single-page app; its file listings come from a JSON API:
        POST https://webapi.grid-india.in/api/v1/file
        {"_source": "GRDW", "_type": "DAILY_PSP_REPORT", "_fileDate": "2026-27", "_month": "00"}
    `_fileDate` is the financial year "YYYY-YY"; `_month` "00" = the whole year.
    `_type` comes from the page metadata: DAILY_PSP_REPORT, DEMAND_FORECAST (week-ahead).
  * Each record has a FilePath; files are served from https://webcdn.grid-india.in/<FilePath>.
    File names carry a random suffix (01.11.26_NLDC_PSP_491.xls), and older years live under
    a different folder (files/grdw/uploads/daily-reports/psp-reports/2023-2024/...), so URLs
    cannot be constructed - they must be looked up in the listing.
  * The old host report.grid-india.in (ReportData/Daily Report/PSP Report/...) no longer
    resolves.
  * A file is listed under the FY/month in which it was *published*; a 31-Mar report is
    therefore in the next FY's listing. Lookups check both.
  * Some listing records are broken (FilePath "files/grdwNA") or 404; these are skipped/logged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

API = "https://webapi.grid-india.in/api/v1/file"
CDN = "https://webcdn.grid-india.in/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) india-power-forecast/0.1 (research data pipeline)"

PSP_TYPE = "DAILY_PSP_REPORT"
FORECAST_TYPE = "DEMAND_FORECAST"

_DDMMYY = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})_NLDC_PSP", re.I)
_PERIOD = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s*to\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")
_MAGIC = {  # extension -> accepted leading bytes
    ".xls": (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"),  # some ".xls" may be zipped xlsx
    ".xlsx": (b"PK\x03\x04",),
    ".pdf": (b"%PDF",),
}


def financial_year(d: dt.date) -> str:
    """2026-11-01 -> '2026-27'; 2027-03-31 -> '2026-27' (Indian FY runs Apr..Mar)."""
    y = d.year if d.month >= 4 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def psp_date_from_name(name: str) -> dt.date | None:
    m = _DDMMYY.search(name)
    if not m:
        return None
    try:
        return dt.date(2000 + int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


def safe_filename(file_path: str) -> str:
    """Last path component of an API FilePath, URL-decoded, with nothing path-like left."""
    name = Path(unquote(file_path)).name
    name = re.sub(r"[^\w.\- ()&]", "_", name).strip()
    if not name or name.startswith("."):
        raise ValueError(f"unsafe file name from API: {file_path!r}")
    return name


class GridIndiaClient:
    """Small polite client: retries with backoff, a fixed delay between requests, and an
    on-disk cache of FY listings (past FYs forever, the current FY for `listing_ttl` seconds)."""

    def __init__(self, cache_dir: str | Path = "data/raw/_index", delay: float = 1.5,
                 retries: int = 4, timeout: float = 60, listing_ttl: float = 6 * 3600):
        self.cache_dir = Path(cache_dir)
        self.delay = delay
        self.timeout = timeout
        self.listing_ttl = listing_ttl
        self._last_request = 0.0
        self._listings: dict[tuple[str, str], list[dict]] = {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        retry = Retry(total=retries, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET", "POST"}), respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # ------------------------------------------------------------------ low level
    def _wait(self) -> None:
        gap = time.monotonic() - self._last_request
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self._last_request = time.monotonic()

    def list_files(self, file_type: str, fy: str, month: str = "00", use_cache: bool = True) -> list[dict]:
        """All records of `file_type` for financial year `fy` ('2026-27'); month '00' = all."""
        key = (file_type, f"{fy}_{month}")
        if use_cache and key in self._listings:
            return self._listings[key]
        cache = self.cache_dir / f"{file_type}_{fy}_{month}.json"
        current_fy = fy == financial_year(dt.date.today())
        if use_cache and cache.exists():
            age = time.time() - cache.stat().st_mtime
            if not current_fy or age < self.listing_ttl:
                self._listings[key] = json.loads(cache.read_text())
                return self._listings[key]

        self._wait()
        r = self.session.post(API, json={"_source": "GRDW", "_type": file_type,
                                         "_fileDate": fy, "_month": month}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if body.get("flagType") != 0:
            raise RuntimeError(f"Grid-India API error for {file_type} {fy}: {body.get('retMessage')}")
        records = body.get("retData") or []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records))
        self._listings[key] = records
        log.info("listed %s %s: %d records", file_type, fy, len(records))
        return records

    def download(self, file_path: str, dest_dir: str | Path, overwrite: bool = False) -> Path:
        """Download one CDN file to dest_dir (skipped if already present). Validates magic bytes,
        writes to a .part file first so an interrupted download never looks complete."""
        dest = Path(dest_dir) / safe_filename(file_path)
        if dest.exists() and dest.stat().st_size > 0 and not overwrite:
            log.debug("exists, skipping %s", dest.name)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = CDN + quote(unquote(file_path.lstrip("/")), safe="/")
        self._wait()
        with self.session.get(url, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            part = dest.with_name(dest.name + ".part")
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        magic = _MAGIC.get(dest.suffix.lower())
        head = part.read_bytes()[:8]
        if magic and not head.startswith(magic):
            part.unlink()
            raise ValueError(f"{url}: unexpected content (starts {head!r}); not a {dest.suffix} file")
        part.replace(dest)
        log.info("downloaded %s (%.0f KB)", dest.name, dest.stat().st_size / 1024)
        return dest

    # ------------------------------------------------------------------ PSP reports
    def psp_records(self, fy: str) -> dict[dt.date, list[dict]]:
        """Data date -> records (xls/xlsx/pdf) from one FY listing."""
        out: dict[dt.date, list[dict]] = {}
        for rec in self.list_files(PSP_TYPE, fy):
            fp = rec.get("FilePath") or ""
            d = psp_date_from_name(fp) or psp_date_from_name(rec.get("Title_") or "")
            if d is None or not Path(unquote(fp)).suffix:
                continue  # broken record, e.g. "files/grdwNA"
            out.setdefault(d, []).append(rec)
        return out

    def find_psp(self, date: dt.date, formats: tuple[str, ...] = (".xls", ".xlsx")) -> dict | None:
        """Best listing record for a data date, in order of `formats` preference."""
        for fy in dict.fromkeys([financial_year(date), financial_year(date + dt.timedelta(days=1))]):
            recs = self.psp_records(fy).get(date, [])
            for ext in formats:
                hits = [r for r in recs if Path(unquote(r["FilePath"])).suffix.lower() == ext]
                if hits:
                    return max(hits, key=lambda r: r.get("Id", 0))  # latest upload wins
        return None

    def download_psp(self, date: dt.date, dest_dir: str | Path = "data/raw/psp",
                     formats: tuple[str, ...] = (".xls", ".xlsx")) -> Path | None:
        """Download the PSP report for `date` (data date). None if not published/listed."""
        existing = [p for p in Path(dest_dir).glob(f"{date:%d.%m.%y}_NLDC_PSP*")
                    if p.suffix.lower() in formats and p.stat().st_size > 0]
        if existing:
            return existing[0]  # skip without touching the network
        rec = self.find_psp(date, formats)
        if rec is None:
            # the current FY listing may be stale: refresh once and retry
            fy = financial_year(date)
            if fy == financial_year(dt.date.today()):
                self._listings.pop((PSP_TYPE, f"{fy}_00"), None)
                self.list_files(PSP_TYPE, fy, use_cache=False)
                rec = self.find_psp(date, formats)
        if rec is None:
            return None
        return self.download(rec["FilePath"], dest_dir)

    # ------------------------------------------------------------------ forecasts
    def forecast_records(self, start: dt.date, end: dt.date) -> list[tuple[dt.date, dict]]:
        """Week-ahead forecast records whose title start date is within [start, end]."""
        fys = sorted({financial_year(start + dt.timedelta(days=i))
                      for i in range(0, (end - start).days + 1, 28)} | {financial_year(end)})
        out, seen = [], set()
        for fy in fys:
            for rec in self.list_files(FORECAST_TYPE, fy):
                fp = rec.get("FilePath") or ""
                m = _PERIOD.search(rec.get("Title_") or "") or _PERIOD.search(unquote(fp))
                if not m or not Path(unquote(fp)).suffix or fp in seen:
                    continue
                seen.add(fp)
                s = dt.datetime.strptime(m[1], "%d-%b-%Y").date()
                if start <= s <= end:
                    out.append((s, rec))
        return sorted(out, key=lambda t: t[0])

    def download_forecasts(self, start: dt.date, end: dt.date,
                           dest_dir: str | Path = "data/raw/forecast") -> list[Path]:
        paths = []
        for s, rec in self.forecast_records(start, end):
            try:
                paths.append(self.download(rec["FilePath"], dest_dir))
            except (requests.RequestException, ValueError) as e:
                log.warning("forecast %s failed: %s", s, e)
        return paths


def fetch_psp(date: dt.date, dest_dir: str | Path = "data/raw/psp", **kw) -> Path | None:
    """Convenience: download the PSP .xls/.xlsx for one data date."""
    return GridIndiaClient(**kw).download_psp(date, dest_dir)
