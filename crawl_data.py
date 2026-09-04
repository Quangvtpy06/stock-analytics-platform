"""
CafeF OHLCV Data Pipeline
=========================
Download, extract, normalize, and clean stock OHLCV data from CafeF
for the three Vietnamese exchanges: HOSE, HNX, UPCOM.

Usage:
    python crawl_data.py
    python crawl_data.py --exchange HOSE HNX
    python crawl_data.py --output-dir my_outputs
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_PAGE = "https://cafef.vn/du-lieu/du-lieu-download.chn"
EXCHANGES = ("HOSE", "HNX", "UPCOM")
REQUIRED_COLUMNS = ["ticker", "trading_date", "open", "high", "low", "close", "volume"]

_EXCHANGE_RE = re.compile(r"(?i)(HOSE|HSX|HNX|UPCOM)")
_EXCHANGE_MAP = {"HSX": "HOSE", "HOSE": "HOSE", "HNX": "HNX", "UPCOM": "UPCOM"}
_URL_RE = re.compile(r"(?i)(?:href|src)\s*=\s*[\"']([^\"']+)")
_DATA_FILE_RE = re.compile(r"(?i)\.(?:zip|rar|7z|csv|xls|xlsx)(?:$|\?)")


@dataclass(frozen=True)
class DownloadedDataset:
    exchange: str
    url: str
    path: Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _exchange_from_text(text: str) -> str | None:
    match = _EXCHANGE_RE.search(text)
    if not match:
        return None
    raw = match.group(1).upper()
    return _EXCHANGE_MAP.get(raw, raw)


def _is_data_file_url(url: str) -> bool:
    """Return True only if the URL points to a downloadable data file."""
    return bool(_DATA_FILE_RE.search(url.split("?")[0]))


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse CafeF dates with explicit formats to keep parsing deterministic."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    for date_format in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        candidate = pd.to_datetime(text, format=date_format, errors="coerce")
        parsed = parsed.fillna(candidate)
    return parsed


# ---------------------------------------------------------------------------
# Discovery & Download
# ---------------------------------------------------------------------------
def discover_download_links(html: str, page_url: str = SOURCE_PAGE) -> dict[str, str]:
    """Return one CafeF archive/download URL per requested exchange.

    Only accepts URLs that point to actual data files (.zip/.rar/.7z/.csv/.xls).
    This prevents picking up HTML page links (e.g. /du-lieu/tracuulichsu2/3/hose/today.chn)
    that CafeF lists as "HOSE" links but are not downloadable data.
    """
    found: dict[str, str] = {}
    upto_links: list[tuple[int, str]] = []

    # Pass 1: scan all href/src attributes for data-file URLs
    for match in _URL_RE.finditer(html):
        raw_url = match.group(1).replace("&amp;", "&")
        full_url = urljoin(page_url, raw_url)

        # ONLY accept actual data file URLs — skip HTML pages
        if not _is_data_file_url(raw_url):
            continue

        exchange = _exchange_from_text(raw_url)
        if exchange and exchange not in found:
            found[exchange] = full_url

        if re.search(r"(?i)(upto|up_to).*\.(?:zip|rar|7z)(?:$|\?)", raw_url):
            date_match = re.search(r"(20\d{6})", raw_url)
            upto_links.append(
                (int(date_match.group(1)) if date_match else 0, full_url)
            )

    # Pass 2: scan anchor tags for labelled links
    for block in re.findall(r"(?is)<a\b[^>]*>.*?</a>", html):
        label = re.sub(r"<[^>]+>", " ", block)
        href = re.search(r"(?i)href\s*=\s*[\"']([^\"']+)", block)
        if not href:
            continue
        link = urljoin(page_url, href.group(1).replace("&amp;", "&"))

        # ONLY accept actual data file URLs
        if not _is_data_file_url(link):
            continue

        exchange = _exchange_from_text(label) or _exchange_from_text(link)
        if exchange and exchange not in found:
            found[exchange] = link
        if re.search(r"(?i)upto\s*3\b", label):
            upto_links.append((0, link))

    # Fallback: if any exchange is missing, use the latest "Upto 3 sàn" archive
    # which contains data for ALL three exchanges
    if upto_links:
        latest_upto = max(upto_links)[1]
        for exchange in EXCHANGES:
            found.setdefault(exchange, latest_upto)

    return found


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (CafeF data pipeline)"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_exchanges(
    exchanges: Iterable[str], raw_dir: Path, source_page: str = SOURCE_PAGE
) -> list[DownloadedDataset]:
    """Discover and download the requested exchanges, rejecting more than 3."""
    selected = [str(exchange).upper() for exchange in exchanges]
    invalid = sorted(set(selected) - set(EXCHANGES))
    if invalid:
        raise ValueError(f"Unsupported exchange(s): {', '.join(invalid)}")
    if not 1 <= len(set(selected)) <= 3:
        raise ValueError("Select between 1 and 3 unique exchanges")

    with urlopen(
        Request(source_page, headers={"User-Agent": "Mozilla/5.0"}), timeout=60
    ) as response:
        html = response.read().decode("utf-8", errors="replace")

    links = discover_download_links(html, source_page)
    print(f"Discovered links: {json.dumps(links, indent=2)}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Group exchanges by URL to avoid downloading the same archive twice
    url_exchanges: dict[str, list[str]] = {}
    for exchange in dict.fromkeys(selected):
        if exchange not in links:
            raise RuntimeError(f"No CafeF download link found for {exchange}")
        url_exchanges.setdefault(links[exchange], []).append(exchange)

    downloaded: list[DownloadedDataset] = []
    for url, linked_exchanges in url_exchanges.items():
        suffix = Path(url.split("?")[0]).suffix or ".download"
        label = (
            linked_exchanges[0].lower()
            if len(linked_exchanges) == 1
            else "upto3"
        )
        path = raw_dir / f"{label}{suffix}"
        print(f"Downloading {url} -> {path}")
        _download(url, path)
        downloaded.append(DownloadedDataset(label.upper(), url, path))

    return downloaded


# ---------------------------------------------------------------------------
# Extraction & Normalization
# ---------------------------------------------------------------------------
def extract_archives(
    files: Iterable[DownloadedDataset], extract_dir: Path
) -> dict[str, list[Path]]:
    """Extract zip files; pass through ordinary files unchanged."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for dataset in files:
        target = extract_dir / dataset.exchange.lower()
        target.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(dataset.path):
            with zipfile.ZipFile(dataset.path) as archive:
                archive.extractall(target)
            result[dataset.exchange] = [p for p in target.rglob("*") if p.is_file()]
        else:
            result[dataset.exchange] = [dataset.path]
    return result


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "cp1258", "cp1252"):
        for separator in (None, ",", ";", "\t"):
            try:
                frame = pd.read_csv(
                    path, sep=separator, engine="python", encoding=encoding
                )
                if len(frame.columns) >= 2:
                    return frame
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                continue
    raise ValueError(f"Could not read tabular file: {path}")


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "ticker": {"ticker", "symbol", "code", "stock", "ma_ck", "ma"},
        "trading_date": {"date", "trading_date", "ngay", "ngay_giao_dich", "time"},
        "open": {"open", "opening", "gia_mo_cua"},
        "high": {"high", "highest", "gia_cao_nhat"},
        "low": {"low", "lowest", "gia_thap_nhat"},
        "close": {"close", "closing", "gia_dong_cua", "adjusted_close", "adj_close"},
        "volume": {"volume", "vol", "khoi_luong", "kl"},
    }
    by_slug = {_slug(column): column for column in frame.columns}
    rename: dict[object, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in by_slug:
                rename[by_slug[name]] = canonical
                break
    # Some CafeF exports label the date column as DTYYYYMMDD or NgayGD.
    for slug, column in by_slug.items():
        if "trading_date" not in rename.values() and (
            "date" in slug or "ngay" in slug or slug in {"dtyyyymmdd", "yyyymmdd"}
        ):
            rename[column] = "trading_date"
            break
    return frame.rename(columns=rename)


def normalize_tables(files_by_exchange: dict[str, list[Path]]) -> pd.DataFrame:
    """Read all stock tables and return the canonical OHLCV schema."""
    frames: list[pd.DataFrame] = []
    for exchange, paths in files_by_exchange.items():
        for path in paths:
            if path.suffix.lower() not in {".csv", ".txt", ".dat", ".xls", ".xlsx"}:
                continue
            frame = _canonical_columns(_read_table(path))
            if "ticker" not in frame:
                frame["ticker"] = path.stem.split("_")[0].upper()
            # A single Upto 3 san archive contains files such as CafeF.HNX.*.
            # Prefer the exchange encoded in the filename over the archive label.
            frame["exchange"] = _exchange_from_text(path.name) or exchange
            missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"{path.name} is missing required columns: {', '.join(missing)}"
                )
            frames.append(frame[REQUIRED_COLUMNS + ["exchange"]])
    if not frames:
        raise ValueError("No supported stock data files were found")
    combined = pd.concat(frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].astype("string").str.strip().str.upper()
    combined["trading_date"] = _parse_dates(combined["trading_date"])
    for column in ["open", "high", "low", "close", "volume"]:
        combined[column] = pd.to_numeric(
            combined[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    return combined


# ---------------------------------------------------------------------------
# Cleaning & Validation
# ---------------------------------------------------------------------------
def clean_and_validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop invalid rows, de-duplicate by instrument/date, and report actions."""
    data = frame.copy()
    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()
    data["trading_date"] = _parse_dates(data["trading_date"])
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(
            data[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    report = {"input_rows": len(data)}
    missing_before = int(data[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    invalid = (
        data["ticker"].isna()
        | ~data["ticker"].str.fullmatch(r"[A-Z0-9.\-]{1,20}", na=False)
        | data["trading_date"].isna()
        | data[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
        | (data[["open", "high", "low", "close", "volume"]] < 0).any(axis=1)
        | (data["high"] < data[["open", "low", "close"]].max(axis=1))
        | (data["low"] > data[["open", "high", "close"]].min(axis=1))
    )
    data = data.loc[~invalid].copy()
    duplicate_count = int(
        data.duplicated(["ticker", "exchange", "trading_date"], keep="last").sum()
    )
    data = data.drop_duplicates(
        ["ticker", "exchange", "trading_date"], keep="last"
    )
    data["trading_date"] = data["trading_date"].dt.strftime("%Y-%m-%d")
    data = data.sort_values(["exchange", "ticker", "trading_date"]).reset_index(
        drop=True
    )
    report.update(
        {
            "missing_or_invalid_rows_removed": int(
                max(missing_before, int(invalid.sum()))
            ),
            "duplicate_rows_removed": duplicate_count,
            "output_rows": len(data),
        }
    )
    return data, report


def upsert_dataset(
    incoming: pd.DataFrame, data_path: Path
) -> tuple[pd.DataFrame, int]:
    """Merge fresh data into an existing dataset, keeping the newest copy of each key."""
    if not data_path.exists():
        return incoming, 0
    previous = pd.read_csv(data_path, encoding="utf-8-sig")
    merged = pd.concat([previous, incoming], ignore_index=True)
    merged, _ = clean_and_validate(merged)
    return merged, len(previous)


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    exchanges: Iterable[str],
    output_dir: Path,
    source_page: str = SOURCE_PAGE,
) -> tuple[Path, Path]:
    """Run acquisition through storage and return dataset/report paths."""
    selected = list(
        dict.fromkeys(str(exchange).upper() for exchange in exchanges)
    )
    work_dir = output_dir / "pipeline_data"
    raw_dir = work_dir / "raw"
    extract_dir = work_dir / "extracted"
    processed_dir = work_dir / "processed"

    downloads = download_exchanges(selected, raw_dir, source_page)
    tables = extract_archives(downloads, extract_dir)
    cleaned, report = clean_and_validate(normalize_tables(tables))
    cleaned = cleaned.loc[cleaned["exchange"].isin(selected)].reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / "cafef_ohlcv.csv"
    report_path = processed_dir / "run_report.json"

    dataset, previous_rows = upsert_dataset(cleaned, data_path)
    dataset.to_csv(data_path, index=False, encoding="utf-8-sig")

    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "exchanges": selected,
            "source_page": source_page,
            "archive_urls": sorted({download.url for download in downloads}),
            "previous_dataset_rows": previous_rows,
            "stored_dataset_rows": len(dataset),
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  Exchanges: {selected}")
    print(f"  Rows stored: {len(dataset):,}")
    print(f"  Dataset: {data_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    return data_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and clean CafeF OHLCV data"
    )
    parser.add_argument(
        "--exchange", nargs="+", choices=EXCHANGES, default=list(EXCHANGES)
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--source-page", default=SOURCE_PAGE)
    args, _ = parser.parse_known_args()
    run_pipeline(args.exchange, args.output_dir, args.source_page)


if __name__ == "__main__":
    main()
