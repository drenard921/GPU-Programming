# barchart.py
import csv
from pathlib import Path


APP_FIELDS = [
    "ticker",
    "price_now",
    "cagr_3y",
    "cagr_5y",
    "dividend_yield",
    "current_holding",
    "reinvest_dividends",
]


def normalize_key(text):
    return str(text).strip().lower()


def clean_row(row):
    return {
        normalize_key(k): ("" if v is None else str(v).strip())
        for k, v in row.items()
        if k is not None
    }

def is_footer_row(row):
    """
    Detect and skip Barchart footer rows like:
    'Downloaded from Barchart.com...'
    """
    for value in row.values():
        if value and "downloaded" in str(value).lower():
            return True
    return False


def parse_float(value, default=0.0):
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    text = text.replace("$", "").replace(",", "")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_bool01(value, default=1):
    if value is None:
        return int(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "off"}:
        return 0
    return int(default)


def load_fallback_stocks(path):
    fallback = {}
    path = Path(path)
    if not path.exists():
        return fallback

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = clean_row(raw)
            ticker = row.get("ticker", "").upper()
            if not ticker:
                continue

            fallback[ticker] = {
                "ticker": ticker,
                "price_now": parse_float(row.get("price_now", ""), 0.0),
                "cagr_3y": parse_float(row.get("cagr_3y", ""), 0.0),
                "cagr_5y": parse_float(row.get("cagr_5y", ""), 0.0),
                "dividend_yield": parse_float(row.get("dividend_yield", ""), 0.0),
            }
    return fallback


def detect_format(fieldnames):
    keys = {normalize_key(x) for x in (fieldnames or [])}

    if {"ticker", "price_now", "cagr_3y", "cagr_5y", "dividend_yield"}.issubset(keys):
        return "app"

    if {"symbol", "latest"}.issubset(keys):
        return "barchart_snapshot"

    return "unknown"


def normalize_app_row(row, fallback_map):
    ticker = row.get("ticker", "").upper().strip()
    if not ticker:
        return None

    base = fallback_map.get(ticker, {})

    return {
        "ticker": ticker,
        "price_now": parse_float(row.get("price_now", ""), base.get("price_now", 0.0)),
        "cagr_3y": parse_float(row.get("cagr_3y", ""), base.get("cagr_3y", 0.0)),
        "cagr_5y": parse_float(row.get("cagr_5y", ""), base.get("cagr_5y", 0.0)),
        "dividend_yield": parse_float(
            row.get("dividend_yield", ""),
            base.get("dividend_yield", 0.0),
        ),
        "current_holding": parse_float(row.get("current_holding", ""), 0.0),
        "reinvest_dividends": parse_bool01(row.get("reinvest_dividends", ""), 1),
    }


def normalize_barchart_snapshot_row(row, fallback_map):
    ticker = row.get("symbol", "").upper().strip()
    if not ticker:
        return None

    base = fallback_map.get(ticker, {})

    return {
        "ticker": ticker,
        "price_now": parse_float(row.get("latest", ""), base.get("price_now", 0.0)),
        "cagr_3y": base.get("cagr_3y", 0.0),
        "cagr_5y": base.get("cagr_5y", 0.0),
        "dividend_yield": base.get("dividend_yield", 0.0),
        "current_holding": 0.0,
        "reinvest_dividends": 1,
    }


def dedupe_rows(rows):
    out = []
    seen = set()
    for row in rows:
        ticker = row["ticker"].upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        out.append(row)
    return out


def normalize_csv_file(input_csv, fallback_csv):
    input_csv = Path(input_csv)
    fallback_map = load_fallback_stocks(fallback_csv)

    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        source_format = detect_format(reader.fieldnames)

        if source_format == "unknown":
            raise ValueError(
                f"Unsupported CSV format for {input_csv.name}. "
                "Expected either app CSV columns or Barchart snapshot columns."
            )

        rows = []
        for raw in reader:
            row = clean_row(raw)
            if is_footer_row(row):
                continue

            if source_format == "app":
                norm = normalize_app_row(row, fallback_map)
            else:
                norm = normalize_barchart_snapshot_row(row, fallback_map)

            if norm is not None:
                rows.append(norm)

    rows = dedupe_rows(rows)

    if len(rows) < 2:
        raise ValueError(
            f"Need at least 2 valid stock rows after normalizing {input_csv.name}."
        )

    return rows, source_format


def normalize_csv_folder(folder_path, fallback_csv):
    folder_path = Path(folder_path)
    csv_paths = sorted(folder_path.glob("*.csv"))
    if not csv_paths:
        raise ValueError("No CSV files were found in the selected folder.")

    all_rows = []
    detected_formats = []

    for csv_path in csv_paths:
        rows, fmt = normalize_csv_file(csv_path, fallback_csv)
        all_rows.extend(rows)
        detected_formats.append(fmt)

    all_rows = dedupe_rows(all_rows)

    if len(all_rows) < 2:
        raise ValueError(
            f"Need at least 2 valid stock rows after normalizing folder {folder_path.name}."
        )

    return all_rows, detected_formats


def write_app_csv(rows, output_csv):
    output_csv = Path(output_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=APP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "ticker": row["ticker"],
                "price_now": f"{row['price_now']:.6f}",
                "cagr_3y": f"{row['cagr_3y']:.6f}",
                "cagr_5y": f"{row['cagr_5y']:.6f}",
                "dividend_yield": f"{row['dividend_yield']:.6f}"})