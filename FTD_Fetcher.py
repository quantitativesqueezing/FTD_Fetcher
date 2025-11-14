import argparse
import datetime as dt
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / ".env"
EXPORT_DIR = REPO_ROOT / "data"
DEFAULT_MAX_DISCORD_ROWS = 25
DISCORD_FIELD_CHAR_LIMIT = 1024


def resolve_log_file(env_data: Dict[str, Union[List[str], str]]) -> Optional[Path]:
    """Determine the log file path configured via .env."""
    raw_value = env_data.get("LOG_FILE_PATH")
    if not raw_value:
        return None
    normalized = normalize_env_path(str(raw_value))
    if not normalized:
        return None
    path = Path(normalized).expanduser()
    return path


def _load_post_log(log_path: Path) -> Dict[str, Union[str, list]]:
    """Load the existing log data if available."""
    if not log_path.exists():
        return {}
    try:
        with log_path.open() as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_post_log(log_path: Path, record: Dict[str, Union[str, list]]) -> None:
    """Persist the log data, ensuring its parent directory exists."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fp:
        json.dump(record, fp, indent=2)


def should_skip_post_for_date(date_str: str, log_path: Optional[Path]) -> bool:
    """Return True if we already posted for the given settlement date."""
    if log_path is None:
        return False
    record = _load_post_log(log_path)
    return record.get("latest_settlement_date") == date_str


def update_post_log(log_path: Optional[Path], date_str: str) -> None:
    """Append metadata for the settlement date we just posted."""
    if log_path is None:
        return
    record = _load_post_log(log_path)
    history = record.get("history", [])
    history.append(
        {
            "settlement_date": date_str,
            "posted_at": dt.datetime.utcnow().isoformat(),
        }
    )
    history = history[-30:]
    payload = {
        "latest_settlement_date": date_str,
        "history": history,
    }
    _write_post_log(log_path, payload)


def strip_quotes(value: str) -> str:
    """Remove matching surrounding quotes from a string."""
    trimmed = value.strip()
    if len(trimmed) >= 2 and ((trimmed[0] == trimmed[-1] == '"') or (trimmed[0] == trimmed[-1] == "'")):
        return trimmed[1:-1]
    return trimmed


def normalize_env_path(value: str) -> str:
    """Expand missing separators and treat escaped spaces like literal spaces."""
    trimmed = strip_quotes(value)
    if not trimmed:
        return ""
    normalized = trimmed.replace("\\ ", " ")
    return os.path.expandvars(normalized)


def load_env_file(env_path: Path) -> Dict[str, Union[List[str], str]]:
    """Read the .env file with support for list-like variables."""
    env_data: Dict[str, Union[List[str], str]] = {}
    if not env_path.exists():
        return env_data

    current_key: Optional[str] = None
    with env_path.open() as fp:
        for raw_line in fp:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if current_key:
                if stripped.startswith("]"):
                    current_key = None
                    continue
                env_data[current_key].append(stripped)
                continue
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("["):
                env_data[key] = []
                current_key = key
                inline = value[1:].strip()
                if inline:
                    if inline.endswith("]"):
                        inline = inline[:-1].strip()
                        if inline:
                            env_data[key].append(inline)
                        current_key = None
                    else:
                        env_data[key].append(inline)
                continue
            env_data[key] = value
    return env_data


def parse_webhook_entries(raw_lines: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Convert raw list entries like '\"url\" => \"thread\"' into tuples."""
    entries: List[Tuple[str, Optional[str]]] = []
    for raw in raw_lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        url = strip_quotes(left.strip())
        thread_value = strip_quotes(right.strip())
        if not url:
            continue
        entries.append((url, thread_value or None))
    return entries


def append_thread_id_to_url(webhook_url: str, thread_id: Optional[str]) -> str:
    """Append or update ?thread_id on webhook URL for forum posts."""
    if not thread_id:
        return webhook_url
    parsed = urlparse(webhook_url)
    query_items = dict(parse_qsl(parsed.query))
    query_items["thread_id"] = thread_id
    new_query = urlencode(query_items)
    return urlunparse(parsed._replace(query=new_query))


def replace_tokens(text: str, context: Dict[str, str]) -> str:
    """Substitute simple {token} placeholders with provided context."""
    result = text
    for token, value in context.items():
        if value is None:
            continue
        placeholder = f"{{{token}}}"
        result = result.replace(placeholder, value)
    return result


def render_template(template: object, context: Dict[str, str]) -> object:
    """Recursively render strings inside JSON-like template structures."""
    if isinstance(template, dict):
        return {key: render_template(val, context) for key, val in template.items()}
    if isinstance(template, list):
        return [render_template(item, context) for item in template]
    if isinstance(template, str):
        return replace_tokens(template, context)
    return template


def resolve_max_discord_rows(env_data: Dict[str, Union[List[str], str]]) -> int:
    """Read MAX_DISCORD_ROWS from the parsed .env data, falling back to a default."""
    raw_value = env_data.get("MAX_DISCORD_ROWS")
    if raw_value is None:
        return DEFAULT_MAX_DISCORD_ROWS
    raw_str = strip_quotes(str(raw_value))
    try:
        parsed = int(raw_str)
        return parsed if parsed > 0 else DEFAULT_MAX_DISCORD_ROWS
    except ValueError:
        return DEFAULT_MAX_DISCORD_ROWS


def build_discord_context(
    dataframe: pd.DataFrame, settlement_date: str, landing_url: Optional[str], max_rows: Optional[int] = None
) -> Dict[str, str]:
    """Build the dictionary of placeholder values for the Discord payload."""
    rows = max_rows if max_rows is not None and max_rows > 0 else DEFAULT_MAX_DISCORD_ROWS
    context: Dict[str, str] = {
        "latest_settlement_date_formatted": settlement_date,
        "latest_ftd_url": landing_url or "",
    }

    subset = dataframe.head(rows)

    def join_column(column: str, prefix: str = "", suffix: str = "") -> str:
        values = []
        for value in subset[column]:
            if pd.isna(value):
                values.append("")
            else:
                raw = str(value)
                formatted = f"{prefix}{raw}{suffix}" if raw else ""
                values.append(formatted)
        joined = "\n".join(values)
        if len(dataframe) > rows:
            joined = f"{joined}\n..."
        if len(joined) > DISCORD_FIELD_CHAR_LIMIT:
            joined = f"{joined[:DISCORD_FIELD_CHAR_LIMIT - 3]}..."
        return joined or "-"

    context["ftd_data.symbol"] = join_column("Symbol", prefix="$")
    context["ftd_data.quantity"] = join_column("QuantityFails", suffix=" shares")
    context["ftd_data.ftd_value"] = join_column("FTD_Value")
    return context


def load_template(template_path_value: str) -> dict:
    """Load the JSON template referenced in the .env."""
    expanded = normalize_env_path(template_path_value)
    if not expanded:
        raise FileNotFoundError("Discord template path is empty")
    template_path = Path(expanded)
    if not template_path.is_absolute():
        template_path = REPO_ROOT / template_path
    template_path = template_path.expanduser()
    if not template_path.exists():
        raise FileNotFoundError(f"Discord template not found at {template_path}")
    with template_path.open() as fp:
        return json.load(fp)


def post_to_discord(
    webhook_url: str, payload: dict, thread_id: Optional[str], attachment_path: Optional[Path]
) -> None:
    """Send the rendered payload to a Discord webhook, optionally attaching a file."""
    target_url = append_thread_id_to_url(webhook_url, thread_id)
    files = {"payload_json": (None, json.dumps(payload))}
    file_handle = None
    if attachment_path and attachment_path.exists():
        file_handle = attachment_path.open("rb")
        files["file"] = (
            attachment_path.name,
            file_handle,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    try:
        response = requests.post(target_url, files=files, timeout=30)
        response.raise_for_status()
    finally:
        if file_handle:
            file_handle.close()


def send_discord_notifications(
    dataframe: pd.DataFrame,
    settlement_date: str,
    landing_url: Optional[str],
    attachment_path: Optional[Path],
    use_test: bool = False,
    env_data: Optional[Dict[str, Union[List[str], str]]] = None,
) -> bool:
    """Render the template and post to every webhook configured in .env."""
    if env_data is None:
        env_data = load_env_file(ENV_FILE)
    template_raw = env_data.get("DISCORD_WEBHOOK_TEMPLATE")
    if not template_raw:
        print("⚠️ Discord template is not configured (.env DISCORD_WEBHOOK_TEMPLATE). Skipping.")
        return False

    try:
        template = load_template(template_raw)
    except FileNotFoundError as exc:
        print(f"⚠️ {exc}")
        return

    row_limit = resolve_max_discord_rows(env_data)
    context = build_discord_context(dataframe, settlement_date, landing_url, max_rows=row_limit)
    payload = render_template(template, context)

    webhook_key = "DISCORD_WEBHOOK_TEST_URL" if use_test else "DISCORD_WEBHOOK_URL"
    raw_targets = env_data.get(webhook_key)
    if not raw_targets:
        print(f"⚠️ No Discord webhook targets defined for {webhook_key}. Skipping.")
        return False

    if isinstance(raw_targets, list):
        target_lines = raw_targets
    else:
        target_lines = [raw_targets]

    targets = parse_webhook_entries(target_lines)
    if not targets:
        print(f"⚠️ Discord webhook list for {webhook_key} is empty after parsing. Skipping.")
        return False

    attachment_to_send: Optional[Path] = None
    if attachment_path:
        if attachment_path.exists():
            attachment_to_send = attachment_path
        else:
            print(f"⚠️ Attachment {attachment_path} not found; sending webhook without file.")

    posted_any = False
    for webhook_url, thread_id in targets:
        try:
            post_to_discord(webhook_url, payload, thread_id, attachment_to_send)
            posted_any = True
            print(f"✅ Posted Discord payload to {webhook_url}")
        except requests.RequestException as exc:
            print(f"⚠️ Failed to post to {webhook_url}: {exc}")

    return posted_any


def build_url(year, month, half):
    if half == "a":  # first half
        start = f"{year}{month:02d}a"
    else:  # second half
        start = f"{year}{month:02d}b"
    return f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{start}.zip"

def get_latest_url():
    today = dt.date.today()
    y, m, d = today.year, today.month, today.day

    if d <= 15:
        # Use previous month’s second half
        if m == 1:
            y -= 1
            m = 12
        else:
            m -= 1
        return build_url(y, m, "a")
    else:
        # Current month’s first half
        return build_url(y, m, "b")

def fetch_top_ftds(num_results=200, export=True):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        ),
        "Referer": "https://www.sec.gov/",
    }

    today = dt.date.today()
    month = today.strftime("%Y%m")
    last_month_date = today.replace(day=1) - dt.timedelta(days=1)
    last_month = last_month_date.strftime("%Y%m")

    candidates = [
        f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{month}a.zip",
        f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{month}b.zip",
        f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{last_month}b.zip",
        f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{last_month}a.zip",
    ]

    resp = None
    download_url: Optional[str] = None
    for potential in candidates:
        print(f"Trying ➤ {potential}")
        try:
            resp = requests.get(potential, headers=headers, timeout=30)
            resp.raise_for_status()
            print(f"✅ Downloading from: {potential}")
            download_url = potential
            break
        except requests.HTTPError as err:
            print(f"  ❌ Request failed (status {err.response.status_code}), trying next...")
            resp = None
    else:
        raise RuntimeError(f"No FTD files accessible within: {candidates}")

    # Save the ZIP file locally
    zip_filename = "latest_ftd.zip"
    with open(zip_filename, "wb") as fzip:
        fzip.write(resp.content)

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    fname = z.namelist()[0]
    # Read the file, decode as latin1 to handle possible encoding
    with z.open(fname) as f:
        raw_data = f.read().decode("latin1")
    from io import StringIO
    df = pd.read_csv(StringIO(raw_data), sep="|", header=0)
    expected_cols = ["SettlementDate","CUSIP","Symbol","QuantityFails","Company","Price"]
    if len(df.columns) >= len(expected_cols):
        df.columns = expected_cols + list(df.columns[len(expected_cols):])
    else:
        df.columns = expected_cols[:len(df.columns)]

    df.dropna(subset=["QuantityFails","Price"], inplace=True)
    df["QuantityFails"] = pd.to_numeric(df["QuantityFails"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["FTD_Value"] = df["QuantityFails"] * df["Price"]

    latest_date = df["SettlementDate"].max()
    latest_date_str = str(latest_date)
    latest_date_obj = dt.datetime.strptime(latest_date_str, "%Y%m%d")
    latest_settlement_date_formatted = latest_date_obj.strftime("%m/%d/%Y")
    df_recent = df[df["SettlementDate"] == latest_date]

    WHITELIST = {"SPY", "QQQ", "USO", "LQD"}
    fundish_substrings = [
        # Common ETF/ETN/fund families
        "etf", "etn", "spdr", "ishares", "vanguard", "invesco", "proshares",
        "global x", "direxion", "wisdomtree", "xtrackers", "vaneck", "pacer",
        "ark", "first trust", "schwab", "select sector", "index",
        # Generic fund terms
        "fund", "trust unit", "unit investment trust", "closed end", "open end",
        # Wealth/private equity terms
        "private equity", "wealth fund", "family office", "sovereign wealth",
        # Bond/fixed income keywords (to exclude bond funds/ETFs/notes)
        "bond", "treasury", "muni", "municipal", "note", "preferred", "fixed income",
        # Other structures often not single operating companies
        "depositary receipt", "adr", "ads", "unit trust", "capital trust", "income trust",
        "reit", "real estate", "partnership", " lp ", " llp ", " mlp ", " etp "
    ]

    def is_single_stock(symbol: str, company: str) -> bool:
        # Normalize symbol safely (guard against NaN/float/None)
        if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
            sym = ""
        else:
            sym = str(symbol)
        sym = sym.upper().strip()
        if sym in WHITELIST:
            return True

        # Normalize company safely
        if company is None or (isinstance(company, float) and pd.isna(company)):
            name = ""
        else:
            name = str(company)
        name = name.lower()
        name_spaced = f" {name} "
        return not any(term in name_spaced for term in fundish_substrings)

    df_recent = df_recent.copy()
    df_recent["Symbol"] = df_recent["Symbol"].astype(str)
    df_recent["Company"] = df_recent["Company"].astype(str)
    df_recent = df_recent[df_recent.apply(lambda r: is_single_stock(r.get("Symbol"), r.get("Company")), axis=1)]
    top_results = (df_recent.sort_values("FTD_Value", ascending=False)
                    .head(num_results)
                    .reset_index(drop=True))

    # Format QuantityFails with thousands separators for display/export
    try:
        # Keep a numeric backup if needed later
        top_results["QuantityFails_numeric"] = top_results["QuantityFails"].astype("Int64")
        top_results["QuantityFails"] = top_results["QuantityFails_numeric"].map(lambda x: f"{x:,}" if pd.notna(x) else "")
    except Exception:
        # Fallback formatting in case of unexpected types
        top_results["QuantityFails"] = top_results["QuantityFails"].apply(lambda x: f"{int(float(x)):,}" if pd.notna(x) else "")

    top_results['FTD_Value'] = top_results['FTD_Value'].map(lambda x: f"${x:,.2f}")

    # Reorder columns for display and export
    top_results = top_results[['SettlementDate','Symbol','Company','CUSIP','Price','QuantityFails','FTD_Value']]

    print(f"\n🎯 Top {num_results} NYSE/NASDAQ by FTD Value on {latest_settlement_date_formatted}:\n")
    print(top_results[['SettlementDate','Symbol','Company','CUSIP','Price','QuantityFails','FTD_Value']].to_string(index=False))

    # Export to Excel format (.xlsx)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = EXPORT_DIR / f"FTD_Top_{num_results}_{latest_date_str}.xlsx"
    top_results.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"\n✅ Exported Excel (XLSX) file: {excel_path}")

    return top_results, latest_settlement_date_formatted, latest_date_str, download_url, excel_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and export top FTD data")
    parser.add_argument("num_results", type=int, help="Number of top results to fetch (must be > 0)")
    parser.add_argument("--no-export", action="store_true", help="Skip file export")
    parser.add_argument("--discord", action="store_true", help="Post the result set to configured Discord webhooks")
    parser.add_argument("--test", action="store_true", help="Use the test Discord webhook list instead of the production list")
    parser.add_argument("--force", action="store_true", help="Force a Discord post even if the latest date has already been sent")

    args = parser.parse_args()

    if args.num_results <= 0:
        print("Error: Number of results must be greater than 0")
        sys.exit(1)

    top_results, latest_settlement_date, latest_date_str, latest_url, latest_export = fetch_top_ftds(
        num_results=args.num_results, export=not args.no_export
    )

    if args.discord:
        env_data = load_env_file(ENV_FILE)
        log_path = resolve_log_file(env_data)
        already_posted = not args.force and should_skip_post_for_date(latest_date_str, log_path)
        if already_posted:
            print(f"⚠️ Discord post skipped because {latest_date_str} is already logged; use --force to override.")
        else:
            posted = send_discord_notifications(
                top_results,
                latest_settlement_date,
                latest_url,
                attachment_path=latest_export,
                use_test=args.test,
                env_data=env_data,
            )
            if posted:
                update_post_log(log_path, latest_date_str)
            else:
                print("⚠️ Discord post failed; log entry not updated so the next run can retry.")
