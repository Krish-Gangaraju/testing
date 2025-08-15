from __future__ import annotations

import io
import os
import urllib.parse
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import warnings


# ── CONFIG ───────────────────────────────────────────────────────────
SQL_SERVER = "riskanalysis-server.database.windows.net"
SQL_DB = "RiskAnalysisDB"
COVER_SHEET = "Cover Page"
DATA_SHEET = "RA_Compilation"
DRIVER_NAME = "ODBC Driver 18 for SQL Server"

# Exact target table columns (exclude identity PK "ID")
# All fields optional; empty → NULL
TABLE_COLUMNS: List[str] = [
    "Project Title",
    "Project Reference No.",
    "Owner",
    "Type",
    "Date",
    "Type of gap",
    "Critical Aspect",
    "Description of Risk",
    "Description of Effects",
    "Severity",
    "Description of Causes",
    "Probability",
    "Eval",
    "Comments on Risk",
    "Type of Action",
    "Description of Actions",
    "Who",
    "When (date/timing)",
    "% Progress",
    "Results",
    "Comments on Action",
    "Residual Severity",
    "Residual Probability",
    "Residual Risk",
    "Comments on Residual",
    "Project Guidance",
]

LABEL_KEYWORDS: Dict[str, List[str]] = {
    "Project Title": ["Title of Affaire"],
    "Owner": ["Sponsor"],
}


# ── UTILITIES ────────────────────────────────────────────────────────
def _as_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def normalize_header(name: str) -> str:
    """
    Map common typos/short forms -> canonical column names.
    """
    s = str(name).strip()
    low = s.lower()
    direct = {
        "sever": "Severity",
        "proba": "Probability",
        "type of risk": "Type",
    }
    if low in direct:
        return direct[low]
    if low.startswith("proba"):
        return "Probability"
    if low.startswith("sever"):
        return "Severity"
    return s


def trim_at_two_blank_rows(df: pd.DataFrame) -> Optional[int]:
    """
    Return the row index to stop BEFORE the first of TWO consecutive fully empty rows.
    If none found, return None.
    """
    if df.empty:
        return None
    empties = df.isna().all(axis=1).to_numpy()
    for i in range(len(empties) - 1):
        if empties[i] and empties[i + 1]:
            return i
    return None


def trim_at_sparse_triplet(df: pd.DataFrame, ignore_cols: List[str]) -> Optional[int]:
    """
    Stop BEFORE the first row of any sequence where the NEXT 3 rows are each >50% empty,
    ignoring columns in `ignore_cols` for the emptiness check. Return index or None.
    """
    if df.empty:
        return None

    consider_cols = [c for c in df.columns if c not in ignore_cols]
    if not consider_cols:
        return None

    sub = df[consider_cols]
    if sub.shape[1] == 0:
        return None

    is_sparse = (sub.isna().mean(axis=1) > 0.5).to_numpy()
    for i in range(0, len(is_sparse) - 2):
        if is_sparse[i] and is_sparse[i + 1] and is_sparse[i + 2]:
            return i
    return None


def extract_label_values(cover_df: pd.DataFrame, keywords: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    # Start with your requested keys, plus the new field
    results: Dict[str, Optional[str]] = {k: None for k in keywords}
    results["Project Reference No."] = None  # NEW

    for idx, row in cover_df.iterrows():
        left  = _as_text(row.iloc[0] if len(row) > 0 else None).strip()  # label-left
        right = _as_text(row.iloc[1] if len(row) > 1 else None).strip()  # label-right

        for field, hints in keywords.items():
            if any(h.lower() in left.lower() or h.lower() in right.lower() for h in hints):
                # main value (D/E in your sheet slice)
                val_d = _as_text(row.iloc[2] if len(row) > 2 else None).strip()
                val_e = _as_text(row.iloc[3] if len(row) > 3 else None).strip()
                combined = " ".join([p for p in (val_d, val_e) if p]).strip()
                results[field] = combined if combined else None

                # If we just matched the Project Title row, also grab the cell just BELOW it
                if field == "Project Title" and (idx + 1) in cover_df.index:
                    below = cover_df.iloc[idx + 1]
                    below_d = _as_text(below.iloc[2] if len(below) > 2 else None).strip()
                    below_e = _as_text(below.iloc[3] if len(below) > 3 else None).strip()
                    below_combined = " ".join([p for p in (below_d, below_e) if p]).strip()
                    if below_combined:
                        results["Project Reference No."] = below_combined

    return results



def _detect_and_map_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify up to three generic 'comments' columns and map them into:
      - Comments on Risk
      - Comments on Action
      - Comments on Residual

    Rules:
      • If there are 3 comment-like columns, assign left→right to Risk/Action/Residual.
      • If fewer:
          - BEFORE 'Type' → Risk
          - AFTER 'Type' and BEFORE 'Residual Severity' → Action
          - AFTER 'Residual Severity' → Residual
      • If boundaries missing, fall back to left→right fill Risk, Action, Residual.

    Uses positional indexing (iloc) to avoid duplicate-name ambiguity.
    """
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    # boundaries
    idx_type = None
    for cand in ("type", "type of risk"):
        if cand in lower_cols:
            idx_type = lower_cols.index(cand)
            break
    idx_res_sev = lower_cols.index("residual severity") if "residual severity" in lower_cols else None

    # candidates
    cand_indices = [i for i, c in enumerate(lower_cols) if "comment" in c]
    for tcol in ("Comments on Risk", "Comments on Action", "Comments on Residual"):
        if tcol not in df.columns:
            df[tcol] = pd.NA

    if not cand_indices:
        return df

    cand_indices.sort()

    def _assign(target_col: str, source_idx: int):
        series = df.iloc[:, source_idx]
        df[target_col] = df[target_col].where(df[target_col].notna(), series)

    if len(cand_indices) >= 3:
        _assign("Comments on Risk", cand_indices[0])
        _assign("Comments on Action", cand_indices[1])
        _assign("Comments on Residual", cand_indices[2])
    else:
        used = set()
        for idx in cand_indices:
            if idx_type is not None and idx < idx_type:
                _assign("Comments on Risk", idx); used.add(idx)
        for idx in cand_indices:
            if idx in used: continue
            if idx_type is not None and idx_res_sev is not None and (idx_type < idx < idx_res_sev):
                _assign("Comments on Action", idx); used.add(idx)
        for idx in cand_indices:
            if idx in used: continue
            if idx_res_sev is not None and idx > idx_res_sev:
                _assign("Comments on Residual", idx); used.add(idx)

        leftovers = [i for i in cand_indices if i not in used]
        fallback_targets = [t for t in ["Comments on Risk", "Comments on Action", "Comments on Residual"] if df[t].isna().all()]
        for i, idx in enumerate(leftovers[: len(fallback_targets)]):
            _assign(fallback_targets[i], idx)

    # drop original raw comment columns (keep standardized)
    drop_names = [cols[i] for i in cand_indices if cols[i] not in ("Comments on Risk", "Comments on Action", "Comments on Residual")]
    df.drop(columns=[c for c in drop_names if c in df.columns], inplace=True, errors="ignore")
    return df


# ── BUILD DATAFRAME ──────────────────────────────────────────────────
def build_dataframe(file_like_or_bytes: bytes | io.BytesIO | str) -> pd.DataFrame:
    """
    Parse one Excel into a cleaned DataFrame (tolerant -> NULLs).
    Accepts a path, bytes, or BytesIO. Uses a single ExcelFile for efficiency.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    if isinstance(file_like_or_bytes, (bytes, bytearray)):
        xfh: io.BytesIO | str = io.BytesIO(file_like_or_bytes)
    else:
        xfh = file_like_or_bytes

    labels: Dict[str, Optional[str]] = {k: None for k in LABEL_KEYWORDS}

    try:
        with pd.ExcelFile(xfh, engine="openpyxl") as xls:
            # Cover sheet (optional)
            try:
                cover = pd.read_excel(xls, sheet_name=COVER_SHEET, header=None, usecols=[1, 2, 3, 4])
                labels = extract_label_values(cover, LABEL_KEYWORDS)
            except Exception:
                pass  # keep None; insert as NULL later

            # Data sheet (expected)
            try:
                df = pd.read_excel(xls, sheet_name=DATA_SHEET)
            except Exception:
                raise RuntimeError(f"cannot read sheet '{DATA_SHEET}'")
    except Exception as e:
        raise RuntimeError(f"Unable to read Excel file: {e}") from e

    # Basic cleanup
    df = df.iloc[3:].reset_index(drop=True)
    pd.set_option("future.no_silent_downcasting", True)
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

    # Early stop on two fully blank rows (pre-header)
    idx_stop = trim_at_two_blank_rows(df)
    if idx_stop is not None:
        df = df.iloc[:idx_stop]
    if len(df) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    # Header normalization
    raw_header = df.iloc[0].astype(str)
    clean_header = raw_header.str.replace(r"\s+", " ", regex=True).str.strip()
    df.columns = [normalize_header(c) for c in clean_header]
    df = df[1:].reset_index(drop=True)

    # Remove empty rows/cols
    df = df.loc[~df.isna().all(axis=1)]
    df = df.loc[:, ~df.isna().all(axis=0)]

    # Drop the first column (legacy index) if present
    if df.shape[1] > 0:
        df.drop(df.columns[0], axis=1, inplace=True, errors="ignore")

    # Insert labels (NULLs if missing)
    df.insert(0, "Project Title", labels.get("Project Title"))
    df.insert(1, "Project Reference No.", labels.get("Project Reference No."))  # NEW, right next to title
    df.insert(2, "Owner", labels.get("Owner"))

    # Map comment columns
    df = _detect_and_map_comments(df)

    # Replace lingering empty strings with NA → SQL NULL
    df = df.replace("", pd.NA)

    # New stop rule: next 3 rows >50% empty (ignore Project Title/Owner)
    idx_sparse = trim_at_sparse_triplet(df, ignore_cols=["Project Title", "Owner"])
    if idx_sparse is not None:
        df = df.iloc[:idx_sparse]

    return df


def align_to_table_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Ensure DataFrame matches TABLE_COLUMNS exactly.
    Returns: (aligned_df, missing_columns, extra_columns)
    """
    df_cols = list(df.columns)
    set_df = set(df_cols)
    set_tbl = set(TABLE_COLUMNS)

    missing = [c for c in TABLE_COLUMNS if c not in set_df]
    extra = [c for c in df_cols if c not in set_tbl]

    for c in missing:
        df[c] = pd.NA  # all optional
    if extra:
        df = df.drop(columns=extra, errors="ignore")

    df = df[TABLE_COLUMNS]
    return df, missing, extra


# ── DB ENGINE ────────────────────────────────────────────────────────
def make_engine():
    load_dotenv()
    client = "krishgangaraju@riskanalysis-server"
    secret = os.getenv("SQL_PASS")
    if not client or not secret:
        raise RuntimeError("Missing SQL_USER or SQL_PASS in environment (.env)")

    conn_params = {
        "Driver": f"{{{DRIVER_NAME}}}",
        "Server": SQL_SERVER,
        "Database": SQL_DB,
        "Encrypt": "yes",
        "TrustServerCertificate": "no",
        "UID": client,
        "PWD": secret,
        "Connect Timeout": "30",
    }
    conn_str = ";".join(f"{k}={v}" for k, v in conn_params.items())
    return create_engine(
        "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str),
        fast_executemany=True,
        pool_pre_ping=True,
    )


# ── SCHEMA MIGRATOR ──────────────────────────────────────────────────
def _q(name: str) -> str:
    """Bracket-escape SQL Server identifiers."""
    return "[" + name.replace("]", "]]") + "]"


def _get_existing_columns(engine: Engine) -> dict:
    sql = """
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'RiskAnalysis'
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {
        r["COLUMN_NAME"]: (r["DATA_TYPE"], r["CHARACTER_MAXIMUM_LENGTH"], r["IS_NULLABLE"])
        for r in rows
    }


def ensure_table_schema(engine: Engine) -> None:
    """
    Bring dbo.RiskAnalysis in line with TABLE_COLUMNS:
      • Add any missing columns as NVARCHAR(MAX) NULL
      • Widen/relax existing to NVARCHAR(MAX) NULL where needed
    Does not drop or rename columns.
    """
    existing = _get_existing_columns(engine)

    # Add missing
    add_stmts = []
    for col in TABLE_COLUMNS:
        if col not in existing:
            add_stmts.append(
                f"ALTER TABLE {_q('dbo')}.{_q('RiskAnalysis')} ADD {_q(col)} NVARCHAR(MAX) NULL;"
            )
    if add_stmts:
        with engine.begin() as conn:
            for stmt in add_stmts:
                conn.execute(text(stmt))

    # Refresh
    existing = _get_existing_columns(engine)

    # Widen/relax to NVARCHAR(MAX) NULL
    alter_stmts = []
    for col in TABLE_COLUMNS:
        if col in existing:
            dtype, maxlen, is_nullable = existing[col]
            needs_widen = not (dtype == "nvarchar" and (maxlen is None or maxlen == -1))
            needs_nullable = (is_nullable != "YES")
            if needs_widen or needs_nullable:
                alter_stmts.append(
                    f"ALTER TABLE {_q('dbo')}.{_q('RiskAnalysis')} "
                    f"ALTER COLUMN {_q(col)} NVARCHAR(MAX) NULL;"
                )
    if alter_stmts:
        with engine.begin() as conn:
            for stmt in alter_stmts:
                conn.execute(text(stmt))


# ── STREAMLIT UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="R2R3 AdR Risk Anaylsis Database Uploader", layout="wide")
st.title("Risk Anaylsis Database Single File Uploader")

with st.expander("Configuration", expanded=False):
    st.write(f"**SQL Server:** {SQL_SERVER}")
    st.write(f"**Database:** {SQL_DB}")
    st.write(f"**Target table:** dbo.RiskAnalysis")
    st.write(f"**Data sheet expected:** {DATA_SHEET}")
    st.write(f"**(Optional) cover sheet:** {COVER_SHEET}")

uploaded = st.file_uploader(
    "Drag & drop a single Excel file (.xlsx or .xlsm)",
    type=["xlsx", "xlsm"],
)

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

file_bytes = uploaded.read()

with st.spinner("Parsing and cleaning..."):
    try:
        df = build_dataframe(file_bytes)
        aligned_df, missing, extra = align_to_table_schema(df)
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

log_msgs = []
if missing:
    log_msgs.append(f"Empty or missing columns added as NULL: {missing}")
if extra:
    log_msgs.append(f"Extra columns dropped: {extra}")
if aligned_df.empty:
    log_msgs.append("No data rows after cleaning.")
if log_msgs:
    st.warning(" • " + "\n • ".join(log_msgs))

# --- Optional user entry for missing Project Reference No. ---
ref_col = "Project Reference No."
if ref_col in aligned_df.columns:
    ref_missing_mask = aligned_df[ref_col].isna() | (aligned_df[ref_col].astype(str).str.strip() == "")
    if ref_missing_mask.any():
        st.warning(
            "It looks like **Project Reference No.** wasn't found in the cover sheet "
            "for some or all rows. If you'd like, enter it here and we'll apply it to the "
            "missing cells before uploading."
        )
        st.text_input("**Project Reference No. (push to database after entering**", key="refno_input")


st.subheader("Extracted DataFrame")
st.dataframe(aligned_df, use_container_width=True)

if 'push_done' not in st.session_state:
    st.session_state.push_done = False

push = st.button("Push to Database", disabled=aligned_df.empty)

if push:
    try:
        engine = make_engine()
    except Exception as e:
        st.error(f"❌ Cannot create DB engine: {e}")
        st.stop()

    # Ensure SQL schema reflects TABLE_COLUMNS
    try:
        # If the user provided a value, fill only missing cells for this upload
        user_ref = st.session_state.get("refno_input", "")
        if user_ref:
            mask = aligned_df[ref_col].isna() | (aligned_df[ref_col].astype(str).str.strip() == "")
            if mask.any():
                aligned_df.loc[mask, ref_col] = user_ref.strip()

        ensure_table_schema(engine)
    except Exception as e:
        st.error(f"⛔ Failed to update table schema: {e}")
        st.stop()

    with st.spinner("Inserting rows in a single transaction..."):
        try:
            with engine.begin() as conn:
                aligned_df.to_sql(
                    "RiskAnalysis",
                    conn,
                    schema="dbo",
                    if_exists="append",
                    index=False,
                    method=None,
                    chunksize=1000,
                )
            st.session_state.push_done = True
            st.success(f"✅ Inserted {len(aligned_df)} rows into dbo.RiskAnalysis.")
        except Exception as e:
            st.error(f"⛔ Insert failed. Transaction rolled back.\n\n**Error:** {e}")

if st.session_state.push_done:
    st.caption("Done. You can upload another file if needed.")

