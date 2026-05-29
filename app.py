"""
G5 Pivot2 (Metric 2) - Standalone Streamlit app.
Multi-month pivot tables, one per hotel (tabs). Upload CSV; hotels derived from data.
Handles different Stay Month formats (e.g. Jan, 2026 / Jan-26 / Jan,26).
"""

import io
import re
from datetime import datetime

import streamlit as st
import pandas as pd


# Canonical month display: "Jan, 2026"
MONTH_FORMATS_TRY = [
    ("%b, %Y", None),   # Jan, 2026
    ("%b-%y", None),   # Jan-26
    ("%b, %y", None),  # Jan, 26
    ("%b,%y", None),   # Jan,26
    ("%b %Y", None),
    ("%b %y", None),
    ("%B, %Y", None),  # January, 2026
    ("%B-%y", None),
    ("%B %Y", None),
    ("%B %y", None),
]


def _normalize_stay_month(raw) -> str | None:
    """Parse various month strings to canonical 'Jan, 2026'. Returns None if unparseable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    s = str(raw).strip()
    if not s:
        return None

    for fmt, _ in MONTH_FORMATS_TRY:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%b, %Y")
        except ValueError:
            continue

    # Fallback: try regex for "MonthYY" or "Month-YY" or "Month, YY" e.g. jan-26, Jan,26
    m = re.match(r"^([A-Za-z]+)[\s\-,\,]+(\d{2,4})$", s)
    if m:
        month_part = m.group(1).strip().title()
        year_part = m.group(2).strip()

        try:
            yr = int(year_part)
            if yr < 100:
                yr += 2000

            try:
                dt = datetime.strptime(f"{month_part} 1 {yr}", "%b %d %Y")
                return dt.strftime("%b, %Y")
            except ValueError:
                dt = datetime.strptime(f"{month_part} 1 {yr}", "%B %d %Y")
                return dt.strftime("%b, %Y")

        except ValueError:
            pass

    return None


# Optional short names for tab labels; unknown hotels use full name from data
HOTEL_SHORT_NAMES = {
    "The Grass Serviced Suites": "TG",
    "Hotel Amber Pattaya": "Amber PTY",
    "Hotel Amber Sukhumvit 85": "Amber 85",
    "Altera Hotel & Residence Pattaya": "Altera",
    "Arbour Hotel and Residence": "Arbour",
    "Arden Hotel & Residence Pattaya": "Arden",
    "Aster Hotel & Residence Pattaya": "Aster",
}


def parse_uploaded_report(uploaded_file) -> tuple[pd.DataFrame, str, str]:
    """Parse uploaded CSV. First 2 rows are title/generated date, row 3 is header."""
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()

    title = lines[0].strip().strip('"').split(",")[-1] if lines else ""
    generated = lines[1].strip().strip('"').split(",")[-1] if len(lines) > 1 else ""

    df = pd.read_csv(io.StringIO("\n".join(lines[2:]) if len(lines) > 2 else ""))

    return df, title, generated


def _build_hotel_table(
    df,
    hotel,
    pivot2_months,
    ref_col_map,
    short_names,
    month_order_p2,
    ref_order_p2,
    metric_short_p2,
):
    """Build pivot table and styler for one hotel. Returns styler or None if no data."""

    month_slice = df[
        (df["Stay Month"].isin(pivot2_months))
        & (df["Hotel"] == hotel)
    ]

    rows_p2 = []

    for _, r in month_slice.iterrows():
        month = r["Stay Month"]

        for ref_label, metric_cols in ref_col_map.items():
            row = {
                "Month": month,
                "Reference": ref_label,
            }

            for short_name in short_names:
                col = metric_cols.get(short_name)
                val = r.get(col) if col else None
                row[short_name] = pd.to_numeric(val, errors="coerce") if pd.notna(val) else None

            rows_p2.append(row)

    if not rows_p2:
        return None

    pivot_df_p2 = pd.DataFrame(rows_p2)

    # Convert occupancy from decimal to percentage
    pivot_df_p2["Occupancy (%)"] = pivot_df_p2["Occupancy"].apply(
        lambda x: round(float(x) * 100, 1) if pd.notna(x) else None
    )

    pivot_df_p2 = pivot_df_p2.drop(columns=["Occupancy"])

    melt_p2 = pivot_df_p2.melt(
        id_vars=["Month", "Reference"],
        value_vars=["Occupancy (%)", "Rooms", "ADR", "Revenue"],
        var_name="Metric",
        value_name="Value",
    )

    pivot_swapped_p2 = melt_p2.pivot_table(
        index=["Month", "Metric"],
        columns="Reference",
        values="Value",
    ).reset_index()

    pivot_swapped_p2["Metric"] = pivot_swapped_p2["Metric"].map(metric_short_p2)

    pivot_swapped_p2["Month"] = pd.Categorical(
        pivot_swapped_p2["Month"],
        categories=month_order_p2,
        ordered=True,
    )

    pivot_swapped_p2["Metric"] = pd.Categorical(
        pivot_swapped_p2["Metric"],
        categories=["Occ", "Room", "ADR", "Rev"],
        ordered=True,
    )

    pivot_swapped_p2 = pivot_swapped_p2.sort_values(["Month", "Metric"])

    ref_cols_p2 = [c for c in ref_order_p2 if c in pivot_swapped_p2.columns]

    pivot_swapped_p2 = pivot_swapped_p2[["Month", "Metric"] + ref_cols_p2]

    # Keep numeric version for calculation / styling logic
    pivot_swapped_p2_numeric = pivot_swapped_p2.copy()

    # Create separate display version for string formatting
    # This avoids Pandas dtype error when assigning "1,234" into numeric columns
    display_df = pivot_swapped_p2.copy()

    mask_occ = display_df["Metric"] == "Occ"

    for c in ref_cols_p2:
        display_df[c] = display_df[c].astype("object")

        display_df.loc[mask_occ, c] = pivot_swapped_p2_numeric.loc[mask_occ, c].apply(
            lambda x: f"{float(x):.1f}" if pd.notna(x) else ""
        )

        display_df.loc[~mask_occ, c] = pivot_swapped_p2_numeric.loc[~mask_occ, c].apply(
            lambda x: f"{int(round(float(x))):,}" if pd.notna(x) else ""
        )

    def _duetto_rev_style_p2(row):
        out = pd.Series("", index=ref_cols_p2)

        if (
            display_df.loc[row.name, "Metric"] != "Rev"
            or "Duetto" not in ref_cols_p2
            or "Budget" not in ref_cols_p2
        ):
            return out

        idx = row.name
        d = pivot_swapped_p2_numeric.loc[idx, "Duetto"]
        b = pivot_swapped_p2_numeric.loc[idx, "Budget"]

        if pd.notna(d) and pd.notna(b):
            if d < b:
                out["Duetto"] = "background-color: #ffcccc"
            elif d > b:
                out["Duetto"] = "background-color: #cce5ff"

        return out

    return (
        display_df.style
        .apply(_duetto_rev_style_p2, axis=1, subset=ref_cols_p2)
        .set_properties(subset=ref_cols_p2, **{"text-align": "right"})
    )


def main():
    st.set_page_config(page_title="G5 Pivot2", layout="wide")

    # Two columns: col1 = upload + Stay Month, col2 = Hotels + View radio
    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Upload report (CSV)",
            type=["csv"],
            key="pivot2_upload",
        )
        month_placeholder = st.empty()

    with col_right:
        hotel_placeholder = st.empty()
        view_placeholder = st.empty()

    if not uploaded:
        st.info("Upload a G5 report CSV to get started.")
        return

    # Cache parsed report so we don't re-parse on every widget interaction
    cache_key = (uploaded.name, uploaded.size)

    if (
        "pivot2_df_cache" not in st.session_state
        or st.session_state.pivot2_df_cache_key != cache_key
    ):
        df, title, generated = parse_uploaded_report(uploaded)

        df = df.rename(
            columns={
                df.columns[0]: "Hotel",
                df.columns[1]: "Stay Month",
            }
        )

        # Normalize Stay Month to canonical "Jan, 2026"
        df["Stay Month"] = df["Stay Month"].apply(
            lambda x: _normalize_stay_month(x) or x
        )

        st.session_state.pivot2_df_cache = (df, title, generated)
        st.session_state.pivot2_df_cache_key = cache_key

    df, title, generated = st.session_state.pivot2_df_cache

    all_hotels = [
        h
        for h in df["Hotel"].dropna().unique()
        if h and str(h).strip() and h != "Total"
    ]

    month_order_p2 = [
        "Jan, 2026",
        "Feb, 2026",
        "Mar, 2026",
        "Apr, 2026",
        "May, 2026",
        "Jun, 2026",
        "Jul, 2026",
        "Aug, 2026",
        "Sep, 2026",
        "Oct, 2026",
        "Nov, 2026",
        "Dec, 2026",
    ]

    months_avail_p2 = [
        m for m in month_order_p2
        if m in df["Stay Month"].values
    ]

    this_month = datetime.now().strftime("%b, %Y")

    default_months = (
        [this_month]
        if this_month in months_avail_p2
        else (months_avail_p2[:1] if months_avail_p2 else [])
    )

    with col_left:
        pivot2_months = month_placeholder.multiselect(
            "Stay Month",
            options=months_avail_p2,
            default=default_months,
            key="pivot2_months",
        )

    with col_right:
        selected_hotels = hotel_placeholder.multiselect(
            "Hotels",
            options=all_hotels,
            default=all_hotels,
            key="pivot2_hotels",
            placeholder="All hotels",
        )

        view_mode = view_placeholder.radio(
            "View",
            options=["Tab view", "List view"],
            index=0,
            key="pivot2_view",
            horizontal=True,
        )

    if not selected_hotels:
        st.warning("Select at least one hotel.")
        return

    st.caption(generated)

    ref_prefixes_p2 = [
        ("Today", "Today"),
        ("STLY", "STLY (DOW)"),
        ("ST2Y", "ST2Y (DOW)"),
        ("ST3Y", "ST3Y (DOW)"),
        ("Duetto", "Duetto Forecast"),
        ("Budget", "Locked Budget"),
        ("Final LY", "Final LY (DOW)"),
        ("Final 2Y", "Final 2Y (DOW)"),
        ("Final 3Y", "Final 3Y (DOW)"),
    ]

    metric_suffixes_p2 = [
        ("Occupancy (Physical)", "Occupancy"),
        ("Rooms (Commit)", "Rooms"),
        ("ADR (Commit)", "ADR"),
        ("Room Revenue (Commit)", "Revenue"),
    ]

    def _find_col_p2(prefix: str, suffix: str):
        for c in df.columns:
            if prefix in c and suffix in c:
                return c
        return None

    # Build column map once
    ref_col_map: dict[str, dict[str, str]] = {}

    for ref_label, col_prefix in ref_prefixes_p2:
        ref_col_map[ref_label] = {}

        for full_suffix, short_name in metric_suffixes_p2:
            col = _find_col_p2(col_prefix, full_suffix)

            if col is not None:
                ref_col_map[ref_label][short_name] = col

    if not pivot2_months:
        st.info("Select at least one month.")
        return

    ref_order_p2 = [
        "Today",
        "STLY",
        "ST2Y",
        "ST3Y",
        "Duetto",
        "Budget",
        "Final LY",
        "Final 2Y",
        "Final 3Y",
    ]

    metric_short_p2 = {
        "Occupancy (%)": "Occ",
        "Rooms": "Room",
        "ADR": "ADR",
        "Revenue": "Rev",
    }

    short_names = [s for _, s in metric_suffixes_p2]

    def render_table(styler):
        if styler is not None:
            st.dataframe(
                styler,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No data for selected months.")

    if view_mode == "Tab view":
        tab_labels = [
            HOTEL_SHORT_NAMES.get(h, h)
            for h in selected_hotels
        ]

        tabs = st.tabs(tab_labels) if tab_labels else []

        for tab, hotel in zip(tabs, selected_hotels):
            with tab:
                styler = _build_hotel_table(
                    df,
                    hotel,
                    pivot2_months,
                    ref_col_map,
                    short_names,
                    month_order_p2,
                    ref_order_p2,
                    metric_short_p2,
                )
                render_table(styler)

    else:
        for hotel in selected_hotels:
            label = HOTEL_SHORT_NAMES.get(hotel, hotel)
            st.subheader(label)

            styler = _build_hotel_table(
                df,
                hotel,
                pivot2_months,
                ref_col_map,
                short_names,
                month_order_p2,
                ref_order_p2,
                metric_short_p2,
            )
            render_table(styler)


if __name__ == "__main__":
    main()
