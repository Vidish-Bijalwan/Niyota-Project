import os
import json
import io
import base64
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

APP_TITLE = "Niyota – Wedding Money Collection"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
EVENTS = [
    "Mehndi",
    "Haldi",
    "Shaadi",
    "Reception",
]

def ensure_storage() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        empty: Dict[str, List[Dict[str, Any]]] = {e: [] for e in EVENTS}
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=2, ensure_ascii=False)


def load_data() -> Dict[str, List[dict[str, Any]]]:
    ensure_storage()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in EVENTS:
            data.setdefault(e, [])
        return data
    except Exception:
        return {e: [] for e in EVENTS}


def save_data(data: Dict[str, List[dict[str, Any]]]) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_image(file) -> str:
    ensure_storage()
    if file is None:
        return ""

    filename = file.name
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(UPLOAD_DIR, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    path = os.path.join(UPLOAD_DIR, candidate)
    with open(path, "wb") as out:
        out.write(file.getbuffer())
    try:
        return os.path.relpath(path, BASE_DIR)
    except Exception:
        return path

def save_photo(file) -> str:
    ensure_storage()
    if file is None:
        return ""
    filename = file.name if hasattr(file, "name") else "captured.png"
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".png"
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(PHOTOS_DIR, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    path = os.path.join(PHOTOS_DIR, candidate)
    with open(path, "wb") as out:
        out.write(file.getbuffer())
    try:
        return os.path.relpath(path, BASE_DIR)
    except Exception:
        return path


def as_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Name", "Address", "Amount", "Photo"])
    df = pd.DataFrame(records)
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).astype(int)
    if "Photo" in df.columns:
        df["Photo"] = df["Photo"].apply(lambda p: p if isinstance(p, str) and len(p) > 0 else None)
    cols = [c for c in ["Name", "Address", "Amount", "Photo"] if c in df.columns]
    return df[cols]


def total_amount(records: List[Dict[str, Any]]) -> int:
    return int(sum(int(r.get("Amount", 0) or 0) for r in records))

def init_session_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("selected_event", None)

def go_event_entry():
    st.session_state["page"] = "event_entry"


def go_dashboard():
    st.session_state["page"] = "dashboard"


def select_event(event_name: str):
    st.session_state["selected_event"] = event_name


def event_entry_view():
    st.header("Event Entry")

    cols = st.columns(4)
    for i, event_name in enumerate(EVENTS):
        if cols[i].button(event_name, key=f"evt_btn_{event_name}", use_container_width=True):
            select_event(event_name)

    current_event = st.session_state.get("selected_event")
    if not current_event:
        st.info("Select an event above to add a contribution entry.")
        return

    st.subheader(f"Add Entry – {current_event}")
    with st.form(key=f"entry_form_{current_event}", clear_on_submit=True):
        name = st.text_input("Name", key=f"name_{current_event}")
        address = st.text_area("Address", key=f"addr_{current_event}")

        amount_default = int(st.session_state.get(f"amt_{current_event}", 0) or 0)
        amount = st.number_input("Amount", min_value=0, step=100, value=amount_default, key=f"amt_{current_event}")
        photo_cap = st.camera_input("Capture Photo", key=f"photo_{current_event}")

        submitted = st.form_submit_button("Save Entry", use_container_width=True)

    if submitted:
        if not name:
            st.error("Name is required.")
            return
        if amount is None or int(amount) <= 0:
            st.error("Amount must be greater than 0.")
            return

        img_path = save_photo(photo_cap) if photo_cap is not None else ""

        data = load_data()
        data[current_event].append(
            {
                "Name": name.strip(),
                "Address": (address or "").strip(),
                "Amount": int(amount),
                "Photo": img_path,
            }
        )
        save_data(data)
        st.success(f"Saved entry for {current_event}.")

def dashboard_view():
    st.header("Dashboard")

    cols = st.columns(4)
    # Persist selected event across reruns
    if "dashboard_event" not in st.session_state:
        st.session_state["dashboard_event"] = None
    for i, event_name in enumerate(EVENTS):
        if cols[i].button(event_name, key=f"dash_btn_{event_name}", use_container_width=True):
            st.session_state["dashboard_event"] = event_name
    chosen = st.session_state.get("dashboard_event")

    data = load_data()

    if not chosen:
        pref = next((e for e in EVENTS if len(data.get(e, [])) > 0), None)
        st.session_state["dashboard_event"] = pref or EVENTS[0]
        chosen = st.session_state["dashboard_event"]

    st.subheader(f"{chosen} – Contributions")
    records = data.get(chosen, [])
    df = as_dataframe(records)
    df_display = df.copy()
    if "Photo" in df_display.columns:
        def _imgval(p: Any) -> Any:
            if isinstance(p, str) and len(p) > 0:
                ap = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
                if os.path.exists(ap):
                    try:
                        with open(ap, "rb") as f:
                            b = f.read()
                        ext = os.path.splitext(ap)[1].lower()
                        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/*"))
                        return f"data:{mime};base64,{base64.b64encode(b).decode()}"
                    except Exception:
                        return None
            return None
        df_display["Photo"] = df_display["Photo"].apply(_imgval)

    st.dataframe(
        df_display,
        use_container_width=True,
        column_config={
            "Photo": st.column_config.ImageColumn(
                "Photo",
                help="Contributor photo (if provided)",
                width="small",
            )
        },
    )

    st.metric(label="Total Amount", value=f"₹ {total_amount(records):,}")

    st.divider()
    st.subheader("Export")
    exp_c1, exp_c2 = st.columns(2)
    # Excel export
    with io.BytesIO() as excel_buffer:
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=chosen)
        excel_buffer.seek(0)
        exp_c1.download_button(
            label="Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"{chosen}_contributions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"dl_xlsx_{chosen}",
        )

    def df_to_pdf_bytes(pdf_title: str, dataframe: pd.DataFrame) -> bytes:
        if not REPORTLAB_AVAILABLE:
            return b""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(pdf_title, styles["Title"]))
        elements.append(Spacer(1, 12))

        headers = list(dataframe.columns)
        rows = []
        for row in dataframe.itertuples(index=False):
            cells = []
            for col, val in zip(headers, row):
                if col == "Photo" and isinstance(val, (str, bytes)):
                    p = val if isinstance(val, str) else ""
                    ap = os.path.join(BASE_DIR, p) if isinstance(p, str) and len(p) > 0 and not os.path.isabs(p) else p
                    if isinstance(ap, str) and os.path.exists(ap):
                        try:
                            img = RLImage(ap, width=60, height=60)
                            cells.append(img)
                            continue
                        except Exception:
                            pass
                cells.append("" if pd.isna(val) else str(val))
            rows.append(cells)
        table_data = [headers] + rows

        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        elements.append(tbl)
        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = df_to_pdf_bytes(f"{chosen} – Contributions", df)
    exp_c2.download_button(
        label=("Download PDF" if REPORTLAB_AVAILABLE else "Download PDF (ReportLab not installed)"),
        data=pdf_bytes,
        file_name=f"{chosen}_contributions.pdf",
        mime="application/pdf",
        disabled=(not REPORTLAB_AVAILABLE),
        use_container_width=True,
        key=f"dl_pdf_{chosen}",
    )

    st.divider()
    st.subheader("Edit Entries")
    if len(records) == 0:
        st.info("No records to edit.")
        return

    editable_df = df.copy()
    if "Delete" not in editable_df.columns:
        editable_df["Delete"] = False
    edited_df = st.data_editor(
        editable_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Name": st.column_config.TextColumn("Name", help="Contributor name", required=True),
            "Address": st.column_config.TextColumn("Address", help="Address (optional)"),
            "Amount": st.column_config.NumberColumn(
                "Amount",
                min_value=0,
                step=100,
                help="Contribution amount (₹)"
            ),
            "Photo": st.column_config.TextColumn("Photo", help="Image path (view-only)"),
            "Delete": st.column_config.CheckboxColumn("Delete", help="Mark to remove this row"),
        },
        disabled=["Photo"],
        key=f"editor_{chosen}",
    )

    if st.button("Save Changes", use_container_width=True, key=f"save_changes_{chosen}"):
        kept_df = edited_df.copy()
        if "Delete" in kept_df.columns:
            kept_df = kept_df[~kept_df["Delete"].fillna(False)].drop(columns=["Delete"], errors="ignore")
        blank_mask = kept_df["Name"].astype(str).str.strip().eq("") & (kept_df["Amount"].isna() | (kept_df["Amount"] == 0))
        kept_df = kept_df[~blank_mask]
        if kept_df.empty and len(records) > 0:
            pass
        elif kept_df["Name"].isna().any() or kept_df["Name"].astype(str).str.strip().eq("").any():
            st.error("All kept rows must have a Name.")
            st.stop()
        try:
            kept_df["Amount"] = pd.to_numeric(kept_df["Amount"], errors="coerce").fillna(0).astype(int)
        except Exception:
            st.error("Amounts must be numeric.")
            st.stop()
        if (kept_df["Amount"] < 0).any():
            st.error("Amounts cannot be negative.")
            st.stop()
        new_records: List[Dict[str, Any]] = []
        for i, row in kept_df.iterrows():
            photo_val = row.get("Photo", None)
            new_records.append(
                {
                    "Name": str(row.get("Name", "")).strip(),
                    "Address": str(row.get("Address", "" or "")).strip(),
                    "Amount": int(row.get("Amount", 0) or 0),
                    "Photo": (photo_val if isinstance(photo_val, str) and len(photo_val) > 0 else ""),
                }
            )

        data[chosen] = new_records
        save_data(data)
        st.success("Changes saved.")
        st.rerun()

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💌", layout="wide")
    init_session_state()

    st.title("Niyota")
    st.caption("A simple wedding money collection app")

    with st.container():
        c1, c2 = st.columns(2)
        if c1.button("Event Entry", key="nav_event", use_container_width=True):
            go_event_entry()
        if c2.button("Dashboard", key="nav_dash", use_container_width=True):
            go_dashboard()

    page = st.session_state.get("page", "home")
    if page == "event_entry":
        event_entry_view()
    elif page == "dashboard":
        dashboard_view()
    else:
        st.subheader("Home")
        st.write(
            "Use the buttons above to navigate to Event Entry to add contributions or Dashboard to view stats."
        )

if __name__ == "__main__":
    main()