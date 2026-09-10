"""ESGLens — AI-powered ESG report analyser (Gemini edition)."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyser import (
    ENV_FIELDS,
    GOV_FIELDS,
    MAX_FILE_MB,
    MODEL_NAME,
    SOCIAL_FIELDS,
    ESGExtraction,
    Environmental,
    Governance,
    Social,
    analyse_report,
    ask_about_reports,
    check_comparability,
    get_api_key,
    get_api_keys,
    report_labels,
    to_rows,
)

GREEN = "#1B8A5A"
GREY = "#d9d9d9"
APP_BUILD_DATE = date.today().strftime("%d %b %Y")  # recomputed each time the app process starts


def logo_svg(size: int = 26) -> str:
    """Small original mark — magnifying glass over an ascending bar chart
    (analysis + data) — used in place of the 🔎 emoji wherever the brand
    appears, so it renders consistently and reads as an actual logo rather
    than emoji. Colour is inherited via `currentColor` from the wrapping
    element, so the same markup works on both dark (hero) and light
    (sidebar) backgrounds."""
    return f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
        xmlns="http://www.w3.org/2000/svg" style="display:block;flex-shrink:0;">
        <circle cx="10" cy="10" r="7.5" stroke="currentColor" stroke-width="2"/>
        <rect x="6.6" y="10.6" width="1.7" height="3.9" rx="0.5" fill="currentColor"/>
        <rect x="9.2" y="8.1" width="1.7" height="6.4" rx="0.5" fill="currentColor"/>
        <rect x="11.8" y="6.1" width="1.7" height="8.4" rx="0.5" fill="currentColor"/>
        <line x1="15.3" y1="15.3" x2="21" y2="21" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
    </svg>'''

st.set_page_config(page_title="ESGLens", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .esglens-hero {
        padding: 1.4rem 1.6rem; border-radius: 14px;
        background: linear-gradient(135deg, #103d2c 0%, #1b8a5a 100%);
        color: #f4fbf7; margin-bottom: 1.2rem; text-align: center;
    }
    .esglens-hero .brand-row { display: flex; align-items: center; justify-content: center; gap: .55rem; }
    .esglens-hero h1 { margin: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: .01em; white-space: nowrap; }
    .esglens-hero p { margin: .5rem auto 0 auto; opacity: .92; font-size: .96rem; max-width: 620px; }
    .sidebar-brand-row { display: flex; align-items: center; gap: .5rem; white-space: nowrap; margin-bottom: .1rem; }
    .sidebar-brand-row .brand-name { font-size: 1.45rem; font-weight: 800; color: #12241C; white-space: nowrap; line-height: 1; }
    .stat-card {
        padding: .65rem .95rem; border: 1px solid #e3ece7; border-radius: 12px;
        background: #fafcfb; margin-bottom: .6rem; min-height: 72px;
    }
    .stat-label { font-size: .76rem; color: #5b6b62; margin-bottom: .2rem; font-weight: 500; }
    .stat-value { font-size: 1.02rem; font-weight: 650; color: #12241C; line-height: 1.35; word-break: break-word; }
    .stat-sub { font-size: .74rem; color: #8a978f; margin-top: .15rem; }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600; margin: 2px 6px 2px 0; }
    .pill-yes { background: #d7f2e3; color: #146c43; }
    .pill-no { background: #f1f1f1; color: #767676; }
    .conflict-box { background: #fff6e5; border: 1px solid #f0c869; border-radius: 10px; padding: .7rem 1rem; margin-bottom: .5rem; font-size: .9rem; }
    .compare-note { background: #fdeeee; border: 1px solid #e8a1a1; border-radius: 10px; padding: .55rem .9rem; margin-bottom: .4rem; font-size: .88rem; color: #7a2020; }
    .extra-box { background: #f4f8f6; border: 1px solid #dbe8e1; border-radius: 10px; padding: .6rem .9rem; margin-bottom: .5rem; }
    .extra-topic { font-weight: 650; font-size: .92rem; }
    .page-note { font-size: .78rem; color: #8a978f; margin: .2rem 0 .8rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# API key(s) — env var or st.secrets only. No manual-entry box: keeps keys
# out of the UI entirely, matching how they're actually configured
# (secrets.toml locally / Secrets on Streamlit Cloud).
#
# Multiple keys (optional): add a GEMINI_API_KEYS list to secrets.toml —
# see .streamlit/secrets.toml.example. Gemini's free-tier daily cap is
# enforced per PROJECT, and each separate Google account normally gets its
# own project, so several personal keys give several independent daily
# budgets; analyser.py rotates to the next one automatically if one hits
# its cap mid-analysis.
# --------------------------------------------------------------------------

if "GEMINI_API_KEYS" not in os.environ and "GEMINI_API_KEY" not in os.environ:
    try:
        secret_multi = st.secrets.get("GEMINI_API_KEYS", None)
        secret_single = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_multi, secret_single = None, ""
    if secret_multi:
        os.environ["GEMINI_API_KEYS"] = ",".join(str(k).strip() for k in secret_multi if str(k).strip())
    elif secret_single:
        os.environ["GEMINI_API_KEY"] = secret_single

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand-row"><span style="color:{GREEN};">{logo_svg(28)}</span>'
        f'<span class="brand-name">ESGLens</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-powered ESG report analyser")
    st.markdown("---")
    mode = st.radio("Mode", ["Single report", "Compare reports"], index=0)
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Upload sustainability report PDF(s)\n"
        "2. Gemini reads the document natively — tables and charts included\n"
        "3. Every value comes back with a page number and a supporting quote\n"
        "4. Nothing is guessed — unfound fields are marked *Not disclosed*"
    )
    st.markdown("---")
    n_keys = len(get_api_keys())
    if n_keys == 0:
        st.caption("⚠️ No Gemini API key found — add GEMINI_API_KEY (or GEMINI_API_KEYS for several) to .streamlit/secrets.toml.")
    elif n_keys > 1:
        st.caption(f"🔑 {n_keys} API keys configured — auto-rotates if one hits today's free-tier cap.")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Stack: Gemini API (native PDF vision) · Streamlit · Plotly")
    st.caption(f"Build: {APP_BUILD_DATE}")

st.markdown(
    f"""
    <div class="esglens-hero">
        <div class="brand-row">{logo_svg(30)}<h1>ESGLens</h1></div>
        <p>Upload a report, get a company's Environmental, Social and Governance story in one dashboard —
        every number page-cited, nothing scored or guessed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Chart + render helpers
# --------------------------------------------------------------------------

def stat_card(label, value, sub=None):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def sv_text(sv):
    return sv.value if sv.found else "Not disclosed"


def _bar(labels, values, unit="", key=None):
    if not values:
        st.caption("No numeric figures found here to chart.")
        return
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=GREEN,
        text=[f"{v:,.1f}" if v % 1 else f"{v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), yaxis_title=unit, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)


def _donut(pct, label, key=None):
    if pct is None:
        return
    pct = max(0.0, min(100.0, pct))
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct], labels=[label, "Remaining"], hole=.65,
        marker_colors=[GREEN, GREY], textinfo="none",
    ))
    fig.update_layout(
        height=230, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
        annotations=[dict(text=f"{pct:.0f}%", x=0.5, y=0.5, font_size=22, showarrow=False)],
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(label)


def render_conflicts(ex: ESGExtraction):
    if ex.conflicts:
        st.markdown("**⚠️ Conflicting values found in this document**")
        for c in ex.conflicts:
            st.markdown(
                f'<div class="conflict-box"><b>{c.field}</b>: '
                f'{" vs. ".join(c.values_found)}<br><span style="opacity:.8">{c.note}</span></div>',
                unsafe_allow_html=True,
            )


def render_environmental(env: Environmental, ns: str):
    st.markdown("#### Emissions")
    labels, values = [], []
    for label, sv in [
        ("Scope 1", env.scope1_emissions),
        ("Scope 2 (location)", env.scope2_location_based),
        ("Scope 2 (market)", env.scope2_market_based),
        ("Scope 2 (total)", env.scope2_total_unspecified),
        ("Scope 3", env.scope3_total),
    ]:
        if sv.numeric is not None:
            labels.append(label)
            values.append(sv.numeric)
    _bar(labels, values, unit="tCO2e", key=f"{ns}_env_emissions_bar")
    if env.scope3_categories_reported.found:
        st.caption(f"Scope 3 categories reported: {env.scope3_categories_reported.value}")

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Net-zero target", sv_text(env.net_zero_target_year))
        stat_card("SBTi status", sv_text(env.sbti_status))
    with c2:
        if env.renewable_energy_pct.numeric is not None:
            _donut(env.renewable_energy_pct.numeric, "Renewable energy", key=f"{ns}_env_renewable_donut")
        else:
            stat_card("Renewable energy", sv_text(env.renewable_energy_pct) if env.renewable_energy_pct.found else sv_text(env.renewable_energy_quantity))
    with c3:
        stat_card("Total energy", sv_text(env.total_energy_consumption))
        stat_card("Total water", sv_text(env.water_consumption))

    st.markdown("#### Circularity")
    c4, c5, c6 = st.columns(3)
    with c4:
        if env.water_recycled_pct.numeric is not None:
            _donut(env.water_recycled_pct.numeric, "Water recycled/reused", key=f"{ns}_env_water_donut")
        else:
            stat_card("Water recycled/reused", sv_text(env.water_recycled_pct))
    with c5:
        if env.waste_recycled_pct.numeric is not None:
            _donut(env.waste_recycled_pct.numeric, "Waste recycled/diverted", key=f"{ns}_env_waste_donut")
        else:
            stat_card("Waste recycled/diverted", sv_text(env.waste_recycled_pct))
    with c6:
        stat_card("Total waste generated", sv_text(env.total_waste))


def render_social(soc: Social, ns: str):
    st.markdown("#### Gender diversity")
    labels, values = [], []
    for label, sv in [
        ("Workforce", soc.women_workforce_pct),
        ("Senior leadership", soc.women_leadership_pct),
    ]:
        if sv.numeric is not None:
            labels.append(label)
            values.append(sv.numeric)
    _bar(labels, values, unit="% women", key=f"{ns}_social_gender_bar")

    st.markdown("#### Workforce & safety")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Total employees", sv_text(soc.total_employees))
        stat_card("Turnover", sv_text(soc.employee_turnover_pct))
    with c2:
        stat_card("LTIFR", sv_text(soc.ltifr), sub="Lost Time Injury Frequency Rate")
        stat_card("TRIR", sv_text(soc.trir), sub="Total Recordable Incident Rate")
    with c3:
        stat_card("Fatality rate", sv_text(soc.fatality_rate))
        stat_card("Training hrs/employee", sv_text(soc.training_hours_per_employee))
    with c4:
        stat_card("Community investment", sv_text(soc.community_investment))


def _policy_pill(label, sv):
    cls = "pill-yes" if sv.found else "pill-no"
    text = sv.value if sv.found else "Not disclosed"
    st.markdown(f'<span class="pill {cls}">{label}: {text}</span>', unsafe_allow_html=True)


def render_governance(gov: Governance, ns: str):
    st.markdown("#### Board composition")
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Board size", sv_text(gov.board_size))
    with c2:
        if gov.board_independence_pct.numeric is not None:
            _donut(gov.board_independence_pct.numeric, "Independent directors", key=f"{ns}_gov_independence_donut")
        else:
            stat_card("Independent directors", sv_text(gov.board_independence_pct))
    with c3:
        if gov.women_on_board_pct.numeric is not None:
            _donut(gov.women_on_board_pct.numeric, "Women on board", key=f"{ns}_gov_women_donut")
        else:
            stat_card("Women on board", sv_text(gov.women_on_board_pct))

    st.markdown("#### Policies & assurance")
    _policy_pill("Anti-corruption policy", gov.anti_corruption_policy)
    _policy_pill("Whistleblower mechanism", gov.whistleblower_mechanism)
    _policy_pill("Data privacy & security", gov.data_privacy_disclosures)
    _policy_pill("ESG-linked exec. pay", gov.esg_linked_compensation)
    st.markdown("")
    stat_card("External assurance provider", sv_text(gov.assurance_provider))


def render_additional(ex: ESGExtraction):
    if not ex.additional_disclosures:
        st.caption("No additional material disclosures beyond the categories above were found.")
        return
    st.markdown("Other material disclosures the report makes, beyond the standard fields above:")
    for d in ex.additional_disclosures:
        page_txt = f" (p.{d.page})" if d.page else ""
        st.markdown(
            f'<div class="extra-box"><span class="extra-topic">{d.topic}</span> '
            f'<span style="opacity:.6">· {d.category}{page_txt}</span><br>{d.value}</div>',
            unsafe_allow_html=True,
        )


def render_detail_table(ex: ESGExtraction, ns: str):
    df = pd.DataFrame(to_rows(ex))
    # Explicit column widths + a fixed height give the grid a real reason to
    # show its horizontal scrollbar — left at auto-width, columns just get
    # squeezed instead, which was cutting off the (wide) Quote column with
    # no way to scroll to it.
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=460,
        key=f"{ns}_full_table",
        column_config={
            "Category": st.column_config.TextColumn(width="small"),
            "Metric": st.column_config.TextColumn(width="medium"),
            "Value": st.column_config.TextColumn(width="medium"),
            "Page": st.column_config.TextColumn(width="small"),
            "Trend": st.column_config.TextColumn(width="medium"),
            "Quote": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "↔️ Drag the scrollbar at the bottom of the table (or click a cell, then use the arrow "
        "keys) to reach the Quote column — it's wide. The CSV/JSON download below has every "
        "column too, if that's easier."
    )
    st.markdown(
        '<div class="page-note">Page numbers follow the report\'s own printed page numbers '
        '(as shown in its footer) — this can differ from your PDF viewer\'s page counter if the '
        "document has unnumbered cover/contents pages before its internal numbering starts. "
        "The quote is the most reliable way to locate a value — search for it directly.</div>",
        unsafe_allow_html=True,
    )


def download_buttons(ex: ESGExtraction, ns: str):
    col1, col2 = st.columns(2)
    df = pd.DataFrame(to_rows(ex))
    with col1:
        st.download_button("📥 Download CSV", data=df.to_csv(index=False),
                            file_name=f"{ex.company_name or 'report'}_ESGLens.csv",
                            mime="text/csv", key=f"{ns}_csv")
    with col2:
        st.download_button("📥 Download JSON", data=ex.model_dump_json(indent=2),
                            file_name=f"{ex.company_name or 'report'}_ESGLens.json",
                            mime="application/json", key=f"{ns}_json")


def render_dashboard(ex: ESGExtraction, ns: str):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Company", ex.company_name or "—")
    with c2:
        stat_card("Reporting year", ex.reporting_year or "—")
    with c3:
        stat_card("Framework", sv_text(ex.reporting_framework))
    with c4:
        stat_card("Sector", ex.industry_sector or "—")
    if ex.material_topics.found:
        st.caption(f"**Material topics:** {ex.material_topics.value}")

    render_conflicts(ex)

    tab_env, tab_soc, tab_gov, tab_extra, tab_data = st.tabs(
        ["🌍 Environmental", "👥 Social", "🏛️ Governance", "➕ Other disclosures", "📄 Full data & sources"]
    )
    with tab_env:
        render_environmental(ex.environmental, ns)
    with tab_soc:
        render_social(ex.social, ns)
    with tab_gov:
        render_governance(ex.governance, ns)
    with tab_extra:
        render_additional(ex)
    with tab_data:
        render_detail_table(ex, ns)
        download_buttons(ex, ns)


def validate_size(file) -> bool:
    size_mb = len(file.getvalue()) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"{file.name} is {size_mb:.1f} MB — Gemini's per-document limit is {MAX_FILE_MB} MB.")
        return False
    return True


def run_with_status(pdf_bytes, filename, status_box, label_prefix=""):
    """Runs analyse_report — ONE Gemini call per report, see analyser.py —
    and turns progress, transient retries, and API-key rotation into
    readable live messages instead of one long silent wait."""
    def on_progress(msg):
        status_box.write(f"{label_prefix}{msg}")

    def on_retry(wait_s):
        status_box.write(f"{label_prefix}Rate limit, a dropped connection, or Google's servers are busy — waiting {wait_s:.0f}s before retrying…")

    def on_key_switch(usable, total):
        status_box.write(
            f"{label_prefix}That API key hit today's free-tier daily cap — "
            f"switching to another configured key ({usable}/{total} still usable today)…"
        )

    return analyse_report(pdf_bytes, filename, on_progress=on_progress, on_retry=on_retry, on_key_switch=on_key_switch)


def explain_error(e: Exception) -> str:
    """Turns common low-level errors into plain language instead of a raw traceback."""
    msg = str(e)
    if "getaddrinfo failed" in msg or "Errno 11001" in msg or "NameResolutionError" in msg:
        return (
            "**Can't reach Google's servers — this is a network/DNS problem on this machine, "
            "not an ESGLens or API issue.** Check your internet connection, and if you're on a "
            "VPN, proxy, or work/college network, try disabling it. Some antivirus tools also "
            "block Python's outbound connections — check your firewall settings."
        )
    if msg.startswith("[quota]"):
        return (
            f"**{msg[len('[quota] '):]}** Add more keys under `GEMINI_API_KEYS` in "
            "`.streamlit/secrets.toml` to keep working past one key's daily cap — see the FAQ below."
        )
    if "API key not valid" in msg or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
        return "**One of your Gemini API keys looks invalid or lacks permission.** Double-check it in `.streamlit/secrets.toml` against the key at aistudio.google.com/apikey."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        return "**Free-tier rate limit hit.** This is usually a short per-minute throttle — try again in a moment."
    if any(s in msg.lower() for s in ("server disconnected", "remoteprotocolerror", "deadline exceeded", "read timed out", "readtimeout")):
        return (
            "**Gemini took too long thinking through this report and the connection was dropped.** "
            "This happens occasionally on long/dense reports. Try again — it's a per-request fluke."
        )
    if "not valid json" in msg.lower() or "parsing step" in msg.lower():
        return "**Gemini's response wasn't valid JSON, even after ESGLens asked it to self-correct once.** This is rare — try again."
    if "UNAVAILABLE" in msg or ("503" in msg and "high demand" in msg.lower()):
        return (
            "**Google's Gemini servers are overloaded right now (503 — high demand), even after "
            "retrying.** This is on Google's side, not your setup, PDF, or API key — it's the same "
            "message you'd get from anyone using this model at the moment. It usually clears within "
            "a few minutes; try again shortly."
        )
    if "400" in msg and "invalid argument" in msg.lower():
        return f"**Gemini rejected the request (400 invalid argument).** This shouldn't happen with the current setup — please retry, and if it keeps recurring, note the debug details: `{msg}`"
    if "password" in msg.lower() or "encrypted" in msg.lower():
        return "**This PDF looks password-protected.** Remove the password and re-upload."
    return f"**Something went wrong:** {msg}"


# --------------------------------------------------------------------------
# Copilot — "ask anything about this report(s)" — new addition, placed in
# the same row as the Analyse/Compare button in each mode below. Uses
# analyser.ask_about_reports(), which shares the same key-pool/retry/quota
# machinery as the main analysis, just for a plain free-text answer.
# --------------------------------------------------------------------------

def render_copilot(files: list, ns: str, disabled: bool):
    chat_key = f"{ns}_copilot_chat"
    st.session_state.setdefault(chat_key, [])

    with st.form(key=f"{ns}_copilot_form", clear_on_submit=True, border=False):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            question = st.text_input(
                "Ask anything about this report",
                placeholder="💬 Ask anything about this report — e.g. What does it say about supply chain risk?",
                disabled=disabled, label_visibility="collapsed", key=f"{ns}_copilot_input",
            )
        with btn_col:
            asked = st.form_submit_button("Ask", disabled=disabled, use_container_width=True)

    if asked and question and question.strip():
        if not files:
            st.warning("Upload a report first, then ask.")
        else:
            with st.spinner("Reading the report to answer…"):
                try:
                    answer = ask_about_reports(files, question.strip())
                except Exception as e:
                    answer = f"⚠️ {explain_error(e)}"
            st.session_state[chat_key].append((question.strip(), answer))

    for q, a in reversed(st.session_state[chat_key]):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)


def render_compare_results(ok_results):
    """Unchanged rendering logic for a completed comparison — factored out
    so it can be called both right after computing it and when re-showing
    a persisted result on a later rerun (e.g. after asking the copilot a
    question), without duplicating the block."""
    labels = report_labels(ok_results)

    notes = check_comparability(ok_results)
    if notes:
        st.markdown("**⚠️ Comparability notes**")
        for n in notes:
            st.markdown(f'<div class="compare-note">{n}</div>', unsafe_allow_html=True)

    st.markdown("### Side-by-side comparison")
    all_fields = (
        [(a, l, lambda r, a=a: getattr(r.environmental, a)) for a, l in ENV_FIELDS]
        + [(a, l, lambda r, a=a: getattr(r.social, a)) for a, l in SOCIAL_FIELDS]
        + [(a, l, lambda r, a=a: getattr(r.governance, a)) for a, l in GOV_FIELDS]
    )
    table = {"Metric": [label for _, label, _ in all_fields]}
    for label, r in zip(labels, ok_results):
        table[label] = [
            (getter(r).value if getter(r).found else "Not disclosed") for _, _, getter in all_fields
        ]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.markdown("### Company dashboards")
    tabs = st.tabs(labels)
    for i, (tab, r) in enumerate(zip(tabs, ok_results)):
        with tab:
            render_dashboard(r, f"cmp_{i}")

    combined_json = json.dumps([r.model_dump() for r in ok_results], indent=2)
    st.download_button("📥 Download full comparison as JSON", data=combined_json,
                        file_name="ESGLens_comparison.json", mime="application/json")


# --------------------------------------------------------------------------
# Single report mode
# --------------------------------------------------------------------------

if mode == "Single report":
    uploaded_file = st.file_uploader(
        "Upload sustainability report PDF", type=["pdf"],
        help="Works with GRI reports, BRSR filings, CDP disclosures, integrated annual reports.",
    )

    btn_col, copilot_col = st.columns([1, 2])
    with btn_col:
        analyse_clicked = st.button("🔍 Analyse report", type="primary",
                                     disabled=(uploaded_file is None or not get_api_key()))
    with copilot_col:
        render_copilot(
            [(uploaded_file.getvalue(), uploaded_file.name)] if uploaded_file else [],
            ns="single", disabled=(uploaded_file is None or not get_api_key()),
        )
    if not get_api_key():
        st.info("No Gemini API key configured — add GEMINI_API_KEY to .streamlit/secrets.toml.")

    if analyse_clicked and uploaded_file and validate_size(uploaded_file):
        with st.status("Analysing report…", expanded=True) as status:
            status.write("Uploading PDF to Gemini (native document understanding)…")
            try:
                result = run_with_status(uploaded_file.getvalue(), uploaded_file.name, status)
                status.update(label="Analysis complete", state="complete", expanded=False)
                st.session_state["single_result"] = result
                st.session_state["single_result_filename"] = uploaded_file.name
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(explain_error(e))
                result = None
        if result:
            render_dashboard(result, "single")
    elif (
        uploaded_file is not None
        and st.session_state.get("single_result") is not None
        and st.session_state.get("single_result_filename") == uploaded_file.name
    ):
        # Re-shows the already-computed dashboard on reruns triggered by
        # something else on the page (e.g. the copilot form below) — so
        # asking a question doesn't blank out results you already paid a
        # request for, and doesn't need a fresh Analyse click to get them back.
        render_dashboard(st.session_state["single_result"], "single")
    else:
        st.info("👆 Upload a PDF and click **Analyse report** to begin.")

    with st.expander("❓ FAQ"):
        st.markdown(
            "**Why is every value page-cited?** So you never have to take the extraction on "
            "faith — check any number against the source page in seconds. Pages follow the "
            "report's own printed page numbers, not your PDF viewer's page count.\n\n"
            "**What does 'Not disclosed' mean?** The model looked for that metric and didn't "
            "find it stated anywhere in the document — it never fills in a guess.\n\n"
            "**Is my data private?** The PDF is sent to Google's Gemini API for analysis and "
            "stored there for up to 48 hours. Don't upload unpublished/confidential reports.\n\n"
            "**Why did it say a key 'hit today's cap'?** Gemini's free tier currently allows "
            "only ~20 requests per day, per Google Cloud project, resetting at midnight Pacific "
            "Time. ESGLens uses one request per report, so this is mainly a compare-mode/heavy-use "
            "thing. Since that cap is per *project* (not per key) and a fresh Google account "
            "normally gets its own project, adding a `GEMINI_API_KEYS = [\"key1\", \"key2\", ...]` "
            "list to `secrets.toml` — one key per Google account — gives ESGLens that many "
            "independent daily budgets; it rotates to the next automatically."
        )


# --------------------------------------------------------------------------
# Compare mode (2–5 reports) — sequential, one Gemini call per report. This
# is also why compare mode is the main place the free tier's ~20-requests/
# day cap actually bites; see the FAQ below for the multi-key workaround.
# --------------------------------------------------------------------------

else:
    uploaded_files = st.file_uploader(
        "Upload 2–5 sustainability report PDFs to compare", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) > 5:
        st.warning("Please upload at most 5 reports at a time — showing the first 5.")
        uploaded_files = uploaded_files[:5]

    ready = uploaded_files and len(uploaded_files) >= 2 and get_api_key()
    copilot_ready = bool(uploaded_files) and bool(get_api_key())
    btn_col, copilot_col = st.columns([1, 2])
    with btn_col:
        compare_clicked = st.button("🔍 Compare reports", type="primary", disabled=not ready)
    with copilot_col:
        render_copilot(
            [(f.getvalue(), f.name) for f in uploaded_files] if uploaded_files else [],
            ns="compare", disabled=not copilot_ready,
        )

    if uploaded_files and len(uploaded_files) == 1:
        st.info("Add at least one more report to compare.")
    if not get_api_key():
        st.info("No Gemini API key configured — add GEMINI_API_KEY to .streamlit/secrets.toml.")

    if compare_clicked and all(validate_size(f) for f in uploaded_files):
        ordered = [None] * len(uploaded_files)
        with st.status(f"Analysing {len(uploaded_files)} reports…", expanded=True) as status:
            for i, f in enumerate(uploaded_files):
                status.write(f"Report {i + 1}/{len(uploaded_files)}: {f.name}…")
                try:
                    ordered[i] = run_with_status(f.getvalue(), f.name, status, label_prefix=f"[{f.name}] ")
                    status.write(f"✓ {f.name} done")
                except Exception as e:
                    ordered[i] = e
                    status.write(f"✗ {f.name} failed: {e}")
            status.update(label="Comparison ready", state="complete", expanded=False)

        ok_results = [r for r in ordered if isinstance(r, ESGExtraction)]
        for i, r in enumerate(ordered):
            if not isinstance(r, ESGExtraction):
                st.error(f"**{uploaded_files[i].name}** could not be analysed. {explain_error(r)}")

        if len(ok_results) >= 2:
            st.session_state["compare_ok_results"] = ok_results
            st.session_state["compare_filenames"] = [f.name for f in uploaded_files]
            render_compare_results(ok_results)
    elif (
        uploaded_files
        and st.session_state.get("compare_ok_results") is not None
        and st.session_state.get("compare_filenames") == [f.name for f in uploaded_files]
    ):
        # Re-shows the already-computed comparison on reruns triggered by
        # something else on the page (e.g. the copilot form above) — so
        # asking a question doesn't blank out results you already paid
        # several requests for, and doesn't need a fresh Compare click
        # (and more quota) to get them back.
        render_compare_results(st.session_state["compare_ok_results"])
    elif not compare_clicked:
        st.info("👆 Upload 2–5 PDFs and click **Compare reports** to begin.")
