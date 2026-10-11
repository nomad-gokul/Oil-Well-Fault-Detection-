"""Live alert dashboard (Streamlit). Run: ``streamlit run src/owfd/dashboard/app.py``.

Talks only to the API (``OWFD_API_URL``; ``OWFD_API_KEY`` for acknowledging alerts and demo
controls), read from Streamlit secrets (Community Cloud) or else environment variables.
"""

from __future__ import annotations

import html
import os
import sys
from collections import Counter
from pathlib import Path

try:  # running from a plain checkout (e.g. Streamlit Community Cloud): make src/ importable
    import owfd  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from owfd.dashboard.client import (  # noqa: E402
    LEVELS,
    ApiUnavailableError,
    Client,
    display_units,
    fault_text,
    plain_status,
    scenario_caption,
    scenario_source,
)

st.set_page_config(page_title="Oil Well Fault Detection", page_icon="🛢️", layout="wide")

SEVERITY_LEVEL = {"high": "critical", "medium": "warning", "low": "watch"}


def setting(name: str) -> str | None:
    """A Streamlit secret (Community Cloud), else the environment variable (Docker, local)."""
    try:
        value = st.secrets.get(name)
    except FileNotFoundError:  # no secrets.toml
        value = None
    return str(value) if value else os.environ.get(name)


@st.cache_resource
def client() -> Client:
    return Client(setting("OWFD_API_URL"), setting("OWFD_API_KEY"))


@st.cache_data(ttl=600, show_spinner=False)
def _demo_scenarios() -> dict[str, dict]:
    return dict(client().demo().get("scenarios") or {})


def demo_scenarios() -> dict[str, dict]:
    """What each bundled demo well replays: true fault class and the real 3W well it comes
    from (``GET /demo``). Empty when the API is unreachable or does not report it yet; errors
    and empty answers are not cached, so a redeployed API is picked up on the next refresh."""
    try:
        scenarios = _demo_scenarios()
    except (ApiUnavailableError, httpx.HTTPStatusError):
        return {}
    if not scenarios:
        _demo_scenarios.clear()
    return scenarios


def api_down(exc: Exception) -> None:
    st.warning(
        f"API unavailable at `{client().base_url}` — it may be waking up (free hosting tiers "
        f"sleep when idle). This page retries automatically. ({exc})",
        icon="⏳",
    )


def sidebar() -> dict | None:
    st.sidebar.title("🛢️ Well monitor")
    try:
        health, model = client().health(), client().model()
    except ApiUnavailableError as exc:
        api_down(exc)
        return None
    st.sidebar.success("System online")
    rule = model["alert"]
    per_well = (
        " (raised per well after a one-hour burn-in)"
        if rule.get("burnin_quantile") is not None
        else ""
    )
    with st.sidebar.expander("Technical details"):
        st.caption(
            f"Model `{health['model']}` · v{health['model_version']}. Alert when "
            f"P(fault) > {rule['threshold']:.2f}{per_well} for {int(rule['k'])} consecutive "
            "minutes."
        )
    st.sidebar.divider()
    st.sidebar.subheader("Demo replay")
    st.sidebar.caption("Replays held-out real wells from the 3W test set, sped up.")
    try:
        demo = client().demo()
    except ApiUnavailableError:
        demo = {"running": False}
    speed = st.sidebar.select_slider("Speed (x real time)", [10, 30, 60, 120, 300], value=60)
    names = model["classes"]
    chosen = st.sidebar.multiselect(
        "Scenarios (empty = all)",
        [int(c) for c in names],
        format_func=lambda c: f"{c} · {names[str(c)]}",
    )
    cols = st.sidebar.columns(2)
    try:
        if cols[0].button("▶ Start", width="stretch"):
            client().demo_start(speed, chosen or None)
            st.rerun()
        if cols[1].button("■ Stop", width="stretch", disabled=not demo.get("running")):
            client().demo_stop()
            st.rerun()
    except httpx.HTTPStatusError as exc:
        st.sidebar.error(
            "Demo controls need OWFD_API_KEY" if exc.response.status_code == 401 else str(exc)
        )
    st.sidebar.caption("● running" if demo.get("running") else "○ stopped")
    return model


def wells_text(n: int) -> str:
    """``"1 well needs"`` / ``"3 wells need"``."""
    return "1 well needs" if n == 1 else f"{n} wells need"


def card(well: str, state: dict, note: str = "") -> str:
    """One well's status as an HTML card: level badge, what is happening, what it means and
    what to do. Colours are translucent so the card reads in light and dark themes."""
    label, _, color, _ = LEVELS[state["level"]]
    esc = html.escape
    since = [f"for {state['duration']}"] if state["duration"] else []
    if state["acknowledged"]:
        since.append("seen by an operator")
    if state["more"]:
        since.append(f"+{state['more']} more issue{'s' if state['more'] > 1 else ''} (see Alerts)")
    lines = [
        "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"
        f"<span style='font-weight:700;font-size:1.05rem'>{esc(well)}</span>"
        f"<span style='background:{color};color:#fff;border-radius:999px;padding:2px 10px;"
        f"font-size:.78rem;font-weight:700;letter-spacing:.04em'>{label.upper()}</span></div>",
        f"<div style='font-size:1.12rem;font-weight:600;margin-top:6px'>"
        f"{esc(state['headline'])}</div>",
    ]
    if since:
        lines.append(f"<div style='opacity:.75;font-size:.85rem'>{esc(' · '.join(since))}</div>")
    if state["level"] not in ("normal", "no_data"):
        lines.append(f"<div style='margin-top:6px;font-size:.92rem'>{esc(state['meaning'])}</div>")
    if state["action"]:
        lines.append(
            f"<div style='margin-top:6px;font-size:.92rem'><b>What to do:</b> "
            f"{esc(state['action'])}</div>"
        )
    if note:
        lines.append(f"<div style='opacity:.6;font-size:.75rem;margin-top:8px'>{esc(note)}</div>")
    return (
        f"<div style='border-left:6px solid {color};background:{color}1a;border-radius:8px;"
        f"padding:12px 14px;margin-bottom:12px'>{''.join(lines)}</div>"
    )


@st.fragment(run_every=3)
def overview(model: dict) -> None:
    """Every well's current state in plain words, worst first."""
    try:
        wells = client().wells()
        alerts = client().alerts(limit=500)
        latest = {well: client().predictions(well, limit=1) for well in wells}
    except ApiUnavailableError as exc:
        api_down(exc)
        return
    if not wells:
        st.info("No wells are reporting yet. Start the demo replay in the sidebar.")
        return
    threshold = model["alert"]["threshold"]
    records = alerts.to_dict("records") if len(alerts) else []
    states = []
    for well in wells:
        preds = latest[well]
        p = float(preds["p_fault"].iloc[-1]) if len(preds) else None
        now = preds["timestamp"].iloc[-1] if len(preds) else None
        mine = [a for a in records if a["well_id"] == well]
        states.append((well, plain_status(mine, p, threshold, now)))
    order = list(LEVELS)
    states.sort(key=lambda ws: (order.index(ws[1]["level"]), ws[0]))

    counts = Counter(state["level"] for _, state in states)
    urgent = counts["critical"] + counts["warning"]
    if counts["critical"]:
        headline = f"{wells_text(counts['critical'])} action now"
    elif urgent:
        headline = f"{wells_text(urgent)} checking today"
    elif counts["watch"]:
        headline = "No serious faults. Some wells to keep an eye on"
    else:
        headline = "All wells are working normally"
    st.subheader(headline)
    shown = [lvl for lvl in order if lvl != "no_data" or counts[lvl]]
    for col, lvl in zip(st.columns(len(shown)), shown, strict=True):
        label, icon, _, _ = LEVELS[lvl]
        col.metric(f"{icon} {label}", counts[lvl])

    scenarios = demo_scenarios()
    for start in range(0, len(states), 3):
        cols = st.columns(3)
        for col, (well, state) in zip(cols, states[start : start + 3], strict=False):
            note = scenario_caption(scenarios[well]) if well in scenarios else ""
            col.markdown(
                card(well, state, f"Demo · {note}" if note else ""), unsafe_allow_html=True
            )

    with st.expander("What do the levels mean?"):
        st.markdown(
            "\n".join(
                f"- {icon} **{label}**: {meaning}" for label, icon, _, meaning in LEVELS.values()
            )
        )
        st.caption(
            "A well's level comes from its most serious open alert. Alerts are raised "
            "automatically from the well's sensors (pressures, temperatures, flows) every minute."
        )


@st.fragment(run_every=3)
def well_detail(model: dict, well: str) -> None:
    try:
        preds = client().predictions(well, limit=180)
        readings = client().readings(well, minutes=120, every_s=10)
        alerts = client().alerts(well_id=well, limit=50)
    except ApiUnavailableError as exc:
        api_down(exc)
        return
    info = demo_scenarios().get(well)
    if info:
        source = scenario_source(info)  # e.g. "WELL-00004, Aug 2014"
        st.info(
            f"Demo replay of a held-out recording from the Petrobras 3W test set"
            f"{f' (real well and month recorded: **{source}**)' if source else ''}. Actual "
            f"condition in this recording: **{info['class_name']}**. Compare it with what the "
            f"model detects below.",
            icon="🛢️",
        )
    if preds.empty:
        st.info("No predictions yet for this well.")
        return
    state = plain_status(
        alerts.to_dict("records") if len(alerts) else [],
        float(preds["p_fault"].iloc[-1]),
        model["alert"]["threshold"],
        preds["timestamp"].iloc[-1],
    )
    st.markdown(card(well, state), unsafe_allow_html=True)
    sensors, units = display_units(readings)
    shown = [c for c in sensors.columns if sensors[c].notna().sum() > 1]
    rows = len(shown) + 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[1.0] * len(shown) + [1.6],
        subplot_titles=[f"{c} [{units[c]}]" for c in shown] + ["P(fault) and alert threshold"],
    )
    for i, col in enumerate(shown, start=1):
        fig.add_trace(
            go.Scatter(
                x=sensors.index,
                y=sensors[col],
                name=col,
                mode="lines",
                line={"width": 1.2},
                showlegend=False,
            ),
            row=i,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=preds["timestamp"],
            y=preds["p_fault"],
            name="P(fault)",
            mode="lines",
            line={"color": "#c0392b", "width": 2},
            showlegend=False,
        ),
        row=rows,
        col=1,
    )
    fig.add_hline(
        y=model["alert"]["threshold"], line_dash="dash", line_color="grey", row=rows, col=1
    )
    # Show only the recent window; clip alert shading to it (old episodes are in the Alerts tab).
    t_min, t_max = preds["timestamp"].min(), preds["timestamp"].max()
    for _, a in alerts.iterrows():
        end = a["resolved_at"] if pd.notna(a["resolved_at"]) else t_max
        if end < t_min or a["opened_at"] > t_max:
            continue
        fig.add_vrect(
            x0=max(a["opened_at"], t_min),
            x1=min(end, t_max),
            fillcolor="#c0392b",
            opacity=0.08,
            line_width=0,
        )
    fig.update_xaxes(range=[t_min, t_max])
    fig.update_yaxes(range=[0, 1], row=rows, col=1)
    fig.update_annotations(font_size=11)
    fig.update_layout(height=130 * rows + 120, margin={"t": 30, "b": 10, "l": 10, "r": 10})
    st.caption("Shaded: alert periods. Sensors in engineering units; pressures in bar.")
    st.plotly_chart(fig, width="stretch")
    latest = preds.iloc[-1]
    probs = pd.Series(latest["probabilities"], index=list(model["classes"].values()))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latest class probabilities**")
        st.bar_chart(probs.sort_values(ascending=False).head(5), horizontal=True, height=200)
    with c2:
        st.markdown("**Why the latest alert fired** (top SHAP drivers)")
        if len(alerts) and isinstance(alerts.iloc[0].get("top_drivers"), list):
            drivers = pd.DataFrame(alerts.iloc[0]["top_drivers"])
            st.dataframe(drivers.round(3), hide_index=True, width="stretch")
        else:
            st.caption("No alert with explanations for this well yet.")


def alerts_tab() -> None:
    try:
        df = client().alerts(limit=1000)
    except ApiUnavailableError as exc:
        api_down(exc)
        return
    if df.empty:
        st.info("No alerts yet.")
        return
    open_ = df[df["status"] == "open"]
    st.subheader(f"Open alerts ({len(open_)})")
    for _, a in open_.iterrows():
        c = st.columns([5, 1, 1, 1])
        label, icon, _, _ = LEVELS[SEVERITY_LEVEL.get(a["severity"], "warning")]
        headline, _, action = fault_text(int(a["event_class"]), a["class_name"], a.get("source"))
        c[0].markdown(
            f"{icon} **{a['well_id']}**: {headline} ({label.lower()}) · since "
            f"{a['opened_at']:%Y-%m-%d %H:%M}  \n{action}"
        )
        if c[1].button("Acknowledge", key=f"ack{a['id']}"):
            try:
                client().acknowledge(int(a["id"]))
                st.rerun()
            except httpx.HTTPStatusError:
                st.error("Acknowledging needs OWFD_API_KEY")
        # Operator verdicts become labels for the next well-profile build.
        for col, (label, text) in zip(
            c[2:], (("confirmed", "Confirm"), ("false_alarm", "False alarm")), strict=True
        ):
            if col.button(text, key=f"{label}{a['id']}"):
                try:
                    client().feedback(int(a["id"]), label)
                    st.toast(f"Recorded: {text.lower()} for alert {a['id']}")
                except httpx.HTTPStatusError:
                    st.error("Feedback needs OWFD_API_KEY")
    st.subheader("History")
    f1, f2 = st.columns(2)
    status = f1.multiselect(
        "Status", ["open", "acknowledged", "resolved"], default=["open", "acknowledged", "resolved"]
    )
    wells = f2.multiselect("Wells", sorted(df["well_id"].unique()))
    view = df[df["status"].isin(status)]
    if wells:
        view = view[view["well_id"].isin(wells)]
    cols = [
        "id",
        "well_id",
        "class_name",
        "severity",
        "status",
        "opened_at",
        "acknowledged_at",
        "resolved_at",
        "peak_probability",
    ]
    st.dataframe(view[cols], hide_index=True, width="stretch")
    st.download_button("Download CSV", view[cols].to_csv(index=False), "alerts.csv", "text/csv")


def model_tab(model: dict) -> None:
    st.subheader(f"Model `{model['model']}` · bundle {model['version']}")
    st.caption(
        f"Window {model['window_s']} s every {model['stride_s']} s · sensors: "
        + ", ".join(model["variables"])
    )
    m = model.get("metrics") or {}
    if m:
        c = st.columns(4)
        c[0].metric("Test macro F1", f"{m.get('test_macro_f1', np.nan):.3f}")
        c[1].metric("Faults detected (test)", f"{m.get('test_detection_rate', np.nan):.0%}")
        c[2].metric("Median lead time", f"{m.get('test_median_lead_min', np.nan):.0f} min")
        c[3].metric("False alarms / day", f"{m.get('test_false_alarms_per_day', np.nan):.2f}")
        st.caption("Evaluated on held-out wells never seen in training (Petrobras 3W).")
    st.json(m, expanded=False)


@st.fragment(run_every=10)
def system_tab() -> None:
    try:
        health = client().health()
        drift = client().drift()
    except ApiUnavailableError as exc:
        api_down(exc)
        return
    c = st.columns(3)
    c[0].metric("API status", health["status"].upper())
    c[1].metric("Uptime", f"{health['uptime_s'] / 3600:.1f} h")
    c[2].metric("Model", health["model_version"])
    st.subheader("Fleet data drift")
    if drift is None:
        st.caption("This model bundle has no reference for drift checks.")
        return
    if drift["drift"]:
        bad = [k for k, v in drift["sensors"].items() if v["status"] == "drift"]
        st.error(f"Fleet-wide drift on: {', '.join(bad)}")
    else:
        st.success("No fleet-wide drift")
    rows = [
        {
            "sensor": k,
            "status": v["status"],
            "wells checked": v["wells_checked"],
            "wells unusual": v["wells_unusual"],
            "share unusual": v["share_unusual"],
        }
        for k, v in drift["sensors"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Per well and sensor, the last 2 h of deviations from the well's own baseline are "
        "compared with the range seen on real training/validation wells (median offset, "
        "spread). Drift = at least half of the wells (>= 3) unusual on the same sensor. "
        "Prometheus metrics: /metrics."
    )


def main() -> None:
    model = sidebar()
    st.title("Oil well monitor")
    st.caption("Which wells have a problem right now, how serious it is, and what to do.")
    if model is None:
        st.stop()
    tabs = st.tabs(["Overview", "Alerts", "Sensor charts", "Model", "System"])
    with tabs[0]:
        overview(model)
    with tabs[1]:
        alerts_tab()
    with tabs[2]:
        try:
            wells = client().wells()
        except ApiUnavailableError:
            wells = []
        if wells:
            well_detail(model, st.selectbox("Well", wells))
        else:
            st.info("No wells yet.")
    with tabs[3]:
        model_tab(model)
    with tabs[4]:
        system_tab()


main()
