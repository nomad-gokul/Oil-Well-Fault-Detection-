"""Live alert dashboard (Streamlit). Run: ``streamlit run src/owfd/dashboard/app.py``.

Talks only to the API (``OWFD_API_URL``; ``OWFD_API_KEY`` for acknowledging alerts and demo
controls), read from Streamlit secrets (Community Cloud) or else environment variables.
"""

from __future__ import annotations

import os
import sys
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
    ApiUnavailableError,
    Client,
    display_units,
    scenario_caption,
    scenario_source,
    well_status,
)

st.set_page_config(page_title="Oil Well Fault Detection", page_icon="🛢️", layout="wide")

COLORS = {"Fault": "#c0392b", "Warning": "#e67e22", "Normal": "#27ae60", "No data": "#7f8c8d"}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


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
    from (``GET /demo``). Empty when the API is unreachable (errors are not cached)."""
    try:
        return _demo_scenarios()
    except (ApiUnavailableError, httpx.HTTPStatusError):
        return {}


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
    st.sidebar.success(f"API online · model `{health['model']}` · v{health['model_version']}")
    rule = model["alert"]
    per_well = (
        " (raised per well after a one-hour burn-in)"
        if rule.get("burnin_quantile") is not None
        else ""
    )
    st.sidebar.caption(
        f"Alert when P(fault) > {rule['threshold']:.2f}{per_well} for "
        f"{int(rule['k'])} consecutive minutes"
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


@st.fragment(run_every=2)
def fleet(model: dict) -> None:
    try:
        wells = client().wells()
        active = client().alerts(status=None, limit=500)
    except ApiUnavailableError as exc:
        api_down(exc)
        return
    if not wells:
        st.info("No wells are streaming yet. Start the demo replay in the sidebar.")
        return
    threshold = model["alert"]["threshold"]
    unresolved = active[active["status"].isin(["open", "acknowledged"])] if len(active) else active
    chronic = (
        set(unresolved.loc[unresolved["source"] == "chronic", "well_id"])
        if len(unresolved) and "source" in unresolved
        else set()
    )
    scenarios = demo_scenarios()
    cols = st.columns(4)
    for i, well in enumerate(wells):
        preds = client().predictions(well, limit=1)
        p = float(preds["p_fault"].iloc[-1]) if len(preds) else None
        alert = unresolved[unresolved["well_id"] == well].head(1) if len(unresolved) else unresolved
        a = alert.iloc[0] if len(alert) else None
        status = well_status(p, threshold, a["status"] if a is not None else None)
        with cols[i % 4].container(border=True):
            st.markdown(
                f"**{well}**  \n<span style='color:{COLORS[status]};font-weight:700'>"
                f"● {status}</span>",
                unsafe_allow_html=True,
            )
            if well in scenarios:  # demo replay: the real source well and its true condition
                st.caption(f"🛢️ {scenario_caption(scenarios[well])}")
            if well in chronic:
                st.caption("〰️ Chronic instability (well profile)")
            if a is not None:
                st.caption(
                    f"{SEVERITY_ICON.get(a['severity'], '')} Alarm: {a['class_name']} · "
                    f"peak P={a['peak_probability']:.2f}"
                )
            elif len(preds):
                cls = model["classes"][str(int(preds["predicted_class"].iloc[-1]))]
                st.caption(f"Model: {cls} · P(fault)={p:.2f}")
            if p is not None:
                st.progress(min(max(p, 0.0), 1.0))


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
        chronic = a.get("source") == "chronic"
        c[0].markdown(
            f"{SEVERITY_ICON.get(a['severity'], '')} **{a['well_id']}** — "
            f"{a['class_name']}{' (chronic, well profile)' if chronic else ''} · "
            f"opened {a['opened_at']:%Y-%m-%d %H:%M} · peak P={a['peak_probability']:.2f}"
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
    st.title("Oil Well Fault Detection — live alerts")
    if model is None:
        st.stop()
    tabs = st.tabs(["Fleet", "Well detail", "Alerts", "Model", "System"])
    with tabs[0]:
        fleet(model)
    with tabs[1]:
        try:
            wells = client().wells()
        except ApiUnavailableError:
            wells = []
        if wells:
            well_detail(model, st.selectbox("Well", wells))
        else:
            st.info("No wells yet.")
    with tabs[2]:
        alerts_tab()
    with tabs[3]:
        model_tab(model)
    with tabs[4]:
        system_tab()


main()
