import re
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

CSV_PATH = "ServiceCharge_Main.csv"
GOOGLE_FORM_URL = "https://forms.gle/jif98v7tmG5ioFnG8"
# --- Settings you can tweak ---
INFLATION_RATE = 0.03     # 3% (your requirement)
FORECAST_YEARS = 5
LOW_BAND = 0.90
HIGH_BAND = 1.10
MIN_N_SECTOR = 5
EXCLUDE_RELIABILITY = {"LOW"}  # excluded from "verified"
# ------------------------------

app = Flask(__name__)

POSTCODE_RE = re.compile(r"^\s*([A-Z]{1,2}\d{1,2}[A-Z]?)\s*\d[A-Z]{2}\s*$", re.I)

# --- Column mapping (your CSV headers) ---
c_postcode = "Postcode"
c_sector = "Sector"
c_reliability = "Coverage/Reliability"
c_annual = "Service Charge Annual Amount (£)"
c_monthly = "Service Charge Monthly Amount (£)"
c_psf = "£/sqft annualised"
c_psm = "£/sqm annualised"
c_end = "Service Charge Period End"
c_year = "Vintage Year"
# ----------------------------------------


def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"£{x:,.0f}"


def fmt_rate(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"£{x:,.2f}"


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1f}%"


def safe_median(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.median()) if len(s2) else None


def safe_count(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return int(len(s2))


def parse_district(postcode: str):
    pc = postcode.strip().upper().replace("  ", " ")
    m = POSTCODE_RE.match(pc)
    if not m:
        return None
    return m.group(1)


def derive_sector_from_postcode(postcode: str, district: str):
    pc = postcode.strip().upper().replace("  ", " ")
    inward_first_digit = re.search(r"\b(\d)[A-Z]{2}\b", pc)
    if not inward_first_digit:
        return None
    return f"{district} {inward_first_digit.group(1)}"


def verdict(sector_med, london_med):
    if sector_med is None or london_med is None:
        return "N/A"
    if sector_med <= london_med * LOW_BAND:
        return "Low"
    elif sector_med <= london_med * HIGH_BAND:
        return "Fair"
    else:
        return "High"


def forecast_year_value(base_yearly, years=FORECAST_YEARS):
    if base_yearly is None:
        return None
    return base_yearly * ((1 + INFLATION_RATE) ** years)


def forecast_rows(base_yearly):
    if base_yearly is None:
        return []
    rows = []
    for i in range(1, FORECAST_YEARS + 1):
        proj = base_yearly * ((1 + INFLATION_RATE) ** i)
        rows.append((i, proj))
    return rows


# --- Load data ---
df = pd.read_csv(CSV_PATH)

required_cols = [
    c_postcode, c_sector, c_reliability,
    c_annual, c_monthly, c_psf, c_psm,
    c_end, c_year
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(
        f"CSV columns missing: {missing}.\n\nYour CSV headers are:\n{df.columns.tolist()}"
    )

df["_rel_clean"] = df[c_reliability].astype(str).str.strip().str.upper()

# Year for trend: prefer Vintage Year; fallback to end-date year
df["_year"] = pd.to_numeric(df[c_year], errors="coerce")
if df["_year"].isna().all():
    end_dt = pd.to_datetime(df[c_end], errors="coerce")
    df["_year"] = end_dt.dt.year


def apply_verified_filter(g: pd.DataFrame):
    return g[~g["_rel_clean"].isin(EXCLUDE_RELIABILITY)].copy()


def compute_stats(g: pd.DataFrame):
    g2 = apply_verified_filter(g)
    return {
        "n": safe_count(g2[c_annual]),
        "median_year": safe_median(g2[c_annual]),
        "median_month": safe_median(g2[c_monthly]),
        "median_psf": safe_median(g2[c_psf]),
        "median_psm": safe_median(g2[c_psm]),
    }


def compute_trend_by_year(g: pd.DataFrame):
    """
    Returns list of dicts: [{year, median_year, n}, ...] using verified rows only.
    """
    g2 = apply_verified_filter(g).dropna(subset=["_year"])
    if g2.empty:
        return []
    out = []
    for yr in sorted(g2["_year"].dropna().unique()):
        gy = g2[g2["_year"] == yr]
        med = safe_median(gy[c_annual])
        n = safe_count(gy[c_annual])
        if med is not None:
            out.append({"year": int(yr), "median_year": med, "n": n})
    return out


def add_yoy(trend_list):
    prev = None
    out = []
    for row in trend_list:
        yoy = None
        if prev is not None and prev != 0:
            yoy = ((row["median_year"] - prev) / prev) * 100.0
        out.append({**row, "yoy": yoy})
        prev = row["median_year"]
    return out


# Pre-compute London stats + London trend (verified rows, across entire dataset)
london_stats_raw = compute_stats(df)
london_trend_raw = compute_trend_by_year(df)


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ServiceChargeCheck</title>
  <style>
    body { font-family: system-ui, -apple-system, Arial; max-width: 1020px; margin: 36px auto; padding: 0 16px; }
    .card { border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px; margin-top: 14px; }
    .muted { color: #6b7280; font-size: 14px; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
    .kpi { flex: 1; min-width: 220px; }
    input { padding: 10px 12px; font-size: 16px; width: 280px; }
    button { padding: 10px 12px; font-size: 16px; cursor: pointer; }
    .btn { display:inline-block; padding: 10px 12px; border: 1px solid #111827; border-radius: 10px; text-decoration:none; color:#111827; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; font-size: 14px; }
    .warn { color: #b45309; }
    .grid2 { display:grid; grid-template-columns: 1.35fr 1fr; gap:16px; align-items: start; }
    .center { text-align: center; }
    .title { font-size: 34px; margin: 0; }
    .subtitle { margin: 6px 0 0 0; }
    .postcode { font-size: 22px; font-weight: 700; margin: 4px 0 0 0; letter-spacing: 0.3px; }
    canvas { width:100%; height:320px; }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  </head>
  <footer style="margin-top:60px; padding:30px 0; text-align:center; font-size:14px; opacity:0.7;">
    <p>
        Built by Ajay Choli |
        <a href="/methodology">Methodology</a> |
        <a href="/disclaimer">Disclaimer</a>
    </p>
  </footer>
<body>

  <div class="center">
    <h1 class="title">ServiceChargeCheck</h1>
    <p class="muted subtitle">Benchmark service charges by postcode sector using verified entries.</p>
  </div>

  <div class="card">
    <form method="GET" action="/search" class="center">
      <input name="postcode" placeholder="e.g., E14 9AZ" value="{{ postcode or '' }}">
      <button type="submit">Search</button>
    </form>
  </div>

  {% if error %}
    <div class="card"><b>Error:</b> {{ error }}</div>
  {% endif %}

  {% if sector %}

    <div class="card center">
      <div class="muted">Results for</div>
      <div class="postcode">{{ postcode_display }}</div>
      <div class="muted" style="margin-top:10px;">
        <b>Sector:</b> {{ sector }} &nbsp;&nbsp; <b>District:</b> {{ district }}
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0;">Sector benchmark <span class="muted">(based on {{ sector_n }} verified entries)</span></h3>
      <div class="row">
        <div class="kpi"><b>Median £/year:</b> {{ sector_stats.median_year }}</div>
        <div class="kpi"><b>Median £/month:</b> {{ sector_stats.median_month }}</div>
        <div class="kpi"><b>Median £/sqft:</b> {{ sector_stats.median_psf }}</div>
        <div class="kpi"><b>Median £/sqm:</b> {{ sector_stats.median_psm }}</div>
      </div>
      {% if sector_n < min_n %}
        <div class="muted warn">Small sample size — interpret with caution.</div>
      {% endif %}
    </div>

    <div class="card">
      <h3 style="margin:0;">London benchmark <span class="muted">(based on {{ london_stats.n }} verified entries)</span></h3>
      <div class="row">
        <div class="kpi"><b>Median £/year:</b> {{ london_stats.median_year }}</div>
        <div class="kpi"><b>Median £/month:</b> {{ london_stats.median_month }}</div>
        <div class="kpi"><b>Median £/sqft:</b> {{ london_stats.median_psf }}</div>
        <div class="kpi"><b>Median £/sqm:</b> {{ london_stats.median_psm }}</div>
      </div>
    </div>

    <!-- Trend vs Verdict split -->
    <div class="card">
      <h3 style="margin:0 0 10px 0;">Benchmark trend vs London</h3>
      <div class="grid2">
        <div>
          <canvas id="benchChart"></canvas>
          <div class="muted" style="margin-top:8px;">
            Lines show median £/year by year for the sector and the London benchmark (verified entries only).
          </div>
        </div>
        <div class="card center">
        <div class="card" style="margin-top:0;">
          <h3 style="margin:0;">Verdict</h3>
          <div class="muted" style="margin-top:6px;">
            Uses a 10% tolerance band around the London benchmark.
          </div>
          <div style="font-size:22px; margin-top:10px;"><b>{{ verdict }}</b></div>
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0;">
        5-year forecast: {{ forecast_headline }}
      </h3>
      <div class="muted">Applies {{ inflation_rate }} inflation to the sector median £/year.</div>

      {% if forecast_rows %}
        <table>
          <thead><tr><th>Year</th><th>Projected £/year</th></tr></thead>
          <tbody>
            {% for yr, val in forecast_rows %}
              <tr><td>{{ yr }}</td><td>{{ val }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <div class="muted">Not enough data to forecast.</div>
      {% endif %}
    </div>

    <div class="card">
      <h3 style="margin:0;">Historic trend</h3>
      <div class="muted">Sector median annualised £/year by year, with YoY % change.</div>

      <div class="grid2" style="margin-top:10px;">
        <div>
          <canvas id="trendChart"></canvas>
        </div>
        <div>
          {% if trend %}
            <table>
              <thead><tr><th>Year</th><th>Median £/year</th><th>YoY %</th><th>(n)</th></tr></thead>
              <tbody>
                {% for t in trend %}
                  <tr>
                    <td>{{ t.year }}</td>
                    <td>{{ t.median_year }}</td>
                    <td>{{ t.yoy }}</td>
                    <td>{{ t.n }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          {% else %}
            <div class="muted">No trend available for this sector.</div>
          {% endif %}
        </div>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0;">Verify your own charge</h3>
      <p class="muted">Curious about your charge? Upload your latest invoice to see where you stand.</p>
      <a class="btn" href="/verify">Upload invoice</a>
      <p style="margin-top:16px;">
      <span style="opacity:0.8;">All submissions are marked <b>Unverified</b> until reviewed.</span>
      </p>
    </div>

    <script>
      // --- Benchmark chart (sector vs London) ---
      const benchYears = {{ bench_years_json | safe }};
      const benchSector = {{ bench_sector_json | safe }};
      const benchLondon = {{ bench_london_json | safe }};

      const benchCtx = document.getElementById('benchChart');
      if (benchCtx && benchYears.length) {
        new Chart(benchCtx, {
          type: 'line',
          data: {
            labels: benchYears,
            datasets: [
              { label: 'Sector median £/year', data: benchSector, tension: 0.25, spanGaps: true },
              { label: 'London median £/year', data: benchLondon, tension: 0.25, spanGaps: true }
            ]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: true } },
            scales: {
              y: { ticks: { callback: (v) => '£' + v.toLocaleString() } }
            }
          }
        });
      }

      // --- Historic chart (sector only) ---
      const years = {{ trend_years_json | safe }};
      const values = {{ trend_values_json | safe }};

      const ctx = document.getElementById('trendChart');
      if (ctx && years.length) {
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: years,
            datasets: [{ label: 'Sector median £/year', data: values, tension: 0.25, spanGaps: true }]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: true } },
            scales: { y: { ticks: { callback: (v) => '£' + v.toLocaleString() } } }
          }
        });
      }
    </script>

  {% endif %}
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML, postcode="", error=None, sector=None)

@app.route("/methodology")
def methodology():
    return """
    <div style="font-family:system-ui; max-width:900px; margin:40px auto; padding:0 20px; line-height:1.75;">
      <h1>Methodology</h1>

      <p><b>Data source:</b> Benchmarks are derived from service charge demands and historic billing records contained within the tracked dataset.
      User submissions are initially categorised as <b>Unverified</b> and then reviewed before being included in the benchmark calculations.</p>

      <p><b>Core metric:</b> The primary benchmark displayed is the annualised service charge (£/year).
      For comparability, the same figure is also expressed as £/month, and where floor area is available as £/sqft and £/sqm.</p>

      <p><b>Normalisation of billing periods:</b> Service charge demands are issued on varying billing cycles (e.g. half-yearly, quarterly, monthly, April–September, etc.).
      To ensure comparability, each invoice is normalised to a daily rate and then annualised as follows:</p>
      <ul>
        <li>Daily rate = Amount billed ÷ number of days in the billing period</li>
        <li>Annualised amount = Daily rate × 365</li>
      </ul>
      <p>This approach avoids doubling half-year invoices, which can misstate the true annual cost.</p>

      <p><b>Intensity metrics:</b> Where floor area is available:</p>
      <ul>
        <li>£/sqft = Annualised amount ÷ floor area (sqft)</li>
        <li>£/sqm = Annualised amount ÷ floor area (sqm)</li>
      </ul>

      <p>If area is in square feet, square metres are calculated using the following conversion (1 sqm = 10.7639 sqft).</p>

      <p><b>Aggregation approach:</b> Service charge datasets often include outliers (e.g. major works, adjustments to the reserve fund, balancing charges).
      Using the median provides a better representation of the typical charges within each sector.</p>

      <p><b>Geography Used:</b> The primary comparison is the postcode sector (e.g. E14 4).
      A London-wide benchmark, calculated from the full dataset, is displayed for context.</p>

      <p><b>Forecasting:</b> The five-year forecast is at 3% per annum, at time of publication.
      This projection is indicative and not predictive of any specific building’s future budget.</p>


      <p style="margin-top:40px;"><a href="/">← Back to home</a></p>
    </div>
    """

@app.route("/disclaimer")
def disclaimer():
    return """
    <div style="font-family:system-ui; max-width:900px; margin:40px auto; padding:0 20px; line-height:1.6;">
        <h1>Disclaimer</h1>

        <p>
        ServiceChargeCheck.com is an independent benchmarking tool for informational purposes only.
        </p>

        <p>
        While reasonable efforts are made to ensure data accuracy, no guarantee is provided regarding
        completeness, reliability, or suitability for any specific purpose.
        </p>

        <p>
        Benchmarks are based on available submitted data and may not represent the full market.
        Individual building characteristics, reserve funding, major works, and managing agent decisions
        may significantly affect service charge levels.
        </p>

        <p>
        This website does not provide legal, financial, or investment advice.
        Users should seek professional advice before making decisions based on this information.
        </p>

        <p>
        Submitted invoices should be redacted before upload. Personal data should not be shared.
        </p>

        <p style="margin-top:40px;">
        <a href="/">← Back to home</a>
        </p>
    </div>
    """
from flask import render_template_string

@app.route("/verify")
def verify():
    return render_template_string(
        """
        <div style="font-family:system-ui; max-width:720px; margin:40px auto; padding:0 16px;">
          <h1>Upload to verify</h1>

          <p>
            Upload your latest service charge invoice/demand to help improve the dataset.
            <b>All submissions are marked Unverified</b> until reviewed.
          </p>

          <div style="padding:14px 16px; border:1px solid rgba(0,0,0,0.12); border-radius:12px; margin:16px 0;">
            <p style="margin:0 0 10px 0;"><b>Before uploading:</b></p>
            <ul style="margin:0; padding-left:18px;">
              <li>Please redact your <b>name</b>, <b>flat number</b>, and any <b>account/reference numbers</b>.</li>
              <li>Keep: postcode, billing period dates, totals, and (if shown) floor area.</li>
            </ul>
          </div>

          <p>
            <a href="{{ google_form_url }}" target="_blank" rel="noopener"
               style="display:inline-block; padding:10px 14px; border-radius:10px; background:#111; color:#fff; text-decoration:none;">
              Open upload form
            </a>
          </p>

          <p style="margin-top:18px;"><a href="/">← Back</a></p>
        </div>
        """,
        google_form_url=GOOGLE_FORM_URL
    )

@app.route("/search")
def search():
    pc = (request.args.get("postcode") or "").strip()
    if not pc:
        return render_template_string(HTML, postcode="", error="Enter a postcode.", sector=None)

    district = parse_district(pc)
    if not district:
        return render_template_string(HTML, postcode=pc, error="Invalid postcode format (try e.g., E14 9AZ).", sector=None)

    sector = derive_sector_from_postcode(pc, district)
    if not sector:
        return render_template_string(HTML, postcode=pc, error="Could not derive sector from postcode.", sector=None)

    # Sector slice
    g_sector = df[df[c_sector].astype(str).str.strip().str.upper() == sector]

    # Stats
    sector_stats_raw = compute_stats(g_sector)
    sector_n = sector_stats_raw["n"]
    london_n = london_stats_raw["n"]

    v = verdict(sector_stats_raw["median_year"], london_stats_raw["median_year"])

    # Forecast headline (year-5 projected annual)
    year5 = forecast_year_value(sector_stats_raw["median_year"], FORECAST_YEARS)
    forecast_headline = fmt_money(year5)

    fc = forecast_rows(sector_stats_raw["median_year"])
    fc_fmt = [(yr, fmt_money(val)) for yr, val in fc]

    # Historic trend (sector)
    sector_trend_raw = add_yoy(compute_trend_by_year(g_sector))
    trend_fmt = []
    trend_years = []
    trend_values = []
    for row in sector_trend_raw:
        trend_years.append(int(row["year"]))
        trend_values.append(float(row["median_year"]))
        trend_fmt.append({
            "year": int(row["year"]),
            "median_year": fmt_money(row["median_year"]),
            "yoy": fmt_pct(row.get("yoy")),
            "n": row["n"],
        })

    # Benchmark chart trend (sector vs London) on the same year axis
    london_trend = {int(r["year"]): float(r["median_year"]) for r in london_trend_raw}
    sector_trend = {int(r["year"]): float(r["median_year"]) for r in compute_trend_by_year(g_sector)}

    all_years = sorted(set(london_trend.keys()) | set(sector_trend.keys()))
    bench_sector_vals = [sector_trend.get(y, None) for y in all_years]
    bench_london_vals = [london_trend.get(y, None) for y in all_years]

    # Format KPI cards
    sector_stats = {
        "median_year": fmt_money(sector_stats_raw["median_year"]),
        "median_month": fmt_money(sector_stats_raw["median_month"]),
        "median_psf": fmt_rate(sector_stats_raw["median_psf"]),
        "median_psm": fmt_rate(sector_stats_raw["median_psm"]),
    }
    london_stats = {
        "n": london_n,
        "median_year": fmt_money(london_stats_raw["median_year"]),
        "median_month": fmt_money(london_stats_raw["median_month"]),
        "median_psf": fmt_rate(london_stats_raw["median_psf"]),
        "median_psm": fmt_rate(london_stats_raw["median_psm"]),
    }

    # JSON for Chart.js injection
    bench_years_json = json.dumps(all_years)
    bench_sector_json = json.dumps(bench_sector_vals)
    bench_london_json = json.dumps(bench_london_vals)

    trend_years_json = json.dumps(trend_years)
    trend_values_json = json.dumps(trend_values)

    return render_template_string(
        HTML,
        postcode=pc,
        postcode_display=pc.strip().upper().replace("  ", " "),
        error=None,
        sector=sector,
        district=district,

        sector_stats=sector_stats,
        sector_n=sector_n,
        london_stats=london_stats,
        verdict=v,

        forecast_headline=forecast_headline,
        forecast_rows=fc_fmt,
        inflation_rate=f"{INFLATION_RATE:.1%}",

        trend=trend_fmt,

        bench_years_json=bench_years_json,
        bench_sector_json=bench_sector_json,
        bench_london_json=bench_london_json,

        trend_years_json=trend_years_json,
        trend_values_json=trend_values_json,

        min_n=MIN_N_SECTOR,
        low_band=LOW_BAND,
        high_band=HIGH_BAND,
        google_form_url=GOOGLE_FORM_URL
    )


if __name__ == "__main__":
    app.run(debug=True)
