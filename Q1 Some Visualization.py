# =============================================================================
# NHG Vehicle Routing — Q1 All Visualizations
# Viz 1 : Waterfall Chart        → Q1_Viz1_Waterfall.png
# Viz 2 : Algorithm Table        → Q1_Viz2_AlgoTable.png
# Viz 3a: Static Route Map       → Q1_Viz3a_RouteMap.png
# Viz 3b: Interactive Route Map  → Q1_Viz3b_RouteMap.html
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MiniMap, Fullscreen
from itertools import combinations
import colorsys
import time
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

raw_orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
raw_locs   = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
raw_dist   = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)

# Orders
orders_df            = raw_orders[raw_orders["ORDERID"] != 0].copy()
orders_df["CUBE"]    = orders_df["CUBE"].astype(int)
orders_df["TOZIP"]   = orders_df["TOZIP"].astype(str).str.zfill(5)
orders_df["FROMZIP"] = orders_df["FROMZIP"].astype(str).str.zfill(5)

# Locations
locs_df            = raw_locs.dropna(subset=["ZIPID"]).copy()
locs_df["ZIPID"]   = locs_df["ZIPID"].astype(int)
locs_df["ZIP_STR"] = locs_df["ZIP"].astype(int).astype(str).str.zfill(5)
locs_df            = locs_df.rename(columns={"Y": "LAT", "X": "LON"})

# Lookups
zip_to_id      = dict(zip(locs_df["ZIP_STR"], locs_df["ZIPID"]))
zipid_to_coord = {int(r["ZIPID"]): (float(r["LAT"]), float(r["LON"]))
                  for _, r in locs_df.iterrows()}
zipid_to_city  = {int(r["ZIPID"]): str(r["CITY"])    for _, r in locs_df.iterrows()}
zipid_to_state = {int(r["ZIPID"]): str(r["STATE"])   for _, r in locs_df.iterrows()}
zipid_to_zip   = {int(r["ZIPID"]): str(r["ZIP_STR"]) for _, r in locs_df.iterrows()}

DEPOT_ZIP = "01887"
DEPOT_ID  = int(zip_to_id[DEPOT_ZIP])

assert not (set(orders_df["TOZIP"]) - set(zip_to_id)), "Unresolved TOZIPs"

# Distance matrix — int() keys fix Windows numpy.int64 mismatch
col_ids  = raw_dist.iloc[1, 2:].astype(int).tolist()
row_ids  = raw_dist.iloc[2:, 1].astype(int).tolist()
dist_arr = raw_dist.iloc[2:, 2:].astype(float).values
assert col_ids == row_ids, "Distance matrix row/col mismatch"
_idx     = {int(zipid): i for i, zipid in enumerate(col_ids)}

def get_dist(a: int, b: int) -> float:
    return dist_arr[_idx[int(a)], _idx[int(b)]]

# Master orders dict
orders = {
    int(r["ORDERID"]): {
        "id":     int(r["ORDERID"]),
        "cube":   int(r["CUBE"]),
        "day":    r["DayOfWeek"],
        "zip_id": int(zip_to_id[r["TOZIP"]]),
    }
    for _, r in orders_df.iterrows()
}

DAYS          = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {d: [oid for oid, o in orders.items() if o["day"] == d]
                 for d in DAYS}

# Safety check
missing = {o["zip_id"] for o in orders.values()} - set(_idx.keys())
assert not missing, f"ZipIDs missing from distance matrix: {missing}"


# =============================================================================
# STEP 2 — CONSTANTS
# =============================================================================

VAN_CAP        = 3200
SPEED_MPH      = 40.0
UNLOAD_RATE    = 0.030
MIN_UNLOAD     = 30.0
MAX_DRIVE_MIN  = 660
MAX_DUTY_MIN   = 840
BREAK_MIN      = 600
WINDOW_OPEN    = 480
WINDOW_CLOSE   = 1080
WEEKS_PER_YEAR = 52


# =============================================================================
# STEP 3 — FEASIBILITY CHECKER
# =============================================================================

def _drive_time(a: int, b: int) -> float:
    return (get_dist(a, b) / SPEED_MPH) * 60.0

def _unload_time(cube: int) -> float:
    return max(MIN_UNLOAD, UNLOAD_RATE * cube)

def simulate_route(order_ids: list, allow_overnight: bool = True) -> dict:
    if not order_ids:
        return {"feasible": True, "total_miles": 0.0, "drive_min": 0.0,
                "duty_min": 0.0, "overnight": False, "end_time": WINDOW_OPEN}
    first_zip     = orders[order_ids[0]]["zip_id"]
    dispatch_time = WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)
    clock = dispatch_time; loc = DEPOT_ID
    miles = drive_acc = duty_acc = 0.0; overnight = False
    for oid in order_ids:
        o = orders[oid]; dest = o["zip_id"]
        leg_mi = get_dist(loc, dest); leg_min = _drive_time(loc, dest)
        ul_min = _unload_time(o["cube"])
        d_left = MAX_DRIVE_MIN - drive_acc; dy_left = MAX_DUTY_MIN - duty_acc
        can_drive = leg_min <= d_left and leg_min <= dy_left
        if not can_drive:
            if not allow_overnight or overnight:
                return {"feasible": False, "total_miles": miles, "overnight": overnight}
            drv_min = min(d_left, dy_left); drv_mi = (drv_min / 60.0) * SPEED_MPH
            clock  += drv_min + BREAK_MIN; miles += drv_mi
            drive_acc = duty_acc = 0.0; overnight = True
            if clock < WINDOW_OPEN + 24 * 60: clock = WINDOW_OPEN + 24 * 60
            rem_mi = leg_mi - drv_mi; rem_min = (rem_mi / SPEED_MPH) * 60.0
            clock += rem_min; drive_acc += rem_min; duty_acc += rem_min
            miles += rem_mi; loc = dest
            if clock > WINDOW_CLOSE + 24 * 60:
                return {"feasible": False, "total_miles": miles, "overnight": overnight}
            clock += ul_min; duty_acc += ul_min; continue
        clock += leg_min; drive_acc += leg_min; duty_acc += leg_min
        miles += leg_mi; loc = dest
        if clock > WINDOW_CLOSE + (24 * 60 if overnight else 0):
            return {"feasible": False, "total_miles": miles, "overnight": overnight}
        clock += ul_min; duty_acc += ul_min
    ret_mi = get_dist(loc, DEPOT_ID); ret_min = _drive_time(loc, DEPOT_ID)
    if drive_acc + ret_min > MAX_DRIVE_MIN or duty_acc + ret_min > MAX_DUTY_MIN:
        if not overnight and allow_overnight:
            clock += BREAK_MIN; drive_acc = duty_acc = 0.0; overnight = True
            if clock < WINDOW_OPEN + 24 * 60: clock = WINDOW_OPEN + 24 * 60
        elif overnight:
            return {"feasible": False, "total_miles": miles, "overnight": overnight}
    clock += ret_min; drive_acc += ret_min; duty_acc += ret_min; miles += ret_mi
    return {"feasible": True, "total_miles": miles, "drive_min": drive_acc,
            "duty_min": duty_acc, "overnight": overnight, "end_time": clock}


# =============================================================================
# STEP 4 — OBJECTIVE FUNCTION
# =============================================================================

def route_miles(route: list) -> float:
    return simulate_route(route)["total_miles"]

def total_miles(routes: list) -> float:
    return sum(route_miles(r) for r in routes)


# =============================================================================
# STEP 5 — ALGORITHMS
# =============================================================================

def nearest_neighbor(day: str) -> list:
    unserved = list(orders_by_day[day]); routes = []
    while unserved:
        route = []; cube = 0; loc = DEPOT_ID
        while True:
            best_oid = best_d = None
            for oid in unserved:
                o = orders[oid]
                if cube + o["cube"] > VAN_CAP: continue
                if not simulate_route(route + [oid])["feasible"]: continue
                d = get_dist(loc, o["zip_id"])
                if best_d is None or d < best_d: best_d, best_oid = d, oid
            if best_oid is None: break
            route.append(best_oid); cube += orders[best_oid]["cube"]
            loc = orders[best_oid]["zip_id"]; unserved.remove(best_oid)
        routes.append(route) if route else routes.append([unserved.pop(0)])
    return routes

def clarke_wright(day: str) -> list:
    day_oids   = list(orders_by_day[day])
    routes     = [[oid] for oid in day_oids]
    route_cube = [orders[oid]["cube"] for oid in day_oids]
    route_of   = {oid: i for i, oid in enumerate(day_oids)}
    savings = []
    for i, j in combinations(day_oids, 2):
        s = (get_dist(DEPOT_ID, orders[i]["zip_id"]) +
             get_dist(DEPOT_ID, orders[j]["zip_id"]) -
             get_dist(orders[i]["zip_id"], orders[j]["zip_id"]))
        savings.append((s, i, j)); savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])
    for s_val, i_oid, j_oid in savings:
        if s_val <= 0: break
        ri, rj = route_of[i_oid], route_of[j_oid]
        if ri == rj or routes[ri] is None or routes[rj] is None: continue
        if routes[ri][-1] != i_oid or routes[rj][0] != j_oid: continue
        if route_cube[ri] + route_cube[rj] > VAN_CAP: continue
        merged = routes[ri] + routes[rj]
        if not simulate_route(merged)["feasible"]: continue
        routes[ri] = merged; route_cube[ri] += route_cube[rj]; routes[rj] = None
        for oid in routes[ri]: route_of[oid] = ri
    return [r for r in routes if r is not None]

def two_opt(route: list) -> list:
    if len(route) < 3: return route
    best = route[:]; best_mi = route_miles(best); improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                res  = simulate_route(cand)
                if res["feasible"] and res["total_miles"] < best_mi - 0.01:
                    best, best_mi, improved = cand, res["total_miles"], True; break
            if improved: break
    return best

def or_opt(routes: list) -> list:
    routes = [r[:] for r in routes]; improved = True
    while improved:
        improved = False
        for r1 in range(len(routes)):
            if not routes[r1]: continue
            for pos in range(len(routes[r1])):
                oid    = routes[r1][pos]
                new_r1 = routes[r1][:pos] + routes[r1][pos+1:]
                for r2 in range(len(routes)):
                    if not routes[r2]: continue
                    cube_r2 = sum(orders[o]["cube"] for o in routes[r2])
                    if r1 != r2 and cube_r2 + orders[oid]["cube"] > VAN_CAP: continue
                    before  = route_miles(routes[r1]) + (route_miles(routes[r2]) if r1 != r2 else 0)
                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = new_r1[:ins] + [oid] + new_r1[ins:]
                            res  = simulate_route(cand)
                            if res["feasible"] and res["total_miles"] < route_miles(routes[r1]) - 0.01:
                                routes[r1] = cand; improved = True; break
                        else:
                            new_r2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            res1 = simulate_route(new_r1); res2 = simulate_route(new_r2)
                            if res1["feasible"] and res2["feasible"]:
                                if res1["total_miles"] + res2["total_miles"] < before - 0.01:
                                    routes[r1] = new_r1; routes[r2] = new_r2
                                    improved = True; break
                    if improved: break
                if improved: break
            if improved: break
    return [r for r in routes if r]

def three_opt(route: list) -> list:
    if len(route) < 5: return two_opt(route)
    best = route[:]; best_mi = route_miles(best); improved = True
    while improved:
        improved = False; n = len(best)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A, B = best[:i+1], best[i+1:j+1]
                    C, D = best[j+1:k+1], best[k+1:]
                    for cand in [A+B+C[::-1]+D, A+B[::-1]+C+D,
                                 A+B[::-1]+C[::-1]+D, A+C+B+D,
                                 A+C+B[::-1]+D, A+C[::-1]+B+D,
                                 A+C[::-1]+B[::-1]+D]:
                        res = simulate_route(cand)
                        if res["feasible"] and res["total_miles"] < best_mi - 0.01:
                            best, best_mi, improved = cand, res["total_miles"], True; break
                    if improved: break
                if improved: break
            if improved: break
    return best

def solve_day(day: str) -> dict:
    """Run all 7 algorithm combinations, return best + full results dict."""
    nn     = nearest_neighbor(day)
    cw     = clarke_wright(day)
    nn_2   = [two_opt(r) for r in nn]
    cw_2   = [two_opt(r) for r in cw]
    nn_opt = or_opt(nn_2)
    cw_opt = or_opt(cw_2)
    cw_3   = [three_opt(r) for r in cw]

    results = {
        "NN":             {"routes": nn,     "miles": total_miles(nn)},
        "CW":             {"routes": cw,     "miles": total_miles(cw)},
        "NN+2Opt":        {"routes": nn_2,   "miles": total_miles(nn_2)},
        "CW+2Opt":        {"routes": cw_2,   "miles": total_miles(cw_2)},
        "NN+2Opt+OrOpt":  {"routes": nn_opt, "miles": total_miles(nn_opt)},
        "CW+2Opt+OrOpt":  {"routes": cw_opt, "miles": total_miles(cw_opt)},
        "CW+3Opt":        {"routes": cw_3,   "miles": total_miles(cw_3)},
    }
    best_key    = min(results, key=lambda k: results[k]["miles"])
    return {"results": results, "best_key": best_key,
            "best_routes": results[best_key]["routes"],
            "best_miles":  results[best_key]["miles"]}


# =============================================================================
# STEP 6 — COMPUTE ALL ROUTES
# =============================================================================

print("=" * 60)
print("  Computing Q1 routes for all days...")
print("=" * 60)

all_day_results = {}
all_best_routes = {}
weekly_miles_q1 = 0.0

for day in DAYS:
    print(f"  {day}...", end=" ", flush=True)
    t0  = time.time()
    res = solve_day(day)
    all_day_results[day] = res
    all_best_routes[day] = res["best_routes"]
    weekly_miles_q1     += res["best_miles"]
    print(f"{res['best_miles']:.0f} mi  [{res['best_key']}]  ({time.time()-t0:.1f}s)")

annual_miles_q1 = weekly_miles_q1 * WEEKS_PER_YEAR
total_routes    = sum(len(v) for v in all_best_routes.values())

# BC1 — point to point
weekly_miles_bc1 = sum(2 * get_dist(DEPOT_ID, o["zip_id"]) for o in orders.values())
annual_miles_bc1 = weekly_miles_bc1 * WEEKS_PER_YEAR

# BC2 — no overnight (CW+2Opt)
print("  BC2 (no overnight)...", end=" ", flush=True)
weekly_miles_bc2 = 0.0
for day in DAYS:
    day_oids = list(orders_by_day[day])
    rts = [[oid] for oid in day_oids]; rc = [orders[oid]["cube"] for oid in day_oids]
    ro  = {oid: i for i, oid in enumerate(day_oids)}
    savings = []
    for i, j in combinations(day_oids, 2):
        s = (get_dist(DEPOT_ID, orders[i]["zip_id"]) +
             get_dist(DEPOT_ID, orders[j]["zip_id"]) -
             get_dist(orders[i]["zip_id"], orders[j]["zip_id"]))
        savings.append((s, i, j)); savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])
    for s_val, i_oid, j_oid in savings:
        if s_val <= 0: break
        ri, rj = ro[i_oid], ro[j_oid]
        if ri == rj or rts[ri] is None or rts[rj] is None: continue
        if rts[ri][-1] != i_oid or rts[rj][0] != j_oid: continue
        if rc[ri] + rc[rj] > VAN_CAP: continue
        merged = rts[ri] + rts[rj]
        if not simulate_route(merged, allow_overnight=False)["feasible"]: continue
        rts[ri] = merged; rc[ri] += rc[rj]; rts[rj] = None
        for oid in rts[ri]: ro[oid] = ri
    day_routes = [two_opt(r) for r in [r for r in rts if r is not None]]
    weekly_miles_bc2 += sum(simulate_route(r, allow_overnight=False)["total_miles"]
                            for r in day_routes)

annual_miles_bc2 = weekly_miles_bc2 * WEEKS_PER_YEAR
print(f"{weekly_miles_bc2:.0f} mi/wk")

print(f"\n  BC1 : {weekly_miles_bc1:>8,.0f} mi/wk  →  {annual_miles_bc1:>12,.0f} mi/yr")
print(f"  BC2 : {weekly_miles_bc2:>8,.0f} mi/wk  →  {annual_miles_bc2:>12,.0f} mi/yr")
print(f"  Q1  : {weekly_miles_q1:>8,.0f} mi/wk  →  {annual_miles_q1:>12,.0f} mi/yr")


# =============================================================================
# VIZ 1 — WATERFALL CHART
# =============================================================================

print("\n[1/4] Generating Waterfall Chart...")

fig1, ax = plt.subplots(figsize=(12, 7))
fig1.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

labels = ["Base Case 1\nPoint-to-Point\n(No Consolidation)",
          "Base Case 2\nVan Routing\n(No Overnight)",
          "Q1 Final\nVan Routing\n(With Overnight)"]
values = [weekly_miles_bc1, weekly_miles_bc2, weekly_miles_q1]
colors = ["#e05252", "#f0a030", "#52b788"]

bars = ax.bar(labels, values, color=colors, width=0.45,
              edgecolor="white", linewidth=0.6, zorder=3)

# Reduction arrows + labels
for i in range(len(values) - 1):
    reduction = values[i] - values[i+1]
    pct       = reduction / values[i] * 100
    ax.annotate("",
                xy=(i + 0.78, values[i+1] + 400),
                xytext=(i + 0.22, values[i] - 400),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=1.8, mutation_scale=16))
    ax.text(i + 0.5, (values[i] + values[i+1]) / 2,
            f"−{pct:.1f}%\n(−{reduction:,.0f} mi/wk)",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2130",
                      edgecolor="white", alpha=0.9))

# Bar value labels
for bar, val, ann_miles in zip(bars, values, [annual_miles_bc1, annual_miles_bc2, annual_miles_q1]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 700,
            f"{val:,.0f} mi/wk\n({ann_miles/1e6:.2f}M mi/yr)",
            ha="center", va="bottom", fontsize=10.5,
            color="white", fontweight="bold")

# Saving annotation
total_saving     = annual_miles_bc1 - annual_miles_q1
total_saving_pct = total_saving / annual_miles_bc1 * 100
ax.text(0.98, 0.97,
        f"Total Annual Saving vs BC1\n"
        f"{total_saving:,.0f} miles  ({total_saving_pct:.1f}% reduction)\n"
        f"Est. saving @ $0.50/mi:  ${total_saving * 0.50:,.0f}/year",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        color="#52b788", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d2018",
                  edgecolor="#52b788", linewidth=1.5))

ax.set_title("NHG Transportation Optimization — Weekly Miles by Scenario",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Weekly Miles", color="white", fontsize=12)
ax.set_ylim(0, max(values) * 1.22)
ax.yaxis.grid(True, color="#2a2a2a", linestyle="--", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors="white", labelsize=11)
for spine in ax.spines.values(): spine.set_color("#444")

plt.tight_layout()
plt.savefig("Q1_Viz1_Waterfall.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  Saved: Q1_Viz1_Waterfall.png")


# =============================================================================
# VIZ 2 — ALGORITHM COMPARISON TABLE
# =============================================================================

print("[2/4] Generating Algorithm Table...")

ALGOS = ["NN", "CW", "NN+2Opt", "CW+2Opt",
         "NN+2Opt+OrOpt", "CW+2Opt+OrOpt", "CW+3Opt"]

data          = {day: {a: all_day_results[day]["results"][a]["miles"] for a in ALGOS}
                 for day in DAYS}
best_per_day  = {day: min(data[day].values()) for day in DAYS}
worst_per_day = {day: max(data[day].values()) for day in DAYS}
best_weekly   = sum(best_per_day.values())

rows = []
for algo in ALGOS:
    day_vals = [data[day][algo] for day in DAYS]
    weekly   = sum(day_vals)
    vs_best  = (weekly - best_weekly) / best_weekly * 100
    rows.append([algo]
                + [f"{m:.0f}" for m in day_vals]
                + [f"{weekly:,.0f}"]
                + ["BEST ✓" if vs_best < 0.01 else f"+{vs_best:.1f}%"])

col_labels = ["Algorithm", "Mon", "Tue", "Wed", "Thu", "Fri",
              "Weekly\nTotal", "vs\nBest"]

fig2, ax = plt.subplots(figsize=(15, 5.5))
fig2.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")
ax.axis("off")

tbl = ax.table(cellText=rows, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 2.5)

# Header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1a3a6e")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
    tbl[0, j].set_edgecolor("#333")

# Data cells
for i, algo in enumerate(ALGOS):
    ri = i + 1
    tbl[ri, 0].set_facecolor("#1a1f2e")
    tbl[ri, 0].set_text_props(color="white", fontweight="bold")
    tbl[ri, 0].set_edgecolor("#333")

    for j, day in enumerate(DAYS):
        cj  = j + 1
        val = data[day][algo]
        tbl[ri, cj].set_edgecolor("#333")
        if abs(val - best_per_day[day]) < 0.5:
            tbl[ri, cj].set_facecolor("#1a5c38")
            tbl[ri, cj].set_text_props(color="#7fff7f", fontweight="bold")
        elif abs(val - worst_per_day[day]) < 0.5:
            tbl[ri, cj].set_facecolor("#5c1a1a")
            tbl[ri, cj].set_text_props(color="#ff8080")
        else:
            norm = (val - best_per_day[day]) / max(1, worst_per_day[day] - best_per_day[day])
            r_val = int(20 + norm * 55)
            tbl[ri, cj].set_facecolor(f"#{r_val:02x}1e{r_val//3:02x}")
            tbl[ri, cj].set_text_props(color="#cccccc")

    tbl[ri, len(DAYS)+1].set_facecolor("#1a1f2e")
    tbl[ri, len(DAYS)+1].set_text_props(color="white")
    tbl[ri, len(DAYS)+1].set_edgecolor("#333")

    is_best = rows[i][-1] == "BEST ✓"
    tbl[ri, len(DAYS)+2].set_edgecolor("#333")
    if is_best:
        tbl[ri, len(DAYS)+2].set_facecolor("#1a5c38")
        tbl[ri, len(DAYS)+2].set_text_props(color="#7fff7f", fontweight="bold")
    else:
        tbl[ri, len(DAYS)+2].set_facecolor("#1a1f2e")
        tbl[ri, len(DAYS)+2].set_text_props(color="#ff8080")

ax.set_title("Q1 — Algorithm Comparison: Miles per Day  "
             "(green = best per column  |  red = worst per column)",
             color="white", fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("Q1_Viz2_AlgoTable.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  Saved: Q1_Viz2_AlgoTable.png")


# =============================================================================
# VIZ 3a — STATIC ROUTE MAP (matplotlib)
# =============================================================================

print("[3/4] Generating Static Route Map...")

DAY_PALETTES = {
    "Mon": plt.cm.Reds,
    "Tue": plt.cm.Oranges,
    "Wed": plt.cm.Blues,
    "Thu": plt.cm.Purples,
    "Fri": plt.cm.Greens,
}

fig3, axes = plt.subplots(1, 5, figsize=(26, 7))
fig3.patch.set_facecolor("#0f1117")
fig3.suptitle("Q1 Route Map — All Routes by Day  "
              "(★ = Wilmington DC  |  each color = one route  |  numbers = stop sequence)",
              color="white", fontsize=13, fontweight="bold", y=1.02)

for ax_idx, day in enumerate(DAYS):
    ax      = axes[ax_idx]
    ax.set_facecolor("#0d1b2a")
    routes  = all_best_routes[day]
    palette = DAY_PALETTES[day]
    n       = len(routes)
    colors  = [palette(0.4 + 0.5 * i / max(1, n - 1)) for i in range(n)]
    day_mi  = sum(simulate_route(r)["total_miles"] for r in routes)
    depot   = zipid_to_coord[DEPOT_ID]

    for r_idx, route in enumerate(routes):
        rc   = colors[r_idx]
        path = [DEPOT_ID] + [orders[oid]["zip_id"] for oid in route] + [DEPOT_ID]
        lons = [zipid_to_coord[z][1] for z in path if z in zipid_to_coord]
        lats = [zipid_to_coord[z][0] for z in path if z in zipid_to_coord]
        ax.plot(lons, lats, "-", color=rc, linewidth=1.5, zorder=3, alpha=0.85)

        # Numbered stop markers
        for seq, oid in enumerate(route):
            coord = zipid_to_coord.get(orders[oid]["zip_id"])
            if coord:
                ax.scatter(coord[1], coord[0], s=28, color=rc, zorder=4)
                ax.text(coord[1], coord[0], str(seq+1),
                        ha="center", va="center", fontsize=5,
                        color="white", fontweight="bold", zorder=5)

    # Depot star
    ax.scatter(depot[1], depot[0], s=250, color="yellow",
               marker="*", zorder=6)
    ax.text(depot[1], depot[0] + 0.05, "DC", ha="center",
            color="yellow", fontsize=7, fontweight="bold", zorder=7)

    ax.set_title(f"{day}  ·  {n} routes  ·  {day_mi:.0f} mi",
                 color="white", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="#556", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#223")
    if ax_idx == 0: ax.set_ylabel("Latitude", color="#778", fontsize=8)
    ax.set_xlabel("Longitude", color="#778", fontsize=8)

plt.tight_layout()
plt.savefig("Q1_Viz3a_RouteMap.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()
print("  Saved: Q1_Viz3a_RouteMap.png")


# =============================================================================
# VIZ 3b — INTERACTIVE ROUTE MAP (folium)
# =============================================================================

print("[4/4] Generating Interactive Folium Map...")

DAY_META = {
    "Mon": {"color": "#e05252", "full": "Monday"},
    "Tue": {"color": "#f0a030", "full": "Tuesday"},
    "Wed": {"color": "#4db8e8", "full": "Wednesday"},
    "Thu": {"color": "#a05ce8", "full": "Thursday"},
    "Fri": {"color": "#40cc70", "full": "Friday"},
}

def shade_variant(hex_color, r_idx, total):
    """Lighter/darker shade per route within same day."""
    h, s, v = colorsys.rgb_to_hsv(
        int(hex_color[1:3], 16) / 255,
        int(hex_color[3:5], 16) / 255,
        int(hex_color[5:7], 16) / 255,
    )
    v2 = max(0.4, min(1.0, v - 0.12 + (r_idx / max(1, total - 1)) * 0.28))
    r, g, b = colorsys.hsv_to_rgb(h, s, v2)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

depot_lat, depot_lon = zipid_to_coord[DEPOT_ID]

m = folium.Map(location=[42.8, -71.8], zoom_start=8,
               tiles="CartoDB dark_matter", prefer_canvas=True)

Fullscreen(position="topright", title="Fullscreen", force_separate_button=True).add_to(m)
MiniMap(toggle_display=True, position="bottomright",
        tile_layer="CartoDB dark_matter").add_to(m)

# ── Depot marker ──────────────────────────────────────────────────────────────
depot_popup = f"""
<div style="font-family:Arial;font-size:13px;width:210px">
  <b style="font-size:15px">🏭 Wilmington DC</b>
  <hr style="margin:5px 0">
  <b>Role:</b> Distribution Center<br>
  <b>ZIP:</b> 01887 · <b>City:</b> Wilmington, MA<br>
  <hr style="margin:5px 0">
  <b>Weekly routes:</b> {total_routes}<br>
  <b>Weekly miles:</b> {weekly_miles_q1:,.0f}<br>
  <b>Annual miles:</b> {annual_miles_q1:,.0f}
</div>"""

folium.Marker(
    location=[depot_lat, depot_lon],
    popup=folium.Popup(depot_popup, max_width=230),
    tooltip="🏭 Wilmington DC (Depot) — click for details",
    icon=folium.DivIcon(
        html="""<div style="font-size:22px;text-align:center;width:36px;
                height:36px;line-height:36px;background:#f5a623;
                border-radius:50%;border:3px solid white;
                box-shadow:0 0 8px rgba(245,166,35,0.8)">🏭</div>""",
        icon_size=(36, 36), icon_anchor=(18, 18))
).add_to(m)

# ── Route layers ──────────────────────────────────────────────────────────────
for day in DAYS:
    routes    = all_best_routes[day]
    meta      = DAY_META[day]
    day_color = meta["color"]
    day_name  = meta["full"]
    day_mi    = total_miles(routes)
    n_routes  = len(routes)
    n_orders  = len(orders_by_day[day])

    group = folium.FeatureGroup(
        name=f"<span style='color:{day_color}'>■</span> "
             f"{day_name} — {n_routes} routes · {day_mi:.0f} mi",
        show=True
    )

    for r_idx, route in enumerate(routes):
        shade     = shade_variant(day_color, r_idx, n_routes)
        res       = simulate_route(route)
        r_miles   = res["total_miles"]
        r_cube    = sum(orders[oid]["cube"] for oid in route)
        util_pct  = r_cube / VAN_CAP * 100
        overnight = "🌙 Yes" if res["overnight"] else "☀️ No"
        stops     = len(route)
        cities    = [zipid_to_city.get(orders[oid]["zip_id"], "?") for oid in route]
        cities_str = " → ".join(cities)

        # Route line
        path   = [DEPOT_ID] + [orders[oid]["zip_id"] for oid in route] + [DEPOT_ID]
        coords = [(zipid_to_coord[z][0], zipid_to_coord[z][1])
                  for z in path if z in zipid_to_coord]

        route_popup = f"""
<div style="font-family:Arial;font-size:12px;width:270px">
  <b style="font-size:14px;color:{shade}">{day_name} — Route {r_idx+1}</b>
  <hr style="margin:5px 0">
  <table style="width:100%;border-collapse:collapse">
    <tr><td><b>Stops</b></td><td>{stops} deliveries</td></tr>
    <tr><td><b>Distance</b></td><td>{r_miles:.1f} miles</td></tr>
    <tr><td><b>Volume</b></td><td>{r_cube:,} / {VAN_CAP:,} ft³ ({util_pct:.0f}%)</td></tr>
    <tr><td><b>Overnight</b></td><td>{overnight}</td></tr>
  </table>
  <hr style="margin:5px 0">
  <b>Delivery sequence:</b><br>
  <span style="font-size:10px;color:#ccc">DC → {cities_str} → DC</span>
</div>"""

        folium.PolyLine(
            coords, color=shade, weight=3, opacity=0.85,
            tooltip=f"{day_name} Route {r_idx+1}  ·  {stops} stops  ·  {r_miles:.0f} mi",
            popup=folium.Popup(route_popup, max_width=290)
        ).add_to(group)

        # Store markers
        for seq, oid in enumerate(route):
            o       = orders[oid]
            coord   = zipid_to_coord.get(o["zip_id"])
            if not coord: continue
            city    = zipid_to_city.get(o["zip_id"], "Unknown")
            state   = zipid_to_state.get(o["zip_id"], "")
            zipcode = zipid_to_zip.get(o["zip_id"], "")
            dist_dc = get_dist(DEPOT_ID, o["zip_id"])

            store_popup = f"""
<div style="font-family:Arial;font-size:12px;width:230px">
  <b style="font-size:14px">📦 {city}, {state}</b>
  <hr style="margin:5px 0">
  <table style="width:100%;border-collapse:collapse">
    <tr><td><b>Order ID</b></td><td>{oid}</td></tr>
    <tr><td><b>ZIP</b></td><td>{zipcode}</td></tr>
    <tr><td><b>Volume</b></td><td>{o["cube"]:,} ft³</td></tr>
    <tr><td><b>Day</b></td><td>{day_name}</td></tr>
    <tr><td><b>Route</b></td><td>{day} Route {r_idx+1}</td></tr>
    <tr><td><b>Stop #</b></td><td>{seq+1} of {stops}</td></tr>
    <tr><td><b>Dist from DC</b></td><td>{dist_dc:.0f} mi</td></tr>
  </table>
</div>"""

            # Circle with stop number inside
            folium.Marker(
                location=coord,
                tooltip=f"Stop {seq+1}: {city}, {state}  ·  OID {oid}  ·  {o['cube']:,} ft³",
                popup=folium.Popup(store_popup, max_width=250),
                icon=folium.DivIcon(
                    html=f"""<div style="font-size:9px;font-weight:bold;
                              color:white;text-align:center;
                              width:18px;height:18px;line-height:18px;
                              background:{shade};border-radius:50%;
                              border:2px solid white;
                              box-shadow:0 0 3px rgba(0,0,0,0.6)">{seq+1}</div>""",
                    icon_size=(18, 18), icon_anchor=(9, 9))
            ).add_to(group)

    group.add_to(m)

# ── Summary panel ─────────────────────────────────────────────────────────────
day_rows_html = "".join([
    f"<tr>"
    f"<td style='color:{DAY_META[d]['color']};font-weight:bold'>"
    f"{DAY_META[d]['full']}</td>"
    f"<td style='text-align:center'>{len(all_best_routes[d])}</td>"
    f"<td style='text-align:center'>{len(orders_by_day[d])}</td>"
    f"<td style='text-align:right'>{total_miles(all_best_routes[d]):,.0f}</td>"
    f"</tr>"
    for d in DAYS
])

summary = f"""
<div style="position:fixed;top:15px;left:15px;z-index:1000;
            background:rgba(15,17,30,0.93);padding:14px 16px;
            border-radius:10px;border:1px solid #444;color:white;
            font-family:Arial;font-size:12px;
            box-shadow:0 4px 12px rgba(0,0,0,0.6);min-width:265px">
  <b style="font-size:14px">NHG Q1 — Route Map</b>
  <hr style="border-color:#444;margin:8px 0">
  <table style="width:100%;border-collapse:collapse">
    <tr style="color:#aaa;font-size:11px">
      <th style="text-align:left">Day</th>
      <th style="text-align:center">Routes</th>
      <th style="text-align:center">Orders</th>
      <th style="text-align:right">Miles</th>
    </tr>
    {day_rows_html}
    <tr style="border-top:1px solid #444;font-weight:bold">
      <td>TOTAL</td>
      <td style="text-align:center">{total_routes}</td>
      <td style="text-align:center">261</td>
      <td style="text-align:right">{weekly_miles_q1:,.0f}</td>
    </tr>
  </table>
  <hr style="border-color:#444;margin:8px 0">
  <span style="font-size:11px;color:#aaa">
    Annual: <b style="color:white">{annual_miles_q1:,.0f} mi</b><br>
    Vehicle: <b style="color:white">Van (3,200 ft³)</b><br>
    Algorithm: <b style="color:white">CW + 2-Opt + Or-Opt</b>
  </span>
  <hr style="border-color:#444;margin:8px 0">
  <span style="font-size:10px;color:#888">
    Numbers = delivery stop sequence<br>
    Click route or store for details<br>
    Toggle days → layer control (top right)
  </span>
</div>"""

m.get_root().html.add_child(folium.Element(summary))
folium.LayerControl(position="topright", collapsed=False).add_to(m)
m.save("Q1_Viz3b_RouteMap.html")
print("  Saved: Q1_Viz3b_RouteMap.html")


# =============================================================================
# DONE
# =============================================================================

print()
print("=" * 60)
print("  ALL 4 VISUALIZATIONS COMPLETE")
print("=" * 60)
print(f"  Q1_Viz1_Waterfall.png     — Waterfall: BC1 → BC2 → Q1")
print(f"  Q1_Viz2_AlgoTable.png     — Algorithm comparison table")
print(f"  Q1_Viz3a_RouteMap.png     — Static 5-panel route map")
print(f"  Q1_Viz3b_RouteMap.html    — Interactive folium map")
print()
print(f"  Weekly Miles  : {weekly_miles_q1:>10,.1f}")
print(f"  Annual Miles  : {annual_miles_q1:>10,.1f}")
print(f"  Total Routes  : {total_routes:>10}")
print("=" * 60)