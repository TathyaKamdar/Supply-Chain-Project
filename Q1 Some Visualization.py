# =============================================================================
# NHG Vehicle Routing — Q1: Vans Only
# Outputs: Q1_Viz1_Waterfall.png | Q1_Viz2_AlgoTable.png
#          Q1_RouteMap_Dark.png  | Q1_RouteMap_Light.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from itertools import combinations
import time, warnings
warnings.filterwarnings("ignore")


# ── load data ─────────────────────────────────────────────────────────────────
orders    = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
location  = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
distances = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)


# ── clean data ────────────────────────────────────────────────────────────────
orders = orders[orders["ORDERID"] != 0].copy()
orders["CUBE"]  = orders["CUBE"].astype(int)
orders["TOZIP"] = orders["TOZIP"].astype(str).str.zfill(5)

location = location.dropna(subset=["ZIPID"]).copy()
location["ZIPID"]   = location["ZIPID"].astype(int)
location["ZIP_STR"] = location["ZIP"].astype(int).astype(str).str.zfill(5)
location = location.rename(columns={"Y": "LAT", "X": "LON"})


# ── build lookups ─────────────────────────────────────────────────────────────
zip_to_id      = dict(zip(location["ZIP_STR"], location["ZIPID"]))
zipid_to_coord = {int(r["ZIPID"]): (float(r["LAT"]), float(r["LON"]))
                  for _, r in location.iterrows()}

depot_zip = "01887"
depot_id  = int(zip_to_id[depot_zip])

col_ids  = distances.iloc[1, 2:].astype(int).tolist()
row_ids  = distances.iloc[2:, 1].astype(int).tolist()
dist_arr = distances.iloc[2:, 2:].astype(float).values
dist_idx = {int(z): i for i, z in enumerate(col_ids)}

def get_dist(a, b):
    return dist_arr[dist_idx[int(a)], dist_idx[int(b)]]

order_dict = {}
for _, row in orders.iterrows():
    oid = int(row["ORDERID"])
    order_dict[oid] = {
        "cube":   int(row["CUBE"]),
        "day":    row["DayOfWeek"],
        "zip_id": int(zip_to_id[row["TOZIP"]]),
    }

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {}
for d in days:
    orders_by_day[d] = [oid for oid, o in order_dict.items() if o["day"] == d]


# ── define constraints ────────────────────────────────────────────────────────
van_capacity   = 3200   # cubic feet
unload_rate    = 0.030  # min per cubic foot
min_unload     = 30     # min per stop
driving_speed  = 40     # mph
window_open    = 480    # 8am in minutes from midnight
window_close   = 1080   # 6pm in minutes from midnight
break_time     = 600    # 10 hours in minutes
max_driving    = 660    # 11 hours in minutes
max_duty       = 840    # 14 hours in minutes
weeks_per_year = 52


# ── helpers ───────────────────────────────────────────────────────────────────
def drive_mins(a, b):
    return (get_dist(a, b) / driving_speed) * 60

def unload_mins(cube):
    return max(min_unload, unload_rate * cube)


# ── feasibility checker ───────────────────────────────────────────────────────
def simulate(route, allow_overnight=True):
    if not route:
        return {"feasible": True, "miles": 0.0, "overnight": False,
                "drive": 0.0, "duty": 0.0}

    if sum(order_dict[o]["cube"] for o in route) > van_capacity:
        return {"feasible": False, "miles": 0.0, "overnight": False}

    first_zip = order_dict[route[0]]["zip_id"]
    clock     = window_open - drive_mins(depot_id, first_zip)
    loc       = depot_id
    miles     = 0.0
    drive_acc = 0.0
    duty_acc  = 0.0
    overnight = False

    for oid in route:
        dest    = order_dict[oid]["zip_id"]
        leg_mi  = get_dist(loc, dest)
        leg_min = drive_mins(loc, dest)
        ul      = unload_mins(order_dict[oid]["cube"])
        d_left  = max_driving - drive_acc
        dy_left = max_duty    - duty_acc

        if leg_min > d_left or leg_min > dy_left:
            if not allow_overnight or overnight:
                return {"feasible": False, "miles": miles, "overnight": overnight}
            drv = min(d_left, dy_left)
            clock     += drv + break_time
            miles     += (drv / 60) * driving_speed
            drive_acc  = 0.0
            duty_acc   = 0.0
            overnight  = True
            if clock < window_open + 1440:
                clock = window_open + 1440
            rem_mi  = leg_mi - (drv / 60) * driving_speed
            rem_min = (rem_mi / driving_speed) * 60
            clock     += rem_min
            drive_acc += rem_min
            duty_acc  += rem_min
            miles     += rem_mi
            loc        = dest
            if clock > window_close + 1440:
                return {"feasible": False, "miles": miles, "overnight": overnight}
            clock    += ul
            duty_acc += ul
            continue

        clock     += leg_min
        drive_acc += leg_min
        duty_acc  += leg_min
        miles     += leg_mi
        loc        = dest
        limit      = window_close + (1440 if overnight else 0)
        if clock > limit:
            return {"feasible": False, "miles": miles, "overnight": overnight}
        clock    += ul
        duty_acc += ul

    # driver must return to depot within remaining DOT budget
    # overnight break on return leg is NOT allowed per project rules
    ret_mi  = get_dist(loc, depot_id)
    ret_min = drive_mins(loc, depot_id)
    if drive_acc + ret_min > max_driving or duty_acc + ret_min > max_duty:
        return {"feasible": False, "miles": miles, "overnight": overnight}

    miles += ret_mi
    return {"feasible": True, "miles": miles, "overnight": overnight,
            "drive": drive_acc + ret_min, "duty": duty_acc + ret_min}


# ── objective function ────────────────────────────────────────────────────────
def get_miles(route):
    res = simulate(route)
    return res["miles"] if res["feasible"] else float("inf")

def total_miles(routes):
    return sum(get_miles(r) for r in routes)


# ── nearest neighbor ──────────────────────────────────────────────────────────
def nearest_neighbor(day):
    unserved = list(orders_by_day[day])
    routes   = []
    while unserved:
        route = []; cube = 0; loc = depot_id
        while True:
            best_oid = best_d = None
            for oid in unserved:
                o = order_dict[oid]
                if cube + o["cube"] > van_capacity: continue
                if not simulate(route + [oid])["feasible"]: continue
                d = get_dist(loc, o["zip_id"])
                if best_d is None or d < best_d:
                    best_d, best_oid = d, oid
            if best_oid is None: break
            route.append(best_oid)
            cube += order_dict[best_oid]["cube"]
            loc   = order_dict[best_oid]["zip_id"]
            unserved.remove(best_oid)
        routes.append(route) if route else routes.append([unserved.pop(0)])
    return routes


# ── clarke-wright savings ─────────────────────────────────────────────────────
def clarke_wright(day):
    day_oids   = list(orders_by_day[day])
    routes     = [[oid] for oid in day_oids]
    route_cube = [order_dict[oid]["cube"] for oid in day_oids]
    route_of   = {oid: i for i, oid in enumerate(day_oids)}

    savings = []
    for i, j in combinations(day_oids, 2):
        s = (get_dist(depot_id, order_dict[i]["zip_id"]) +
             get_dist(depot_id, order_dict[j]["zip_id"]) -
             get_dist(order_dict[i]["zip_id"], order_dict[j]["zip_id"]))
        savings.append((s, i, j))
        savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])

    for s_val, i_oid, j_oid in savings:
        if s_val <= 0: break
        ri = route_of[i_oid]; rj = route_of[j_oid]
        if ri == rj or routes[ri] is None or routes[rj] is None: continue
        if routes[ri][-1] != i_oid or routes[rj][0] != j_oid: continue
        if route_cube[ri] + route_cube[rj] > van_capacity: continue
        merged = routes[ri] + routes[rj]
        if not simulate(merged)["feasible"]: continue
        routes[ri]     = merged
        route_cube[ri] = route_cube[ri] + route_cube[rj]
        routes[rj]     = None
        for oid in routes[ri]:
            route_of[oid] = ri

    return [r for r in routes if r is not None]


# ── 2-opt ─────────────────────────────────────────────────────────────────────
def two_opt(route):
    if len(route) < 3: return route
    best     = route[:]
    best_mi  = get_miles(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                res  = simulate(cand)
                if res["feasible"] and res["miles"] < best_mi - 0.01:
                    best     = cand
                    best_mi  = res["miles"]
                    improved = True
                    break
            if improved: break
    return best


# ── or-opt ────────────────────────────────────────────────────────────────────
def or_opt(routes):
    routes   = [r[:] for r in routes]
    improved = True
    while improved:
        improved = False
        for r1 in range(len(routes)):
            if not routes[r1]: continue
            for pos in range(len(routes[r1])):
                oid    = routes[r1][pos]
                new_r1 = routes[r1][:pos] + routes[r1][pos+1:]
                for r2 in range(len(routes)):
                    if not routes[r2]: continue
                    cube_r2 = sum(order_dict[o]["cube"] for o in routes[r2])
                    if r1 != r2 and cube_r2 + order_dict[oid]["cube"] > van_capacity:
                        continue
                    before = get_miles(routes[r1])
                    if r1 != r2: before += get_miles(routes[r2])
                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = new_r1[:ins] + [oid] + new_r1[ins:]
                            res  = simulate(cand)
                            if res["feasible"] and res["miles"] < get_miles(routes[r1]) - 0.01:
                                routes[r1] = cand; improved = True; break
                        else:
                            new_r2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            r1r = simulate(new_r1)
                            r2r = simulate(new_r2)
                            if r1r["feasible"] and r2r["feasible"]:
                                if r1r["miles"] + r2r["miles"] < before - 0.01:
                                    routes[r1] = new_r1; routes[r2] = new_r2
                                    improved = True; break
                    if improved: break
                if improved: break
            if improved: break
    return [r for r in routes if r]


# ── 3-opt ─────────────────────────────────────────────────────────────────────
def three_opt(route):
    if len(route) < 5: return two_opt(route)
    best     = route[:]
    best_mi  = get_miles(best)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A, B = best[:i+1], best[i+1:j+1]
                    C, D = best[j+1:k+1], best[k+1:]
                    for cand in [A+B+C[::-1]+D, A+B[::-1]+C+D,
                                 A+B[::-1]+C[::-1]+D, A+C+B+D,
                                 A+C+B[::-1]+D, A+C[::-1]+B+D,
                                 A+C[::-1]+B[::-1]+D]:
                        res = simulate(cand)
                        if res["feasible"] and res["miles"] < best_mi - 0.01:
                            best, best_mi, improved = cand, res["miles"], True; break
                    if improved: break
                if improved: break
            if improved: break
    return best


# ── solve one day (all 7 algorithm combos) ────────────────────────────────────
def solve_day(day):
    nn  = nearest_neighbor(day)
    cw  = clarke_wright(day)
    nn2 = [two_opt(r) for r in nn]
    cw2 = [two_opt(r) for r in cw]
    nno = or_opt(nn2)
    cwo = or_opt(cw2)
    cw3 = [three_opt(r) for r in cw]

    results = {
        "NN":            {"routes": nn,  "miles": total_miles(nn)},
        "CW":            {"routes": cw,  "miles": total_miles(cw)},
        "NN+2Opt":       {"routes": nn2, "miles": total_miles(nn2)},
        "CW+2Opt":       {"routes": cw2, "miles": total_miles(cw2)},
        "NN+2Opt+OrOpt": {"routes": nno, "miles": total_miles(nno)},
        "CW+2Opt+OrOpt": {"routes": cwo, "miles": total_miles(cwo)},
        "CW+3Opt":       {"routes": cw3, "miles": total_miles(cw3)},
    }
    best_key = min(results, key=lambda k: results[k]["miles"])
    return {
        "results":     results,
        "best_key":    best_key,
        "best_routes": results[best_key]["routes"],
        "best_miles":  results[best_key]["miles"],
    }


# ── run q1 ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Q1 — VANS ONLY (CW + 2-Opt + Or-Opt)")
print("=" * 65)

all_results = {}
all_routes  = {}
weekly_q1   = 0.0

for day in days:
    print(f"\n  Solving {day}...", end=" ", flush=True)
    t0  = time.time()
    res = solve_day(day)
    all_results[day] = res
    all_routes[day]  = res["best_routes"]
    weekly_q1       += res["best_miles"]
    print(f"done ({time.time()-t0:.1f}s)  [{res['best_key']}]")
    print(f"    {len(res['best_routes'])} routes  {res['best_miles']:.0f} mi")

annual_q1   = weekly_q1 * weeks_per_year
total_routes = sum(len(v) for v in all_routes.values())

# base case 1: point-to-point (no consolidation)
weekly_bc1 = sum(2 * get_dist(depot_id, o["zip_id"]) for o in order_dict.values())
annual_bc1 = weekly_bc1 * weeks_per_year

# base case 2: no overnight (CW+2Opt only)
print("\n  Computing BC2 (no overnight)...", end=" ", flush=True)
weekly_bc2 = 0.0
for day in days:
    day_oids   = list(orders_by_day[day])
    rts        = [[oid] for oid in day_oids]
    rc         = [order_dict[oid]["cube"] for oid in day_oids]
    ro         = {oid: i for i, oid in enumerate(day_oids)}
    sav        = []
    for i, j in combinations(day_oids, 2):
        s = (get_dist(depot_id, order_dict[i]["zip_id"]) +
             get_dist(depot_id, order_dict[j]["zip_id"]) -
             get_dist(order_dict[i]["zip_id"], order_dict[j]["zip_id"]))
        sav.append((s, i, j)); sav.append((s, j, i))
    sav.sort(key=lambda x: -x[0])
    for s, i, j in sav:
        if s <= 0: break
        ri, rj = ro[i], ro[j]
        if ri == rj or rts[ri] is None or rts[rj] is None: continue
        if rts[ri][-1] != i or rts[rj][0] != j: continue
        if rc[ri] + rc[rj] > van_capacity: continue
        merged = rts[ri] + rts[rj]
        if not simulate(merged, allow_overnight=False)["feasible"]: continue
        rts[ri] = merged; rc[ri] += rc[rj]; rts[rj] = None
        for oid in rts[ri]: ro[oid] = ri
    day_routes  = [two_opt(r) for r in [r for r in rts if r is not None]]
    weekly_bc2 += sum(simulate(r, allow_overnight=False)["miles"] for r in day_routes)
annual_bc2 = weekly_bc2 * weeks_per_year
print("done")

print("\n" + "=" * 65)
print("  WEEKLY SUMMARY")
print("=" * 65)
print(f"  {'Day':<6} {'Routes':>6} {'Miles':>9} {'Algorithm'}")
print(f"  {'-'*6} {'-'*6} {'-'*9} {'-'*20}")
for day in days:
    print(f"  {day:<6} {len(all_routes[day]):>6} "
          f"{all_results[day]['best_miles']:>9.0f}  {all_results[day]['best_key']}")
print(f"  {'─'*6} {'─'*6} {'─'*9}")
print(f"  {'TOTAL':<6} {total_routes:>6} {weekly_q1:>9.0f}")
print(f"\n  Weekly Miles  : {weekly_q1:>10,.0f}")
print(f"  Annual Miles  : {annual_q1:>10,.0f}  (× {weeks_per_year} weeks)")
print(f"  BC1 Weekly    : {weekly_bc1:>10,.0f}")
print(f"  BC2 Weekly    : {weekly_bc2:>10,.0f}")
print("=" * 65)


# ── visualizations ────────────────────────────────────────────────────────────

# viz 1: waterfall bc1 → bc2 → q1
print("\n[1/4] Waterfall Chart...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")

labels  = ["BC1\nPoint-to-Point\n(No Consolidation)",
           "BC2\nVan Routing\n(No Overnight)",
           "Q1\nVan Routing\n(With Overnight)"]
values  = [weekly_bc1, weekly_bc2, weekly_q1]
colors  = ["#e05252", "#f0a030", "#52b788"]
ann_mi  = [annual_bc1, annual_bc2, annual_q1]

bars = ax.bar(labels, values, color=colors, width=0.45,
              edgecolor="white", linewidth=0.6, zorder=3)

for i in range(len(values) - 1):
    reduction = values[i] - values[i+1]
    pct       = reduction / values[i] * 100
    ax.annotate("", xy=(i + 0.78, values[i+1] + 400),
                xytext=(i + 0.22, values[i] - 400),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=1.8, mutation_scale=16))
    ax.text(i + 0.5, (values[i] + values[i+1]) / 2,
            f"-{pct:.1f}%\n(-{reduction:,.0f} mi/wk)",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2130",
                      edgecolor="white", alpha=0.9))

for bar, val, ann in zip(bars, values, ann_mi):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 700,
            f"{val:,.0f} mi/wk\n({ann/1e6:.2f}M mi/yr)",
            ha="center", va="bottom", fontsize=10.5,
            color="white", fontweight="bold")

total_saving     = annual_bc1 - annual_q1
total_saving_pct = total_saving / annual_bc1 * 100
ax.text(0.98, 0.97,
        f"Total Annual Saving vs BC1\n"
        f"{total_saving:,.0f} miles  ({total_saving_pct:.1f}% reduction)\n"
        f"Est. saving @ $0.50/mi:  ${total_saving * 0.50:,.0f}/yr",
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
for sp in ax.spines.values(): sp.set_color("#444")
plt.tight_layout()
plt.savefig("Q1_Viz1_Waterfall.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q1_Viz1_Waterfall.png")


# viz 2: algorithm comparison table
print("[2/4] Algorithm Table...")

algos = ["NN", "CW", "NN+2Opt", "CW+2Opt",
         "NN+2Opt+OrOpt", "CW+2Opt+OrOpt", "CW+3Opt"]

data         = {day: {a: all_results[day]["results"][a]["miles"] for a in algos}
                for day in days}
best_per_day  = {day: min(data[day].values()) for day in days}
worst_per_day = {day: max(data[day].values()) for day in days}
best_weekly   = sum(best_per_day.values())

rows = []
for algo in algos:
    day_vals = [data[day][algo] for day in days]
    weekly   = sum(day_vals)
    vs_best  = (weekly - best_weekly) / best_weekly * 100
    rows.append([algo] + [f"{m:.0f}" for m in day_vals]
                + [f"{weekly:,.0f}"]
                + ["BEST ✓" if vs_best < 0.01 else f"+{vs_best:.1f}%"])

col_labels = ["Algorithm", "Mon", "Tue", "Wed", "Thu", "Fri",
              "Weekly\nTotal", "vs\nBest"]

fig, ax = plt.subplots(figsize=(15, 5.5))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117"); ax.axis("off")

tbl = ax.table(cellText=rows, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.5)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1a3a6e")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
    tbl[0, j].set_edgecolor("#333")

for i, algo in enumerate(algos):
    ri = i + 1
    tbl[ri, 0].set_facecolor("#1a1f2e")
    tbl[ri, 0].set_text_props(color="white", fontweight="bold")
    tbl[ri, 0].set_edgecolor("#333")
    for j, day in enumerate(days):
        cj  = j + 1; val = data[day][algo]
        tbl[ri, cj].set_edgecolor("#333")
        if abs(val - best_per_day[day]) < 0.5:
            tbl[ri, cj].set_facecolor("#1a5c38")
            tbl[ri, cj].set_text_props(color="#7fff7f", fontweight="bold")
        elif abs(val - worst_per_day[day]) < 0.5:
            tbl[ri, cj].set_facecolor("#5c1a1a")
            tbl[ri, cj].set_text_props(color="#ff8080")
        else:
            norm  = (val - best_per_day[day]) / max(1, worst_per_day[day] - best_per_day[day])
            r_val = int(20 + norm * 55)
            tbl[ri, cj].set_facecolor(f"#{r_val:02x}1e{r_val//3:02x}")
            tbl[ri, cj].set_text_props(color="#cccccc")
    tbl[ri, len(days)+1].set_facecolor("#1a1f2e")
    tbl[ri, len(days)+1].set_text_props(color="white")
    tbl[ri, len(days)+1].set_edgecolor("#333")
    is_best = rows[i][-1] == "BEST ✓"
    tbl[ri, len(days)+2].set_edgecolor("#333")
    if is_best:
        tbl[ri, len(days)+2].set_facecolor("#1a5c38")
        tbl[ri, len(days)+2].set_text_props(color="#7fff7f", fontweight="bold")
    else:
        tbl[ri, len(days)+2].set_facecolor("#1a1f2e")
        tbl[ri, len(days)+2].set_text_props(color="#ff8080")

ax.set_title("Q1 — Algorithm Comparison: Miles per Day  "
             "(green = best  |  red = worst per column)",
             color="white", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("Q1_Viz2_AlgoTable.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q1_Viz2_AlgoTable.png")


# viz 3: route maps (dark + light)
print("[3/4] Route Map (dark)...")
print("[4/4] Route Map (light)...")

DAY_CM_DARK  = {"Mon": plt.cm.Reds, "Tue": plt.cm.Oranges,
                "Wed": plt.cm.Blues, "Thu": plt.cm.Purples, "Fri": plt.cm.Greens}
DAY_CM_LIGHT = {"Mon": plt.cm.Reds, "Tue": plt.cm.YlOrBr,
                "Wed": plt.cm.Blues, "Thu": plt.cm.Purples, "Fri": plt.cm.Greens}

depot_coord = zipid_to_coord[depot_id]

suptitle = (f"Q1 Route Map — All Routes by Day  "
            f"(★ = Wilmington DC  ·  numbers = stop sequence  ·  dashed = overnight)\n"
            f"Weekly: {weekly_q1:,.0f} mi  ·  Annual: {annual_q1:,.0f} mi  ·  "
            f"Routes: {total_routes}  ·  Algorithm: CW + 2-Opt + Or-Opt")

legend_handles = [
    mlines.Line2D([], [], color="#888", lw=2, ls="-",  label="Day route (solid)"),
    mlines.Line2D([], [], color="#888", lw=2, ls="--", label="Overnight route (dashed)"),
    mpatches.Patch(facecolor="#f5a623", label="★ Wilmington DC"),
]

def draw_map(axes, bg, text, grid, spine, cm_dict):
    for ax_idx, day in enumerate(days):
        ax      = axes[ax_idx]; ax.set_facecolor(bg)
        routes  = all_routes[day]; n = len(routes)
        palette = cm_dict[day]
        colors  = [palette(0.40 + 0.45 * i / max(1, n-1)) for i in range(n)]
        day_mi  = sum(get_miles(r) for r in routes)
        n_on    = sum(1 for r in routes if simulate(r)["overnight"])

        for oid in orders_by_day[day]:
            coord = zipid_to_coord.get(order_dict[oid]["zip_id"])
            if coord: ax.scatter(coord[1], coord[0], s=6, color=grid, zorder=2, alpha=0.5)

        for ri, route in enumerate(routes):
            rc   = colors[ri]; res = simulate(route)
            path = [depot_id] + [order_dict[o]["zip_id"] for o in route] + [depot_id]
            lons = [zipid_to_coord[z][1] for z in path if z in zipid_to_coord]
            lats = [zipid_to_coord[z][0] for z in path if z in zipid_to_coord]
            ls   = "--" if res["overnight"] else "-"
            ax.plot(lons, lats, ls, color=rc, linewidth=1.6, zorder=3, alpha=0.88)
            for seq, oid in enumerate(route):
                coord = zipid_to_coord.get(order_dict[oid]["zip_id"])
                if coord:
                    ax.scatter(coord[1], coord[0], s=30, color=rc, zorder=4)
                    ax.text(coord[1], coord[0], str(seq+1),
                            ha="center", va="center", fontsize=4.5,
                            color="white", fontweight="bold", zorder=5)
            ax.plot([], [], color=rc, lw=2, label=f"R{ri+1}")

        ax.scatter(depot_coord[1], depot_coord[0], s=250,
                   color="#f5a623", marker="*", zorder=6)
        ax.text(depot_coord[1], depot_coord[0]+0.06, "DC",
                ha="center", fontsize=7, color="#f5a623",
                fontweight="bold", zorder=7)
        ax.legend(loc="lower right", fontsize=5.5, framealpha=0.7,
                  facecolor=bg, labelcolor=text, edgecolor=spine)

        title = f"{day}  ·  {n} routes  ·  {day_mi:.0f} mi"
        if n_on: title += f"  ({n_on} overnight)"
        ax.set_title(title, color=text, fontsize=9.5, fontweight="bold", pad=6)
        ax.tick_params(colors=grid, labelsize=6.5)
        for sp in ax.spines.values(): sp.set_color(spine)
        if ax_idx == 0: ax.set_ylabel("Latitude", color=grid, fontsize=8)
        ax.set_xlabel("Longitude", color=grid, fontsize=7)

# dark version
fig_d, axes_d = plt.subplots(1, 5, figsize=(26, 7))
fig_d.patch.set_facecolor("#0f1117")
draw_map(axes_d, "#0d1b2a", "white", "#445566", "#223344", DAY_CM_DARK)
fig_d.suptitle(suptitle, color="white", fontsize=11, fontweight="bold", y=1.02)
fig_d.legend(handles=legend_handles, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), facecolor="#1e2130",
             labelcolor="white", edgecolor="#444", fontsize=9)
plt.tight_layout()
plt.savefig("Q1_RouteMap_Dark.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q1_RouteMap_Dark.png")

# light version
fig_l, axes_l = plt.subplots(1, 5, figsize=(26, 7))
fig_l.patch.set_facecolor("white")
draw_map(axes_l, "#f8f9fa", "#111111", "#aaaaaa", "#cccccc", DAY_CM_LIGHT)
fig_l.suptitle(suptitle, color="#111111", fontsize=11, fontweight="bold", y=1.02)
fig_l.legend(handles=legend_handles, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.04), facecolor="white",
             labelcolor="#111", edgecolor="#ccc", fontsize=9)
plt.tight_layout()
plt.savefig("Q1_RouteMap_Light.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("   Saved: Q1_RouteMap_Light.png")


print()
print("=" * 65)
print("  Q1 COMPLETE")
print("=" * 65)
print("  Q1_Viz1_Waterfall.png    — BC1 → BC2 → Q1 waterfall")
print("  Q1_Viz2_AlgoTable.png    — Algorithm comparison table")
print("  Q1_RouteMap_Dark.png     — Route map (dark, for screen)")
print("  Q1_RouteMap_Light.png    — Route map (light, for report)")
print()
print(f"  BC1 Weekly : {weekly_bc1:>10,.0f} mi")
print(f"  BC2 Weekly : {weekly_bc2:>10,.0f} mi")
print(f"  Q1  Weekly : {weekly_q1:>10,.0f} mi")
print(f"  Q1  Annual : {annual_q1:>10,.0f} mi")
print(f"  Routes     : {total_routes:>10}")
print("=" * 65)