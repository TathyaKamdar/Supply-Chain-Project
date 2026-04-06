# Q2 - Heterogeneous Fleet (Vans + Straight Trucks)
# Unified Clarke-Wright + Cross-Fleet Improvement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time, warnings
warnings.filterwarnings("ignore")


# ── load data ─────────────────────────────────────────────────────────────────
orders    = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
location  = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
distances = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)


# ── clean data ────────────────────────────────────────────────────────────────
orders = orders[orders["ORDERID"] != 0].copy()
orders["CUBE"]   = orders["CUBE"].astype(int)
orders["TOZIP"]  = orders["TOZIP"].astype(str).str.zfill(5)
orders["ST_REQ"] = orders["ST required?"].str.lower() == "yes"

location = location.dropna(subset=["ZIPID"]).copy()
location["ZIPID"]   = location["ZIPID"].astype(int)
location["ZIP_STR"] = location["ZIP"].astype(int).astype(str).str.zfill(5)
location = location.rename(columns={"Y": "LAT", "X": "LON"})


# ── build lookups ─────────────────────────────────────────────────────────────
zip_to_id      = dict(zip(location["ZIP_STR"], location["ZIPID"]))
zipid_to_coord = {}
zipid_to_city  = {}
zipid_to_state = {}
zipid_to_zip   = {}

for _, row in location.iterrows():
    zid = int(row["ZIPID"])
    zipid_to_coord[zid] = (float(row["LAT"]), float(row["LON"]))
    zipid_to_city[zid]  = str(row["CITY"])
    zipid_to_state[zid] = str(row["STATE"])
    zipid_to_zip[zid]   = str(row["ZIP_STR"])

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
        "st_req": bool(row["ST_REQ"]),
    }

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {}
for d in days:
    orders_by_day[d] = [oid for oid, o in order_dict.items() if o["day"] == d]


# ── define constraints ────────────────────────────────────────────────────────
van_capacity   = 3200
st_capacity    = 1400
unload_van     = 0.030
unload_st      = 0.043
min_unload     = 30
driving_speed  = 40
window_open    = 480
window_close   = 1080
break_time     = 600
max_driving    = 660
max_duty       = 840
weeks_per_year = 52


# ── helpers ───────────────────────────────────────────────────────────────────
def drive_mins(a, b):
    return (get_dist(a, b) / driving_speed) * 60

def unload_mins(cube, vehicle):
    rate = unload_van if vehicle == "van" else unload_st
    return max(min_unload, rate * cube)

def get_cap(vehicle):
    return van_capacity if vehicle == "van" else st_capacity


# ── feasibility checker ───────────────────────────────────────────────────────
def simulate(route, vehicle, allow_overnight=True):
    cap = get_cap(vehicle)
    if not route:
        return {"feasible": True, "miles": 0.0, "overnight": False}
    if sum(order_dict[o]["cube"] for o in route) > cap:
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
        ul      = unload_mins(order_dict[oid]["cube"], vehicle)
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

    ret_mi  = get_dist(loc, depot_id)
    ret_min = drive_mins(loc, depot_id)
    if drive_acc + ret_min > max_driving or duty_acc + ret_min > max_duty:
        if not overnight and allow_overnight:
            clock     += break_time
            drive_acc  = 0.0
            duty_acc   = 0.0
            overnight  = True
            if clock < window_open + 1440:
                clock = window_open + 1440
        elif overnight:
            return {"feasible": False, "miles": miles, "overnight": overnight}

    miles += ret_mi
    return {"feasible": True, "miles": miles, "overnight": overnight}

def get_miles(route, vehicle):
    res = simulate(route, vehicle)
    return res["miles"] if res["feasible"] else float("inf")


# ── objective function ────────────────────────────────────────────────────────
def total_q2_miles(van_routes, st_routes):
    van_mi = sum(get_miles(r, "van") for r in van_routes)
    st_mi  = sum(get_miles(r, "st")  for r in st_routes)
    return van_mi + st_mi


# ── 2-opt improvement ─────────────────────────────────────────────────────────
def two_opt(route, vehicle):
    if len(route) < 3:
        return route
    best     = route[:]
    best_mi  = get_miles(best, vehicle)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                res  = simulate(cand, vehicle)
                if res["feasible"] and res["miles"] < best_mi - 0.01:
                    best     = cand
                    best_mi  = res["miles"]
                    improved = True
                    break
            if improved:
                break
    return best


# ── or-opt improvement ────────────────────────────────────────────────────────
def or_opt(routes, vehicle):
    cap      = get_cap(vehicle)
    routes   = [r[:] for r in routes]
    improved = True
    while improved:
        improved = False
        for r1 in range(len(routes)):
            if not routes[r1]:
                continue
            for pos in range(len(routes[r1])):
                oid    = routes[r1][pos]
                new_r1 = routes[r1][:pos] + routes[r1][pos+1:]
                for r2 in range(len(routes)):
                    if not routes[r2]:
                        continue
                    cube_r2 = sum(order_dict[o]["cube"] for o in routes[r2])
                    if r1 != r2 and cube_r2 + order_dict[oid]["cube"] > cap:
                        continue
                    before = get_miles(routes[r1], vehicle)
                    if r1 != r2:
                        before += get_miles(routes[r2], vehicle)
                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = new_r1[:ins] + [oid] + new_r1[ins:]
                            res  = simulate(cand, vehicle)
                            if res["feasible"] and res["miles"] < get_miles(routes[r1], vehicle) - 0.01:
                                routes[r1] = cand
                                improved   = True
                                break
                        else:
                            new_r2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            r1r = simulate(new_r1, vehicle)
                            r2r = simulate(new_r2, vehicle)
                            if r1r["feasible"] and r2r["feasible"]:
                                if r1r["miles"] + r2r["miles"] < before - 0.01:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved   = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return [r for r in routes if r]


# ── unified clarke-wright (all orders together, fixes flaw 1) ─────────────────
def unified_cw(day):
    all_oids   = orders_by_day[day]
    routes     = [[oid] for oid in all_oids]
    route_cube = [order_dict[oid]["cube"] for oid in all_oids]
    route_of   = {oid: i for i, oid in enumerate(all_oids)}

    # track vehicle type per route index
    route_veh = []
    for oid in all_oids:
        if order_dict[oid]["st_req"]:
            route_veh.append("st")
        elif order_dict[oid]["cube"] > st_capacity:
            route_veh.append("van")
        else:
            route_veh.append("flex")

    # savings both orientations
    savings = []
    for i, j in combinations(all_oids, 2):
        s = (get_dist(depot_id, order_dict[i]["zip_id"]) +
             get_dist(depot_id, order_dict[j]["zip_id"]) -
             get_dist(order_dict[i]["zip_id"], order_dict[j]["zip_id"]))
        savings.append((s, i, j))
        savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])

    for s_val, i_oid, j_oid in savings:
        if s_val <= 0:
            break

        ri = route_of[i_oid]
        rj = route_of[j_oid]

        if ri == rj or routes[ri] is None or routes[rj] is None:
            continue
        if routes[ri][-1] != i_oid or routes[rj][0] != j_oid:
            continue

        # vehicle type compatibility check
        vi = route_veh[ri]
        vj = route_veh[rj]

        # st-required and van-forced cannot merge
        if (vi == "st" and vj == "van") or (vi == "van" and vj == "st"):
            continue

        if vi == "st" or vj == "st":
            merged_veh = "st"
        elif vi == "van" or vj == "van":
            merged_veh = "van"
        else:
            merged_veh = "flex"   # all flexible, defaults to van

        # capacity check for merged vehicle type
        check_vehicle = "st" if merged_veh == "st" else "van"
        cap = get_cap(check_vehicle)
        if route_cube[ri] + route_cube[rj] > cap:
            continue

        # feasibility check
        merged = routes[ri] + routes[rj]
        if not simulate(merged, check_vehicle)["feasible"]:
            continue

        routes[ri]     = merged
        route_cube[ri] = route_cube[ri] + route_cube[rj]
        route_veh[ri]  = merged_veh
        routes[rj]     = None
        for oid in routes[ri]:
            route_of[oid] = ri

    # split into van and st lists
    van_routes = []
    st_routes  = []
    for i, route in enumerate(routes):
        if route is None:
            continue
        if route_veh[i] == "st":
            st_routes.append(route)
        else:
            van_routes.append(route)   # van or flex both go to van

    return van_routes, st_routes


# ── cross-fleet improvement (fixes flaw 3) ────────────────────────────────────
def cross_fleet_improve(van_routes, st_routes):
    improved = True
    while improved:
        improved = False

        # try moving flexible orders van → st
        for v_idx in range(len(van_routes)):
            if not van_routes[v_idx]:
                continue
            for pos in range(len(van_routes[v_idx])):
                oid = van_routes[v_idx][pos]
                if order_dict[oid]["st_req"] or order_dict[oid]["cube"] > st_capacity:
                    continue

                new_van = van_routes[v_idx][:pos] + van_routes[v_idx][pos+1:]

                for s_idx in range(len(st_routes)):
                    if not st_routes[s_idx]:
                        continue
                    if sum(order_dict[o]["cube"] for o in st_routes[s_idx]) + order_dict[oid]["cube"] > st_capacity:
                        continue
                    before = get_miles(van_routes[v_idx], "van") + get_miles(st_routes[s_idx], "st")
                    for ins in range(len(st_routes[s_idx]) + 1):
                        new_st = st_routes[s_idx][:ins] + [oid] + st_routes[s_idx][ins:]
                        vr = simulate(new_van, "van")
                        sr = simulate(new_st,  "st")
                        if vr["feasible"] and sr["feasible"]:
                            if vr["miles"] + sr["miles"] < before - 0.01:
                                van_routes[v_idx] = new_van
                                st_routes[s_idx]  = new_st
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # try moving flexible orders st → van
        for s_idx in range(len(st_routes)):
            if not st_routes[s_idx]:
                continue
            for pos in range(len(st_routes[s_idx])):
                oid = st_routes[s_idx][pos]
                if order_dict[oid]["st_req"]:
                    continue   # st-required stays on st

                new_st = st_routes[s_idx][:pos] + st_routes[s_idx][pos+1:]

                for v_idx in range(len(van_routes)):
                    if not van_routes[v_idx]:
                        continue
                    if sum(order_dict[o]["cube"] for o in van_routes[v_idx]) + order_dict[oid]["cube"] > van_capacity:
                        continue
                    before = get_miles(st_routes[s_idx], "st") + get_miles(van_routes[v_idx], "van")
                    for ins in range(len(van_routes[v_idx]) + 1):
                        new_van = van_routes[v_idx][:ins] + [oid] + van_routes[v_idx][ins:]
                        sr = simulate(new_st,  "st")
                        vr = simulate(new_van, "van")
                        if sr["feasible"] and vr["feasible"]:
                            if sr["miles"] + vr["miles"] < before - 0.01:
                                st_routes[s_idx]  = new_st
                                van_routes[v_idx] = new_van
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    van_routes = [r for r in van_routes if r]
    st_routes  = [r for r in st_routes  if r]
    return van_routes, st_routes


# ── solve one day ─────────────────────────────────────────────────────────────
def solve_day_q2(day):
    van_routes, st_routes = unified_cw(day)
    van_routes = or_opt([two_opt(r, "van") for r in van_routes], "van")
    st_routes  = or_opt([two_opt(r, "st")  for r in st_routes],  "st")
    van_routes, st_routes = cross_fleet_improve(van_routes, st_routes)

    van_mi = sum(get_miles(r, "van") for r in van_routes)
    st_mi  = sum(get_miles(r, "st")  for r in st_routes)

    return {
        "van_routes": van_routes, "st_routes": st_routes,
        "van_miles":  van_mi,     "st_miles":  st_mi,
        "total":      van_mi + st_mi,
        "n_van":      len(van_routes), "n_st": len(st_routes),
    }


# ── run q2 ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Q2 — HETEROGENEOUS FLEET (UNIFIED CW + CROSS-FLEET IMPROVEMENT)")
print("=" * 65)

all_van      = {}
all_st       = {}
weekly_total = 0.0
weekly_van   = 0.0
weekly_st    = 0.0
day_summary  = []

for day in days:
    print(f"\n  Solving {day}...", end=" ", flush=True)
    t0  = time.time()
    res = solve_day_q2(day)
    print(f"done ({time.time()-t0:.1f}s)")

    all_van[day] = res["van_routes"]
    all_st[day]  = res["st_routes"]
    weekly_van   += res["van_miles"]
    weekly_st    += res["st_miles"]
    weekly_total += res["total"]

    print(f"    Van: {res['n_van']} routes  {res['van_miles']:.1f} mi")
    print(f"    ST : {res['n_st']} routes  {res['st_miles']:.1f} mi")
    print(f"    Day total: {res['total']:.1f} mi")

    day_summary.append({
        "day":       day,
        "n_van":     res["n_van"],     "n_st":      res["n_st"],
        "van_miles": res["van_miles"], "st_miles":  res["st_miles"],
        "total":     res["total"],
    })

annual_total = weekly_total * weeks_per_year
annual_van   = weekly_van   * weeks_per_year
annual_st    = weekly_st    * weeks_per_year
n_van_total  = sum(s["n_van"] for s in day_summary)
n_st_total   = sum(s["n_st"]  for s in day_summary)
q1_weekly    = 6566.0
diff_pct     = (weekly_total - q1_weekly) / q1_weekly * 100

print("\n" + "=" * 65)
print("  WEEKLY SUMMARY")
print("=" * 65)
print(f"  {'Day':<6} {'Van Rt':>6} {'ST Rt':>6} {'Van Mi':>9} {'ST Mi':>9} {'Total':>9}")
print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*9}")
for s in day_summary:
    print(f"  {s['day']:<6} {s['n_van']:>6} {s['n_st']:>6} "
          f"{s['van_miles']:>9.1f} {s['st_miles']:>9.1f} {s['total']:>9.1f}")
print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*9} {'─'*9}")
print(f"  {'TOTAL':<6} {n_van_total:>6} {n_st_total:>6} "
      f"{weekly_van:>9.1f} {weekly_st:>9.1f} {weekly_total:>9.1f}")
print(f"\n  Weekly Miles  : {weekly_total:>10,.1f}")
print(f"  Annual Miles  : {annual_total:>10,.1f}  (× {weeks_per_year} weeks)")
print(f"  Annual Van    : {annual_van:>10,.1f}")
print(f"  Annual ST     : {annual_st:>10,.1f}")
print(f"  vs Q1         : {diff_pct:>+10.1f}%")
print("=" * 65)


# ── visualizations ────────────────────────────────────────────────────────────

van_color = "#4db8e8"
st_color  = "#f0a030"
day_full  = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday",
             "Thu":"Thursday","Fri":"Friday"}

# viz a: fleet composition
print("\n[1/4] Fleet Composition Chart...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")
x = np.arange(len(days))
w = 0.35

ax = axes[0]
ax.set_facecolor("#0f1117")
ax.bar(x - w/2, [s["n_van"] for s in day_summary], w,
       color=van_color, label="Van", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, [s["n_st"]  for s in day_summary], w,
       color=st_color,  label="ST",  edgecolor="white", linewidth=0.5)
ax.set_title("Routes per Day", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(days)
ax.set_ylabel("Routes", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white")
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for sp in ax.spines.values(): sp.set_color("#444")

ax = axes[1]
ax.set_facecolor("#0f1117")
ax.bar(days, [s["van_miles"] for s in day_summary],
       color=van_color, label="Van Miles", edgecolor="white", linewidth=0.5)
ax.bar(days, [s["st_miles"]  for s in day_summary],
       bottom=[s["van_miles"] for s in day_summary],
       color=st_color, label="ST Miles", edgecolor="white", linewidth=0.5)
ax.set_title("Miles per Day (Van + ST)", color="white", fontweight="bold")
ax.set_ylabel("Miles", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white")
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for sp in ax.spines.values(): sp.set_color("#444")

ax = axes[2]
ax.set_facecolor("#0f1117")
van_util = []
st_util  = []
for s in day_summary:
    vc = sum(order_dict[o]["cube"] for r in all_van[s["day"]] for o in r)
    sc = sum(order_dict[o]["cube"] for r in all_st[s["day"]]  for o in r)
    van_util.append(vc / max(1, s["n_van"] * van_capacity) * 100)
    st_util.append( sc / max(1, s["n_st"]  * st_capacity)  * 100)
ax.bar(x - w/2, van_util, w, color=van_color, label="Van %", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, st_util,  w, color=st_color,  label="ST %",  edgecolor="white", linewidth=0.5)
ax.axhline(100, color="#e05252", linestyle="--", linewidth=1)
ax.set_title("Avg Capacity Utilization %", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(days)
ax.set_ylabel("Utilization %", color="white"); ax.set_ylim(0, 115)
ax.legend(facecolor="#1e2130", labelcolor="white")
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for sp in ax.spines.values(): sp.set_color("#444")

fig.suptitle("Q2 — Fleet Composition by Day", color="white",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Q2_VizA_FleetComposition.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q2_VizA_FleetComposition.png")

# viz b: stacked + q1 overlay
print("[2/4] Stacked Comparison Chart...")
q1_by_day = [1330, 1469, 1144, 1423, 1200]
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
van_mi = [s["van_miles"] for s in day_summary]
st_mi  = [s["st_miles"]  for s in day_summary]
ax.bar(days, van_mi, color=van_color, label="Q2 Van Miles",
       edgecolor="white", linewidth=0.5, width=0.5)
ax.bar(days, st_mi, bottom=van_mi, color=st_color, label="Q2 ST Miles",
       edgecolor="white", linewidth=0.5, width=0.5)
ax.plot(days, q1_by_day, "o--", color="#52e07f", linewidth=2,
        markersize=8, label="Q1 Total", zorder=5)
for i, val in enumerate(q1_by_day):
    ax.text(i, val + 30, f"{val:,}", ha="center",
            color="#52e07f", fontsize=9, fontweight="bold")
for i, (v, s) in enumerate(zip(van_mi, st_mi)):
    ax.text(i, v + s + 30, f"{v+s:,.0f}", ha="center",
            color="white", fontsize=9, fontweight="bold")
ax.set_title("Q2 vs Q1 — Daily Miles", color="white",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Miles", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=10)
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for sp in ax.spines.values(): sp.set_color("#444")
ax.text(0.98, 0.96,
        f"Q1: {q1_weekly:,.0f} mi/wk\nQ2: {weekly_total:,.0f} mi/wk\n"
        f"Diff: {diff_pct:+.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2130",
                  edgecolor=van_color))
plt.tight_layout()
plt.savefig("Q2_VizB_StackedComparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q2_VizB_StackedComparison.png")

# viz e: summary + pies
print("[3/4] Summary Chart...")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")

ax = axes[0]; ax.set_facecolor("#0f1117")
bars = ax.bar(["Q1\nVans Only", "Q2\nMixed Fleet"],
              [q1_weekly * weeks_per_year, annual_total],
              color=["#52b788", van_color], width=0.45,
              edgecolor="white", linewidth=0.6)
for bar, val in zip(bars, [q1_weekly * 52, annual_total]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
            f"{val:,.0f}", ha="center", color="white",
            fontsize=10, fontweight="bold")
ax.text(0.98, 0.97, f"Change: {diff_pct:+.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        color=st_color, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2130",
                  edgecolor=st_color))
ax.set_title("Annual Miles: Q1 vs Q2", color="white", fontweight="bold")
ax.set_ylabel("Annual Miles", color="white")
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for sp in ax.spines.values(): sp.set_color("#444")

ax = axes[1]; ax.set_facecolor("#0f1117")
_, _, at1 = ax.pie([n_van_total, n_st_total],
    labels=[f"Van\n({n_van_total})", f"ST\n({n_st_total})"],
    colors=[van_color, st_color], autopct="%1.1f%%", startangle=90,
    textprops={"color":"white","fontsize":10},
    wedgeprops={"edgecolor":"white","linewidth":1})
for at in at1: at.set_color("white"); at.set_fontweight("bold")
ax.set_title("Fleet Split — Routes", color="white", fontweight="bold")

ax = axes[2]; ax.set_facecolor("#0f1117")
_, _, at2 = ax.pie([weekly_van, weekly_st],
    labels=[f"Van\n({weekly_van:,.0f} mi)", f"ST\n({weekly_st:,.0f} mi)"],
    colors=[van_color, st_color], autopct="%1.1f%%", startangle=90,
    textprops={"color":"white","fontsize":10},
    wedgeprops={"edgecolor":"white","linewidth":1})
for at in at2: at.set_color("white"); at.set_fontweight("bold")
ax.set_title("Fleet Split — Miles", color="white", fontweight="bold")

fig.suptitle("Q2 Summary — Annual Miles vs Q1  |  Van vs ST Split",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("Q2_VizE_Summary.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q2_VizE_Summary.png")

# viz: static route map (matplotlib, van=blue solid, st=orange dashed)
print("[4/4] Static Route Map...")

DAY_PALETTES_VAN = {"Mon": "#4db8e8", "Tue": "#3da0d0", "Wed": "#5cc8f8",
                    "Thu": "#2d90c0", "Fri": "#6dd8ff"}
DAY_PALETTES_ST  = {"Mon": "#f0a030", "Tue": "#d88820", "Wed": "#ffb840",
                    "Thu": "#c87010", "Fri": "#ffc850"}

fig, axes = plt.subplots(2, 5, figsize=(26, 10))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Q2 Route Map — Van routes (top, solid blue) · ST routes (bottom, dashed orange)\n"
             "★ = Wilmington DC  ·  numbers = delivery sequence",
             color="white", fontsize=12, fontweight="bold", y=1.01)

depot_coord = zipid_to_coord[depot_id]

for col, day in enumerate(days):
    for row, (route_list, vehicle, color_key, palette) in enumerate([
        (all_van[day], "van", DAY_PALETTES_VAN, plt.cm.Blues),
        (all_st[day],  "st",  DAY_PALETTES_ST,  plt.cm.Oranges),
    ]):
        ax = axes[row][col]
        ax.set_facecolor("#0d1b2a")
        n      = len(route_list)
        colors = [palette(0.45 + 0.45 * i / max(1, n-1)) for i in range(n)]
        day_mi = sum(get_miles(r, vehicle) for r in route_list)

        for ri, route in enumerate(route_list):
            rc   = colors[ri]
            path = [depot_id] + [order_dict[o]["zip_id"] for o in route] + [depot_id]
            lons = [zipid_to_coord[z][1] for z in path if z in zipid_to_coord]
            lats = [zipid_to_coord[z][0] for z in path if z in zipid_to_coord]

            ls = "-" if vehicle == "van" else "--"
            ax.plot(lons, lats, ls, color=rc, linewidth=1.5, zorder=3, alpha=0.85)

            for seq, oid in enumerate(route):
                coord = zipid_to_coord.get(order_dict[oid]["zip_id"])
                if not coord:
                    continue
                marker = "o" if vehicle == "van" else "s"
                ax.scatter(coord[1], coord[0], s=28, color=rc,
                           marker=marker, zorder=4)
                ax.text(coord[1], coord[0], str(seq+1),
                        ha="center", va="center", fontsize=5,
                        color="white", fontweight="bold", zorder=5)

        # depot star
        ax.scatter(depot_coord[1], depot_coord[0], s=200,
                   color="yellow", marker="*", zorder=6)
        ax.text(depot_coord[1], depot_coord[0] + 0.05, "DC",
                ha="center", color="yellow", fontsize=7,
                fontweight="bold", zorder=7)

        label   = "Van" if vehicle == "van" else "ST"
        ax.set_title(f"{day} {label}  ·  {n} routes  ·  {day_mi:.0f} mi",
                     color="white", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(colors="#556", labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#223")
        if col == 0:
            ax.set_ylabel("Van Routes" if row == 0 else "ST Routes",
                          color="#aaa", fontsize=8)

plt.tight_layout()
plt.savefig("Q2_Viz_RouteMap.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("   Saved: Q2_Viz_RouteMap.png")

print()
print("=" * 65)
print("  Q2 COMPLETE")
print("=" * 65)
print("  Q2_VizA_FleetComposition.png")
print("  Q2_VizB_StackedComparison.png")
print("  Q2_VizE_Summary.png")
print("  Q2_Viz_RouteMap.png")
print()
print(f"  Q1 Weekly  : {q1_weekly:>10,.1f} mi")
print(f"  Q2 Weekly  : {weekly_total:>10,.1f} mi  ({diff_pct:+.1f}% vs Q1)")
print(f"  Q2 Annual  : {annual_total:>10,.1f} mi")
print(f"  Van routes : {n_van_total}  ·  ST routes : {n_st_total}")
print("=" * 65)