# =============================================================================
# NHG Vehicle Routing — Q2: Heterogeneous Fleet (Vans + Straight Trucks)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import MiniMap, Fullscreen
from itertools import combinations
import colorsys, time, warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

raw_orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
raw_locs   = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
raw_dist   = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)

orders_df            = raw_orders[raw_orders["ORDERID"] != 0].copy()
orders_df["CUBE"]    = orders_df["CUBE"].astype(int)
orders_df["TOZIP"]   = orders_df["TOZIP"].astype(str).str.zfill(5)
orders_df["FROMZIP"] = orders_df["FROMZIP"].astype(str).str.zfill(5)
orders_df["ST_REQ"]  = orders_df["ST required?"].str.lower() == "yes"

locs_df            = raw_locs.dropna(subset=["ZIPID"]).copy()
locs_df["ZIPID"]   = locs_df["ZIPID"].astype(int)
locs_df["ZIP_STR"] = locs_df["ZIP"].astype(int).astype(str).str.zfill(5)
locs_df            = locs_df.rename(columns={"Y": "LAT", "X": "LON"})

zip_to_id      = dict(zip(locs_df["ZIP_STR"], locs_df["ZIPID"]))
zipid_to_coord = {int(r["ZIPID"]): (float(r["LAT"]), float(r["LON"]))
                  for _, r in locs_df.iterrows()}
zipid_to_city  = {int(r["ZIPID"]): str(r["CITY"])    for _, r in locs_df.iterrows()}
zipid_to_state = {int(r["ZIPID"]): str(r["STATE"])   for _, r in locs_df.iterrows()}
zipid_to_zip   = {int(r["ZIPID"]): str(r["ZIP_STR"]) for _, r in locs_df.iterrows()}

DEPOT_ZIP = "01887"
DEPOT_ID  = int(zip_to_id[DEPOT_ZIP])

assert not (set(orders_df["TOZIP"]) - set(zip_to_id)), "Unresolved TOZIPs"

col_ids  = raw_dist.iloc[1, 2:].astype(int).tolist()
row_ids  = raw_dist.iloc[2:, 1].astype(int).tolist()
dist_arr = raw_dist.iloc[2:, 2:].astype(float).values
assert col_ids == row_ids, "Distance matrix mismatch"
_idx     = {int(z): i for i, z in enumerate(col_ids)}

def get_dist(a: int, b: int) -> float:
    return dist_arr[_idx[int(a)], _idx[int(b)]]

orders = {
    int(r["ORDERID"]): {
        "id":     int(r["ORDERID"]),
        "cube":   int(r["CUBE"]),
        "day":    r["DayOfWeek"],
        "zip_id": int(zip_to_id[r["TOZIP"]]),
        "st_req": bool(r["ST_REQ"]),
    }
    for _, r in orders_df.iterrows()
}

DAYS          = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {d: [oid for oid, o in orders.items() if o["day"] == d]
                 for d in DAYS}

missing = {o["zip_id"] for o in orders.values()} - set(_idx.keys())
assert not missing, f"ZipIDs missing from distance matrix: {missing}"


# =============================================================================
# STEP 2 — CONSTANTS
# =============================================================================

VAN_CAP        = 3200
ST_CAP         = 1400
SPEED_MPH      = 40.0
UNLOAD_VAN     = 0.030   # min/ft³
UNLOAD_ST      = 0.043   # min/ft³
MIN_UNLOAD     = 30.0
MAX_DRIVE_MIN  = 660
MAX_DUTY_MIN   = 840
BREAK_MIN      = 600
WINDOW_OPEN    = 480
WINDOW_CLOSE   = 1080
WEEKS_PER_YEAR = 52


# =============================================================================
# STEP 3 — CONSTRAINTS: ORDER CATEGORIZATION
# =============================================================================

def categorize_day(day: str) -> tuple:
    day_oids   = orders_by_day[day]
    st_req     = [oid for oid in day_oids if orders[oid]["st_req"]]
    van_forced = [oid for oid in day_oids
                  if not orders[oid]["st_req"] and orders[oid]["cube"] > ST_CAP]
    flexible   = [oid for oid in day_oids
                  if not orders[oid]["st_req"] and orders[oid]["cube"] <= ST_CAP]
    return st_req, van_forced, flexible


# =============================================================================
# STEP 4 — FEASIBILITY CHECKER (vehicle-aware)
# =============================================================================

def _drive_time(a: int, b: int) -> float:
    return (get_dist(a, b) / SPEED_MPH) * 60.0

def _unload_time(cube: int, vehicle: str) -> float:
    rate = UNLOAD_VAN if vehicle == "van" else UNLOAD_ST
    return max(MIN_UNLOAD, rate * cube)

def simulate_route(order_ids: list, vehicle: str = "van",
                   allow_overnight: bool = True) -> dict:
    cap = VAN_CAP if vehicle == "van" else ST_CAP
    if not order_ids:
        return {"feasible": True, "total_miles": 0.0, "overnight": False,
                "drive_min": 0.0, "duty_min": 0.0, "end_time": WINDOW_OPEN}

    if sum(orders[o]["cube"] for o in order_ids) > cap:
        return {"feasible": False, "total_miles": 0.0, "overnight": False,
                "reason": "capacity exceeded"}

    first_zip     = orders[order_ids[0]]["zip_id"]
    dispatch_time = WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)

    clock = dispatch_time; loc = DEPOT_ID
    miles = drive_acc = duty_acc = 0.0
    overnight = False

    for oid in order_ids:
        o = orders[oid]; dest = o["zip_id"]
        leg_mi = get_dist(loc, dest); leg_min = _drive_time(loc, dest)
        ul_min = _unload_time(o["cube"], vehicle)
        d_left = MAX_DRIVE_MIN - drive_acc; dy_left = MAX_DUTY_MIN - duty_acc
        can_drive = leg_min <= d_left and leg_min <= dy_left

        if not can_drive:
            if not allow_overnight or overnight:
                return {"feasible": False, "total_miles": miles,
                        "overnight": overnight, "reason": f"DOT limit oid={oid}"}
            drv_min = min(d_left, dy_left)
            drv_mi  = (drv_min / 60.0) * SPEED_MPH
            clock  += drv_min + BREAK_MIN; miles += drv_mi
            drive_acc = duty_acc = 0.0; overnight = True
            if clock < WINDOW_OPEN + 24 * 60: clock = WINDOW_OPEN + 24 * 60
            rem_mi = leg_mi - drv_mi; rem_min = (rem_mi / SPEED_MPH) * 60.0
            clock += rem_min; drive_acc += rem_min; duty_acc += rem_min
            miles += rem_mi; loc = dest
            if clock > WINDOW_CLOSE + 24 * 60:
                return {"feasible": False, "total_miles": miles,
                        "overnight": overnight, "reason": f"oid={oid} after 6pm Day2"}
            clock += ul_min; duty_acc += ul_min; continue

        clock += leg_min; drive_acc += leg_min; duty_acc += leg_min
        miles += leg_mi; loc = dest
        if clock > WINDOW_CLOSE + (24 * 60 if overnight else 0):
            return {"feasible": False, "total_miles": miles,
                    "overnight": overnight, "reason": f"oid={oid} after 6pm"}
        clock += ul_min; duty_acc += ul_min

    ret_mi  = get_dist(loc, DEPOT_ID); ret_min = _drive_time(loc, DEPOT_ID)
    if drive_acc + ret_min > MAX_DRIVE_MIN or duty_acc + ret_min > MAX_DUTY_MIN:
        if not overnight and allow_overnight:
            clock += BREAK_MIN; drive_acc = duty_acc = 0.0; overnight = True
            if clock < WINDOW_OPEN + 24 * 60: clock = WINDOW_OPEN + 24 * 60
        elif overnight:
            return {"feasible": False, "total_miles": miles,
                    "overnight": overnight, "reason": "DOT limit on return"}
    clock += ret_min; drive_acc += ret_min; duty_acc += ret_min; miles += ret_mi

    return {"feasible": True, "total_miles": miles, "drive_min": drive_acc,
            "duty_min": duty_acc, "overnight": overnight, "end_time": clock}


# =============================================================================
# STEP 5 — OBJECTIVE FUNCTION
# =============================================================================

def route_miles(route: list, vehicle: str) -> float:
    return simulate_route(route, vehicle)["total_miles"]

def total_miles_fleet(van_routes: list, st_routes: list) -> float:
    return (sum(route_miles(r, "van") for r in van_routes) +
            sum(route_miles(r, "st")  for r in st_routes))


# =============================================================================
# STEP 6 — ALGORITHMS (CW + 2Opt + OrOpt)
# =============================================================================

def clarke_wright(oid_list: list, vehicle: str) -> list:
    if not oid_list: return []
    cap      = VAN_CAP if vehicle == "van" else ST_CAP
    routes   = [[oid] for oid in oid_list]
    rcube    = [orders[oid]["cube"] for oid in oid_list]
    route_of = {oid: i for i, oid in enumerate(oid_list)}

    savings = []
    for i, j in combinations(oid_list, 2):
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
        if rcube[ri] + rcube[rj] > cap: continue
        merged = routes[ri] + routes[rj]
        if not simulate_route(merged, vehicle)["feasible"]: continue
        routes[ri] = merged; rcube[ri] += rcube[rj]; routes[rj] = None
        for oid in routes[ri]: route_of[oid] = ri

    return [r for r in routes if r is not None]

def two_opt(route: list, vehicle: str) -> list:
    if len(route) < 3: return route
    best = route[:]; best_mi = route_miles(best, vehicle); improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                res  = simulate_route(cand, vehicle)
                if res["feasible"] and res["total_miles"] < best_mi - 0.01:
                    best, best_mi, improved = cand, res["total_miles"], True; break
            if improved: break
    return best

def or_opt(routes: list, vehicle: str) -> list:
    cap      = VAN_CAP if vehicle == "van" else ST_CAP
    routes   = [r[:] for r in routes]; improved = True
    while improved:
        improved = False
        for r1 in range(len(routes)):
            if not routes[r1]: continue
            for pos in range(len(routes[r1])):
                oid    = routes[r1][pos]
                new_r1 = routes[r1][:pos] + routes[r1][pos+1:]
                for r2 in range(len(routes)):
                    if not routes[r2]: continue
                    if r1 != r2 and sum(orders[o]["cube"] for o in routes[r2]) + orders[oid]["cube"] > cap:
                        continue
                    before = route_miles(routes[r1], vehicle) + (route_miles(routes[r2], vehicle) if r1 != r2 else 0)
                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = new_r1[:ins] + [oid] + new_r1[ins:]
                            res  = simulate_route(cand, vehicle)
                            if res["feasible"] and res["total_miles"] < route_miles(routes[r1], vehicle) - 0.01:
                                routes[r1] = cand; improved = True; break
                        else:
                            new_r2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            r1_ = simulate_route(new_r1, vehicle)
                            r2_ = simulate_route(new_r2, vehicle)
                            if r1_["feasible"] and r2_["feasible"]:
                                if r1_["total_miles"] + r2_["total_miles"] < before - 0.01:
                                    routes[r1] = new_r1; routes[r2] = new_r2
                                    improved = True; break
                    if improved: break
                if improved: break
            if improved: break
    return [r for r in routes if r]


# =============================================================================
# STEP 7 — FLEXIBLE ORDER ASSIGNMENT (savings-sorted best-fit)
# =============================================================================

def insertion_cost(oid: int, routes: list, vehicle: str) -> tuple:
    # Cheapest extra miles to insert oid into any existing route
    # Returns (best_extra_miles, best_route_idx, best_position)
    cap       = VAN_CAP if vehicle == "van" else ST_CAP
    best_cost = float("inf"); best_ri = -1; best_pos = -1

    for ri, route in enumerate(routes):
        if sum(orders[o]["cube"] for o in route) + orders[oid]["cube"] > cap:
            continue
        for ins in range(len(route) + 1):
            cand = route[:ins] + [oid] + route[ins:]
            res  = simulate_route(cand, vehicle)
            if res["feasible"]:
                extra = res["total_miles"] - route_miles(route, vehicle)
                if extra < best_cost:
                    best_cost, best_ri, best_pos = extra, ri, ins

    return best_cost, best_ri, best_pos

def assign_flexible(flex_oids: list, van_routes: list, st_routes: list) -> tuple:
    # Compute marginal cost savings (van_cost - st_cost) for each flexible order
    savings_list = []
    for oid in flex_oids:
        van_cost, _, _ = insertion_cost(oid, van_routes, "van")
        st_cost,  _, _ = insertion_cost(oid, st_routes,  "st")
        saving = van_cost - st_cost   # positive = ST cheaper
        savings_list.append((saving, oid, van_cost, st_cost))

    # Sort: highest ST-preference first
    savings_list.sort(key=lambda x: -x[0])

    for saving, oid, van_cost, st_cost in savings_list:
        st_cost_now, st_ri, st_ins = insertion_cost(oid, st_routes, "st")
        van_cost_now, van_ri, van_ins = insertion_cost(oid, van_routes, "van")

        # Assign to ST if cheaper and has room, else Van
        if st_cost_now < van_cost_now and st_ri >= 0:
            st_routes[st_ri] = st_routes[st_ri][:st_ins] + [oid] + st_routes[st_ri][st_ins:]
        elif van_ri >= 0:
            van_routes[van_ri] = van_routes[van_ri][:van_ins] + [oid] + van_routes[van_ri][van_ins:]
        else:
            # No existing route has room — open new Van route
            van_routes.append([oid])

    return van_routes, st_routes


# =============================================================================
# STEP 8 — MAIN SOLVER (Q2)
# =============================================================================

def solve_day_q2(day: str) -> dict:
    st_req, van_forced, flexible = categorize_day(day)

    # Phase 1 — build initial routes for fixed orders
    st_routes  = clarke_wright(st_req, "st")
    van_routes = clarke_wright(van_forced, "van")

    # Seed empty route lists so insertion always has somewhere to go
    if not st_routes  and st_req:   st_routes  = [[oid] for oid in st_req]
    if not van_routes:              van_routes  = [[]]

    # Phase 2 — assign flexible orders (savings-sorted best-fit)
    van_routes, st_routes = assign_flexible(flexible, van_routes, st_routes)

    # Remove any empty placeholder routes
    van_routes = [r for r in van_routes if r]
    st_routes  = [r for r in st_routes  if r]

    # Phase 3 — improve all routes
    van_routes = or_opt([two_opt(r, "van") for r in van_routes], "van")
    st_routes  = or_opt([two_opt(r, "st")  for r in st_routes],  "st")

    van_miles = sum(route_miles(r, "van") for r in van_routes)
    st_miles  = sum(route_miles(r, "st")  for r in st_routes)

    return {"van_routes": van_routes, "st_routes": st_routes,
            "van_miles": van_miles,   "st_miles": st_miles,
            "total_miles": van_miles + st_miles,
            "n_van": len(van_routes), "n_st": len(st_routes)}


def solve_q2() -> dict:
    print("\n" + "=" * 65)
    print("  Q2 — HETEROGENEOUS FLEET (VANS + STRAIGHT TRUCKS)")
    print("=" * 65)

    all_van    = {}; all_st = {}
    weekly_van = weekly_st = weekly_total = 0.0
    day_summary = []

    for day in DAYS:
        print(f"\n  Solving {day}...", end=" ", flush=True)
        t0  = time.time()
        res = solve_day_q2(day)
        print(f"done ({time.time()-t0:.1f}s)")

        all_van[day] = res["van_routes"]
        all_st[day]  = res["st_routes"]
        weekly_van   += res["van_miles"]
        weekly_st    += res["st_miles"]
        weekly_total += res["total_miles"]

        st_req, van_forced, flexible = categorize_day(day)
        print(f"  {'':3} Van routes: {res['n_van']:>2}  ({res['van_miles']:>7.1f} mi)")
        print(f"  {'':3} ST  routes: {res['n_st']:>2}  ({res['st_miles']:>7.1f} mi)")
        print(f"  {'':3} Day total : {res['total_miles']:>8.1f} mi")

        day_summary.append({
            "day": day, "n_van": res["n_van"], "n_st": res["n_st"],
            "van_miles": res["van_miles"], "st_miles": res["st_miles"],
            "total": res["total_miles"],
            "st_req": len(st_req), "van_forced": len(van_forced),
            "flexible": len(flexible)
        })

    annual_total = weekly_total * WEEKS_PER_YEAR
    annual_van   = weekly_van   * WEEKS_PER_YEAR
    annual_st    = weekly_st    * WEEKS_PER_YEAR

    print("\n" + "=" * 65)
    print("  WEEKLY SUMMARY")
    print("=" * 65)
    print(f"  {'Day':<6} {'Van Rt':>6} {'ST Rt':>6} {'Van Mi':>9} {'ST Mi':>9} {'Total':>9}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*9}")
    for s in day_summary:
        print(f"  {s['day']:<6} {s['n_van']:>6} {s['n_st']:>6} "
              f"{s['van_miles']:>9.1f} {s['st_miles']:>9.1f} {s['total']:>9.1f}")
    print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*9} {'─'*9}")
    n_van_total = sum(s["n_van"] for s in day_summary)
    n_st_total  = sum(s["n_st"]  for s in day_summary)
    print(f"  {'TOTAL':<6} {n_van_total:>6} {n_st_total:>6} "
          f"{weekly_van:>9.1f} {weekly_st:>9.1f} {weekly_total:>9.1f}")

    print(f"\n  Weekly Miles  : {weekly_total:>10,.1f}  "
          f"(Van: {weekly_van:,.1f}  ST: {weekly_st:,.1f})")
    print(f"  Annual Miles  : {annual_total:>10,.1f}  (× {WEEKS_PER_YEAR} weeks)")
    print(f"  Annual Van    : {annual_van:>10,.1f}")
    print(f"  Annual ST     : {annual_st:>10,.1f}")
    print("=" * 65)

    return {"all_van": all_van, "all_st": all_st,
            "weekly_total": weekly_total, "weekly_van": weekly_van,
            "weekly_st": weekly_st, "annual_total": annual_total,
            "day_summary": day_summary,
            "n_van": n_van_total, "n_st": n_st_total}


# =============================================================================
# STEP 9 — RUN SOLVER
# =============================================================================

Q1_WEEKLY = 6566.0   # Q1 result for comparison

print("Running Q2 solver...")
q2 = solve_q2()


# =============================================================================
# VIZ A — FLEET COMPOSITION BARS (Van vs ST per day)
# =============================================================================

print("\n[1/4] Generating Fleet Composition Chart...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")

ds  = q2["day_summary"]
days_labels = [s["day"] for s in ds]
x   = np.arange(len(days_labels))
w   = 0.35

# Panel 1 — Routes per day
ax = axes[0]; ax.set_facecolor("#0f1117")
ax.bar(x - w/2, [s["n_van"] for s in ds], w, label="Van",
       color="#4db8e8", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, [s["n_st"]  for s in ds], w, label="ST",
       color="#f0a030", edgecolor="white", linewidth=0.5)
ax.set_title("Routes per Day", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(days_labels)
ax.set_ylabel("Routes", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white", framealpha=0.8)
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for spine in ax.spines.values(): spine.set_color("#444")

# Panel 2 — Miles per day stacked
ax = axes[1]; ax.set_facecolor("#0f1117")
ax.bar(days_labels, [s["van_miles"] for s in ds],
       color="#4db8e8", label="Van Miles", edgecolor="white", linewidth=0.5)
ax.bar(days_labels, [s["st_miles"]  for s in ds],
       bottom=[s["van_miles"] for s in ds],
       color="#f0a030", label="ST Miles", edgecolor="white", linewidth=0.5)
ax.set_title("Miles per Day (Van + ST)", color="white", fontweight="bold")
ax.set_ylabel("Miles", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white", framealpha=0.8)
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for spine in ax.spines.values(): spine.set_color("#444")

# Panel 3 — Utilization per day
ax = axes[2]; ax.set_facecolor("#0f1117")
van_util = [sum(orders[oid]["cube"] for r in q2["all_van"][s["day"]] for oid in r)
            / (s["n_van"] * VAN_CAP) * 100 if s["n_van"] else 0 for s in ds]
st_util  = [sum(orders[oid]["cube"] for r in q2["all_st"][s["day"]]  for oid in r)
            / (s["n_st"]  * ST_CAP)  * 100 if s["n_st"]  else 0 for s in ds]
ax.bar(x - w/2, van_util, w, color="#4db8e8", label="Van %", edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, st_util,  w, color="#f0a030", label="ST %",  edgecolor="white", linewidth=0.5)
ax.axhline(100, color="#e05252", linestyle="--", linewidth=1, label="100% capacity")
ax.set_title("Avg Capacity Utilization %", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(days_labels)
ax.set_ylabel("Utilization %", color="white")
ax.set_ylim(0, 115)
ax.legend(facecolor="#1e2130", labelcolor="white", framealpha=0.8)
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for spine in ax.spines.values(): spine.set_color("#444")

fig.suptitle("Q2 — Fleet Composition by Day", color="white",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("Q2_VizA_FleetComposition.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  Saved: Q2_VizA_FleetComposition.png")


# =============================================================================
# VIZ B — STACKED MILES BAR WITH Q1 OVERLAY
# =============================================================================

print("[2/4] Generating Stacked Miles Comparison Chart...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

van_mi  = [s["van_miles"] for s in ds]
st_mi   = [s["st_miles"]  for s in ds]
q1_mi   = [1330, 1469, 1144, 1423, 1200]   # Q1 best per day

ax.bar(days_labels, van_mi, color="#4db8e8", label="Q2 Van Miles",
       edgecolor="white", linewidth=0.5, width=0.5)
ax.bar(days_labels, st_mi, bottom=van_mi, color="#f0a030",
       label="Q2 ST Miles", edgecolor="white", linewidth=0.5, width=0.5)
ax.plot(days_labels, q1_mi, "o--", color="#52e07f", linewidth=2,
        markersize=8, label="Q1 Total (Vans only)", zorder=5)

# Value labels on Q1 line
for i, val in enumerate(q1_mi):
    ax.text(i, val + 40, f"{val:,}", ha="center", va="bottom",
            color="#52e07f", fontsize=9, fontweight="bold")

# Q2 total labels on stacked bars
for i, (v, s) in enumerate(zip(van_mi, st_mi)):
    ax.text(i, v + s + 40, f"{v+s:,.0f}", ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold")

ax.set_title("Q2 vs Q1 — Daily Miles Comparison\n"
             "(stacked = Van + ST breakdown  |  line = Q1 baseline)",
             color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("Miles", color="white")
ax.legend(facecolor="#1e2130", labelcolor="white", framealpha=0.8, fontsize=10)
ax.tick_params(colors="white")
ax.yaxis.grid(True, color="#333", linestyle="--")
for spine in ax.spines.values(): spine.set_color("#444")

# Annotation box
diff_pct = (q2["weekly_total"] - Q1_WEEKLY) / Q1_WEEKLY * 100
ax.text(0.98, 0.96,
        f"Q1 Weekly: {Q1_WEEKLY:,.0f} mi\n"
        f"Q2 Weekly: {q2['weekly_total']:,.0f} mi\n"
        f"Difference: {diff_pct:+.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2130",
                  edgecolor="#4db8e8", linewidth=1.5))

plt.tight_layout()
plt.savefig("Q2_VizB_StackedComparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  Saved: Q2_VizB_StackedComparison.png")


# =============================================================================
# VIZ E — Q1 vs Q2 SUMMARY + FLEET SPLIT PIE
# =============================================================================

print("[3/4] Generating Summary Comparison Chart...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1117")

# Panel 1 — Q1 vs Q2 annual miles bar
ax = axes[0]; ax.set_facecolor("#0f1117")
scenarios = ["Q1\nVans Only", "Q2\nMixed Fleet"]
values    = [Q1_WEEKLY * 52, q2["annual_total"]]
colors_b  = ["#52b788", "#4db8e8"]
bars = ax.bar(scenarios, values, color=colors_b, width=0.45,
              edgecolor="white", linewidth=0.6)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"{val:,.0f}", ha="center", va="bottom",
            color="white", fontsize=10, fontweight="bold")
diff = (values[1] - values[0]) / values[0] * 100
ax.text(0.98, 0.97, f"Change: {diff:+.1f}%", transform=ax.transAxes,
        ha="right", va="top", color="#f0a030", fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e2130",
                  edgecolor="#f0a030"))
ax.set_title("Annual Miles: Q1 vs Q2", color="white", fontweight="bold")
ax.set_ylabel("Annual Miles", color="white")
ax.tick_params(colors="white"); ax.yaxis.grid(True, color="#333", linestyle="--")
for spine in ax.spines.values(): spine.set_color("#444")

# Panel 2 — Fleet split pie (routes)
ax = axes[1]; ax.set_facecolor("#0f1117")
pie_vals   = [q2["n_van"], q2["n_st"]]
pie_labels = [f"Van Routes\n({q2['n_van']})", f"ST Routes\n({q2['n_st']})"]
pie_colors = ["#4db8e8", "#f0a030"]
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_labels, colors=pie_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "white", "fontsize": 10},
    wedgeprops={"edgecolor": "white", "linewidth": 1})
for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
ax.set_title("Fleet Split — Routes", color="white", fontweight="bold")

# Panel 3 — Fleet split pie (miles)
ax = axes[2]; ax.set_facecolor("#0f1117")
pie_vals2   = [q2["weekly_van"], q2["weekly_st"]]
pie_labels2 = [f"Van Miles\n({q2['weekly_van']:,.0f})",
               f"ST Miles\n({q2['weekly_st']:,.0f})"]
wedges2, texts2, autotexts2 = ax.pie(
    pie_vals2, labels=pie_labels2, colors=pie_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "white", "fontsize": 10},
    wedgeprops={"edgecolor": "white", "linewidth": 1})
for at in autotexts2: at.set_color("white"); at.set_fontweight("bold")
ax.set_title("Fleet Split — Miles", color="white", fontweight="bold")

fig.suptitle("Q2 Summary — Annual Miles vs Q1  |  Van vs ST Fleet Split",
             color="white", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("Q2_VizE_Summary.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  Saved: Q2_VizE_Summary.png")


# =============================================================================
# VIZ — FOLIUM INTERACTIVE MAP (Van=blue, ST=orange)
# =============================================================================

print("[4/4] Generating Interactive Folium Map...")

def shade_color(hex_color, idx, total):
    h, s, v = colorsys.rgb_to_hsv(
        int(hex_color[1:3], 16)/255,
        int(hex_color[3:5], 16)/255,
        int(hex_color[5:7], 16)/255)
    v2 = max(0.4, min(1.0, v - 0.1 + (idx / max(1, total-1)) * 0.25))
    r, g, b = colorsys.hsv_to_rgb(h, s, v2)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

VAN_COLOR = "#4db8e8"
ST_COLOR  = "#f0a030"
DAY_FULL  = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday",
             "Thu":"Thursday","Fri":"Friday"}

depot_lat, depot_lon = zipid_to_coord[DEPOT_ID]
m = folium.Map(location=[42.8, -71.8], zoom_start=8,
               tiles="CartoDB dark_matter", prefer_canvas=True)
Fullscreen(position="topright").add_to(m)
MiniMap(toggle_display=True, position="bottomright",
        tile_layer="CartoDB dark_matter").add_to(m)

# Depot
folium.Marker(
    location=[depot_lat, depot_lon],
    tooltip="🏭 Wilmington DC (Depot)",
    popup=folium.Popup(
        f"<b>🏭 Wilmington DC</b><br>ZIP: 01887<br>"
        f"Weekly routes: {q2['n_van']+q2['n_st']}<br>"
        f"Weekly miles: {q2['weekly_total']:,.0f}", max_width=200),
    icon=folium.DivIcon(
        html="""<div style="font-size:20px;text-align:center;width:34px;
                height:34px;line-height:34px;background:#f5a623;
                border-radius:50%;border:3px solid white;
                box-shadow:0 0 8px rgba(245,166,35,0.8)">🏭</div>""",
        icon_size=(34, 34), icon_anchor=(17, 17))
).add_to(m)

for day in DAYS:
    van_routes = q2["all_van"][day]
    st_routes  = q2["all_st"][day]
    day_name   = DAY_FULL[day]

    # Van group
    if van_routes:
        van_group = folium.FeatureGroup(
            name=f"<span style='color:{VAN_COLOR}'>■</span> "
                 f"{day_name} — Van ({len(van_routes)} routes)", show=True)
        for ri, route in enumerate(van_routes):
            shade = shade_color(VAN_COLOR, ri, len(van_routes))
            res   = simulate_route(route, "van")
            r_mi  = res["total_miles"]
            r_cube = sum(orders[o]["cube"] for o in route)
            path  = [DEPOT_ID] + [orders[o]["zip_id"] for o in route] + [DEPOT_ID]
            coords = [(zipid_to_coord[z][0], zipid_to_coord[z][1])
                      for z in path if z in zipid_to_coord]
            cities = " → ".join(zipid_to_city.get(orders[o]["zip_id"],"?") for o in route)

            folium.PolyLine(
                coords, color=shade, weight=3, opacity=0.85,
                tooltip=f"{day_name} Van Route {ri+1}  ·  {len(route)} stops  ·  {r_mi:.0f} mi",
                popup=folium.Popup(
                    f"<b style='color:{shade}'>{day_name} — Van Route {ri+1}</b><br>"
                    f"Stops: {len(route)}<br>Miles: {r_mi:.1f}<br>"
                    f"Volume: {r_cube:,}/{VAN_CAP:,} ft³ ({r_cube/VAN_CAP*100:.0f}%)<br>"
                    f"Overnight: {'🌙 Yes' if res['overnight'] else '☀️ No'}<br>"
                    f"<i>{cities}</i>", max_width=300)
            ).add_to(van_group)

            for seq, oid in enumerate(route):
                coord = zipid_to_coord.get(orders[oid]["zip_id"])
                if not coord: continue
                city  = zipid_to_city.get(orders[oid]["zip_id"], "?")
                state = zipid_to_state.get(orders[oid]["zip_id"], "")
                folium.Marker(
                    location=coord,
                    tooltip=f"Van Stop {seq+1}: {city}, {state}  ·  {orders[oid]['cube']:,} ft³",
                    popup=folium.Popup(
                        f"<b>📦 {city}, {state}</b><br>"
                        f"OID: {oid} · ZIP: {zipid_to_zip.get(orders[oid]['zip_id'],'')}<br>"
                        f"Volume: {orders[oid]['cube']:,} ft³<br>"
                        f"Vehicle: Van · Stop {seq+1}/{len(route)}", max_width=220),
                    icon=folium.DivIcon(
                        html=f"""<div style="font-size:8px;font-weight:bold;
                                color:white;text-align:center;width:16px;height:16px;
                                line-height:16px;background:{shade};border-radius:50%;
                                border:2px solid white">{seq+1}</div>""",
                        icon_size=(16,16), icon_anchor=(8,8))
                ).add_to(van_group)
        van_group.add_to(m)

    # ST group
    if st_routes:
        st_group = folium.FeatureGroup(
            name=f"<span style='color:{ST_COLOR}'>■</span> "
                 f"{day_name} — ST ({len(st_routes)} routes)", show=True)
        for ri, route in enumerate(st_routes):
            shade = shade_color(ST_COLOR, ri, len(st_routes))
            res   = simulate_route(route, "st")
            r_mi  = res["total_miles"]
            r_cube = sum(orders[o]["cube"] for o in route)
            path  = [DEPOT_ID] + [orders[o]["zip_id"] for o in route] + [DEPOT_ID]
            coords = [(zipid_to_coord[z][0], zipid_to_coord[z][1])
                      for z in path if z in zipid_to_coord]
            cities = " → ".join(zipid_to_city.get(orders[o]["zip_id"],"?") for o in route)

            folium.PolyLine(
                coords, color=shade, weight=2.5, opacity=0.85,
                dash_array="6 4",   # dashed line = ST routes
                tooltip=f"{day_name} ST Route {ri+1}  ·  {len(route)} stops  ·  {r_mi:.0f} mi",
                popup=folium.Popup(
                    f"<b style='color:{shade}'>{day_name} — ST Route {ri+1}</b><br>"
                    f"Stops: {len(route)}<br>Miles: {r_mi:.1f}<br>"
                    f"Volume: {r_cube:,}/{ST_CAP:,} ft³ ({r_cube/ST_CAP*100:.0f}%)<br>"
                    f"Overnight: {'🌙 Yes' if res['overnight'] else '☀️ No'}<br>"
                    f"<i>{cities}</i>", max_width=300)
            ).add_to(st_group)

            for seq, oid in enumerate(route):
                coord = zipid_to_coord.get(orders[oid]["zip_id"])
                if not coord: continue
                city  = zipid_to_city.get(orders[oid]["zip_id"], "?")
                state = zipid_to_state.get(orders[oid]["zip_id"], "")
                st_req_flag = "⚠️ ST-req" if orders[oid]["st_req"] else "✓ flexible"
                folium.Marker(
                    location=coord,
                    tooltip=f"ST Stop {seq+1}: {city}, {state}  ·  {orders[oid]['cube']:,} ft³",
                    popup=folium.Popup(
                        f"<b>📦 {city}, {state}</b><br>"
                        f"OID: {oid} · ZIP: {zipid_to_zip.get(orders[oid]['zip_id'],'')}<br>"
                        f"Volume: {orders[oid]['cube']:,} ft³<br>"
                        f"Vehicle: Straight Truck · {st_req_flag}<br>"
                        f"Stop {seq+1}/{len(route)}", max_width=220),
                    icon=folium.DivIcon(
                        html=f"""<div style="font-size:8px;font-weight:bold;
                                color:white;text-align:center;width:16px;height:16px;
                                line-height:16px;background:{shade};
                                border-radius:3px;border:2px solid white">{seq+1}</div>""",
                        icon_size=(16,16), icon_anchor=(8,8))
                ).add_to(st_group)
        st_group.add_to(m)

# Summary panel
day_rows = "".join([
    f"<tr><td style='color:#aaa'>{DAY_FULL[s['day']]}</td>"
    f"<td style='text-align:center;color:{VAN_COLOR}'>{s['n_van']}</td>"
    f"<td style='text-align:center;color:{ST_COLOR}'>{s['n_st']}</td>"
    f"<td style='text-align:right'>{s['total']:,.0f}</td></tr>"
    for s in q2["day_summary"]])

summary = f"""
<div style="position:fixed;top:15px;left:15px;z-index:1000;
            background:rgba(15,17,30,0.93);padding:14px 16px;
            border-radius:10px;border:1px solid #444;color:white;
            font-family:Arial;font-size:12px;min-width:270px">
  <b style="font-size:14px">NHG Q2 — Heterogeneous Fleet Map</b>
  <hr style="border-color:#444;margin:8px 0">
  <table style="width:100%;border-collapse:collapse">
    <tr style="color:#aaa;font-size:11px">
      <th style="text-align:left">Day</th>
      <th style="text-align:center;color:{VAN_COLOR}">Vans</th>
      <th style="text-align:center;color:{ST_COLOR}">STs</th>
      <th style="text-align:right">Miles</th>
    </tr>
    {day_rows}
    <tr style="border-top:1px solid #444;font-weight:bold">
      <td>TOTAL</td>
      <td style="text-align:center;color:{VAN_COLOR}">{q2['n_van']}</td>
      <td style="text-align:center;color:{ST_COLOR}">{q2['n_st']}</td>
      <td style="text-align:right">{q2['weekly_total']:,.0f}</td>
    </tr>
  </table>
  <hr style="border-color:#444;margin:8px 0">
  <span style="font-size:11px;color:#aaa">
    Annual: <b style="color:white">{q2['annual_total']:,.0f} mi</b><br>
    <span style="color:{VAN_COLOR}">━━</span> Van routes (solid line)<br>
    <span style="color:{ST_COLOR}">┅┅</span> ST routes (dashed line)<br>
    Circles = Van stops · Squares = ST stops
  </span>
  <hr style="border-color:#444;margin:8px 0">
  <span style="font-size:10px;color:#888">
    Click route or stop for details<br>
    Toggle days/fleets in layer control →
  </span>
</div>"""

m.get_root().html.add_child(folium.Element(summary))
folium.LayerControl(position="topright", collapsed=False).add_to(m)
m.save("Q2_RouteMap.html")
print("  Saved: Q2_RouteMap.html")


# =============================================================================
# DONE
# =============================================================================

print()
print("=" * 65)
print("  ALL Q2 OUTPUTS COMPLETE")
print("=" * 65)
print("  Q2_VizA_FleetComposition.png  — Van vs ST bars per day")
print("  Q2_VizB_StackedComparison.png — Stacked miles + Q1 overlay")
print("  Q2_VizE_Summary.png           — Annual compare + pie charts")
print("  Q2_RouteMap.html              — Interactive folium map")
print()
print(f"  Q1 Weekly  : {Q1_WEEKLY:>10,.1f} mi")
print(f"  Q2 Weekly  : {q2['weekly_total']:>10,.1f} mi")
diff = (q2['weekly_total'] - Q1_WEEKLY) / Q1_WEEKLY * 100
print(f"  Difference : {diff:>+10.1f}%")
print(f"  Q2 Annual  : {q2['annual_total']:>10,.1f} mi")
print("=" * 65)