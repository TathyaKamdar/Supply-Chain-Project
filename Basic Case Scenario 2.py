# =============================================================================
# NHG Vehicle Routing — Base Case 2: No Overnight Routes
# Same algorithm as Q1 (CW+2Opt+OrOpt) but allow_overnight=False
# Represents operation without sleeper cabs / overnight capability
# Shows the value of overnight routing vs Q1
# =============================================================================

import pandas as pd
import numpy as np
from itertools import combinations
import time

# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

raw_orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
raw_locs   = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
raw_dist   = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)

# Orders — drop depot row, fix dtypes, pad ZIPs
orders_df            = raw_orders[raw_orders["ORDERID"] != 0].copy()
orders_df["CUBE"]    = orders_df["CUBE"].astype(int)
orders_df["TOZIP"]   = orders_df["TOZIP"].astype(str).str.zfill(5)
orders_df["FROMZIP"] = orders_df["FROMZIP"].astype(str).str.zfill(5)

# Locations — drop garbage rows, normalize ZIP
locs_df            = raw_locs.dropna(subset=["ZIPID"]).copy()
locs_df["ZIPID"]   = locs_df["ZIPID"].astype(int)
locs_df["ZIP_STR"] = locs_df["ZIP"].astype(int).astype(str).str.zfill(5)

# ZIP → ZipID lookup
zip_to_id = dict(zip(locs_df["ZIP_STR"], locs_df["ZIPID"]))
DEPOT_ZIP = "01887"
DEPOT_ID  = zip_to_id[DEPOT_ZIP]

assert not (set(orders_df["TOZIP"]) - set(zip_to_id)), "Unresolved TOZIPs"

# Distance matrix — two header rows, data from row 2 onward
col_ids  = raw_dist.iloc[1, 2:].astype(int).tolist()
row_ids  = raw_dist.iloc[2:, 1].astype(int).tolist()
dist_arr = raw_dist.iloc[2:, 2:].astype(float).values
assert col_ids == row_ids, "Distance matrix index mismatch"
_idx     = {zipid: i for i, zipid in enumerate(col_ids)}

def get_dist(a: int, b: int) -> float:
    return dist_arr[_idx[a], _idx[b]]

# Master orders dict
orders = {
    int(r["ORDERID"]): {
        "id":     int(r["ORDERID"]),
        "cube":   r["CUBE"],
        "day":    r["DayOfWeek"],
        "zip_id": zip_to_id[r["TOZIP"]],
    }
    for _, r in orders_df.iterrows()
}

DAYS          = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {d: [oid for oid, o in orders.items() if o["day"] == d] for d in DAYS}


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

# Overnight explicitly disabled for this base case
ALLOW_OVERNIGHT = False


# =============================================================================
# STEP 3 — FEASIBILITY CHECKER (OVERNIGHT DISABLED)
# =============================================================================

def _drive_time(a: int, b: int) -> float:
    return (get_dist(a, b) / SPEED_MPH) * 60.0

def _unload_time(cube: int) -> float:
    return max(MIN_UNLOAD, UNLOAD_RATE * cube)

def simulate_route(order_ids: list) -> dict:
    # Overnight always False — single shift routes only
    if not order_ids:
        return {"feasible": True, "total_miles": 0.0, "drive_min": 0.0,
                "duty_min": 0.0, "end_time": WINDOW_OPEN}

    first_zip     = orders[order_ids[0]]["zip_id"]
    dispatch_time = WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)

    clock     = dispatch_time
    loc       = DEPOT_ID
    miles     = 0.0
    drive_acc = 0.0
    duty_acc  = 0.0
    timeline  = [{"event": "DISPATCH", "clock": clock}]

    for oid in order_ids:
        o         = orders[oid]
        dest      = o["zip_id"]
        leg_mi    = get_dist(loc, dest)
        leg_min   = _drive_time(loc, dest)
        ul_min    = _unload_time(o["cube"])
        d_left    = MAX_DRIVE_MIN - drive_acc
        dy_left   = MAX_DUTY_MIN  - duty_acc
        can_drive = leg_min <= d_left and leg_min <= dy_left

        if not can_drive:
            # No overnight allowed — route infeasible
            return {"feasible": False, "total_miles": miles,
                    "reason": f"DOT limit oid={oid}, no overnight allowed"}

        clock     += leg_min
        drive_acc += leg_min
        duty_acc  += leg_min
        miles     += leg_mi
        loc        = dest

        if clock > WINDOW_CLOSE:
            return {"feasible": False, "total_miles": miles,
                    "reason": f"oid={oid} after 6pm"}

        clock    += ul_min
        duty_acc += ul_min
        timeline.append({"event": f"DELIVER oid={oid}", "clock": clock})

    # Return leg
    ret_mi  = get_dist(loc, DEPOT_ID)
    ret_min = _drive_time(loc, DEPOT_ID)

    if drive_acc + ret_min > MAX_DRIVE_MIN or duty_acc + ret_min > MAX_DUTY_MIN:
        return {"feasible": False, "total_miles": miles,
                "reason": "DOT limit on return leg"}

    clock     += ret_min
    drive_acc += ret_min
    duty_acc  += ret_min
    miles     += ret_mi
    timeline.append({"event": "RETURN DEPOT", "clock": clock})

    return {"feasible":  True,      "total_miles": miles,
            "drive_min": drive_acc, "duty_min":    duty_acc,
            "end_time":  clock,     "timeline":    timeline}


# =============================================================================
# STEP 4 — OBJECTIVE FUNCTION
# =============================================================================

def route_miles(route: list) -> float:
    return simulate_route(route)["total_miles"]

def total_miles(routes: list) -> float:
    return sum(route_miles(r) for r in routes)


# =============================================================================
# STEP 5 — NEAREST NEIGHBOR (NO OVERNIGHT)
# =============================================================================

def nearest_neighbor(day: str) -> list:
    unserved = list(orders_by_day[day])
    routes   = []

    while unserved:
        route = []
        cube  = 0
        loc   = DEPOT_ID

        while True:
            best_oid = best_d = None
            for oid in unserved:
                o = orders[oid]
                if cube + o["cube"] > VAN_CAP:
                    continue
                if not simulate_route(route + [oid])["feasible"]:
                    continue
                d = get_dist(loc, o["zip_id"])
                if best_d is None or d < best_d:
                    best_d, best_oid = d, oid

            if best_oid is None:
                break
            route.append(best_oid)
            cube += orders[best_oid]["cube"]
            loc   = orders[best_oid]["zip_id"]
            unserved.remove(best_oid)

        routes.append(route) if route else routes.append([unserved.pop(0)])

    return routes


# =============================================================================
# STEP 6 — CLARKE-WRIGHT SAVINGS (NO OVERNIGHT)
# =============================================================================

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
        savings.append((s, i, j))
        savings.append((s, j, i))
    savings.sort(key=lambda x: -x[0])

    for s_val, i_oid, j_oid in savings:
        if s_val <= 0:
            break
        ri, rj = route_of[i_oid], route_of[j_oid]
        if ri == rj or routes[ri] is None or routes[rj] is None:
            continue
        if routes[ri][-1] != i_oid or routes[rj][0] != j_oid:
            continue
        if route_cube[ri] + route_cube[rj] > VAN_CAP:
            continue
        merged = routes[ri] + routes[rj]
        if not simulate_route(merged)["feasible"]:
            continue
        routes[ri] = merged
        route_cube[ri] += route_cube[rj]
        routes[rj] = None
        for oid in routes[ri]:
            route_of[oid] = ri

    return [r for r in routes if r is not None]


# =============================================================================
# STEP 7 — 2-OPT (NO OVERNIGHT)
# =============================================================================

def two_opt(route: list) -> list:
    if len(route) < 3:
        return route
    best     = route[:]
    best_mi  = route_miles(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand   = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                result = simulate_route(cand)
                if result["feasible"] and result["total_miles"] < best_mi - 0.01:
                    best, best_mi, improved = cand, result["total_miles"], True
                    break
            if improved:
                break
    return best


# =============================================================================
# STEP 8 — OR-OPT (NO OVERNIGHT)
# =============================================================================

def or_opt(routes: list) -> list:
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
                    cube_r2 = sum(orders[o]["cube"] for o in routes[r2])
                    if r1 != r2 and cube_r2 + orders[oid]["cube"] > VAN_CAP:
                        continue

                    before = route_miles(routes[r1]) + (route_miles(routes[r2]) if r1 != r2 else 0)

                    for ins in range(len(routes[r2]) + 1):
                        if r1 == r2:
                            cand = new_r1[:ins] + [oid] + new_r1[ins:]
                            res  = simulate_route(cand)
                            if res["feasible"] and res["total_miles"] < route_miles(routes[r1]) - 0.01:
                                routes[r1] = cand
                                improved   = True
                                break
                        else:
                            new_r2 = routes[r2][:ins] + [oid] + routes[r2][ins:]
                            res1   = simulate_route(new_r1)
                            res2   = simulate_route(new_r2)
                            if res1["feasible"] and res2["feasible"]:
                                if res1["total_miles"] + res2["total_miles"] < before - 0.01:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved   = True
                                    break
                    if improved: break
                if improved: break
            if improved: break

    return [r for r in routes if r]


# =============================================================================
# STEP 9 — 3-OPT (NO OVERNIGHT)
# =============================================================================

def three_opt(route: list) -> list:
    if len(route) < 5:
        return two_opt(route)
    best     = route[:]
    best_mi  = route_miles(best)
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
                        res = simulate_route(cand)
                        if res["feasible"] and res["total_miles"] < best_mi - 0.01:
                            best, best_mi, improved = cand, res["total_miles"], True
                            break
                    if improved: break
                if improved: break
            if improved: break
    return best


# =============================================================================
# STEP 10 — SOLVE DAY + FULL PIPELINE (NO OVERNIGHT)
# =============================================================================

def solve_day(day: str) -> dict:
    results = {}

    t0 = time.time()
    nn = nearest_neighbor(day)
    results["NN"] = {"routes": nn, "miles": total_miles(nn), "time": time.time()-t0}

    t0 = time.time()
    cw = clarke_wright(day)
    results["CW"] = {"routes": cw, "miles": total_miles(cw), "time": time.time()-t0}

    t0 = time.time()
    nn_2opt = [two_opt(r) for r in nn]
    results["NN+2Opt"] = {"routes": nn_2opt, "miles": total_miles(nn_2opt), "time": time.time()-t0}

    t0 = time.time()
    cw_2opt = [two_opt(r) for r in cw]
    results["CW+2Opt"] = {"routes": cw_2opt, "miles": total_miles(cw_2opt), "time": time.time()-t0}

    t0 = time.time()
    nn_oropt = or_opt(nn_2opt)
    results["NN+2Opt+OrOpt"] = {"routes": nn_oropt, "miles": total_miles(nn_oropt), "time": time.time()-t0}

    t0 = time.time()
    cw_oropt = or_opt(cw_2opt)
    results["CW+2Opt+OrOpt"] = {"routes": cw_oropt, "miles": total_miles(cw_oropt), "time": time.time()-t0}

    t0 = time.time()
    cw_3opt = [three_opt(r) for r in cw]
    results["CW+3Opt"] = {"routes": cw_3opt, "miles": total_miles(cw_3opt), "time": time.time()-t0}

    best_key    = min(results, key=lambda k: results[k]["miles"])
    best_routes = results[best_key]["routes"]
    best_miles  = results[best_key]["miles"]

    return {"day": day, "results": results,
            "best_key": best_key, "best_routes": best_routes, "best_miles": best_miles}


def solve_base_case_2() -> dict:
    print("\n" + "=" * 65)
    print("  BASE CASE 2 — NO OVERNIGHT ROUTES")
    print("  Same algorithm as Q1 but overnight disabled")
    print("  All routes must complete within single shift")
    print("=" * 65)

    all_best_routes = {}
    weekly_miles    = 0.0
    day_summary     = []

    for day in DAYS:
        print(f"\n  Solving {day}...", end=" ", flush=True)
        t0         = time.time()
        day_result = solve_day(day)
        print(f"done ({time.time()-t0:.1f}s)")

        all_best_routes[day] = day_result["best_routes"]
        weekly_miles        += day_result["best_miles"]

        print(f"\n  {day} — Algorithm Comparison (No Overnight)")
        print(f"  {'Algorithm':<22} {'Routes':>7} {'Miles':>10} {'Time(s)':>9}")
        print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*9}")
        for algo, res in day_result["results"].items():
            marker = " ←" if algo == day_result["best_key"] else ""
            print(f"  {algo:<22} {len(res['routes']):>7} {res['miles']:>10.1f} "
                  f"{res['time']:>9.2f}{marker}")

        day_summary.append({"day":    day,
                             "orders": len(orders_by_day[day]),
                             "routes": len(day_result["best_routes"]),
                             "miles":  day_result["best_miles"],
                             "best":   day_result["best_key"]})

    annual_miles = weekly_miles * WEEKS_PER_YEAR

    print("\n" + "=" * 65)
    print("  WEEKLY SUMMARY — NO OVERNIGHT")
    print("=" * 65)
    print(f"  {'Day':<6} {'Orders':>7} {'Routes':>7} {'Miles':>10} {'Best Algorithm':<25}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*25}")
    for s in day_summary:
        print(f"  {s['day']:<6} {s['orders']:>7} {s['routes']:>7} "
              f"{s['miles']:>10.1f} {s['best']:<25}")
    print(f"  {'─'*6} {'─'*7} {'─'*7} {'─'*10}")
    print(f"  {'TOTAL':<6} {'261':>7} "
          f"{sum(s['routes'] for s in day_summary):>7} {weekly_miles:>10.1f}")

    print(f"\n  Weekly Miles  : {weekly_miles:>10,.1f}")
    print(f"  Annual Miles  : {annual_miles:>10,.1f}  (× {WEEKS_PER_YEAR} weeks)")

    # Resource requirements — no sleeper cabs needed
    drivers, vehicles = compute_resources(all_best_routes)
    print(f"\n  RESOURCE REQUIREMENTS (No Overnight)")
    print(f"  {'Metric':<25} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8}")
    print(f"  {'Min Drivers':<25} {drivers:>8}")
    print(f"  {'Min Vehicles':<25} {vehicles:>8}")
    print(f"  {'Sleeper Cabs':<25} {'0':>8}  (not needed — no overnight)")
    print(f"  {'Day Cabs':<25} {vehicles:>8}  (all day cabs)")
    print("=" * 65)

    return {"routes": all_best_routes, "weekly_miles": weekly_miles,
            "annual_miles": annual_miles, "drivers": drivers, "vehicles": vehicles}


# =============================================================================
# STEP 11 — RESOURCE REQUIREMENTS (NO OVERNIGHT)
# =============================================================================

def compute_resources(all_routes: dict) -> tuple:
    # No overnight routes — simpler than Q1, no sleeper cab logic needed
    day_num = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    events  = []

    for day, routes in all_routes.items():
        d = day_num[day]
        for route in routes:
            result       = simulate_route(route)
            first_zip    = orders[route[0]]["zip_id"]
            dispatch_abs = d * 24 * 60 + WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)
            end_abs      = d * 24 * 60 + result["end_time"]
            events.append({"start": dispatch_abs, "end": end_abs})

    events.sort(key=lambda x: x["start"])

    # Min drivers — greedy bin-packing
    driver_ends = []
    for e in events:
        assigned = False
        for i, end_t in enumerate(driver_ends):
            if e["start"] >= end_t + BREAK_MIN:
                driver_ends[i] = e["end"]
                assigned = True
                break
        if not assigned:
            driver_ends.append(e["end"])

    # Max concurrent vehicles — sweep line
    sweep = [(e["start"], +1) for e in events] + [(e["end"], -1) for e in events]
    sweep.sort()
    curr = max_v = 0
    for _, delta in sweep:
        curr += delta
        max_v = max(max_v, curr)

    return len(driver_ends), max_v


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    bc2 = solve_base_case_2()

    print("\n  COMPARISON: Base Case 2 vs Q1")
    print(f"  {'Scenario':<35} {'Weekly':>10} {'Annual':>12} {'Routes':>8} {'Sleepers':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")
    print(f"  {'Base Case 2 (No Overnight)':<35} "
          f"{bc2['weekly_miles']:>10,.1f} {bc2['annual_miles']:>12,.1f} "
          f"{sum(len(v) for v in bc2['routes'].values()):>8} {'0':>10}")
    print(f"  {'Q1 (With Overnight)':<35} "
          f"{'6,566':>10} {'341,432':>12} {'26':>8} {'8':>10}")

    q1_weekly = 6566
    saving    = bc2['weekly_miles'] - q1_weekly
    saving_pct = saving / bc2['weekly_miles'] * 100
    print(f"\n  Overnight routing saves {saving:.1f} miles/week "
          f"({saving_pct:.1f}% reduction)")
    print(f"  Annual saving from overnight capability: "
          f"{saving * WEEKS_PER_YEAR:,.1f} miles")