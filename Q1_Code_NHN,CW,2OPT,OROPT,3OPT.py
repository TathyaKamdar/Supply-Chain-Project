
# NHG Vehicle Routing — Q1: Base Case (Vans Only)
# =============================================================================

import pandas as pd
import numpy as np
from itertools import combinations
import time


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

# Locations — drop garbage rows, normalize ZIP to 5-digit string
locs_df            = raw_locs.dropna(subset=["ZIPID"]).copy()
locs_df["ZIPID"]   = locs_df["ZIPID"].astype(int)
locs_df["ZIP_STR"] = locs_df["ZIP"].astype(int).astype(str).str.zfill(5)

# ZIP → ZipID lookup
zip_to_id = dict(zip(locs_df["ZIP_STR"], locs_df["ZIPID"]))
DEPOT_ZIP = "01887"
DEPOT_ID  = zip_to_id[DEPOT_ZIP]   # ZipID 20

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
UNLOAD_RATE    = 0.030     # min/ft³
MIN_UNLOAD     = 30.0      # min
MAX_DRIVE_MIN  = 660       # 11 hrs
MAX_DUTY_MIN   = 840       # 14 hrs
BREAK_MIN      = 600       # 10 hrs
WINDOW_OPEN    = 480       # 8:00 am in minutes from midnight
WINDOW_CLOSE   = 1080      # 6:00 pm in minutes from midnight
WEEKS_PER_YEAR = 52


# =============================================================================
# STEP 3 — FEASIBILITY CHECKER
# =============================================================================

def _drive_time(a: int, b: int) -> float:
    return (get_dist(a, b) / SPEED_MPH) * 60.0

def _unload_time(cube: int) -> float:
    return max(MIN_UNLOAD, UNLOAD_RATE * cube)

def simulate_route(order_ids: list, allow_overnight: bool = True) -> dict:
    # Returns: feasible, total_miles, drive_min, duty_min, overnight, end_time, timeline
    if not order_ids:
        return {"feasible": True, "total_miles": 0.0, "drive_min": 0.0,
                "duty_min": 0.0, "overnight": False, "end_time": WINDOW_OPEN}

    first_zip     = orders[order_ids[0]]["zip_id"]
    dispatch_time = WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)

    clock     = dispatch_time
    loc       = DEPOT_ID
    miles     = 0.0
    drive_acc = 0.0
    duty_acc  = 0.0
    overnight = False
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
            if not allow_overnight or overnight:
                return {"feasible": False, "total_miles": miles,
                        "reason": f"DOT limit oid={oid}", "overnight": overnight}

            # Drive to legal limit then take 10hr break
            drv_min  = min(d_left, dy_left)
            drv_mi   = (drv_min / 60.0) * SPEED_MPH
            clock   += drv_min + BREAK_MIN
            miles   += drv_mi
            drive_acc = duty_acc = 0.0
            overnight = True
            timeline.append({"event": "OVERNIGHT BREAK", "clock": clock})

            # Wait until 8am next day if needed
            if clock < WINDOW_OPEN + 24 * 60:
                clock = WINDOW_OPEN + 24 * 60

            # Finish remaining leg to destination
            rem_mi    = leg_mi - drv_mi
            rem_min   = (rem_mi / SPEED_MPH) * 60.0
            clock    += rem_min
            drive_acc += rem_min
            duty_acc  += rem_min
            miles    += rem_mi
            loc       = dest

            if clock > WINDOW_CLOSE + 24 * 60:
                return {"feasible": False, "total_miles": miles,
                        "reason": f"oid={oid} after 6pm Day2", "overnight": overnight}

            clock    += ul_min
            duty_acc += ul_min
            timeline.append({"event": f"DELIVER oid={oid} Day2", "clock": clock})
            continue

        # Normal leg
        clock     += leg_min
        drive_acc += leg_min
        duty_acc  += leg_min
        miles     += leg_mi
        loc        = dest

        day_close = WINDOW_CLOSE + (24 * 60 if overnight else 0)
        if clock > day_close:
            return {"feasible": False, "total_miles": miles,
                    "reason": f"oid={oid} after 6pm", "overnight": overnight}

        clock    += ul_min
        duty_acc += ul_min
        timeline.append({"event": f"DELIVER oid={oid}", "clock": clock})

    # Return leg to depot
    ret_mi  = get_dist(loc, DEPOT_ID)
    ret_min = _drive_time(loc, DEPOT_ID)

    if drive_acc + ret_min > MAX_DRIVE_MIN or duty_acc + ret_min > MAX_DUTY_MIN:
        if not overnight and allow_overnight:
            clock    += BREAK_MIN
            drive_acc = duty_acc = 0.0
            overnight = True
            if clock < WINDOW_OPEN + 24 * 60:
                clock = WINDOW_OPEN + 24 * 60
        elif overnight:
            return {"feasible": False, "total_miles": miles,
                    "reason": "DOT limit on return", "overnight": overnight}

    clock     += ret_min
    drive_acc += ret_min
    duty_acc  += ret_min
    miles     += ret_mi
    timeline.append({"event": "RETURN DEPOT", "clock": clock})

    return {"feasible":  True,      "total_miles": miles,
            "drive_min": drive_acc, "duty_min":    duty_acc,
            "overnight": overnight, "end_time":    clock,
            "timeline":  timeline}


# =============================================================================
# STEP 4 — OBJECTIVE FUNCTION
# =============================================================================

def route_miles(route: list) -> float:
    return simulate_route(route)["total_miles"]

def total_miles(routes: list) -> float:
    return sum(route_miles(r) for r in routes)


# =============================================================================
# STEP 5 — TABLE 3 VALIDATION
# =============================================================================

def validate_table3() -> bool:
    sample = [255, 209, 244, 67, 217, 180, 20, 201]
    result = simulate_route(sample, allow_overnight=False)
    cube   = sum(orders[o]["cube"] for o in sample)
    disp   = WINDOW_OPEN - _drive_time(DEPOT_ID, orders[sample[0]]["zip_id"])

    def fmt(m):
        h = int(m // 60) % 24; mi = int(m % 60)
        return f"{h%12 or 12}:{mi:02d} {'am' if h<12 else 'pm'}"

    print("=" * 52)
    print("  TABLE 3 VALIDATION")
    print("=" * 52)
    print(f"  {'Metric':<22} {'Result':>10} {'Expected':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    print(f"  {'Cube (ft³)':<22} {cube:>10,} {'1,245':>10}")
    print(f"  {'Dispatch':<22} {fmt(disp):>10} {'7:36 am':>10}")
    print(f"  {'Return':<22} {fmt(result['end_time']):>10} {'5:45 pm':>10}")
    print(f"  {'Drive (hrs)':<22} {result['drive_min']/60:>10.2f} {'6.30':>10}")
    print(f"  {'Duty (hrs)':<22} {result['duty_min']/60:>10.2f} {'10.30':>10}")

    passed = (cube == 1245 and
              abs(disp - 456) < 2 and
              abs(result["drive_min"] / 60 - 6.3)  < 0.1 and
              abs(result["duty_min"]  / 60 - 10.3) < 0.1 and
              result["feasible"])

    print(f"\n  {'✓ ALL CHECKS PASSED' if passed else '✗ CHECKS FAILED'}")
    print("=" * 52)
    return passed


# =============================================================================
# STEP 6 — NEAREST NEIGHBOR
# =============================================================================

def nearest_neighbor(day: str) -> list:
    # Greedy: always extend to nearest feasible unserved store
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
# STEP 7 — CLARKE-WRIGHT SAVINGS
# =============================================================================

def clarke_wright(day: str) -> list:
    # Parallel CW: s(i,j) = d(depot,i) + d(depot,j) - d(i,j), greedy merge
    day_oids   = list(orders_by_day[day])
    routes     = [[oid] for oid in day_oids]
    route_cube = [orders[oid]["cube"] for oid in day_oids]
    route_of   = {oid: i for i, oid in enumerate(day_oids)}

    # Both orientations prevents 94% tail/head merge misses
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
# STEP 8 — 2-OPT
# =============================================================================

def two_opt(route: list) -> list:
    # Reverse segments between all pairs; accept only if feasible + shorter
    if len(route) < 3:
        return route
    best       = route[:]
    best_mi    = route_miles(best)
    improved   = True
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
# STEP 9 — OR-OPT
# =============================================================================

def or_opt(routes: list) -> list:
    # Relocate single stops between routes to reduce total miles
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
# STEP 10 — 3-OPT
# =============================================================================

def three_opt(route: list) -> list:
    # Try all 3-edge removals (8 reconnections each); O(n³) per pass
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
# STEP 11 — SOLVE DAY + FULL PIPELINE
# =============================================================================

def solve_day(day: str) -> dict:
    results = {}

    # Construction
    t0 = time.time()
    nn = nearest_neighbor(day)
    results["NN"] = {"routes": nn, "miles": total_miles(nn), "time": time.time()-t0}

    t0 = time.time()
    cw = clarke_wright(day)
    results["CW"] = {"routes": cw, "miles": total_miles(cw), "time": time.time()-t0}

    # 2-Opt
    t0 = time.time()
    nn_2opt = [two_opt(r) for r in nn]
    results["NN+2Opt"] = {"routes": nn_2opt, "miles": total_miles(nn_2opt), "time": time.time()-t0}

    t0 = time.time()
    cw_2opt = [two_opt(r) for r in cw]
    results["CW+2Opt"] = {"routes": cw_2opt, "miles": total_miles(cw_2opt), "time": time.time()-t0}

    # Or-Opt
    t0 = time.time()
    nn_oropt = or_opt(nn_2opt)
    results["NN+2Opt+OrOpt"] = {"routes": nn_oropt, "miles": total_miles(nn_oropt), "time": time.time()-t0}

    t0 = time.time()
    cw_oropt = or_opt(cw_2opt)
    results["CW+2Opt+OrOpt"] = {"routes": cw_oropt, "miles": total_miles(cw_oropt), "time": time.time()-t0}

    # 3-Opt
    t0 = time.time()
    cw_3opt = [three_opt(r) for r in cw]
    results["CW+3Opt"] = {"routes": cw_3opt, "miles": total_miles(cw_3opt), "time": time.time()-t0}

    best_key    = min(results, key=lambda k: results[k]["miles"])
    best_routes = results[best_key]["routes"]
    best_miles  = results[best_key]["miles"]

    return {"day": day, "results": results,
            "best_key": best_key, "best_routes": best_routes, "best_miles": best_miles}


def solve_q1() -> dict:
    print("\n" + "=" * 65)
    print("  Q1 — BASE CASE: VANS ONLY")
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

        print(f"\n  {day} — Algorithm Comparison")
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
    print("  WEEKLY SUMMARY")
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

    drivers, vehicles, sleepers = compute_resources(all_best_routes)
    print(f"\n  RESOURCE REQUIREMENTS")
    print(f"  {'Metric':<25} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8}")
    print(f"  {'Min Drivers':<25} {drivers:>8}")
    print(f"  {'Min Vehicles':<25} {vehicles:>8}")
    print(f"  {'Sleeper Cabs':<25} {sleepers:>8}")
    print(f"  {'Day Cabs':<25} {max(0, vehicles-sleepers):>8}")
    print("=" * 65)

    return {"routes": all_best_routes, "weekly_miles": weekly_miles,
            "annual_miles": annual_miles, "drivers": drivers, "vehicles": vehicles}


# =============================================================================
# STEP 12 — RESOURCE REQUIREMENTS
# =============================================================================

def compute_resources(all_routes: dict) -> tuple:
    # Returns (min_drivers, max_concurrent_vehicles, sleeper_cabs)
    day_num = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    events  = []

    for day, routes in all_routes.items():
        d = day_num[day]
        for route in routes:
            result       = simulate_route(route)
            first_zip    = orders[route[0]]["zip_id"]
            dispatch_abs = d * 24 * 60 + WINDOW_OPEN - _drive_time(DEPOT_ID, first_zip)
            end_abs      = d * 24 * 60 + result["end_time"]
            if result["overnight"]:
                end_abs += 24 * 60
            # Sleeper only needed for mid-route overnight (not return-leg only)
            sleeper = any("Day2" in e["event"] for e in result.get("timeline", []))
            events.append({"start": dispatch_abs, "end": end_abs, "sleeper": sleeper})

    events.sort(key=lambda x: x["start"])

    # Min drivers — greedy bin-packing by rest constraint
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

    sleepers = sum(1 for e in events if e["sleeper"])
    return len(driver_ends), max_v, sleepers


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ok = validate_table3()
    if not ok:
        raise SystemExit("Validation failed — fix data pipeline before proceeding.")
    q1 = solve_q1()