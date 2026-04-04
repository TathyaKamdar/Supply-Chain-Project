import pandas as pd
import numpy as np

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
WEEKS_PER_YEAR = 52


# =============================================================================
# STEP 3 — OBJECTIVE FUNCTION (POINT-TO-POINT)
# =============================================================================

def point_to_point_miles(order_ids: list) -> float:
    # Round trip miles per order: depot → store → depot
    return sum(2 * get_dist(DEPOT_ID, orders[oid]["zip_id"]) for oid in order_ids)


# =============================================================================
# STEP 4 — SOLVE BASE CASE 1
# =============================================================================

def solve_base_case_1() -> dict:
    print("\n" + "=" * 65)
    print("  BASE CASE 1 — POINT-TO-POINT (NO CONSOLIDATION)")
    print("  Every order = dedicated round trip, no route sharing")
    print("=" * 65)

    weekly_miles = 0.0
    day_summary  = []

    print(f"\n  {'Day':<6} {'Orders':>7} {'Routes':>7} {'Miles':>12}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*12}")

    for day in DAYS:
        day_oids  = orders_by_day[day]
        day_miles = point_to_point_miles(day_oids)
        weekly_miles += day_miles
        day_summary.append({"day": day, "orders": len(day_oids),
                             "miles": day_miles})
        print(f"  {day:<6} {len(day_oids):>7} {len(day_oids):>7} {day_miles:>12.1f}")

    print(f"  {'─'*6} {'─'*7} {'─'*7} {'─'*12}")
    print(f"  {'TOTAL':<6} {'261':>7} {'261':>7} {weekly_miles:>12.1f}")

    annual_miles = weekly_miles * WEEKS_PER_YEAR

    print(f"\n  Weekly Miles  : {weekly_miles:>12,.1f}")
    print(f"  Annual Miles  : {annual_miles:>12,.1f}  (× {WEEKS_PER_YEAR} weeks)")
    print(f"  Total Routes  : {len(orders):>12}  (1 per order, no sharing)")

    # Miles per order stats
    all_rt = [2 * get_dist(DEPOT_ID, o["zip_id"]) for o in orders.values()]
    print(f"\n  Per-Order Round Trip Stats:")
    print(f"  {'Metric':<20} {'Miles':>10}")
    print(f"  {'-'*20} {'-'*10}")
    print(f"  {'Min':<20} {min(all_rt):>10.1f}")
    print(f"  {'Max':<20} {max(all_rt):>10.1f}")
    print(f"  {'Average':<20} {sum(all_rt)/len(all_rt):>10.1f}")
    print(f"  {'Median':<20} {sorted(all_rt)[len(all_rt)//2]:>10.1f}")
    print("=" * 65)

    return {"weekly_miles": weekly_miles, "annual_miles": annual_miles,
            "routes": len(orders), "day_summary": day_summary}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    bc1 = solve_base_case_1()

    print("\n  NOTE: Compare against Q1 (CW+2Opt) results:")
    print(f"  {'Scenario':<30} {'Weekly Miles':>14} {'Annual Miles':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*14}")
    print(f"  {'Base Case 1 (Point-to-Point)':<30} {bc1['weekly_miles']:>14,.1f} "
          f"{bc1['annual_miles']:>14,.1f}")
    print(f"  {'Q1 (CW+2Opt, with overnight)':<30} {'6,566':>14} {'341,432':>14}")
    q1_weekly = 6566
    saving_pct = (bc1['weekly_miles'] - q1_weekly) / bc1['weekly_miles'] * 100
    print(f"\n  Optimization reduces miles by {saving_pct:.1f}% vs naive routing")