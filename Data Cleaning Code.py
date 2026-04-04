import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING  ·  NHG Vehicle Routing Project
# Covers all three questions (Q1, Q2, Q3)
# ══════════════════════════════════════════════════════════════════════════════


# ── 1. LOAD RAW FILES ─────────────────────────────────────────────────────────
raw_orders = pd.read_excel("deliveries.xlsx", sheet_name="OrderTable")
raw_locs   = pd.read_excel("deliveries.xlsx", sheet_name="LocationTable")
raw_dist   = pd.read_excel("distances.xlsx",  sheet_name="Sheet1", header=None)


# ── 2. CLEAN ORDERS ───────────────────────────────────────────────────────────

# Drop depot placeholder — its "(Depot)" string corrupts CUBE dtype
orders_df = raw_orders[raw_orders["ORDERID"] != 0].copy()

# Cast after removing the non-numeric row
orders_df["CUBE"] = orders_df["CUBE"].astype(int)

# Standardize both ZIPs to 5-digit strings for consistent matching
orders_df["TOZIP"]   = orders_df["TOZIP"].astype(str).str.zfill(5)
orders_df["FROMZIP"] = orders_df["FROMZIP"].astype(str).str.zfill(5)

# Boolean flag used in Q2 to separate ST-required orders from flexible ones
orders_df["ST_REQ"] = orders_df["ST required?"].str.lower() == "yes"


# ── 3. CLEAN LOCATIONS ────────────────────────────────────────────────────────

# Drop trailing garbage rows — they have no ZIPID and break all lookups
locs_df = raw_locs.dropna(subset=["ZIPID"]).copy()
locs_df["ZIPID"] = locs_df["ZIPID"].astype(int)

# Normalize ZIP to 5-digit string to align with orders format
locs_df["ZIP_STR"] = locs_df["ZIP"].astype(int).astype(str).str.zfill(5)

# Retain lat/lon for Q3 geographic clustering (Y=latitude, X=longitude)
locs_df = locs_df.rename(columns={"Y": "LAT", "X": "LON"})


# ── 4. BUILD LOOKUPS ──────────────────────────────────────────────────────────

# ZIP string → ZipID  (used to query distance matrix)
zip_to_id = dict(zip(locs_df["ZIP_STR"], locs_df["ZIPID"]))

# ZipID → (lat, lon)  (used in Q3 KMeans clustering)
zipid_to_coord = {
    row["ZIPID"]: (row["LAT"], row["LON"])
    for _, row in locs_df.iterrows()
}

DEPOT_ZIP = "01887"
DEPOT_ID  = zip_to_id[DEPOT_ZIP]   # ZipID = 20


# ── 5. VERIFY ALL TOZIPS RESOLVE ──────────────────────────────────────────────
missing = set(orders_df["TOZIP"]) - set(zip_to_id)
assert not missing, f"Unresolved TOZIPs (ZIP format mismatch): {missing}"


# ── 6. BUILD MASTER ORDERS DICT ───────────────────────────────────────────────
# Single source of truth used by Q1, Q2, and Q3 algorithms
orders = {
    int(row["ORDERID"]): {
        "id":     int(row["ORDERID"]),
        "cube":   row["CUBE"],
        "day":    row["DayOfWeek"],              # 'Mon' | 'Tue' | 'Wed' | 'Thu' | 'Fri'
        "zip_id": zip_to_id[row["TOZIP"]],
        "st_req": row["ST_REQ"],                 # Q2: True = must use Straight Truck
    }
    for _, row in orders_df.iterrows()
}

# Orders grouped by delivery day — used by all three questions
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
orders_by_day = {
    day: [oid for oid, o in orders.items() if o["day"] == day]
    for day in DAYS
}

# Q2 pre-categorized order lists (avoids recomputing inside the algorithm)
# Van-forced: not ST-required but cube > ST capacity — must go on Van
# Flexible:   not ST-required and cube ≤ ST capacity — can go on either
ST_CAPACITY  = 1400
VAN_CAPACITY = 3200

st_required = [oid for oid, o in orders.items() if o["st_req"]]
van_forced  = [oid for oid, o in orders.items() if not o["st_req"] and o["cube"] > ST_CAPACITY]
flexible    = [oid for oid, o in orders.items() if not o["st_req"] and o["cube"] <= ST_CAPACITY]


# ── 7. BUILD DISTANCE MATRIX ──────────────────────────────────────────────────

# Row 0 = zip code labels, Row 1 = ZipID labels — actual data starts at row 2
col_ids  = raw_dist.iloc[1, 2:].astype(int).tolist()   # ZipIDs for columns
row_ids  = raw_dist.iloc[2:, 1].astype(int).tolist()   # ZipIDs for rows
dist_arr = raw_dist.iloc[2:, 2:].astype(float).values  # 123×123 numeric matrix

assert col_ids == row_ids, "Distance matrix row/column ZipIDs do not match"

# ZipID → matrix index for O(1) numpy lookup
_idx = {zipid: i for i, zipid in enumerate(col_ids)}


def get_dist(a: int, b: int) -> float:
    """Road distance in miles between two locations by ZipID."""
    return dist_arr[_idx[a], _idx[b]]


# ── 8. SUMMARY PRINTOUT ───────────────────────────────────────────────────────
print("═" * 50)
print("  DATA LOADED SUCCESSFULLY")
print("═" * 50)
print(f"  Depot        :  ZIP {DEPOT_ZIP}  →  ZipID {DEPOT_ID}")
print(f"  Total orders :  {len(orders)}")
print(f"  Locations    :  {len(locs_df)}  (stores + depot)")
print(f"  Dist matrix  :  {len(col_ids)} × {len(col_ids)}")
print()
print(f"  {'Day':<6} {'Orders':>7} {'Volume (ft³)':>14}")
print(f"  {'─'*6} {'─'*7} {'─'*14}")
for day in DAYS:
    n   = len(orders_by_day[day])
    vol = sum(orders[o]["cube"] for o in orders_by_day[day])
    print(f"  {day:<6} {n:>7} {vol:>14,}")

print()
print("  Q2 Fleet Breakdown:")
print(f"  ST-required  :  {len(st_required):>3}  (must use Straight Truck)")
print(f"  Van-forced   :  {len(van_forced):>3}  (cube > 1400, must use Van)")
print(f"  Flexible     :  {len(flexible):>3}  (can use either vehicle)")
print("═" * 50)