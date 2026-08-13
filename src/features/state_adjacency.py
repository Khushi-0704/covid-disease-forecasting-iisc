"""
state_adjacency.py

Land-border adjacency mapping between Indian states/UTs, used to compute
spatial-lag features (neighboring states' case activity) and spatial
autocorrelation (Moran's I) diagnostics.

Island territories (Andaman and Nicobar Islands, Lakshadweep) have no land
neighbors -- their spatial-lag feature will be NaN, an expected/correct
limitation, not a bug.
"""

STATE_ADJACENCY = {
    "Andaman and Nicobar Islands": [],
    "Andhra Pradesh": ["Telangana", "Odisha", "Chhattisgarh", "Karnataka", "Tamil Nadu"],
    "Arunachal Pradesh": ["Assam", "Nagaland"],
    "Assam": ["Arunachal Pradesh", "Nagaland", "Manipur", "Meghalaya", "Tripura", "Mizoram", "West Bengal"],
    "Bihar": ["Uttar Pradesh", "Jharkhand", "West Bengal"],
    "Chandigarh": ["Punjab", "Haryana"],
    "Chhattisgarh": ["Madhya Pradesh", "Maharashtra", "Telangana", "Andhra Pradesh", "Odisha", "Jharkhand", "Uttar Pradesh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Gujarat", "Maharashtra"],
    "Delhi": ["Haryana", "Uttar Pradesh"],
    "Goa": ["Maharashtra", "Karnataka"],
    "Gujarat": ["Rajasthan", "Madhya Pradesh", "Maharashtra", "Dadra and Nagar Haveli and Daman and Diu"],
    "Haryana": ["Punjab", "Himachal Pradesh", "Uttar Pradesh", "Rajasthan", "Delhi", "Chandigarh"],
    "Himachal Pradesh": ["Jammu and Kashmir", "Punjab", "Haryana", "Uttarakhand", "Ladakh"],
    "Jammu and Kashmir": ["Ladakh", "Himachal Pradesh", "Punjab"],
    "Jharkhand": ["Bihar", "West Bengal", "Odisha", "Chhattisgarh", "Uttar Pradesh"],
    "Karnataka": ["Goa", "Maharashtra", "Telangana", "Andhra Pradesh", "Tamil Nadu", "Kerala"],
    "Kerala": ["Karnataka", "Tamil Nadu", "Puducherry"],
    "Ladakh": ["Jammu and Kashmir", "Himachal Pradesh"],
    "Lakshadweep": [],
    "Madhya Pradesh": ["Uttar Pradesh", "Chhattisgarh", "Maharashtra", "Gujarat", "Rajasthan"],
    "Maharashtra": ["Gujarat", "Madhya Pradesh", "Chhattisgarh", "Telangana", "Karnataka", "Goa", "Dadra and Nagar Haveli and Daman and Diu"],
    "Manipur": ["Nagaland", "Mizoram", "Assam"],
    "Meghalaya": ["Assam"],
    "Mizoram": ["Assam", "Manipur", "Tripura"],
    "Nagaland": ["Assam", "Manipur", "Arunachal Pradesh"],
    "Odisha": ["West Bengal", "Jharkhand", "Chhattisgarh", "Andhra Pradesh"],
    "Puducherry": ["Tamil Nadu", "Kerala"],
    "Punjab": ["Jammu and Kashmir", "Himachal Pradesh", "Haryana", "Rajasthan", "Chandigarh"],
    "Rajasthan": ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Gujarat"],
    "Sikkim": ["West Bengal"],
    "Tamil Nadu": ["Andhra Pradesh", "Karnataka", "Kerala", "Puducherry"],
    "Telangana": ["Maharashtra", "Chhattisgarh", "Andhra Pradesh", "Karnataka"],
    "Tripura": ["Assam", "Mizoram"],
    "Uttar Pradesh": ["Uttarakhand", "Haryana", "Delhi", "Rajasthan", "Madhya Pradesh", "Chhattisgarh", "Jharkhand", "Bihar"],
    "Uttarakhand": ["Himachal Pradesh", "Uttar Pradesh"],
    "West Bengal": ["Odisha", "Jharkhand", "Bihar", "Sikkim", "Assam"],
}


def validate_symmetry():
    issues = []
    for state, neighbors in STATE_ADJACENCY.items():
        for n in neighbors:
            if n not in STATE_ADJACENCY:
                issues.append(f"{n} (neighbor of {state}) not in STATE_ADJACENCY keys")
            elif state not in STATE_ADJACENCY.get(n, []):
                issues.append(f"Asymmetry: {state} lists {n} as neighbor, but not vice versa")
    return issues


if __name__ == "__main__":
    issues = validate_symmetry()
    if issues:
        print("Adjacency issues found:")
        for i in issues:
            print(" -", i)
    else:
        print("Adjacency map is symmetric and internally consistent.")