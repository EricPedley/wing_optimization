"""Maps the short prop labels used in T-Motor bench-sweep CSVs to the full
names in prop_datasheets.csv, so a fitting script can look up real
(diameter, pitch, blade_count) geometry for each bench row.

Kept as a small standalone module (not touching prop_datasheets.csv's schema)
since the bench CSVs' labels are vendor shorthand ("GF3016") rather than this
repo's fuller catalogue names ("Gemfan Hurricane 3016").
"""

PROP_NAME_TO_CATALOGUE = {
    "GF1635-3": "Gemfan GF1635-3",
    "GF1636-4": "Gemfan GF1636-4",
    "GF1636-3(?)": "Gemfan GF1635-3",  # orphaned row, uncertain -- see tmotor_m1103_throttle_sweep.csv note
    "GF2015-2": "Gemfan GF2015-2",
    "GF2023-3": "Gemfan Hurricane 2023-3",
    "GF65MMS-2": "Gemfan GF65MMS-2",
    "GF63MM-3": "Gemfan GF63MM-3 Flash",
    "GFD63-3": "Gemfan GF63MM-3 Flash",
    "GF3018-2": "Gemfan Hurricane 3018-2",
    "G3018-2": "Gemfan Hurricane 3018-2",
    "GF3016": "Gemfan Hurricane 3016",
    "GF2020-4": "Gemfan GF2020-4",
    "GF2020-4(?)": "Gemfan GF2020-4",
    "GF3028-3": "Gemfan GF3028-3 WinDancer",
    "M12199-3": "T-Motor M12199-3",
    "GF1610-2": "Gemfan GF1610-2",
    "GF35mm-3": "Gemfan GF35mm-3",
    "GF1608-3": "Gemfan GF1608-3",
    "GF1609-4": "Gemfan GF1609-4",
    "GF2540": "Gemfan Flash 2540",
    "HQ3020": "HQProp T3x2x3",
    "HQ3018": "HQProp T3x1.8x3 3018",
    "HQ3018(?)": "HQProp T3x1.8x3 3018",
}
