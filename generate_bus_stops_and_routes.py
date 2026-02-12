#!/usr/bin/env python3
"""
generate_bus_stops_and_routes.py

Reads bus_6x6_with_ped.net.xml and generates:
 - bus_stops.add.xml   (additional file with <busStop .../>)
 - pt_routes.rou.xml   (routes file with pt routes & vehicles that stop)

Heuristic:
 - choose non-internal edges
 - pick 'horizontal' edges (abs(dx) > abs(dy))
 - place a stop on the first lane of the edge (startPos ~= 10% of lane length)
 - create a route per row by chaining horizontal edges sorted by x
"""

import sumolib
import xml.etree.ElementTree as ET
from collections import defaultdict
import math

NETFILE = "bus_6x6_tls.net.xml"
OUT_STOPS = "bus_stops.add.xml"
OUT_ROUTES = "pt_routes.rou.xml"

print("Reading net:", NETFILE)
net = sumolib.net.readNet(NETFILE, withInternal=False)  # skip internal edges

# collect candidate edges and group by approximate Y (rows)
rows = defaultdict(list)

for edge in net.getEdges():
    eid = edge.getID()
    # skip internal / special edges (sumolib readNet withInternal=False should help, but keep safe)
    if eid.startswith(":"):
        continue
    # compute approximate direction from lane shape
    # get center coords of edge shape
    shape = edge.getShape()  # list of (x,y)
    if not shape or len(shape) < 2:
        continue
    x0, y0 = shape[0]
    x1, y1 = shape[-1]
    dx = x1 - x0
    dy = y1 - y0
    # group by rounded y to form rows (tolerance ~50m)
    row_key = round(((y0 + y1)/2) / 50) * 50
    # mark horizontal-like edges (so route rows are mostly continuous)
    if abs(dx) >= abs(dy):
        lanes = edge.getLanes()
        if lanes:
            lane = lanes[0]
            lane_len = lane.getLength()
            rows[row_key].append((min(x0,x1), edge, lane, lane_len))

# write busStop additional file
add_root = ET.Element("additional")
stop_count = 0
for row_key in sorted(rows.keys()):
    # sort edges left->right
    entries = sorted(rows[row_key], key=lambda t: t[0])
    for i, (x, edge, lane, lane_len) in enumerate(entries):
        # place stop at 10% along lane, length 10-20m
        startPos = max(2.0, lane_len * 0.10)
        stopLength = min(12.0, lane_len * 0.12)
        endPos = startPos + stopLength
        stop_id = f"bs_{edge.getID()}"
        bs_elem = ET.SubElement(add_root, "busStop", {
            "id": stop_id,
            "lane": lane.getID(),
            "startPos": f"{startPos:.2f}",
            "endPos": f"{endPos:.2f}"
        })
        stop_count += 1

# save additional
tree = ET.ElementTree(add_root)
tree.write(OUT_STOPS, encoding="utf-8", xml_declaration=True)
print(f"Wrote {OUT_STOPS} with {stop_count} busStops")

# build simple horizontal routes (one route per row)
routes_root = ET.Element("routes")
# define a bus vehicle type
vtype = ET.SubElement(routes_root, "vType", {
    "id": "bus",
    "vClass": "bus",
    "length": "12.0",
    "width": "2.5",
    "accel": "1.0",
    "decel": "3.0",
    "color": "1,0,0"
})

route_count = 0
vehicle_count = 0
for r_i, row_key in enumerate(sorted(rows.keys())):
    entries = sorted(rows[row_key], key=lambda t: t[0])
    if len(entries) < 2:
        continue
    # build edge sequence
    edge_ids = [e.getID() for (_, e, _, _) in entries]
    route_id = f"line_row_{r_i}"
    route_elem = ET.SubElement(routes_root, "route", {
        "id": route_id,
        "edges": " ".join(edge_ids)
    })
    route_count += 1

    # create a repeating vehicle flow using that route (simple periodic buses)
    # each vehicle will include <stop busStop="..." duration="..."/> entries referencing the stops we created
    # for simplicity, add stops in same order as edges
    for k in range(0, len(edge_ids), max(1, len(edge_ids)//3)):  # place 3 stops per route-ish
        pass

    # Create one vehicle with explicit stops at every route edge's stop
    veh = ET.SubElement(routes_root, "vehicle", {
        "id": f"bus_{route_id}_0",
        "type": "bus",
        "route": route_id,
        "depart": "0"
    })
    # add stop elements referencing busStop ids
    for (_, edge, lane, lane_len) in entries:
        stop_id = f"bs_{edge.getID()}"
        s = ET.SubElement(veh, "stop", {"busStop": stop_id, "duration": "10"})
    vehicle_count += 1

# write routes file
tree2 = ET.ElementTree(routes_root)
tree2.write(OUT_ROUTES, encoding="utf-8", xml_declaration=True)
print(f"Wrote {OUT_ROUTES} with {route_count} routes and {vehicle_count} vehicles")
