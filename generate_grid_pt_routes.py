#!/usr/bin/env python3
import os
import sys
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import sumolib

# ---------------- CONFIG ----------------
NETFILE = "bus_5x5_250_with_ped_staticTLS.net.xml"   # change to your net filename
STOPFILE = "bus_5x5.add.xml"                   # change to your stops file
OUT_ROUTES = "bus_routes_fixed_5x5.rou.xml"

BUS_VTYPE_ID = "bus"
BUS_CAPACITY = "85"
BUS_DWELL_S = 10

SIM_BEGIN = 0
SIM_END = 3600
HEADWAY_S = 300          # 5 min

INCLUDE_STOPS = True
EXPECTED_GRID_N = 5      # 5x5 junction grid
# ----------------------------------------


def read_busstop_map(stopfile: str) -> dict[str, str]:
    """
    Map directed edge_id -> busStop_id, derived from busStop@lane.
    lane="A2B2_1" -> edge "A2B2"
    """
    tree = ET.parse(stopfile)
    root = tree.getroot()
    edge_to_stop = {}
    for bs in root.findall("busStop"):
        bs_id = bs.get("id")
        lane_id = bs.get("lane")
        if not bs_id or not lane_id:
            continue
        edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
        # First one wins (adjust if you ever put multiple stops per edge)
        edge_to_stop.setdefault(edge_id, bs_id)
    return edge_to_stop


def cluster(values, tol):
    values = sorted(values)
    clusters = []
    cur = []
    for v in values:
        if not cur or abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(sum(cur) / len(cur))
            cur = [v]
    if cur:
        clusters.append(sum(cur) / len(cur))
    return clusters


def snap(v, centers):
    return min(range(len(centers)), key=lambda i: abs(v - centers[i]))


def build_grid_nodes(net: sumolib.net.Net, n: int) -> dict[tuple[int, int], sumolib.net.node.Node]:
    """Index nodes into an n×n grid based on coordinates."""
    nodes = [nd for nd in net.getNodes() if not nd.getID().startswith(":")]
    xs = [nd.getCoord()[0] for nd in nodes]
    ys = [nd.getCoord()[1] for nd in nodes]

    xspread = max(xs) - min(xs) if len(xs) > 1 else 1.0
    yspread = max(ys) - min(ys) if len(ys) > 1 else 1.0

    xspacing = xspread / max(1, (n - 1))
    yspacing = yspread / max(1, (n - 1))
    tol = 0.30 * min(xspacing, yspacing)

    x_centers = cluster(xs, tol)
    y_centers = cluster(ys, tol)

    if len(x_centers) != n or len(y_centers) != n:
        # relax tolerance once
        tol = 0.45 * min(xspacing, yspacing)
        x_centers = cluster(xs, tol)
        y_centers = cluster(ys, tol)

    if len(x_centers) != n or len(y_centers) != n:
        raise RuntimeError(f"Could not infer a {n}x{n} grid (got {len(x_centers)}x{len(y_centers)}).")

    grid = {}
    for nd in nodes:
        x, y = nd.getCoord()
        ix = snap(x, x_centers)
        iy = snap(y, y_centers)
        grid[(ix, iy)] = nd

    missing = [(ix, iy) for ix in range(n) for iy in range(n) if (ix, iy) not in grid]
    if missing:
        raise RuntimeError(f"Missing grid nodes: {missing}")

    return grid


def find_edge(a: sumolib.net.node.Node, b: sumolib.net.node.Node) -> sumolib.net.edge.Edge:
    """Find the directed (non-internal) edge from a->b."""
    for e in a.getOutgoing():
        if e.getID().startswith(":"):
            continue
        if e.getToNode().getID() == b.getID():
            return e
    raise RuntimeError(f"No directed edge from {a.getID()} to {b.getID()}")


def node_path_to_edges(node_path):
    edge_ids = []
    for i in range(len(node_path) - 1):
        edge_ids.append(find_edge(node_path[i], node_path[i + 1]).getID())
    return edge_ids


def make_perimeter_node_path(grid, n):
    """Clockwise perimeter starting at top-right (n-1,n-1)."""
    path = [grid[(n - 1, n - 1)]]
    # top row right->left
    for ix in range(n - 2, -1, -1):
        path.append(grid[(ix, n - 1)])
    # left col top->bottom
    for iy in range(n - 2, -1, -1):
        path.append(grid[(0, iy)])
    # bottom row left->right
    for ix in range(1, n):
        path.append(grid[(ix, 0)])
    # right col bottom->top (back to start)
    for iy in range(1, n):
        path.append(grid[(n - 1, iy)])
    return path


def add_route(parent, route_id, edge_ids, edge_to_stop):
    r = ET.SubElement(parent, "route", {"id": route_id, "edges": " ".join(edge_ids)})
    if INCLUDE_STOPS:
        for eid in edge_ids:
            bs = edge_to_stop.get(eid)
            if bs:
                ET.SubElement(r, "stop", {"busStop": bs, "duration": str(BUS_DWELL_S)})


def add_flow(parent, flow_id, route_id):
    ET.SubElement(parent, "flow", {
        "id": flow_id,
        "type": BUS_VTYPE_ID,
        "route": route_id,
        "begin": str(SIM_BEGIN),
        "end": str(SIM_END),
        "period": str(HEADWAY_S),
    })


def main():
    if not os.path.exists(NETFILE):
        print(f"ERROR: net not found: {NETFILE}")
        sys.exit(1)
    if not os.path.exists(STOPFILE):
        print(f"ERROR: stops not found: {STOPFILE}")
        sys.exit(1)

    net = sumolib.net.readNet(NETFILE, withInternal=False)
    edge_to_stop = read_busstop_map(STOPFILE)

    grid = build_grid_nodes(net, EXPECTED_GRID_N)
    n = EXPECTED_GRID_N
    mid = n // 2

    routes_root = ET.Element("routes")

    # vType
    ET.SubElement(routes_root, "vType", {
        "id": BUS_VTYPE_ID,
        "vClass": "bus",
        "personCapacity": BUS_CAPACITY,
        "accel": "1.0",
        "decel": "3.0",
        "length": "12.0",
    })

    # -------- Perimeter (CW + CCW) --------
    perim_nodes_cw = make_perimeter_node_path(grid, n)
    perim_edges_cw = node_path_to_edges(perim_nodes_cw)

    # CCW is NOT "reverse(edge_ids)". It is the reverse NODE PATH, then compute edges.
    perim_nodes_ccw = list(reversed(perim_nodes_cw))
    perim_edges_ccw = node_path_to_edges(perim_nodes_ccw)

    add_route(routes_root, "perimeter_cw", perim_edges_cw, edge_to_stop)
    add_route(routes_root, "perimeter_ccw", perim_edges_ccw, edge_to_stop)

    # -------- Trunks (each direction computed from node path) --------
    # Horizontal trunk: left->right along middle row
    h_nodes_lr = [grid[(ix, mid)] for ix in range(0, n)]
    h_nodes_rl = list(reversed(h_nodes_lr))
    h_edges_lr = node_path_to_edges(h_nodes_lr)
    h_edges_rl = node_path_to_edges(h_nodes_rl)

    add_route(routes_root, "trunk_h_lr", h_edges_lr, edge_to_stop)
    add_route(routes_root, "trunk_h_rl", h_edges_rl, edge_to_stop)

    # Vertical trunk: bottom->top along middle column
    v_nodes_bt = [grid[(mid, iy)] for iy in range(0, n)]
    v_nodes_tb = list(reversed(v_nodes_bt))
    v_edges_bt = node_path_to_edges(v_nodes_bt)
    v_edges_tb = node_path_to_edges(v_nodes_tb)

    add_route(routes_root, "trunk_v_bt", v_edges_bt, edge_to_stop)
    add_route(routes_root, "trunk_v_tb", v_edges_tb, edge_to_stop)

    # Flows
    add_flow(routes_root, "f_perimeter_cw", "perimeter_cw")
    add_flow(routes_root, "f_perimeter_ccw", "perimeter_ccw")
    add_flow(routes_root, "f_trunk_h_lr", "trunk_h_lr")
    add_flow(routes_root, "f_trunk_h_rl", "trunk_h_rl")
    add_flow(routes_root, "f_trunk_v_bt", "trunk_v_bt")
    add_flow(routes_root, "f_trunk_v_tb", "trunk_v_tb")

    ET.ElementTree(routes_root).write(OUT_ROUTES, encoding="utf-8", xml_declaration=True)
    print(f"Wrote: {OUT_ROUTES}")
    print("Note: XML is compact (one line).")


if __name__ == "__main__":
    main()
