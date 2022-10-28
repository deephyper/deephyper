import sys
import socket


def _theta_nodelist(node_str):
    # string like: 1001-1005,1030,1034-1200
    node_ids = []
    ranges = node_str.split(",")
    lo = None
    hi = None
    for node_range in ranges:
        lo, *hi = node_range.split("-")
        lo = int(lo)
        if hi:
            hi = int(hi[0])
            node_ids.extend(list(range(lo, hi + 1)))
        else:
            node_ids.append(lo)
    return [f"nid{node_id:05d}" for node_id in node_ids]


def expand_nodelist(system, node_str):

    hostname = socket.gethostname()
    if "theta" in hostname:
        node_list = _theta_nodelist(node_str)
    else:
        node_list = [node_str]

    return node_list


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("No argument provided")

    node_list = expand_nodelist(sys.argv[1], sys.argv[2])
    node_list = str(node_list).replace(", ", " ").replace("'", "")
    print(node_list)
