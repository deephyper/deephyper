import os
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


# def _bebop_nodelist(node_str):
#     node_str = os.environ["SLURM_JOB_NODELIST"]
#     node_ids = []
#     # string like: bdw-[0123,0124-0126]
#     # NOTE: the following is not yet generic enough,
#     # technically we could have prefix other than bdw- when
#     # working with a different type of node, and we could
#     # have mixed prefixes (and not have the nodes in brakets also)/
#     # Something like this could be possible:
#     # bdw-[0123,0124-0126],bdwd-1239
#     ranges = node_str.split('[')[1].split(']')[0].split(',')
#     for node_range in ranges:
#         lo, *hi = node_range.split("-")
#         lo = int(lo)
#         if hi:
#             hi = int(hi[0])
#             node_ids.extend(list(range(lo, hi + 1)))
#         else:
#             node_ids.append(lo)
#     return [f"bdw-{node_id:04d}" for node_id in node_ids]

# HOSTNAME=thetamom1
# COBALT_BLOCKNAME=3824-3827
# COBALT_PARTNAME=3824-3827
# COBALT_JOBSIZE=4
# COBALT_PARTSIZE=4
# COBALT_JOBID=550270
# COBALT_BLOCKSIZE=4
# COBALT_NODEFILE=/var/tmp/cobalt.550270


def expand_nodelist(system, node_str):

    hostname = socket.gethostname()
    if "theta" in hostname:
        l = _theta_nodelist(node_str)
    else:
        l = [node_str]

    return l


#

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("No argument provided")

    l = expand_nodelist(sys.argv[1], sys.argv[2])
    l = str(l).replace(", ", " ").replace("'", "")
    print(l)
