import os

# Adapted from 'get_job_nodelist()' found in the following project:
# https://github.com/argonne-lcf/balsam/blob/main/balsam/platform/compute_node/alcf_thetaknl_node.py


def nodelist():
    """Get all compute nodes allocated in the current job context.

    :meta private:
    """
    node_str = os.environ["COBALT_PARTNAME"]
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

    print([f"nid{node_id:05d}" for node_id in node_ids])


if __name__ == "__main__":
    nodelist()
