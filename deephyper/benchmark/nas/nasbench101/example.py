import random
from deephyper.benchmark.nas.nasbench101.util import create_search_space, evaluate_ops

ss = create_search_space()

for i in range(100):
    ops = [random.random() for _ in range(ss.num_nodes)]

    val_acc = evaluate_ops(ss, ops)

    print(f"{i} - val_acc: {val_acc}")