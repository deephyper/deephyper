min_budget = 30
min_competing = 0
min_fully_completed = 0
reduction_factor = 3
min_early_stopping_rate = 0

for rung in range(5):
    val = min_budget * reduction_factor ** (min_early_stopping_rate + rung)
    print(rung, val)
