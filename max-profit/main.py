def max_profit(n):
    buildings = {
        "T": (5, 1500),
        "P": (4, 1000),
        "C": (10, 2000)
    }

    dp = [0] * (n + 1)
    choice = [None] * (n + 1)

    for t in range(1, n + 1):
        for b, (bt, earn) in buildings.items():
            if t >= bt:
                profit = dp[t - bt] + (n - t) * earn
                if profit > dp[t]:
                    dp[t] = profit
                    choice[t] = b

    # Reconstruct solution
    t = max(range(n + 1), key=lambda x: dp[x])
    count = {"T": 0, "P": 0, "C": 0}

    while t > 0 and choice[t]:
        b = choice[t]
        count[b] += 1
        t -= buildings[b][0]

    return dp[max(range(n + 1), key=lambda x: dp[x])], count
