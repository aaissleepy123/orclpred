import numpy as np
import math

def optimal_trade_schedule(N, T, alpha, pi):
    dp = np.zeros((T+1, N+1))
    choice = np.zeros((T, N+1), dtype=int)

    # Backward DP
    for t in range(T-1, -1, -1):
        remaining_steps = T - t
        for m in range(N+1):
            max_n = math.ceil(m / remaining_steps)
            n_range = np.arange(max_n+1)  # candidate trades

            traded = np.ceil((1 - alpha * (n_range**pi)) * n_range)
            future = dp[t+1, m - n_range]

            vals = traded + future
            idx = np.argmax(vals)

            dp[t, m] = vals[idx]
            choice[t, m] = n_range[idx]

    # Reconstruct optimal schedule and compute total traded
    n_schedule = []
    successful_trades = 0
    m = N
    for t in range(T):
        trade = choice[t, m]
        n_schedule.append(int(trade))
        traded_now = math.ceil((1 - alpha * (trade**pi)) * trade)
        successful_trades += traded_now
        m -= trade

    return n_schedule, dp[0, N], successful_trades


# Example
if __name__ == "__main__":
    N2, T2 = 10000, 10
    alpha2, pi2 = 0.001, 0.5

    schedule, dp_val, total_traded = optimal_trade_schedule(N2, T2, alpha2, pi2)
    print("Trade schedule:", schedule)
    print("DP[0, N]:", dp_val)
    print("Total traded:", total_traded)
