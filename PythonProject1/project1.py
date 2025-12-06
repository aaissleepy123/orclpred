import numpy as np
import math
import pandas as pd
from collections import defaultdict

# -----------------------------
# Load TSLA CSV
# -----------------------------
TSLA_chart = pd.read_csv("TSLA.csv", skiprows=3, dtype=str)

date_col = "Dates"
vol_col = "Volume"

df = TSLA_chart.rename(columns={date_col: "timestamps", vol_col: "volume"})
df["timestamps"] = pd.to_datetime(df["timestamps"], format="%m/%d/%y %H:%M", errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

print("TSLA head:\n", df.head(3))

# -----------------------------
# Preprocess volume intervals
# -----------------------------
mask = (
    ((df["timestamps"].dt.hour == 9) & (df["timestamps"].dt.minute >= 30))
    | (df["timestamps"].dt.hour == 10)
    | ((df["timestamps"].dt.hour == 11) & (df["timestamps"].dt.minute <= 29))
)
df2 = df.loc[mask].copy()

all_daily_interval_volumes = []
grouped_by_day = df2.groupby(df2["timestamps"].dt.date)

for _, group in grouped_by_day:
    day_sorted = group.sort_values("timestamps")
    if len(day_sorted) >= 120:
        first_two_hours_volume = day_sorted["volume"].to_list()[:120]
        for j in range(12):
            start = j * 10
            ten_minute_interval = first_two_hours_volume[start:start + 10]
            all_daily_interval_volumes.append(ten_minute_interval)

days_half = len(all_daily_interval_volumes) // 2
volumes = all_daily_interval_volumes[:days_half]

print("Number of 10 minute intervals:", len(volumes))
print("Each interval length:", len(volumes[0]))

# -----------------------------
# Scoring function
# -----------------------------
def total_score(volume, schedule):
    """
    Compute schedule score against historical volumes.
    Caps trades at 1% of interval volume.
    """
    T = len(schedule)
    total_score = 0

    for interval in volume:
        if len(interval) != T:
            raise TypeError(f"each volume interval must have {T} elements")

        interval_score = 0
        for i in range(T):
            interval_score += min(schedule[i], interval[i] / 100)
        total_score += interval_score

    return total_score / len(volume)


# -----------------------------
# Memory-based DP (bucketed)
# -----------------------------
def optimal_trade_schedule_with_memory(N, T, alpha, pi, M_bins=100):
    """
    Sparse DP with bucketed memory for tractability.
    M (liquidity memory) is bucketed into M_bins ranges.
    """

    # Define bin edges for M
    M_max = N
    bin_edges = np.linspace(0, M_max, M_bins + 1)

    def bucket(M):
        """Find the bin index for a given M"""
        return int(np.searchsorted(bin_edges, M, side="right") - 1)

    # dp[t] = dict mapping (m, M_bucket) -> max executed trades
    dp = [defaultdict(lambda: -1) for _ in range(T + 1)]
    choice = [{} for _ in range(T)]

    # Base case
    dp[T][(0, bucket(0))] = 0

    # Work backwards
    for t in range(T - 1, -1, -1):
        remaining_steps = T - t
        for (m_next, M_bucket_next), future_val in dp[t + 1].items():
            # Try all possible n that could have led to m_next
            for n in range(m_next, N + 1):
                m = m_next + n
                if m > N:
                    continue

                # Reconstruct approximate previous M
                for trialM in np.linspace(0, M_max, M_bins):
                    traded = math.ceil((1 - alpha * (trialM**pi)) * n)
                    newM = math.ceil(0.1 * trialM + 0.9 * n)
                    if bucket(newM) != M_bucket_next:
                        continue

                    val = traded + future_val
                    key = (m, bucket(trialM))
                    if val > dp[t][key]:
                        dp[t][key] = val
                        choice[t][key] = (n, newM)

    # Reconstruct schedule (start with N shares, M=0)
    n_schedule = []
    successful_trades = 0
    m, M = N, 0
    for t in range(T):
        key = (m, bucket(M))
        if key not in choice[t]:
            break
        n, newM = choice[t][key]
        traded_now = math.ceil((1 - alpha * (M**pi)) * n)
        successful_trades += traded_now
        n_schedule.append(n)
        m -= n
        M = newM

    return n_schedule, dp[0].get((N, bucket(0)), -1), successful_trades


# -----------------------------
# Pi sweep
# -----------------------------
def pi_maximize_total_s(N, T, alpha):
    pi_values = np.linspace(0.3, 0.7, num=5)
    total_scores = []
    for pi in pi_values:
        n_schedule, dp_val, successful_trades = optimal_trade_schedule_with_memory(N, T, alpha, pi)
        score = total_score(volumes, n_schedule)
        total_scores.append(score)
    max_total_score = np.max(total_scores)
    return max_total_score


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    N2, T2 = 10000, 10
    alpha2, pi2 = 0.001, 0.5
    schedule, dp_val, total_traded = optimal_trade_schedule_with_memory(N2, T2, alpha2, pi2)

    print("Trade schedule:", schedule)
    print("DP[0, N]:", dp_val)
    print("Total traded:", total_traded)

    TOTAL_SCORE = pi_maximize_total_s(N=100000, T=10, alpha=0.001)
    print("Max total score across pi sweep:", TOTAL_SCORE)
