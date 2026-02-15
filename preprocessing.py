import os
import pandas as pd
import pm4py


# ensure directory exists
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# get variants and their counts
def variants_stats_fallback(event_log):
    variants = pm4py.get_variants_as_tuples(event_log)
    stats = []
    for var, traces in variants.items():
        count = len(traces) if hasattr(traces, "__len__") and not isinstance(
            traces, int) else int(traces)
        stats.append((var, count))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats  # list[(tuple(activity...), count)]


# keep only the top variants that cover at least the given percentage of cases
def keep_top_variants(event_log, coverage: float = 0.9):
    df = pm4py.convert_to_dataframe(event_log)

    df = df.sort_values(["case:concept:name", "time:timestamp"])
    variants = df.groupby("case:concept:name")["concept:name"].apply(tuple)

    # count variants and keep top ones until coverage is reached
    counts = variants.value_counts()
    total = counts.sum()
    running = 0
    keep_variants = set()
    for var, cnt in counts.items():
        keep_variants.add(var)
        running += cnt
        if running / total >= coverage:
            break

    # keep cases that belong to the selected variants
    keep_cases = variants[variants.isin(
        keep_variants)].index.astype(str).tolist()

    return pm4py.filter_event_attribute_values(
        event_log,
        "case:concept:name",
        keep_cases,
        level="case",
        retain=True
    )


# overview of the log with number of cases, events, activities, time range, top activities and variants
def data_overview(event_log, title: str = "OVERVIEW", top_k: int = 10):
    df = pm4py.convert_to_dataframe(event_log)

    n_cases = df["case:concept:name"].nunique()
    n_events = len(df)
    n_acts = df["concept:name"].nunique()
    t_min, t_max = df["time:timestamp"].min(), df["time:timestamp"].max()

    print(f"\n=== {title} ===")
    print("cases:", n_cases, "| events:", n_events, "| activities:", n_acts)
    print("time range:", t_min, "->", t_max)

    print("\nTop activities:")
    print(df["concept:name"].value_counts().head(top_k).to_string())

    var_stats = variants_stats_fallback(event_log)
    print("\nvariants:", len(var_stats))
    print("top variants (first 5):")
    for var_tuple, cnt in var_stats[:5]:
        print("-", " -> ".join(var_tuple), "| cases:", cnt)

    avg_events = n_events / n_cases if n_cases else 0
    print("\navg events per case:", round(avg_events, 2))
