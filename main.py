import pm4py
from config import *
from preprocessing import *
from discovery_eval import *
import pandas as pd


def main():
    ensure_dir(OUT_DIR+"/raw")
    ensure_dir(OUT_DIR+"/filtered")

    raw_log = pm4py.read_xes(LOG_PATH)

    # overview of raw log
    data_overview(raw_log, "RAW LOG OVERVIEW")
    df_raw = pm4py.convert_to_dataframe(raw_log)
    raw_cases = df_raw["case:concept:name"].nunique()
    raw_events = len(df_raw)
    raw_activities = set(df_raw["concept:name"].unique())

    # filter log to keep only top variants covering 90% of cases
    filtered_log = keep_top_variants(raw_log)

    # overview of filtered log
    data_overview(filtered_log, "FILTERED LOG OVERVIEW")
    df_filtered = pm4py.convert_to_dataframe(filtered_log)
    filtered_cases = df_filtered["case:concept:name"].nunique()
    filtered_events = len(df_filtered)
    filtered_activities = set(df_filtered["concept:name"].unique())

    # summary of filtering
    print(f"Cases kept: {filtered_cases} / {raw_cases} "
          f"({round(filtered_cases / raw_cases * 100, 2)}%)")
    print(f"Events kept: {filtered_events} / {raw_events} "
          f"({round(filtered_events / raw_events * 100, 2)}%)")
    removed_activities = raw_activities - filtered_activities
    print("\nRemoved activities:")
    for act in sorted(removed_activities):
        print("-", act)

    # discovery and evaluation of models on raw log
    models = discover_three_models(raw_log)
    save_dfg(raw_log, OUT_DIR+"/raw")
    save_models(models, OUT_DIR+"/raw")
    results_raw = evaluate_all(raw_log, models, OUT_DIR+"/raw")

    models = discover_three_models(filtered_log)
    save_dfg(filtered_log, OUT_DIR+"/filtered")
    save_models(models, OUT_DIR+"/filtered")
    results_filtered = evaluate_all(raw_log, models, OUT_DIR+"/filtered")

    # put all results in a dataframe and save as csv
    rows = []
    for alg in results_raw.keys():
        row = {
            "algorithm": alg,
            "raw_fitness": results_raw[alg]["fitness"],
            "raw_precision": results_raw[alg]["precision"],
            "raw_generalization": results_raw[alg]["generalization"],
            "raw_simplicity": results_raw[alg]["simplicity"],
            "filtered_fitness": results_filtered[alg]["fitness"],
            "filtered_precision": results_filtered[alg]["precision"],
            "filtered_generalization": results_filtered[alg]["generalization"],
            "filtered_simplicity": results_filtered[alg]["simplicity"],
        }
        rows.append(row)
    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "metrics_summary.csv")
    results_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
