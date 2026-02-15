import os
import pm4py
import matplotlib.pyplot as plt


# save visualization of the dfg
def save_dfg(event_log, out_path):
    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
    pm4py.save_vis_dfg(
        dfg,
        start_activities,
        end_activities,
        os.path.join(out_path, "dfg.png")
    )


# discovery of the petri net with three algorithms
def discover_three_models(event_log):
    net_a, im_a, fm_a = pm4py.discover_petri_net_alpha(event_log)
    net_h, im_h, fm_h = pm4py.discover_petri_net_heuristics(event_log)
    net_i, im_i, fm_i = pm4py.discover_petri_net_inductive(event_log)
    return {
        "alpha": (net_a, im_a, fm_a),
        "heuristics": (net_h, im_h, fm_h),
        "inductive": (net_i, im_i, fm_i),
    }


# save visualizations of the models
def save_models(models: dict, out_dir: str):
    for name, (net, im, fm) in models.items():
        pm4py.save_vis_petri_net(
            net, im, fm, os.path.join(out_dir, f"{name}.png"))


# evaluation of a model with fitness, precision, generalization, simplicity
def evaluate_model(event_log, net, im, fm):
    fitness = pm4py.fitness_token_based_replay(
        event_log, net, im, fm)["log_fitness"]
    precision = pm4py.precision_token_based_replay(event_log, net, im, fm)
    generalization = pm4py.algo.evaluation.generalization.algorithm.apply(
        event_log, net, im, fm)
    simplicity = pm4py.algo.evaluation.simplicity.algorithm.apply(net)
    return {
        "fitness": float(fitness),
        "precision": float(precision),
        "generalization": float(generalization),
        "simplicity": float(simplicity),
    }


# evaluate all models and return results
def evaluate_all(event_log, models: dict, out_dir):
    results = {}
    for name, (net, im, fm) in models.items():
        metrics = evaluate_model(event_log, net, im, fm)
        results[name] = metrics
        save_metrics(name, metrics, out_dir)
    return results


# save metrics as bar chart
def save_metrics(name, metrics, out_dir):
    keys = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure()
    plt.bar(keys, values, color=colors)
    plt.ylim(0, 1)
    plt.title(name)
    plt.savefig(os.path.join(out_dir, f"{name}_metrics.png"))
    plt.close()
