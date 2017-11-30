import plotting

fn1 = '/mnt/results/super_mnist/eval_epoch_0101.json'
fn2 = '/mnt/results/super_mnist/method_5_training_gt/eval_epoch_0101.json'
kwargs = dict(
    this_name="Training with Counterfactuals",
    baseline_name="Training with Ground Truth"
)
"""
plotting.compare_active_learning(fn1, fn2,
        title='MNIST Accuracy (0-5)',
        prefix='grid',
        statistic='classification_accuracy', **kwargs)
"""
plotting.compare_active_learning(fn1, fn2,
        title='Open Set AUC (MNIST 0-5 vs 6-9)',
        prefix='grid',
        statistic='openset_auc',
        **kwargs)
