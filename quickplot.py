import plotting

fn1 = '/mnt/results/super_mnist/eval_epoch_0101.json'
fn2 = '/mnt/results/super_mnist/method_4_hinge_with_margin/eval_epoch_0101.json'
kwargs = dict(
    title='MNIST Accuracy (0-5)',
    this_name="Hinge Only",
    baseline_name="Method 4: Two-Sided Hinge Loss w/ Margin"
)
plotting.compare_active_learning(fn1, fn2,
        prefix='grid',
        statistic='classification_accuracy', **kwargs)
"""
plotting.compare_active_learning(fn1, fn2,
        prefix='grid',
        statistic='openset_auc',
        **kwargs)
"""
