import plotting

fn1 = '/mnt/results/super_mnist/method_1/eval_epoch_0101.json'
fn2 = '/mnt/results/super_mnist/method_2/eval_epoch_0101.json'
"""
plotting.compare_active_learning(fn1, fn2,
        prefix='grid',
        statistic='classification_accuracy',
        title='MNIST 6-class test accuracy',
        this_name="Method 1: K+1 Softmax",
        baseline_name="Method 2: Hinge Loss")
"""
plotting.compare_active_learning(fn1, fn2,
        prefix='grid',
        statistic='openset_auc',
        title='MNIST 6-class test accuracy',
        this_name="Method 1: K+1 Softmax",
        baseline_name="Method 2: Hinge Loss")
