import plotting

fn1 = '/mnt/results/super_mnist/method_1/eval_epoch_0101.json'
fn2 = '/mnt/results/super_mnist/method_2/eval_epoch_0101.json'
plotting.compare_active_learning(fn1, fn2, prefix='grid', statistic='classification_accuracy')
#plotting.compare_active_learning(fn, fn, prefix='grid', statistic='openset_accuracy')
