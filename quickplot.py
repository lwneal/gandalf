import plotting

fn = '/mnt/results/super_mnist/eval_epoch_0101.json'
plotting.compare_active_learning(fn, fn, prefix='grid', statistic='classification_accuracy')
plotting.compare_active_learning(fn, fn, prefix='grid', statistic='openset_accuracy')
