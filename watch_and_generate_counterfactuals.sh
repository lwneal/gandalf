#!/bin/bash

RESULT_DIR=/mnt/results/super_mnist
COUNTERFACTUAL_COUNT=`ls $RESULT_DIR/trajectories | grep mp4 | wc -l`
LABEL_COUNT=`ls $RESULT_DIR/labels | wc -l`

if [ $COUNTERFACTUAL_COUNT -lt $(($LABEL_COUNT + 5)) ]; then
    echo "Out of $COUNTERFACTUAL_COUNT counterfactuals $LABEL_COUNT have been labeled; generating more"
    python experiments/generate_counterfactual.py --result_dir $RESULT_DIR --strategy uncertainty-nearest --count 6 --counterfactual_max_iters 100
else
    echo "Labeled: $LABEL_COUNT Total: $COUNTERFACTUAL_COUNT"
fi

