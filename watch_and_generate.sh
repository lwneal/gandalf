#!/bin/bash
RESULT_DIR=/mnt/results/mnist_standalone_classifier

echo "Watching $RESULT_DIR..."
while true; do
    sleep 3
    LABEL_COUNT=`ls $RESULT_DIR/labels | wc -l`
    TRAJECTORY_COUNT=`ls $RESULT_DIR/trajectories | grep npy$ | wc -l`

    echo Out of $TRAJECTORY_COUNT trajectories, $LABEL_COUNT are labeled

    if (( $TRAJECTORY_COUNT < $(($LABEL_COUNT + 2)) )); then
        python experiments/generate_counterfactual.py --result_dir $RESULT_DIR
    fi
done
