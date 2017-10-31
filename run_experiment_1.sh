#!/bin/bash
set -e
set -x
TARGET_DIR=$1
MODE=$2
STRATEGY=$3
RUN_NAME=$4

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 target_dir oracle_dir mode use_trajectories"
    echo "target_dir: a result_dir with a good unsupervised model"
    echo "mode: uncertainty_sampling, counterfactual"
    echo "strategy: random, uncertainty-random, random-nearest, ..."
    echo "run_name: unique name for this run"
    exit
fi

if (echo $TARGET_DIR | grep emnist); then
    echo "EMNIST character dataset detected"
    CLASSES="A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
else
    CLASSES="0 1 2 3 4 5 6 7 8 9"
fi
echo "Generating seed data for $CLASS_COUNT classes"

echo "Running Experiment 1:"
echo "  TARGET_DIR $TARGET_DIR"
echo "  STRATEGY $STRATEGY"
echo "  MODE $MODE"
echo ""
echo "WARNING: Deleting all active learning labels in $TARGET_DIR and beginning experiment 1 in 5 seconds..."
sleep 5

# Delete all trajectories and labels
rm -Rf /mnt/results/$TARGET_DIR/trajectories
rm -Rf /mnt/results/$TARGET_DIR/labels

# Reset active classifier
rm -f /mnt/results/$TARGET_DIR/active_learning_classifier_*.pth

INIT_LABELS=1000

MAX_ITERS=1000
FRAME_COUNT=60
if [[ "$MODE" = "uncertainty_sampling" ]]; then
    FRAME_COUNT=1
    MAX_ITERS=1
fi

# Seed labels are now included within train_active_classifier
python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR --experiment_type $MODE --classifier_name active_learning_classifier --init_label_count $INIT_LABELS

echo "Generating additional labels using sampling mode: $MODE"
for i in `seq 0 5 200`; do
    echo "Generating labels $i"
    python experiments/generate_counterfactual.py --result_dir /mnt/results/$TARGET_DIR --classifier_name active_learning_classifier --counterfactual_frame_count $FRAME_COUNT --counterfactual_max_iters $MAX_ITERS --count 5 --strategy $STRATEGY
    python experiments/oracle.py --result_dir /mnt/results/$TARGET_DIR --oracle_pth /mnt/results/$TARGET_DIR/oracle.pth
    python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR --experiment_type $MODE --classifier_name active_learning_classifier --init_label_count $INIT_LABELS
done
cp /mnt/results/$TARGET_DIR/eval_epoch_0025.json /mnt/results/$TARGET_DIR/eval_$RUN_NAME.json
