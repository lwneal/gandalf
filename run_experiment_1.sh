#!/bin/bash
set -e
TARGET_DIR=$1
ORACLE_DIR=$2
MODE=$3

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 target_dir oracle_dir mode use_trajectories"
    echo "target_dir: a result_dir with a good unsupervised model"
    echo "oracle_dir: a result_dir with a high classification accuracy"
    echo "mode: semisupervised, uncertainty_sampling, counterfactual"
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
echo "  ORACLE_DIR $ORACLE_DIR"
echo "  MODE $MODE"
echo "  TRAJECTORIES $USE_TRAJECTORIES "
echo ""
echo "WARNING: Deleting all active learning labels in $TARGET_DIR and beginning experiment 1 in 5 seconds..."
sleep 5

# Delete all trajectories and labels
rm -Rf /mnt/results/$TARGET_DIR/trajectories
rm -Rf /mnt/results/$TARGET_DIR/labels

# Reset active classifier
rm -f /mnt/results/$TARGET_DIR/active_learning_classifier_*.pth


# Seed labels are now included within train_active_classifier

echo "Generating additional labels using sampling mode: $MODE"
for i in `seq 0 5 150`; do
    echo "Generating labels $i / 150"
    for j in `seq 5`; do
        python experiments/generate_counterfactual.py --result_dir /mnt/results/$TARGET_DIR --classifier_name active_learning_classifier
    done
    python experiments/oracle.py --result_dir /mnt/results/$TARGET_DIR --oracle_result_dir /mnt/results/$ORACLE_DIR
    python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR --experiment_type $MODE --classifier_name active_learning_classifier
done

