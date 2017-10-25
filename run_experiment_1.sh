#!/bin/bash
TARGET_DIR=$1
ORACLE_DIR=$2
MODE=$3
USE_TRAJECTORIES=$4
CLASS_COUNT=10

echo "Running Experiment 1:"
echo "  TARGET_DIR $TARGET_DIR"
echo "  ORACLE_DIR $ORACLE_DIR"
echo "  MODE $MODE"
echo "  TRAJECTORIES $USE_TRAJECTORIES "
echo ""
echo "WARNING: Deleting all active learning labels in $TARGET_DIR and beginning experiment 1 in 5 seconds..."
sleep 5

# Delete all trajectories and labels
rm -R /mnt/results/$TARGET_DIR/trajectories
rm -R /mnt/results/$TARGET_DIR/labels

# Reset active classifier
rm /mnt/results/$TARGET_DIR/active_learning_classifier_*.pth


echo "Generating $CLASS_COUNT seed labels to initialize..."
for i in `seq 0 9`; do
    python experiments/generate_counterfactual.py --result_dir /mnt/results/$TARGET_DIR --mode active --start_class $i --target_class $i --speed 0
done
python experiments/oracle.py --result_dir /mnt/results/$TARGET_DIR --oracle_result_dir /mnt/results/$ORACLE_DIR
python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR 

echo "Generating additional labels using counterfactual uncertainty sampling"
for i in `seq $CLASS_COUNT 3 150`; do
    echo "Generating labels $i / 150"
    for j in `seq 3`; do
        python experiments/generate_counterfactual.py --result_dir /mnt/results/$TARGET_DIR --mode uncertainty
    done
    python experiments/oracle.py --result_dir /mnt/results/$TARGET_DIR --oracle_result_dir /mnt/results/$ORACLE_DIR
    python experiments/train_active_classifier.py --result_dir /mnt/results/$TARGET_DIR 
done

