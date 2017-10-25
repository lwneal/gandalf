#!/bin/bash
TARGET_DIR=mnist_28x28_e24de477
ORACLE_DIR=mnist_28x28_e2a0b34a
CLASS_COUNT=10

echo "Running Experiment 1 with target directory $TARGET_DIR and oracle $ORACLE_DIR..."
echo "WARNING: Deleting all active learning labels in $TARGET_DIR and restarting experiment in 5..."
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

