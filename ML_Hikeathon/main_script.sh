#!/bin/bash
pip install -r requirement.txt
python ./code/undirected_graph_features.py
python ./code/directed_degrees_feature_creation.py
python ./code/neighbours_features.py
python ./code/leak_analysis.py
python ./code/freq_train_test.py
python ./code/train_script.py
python ./code/test_script.py
