How to process tor .pcaps
== Req. ==
python 3.6.7 (i tested only this version but newer versions should be ok too)
Pyhton Libraries: see requirements.txt file.


1. We first need a .pkl file with entry Nodes. ==
To create a .pkl file with nodes you can do the following in a python shell:
```pyhton
import sys
sys.path.append('PATH_TO_NodeGuard.py')
import NodeGuard
nodes = NodeGuard.NodeGuard()
nodes.loadNodes()
nodes.saveNodesToFile("nodes")
```
File nodes.pkl contains all entry nodes ready.

2. Main function to process .pcaps is ProcessPcapTor (in util.py) ==
util.ProcessPcapTor(PCAP_FILE, output_folder=OUTPUT_FOLDER, label=APP_NAME, category=APP_CATEGORY,
                    Timeout=FLOW_TIMEOUT, ActivityTimeout=ACTIVITY_TIMEOUT, TorPickle=NODES.pkl)
You can edit and use tor_pcaps_to_csv_flows.py to batch process more files at once.

3. MachineLearning & app recognition ==
You can use run_ml.py, just edit the following varibles:
csv_path: path to csvs from 1.
output_filename_prefix: output files file name prefix
flow_timeout: flow timeout value
activity_timeout: activity timeout ValueError
Output: * a multipage .pdf that contain Confusion Matrices
        * some .csv files *CLASSIFIER_micro_macro.csv, it contains micro and macro results for the CLASSIFIER
        * some .csv files *CLASSIFIER.csv, it contains classifier performances per app and TP,TN,FP,FN,
             Precision, Recall, F1, Accuracy
