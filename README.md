How to process tor .pcaps
== Req. ==
python 3.6.7 (i tested only this version but newer versions should be ok too)
Pyhton Libraries (these are the tested ones, newer should be ok too):
future                   0.17.1    
lxml                     4.2.5     
matplotlib               3.0.0     
numpy                    1.15.2    
pandas                   0.23.4    
requests                 2.19.1    
scapy                    2.4.0     
scikit-learn             0.20.0    
scipy                    1.1.0     
sklearn-pandas           1.7.0     
urllib3                  1.23      

== 1. We first need a .pkl file with entry Nodes. ==
To create a .pkl file with nodes you can do the following in a python shell:
import sys
sys.path.append('PATH_TO_NodeGuard.py')
import NodeGuard
nodes = NodeGuard.NodeGuard()
nodes.loadNodes()
nodes.saveNodesToFile("nodes")
File nodes.pkl contains all entry nodes ready.

== 2. Main function to process .pcaps is ProcessPcapTor (in util.py) ==
util.ProcessPcapTor(PCAP_FILE, output_folder=OUTPUT_FOLDER, label=APP_NAME, category=APP_CATEGORY,
                    Timeout=FLOW_TIMEOUT, ActivityTimeout=ACTIVITY_TIMEOUT, TorPickle=NODES.pkl)
You can edit and use tor_pcaps_to_csv_flows.py to batch process more files at once.

== 3. MachineLearning & app recognition ==
You can use run_ml.py, just edit the following varibles:
csv_path: path to csvs from 1.
output_filename_prefix: output files file name prefix
flow_timeout: flow timeout value
activity_timeout: activity timeout ValueError
Output: -- a multipage .pdf that contain Confusion Matrices
        -- some .csv files *CLASSIFIER_micro_macro.csv, it contains micro and macro results for the CLASSIFIER
        -- some .csv files *CLASSIFIER.csv, it contains classifier performances per app and TP,TN,FP,FN,
             Precision, Recall, F1, Accuracy
