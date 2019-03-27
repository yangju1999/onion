# Peel the Onion
Proof of Concept to de-anonimize apps used on android device through a TOR proxy. 
## How to process tor .pcaps
#### 1. Requirements
  * Python 3.6.7 (i tested only this version but newer versions should be ok too).
  * Python Libraries: see `requirements.txt` file.

#### 2. We first need a .pkl file with entry Nodes.
To create a .pkl file with nodes you can do the following in a python shell:
```python
import sys
sys.path.append('PATH_TO_NodeGuard.py')
import NodeGuard
nodes = NodeGuard.NodeGuard()
nodes.loadNodes()
nodes.saveNodesToFile("nodes")
```
Or you can use `create_nodes_pickle.py` file.  
File nodes.pkl contains all entry nodes ready.

#### 3. Main function to process .pcaps is __ProcessPcapTor__ (in util.py).  
`util.ProcessPcapTor(PCAP_FILE, output_folder=OUTPUT_FOLDER, label=APP_NAME, category=APP_CATEGORY,Timeout=FLOW_TIMEOUT, ActivityTimeout=ACTIVITY_TIMEOUT, TorPickle=NODES.pkl)`  
You can edit and use `tor_pcaps_to_csv_flows.py` to batch process more files at once.

#### 4. MachineLearning & app recognition  
You can use run_ml.py, just edit the following variables:
  * csv_path: path to csvs from 1.
  * output_filename_prefix: output files file name prefix
  * flow_timeout: flow timeout value
  * activity_timeout: activity timeout ValueError  
Output:
  * a multipage .pdf that contain Confusion Matrices
  * some .csv files *CLASSIFIER_micro_macro.csv, it contains micro and macro results for the CLASSIFIER
  * some .csv files *CLASSIFIER.csv, it contains classifier performances per app and TP,TN,FP,FN,
    Precision, Recall, F1, Accuracy

### Dataset
A copy of the dataset is available at www.cis.uniroma1.it/peel-the-onion

## Acknowledgment
Dataset and code are part of the work "Peel the onion: Recognition of Android apps behind the Tor Network" [https://arxiv.org/abs/1901.04434]. Please cite it if you use something for your work.
