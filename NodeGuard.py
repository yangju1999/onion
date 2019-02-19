# author: EManuele/Immanuel

import requests
import requests.exceptions
import logging
from lxml import etree, html
import re
import pickle

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
TOR_NODES_LIST_URL = "https://www.dan.me.uk/tornodes"


class NodeGuard:

    def __init__(self):
        self.guards = {}

    def loadNodes(self):
        page = requests.get(TOR_NODES_LIST_URL)
        tmp_guards = {}
        if page.status_code == requests.codes.ok:
            logging.info("Grabbed List from %s" % (TOR_NODES_LIST_URL))
            tree = html.fromstring(page.content)
            page_list = tree.xpath('//div[@class="article box"]/text()')
            pattern = re.compile("^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

            for line in page_list:
                server = line.strip().split("|")
                if pattern.match(server[0]) and "G" in server[4]:
                    #print("Guard node IP: %s attrs: %s" %(server[0], server[4]))
                    tmp_guards[server[0]] = server[4]
            logging.info("Downloaded %d Nodes" % (len(tmp_guards)))
        if len(tmp_guards) > 0:
            self.guards = tmp_guards
        else:
            logging.info("No nodes loaded, list empty.")

    def loadNodesFromPickle(self, pickle_file):
        with open(pickle_file + '.pkl', 'rb') as f:
            self.guards = pickle.load(f)
        logging.info("%d Nodes loaded from %s." % (len(self.guards), pickle_file))

    def saveNodesToFile(self, pickle_file):
        with open(pickle_file + '.pkl', 'wb') as f:
            pickle.dump(self.guards, f, 0)
        logging.info("Nodes saved to %s" % (pickle_file))

    # add an IP to NodeGuard obj
    def manually_add(self, ip):
        self.guards[ip] = "G"

    def isGuardNode(self, nodeip):
        return nodeip in self.guards
