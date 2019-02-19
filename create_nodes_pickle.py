# author: EManuele/Immanuel

import NodeGuard

if __name__ == '__main__':
    nodes =  NodeGuard.NodeGuard()
    nodes.loadNodes()
    nodes.saveNodesToFile("entry_nodes")
