# author: EManuele/Immanuel

import SimplePacket
import PacketCapture
import Flow
import NodeGuard
import pandas
import os
from functools import reduce, partial
import pickle
import time

def detectFlows(stream, Timeout=120):
    subflows = []
    # So timeout is now a float
    Timeout *= 1.0
    current_subflow = 0
    if len(stream) > 0:
        start_ts_flow = 0
        active_flow = False
        subflows.append([])
        for index, packet in enumerate(stream):
            if not active_flow:
                active_flow = True
                start_ts_flow = packet.timestamp
            # Split the flow, timeout reached
            if active_flow and (packet.timestamp - start_ts_flow > Timeout):
                active_flow = False
                subflows.append([])
                current_subflow += 1
            # TCP connection end, we got a FIN
            elif active_flow and packet.FIN and packet.ACK:
                active_flow = False
                subflows.append([])
            subflows[current_subflow].append(packet)

    return subflows


# args: list of IP:srcPort IP:dstport, NodeGuard
def detectTorIP(listIP, nodeguard):
    tortraffic = []
    for ips in listIP:
        # print(ips)
        if nodeguard.isGuardNode(ips[0]) or nodeguard.isGuardNode(ips[2]):
            ret_ips = ()
            # we return the original tuple + tor IP
            # so: ips[0],ips[1] -> src:port
            #     ips[2],ips[3] -> dst:port
            #     ips[4] -> TOR IP
            if nodeguard.isGuardNode(ips[0]):
                ret_ips = (ips[0], ips[1], ips[2], ips[3], ips[0])
            else:
                ret_ips = (ips[0], ips[1], ips[2], ips[3], ips[2])
            tortraffic.append(ret_ips)
            print("Tor traffic detected: %s:%s<->%s:%s -- [TOR DST: %s]" % (
                str(ret_ips[0]), str(ret_ips[1]),
                str(ret_ips[2]), str(ret_ips[3]),
                str(ret_ips[4])))
    return tortraffic

def exportflowsToCsv(flows, filename, label=None, category=None):
    data = []
    header = []
    for flow in flows:
        if label is not None:
            flow.label = label
        if category is not None:
            flow.category = category
        to_list = flow.flowToList()
        # to_list: all data, last row contains HEADER
        flow_values = to_list[:-1]
        flow_header = to_list[-1]
        # header columns
        header = flow_header
        data.append(flow_values)
    output = pandas.DataFrame(data=data, columns=header)
    output.to_csv(filename, sep=",", encoding='utf-8', index=False)


# Takes a .pcap, outputs a .csv  with flows (and returns a list of flows)
# Params: output_folder: save all flow related files to output_folder
#         flow_length: filter flows with less than X packets (default: 10)
#         label: assign  label to flow, if none then...empty label
#         category: assign category to flow, if none...then category
#         Timeout: flow timeout (default: 30)
#         TorDetect: save only Tor traffic flows
#         TorPickle: pickle with tor entry nodes, if None then tor nodes will be downloaded
#         output_sizes_stat: outputs a csv with packet_size:#packets (default: False)
#         pickle_flows: outputs a .pkl with detected Flows (in contains a list of flows) (default: False)
#         filter_packets_eq_less_than: removes from flows packets <= target_size (default -1, disabled)
def ProcessPcap(filename, output_folder="", flow_length=10,
                label=None, category=None, Timeout=30, ActivityTimeout=5,
                TorDetect=False, TorPickle=None,
                output_sizes_stat=False, pickle_flows=False,
                filter_packets_eq_less_than=-1):
    df_sizes = pandas.DataFrame()
    list_sizes = []
    try:
        input_pcap = PacketCapture.PacketCapture(filename)
    except (OSError, FileNotFoundError, FileExistsError) as e:
        print(e)
        print("Error while accessing file %s." % (filename))
        exit(1)
    Flows = []
    time.sleep(1)
    if (TorDetect):
        tornodes = NodeGuard.NodeGuard()
        if(TorPickle):
            tornodes.loadNodesFromPickle(TorPickle)
        else:
            tornodes.loadNodes()

        tor_traffic = detectTorIP(input_pcap.streamslist, tornodes)
        for flow in tor_traffic:
                subflows = detectFlows(input_pcap.streams[(flow[0], flow[1], flow[2], flow[3])], Timeout=Timeout)
                # set up desttination (Tor EntryNode) and source
                flow_dest = flow[4]
                flow_source = None
                # from tcpdump point of view dest is Tor node
                if flow_dest == flow[2]:
                    flow_source = flow[0]
                else:
                # then tcpdump saw Tor node as source and we don't like it..
                    flow_source = flow[2]
                for f in subflows:
                    # process flows with at least 10 packets
                    if len(f) > flow_length:
                        ff = Flow.Flow(FlowTimeout=Timeout, ActivityTimeout=ActivityTimeout, \
                                        ipsrc=flow_source, ipdst=flow_dest)
                        ff.loadPackets(f)
                        if (filter_packets_eq_less_than != -1):
                            ff.filter_packets_eq_less_than(filter_packets_eq_less_than)
                        Flows.append(ff)

                        if output_sizes_stat:
                            list_sizes.append(ff.packets_size_to_pandas())
    # No TOR detection
    else:
        for flow in input_pcap.streams:
            subflows = detectFlows(input_pcap.streams[flow], Timeout=Timeout)
            for f in subflows:
                if len(f) > flow_length:
                    ff = Flow.Flow(FlowTimeout=Timeout, ActivityTimeout=ActivityTimeout)
                    ff.loadPackets(f)
                    if (filter_packets_eq_less_than != -1):
                        ff.filter_packets_eq_less_than(filter_packets_eq_less_than)
                    Flows.append(ff)
                    if output_sizes_stat:
                        list_sizes.append(ff.packets_size_to_pandas())
    print("Flows extracted: %d" %(len(Flows)) )
    name = os.path.basename(filename)
    exportflowsToCsv(Flows, output_folder + name + "_flows_" + str(Timeout) + "_" + \
                                                str(ActivityTimeout) + ".csv", label=label, category=category)
    if output_sizes_stat and len(list_sizes) > 0:
        df_sizes = reduce(partial(pandas.DataFrame.add, fill_value=0), list_sizes)
        df_sizes.to_csv(output_folder + name + "_flows_" + str(Timeout) +  "_" + str(ActivityTimeout) + "_size_stats.csv", sep=",", encoding='utf-8', index=True)
    if pickle_flows:
        with open(output_folder + name + "_flows_" + str(Timeout) +  "_" + str(ActivityTimeout) + ".pkl", 'wb' ) as f:
            pickle.dump(Flows, f, 0)
    return Flows

# Default Tor processing function + parameters
def ProcessPcapTor(filename, output_folder, label,
                    category, Timeout, ActivityTimeout,
                    TorPickle):
    return ProcessPcap(filename, output_folder, flow_length=10, label=label, category=category,
                    Timeout=Timeout, ActivityTimeout=ActivityTimeout, TorDetect=True, TorPickle=TorPickle,
                    pickle_flows=False, filter_packets_eq_less_than=80, output_sizes_stat=False)
