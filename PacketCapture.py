# author: EManuele/Immanuel

import SimplePacket
import pandas
from scapy.all import *
from timeit import default_timer as timer
import gc

class PacketCapture:

    def __init__(self, pcap_file, filter_dupes=False):
        start = timer()
        packets = []
        pkt_num = 0
        tot_pkt = 0
        # list of TCP Streams (each one identified by SrcIP:srcPort DstIP:dstPort)
        self.streamslist = []
        # dict with TCP Stream -> traffic for esch stream (in SimplePacket format)
        self.streams = dict()
        file = rdpcap(pcap_file)
        for packet in file:
            try:
                packets.append(SimplePacket.SimplePacket(packet))
                pkt_num += 1
            #It's not an IP+TCP packet
            except ValueError as e:
                # print(e)
                pass

            tot_pkt += 1
        print("Packets Parsed: %d, Valid: %d" % (tot_pkt, pkt_num))
        # All Packets (in SimplePacket format) from the .pcap
        self.Packets = sorted(packets, key=lambda SimplePacket: SimplePacket.timestamp)
        self._sortStreams()
        end = timer()
        print("Elapsed: %.2f" % (end - start))

    # Prints all packet in current PacketCapture
    def print(self):
        for packet in self.Packets:
            packet.print()

    # Return the numerber of packets
    def size(self):
        return len(self.Packets)

    # Internal method used to process all packets and split traffic stream
    def _sortStreams(self):
        for packet in self.Packets:
            key = packet.ipSrc, packet.srcPort, packet.ipDst, packet.dstPort
            if key in self.streams:
                self.streams[key].append(packet)
            elif (packet.ipDst, packet.dstPort, packet.ipSrc, packet.srcPort) in self.streams:
                # pkts[packet.ipDst, packet.ipSrc] += 1
                self.streams[packet.ipDst, packet.dstPort, packet.ipSrc, packet.srcPort].append(packet)
            else:
                self.streams[key] = []
                self.streams[key].append(packet)
        self.streamslist = list(self.streams.keys())

    # Prints Streams (TCP sessions)
    def liststreams(self):
        for index, stream in enumerate(self.streamslist):
            print("[%d] %s, packets: %d" % (index, stream, len(self.streams[stream])))

    # Returns a Stream from its position (see liststreams())
    def getstream(self, stream_number):
        if stream_number in range(len(self.streamslist)):
            print("Getting stream %d" % (stream_number))
            return self.streams[self.streamslist[stream_number]]
        else:
            print("Error, invalid stream number.")
