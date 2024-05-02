# author: EManuele/Immanuel

import SimplePacket
import PacketCapture
import numpy
import constants
import pandas
from itertools import groupby
from collections import OrderedDict
from collections import Counter
import pdb 

class Flow:
    # FlowTimeout: (where do we cut a long TCP stream into smaller chunks, default: 120sec/chunk)
    # ActivityTimeout: (After how much time do we consider the stream IDLE, default 5sec)
    # first_x_packets_direction: for how many packets do we save the direction in an array? (default: 10)
    # first_x_burst_length: how many of the first  bursts length do we save? (default: 10)
    def __init__(self, FlowTimeout=120, ActivityTimeout=5, first_x_packets_direction=10, first_x_bursts_length=10, \
                ipsrc=None, ipdst=None):
        self.ipSRC = ipsrc
        self.ipDST = ipdst
        self.srcPort = None
        self.dstPort = None
        # Array with all incoming/outgoing/total packets
        self.InPackets = []
        self.OutPackets = []
        self.AllPackets = []
        # TCP Stream timeout (if no FIN found)
        self.FlowTimeout = FlowTimeout
        self.ActivityTimeout = ActivityTimeout
        # Backward/Forward/Both Packet Interarrival Time
        # (Not yet completely defined..)
        # Mean, Min, Max, Std
        self.BIAT = [0, 0, 0, 0]
        self.FIAT = [0, 0, 0, 0]
        self.FLOWIAT = [0, 0, 0, 0]
        # TODO - DONE flow bytes per second
        self.FB_SEC = -1
        # TODO - DONE flow packets per second
        self.FP_SEC = -1
        # Duration (whole flow)
        self.DURATION = 0.0
        # TODO active/idle
        self.ACTIVE = [0, 0, 0, 0]
        self.IDLE = [0, 0, 0, 0]
        # Total traffic in bytes In/Out/Both
        self.totInBytes = 0
        self.totOutBytes = 0
        self.totInOutBytes = 0
        # Total traffic in packets In/Out/Both
        self.totInPackets = 0
        self.totOutPackets = 0
        self.TotalPackets = 0
        # Timestamps first/last packet
        self.firstPacketTS = 0
        self.lastPacketTS = 0
        # Timestamps first/last packet Incoming
        self.firstInPacketTS = 0
        self.lastInPacketTS = 0
        # Timestamps first/last packet Outgoing
        self.firstOutPacketTS = 0
        self.lastOutPacketTS = 0
        # label = APP
        self.label = ""
        # category =  category [social, streaming...]
        self.category = ""
        # a dict where key is packet size and value #packets of that size
        self.packets_size = dict()
        # value of first packets to check for direction
        self.first_x_packets = first_x_packets_direction
        # First X packets and direction (+1 outgoing, -1 incoming)
        # init it to 0 (no n-th packet)
        self.first_packets_direction = [0] * first_x_packets_direction
        # bursts incoming stats [Mean, #bursts, Max]
        self.bursts_incoming = [0, 0, 0]
        # bursts outgoing stats [Mean, #bursts, Max]
        self.bursts_outgoing = [0, 0, 0]
        self.first_bursts_len_to_save = first_x_bursts_length
        self.first_incoming_bursts_len = [0] * first_x_bursts_length
        self.first_outgoing_bursts_len = [0] * first_x_bursts_length
        # packet sizes stats, a dict with size:count
        self.packet_sizes_tor = OrderedDict([(583,0), (595,0), (1500,0), (1097,0),(1138,0),
                                            (1109,0), (151,0), (1126,0), (233,0)])

    def computeFIAT(self):
        self._interArrivalTime(self.OutPackets, self.FIAT)

    def computeBIAT(self):
        self._interArrivalTime(self.InPackets, self.BIAT)

    def computeFLOWIAT(self):
        self._interArrivalTime(self.AllPackets, self.FLOWIAT, True)

    # New FIAT, BIAT and FLOWIAT, and compute also ActivityTimeout (if required)
    def _interArrivalTime(self, tsSortedPacketList, iatList, computeActivityTimeout=False):
        iat = []
        active = []
        idle = []
        if len(tsSortedPacketList) > 0:
            # First packet timestamp used as reference
            for i in range(len(tsSortedPacketList)):
                # First packet so the IAT is 0
                if i == 0:
                    iat.append(0)
                else:
                    # Other packets, so i > 0 and we compute IAT: current packet ts - prev packet ts
                    iat.append(tsSortedPacketList[i].timestamp - tsSortedPacketList[i - 1].timestamp)
            # This FIXES the min, now min must be > 0 (unless)
            if len(iat) > 1:
                # iatList[constants.MIN] = numpy.min(iat[1:])
                try:
                    iatList[constants.MIN] = min(x for x in iat if x != 0)
                except ValueError:
                    # we are really unlucky and all values in IAT are 0...
                    # but really really really unlucky...
                    # \___________/
                    #   | | | | |
                    iatList[constants.MIN] = 0
                iat_float = [float(x) for x in iat] 
                iatList[constants.MAX] = numpy.max(iat)
                iatList[constants.STD] = numpy.std(iat_float)
                iatList[constants.MEAN] = numpy.mean(iat_float)

            if computeActivityTimeout:
                idle_values = []
                active_values = []
                active = True
                for i in iat[1:]:
                    # case 1 - Idle
                    #print("Fin i %d: %d" % (index, self.AllPackets[index].FIN))
                    #if self.AllPackets[index].FIN == False:
                    if i > self.ActivityTimeout:
                        # Idle after some active time
                        if active:
                            active = False
                            # add new idle time
                            idle_values.append(i)
                        # consecutive idle time
                        # so we sum this idle time to the last idle value
                        else:
                            # Using a try block just to be sure it doesn't throw exception..
                            # but this ** should never happen ** here
                            try:
                                # update the last idle value
                                idle_values[-1] += i
                            except IndexError:
                                idle_values.append(i)
                    # we are not idle
                    else:
                        # consecutive active time
                        if active:
                            try:
                                active_values[-1] += i
                            # This happens when it's the first active value..
                            except IndexError:
                                active_values.append(i)
                        # after an idle time so we add a new idle time elem
                        else:
                            active = True
                            active_values.append(i)

                if len(active_values) > 0:
                    active_values_float = [float(x) for x in active_values] 
                    self.ACTIVE[constants.MEAN] = numpy.mean(active_values_float)
                    self.ACTIVE[constants.MIN] = numpy.min(active_values)
                    self.ACTIVE[constants.MAX] = numpy.max(active_values)
                    self.ACTIVE[constants.STD] = numpy.std(active_values_float)
                if len(idle_values) > 0:
                    idle_values_float = [float(x) for x in idle_values] 
                    self.IDLE[constants.MEAN] = numpy.mean(idle_values_float)
                    self.IDLE[constants.MIN] = numpy.min(idle_values)
                    self.IDLE[constants.MAX] = numpy.max(idle_values)
                    self.IDLE[constants.STD] = numpy.std(idle_values_float)

    def _updateDuration(self):
        if self.TotalPackets > 1:
            self.DURATION = self.lastPacketTS - self.firstPacketTS

    def _updateBytesCounter(self, pkt, counter):
        counter += pkt.len

    # Load traffic cpture from a PacketCapture
    def loadPackets(self, sequence):
        if len(sequence) > 0:
            self.AllPackets = sequence
            self.TotalPackets = len(sequence)
            self._initInOut()
            self.computeFIAT()
            self.computeBIAT()
            self.computeFLOWIAT()
            self._computeFP_SEC()
            self._computeFB_SEC()
            self._packets_size_update()
            #print(self.first_x_packets)
            self._direction_first_x_packets(self.first_x_packets)
            self._compute_burst()

    def print(self):
        # print("SRC: %s:%d - DST: %s:%d packets: %d" % (self.AllPackets[0].ipSrc,self.AllPackets[0].srcPort, self.AllPackets[0].ipDst, self.AllPackets[0].dstPort, self.TotalPackets ))
        print("SRC: %s:%d - DST: %s:%d packets: %d bytes: %f" % (self.ipSRC, self.srcPort, self.ipDST, self.dstPort, self.TotalPackets, self.totInOutBytes))
        print("Time: %f First packet ts: %f Last packet ts: %f" % (self.DURATION, self.firstPacketTS, self.lastPacketTS))
        print("Flow Packets per seconds: %f Flow Bytes per seconds: %f" % (self.FP_SEC, self.FB_SEC))
        print("Incoming traffic - Packets: %d Bytes: %d First packet ts: %f Last packet ts: %f" % (self.totInPackets, self.totInBytes, self.firstInPacketTS, self.lastInPacketTS))
        print("BIAT mean: %f min: %f max: %f std: %f" % (self.BIAT[constants.MEAN], self.BIAT[constants.MIN], self.BIAT[constants.MAX], self.BIAT[constants.STD]))
        print("Outgoing traffic - Packets: %d Bytes: %d First packet ts: %f Last packet ts: %f" % (self.totOutPackets, self.totInPackets, self.firstOutPacketTS, self.lastOutPacketTS))
        print("FIAT mean: %f min: %f max: %f std: %f" % (self.FIAT[constants.MEAN], self.FIAT[constants.MIN], self.FIAT[constants.MAX], self.FIAT[constants.STD]))
        print("FLOWIAT mean: %f min: %f max: %f std: %f" % (self.FLOWIAT[constants.MEAN], self.FLOWIAT[constants.MIN], self.FLOWIAT[constants.MAX], self.FLOWIAT[constants.STD]))
        print("ACTIVE mean: %f min: %f max: %f std: %f" % (self.ACTIVE[constants.MEAN], self.ACTIVE[constants.MIN], self.ACTIVE[constants.MAX], self.ACTIVE[constants.STD]))
        print("IDLE mean: %f min: %f max: %f std: %f" % (self.IDLE[constants.MEAN], self.IDLE[constants.MIN], self.IDLE[constants.MAX], self.IDLE[constants.STD]))
        print("FIRST X Packets direction: %s" % (self.first_packets_direction))
        print("BURST INCOMING mean: %f, #bursts: %f, max: %f" % (self.bursts_incoming[constants.MEAN], self.bursts_incoming[constants.COUNT], self.bursts_incoming[constants.MAX]))
        print("First %d INCOMING BURSTS LEN: %s" % (self.first_bursts_len_to_save, self.first_incoming_bursts_len))
        print("BURST OUTGOING mean: %f, #bursts: %f, max: %f" % (self.bursts_outgoing[constants.MEAN], self.bursts_outgoing[constants.COUNT], self.bursts_outgoing[constants.MAX]))
        print("First %d OUTGOING BURSTS LEN: %s" % (self.first_bursts_len_to_save, self.first_outgoing_bursts_len))



    # Total packets/ Duration -> packts/seconds
    def _computeFP_SEC(self):
        if self.TotalPackets > 0 and self.DURATION > 0:
            self.FP_SEC = self.TotalPackets / self.DURATION

    def _computeFB_SEC(self):
        if self.totInBytes > 0 and self.DURATION > 0:
            self.FB_SEC = self.totInOutBytes / self.DURATION

    def _initInOut(self):
        if len(self.AllPackets) > 0:

            first_packet = self.AllPackets[0]
            # Common: we don't set src/dst so autodetect from first packet
            if self.ipSRC is None or self.ipDST is None:
                # First packet (and it's a SimplePacket obj)
                self.ipSRC = first_packet.ipSrc
                self.ipDST = first_packet.ipDst
                self.srcPort = first_packet.srcPort
                self.dstPort = first_packet.dstPort
            # TOR: we manually set IP and DST (DST is tor node)
            else:
                # we need srcPort and dstPort..
                if self.ipSRC == first_packet.ipSrc:
                    self.srcPort = first_packet.srcPort
                    self.dstPort = first_packet.dstPort
                else:
                    self.srcPort = first_packet.dstPort
                    self.dstPort = first_packet.srcPort
            # set ipSrc and ipDst
            for packet in self.AllPackets:
                # if packet.ipSrc == dst:
                if packet.ipSrc == self.ipDST:
                    self.InPackets.append(packet)
                    self.totInBytes += packet.len
                else:
                    self.OutPackets.append(packet)
                    self.totOutBytes += packet.len
            if len(self.InPackets) > 0:
                self.firstInPacketTS = self.InPackets[0].timestamp
                self.lastInPacketTS = self.InPackets[len(self.InPackets) - 1].timestamp
            if len(self.OutPackets) > 0:
                self.firstOutPacketTS = self.OutPackets[0].timestamp
                self.lastOutPacketTS = self.OutPackets[len(self.OutPackets) - 1].timestamp
            # If we have only incoming packets OR only outgoing packets
            # then self.firstPacketTS would be 0 unless we do the following..
            if self.firstInPacketTS == 0 or self.firstOutPacketTS == 0:
                self.firstPacketTS = max(self.firstInPacketTS, self.firstOutPacketTS)
            else:
                self.firstPacketTS = min(self.firstInPacketTS, self.firstOutPacketTS)
            self.lastPacketTS = max(self.lastInPacketTS, self.lastOutPacketTS)
            self._updateDuration()
            self.totInOutBytes = self.totInBytes + self.totOutBytes
            self.totInPackets = len(self.InPackets)
            self.totOutPackets = len(self.OutPackets)

    #returns a list where the last element it's a list with columns header
    def flowToList(self):

        flowentry = []
        flowentry.append(self.label)
        # added
        flowentry.append(self.category)
        #end added
        flowentry += self.FLOWIAT + self.FIAT + self.BIAT + self.ACTIVE + self.IDLE
        flowentry.append(self.DURATION)
        flowentry.append(self.FB_SEC)
        flowentry.append(self.FP_SEC)
        # added
        flowentry += self.first_packets_direction + self.bursts_incoming + self.bursts_outgoing + self.first_incoming_bursts_len + self.first_outgoing_bursts_len
        # add here  packet sizes tor
        flowentry += list(self.packet_sizes_tor.values())
        flowentry.append(self.totInPackets)
        flowentry.append(self.totOutPackets)
        flowentry.append(self.TotalPackets)
        #end added
        flowentry.append(self.totInOutBytes)
        flowentry.append(self.ipSRC)
        flowentry.append(self.ipDST)
        header = []

        # column headers
        first_packet_direction_columns_header = ['FIRST_PACKET_DIR_' + str(i) for i in range(0, self.first_x_packets)]
        bursts_incoming_col = ['INCOMING_BURSTS_MEAN', 'INCOMING_BURSTS_COUNT', 'INCOMING_BURSTS_LONGEST']
        bursts_outgoing_col = ['OUTGOING_BURSTS_MEAN', 'OUTGOING_BURSTS_COUNT', 'OUTGOING_BURSTS_LONGEST']
        first_incoming_bursts_len_col = ['FIRST_INCOMING_BURST_' + str(i) for i in range(0, self.first_bursts_len_to_save)]
        first_outgoing_bursts_len_col = ['FIRST_OUTGOING_BURST_' + str(i) for i in range(0, self.first_bursts_len_to_save)]
        packet_sizes_tor_col = ['SIZE_583', 'SIZE_595', 'SIZE_1500', 'SIZE_1097',
                                'SIZE_1138', 'SIZE_1109', 'SIZE_151', 'SIZE_1126',
                                'SIZE_233']
        packets_col = ['TOT_IN_PACKETS', 'TOT_OUT_PACKETS', 'TOT_PACKETS', 'TOT_BYTES', 'IP_SRC', 'IP_DST']
        columns_pt1 = ['LABEL', 'CATEGORY','FLOWIAT_MEAN', 'FLOWIAT_MIN', 'FLOWIAT_MAX', 'FLOWIAT_STD',
                   'FIAT_MEAN', 'FIAT_MIN', 'FIAT_MAX', 'FIAT_STD',
                   'BIAT_MEAN', 'BIAT_MIN', 'BIAT_MAX', 'BIAT_STD',
                   'ACTIVE_MEAN', 'ACTIVE_MIN', 'ACTIVE_MAX', 'ACTIVE_STD',
                   'IDLE_MEAN', 'IDLE_MIN', 'IDLE_MAX', 'IDLE_STD',
                   'DURATION', 'FB_PSEC', 'FP_SEC']
        columns = columns_pt1 + first_packet_direction_columns_header + bursts_incoming_col + bursts_outgoing_col \
                    + first_incoming_bursts_len_col + first_outgoing_bursts_len_col + packet_sizes_tor_col \
                    + packets_col
        flowentry.append(columns)
        return flowentry

    # inits the dict with packets size
    def _packets_size_update(self):
        self.packets_size = Counter([x.len for x in self.AllPackets])

    # outputs a .csv where each row is a [packet size, #packets of that size]
    def csv_with_packets_size(self, filename):
        df = pandas.DataFrame.from_dict(self.packets_size, orient='index')
        df.to_csv(filename, sep=',', encoding='utf-8', index=True)

    # returns a pandas DataFrame with [packet size, #packets with that size]
    def packets_size_to_pandas(self):
        return pandas.DataFrame.from_dict(self.packets_size, orient='index', columns=['n'])

    # filters packate by size
    # first we create a new list filtered, after we have to reset and recalculate  all stats
    def filter_packets_eq_less_than(self, target_size):
        if len(self.AllPackets) > 0:
            self.AllPackets = [pkt for pkt in self.AllPackets if pkt.len > target_size]
            self._reset()
            self.TotalPackets = len(self.AllPackets)
            self._initInOut()
            self.computeFIAT()
            self.computeBIAT()
            self.computeFLOWIAT()
            self._computeFP_SEC()
            self._computeFB_SEC()
            self._packets_size_update()
            self._compute_burst()
            self._direction_first_x_packets(self.first_x_packets)
            self._count_packet_sizes_tor()

    # bursts stats
    def _compute_burst(self):
        incoming_bursts = []
        outgoing_bursts = []

        #direction: True if outgoing, False incoming
        #direction = True if self.AllPackets[0].ipSrc == self.ipSRC else False
        # +1 -> outgoing, -1 incoming
        bursts_list = []
        for packet in self.AllPackets:
            # outgoing packet
            if packet.ipSrc == self.ipSRC:
                bursts_list.append(1)
            else:
            # incoming packet
                bursts_list.append(-1)
        burst_np = numpy.array(bursts_list)
        # lists with bursts length
        bursts_count_incoming = numpy.array([len(list(g[1]))  for g in groupby(bursts_list) if g[0] == -1])
        bursts_count_outgoing = numpy.array([len(list(g[1]))  for g in groupby(bursts_list) if g[0] == 1])
        bursts_in = None
        bursts_out = None
        if len(bursts_count_incoming) > 0:
            bursts_in = bursts_count_incoming[numpy.where(bursts_count_incoming > 1)]
            if len(bursts_in) > 0:
                self.bursts_incoming[constants.MEAN] = numpy.mean(bursts_in)
                self.bursts_incoming[constants.COUNT] = len(bursts_in)
                self.bursts_incoming[constants.MAX] = numpy.max(bursts_in)
                if len(bursts_in) >= self.first_bursts_len_to_save:
                    self.first_incoming_bursts_len = bursts_in[:self.first_bursts_len_to_save].tolist()
                else:
                    copied = len(bursts_in)
                    self.first_incoming_bursts_len = bursts_in[:].tolist()
                    self.first_incoming_bursts_len[copied:] = [0] * (self.first_bursts_len_to_save - copied)
        if len(bursts_count_outgoing) > 0:
            bursts_out = bursts_count_outgoing[numpy.where(bursts_count_outgoing > 1)]
            if len(bursts_out) > 0:
                # save mean, #bursts, max
                self.bursts_outgoing[constants.MEAN] = numpy.mean(bursts_out)
                self.bursts_outgoing[constants.COUNT] = len(bursts_out)
                self.bursts_outgoing[constants.MAX] = numpy.max(bursts_out)
                if len(bursts_out) >= self.first_bursts_len_to_save:
                    self.first_outgoing_bursts_len = bursts_out[:self.first_bursts_len_to_save].tolist()
                else:
                    copied = len(bursts_out)
                    self.first_outgoing_bursts_len = bursts_out[:].tolist()
                    self.first_outgoing_bursts_len[len(bursts_out):] = [0] * (self.first_bursts_len_to_save - copied)

    def _direction_first_x_packets(self, number_of_packets):
    # 0 -> no x-th packet
    # 1 -> x-th outgoing packet
    # -1 -> x-th incoming packet
        all_packets = len(self.AllPackets)
        n_packets = number_of_packets if all_packets > number_of_packets else all_packets
        for i in range(0, n_packets):
            if self.AllPackets[i].ipSrc == self.ipSRC:
                self.first_packets_direction[i] = 1
            else:
                self.first_packets_direction[i] = -1

    # Tor packet sizes features
    def _count_packet_sizes_tor(self):
        size_counts = Counter([x.len for x in self.AllPackets])
        for size in self.packet_sizes_tor.keys():
            self.packet_sizes_tor[size] = size_counts[size]

    # resets all Flow stats attributes except: AllPackets, ipSRC, ipDST, srcport, dstport )
    def _reset(self):
        self.InPackets.clear()
        self.OutPackets.clear()
        self.BIAT = [0, 0, 0, 0]
        self.FIAT = [0, 0, 0, 0]
        self.FLOWIAT = [0, 0, 0, 0]
        self.FB_SEC = -1
        self.FP_SEC = -1
        self.SYN = False
        self.FIN = False
        self.ACTIVE = [0, 0, 0, 0]
        self.IDLE = [0, 0, 0, 0]
        self.DURATION = 0.0
        self.totInBytes = 0
        self.totOutBytes = 0
        self.totInOutBytes = 0
        self.totInPackets = 0
        self.totOutPackets = 0
        self.TotalPackets = 0
        self.firstPacketTS = 0
        self.lastPacketTS = 0
        self.firstInPacketTS = 0
        self.lastInPacketTS = 0
        self.firstOutPacketTS = 0
        self.lastOutPacketTS = 0
        self.packets_size.clear()
        self.bursts_incoming = [0, 0, 0]
        self.bursts_outgoing = [0, 0, 0]
        self.first_packets_direction = [0] * self.first_x_packets
        self.first_incoming_bursts_len = [0] * self.first_bursts_len_to_save
        self.first_outgoing_bursts_len = [0] * self.first_bursts_len_to_save
        for size in self.packet_sizes_tor.keys():
            self.packet_sizes_tor[size] = 0
