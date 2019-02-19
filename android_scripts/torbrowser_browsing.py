#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2013-2018  Diego Torres Milano
Created on 2018-12-02 by CulebraTester
                      __    __    __    __
                     /  \  /  \  /  \  /  \
____________________/  __\/  __\/  __\/  __\_____________________________
___________________/  /__/  /__/  /__/  /________________________________
                   | / \   / \   / \   / \   \___
                   |/   \_/   \_/   \_/   \    o \
                                           \_____/--<
@author: Diego Torres Milano
@author: Jennifer E. Swofford (ascii art snake)
'''
# author: EManuele/Immanuel

import re
import sys
import os
import random
import io

import unittest
try:
    sys.path.insert(0, os.path.join(os.environ['ANDROID_VIEW_CLIENT_HOME'], 'src'))
except:
    pass

import pkg_resources
pkg_resources.require('androidviewclient>=12.4.0')
from com.dtmilano.android.viewclient import ViewClient, CulebraTestCase
from com.dtmilano.android.uiautomator.uiautomatorhelper import UiAutomatorHelper, UiScrollable, UiObject, UiObject2

TAG = 'CULEBRA'

# minumum time visit for a page
min_browse = 20
# maximum time visit for a page
max_browse = 30
# sites to Visit
how_many_sites = 200

class CulebraTests(CulebraTestCase):
    @classmethod
    def setUpClass(cls):
        cls.kwargs1 = {'ignoreversioncheck': False, 'verbose': False, 'ignoresecuredevice': False}
        cls.kwargs2 = {'forceviewserveruse': False, 'useuiautomatorhelper': True, 'ignoreuiautomatorkilled': True, 'autodump': False, 'startviewserver': True, 'compresseddump': True}
        cls.options = {'start-activity': None, 'concertina': False, 'device-art': None, 'use-jar': False, 'multi-device': False, 'unit-test-class': True, 'save-screenshot': None, 'use-dictionary': False, 'glare': False, 'dictionary-keys-from': 'id', 'scale': 1, 'find-views-with-content-description': True, 'window': -1, 'orientation-locked': None, 'save-view-screenshots': None, 'find-views-by-id': True, 'log-actions': False, 'use-regexps': False, 'null-back-end': False, 'auto-regexps': None, 'do-not-verify-screen-dump': True, 'verbose-comments': False, 'gui': False, 'find-views-with-text': True, 'prepend-to-sys-path': False, 'install-apk': None, 'drop-shadow': False, 'output': None, 'unit-test-method': None, 'interactive': False}
        cls.sleep = 5

    def setUp(self):
        super(CulebraTests, self).setUp()

    def tearDown(self):
        super(CulebraTests, self).tearDown()

    def preconditions(self):
        if not super(CulebraTests, self).preconditions():
            return False
        return True
    # motorola app screen coords
    def startTorBrowser_fromApps(self):
        package = 'org.torproject.torbrowser_alpha'
        activity = '.App'
        component = package +"/"+ activity
        device, serialno = ViewClient.connectToDeviceOrExit()
        device.startActivity(component=component)
        self.vc.sleep(3)


    def load_sites(self, sites_file):
        sites_list = []
        with open(sites_file, "r") as f:
        #with io.open(sites_file, 'r', encoding="utf-8", newline='\n') as f:
            for line in f:
                sites_list.append(line.strip())
        return sites_list

    def random_nav(self, moves):
        for i in range(0,moves):
            if random.randint(0,100) > 15:
                self.scroll_down()
            else:
                self.scroll_up()

            self.vc.sleep(random.randint(2,5))
        self.vc.sleep(random.randint(2,6))

    def go_to_url(self, my_url):
        # Ohhhhhh YEEEEAH!
        _s = CulebraTests.sleep
        _v = CulebraTests.verbose
        print("Visiting %s" % (str(my_url)))
        try:
            self.vc.uiAutomatorHelper.findObject(bySelector='res@org.torproject.torbrowser_alpha:id/url_bar_title_scroll_view').clickAndWait(eventCondition='until:newWindow', timeout=_s*1000)
            self.vc.uiAutomatorHelper.findObject(bySelector='res@org.torproject.torbrowser_alpha:id/url_edit_text').setText(my_url)
            self.vc.sleep(1)
            self.vc.uiAutomatorHelper.findObject(bySelector='res@com.google.android.inputmethod.latin:id/key_pos_ime_action').clickAndWait(eventCondition='until:newWindow', timeout=_s*1000)
        except RuntimeError as e:
            print("Sorry, something went wrong :/")
            print(e)

    def testSomething(self):
        if not self.preconditions():
            self.fail('Preconditions failed')
        # load sites..
        sites = self.load_sites("10k_top_sites.txt")
        print("Sites loaded..")
        n_sites = len(sites)
        self.startTorBrowser_fromApps()
        self.vc.sleep(4)
        print("Tor Browser should be running...")
        i = 0
        while i < how_many_sites:
            self.go_to_url(sites[random.randint(0,n_sites)])
            self.vc.sleep(random.randint(min_browse, max_browse))
            i += 1
            print("Visited: %d, to visit: %d" % (i, how_many_sites - i))


if __name__ == '__main__':
    CulebraTests.main()
