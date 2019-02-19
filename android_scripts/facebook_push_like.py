#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C) 2013-2018  Diego Torres Milano
Created on 2018-11-11 by CulebraTester
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
# target likes
likes = 30
# post push btn sleep
push_sleep = 3
# startup sleep
startup_sleep = 5
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
    def startFB_fromApps(self):
        package = 'com.facebook.katana'
        activity = '.LoginActivity'
        component = package +"/"+ activity
        device, serialno = ViewClient.connectToDeviceOrExit()
        device.startActivity(component=component)
        self.vc.sleep(startup_sleep)


    def scroll_down(self, speed=10):
        self.vc.swipe(start=(347, 1164), end=(326, 261), steps=speed)

    def scroll_up(self, speed=10):
        self.vc.swipe(start=(364, 263), end=(331, 1160), steps=speed)

    def pushlike(self):
        liked = True
        try:
            UiScrollable(self.vc.uiAutomatorHelper, uiSelector='clazz@android.support.v7.widget.RecyclerView,res@android:id/list,index@0,parentIndex@0,package@com.facebook.katana').getChildByDescription(uiSelector='desc@Like button. Double tap and hold to react.', description="Like button. Double tap and hold to react.", allowScrollSearch=True).click()
            self.vc.sleep(push_sleep)
        except RuntimeError as e:
            print("Cazz...(no like btn or already liked post)")
            print(e)
            liked = False
            pass
        return liked

    def random_nav(self, moves):
        for i in range(0,moves):
            # 60% prob to scroll down
            if random.randint(0,100) >= 40:
                self.scroll_down()
            else:
                self.scroll_up()

            self.vc.sleep(random.randint(2,3))
        self.vc.sleep(random.randint(2,4))

    def testSomething(self):
        # how many likes before stop?
        print("Annamo a fa' sta sceneggiata! [cit.]")

        if not self.preconditions():
            self.fail("Preconditions Failed")
        self.startFB_fromApps()
        self.scroll_down()
        self.vc.sleep(2)
        self.scroll_down()
        self.vc.sleep(2)
        self.scroll_down()
        self.vc.sleep(4)
        i = 0
        while(i < likes):
            self.random_nav(random.randint(2, 20))
            # then like or not if >= 7
            if random.randint(0, 10) >= 7:
                self.vc.sleep(2)
                if self.pushlike():
                    self.vc.sleep(2)
                    i += 1
                    print("Given Likes: " + str(i) + " remaining: " + str(likes - i))




if __name__ == '__main__':
    CulebraTests.main()
    exit(0)
