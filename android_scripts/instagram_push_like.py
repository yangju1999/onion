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
# sleep after clicking some tab/key/
sleep_after_op = 3
likes = 100

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


    def startInstagram(self):
        package = 'com.instagram.android'
        activity = '.activity.MainTabActivity'
        component = package +"/"+ activity
        device, serialno = ViewClient.connectToDeviceOrExit()
        device.startActivity(component=component)
        self.vc.sleep(5)
        try:
            self.vc.uiAutomatorHelper.findObject(bySelector='desc@Home,clazz@android.widget.FrameLayout,text@$,package@com.instagram.android').clickAndWait(eventCondition='until:newWindow', timeout=4000)
        except (RuntimeError, AttributeError) as e:
            pass
        self.vc.sleep(5)


    def scroll_down(self, speed=10):
        self.vc.swipe(start=(54, 1069), end=(54, 459), steps=speed)


    def scroll_up(self, speed=10):
        self.vc.swipe(start=(54, 459), end=(54, 1069), steps=speed)

    def visit_profile(self):
        try:
            #click profile button (if visible)
            UiScrollable(self.vc.uiAutomatorHelper, uiSelector='clazz@android.widget.ListView,res@android:id/list,index@0,parentIndex@0,package@com.instagram.android').getChildByDescription(uiSelector='clazz@android.widget.LinearLayout,package@com.instagram.android', description="Profile picture", allowScrollSearch=True).click()
            self.vc.sleep(sleep_after_op)
            self.random_nav(random.randint(2, 9))
            self.vc.sleep(sleep_after_op)

        except  (RuntimeError, AttributeError) as e:
            print("no clickable profile.")
            return
        try:
            self.vc.uiAutomatorHelper.findObject(bySelector='desc@Home,clazz@android.widget.FrameLayout,text@$,package@com.instagram.android').click()

        except (RuntimeError, AttributeError) as e:
            print("Cazzoo cazzo cazzo! stocazzo stocazzo stocazzo! can't find home btn??")
            self.vc.sleep(sleep_after_op)
            self.vc.uiAutomatorHelper.pressBack()
            pass

    def pushlike(self):
        liked = True
        try:
            self.vc.sleep(2)
            UiScrollable(self.vc.uiAutomatorHelper, uiSelector='clazz@android.widget.ListView,res@android:id/list,index@0,parentIndex@0,package@com.instagram.android').getChildByDescription(uiSelector='desc@Like', description="Like", allowScrollSearch=True).click()
            #self.vc.sleep(3)
        except (RuntimeError, AttributeError) as e:
            print("Cazz...")
            #self.vc.sleep(3)
            liked = False
            pass
        return liked

    def random_nav(self, moves):
        for i in range(0,moves):
            if random.randint(0,100) > 15:
                self.scroll_down()
            else:
                self.scroll_up()

            self.vc.sleep(random.randint(2,5))
        self.vc.sleep(random.randint(2,6))


    def check_if_switched_to_camera(self):
        try:
            self.vc.uiAutomatorHelper.findObject(bySelector='res@com.instagram.android:id/camera_cover,clazz@android.view.View,text@$,package@com.instagram.android').click()
            print("Damn instagram camera...we have to press back.")
            self.vc.uiAutomatorHelper.pressBack()
        except AttributeError:
            pass


    def testSomething(self):
        #how many laikes before stop?
        print("Annamo a fa' sta sceneggiata (pure su instagram)! [cit.]")
        print("Target likes: " + str(likes))
        #likes = 400
        if not self.preconditions():
            self.fail("Preconditions Failed")
        print("Starting Instagram...")
        self.startInstagram()
        self.scroll_down()
        self.vc.sleep(1)
        self.scroll_down()
        self.vc.sleep(1)
        self.scroll_down()
        self.vc.sleep(4)
        print("Started.")
        i = 0

        while(i < likes):
            # 0-10, if >= 7 visit profile page
            self.check_if_switched_to_camera()
            if random.randint(0,10) >= 7:
                self.visit_profile()
                self.check_if_switched_to_camera()
            else:
                self.random_nav(random.randint(2, 20))
                self.check_if_switched_to_camera()
                # then like or not (like if >=4)
                if random.randint(0, 10) >= 3:
                    if self.pushlike():
                        i+=1
                        print("Given Likes: "+ str(i) + " remaining: " + str(likes-i))
                    else:
                        print("Post already liked or Error.")
            self.vc.sleep(sleep_after_op)

if __name__ == '__main__':
    CulebraTests.main()
    exit(0)
