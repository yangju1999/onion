To run android scripts you need:
-- An Android phone (tested on Android 6.0.1, i don't know if it works on previous/next versions)
-- Python 2.7 (yes, 2.7)
-- Required Python libraries (see requirements_android_scripts.txt)
-- Android adb (and other tools) installed
-- The following instructions work on MacOS X and should work on Linux, i didn't test on Windows.
Part of these instructions are from [2], so refer to [2] for more info.

-- Smartphone setup
1. Install from play store Culebra Tester Instrumentation
2. Install Culebra Tester (not available on play store, get a copy here [1] or from other sources)
3. Enable Developer Options on Android Phone
4. Enable USB Debugging (Settings > Developer Options)
5. Connect the smartphone to computer using an USB cable, on smartphone should pop up a request: "Allow USB debugging."
   Tick the option "Always allow from this computer" and click "OK"

-- Connect and run the scripts   
  1 - Start Culebra Tester
  Now there are 2 choices:
  a. Use Culebra via Network.
     a1. - Smartphone and Computer are in the same network (smartphone can reach computer and viceversa)
     a2. - Get Smartphone's IP, then open a browser and connect to http://Smartphone_IP:9999
     a3. - Follow the instructions on screen: open a terminal, paste the command "curl -s http://Smartphone_IP:9999/Assets/runinstrumentation.sh | bash"
     and execute it. [DON'T close the terminal]
     a4. - Now you should see smartphone's screen mirrored in web browser.
     a5. - Done.

  b. Use Culebra via USB.
     b1. - Open a terminal
     b2. - Enter the following command: "adb forward tcp:9999 tcp:9999" (this works if there is only one smartphone connected via USB)
     b3. - Open a browser and connect to http://localhost:9999
     b4. - Follow the instructions on screen: open a terminal, paste the command "curl -s http://localhost:9999/Assets/runinstrumentation.sh | bash"
     and execute it. [DON'T close the terminal]
     b5. - Now you should see smartphone's screen mirrored in web browser.
     b6. - Done.

Steps from a1. to a3. and from b1. to b3. are needed only to check if everything is working properly, you can skip them after first time setup.     
  2 - Run scripts
    - open a terminal and go to android_scripts folder
    - execute: "python script_name.py"

If a script crashes you (may) have to:
 - kill the script (if it's stuck)
 - restart from a3. or b3., if Culebra Tester is still running on smarphone, otherwise restart from a1. or b1.

[1] https://apkpure.com/culebra-tester/com.dtmilano.android.culebratester
[2] https://github.com/dtmilano/CulebraTester-public/wiki/Running-the-Instrumentation
