Setup smartphone and computer as explained in `README_android_scripts.md`.
1. Requirements
  * Facebook app must be installed and fb account already logged in.
  * You must have some friends and/or some liked pages (otherwise home will be empty)
  * In case you are using a new and empty account you can add pages in `top_500_fb.txt` to your liked pages.
2. Edit `facebook_push_like.py` (if needed), parmeters:
    * `likes`, how many likes before stopping (default: 30)
    * `push_sleep`, how many seconds wait after a like (default: 3)
3. Run script from terminal: `python facebook_push_like.py`
