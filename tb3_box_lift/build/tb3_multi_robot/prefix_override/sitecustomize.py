import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abhay/Desktop/varun/tb3_box_lift/install/tb3_multi_robot'
