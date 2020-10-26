#run all
import os
addr = 'C:/Rebalancing/nowModel/centralCode/'
os.system('python '+addr+'Beta.py')
os.system('python '+addr+'Mu.py')
os.system('python '+addr+'Gamma.py')
os.system('python '+addr+'Delta.py')
os.system('python '+addr+'N.py')
os.system('python '+addr+'Bbar.py')
os.system('python '+addr+'Dbar.py')