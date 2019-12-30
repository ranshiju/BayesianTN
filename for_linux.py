from sys import path
import os

proj = 'TNML'
cur_path = os.path.abspath(os.path.dirname(__file__))
proj = cur_path[:cur_path.find(proj) + len(proj)]
path.append(proj)
print('Added \'' + proj + '\' as a system path')
