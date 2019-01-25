#teste

import glob
import os

# define the path
currentDirectory = '/Exp/tensor_flow/1_s/running/svm.predictions/etf/test/plot'

# define the pattern
currentFilePattern = "analisis_*.dat"
output_file_name = "consolidation.dat"

output_file_path = os.path.join(currentDirectory, output_file_name)
currentPattern = os.path.join(currentDirectory, currentFilePattern)
print(currentPattern)

content = []
for currentFile in glob.glob(currentPattern): 
    print(currentFile)
    with open(currentFile, 'r') as f:
      content += f.readlines()

with open(output_file_path, 'w') as fo:
  fo.writelines(content)

