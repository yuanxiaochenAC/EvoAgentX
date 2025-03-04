from evoagentx.tools.interpreter_docker import DockerInterpreter

print("Creating DockerInterpreter instance...")
interpreter = DockerInterpreter(require_confirm=False, print_stdout=True, image_tag = "fundingsocietiesdocker/python3.9-slim")
code = """

#import numpy as np
#print(np.array([1, 2, 3]))
import os

print('Hello from Docker!')
"""

print("Code: {}".format(code))
result = interpreter.run(code, "python")
print("Execution Result:", result)