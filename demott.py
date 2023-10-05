import pathlib
import sys
import os
maindir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(maindir, "LIHQ", "first_order_model"))
sys.path.append(os.path.join(maindir, "LIHQ", "procedures"))

from LIHQ import runLIHQ
runLIHQ.run(face="123456.png")