import os
import qililab as ql
from qibo import Circuit, gates
ql.logger.setLevel(40)  # Set qililab's logger to a higher level so it only shows error messages
# Load the runcard
PLATFORM_PATH = os.getenv("RUNCARD")

# Construct the circuit
circuit = Circuit(2)

# Add some gates
circuit.add(gates.X(0))
circuit.add(gates.H(1))
circuit.add(gates.M(0,1))

# Execute the circuit
result = ql.execute(circuit,PLATFORM_PATH,nshots=1000)
print(result.counts())
