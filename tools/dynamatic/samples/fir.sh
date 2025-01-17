# Sample sequence of commands for Dynamatic frontend

# Indicate the path to Dynamatic's top-level directory here (leave unchanged if
# running the frontend from the top-level directory)
set-dynamatic-path  .

# Indicate the path the legacy Dynamatic's top-level directory here (required
# for write-hdl)
set-legacy-path     ../dynamatic-utils/legacy-dynamatic/dhls/etc/dynamatic

# Set the source file to run
set-src             integration-test/src/fir/fir.c

# Synthesize (from source to Handshake/DOT)
# Remove the flag to run smart buffer placement (requires Gurobi)
synthesize          --simple-buffers

# Generate the VHDL for the dataflow circuit
write-hdl

# Simulate using Modelsim
simulate

# Synthesize using Vivado
logic-synthesize

# Exit the frontend
exit