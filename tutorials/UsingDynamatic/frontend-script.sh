# Set the source file to run
set-src             tutorials/UsingDynamatic/loop_accumulate.c

# Compile (from source to Handshake IR/DOT)
# Provide the --simple-buffers flag to `compile` to use a simple buffering
# strategy which does not require an MILP solver.
compile             

# Generate the VHDL design corresponding to the source code
write-hdl

# Simulate and verify using Modelsim
simulate

# Exit the frontend
exit
