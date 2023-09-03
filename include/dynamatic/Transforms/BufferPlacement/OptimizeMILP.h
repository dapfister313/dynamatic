//===- OptimizeMILP.h - optimize MILP model over CFDFC  ---------*- C++ -*-===//
//
// This file declares functions of MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractCFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {
namespace buffer {

/// Holds information about what type of buffer should be placed on a specific
/// channel.
struct PlacementResult {
  /// The number of transparent buffer slots that should be placed.
  unsigned numTrans = 0;
  /// The number of opaque buffer slots that should be placed.
  unsigned numOpaque = 0;
};

/// Holds MILP variables associated to every CFDFC unit. Note that a unit may
/// appear in multiple CFDFCs and so may have multiple sets of these variables.
struct UnitVars {
  /// Fluid retiming of tokens at unit's input (real).
  GRBVar retIn;
  /// Fluid retiming of tokens at unit's output. Identical to retiming at unit's
  /// input if the latter is combinational (real).
  GRBVar retOut;
};

/// Holds all MILP variables associated to a channel.
struct ChannelVars {
  /// Arrival time at channel's input (real).
  GRBVar tPathIn;
  /// Arrival time at channel's output (real).
  GRBVar tPathOut;
  /// Elastic arrival time at channel's input (real).
  GRBVar tElasIn;
  /// Elastic arrival time at channel's output (real).
  GRBVar tElasOut;
  /// Whether there is a buffer of any kind on the channel (binary).
  GRBVar bufPresent;
  /// Whether the buffer on the channel is opaque (binary).
  GRBVar bufIsOpaque;
  /// Number of buffer slots on the channel (integer).
  GRBVar bufNumSlots;
};

/// Holds all variables associated to a CFDFC. These are a set of variables for
/// each unit inside the CFDFC, a throughput variable for each channel inside
/// the CFDFC, and a CFDFC throughput varriable.
struct CFDFCVars {
  /// Maps each CFDFC unit to its retiming variables.
  DenseMap<Operation *, UnitVars> units;
  /// Channel throughput variables  (real).
  DenseMap<Value, GRBVar> channelThroughputs;
  /// CFDFC throughput (real).
  GRBVar thoughput;
};

/// Holds all variables that may be used in the MILP. These are a set of
/// variables for each CFDFC and a set of variables for each channel in the
/// function.
struct MILPVars {
  /// Mapping between each CFDFC and their related variables.
  DenseMap<CFDFC *, CFDFCVars> cfdfcs;
  /// Mapping between each circuit channel and their related variables.
  DenseMap<Value, ChannelVars> channels;
};

/// Holds the bulk of the logic for the smart buffer placement pass, which
/// expresses the buffer placement problem in dataflow circuits as an MILP
/// (mixed-integer linear program) whose solution indicates the location and
/// nature of buffers that must be placed in the circuit to achieve functional
/// correctness and high performance. This class relies on the prior
/// identification of all CFDFCs (choice-free dataflow circuits) inside an input
/// dataflow circuit to create throughput constraints and set the MILP's
/// objective to maximize. Gurobi's C++ API is used internally to manage the
/// MILP.
class BufferPlacementMILP {
public:
  /// Target clock period.
  const double targetPeriod;
  /// Maximum clock period.
  const double maxPeriod;

  /// Assumes that every value is used exactly once. Assumes that CFDFCs were
  /// extracted from the passed function.
  BufferPlacementMILP(circt::handshake::FuncOp funcOp,
                      llvm::MapVector<CFDFC *, bool> &cfdfcs,
                      TimingDatabase &timingDB, double targetPeriod,
                      double maxPeriod, GRBEnv &env, double timeLimit);

  /// Returns whether the custom buffer placement constraints derived from
  /// custom channel buffering properties attached to IR operations are
  /// satisfiable with respect to the component descriptions that the MILP
  /// constructor was called with.
  bool arePlacementConstraintsSatisfiable();

  /// Setups the entire MILP, first creating all variables, the all constraints,
  /// and finally setting the system's objective. After calling this function,
  /// the MILP is ready to be optimized.
  LogicalResult setup();

  /// Optimizes the MILP, which the function asssumes must have been setup
  /// before. On success, fills in the provided map with the buffer placement
  /// results, telling how each channel (equivalently, each MLIR value) must
  /// be bufferized according to the MILP solution.
  LogicalResult optimize(DenseMap<Value, PlacementResult> &placement);

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy constructor is deleted.
  BufferPlacementMILP(const BufferPlacementMILP &) = delete;

  /// The class manages a Gurobi model which should not be copied, hence the
  /// copy-assignment constructor is deleted.
  BufferPlacementMILP &operator=(const BufferPlacementMILP &) = delete;

protected:
  /// The Handshake function in which to place buffers.
  circt::handshake::FuncOp funcOp;
  /// Maps each CFDFC in the function to a boolean indicating whether it should
  /// be optimized. The CFDFCs must be included inside the Handshake function
  /// that was provide to the constructor.
  llvm::MapVector<CFDFC *, bool> &cfdfcs;
  /// Contains timing characterizations for dataflow components required to
  /// create the MILP constraints.
  TimingDatabase &timingDB;
  /// After construction, maps all channels (i.e, values) defined in the
  /// function to their specific channel buffering properties (unconstraining
  /// properties if none were explicitly specified).
  DenseMap<Value, ChannelBufProps> channels;
  /// Number of units (i.e., operations) in the function.
  unsigned numUnits;
  /// Gurobi model for creating/solving the MILP.
  GRBModel model;
  /// Contains all the variables used in the MILP.
  MILPVars vars;

  /// Adds all variables used in the MILP to the Gurobi model.
  LogicalResult createVars();

  /// Adds all variables related to the passed CFDFC to the Gurobi model. Each
  /// time this method is called, it must be with a different uid which is used
  /// to unique the name of each created variable. The CFDFC must be part of
  /// those that were provided to the constructor.
  LogicalResult createCFDFCVars(CFDFC &cfdfc, unsigned uid);

  /// Adds all variables related to all channels (regardless of whether they are
  /// part of a CFDFC) to the Gurobi model.
  LogicalResult createChannelVars();

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  LogicalResult addCustomChannelConstraints(ValueRange customChannels);

  /// Adds path constraints for all provided channels and units to the Gurobi
  /// model. All channels and units must be part of the Handshake function under
  /// consideration.
  LogicalResult addPathConstraints(ValueRange pathChannels,
                                   ArrayRef<Operation *> pathUnits);

  /// Adds elasticity constraints for all provided channels and units to the
  /// Gurobi model. All channels and units must be part of the Handshake
  /// function under consideration.
  LogicalResult addElasticityConstraints(ValueRange elasticChannels,
                                         ArrayRef<Operation *> elasticUnits);

  /// Adds throughput constraints for the provided CFDFC to the Gurobi model.
  /// The CFDFC must be part of those that were provided to the constructor.
  LogicalResult addThroughputConstraints(CFDFC &cfdfc);

  /// Adds the objective to the Gurobi model.
  LogicalResult addObjective();

  /// Adds pre-existing buffers that may exist as part of the units the channel
  /// connects to to the buffering properties. These are added to the minimum
  /// numbers of transparent and opaque slots so that the MILP is forced to
  /// place at least a certain quantity of slots on the channel and can take
  /// them into account in its constraints. Fails when buffering properties
  /// become unsatisfiable due to an increase in the minimum number of slots;
  /// succeeds otherwise.
  LogicalResult addInternalBuffers(Channel &channel);

  /// Removes pre-existing buffers that may exist as part of the units the
  /// channel connects to from the placement results. These are deducted from
  /// the numbers of transparent and opaque slots stored in the placement
  /// results. The latter are expected to specify more slots than what is going
  /// to be deducted (which should be guaranteed by the MILP constraints).
  void deductInternalBuffers(Channel &channel, PlacementResult &result);

private:
  /// Whether the MILP is unsatisfiable due to a conflict between user-defined
  /// channel properties and buffers internal to units (e.g., a channel declares
  /// that it should not be buffered yet the unit's IO which it connects to has
  /// a one-slot transparent buffer). Set by the class constructor.
  bool unsatisfiable = false;

  /// Helper method to run a closure on each input/output port pair of the
  /// provided operation.
  void forEachIOPair(Operation *op,
                     const std::function<void(Value, Value)> &callback);
};

} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
