//===- UtilsForPlaceBuffers.cpp - functions for placing buffer  -*- C++ -*-===//
//
// This file implements function supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

void buffer::channel::print() {
  llvm::errs() << "opSrc: " << *(unitSrc->op) << " ---> ";
  llvm::errs() << "opDst: " << *(unitDst->op) << "\n";
}

bool buffer::isEntryOp(Operation *op, std::vector<Operation *> &visitedOp) {
  for (auto operand : op->getOperands())
    if (!operand.getDefiningOp())
      return true;
  return false;
}

int buffer::getBBIndex(Operation *op) {
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == BB_ATTR)
      return dyn_cast<IntegerAttr>(attr.getValue()).getValue().getZExtValue();
  }
  return -1;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  if (opDst->isProperAncestor(opSrc))
    return true;

  return false;
}

unit *buffer::getUnitWithOp(Operation *op, std::vector<unit *> &unitList) {
  for (auto u : unitList) {
    if (u->op == op)
      return u;
  }
  return nullptr;
}

void buffer::connectInChannel(unit *unitNode, channel *inChannel) {
  port *inPort = new port(inChannel->valPort);
  inPort->cntChannels.push_back(inChannel);
  unitNode->inPorts.push_back(inPort);
}

void buffer::dfsHandshakeGraph(Operation *opNode, std::vector<unit *> &unitList,
                               std::vector<Operation *> &visited,
                               channel *inChannel) {

  // ensure inChannel is marked as connected to the unit
  if (inChannel != nullptr)
    if (unit *unitNode = getUnitWithOp(opNode, unitList); unitNode != nullptr) {
      connectInChannel(unitNode, inChannel);
    }

  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }

  // marked as visited
  visited.push_back(opNode);

  // initialize the unit, if the unit is initialized by its predecessor,
  // get from the list. Otherwise, create a new unit and add to the list
  unit *unitNode = getUnitWithOp(opNode, unitList);
  if (!unitNode) {
    unitNode = new unit(opNode);
    unitList.push_back(unitNode);
  }

  if (inChannel != nullptr)
    connectInChannel(unitNode, inChannel);

  // dfs the successor operation
  for (auto resOperand : opNode->getResults()) {
    // initialize the out port
    port *outPort = new port(&resOperand);
    for (auto sucOp : resOperand.getUsers()) {
      // create the channel connected to the outport
      unit *sucUnit  = getUnitWithOp(sucOp, unitList);
      // if the successor unit not exists, create a new one
      if (!sucUnit) {
        sucUnit = new unit(sucOp);
        unitList.push_back(sucUnit);
      }
      channel *outChannel = new channel(unitNode, sucUnit, &resOperand);
      outChannel->isBackEdge = isBackEdge(opNode, sucOp);
      outPort->cntChannels.push_back(outChannel);

      // dfs the successor operation
      dfsHandshakeGraph(sucOp, unitList, visited, outChannel);
    }
    unitNode->outPorts.push_back(outPort);
  }
}

/// ================== dataFlowCircuit Function ================== ///
void buffer::dataFlowCircuit::printCircuits() {
  for (auto unit : units) {
    llvm::errs() << "===========================\n";
    llvm::errs() << "operation: " << *(unit->op) << "\n";
    llvm::errs() << "-------------inPorts: \n";
    for (auto port : unit->outPorts)
      for (auto ch : port->cntChannels)
        ch->print();
    llvm::errs() << "-------------outPorts: \n";
    for (auto port : unit->inPorts)
      for (auto ch : port->cntChannels)
        ch->print();
  }
}

std::vector<std::vector<float>>
buffer::dataFlowCircuit::readInfoFromFile(const std::string &filename) {
  std::vector<std::vector<float>> info;

  std::ifstream file(filename);
  assert(file.is_open() && "Error opening delay info file");

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
      float num = std::stof(value);
      row.push_back(num);
    }

    assert(!row.empty() && "Error reading delay info file");
    info.push_back(row);
  }

  file.close();

  return info;
}

void buffer::dataFlowCircuit::createMILPVars(GRBModel &modelMILP, 
                          std::vector<unit *> &units,
                          std::vector<channel *> &channels,
                          std::vector<port *> &ports,
                          std::map<std::string, GRBVar> &timeVars,
                          std::map<std::string, GRBVar> &elasticVars,
                          std::map<std::string, GRBVar> &thrptVars,
                          std::map<std::string, GRBVar> &bufferVars,
                          std::map<std::string, GRBVar> &retimeVars,
                          std::map<std::string, GRBVar> &outputVars) {
  // clear the variables
  timeVars.clear();
  thrptVars.clear();
  bufferVars.clear();
  retimeVars.clear();
  // create the the circuit throuput variables
  thrptVars["thrpt"] = modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);

  for (size_t i = 0; i < units.size(); i++) {
    unit *unit = units[i];
    // create the retiming variables for the throughput
    retimeVars["retimeIn_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    retimeVars["retimeOut_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    retimeVars["retimeBubble_" + std::to_string(i)] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);

    // time path defined in input port
    for (size_t j = 0; j < unit->inPorts.size(); j++) {
      port *inP = unit->inPorts[j];
      if (std::find(ports.begin(), ports.end(), inP) != ports.end()) {
        timeVars["timeIn_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeInValid_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeInReady_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        elasticVars["timeInElastic_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
      }
    }

    // time path defined in output port 
    for (size_t j = 0; j < unit->outPorts.size(); j++) {
      port *outP = unit->outPorts[j];
      // if the port is in the path, create time path variables
      if (std::find(ports.begin(), ports.end(), outP) != ports.end()) {
        timeVars["timeOut_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeOutValid_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        timeVars["timeOutReady_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
        elasticVars["timeOutElastic_" + std::to_string(i) + "_" + std::to_string(j)] =
            modelMILP.addVar(0.0, INFINITY, 0.0, GRB_CONTINUOUS);
      }
    }

  }

  // create throughput variables for the channels
  for(int i = 0; i < channels.size(); i++) {
    channel *channel = channels[i];
    int srcIndex = findUnitIndex(channel->unitSrc->op);
    int dstIndex = findUnitIndex(channel->unitDst->op);
    assert((srcIndex != -1 && dstIndex != -1) && "Error finding unit index");

    std::string varName = std::to_string(i) + 
                          "_u" + std::to_string(srcIndex) + 
                          "_u" + std::to_string(dstIndex);
    thrptVars["thTokens_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    thrptVars["thBubbles_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
    thrptVars["bufferFlopValid_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    thrptVars["bufferFlopReady_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    outputVars["bufferFlop_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
    outputVars["bufferSlots_" + varName] =
        modelMILP.addVar(0.0, INFINITY, 0.0, GRB_INTEGER);
    outputVars["hasBuffer_" + varName] =
        modelMILP.addVar(0.0, 1.0, 0.0, GRB_BINARY);
  }
}

static port* getSrcPort(channel *ch, unit * unitNode ) {
  for (auto p : unitNode->outPorts) 
    if (p->opVal == ch->valPort) 
      return p;
    
  return nullptr;
}

static int getUnitIndex(unit *unitNode, std::vector<unit *> &unitList) {
  for (int i = 0; i < unitList.size(); i++) {
    if (unitList[i] == unitNode)
      return i;
  }
  return -1;
}

static int getPortIndex(Value *val, std::vector<port *> &portList) {
  for (int i = 0; i < portList.size(); i++) {
    if (portList[i]->opVal == val)
      return i;
  }
  return -1;
}

static bool hasConectedChannels(port *p, std::vector<channel *> &channels) {
  for (auto ch : channels) 
    if (ch->valPort == p->opVal)
      return true;
  
  return false;
}


void buffer::dataFlowCircuit::createPathConstraints(GRBModel &modelMILP, 
                                  std::map<std::string, GRBVar> &timeVars,
                                  std::map<std::string, GRBVar> &bufferVars) {
  // create constraints in the path alongside the channels
  for (size_t i = 0; i < channels.size(); i++) {
    channel *ch = channels[i];
    int srcIndex = getUnitIndex(ch->unitSrc, this->units);
    int j = getPortIndex(ch->valPort, ch->unitSrc->outPorts);

    assert((srcIndex != -1 && j != -1) && "Unit or port not found in the list");
    GRBVar &timeIn = timeVars["timeOut_" + std::to_string(srcIndex) + "_" + std::to_string(j)];
    // timeIn <= period
    modelMILP.addConstr(timeIn <= this->targetCP);

    int dstIndex = getUnitIndex(ch->unitDst, this->units);
    j = getPortIndex(ch->valPort, ch->unitDst->inPorts);
    assert((dstIndex != -1 && j != -1) && "Unit or port not found in the list");
    GRBVar &timeOut = timeVars["timeIn_" + std::to_string(dstIndex) + "_" + std::to_string(j)];
    // timeOut <= period
    modelMILP.addConstr(timeOut <= this->targetCP);

    std::string varName = std::to_string(i) + 
                          "_u" + std::to_string(srcIndex) + 
                          "_u" + std::to_string(dstIndex);
    GRBVar &R_flop = bufferVars["bufferFlop_" + varName];
    // v2 >= v1 - 2*period*R
    modelMILP.addConstr(timeOut >= timeIn - 2 * this->targetCP * R_flop);

    // v2 >= Buffer Delay
    modelMILP.addConstr(timeOut >= this->bufferDelay);
  }

  // create constraints for the units
  for (size_t i = 0; i < units.size(); i++) {
    unit *uNode = units[i];
    assert (uNode->delay <= this->targetCP 
            && "Unit delay is greater than the target period");
    
    for (size_t j = 0; j < uNode->inPorts.size(); j++) 
      // get the input port(if used) of the unit
      if (hasConectedChannels(uNode->inPorts[j], this->channels)) {
        GRBVar &timeIn = timeVars["timeIn_" + std::to_string(i) + "_" + std::to_string(j)];
        for (size_t k = 0; k < uNode->outPorts.size(); k++) 
          // get the output port(if used) of the unit
          if (hasConectedChannels(uNode->outPorts[k], this->channels)) {
            GRBVar &timeOut = timeVars["timeOut_" + std::to_string(i) + "_" + std::to_string(k)];

            // define time constraints for combinational units
            // tIn |--> input port -> units -> output port --> |tOut
            // t_out >= t_in + d_in + d + d_out
            if (uNode->latency == 0.0) 
              modelMILP.addConstr(timeOut >= timeIn + 
                                          uNode->inPorts[j]->portDelay + uNode->delay + 
                                          uNode->outPorts[k]->portDelay);
            // define time constraints for sequential units
            else {
              // t_out = d_out
              modelMILP.addConstr(timeOut == uNode->outPorts[k]->portDelay);
              // t_in + d_in <= period
              modelMILP.addConstr(timeIn + uNode->inPorts[j]->portDelay <= this->targetCP);
            }
          }
      }
  }  

}

void buffer::dataFlowCircuit::createMILPModel(BufferPlacementStrategy &strategy,
                          std::map<std::string, GRBVar> &outputVars) {
  // init the model
  GRBEnv env = GRBEnv(true);
  env.start();
  GRBModel modelMILP = GRBModel(env);

  // double period = strategy.period;
  // double periodMax = strategy.periodMax;

  // internal variables
  std::map<std::string, GRBVar> timeVars;
  std::map<std::string, GRBVar> elasticVars;
  std::map<std::string, GRBVar> bufferVars;
  std::map<std::string, GRBVar> retimeVars;
  std::map<std::string, GRBVar> thrptVars;

  createMILPVars(modelMILP, this->units, this->channels, this->ports, 
                 timeVars, elasticVars, thrptVars, bufferVars, retimeVars, outputVars);

  createPathConstraints(modelMILP, timeVars, bufferVars);

  // create the variables
}