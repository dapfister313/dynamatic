//===- ParseCircuitJson.cpp -Parse circuit json file  -----------*- C++ -*-===//
//
// This file contains functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

using namespace dynamatic;
using namespace dynamatic::buffer;

std::string buffer::getOperationShortStrName(Operation *op) {
  std::string fullName = op->getName().getStringRef().str();
  size_t pos = fullName.find('.');
  return fullName.substr(pos + 1);
}

static inline unsigned getPortWidth(Value channel) {
  unsigned portBitWidth = bitwidth::CPP_MAX_WIDTH;
  if (isa<NoneType>(channel.getType()))
    return 0;
  if (isa<IntegerType>(channel.getType()))
    portBitWidth = channel.getType().getIntOrFloatBitWidth();
  return portBitWidth;
}

static double
getBitWidthMatchedTimeInfo(unsigned bitWidth,
                           std::vector<std::pair<unsigned, double>> &timeInfo) {
  double delay;
  // Sort the vector based on pair.first (unsigned)
  std::sort(
      timeInfo.begin(), timeInfo.end(),
      [](const std::pair<unsigned, double> &a,
         const std::pair<unsigned, double> &b) { return a.first < b.first; });
  for (const auto &[width, opDelay] : timeInfo)
    if (width >= bitWidth)
      return opDelay;

  // return the delay of the largest bitwidth
  return timeInfo.end()->second;
}

double buffer::getPortDelay(Value channel,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
                            std::string direction) {
  std::string opName;
  if (direction == "in") {
    opName = getOperationShortStrName(channel.getDefiningOp());
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].inPortDataDelay);

  } else if (direction == "out") {
    auto dstOp = channel.getUsers().begin();
    // TODO: handle multiple users
    // assert(dstOp == channel.getUsers().end() && "There are multiple users!");
    opName = getOperationShortStrName(*dstOp);
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].outPortDataDelay);
  }
  return 0.0;
}

double buffer::getUnitDelay(Operation *op,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
                            std::string type) {
  double delay;
  std::string opName = getOperationShortStrName(op);
  // check whether delay information exists
  if (unitInfo.find(opName) == unitInfo.end()) {
    llvm::errs() << "Cannot find " << opName << " in unitInfo\n";
    return 0.0;
  }

  // get delay w.r.t to bitwidth
  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  if (type == "data")
    delay =
        getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].dataDelay);
  else if (type == "valid")
    delay = unitInfo[opName].validDelay;
  else if (type == "ready")
    delay = unitInfo[opName].readyDelay;
  // llvm::errs() << opName << " : " << delay << "\n";
  return delay;
}

double
buffer::getCombinationalDelay(Operation *op,
                              std::map<std::string, buffer::UnitInfo> &unitInfo,
                              std::string type) {
  std::string opName = getOperationShortStrName(op);
  if (unitInfo.find(getOperationShortStrName(op)) == unitInfo.end())
    return 0.0;

  double inPortDelay, outPortDelay;
  double unitDelay = getUnitDelay(op, unitInfo, type);

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));

  if (type == "data") {
    inPortDelay = getBitWidthMatchedTimeInfo(unitBitWidth,
                                             unitInfo[opName].inPortDataDelay);
    outPortDelay = getBitWidthMatchedTimeInfo(
        unitBitWidth, unitInfo[opName].outPortDataDelay);
  } else if (type == "valid") {
    inPortDelay = unitInfo[opName].inPortValidDelay;
    outPortDelay = unitInfo[opName].outPortValidDelay;
  } else if (type == "ready") {
    inPortDelay = unitInfo[opName].inPortReadyDelay;
    outPortDelay = unitInfo[opName].outPortReadyDelay;
  }
  return unitDelay + inPortDelay + outPortDelay;
}

double
buffer::getUnitLatency(Operation *op,
                       std::map<std::string, buffer::UnitInfo> &unitInfo) {
  std::string opName = getOperationShortStrName(op);
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  double latency =
      getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].latency);
  return latency;
}

static void parseBitWidthPair(json jsonData,
                              std::vector<std::pair<unsigned, double>> &data) {
  for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
    auto key = stoi(it.key());
    double value = it.value();
    data.emplace_back(key, value);
  }
}

// Function to parse the JSON string and extract the required data
LogicalResult buffer::parseJson(const std::string &jsonFile,
                                std::map<std::string, UnitInfo> &unitInfo) {

  // reserve for read buffers information to be used in the channel
  // constraints
  size_t pos = 0;
  std::vector<std::string> opNames = {
      "cmpi",          "addi",    "subi",   "muli",    "extsi",
      "d_load",        "d_store", "merge",  "addf",    "subf",
      "mulf",          "divui",   "divsi",  "divf",    "cmpf",
      "control_merge", "fork",    "return", "cond_br", "end",
      "andi",          "ori",     "xori",   "shli",    "shrsi",
      "shrui",         "select",  "mux"};
  std::string opName;

  std::ifstream file(jsonFile);
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file.\n";
  }

  // Read the file contents into a string
  json data;
  file >> data;
  for (auto op : opNames) {
    auto unitInfoJson = data[op];
    auto latencyJson = unitInfoJson["latency"];
    // parse the bitwidth and its corresponding latency
    parseBitWidthPair(unitInfoJson["latency"], unitInfo[op].latency);
    parseBitWidthPair(unitInfoJson["delay"]["data"], unitInfo[op].dataDelay);
    parseBitWidthPair(unitInfoJson["inport"]["delay"]["data"],
                      unitInfo[op].inPortDataDelay);
    parseBitWidthPair(unitInfoJson["outport"]["delay"]["data"],
                      unitInfo[op].outPortDataDelay);

    double validDelay = unitInfoJson["delay"]["valid"]["1"];
    unitInfo[op].validDelay = validDelay;
    double readyDelay = unitInfoJson["delay"]["ready"]["1"];
    unitInfo[op].readyDelay = readyDelay;

    unitInfo[op].inPortValidDelay =
        unitInfoJson["inport"]["delay"]["valid"]["1"];
    unitInfo[op].inPortReadyDelay =
        unitInfoJson["inport"]["delay"]["ready"]["1"];

    unitInfo[op].outPortValidDelay =
        unitInfoJson["outport"]["delay"]["valid"]["1"];
    unitInfo[op].outPortReadyDelay =
        unitInfoJson["outport"]["delay"]["ready"]["1"];

    unitInfo[op].inPortTransBuf = unitInfoJson["inport"]["transparentBuffer"];
    unitInfo[op].inPortOpBuf = unitInfoJson["inport"]["opaqueBuffer"];

    unitInfo[op].outPortTransBuf = unitInfoJson["outport"]["transparentBuffer"];
    unitInfo[op].outPortOpBuf = unitInfoJson["outport"]["opaqueBuffer"];

    if (unitInfoJson.is_discarded())
      return failure();
  }

  return success();
}
