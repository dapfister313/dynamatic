//===- ParseCircuitJson.cpp - Parse circuit json file  ----------*- C++ -*-===//
//
// This file contains functions to parse the elements in the circuit json file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
<<<<<<< HEAD
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
=======
#include "llvm/Support/JSON.h"
#include <fstream>
#include <iostream>
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b

using namespace dynamatic;
using namespace dynamatic::buffer;

/// Get the full operation name
static std::string getOperationFullName(Operation *op) {
  std::string fullName = op->getName().getStringRef().str();
  return fullName;
}

std::string buffer::getOperationShortStrName(Operation *op) {
  std::string fullName = getOperationFullName(op);
  size_t pos = fullName.find('.');
  return fullName.substr(pos + 1);
}

/// For a channel, indicated by a value, get its port width if exists,
/// otherwise, return CPP_MAX_WIDTH
static unsigned getPortWidth(Value channel) {
  unsigned portBitWidth = bitwidth::CPP_MAX_WIDTH;
  if (isa<NoneType>(channel.getType()))
    return 0;
  if (isa<IntegerType, FloatType>(channel.getType()))
    portBitWidth = channel.getType().getIntOrFloatBitWidth();
  return portBitWidth;
}

/// Get the precise time information w.r.t to the bitwidth from a vector store
/// the {bitwidth, time} info.
static double
getBitWidthMatchedTimeInfo(unsigned bitWidth,
                           std::vector<std::pair<unsigned, double>> &timeInfo) {
<<<<<<< HEAD
  double delay;
=======
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  // Sort the vector based on pair.first (unsigned)
  std::sort(
      timeInfo.begin(), timeInfo.end(),
      [](const std::pair<unsigned, double> &a,
         const std::pair<unsigned, double> &b) { return a.first < b.first; });
  for (const auto &[width, opDelay] : timeInfo)
    if (width >= bitWidth)
      return opDelay;

  // return the delay of the largest bitwidth
<<<<<<< HEAD
  return timeInfo.end()->second;
}

double buffer::getMixedDelay(Operation *op,
                             std::map<std::string, buffer::UnitInfo> &unitInfo,
                             std::string type) {
  double delay;
  std::string opName = getOperationFullName(op);
  // check whether delay information exists
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  if (type == "VR")
    return unitInfo[opName].VR;
  else if (type == "DV")
    return unitInfo[opName].DV;
  else if (type == "DR")
    return unitInfo[opName].DR;
  else if (type == "VC")
    return unitInfo[opName].VC;
  else if (type == "VD")
    return unitInfo[opName].VD;

  return 0.0;
=======
  return (timeInfo.end() - 1)->second;
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
}

double buffer::getPortDelay(Value channel,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
<<<<<<< HEAD
                            std::string direction) {
=======
                            std::string &direction) {
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  std::string opName;
  if (direction == "in") {
    opName = getOperationFullName(channel.getDefiningOp());
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].inPortDataDelay);

  } else if (direction == "out") {
    auto dstOp = channel.getUsers().begin();
    opName = getOperationFullName(*dstOp);
    unsigned portBitWidth = getPortWidth(channel);
    if (unitInfo.find(opName) != unitInfo.end())
      return getBitWidthMatchedTimeInfo(portBitWidth,
                                        unitInfo[opName].outPortDataDelay);
  }
  return 0.0;
}

double buffer::getUnitDelay(Operation *op,
                            std::map<std::string, buffer::UnitInfo> &unitInfo,
<<<<<<< HEAD
                            std::string type) {
=======
                            std::string &type) {
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  double delay;
  std::string opName = getOperationFullName(op);
  // check whether delay information exists
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  // get delay w.r.t to bitwidth
  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
  if (type == "data")
    delay =
        getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].dataDelay);
  else if (type == "valid")
    delay = unitInfo[opName].validDelay;
  else if (type == "ready")
    delay = unitInfo[opName].readyDelay;
<<<<<<< HEAD
=======
  else
    delay = 0.0;

>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  return delay;
}

double
buffer::getCombinationalDelay(Operation *op,
                              std::map<std::string, buffer::UnitInfo> &unitInfo,
                              std::string type) {
  std::string opName = getOperationFullName(op);
  if (unitInfo.find(getOperationFullName(op)) == unitInfo.end())
    return 0.0;

<<<<<<< HEAD
  double inPortDelay, outPortDelay;
  double unitDelay = getUnitDelay(op, unitInfo, type);

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));
=======
  double inPortDelay = 0.0;
  double outPortDelay = 0.0;
  double unitDelay = getUnitDelay(op, unitInfo, type);

  unsigned unitBitWidth = 1;
  for (auto operand : op->getOperands())
    unitBitWidth = std::max(unitBitWidth, getPortWidth(operand));
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b

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
<<<<<<< HEAD
=======

>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  return unitDelay + inPortDelay + outPortDelay;
}

double
buffer::getUnitLatency(Operation *op,
                       std::map<std::string, buffer::UnitInfo> &unitInfo) {
  std::string opName = getOperationFullName(op);
  if (unitInfo.find(opName) == unitInfo.end())
    return 0.0;

  unsigned unitBitWidth = getPortWidth(op->getOperand(0));

  double latency =
      getBitWidthMatchedTimeInfo(unitBitWidth, unitInfo[opName].latency);

  return latency;
}

LogicalResult
buffer::setChannelBufProps(std::vector<Value> &channels,
<<<<<<< HEAD
                           DenseMap<Value, ChannelBufProps> &ChannelBufProps,
=======
                           DenseMap<Value, ChannelBufProps> &channelBufProps,
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
                           std::map<std::string, UnitInfo> &unitInfo) {
  for (auto &ch : channels) {
    Operation *srcOp = ch.getDefiningOp();
    Operation *dstOp = *(ch.getUsers().begin());

    // skip the channel that is the block argument
    if (!srcOp || !dstOp)
      continue;

    std::string srcName = srcOp->getName().getStringRef().str();
    std::string dstName = dstOp->getName().getStringRef().str();
    // set merge with multiple input to have at least one transparent buffer
    if (isa<handshake::MergeOp>(srcOp) && srcOp->getNumOperands() > 1)
<<<<<<< HEAD
      ChannelBufProps[ch].minTrans = 1;

    // TODO: set selectOp always select the frequent input
    if (isa<arith::SelectOp>(srcOp))
      if (srcOp->getResult(0) == ch) {
        ChannelBufProps[ch].maxTrans = 0;
        ChannelBufProps[ch].minOpaque = 0;
=======
      channelBufProps[ch].minTrans = 1;

    // TODO: set selectOp always select the frequent input
    if (isa<arith::SelectOp>(dstOp))
      if (dstOp->getOperand(2) == ch) {
        channelBufProps[ch].maxTrans = 0;
        channelBufProps[ch].minOpaque = 0;
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
      }

    if (isa<handshake::MemoryControllerOp>(srcOp) ||
        isa<handshake::MemoryControllerOp>(dstOp)) {
<<<<<<< HEAD
      ChannelBufProps[ch].minOpaque = 0;
      ChannelBufProps[ch].maxTrans = 0;
=======
      channelBufProps[ch].maxOpaque = 0;
      channelBufProps[ch].maxTrans = 0;
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
    }

    // set channel buffer properties w.r.t to input file
    if (unitInfo.count(srcName) > 0) {
<<<<<<< HEAD
      ChannelBufProps[ch].minTrans += unitInfo[srcName].outPortTransBuf;
      ChannelBufProps[ch].minOpaque += unitInfo[srcName].outPortOpBuf;
    }

    if (unitInfo.count(dstName) > 0) {
      ChannelBufProps[ch].minTrans += unitInfo[dstName].inPortTransBuf;
      ChannelBufProps[ch].minOpaque += unitInfo[dstName].inPortOpBuf;
    }

    if (ChannelBufProps[ch].minTrans > 0 && ChannelBufProps[ch].minOpaque > 0)
=======
      channelBufProps[ch].minTrans += unitInfo[srcName].outPortTransBuf;
      channelBufProps[ch].minOpaque += unitInfo[srcName].outPortOpBuf;
    }

    if (unitInfo.count(dstName) > 0) {
      channelBufProps[ch].minTrans += unitInfo[dstName].inPortTransBuf;
      channelBufProps[ch].minOpaque += unitInfo[dstName].inPortOpBuf;
    }

    if (channelBufProps[ch].minTrans > 0 && channelBufProps[ch].minOpaque > 0)
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
      return failure(); // cannot satisfy the constraint
  }
  return success();
}

<<<<<<< HEAD
/// Parse the JSON data to a vector of pair {bitwidth, info}
static void parseBitWidthPair(json jsonData,
                              std::vector<std::pair<unsigned, double>> &data) {
  for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
    auto key = stoi(it.key());
    double value = it.value();
    data.emplace_back(key, value);
=======
static void parseBitWidthPair(llvm::json::Object &jsonData,
                              std::vector<std::pair<unsigned, double>> &data) {
  for (const auto &[bitWidth, value] : jsonData) {
    llvm::StringRef bitKey(bitWidth);
    unsigned key = std::stoi(bitKey.str());
    double info = value.getAsNumber().value();
    data.emplace_back(key, info);
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  }
}

LogicalResult buffer::parseJson(const std::string &jsonFile,
                                std::map<std::string, UnitInfo> &unitInfo) {

  // Operations that is supported to use its time information.
<<<<<<< HEAD
  size_t pos = 0;
=======
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  std::vector<std::string> opNames = {
      "arith.cmpi",        "arith.addi",
      "arith.subi",        "arith.muli",
      "arith.extsi",       "handshake.d_load",
      "handshake.d_store", "handshake.merge",
      "arith.addf",        "arith.subf",
      "arith.mulf",        "arith.divui",
      "arith.divsi",       "arith.divf",
      "arith.cmpf",        "handshake.control_merge",
      "handshake.fork",    "handshake.d_return",
      "handshake.cond_br", "handshake.end",
      "arith.andi",        "arith.ori",
      "arith.xori",        "arith.shli",
      "arith.shrsi",       "arith.shrui",
      "arith.select",      "handshake.mux"};
  std::string opName;

<<<<<<< HEAD
  std::ifstream file(jsonFile);
  if (!file.is_open()) {
=======
  std::ifstream inputFile(jsonFile);
  if (!inputFile.is_open()) {
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
    llvm::errs() << "Failed to open file.\n";
    return failure();
  }

<<<<<<< HEAD
  // Read the file contents into a string
  json data;
  file >> data;
  for (std::string &op : opNames) {
    auto unitInfoJson = data[op];
    auto latencyJson = unitInfoJson["latency"];
    // parse the bitwidth and its corresponding latency for data
    parseBitWidthPair(unitInfoJson["latency"], unitInfo[op].latency);
    parseBitWidthPair(unitInfoJson["delay"]["data"], unitInfo[op].dataDelay);
    parseBitWidthPair(unitInfoJson["inport"]["delay"]["data"],
                      unitInfo[op].inPortDataDelay);
    parseBitWidthPair(unitInfoJson["outport"]["delay"]["data"],
                      unitInfo[op].outPortDataDelay);

    // parse the bitwidth and its corresponding latency for valid and ready
    // The valid and ready signal is 1 bit
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

    unitInfo[op].VR = unitInfoJson["delay"]["VR"];
    unitInfo[op].DV = unitInfoJson["delay"]["CV"];
    unitInfo[op].DR = unitInfoJson["delay"]["CR"];
    unitInfo[op].VD = unitInfoJson["delay"]["VD"];
    unitInfo[op].VC = unitInfoJson["delay"]["VC"];

    unitInfo[op].inPortTransBuf = unitInfoJson["inport"]["transparentBuffer"];
    unitInfo[op].inPortOpBuf = unitInfoJson["inport"]["opaqueBuffer"];

    unitInfo[op].outPortTransBuf = unitInfoJson["outport"]["transparentBuffer"];
    unitInfo[op].outPortOpBuf = unitInfoJson["outport"]["opaqueBuffer"];

    if (unitInfoJson.is_discarded())
      return failure();
=======
  // Read the JSON content from the file
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line)) {
    jsonString += line;
  }

  // Parse the JSON
  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonString);
  if (!jsonValue)
    return failure();

  llvm::json::Object *data = jsonValue->getAsObject();
  for (std::string &op : opNames) {
    llvm::json::Object *unitInfoJson = data->getObject(op);
    // parse the bitwidth and its corresponding latency for data
    parseBitWidthPair(*unitInfoJson->getObject("latency"),
                      unitInfo[op].latency);
    parseBitWidthPair(*unitInfoJson->getObject("delay")->getObject("data"),
                      unitInfo[op].dataDelay);
    parseBitWidthPair(
        *unitInfoJson->getObject("inport")->getObject("delay")->getObject(
            "data"),
        unitInfo[op].inPortDataDelay);
    parseBitWidthPair(
        *unitInfoJson->getObject("outport")->getObject("delay")->getObject(
            "data"),
        unitInfo[op].outPortDataDelay);

    // parse the bitwidth and its corresponding latency for valid and ready
    // The valid and ready signal is 1 bit
    unitInfo[op].validDelay = unitInfoJson->getObject("delay")
                                  ->getObject("valid")
                                  ->getNumber("1")
                                  .value();
    unitInfo[op].readyDelay = unitInfoJson->getObject("delay")
                                  ->getObject("ready")
                                  ->getNumber("1")
                                  .value();
    unitInfo[op].inPortValidDelay = unitInfoJson->getObject("inport")
                                        ->getObject("delay")
                                        ->getObject("valid")
                                        ->getNumber("1")
                                        .value();
    unitInfo[op].inPortReadyDelay = unitInfoJson->getObject("inport")
                                        ->getObject("delay")
                                        ->getObject("ready")
                                        ->getNumber("1")
                                        .value();
    unitInfo[op].outPortValidDelay = unitInfoJson->getObject("outport")
                                         ->getObject("delay")
                                         ->getObject("valid")
                                         ->getNumber("1")
                                         .value();
    unitInfo[op].outPortReadyDelay = unitInfoJson->getObject("outport")
                                         ->getObject("delay")
                                         ->getObject("ready")
                                         ->getNumber("1")
                                         .value();

    unitInfo[op].inPortTransBuf = unitInfoJson->getObject("inport")
                                      ->getNumber("transparentBuffer")
                                      .value();
    unitInfo[op].inPortOpBuf =
        unitInfoJson->getObject("inport")->getNumber("opaqueBuffer").value();

    unitInfo[op].outPortTransBuf = unitInfoJson->getObject("outport")
                                       ->getNumber("transparentBuffer")
                                       .value();
    unitInfo[op].outPortOpBuf =
        unitInfoJson->getObject("outport")->getNumber("opaqueBuffer").value();
>>>>>>> a152111d2491cfda88e44e7dad2a61ecc46f296b
  }

  return success();
}
