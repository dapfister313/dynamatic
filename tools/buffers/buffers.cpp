//===- buffers.cpp - Export Handshake-level IR to DOT --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the export-dot tool, which outputs on stdout the
// Graphviz-formatted representation on an input Handshake-level IR. The tool
// may be configured so that its output is compatible with .dot files expected
// by legacy Dynamatic, assuming the the inpur IR respects some constraints
// imposed in legacy dataflow circuits. This tools enables the creation of a
// bridge between Dynamatic and legacy Dynamatic, which is very useful in
// practice.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string>
    bufferInfoFilepath("buffer-info", cl::Optional,
                       cl::desc("Path to CSV-formatted file containing "
                                "per-channel buffering constraints."),
                       cl::init(""), cl::cat(mainCategory));

static cl::opt<std::string>
    frequenciesFilepath("frequencies", cl::Optional,
                        cl::desc("BB transition frequencies"), cl::init(""),
                        cl::cat(mainCategory));

static cl::opt<std::string> timingDBFilepath(
    "timing-models", cl::Optional,
    cl::desc(
        "Relative path to JSON-formatted file containing timing models for "
        "dataflow components. The tool only tries to read from this file if it "
        "is ran in one of the legacy-compatible modes, where timing "
        "annotations are given to all nodes in the graph. By default, contains "
        "the relative path (from the project's top-level directory) to the "
        "file defining the default timing models in Dynamatic."),
    cl::init("data/components.json"), cl::cat(mainCategory));

static cl::opt<double> targetCP("period", cl::Optional,
                                cl::desc("Target period, in ns"), cl::init(4.0),
                                cl::cat(mainCategory));

static cl::opt<double> timeout("timeout", cl::Optional,
                               cl::desc("Gurobi timeout, in s"), cl::init(180),
                               cl::cat(mainCategory));

namespace {

using NameToOp = std::unordered_map<std::string, Operation *>;

struct ChannelInfo {
  std::string src;
  unsigned srcPort;
  std::string dst;
  unsigned dstPort;
  double delay;
  unsigned minTrans;
  std::optional<unsigned> maxTrans;
  unsigned minOpaque;
  std::optional<unsigned> maxOpaque;
};
} // namespace

/// Determines whether a string is a valid unsigned integer.
static bool isUnsigned(const std::string &str) {
  return std::all_of(str.begin(), str.end(), [](char c) { return isdigit(c); });
}

static bool isDouble(const std::string &str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return isdigit(c) || c == '.'; });
}

static std::string strip(const std::string &str) {
  unsigned startIdx = 0, endIdx = str.size();
  bool leading = true;
  for (auto [idx, c] : llvm::enumerate(str)) {
    if (leading) {
      if (!std::isspace(c)) {
        leading = false;
        startIdx = idx;
      }
    } else {
      if (std::isspace(c)) {
        endIdx = idx;
        break;
      }
    }
  }
  return str.substr(startIdx, endIdx - startIdx);
}

static mlir::Value findChannel(ChannelInfo &info, NameToOp &names) {
  // Identify the channel referenced by the information
  if (names.find(info.src) == names.end()) {
    llvm::errs() << "No operation named " << info.src << " could be found\n";
    return nullptr;
  }
  if (names.find(info.dst) == names.end()) {
    llvm::errs() << "No operation named " << info.dst << " could be found\n";
    return nullptr;
  }
  Operation *srcOp = names[info.src], *dstOp = names[info.dst];
  if (info.srcPort > srcOp->getNumResults()) {
    srcOp->emitError() << "Operation has " << srcOp->getNumResults()
                       << " results but source port is " << info.srcPort << "";
    return nullptr;
  }
  if (info.dstPort > dstOp->getNumOperands()) {
    dstOp->emitError() << "Operation has " << dstOp->getNumOperands()
                       << " operands but destination port is " << info.dstPort
                       << "";
    return nullptr;
  }
  if (srcOp->getResult(info.srcPort) != dstOp->getOperand(info.dstPort)) {
    llvm::errs() << "Port " << info.srcPort << " of " << info.src
                 << " and port " << info.dstPort << " of " << info.dst
                 << " do not appear to match.\n";
    return nullptr;
  }

  return srcOp->getResult(info.srcPort);
}

static LogicalResult annotateFunc(handshake::FuncOp funcOp) {
  // Re-derive operation names in the same way as the DOT does
  std::map<std::string, unsigned> opTypeCntrs;
  NameToOp names;
  for (auto &op : funcOp.getOps()) {
    std::string opFullName = op.getName().getStringRef().str();
    auto startIdx = opFullName.find('.');
    if (startIdx == std::string::npos)
      startIdx = 0;
    unsigned id;
    if (auto memOp = dyn_cast<handshake::MemoryControllerOp>(op))
      // Memories already have unique IDs, so make their name match it
      id = memOp.getId();
    else
      id = opTypeCntrs[op.getName().getStringRef().str()]++;
    names[opFullName.substr(startIdx + 1) + std::to_string(id)] = &op;
  }

  // Open the buffer information file
  std::ifstream inputFile(bufferInfoFilepath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open buffer information\n";
    return failure();
  }

  std::string token;

  auto parseDouble = [&](std::istringstream &iss,
                         double &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (!isDouble(token))
      return failure();
    value = std::stod(token);
    return success();
  };

  auto parseUnsigned = [&](std::istringstream &iss,
                           unsigned &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (!isUnsigned(token))
      return failure();
    value = std::stoi(token);
    return success();
  };

  auto parseOptUnsigned = [&](std::istringstream &iss,
                              std::optional<unsigned> &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (token.empty()) {
      value = std::nullopt;
      return success();
    }
    if (!isUnsigned(token))
      return failure();
    value = std::stoi(token);
    return success();
  };

  auto parseString = [&](std::istringstream &iss,
                         std::string &value) -> ParseResult {
    std::getline(iss, token, ',');
    value = strip(token);
    return success();
  };

  // Skip the header line
  std::string line;
  std::getline(inputFile, line);

  // Parse lines one by one, creating an ArchBB for each
  while (std::getline(inputFile, line)) {
    std::istringstream iss(line);

    // Parse all columns
    ChannelInfo info;
    if (parseString(iss, info.src) || parseUnsigned(iss, info.srcPort) ||
        parseString(iss, info.dst) || parseUnsigned(iss, info.dstPort) ||
        parseDouble(iss, info.delay) || parseUnsigned(iss, info.minTrans) ||
        parseOptUnsigned(iss, info.maxTrans) ||
        parseUnsigned(iss, info.minOpaque) ||
        parseOptUnsigned(iss, info.maxOpaque)) {
      llvm::errs() << "Failed to parse CSV line: " << line << "\n";
      return failure();
    }

    // Find the value that the information indicates
    mlir::Value channel = findChannel(info, names);
    if (!channel)
      return failure();

    // Set buffering properties for the channel
    ChannelBufProps props(info.minTrans, info.maxTrans, info.minOpaque,
                          info.maxOpaque, info.delay);
    if (failed(dynamatic::buffer::replaceBufProps(channel, props)))
      return failure();
  }
  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a DOT graph corresponding to the module for visualization\n"
      "and legacy-compatibility purposes.The pass only supports exporting\n"
      "the graph of a single Handshake function at the moment, and will "
      "fail\n"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these cases
  MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                      arith::ArithDialect, LLVM::LLVMDialect,
                      handshake::HandshakeDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return 1;
  mlir::ModuleOp modOp = *module;
  auto funcs = modOp.getOps<handshake::FuncOp>();
  if (std::distance(funcs.begin(), funcs.end()) != 1) {
    modOp->emitError() << "We only support one Handshake function per module";
    return 1;
  }
  handshake::FuncOp funcOp = *funcs.begin();
  if (failed(annotateFunc(funcOp))) {
    funcOp->emitError() << "Failed to annotate function";
    return 1;
  }

  // Run the buffer placement pass
  PassManager pm(&context);
  pm.addPass(dynamatic::buffer::createHandshakePlaceBuffersPass(
      frequenciesFilepath, timingDBFilepath, false, targetCP, timeout, true));
  if (failed(pm.run(modOp))) {
    llvm::errs() << "Failed to run buffer placement\n.";
    return 1;
  }

  // Parse timing models for DOT printer
  TimingDatabase timingDB(&context);
  if (failed(TimingDatabase::readFromJSON(timingDBFilepath, timingDB))) {
    llvm::errs() << "Failed to read timing database at \"" << timingDBFilepath
                 << "\"\n";
    return 1;
  }
  DOTPrinter printer(DOTPrinter::Mode::LEGACY, DOTPrinter::EdgeStyle::SPLINE,
                     &timingDB);
  return failed(printer.printDOT(*module));
}
