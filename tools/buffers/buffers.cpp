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
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakeSetBufferingProperties.h"
#include "dynamatic/Transforms/HandshakeCanonicalize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>
#include <fstream>
#include <iterator>
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
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
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

  // Extract the single function from the module
  auto funcs = modOp.getOps<handshake::FuncOp>();
  if (std::distance(funcs.begin(), funcs.end()) != 1) {
    modOp->emitError() << "We only support one Handshake function per module";
    return 1;
  }
  // handshake::FuncOp funcOp = *funcs.begin();
  // if (failed(annotateFunc(funcOp))) {
  //   funcOp->emitError() << "Failed to annotate function";
  //   return 1;
  // }

  // Run the buffer placement pass
  PassManager pm(&context);
  pm.addPass(dynamatic::buffer::createHandshakeSetBufferingProperties());
  pm.addPass(dynamatic::buffer::createHandshakePlaceBuffersPass(
      frequenciesFilepath, timingDBFilepath, false, targetCP, timeout, true,
      bufferInfoFilepath));
  pm.addPass(dynamatic::createHandshakeCanonicalize());
  if (failed(pm.run(modOp))) {
    llvm::errs() << "Failed to run buffer placement.\n";
    return 1;
  }

  // Parse timing models for DOT printer
  TimingDatabase timingDB(&context);
  if (failed(TimingDatabase::readFromJSON(timingDBFilepath, timingDB))) {
    llvm::errs() << "Failed to read timing database at '" << timingDBFilepath
                 << "'.\n";
    return 1;
  }
  DOTPrinter printer(DOTPrinter::Mode::LEGACY, DOTPrinter::EdgeStyle::SPLINE,
                     &timingDB);
  return failed(printer.printDOT(*module));
}
