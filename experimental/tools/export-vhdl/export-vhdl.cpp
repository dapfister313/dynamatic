//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Experimental tool that exports VHDL from a netlist-level IR expressed in a
// combination of the HW and ESI dialects.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"


using namespace llvm;
using namespace mlir;
using namespace circt;

void traverseIRFromOperation(mlir::Operation *op, int k = 0) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks()) 
      for (mlir::Operation &nestedOp : block.getOperations()) {
        llvm::outs() << k << ") " << "Traversing operation " << op << "\n";
        traverseIRFromOperation(&nestedOp, k++);
      }
}

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

int main(int argc, char **argv) {
  // Initialize LLVM and parse command line arguments
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "VHDL exporter\n\n"
      "This tool prints on stdout the VHDL design corresponding to the input "
      "netlist-level MLIR representation of a dataflow circuit.\n");

  // Read the input IR in memory
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
  MLIRContext context;
  context.loadDialect<circt::hw::HWDialect, circt::esi::ESIDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module in memory
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return 1;            

  // Just print the @name of all external hardware modules in the input

  int k = 0;
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {   
    llvm::outs() << "-------------------\n" << ++k << ")" << "\n";     
    auto s = extModOp.getVerilogModuleName();
    StringRef p("_in_\0");
    size_t flag = 0;
    if (s.find(StringRef("handshake")) >= s.size()) {
      flag = 1;
    }
    auto first = s.find('_', 0);
    ++first;
    auto second = s.find(p, 0);
    if (second >= s.size()) {
      s = StringRef("start\0");
    } else {
      s = s.substr(first, second - first);
    }
    llvm::outs() << "The component's characteristic: ";
    if (flag) {
      llvm::outs() << "arithmetic,\n";
    } else {
      llvm::outs() << "dataflow,\n";
    }
    llvm::outs() << "The component's type: " << s << ",\n";
    llvm::outs() << "Number of inputs: " << extModOp.getNumInputs() << ", these are: " << extModOp.getArgNames() << " "
      << "with types: " << extModOp.getArgumentTypes() << " respectively,\n";
    llvm::outs() << "Number of outputs: " << extModOp.getNumOutputs() << ", these are: " << extModOp.getResultNames() << " " 
      << "with types: " << extModOp.getResultTypes() << " respectively.\n";
    llvm::outs() << "-------------------" << "\n";

    // TODO look at the attributes/operands of each external module and
    // identify:
    // - which handshake/arith operation it maps to
    // - the characteristic properties of the operation type (e.g., number of
    // fork outputs) which are gonna be needed to concretize the correct VHDL
    // modules from the templates
  }

  return 0;
}
