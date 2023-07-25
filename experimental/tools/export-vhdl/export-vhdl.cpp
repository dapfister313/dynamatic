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

#include <cstdio>
#include <list>
#include <string>
#include <map>
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace circt;

// TIPS
// 1) It's better to use MLIR analogues instead of cpp ones, because they safe memory:
//  StringRef <=> string (& organised but may fall in case of memory, we assume that we don't fail)
//  DenseMap <=> map
//  SmallVector <=> vector / list. [however std::vector smtms may be useful]
//  But if u have no option -> use cpp data structures
// 2) Concretization mathod: basically generics. 
//  Generator: start thinking, may be difficult as we work with binaries
//  Text: for future, don't care at the moment
// 3) path to resource: separate vhds to separate files and give the string int ../../../path_format so on
// 4) 1st step of GENERAL PLAN is already implemented: take .hpp from the link Lucas sent you, 
//  put it into include/dynamatic/support, read how to use it and celebrate
// 5) About inheritance: don't use it, take the components as VHDLComponents, but specify the structure later.
// 6) For components with equal structure it's no use in specifying them all in .json. Just link lazy_fork to fork and be happy


// GENERAL PLAN v1.0
// 1) parse the JSON and get the map of components description -> [special 
// 2) extract ext_names from .mlir as string / stringrefs -> [very EASY, already done it on one of the first stages]
// 3) merge the map with data from .mlir using getMod. At this stage we obtain VHDLModules. 
// 4) construct a real vhdl module from each VHDLModule [seems fairly hard]

// ESPECIAL PLAN v1.0
// 1) writing 3-5 components, Better to include mem_controller as the most difficult one.
// 2) read about reading json from cpp
// 3) Try to implement it
// 4) Read about MLIR data structures in detail (DenseMap's almoust unexplored)
// 5) Start writing the core of functions

struct VHDLComponent
{
  // here we somehow describe each component: fork, merge etc
  // [switch / sequence of if / maybe inheritance to concretise ?]
  // because the structure of fork differs from the structure of mem_controller, for instance.
};
struct VHDLModule
{
  std::list<std::string> input_ports;

};

typedef std::map <std::string, VHDLComponent> VHDLComponentLibrary;

// [not needed]
VHDLComponentLibrary 
parseJSON (std::ifstream &jsonLib) { //in case we change path to json
  // parsing json to get a more convenient cpp representation
}
VHDLModule
getMod (std::string extName, VHDLComponentLibrary jsonLib) {

}


// if needed
struct VHDL_MODULE_Description
{
  std::string pathToResource;
  std::string concretizationMethod;

};

///
/// std::string modName;
/// std::string modParameters;


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
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {
    llvm::outs() << extModOp.getVerilogModuleName() << "\n";

    // TODO look at the attributes/operands of each external module and
    // identify:
    // - which handshake/arith operation it maps to
    // - the characteristic properties of the operation type (e.g., number of
    // fork outputs) which are gonna be needed to concretize the correct VHDL
    // modules from the templates
  }

  return 0;
}
