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
#include "llvm/ADT/TypeSwitch.h"
#include <string>
#include <iostream>
#include <vector>


using namespace llvm;
using namespace mlir;
using namespace circt;

/*
void traverseIRFromOperation(mlir::Operation *op, int k = 0) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks()) 
      for (mlir::Operation &nestedOp : block.getOperations()) {
        llvm::outs() << k << ") " << "Traversing operation " << op << "\n";
        traverseIRFromOperation(&nestedOp, k++);
      }
}
*/

namespace
tools 
{
  StringRef in("_in_\0");
  StringRef out("_out_\n");
  StringRef handshake("handshake");
  StringRef start("start\0");
};

namespace
dcomponents
{
    StringRef 
    func = StringRef("func"), //1.1

    instance = StringRef("instance"), //2.1
    h_return = StringRef("return"), //2.2
    buffer = StringRef("buffer"), //2.3
    fork = StringRef("fork"), //2.4
    lazy_fork = StringRef("lazy_fork"), //2.5
    merge = StringRef("merge"), //2.6
    mux = StringRef("mux"), //2.7
    control_merge = StringRef("control_merge"), //2.8
    br = StringRef("br"), //2.9
    cond_br = StringRef("cond_br"), //2.10
    sink = StringRef("sink"), //2.11
    source = StringRef("source"), //2.12
    never = StringRef("never"), //2.13
    constant = StringRef("constant"), //2.14
    extmemory = StringRef("extmemory"), //2.16
    load = StringRef("load"), //2.17
    store = StringRef("store"), //2.18
    join = StringRef("join"), //2.19
    sync = StringRef("sync"), //2.20
    unpack = StringRef("unpack"), //2.21
    pack = StringRef("pack"), //2.22

    mem_controller = StringRef("mem_controller"), //3.1
    d_load = StringRef("d_load"), //3.2
    d_store = StringRef("d_store"), //3.3
    d_return = StringRef("d_return"), //3.4
    end = StringRef("end"); //3.5

};

/*
mlir::Type myType;
  std::string typeToTxt = llvm::TypeSwitch<Type, std::string>(myType)
                              .Case<NoneType>([&](auto) { return "none"; }) // !esi.channel<something> -> channel<iX>
                              .Case<IndexType>([&](auto) { return "index"; }) // iX / fX (f32)
                              .Default([&](auto) { return "default"; // this should fail the tool 
                              });
*/


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
    size_t flag = 0;
    if (s.find(tools::handshake) >= s.size()) {
      flag = 1;
    }
    auto first = s.find('_', 0);
    ++first;
    auto second = s.find(tools::in, 0);
    if (second >= s.size()) {
      s = tools::start;
    } else {
      s = s.substr(first, second - first);
    }

    if (flag) {
      // arithmetic
      llvm::outs() << "### arith.";
    } else {
      // dataflow
      llvm::outs() << "### handshake.";
    }
    llvm::outs() << s << "\n";
    auto in = s.find(tools::in, 0);
    auto out = s.find(tools::out, 0);
    std::vector <StringRef> inputs{};
    std::vector <StringRef> outputs{};
    
    size_t counter = in;
    counter += 4;
    size_t prev = counter;
    while (s[counter] < ')') {
      if (s[counter] == '_' && counter < out) {
        inputs.push_back(s.substr(prev, counter));
        prev = counter;
      }
      counter++;
    }
    if (in < s.size()) {
      
    }
    if (out < s.size()) {
      //
    }



    if (s == dcomponents::fork || s == dcomponents::lazy_fork) {
      auto n_o = extModOp.getNumOutputs();
      mlir::Type dw = extModOp.getResultTypes()[0];
      
      
      

      
      llvm::outs() << "- number of outputs: " << n_o << "\n";
      llvm::outs() << "- datawidth: " << dw << "\n";
    } else 
    if (s == dcomponents::merge || s == dcomponents::mux || s == dcomponents::control_merge || 
      s == dcomponents::join) {
      //
    } else if (s == dcomponents::br || s == dcomponents::sink || s == dcomponents::source) {
      //
    } else if (s == dcomponents::constant) {
      //
    } else if (s == dcomponents::sync) {
      //
    } else {
      llvm::outs() << "Not implemented yet!\n";
    }


    /// old input:
    /*
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
    */
    // TODO look at the attributes/operands of each external module and
    // identify:
    // - which handshake/arith operation it maps to
    // - the characteristic properties of the operation type (e.g., number of
    // fork outputs) which are gonna be needed to concretize the correct VHDL
    // modules from the templates
  }

  return 0;
}
