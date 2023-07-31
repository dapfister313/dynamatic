//===- export-vhdl.cpp - Export VHDL from netlist-level IR ------*- C++ -*-===//
//
// Experimental tool that exports VHDL from a netlist-level IR expressed in a
// combination of the HW and ESI dialects.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/type_traits.h"

#include <cstdio>
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#define LIBRARY_PATH "experimental/tools/export-vhdl/library.json"

using namespace llvm;
using namespace mlir;
using namespace circt;

// TIPS
// 1) It's better to use MLIR analogues instead of cpp ones, because they safe
// memory:
//  StringRef <=> string (& organised but may fall in case of memory, we assume
//  that we don't fail) DenseMap <=> map SmallVector <=> vector / list. [however
//  std::vector smtms may be useful] But if u have no option -> use cpp data
//  structures
// 2) Concretization mathod: basically generics.
//  Generator: start thinking, may be difficult as we work with binaries
//  Text: for future, don't care at the moment
// 3) path to resource: separate vhds to separate files and give the string int
// ../../../path_format so on 4) 1st step of GENERAL PLAN is already
// implemented: take .hpp from the link Lucas sent you,
//  put it into include/dynamatic/support, read how to use it and celebrate
// 5) About inheritance: don't use it, take the components as VHDLComponents,
// but specify the structure later. 6) For components with equal structure it's
// no use in specifying them all in .json. Just link lazy_fork to fork and be
// happy

// GENERAL PLAN v1.0
// 1) parse the JSON and get the map of components description -> [special
// 2) extract ext_names from .mlir as string / stringrefs -> [very EASY, already
// done it on one of the first stages] 3) merge the map with data from .mlir
// using getMod. At this stage we obtain VHDLModules. 4) construct a real vhdl
// module from each VHDLModule [seems fairly hard]

// ESPECIAL PLAN v1.0
// 1) writing 3-5 components, Better to include mem_controller as the most
// difficult one. 2) read about reading json from cpp 3) Try to implement it 4)
// Read about MLIR data structures in detail (DenseMap's almoust unexplored) 5)
// Start writing the core of functions

struct VHDLParameter;
struct VHDLComponent;

struct VHDLParameter {
  VHDLParameter(std::string temp_name = "", std::string temp_type = "")
      : name{temp_name}, type{temp_type} {};
  std::string getName() const { return name; }
  std::string getType() const { return type; }

private:
  std::string name;
  std::string type;
};

struct VHDLComponent {

  VHDLComponent(std::string temp_path = {}, std::string temp_concr_method = {},
                llvm::SmallVector<VHDLParameter> temp_parameters = {})
      : path(temp_path), concretization_method(temp_concr_method),
        parameters(temp_parameters) {}
  std::string getPath() const { return path; }
  std::string getConcretization_method() const {
    return concretization_method;
  };
  llvm::SmallVector<VHDLParameter> getParameters() const { return parameters; }

private:
  std::string path;
  std::string concretization_method;
  llvm::SmallVector<VHDLParameter> parameters;
};

typedef llvm::StringMap<VHDLComponent> VHDLComponentLibrary;

struct VHDLModule {
  std::list<std::string> input_ports;
};

VHDLModule getMod(std::string extName, VHDLComponentLibrary jsonLib){

};

VHDLComponentLibrary parseJSON() {
  // Load JSON library
  std::ifstream lib;
  lib.open(LIBRARY_PATH);
  // VHDLComponentLibrary m{};
  VHDLComponentLibrary m{};
  if (!lib.is_open()) {
    errs() << "Filepath is uncorrect\n";
    return m;
  }
  // Read as file
  std::stringstream buffer;
  buffer << lib.rdbuf();
  std::string jsonStr = buffer.str();

  // Parse the library
  auto jsonLib = llvm::json::parse(StringRef(jsonStr));

  if (!jsonLib) {
    errs() << "Library JSON could not be parsed"
           << "\n";
    return m;
  }

  if (!jsonLib->getAsObject()) {
    errs() << "Library JSON is not a valid JSON"
           << "\n";
    return m;
  }
  for (auto item : *jsonLib->getAsObject()) {
    auto key_name = item.first.str();
    auto path = item.second.getAsObject()
                    ->find("path")
                    ->second.getAsString()
                    .value()
                    .str();
    auto concretization_method = item.second.getAsObject()
                                     ->find("concretization_method")
                                     ->second.getAsString()
                                     .value()
                                     .str();
    auto parameters =
        item.getSecond().getAsObject()->get("parameters")->getAsArray();
    llvm::SmallVector<VHDLParameter> components{};
    for (auto i = parameters->begin(); i != parameters->end(); ++i) {
      auto obj = i->getAsObject();
      auto name = obj->get("name")->getAsString().value().str();
      auto type = obj->get("type")->getAsString().value().str();
      components.push_back(VHDLParameter(name, type));
    }
    m.insert(std::pair(key_name,
                       VHDLComponent(path, concretization_method, components)));
  }

  return m;
}

void testLib(VHDLComponentLibrary &m) {
  for (auto &[keyl, val] : m) {
    llvm::outs() << "---\n"
                 << keyl << " "
                 << "\npath: " << val.getPath()
                 << "\nconcr_method: " << val.getConcretization_method()
                 << "\nparameters:\n";
    for (auto &i : val.getParameters()) {
      llvm::outs() << "[" << i.getName() << "," << i.getType() << "]\n";
    }
  }
}

// VHDLModule getMod(std::string extName, VHDLComponentLibrary jsonLib){};

// if needed
/*
struct VHDL_MODULE_Description {
  std::string pathToResource;
  std::string concretizationMethod;
};
*/
///
/// std::string modName;
/// std::string modParameters;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

int main(int argc, char **argv) {
  auto m = parseJSON();
  auto i = m.find("handshake.fork");

  testLib(m);

  //////////////////////////////////////////////[][][][][][][][][]///////////////////////////////////////////ss
  // Initialize LLVM and parse command line arguments
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "VHDL exporter\n\n"
      "This tool prints on stdout the VHDL design corresponding to the input",
      "netlist-level MLIR representation of a dataflow circuit. \n");

  // Read the input IR in memory
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er)
  // level
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
    // - the characteristic properties of the operation type (e.g., number
    // of
    //  fork outputs) which are gonna be needed to concretize the correct
    // VHDL
    //  modules from the templates
  }

  return 0;
}
