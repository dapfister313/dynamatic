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

#include <any>
#include <cstdio>
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <cstdlib>

#define LIBRARY_PATH "experimental/tools/export-vhdl/library.json"

using namespace llvm;
using namespace mlir;
using namespace circt;

struct VHDLParameter;
struct VHDLModuleDescription;
struct VHDLModule;
typedef llvm::StringMap<VHDLModuleDescription> VHDLComponentLibrary;
typedef llvm::StringMap<size_t> StoreComponentNumbers;
typedef llvm::SmallVector<std::string> string_arr;
typedef llvm::SmallVector<int> integer_arr;

//===----------------------------------------------------------------------===//
// GENERAL
//===----------------------------------------------------------------------===//

// Just an inner description in VHDLComponentLibrary lib
struct VHDLParameter {
  VHDLParameter(std::string tempName = "", std::string tempType = "",
                std::string tempSize = "")
      : name{tempName}, type{tempType}, size{tempSize} {};
  std::string getName() const { return name; }
  std::string getType() const { return type; }
  std::string getSize() const {
    if (size.empty())
      return "1";
    else
      return size;
  }

private:
  std::string name;
  std::string type;
  std::string size;
};

// Description of the component in the lib
struct VHDLModuleDescription {

  VHDLModuleDescription(std::string tempPath = {},
                        std::string tempConcrMethod = {},
                        llvm::SmallVector<std::string> tempGenerators = {},
                        llvm::SmallVector<std::string> tempGenerics = {},
                        llvm::SmallVector<std::string> tempExtras = {},
                        llvm::SmallVector<VHDLParameter> tempInputPorts = {},
                        llvm::SmallVector<VHDLParameter> tempOutputPorts = {})
      : path(tempPath), concretizationMethod(tempConcrMethod),
        generators(tempGenerators), generics(tempGenerics), extras(tempExtras),
        inputPorts(tempInputPorts), outputPorts(tempOutputPorts) {}
  std::string getPath() const { return path; }
  std::string getConcretizationMethod() const { return concretizationMethod; };
  const llvm::SmallVector<std::string> &getGenerators() const {
    return generators;
  }
  const llvm::SmallVector<std::string> &getGenerics() const { return generics; }
  const llvm::SmallVector<std::string> &getExtras() const { return extras; }
  const llvm::SmallVector<VHDLParameter> &getInputPorts() const {
    return inputPorts;
  }
  const llvm::SmallVector<VHDLParameter> &getOutputPorts() const {
    return outputPorts;
  }
  VHDLModule concretize(std::string modName, std::string modParameters) const;

private:
  std::string path;
  std::string concretizationMethod;
  // parameters used in generators. Values're taken from discriminating
  // parameters
  llvm::SmallVector<std::string> generators;
  // parameters for instantiation
  llvm::SmallVector<std::string> generics;
  // extra parameters that we may need in future
  llvm::SmallVector<std::string> extras;
  // all input ports
  llvm::SmallVector<VHDLParameter> inputPorts;
  // all output ports
  llvm::SmallVector<VHDLParameter> outputPorts;
};

// VHDL module, that is VHDLModuleDescription + actual parameters data
struct VHDLModule {
  VHDLModule(std::string tempModName, std::string tempMtext,
             llvm::SmallVector<std::string> tempModParameters,
             const VHDLModuleDescription &tempModDesc)
      : modName(tempModName), modText(tempMtext),
        modParameters(tempModParameters), modDesc(tempModDesc) {}
  std::string getModName() const { return modName; }
  const std::string &getModText() const { return modText; }
  const llvm::SmallVector<std::string> &getModParameters() const {
    return modParameters;
  }
  const VHDLModuleDescription &getModDesc() const { return modDesc; }

private:
  // component's name
  std::string modName;
  // component's definition & architecture
  std::string modText;
  // discriminating parameters
  llvm::SmallVector<std::string> modParameters;
  // reference to the corresponding template in VHDLComponentLibrary
  const VHDLModuleDescription &modDesc;
};

// split the string with discriminating parameters into string vector for
// convenience
llvm::SmallVector<std::string>
parseDiscriminatingParameters(std::string &modParameters) {
  llvm::SmallVector<std::string> s{};
  std::stringstream str(modParameters);
  std::string temp;
  while (str.good()) {
    std::getline(str, temp, '_');
    s.push_back(temp);
  }
  return s;
}

// Get a module corresponding the given component and data
VHDLModule VHDLModuleDescription::concretize(std::string modName,
                                             std::string modParameters) const {
  llvm::SmallVector<std::string> modParametersVec =
      parseDiscriminatingParameters(modParameters);
  std::string modText{};
  // open a file with component concretization data
  std::ifstream file;
  // read as file
  std::stringstream buffer;
  if (concretizationMethod == "GENERATOR") {
    // in case of generator we're looking for binary
    std::string commandLineArguments;
    auto i = modParametersVec.begin();
    size_t k = generators.size();
    // collecting discriminating params for command line
    while (k > 0) {
      commandLineArguments += " " + (*i);
      ++i;
      --k;
    }
    // create a temporary text.txt
    std::string resultPath = path + commandLineArguments + " > test.txt";
    std::system(resultPath.c_str());
    file.open("test.txt");
    // ... and delete it
    std::system("rm test.txt");
  } else if (concretizationMethod == "GENERIC") {
    // in case of generic we're looking for ordinary file
    file.open(path);
  } else
    // error
    llvm::errs() << "Wrong concredization method";
  buffer << file.rdbuf();
  modText = buffer.str();
  return VHDLModule(modName, modText, modParametersVec, *this);
}

// Get a cpp representation for given .json file
VHDLComponentLibrary parseJSON() {
  // Load JSON library
  std::ifstream lib;
  lib.open(LIBRARY_PATH);

  VHDLComponentLibrary m{};
  if (!lib.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    return m;
  }
  // Read as file
  std::stringstream buffer;
  buffer << lib.rdbuf();
  std::string jsonStr = buffer.str();

  // Parse the library
  auto jsonLib = llvm::json::parse(StringRef(jsonStr));

  if (!jsonLib) {
    llvm::errs() << "Library JSON could not be parsed"
                 << "\n";
    return m;
  }

  if (!jsonLib->getAsObject()) {
    llvm::errs() << "Library JSON is not a valid JSON"
                 << "\n";
    return m;
  }
  // parse elements in json
  for (auto item : *jsonLib->getAsObject()) {
    // moduleName is either "arith" or "handshake"
    std::string moduleName = item.getFirst().str();
    auto moduleArray = item.getSecond().getAsArray();
    for (auto c = moduleArray->begin(); c != moduleArray->end(); ++c) {
      // c is iterator, which points on a specific component's scheme inside
      // arith / handshake class
      auto obj = c->getAsObject();
      auto jsonComponents = obj->get("components")->getAsArray();
      auto jsonConcretizationMethod =
          obj->get("concretization_method")->getAsString();
      auto jsonGenerators = obj->get("generators")->getAsArray();
      auto jsonGenerics = obj->get("generics")->getAsArray();
      auto jsonExtras = obj->get("extras")->getAsArray();
      auto jsonPorts = obj->get("ports")->getAsObject();
      // creating corresponding VHDLModuleDescription variables
      std::string concretizationMethod = jsonConcretizationMethod.value().str();

      llvm::SmallVector<std::string> generators{};
      for (auto i = jsonGenerators->begin(); i != jsonGenerators->end(); ++i)
        generators.push_back(i->getAsString().value().str());

      llvm::SmallVector<std::string> generics{};
      for (auto i = jsonGenerics->begin(); i != jsonGenerics->end(); ++i)
        generics.push_back(i->getAsString().value().str());

      llvm::SmallVector<std::string> extras{};
      for (auto i = jsonExtras->begin(); i != jsonExtras->end(); ++i)
        extras.push_back(i->getAsString().value().str());

      llvm::SmallVector<VHDLParameter> inputPorts{};
      auto jsonInputPorts = jsonPorts->get("in")->getAsArray();
      for (auto i = jsonInputPorts->begin(); i != jsonInputPorts->end(); ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto type = ob->get("type")->getAsString().value().str();
        std::string size{};
        if (ob->find("size") != ob->end())
          size = ob->get("size")->getAsString().value().str();
        inputPorts.push_back(VHDLParameter(name, type, size));
      }

      llvm::SmallVector<VHDLParameter> outputPorts{};
      auto jsonOutputPorts = jsonPorts->get("out")->getAsArray();
      for (auto i = jsonOutputPorts->begin(); i != jsonOutputPorts->end();
           ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto type = ob->get("type")->getAsString().value().str();
        std::string size{};
        if (ob->find("size") != ob->end())
          size = ob->get("size")->getAsString().value().str();
        outputPorts.push_back(VHDLParameter(name, type, size));
      }

      for (auto i = jsonComponents->begin(); i != jsonComponents->end(); ++i) {
        auto ob = i->getAsObject();
        auto name = ob->get("name")->getAsString().value().str();
        auto path = ob->get("path")->getAsString().value().str();
        std::string key_name = moduleName + "_" + name;
        // inserting our component into library
        m.insert(std::pair(key_name,
                           VHDLModuleDescription(path, concretizationMethod,
                                                 generators, generics, extras,
                                                 inputPorts, outputPorts)));
      }
    }
  }
  lib.close();
  return m;
}

// Check if library is correct
void testLib(VHDLComponentLibrary &m) {
  int num = 1;
  for (auto &[keyl, val] : m) {
    llvm::outs() << "# "
                    "===-------------------------------------------------------"
                    "---------------=== #\n";
    llvm::outs() << "(" << num << ") " << keyl << "\n";
    ++num;
    llvm::outs() << "# "
                    "===-------------------------------------------------------"
                    "---------------=== #\n";
    std::ifstream file;
    file.open(val.getPath());

    if (!file.is_open()) {
      errs() << "Filepath is uncorrect\n";
      file.close();
      return;
    }
    file.close();
    llvm::outs() << "Component:" << keyl << "\n"
                 << "path: " << val.getPath()
                 << "\nconcr method: " << val.getConcretizationMethod()
                 << "\ngenerator params:\n";
    for (auto &i : val.getGenerators()) {
      llvm::outs() << i << " ";
    }
    llvm::outs() << "\ngenerics params:\n ";
    for (auto &i : val.getGenerics()) {
      llvm::outs() << i << " ";
    }
    llvm::outs() << "\nextra params:\n";
    for (auto &i : val.getExtras()) {
      llvm::outs() << i << " ";
    }
    llvm::outs() << "\ninput ports:\n";
    for (auto &i : val.getInputPorts()) {
      llvm::outs() << "[" << i.getName() << ", " << i.getType() << "]\n";
    }
    llvm::outs() << "output ports:\n";
    for (auto &i : val.getOutputPorts()) {
      llvm::outs() << "[" << i.getName() << ", " << i.getType() << ", "
                   << i.getSize() << "]\n";
    }
  }
}
//===----------------------------------------------------------------------===//
// CONCRETIZATION
//===----------------------------------------------------------------------===//

// extName consists of modName and modParameters (e.g. handshake_fork_3_32) and
// this function splits these parameters into 2 strings (in our example
// handshake_fork and 3_32 respectively)
std::pair<std::string, std::string> splitExtName(StringRef extName) {
  size_t first_ = extName.find('_', 0);
  size_t second_ = extName.find('_', first_ + 1);
  std::string componentPart =
      extName.substr(first_ + 1, second_ - first_ - 1).str();
  if (componentPart == "lazy" || componentPart == "control" ||
      componentPart == "cond" || componentPart == "d" ||
      componentPart == "mem" || componentPart == "start" ||
      componentPart == "end") {
    second_ = extName.find('_', second_ + 1);
  }
  std::string modName = extName.substr(0, second_).str();
  std::string modParameters = extName.substr(second_ + 1).str();
  return std::pair(modName, modParameters);
}

// get .vhd module description
VHDLModule getMod(StringRef extName, VHDLComponentLibrary &jsonLib) {
  auto p = splitExtName(extName);
  std::string modName = p.first;
  std::string modParameters = p.second;

  // find external module in VHDLComponentLibrary
  llvm::StringMapIterator<VHDLModuleDescription> comp = jsonLib.find(modName);

  if (comp == jsonLib.end()) {
    llvm::errs() << "Unable to find the element in the library\n";
    return VHDLModule({}, {}, {}, {});
  }
  const VHDLModuleDescription &desc = (*comp).second;
  auto mod = desc.concretize(modName, modParameters);

  return mod;
};

// Test how modules are printed on concretization phase
void testModulesConcretization(mlir::OwningOpRef<mlir::ModuleOp> &module,
                               VHDLComponentLibrary &m) {
  StoreComponentNumbers comp{};
  int num = 1;
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {
    auto extName = extModOp.getModuleName();
    llvm::outs() << "# "
                    "===-------------------------------------------------------"
                    "---------------=== #\n";
    auto i = getMod(extName, m);

    if (i.getModText().empty()) {
      llvm::outs() << "(" << num << ") " << extName
                   << " still doesn't exist in the lib\n";
      llvm::outs() << "# "
                      "===-----------------------------------------------------"
                      "-----------------=== #\n";
    } else {
      llvm::outs() << "(" << num << ") " << i.getModName() << "\n";
      llvm::outs() << "# "
                      "===-----------------------------------------------------"
                      "-----------------=== #\n";
      llvm::outs() << i.getModText() << "\n";
    }
    ++num;
  }
}

//===----------------------------------------------------------------------===//
// INSTANTIATION
//===----------------------------------------------------------------------===//

// For component's name on instantiation: get a number of next similar component
// (e.g fork) or add a new component to SoreComponentNumbers library (numeration
// starts with 0)
size_t getModNumber(std::string modName, StoreComponentNumbers &n) {
  auto it = n.find(modName);
  if (it == n.end()) {
    n.insert(std::pair(modName, 0));
    return 0;
  } else {
    ++it->second;
    return it->second;
  }
}

// Get full description = instantiation = for a given module
std::string getInstanceDeclaration(VHDLModule &mod, size_t modNumber) {
  // header
  std::string instance{};
  std::string shortName =
      mod.getModName().substr(mod.getModName().find('_') + 1);
  std::string numberedName = shortName + "_n" + std::to_string(modNumber);
  instance +=
      numberedName + " : entity work." + shortName + "(arch) generic map (";
  auto startInd = mod.getModDesc().getGenerators().size();
  auto generics = mod.getModDesc().getGenerics();
  auto modParameters = mod.getModParameters();
  auto n = startInd + generics.size();
  for (auto i = startInd; i < n; ++i) {
    instance += modParameters[i];
    if (i == n - 1)
      instance += ")";
    else
      instance += ", ";
  }
  instance += "\nport map(\n";
  // body
  auto inputPorts = mod.getModDesc().getInputPorts();
  auto outputPorts = mod.getModDesc().getOutputPorts();
  // clock & reset
  instance += "clk => " + numberedName + "_clk,\n" + "rst => " + numberedName +
              "_rst,\n";
  // inputs
  for (auto i : inputPorts) {
    std::string paramName = i.getName();
    std::string paramType = i.getType();
    std::string paramSize = i.getSize();
    size_t inp = 0;
    // find the size of array if the component's multidimensional
    if (paramSize != "1") {
      size_t k = 0;
      for (auto &j : generics) {
        if (j == paramSize) {
          inp = std::stoi(modParameters[startInd + k]);
          break;
        }
        ++k;
      }
    }
    if (paramType == "dataflow") {
      // have param, param_valid, param_ready
      if (inp > 0) {
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_" + std::to_string(j) +
                      ",\n";
        }
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "_valid(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_valid_" +
                      std::to_string(j) + ",\n";
        }
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "_ready(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_ready_" +
                      std::to_string(j) + ",\n";
        }
      } else {
        instance += paramName + " => " + numberedName + "_" + paramName + "\n";
        instance += paramName + "_valid => " + numberedName + "_" + paramName +
                    "_valid,\n";
        instance += paramName + "_ready => " + numberedName + "_" + paramName +
                    "_ready,\n";
      }
    } else if (paramType == "control") {
      // have only param_ready & param_valid
      if (inp > 0) {
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "_valid(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_valid_" +
                      std::to_string(j) + ",\n";
        }
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "_ready(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_ready_" +
                      std::to_string(j) + ",\n";
        }
      } else {
        instance += paramName + "_valid => " + numberedName + "_" + paramName +
                    "_valid,\n";
        instance += paramName + "_ready => " + numberedName + "_" + paramName +
                    "_ready,\n";
      }
    } else if (paramType == "signal") {
      // only param, without valid & ready signals
      if (inp > 0) {
        for (auto j = (size_t)0; j < inp; ++j) {
          instance += paramName + "(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_" + std::to_string(j) +
                      ",\n";
        }
      } else {
        instance += paramName + " => " + numberedName + "_" + paramName + ",\n";
      }
    } else
      // error
      llvm::errs() << "Wrong input port's type\n";
  }
  // outputs
  for (auto i : outputPorts) {
    std::string paramName = i.getName();
    std::string paramType = i.getType();
    std::string paramSize = i.getSize();
    size_t outp = 0;
    // find the size of array if the component's multidimensional
    if (paramSize != "1") {
      size_t k = 0;
      for (auto &j : generics) {
        if (j == paramSize) {
          outp = std::stoi(modParameters[startInd + k]);
          break;
        }
        ++k;
      }
    }
    if (paramType == "dataflow") {
      // have param, param_valid, param_ready
      if (outp > 0) {
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_" + std::to_string(j) +
                      ",\n";
        }
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "_valid(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_valid_" +
                      std::to_string(j) + ",\n";
        }
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "_ready(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_ready_" +
                      std::to_string(j) + ",\n";
        }
      } else {
        instance += paramName + " => " + numberedName + "_" + paramName + "\n";
        instance += paramName + "_valid => " + numberedName + "_" + paramName +
                    "_valid,\n";
        instance += paramName + "_ready => " + numberedName + "_" + paramName +
                    "_ready,\n";
      }
    } else if (paramType == "control") {
      // have only param_ready & param_valid
      if (outp > 0) {
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "_valid(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_valid_" +
                      std::to_string(j) + ",\n";
        }
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "_ready(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_ready_" +
                      std::to_string(j) + ",\n";
        }
      } else {
        instance += paramName + "_valid => " + numberedName + "_" + paramName +
                    "_valid,\n";
        instance += paramName + "_ready => " + numberedName + "_" + paramName +
                    "_ready,\n";
      }
    } else if (paramType == "signal") {
      // only param, without valid & ready signals
      if (outp > 0) {
        for (auto j = (size_t)0; j < outp; ++j) {
          instance += paramName + "(" + std::to_string(j) + ") => " +
                      numberedName + "_" + paramName + "_" + std::to_string(j) +
                      ",\n";
        }
      } else {
        instance += paramName + " => " + numberedName + "_" + paramName + ",\n";
      }
    } else
      // error
      llvm::errs() << "Wrong output port's type\n";
  }
  if ((*(instance.end() - 1)) == '\n') {
    (*(instance.end() - 2)) = ')';
    (*(instance.end() - 1)) = ';';
  }
  instance += "\n";
  return instance;
}

// Test modules instantiation description
void testModulesInstantiation(mlir::OwningOpRef<mlir::ModuleOp> &module,
                              VHDLComponentLibrary &m) {
  StoreComponentNumbers comp{};
  for (auto extModOp : module->getOps<hw::HWModuleExternOp>()) {
    auto extName = extModOp.getModuleName();
    llvm::outs() << "# "
                    "===-------------------------------------------------------"
                    "---------------=== #\n";
    auto i = getMod(extName, m);

    if (i.getModText().empty()) {
      llvm::outs() << i.getModName() << " still doesn't exist in the lib\n";
      llvm::outs()
          << "# "
             "===-------------------------------------------------------"
             "---------------=== #\n";
      continue;
    } else {
      llvm::outs() << i.getModName() << "\n";
      llvm::outs()
          << "# "
             "===-------------------------------------------------------"
             "---------------=== #\n";
      auto instance =
          getInstanceDeclaration(i, getModNumber(i.getModName(), comp));
      llvm::outs() << instance << "\n";
    }
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
      "This tool prints on stdout the VHDL design corresponding to the input"
      "netlist-level MLIR representation of a dataflow circuit.\n");

  // Read the input IR in memory
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er)
  // level dialects or parsers. Allow unregistered dialects to not fail in
  // these cases
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

  //////////////////////////////////////
  auto m = parseJSON();
  // testLib(m);
  // testModulesConcretization(module, m);

  testModulesInstantiation(module, m);
  return 0;
}