
#pragma once

#include <string>
#include <iostream>



#include "includes/registry.h"

namespace Kratos
{








class RegistryAuxiliaries final
{
public:

KRATOS_CLASS_POINTER_DEFINITION(RegistryAuxiliaries);


RegistryAuxiliaries() = delete;

RegistryAuxiliaries(RegistryAuxiliaries const& rOther) = delete;

~RegistryAuxiliaries() = delete;




template<typename TPrototypeType>
static void RegisterProcessWithPrototype(
const std::string ModuleName,
const std::string ProcessName,
TPrototypeType rProcessPrototype)
{
const std::string all_path = std::string("Processes.All.") + ProcessName;
RegisterPrototype(all_path, rProcessPrototype);
const std::string module_path = std::string("Processes.") + ModuleName + std::string(".") + ProcessName;
RegisterPrototype(module_path, rProcessPrototype);
}

template<typename TPrototypeType>
static void RegisterOperationWithPrototype(
const std::string ModuleName,
const std::string OperationName,
TPrototypeType rOperationPrototype)
{
const std::string all_path = std::string("Operations.All.") + OperationName;
RegisterPrototype(all_path, rOperationPrototype);
const std::string module_path = std::string("Operations.") + ModuleName + std::string(".") + OperationName;
RegisterPrototype(module_path, rOperationPrototype);
}

template<typename TPrototypeType>
static void RegisterControllerWithPrototype(
const std::string ModuleName,
const std::string ControllerName,
TPrototypeType rControllerPrototype)
{
const std::string all_path = std::string("Controllers.All.") + ControllerName;
RegisterPrototype(all_path, rControllerPrototype);
const std::string module_path = std::string("Controllers.") + ModuleName + std::string(".") + ControllerName;
RegisterPrototype(module_path, rControllerPrototype);
}









private:







template<typename TPrototypeType>
static void RegisterPrototype(
const std::string RegistryEntryName,
TPrototypeType rPrototype)
{
if (!Registry::HasItem(RegistryEntryName)) {
auto& r_item = Registry::AddItem<RegistryItem>(RegistryEntryName);
r_item.AddItem<TPrototypeType>("Prototype", rPrototype);
} else {
KRATOS_ERROR << "'" << RegistryEntryName << "' is already registered." << std::endl;
}
}







}; 







}  
