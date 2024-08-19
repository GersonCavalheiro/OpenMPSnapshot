
#pragma once

#include <stdexcept>
#include <sstream>



#include "includes/registry.h"
#include "includes/registry_item.h"

#define KRATOS_REGISTRY_NAME_(A,B) A##B
#define KRATOS_REGISTRY_NAME(A,B) KRATOS_REGISTRY_NAME_(A,B)


#define KRATOS_REGISTRY_ADD_PROTOTYPE(NAME, X)                                            \
static inline bool KRATOS_REGISTRY_NAME(_is_registered_, __LINE__) = []() -> bool {   \
using TFunctionType = std::function<std::shared_ptr<X>()>;                        \
std::string key_name = NAME + std::string(".") + std::string(#X);                 \
if (!Registry::HasItem(key_name))                                                 \
{                                                                                 \
auto &r_item = Registry::AddItem<RegistryItem>(key_name);                     \
TFunctionType dispatcher = [](){return std::make_shared<X>();};               \
r_item.AddItem<TFunctionType>("Prototype", std::move(dispatcher));            \
}                                                                                 \
return Registry::HasItem(key_name);                                               \
}();
