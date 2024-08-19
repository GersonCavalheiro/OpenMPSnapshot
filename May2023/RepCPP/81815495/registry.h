
#pragma once

#include <string>
#include <iostream>


#include "includes/registry_item.h"
#include "utilities/parallel_utilities.h"
#include "utilities/string_utilities.h"

namespace Kratos
{




class KRATOS_API(KRATOS_CORE) Registry final
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Registry);


Registry(){}

~Registry(){}




template<typename TItemType, class... TArgumentsList >
static RegistryItem& AddItem(
std::string const& rItemFullName,
TArgumentsList&&... Arguments)
{

const std::lock_guard<LockObject> scope_lock(ParallelUtilities::GetGlobalLock());

auto item_path = StringUtilities::SplitStringByDelimiter(rItemFullName, '.');
KRATOS_ERROR_IF(item_path.empty()) << "The item full name is empty" << std::endl;

RegistryItem* p_current_item = &GetRootRegistryItem();

for(std::size_t i = 0 ; i < item_path.size() - 1 ; i++){
auto& r_item_name = item_path[i];
if(p_current_item->HasItem(r_item_name)){
p_current_item = &p_current_item->GetItem(r_item_name);
}
else{
p_current_item = &p_current_item->AddItem<RegistryItem>(r_item_name);
}
}

auto& r_item_name = item_path.back();
if(p_current_item->HasItem(r_item_name)){
KRATOS_ERROR << "The item \"" << rItemFullName << "\" is already registered." << std::endl;
}
else{
p_current_item = &p_current_item->AddItem<TItemType>(r_item_name, std::forward<TArgumentsList>(Arguments)...);
}

return *p_current_item;
}


static auto begin()
{
return mspRootRegistryItem->begin();
}

static auto cbegin()
{
return mspRootRegistryItem->cbegin();
}

static auto end()
{
return mspRootRegistryItem->end();
}

static auto const cend()
{
return mspRootRegistryItem->cend();
}

static RegistryItem& GetItem(std::string const& rItemFullName);

template<typename TDataType>
static TDataType const& GetValue(std::string const& rItemFullName)
{
return GetItem(rItemFullName).GetValue<TDataType>();
}

static void RemoveItem(std::string const& ItemName);


static std::size_t size();

static bool HasItem(std::string const& rItemFullName);

static bool HasValue(std::string const& rItemFullName);

static bool HasItems(std::string const& rItemFullName);


std::string Info() const;

void PrintInfo(std::ostream& rOStream) const;

void PrintData(std::ostream& rOStream) const;

std::string ToJson(std::string const& Indentation) const;



private:

static RegistryItem* mspRootRegistryItem;








static RegistryItem& GetRootRegistryItem();

static std::vector<std::string> SplitFullName(std::string const& FullName);




Registry& operator=(Registry const& rOther);

Registry(Registry const& rOther);

}; 






inline std::istream& operator >> (
std::istream& rIStream,
Registry& rThis);

inline std::ostream& operator << (
std::ostream& rOStream,
const Registry& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
