
#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <any>



#include "includes/define.h"

namespace Kratos
{


class KRATOS_API(KRATOS_CORE) RegistryItem
{
public:

KRATOS_CLASS_POINTER_DEFINITION(RegistryItem);

using SubRegistryItemType = std::unordered_map<std::string, Kratos::shared_ptr<RegistryItem>>;

using SubRegistryItemPointerType = Kratos::shared_ptr<SubRegistryItemType>;

class KeyReturnConstIterator
{
public:

using BaseIterator      = SubRegistryItemType::const_iterator;
using iterator_category = std::forward_iterator_tag;
using difference_type   = BaseIterator::difference_type;
using value_type        = SubRegistryItemType::key_type;
using const_pointer     = const value_type*;
using const_reference   = const value_type&;


KeyReturnConstIterator()
{}

KeyReturnConstIterator(const BaseIterator Iterator)
: mIterator(Iterator)
{}

KeyReturnConstIterator(const KeyReturnConstIterator& rIterator)
: mIterator(rIterator.mIterator)
{}


KeyReturnConstIterator& operator=(const KeyReturnConstIterator& rIterator)
{
this->mIterator = rIterator.mIterator;
return *this;
}

const_reference operator*() const
{
return mIterator->first;
}

const_pointer operator->() const
{
return &(mIterator->first);
}

KeyReturnConstIterator& operator++()
{
++mIterator;
return *this;
}

KeyReturnConstIterator operator++(int)
{
KeyReturnConstIterator tmp(*this);
++(*this);
return tmp;
}

bool operator==(const KeyReturnConstIterator& rIterator) const
{
return this->mIterator == rIterator.mIterator;
}

bool operator!=(const KeyReturnConstIterator& rIterator) const
{
return this->mIterator != rIterator.mIterator;
}

private:

BaseIterator mIterator;

};


RegistryItem() = delete;

RegistryItem(const std::string& rName)
: mName(rName),
mpValue(Kratos::make_shared<SubRegistryItemType>()),
mGetValueStringMethod(&RegistryItem::GetRegistryItemType) {}

template <typename TItemType, typename... TArgs>
RegistryItem(
const std::string &rName,
const std::function<std::shared_ptr<TItemType>(TArgs...)> &rValue)
: mName(rName),
mpValue(rValue()),
mGetValueStringMethod(&RegistryItem::GetItemString<TItemType>) {}

template<class TItemType>
RegistryItem(
const std::string&  rName,
const TItemType& rValue)
: mName(rName),
mpValue(Kratos::make_shared<TItemType>(rValue)),
mGetValueStringMethod(&RegistryItem::GetItemString<TItemType>) {}

template<class TItemType>
RegistryItem(
const std::string&  rName,
const shared_ptr<TItemType>& pValue)
: mName(rName),
mpValue(pValue),
mGetValueStringMethod(&RegistryItem::GetItemString<TItemType>) {}

RegistryItem(RegistryItem const& rOther) = delete;

~RegistryItem() = default;

RegistryItem& operator=(RegistryItem& rOther) = delete;


template<typename TItemType, class... TArgumentsList>
RegistryItem& AddItem(
std::string const& ItemName,
TArgumentsList&&... Arguments)
{
KRATOS_ERROR_IF(this->HasItem(ItemName))
<< "The RegistryItem '" << this->Name() << "' already has an item with name "
<< ItemName << "." << std::endl;

using ValueType = typename std::conditional<std::is_same<TItemType, RegistryItem>::value, SubRegistryItemFunctor, SubValueItemFunctor<TItemType>>::type;

auto insert_result = GetSubRegistryItemMap().emplace(
std::make_pair(
ItemName,
ValueType::Create(ItemName, std::forward<TArgumentsList>(Arguments)...)
)
);

KRATOS_ERROR_IF_NOT(insert_result.second)
<< "Error in inserting '" << ItemName
<< "' in registry item with name '" << this->Name() << "'." << std::endl;

return *insert_result.first->second;
}


SubRegistryItemType::iterator begin();

SubRegistryItemType::const_iterator cbegin() const;

SubRegistryItemType::iterator end();

SubRegistryItemType::const_iterator cend() const;

KeyReturnConstIterator KeyConstBegin() const;

KeyReturnConstIterator KeyConstEnd() const;

const std::string& Name() const  { return mName; }

RegistryItem const& GetItem(std::string const& rItemName) const;

RegistryItem& GetItem(std::string const& rItemName);

template<typename TDataType> TDataType const& GetValue() const
{
KRATOS_TRY

return *(std::any_cast<std::shared_ptr<TDataType>>(mpValue));

KRATOS_CATCH("");
}

void RemoveItem(std::string const& rItemName);


std::size_t size();

bool HasValue() const;

bool HasItems() const;

bool HasItem(std::string const& rItemName) const;


std::string Info() const;

void PrintInfo(std::ostream& rOStream) const;

void PrintData(std::ostream& rOStream) const;

std::string ToJson(std::string const& rTabSpacing = "", const std::size_t Level = 0) const;

private:

std::string mName;
std::any mpValue;
std::string (RegistryItem::*mGetValueStringMethod)() const;


std::string GetRegistryItemType() const
{
return mpValue.type().name();
}

template<class TItemType>
std::string GetItemString() const
{
std::stringstream buffer;
buffer << this->GetValue<TItemType>();
return buffer.str();
}


class SubRegistryItemFunctor
{
public:
template<class... TArgumentsList>
static inline RegistryItem::Pointer Create(
std::string const& ItemName,
TArgumentsList&&... Arguments)
{
return Kratos::make_shared<RegistryItem>(ItemName);
}
};

template<typename TItemType>
class SubValueItemFunctor
{
public:
template<class... TArgumentsList, class TFunctionType = std::function<std::shared_ptr<TItemType>(TArgumentsList...)>>
static inline RegistryItem::Pointer Create(
std::string const& ItemName,
TFunctionType && Function)
{
return Kratos::make_shared<RegistryItem>(ItemName, std::forward<TFunctionType>(Function));
}

template<class... TArgumentsList>
static inline RegistryItem::Pointer Create(
std::string const& ItemName,
TArgumentsList&&... Arguments)
{
return Kratos::make_shared<RegistryItem>(ItemName, Kratos::make_shared<TItemType>(std::forward<TArgumentsList>(Arguments)...));
}

};


std::string GetValueString() const;

SubRegistryItemType& GetSubRegistryItemMap();

SubRegistryItemType& GetSubRegistryItemMap() const;


}; 


inline std::istream& operator >> (
std::istream& rIStream,
RegistryItem& rThis);

inline std::ostream& operator << (
std::ostream& rOStream,
const RegistryItem& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
