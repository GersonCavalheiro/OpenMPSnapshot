
# pragma once

#include <string>
#include <iostream>
#include <cstddef>
#include <unordered_map>


#include "includes/define.h"
#include "includes/accessor.h"
#include "includes/node.h"
#include "includes/indexed_object.h"
#include "containers/data_value_container.h"
#include "includes/process_info.h"
#include "includes/table.h"
#include "utilities/string_utilities.h"

namespace Kratos
{







class Properties : public IndexedObject
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Properties);

#ifdef  _WIN32 
using int64_t = __int64;
#endif
using BaseType = IndexedObject;

using ContainerType = DataValueContainer;

using GeometryType = Geometry<Node> ;

using IndexType = std::size_t;

using TableType = Table<double>;

using KeyType = IndexType;

using AccessorPointerType = Accessor::UniquePointer;

using AccessorsContainerType = std::unordered_map<KeyType, AccessorPointerType>;

using TablesContainerType = std::unordered_map<std::size_t, TableType>; 

using SubPropertiesContainerType = PointerVectorSet<Properties, IndexedObject>;


explicit Properties(IndexType NewId = 0)
: BaseType(NewId)
, mData()
, mTables()
, mSubPropertiesList()
, mAccessors() {}

explicit Properties(IndexType NewId, const SubPropertiesContainerType& SubPropertiesList)
: BaseType(NewId)
, mData()
, mTables()
, mSubPropertiesList(SubPropertiesList)
, mAccessors() {}

Properties(const Properties& rOther)
: BaseType(rOther)
, mData(rOther.mData)
, mTables(rOther.mTables)
, mSubPropertiesList(rOther.mSubPropertiesList)
{
for (auto& r_item : rOther.mAccessors) {
const auto key = r_item.first;
const auto& rp_accessor = r_item.second;
mAccessors.emplace(key, rp_accessor->Clone());
}
}

~Properties() override {}


Properties& operator=(const Properties& rOther)
{
BaseType::operator=(rOther);
mData = rOther.mData;
mTables = rOther.mTables;
mSubPropertiesList = rOther.mSubPropertiesList;
for (auto& r_item : rOther.mAccessors) {
const auto key = r_item.first;
const auto& rp_accessor = r_item.second;
mAccessors.emplace(key, rp_accessor->Clone());
}
return *this;
}

template<class TVariableType>
typename TVariableType::Type& operator()(const TVariableType& rV)
{
return GetValue(rV);
}

template<class TVariableType>
typename TVariableType::Type const& operator()(const TVariableType& rV) const
{
return GetValue(rV);
}

template<class TVariableType>
typename TVariableType::Type& operator[](const TVariableType& rV)
{
return GetValue(rV);
}

template<class TVariableType>
typename TVariableType::Type const& operator[](const TVariableType& rV) const
{
return GetValue(rV);
}

template<class TVariableType>
typename TVariableType::Type& operator()(const TVariableType& rV, Node& rThisNode)
{
return GetValue(rV, rThisNode);
}

template<class TVariableType>
typename TVariableType::Type const& operator()(const TVariableType& rV, Node const& rThisNode) const
{
return GetValue(rV, rThisNode);
}

template<class TVariableType>
typename TVariableType::Type& operator()(const TVariableType& rV, Node& rThisNode, IndexType SolutionStepIndex)
{
return GetValue(rV, rThisNode, SolutionStepIndex);
}

template<class TVariableType>
typename TVariableType::Type const& operator()(const TVariableType& rV, Node const& rThisNode, IndexType SolutionStepIndex) const
{
return GetValue(rV, rThisNode, SolutionStepIndex);
}

template<class TVariableType>
typename TVariableType::Type& operator()(const TVariableType& rV, Node& rThisNode, ProcessInfo const& rCurrentProcessInfo)
{
return GetValue(rV, rThisNode, rCurrentProcessInfo.GetSolutionStepIndex());
}

template<class TVariableType>
typename TVariableType::Type const& operator()(const TVariableType& rV, Node const& rThisNode, ProcessInfo const& rCurrentProcessInfo) const
{
return GetValue(rV, rThisNode, rCurrentProcessInfo.GetSolutionStepIndex());
}


template<class TVariableType>
void Erase(const TVariableType& rV)
{
mData.Erase(rV);
}

template<class TVariableType>
typename TVariableType::Type& GetValue(const TVariableType& rVariable)
{
return mData.GetValue(rVariable);
}

template<class TVariableType>
typename TVariableType::Type const& GetValue(const TVariableType& rVariable) const
{

return mData.GetValue(rVariable);
}

template<class TVariableType>
typename TVariableType::Type& GetValue(const TVariableType& rVariable, Node& rThisNode)
{
if (mData.Has(rVariable))
return mData.GetValue(rVariable);
return rThisNode.GetValue(rVariable);
}

template<class TVariableType>
typename TVariableType::Type const& GetValue(const TVariableType& rVariable, Node const& rThisNode) const
{
if (mData.Has(rVariable))
return mData.GetValue(rVariable);
return rThisNode.GetValue(rVariable);
}

template<class TVariableType>
typename TVariableType::Type& GetValue(const TVariableType& rVariable, Node& rThisNode, IndexType SolutionStepIndex)
{
if (mData.Has(rVariable))
return mData.GetValue(rVariable);
return rThisNode.GetValue(rVariable, SolutionStepIndex);
}

template<class TVariableType>
typename TVariableType::Type const& GetValue(const TVariableType& rVariable, Node const& rThisNode, IndexType SolutionStepIndex) const
{
if (mData.Has(rVariable))
return mData.GetValue(rVariable);
return rThisNode.GetValue(rVariable, SolutionStepIndex);
}


template<class TVariableType>
typename TVariableType::Type GetValue(const TVariableType& rVariable, const GeometryType& rGeometry, const Vector& rShapeFunctionVector, const ProcessInfo& rProcessInfo) const
{
auto it_value = mAccessors.find(rVariable.Key());
if (it_value != mAccessors.end()) {
return (it_value->second)->GetValue(rVariable, *this, rGeometry, rShapeFunctionVector, rProcessInfo);
} else {
return mData.GetValue(rVariable);
}
}


template<class TVariableType>
void SetAccessor(const TVariableType& rVariable, AccessorPointerType pAccessor)
{
mAccessors.emplace(rVariable.Key(), std::move(pAccessor));
}

template<class TVariableType>
void SetValue(TVariableType const& rV, typename TVariableType::Type const& rValue)
{
mData.SetValue(rV, rValue);
}

bool HasVariables() const
{
return !mData.IsEmpty();
}

template<class TXVariableType, class TYVariableType>
TableType& GetTable(const TXVariableType& XVariable, const TYVariableType& YVariable)
{
return mTables[Key(XVariable.Key(), YVariable.Key())];
}

template<class TXVariableType, class TYVariableType>
TableType const& GetTable(const TXVariableType& XVariable, const TYVariableType& YVariable) const
{
return mTables.at(Key(XVariable.Key(), YVariable.Key()));
}

template<class TXVariableType, class TYVariableType>
void SetTable(const TXVariableType& XVariable, const TYVariableType& YVariable, TableType const& rThisTable)
{
mTables[Key(XVariable.Key(), YVariable.Key())] = rThisTable;
}

bool HasTables() const
{
return !mTables.empty();
}

bool IsEmpty() const
{
return !( HasVariables() || HasTables() );
}

int64_t Key(std::size_t XKey, std::size_t YKey) const
{
int64_t result_key = XKey;
result_key = result_key << 32;
result_key |= YKey; 
return result_key;
}


std::size_t NumberOfSubproperties() const
{
return mSubPropertiesList.size();
}


void AddSubProperties(Properties::Pointer pNewSubProperty)
{
KRATOS_DEBUG_ERROR_IF(this->HasSubProperties(pNewSubProperty->Id())) << "SubProperties with ID: " << pNewSubProperty->Id() << " already defined" << std::endl;
mSubPropertiesList.insert(mSubPropertiesList.begin(), pNewSubProperty);
}


bool HasSubProperties(const IndexType SubPropertyIndex) const
{
return mSubPropertiesList.find(SubPropertyIndex) != mSubPropertiesList.end();
}


Properties::Pointer pGetSubProperties(const IndexType SubPropertyIndex)
{
auto property_iterator = mSubPropertiesList.find(SubPropertyIndex);
if (property_iterator != mSubPropertiesList.end()) {
return *(property_iterator.base());
} else {
KRATOS_ERROR << "Subproperty ID: " << SubPropertyIndex << " is not defined on the current Properties ID: " << this->Id() << " creating a new one with ID: " << SubPropertyIndex << std::endl;
return nullptr;
}
}


const Properties::Pointer pGetSubProperties(const IndexType SubPropertyIndex) const
{
auto property_iterator = mSubPropertiesList.find(SubPropertyIndex);
if (property_iterator != mSubPropertiesList.end()) {
return *(property_iterator.base());
} else {
KRATOS_ERROR << "Subproperty ID: " << SubPropertyIndex << " is not defined on the current Properties ID: " << this->Id() << " creating a new one with ID: " << SubPropertyIndex << std::endl;
return nullptr;
}
}


Properties& GetSubProperties(const IndexType SubPropertyIndex)
{
auto property_iterator = mSubPropertiesList.find(SubPropertyIndex);
if (property_iterator != mSubPropertiesList.end()) {
return *(property_iterator);
} else {
KRATOS_ERROR << "Subproperty ID: " << SubPropertyIndex << " is not defined on the current Properties ID: " << this->Id() << " creating a new one with ID: " << SubPropertyIndex << std::endl;
return *this;
}
}


const Properties& GetSubProperties(const IndexType SubPropertyIndex) const
{
if (mSubPropertiesList.find(SubPropertyIndex) != mSubPropertiesList.end()) {
return *(mSubPropertiesList.find(SubPropertyIndex));
} else {
KRATOS_ERROR << "Subproperty ID: " << SubPropertyIndex << " is not defined on the current Properties ID: " << this->Id() << std::endl;
}
}


SubPropertiesContainerType& GetSubProperties()
{
return mSubPropertiesList;
}


SubPropertiesContainerType const& GetSubProperties() const
{
return mSubPropertiesList;
}


void SetSubProperties(SubPropertiesContainerType& rSubPropertiesList)
{
mSubPropertiesList = rSubPropertiesList;
}



ContainerType& Data()
{
return mData;
}


ContainerType const& Data() const
{
return mData;
}


TablesContainerType& Tables()
{
return mTables;
}


TablesContainerType const& Tables() const
{
return mTables;
}


template<class TVariableType>
bool Has(TVariableType const& rThisVariable) const
{
return mData.Has(rThisVariable);
}

template<class TXVariableType, class TYVariableType>
bool HasTable(const TXVariableType& XVariable, const TYVariableType& YVariable) const
{
return (mTables.find(Key(XVariable.Key(), YVariable.Key())) != mTables.end());
}


std::string Info() const override
{
return "Properties";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream <<  "Properties";
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << "Id : " << this->Id() << "\n";

mData.PrintData(rOStream);

if (mTables.size() > 0) {
rOStream << "This properties contains " << mTables.size() << " tables\n";
for (auto& r_table : mTables) {
rOStream << "Table key: " << r_table.first << "\n";
StringUtilities::PrintDataWithIdentation(rOStream, r_table.second);
}
}

if (mSubPropertiesList.size() > 0) {
rOStream << "\nThis properties contains " << mSubPropertiesList.size() << " subproperties\n";
for (auto& r_subprop : mSubPropertiesList) {
StringUtilities::PrintDataWithIdentation(rOStream, r_subprop);
}
}

if (mAccessors.size() > 0) {
rOStream << "\nThis properties contains " << mAccessors.size() << " accessors\n";
for (auto& r_entry : mAccessors) {
rOStream << "Accessor for variable key: " << r_entry.first << "\n";
StringUtilities::PrintDataWithIdentation(rOStream, *r_entry.second);
}
}
}


protected:







private:


ContainerType mData;                        

TablesContainerType mTables;                

SubPropertiesContainerType mSubPropertiesList; 

AccessorsContainerType mAccessors = {}; 




friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, IndexedObject );
rSerializer.save("Data", mData);
rSerializer.save("Tables", mTables);
rSerializer.save("SubPropertiesList", mSubPropertiesList);
std::vector<std::pair<const KeyType, Accessor*>> aux_accessors_container;
for (auto& r_item : mAccessors) {
const auto key = r_item.first;
const auto& rp_accessor = r_item.second;
aux_accessors_container.push_back(std::make_pair(key, &(*rp_accessor)));
}
rSerializer.save("Accessors", aux_accessors_container);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, IndexedObject );
rSerializer.load("Data", mData);
rSerializer.load("Tables", mTables);
rSerializer.load("SubPropertiesList", mSubPropertiesList);
std::vector<std::pair<const KeyType, Accessor*>> aux_accessors_container;
rSerializer.load("Accessors", aux_accessors_container);
for (auto& r_item : aux_accessors_container) {
const auto key = r_item.first;
const auto& rp_accessor = r_item.second;
mAccessors.emplace(key, rp_accessor->Clone());
}
}





}; 




inline std::istream& operator >> (std::istream& rIStream,
Properties& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const Properties& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  