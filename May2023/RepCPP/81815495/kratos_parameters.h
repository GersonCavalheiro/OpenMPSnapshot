
#pragma once

#include <filesystem>

#include "json/json_fwd.hpp" 

#include "includes/serializer.h"
#include "includes/ublas_interface.h"

namespace Kratos
{






class KRATOS_API(KRATOS_CORE) Parameters
{
private:

class KRATOS_API(KRATOS_CORE) iterator_adaptor
{
public:

using iterator_category = std::forward_iterator_tag;
using difference_type   = std::ptrdiff_t;
using value_type        = Parameters;
using pointer           = Parameters*;
using reference         = Parameters&;

using value_iterator = nlohmann::detail::iter_impl<nlohmann::json>; 



iterator_adaptor(value_iterator itValue, nlohmann::json* pValue,  Kratos::shared_ptr<nlohmann::json> pRoot);


iterator_adaptor(const iterator_adaptor& itValue);



iterator_adaptor& operator++();


iterator_adaptor operator++(int);


bool operator==(const iterator_adaptor& rhs) const;


bool operator!=(const iterator_adaptor& rhs) const;


Parameters& operator*() const;


Parameters* operator->() const;



inline value_iterator GetCurrentIterator() const;


const std::string name();


private:

std::size_t mDistance = 0;                       
nlohmann::json& mrValue;                         
std::unique_ptr<Parameters> mpParameters;        

};


class KRATOS_API(KRATOS_CORE) const_iterator_adaptor
{
public:

using iterator_category = std::forward_iterator_tag;
using difference_type   = std::ptrdiff_t;
using value_type        = Parameters;
using pointer           = const Parameters*;
using reference         = const Parameters&;

using value_iterator = nlohmann::detail::iter_impl<const nlohmann::json>; 



const_iterator_adaptor(value_iterator itValue, nlohmann::json* pValue,  Kratos::shared_ptr<nlohmann::json> pRoot);


const_iterator_adaptor(const const_iterator_adaptor& itValue);



const_iterator_adaptor& operator++();


const_iterator_adaptor operator++(int);


bool operator==(const const_iterator_adaptor& rhs) const;


bool operator!=(const const_iterator_adaptor& rhs) const;


const Parameters& operator*() const;


const Parameters* operator->() const;



inline value_iterator GetCurrentIterator() const;


const std::string name();

private:

std::size_t mDistance = 0;                       
nlohmann::json& mrValue;                         
std::unique_ptr<Parameters> mpParameters;        

};


public:

typedef std::size_t IndexType;

typedef std::size_t SizeType;

KRATOS_CLASS_POINTER_DEFINITION(Parameters);

using iterator = iterator_adaptor;
using const_iterator = const_iterator_adaptor;

typedef nlohmann::detail::iter_impl<nlohmann::json> json_iterator;
typedef nlohmann::detail::iter_impl<const nlohmann::json> json_const_iterator;
typedef nlohmann::detail::iteration_proxy<json_iterator> json_iteration_proxy;
typedef nlohmann::detail::iteration_proxy<json_const_iterator> json_const_iteration_proxy;



Parameters();


Parameters(const std::string& rJsonString);


Parameters(std::ifstream& rStringStream);

Parameters(Parameters const& rOther);

Parameters(Parameters&& rOther);

virtual ~Parameters()
{
}


Parameters& operator=(Parameters const& rOther);


Parameters operator[](const std::string& rEntry);


Parameters operator[](const std::string& rEntry) const;


Parameters operator[](const IndexType Index);


Parameters operator[](const IndexType Index) const;


Parameters& operator=(Parameters&& rOther);



Parameters Clone() const;


const std::string WriteJsonString() const;


const std::string PrettyPrintJsonString() const;


Parameters GetValue(const std::string& rEntry);


Parameters GetValue(const std::string& rEntry) const;


void SetValue(
const std::string& rEntry,
const Parameters& rOtherValue
);


void AddValue(
const std::string& rEntry,
const Parameters& rOtherValue
);


Parameters AddEmptyValue(const std::string& rEntry);


bool RemoveValue(const std::string& rEntry);


bool RemoveValues(const std::vector<std::string>& rEntries);


json_iteration_proxy items() noexcept;


json_const_iteration_proxy items() const noexcept;


bool Has(const std::string& rEntry) const;


bool IsNull() const;


bool IsNumber() const;


bool IsDouble() const;


bool IsInt() const;


bool IsBool() const;


bool IsString() const;


bool IsArray() const;


bool IsVector() const;


bool IsMatrix() const;


template <class TValue>
bool Is() const;


bool IsSubParameter() const;


double GetDouble() const;


int GetInt() const;


bool GetBool() const;


std::string GetString() const;


std::vector<std::string> GetStringArray() const;


Vector GetVector() const;


Matrix GetMatrix() const;


template <class TValue>
TValue Get() const;


void SetDouble(const double Value);


void SetInt(const int Value);


void SetBool(const bool Value);


void SetString(const std::string& rValue);


void SetStringArray(const std::vector<std::string>& rValue);


void SetVector(const Vector& rValue);


void SetMatrix(const Matrix& rValue);


template <class TValue>
void Set(const TValue& rValue);


void AddDouble(
const std::string& rEntry,
const double Value
);


void AddInt(
const std::string& rEntry,
const int Value
);


void AddBool(
const std::string& rEntry,
const bool Value
);


void AddString(
const std::string& rEntry,
const std::string& rValue
);


void AddStringArray(
const std::string& rEntry,
const std::vector<std::string>& rValue
);


void AddVector(
const std::string& rEntry,
const Vector& rValue
);


void AddMatrix(
const std::string& rEntry,
const Matrix& rValue
);


iterator begin();


iterator end();


const_iterator begin() const;


const_iterator end() const;


SizeType size() const;


void swap(Parameters& rOther) noexcept;


void Reset() noexcept;


Parameters GetArrayItem(const IndexType Index);


Parameters GetArrayItem(const IndexType Index) const;


void SetArrayItem(
const IndexType Index,
const Parameters& rOtherArrayItem
);


void AddEmptyArray(const std::string& rEntry);


void Append(const double Value);


void Append(const int Value);


void Append(const bool Value);


void Append(const std::string& rValue);


void Append(const Vector& rValue);


void Append(const Matrix& rValue);


void Append(const Parameters& rValue);


void CopyValuesFromExistingParameters(
const Parameters OriginParameters,
const std::vector<std::string>& rListParametersToCopy
);


void RecursivelyFindValue(
const nlohmann::json& rBaseValue,
const nlohmann::json& rValueToFind
) const;


bool IsEquivalentTo(Parameters& rParameters);


bool HasSameKeysAndTypeOfValuesAs(Parameters& rParameters);


void ValidateAndAssignDefaults(const Parameters& rDefaultParameters);


void RecursivelyValidateAndAssignDefaults(const Parameters& rDefaultParameters);


void AddMissingParameters(const Parameters& rDefaultParameters);


void RecursivelyAddMissingParameters(const Parameters& rDefaultParameters);


void ValidateDefaults(const Parameters& rDefaultParameters) const;


void RecursivelyValidateDefaults(const Parameters& rDefaultParameters) const;





virtual std::string Info() const
{
return this->PrettyPrintJsonString();
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "Parameters Object " << Info();
}

virtual void PrintData(std::ostream& rOStream) const
{
};

private:

friend class Serializer;

void save(Serializer& rSerializer) const;

void load(Serializer& rSerializer);


nlohmann::json* mpValue;                   
Kratos::shared_ptr<nlohmann::json> mpRoot; 




Parameters(nlohmann::json* pValue, Kratos::shared_ptr<nlohmann::json> pRoot);


Parameters(json_iterator itValue, nlohmann::json* pValue, Kratos::shared_ptr<nlohmann::json> pRoot);


Parameters(json_const_iterator itValue, nlohmann::json* pValue, Kratos::shared_ptr<nlohmann::json> pRoot);


nlohmann::json* GetUnderlyingStorage();


nlohmann::json* GetUnderlyingStorage() const;


void SetUnderlyingSotrage(nlohmann::json* pNewValue);


Kratos::shared_ptr<nlohmann::json> GetUnderlyingRootStorage();


Kratos::shared_ptr<nlohmann::json> GetUnderlyingRootStorage() const;


void SetUnderlyingRootStorage(Kratos::shared_ptr<nlohmann::json> pNewValue);

void InternalSetValue(const Parameters& rOtherValue);


void SolveIncludes(nlohmann::json& rJson, const std::filesystem::path& rFileName, std::vector<std::filesystem::path>& rIncludeSequence);


nlohmann::json ReadFile(const std::filesystem::path& rFileName);

}; 





inline std::istream& operator >> (std::istream& rIStream,
Parameters& rThis)
{
return rIStream;
}

inline std::ostream& operator << (std::ostream& rOStream,
const Parameters& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
