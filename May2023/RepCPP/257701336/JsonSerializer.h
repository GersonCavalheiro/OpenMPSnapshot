

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/Array.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/external/json-cpp/json.h>

#include <utility>

namespace Aws
{
namespace Utils
{
namespace Json
{

class AWS_CORE_API JsonValue
{
public:

JsonValue();



JsonValue(const Aws::String& value);


JsonValue(Aws::IStream& istream);


JsonValue(const JsonValue& value);


JsonValue(JsonValue&& value);

~JsonValue();

JsonValue& operator=(const JsonValue& other);

JsonValue& operator=(JsonValue&& other);

bool operator!=(const JsonValue& other)
{
return m_value != other.m_value;
}

bool operator==(const JsonValue& other)
{
return m_value == other.m_value;
}


Aws::String GetString(const Aws::String& key) const;
Aws::String GetString(const char* key) const;


JsonValue& WithString(const Aws::String& key, const Aws::String& value);
JsonValue& WithString(const char* key, const Aws::String& value);


JsonValue& AsString(const Aws::String& value);


Aws::String AsString() const;


bool GetBool(const Aws::String& key) const;
bool GetBool(const char* key) const;


JsonValue& WithBool(const Aws::String& key, bool value);
JsonValue& WithBool(const char* key, bool value);


JsonValue& AsBool(bool value);


bool AsBool() const;


int GetInteger(const Aws::String& key) const;
int GetInteger(const char* key) const;


JsonValue& WithInteger(const Aws::String& key, int value);
JsonValue& WithInteger(const char* key, int value);


JsonValue& AsInteger(int value);


int AsInteger() const;


long long GetInt64(const Aws::String& key) const;
long long GetInt64(const char* key) const;


JsonValue& WithInt64(const Aws::String& key, long long value);
JsonValue& WithInt64(const char* key, long long value);


JsonValue& AsInt64(long long value);


long long AsInt64() const;


double GetDouble(const Aws::String& key) const;
double GetDouble(const char* key) const;


JsonValue& WithDouble(const Aws::String& key, double value);
JsonValue& WithDouble(const char* key, double value);


JsonValue& AsDouble(double value);


double AsDouble() const;


Array<JsonValue> GetArray(const Aws::String& key) const;
Array<JsonValue> GetArray(const char* key) const;


JsonValue& WithArray(const Aws::String& key, const Array<Aws::String>& array);
JsonValue& WithArray(const char* key, const Array<Aws::String>& array);


JsonValue& WithArray(const Aws::String& key, Array<Aws::String>&& array);


JsonValue& WithArray(const Aws::String& key, const Array<JsonValue>& array);


JsonValue& WithArray(const Aws::String& key, Array<JsonValue>&& array);


JsonValue& AsArray(const Array<JsonValue>& array);


JsonValue& AsArray(Array<JsonValue>&& array);


Array<JsonValue> AsArray() const;


JsonValue GetObject(const char* key) const;
JsonValue GetObject(const Aws::String& key) const;


JsonValue& WithObject(const Aws::String& key, const JsonValue& value);
JsonValue& WithObject(const char* key, const JsonValue& value);


JsonValue& WithObject(const Aws::String& key, const JsonValue&& value);
JsonValue& WithObject(const char* key, const JsonValue&& value);


JsonValue& AsObject(const JsonValue& value);


JsonValue& AsObject(JsonValue&& value);


JsonValue AsObject() const;


Aws::Map<Aws::String, JsonValue> GetAllObjects() const;


bool ValueExists(const char* key) const;
bool ValueExists(const Aws::String& key) const;


Aws::String WriteCompact(bool treatAsObject = true) const;


void WriteCompact(Aws::OStream& ostream, bool treatAsObject = true) const;


Aws::String WriteReadable(bool treatAsObject = true) const;


void WriteReadable(Aws::OStream& ostream, bool treatAsObject = true) const;


inline bool WasParseSuccessful() const
{
return m_wasParseSuccessful;
}


inline const Aws::String& GetErrorMessage() const
{
return m_errorMessage;
}

void AppendValue(const JsonValue& value);

bool IsObject() const;
bool IsBool() const;
bool IsString() const;
bool IsIntegerType() const;
bool IsFloatingPointType() const;
bool IsListType() const;

Aws::External::Json::Value& ModifyRawValue() { return m_value; }

private:
JsonValue(const Aws::External::Json::Value& value);

JsonValue& operator=(Aws::External::Json::Value& other);

mutable Aws::External::Json::Value m_value;
bool m_wasParseSuccessful;
Aws::String m_errorMessage;
};

} 
} 
} 

