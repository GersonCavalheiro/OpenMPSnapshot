

#pragma once

#include <Server/Components/Databases/databases.hpp>
#include <types.hpp>

using namespace Impl;

class DatabaseResultSetRow final : public IDatabaseResultSetRow
{
private:
DynamicArray<Pair<String, String>> fields;

FlatHashMap<String, std::size_t> fieldNameToFieldIndexLookup;

public:
bool addField(StringView value, StringView fieldName);

std::size_t getFieldCount() const override;

bool isFieldNameAvailable(StringView fieldName) const override;

StringView getFieldName(std::size_t fieldIndex) const override;

StringView getFieldString(std::size_t fieldIndex) const override;

long getFieldInt(std::size_t fieldIndex) const override;

double getFieldFloat(std::size_t fieldIndex) const override;

StringView getFieldStringByName(StringView fieldName) const override;

long getFieldIntByName(StringView fieldName) const override;

double getFieldFloatByName(StringView fieldName) const override;
};
