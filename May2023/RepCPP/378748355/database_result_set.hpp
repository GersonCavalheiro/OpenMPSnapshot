

#pragma once

#include "database_result_set_row.hpp"
#include <Impl/pool_impl.hpp>
#include <queue>

using namespace Impl;

class LegacyDBResultImpl : public LegacyDBResult
{
private:
DynamicArray<char*> results_;
bool fieldsAreAdded = false;

public:
void addColumns(int fieldCount, char** fieldNames, char** values)
{
if (!fieldsAreAdded)
{
for (int field_index(0); field_index < fieldCount; field_index++)
{
results_.push_back(fieldNames[field_index]);
}
columns = fieldCount;
fieldsAreAdded = true;
}
results = results_.data();
}

void addField(char* value)
{
results_.push_back(const_cast<char*>(value ? value : ""));
results = results_.data();
}
};

class DatabaseResultSet final : public IDatabaseResultSet, public PoolIDProvider, public NoCopy
{
private:
std::queue<DatabaseResultSetRow> rows;

std::size_t rowCount;

LegacyDBResultImpl legacyDbResult;

public:
bool addRow(int fieldCount, char** values, char** fieldNames);

int getID() const override;

std::size_t getRowCount() const override;

bool selectNextRow() override;

std::size_t getFieldCount() const override;

bool isFieldNameAvailable(StringView fieldName) const override;

StringView getFieldName(std::size_t fieldIndex) const override;

StringView getFieldString(std::size_t fieldIndex) const override;

long getFieldInt(std::size_t fieldIndex) const override;

double getFieldFloat(std::size_t fieldIndex) const override;

StringView getFieldStringByName(StringView fieldName) const override;

long getFieldIntByName(StringView fieldName) const override;

double getFieldFloatByName(StringView fieldName) const override;

LegacyDBResult& getLegacyDBResult() override;
};
