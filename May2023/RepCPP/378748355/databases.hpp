#pragma once

#include <sdk.hpp>

struct LegacyDBResult
{
int rows;
int columns;
char** results;
};

struct IDatabaseResultSetRow
{

virtual std::size_t getFieldCount() const = 0;

virtual bool isFieldNameAvailable(StringView fieldName) const = 0;

virtual StringView getFieldName(std::size_t fieldIndex) const = 0;

virtual StringView getFieldString(std::size_t fieldIndex) const = 0;

virtual long getFieldInt(std::size_t fieldIndex) const = 0;

virtual double getFieldFloat(std::size_t fieldIndex) const = 0;

virtual StringView getFieldStringByName(StringView fieldName) const = 0;

virtual long getFieldIntByName(StringView fieldName) const = 0;

virtual double getFieldFloatByName(StringView fieldName) const = 0;
};

struct IDatabaseResultSet : public IExtensible, public IIDProvider
{

virtual std::size_t getRowCount() const = 0;

virtual bool selectNextRow() = 0;

virtual std::size_t getFieldCount() const = 0;

virtual bool isFieldNameAvailable(StringView fieldName) const = 0;

virtual StringView getFieldName(std::size_t fieldIndex) const = 0;

virtual StringView getFieldString(std::size_t fieldIndex) const = 0;

virtual long getFieldInt(std::size_t fieldIndex) const = 0;

virtual double getFieldFloat(std::size_t fieldIndex) const = 0;

virtual StringView getFieldStringByName(StringView fieldName) const = 0;

virtual long getFieldIntByName(StringView fieldName) const = 0;

virtual double getFieldFloatByName(StringView fieldName) const = 0;

virtual LegacyDBResult& getLegacyDBResult() = 0;
};

struct IDatabaseConnection : public IExtensible, public IIDProvider
{

virtual bool close() = 0;

virtual IDatabaseResultSet* executeQuery(StringView query) = 0;
};

static const UID DatabasesComponent_UID = UID(0x80092e7eb5821a96 );
struct IDatabasesComponent : public IComponent
{
PROVIDE_UID(DatabasesComponent_UID);

virtual IDatabaseConnection* open(StringView path, int flags = 0) = 0;

virtual bool close(IDatabaseConnection& connection) = 0;

virtual bool freeResultSet(IDatabaseResultSet& resultSet) = 0;

virtual std::size_t getDatabaseConnectionCount() const = 0;

virtual bool isDatabaseConnectionIDValid(int databaseConnectionID) const = 0;

virtual IDatabaseConnection& getDatabaseConnectionByID(int databaseConnectionID) = 0;

virtual std::size_t getDatabaseResultSetCount() const = 0;

virtual bool isDatabaseResultSetIDValid(int databaseResultSetID) const = 0;

virtual IDatabaseResultSet& getDatabaseResultSetByID(int databaseResultSetID) = 0;
};
