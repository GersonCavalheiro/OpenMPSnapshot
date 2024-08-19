

#pragma once

#include "database_connection.hpp"
#include <Impl/pool_impl.hpp>

using namespace Impl;

class DatabasesComponent final : public IDatabasesComponent, public NoCopy
{
private:
DynamicPoolStorage<DatabaseConnection, IDatabaseConnection, 1, 1025> databaseConnections;

DynamicPoolStorage<DatabaseResultSet, IDatabaseResultSet, 1, 2049> databaseResultSets;

bool* logSQLite_;
bool* logSQLiteQueries_;

ICore* core_;

public:
IDatabaseResultSet* createResultSet();

DatabasesComponent();

StringView componentName() const override
{
return "Databases";
}

ComponentType componentType() const override
{
return ComponentType::Other;
}

SemanticVersion componentVersion() const override
{
return SemanticVersion(OMP_VERSION_MAJOR, OMP_VERSION_MINOR, OMP_VERSION_PATCH, BUILD_NUMBER);
}

void onLoad(ICore* c) override;

IDatabaseConnection* open(StringView path, int flags = 0) override;

bool close(IDatabaseConnection& connection) override;

bool freeResultSet(IDatabaseResultSet& resultSet) override;

std::size_t getDatabaseConnectionCount() const override;

bool isDatabaseConnectionIDValid(int databaseConnectionID) const override;

IDatabaseConnection& getDatabaseConnectionByID(int databaseConnectionID) override;

std::size_t getDatabaseResultSetCount() const override;

bool isDatabaseResultSetIDValid(int databaseResultSetID) const override;

IDatabaseResultSet& getDatabaseResultSetByID(int databaseResultSetID) override;

void log(LogLevel level, const char* fmt, ...) const;

void logQuery(const char* fmt, ...) const;

void free() override
{
delete this;
}

void reset() override
{
}
};
