

#pragma once

#include <pool.hpp>
#include <sqlite3.h>

#include "database_result_set.hpp"
#include <Impl/pool_impl.hpp>

using namespace Impl;

class DatabasesComponent;

class DatabaseConnection final : public IDatabaseConnection, public PoolIDProvider, public NoCopy
{
private:
DatabasesComponent* parentDatabasesComponent;

sqlite3* databaseConnectionHandle;

public:
DatabaseConnection(DatabasesComponent* parentDatabasesComponent, sqlite3* databaseConnectionHandle);

int getID() const override;

bool close() override;

IDatabaseResultSet* executeQuery(StringView query) override;

private:
static int queryStepExecuted(void* userData, int fieldCount, char** values, char** fieldNames);
};
