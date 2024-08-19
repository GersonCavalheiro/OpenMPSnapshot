

#ifndef DB_H
#define DB_H 1

#include "Common/InsOrderedMap.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <config.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#if _SQL
# include <sqlite3.h>
#endif

typedef std::vector<std::vector<std::string> > dbVec; 
typedef InsOrderedMap<std::string,int> dbMap; 
typedef std::vector<std::string> dbVars;

class DB {
private:

dbMap statMap;
dbVars initVars, peVars;
#if _SQL
sqlite3* db;
sqlite3_stmt* stmt;
#else
void* db;
void* stmt;
#endif
std::string prog, cmd;
int exp;

int verbose_val;

std::string getSqliteName()
{
std::string name("dummy"); 
std::fstream fl("name.txt");
if (fl)
fl >> name;
return name;
}

void openDB(
const char* c, const int& v)
{
#if _SQL
verbose_val = v;
if (sqlite3_open(c, &db)) {
std::cerr << "[" << prog << "] Can't open DB.\n";
exit(EXIT_FAILURE);
} else {
if (verbose_val >= 2 && exp != READ)
std::cerr << "[" << prog << "] DB opened.\n";
}
#else
(void)v;
(void)c;
#endif
}

static int callback(
void* info, int argc, char** argv, char** colName)
{
int i;
std::cerr << (const char*)info << std::endl;
for (i=0; i<argc; i++) {
std::cerr << "%s = %s\n" << colName[i] << (argv[i] ? argv[i] : "NULL");
}
std::cerr << "\n";
return 0;
}

static unsigned int getRand()
{
srand(time(NULL));
return(rand() % 2000 + 200); 
}

void closeDB()
{
#if _SQL
if (sqlite3_close(db)) {
std::cerr << "[" << prog << "] Can't close DB.\n";
exit(EXIT_FAILURE);
} else {
if (verbose_val >= 2 && exp != READ)
std::cerr << "[" << prog << "] DB closed.\n";
}
#else
return;
#endif
}

void createTables();
void insertToMetaTables(const dbVars&);
std::string initializeRun();
std::string getPath(const std::string&);
bool definePeVars(const std::string&);
void assemblyStatsToDb();

public:

enum { NO_INIT, INIT, READ };

DB()
{
(void)db;
(void)stmt;

initVars.resize(3);
peVars.resize(3);
exp = NO_INIT;
}

~DB()
{
if (exp == INIT)
assemblyStatsToDb();
if (exp != NO_INIT)
closeDB();
}

friend void init(
DB& d, const std::string& path)
{
#if _SQL
d.exp = READ;
d.openDB(path.c_str(), 0);
#else
(void)d;
(void)path;
std::cerr << "error: `db` parameter has been used, but ABySS has not been configured to support SQLite.\n";
exit(EXIT_FAILURE);
#endif
}

friend void init(
DB& d,
const std::string& path,
const int& v,
const std::string& program,
const std::string& command,
const dbVars& vars)
{
#if _SQL
d.prog = program;
d.cmd = command;
d.initVars = vars;
d.exp = INIT;

std::string name(d.getSqliteName());
d.openDB(path.empty() ? name.c_str() : path.c_str(), v);
#else
(void)d;
(void)path;
(void)v;
(void)program;
(void)command;
(void)vars;
std::cerr << "error: `db` parameter has been used, but ABySS has not been configured to support SQLite.\n";
exit(EXIT_FAILURE);
#endif
}

std::string activateForeignKey(const std::string& s)
{
std::string s_pragma("pragma foreign_keys=on; ");
return s_pragma += s;
}

bool query(const std::string& s)
{
#if _SQL
int rc;
unsigned long int n;
char* errMsg = 0;
std::string new_s(activateForeignKey(s));
const char* statement = new_s.c_str();
n = 1;
do {
rc = sqlite3_exec(db, statement, callback, 0, &errMsg);
if (rc == SQLITE_OK)
return true;
n++;
usleep(getRand());
} while (rc == SQLITE_BUSY && n < 30000000); 
if (rc != SQLITE_OK) {
std::cerr << "[" << prog << "] SQL error: " << errMsg << std::endl;
sqlite3_free(errMsg);
exit(EXIT_FAILURE);
}
#else
(void)s;
#endif
return true;
}

friend void addToDb(
DB& d, const std::string& key, const int& value)
{
d.statMap.push_back(key, value);
}

friend void addToDb(
DB& d, dbMap m)
{
d.statMap.insert(m.getAC());
m.clear();
}

dbVec readSqlToVec(const std::string&);
std::string getProperTableName(const std::string&);
};

#endif
