
#if (defined _DEBUG) && (defined _MSC_VER) && (defined VLD)
#include "vld.h"
#endif 
#ifdef _MSC_VER
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#pragma warning(disable : 4996) 
#pragma warning(disable : 4127) 
#endif
#include "gtest/gtest.h"
#ifdef USE_GDAL
#include <gdal.h>
#include <utility>
#endif

#include "test_global.h"
#include "../src/basic.h"
#include "../src/utils_filesystem.h"
#include "../src/utils_string.h"
#include "../src/db_mongoc.h"

using namespace ccgl;
using namespace utils_filesystem;

GlobalEnvironment* GlobalEnv;

int main(int argc, char** argv) {
::testing::InitGoogleTest(&argc, argv);
int i = 1;
char* strend = nullptr;
string mongo_host = "127.0.0.1";
vint16_t mongo_port = 27017;
while (argc > i) {
if (utils_string::StringMatch(argv[i], "-host")) {
i++;
if (argc > i) {
mongo_host = argv[i];
i++;
}
}
else if (utils_string::StringMatch(argv[i], "-port")) {
i++;
if (argc > i) {
mongo_port = static_cast<vint16_t>(strtol(argv[i], &strend, 10));
i++;
}
}
}
#ifdef USE_MONGODB
using namespace db_mongoc;
MongoClient* client_ = MongoClient::Init(mongo_host.c_str(), mongo_port);
MongoGridFs* gfs_ = new MongoGridFs(client_->GetGridFs("test", "spatial"));
GlobalEnv = new GlobalEnvironment(client_, gfs_);
::testing::AddGlobalTestEnvironment(GlobalEnv);
#endif

SetDefaultOpenMPThread();

#ifdef USE_GDAL
GDALAllRegister(); 
#endif
string apppath = GetAppPath();
string resultpath = apppath + "./data/raster/result";
if (!DirectoryExists(resultpath)) { CleanDirectory(resultpath); }

#if (defined _DEBUG) && (defined _MSC_VER) && (defined VLD)
_CrtMemState memoryState = { 0 };
(void)memoryState;
_CrtMemCheckpoint(&memoryState);
#endif 

int retval = RUN_ALL_TESTS();

#if (defined _DEBUG) && (defined _MSC_VER) && (defined VLD)
_CrtMemDumpAllObjectsSince(&memoryState);
#endif 
return retval;
}
