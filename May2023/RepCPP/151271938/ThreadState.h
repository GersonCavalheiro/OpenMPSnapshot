#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <faabric/proto/faabric.pb.h>
#include <faabric/util/environment.h>
#include <faabric/util/locks.h>

namespace threads {

class Level
{
public:
int32_t depth = 0;

int32_t activeLevels = 0;

int32_t maxActiveLevels = 1;

int32_t numThreads = 1;

int32_t wantedThreads = -1;

int32_t pushedThreads = -1;

int32_t globalTidOffset = 0;

uint32_t nSharedVarOffsets = 0;
std::unique_ptr<uint32_t[]> sharedVarOffsets;
static_assert(sizeof(sharedVarOffsets) == sizeof(uint32_t*));

static std::shared_ptr<Level> deserialise(
const std::vector<uint8_t>& bytes);

Level(int32_t numThreadsIn);

std::vector<uint32_t> getSharedVarOffsets();

void setSharedVarOffsets(uint32_t* ptr, int nVars);

void fromParentLevel(const std::shared_ptr<Level>& parent);

int getMaxThreadsAtNextLevel() const;

std::vector<uint8_t> serialise();

int getLocalThreadNum(const faabric::Message* msg);

int getGlobalThreadNum(int localThreadNum);

int getGlobalThreadNum(const faabric::Message* msg);

std::string toString();
};

class PthreadCall
{
public:
int32_t pthreadPtr;
int32_t entryFunc;
int32_t argsPtr;
};

std::shared_ptr<Level> levelFromBatchRequest(
const std::shared_ptr<faabric::BatchExecuteRequest>& req);

std::shared_ptr<Level> getCurrentOpenMPLevel();

void setCurrentOpenMPLevel(
const std::shared_ptr<faabric::BatchExecuteRequest> req);

void setCurrentOpenMPLevel(const std::shared_ptr<Level>& level);
}
