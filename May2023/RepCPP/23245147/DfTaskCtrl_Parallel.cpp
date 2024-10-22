
#include "DfTaskCtrl_Parallel.h"
#include "TlCommunicate.h"

DfTaskCtrl_Parallel::DfTaskCtrl_Parallel(TlSerializeData* pPdfParam)
: DfTaskCtrl(pPdfParam) {
const int input_numOfSessions = (*pPdfParam)["num_of_sessions"].getInt();
this->numOfSessions_ = (input_numOfSessions > 0) ? input_numOfSessions : 1;

if (this->isMasterSlave_) {
this->log_.info("task control: parallel, manager-worker");
} else {
this->log_.info("task control: parallel, divide & conquer");
}

this->log_.debug("DfTaskCtrl_Parallel::DfTaskCtrl_Parallel() called.");
}

DfTaskCtrl_Parallel::~DfTaskCtrl_Parallel() {}

void DfTaskCtrl_Parallel::prescreeningReport() {
TlCommunicate& rComm = TlCommunicate::getInstance();
rComm.allReduce_SUM(cutoffAll_density_);
rComm.allReduce_SUM(cutoffAlive_density_);
rComm.allReduce_SUM(cutoffAll_distribution_);
rComm.allReduce_SUM(cutoffAlive_distribution_);

if (rComm.isMaster() == true) {
DfTaskCtrl::prescreeningReport();
}
}

void DfTaskCtrl_Parallel::cutoffReport() {
if (this->isMasterSlave_ != true) {
TlCommunicate& rComm = TlCommunicate::getInstance();
rComm.allReduce_SUM(cutoffAll_schwarz_);
rComm.allReduce_SUM(cutoffAlive_schwarz_);

if (rComm.isMaster() == true) {
DfTaskCtrl::cutoffReport();
}
}
}

void DfTaskCtrl_Parallel::cutoffReport_MS() {
TlCommunicate& rComm = TlCommunicate::getInstance();
rComm.allReduce_SUM(cutoffAll_schwarz_);
rComm.allReduce_SUM(cutoffAlive_schwarz_);

if (rComm.isMaster() == true) {
DfTaskCtrl::cutoffReport();
}
}
bool DfTaskCtrl_Parallel::getQueue(const std::size_t maxIndeces,
const std::size_t maxGrainSize,
std::vector<std::size_t>* pTasks,
bool initialize) {
bool answer = false;

if (this->isMasterSlave_ == true) {
this->log_.info("sorry, DC only");
answer =
this->getQueue_DC(maxIndeces, maxGrainSize, pTasks, initialize);
} else {
answer =
this->getQueue_DC(maxIndeces, maxGrainSize, pTasks, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_DC(const std::size_t maxIndeces,
const std::size_t maxGrainSize,
std::vector<std::size_t>* pTasks,
bool initialize) {
assert(pTasks != NULL);

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<std::size_t> globalTasks;
bool answer = DfTaskCtrl::getQueue(maxIndeces, globalMaxGrainSize,
&globalTasks, initialize);

if (answer == true) {
const std::size_t grainSize = globalTasks.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTasks->resize(end - begin);
std::copy(globalTasks.begin() + begin, globalTasks.begin() + end,
pTasks->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue(const TlOrbitalInfoObject& orbitalInfo,
const int maxGrainSize,
std::vector<Task>* pTask, bool initialize) {
bool answer = false;

if (this->isMasterSlave_ == true) {
this->log_.info("sorry, DC only");
answer =
this->getQueue_DC(orbitalInfo, maxGrainSize, pTask, initialize);
} else {
answer =
this->getQueue_DC(orbitalInfo, maxGrainSize, pTask, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_DC(const TlOrbitalInfoObject& orbitalInfo,
const int maxGrainSize,
std::vector<Task>* pTask,
bool initialize) {
assert(pTask != NULL);
pTask->clear();

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<Task> globalTask;
bool answer = false;
answer = DfTaskCtrl::getQueue(orbitalInfo, globalMaxGrainSize, &globalTask,
initialize);
if (answer == true) {
const std::size_t grainSize = globalTask.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTask->resize(end - begin);
std::copy(globalTask.begin() + begin, globalTask.begin() + end,
pTask->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2(const TlOrbitalInfoObject& orbitalInfo,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTask,
bool initialize) {
bool answer = false;
if (this->isMasterSlave_ == true) {
answer = this->getQueue2_MS(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTask, initialize);
} else {
answer = this->getQueue2_DC(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTask, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_DC(const TlOrbitalInfoObject& orbitalInfo,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTask,
bool initialize) {
assert(pTask != NULL);
pTask->clear();

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<Task2> globalTask;
bool answer = false;
answer = DfTaskCtrl::getQueue2(orbitalInfo, isCutoffDistribution,
globalMaxGrainSize, &globalTask, initialize);
if (answer == true) {
const std::size_t grainSize = globalTask.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTask->resize(end - begin);
std::copy(globalTask.begin() + begin, globalTask.begin() + end,
pTask->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS(const TlOrbitalInfoObject& orbitalInfo,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTaskList,
bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();
if (rComm.isMaster() == true) {
answer = this->getQueue2_MS_master(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTaskList, initialize);
} else {
answer = this->getQueue2_MS_slave(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS_master(
const TlOrbitalInfoObject& orbitalInfo, const bool isCutoffDistribution,
const int maxGrainSize, std::vector<Task2>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

int rank = 0;

static int numOfFinishStateSlaves = 0;
if (initialize == true) {
numOfFinishStateSlaves = 0;

answer = DfTaskCtrl::getQueue2(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTaskList, true);
return answer;
}

static bool isRecvFinishMsg = false;
static int finishMsg = 0;
if (isRecvFinishMsg != true) {
rComm.iReceiveDataFromAnySource(finishMsg, DFTC_FINISH);
isRecvFinishMsg = true;
}
if ((isRecvFinishMsg == true) && (rComm.test(finishMsg, &rank) == true)) {
rComm.wait(finishMsg);
++numOfFinishStateSlaves;
isRecvFinishMsg = false;
}

static RequestListType requestList;
static bool isRecvSessionID = false;
static int sessionID = 0;
static const int maxContinuousSessionRequests = 5;  
for (int i = 0; i < maxContinuousSessionRequests; ++i) {
if (isRecvSessionID != true) {
rComm.iReceiveDataFromAnySource(sessionID, DFTC_SESSION_ID);
isRecvSessionID = true;
}

if ((isRecvSessionID == true) &&
(rComm.test(sessionID, &rank) == true)) {
rComm.wait(sessionID);
isRecvSessionID = false;

Request request;
request.rank = rank;
request.sessionID = sessionID;
request.state.reset();
requestList.push_back(request);
} else {
break;
}
}

RequestListType::iterator itEnd = requestList.end();
for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
std::vector<Task2> tmpTaskList;
const bool hasTask = DfTaskCtrl::getQueue2(
orbitalInfo, isCutoffDistribution, maxGrainSize, &tmpTaskList);
if (hasTask == true) {
const std::size_t numOfTasks = tmpTaskList.size();
const std::size_t unpackTaskListSize = numOfTasks * 2;
std::vector<index_type> unpackTaskList(unpackTaskListSize);
#pragma omp parallel for
for (std::size_t i = 0; i < numOfTasks; ++i) {
const Task2& task = tmpTaskList[i];
unpackTaskList[i * 2] = task.shellIndex1;
unpackTaskList[i * 2 + 1] = task.shellIndex2;
}
it->taskSize = unpackTaskList.size();
it->unpackTaskList = unpackTaskList;
it->state[Request::HAS_TASK] = true;
} else {
it->taskSize = 0;
it->state[Request::HAS_TASK] = false;
}

it->state[Request::TASK_ASSIGNED] = true;
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
continue;
}

if (it->state[Request::SEND_ANSWER] != true) {
const int rank = it->rank;
const int sessionID = it->sessionID;

if (it->state[Request::HAS_TASK] == true) {
rComm.iSendData(it->taskSize, rank, DFTC_TASK_SIZE + sessionID);
rComm.iSendDataX((index_type*)&(it->unpackTaskList[0]),
it->taskSize, rank, DFTC_TASK + sessionID);
} else {
it->finish = 9999;
rComm.iSendData(it->finish, rank, DFTC_FINISH + sessionID);
}

it->state[Request::SEND_ANSWER] = true;
} else if (it->state[Request::SESSION_END] != true) {
if (it->state[Request::HAS_TASK] == true) {
if ((it->state[Request::WAIT_TASK_SIZE] != true) &&
(rComm.test(it->taskSize) == true)) {
rComm.wait(it->taskSize);
it->state[Request::WAIT_TASK_SIZE] = true;
}
if ((it->state[Request::WAIT_TASK] != true) &&
(rComm.test(&(it->unpackTaskList[0])) == true)) {
rComm.wait(&(it->unpackTaskList[0]));
it->state[Request::WAIT_TASK] = true;
}

if ((it->state[Request::WAIT_TASK_SIZE] == true) &&
(it->state[Request::WAIT_TASK] == true)) {
it->state[Request::SESSION_END] = true;
}
} else {
if (rComm.test(it->finish) == true) {
rComm.wait(it->finish);
it->state[Request::SESSION_END] = true;
}
}
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;) {
if (it->state[Request::SESSION_END] == true) {
it = requestList.erase(it);
continue;
}
++it;
}

const int numOfSlaves = rComm.getNumOfProcs() - 1;
if (numOfFinishStateSlaves >= numOfSlaves) {
answer = false;
this->cutoffReport_MS();

if (isRecvFinishMsg == true) {
rComm.cancel(&finishMsg);
isRecvFinishMsg = false;
}
if (isRecvSessionID == true) {
rComm.cancel(&sessionID);
isRecvSessionID = false;
}

assert(requestList.empty() == true);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS_slave(
const TlOrbitalInfoObject& orbitalInfo, const bool isCutoffDistribution,
const int maxGrainSize, std::vector<Task2>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

const int numOfSessions = this->numOfSessions_;
static bool isAllSessionFinished = false;
static std::vector<Session> sessions(numOfSessions);
if (initialize == true) {
isAllSessionFinished = false;
for (int i = 0; i < numOfSessions; ++i) {
sessions[i].state.reset();
}

answer = DfTaskCtrl::getQueue2(orbitalInfo, isCutoffDistribution,
maxGrainSize, pTaskList, true);
return answer;
}

if (isAllSessionFinished == true) {
return false;
}

assert(pTaskList != NULL);
pTaskList->clear();
const int master = 0;

for (int i = 0; i < numOfSessions; ++i) {
Session& session = sessions[i];
if (session.state[Session::SESSION_END] == true) {
continue;
}

if (session.state[Session::RECV_FINISH] != true) {
rComm.iReceiveData(session.recvFinishMsg, master, DFTC_FINISH + i);
session.state[Session::RECV_FINISH] = true;
}
if ((session.state[Session::RECV_FINISH] == true) &&
(session.state[Session::SESSION_END] != true)) {
if (rComm.test(session.recvFinishMsg) == true) {
rComm.wait(session.recvFinishMsg);
session.state[Session::SESSION_END] = true;
}
}

if (session.state[Session::SEND_SESSION] != true) {
session.id = i;
rComm.iSendData(session.id, master, DFTC_SESSION_ID);
session.state[Session::SEND_SESSION] = true;
} else if (session.state[Session::WAIT_SESSION] != true) {
if (rComm.test(session.id) == true) {
rComm.wait(session.id);
sessions[i].state[Session::WAIT_SESSION] = true;
}
}

if (session.state[Session::RECV_SIZE] != true) {
rComm.iReceiveData(session.size, master, DFTC_TASK_SIZE + i);
session.state[Session::RECV_SIZE] = true;
} else if (session.state[Session::WAIT_SIZE] != true) {
if (rComm.test(session.size) == true) {
rComm.wait(session.size);
session.unpackTaskList.resize(session.size);
session.state[Session::WAIT_SIZE] = true;
}
}

if ((session.state[Session::WAIT_SIZE] == true) &&
(session.state[Session::RECV_TASK_LIST] != true)) {
session.unpackTaskList.resize(session.size);
rComm.iReceiveDataX(&(session.unpackTaskList[0]), session.size,
master, DFTC_TASK + i);
session.state[Session::RECV_TASK_LIST] = true;
}
if ((session.state[Session::RECV_TASK_LIST] == true) &&
(session.state[Session::WAIT_TASK_LIST] != true)) {
if (rComm.test(&(session.unpackTaskList[0])) == true) {
rComm.wait(&(session.unpackTaskList[0]));
session.state[Session::WAIT_TASK_LIST] = true;

const std::size_t taskListSize = session.size / 2;
pTaskList->resize(taskListSize);
Task2 task;
#pragma omp parallel for private(task)
for (std::size_t i = 0; i < taskListSize; ++i) {
task.shellIndex1 = session.unpackTaskList[i * 2];
task.shellIndex2 = session.unpackTaskList[i * 2 + 1];
(*pTaskList)[i] = task;
}

session.state[Session::SEND_SESSION] = false;
session.state[Session::WAIT_SESSION] = false;
session.state[Session::RECV_SIZE] = false;
session.state[Session::WAIT_SIZE] = false;
session.state[Session::RECV_TASK_LIST] = false;
session.state[Session::WAIT_TASK_LIST] = false;

return true;
}
}
}

{
bool allFinished = true;
for (int i = 0; i < numOfSessions; ++i) {
if (sessions[i].state[Session::SESSION_END] != true) {
allFinished = false;
break;
}
}
if (allFinished == true) {
const int sendExitMsg = 9999;
rComm.sendData(sendExitMsg, master, DFTC_FINISH);

for (int i = 0; i < numOfSessions; ++i) {
if ((sessions[i].state[Session::SEND_SESSION] == true) &&
(sessions[i].state[Session::WAIT_SESSION] != true)) {
rComm.cancel(sessions[i].id);
}
if ((sessions[i].state[Session::RECV_SIZE] == true) &&
(sessions[i].state[Session::WAIT_SIZE] != true)) {
rComm.cancel(sessions[i].size);
}
}

isAllSessionFinished = true;
answer = false;
this->cutoffReport_MS();
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2(const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTask,
bool initialize) {
bool answer = false;
if (this->isMasterSlave_ == true) {
answer =
this->getQueue2_MS(orbitalInfo1, orbitalInfo2, isCutoffDistribution,
maxGrainSize, pTask, initialize);
} else {
answer =
this->getQueue2_DC(orbitalInfo1, orbitalInfo2, isCutoffDistribution,
maxGrainSize, pTask, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_DC(const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTask,
bool initialize) {
assert(pTask != NULL);
pTask->clear();

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<Task2> globalTask;
bool answer = false;
answer =
DfTaskCtrl::getQueue2(orbitalInfo1, orbitalInfo2, isCutoffDistribution,
globalMaxGrainSize, &globalTask, initialize);
if (answer == true) {
const std::size_t grainSize = globalTask.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTask->resize(end - begin);
std::copy(globalTask.begin() + begin, globalTask.begin() + end,
pTask->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS(const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
const bool isCutoffDistribution,
const int maxGrainSize,
std::vector<Task2>* pTaskList,
bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();
if (rComm.isMaster() == true) {
answer = this->getQueue2_MS_master(orbitalInfo1, orbitalInfo2,
isCutoffDistribution, maxGrainSize,
pTaskList, initialize);
} else {
answer = this->getQueue2_MS_slave(orbitalInfo1, orbitalInfo2,
isCutoffDistribution, maxGrainSize,
pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS_master(
const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2, const bool isCutoffDistribution,
const int maxGrainSize, std::vector<Task2>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

int rank = 0;

static int numOfFinishStateSlaves = 0;
if (initialize == true) {
numOfFinishStateSlaves = 0;

answer = DfTaskCtrl::getQueue2(orbitalInfo1, orbitalInfo2,
isCutoffDistribution, maxGrainSize,
pTaskList, true);
return answer;
}

static bool isRecvFinishMsg = false;
static int finishMsg = 0;
if (isRecvFinishMsg != true) {
rComm.iReceiveDataFromAnySource(finishMsg, DFTC_FINISH);
isRecvFinishMsg = true;
}
if ((isRecvFinishMsg == true) && (rComm.test(finishMsg, &rank) == true)) {
rComm.wait(finishMsg);
++numOfFinishStateSlaves;
isRecvFinishMsg = false;
}

static RequestListType requestList;
static bool isRecvSessionID = false;
static int sessionID = 0;
static const int maxContinuousSessionRequests = 5;  
for (int i = 0; i < maxContinuousSessionRequests; ++i) {
if (isRecvSessionID != true) {
rComm.iReceiveDataFromAnySource(sessionID, DFTC_SESSION_ID);
isRecvSessionID = true;
}

if ((isRecvSessionID == true) &&
(rComm.test(sessionID, &rank) == true)) {
rComm.wait(sessionID);
isRecvSessionID = false;

Request request;
request.rank = rank;
request.sessionID = sessionID;
request.state.reset();
requestList.push_back(request);
} else {
break;
}
}

RequestListType::iterator itEnd = requestList.end();
for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
std::vector<Task2> tmpTaskList;
const bool hasTask = DfTaskCtrl::getQueue2(
orbitalInfo1, orbitalInfo2, isCutoffDistribution, maxGrainSize,
&tmpTaskList);
if (hasTask == true) {
const std::size_t numOfTasks = tmpTaskList.size();
const std::size_t unpackTaskListSize = numOfTasks * 2;
std::vector<index_type> unpackTaskList(unpackTaskListSize);
#pragma omp parallel for
for (std::size_t i = 0; i < numOfTasks; ++i) {
const Task2& task = tmpTaskList[i];
unpackTaskList[i * 2] = task.shellIndex1;
unpackTaskList[i * 2 + 1] = task.shellIndex2;
}
it->taskSize = unpackTaskList.size();
it->unpackTaskList = unpackTaskList;
it->state[Request::HAS_TASK] = true;
} else {
it->taskSize = 0;
it->state[Request::HAS_TASK] = false;
}

it->state[Request::TASK_ASSIGNED] = true;
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
continue;
}

if (it->state[Request::SEND_ANSWER] != true) {
const int rank = it->rank;
const int sessionID = it->sessionID;

if (it->state[Request::HAS_TASK] == true) {
rComm.iSendData(it->taskSize, rank, DFTC_TASK_SIZE + sessionID);
rComm.iSendDataX((index_type*)&(it->unpackTaskList[0]),
it->taskSize, rank, DFTC_TASK + sessionID);
} else {
it->finish = 9999;
rComm.iSendData(it->finish, rank, DFTC_FINISH + sessionID);
}

it->state[Request::SEND_ANSWER] = true;
} else if (it->state[Request::SESSION_END] != true) {
if (it->state[Request::HAS_TASK] == true) {
if ((it->state[Request::WAIT_TASK_SIZE] != true) &&
(rComm.test(it->taskSize) == true)) {
rComm.wait(it->taskSize);
it->state[Request::WAIT_TASK_SIZE] = true;
}
if ((it->state[Request::WAIT_TASK] != true) &&
(rComm.test(&(it->unpackTaskList[0])) == true)) {
rComm.wait(&(it->unpackTaskList[0]));
it->state[Request::WAIT_TASK] = true;
}

if ((it->state[Request::WAIT_TASK_SIZE] == true) &&
(it->state[Request::WAIT_TASK] == true)) {
it->state[Request::SESSION_END] = true;
}
} else {
if (rComm.test(it->finish) == true) {
rComm.wait(it->finish);
it->state[Request::SESSION_END] = true;
}
}
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;) {
if (it->state[Request::SESSION_END] == true) {
it = requestList.erase(it);
continue;
}
++it;
}

const int numOfSlaves = rComm.getNumOfProcs() - 1;
if (numOfFinishStateSlaves >= numOfSlaves) {
answer = false;
this->cutoffReport_MS();

if (isRecvFinishMsg == true) {
rComm.cancel(&finishMsg);
isRecvFinishMsg = false;
}
if (isRecvSessionID == true) {
rComm.cancel(&sessionID);
isRecvSessionID = false;
}

assert(requestList.empty() == true);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue2_MS_slave(
const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2, const bool isCutoffDistribution,
const int maxGrainSize, std::vector<Task2>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

const int numOfSessions = this->numOfSessions_;
static bool isAllSessionFinished = false;
static std::vector<Session> sessions(numOfSessions);
if (initialize == true) {
isAllSessionFinished = false;
for (int i = 0; i < numOfSessions; ++i) {
sessions[i].state.reset();
}

answer = DfTaskCtrl::getQueue2(orbitalInfo1, orbitalInfo2,
isCutoffDistribution, maxGrainSize,
pTaskList, true);
return answer;
}

if (isAllSessionFinished == true) {
return false;
}

assert(pTaskList != NULL);
pTaskList->clear();
const int master = 0;

for (int i = 0; i < numOfSessions; ++i) {
Session& session = sessions[i];
if (session.state[Session::SESSION_END] == true) {
continue;
}

if (session.state[Session::RECV_FINISH] != true) {
rComm.iReceiveData(session.recvFinishMsg, master, DFTC_FINISH + i);
session.state[Session::RECV_FINISH] = true;
}
if ((session.state[Session::RECV_FINISH] == true) &&
(session.state[Session::SESSION_END] != true)) {
if (rComm.test(session.recvFinishMsg) == true) {
rComm.wait(session.recvFinishMsg);
session.state[Session::SESSION_END] = true;
}
}

if (session.state[Session::SEND_SESSION] != true) {
session.id = i;
rComm.iSendData(session.id, master, DFTC_SESSION_ID);
session.state[Session::SEND_SESSION] = true;
} else if (session.state[Session::WAIT_SESSION] != true) {
if (rComm.test(session.id) == true) {
rComm.wait(session.id);
sessions[i].state[Session::WAIT_SESSION] = true;
}
}

if (session.state[Session::RECV_SIZE] != true) {
rComm.iReceiveData(session.size, master, DFTC_TASK_SIZE + i);
session.state[Session::RECV_SIZE] = true;
} else if (session.state[Session::WAIT_SIZE] != true) {
if (rComm.test(session.size) == true) {
rComm.wait(session.size);
session.unpackTaskList.resize(session.size);
session.state[Session::WAIT_SIZE] = true;
}
}

if ((session.state[Session::WAIT_SIZE] == true) &&
(session.state[Session::RECV_TASK_LIST] != true)) {
session.unpackTaskList.resize(session.size);
rComm.iReceiveDataX(&(session.unpackTaskList[0]), session.size,
master, DFTC_TASK + i);
session.state[Session::RECV_TASK_LIST] = true;
}
if ((session.state[Session::RECV_TASK_LIST] == true) &&
(session.state[Session::WAIT_TASK_LIST] != true)) {
if (rComm.test(&(session.unpackTaskList[0])) == true) {
rComm.wait(&(session.unpackTaskList[0]));
session.state[Session::WAIT_TASK_LIST] = true;

const std::size_t taskListSize = session.size / 2;
pTaskList->resize(taskListSize);
Task2 task;
#pragma omp parallel for private(task)
for (std::size_t i = 0; i < taskListSize; ++i) {
task.shellIndex1 = session.unpackTaskList[i * 2];
task.shellIndex2 = session.unpackTaskList[i * 2 + 1];
(*pTaskList)[i] = task;
}

session.state[Session::SEND_SESSION] = false;
session.state[Session::WAIT_SESSION] = false;
session.state[Session::RECV_SIZE] = false;
session.state[Session::WAIT_SIZE] = false;
session.state[Session::RECV_TASK_LIST] = false;
session.state[Session::WAIT_TASK_LIST] = false;

return true;
}
}
}

{
bool allFinished = true;
for (int i = 0; i < numOfSessions; ++i) {
if (sessions[i].state[Session::SESSION_END] != true) {
allFinished = false;
break;
}
}
if (allFinished == true) {
const int sendExitMsg = 9999;
rComm.sendData(sendExitMsg, master, DFTC_FINISH);

for (int i = 0; i < numOfSessions; ++i) {
if ((sessions[i].state[Session::SEND_SESSION] == true) &&
(sessions[i].state[Session::WAIT_SESSION] != true)) {
rComm.cancel(sessions[i].id);
}
if ((sessions[i].state[Session::RECV_SIZE] == true) &&
(sessions[i].state[Session::WAIT_SIZE] != true)) {
rComm.cancel(sessions[i].size);
}
}

isAllSessionFinished = true;
answer = false;
this->cutoffReport_MS();
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue4(const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable,
const int maxGrainSize,
std::vector<Task4>* pTaskList,
bool initialize) {
bool answer = false;
if (this->isMasterSlave_ == true) {
answer = this->getQueue4_MS(orbitalInfo, schwarzTable, maxGrainSize,
pTaskList, initialize);
} else {
answer = this->getQueue4_DC(orbitalInfo, schwarzTable, maxGrainSize,
pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue4_DC(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTask, bool initialize) {
TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<Task4> globalTask;
bool answer = false;
answer = DfTaskCtrl::getQueue4(orbitalInfo, schwarzTable,
globalMaxGrainSize, &globalTask, initialize);
if (answer == true) {
const std::size_t grainSize = globalTask.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTask->resize(end - begin);
std::copy(globalTask.begin() + begin, globalTask.begin() + end,
pTask->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue4_MS(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();
if (rComm.isMaster() == true) {
answer = this->getQueue4_MS_master(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, initialize);
} else {
answer = this->getQueue4_MS_slave(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue4_MS_master(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

int rank = 0;

static int numOfFinishStateSlaves = 0;
if (initialize == true) {
numOfFinishStateSlaves = 0;

answer = DfTaskCtrl::getQueue4(orbitalInfo, schwarzTable, maxGrainSize,
pTaskList, true);
return answer;
}

static bool isRecvFinishMsg = false;
static int finishMsg = 0;
if (isRecvFinishMsg != true) {
rComm.iReceiveDataFromAnySource(finishMsg, DFTC_FINISH);
isRecvFinishMsg = true;
}
if ((isRecvFinishMsg == true) && (rComm.test(finishMsg, &rank) == true)) {
rComm.wait(finishMsg);
++numOfFinishStateSlaves;
isRecvFinishMsg = false;
}

static RequestListType requestList;
static bool isRecvSessionID = false;
static int sessionID = 0;
static const int maxContinuousSessionRequests = 5;  
for (int i = 0; i < maxContinuousSessionRequests; ++i) {
if (isRecvSessionID != true) {
rComm.iReceiveDataFromAnySource(sessionID, DFTC_SESSION_ID);
isRecvSessionID = true;
}

if ((isRecvSessionID == true) &&
(rComm.test(sessionID, &rank) == true)) {
rComm.wait(sessionID);
isRecvSessionID = false;

Request request;
request.rank = rank;
request.sessionID = sessionID;
request.state.reset();
requestList.push_back(request);
} else {
break;
}
}

RequestListType::iterator itEnd = requestList.end();
for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
std::vector<Task4> tmpTaskList;
const bool hasTask = DfTaskCtrl::getQueue4(
orbitalInfo, schwarzTable, maxGrainSize, &tmpTaskList);
if (hasTask == true) {
const std::size_t numOfTasks = tmpTaskList.size();
const std::size_t unpackTaskListSize = numOfTasks * 4;
std::vector<index_type> unpackTaskList(unpackTaskListSize);
#pragma omp parallel for
for (std::size_t i = 0; i < numOfTasks; ++i) {
const Task4& task = tmpTaskList[i];
unpackTaskList[i * 4] = task.shellIndex1;
unpackTaskList[i * 4 + 1] = task.shellIndex2;
unpackTaskList[i * 4 + 2] = task.shellIndex3;
unpackTaskList[i * 4 + 3] = task.shellIndex4;
}
it->taskSize = unpackTaskList.size();
it->unpackTaskList = unpackTaskList;
it->state[Request::HAS_TASK] = true;
} else {
it->taskSize = 0;
it->state[Request::HAS_TASK] = false;
}

it->state[Request::TASK_ASSIGNED] = true;
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
continue;
}

if (it->state[Request::SEND_ANSWER] != true) {
const int rank = it->rank;
const int sessionID = it->sessionID;

if (it->state[Request::HAS_TASK] == true) {
rComm.iSendData(it->taskSize, rank, DFTC_TASK_SIZE + sessionID);
rComm.iSendDataX((index_type*)&(it->unpackTaskList[0]),
it->taskSize, rank, DFTC_TASK + sessionID);
} else {
it->finish = 9999;
rComm.iSendData(it->finish, rank, DFTC_FINISH + sessionID);
}

it->state[Request::SEND_ANSWER] = true;
} else if (it->state[Request::SESSION_END] != true) {
if (it->state[Request::HAS_TASK] == true) {
if ((it->state[Request::WAIT_TASK_SIZE] != true) &&
(rComm.test(it->taskSize) == true)) {
rComm.wait(it->taskSize);
it->state[Request::WAIT_TASK_SIZE] = true;
}
if ((it->state[Request::WAIT_TASK] != true) &&
(rComm.test(&(it->unpackTaskList[0])) == true)) {
rComm.wait(&(it->unpackTaskList[0]));
it->state[Request::WAIT_TASK] = true;
}

if ((it->state[Request::WAIT_TASK_SIZE] == true) &&
(it->state[Request::WAIT_TASK] == true)) {
it->state[Request::SESSION_END] = true;
}
} else {
if (rComm.test(it->finish) == true) {
rComm.wait(it->finish);
it->state[Request::SESSION_END] = true;
}
}
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;) {
if (it->state[Request::SESSION_END] == true) {
it = requestList.erase(it);
continue;
}
++it;
}

const int numOfSlaves = rComm.getNumOfProcs() - 1;
if (numOfFinishStateSlaves >= numOfSlaves) {
answer = false;
this->cutoffReport_MS();

if (isRecvFinishMsg == true) {
rComm.cancel(&finishMsg);
isRecvFinishMsg = false;
}
if (isRecvSessionID == true) {
rComm.cancel(&sessionID);
isRecvSessionID = false;
}

assert(requestList.empty() == true);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue4_MS_slave(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

const int numOfSessions = this->numOfSessions_;
static bool isAllSessionFinished = false;
static std::vector<Session> sessions(numOfSessions);
if (initialize == true) {
isAllSessionFinished = false;
for (int i = 0; i < numOfSessions; ++i) {
sessions[i].state.reset();
}

answer = DfTaskCtrl::getQueue4(orbitalInfo, schwarzTable, maxGrainSize,
pTaskList, true);
return answer;
}

if (isAllSessionFinished == true) {
return false;
}

assert(pTaskList != NULL);
pTaskList->clear();
const int master = 0;

for (int i = 0; i < numOfSessions; ++i) {
Session& session = sessions[i];
if (session.state[Session::SESSION_END] == true) {
continue;
}

if (session.state[Session::RECV_FINISH] != true) {
rComm.iReceiveData(session.recvFinishMsg, master, DFTC_FINISH + i);
session.state[Session::RECV_FINISH] = true;
}
if ((session.state[Session::RECV_FINISH] == true) &&
(session.state[Session::SESSION_END] != true)) {
if (rComm.test(session.recvFinishMsg) == true) {
rComm.wait(session.recvFinishMsg);
session.state[Session::SESSION_END] = true;
}
}

if (session.state[Session::SEND_SESSION] != true) {
session.id = i;
rComm.iSendData(session.id, master, DFTC_SESSION_ID);
session.state[Session::SEND_SESSION] = true;
} else if (session.state[Session::WAIT_SESSION] != true) {
if (rComm.test(session.id) == true) {
rComm.wait(session.id);
sessions[i].state[Session::WAIT_SESSION] = true;
}
}

if (session.state[Session::RECV_SIZE] != true) {
rComm.iReceiveData(session.size, master, DFTC_TASK_SIZE + i);
session.state[Session::RECV_SIZE] = true;
} else if (session.state[Session::WAIT_SIZE] != true) {
if (rComm.test(session.size) == true) {
rComm.wait(session.size);
session.unpackTaskList.resize(session.size);
session.state[Session::WAIT_SIZE] = true;
}
}

if ((session.state[Session::WAIT_SIZE] == true) &&
(session.state[Session::RECV_TASK_LIST] != true)) {
session.unpackTaskList.resize(session.size);
rComm.iReceiveDataX(&(session.unpackTaskList[0]), session.size,
master, DFTC_TASK + i);
session.state[Session::RECV_TASK_LIST] = true;
}
if ((session.state[Session::RECV_TASK_LIST] == true) &&
(session.state[Session::WAIT_TASK_LIST] != true)) {
if (rComm.test(&(session.unpackTaskList[0])) == true) {
rComm.wait(&(session.unpackTaskList[0]));
session.state[Session::WAIT_TASK_LIST] = true;

const std::size_t taskListSize = session.size / 4;
pTaskList->resize(taskListSize);
Task4 task;
#pragma omp parallel for private(task)
for (std::size_t i = 0; i < taskListSize; ++i) {
task.shellIndex1 = session.unpackTaskList[i * 4];
task.shellIndex2 = session.unpackTaskList[i * 4 + 1];
task.shellIndex3 = session.unpackTaskList[i * 4 + 2];
task.shellIndex4 = session.unpackTaskList[i * 4 + 3];
(*pTaskList)[i] = task;
}

session.state[Session::SEND_SESSION] = false;
session.state[Session::WAIT_SESSION] = false;
session.state[Session::RECV_SIZE] = false;
session.state[Session::WAIT_SIZE] = false;
session.state[Session::RECV_TASK_LIST] = false;
session.state[Session::WAIT_TASK_LIST] = false;

return true;
}
}
}

{
bool allFinished = true;
for (int i = 0; i < numOfSessions; ++i) {
if (sessions[i].state[Session::SESSION_END] != true) {
allFinished = false;
break;
}
}
if (allFinished == true) {
const int sendExitMsg = 9999;
rComm.sendData(sendExitMsg, master, DFTC_FINISH);

for (int i = 0; i < numOfSessions; ++i) {
if ((sessions[i].state[Session::SEND_SESSION] == true) &&
(sessions[i].state[Session::WAIT_SESSION] != true)) {
rComm.cancel(sessions[i].id);
}
if ((sessions[i].state[Session::RECV_SIZE] == true) &&
(sessions[i].state[Session::WAIT_SIZE] != true)) {
rComm.cancel(sessions[i].size);
}
}

isAllSessionFinished = true;
answer = false;
this->cutoffReport_MS();
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_Force4(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = false;
if (this->isMasterSlave_ == true) {
answer = this->getQueue_Force4_MS(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, initialize);
} else {
answer = this->getQueue_Force4_DC(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_Force4_DC(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTask, bool initialize) {
assert(pTask != NULL);
pTask->clear();

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProcs();
const int globalMaxGrainSize = maxGrainSize * numOfProcs;

std::vector<Task4> globalTask;
bool answer = false;
answer = DfTaskCtrl::getQueue_Force4(
orbitalInfo, schwarzTable, globalMaxGrainSize, &globalTask, initialize);
if (answer == true) {
const std::size_t grainSize = globalTask.size();
const std::size_t localGrainSize =
(grainSize + numOfProcs - 1) / numOfProcs;

const int rank = rComm.getRank();
const std::size_t begin = localGrainSize * rank;
const std::size_t end =
std::min(localGrainSize * (rank + 1), grainSize);

if (begin < end) {
pTask->resize(end - begin);
std::copy(globalTask.begin() + begin, globalTask.begin() + end,
pTask->begin());
}
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_Force4_MS(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();
if (rComm.isMaster() == true) {
answer = this->getQueue_Force4_MS_master(
orbitalInfo, schwarzTable, maxGrainSize, pTaskList, initialize);
} else {
answer = this->getQueue_Force4_MS_slave(
orbitalInfo, schwarzTable, maxGrainSize, pTaskList, initialize);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_Force4_MS_master(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
TlCommunicate& rComm = TlCommunicate::getInstance();

int rank = 0;

static int numOfFinishStateSlaves = 0;
if (initialize == true) {
numOfFinishStateSlaves = 0;

answer = DfTaskCtrl::getQueue_Force4(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, true);
return answer;
}

static bool isRecvFinishMsg = false;
static int finishMsg = 0;
if (isRecvFinishMsg != true) {
rComm.iReceiveDataFromAnySource(finishMsg, DFTC_FINISH);
isRecvFinishMsg = true;
}
if ((isRecvFinishMsg == true) && (rComm.test(finishMsg, &rank) == true)) {
rComm.wait(finishMsg);
++numOfFinishStateSlaves;
isRecvFinishMsg = false;
}

static RequestListType requestList;
static bool isRecvSessionID = false;
static int sessionID = 0;
static const int maxContinuousSessionRequests = 5;  
for (int i = 0; i < maxContinuousSessionRequests; ++i) {
if (isRecvSessionID != true) {
rComm.iReceiveDataFromAnySource(sessionID, DFTC_SESSION_ID);
isRecvSessionID = true;
}

if ((isRecvSessionID == true) &&
(rComm.test(sessionID, &rank) == true)) {
rComm.wait(sessionID);
isRecvSessionID = false;

Request request;
request.rank = rank;
request.sessionID = sessionID;
request.state.reset();
requestList.push_back(request);
} else {
break;
}
}

RequestListType::iterator itEnd = requestList.end();
for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
std::vector<Task4> tmpTaskList;
const bool hasTask = DfTaskCtrl::getQueue_Force4(
orbitalInfo, schwarzTable, maxGrainSize, &tmpTaskList);
if (hasTask == true) {
const std::size_t numOfTasks = tmpTaskList.size();
const std::size_t unpackTaskListSize = numOfTasks * 4;
std::vector<index_type> unpackTaskList(unpackTaskListSize);
#pragma omp parallel for
for (std::size_t i = 0; i < numOfTasks; ++i) {
const Task4& task = tmpTaskList[i];
unpackTaskList[i * 4] = task.shellIndex1;
unpackTaskList[i * 4 + 1] = task.shellIndex2;
unpackTaskList[i * 4 + 2] = task.shellIndex3;
unpackTaskList[i * 4 + 3] = task.shellIndex4;
}
it->taskSize = unpackTaskList.size();
it->unpackTaskList = unpackTaskList;
it->state[Request::HAS_TASK] = true;
} else {
it->taskSize = 0;
it->state[Request::HAS_TASK] = false;
}

it->state[Request::TASK_ASSIGNED] = true;
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;
++it) {
if (it->state[Request::TASK_ASSIGNED] != true) {
continue;
}

if (it->state[Request::SEND_ANSWER] != true) {
const int rank = it->rank;
const int sessionID = it->sessionID;

if (it->state[Request::HAS_TASK] == true) {
rComm.iSendData(it->taskSize, rank, DFTC_TASK_SIZE + sessionID);
rComm.iSendDataX((index_type*)&(it->unpackTaskList[0]),
it->taskSize, rank, DFTC_TASK + sessionID);
} else {
it->finish = 9999;
rComm.iSendData(it->finish, rank, DFTC_FINISH + sessionID);
}

it->state[Request::SEND_ANSWER] = true;
} else if (it->state[Request::SESSION_END] != true) {
if (it->state[Request::HAS_TASK] == true) {
if ((it->state[Request::WAIT_TASK_SIZE] != true) &&
(rComm.test(it->taskSize) == true)) {
rComm.wait(it->taskSize);
it->state[Request::WAIT_TASK_SIZE] = true;
}
if ((it->state[Request::WAIT_TASK] != true) &&
(rComm.test(&(it->unpackTaskList[0])) == true)) {
rComm.wait(&(it->unpackTaskList[0]));
it->state[Request::WAIT_TASK] = true;
}

if ((it->state[Request::WAIT_TASK_SIZE] == true) &&
(it->state[Request::WAIT_TASK] == true)) {
it->state[Request::SESSION_END] = true;
}
} else {
if (rComm.test(it->finish) == true) {
rComm.wait(it->finish);
it->state[Request::SESSION_END] = true;
}
}
}
}

for (RequestListType::iterator it = requestList.begin(); it != itEnd;) {
if (it->state[Request::SESSION_END] == true) {
it = requestList.erase(it);
continue;
}
++it;
}

const int numOfSlaves = rComm.getNumOfProcs() - 1;
if (numOfFinishStateSlaves >= numOfSlaves) {
answer = false;

if (isRecvFinishMsg == true) {
rComm.cancel(&finishMsg);
isRecvFinishMsg = false;
}
if (isRecvSessionID == true) {
rComm.cancel(&sessionID);
isRecvSessionID = false;
}

assert(requestList.empty() == true);
}

return answer;
}

bool DfTaskCtrl_Parallel::getQueue_Force4_MS_slave(
const TlOrbitalInfoObject& orbitalInfo,
const TlSparseSymmetricMatrix& schwarzTable, const int maxGrainSize,
std::vector<Task4>* pTaskList, bool initialize) {
bool answer = true;
pTaskList->clear();
TlCommunicate& rComm = TlCommunicate::getInstance();

const int numOfSessions = this->numOfSessions_;
static bool isAllSessionFinished = false;
static std::vector<Session> sessions(numOfSessions);
if (initialize == true) {
isAllSessionFinished = false;
for (int i = 0; i < numOfSessions; ++i) {
sessions[i].state.reset();
}

answer = DfTaskCtrl::getQueue_Force4(orbitalInfo, schwarzTable,
maxGrainSize, pTaskList, true);
return answer;
}

if (isAllSessionFinished == true) {
return false;
}

const int master = 0;

for (int i = 0; i < numOfSessions; ++i) {
Session& session = sessions[i];
if (session.state[Session::SESSION_END] == true) {
continue;
}

if (session.state[Session::RECV_FINISH] != true) {
rComm.iReceiveData(session.recvFinishMsg, master, DFTC_FINISH + i);
session.state[Session::RECV_FINISH] = true;
}
if ((session.state[Session::RECV_FINISH] == true) &&
(session.state[Session::SESSION_END] != true)) {
if (rComm.test(session.recvFinishMsg) == true) {
rComm.wait(session.recvFinishMsg);
session.state[Session::SESSION_END] = true;
}
}

if (session.state[Session::SEND_SESSION] != true) {
session.id = i;
rComm.iSendData(session.id, master, DFTC_SESSION_ID);
session.state[Session::SEND_SESSION] = true;
} else if (session.state[Session::WAIT_SESSION] != true) {
if (rComm.test(session.id) == true) {
rComm.wait(session.id);
sessions[i].state[Session::WAIT_SESSION] = true;
}
}

if (session.state[Session::RECV_SIZE] != true) {
rComm.iReceiveData(session.size, master, DFTC_TASK_SIZE + i);
session.state[Session::RECV_SIZE] = true;
} else if (session.state[Session::WAIT_SIZE] != true) {
if (rComm.test(session.size) == true) {
rComm.wait(session.size);
session.unpackTaskList.resize(session.size);
session.state[Session::WAIT_SIZE] = true;
}
}

if ((session.state[Session::WAIT_SIZE] == true) &&
(session.state[Session::RECV_TASK_LIST] != true)) {
session.unpackTaskList.resize(session.size);
rComm.iReceiveDataX(&(session.unpackTaskList[0]), session.size,
master, DFTC_TASK + i);
session.state[Session::RECV_TASK_LIST] = true;
}
if ((session.state[Session::RECV_TASK_LIST] == true) &&
(session.state[Session::WAIT_TASK_LIST] != true)) {
if (rComm.test(&(session.unpackTaskList[0])) == true) {
rComm.wait(&(session.unpackTaskList[0]));
session.state[Session::WAIT_TASK_LIST] = true;

const std::size_t taskListSize = session.size / 4;
pTaskList->resize(taskListSize);
Task4 task;
#pragma omp parallel for private(task)
for (std::size_t i = 0; i < taskListSize; ++i) {
task.shellIndex1 = session.unpackTaskList[i * 4];
task.shellIndex2 = session.unpackTaskList[i * 4 + 1];
task.shellIndex3 = session.unpackTaskList[i * 4 + 2];
task.shellIndex4 = session.unpackTaskList[i * 4 + 3];
(*pTaskList)[i] = task;
}

session.state[Session::SEND_SESSION] = false;
session.state[Session::WAIT_SESSION] = false;
session.state[Session::RECV_SIZE] = false;
session.state[Session::WAIT_SIZE] = false;
session.state[Session::RECV_TASK_LIST] = false;
session.state[Session::WAIT_TASK_LIST] = false;

return true;
}
}
}

{
bool allFinished = true;
for (int i = 0; i < numOfSessions; ++i) {
if (sessions[i].state[Session::SESSION_END] != true) {
allFinished = false;
break;
}
}
if (allFinished == true) {
const int sendExitMsg = 9999;
rComm.sendData(sendExitMsg, master, DFTC_FINISH);

for (int i = 0; i < numOfSessions; ++i) {
if ((sessions[i].state[Session::SEND_SESSION] == true) &&
(sessions[i].state[Session::WAIT_SESSION] != true)) {
rComm.cancel(sessions[i].id);
}
if ((sessions[i].state[Session::RECV_SIZE] == true) &&
(sessions[i].state[Session::WAIT_SIZE] != true)) {
rComm.cancel(sessions[i].size);
}
}

isAllSessionFinished = true;
answer = false;
}
}

return answer;
}
