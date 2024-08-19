
#include "TlCommunicate.h"

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <typeinfo>

#include "TlMsgPack.h"
#include "TlSerializeData.h"
#include "tl_dense_general_matrix_lapack.h"
#include "tl_dense_symmetric_matrix_lapack.h"
#include "tl_dense_vector_impl_lapack.h"
#include "tl_dense_vector_lapack.h"

#ifdef HAVE_EIGEN
#include "tl_dense_general_matrix_eigen.h"
#include "tl_dense_symmetric_matrix_eigen.h"
#include "tl_dense_vector_eigen.h"
#endif  

#include "tl_dense_matrix_io_object.h"
#include "tl_dense_symmetric_matrix_io.h"
#include "tl_matrix_object.h"
#include "tl_scalapack_context.h"
#include "tl_sparse_matrix.h"
#include "tl_vector_object.h"

#define DEFAULT_WORK_MEM_SIZE (400UL * 1024UL * 1024UL)


TlCommunicate* TlCommunicate::m_pTlCommunicateInstance = NULL;
TlCommunicate::NonBlockingCommParamTableType
TlCommunicate::nonBlockingCommParamTable_;

TlCommunicate& TlCommunicate::getInstance(int argc, char* argv[]) {
if (TlCommunicate::m_pTlCommunicateInstance == NULL) {
TlCommunicate::m_pTlCommunicateInstance = new TlCommunicate();

TlCommunicate::m_pTlCommunicateInstance->initialize(argc, argv);
}

return *(TlCommunicate::m_pTlCommunicateInstance);
}

TlCommunicate& TlCommunicate::getInstance() {
assert(TlCommunicate::m_pTlCommunicateInstance != NULL);

return *(TlCommunicate::m_pTlCommunicateInstance);
}

void TlCommunicate::setWorkMemSize(const std::size_t workMemSize) {
this->workMemSize_ =
std::max<std::size_t>(workMemSize, DEFAULT_WORK_MEM_SIZE);
}

std::size_t TlCommunicate::getWorkMemSize() const {
return this->workMemSize_;
}

std::string TlCommunicate::getReport() const {
std::string answer = TlUtils::format(" #%d:\n", this->getRank());
answer += TlUtils::format(" barrier:     %16.2f sec. %16ld times\n",
this->time_barrier_.getElapseTime(),
this->counter_barrier_);
answer +=
TlUtils::format(" test:        %16.2f sec. %16ld times\n",
this->time_test_.getElapseTime(), this->counter_test_);
answer +=
TlUtils::format(" wait:        %16.2f sec. %16ld times\n",
this->time_wait_.getElapseTime(), this->counter_wait_);
answer += TlUtils::format(" all_reduce:  %16.2f sec. %16ld times\n",
this->time_allreduce_.getElapseTime(),
this->counter_allreduce_);
answer += TlUtils::format(" iall_reduce: %16.2f sec. %16ld times\n",
this->time_iallreduce_.getElapseTime(),
this->counter_iallreduce_);

return answer;
}

std::string TlCommunicate::getHostName() const {
int name_length = 0;
char* name = new char[MPI_MAX_PROCESSOR_NAME];

MPI_Get_processor_name(name, &name_length);
return std::string(name);
}

int TlCommunicate::barrier(bool isDebugOut) const {
this->time_barrier_.start();
++(this->counter_barrier_);

if (isDebugOut == true) {
this->log_.debug(TlUtils::format("barrier called. times=%ld",
this->counter_barrier_));
}

const int answer = MPI_Barrier(MPI_COMM_WORLD);

this->time_barrier_.stop();
return answer;
}

double TlCommunicate::getTime() {
this->barrier();
return MPI_Wtime();
}

bool TlCommunicate::checkNonBlockingTableCollision(
uintptr_t key, const NonBlockingCommParam& param, int line) const {
bool answer = true;

#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{
if (this->nonBlockingCommParamTable_.find(key) !=
this->nonBlockingCommParamTable_.end()) {
const char isSendRecv =
((param.property & NonBlockingCommParam::SEND) != 0) ? 'S'
: 'R';
this->log_.critical(TlUtils::format(
"[%5d/%5d WARN] non-blocking table collision(%c) "
"found in TlCommunicate: tag=%d, line=%d",
this->getRank(), this->getNumOfProcs() - 1, isSendRecv,
param.tag, line));
answer = false;
}
}

return answer;
}

bool TlCommunicate::checkNonBlockingCommunications() const {
bool answer = true;
TlLogging& log = TlLogging::getInstance();

#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{
if (this->nonBlockingCommParamTable_.empty() == false) {
answer = false;
NonBlockingCommParamTableType::const_iterator itEnd =
this->nonBlockingCommParamTable_.end();
for (NonBlockingCommParamTableType::const_iterator it =
this->nonBlockingCommParamTable_.begin();
it != itEnd; ++it) {
const char isSendRecv =
((it->second.property & NonBlockingCommParam::SEND) != 0)
? 'S'
: 'R';
log.warn(TlUtils::format(
"[%5d/%5d WARN] rest waiting communication(%c) in "
"TlCommunicate: TAG=%d",
this->getRank(), this->getNumOfProcs() - 1, isSendRecv,
it->second.tag));
}
}
}

return answer;
}

template <typename T>
int TlCommunicate::reduce(T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const MPI_Op mpiOp, const int root) {
int answer = 0;

#ifdef DIV_COMM
{
const long length = end - start;
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);
T* pBuf = new T[bufCount];

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Reduce(pData + startIndex, pBuf, bufCount, mpiType,
mpiOp, root, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + bufCount, pData + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

if (tmp.rem != 0) {
const std::size_t remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Reduce(pData + startIndex, pBuf, remain, mpiType,
mpiOp, root, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + remain, pData + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

delete[] pBuf;
pBuf = NULL;
}
#else
{
T* pBuf = new T[end - start];
answer = MPI_Reduce(pData, pBuf, end - start, mpiType, mpiOp, root,
MPI_COMM_WORLD);
std::copy(pBuf, pBuf + (end - start), pData);
delete[] pBuf;
pBuf = NULL;
}
#endif  

return answer;
}

int TlCommunicate::reduce_SUM(unsigned int* pData, std::size_t size, int root) {
return this->reduce(pData, MPI_UNSIGNED, 0, size, MPI_SUM, root);
}

int TlCommunicate::reduce_SUM(unsigned long* pData, std::size_t size,
int root) {
return this->reduce(pData, MPI_UNSIGNED_LONG, 0, size, MPI_SUM, root);
}

int TlCommunicate::reduce_MAXLOC(double* pValue, int* pIndex, int root) {
struct DoubleInt {
double value;
int index;
};
DoubleInt in;
DoubleInt out;

in.value = *pValue;
in.index = *pIndex;

const int answer = MPI_Reduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
root, MPI_COMM_WORLD);
*pValue = out.value;
*pIndex = out.index;

return answer;
}


int TlCommunicate::allReduce_SUM(int& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

const int nSize = 1;

int nReceive = 0;
int nError = MPI_Allreduce(&rData, &nReceive, nSize, MPI_INT, MPI_SUM,
MPI_COMM_WORLD);

rData = nReceive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_SUM(unsigned int& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

const int nSize = 1;

int nReceive = 0;
int nError = MPI_Allreduce(&rData, &nReceive, nSize, MPI_UNSIGNED, MPI_SUM,
MPI_COMM_WORLD);

rData = nReceive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_SUM(long& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

const int nSize = 1;

long nReceive = 0;
int nError = MPI_Allreduce(&rData, &nReceive, nSize, MPI_LONG, MPI_SUM,
MPI_COMM_WORLD);

rData = nReceive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_SUM(unsigned long& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

const int nSize = 1;

unsigned long nReceive = 0;
int nError = MPI_Allreduce(&rData, &nReceive, nSize, MPI_UNSIGNED_LONG,
MPI_SUM, MPI_COMM_WORLD);

rData = nReceive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_SUM(double& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

const int nSize = 1;

double dReceive = 0.0;
int nError = MPI_Allreduce(&rData, &dReceive, nSize, MPI_DOUBLE, MPI_SUM,
MPI_COMM_WORLD);

rData = dReceive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_SUM(std::vector<int>& rData) {
return this->allReduce_SUM(rData, MPI_INT, 0, rData.size());
}

int TlCommunicate::allReduce_SUM(std::vector<unsigned int>& rData) {
return this->allReduce_SUM(rData, MPI_UNSIGNED, 0, rData.size());
}

int TlCommunicate::allReduce_SUM(std::vector<long>& rData) {
return this->allReduce_SUM(rData, MPI_LONG, 0, rData.size());
}

int TlCommunicate::allReduce_SUM(std::vector<unsigned long>& rData) {
return this->allReduce_SUM(rData, MPI_UNSIGNED_LONG, 0, rData.size());
}

int TlCommunicate::allReduce_SUM(std::vector<double>& rData) {
return this->allReduce_SUM(rData, MPI_DOUBLE, 0, rData.size());
}

template <typename T>
int TlCommunicate::allReduce_SUM(std::vector<T>& data,
const MPI_Datatype mpiType,
const std::size_t start,
const std::size_t end) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
std::vector<T> buf(bufCount);
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Allreduce(&(data[startIndex]), &(buf[0]), bufCount,
mpiType, MPI_SUM, MPI_COMM_WORLD);
std::copy(buf.begin(), buf.end(), data.begin() + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Allreduce(&(data[startIndex]), &(buf[0]), remain,
mpiType, MPI_SUM, MPI_COMM_WORLD);
std::copy(buf.begin(), buf.begin() + remain,
data.begin() + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}
}
#else
{
T* pBuf = new T[end - start];
answer = MPI_Allreduce(&(data[0]), pBuf, end - start, mpiType, MPI_SUM,
MPI_COMM_WORLD);
std::copy(pBuf, pBuf + (end - start), &(data[0]));
delete[] pBuf;
pBuf = NULL;
}
#endif  

this->time_allreduce_.stop();
return answer;
}

int TlCommunicate::allReduce_SUM(std::valarray<double>& rData) {
return this->allReduce_SUM(rData, 0, rData.size());
}

int TlCommunicate::allReduce_SUM(std::valarray<double>& data,
const std::size_t start,
const std::size_t end) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount =
static_cast<int>(this->workMemSize_ / sizeof(double));
const ldiv_t tmp = std::ldiv(length, bufCount);
double* pBuf = new double[bufCount];

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Allreduce((void*)&(data[startIndex]), pBuf, bufCount,
MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + bufCount, &(data[0]) + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Allreduce((void*)&(data[startIndex]), pBuf, remain,
MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + remain, &(data[0]) + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

delete[] pBuf;
pBuf = NULL;
}
#else
{
double* pBuf = new double[end - start];
answer = MPI_Allreduce((void*)&(data[0]), pBuf, end - start, MPI_DOUBLE,
MPI_SUM, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + (end - start), &(data[0]));
delete[] pBuf;
pBuf = NULL;
}
#endif  

this->time_allreduce_.stop();
return answer;
}

template <typename T>
int TlCommunicate::allReduce(T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const MPI_Op mpiOp) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

int answer = 0;

#ifdef DIV_COMM
{
const long length = end - start;
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);
T* pBuf = new T[bufCount];

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Allreduce(pData + startIndex, pBuf, bufCount, mpiType,
mpiOp, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + bufCount, pData + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

if (tmp.rem != 0) {
const std::size_t remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Allreduce(pData + startIndex, pBuf, remain, mpiType,
mpiOp, MPI_COMM_WORLD);
std::copy(pBuf, pBuf + remain, pData + startIndex);
if (answer != 0) {
this->log_.critical(TlUtils::format(
"MPI error. %s:%d answer=%d", __FILE__, __LINE__, answer));
}
}

delete[] pBuf;
pBuf = NULL;
}
#else
{
T* pBuf = new T[end - start];
answer = MPI_Allreduce(pData, pBuf, end - start, mpiType, mpiOp,
MPI_COMM_WORLD);
std::copy(pBuf, pBuf + (end - start), pData);
delete[] pBuf;
pBuf = NULL;
}
#endif  

this->time_allreduce_.stop();
return answer;
}

int TlCommunicate::allReduce_SUM(int* pData, std::size_t length) {
return this->allReduce(pData, MPI_INT, 0, length, MPI_SUM);
}

int TlCommunicate::allReduce_SUM(double* pData, std::size_t length) {
return this->allReduce(pData, MPI_DOUBLE, 0, length, MPI_SUM);
}


template <typename T>
int TlCommunicate::iAllReduce(const T* pSendBuf, T* pRecvBuf, int count,
const MPI_Datatype mpiType, const MPI_Op mpiOp) {
this->time_iallreduce_.start();
++(this->counter_iallreduce_);

int answer = 0;

#ifdef DIV_COMM
{
#error  
}
#else
{
MPI_Request* pRequest = new MPI_Request;
const int tag = 0;
answer = MPI_Iallreduce(pSendBuf, pRecvBuf, count, mpiType, mpiOp,
MPI_COMM_WORLD, pRequest);

std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)pRecvBuf);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::SEND);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}
#endif  

this->time_iallreduce_.stop();
return answer;
}

int TlCommunicate::iAllReduce_SUM(const double* pSendBuf, double* pRecvBuf,
int count) {
return this->iAllReduce(pSendBuf, pRecvBuf, count, MPI_DOUBLE, MPI_SUM);
}


int TlCommunicate::allReduce_SUM(TlDenseGeneralMatrix_Lapack* pMatrix) {
return this->allReduce_SUM(pMatrix->data(), pMatrix->getNumOfElements());
}

int TlCommunicate::allReduce_SUM(TlDenseSymmetricMatrix_Lapack* pMatrix) {
return this->allReduce_SUM(pMatrix->data(), pMatrix->getNumOfElements());
}

int TlCommunicate::allReduce_SUM(TlDenseVector_Lapack* pVector) {
return this->allReduce_SUM(
pVector->data(),
pVector->getSize());  
}

int TlCommunicate::allReduce_SUM(TlSparseMatrix& rMatrix) {
int err = this->reduce_SUM(rMatrix);
if (err == 0) {
err = this->broadcast(rMatrix, 0);
}

return err;
}

int TlCommunicate::reduce_SUM(TlSparseMatrix& rMatrix, int root) {
const int numOfProcs = this->getNumOfProcs();
const int myRank = this->getRank();
assert(0 <= root);
assert(root < numOfProcs);

const int tag_size = TLC_REDUCE_SUM_TLSPARSEMATRIX_SIZE;
const int tag_data = TLC_REDUCE_SUM_TLSPARSEMATRIX_DATA;

if (myRank == root) {
std::vector<unsigned long> sizes(numOfProcs);
{
for (int i = 0; i < numOfProcs - 1; ++i) {
int src = 0;
unsigned long numOfSize = 0;
this->receiveDataFromAnySource(numOfSize, &src, tag_size);
sizes[src] = numOfSize;
}
}

std::vector<std::vector<TlMatrixObject::MatrixElement> > bufs(
numOfProcs);
for (int src = 0; src < numOfProcs; ++src) {
if (src != myRank) {
bufs[src].resize(sizes[src]);
this->iReceiveDataX(&(bufs[src][0]), this->MPI_MATRIXELEMENT, 0,
sizes[src], src, tag_data);
}
}

std::vector<int> recvd(numOfProcs, 0);
recvd[root] = 1;  
while (true) {
for (int src = 0; src < numOfProcs; ++src) {
if (recvd[src] == 0) {
if (this->test(&(bufs[src][0]))) {
this->wait(&(bufs[src][0]));

for (std::size_t i = 0; i < sizes[src]; ++i) {
rMatrix.add(bufs[src][i].row, bufs[src][i].col,
bufs[src][i].value);
}

recvd[src] = 1;
}
}
}

if (std::accumulate(recvd.begin(), recvd.end(), 0) == numOfProcs) {
break;
}
}

} else {
unsigned long numOfSize = rMatrix.getSize();
this->sendData(numOfSize, root, tag_size);

TlSparseMatrix::const_iterator itEnd = rMatrix.end();
std::size_t count = 0;
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);
for (TlSparseMatrix::const_iterator it = rMatrix.begin(); it != itEnd;
++it) {
buf[count].row = it->first.row;
buf[count].col = it->first.col;
buf[count].value = it->second;

++count;
}
assert(count == numOfSize);

this->sendDataX(&(buf[0]), this->MPI_MATRIXELEMENT, 0, numOfSize, root,
tag_data);
}

return 0;
}

int TlCommunicate::broadcast(TlSparseMatrix& rMatrix, const int root) {
unsigned long header[3];
std::size_t numOfSize = 0;

if (this->getRank() == root) {
numOfSize = rMatrix.getSize();

header[0] = rMatrix.getNumOfRows();
header[1] = rMatrix.getNumOfCols();
header[2] = numOfSize;

TlSparseMatrix::const_iterator itEnd = rMatrix.end();
std::size_t count = 0;
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);
for (TlSparseMatrix::const_iterator it = rMatrix.begin(); it != itEnd;
++it) {
buf[count].row = it->first.row;
buf[count].col = it->first.col;
buf[count].value = it->second;

++count;
}
assert(count == numOfSize);

this->broadcast(header, MPI_UNSIGNED_LONG, 0, 3, root);
this->broadcast(&(buf[0]), this->MPI_MATRIXELEMENT, 0, numOfSize, root);
} else {
this->broadcast(header, MPI_UNSIGNED_LONG, 0, 3, root);

rMatrix.clear();
rMatrix.resize(header[0], header[1]);

numOfSize = header[2];
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);
this->broadcast(&(buf[0]), this->MPI_MATRIXELEMENT, 0, numOfSize, root);

for (std::size_t i = 0; i < numOfSize; ++i) {
rMatrix.set(buf[i].row, buf[i].col, buf[i].value);
}
}

return 0;
}

int TlCommunicate::allReduce_AND(bool& rData) {
int nData = (rData == true) ? 1 : 0;
int nError = this->allReduce_SUM(nData);

rData = (nData != 0) ? true : false;

return nError;
}

int TlCommunicate::allReduce_MAX(int& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

int receive = 0;
const int count = 1;
int nError = MPI_Allreduce(&rData, &receive, count, MPI_INT, MPI_MAX,
MPI_COMM_WORLD);

rData = receive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_MIN(int& rData) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

int receive = 0;
const int count = 1;
int nError = MPI_Allreduce(&rData, &receive, count, MPI_INT, MPI_MIN,
MPI_COMM_WORLD);

rData = receive;

this->time_allreduce_.stop();
return nError;
}

int TlCommunicate::allReduce_MAXLOC(double* pValue, int* pIndex) {
this->time_allreduce_.start();
++(this->counter_allreduce_);

struct DoubleInt {
double value;
int index;
};
DoubleInt in;
DoubleInt out;

in.value = *pValue;
in.index = *pIndex;

const int answer =
MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
*pValue = out.value;
*pIndex = out.index;

this->time_allreduce_.stop();
return answer;
}

int TlCommunicate::sendData(bool data, int destination, int tag) {
const int yn = (data == false) ? 0 : 1;
return this->sendData(yn, destination, tag);
}

int TlCommunicate::sendData(int data, int nDestination, int nTag) {
int nErr =
MPI_Send((void*)&data, 1, MPI_INT, nDestination, nTag, MPI_COMM_WORLD);
return nErr;
}

int TlCommunicate::sendData(unsigned int data, int nDestination, int nTag) {
int nErr = MPI_Send((void*)&data, 1, MPI_UNSIGNED, nDestination, nTag,
MPI_COMM_WORLD);
return nErr;
}

int TlCommunicate::sendData(long data, int nDestination, int nTag) {
int nErr =
MPI_Send((void*)&data, 1, MPI_LONG, nDestination, nTag, MPI_COMM_WORLD);
return nErr;
}

int TlCommunicate::sendData(unsigned long data, int nDestination, int nTag) {
int nErr = MPI_Send((void*)&data, 1, MPI_UNSIGNED_LONG, nDestination, nTag,
MPI_COMM_WORLD);
return nErr;
}

int TlCommunicate::sendData(double data, int nDestination, int nTag) {
int nErr = MPI_Send((void*)&data, 1, MPI_DOUBLE, nDestination, nTag,
MPI_COMM_WORLD);
return nErr;
}

template <typename T>
int TlCommunicate::sendData(const std::vector<T>& data,
const MPI_Datatype mpiType, const std::size_t start,
const std::size_t end, const int destination,
const int tag) {
int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Send((void*)&(data[startIndex]), bufCount, mpiType,
destination, tag, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Send((void*)&(data[startIndex]), remain, mpiType,
destination, tag, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Send((void*)&(data[0]), end - start, mpiType, destination,
tag, MPI_COMM_WORLD);
}
#endif  

return answer;
}

int TlCommunicate::sendData(const std::vector<int>& data, int destination,
int tag) {
const std::size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer = this->sendData(data, MPI_INT, 0, size, destination, tag);
}
return answer;
}

int TlCommunicate::sendData(const std::vector<unsigned int>& data,
int destination, int tag) {
const std::size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer = this->sendData(data, MPI_UNSIGNED, 0, size, destination, tag);
}
return answer;
}

int TlCommunicate::sendData(const std::vector<long>& data, int destination,
int tag) {
const std::size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer = this->sendData(data, MPI_LONG, 0, size, destination, tag);
}
return answer;
}

int TlCommunicate::sendData(const std::vector<unsigned long>& data,
int destination, int tag) {
const std::size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer =
this->sendData(data, MPI_UNSIGNED_LONG, 0, size, destination, tag);
}
return answer;
}

int TlCommunicate::sendData(const std::vector<double>& data, int destination,
int tag) {
const std::size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer = this->sendData(data, MPI_DOUBLE, 0, size, destination, tag);
}
return answer;
}

int TlCommunicate::sendData(const std::valarray<double>& data, int destination,
int tag) {
const size_t size = data.size();
int answer = this->sendData(size, destination, tag);

if ((answer == 0) && (size > 0)) {
answer = this->sendData(data, 0, size, destination, tag);
}

return answer;
}

int TlCommunicate::sendData(const std::valarray<double>& data,
const std::size_t start, const std::size_t end,
int destination, int tag) {
std::valarray<double>& data_tmp = const_cast<std::valarray<double>&>(data);

int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount =
static_cast<int>(this->workMemSize_ / sizeof(double));
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Send((void*)&(data_tmp[startIndex]), bufCount,
MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Send((void*)&(data_tmp[startIndex]), remain,
MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Send((void*)&(data_tmp[0]), end - start, MPI_DOUBLE,
destination, tag, MPI_COMM_WORLD);
}
#endif  

return answer;
}

int TlCommunicate::sendData(const std::string& data, int nDestination,
int nTag) {
const int nSize = static_cast<int>(data.length());
char* pBuff = new char[nSize];
std::strncpy(pBuff, data.c_str(), nSize);

this->sendData(nSize, nDestination, nTag);
int nErr = MPI_Send((void*)pBuff, nSize, MPI_CHAR, nDestination, nTag,
MPI_COMM_WORLD);

delete[] pBuff;
pBuff = NULL;

return nErr;
}




int TlCommunicate::sendData(const TlDenseGeneralMatrix_Lapack& data,
const int destination, const int tag) {
const int headerSize = 2;
TlMatrixObject::index_type* pHeader =
new TlMatrixObject::index_type[headerSize];
pHeader[0] = data.getNumOfRows();
pHeader[1] = data.getNumOfCols();
int err = this->sendDataX(pHeader, headerSize, destination, tag);
this->log_.debug(TlUtils::format(
"TlCommunicate::sendData(TlDenseGeneralMatrix_Lapack): dest=%d, "
"tag=%d, "
"row=%d, "
"col=%d, "
"err=%d.",
destination, tag, data.getNumOfRows(), data.getNumOfCols(), err));
delete[] pHeader;
pHeader = NULL;
if (err != 0) {
return err;
}

err =
this->sendDataX(data.data(), data.getNumOfElements(), destination, tag);
this->log_.debug(TlUtils::format(
"TlCommunicate::sendData(TlDenseGeneralMatrix_Lapack): dest=%d, "
"tag=%d, size=%ld, err=%d.",
destination, tag, data.getNumOfElements(), err));
return err;
}

int TlCommunicate::sendData(const TlDenseSymmetricMatrix_Lapack& data,
int nDestination, int nTag) {
const std::size_t dim = data.getNumOfRows();
int nErr = this->sendData(dim, nDestination, nTag);
if (nErr == 0) {
nErr = this->sendDataX(data.data(), MPI_DOUBLE, 0,
data.getNumOfElements(), nDestination, nTag);
}

return nErr;
}

int TlCommunicate::sendData(const TlDenseVector_Lapack& data, int destination,
int nTag) {
const std::size_t dim = data.getSize();
int nErr = this->sendData(dim, destination, nTag);
if (nErr == 0) {
const double* buf =
dynamic_cast<TlDenseVector_ImplLapack*>(data.pImpl_)->vector_;
nErr = this->sendDataX(buf, MPI_DOUBLE, 0, dim, destination, nTag);
}

return nErr;
}

int TlCommunicate::sendData(const TlSparseMatrix& data, int dest, int tag) {
const std::size_t numOfSize = data.getSize();

unsigned long header[3];
header[0] = static_cast<unsigned long>(data.getNumOfRows());
header[1] = static_cast<unsigned long>(data.getNumOfCols());
header[2] = static_cast<unsigned long>(numOfSize);
int err = this->sendDataX(&(header[0]), 3, dest, tag);

if (err == 0) {
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);

TlSparseMatrix::const_iterator itEnd = data.end();
std::size_t count = 0;
for (TlSparseMatrix::const_iterator it = data.begin(); it != itEnd;
++it) {
buf[count].row = it->first.row;
buf[count].col = it->first.col;
buf[count].value = it->second;

++count;
}
assert(count == numOfSize);

err = this->sendDataX(&(buf[0]), this->MPI_MATRIXELEMENT, 0, numOfSize,
dest, tag);
}

return err;
}

int TlCommunicate::sendDataX(const int* pData, const std::size_t size,
const int dest, const int tag) {
return this->sendDataX(pData, MPI_INT, 0, size, dest, tag);
}

int TlCommunicate::sendDataX(const unsigned int* pData, const std::size_t size,
const int dest, const int tag) {
return this->sendDataX(pData, MPI_UNSIGNED, 0, size, dest, tag);
}

int TlCommunicate::sendDataX(const unsigned long* pData, const std::size_t size,
const int dest, const int tag) {
return this->sendDataX(pData, MPI_UNSIGNED_LONG, 0, size, dest, tag);
}

int TlCommunicate::sendDataX(const double* pData, const std::size_t size,
const int dest, const int tag) {
return this->sendDataX(pData, MPI_DOUBLE, 0, size, dest, tag);
}

template <typename T>
int TlCommunicate::sendDataX(const T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int dest, const int tag) {
const int answer = MPI_Send((void*)(pData + start), (end - start), mpiType,
dest, tag, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}

return answer;
}

int TlCommunicate::receiveData(bool& data, int src, int tag) {
int yn = 0;
int err = this->receiveData(yn, src, tag);
data = (yn != 0);

return err;
}

int TlCommunicate::receiveData(int& rData, int nSrc, int nTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&rData, 1, MPI_INT, nSrc, nTag, MPI_COMM_WORLD,
&status);

return nErr;
}

int TlCommunicate::receiveData(unsigned int& rData, int nSrc, int nTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&rData, 1, MPI_UNSIGNED, nSrc, nTag,
MPI_COMM_WORLD, &status);

return nErr;
}

int TlCommunicate::receiveData(long& rData, int nSrc, int nTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&rData, 1, MPI_LONG, nSrc, nTag, MPI_COMM_WORLD,
&status);

return nErr;
}

int TlCommunicate::receiveData(unsigned long& rData, int nSrc, int nTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&rData, 1, MPI_UNSIGNED_LONG, nSrc, nTag,
MPI_COMM_WORLD, &status);

return nErr;
}

int TlCommunicate::receiveData(double& rData, int nSrc, int nTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&rData, 1, MPI_DOUBLE, nSrc, nTag,
MPI_COMM_WORLD, &status);

return nErr;
}

template <typename T>
int TlCommunicate::receiveData(std::vector<T>& data, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int src, const int tag) {
int answer = 0;
MPI_Status status;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Recv(&(data[startIndex]), bufCount, mpiType, src, tag,
MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Recv(&(data[startIndex]), remain, mpiType, src, tag,
MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Recv(&(data[0]), end - start, mpiType, src, tag,
MPI_COMM_WORLD, &status);
}
#endif  

return answer = 0;
}





int TlCommunicate::receiveData(std::vector<int>& data, int src, int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, MPI_INT, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::vector<unsigned int>& data, int src,
int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, MPI_UNSIGNED, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::vector<long>& data, int src, int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, MPI_LONG, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::vector<unsigned long>& data, int src,
int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, MPI_UNSIGNED_LONG, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::vector<double>& data, int src, int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, MPI_DOUBLE, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::valarray<double>& data,
const std::size_t start, const std::size_t end,
const int src, const int tag) {
int answer = 0;
MPI_Status status;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount =
static_cast<int>(this->workMemSize_ / sizeof(double));
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Recv((void*)&(data[startIndex]), bufCount, MPI_DOUBLE,
src, tag, MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Recv((void*)&(data[startIndex]), remain, MPI_DOUBLE,
src, tag, MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Recv((void*)&(data[0]), end - start, MPI_DOUBLE, src, tag,
MPI_COMM_WORLD, &status);
}
#endif  

return answer;
}

int TlCommunicate::receiveData(std::valarray<double>& data, int src, int tag) {
std::size_t size = 0;
this->receiveData(size, src, tag);
data.resize(size);

return this->receiveData(data, 0, size, src, tag);
}

int TlCommunicate::receiveData(std::string& rData, int nSrc, int nTag) {
int nErr = 0;

int nSize = 0;
this->receiveData(nSize, nSrc, nTag);

char* pBuff = new char[nSize];
MPI_Status status;
nErr = MPI_Recv((void*)pBuff, nSize, MPI_CHAR, nSrc, nTag, MPI_COMM_WORLD,
&status);
rData = std::string(pBuff, nSize);

delete[] pBuff;
pBuff = NULL;

return nErr;
}




int TlCommunicate::receiveData(TlDenseGeneralMatrix_Lapack& data, const int src,
const int tag) {
const int headerSize = 2;
TlMatrixObject::index_type* pHeader =
new TlMatrixObject::index_type[headerSize];
int err = this->receiveDataX(pHeader, headerSize, src, tag);
const TlMatrixObject::index_type numOfRows = pHeader[0];
const TlMatrixObject::index_type numOfCols = pHeader[1];
this->log_.debug(TlUtils::format(
"TlCommunicate::recvData(TlDenseGeneralMatrix_Lapack): src=%d, tag=%d, "
"row=%d, col=%d, err=%d.",
src, tag, numOfRows, numOfCols, err));

delete[] pHeader;
pHeader = NULL;
if (err != 0) {
return err;
}

data.resize(numOfRows, numOfCols);
err = this->receiveDataX(data.data(), MPI_DOUBLE, 0,
data.getNumOfElements(), src, tag);
this->log_.debug(TlUtils::format(
"TlCommunicate::recvData(TlDenseGeneralMatrix_Lapack): src=%d, tag=%d, "
"size=%ld, err=%d.",
src, tag, data.getNumOfElements(), err));

return err;
}

int TlCommunicate::receiveData(TlDenseSymmetricMatrix_Lapack& rData, int nSrc,
int nTag) {
std::size_t dim = 0;
int nErr = this->receiveData(dim, nSrc, nTag);
if (nErr == 0) {
rData.resize(dim);
nErr = this->receiveDataX(rData.data(), MPI_DOUBLE, 0,
rData.getNumOfElements(), nSrc, nTag);
}

return nErr;
}

int TlCommunicate::receiveData(TlDenseVector_Lapack& rData, int src, int tag) {
std::size_t dim = 0;
int nErr = this->receiveData(dim, src, tag);
if (nErr == 0) {
rData.resize(dim);
double* buf =
dynamic_cast<TlDenseVector_ImplLapack*>(rData.pImpl_)->vector_;
nErr = this->receiveDataX(buf, MPI_DOUBLE, 0, dim, src, tag);
}

return nErr;
}

int TlCommunicate::receiveData(TlSparseMatrix& rData, int src, int tag) {
unsigned long header[3];
int err = this->receiveDataX(header, MPI_UNSIGNED_LONG, 0, 3, src, tag);

if (err == 0) {
const std::size_t numOfSize = header[2];
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);

err = this->receiveDataX(&(buf[0]), this->MPI_MATRIXELEMENT, 0,
numOfSize, src, tag);

rData.clear();
rData.resize(header[0], header[1]);
for (std::size_t i = 0; i < numOfSize; ++i) {
rData.set(buf[i].row, buf[i].col, buf[i].value);
}
}

return err;
}

int TlCommunicate::receiveDataX(int* pData, const std::size_t size,
const int src, const int tag) {
return this->receiveDataX(pData, MPI_INT, 0, size, src, tag);
}

int TlCommunicate::receiveDataX(unsigned int* pData, const std::size_t size,
const int src, const int tag) {
return this->receiveDataX(pData, MPI_UNSIGNED, 0, size, src, tag);
}

int TlCommunicate::receiveDataX(double* pData, const std::size_t size,
const int src, const int tag) {
return this->receiveDataX(pData, MPI_DOUBLE, 0, size, src, tag);
}

int TlCommunicate::receiveDataX(TlMatrixObject::MatrixElement* pData,
const std::size_t size, const int src,
const int tag) {
return this->receiveDataX(pData, this->MPI_MATRIXELEMENT, 0, size, src,
tag);
}

int TlCommunicate::receiveDataX(TlVectorObject::VectorElement* pData,
const std::size_t size, const int src,
const int tag) {
return this->receiveDataX(pData, this->MPI_VECTORELEMENT, 0, size, src,
tag);
}

template <typename T>
int TlCommunicate::receiveDataX(T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int src, const int tag) {
int answer = 0;
MPI_Status status;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Recv((void*)(pData + startIndex), bufCount, mpiType,
src, tag, MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Recv((void*)(pData + startIndex), remain, mpiType, src,
tag, MPI_COMM_WORLD, &status);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Recv((void*)(pData), end - start, mpiType, src, tag,
MPI_COMM_WORLD, &status);
}
#endif  

return answer;
}

template <typename T>
int TlCommunicate::receiveDataFromAnySource(T& data, const MPI_Datatype mpiType,
int* pSrc, const int tag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&data, 1, mpiType, MPI_ANY_SOURCE, tag,
MPI_COMM_WORLD, &status);

assert(pSrc != NULL);
*pSrc = status.MPI_SOURCE;

return nErr;
}

template <typename T>
int TlCommunicate::receiveDataFromAnySource(T& data, const MPI_Datatype mpiType,
int* pSrc, int* pTag) {
MPI_Status status;
int nErr = MPI_Recv((void*)&data, 1, mpiType, MPI_ANY_SOURCE, MPI_ANY_TAG,
MPI_COMM_WORLD, &status);

assert(pSrc != NULL);
*pSrc = status.MPI_SOURCE;
if (pTag != NULL) {
*pTag = status.MPI_TAG;
}

return nErr;
}

int TlCommunicate::receiveDataFromAnySource(int& data, int* pSrc,
const int tag) {
return this->receiveDataFromAnySource(data, MPI_INT, pSrc, tag);
}

int TlCommunicate::receiveDataFromAnySource(unsigned long& data, int* pSrc,
const int tag) {
return this->receiveDataFromAnySource(data, MPI_UNSIGNED_LONG, pSrc, tag);
}

int TlCommunicate::receiveDataFromAnySource(int& data, int* pSrc, int* pTag) {
return this->receiveDataFromAnySource(data, MPI_INT, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(unsigned int& data, int* pSrc,
int* pTag) {
return this->receiveDataFromAnySource(data, MPI_UNSIGNED, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(long& data, int* pSrc, int* pTag) {
return this->receiveDataFromAnySource(data, MPI_LONG, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(unsigned long& data, int* pSrc,
int* pTag) {
return this->receiveDataFromAnySource(data, MPI_UNSIGNED_LONG, pSrc, pTag);
}

template <typename T>
int TlCommunicate::receiveDataFromAnySource(std::vector<T>& data,
const MPI_Datatype mpiType,
int* pSrc, int* pTag) {
std::size_t size = 0;
int src = 0;
int tag = 0;
this->receiveDataFromAnySource(size, &src, &tag);
data.resize(size);

const int answer = this->receiveData(data, mpiType, 0, size, src, tag);

assert(pSrc != NULL);
*pSrc = src;
if (pTag != NULL) {
*pTag = tag;
}

return answer;
}

int TlCommunicate::receiveDataFromAnySource(std::vector<int>& data, int* pSrc,
int* pTag) {
return this->receiveDataFromAnySource(data, MPI_INT, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(std::vector<unsigned int>& data,
int* pSrc, int* pTag) {
return this->receiveDataFromAnySource(data, MPI_UNSIGNED, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(std::vector<long>& data, int* pSrc,
int* pTag) {
return this->receiveDataFromAnySource(data, MPI_LONG, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(std::vector<unsigned long>& data,
int* pSrc, int* pTag) {
return this->receiveDataFromAnySource(data, MPI_UNSIGNED_LONG, pSrc, pTag);
}

int TlCommunicate::receiveDataFromAnySource(std::vector<double>& data,
int* pSrc, int* pTag) {
return this->receiveDataFromAnySource(data, MPI_DOUBLE, pSrc, pTag);
}


int TlCommunicate::receiveDataFromAnySource(TlDenseGeneralMatrix_Lapack& data,
int* pSrc, int tag) {
assert(pSrc != NULL);

const int headerSize = 2;
TlMatrixObject::index_type* pHeader =
new TlMatrixObject::index_type[headerSize];
int src = 0;
int err = this->receiveDataFromAnySourceX(pHeader, headerSize, &src, tag);
if (err != 0) {
return err;
}

*pSrc = src;

const TlMatrixObject::index_type numOfRows = pHeader[0];
const TlMatrixObject::index_type numOfCols = pHeader[1];
delete[] pHeader;
pHeader = NULL;

data.resize(numOfRows, numOfCols);
return this->receiveDataX(data.data(), MPI_DOUBLE, 0,
data.getNumOfElements(), src, tag);
}

int TlCommunicate::receiveDataFromAnySource(TlSparseMatrix* pData, int* pSrc,
int tag) {
assert(pData != NULL);
int src = 0;

unsigned long header[3];
int err = this->receiveDataFromAnySourceX(header, MPI_UNSIGNED_LONG, 0, 3,
&src, tag);

assert(pSrc != NULL);
*pSrc = src;

const int numOfRows = header[0];
const int numOfCols = header[1];
if (err == 0) {
const std::size_t numOfSize = header[2];
std::vector<TlMatrixObject::MatrixElement> buf(numOfSize);

err = this->receiveDataX(&(buf[0]), this->MPI_MATRIXELEMENT, 0,
numOfSize, src, tag);
assert(err == 0);

pData->clear();
pData->resize(numOfRows, numOfCols);
for (std::size_t i = 0; i < numOfSize; ++i) {
const int r = buf[i].row;
const int c = buf[i].col;
assert((0 <= r) && (r < numOfRows));
assert((0 <= c) && (c < numOfCols));
pData->set(r, c, buf[i].value);
}
}

return err;
}








template <typename T>
int TlCommunicate::receiveDataFromAnySourceX(T* pData,
const MPI_Datatype mpiType,
const std::size_t start,
const std::size_t end, int* pSrc,
const int tag) {
int answer = 0;
MPI_Status status;
int src = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);
bool isSrcDefined = false;

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
if (isSrcDefined != true) {
answer =
MPI_Recv((void*)(pData + startIndex), bufCount, mpiType,
MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
isSrcDefined = true;
} else {
answer = MPI_Recv((void*)(pData + startIndex), bufCount,
mpiType, src, tag, MPI_COMM_WORLD, &status);
}
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
if (isSrcDefined != true) {
answer = MPI_Recv((void*)(pData + startIndex), remain, mpiType,
MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
isSrcDefined = true;
} else {
answer = MPI_Recv((void*)(pData + startIndex), remain, mpiType,
src, tag, MPI_COMM_WORLD, &status);
}
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Recv((void*)(pData), end - start, mpiType, MPI_ANY_SOURCE,
tag, MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
}
#endif  

if (pSrc != NULL) {
*pSrc = src;
}

return answer;
}

int TlCommunicate::receiveDataFromAnySourceX(int* pData, const std::size_t size,
int* pSrc, const int tag) {
return this->receiveDataFromAnySourceX(pData, MPI_INT, 0, size, pSrc, tag);
}

int TlCommunicate::receiveDataFromAnySourceX(unsigned long* pData,
const std::size_t size, int* pSrc,
const int tag) {
return this->receiveDataFromAnySourceX(pData, MPI_UNSIGNED_LONG, 0, size,
pSrc, tag);
}

template <typename T>
int TlCommunicate::receiveDataFromAnySourceAnyTagX(T* pData,
const MPI_Datatype mpiType,
const std::size_t start,
const std::size_t end,
int* pSrc, int* pTag) {
int answer = 0;
MPI_Status status;
int src = 0;
int tag = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);
bool isSrcDefined = false;

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
if (isSrcDefined != true) {
answer = MPI_Recv((void*)(pData + startIndex), bufCount,
mpiType, MPI_ANY_SOURCE, MPI_ANY_TAG,
MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
tag = status.MPI_TAG;
isSrcDefined = true;
} else {
answer = MPI_Recv((void*)(pData + startIndex), bufCount,
mpiType, src, tag, MPI_COMM_WORLD, &status);
}
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
if (isSrcDefined != true) {
answer = MPI_Recv((void*)(pData + startIndex), remain, mpiType,
MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
isSrcDefined = true;
} else {
answer = MPI_Recv((void*)(pData + startIndex), remain, mpiType,
src, tag, MPI_COMM_WORLD, &status);
}
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Recv((void*)(pData), end - start, mpiType, MPI_ANY_SOURCE,
MPI_ANY_TAG, MPI_COMM_WORLD, &status);
src = status.MPI_SOURCE;
tag = status.MPI_TAG;
}
#endif  

if (pSrc != NULL) {
*pSrc = src;
}
if (pTag != NULL) {
*pTag = tag;
}

return answer;
}

template <typename T>
int TlCommunicate::iSendData(const T& data, const MPI_Datatype mpiType,
const int destination, const int tag) {
MPI_Request* pRequest = new MPI_Request;
const int answer = MPI_Isend((void*)&data, 1, mpiType, destination, tag,
MPI_COMM_WORLD, pRequest);
if (answer == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)&data);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::SEND);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}


return answer;
}

int TlCommunicate::iSendData(const int& data, const int destination,
const int tag) {
return this->iSendData(data, MPI_INT, destination, tag);
}

int TlCommunicate::iSendData(const unsigned int& data, const int destination,
const int tag) {
return this->iSendData(data, MPI_UNSIGNED, destination, tag);
}

int TlCommunicate::iSendData(const long& data, const int destination,
const int tag) {
return this->iSendData(data, MPI_LONG, destination, tag);
}

int TlCommunicate::iSendData(const unsigned long& data, const int destination,
const int tag) {
return this->iSendData(data, MPI_UNSIGNED_LONG, destination, tag);
}

template <typename T>
int TlCommunicate::iSendData(const std::vector<T>& data,
const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int destination, const int tag) {
int answer = 0;
std::vector<uintptr_t> requests;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
std::vector<T> buf(bufCount);
const ldiv_t tmp = std::ldiv(length, bufCount);

if (tmp.quot != 0) {
for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
MPI_Request* pRequest = new MPI_Request;
answer =
MPI_Isend((void*)&(data[startIndex]), bufCount, mpiType,
destination, tag, MPI_COMM_WORLD, pRequest);
requests.push_back(
reinterpret_cast<uintptr_t>((void*)pRequest));
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
MPI_Request* pRequest = new MPI_Request;
answer = MPI_Isend((void*)&(data[startIndex]), remain, mpiType,
destination, tag, MPI_COMM_WORLD, pRequest);
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
MPI_Request* pRequest = new MPI_Request;
answer = MPI_Isend((void*)&(data[0]), end - start, mpiType, destination,
tag, MPI_COMM_WORLD, pRequest);
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
}
#endif  

const uintptr_t key = reinterpret_cast<uintptr_t>((void*)&data[start]);
const NonBlockingCommParam param(requests, tag, NonBlockingCommParam::SEND);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }

return answer;
}

int TlCommunicate::iSendData(const std::vector<int>& data,
const int destination, const int tag) {
const std::size_t size = data.size();
int answer = this->iSendData(size, destination, tag);
this->wait(size);

answer |= this->iSendData(data, MPI_INT, 0, size, destination, tag);
return answer;
}

int TlCommunicate::iSendData(const std::vector<double>& data, int destination,
int tag) {
const std::size_t size = data.size();
int answer = this->iSendData(size, destination, tag);
this->wait(size);

answer |= this->iSendData(data, MPI_DOUBLE, 0, size, destination, tag);
return answer;
}











template <typename T>
int TlCommunicate::iSendDataX(const T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int dest, const int tag) {
MPI_Request* pRequest = new MPI_Request;
const int answer = MPI_Isend((void*)(pData + start), (end - start), mpiType,
dest, tag, MPI_COMM_WORLD, pRequest);

if (answer == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)pData);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::SEND);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}

return answer;
}

int TlCommunicate::iSendDataX(const int* pData, const std::size_t size,
const int dest, const int tag) {
return this->iSendDataX(pData, MPI_INT, 0, size, dest, tag);
}

int TlCommunicate::iSendDataX(const double* pData, const std::size_t size,
const int dest, const int tag) {
return this->iSendDataX(pData, MPI_DOUBLE, 0, size, dest, tag);
}

int TlCommunicate::iSendDataX(const TlMatrixObject::MatrixElement* pData,
const std::size_t size, const int dest,
const int tag) {
return this->iSendDataX(pData, this->MPI_MATRIXELEMENT, 0, size, dest, tag);
}

int TlCommunicate::iSendDataX(const TlVectorObject::VectorElement* pData,
const std::size_t size, const int dest,
const int tag) {
return this->iSendDataX(pData, this->MPI_VECTORELEMENT, 0, size, dest, tag);
}

template <typename T>
int TlCommunicate::iReceiveData(T& data, const MPI_Datatype mpiType,
const int src, const int tag) {
MPI_Request* pRequest = new MPI_Request;
int err =
MPI_Irecv((void*)&data, 1, mpiType, src, tag, MPI_COMM_WORLD, pRequest);
if (err == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)&data);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::RECV);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}

return err;
}

int TlCommunicate::iReceiveData(int& data, const int src, const int tag) {
return this->iReceiveData(data, MPI_INT, src, tag);
}

int TlCommunicate::iReceiveData(unsigned int& data, const int src,
const int tag) {
return this->iReceiveData(data, MPI_UNSIGNED, src, tag);
}

int TlCommunicate::iReceiveData(long& data, const int src, const int tag) {
return this->iReceiveData(data, MPI_LONG, src, tag);
}

int TlCommunicate::iReceiveData(unsigned long& data, const int src,
const int tag) {
return this->iReceiveData(data, MPI_UNSIGNED_LONG, src, tag);
}

template <typename T>
int TlCommunicate::iReceiveData(std::vector<T>& data,
const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int src, const int tag) {
int answer = 0;
std::vector<uintptr_t> requests;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
std::vector<T> buf(bufCount);
const ldiv_t tmp = std::ldiv(length, bufCount);

if (tmp.quot != 0) {
for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
MPI_Request* pRequest = new MPI_Request;
answer = MPI_Irecv(&(data[startIndex]), bufCount, mpiType, src,
tag, MPI_COMM_WORLD, pRequest);
requests.push_back(
reinterpret_cast<uintptr_t>((void*)pRequest));
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
MPI_Request* pRequest = new MPI_Request;
answer = MPI_Irecv(&(data[startIndex]), remain, mpiType, src, tag,
MPI_COMM_WORLD, pRequest);
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
MPI_Request* pRequest = new MPI_Request;
answer = MPI_Irecv(&(data[0]), end - start, mpiType, src, tag,
MPI_COMM_WORLD, pRequest);
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
}
#endif  

const uintptr_t key = reinterpret_cast<uintptr_t>((void*)&(data[start]));
const NonBlockingCommParam param(requests, tag, NonBlockingCommParam::RECV);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }

return answer;
}

int TlCommunicate::iReceiveData(std::vector<int>& data, const int src,
const int tag) {
std::size_t size = 0;
this->iReceiveData(size, src, tag);
this->wait(size);
data.resize(size);

return this->iReceiveData(data, MPI_INT, 0, size, src, tag);
}

int TlCommunicate::iReceiveData(std::vector<double>& data, int src, int tag) {
std::size_t size = 0;
this->iReceiveData(size, src, tag);
this->wait(size);
data.resize(size);

return this->iReceiveData(data, MPI_DOUBLE, 0, size, src, tag);
}






template <typename T>
int TlCommunicate::iReceiveDataX(T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
const int src, const int tag) {
MPI_Request* pRequest = new MPI_Request;
const int answer = MPI_Irecv((void*)(pData + start), (end - start), mpiType,
src, tag, MPI_COMM_WORLD, pRequest);

if (answer == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)pData);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::RECV);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}

return answer;
}

int TlCommunicate::iReceiveDataX(int* pData, const std::size_t size,
const int src, const int tag) {
return this->iReceiveDataX(pData, MPI_INT, 0, size, src, tag);
}

int TlCommunicate::iReceiveDataX(double* pData, const std::size_t size,
const int src, const int tag) {
return this->iReceiveDataX(pData, MPI_DOUBLE, 0, size, src, tag);
}


int TlCommunicate::iReceiveDataFromAnySource(int& data, const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_INT, tag);
}

int TlCommunicate::iReceiveDataFromAnySource(unsigned int& data,
const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_UNSIGNED, tag);
}

int TlCommunicate::iReceiveDataFromAnySource(long& data, const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_LONG, tag);
}

int TlCommunicate::iReceiveDataFromAnySource(unsigned long& data,
const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_UNSIGNED_LONG, tag);
}

template <typename T>
int TlCommunicate::iReceiveDataFromAnySource(T& data,
const MPI_Datatype mpiType,
const int tag) {
return this->iReceiveData(data, mpiType, MPI_ANY_SOURCE, tag);
}

int TlCommunicate::iReceiveDataFromAnySource(std::vector<int>& data,
const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_INT, tag);
}

int TlCommunicate::iReceiveDataFromAnySource(std::vector<double>& data,
const int tag) {
return this->iReceiveDataFromAnySource(data, MPI_DOUBLE, tag);
}

template <typename T>
int TlCommunicate::iReceiveDataFromAnySource(std::vector<T>& data,
const MPI_Datatype mpiType,
const int tag) {
std::size_t size = 0;
this->iReceiveDataFromAnySource(size, tag);
int src = 0;
this->wait(size, &src);
data.resize(size);

return this->iReceiveData(data, mpiType, 0, size, src, tag);
}

template <typename T>
int TlCommunicate::iReceiveDataFromAnySourceX(T* pData,
const MPI_Datatype mpiType,
const std::size_t start,
const std::size_t end,
const int tag) {
int answer = 0;


MPI_Request* pRequest = new MPI_Request;
answer = MPI_Irecv((void*)(pData + start), (end - start), mpiType,
MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, pRequest);

if (answer == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)pData);
const NonBlockingCommParam param(requests, tag,
NonBlockingCommParam::RECV);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}

return answer;
}

int TlCommunicate::iReceiveDataFromAnySourceX(int* pData,
const std::size_t size,
const int tag) {
return this->iReceiveDataFromAnySourceX(pData, MPI_INT, 0, size, tag);
}

template <typename T>
int TlCommunicate::iReceiveDataFromAnySourceAnyTag(T* pData,
const MPI_Datatype mpiType,
const int start,
const int end) {
int answer = 0;

MPI_Request* pRequest = new MPI_Request;
answer = MPI_Irecv((void*)(pData + start), (end - start), mpiType,
MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, pRequest);

if (answer == 0) {
std::vector<uintptr_t> requests;
requests.push_back(reinterpret_cast<uintptr_t>((void*)pRequest));
const uintptr_t key = reinterpret_cast<uintptr_t>((void*)pData);
const NonBlockingCommParam param(requests, MPI_ANY_TAG,
NonBlockingCommParam::RECV);
this->checkNonBlockingTableCollision(key, param, __LINE__);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{ this->nonBlockingCommParamTable_[key] = param; }
}

return answer;
}

int TlCommunicate::iReceiveDataFromAnySourceAnyTag(double* pData,
const int count) {
return this->iReceiveDataFromAnySourceAnyTag<double>(pData, MPI_DOUBLE, 0,
count);
}

bool TlCommunicate::cancel(void* pData) {
bool answer = false;

uintptr_t key = reinterpret_cast<uintptr_t>(pData);
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{
NonBlockingCommParamTableType::iterator it =
this->nonBlockingCommParamTable_.find(key);
if (it != this->nonBlockingCommParamTable_.end()) {
bool isComplete = true;
std::vector<uintptr_t>::iterator reqEnd = it->second.requests.end();
for (std::vector<uintptr_t>::iterator req =
it->second.requests.begin();
req != reqEnd; ++req) {
MPI_Request* pRequest = reinterpret_cast<MPI_Request*>(*req);
const int retval = MPI_Cancel(pRequest);
if (retval != MPI_SUCCESS) {
isComplete = false;
break;
}
}
answer = isComplete;
} else {
std::cerr
<< "PROGRAM ERROR(TlCommunicate::cancel()): unknown key. file:"
<< __FILE__ << ", l." << __LINE__ << std::endl;
std::abort();
}
}
if (answer == true) {
this->wait(pData);
}

return answer;
}

bool TlCommunicate::test(void* pData, int* pSrc) {
this->time_test_.start();
++(this->counter_test_);

bool answer = false;

uintptr_t key = reinterpret_cast<uintptr_t>(pData);

NonBlockingCommParamTableType::iterator it;
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{
it = this->nonBlockingCommParamTable_.find(key);
if (it != this->nonBlockingCommParamTable_.end()) {
const NonBlockingCommParam& param = it->second;
if ((param.property & NonBlockingCommParam::COMPLETE) != 0) {
answer = true;
if (pSrc != NULL) {
*pSrc = param.source;
}
} else {
bool isComplete = true;
int source = -1;

const std::size_t numOfRequests = param.requests.size();
assert(numOfRequests == param.requestStates.size());
for (std::size_t i = 0; i < numOfRequests; ++i) {
const unsigned int state = param.requestStates[i];
if (state == 0) {
MPI_Request* pRequest =
reinterpret_cast<MPI_Request*>(param.requests[i]);
int flag = 0;
MPI_Status status;
MPI_Test(pRequest, &flag, &status);
if (flag == 0) {
isComplete = false;
} else {
it->second.requestStates[i] = 1;
source = status.MPI_SOURCE;
}
}
}

if (isComplete == true) {
it->second.property |= NonBlockingCommParam::COMPLETE;
it->second.source = source;
if (pSrc != NULL) {
*pSrc = source;
}
}
answer = isComplete;
}
} else {
std::cerr
<< "PROGRAM ERROR(TlCommunicate::test()): unknown key. file:"
<< __FILE__ << ", l." << __LINE__ << std::endl;
std::abort();
}
}

this->time_test_.stop();
return answer;
}

int TlCommunicate::wait(void* pData, int* pSrc, int* pTag) {
this->time_wait_.start();
++(this->counter_wait_);

int answer = 0;

uintptr_t key = reinterpret_cast<uintptr_t>(pData);
MPI_Status status;

NonBlockingCommParamTableType::iterator it;
#pragma omp critical(TlCommunicate_nonBlockingCommParamTable_update)
{
it = this->nonBlockingCommParamTable_.find(key);
if (it != this->nonBlockingCommParamTable_.end()) {
std::vector<uintptr_t>::iterator reqEnd = it->second.requests.end();
for (std::vector<uintptr_t>::iterator req =
it->second.requests.begin();
req != reqEnd; ++req) {
MPI_Request* pRequest = reinterpret_cast<MPI_Request*>(*req);

int err = MPI_Wait(pRequest, &status);
if (err != 0) {
std::cerr << " MPI wait error. " << __FILE__ << ":"
<< __LINE__ << " err=" << err << std::endl;
}
answer |= err;

delete pRequest;
pRequest = NULL;
}

this->nonBlockingCommParamTable_.erase(it);

if (pSrc != NULL) {
*pSrc = status.MPI_SOURCE;
}

if (pTag != NULL) {
*pTag = status.MPI_TAG;
}

} else {
this->log_.critical(
TlUtils::format("TlCommunicate::wait(): cannot find: %ld",
this->getRank(), key));
}
}

this->time_wait_.stop();
return answer;
}

TlCommunicate::TlCommunicate()
: workMemSize_(DEFAULT_WORK_MEM_SIZE), log_(TlLogging::getInstance()) {}

TlCommunicate::TlCommunicate(const TlCommunicate& rhs)
: log_(TlLogging::getInstance()) {}

TlCommunicate::~TlCommunicate() {
(void)this->finalize();
}

int TlCommunicate::initialize(int argc, char* argv[]) {
int nAnswer = MPI_Init(&argc, &argv);

MPI_Comm_size(MPI_COMM_WORLD, &(this->m_nProc));
MPI_Comm_rank(MPI_COMM_WORLD, &(this->m_nRank));

this->register_MatrixElement();
this->register_VectorElement();

this->counter_barrier_ = 0;
this->counter_test_ = 0;
this->counter_wait_ = 0;
this->counter_allreduce_ = 0;
this->time_barrier_.stop();
this->time_test_.stop();
this->time_wait_.stop();
this->time_allreduce_.stop();
this->time_barrier_.reset();
this->time_test_.reset();
this->time_wait_.reset();
this->time_allreduce_.reset();

return nAnswer;
}

int TlCommunicate::finalize() {
this->unregister_MatrixElement();
this->unregister_VectorElement();

TlScalapackContext::finalize();

const int retcode = MPI_Finalize();


return retcode;
}

int TlCommunicate::abort(const int errorCode) {
int answer = MPI_Abort(MPI_COMM_WORLD, errorCode);
this->finalize();
return answer;
}

int TlCommunicate::getNumOfProcs() const {
return this->m_nProc;
}

int TlCommunicate::getRank() const {
return this->m_nRank;
}


bool TlCommunicate::isMaster() const {
return (this->m_nRank == 0);
}

bool TlCommunicate::isSlave() const {
return !(this->isMaster());
}

void TlCommunicate::register_MatrixElement() {
const int numOfItems = 3;
int blocklengths[3] = {1, 1, 1};
MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
MPI_Aint offsets[3];
offsets[0] = offsetof(struct TlMatrixObject::MatrixElement, row);
offsets[1] = offsetof(struct TlMatrixObject::MatrixElement, col);
offsets[2] = offsetof(struct TlMatrixObject::MatrixElement, value);

int err = MPI_Type_create_struct(numOfItems, blocklengths, offsets, types,
&(this->MPI_MATRIXELEMENT));

if (err != MPI_SUCCESS) {
this->log_.critical("cannot register MPI_MATRIXELEMENT.");
}

MPI_Type_commit(&(this->MPI_MATRIXELEMENT));
}

void TlCommunicate::unregister_MatrixElement() {
MPI_Type_free(&(this->MPI_MATRIXELEMENT));
}

void TlCommunicate::register_VectorElement() {
const int numOfItems = 2;
int blocklengths[2] = {1, 1};
MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
MPI_Aint offsets[2];
offsets[0] = offsetof(struct TlVectorObject::VectorElement, index);
offsets[1] = offsetof(struct TlVectorObject::VectorElement, value);

int err = MPI_Type_create_struct(numOfItems, blocklengths, offsets, types,
&(this->MPI_VECTORELEMENT));

if (err != MPI_SUCCESS) {
this->log_.critical("cannot register MPI_VECTORELEMENT.");
}

MPI_Type_commit(&(this->MPI_VECTORELEMENT));
}

void TlCommunicate::unregister_VectorElement() {
MPI_Type_free(&(this->MPI_VECTORELEMENT));
}

int TlCommunicate::broadcast(bool& rData) {
int nData = 0;

if (this->isMaster() == true) {
nData = (rData == true) ? 1 : 0;
}

int nError = this->broadcast(nData);

rData = ((nData == 1) ? true : false);

return nError;
}

int TlCommunicate::broadcast(int& data, int root) {
return this->broadcast(data, MPI_INT, root);
}

int TlCommunicate::broadcast(unsigned int& data, int root) {
return this->broadcast(data, MPI_UNSIGNED, root);
}

int TlCommunicate::broadcast(long& data, int root) {
return this->broadcast(data, MPI_LONG, root);
}

int TlCommunicate::broadcast(unsigned long& data, int root) {
return this->broadcast(data, MPI_UNSIGNED_LONG, root);
}

int TlCommunicate::broadcast(double& data, int root) {
return this->broadcast(data, MPI_DOUBLE, root);
}

int TlCommunicate::broadcast(std::string& rData) {
int nCount = 0;
if (this->isMaster() == true) {
nCount = rData.size();
}
this->broadcast(nCount);

char* pBuffer = new char[nCount];
if (this->isMaster() == true) {
rData.copy(pBuffer, nCount);
}
const int nRoot = 0;

int nError = MPI_Bcast(pBuffer, nCount, MPI_CHAR, nRoot, MPI_COMM_WORLD);

rData = std::string(pBuffer, nCount);

delete[] pBuffer;
pBuffer = NULL;

return nError;
}

int TlCommunicate::broadcast(std::vector<int>& rData, int nRoot) {
std::size_t size = 0;
if (this->getRank() == nRoot) {
size = rData.size();
}
this->broadcast(size, nRoot);
rData.resize(size);

return this->broadcast(rData, MPI_INT, 0, size, nRoot);
}

int TlCommunicate::broadcast(std::vector<long>& data, int root) {
std::size_t size = 0;
if (this->isMaster() == true) {
size = data.size();
}
this->broadcast(size);
data.resize(size);

return this->broadcast(data, MPI_LONG, 0, size, root);
}

int TlCommunicate::broadcast(std::vector<unsigned long>& rData,
const int nRoot) {
std::size_t size = 0;
if (this->getRank() == nRoot) {
size = rData.size();
}
this->broadcast(size, nRoot);
rData.resize(size);

return this->broadcast(rData, MPI_UNSIGNED_LONG, 0, size, nRoot);
}

int TlCommunicate::broadcast(std::vector<double>& rData, const int nRoot) {
std::size_t size = 0;
if (this->getRank() == nRoot) {
size = rData.size();
}
this->broadcast(size, nRoot);
rData.resize(size);

return this->broadcast(rData, MPI_DOUBLE, 0, size, nRoot);
}

int TlCommunicate::broadcast(std::valarray<double>& rData, const int nRoot) {
std::size_t size = 0;
if (this->getRank() == nRoot) {
size = rData.size();
}
int answer = this->broadcast(size, nRoot);

if (this->getRank() != nRoot) {
rData.resize(size);
}

if (answer == 0) {
answer = this->broadcast(rData, MPI_DOUBLE, 0, size, nRoot);
}

return answer;
}

int TlCommunicate::broadcast(std::vector<std::string>& rData) {
int nArraySize = 0;
if (this->isMaster() == true) {
nArraySize = rData.size();
}
this->broadcast(nArraySize);
rData.resize(nArraySize);

for (int i = 0; i < nArraySize; ++i) {
std::string str = "";
if (this->isMaster() == true) {
str = rData[i];
}
this->broadcast(str);

rData[i] = str;
}

return 0;
}

#ifdef HAVE_LAPACK
int TlCommunicate::broadcast(TlDenseGeneralMatrix_Lapack* pMatrix,
const int root) {
std::vector<TlMatrixObject::index_type> rowcol(2);

if (this->getRank() == root) {
rowcol[0] = pMatrix->getNumOfRows();
rowcol[1] = pMatrix->getNumOfCols();
}
int answer = 0;
answer = this->broadcast(rowcol, root);

if (answer == 0) {
if (this->getRank() != root) {
pMatrix->resize(rowcol[0], rowcol[1]);
}
this->broadcast(pMatrix->data(), MPI_DOUBLE, 0,
pMatrix->getNumOfElements(), root);
}

return answer;
}

int TlCommunicate::broadcast(TlDenseSymmetricMatrix_Lapack* pMatrix, int root) {
const int myRank = this->getRank();
TlMatrixObject::index_type dim = 0;

if (myRank == root) {
dim = pMatrix->getNumOfRows();
}
int answer = this->broadcast(dim, root);

std::vector<double> buf;
if (answer == 0) {
if (myRank == root) {
buf = *pMatrix;
}
pMatrix->resize(dim);
answer = this->broadcast(buf, root);

if (myRank != root) {
*pMatrix = TlDenseSymmetricMatrix_Lapack(dim, buf.data());
}
}

return answer;
}

int TlCommunicate::broadcast(TlDenseVector_Lapack* pMatrix, int root) {
return this->broadcast_denseVector(pMatrix, root);
}
#endif  

#ifdef HAVE_EIGEN
int TlCommunicate::broadcast(TlDenseGeneralMatrix_Eigen* pMatrix,
const int root) {
const int myRank = this->getRank();

std::vector<TlMatrixObject::index_type> rowcol(2);
if (this->getRank() == root) {
rowcol[0] = pMatrix->getNumOfRows();
rowcol[1] = pMatrix->getNumOfCols();
}
int answer = 0;
answer = this->broadcast(rowcol, root);

if (answer == 0) {
std::vector<double> buf;
if (myRank == root) {
buf = *pMatrix;
} else {
pMatrix->resize(rowcol[0], rowcol[1]);
}

answer = this->broadcast(buf, root);

if (myRank != root) {
*pMatrix =
TlDenseGeneralMatrix_Eigen(rowcol[0], rowcol[1], buf.data());
}
}

return answer;
}

int TlCommunicate::broadcast(TlDenseSymmetricMatrix_Eigen* pMatrix, int root) {
const int myRank = this->getRank();
TlMatrixObject::index_type dim = 0;

if (this->getRank() == root) {
dim = pMatrix->getNumOfRows();
}
int answer = this->broadcast(dim, root);

std::vector<double> buf;
if (answer == 0) {
if (myRank == root) {
buf = *pMatrix;
}
answer = this->broadcast(buf, root);

if (myRank != root) {
*pMatrix = TlDenseSymmetricMatrix_Eigen(dim, buf.data());
}
}

return answer;
}

int TlCommunicate::broadcast(TlDenseVector_Eigen* pMatrix, int root) {
return this->broadcast_denseVector(pMatrix, root);
}
#endif  

int TlCommunicate::broadcast(int* pBuf, const std::size_t size, int root) {
return this->broadcast(pBuf, MPI_INT, 0, size, root);
}

int TlCommunicate::broadcast(double* pBuf, const std::size_t size, int root) {
return this->broadcast(pBuf, MPI_DOUBLE, 0, size, root);
}

int TlCommunicate::broadcast(TlSerializeData& data) {
std::string buf = "";
if (this->isMaster() == true) {
TlMsgPack mpac(data);
buf = mpac.dump();
}

int ans = this->broadcast(buf);

if ((ans == 0) && (this->isMaster() != true)) {
TlMsgPack mpac;
mpac.pack(buf);
data = mpac.getSerializeData();
}

return ans;
}

template <typename T>
int TlCommunicate::broadcast(T& data, const MPI_Datatype mpiType,
const int root) {
this->log_.debug(
TlUtils::format("TlCommunicate::broadcast(): type=%s, root=%d",
TlUtils::xtos(mpiType).c_str(), root));
return MPI_Bcast(&data, 1, mpiType, root, MPI_COMM_WORLD);
}

template <typename T>
int TlCommunicate::broadcast(std::vector<T>& data, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
int root) {
int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ /
sizeof(T));  
std::vector<T> buf(bufCount);
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Bcast(&(data[startIndex]), bufCount, mpiType, root,
MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Bcast(&(data[startIndex]), remain, mpiType, root,
MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer =
MPI_Bcast(&(data[0]), end - start, mpiType, root, MPI_COMM_WORLD);
}
#endif  

return answer;
}

template <typename T>
int TlCommunicate::broadcast(std::valarray<T>& data, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
int root) {
int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount = static_cast<int>(this->workMemSize_ / sizeof(T));
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
const std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Bcast((void*)(&data[startIndex]), bufCount, mpiType,
root, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
const std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Bcast((void*)(&data[startIndex]), remain, mpiType,
root, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Bcast((void*)(&data[0]), end - start, mpiType, root,
MPI_COMM_WORLD);
}
#endif  

return answer;
}

template <typename T>
int TlCommunicate::broadcast(T* pData, const MPI_Datatype mpiType,
const std::size_t start, const std::size_t end,
int root) {
int answer = 0;

#ifdef DIV_COMM
{
const long length = static_cast<long>(end - start);
const int bufCount =
static_cast<int>(this->workMemSize_ / sizeof(double));
const ldiv_t tmp = std::ldiv(length, bufCount);

for (long i = 0; i < tmp.quot; ++i) {
const std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * i);
answer = MPI_Bcast((void*)(pData + startIndex), bufCount, mpiType,
root, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}

if (tmp.rem != 0) {
const int remain = tmp.rem;
const std::size_t startIndex =
start + static_cast<std::size_t>(bufCount * tmp.quot);
answer = MPI_Bcast((void*)(pData + startIndex), remain, mpiType,
root, MPI_COMM_WORLD);
if (answer != 0) {
std::cerr << " MPI error. " << __FILE__ << ":" << __LINE__
<< " anwer=" << answer << std::endl;
}
}
}
#else
{
answer = MPI_Bcast((void*)(pData), end - start, mpiType, root,
MPI_COMM_WORLD);
}
#endif  

return answer;
}

template <class DenseGeneralMatrix>
int TlCommunicate::broadcast_denseGeneralMatrix(DenseGeneralMatrix* pMatrix,
const int root) {
const int myRank = this->getRank();
TlMatrixObject::index_type rowcol[2];

if (myRank == root) {
rowcol[0] = pMatrix->getNumOfRows();
rowcol[1] = pMatrix->getNumOfCols();
}
int answer = 0;
answer = this->broadcast(rowcol, 2, root);

const std::size_t row = rowcol[0];
const std::size_t col = rowcol[1];
const std::size_t bufSize = row * col;
std::vector<double> buf(bufSize);
if (answer == 0) {
if (myRank == root) {
buf = *pMatrix;
}

answer = this->broadcast(&(buf[0]), bufSize, root);

if (myRank != root) {
*pMatrix = DenseGeneralMatrix(row, col, buf);
}
}

return answer;
}

template <class DenseSymmetricMatrix>
int TlCommunicate::broadcast_denseSymmetricMatrix(DenseSymmetricMatrix* pMatrix,
int root) {
const int myRank = this->getRank();
TlMatrixObject::index_type rowcol;

if (myRank == root) {
rowcol = pMatrix->getNumOfRows();
}
int answer = this->broadcast(rowcol, root);

const std::size_t row = rowcol;
const std::size_t bufSize = row * (row + 1) / 2;
std::vector<double> buf(bufSize);
if (answer == 0) {
if (myRank == root) {
buf = *pMatrix;
}

answer = this->broadcast(&(buf[0]), bufSize, root);

if (myRank != root) {
*pMatrix = DenseSymmetricMatrix(row, buf.data());
}
}

return answer;
}

template <class DenseVector>
int TlCommunicate::broadcast_denseVector(DenseVector* pVector, int root) {
const int myRank = this->getRank();
TlDenseVectorObject::index_type dim = 0;

if (myRank == root) {
dim = pVector->getSize();
}
int answer = this->broadcast(dim, root);

std::vector<double> buf;
if (answer == 0) {
if (myRank == root) {
buf = std::vector<double>(*pVector);
}
pVector->resize(dim);
answer = this->broadcast(buf, root);

if (myRank != root) {
*pVector = DenseVector(buf);
}
}

return answer;
}

