







#ifndef FF_STENCILREDUCE_OCL_HPP
#define FF_STENCILREDUCE_OCL_HPP

#ifdef FF_OPENCL

#include <string>
#include <fstream>
#include <tuple>
#include <algorithm>
#include <ff/bitflags.hpp>
#include <ff/oclnode.hpp>
#include <ff/node.hpp>
#include <ff/oclallocator.hpp>
#include <ff/stencilReduceOCL_macros.hpp>

namespace ff {

enum reduceMode { REDUCE_INPUT, REDUCE_OUTPUT };



template<typename TaskT_, typename Tin_, typename Tout_ = Tin_>
class baseOCLTask {
public:
typedef TaskT_  TaskT;
typedef Tin_    Tin;
typedef Tout_   Tout;

baseOCLTask(): inPtr(NULL),outPtr(NULL),reduceVar(NULL),
size_in(0),size_out(0),iter(0),
tuple_in(std::make_tuple(true,false,false)),
tuple_out(std::make_tuple(true,false,false))   { }   
virtual ~baseOCLTask() { }

virtual void setTask(TaskT *t) = 0;



virtual void   releaseTask(TaskT *t) {}
virtual bool   iterCondition(const Tout&, size_t) { return false; } 
virtual Tout   combinator(const Tout&, const Tout&)  { return Tout(); }
virtual void   incIter()                     { ++iter; }
virtual size_t getIter() const               { return iter; }
virtual void   resetIter(const size_t val=0) { iter = val; } 



void resetTask() {
envPtr.resize(0);
copyEnv.resize(0);
}


void setInPtr(Tin* _inPtr, size_t sizeIn, 
const CopyFlags    copy   =CopyFlags::COPY, 
const ReuseFlags   reuse  =ReuseFlags::DONTREUSE, 
const ReleaseFlags release=ReleaseFlags::DONTRELEASE)  { 
inPtr  = _inPtr; size_in = sizeIn; 
tuple_in = std::make_tuple(copy==CopyFlags::COPY,
reuse==ReuseFlags::REUSE,
release==ReleaseFlags::RELEASE);
}


void setInPtr(Tin* _inPtr, size_t sizeIn, const MemoryFlags &flags) { 
inPtr  = _inPtr; size_in = sizeIn; 
tuple_in = std::make_tuple(flags.copy==CopyFlags::COPY,
flags.reuse==ReuseFlags::REUSE,
flags.release==ReleaseFlags::RELEASE);
}


void setOutPtr(Tout* _outPtr, size_t sizeOut, 
const CopyFlags copyback    =CopyFlags::COPY, 
const ReuseFlags reuse      =ReuseFlags::DONTREUSE, 
const ReleaseFlags release  =ReleaseFlags::DONTRELEASE)  { 
outPtr = _outPtr; size_out = sizeOut; 
tuple_out = std::make_tuple(copyback==CopyFlags::COPY,
reuse==ReuseFlags::REUSE,
release==ReleaseFlags::RELEASE);
}


void setOutPtr(Tout* _outPtr, size_t sizeOut, const MemoryFlags &flags ) { 
outPtr = _outPtr; size_out = sizeOut; 
tuple_out = std::make_tuple(flags.copy==CopyFlags::COPY,
flags.reuse==ReuseFlags::REUSE,
flags.release==ReleaseFlags::RELEASE);
}


template<typename ptrT>
void setEnvPtr(const ptrT* _envPtr, size_t size, 
const CopyFlags copy   =CopyFlags::COPY, 
const ReuseFlags reuse  =ReuseFlags::DONTREUSE, 
const ReleaseFlags release=ReleaseFlags::DONTRELEASE)  { 
assert(envPtr.size() == copyEnv.size());
envPtr.push_back(std::make_pair((void*)_envPtr,size*sizeof(ptrT)));
copyEnv.push_back(std::make_tuple(sizeof(ptrT), 
copy==CopyFlags::COPY,
reuse==ReuseFlags::REUSE,
release==ReleaseFlags::RELEASE));                   
}


template<typename ptrT>
void setEnvPtr(const ptrT* _envPtr, size_t size, const MemoryFlags &flags) { 
assert(envPtr.size() == copyEnv.size());
envPtr.push_back(std::make_pair((void*)_envPtr,size*sizeof(ptrT)));
copyEnv.push_back(std::make_tuple(sizeof(ptrT), 
flags.copy==CopyFlags::COPY,
flags.reuse==ReuseFlags::REUSE,
flags.release==ReleaseFlags::RELEASE));
}

Tin *   getInPtr()    const { return inPtr;  }
Tout *  getOutPtr()   const { return outPtr; }
template<typename ptrT>
void getEnvPtr(const size_t idx, ptrT *& ptr)  const { 
assert(idx < envPtr.size());
ptr = reinterpret_cast<ptrT*>(envPtr[idx].first);
}

size_t  getEnvNum() const { 
assert(envPtr.size() == copyEnv.size());
return envPtr.size();
}

bool  getCopyEnv(const size_t idx) const { 
assert(idx < copyEnv.size());
return std::get<1>(copyEnv[idx]);
}
bool  getReuseEnv(const size_t idx) const { 
assert(idx < copyEnv.size());
return std::get<2>(copyEnv[idx]);
}
bool  getReleaseEnv(const size_t idx) const { 
assert(idx < copyEnv.size());
return std::get<3>(copyEnv[idx]);
}

bool getCopyIn()     const { return std::get<0>(tuple_in); }
bool getReuseIn()    const { return std::get<1>(tuple_in); }
bool getReleaseIn()  const { return std::get<2>(tuple_in); }

bool getCopyOut()    const { return std::get<0>(tuple_out); }
bool getReuseOut()   const { return std::get<1>(tuple_out); }
bool getReleaseOut() const { return std::get<2>(tuple_out); }


size_t getSizeIn()   const { return size_in;   }
size_t getSizeOut()  const { return (size_out==0)?size_in:size_out;  }
size_t getSizeEnv(const size_t idx)  const { 
assert(idx < copyEnv.size());
return std::get<0>(copyEnv[idx]);
}

size_t getBytesizeIn()    const { return getSizeIn() * sizeof(Tin); }
size_t getBytesizeOut()   const { return getSizeOut() * sizeof(Tout); }
size_t getBytesizeEnv(const size_t idx)  const { 
assert(idx < envPtr.size());
return envPtr[idx].second;
}

void  setReduceVar(const Tout *r) { reduceVar = (Tout*)r;  } 
Tout *getReduceVar() const { return reduceVar;  }
void  writeReduceVar(const Tout &r) { *reduceVar = r; }    

void  setIdentityVal(const Tout &x) { identityVal = x;} 
Tout  getIdentityVal() const        { return identityVal; }

bool  iterCondition_aux() { 
return iterCondition(*reduceVar, iter);   
}           

protected:
Tin     *inPtr;
Tout    *outPtr;
Tout    *reduceVar, identityVal;
size_t   size_in, size_out, iter;
std::tuple<bool,bool,bool> tuple_in;
std::tuple<bool,bool,bool> tuple_out;

std::vector<std::pair<void*,size_t> > envPtr;             
std::vector<std::tuple<size_t,bool,bool,bool> > copyEnv;  
};




template<typename T, typename TOCL = T>
class ff_oclAccelerator {
public:
typedef typename TOCL::Tin  Tin;
typedef typename TOCL::Tout Tout;

ff_oclAccelerator(ff_oclallocator *alloc, const size_t width_, const Tout &identityVal, const bool from_source=false) :
from_source(from_source), my_own_allocator(false), allocator(alloc), halo_half(width_), identityVal(identityVal), events_h2d(16), deviceId(NULL) {
wgsize_map_static = wgsize_reduce_static = 0;
wgsize_map_max = wgsize_reduce_max = 0;
inputBuffer = outputBuffer = reduceBuffer = NULL;

sizeInput = sizeInput_padded = 0;
lenInput = offset1_in = halo_in_left = halo_in_right = lenInput_global = 0;

sizeOutput = sizeOutput_padded = 0;
lenOutput = offset1_out = halo_out_left = halo_out_right = lenOutput_global = 0;

nevents_h2d = nevents_map = 0;
event_d2h = event_map = event_reduce1 = event_reduce2 = NULL;
wgsize_map = nthreads_map = 0;
wgsize_reduce = nthreads_reduce = nwg_reduce = wg_red_mem = 0;
reduceVar = identityVal;
kernel_map = kernel_reduce = kernel_init = NULL;
context = NULL;
program = NULL;
cmd_queue = NULL;

reduce_mode = REDUCE_OUTPUT;

if (!allocator) {
my_own_allocator = true;
allocator = new ff_oclallocator;
assert(allocator);
}
}

virtual ~ff_oclAccelerator() {
if (my_own_allocator) {
allocator->releaseAllBuffers(context);
delete allocator;
allocator = NULL;
my_own_allocator = false;
}
}

int init(cl_device_id dId, reduceMode m, const std::string &kernel_code, const std::string &kernel_name1,
const std::string &kernel_name2, const bool save_binary, const bool reuse_binary) {
#ifdef FF_OPENCL_LOG
fprintf(stderr, "initializing virtual accelerator @%p mapped to device:\n", this);
std::cerr << ff::clEnvironment::instance()->getDeviceInfo(dId) << std::endl;
#endif
reduce_mode = m;
deviceId = dId;
const oclParameter *param = clEnvironment::instance()->getParameter(deviceId);
assert(param);
context    = param->context;
cmd_queue  = param->commandQueue;
cl_int status = buildKernels(kernel_code, kernel_name1,kernel_name2, from_source, save_binary, reuse_binary);
checkResult(status, "build kernels");
setSizingHeuristics();
return status == CL_SUCCESS;
}

void releaseAll() { if (deviceId) { svc_releaseOclObjects(); deviceId = NULL; }}
void releaseInput(const Tin *inPtr)     { 
if (allocator->releaseBuffer(inPtr, context, inputBuffer) != CL_SUCCESS)
checkResult(CL_INVALID_MEM_OBJECT, "releaseInput");
inputBuffer = NULL;
}
void releaseOutput(const Tout *outPtr)  {
if (allocator->releaseBuffer(outPtr, context, outputBuffer) != CL_SUCCESS)
checkResult(CL_INVALID_MEM_OBJECT, "releaseOutput");
outputBuffer = NULL;
}
void releaseEnv(size_t idx, const void *envPtr) {
if (allocator->releaseBuffer(envPtr, context, envBuffer[idx].first) != CL_SUCCESS)
checkResult(CL_INVALID_MEM_OBJECT, "releaseEnv");
envBuffer[idx].first = NULL, envBuffer[idx].second = 0;
}

void swapBuffers() {
cl_mem tmp = inputBuffer;
inputBuffer = outputBuffer;
outputBuffer = tmp;
}

void setSizingHeuristics() {
cl_int status;
size_t max_device_wgsize;
status = clGetDeviceInfo(deviceId,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_device_wgsize),&max_device_wgsize, NULL);
checkResult(status, "clGetDeviceInfo (map)");
if(kernel_map) { 
size_t max_kernel_wgsize;
status = clGetKernelWorkGroupInfo(kernel_map, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_kernel_wgsize, 0);
checkResult(status, "GetKernelWorkGroupInfo (map)");
wgsize_map_max = std::min<size_t>(max_device_wgsize,max_kernel_wgsize);
size_t wg_multiple;
status = clGetKernelWorkGroupInfo(kernel_map, deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &wg_multiple, 0);
wgsize_map_static = std::max<size_t>(64, wg_multiple * 4); 
wgsize_map_static = std::min<size_t>(wgsize_map_static,wgsize_map_max);
}
if(kernel_reduce) { 
size_t max_kernel_wgsize;
status = clGetKernelWorkGroupInfo(kernel_reduce, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_kernel_wgsize, 0);
checkResult(status, "GetKernelWorkGroupInfo (reduce)");
wgsize_reduce_max = std::min<size_t>(max_device_wgsize,max_kernel_wgsize);
size_t wg_multiple;
status = clGetKernelWorkGroupInfo(kernel_reduce, deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &wg_multiple, 0);
wgsize_reduce_static = std::max<size_t>(64, wg_multiple * 4); 
wgsize_reduce_static = std::min<size_t>(wgsize_reduce_static,wgsize_reduce_max);
}
#ifdef FF_OPENCL_LOG
std::cerr <<  "[virtual accelerator @"<<this<<"]\n";
std::cerr <<  "+ static heuristics for kernel sizing parameters:\n";
std::cerr <<  "- MAP workgroup-size = " <<wgsize_map_static<< "\n";
std::cerr <<  "- MAP max workgroup-size = " <<wgsize_map_max<< "\n";
std::cerr <<  "- RED workgroup-size = " <<wgsize_reduce_static<< " \n";
std::cerr <<  "- RED max workgroup-size = " <<wgsize_reduce_max<< " \n";
#endif
}

void adjustInputBufferOffset(const Tin *newPtr, const Tin *oldPtr, std::pair<size_t, size_t> &P, size_t len_global) {
offset1_in = P.first;
lenInput   = P.second;
lenInput_global = len_global;
halo_in_left = (std::min)(halo_half, offset1_in);
halo_in_right = (std::min)(halo_half, lenInput_global - lenInput - offset1_in);
sizeInput = lenInput * sizeof(Tin);
sizeInput_padded = sizeInput + (halo_in_left + halo_in_right) * sizeof(Tin);

if (oldPtr != NULL) allocator->updateKey(oldPtr, newPtr, context);
}

void relocateInputBuffer(const Tin *inPtr, const bool reuseIn, const Tout *reducePtr) {
cl_int status;

if (reuseIn) {
inputBuffer = allocator->createBufferUnique(inPtr, context,
CL_MEM_READ_WRITE, sizeInput_padded, &status);
checkResult(status, "CreateBuffer(Unique) input");
} else {
if (inputBuffer)  allocator->releaseBuffer(inPtr, context, inputBuffer);
inputBuffer = allocator->createBuffer(inPtr, context,
CL_MEM_READ_WRITE, sizeInput_padded, &status);
checkResult(status, "CreateBuffer input");
}

if(kernel_map) {

if (lenInput < lenOutput) {
wgsize_map = std::min<size_t>(lenOutput, wgsize_map_static);
nthreads_map = wgsize_map * ((lenOutput + wgsize_map - 1) / wgsize_map); 
}
else {
wgsize_map = std::min<size_t>(lenInput, wgsize_map_static);
nthreads_map = wgsize_map * ((lenInput + wgsize_map - 1) / wgsize_map); 
}
#ifdef FF_OPENCL_LOG
std::cerr <<  "[virtual accelerator @"<<this<<"]\n";
std::cerr << "+ computed MAP kernel sizing parameters:\n";
std::cerr << "- MAP workgroup-size = " << wgsize_map << "\n";
std::cerr << "- MAP n. threads     = " << nthreads_map << " \n";
#endif
}
if (kernel_reduce && reduce_mode == REDUCE_INPUT) {
resetReduce(lenInput, sizeof(Tin), (void *)reducePtr);
}
}

void adjustOutputBufferOffset(const Tout *newPtr, const Tout *oldPtr, std::pair<size_t, size_t> &P, size_t len_global) {
offset1_out = P.first;
lenOutput   = P.second;
lenOutput_global = len_global;
halo_out_left = (std::min)(halo_half, offset1_out);
halo_out_right = (std::min)(halo_half, lenOutput_global - lenOutput - offset1_out);
sizeOutput = lenOutput * sizeof(Tout);
sizeOutput_padded = sizeOutput + (halo_out_left + halo_out_right) * sizeof(Tout);

if (oldPtr != NULL) allocator->updateKey(oldPtr, newPtr, context);
}

void relocateOutputBuffer(const Tout  *outPtr, const Tout *reducePtr) {
cl_int status;
if (outputBuffer) allocator->releaseBuffer(outPtr, context, outputBuffer);
outputBuffer = allocator->createBuffer(outPtr, context,
CL_MEM_READ_WRITE, sizeOutput_padded,&status);
checkResult(status, "CreateBuffer output");

if (kernel_reduce && reduce_mode == REDUCE_OUTPUT) {
resetReduce(lenOutput, sizeof(Tout), (void *)reducePtr);
}
}

void relocateEnvBuffer(const void *envptr, const bool reuseEnv, const size_t idx, const size_t envbytesize) {
cl_int status = CL_SUCCESS;

if (idx >= envBuffer.size()) {
cl_mem envb;
if (reuseEnv) 
envb = allocator->createBufferUnique(envptr, context,
CL_MEM_READ_WRITE, envbytesize, &status);
else 
envb = allocator->createBuffer(envptr, context,
CL_MEM_READ_WRITE, envbytesize, &status);
if (checkResult(status, "CreateBuffer envBuffer"))
envBuffer.push_back(std::make_pair(envb,envbytesize));
} else { 

if (reuseEnv) {
envBuffer[idx].first = allocator->createBufferUnique(envptr, context,
CL_MEM_READ_WRITE, envbytesize, &status);
if (checkResult(status, "CreateBuffer envBuffer"))
envBuffer[idx].second = envbytesize;
} else {
if (envBuffer[idx].second < envbytesize) {
if (envBuffer[idx].first) allocator->releaseBuffer(envptr, context, envBuffer[idx].first);
envBuffer[idx].first = allocator->createBuffer(envptr, context,
CL_MEM_READ_WRITE, envbytesize, &status);
if (checkResult(status, "CreateBuffer envBuffer"))
envBuffer[idx].second = envbytesize;
}
}
}
}

void setInPlace(Tout *reducePtr) {
outputBuffer      = inputBuffer;          
lenOutput         = lenInput;
lenOutput_global  = lenInput_global;
halo_out_left          = halo_in_left;
halo_out_right          = halo_in_right;
offset1_out       = offset1_in;
sizeOutput        = sizeInput;
sizeOutput_padded = sizeInput_padded;

if (kernel_reduce && reduce_mode == REDUCE_OUTPUT) {
resetReduce(lenOutput, sizeof(Tout), reducePtr);
}
}

void swap() {
cl_mem tmp = inputBuffer;
inputBuffer = outputBuffer;
outputBuffer = tmp;
cl_int status = clSetKernelArg(kernel_map, 0, sizeof(cl_mem), &inputBuffer);
checkResult(status, "setKernelArg input");
status = clSetKernelArg(kernel_map, 1, sizeof(cl_mem), &outputBuffer);
checkResult(status, "setKernelArg output");
}

virtual void setMapKernelArgs(const size_t envSize) {
cl_uint idx = 0;
cl_int status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &inputBuffer);
checkResult(status, "setKernelArg input");
status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &outputBuffer);
checkResult(status, "setKernelArg output");

status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &lenInput_global);
checkResult(status, "setKernelArg global input length");

status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &lenOutput);
checkResult(status, "setKernelArg local input length");
status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &offset1_in);
checkResult(status, "setKernelArg offset");
status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &halo_out_left); 
checkResult(status, "setKernelArg pad");

for(size_t k=0; k < envSize; ++k) {
status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &envBuffer[k].first);
checkResult(status, "setKernelArg env");
}
}

size_t asyncH2Dinput(Tin *p) {
if (nevents_h2d >= events_h2d.size()) events_h2d.reserve(nevents_h2d);
p += offset1_in - halo_in_left;
cl_int status = clEnqueueWriteBuffer(cmd_queue, inputBuffer, CL_FALSE, 0,
sizeInput_padded, p, 0, NULL, &events_h2d[nevents_h2d++]);
checkResult(status, "copying Task to device input-buffer");
return sizeInput_padded;
}

size_t asyncH2Denv(const size_t idx, char *p) {
if (nevents_h2d >= events_h2d.size()) events_h2d.reserve(nevents_h2d);
cl_int status = clEnqueueWriteBuffer(cmd_queue, envBuffer[idx].first, CL_FALSE, 0,
envBuffer[idx].second, p, 0, NULL, &events_h2d[nevents_h2d++]);    
checkResult(status, "copying Task to device env-buffer");
return envBuffer[idx].second;
}

size_t  asyncH2Dborders(Tout *p) {
if (halo_out_left) {
cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE, 0,
halo_out_left * sizeof(Tout), p + offset1_out - halo_out_left, 0, NULL,
&events_h2d[nevents_h2d++]);
checkResult(status, "copying left border to device");
return halo_out_left * sizeof(Tout);
}
if (halo_out_right) {
cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE,
(halo_out_left + lenOutput) * sizeof(Tout), halo_out_right * sizeof(Tout),  
p + offset1_out + lenOutput, 0, NULL, &events_h2d[nevents_h2d++]);
checkResult(status, "copying right border to device");
}
return halo_out_left * sizeof(Tout);
}

size_t asyncD2Houtput(Tout *p) {
cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
halo_out_left * sizeof(Tout), sizeOutput, p + offset1_out, 0, NULL, &event_d2h);
checkResult(status, "copying output back from device");
return sizeOutput;
}

size_t asyncD2Hborders(Tout *p) {
cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
halo_out_left * sizeof(Tout), halo_half * sizeof(Tout), p + offset1_out, 0, NULL,
&events_h2d[0]);
checkResult(status, "copying border1 back from device");
++nevents_h2d;
status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
(halo_out_left + lenOutput - halo_half) * sizeof(Tout), halo_half * sizeof(Tout),
p + offset1_out + lenOutput - halo_half, 0, NULL, &event_d2h);
checkResult(status, "copying border2 back from device");
return halo_half * sizeof(Tout);
}



void initReduce() {
int idx = 0;
cl_mem tmp = (reduce_mode == REDUCE_OUTPUT) ? outputBuffer : inputBuffer;
cl_uint len = (reduce_mode == REDUCE_OUTPUT) ? lenOutput : lenInput;
cl_int status  = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &tmp);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &halo_in_left);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),	(void *) &len);
status        |= clSetKernelArg(kernel_reduce, idx++, wg_red_mem, NULL);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tout), (void *) &identityVal);
checkResult(status, "setKernelArg reduce-1");
}

void asyncExecMapKernel() {
cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_map, 1, NULL,
&nthreads_map, &wgsize_map, 0, NULL, &event_map);
checkResult(status, "executing map kernel");
++nevents_map;
}

void asyncExecReduceKernel1() {
cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
&nthreads_reduce, &wgsize_reduce, nevents_map,
(nevents_map==0)?NULL:&event_map,
&event_reduce1);
checkResult(status, "exec kernel reduce-1");
nevents_map = 0;
}

void asyncExecReduceKernel2() {
cl_uint zeropad = 0;
int idx = 0;
cl_int status  = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &zeropad);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),(void*) &nwg_reduce);
status        |= clSetKernelArg(kernel_reduce, idx++, wg_red_mem, NULL);
status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tout), (void *) &identityVal);
checkResult(status, "setKernelArg reduce-2");
status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
&wgsize_reduce, &wgsize_reduce, 1, &event_reduce1,
&event_reduce2);
checkResult(status, "exec kernel reduce-2");
}

Tout getReduceVar() {
cl_int status = clEnqueueReadBuffer(cmd_queue, reduceBuffer, CL_TRUE, 0,
sizeof(Tout), &reduceVar, 0, NULL, NULL);
checkResult(status, "d2h reduceVar");
return reduceVar;
}

void waitforh2d() {
if (nevents_h2d>0) {
cl_int status = clWaitForEvents(nevents_h2d, events_h2d.data());
checkResult(status, "h2d wait for");
nevents_h2d = 0;
}
}

void waitford2h() {
cl_int status = clWaitForEvents(1, &event_d2h);
checkResult(status, "d2h wait for");
}

void waitforreduce() {
cl_int status = clWaitForEvents(1, &event_reduce2);
checkResult(status, "wait for reduce");
}

void waitformap() {
cl_int status = clWaitForEvents(nevents_map, &event_map);
nevents_map = 0;
checkResult(status, "wait for map");
}


void setHaloHalf(const size_t h) {
halo_half = h;
}

private:
int buildProgram(cl_device_id dId) {
cl_int status = clBuildProgram(program, 1, &dId, NULL, NULL,NULL);
checkResult(status, "building program");

if (status != CL_SUCCESS) {
printf("\nFail to build the program\n");
size_t len;
clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
printf("LOG len %ld\n", len);
char *buffer = (char*) calloc(len, sizeof(char));
assert(buffer);
clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, len * sizeof(char),
buffer, NULL);
printf("LOG: %s\n\n", buffer);

return -1;
}
return 0;
}

int buildKernelCode(const std::string &kc, cl_device_id dId) {
cl_int status;

size_t sourceSize = kc.length();
const char* code = kc.c_str();

#ifdef FF_OPENCL_LOG
printf("\n");
printf("buildKernelCode:\n%s\n", code);
printf("\n");
#endif
program = clCreateProgramWithSource(context, 1, &code, &sourceSize, &status);
if (!program) {
checkResult(status, "creating program with source");
return -1;
}
return buildProgram(dId);
}

int createProgram(const std::string &filepath, cl_device_id dId, const bool save_binary, const bool reuse_binary) {
cl_int status, binaryStatus;
bool binary = false;
const std::string binpath = filepath + ".bin";

std::ifstream ifs;
if (reuse_binary) {
ifs.open(binpath, std::ios::binary );
if (!ifs.is_open()) { 
ifs.open(filepath, std::ios::binary);
if (!ifs.is_open()) {
error("createProgram: cannot open %s (nor %s)\n", filepath.c_str(), binpath.c_str());
return -1;
}
} else binary = true;
} else {
ifs.open(filepath, std::ios::binary);
if (!ifs.is_open()) {
error("createProgram: cannot open source file %s\n", filepath.c_str());
return -1;
}
}
std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
(std::istreambuf_iterator<char>()));
ifs.close();
size_t bufsize      = buf.size();
const char *bufptr  = buf.data();

status = CL_INVALID_BINARY;
if (binary) {
program = clCreateProgramWithBinary(context, 1, &dId, &bufsize,
reinterpret_cast<const unsigned char **>(&bufptr),
&binaryStatus, &status);
}        
if (status != CL_SUCCESS) { 
program = clCreateProgramWithSource(context, 1,&bufptr, &bufsize, &status);
if (!program) {
checkResult(status, "creating program with source");
return -1;  
}
if (buildProgram(dId)<0) return -1;
if (save_binary) { 
size_t programBinarySize;
status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
sizeof(size_t) * 1,
&programBinarySize, NULL);
checkResult(status, "createProgram clGetProgramInfo (binary size)");                    

std::vector<char> binbuf(programBinarySize);
const char *binbufptr = binbuf.data();
status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char*) * 1,
&binbufptr, NULL);
checkResult(status, "createProgram clGetProgramInfo (binary data)");                          

std::ofstream ofs(binpath, std::ios::out | std::ios::binary);
ofs.write(binbuf.data(), binbuf.size());
ofs.close();
}
return 0;
}             
return buildProgram(dId);
}


cl_int buildKernels(const std::string &kernel_code,
const std::string &kernel_name1, const std::string &kernel_name2,
const bool from_source, const bool save_binary, const bool reuse_binary) {

cl_int status_ = CL_SUCCESS;

if (!from_source) { 
if (buildKernelCode(kernel_code, deviceId)<0) return -1;
} else {  
if (createProgram(kernel_code, deviceId, save_binary, reuse_binary)<0) return -1;
}

if (kernel_name1 != "") { 
cl_int status;
kernel_map = clCreateKernel(program, kernel_name1.c_str(), &status);
checkResult(status, "CreateKernel (map)");
status_ |= status;
}
if (kernel_name2 != "") { 
cl_int status;
kernel_reduce = clCreateKernel(program, kernel_name2.c_str(), &status);
checkResult(status, "CreateKernel (reduce)");
status_ |= status;
}

return status_;
}

void svc_releaseOclObjects() {
if (kernel_map)     clReleaseKernel(kernel_map);
if (kernel_reduce)	clReleaseKernel(kernel_reduce);
clReleaseProgram(program);

allocator->releaseAllBuffers(context);
}

void resetReduce(size_t lenReduceInput, size_t elem_size, const void *reducePtr) {
nthreads_reduce = lenReduceInput;
if(!isPowerOf2(nthreads_reduce))
nthreads_reduce = nextPowerOf2(nthreads_reduce);
wgsize_reduce = std::min<size_t>(nthreads_reduce, wgsize_reduce_static);
nwg_reduce = nthreads_reduce / wgsize_reduce;

wg_red_mem = (wgsize_reduce * elem_size)
+ (wgsize_reduce <= 32) * (wgsize_reduce * elem_size);

size_t global_red_mem = nwg_reduce * elem_size;

cl_int status;
if (reduceBuffer)
allocator->releaseBuffer(reducePtr, context, reduceBuffer);
reduceBuffer = allocator->createBuffer(reducePtr, context,
CL_MEM_READ_WRITE, global_red_mem, &status);
checkResult(status, "CreateBuffer reduce");

#ifdef FF_OPENCL_LOG
std::cerr << "[virtual accelerator @"<<this<<"]\n";
std::cerr << "+ computed REDUCE kernel sizing parameters:\n";
std::cerr << "- REDUCE workgroup-size = " <<wgsize_reduce<< " \n";
std::cerr << "- REDUCE n. threads     = " <<nthreads_reduce<< " \n";
std::cerr << "- REDUCE n. workgroups  = " <<nwg_reduce<< " \n";
std::cerr << "- REDUCE per-wg memory  = " <<wg_red_mem<< " \n";
#endif
}


inline void getBlocksAndThreads(const size_t size, const size_t maxBlocks,
const size_t maxThreads, size_t & blocks, size_t &threads) {
const size_t half = (size + 1) / 2;
threads =
(size < maxThreads * 2) ?
(isPowerOf2(half) ? nextPowerOf2(half + 1) : nextPowerOf2(half)) :
maxThreads;
blocks = (size + (threads * 2 - 1)) / (threads * 2);
blocks = std::min(maxBlocks, blocks);
}

protected:  
cl_context context;
cl_program program;
cl_command_queue cmd_queue;
cl_mem reduceBuffer;
cl_kernel kernel_map, kernel_reduce, kernel_init;

protected:
const bool from_source;
bool my_own_allocator; 
ff_oclallocator *allocator;
size_t halo_half; 
const Tout identityVal;
Tout reduceVar;

cl_mem inputBuffer, outputBuffer;
std::vector<std::pair<cl_mem, size_t> > envBuffer;


size_t sizeInput; 
size_t sizeInput_padded; 
size_t lenInput; 
size_t offset1_in; 
size_t halo_in_left; 
size_t halo_in_right; 
size_t lenInput_global; 

size_t sizeOutput, sizeOutput_padded;
size_t lenOutput, offset1_out, halo_out_left, halo_out_right, lenOutput_global;

size_t wgsize_map_static, wgsize_reduce_static;
size_t wgsize_map_max, wgsize_reduce_max;
size_t wgsize_map, wgsize_reduce;
size_t nthreads_map, nthreads_reduce;
size_t nwg_reduce;
size_t wg_red_mem;

std::vector<cl_event> events_h2d;
size_t nevents_h2d, nevents_map;
cl_event event_d2h, event_map, event_reduce1, event_reduce2;

reduceMode reduce_mode;

cl_device_id deviceId;
};





template<typename T, typename TOCL = T, typename accelerator_t = ff_oclAccelerator<T, TOCL> >
class ff_stencilReduceLoopOCL_1D: public ff_oclNode_t<T> {
public:
typedef typename TOCL::Tin         Tin;
typedef typename TOCL::Tout        Tout;

ff_stencilReduceLoopOCL_1D(const std::string &mapf,			              
const std::string &reducef = std::string(""),  
const Tout &identityVal = Tout(),               
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1, const int width = 1) :
oneshot(false), saveBinary(false), reuseBinary(false), 
accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
stencil_width_half(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL),
oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
setcode(mapf, reducef);
for(size_t i = 0; i< NACCELERATORS; ++i)
accelerators[i]= new accelerator_t(allocator, width,identityVal);
#ifdef FF_OPENCL_LOG
fprintf(stderr,"[ff_stencilReduceLoopOCL_1D node @%p]\n",this);
fprintf(stderr,"map-kernel code:\n%s\n", mapf.c_str());
fprintf(stderr,"reduce-kernel code:\n%s\n", reducef.c_str());
#endif
}

ff_stencilReduceLoopOCL_1D(const std::string &kernels_source,            
const std::string &mapf_name,                 
const std::string &reducef_name,              
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1, const int width = 1) :
oneshot(false), saveBinary(false), reuseBinary(false), 
accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
stencil_width_half(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL),
oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
setsourcecode(kernels_source, mapf_name, reducef_name);
for(size_t i = 0; i< NACCELERATORS; ++i)
accelerators[i]= new accelerator_t(allocator, width, identityVal, true);
}

ff_stencilReduceLoopOCL_1D(const T &task, 
const std::string &mapf,	               
const std::string &reducef = std::string(""),	
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1, const int width = 1) :
oneshot(true), saveBinary(false), reuseBinary(false), 
accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
stencil_width_half(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL),
oldBytesizeIn(0), oldSizeOut(0), oldSizeReduce(0) {
ff_node::skipfirstpop(true);
setcode(mapf, reducef);
setTask(const_cast<T&>(task));
for(size_t i = 0; i< NACCELERATORS; ++i)
accelerators[i]= new accelerator_t(allocator, width,identityVal);
#ifdef FF_OPENCL_LOG
fprintf(stderr,"[ff_stencilReduceLoopOCL_1D node @%p]\n",this);
fprintf(stderr,"map-kernel code:\n%s\n", mapf.c_str());
fprintf(stderr,"reduce-kernel code:\n%s\n", reducef.c_str());
#endif
}

ff_stencilReduceLoopOCL_1D(const T &task, 
const std::string &kernels_source,            
const std::string &mapf_name,           
const std::string &reducef_name,        
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1, const int width = 1) :
oneshot(true), saveBinary(false), reuseBinary(false), 
accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
stencil_width_half(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL),
oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
setsourcecode(kernels_source, mapf_name, reducef_name);
setTask(const_cast<T&>(task));
for(size_t i = 0; i< NACCELERATORS; ++i)
accelerators[i]= new accelerator_t(allocator, width, identityVal, true);
}


virtual ~ff_stencilReduceLoopOCL_1D() {
for(size_t i = 0; i< accelerators.size(); ++i)
if (accelerators[i]) delete accelerators[i];
}

void setTask(T &task) {
Task.resetTask();
Task.setTask(&task);
}


void setDevices(std::vector<cl_device_id> &dev) {
devices = dev;
}

void pickCPU () {
ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_CPU);
}

void pickGPU (size_t offset=0 ) {
offset_dev=offset; 
ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_GPU);
}

void saveBinaryFile() { saveBinary = true; }

void reuseBinaryFile() { reuseBinary = true; }

virtual int run(bool = false) {
return ff_node::run();
}

virtual int wait() {
return ff_node::wait();
}

virtual int run_and_wait_end() {
if (run() < 0)
return -1;
if (wait() < 0)
return -1;
return 0;
}

virtual int run_then_freeze() {
if (ff_node::isfrozen()) {
ff_node::thaw(true);
return 0;
}
return ff_node::freeze_and_run();
}
virtual int wait_freezing() {
return ff_node::wait_freezing();
}

const T* getTask() const {
return &Task;
}

unsigned int getIter() {
return Task.getIter();
}

Tout *getReduceVar() {
assert(oneshot);
return Task.getReduceVar();
}


int nodeInit() {
if (ff_oclNode_t<T>::oclId < 0) { 
ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();
if (devices.size() == 0) { 
switch (ff_oclNode_t<T>::getDeviceType()) {
case CL_DEVICE_TYPE_ALL:
case CL_DEVICE_TYPE_GPU: {
std::vector<ssize_t> logdev =
clEnvironment::instance()->coAllocateGPUDeviceRR(accelerators.size(),
(offset_dev == 0) ? -1 : offset_dev);
if (logdev.size() == 0) {
if(ff_oclNode_t<T>::getDeviceType() == CL_DEVICE_TYPE_GPU) {
error("not enough GPUs found !\n");
return -1;
}
} else {
devices.clear();
for (size_t i = 0; i < logdev.size(); ++i)
devices.push_back(clEnvironment::instance()->getDevice(logdev[i]));
for (size_t i = 0; i < devices.size(); ++i)
if (accelerators[i]->init(devices[i], getReduceMode(), kernel_code,
kernel_name1, kernel_name2, saveBinary, reuseBinary) < 0)
return -1;
break;
}
break;
}
case CL_DEVICE_TYPE_CPU: {
if (accelerators.size() > 1) {
error(
"Multiple (>1) virtual accelerators on CPU are requested. Not yet implemented.\n");
return -1;
} else {
devices.clear();
devices.push_back(clEnvironment::instance()->getDevice( 
clEnvironment::instance()->getCPUDevice())); 
if (accelerators[0]->init(devices[0], getReduceMode(), kernel_code,
kernel_name1, kernel_name2, saveBinary, reuseBinary) < 0)
return -1;
}
}
break;
default: {
error(
"stencilReduceOCL::Other device. Not yet implemented.\n");
return -1;
}
} 
} else {
if (devices.size() > accelerators.size()) {
error(
"stencilReduceOCL::nodeInit: Too many devices requested, increase the number of accelerators!\n");
return -1;
}
for (size_t i = 0; i < devices.size(); ++i)
accelerators[i]->init(devices[i], getReduceMode(), kernel_code, kernel_name1,
kernel_name2, saveBinary, reuseBinary);
}
}


return 0;
}

void nodeEnd() {}

#if defined(FF_REPARA)

size_t rpr_get_sizeIn()  const { return ff_node::rpr_sizeIn; }


size_t rpr_get_sizeOut() const { return ff_node::rpr_sizeOut; }
#endif

protected:

virtual bool isPureMap() const    { return false; }
virtual bool isPureReduce() const { return false; }
reduceMode getReduceMode() {
return isPureReduce() ? REDUCE_INPUT : REDUCE_OUTPUT;
}

virtual int svc_init() { return nodeInit(); }

#if 0    
virtual int svc_init() {
if (ff_oclNode_t<T>::oclId < 0) {
ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();

switch(ff_oclNode_t<T>::getDeviceType()) {
case CL_DEVICE_TYPE_ALL:
fprintf(stderr,"STATUS: requested ALL\n");
case CL_DEVICE_TYPE_GPU: {
std::vector<ssize_t> logdev = clEnvironment::instance()->coAllocateGPUDeviceRR(accelerators.size());
devices.clear();
for (size_t i = 0; i < logdev.size(); ++i)
devices.push_back(clEnvironment::instance()->getDevice(logdev[i]));
if (devices.size() == 0) {
error("stencilReduceOCL::svc_init:not enough GPUs found !\n");
return -1;
} else {
for (size_t i = 0; i < devices.size(); ++i)
accelerators[i]->init(devices[i], kernel_code, kernel_name1,kernel_name2);
break;
}
}
case CL_DEVICE_TYPE_CPU: {
if (accelerators.size()>1) {
error ("Multiple (>1) virtual accelerators on CPU are requested. Not yet implemented.\n");
return -1;
} else {
devices.clear();
devices.push_back(clEnvironment::instance()->getDevice(clEnvironment::instance()->getCPUDevice()));
accelerators[0]->init(devices[0], kernel_code, kernel_name1,kernel_name2);
}
} break;
default: {
error("stencilReduceOCL::Other device. Not yet implemented.\n");
} break;
}
}
return 0;
}
#endif 

virtual void svc_end() {
if (!ff::ff_node::isfrozen()) nodeEnd();
}

T *svc(T *task) {
if (task) setTask(*task);
Tin   *inPtr     = Task.getInPtr();
Tout  *outPtr    = Task.getOutPtr();
Tout  *reducePtr = Task.getReduceVar();
const size_t envSize = Task.getEnvNum(); 

#if defined(FF_REPARA)
ff_node::rpr_sizeIn = ff_node::rpr_sizeOut = 0;
#endif

if ((void*)inPtr != (void*)outPtr) {

if (oldSizeOut != Task.getBytesizeOut()) {
compute_accmem(Task.getSizeOut(), acc_out);

const bool memorychange = (oldSizeOut < Task.getBytesizeOut());

for (size_t i = 0; i < accelerators.size(); ++i) {
accelerators[i]->adjustOutputBufferOffset(outPtr, (memorychange?old_outPtr:NULL), acc_out[i], Task.getSizeOut());
}

if (memorychange) {
for (size_t i = 0; i < accelerators.size(); ++i) {
accelerators[i]->relocateOutputBuffer(outPtr, reducePtr);
}
oldSizeOut = Task.getBytesizeOut();
old_outPtr = outPtr;
}
}                                    
} 
if (oldBytesizeIn != Task.getBytesizeIn()) {
compute_accmem(Task.getSizeIn(),  acc_in);
const bool memorychange = (oldBytesizeIn < Task.getBytesizeIn());
adjustInputBufferOffset(memorychange);
if (memorychange) {
for (size_t i = 0; i < accelerators.size(); ++i) {
accelerators[i]->relocateInputBuffer(inPtr, Task.getReuseIn(), reducePtr);
}
oldBytesizeIn = Task.getBytesizeIn();
old_inPtr = inPtr;
}            
}

if (((void*)inPtr == (void*)outPtr) && ( oldSizeOut != Task.getBytesizeOut())) {
for (size_t i = 0; i < accelerators.size(); ++i) {
accelerators[i]->setInPlace(reducePtr);
}
}                        


for (size_t i = 0; i < accelerators.size(); ++i)
for(size_t k=0; k < envSize; ++k) {
char *envptr;
Task.getEnvPtr(k, envptr);
accelerators[i]->relocateEnvBuffer(envptr, Task.getReuseEnv(k), k, Task.getBytesizeEnv(k));
}        

if (!isPureReduce())  
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->setMapKernelArgs(envSize);

for (size_t i = 0; i < accelerators.size(); ++i) {

if (Task.getCopyIn()) {
#if defined(FF_REPARA)
ff_node::rpr_sizeIn += accelerators[i]->asyncH2Dinput(Task.getInPtr());  
#else
accelerators[i]->asyncH2Dinput(Task.getInPtr());  
#endif
}

for(size_t k=0; k < envSize; ++k) {
if (Task.getCopyEnv(k)) {
char *envptr;
Task.getEnvPtr(k, envptr);    
#if defined(FF_REPARA)
ff_node::rpr_sizeIn += accelerators[i]->asyncH2Denv(k, envptr);
#else
accelerators[i]->asyncH2Denv(k, envptr);
#endif
}
}
} 

if (isPureReduce()) {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->initReduce();

waitforh2d();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecReduceKernel1();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecReduceKernel2();

waitforreduce(); 

Tout redVar = accelerators[0]->getReduceVar();
for (size_t i = 1; i < accelerators.size(); ++i)
redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());
Task.writeReduceVar(redVar);
} else {
Task.resetIter();

if (isPureMap()) {
waitforh2d();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecMapKernel();
Task.incIter();

waitformap(); 

} else { 

for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i]->swap();

bool go = true;
do {

for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i]->swap();

waitforh2d();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecMapKernel();
Task.incIter();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->initReduce();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecReduceKernel1();

for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncExecReduceKernel2();

waitforreduce();

Tout redVar = accelerators[0]->getReduceVar();
for (size_t i = 1; i < accelerators.size(); ++i) 
redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());
Task.writeReduceVar(redVar);

go = Task.iterCondition_aux();
if (go) {
assert(outPtr);
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncD2Hborders(outPtr);
waitford2h(); 
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->asyncH2Dborders(outPtr);
}


} while (go);
}

if (outPtr && Task.getCopyOut()) { 
for (size_t i = 0; i < accelerators.size(); ++i) {
#if defined(FF_REPARA)
ff_node::rpr_sizeOut += accelerators[i]->asyncD2Houtput(outPtr);
#else
accelerators[i]->asyncD2Houtput(outPtr);
#endif
}
waitford2h(); 
}
}


if (Task.getReleaseIn() && (void *)outPtr != (void *)inPtr) {
for (size_t i = 0; i < accelerators.size(); ++i) 
accelerators[i]->releaseInput(inPtr);
oldBytesizeIn = 0;
old_inPtr = NULL;
}
if ( Task.getReleaseOut() ) {
for (size_t i = 0; i < accelerators.size(); ++i) 
accelerators[i]->releaseOutput(outPtr);
oldSizeOut = 0;
old_outPtr = NULL;
}

for(size_t k=0; k < envSize; ++k) {
if (Task.getReleaseEnv(k)) {
char *envptr;
Task.getEnvPtr(k, envptr);
if ((void*)envptr != (void*)outPtr) { 
for (size_t i = 0; i < accelerators.size(); ++i) {
accelerators[i]->releaseEnv(k,envptr);
}
}
}

}

Task.releaseTask(task);

return (oneshot ? NULL : task);
}

protected:
virtual void adjustInputBufferOffset(const bool memorychange) {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->adjustInputBufferOffset(Task.getInPtr(),
(memorychange ? old_inPtr : NULL), acc_in[i],
Task.getSizeIn());
}

void setcode(const std::string &codestr1, const std::string &codestr2) {
int n = 0;
if (codestr1 != "") {
n = codestr1.find_first_of("|");
assert(n > 0);
kernel_name1 = codestr1.substr(0, n);
const std::string &tmpstr = codestr1.substr(n + 1);
n = tmpstr.find_first_of("|");
assert(n > 0);

if (tmpstr.substr(0, n) == "double") {
kernel_code = "\n#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n"
+ tmpstr.substr(n + 1);
} else
kernel_code = "\n" + tmpstr.substr(n + 1);
}

std::ifstream ifs(FF_OPENCL_DATATYPES_FILE);
if (ifs.is_open())
kernel_code.insert(kernel_code.begin(), std::istreambuf_iterator<char>(ifs),
std::istreambuf_iterator<char>());

if (codestr2 != "") {
n = codestr2.find("|");
assert(n > 0);
kernel_name2 += codestr2.substr(0, n);
const std::string &tmpstr = codestr2.substr(n + 1);
n = tmpstr.find("|");
assert(n > 0);

if (tmpstr.substr(0, n) == "double") {
kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
+ tmpstr.substr(n + 1);
} else
kernel_code += tmpstr.substr(n + 1);
}
}

void setsourcecode(const std::string &source, const std::string &kernel1, const std::string &kernel2) {		
if (kernel1 != "") kernel_name1 = "kern_"+kernel1;
if (kernel2 != "") kernel_name2 = "kern_"+kernel2;
kernel_code  = source;
}

void compute_accmem(const size_t len, std::vector<std::pair<size_t,size_t> > &acc) {
size_t start = 0, step = (len + accelerators.size() - 1) / accelerators.size();
size_t i = 0;
for (; i < accelerators.size() - 1; ++i) {
acc[i]=std::make_pair(start, step);
start += step;
}
acc[i]=std::make_pair(start, len-start);
}

void waitforh2d() {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->waitforh2d();
}

void waitford2h() {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->waitford2h();
}

void waitforreduce() {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->waitforreduce();
}

void waitformap() {
for (size_t i = 0; i < accelerators.size(); ++i)
accelerators[i]->waitformap();
}

TOCL Task;
const bool oneshot;
bool saveBinary, reuseBinary;    
std::vector<accelerator_t*> accelerators;
std::vector<std::pair<size_t, size_t> > acc_in;
std::vector<std::pair<size_t, size_t> > acc_out;
std::vector<cl_device_id> devices;
int stencil_width_half;
size_t offset_dev;

std::string kernel_code;
std::string kernel_name1;
std::string kernel_name2;

size_t forced_cpu;
size_t forced_gpu;
size_t forced_other;

Tin   *old_inPtr;
Tout  *old_outPtr;
size_t oldBytesizeIn, oldSizeOut, oldSizeReduce;
};




template<typename T, typename TOCL = T>
class ff_mapOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
ff_mapOCL_1D(std::string mapf, ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, "", 0, alloc, NACCELERATORS, 0) {
}

ff_mapOCL_1D(const std::string &kernels_source, const std::string &mapf_name, 
ff_oclallocator *alloc=nullptr, const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_source, mapf_name, "", 0, alloc, NACCELERATORS, 0) {
}


ff_mapOCL_1D(const T &task, std::string mapf, 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, "", 0, alloc, NACCELERATORS, 0) {
}
ff_mapOCL_1D(const T &task, const std::string &kernels_source, const std::string &mapf_name, 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_source, mapf_name, "", 0, alloc, NACCELERATORS, 0) {
}

bool isPureMap() const { return true; }
};


template<typename T, typename TOCL = T>
class ff_reduceOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
typedef typename TOCL::Tin  Tin;
typedef typename TOCL::Tout Tout;

ff_reduceOCL_1D(std::string reducef, const Tout &identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>("", reducef, identityVal, alloc, NACCELERATORS, 0) {
}
ff_reduceOCL_1D(const std::string &kernels_source, const std::string &reducef_name, const Tout &identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_source, "", reducef_name, identityVal, alloc, NACCELERATORS, 0) {
}

ff_reduceOCL_1D(const T &task, std::string reducef, const Tout identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, "", reducef, identityVal, alloc, NACCELERATORS, 0) {
}
ff_reduceOCL_1D(const T &task, const std::string &kernels_source,const std::string &reducef_name, 
const Tout identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_source, "", reducef_name, identityVal, alloc, NACCELERATORS, 0) {
}
bool isPureReduce() const { return true; }
};


template<typename T, typename TOCL = T>
class ff_mapReduceOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
typedef typename TOCL::Tin  Tin;
typedef typename TOCL::Tout Tout;

ff_mapReduceOCL_1D(std::string mapf, std::string reducef, const Tout &identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
}

ff_mapReduceOCL_1D(const std::string &kernels_code, const std::string &mapf_name, 
const std::string &reducef_name, const Tout &identityVal = Tout(),
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_code, mapf_name, reducef_name, identityVal, alloc, NACCELERATORS, 0) {
}


ff_mapReduceOCL_1D(const T &task, std::string mapf, std::string reducef,
const Tout &identityVal = Tout(), 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
}
ff_mapReduceOCL_1D(const T &task, const std::string &kernels_code, const std::string &mapf_name, 
const std::string &reducef_name, const Tout &identityVal = Tout(), 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_code, mapf_name, reducef_name, identityVal, alloc, NACCELERATORS, 0) {
}

};






template<typename TaskT_, typename Tin_, typename Tout_ = Tin_>
class baseOCLTask_2D: public baseOCLTask<TaskT_, Tin_, Tout_> {
public:

void setHeight(size_t h) {
height = h;
}


void setWidth(size_t w) {
width = w;
}

size_t getHeight() const { return height;}
size_t getWidth() const {return width;}

protected:
size_t height, width;
};




template<typename T, typename TOCL = T>
class ff_oclAccelerator_2D : public ff_oclAccelerator<T, TOCL> {
public:
typedef typename TOCL::Tin  Tin;
typedef typename TOCL::Tout Tout;

ff_oclAccelerator_2D(ff_oclallocator *alloc, const size_t halo_width_, const Tout &identityVal, const bool from_source=false) :
ff_oclAccelerator<T,TOCL>(alloc, halo_width_, identityVal, from_source) {
heightInput_global = 0;
widthInput_global = 0;
}

void setMapKernelArgs(const size_t envSize) {
cl_uint idx = 0;
cl_int status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_mem), &this->inputBuffer);
checkResult(status, "setKernelArg input");
status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_mem), &this->outputBuffer);
checkResult(status, "setKernelArg output");

status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_uint), (void *) &heightInput_global);
checkResult(status, "setKernelArg global input height");
status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_uint), (void *) &widthInput_global);
checkResult(status, "setKernelArg global input width");

status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_uint), (void *) &this->lenOutput);
checkResult(status, "setKernelArg local input length");
status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_uint), (void *) &this->offset1_in);
checkResult(status, "setKernelArg offset");
status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_uint), (void *) &this->halo_out_left);
checkResult(status, "setKernelArg halo");

for(size_t k=0; k < envSize; ++k) {
status = clSetKernelArg(this->kernel_map, idx++, sizeof(cl_mem), &(this->envBuffer[k].first));
checkResult(status, "setKernelArg env");
}
}


void setHeight(size_t h) {
heightInput_global = h;
}

void setWidth(size_t w) {
widthInput_global = w;
}

private:
size_t heightInput_global, widthInput_global;
};




template<typename T, typename TOCL = T>
class ff_stencilReduceLoopOCL_2D: public ff_stencilReduceLoopOCL_1D<T,TOCL,ff_oclAccelerator_2D<T,TOCL> > {
private:
typedef typename TOCL::Tin Tin;
typedef typename TOCL::Tout Tout;
typedef ff_oclAccelerator_2D<T,TOCL> accelerator_t;
typedef ff::ff_stencilReduceLoopOCL_1D<T,TOCL,accelerator_t> base_srl_t;

public:
ff_stencilReduceLoopOCL_2D(const std::string &mapf,			              
const std::string &reducef = std::string(""),  
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1,
const int stencil_width_half_ = 1,
const int stencil_height_half_ = 1) :
base_srl_t(mapf, reducef, identityVal, allocator, NACCELERATORS, 0),
stencil_width_half(stencil_width_half_), stencil_height_half(stencil_height_half_) {}

ff_stencilReduceLoopOCL_2D(const std::string &kernels_source,            
const std::string &mapf_name,                 
const std::string &reducef_name,              
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1,
const int stencil_width_half_ = 1,
const int stencil_height_half_ = 1) :
base_srl_t(kernels_source, mapf_name, reducef_name, identityVal, allocator, NACCELERATORS, 0),
stencil_width_half(stencil_width_half_), stencil_height_half(stencil_height_half_) {}

ff_stencilReduceLoopOCL_2D(const T &task,
const std::string &mapf,
const std::string &reducef = std::string(""),
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1,
const int stencil_width_half_ = 1,
const int stencil_height_half_ = 1) :
base_srl_t(task, mapf, reducef, identityVal, allocator, NACCELERATORS, 0),
stencil_width_half(stencil_width_half_),stencil_height_half(stencil_height_half_) {}

ff_stencilReduceLoopOCL_2D(const T &task,
const std::string &kernels_source,            
const std::string &mapf_name,           
const std::string &reducef_name,        
const Tout &identityVal = Tout(),
ff_oclallocator *allocator = nullptr,
const size_t NACCELERATORS = 1,
const int stencil_width_half_ = 1,
const int stencil_height_half_ = 1) :
base_srl_t(task, kernels_source, mapf_name, reducef_name, identityVal, allocator, NACCELERATORS, 0),
stencil_width_half(stencil_width_half_), stencil_height_half(stencil_height_half_) {}

protected:
void adjustInputBufferOffset(const bool memorychange) {
for (size_t i = 0; i < this->accelerators.size(); ++i) {
this->accelerators[i]->setWidth(this->Task.getWidth());
this->accelerators[i]->setHeight(this->Task.getHeight());
this->accelerators[i]->setHaloHalf(halo_half(this->Task.getWidth()));
}
base_srl_t::adjustInputBufferOffset(memorychange);
}
private:

const size_t halo_half(const size_t width) {
return stencil_height_half * width + stencil_width_half;
}


const int stencil_width_half;
const int stencil_height_half;
};



template<typename T, typename TOCL = T>
class ff_mapOCL_2D: public ff_stencilReduceLoopOCL_2D<T, TOCL> {
public:
ff_mapOCL_2D(std::string mapf, ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_2D<T, TOCL>(mapf, "", 0, alloc, NACCELERATORS) {
}

ff_mapOCL_2D(const std::string &kernels_source, const std::string &mapf_name, 
ff_oclallocator *alloc=nullptr, const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_2D<T, TOCL>(kernels_source, mapf_name, "", 0, alloc, NACCELERATORS) {
}


ff_mapOCL_2D(const T &task, std::string mapf, 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_2D<T, TOCL>(task, mapf, "", 0, alloc, NACCELERATORS) {
}
ff_mapOCL_2D(const T &task, const std::string &kernels_source, const std::string &mapf_name, 
ff_oclallocator *alloc=nullptr,
const size_t NACCELERATORS = 1) :
ff_stencilReduceLoopOCL_2D<T, TOCL>(task, kernels_source, mapf_name, "", 0, alloc, NACCELERATORS) {
}

bool isPureMap() const { return true; }
};    





}

#endif 
#endif 

