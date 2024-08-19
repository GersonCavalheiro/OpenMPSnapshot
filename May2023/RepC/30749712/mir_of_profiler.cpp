#include "pin.H"
#include "portability.H"
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <stack>
#include <string>
#include <sstream>
#include <algorithm>
#include <omp.h>
#include <sys/shm.h>
#include <assert.h>
KNOB<string> KnobOutputFileSuffix(KNOB_MODE_WRITEONCE, "pintool",
"pr", "mir-ofp", "specify output file prefix");
KNOB<BOOL> KnobPid(KNOB_MODE_WRITEONCE, "pintool",
"pi", "0", "append pid to output");
KNOB<string> KnobFunctionNames(KNOB_MODE_WRITEONCE, "pintool",
"of", "", "specify outline functions (csv)");
KNOB<string> KnobCalledFunctionNames(KNOB_MODE_WRITEONCE, "pintool",
"cf", "", "specify functions called (csv) from outline functions");
KNOB<string> KnobDynamicallyCalledFunctionNames(KNOB_MODE_WRITEONCE, "pintool",
"df", "", "specify functions dynamically called (csv) from outline functions");
KNOB<BOOL> KnobCalcMemShare(KNOB_MODE_WRITEONCE, "pintool",
"ds", "0", "calculate data sharing (NOTE: a time consuming process!)");
KNOB<BOOL> KnobDisableIgnoreContextDetection(KNOB_MODE_WRITEONCE, "pintool",
"ni", "0", "disable ignorable context detection");
#define EXCLUDE_STACK_INS_FROM_MEM_FP 1
#define GET_INS_MIX 1
#define MIR_SHM_KEY 31415926
#define MIR_SHM_SIZE 16
#define MIR_SHM_SIGREAD '*'
bool g_shmat_done = false;
int g_shmid;
char* g_shm;
#if 0
int g_id = 0;
#endif
void tokenize(const std::string& str, const std::string& delimiters, std::vector<std::string>& tokens)
{ 
std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
std::string::size_type pos = str.find_first_of(delimiters, lastPos);
while (std::string::npos != pos || std::string::npos != lastPos) {
tokens.push_back(str.substr(lastPos, pos - lastPos));
lastPos = str.find_first_not_of(delimiters, pos);
pos = str.find_first_of(delimiters, lastPos);
}
} 
UINT64 get_cycles()
{
unsigned a, d;
__asm__ volatile("rdtsc"
: "=a"(a), "=d"(d));
return ((uint64_t)a) | (((uint64_t)d) << 32);
}
#define NAME_SIZE 256
typedef struct _MIR_FUNCTION_STAT_ { 
UINT64 id;
UINT64 ins_count;
UINT64 stack_read;
UINT64 stack_write;
UINT64 mem_read;
UINT64 mem_write;
std::set<VOID*> mem_fp;
#ifdef GET_INS_MIX
UINT64 ins_mix[XED_CATEGORY_LAST];
#endif
char name[NAME_SIZE];
UINT64 ccr; 
UINT64 clr; 
UINT64 mem_fp_sz;
INT64 ignore_context_count;
std::vector<UINT64> mem_share;
std::vector<UINT64> task_create_instant;
std::vector<UINT64> task_wait_instant;
struct _MIR_FUNCTION_STAT_* next;
} MIR_FUNCTION_STAT; 
MIR_FUNCTION_STAT* g_stat_list = NULL;
MIR_FUNCTION_STAT* g_current_stat = NULL;
std::stack<MIR_FUNCTION_STAT*> g_stat_stack;
VOID MIROutlineFunctionUpdateMemRefRead(VOID* memp)
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0) {
g_current_stat->mem_read++;
g_current_stat->mem_fp.insert(memp);
}
} 
VOID MIROutlineFunctionUpdateMemRefWrite(VOID* memp)
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0) {
g_current_stat->mem_write++;
g_current_stat->mem_fp.insert(memp);
}
} 
#ifdef GET_INS_MIX
VOID MIROutlineFunctionUpdateInsMix(INT32 index)
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0)
g_current_stat->ins_mix[index]++;
} 
#endif
VOID MIROutlineFunctionIgnoreContextEntry()
{
if(g_current_stat)
g_current_stat->ignore_context_count++;
}
VOID MIROutlineFunctionIgnoreContextExit()
{
if(g_current_stat)
g_current_stat->ignore_context_count--;
}
VOID MIROutlineFunctionUpdateInsCount()
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0)
g_current_stat->ins_count++;
} 
VOID MIROutlineFunctionUpdateStackRead()
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0)
g_current_stat->stack_read++;
} 
VOID MIROutlineFunctionUpdateStackWrite()
{ 
if (g_current_stat && g_current_stat->ignore_context_count == 0)
g_current_stat->stack_write++;
} 
VOID MIRPrintFunctionStat(MIR_FUNCTION_STAT* stat)
{
std::cerr << "ol = " << stat->name << ", id = " << stat->id << ", ignore_context_count = " << stat->ignore_context_count << std::endl;
}
VOID MIROutlineFunctionEntry(VOID* name)
{ 
if (g_shmat_done == false) {
g_shmat_done = true;
if ((g_shmid = shmget(MIR_SHM_KEY, MIR_SHM_SIZE, 0666)) < 0) {
std::cerr << "Call to shmget failed!" << std::endl;
exit(1);
}
g_shm = (char*)shmat(g_shmid, NULL, 0);
if (g_shm == NULL) {
std::cerr << "No shared memory. Call to shmat returned NULL!" << std::endl;
exit(1);
}
}
char buf[MIR_SHM_SIZE];
for (int i = 0; i < MIR_SHM_SIZE; i++)
buf[i] = g_shm[i];
*g_shm = MIR_SHM_SIGREAD;
MIR_FUNCTION_STAT* stat = new MIR_FUNCTION_STAT;
stat->id = atoi(buf);
#if 0
stat->id = g_id++;
#endif
stat->ins_count = 0;
stat->stack_read = 0;
stat->stack_write = 0;
stat->mem_read = 0;
stat->mem_write = 0;
memcpy(stat->name, name, sizeof(char) * NAME_SIZE);
#ifdef GET_INS_MIX
memset(&stat->ins_mix, 0, sizeof(UINT64) * XED_CATEGORY_LAST);
#endif
stat->ignore_context_count = 0;
stat->next = g_stat_list;
g_stat_list = stat;
g_stat_stack.push(g_stat_list);
g_current_stat = g_stat_list;
} 
VOID MIROutlineFunctionExit()
{ 
if (g_current_stat) {
g_current_stat->mem_fp_sz = g_current_stat->mem_fp.size();
g_current_stat->mem_fp.clear();
if(g_current_stat->ignore_context_count != 0) {
std::cerr << "Ignore context count at outline function exit is non-zero!" << std::endl;
MIRPrintFunctionStat(g_current_stat);
exit(1);
}
}
g_stat_stack.pop();
if (!g_stat_stack.empty())
g_current_stat = g_stat_stack.top();
else
g_current_stat = NULL;
} 
VOID MIRTaskCreateBefore()
{ 
if (g_current_stat)
g_current_stat->task_create_instant.push_back(g_current_stat->ins_count);
} 
VOID MIRTaskWaitAfter()
{ 
if (g_current_stat)
g_current_stat->task_wait_instant.push_back(g_current_stat->ins_count);
} 
VOID Image(IMG img, VOID* v)
{ 
std::cout << "Analyzing loaded image: " << IMG_Name(img) << std::endl;
std::string delims = ",";
std::vector<string>::iterator it;
std::string outline_functions_csv = KnobFunctionNames.Value();
std::vector<std::string> outline_functions;
tokenize(outline_functions_csv, delims, outline_functions);
if (outline_functions.size() == 0) {
std::cout << "Error: Outline function list is empty." << std::endl;
exit(1);
}
for (it = outline_functions.begin(); it != outline_functions.end(); it++) {
RTN mirRtn = RTN_FindByName(img, (*it).c_str());
if (RTN_Valid(mirRtn)) { 
std::cout << "Adding profiling hooks to outline function: " << *it << std::endl;
RTN_Open(mirRtn);
RTN_InsertCall(mirRtn, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionEntry, IARG_PTR, RTN_Name(mirRtn).c_str(), IARG_END);
RTN_InsertCall(mirRtn, IPOINT_AFTER, (AFUNPTR)MIROutlineFunctionExit, IARG_END);
for (INS ins = RTN_InsHead(mirRtn); INS_Valid(ins); ins = INS_Next(ins)) {
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsCount, IARG_END);
#ifdef GET_INS_MIX
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsMix, IARG_UINT32, INS_Category(ins), IARG_END);
#endif
if (INS_IsStackRead(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackRead, IARG_END);
if (INS_IsStackWrite(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackWrite, IARG_END);
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
if (!INS_IsStackRead(ins) && !INS_IsStackWrite(ins)) {
#endif
UINT32 memOperands = INS_MemoryOperandCount(ins);
for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
if (INS_MemoryOperandIsRead(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefRead,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
if (INS_MemoryOperandIsWritten(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefWrite,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
}
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
}
#endif
}
RTN_Close(mirRtn);
} 
}
std::string called_functions_csv = KnobCalledFunctionNames.Value();
std::vector<std::string> called_functions;
tokenize(called_functions_csv, delims, called_functions);
if (called_functions.size() == 0)
std::cout << "Note: Called function list is empty." << std::endl;
for (it = called_functions.begin(); it != called_functions.end(); it++) {
RTN mirRtn = RTN_FindByName(img, (*it).c_str());
if (RTN_Valid(mirRtn)) { 
std::cout << "Adding profiling hooks to called function: " << *it << std::endl;
RTN_Open(mirRtn);
for (INS ins = RTN_InsHead(mirRtn); INS_Valid(ins); ins = INS_Next(ins)) {
if(!KnobDisableIgnoreContextDetection) {
if (INS_IsMov(ins) && INS_FullRegWContain(ins, REG_BX) && INS_FullRegRContain(ins, REG_BX)) {
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionIgnoreContextEntry, IARG_END);
continue;
}
if (INS_IsMov(ins) && INS_FullRegWContain(ins, REG_CX) && INS_FullRegRContain(ins, REG_CX)) {
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionIgnoreContextExit, IARG_END);
continue;
}
}
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsCount, IARG_END);
#ifdef GET_INS_MIX
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsMix, IARG_UINT32, INS_Category(ins), IARG_END);
#endif
if (INS_IsStackRead(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackRead, IARG_END);
if (INS_IsStackWrite(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackWrite, IARG_END);
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
if (!INS_IsStackRead(ins) && !INS_IsStackWrite(ins)) {
#endif
UINT32 memOperands = INS_MemoryOperandCount(ins);
for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
if (INS_MemoryOperandIsRead(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefRead,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
if (INS_MemoryOperandIsWritten(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefWrite,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
}
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
}
#endif
}
RTN_Close(mirRtn);
} 
}
std::string dynamically_called_functions_csv = KnobDynamicallyCalledFunctionNames.Value();
std::vector<std::string> dynamically_called_functions;
tokenize(dynamically_called_functions_csv, delims, dynamically_called_functions);
if (dynamically_called_functions.size() == 0)
std::cout << "Note: Dynamically called function list is empty." << std::endl;
for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym)) {
std::string undFuncName = PIN_UndecorateSymbolName(SYM_Name(sym), UNDECORATION_NAME_ONLY);
for (it = dynamically_called_functions.begin(); it != dynamically_called_functions.end(); it++) {
if (undFuncName == (*it).c_str()) {
RTN mirRtn = RTN_FindByAddress(IMG_LowAddress(img) + SYM_Value(sym));
if (RTN_Valid(mirRtn)) { 
std::cout << "Adding profiling hooks to dynamically called function: " << *it << std::endl;
RTN_Open(mirRtn);
for (INS ins = RTN_InsHead(mirRtn); INS_Valid(ins); ins = INS_Next(ins)) {
if (!KnobDisableIgnoreContextDetection) {
if (INS_IsMov(ins) && INS_FullRegWContain(ins, REG_BX) && INS_FullRegRContain(ins, REG_BX)) {
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionIgnoreContextEntry, IARG_END);
continue;
}
if (INS_IsMov(ins) && INS_FullRegWContain(ins, REG_CX) && INS_FullRegRContain(ins, REG_CX)) {
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionIgnoreContextExit, IARG_END);
continue;
}
}
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsCount, IARG_END);
#ifdef GET_INS_MIX
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateInsMix, IARG_UINT32, INS_Category(ins), IARG_END);
#endif
if (INS_IsStackRead(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackRead, IARG_END);
if (INS_IsStackWrite(ins))
INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateStackWrite, IARG_END);
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
if (!INS_IsStackRead(ins) && !INS_IsStackWrite(ins)) {
#endif
UINT32 memOperands = INS_MemoryOperandCount(ins);
for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
if (INS_MemoryOperandIsRead(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefRead,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
if (INS_MemoryOperandIsWritten(ins, memOp)) {
INS_InsertPredicatedCall(
ins, IPOINT_BEFORE, (AFUNPTR)MIROutlineFunctionUpdateMemRefWrite,
IARG_MEMORYOP_EA, memOp,
IARG_END);
}
}
#ifdef EXCLUDE_STACK_INS_FROM_MEM_FP
}
#endif
}
RTN_Close(mirRtn);
} 
break;
}
}
}
const int num_task_create_functions = 2;
const char* task_create_functions[num_task_create_functions] = {"mir_task_create", "GOMP_task"};
for (int i=0; i<num_task_create_functions; i++) {
RTN mirTaskCreateRtn = RTN_FindByName(img, task_create_functions[i]);
if (RTN_Valid(mirTaskCreateRtn)) {
std::cout << "Adding profiling hooks to runtime system function: " << task_create_functions[i] << std::endl;
RTN_Open(mirTaskCreateRtn);
RTN_InsertCall(mirTaskCreateRtn, IPOINT_BEFORE, (AFUNPTR)MIRTaskCreateBefore, IARG_END);
RTN_Close(mirTaskCreateRtn);
}
}
const int num_task_wait_functions = 2;
const char* task_wait_functions[num_task_wait_functions] = {"mir_task_wait", "GOMP_taskwait"};
for (int i=0; i<num_task_wait_functions; i++) {
RTN mirTaskWaitRtn = RTN_FindByName(img, task_wait_functions[i]);
if (RTN_Valid(mirTaskWaitRtn)) {
std::cout << "Adding profiling hooks to runtime system function: " << task_wait_functions[i] << std::endl;
RTN_Open(mirTaskWaitRtn);
RTN_InsertCall(mirTaskWaitRtn, IPOINT_AFTER, (AFUNPTR)MIRTaskWaitAfter, IARG_END);
RTN_Close(mirTaskWaitRtn);
}
}
} 
VOID MIROutlineFunctionUpdateMemFp(MIR_FUNCTION_STAT* stat)
{ 
UINT64 communication = stat->mem_read + stat->mem_write;
if (communication == 0)
stat->ccr = stat->ins_count;
else
stat->ccr = (int)((double)(stat->ins_count) / communication + 0.5);
UINT64 load = stat->mem_read;
if (load == 0)
stat->clr = stat->ins_count;
else
stat->clr = (int)((double)(stat->ins_count) / load + 0.5);
} 
VOID MIROutlineFunctionUpdateMemShare(MIR_FUNCTION_STAT* this_stat, size_t cutoff)
{ 
size_t sz = 0;
for (MIR_FUNCTION_STAT* stat = g_stat_list; sz <= cutoff; stat = stat->next, sz++) {
if (sz != cutoff) {
std::set<VOID*> mem_fp_intersection;
std::insert_iterator<std::set<VOID*> > ins_it(mem_fp_intersection, mem_fp_intersection.begin());
std::set_intersection(stat->mem_fp.begin(), stat->mem_fp.end(), this_stat->mem_fp.begin(), this_stat->mem_fp.end(), ins_it);
this_stat->mem_share.push_back(mem_fp_intersection.size());
}
else {
this_stat->mem_share.push_back(this_stat->mem_fp.size());
}
}
} 
VOID Fini(INT32 code, VOID* v)
{ 
std::cout << "Finalizing ..." << std::endl;
std::string filename;
filename.clear();
filename = KnobOutputFileSuffix.Value();
if (KnobPid)
filename += "." + decstr(getpid_portable());
filename += "-mem-map";
std::cout << "Writing memory map (/proc/<pid>/maps) to file: " << filename << " ..." << std::endl;
char cmd[256];
sprintf(cmd, "cat /proc/%d/maps > %s", getpid_portable(), filename.c_str());
int rv = system(cmd);
if (rv == 127)
std::cerr << "Could not write memory map to " << filename << std::endl;
std::cout << "Updating statistics in parallel ..." << std::endl;
UINT64 num_instances = 0;
#pragma omp parallel
{
#pragma omp single
{
std::cout << "Updating memory footprint ..." << std::endl;
for (MIR_FUNCTION_STAT* stat = g_stat_list; stat; stat = stat->next) {
num_instances++;
{
MIROutlineFunctionUpdateMemFp(stat);
}
}
if (KnobCalcMemShare) {
std::cout << "Updating memory sharing ..." << std::endl;
std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
size_t cutoff = 0;
for (MIR_FUNCTION_STAT* stat = g_stat_list; stat; stat = stat->next, cutoff++) {
#pragma omp task
{
MIROutlineFunctionUpdateMemShare(stat, cutoff);
}
}
#pragma omp taskwait
}
}
}
filename.clear();
filename = KnobOutputFileSuffix.Value(); 
if (KnobPid)
filename += "." + decstr(getpid_portable());
filename += "-instructions";
std::ofstream out;
out.open(filename.c_str());
std::cout << "Writing call graph information to file: " << filename << " ..." << std::endl;
const char* fileheader = "task,ins_count,stack_read,stack_write,mem_fp,ccr,clr,mem_read,mem_write,outl_func";
#ifdef GET_INS_MIX
string ins_catg = ",";
for (unsigned int c = 0; c < XED_CATEGORY_LAST; c++) {
if (c != 0) {
ins_catg += ",";
ins_catg += xed_category_enum_t2str((const xed_category_enum_t)c);
}
else
ins_catg += xed_category_enum_t2str((const xed_category_enum_t)c);
}
out << fileheader << ins_catg << std::endl;
#else
out << fileheader << std::endl;
#endif
for (MIR_FUNCTION_STAT* stat = g_stat_list; stat; stat = stat->next) {
out << stat->id << ","
<< stat->ins_count << ","
<< stat->stack_read << ","
<< stat->stack_write << ","
<< stat->mem_fp_sz << ","
<< stat->ccr << ","
<< stat->clr << ","
<< stat->mem_read << ","
<< stat->mem_write << ","
<< stat->name;
#ifdef GET_INS_MIX
out << ",";
for (int c = 0; c < XED_CATEGORY_LAST; c++) {
if (c != 0)
out << "," << stat->ins_mix[c];
else
out << stat->ins_mix[c];
}
#endif
out << std::endl;
}
out.close();
if (KnobCalcMemShare) {
filename.clear();
filename = KnobOutputFileSuffix.Value(); 
if (KnobPid)
filename += "." + decstr(getpid_portable());
filename += "-mem-share";
out.open(filename.c_str());
std::cout << "Writing memory sharing information to file: " << filename << " ..." << std::endl;
for (MIR_FUNCTION_STAT* stat = g_stat_list; stat; stat = stat->next) {
UINT64 count = num_instances;
std::vector<UINT64>::iterator it;
const char* padding = "";
for (it = stat->mem_share.begin(); it != stat->mem_share.end(); it++) {
out << padding << *it;
padding = ",";
count--;
}
while (count != 0) {
out << ",";
count--;
}
out << std::endl;
}
out.close();
}
filename.clear();
filename = KnobOutputFileSuffix.Value(); 
if (KnobPid)
filename += "." + decstr(getpid_portable());
filename += "-events";
std::cout << "Writing task create and wait invocation information to file: " << filename << " ..." << std::endl;
out.open(filename.c_str());
out << "task,ins_count,[create],[wait]" << std::endl;
for (MIR_FUNCTION_STAT* stat = g_stat_list; stat; stat = stat->next) {
out << stat->id << "," << stat->ins_count << ",[";
std::vector<UINT64>::iterator it;
for (it = stat->task_create_instant.begin(); it != stat->task_create_instant.end(); it++) {
out << *it << ",";
}
out << "],[";
for (it = stat->task_wait_instant.begin(); it != stat->task_wait_instant.end(); it++) {
out << *it << ",";
}
out << "]";
out << std::endl;
}
out.close();
} 
INT32 Usage()
{ 
cerr << "This PIN tool profiles calls to task outlined functions. Usage:" << endl;
cerr << endl
<< KNOB_BASE::StringKnobSummary() << endl;
return -1;
} 
int main(int argc, char* argv[])
{ 
PIN_InitSymbols();
if (PIN_Init(argc, argv)) {
return Usage();
}
IMG_AddInstrumentFunction(Image, 0);
PIN_AddFiniFunction(Fini, 0);
PIN_StartProgram();
return 0;
} 
