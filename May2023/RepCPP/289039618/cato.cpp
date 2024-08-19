#include <llvm/Pass.h>

#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <memory>
#include <set>

#include "UserTree.h"
#include "cato.hpp"
#include "debug.h"
#include "helper.h"

using namespace llvm;

static cl::opt<bool> cato_logging("cato-logging", cl::init(0), cl::Hidden,
cl::desc("Enable CATO logging"));

PreservedAnalyses CatoPass::run(Module &M, ModuleAnalysisManager &)
{
bool Changed = runOnModule(M);

return (Changed ? PreservedAnalyses::none() : PreservedAnalyses::all());
}


std::vector<std::unique_ptr<MemoryAllocation>> CatoPass::find_memory_allocations(Module &M)
{
std::vector<std::unique_ptr<MemoryAllocation>> memory_allocations;

std::vector<StringRef> allocation_functions = {"malloc", "calloc", "_Znam", "_Znwm"};

for (auto alloc_func : allocation_functions)
{
auto alloc_users = get_function_users(M, alloc_func);

for (auto &user : alloc_users)
{
if (auto *call = dyn_cast<CallInst>(user))
{
memory_allocations.push_back(std::make_unique<MemoryAllocation>(call));
}
}
}

return memory_allocations;
}


std::vector<CallInst *> CatoPass::find_memory_deallocations(Module &M)
{
std::vector<CallInst *> memory_deallocations;

std::vector<StringRef> deallocation_functions = {"free"};

for (auto dealloc_func : deallocation_functions)
{
auto dealloc_users = get_function_users(M, dealloc_func);

for (auto &user : dealloc_users)
{
if (auto *call = dyn_cast<CallInst>(user))
{
memory_deallocations.push_back(call);
}
}
}

return memory_deallocations;
}


void CatoPass::categorize_memory_access_paths(
std::vector<std::vector<Value *>> &paths,
std::vector<std::pair<int, std::vector<Value *>>> *store_paths,
std::vector<std::pair<int, std::vector<Value *>>> *load_paths,
std::vector<std::pair<int, std::vector<Value *>>> *ptr_store_paths,
std::vector<std::vector<Value *>> *free_paths)
{
std::set<Value *> categorized_instructions;

for (auto &path : paths)
{
for (unsigned int i = 0; i < path.size(); i++)
{
Value *u = path[i];

if (auto *store = dyn_cast<StoreInst>(u))
{
if (!store->getValueOperand()->getType()->isPointerTy())
{
auto *store_value = store->getValueOperand();
if (std::find(path.begin(), path.begin() + i, store_value) !=
path.begin() + i)
{
Debug(errs() << "THIS PATH STORES THE VALUE OF THE MEMORY "
"ABSTRACTION AND IS NOT A STORE TO IT!\n";);
Debug(errs() << "    ";);
Debug(store->dump(););
}
else if (categorized_instructions.insert(u).second)
{
store_paths->push_back({i, path});
}
}
else
{
Value *store_destination = store->getPointerOperand();
if (std::find(path.begin(), path.begin() + i, store_destination) !=
path.begin() + i)
{
Debug(errs() << "Pointer value store to shared memory:\n";);
Debug(u->dump(););
if (categorized_instructions.insert(u).second)
{
ptr_store_paths->push_back({i, path});
}
}
else
{
Debug(errs() << "Store dest: ";);
Debug(store_destination->dump(););
Debug(errs() << "Not a store to memeory abstraction.\n";);
}
}
}
else if (auto *load = dyn_cast<LoadInst>(u))
{
if (!load->getType()->isPointerTy())
{
if (categorized_instructions.insert(u).second)
{
load_paths->push_back({i, path});
}
}
}
else if (auto *call_inst = dyn_cast<CallInst>(u))
{
if (call_inst->getCalledFunction()->getName().equals("free"))
{
if (categorized_instructions.insert(u).second)
{
Debug(errs() << "Free call on shared memory:\n";);
Debug(call_inst->dump(););
free_paths->push_back(path);
}
}
}
}
}
}


template <class T>
std::vector<Value *>
CatoPass::get_memory_access_indices(Module &M,
std::pair<int, std::vector<Value *>> &categorized_path)
{
auto &path = categorized_path.second;
auto *instruction = dyn_cast<T>(path[categorized_path.first]);

std::vector<Value *> indices;

IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();

Debug(errs() << "Analyzing memory access:\n";);
Debug(instruction->dump(););
for (int i = categorized_path.first - 1; i >= 0; i--)
{
Debug(errs() << " |";);
Debug(path[i]->dump(););
}

Type *ptr_type = nullptr;
if (auto *first_inst = dyn_cast<LoadInst>(path[0]))
{
ptr_type = path[0]->getType();
}
else
{
for (int i = 0; i < categorized_path.first; i++)
{
Value *user = path[i];
if (auto *cast = dyn_cast<BitCastInst>(user))
{
ptr_type = cast->getType();
}
}
}

if (ptr_type == nullptr)
{
errs() << "Error: could not find out type of memory access\n";
}

int ptr_depth = get_pointer_depth(ptr_type);

builder.SetInsertPoint(instruction);

if (ptr_depth == 1)
{
if (auto *gep = dyn_cast<GetElementPtrInst>(instruction->getPointerOperand()))
{
if (auto *gep2 = dyn_cast<GetElementPtrInst>(gep->getOperand(0)))
{
errs() << "Error: Unknown IR Pattern for evaluation of accessing index "
"for 1d array.\n";
gep2->dump();
}

Value *index = gep->getOperand(1);
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: ";);
Debug(index->dump(););

indices.push_back(builder.getInt32(1));
indices.push_back(index);
}
else
{
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: 0\n";);

indices.push_back(builder.getInt32(1));
indices.push_back(builder.getInt64(0));
}
}
else if (ptr_depth == 2)
{
Debug(errs() << "Pointer depth of 2\n";);
if (auto *last_inst = dyn_cast<LoadInst>(instruction->getPointerOperand()))
{
if (auto *gep = dyn_cast<GetElementPtrInst>(last_inst->getPointerOperand()))
{
Value *index = gep->getOperand(1);
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: ";);
Debug(index->dump(););
Debug(errs() << "  ->index2: 0\n";);

indices.push_back(builder.getInt32(2));
indices.push_back(index);
indices.push_back(builder.getInt64(0));
}
else
{
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: 0\n";);
Debug(errs() << "  ->index2: 0\n";);

indices.push_back(builder.getInt32(2));
indices.push_back(builder.getInt64(0));
indices.push_back(builder.getInt64(0));
}
}
else if (auto *gep2 = dyn_cast<GetElementPtrInst>(instruction->getPointerOperand()))
{
if (auto *inst = dyn_cast<LoadInst>(gep2->getPointerOperand()))
{
if (auto *gep1 = dyn_cast<GetElementPtrInst>(inst->getPointerOperand()))
{
Value *index1 = gep1->getOperand(1);
Value *index2 = gep2->getOperand(1);
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: ";);
Debug(index1->dump(););
Debug(errs() << "  ->index2: ";);
Debug(index2->dump(););

indices.push_back(builder.getInt32(2));
indices.push_back(index1);
indices.push_back(index2);
}
else
{
Value *index2 = gep2->getOperand(1);
Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  ->index1: 0\n";);
Debug(errs() << "  ->index2: ";);
Debug(index2->dump(););

indices.push_back(builder.getInt32(2));
indices.push_back(builder.getInt64(0));
indices.push_back(index2);
}
}
else
{
errs() << "Unrecognized pointer access pattern!\n";
}
}
else
{
errs() << "Unrecognized pointer access pattern!\n";
}
}
else if (ptr_depth == 3)
{
Debug(errs() << "Pointer depth of 3\n";);

if (auto *load = dyn_cast<LoadInst>(instruction->getPointerOperand()))
{
if (auto *load1 = dyn_cast<LoadInst>(load->getPointerOperand()))
{
if (auto *gep2 = dyn_cast<GetElementPtrInst>(load1->getPointerOperand()))
{
indices.push_back(builder.getInt32(3));
indices.push_back(gep2->getOperand(1));
indices.push_back(builder.getInt64(0));
indices.push_back(builder.getInt64(0));
}
else
{
indices.push_back(builder.getInt32(3));
indices.push_back(builder.getInt64(0));
indices.push_back(builder.getInt64(0));
indices.push_back(builder.getInt64(0));
}
}
else if (auto *gep1 = dyn_cast<GetElementPtrInst>(load->getPointerOperand()))
{
if (auto *load2 = dyn_cast<LoadInst>(gep1->getPointerOperand()))
{
if (auto *gep2 = dyn_cast<GetElementPtrInst>(load2->getPointerOperand()))
{
indices.push_back(builder.getInt32(3));
indices.push_back(gep2->getOperand(1));
indices.push_back(gep1->getOperand(1));
indices.push_back(builder.getInt64(0));
}
else
{
indices.push_back(builder.getInt32(3));
indices.push_back(builder.getInt64(0));
indices.push_back(gep1->getOperand(1));
indices.push_back(builder.getInt64(0));
}
}
else
{
errs() << "Error in 3d index calculation\n";
}
}
}
else if (auto *gep = dyn_cast<GetElementPtrInst>(instruction->getPointerOperand()))
{
if (auto *load1 = dyn_cast<LoadInst>(gep->getPointerOperand()))
{
if (auto *load2 = dyn_cast<LoadInst>(load1->getPointerOperand()))
{
if (auto *gep2 = dyn_cast<GetElementPtrInst>(load2->getPointerOperand()))
{
indices.push_back(builder.getInt32(3));
indices.push_back(gep2->getOperand(1));
indices.push_back(builder.getInt64(0));
indices.push_back(gep->getOperand(1));
}
else
{
indices.push_back(builder.getInt32(3));
indices.push_back(builder.getInt64(0));
indices.push_back(builder.getInt64(0));
indices.push_back(gep->getOperand(1));
}
}
else if (auto *gep2 = dyn_cast<GetElementPtrInst>(load1->getPointerOperand()))
{
if (auto *load2 = dyn_cast<LoadInst>(gep2->getPointerOperand()))
{
if (auto *gep3 =
dyn_cast<GetElementPtrInst>(load2->getPointerOperand()))
{
indices.push_back(builder.getInt32(3));
indices.push_back(gep3->getOperand(1));
indices.push_back(gep2->getOperand(1));
indices.push_back(gep->getOperand(1));
}
else
{
indices.push_back(builder.getInt32(3));
indices.push_back(builder.getInt64(0));
indices.push_back(gep2->getOperand(1));
indices.push_back(gep->getOperand(1));
}
}
else
{
errs() << "Error in 3d index calculation\n";
}
}
}
else
{
errs() << "Error in 3d index calculation\n";
}
}

Debug(errs() << "Pointer access at index:\n";);
Debug(errs() << "  -> index1: ";);
Debug(indices[1]->dump(););
Debug(errs() << "  -> index2: ";);
Debug(indices[2]->dump(););
Debug(errs() << "  -> index3: ";);
Debug(indices[3]->dump(););
}
else
{
errs() << "Error: Pointer depth > 3 not supported at the moment!\n";
}

return indices;
}


void CatoPass::replace_sequential_shared_memory_accesses(Module &M, RuntimeHandler &runtime)
{
std::vector<std::vector<Value *>> paths;

for (auto *allocate_call : runtime.functions.allocate_shared_memory->users())
{
UserTree T(allocate_call);
auto tmp_paths = T.get_all_paths();
paths.insert(paths.end(), tmp_paths.begin(), tmp_paths.end());
}

std::vector<Value *> struct_geps;
int path_count_before = paths.size();
for (int j = 0; j < path_count_before; j++)
{
auto &path = paths[j];
for (unsigned int i = 0; i < path.size(); i++)
{
Value *u = path[i];

if (auto *store = dyn_cast<StoreInst>(u))
{
if (store->getValueOperand()->getType()->isPointerTy())
{
Value *store_destination = store->getPointerOperand();

if (auto *gep = dyn_cast<GetElementPtrInst>(store_destination))
{
if (gep->getPointerOperandType()
->getPointerElementType()
->isStructTy())
{
errs() << "FOUND STORE INTO STRUCT:\n";
errs() << "    ";
gep->getPointerOperand()->dump();

for (auto *user : gep->getPointerOperand()->users())
{
if (auto *gep2 = dyn_cast<GetElementPtrInst>(user))
{
if (gep2->getOperand(1) == gep->getOperand(1) &&
gep2->getOperand(2) == gep->getOperand(2))
{
for (auto *user : gep2->users())
{
if (auto *load = dyn_cast<LoadInst>(user))
{
UserTree T(load);
auto tmp_paths = T.get_all_paths();
paths.insert(paths.end(), tmp_paths.begin(),
tmp_paths.end());
for (auto &p : tmp_paths)
{
errs() << "STRUCT PATH:\n";
for (auto *e : p)
{
e->dump();
}
}
}
}
}
}
}
}
}
}
}
}
}

std::vector<StoreInst *> base_pointer_stores;
for (auto &path : paths)
{
for (unsigned int i = 0; i < path.size(); i++)
{
Value *u = path[i];

if (auto *store = dyn_cast<StoreInst>(u))
{
if (store->getValueOperand()->getType()->isPointerTy())
{
Value *store_destination = store->getPointerOperand();
if (std::find(path.begin(), path.begin() + i, store_destination) ==
path.begin() + i)
{
if (auto *alloca = dyn_cast<AllocaInst>(store_destination))
{
base_pointer_stores.push_back(store);
}
}
}
}
}
}

replace_sequential_pointers_to_shared_memory(M, runtime, base_pointer_stores);

for (auto &path : paths)
{
Debug(errs() << "PATH:\n";);
for (auto &u : path)
{
Debug(u->dump(););
}
Debug(errs() << "PATH END\n";);
}
Debug(errs() << "========================================\n";);


std::vector<std::pair<int, std::vector<Value *>>> store_paths, load_paths, ptr_store_paths;
std::vector<std::vector<Value *>> free_paths;

categorize_memory_access_paths(paths, &store_paths, &load_paths, &ptr_store_paths,
&free_paths);

IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();
for (auto &p : load_paths)
{
auto &path = p.second;
auto *load = dyn_cast<LoadInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<LoadInst>(M, p);

Function *current_func = load->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j < called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(load);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else if (!path[0]->getType()->isVoidTy())
{
builder.SetInsertPoint(load);
Value *void_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
}
else
{
args.insert(args.begin(), path[0]);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(load->getFunction()->getEntryBlock().getFirstNonPHI());
Value *load_value = builder.CreateAlloca(load->getType());
builder.SetInsertPoint(load);
Value *void_ptr = builder.CreateBitCast(load_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
CallInst *load_call =
builder.CreateCall(runtime.functions.shared_memory_sequential_load, args);
Value *bitcast = builder.CreateBitCast(void_ptr, load->getPointerOperandType());
Value *new_load = builder.CreateLoad(bitcast->getType()->getPointerElementType(),
bitcast, "CATO: New Load Call");
load->replaceAllUsesWith(new_load);
load->eraseFromParent();
}
}

for (auto &p : store_paths)
{
auto &path = p.second;
auto *store = dyn_cast<StoreInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<StoreInst>(M, p);

Function *current_func = store->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j < called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(store);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else if (!path[0]->getType()->isVoidTy())
{
builder.SetInsertPoint(store);
Value *void_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
}
else
{
args.insert(args.begin(), path[0]);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(store->getFunction()->getEntryBlock().getFirstNonPHI());
Value *store_value = builder.CreateAlloca(store->getValueOperand()->getType());
builder.SetInsertPoint(store);
builder.CreateStore(store->getOperand(0), store_value);
Value *void_ptr = builder.CreateBitCast(store_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
Value *new_store_call =
builder.CreateCall(runtime.functions.shared_memory_sequential_store, args);
store->replaceAllUsesWith(new_store_call);
store->eraseFromParent();
}
}

for (auto &p : ptr_store_paths)
{
auto &path = p.second;
auto *store = dyn_cast<StoreInst>(path[p.first]);

Debug(errs() << "Pointer store Path:\n";);
for (auto &x : path)
{
Debug(x->dump(););
}

if (get_pointer_depth(store->getValueOperand()->getType()) < 3)
{
builder.SetInsertPoint(store);

Value *index = nullptr;

if (auto *gep = dyn_cast<GetElementPtrInst>(store->getPointerOperand()))
{
index = gep->getOperand(1);
}
else
{
index = builder.getInt64(0);
}

Value *dest_ptr =
builder.CreateBitCast(store->getPointerOperand(), Type::getInt8PtrTy(Ctx));
Value *source_ptr =
builder.CreateBitCast(store->getValueOperand(), Type::getInt8PtrTy(Ctx));
std::vector<Value *> args = {dest_ptr, source_ptr, index};

auto *new_call =
builder.CreateCall(runtime.functions.shared_memory_pointer_store, args);
store->replaceAllUsesWith(new_call);
store->eraseFromParent();
}
}

for (auto &path : free_paths)
{
if (auto *free_call = dyn_cast<CallInst>(path.back()))
{
Debug(errs() << "Analyzing free call instruction:\n";);
Debug(free_call->dump(););

for (int i = path.size() - 2; i >= 0; i--)
{
Debug(errs() << " |";);
Debug(path[i]->dump(););
}

builder.SetInsertPoint(free_call);
auto new_free_call = builder.CreateCall(runtime.functions.shared_memory_free,
free_call->getOperand(0));
free_call->replaceAllUsesWith(new_free_call);
free_call->eraseFromParent();
}
else
{
errs() << "Error: free path ends in non call instruction\n";
}
}
}


void CatoPass::replace_sequential_pointers_to_shared_memory(
Module &M, RuntimeHandler &runtime, std::vector<StoreInst *> &base_ptr_stores)
{
std::vector<Value *> pointers_to_shared_memory;

for (auto &ptr_store : base_ptr_stores)
{
Debug(errs() << "BASE_PTR_STORE: ";);
Debug(ptr_store->dump(););
Debug(errs() << "-->";);
Debug(ptr_store->getPointerOperand()->dump(););
pointers_to_shared_memory.push_back(ptr_store->getPointerOperand());
}

std::vector<std::vector<Value *>> paths;

for (auto &value : pointers_to_shared_memory)
{
for (auto *user : value->users())
{
if (auto *load = dyn_cast<LoadInst>(user))
{
UserTree T(load);
auto tmp_paths = T.get_all_paths();
paths.insert(paths.end(), tmp_paths.begin(), tmp_paths.end());
}
}
}

Debug(errs() << "POINTER TO SHARED MEMORY PATHS: \n";);
for (auto &path : paths)
{
Debug(errs() << "PATH BEGIN\n";);
for (auto &u : path)
{
Debug(errs() << "    ");
Debug(u->dump(););
}
Debug(errs() << "PATH END\n";);
}

std::vector<std::pair<int, std::vector<Value *>>> store_paths;
std::vector<std::pair<int, std::vector<Value *>>> load_paths;
std::set<Value *> categorized_instructions;

for (auto &path : paths)
{
for (unsigned int i = 0; i < path.size(); i++)
{
Value *u = path[i];

if (auto *store = dyn_cast<StoreInst>(u))
{
if (!store->getValueOperand()->getType()->isPointerTy())
{
if (categorized_instructions.insert(u).second)
{
store_paths.push_back({i, path});
}
}
}
else if (auto *load = dyn_cast<LoadInst>(u))
{
if (!load->getType()->isPointerTy())
{
if (categorized_instructions.insert(u).second)
{
load_paths.push_back({i, path});
}
}
}
}
}

IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();

for (auto &p : load_paths)
{
auto &path = p.second;
auto *load = dyn_cast<LoadInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<LoadInst>(M, p);

Function *current_func = load->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
errs() << "   CallInst: ";
call_inst->dump();

auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j < called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(load);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else
{
builder.SetInsertPoint(load);
Value *void_base_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));

args.insert(args.begin(), void_base_ptr);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(load->getFunction()->getEntryBlock().getFirstNonPHI());
Value *load_value = builder.CreateAlloca(load->getType());
builder.SetInsertPoint(load);
Value *void_ptr = builder.CreateBitCast(load_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
auto *load_call =
builder.CreateCall(runtime.functions.shared_memory_sequential_load, args);
Value *bitcast = builder.CreateBitCast(void_ptr, load->getPointerOperandType());
Value *new_load = builder.CreateLoad(bitcast->getType()->getPointerElementType(),
bitcast, "CATO: Replacement of load call");
load->replaceAllUsesWith(new_load);
load->eraseFromParent();
}
}

for (auto &p : store_paths)
{
auto &path = p.second;
auto *store = dyn_cast<StoreInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<StoreInst>(M, p);

Function *current_func = store->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
errs() << "   CallInst: ";
call_inst->dump();

auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j < called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(store);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else
{
builder.SetInsertPoint(store);
Value *void_base_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));

args.insert(args.begin(), void_base_ptr);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(store->getFunction()->getEntryBlock().getFirstNonPHI());
Value *store_value = builder.CreateAlloca(store->getValueOperand()->getType());
builder.SetInsertPoint(store);
builder.CreateStore(store->getOperand(0), store_value);
Value *void_ptr = builder.CreateBitCast(store_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
Value *new_store_call =
builder.CreateCall(runtime.functions.shared_memory_sequential_store, args);
store->replaceAllUsesWith(new_store_call);
store->eraseFromParent();
}
}
}


void CatoPass::replace_microtask_shared_memory_accesses(
Module &M, RuntimeHandler &runtime, std::vector<std::unique_ptr<Microtask>> &microtasks)
{
for (auto &microtask : microtasks)
{
std::vector<Value *> shared_value_variables, shared_pointer_variables,
shared_struct_variables;

for (auto &shared_variable : microtask->get_shared_variables())
{
if (get_pointer_depth(shared_variable) == 1)
{
if (shared_variable->getType()->getPointerElementType()->isStructTy())
{
shared_struct_variables.push_back(shared_variable);
Debug(errs() << "Shared struct variable in Microtask: ";);
Debug(shared_variable->dump(););
}
else
{
shared_value_variables.push_back(shared_variable);
Debug(errs() << "Shared value variable in Microtask: ";);
Debug(shared_variable->dump(););
}
}
else
{
shared_pointer_variables.push_back(shared_variable);
Debug(errs() << "Shared pointer variable in Microtask: ";);
Debug(shared_variable->dump(););
}
}

for (auto *struct_var : shared_struct_variables)
{
for (auto *user : struct_var->users())
{
if (get_pointer_depth(user->getType()) == 1)
{
shared_value_variables.push_back(user);
}
else if (get_pointer_depth(user->getType()) > 1)
{
shared_pointer_variables.push_back(user);
}
}
}

for (auto &single_value_var : shared_value_variables)
{
Debug(errs() << "Analysing single value shared variable: ";);
Debug(single_value_var->dump(););

bool has_stores = false;
for (auto *user : single_value_var->users())
{
if (auto *store = dyn_cast<StoreInst>(user))
{
has_stores = true;
break;
}
}

if (has_stores)
{
IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();
builder.SetInsertPoint(
microtask->get_function()->getEntryBlock().getFirstNonPHI());

Value *void_ptr = nullptr;
if (auto *gep = dyn_cast<GetElementPtrInst>(single_value_var))
{
auto *new_gep =
builder.CreateGEP(nullptr, gep->getPointerOperand(),
{gep->getOperand(1), gep->getOperand(2)});
void_ptr = builder.CreateBitCast(new_gep, Type::getInt8PtrTy(Ctx));
}
else
{
void_ptr =
builder.CreateBitCast(single_value_var, Type::getInt8PtrTy(Ctx));
}
Type *type = single_value_var->getType();

std::vector<Value *> args = {void_ptr,
builder.getInt32(get_mpi_datatype(type))};

builder.CreateCall(runtime.functions.allocate_shared_value, args);


Debug(errs() << "Users for shared value:\n";);
for (auto *user : single_value_var->users())
{
if (auto *store = dyn_cast<StoreInst>(user))
{
Debug(errs() << "Replacing store instruction for shared value: ";);
Debug(store->dump(););

builder.SetInsertPoint(
store->getFunction()->getEntryBlock().getFirstNonPHI());
Value *store_value =
builder.CreateAlloca(store->getValueOperand()->getType());
builder.SetInsertPoint(store);
builder.CreateStore(store->getOperand(0), store_value);
Value *store_value_ptr =
builder.CreateBitCast(store_value, Type::getInt8PtrTy(Ctx));

std::vector<Value *> args = {void_ptr, store_value_ptr};

Value *new_store_call =
builder.CreateCall(runtime.functions.shared_value_store, args);
store->replaceAllUsesWith(new_store_call);
store->eraseFromParent();
}
else if (auto *load = dyn_cast<LoadInst>(user))
{
Debug(errs() << "Replacing load instruction for shared value: ";);
Debug(load->dump(););

std::vector<Value *> args = {void_ptr, void_ptr};

builder.SetInsertPoint(load);
builder.CreateCall(runtime.functions.shared_value_load, args);
}
}

auto return_instructions =
get_instruction_in_function<ReturnInst>(microtask->get_function());
for (auto &ret : return_instructions)
{
builder.SetInsertPoint(ret);
builder.CreateCall(runtime.functions.shared_value_synchronize, void_ptr);
}
}
}

std::vector<std::vector<Value *>> paths;
for (auto *pointer_var : shared_pointer_variables)
{
for (auto *user : pointer_var->users())
{
Debug(errs() << "USER: ";);
Debug(user->dump(););
UserTree T(user);
auto tmp_paths = T.get_all_paths();
paths.insert(paths.end(), tmp_paths.begin(), tmp_paths.end());
}
}

for (auto &path : paths)
{
Debug(errs() << "MICROTASK PATH:\n";);
for (auto &u : path)
{
Debug(u->dump(););
}
Debug(errs() << "MICROTASK PATH END\n";);
}

Debug(errs() << "========================================\n";);

std::vector<std::pair<int, std::vector<Value *>>> store_paths, load_paths,
ptr_store_paths;
std::vector<std::vector<Value *>> free_paths;

categorize_memory_access_paths(paths, &store_paths, &load_paths, &ptr_store_paths,
&free_paths);

IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();
for (auto &p : load_paths)
{
auto &path = p.second;
auto *load = dyn_cast<LoadInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<LoadInst>(M, p);

Function *current_func = load->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
errs() << "   CallInst: ";
call_inst->dump();

auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j <
called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(load);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else
{
builder.SetInsertPoint(load);
Value *void_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(load->getFunction()->getEntryBlock().getFirstNonPHI());
Value *load_value = builder.CreateAlloca(load->getType());
builder.SetInsertPoint(load);
Value *void_ptr = builder.CreateBitCast(load_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
CallInst *load_call =
builder.CreateCall(runtime.functions.shared_memory_load, args);
Value *bitcast =
builder.CreateBitCast(void_ptr, load->getPointerOperandType());
LoadInst *new_load = builder.CreateLoad(bitcast->getType(), bitcast);
load->replaceAllUsesWith(new_load);
load->eraseFromParent();
}
}

for (auto &p : store_paths)
{
auto &path = p.second;
auto *store = dyn_cast<StoreInst>(path[p.first]);

std::vector<Value *> args = get_memory_access_indices<StoreInst>(M, p);

Function *current_func = store->getFunction();
if (dyn_cast<Instruction>(path[0])->getFunction() != current_func)
{
for (int i = p.first; i > 0; i--)
{
if (auto *call_inst = dyn_cast<CallInst>(path[i]))
{
errs() << "   CallInst: ";
call_inst->dump();

auto *last_inst = path[i - 1];
Value *matching_argument = nullptr;

for (unsigned int j = 0; j < call_inst->arg_size(); j++)
{
if (last_inst == call_inst->getArgOperand(j))
{
Function *called_function = call_inst->getCalledFunction();
if (called_function->arg_begin() + j <
called_function->arg_end())
{
matching_argument = called_function->arg_begin() + j;
}
}
}

builder.SetInsertPoint(store);
Value *void_ptr =
builder.CreateBitCast(matching_argument, Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
break;
}
}
}
else
{
builder.SetInsertPoint(store);
Value *void_ptr = builder.CreateBitCast(path[0], Type::getInt8PtrTy(Ctx));
args.insert(args.begin(), void_ptr);
}

if (args.size() >= 3)
{
builder.SetInsertPoint(store->getFunction()->getEntryBlock().getFirstNonPHI());
Value *store_value = builder.CreateAlloca(store->getValueOperand()->getType());
builder.SetInsertPoint(store);
builder.CreateStore(store->getOperand(0), store_value);
Value *void_ptr = builder.CreateBitCast(store_value, Type::getInt8PtrTy(Ctx));
args.insert(args.begin() + 1, void_ptr);
Value *new_store_call =
builder.CreateCall(runtime.functions.shared_memory_store, args);
store->replaceAllUsesWith(new_store_call);
store->eraseFromParent();
}
}

for (auto &path : free_paths)
{
if (auto *free_call = dyn_cast<CallInst>(path.back()))
{
Debug(errs() << "Analyzing free call instruction:\n";);
Debug(free_call->dump(););

for (int i = path.size() - 2; i >= 0; i--)
{
Debug(errs() << " |";);
Debug(path[i]->dump(););
}

builder.SetInsertPoint(free_call);
auto new_free_call = builder.CreateCall(runtime.functions.shared_memory_free,
free_call->getOperand(0));
free_call->replaceAllUsesWith(new_free_call);
free_call->eraseFromParent();
}
else
{
errs() << "Error: free path ends in non call instruction\n";
}
}
}
}


std::vector<std::unique_ptr<Microtask>> CatoPass::find_microtasks(Module &M)
{
std::vector<std::unique_ptr<Microtask>> microtasks;
auto kmpc_fork_call_users = get_function_users(M, "__kmpc_fork_call");
for (auto &fork_call : kmpc_fork_call_users)
{
if (auto *fork_call_inst = dyn_cast<CallInst>(fork_call))
{
microtasks.push_back(std::make_unique<Microtask>(fork_call_inst));
}
}

return microtasks;
}


void CatoPass::replace_fork_calls(Module &M, RuntimeHandler &runtime,
std::vector<std::unique_ptr<Microtask>> &microtasks)
{
LLVMContext &Ctx = M.getContext();
IRBuilder<> builder(Ctx);

for (auto &microtask : microtasks)
{
auto *fork_call_inst = microtask->get_fork_call();

if (auto *num_shared_vars = dyn_cast<ConstantInt>(fork_call_inst->getArgOperand(1)))
{
std::vector<Value *> args;

args.push_back(Constant::getNullValue(Type::getInt32PtrTy(Ctx)));
args.push_back(Constant::getNullValue(Type::getInt32PtrTy(Ctx)));

for (int i = 0; i < num_shared_vars->getSExtValue(); i++)
{
args.push_back(fork_call_inst->getArgOperand(3 + i));
}

builder.SetInsertPoint(fork_call_inst);
builder.CreateCall(microtask->get_function(), args);
builder.CreateCall(runtime.functions.mpi_barrier);
fork_call_inst->eraseFromParent();
}
}
}


void CatoPass::replace_memory_allocations(Module &M, RuntimeHandler &runtime)
{
LLVMContext &Ctx = M.getContext();
IRBuilder<> builder(Ctx);

auto alloc_calls = find_memory_allocations(M);

for (auto &allocation : alloc_calls)
{
CallInst *inst = allocation->get_allocation_call();
Value *size = allocation->get_allocation_size();
Type *type = allocation->get_allocation_type();

builder.SetInsertPoint(inst);

std::vector<Value *> args = {size, builder.getInt32(get_mpi_datatype(type)),
builder.getInt32(get_pointer_depth(type))};

auto new_call = builder.CreateCall(runtime.functions.allocate_shared_memory, args);
new_call->takeName(inst);
inst->replaceAllUsesWith(new_call);
inst->eraseFromParent();
}
}


void CatoPass::replace_parallel_for(Module &M, RuntimeHandler &runtime,
std::vector<std::unique_ptr<Microtask>> &microtasks)
{
for (auto &microtask : microtasks)
{
std::vector<ParallelForData> *parallel_for_data_vec = microtask->get_parallel_for();
if (parallel_for_data_vec != nullptr)
{
for (ParallelForData parallel_for_data : *parallel_for_data_vec)
{
Debug(errs() << "Replacing parallel for in Microtask.\n";);
Debug(errs() << "    ";);
Debug(parallel_for_data.init->dump(););
Debug(errs() << "    ";);
Debug(parallel_for_data.fini->dump(););

Value *lower_bound = parallel_for_data.init->getOperand(4);
Value *upper_bound = parallel_for_data.init->getOperand(5);
Value *increment = parallel_for_data.init->getOperand(7);
Debug(errs() << "Loop lower bound: ");
Debug(lower_bound->dump(););
Debug(errs() << "Loop upper bound: ";);
Debug(upper_bound->dump(););
Debug(errs() << "Loop increment: ";);
Debug(increment->dump(););

LLVMContext &Ctx = M.getContext();
IRBuilder<> builder(parallel_for_data.init);

std::vector<Value *> args = {lower_bound, upper_bound, increment};

CallInst *new_call = nullptr;
if (lower_bound->getType() == Type::getInt32PtrTy(Ctx))
{
new_call = builder.CreateCall(
runtime.functions.modify_parallel_for_bounds_4, args);
}
else if (lower_bound->getType() == Type::getInt64PtrTy(Ctx))
{
new_call = builder.CreateCall(
runtime.functions.modify_parallel_for_bounds_8, args);
}
parallel_for_data.init->replaceAllUsesWith(new_call);
parallel_for_data.init->eraseFromParent();
parallel_for_data.fini->eraseFromParent();
}
}
}
}


void CatoPass::replace_reductions(Module &M, RuntimeHandler &runtime,
std::vector<std::unique_ptr<Microtask>> &microtasks)
{
for (auto &microtask : microtasks)
{
std::vector<ReductionData> *reduction_data_vec = microtask->get_reductions();
if (reduction_data_vec != nullptr)
{
for (ReductionData reduction_data : *reduction_data_vec)
{
Debug(errs() << "Microtask contains a reduction operation.\n";);
Debug(errs() << "    Reduce: ";);
Debug(reduction_data.reduce->dump(););
Debug(errs() << "    End Reduce: ";);
Debug(reduction_data.end_reduce->dump(););

Value *num_reduction_variables = reduction_data.reduce->getOperand(2);
Value *reduction_data_size = reduction_data.reduce->getOperand(3);
Function *reduction_function = dyn_cast<Function>(
reduction_data.reduce->getOperand(5)->stripPointerCasts());
Value *reduction_lst = nullptr;
if (auto *bitcast =
dyn_cast<BitCastInst>(reduction_data.reduce->getOperand(4)))
{
reduction_lst = bitcast->getOperand(0);
}
Value *local_reduction_var = nullptr;
for (User *user : reduction_lst->users())
{
if (auto *gep = dyn_cast<GetElementPtrInst>(user))
{
for (User *user2 : gep->users())
{
if (auto *store = dyn_cast<StoreInst>(user2))
{
local_reduction_var = store->getValueOperand();
}
}
}
}

Debug(errs() << "    Number reduction variables: ";);
Debug(num_reduction_variables->dump(););
Debug(errs() << "    Reduction data size: ";);
Debug(reduction_data_size->dump(););
Debug(errs() << "    Reduction reduction_lst: ";);
Debug(reduction_lst->dump(););
Debug(errs() << "    Reduction local var: ";);
Debug(local_reduction_var->dump(););

if (auto *switch_inst =
dyn_cast<SwitchInst>(reduction_data.reduce->getNextNode()))
{
BasicBlock *case_default = switch_inst->getDefaultDest();
BasicBlock *case1 = switch_inst->getSuccessor(1);
BasicBlock *case2 = switch_inst->getSuccessor(2);
AtomicRMWInst::BinOp bin_op = AtomicRMWInst::BinOp::BAD_BINOP;
Value *reduction_target = nullptr;
bool integer_type = true;

Debug(errs() << "";);
for (Instruction &instr : *case2)
{
if (auto *atomicrmw = dyn_cast<AtomicRMWInst>(&instr))
{
bin_op = atomicrmw->getOperation();
switch (bin_op)
{
case AtomicRMWInst::BinOp::Add:
Debug(errs() << "    Reduction operation: ADD\n";);
break;
case AtomicRMWInst::BinOp::Max:
Debug(errs() << "    Reduction operation: MAX\n";);
break;
case AtomicRMWInst::BinOp::Min:
Debug(errs() << "    Reduction operation: MIN\n";);
break;
default:
errs() << "    Reduction operation: UNKWOWN\n";
break;
}

reduction_target = atomicrmw->getOperand(0);
Debug(errs() << "    Shared variable reduction target: ";);
Debug(reduction_target->dump());
}
else if (auto *branch = dyn_cast<BranchInst>(&instr))
{
if (bin_op == AtomicRMWInst::BinOp::BAD_BINOP)
{
integer_type = false;
Debug(errs() << "   Reduction on non integer type\n";);

for (auto &instr : *(branch->getSuccessor(0)))
{
if (auto *binary_op = dyn_cast<BinaryOperator>(&instr))
{
if (binary_op->getOpcode() ==
Instruction::BinaryOps::FAdd)
{
Debug(errs() << "    Reduction operation: ADD\n";);
bin_op = AtomicRMWInst::BinOp::Add;
break;
}
}
else if (auto *cmp_inst = dyn_cast<FCmpInst>(&instr))
{
if (cmp_inst->getPredicate() ==
CmpInst::Predicate::FCMP_OGT)
{
Debug(errs() << "    Reduction operation: MAX\n";);
bin_op = AtomicRMWInst::BinOp::Max;
break;
}
else if (cmp_inst->getPredicate() ==
CmpInst::Predicate::FCMP_OLT)
{
Debug(errs() << "    Reduction operation: MIN\n";);
bin_op = AtomicRMWInst::BinOp::Min;
break;
}
else
{
errs() << "Reduction operation: UNKNOWN\n";
}
}
}

for (Instruction &instr : *case2)
{
if (auto *load = dyn_cast<LoadInst>(&instr))
{
if (auto *bitcast =
dyn_cast<BitCastInst>(load->getOperand(0)))
{
reduction_target = bitcast->getOperand(0);
Debug(errs() << "    Shared variable reduction "
"target: ";);
Debug(reduction_target->dump());
}
}
}
}
}
}

LLVMContext &Ctx = M.getContext();
IRBuilder<> builder(switch_inst);

builder.CreateBr(case_default);
switch_inst->eraseFromParent();
reduction_data.reduce->eraseFromParent();
reduction_data.end_reduce->eraseFromParent();

std::vector<Value *> args = {
local_reduction_var, builder.getInt32(bin_op),
builder.getInt32(get_mpi_datatype(reduction_target))};
builder.SetInsertPoint(case_default->getFirstNonPHI());
builder.CreateCall(runtime.functions.reduce_local_vars, args);

CallInst *mpi_rank = builder.CreateCall(runtime.functions.get_mpi_rank);
BasicBlock *split_block =
SplitBlock(case_default, &*builder.GetInsertPoint());
BasicBlock *master_reduction_block =
BasicBlock::Create(Ctx, "master_reduction", case_default->getParent());
builder.SetInsertPoint(case_default->getTerminator());
Value *condition = builder.CreateICmpEQ(mpi_rank, builder.getInt32(0));
builder.CreateCondBr(condition, master_reduction_block, split_block);
case_default->getTerminator()->eraseFromParent();

builder.SetInsertPoint(master_reduction_block);

Value *load = builder.CreateLoad(
dyn_cast<BitCastInst>(local_reduction_var)
->getOperand(0)
->getType()
->getPointerElementType(),
dyn_cast<BitCastInst>(local_reduction_var)->getOperand(0));

if (integer_type)
{
auto *add =
builder.CreateAtomicRMW(bin_op, reduction_target, load,
MaybeAlign(), AtomicOrdering::Monotonic);
}
else
{
if (bin_op == AtomicRMWInst::BinOp::Add)
{
auto *add = builder.CreateAtomicRMW(
AtomicRMWInst::BinOp::FAdd, reduction_target, load,
MaybeAlign(), AtomicOrdering::Monotonic);
}
else if (bin_op == AtomicRMWInst::BinOp::Max)
{
auto *target_load = builder.CreateLoad(
reduction_target->getType()->getPointerElementType(),
reduction_target);
auto *fcmp = builder.CreateFCmp(CmpInst::Predicate::FCMP_OGT,
target_load, load);
auto *select = builder.CreateSelect(fcmp, target_load, load);
auto *store = builder.CreateStore(select, reduction_target);
}
else if (bin_op == AtomicRMWInst::BinOp::Min)
{
auto *target_load = builder.CreateLoad(
reduction_target->getType()->getPointerElementType(),
reduction_target);
auto *fcmp = builder.CreateFCmp(CmpInst::Predicate::FCMP_OLT,
target_load, load);
auto *select = builder.CreateSelect(fcmp, target_load, load);
auto *store = builder.CreateStore(select, reduction_target);
}
else
{
}
}
builder.CreateBr(split_block);

reduction_function->eraseFromParent();
}
else
{
errs() << "Error: Unknown IR pattern for a reduction pragma!\n";
}
}
}
}
}


void CatoPass::replace_criticals(Module &M, RuntimeHandler &runtime,
std::vector<std::unique_ptr<Microtask>> &microtasks)
{
for (auto &microtask : microtasks)
{
std::vector<CriticalData> *critical_data_vec = microtask->get_critical();
if (critical_data_vec != nullptr)
{
IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();

builder.SetInsertPoint(
microtask->get_function()->getEntryBlock().getFirstNonPHI());
Value *mpi_mutex_call =
builder.CreateCall(runtime.functions.critical_section_init);

for (CriticalData critical_data : *critical_data_vec)
{
Debug(errs() << "Microtask contains a critical section.\n";);
Debug(errs() << "    Critical: ";);
Debug(critical_data.critical->dump(););
Debug(errs() << "    End Critical: ";);
Debug(critical_data.end_critical->dump(););

builder.SetInsertPoint(critical_data.critical);
Value *critical_enter_call = builder.CreateCall(
runtime.functions.critical_section_enter, mpi_mutex_call);
critical_data.critical->replaceAllUsesWith(critical_enter_call);
critical_data.critical->eraseFromParent();

builder.SetInsertPoint(critical_data.end_critical);
Value *critical_leave_call = builder.CreateCall(
runtime.functions.critical_section_leave, mpi_mutex_call);
critical_data.end_critical->replaceAllUsesWith(critical_leave_call);
critical_data.end_critical->eraseFromParent();
}

auto return_instructions =
get_instruction_in_function<ReturnInst>(microtask->get_function());
for (auto &ret : return_instructions)
{
builder.SetInsertPoint(ret);
builder.CreateCall(runtime.functions.critical_section_finalize,
mpi_mutex_call);
}
}
}
}


void CatoPass::replace_memory_deallocations(Module &M, RuntimeHandler &runtime)
{
LLVMContext &Ctx = M.getContext();
IRBuilder<> builder(Ctx);

std::vector<CallInst *> dealloc_calls = find_memory_deallocations(M);

for (CallInst *&deallocation : dealloc_calls)
{
builder.SetInsertPoint(deallocation);

Value *new_dealloc_call = builder.CreateCall(runtime.functions.shared_memory_free,
deallocation->getOperand(0));

deallocation->replaceAllUsesWith(new_dealloc_call);
deallocation->eraseFromParent();
}
}


void CatoPass::insert_test_func(Module &M, RuntimeHandler &runtime)
{
IRBuilder<> builder(M.getContext());
LLVMContext &Ctx = M.getContext();
auto *entry_block = runtime.get_entry_block();

builder.SetInsertPoint(entry_block->getTerminator());

std::vector<Value *> args;
args.push_back(builder.getInt32(3));
args.push_back(builder.getInt64(0));
args.push_back(builder.getInt64(100));
args.push_back(builder.getInt64(42));

builder.CreateCall(runtime.functions.test_func, args);
}


bool CatoPass::runOnModule(Module &M)
{

Debug(errs() << "*----------------------------------*\n";);
Debug(errs() << "|      IR CODE BEFORE THE PASS:    |\n";);
Debug(errs() << "*----------------------------------*\n";);
Debug(M.dump(););

Debug(errs() << "*----------------------------------*\n";);
Debug(errs() << "|            DEBUG OUTPUT:         |\n";);
Debug(errs() << "*----------------------------------*\n";);

RuntimeHandler runtime(M);

if (cato_logging)
{
runtime.insert_cato_init_and_fin(nullptr, true);
}
else
{
runtime.insert_cato_init_and_fin();
}

runtime.replace_omp_functions();

std::vector<std::unique_ptr<Microtask>> microtasks = find_microtasks(M);

replace_fork_calls(M, runtime, microtasks);

replace_memory_allocations(M, runtime);

replace_sequential_shared_memory_accesses(M, runtime);

replace_microtask_shared_memory_accesses(M, runtime, microtasks);

replace_parallel_for(M, runtime, microtasks);

replace_reductions(M, runtime, microtasks);

replace_criticals(M, runtime, microtasks);

replace_memory_deallocations(M, runtime);

Debug(errs() << "*----------------------------------*\n";);
Debug(errs() << "|      IR CODE AFTER THE PASS:     |\n";);
Debug(errs() << "*----------------------------------*\n";);
Debug(M.dump(););

return true;
}


PassPluginLibraryInfo getCatoPluginInfo()
{
return {LLVM_PLUGIN_API_VERSION, "Cato", LLVM_VERSION_STRING, [](PassBuilder &PB) {
PB.registerPipelineParsingCallback([](StringRef Name, ModulePassManager &MPM,
ArrayRef<PassBuilder::PipelineElement>) {
if (Name == "Cato")
{
MPM.addPass(CatoPass());
return true;
}
return false;
});
}};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
return getCatoPluginInfo();
}