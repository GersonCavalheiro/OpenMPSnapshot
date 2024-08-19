#ifndef CATO_MICROTASK_H
#define CATO_MICROTASK_H

#include <memory>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include "helper.h"

#include "SharedVariable.h"


struct ParallelForData
{
llvm::CallInst *init;
llvm::CallInst *fini;
};


struct ReductionData
{
llvm::CallInst *reduce;
llvm::CallInst *end_reduce;
};


struct CriticalData
{
llvm::CallInst *critical;
llvm::CallInst *end_critical;
};


class Microtask
{
private:
llvm::CallInst *_fork_call;

llvm::Function *_function;

std::vector<llvm::Value *> _shared_variables;

std::vector<ParallelForData> _parallel_for;

std::vector<ReductionData> _reduction;

std::vector<CriticalData> _critical;

public:

Microtask(llvm::CallInst *fork_call);

~Microtask();

llvm::CallInst *get_fork_call();

llvm::Function *get_function();

std::vector<ParallelForData> *get_parallel_for();

std::vector<ReductionData> *get_reductions();

std::vector<CriticalData> *get_critical();

bool has_shared_variables();

std::vector<llvm::Value *> &get_shared_variables();
};

#endif
