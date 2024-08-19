










#ifndef LLVM_ABI_BREAKING_CHECKS_H
#define LLVM_ABI_BREAKING_CHECKS_H


#define LLVM_ENABLE_ABI_BREAKING_CHECKS 0


#define LLVM_ENABLE_REVERSE_ITERATION 0


#if !LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING

#if defined(_MSC_VER)
#define LLVM_XSTR(s) LLVM_STR(s)
#define LLVM_STR(s) #s
#pragma detect_mismatch("LLVM_ENABLE_ABI_BREAKING_CHECKS", LLVM_XSTR(LLVM_ENABLE_ABI_BREAKING_CHECKS))
#undef LLVM_XSTR
#undef LLVM_STR
#elif defined(_WIN32) || defined(__CYGWIN__) 
#elif defined(__cplusplus)
namespace llvm {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
extern int EnableABIBreakingChecks;
__attribute__((weak, visibility ("hidden"))) int *VerifyEnableABIBreakingChecks = &EnableABIBreakingChecks;
#else
extern int DisableABIBreakingChecks;
__attribute__((weak, visibility ("hidden"))) int *VerifyDisableABIBreakingChecks = &DisableABIBreakingChecks;
#endif
}
#endif 

#endif 

#endif
