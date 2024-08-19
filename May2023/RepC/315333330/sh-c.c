#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-family/c-common.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "attribs.h"
static void
sh_add_function_attribute (const char *attr)
{
tree id = get_identifier (attr);
if (current_function_decl)
decl_attributes (&current_function_decl,
tree_cons (id, NULL_TREE, NULL_TREE), 0);
else
{
*sh_deferred_function_attributes_tail
= tree_cons (id, NULL_TREE, *sh_deferred_function_attributes_tail);
sh_deferred_function_attributes_tail
= &TREE_CHAIN (*sh_deferred_function_attributes_tail);
}
}
void
sh_pr_interrupt (struct cpp_reader *pfile ATTRIBUTE_UNUSED)
{
sh_add_function_attribute ("interrupt_handler");
}
void
sh_pr_trapa (struct cpp_reader *pfile ATTRIBUTE_UNUSED)
{
sh_add_function_attribute ("trapa_handler");
}
void
sh_pr_nosave_low_regs (struct cpp_reader *pfile ATTRIBUTE_UNUSED)
{
sh_add_function_attribute ("nosave_low_regs");
}
#define builtin_define(TXT) cpp_define (pfile, TXT)
#define builtin_assert(TXT) cpp_assert (pfile, TXT)
void
sh_cpu_cpp_builtins (cpp_reader* pfile)
{
builtin_define ("__sh__");
builtin_assert ("cpu=sh");
builtin_assert ("machine=sh");
switch ((int) sh_cpu)
{
case PROCESSOR_SH1:
builtin_define ("__sh1__");
builtin_define ("__SH1__");
break;
case PROCESSOR_SH2:
builtin_define ("__sh2__");
builtin_define ("__SH2__");
break;
case PROCESSOR_SH2E:
builtin_define ("__SH2E__");
break;
case PROCESSOR_SH2A:
builtin_define ("__SH2A__");
if (TARGET_SH2A_DOUBLE)
builtin_define (TARGET_FPU_SINGLE
? "__SH2A_SINGLE__" : "__SH2A_DOUBLE__");
else
builtin_define (TARGET_FPU_ANY
? "__SH2A_SINGLE_ONLY__" : "__SH2A_NOFPU__");
break;
case PROCESSOR_SH3:
builtin_define ("__sh3__");
builtin_define ("__SH3__");
if (TARGET_HARD_SH4)
builtin_define ("__SH4_NOFPU__");
break;
case PROCESSOR_SH3E:
builtin_define (TARGET_HARD_SH4 ? "__SH4_SINGLE_ONLY__" : "__SH3E__");
break;
case PROCESSOR_SH4:
builtin_define (TARGET_FPU_SINGLE ? "__SH4_SINGLE__" : "__SH4__");
break;
case PROCESSOR_SH4A: \
builtin_define ("__SH4A__");
builtin_define (TARGET_SH4
? (TARGET_FPU_SINGLE ? "__SH4_SINGLE__" : "__SH4__")
: TARGET_FPU_ANY ? "__SH4_SINGLE_ONLY__"
: "__SH4_NOFPU__");
break;
}
if (TARGET_FPU_ANY)
builtin_define ("__SH_FPU_ANY__");
if (TARGET_FPU_DOUBLE)
builtin_define ("__SH_FPU_DOUBLE__");
if (TARGET_HITACHI)
builtin_define ("__HITACHI__");
if (TARGET_FMOVD)
builtin_define ("__FMOVD_ENABLED__");
if (TARGET_FDPIC)
{
builtin_define ("__SH_FDPIC__");
builtin_define ("__FDPIC__");
}
builtin_define (TARGET_LITTLE_ENDIAN
? "__LITTLE_ENDIAN__" : "__BIG_ENDIAN__");
cpp_define_formatted (pfile, "__SH_ATOMIC_MODEL_%s__",
selected_atomic_model ().cdef_name);
}
