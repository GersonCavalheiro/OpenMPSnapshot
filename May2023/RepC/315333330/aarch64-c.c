#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "input.h"
#include "memmodel.h"
#include "tm_p.h"
#include "flags.h"
#include "c-family/c-common.h"
#include "cpplib.h"
#include "c-family/c-pragma.h"
#include "langhooks.h"
#include "target.h"
#define builtin_define(TXT) cpp_define (pfile, TXT)
#define builtin_assert(TXT) cpp_assert (pfile, TXT)
static void
aarch64_def_or_undef (bool def_p, const char *macro, cpp_reader *pfile)
{
if (def_p)
cpp_define (pfile, macro);
else
cpp_undef (pfile, macro);
}
static void
aarch64_define_unconditional_macros (cpp_reader *pfile)
{
builtin_define ("__aarch64__");
builtin_define ("__ARM_64BIT_STATE");
builtin_define ("__ARM_ARCH_ISA_A64");
builtin_define_with_int_value ("__ARM_ALIGN_MAX_PWR", 28);
builtin_define_with_int_value ("__ARM_ALIGN_MAX_STACK_PWR", 16);
builtin_define ("__ARM_ARCH_8A");
builtin_define_with_int_value ("__ARM_ARCH_PROFILE", 'A');
builtin_define ("__ARM_FEATURE_CLZ");
builtin_define ("__ARM_FEATURE_IDIV");
builtin_define ("__ARM_FEATURE_UNALIGNED");
builtin_define ("__ARM_PCS_AAPCS64");
builtin_define_with_int_value ("__ARM_SIZEOF_WCHAR_T", WCHAR_TYPE_SIZE / 8);
}
static void
aarch64_update_cpp_builtins (cpp_reader *pfile)
{
aarch64_def_or_undef (flag_unsafe_math_optimizations, "__ARM_FP_FAST", pfile);
builtin_define_with_int_value ("__ARM_ARCH", aarch64_architecture_version);
builtin_define_with_int_value ("__ARM_SIZEOF_MINIMAL_ENUM",
flag_short_enums ? 1 : 4);
aarch64_def_or_undef (TARGET_BIG_END, "__AARCH64EB__", pfile);
aarch64_def_or_undef (TARGET_BIG_END, "__ARM_BIG_ENDIAN", pfile);
aarch64_def_or_undef (!TARGET_BIG_END, "__AARCH64EL__", pfile);
aarch64_def_or_undef (TARGET_FLOAT, "__ARM_FEATURE_FMA", pfile);
if (TARGET_FLOAT || TARGET_SIMD)
{
builtin_define_with_int_value ("__ARM_FP", 0x0E);
builtin_define ("__ARM_FP16_FORMAT_IEEE");
builtin_define ("__ARM_FP16_ARGS");
}
else
cpp_undef (pfile, "__ARM_FP");
aarch64_def_or_undef (TARGET_FP_F16INST,
"__ARM_FEATURE_FP16_SCALAR_ARITHMETIC", pfile);
aarch64_def_or_undef (TARGET_SIMD_F16INST,
"__ARM_FEATURE_FP16_VECTOR_ARITHMETIC", pfile);
aarch64_def_or_undef (TARGET_SIMD, "__ARM_FEATURE_NUMERIC_MAXMIN", pfile);
aarch64_def_or_undef (TARGET_SIMD, "__ARM_NEON", pfile);
aarch64_def_or_undef (TARGET_CRC32, "__ARM_FEATURE_CRC32", pfile);
aarch64_def_or_undef (TARGET_DOTPROD, "__ARM_FEATURE_DOTPROD", pfile);
cpp_undef (pfile, "__AARCH64_CMODEL_TINY__");
cpp_undef (pfile, "__AARCH64_CMODEL_SMALL__");
cpp_undef (pfile, "__AARCH64_CMODEL_LARGE__");
switch (aarch64_cmodel)
{
case AARCH64_CMODEL_TINY:
case AARCH64_CMODEL_TINY_PIC:
builtin_define ("__AARCH64_CMODEL_TINY__");
break;
case AARCH64_CMODEL_SMALL:
case AARCH64_CMODEL_SMALL_PIC:
builtin_define ("__AARCH64_CMODEL_SMALL__");
break;
case AARCH64_CMODEL_LARGE:
builtin_define ("__AARCH64_CMODEL_LARGE__");
break;
default:
break;
}
aarch64_def_or_undef (TARGET_ILP32, "_ILP32", pfile);
aarch64_def_or_undef (TARGET_ILP32, "__ILP32__", pfile);
aarch64_def_or_undef (TARGET_CRYPTO, "__ARM_FEATURE_CRYPTO", pfile);
aarch64_def_or_undef (TARGET_SIMD_RDMA, "__ARM_FEATURE_QRDMX", pfile);
aarch64_def_or_undef (TARGET_SVE, "__ARM_FEATURE_SVE", pfile);
cpp_undef (pfile, "__ARM_FEATURE_SVE_BITS");
if (TARGET_SVE)
{
int bits;
if (!BITS_PER_SVE_VECTOR.is_constant (&bits))
bits = 0;
builtin_define_with_int_value ("__ARM_FEATURE_SVE_BITS", bits);
}
aarch64_def_or_undef (TARGET_AES, "__ARM_FEATURE_AES", pfile);
aarch64_def_or_undef (TARGET_SHA2, "__ARM_FEATURE_SHA2", pfile);
aarch64_def_or_undef (TARGET_SHA3, "__ARM_FEATURE_SHA3", pfile);
aarch64_def_or_undef (TARGET_SHA3, "__ARM_FEATURE_SHA512", pfile);
aarch64_def_or_undef (TARGET_SM4, "__ARM_FEATURE_SM3", pfile);
aarch64_def_or_undef (TARGET_SM4, "__ARM_FEATURE_SM4", pfile);
aarch64_def_or_undef (TARGET_F16FML, "__ARM_FEATURE_FP16_FML", pfile);
cpp_undef (pfile, "__FLT_EVAL_METHOD__");
builtin_define_with_int_value ("__FLT_EVAL_METHOD__",
c_flt_eval_method (true));
cpp_undef (pfile, "__FLT_EVAL_METHOD_C99__");
builtin_define_with_int_value ("__FLT_EVAL_METHOD_C99__",
c_flt_eval_method (false));
}
void
aarch64_cpu_cpp_builtins (cpp_reader *pfile)
{
aarch64_define_unconditional_macros (pfile);
aarch64_update_cpp_builtins (pfile);
}
static bool
aarch64_pragma_target_parse (tree args, tree pop_target)
{
if (args)
{
if (!aarch64_process_target_attr (args))
return false;
aarch64_override_options_internal (&global_options);
}
else
{
pop_target = pop_target ? pop_target : target_option_default_node;
cl_target_option_restore (&global_options,
TREE_TARGET_OPTION (pop_target));
}
target_option_current_node
= build_target_option_node (&global_options);
aarch64_reset_previous_fndecl ();
cpp_options *cpp_opts = cpp_get_options (parse_in);
unsigned char saved_warn_unused_macros = cpp_opts->warn_unused_macros;
cpp_opts->warn_unused_macros = 0;
aarch64_update_cpp_builtins (parse_in);
cpp_opts->warn_unused_macros = saved_warn_unused_macros;
if (pop_target)
aarch64_save_restore_target_globals (pop_target);
if (TARGET_SIMD)
{
tree saved_current_target_pragma = current_target_pragma;
current_target_pragma = NULL;
aarch64_init_simd_builtins ();
current_target_pragma = saved_current_target_pragma;
}
return true;
}
void
aarch64_register_pragmas (void)
{
targetm.target_option.pragma_parse = aarch64_pragma_target_parse;
}
