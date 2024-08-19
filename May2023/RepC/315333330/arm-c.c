#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-family/c-common.h"
#include "memmodel.h"
#include "tm_p.h"
#include "c-family/c-pragma.h"
#include "stringpool.h"
static void
arm_output_c_attributes (void)
{
int wchar_size = (int)(TYPE_PRECISION (wchar_type_node) / BITS_PER_UNIT);
arm_emit_eabi_attribute ("Tag_ABI_PCS_wchar_t", 18, wchar_size);
}
void
arm_lang_object_attributes_init (void)
{
arm_lang_output_object_attributes_hook = arm_output_c_attributes;
}
#define builtin_define(TXT) cpp_define (pfile, TXT)
#define builtin_assert(TXT) cpp_assert (pfile, TXT)
static void
def_or_undef_macro(struct cpp_reader* pfile, const char *name, bool def_p)
{
if (def_p)
cpp_define (pfile, name);
else
cpp_undef (pfile, name);
}
static void
arm_cpu_builtins (struct cpp_reader* pfile)
{
def_or_undef_macro (pfile, "__ARM_FEATURE_DSP", TARGET_DSP_MULTIPLY);
def_or_undef_macro (pfile, "__ARM_FEATURE_QBIT", TARGET_ARM_QBIT);
def_or_undef_macro (pfile, "__ARM_FEATURE_SAT", TARGET_ARM_SAT);
def_or_undef_macro (pfile, "__ARM_FEATURE_CRYPTO", TARGET_CRYPTO);
def_or_undef_macro (pfile, "__ARM_FEATURE_UNALIGNED", unaligned_access);
def_or_undef_macro (pfile, "__ARM_FEATURE_QRDMX", TARGET_NEON_RDMA);
def_or_undef_macro (pfile, "__ARM_FEATURE_CRC32", TARGET_CRC32);
def_or_undef_macro (pfile, "__ARM_FEATURE_DOTPROD", TARGET_DOTPROD);
def_or_undef_macro (pfile, "__ARM_32BIT_STATE", TARGET_32BIT);
cpp_undef (pfile, "__ARM_FEATURE_CMSE");
if (arm_arch8 && !arm_arch_notm)
{
if (arm_arch_cmse && use_cmse)
builtin_define_with_int_value ("__ARM_FEATURE_CMSE", 3);
else
builtin_define ("__ARM_FEATURE_CMSE");
}
cpp_undef (pfile, "__ARM_FEATURE_LDREX");
if (TARGET_ARM_FEATURE_LDREX)
builtin_define_with_int_value ("__ARM_FEATURE_LDREX",
TARGET_ARM_FEATURE_LDREX);
def_or_undef_macro (pfile, "__ARM_FEATURE_CLZ",
((TARGET_ARM_ARCH >= 5 && !TARGET_THUMB)
|| TARGET_ARM_ARCH_ISA_THUMB >=2));
def_or_undef_macro (pfile, "__ARM_FEATURE_NUMERIC_MAXMIN",
TARGET_ARM_ARCH >= 8 && TARGET_NEON && TARGET_VFP5);
def_or_undef_macro (pfile, "__ARM_FEATURE_SIMD32", TARGET_INT_SIMD);
builtin_define_with_int_value ("__ARM_SIZEOF_MINIMAL_ENUM",
flag_short_enums ? 1 : 4);
builtin_define_type_sizeof ("__ARM_SIZEOF_WCHAR_T", wchar_type_node);
cpp_undef (pfile, "__ARM_ARCH_PROFILE");
if (TARGET_ARM_ARCH_PROFILE)
builtin_define_with_int_value ("__ARM_ARCH_PROFILE",
TARGET_ARM_ARCH_PROFILE);
builtin_define ("__arm__");
if (TARGET_ARM_ARCH)
{
cpp_undef (pfile, "__ARM_ARCH");
builtin_define_with_int_value ("__ARM_ARCH", TARGET_ARM_ARCH);
}
if (arm_arch_notm)
builtin_define ("__ARM_ARCH_ISA_ARM");
builtin_define ("__APCS_32__");
def_or_undef_macro (pfile, "__thumb__", TARGET_THUMB);
def_or_undef_macro (pfile, "__thumb2__", TARGET_THUMB2);
if (TARGET_BIG_END)
def_or_undef_macro (pfile, "__THUMBEB__", TARGET_THUMB);
else
def_or_undef_macro (pfile, "__THUMBEL__", TARGET_THUMB);
cpp_undef (pfile, "__ARM_ARCH_ISA_THUMB");
if (TARGET_ARM_ARCH_ISA_THUMB)
builtin_define_with_int_value ("__ARM_ARCH_ISA_THUMB",
TARGET_ARM_ARCH_ISA_THUMB);
if (TARGET_BIG_END)
{
builtin_define ("__ARMEB__");
builtin_define ("__ARM_BIG_ENDIAN");
}
else
{
builtin_define ("__ARMEL__");
}
if (TARGET_SOFT_FLOAT)
builtin_define ("__SOFTFP__");
builtin_define ("__VFP_FP__");
cpp_undef (pfile, "__ARM_FP");
if (TARGET_ARM_FP)
builtin_define_with_int_value ("__ARM_FP", TARGET_ARM_FP);
def_or_undef_macro (pfile, "__ARM_FP16_FORMAT_IEEE",
arm_fp16_format == ARM_FP16_FORMAT_IEEE);
def_or_undef_macro (pfile, "__ARM_FP16_FORMAT_ALTERNATIVE",
arm_fp16_format == ARM_FP16_FORMAT_ALTERNATIVE);
def_or_undef_macro (pfile, "__ARM_FP16_ARGS",
arm_fp16_format != ARM_FP16_FORMAT_NONE);
def_or_undef_macro (pfile, "__ARM_FEATURE_FP16_SCALAR_ARITHMETIC",
TARGET_VFP_FP16INST);
def_or_undef_macro (pfile, "__ARM_FEATURE_FP16_VECTOR_ARITHMETIC",
TARGET_NEON_FP16INST);
def_or_undef_macro (pfile, "__ARM_FEATURE_FP16_FML", TARGET_FP16FML);
def_or_undef_macro (pfile, "__ARM_FEATURE_FMA", TARGET_FMA);
def_or_undef_macro (pfile, "__ARM_NEON__", TARGET_NEON);
def_or_undef_macro (pfile, "__ARM_NEON", TARGET_NEON);
cpp_undef (pfile, "__ARM_NEON_FP");
if (TARGET_NEON_FP)
builtin_define_with_int_value ("__ARM_NEON_FP", TARGET_NEON_FP);
if (arm_cpp_interwork)
builtin_define ("__THUMB_INTERWORK__");
builtin_define (arm_arch_name);
if (arm_arch_xscale)
builtin_define ("__XSCALE__");
if (arm_arch_iwmmxt)
{
builtin_define ("__IWMMXT__");
builtin_define ("__ARM_WMMX");
}
if (arm_arch_iwmmxt2)
builtin_define ("__IWMMXT2__");
if (arm_arch6kz)
builtin_define ("__ARM_ARCH_6ZK__");
if (TARGET_AAPCS_BASED)
{
if (arm_pcs_default == ARM_PCS_AAPCS_VFP)
builtin_define ("__ARM_PCS_VFP");
else if (arm_pcs_default == ARM_PCS_AAPCS)
builtin_define ("__ARM_PCS");
builtin_define ("__ARM_EABI__");
}
def_or_undef_macro (pfile, "__ARM_ARCH_EXT_IDIV__", TARGET_IDIV);
def_or_undef_macro (pfile, "__ARM_FEATURE_IDIV", TARGET_IDIV);
def_or_undef_macro (pfile, "__ARM_ASM_SYNTAX_UNIFIED__", inline_asm_unified);
cpp_undef (pfile, "__ARM_FEATURE_COPROC");
if (TARGET_32BIT && arm_arch4 && !(arm_arch8 && arm_arch_notm))
{
int coproc_level = 0x1;
if (arm_arch5)
coproc_level |= 0x2;
if (arm_arch5e)
coproc_level |= 0x4;
if (arm_arch6)
coproc_level |= 0x8;
builtin_define_with_int_value ("__ARM_FEATURE_COPROC", coproc_level);
}
}
void
arm_cpu_cpp_builtins (struct cpp_reader * pfile)
{
builtin_assert ("cpu=arm");
builtin_assert ("machine=arm");
arm_cpu_builtins (pfile);
}
static bool
arm_pragma_target_parse (tree args, tree pop_target)
{
tree prev_tree = target_option_current_node;
tree cur_tree;
struct cl_target_option *prev_opt;
struct cl_target_option *cur_opt;
if (! args)
{
cur_tree = ((pop_target) ? pop_target : target_option_default_node);
cl_target_option_restore (&global_options,
TREE_TARGET_OPTION (cur_tree));
}
else
{
cur_tree = arm_valid_target_attribute_tree (args, &global_options,
&global_options_set);
if (cur_tree == NULL_TREE)
{
cl_target_option_restore (&global_options,
TREE_TARGET_OPTION (prev_tree));
return false;
}
target_option_current_node = cur_tree;
arm_configure_build_target (&arm_active_target,
TREE_TARGET_OPTION (cur_tree),
&global_options_set, false);
}
prev_opt = TREE_TARGET_OPTION (prev_tree);
cur_opt  = TREE_TARGET_OPTION (cur_tree);
gcc_assert (prev_opt);
gcc_assert (cur_opt);
if (cur_opt != prev_opt)
{
cpp_options *cpp_opts = cpp_get_options (parse_in);
unsigned char saved_warn_unused_macros = cpp_opts->warn_unused_macros;
cpp_opts->warn_unused_macros = 0;
gcc_assert (cur_opt->x_target_flags == target_flags);
tree acond_macro = get_identifier ("__ARM_NEON_FP");
C_CPP_HASHNODE (acond_macro)->flags |= NODE_CONDITIONAL ;
acond_macro = get_identifier ("__ARM_FP");
C_CPP_HASHNODE (acond_macro)->flags |= NODE_CONDITIONAL;
acond_macro = get_identifier ("__ARM_FEATURE_LDREX");
C_CPP_HASHNODE (acond_macro)->flags |= NODE_CONDITIONAL;
arm_cpu_builtins (parse_in);
cpp_opts->warn_unused_macros = saved_warn_unused_macros;
arm_reset_previous_fndecl ();
if (cur_tree == target_option_default_node)
save_restore_target_globals (cur_tree);
}
return true;
}
void
arm_register_target_pragmas (void)
{
targetm.target_option.pragma_parse = arm_pragma_target_parse;
#ifdef REGISTER_SUBTARGET_PRAGMAS
REGISTER_SUBTARGET_PRAGMAS ();
#endif
}
