#define IN_TARGET_CODE 1
#include "config.h"
#define INCLUDE_STRING
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "gimple.h"
#include "cfghooks.h"
#include "cfgloop.h"
#include "df.h"
#include "tm_p.h"
#include "stringpool.h"
#include "attribs.h"
#include "optabs.h"
#include "regs.h"
#include "emit-rtl.h"
#include "recog.h"
#include "diagnostic.h"
#include "insn-attr.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "calls.h"
#include "varasm.h"
#include "output.h"
#include "flags.h"
#include "explow.h"
#include "expr.h"
#include "reload.h"
#include "langhooks.h"
#include "opts.h"
#include "params.h"
#include "gimplify.h"
#include "dwarf2.h"
#include "gimple-iterator.h"
#include "tree-vectorizer.h"
#include "aarch64-cost-tables.h"
#include "dumpfile.h"
#include "builtins.h"
#include "rtl-iter.h"
#include "tm-constrs.h"
#include "sched-int.h"
#include "target-globals.h"
#include "common/common-target.h"
#include "cfgrtl.h"
#include "selftest.h"
#include "selftest-rtl.h"
#include "rtx-vector-builder.h"
#include "target-def.h"
#define POINTER_BYTES (POINTER_SIZE / BITS_PER_UNIT)
enum aarch64_address_type {
ADDRESS_REG_IMM,
ADDRESS_REG_WB,
ADDRESS_REG_REG,
ADDRESS_REG_UXTW,
ADDRESS_REG_SXTW,
ADDRESS_LO_SUM,
ADDRESS_SYMBOLIC
};
struct aarch64_address_info {
enum aarch64_address_type type;
rtx base;
rtx offset;
poly_int64 const_offset;
int shift;
enum aarch64_symbol_type symbol_type;
};
struct simd_immediate_info
{
enum insn_type { MOV, MVN };
enum modifier_type { LSL, MSL };
simd_immediate_info () {}
simd_immediate_info (scalar_float_mode, rtx);
simd_immediate_info (scalar_int_mode, unsigned HOST_WIDE_INT,
insn_type = MOV, modifier_type = LSL,
unsigned int = 0);
simd_immediate_info (scalar_mode, rtx, rtx);
scalar_mode elt_mode;
rtx value;
rtx step;
insn_type insn;
modifier_type modifier;
unsigned int shift;
};
inline simd_immediate_info
::simd_immediate_info (scalar_float_mode elt_mode_in, rtx value_in)
: elt_mode (elt_mode_in), value (value_in), step (NULL_RTX), insn (MOV),
modifier (LSL), shift (0)
{}
inline simd_immediate_info
::simd_immediate_info (scalar_int_mode elt_mode_in,
unsigned HOST_WIDE_INT value_in,
insn_type insn_in, modifier_type modifier_in,
unsigned int shift_in)
: elt_mode (elt_mode_in), value (gen_int_mode (value_in, elt_mode_in)),
step (NULL_RTX), insn (insn_in), modifier (modifier_in), shift (shift_in)
{}
inline simd_immediate_info
::simd_immediate_info (scalar_mode elt_mode_in, rtx value_in, rtx step_in)
: elt_mode (elt_mode_in), value (value_in), step (step_in), insn (MOV),
modifier (LSL), shift (0)
{}
enum aarch64_code_model aarch64_cmodel;
poly_uint16 aarch64_sve_vg;
#ifdef HAVE_AS_TLS
#undef TARGET_HAVE_TLS
#define TARGET_HAVE_TLS 1
#endif
static bool aarch64_composite_type_p (const_tree, machine_mode);
static bool aarch64_vfp_is_call_or_return_candidate (machine_mode,
const_tree,
machine_mode *, int *,
bool *);
static void aarch64_elf_asm_constructor (rtx, int) ATTRIBUTE_UNUSED;
static void aarch64_elf_asm_destructor (rtx, int) ATTRIBUTE_UNUSED;
static void aarch64_override_options_after_change (void);
static bool aarch64_vector_mode_supported_p (machine_mode);
static int aarch64_address_cost (rtx, machine_mode, addr_space_t, bool);
static bool aarch64_builtin_support_vector_misalignment (machine_mode mode,
const_tree type,
int misalignment,
bool is_packed);
static machine_mode aarch64_simd_container_mode (scalar_mode, poly_int64);
static bool aarch64_print_ldpstp_address (FILE *, machine_mode, rtx);
unsigned aarch64_architecture_version;
enum aarch64_processor aarch64_tune = cortexa53;
unsigned long aarch64_tune_flags = 0;
bool aarch64_pcrelative_literal_loads;
struct aarch64_flag_desc
{
const char* name;
unsigned int flag;
};
#define AARCH64_FUSION_PAIR(name, internal_name) \
{ name, AARCH64_FUSE_##internal_name },
static const struct aarch64_flag_desc aarch64_fusible_pairs[] =
{
{ "none", AARCH64_FUSE_NOTHING },
#include "aarch64-fusion-pairs.def"
{ "all", AARCH64_FUSE_ALL },
{ NULL, AARCH64_FUSE_NOTHING }
};
#define AARCH64_EXTRA_TUNING_OPTION(name, internal_name) \
{ name, AARCH64_EXTRA_TUNE_##internal_name },
static const struct aarch64_flag_desc aarch64_tuning_flags[] =
{
{ "none", AARCH64_EXTRA_TUNE_NONE },
#include "aarch64-tuning-flags.def"
{ "all", AARCH64_EXTRA_TUNE_ALL },
{ NULL, AARCH64_EXTRA_TUNE_NONE }
};
static const struct cpu_addrcost_table generic_addrcost_table =
{
{
1, 
0, 
0, 
1, 
},
0, 
0, 
0, 
0, 
0, 
0 
};
static const struct cpu_addrcost_table exynosm1_addrcost_table =
{
{
0, 
0, 
0, 
2, 
},
0, 
0, 
1, 
1, 
2, 
0, 
};
static const struct cpu_addrcost_table xgene1_addrcost_table =
{
{
1, 
0, 
0, 
1, 
},
1, 
0, 
0, 
1, 
1, 
0, 
};
static const struct cpu_addrcost_table thunderx2t99_addrcost_table =
{
{
1, 
1, 
1, 
2, 
},
0, 
0, 
2, 
3, 
3, 
0, 
};
static const struct cpu_addrcost_table qdf24xx_addrcost_table =
{
{
1, 
1, 
1, 
2, 
},
1, 
1, 
3, 
3, 
3, 
2, 
};
static const struct cpu_regmove_cost generic_regmove_cost =
{
1, 
5, 
5, 
2 
};
static const struct cpu_regmove_cost cortexa57_regmove_cost =
{
1, 
5, 
5, 
2 
};
static const struct cpu_regmove_cost cortexa53_regmove_cost =
{
1, 
5, 
5, 
2 
};
static const struct cpu_regmove_cost exynosm1_regmove_cost =
{
1, 
9, 
9, 
1 
};
static const struct cpu_regmove_cost thunderx_regmove_cost =
{
2, 
2, 
6, 
4 
};
static const struct cpu_regmove_cost xgene1_regmove_cost =
{
1, 
8, 
8, 
2 
};
static const struct cpu_regmove_cost qdf24xx_regmove_cost =
{
2, 
6, 
6, 
4 
};
static const struct cpu_regmove_cost thunderx2t99_regmove_cost =
{
1, 
8, 
8, 
4  
};
static const struct cpu_vector_cost generic_vector_cost =
{
1, 
1, 
1, 
1, 
1, 
1, 
2, 
1, 
1, 
1, 
1, 
1, 
1, 
3, 
1 
};
static const struct cpu_vector_cost qdf24xx_vector_cost =
{
1, 
1, 
1, 
1, 
1, 
3, 
2, 
1, 
1, 
1, 
1, 
1, 
1, 
3, 
1 
};
static const struct cpu_vector_cost thunderx_vector_cost =
{
1, 
1, 
3, 
1, 
4, 
1, 
4, 
2, 
2, 
3, 
5, 
5, 
1, 
3, 
3 
};
static const struct cpu_vector_cost cortexa57_vector_cost =
{
1, 
1, 
4, 
1, 
2, 
2, 
3, 
8, 
8, 
4, 
4, 
1, 
1, 
1, 
1 
};
static const struct cpu_vector_cost exynosm1_vector_cost =
{
1, 
1, 
5, 
1, 
3, 
3, 
3, 
3, 
3, 
5, 
5, 
1, 
1, 
1, 
1 
};
static const struct cpu_vector_cost xgene1_vector_cost =
{
1, 
1, 
5, 
1, 
2, 
2, 
2, 
4, 
4, 
10, 
10, 
2, 
2, 
2, 
1 
};
static const struct cpu_vector_cost thunderx2t99_vector_cost =
{
1, 
6, 
4, 
1, 
5, 
6, 
3, 
6, 
5, 
8, 
8, 
4, 
4, 
2, 
1  
};
static const struct cpu_branch_cost generic_branch_cost =
{
1,  
3   
};
static const cpu_approx_modes generic_approx_modes =
{
AARCH64_APPROX_NONE,	
AARCH64_APPROX_NONE,	
AARCH64_APPROX_NONE	
};
static const cpu_approx_modes exynosm1_approx_modes =
{
AARCH64_APPROX_NONE,	
AARCH64_APPROX_ALL,	
AARCH64_APPROX_ALL	
};
static const cpu_approx_modes xgene1_approx_modes =
{
AARCH64_APPROX_NONE,	
AARCH64_APPROX_NONE,	
AARCH64_APPROX_ALL	
};
static const cpu_prefetch_tune generic_prefetch_tune =
{
0,			
-1,			
-1,			
-1,			
-1			
};
static const cpu_prefetch_tune exynosm1_prefetch_tune =
{
0,			
-1,			
64,			
-1,			
-1			
};
static const cpu_prefetch_tune qdf24xx_prefetch_tune =
{
4,			
32,			
64,			
1024,			
-1			
};
static const cpu_prefetch_tune thunderxt88_prefetch_tune =
{
8,			
32,			
128,			
16*1024,		
3			
};
static const cpu_prefetch_tune thunderx_prefetch_tune =
{
8,			
32,			
128,			
-1,			
-1			
};
static const cpu_prefetch_tune thunderx2t99_prefetch_tune =
{
8,			
32,			
64,			
256,			
-1			
};
static const struct tune_params generic_tunings =
{
&cortexa57_extra_costs,
&generic_addrcost_table,
&generic_regmove_cost,
&generic_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
2, 
(AARCH64_FUSE_AES_AESMC), 
8,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params cortexa35_tunings =
{
&cortexa53_extra_costs,
&generic_addrcost_table,
&cortexa53_regmove_cost,
&generic_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
1, 
(AARCH64_FUSE_AES_AESMC | AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK | AARCH64_FUSE_ADRP_LDR), 
16,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params cortexa53_tunings =
{
&cortexa53_extra_costs,
&generic_addrcost_table,
&cortexa53_regmove_cost,
&generic_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
2, 
(AARCH64_FUSE_AES_AESMC | AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK | AARCH64_FUSE_ADRP_LDR), 
16,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params cortexa57_tunings =
{
&cortexa57_extra_costs,
&generic_addrcost_table,
&cortexa57_regmove_cost,
&cortexa57_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
3, 
(AARCH64_FUSE_AES_AESMC | AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK), 
16,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_RENAME_FMA_REGS),	
&generic_prefetch_tune
};
static const struct tune_params cortexa72_tunings =
{
&cortexa57_extra_costs,
&generic_addrcost_table,
&cortexa57_regmove_cost,
&cortexa57_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
3, 
(AARCH64_FUSE_AES_AESMC | AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK), 
16,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params cortexa73_tunings =
{
&cortexa57_extra_costs,
&generic_addrcost_table,
&cortexa57_regmove_cost,
&cortexa57_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
2, 
(AARCH64_FUSE_AES_AESMC | AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK | AARCH64_FUSE_ADRP_LDR), 
16,	
4,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params exynosm1_tunings =
{
&exynosm1_extra_costs,
&exynosm1_addrcost_table,
&exynosm1_regmove_cost,
&exynosm1_vector_cost,
&generic_branch_cost,
&exynosm1_approx_modes,
4,	
3,	
(AARCH64_FUSE_AES_AESMC), 
4,	
4,	
4,	
2,	
4,	
1,	
2,	
2,	
48,	
tune_params::AUTOPREFETCHER_WEAK, 
(AARCH64_EXTRA_TUNE_NONE), 
&exynosm1_prefetch_tune
};
static const struct tune_params thunderxt88_tunings =
{
&thunderx_extra_costs,
&generic_addrcost_table,
&thunderx_regmove_cost,
&thunderx_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
6, 
2, 
AARCH64_FUSE_CMP_BRANCH, 
8,	
8,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_OFF,	
(AARCH64_EXTRA_TUNE_SLOW_UNALIGNED_LDPW),	
&thunderxt88_prefetch_tune
};
static const struct tune_params thunderx_tunings =
{
&thunderx_extra_costs,
&generic_addrcost_table,
&thunderx_regmove_cost,
&thunderx_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
6, 
2, 
AARCH64_FUSE_CMP_BRANCH, 
8,	
8,	
8,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_OFF,	
(AARCH64_EXTRA_TUNE_SLOW_UNALIGNED_LDPW
| AARCH64_EXTRA_TUNE_CHEAP_SHIFT_EXTEND),	
&thunderx_prefetch_tune
};
static const struct tune_params xgene1_tunings =
{
&xgene1_extra_costs,
&xgene1_addrcost_table,
&xgene1_regmove_cost,
&xgene1_vector_cost,
&generic_branch_cost,
&xgene1_approx_modes,
6, 
4, 
AARCH64_FUSE_NOTHING, 
16,	
8,	
16,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_OFF,	
(AARCH64_EXTRA_TUNE_NONE),	
&generic_prefetch_tune
};
static const struct tune_params qdf24xx_tunings =
{
&qdf24xx_extra_costs,
&qdf24xx_addrcost_table,
&qdf24xx_regmove_cost,
&qdf24xx_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
4, 
(AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK), 
16,	
8,	
16,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),		
&qdf24xx_prefetch_tune
};
static const struct tune_params saphira_tunings =
{
&generic_extra_costs,
&generic_addrcost_table,
&generic_regmove_cost,
&generic_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
4, 
(AARCH64_FUSE_MOV_MOVK | AARCH64_FUSE_ADRP_ADD
| AARCH64_FUSE_MOVK_MOVK), 
16,	
8,	
16,	
2,	
4,	
1,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),		
&generic_prefetch_tune
};
static const struct tune_params thunderx2t99_tunings =
{
&thunderx2t99_extra_costs,
&thunderx2t99_addrcost_table,
&thunderx2t99_regmove_cost,
&thunderx2t99_vector_cost,
&generic_branch_cost,
&generic_approx_modes,
4, 
4, 
(AARCH64_FUSE_CMP_BRANCH | AARCH64_FUSE_AES_AESMC
| AARCH64_FUSE_ALU_BRANCH), 
16,	
8,	
16,	
3,	
2,	
2,	
2,	
2,	
0,	
tune_params::AUTOPREFETCHER_WEAK,	
(AARCH64_EXTRA_TUNE_NONE),	
&thunderx2t99_prefetch_tune
};
struct aarch64_tuning_override_function
{
const char* name;
void (*parse_override)(const char*, struct tune_params*);
};
static void aarch64_parse_fuse_string (const char*, struct tune_params*);
static void aarch64_parse_tune_string (const char*, struct tune_params*);
static const struct aarch64_tuning_override_function
aarch64_tuning_override_functions[] =
{
{ "fuse", aarch64_parse_fuse_string },
{ "tune", aarch64_parse_tune_string },
{ NULL, NULL }
};
struct processor
{
const char *const name;
enum aarch64_processor ident;
enum aarch64_processor sched_core;
enum aarch64_arch arch;
unsigned architecture_version;
const unsigned long flags;
const struct tune_params *const tune;
};
static const struct processor all_architectures[] =
{
#define AARCH64_ARCH(NAME, CORE, ARCH_IDENT, ARCH_REV, FLAGS) \
{NAME, CORE, CORE, AARCH64_ARCH_##ARCH_IDENT, ARCH_REV, FLAGS, NULL},
#include "aarch64-arches.def"
{NULL, aarch64_none, aarch64_none, aarch64_no_arch, 0, 0, NULL}
};
static const struct processor all_cores[] =
{
#define AARCH64_CORE(NAME, IDENT, SCHED, ARCH, FLAGS, COSTS, IMP, PART, VARIANT) \
{NAME, IDENT, SCHED, AARCH64_ARCH_##ARCH,				\
all_architectures[AARCH64_ARCH_##ARCH].architecture_version,	\
FLAGS, &COSTS##_tunings},
#include "aarch64-cores.def"
{"generic", generic, cortexa53, AARCH64_ARCH_8A, 8,
AARCH64_FL_FOR_ARCH8, &generic_tunings},
{NULL, aarch64_none, aarch64_none, aarch64_no_arch, 0, 0, NULL}
};
static const struct processor *selected_arch;
static const struct processor *selected_cpu;
static const struct processor *selected_tune;
struct tune_params aarch64_tune_params = generic_tunings;
#define AARCH64_CPU_DEFAULT_FLAGS ((selected_cpu) ? selected_cpu->flags : 0)
struct aarch64_option_extension
{
const char *const name;
const unsigned long flags_on;
const unsigned long flags_off;
};
typedef enum aarch64_cond_code
{
AARCH64_EQ = 0, AARCH64_NE, AARCH64_CS, AARCH64_CC, AARCH64_MI, AARCH64_PL,
AARCH64_VS, AARCH64_VC, AARCH64_HI, AARCH64_LS, AARCH64_GE, AARCH64_LT,
AARCH64_GT, AARCH64_LE, AARCH64_AL, AARCH64_NV
}
aarch64_cc;
#define AARCH64_INVERSE_CONDITION_CODE(X) ((aarch64_cc) (((int) X) ^ 1))
static const char * const aarch64_condition_codes[] =
{
"eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
"hi", "ls", "ge", "lt", "gt", "le", "al", "nv"
};
const char *
aarch64_gen_far_branch (rtx * operands, int pos_label, const char * dest,
const char * branch_format)
{
rtx_code_label * tmp_label = gen_label_rtx ();
char label_buf[256];
char buffer[128];
ASM_GENERATE_INTERNAL_LABEL (label_buf, dest,
CODE_LABEL_NUMBER (tmp_label));
const char *label_ptr = targetm.strip_name_encoding (label_buf);
rtx dest_label = operands[pos_label];
operands[pos_label] = tmp_label;
snprintf (buffer, sizeof (buffer), "%s%s", branch_format, label_ptr);
output_asm_insn (buffer, operands);
snprintf (buffer, sizeof (buffer), "b\t%%l%d\n%s:", pos_label, label_ptr);
operands[pos_label] = dest_label;
output_asm_insn (buffer, operands);
return "";
}
void
aarch64_err_no_fpadvsimd (machine_mode mode, const char *msg)
{
const char *mc = FLOAT_MODE_P (mode) ? "floating-point" : "vector";
if (TARGET_GENERAL_REGS_ONLY)
error ("%qs is incompatible with %s %s", "-mgeneral-regs-only", mc, msg);
else
error ("%qs feature modifier is incompatible with %s %s", "+nofp", mc, msg);
}
static reg_class_t
aarch64_ira_change_pseudo_allocno_class (int regno, reg_class_t allocno_class,
reg_class_t best_class)
{
machine_mode mode;
if (allocno_class != ALL_REGS)
return allocno_class;
if (best_class != ALL_REGS)
return best_class;
mode = PSEUDO_REGNO_MODE (regno);
return FLOAT_MODE_P (mode) || VECTOR_MODE_P (mode) ? FP_REGS : GENERAL_REGS;
}
static unsigned int
aarch64_min_divisions_for_recip_mul (machine_mode mode)
{
if (GET_MODE_UNIT_SIZE (mode) == 4)
return aarch64_tune_params.min_div_recip_mul_sf;
return aarch64_tune_params.min_div_recip_mul_df;
}
static int
aarch64_reassociation_width (unsigned opc, machine_mode mode)
{
if (VECTOR_MODE_P (mode))
return aarch64_tune_params.vec_reassoc_width;
if (INTEGRAL_MODE_P (mode))
return aarch64_tune_params.int_reassoc_width;
if (FLOAT_MODE_P (mode) && opc != PLUS_EXPR)
return aarch64_tune_params.fp_reassoc_width;
return 1;
}
unsigned
aarch64_dbx_register_number (unsigned regno)
{
if (GP_REGNUM_P (regno))
return AARCH64_DWARF_R0 + regno - R0_REGNUM;
else if (regno == SP_REGNUM)
return AARCH64_DWARF_SP;
else if (FP_REGNUM_P (regno))
return AARCH64_DWARF_V0 + regno - V0_REGNUM;
else if (PR_REGNUM_P (regno))
return AARCH64_DWARF_P0 + regno - P0_REGNUM;
else if (regno == VG_REGNUM)
return AARCH64_DWARF_VG;
return DWARF_FRAME_REGISTERS;
}
static bool
aarch64_advsimd_struct_mode_p (machine_mode mode)
{
return (TARGET_SIMD
&& (mode == OImode || mode == CImode || mode == XImode));
}
static bool
aarch64_sve_pred_mode_p (machine_mode mode)
{
return (TARGET_SVE
&& (mode == VNx16BImode
|| mode == VNx8BImode
|| mode == VNx4BImode
|| mode == VNx2BImode));
}
const unsigned int VEC_ADVSIMD  = 1;
const unsigned int VEC_SVE_DATA = 2;
const unsigned int VEC_SVE_PRED = 4;
const unsigned int VEC_STRUCT   = 8;
const unsigned int VEC_ANY_SVE  = VEC_SVE_DATA | VEC_SVE_PRED;
const unsigned int VEC_ANY_DATA = VEC_ADVSIMD | VEC_SVE_DATA;
static unsigned int
aarch64_classify_vector_mode (machine_mode mode)
{
if (aarch64_advsimd_struct_mode_p (mode))
return VEC_ADVSIMD | VEC_STRUCT;
if (aarch64_sve_pred_mode_p (mode))
return VEC_SVE_PRED;
scalar_mode inner = GET_MODE_INNER (mode);
if (VECTOR_MODE_P (mode)
&& (inner == QImode
|| inner == HImode
|| inner == HFmode
|| inner == SImode
|| inner == SFmode
|| inner == DImode
|| inner == DFmode))
{
if (TARGET_SVE)
{
if (known_eq (GET_MODE_BITSIZE (mode), BITS_PER_SVE_VECTOR))
return VEC_SVE_DATA;
if (known_eq (GET_MODE_BITSIZE (mode), BITS_PER_SVE_VECTOR * 2)
|| known_eq (GET_MODE_BITSIZE (mode), BITS_PER_SVE_VECTOR * 3)
|| known_eq (GET_MODE_BITSIZE (mode), BITS_PER_SVE_VECTOR * 4))
return VEC_SVE_DATA | VEC_STRUCT;
}
if (TARGET_SIMD
&& (known_eq (GET_MODE_BITSIZE (mode), 64)
|| known_eq (GET_MODE_BITSIZE (mode), 128)))
return VEC_ADVSIMD;
}
return 0;
}
static bool
aarch64_vector_data_mode_p (machine_mode mode)
{
return aarch64_classify_vector_mode (mode) & VEC_ANY_DATA;
}
static bool
aarch64_sve_data_mode_p (machine_mode mode)
{
return aarch64_classify_vector_mode (mode) & VEC_SVE_DATA;
}
static opt_machine_mode
aarch64_array_mode (machine_mode mode, unsigned HOST_WIDE_INT nelems)
{
if (aarch64_classify_vector_mode (mode) == VEC_SVE_DATA
&& IN_RANGE (nelems, 2, 4))
return mode_for_vector (GET_MODE_INNER (mode),
GET_MODE_NUNITS (mode) * nelems);
return opt_machine_mode ();
}
static bool
aarch64_array_mode_supported_p (machine_mode mode,
unsigned HOST_WIDE_INT nelems)
{
if (TARGET_SIMD
&& (AARCH64_VALID_SIMD_QREG_MODE (mode)
|| AARCH64_VALID_SIMD_DREG_MODE (mode))
&& (nelems >= 2 && nelems <= 4))
return true;
return false;
}
opt_machine_mode
aarch64_sve_pred_mode (unsigned int elem_nbytes)
{
if (TARGET_SVE)
{
if (elem_nbytes == 1)
return VNx16BImode;
if (elem_nbytes == 2)
return VNx8BImode;
if (elem_nbytes == 4)
return VNx4BImode;
if (elem_nbytes == 8)
return VNx2BImode;
}
return opt_machine_mode ();
}
static opt_machine_mode
aarch64_get_mask_mode (poly_uint64 nunits, poly_uint64 nbytes)
{
if (TARGET_SVE && known_eq (nbytes, BYTES_PER_SVE_VECTOR))
{
unsigned int elem_nbytes = vector_element_size (nbytes, nunits);
machine_mode pred_mode;
if (aarch64_sve_pred_mode (elem_nbytes).exists (&pred_mode))
return pred_mode;
}
return default_get_mask_mode (nunits, nbytes);
}
static unsigned int
aarch64_hard_regno_nregs (unsigned regno, machine_mode mode)
{
HOST_WIDE_INT lowest_size = constant_lower_bound (GET_MODE_SIZE (mode));
switch (aarch64_regno_regclass (regno))
{
case FP_REGS:
case FP_LO_REGS:
if (aarch64_sve_data_mode_p (mode))
return exact_div (GET_MODE_SIZE (mode),
BYTES_PER_SVE_VECTOR).to_constant ();
return CEIL (lowest_size, UNITS_PER_VREG);
case PR_REGS:
case PR_LO_REGS:
case PR_HI_REGS:
return 1;
default:
return CEIL (lowest_size, UNITS_PER_WORD);
}
gcc_unreachable ();
}
static bool
aarch64_hard_regno_mode_ok (unsigned regno, machine_mode mode)
{
if (GET_MODE_CLASS (mode) == MODE_CC)
return regno == CC_REGNUM;
if (regno == VG_REGNUM)
return mode == DImode;
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
if (vec_flags & VEC_SVE_PRED)
return PR_REGNUM_P (regno);
if (PR_REGNUM_P (regno))
return 0;
if (regno == SP_REGNUM)
return mode == Pmode || mode == ptr_mode;
if (regno == FRAME_POINTER_REGNUM || regno == ARG_POINTER_REGNUM)
return mode == Pmode;
if (GP_REGNUM_P (regno) && known_le (GET_MODE_SIZE (mode), 16))
return true;
if (FP_REGNUM_P (regno))
{
if (vec_flags & VEC_STRUCT)
return end_hard_regno (mode, regno) - 1 <= V31_REGNUM;
else
return !VECTOR_MODE_P (mode) || vec_flags != 0;
}
return false;
}
static bool
aarch64_hard_regno_call_part_clobbered (unsigned int regno, machine_mode mode)
{
return FP_REGNUM_P (regno) && maybe_gt (GET_MODE_SIZE (mode), 8);
}
poly_uint64
aarch64_regmode_natural_size (machine_mode mode)
{
if (!aarch64_sve_vg.is_constant ())
{
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
if (vec_flags & VEC_SVE_PRED)
return BYTES_PER_SVE_PRED;
if (vec_flags & VEC_SVE_DATA)
return BYTES_PER_SVE_VECTOR;
}
return UNITS_PER_WORD;
}
machine_mode
aarch64_hard_regno_caller_save_mode (unsigned regno, unsigned,
machine_mode mode)
{
if (PR_REGNUM_P (regno))
return mode;
if (known_ge (GET_MODE_SIZE (mode), 4))
return mode;
else
return SImode;
}
static HOST_WIDE_INT
aarch64_constant_alignment (const_tree exp, HOST_WIDE_INT align)
{
if (TREE_CODE (exp) == STRING_CST && !optimize_size)
return MAX (align, BITS_PER_WORD);
return align;
}
static bool
aarch64_decl_is_long_call_p (const_tree decl ATTRIBUTE_UNUSED)
{
return false;
}
bool
aarch64_is_long_call_p (rtx sym)
{
return aarch64_decl_is_long_call_p (SYMBOL_REF_DECL (sym));
}
bool
aarch64_is_noplt_call_p (rtx sym)
{
const_tree decl = SYMBOL_REF_DECL (sym);
if (flag_pic
&& decl
&& (!flag_plt
|| lookup_attribute ("noplt", DECL_ATTRIBUTES (decl)))
&& !targetm.binds_local_p (decl))
return true;
return false;
}
bool
aarch64_is_extend_from_extract (scalar_int_mode mode, rtx mult_imm,
rtx extract_imm)
{
HOST_WIDE_INT mult_val, extract_val;
if (! CONST_INT_P (mult_imm) || ! CONST_INT_P (extract_imm))
return false;
mult_val = INTVAL (mult_imm);
extract_val = INTVAL (extract_imm);
if (extract_val > 8
&& extract_val < GET_MODE_BITSIZE (mode)
&& exact_log2 (extract_val & ~7) > 0
&& (extract_val & 7) <= 4
&& mult_val == (1 << (extract_val & 7)))
return true;
return false;
}
inline static rtx_insn *
emit_set_insn (rtx x, rtx y)
{
return emit_insn (gen_rtx_SET (x, y));
}
rtx
aarch64_gen_compare_reg (RTX_CODE code, rtx x, rtx y)
{
machine_mode mode = SELECT_CC_MODE (code, x, y);
rtx cc_reg = gen_rtx_REG (mode, CC_REGNUM);
emit_set_insn (cc_reg, gen_rtx_COMPARE (mode, x, y));
return cc_reg;
}
static GTY(()) rtx tls_get_addr_libfunc;
rtx
aarch64_tls_get_addr (void)
{
if (!tls_get_addr_libfunc)
tls_get_addr_libfunc = init_one_libfunc ("__tls_get_addr");
return tls_get_addr_libfunc;
}
static enum tls_model
tls_symbolic_operand_type (rtx addr)
{
enum tls_model tls_kind = TLS_MODEL_NONE;
if (GET_CODE (addr) == CONST)
{
poly_int64 addend;
rtx sym = strip_offset (addr, &addend);
if (GET_CODE (sym) == SYMBOL_REF)
tls_kind = SYMBOL_REF_TLS_MODEL (sym);
}
else if (GET_CODE (addr) == SYMBOL_REF)
tls_kind = SYMBOL_REF_TLS_MODEL (addr);
return tls_kind;
}
static void
aarch64_load_symref_appropriately (rtx dest, rtx imm,
enum aarch64_symbol_type type)
{
switch (type)
{
case SYMBOL_SMALL_ABSOLUTE:
{
rtx tmp_reg = dest;
machine_mode mode = GET_MODE (dest);
gcc_assert (mode == Pmode || mode == ptr_mode);
if (can_create_pseudo_p ())
tmp_reg = gen_reg_rtx (mode);
emit_move_insn (tmp_reg, gen_rtx_HIGH (mode, imm));
emit_insn (gen_add_losym (dest, tmp_reg, imm));
return;
}
case SYMBOL_TINY_ABSOLUTE:
emit_insn (gen_rtx_SET (dest, imm));
return;
case SYMBOL_SMALL_GOT_28K:
{
machine_mode mode = GET_MODE (dest);
rtx gp_rtx = pic_offset_table_rtx;
rtx insn;
rtx mem;
if (gp_rtx != NULL)
{
rtx s = gen_rtx_SYMBOL_REF (Pmode, "_GLOBAL_OFFSET_TABLE_");
crtl->uses_pic_offset_table = 1;
emit_move_insn (gp_rtx, gen_rtx_HIGH (Pmode, s));
if (mode != GET_MODE (gp_rtx))
gp_rtx = gen_lowpart (mode, gp_rtx);
}
if (mode == ptr_mode)
{
if (mode == DImode)
insn = gen_ldr_got_small_28k_di (dest, gp_rtx, imm);
else
insn = gen_ldr_got_small_28k_si (dest, gp_rtx, imm);
mem = XVECEXP (SET_SRC (insn), 0, 0);
}
else
{
gcc_assert (mode == Pmode);
insn = gen_ldr_got_small_28k_sidi (dest, gp_rtx, imm);
mem = XVECEXP (XEXP (SET_SRC (insn), 0), 0, 0);
}
gcc_assert (GET_CODE (mem) == MEM);
MEM_READONLY_P (mem) = 1;
MEM_NOTRAP_P (mem) = 1;
emit_insn (insn);
return;
}
case SYMBOL_SMALL_GOT_4G:
{
rtx insn;
rtx mem;
rtx tmp_reg = dest;
machine_mode mode = GET_MODE (dest);
if (can_create_pseudo_p ())
tmp_reg = gen_reg_rtx (mode);
emit_move_insn (tmp_reg, gen_rtx_HIGH (mode, imm));
if (mode == ptr_mode)
{
if (mode == DImode)
insn = gen_ldr_got_small_di (dest, tmp_reg, imm);
else
insn = gen_ldr_got_small_si (dest, tmp_reg, imm);
mem = XVECEXP (SET_SRC (insn), 0, 0);
}
else
{
gcc_assert (mode == Pmode);
insn = gen_ldr_got_small_sidi (dest, tmp_reg, imm);
mem = XVECEXP (XEXP (SET_SRC (insn), 0), 0, 0);
}
gcc_assert (GET_CODE (mem) == MEM);
MEM_READONLY_P (mem) = 1;
MEM_NOTRAP_P (mem) = 1;
emit_insn (insn);
return;
}
case SYMBOL_SMALL_TLSGD:
{
rtx_insn *insns;
machine_mode mode = GET_MODE (dest);
rtx result = gen_rtx_REG (mode, R0_REGNUM);
start_sequence ();
if (TARGET_ILP32)
aarch64_emit_call_insn (gen_tlsgd_small_si (result, imm));
else
aarch64_emit_call_insn (gen_tlsgd_small_di (result, imm));
insns = get_insns ();
end_sequence ();
RTL_CONST_CALL_P (insns) = 1;
emit_libcall_block (insns, dest, result, imm);
return;
}
case SYMBOL_SMALL_TLSDESC:
{
machine_mode mode = GET_MODE (dest);
rtx x0 = gen_rtx_REG (mode, R0_REGNUM);
rtx tp;
gcc_assert (mode == Pmode || mode == ptr_mode);
if (TARGET_ILP32)
emit_insn (gen_tlsdesc_small_si (imm));
else
emit_insn (gen_tlsdesc_small_di (imm));
tp = aarch64_load_tp (NULL);
if (mode != Pmode)
tp = gen_lowpart (mode, tp);
emit_insn (gen_rtx_SET (dest, gen_rtx_PLUS (mode, tp, x0)));
if (REG_P (dest))
set_unique_reg_note (get_last_insn (), REG_EQUIV, imm);
return;
}
case SYMBOL_SMALL_TLSIE:
{
machine_mode mode = GET_MODE (dest);
rtx tmp_reg = gen_reg_rtx (mode);
rtx tp = aarch64_load_tp (NULL);
if (mode == ptr_mode)
{
if (mode == DImode)
emit_insn (gen_tlsie_small_di (tmp_reg, imm));
else
{
emit_insn (gen_tlsie_small_si (tmp_reg, imm));
tp = gen_lowpart (mode, tp);
}
}
else
{
gcc_assert (mode == Pmode);
emit_insn (gen_tlsie_small_sidi (tmp_reg, imm));
}
emit_insn (gen_rtx_SET (dest, gen_rtx_PLUS (mode, tp, tmp_reg)));
if (REG_P (dest))
set_unique_reg_note (get_last_insn (), REG_EQUIV, imm);
return;
}
case SYMBOL_TLSLE12:
case SYMBOL_TLSLE24:
case SYMBOL_TLSLE32:
case SYMBOL_TLSLE48:
{
machine_mode mode = GET_MODE (dest);
rtx tp = aarch64_load_tp (NULL);
if (mode != Pmode)
tp = gen_lowpart (mode, tp);
switch (type)
{
case SYMBOL_TLSLE12:
emit_insn ((mode == DImode ? gen_tlsle12_di : gen_tlsle12_si)
(dest, tp, imm));
break;
case SYMBOL_TLSLE24:
emit_insn ((mode == DImode ? gen_tlsle24_di : gen_tlsle24_si)
(dest, tp, imm));
break;
case SYMBOL_TLSLE32:
emit_insn ((mode == DImode ? gen_tlsle32_di : gen_tlsle32_si)
(dest, imm));
emit_insn ((mode == DImode ? gen_adddi3 : gen_addsi3)
(dest, dest, tp));
break;
case SYMBOL_TLSLE48:
emit_insn ((mode == DImode ? gen_tlsle48_di : gen_tlsle48_si)
(dest, imm));
emit_insn ((mode == DImode ? gen_adddi3 : gen_addsi3)
(dest, dest, tp));
break;
default:
gcc_unreachable ();
}
if (REG_P (dest))
set_unique_reg_note (get_last_insn (), REG_EQUIV, imm);
return;
}
case SYMBOL_TINY_GOT:
emit_insn (gen_ldr_got_tiny (dest, imm));
return;
case SYMBOL_TINY_TLSIE:
{
machine_mode mode = GET_MODE (dest);
rtx tp = aarch64_load_tp (NULL);
if (mode == ptr_mode)
{
if (mode == DImode)
emit_insn (gen_tlsie_tiny_di (dest, imm, tp));
else
{
tp = gen_lowpart (mode, tp);
emit_insn (gen_tlsie_tiny_si (dest, imm, tp));
}
}
else
{
gcc_assert (mode == Pmode);
emit_insn (gen_tlsie_tiny_sidi (dest, imm, tp));
}
if (REG_P (dest))
set_unique_reg_note (get_last_insn (), REG_EQUIV, imm);
return;
}
default:
gcc_unreachable ();
}
}
static rtx
aarch64_emit_move (rtx dest, rtx src)
{
return (can_create_pseudo_p ()
? emit_move_insn (dest, src)
: emit_move_insn_1 (dest, src));
}
void
aarch64_split_128bit_move (rtx dst, rtx src)
{
rtx dst_lo, dst_hi;
rtx src_lo, src_hi;
machine_mode mode = GET_MODE (dst);
gcc_assert (mode == TImode || mode == TFmode);
gcc_assert (!(side_effects_p (src) || side_effects_p (dst)));
gcc_assert (mode == GET_MODE (src) || GET_MODE (src) == VOIDmode);
if (REG_P (dst) && REG_P (src))
{
int src_regno = REGNO (src);
int dst_regno = REGNO (dst);
if (FP_REGNUM_P (dst_regno) && GP_REGNUM_P (src_regno))
{
src_lo = gen_lowpart (word_mode, src);
src_hi = gen_highpart (word_mode, src);
if (mode == TImode)
{
emit_insn (gen_aarch64_movtilow_di (dst, src_lo));
emit_insn (gen_aarch64_movtihigh_di (dst, src_hi));
}
else
{
emit_insn (gen_aarch64_movtflow_di (dst, src_lo));
emit_insn (gen_aarch64_movtfhigh_di (dst, src_hi));
}
return;
}
else if (GP_REGNUM_P (dst_regno) && FP_REGNUM_P (src_regno))
{
dst_lo = gen_lowpart (word_mode, dst);
dst_hi = gen_highpart (word_mode, dst);
if (mode == TImode)
{
emit_insn (gen_aarch64_movdi_tilow (dst_lo, src));
emit_insn (gen_aarch64_movdi_tihigh (dst_hi, src));
}
else
{
emit_insn (gen_aarch64_movdi_tflow (dst_lo, src));
emit_insn (gen_aarch64_movdi_tfhigh (dst_hi, src));
}
return;
}
}
dst_lo = gen_lowpart (word_mode, dst);
dst_hi = gen_highpart (word_mode, dst);
src_lo = gen_lowpart (word_mode, src);
src_hi = gen_highpart_mode (word_mode, mode, src);
if (reg_overlap_mentioned_p (dst_lo, src_hi))
{
aarch64_emit_move (dst_hi, src_hi);
aarch64_emit_move (dst_lo, src_lo);
}
else
{
aarch64_emit_move (dst_lo, src_lo);
aarch64_emit_move (dst_hi, src_hi);
}
}
bool
aarch64_split_128bit_move_p (rtx dst, rtx src)
{
return (! REG_P (src)
|| ! (FP_REGNUM_P (REGNO (dst)) && FP_REGNUM_P (REGNO (src))));
}
void
aarch64_split_simd_combine (rtx dst, rtx src1, rtx src2)
{
machine_mode src_mode = GET_MODE (src1);
machine_mode dst_mode = GET_MODE (dst);
gcc_assert (VECTOR_MODE_P (dst_mode));
gcc_assert (register_operand (dst, dst_mode)
&& register_operand (src1, src_mode)
&& register_operand (src2, src_mode));
rtx (*gen) (rtx, rtx, rtx);
switch (src_mode)
{
case E_V8QImode:
gen = gen_aarch64_simd_combinev8qi;
break;
case E_V4HImode:
gen = gen_aarch64_simd_combinev4hi;
break;
case E_V2SImode:
gen = gen_aarch64_simd_combinev2si;
break;
case E_V4HFmode:
gen = gen_aarch64_simd_combinev4hf;
break;
case E_V2SFmode:
gen = gen_aarch64_simd_combinev2sf;
break;
case E_DImode:
gen = gen_aarch64_simd_combinedi;
break;
case E_DFmode:
gen = gen_aarch64_simd_combinedf;
break;
default:
gcc_unreachable ();
}
emit_insn (gen (dst, src1, src2));
return;
}
void
aarch64_split_simd_move (rtx dst, rtx src)
{
machine_mode src_mode = GET_MODE (src);
machine_mode dst_mode = GET_MODE (dst);
gcc_assert (VECTOR_MODE_P (dst_mode));
if (REG_P (dst) && REG_P (src))
{
rtx (*gen) (rtx, rtx);
gcc_assert (VECTOR_MODE_P (src_mode));
switch (src_mode)
{
case E_V16QImode:
gen = gen_aarch64_split_simd_movv16qi;
break;
case E_V8HImode:
gen = gen_aarch64_split_simd_movv8hi;
break;
case E_V4SImode:
gen = gen_aarch64_split_simd_movv4si;
break;
case E_V2DImode:
gen = gen_aarch64_split_simd_movv2di;
break;
case E_V8HFmode:
gen = gen_aarch64_split_simd_movv8hf;
break;
case E_V4SFmode:
gen = gen_aarch64_split_simd_movv4sf;
break;
case E_V2DFmode:
gen = gen_aarch64_split_simd_movv2df;
break;
default:
gcc_unreachable ();
}
emit_insn (gen (dst, src));
return;
}
}
bool
aarch64_zero_extend_const_eq (machine_mode xmode, rtx x,
machine_mode ymode, rtx y)
{
rtx r = simplify_const_unary_operation (ZERO_EXTEND, xmode, y, ymode);
gcc_assert (r != NULL);
return rtx_equal_p (x, r);
}
static rtx
aarch64_force_temporary (machine_mode mode, rtx x, rtx value)
{
if (can_create_pseudo_p ())
return force_reg (mode, value);
else
{
gcc_assert (x);
aarch64_emit_move (x, value);
return x;
}
}
static bool
aarch64_sve_cnt_immediate_p (poly_int64 value)
{
HOST_WIDE_INT factor = value.coeffs[0];
return (value.coeffs[1] == factor
&& IN_RANGE (factor, 2, 16 * 16)
&& (factor & 1) == 0
&& factor <= 16 * (factor & -factor));
}
bool
aarch64_sve_cnt_immediate_p (rtx x)
{
poly_int64 value;
return poly_int_rtx_p (x, &value) && aarch64_sve_cnt_immediate_p (value);
}
static char *
aarch64_output_sve_cnt_immediate (const char *prefix, const char *operands,
unsigned int factor,
unsigned int nelts_per_vq)
{
static char buffer[sizeof ("sqincd\t%x0, %w0, all, mul #16")];
if (nelts_per_vq == 0)
nelts_per_vq = factor & -factor;
int shift = std::min (exact_log2 (nelts_per_vq), 4);
gcc_assert (IN_RANGE (shift, 1, 4));
char suffix = "dwhb"[shift - 1];
factor >>= shift;
unsigned int written;
if (factor == 1)
written = snprintf (buffer, sizeof (buffer), "%s%c\t%s",
prefix, suffix, operands);
else
written = snprintf (buffer, sizeof (buffer), "%s%c\t%s, all, mul #%d",
prefix, suffix, operands, factor);
gcc_assert (written < sizeof (buffer));
return buffer;
}
char *
aarch64_output_sve_cnt_immediate (const char *prefix, const char *operands,
rtx x)
{
poly_int64 value = rtx_to_poly_int64 (x);
gcc_assert (aarch64_sve_cnt_immediate_p (value));
return aarch64_output_sve_cnt_immediate (prefix, operands,
value.coeffs[1], 0);
}
static bool
aarch64_sve_addvl_addpl_immediate_p (poly_int64 value)
{
HOST_WIDE_INT factor = value.coeffs[0];
if (factor == 0 || value.coeffs[1] != factor)
return false;
return (((factor & 15) == 0 && IN_RANGE (factor, -32 * 16, 31 * 16))
|| ((factor & 1) == 0 && IN_RANGE (factor, -32 * 2, 31 * 2)));
}
bool
aarch64_sve_addvl_addpl_immediate_p (rtx x)
{
poly_int64 value;
return (poly_int_rtx_p (x, &value)
&& aarch64_sve_addvl_addpl_immediate_p (value));
}
char *
aarch64_output_sve_addvl_addpl (rtx dest, rtx base, rtx offset)
{
static char buffer[sizeof ("addpl\t%x0, %x1, #-") + 3 * sizeof (int)];
poly_int64 offset_value = rtx_to_poly_int64 (offset);
gcc_assert (aarch64_sve_addvl_addpl_immediate_p (offset_value));
if (rtx_equal_p (dest, base) && GP_REGNUM_P (REGNO (dest)))
{
if (aarch64_sve_cnt_immediate_p (offset_value))
return aarch64_output_sve_cnt_immediate ("inc", "%x0",
offset_value.coeffs[1], 0);
if (aarch64_sve_cnt_immediate_p (-offset_value))
return aarch64_output_sve_cnt_immediate ("dec", "%x0",
-offset_value.coeffs[1], 0);
}
int factor = offset_value.coeffs[1];
if ((factor & 15) == 0)
snprintf (buffer, sizeof (buffer), "addvl\t%%x0, %%x1, #%d", factor / 16);
else
snprintf (buffer, sizeof (buffer), "addpl\t%%x0, %%x1, #%d", factor / 2);
return buffer;
}
bool
aarch64_sve_inc_dec_immediate_p (rtx x, int *factor_out,
unsigned int *nelts_per_vq_out)
{
rtx elt;
poly_int64 value;
if (!const_vec_duplicate_p (x, &elt)
|| !poly_int_rtx_p (elt, &value))
return false;
unsigned int nelts_per_vq = 128 / GET_MODE_UNIT_BITSIZE (GET_MODE (x));
if (nelts_per_vq != 8 && nelts_per_vq != 4 && nelts_per_vq != 2)
return false;
HOST_WIDE_INT factor = value.coeffs[0];
if (value.coeffs[1] != factor)
return false;
if ((factor % nelts_per_vq) != 0
|| !IN_RANGE (abs (factor), nelts_per_vq, 16 * nelts_per_vq))
return false;
if (factor_out)
*factor_out = factor;
if (nelts_per_vq_out)
*nelts_per_vq_out = nelts_per_vq;
return true;
}
bool
aarch64_sve_inc_dec_immediate_p (rtx x)
{
return aarch64_sve_inc_dec_immediate_p (x, NULL, NULL);
}
char *
aarch64_output_sve_inc_dec_immediate (const char *operands, rtx x)
{
int factor;
unsigned int nelts_per_vq;
if (!aarch64_sve_inc_dec_immediate_p (x, &factor, &nelts_per_vq))
gcc_unreachable ();
if (factor < 0)
return aarch64_output_sve_cnt_immediate ("dec", operands, -factor,
nelts_per_vq);
else
return aarch64_output_sve_cnt_immediate ("inc", operands, factor,
nelts_per_vq);
}
static int
aarch64_internal_mov_immediate (rtx dest, rtx imm, bool generate,
scalar_int_mode mode)
{
int i;
unsigned HOST_WIDE_INT val, val2, mask;
int one_match, zero_match;
int num_insns;
val = INTVAL (imm);
if (aarch64_move_imm (val, mode))
{
if (generate)
emit_insn (gen_rtx_SET (dest, imm));
return 1;
}
val2 = val & 0xffffffff;
if (mode == DImode
&& aarch64_move_imm (val2, SImode)
&& (((val >> 32) & 0xffff) == 0 || (val >> 48) == 0))
{
if (generate)
emit_insn (gen_rtx_SET (dest, GEN_INT (val2)));
if (val == val2)
return 1;
i = (val >> 48) ? 48 : 32;
if (generate)
emit_insn (gen_insv_immdi (dest, GEN_INT (i),
GEN_INT ((val >> i) & 0xffff)));
return 2;
}
if ((val >> 32) == 0 || mode == SImode)
{
if (generate)
{
emit_insn (gen_rtx_SET (dest, GEN_INT (val & 0xffff)));
if (mode == SImode)
emit_insn (gen_insv_immsi (dest, GEN_INT (16),
GEN_INT ((val >> 16) & 0xffff)));
else
emit_insn (gen_insv_immdi (dest, GEN_INT (16),
GEN_INT ((val >> 16) & 0xffff)));
}
return 2;
}
mask = 0xffff;
zero_match = ((val & mask) == 0) + ((val & (mask << 16)) == 0) +
((val & (mask << 32)) == 0) + ((val & (mask << 48)) == 0);
one_match = ((~val & mask) == 0) + ((~val & (mask << 16)) == 0) +
((~val & (mask << 32)) == 0) + ((~val & (mask << 48)) == 0);
if (zero_match != 2 && one_match != 2)
{
for (i = 0; i < 64; i += 16, mask <<= 16)
{
val2 = val & ~mask;
if (val2 != val && aarch64_bitmask_imm (val2, mode))
break;
val2 = val | mask;
if (val2 != val && aarch64_bitmask_imm (val2, mode))
break;
val2 = val2 & ~mask;
val2 = val2 | (((val2 >> 32) | (val2 << 32)) & mask);
if (val2 != val && aarch64_bitmask_imm (val2, mode))
break;
}
if (i != 64)
{
if (generate)
{
emit_insn (gen_rtx_SET (dest, GEN_INT (val2)));
emit_insn (gen_insv_immdi (dest, GEN_INT (i),
GEN_INT ((val >> i) & 0xffff)));
}
return 2;
}
}
num_insns = 1;
mask = 0xffff;
val2 = one_match > zero_match ? ~val : val;
i = (val2 & mask) != 0 ? 0 : (val2 & (mask << 16)) != 0 ? 16 : 32;
if (generate)
emit_insn (gen_rtx_SET (dest, GEN_INT (one_match > zero_match
? (val | ~(mask << i))
: (val & (mask << i)))));
for (i += 16; i < 64; i += 16)
{
if ((val2 & (mask << i)) == 0)
continue;
if (generate)
emit_insn (gen_insv_immdi (dest, GEN_INT (i),
GEN_INT ((val >> i) & 0xffff)));
num_insns ++;
}
return num_insns;
}
bool
aarch64_mov128_immediate (rtx imm)
{
if (GET_CODE (imm) == CONST_INT)
return true;
gcc_assert (CONST_WIDE_INT_NUNITS (imm) == 2);
rtx lo = GEN_INT (CONST_WIDE_INT_ELT (imm, 0));
rtx hi = GEN_INT (CONST_WIDE_INT_ELT (imm, 1));
return aarch64_internal_mov_immediate (NULL_RTX, lo, false, DImode)
+ aarch64_internal_mov_immediate (NULL_RTX, hi, false, DImode) <= 4;
}
static unsigned int
aarch64_add_offset_1_temporaries (HOST_WIDE_INT offset)
{
return abs_hwi (offset) < 0x1000000 ? 0 : 1;
}
static void
aarch64_add_offset_1 (scalar_int_mode mode, rtx dest,
rtx src, HOST_WIDE_INT offset, rtx temp1,
bool frame_related_p, bool emit_move_imm)
{
gcc_assert (emit_move_imm || temp1 != NULL_RTX);
gcc_assert (temp1 == NULL_RTX || !reg_overlap_mentioned_p (temp1, src));
HOST_WIDE_INT moffset = abs_hwi (offset);
rtx_insn *insn;
if (!moffset)
{
if (!rtx_equal_p (dest, src))
{
insn = emit_insn (gen_rtx_SET (dest, src));
RTX_FRAME_RELATED_P (insn) = frame_related_p;
}
return;
}
if (aarch64_uimm12_shift (moffset))
{
insn = emit_insn (gen_add3_insn (dest, src, GEN_INT (offset)));
RTX_FRAME_RELATED_P (insn) = frame_related_p;
return;
}
if (moffset < 0x1000000
&& ((!temp1 && !can_create_pseudo_p ())
|| !aarch64_move_imm (moffset, mode)))
{
HOST_WIDE_INT low_off = moffset & 0xfff;
low_off = offset < 0 ? -low_off : low_off;
insn = emit_insn (gen_add3_insn (dest, src, GEN_INT (low_off)));
RTX_FRAME_RELATED_P (insn) = frame_related_p;
insn = emit_insn (gen_add2_insn (dest, GEN_INT (offset - low_off)));
RTX_FRAME_RELATED_P (insn) = frame_related_p;
return;
}
if (emit_move_imm)
{
gcc_assert (temp1 != NULL_RTX || can_create_pseudo_p ());
temp1 = aarch64_force_temporary (mode, temp1, GEN_INT (moffset));
}
insn = emit_insn (offset < 0
? gen_sub3_insn (dest, src, temp1)
: gen_add3_insn (dest, src, temp1));
if (frame_related_p)
{
RTX_FRAME_RELATED_P (insn) = frame_related_p;
rtx adj = plus_constant (mode, src, offset);
add_reg_note (insn, REG_CFA_ADJUST_CFA, gen_rtx_SET (dest, adj));
}
}
static unsigned int
aarch64_offset_temporaries (bool add_p, poly_int64 offset)
{
if (add_p && aarch64_sve_addvl_addpl_immediate_p (offset))
return 0;
unsigned int count = 0;
HOST_WIDE_INT factor = offset.coeffs[1];
HOST_WIDE_INT constant = offset.coeffs[0] - factor;
poly_int64 poly_offset (factor, factor);
if (add_p && aarch64_sve_addvl_addpl_immediate_p (poly_offset))
count += 1;
else if (factor != 0)
{
factor = abs (factor);
if (factor > 16 * (factor & -factor))
return 2;
count += 1;
}
return count + aarch64_add_offset_1_temporaries (constant);
}
int
aarch64_add_offset_temporaries (rtx x)
{
poly_int64 offset;
if (!poly_int_rtx_p (x, &offset))
return -1;
return aarch64_offset_temporaries (true, offset);
}
static void
aarch64_add_offset (scalar_int_mode mode, rtx dest, rtx src,
poly_int64 offset, rtx temp1, rtx temp2,
bool frame_related_p, bool emit_move_imm = true)
{
gcc_assert (emit_move_imm || temp1 != NULL_RTX);
gcc_assert (temp1 == NULL_RTX || !reg_overlap_mentioned_p (temp1, src));
gcc_assert (temp1 == NULL_RTX
|| !frame_related_p
|| !reg_overlap_mentioned_p (temp1, dest));
gcc_assert (temp2 == NULL_RTX || !reg_overlap_mentioned_p (dest, temp2));
if (src != const0_rtx && aarch64_sve_addvl_addpl_immediate_p (offset))
{
rtx offset_rtx = gen_int_mode (offset, mode);
rtx_insn *insn = emit_insn (gen_add3_insn (dest, src, offset_rtx));
RTX_FRAME_RELATED_P (insn) = frame_related_p;
return;
}
HOST_WIDE_INT factor = offset.coeffs[1];
HOST_WIDE_INT constant = offset.coeffs[0] - factor;
poly_int64 poly_offset (factor, factor);
if (src != const0_rtx
&& aarch64_sve_addvl_addpl_immediate_p (poly_offset))
{
rtx offset_rtx = gen_int_mode (poly_offset, mode);
if (frame_related_p)
{
rtx_insn *insn = emit_insn (gen_add3_insn (dest, src, offset_rtx));
RTX_FRAME_RELATED_P (insn) = true;
src = dest;
}
else
{
rtx addr = gen_rtx_PLUS (mode, src, offset_rtx);
src = aarch64_force_temporary (mode, temp1, addr);
temp1 = temp2;
temp2 = NULL_RTX;
}
}
else if (factor != 0)
{
rtx_code code = PLUS;
if (factor < 0)
{
factor = -factor;
code = MINUS;
}
rtx val;
int shift = 0;
if (factor & 1)
shift = -1;
else
factor /= 2;
HOST_WIDE_INT low_bit = factor & -factor;
if (factor <= 16 * low_bit)
{
if (factor > 16 * 8)
{
int extra_shift = exact_log2 (low_bit);
shift += extra_shift;
factor >>= extra_shift;
}
val = gen_int_mode (poly_int64 (factor * 2, factor * 2), mode);
}
else
{
val = gen_int_mode (poly_int64 (2, 2), mode);
val = aarch64_force_temporary (mode, temp1, val);
if (code == MINUS && src == const0_rtx)
{
factor = -factor;
code = PLUS;
}
rtx coeff1 = gen_int_mode (factor, mode);
coeff1 = aarch64_force_temporary (mode, temp2, coeff1);
val = gen_rtx_MULT (mode, val, coeff1);
}
if (shift > 0)
{
val = aarch64_force_temporary (mode, temp1, val);
val = gen_rtx_ASHIFT (mode, val, GEN_INT (shift));
}
else if (shift == -1)
{
val = aarch64_force_temporary (mode, temp1, val);
val = gen_rtx_ASHIFTRT (mode, val, const1_rtx);
}
if (src != const0_rtx)
{
val = aarch64_force_temporary (mode, temp1, val);
val = gen_rtx_fmt_ee (code, mode, src, val);
}
else if (code == MINUS)
{
val = aarch64_force_temporary (mode, temp1, val);
val = gen_rtx_NEG (mode, val);
}
if (constant == 0 || frame_related_p)
{
rtx_insn *insn = emit_insn (gen_rtx_SET (dest, val));
if (frame_related_p)
{
RTX_FRAME_RELATED_P (insn) = true;
add_reg_note (insn, REG_CFA_ADJUST_CFA,
gen_rtx_SET (dest, plus_constant (Pmode, src,
poly_offset)));
}
src = dest;
if (constant == 0)
return;
}
else
{
src = aarch64_force_temporary (mode, temp1, val);
temp1 = temp2;
temp2 = NULL_RTX;
}
emit_move_imm = true;
}
aarch64_add_offset_1 (mode, dest, src, constant, temp1,
frame_related_p, emit_move_imm);
}
void
aarch64_split_add_offset (scalar_int_mode mode, rtx dest, rtx src,
rtx offset_rtx, rtx temp1, rtx temp2)
{
aarch64_add_offset (mode, dest, src, rtx_to_poly_int64 (offset_rtx),
temp1, temp2, false);
}
static inline void
aarch64_add_sp (rtx temp1, rtx temp2, poly_int64 delta, bool emit_move_imm)
{
aarch64_add_offset (Pmode, stack_pointer_rtx, stack_pointer_rtx, delta,
temp1, temp2, true, emit_move_imm);
}
static inline void
aarch64_sub_sp (rtx temp1, rtx temp2, poly_int64 delta, bool frame_related_p)
{
aarch64_add_offset (Pmode, stack_pointer_rtx, stack_pointer_rtx, -delta,
temp1, temp2, frame_related_p);
}
static void
aarch64_expand_vec_series (rtx dest, rtx base, rtx step)
{
machine_mode mode = GET_MODE (dest);
scalar_mode inner = GET_MODE_INNER (mode);
if (!aarch64_sve_index_immediate_p (base))
base = force_reg (inner, base);
if (!aarch64_sve_index_immediate_p (step))
step = force_reg (inner, step);
emit_set_insn (dest, gen_rtx_VEC_SERIES (mode, base, step));
}
static bool
aarch64_expand_sve_widened_duplicate (rtx dest, scalar_int_mode src_mode,
rtx src)
{
if (src_mode != TImode)
{
poly_uint64 count = exact_div (GET_MODE_SIZE (GET_MODE (dest)),
GET_MODE_SIZE (src_mode));
machine_mode dup_mode = mode_for_vector (src_mode, count).require ();
emit_move_insn (gen_lowpart (dup_mode, dest),
gen_const_vec_duplicate (dup_mode, src));
return true;
}
src = force_const_mem (src_mode, src);
if (!src)
return false;
if (!aarch64_sve_ld1r_operand_p (src))
{
rtx addr = force_reg (Pmode, XEXP (src, 0));
src = replace_equiv_address (src, addr);
}
machine_mode mode = GET_MODE (dest);
unsigned int elem_bytes = GET_MODE_UNIT_SIZE (mode);
machine_mode pred_mode = aarch64_sve_pred_mode (elem_bytes).require ();
rtx ptrue = force_reg (pred_mode, CONSTM1_RTX (pred_mode));
src = gen_rtx_UNSPEC (mode, gen_rtvec (2, ptrue, src), UNSPEC_LD1RQ);
emit_insn (gen_rtx_SET (dest, src));
return true;
}
static void
aarch64_expand_sve_const_vector (rtx dest, rtx src)
{
machine_mode mode = GET_MODE (src);
unsigned int npatterns = CONST_VECTOR_NPATTERNS (src);
unsigned int nelts_per_pattern = CONST_VECTOR_NELTS_PER_PATTERN (src);
gcc_assert (npatterns > 1);
if (nelts_per_pattern == 1)
{
scalar_int_mode int_mode;
if (BYTES_BIG_ENDIAN)
int_mode = TImode;
else
{
unsigned int int_bits = GET_MODE_UNIT_BITSIZE (mode) * npatterns;
gcc_assert (int_bits <= 128);
int_mode = int_mode_for_size (int_bits, 0).require ();
}
rtx int_value = simplify_gen_subreg (int_mode, src, mode, 0);
if (int_value
&& aarch64_expand_sve_widened_duplicate (dest, int_mode, int_value))
return;
}
rtx_vector_builder builder;
auto_vec<rtx, 16> vectors (npatterns);
for (unsigned int i = 0; i < npatterns; ++i)
{
builder.new_vector (mode, 1, nelts_per_pattern);
for (unsigned int j = 0; j < nelts_per_pattern; ++j)
builder.quick_push (CONST_VECTOR_ELT (src, i + j * npatterns));
vectors.quick_push (force_reg (mode, builder.build ()));
}
while (npatterns > 1)
{
npatterns /= 2;
for (unsigned int i = 0; i < npatterns; ++i)
{
rtx tmp = (npatterns == 1 ? dest : gen_reg_rtx (mode));
rtvec v = gen_rtvec (2, vectors[i], vectors[i + npatterns]);
emit_set_insn (tmp, gen_rtx_UNSPEC (mode, v, UNSPEC_ZIP1));
vectors[i] = tmp;
}
}
gcc_assert (vectors[0] == dest);
}
void
aarch64_expand_mov_immediate (rtx dest, rtx imm,
rtx (*gen_vec_duplicate) (rtx, rtx))
{
machine_mode mode = GET_MODE (dest);
scalar_int_mode int_mode;
if ((GET_CODE (imm) == SYMBOL_REF
|| GET_CODE (imm) == LABEL_REF
|| GET_CODE (imm) == CONST
|| GET_CODE (imm) == CONST_POLY_INT)
&& is_a <scalar_int_mode> (mode, &int_mode))
{
rtx mem;
poly_int64 offset;
HOST_WIDE_INT const_offset;
enum aarch64_symbol_type sty;
rtx base = strip_offset (imm, &offset);
if (!offset.is_constant (&const_offset))
{
if (base == const0_rtx && aarch64_sve_cnt_immediate_p (offset))
emit_insn (gen_rtx_SET (dest, imm));
else
{
if (partial_subreg_p (int_mode, SImode))
{
gcc_assert (base == const0_rtx);
dest = gen_lowpart (SImode, dest);
int_mode = SImode;
}
if (base != const0_rtx)
{
base = aarch64_force_temporary (int_mode, dest, base);
aarch64_add_offset (int_mode, dest, base, offset,
NULL_RTX, NULL_RTX, false);
}
else
aarch64_add_offset (int_mode, dest, base, offset,
dest, NULL_RTX, false);
}
return;
}
sty = aarch64_classify_symbol (base, const_offset);
switch (sty)
{
case SYMBOL_FORCE_TO_MEM:
if (const_offset != 0
&& targetm.cannot_force_const_mem (int_mode, imm))
{
gcc_assert (can_create_pseudo_p ());
base = aarch64_force_temporary (int_mode, dest, base);
aarch64_add_offset (int_mode, dest, base, const_offset,
NULL_RTX, NULL_RTX, false);
return;
}
mem = force_const_mem (ptr_mode, imm);
gcc_assert (mem);
if (!aarch64_pcrelative_literal_loads)
{
gcc_assert (can_create_pseudo_p ());
base = gen_reg_rtx (ptr_mode);
aarch64_expand_mov_immediate (base, XEXP (mem, 0));
if (ptr_mode != Pmode)
base = convert_memory_address (Pmode, base);
mem = gen_rtx_MEM (ptr_mode, base);
}
if (int_mode != ptr_mode)
mem = gen_rtx_ZERO_EXTEND (int_mode, mem);
emit_insn (gen_rtx_SET (dest, mem));
return;
case SYMBOL_SMALL_TLSGD:
case SYMBOL_SMALL_TLSDESC:
case SYMBOL_SMALL_TLSIE:
case SYMBOL_SMALL_GOT_28K:
case SYMBOL_SMALL_GOT_4G:
case SYMBOL_TINY_GOT:
case SYMBOL_TINY_TLSIE:
if (const_offset != 0)
{
gcc_assert(can_create_pseudo_p ());
base = aarch64_force_temporary (int_mode, dest, base);
aarch64_add_offset (int_mode, dest, base, const_offset,
NULL_RTX, NULL_RTX, false);
return;
}
case SYMBOL_SMALL_ABSOLUTE:
case SYMBOL_TINY_ABSOLUTE:
case SYMBOL_TLSLE12:
case SYMBOL_TLSLE24:
case SYMBOL_TLSLE32:
case SYMBOL_TLSLE48:
aarch64_load_symref_appropriately (dest, imm, sty);
return;
default:
gcc_unreachable ();
}
}
if (!CONST_INT_P (imm))
{
rtx base, step, value;
if (GET_CODE (imm) == HIGH
|| aarch64_simd_valid_immediate (imm, NULL))
emit_insn (gen_rtx_SET (dest, imm));
else if (const_vec_series_p (imm, &base, &step))
aarch64_expand_vec_series (dest, base, step);
else if (const_vec_duplicate_p (imm, &value))
{
scalar_mode inner_mode = GET_MODE_INNER (mode);
rtx op = force_const_mem (inner_mode, value);
if (!op)
op = force_reg (inner_mode, value);
else if (!aarch64_sve_ld1r_operand_p (op))
{
rtx addr = force_reg (Pmode, XEXP (op, 0));
op = replace_equiv_address (op, addr);
}
emit_insn (gen_vec_duplicate (dest, op));
}
else if (GET_CODE (imm) == CONST_VECTOR
&& !GET_MODE_NUNITS (GET_MODE (imm)).is_constant ())
aarch64_expand_sve_const_vector (dest, imm);
else
{
rtx mem = force_const_mem (mode, imm);
gcc_assert (mem);
emit_move_insn (dest, mem);
}
return;
}
aarch64_internal_mov_immediate (dest, imm, true,
as_a <scalar_int_mode> (mode));
}
void
aarch64_emit_sve_pred_move (rtx dest, rtx pred, rtx src)
{
emit_insn (gen_rtx_SET (dest, gen_rtx_UNSPEC (GET_MODE (dest),
gen_rtvec (2, pred, src),
UNSPEC_MERGE_PTRUE)));
}
void
aarch64_expand_sve_mem_move (rtx dest, rtx src, machine_mode pred_mode)
{
machine_mode mode = GET_MODE (dest);
rtx ptrue = force_reg (pred_mode, CONSTM1_RTX (pred_mode));
if (!register_operand (src, mode)
&& !register_operand (dest, mode))
{
rtx tmp = gen_reg_rtx (mode);
if (MEM_P (src))
aarch64_emit_sve_pred_move (tmp, ptrue, src);
else
emit_move_insn (tmp, src);
src = tmp;
}
aarch64_emit_sve_pred_move (dest, ptrue, src);
}
bool
aarch64_maybe_expand_sve_subreg_move (rtx dest, rtx src)
{
gcc_assert (BYTES_BIG_ENDIAN);
if (GET_CODE (dest) == SUBREG)
dest = SUBREG_REG (dest);
if (GET_CODE (src) == SUBREG)
src = SUBREG_REG (src);
if (!REG_P (dest)
|| !REG_P (src)
|| aarch64_classify_vector_mode (GET_MODE (dest)) != VEC_SVE_DATA
|| aarch64_classify_vector_mode (GET_MODE (src)) != VEC_SVE_DATA
|| (GET_MODE_UNIT_SIZE (GET_MODE (dest))
== GET_MODE_UNIT_SIZE (GET_MODE (src))))
return false;
rtx ptrue = force_reg (VNx16BImode, CONSTM1_RTX (VNx16BImode));
rtx unspec = gen_rtx_UNSPEC (GET_MODE (dest), gen_rtvec (2, ptrue, src),
UNSPEC_REV_SUBREG);
emit_insn (gen_rtx_SET (dest, unspec));
return true;
}
static rtx
aarch64_replace_reg_mode (rtx x, machine_mode mode)
{
if (GET_MODE (x) == mode)
return x;
x = shallow_copy_rtx (x);
set_mode_and_regno (x, mode, REGNO (x));
return x;
}
void
aarch64_split_sve_subreg_move (rtx dest, rtx ptrue, rtx src)
{
machine_mode mode_with_wider_elts = GET_MODE (dest);
machine_mode mode_with_narrower_elts = GET_MODE (src);
if (GET_MODE_UNIT_SIZE (mode_with_wider_elts)
< GET_MODE_UNIT_SIZE (mode_with_narrower_elts))
std::swap (mode_with_wider_elts, mode_with_narrower_elts);
unsigned int wider_bytes = GET_MODE_UNIT_SIZE (mode_with_wider_elts);
unsigned int unspec;
if (wider_bytes == 8)
unspec = UNSPEC_REV64;
else if (wider_bytes == 4)
unspec = UNSPEC_REV32;
else if (wider_bytes == 2)
unspec = UNSPEC_REV16;
else
gcc_unreachable ();
machine_mode pred_mode = aarch64_sve_pred_mode (wider_bytes).require ();
ptrue = gen_lowpart (pred_mode, ptrue);
dest = aarch64_replace_reg_mode (dest, mode_with_narrower_elts);
src = aarch64_replace_reg_mode (src, mode_with_narrower_elts);
src = gen_rtx_UNSPEC (mode_with_narrower_elts, gen_rtvec (1, src), unspec);
src = gen_rtx_UNSPEC (mode_with_narrower_elts, gen_rtvec (2, ptrue, src),
UNSPEC_MERGE_PTRUE);
emit_insn (gen_rtx_SET (dest, src));
}
static bool
aarch64_function_ok_for_sibcall (tree decl ATTRIBUTE_UNUSED,
tree exp ATTRIBUTE_UNUSED)
{
return true;
}
static bool
aarch64_pass_by_reference (cumulative_args_t pcum ATTRIBUTE_UNUSED,
machine_mode mode,
const_tree type,
bool named ATTRIBUTE_UNUSED)
{
HOST_WIDE_INT size;
machine_mode dummymode;
int nregs;
if (mode == BLKmode && type)
size = int_size_in_bytes (type);
else
size = GET_MODE_SIZE (mode).to_constant ();
if (type && AGGREGATE_TYPE_P (type))
{
size = int_size_in_bytes (type);
}
if (size < 0)
return true;
if (aarch64_vfp_is_call_or_return_candidate (mode, type,
&dummymode, &nregs,
NULL))
return false;
return size > 2 * UNITS_PER_WORD;
}
static bool
aarch64_return_in_msb (const_tree valtype)
{
machine_mode dummy_mode;
int dummy_int;
if (!BYTES_BIG_ENDIAN)
return false;
if (!aarch64_composite_type_p (valtype, TYPE_MODE (valtype))
|| int_size_in_bytes (valtype) <= 0
|| int_size_in_bytes (valtype) > 16)
return false;
if (aarch64_vfp_is_call_or_return_candidate (TYPE_MODE (valtype), valtype,
&dummy_mode, &dummy_int, NULL))
return false;
return true;
}
static rtx
aarch64_function_value (const_tree type, const_tree func,
bool outgoing ATTRIBUTE_UNUSED)
{
machine_mode mode;
int unsignedp;
int count;
machine_mode ag_mode;
mode = TYPE_MODE (type);
if (INTEGRAL_TYPE_P (type))
mode = promote_function_mode (type, mode, &unsignedp, func, 1);
if (aarch64_return_in_msb (type))
{
HOST_WIDE_INT size = int_size_in_bytes (type);
if (size % UNITS_PER_WORD != 0)
{
size += UNITS_PER_WORD - size % UNITS_PER_WORD;
mode = int_mode_for_size (size * BITS_PER_UNIT, 0).require ();
}
}
if (aarch64_vfp_is_call_or_return_candidate (mode, type,
&ag_mode, &count, NULL))
{
if (!aarch64_composite_type_p (type, mode))
{
gcc_assert (count == 1 && mode == ag_mode);
return gen_rtx_REG (mode, V0_REGNUM);
}
else
{
int i;
rtx par;
par = gen_rtx_PARALLEL (mode, rtvec_alloc (count));
for (i = 0; i < count; i++)
{
rtx tmp = gen_rtx_REG (ag_mode, V0_REGNUM + i);
rtx offset = gen_int_mode (i * GET_MODE_SIZE (ag_mode), Pmode);
tmp = gen_rtx_EXPR_LIST (VOIDmode, tmp, offset);
XVECEXP (par, 0, i) = tmp;
}
return par;
}
}
else
return gen_rtx_REG (mode, R0_REGNUM);
}
static bool
aarch64_function_value_regno_p (const unsigned int regno)
{
if (regno == R0_REGNUM || regno == R1_REGNUM)
return true;
if (regno >= V0_REGNUM && regno < V0_REGNUM + HA_MAX_NUM_FLDS)
return TARGET_FLOAT;
return false;
}
static bool
aarch64_return_in_memory (const_tree type, const_tree fndecl ATTRIBUTE_UNUSED)
{
HOST_WIDE_INT size;
machine_mode ag_mode;
int count;
if (!AGGREGATE_TYPE_P (type)
&& TREE_CODE (type) != COMPLEX_TYPE
&& TREE_CODE (type) != VECTOR_TYPE)
return false;
if (aarch64_vfp_is_call_or_return_candidate (TYPE_MODE (type),
type,
&ag_mode,
&count,
NULL))
return false;
size = int_size_in_bytes (type);
return (size < 0 || size > 2 * UNITS_PER_WORD);
}
static bool
aarch64_vfp_is_call_candidate (cumulative_args_t pcum_v, machine_mode mode,
const_tree type, int *nregs)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
return aarch64_vfp_is_call_or_return_candidate (mode,
type,
&pcum->aapcs_vfp_rmode,
nregs,
NULL);
}
static unsigned int
aarch64_function_arg_alignment (machine_mode mode, const_tree type)
{
if (!type)
return GET_MODE_ALIGNMENT (mode);
if (integer_zerop (TYPE_SIZE (type)))
return 0;
gcc_assert (TYPE_MODE (type) == mode);
if (!AGGREGATE_TYPE_P (type))
return TYPE_ALIGN (TYPE_MAIN_VARIANT (type));
if (TREE_CODE (type) == ARRAY_TYPE)
return TYPE_ALIGN (TREE_TYPE (type));
unsigned int alignment = 0;
for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
alignment = std::max (alignment, DECL_ALIGN (field));
return alignment;
}
static void
aarch64_layout_arg (cumulative_args_t pcum_v, machine_mode mode,
const_tree type,
bool named ATTRIBUTE_UNUSED)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
int ncrn, nvrn, nregs;
bool allocate_ncrn, allocate_nvrn;
HOST_WIDE_INT size;
if (pcum->aapcs_arg_processed)
return;
pcum->aapcs_arg_processed = true;
if (type)
size = int_size_in_bytes (type);
else
size = GET_MODE_SIZE (mode).to_constant ();
size = ROUND_UP (size, UNITS_PER_WORD);
allocate_ncrn = (type) ? !(FLOAT_TYPE_P (type)) : !FLOAT_MODE_P (mode);
allocate_nvrn = aarch64_vfp_is_call_candidate (pcum_v,
mode,
type,
&nregs);
nvrn = pcum->aapcs_nvrn;
if (allocate_nvrn)
{
if (!TARGET_FLOAT)
aarch64_err_no_fpadvsimd (mode, "argument");
if (nvrn + nregs <= NUM_FP_ARG_REGS)
{
pcum->aapcs_nextnvrn = nvrn + nregs;
if (!aarch64_composite_type_p (type, mode))
{
gcc_assert (nregs == 1);
pcum->aapcs_reg = gen_rtx_REG (mode, V0_REGNUM + nvrn);
}
else
{
rtx par;
int i;
par = gen_rtx_PARALLEL (mode, rtvec_alloc (nregs));
for (i = 0; i < nregs; i++)
{
rtx tmp = gen_rtx_REG (pcum->aapcs_vfp_rmode,
V0_REGNUM + nvrn + i);
rtx offset = gen_int_mode
(i * GET_MODE_SIZE (pcum->aapcs_vfp_rmode), Pmode);
tmp = gen_rtx_EXPR_LIST (VOIDmode, tmp, offset);
XVECEXP (par, 0, i) = tmp;
}
pcum->aapcs_reg = par;
}
return;
}
else
{
pcum->aapcs_nextnvrn = NUM_FP_ARG_REGS;
goto on_stack;
}
}
ncrn = pcum->aapcs_ncrn;
nregs = size / UNITS_PER_WORD;
if (allocate_ncrn && (ncrn + nregs <= NUM_ARG_REGS))
{
gcc_assert (nregs == 0 || nregs == 1 || nregs == 2);
if (nregs == 2
&& ncrn % 2
&& aarch64_function_arg_alignment (mode, type) == 16 * BITS_PER_UNIT)
{
++ncrn;
gcc_assert (ncrn + nregs <= NUM_ARG_REGS);
}
if (nregs == 0 || nregs == 1 || GET_MODE_CLASS (mode) == MODE_INT)
pcum->aapcs_reg = gen_rtx_REG (mode, R0_REGNUM + ncrn);
else
{
rtx par;
int i;
par = gen_rtx_PARALLEL (mode, rtvec_alloc (nregs));
for (i = 0; i < nregs; i++)
{
rtx tmp = gen_rtx_REG (word_mode, R0_REGNUM + ncrn + i);
tmp = gen_rtx_EXPR_LIST (VOIDmode, tmp,
GEN_INT (i * UNITS_PER_WORD));
XVECEXP (par, 0, i) = tmp;
}
pcum->aapcs_reg = par;
}
pcum->aapcs_nextncrn = ncrn + nregs;
return;
}
pcum->aapcs_nextncrn = NUM_ARG_REGS;
on_stack:
pcum->aapcs_stack_words = size / UNITS_PER_WORD;
if (aarch64_function_arg_alignment (mode, type) == 16 * BITS_PER_UNIT)
pcum->aapcs_stack_size = ROUND_UP (pcum->aapcs_stack_size,
16 / UNITS_PER_WORD);
return;
}
static rtx
aarch64_function_arg (cumulative_args_t pcum_v, machine_mode mode,
const_tree type, bool named)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
gcc_assert (pcum->pcs_variant == ARM_PCS_AAPCS64);
if (mode == VOIDmode)
return NULL_RTX;
aarch64_layout_arg (pcum_v, mode, type, named);
return pcum->aapcs_reg;
}
void
aarch64_init_cumulative_args (CUMULATIVE_ARGS *pcum,
const_tree fntype ATTRIBUTE_UNUSED,
rtx libname ATTRIBUTE_UNUSED,
const_tree fndecl ATTRIBUTE_UNUSED,
unsigned n_named ATTRIBUTE_UNUSED)
{
pcum->aapcs_ncrn = 0;
pcum->aapcs_nvrn = 0;
pcum->aapcs_nextncrn = 0;
pcum->aapcs_nextnvrn = 0;
pcum->pcs_variant = ARM_PCS_AAPCS64;
pcum->aapcs_reg = NULL_RTX;
pcum->aapcs_arg_processed = false;
pcum->aapcs_stack_words = 0;
pcum->aapcs_stack_size = 0;
if (!TARGET_FLOAT
&& fndecl && TREE_PUBLIC (fndecl)
&& fntype && fntype != error_mark_node)
{
const_tree type = TREE_TYPE (fntype);
machine_mode mode ATTRIBUTE_UNUSED; 
int nregs ATTRIBUTE_UNUSED; 
if (aarch64_vfp_is_call_or_return_candidate (TYPE_MODE (type), type,
&mode, &nregs, NULL))
aarch64_err_no_fpadvsimd (TYPE_MODE (type), "return type");
}
return;
}
static void
aarch64_function_arg_advance (cumulative_args_t pcum_v,
machine_mode mode,
const_tree type,
bool named)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
if (pcum->pcs_variant == ARM_PCS_AAPCS64)
{
aarch64_layout_arg (pcum_v, mode, type, named);
gcc_assert ((pcum->aapcs_reg != NULL_RTX)
!= (pcum->aapcs_stack_words != 0));
pcum->aapcs_arg_processed = false;
pcum->aapcs_ncrn = pcum->aapcs_nextncrn;
pcum->aapcs_nvrn = pcum->aapcs_nextnvrn;
pcum->aapcs_stack_size += pcum->aapcs_stack_words;
pcum->aapcs_stack_words = 0;
pcum->aapcs_reg = NULL_RTX;
}
}
bool
aarch64_function_arg_regno_p (unsigned regno)
{
return ((GP_REGNUM_P (regno) && regno < R0_REGNUM + NUM_ARG_REGS)
|| (FP_REGNUM_P (regno) && regno < V0_REGNUM + NUM_FP_ARG_REGS));
}
static unsigned int
aarch64_function_arg_boundary (machine_mode mode, const_tree type)
{
unsigned int alignment = aarch64_function_arg_alignment (mode, type);
return MIN (MAX (alignment, PARM_BOUNDARY), STACK_BOUNDARY);
}
static fixed_size_mode
aarch64_get_reg_raw_mode (int regno)
{
if (TARGET_SVE && FP_REGNUM_P (regno))
return as_a <fixed_size_mode> (V16QImode);
return default_get_reg_raw_mode (regno);
}
static pad_direction
aarch64_function_arg_padding (machine_mode mode, const_tree type)
{
if (!BYTES_BIG_ENDIAN)
return PAD_UPWARD;
if (type
? (INTEGRAL_TYPE_P (type) || SCALAR_FLOAT_TYPE_P (type)
|| POINTER_TYPE_P (type))
: (SCALAR_INT_MODE_P (mode) || SCALAR_FLOAT_MODE_P (mode)))
return PAD_DOWNWARD;
return PAD_UPWARD;
}
bool
aarch64_pad_reg_upward (machine_mode mode, const_tree type,
bool first ATTRIBUTE_UNUSED)
{
if (BYTES_BIG_ENDIAN && aarch64_composite_type_p (type, mode))
{
HOST_WIDE_INT size;
if (type)
size = int_size_in_bytes (type);
else
size = GET_MODE_SIZE (mode).to_constant ();
if (size < 2 * UNITS_PER_WORD)
return true;
}
return !BYTES_BIG_ENDIAN;
}
static scalar_int_mode
aarch64_libgcc_cmp_return_mode (void)
{
return SImode;
}
#define PROBE_INTERVAL (1 << STACK_CHECK_PROBE_INTERVAL_EXP)
#define ARITH_FACTOR 4096
#if (PROBE_INTERVAL % ARITH_FACTOR) != 0
#error Cannot use simple address calculation for stack probing
#endif
#define PROBE_STACK_FIRST_REG  9
#define PROBE_STACK_SECOND_REG 10
static void
aarch64_emit_probe_stack_range (HOST_WIDE_INT first, poly_int64 poly_size)
{
HOST_WIDE_INT size;
if (!poly_size.is_constant (&size))
{
sorry ("stack probes for SVE frames");
return;
}
rtx reg1 = gen_rtx_REG (Pmode, PROBE_STACK_FIRST_REG);
gcc_assert ((first % ARITH_FACTOR) == 0);
if (size <= PROBE_INTERVAL)
{
const HOST_WIDE_INT base = ROUND_UP (size, ARITH_FACTOR);
emit_set_insn (reg1,
plus_constant (Pmode,
stack_pointer_rtx, -(first + base)));
emit_stack_probe (plus_constant (Pmode, reg1, base - size));
}
else if (size <= 4 * PROBE_INTERVAL)
{
HOST_WIDE_INT i, rem;
emit_set_insn (reg1,
plus_constant (Pmode,
stack_pointer_rtx,
-(first + PROBE_INTERVAL)));
emit_stack_probe (reg1);
for (i = 2 * PROBE_INTERVAL; i < size; i += PROBE_INTERVAL)
{
emit_set_insn (reg1,
plus_constant (Pmode, reg1, -PROBE_INTERVAL));
emit_stack_probe (reg1);
}
rem = size - (i - PROBE_INTERVAL);
if (rem > 256)
{
const HOST_WIDE_INT base = ROUND_UP (rem, ARITH_FACTOR);
emit_set_insn (reg1, plus_constant (Pmode, reg1, -base));
emit_stack_probe (plus_constant (Pmode, reg1, base - rem));
}
else
emit_stack_probe (plus_constant (Pmode, reg1, -rem));
}
else
{
rtx reg2 = gen_rtx_REG (Pmode, PROBE_STACK_SECOND_REG);
HOST_WIDE_INT rounded_size = size & -PROBE_INTERVAL;
emit_set_insn (reg1,
plus_constant (Pmode, stack_pointer_rtx, -first));
HOST_WIDE_INT adjustment = - (first + rounded_size);
if (! aarch64_uimm12_shift (adjustment))
{
aarch64_internal_mov_immediate (reg2, GEN_INT (adjustment),
true, Pmode);
emit_set_insn (reg2, gen_rtx_PLUS (Pmode, stack_pointer_rtx, reg2));
}
else
emit_set_insn (reg2,
plus_constant (Pmode, stack_pointer_rtx, adjustment));
emit_insn (gen_probe_stack_range (reg1, reg1, reg2));
if (size != rounded_size)
{
HOST_WIDE_INT rem = size - rounded_size;
if (rem > 256)
{
const HOST_WIDE_INT base = ROUND_UP (rem, ARITH_FACTOR);
emit_set_insn (reg2, plus_constant (Pmode, reg2, -base));
emit_stack_probe (plus_constant (Pmode, reg2, base - rem));
}
else
emit_stack_probe (plus_constant (Pmode, reg2, -rem));
}
}
emit_insn (gen_blockage ());
}
const char *
aarch64_output_probe_stack_range (rtx reg1, rtx reg2)
{
static int labelno = 0;
char loop_lab[32];
rtx xops[2];
ASM_GENERATE_INTERNAL_LABEL (loop_lab, "LPSRL", labelno++);
ASM_OUTPUT_INTERNAL_LABEL (asm_out_file, loop_lab);
xops[0] = reg1;
xops[1] = GEN_INT (PROBE_INTERVAL);
output_asm_insn ("sub\t%0, %0, %1", xops);
output_asm_insn ("str\txzr, [%0]", xops);
xops[1] = reg2;
output_asm_insn ("cmp\t%0, %1", xops);
fputs ("\tb.ne\t", asm_out_file);
assemble_name_raw (asm_out_file, loop_lab);
fputc ('\n', asm_out_file);
return "";
}
static void
aarch64_layout_frame (void)
{
HOST_WIDE_INT offset = 0;
int regno, last_fp_reg = INVALID_REGNUM;
if (reload_completed && cfun->machine->frame.laid_out)
return;
cfun->machine->frame.emit_frame_chain
= frame_pointer_needed || crtl->calls_eh_return;
if (flag_omit_frame_pointer == 2
&& !(flag_omit_leaf_frame_pointer && crtl->is_leaf
&& !df_regs_ever_live_p (LR_REGNUM)))
cfun->machine->frame.emit_frame_chain = true;
#define SLOT_NOT_REQUIRED (-2)
#define SLOT_REQUIRED     (-1)
cfun->machine->frame.wb_candidate1 = INVALID_REGNUM;
cfun->machine->frame.wb_candidate2 = INVALID_REGNUM;
for (regno = R0_REGNUM; regno <= R30_REGNUM; regno++)
cfun->machine->frame.reg_offset[regno] = SLOT_NOT_REQUIRED;
for (regno = V0_REGNUM; regno <= V31_REGNUM; regno++)
cfun->machine->frame.reg_offset[regno] = SLOT_NOT_REQUIRED;
if (crtl->calls_eh_return)
for (regno = 0; EH_RETURN_DATA_REGNO (regno) != INVALID_REGNUM; regno++)
cfun->machine->frame.reg_offset[EH_RETURN_DATA_REGNO (regno)]
= SLOT_REQUIRED;
for (regno = R0_REGNUM; regno <= R30_REGNUM; regno++)
if (df_regs_ever_live_p (regno)
&& (regno == R30_REGNUM
|| !call_used_regs[regno]))
cfun->machine->frame.reg_offset[regno] = SLOT_REQUIRED;
for (regno = V0_REGNUM; regno <= V31_REGNUM; regno++)
if (df_regs_ever_live_p (regno)
&& !call_used_regs[regno])
{
cfun->machine->frame.reg_offset[regno] = SLOT_REQUIRED;
last_fp_reg = regno;
}
if (cfun->machine->frame.emit_frame_chain)
{
cfun->machine->frame.reg_offset[R29_REGNUM] = 0;
cfun->machine->frame.wb_candidate1 = R29_REGNUM;
cfun->machine->frame.reg_offset[R30_REGNUM] = UNITS_PER_WORD;
cfun->machine->frame.wb_candidate2 = R30_REGNUM;
offset = 2 * UNITS_PER_WORD;
}
for (regno = R0_REGNUM; regno <= R30_REGNUM; regno++)
if (cfun->machine->frame.reg_offset[regno] == SLOT_REQUIRED)
{
cfun->machine->frame.reg_offset[regno] = offset;
if (cfun->machine->frame.wb_candidate1 == INVALID_REGNUM)
cfun->machine->frame.wb_candidate1 = regno;
else if (cfun->machine->frame.wb_candidate2 == INVALID_REGNUM)
cfun->machine->frame.wb_candidate2 = regno;
offset += UNITS_PER_WORD;
}
HOST_WIDE_INT max_int_offset = offset;
offset = ROUND_UP (offset, STACK_BOUNDARY / BITS_PER_UNIT);
bool has_align_gap = offset != max_int_offset;
for (regno = V0_REGNUM; regno <= V31_REGNUM; regno++)
if (cfun->machine->frame.reg_offset[regno] == SLOT_REQUIRED)
{
if (regno == last_fp_reg && has_align_gap && (offset & 8) == 0)
{
cfun->machine->frame.reg_offset[regno] = max_int_offset;
break;
}
cfun->machine->frame.reg_offset[regno] = offset;
if (cfun->machine->frame.wb_candidate1 == INVALID_REGNUM)
cfun->machine->frame.wb_candidate1 = regno;
else if (cfun->machine->frame.wb_candidate2 == INVALID_REGNUM
&& cfun->machine->frame.wb_candidate1 >= V0_REGNUM)
cfun->machine->frame.wb_candidate2 = regno;
offset += UNITS_PER_WORD;
}
offset = ROUND_UP (offset, STACK_BOUNDARY / BITS_PER_UNIT);
cfun->machine->frame.saved_regs_size = offset;
HOST_WIDE_INT varargs_and_saved_regs_size
= offset + cfun->machine->frame.saved_varargs_size;
cfun->machine->frame.hard_fp_offset
= aligned_upper_bound (varargs_and_saved_regs_size
+ get_frame_size (),
STACK_BOUNDARY / BITS_PER_UNIT);
gcc_assert (multiple_p (crtl->outgoing_args_size,
STACK_BOUNDARY / BITS_PER_UNIT));
cfun->machine->frame.frame_size
= (cfun->machine->frame.hard_fp_offset
+ crtl->outgoing_args_size);
cfun->machine->frame.locals_offset = cfun->machine->frame.saved_varargs_size;
cfun->machine->frame.initial_adjust = 0;
cfun->machine->frame.final_adjust = 0;
cfun->machine->frame.callee_adjust = 0;
cfun->machine->frame.callee_offset = 0;
HOST_WIDE_INT max_push_offset = 0;
if (cfun->machine->frame.wb_candidate2 != INVALID_REGNUM)
max_push_offset = 512;
else if (cfun->machine->frame.wb_candidate1 != INVALID_REGNUM)
max_push_offset = 256;
HOST_WIDE_INT const_size, const_fp_offset;
if (cfun->machine->frame.frame_size.is_constant (&const_size)
&& const_size < max_push_offset
&& known_eq (crtl->outgoing_args_size, 0))
{
cfun->machine->frame.callee_adjust = const_size;
}
else if (known_lt (crtl->outgoing_args_size
+ cfun->machine->frame.saved_regs_size, 512)
&& !(cfun->calls_alloca
&& known_lt (cfun->machine->frame.hard_fp_offset,
max_push_offset)))
{
cfun->machine->frame.initial_adjust = cfun->machine->frame.frame_size;
cfun->machine->frame.callee_offset
= cfun->machine->frame.frame_size - cfun->machine->frame.hard_fp_offset;
}
else if (cfun->machine->frame.hard_fp_offset.is_constant (&const_fp_offset)
&& const_fp_offset < max_push_offset)
{
cfun->machine->frame.callee_adjust = const_fp_offset;
cfun->machine->frame.final_adjust
= cfun->machine->frame.frame_size - cfun->machine->frame.callee_adjust;
}
else
{
cfun->machine->frame.initial_adjust = cfun->machine->frame.hard_fp_offset;
cfun->machine->frame.final_adjust
= cfun->machine->frame.frame_size - cfun->machine->frame.initial_adjust;
}
cfun->machine->frame.laid_out = true;
}
static bool
aarch64_register_saved_on_entry (int regno)
{
return cfun->machine->frame.reg_offset[regno] >= 0;
}
static unsigned
aarch64_next_callee_save (unsigned regno, unsigned limit)
{
while (regno <= limit && !aarch64_register_saved_on_entry (regno))
regno ++;
return regno;
}
static void
aarch64_pushwb_single_reg (machine_mode mode, unsigned regno,
HOST_WIDE_INT adjustment)
{
rtx base_rtx = stack_pointer_rtx;
rtx insn, reg, mem;
reg = gen_rtx_REG (mode, regno);
mem = gen_rtx_PRE_MODIFY (Pmode, base_rtx,
plus_constant (Pmode, base_rtx, -adjustment));
mem = gen_frame_mem (mode, mem);
insn = emit_move_insn (mem, reg);
RTX_FRAME_RELATED_P (insn) = 1;
}
static rtx
aarch64_gen_storewb_pair (machine_mode mode, rtx base, rtx reg, rtx reg2,
HOST_WIDE_INT adjustment)
{
switch (mode)
{
case E_DImode:
return gen_storewb_pairdi_di (base, base, reg, reg2,
GEN_INT (-adjustment),
GEN_INT (UNITS_PER_WORD - adjustment));
case E_DFmode:
return gen_storewb_pairdf_di (base, base, reg, reg2,
GEN_INT (-adjustment),
GEN_INT (UNITS_PER_WORD - adjustment));
default:
gcc_unreachable ();
}
}
static void
aarch64_push_regs (unsigned regno1, unsigned regno2, HOST_WIDE_INT adjustment)
{
rtx_insn *insn;
machine_mode mode = (regno1 <= R30_REGNUM) ? E_DImode : E_DFmode;
if (regno2 == INVALID_REGNUM)
return aarch64_pushwb_single_reg (mode, regno1, adjustment);
rtx reg1 = gen_rtx_REG (mode, regno1);
rtx reg2 = gen_rtx_REG (mode, regno2);
insn = emit_insn (aarch64_gen_storewb_pair (mode, stack_pointer_rtx, reg1,
reg2, adjustment));
RTX_FRAME_RELATED_P (XVECEXP (PATTERN (insn), 0, 2)) = 1;
RTX_FRAME_RELATED_P (XVECEXP (PATTERN (insn), 0, 1)) = 1;
RTX_FRAME_RELATED_P (insn) = 1;
}
static rtx
aarch64_gen_loadwb_pair (machine_mode mode, rtx base, rtx reg, rtx reg2,
HOST_WIDE_INT adjustment)
{
switch (mode)
{
case E_DImode:
return gen_loadwb_pairdi_di (base, base, reg, reg2, GEN_INT (adjustment),
GEN_INT (UNITS_PER_WORD));
case E_DFmode:
return gen_loadwb_pairdf_di (base, base, reg, reg2, GEN_INT (adjustment),
GEN_INT (UNITS_PER_WORD));
default:
gcc_unreachable ();
}
}
static void
aarch64_pop_regs (unsigned regno1, unsigned regno2, HOST_WIDE_INT adjustment,
rtx *cfi_ops)
{
machine_mode mode = (regno1 <= R30_REGNUM) ? E_DImode : E_DFmode;
rtx reg1 = gen_rtx_REG (mode, regno1);
*cfi_ops = alloc_reg_note (REG_CFA_RESTORE, reg1, *cfi_ops);
if (regno2 == INVALID_REGNUM)
{
rtx mem = plus_constant (Pmode, stack_pointer_rtx, adjustment);
mem = gen_rtx_POST_MODIFY (Pmode, stack_pointer_rtx, mem);
emit_move_insn (reg1, gen_frame_mem (mode, mem));
}
else
{
rtx reg2 = gen_rtx_REG (mode, regno2);
*cfi_ops = alloc_reg_note (REG_CFA_RESTORE, reg2, *cfi_ops);
emit_insn (aarch64_gen_loadwb_pair (mode, stack_pointer_rtx, reg1,
reg2, adjustment));
}
}
static rtx
aarch64_gen_store_pair (machine_mode mode, rtx mem1, rtx reg1, rtx mem2,
rtx reg2)
{
switch (mode)
{
case E_DImode:
return gen_store_pairdi (mem1, reg1, mem2, reg2);
case E_DFmode:
return gen_store_pairdf (mem1, reg1, mem2, reg2);
default:
gcc_unreachable ();
}
}
static rtx
aarch64_gen_load_pair (machine_mode mode, rtx reg1, rtx mem1, rtx reg2,
rtx mem2)
{
switch (mode)
{
case E_DImode:
return gen_load_pairdi (reg1, mem1, reg2, mem2);
case E_DFmode:
return gen_load_pairdf (reg1, mem1, reg2, mem2);
default:
gcc_unreachable ();
}
}
bool
aarch64_return_address_signing_enabled (void)
{
gcc_assert (cfun->machine->frame.laid_out);
return (aarch64_ra_sign_scope == AARCH64_FUNCTION_ALL
|| (aarch64_ra_sign_scope == AARCH64_FUNCTION_NON_LEAF
&& cfun->machine->frame.reg_offset[LR_REGNUM] >= 0));
}
static void
aarch64_save_callee_saves (machine_mode mode, poly_int64 start_offset,
unsigned start, unsigned limit, bool skip_wb)
{
rtx_insn *insn;
unsigned regno;
unsigned regno2;
for (regno = aarch64_next_callee_save (start, limit);
regno <= limit;
regno = aarch64_next_callee_save (regno + 1, limit))
{
rtx reg, mem;
poly_int64 offset;
if (skip_wb
&& (regno == cfun->machine->frame.wb_candidate1
|| regno == cfun->machine->frame.wb_candidate2))
continue;
if (cfun->machine->reg_is_wrapped_separately[regno])
continue;
reg = gen_rtx_REG (mode, regno);
offset = start_offset + cfun->machine->frame.reg_offset[regno];
mem = gen_frame_mem (mode, plus_constant (Pmode, stack_pointer_rtx,
offset));
regno2 = aarch64_next_callee_save (regno + 1, limit);
if (regno2 <= limit
&& !cfun->machine->reg_is_wrapped_separately[regno2]
&& ((cfun->machine->frame.reg_offset[regno] + UNITS_PER_WORD)
== cfun->machine->frame.reg_offset[regno2]))
{
rtx reg2 = gen_rtx_REG (mode, regno2);
rtx mem2;
offset = start_offset + cfun->machine->frame.reg_offset[regno2];
mem2 = gen_frame_mem (mode, plus_constant (Pmode, stack_pointer_rtx,
offset));
insn = emit_insn (aarch64_gen_store_pair (mode, mem, reg, mem2,
reg2));
RTX_FRAME_RELATED_P (XVECEXP (PATTERN (insn), 0, 1)) = 1;
regno = regno2;
}
else
insn = emit_move_insn (mem, reg);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
static void
aarch64_restore_callee_saves (machine_mode mode,
poly_int64 start_offset, unsigned start,
unsigned limit, bool skip_wb, rtx *cfi_ops)
{
rtx base_rtx = stack_pointer_rtx;
unsigned regno;
unsigned regno2;
poly_int64 offset;
for (regno = aarch64_next_callee_save (start, limit);
regno <= limit;
regno = aarch64_next_callee_save (regno + 1, limit))
{
if (cfun->machine->reg_is_wrapped_separately[regno])
continue;
rtx reg, mem;
if (skip_wb
&& (regno == cfun->machine->frame.wb_candidate1
|| regno == cfun->machine->frame.wb_candidate2))
continue;
reg = gen_rtx_REG (mode, regno);
offset = start_offset + cfun->machine->frame.reg_offset[regno];
mem = gen_frame_mem (mode, plus_constant (Pmode, base_rtx, offset));
regno2 = aarch64_next_callee_save (regno + 1, limit);
if (regno2 <= limit
&& !cfun->machine->reg_is_wrapped_separately[regno2]
&& ((cfun->machine->frame.reg_offset[regno] + UNITS_PER_WORD)
== cfun->machine->frame.reg_offset[regno2]))
{
rtx reg2 = gen_rtx_REG (mode, regno2);
rtx mem2;
offset = start_offset + cfun->machine->frame.reg_offset[regno2];
mem2 = gen_frame_mem (mode, plus_constant (Pmode, base_rtx, offset));
emit_insn (aarch64_gen_load_pair (mode, reg, mem, reg2, mem2));
*cfi_ops = alloc_reg_note (REG_CFA_RESTORE, reg2, *cfi_ops);
regno = regno2;
}
else
emit_move_insn (reg, mem);
*cfi_ops = alloc_reg_note (REG_CFA_RESTORE, reg, *cfi_ops);
}
}
static inline bool
offset_4bit_signed_scaled_p (machine_mode mode, poly_int64 offset)
{
HOST_WIDE_INT multiple;
return (constant_multiple_p (offset, GET_MODE_SIZE (mode), &multiple)
&& IN_RANGE (multiple, -8, 7));
}
static inline bool
offset_6bit_unsigned_scaled_p (machine_mode mode, poly_int64 offset)
{
HOST_WIDE_INT multiple;
return (constant_multiple_p (offset, GET_MODE_SIZE (mode), &multiple)
&& IN_RANGE (multiple, 0, 63));
}
bool
aarch64_offset_7bit_signed_scaled_p (machine_mode mode, poly_int64 offset)
{
HOST_WIDE_INT multiple;
return (constant_multiple_p (offset, GET_MODE_SIZE (mode), &multiple)
&& IN_RANGE (multiple, -64, 63));
}
static inline bool
offset_9bit_signed_unscaled_p (machine_mode mode ATTRIBUTE_UNUSED,
poly_int64 offset)
{
HOST_WIDE_INT const_offset;
return (offset.is_constant (&const_offset)
&& IN_RANGE (const_offset, -256, 255));
}
static inline bool
offset_9bit_signed_scaled_p (machine_mode mode, poly_int64 offset)
{
HOST_WIDE_INT multiple;
return (constant_multiple_p (offset, GET_MODE_SIZE (mode), &multiple)
&& IN_RANGE (multiple, -256, 255));
}
static inline bool
offset_12bit_unsigned_scaled_p (machine_mode mode, poly_int64 offset)
{
HOST_WIDE_INT multiple;
return (constant_multiple_p (offset, GET_MODE_SIZE (mode), &multiple)
&& IN_RANGE (multiple, 0, 4095));
}
static sbitmap
aarch64_get_separate_components (void)
{
aarch64_layout_frame ();
sbitmap components = sbitmap_alloc (LAST_SAVED_REGNUM + 1);
bitmap_clear (components);
for (unsigned regno = 0; regno <= LAST_SAVED_REGNUM; regno++)
if (aarch64_register_saved_on_entry (regno))
{
poly_int64 offset = cfun->machine->frame.reg_offset[regno];
if (!frame_pointer_needed)
offset += cfun->machine->frame.frame_size
- cfun->machine->frame.hard_fp_offset;
if (offset_12bit_unsigned_scaled_p (DImode, offset))
bitmap_set_bit (components, regno);
}
if (frame_pointer_needed)
bitmap_clear_bit (components, HARD_FRAME_POINTER_REGNUM);
unsigned reg1 = cfun->machine->frame.wb_candidate1;
unsigned reg2 = cfun->machine->frame.wb_candidate2;
if (reg2 != INVALID_REGNUM)
bitmap_clear_bit (components, reg2);
if (reg1 != INVALID_REGNUM)
bitmap_clear_bit (components, reg1);
bitmap_clear_bit (components, LR_REGNUM);
bitmap_clear_bit (components, SP_REGNUM);
return components;
}
static sbitmap
aarch64_components_for_bb (basic_block bb)
{
bitmap in = DF_LIVE_IN (bb);
bitmap gen = &DF_LIVE_BB_INFO (bb)->gen;
bitmap kill = &DF_LIVE_BB_INFO (bb)->kill;
sbitmap components = sbitmap_alloc (LAST_SAVED_REGNUM + 1);
bitmap_clear (components);
for (unsigned regno = 0; regno <= LAST_SAVED_REGNUM; regno++)
if ((!call_used_regs[regno])
&& (bitmap_bit_p (in, regno)
|| bitmap_bit_p (gen, regno)
|| bitmap_bit_p (kill, regno)))
{
unsigned regno2, offset, offset2;
bitmap_set_bit (components, regno);
offset = cfun->machine->frame.reg_offset[regno];
regno2 = ((offset & 8) == 0) ? regno + 1 : regno - 1;
if (regno2 <= LAST_SAVED_REGNUM)
{
offset2 = cfun->machine->frame.reg_offset[regno2];
if ((offset & ~8) == (offset2 & ~8))
bitmap_set_bit (components, regno2);
}
}
return components;
}
static void
aarch64_disqualify_components (sbitmap, edge, sbitmap, bool)
{
}
static unsigned int
aarch64_get_next_set_bit (sbitmap bmp, unsigned int start)
{
unsigned int nbits = SBITMAP_SIZE (bmp);
if (start == nbits)
return start;
gcc_assert (start < nbits);
for (unsigned int i = start; i < nbits; i++)
if (bitmap_bit_p (bmp, i))
return i;
return nbits;
}
static void
aarch64_process_components (sbitmap components, bool prologue_p)
{
rtx ptr_reg = gen_rtx_REG (Pmode, frame_pointer_needed
? HARD_FRAME_POINTER_REGNUM
: STACK_POINTER_REGNUM);
unsigned last_regno = SBITMAP_SIZE (components);
unsigned regno = aarch64_get_next_set_bit (components, R0_REGNUM);
rtx_insn *insn = NULL;
while (regno != last_regno)
{
machine_mode mode = GP_REGNUM_P (regno) ? E_DImode : E_DFmode;
rtx reg = gen_rtx_REG (mode, regno);
poly_int64 offset = cfun->machine->frame.reg_offset[regno];
if (!frame_pointer_needed)
offset += cfun->machine->frame.frame_size
- cfun->machine->frame.hard_fp_offset;
rtx addr = plus_constant (Pmode, ptr_reg, offset);
rtx mem = gen_frame_mem (mode, addr);
rtx set = prologue_p ? gen_rtx_SET (mem, reg) : gen_rtx_SET (reg, mem);
unsigned regno2 = aarch64_get_next_set_bit (components, regno + 1);
if (regno2 == last_regno)
{
insn = emit_insn (set);
RTX_FRAME_RELATED_P (insn) = 1;
if (prologue_p)
add_reg_note (insn, REG_CFA_OFFSET, copy_rtx (set));
else
add_reg_note (insn, REG_CFA_RESTORE, reg);
break;
}
poly_int64 offset2 = cfun->machine->frame.reg_offset[regno2];
if (!satisfies_constraint_Ump (mem)
|| GP_REGNUM_P (regno) != GP_REGNUM_P (regno2)
|| maybe_ne ((offset2 - cfun->machine->frame.reg_offset[regno]),
GET_MODE_SIZE (mode)))
{
insn = emit_insn (set);
RTX_FRAME_RELATED_P (insn) = 1;
if (prologue_p)
add_reg_note (insn, REG_CFA_OFFSET, copy_rtx (set));
else
add_reg_note (insn, REG_CFA_RESTORE, reg);
regno = regno2;
continue;
}
rtx reg2 = gen_rtx_REG (mode, regno2);
if (!frame_pointer_needed)
offset2 += cfun->machine->frame.frame_size
- cfun->machine->frame.hard_fp_offset;
rtx addr2 = plus_constant (Pmode, ptr_reg, offset2);
rtx mem2 = gen_frame_mem (mode, addr2);
rtx set2 = prologue_p ? gen_rtx_SET (mem2, reg2)
: gen_rtx_SET (reg2, mem2);
if (prologue_p)
insn = emit_insn (aarch64_gen_store_pair (mode, mem, reg, mem2, reg2));
else
insn = emit_insn (aarch64_gen_load_pair (mode, reg, mem, reg2, mem2));
RTX_FRAME_RELATED_P (insn) = 1;
if (prologue_p)
{
add_reg_note (insn, REG_CFA_OFFSET, set);
add_reg_note (insn, REG_CFA_OFFSET, set2);
}
else
{
add_reg_note (insn, REG_CFA_RESTORE, reg);
add_reg_note (insn, REG_CFA_RESTORE, reg2);
}
regno = aarch64_get_next_set_bit (components, regno2 + 1);
}
}
static void
aarch64_emit_prologue_components (sbitmap components)
{
aarch64_process_components (components, true);
}
static void
aarch64_emit_epilogue_components (sbitmap components)
{
aarch64_process_components (components, false);
}
static void
aarch64_set_handled_components (sbitmap components)
{
for (unsigned regno = 0; regno <= LAST_SAVED_REGNUM; regno++)
if (bitmap_bit_p (components, regno))
cfun->machine->reg_is_wrapped_separately[regno] = true;
}
static void
aarch64_add_cfa_expression (rtx_insn *insn, unsigned int reg,
rtx base, poly_int64 offset)
{
rtx mem = gen_frame_mem (DImode, plus_constant (Pmode, base, offset));
add_reg_note (insn, REG_CFA_EXPRESSION,
gen_rtx_SET (mem, regno_reg_rtx[reg]));
}
void
aarch64_expand_prologue (void)
{
aarch64_layout_frame ();
poly_int64 frame_size = cfun->machine->frame.frame_size;
poly_int64 initial_adjust = cfun->machine->frame.initial_adjust;
HOST_WIDE_INT callee_adjust = cfun->machine->frame.callee_adjust;
poly_int64 final_adjust = cfun->machine->frame.final_adjust;
poly_int64 callee_offset = cfun->machine->frame.callee_offset;
unsigned reg1 = cfun->machine->frame.wb_candidate1;
unsigned reg2 = cfun->machine->frame.wb_candidate2;
bool emit_frame_chain = cfun->machine->frame.emit_frame_chain;
rtx_insn *insn;
if (aarch64_return_address_signing_enabled ())
{
insn = emit_insn (gen_pacisp ());
add_reg_note (insn, REG_CFA_TOGGLE_RA_MANGLE, const0_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
if (flag_stack_usage_info)
current_function_static_stack_size = constant_lower_bound (frame_size);
if (flag_stack_check == STATIC_BUILTIN_STACK_CHECK)
{
if (crtl->is_leaf && !cfun->calls_alloca)
{
if (maybe_gt (frame_size, PROBE_INTERVAL)
&& maybe_gt (frame_size, get_stack_check_protect ()))
aarch64_emit_probe_stack_range (get_stack_check_protect (),
(frame_size
- get_stack_check_protect ()));
}
else if (maybe_gt (frame_size, 0))
aarch64_emit_probe_stack_range (get_stack_check_protect (), frame_size);
}
rtx ip0_rtx = gen_rtx_REG (Pmode, IP0_REGNUM);
rtx ip1_rtx = gen_rtx_REG (Pmode, IP1_REGNUM);
aarch64_sub_sp (ip0_rtx, ip1_rtx, initial_adjust, true);
if (callee_adjust != 0)
aarch64_push_regs (reg1, reg2, callee_adjust);
if (emit_frame_chain)
{
poly_int64 reg_offset = callee_adjust;
if (callee_adjust == 0)
{
reg1 = R29_REGNUM;
reg2 = R30_REGNUM;
reg_offset = callee_offset;
aarch64_save_callee_saves (DImode, reg_offset, reg1, reg2, false);
}
aarch64_add_offset (Pmode, hard_frame_pointer_rtx,
stack_pointer_rtx, callee_offset,
ip1_rtx, ip0_rtx, frame_pointer_needed);
if (frame_pointer_needed && !frame_size.is_constant ())
{
rtx_insn *insn = get_last_insn ();
gcc_assert (RTX_FRAME_RELATED_P (insn));
if (!find_reg_note (insn, REG_CFA_ADJUST_CFA, NULL_RTX))
{
rtx src = plus_constant (Pmode, stack_pointer_rtx,
callee_offset);
add_reg_note (insn, REG_CFA_ADJUST_CFA,
gen_rtx_SET (hard_frame_pointer_rtx, src));
}
reg_offset -= callee_offset;
aarch64_add_cfa_expression (insn, reg2, hard_frame_pointer_rtx,
reg_offset + UNITS_PER_WORD);
aarch64_add_cfa_expression (insn, reg1, hard_frame_pointer_rtx,
reg_offset);
}
emit_insn (gen_stack_tie (stack_pointer_rtx, hard_frame_pointer_rtx));
}
aarch64_save_callee_saves (DImode, callee_offset, R0_REGNUM, R30_REGNUM,
callee_adjust != 0 || emit_frame_chain);
aarch64_save_callee_saves (DFmode, callee_offset, V0_REGNUM, V31_REGNUM,
callee_adjust != 0 || emit_frame_chain);
aarch64_sub_sp (ip1_rtx, ip0_rtx, final_adjust, !frame_pointer_needed);
}
bool
aarch64_use_return_insn_p (void)
{
if (!reload_completed)
return false;
if (crtl->profile)
return false;
aarch64_layout_frame ();
return known_eq (cfun->machine->frame.frame_size, 0);
}
void
aarch64_expand_epilogue (bool for_sibcall)
{
aarch64_layout_frame ();
poly_int64 initial_adjust = cfun->machine->frame.initial_adjust;
HOST_WIDE_INT callee_adjust = cfun->machine->frame.callee_adjust;
poly_int64 final_adjust = cfun->machine->frame.final_adjust;
poly_int64 callee_offset = cfun->machine->frame.callee_offset;
unsigned reg1 = cfun->machine->frame.wb_candidate1;
unsigned reg2 = cfun->machine->frame.wb_candidate2;
rtx cfi_ops = NULL;
rtx_insn *insn;
bool can_inherit_p = (initial_adjust.is_constant ()
&& final_adjust.is_constant ()
&& !flag_stack_clash_protection);
bool need_barrier_p
= maybe_ne (get_frame_size ()
+ cfun->machine->frame.saved_varargs_size, 0);
if (maybe_gt (final_adjust, crtl->outgoing_args_size)
|| cfun->calls_alloca
|| crtl->calls_eh_return)
{
emit_insn (gen_stack_tie (stack_pointer_rtx, stack_pointer_rtx));
need_barrier_p = false;
}
rtx ip0_rtx = gen_rtx_REG (Pmode, IP0_REGNUM);
rtx ip1_rtx = gen_rtx_REG (Pmode, IP1_REGNUM);
if (frame_pointer_needed
&& (maybe_ne (final_adjust, 0) || cfun->calls_alloca))
aarch64_add_offset (Pmode, stack_pointer_rtx,
hard_frame_pointer_rtx, -callee_offset,
ip1_rtx, ip0_rtx, callee_adjust == 0);
else
aarch64_add_sp (ip1_rtx, ip0_rtx, final_adjust,
!can_inherit_p || df_regs_ever_live_p (IP1_REGNUM));
aarch64_restore_callee_saves (DImode, callee_offset, R0_REGNUM, R30_REGNUM,
callee_adjust != 0, &cfi_ops);
aarch64_restore_callee_saves (DFmode, callee_offset, V0_REGNUM, V31_REGNUM,
callee_adjust != 0, &cfi_ops);
if (need_barrier_p)
emit_insn (gen_stack_tie (stack_pointer_rtx, stack_pointer_rtx));
if (callee_adjust != 0)
aarch64_pop_regs (reg1, reg2, callee_adjust, &cfi_ops);
if (callee_adjust != 0 || maybe_gt (initial_adjust, 65536))
{
insn = get_last_insn ();
rtx new_cfa = plus_constant (Pmode, stack_pointer_rtx, initial_adjust);
REG_NOTES (insn) = alloc_reg_note (REG_CFA_DEF_CFA, new_cfa, cfi_ops);
RTX_FRAME_RELATED_P (insn) = 1;
cfi_ops = NULL;
}
aarch64_add_sp (ip0_rtx, ip1_rtx, initial_adjust,
!can_inherit_p || df_regs_ever_live_p (IP0_REGNUM));
if (cfi_ops)
{
insn = get_last_insn ();
cfi_ops = alloc_reg_note (REG_CFA_DEF_CFA, stack_pointer_rtx, cfi_ops);
REG_NOTES (insn) = cfi_ops;
RTX_FRAME_RELATED_P (insn) = 1;
}
if (aarch64_return_address_signing_enabled ()
&& (for_sibcall || !TARGET_ARMV8_3 || crtl->calls_eh_return))
{
insn = emit_insn (gen_autisp ());
add_reg_note (insn, REG_CFA_TOGGLE_RA_MANGLE, const0_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
if (crtl->calls_eh_return)
{
emit_insn (gen_add2_insn (stack_pointer_rtx, EH_RETURN_STACKADJ_RTX));
}
emit_use (gen_rtx_REG (DImode, LR_REGNUM));
if (!for_sibcall)
emit_jump_insn (ret_rtx);
}
rtx
aarch64_eh_return_handler_rtx (void)
{
rtx tmp = gen_frame_mem (Pmode,
plus_constant (Pmode, hard_frame_pointer_rtx, UNITS_PER_WORD));
MEM_VOLATILE_P (tmp) = true;
return tmp;
}
static void
aarch64_output_mi_thunk (FILE *file, tree thunk ATTRIBUTE_UNUSED,
HOST_WIDE_INT delta,
HOST_WIDE_INT vcall_offset,
tree function)
{
int this_regno = R0_REGNUM;
rtx this_rtx, temp0, temp1, addr, funexp;
rtx_insn *insn;
reload_completed = 1;
emit_note (NOTE_INSN_PROLOGUE_END);
this_rtx = gen_rtx_REG (Pmode, this_regno);
temp0 = gen_rtx_REG (Pmode, IP0_REGNUM);
temp1 = gen_rtx_REG (Pmode, IP1_REGNUM);
if (vcall_offset == 0)
aarch64_add_offset (Pmode, this_rtx, this_rtx, delta, temp1, temp0, false);
else
{
gcc_assert ((vcall_offset & (POINTER_BYTES - 1)) == 0);
addr = this_rtx;
if (delta != 0)
{
if (delta >= -256 && delta < 256)
addr = gen_rtx_PRE_MODIFY (Pmode, this_rtx,
plus_constant (Pmode, this_rtx, delta));
else
aarch64_add_offset (Pmode, this_rtx, this_rtx, delta,
temp1, temp0, false);
}
if (Pmode == ptr_mode)
aarch64_emit_move (temp0, gen_rtx_MEM (ptr_mode, addr));
else
aarch64_emit_move (temp0,
gen_rtx_ZERO_EXTEND (Pmode,
gen_rtx_MEM (ptr_mode, addr)));
if (vcall_offset >= -256 && vcall_offset < 4096 * POINTER_BYTES)
addr = plus_constant (Pmode, temp0, vcall_offset);
else
{
aarch64_internal_mov_immediate (temp1, GEN_INT (vcall_offset), true,
Pmode);
addr = gen_rtx_PLUS (Pmode, temp0, temp1);
}
if (Pmode == ptr_mode)
aarch64_emit_move (temp1, gen_rtx_MEM (ptr_mode,addr));
else
aarch64_emit_move (temp1,
gen_rtx_SIGN_EXTEND (Pmode,
gen_rtx_MEM (ptr_mode, addr)));
emit_insn (gen_add2_insn (this_rtx, temp1));
}
if (!TREE_USED (function))
{
assemble_external (function);
TREE_USED (function) = 1;
}
funexp = XEXP (DECL_RTL (function), 0);
funexp = gen_rtx_MEM (FUNCTION_MODE, funexp);
insn = emit_call_insn (gen_sibcall (funexp, const0_rtx, NULL_RTX));
SIBLING_CALL_P (insn) = 1;
insn = get_insns ();
shorten_branches (insn);
final_start_function (insn, file, 1);
final (insn, file, 1);
final_end_function ();
reload_completed = 0;
}
static bool
aarch64_tls_referenced_p (rtx x)
{
if (!TARGET_HAVE_TLS)
return false;
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, x, ALL)
{
const_rtx x = *iter;
if (GET_CODE (x) == SYMBOL_REF && SYMBOL_REF_TLS_MODEL (x) != 0)
return true;
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
iter.skip_subrtxes ();
}
return false;
}
bool
aarch64_uimm12_shift (HOST_WIDE_INT val)
{
return ((val & (((HOST_WIDE_INT) 0xfff) << 0)) == val
|| (val & (((HOST_WIDE_INT) 0xfff) << 12)) == val
);
}
static bool
aarch64_movw_imm (HOST_WIDE_INT val, scalar_int_mode mode)
{
if (GET_MODE_SIZE (mode) > 4)
{
if ((val & (((HOST_WIDE_INT) 0xffff) << 32)) == val
|| (val & (((HOST_WIDE_INT) 0xffff) << 48)) == val)
return 1;
}
else
{
val &= (HOST_WIDE_INT) 0xffffffff;
}
return ((val & (((HOST_WIDE_INT) 0xffff) << 0)) == val
|| (val & (((HOST_WIDE_INT) 0xffff) << 16)) == val);
}
static unsigned HOST_WIDE_INT
aarch64_replicate_bitmask_imm (unsigned HOST_WIDE_INT val, machine_mode mode)
{
unsigned int size = GET_MODE_UNIT_PRECISION (mode);
while (size < 64)
{
val &= (HOST_WIDE_INT_1U << size) - 1;
val |= val << size;
size *= 2;
}
return val;
}
static const unsigned HOST_WIDE_INT bitmask_imm_mul[] =
{
0x0000000100000001ull,
0x0001000100010001ull,
0x0101010101010101ull,
0x1111111111111111ull,
0x5555555555555555ull,
};
bool
aarch64_bitmask_imm (HOST_WIDE_INT val_in, machine_mode mode)
{
unsigned HOST_WIDE_INT val, tmp, mask, first_one, next_one;
int bits;
val = aarch64_replicate_bitmask_imm (val_in, mode);
tmp = val + (val & -val);
if (tmp == (tmp & -tmp))
return (val + 1) > 1;
if (mode == SImode)
val = (val << 32) | (val & 0xffffffff);
if (val & 1)
val = ~val;
first_one = val & -val;
tmp = val & (val + first_one);
if (tmp == 0)
return true;
next_one = tmp & -tmp;
bits = clz_hwi (first_one) - clz_hwi (next_one);
mask = val ^ tmp;
if ((mask >> bits) != 0 || bits != (bits & -bits))
return false;
return val == mask * bitmask_imm_mul[__builtin_clz (bits) - 26];
}
unsigned HOST_WIDE_INT
aarch64_and_split_imm1 (HOST_WIDE_INT val_in)
{
int lowest_bit_set = ctz_hwi (val_in);
int highest_bit_set = floor_log2 (val_in);
gcc_assert (val_in != 0);
return ((HOST_WIDE_INT_UC (2) << highest_bit_set) -
(HOST_WIDE_INT_1U << lowest_bit_set));
}
unsigned HOST_WIDE_INT
aarch64_and_split_imm2 (HOST_WIDE_INT val_in)
{
return val_in | ~aarch64_and_split_imm1 (val_in);
}
bool
aarch64_and_bitmask_imm (unsigned HOST_WIDE_INT val_in, machine_mode mode)
{
scalar_int_mode int_mode;
if (!is_a <scalar_int_mode> (mode, &int_mode))
return false;
if (aarch64_bitmask_imm (val_in, int_mode))
return false;
if (aarch64_move_imm (val_in, int_mode))
return false;
unsigned HOST_WIDE_INT imm2 = aarch64_and_split_imm2 (val_in);
return aarch64_bitmask_imm (imm2, int_mode);
}
bool
aarch64_move_imm (HOST_WIDE_INT val, machine_mode mode)
{
scalar_int_mode int_mode;
if (!is_a <scalar_int_mode> (mode, &int_mode))
return false;
if (aarch64_movw_imm (val, int_mode) || aarch64_movw_imm (~val, int_mode))
return 1;
return aarch64_bitmask_imm (val, int_mode);
}
static bool
aarch64_cannot_force_const_mem (machine_mode mode ATTRIBUTE_UNUSED, rtx x)
{
rtx base, offset;
if (GET_CODE (x) == HIGH)
return true;
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, x, ALL)
if (GET_CODE (*iter) == CONST_POLY_INT)
return true;
split_const (x, &base, &offset);
if (GET_CODE (base) == SYMBOL_REF || GET_CODE (base) == LABEL_REF)
{
if (aarch64_classify_symbol (base, INTVAL (offset))
!= SYMBOL_FORCE_TO_MEM)
return true;
else
return mode != ptr_mode;
}
return aarch64_tls_referenced_p (x);
}
static unsigned int
aarch64_case_values_threshold (void)
{
if (optimize > 2
&& selected_cpu->tune->max_case_values != 0)
return selected_cpu->tune->max_case_values;
else
return optimize_size ? default_case_values_threshold () : 17;
}
bool
aarch64_regno_ok_for_index_p (int regno, bool strict_p)
{
if (!HARD_REGISTER_NUM_P (regno))
{
if (!strict_p)
return true;
if (!reg_renumber)
return false;
regno = reg_renumber[regno];
}
return GP_REGNUM_P (regno);
}
bool
aarch64_regno_ok_for_base_p (int regno, bool strict_p)
{
if (!HARD_REGISTER_NUM_P (regno))
{
if (!strict_p)
return true;
if (!reg_renumber)
return false;
regno = reg_renumber[regno];
}
return (GP_REGNUM_P (regno)
|| regno == SP_REGNUM
|| regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM);
}
static bool
aarch64_base_register_rtx_p (rtx x, bool strict_p)
{
if (!strict_p
&& GET_CODE (x) == SUBREG
&& contains_reg_of_mode[GENERAL_REGS][GET_MODE (SUBREG_REG (x))])
x = SUBREG_REG (x);
return (REG_P (x) && aarch64_regno_ok_for_base_p (REGNO (x), strict_p));
}
static bool
aarch64_classify_index (struct aarch64_address_info *info, rtx x,
machine_mode mode, bool strict_p)
{
enum aarch64_address_type type;
rtx index;
int shift;
if ((REG_P (x) || GET_CODE (x) == SUBREG)
&& GET_MODE (x) == Pmode)
{
type = ADDRESS_REG_REG;
index = x;
shift = 0;
}
else if ((GET_CODE (x) == SIGN_EXTEND
|| GET_CODE (x) == ZERO_EXTEND)
&& GET_MODE (x) == DImode
&& GET_MODE (XEXP (x, 0)) == SImode)
{
type = (GET_CODE (x) == SIGN_EXTEND)
? ADDRESS_REG_SXTW : ADDRESS_REG_UXTW;
index = XEXP (x, 0);
shift = 0;
}
else if (GET_CODE (x) == MULT
&& (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND
|| GET_CODE (XEXP (x, 0)) == ZERO_EXTEND)
&& GET_MODE (XEXP (x, 0)) == DImode
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == SImode
&& CONST_INT_P (XEXP (x, 1)))
{
type = (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND)
? ADDRESS_REG_SXTW : ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = exact_log2 (INTVAL (XEXP (x, 1)));
}
else if (GET_CODE (x) == ASHIFT
&& (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND
|| GET_CODE (XEXP (x, 0)) == ZERO_EXTEND)
&& GET_MODE (XEXP (x, 0)) == DImode
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == SImode
&& CONST_INT_P (XEXP (x, 1)))
{
type = (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND)
? ADDRESS_REG_SXTW : ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = INTVAL (XEXP (x, 1));
}
else if ((GET_CODE (x) == SIGN_EXTRACT
|| GET_CODE (x) == ZERO_EXTRACT)
&& GET_MODE (x) == DImode
&& GET_CODE (XEXP (x, 0)) == MULT
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == DImode
&& CONST_INT_P (XEXP (XEXP (x, 0), 1)))
{
type = (GET_CODE (x) == SIGN_EXTRACT)
? ADDRESS_REG_SXTW : ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = exact_log2 (INTVAL (XEXP (XEXP (x, 0), 1)));
if (INTVAL (XEXP (x, 1)) != 32 + shift
|| INTVAL (XEXP (x, 2)) != 0)
shift = -1;
}
else if (GET_CODE (x) == AND
&& GET_MODE (x) == DImode
&& GET_CODE (XEXP (x, 0)) == MULT
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == DImode
&& CONST_INT_P (XEXP (XEXP (x, 0), 1))
&& CONST_INT_P (XEXP (x, 1)))
{
type = ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = exact_log2 (INTVAL (XEXP (XEXP (x, 0), 1)));
if (INTVAL (XEXP (x, 1)) != (HOST_WIDE_INT)0xffffffff << shift)
shift = -1;
}
else if ((GET_CODE (x) == SIGN_EXTRACT
|| GET_CODE (x) == ZERO_EXTRACT)
&& GET_MODE (x) == DImode
&& GET_CODE (XEXP (x, 0)) == ASHIFT
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == DImode
&& CONST_INT_P (XEXP (XEXP (x, 0), 1)))
{
type = (GET_CODE (x) == SIGN_EXTRACT)
? ADDRESS_REG_SXTW : ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = INTVAL (XEXP (XEXP (x, 0), 1));
if (INTVAL (XEXP (x, 1)) != 32 + shift
|| INTVAL (XEXP (x, 2)) != 0)
shift = -1;
}
else if (GET_CODE (x) == AND
&& GET_MODE (x) == DImode
&& GET_CODE (XEXP (x, 0)) == ASHIFT
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == DImode
&& CONST_INT_P (XEXP (XEXP (x, 0), 1))
&& CONST_INT_P (XEXP (x, 1)))
{
type = ADDRESS_REG_UXTW;
index = XEXP (XEXP (x, 0), 0);
shift = INTVAL (XEXP (XEXP (x, 0), 1));
if (INTVAL (XEXP (x, 1)) != (HOST_WIDE_INT)0xffffffff << shift)
shift = -1;
}
else if (GET_CODE (x) == MULT
&& GET_MODE (x) == Pmode
&& GET_MODE (XEXP (x, 0)) == Pmode
&& CONST_INT_P (XEXP (x, 1)))
{
type = ADDRESS_REG_REG;
index = XEXP (x, 0);
shift = exact_log2 (INTVAL (XEXP (x, 1)));
}
else if (GET_CODE (x) == ASHIFT
&& GET_MODE (x) == Pmode
&& GET_MODE (XEXP (x, 0)) == Pmode
&& CONST_INT_P (XEXP (x, 1)))
{
type = ADDRESS_REG_REG;
index = XEXP (x, 0);
shift = INTVAL (XEXP (x, 1));
}
else
return false;
if (!strict_p
&& GET_CODE (index) == SUBREG
&& contains_reg_of_mode[GENERAL_REGS][GET_MODE (SUBREG_REG (index))])
index = SUBREG_REG (index);
if (aarch64_sve_data_mode_p (mode))
{
if (type != ADDRESS_REG_REG
|| (1 << shift) != GET_MODE_UNIT_SIZE (mode))
return false;
}
else
{
if (shift != 0
&& !(IN_RANGE (shift, 1, 3)
&& known_eq (1 << shift, GET_MODE_SIZE (mode))))
return false;
}
if (REG_P (index)
&& aarch64_regno_ok_for_index_p (REGNO (index), strict_p))
{
info->type = type;
info->offset = index;
info->shift = shift;
return true;
}
return false;
}
static bool
aarch64_mode_valid_for_sched_fusion_p (machine_mode mode)
{
return mode == SImode || mode == DImode
|| mode == SFmode || mode == DFmode
|| (aarch64_vector_mode_supported_p (mode)
&& known_eq (GET_MODE_SIZE (mode), 8));
}
static bool
virt_or_elim_regno_p (unsigned regno)
{
return ((regno >= FIRST_VIRTUAL_REGISTER
&& regno <= LAST_VIRTUAL_POINTER_REGISTER)
|| regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM);
}
static bool
aarch64_classify_address (struct aarch64_address_info *info,
rtx x, machine_mode mode, bool strict_p,
aarch64_addr_query_type type = ADDR_QUERY_M)
{
enum rtx_code code = GET_CODE (x);
rtx op0, op1;
poly_int64 offset;
HOST_WIDE_INT const_size;
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
bool advsimd_struct_p = (vec_flags == (VEC_ADVSIMD | VEC_STRUCT));
bool load_store_pair_p = (type == ADDR_QUERY_LDP_STP
|| mode == TImode
|| mode == TFmode
|| (BYTES_BIG_ENDIAN && advsimd_struct_p));
bool allow_reg_index_p = (!load_store_pair_p
&& (known_lt (GET_MODE_SIZE (mode), 16)
|| vec_flags == VEC_ADVSIMD
|| vec_flags == VEC_SVE_DATA));
if ((vec_flags & (VEC_SVE_DATA | VEC_SVE_PRED)) != 0
&& (code != REG && code != PLUS))
return false;
if (advsimd_struct_p
&& !BYTES_BIG_ENDIAN
&& (code != POST_INC && code != REG))
return false;
gcc_checking_assert (GET_MODE (x) == VOIDmode
|| SCALAR_INT_MODE_P (GET_MODE (x)));
switch (code)
{
case REG:
case SUBREG:
info->type = ADDRESS_REG_IMM;
info->base = x;
info->offset = const0_rtx;
info->const_offset = 0;
return aarch64_base_register_rtx_p (x, strict_p);
case PLUS:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (! strict_p
&& REG_P (op0)
&& virt_or_elim_regno_p (REGNO (op0))
&& poly_int_rtx_p (op1, &offset))
{
info->type = ADDRESS_REG_IMM;
info->base = op0;
info->offset = op1;
info->const_offset = offset;
return true;
}
if (maybe_ne (GET_MODE_SIZE (mode), 0)
&& aarch64_base_register_rtx_p (op0, strict_p)
&& poly_int_rtx_p (op1, &offset))
{
info->type = ADDRESS_REG_IMM;
info->base = op0;
info->offset = op1;
info->const_offset = offset;
if (mode == TImode || mode == TFmode)
return (aarch64_offset_7bit_signed_scaled_p (DImode, offset)
&& (offset_9bit_signed_unscaled_p (mode, offset)
|| offset_12bit_unsigned_scaled_p (mode, offset)));
if (mode == OImode)
return aarch64_offset_7bit_signed_scaled_p (TImode, offset);
if (mode == CImode)
return (aarch64_offset_7bit_signed_scaled_p (TImode, offset)
&& (offset_9bit_signed_unscaled_p (V16QImode, offset + 32)
|| offset_12bit_unsigned_scaled_p (V16QImode,
offset + 32)));
if (mode == XImode)
return (aarch64_offset_7bit_signed_scaled_p (TImode, offset)
&& aarch64_offset_7bit_signed_scaled_p (TImode,
offset + 32));
if (vec_flags == VEC_SVE_DATA)
return (type == ADDR_QUERY_M
? offset_4bit_signed_scaled_p (mode, offset)
: offset_9bit_signed_scaled_p (mode, offset));
if (vec_flags == (VEC_SVE_DATA | VEC_STRUCT))
{
poly_int64 end_offset = (offset
+ GET_MODE_SIZE (mode)
- BYTES_PER_SVE_VECTOR);
return (type == ADDR_QUERY_M
? offset_4bit_signed_scaled_p (mode, offset)
: (offset_9bit_signed_scaled_p (SVE_BYTE_MODE, offset)
&& offset_9bit_signed_scaled_p (SVE_BYTE_MODE,
end_offset)));
}
if (vec_flags == VEC_SVE_PRED)
return offset_9bit_signed_scaled_p (mode, offset);
if (load_store_pair_p)
return ((known_eq (GET_MODE_SIZE (mode), 4)
|| known_eq (GET_MODE_SIZE (mode), 8))
&& aarch64_offset_7bit_signed_scaled_p (mode, offset));
else
return (offset_9bit_signed_unscaled_p (mode, offset)
|| offset_12bit_unsigned_scaled_p (mode, offset));
}
if (allow_reg_index_p)
{
if (aarch64_base_register_rtx_p (op0, strict_p)
&& aarch64_classify_index (info, op1, mode, strict_p))
{
info->base = op0;
return true;
}
if (aarch64_base_register_rtx_p (op1, strict_p)
&& aarch64_classify_index (info, op0, mode, strict_p))
{
info->base = op1;
return true;
}
}
return false;
case POST_INC:
case POST_DEC:
case PRE_INC:
case PRE_DEC:
info->type = ADDRESS_REG_WB;
info->base = XEXP (x, 0);
info->offset = NULL_RTX;
return aarch64_base_register_rtx_p (info->base, strict_p);
case POST_MODIFY:
case PRE_MODIFY:
info->type = ADDRESS_REG_WB;
info->base = XEXP (x, 0);
if (GET_CODE (XEXP (x, 1)) == PLUS
&& poly_int_rtx_p (XEXP (XEXP (x, 1), 1), &offset)
&& rtx_equal_p (XEXP (XEXP (x, 1), 0), info->base)
&& aarch64_base_register_rtx_p (info->base, strict_p))
{
info->offset = XEXP (XEXP (x, 1), 1);
info->const_offset = offset;
if (mode == TImode || mode == TFmode)
return (aarch64_offset_7bit_signed_scaled_p (mode, offset)
&& offset_9bit_signed_unscaled_p (mode, offset));
if (load_store_pair_p)
return ((known_eq (GET_MODE_SIZE (mode), 4)
|| known_eq (GET_MODE_SIZE (mode), 8))
&& aarch64_offset_7bit_signed_scaled_p (mode, offset));
else
return offset_9bit_signed_unscaled_p (mode, offset);
}
return false;
case CONST:
case SYMBOL_REF:
case LABEL_REF:
info->type = ADDRESS_SYMBOLIC;
if (!load_store_pair_p
&& GET_MODE_SIZE (mode).is_constant (&const_size)
&& const_size >= 4)
{
rtx sym, addend;
split_const (x, &sym, &addend);
return ((GET_CODE (sym) == LABEL_REF
|| (GET_CODE (sym) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (sym)
&& aarch64_pcrelative_literal_loads)));
}
return false;
case LO_SUM:
info->type = ADDRESS_LO_SUM;
info->base = XEXP (x, 0);
info->offset = XEXP (x, 1);
if (allow_reg_index_p
&& aarch64_base_register_rtx_p (info->base, strict_p))
{
rtx sym, offs;
split_const (info->offset, &sym, &offs);
if (GET_CODE (sym) == SYMBOL_REF
&& (aarch64_classify_symbol (sym, INTVAL (offs))
== SYMBOL_SMALL_ABSOLUTE))
{
unsigned int align;
if (CONSTANT_POOL_ADDRESS_P (sym))
align = GET_MODE_ALIGNMENT (get_pool_mode (sym));
else if (TREE_CONSTANT_POOL_ADDRESS_P (sym))
{
tree exp = SYMBOL_REF_DECL (sym);
align = TYPE_ALIGN (TREE_TYPE (exp));
align = aarch64_constant_alignment (exp, align);
}
else if (SYMBOL_REF_DECL (sym))
align = DECL_ALIGN (SYMBOL_REF_DECL (sym));
else if (SYMBOL_REF_HAS_BLOCK_INFO_P (sym)
&& SYMBOL_REF_BLOCK (sym) != NULL)
align = SYMBOL_REF_BLOCK (sym)->alignment;
else
align = BITS_PER_UNIT;
poly_int64 ref_size = GET_MODE_SIZE (mode);
if (known_eq (ref_size, 0))
ref_size = GET_MODE_SIZE (DImode);
return (multiple_p (INTVAL (offs), ref_size)
&& multiple_p (align / BITS_PER_UNIT, ref_size));
}
}
return false;
default:
return false;
}
}
bool
aarch64_address_valid_for_prefetch_p (rtx x, bool strict_p)
{
struct aarch64_address_info addr;
bool res = aarch64_classify_address (&addr, x, DImode, strict_p);
if (!res)
return false;
return addr.type != ADDRESS_REG_WB;
}
bool
aarch64_symbolic_address_p (rtx x)
{
rtx offset;
split_const (x, &x, &offset);
return GET_CODE (x) == SYMBOL_REF || GET_CODE (x) == LABEL_REF;
}
enum aarch64_symbol_type
aarch64_classify_symbolic_expression (rtx x)
{
rtx offset;
split_const (x, &x, &offset);
return aarch64_classify_symbol (x, INTVAL (offset));
}
static bool
aarch64_legitimate_address_hook_p (machine_mode mode, rtx x, bool strict_p)
{
struct aarch64_address_info addr;
return aarch64_classify_address (&addr, x, mode, strict_p);
}
bool
aarch64_legitimate_address_p (machine_mode mode, rtx x, bool strict_p,
aarch64_addr_query_type type)
{
struct aarch64_address_info addr;
return aarch64_classify_address (&addr, x, mode, strict_p, type);
}
static bool
aarch64_legitimize_address_displacement (rtx *offset1, rtx *offset2,
poly_int64 orig_offset,
machine_mode mode)
{
HOST_WIDE_INT size;
if (GET_MODE_SIZE (mode).is_constant (&size))
{
HOST_WIDE_INT const_offset, second_offset;
const_offset = orig_offset.coeffs[0] - orig_offset.coeffs[1];
if (mode == TImode || mode == TFmode)
second_offset = ((const_offset + 0x100) & 0x1f8) - 0x100;
else if ((const_offset & (size - 1)) != 0)
second_offset = ((const_offset + 0x100) & 0x1ff) - 0x100;
else
second_offset = const_offset & (size < 4 ? 0xfff : 0x3ffc);
if (second_offset == 0 || known_eq (orig_offset, second_offset))
return false;
*offset1 = gen_int_mode (orig_offset - second_offset, Pmode);
*offset2 = gen_int_mode (second_offset, Pmode);
return true;
}
else
{
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
machine_mode step_mode
= (vec_flags & VEC_STRUCT) != 0 ? SVE_BYTE_MODE : mode;
HOST_WIDE_INT factor = GET_MODE_SIZE (step_mode).coeffs[1];
HOST_WIDE_INT vnum = orig_offset.coeffs[1] / factor;
if (vec_flags & VEC_SVE_DATA)
vnum = ((vnum + 128) & 255) - 128;
else
vnum = ((vnum + 256) & 511) - 256;
if (vnum == 0)
return false;
poly_int64 second_offset = GET_MODE_SIZE (step_mode) * vnum;
if (known_eq (second_offset, orig_offset))
return false;
*offset1 = gen_int_mode (orig_offset - second_offset, Pmode);
*offset2 = gen_int_mode (second_offset, Pmode);
return true;
}
}
bool
aarch64_reinterpret_float_as_int (rtx value, unsigned HOST_WIDE_INT *intval)
{
if (aarch64_float_const_zero_rtx_p (value))
{
*intval = 0;
return true;
}
scalar_float_mode mode;
if (GET_CODE (value) != CONST_DOUBLE
|| !is_a <scalar_float_mode> (GET_MODE (value), &mode)
|| GET_MODE_BITSIZE (mode) > HOST_BITS_PER_WIDE_INT
|| GET_MODE_BITSIZE (mode) > GET_MODE_BITSIZE (DFmode))
return false;
unsigned HOST_WIDE_INT ival = 0;
long res[2];
real_to_target (res,
CONST_DOUBLE_REAL_VALUE (value),
REAL_MODE_FORMAT (mode));
if (mode == DFmode)
{
int order = BYTES_BIG_ENDIAN ? 1 : 0;
ival = zext_hwi (res[order], 32);
ival |= (zext_hwi (res[1 - order], 32) << 32);
}
else
ival = zext_hwi (res[0], 32);
*intval = ival;
return true;
}
bool
aarch64_float_const_rtx_p (rtx x)
{
machine_mode mode = GET_MODE (x);
if (mode == VOIDmode)
return false;
unsigned HOST_WIDE_INT ival;
if (GET_CODE (x) == CONST_DOUBLE
&& SCALAR_FLOAT_MODE_P (mode)
&& aarch64_reinterpret_float_as_int (x, &ival))
{
scalar_int_mode imode = (mode == HFmode
? SImode
: int_mode_for_mode (mode).require ());
int num_instr = aarch64_internal_mov_immediate
(NULL_RTX, gen_int_mode (ival, imode), false, imode);
return num_instr < 3;
}
return false;
}
bool
aarch64_float_const_zero_rtx_p (rtx x)
{
if (GET_MODE (x) == VOIDmode)
return false;
if (REAL_VALUE_MINUS_ZERO (*CONST_DOUBLE_REAL_VALUE (x)))
return !HONOR_SIGNED_ZEROS (GET_MODE (x));
return real_equal (CONST_DOUBLE_REAL_VALUE (x), &dconst0);
}
bool
aarch64_can_const_movi_rtx_p (rtx x, machine_mode mode)
{
if (!TARGET_SIMD)
return false;
machine_mode vmode;
scalar_int_mode imode;
unsigned HOST_WIDE_INT ival;
if (GET_CODE (x) == CONST_DOUBLE
&& SCALAR_FLOAT_MODE_P (mode))
{
if (!aarch64_reinterpret_float_as_int (x, &ival))
return false;
if (aarch64_float_const_zero_rtx_p (x))
return true;
imode = int_mode_for_mode (mode).require ();
}
else if (GET_CODE (x) == CONST_INT
&& is_a <scalar_int_mode> (mode, &imode))
ival = INTVAL (x);
else
return false;
int width = GET_MODE_BITSIZE (imode) == 64 ? 128 : 64;
vmode = aarch64_simd_container_mode (imode, width);
rtx v_op = aarch64_simd_gen_const_vector_dup (vmode, ival);
return aarch64_simd_valid_immediate (v_op, NULL);
}
static bool
aarch64_fixed_condition_code_regs (unsigned int *p1, unsigned int *p2)
{
*p1 = CC_REGNUM;
*p2 = INVALID_REGNUM;
return true;
}
void
aarch64_expand_call (rtx result, rtx mem, bool sibcall)
{
rtx call, callee, tmp;
rtvec vec;
machine_mode mode;
gcc_assert (MEM_P (mem));
callee = XEXP (mem, 0);
mode = GET_MODE (callee);
gcc_assert (mode == Pmode);
if (SYMBOL_REF_P (callee)
? (aarch64_is_long_call_p (callee)
|| aarch64_is_noplt_call_p (callee))
: !REG_P (callee))
XEXP (mem, 0) = force_reg (mode, callee);
call = gen_rtx_CALL (VOIDmode, mem, const0_rtx);
if (result != NULL_RTX)
call = gen_rtx_SET (result, call);
if (sibcall)
tmp = ret_rtx;
else
tmp = gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, LR_REGNUM));
vec = gen_rtvec (2, call, tmp);
call = gen_rtx_PARALLEL (VOIDmode, vec);
aarch64_emit_call_insn (call);
}
void
aarch64_emit_call_insn (rtx pat)
{
rtx insn = emit_call_insn (pat);
rtx *fusage = &CALL_INSN_FUNCTION_USAGE (insn);
clobber_reg (fusage, gen_rtx_REG (word_mode, IP0_REGNUM));
clobber_reg (fusage, gen_rtx_REG (word_mode, IP1_REGNUM));
}
machine_mode
aarch64_select_cc_mode (RTX_CODE code, rtx x, rtx y)
{
if (GET_MODE_CLASS (GET_MODE (x)) == MODE_FLOAT)
{
switch (code)
{
case EQ:
case NE:
case UNORDERED:
case ORDERED:
case UNLT:
case UNLE:
case UNGT:
case UNGE:
case UNEQ:
return CCFPmode;
case LT:
case LE:
case GT:
case GE:
case LTGT:
return CCFPEmode;
default:
gcc_unreachable ();
}
}
if (y == const0_rtx && REG_P (x)
&& (code == EQ || code == NE)
&& (GET_MODE (x) == HImode || GET_MODE (x) == QImode))
return CC_NZmode;
if (y == const0_rtx && GET_CODE (x) == ZERO_EXTEND
&& (GET_MODE (x) == SImode || GET_MODE (x) == DImode)
&& (GET_MODE (XEXP (x, 0)) == HImode || GET_MODE (XEXP (x, 0)) == QImode)
&& (code == EQ || code == NE))
return CC_NZmode;
if ((GET_MODE (x) == SImode || GET_MODE (x) == DImode)
&& y == const0_rtx
&& (code == EQ || code == NE || code == LT || code == GE)
&& (GET_CODE (x) == PLUS || GET_CODE (x) == MINUS || GET_CODE (x) == AND
|| GET_CODE (x) == NEG
|| (GET_CODE (x) == ZERO_EXTRACT && CONST_INT_P (XEXP (x, 1))
&& CONST_INT_P (XEXP (x, 2)))))
return CC_NZmode;
if ((GET_MODE (x) == SImode || GET_MODE (x) == DImode)
&& (REG_P (y) || GET_CODE (y) == SUBREG || y == const0_rtx)
&& (GET_CODE (x) == ASHIFT || GET_CODE (x) == ASHIFTRT
|| GET_CODE (x) == LSHIFTRT
|| GET_CODE (x) == ZERO_EXTEND || GET_CODE (x) == SIGN_EXTEND))
return CC_SWPmode;
if ((GET_MODE (x) == SImode || GET_MODE (x) == DImode)
&& (REG_P (y) || GET_CODE (y) == SUBREG)
&& (code == EQ || code == NE)
&& GET_CODE (x) == NEG)
return CC_Zmode;
if ((GET_MODE (x) == DImode || GET_MODE (x) == TImode)
&& code == NE
&& GET_CODE (x) == PLUS
&& GET_CODE (y) == ZERO_EXTEND)
return CC_Cmode;
return CCmode;
}
static int
aarch64_get_condition_code_1 (machine_mode, enum rtx_code);
int
aarch64_get_condition_code (rtx x)
{
machine_mode mode = GET_MODE (XEXP (x, 0));
enum rtx_code comp_code = GET_CODE (x);
if (GET_MODE_CLASS (mode) != MODE_CC)
mode = SELECT_CC_MODE (comp_code, XEXP (x, 0), XEXP (x, 1));
return aarch64_get_condition_code_1 (mode, comp_code);
}
static int
aarch64_get_condition_code_1 (machine_mode mode, enum rtx_code comp_code)
{
switch (mode)
{
case E_CCFPmode:
case E_CCFPEmode:
switch (comp_code)
{
case GE: return AARCH64_GE;
case GT: return AARCH64_GT;
case LE: return AARCH64_LS;
case LT: return AARCH64_MI;
case NE: return AARCH64_NE;
case EQ: return AARCH64_EQ;
case ORDERED: return AARCH64_VC;
case UNORDERED: return AARCH64_VS;
case UNLT: return AARCH64_LT;
case UNLE: return AARCH64_LE;
case UNGT: return AARCH64_HI;
case UNGE: return AARCH64_PL;
default: return -1;
}
break;
case E_CCmode:
switch (comp_code)
{
case NE: return AARCH64_NE;
case EQ: return AARCH64_EQ;
case GE: return AARCH64_GE;
case GT: return AARCH64_GT;
case LE: return AARCH64_LE;
case LT: return AARCH64_LT;
case GEU: return AARCH64_CS;
case GTU: return AARCH64_HI;
case LEU: return AARCH64_LS;
case LTU: return AARCH64_CC;
default: return -1;
}
break;
case E_CC_SWPmode:
switch (comp_code)
{
case NE: return AARCH64_NE;
case EQ: return AARCH64_EQ;
case GE: return AARCH64_LE;
case GT: return AARCH64_LT;
case LE: return AARCH64_GE;
case LT: return AARCH64_GT;
case GEU: return AARCH64_LS;
case GTU: return AARCH64_CC;
case LEU: return AARCH64_CS;
case LTU: return AARCH64_HI;
default: return -1;
}
break;
case E_CC_NZmode:
switch (comp_code)
{
case NE: return AARCH64_NE;
case EQ: return AARCH64_EQ;
case GE: return AARCH64_PL;
case LT: return AARCH64_MI;
default: return -1;
}
break;
case E_CC_Zmode:
switch (comp_code)
{
case NE: return AARCH64_NE;
case EQ: return AARCH64_EQ;
default: return -1;
}
break;
case E_CC_Cmode:
switch (comp_code)
{
case NE: return AARCH64_CS;
case EQ: return AARCH64_CC;
default: return -1;
}
break;
default:
return -1;
}
return -1;
}
bool
aarch64_const_vec_all_same_in_range_p (rtx x,
HOST_WIDE_INT minval,
HOST_WIDE_INT maxval)
{
rtx elt;
return (const_vec_duplicate_p (x, &elt)
&& CONST_INT_P (elt)
&& IN_RANGE (INTVAL (elt), minval, maxval));
}
bool
aarch64_const_vec_all_same_int_p (rtx x, HOST_WIDE_INT val)
{
return aarch64_const_vec_all_same_in_range_p (x, val, val);
}
static bool
aarch64_const_vec_all_in_range_p (rtx vec,
HOST_WIDE_INT minval,
HOST_WIDE_INT maxval)
{
if (GET_CODE (vec) != CONST_VECTOR
|| GET_MODE_CLASS (GET_MODE (vec)) != MODE_VECTOR_INT)
return false;
int nunits;
if (!CONST_VECTOR_STEPPED_P (vec))
nunits = const_vector_encoded_nelts (vec);
else if (!CONST_VECTOR_NUNITS (vec).is_constant (&nunits))
return false;
for (int i = 0; i < nunits; i++)
{
rtx vec_elem = CONST_VECTOR_ELT (vec, i);
if (!CONST_INT_P (vec_elem)
|| !IN_RANGE (INTVAL (vec_elem), minval, maxval))
return false;
}
return true;
}
#define AARCH64_CC_V 1
#define AARCH64_CC_C (1 << 1)
#define AARCH64_CC_Z (1 << 2)
#define AARCH64_CC_N (1 << 3)
static const int aarch64_nzcv_codes[] =
{
0,		
AARCH64_CC_Z,	
0,		
AARCH64_CC_C,	
0,		
AARCH64_CC_N, 
0,		
AARCH64_CC_V, 
0,		
AARCH64_CC_C,	
AARCH64_CC_V,	
0,		
AARCH64_CC_Z, 
0,		
0,		
0		
};
static bool
aarch64_print_vector_float_operand (FILE *f, rtx x, bool negate)
{
rtx elt;
if (!const_vec_duplicate_p (x, &elt))
return false;
REAL_VALUE_TYPE r = *CONST_DOUBLE_REAL_VALUE (elt);
if (negate)
r = real_value_negate (&r);
if (real_equal (&r, &dconst0))
asm_fprintf (f, "0.0");
else if (real_equal (&r, &dconst1))
asm_fprintf (f, "1.0");
else if (real_equal (&r, &dconsthalf))
asm_fprintf (f, "0.5");
else
return false;
return true;
}
static char
sizetochar (int size)
{
switch (size)
{
case 64: return 'd';
case 32: return 's';
case 16: return 'h';
case 8 : return 'b';
default: gcc_unreachable ();
}
}
static void
aarch64_print_operand (FILE *f, rtx x, int code)
{
rtx elt;
switch (code)
{
case 'c':
switch (GET_CODE (x))
{
case CONST_INT:
fprintf (f, HOST_WIDE_INT_PRINT_DEC, INTVAL (x));
break;
case SYMBOL_REF:
output_addr_const (f, x);
break;
case CONST:
if (GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF)
{
output_addr_const (f, x);
break;
}
default:
output_operand_lossage ("unsupported operand for code '%c'", code);
}
break;
case 'e':
{
int n;
if (!CONST_INT_P (x)
|| (n = exact_log2 (INTVAL (x) & ~7)) <= 0)
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
switch (n)
{
case 3:
fputc ('b', f);
break;
case 4:
fputc ('h', f);
break;
case 5:
fputc ('w', f);
break;
default:
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
}
break;
case 'p':
{
int n;
if (!CONST_INT_P (x) || (n = exact_log2 (INTVAL (x))) < 0)
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
asm_fprintf (f, "%d", n);
}
break;
case 'P':
if (!CONST_INT_P (x))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
asm_fprintf (f, "%u", popcount_hwi (INTVAL (x)));
break;
case 'H':
if (!REG_P (x) || !GP_REGNUM_P (REGNO (x) + 1))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
asm_fprintf (f, "%s", reg_names [REGNO (x) + 1]);
break;
case 'M':
case 'm':
{
int cond_code;
if (x == const_true_rtx)
{
if (code == 'M')
fputs ("nv", f);
return;
}
if (!COMPARISON_P (x))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
cond_code = aarch64_get_condition_code (x);
gcc_assert (cond_code >= 0);
if (code == 'M')
cond_code = AARCH64_INVERSE_CONDITION_CODE (cond_code);
fputs (aarch64_condition_codes[cond_code], f);
}
break;
case 'N':
if (!const_vec_duplicate_p (x, &elt))
{
output_operand_lossage ("invalid vector constant");
return;
}
if (GET_MODE_CLASS (GET_MODE (x)) == MODE_VECTOR_INT)
asm_fprintf (f, "%wd", -INTVAL (elt));
else if (GET_MODE_CLASS (GET_MODE (x)) == MODE_VECTOR_FLOAT
&& aarch64_print_vector_float_operand (f, x, true))
;
else
{
output_operand_lossage ("invalid vector constant");
return;
}
break;
case 'b':
case 'h':
case 's':
case 'd':
case 'q':
if (!REG_P (x) || !FP_REGNUM_P (REGNO (x)))
{
output_operand_lossage ("incompatible floating point / vector register operand for '%%%c'", code);
return;
}
asm_fprintf (f, "%c%d", code, REGNO (x) - V0_REGNUM);
break;
case 'S':
case 'T':
case 'U':
case 'V':
if (!REG_P (x) || !FP_REGNUM_P (REGNO (x)))
{
output_operand_lossage ("incompatible floating point / vector register operand for '%%%c'", code);
return;
}
asm_fprintf (f, "%c%d",
aarch64_sve_data_mode_p (GET_MODE (x)) ? 'z' : 'v',
REGNO (x) - V0_REGNUM + (code - 'S'));
break;
case 'R':
if (!REG_P (x) || !FP_REGNUM_P (REGNO (x)))
{
output_operand_lossage ("incompatible floating point / vector register operand for '%%%c'", code);
return;
}
asm_fprintf (f, "q%d", REGNO (x) - V0_REGNUM + 1);
break;
case 'X':
if (!CONST_INT_P (x))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
asm_fprintf (f, "0x%wx", UINTVAL (x) & 0xffff);
break;
case 'C':
{
if (!const_vec_duplicate_p (x, &elt) || !CONST_INT_P (elt))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
scalar_mode inner_mode = GET_MODE_INNER (GET_MODE (x));
asm_fprintf (f, "0x%wx", UINTVAL (elt) & GET_MODE_MASK (inner_mode));
}
break;
case 'D':
{
if (!const_vec_duplicate_p (x, &elt) || !CONST_INT_P (elt))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
scalar_mode inner_mode = GET_MODE_INNER (GET_MODE (x));
asm_fprintf (f, "%wd", UINTVAL (elt) & GET_MODE_MASK (inner_mode));
}
break;
case 'w':
case 'x':
if (x == const0_rtx
|| (CONST_DOUBLE_P (x) && aarch64_float_const_zero_rtx_p (x)))
{
asm_fprintf (f, "%czr", code);
break;
}
if (REG_P (x) && GP_REGNUM_P (REGNO (x)))
{
asm_fprintf (f, "%c%d", code, REGNO (x) - R0_REGNUM);
break;
}
if (REG_P (x) && REGNO (x) == SP_REGNUM)
{
asm_fprintf (f, "%ssp", code == 'w' ? "w" : "");
break;
}
case 0:
if (x == NULL)
{
output_operand_lossage ("missing operand");
return;
}
switch (GET_CODE (x))
{
case REG:
if (aarch64_sve_data_mode_p (GET_MODE (x)))
{
if (REG_NREGS (x) == 1)
asm_fprintf (f, "z%d", REGNO (x) - V0_REGNUM);
else
{
char suffix
= sizetochar (GET_MODE_UNIT_BITSIZE (GET_MODE (x)));
asm_fprintf (f, "{z%d.%c - z%d.%c}",
REGNO (x) - V0_REGNUM, suffix,
END_REGNO (x) - V0_REGNUM - 1, suffix);
}
}
else
asm_fprintf (f, "%s", reg_names [REGNO (x)]);
break;
case MEM:
output_address (GET_MODE (x), XEXP (x, 0));
break;
case LABEL_REF:
case SYMBOL_REF:
output_addr_const (asm_out_file, x);
break;
case CONST_INT:
asm_fprintf (f, "%wd", INTVAL (x));
break;
case CONST:
if (!VECTOR_MODE_P (GET_MODE (x)))
{
output_addr_const (asm_out_file, x);
break;
}
case CONST_VECTOR:
if (!const_vec_duplicate_p (x, &elt))
{
output_operand_lossage ("invalid vector constant");
return;
}
if (GET_MODE_CLASS (GET_MODE (x)) == MODE_VECTOR_INT)
asm_fprintf (f, "%wd", INTVAL (elt));
else if (GET_MODE_CLASS (GET_MODE (x)) == MODE_VECTOR_FLOAT
&& aarch64_print_vector_float_operand (f, x, false))
;
else
{
output_operand_lossage ("invalid vector constant");
return;
}
break;
case CONST_DOUBLE:
gcc_assert (GET_MODE (x) != VOIDmode);
if (aarch64_float_const_zero_rtx_p (x))
{
fputc ('0', f);
break;
}
else if (aarch64_float_const_representable_p (x))
{
#define buf_size 20
char float_buf[buf_size] = {'\0'};
real_to_decimal_for_mode (float_buf,
CONST_DOUBLE_REAL_VALUE (x),
buf_size, buf_size,
1, GET_MODE (x));
asm_fprintf (asm_out_file, "%s", float_buf);
break;
#undef buf_size
}
output_operand_lossage ("invalid constant");
return;
default:
output_operand_lossage ("invalid operand");
return;
}
break;
case 'A':
if (GET_CODE (x) == HIGH)
x = XEXP (x, 0);
switch (aarch64_classify_symbolic_expression (x))
{
case SYMBOL_SMALL_GOT_4G:
asm_fprintf (asm_out_file, ":got:");
break;
case SYMBOL_SMALL_TLSGD:
asm_fprintf (asm_out_file, ":tlsgd:");
break;
case SYMBOL_SMALL_TLSDESC:
asm_fprintf (asm_out_file, ":tlsdesc:");
break;
case SYMBOL_SMALL_TLSIE:
asm_fprintf (asm_out_file, ":gottprel:");
break;
case SYMBOL_TLSLE24:
asm_fprintf (asm_out_file, ":tprel:");
break;
case SYMBOL_TINY_GOT:
gcc_unreachable ();
break;
default:
break;
}
output_addr_const (asm_out_file, x);
break;
case 'L':
switch (aarch64_classify_symbolic_expression (x))
{
case SYMBOL_SMALL_GOT_4G:
asm_fprintf (asm_out_file, ":lo12:");
break;
case SYMBOL_SMALL_TLSGD:
asm_fprintf (asm_out_file, ":tlsgd_lo12:");
break;
case SYMBOL_SMALL_TLSDESC:
asm_fprintf (asm_out_file, ":tlsdesc_lo12:");
break;
case SYMBOL_SMALL_TLSIE:
asm_fprintf (asm_out_file, ":gottprel_lo12:");
break;
case SYMBOL_TLSLE12:
asm_fprintf (asm_out_file, ":tprel_lo12:");
break;
case SYMBOL_TLSLE24:
asm_fprintf (asm_out_file, ":tprel_lo12_nc:");
break;
case SYMBOL_TINY_GOT:
asm_fprintf (asm_out_file, ":got:");
break;
case SYMBOL_TINY_TLSIE:
asm_fprintf (asm_out_file, ":gottprel:");
break;
default:
break;
}
output_addr_const (asm_out_file, x);
break;
case 'G':
switch (aarch64_classify_symbolic_expression (x))
{
case SYMBOL_TLSLE24:
asm_fprintf (asm_out_file, ":tprel_hi12:");
break;
default:
break;
}
output_addr_const (asm_out_file, x);
break;
case 'k':
{
HOST_WIDE_INT cond_code;
if (!CONST_INT_P (x))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
cond_code = INTVAL (x);
gcc_assert (cond_code >= 0 && cond_code <= AARCH64_NV);
asm_fprintf (f, "%d", aarch64_nzcv_codes[cond_code]);
}
break;
case 'y':
case 'z':
{
machine_mode mode = GET_MODE (x);
if (GET_CODE (x) != MEM
|| (code == 'y' && maybe_ne (GET_MODE_SIZE (mode), 16)))
{
output_operand_lossage ("invalid operand for '%%%c'", code);
return;
}
if (code == 'y')
mode = DFmode;
if (!aarch64_print_ldpstp_address (f, mode, XEXP (x, 0)))
output_operand_lossage ("invalid operand prefix '%%%c'", code);
}
break;
default:
output_operand_lossage ("invalid operand prefix '%%%c'", code);
return;
}
}
static bool
aarch64_print_address_internal (FILE *f, machine_mode mode, rtx x,
aarch64_addr_query_type type)
{
struct aarch64_address_info addr;
unsigned int size;
if (GET_MODE (x) != Pmode
&& (!CONST_INT_P (x)
|| trunc_int_for_mode (INTVAL (x), Pmode) != INTVAL (x)))
{
output_operand_lossage ("invalid address mode");
return false;
}
if (aarch64_classify_address (&addr, x, mode, true, type))
switch (addr.type)
{
case ADDRESS_REG_IMM:
if (known_eq (addr.const_offset, 0))
asm_fprintf (f, "[%s]", reg_names [REGNO (addr.base)]);
else if (aarch64_sve_data_mode_p (mode))
{
HOST_WIDE_INT vnum
= exact_div (addr.const_offset,
BYTES_PER_SVE_VECTOR).to_constant ();
asm_fprintf (f, "[%s, #%wd, mul vl]",
reg_names[REGNO (addr.base)], vnum);
}
else if (aarch64_sve_pred_mode_p (mode))
{
HOST_WIDE_INT vnum
= exact_div (addr.const_offset,
BYTES_PER_SVE_PRED).to_constant ();
asm_fprintf (f, "[%s, #%wd, mul vl]",
reg_names[REGNO (addr.base)], vnum);
}
else
asm_fprintf (f, "[%s, %wd]", reg_names [REGNO (addr.base)],
INTVAL (addr.offset));
return true;
case ADDRESS_REG_REG:
if (addr.shift == 0)
asm_fprintf (f, "[%s, %s]", reg_names [REGNO (addr.base)],
reg_names [REGNO (addr.offset)]);
else
asm_fprintf (f, "[%s, %s, lsl %u]", reg_names [REGNO (addr.base)],
reg_names [REGNO (addr.offset)], addr.shift);
return true;
case ADDRESS_REG_UXTW:
if (addr.shift == 0)
asm_fprintf (f, "[%s, w%d, uxtw]", reg_names [REGNO (addr.base)],
REGNO (addr.offset) - R0_REGNUM);
else
asm_fprintf (f, "[%s, w%d, uxtw %u]", reg_names [REGNO (addr.base)],
REGNO (addr.offset) - R0_REGNUM, addr.shift);
return true;
case ADDRESS_REG_SXTW:
if (addr.shift == 0)
asm_fprintf (f, "[%s, w%d, sxtw]", reg_names [REGNO (addr.base)],
REGNO (addr.offset) - R0_REGNUM);
else
asm_fprintf (f, "[%s, w%d, sxtw %u]", reg_names [REGNO (addr.base)],
REGNO (addr.offset) - R0_REGNUM, addr.shift);
return true;
case ADDRESS_REG_WB:
size = GET_MODE_SIZE (mode).to_constant ();
switch (GET_CODE (x))
{
case PRE_INC:
asm_fprintf (f, "[%s, %d]!", reg_names [REGNO (addr.base)], size);
return true;
case POST_INC:
asm_fprintf (f, "[%s], %d", reg_names [REGNO (addr.base)], size);
return true;
case PRE_DEC:
asm_fprintf (f, "[%s, -%d]!", reg_names [REGNO (addr.base)], size);
return true;
case POST_DEC:
asm_fprintf (f, "[%s], -%d", reg_names [REGNO (addr.base)], size);
return true;
case PRE_MODIFY:
asm_fprintf (f, "[%s, %wd]!", reg_names[REGNO (addr.base)],
INTVAL (addr.offset));
return true;
case POST_MODIFY:
asm_fprintf (f, "[%s], %wd", reg_names[REGNO (addr.base)],
INTVAL (addr.offset));
return true;
default:
break;
}
break;
case ADDRESS_LO_SUM:
asm_fprintf (f, "[%s, #:lo12:", reg_names [REGNO (addr.base)]);
output_addr_const (f, addr.offset);
asm_fprintf (f, "]");
return true;
case ADDRESS_SYMBOLIC:
output_addr_const (f, x);
return true;
}
return false;
}
static bool
aarch64_print_ldpstp_address (FILE *f, machine_mode mode, rtx x)
{
return aarch64_print_address_internal (f, mode, x, ADDR_QUERY_LDP_STP);
}
static void
aarch64_print_operand_address (FILE *f, machine_mode mode, rtx x)
{
if (!aarch64_print_address_internal (f, mode, x, ADDR_QUERY_ANY))
output_addr_const (f, x);
}
bool
aarch64_label_mentioned_p (rtx x)
{
const char *fmt;
int i;
if (GET_CODE (x) == LABEL_REF)
return true;
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
return false;
fmt = GET_RTX_FORMAT (GET_CODE (x));
for (i = GET_RTX_LENGTH (GET_CODE (x)) - 1; i >= 0; i--)
{
if (fmt[i] == 'E')
{
int j;
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
if (aarch64_label_mentioned_p (XVECEXP (x, i, j)))
return 1;
}
else if (fmt[i] == 'e' && aarch64_label_mentioned_p (XEXP (x, i)))
return 1;
}
return 0;
}
enum reg_class
aarch64_regno_regclass (unsigned regno)
{
if (GP_REGNUM_P (regno))
return GENERAL_REGS;
if (regno == SP_REGNUM)
return STACK_REG;
if (regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM)
return POINTER_REGS;
if (FP_REGNUM_P (regno))
return FP_LO_REGNUM_P (regno) ?  FP_LO_REGS : FP_REGS;
if (PR_REGNUM_P (regno))
return PR_LO_REGNUM_P (regno) ? PR_LO_REGS : PR_HI_REGS;
return NO_REGS;
}
static HOST_WIDE_INT
aarch64_anchor_offset (HOST_WIDE_INT offset, HOST_WIDE_INT size,
machine_mode mode)
{
if (size > 16)
return (offset + 0x400) & ~0x7f0;
if (offset & (size - 1))
{
if (mode == BLKmode)
return (offset + 512) & ~0x3ff;
return (offset + 0x100) & ~0x1ff;
}
if (IN_RANGE (offset, -256, 0))
return 0;
if (mode == TImode || mode == TFmode)
return (offset + 0x100) & ~0x1ff;
return offset & (~0xfff * size);
}
static rtx
aarch64_legitimize_address (rtx x, rtx , machine_mode mode)
{
if (GET_CODE (x) == PLUS && CONST_INT_P (XEXP (x, 1)))
{
rtx base = XEXP (x, 0);
rtx offset_rtx = XEXP (x, 1);
HOST_WIDE_INT offset = INTVAL (offset_rtx);
if (GET_CODE (base) == PLUS)
{
rtx op0 = XEXP (base, 0);
rtx op1 = XEXP (base, 1);
op0 = force_reg (Pmode, op0);
op1 = force_reg (Pmode, op1);
if (REG_POINTER (op1))
std::swap (op0, op1);
if (virt_or_elim_regno_p (REGNO (op0)))
{
base = expand_binop (Pmode, add_optab, op0, offset_rtx,
NULL_RTX, true, OPTAB_DIRECT);
return gen_rtx_PLUS (Pmode, base, op1);
}
base = expand_binop (Pmode, add_optab, op0, op1,
NULL_RTX, true, OPTAB_DIRECT);
x = gen_rtx_PLUS (Pmode, base, offset_rtx);
}
HOST_WIDE_INT size;
if (GET_MODE_SIZE (mode).is_constant (&size))
{
HOST_WIDE_INT base_offset = aarch64_anchor_offset (offset, size,
mode);
if (base_offset != 0)
{
base = plus_constant (Pmode, base, base_offset);
base = force_operand (base, NULL_RTX);
return plus_constant (Pmode, base, offset - base_offset);
}
}
}
return x;
}
static enum insn_code
aarch64_constant_pool_reload_icode (machine_mode mode)
{
switch (mode)
{
case E_SFmode:
return CODE_FOR_aarch64_reload_movcpsfdi;
case E_DFmode:
return CODE_FOR_aarch64_reload_movcpdfdi;
case E_TFmode:
return CODE_FOR_aarch64_reload_movcptfdi;
case E_V8QImode:
return CODE_FOR_aarch64_reload_movcpv8qidi;
case E_V16QImode:
return CODE_FOR_aarch64_reload_movcpv16qidi;
case E_V4HImode:
return CODE_FOR_aarch64_reload_movcpv4hidi;
case E_V8HImode:
return CODE_FOR_aarch64_reload_movcpv8hidi;
case E_V2SImode:
return CODE_FOR_aarch64_reload_movcpv2sidi;
case E_V4SImode:
return CODE_FOR_aarch64_reload_movcpv4sidi;
case E_V2DImode:
return CODE_FOR_aarch64_reload_movcpv2didi;
case E_V2DFmode:
return CODE_FOR_aarch64_reload_movcpv2dfdi;
default:
gcc_unreachable ();
}
gcc_unreachable ();
}
static reg_class_t
aarch64_secondary_reload (bool in_p ATTRIBUTE_UNUSED, rtx x,
reg_class_t rclass,
machine_mode mode,
secondary_reload_info *sri)
{
if (BYTES_BIG_ENDIAN
&& reg_class_subset_p (rclass, FP_REGS)
&& !((REG_P (x) && HARD_REGISTER_P (x))
|| aarch64_simd_valid_immediate (x, NULL))
&& aarch64_sve_data_mode_p (mode))
{
sri->icode = CODE_FOR_aarch64_sve_reload_be;
return NO_REGS;
}
if (MEM_P (x) && GET_CODE (x) == SYMBOL_REF && CONSTANT_POOL_ADDRESS_P (x)
&& (SCALAR_FLOAT_MODE_P (GET_MODE (x))
|| targetm.vector_mode_supported_p (GET_MODE (x)))
&& !aarch64_pcrelative_literal_loads)
{
sri->icode = aarch64_constant_pool_reload_icode (mode);
return NO_REGS;
}
if (REG_P (x) && (mode == TFmode || mode == TImode) && mode == GET_MODE (x)
&& FP_REGNUM_P (REGNO (x)) && !TARGET_SIMD
&& reg_class_subset_p (rclass, FP_REGS))
{
if (mode == TFmode)
sri->icode = CODE_FOR_aarch64_reload_movtf;
else if (mode == TImode)
sri->icode = CODE_FOR_aarch64_reload_movti;
return NO_REGS;
}
if (TARGET_FLOAT && rclass == GENERAL_REGS
&& known_eq (GET_MODE_SIZE (mode), 16) && MEM_P (x))
return FP_REGS;
if (rclass == FP_REGS && (mode == TImode || mode == TFmode) && CONSTANT_P(x))
return GENERAL_REGS;
return NO_REGS;
}
static bool
aarch64_can_eliminate (const int from ATTRIBUTE_UNUSED, const int to)
{
gcc_assert (from == ARG_POINTER_REGNUM || from == FRAME_POINTER_REGNUM);
if (frame_pointer_needed)
return to == HARD_FRAME_POINTER_REGNUM;
return true;
}
poly_int64
aarch64_initial_elimination_offset (unsigned from, unsigned to)
{
aarch64_layout_frame ();
if (to == HARD_FRAME_POINTER_REGNUM)
{
if (from == ARG_POINTER_REGNUM)
return cfun->machine->frame.hard_fp_offset;
if (from == FRAME_POINTER_REGNUM)
return cfun->machine->frame.hard_fp_offset
- cfun->machine->frame.locals_offset;
}
if (to == STACK_POINTER_REGNUM)
{
if (from == FRAME_POINTER_REGNUM)
return cfun->machine->frame.frame_size
- cfun->machine->frame.locals_offset;
}
return cfun->machine->frame.frame_size;
}
rtx
aarch64_return_addr (int count, rtx frame ATTRIBUTE_UNUSED)
{
if (count != 0)
return const0_rtx;
return get_hard_reg_initial_val (Pmode, LR_REGNUM);
}
static void
aarch64_asm_trampoline_template (FILE *f)
{
if (TARGET_ILP32)
{
asm_fprintf (f, "\tldr\tw%d, .+16\n", IP1_REGNUM - R0_REGNUM);
asm_fprintf (f, "\tldr\tw%d, .+16\n", STATIC_CHAIN_REGNUM - R0_REGNUM);
}
else
{
asm_fprintf (f, "\tldr\t%s, .+16\n", reg_names [IP1_REGNUM]);
asm_fprintf (f, "\tldr\t%s, .+20\n", reg_names [STATIC_CHAIN_REGNUM]);
}
asm_fprintf (f, "\tbr\t%s\n", reg_names [IP1_REGNUM]);
assemble_aligned_integer (4, const0_rtx);
assemble_aligned_integer (POINTER_BYTES, const0_rtx);
assemble_aligned_integer (POINTER_BYTES, const0_rtx);
}
static void
aarch64_trampoline_init (rtx m_tramp, tree fndecl, rtx chain_value)
{
rtx fnaddr, mem, a_tramp;
const int tramp_code_sz = 16;
emit_block_move (m_tramp, assemble_trampoline_template (),
GEN_INT (tramp_code_sz), BLOCK_OP_NORMAL);
mem = adjust_address (m_tramp, ptr_mode, tramp_code_sz);
fnaddr = XEXP (DECL_RTL (fndecl), 0);
if (GET_MODE (fnaddr) != ptr_mode)
fnaddr = convert_memory_address (ptr_mode, fnaddr);
emit_move_insn (mem, fnaddr);
mem = adjust_address (m_tramp, ptr_mode, tramp_code_sz + POINTER_BYTES);
emit_move_insn (mem, chain_value);
a_tramp = XEXP (m_tramp, 0);
emit_library_call (gen_rtx_SYMBOL_REF (Pmode, "__clear_cache"),
LCT_NORMAL, VOIDmode, a_tramp, ptr_mode,
plus_constant (ptr_mode, a_tramp, TRAMPOLINE_SIZE),
ptr_mode);
}
static unsigned char
aarch64_class_max_nregs (reg_class_t regclass, machine_mode mode)
{
HOST_WIDE_INT lowest_size = constant_lower_bound (GET_MODE_SIZE (mode));
unsigned int nregs;
switch (regclass)
{
case TAILCALL_ADDR_REGS:
case POINTER_REGS:
case GENERAL_REGS:
case ALL_REGS:
case POINTER_AND_FP_REGS:
case FP_REGS:
case FP_LO_REGS:
if (aarch64_sve_data_mode_p (mode)
&& constant_multiple_p (GET_MODE_SIZE (mode),
BYTES_PER_SVE_VECTOR, &nregs))
return nregs;
return (aarch64_vector_data_mode_p (mode)
? CEIL (lowest_size, UNITS_PER_VREG)
: CEIL (lowest_size, UNITS_PER_WORD));
case STACK_REG:
case PR_REGS:
case PR_LO_REGS:
case PR_HI_REGS:
return 1;
case NO_REGS:
return 0;
default:
break;
}
gcc_unreachable ();
}
static reg_class_t
aarch64_preferred_reload_class (rtx x, reg_class_t regclass)
{
if (regclass == POINTER_REGS)
return GENERAL_REGS;
if (regclass == STACK_REG)
{
if (REG_P(x)
&& reg_class_subset_p (REGNO_REG_CLASS (REGNO (x)), POINTER_REGS))
return regclass;
return NO_REGS;
}
if (! reg_class_subset_p (regclass, GENERAL_REGS) && GET_CODE (x) == PLUS)
{
rtx lhs = XEXP (x, 0);
if (GET_CODE (lhs) == SUBREG)
lhs = SUBREG_REG (lhs);
gcc_assert (REG_P (lhs));
gcc_assert (reg_class_subset_p (REGNO_REG_CLASS (REGNO (lhs)),
POINTER_REGS));
return NO_REGS;
}
return regclass;
}
void
aarch64_asm_output_labelref (FILE* f, const char *name)
{
asm_fprintf (f, "%U%s", name);
}
static void
aarch64_elf_asm_constructor (rtx symbol, int priority)
{
if (priority == DEFAULT_INIT_PRIORITY)
default_ctor_section_asm_out_constructor (symbol, priority);
else
{
section *s;
char buf[23];
snprintf (buf, sizeof (buf), ".init_array.%.5u", priority);
s = get_section (buf, SECTION_WRITE | SECTION_NOTYPE, NULL);
switch_to_section (s);
assemble_align (POINTER_SIZE);
assemble_aligned_integer (POINTER_BYTES, symbol);
}
}
static void
aarch64_elf_asm_destructor (rtx symbol, int priority)
{
if (priority == DEFAULT_INIT_PRIORITY)
default_dtor_section_asm_out_destructor (symbol, priority);
else
{
section *s;
char buf[23];
snprintf (buf, sizeof (buf), ".fini_array.%.5u", priority);
s = get_section (buf, SECTION_WRITE | SECTION_NOTYPE, NULL);
switch_to_section (s);
assemble_align (POINTER_SIZE);
assemble_aligned_integer (POINTER_BYTES, symbol);
}
}
const char*
aarch64_output_casesi (rtx *operands)
{
char buf[100];
char label[100];
rtx diff_vec = PATTERN (NEXT_INSN (as_a <rtx_insn *> (operands[2])));
int index;
static const char *const patterns[4][2] =
{
{
"ldrb\t%w3, [%0,%w1,uxtw]",
"add\t%3, %4, %w3, sxtb #2"
},
{
"ldrh\t%w3, [%0,%w1,uxtw #1]",
"add\t%3, %4, %w3, sxth #2"
},
{
"ldr\t%w3, [%0,%w1,uxtw #2]",
"add\t%3, %4, %w3, sxtw #2"
},
{
"ldr\t%w3, [%0,%w1,uxtw #2]",
"add\t%3, %4, %w3, sxtw #2"
}
};
gcc_assert (GET_CODE (diff_vec) == ADDR_DIFF_VEC);
scalar_int_mode mode = as_a <scalar_int_mode> (GET_MODE (diff_vec));
index = exact_log2 (GET_MODE_SIZE (mode));
gcc_assert (index >= 0 && index <= 3);
output_asm_insn (patterns[index][0], operands);
ASM_GENERATE_INTERNAL_LABEL (label, "Lrtx", CODE_LABEL_NUMBER (operands[2]));
snprintf (buf, sizeof (buf),
"adr\t%%4, %s", targetm.strip_name_encoding (label));
output_asm_insn (buf, operands);
output_asm_insn (patterns[index][1], operands);
output_asm_insn ("br\t%3", operands);
assemble_label (asm_out_file, label);
return "";
}
int
aarch64_uxt_size (int shift, HOST_WIDE_INT mask)
{
if (shift >= 0 && shift <= 3)
{
int size;
for (size = 8; size <= 32; size *= 2)
{
HOST_WIDE_INT bits = ((HOST_WIDE_INT)1U << size) - 1;
if (mask == bits << shift)
return size;
}
}
return 0;
}
static inline bool
aarch64_can_use_per_function_literal_pools_p (void)
{
return (aarch64_pcrelative_literal_loads
|| aarch64_cmodel == AARCH64_CMODEL_LARGE);
}
static bool
aarch64_use_blocks_for_constant_p (machine_mode, const_rtx)
{
return !aarch64_can_use_per_function_literal_pools_p ();
}
static section *
aarch64_select_rtx_section (machine_mode mode,
rtx x,
unsigned HOST_WIDE_INT align)
{
if (aarch64_can_use_per_function_literal_pools_p ())
return function_section (current_function_decl);
return default_elf_select_rtx_section (mode, x, align);
}
void
aarch64_asm_output_pool_epilogue (FILE *f, const char *, tree,
HOST_WIDE_INT offset)
{
if ((offset & 3) && aarch64_can_use_per_function_literal_pools_p ())
ASM_OUTPUT_ALIGN (f, 2);
}
static rtx
aarch64_strip_shift (rtx x)
{
rtx op = x;
if ((GET_CODE (op) == ASHIFT
|| GET_CODE (op) == ASHIFTRT
|| GET_CODE (op) == LSHIFTRT
|| GET_CODE (op) == ROTATERT
|| GET_CODE (op) == ROTATE)
&& CONST_INT_P (XEXP (op, 1)))
return XEXP (op, 0);
if (GET_CODE (op) == MULT
&& CONST_INT_P (XEXP (op, 1))
&& ((unsigned) exact_log2 (INTVAL (XEXP (op, 1)))) < 64)
return XEXP (op, 0);
return x;
}
static rtx
aarch64_strip_extend (rtx x, bool strip_shift)
{
scalar_int_mode mode;
rtx op = x;
if (!is_a <scalar_int_mode> (GET_MODE (op), &mode))
return op;
if ((GET_CODE (op) == ZERO_EXTRACT || GET_CODE (op) == SIGN_EXTRACT)
&& XEXP (op, 2) == const0_rtx
&& GET_CODE (XEXP (op, 0)) == MULT
&& aarch64_is_extend_from_extract (mode, XEXP (XEXP (op, 0), 1),
XEXP (op, 1)))
return XEXP (XEXP (op, 0), 0);
if (GET_CODE (op) == AND
&& GET_CODE (XEXP (op, 0)) == MULT
&& CONST_INT_P (XEXP (XEXP (op, 0), 1))
&& CONST_INT_P (XEXP (op, 1))
&& aarch64_uxt_size (exact_log2 (INTVAL (XEXP (XEXP (op, 0), 1))),
INTVAL (XEXP (op, 1))) != 0)
return XEXP (XEXP (op, 0), 0);
if (strip_shift
&& GET_CODE (op) == ASHIFT
&& CONST_INT_P (XEXP (op, 1))
&& ((unsigned HOST_WIDE_INT) INTVAL (XEXP (op, 1))) <= 4)
op = XEXP (op, 0);
if (GET_CODE (op) == ZERO_EXTEND
|| GET_CODE (op) == SIGN_EXTEND)
op = XEXP (op, 0);
if (op != x)
return op;
return x;
}
static bool
aarch64_shift_p (enum rtx_code code)
{
return code == ASHIFT || code == ASHIFTRT || code == LSHIFTRT;
}
static bool
aarch64_cheap_mult_shift_p (rtx x)
{
rtx op0, op1;
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (!(aarch64_tune_params.extra_tuning_flags
& AARCH64_EXTRA_TUNE_CHEAP_SHIFT_EXTEND))
return false;
if (GET_CODE (op0) == SIGN_EXTEND)
return false;
if (GET_CODE (x) == ASHIFT && CONST_INT_P (op1)
&& UINTVAL (op1) <= 4)
return true;
if (GET_CODE (x) != MULT || !CONST_INT_P (op1))
return false;
HOST_WIDE_INT l2 = exact_log2 (INTVAL (op1));
if (l2 > 0 && l2 <= 4)
return true;
return false;
}
static int
aarch64_rtx_mult_cost (rtx x, enum rtx_code code, int outer, bool speed)
{
rtx op0, op1;
const struct cpu_cost_table *extra_cost
= aarch64_tune_params.insn_extra_cost;
int cost = 0;
bool compound_p = (outer == PLUS || outer == MINUS);
machine_mode mode = GET_MODE (x);
gcc_checking_assert (code == MULT);
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (VECTOR_MODE_P (mode))
mode = GET_MODE_INNER (mode);
if (GET_MODE_CLASS (mode) == MODE_INT)
{
if (aarch64_shift_p (GET_CODE (x))
|| (CONST_INT_P (op1)
&& exact_log2 (INTVAL (op1)) > 0))
{
bool is_extend = GET_CODE (op0) == ZERO_EXTEND
|| GET_CODE (op0) == SIGN_EXTEND;
if (speed)
{
if (compound_p)
{
if (aarch64_cheap_mult_shift_p (x))
;
else if (REG_P (op1))
cost += extra_cost->alu.arith_shift_reg;
else if (is_extend)
cost += extra_cost->alu.extend_arith;
else
cost += extra_cost->alu.arith_shift;
}
else
cost += extra_cost->alu.shift;
}
if (is_extend)
op0 = aarch64_strip_extend (op0, true);
cost += rtx_cost (op0, VOIDmode, code, 0, speed);
return cost;
}
if (GET_CODE (op0) == NEG)
{
op0 = XEXP (op0, 0);
compound_p = true;
}
if ((GET_CODE (op0) == ZERO_EXTEND
&& GET_CODE (op1) == ZERO_EXTEND)
|| (GET_CODE (op0) == SIGN_EXTEND
&& GET_CODE (op1) == SIGN_EXTEND))
{
cost += rtx_cost (XEXP (op0, 0), VOIDmode, MULT, 0, speed);
cost += rtx_cost (XEXP (op1, 0), VOIDmode, MULT, 1, speed);
if (speed)
{
if (compound_p)
cost += extra_cost->mult[0].extend_add;
else
cost += extra_cost->mult[0].extend;
}
return cost;
}
cost += rtx_cost (op0, mode, MULT, 0, speed);
cost += rtx_cost (op1, mode, MULT, 1, speed);
if (speed)
{
if (compound_p)
cost += extra_cost->mult[mode == DImode].add;
else
cost += extra_cost->mult[mode == DImode].simple;
}
return cost;
}
else
{
if (speed)
{
bool neg0 = GET_CODE (op0) == NEG;
bool neg1 = GET_CODE (op1) == NEG;
if (compound_p || !flag_rounding_math || (neg0 && neg1))
{
if (neg0)
op0 = XEXP (op0, 0);
if (neg1)
op1 = XEXP (op1, 0);
}
if (compound_p)
cost += extra_cost->fp[mode == DFmode].fma;
else
cost += extra_cost->fp[mode == DFmode].mult;
}
cost += rtx_cost (op0, mode, MULT, 0, speed);
cost += rtx_cost (op1, mode, MULT, 1, speed);
return cost;
}
}
static int
aarch64_address_cost (rtx x,
machine_mode mode,
addr_space_t as ATTRIBUTE_UNUSED,
bool speed)
{
enum rtx_code c = GET_CODE (x);
const struct cpu_addrcost_table *addr_cost = aarch64_tune_params.addr_cost;
struct aarch64_address_info info;
int cost = 0;
info.shift = 0;
if (!aarch64_classify_address (&info, x, mode, false))
{
if (GET_CODE (x) == CONST || GET_CODE (x) == SYMBOL_REF)
{
int cost_symbol_ref = rtx_cost (x, Pmode, MEM, 1, speed);
cost_symbol_ref /= COSTS_N_INSNS (1);
return cost_symbol_ref + addr_cost->imm_offset;
}
else
{
return addr_cost->register_offset;
}
}
switch (info.type)
{
case ADDRESS_LO_SUM:
case ADDRESS_SYMBOLIC:
case ADDRESS_REG_IMM:
cost += addr_cost->imm_offset;
break;
case ADDRESS_REG_WB:
if (c == PRE_INC || c == PRE_DEC || c == PRE_MODIFY)
cost += addr_cost->pre_modify;
else if (c == POST_INC || c == POST_DEC || c == POST_MODIFY)
cost += addr_cost->post_modify;
else
gcc_unreachable ();
break;
case ADDRESS_REG_REG:
cost += addr_cost->register_offset;
break;
case ADDRESS_REG_SXTW:
cost += addr_cost->register_sextend;
break;
case ADDRESS_REG_UXTW:
cost += addr_cost->register_zextend;
break;
default:
gcc_unreachable ();
}
if (info.shift > 0)
{
if (known_eq (GET_MODE_BITSIZE (mode), 16))
cost += addr_cost->addr_scale_costs.hi;
else if (known_eq (GET_MODE_BITSIZE (mode), 32))
cost += addr_cost->addr_scale_costs.si;
else if (known_eq (GET_MODE_BITSIZE (mode), 64))
cost += addr_cost->addr_scale_costs.di;
else
cost += addr_cost->addr_scale_costs.ti;
}
return cost;
}
int
aarch64_branch_cost (bool speed_p, bool predictable_p)
{
const struct cpu_branch_cost *branch_costs =
aarch64_tune_params.branch_costs;
if (!speed_p || predictable_p)
return branch_costs->predictable;
else
return branch_costs->unpredictable;
}
static bool
aarch64_rtx_arith_op_extract_p (rtx x, scalar_int_mode mode)
{
if (GET_CODE (x) == SIGN_EXTRACT
|| GET_CODE (x) == ZERO_EXTRACT)
{
rtx op0 = XEXP (x, 0);
rtx op1 = XEXP (x, 1);
rtx op2 = XEXP (x, 2);
if (GET_CODE (op0) == MULT
&& CONST_INT_P (op1)
&& op2 == const0_rtx
&& CONST_INT_P (XEXP (op0, 1))
&& aarch64_is_extend_from_extract (mode,
XEXP (op0, 1),
op1))
{
return true;
}
}
else if (GET_CODE (x) == SIGN_EXTEND
|| GET_CODE (x) == ZERO_EXTEND)
return REG_P (XEXP (x, 0));
return false;
}
static bool
aarch64_frint_unspec_p (unsigned int u)
{
switch (u)
{
case UNSPEC_FRINTZ:
case UNSPEC_FRINTP:
case UNSPEC_FRINTM:
case UNSPEC_FRINTA:
case UNSPEC_FRINTN:
case UNSPEC_FRINTX:
case UNSPEC_FRINTI:
return true;
default:
return false;
}
}
static bool
aarch64_extr_rtx_p (rtx x, rtx *res_op0, rtx *res_op1)
{
rtx op0, op1;
scalar_int_mode mode;
if (!is_a <scalar_int_mode> (GET_MODE (x), &mode))
return false;
*res_op0 = NULL_RTX;
*res_op1 = NULL_RTX;
if (GET_CODE (x) != IOR)
return false;
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if ((GET_CODE (op0) == ASHIFT && GET_CODE (op1) == LSHIFTRT)
|| (GET_CODE (op1) == ASHIFT && GET_CODE (op0) == LSHIFTRT))
{
if (GET_CODE (op1) == ASHIFT)
std::swap (op0, op1);
if (!CONST_INT_P (XEXP (op0, 1)) || !CONST_INT_P (XEXP (op1, 1)))
return false;
unsigned HOST_WIDE_INT shft_amnt_0 = UINTVAL (XEXP (op0, 1));
unsigned HOST_WIDE_INT shft_amnt_1 = UINTVAL (XEXP (op1, 1));
if (shft_amnt_0 < GET_MODE_BITSIZE (mode)
&& shft_amnt_0 + shft_amnt_1 == GET_MODE_BITSIZE (mode))
{
*res_op0 = XEXP (op0, 0);
*res_op1 = XEXP (op1, 0);
return true;
}
}
return false;
}
static bool
aarch64_if_then_else_costs (rtx op0, rtx op1, rtx op2, int *cost, bool speed)
{
rtx inner;
rtx comparator;
enum rtx_code cmpcode;
if (COMPARISON_P (op0))
{
inner = XEXP (op0, 0);
comparator = XEXP (op0, 1);
cmpcode = GET_CODE (op0);
}
else
{
inner = op0;
comparator = const0_rtx;
cmpcode = NE;
}
if (GET_CODE (op1) == PC || GET_CODE (op2) == PC)
{
if (GET_MODE_CLASS (GET_MODE (inner)) == MODE_CC)
return true;
else
{
if (cmpcode == NE || cmpcode == EQ)
{
if (comparator == const0_rtx)
{
if (GET_CODE (inner) == ZERO_EXTRACT)
*cost += rtx_cost (XEXP (inner, 0), VOIDmode,
ZERO_EXTRACT, 0, speed);
else
*cost += rtx_cost (inner, VOIDmode, cmpcode, 0, speed);
return true;
}
}
else if (cmpcode == LT || cmpcode == GE)
{
if (comparator == const0_rtx)
return true;
}
}
}
else if (GET_MODE_CLASS (GET_MODE (inner)) == MODE_CC)
{
if (GET_CODE (op1) == COMPARE)
{
if (XEXP (op1, 1) == const0_rtx)
*cost += 1;
if (speed)
{
machine_mode mode = GET_MODE (XEXP (op1, 0));
const struct cpu_cost_table *extra_cost
= aarch64_tune_params.insn_extra_cost;
if (GET_MODE_CLASS (mode) == MODE_INT)
*cost += extra_cost->alu.arith;
else
*cost += extra_cost->fp[mode == DFmode].compare;
}
return true;
}
if (GET_CODE (op1) == NEG
|| GET_CODE (op1) == NOT
|| (GET_CODE (op1) == PLUS && XEXP (op1, 1) == const1_rtx))
op1 = XEXP (op1, 0);
else if (GET_CODE (op1) == ZERO_EXTEND && GET_CODE (op2) == ZERO_EXTEND)
{
op1 = XEXP (op1, 0);
op2 = XEXP (op2, 0);
}
*cost += rtx_cost (op1, VOIDmode, IF_THEN_ELSE, 1, speed);
*cost += rtx_cost (op2, VOIDmode, IF_THEN_ELSE, 2, speed);
return true;
}
return false;
}
static rtx
aarch64_extend_bitfield_pattern_p (rtx x)
{
rtx_code outer_code = GET_CODE (x);
machine_mode outer_mode = GET_MODE (x);
if (outer_code != ZERO_EXTEND && outer_code != SIGN_EXTEND
&& outer_mode != SImode && outer_mode != DImode)
return NULL_RTX;
rtx inner = XEXP (x, 0);
rtx_code inner_code = GET_CODE (inner);
machine_mode inner_mode = GET_MODE (inner);
rtx op = NULL_RTX;
switch (inner_code)
{
case ASHIFT:
if (CONST_INT_P (XEXP (inner, 1))
&& (inner_mode == QImode || inner_mode == HImode))
op = XEXP (inner, 0);
break;
case LSHIFTRT:
if (outer_code == ZERO_EXTEND && CONST_INT_P (XEXP (inner, 1))
&& (inner_mode == QImode || inner_mode == HImode))
op = XEXP (inner, 0);
break;
case ASHIFTRT:
if (outer_code == SIGN_EXTEND && CONST_INT_P (XEXP (inner, 1))
&& (inner_mode == QImode || inner_mode == HImode))
op = XEXP (inner, 0);
break;
default:
break;
}
return op;
}
bool
aarch64_mask_and_shift_for_ubfiz_p (scalar_int_mode mode, rtx mask,
rtx shft_amnt)
{
return CONST_INT_P (mask) && CONST_INT_P (shft_amnt)
&& INTVAL (shft_amnt) < GET_MODE_BITSIZE (mode)
&& exact_log2 ((INTVAL (mask) >> INTVAL (shft_amnt)) + 1) >= 0
&& (INTVAL (mask)
& ((HOST_WIDE_INT_1U << INTVAL (shft_amnt)) - 1)) == 0;
}
static bool
aarch64_rtx_costs (rtx x, machine_mode mode, int outer ATTRIBUTE_UNUSED,
int param ATTRIBUTE_UNUSED, int *cost, bool speed)
{
rtx op0, op1, op2;
const struct cpu_cost_table *extra_cost
= aarch64_tune_params.insn_extra_cost;
int code = GET_CODE (x);
scalar_int_mode int_mode;
*cost = COSTS_N_INSNS (1);
switch (code)
{
case SET:
*cost = 0;
op0 = SET_DEST (x);
op1 = SET_SRC (x);
switch (GET_CODE (op0))
{
case MEM:
if (speed)
{
rtx address = XEXP (op0, 0);
if (VECTOR_MODE_P (mode))
*cost += extra_cost->ldst.storev;
else if (GET_MODE_CLASS (mode) == MODE_INT)
*cost += extra_cost->ldst.store;
else if (mode == SFmode)
*cost += extra_cost->ldst.storef;
else if (mode == DFmode)
*cost += extra_cost->ldst.stored;
*cost +=
COSTS_N_INSNS (aarch64_address_cost (address, mode,
0, speed));
}
*cost += rtx_cost (op1, mode, SET, 1, speed);
return true;
case SUBREG:
if (! REG_P (SUBREG_REG (op0)))
*cost += rtx_cost (SUBREG_REG (op0), VOIDmode, SET, 0, speed);
case REG:
if (VECTOR_MODE_P (GET_MODE (op0)) && REG_P (op1))
{
int nregs = aarch64_hard_regno_nregs (V0_REGNUM, GET_MODE (op0));
*cost = COSTS_N_INSNS (nregs);
}
else if (REG_P (op1) || op1 == const0_rtx)
{
int nregs = aarch64_hard_regno_nregs (R0_REGNUM, GET_MODE (op0));
*cost = COSTS_N_INSNS (nregs);
}
else
*cost += rtx_cost (op1, mode, SET, 1, speed);
return true;
case ZERO_EXTRACT:
case SIGN_EXTRACT:
if (GET_CODE (op1) == SUBREG)
op1 = SUBREG_REG (op1);
if ((GET_CODE (op1) == ZERO_EXTEND
|| GET_CODE (op1) == SIGN_EXTEND)
&& CONST_INT_P (XEXP (op0, 1))
&& is_a <scalar_int_mode> (GET_MODE (XEXP (op1, 0)), &int_mode)
&& GET_MODE_BITSIZE (int_mode) >= INTVAL (XEXP (op0, 1)))
op1 = XEXP (op1, 0);
if (CONST_INT_P (op1))
{
*cost = COSTS_N_INSNS (1);
}
else
{
if (speed)
*cost += extra_cost->alu.bfi;
*cost += rtx_cost (op1, VOIDmode, (enum rtx_code) code, 1, speed);
}
return true;
default:
*cost = COSTS_N_INSNS (1);
return false;
}
return false;
case CONST_INT:
if (x == const0_rtx)
*cost = 0;
else
{
if (!is_a <scalar_int_mode> (mode, &int_mode))
int_mode = word_mode;
*cost = COSTS_N_INSNS (aarch64_internal_mov_immediate
(NULL_RTX, x, false, int_mode));
}
return true;
case CONST_DOUBLE:
if (!aarch64_float_const_representable_p (x)
&& !aarch64_can_const_movi_rtx_p (x, mode)
&& aarch64_float_const_rtx_p (x))
{
unsigned HOST_WIDE_INT ival;
bool succeed = aarch64_reinterpret_float_as_int (x, &ival);
gcc_assert (succeed);
scalar_int_mode imode = (mode == HFmode
? SImode
: int_mode_for_mode (mode).require ());
int ncost = aarch64_internal_mov_immediate
(NULL_RTX, gen_int_mode (ival, imode), false, imode);
*cost += COSTS_N_INSNS (ncost);
return true;
}
if (speed)
{
if (aarch64_float_const_representable_p (x))
*cost += extra_cost->fp[mode == DFmode].fpconst;
else if (!aarch64_float_const_zero_rtx_p (x))
{
if (mode == DFmode)
*cost += extra_cost->ldst.loadd;
else
*cost += extra_cost->ldst.loadf;
}
else
{
}
}
return true;
case MEM:
if (speed)
{
rtx address = XEXP (x, 0);
if (VECTOR_MODE_P (mode))
*cost += extra_cost->ldst.loadv;
else if (GET_MODE_CLASS (mode) == MODE_INT)
*cost += extra_cost->ldst.load;
else if (mode == SFmode)
*cost += extra_cost->ldst.loadf;
else if (mode == DFmode)
*cost += extra_cost->ldst.loadd;
*cost +=
COSTS_N_INSNS (aarch64_address_cost (address, mode,
0, speed));
}
return true;
case NEG:
op0 = XEXP (x, 0);
if (VECTOR_MODE_P (mode))
{
if (speed)
{
*cost += extra_cost->vect.alu;
}
return false;
}
if (GET_MODE_CLASS (mode) == MODE_INT)
{
if (GET_RTX_CLASS (GET_CODE (op0)) == RTX_COMPARE
|| GET_RTX_CLASS (GET_CODE (op0)) == RTX_COMM_COMPARE)
{
*cost += rtx_cost (XEXP (op0, 0), VOIDmode, NEG, 0, speed);
return true;
}
op0 = CONST0_RTX (mode);
op1 = XEXP (x, 0);
goto cost_minus;
}
if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
if (GET_CODE (op0) == FMA && !HONOR_SIGNED_ZEROS (GET_MODE (op0)))
{
*cost = rtx_cost (op0, mode, NEG, 0, speed);
return true;
}
if (GET_CODE (op0) == MULT)
{
*cost = rtx_cost (op0, mode, NEG, 0, speed);
return true;
}
if (speed)
*cost += extra_cost->fp[mode == DFmode].neg;
return false;
}
return false;
case CLRSB:
case CLZ:
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.clz;
}
return false;
case COMPARE:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (op1 == const0_rtx
&& GET_CODE (op0) == AND)
{
x = op0;
mode = GET_MODE (op0);
goto cost_logic;
}
if (GET_MODE_CLASS (GET_MODE (op0)) == MODE_INT)
{
mode = GET_MODE (op0);
if (GET_CODE (op0) == AND)
{
x = op0;
goto cost_logic;
}
if (GET_CODE (op0) == PLUS)
{
x = op0;
goto cost_plus;
}
if (GET_CODE (op0) == MINUS)
{
x = op0;
goto cost_minus;
}
if (GET_CODE (op0) == ZERO_EXTRACT && op1 == const0_rtx
&& GET_MODE (x) == CC_NZmode && CONST_INT_P (XEXP (op0, 1))
&& CONST_INT_P (XEXP (op0, 2)))
{
if (speed)
*cost += extra_cost->alu.logical;
*cost += rtx_cost (XEXP (op0, 0), GET_MODE (op0),
ZERO_EXTRACT, 0, speed);
return true;
}
if (GET_CODE (op1) == NEG)
{
if (speed)
*cost += extra_cost->alu.arith;
*cost += rtx_cost (op0, mode, COMPARE, 0, speed);
*cost += rtx_cost (XEXP (op1, 0), mode, NEG, 1, speed);
return true;
}
if (! (REG_P (op0)
|| (GET_CODE (op0) == SUBREG && REG_P (SUBREG_REG (op0)))))
{
op0 = XEXP (x, 1);
op1 = XEXP (x, 0);
}
goto cost_minus;
}
if (GET_MODE_CLASS (GET_MODE (op0)) == MODE_FLOAT)
{
if (speed)
*cost += extra_cost->fp[mode == DFmode].compare;
if (CONST_DOUBLE_P (op1) && aarch64_float_const_zero_rtx_p (op1))
{
*cost += rtx_cost (op0, VOIDmode, COMPARE, 0, speed);
return true;
}
return false;
}
if (VECTOR_MODE_P (mode))
{
if (speed)
*cost += extra_cost->vect.alu;
if (aarch64_float_const_zero_rtx_p (op1))
{
return true;
}
return false;
}
return false;
case MINUS:
{
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
cost_minus:
*cost += rtx_cost (op0, mode, MINUS, 0, speed);
if ((GET_MODE_CLASS (mode) == MODE_INT
|| (GET_MODE_CLASS (mode) == MODE_CC
&& GET_MODE_CLASS (GET_MODE (op0)) == MODE_INT))
&& CONST_INT_P (op1)
&& aarch64_uimm12_shift (INTVAL (op1)))
{
if (speed)
*cost += extra_cost->alu.arith;
return true;
}
if (is_a <scalar_int_mode> (mode, &int_mode)
&& aarch64_rtx_arith_op_extract_p (op1, int_mode))
{
if (speed)
*cost += extra_cost->alu.extend_arith;
op1 = aarch64_strip_extend (op1, true);
*cost += rtx_cost (op1, VOIDmode,
(enum rtx_code) GET_CODE (op1), 0, speed);
return true;
}
rtx new_op1 = aarch64_strip_extend (op1, false);
if ((GET_CODE (new_op1) == MULT
|| aarch64_shift_p (GET_CODE (new_op1)))
&& code != COMPARE)
{
*cost += aarch64_rtx_mult_cost (new_op1, MULT,
(enum rtx_code) code,
speed);
return true;
}
*cost += rtx_cost (new_op1, VOIDmode, MINUS, 1, speed);
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else if (GET_MODE_CLASS (mode) == MODE_INT)
{
*cost += extra_cost->alu.arith;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost += extra_cost->fp[mode == DFmode].addsub;
}
}
return true;
}
case PLUS:
{
rtx new_op0;
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
cost_plus:
if (GET_RTX_CLASS (GET_CODE (op0)) == RTX_COMPARE
|| GET_RTX_CLASS (GET_CODE (op0)) == RTX_COMM_COMPARE)
{
*cost += rtx_cost (XEXP (op0, 0), mode, PLUS, 0, speed);
*cost += rtx_cost (op1, mode, PLUS, 1, speed);
return true;
}
if (GET_MODE_CLASS (mode) == MODE_INT
&& ((CONST_INT_P (op1) && aarch64_uimm12_shift (INTVAL (op1)))
|| aarch64_sve_addvl_addpl_immediate (op1, mode)))
{
*cost += rtx_cost (op0, mode, PLUS, 0, speed);
if (speed)
*cost += extra_cost->alu.arith;
return true;
}
*cost += rtx_cost (op1, mode, PLUS, 1, speed);
if (is_a <scalar_int_mode> (mode, &int_mode)
&& aarch64_rtx_arith_op_extract_p (op0, int_mode))
{
if (speed)
*cost += extra_cost->alu.extend_arith;
op0 = aarch64_strip_extend (op0, true);
*cost += rtx_cost (op0, VOIDmode,
(enum rtx_code) GET_CODE (op0), 0, speed);
return true;
}
new_op0 = aarch64_strip_extend (op0, false);
if (GET_CODE (new_op0) == MULT
|| aarch64_shift_p (GET_CODE (new_op0)))
{
*cost += aarch64_rtx_mult_cost (new_op0, MULT, PLUS,
speed);
return true;
}
*cost += rtx_cost (new_op0, VOIDmode, PLUS, 0, speed);
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else if (GET_MODE_CLASS (mode) == MODE_INT)
{
*cost += extra_cost->alu.arith;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost += extra_cost->fp[mode == DFmode].addsub;
}
}
return true;
}
case BSWAP:
*cost = COSTS_N_INSNS (1);
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.rev;
}
return false;
case IOR:
if (aarch_rev16_p (x))
{
*cost = COSTS_N_INSNS (1);
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.rev;
}
return true;
}
if (aarch64_extr_rtx_p (x, &op0, &op1))
{
*cost += rtx_cost (op0, mode, IOR, 0, speed);
*cost += rtx_cost (op1, mode, IOR, 1, speed);
if (speed)
*cost += extra_cost->alu.shift;
return true;
}
case XOR:
case AND:
cost_logic:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (VECTOR_MODE_P (mode))
{
if (speed)
*cost += extra_cost->vect.alu;
return true;
}
if (code == AND
&& GET_CODE (op0) == MULT
&& CONST_INT_P (XEXP (op0, 1))
&& CONST_INT_P (op1)
&& aarch64_uxt_size (exact_log2 (INTVAL (XEXP (op0, 1))),
INTVAL (op1)) != 0)
{
*cost += rtx_cost (XEXP (op0, 0), mode, ZERO_EXTRACT, 0, speed);
if (speed)
*cost += extra_cost->alu.bfx;
return true;
}
if (is_int_mode (mode, &int_mode))
{
if (CONST_INT_P (op1))
{
if (GET_CODE (op0) == ASHIFT
&& aarch64_mask_and_shift_for_ubfiz_p (int_mode, op1,
XEXP (op0, 1)))
{
*cost += rtx_cost (XEXP (op0, 0), int_mode,
(enum rtx_code) code, 0, speed);
if (speed)
*cost += extra_cost->alu.bfx;
return true;
}
else if (aarch64_bitmask_imm (INTVAL (op1), int_mode))
{
*cost += rtx_cost (op0, int_mode,
(enum rtx_code) code, 0, speed);
if (speed)
*cost += extra_cost->alu.logical;
return true;
}
}
else
{
rtx new_op0 = op0;
if (GET_CODE (op0) == NOT)
op0 = XEXP (op0, 0);
new_op0 = aarch64_strip_shift (op0);
if (speed)
{
if (new_op0 != op0)
{
if (CONST_INT_P (XEXP (op0, 1)))
*cost += extra_cost->alu.log_shift;
else
*cost += extra_cost->alu.log_shift_reg;
}
else
*cost += extra_cost->alu.logical;
}
*cost += rtx_cost (new_op0, int_mode, (enum rtx_code) code,
0, speed);
*cost += rtx_cost (op1, int_mode, (enum rtx_code) code,
1, speed);
return true;
}
}
return false;
case NOT:
x = XEXP (x, 0);
op0 = aarch64_strip_shift (x);
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
return false;
}
if (op0 != x)
{
*cost += rtx_cost (op0, mode, (enum rtx_code) code, 0, speed);
if (speed)
*cost += extra_cost->alu.log_shift;
return true;
}
else if (GET_CODE (op0) == XOR)
{
rtx newop0 = XEXP (op0, 0);
rtx newop1 = XEXP (op0, 1);
rtx op0_stripped = aarch64_strip_shift (newop0);
*cost += rtx_cost (newop1, mode, (enum rtx_code) code, 1, speed);
*cost += rtx_cost (op0_stripped, mode, XOR, 0, speed);
if (speed)
{
if (op0_stripped != newop0)
*cost += extra_cost->alu.log_shift;
else
*cost += extra_cost->alu.logical;
}
return true;
}
if (speed)
*cost += extra_cost->alu.logical;
return false;
case ZERO_EXTEND:
op0 = XEXP (x, 0);
if (mode == DImode
&& GET_MODE (op0) == SImode
&& outer == SET)
{
int op_cost = rtx_cost (op0, VOIDmode, ZERO_EXTEND, 0, speed);
if (op_cost)
*cost = op_cost;
return true;
}
else if (MEM_P (op0))
{
*cost = rtx_cost (op0, VOIDmode, ZERO_EXTEND, param, speed);
return true;
}
op0 = aarch64_extend_bitfield_pattern_p (x);
if (op0)
{
*cost += rtx_cost (op0, mode, ZERO_EXTEND, 0, speed);
if (speed)
*cost += extra_cost->alu.bfx;
return true;
}
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else
{
*cost += extra_cost->alu.logical;
}
}
return false;
case SIGN_EXTEND:
if (MEM_P (XEXP (x, 0)))
{
if (speed)
{
rtx address = XEXP (XEXP (x, 0), 0);
*cost += extra_cost->ldst.load_sign_extend;
*cost +=
COSTS_N_INSNS (aarch64_address_cost (address, mode,
0, speed));
}
return true;
}
op0 = aarch64_extend_bitfield_pattern_p (x);
if (op0)
{
*cost += rtx_cost (op0, mode, SIGN_EXTEND, 0, speed);
if (speed)
*cost += extra_cost->alu.bfx;
return true;
}
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.extend;
}
return false;
case ASHIFT:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (CONST_INT_P (op1))
{
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else
{
*cost += extra_cost->alu.shift;
}
}
if (GET_CODE (op0) == ZERO_EXTEND
|| GET_CODE (op0) == SIGN_EXTEND)
op0 = XEXP (op0, 0);
*cost += rtx_cost (op0, VOIDmode, ASHIFT, 0, speed);
return true;
}
else
{
if (VECTOR_MODE_P (mode))
{
if (speed)
*cost += extra_cost->vect.alu;
}
else
{
if (speed)
*cost += extra_cost->alu.shift_reg;
if (GET_CODE (op1) == AND && REG_P (XEXP (op1, 0))
&& CONST_INT_P (XEXP (op1, 1))
&& known_eq (INTVAL (XEXP (op1, 1)),
GET_MODE_BITSIZE (mode) - 1))
{
*cost += rtx_cost (op0, mode, (rtx_code) code, 0, speed);
return true;
}
}
return false;  
}
case ROTATE:
case ROTATERT:
case LSHIFTRT:
case ASHIFTRT:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
if (CONST_INT_P (op1))
{
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.shift;
}
*cost += rtx_cost (op0, mode, (enum rtx_code) code, 0, speed);
return true;
}
else
{
if (VECTOR_MODE_P (mode))
{
if (speed)
*cost += extra_cost->vect.alu;
}
else
{
if (speed)
*cost += extra_cost->alu.shift_reg;
if (GET_CODE (op1) == AND && REG_P (XEXP (op1, 0))
&& CONST_INT_P (XEXP (op1, 1))
&& known_eq (INTVAL (XEXP (op1, 1)),
GET_MODE_BITSIZE (mode) - 1))
{
*cost += rtx_cost (op0, mode, (rtx_code) code, 0, speed);
return true;
}
}
return false;  
}
case SYMBOL_REF:
if (aarch64_cmodel == AARCH64_CMODEL_LARGE
|| aarch64_cmodel == AARCH64_CMODEL_SMALL_SPIC)
{
if (speed)
*cost += extra_cost->ldst.load;
}
else if (aarch64_cmodel == AARCH64_CMODEL_SMALL
|| aarch64_cmodel == AARCH64_CMODEL_SMALL_PIC)
{
*cost += COSTS_N_INSNS (1);
if (speed)
*cost += 2 * extra_cost->alu.arith;
}
else if (aarch64_cmodel == AARCH64_CMODEL_TINY
|| aarch64_cmodel == AARCH64_CMODEL_TINY_PIC)
{
if (speed)
*cost += extra_cost->alu.arith;
}
if (flag_pic)
{
*cost += COSTS_N_INSNS (1);
if (speed)
*cost += extra_cost->ldst.load;
}
return true;
case HIGH:
case LO_SUM:
if (speed)
*cost += extra_cost->alu.arith;
return true;
case ZERO_EXTRACT:
case SIGN_EXTRACT:
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->alu.bfx;
}
*cost += rtx_cost (XEXP (x, 0), VOIDmode, (enum rtx_code) code, 0, speed);
return true;
case MULT:
*cost += aarch64_rtx_mult_cost (x, MULT, 0, speed);
return true;
case MOD:
if (CONST_INT_P (XEXP (x, 1))
&& exact_log2 (INTVAL (XEXP (x, 1))) > 0
&& (mode == SImode || mode == DImode))
{
*cost = COSTS_N_INSNS (4);
if (speed)
*cost += 2 * extra_cost->alu.logical
+ 2 * extra_cost->alu.arith;
return true;
}
case UMOD:
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else if (GET_MODE_CLASS (mode) == MODE_INT)
*cost += (extra_cost->mult[mode == DImode].add
+ extra_cost->mult[mode == DImode].idiv
+ (code == MOD ? 1 : 0));
}
return false;  
case DIV:
case UDIV:
case SQRT:
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else if (GET_MODE_CLASS (mode) == MODE_INT)
*cost += (extra_cost->mult[mode == DImode].idiv
+ (code == DIV ? 1 : 0));
else
*cost += extra_cost->fp[mode == DFmode].div;
}
return false;  
case IF_THEN_ELSE:
return aarch64_if_then_else_costs (XEXP (x, 0), XEXP (x, 1),
XEXP (x, 2), cost, speed);
case EQ:
case NE:
case GT:
case GTU:
case LT:
case LTU:
case GE:
case GEU:
case LE:
case LEU:
return false; 
case FMA:
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
op2 = XEXP (x, 2);
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->fp[mode == DFmode].fma;
}
if (GET_CODE (op0) == NEG)
op0 = XEXP (op0, 0);
if (GET_CODE (op2) == NEG)
op2 = XEXP (op2, 0);
if (GET_CODE (op1) == NEG)
op1 = XEXP (op1, 0);
if (GET_CODE (op0) == VEC_DUPLICATE)
op0 = XEXP (op0, 0);
else if (GET_CODE (op1) == VEC_DUPLICATE)
op1 = XEXP (op1, 0);
if (GET_CODE (op0) == VEC_SELECT)
op0 = XEXP (op0, 0);
else if (GET_CODE (op1) == VEC_SELECT)
op1 = XEXP (op1, 0);
*cost += rtx_cost (op0, mode, FMA, 0, speed);
*cost += rtx_cost (op1, mode, FMA, 1, speed);
*cost += rtx_cost (op2, mode, FMA, 2, speed);
return true;
case FLOAT:
case UNSIGNED_FLOAT:
if (speed)
*cost += extra_cost->fp[mode == DFmode].fromint;
return false;
case FLOAT_EXTEND:
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else
*cost += extra_cost->fp[mode == DFmode].widen;
}
return false;
case FLOAT_TRUNCATE:
if (speed)
{
if (VECTOR_MODE_P (mode))
{
*cost += extra_cost->vect.alu;
}
else
*cost += extra_cost->fp[mode == DFmode].narrow;
}
return false;
case FIX:
case UNSIGNED_FIX:
x = XEXP (x, 0);
if (GET_CODE (x) == UNSPEC)
{
unsigned int uns_code = XINT (x, 1);
if (uns_code == UNSPEC_FRINTA
|| uns_code == UNSPEC_FRINTM
|| uns_code == UNSPEC_FRINTN
|| uns_code == UNSPEC_FRINTP
|| uns_code == UNSPEC_FRINTZ)
x = XVECEXP (x, 0, 0);
}
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
*cost += extra_cost->fp[GET_MODE (x) == DFmode].toint;
}
if (GET_CODE (x) == MULT
&& ((VECTOR_MODE_P (mode)
&& aarch64_vec_fpconst_pow_of_2 (XEXP (x, 1)) > 0)
|| aarch64_fpconst_pow_of_2 (XEXP (x, 1)) > 0))
{
*cost += rtx_cost (XEXP (x, 0), VOIDmode, (rtx_code) code,
0, speed);
return true;
}
*cost += rtx_cost (x, VOIDmode, (enum rtx_code) code, 0, speed);
return true;
case ABS:
if (VECTOR_MODE_P (mode))
{
if (speed)
*cost += extra_cost->vect.alu;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
op0 = XEXP (x, 0);
if (GET_CODE (op0) == MINUS)
{
*cost += rtx_cost (XEXP (op0, 0), mode, MINUS, 0, speed);
*cost += rtx_cost (XEXP (op0, 1), mode, MINUS, 1, speed);
if (speed)
*cost += extra_cost->fp[mode == DFmode].addsub;
return true;
}
if (speed)
*cost += extra_cost->fp[mode == DFmode].neg;
}
else
{
*cost = COSTS_N_INSNS (2);
if (speed)
*cost += 2 * extra_cost->alu.arith;
}
return false;
case SMAX:
case SMIN:
if (speed)
{
if (VECTOR_MODE_P (mode))
*cost += extra_cost->vect.alu;
else
{
*cost += extra_cost->fp[mode == DFmode].addsub;
}
}
return false;
case UNSPEC:
if (aarch64_frint_unspec_p (XINT (x, 1)))
{
if (speed)
*cost += extra_cost->fp[mode == DFmode].roundint;
return false;
}
if (XINT (x, 1) == UNSPEC_RBIT)
{
if (speed)
*cost += extra_cost->alu.rev;
return false;
}
break;
case TRUNCATE:
if (
mode == DImode
&& GET_MODE (XEXP (x, 0)) == TImode
&& GET_CODE (XEXP (x, 0)) == LSHIFTRT
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == MULT
&& ((GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 0)) == ZERO_EXTEND
&& GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 1)) == ZERO_EXTEND)
|| (GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 0)) == SIGN_EXTEND
&& GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 1)) == SIGN_EXTEND))
&& GET_MODE (XEXP (XEXP (XEXP (XEXP (x, 0), 0), 0), 0)) == DImode
&& GET_MODE (XEXP (XEXP (XEXP (XEXP (x, 0), 0), 1), 0)) == DImode
&& CONST_INT_P (XEXP (XEXP (x, 0), 1))
&& UINTVAL (XEXP (XEXP (x, 0), 1)) == 64)
{
if (speed)
*cost += extra_cost->mult[mode == DImode].extend;
*cost += rtx_cost (XEXP (XEXP (XEXP (XEXP (x, 0), 0), 0), 0),
mode, MULT, 0, speed);
*cost += rtx_cost (XEXP (XEXP (XEXP (XEXP (x, 0), 0), 1), 0),
mode, MULT, 1, speed);
return true;
}
default:
break;
}
if (dump_file
&& flag_aarch64_verbose_cost)
fprintf (dump_file,
"\nFailed to cost RTX.  Assuming default cost.\n");
return true;
}
static bool
aarch64_rtx_costs_wrapper (rtx x, machine_mode mode, int outer,
int param, int *cost, bool speed)
{
bool result = aarch64_rtx_costs (x, mode, outer, param, cost, speed);
if (dump_file
&& flag_aarch64_verbose_cost)
{
print_rtl_single (dump_file, x);
fprintf (dump_file, "\n%s cost: %d (%s)\n",
speed ? "Hot" : "Cold",
*cost, result ? "final" : "partial");
}
return result;
}
static int
aarch64_register_move_cost (machine_mode mode,
reg_class_t from_i, reg_class_t to_i)
{
enum reg_class from = (enum reg_class) from_i;
enum reg_class to = (enum reg_class) to_i;
const struct cpu_regmove_cost *regmove_cost
= aarch64_tune_params.regmove_cost;
if (to == TAILCALL_ADDR_REGS || to == POINTER_REGS)
to = GENERAL_REGS;
if (from == TAILCALL_ADDR_REGS || from == POINTER_REGS)
from = GENERAL_REGS;
if ((from == GENERAL_REGS && to == STACK_REG)
|| (to == GENERAL_REGS && from == STACK_REG))
return regmove_cost->GP2GP;
if (to == STACK_REG || from == STACK_REG)
return aarch64_register_move_cost (mode, from, GENERAL_REGS)
+ aarch64_register_move_cost (mode, GENERAL_REGS, to);
if (known_eq (GET_MODE_SIZE (mode), 16))
{
if (from == GENERAL_REGS && to == GENERAL_REGS)
return regmove_cost->GP2GP * 2;
else if (from == GENERAL_REGS)
return regmove_cost->GP2FP * 2;
else if (to == GENERAL_REGS)
return regmove_cost->FP2GP * 2;
if (! TARGET_SIMD)
return regmove_cost->GP2FP + regmove_cost->FP2GP + regmove_cost->FP2FP;
return regmove_cost->FP2FP;
}
if (from == GENERAL_REGS && to == GENERAL_REGS)
return regmove_cost->GP2GP;
else if (from == GENERAL_REGS)
return regmove_cost->GP2FP;
else if (to == GENERAL_REGS)
return regmove_cost->FP2GP;
return regmove_cost->FP2FP;
}
static int
aarch64_memory_move_cost (machine_mode mode ATTRIBUTE_UNUSED,
reg_class_t rclass ATTRIBUTE_UNUSED,
bool in ATTRIBUTE_UNUSED)
{
return aarch64_tune_params.memmov_cost;
}
static bool
use_rsqrt_p (machine_mode mode)
{
return (!flag_trapping_math
&& flag_unsafe_math_optimizations
&& ((aarch64_tune_params.approx_modes->recip_sqrt
& AARCH64_APPROX_MODE (mode))
|| flag_mrecip_low_precision_sqrt));
}
static tree
aarch64_builtin_reciprocal (tree fndecl)
{
machine_mode mode = TYPE_MODE (TREE_TYPE (fndecl));
if (!use_rsqrt_p (mode))
return NULL_TREE;
return aarch64_builtin_rsqrt (DECL_FUNCTION_CODE (fndecl));
}
typedef rtx (*rsqrte_type) (rtx, rtx);
static rsqrte_type
get_rsqrte_type (machine_mode mode)
{
switch (mode)
{
case E_DFmode:   return gen_aarch64_rsqrtedf;
case E_SFmode:   return gen_aarch64_rsqrtesf;
case E_V2DFmode: return gen_aarch64_rsqrtev2df;
case E_V2SFmode: return gen_aarch64_rsqrtev2sf;
case E_V4SFmode: return gen_aarch64_rsqrtev4sf;
default: gcc_unreachable ();
}
}
typedef rtx (*rsqrts_type) (rtx, rtx, rtx);
static rsqrts_type
get_rsqrts_type (machine_mode mode)
{
switch (mode)
{
case E_DFmode:   return gen_aarch64_rsqrtsdf;
case E_SFmode:   return gen_aarch64_rsqrtssf;
case E_V2DFmode: return gen_aarch64_rsqrtsv2df;
case E_V2SFmode: return gen_aarch64_rsqrtsv2sf;
case E_V4SFmode: return gen_aarch64_rsqrtsv4sf;
default: gcc_unreachable ();
}
}
bool
aarch64_emit_approx_sqrt (rtx dst, rtx src, bool recp)
{
machine_mode mode = GET_MODE (dst);
if (GET_MODE_INNER (mode) == HFmode)
{
gcc_assert (!recp);
return false;
}
if (!recp)
{
if (!(flag_mlow_precision_sqrt
|| (aarch64_tune_params.approx_modes->sqrt
& AARCH64_APPROX_MODE (mode))))
return false;
if (flag_finite_math_only
|| flag_trapping_math
|| !flag_unsafe_math_optimizations
|| optimize_function_for_size_p (cfun))
return false;
}
else
gcc_assert (use_rsqrt_p (mode));
machine_mode mmsk = mode_for_int_vector (mode).require ();
rtx xmsk = gen_reg_rtx (mmsk);
if (!recp)
emit_insn (gen_rtx_SET (xmsk,
gen_rtx_NEG (mmsk,
gen_rtx_EQ (mmsk, src,
CONST0_RTX (mode)))));
rtx xdst = gen_reg_rtx (mode);
emit_insn ((*get_rsqrte_type (mode)) (xdst, src));
int iterations = (GET_MODE_INNER (mode) == DFmode) ? 3 : 2;
if ((recp && flag_mrecip_low_precision_sqrt)
|| (!recp && flag_mlow_precision_sqrt))
iterations--;
rtx x1 = gen_reg_rtx (mode);
while (iterations--)
{
rtx x2 = gen_reg_rtx (mode);
emit_set_insn (x2, gen_rtx_MULT (mode, xdst, xdst));
emit_insn ((*get_rsqrts_type (mode)) (x1, src, x2));
if (iterations > 0)
emit_set_insn (xdst, gen_rtx_MULT (mode, xdst, x1));
}
if (!recp)
{
rtx xtmp = gen_reg_rtx (mmsk);
emit_set_insn (xtmp, gen_rtx_AND (mmsk, gen_rtx_NOT (mmsk, xmsk),
gen_rtx_SUBREG (mmsk, xdst, 0)));
emit_move_insn (xdst, gen_rtx_SUBREG (mode, xtmp, 0));
emit_set_insn (xdst, gen_rtx_MULT (mode, xdst, src));
}
emit_set_insn (dst, gen_rtx_MULT (mode, xdst, x1));
return true;
}
typedef rtx (*recpe_type) (rtx, rtx);
static recpe_type
get_recpe_type (machine_mode mode)
{
switch (mode)
{
case E_SFmode:   return (gen_aarch64_frecpesf);
case E_V2SFmode: return (gen_aarch64_frecpev2sf);
case E_V4SFmode: return (gen_aarch64_frecpev4sf);
case E_DFmode:   return (gen_aarch64_frecpedf);
case E_V2DFmode: return (gen_aarch64_frecpev2df);
default:         gcc_unreachable ();
}
}
typedef rtx (*recps_type) (rtx, rtx, rtx);
static recps_type
get_recps_type (machine_mode mode)
{
switch (mode)
{
case E_SFmode:   return (gen_aarch64_frecpssf);
case E_V2SFmode: return (gen_aarch64_frecpsv2sf);
case E_V4SFmode: return (gen_aarch64_frecpsv4sf);
case E_DFmode:   return (gen_aarch64_frecpsdf);
case E_V2DFmode: return (gen_aarch64_frecpsv2df);
default:         gcc_unreachable ();
}
}
bool
aarch64_emit_approx_div (rtx quo, rtx num, rtx den)
{
machine_mode mode = GET_MODE (quo);
if (GET_MODE_INNER (mode) == HFmode)
return false;
bool use_approx_division_p = (flag_mlow_precision_div
|| (aarch64_tune_params.approx_modes->division
& AARCH64_APPROX_MODE (mode)));
if (!flag_finite_math_only
|| flag_trapping_math
|| !flag_unsafe_math_optimizations
|| optimize_function_for_size_p (cfun)
|| !use_approx_division_p)
return false;
if (!TARGET_SIMD && VECTOR_MODE_P (mode))
return false;
rtx xrcp = gen_reg_rtx (mode);
emit_insn ((*get_recpe_type (mode)) (xrcp, den));
int iterations = (GET_MODE_INNER (mode) == DFmode) ? 3 : 2;
if (flag_mlow_precision_div)
iterations--;
rtx xtmp = gen_reg_rtx (mode);
while (iterations--)
{
emit_insn ((*get_recps_type (mode)) (xtmp, xrcp, den));
if (iterations > 0)
emit_set_insn (xrcp, gen_rtx_MULT (mode, xrcp, xtmp));
}
if (num != CONST1_RTX (mode))
{
rtx xnum = force_reg (mode, num);
emit_set_insn (xrcp, gen_rtx_MULT (mode, xrcp, xnum));
}
emit_set_insn (quo, gen_rtx_MULT (mode, xrcp, xtmp));
return true;
}
static int
aarch64_sched_issue_rate (void)
{
return aarch64_tune_params.issue_rate;
}
static int
aarch64_sched_first_cycle_multipass_dfa_lookahead (void)
{
int issue_rate = aarch64_sched_issue_rate ();
return issue_rate > 1 && !sched_fusion ? issue_rate : 0;
}
static int
aarch64_first_cycle_multipass_dfa_lookahead_guard (rtx_insn *insn,
int ready_index)
{
return autopref_multipass_dfa_lookahead_guard (insn, ready_index);
}
static int
aarch64_builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype,
int misalign ATTRIBUTE_UNUSED)
{
unsigned elements;
const cpu_vector_cost *costs = aarch64_tune_params.vec_costs;
bool fp = false;
if (vectype != NULL)
fp = FLOAT_TYPE_P (vectype);
switch (type_of_cost)
{
case scalar_stmt:
return fp ? costs->scalar_fp_stmt_cost : costs->scalar_int_stmt_cost;
case scalar_load:
return costs->scalar_load_cost;
case scalar_store:
return costs->scalar_store_cost;
case vector_stmt:
return fp ? costs->vec_fp_stmt_cost : costs->vec_int_stmt_cost;
case vector_load:
return costs->vec_align_load_cost;
case vector_store:
return costs->vec_store_cost;
case vec_to_scalar:
return costs->vec_to_scalar_cost;
case scalar_to_vec:
return costs->scalar_to_vec_cost;
case unaligned_load:
case vector_gather_load:
return costs->vec_unalign_load_cost;
case unaligned_store:
case vector_scatter_store:
return costs->vec_unalign_store_cost;
case cond_branch_taken:
return costs->cond_taken_branch_cost;
case cond_branch_not_taken:
return costs->cond_not_taken_branch_cost;
case vec_perm:
return costs->vec_permute_cost;
case vec_promote_demote:
return fp ? costs->vec_fp_stmt_cost : costs->vec_int_stmt_cost;
case vec_construct:
elements = estimated_poly_value (TYPE_VECTOR_SUBPARTS (vectype));
return elements / 2 + 1;
default:
gcc_unreachable ();
}
}
static unsigned
aarch64_add_stmt_cost (void *data, int count, enum vect_cost_for_stmt kind,
struct _stmt_vec_info *stmt_info, int misalign,
enum vect_cost_model_location where)
{
unsigned *cost = (unsigned *) data;
unsigned retval = 0;
if (flag_vect_cost_model)
{
tree vectype = stmt_info ? stmt_vectype (stmt_info) : NULL_TREE;
int stmt_cost =
aarch64_builtin_vectorization_cost (kind, vectype, misalign);
if (where == vect_body && stmt_info && stmt_in_inner_loop_p (stmt_info))
count *= 50; 
retval = (unsigned) (count * stmt_cost);
cost[where] += retval;
}
return retval;
}
static void initialize_aarch64_code_model (struct gcc_options *);
static enum aarch64_parse_opt_result
aarch64_parse_arch (const char *to_parse, const struct processor **res,
unsigned long *isa_flags)
{
char *ext;
const struct processor *arch;
char *str = (char *) alloca (strlen (to_parse) + 1);
size_t len;
strcpy (str, to_parse);
ext = strchr (str, '+');
if (ext != NULL)
len = ext - str;
else
len = strlen (str);
if (len == 0)
return AARCH64_PARSE_MISSING_ARG;
for (arch = all_architectures; arch->name != NULL; arch++)
{
if (strlen (arch->name) == len && strncmp (arch->name, str, len) == 0)
{
unsigned long isa_temp = arch->flags;
if (ext != NULL)
{
enum aarch64_parse_opt_result ext_res
= aarch64_parse_extension (ext, &isa_temp);
if (ext_res != AARCH64_PARSE_OK)
return ext_res;
}
*res = arch;
*isa_flags = isa_temp;
return AARCH64_PARSE_OK;
}
}
return AARCH64_PARSE_INVALID_ARG;
}
static enum aarch64_parse_opt_result
aarch64_parse_cpu (const char *to_parse, const struct processor **res,
unsigned long *isa_flags)
{
char *ext;
const struct processor *cpu;
char *str = (char *) alloca (strlen (to_parse) + 1);
size_t len;
strcpy (str, to_parse);
ext = strchr (str, '+');
if (ext != NULL)
len = ext - str;
else
len = strlen (str);
if (len == 0)
return AARCH64_PARSE_MISSING_ARG;
for (cpu = all_cores; cpu->name != NULL; cpu++)
{
if (strlen (cpu->name) == len && strncmp (cpu->name, str, len) == 0)
{
unsigned long isa_temp = cpu->flags;
if (ext != NULL)
{
enum aarch64_parse_opt_result ext_res
= aarch64_parse_extension (ext, &isa_temp);
if (ext_res != AARCH64_PARSE_OK)
return ext_res;
}
*res = cpu;
*isa_flags = isa_temp;
return AARCH64_PARSE_OK;
}
}
return AARCH64_PARSE_INVALID_ARG;
}
static enum aarch64_parse_opt_result
aarch64_parse_tune (const char *to_parse, const struct processor **res)
{
const struct processor *cpu;
char *str = (char *) alloca (strlen (to_parse) + 1);
strcpy (str, to_parse);
for (cpu = all_cores; cpu->name != NULL; cpu++)
{
if (strcmp (cpu->name, str) == 0)
{
*res = cpu;
return AARCH64_PARSE_OK;
}
}
return AARCH64_PARSE_INVALID_ARG;
}
static unsigned int
aarch64_parse_one_option_token (const char *token,
size_t length,
const struct aarch64_flag_desc *flag,
const char *option_name)
{
for (; flag->name != NULL; flag++)
{
if (length == strlen (flag->name)
&& !strncmp (flag->name, token, length))
return flag->flag;
}
error ("unknown flag passed in -moverride=%s (%s)", option_name, token);
return 0;
}
static unsigned int
aarch64_parse_boolean_options (const char *option,
const struct aarch64_flag_desc *flags,
unsigned int initial_state,
const char *option_name)
{
const char separator = '.';
const char* specs = option;
const char* ntoken = option;
unsigned int found_flags = initial_state;
while ((ntoken = strchr (specs, separator)))
{
size_t token_length = ntoken - specs;
unsigned token_ops = aarch64_parse_one_option_token (specs,
token_length,
flags,
option_name);
if (!token_ops)
found_flags = 0;
found_flags |= token_ops;
specs = ++ntoken;
}
if (!(*specs))
{
error ("%s string ill-formed\n", option_name);
return 0;
}
size_t token_length = strlen (specs);
unsigned token_ops = aarch64_parse_one_option_token (specs,
token_length,
flags,
option_name);
if (!token_ops)
found_flags = 0;
found_flags |= token_ops;
return found_flags;
}
static void
aarch64_parse_fuse_string (const char *fuse_string,
struct tune_params *tune)
{
tune->fusible_ops = aarch64_parse_boolean_options (fuse_string,
aarch64_fusible_pairs,
tune->fusible_ops,
"fuse=");
}
static void
aarch64_parse_tune_string (const char *tune_string,
struct tune_params *tune)
{
tune->extra_tuning_flags
= aarch64_parse_boolean_options (tune_string,
aarch64_tuning_flags,
tune->extra_tuning_flags,
"tune=");
}
void
aarch64_parse_one_override_token (const char* token,
size_t length,
struct tune_params *tune)
{
const struct aarch64_tuning_override_function *fn
= aarch64_tuning_override_functions;
const char *option_part = strchr (token, '=');
if (!option_part)
{
error ("tuning string missing in option (%s)", token);
return;
}
length = option_part - token;
option_part++;
for (; fn->name != NULL; fn++)
{
if (!strncmp (fn->name, token, length))
{
fn->parse_override (option_part, tune);
return;
}
}
error ("unknown tuning option (%s)",token);
return;
}
static void
initialize_aarch64_tls_size (struct gcc_options *opts)
{
if (aarch64_tls_size == 0)
aarch64_tls_size = 24;
switch (opts->x_aarch64_cmodel_var)
{
case AARCH64_CMODEL_TINY:
if (aarch64_tls_size > 24)
aarch64_tls_size = 24;
break;
case AARCH64_CMODEL_SMALL:
if (aarch64_tls_size > 32)
aarch64_tls_size = 32;
break;
case AARCH64_CMODEL_LARGE:
if (aarch64_tls_size > 48)
aarch64_tls_size = 48;
break;
default:
gcc_unreachable ();
}
return;
}
static void
aarch64_parse_override_string (const char* input_string,
struct tune_params* tune)
{
const char separator = ':';
size_t string_length = strlen (input_string) + 1;
char *string_root = (char *) xmalloc (sizeof (*string_root) * string_length);
char *string = string_root;
strncpy (string, input_string, string_length);
string[string_length - 1] = '\0';
char* ntoken = string;
while ((ntoken = strchr (string, separator)))
{
size_t token_length = ntoken - string;
*ntoken = '\0';
aarch64_parse_one_override_token (string, token_length, tune);
string = ++ntoken;
}
aarch64_parse_one_override_token (string, strlen (string), tune);
free (string_root);
}
static void
aarch64_override_options_after_change_1 (struct gcc_options *opts)
{
if (opts->x_flag_omit_frame_pointer == 0)
opts->x_flag_omit_frame_pointer = 2;
if (!opts->x_optimize_size)
{
if (opts->x_align_loops <= 0)
opts->x_align_loops = aarch64_tune_params.loop_align;
if (opts->x_align_jumps <= 0)
opts->x_align_jumps = aarch64_tune_params.jump_align;
if (opts->x_align_functions <= 0)
opts->x_align_functions = aarch64_tune_params.function_align;
}
aarch64_pcrelative_literal_loads = false;
if (opts->x_pcrelative_literal_loads == 1)
aarch64_pcrelative_literal_loads = true;
if (aarch64_cmodel == AARCH64_CMODEL_TINY
|| aarch64_cmodel == AARCH64_CMODEL_TINY_PIC)
aarch64_pcrelative_literal_loads = true;
if (flag_mlow_precision_sqrt)
flag_mrecip_low_precision_sqrt = true;
}
void
aarch64_override_options_internal (struct gcc_options *opts)
{
aarch64_tune_flags = selected_tune->flags;
aarch64_tune = selected_tune->sched_core;
aarch64_tune_params = *(selected_tune->tune);
aarch64_architecture_version = selected_arch->architecture_version;
if (opts->x_aarch64_override_tune_string)
aarch64_parse_override_string (opts->x_aarch64_override_tune_string,
&aarch64_tune_params);
if (opts->x_flag_strict_volatile_bitfields < 0 && abi_version_at_least (2))
opts->x_flag_strict_volatile_bitfields = 1;
initialize_aarch64_code_model (opts);
initialize_aarch64_tls_size (opts);
int queue_depth = 0;
switch (aarch64_tune_params.autoprefetcher_model)
{
case tune_params::AUTOPREFETCHER_OFF:
queue_depth = -1;
break;
case tune_params::AUTOPREFETCHER_WEAK:
queue_depth = 0;
break;
case tune_params::AUTOPREFETCHER_STRONG:
queue_depth = max_insn_queue_index + 1;
break;
default:
gcc_unreachable ();
}
maybe_set_param_value (PARAM_SCHED_AUTOPREF_QUEUE_DEPTH,
queue_depth,
opts->x_param_values,
global_options_set.x_param_values);
if (aarch64_tune_params.prefetch->num_slots > 0)
maybe_set_param_value (PARAM_SIMULTANEOUS_PREFETCHES,
aarch64_tune_params.prefetch->num_slots,
opts->x_param_values,
global_options_set.x_param_values);
if (aarch64_tune_params.prefetch->l1_cache_size >= 0)
maybe_set_param_value (PARAM_L1_CACHE_SIZE,
aarch64_tune_params.prefetch->l1_cache_size,
opts->x_param_values,
global_options_set.x_param_values);
if (aarch64_tune_params.prefetch->l1_cache_line_size >= 0)
maybe_set_param_value (PARAM_L1_CACHE_LINE_SIZE,
aarch64_tune_params.prefetch->l1_cache_line_size,
opts->x_param_values,
global_options_set.x_param_values);
if (aarch64_tune_params.prefetch->l2_cache_size >= 0)
maybe_set_param_value (PARAM_L2_CACHE_SIZE,
aarch64_tune_params.prefetch->l2_cache_size,
opts->x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_SCHED_PRESSURE_ALGORITHM, SCHED_PRESSURE_MODEL,
opts->x_param_values,
global_options_set.x_param_values);
if (opts->x_flag_prefetch_loop_arrays < 0
&& !opts->x_optimize_size
&& aarch64_tune_params.prefetch->default_opt_level >= 0
&& opts->x_optimize >= aarch64_tune_params.prefetch->default_opt_level)
opts->x_flag_prefetch_loop_arrays = 1;
aarch64_override_options_after_change_1 (opts);
}
static void
aarch64_print_hint_for_core_or_arch (const char *str, bool arch)
{
auto_vec<const char *> candidates;
const struct processor *entry = arch ? all_architectures : all_cores;
for (; entry->name != NULL; entry++)
candidates.safe_push (entry->name);
#ifdef HAVE_LOCAL_CPU_DETECT
if (arch)
candidates.safe_push ("native");
#endif
char *s;
const char *hint = candidates_list_and_hint (str, s, candidates);
if (hint)
inform (input_location, "valid arguments are: %s;"
" did you mean %qs?", s, hint);
else
inform (input_location, "valid arguments are: %s", s);
XDELETEVEC (s);
}
inline static void
aarch64_print_hint_for_core (const char *str)
{
aarch64_print_hint_for_core_or_arch (str, false);
}
inline static void
aarch64_print_hint_for_arch (const char *str)
{
aarch64_print_hint_for_core_or_arch (str, true);
}
static bool
aarch64_validate_mcpu (const char *str, const struct processor **res,
unsigned long *isa_flags)
{
enum aarch64_parse_opt_result parse_res
= aarch64_parse_cpu (str, res, isa_flags);
if (parse_res == AARCH64_PARSE_OK)
return true;
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing cpu name in %<-mcpu=%s%>", str);
break;
case AARCH64_PARSE_INVALID_ARG:
error ("unknown value %qs for -mcpu", str);
aarch64_print_hint_for_core (str);
break;
case AARCH64_PARSE_INVALID_FEATURE:
error ("invalid feature modifier in %<-mcpu=%s%>", str);
break;
default:
gcc_unreachable ();
}
return false;
}
static bool
aarch64_validate_march (const char *str, const struct processor **res,
unsigned long *isa_flags)
{
enum aarch64_parse_opt_result parse_res
= aarch64_parse_arch (str, res, isa_flags);
if (parse_res == AARCH64_PARSE_OK)
return true;
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing arch name in %<-march=%s%>", str);
break;
case AARCH64_PARSE_INVALID_ARG:
error ("unknown value %qs for -march", str);
aarch64_print_hint_for_arch (str);
break;
case AARCH64_PARSE_INVALID_FEATURE:
error ("invalid feature modifier in %<-march=%s%>", str);
break;
default:
gcc_unreachable ();
}
return false;
}
static bool
aarch64_validate_mtune (const char *str, const struct processor **res)
{
enum aarch64_parse_opt_result parse_res
= aarch64_parse_tune (str, res);
if (parse_res == AARCH64_PARSE_OK)
return true;
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing cpu name in %<-mtune=%s%>", str);
break;
case AARCH64_PARSE_INVALID_ARG:
error ("unknown value %qs for -mtune", str);
aarch64_print_hint_for_core (str);
break;
default:
gcc_unreachable ();
}
return false;
}
static const struct processor *
aarch64_get_tune_cpu (enum aarch64_processor cpu)
{
if (cpu != aarch64_none)
return &all_cores[cpu];
return &all_cores[TARGET_CPU_DEFAULT & 0x3f];
}
static const struct processor *
aarch64_get_arch (enum aarch64_arch arch)
{
if (arch != aarch64_no_arch)
return &all_architectures[arch];
const struct processor *cpu = &all_cores[TARGET_CPU_DEFAULT & 0x3f];
return &all_architectures[cpu->arch];
}
static poly_uint16
aarch64_convert_sve_vector_bits (aarch64_sve_vector_bits_enum value)
{
if (value == SVE_SCALABLE || value == SVE_128)
return poly_uint16 (2, 2);
else
return (int) value / 64;
}
static void
aarch64_override_options (void)
{
unsigned long cpu_isa = 0;
unsigned long arch_isa = 0;
aarch64_isa_flags = 0;
bool valid_cpu = true;
bool valid_tune = true;
bool valid_arch = true;
selected_cpu = NULL;
selected_arch = NULL;
selected_tune = NULL;
if (aarch64_cpu_string)
valid_cpu = aarch64_validate_mcpu (aarch64_cpu_string, &selected_cpu,
&cpu_isa);
if (aarch64_arch_string)
valid_arch = aarch64_validate_march (aarch64_arch_string, &selected_arch,
&arch_isa);
if (aarch64_tune_string)
valid_tune = aarch64_validate_mtune (aarch64_tune_string, &selected_tune);
if (!selected_cpu)
{
if (selected_arch)
{
selected_cpu = &all_cores[selected_arch->ident];
aarch64_isa_flags = arch_isa;
explicit_arch = selected_arch->arch;
}
else
{
selected_cpu = aarch64_get_tune_cpu (aarch64_none);
aarch64_isa_flags = TARGET_CPU_DEFAULT >> 6;
}
if (selected_tune)
explicit_tune_core = selected_tune->ident;
}
else if (selected_arch)
{
if (selected_arch->arch != selected_cpu->arch)
{
warning (0, "switch -mcpu=%s conflicts with -march=%s switch",
all_architectures[selected_cpu->arch].name,
selected_arch->name);
}
aarch64_isa_flags = arch_isa;
explicit_arch = selected_arch->arch;
explicit_tune_core = selected_tune ? selected_tune->ident
: selected_cpu->ident;
}
else
{
aarch64_isa_flags = cpu_isa;
explicit_tune_core = selected_tune ? selected_tune->ident
: selected_cpu->ident;
gcc_assert (selected_cpu);
selected_arch = &all_architectures[selected_cpu->arch];
explicit_arch = selected_arch->arch;
}
if (!selected_arch)
{
gcc_assert (selected_cpu);
selected_arch = &all_architectures[selected_cpu->arch];
}
if (!selected_tune)
selected_tune = selected_cpu;
#ifndef HAVE_AS_MABI_OPTION
if (TARGET_ILP32)
error ("assembler does not support -mabi=ilp32");
#endif
aarch64_sve_vg = aarch64_convert_sve_vector_bits (aarch64_sve_vector_bits);
if (aarch64_ra_sign_scope != AARCH64_FUNCTION_NONE && TARGET_ILP32)
sorry ("return address signing is only supported for -mabi=lp64");
if ((aarch64_cpu_string && valid_cpu)
|| (aarch64_tune_string && valid_tune))
gcc_assert (explicit_tune_core != aarch64_none);
if ((aarch64_cpu_string && valid_cpu)
|| (aarch64_arch_string && valid_arch))
gcc_assert (explicit_arch != aarch64_no_arch);
aarch64_override_options_internal (&global_options);
target_option_default_node = target_option_current_node
= build_target_option_node (&global_options);
}
static void
aarch64_override_options_after_change (void)
{
aarch64_override_options_after_change_1 (&global_options);
}
static struct machine_function *
aarch64_init_machine_status (void)
{
struct machine_function *machine;
machine = ggc_cleared_alloc<machine_function> ();
return machine;
}
void
aarch64_init_expanders (void)
{
init_machine_status = aarch64_init_machine_status;
}
static void
initialize_aarch64_code_model (struct gcc_options *opts)
{
if (opts->x_flag_pic)
{
switch (opts->x_aarch64_cmodel_var)
{
case AARCH64_CMODEL_TINY:
aarch64_cmodel = AARCH64_CMODEL_TINY_PIC;
break;
case AARCH64_CMODEL_SMALL:
#ifdef HAVE_AS_SMALL_PIC_RELOCS
aarch64_cmodel = (flag_pic == 2
? AARCH64_CMODEL_SMALL_PIC
: AARCH64_CMODEL_SMALL_SPIC);
#else
aarch64_cmodel = AARCH64_CMODEL_SMALL_PIC;
#endif
break;
case AARCH64_CMODEL_LARGE:
sorry ("code model %qs with -f%s", "large",
opts->x_flag_pic > 1 ? "PIC" : "pic");
break;
default:
gcc_unreachable ();
}
}
else
aarch64_cmodel = opts->x_aarch64_cmodel_var;
}
static void
aarch64_option_save (struct cl_target_option *ptr, struct gcc_options *opts)
{
ptr->x_aarch64_override_tune_string = opts->x_aarch64_override_tune_string;
}
static void
aarch64_option_restore (struct gcc_options *opts, struct cl_target_option *ptr)
{
opts->x_explicit_tune_core = ptr->x_explicit_tune_core;
selected_tune = aarch64_get_tune_cpu (ptr->x_explicit_tune_core);
opts->x_explicit_arch = ptr->x_explicit_arch;
selected_arch = aarch64_get_arch (ptr->x_explicit_arch);
opts->x_aarch64_override_tune_string = ptr->x_aarch64_override_tune_string;
aarch64_override_options_internal (opts);
}
static void
aarch64_option_print (FILE *file, int indent, struct cl_target_option *ptr)
{
const struct processor *cpu
= aarch64_get_tune_cpu (ptr->x_explicit_tune_core);
unsigned long isa_flags = ptr->x_aarch64_isa_flags;
const struct processor *arch = aarch64_get_arch (ptr->x_explicit_arch);
std::string extension
= aarch64_get_extension_string_for_isa_flags (isa_flags, arch->flags);
fprintf (file, "%*sselected tune = %s\n", indent, "", cpu->name);
fprintf (file, "%*sselected arch = %s%s\n", indent, "",
arch->name, extension.c_str ());
}
static GTY(()) tree aarch64_previous_fndecl;
void
aarch64_reset_previous_fndecl (void)
{
aarch64_previous_fndecl = NULL;
}
void
aarch64_save_restore_target_globals (tree new_tree)
{
if (TREE_TARGET_GLOBALS (new_tree))
restore_target_globals (TREE_TARGET_GLOBALS (new_tree));
else if (new_tree == target_option_default_node)
restore_target_globals (&default_target_globals);
else
TREE_TARGET_GLOBALS (new_tree) = save_target_globals_default_opts ();
}
static void
aarch64_set_current_function (tree fndecl)
{
if (!fndecl || fndecl == aarch64_previous_fndecl)
return;
tree old_tree = (aarch64_previous_fndecl
? DECL_FUNCTION_SPECIFIC_TARGET (aarch64_previous_fndecl)
: NULL_TREE);
tree new_tree = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
if (!new_tree && old_tree)
new_tree = target_option_default_node;
if (old_tree == new_tree)
return;
aarch64_previous_fndecl = fndecl;
cl_target_option_restore (&global_options, TREE_TARGET_OPTION (new_tree));
aarch64_save_restore_target_globals (new_tree);
}
enum aarch64_attr_opt_type
{
aarch64_attr_mask,	
aarch64_attr_bool,	
aarch64_attr_enum,	
aarch64_attr_custom	
};
struct aarch64_attribute_info
{
const char *name;
enum aarch64_attr_opt_type attr_type;
bool allow_neg;
bool (*handler) (const char *);
enum opt_code opt_num;
};
static bool
aarch64_handle_attr_arch (const char *str)
{
const struct processor *tmp_arch = NULL;
enum aarch64_parse_opt_result parse_res
= aarch64_parse_arch (str, &tmp_arch, &aarch64_isa_flags);
if (parse_res == AARCH64_PARSE_OK)
{
gcc_assert (tmp_arch);
selected_arch = tmp_arch;
explicit_arch = selected_arch->arch;
return true;
}
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing name in %<target(\"arch=\")%> pragma or attribute");
break;
case AARCH64_PARSE_INVALID_ARG:
error ("invalid name (\"%s\") in %<target(\"arch=\")%> pragma or attribute", str);
aarch64_print_hint_for_arch (str);
break;
case AARCH64_PARSE_INVALID_FEATURE:
error ("invalid value (\"%s\") in %<target()%> pragma or attribute", str);
break;
default:
gcc_unreachable ();
}
return false;
}
static bool
aarch64_handle_attr_cpu (const char *str)
{
const struct processor *tmp_cpu = NULL;
enum aarch64_parse_opt_result parse_res
= aarch64_parse_cpu (str, &tmp_cpu, &aarch64_isa_flags);
if (parse_res == AARCH64_PARSE_OK)
{
gcc_assert (tmp_cpu);
selected_tune = tmp_cpu;
explicit_tune_core = selected_tune->ident;
selected_arch = &all_architectures[tmp_cpu->arch];
explicit_arch = selected_arch->arch;
return true;
}
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing name in %<target(\"cpu=\")%> pragma or attribute");
break;
case AARCH64_PARSE_INVALID_ARG:
error ("invalid name (\"%s\") in %<target(\"cpu=\")%> pragma or attribute", str);
aarch64_print_hint_for_core (str);
break;
case AARCH64_PARSE_INVALID_FEATURE:
error ("invalid value (\"%s\") in %<target()%> pragma or attribute", str);
break;
default:
gcc_unreachable ();
}
return false;
}
static bool
aarch64_handle_attr_tune (const char *str)
{
const struct processor *tmp_tune = NULL;
enum aarch64_parse_opt_result parse_res
= aarch64_parse_tune (str, &tmp_tune);
if (parse_res == AARCH64_PARSE_OK)
{
gcc_assert (tmp_tune);
selected_tune = tmp_tune;
explicit_tune_core = selected_tune->ident;
return true;
}
switch (parse_res)
{
case AARCH64_PARSE_INVALID_ARG:
error ("invalid name (\"%s\") in %<target(\"tune=\")%> pragma or attribute", str);
aarch64_print_hint_for_core (str);
break;
default:
gcc_unreachable ();
}
return false;
}
static bool
aarch64_handle_attr_isa_flags (char *str)
{
enum aarch64_parse_opt_result parse_res;
unsigned long isa_flags = aarch64_isa_flags;
if (strncmp ("+nothing", str, 8) == 0)
{
isa_flags = 0;
str += 8;
}
parse_res = aarch64_parse_extension (str, &isa_flags);
if (parse_res == AARCH64_PARSE_OK)
{
aarch64_isa_flags = isa_flags;
return true;
}
switch (parse_res)
{
case AARCH64_PARSE_MISSING_ARG:
error ("missing value in %<target()%> pragma or attribute");
break;
case AARCH64_PARSE_INVALID_FEATURE:
error ("invalid value (\"%s\") in %<target()%> pragma or attribute", str);
break;
default:
gcc_unreachable ();
}
return false;
}
static const struct aarch64_attribute_info aarch64_attributes[] =
{
{ "general-regs-only", aarch64_attr_mask, false, NULL,
OPT_mgeneral_regs_only },
{ "fix-cortex-a53-835769", aarch64_attr_bool, true, NULL,
OPT_mfix_cortex_a53_835769 },
{ "fix-cortex-a53-843419", aarch64_attr_bool, true, NULL,
OPT_mfix_cortex_a53_843419 },
{ "cmodel", aarch64_attr_enum, false, NULL, OPT_mcmodel_ },
{ "strict-align", aarch64_attr_mask, false, NULL, OPT_mstrict_align },
{ "omit-leaf-frame-pointer", aarch64_attr_bool, true, NULL,
OPT_momit_leaf_frame_pointer },
{ "tls-dialect", aarch64_attr_enum, false, NULL, OPT_mtls_dialect_ },
{ "arch", aarch64_attr_custom, false, aarch64_handle_attr_arch,
OPT_march_ },
{ "cpu", aarch64_attr_custom, false, aarch64_handle_attr_cpu, OPT_mcpu_ },
{ "tune", aarch64_attr_custom, false, aarch64_handle_attr_tune,
OPT_mtune_ },
{ "sign-return-address", aarch64_attr_enum, false, NULL,
OPT_msign_return_address_ },
{ NULL, aarch64_attr_custom, false, NULL, OPT____ }
};
static bool
aarch64_process_one_target_attr (char *arg_str)
{
bool invert = false;
size_t len = strlen (arg_str);
if (len == 0)
{
error ("malformed %<target()%> pragma or attribute");
return false;
}
char *str_to_check = (char *) alloca (len + 1);
strcpy (str_to_check, arg_str);
while (*str_to_check == ' ' || *str_to_check == '\t')
str_to_check++;
if (*str_to_check == '+')
return aarch64_handle_attr_isa_flags (str_to_check);
if (len > 3 && strncmp (str_to_check, "no-", 3) == 0)
{
invert = true;
str_to_check += 3;
}
char *arg = strchr (str_to_check, '=');
if (arg)
{
*arg = '\0';
arg++;
}
const struct aarch64_attribute_info *p_attr;
bool found = false;
for (p_attr = aarch64_attributes; p_attr->name; p_attr++)
{
if (strcmp (str_to_check, p_attr->name) != 0)
continue;
found = true;
bool attr_need_arg_p = p_attr->attr_type == aarch64_attr_custom
|| p_attr->attr_type == aarch64_attr_enum;
if (attr_need_arg_p ^ (arg != NULL))
{
error ("pragma or attribute %<target(\"%s\")%> does not accept an argument", str_to_check);
return false;
}
if (invert && !p_attr->allow_neg)
{
error ("pragma or attribute %<target(\"%s\")%> does not allow a negated form", str_to_check);
return false;
}
switch (p_attr->attr_type)
{
case aarch64_attr_custom:
gcc_assert (p_attr->handler);
if (!p_attr->handler (arg))
return false;
break;
case aarch64_attr_bool:
{
struct cl_decoded_option decoded;
generate_option (p_attr->opt_num, NULL, !invert,
CL_TARGET, &decoded);
aarch64_handle_option (&global_options, &global_options_set,
&decoded, input_location);
break;
}
case aarch64_attr_mask:
{
struct cl_decoded_option decoded;
decoded.opt_index = p_attr->opt_num;
decoded.value = !invert;
aarch64_handle_option (&global_options, &global_options_set,
&decoded, input_location);
break;
}
case aarch64_attr_enum:
{
gcc_assert (arg);
bool valid;
int value;
valid = opt_enum_arg_to_value (p_attr->opt_num, arg,
&value, CL_TARGET);
if (valid)
{
set_option (&global_options, NULL, p_attr->opt_num, value,
NULL, DK_UNSPECIFIED, input_location,
global_dc);
}
else
{
error ("pragma or attribute %<target(\"%s=%s\")%> is not valid", str_to_check, arg);
}
break;
}
default:
gcc_unreachable ();
}
}
return found;
}
static unsigned int
num_occurences_in_str (char c, char *str)
{
unsigned int res = 0;
while (*str != '\0')
{
if (*str == c)
res++;
str++;
}
return res;
}
bool
aarch64_process_target_attr (tree args)
{
if (TREE_CODE (args) == TREE_LIST)
{
do
{
tree head = TREE_VALUE (args);
if (head)
{
if (!aarch64_process_target_attr (head))
return false;
}
args = TREE_CHAIN (args);
} while (args);
return true;
}
if (TREE_CODE (args) != STRING_CST)
{
error ("attribute %<target%> argument not a string");
return false;
}
size_t len = strlen (TREE_STRING_POINTER (args));
char *str_to_check = (char *) alloca (len + 1);
strcpy (str_to_check, TREE_STRING_POINTER (args));
if (len == 0)
{
error ("malformed %<target()%> pragma or attribute");
return false;
}
unsigned int num_commas = num_occurences_in_str (',', str_to_check);
char *token = strtok (str_to_check, ",");
unsigned int num_attrs = 0;
while (token)
{
num_attrs++;
if (!aarch64_process_one_target_attr (token))
{
error ("pragma or attribute %<target(\"%s\")%> is not valid", token);
return false;
}
token = strtok (NULL, ",");
}
if (num_attrs != num_commas + 1)
{
error ("malformed %<target(\"%s\")%> pragma or attribute", TREE_STRING_POINTER (args));
return false;
}
return true;
}
static bool
aarch64_option_valid_attribute_p (tree fndecl, tree, tree args, int)
{
struct cl_target_option cur_target;
bool ret;
tree old_optimize;
tree new_target, new_optimize;
tree existing_target = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
if (!existing_target && args == current_target_pragma)
{
DECL_FUNCTION_SPECIFIC_TARGET (fndecl) = target_option_current_node;
return true;
}
tree func_optimize = DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl);
old_optimize = build_optimization_node (&global_options);
func_optimize = DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl);
if (func_optimize && func_optimize != old_optimize)
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (func_optimize));
cl_target_option_save (&cur_target, &global_options);
if (existing_target)
{
struct cl_target_option *existing_options
= TREE_TARGET_OPTION (existing_target);
if (existing_options)
cl_target_option_restore (&global_options, existing_options);
}
else
cl_target_option_restore (&global_options,
TREE_TARGET_OPTION (target_option_current_node));
ret = aarch64_process_target_attr (args);
if (ret)
{
aarch64_override_options_internal (&global_options);
if (TARGET_SIMD)
{
tree saved_current_target_pragma = current_target_pragma;
current_target_pragma = NULL;
aarch64_init_simd_builtins ();
current_target_pragma = saved_current_target_pragma;
}
new_target = build_target_option_node (&global_options);
}
else
new_target = NULL;
new_optimize = build_optimization_node (&global_options);
if (fndecl && ret)
{
DECL_FUNCTION_SPECIFIC_TARGET (fndecl) = new_target;
if (old_optimize != new_optimize)
DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl) = new_optimize;
}
cl_target_option_restore (&global_options, &cur_target);
if (old_optimize != new_optimize)
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (old_optimize));
return ret;
}
static bool
aarch64_tribools_ok_for_inlining_p (int caller, int callee,
int dont_care, int def)
{
if (callee == dont_care)
return true;
if (caller == dont_care)
return true;
return (callee == caller || callee == def);
}
static bool
aarch64_can_inline_p (tree caller, tree callee)
{
tree caller_tree = DECL_FUNCTION_SPECIFIC_TARGET (caller);
tree callee_tree = DECL_FUNCTION_SPECIFIC_TARGET (callee);
if (!callee_tree)
return true;
struct cl_target_option *caller_opts
= TREE_TARGET_OPTION (caller_tree ? caller_tree
: target_option_default_node);
struct cl_target_option *callee_opts = TREE_TARGET_OPTION (callee_tree);
if ((caller_opts->x_aarch64_isa_flags & callee_opts->x_aarch64_isa_flags)
!= callee_opts->x_aarch64_isa_flags)
return false;
if ((TARGET_STRICT_ALIGN_P (caller_opts->x_target_flags)
!= TARGET_STRICT_ALIGN_P (callee_opts->x_target_flags))
&& !(!TARGET_STRICT_ALIGN_P (callee_opts->x_target_flags)
&& TARGET_STRICT_ALIGN_P (caller_opts->x_target_flags)))
return false;
bool always_inline = lookup_attribute ("always_inline",
DECL_ATTRIBUTES (callee));
if (always_inline)
return true;
if (caller_opts->x_aarch64_cmodel_var
!= callee_opts->x_aarch64_cmodel_var)
return false;
if (caller_opts->x_aarch64_tls_dialect
!= callee_opts->x_aarch64_tls_dialect)
return false;
if (!aarch64_tribools_ok_for_inlining_p (
caller_opts->x_aarch64_fix_a53_err835769,
callee_opts->x_aarch64_fix_a53_err835769,
2, TARGET_FIX_ERR_A53_835769_DEFAULT))
return false;
if (!aarch64_tribools_ok_for_inlining_p (
caller_opts->x_aarch64_fix_a53_err843419,
callee_opts->x_aarch64_fix_a53_err843419,
2, TARGET_FIX_ERR_A53_843419))
return false;
if (!aarch64_tribools_ok_for_inlining_p (
caller_opts->x_flag_omit_leaf_frame_pointer,
callee_opts->x_flag_omit_leaf_frame_pointer,
2, 1))
return false;
if (callee_opts->x_aarch64_override_tune_string != NULL
&& caller_opts->x_aarch64_override_tune_string == NULL)
return false;
if (callee_opts->x_aarch64_override_tune_string
&& caller_opts->x_aarch64_override_tune_string
&& (strcmp (callee_opts->x_aarch64_override_tune_string,
caller_opts->x_aarch64_override_tune_string) != 0))
return false;
return true;
}
static bool
aarch64_symbol_binds_local_p (const_rtx x)
{
return (SYMBOL_REF_DECL (x)
? targetm.binds_local_p (SYMBOL_REF_DECL (x))
: SYMBOL_REF_LOCAL_P (x));
}
static bool
aarch64_tls_symbol_p (rtx x)
{
if (! TARGET_HAVE_TLS)
return false;
if (GET_CODE (x) != SYMBOL_REF)
return false;
return SYMBOL_REF_TLS_MODEL (x) != 0;
}
enum aarch64_symbol_type
aarch64_classify_tls_symbol (rtx x)
{
enum tls_model tls_kind = tls_symbolic_operand_type (x);
switch (tls_kind)
{
case TLS_MODEL_GLOBAL_DYNAMIC:
case TLS_MODEL_LOCAL_DYNAMIC:
return TARGET_TLS_DESC ? SYMBOL_SMALL_TLSDESC : SYMBOL_SMALL_TLSGD;
case TLS_MODEL_INITIAL_EXEC:
switch (aarch64_cmodel)
{
case AARCH64_CMODEL_TINY:
case AARCH64_CMODEL_TINY_PIC:
return SYMBOL_TINY_TLSIE;
default:
return SYMBOL_SMALL_TLSIE;
}
case TLS_MODEL_LOCAL_EXEC:
if (aarch64_tls_size == 12)
return SYMBOL_TLSLE12;
else if (aarch64_tls_size == 24)
return SYMBOL_TLSLE24;
else if (aarch64_tls_size == 32)
return SYMBOL_TLSLE32;
else if (aarch64_tls_size == 48)
return SYMBOL_TLSLE48;
else
gcc_unreachable ();
case TLS_MODEL_EMULATED:
case TLS_MODEL_NONE:
return SYMBOL_FORCE_TO_MEM;
default:
gcc_unreachable ();
}
}
enum aarch64_symbol_type
aarch64_classify_symbol (rtx x, HOST_WIDE_INT offset)
{
if (GET_CODE (x) == LABEL_REF)
{
switch (aarch64_cmodel)
{
case AARCH64_CMODEL_LARGE:
return SYMBOL_FORCE_TO_MEM;
case AARCH64_CMODEL_TINY_PIC:
case AARCH64_CMODEL_TINY:
return SYMBOL_TINY_ABSOLUTE;
case AARCH64_CMODEL_SMALL_SPIC:
case AARCH64_CMODEL_SMALL_PIC:
case AARCH64_CMODEL_SMALL:
return SYMBOL_SMALL_ABSOLUTE;
default:
gcc_unreachable ();
}
}
if (GET_CODE (x) == SYMBOL_REF)
{
if (aarch64_tls_symbol_p (x))
return aarch64_classify_tls_symbol (x);
switch (aarch64_cmodel)
{
case AARCH64_CMODEL_TINY:
if ((SYMBOL_REF_WEAK (x)
&& !aarch64_symbol_binds_local_p (x))
|| !IN_RANGE (offset, -1048575, 1048575))
return SYMBOL_FORCE_TO_MEM;
return SYMBOL_TINY_ABSOLUTE;
case AARCH64_CMODEL_SMALL:
if ((SYMBOL_REF_WEAK (x)
&& !aarch64_symbol_binds_local_p (x))
|| !IN_RANGE (offset, HOST_WIDE_INT_C (-4294967263),
HOST_WIDE_INT_C (4294967264)))
return SYMBOL_FORCE_TO_MEM;
return SYMBOL_SMALL_ABSOLUTE;
case AARCH64_CMODEL_TINY_PIC:
if (!aarch64_symbol_binds_local_p (x))
return SYMBOL_TINY_GOT;
return SYMBOL_TINY_ABSOLUTE;
case AARCH64_CMODEL_SMALL_SPIC:
case AARCH64_CMODEL_SMALL_PIC:
if (!aarch64_symbol_binds_local_p (x))
return (aarch64_cmodel == AARCH64_CMODEL_SMALL_SPIC
?  SYMBOL_SMALL_GOT_28K : SYMBOL_SMALL_GOT_4G);
return SYMBOL_SMALL_ABSOLUTE;
case AARCH64_CMODEL_LARGE:
if (!aarch64_pcrelative_literal_loads && CONSTANT_POOL_ADDRESS_P (x))
return SYMBOL_SMALL_ABSOLUTE;
else
return SYMBOL_FORCE_TO_MEM;
default:
gcc_unreachable ();
}
}
return SYMBOL_FORCE_TO_MEM;
}
bool
aarch64_constant_address_p (rtx x)
{
return (CONSTANT_P (x) && memory_address_p (DImode, x));
}
bool
aarch64_legitimate_pic_operand_p (rtx x)
{
if (GET_CODE (x) == SYMBOL_REF
|| (GET_CODE (x) == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF))
return false;
return true;
}
static bool
aarch64_legitimate_constant_p (machine_mode mode, rtx x)
{
if (CONST_INT_P (x)
|| (CONST_DOUBLE_P (x) && GET_MODE_CLASS (mode) == MODE_FLOAT)
|| GET_CODE (x) == CONST_VECTOR)
return true;
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
if (vec_flags == (VEC_ADVSIMD | VEC_STRUCT))
return false;
if (vec_flags & VEC_ANY_SVE)
return aarch64_simd_valid_immediate (x, NULL);
if (GET_CODE (x) == HIGH)
x = XEXP (x, 0);
poly_int64 offset;
if (poly_int_rtx_p (x, &offset))
return aarch64_offset_temporaries (false, offset) <= 1;
x = strip_offset (x, &offset);
if (!offset.is_constant () && aarch64_offset_temporaries (true, offset) > 0)
return false;
if (maybe_ne (offset, 0) && SYMBOL_REF_P (x) && SYMBOL_REF_ANCHOR_P (x))
return false;
if (SYMBOL_REF_P (x) && !SYMBOL_REF_TLS_MODEL (x))
return true;
if (GET_CODE (x) == LABEL_REF)
return true;
return false;
}
rtx
aarch64_load_tp (rtx target)
{
if (!target
|| GET_MODE (target) != Pmode
|| !register_operand (target, Pmode))
target = gen_reg_rtx (Pmode);
emit_insn (gen_aarch64_load_tp_hard (target));
return target;
}
static GTY(()) tree va_list_type;
static tree
aarch64_build_builtin_va_list (void)
{
tree va_list_name;
tree f_stack, f_grtop, f_vrtop, f_groff, f_vroff;
va_list_type = lang_hooks.types.make_type (RECORD_TYPE);
va_list_name = build_decl (BUILTINS_LOCATION,
TYPE_DECL,
get_identifier ("__va_list"),
va_list_type);
DECL_ARTIFICIAL (va_list_name) = 1;
TYPE_NAME (va_list_type) = va_list_name;
TYPE_STUB_DECL (va_list_type) = va_list_name;
f_stack = build_decl (BUILTINS_LOCATION,
FIELD_DECL, get_identifier ("__stack"),
ptr_type_node);
f_grtop = build_decl (BUILTINS_LOCATION,
FIELD_DECL, get_identifier ("__gr_top"),
ptr_type_node);
f_vrtop = build_decl (BUILTINS_LOCATION,
FIELD_DECL, get_identifier ("__vr_top"),
ptr_type_node);
f_groff = build_decl (BUILTINS_LOCATION,
FIELD_DECL, get_identifier ("__gr_offs"),
integer_type_node);
f_vroff = build_decl (BUILTINS_LOCATION,
FIELD_DECL, get_identifier ("__vr_offs"),
integer_type_node);
va_list_gpr_counter_field = f_groff;
va_list_fpr_counter_field = f_vroff;
DECL_ARTIFICIAL (f_stack) = 1;
DECL_ARTIFICIAL (f_grtop) = 1;
DECL_ARTIFICIAL (f_vrtop) = 1;
DECL_ARTIFICIAL (f_groff) = 1;
DECL_ARTIFICIAL (f_vroff) = 1;
DECL_FIELD_CONTEXT (f_stack) = va_list_type;
DECL_FIELD_CONTEXT (f_grtop) = va_list_type;
DECL_FIELD_CONTEXT (f_vrtop) = va_list_type;
DECL_FIELD_CONTEXT (f_groff) = va_list_type;
DECL_FIELD_CONTEXT (f_vroff) = va_list_type;
TYPE_FIELDS (va_list_type) = f_stack;
DECL_CHAIN (f_stack) = f_grtop;
DECL_CHAIN (f_grtop) = f_vrtop;
DECL_CHAIN (f_vrtop) = f_groff;
DECL_CHAIN (f_groff) = f_vroff;
layout_type (va_list_type);
return va_list_type;
}
static void
aarch64_expand_builtin_va_start (tree valist, rtx nextarg ATTRIBUTE_UNUSED)
{
const CUMULATIVE_ARGS *cum;
tree f_stack, f_grtop, f_vrtop, f_groff, f_vroff;
tree stack, grtop, vrtop, groff, vroff;
tree t;
int gr_save_area_size = cfun->va_list_gpr_size;
int vr_save_area_size = cfun->va_list_fpr_size;
int vr_offset;
cum = &crtl->args.info;
if (cfun->va_list_gpr_size)
gr_save_area_size = MIN ((NUM_ARG_REGS - cum->aapcs_ncrn) * UNITS_PER_WORD,
cfun->va_list_gpr_size);
if (cfun->va_list_fpr_size)
vr_save_area_size = MIN ((NUM_FP_ARG_REGS - cum->aapcs_nvrn)
* UNITS_PER_VREG, cfun->va_list_fpr_size);
if (!TARGET_FLOAT)
{
gcc_assert (cum->aapcs_nvrn == 0);
vr_save_area_size = 0;
}
f_stack = TYPE_FIELDS (va_list_type_node);
f_grtop = DECL_CHAIN (f_stack);
f_vrtop = DECL_CHAIN (f_grtop);
f_groff = DECL_CHAIN (f_vrtop);
f_vroff = DECL_CHAIN (f_groff);
stack = build3 (COMPONENT_REF, TREE_TYPE (f_stack), valist, f_stack,
NULL_TREE);
grtop = build3 (COMPONENT_REF, TREE_TYPE (f_grtop), valist, f_grtop,
NULL_TREE);
vrtop = build3 (COMPONENT_REF, TREE_TYPE (f_vrtop), valist, f_vrtop,
NULL_TREE);
groff = build3 (COMPONENT_REF, TREE_TYPE (f_groff), valist, f_groff,
NULL_TREE);
vroff = build3 (COMPONENT_REF, TREE_TYPE (f_vroff), valist, f_vroff,
NULL_TREE);
t = make_tree (TREE_TYPE (stack), virtual_incoming_args_rtx);
if (cum->aapcs_stack_size > 0)
t = fold_build_pointer_plus_hwi (t, cum->aapcs_stack_size * UNITS_PER_WORD);
t = build2 (MODIFY_EXPR, TREE_TYPE (stack), stack, t);
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
t = make_tree (TREE_TYPE (grtop), virtual_incoming_args_rtx);
t = build2 (MODIFY_EXPR, TREE_TYPE (grtop), grtop, t);
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
t = make_tree (TREE_TYPE (vrtop), virtual_incoming_args_rtx);
vr_offset = ROUND_UP (gr_save_area_size,
STACK_BOUNDARY / BITS_PER_UNIT);
if (vr_offset)
t = fold_build_pointer_plus_hwi (t, -vr_offset);
t = build2 (MODIFY_EXPR, TREE_TYPE (vrtop), vrtop, t);
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
t = build2 (MODIFY_EXPR, TREE_TYPE (groff), groff,
build_int_cst (TREE_TYPE (groff), -gr_save_area_size));
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
t = build2 (MODIFY_EXPR, TREE_TYPE (vroff), vroff,
build_int_cst (TREE_TYPE (vroff), -vr_save_area_size));
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
}
static tree
aarch64_gimplify_va_arg_expr (tree valist, tree type, gimple_seq *pre_p,
gimple_seq *post_p ATTRIBUTE_UNUSED)
{
tree addr;
bool indirect_p;
bool is_ha;		
bool dw_align;	
machine_mode ag_mode = VOIDmode;
int nregs;
machine_mode mode;
tree f_stack, f_grtop, f_vrtop, f_groff, f_vroff;
tree stack, f_top, f_off, off, arg, roundup, on_stack;
HOST_WIDE_INT size, rsize, adjust, align;
tree t, u, cond1, cond2;
indirect_p = pass_by_reference (NULL, TYPE_MODE (type), type, false);
if (indirect_p)
type = build_pointer_type (type);
mode = TYPE_MODE (type);
f_stack = TYPE_FIELDS (va_list_type_node);
f_grtop = DECL_CHAIN (f_stack);
f_vrtop = DECL_CHAIN (f_grtop);
f_groff = DECL_CHAIN (f_vrtop);
f_vroff = DECL_CHAIN (f_groff);
stack = build3 (COMPONENT_REF, TREE_TYPE (f_stack), unshare_expr (valist),
f_stack, NULL_TREE);
size = int_size_in_bytes (type);
align = aarch64_function_arg_alignment (mode, type) / BITS_PER_UNIT;
dw_align = false;
adjust = 0;
if (aarch64_vfp_is_call_or_return_candidate (mode,
type,
&ag_mode,
&nregs,
&is_ha))
{
unsigned int ag_size = GET_MODE_SIZE (ag_mode).to_constant ();
if (!TARGET_FLOAT)
aarch64_err_no_fpadvsimd (mode, "varargs");
f_top = build3 (COMPONENT_REF, TREE_TYPE (f_vrtop),
unshare_expr (valist), f_vrtop, NULL_TREE);
f_off = build3 (COMPONENT_REF, TREE_TYPE (f_vroff),
unshare_expr (valist), f_vroff, NULL_TREE);
rsize = nregs * UNITS_PER_VREG;
if (is_ha)
{
if (BYTES_BIG_ENDIAN && ag_size < UNITS_PER_VREG)
adjust = UNITS_PER_VREG - ag_size;
}
else if (BLOCK_REG_PADDING (mode, type, 1) == PAD_DOWNWARD
&& size < UNITS_PER_VREG)
{
adjust = UNITS_PER_VREG - size;
}
}
else
{
f_top = build3 (COMPONENT_REF, TREE_TYPE (f_grtop),
unshare_expr (valist), f_grtop, NULL_TREE);
f_off = build3 (COMPONENT_REF, TREE_TYPE (f_groff),
unshare_expr (valist), f_groff, NULL_TREE);
rsize = ROUND_UP (size, UNITS_PER_WORD);
nregs = rsize / UNITS_PER_WORD;
if (align > 8)
dw_align = true;
if (BLOCK_REG_PADDING (mode, type, 1) == PAD_DOWNWARD
&& size < UNITS_PER_WORD)
{
adjust = UNITS_PER_WORD  - size;
}
}
off = get_initialized_tmp_var (f_off, pre_p, NULL);
t = build2 (GE_EXPR, boolean_type_node, off,
build_int_cst (TREE_TYPE (off), 0));
cond1 = build3 (COND_EXPR, ptr_type_node, t, NULL_TREE, NULL_TREE);
if (dw_align)
{
t = build2 (PLUS_EXPR, TREE_TYPE (off), off,
build_int_cst (TREE_TYPE (off), 15));
t = build2 (BIT_AND_EXPR, TREE_TYPE (off), t,
build_int_cst (TREE_TYPE (off), -16));
roundup = build2 (MODIFY_EXPR, TREE_TYPE (off), off, t);
}
else
roundup = NULL;
t = build2 (PLUS_EXPR, TREE_TYPE (off), off,
build_int_cst (TREE_TYPE (off), rsize));
t = build2 (MODIFY_EXPR, TREE_TYPE (f_off), unshare_expr (f_off), t);
if (roundup)
t = build2 (COMPOUND_EXPR, TREE_TYPE (t), roundup, t);
u = build2 (GT_EXPR, boolean_type_node, unshare_expr (f_off),
build_int_cst (TREE_TYPE (f_off), 0));
cond2 = build3 (COND_EXPR, ptr_type_node, u, NULL_TREE, NULL_TREE);
t = build2 (COMPOUND_EXPR, TREE_TYPE (cond2), t, cond2);
COND_EXPR_ELSE (cond1) = t;
arg = get_initialized_tmp_var (stack, pre_p, NULL);
if (align > 8)
{
t = fold_convert (intDI_type_node, arg);
t = build2 (PLUS_EXPR, TREE_TYPE (t), t,
build_int_cst (TREE_TYPE (t), 15));
t = build2 (BIT_AND_EXPR, TREE_TYPE (t), t,
build_int_cst (TREE_TYPE (t), -16));
t = fold_convert (TREE_TYPE (arg), t);
roundup = build2 (MODIFY_EXPR, TREE_TYPE (arg), arg, t);
}
else
roundup = NULL;
t = fold_convert (intDI_type_node, arg);
t = build2 (PLUS_EXPR, TREE_TYPE (t), t,
build_int_cst (TREE_TYPE (t), size + 7));
t = build2 (BIT_AND_EXPR, TREE_TYPE (t), t,
build_int_cst (TREE_TYPE (t), -8));
t = fold_convert (TREE_TYPE (arg), t);
t = build2 (MODIFY_EXPR, TREE_TYPE (stack), unshare_expr (stack), t);
if (roundup)
t = build2 (COMPOUND_EXPR, TREE_TYPE (t), roundup, t);
on_stack = build2 (COMPOUND_EXPR, TREE_TYPE (arg), t, arg);
if (BLOCK_REG_PADDING (mode, type, 1) == PAD_DOWNWARD
&& size < UNITS_PER_WORD)
{
t = build2 (POINTER_PLUS_EXPR, TREE_TYPE (arg), arg,
size_int (UNITS_PER_WORD - size));
on_stack = build2 (COMPOUND_EXPR, TREE_TYPE (arg), on_stack, t);
}
COND_EXPR_THEN (cond1) = unshare_expr (on_stack);
COND_EXPR_THEN (cond2) = unshare_expr (on_stack);
t = off;
if (adjust)
t = build2 (PREINCREMENT_EXPR, TREE_TYPE (off), off,
build_int_cst (TREE_TYPE (off), adjust));
t = fold_convert (sizetype, t);
t = build2 (POINTER_PLUS_EXPR, TREE_TYPE (f_top), f_top, t);
if (is_ha)
{
int i;
tree tmp_ha, field_t, field_ptr_t;
tmp_ha = create_tmp_var_raw (type, "ha");
gimple_add_tmp_var (tmp_ha);
switch (ag_mode)
{
case E_SFmode:
field_t = float_type_node;
field_ptr_t = float_ptr_type_node;
break;
case E_DFmode:
field_t = double_type_node;
field_ptr_t = double_ptr_type_node;
break;
case E_TFmode:
field_t = long_double_type_node;
field_ptr_t = long_double_ptr_type_node;
break;
case E_HFmode:
field_t = aarch64_fp16_type_node;
field_ptr_t = aarch64_fp16_ptr_type_node;
break;
case E_V2SImode:
case E_V4SImode:
{
tree innertype = make_signed_type (GET_MODE_PRECISION (SImode));
field_t = build_vector_type_for_mode (innertype, ag_mode);
field_ptr_t = build_pointer_type (field_t);
}
break;
default:
gcc_assert (0);
}
tmp_ha = build1 (ADDR_EXPR, field_ptr_t, tmp_ha);
addr = t;
t = fold_convert (field_ptr_t, addr);
t = build2 (MODIFY_EXPR, field_t,
build1 (INDIRECT_REF, field_t, tmp_ha),
build1 (INDIRECT_REF, field_t, t));
for (i = 1; i < nregs; ++i)
{
addr = fold_build_pointer_plus_hwi (addr, UNITS_PER_VREG);
u = fold_convert (field_ptr_t, addr);
u = build2 (MODIFY_EXPR, field_t,
build2 (MEM_REF, field_t, tmp_ha,
build_int_cst (field_ptr_t,
(i *
int_size_in_bytes (field_t)))),
build1 (INDIRECT_REF, field_t, u));
t = build2 (COMPOUND_EXPR, TREE_TYPE (t), t, u);
}
u = fold_convert (TREE_TYPE (f_top), tmp_ha);
t = build2 (COMPOUND_EXPR, TREE_TYPE (f_top), t, u);
}
COND_EXPR_ELSE (cond2) = t;
addr = fold_convert (build_pointer_type (type), cond1);
addr = build_va_arg_indirect_ref (addr);
if (indirect_p)
addr = build_va_arg_indirect_ref (addr);
return addr;
}
static void
aarch64_setup_incoming_varargs (cumulative_args_t cum_v, machine_mode mode,
tree type, int *pretend_size ATTRIBUTE_UNUSED,
int no_rtl)
{
CUMULATIVE_ARGS *cum = get_cumulative_args (cum_v);
CUMULATIVE_ARGS local_cum;
int gr_saved = cfun->va_list_gpr_size;
int vr_saved = cfun->va_list_fpr_size;
local_cum = *cum;
aarch64_function_arg_advance (pack_cumulative_args(&local_cum), mode, type, true);
if (cfun->va_list_gpr_size)
gr_saved = MIN (NUM_ARG_REGS - local_cum.aapcs_ncrn,
cfun->va_list_gpr_size / UNITS_PER_WORD);
if (cfun->va_list_fpr_size)
vr_saved = MIN (NUM_FP_ARG_REGS - local_cum.aapcs_nvrn,
cfun->va_list_fpr_size / UNITS_PER_VREG);
if (!TARGET_FLOAT)
{
gcc_assert (local_cum.aapcs_nvrn == 0);
vr_saved = 0;
}
if (!no_rtl)
{
if (gr_saved > 0)
{
rtx ptr, mem;
ptr = plus_constant (Pmode, virtual_incoming_args_rtx,
- gr_saved * UNITS_PER_WORD);
mem = gen_frame_mem (BLKmode, ptr);
set_mem_alias_set (mem, get_varargs_alias_set ());
move_block_from_reg (local_cum.aapcs_ncrn + R0_REGNUM,
mem, gr_saved);
}
if (vr_saved > 0)
{
machine_mode mode = TImode;
int off, i, vr_start;
off = -ROUND_UP (gr_saved * UNITS_PER_WORD,
STACK_BOUNDARY / BITS_PER_UNIT);
off -= vr_saved * UNITS_PER_VREG;
vr_start = V0_REGNUM + local_cum.aapcs_nvrn;
for (i = 0; i < vr_saved; ++i)
{
rtx ptr, mem;
ptr = plus_constant (Pmode, virtual_incoming_args_rtx, off);
mem = gen_frame_mem (mode, ptr);
set_mem_alias_set (mem, get_varargs_alias_set ());
aarch64_emit_move (mem, gen_rtx_REG (mode, vr_start + i));
off += UNITS_PER_VREG;
}
}
}
cfun->machine->frame.saved_varargs_size
= (ROUND_UP (gr_saved * UNITS_PER_WORD,
STACK_BOUNDARY / BITS_PER_UNIT)
+ vr_saved * UNITS_PER_VREG);
}
static void
aarch64_conditional_register_usage (void)
{
int i;
if (!TARGET_FLOAT)
{
for (i = V0_REGNUM; i <= V31_REGNUM; i++)
{
fixed_regs[i] = 1;
call_used_regs[i] = 1;
}
}
if (!TARGET_SVE)
for (i = P0_REGNUM; i <= P15_REGNUM; i++)
{
fixed_regs[i] = 1;
call_used_regs[i] = 1;
}
}
static int
aapcs_vfp_sub_candidate (const_tree type, machine_mode *modep)
{
machine_mode mode;
HOST_WIDE_INT size;
switch (TREE_CODE (type))
{
case REAL_TYPE:
mode = TYPE_MODE (type);
if (mode != DFmode && mode != SFmode
&& mode != TFmode && mode != HFmode)
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 1;
break;
case COMPLEX_TYPE:
mode = TYPE_MODE (TREE_TYPE (type));
if (mode != DFmode && mode != SFmode
&& mode != TFmode && mode != HFmode)
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 2;
break;
case VECTOR_TYPE:
size = int_size_in_bytes (type);
switch (size)
{
case 8:
mode = V2SImode;
break;
case 16:
mode = V4SImode;
break;
default:
return -1;
}
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 1;
break;
case ARRAY_TYPE:
{
int count;
tree index = TYPE_DOMAIN (type);
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
count = aapcs_vfp_sub_candidate (TREE_TYPE (type), modep);
if (count == -1
|| !index
|| !TYPE_MAX_VALUE (index)
|| !tree_fits_uhwi_p (TYPE_MAX_VALUE (index))
|| !TYPE_MIN_VALUE (index)
|| !tree_fits_uhwi_p (TYPE_MIN_VALUE (index))
|| count < 0)
return -1;
count *= (1 + tree_to_uhwi (TYPE_MAX_VALUE (index))
- tree_to_uhwi (TYPE_MIN_VALUE (index)));
if (maybe_ne (wi::to_poly_wide (TYPE_SIZE (type)),
count * GET_MODE_BITSIZE (*modep)))
return -1;
return count;
}
case RECORD_TYPE:
{
int count = 0;
int sub_count;
tree field;
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
for (field = TYPE_FIELDS (type); field; field = TREE_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
sub_count = aapcs_vfp_sub_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count += sub_count;
}
if (maybe_ne (wi::to_poly_wide (TYPE_SIZE (type)),
count * GET_MODE_BITSIZE (*modep)))
return -1;
return count;
}
case UNION_TYPE:
case QUAL_UNION_TYPE:
{
int count = 0;
int sub_count;
tree field;
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
for (field = TYPE_FIELDS (type); field; field = TREE_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
sub_count = aapcs_vfp_sub_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count = count > sub_count ? count : sub_count;
}
if (maybe_ne (wi::to_poly_wide (TYPE_SIZE (type)),
count * GET_MODE_BITSIZE (*modep)))
return -1;
return count;
}
default:
break;
}
return -1;
}
static bool
aarch64_short_vector_p (const_tree type,
machine_mode mode)
{
poly_int64 size = -1;
if (type && TREE_CODE (type) == VECTOR_TYPE)
size = int_size_in_bytes (type);
else if (GET_MODE_CLASS (mode) == MODE_VECTOR_INT
|| GET_MODE_CLASS (mode) == MODE_VECTOR_FLOAT)
size = GET_MODE_SIZE (mode);
return known_eq (size, 8) || known_eq (size, 16);
}
static bool
aarch64_composite_type_p (const_tree type,
machine_mode mode)
{
if (aarch64_short_vector_p (type, mode))
return false;
if (type && (AGGREGATE_TYPE_P (type) || TREE_CODE (type) == COMPLEX_TYPE))
return true;
if (mode == BLKmode
|| GET_MODE_CLASS (mode) == MODE_COMPLEX_FLOAT
|| GET_MODE_CLASS (mode) == MODE_COMPLEX_INT)
return true;
return false;
}
static bool
aarch64_vfp_is_call_or_return_candidate (machine_mode mode,
const_tree type,
machine_mode *base_mode,
int *count,
bool *is_ha)
{
machine_mode new_mode = VOIDmode;
bool composite_p = aarch64_composite_type_p (type, mode);
if (is_ha != NULL) *is_ha = false;
if ((!composite_p && GET_MODE_CLASS (mode) == MODE_FLOAT)
|| aarch64_short_vector_p (type, mode))
{
*count = 1;
new_mode = mode;
}
else if (GET_MODE_CLASS (mode) == MODE_COMPLEX_FLOAT)
{
if (is_ha != NULL) *is_ha = true;
*count = 2;
new_mode = GET_MODE_INNER (mode);
}
else if (type && composite_p)
{
int ag_count = aapcs_vfp_sub_candidate (type, &new_mode);
if (ag_count > 0 && ag_count <= HA_MAX_NUM_FLDS)
{
if (is_ha != NULL) *is_ha = true;
*count = ag_count;
}
else
return false;
}
else
return false;
*base_mode = new_mode;
return true;
}
static rtx
aarch64_struct_value_rtx (tree fndecl ATTRIBUTE_UNUSED,
int incoming ATTRIBUTE_UNUSED)
{
return gen_rtx_REG (Pmode, AARCH64_STRUCT_VALUE_REGNUM);
}
static bool
aarch64_vector_mode_supported_p (machine_mode mode)
{
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
return vec_flags != 0 && (vec_flags & VEC_STRUCT) == 0;
}
static machine_mode
aarch64_simd_container_mode (scalar_mode mode, poly_int64 width)
{
if (TARGET_SVE && known_eq (width, BITS_PER_SVE_VECTOR))
switch (mode)
{
case E_DFmode:
return VNx2DFmode;
case E_SFmode:
return VNx4SFmode;
case E_HFmode:
return VNx8HFmode;
case E_DImode:
return VNx2DImode;
case E_SImode:
return VNx4SImode;
case E_HImode:
return VNx8HImode;
case E_QImode:
return VNx16QImode;
default:
return word_mode;
}
gcc_assert (known_eq (width, 64) || known_eq (width, 128));
if (TARGET_SIMD)
{
if (known_eq (width, 128))
switch (mode)
{
case E_DFmode:
return V2DFmode;
case E_SFmode:
return V4SFmode;
case E_HFmode:
return V8HFmode;
case E_SImode:
return V4SImode;
case E_HImode:
return V8HImode;
case E_QImode:
return V16QImode;
case E_DImode:
return V2DImode;
default:
break;
}
else
switch (mode)
{
case E_SFmode:
return V2SFmode;
case E_HFmode:
return V4HFmode;
case E_SImode:
return V2SImode;
case E_HImode:
return V4HImode;
case E_QImode:
return V8QImode;
default:
break;
}
}
return word_mode;
}
static machine_mode
aarch64_preferred_simd_mode (scalar_mode mode)
{
poly_int64 bits = TARGET_SVE ? BITS_PER_SVE_VECTOR : 128;
return aarch64_simd_container_mode (mode, bits);
}
static void
aarch64_autovectorize_vector_sizes (vector_sizes *sizes)
{
if (TARGET_SVE)
sizes->safe_push (BYTES_PER_SVE_VECTOR);
sizes->safe_push (16);
sizes->safe_push (8);
}
static const char *
aarch64_mangle_type (const_tree type)
{
if (lang_hooks.types_compatible_p (CONST_CAST_TREE (type), va_list_type))
return "St9__va_list";
if (TREE_CODE (type) == REAL_TYPE && TYPE_PRECISION (type) == 16)
return "Dh";
if (TYPE_NAME (type) != NULL)
return aarch64_mangle_builtin_type (type);
return NULL;
}
static rtx_insn *
aarch64_prev_real_insn (rtx_insn *insn)
{
if (!insn)
return NULL;
do
{
insn = prev_real_insn (insn);
}
while (insn && recog_memoized (insn) < 0);
return insn;
}
static bool
is_madd_op (enum attr_type t1)
{
unsigned int i;
enum attr_type mlatypes[] = {
TYPE_MLA, TYPE_MLAS, TYPE_SMLAD, TYPE_SMLADX, TYPE_SMLAL, TYPE_SMLALD,
TYPE_SMLALS, TYPE_SMLALXY, TYPE_SMLAWX, TYPE_SMLAWY, TYPE_SMLAXY,
TYPE_SMMLA, TYPE_UMLAL, TYPE_UMLALS,TYPE_SMLSD, TYPE_SMLSDX, TYPE_SMLSLD
};
for (i = 0; i < sizeof (mlatypes) / sizeof (enum attr_type); i++)
{
if (t1 == mlatypes[i])
return true;
}
return false;
}
static bool
dep_between_memop_and_curr (rtx memop)
{
rtx load_reg;
int opno;
gcc_assert (GET_CODE (memop) == SET);
if (!REG_P (SET_DEST (memop)))
return false;
load_reg = SET_DEST (memop);
for (opno = 1; opno < recog_data.n_operands; opno++)
{
rtx operand = recog_data.operand[opno];
if (REG_P (operand)
&& reg_overlap_mentioned_p (load_reg, operand))
return true;
}
return false;
}
bool
aarch64_madd_needs_nop (rtx_insn* insn)
{
enum attr_type attr_type;
rtx_insn *prev;
rtx body;
if (!TARGET_FIX_ERR_A53_835769)
return false;
if (!INSN_P (insn) || recog_memoized (insn) < 0)
return false;
attr_type = get_attr_type (insn);
if (!is_madd_op (attr_type))
return false;
prev = aarch64_prev_real_insn (insn);
extract_constrain_insn_cached (insn);
if (!prev || !contains_mem_rtx_p (PATTERN (prev)))
return false;
body = single_set (prev);
if (GET_MODE (recog_data.operand[0]) == DImode
&& (!body || !dep_between_memop_and_curr (body)))
return true;
return false;
}
void
aarch64_final_prescan_insn (rtx_insn *insn)
{
if (aarch64_madd_needs_nop (insn))
fprintf (asm_out_file, "\tnop 
}
bool
aarch64_sve_index_immediate_p (rtx base_or_step)
{
return (CONST_INT_P (base_or_step)
&& IN_RANGE (INTVAL (base_or_step), -16, 15));
}
bool
aarch64_sve_arith_immediate_p (rtx x, bool negate_p)
{
rtx elt;
if (!const_vec_duplicate_p (x, &elt)
|| !CONST_INT_P (elt))
return false;
HOST_WIDE_INT val = INTVAL (elt);
if (negate_p)
val = -val;
val &= GET_MODE_MASK (GET_MODE_INNER (GET_MODE (x)));
if (val & 0xff)
return IN_RANGE (val, 0, 0xff);
return IN_RANGE (val, 0, 0xff00);
}
bool
aarch64_sve_bitmask_immediate_p (rtx x)
{
rtx elt;
return (const_vec_duplicate_p (x, &elt)
&& CONST_INT_P (elt)
&& aarch64_bitmask_imm (INTVAL (elt),
GET_MODE_INNER (GET_MODE (x))));
}
bool
aarch64_sve_dup_immediate_p (rtx x)
{
rtx elt;
if (!const_vec_duplicate_p (x, &elt)
|| !CONST_INT_P (elt))
return false;
HOST_WIDE_INT val = INTVAL (elt);
if (val & 0xff)
return IN_RANGE (val, -0x80, 0x7f);
return IN_RANGE (val, -0x8000, 0x7f00);
}
bool
aarch64_sve_cmp_immediate_p (rtx x, bool signed_p)
{
rtx elt;
return (const_vec_duplicate_p (x, &elt)
&& CONST_INT_P (elt)
&& (signed_p
? IN_RANGE (INTVAL (elt), -16, 15)
: IN_RANGE (INTVAL (elt), 0, 127)));
}
bool
aarch64_sve_float_arith_immediate_p (rtx x, bool negate_p)
{
rtx elt;
REAL_VALUE_TYPE r;
if (!const_vec_duplicate_p (x, &elt)
|| GET_CODE (elt) != CONST_DOUBLE)
return false;
r = *CONST_DOUBLE_REAL_VALUE (elt);
if (negate_p)
r = real_value_negate (&r);
if (real_equal (&r, &dconst1))
return true;
if (real_equal (&r, &dconsthalf))
return true;
return false;
}
bool
aarch64_sve_float_mul_immediate_p (rtx x)
{
rtx elt;
return (const_vec_duplicate_p (x, &elt)
&& GET_CODE (elt) == CONST_DOUBLE
&& real_equal (CONST_DOUBLE_REAL_VALUE (elt), &dconsthalf));
}
static bool
aarch64_advsimd_valid_immediate_hs (unsigned int val32,
simd_immediate_info *info,
enum simd_immediate_check which,
simd_immediate_info::insn_type insn)
{
for (unsigned int shift = 0; shift < 32; shift += 8)
if ((val32 & (0xff << shift)) == val32)
{
if (info)
*info = simd_immediate_info (SImode, val32 >> shift, insn,
simd_immediate_info::LSL, shift);
return true;
}
unsigned int imm16 = val32 & 0xffff;
if (imm16 == (val32 >> 16))
for (unsigned int shift = 0; shift < 16; shift += 8)
if ((imm16 & (0xff << shift)) == imm16)
{
if (info)
*info = simd_immediate_info (HImode, imm16 >> shift, insn,
simd_immediate_info::LSL, shift);
return true;
}
if (which == AARCH64_CHECK_MOV)
for (unsigned int shift = 8; shift < 24; shift += 8)
{
unsigned int low = (1 << shift) - 1;
if (((val32 & (0xff << shift)) | low) == val32)
{
if (info)
*info = simd_immediate_info (SImode, val32 >> shift, insn,
simd_immediate_info::MSL, shift);
return true;
}
}
return false;
}
static bool
aarch64_advsimd_valid_immediate (unsigned HOST_WIDE_INT val64,
simd_immediate_info *info,
enum simd_immediate_check which)
{
unsigned int val32 = val64 & 0xffffffff;
unsigned int val16 = val64 & 0xffff;
unsigned int val8 = val64 & 0xff;
if (val32 == (val64 >> 32))
{
if ((which & AARCH64_CHECK_ORR) != 0
&& aarch64_advsimd_valid_immediate_hs (val32, info, which,
simd_immediate_info::MOV))
return true;
if ((which & AARCH64_CHECK_BIC) != 0
&& aarch64_advsimd_valid_immediate_hs (~val32, info, which,
simd_immediate_info::MVN))
return true;
if (which == AARCH64_CHECK_MOV
&& val16 == (val32 >> 16)
&& val8 == (val16 >> 8))
{
if (info)
*info = simd_immediate_info (QImode, val8);
return true;
}
}
if (which == AARCH64_CHECK_MOV)
{
unsigned int i;
for (i = 0; i < 64; i += 8)
{
unsigned char byte = (val64 >> i) & 0xff;
if (byte != 0 && byte != 0xff)
break;
}
if (i == 64)
{
if (info)
*info = simd_immediate_info (DImode, val64);
return true;
}
}
return false;
}
static bool
aarch64_sve_valid_immediate (unsigned HOST_WIDE_INT val64,
simd_immediate_info *info)
{
scalar_int_mode mode = DImode;
unsigned int val32 = val64 & 0xffffffff;
if (val32 == (val64 >> 32))
{
mode = SImode;
unsigned int val16 = val32 & 0xffff;
if (val16 == (val32 >> 16))
{
mode = HImode;
unsigned int val8 = val16 & 0xff;
if (val8 == (val16 >> 8))
mode = QImode;
}
}
HOST_WIDE_INT val = trunc_int_for_mode (val64, mode);
if (IN_RANGE (val, -0x80, 0x7f))
{
if (info)
*info = simd_immediate_info (mode, val);
return true;
}
if ((val & 0xff) == 0 && IN_RANGE (val, -0x8000, 0x7f00))
{
if (info)
*info = simd_immediate_info (mode, val);
return true;
}
if (aarch64_bitmask_imm (val64, mode))
{
if (info)
*info = simd_immediate_info (mode, val);
return true;
}
return false;
}
bool
aarch64_simd_valid_immediate (rtx op, simd_immediate_info *info,
enum simd_immediate_check which)
{
machine_mode mode = GET_MODE (op);
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
if (vec_flags == 0 || vec_flags == (VEC_ADVSIMD | VEC_STRUCT))
return false;
scalar_mode elt_mode = GET_MODE_INNER (mode);
rtx base, step;
unsigned int n_elts;
if (GET_CODE (op) == CONST_VECTOR
&& CONST_VECTOR_DUPLICATE_P (op))
n_elts = CONST_VECTOR_NPATTERNS (op);
else if ((vec_flags & VEC_SVE_DATA)
&& const_vec_series_p (op, &base, &step))
{
gcc_assert (GET_MODE_CLASS (mode) == MODE_VECTOR_INT);
if (!aarch64_sve_index_immediate_p (base)
|| !aarch64_sve_index_immediate_p (step))
return false;
if (info)
*info = simd_immediate_info (elt_mode, base, step);
return true;
}
else if (GET_CODE (op) == CONST_VECTOR
&& CONST_VECTOR_NUNITS (op).is_constant (&n_elts))
;
else
return false;
if (vec_flags & VEC_SVE_PRED)
return (op == CONST0_RTX (mode)
|| op == CONSTM1_RTX (mode));
scalar_float_mode elt_float_mode;
if (n_elts == 1
&& is_a <scalar_float_mode> (elt_mode, &elt_float_mode))
{
rtx elt = CONST_VECTOR_ENCODED_ELT (op, 0);
if (aarch64_float_const_zero_rtx_p (elt)
|| aarch64_float_const_representable_p (elt))
{
if (info)
*info = simd_immediate_info (elt_float_mode, elt);
return true;
}
}
unsigned int elt_size = GET_MODE_SIZE (elt_mode);
if (elt_size > 8)
return false;
scalar_int_mode elt_int_mode = int_mode_for_mode (elt_mode).require ();
auto_vec<unsigned char, 16> bytes;
bytes.reserve (n_elts * elt_size);
for (unsigned int i = 0; i < n_elts; i++)
{
bool swap_p = ((vec_flags & VEC_ADVSIMD) != 0 && BYTES_BIG_ENDIAN);
rtx elt = CONST_VECTOR_ELT (op, swap_p ? (n_elts - 1 - i) : i);
if (elt_mode != elt_int_mode)
elt = gen_lowpart (elt_int_mode, elt);
if (!CONST_INT_P (elt))
return false;
unsigned HOST_WIDE_INT elt_val = INTVAL (elt);
for (unsigned int byte = 0; byte < elt_size; byte++)
{
bytes.quick_push (elt_val & 0xff);
elt_val >>= BITS_PER_UNIT;
}
}
unsigned int nbytes = bytes.length ();
for (unsigned i = 8; i < nbytes; ++i)
if (bytes[i] != bytes[i - 8])
return false;
unsigned HOST_WIDE_INT val64 = 0;
for (unsigned int i = 0; i < 8; i++)
val64 |= ((unsigned HOST_WIDE_INT) bytes[i % nbytes]
<< (i * BITS_PER_UNIT));
if (vec_flags & VEC_SVE_DATA)
return aarch64_sve_valid_immediate (val64, info);
else
return aarch64_advsimd_valid_immediate (val64, info, which);
}
rtx
aarch64_check_zero_based_sve_index_immediate (rtx x)
{
rtx base, step;
if (const_vec_series_p (x, &base, &step)
&& base == const0_rtx
&& aarch64_sve_index_immediate_p (step))
return step;
return NULL_RTX;
}
bool
aarch64_simd_shift_imm_p (rtx x, machine_mode mode, bool left)
{
int bit_width = GET_MODE_UNIT_SIZE (mode) * BITS_PER_UNIT;
if (left)
return aarch64_const_vec_all_same_in_range_p (x, 0, bit_width - 1);
else
return aarch64_const_vec_all_same_in_range_p (x, 1, bit_width);
}
rtx
aarch64_mask_from_zextract_ops (rtx width, rtx pos)
{
gcc_assert (CONST_INT_P (width));
gcc_assert (CONST_INT_P (pos));
unsigned HOST_WIDE_INT mask
= ((unsigned HOST_WIDE_INT) 1 << UINTVAL (width)) - 1;
return GEN_INT (mask << UINTVAL (pos));
}
bool
aarch64_mov_operand_p (rtx x, machine_mode mode)
{
if (GET_CODE (x) == HIGH
&& aarch64_valid_symref (XEXP (x, 0), GET_MODE (XEXP (x, 0))))
return true;
if (CONST_INT_P (x))
return true;
if (VECTOR_MODE_P (GET_MODE (x)))
return aarch64_simd_valid_immediate (x, NULL);
if (GET_CODE (x) == SYMBOL_REF && mode == DImode && CONSTANT_ADDRESS_P (x))
return true;
if (aarch64_sve_cnt_immediate_p (x))
return true;
return aarch64_classify_symbolic_expression (x)
== SYMBOL_TINY_ABSOLUTE;
}
rtx
aarch64_simd_gen_const_vector_dup (machine_mode mode, HOST_WIDE_INT val)
{
rtx c = gen_int_mode (val, GET_MODE_INNER (mode));
return gen_const_vec_duplicate (mode, c);
}
bool
aarch64_simd_scalar_immediate_valid_for_move (rtx op, scalar_int_mode mode)
{
machine_mode vmode;
vmode = aarch64_simd_container_mode (mode, 64);
rtx op_v = aarch64_simd_gen_const_vector_dup (vmode, INTVAL (op));
return aarch64_simd_valid_immediate (op_v, NULL);
}
rtx
aarch64_simd_vect_par_cnst_half (machine_mode mode, int nunits, bool high)
{
rtvec v = rtvec_alloc (nunits / 2);
int high_base = nunits / 2;
int low_base = 0;
int base;
rtx t1;
int i;
if (BYTES_BIG_ENDIAN)
base = high ? low_base : high_base;
else
base = high ? high_base : low_base;
for (i = 0; i < nunits / 2; i++)
RTVEC_ELT (v, i) = GEN_INT (base + i);
t1 = gen_rtx_PARALLEL (mode, v);
return t1;
}
bool
aarch64_simd_check_vect_par_cnst_half (rtx op, machine_mode mode,
bool high)
{
int nelts;
if (!VECTOR_MODE_P (mode) || !GET_MODE_NUNITS (mode).is_constant (&nelts))
return false;
rtx ideal = aarch64_simd_vect_par_cnst_half (mode, nelts, high);
HOST_WIDE_INT count_op = XVECLEN (op, 0);
HOST_WIDE_INT count_ideal = XVECLEN (ideal, 0);
int i = 0;
if (count_op != count_ideal)
return false;
for (i = 0; i < count_ideal; i++)
{
rtx elt_op = XVECEXP (op, 0, i);
rtx elt_ideal = XVECEXP (ideal, 0, i);
if (!CONST_INT_P (elt_op)
|| INTVAL (elt_ideal) != INTVAL (elt_op))
return false;
}
return true;
}
void
aarch64_simd_lane_bounds (rtx operand, HOST_WIDE_INT low, HOST_WIDE_INT high,
const_tree exp)
{
HOST_WIDE_INT lane;
gcc_assert (CONST_INT_P (operand));
lane = INTVAL (operand);
if (lane < low || lane >= high)
{
if (exp)
error ("%Klane %wd out of range %wd - %wd", exp, lane, low, high - 1);
else
error ("lane %wd out of range %wd - %wd", lane, low, high - 1);
}
}
rtx
aarch64_endian_lane_rtx (machine_mode mode, unsigned int n)
{
return gen_int_mode (ENDIAN_LANE_N (GET_MODE_NUNITS (mode), n), SImode);
}
bool
aarch64_simd_mem_operand_p (rtx op)
{
return MEM_P (op) && (GET_CODE (XEXP (op, 0)) == POST_INC
|| REG_P (XEXP (op, 0)));
}
bool
aarch64_sve_ld1r_operand_p (rtx op)
{
struct aarch64_address_info addr;
scalar_mode mode;
return (MEM_P (op)
&& is_a <scalar_mode> (GET_MODE (op), &mode)
&& aarch64_classify_address (&addr, XEXP (op, 0), mode, false)
&& addr.type == ADDRESS_REG_IMM
&& offset_6bit_unsigned_scaled_p (mode, addr.const_offset));
}
bool
aarch64_sve_ldr_operand_p (rtx op)
{
struct aarch64_address_info addr;
return (MEM_P (op)
&& aarch64_classify_address (&addr, XEXP (op, 0), GET_MODE (op),
false, ADDR_QUERY_ANY)
&& addr.type == ADDRESS_REG_IMM);
}
bool
aarch64_sve_struct_memory_operand_p (rtx op)
{
if (!MEM_P (op))
return false;
machine_mode mode = GET_MODE (op);
struct aarch64_address_info addr;
if (!aarch64_classify_address (&addr, XEXP (op, 0), SVE_BYTE_MODE, false,
ADDR_QUERY_ANY)
|| addr.type != ADDRESS_REG_IMM)
return false;
poly_int64 first = addr.const_offset;
poly_int64 last = first + GET_MODE_SIZE (mode) - BYTES_PER_SVE_VECTOR;
return (offset_4bit_signed_scaled_p (SVE_BYTE_MODE, first)
&& offset_4bit_signed_scaled_p (SVE_BYTE_MODE, last));
}
void
aarch64_simd_emit_reg_reg_move (rtx *operands, machine_mode mode,
unsigned int count)
{
unsigned int i;
int rdest = REGNO (operands[0]);
int rsrc = REGNO (operands[1]);
if (!reg_overlap_mentioned_p (operands[0], operands[1])
|| rdest < rsrc)
for (i = 0; i < count; i++)
emit_move_insn (gen_rtx_REG (mode, rdest + i),
gen_rtx_REG (mode, rsrc + i));
else
for (i = 0; i < count; i++)
emit_move_insn (gen_rtx_REG (mode, rdest + count - i - 1),
gen_rtx_REG (mode, rsrc + count - i - 1));
}
int
aarch64_simd_attr_length_rglist (machine_mode mode)
{
return (GET_MODE_SIZE (mode).to_constant () / UNITS_PER_VREG) * 4;
}
static HOST_WIDE_INT
aarch64_simd_vector_alignment (const_tree type)
{
if (TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return GET_MODE_CLASS (TYPE_MODE (type)) == MODE_VECTOR_BOOL ? 16 : 128;
HOST_WIDE_INT align = tree_to_shwi (TYPE_SIZE (type));
return MIN (align, 128);
}
static HOST_WIDE_INT
aarch64_vectorize_preferred_vector_alignment (const_tree type)
{
if (aarch64_sve_data_mode_p (TYPE_MODE (type)))
{
HOST_WIDE_INT result;
if (!BITS_PER_SVE_VECTOR.is_constant (&result))
result = TYPE_ALIGN (TREE_TYPE (type));
return result;
}
return TYPE_ALIGN (type);
}
static bool
aarch64_simd_vector_alignment_reachable (const_tree type, bool is_packed)
{
if (is_packed)
return false;
if (TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST
&& (wi::to_widest (TYPE_SIZE (type))
!= aarch64_vectorize_preferred_vector_alignment (type)))
return false;
return true;
}
static bool
aarch64_builtin_support_vector_misalignment (machine_mode mode,
const_tree type, int misalignment,
bool is_packed)
{
if (TARGET_SIMD && STRICT_ALIGNMENT)
{
if (optab_handler (movmisalign_optab, mode) == CODE_FOR_nothing)
return false;
if (misalignment == -1)
return false;
}
return default_builtin_support_vector_misalignment (mode, type, misalignment,
is_packed);
}
static rtx
aarch64_simd_dup_constant (rtx vals)
{
machine_mode mode = GET_MODE (vals);
machine_mode inner_mode = GET_MODE_INNER (mode);
rtx x;
if (!const_vec_duplicate_p (vals, &x))
return NULL_RTX;
x = copy_to_mode_reg (inner_mode, x);
return gen_vec_duplicate (mode, x);
}
static rtx
aarch64_simd_make_constant (rtx vals)
{
machine_mode mode = GET_MODE (vals);
rtx const_dup;
rtx const_vec = NULL_RTX;
int n_const = 0;
int i;
if (GET_CODE (vals) == CONST_VECTOR)
const_vec = vals;
else if (GET_CODE (vals) == PARALLEL)
{
int n_elts = XVECLEN (vals, 0);
for (i = 0; i < n_elts; ++i)
{
rtx x = XVECEXP (vals, 0, i);
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
n_const++;
}
if (n_const == n_elts)
const_vec = gen_rtx_CONST_VECTOR (mode, XVEC (vals, 0));
}
else
gcc_unreachable ();
if (const_vec != NULL_RTX
&& aarch64_simd_valid_immediate (const_vec, NULL))
return const_vec;
else if ((const_dup = aarch64_simd_dup_constant (vals)) != NULL_RTX)
return const_dup;
else if (const_vec != NULL_RTX)
return const_vec;
else
return NULL_RTX;
}
void
aarch64_expand_vector_init (rtx target, rtx vals)
{
machine_mode mode = GET_MODE (target);
scalar_mode inner_mode = GET_MODE_INNER (mode);
int n_elts = XVECLEN (vals, 0);
int n_var = 0;
rtx any_const = NULL_RTX;
rtx v0 = XVECEXP (vals, 0, 0);
bool all_same = true;
for (int i = 0; i < n_elts; ++i)
{
rtx x = XVECEXP (vals, 0, i);
if (!(CONST_INT_P (x) || CONST_DOUBLE_P (x)))
++n_var;
else
any_const = x;
all_same &= rtx_equal_p (x, v0);
}
if (n_var == 0)
{
rtx constant = aarch64_simd_make_constant (vals);
if (constant != NULL_RTX)
{
emit_move_insn (target, constant);
return;
}
}
if (all_same)
{
rtx x = copy_to_mode_reg (inner_mode, v0);
aarch64_emit_move (target, gen_vec_duplicate (mode, x));
return;
}
enum insn_code icode = optab_handler (vec_set_optab, mode);
gcc_assert (icode != CODE_FOR_nothing);
if (n_var == n_elts && n_elts <= 16)
{
int matches[16][2] = {0};
for (int i = 0; i < n_elts; i++)
{
for (int j = 0; j <= i; j++)
{
if (rtx_equal_p (XVECEXP (vals, 0, i), XVECEXP (vals, 0, j)))
{
matches[i][0] = j;
matches[j][1]++;
break;
}
}
}
int maxelement = 0;
int maxv = 0;
for (int i = 0; i < n_elts; i++)
if (matches[i][1] > maxv)
{
maxelement = i;
maxv = matches[i][1];
}
rtx x = copy_to_mode_reg (inner_mode, XVECEXP (vals, 0, maxelement));
aarch64_emit_move (target, gen_vec_duplicate (mode, x));
for (int i = 0; i < n_elts; i++)
{
rtx x = XVECEXP (vals, 0, i);
if (matches[i][0] == maxelement)
continue;
x = copy_to_mode_reg (inner_mode, x);
emit_insn (GEN_FCN (icode) (target, x, GEN_INT (i)));
}
return;
}
if (n_var != n_elts)
{
rtx copy = copy_rtx (vals);
for (int i = 0; i < n_elts; i++)
{
rtx x = XVECEXP (vals, 0, i);
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
continue;
rtx subst = any_const;
for (int bit = n_elts / 2; bit > 0; bit /= 2)
{
rtx test = XVECEXP (copy, 0, i ^ bit);
if (CONST_INT_P (test) || CONST_DOUBLE_P (test))
{
subst = test;
break;
}
}
XVECEXP (copy, 0, i) = subst;
}
aarch64_expand_vector_init (target, copy);
}
for (int i = 0; i < n_elts; i++)
{
rtx x = XVECEXP (vals, 0, i);
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
continue;
x = copy_to_mode_reg (inner_mode, x);
emit_insn (GEN_FCN (icode) (target, x, GEN_INT (i)));
}
}
static unsigned HOST_WIDE_INT
aarch64_shift_truncation_mask (machine_mode mode)
{
if (!SHIFT_COUNT_TRUNCATED || aarch64_vector_data_mode_p (mode))
return 0;
return GET_MODE_UNIT_BITSIZE (mode) - 1;
}
int
aarch64_asm_preferred_eh_data_format (int code ATTRIBUTE_UNUSED, int global)
{
int type;
switch (aarch64_cmodel)
{
case AARCH64_CMODEL_TINY:
case AARCH64_CMODEL_TINY_PIC:
case AARCH64_CMODEL_SMALL:
case AARCH64_CMODEL_SMALL_PIC:
case AARCH64_CMODEL_SMALL_SPIC:
type = DW_EH_PE_sdata4;
break;
default:
type = DW_EH_PE_sdata8;
break;
}
return (global ? DW_EH_PE_indirect : 0) | DW_EH_PE_pcrel | type;
}
static std::string aarch64_last_printed_arch_string;
static std::string aarch64_last_printed_tune_string;
void
aarch64_declare_function_name (FILE *stream, const char* name,
tree fndecl)
{
tree target_parts = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
struct cl_target_option *targ_options;
if (target_parts)
targ_options = TREE_TARGET_OPTION (target_parts);
else
targ_options = TREE_TARGET_OPTION (target_option_current_node);
gcc_assert (targ_options);
const struct processor *this_arch
= aarch64_get_arch (targ_options->x_explicit_arch);
unsigned long isa_flags = targ_options->x_aarch64_isa_flags;
std::string extension
= aarch64_get_extension_string_for_isa_flags (isa_flags,
this_arch->flags);
std::string to_print = this_arch->name + extension;
if (to_print != aarch64_last_printed_arch_string)
{
asm_fprintf (asm_out_file, "\t.arch %s\n", to_print.c_str ());
aarch64_last_printed_arch_string = to_print;
}
const struct processor *this_tune
= aarch64_get_tune_cpu (targ_options->x_explicit_tune_core);
if (flag_debug_asm && aarch64_last_printed_tune_string != this_tune->name)
{
asm_fprintf (asm_out_file, "\t" ASM_COMMENT_START ".tune %s\n",
this_tune->name);
aarch64_last_printed_tune_string = this_tune->name;
}
ASM_OUTPUT_TYPE_DIRECTIVE (stream, name, "function");
ASM_OUTPUT_LABEL (stream, name);
}
static void
aarch64_start_file (void)
{
struct cl_target_option *default_options
= TREE_TARGET_OPTION (target_option_default_node);
const struct processor *default_arch
= aarch64_get_arch (default_options->x_explicit_arch);
unsigned long default_isa_flags = default_options->x_aarch64_isa_flags;
std::string extension
= aarch64_get_extension_string_for_isa_flags (default_isa_flags,
default_arch->flags);
aarch64_last_printed_arch_string = default_arch->name + extension;
aarch64_last_printed_tune_string = "";
asm_fprintf (asm_out_file, "\t.arch %s\n",
aarch64_last_printed_arch_string.c_str ());
default_file_start ();
}
static void
aarch64_emit_load_exclusive (machine_mode mode, rtx rval,
rtx mem, rtx model_rtx)
{
rtx (*gen) (rtx, rtx, rtx);
switch (mode)
{
case E_QImode: gen = gen_aarch64_load_exclusiveqi; break;
case E_HImode: gen = gen_aarch64_load_exclusivehi; break;
case E_SImode: gen = gen_aarch64_load_exclusivesi; break;
case E_DImode: gen = gen_aarch64_load_exclusivedi; break;
default:
gcc_unreachable ();
}
emit_insn (gen (rval, mem, model_rtx));
}
static void
aarch64_emit_store_exclusive (machine_mode mode, rtx bval,
rtx rval, rtx mem, rtx model_rtx)
{
rtx (*gen) (rtx, rtx, rtx, rtx);
switch (mode)
{
case E_QImode: gen = gen_aarch64_store_exclusiveqi; break;
case E_HImode: gen = gen_aarch64_store_exclusivehi; break;
case E_SImode: gen = gen_aarch64_store_exclusivesi; break;
case E_DImode: gen = gen_aarch64_store_exclusivedi; break;
default:
gcc_unreachable ();
}
emit_insn (gen (bval, rval, mem, model_rtx));
}
static void
aarch64_emit_unlikely_jump (rtx insn)
{
rtx_insn *jump = emit_jump_insn (insn);
add_reg_br_prob_note (jump, profile_probability::very_unlikely ());
}
void
aarch64_expand_compare_and_swap (rtx operands[])
{
rtx bval, rval, mem, oldval, newval, is_weak, mod_s, mod_f, x;
machine_mode mode, cmp_mode;
typedef rtx (*gen_cas_fn) (rtx, rtx, rtx, rtx, rtx, rtx, rtx);
int idx;
gen_cas_fn gen;
const gen_cas_fn split_cas[] =
{
gen_aarch64_compare_and_swapqi,
gen_aarch64_compare_and_swaphi,
gen_aarch64_compare_and_swapsi,
gen_aarch64_compare_and_swapdi
};
const gen_cas_fn atomic_cas[] =
{
gen_aarch64_compare_and_swapqi_lse,
gen_aarch64_compare_and_swaphi_lse,
gen_aarch64_compare_and_swapsi_lse,
gen_aarch64_compare_and_swapdi_lse
};
bval = operands[0];
rval = operands[1];
mem = operands[2];
oldval = operands[3];
newval = operands[4];
is_weak = operands[5];
mod_s = operands[6];
mod_f = operands[7];
mode = GET_MODE (mem);
cmp_mode = mode;
if (is_mm_acquire (memmodel_from_int (INTVAL (mod_f)))
&& is_mm_release (memmodel_from_int (INTVAL (mod_s))))
mod_s = GEN_INT (MEMMODEL_ACQ_REL);
switch (mode)
{
case E_QImode:
case E_HImode:
cmp_mode = SImode;
rval = gen_reg_rtx (SImode);
oldval = convert_modes (SImode, mode, oldval, true);
case E_SImode:
case E_DImode:
if (!aarch64_plus_operand (oldval, mode))
oldval = force_reg (cmp_mode, oldval);
break;
default:
gcc_unreachable ();
}
switch (mode)
{
case E_QImode: idx = 0; break;
case E_HImode: idx = 1; break;
case E_SImode: idx = 2; break;
case E_DImode: idx = 3; break;
default:
gcc_unreachable ();
}
if (TARGET_LSE)
gen = atomic_cas[idx];
else
gen = split_cas[idx];
emit_insn (gen (rval, mem, oldval, newval, is_weak, mod_s, mod_f));
if (mode == QImode || mode == HImode)
emit_move_insn (operands[1], gen_lowpart (mode, rval));
x = gen_rtx_REG (CCmode, CC_REGNUM);
x = gen_rtx_EQ (SImode, x, const0_rtx);
emit_insn (gen_rtx_SET (bval, x));
}
bool
aarch64_atomic_ldop_supported_p (enum rtx_code code)
{
if (!TARGET_LSE)
return false;
switch (code)
{
case SET:
case AND:
case IOR:
case XOR:
case MINUS:
case PLUS:
return true;
default:
return false;
}
}
static void
aarch64_emit_post_barrier (enum memmodel model)
{
const enum memmodel base_model = memmodel_base (model);
if (is_mm_sync (model)
&& (base_model == MEMMODEL_ACQUIRE
|| base_model == MEMMODEL_ACQ_REL
|| base_model == MEMMODEL_SEQ_CST))
{
emit_insn (gen_mem_thread_fence (GEN_INT (MEMMODEL_SEQ_CST)));
}
}
void
aarch64_gen_atomic_cas (rtx rval, rtx mem,
rtx expected, rtx desired,
rtx model)
{
rtx (*gen) (rtx, rtx, rtx, rtx);
machine_mode mode;
mode = GET_MODE (mem);
switch (mode)
{
case E_QImode: gen = gen_aarch64_atomic_casqi; break;
case E_HImode: gen = gen_aarch64_atomic_cashi; break;
case E_SImode: gen = gen_aarch64_atomic_cassi; break;
case E_DImode: gen = gen_aarch64_atomic_casdi; break;
default:
gcc_unreachable ();
}
emit_insn (gen_rtx_SET (rval, expected));
emit_insn (gen (rval, mem, desired, model));
aarch64_gen_compare_reg (EQ, rval, expected);
}
void
aarch64_split_compare_and_swap (rtx operands[])
{
rtx rval, mem, oldval, newval, scratch;
machine_mode mode;
bool is_weak;
rtx_code_label *label1, *label2;
rtx x, cond;
enum memmodel model;
rtx model_rtx;
rval = operands[0];
mem = operands[1];
oldval = operands[2];
newval = operands[3];
is_weak = (operands[4] != const0_rtx);
model_rtx = operands[5];
scratch = operands[7];
mode = GET_MODE (mem);
model = memmodel_from_int (INTVAL (model_rtx));
bool strong_zero_p = !is_weak && oldval == const0_rtx;
label1 = NULL;
if (!is_weak)
{
label1 = gen_label_rtx ();
emit_label (label1);
}
label2 = gen_label_rtx ();
if (is_mm_sync (model))
aarch64_emit_load_exclusive (mode, rval, mem,
GEN_INT (MEMMODEL_RELAXED));
else
aarch64_emit_load_exclusive (mode, rval, mem, model_rtx);
if (strong_zero_p)
{
x = gen_rtx_NE (VOIDmode, rval, const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (Pmode, label2), pc_rtx);
aarch64_emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
}
else
{
cond = aarch64_gen_compare_reg (NE, rval, oldval);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (Pmode, label2), pc_rtx);
aarch64_emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
}
aarch64_emit_store_exclusive (mode, scratch, mem, newval, model_rtx);
if (!is_weak)
{
x = gen_rtx_NE (VOIDmode, scratch, const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (Pmode, label1), pc_rtx);
aarch64_emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
}
else
{
cond = gen_rtx_REG (CCmode, CC_REGNUM);
x = gen_rtx_COMPARE (CCmode, scratch, const0_rtx);
emit_insn (gen_rtx_SET (cond, x));
}
emit_label (label2);
if (strong_zero_p)
{
cond = gen_rtx_REG (CCmode, CC_REGNUM);
x = gen_rtx_COMPARE (CCmode, rval, const0_rtx);
emit_insn (gen_rtx_SET (cond, x));
}
if (is_mm_sync (model))
aarch64_emit_post_barrier (model);
}
static void
aarch64_emit_bic (machine_mode mode, rtx dst, rtx s1, rtx s2, int shift)
{
rtx shift_rtx = GEN_INT (shift);
rtx (*gen) (rtx, rtx, rtx, rtx);
switch (mode)
{
case E_SImode: gen = gen_and_one_cmpl_lshrsi3; break;
case E_DImode: gen = gen_and_one_cmpl_lshrdi3; break;
default:
gcc_unreachable ();
}
emit_insn (gen (dst, s2, shift_rtx, s1));
}
static void
aarch64_emit_atomic_swap (machine_mode mode, rtx dst, rtx value,
rtx mem, rtx model)
{
rtx (*gen) (rtx, rtx, rtx, rtx);
switch (mode)
{
case E_QImode: gen = gen_aarch64_atomic_swpqi; break;
case E_HImode: gen = gen_aarch64_atomic_swphi; break;
case E_SImode: gen = gen_aarch64_atomic_swpsi; break;
case E_DImode: gen = gen_aarch64_atomic_swpdi; break;
default:
gcc_unreachable ();
}
emit_insn (gen (dst, mem, value, model));
}
enum aarch64_atomic_load_op_code
{
AARCH64_LDOP_PLUS,	
AARCH64_LDOP_XOR,	
AARCH64_LDOP_OR,	
AARCH64_LDOP_BIC	
};
static void
aarch64_emit_atomic_load_op (enum aarch64_atomic_load_op_code code,
machine_mode mode, rtx dst, rtx src,
rtx mem, rtx model)
{
typedef rtx (*aarch64_atomic_load_op_fn) (rtx, rtx, rtx, rtx);
const aarch64_atomic_load_op_fn plus[] =
{
gen_aarch64_atomic_loadaddqi,
gen_aarch64_atomic_loadaddhi,
gen_aarch64_atomic_loadaddsi,
gen_aarch64_atomic_loadadddi
};
const aarch64_atomic_load_op_fn eor[] =
{
gen_aarch64_atomic_loadeorqi,
gen_aarch64_atomic_loadeorhi,
gen_aarch64_atomic_loadeorsi,
gen_aarch64_atomic_loadeordi
};
const aarch64_atomic_load_op_fn ior[] =
{
gen_aarch64_atomic_loadsetqi,
gen_aarch64_atomic_loadsethi,
gen_aarch64_atomic_loadsetsi,
gen_aarch64_atomic_loadsetdi
};
const aarch64_atomic_load_op_fn bic[] =
{
gen_aarch64_atomic_loadclrqi,
gen_aarch64_atomic_loadclrhi,
gen_aarch64_atomic_loadclrsi,
gen_aarch64_atomic_loadclrdi
};
aarch64_atomic_load_op_fn gen;
int idx = 0;
switch (mode)
{
case E_QImode: idx = 0; break;
case E_HImode: idx = 1; break;
case E_SImode: idx = 2; break;
case E_DImode: idx = 3; break;
default:
gcc_unreachable ();
}
switch (code)
{
case AARCH64_LDOP_PLUS: gen = plus[idx]; break;
case AARCH64_LDOP_XOR: gen = eor[idx]; break;
case AARCH64_LDOP_OR: gen = ior[idx]; break;
case AARCH64_LDOP_BIC: gen = bic[idx]; break;
default:
gcc_unreachable ();
}
emit_insn (gen (dst, mem, src, model));
}
void
aarch64_gen_atomic_ldop (enum rtx_code code, rtx out_data, rtx out_result,
rtx mem, rtx value, rtx model_rtx)
{
machine_mode mode = GET_MODE (mem);
machine_mode wmode = (mode == DImode ? DImode : SImode);
const bool short_mode = (mode < SImode);
aarch64_atomic_load_op_code ldop_code;
rtx src;
rtx x;
if (out_data)
out_data = gen_lowpart (mode, out_data);
if (out_result)
out_result = gen_lowpart (mode, out_result);
if (!register_operand (value, mode)
|| code == AND || code == MINUS)
{
src = out_result ? out_result : out_data;
emit_move_insn (src, gen_lowpart (mode, value));
}
else
src = value;
gcc_assert (register_operand (src, mode));
switch (code)
{
case SET:
aarch64_emit_atomic_swap (mode, out_data, src, mem, model_rtx);
return;
case MINUS:
{
rtx neg_src;
if (short_mode)
src = gen_lowpart (wmode, src);
neg_src = gen_rtx_NEG (wmode, src);
emit_insn (gen_rtx_SET (src, neg_src));
if (short_mode)
src = gen_lowpart (mode, src);
}
case PLUS:
ldop_code = AARCH64_LDOP_PLUS;
break;
case IOR:
ldop_code = AARCH64_LDOP_OR;
break;
case XOR:
ldop_code = AARCH64_LDOP_XOR;
break;
case AND:
{
rtx not_src;
if (short_mode)
src = gen_lowpart (wmode, src);
not_src = gen_rtx_NOT (wmode, src);
emit_insn (gen_rtx_SET (src, not_src));
if (short_mode)
src = gen_lowpart (mode, src);
}
ldop_code = AARCH64_LDOP_BIC;
break;
default:
gcc_unreachable ();
}
aarch64_emit_atomic_load_op (ldop_code, mode, out_data, src, mem, model_rtx);
if (!out_result)
return;
if (short_mode)
{
src = gen_lowpart (wmode, src);
out_data = gen_lowpart (wmode, out_data);
out_result = gen_lowpart (wmode, out_result);
}
x = NULL_RTX;
switch (code)
{
case MINUS:
case PLUS:
x = gen_rtx_PLUS (wmode, out_data, src);
break;
case IOR:
x = gen_rtx_IOR (wmode, out_data, src);
break;
case XOR:
x = gen_rtx_XOR (wmode, out_data, src);
break;
case AND:
aarch64_emit_bic (wmode, out_result, out_data, src, 0);
return;
default:
gcc_unreachable ();
}
emit_set_insn (out_result, x);
return;
}
void
aarch64_split_atomic_op (enum rtx_code code, rtx old_out, rtx new_out, rtx mem,
rtx value, rtx model_rtx, rtx cond)
{
machine_mode mode = GET_MODE (mem);
machine_mode wmode = (mode == DImode ? DImode : SImode);
const enum memmodel model = memmodel_from_int (INTVAL (model_rtx));
const bool is_sync = is_mm_sync (model);
rtx_code_label *label;
rtx x;
label = gen_label_rtx ();
emit_label (label);
if (new_out)
new_out = gen_lowpart (wmode, new_out);
if (old_out)
old_out = gen_lowpart (wmode, old_out);
else
old_out = new_out;
value = simplify_gen_subreg (wmode, value, mode, 0);
if (is_sync)
aarch64_emit_load_exclusive (mode, old_out, mem,
GEN_INT (MEMMODEL_RELAXED));
else
aarch64_emit_load_exclusive (mode, old_out, mem, model_rtx);
switch (code)
{
case SET:
new_out = value;
break;
case NOT:
x = gen_rtx_AND (wmode, old_out, value);
emit_insn (gen_rtx_SET (new_out, x));
x = gen_rtx_NOT (wmode, new_out);
emit_insn (gen_rtx_SET (new_out, x));
break;
case MINUS:
if (CONST_INT_P (value))
{
value = GEN_INT (-INTVAL (value));
code = PLUS;
}
default:
x = gen_rtx_fmt_ee (code, wmode, old_out, value);
emit_insn (gen_rtx_SET (new_out, x));
break;
}
aarch64_emit_store_exclusive (mode, cond, mem,
gen_lowpart (mode, new_out), model_rtx);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (Pmode, label), pc_rtx);
aarch64_emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
if (is_sync)
aarch64_emit_post_barrier (model);
}
static void
aarch64_init_libfuncs (void)
{
set_conv_libfunc (trunc_optab, HFmode, SFmode, "__gnu_f2h_ieee");
set_conv_libfunc (sext_optab, SFmode, HFmode, "__gnu_h2f_ieee");
set_optab_libfunc (add_optab, HFmode, NULL);
set_optab_libfunc (sdiv_optab, HFmode, NULL);
set_optab_libfunc (smul_optab, HFmode, NULL);
set_optab_libfunc (neg_optab, HFmode, NULL);
set_optab_libfunc (sub_optab, HFmode, NULL);
set_optab_libfunc (eq_optab, HFmode, NULL);
set_optab_libfunc (ne_optab, HFmode, NULL);
set_optab_libfunc (lt_optab, HFmode, NULL);
set_optab_libfunc (le_optab, HFmode, NULL);
set_optab_libfunc (ge_optab, HFmode, NULL);
set_optab_libfunc (gt_optab, HFmode, NULL);
set_optab_libfunc (unord_optab, HFmode, NULL);
}
static machine_mode
aarch64_c_mode_for_suffix (char suffix)
{
if (suffix == 'q')
return TFmode;
return VOIDmode;
}
bool
aarch64_float_const_representable_p (rtx x)
{
int point_pos = 2 * HOST_BITS_PER_WIDE_INT - 1;
int exponent;
unsigned HOST_WIDE_INT mantissa, mask;
REAL_VALUE_TYPE r, m;
bool fail;
if (!CONST_DOUBLE_P (x))
return false;
if (GET_MODE (x) == VOIDmode || GET_MODE (x) == HFmode)
return false;
r = *CONST_DOUBLE_REAL_VALUE (x);
if (REAL_VALUE_ISINF (r) || REAL_VALUE_ISNAN (r)
|| REAL_VALUE_MINUS_ZERO (r))
return false;
r = real_value_abs (&r);
exponent = REAL_EXP (&r);
real_ldexp (&m, &r, point_pos - exponent);
wide_int w = real_to_integer (&m, &fail, HOST_BITS_PER_WIDE_INT * 2);
if (w.ulow () != 0)
return false;
mantissa = w.elt (1);
point_pos -= HOST_BITS_PER_WIDE_INT;
mask = ((unsigned HOST_WIDE_INT)1 << (point_pos - 5)) - 1;
if ((mantissa & mask) != 0)
return false;
mantissa >>= point_pos - 5;
if (mantissa == 0)
return false;
mantissa &= ~(1 << 4);
gcc_assert (mantissa <= 15);
exponent = 5 - exponent;
return (exponent >= 0 && exponent <= 7);
}
char*
aarch64_output_simd_mov_immediate (rtx const_vector, unsigned width,
enum simd_immediate_check which)
{
bool is_valid;
static char templ[40];
const char *mnemonic;
const char *shift_op;
unsigned int lane_count = 0;
char element_char;
struct simd_immediate_info info;
is_valid = aarch64_simd_valid_immediate (const_vector, &info, which);
gcc_assert (is_valid);
element_char = sizetochar (GET_MODE_BITSIZE (info.elt_mode));
lane_count = width / GET_MODE_BITSIZE (info.elt_mode);
if (GET_MODE_CLASS (info.elt_mode) == MODE_FLOAT)
{
gcc_assert (info.shift == 0 && info.insn == simd_immediate_info::MOV);
if (aarch64_float_const_zero_rtx_p (info.value))
info.value = GEN_INT (0);
else
{
const unsigned int buf_size = 20;
char float_buf[buf_size] = {'\0'};
real_to_decimal_for_mode (float_buf,
CONST_DOUBLE_REAL_VALUE (info.value),
buf_size, buf_size, 1, info.elt_mode);
if (lane_count == 1)
snprintf (templ, sizeof (templ), "fmov\t%%d0, %s", float_buf);
else
snprintf (templ, sizeof (templ), "fmov\t%%0.%d%c, %s",
lane_count, element_char, float_buf);
return templ;
}
}
gcc_assert (CONST_INT_P (info.value));
if (which == AARCH64_CHECK_MOV)
{
mnemonic = info.insn == simd_immediate_info::MVN ? "mvni" : "movi";
shift_op = info.modifier == simd_immediate_info::MSL ? "msl" : "lsl";
if (lane_count == 1)
snprintf (templ, sizeof (templ), "%s\t%%d0, " HOST_WIDE_INT_PRINT_HEX,
mnemonic, UINTVAL (info.value));
else if (info.shift)
snprintf (templ, sizeof (templ), "%s\t%%0.%d%c, "
HOST_WIDE_INT_PRINT_HEX ", %s %d", mnemonic, lane_count,
element_char, UINTVAL (info.value), shift_op, info.shift);
else
snprintf (templ, sizeof (templ), "%s\t%%0.%d%c, "
HOST_WIDE_INT_PRINT_HEX, mnemonic, lane_count,
element_char, UINTVAL (info.value));
}
else
{
mnemonic = info.insn == simd_immediate_info::MVN ? "bic" : "orr";
if (info.shift)
snprintf (templ, sizeof (templ), "%s\t%%0.%d%c, #"
HOST_WIDE_INT_PRINT_DEC ", %s #%d", mnemonic, lane_count,
element_char, UINTVAL (info.value), "lsl", info.shift);
else
snprintf (templ, sizeof (templ), "%s\t%%0.%d%c, #"
HOST_WIDE_INT_PRINT_DEC, mnemonic, lane_count,
element_char, UINTVAL (info.value));
}
return templ;
}
char*
aarch64_output_scalar_simd_mov_immediate (rtx immediate, scalar_int_mode mode)
{
if (CONST_DOUBLE_P (immediate) && GET_MODE_CLASS (mode) == MODE_INT)
{
unsigned HOST_WIDE_INT ival;
if (!aarch64_reinterpret_float_as_int (immediate, &ival))
gcc_unreachable ();
immediate = gen_int_mode (ival, mode);
}
machine_mode vmode;
int width = GET_MODE_BITSIZE (mode) == 64 ? 128 : 64;
vmode = aarch64_simd_container_mode (mode, width);
rtx v_op = aarch64_simd_gen_const_vector_dup (vmode, INTVAL (immediate));
return aarch64_output_simd_mov_immediate (v_op, width);
}
char *
aarch64_output_sve_mov_immediate (rtx const_vector)
{
static char templ[40];
struct simd_immediate_info info;
char element_char;
bool is_valid = aarch64_simd_valid_immediate (const_vector, &info);
gcc_assert (is_valid);
element_char = sizetochar (GET_MODE_BITSIZE (info.elt_mode));
if (info.step)
{
snprintf (templ, sizeof (templ), "index\t%%0.%c, #"
HOST_WIDE_INT_PRINT_DEC ", #" HOST_WIDE_INT_PRINT_DEC,
element_char, INTVAL (info.value), INTVAL (info.step));
return templ;
}
if (GET_MODE_CLASS (info.elt_mode) == MODE_FLOAT)
{
if (aarch64_float_const_zero_rtx_p (info.value))
info.value = GEN_INT (0);
else
{
const int buf_size = 20;
char float_buf[buf_size] = {};
real_to_decimal_for_mode (float_buf,
CONST_DOUBLE_REAL_VALUE (info.value),
buf_size, buf_size, 1, info.elt_mode);
snprintf (templ, sizeof (templ), "fmov\t%%0.%c, #%s",
element_char, float_buf);
return templ;
}
}
snprintf (templ, sizeof (templ), "mov\t%%0.%c, #" HOST_WIDE_INT_PRINT_DEC,
element_char, INTVAL (info.value));
return templ;
}
char *
aarch64_output_ptrue (machine_mode mode, char suffix)
{
unsigned int nunits;
static char buf[sizeof ("ptrue\t%0.N, vlNNNNN")];
if (GET_MODE_NUNITS (mode).is_constant (&nunits))
snprintf (buf, sizeof (buf), "ptrue\t%%0.%c, vl%d", suffix, nunits);
else
snprintf (buf, sizeof (buf), "ptrue\t%%0.%c, all", suffix);
return buf;
}
void
aarch64_split_combinev16qi (rtx operands[3])
{
unsigned int dest = REGNO (operands[0]);
unsigned int src1 = REGNO (operands[1]);
unsigned int src2 = REGNO (operands[2]);
machine_mode halfmode = GET_MODE (operands[1]);
unsigned int halfregs = REG_NREGS (operands[1]);
rtx destlo, desthi;
gcc_assert (halfmode == V16QImode);
if (src1 == dest && src2 == dest + halfregs)
{
emit_note (NOTE_INSN_DELETED);
return;
}
destlo = gen_rtx_REG_offset (operands[0], halfmode, dest, 0);
desthi = gen_rtx_REG_offset (operands[0], halfmode, dest + halfregs,
GET_MODE_SIZE (halfmode));
if (reg_overlap_mentioned_p (operands[2], destlo)
&& reg_overlap_mentioned_p (operands[1], desthi))
{
emit_insn (gen_xorv16qi3 (operands[1], operands[1], operands[2]));
emit_insn (gen_xorv16qi3 (operands[2], operands[1], operands[2]));
emit_insn (gen_xorv16qi3 (operands[1], operands[1], operands[2]));
}
else if (!reg_overlap_mentioned_p (operands[2], destlo))
{
if (src1 != dest)
emit_move_insn (destlo, operands[1]);
if (src2 != dest + halfregs)
emit_move_insn (desthi, operands[2]);
}
else
{
if (src2 != dest + halfregs)
emit_move_insn (desthi, operands[2]);
if (src1 != dest)
emit_move_insn (destlo, operands[1]);
}
}
struct expand_vec_perm_d
{
rtx target, op0, op1;
vec_perm_indices perm;
machine_mode vmode;
unsigned int vec_flags;
bool one_vector_p;
bool testing_p;
};
static void
aarch64_expand_vec_perm_1 (rtx target, rtx op0, rtx op1, rtx sel)
{
machine_mode vmode = GET_MODE (target);
bool one_vector_p = rtx_equal_p (op0, op1);
gcc_checking_assert (vmode == V8QImode || vmode == V16QImode);
gcc_checking_assert (GET_MODE (op0) == vmode);
gcc_checking_assert (GET_MODE (op1) == vmode);
gcc_checking_assert (GET_MODE (sel) == vmode);
gcc_checking_assert (TARGET_SIMD);
if (one_vector_p)
{
if (vmode == V8QImode)
{
rtx pair = gen_reg_rtx (V16QImode);
emit_insn (gen_aarch64_combinev8qi (pair, op0, op0));
emit_insn (gen_aarch64_tbl1v8qi (target, pair, sel));
}
else
{
emit_insn (gen_aarch64_tbl1v16qi (target, op0, sel));
}
}
else
{
rtx pair;
if (vmode == V8QImode)
{
pair = gen_reg_rtx (V16QImode);
emit_insn (gen_aarch64_combinev8qi (pair, op0, op1));
emit_insn (gen_aarch64_tbl1v8qi (target, pair, sel));
}
else
{
pair = gen_reg_rtx (OImode);
emit_insn (gen_aarch64_combinev16qi (pair, op0, op1));
emit_insn (gen_aarch64_tbl2v16qi (target, pair, sel));
}
}
}
void
aarch64_expand_vec_perm (rtx target, rtx op0, rtx op1, rtx sel,
unsigned int nelt)
{
machine_mode vmode = GET_MODE (target);
bool one_vector_p = rtx_equal_p (op0, op1);
rtx mask;
mask = aarch64_simd_gen_const_vector_dup (vmode,
one_vector_p ? nelt - 1 : 2 * nelt - 1);
sel = expand_simple_binop (vmode, AND, sel, mask, NULL, 0, OPTAB_LIB_WIDEN);
if (BYTES_BIG_ENDIAN)
{
if (!one_vector_p)
mask = aarch64_simd_gen_const_vector_dup (vmode, nelt - 1);
sel = expand_simple_binop (vmode, XOR, sel, mask,
NULL, 0, OPTAB_LIB_WIDEN);
}
aarch64_expand_vec_perm_1 (target, op0, op1, sel);
}
static void
emit_unspec2 (rtx target, int code, rtx op0, rtx op1)
{
emit_insn (gen_rtx_SET (target,
gen_rtx_UNSPEC (GET_MODE (target),
gen_rtvec (2, op0, op1), code)));
}
void
aarch64_expand_sve_vec_perm (rtx target, rtx op0, rtx op1, rtx sel)
{
machine_mode data_mode = GET_MODE (target);
machine_mode sel_mode = GET_MODE (sel);
int nunits = GET_MODE_NUNITS (sel_mode).to_constant ();
rtx sel_reg = force_reg (sel_mode, sel);
if (GET_CODE (sel) == CONST_VECTOR
&& aarch64_const_vec_all_in_range_p (sel, 0, nunits - 1))
{
emit_unspec2 (target, UNSPEC_TBL, op0, sel_reg);
return;
}
if (rtx_equal_p (op0, op1))
{
rtx max_sel = aarch64_simd_gen_const_vector_dup (sel_mode, nunits - 1);
rtx sel_mod = expand_simple_binop (sel_mode, AND, sel_reg, max_sel,
NULL, 0, OPTAB_DIRECT);
emit_unspec2 (target, UNSPEC_TBL, op0, sel_mod);
return;
}
rtx res0 = gen_reg_rtx (data_mode);
rtx res1 = gen_reg_rtx (data_mode);
rtx neg_num_elems = aarch64_simd_gen_const_vector_dup (sel_mode, -nunits);
if (GET_CODE (sel) != CONST_VECTOR
|| !aarch64_const_vec_all_in_range_p (sel, 0, 2 * nunits - 1))
{
rtx max_sel = aarch64_simd_gen_const_vector_dup (sel_mode,
2 * nunits - 1);
sel_reg = expand_simple_binop (sel_mode, AND, sel_reg, max_sel,
NULL, 0, OPTAB_DIRECT);
}
emit_unspec2 (res0, UNSPEC_TBL, op0, sel_reg);
rtx sel_sub = expand_simple_binop (sel_mode, PLUS, sel_reg, neg_num_elems,
NULL, 0, OPTAB_DIRECT);
emit_unspec2 (res1, UNSPEC_TBL, op1, sel_sub);
if (GET_MODE_CLASS (data_mode) == MODE_VECTOR_INT)
emit_insn (gen_rtx_SET (target, gen_rtx_IOR (data_mode, res0, res1)));
else
emit_unspec2 (target, UNSPEC_IORF, res0, res1);
}
static bool
aarch64_evpc_trn (struct expand_vec_perm_d *d)
{
HOST_WIDE_INT odd;
poly_uint64 nelt = d->perm.length ();
rtx out, in0, in1, x;
machine_mode vmode = d->vmode;
if (GET_MODE_UNIT_SIZE (vmode) > 8)
return false;
if (!d->perm[0].is_constant (&odd)
|| (odd != 0 && odd != 1)
|| !d->perm.series_p (0, 2, odd, 2)
|| !d->perm.series_p (1, 2, nelt + odd, 2))
return false;
if (d->testing_p)
return true;
in0 = d->op0;
in1 = d->op1;
if (BYTES_BIG_ENDIAN && d->vec_flags == VEC_ADVSIMD)
{
x = in0, in0 = in1, in1 = x;
odd = !odd;
}
out = d->target;
emit_set_insn (out, gen_rtx_UNSPEC (vmode, gen_rtvec (2, in0, in1),
odd ? UNSPEC_TRN2 : UNSPEC_TRN1));
return true;
}
static bool
aarch64_evpc_uzp (struct expand_vec_perm_d *d)
{
HOST_WIDE_INT odd;
rtx out, in0, in1, x;
machine_mode vmode = d->vmode;
if (GET_MODE_UNIT_SIZE (vmode) > 8)
return false;
if (!d->perm[0].is_constant (&odd)
|| (odd != 0 && odd != 1)
|| !d->perm.series_p (0, 1, odd, 2))
return false;
if (d->testing_p)
return true;
in0 = d->op0;
in1 = d->op1;
if (BYTES_BIG_ENDIAN && d->vec_flags == VEC_ADVSIMD)
{
x = in0, in0 = in1, in1 = x;
odd = !odd;
}
out = d->target;
emit_set_insn (out, gen_rtx_UNSPEC (vmode, gen_rtvec (2, in0, in1),
odd ? UNSPEC_UZP2 : UNSPEC_UZP1));
return true;
}
static bool
aarch64_evpc_zip (struct expand_vec_perm_d *d)
{
unsigned int high;
poly_uint64 nelt = d->perm.length ();
rtx out, in0, in1, x;
machine_mode vmode = d->vmode;
if (GET_MODE_UNIT_SIZE (vmode) > 8)
return false;
poly_uint64 first = d->perm[0];
if ((maybe_ne (first, 0U) && maybe_ne (first * 2, nelt))
|| !d->perm.series_p (0, 2, first, 1)
|| !d->perm.series_p (1, 2, first + nelt, 1))
return false;
high = maybe_ne (first, 0U);
if (d->testing_p)
return true;
in0 = d->op0;
in1 = d->op1;
if (BYTES_BIG_ENDIAN && d->vec_flags == VEC_ADVSIMD)
{
x = in0, in0 = in1, in1 = x;
high = !high;
}
out = d->target;
emit_set_insn (out, gen_rtx_UNSPEC (vmode, gen_rtvec (2, in0, in1),
high ? UNSPEC_ZIP2 : UNSPEC_ZIP1));
return true;
}
static bool
aarch64_evpc_ext (struct expand_vec_perm_d *d)
{
HOST_WIDE_INT location;
rtx offset;
if (d->vec_flags == VEC_SVE_PRED
|| !d->perm[0].is_constant (&location)
|| !d->perm.series_p (0, 1, location, 1))
return false;
if (d->testing_p)
return true;
if (BYTES_BIG_ENDIAN && location != 0 && d->vec_flags == VEC_ADVSIMD)
{
std::swap (d->op0, d->op1);
location = d->perm.length ().to_constant () - location;
}
offset = GEN_INT (location);
emit_set_insn (d->target,
gen_rtx_UNSPEC (d->vmode,
gen_rtvec (3, d->op0, d->op1, offset),
UNSPEC_EXT));
return true;
}
static bool
aarch64_evpc_rev_local (struct expand_vec_perm_d *d)
{
HOST_WIDE_INT diff;
unsigned int i, size, unspec;
machine_mode pred_mode;
if (d->vec_flags == VEC_SVE_PRED
|| !d->one_vector_p
|| !d->perm[0].is_constant (&diff))
return false;
size = (diff + 1) * GET_MODE_UNIT_SIZE (d->vmode);
if (size == 8)
{
unspec = UNSPEC_REV64;
pred_mode = VNx2BImode;
}
else if (size == 4)
{
unspec = UNSPEC_REV32;
pred_mode = VNx4BImode;
}
else if (size == 2)
{
unspec = UNSPEC_REV16;
pred_mode = VNx8BImode;
}
else
return false;
unsigned int step = diff + 1;
for (i = 0; i < step; ++i)
if (!d->perm.series_p (i, step, diff - i, step))
return false;
if (d->testing_p)
return true;
rtx src = gen_rtx_UNSPEC (d->vmode, gen_rtvec (1, d->op0), unspec);
if (d->vec_flags == VEC_SVE_DATA)
{
rtx pred = force_reg (pred_mode, CONSTM1_RTX (pred_mode));
src = gen_rtx_UNSPEC (d->vmode, gen_rtvec (2, pred, src),
UNSPEC_MERGE_PTRUE);
}
emit_set_insn (d->target, src);
return true;
}
static bool
aarch64_evpc_rev_global (struct expand_vec_perm_d *d)
{
poly_uint64 nelt = d->perm.length ();
if (!d->one_vector_p || d->vec_flags != VEC_SVE_DATA)
return false;
if (!d->perm.series_p (0, 1, nelt - 1, -1))
return false;
if (d->testing_p)
return true;
rtx src = gen_rtx_UNSPEC (d->vmode, gen_rtvec (1, d->op0), UNSPEC_REV);
emit_set_insn (d->target, src);
return true;
}
static bool
aarch64_evpc_dup (struct expand_vec_perm_d *d)
{
rtx out = d->target;
rtx in0;
HOST_WIDE_INT elt;
machine_mode vmode = d->vmode;
rtx lane;
if (d->vec_flags == VEC_SVE_PRED
|| d->perm.encoding ().encoded_nelts () != 1
|| !d->perm[0].is_constant (&elt))
return false;
if (d->vec_flags == VEC_SVE_DATA && elt >= 64 * GET_MODE_UNIT_SIZE (vmode))
return false;
if (d->testing_p)
return true;
in0 = d->op0;
lane = GEN_INT (elt); 
rtx parallel = gen_rtx_PARALLEL (vmode, gen_rtvec (1, lane));
rtx select = gen_rtx_VEC_SELECT (GET_MODE_INNER (vmode), in0, parallel);
emit_set_insn (out, gen_rtx_VEC_DUPLICATE (vmode, select));
return true;
}
static bool
aarch64_evpc_tbl (struct expand_vec_perm_d *d)
{
rtx rperm[MAX_COMPILE_TIME_VEC_BYTES], sel;
machine_mode vmode = d->vmode;
unsigned int encoded_nelts = d->perm.encoding ().encoded_nelts ();
for (unsigned int i = 0; i < encoded_nelts; ++i)
if (!d->perm[i].is_constant ())
return false;
if (d->testing_p)
return true;
if (vmode != V8QImode && vmode != V16QImode)
return false;
unsigned int nelt = d->perm.length ().to_constant ();
for (unsigned int i = 0; i < nelt; ++i)
rperm[i] = GEN_INT (BYTES_BIG_ENDIAN
? d->perm[i].to_constant () ^ (nelt - 1)
: d->perm[i].to_constant ());
sel = gen_rtx_CONST_VECTOR (vmode, gen_rtvec_v (nelt, rperm));
sel = force_reg (vmode, sel);
aarch64_expand_vec_perm_1 (d->target, d->op0, d->op1, sel);
return true;
}
static bool
aarch64_evpc_sve_tbl (struct expand_vec_perm_d *d)
{
unsigned HOST_WIDE_INT nelt;
if (!d->one_vector_p && !d->perm.length ().is_constant (&nelt))
return false;
if (d->testing_p)
return true;
machine_mode sel_mode = mode_for_int_vector (d->vmode).require ();
rtx sel = vec_perm_indices_to_rtx (sel_mode, d->perm);
aarch64_expand_sve_vec_perm (d->target, d->op0, d->op1, sel);
return true;
}
static bool
aarch64_expand_vec_perm_const_1 (struct expand_vec_perm_d *d)
{
poly_int64 nelt = d->perm.length ();
if (known_ge (d->perm[0], nelt))
{
d->perm.rotate_inputs (1);
std::swap (d->op0, d->op1);
}
if ((d->vec_flags == VEC_ADVSIMD
|| d->vec_flags == VEC_SVE_DATA
|| d->vec_flags == VEC_SVE_PRED)
&& known_gt (nelt, 1))
{
if (aarch64_evpc_rev_local (d))
return true;
else if (aarch64_evpc_rev_global (d))
return true;
else if (aarch64_evpc_ext (d))
return true;
else if (aarch64_evpc_dup (d))
return true;
else if (aarch64_evpc_zip (d))
return true;
else if (aarch64_evpc_uzp (d))
return true;
else if (aarch64_evpc_trn (d))
return true;
if (d->vec_flags == VEC_SVE_DATA)
return aarch64_evpc_sve_tbl (d);
else if (d->vec_flags == VEC_SVE_DATA)
return aarch64_evpc_tbl (d);
}
return false;
}
static bool
aarch64_vectorize_vec_perm_const (machine_mode vmode, rtx target, rtx op0,
rtx op1, const vec_perm_indices &sel)
{
struct expand_vec_perm_d d;
if (op0 && rtx_equal_p (op0, op1))
d.one_vector_p = true;
else if (sel.all_from_input_p (0))
{
d.one_vector_p = true;
op1 = op0;
}
else if (sel.all_from_input_p (1))
{
d.one_vector_p = true;
op0 = op1;
}
else
d.one_vector_p = false;
d.perm.new_vector (sel.encoding (), d.one_vector_p ? 1 : 2,
sel.nelts_per_input ());
d.vmode = vmode;
d.vec_flags = aarch64_classify_vector_mode (d.vmode);
d.target = target;
d.op0 = op0;
d.op1 = op1;
d.testing_p = !target;
if (!d.testing_p)
return aarch64_expand_vec_perm_const_1 (&d);
rtx_insn *last = get_last_insn ();
bool ret = aarch64_expand_vec_perm_const_1 (&d);
gcc_assert (last == get_last_insn ());
return ret;
}
rtx
aarch64_reverse_mask (machine_mode mode, unsigned int nunits)
{
rtx mask;
rtvec v = rtvec_alloc (16);
unsigned int i, j;
unsigned int usize = GET_MODE_UNIT_SIZE (mode);
gcc_assert (BYTES_BIG_ENDIAN);
gcc_assert (AARCH64_VALID_SIMD_QREG_MODE (mode));
for (i = 0; i < nunits; i++)
for (j = 0; j < usize; j++)
RTVEC_ELT (v, i * usize + j) = GEN_INT ((i + 1) * usize - 1 - j);
mask = gen_rtx_CONST_VECTOR (V16QImode, v);
return force_reg (V16QImode, mask);
}
static bool
aarch64_sve_cmp_operand_p (rtx_code op_code, rtx x)
{
if (register_operand (x, VOIDmode))
return true;
switch (op_code)
{
case LTU:
case LEU:
case GEU:
case GTU:
return aarch64_sve_cmp_immediate_p (x, false);
case LT:
case LE:
case GE:
case GT:
case NE:
case EQ:
return aarch64_sve_cmp_immediate_p (x, true);
default:
gcc_unreachable ();
}
}
static unsigned int
aarch64_unspec_cond_code (rtx_code code)
{
switch (code)
{
case NE:
return UNSPEC_COND_NE;
case EQ:
return UNSPEC_COND_EQ;
case LT:
return UNSPEC_COND_LT;
case GT:
return UNSPEC_COND_GT;
case LE:
return UNSPEC_COND_LE;
case GE:
return UNSPEC_COND_GE;
case LTU:
return UNSPEC_COND_LO;
case GTU:
return UNSPEC_COND_HI;
case LEU:
return UNSPEC_COND_LS;
case GEU:
return UNSPEC_COND_HS;
case UNORDERED:
return UNSPEC_COND_UO;
default:
gcc_unreachable ();
}
}
static rtx
aarch64_gen_unspec_cond (rtx_code code, machine_mode pred_mode,
rtx pred, rtx op0, rtx op1)
{
rtvec vec = gen_rtvec (3, pred, op0, op1);
return gen_rtx_UNSPEC (pred_mode, vec, aarch64_unspec_cond_code (code));
}
void
aarch64_expand_sve_vec_cmp_int (rtx target, rtx_code code, rtx op0, rtx op1)
{
machine_mode pred_mode = GET_MODE (target);
machine_mode data_mode = GET_MODE (op0);
if (!aarch64_sve_cmp_operand_p (code, op1))
op1 = force_reg (data_mode, op1);
rtx ptrue = force_reg (pred_mode, CONSTM1_RTX (pred_mode));
rtx unspec = aarch64_gen_unspec_cond (code, pred_mode, ptrue, op0, op1);
emit_insn (gen_set_clobber_cc (target, unspec));
}
static void
aarch64_emit_unspec_cond (rtx target, rtx_code code, machine_mode pred_mode,
rtx pred, rtx op0, rtx op1)
{
rtx unspec = aarch64_gen_unspec_cond (code, pred_mode, pred, op0, op1);
emit_set_insn (target, unspec);
}
static void
aarch64_emit_unspec_cond_or (rtx target, rtx_code code1, rtx_code code2,
machine_mode pred_mode, rtx ptrue,
rtx op0, rtx op1)
{
rtx tmp1 = gen_reg_rtx (pred_mode);
aarch64_emit_unspec_cond (tmp1, code1, pred_mode, ptrue, op0, op1);
rtx tmp2 = gen_reg_rtx (pred_mode);
aarch64_emit_unspec_cond (tmp2, code2, pred_mode, ptrue, op0, op1);
emit_set_insn (target, gen_rtx_AND (pred_mode,
gen_rtx_IOR (pred_mode, tmp1, tmp2),
ptrue));
}
static void
aarch64_emit_inverted_unspec_cond (rtx target, rtx_code code,
machine_mode pred_mode, rtx ptrue, rtx pred,
rtx op0, rtx op1, bool can_invert_p)
{
if (can_invert_p)
aarch64_emit_unspec_cond (target, code, pred_mode, pred, op0, op1);
else
{
rtx tmp = gen_reg_rtx (pred_mode);
aarch64_emit_unspec_cond (tmp, code, pred_mode, pred, op0, op1);
emit_set_insn (target, gen_rtx_AND (pred_mode,
gen_rtx_NOT (pred_mode, tmp),
ptrue));
}
}
bool
aarch64_expand_sve_vec_cmp_float (rtx target, rtx_code code,
rtx op0, rtx op1, bool can_invert_p)
{
machine_mode pred_mode = GET_MODE (target);
machine_mode data_mode = GET_MODE (op0);
rtx ptrue = force_reg (pred_mode, CONSTM1_RTX (pred_mode));
switch (code)
{
case UNORDERED:
op1 = force_reg (data_mode, op1);
aarch64_emit_unspec_cond (target, code, pred_mode, ptrue, op0, op1);
return false;
case LT:
case LE:
case GT:
case GE:
case EQ:
case NE:
aarch64_emit_unspec_cond (target, code, pred_mode, ptrue, op0, op1);
return false;
case ORDERED:
op1 = force_reg (data_mode, op1);
aarch64_emit_inverted_unspec_cond (target, UNORDERED,
pred_mode, ptrue, ptrue, op0, op1,
can_invert_p);
return can_invert_p;
case LTGT:
aarch64_emit_unspec_cond_or (target, LT, GT, pred_mode, ptrue, op0, op1);
return false;
case UNEQ:
if (!flag_trapping_math)
{
op1 = force_reg (data_mode, op1);
aarch64_emit_unspec_cond_or (target, UNORDERED, EQ,
pred_mode, ptrue, op0, op1);
return false;
}
case UNLT:
case UNLE:
case UNGT:
case UNGE:
{
rtx ordered = ptrue;
if (flag_trapping_math)
{
ordered = gen_reg_rtx (pred_mode);
op1 = force_reg (data_mode, op1);
aarch64_emit_inverted_unspec_cond (ordered, UNORDERED, pred_mode,
ptrue, ptrue, op0, op1, false);
}
if (code == UNEQ)
code = NE;
else
code = reverse_condition_maybe_unordered (code);
aarch64_emit_inverted_unspec_cond (target, code, pred_mode, ptrue,
ordered, op0, op1, can_invert_p);
return can_invert_p;
}
default:
gcc_unreachable ();
}
}
void
aarch64_expand_sve_vcond (machine_mode data_mode, machine_mode cmp_mode,
rtx *ops)
{
machine_mode pred_mode
= aarch64_get_mask_mode (GET_MODE_NUNITS (cmp_mode),
GET_MODE_SIZE (cmp_mode)).require ();
rtx pred = gen_reg_rtx (pred_mode);
if (FLOAT_MODE_P (cmp_mode))
{
if (aarch64_expand_sve_vec_cmp_float (pred, GET_CODE (ops[3]),
ops[4], ops[5], true))
std::swap (ops[1], ops[2]);
}
else
aarch64_expand_sve_vec_cmp_int (pred, GET_CODE (ops[3]), ops[4], ops[5]);
rtvec vec = gen_rtvec (3, pred, ops[1], ops[2]);
emit_set_insn (ops[0], gen_rtx_UNSPEC (data_mode, vec, UNSPEC_SEL));
}
static bool
aarch64_modes_tieable_p (machine_mode mode1, machine_mode mode2)
{
if (GET_MODE_CLASS (mode1) == GET_MODE_CLASS (mode2))
return true;
if (aarch64_vector_data_mode_p (mode1)
&& aarch64_vector_data_mode_p (mode2))
return true;
if (aarch64_vector_mode_supported_p (mode1)
|| aarch64_vector_mode_supported_p (mode2))
return true;
return false;
}
static rtx
aarch64_move_pointer (rtx pointer, poly_int64 amount)
{
rtx next = plus_constant (Pmode, XEXP (pointer, 0), amount);
return adjust_automodify_address (pointer, GET_MODE (pointer),
next, amount);
}
static rtx
aarch64_progress_pointer (rtx pointer)
{
return aarch64_move_pointer (pointer, GET_MODE_SIZE (GET_MODE (pointer)));
}
static void
aarch64_copy_one_block_and_progress_pointers (rtx *src, rtx *dst,
machine_mode mode)
{
rtx reg = gen_reg_rtx (mode);
*src = adjust_address (*src, mode, 0);
*dst = adjust_address (*dst, mode, 0);
emit_move_insn (reg, *src);
emit_move_insn (*dst, reg);
*src = aarch64_progress_pointer (*src);
*dst = aarch64_progress_pointer (*dst);
}
bool
aarch64_expand_movmem (rtx *operands)
{
unsigned int n;
rtx dst = operands[0];
rtx src = operands[1];
rtx base;
bool speed_p = !optimize_function_for_size_p (cfun);
unsigned int max_instructions = (speed_p ? 15 : AARCH64_CALL_RATIO) / 2;
if (!CONST_INT_P (operands[2]))
return false;
n = UINTVAL (operands[2]);
if (((n / 16) + (n % 16 ? 2 : 0)) > max_instructions)
return false;
base = copy_to_mode_reg (Pmode, XEXP (dst, 0));
dst = adjust_automodify_address (dst, VOIDmode, base, 0);
base = copy_to_mode_reg (Pmode, XEXP (src, 0));
src = adjust_automodify_address (src, VOIDmode, base, 0);
if (n < 4)
{
if (n >= 2)
{
aarch64_copy_one_block_and_progress_pointers (&src, &dst, HImode);
n -= 2;
}
if (n == 1)
aarch64_copy_one_block_and_progress_pointers (&src, &dst, QImode);
return true;
}
if (n < 8)
{
aarch64_copy_one_block_and_progress_pointers (&src, &dst, SImode);
n -= 4;
if (n > 0)
{
int move = n - 4;
src = aarch64_move_pointer (src, move);
dst = aarch64_move_pointer (dst, move);
aarch64_copy_one_block_and_progress_pointers (&src, &dst, SImode);
}
return true;
}
while (n >= 8)
{
if (n / 16)
{
aarch64_copy_one_block_and_progress_pointers (&src, &dst, TImode);
n -= 16;
}
else
{
aarch64_copy_one_block_and_progress_pointers (&src, &dst, DImode);
n -= 8;
}
}
if (n == 0)
return true;
else if (n == 1)
aarch64_copy_one_block_and_progress_pointers (&src, &dst, QImode);
else if (n == 2)
aarch64_copy_one_block_and_progress_pointers (&src, &dst, HImode);
else if (n == 4)
aarch64_copy_one_block_and_progress_pointers (&src, &dst, SImode);
else
{
if (n == 3)
{
src = aarch64_move_pointer (src, -1);
dst = aarch64_move_pointer (dst, -1);
aarch64_copy_one_block_and_progress_pointers (&src, &dst, SImode);
}
else
{
int move = n - 8;
src = aarch64_move_pointer (src, move);
dst = aarch64_move_pointer (dst, move);
aarch64_copy_one_block_and_progress_pointers (&src, &dst, DImode);
}
}
return true;
}
bool
aarch64_split_dimode_const_store (rtx dst, rtx src)
{
rtx lo = gen_lowpart (SImode, src);
rtx hi = gen_highpart_mode (SImode, DImode, src);
bool size_p = optimize_function_for_size_p (cfun);
if (!rtx_equal_p (lo, hi))
return false;
unsigned int orig_cost
= aarch64_internal_mov_immediate (NULL_RTX, src, false, DImode);
unsigned int lo_cost
= aarch64_internal_mov_immediate (NULL_RTX, lo, false, SImode);
if (size_p && orig_cost <= lo_cost)
return false;
if (!size_p
&& (orig_cost <= lo_cost + 1))
return false;
rtx mem_lo = adjust_address (dst, SImode, 0);
if (!aarch64_mem_pair_operand (mem_lo, SImode))
return false;
rtx tmp_reg = gen_reg_rtx (SImode);
aarch64_expand_mov_immediate (tmp_reg, lo);
rtx mem_hi = aarch64_move_pointer (mem_lo, GET_MODE_SIZE (SImode));
emit_move_insn (mem_lo, tmp_reg);
emit_move_insn (mem_hi, tmp_reg);
return true;
}
static unsigned HOST_WIDE_INT
aarch64_asan_shadow_offset (void)
{
return (HOST_WIDE_INT_1 << 36);
}
static rtx
aarch64_gen_ccmp_first (rtx_insn **prep_seq, rtx_insn **gen_seq,
int code, tree treeop0, tree treeop1)
{
machine_mode op_mode, cmp_mode, cc_mode = CCmode;
rtx op0, op1;
int unsignedp = TYPE_UNSIGNED (TREE_TYPE (treeop0));
insn_code icode;
struct expand_operand ops[4];
start_sequence ();
expand_operands (treeop0, treeop1, NULL_RTX, &op0, &op1, EXPAND_NORMAL);
op_mode = GET_MODE (op0);
if (op_mode == VOIDmode)
op_mode = GET_MODE (op1);
switch (op_mode)
{
case E_QImode:
case E_HImode:
case E_SImode:
cmp_mode = SImode;
icode = CODE_FOR_cmpsi;
break;
case E_DImode:
cmp_mode = DImode;
icode = CODE_FOR_cmpdi;
break;
case E_SFmode:
cmp_mode = SFmode;
cc_mode = aarch64_select_cc_mode ((rtx_code) code, op0, op1);
icode = cc_mode == CCFPEmode ? CODE_FOR_fcmpesf : CODE_FOR_fcmpsf;
break;
case E_DFmode:
cmp_mode = DFmode;
cc_mode = aarch64_select_cc_mode ((rtx_code) code, op0, op1);
icode = cc_mode == CCFPEmode ? CODE_FOR_fcmpedf : CODE_FOR_fcmpdf;
break;
default:
end_sequence ();
return NULL_RTX;
}
op0 = prepare_operand (icode, op0, 0, op_mode, cmp_mode, unsignedp);
op1 = prepare_operand (icode, op1, 1, op_mode, cmp_mode, unsignedp);
if (!op0 || !op1)
{
end_sequence ();
return NULL_RTX;
}
*prep_seq = get_insns ();
end_sequence ();
create_fixed_operand (&ops[0], op0);
create_fixed_operand (&ops[1], op1);
start_sequence ();
if (!maybe_expand_insn (icode, 2, ops))
{
end_sequence ();
return NULL_RTX;
}
*gen_seq = get_insns ();
end_sequence ();
return gen_rtx_fmt_ee ((rtx_code) code, cc_mode,
gen_rtx_REG (cc_mode, CC_REGNUM), const0_rtx);
}
static rtx
aarch64_gen_ccmp_next (rtx_insn **prep_seq, rtx_insn **gen_seq, rtx prev,
int cmp_code, tree treeop0, tree treeop1, int bit_code)
{
rtx op0, op1, target;
machine_mode op_mode, cmp_mode, cc_mode = CCmode;
int unsignedp = TYPE_UNSIGNED (TREE_TYPE (treeop0));
insn_code icode;
struct expand_operand ops[6];
int aarch64_cond;
push_to_sequence (*prep_seq);
expand_operands (treeop0, treeop1, NULL_RTX, &op0, &op1, EXPAND_NORMAL);
op_mode = GET_MODE (op0);
if (op_mode == VOIDmode)
op_mode = GET_MODE (op1);
switch (op_mode)
{
case E_QImode:
case E_HImode:
case E_SImode:
cmp_mode = SImode;
icode = CODE_FOR_ccmpsi;
break;
case E_DImode:
cmp_mode = DImode;
icode = CODE_FOR_ccmpdi;
break;
case E_SFmode:
cmp_mode = SFmode;
cc_mode = aarch64_select_cc_mode ((rtx_code) cmp_code, op0, op1);
icode = cc_mode == CCFPEmode ? CODE_FOR_fccmpesf : CODE_FOR_fccmpsf;
break;
case E_DFmode:
cmp_mode = DFmode;
cc_mode = aarch64_select_cc_mode ((rtx_code) cmp_code, op0, op1);
icode = cc_mode == CCFPEmode ? CODE_FOR_fccmpedf : CODE_FOR_fccmpdf;
break;
default:
end_sequence ();
return NULL_RTX;
}
op0 = prepare_operand (icode, op0, 2, op_mode, cmp_mode, unsignedp);
op1 = prepare_operand (icode, op1, 3, op_mode, cmp_mode, unsignedp);
if (!op0 || !op1)
{
end_sequence ();
return NULL_RTX;
}
*prep_seq = get_insns ();
end_sequence ();
target = gen_rtx_REG (cc_mode, CC_REGNUM);
aarch64_cond = aarch64_get_condition_code_1 (cc_mode, (rtx_code) cmp_code);
if (bit_code != AND)
{
prev = gen_rtx_fmt_ee (REVERSE_CONDITION (GET_CODE (prev),
GET_MODE (XEXP (prev, 0))),
VOIDmode, XEXP (prev, 0), const0_rtx);
aarch64_cond = AARCH64_INVERSE_CONDITION_CODE (aarch64_cond);
}
create_fixed_operand (&ops[0], XEXP (prev, 0));
create_fixed_operand (&ops[1], target);
create_fixed_operand (&ops[2], op0);
create_fixed_operand (&ops[3], op1);
create_fixed_operand (&ops[4], prev);
create_fixed_operand (&ops[5], GEN_INT (aarch64_cond));
push_to_sequence (*gen_seq);
if (!maybe_expand_insn (icode, 6, ops))
{
end_sequence ();
return NULL_RTX;
}
*gen_seq = get_insns ();
end_sequence ();
return gen_rtx_fmt_ee ((rtx_code) cmp_code, VOIDmode, target, const0_rtx);
}
#undef TARGET_GEN_CCMP_FIRST
#define TARGET_GEN_CCMP_FIRST aarch64_gen_ccmp_first
#undef TARGET_GEN_CCMP_NEXT
#define TARGET_GEN_CCMP_NEXT aarch64_gen_ccmp_next
static bool
aarch64_macro_fusion_p (void)
{
return aarch64_tune_params.fusible_ops != AARCH64_FUSE_NOTHING;
}
static bool
aarch_macro_fusion_pair_p (rtx_insn *prev, rtx_insn *curr)
{
rtx set_dest;
rtx prev_set = single_set (prev);
rtx curr_set = single_set (curr);
bool simple_sets_p = prev_set && curr_set && !any_condjump_p (curr);
if (!aarch64_macro_fusion_p ())
return false;
if (simple_sets_p && aarch64_fusion_enabled_p (AARCH64_FUSE_MOV_MOVK))
{
set_dest = SET_DEST (curr_set);
if (GET_CODE (set_dest) == ZERO_EXTRACT
&& CONST_INT_P (SET_SRC (curr_set))
&& CONST_INT_P (SET_SRC (prev_set))
&& CONST_INT_P (XEXP (set_dest, 2))
&& INTVAL (XEXP (set_dest, 2)) == 16
&& REG_P (XEXP (set_dest, 0))
&& REG_P (SET_DEST (prev_set))
&& REGNO (XEXP (set_dest, 0)) == REGNO (SET_DEST (prev_set)))
{
return true;
}
}
if (simple_sets_p && aarch64_fusion_enabled_p (AARCH64_FUSE_ADRP_ADD))
{
if (satisfies_constraint_Ush (SET_SRC (prev_set))
&& REG_P (SET_DEST (prev_set)) && REG_P (SET_DEST (curr_set)))
{
if (GET_CODE (SET_SRC (curr_set)) == LO_SUM
&& REG_P (XEXP (SET_SRC (curr_set), 0))
&& REGNO (XEXP (SET_SRC (curr_set), 0))
== REGNO (SET_DEST (prev_set))
&& rtx_equal_p (XEXP (SET_SRC (prev_set), 0),
XEXP (SET_SRC (curr_set), 1)))
return true;
}
}
if (simple_sets_p && aarch64_fusion_enabled_p (AARCH64_FUSE_MOVK_MOVK))
{
if (GET_CODE (SET_DEST (prev_set)) == ZERO_EXTRACT
&& GET_CODE (SET_DEST (curr_set)) == ZERO_EXTRACT
&& REG_P (XEXP (SET_DEST (prev_set), 0))
&& REG_P (XEXP (SET_DEST (curr_set), 0))
&& REGNO (XEXP (SET_DEST (prev_set), 0))
== REGNO (XEXP (SET_DEST (curr_set), 0))
&& CONST_INT_P (XEXP (SET_DEST (prev_set), 2))
&& CONST_INT_P (XEXP (SET_DEST (curr_set), 2))
&& INTVAL (XEXP (SET_DEST (prev_set), 2)) == 32
&& INTVAL (XEXP (SET_DEST (curr_set), 2)) == 48
&& CONST_INT_P (SET_SRC (prev_set))
&& CONST_INT_P (SET_SRC (curr_set)))
return true;
}
if (simple_sets_p && aarch64_fusion_enabled_p (AARCH64_FUSE_ADRP_LDR))
{
if (satisfies_constraint_Ush (SET_SRC (prev_set))
&& REG_P (SET_DEST (prev_set)) && REG_P (SET_DEST (curr_set)))
{
rtx curr_src = SET_SRC (curr_set);
if (GET_CODE (curr_src) == ZERO_EXTEND)
curr_src = XEXP (curr_src, 0);
if (MEM_P (curr_src) && GET_CODE (XEXP (curr_src, 0)) == LO_SUM
&& REG_P (XEXP (XEXP (curr_src, 0), 0))
&& REGNO (XEXP (XEXP (curr_src, 0), 0))
== REGNO (SET_DEST (prev_set))
&& rtx_equal_p (XEXP (XEXP (curr_src, 0), 1),
XEXP (SET_SRC (prev_set), 0)))
return true;
}
}
if (aarch64_fusion_enabled_p (AARCH64_FUSE_AES_AESMC)
&& aarch_crypto_can_dual_issue (prev, curr))
return true;
if (aarch64_fusion_enabled_p (AARCH64_FUSE_CMP_BRANCH)
&& any_condjump_p (curr))
{
unsigned int condreg1, condreg2;
rtx cc_reg_1;
aarch64_fixed_condition_code_regs (&condreg1, &condreg2);
cc_reg_1 = gen_rtx_REG (CCmode, condreg1);
if (reg_referenced_p (cc_reg_1, PATTERN (curr))
&& prev
&& modified_in_p (cc_reg_1, prev))
{
enum attr_type prev_type = get_attr_type (prev);
if (prev_type == TYPE_ALUS_SREG
|| prev_type == TYPE_ALUS_IMM
|| prev_type == TYPE_LOGICS_REG
|| prev_type == TYPE_LOGICS_IMM)
return true;
}
}
if (prev_set
&& curr_set
&& aarch64_fusion_enabled_p (AARCH64_FUSE_ALU_BRANCH)
&& any_condjump_p (curr))
{
if (SET_DEST (curr_set) == (pc_rtx)
&& GET_CODE (SET_SRC (curr_set)) == IF_THEN_ELSE
&& REG_P (XEXP (XEXP (SET_SRC (curr_set), 0), 0))
&& REG_P (SET_DEST (prev_set))
&& REGNO (SET_DEST (prev_set))
== REGNO (XEXP (XEXP (SET_SRC (curr_set), 0), 0)))
{
switch (get_attr_type (prev))
{
case TYPE_ALU_IMM:
case TYPE_ALU_SREG:
case TYPE_ADC_REG:
case TYPE_ADC_IMM:
case TYPE_ADCS_REG:
case TYPE_ADCS_IMM:
case TYPE_LOGIC_REG:
case TYPE_LOGIC_IMM:
case TYPE_CSEL:
case TYPE_ADR:
case TYPE_MOV_IMM:
case TYPE_SHIFT_REG:
case TYPE_SHIFT_IMM:
case TYPE_BFM:
case TYPE_RBIT:
case TYPE_REV:
case TYPE_EXTEND:
return true;
default:;
}
}
}
return false;
}
bool
aarch64_fusion_enabled_p (enum aarch64_fusion_pairs op)
{
return (aarch64_tune_params.fusible_ops & op) != 0;
}
bool
extract_base_offset_in_addr (rtx mem, rtx *base, rtx *offset)
{
rtx addr;
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
if (REG_P (addr))
{
*base = addr;
*offset = const0_rtx;
return true;
}
if (GET_CODE (addr) == PLUS
&& REG_P (XEXP (addr, 0)) && CONST_INT_P (XEXP (addr, 1)))
{
*base = XEXP (addr, 0);
*offset = XEXP (addr, 1);
return true;
}
*base = NULL_RTX;
*offset = NULL_RTX;
return false;
}
enum sched_fusion_type
{
SCHED_FUSION_NONE = 0,
SCHED_FUSION_LD_SIGN_EXTEND,
SCHED_FUSION_LD_ZERO_EXTEND,
SCHED_FUSION_LD,
SCHED_FUSION_ST,
SCHED_FUSION_NUM
};
static enum sched_fusion_type
fusion_load_store (rtx_insn *insn, rtx *base, rtx *offset)
{
rtx x, dest, src;
enum sched_fusion_type fusion = SCHED_FUSION_LD;
gcc_assert (INSN_P (insn));
x = PATTERN (insn);
if (GET_CODE (x) != SET)
return SCHED_FUSION_NONE;
src = SET_SRC (x);
dest = SET_DEST (x);
machine_mode dest_mode = GET_MODE (dest);
if (!aarch64_mode_valid_for_sched_fusion_p (dest_mode))
return SCHED_FUSION_NONE;
if (GET_CODE (src) == SIGN_EXTEND)
{
fusion = SCHED_FUSION_LD_SIGN_EXTEND;
src = XEXP (src, 0);
if (GET_CODE (src) != MEM || GET_MODE (src) != SImode)
return SCHED_FUSION_NONE;
}
else if (GET_CODE (src) == ZERO_EXTEND)
{
fusion = SCHED_FUSION_LD_ZERO_EXTEND;
src = XEXP (src, 0);
if (GET_CODE (src) != MEM || GET_MODE (src) != SImode)
return SCHED_FUSION_NONE;
}
if (GET_CODE (src) == MEM && REG_P (dest))
extract_base_offset_in_addr (src, base, offset);
else if (GET_CODE (dest) == MEM && (REG_P (src) || src == const0_rtx))
{
fusion = SCHED_FUSION_ST;
extract_base_offset_in_addr (dest, base, offset);
}
else
return SCHED_FUSION_NONE;
if (*base == NULL_RTX || *offset == NULL_RTX)
fusion = SCHED_FUSION_NONE;
return fusion;
}
static void
aarch64_sched_fusion_priority (rtx_insn *insn, int max_pri,
int *fusion_pri, int *pri)
{
int tmp, off_val;
rtx base, offset;
enum sched_fusion_type fusion;
gcc_assert (INSN_P (insn));
tmp = max_pri - 1;
fusion = fusion_load_store (insn, &base, &offset);
if (fusion == SCHED_FUSION_NONE)
{
*pri = tmp;
*fusion_pri = tmp;
return;
}
*fusion_pri = tmp - fusion * FIRST_PSEUDO_REGISTER - REGNO (base);
tmp /= 2;
off_val = (int)(INTVAL (offset));
if (off_val >= 0)
tmp -= (off_val & 0xfffff);
else
tmp += ((- off_val) & 0xfffff);
*pri = tmp;
return;
}
static int
aarch64_sched_adjust_priority (rtx_insn *insn, int priority)
{
rtx x = PATTERN (insn);
if (GET_CODE (x) == SET)
{
x = SET_SRC (x);
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_SHA1H)
return priority + 10;
}
return priority;
}
bool
aarch64_operands_ok_for_ldpstp (rtx *operands, bool load,
machine_mode mode)
{
HOST_WIDE_INT offval_1, offval_2, msize;
enum reg_class rclass_1, rclass_2;
rtx mem_1, mem_2, reg_1, reg_2, base_1, base_2, offset_1, offset_2;
if (load)
{
mem_1 = operands[1];
mem_2 = operands[3];
reg_1 = operands[0];
reg_2 = operands[2];
gcc_assert (REG_P (reg_1) && REG_P (reg_2));
if (REGNO (reg_1) == REGNO (reg_2))
return false;
}
else
{
mem_1 = operands[0];
mem_2 = operands[2];
reg_1 = operands[1];
reg_2 = operands[3];
}
if (MEM_VOLATILE_P (mem_1) || MEM_VOLATILE_P (mem_2))
return false;
if (mode == SImode
&& (aarch64_tune_params.extra_tuning_flags
& AARCH64_EXTRA_TUNE_SLOW_UNALIGNED_LDPW)
&& !optimize_size
&& MEM_ALIGN (mem_1) < 8 * BITS_PER_UNIT)
return false;
extract_base_offset_in_addr (mem_1, &base_1, &offset_1);
if (base_1 == NULL_RTX || offset_1 == NULL_RTX)
return false;
extract_base_offset_in_addr (mem_2, &base_2, &offset_2);
if (base_2 == NULL_RTX || offset_2 == NULL_RTX)
return false;
if (!rtx_equal_p (base_1, base_2))
return false;
offval_1 = INTVAL (offset_1);
offval_2 = INTVAL (offset_2);
msize = GET_MODE_SIZE (mode).to_constant ();
if (offval_1 != (offval_2 + msize) && offval_2 != (offval_1 + msize))
return false;
if (load)
{
if (reg_mentioned_p (reg_1, mem_1))
return false;
if (offval_1 > offval_2 && reg_mentioned_p (reg_2, mem_2))
return false;
}
if (REG_P (reg_1) && FP_REGNUM_P (REGNO (reg_1)))
rclass_1 = FP_REGS;
else
rclass_1 = GENERAL_REGS;
if (REG_P (reg_2) && FP_REGNUM_P (REGNO (reg_2)))
rclass_2 = FP_REGS;
else
rclass_2 = GENERAL_REGS;
if (rclass_1 != rclass_2)
return false;
return true;
}
bool
aarch64_operands_adjust_ok_for_ldpstp (rtx *operands, bool load,
scalar_mode mode)
{
enum reg_class rclass_1, rclass_2, rclass_3, rclass_4;
HOST_WIDE_INT offval_1, offval_2, offval_3, offval_4, msize;
rtx mem_1, mem_2, mem_3, mem_4, reg_1, reg_2, reg_3, reg_4;
rtx base_1, base_2, base_3, base_4, offset_1, offset_2, offset_3, offset_4;
if (load)
{
reg_1 = operands[0];
mem_1 = operands[1];
reg_2 = operands[2];
mem_2 = operands[3];
reg_3 = operands[4];
mem_3 = operands[5];
reg_4 = operands[6];
mem_4 = operands[7];
gcc_assert (REG_P (reg_1) && REG_P (reg_2)
&& REG_P (reg_3) && REG_P (reg_4));
if (REGNO (reg_1) == REGNO (reg_2) || REGNO (reg_3) == REGNO (reg_4))
return false;
}
else
{
mem_1 = operands[0];
reg_1 = operands[1];
mem_2 = operands[2];
reg_2 = operands[3];
mem_3 = operands[4];
reg_3 = operands[5];
mem_4 = operands[6];
reg_4 = operands[7];
}
if (!MEM_P (mem_1) || aarch64_mem_pair_operand (mem_1, mode))
return false;
if (MEM_VOLATILE_P (mem_1) || MEM_VOLATILE_P (mem_2)
|| MEM_VOLATILE_P (mem_3) ||MEM_VOLATILE_P (mem_4))
return false;
extract_base_offset_in_addr (mem_1, &base_1, &offset_1);
if (base_1 == NULL_RTX || offset_1 == NULL_RTX)
return false;
extract_base_offset_in_addr (mem_2, &base_2, &offset_2);
if (base_2 == NULL_RTX || offset_2 == NULL_RTX)
return false;
extract_base_offset_in_addr (mem_3, &base_3, &offset_3);
if (base_3 == NULL_RTX || offset_3 == NULL_RTX)
return false;
extract_base_offset_in_addr (mem_4, &base_4, &offset_4);
if (base_4 == NULL_RTX || offset_4 == NULL_RTX)
return false;
if (!rtx_equal_p (base_1, base_2)
|| !rtx_equal_p (base_2, base_3)
|| !rtx_equal_p (base_3, base_4))
return false;
offval_1 = INTVAL (offset_1);
offval_2 = INTVAL (offset_2);
offval_3 = INTVAL (offset_3);
offval_4 = INTVAL (offset_4);
msize = GET_MODE_SIZE (mode);
if ((offval_1 != (offval_2 + msize)
|| offval_1 != (offval_3 + msize * 2)
|| offval_1 != (offval_4 + msize * 3))
&& (offval_4 != (offval_3 + msize)
|| offval_4 != (offval_2 + msize * 2)
|| offval_4 != (offval_1 + msize * 3)))
return false;
if (load)
{
if (reg_mentioned_p (reg_1, mem_1)
|| reg_mentioned_p (reg_2, mem_2)
|| reg_mentioned_p (reg_3, mem_3))
return false;
if (offval_1 > offval_2 && reg_mentioned_p (reg_4, mem_4))
return false;
}
if (mode == SImode
&& (aarch64_tune_params.extra_tuning_flags
& AARCH64_EXTRA_TUNE_SLOW_UNALIGNED_LDPW)
&& !optimize_size
&& MEM_ALIGN (mem_1) < 8 * BITS_PER_UNIT)
return false;
if (REG_P (reg_1) && FP_REGNUM_P (REGNO (reg_1)))
rclass_1 = FP_REGS;
else
rclass_1 = GENERAL_REGS;
if (REG_P (reg_2) && FP_REGNUM_P (REGNO (reg_2)))
rclass_2 = FP_REGS;
else
rclass_2 = GENERAL_REGS;
if (REG_P (reg_3) && FP_REGNUM_P (REGNO (reg_3)))
rclass_3 = FP_REGS;
else
rclass_3 = GENERAL_REGS;
if (REG_P (reg_4) && FP_REGNUM_P (REGNO (reg_4)))
rclass_4 = FP_REGS;
else
rclass_4 = GENERAL_REGS;
if (rclass_1 != rclass_2 || rclass_2 != rclass_3 || rclass_3 != rclass_4)
return false;
return true;
}
bool
aarch64_gen_adjusted_ldpstp (rtx *operands, bool load,
scalar_mode mode, RTX_CODE code)
{
rtx base, offset, t1, t2;
rtx mem_1, mem_2, mem_3, mem_4;
HOST_WIDE_INT off_val, abs_off, adj_off, new_off, stp_off_limit, msize;
if (load)
{
mem_1 = operands[1];
mem_2 = operands[3];
mem_3 = operands[5];
mem_4 = operands[7];
}
else
{
mem_1 = operands[0];
mem_2 = operands[2];
mem_3 = operands[4];
mem_4 = operands[6];
gcc_assert (code == UNKNOWN);
}
extract_base_offset_in_addr (mem_1, &base, &offset);
gcc_assert (base != NULL_RTX && offset != NULL_RTX);
msize = GET_MODE_SIZE (mode);
stp_off_limit = msize * 0x40;
off_val = INTVAL (offset);
abs_off = (off_val < 0) ? -off_val : off_val;
new_off = abs_off % stp_off_limit;
adj_off = abs_off - new_off;
if ((new_off + msize * 2) >= stp_off_limit)
{
adj_off += stp_off_limit;
new_off -= stp_off_limit;
}
if (adj_off >= 0x1000)
return false;
if (off_val < 0)
{
adj_off = -adj_off;
new_off = -new_off;
}
mem_1 = change_address (mem_1, VOIDmode,
plus_constant (DImode, operands[8], new_off));
if (!aarch64_mem_pair_operand (mem_1, mode))
return false;
msize = GET_MODE_SIZE (mode);
mem_2 = change_address (mem_2, VOIDmode,
plus_constant (DImode,
operands[8],
new_off + msize));
mem_3 = change_address (mem_3, VOIDmode,
plus_constant (DImode,
operands[8],
new_off + msize * 2));
mem_4 = change_address (mem_4, VOIDmode,
plus_constant (DImode,
operands[8],
new_off + msize * 3));
if (code == ZERO_EXTEND)
{
mem_1 = gen_rtx_ZERO_EXTEND (DImode, mem_1);
mem_2 = gen_rtx_ZERO_EXTEND (DImode, mem_2);
mem_3 = gen_rtx_ZERO_EXTEND (DImode, mem_3);
mem_4 = gen_rtx_ZERO_EXTEND (DImode, mem_4);
}
else if (code == SIGN_EXTEND)
{
mem_1 = gen_rtx_SIGN_EXTEND (DImode, mem_1);
mem_2 = gen_rtx_SIGN_EXTEND (DImode, mem_2);
mem_3 = gen_rtx_SIGN_EXTEND (DImode, mem_3);
mem_4 = gen_rtx_SIGN_EXTEND (DImode, mem_4);
}
if (load)
{
operands[1] = mem_1;
operands[3] = mem_2;
operands[5] = mem_3;
operands[7] = mem_4;
}
else
{
operands[0] = mem_1;
operands[2] = mem_2;
operands[4] = mem_3;
operands[6] = mem_4;
}
emit_insn (gen_rtx_SET (operands[8], plus_constant (DImode, base, adj_off)));
t1 = gen_rtx_SET (operands[0], operands[1]);
t2 = gen_rtx_SET (operands[2], operands[3]);
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, t1, t2)));
t1 = gen_rtx_SET (operands[4], operands[5]);
t2 = gen_rtx_SET (operands[6], operands[7]);
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, t1, t2)));
return true;
}
static bool
aarch64_empty_mask_is_expensive (unsigned)
{
return false;
}
bool
aarch64_use_pseudo_pic_reg (void)
{
return aarch64_cmodel == AARCH64_CMODEL_SMALL_SPIC;
}
static int
aarch64_unspec_may_trap_p (const_rtx x, unsigned flags)
{
switch (XINT (x, 1))
{
case UNSPEC_GOTSMALLPIC:
case UNSPEC_GOTSMALLPIC28K:
case UNSPEC_GOTTINYPIC:
return 0;
default:
break;
}
return default_unspec_may_trap_p (x, flags);
}
int
aarch64_fpconst_pow_of_2 (rtx x)
{
const REAL_VALUE_TYPE *r;
if (!CONST_DOUBLE_P (x))
return -1;
r = CONST_DOUBLE_REAL_VALUE (x);
if (REAL_VALUE_NEGATIVE (*r)
|| REAL_VALUE_ISNAN (*r)
|| REAL_VALUE_ISINF (*r)
|| !real_isinteger (r, DFmode))
return -1;
return exact_log2 (real_to_integer (r));
}
int
aarch64_vec_fpconst_pow_of_2 (rtx x)
{
int nelts;
if (GET_CODE (x) != CONST_VECTOR
|| !CONST_VECTOR_NUNITS (x).is_constant (&nelts))
return -1;
if (GET_MODE_CLASS (GET_MODE (x)) != MODE_VECTOR_FLOAT)
return -1;
int firstval = aarch64_fpconst_pow_of_2 (CONST_VECTOR_ELT (x, 0));
if (firstval <= 0)
return -1;
for (int i = 1; i < nelts; i++)
if (aarch64_fpconst_pow_of_2 (CONST_VECTOR_ELT (x, i)) != firstval)
return -1;
return firstval;
}
static tree
aarch64_promoted_type (const_tree t)
{
if (SCALAR_FLOAT_TYPE_P (t)
&& TYPE_MAIN_VARIANT (t) == aarch64_fp16_type_node)
return float_type_node;
return NULL_TREE;
}
static bool
aarch64_optab_supported_p (int op, machine_mode mode1, machine_mode,
optimization_type opt_type)
{
switch (op)
{
case rsqrt_optab:
return opt_type == OPTIMIZE_FOR_SPEED && use_rsqrt_p (mode1);
default:
return true;
}
}
static unsigned int
aarch64_dwarf_poly_indeterminate_value (unsigned int i, unsigned int *factor,
int *offset)
{
gcc_assert (i == 1);
*factor = 2;
*offset = 1;
return AARCH64_DWARF_VG;
}
static bool
aarch64_libgcc_floating_mode_supported_p (scalar_float_mode mode)
{
return (mode == HFmode
? true
: default_libgcc_floating_mode_supported_p (mode));
}
static bool
aarch64_scalar_mode_supported_p (scalar_mode mode)
{
return (mode == HFmode
? true
: default_scalar_mode_supported_p (mode));
}
static enum flt_eval_method
aarch64_excess_precision (enum excess_precision_type type)
{
switch (type)
{
case EXCESS_PRECISION_TYPE_FAST:
case EXCESS_PRECISION_TYPE_STANDARD:
return (TARGET_FP_F16INST
? FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16
: FLT_EVAL_METHOD_PROMOTE_TO_FLOAT);
case EXCESS_PRECISION_TYPE_IMPLICIT:
return FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16;
default:
gcc_unreachable ();
}
return FLT_EVAL_METHOD_UNPREDICTABLE;
}
static bool
aarch64_sched_can_speculate_insn (rtx_insn *insn)
{
switch (get_attr_type (insn))
{
case TYPE_SDIV:
case TYPE_UDIV:
case TYPE_FDIVS:
case TYPE_FDIVD:
case TYPE_FSQRTS:
case TYPE_FSQRTD:
case TYPE_NEON_FP_SQRT_S:
case TYPE_NEON_FP_SQRT_D:
case TYPE_NEON_FP_SQRT_S_Q:
case TYPE_NEON_FP_SQRT_D_Q:
case TYPE_NEON_FP_DIV_S:
case TYPE_NEON_FP_DIV_D:
case TYPE_NEON_FP_DIV_S_Q:
case TYPE_NEON_FP_DIV_D_Q:
return false;
default:
return true;
}
}
static int
aarch64_compute_pressure_classes (reg_class *classes)
{
int i = 0;
classes[i++] = GENERAL_REGS;
classes[i++] = FP_REGS;
classes[i++] = PR_LO_REGS;
classes[i++] = PR_HI_REGS;
return i;
}
static bool
aarch64_can_change_mode_class (machine_mode from,
machine_mode to, reg_class_t)
{
if (BYTES_BIG_ENDIAN)
{
bool from_sve_p = aarch64_sve_data_mode_p (from);
bool to_sve_p = aarch64_sve_data_mode_p (to);
if (from_sve_p != to_sve_p)
return false;
if (from_sve_p && GET_MODE_UNIT_SIZE (from) != GET_MODE_UNIT_SIZE (to))
return false;
}
return true;
}
static void
aarch64_select_early_remat_modes (sbitmap modes)
{
for (int i = 0; i < NUM_MACHINE_MODES; ++i)
{
machine_mode mode = (machine_mode) i;
unsigned int vec_flags = aarch64_classify_vector_mode (mode);
if (vec_flags & VEC_ANY_SVE)
bitmap_set_bit (modes, i);
}
}
#if CHECKING_P
namespace selftest {
static void
aarch64_test_loading_full_dump ()
{
rtl_dump_test t (SELFTEST_LOCATION, locate_file ("aarch64/times-two.rtl"));
ASSERT_STREQ ("times_two", IDENTIFIER_POINTER (DECL_NAME (cfun->decl)));
rtx_insn *insn_1 = get_insn_by_uid (1);
ASSERT_EQ (NOTE, GET_CODE (insn_1));
rtx_insn *insn_15 = get_insn_by_uid (15);
ASSERT_EQ (INSN, GET_CODE (insn_15));
ASSERT_EQ (USE, GET_CODE (PATTERN (insn_15)));
ASSERT_EQ (REG, GET_CODE (crtl->return_rtx));
ASSERT_EQ (0, REGNO (crtl->return_rtx));
ASSERT_EQ (SImode, GET_MODE (crtl->return_rtx));
}
static void
aarch64_run_selftests (void)
{
aarch64_test_loading_full_dump ();
}
} 
#endif 
#undef TARGET_ADDRESS_COST
#define TARGET_ADDRESS_COST aarch64_address_cost
#undef TARGET_ALIGN_ANON_BITFIELD
#define TARGET_ALIGN_ANON_BITFIELD hook_bool_void_true
#undef TARGET_ASM_ALIGNED_DI_OP
#define TARGET_ASM_ALIGNED_DI_OP "\t.xword\t"
#undef TARGET_ASM_ALIGNED_HI_OP
#define TARGET_ASM_ALIGNED_HI_OP "\t.hword\t"
#undef TARGET_ASM_ALIGNED_SI_OP
#define TARGET_ASM_ALIGNED_SI_OP "\t.word\t"
#undef TARGET_ASM_CAN_OUTPUT_MI_THUNK
#define TARGET_ASM_CAN_OUTPUT_MI_THUNK \
hook_bool_const_tree_hwi_hwi_const_tree_true
#undef TARGET_ASM_FILE_START
#define TARGET_ASM_FILE_START aarch64_start_file
#undef TARGET_ASM_OUTPUT_MI_THUNK
#define TARGET_ASM_OUTPUT_MI_THUNK aarch64_output_mi_thunk
#undef TARGET_ASM_SELECT_RTX_SECTION
#define TARGET_ASM_SELECT_RTX_SECTION aarch64_select_rtx_section
#undef TARGET_ASM_TRAMPOLINE_TEMPLATE
#define TARGET_ASM_TRAMPOLINE_TEMPLATE aarch64_asm_trampoline_template
#undef TARGET_BUILD_BUILTIN_VA_LIST
#define TARGET_BUILD_BUILTIN_VA_LIST aarch64_build_builtin_va_list
#undef TARGET_CALLEE_COPIES
#define TARGET_CALLEE_COPIES hook_bool_CUMULATIVE_ARGS_mode_tree_bool_false
#undef TARGET_CAN_ELIMINATE
#define TARGET_CAN_ELIMINATE aarch64_can_eliminate
#undef TARGET_CAN_INLINE_P
#define TARGET_CAN_INLINE_P aarch64_can_inline_p
#undef TARGET_CANNOT_FORCE_CONST_MEM
#define TARGET_CANNOT_FORCE_CONST_MEM aarch64_cannot_force_const_mem
#undef TARGET_CASE_VALUES_THRESHOLD
#define TARGET_CASE_VALUES_THRESHOLD aarch64_case_values_threshold
#undef TARGET_CONDITIONAL_REGISTER_USAGE
#define TARGET_CONDITIONAL_REGISTER_USAGE aarch64_conditional_register_usage
#undef TARGET_CXX_GUARD_MASK_BIT
#define TARGET_CXX_GUARD_MASK_BIT hook_bool_void_true
#undef TARGET_C_MODE_FOR_SUFFIX
#define TARGET_C_MODE_FOR_SUFFIX aarch64_c_mode_for_suffix
#ifdef TARGET_BIG_ENDIAN_DEFAULT
#undef  TARGET_DEFAULT_TARGET_FLAGS
#define TARGET_DEFAULT_TARGET_FLAGS (MASK_BIG_END)
#endif
#undef TARGET_CLASS_MAX_NREGS
#define TARGET_CLASS_MAX_NREGS aarch64_class_max_nregs
#undef TARGET_BUILTIN_DECL
#define TARGET_BUILTIN_DECL aarch64_builtin_decl
#undef TARGET_BUILTIN_RECIPROCAL
#define TARGET_BUILTIN_RECIPROCAL aarch64_builtin_reciprocal
#undef TARGET_C_EXCESS_PRECISION
#define TARGET_C_EXCESS_PRECISION aarch64_excess_precision
#undef  TARGET_EXPAND_BUILTIN
#define TARGET_EXPAND_BUILTIN aarch64_expand_builtin
#undef TARGET_EXPAND_BUILTIN_VA_START
#define TARGET_EXPAND_BUILTIN_VA_START aarch64_expand_builtin_va_start
#undef TARGET_FOLD_BUILTIN
#define TARGET_FOLD_BUILTIN aarch64_fold_builtin
#undef TARGET_FUNCTION_ARG
#define TARGET_FUNCTION_ARG aarch64_function_arg
#undef TARGET_FUNCTION_ARG_ADVANCE
#define TARGET_FUNCTION_ARG_ADVANCE aarch64_function_arg_advance
#undef TARGET_FUNCTION_ARG_BOUNDARY
#define TARGET_FUNCTION_ARG_BOUNDARY aarch64_function_arg_boundary
#undef TARGET_FUNCTION_ARG_PADDING
#define TARGET_FUNCTION_ARG_PADDING aarch64_function_arg_padding
#undef TARGET_GET_RAW_RESULT_MODE
#define TARGET_GET_RAW_RESULT_MODE aarch64_get_reg_raw_mode
#undef TARGET_GET_RAW_ARG_MODE
#define TARGET_GET_RAW_ARG_MODE aarch64_get_reg_raw_mode
#undef TARGET_FUNCTION_OK_FOR_SIBCALL
#define TARGET_FUNCTION_OK_FOR_SIBCALL aarch64_function_ok_for_sibcall
#undef TARGET_FUNCTION_VALUE
#define TARGET_FUNCTION_VALUE aarch64_function_value
#undef TARGET_FUNCTION_VALUE_REGNO_P
#define TARGET_FUNCTION_VALUE_REGNO_P aarch64_function_value_regno_p
#undef TARGET_GIMPLE_FOLD_BUILTIN
#define TARGET_GIMPLE_FOLD_BUILTIN aarch64_gimple_fold_builtin
#undef TARGET_GIMPLIFY_VA_ARG_EXPR
#define TARGET_GIMPLIFY_VA_ARG_EXPR aarch64_gimplify_va_arg_expr
#undef  TARGET_INIT_BUILTINS
#define TARGET_INIT_BUILTINS  aarch64_init_builtins
#undef TARGET_IRA_CHANGE_PSEUDO_ALLOCNO_CLASS
#define TARGET_IRA_CHANGE_PSEUDO_ALLOCNO_CLASS \
aarch64_ira_change_pseudo_allocno_class
#undef TARGET_LEGITIMATE_ADDRESS_P
#define TARGET_LEGITIMATE_ADDRESS_P aarch64_legitimate_address_hook_p
#undef TARGET_LEGITIMATE_CONSTANT_P
#define TARGET_LEGITIMATE_CONSTANT_P aarch64_legitimate_constant_p
#undef TARGET_LEGITIMIZE_ADDRESS_DISPLACEMENT
#define TARGET_LEGITIMIZE_ADDRESS_DISPLACEMENT \
aarch64_legitimize_address_displacement
#undef TARGET_LIBGCC_CMP_RETURN_MODE
#define TARGET_LIBGCC_CMP_RETURN_MODE aarch64_libgcc_cmp_return_mode
#undef TARGET_LIBGCC_FLOATING_MODE_SUPPORTED_P
#define TARGET_LIBGCC_FLOATING_MODE_SUPPORTED_P \
aarch64_libgcc_floating_mode_supported_p
#undef TARGET_MANGLE_TYPE
#define TARGET_MANGLE_TYPE aarch64_mangle_type
#undef TARGET_MEMORY_MOVE_COST
#define TARGET_MEMORY_MOVE_COST aarch64_memory_move_cost
#undef TARGET_MIN_DIVISIONS_FOR_RECIP_MUL
#define TARGET_MIN_DIVISIONS_FOR_RECIP_MUL aarch64_min_divisions_for_recip_mul
#undef TARGET_MUST_PASS_IN_STACK
#define TARGET_MUST_PASS_IN_STACK must_pass_in_stack_var_size
#undef TARGET_NARROW_VOLATILE_BITFIELD
#define TARGET_NARROW_VOLATILE_BITFIELD hook_bool_void_false
#undef  TARGET_OPTION_OVERRIDE
#define TARGET_OPTION_OVERRIDE aarch64_override_options
#undef TARGET_OVERRIDE_OPTIONS_AFTER_CHANGE
#define TARGET_OVERRIDE_OPTIONS_AFTER_CHANGE \
aarch64_override_options_after_change
#undef TARGET_OPTION_SAVE
#define TARGET_OPTION_SAVE aarch64_option_save
#undef TARGET_OPTION_RESTORE
#define TARGET_OPTION_RESTORE aarch64_option_restore
#undef TARGET_OPTION_PRINT
#define TARGET_OPTION_PRINT aarch64_option_print
#undef TARGET_OPTION_VALID_ATTRIBUTE_P
#define TARGET_OPTION_VALID_ATTRIBUTE_P aarch64_option_valid_attribute_p
#undef TARGET_SET_CURRENT_FUNCTION
#define TARGET_SET_CURRENT_FUNCTION aarch64_set_current_function
#undef TARGET_PASS_BY_REFERENCE
#define TARGET_PASS_BY_REFERENCE aarch64_pass_by_reference
#undef TARGET_PREFERRED_RELOAD_CLASS
#define TARGET_PREFERRED_RELOAD_CLASS aarch64_preferred_reload_class
#undef TARGET_SCHED_REASSOCIATION_WIDTH
#define TARGET_SCHED_REASSOCIATION_WIDTH aarch64_reassociation_width
#undef TARGET_PROMOTED_TYPE
#define TARGET_PROMOTED_TYPE aarch64_promoted_type
#undef TARGET_SECONDARY_RELOAD
#define TARGET_SECONDARY_RELOAD aarch64_secondary_reload
#undef TARGET_SHIFT_TRUNCATION_MASK
#define TARGET_SHIFT_TRUNCATION_MASK aarch64_shift_truncation_mask
#undef TARGET_SETUP_INCOMING_VARARGS
#define TARGET_SETUP_INCOMING_VARARGS aarch64_setup_incoming_varargs
#undef TARGET_STRUCT_VALUE_RTX
#define TARGET_STRUCT_VALUE_RTX   aarch64_struct_value_rtx
#undef TARGET_REGISTER_MOVE_COST
#define TARGET_REGISTER_MOVE_COST aarch64_register_move_cost
#undef TARGET_RETURN_IN_MEMORY
#define TARGET_RETURN_IN_MEMORY aarch64_return_in_memory
#undef TARGET_RETURN_IN_MSB
#define TARGET_RETURN_IN_MSB aarch64_return_in_msb
#undef TARGET_RTX_COSTS
#define TARGET_RTX_COSTS aarch64_rtx_costs_wrapper
#undef TARGET_SCALAR_MODE_SUPPORTED_P
#define TARGET_SCALAR_MODE_SUPPORTED_P aarch64_scalar_mode_supported_p
#undef TARGET_SCHED_ISSUE_RATE
#define TARGET_SCHED_ISSUE_RATE aarch64_sched_issue_rate
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD \
aarch64_sched_first_cycle_multipass_dfa_lookahead
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD \
aarch64_first_cycle_multipass_dfa_lookahead_guard
#undef TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS
#define TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS \
aarch64_get_separate_components
#undef TARGET_SHRINK_WRAP_COMPONENTS_FOR_BB
#define TARGET_SHRINK_WRAP_COMPONENTS_FOR_BB \
aarch64_components_for_bb
#undef TARGET_SHRINK_WRAP_DISQUALIFY_COMPONENTS
#define TARGET_SHRINK_WRAP_DISQUALIFY_COMPONENTS \
aarch64_disqualify_components
#undef TARGET_SHRINK_WRAP_EMIT_PROLOGUE_COMPONENTS
#define TARGET_SHRINK_WRAP_EMIT_PROLOGUE_COMPONENTS \
aarch64_emit_prologue_components
#undef TARGET_SHRINK_WRAP_EMIT_EPILOGUE_COMPONENTS
#define TARGET_SHRINK_WRAP_EMIT_EPILOGUE_COMPONENTS \
aarch64_emit_epilogue_components
#undef TARGET_SHRINK_WRAP_SET_HANDLED_COMPONENTS
#define TARGET_SHRINK_WRAP_SET_HANDLED_COMPONENTS \
aarch64_set_handled_components
#undef TARGET_TRAMPOLINE_INIT
#define TARGET_TRAMPOLINE_INIT aarch64_trampoline_init
#undef TARGET_USE_BLOCKS_FOR_CONSTANT_P
#define TARGET_USE_BLOCKS_FOR_CONSTANT_P aarch64_use_blocks_for_constant_p
#undef TARGET_VECTOR_MODE_SUPPORTED_P
#define TARGET_VECTOR_MODE_SUPPORTED_P aarch64_vector_mode_supported_p
#undef TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT
#define TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT \
aarch64_builtin_support_vector_misalignment
#undef TARGET_ARRAY_MODE
#define TARGET_ARRAY_MODE aarch64_array_mode
#undef TARGET_ARRAY_MODE_SUPPORTED_P
#define TARGET_ARRAY_MODE_SUPPORTED_P aarch64_array_mode_supported_p
#undef TARGET_VECTORIZE_ADD_STMT_COST
#define TARGET_VECTORIZE_ADD_STMT_COST aarch64_add_stmt_cost
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST
#define TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST \
aarch64_builtin_vectorization_cost
#undef TARGET_VECTORIZE_PREFERRED_SIMD_MODE
#define TARGET_VECTORIZE_PREFERRED_SIMD_MODE aarch64_preferred_simd_mode
#undef TARGET_VECTORIZE_BUILTINS
#define TARGET_VECTORIZE_BUILTINS
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION
#define TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION \
aarch64_builtin_vectorized_function
#undef TARGET_VECTORIZE_AUTOVECTORIZE_VECTOR_SIZES
#define TARGET_VECTORIZE_AUTOVECTORIZE_VECTOR_SIZES \
aarch64_autovectorize_vector_sizes
#undef TARGET_ATOMIC_ASSIGN_EXPAND_FENV
#define TARGET_ATOMIC_ASSIGN_EXPAND_FENV \
aarch64_atomic_assign_expand_fenv
#undef TARGET_MIN_ANCHOR_OFFSET
#define TARGET_MIN_ANCHOR_OFFSET -256
#undef TARGET_MAX_ANCHOR_OFFSET
#define TARGET_MAX_ANCHOR_OFFSET 4095
#undef TARGET_VECTOR_ALIGNMENT
#define TARGET_VECTOR_ALIGNMENT aarch64_simd_vector_alignment
#undef TARGET_VECTORIZE_PREFERRED_VECTOR_ALIGNMENT
#define TARGET_VECTORIZE_PREFERRED_VECTOR_ALIGNMENT \
aarch64_vectorize_preferred_vector_alignment
#undef TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE
#define TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE \
aarch64_simd_vector_alignment_reachable
#undef TARGET_VECTORIZE_VEC_PERM_CONST
#define TARGET_VECTORIZE_VEC_PERM_CONST \
aarch64_vectorize_vec_perm_const
#undef TARGET_VECTORIZE_GET_MASK_MODE
#define TARGET_VECTORIZE_GET_MASK_MODE aarch64_get_mask_mode
#undef TARGET_VECTORIZE_EMPTY_MASK_IS_EXPENSIVE
#define TARGET_VECTORIZE_EMPTY_MASK_IS_EXPENSIVE \
aarch64_empty_mask_is_expensive
#undef TARGET_INIT_LIBFUNCS
#define TARGET_INIT_LIBFUNCS aarch64_init_libfuncs
#undef TARGET_FIXED_CONDITION_CODE_REGS
#define TARGET_FIXED_CONDITION_CODE_REGS aarch64_fixed_condition_code_regs
#undef TARGET_FLAGS_REGNUM
#define TARGET_FLAGS_REGNUM CC_REGNUM
#undef TARGET_CALL_FUSAGE_CONTAINS_NON_CALLEE_CLOBBERS
#define TARGET_CALL_FUSAGE_CONTAINS_NON_CALLEE_CLOBBERS true
#undef TARGET_ASAN_SHADOW_OFFSET
#define TARGET_ASAN_SHADOW_OFFSET aarch64_asan_shadow_offset
#undef TARGET_LEGITIMIZE_ADDRESS
#define TARGET_LEGITIMIZE_ADDRESS aarch64_legitimize_address
#undef TARGET_SCHED_CAN_SPECULATE_INSN
#define TARGET_SCHED_CAN_SPECULATE_INSN aarch64_sched_can_speculate_insn
#undef TARGET_CAN_USE_DOLOOP_P
#define TARGET_CAN_USE_DOLOOP_P can_use_doloop_if_innermost
#undef TARGET_SCHED_ADJUST_PRIORITY
#define TARGET_SCHED_ADJUST_PRIORITY aarch64_sched_adjust_priority
#undef TARGET_SCHED_MACRO_FUSION_P
#define TARGET_SCHED_MACRO_FUSION_P aarch64_macro_fusion_p
#undef TARGET_SCHED_MACRO_FUSION_PAIR_P
#define TARGET_SCHED_MACRO_FUSION_PAIR_P aarch_macro_fusion_pair_p
#undef TARGET_SCHED_FUSION_PRIORITY
#define TARGET_SCHED_FUSION_PRIORITY aarch64_sched_fusion_priority
#undef TARGET_UNSPEC_MAY_TRAP_P
#define TARGET_UNSPEC_MAY_TRAP_P aarch64_unspec_may_trap_p
#undef TARGET_USE_PSEUDO_PIC_REG
#define TARGET_USE_PSEUDO_PIC_REG aarch64_use_pseudo_pic_reg
#undef TARGET_PRINT_OPERAND
#define TARGET_PRINT_OPERAND aarch64_print_operand
#undef TARGET_PRINT_OPERAND_ADDRESS
#define TARGET_PRINT_OPERAND_ADDRESS aarch64_print_operand_address
#undef TARGET_OPTAB_SUPPORTED_P
#define TARGET_OPTAB_SUPPORTED_P aarch64_optab_supported_p
#undef TARGET_OMIT_STRUCT_RETURN_REG
#define TARGET_OMIT_STRUCT_RETURN_REG true
#undef TARGET_DWARF_POLY_INDETERMINATE_VALUE
#define TARGET_DWARF_POLY_INDETERMINATE_VALUE \
aarch64_dwarf_poly_indeterminate_value
#undef TARGET_CUSTOM_FUNCTION_DESCRIPTORS
#define TARGET_CUSTOM_FUNCTION_DESCRIPTORS 4
#undef TARGET_HARD_REGNO_NREGS
#define TARGET_HARD_REGNO_NREGS aarch64_hard_regno_nregs
#undef TARGET_HARD_REGNO_MODE_OK
#define TARGET_HARD_REGNO_MODE_OK aarch64_hard_regno_mode_ok
#undef TARGET_MODES_TIEABLE_P
#define TARGET_MODES_TIEABLE_P aarch64_modes_tieable_p
#undef TARGET_HARD_REGNO_CALL_PART_CLOBBERED
#define TARGET_HARD_REGNO_CALL_PART_CLOBBERED \
aarch64_hard_regno_call_part_clobbered
#undef TARGET_CONSTANT_ALIGNMENT
#define TARGET_CONSTANT_ALIGNMENT aarch64_constant_alignment
#undef TARGET_COMPUTE_PRESSURE_CLASSES
#define TARGET_COMPUTE_PRESSURE_CLASSES aarch64_compute_pressure_classes
#undef TARGET_CAN_CHANGE_MODE_CLASS
#define TARGET_CAN_CHANGE_MODE_CLASS aarch64_can_change_mode_class
#undef TARGET_SELECT_EARLY_REMAT_MODES
#define TARGET_SELECT_EARLY_REMAT_MODES aarch64_select_early_remat_modes
#if CHECKING_P
#undef TARGET_RUN_TARGET_SELFTESTS
#define TARGET_RUN_TARGET_SELFTESTS selftest::aarch64_run_selftests
#endif 
struct gcc_target targetm = TARGET_INITIALIZER;
#include "gt-aarch64.h"
