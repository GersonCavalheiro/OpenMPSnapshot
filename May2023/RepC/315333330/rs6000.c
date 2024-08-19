#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "gimple.h"
#include "cfghooks.h"
#include "cfgloop.h"
#include "df.h"
#include "tm_p.h"
#include "stringpool.h"
#include "expmed.h"
#include "optabs.h"
#include "regs.h"
#include "ira.h"
#include "recog.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "insn-attr.h"
#include "flags.h"
#include "alias.h"
#include "fold-const.h"
#include "attribs.h"
#include "stor-layout.h"
#include "calls.h"
#include "print-tree.h"
#include "varasm.h"
#include "explow.h"
#include "expr.h"
#include "output.h"
#include "dbxout.h"
#include "common/common-target.h"
#include "langhooks.h"
#include "reload.h"
#include "sched-int.h"
#include "gimplify.h"
#include "gimple-fold.h"
#include "gimple-iterator.h"
#include "gimple-ssa.h"
#include "gimple-walk.h"
#include "intl.h"
#include "params.h"
#include "tm-constrs.h"
#include "tree-vectorizer.h"
#include "target-globals.h"
#include "builtins.h"
#include "tree-vector-builder.h"
#include "context.h"
#include "tree-pass.h"
#include "except.h"
#if TARGET_XCOFF
#include "xcoffout.h"  
#endif
#if TARGET_MACHO
#include "gstab.h"  
#endif
#include "case-cfn-macros.h"
#include "ppc-auxv.h"
#include "tree-ssa-propagate.h"
#include "tree-vrp.h"
#include "tree-ssanames.h"
#include "target-def.h"
#ifndef TARGET_NO_PROTOTYPE
#define TARGET_NO_PROTOTYPE 0
#endif
#ifndef TARGET_IEEEQUAD_DEFAULT
#if !defined (POWERPC_LINUX) && !defined (POWERPC_FREEBSD)
#define TARGET_IEEEQUAD_DEFAULT 1
#else
#define TARGET_IEEEQUAD_DEFAULT 0
#endif
#endif
#define min(A,B)	((A) < (B) ? (A) : (B))
#define max(A,B)	((A) > (B) ? (A) : (B))
static pad_direction rs6000_function_arg_padding (machine_mode, const_tree);
typedef struct rs6000_stack {
int reload_completed;		
int first_gp_reg_save;	
int first_fp_reg_save;	
int first_altivec_reg_save;	
int lr_save_p;		
int cr_save_p;		
unsigned int vrsave_mask;	
int push_p;			
int calls_p;			
int world_save_p;		
enum rs6000_abi abi;		
int gp_save_offset;		
int fp_save_offset;		
int altivec_save_offset;	
int lr_save_offset;		
int cr_save_offset;		
int vrsave_save_offset;	
int varargs_save_offset;	
int ehrd_offset;		
int ehcr_offset;		
int reg_size;			
HOST_WIDE_INT vars_size;	
int parm_size;		
int save_size;		
int fixed_size;		
int gp_size;			
int fp_size;			
int altivec_size;		
int cr_size;			
int vrsave_size;		
int altivec_padding_size;	
HOST_WIDE_INT total_size;	
int savres_strategy;
} rs6000_stack_t;
typedef struct GTY(()) machine_function
{
int ra_needs_full_frame;
int ra_need_lr;
int lr_save_state;
bool save_toc_in_prologue;
HOST_WIDE_INT varargs_save_offset;
rtx split_stack_arg_pointer;
bool split_stack_argp_used;
bool r2_setup_needed;
int n_components;
bool gpr_is_wrapped_separately[32];
bool fpr_is_wrapped_separately[32];
bool lr_is_wrapped_separately;
bool toc_is_wrapped_separately;
} machine_function;
static GTY(()) tree altivec_builtin_mask_for_load;
static GTY(()) int common_mode_defined;
static int rs6000_pic_labelno;
#ifdef USING_ELFOS_H
int fixuplabelno = 0;
#endif
int dot_symbols;
scalar_int_mode rs6000_pmode;
#if TARGET_ELF
static bool rs6000_passes_ieee128;
#endif
static bool ieee128_mangling_gcc_8_1;
unsigned rs6000_pointer_size;
#ifdef HAVE_AS_GNU_ATTRIBUTE
# ifndef HAVE_LD_PPC_GNU_ATTR_LONG_DOUBLE
# define HAVE_LD_PPC_GNU_ATTR_LONG_DOUBLE 0
# endif
static bool rs6000_passes_float;
static bool rs6000_passes_long_double;
static bool rs6000_passes_vector;
static bool rs6000_returns_struct;
#endif
static bool rs6000_hard_regno_mode_ok_p
[NUM_MACHINE_MODES][FIRST_PSEUDO_REGISTER];
unsigned char rs6000_class_max_nregs[NUM_MACHINE_MODES][LIM_REG_CLASSES];
unsigned char rs6000_hard_regno_nregs[NUM_MACHINE_MODES][FIRST_PSEUDO_REGISTER];
enum reg_class rs6000_regno_regclass[FIRST_PSEUDO_REGISTER];
static int dbg_cost_ctrl;
tree rs6000_builtin_types[RS6000_BTI_MAX];
tree rs6000_builtin_decls[RS6000_BUILTIN_COUNT];
int toc_initialized, need_toc_init;
char toc_label_name[10];
static short cached_can_issue_more;
static GTY(()) section *read_only_data_section;
static GTY(()) section *private_data_section;
static GTY(()) section *tls_data_section;
static GTY(()) section *tls_private_data_section;
static GTY(()) section *read_only_private_data_section;
static GTY(()) section *sdata2_section;
static GTY(()) section *toc_section;
struct builtin_description
{
const HOST_WIDE_INT mask;
const enum insn_code icode;
const char *const name;
const enum rs6000_builtins code;
};
enum rs6000_vector rs6000_vector_unit[NUM_MACHINE_MODES];
enum rs6000_vector rs6000_vector_mem[NUM_MACHINE_MODES];
enum reg_class rs6000_constraints[RS6000_CONSTRAINT_MAX];
int rs6000_vector_align[NUM_MACHINE_MODES];
static GTY(()) tree builtin_mode_to_type[MAX_MACHINE_MODE][2];
unsigned char rs6000_recip_bits[MAX_MACHINE_MODE];
enum rs6000_recip_mask {
RECIP_SF_DIV		= 0x001,	
RECIP_DF_DIV		= 0x002,
RECIP_V4SF_DIV	= 0x004,
RECIP_V2DF_DIV	= 0x008,
RECIP_SF_RSQRT	= 0x010,	
RECIP_DF_RSQRT	= 0x020,
RECIP_V4SF_RSQRT	= 0x040,
RECIP_V2DF_RSQRT	= 0x080,
RECIP_NONE		= 0,
RECIP_ALL		= (RECIP_SF_DIV | RECIP_DF_DIV | RECIP_V4SF_DIV
| RECIP_V2DF_DIV | RECIP_SF_RSQRT | RECIP_DF_RSQRT
| RECIP_V4SF_RSQRT | RECIP_V2DF_RSQRT),
RECIP_HIGH_PRECISION	= RECIP_ALL,
RECIP_LOW_PRECISION	= (RECIP_ALL & ~(RECIP_DF_RSQRT | RECIP_V2DF_RSQRT))
};
static struct
{
const char *string;		
unsigned int mask;		
} recip_options[] = {
{ "all",	 RECIP_ALL },
{ "none",	 RECIP_NONE },
{ "div",	 (RECIP_SF_DIV | RECIP_DF_DIV | RECIP_V4SF_DIV
| RECIP_V2DF_DIV) },
{ "divf",	 (RECIP_SF_DIV | RECIP_V4SF_DIV) },
{ "divd",	 (RECIP_DF_DIV | RECIP_V2DF_DIV) },
{ "rsqrt",	 (RECIP_SF_RSQRT | RECIP_DF_RSQRT | RECIP_V4SF_RSQRT
| RECIP_V2DF_RSQRT) },
{ "rsqrtf",	 (RECIP_SF_RSQRT | RECIP_V4SF_RSQRT) },
{ "rsqrtd",	 (RECIP_DF_RSQRT | RECIP_V2DF_RSQRT) },
};
static const struct
{
const char *cpu;
unsigned int cpuid;
} cpu_is_info[] = {
{ "power9",	   PPC_PLATFORM_POWER9 },
{ "power8",	   PPC_PLATFORM_POWER8 },
{ "power7",	   PPC_PLATFORM_POWER7 },
{ "power6x",	   PPC_PLATFORM_POWER6X },
{ "power6",	   PPC_PLATFORM_POWER6 },
{ "power5+",	   PPC_PLATFORM_POWER5_PLUS },
{ "power5",	   PPC_PLATFORM_POWER5 },
{ "ppc970",	   PPC_PLATFORM_PPC970 },
{ "power4",	   PPC_PLATFORM_POWER4 },
{ "ppca2",	   PPC_PLATFORM_PPCA2 },
{ "ppc476",	   PPC_PLATFORM_PPC476 },
{ "ppc464",	   PPC_PLATFORM_PPC464 },
{ "ppc440",	   PPC_PLATFORM_PPC440 },
{ "ppc405",	   PPC_PLATFORM_PPC405 },
{ "ppc-cell-be", PPC_PLATFORM_CELL_BE }
};
static const struct
{
const char *hwcap;
int mask;
unsigned int id;
} cpu_supports_info[] = {
{ "4xxmac",		PPC_FEATURE_HAS_4xxMAC,		0 },
{ "altivec",		PPC_FEATURE_HAS_ALTIVEC,	0 },
{ "arch_2_05",	PPC_FEATURE_ARCH_2_05,		0 },
{ "arch_2_06",	PPC_FEATURE_ARCH_2_06,		0 },
{ "archpmu",		PPC_FEATURE_PERFMON_COMPAT,	0 },
{ "booke",		PPC_FEATURE_BOOKE,		0 },
{ "cellbe",		PPC_FEATURE_CELL_BE,		0 },
{ "dfp",		PPC_FEATURE_HAS_DFP,		0 },
{ "efpdouble",	PPC_FEATURE_HAS_EFP_DOUBLE,	0 },
{ "efpsingle",	PPC_FEATURE_HAS_EFP_SINGLE,	0 },
{ "fpu",		PPC_FEATURE_HAS_FPU,		0 },
{ "ic_snoop",		PPC_FEATURE_ICACHE_SNOOP,	0 },
{ "mmu",		PPC_FEATURE_HAS_MMU,		0 },
{ "notb",		PPC_FEATURE_NO_TB,		0 },
{ "pa6t",		PPC_FEATURE_PA6T,		0 },
{ "power4",		PPC_FEATURE_POWER4,		0 },
{ "power5",		PPC_FEATURE_POWER5,		0 },
{ "power5+",		PPC_FEATURE_POWER5_PLUS,	0 },
{ "power6x",		PPC_FEATURE_POWER6_EXT,		0 },
{ "ppc32",		PPC_FEATURE_32,			0 },
{ "ppc601",		PPC_FEATURE_601_INSTR,		0 },
{ "ppc64",		PPC_FEATURE_64,			0 },
{ "ppcle",		PPC_FEATURE_PPC_LE,		0 },
{ "smt",		PPC_FEATURE_SMT,		0 },
{ "spe",		PPC_FEATURE_HAS_SPE,		0 },
{ "true_le",		PPC_FEATURE_TRUE_LE,		0 },
{ "ucache",		PPC_FEATURE_UNIFIED_CACHE,	0 },
{ "vsx",		PPC_FEATURE_HAS_VSX,		0 },
{ "arch_2_07",	PPC_FEATURE2_ARCH_2_07,		1 },
{ "dscr",		PPC_FEATURE2_HAS_DSCR,		1 },
{ "ebb",		PPC_FEATURE2_HAS_EBB,		1 },
{ "htm",		PPC_FEATURE2_HAS_HTM,		1 },
{ "htm-nosc",		PPC_FEATURE2_HTM_NOSC,		1 },
{ "htm-no-suspend",	PPC_FEATURE2_HTM_NO_SUSPEND,	1 },
{ "isel",		PPC_FEATURE2_HAS_ISEL,		1 },
{ "tar",		PPC_FEATURE2_HAS_TAR,		1 },
{ "vcrypto",		PPC_FEATURE2_HAS_VEC_CRYPTO,	1 },
{ "arch_3_00",	PPC_FEATURE2_ARCH_3_00,		1 },
{ "ieee128",		PPC_FEATURE2_HAS_IEEE128,	1 },
{ "darn",		PPC_FEATURE2_DARN,		1 },
{ "scv",		PPC_FEATURE2_SCV,		1 }
};
enum {
CLONE_DEFAULT		= 0,		
CLONE_ISA_2_05,			
CLONE_ISA_2_06,			
CLONE_ISA_2_07,			
CLONE_ISA_3_00,			
CLONE_MAX
};
struct clone_map {
HOST_WIDE_INT isa_mask;	
const char *name;		
};
static const struct clone_map rs6000_clone_map[CLONE_MAX] = {
{ 0,				"" },		
{ OPTION_MASK_CMPB,		"arch_2_05" },	
{ OPTION_MASK_POPCNTD,	"arch_2_06" },	
{ OPTION_MASK_P8_VECTOR,	"arch_2_07" },	
{ OPTION_MASK_P9_VECTOR,	"arch_3_00" },	
};
const char *tcb_verification_symbol = "__parse_hwcap_and_convert_at_platform";
bool cpu_builtin_p;
void (*rs6000_target_modify_macros_ptr) (bool, HOST_WIDE_INT, HOST_WIDE_INT);
enum rs6000_reg_type {
NO_REG_TYPE,
PSEUDO_REG_TYPE,
GPR_REG_TYPE,
VSX_REG_TYPE,
ALTIVEC_REG_TYPE,
FPR_REG_TYPE,
SPR_REG_TYPE,
CR_REG_TYPE
};
static enum rs6000_reg_type reg_class_to_reg_type[N_REG_CLASSES];
#define IS_STD_REG_TYPE(RTYPE) IN_RANGE(RTYPE, GPR_REG_TYPE, FPR_REG_TYPE)
#define IS_FP_VECT_REG_TYPE(RTYPE) IN_RANGE(RTYPE, VSX_REG_TYPE, FPR_REG_TYPE)
enum rs6000_reload_reg_type {
RELOAD_REG_GPR,			
RELOAD_REG_FPR,			
RELOAD_REG_VMX,			
RELOAD_REG_ANY,			
N_RELOAD_REG
};
#define FIRST_RELOAD_REG_CLASS	RELOAD_REG_GPR
#define LAST_RELOAD_REG_CLASS	RELOAD_REG_VMX
struct reload_reg_map_type {
const char *name;			
int reg;				
};
static const struct reload_reg_map_type reload_reg_map[N_RELOAD_REG] = {
{ "Gpr",	FIRST_GPR_REGNO },	
{ "Fpr",	FIRST_FPR_REGNO },	
{ "VMX",	FIRST_ALTIVEC_REGNO },	
{ "Any",	-1 },			
};
typedef unsigned char addr_mask_type;
#define RELOAD_REG_VALID	0x01	
#define RELOAD_REG_MULTIPLE	0x02	
#define RELOAD_REG_INDEXED	0x04	
#define RELOAD_REG_OFFSET	0x08	
#define RELOAD_REG_PRE_INCDEC	0x10	
#define RELOAD_REG_PRE_MODIFY	0x20	
#define RELOAD_REG_AND_M16	0x40	
#define RELOAD_REG_QUAD_OFFSET	0x80	
struct rs6000_reg_addr {
enum insn_code reload_load;		
enum insn_code reload_store;		
enum insn_code reload_fpr_gpr;	
enum insn_code reload_gpr_vsx;	
enum insn_code reload_vsx_gpr;	
enum insn_code fusion_gpr_ld;		
enum insn_code fusion_addi_ld[(int)N_RELOAD_REG];
enum insn_code fusion_addi_st[(int)N_RELOAD_REG];
enum insn_code fusion_addis_ld[(int)N_RELOAD_REG];
enum insn_code fusion_addis_st[(int)N_RELOAD_REG];
addr_mask_type addr_mask[(int)N_RELOAD_REG]; 
bool scalar_in_vmx_p;			
bool fused_toc;			
};
static struct rs6000_reg_addr reg_addr[NUM_MACHINE_MODES];
static inline bool
mode_supports_pre_incdec_p (machine_mode mode)
{
return ((reg_addr[mode].addr_mask[RELOAD_REG_ANY] & RELOAD_REG_PRE_INCDEC)
!= 0);
}
static inline bool
mode_supports_pre_modify_p (machine_mode mode)
{
return ((reg_addr[mode].addr_mask[RELOAD_REG_ANY] & RELOAD_REG_PRE_MODIFY)
!= 0);
}
int
rs6000_store_data_bypass_p (rtx_insn *out_insn, rtx_insn *in_insn)
{
rtx out_set, in_set;
rtx out_pat, in_pat;
rtx out_exp, in_exp;
int i, j;
in_set = single_set (in_insn);
if (in_set)
{
if (MEM_P (SET_DEST (in_set)))
{
out_set = single_set (out_insn);
if (!out_set)
{
out_pat = PATTERN (out_insn);
if (GET_CODE (out_pat) == PARALLEL)
{
for (i = 0; i < XVECLEN (out_pat, 0); i++)
{
out_exp = XVECEXP (out_pat, 0, i);
if ((GET_CODE (out_exp) == CLOBBER)
|| (GET_CODE (out_exp) == USE))
continue;
else if (GET_CODE (out_exp) != SET)
return false;
}
}
}
}
}
else
{
in_pat = PATTERN (in_insn);
if (GET_CODE (in_pat) != PARALLEL)
return false;
for (i = 0; i < XVECLEN (in_pat, 0); i++)
{
in_exp = XVECEXP (in_pat, 0, i);
if ((GET_CODE (in_exp) == CLOBBER) || (GET_CODE (in_exp) == USE))
continue;
else if (GET_CODE (in_exp) != SET)
return false;
if (MEM_P (SET_DEST (in_exp)))
{
out_set = single_set (out_insn);
if (!out_set)
{
out_pat = PATTERN (out_insn);
if (GET_CODE (out_pat) != PARALLEL)
return false;
for (j = 0; j < XVECLEN (out_pat, 0); j++)
{
out_exp = XVECEXP (out_pat, 0, j);
if ((GET_CODE (out_exp) == CLOBBER)
|| (GET_CODE (out_exp) == USE))
continue;
else if (GET_CODE (out_exp) != SET)
return false;
}
}
}
}
}
return store_data_bypass_p (out_insn, in_insn);
}
static inline bool
mode_supports_vmx_dform (machine_mode mode)
{
return ((reg_addr[mode].addr_mask[RELOAD_REG_VMX] & RELOAD_REG_OFFSET) != 0);
}
static inline bool
mode_supports_vsx_dform_quad (machine_mode mode)
{
return ((reg_addr[mode].addr_mask[RELOAD_REG_ANY] & RELOAD_REG_QUAD_OFFSET)
!= 0);
}

const struct processor_costs *rs6000_cost;
static const
struct processor_costs size32_cost = {
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
32,			
0,			
0,			
0,			
0,			
};
static const
struct processor_costs size64_cost = {
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
COSTS_N_INSNS (1),    
128,			
0,			
0,			
0,			
0,			
};
static const
struct processor_costs rs64a_cost = {
COSTS_N_INSNS (20),   
COSTS_N_INSNS (12),   
COSTS_N_INSNS (8),    
COSTS_N_INSNS (34),   
COSTS_N_INSNS (65),   
COSTS_N_INSNS (67),   
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (31),   
COSTS_N_INSNS (31),   
128,			
128,			
2048,			
1,			
0,			
};
static const
struct processor_costs mpccore_cost = {
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (6),    
COSTS_N_INSNS (6),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (10),   
COSTS_N_INSNS (17),   
32,			
4,			
16,			
1,			
0,			
};
static const
struct processor_costs ppc403_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (33),   
COSTS_N_INSNS (33),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
32,			
4,			
16,			
1,			
0,			
};
static const
struct processor_costs ppc405_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (35),   
COSTS_N_INSNS (35),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
32,			
16,			
128,			
1,			
0,			
};
static const
struct processor_costs ppc440_cost = {
COSTS_N_INSNS (3),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (34),   
COSTS_N_INSNS (34),   
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (19),   
COSTS_N_INSNS (33),   
32,			
32,			
256,			
1,			
0,			
};
static const
struct processor_costs ppc476_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (11),   
COSTS_N_INSNS (11),   
COSTS_N_INSNS (6),    
COSTS_N_INSNS (6),    
COSTS_N_INSNS (19),   
COSTS_N_INSNS (33),   
32,			
32,			
512,			
1,			
0,			
};
static const
struct processor_costs ppc601_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (36),   
COSTS_N_INSNS (36),   
COSTS_N_INSNS (4),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (17),   
COSTS_N_INSNS (31),   
32,			
32,			
256,			
1,			
0,			
};
static const
struct processor_costs ppc603_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (37),   
COSTS_N_INSNS (37),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (33),   
32,			
8,			
64,			
1,			
0,			
};
static const
struct processor_costs ppc604_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (20),   
COSTS_N_INSNS (20),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (32),   
32,			
16,			
512,			
1,			
0,			
};
static const
struct processor_costs ppc604e_cost = {
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (20),   
COSTS_N_INSNS (20),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (32),   
32,			
32,			
1024,			
1,			
0,			
};
static const
struct processor_costs ppc620_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (7),    
COSTS_N_INSNS (21),   
COSTS_N_INSNS (37),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (32),   
128,			
32,			
1024,			
1,			
0,			
};
static const
struct processor_costs ppc630_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (7),    
COSTS_N_INSNS (21),   
COSTS_N_INSNS (37),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (17),   
COSTS_N_INSNS (21),   
128,			
64,			
1024,			
1,			
0,			
};
static const
struct processor_costs ppccell_cost = {
COSTS_N_INSNS (9/2)+2,    
COSTS_N_INSNS (6/2),    
COSTS_N_INSNS (6/2),    
COSTS_N_INSNS (15/2)+2,   
COSTS_N_INSNS (38/2),   
COSTS_N_INSNS (70/2),   
COSTS_N_INSNS (10/2),   
COSTS_N_INSNS (10/2),   
COSTS_N_INSNS (74/2),   
COSTS_N_INSNS (74/2),   
128,			
32,			
512,			
6,			
0,			
};
static const
struct processor_costs ppc750_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (17),   
COSTS_N_INSNS (17),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (17),   
COSTS_N_INSNS (31),   
32,			
32,			
512,			
1,			
0,			
};
static const
struct processor_costs ppc7450_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (23),   
COSTS_N_INSNS (23),   
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (21),   
COSTS_N_INSNS (35),   
32,			
32,			
1024,			
1,			
0,			
};
static const
struct processor_costs ppc8540_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (19),   
COSTS_N_INSNS (19),   
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (29),   
COSTS_N_INSNS (29),   
32,			
32,			
256,			
1,			
0,			
};
static const
struct processor_costs ppce300c2c3_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (19),   
COSTS_N_INSNS (19),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (33),   
32,
16,			
16,			
1,			
0,			
};
static const
struct processor_costs ppce500mc_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (14),   
COSTS_N_INSNS (14),   
COSTS_N_INSNS (8),    
COSTS_N_INSNS (10),   
COSTS_N_INSNS (36),   
COSTS_N_INSNS (66),   
64,			
32,			
128,			
1,			
0,			
};
static const
struct processor_costs ppce500mc64_cost = {
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (14),   
COSTS_N_INSNS (14),   
COSTS_N_INSNS (4),    
COSTS_N_INSNS (10),   
COSTS_N_INSNS (36),   
COSTS_N_INSNS (66),   
64,			
32,			
128,			
1,			
0,			
};
static const
struct processor_costs ppce5500_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (14),   
COSTS_N_INSNS (14),   
COSTS_N_INSNS (7),    
COSTS_N_INSNS (10),   
COSTS_N_INSNS (36),   
COSTS_N_INSNS (66),   
64,			
32,			
128,			
1,			
0,			
};
static const
struct processor_costs ppce6500_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (14),   
COSTS_N_INSNS (14),   
COSTS_N_INSNS (7),    
COSTS_N_INSNS (10),   
COSTS_N_INSNS (36),   
COSTS_N_INSNS (66),   
64,			
32,			
128,			
1,			
0,			
};
static const
struct processor_costs titan_cost = {
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (5),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (18),   
COSTS_N_INSNS (10),   
COSTS_N_INSNS (10),   
COSTS_N_INSNS (46),   
COSTS_N_INSNS (72),   
32,			
32,			
512,			
1,			
0,			
};
static const
struct processor_costs power4_cost = {
COSTS_N_INSNS (3),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (2),    
COSTS_N_INSNS (4),    
COSTS_N_INSNS (18),   
COSTS_N_INSNS (34),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (17),   
COSTS_N_INSNS (17),   
128,			
32,			
1024,			
8,			
0,			
};
static const
struct processor_costs power6_cost = {
COSTS_N_INSNS (8),    
COSTS_N_INSNS (8),    
COSTS_N_INSNS (8),    
COSTS_N_INSNS (8),    
COSTS_N_INSNS (22),   
COSTS_N_INSNS (28),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (13),   
COSTS_N_INSNS (16),   
128,			
64,			
2048,			
16,			
0,			
};
static const
struct processor_costs power7_cost = {
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (18),	
COSTS_N_INSNS (34),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (13),	
COSTS_N_INSNS (16),	
128,			
32,			
256,			
12,			
COSTS_N_INSNS (3),	
};
static const
struct processor_costs power8_cost = {
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (19),	
COSTS_N_INSNS (35),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (14),	
COSTS_N_INSNS (17),	
128,			
32,			
256,			
12,			
COSTS_N_INSNS (3),	
};
static const
struct processor_costs power9_cost = {
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (12),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (13),	
COSTS_N_INSNS (18),	
128,			
32,			
512,			
8,			
COSTS_N_INSNS (3),	
};
static const
struct processor_costs ppca2_cost = {
COSTS_N_INSNS (16),    
COSTS_N_INSNS (16),    
COSTS_N_INSNS (16),    
COSTS_N_INSNS (16),   
COSTS_N_INSNS (22),   
COSTS_N_INSNS (28),   
COSTS_N_INSNS (3),    
COSTS_N_INSNS (3),    
COSTS_N_INSNS (59),   
COSTS_N_INSNS (72),   
64,
16,			
2048,			
16,			
0,			
};

#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE) \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE) \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)  \
{ NAME, ICODE, MASK, ATTR },
struct rs6000_builtin_info_type {
const char *name;
const enum insn_code icode;
const HOST_WIDE_INT mask;
const unsigned attr;
};
static const struct rs6000_builtin_info_type rs6000_builtin_info[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
static tree (*rs6000_veclib_handler) (combined_fn, tree, tree);

static bool rs6000_debug_legitimate_address_p (machine_mode, rtx, bool);
static struct machine_function * rs6000_init_machine_status (void);
static int rs6000_ra_ever_killed (void);
static tree rs6000_handle_longcall_attribute (tree *, tree, tree, int, bool *);
static tree rs6000_handle_altivec_attribute (tree *, tree, tree, int, bool *);
static tree rs6000_handle_struct_attribute (tree *, tree, tree, int, bool *);
static tree rs6000_builtin_vectorized_libmass (combined_fn, tree, tree);
static void rs6000_emit_set_long_const (rtx, HOST_WIDE_INT);
static int rs6000_memory_move_cost (machine_mode, reg_class_t, bool);
static bool rs6000_debug_rtx_costs (rtx, machine_mode, int, int, int *, bool);
static int rs6000_debug_address_cost (rtx, machine_mode, addr_space_t,
bool);
static int rs6000_debug_adjust_cost (rtx_insn *, int, rtx_insn *, int,
unsigned int);
static bool is_microcoded_insn (rtx_insn *);
static bool is_nonpipeline_insn (rtx_insn *);
static bool is_cracked_insn (rtx_insn *);
static bool is_load_insn (rtx, rtx *);
static bool is_store_insn (rtx, rtx *);
static bool set_to_load_agen (rtx_insn *,rtx_insn *);
static bool insn_terminates_group_p (rtx_insn *, enum group_termination);
static bool insn_must_be_first_in_group (rtx_insn *);
static bool insn_must_be_last_in_group (rtx_insn *);
static void altivec_init_builtins (void);
static tree builtin_function_type (machine_mode, machine_mode,
machine_mode, machine_mode,
enum rs6000_builtins, const char *name);
static void rs6000_common_init_builtins (void);
static void paired_init_builtins (void);
static rtx paired_expand_predicate_builtin (enum insn_code, tree, rtx);
static void htm_init_builtins (void);
static rs6000_stack_t *rs6000_stack_info (void);
static void is_altivec_return_reg (rtx, void *);
int easy_vector_constant (rtx, machine_mode);
static rtx rs6000_debug_legitimize_address (rtx, rtx, machine_mode);
static rtx rs6000_legitimize_tls_address (rtx, enum tls_model);
static rtx rs6000_darwin64_record_arg (CUMULATIVE_ARGS *, const_tree,
bool, bool);
#if TARGET_MACHO
static void macho_branch_islands (void);
#endif
static rtx rs6000_legitimize_reload_address (rtx, machine_mode, int, int,
int, int *);
static rtx rs6000_debug_legitimize_reload_address (rtx, machine_mode, int,
int, int, int *);
static bool rs6000_mode_dependent_address (const_rtx);
static bool rs6000_debug_mode_dependent_address (const_rtx);
static bool rs6000_offsettable_memref_p (rtx, machine_mode, bool);
static enum reg_class rs6000_secondary_reload_class (enum reg_class,
machine_mode, rtx);
static enum reg_class rs6000_debug_secondary_reload_class (enum reg_class,
machine_mode,
rtx);
static enum reg_class rs6000_preferred_reload_class (rtx, enum reg_class);
static enum reg_class rs6000_debug_preferred_reload_class (rtx,
enum reg_class);
static bool rs6000_debug_secondary_memory_needed (machine_mode,
reg_class_t,
reg_class_t);
static bool rs6000_debug_can_change_mode_class (machine_mode,
machine_mode,
reg_class_t);
static bool rs6000_save_toc_in_prologue_p (void);
static rtx rs6000_internal_arg_pointer (void);
rtx (*rs6000_legitimize_reload_address_ptr) (rtx, machine_mode, int, int,
int, int *)
= rs6000_legitimize_reload_address;
static bool (*rs6000_mode_dependent_address_ptr) (const_rtx)
= rs6000_mode_dependent_address;
enum reg_class (*rs6000_secondary_reload_class_ptr) (enum reg_class,
machine_mode, rtx)
= rs6000_secondary_reload_class;
enum reg_class (*rs6000_preferred_reload_class_ptr) (rtx, enum reg_class)
= rs6000_preferred_reload_class;
const int INSN_NOT_AVAILABLE = -1;
static void rs6000_print_isa_options (FILE *, int, const char *,
HOST_WIDE_INT);
static void rs6000_print_builtin_options (FILE *, int, const char *,
HOST_WIDE_INT);
static HOST_WIDE_INT rs6000_disable_incompatible_switches (void);
static enum rs6000_reg_type register_to_reg_type (rtx, bool *);
static bool rs6000_secondary_reload_move (enum rs6000_reg_type,
enum rs6000_reg_type,
machine_mode,
secondary_reload_info *,
bool);
rtl_opt_pass *make_pass_analyze_swaps (gcc::context*);
static bool rs6000_keep_leaf_when_profiled () __attribute__ ((unused));
static tree rs6000_fold_builtin (tree, int, tree *, bool);
struct GTY((for_user)) toc_hash_struct
{
rtx key;
machine_mode key_mode;
int labelno;
};
struct toc_hasher : ggc_ptr_hash<toc_hash_struct>
{
static hashval_t hash (toc_hash_struct *);
static bool equal (toc_hash_struct *, toc_hash_struct *);
};
static GTY (()) hash_table<toc_hasher> *toc_hash_table;
struct GTY((for_user)) builtin_hash_struct
{
tree type;
machine_mode mode[4];	
unsigned char uns_p[4];	
};
struct builtin_hasher : ggc_ptr_hash<builtin_hash_struct>
{
static hashval_t hash (builtin_hash_struct *);
static bool equal (builtin_hash_struct *, builtin_hash_struct *);
};
static GTY (()) hash_table<builtin_hasher> *builtin_hash_table;

char rs6000_reg_names[][8] =
{
"0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",
"8",  "9", "10", "11", "12", "13", "14", "15",
"16", "17", "18", "19", "20", "21", "22", "23",
"24", "25", "26", "27", "28", "29", "30", "31",
"0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",
"8",  "9", "10", "11", "12", "13", "14", "15",
"16", "17", "18", "19", "20", "21", "22", "23",
"24", "25", "26", "27", "28", "29", "30", "31",
"mq", "lr", "ctr","ap",
"0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",
"ca",
"0",  "1",  "2",  "3",  "4",  "5",  "6", "7",
"8",  "9",  "10", "11", "12", "13", "14", "15",
"16", "17", "18", "19", "20", "21", "22", "23",
"24", "25", "26", "27", "28", "29", "30", "31",
"vrsave", "vscr",
"sfp",
"tfhar", "tfiar", "texasr"
};
#ifdef TARGET_REGNAMES
static const char alt_reg_names[][8] =
{
"%r0",   "%r1",  "%r2",  "%r3",  "%r4",  "%r5",  "%r6",  "%r7",
"%r8",   "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",
"%r16",  "%r17", "%r18", "%r19", "%r20", "%r21", "%r22", "%r23",
"%r24",  "%r25", "%r26", "%r27", "%r28", "%r29", "%r30", "%r31",
"%f0",   "%f1",  "%f2",  "%f3",  "%f4",  "%f5",  "%f6",  "%f7",
"%f8",   "%f9", "%f10", "%f11", "%f12", "%f13", "%f14", "%f15",
"%f16",  "%f17", "%f18", "%f19", "%f20", "%f21", "%f22", "%f23",
"%f24",  "%f25", "%f26", "%f27", "%f28", "%f29", "%f30", "%f31",
"mq",    "lr",  "ctr",   "ap",
"%cr0",  "%cr1", "%cr2", "%cr3", "%cr4", "%cr5", "%cr6", "%cr7",
"ca",
"%v0",  "%v1",  "%v2",  "%v3",  "%v4",  "%v5",  "%v6", "%v7",
"%v8",  "%v9", "%v10", "%v11", "%v12", "%v13", "%v14", "%v15",
"%v16", "%v17", "%v18", "%v19", "%v20", "%v21", "%v22", "%v23",
"%v24", "%v25", "%v26", "%v27", "%v28", "%v29", "%v30", "%v31",
"vrsave", "vscr",
"sfp",
"tfhar", "tfiar", "texasr"
};
#endif
static const struct attribute_spec rs6000_attribute_table[] =
{
{ "altivec",   1, 1, false, true,  false, false,
rs6000_handle_altivec_attribute, NULL },
{ "longcall",  0, 0, false, true,  true,  false,
rs6000_handle_longcall_attribute, NULL },
{ "shortcall", 0, 0, false, true,  true,  false,
rs6000_handle_longcall_attribute, NULL },
{ "ms_struct", 0, 0, false, false, false, false,
rs6000_handle_struct_attribute, NULL },
{ "gcc_struct", 0, 0, false, false, false, false,
rs6000_handle_struct_attribute, NULL },
#ifdef SUBTARGET_ATTRIBUTE_TABLE
SUBTARGET_ATTRIBUTE_TABLE,
#endif
{ NULL,        0, 0, false, false, false, false, NULL, NULL }
};

#ifndef TARGET_PROFILE_KERNEL
#define TARGET_PROFILE_KERNEL 0
#endif
#define ALTIVEC_REG_BIT(REGNO) (0x80000000 >> ((REGNO) - FIRST_ALTIVEC_REGNO))

#undef TARGET_ATTRIBUTE_TABLE
#define TARGET_ATTRIBUTE_TABLE rs6000_attribute_table
#undef TARGET_SET_DEFAULT_TYPE_ATTRIBUTES
#define TARGET_SET_DEFAULT_TYPE_ATTRIBUTES rs6000_set_default_type_attributes
#undef TARGET_ATTRIBUTE_TAKES_IDENTIFIER_P
#define TARGET_ATTRIBUTE_TAKES_IDENTIFIER_P rs6000_attribute_takes_identifier_p
#undef TARGET_ASM_ALIGNED_DI_OP
#define TARGET_ASM_ALIGNED_DI_OP DOUBLE_INT_ASM_OP
#ifndef OBJECT_FORMAT_ELF
#if TARGET_XCOFF
#undef TARGET_ASM_UNALIGNED_HI_OP
#define TARGET_ASM_UNALIGNED_HI_OP "\t.vbyte\t2,"
#undef TARGET_ASM_UNALIGNED_SI_OP
#define TARGET_ASM_UNALIGNED_SI_OP "\t.vbyte\t4,"
#undef TARGET_ASM_UNALIGNED_DI_OP
#define TARGET_ASM_UNALIGNED_DI_OP "\t.vbyte\t8,"
#else
#undef TARGET_ASM_UNALIGNED_HI_OP
#define TARGET_ASM_UNALIGNED_HI_OP "\t.short\t"
#undef TARGET_ASM_UNALIGNED_SI_OP
#define TARGET_ASM_UNALIGNED_SI_OP "\t.long\t"
#undef TARGET_ASM_UNALIGNED_DI_OP
#define TARGET_ASM_UNALIGNED_DI_OP "\t.quad\t"
#undef TARGET_ASM_ALIGNED_DI_OP
#define TARGET_ASM_ALIGNED_DI_OP "\t.quad\t"
#endif
#endif
#undef TARGET_ASM_INTEGER
#define TARGET_ASM_INTEGER rs6000_assemble_integer
#if defined (HAVE_GAS_HIDDEN) && !TARGET_MACHO
#undef TARGET_ASM_ASSEMBLE_VISIBILITY
#define TARGET_ASM_ASSEMBLE_VISIBILITY rs6000_assemble_visibility
#endif
#undef TARGET_SET_UP_BY_PROLOGUE
#define TARGET_SET_UP_BY_PROLOGUE rs6000_set_up_by_prologue
#undef TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS
#define TARGET_SHRINK_WRAP_GET_SEPARATE_COMPONENTS rs6000_get_separate_components
#undef TARGET_SHRINK_WRAP_COMPONENTS_FOR_BB
#define TARGET_SHRINK_WRAP_COMPONENTS_FOR_BB rs6000_components_for_bb
#undef TARGET_SHRINK_WRAP_DISQUALIFY_COMPONENTS
#define TARGET_SHRINK_WRAP_DISQUALIFY_COMPONENTS rs6000_disqualify_components
#undef TARGET_SHRINK_WRAP_EMIT_PROLOGUE_COMPONENTS
#define TARGET_SHRINK_WRAP_EMIT_PROLOGUE_COMPONENTS rs6000_emit_prologue_components
#undef TARGET_SHRINK_WRAP_EMIT_EPILOGUE_COMPONENTS
#define TARGET_SHRINK_WRAP_EMIT_EPILOGUE_COMPONENTS rs6000_emit_epilogue_components
#undef TARGET_SHRINK_WRAP_SET_HANDLED_COMPONENTS
#define TARGET_SHRINK_WRAP_SET_HANDLED_COMPONENTS rs6000_set_handled_components
#undef TARGET_EXTRA_LIVE_ON_ENTRY
#define TARGET_EXTRA_LIVE_ON_ENTRY rs6000_live_on_entry
#undef TARGET_INTERNAL_ARG_POINTER
#define TARGET_INTERNAL_ARG_POINTER rs6000_internal_arg_pointer
#undef TARGET_HAVE_TLS
#define TARGET_HAVE_TLS HAVE_AS_TLS
#undef TARGET_CANNOT_FORCE_CONST_MEM
#define TARGET_CANNOT_FORCE_CONST_MEM rs6000_cannot_force_const_mem
#undef TARGET_DELEGITIMIZE_ADDRESS
#define TARGET_DELEGITIMIZE_ADDRESS rs6000_delegitimize_address
#undef TARGET_CONST_NOT_OK_FOR_DEBUG_P
#define TARGET_CONST_NOT_OK_FOR_DEBUG_P rs6000_const_not_ok_for_debug_p
#undef TARGET_LEGITIMATE_COMBINED_INSN
#define TARGET_LEGITIMATE_COMBINED_INSN rs6000_legitimate_combined_insn
#undef TARGET_ASM_FUNCTION_PROLOGUE
#define TARGET_ASM_FUNCTION_PROLOGUE rs6000_output_function_prologue
#undef TARGET_ASM_FUNCTION_EPILOGUE
#define TARGET_ASM_FUNCTION_EPILOGUE rs6000_output_function_epilogue
#undef TARGET_ASM_OUTPUT_ADDR_CONST_EXTRA
#define TARGET_ASM_OUTPUT_ADDR_CONST_EXTRA rs6000_output_addr_const_extra
#undef TARGET_LEGITIMIZE_ADDRESS
#define TARGET_LEGITIMIZE_ADDRESS rs6000_legitimize_address
#undef  TARGET_SCHED_VARIABLE_ISSUE
#define TARGET_SCHED_VARIABLE_ISSUE rs6000_variable_issue
#undef TARGET_SCHED_ISSUE_RATE
#define TARGET_SCHED_ISSUE_RATE rs6000_issue_rate
#undef TARGET_SCHED_ADJUST_COST
#define TARGET_SCHED_ADJUST_COST rs6000_adjust_cost
#undef TARGET_SCHED_ADJUST_PRIORITY
#define TARGET_SCHED_ADJUST_PRIORITY rs6000_adjust_priority
#undef TARGET_SCHED_IS_COSTLY_DEPENDENCE
#define TARGET_SCHED_IS_COSTLY_DEPENDENCE rs6000_is_costly_dependence
#undef TARGET_SCHED_INIT
#define TARGET_SCHED_INIT rs6000_sched_init
#undef TARGET_SCHED_FINISH
#define TARGET_SCHED_FINISH rs6000_sched_finish
#undef TARGET_SCHED_REORDER
#define TARGET_SCHED_REORDER rs6000_sched_reorder
#undef TARGET_SCHED_REORDER2
#define TARGET_SCHED_REORDER2 rs6000_sched_reorder2
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD rs6000_use_sched_lookahead
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD rs6000_use_sched_lookahead_guard
#undef TARGET_SCHED_ALLOC_SCHED_CONTEXT
#define TARGET_SCHED_ALLOC_SCHED_CONTEXT rs6000_alloc_sched_context
#undef TARGET_SCHED_INIT_SCHED_CONTEXT
#define TARGET_SCHED_INIT_SCHED_CONTEXT rs6000_init_sched_context
#undef TARGET_SCHED_SET_SCHED_CONTEXT
#define TARGET_SCHED_SET_SCHED_CONTEXT rs6000_set_sched_context
#undef TARGET_SCHED_FREE_SCHED_CONTEXT
#define TARGET_SCHED_FREE_SCHED_CONTEXT rs6000_free_sched_context
#undef TARGET_SCHED_CAN_SPECULATE_INSN
#define TARGET_SCHED_CAN_SPECULATE_INSN rs6000_sched_can_speculate_insn
#undef TARGET_VECTORIZE_BUILTIN_MASK_FOR_LOAD
#define TARGET_VECTORIZE_BUILTIN_MASK_FOR_LOAD rs6000_builtin_mask_for_load
#undef TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT
#define TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT		\
rs6000_builtin_support_vector_misalignment
#undef TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE
#define TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE rs6000_vector_alignment_reachable
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST
#define TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST \
rs6000_builtin_vectorization_cost
#undef TARGET_VECTORIZE_PREFERRED_SIMD_MODE
#define TARGET_VECTORIZE_PREFERRED_SIMD_MODE \
rs6000_preferred_simd_mode
#undef TARGET_VECTORIZE_INIT_COST
#define TARGET_VECTORIZE_INIT_COST rs6000_init_cost
#undef TARGET_VECTORIZE_ADD_STMT_COST
#define TARGET_VECTORIZE_ADD_STMT_COST rs6000_add_stmt_cost
#undef TARGET_VECTORIZE_FINISH_COST
#define TARGET_VECTORIZE_FINISH_COST rs6000_finish_cost
#undef TARGET_VECTORIZE_DESTROY_COST_DATA
#define TARGET_VECTORIZE_DESTROY_COST_DATA rs6000_destroy_cost_data
#undef TARGET_INIT_BUILTINS
#define TARGET_INIT_BUILTINS rs6000_init_builtins
#undef TARGET_BUILTIN_DECL
#define TARGET_BUILTIN_DECL rs6000_builtin_decl
#undef TARGET_FOLD_BUILTIN
#define TARGET_FOLD_BUILTIN rs6000_fold_builtin
#undef TARGET_GIMPLE_FOLD_BUILTIN
#define TARGET_GIMPLE_FOLD_BUILTIN rs6000_gimple_fold_builtin
#undef TARGET_EXPAND_BUILTIN
#define TARGET_EXPAND_BUILTIN rs6000_expand_builtin
#undef TARGET_MANGLE_TYPE
#define TARGET_MANGLE_TYPE rs6000_mangle_type
#undef TARGET_INIT_LIBFUNCS
#define TARGET_INIT_LIBFUNCS rs6000_init_libfuncs
#if TARGET_MACHO
#undef TARGET_BINDS_LOCAL_P
#define TARGET_BINDS_LOCAL_P darwin_binds_local_p
#endif
#undef TARGET_MS_BITFIELD_LAYOUT_P
#define TARGET_MS_BITFIELD_LAYOUT_P rs6000_ms_bitfield_layout_p
#undef TARGET_ASM_OUTPUT_MI_THUNK
#define TARGET_ASM_OUTPUT_MI_THUNK rs6000_output_mi_thunk
#undef TARGET_ASM_CAN_OUTPUT_MI_THUNK
#define TARGET_ASM_CAN_OUTPUT_MI_THUNK hook_bool_const_tree_hwi_hwi_const_tree_true
#undef TARGET_FUNCTION_OK_FOR_SIBCALL
#define TARGET_FUNCTION_OK_FOR_SIBCALL rs6000_function_ok_for_sibcall
#undef TARGET_REGISTER_MOVE_COST
#define TARGET_REGISTER_MOVE_COST rs6000_register_move_cost
#undef TARGET_MEMORY_MOVE_COST
#define TARGET_MEMORY_MOVE_COST rs6000_memory_move_cost
#undef TARGET_CANNOT_COPY_INSN_P
#define TARGET_CANNOT_COPY_INSN_P rs6000_cannot_copy_insn_p
#undef TARGET_RTX_COSTS
#define TARGET_RTX_COSTS rs6000_rtx_costs
#undef TARGET_ADDRESS_COST
#define TARGET_ADDRESS_COST hook_int_rtx_mode_as_bool_0
#undef TARGET_INSN_COST
#define TARGET_INSN_COST rs6000_insn_cost
#undef TARGET_INIT_DWARF_REG_SIZES_EXTRA
#define TARGET_INIT_DWARF_REG_SIZES_EXTRA rs6000_init_dwarf_reg_sizes_extra
#undef TARGET_PROMOTE_FUNCTION_MODE
#define TARGET_PROMOTE_FUNCTION_MODE rs6000_promote_function_mode
#undef TARGET_RETURN_IN_MEMORY
#define TARGET_RETURN_IN_MEMORY rs6000_return_in_memory
#undef TARGET_RETURN_IN_MSB
#define TARGET_RETURN_IN_MSB rs6000_return_in_msb
#undef TARGET_SETUP_INCOMING_VARARGS
#define TARGET_SETUP_INCOMING_VARARGS setup_incoming_varargs
#undef TARGET_STRICT_ARGUMENT_NAMING
#define TARGET_STRICT_ARGUMENT_NAMING hook_bool_CUMULATIVE_ARGS_true
#undef TARGET_PRETEND_OUTGOING_VARARGS_NAMED
#define TARGET_PRETEND_OUTGOING_VARARGS_NAMED hook_bool_CUMULATIVE_ARGS_true
#undef TARGET_SPLIT_COMPLEX_ARG
#define TARGET_SPLIT_COMPLEX_ARG hook_bool_const_tree_true
#undef TARGET_MUST_PASS_IN_STACK
#define TARGET_MUST_PASS_IN_STACK rs6000_must_pass_in_stack
#undef TARGET_PASS_BY_REFERENCE
#define TARGET_PASS_BY_REFERENCE rs6000_pass_by_reference
#undef TARGET_ARG_PARTIAL_BYTES
#define TARGET_ARG_PARTIAL_BYTES rs6000_arg_partial_bytes
#undef TARGET_FUNCTION_ARG_ADVANCE
#define TARGET_FUNCTION_ARG_ADVANCE rs6000_function_arg_advance
#undef TARGET_FUNCTION_ARG
#define TARGET_FUNCTION_ARG rs6000_function_arg
#undef TARGET_FUNCTION_ARG_PADDING
#define TARGET_FUNCTION_ARG_PADDING rs6000_function_arg_padding
#undef TARGET_FUNCTION_ARG_BOUNDARY
#define TARGET_FUNCTION_ARG_BOUNDARY rs6000_function_arg_boundary
#undef TARGET_BUILD_BUILTIN_VA_LIST
#define TARGET_BUILD_BUILTIN_VA_LIST rs6000_build_builtin_va_list
#undef TARGET_EXPAND_BUILTIN_VA_START
#define TARGET_EXPAND_BUILTIN_VA_START rs6000_va_start
#undef TARGET_GIMPLIFY_VA_ARG_EXPR
#define TARGET_GIMPLIFY_VA_ARG_EXPR rs6000_gimplify_va_arg
#undef TARGET_EH_RETURN_FILTER_MODE
#define TARGET_EH_RETURN_FILTER_MODE rs6000_eh_return_filter_mode
#undef TARGET_SCALAR_MODE_SUPPORTED_P
#define TARGET_SCALAR_MODE_SUPPORTED_P rs6000_scalar_mode_supported_p
#undef TARGET_VECTOR_MODE_SUPPORTED_P
#define TARGET_VECTOR_MODE_SUPPORTED_P rs6000_vector_mode_supported_p
#undef TARGET_FLOATN_MODE
#define TARGET_FLOATN_MODE rs6000_floatn_mode
#undef TARGET_INVALID_ARG_FOR_UNPROTOTYPED_FN
#define TARGET_INVALID_ARG_FOR_UNPROTOTYPED_FN invalid_arg_for_unprototyped_fn
#undef TARGET_ASM_LOOP_ALIGN_MAX_SKIP
#define TARGET_ASM_LOOP_ALIGN_MAX_SKIP rs6000_loop_align_max_skip
#undef TARGET_MD_ASM_ADJUST
#define TARGET_MD_ASM_ADJUST rs6000_md_asm_adjust
#undef TARGET_OPTION_OVERRIDE
#define TARGET_OPTION_OVERRIDE rs6000_option_override
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION
#define TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION \
rs6000_builtin_vectorized_function
#undef TARGET_VECTORIZE_BUILTIN_MD_VECTORIZED_FUNCTION
#define TARGET_VECTORIZE_BUILTIN_MD_VECTORIZED_FUNCTION \
rs6000_builtin_md_vectorized_function
#undef TARGET_STACK_PROTECT_GUARD
#define TARGET_STACK_PROTECT_GUARD rs6000_init_stack_protect_guard
#if !TARGET_MACHO
#undef TARGET_STACK_PROTECT_FAIL
#define TARGET_STACK_PROTECT_FAIL rs6000_stack_protect_fail
#endif
#ifdef HAVE_AS_TLS
#undef TARGET_ASM_OUTPUT_DWARF_DTPREL
#define TARGET_ASM_OUTPUT_DWARF_DTPREL rs6000_output_dwarf_dtprel
#endif
#undef TARGET_MIN_ANCHOR_OFFSET
#define TARGET_MIN_ANCHOR_OFFSET -0x7fffffff - 1
#undef TARGET_MAX_ANCHOR_OFFSET
#define TARGET_MAX_ANCHOR_OFFSET 0x7fffffff
#undef TARGET_USE_BLOCKS_FOR_CONSTANT_P
#define TARGET_USE_BLOCKS_FOR_CONSTANT_P rs6000_use_blocks_for_constant_p
#undef TARGET_USE_BLOCKS_FOR_DECL_P
#define TARGET_USE_BLOCKS_FOR_DECL_P rs6000_use_blocks_for_decl_p
#undef TARGET_BUILTIN_RECIPROCAL
#define TARGET_BUILTIN_RECIPROCAL rs6000_builtin_reciprocal
#undef TARGET_SECONDARY_RELOAD
#define TARGET_SECONDARY_RELOAD rs6000_secondary_reload
#undef TARGET_SECONDARY_MEMORY_NEEDED
#define TARGET_SECONDARY_MEMORY_NEEDED rs6000_secondary_memory_needed
#undef TARGET_SECONDARY_MEMORY_NEEDED_MODE
#define TARGET_SECONDARY_MEMORY_NEEDED_MODE rs6000_secondary_memory_needed_mode
#undef TARGET_LEGITIMATE_ADDRESS_P
#define TARGET_LEGITIMATE_ADDRESS_P rs6000_legitimate_address_p
#undef TARGET_MODE_DEPENDENT_ADDRESS_P
#define TARGET_MODE_DEPENDENT_ADDRESS_P rs6000_mode_dependent_address_p
#undef TARGET_COMPUTE_PRESSURE_CLASSES
#define TARGET_COMPUTE_PRESSURE_CLASSES rs6000_compute_pressure_classes
#undef TARGET_CAN_ELIMINATE
#define TARGET_CAN_ELIMINATE rs6000_can_eliminate
#undef TARGET_CONDITIONAL_REGISTER_USAGE
#define TARGET_CONDITIONAL_REGISTER_USAGE rs6000_conditional_register_usage
#undef TARGET_SCHED_REASSOCIATION_WIDTH
#define TARGET_SCHED_REASSOCIATION_WIDTH rs6000_reassociation_width
#undef TARGET_TRAMPOLINE_INIT
#define TARGET_TRAMPOLINE_INIT rs6000_trampoline_init
#undef TARGET_FUNCTION_VALUE
#define TARGET_FUNCTION_VALUE rs6000_function_value
#undef TARGET_OPTION_VALID_ATTRIBUTE_P
#define TARGET_OPTION_VALID_ATTRIBUTE_P rs6000_valid_attribute_p
#undef TARGET_OPTION_SAVE
#define TARGET_OPTION_SAVE rs6000_function_specific_save
#undef TARGET_OPTION_RESTORE
#define TARGET_OPTION_RESTORE rs6000_function_specific_restore
#undef TARGET_OPTION_PRINT
#define TARGET_OPTION_PRINT rs6000_function_specific_print
#undef TARGET_CAN_INLINE_P
#define TARGET_CAN_INLINE_P rs6000_can_inline_p
#undef TARGET_SET_CURRENT_FUNCTION
#define TARGET_SET_CURRENT_FUNCTION rs6000_set_current_function
#undef TARGET_LEGITIMATE_CONSTANT_P
#define TARGET_LEGITIMATE_CONSTANT_P rs6000_legitimate_constant_p
#undef TARGET_VECTORIZE_VEC_PERM_CONST
#define TARGET_VECTORIZE_VEC_PERM_CONST rs6000_vectorize_vec_perm_const
#undef TARGET_CAN_USE_DOLOOP_P
#define TARGET_CAN_USE_DOLOOP_P can_use_doloop_if_innermost
#undef TARGET_ATOMIC_ASSIGN_EXPAND_FENV
#define TARGET_ATOMIC_ASSIGN_EXPAND_FENV rs6000_atomic_assign_expand_fenv
#undef TARGET_LIBGCC_CMP_RETURN_MODE
#define TARGET_LIBGCC_CMP_RETURN_MODE rs6000_abi_word_mode
#undef TARGET_LIBGCC_SHIFT_COUNT_MODE
#define TARGET_LIBGCC_SHIFT_COUNT_MODE rs6000_abi_word_mode
#undef TARGET_UNWIND_WORD_MODE
#define TARGET_UNWIND_WORD_MODE rs6000_abi_word_mode
#undef TARGET_OFFLOAD_OPTIONS
#define TARGET_OFFLOAD_OPTIONS rs6000_offload_options
#undef TARGET_C_MODE_FOR_SUFFIX
#define TARGET_C_MODE_FOR_SUFFIX rs6000_c_mode_for_suffix
#undef TARGET_INVALID_BINARY_OP
#define TARGET_INVALID_BINARY_OP rs6000_invalid_binary_op
#undef TARGET_OPTAB_SUPPORTED_P
#define TARGET_OPTAB_SUPPORTED_P rs6000_optab_supported_p
#undef TARGET_CUSTOM_FUNCTION_DESCRIPTORS
#define TARGET_CUSTOM_FUNCTION_DESCRIPTORS 1
#undef TARGET_COMPARE_VERSION_PRIORITY
#define TARGET_COMPARE_VERSION_PRIORITY rs6000_compare_version_priority
#undef TARGET_GENERATE_VERSION_DISPATCHER_BODY
#define TARGET_GENERATE_VERSION_DISPATCHER_BODY				\
rs6000_generate_version_dispatcher_body
#undef TARGET_GET_FUNCTION_VERSIONS_DISPATCHER
#define TARGET_GET_FUNCTION_VERSIONS_DISPATCHER				\
rs6000_get_function_versions_dispatcher
#undef TARGET_OPTION_FUNCTION_VERSIONS
#define TARGET_OPTION_FUNCTION_VERSIONS common_function_versions
#undef TARGET_HARD_REGNO_NREGS
#define TARGET_HARD_REGNO_NREGS rs6000_hard_regno_nregs_hook
#undef TARGET_HARD_REGNO_MODE_OK
#define TARGET_HARD_REGNO_MODE_OK rs6000_hard_regno_mode_ok
#undef TARGET_MODES_TIEABLE_P
#define TARGET_MODES_TIEABLE_P rs6000_modes_tieable_p
#undef TARGET_HARD_REGNO_CALL_PART_CLOBBERED
#define TARGET_HARD_REGNO_CALL_PART_CLOBBERED \
rs6000_hard_regno_call_part_clobbered
#undef TARGET_SLOW_UNALIGNED_ACCESS
#define TARGET_SLOW_UNALIGNED_ACCESS rs6000_slow_unaligned_access
#undef TARGET_CAN_CHANGE_MODE_CLASS
#define TARGET_CAN_CHANGE_MODE_CLASS rs6000_can_change_mode_class
#undef TARGET_CONSTANT_ALIGNMENT
#define TARGET_CONSTANT_ALIGNMENT rs6000_constant_alignment
#undef TARGET_STARTING_FRAME_OFFSET
#define TARGET_STARTING_FRAME_OFFSET rs6000_starting_frame_offset
#if TARGET_ELF && RS6000_WEAK
#undef TARGET_ASM_GLOBALIZE_DECL_NAME
#define TARGET_ASM_GLOBALIZE_DECL_NAME rs6000_globalize_decl_name
#endif

struct rs6000_ptt
{
const char *const name;		
const enum processor_type processor;	
const HOST_WIDE_INT target_enable;	
};
static struct rs6000_ptt const processor_target_table[] =
{
#define RS6000_CPU(NAME, CPU, FLAGS) { NAME, CPU, FLAGS },
#include "rs6000-cpus.def"
#undef RS6000_CPU
};
static int
rs6000_cpu_name_lookup (const char *name)
{
size_t i;
if (name != NULL)
{
for (i = 0; i < ARRAY_SIZE (processor_target_table); i++)
if (! strcmp (name, processor_target_table[i].name))
return (int)i;
}
return -1;
}

static int
rs6000_hard_regno_nregs_internal (int regno, machine_mode mode)
{
unsigned HOST_WIDE_INT reg_size;
if (FP_REGNO_P (regno))
reg_size = (VECTOR_MEM_VSX_P (mode) || FLOAT128_VECTOR_P (mode)
? UNITS_PER_VSX_WORD
: UNITS_PER_FP_WORD);
else if (ALTIVEC_REGNO_P (regno))
reg_size = UNITS_PER_ALTIVEC_WORD;
else
reg_size = UNITS_PER_WORD;
return (GET_MODE_SIZE (mode) + reg_size - 1) / reg_size;
}
static int
rs6000_hard_regno_mode_ok_uncached (int regno, machine_mode mode)
{
int last_regno = regno + rs6000_hard_regno_nregs[mode][regno] - 1;
if (COMPLEX_MODE_P (mode))
mode = GET_MODE_INNER (mode);
if (mode == PTImode)
return (IN_RANGE (regno, FIRST_GPR_REGNO, LAST_GPR_REGNO)
&& IN_RANGE (last_regno, FIRST_GPR_REGNO, LAST_GPR_REGNO)
&& ((regno & 1) == 0));
if (TARGET_VSX && VSX_REGNO_P (regno)
&& (VECTOR_MEM_VSX_P (mode)
|| FLOAT128_VECTOR_P (mode)
|| reg_addr[mode].scalar_in_vmx_p
|| mode == TImode
|| (TARGET_VADDUQM && mode == V1TImode)))
{
if (FP_REGNO_P (regno))
return FP_REGNO_P (last_regno);
if (ALTIVEC_REGNO_P (regno))
{
if (GET_MODE_SIZE (mode) != 16 && !reg_addr[mode].scalar_in_vmx_p)
return 0;
return ALTIVEC_REGNO_P (last_regno);
}
}
if (INT_REGNO_P (regno))
return INT_REGNO_P (last_regno);
if (FP_REGNO_P (regno))
{
if (FLOAT128_VECTOR_P (mode))
return false;
if (SCALAR_FLOAT_MODE_P (mode)
&& (mode != TDmode || (regno % 2) == 0)
&& FP_REGNO_P (last_regno))
return 1;
if (GET_MODE_CLASS (mode) == MODE_INT)
{
if(GET_MODE_SIZE (mode) == UNITS_PER_FP_WORD)
return 1;
if (TARGET_P8_VECTOR && (mode == SImode))
return 1;
if (TARGET_P9_VECTOR && (mode == QImode || mode == HImode))
return 1;
}
if (PAIRED_SIMD_REGNO_P (regno) && TARGET_PAIRED_FLOAT
&& PAIRED_VECTOR_MODE (mode))
return 1;
return 0;
}
if (CR_REGNO_P (regno))
return GET_MODE_CLASS (mode) == MODE_CC;
if (CA_REGNO_P (regno))
return mode == Pmode || mode == SImode;
if (ALTIVEC_REGNO_P (regno))
return (VECTOR_MEM_ALTIVEC_OR_VSX_P (mode)
|| mode == V1TImode);
return GET_MODE_SIZE (mode) <= UNITS_PER_WORD;
}
static unsigned int
rs6000_hard_regno_nregs_hook (unsigned int regno, machine_mode mode)
{
return rs6000_hard_regno_nregs[mode][regno];
}
static bool
rs6000_hard_regno_mode_ok (unsigned int regno, machine_mode mode)
{
return rs6000_hard_regno_mode_ok_p[mode][regno];
}
static bool
rs6000_modes_tieable_p (machine_mode mode1, machine_mode mode2)
{
if (mode1 == PTImode)
return mode2 == PTImode;
if (mode2 == PTImode)
return false;
if (ALTIVEC_OR_VSX_VECTOR_MODE (mode1))
return ALTIVEC_OR_VSX_VECTOR_MODE (mode2);
if (ALTIVEC_OR_VSX_VECTOR_MODE (mode2))
return false;
if (SCALAR_FLOAT_MODE_P (mode1))
return SCALAR_FLOAT_MODE_P (mode2);
if (SCALAR_FLOAT_MODE_P (mode2))
return false;
if (GET_MODE_CLASS (mode1) == MODE_CC)
return GET_MODE_CLASS (mode2) == MODE_CC;
if (GET_MODE_CLASS (mode2) == MODE_CC)
return false;
if (PAIRED_VECTOR_MODE (mode1))
return PAIRED_VECTOR_MODE (mode2);
if (PAIRED_VECTOR_MODE (mode2))
return false;
return true;
}
static bool
rs6000_hard_regno_call_part_clobbered (unsigned int regno, machine_mode mode)
{
if (TARGET_32BIT
&& TARGET_POWERPC64
&& GET_MODE_SIZE (mode) > 4
&& INT_REGNO_P (regno))
return true;
if (TARGET_VSX
&& FP_REGNO_P (regno)
&& GET_MODE_SIZE (mode) > 8
&& !FLOAT128_2REG_P (mode))
return true;
return false;
}
static void
rs6000_debug_reg_print (int first_regno, int last_regno, const char *reg_name)
{
int r, m;
for (r = first_regno; r <= last_regno; ++r)
{
const char *comma = "";
int len;
if (first_regno == last_regno)
fprintf (stderr, "%s:\t", reg_name);
else
fprintf (stderr, "%s%d:\t", reg_name, r - first_regno);
len = 8;
for (m = 0; m < NUM_MACHINE_MODES; ++m)
if (rs6000_hard_regno_mode_ok_p[m][r] && rs6000_hard_regno_nregs[m][r])
{
if (len > 70)
{
fprintf (stderr, ",\n\t");
len = 8;
comma = "";
}
if (rs6000_hard_regno_nregs[m][r] > 1)
len += fprintf (stderr, "%s%s/%d", comma, GET_MODE_NAME (m),
rs6000_hard_regno_nregs[m][r]);
else
len += fprintf (stderr, "%s%s", comma, GET_MODE_NAME (m));
comma = ", ";
}
if (call_used_regs[r])
{
if (len > 70)
{
fprintf (stderr, ",\n\t");
len = 8;
comma = "";
}
len += fprintf (stderr, "%s%s", comma, "call-used");
comma = ", ";
}
if (fixed_regs[r])
{
if (len > 70)
{
fprintf (stderr, ",\n\t");
len = 8;
comma = "";
}
len += fprintf (stderr, "%s%s", comma, "fixed");
comma = ", ";
}
if (len > 70)
{
fprintf (stderr, ",\n\t");
comma = "";
}
len += fprintf (stderr, "%sreg-class = %s", comma,
reg_class_names[(int)rs6000_regno_regclass[r]]);
comma = ", ";
if (len > 70)
{
fprintf (stderr, ",\n\t");
comma = "";
}
fprintf (stderr, "%sregno = %d\n", comma, r);
}
}
static const char *
rs6000_debug_vector_unit (enum rs6000_vector v)
{
const char *ret;
switch (v)
{
case VECTOR_NONE:	   ret = "none";      break;
case VECTOR_ALTIVEC:   ret = "altivec";   break;
case VECTOR_VSX:	   ret = "vsx";       break;
case VECTOR_P8_VECTOR: ret = "p8_vector"; break;
case VECTOR_PAIRED:	   ret = "paired";    break;
case VECTOR_OTHER:	   ret = "other";     break;
default:		   ret = "unknown";   break;
}
return ret;
}
DEBUG_FUNCTION char *
rs6000_debug_addr_mask (addr_mask_type mask, bool keep_spaces)
{
static char ret[8];
char *p = ret;
if ((mask & RELOAD_REG_VALID) != 0)
*p++ = 'v';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_MULTIPLE) != 0)
*p++ = 'm';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_INDEXED) != 0)
*p++ = 'i';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_QUAD_OFFSET) != 0)
*p++ = 'O';
else if ((mask & RELOAD_REG_OFFSET) != 0)
*p++ = 'o';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_PRE_INCDEC) != 0)
*p++ = '+';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_PRE_MODIFY) != 0)
*p++ = '+';
else if (keep_spaces)
*p++ = ' ';
if ((mask & RELOAD_REG_AND_M16) != 0)
*p++ = '&';
else if (keep_spaces)
*p++ = ' ';
*p = '\0';
return ret;
}
DEBUG_FUNCTION void
rs6000_debug_print_mode (ssize_t m)
{
ssize_t rc;
int spaces = 0;
bool fuse_extra_p;
fprintf (stderr, "Mode: %-5s", GET_MODE_NAME (m));
for (rc = 0; rc < N_RELOAD_REG; rc++)
fprintf (stderr, " %s: %s", reload_reg_map[rc].name,
rs6000_debug_addr_mask (reg_addr[m].addr_mask[rc], true));
if ((reg_addr[m].reload_store != CODE_FOR_nothing)
|| (reg_addr[m].reload_load != CODE_FOR_nothing))
fprintf (stderr, "  Reload=%c%c",
(reg_addr[m].reload_store != CODE_FOR_nothing) ? 's' : '*',
(reg_addr[m].reload_load != CODE_FOR_nothing) ? 'l' : '*');
else
spaces += sizeof ("  Reload=sl") - 1;
if (reg_addr[m].scalar_in_vmx_p)
{
fprintf (stderr, "%*s  Upper=y", spaces, "");
spaces = 0;
}
else
spaces += sizeof ("  Upper=y") - 1;
fuse_extra_p = ((reg_addr[m].fusion_gpr_ld != CODE_FOR_nothing)
|| reg_addr[m].fused_toc);
if (!fuse_extra_p)
{
for (rc = 0; rc < N_RELOAD_REG; rc++)
{
if (rc != RELOAD_REG_ANY)
{
if (reg_addr[m].fusion_addi_ld[rc]     != CODE_FOR_nothing
|| reg_addr[m].fusion_addi_ld[rc]  != CODE_FOR_nothing
|| reg_addr[m].fusion_addi_st[rc]  != CODE_FOR_nothing
|| reg_addr[m].fusion_addis_ld[rc] != CODE_FOR_nothing
|| reg_addr[m].fusion_addis_st[rc] != CODE_FOR_nothing)
{
fuse_extra_p = true;
break;
}
}
}
}
if (fuse_extra_p)
{
fprintf (stderr, "%*s  Fuse:", spaces, "");
spaces = 0;
for (rc = 0; rc < N_RELOAD_REG; rc++)
{
if (rc != RELOAD_REG_ANY)
{
char load, store;
if (reg_addr[m].fusion_addis_ld[rc] != CODE_FOR_nothing)
load = 'l';
else if (reg_addr[m].fusion_addi_ld[rc] != CODE_FOR_nothing)
load = 'L';
else
load = '-';
if (reg_addr[m].fusion_addis_st[rc] != CODE_FOR_nothing)
store = 's';
else if (reg_addr[m].fusion_addi_st[rc] != CODE_FOR_nothing)
store = 'S';
else
store = '-';
if (load == '-' && store == '-')
spaces += 5;
else
{
fprintf (stderr, "%*s%c=%c%c", (spaces + 1), "",
reload_reg_map[rc].name[0], load, store);
spaces = 0;
}
}
}
if (reg_addr[m].fusion_gpr_ld != CODE_FOR_nothing)
{
fprintf (stderr, "%*sP8gpr", (spaces + 1), "");
spaces = 0;
}
else
spaces += sizeof (" P8gpr") - 1;
if (reg_addr[m].fused_toc)
{
fprintf (stderr, "%*sToc", (spaces + 1), "");
spaces = 0;
}
else
spaces += sizeof (" Toc") - 1;
}
else
spaces += sizeof ("  Fuse: G=ls F=ls v=ls P8gpr Toc") - 1;
if (rs6000_vector_unit[m] != VECTOR_NONE
|| rs6000_vector_mem[m] != VECTOR_NONE)
{
fprintf (stderr, "%*s  vector: arith=%-10s mem=%s",
spaces, "",
rs6000_debug_vector_unit (rs6000_vector_unit[m]),
rs6000_debug_vector_unit (rs6000_vector_mem[m]));
}
fputs ("\n", stderr);
}
#define DEBUG_FMT_ID "%-32s= "
#define DEBUG_FMT_D   DEBUG_FMT_ID "%d\n"
#define DEBUG_FMT_WX  DEBUG_FMT_ID "%#.12" HOST_WIDE_INT_PRINT "x: "
#define DEBUG_FMT_S   DEBUG_FMT_ID "%s\n"
static void
rs6000_debug_reg_global (void)
{
static const char *const tf[2] = { "false", "true" };
const char *nl = (const char *)0;
int m;
size_t m1, m2, v;
char costly_num[20];
char nop_num[20];
char flags_buffer[40];
const char *costly_str;
const char *nop_str;
const char *trace_str;
const char *abi_str;
const char *cmodel_str;
struct cl_target_option cl_opts;
static const machine_mode print_tieable_modes[] = {
QImode,
HImode,
SImode,
DImode,
TImode,
PTImode,
SFmode,
DFmode,
TFmode,
IFmode,
KFmode,
SDmode,
DDmode,
TDmode,
V2SImode,
V16QImode,
V8HImode,
V4SImode,
V2DImode,
V1TImode,
V32QImode,
V16HImode,
V8SImode,
V4DImode,
V2TImode,
V2SFmode,
V4SFmode,
V2DFmode,
V8SFmode,
V4DFmode,
CCmode,
CCUNSmode,
CCEQmode,
};
const static struct {
int regno;			
const char *name;		
} virtual_regs[] = {
{ STACK_POINTER_REGNUM,			"stack pointer:" },
{ TOC_REGNUM,				"toc:          " },
{ STATIC_CHAIN_REGNUM,			"static chain: " },
{ RS6000_PIC_OFFSET_TABLE_REGNUM,		"pic offset:   " },
{ HARD_FRAME_POINTER_REGNUM,		"hard frame:   " },
{ ARG_POINTER_REGNUM,			"arg pointer:  " },
{ FRAME_POINTER_REGNUM,			"frame pointer:" },
{ FIRST_PSEUDO_REGISTER,			"first pseudo: " },
{ FIRST_VIRTUAL_REGISTER,			"first virtual:" },
{ VIRTUAL_INCOMING_ARGS_REGNUM,		"incoming_args:" },
{ VIRTUAL_STACK_VARS_REGNUM,		"stack_vars:   " },
{ VIRTUAL_STACK_DYNAMIC_REGNUM,		"stack_dynamic:" },
{ VIRTUAL_OUTGOING_ARGS_REGNUM,		"outgoing_args:" },
{ VIRTUAL_CFA_REGNUM,			"cfa (frame):  " },
{ VIRTUAL_PREFERRED_STACK_BOUNDARY_REGNUM,	"stack boundry:" },
{ LAST_VIRTUAL_REGISTER,			"last virtual: " },
};
fputs ("\nHard register information:\n", stderr);
rs6000_debug_reg_print (FIRST_GPR_REGNO, LAST_GPR_REGNO, "gr");
rs6000_debug_reg_print (FIRST_FPR_REGNO, LAST_FPR_REGNO, "fp");
rs6000_debug_reg_print (FIRST_ALTIVEC_REGNO,
LAST_ALTIVEC_REGNO,
"vs");
rs6000_debug_reg_print (LR_REGNO, LR_REGNO, "lr");
rs6000_debug_reg_print (CTR_REGNO, CTR_REGNO, "ctr");
rs6000_debug_reg_print (CR0_REGNO, CR7_REGNO, "cr");
rs6000_debug_reg_print (CA_REGNO, CA_REGNO, "ca");
rs6000_debug_reg_print (VRSAVE_REGNO, VRSAVE_REGNO, "vrsave");
rs6000_debug_reg_print (VSCR_REGNO, VSCR_REGNO, "vscr");
fputs ("\nVirtual/stack/frame registers:\n", stderr);
for (v = 0; v < ARRAY_SIZE (virtual_regs); v++)
fprintf (stderr, "%s regno = %3d\n", virtual_regs[v].name, virtual_regs[v].regno);
fprintf (stderr,
"\n"
"d  reg_class = %s\n"
"f  reg_class = %s\n"
"v  reg_class = %s\n"
"wa reg_class = %s\n"
"wb reg_class = %s\n"
"wd reg_class = %s\n"
"we reg_class = %s\n"
"wf reg_class = %s\n"
"wg reg_class = %s\n"
"wh reg_class = %s\n"
"wi reg_class = %s\n"
"wj reg_class = %s\n"
"wk reg_class = %s\n"
"wl reg_class = %s\n"
"wm reg_class = %s\n"
"wo reg_class = %s\n"
"wp reg_class = %s\n"
"wq reg_class = %s\n"
"wr reg_class = %s\n"
"ws reg_class = %s\n"
"wt reg_class = %s\n"
"wu reg_class = %s\n"
"wv reg_class = %s\n"
"ww reg_class = %s\n"
"wx reg_class = %s\n"
"wy reg_class = %s\n"
"wz reg_class = %s\n"
"wA reg_class = %s\n"
"wH reg_class = %s\n"
"wI reg_class = %s\n"
"wJ reg_class = %s\n"
"wK reg_class = %s\n"
"\n",
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_d]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_f]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_v]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wa]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wb]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wd]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_we]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wf]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wg]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wh]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wi]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wj]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wk]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wl]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wm]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wo]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wp]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wq]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wr]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_ws]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wt]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wu]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wv]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_ww]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wx]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wy]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wz]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wA]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wH]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wI]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wJ]],
reg_class_names[rs6000_constraints[RS6000_CONSTRAINT_wK]]);
nl = "\n";
for (m = 0; m < NUM_MACHINE_MODES; ++m)
rs6000_debug_print_mode (m);
fputs ("\n", stderr);
for (m1 = 0; m1 < ARRAY_SIZE (print_tieable_modes); m1++)
{
machine_mode mode1 = print_tieable_modes[m1];
bool first_time = true;
nl = (const char *)0;
for (m2 = 0; m2 < ARRAY_SIZE (print_tieable_modes); m2++)
{
machine_mode mode2 = print_tieable_modes[m2];
if (mode1 != mode2 && rs6000_modes_tieable_p (mode1, mode2))
{
if (first_time)
{
fprintf (stderr, "Tieable modes %s:", GET_MODE_NAME (mode1));
nl = "\n";
first_time = false;
}
fprintf (stderr, " %s", GET_MODE_NAME (mode2));
}
}
if (!first_time)
fputs ("\n", stderr);
}
if (nl)
fputs (nl, stderr);
if (rs6000_recip_control)
{
fprintf (stderr, "\nReciprocal mask = 0x%x\n", rs6000_recip_control);
for (m = 0; m < NUM_MACHINE_MODES; ++m)
if (rs6000_recip_bits[m])
{
fprintf (stderr,
"Reciprocal estimate mode: %-5s divide: %s rsqrt: %s\n",
GET_MODE_NAME (m),
(RS6000_RECIP_AUTO_RE_P (m)
? "auto"
: (RS6000_RECIP_HAVE_RE_P (m) ? "have" : "none")),
(RS6000_RECIP_AUTO_RSQRTE_P (m)
? "auto"
: (RS6000_RECIP_HAVE_RSQRTE_P (m) ? "have" : "none")));
}
fputs ("\n", stderr);
}
if (rs6000_cpu_index >= 0)
{
const char *name = processor_target_table[rs6000_cpu_index].name;
HOST_WIDE_INT flags
= processor_target_table[rs6000_cpu_index].target_enable;
sprintf (flags_buffer, "-mcpu=%s flags", name);
rs6000_print_isa_options (stderr, 0, flags_buffer, flags);
}
else
fprintf (stderr, DEBUG_FMT_S, "cpu", "<none>");
if (rs6000_tune_index >= 0)
{
const char *name = processor_target_table[rs6000_tune_index].name;
HOST_WIDE_INT flags
= processor_target_table[rs6000_tune_index].target_enable;
sprintf (flags_buffer, "-mtune=%s flags", name);
rs6000_print_isa_options (stderr, 0, flags_buffer, flags);
}
else
fprintf (stderr, DEBUG_FMT_S, "tune", "<none>");
cl_target_option_save (&cl_opts, &global_options);
rs6000_print_isa_options (stderr, 0, "rs6000_isa_flags",
rs6000_isa_flags);
rs6000_print_isa_options (stderr, 0, "rs6000_isa_flags_explicit",
rs6000_isa_flags_explicit);
rs6000_print_builtin_options (stderr, 0, "rs6000_builtin_mask",
rs6000_builtin_mask);
rs6000_print_isa_options (stderr, 0, "TARGET_DEFAULT", TARGET_DEFAULT);
fprintf (stderr, DEBUG_FMT_S, "--with-cpu default",
OPTION_TARGET_CPU_DEFAULT ? OPTION_TARGET_CPU_DEFAULT : "<none>");
switch (rs6000_sched_costly_dep)
{
case max_dep_latency:
costly_str = "max_dep_latency";
break;
case no_dep_costly:
costly_str = "no_dep_costly";
break;
case all_deps_costly:
costly_str = "all_deps_costly";
break;
case true_store_to_load_dep_costly:
costly_str = "true_store_to_load_dep_costly";
break;
case store_to_load_dep_costly:
costly_str = "store_to_load_dep_costly";
break;
default:
costly_str = costly_num;
sprintf (costly_num, "%d", (int)rs6000_sched_costly_dep);
break;
}
fprintf (stderr, DEBUG_FMT_S, "sched_costly_dep", costly_str);
switch (rs6000_sched_insert_nops)
{
case sched_finish_regroup_exact:
nop_str = "sched_finish_regroup_exact";
break;
case sched_finish_pad_groups:
nop_str = "sched_finish_pad_groups";
break;
case sched_finish_none:
nop_str = "sched_finish_none";
break;
default:
nop_str = nop_num;
sprintf (nop_num, "%d", (int)rs6000_sched_insert_nops);
break;
}
fprintf (stderr, DEBUG_FMT_S, "sched_insert_nops", nop_str);
switch (rs6000_sdata)
{
default:
case SDATA_NONE:
break;
case SDATA_DATA:
fprintf (stderr, DEBUG_FMT_S, "sdata", "data");
break;
case SDATA_SYSV:
fprintf (stderr, DEBUG_FMT_S, "sdata", "sysv");
break;
case SDATA_EABI:
fprintf (stderr, DEBUG_FMT_S, "sdata", "eabi");
break;
}
switch (rs6000_traceback)
{
case traceback_default:	trace_str = "default";	break;
case traceback_none:	trace_str = "none";	break;
case traceback_part:	trace_str = "part";	break;
case traceback_full:	trace_str = "full";	break;
default:			trace_str = "unknown";	break;
}
fprintf (stderr, DEBUG_FMT_S, "traceback", trace_str);
switch (rs6000_current_cmodel)
{
case CMODEL_SMALL:	cmodel_str = "small";	break;
case CMODEL_MEDIUM:	cmodel_str = "medium";	break;
case CMODEL_LARGE:	cmodel_str = "large";	break;
default:		cmodel_str = "unknown";	break;
}
fprintf (stderr, DEBUG_FMT_S, "cmodel", cmodel_str);
switch (rs6000_current_abi)
{
case ABI_NONE:	abi_str = "none";	break;
case ABI_AIX:	abi_str = "aix";	break;
case ABI_ELFv2:	abi_str = "ELFv2";	break;
case ABI_V4:	abi_str = "V4";		break;
case ABI_DARWIN:	abi_str = "darwin";	break;
default:		abi_str = "unknown";	break;
}
fprintf (stderr, DEBUG_FMT_S, "abi", abi_str);
if (rs6000_altivec_abi)
fprintf (stderr, DEBUG_FMT_S, "altivec_abi", "true");
if (rs6000_darwin64_abi)
fprintf (stderr, DEBUG_FMT_S, "darwin64_abi", "true");
fprintf (stderr, DEBUG_FMT_S, "single_float",
(TARGET_SINGLE_FLOAT ? "true" : "false"));
fprintf (stderr, DEBUG_FMT_S, "double_float",
(TARGET_DOUBLE_FLOAT ? "true" : "false"));
fprintf (stderr, DEBUG_FMT_S, "soft_float",
(TARGET_SOFT_FLOAT ? "true" : "false"));
if (TARGET_LINK_STACK)
fprintf (stderr, DEBUG_FMT_S, "link_stack", "true");
if (TARGET_P8_FUSION)
{
char options[80];
strcpy (options, (TARGET_P9_FUSION) ? "power9" : "power8");
if (TARGET_TOC_FUSION)
strcat (options, ", toc");
if (TARGET_P8_FUSION_SIGN)
strcat (options, ", sign");
fprintf (stderr, DEBUG_FMT_S, "fusion", options);
}
fprintf (stderr, DEBUG_FMT_S, "plt-format",
TARGET_SECURE_PLT ? "secure" : "bss");
fprintf (stderr, DEBUG_FMT_S, "struct-return",
aix_struct_return ? "aix" : "sysv");
fprintf (stderr, DEBUG_FMT_S, "always_hint", tf[!!rs6000_always_hint]);
fprintf (stderr, DEBUG_FMT_S, "sched_groups", tf[!!rs6000_sched_groups]);
fprintf (stderr, DEBUG_FMT_S, "align_branch",
tf[!!rs6000_align_branch_targets]);
fprintf (stderr, DEBUG_FMT_D, "tls_size", rs6000_tls_size);
fprintf (stderr, DEBUG_FMT_D, "long_double_size",
rs6000_long_double_type_size);
if (rs6000_long_double_type_size > 64)
{
fprintf (stderr, DEBUG_FMT_S, "long double type",
TARGET_IEEEQUAD ? "IEEE" : "IBM");
fprintf (stderr, DEBUG_FMT_S, "default long double type",
TARGET_IEEEQUAD_DEFAULT ? "IEEE" : "IBM");
}
fprintf (stderr, DEBUG_FMT_D, "sched_restricted_insns_priority",
(int)rs6000_sched_restricted_insns_priority);
fprintf (stderr, DEBUG_FMT_D, "Number of standard builtins",
(int)END_BUILTINS);
fprintf (stderr, DEBUG_FMT_D, "Number of rs6000 builtins",
(int)RS6000_BUILTIN_COUNT);
fprintf (stderr, DEBUG_FMT_D, "Enable float128 on VSX",
(int)TARGET_FLOAT128_ENABLE_TYPE);
if (TARGET_VSX)
fprintf (stderr, DEBUG_FMT_D, "VSX easy 64-bit scalar element",
(int)VECTOR_ELEMENT_SCALAR_64BIT);
if (TARGET_DIRECT_MOVE_128)
fprintf (stderr, DEBUG_FMT_D, "VSX easy 64-bit mfvsrld element",
(int)VECTOR_ELEMENT_MFVSRLD_64BIT);
}

static void
rs6000_setup_reg_addr_masks (void)
{
ssize_t rc, reg, m, nregs;
addr_mask_type any_addr_mask, addr_mask;
for (m = 0; m < NUM_MACHINE_MODES; ++m)
{
machine_mode m2 = (machine_mode) m;
bool complex_p = false;
bool small_int_p = (m2 == QImode || m2 == HImode || m2 == SImode);
size_t msize;
if (COMPLEX_MODE_P (m2))
{
complex_p = true;
m2 = GET_MODE_INNER (m2);
}
msize = GET_MODE_SIZE (m2);
bool indexed_only_p = (m == SDmode && TARGET_NO_SDMODE_STACK);
any_addr_mask = 0;
for (rc = FIRST_RELOAD_REG_CLASS; rc <= LAST_RELOAD_REG_CLASS; rc++)
{
addr_mask = 0;
reg = reload_reg_map[rc].reg;
if (reg >= 0 && rs6000_hard_regno_mode_ok_p[m][reg])
{
bool small_int_vsx_p = (small_int_p
&& (rc == RELOAD_REG_FPR
|| rc == RELOAD_REG_VMX));
nregs = rs6000_hard_regno_nregs[m][reg];
addr_mask |= RELOAD_REG_VALID;
if (small_int_vsx_p)
addr_mask |= RELOAD_REG_INDEXED;
else if (nregs > 1 || m == BLKmode || complex_p)
addr_mask |= RELOAD_REG_MULTIPLE;
else
addr_mask |= RELOAD_REG_INDEXED;
if (TARGET_UPDATE
&& (rc == RELOAD_REG_GPR || rc == RELOAD_REG_FPR)
&& msize <= 8
&& !VECTOR_MODE_P (m2)
&& !FLOAT128_VECTOR_P (m2)
&& !complex_p
&& (m != E_DFmode || !TARGET_VSX)
&& (m != E_SFmode || !TARGET_P8_VECTOR)
&& !small_int_vsx_p)
{
addr_mask |= RELOAD_REG_PRE_INCDEC;
switch (m)
{
default:
addr_mask |= RELOAD_REG_PRE_MODIFY;
break;
case E_DImode:
if (TARGET_POWERPC64)
addr_mask |= RELOAD_REG_PRE_MODIFY;
break;
case E_DFmode:
case E_DDmode:
if (TARGET_DF_INSN)
addr_mask |= RELOAD_REG_PRE_MODIFY;
break;
}
}
}
if ((addr_mask != 0) && !indexed_only_p
&& msize <= 8
&& (rc == RELOAD_REG_GPR
|| ((msize == 8 || m2 == SFmode)
&& (rc == RELOAD_REG_FPR
|| (rc == RELOAD_REG_VMX && TARGET_P9_VECTOR)))))
addr_mask |= RELOAD_REG_OFFSET;
else if ((addr_mask != 0) && !indexed_only_p
&& msize == 16 && TARGET_P9_VECTOR
&& (ALTIVEC_OR_VSX_VECTOR_MODE (m2)
|| (m2 == TImode && TARGET_VSX)))
{
addr_mask |= RELOAD_REG_OFFSET;
if (rc == RELOAD_REG_FPR || rc == RELOAD_REG_VMX)
addr_mask |= RELOAD_REG_QUAD_OFFSET;
}
if (rc == RELOAD_REG_VMX && msize == 16
&& (addr_mask & RELOAD_REG_VALID) != 0)
addr_mask |= RELOAD_REG_AND_M16;
reg_addr[m].addr_mask[rc] = addr_mask;
any_addr_mask |= addr_mask;
}
reg_addr[m].addr_mask[RELOAD_REG_ANY] = any_addr_mask;
}
}

static void
rs6000_init_hard_regno_mode_ok (bool global_init_p)
{
ssize_t r, m, c;
int align64;
int align32;
rs6000_regno_regclass[0] = GENERAL_REGS;
for (r = 1; r < 32; ++r)
rs6000_regno_regclass[r] = BASE_REGS;
for (r = 32; r < 64; ++r)
rs6000_regno_regclass[r] = FLOAT_REGS;
for (r = 64; r < FIRST_PSEUDO_REGISTER; ++r)
rs6000_regno_regclass[r] = NO_REGS;
for (r = FIRST_ALTIVEC_REGNO; r <= LAST_ALTIVEC_REGNO; ++r)
rs6000_regno_regclass[r] = ALTIVEC_REGS;
rs6000_regno_regclass[CR0_REGNO] = CR0_REGS;
for (r = CR1_REGNO; r <= CR7_REGNO; ++r)
rs6000_regno_regclass[r] = CR_REGS;
rs6000_regno_regclass[LR_REGNO] = LINK_REGS;
rs6000_regno_regclass[CTR_REGNO] = CTR_REGS;
rs6000_regno_regclass[CA_REGNO] = NO_REGS;
rs6000_regno_regclass[VRSAVE_REGNO] = VRSAVE_REGS;
rs6000_regno_regclass[VSCR_REGNO] = VRSAVE_REGS;
rs6000_regno_regclass[TFHAR_REGNO] = SPR_REGS;
rs6000_regno_regclass[TFIAR_REGNO] = SPR_REGS;
rs6000_regno_regclass[TEXASR_REGNO] = SPR_REGS;
rs6000_regno_regclass[ARG_POINTER_REGNUM] = BASE_REGS;
rs6000_regno_regclass[FRAME_POINTER_REGNUM] = BASE_REGS;
for (c = 0; c < N_REG_CLASSES; c++)
reg_class_to_reg_type[c] = NO_REG_TYPE;
reg_class_to_reg_type[(int)GENERAL_REGS] = GPR_REG_TYPE;
reg_class_to_reg_type[(int)BASE_REGS] = GPR_REG_TYPE;
reg_class_to_reg_type[(int)VSX_REGS] = VSX_REG_TYPE;
reg_class_to_reg_type[(int)VRSAVE_REGS] = SPR_REG_TYPE;
reg_class_to_reg_type[(int)VSCR_REGS] = SPR_REG_TYPE;
reg_class_to_reg_type[(int)LINK_REGS] = SPR_REG_TYPE;
reg_class_to_reg_type[(int)CTR_REGS] = SPR_REG_TYPE;
reg_class_to_reg_type[(int)LINK_OR_CTR_REGS] = SPR_REG_TYPE;
reg_class_to_reg_type[(int)CR_REGS] = CR_REG_TYPE;
reg_class_to_reg_type[(int)CR0_REGS] = CR_REG_TYPE;
if (TARGET_VSX)
{
reg_class_to_reg_type[(int)FLOAT_REGS] = VSX_REG_TYPE;
reg_class_to_reg_type[(int)ALTIVEC_REGS] = VSX_REG_TYPE;
}
else
{
reg_class_to_reg_type[(int)FLOAT_REGS] = FPR_REG_TYPE;
reg_class_to_reg_type[(int)ALTIVEC_REGS] = ALTIVEC_REG_TYPE;
}
gcc_assert ((int)VECTOR_NONE == 0);
memset ((void *) &rs6000_vector_unit[0], '\0', sizeof (rs6000_vector_unit));
memset ((void *) &rs6000_vector_mem[0], '\0', sizeof (rs6000_vector_unit));
gcc_assert ((int)CODE_FOR_nothing == 0);
memset ((void *) &reg_addr[0], '\0', sizeof (reg_addr));
gcc_assert ((int)NO_REGS == 0);
memset ((void *) &rs6000_constraints[0], '\0', sizeof (rs6000_constraints));
if (TARGET_VSX && !TARGET_VSX_ALIGN_128)
{
align64 = 64;
align32 = 32;
}
else
{
align64 = 128;
align32 = 128;
}
if (TARGET_FLOAT128_TYPE)
{
rs6000_vector_mem[KFmode] = VECTOR_VSX;
rs6000_vector_align[KFmode] = 128;
if (FLOAT128_IEEE_P (TFmode))
{
rs6000_vector_mem[TFmode] = VECTOR_VSX;
rs6000_vector_align[TFmode] = 128;
}
}
if (TARGET_VSX)
{
rs6000_vector_unit[V2DFmode] = VECTOR_VSX;
rs6000_vector_mem[V2DFmode] = VECTOR_VSX;
rs6000_vector_align[V2DFmode] = align64;
}
if (TARGET_VSX)
{
rs6000_vector_unit[V4SFmode] = VECTOR_VSX;
rs6000_vector_mem[V4SFmode] = VECTOR_VSX;
rs6000_vector_align[V4SFmode] = align32;
}
else if (TARGET_ALTIVEC)
{
rs6000_vector_unit[V4SFmode] = VECTOR_ALTIVEC;
rs6000_vector_mem[V4SFmode] = VECTOR_ALTIVEC;
rs6000_vector_align[V4SFmode] = align32;
}
if (TARGET_ALTIVEC)
{
rs6000_vector_unit[V4SImode] = VECTOR_ALTIVEC;
rs6000_vector_unit[V8HImode] = VECTOR_ALTIVEC;
rs6000_vector_unit[V16QImode] = VECTOR_ALTIVEC;
rs6000_vector_align[V4SImode] = align32;
rs6000_vector_align[V8HImode] = align32;
rs6000_vector_align[V16QImode] = align32;
if (TARGET_VSX)
{
rs6000_vector_mem[V4SImode] = VECTOR_VSX;
rs6000_vector_mem[V8HImode] = VECTOR_VSX;
rs6000_vector_mem[V16QImode] = VECTOR_VSX;
}
else
{
rs6000_vector_mem[V4SImode] = VECTOR_ALTIVEC;
rs6000_vector_mem[V8HImode] = VECTOR_ALTIVEC;
rs6000_vector_mem[V16QImode] = VECTOR_ALTIVEC;
}
}
if (TARGET_VSX)
{
rs6000_vector_mem[V2DImode] = VECTOR_VSX;
rs6000_vector_unit[V2DImode]
= (TARGET_P8_VECTOR) ? VECTOR_P8_VECTOR : VECTOR_NONE;
rs6000_vector_align[V2DImode] = align64;
rs6000_vector_mem[V1TImode] = VECTOR_VSX;
rs6000_vector_unit[V1TImode]
= (TARGET_P8_VECTOR) ? VECTOR_P8_VECTOR : VECTOR_NONE;
rs6000_vector_align[V1TImode] = 128;
}
if (TARGET_VSX)
{
rs6000_vector_unit[DFmode] = VECTOR_VSX;
rs6000_vector_align[DFmode] = 64;
}
if (TARGET_P8_VECTOR)
{
rs6000_vector_unit[SFmode] = VECTOR_VSX;
rs6000_vector_align[SFmode] = 32;
}
if (TARGET_VSX)
{
rs6000_vector_mem[TImode] = VECTOR_VSX;
rs6000_vector_align[TImode] = align64;
}
if (TARGET_HARD_FLOAT)
rs6000_constraints[RS6000_CONSTRAINT_f] = FLOAT_REGS;	
if (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
rs6000_constraints[RS6000_CONSTRAINT_d]  = FLOAT_REGS;	
if (TARGET_VSX)
{
rs6000_constraints[RS6000_CONSTRAINT_wa] = VSX_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wd] = VSX_REGS;	
rs6000_constraints[RS6000_CONSTRAINT_wf] = VSX_REGS;	
rs6000_constraints[RS6000_CONSTRAINT_ws] = VSX_REGS;	
rs6000_constraints[RS6000_CONSTRAINT_wv] = ALTIVEC_REGS;	
rs6000_constraints[RS6000_CONSTRAINT_wi] = VSX_REGS;	
rs6000_constraints[RS6000_CONSTRAINT_wt] = VSX_REGS;	
}
if (TARGET_ALTIVEC)
rs6000_constraints[RS6000_CONSTRAINT_v] = ALTIVEC_REGS;
if (TARGET_MFPGPR)						
rs6000_constraints[RS6000_CONSTRAINT_wg] = FLOAT_REGS;
if (TARGET_LFIWAX)
rs6000_constraints[RS6000_CONSTRAINT_wl] = FLOAT_REGS;	
if (TARGET_DIRECT_MOVE)
{
rs6000_constraints[RS6000_CONSTRAINT_wh] = FLOAT_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wj]			
= rs6000_constraints[RS6000_CONSTRAINT_wi];
rs6000_constraints[RS6000_CONSTRAINT_wk]			
= rs6000_constraints[RS6000_CONSTRAINT_ws];
rs6000_constraints[RS6000_CONSTRAINT_wm] = VSX_REGS;
}
if (TARGET_POWERPC64)
{
rs6000_constraints[RS6000_CONSTRAINT_wr] = GENERAL_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wA] = BASE_REGS;
}
if (TARGET_P8_VECTOR)						
{
rs6000_constraints[RS6000_CONSTRAINT_wu] = ALTIVEC_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wy] = VSX_REGS;
rs6000_constraints[RS6000_CONSTRAINT_ww] = VSX_REGS;
}
else if (TARGET_VSX)
rs6000_constraints[RS6000_CONSTRAINT_ww] = FLOAT_REGS;
if (TARGET_STFIWX)
rs6000_constraints[RS6000_CONSTRAINT_wx] = FLOAT_REGS;	
if (TARGET_LFIWZX)
rs6000_constraints[RS6000_CONSTRAINT_wz] = FLOAT_REGS;	
if (TARGET_FLOAT128_TYPE)
{
rs6000_constraints[RS6000_CONSTRAINT_wq] = VSX_REGS;	
if (FLOAT128_IEEE_P (TFmode))
rs6000_constraints[RS6000_CONSTRAINT_wp] = VSX_REGS;	
}
if (TARGET_P9_VECTOR)
{
rs6000_constraints[RS6000_CONSTRAINT_wb] = ALTIVEC_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wo] = VSX_REGS;
}
if (TARGET_DIRECT_MOVE_128)
rs6000_constraints[RS6000_CONSTRAINT_we] = VSX_REGS;
if (TARGET_P8_VECTOR)
{
rs6000_constraints[RS6000_CONSTRAINT_wH] = ALTIVEC_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wI] = FLOAT_REGS;
if (TARGET_P9_VECTOR)
{
rs6000_constraints[RS6000_CONSTRAINT_wJ] = FLOAT_REGS;
rs6000_constraints[RS6000_CONSTRAINT_wK] = ALTIVEC_REGS;
}
}
if (TARGET_VSX || TARGET_ALTIVEC)
{
if (TARGET_64BIT)
{
reg_addr[V16QImode].reload_store = CODE_FOR_reload_v16qi_di_store;
reg_addr[V16QImode].reload_load  = CODE_FOR_reload_v16qi_di_load;
reg_addr[V8HImode].reload_store  = CODE_FOR_reload_v8hi_di_store;
reg_addr[V8HImode].reload_load   = CODE_FOR_reload_v8hi_di_load;
reg_addr[V4SImode].reload_store  = CODE_FOR_reload_v4si_di_store;
reg_addr[V4SImode].reload_load   = CODE_FOR_reload_v4si_di_load;
reg_addr[V2DImode].reload_store  = CODE_FOR_reload_v2di_di_store;
reg_addr[V2DImode].reload_load   = CODE_FOR_reload_v2di_di_load;
reg_addr[V1TImode].reload_store  = CODE_FOR_reload_v1ti_di_store;
reg_addr[V1TImode].reload_load   = CODE_FOR_reload_v1ti_di_load;
reg_addr[V4SFmode].reload_store  = CODE_FOR_reload_v4sf_di_store;
reg_addr[V4SFmode].reload_load   = CODE_FOR_reload_v4sf_di_load;
reg_addr[V2DFmode].reload_store  = CODE_FOR_reload_v2df_di_store;
reg_addr[V2DFmode].reload_load   = CODE_FOR_reload_v2df_di_load;
reg_addr[DFmode].reload_store    = CODE_FOR_reload_df_di_store;
reg_addr[DFmode].reload_load     = CODE_FOR_reload_df_di_load;
reg_addr[DDmode].reload_store    = CODE_FOR_reload_dd_di_store;
reg_addr[DDmode].reload_load     = CODE_FOR_reload_dd_di_load;
reg_addr[SFmode].reload_store    = CODE_FOR_reload_sf_di_store;
reg_addr[SFmode].reload_load     = CODE_FOR_reload_sf_di_load;
if (FLOAT128_VECTOR_P (KFmode))
{
reg_addr[KFmode].reload_store = CODE_FOR_reload_kf_di_store;
reg_addr[KFmode].reload_load  = CODE_FOR_reload_kf_di_load;
}
if (FLOAT128_VECTOR_P (TFmode))
{
reg_addr[TFmode].reload_store = CODE_FOR_reload_tf_di_store;
reg_addr[TFmode].reload_load  = CODE_FOR_reload_tf_di_load;
}
if (TARGET_NO_SDMODE_STACK)
{
reg_addr[SDmode].reload_store = CODE_FOR_reload_sd_di_store;
reg_addr[SDmode].reload_load  = CODE_FOR_reload_sd_di_load;
}
if (TARGET_VSX)
{
reg_addr[TImode].reload_store  = CODE_FOR_reload_ti_di_store;
reg_addr[TImode].reload_load   = CODE_FOR_reload_ti_di_load;
}
if (TARGET_DIRECT_MOVE && !TARGET_DIRECT_MOVE_128)
{
reg_addr[TImode].reload_gpr_vsx    = CODE_FOR_reload_gpr_from_vsxti;
reg_addr[V1TImode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv1ti;
reg_addr[V2DFmode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv2df;
reg_addr[V2DImode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv2di;
reg_addr[V4SFmode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv4sf;
reg_addr[V4SImode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv4si;
reg_addr[V8HImode].reload_gpr_vsx  = CODE_FOR_reload_gpr_from_vsxv8hi;
reg_addr[V16QImode].reload_gpr_vsx = CODE_FOR_reload_gpr_from_vsxv16qi;
reg_addr[SFmode].reload_gpr_vsx    = CODE_FOR_reload_gpr_from_vsxsf;
reg_addr[TImode].reload_vsx_gpr    = CODE_FOR_reload_vsx_from_gprti;
reg_addr[V1TImode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv1ti;
reg_addr[V2DFmode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv2df;
reg_addr[V2DImode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv2di;
reg_addr[V4SFmode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv4sf;
reg_addr[V4SImode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv4si;
reg_addr[V8HImode].reload_vsx_gpr  = CODE_FOR_reload_vsx_from_gprv8hi;
reg_addr[V16QImode].reload_vsx_gpr = CODE_FOR_reload_vsx_from_gprv16qi;
reg_addr[SFmode].reload_vsx_gpr    = CODE_FOR_reload_vsx_from_gprsf;
if (FLOAT128_VECTOR_P (KFmode))
{
reg_addr[KFmode].reload_gpr_vsx = CODE_FOR_reload_gpr_from_vsxkf;
reg_addr[KFmode].reload_vsx_gpr = CODE_FOR_reload_vsx_from_gprkf;
}
if (FLOAT128_VECTOR_P (TFmode))
{
reg_addr[TFmode].reload_gpr_vsx = CODE_FOR_reload_gpr_from_vsxtf;
reg_addr[TFmode].reload_vsx_gpr = CODE_FOR_reload_vsx_from_gprtf;
}
}
}
else
{
reg_addr[V16QImode].reload_store = CODE_FOR_reload_v16qi_si_store;
reg_addr[V16QImode].reload_load  = CODE_FOR_reload_v16qi_si_load;
reg_addr[V8HImode].reload_store  = CODE_FOR_reload_v8hi_si_store;
reg_addr[V8HImode].reload_load   = CODE_FOR_reload_v8hi_si_load;
reg_addr[V4SImode].reload_store  = CODE_FOR_reload_v4si_si_store;
reg_addr[V4SImode].reload_load   = CODE_FOR_reload_v4si_si_load;
reg_addr[V2DImode].reload_store  = CODE_FOR_reload_v2di_si_store;
reg_addr[V2DImode].reload_load   = CODE_FOR_reload_v2di_si_load;
reg_addr[V1TImode].reload_store  = CODE_FOR_reload_v1ti_si_store;
reg_addr[V1TImode].reload_load   = CODE_FOR_reload_v1ti_si_load;
reg_addr[V4SFmode].reload_store  = CODE_FOR_reload_v4sf_si_store;
reg_addr[V4SFmode].reload_load   = CODE_FOR_reload_v4sf_si_load;
reg_addr[V2DFmode].reload_store  = CODE_FOR_reload_v2df_si_store;
reg_addr[V2DFmode].reload_load   = CODE_FOR_reload_v2df_si_load;
reg_addr[DFmode].reload_store    = CODE_FOR_reload_df_si_store;
reg_addr[DFmode].reload_load     = CODE_FOR_reload_df_si_load;
reg_addr[DDmode].reload_store    = CODE_FOR_reload_dd_si_store;
reg_addr[DDmode].reload_load     = CODE_FOR_reload_dd_si_load;
reg_addr[SFmode].reload_store    = CODE_FOR_reload_sf_si_store;
reg_addr[SFmode].reload_load     = CODE_FOR_reload_sf_si_load;
if (FLOAT128_VECTOR_P (KFmode))
{
reg_addr[KFmode].reload_store = CODE_FOR_reload_kf_si_store;
reg_addr[KFmode].reload_load  = CODE_FOR_reload_kf_si_load;
}
if (FLOAT128_IEEE_P (TFmode))
{
reg_addr[TFmode].reload_store = CODE_FOR_reload_tf_si_store;
reg_addr[TFmode].reload_load  = CODE_FOR_reload_tf_si_load;
}
if (TARGET_NO_SDMODE_STACK)
{
reg_addr[SDmode].reload_store = CODE_FOR_reload_sd_si_store;
reg_addr[SDmode].reload_load  = CODE_FOR_reload_sd_si_load;
}
if (TARGET_VSX)
{
reg_addr[TImode].reload_store  = CODE_FOR_reload_ti_si_store;
reg_addr[TImode].reload_load   = CODE_FOR_reload_ti_si_load;
}
if (TARGET_DIRECT_MOVE)
{
reg_addr[DImode].reload_fpr_gpr = CODE_FOR_reload_fpr_from_gprdi;
reg_addr[DDmode].reload_fpr_gpr = CODE_FOR_reload_fpr_from_gprdd;
reg_addr[DFmode].reload_fpr_gpr = CODE_FOR_reload_fpr_from_gprdf;
}
}
reg_addr[DFmode].scalar_in_vmx_p = true;
reg_addr[DImode].scalar_in_vmx_p = true;
if (TARGET_P8_VECTOR)
{
reg_addr[SFmode].scalar_in_vmx_p = true;
reg_addr[SImode].scalar_in_vmx_p = true;
if (TARGET_P9_VECTOR)
{
reg_addr[HImode].scalar_in_vmx_p = true;
reg_addr[QImode].scalar_in_vmx_p = true;
}
}
}
if (TARGET_P8_FUSION)
{
reg_addr[QImode].fusion_gpr_ld = CODE_FOR_fusion_gpr_load_qi;
reg_addr[HImode].fusion_gpr_ld = CODE_FOR_fusion_gpr_load_hi;
reg_addr[SImode].fusion_gpr_ld = CODE_FOR_fusion_gpr_load_si;
if (TARGET_64BIT)
reg_addr[DImode].fusion_gpr_ld = CODE_FOR_fusion_gpr_load_di;
}
if (TARGET_P9_FUSION)
{
struct fuse_insns {
enum machine_mode mode;			
enum machine_mode pmode;		
enum rs6000_reload_reg_type rtype;	
enum insn_code load;			
enum insn_code store;			
};
static const struct fuse_insns addis_insns[] = {
{ E_SFmode, E_DImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_di_sf_load,
CODE_FOR_fusion_vsx_di_sf_store },
{ E_SFmode, E_SImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_si_sf_load,
CODE_FOR_fusion_vsx_si_sf_store },
{ E_DFmode, E_DImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_di_df_load,
CODE_FOR_fusion_vsx_di_df_store },
{ E_DFmode, E_SImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_si_df_load,
CODE_FOR_fusion_vsx_si_df_store },
{ E_DImode, E_DImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_di_di_load,
CODE_FOR_fusion_vsx_di_di_store },
{ E_DImode, E_SImode, RELOAD_REG_FPR,
CODE_FOR_fusion_vsx_si_di_load,
CODE_FOR_fusion_vsx_si_di_store },
{ E_QImode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_qi_load,
CODE_FOR_fusion_gpr_di_qi_store },
{ E_QImode, E_SImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_si_qi_load,
CODE_FOR_fusion_gpr_si_qi_store },
{ E_HImode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_hi_load,
CODE_FOR_fusion_gpr_di_hi_store },
{ E_HImode, E_SImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_si_hi_load,
CODE_FOR_fusion_gpr_si_hi_store },
{ E_SImode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_si_load,
CODE_FOR_fusion_gpr_di_si_store },
{ E_SImode, E_SImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_si_si_load,
CODE_FOR_fusion_gpr_si_si_store },
{ E_SFmode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_sf_load,
CODE_FOR_fusion_gpr_di_sf_store },
{ E_SFmode, E_SImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_si_sf_load,
CODE_FOR_fusion_gpr_si_sf_store },
{ E_DImode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_di_load,
CODE_FOR_fusion_gpr_di_di_store },
{ E_DFmode, E_DImode, RELOAD_REG_GPR,
CODE_FOR_fusion_gpr_di_df_load,
CODE_FOR_fusion_gpr_di_df_store },
};
machine_mode cur_pmode = Pmode;
size_t i;
for (i = 0; i < ARRAY_SIZE (addis_insns); i++)
{
machine_mode xmode = addis_insns[i].mode;
enum rs6000_reload_reg_type rtype = addis_insns[i].rtype;
if (addis_insns[i].pmode != cur_pmode)
continue;
if (rtype == RELOAD_REG_FPR && !TARGET_HARD_FLOAT)
continue;
reg_addr[xmode].fusion_addis_ld[rtype] = addis_insns[i].load;
reg_addr[xmode].fusion_addis_st[rtype] = addis_insns[i].store;
if (rtype == RELOAD_REG_FPR && TARGET_P9_VECTOR)
{
reg_addr[xmode].fusion_addis_ld[RELOAD_REG_VMX]
= addis_insns[i].load;
reg_addr[xmode].fusion_addis_st[RELOAD_REG_VMX]
= addis_insns[i].store;
}
}
}
if (TARGET_P8_FUSION && TARGET_TOC_FUSION && TARGET_POWERPC64
&& (TARGET_CMODEL != CMODEL_SMALL))
{
reg_addr[QImode].fused_toc = true;
reg_addr[HImode].fused_toc = true;
reg_addr[SImode].fused_toc = true;
reg_addr[DImode].fused_toc = true;
if (TARGET_HARD_FLOAT)
{
if (TARGET_SINGLE_FLOAT)
reg_addr[SFmode].fused_toc = true;
if (TARGET_DOUBLE_FLOAT)
reg_addr[DFmode].fused_toc = true;
}
}
for (r = 0; r < FIRST_PSEUDO_REGISTER; ++r)
for (m = 0; m < NUM_MACHINE_MODES; ++m)
rs6000_hard_regno_nregs[m][r]
= rs6000_hard_regno_nregs_internal (r, (machine_mode)m);
for (r = 0; r < FIRST_PSEUDO_REGISTER; ++r)
for (m = 0; m < NUM_MACHINE_MODES; ++m)
if (rs6000_hard_regno_mode_ok_uncached (r, (machine_mode)m))
rs6000_hard_regno_mode_ok_p[m][r] = true;
for (c = 0; c < LIM_REG_CLASSES; ++c)
{
int reg_size;
if (TARGET_VSX && VSX_REG_CLASS_P (c))
reg_size = UNITS_PER_VSX_WORD;
else if (c == ALTIVEC_REGS)
reg_size = UNITS_PER_ALTIVEC_WORD;
else if (c == FLOAT_REGS)
reg_size = UNITS_PER_FP_WORD;
else
reg_size = UNITS_PER_WORD;
for (m = 0; m < NUM_MACHINE_MODES; ++m)
{
machine_mode m2 = (machine_mode)m;
int reg_size2 = reg_size;
if (TARGET_VSX && VSX_REG_CLASS_P (c) && FLOAT128_2REG_P (m))
reg_size2 = UNITS_PER_FP_WORD;
rs6000_class_max_nregs[m][c]
= (GET_MODE_SIZE (m2) + reg_size2 - 1) / reg_size2;
}
}
memset (rs6000_recip_bits, 0, sizeof (rs6000_recip_bits));
if (TARGET_FRES)
rs6000_recip_bits[SFmode] = RS6000_RECIP_MASK_HAVE_RE;
if (TARGET_FRE)
rs6000_recip_bits[DFmode] = RS6000_RECIP_MASK_HAVE_RE;
if (VECTOR_UNIT_ALTIVEC_OR_VSX_P (V4SFmode))
rs6000_recip_bits[V4SFmode] = RS6000_RECIP_MASK_HAVE_RE;
if (VECTOR_UNIT_VSX_P (V2DFmode))
rs6000_recip_bits[V2DFmode] = RS6000_RECIP_MASK_HAVE_RE;
if (TARGET_FRSQRTES)
rs6000_recip_bits[SFmode] |= RS6000_RECIP_MASK_HAVE_RSQRTE;
if (TARGET_FRSQRTE)
rs6000_recip_bits[DFmode] |= RS6000_RECIP_MASK_HAVE_RSQRTE;
if (VECTOR_UNIT_ALTIVEC_OR_VSX_P (V4SFmode))
rs6000_recip_bits[V4SFmode] |= RS6000_RECIP_MASK_HAVE_RSQRTE;
if (VECTOR_UNIT_VSX_P (V2DFmode))
rs6000_recip_bits[V2DFmode] |= RS6000_RECIP_MASK_HAVE_RSQRTE;
if (rs6000_recip_control)
{
if (!flag_finite_math_only)
warning (0, "%qs requires %qs or %qs", "-mrecip", "-ffinite-math",
"-ffast-math");
if (flag_trapping_math)
warning (0, "%qs requires %qs or %qs", "-mrecip",
"-fno-trapping-math", "-ffast-math");
if (!flag_reciprocal_math)
warning (0, "%qs requires %qs or %qs", "-mrecip", "-freciprocal-math",
"-ffast-math");
if (flag_finite_math_only && !flag_trapping_math && flag_reciprocal_math)
{
if (RS6000_RECIP_HAVE_RE_P (SFmode)
&& (rs6000_recip_control & RECIP_SF_DIV) != 0)
rs6000_recip_bits[SFmode] |= RS6000_RECIP_MASK_AUTO_RE;
if (RS6000_RECIP_HAVE_RE_P (DFmode)
&& (rs6000_recip_control & RECIP_DF_DIV) != 0)
rs6000_recip_bits[DFmode] |= RS6000_RECIP_MASK_AUTO_RE;
if (RS6000_RECIP_HAVE_RE_P (V4SFmode)
&& (rs6000_recip_control & RECIP_V4SF_DIV) != 0)
rs6000_recip_bits[V4SFmode] |= RS6000_RECIP_MASK_AUTO_RE;
if (RS6000_RECIP_HAVE_RE_P (V2DFmode)
&& (rs6000_recip_control & RECIP_V2DF_DIV) != 0)
rs6000_recip_bits[V2DFmode] |= RS6000_RECIP_MASK_AUTO_RE;
if (RS6000_RECIP_HAVE_RSQRTE_P (SFmode)
&& (rs6000_recip_control & RECIP_SF_RSQRT) != 0)
rs6000_recip_bits[SFmode] |= RS6000_RECIP_MASK_AUTO_RSQRTE;
if (RS6000_RECIP_HAVE_RSQRTE_P (DFmode)
&& (rs6000_recip_control & RECIP_DF_RSQRT) != 0)
rs6000_recip_bits[DFmode] |= RS6000_RECIP_MASK_AUTO_RSQRTE;
if (RS6000_RECIP_HAVE_RSQRTE_P (V4SFmode)
&& (rs6000_recip_control & RECIP_V4SF_RSQRT) != 0)
rs6000_recip_bits[V4SFmode] |= RS6000_RECIP_MASK_AUTO_RSQRTE;
if (RS6000_RECIP_HAVE_RSQRTE_P (V2DFmode)
&& (rs6000_recip_control & RECIP_V2DF_RSQRT) != 0)
rs6000_recip_bits[V2DFmode] |= RS6000_RECIP_MASK_AUTO_RSQRTE;
}
}
rs6000_setup_reg_addr_masks ();
if (global_init_p || TARGET_DEBUG_TARGET)
{
if (TARGET_DEBUG_REG)
rs6000_debug_reg_global ();
if (TARGET_DEBUG_COST || TARGET_DEBUG_REG)
fprintf (stderr,
"SImode variable mult cost       = %d\n"
"SImode constant mult cost       = %d\n"
"SImode short constant mult cost = %d\n"
"DImode multipliciation cost     = %d\n"
"SImode division cost            = %d\n"
"DImode division cost            = %d\n"
"Simple fp operation cost        = %d\n"
"DFmode multiplication cost      = %d\n"
"SFmode division cost            = %d\n"
"DFmode division cost            = %d\n"
"cache line size                 = %d\n"
"l1 cache size                   = %d\n"
"l2 cache size                   = %d\n"
"simultaneous prefetches         = %d\n"
"\n",
rs6000_cost->mulsi,
rs6000_cost->mulsi_const,
rs6000_cost->mulsi_const9,
rs6000_cost->muldi,
rs6000_cost->divsi,
rs6000_cost->divdi,
rs6000_cost->fp,
rs6000_cost->dmul,
rs6000_cost->sdiv,
rs6000_cost->ddiv,
rs6000_cost->cache_line_size,
rs6000_cost->l1_cache_size,
rs6000_cost->l2_cache_size,
rs6000_cost->simultaneous_prefetches);
}
}
#if TARGET_MACHO
static void
darwin_rs6000_override_options (void)
{
rs6000_altivec_abi = 1;
TARGET_ALTIVEC_VRSAVE = 1;
rs6000_current_abi = ABI_DARWIN;
if (DEFAULT_ABI == ABI_DARWIN
&& TARGET_64BIT)
darwin_one_byte_bool = 1;
if (TARGET_64BIT && ! TARGET_POWERPC64)
{
rs6000_isa_flags |= OPTION_MASK_POWERPC64;
warning (0, "%qs requires PowerPC64 architecture, enabling", "-m64");
}
if (flag_mkernel)
{
rs6000_default_long_calls = 1;
rs6000_isa_flags |= OPTION_MASK_SOFT_FLOAT;
}
if (!flag_mkernel && !flag_apple_kext
&& TARGET_64BIT
&& ! (rs6000_isa_flags_explicit & OPTION_MASK_ALTIVEC))
rs6000_isa_flags |= OPTION_MASK_ALTIVEC;
if (!flag_mkernel
&& !flag_apple_kext
&& strverscmp (darwin_macosx_version_min, "10.5") >= 0
&& ! (rs6000_isa_flags_explicit & OPTION_MASK_ALTIVEC)
&& ! global_options_set.x_rs6000_cpu_index)
{
rs6000_isa_flags |= OPTION_MASK_ALTIVEC;
}
}
#endif
#ifndef RS6000_DEFAULT_LONG_DOUBLE_SIZE
#define RS6000_DEFAULT_LONG_DOUBLE_SIZE 64
#endif
HOST_WIDE_INT
rs6000_builtin_mask_calculate (void)
{
return (((TARGET_ALTIVEC)		    ? RS6000_BTM_ALTIVEC   : 0)
| ((TARGET_CMPB)		    ? RS6000_BTM_CMPB	   : 0)
| ((TARGET_VSX)		    ? RS6000_BTM_VSX	   : 0)
| ((TARGET_PAIRED_FLOAT)	    ? RS6000_BTM_PAIRED	   : 0)
| ((TARGET_FRE)		    ? RS6000_BTM_FRE	   : 0)
| ((TARGET_FRES)		    ? RS6000_BTM_FRES	   : 0)
| ((TARGET_FRSQRTE)		    ? RS6000_BTM_FRSQRTE   : 0)
| ((TARGET_FRSQRTES)		    ? RS6000_BTM_FRSQRTES  : 0)
| ((TARGET_POPCNTD)		    ? RS6000_BTM_POPCNTD   : 0)
| ((rs6000_cpu == PROCESSOR_CELL) ? RS6000_BTM_CELL      : 0)
| ((TARGET_P8_VECTOR)		    ? RS6000_BTM_P8_VECTOR : 0)
| ((TARGET_P9_VECTOR)		    ? RS6000_BTM_P9_VECTOR : 0)
| ((TARGET_P9_MISC)		    ? RS6000_BTM_P9_MISC   : 0)
| ((TARGET_MODULO)		    ? RS6000_BTM_MODULO    : 0)
| ((TARGET_64BIT)		    ? RS6000_BTM_64BIT     : 0)
| ((TARGET_POWERPC64)		    ? RS6000_BTM_POWERPC64 : 0)
| ((TARGET_CRYPTO)		    ? RS6000_BTM_CRYPTO	   : 0)
| ((TARGET_HTM)		    ? RS6000_BTM_HTM	   : 0)
| ((TARGET_DFP)		    ? RS6000_BTM_DFP	   : 0)
| ((TARGET_HARD_FLOAT)	    ? RS6000_BTM_HARD_FLOAT : 0)
| ((TARGET_LONG_DOUBLE_128
&& TARGET_HARD_FLOAT
&& !TARGET_IEEEQUAD)	    ? RS6000_BTM_LDBL128   : 0)
| ((TARGET_FLOAT128_TYPE)	    ? RS6000_BTM_FLOAT128  : 0)
| ((TARGET_FLOAT128_HW)	    ? RS6000_BTM_FLOAT128_HW : 0));
}
static rtx_insn *
rs6000_md_asm_adjust (vec<rtx> &, vec<rtx> &,
vec<const char *> &,
vec<rtx> &clobbers, HARD_REG_SET &clobbered_regs)
{
clobbers.safe_push (gen_rtx_REG (SImode, CA_REGNO));
SET_HARD_REG_BIT (clobbered_regs, CA_REGNO);
return NULL;
}
static bool
rs6000_option_override_internal (bool global_init_p)
{
bool ret = true;
HOST_WIDE_INT set_masks;
HOST_WIDE_INT ignore_masks;
int cpu_index = -1;
int tune_index;
struct cl_target_option *main_target_opt
= ((global_init_p || target_option_default_node == NULL)
? NULL : TREE_TARGET_OPTION (target_option_default_node));
if ((TARGET_DEBUG_REG || TARGET_DEBUG_TARGET) && global_init_p)
rs6000_print_isa_options (stderr, 0, "TARGET_DEFAULT", TARGET_DEFAULT);
if (global_init_p)
rs6000_isa_flags_explicit = global_options_set.x_rs6000_isa_flags;
if (global_init_p
&& rs6000_altivec_element_order == 2)
warning (0, "%qs command-line option is deprecated",
"-maltivec=be");
if (global_options_set.x_rs6000_alignment_flags
&& rs6000_alignment_flags == MASK_ALIGN_POWER
&& DEFAULT_ABI == ABI_DARWIN
&& TARGET_64BIT)
warning (0, "%qs is not supported for 64-bit Darwin;"
" it is incompatible with the installed C and C++ libraries",
"-malign-power");
if (optimize >= 3 && global_init_p
&& !global_options_set.x_flag_ira_loop_pressure)
flag_ira_loop_pressure = 1;
if (flag_sanitize & SANITIZE_USER_ADDRESS
&& !global_options_set.x_flag_asynchronous_unwind_tables)
flag_asynchronous_unwind_tables = 1;
if (TARGET_64BIT)
{
rs6000_pmode = DImode;
rs6000_pointer_size = 64;
}
else
{
rs6000_pmode = SImode;
rs6000_pointer_size = 32;
}
set_masks = POWERPC_MASKS;
#ifdef OS_MISSING_POWERPC64
if (OS_MISSING_POWERPC64)
set_masks &= ~OPTION_MASK_POWERPC64;
#endif
#ifdef OS_MISSING_ALTIVEC
if (OS_MISSING_ALTIVEC)
set_masks &= ~(OPTION_MASK_ALTIVEC | OPTION_MASK_VSX
| OTHER_VSX_VECTOR_MASKS);
#endif
set_masks &= ~rs6000_isa_flags_explicit;
if (rs6000_cpu_index >= 0)
cpu_index = rs6000_cpu_index;
else if (main_target_opt != NULL && main_target_opt->x_rs6000_cpu_index >= 0)
cpu_index = main_target_opt->x_rs6000_cpu_index;
else if (OPTION_TARGET_CPU_DEFAULT)
cpu_index = rs6000_cpu_name_lookup (OPTION_TARGET_CPU_DEFAULT);
if (cpu_index >= 0)
{
const char *unavailable_cpu = NULL;
switch (processor_target_table[cpu_index].processor)
{
#ifndef HAVE_AS_POWER9
case PROCESSOR_POWER9:
unavailable_cpu = "power9";
break;
#endif
#ifndef HAVE_AS_POWER8
case PROCESSOR_POWER8:
unavailable_cpu = "power8";
break;
#endif
#ifndef HAVE_AS_POPCNTD
case PROCESSOR_POWER7:
unavailable_cpu = "power7";
break;
#endif
#ifndef HAVE_AS_DFP
case PROCESSOR_POWER6:
unavailable_cpu = "power6";
break;
#endif
#ifndef HAVE_AS_POPCNTB
case PROCESSOR_POWER5:
unavailable_cpu = "power5";
break;
#endif
default:
break;
}
if (unavailable_cpu)
{
cpu_index = -1;
warning (0, "will not generate %qs instructions because "
"assembler lacks %qs support", unavailable_cpu,
unavailable_cpu);
}
}
if (cpu_index >= 0)
{
rs6000_cpu_index = cpu_index;
rs6000_isa_flags &= ~set_masks;
rs6000_isa_flags |= (processor_target_table[cpu_index].target_enable
& set_masks);
}
else
{
HOST_WIDE_INT flags;
if (TARGET_DEFAULT)
flags = TARGET_DEFAULT;
else
{
const char *default_cpu = (!TARGET_POWERPC64
? "powerpc"
: (BYTES_BIG_ENDIAN
? "powerpc64"
: "powerpc64le"));
int default_cpu_index = rs6000_cpu_name_lookup (default_cpu);
flags = processor_target_table[default_cpu_index].target_enable;
}
rs6000_isa_flags |= (flags & ~rs6000_isa_flags_explicit);
}
if (rs6000_tune_index >= 0)
tune_index = rs6000_tune_index;
else if (cpu_index >= 0)
rs6000_tune_index = tune_index = cpu_index;
else
{
size_t i;
enum processor_type tune_proc
= (TARGET_POWERPC64 ? PROCESSOR_DEFAULT64 : PROCESSOR_DEFAULT);
tune_index = -1;
for (i = 0; i < ARRAY_SIZE (processor_target_table); i++)
if (processor_target_table[i].processor == tune_proc)
{
tune_index = i;
break;
}
}
if (cpu_index >= 0)
rs6000_cpu = processor_target_table[cpu_index].processor;
else
rs6000_cpu = TARGET_POWERPC64 ? PROCESSOR_DEFAULT64 : PROCESSOR_DEFAULT;
gcc_assert (tune_index >= 0);
rs6000_tune = processor_target_table[tune_index].processor;
if (rs6000_cpu == PROCESSOR_PPCE300C2 || rs6000_cpu == PROCESSOR_PPCE300C3
|| rs6000_cpu == PROCESSOR_PPCE500MC || rs6000_cpu == PROCESSOR_PPCE500MC64
|| rs6000_cpu == PROCESSOR_PPCE5500)
{
if (TARGET_ALTIVEC)
error ("AltiVec not supported in this target");
}
if (BYTES_BIG_ENDIAN && optimize_size)
rs6000_isa_flags |= ~rs6000_isa_flags_explicit & OPTION_MASK_MULTIPLE;
if (!BYTES_BIG_ENDIAN && rs6000_cpu != PROCESSOR_PPC750 && TARGET_MULTIPLE)
{
rs6000_isa_flags &= ~OPTION_MASK_MULTIPLE;
if ((rs6000_isa_flags_explicit & OPTION_MASK_MULTIPLE) != 0)
warning (0, "%qs is not supported on little endian systems",
"-mmultiple");
}
if (!BYTES_BIG_ENDIAN
&& !(processor_target_table[tune_index].target_enable & OPTION_MASK_HTM))
rs6000_isa_flags |= ~rs6000_isa_flags_explicit & OPTION_MASK_STRICT_ALIGN;
if (rs6000_altivec_element_order != 0)
rs6000_isa_flags |= OPTION_MASK_ALTIVEC;
if (BYTES_BIG_ENDIAN && rs6000_altivec_element_order == 1)
{
warning (0, N_("-maltivec=le not allowed for big-endian targets"));
rs6000_altivec_element_order = 0;
}
if (!rs6000_fold_gimple)
fprintf (stderr,
"gimple folding of rs6000 builtins has been disabled.\n");
if (TARGET_VSX)
{
const char *msg = NULL;
if (!TARGET_HARD_FLOAT || !TARGET_SINGLE_FLOAT || !TARGET_DOUBLE_FLOAT)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_VSX)
msg = N_("-mvsx requires hardware floating point");
else
{
rs6000_isa_flags &= ~ OPTION_MASK_VSX;
rs6000_isa_flags_explicit |= OPTION_MASK_VSX;
}
}
else if (TARGET_PAIRED_FLOAT)
msg = N_("-mvsx and -mpaired are incompatible");
else if (TARGET_AVOID_XFORM > 0)
msg = N_("-mvsx needs indexed addressing");
else if (!TARGET_ALTIVEC && (rs6000_isa_flags_explicit
& OPTION_MASK_ALTIVEC))
{
if (rs6000_isa_flags_explicit & OPTION_MASK_VSX)
msg = N_("-mvsx and -mno-altivec are incompatible");
else
msg = N_("-mno-altivec disables vsx");
}
if (msg)
{
warning (0, msg);
rs6000_isa_flags &= ~ OPTION_MASK_VSX;
rs6000_isa_flags_explicit |= OPTION_MASK_VSX;
}
}
if ((!TARGET_HARD_FLOAT || !TARGET_ALTIVEC || !TARGET_VSX)
&& (rs6000_isa_flags_explicit & (OPTION_MASK_SOFT_FLOAT
| OPTION_MASK_ALTIVEC
| OPTION_MASK_VSX)) != 0)
rs6000_isa_flags &= ~((OPTION_MASK_P8_VECTOR | OPTION_MASK_CRYPTO
| OPTION_MASK_DIRECT_MOVE)
& ~rs6000_isa_flags_explicit);
if (TARGET_DEBUG_REG || TARGET_DEBUG_TARGET)
rs6000_print_isa_options (stderr, 0, "before defaults", rs6000_isa_flags);
ignore_masks = rs6000_disable_incompatible_switches ();
if (TARGET_P9_VECTOR || TARGET_MODULO || TARGET_P9_MISC)
rs6000_isa_flags |= (ISA_3_0_MASKS_SERVER & ~ignore_masks);
else if (TARGET_P9_MINMAX)
{
if (cpu_index >= 0)
{
if (cpu_index == PROCESSOR_POWER9)
{
rs6000_isa_flags |= (ISA_3_0_MASKS_SERVER & ~ignore_masks);
}
else
error ("power9 target option is incompatible with %<%s=<xxx>%> "
"for <xxx> less than power9", "-mcpu");
}
else if ((ISA_3_0_MASKS_SERVER & rs6000_isa_flags_explicit)
!= (ISA_3_0_MASKS_SERVER & rs6000_isa_flags
& rs6000_isa_flags_explicit))
error ("%qs incompatible with explicitly disabled options",
"-mpower9-minmax");
else
rs6000_isa_flags |= ISA_3_0_MASKS_SERVER;
}
else if (TARGET_P8_VECTOR || TARGET_DIRECT_MOVE || TARGET_CRYPTO)
rs6000_isa_flags |= (ISA_2_7_MASKS_SERVER & ~ignore_masks);
else if (TARGET_VSX)
rs6000_isa_flags |= (ISA_2_6_MASKS_SERVER & ~ignore_masks);
else if (TARGET_POPCNTD)
rs6000_isa_flags |= (ISA_2_6_MASKS_EMBEDDED & ~ignore_masks);
else if (TARGET_DFP)
rs6000_isa_flags |= (ISA_2_5_MASKS_SERVER & ~ignore_masks);
else if (TARGET_CMPB)
rs6000_isa_flags |= (ISA_2_5_MASKS_EMBEDDED & ~ignore_masks);
else if (TARGET_FPRND)
rs6000_isa_flags |= (ISA_2_4_MASKS & ~ignore_masks);
else if (TARGET_POPCNTB)
rs6000_isa_flags |= (ISA_2_2_MASKS & ~ignore_masks);
else if (TARGET_ALTIVEC)
rs6000_isa_flags |= (OPTION_MASK_PPC_GFXOPT & ~ignore_masks);
if (TARGET_CRYPTO && !TARGET_ALTIVEC)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_CRYPTO)
error ("%qs requires %qs", "-mcrypto", "-maltivec");
rs6000_isa_flags &= ~OPTION_MASK_CRYPTO;
}
if (TARGET_DIRECT_MOVE && !TARGET_VSX)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_DIRECT_MOVE)
error ("%qs requires %qs", "-mdirect-move", "-mvsx");
rs6000_isa_flags &= ~OPTION_MASK_DIRECT_MOVE;
}
if (TARGET_P8_VECTOR && !TARGET_ALTIVEC)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_P8_VECTOR)
error ("%qs requires %qs", "-mpower8-vector", "-maltivec");
rs6000_isa_flags &= ~OPTION_MASK_P8_VECTOR;
}
if (TARGET_P8_VECTOR && !TARGET_VSX)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_P8_VECTOR)
&& (rs6000_isa_flags_explicit & OPTION_MASK_VSX))
error ("%qs requires %qs", "-mpower8-vector", "-mvsx");
else if ((rs6000_isa_flags_explicit & OPTION_MASK_P8_VECTOR) == 0)
{
rs6000_isa_flags &= ~OPTION_MASK_P8_VECTOR;
if (rs6000_isa_flags_explicit & OPTION_MASK_VSX)
rs6000_isa_flags_explicit |= OPTION_MASK_P8_VECTOR;
}
else
{
rs6000_isa_flags |= OPTION_MASK_VSX;
rs6000_isa_flags_explicit |= OPTION_MASK_VSX;
}
}
if (TARGET_DFP && !TARGET_HARD_FLOAT)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_DFP)
error ("%qs requires %qs", "-mhard-dfp", "-mhard-float");
rs6000_isa_flags &= ~OPTION_MASK_DFP;
}
if ((TARGET_QUAD_MEMORY || TARGET_QUAD_MEMORY_ATOMIC) && !TARGET_POWERPC64)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_QUAD_MEMORY) != 0)
warning (0, N_("-mquad-memory requires 64-bit mode"));
if ((rs6000_isa_flags_explicit & OPTION_MASK_QUAD_MEMORY_ATOMIC) != 0)
warning (0, N_("-mquad-memory-atomic requires 64-bit mode"));
rs6000_isa_flags &= ~(OPTION_MASK_QUAD_MEMORY
| OPTION_MASK_QUAD_MEMORY_ATOMIC);
}
if (TARGET_QUAD_MEMORY && !WORDS_BIG_ENDIAN)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_QUAD_MEMORY) != 0)
warning (0, N_("-mquad-memory is not available in little endian "
"mode"));
rs6000_isa_flags &= ~OPTION_MASK_QUAD_MEMORY;
}
if (TARGET_QUAD_MEMORY
&& !TARGET_QUAD_MEMORY_ATOMIC
&& ((rs6000_isa_flags_explicit & OPTION_MASK_QUAD_MEMORY_ATOMIC) == 0))
rs6000_isa_flags |= OPTION_MASK_QUAD_MEMORY_ATOMIC;
if ((rs6000_isa_flags_explicit & OPTION_MASK_SAVE_TOC_INDIRECT) == 0
&& flag_shrink_wrap_separate
&& optimize_function_for_speed_p (cfun))
rs6000_isa_flags |= OPTION_MASK_SAVE_TOC_INDIRECT;
if (!(rs6000_isa_flags_explicit & OPTION_MASK_P8_FUSION))
rs6000_isa_flags |= (processor_target_table[tune_index].target_enable
& OPTION_MASK_P8_FUSION);
if (!TARGET_P8_FUSION && (TARGET_P8_FUSION_SIGN || TARGET_TOC_FUSION))
{
if (rs6000_isa_flags_explicit & OPTION_MASK_P8_FUSION)
{
if (TARGET_P8_FUSION_SIGN)
error ("%qs requires %qs", "-mpower8-fusion-sign",
"-mpower8-fusion");
if (TARGET_TOC_FUSION)
error ("%qs requires %qs", "-mtoc-fusion", "-mpower8-fusion");
rs6000_isa_flags &= ~OPTION_MASK_P8_FUSION;
}
else
rs6000_isa_flags |= OPTION_MASK_P8_FUSION;
}
if (TARGET_P9_FUSION && !TARGET_P8_FUSION)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_P8_FUSION)
{
error ("%qs requires %qs", "-mpower9-fusion", "-mpower8-fusion");
rs6000_isa_flags &= ~OPTION_MASK_P9_FUSION;
}
else
rs6000_isa_flags |= OPTION_MASK_P8_FUSION;
}
if (!(rs6000_isa_flags_explicit & OPTION_MASK_P9_FUSION))
rs6000_isa_flags |= (processor_target_table[tune_index].target_enable
& OPTION_MASK_P9_FUSION);
if (TARGET_P8_FUSION
&& !(rs6000_isa_flags_explicit & OPTION_MASK_P8_FUSION_SIGN)
&& optimize_function_for_speed_p (cfun)
&& optimize >= 3)
rs6000_isa_flags |= OPTION_MASK_P8_FUSION_SIGN;
if (TARGET_TOC_FUSION && !TARGET_POWERPC64)
{
rs6000_isa_flags &= ~OPTION_MASK_TOC_FUSION;
if ((rs6000_isa_flags_explicit & OPTION_MASK_TOC_FUSION) != 0)
warning (0, N_("-mtoc-fusion requires 64-bit"));
}
if (TARGET_TOC_FUSION && (TARGET_CMODEL == CMODEL_SMALL))
{
rs6000_isa_flags &= ~OPTION_MASK_TOC_FUSION;
if ((rs6000_isa_flags_explicit & OPTION_MASK_TOC_FUSION) != 0)
warning (0, N_("-mtoc-fusion requires medium/large code model"));
}
if (TARGET_P8_FUSION && !TARGET_TOC_FUSION && TARGET_POWERPC64
&& (TARGET_CMODEL != CMODEL_SMALL)
&& !(rs6000_isa_flags_explicit & OPTION_MASK_TOC_FUSION))
rs6000_isa_flags |= OPTION_MASK_TOC_FUSION;
if (TARGET_P9_VECTOR && !TARGET_P8_VECTOR)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_P9_VECTOR) &&
(rs6000_isa_flags_explicit & OPTION_MASK_P8_VECTOR))
error ("%qs requires %qs", "-mpower9-vector", "-mpower8-vector");
else if ((rs6000_isa_flags_explicit & OPTION_MASK_P9_VECTOR) == 0)
{
rs6000_isa_flags &= ~OPTION_MASK_P9_VECTOR;
if (rs6000_isa_flags_explicit & OPTION_MASK_P8_VECTOR)
rs6000_isa_flags_explicit |= OPTION_MASK_P9_VECTOR;
}
else
{
rs6000_isa_flags |= OPTION_MASK_P8_VECTOR;
rs6000_isa_flags_explicit |= OPTION_MASK_P8_VECTOR;
}
}
if (TARGET_ALLOW_MOVMISALIGN == -1 && TARGET_P8_VECTOR && TARGET_DIRECT_MOVE)
TARGET_ALLOW_MOVMISALIGN = 1;
else if (TARGET_ALLOW_MOVMISALIGN && !TARGET_VSX)
{
if (TARGET_ALLOW_MOVMISALIGN > 0
&& global_options_set.x_TARGET_ALLOW_MOVMISALIGN)
error ("%qs requires %qs", "-mallow-movmisalign", "-mvsx");
TARGET_ALLOW_MOVMISALIGN = 0;
}
if (TARGET_EFFICIENT_UNALIGNED_VSX)
{
if (!TARGET_VSX)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_EFFICIENT_UNALIGNED_VSX)
error ("%qs requires %qs", "-mefficient-unaligned-vsx", "-mvsx");
rs6000_isa_flags &= ~OPTION_MASK_EFFICIENT_UNALIGNED_VSX;
}
else if (!TARGET_ALLOW_MOVMISALIGN)
{
if (rs6000_isa_flags_explicit & OPTION_MASK_EFFICIENT_UNALIGNED_VSX)
error ("%qs requires %qs", "-munefficient-unaligned-vsx",
"-mallow-movmisalign");
rs6000_isa_flags &= ~OPTION_MASK_EFFICIENT_UNALIGNED_VSX;
}
}
int default_long_double_size = (RS6000_DEFAULT_LONG_DOUBLE_SIZE == 64
? 64
: FLOAT_PRECISION_TFmode);
if (!global_options_set.x_rs6000_long_double_type_size)
{
if (main_target_opt != NULL
&& (main_target_opt->x_rs6000_long_double_type_size
!= default_long_double_size))
error ("target attribute or pragma changes long double size");
else
rs6000_long_double_type_size = default_long_double_size;
}
else if (rs6000_long_double_type_size == 128)
rs6000_long_double_type_size = FLOAT_PRECISION_TFmode;
else if (global_options_set.x_rs6000_ieeequad)
{
if (global_options.x_rs6000_ieeequad)
error ("%qs requires %qs", "-mabi=ieeelongdouble", "-mlong-double-128");
else
error ("%qs requires %qs", "-mabi=ibmlongdouble", "-mlong-double-128");
}
if (!global_options_set.x_rs6000_ieeequad)
rs6000_ieeequad = TARGET_IEEEQUAD_DEFAULT;
else
{
if (global_options.x_rs6000_ieeequad
&& (!TARGET_POPCNTD || !TARGET_VSX))
error ("%qs requires full ISA 2.06 support", "-mabi=ieeelongdouble");
if (rs6000_ieeequad != TARGET_IEEEQUAD_DEFAULT && TARGET_LONG_DOUBLE_128)
{
static bool warned_change_long_double;
if (!warned_change_long_double)
{
warned_change_long_double = true;
if (TARGET_IEEEQUAD)
warning (OPT_Wpsabi, "Using IEEE extended precision long double");
else
warning (OPT_Wpsabi, "Using IBM extended precision long double");
}
}
}
TARGET_FLOAT128_TYPE = TARGET_FLOAT128_ENABLE_TYPE && TARGET_VSX;
if (TARGET_FLOAT128_KEYWORD)
{
if (!TARGET_VSX)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_FLOAT128_KEYWORD) != 0)
error ("%qs requires VSX support", "-mfloat128");
TARGET_FLOAT128_TYPE = 0;
rs6000_isa_flags &= ~(OPTION_MASK_FLOAT128_KEYWORD
| OPTION_MASK_FLOAT128_HW);
}
else if (!TARGET_FLOAT128_TYPE)
{
TARGET_FLOAT128_TYPE = 1;
warning (0, "The -mfloat128 option may not be fully supported");
}
}
if (TARGET_FLOAT128_TYPE && !TARGET_FLOAT128_KEYWORD
&& (rs6000_isa_flags_explicit & OPTION_MASK_FLOAT128_KEYWORD) == 0)
rs6000_isa_flags |= OPTION_MASK_FLOAT128_KEYWORD;
if (TARGET_FLOAT128_TYPE && !TARGET_FLOAT128_HW && TARGET_64BIT
&& (rs6000_isa_flags & ISA_3_0_MASKS_IEEE) == ISA_3_0_MASKS_IEEE
&& !(rs6000_isa_flags_explicit & OPTION_MASK_FLOAT128_HW))
rs6000_isa_flags |= OPTION_MASK_FLOAT128_HW;
if (TARGET_FLOAT128_HW
&& (rs6000_isa_flags & ISA_3_0_MASKS_IEEE) != ISA_3_0_MASKS_IEEE)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_FLOAT128_HW) != 0)
error ("%qs requires full ISA 3.0 support", "-mfloat128-hardware");
rs6000_isa_flags &= ~OPTION_MASK_FLOAT128_HW;
}
if (TARGET_FLOAT128_HW && !TARGET_64BIT)
{
if ((rs6000_isa_flags_explicit & OPTION_MASK_FLOAT128_HW) != 0)
error ("%qs requires %qs", "-mfloat128-hardware", "-m64");
rs6000_isa_flags &= ~OPTION_MASK_FLOAT128_HW;
}
if (TARGET_DEBUG_REG || TARGET_DEBUG_TARGET)
rs6000_print_isa_options (stderr, 0, "after defaults", rs6000_isa_flags);
if (rs6000_block_move_inline_limit == 0
&& (rs6000_tune == PROCESSOR_PPCE500MC
|| rs6000_tune == PROCESSOR_PPCE500MC64
|| rs6000_tune == PROCESSOR_PPCE5500
|| rs6000_tune == PROCESSOR_PPCE6500))
rs6000_block_move_inline_limit = 128;
if (rs6000_block_move_inline_limit < (TARGET_POWERPC64 ? 64 : 32))
rs6000_block_move_inline_limit = (TARGET_POWERPC64 ? 64 : 32);
if (global_init_p)
{
if (TARGET_DEBUG_COST)
{
targetm.rtx_costs = rs6000_debug_rtx_costs;
targetm.address_cost = rs6000_debug_address_cost;
targetm.sched.adjust_cost = rs6000_debug_adjust_cost;
}
if (TARGET_DEBUG_ADDR)
{
targetm.legitimate_address_p = rs6000_debug_legitimate_address_p;
targetm.legitimize_address = rs6000_debug_legitimize_address;
rs6000_secondary_reload_class_ptr
= rs6000_debug_secondary_reload_class;
targetm.secondary_memory_needed
= rs6000_debug_secondary_memory_needed;
targetm.can_change_mode_class
= rs6000_debug_can_change_mode_class;
rs6000_preferred_reload_class_ptr
= rs6000_debug_preferred_reload_class;
rs6000_legitimize_reload_address_ptr
= rs6000_debug_legitimize_reload_address;
rs6000_mode_dependent_address_ptr
= rs6000_debug_mode_dependent_address;
}
if (rs6000_veclibabi_name)
{
if (strcmp (rs6000_veclibabi_name, "mass") == 0)
rs6000_veclib_handler = rs6000_builtin_vectorized_libmass;
else
{
error ("unknown vectorization library ABI type (%qs) for "
"%qs switch", rs6000_veclibabi_name, "-mveclibabi=");
ret = false;
}
}
}
if (main_target_opt != NULL && !main_target_opt->x_rs6000_altivec_abi)
{
TARGET_FLOAT128_TYPE = 0;
rs6000_isa_flags &= ~((OPTION_MASK_VSX | OPTION_MASK_ALTIVEC
| OPTION_MASK_FLOAT128_KEYWORD)
& ~rs6000_isa_flags_explicit);
}
if (TARGET_XCOFF && (TARGET_ALTIVEC || TARGET_VSX))
{
if (main_target_opt != NULL && !main_target_opt->x_rs6000_altivec_abi)
error ("target attribute or pragma changes AltiVec ABI");
else
rs6000_altivec_abi = 1;
}
if (TARGET_ELF)
{
if (!global_options_set.x_rs6000_altivec_abi
&& (TARGET_64BIT || TARGET_ALTIVEC || TARGET_VSX))
{
if (main_target_opt != NULL &&
!main_target_opt->x_rs6000_altivec_abi)
error ("target attribute or pragma changes AltiVec ABI");
else
rs6000_altivec_abi = 1;
}
}
if (TARGET_MACHO
&& DEFAULT_ABI == ABI_DARWIN 
&& TARGET_64BIT)
{
if (main_target_opt != NULL && !main_target_opt->x_rs6000_darwin64_abi)
error ("target attribute or pragma changes darwin64 ABI");
else
{
rs6000_darwin64_abi = 1;
rs6000_alignment_flags = MASK_ALIGN_NATURAL;
}
}
if (flag_section_anchors
&& !global_options_set.x_TARGET_NO_FP_IN_TOC)
TARGET_NO_FP_IN_TOC = 1;
if (TARGET_DEBUG_REG || TARGET_DEBUG_TARGET)
rs6000_print_isa_options (stderr, 0, "before subtarget", rs6000_isa_flags);
#ifdef SUBTARGET_OVERRIDE_OPTIONS
SUBTARGET_OVERRIDE_OPTIONS;
#endif
#ifdef SUBSUBTARGET_OVERRIDE_OPTIONS
SUBSUBTARGET_OVERRIDE_OPTIONS;
#endif
#ifdef SUB3TARGET_OVERRIDE_OPTIONS
SUB3TARGET_OVERRIDE_OPTIONS;
#endif
if (TARGET_DEBUG_REG || TARGET_DEBUG_TARGET)
rs6000_print_isa_options (stderr, 0, "after subtarget", rs6000_isa_flags);
if (main_target_opt)
{
if (main_target_opt->x_rs6000_single_float != rs6000_single_float)
error ("target attribute or pragma changes single precision floating "
"point");
if (main_target_opt->x_rs6000_double_float != rs6000_double_float)
error ("target attribute or pragma changes double precision floating "
"point");
}
rs6000_always_hint = (rs6000_tune != PROCESSOR_POWER4
&& rs6000_tune != PROCESSOR_POWER5
&& rs6000_tune != PROCESSOR_POWER6
&& rs6000_tune != PROCESSOR_POWER7
&& rs6000_tune != PROCESSOR_POWER8
&& rs6000_tune != PROCESSOR_POWER9
&& rs6000_tune != PROCESSOR_PPCA2
&& rs6000_tune != PROCESSOR_CELL
&& rs6000_tune != PROCESSOR_PPC476);
rs6000_sched_groups = (rs6000_tune == PROCESSOR_POWER4
|| rs6000_tune == PROCESSOR_POWER5
|| rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8);
rs6000_align_branch_targets = (rs6000_tune == PROCESSOR_POWER4
|| rs6000_tune == PROCESSOR_POWER5
|| rs6000_tune == PROCESSOR_POWER6
|| rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8
|| rs6000_tune == PROCESSOR_POWER9
|| rs6000_tune == PROCESSOR_PPCE500MC
|| rs6000_tune == PROCESSOR_PPCE500MC64
|| rs6000_tune == PROCESSOR_PPCE5500
|| rs6000_tune == PROCESSOR_PPCE6500);
if (TARGET_ALWAYS_HINT >= 0)
rs6000_always_hint = TARGET_ALWAYS_HINT;
if (TARGET_SCHED_GROUPS >= 0)
rs6000_sched_groups = TARGET_SCHED_GROUPS;
if (TARGET_ALIGN_BRANCH_TARGETS >= 0)
rs6000_align_branch_targets = TARGET_ALIGN_BRANCH_TARGETS;
rs6000_sched_restricted_insns_priority
= (rs6000_sched_groups ? 1 : 0);
rs6000_sched_costly_dep
= (rs6000_sched_groups ? true_store_to_load_dep_costly : no_dep_costly);
if (rs6000_sched_costly_dep_str)
{
if (! strcmp (rs6000_sched_costly_dep_str, "no"))
rs6000_sched_costly_dep = no_dep_costly;
else if (! strcmp (rs6000_sched_costly_dep_str, "all"))
rs6000_sched_costly_dep = all_deps_costly;
else if (! strcmp (rs6000_sched_costly_dep_str, "true_store_to_load"))
rs6000_sched_costly_dep = true_store_to_load_dep_costly;
else if (! strcmp (rs6000_sched_costly_dep_str, "store_to_load"))
rs6000_sched_costly_dep = store_to_load_dep_costly;
else
rs6000_sched_costly_dep = ((enum rs6000_dependence_cost)
atoi (rs6000_sched_costly_dep_str));
}
rs6000_sched_insert_nops
= (rs6000_sched_groups ? sched_finish_regroup_exact : sched_finish_none);
if (rs6000_sched_insert_nops_str)
{
if (! strcmp (rs6000_sched_insert_nops_str, "no"))
rs6000_sched_insert_nops = sched_finish_none;
else if (! strcmp (rs6000_sched_insert_nops_str, "pad"))
rs6000_sched_insert_nops = sched_finish_pad_groups;
else if (! strcmp (rs6000_sched_insert_nops_str, "regroup_exact"))
rs6000_sched_insert_nops = sched_finish_regroup_exact;
else
rs6000_sched_insert_nops = ((enum rs6000_nop_insertion)
atoi (rs6000_sched_insert_nops_str));
}
if (!global_options_set.x_rs6000_stack_protector_guard)
#ifdef TARGET_THREAD_SSP_OFFSET
rs6000_stack_protector_guard = SSP_TLS;
#else
rs6000_stack_protector_guard = SSP_GLOBAL;
#endif
#ifdef TARGET_THREAD_SSP_OFFSET
rs6000_stack_protector_guard_offset = TARGET_THREAD_SSP_OFFSET;
rs6000_stack_protector_guard_reg = TARGET_64BIT ? 13 : 2;
#endif
if (global_options_set.x_rs6000_stack_protector_guard_offset_str)
{
char *endp;
const char *str = rs6000_stack_protector_guard_offset_str;
errno = 0;
long offset = strtol (str, &endp, 0);
if (!*str || *endp || errno)
error ("%qs is not a valid number in %qs", str,
"-mstack-protector-guard-offset=");
if (!IN_RANGE (offset, -0x8000, 0x7fff)
|| (TARGET_64BIT && (offset & 3)))
error ("%qs is not a valid offset in %qs", str,
"-mstack-protector-guard-offset=");
rs6000_stack_protector_guard_offset = offset;
}
if (global_options_set.x_rs6000_stack_protector_guard_reg_str)
{
const char *str = rs6000_stack_protector_guard_reg_str;
int reg = decode_reg_name (str);
if (!IN_RANGE (reg, 1, 31))
error ("%qs is not a valid base register in %qs", str,
"-mstack-protector-guard-reg=");
rs6000_stack_protector_guard_reg = reg;
}
if (rs6000_stack_protector_guard == SSP_TLS
&& !IN_RANGE (rs6000_stack_protector_guard_reg, 1, 31))
error ("%qs needs a valid base register", "-mstack-protector-guard=tls");
if (global_init_p)
{
#ifdef TARGET_REGNAMES
if (TARGET_REGNAMES)
memcpy (rs6000_reg_names, alt_reg_names, sizeof (rs6000_reg_names));
#endif
if (!global_options_set.x_aix_struct_return)
aix_struct_return = (DEFAULT_ABI != ABI_V4 || DRAFT_V4_STRUCT_RET);
#if 0
if (TARGET_XL_COMPAT)
flag_signed_bitfields = 0;
#endif
if (TARGET_LONG_DOUBLE_128 && !TARGET_IEEEQUAD)
REAL_MODE_FORMAT (TFmode) = &ibm_extended_format;
ASM_GENERATE_INTERNAL_LABEL (toc_label_name, "LCTOC", 1);
if (!TARGET_64BIT)
{
targetm.asm_out.aligned_op.di = NULL;
targetm.asm_out.unaligned_op.di = NULL;
}
if (!optimize_size)
{
if (rs6000_tune == PROCESSOR_TITAN
|| rs6000_tune == PROCESSOR_CELL)
{
if (align_functions <= 0)
align_functions = 8;
if (align_jumps <= 0)
align_jumps = 8;
if (align_loops <= 0)
align_loops = 8;
}
if (rs6000_align_branch_targets)
{
if (align_functions <= 0)
align_functions = 16;
if (align_jumps <= 0)
align_jumps = 16;
if (align_loops <= 0)
{
can_override_loop_align = 1;
align_loops = 16;
}
}
if (align_jumps_max_skip <= 0)
align_jumps_max_skip = 15;
if (align_loops_max_skip <= 0)
align_loops_max_skip = 15;
}
init_machine_status = rs6000_init_machine_status;
if (DEFAULT_ABI == ABI_V4 || DEFAULT_ABI == ABI_DARWIN)
targetm.calls.split_complex_arg = NULL;
if (DEFAULT_ABI == ABI_AIX)
targetm.calls.custom_function_descriptors = 0;
}
if (optimize_size)
rs6000_cost = TARGET_POWERPC64 ? &size64_cost : &size32_cost;
else
switch (rs6000_tune)
{
case PROCESSOR_RS64A:
rs6000_cost = &rs64a_cost;
break;
case PROCESSOR_MPCCORE:
rs6000_cost = &mpccore_cost;
break;
case PROCESSOR_PPC403:
rs6000_cost = &ppc403_cost;
break;
case PROCESSOR_PPC405:
rs6000_cost = &ppc405_cost;
break;
case PROCESSOR_PPC440:
rs6000_cost = &ppc440_cost;
break;
case PROCESSOR_PPC476:
rs6000_cost = &ppc476_cost;
break;
case PROCESSOR_PPC601:
rs6000_cost = &ppc601_cost;
break;
case PROCESSOR_PPC603:
rs6000_cost = &ppc603_cost;
break;
case PROCESSOR_PPC604:
rs6000_cost = &ppc604_cost;
break;
case PROCESSOR_PPC604e:
rs6000_cost = &ppc604e_cost;
break;
case PROCESSOR_PPC620:
rs6000_cost = &ppc620_cost;
break;
case PROCESSOR_PPC630:
rs6000_cost = &ppc630_cost;
break;
case PROCESSOR_CELL:
rs6000_cost = &ppccell_cost;
break;
case PROCESSOR_PPC750:
case PROCESSOR_PPC7400:
rs6000_cost = &ppc750_cost;
break;
case PROCESSOR_PPC7450:
rs6000_cost = &ppc7450_cost;
break;
case PROCESSOR_PPC8540:
case PROCESSOR_PPC8548:
rs6000_cost = &ppc8540_cost;
break;
case PROCESSOR_PPCE300C2:
case PROCESSOR_PPCE300C3:
rs6000_cost = &ppce300c2c3_cost;
break;
case PROCESSOR_PPCE500MC:
rs6000_cost = &ppce500mc_cost;
break;
case PROCESSOR_PPCE500MC64:
rs6000_cost = &ppce500mc64_cost;
break;
case PROCESSOR_PPCE5500:
rs6000_cost = &ppce5500_cost;
break;
case PROCESSOR_PPCE6500:
rs6000_cost = &ppce6500_cost;
break;
case PROCESSOR_TITAN:
rs6000_cost = &titan_cost;
break;
case PROCESSOR_POWER4:
case PROCESSOR_POWER5:
rs6000_cost = &power4_cost;
break;
case PROCESSOR_POWER6:
rs6000_cost = &power6_cost;
break;
case PROCESSOR_POWER7:
rs6000_cost = &power7_cost;
break;
case PROCESSOR_POWER8:
rs6000_cost = &power8_cost;
break;
case PROCESSOR_POWER9:
rs6000_cost = &power9_cost;
break;
case PROCESSOR_PPCA2:
rs6000_cost = &ppca2_cost;
break;
default:
gcc_unreachable ();
}
if (global_init_p)
{
maybe_set_param_value (PARAM_SIMULTANEOUS_PREFETCHES,
rs6000_cost->simultaneous_prefetches,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_L1_CACHE_SIZE, rs6000_cost->l1_cache_size,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_L1_CACHE_LINE_SIZE,
rs6000_cost->cache_line_size,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_L2_CACHE_SIZE, rs6000_cost->l2_cache_size,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_MAX_PEELED_INSNS, 400,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_MAX_COMPLETELY_PEELED_INSNS, 400,
global_options.x_param_values,
global_options_set.x_param_values);
maybe_set_param_value (PARAM_SCHED_PRESSURE_ALGORITHM,
SCHED_PRESSURE_MODEL,
global_options.x_param_values,
global_options_set.x_param_values);
if (DEFAULT_ABI != ABI_V4)
targetm.expand_builtin_va_start = NULL;
}
if (TARGET_HARD_FLOAT && rs6000_single_float == 0 && rs6000_double_float == 0)
rs6000_single_float = rs6000_double_float = 1;
if (TARGET_AVOID_XFORM == -1)
TARGET_AVOID_XFORM = (rs6000_tune == PROCESSOR_POWER6 && TARGET_CMPB
&& !TARGET_ALTIVEC);
if (rs6000_recip_name)
{
char *p = ASTRDUP (rs6000_recip_name);
char *q;
unsigned int mask, i;
bool invert;
while ((q = strtok (p, ",")) != NULL)
{
p = NULL;
if (*q == '!')
{
invert = true;
q++;
}
else
invert = false;
if (!strcmp (q, "default"))
mask = ((TARGET_RECIP_PRECISION)
? RECIP_HIGH_PRECISION : RECIP_LOW_PRECISION);
else
{
for (i = 0; i < ARRAY_SIZE (recip_options); i++)
if (!strcmp (q, recip_options[i].string))
{
mask = recip_options[i].mask;
break;
}
if (i == ARRAY_SIZE (recip_options))
{
error ("unknown option for %<%s=%s%>", "-mrecip", q);
invert = false;
mask = 0;
ret = false;
}
}
if (invert)
rs6000_recip_control &= ~mask;
else
rs6000_recip_control |= mask;
}
}
rs6000_builtin_mask = rs6000_builtin_mask_calculate ();
if (TARGET_DEBUG_BUILTIN || TARGET_DEBUG_TARGET)
rs6000_print_builtin_options (stderr, 0, "builtin mask",
rs6000_builtin_mask);
rs6000_init_hard_regno_mode_ok (global_init_p);
if (global_init_p)
target_option_default_node = target_option_current_node
= build_target_option_node (&global_options);
if (TARGET_LINK_STACK == -1)
SET_TARGET_LINK_STACK (rs6000_tune == PROCESSOR_PPC476 && flag_pic);
if (!rs6000_speculate_indirect_jumps)
warning (0, "%qs is deprecated and not recommended in any circumstances",
"-mno-speculate-indirect-jumps");
return ret;
}
static void
rs6000_option_override (void)
{
(void) rs6000_option_override_internal (true);
}

static tree
rs6000_builtin_mask_for_load (void)
{
if ((TARGET_ALTIVEC && !TARGET_VSX)
|| (TARGET_VSX && !TARGET_EFFICIENT_UNALIGNED_VSX))
return altivec_builtin_mask_for_load;
else
return 0;
}
int
rs6000_loop_align (rtx label)
{
basic_block bb;
int ninsns;
if (!can_override_loop_align)
return align_loops_log;
bb = BLOCK_FOR_INSN (label);
ninsns = num_loop_insns(bb->loop_father);
if (ninsns > 4 && ninsns <= 8
&& (rs6000_tune == PROCESSOR_POWER4
|| rs6000_tune == PROCESSOR_POWER5
|| rs6000_tune == PROCESSOR_POWER6
|| rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8))
return 5;
else
return align_loops_log;
}
static int
rs6000_loop_align_max_skip (rtx_insn *label)
{
return (1 << rs6000_loop_align (label)) - 1;
}
static bool
rs6000_vector_alignment_reachable (const_tree type ATTRIBUTE_UNUSED, bool is_packed)
{
if (is_packed)
return false;
if (TARGET_32BIT)
{
if (rs6000_alignment_flags == MASK_ALIGN_NATURAL)
return true;
if (rs6000_alignment_flags ==  MASK_ALIGN_POWER)
return true;
return false;
}
else
{
if (TARGET_MACHO)
return false;
return true;
}
}
static bool
rs6000_builtin_support_vector_misalignment (machine_mode mode,
const_tree type,
int misalignment,
bool is_packed)
{
if (TARGET_VSX)
{
if (TARGET_EFFICIENT_UNALIGNED_VSX)
return true;
if (optab_handler (movmisalign_optab, mode) == CODE_FOR_nothing)
return false;
if (misalignment == -1)
{
if (rs6000_vector_alignment_reachable (type, is_packed))
{
int element_size = TREE_INT_CST_LOW (TYPE_SIZE (type));
if (element_size == 64 || element_size == 32)
return true;
}
return false;
}
if (misalignment % 4 == 0)
return true;
}
return false;
}
static int
rs6000_builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype, int misalign)
{
unsigned elements;
tree elem_type;
switch (type_of_cost)
{
case scalar_stmt:
case scalar_load:
case scalar_store:
case vector_stmt:
case vector_load:
case vector_store:
case vec_to_scalar:
case scalar_to_vec:
case cond_branch_not_taken:
return 1;
case vec_perm:
if (TARGET_VSX)
return 3;
else
return 1;
case vec_promote_demote:
if (TARGET_VSX)
return 4;
else
return 1;
case cond_branch_taken:
return 3;
case unaligned_load:
case vector_gather_load:
if (TARGET_EFFICIENT_UNALIGNED_VSX)
return 1;
if (TARGET_VSX && TARGET_ALLOW_MOVMISALIGN)
{
elements = TYPE_VECTOR_SUBPARTS (vectype);
if (elements == 2)
return 2;
if (elements == 4)
{
switch (misalign)
{
case 8:
return 2;
case -1:
case 4:
case 12:
return 22;
default:
gcc_unreachable ();
}
}
}
if (TARGET_ALTIVEC)
gcc_unreachable ();
return 2;
case unaligned_store:
case vector_scatter_store:
if (TARGET_EFFICIENT_UNALIGNED_VSX)
return 1;
if (TARGET_VSX && TARGET_ALLOW_MOVMISALIGN)
{
elements = TYPE_VECTOR_SUBPARTS (vectype);
if (elements == 2)
return 2;
if (elements == 4)
{
switch (misalign)
{
case 8:
return 2;
case -1:
case 4:
case 12:
return 23;
default:
gcc_unreachable ();
}
}
}
if (TARGET_ALTIVEC)
gcc_unreachable ();
return 2;
case vec_construct:
elem_type = TREE_TYPE (vectype);
if (SCALAR_FLOAT_TYPE_P (elem_type)
&& TYPE_PRECISION (elem_type) == 32)
return 5;
else if (INTEGRAL_TYPE_P (elem_type))
{
if (TARGET_P9_VECTOR)
return TYPE_VECTOR_SUBPARTS (vectype) - 1 + 2;
else
return TYPE_VECTOR_SUBPARTS (vectype) - 1 + 5;
}
else
return 2;
default:
gcc_unreachable ();
}
}
static machine_mode
rs6000_preferred_simd_mode (scalar_mode mode)
{
if (TARGET_VSX)
switch (mode)
{
case E_DFmode:
return V2DFmode;
default:;
}
if (TARGET_ALTIVEC || TARGET_VSX)
switch (mode)
{
case E_SFmode:
return V4SFmode;
case E_TImode:
return V1TImode;
case E_DImode:
return V2DImode;
case E_SImode:
return V4SImode;
case E_HImode:
return V8HImode;
case E_QImode:
return V16QImode;
default:;
}
if (TARGET_PAIRED_FLOAT
&& mode == SFmode)
return V2SFmode;
return word_mode;
}
typedef struct _rs6000_cost_data
{
struct loop *loop_info;
unsigned cost[3];
} rs6000_cost_data;
static void
rs6000_density_test (rs6000_cost_data *data)
{
const int DENSITY_PCT_THRESHOLD = 85;
const int DENSITY_SIZE_THRESHOLD = 70;
const int DENSITY_PENALTY = 10;
struct loop *loop = data->loop_info;
basic_block *bbs = get_loop_body (loop);
int nbbs = loop->num_nodes;
int vec_cost = data->cost[vect_body], not_vec_cost = 0;
int i, density_pct;
for (i = 0; i < nbbs; i++)
{
basic_block bb = bbs[i];
gimple_stmt_iterator gsi;
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
if (!STMT_VINFO_RELEVANT_P (stmt_info)
&& !STMT_VINFO_IN_PATTERN_P (stmt_info))
not_vec_cost++;
}
}
free (bbs);
density_pct = (vec_cost * 100) / (vec_cost + not_vec_cost);
if (density_pct > DENSITY_PCT_THRESHOLD
&& vec_cost + not_vec_cost > DENSITY_SIZE_THRESHOLD)
{
data->cost[vect_body] = vec_cost * (100 + DENSITY_PENALTY) / 100;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"density %d%%, cost %d exceeds threshold, penalizing "
"loop body cost by %d%%", density_pct,
vec_cost + not_vec_cost, DENSITY_PENALTY);
}
}
static bool rs6000_vect_nonmem;
static void *
rs6000_init_cost (struct loop *loop_info)
{
rs6000_cost_data *data = XNEW (struct _rs6000_cost_data);
data->loop_info = loop_info;
data->cost[vect_prologue] = 0;
data->cost[vect_body]     = 0;
data->cost[vect_epilogue] = 0;
rs6000_vect_nonmem = false;
return data;
}
static unsigned
rs6000_add_stmt_cost (void *data, int count, enum vect_cost_for_stmt kind,
struct _stmt_vec_info *stmt_info, int misalign,
enum vect_cost_model_location where)
{
rs6000_cost_data *cost_data = (rs6000_cost_data*) data;
unsigned retval = 0;
if (flag_vect_cost_model)
{
tree vectype = stmt_info ? stmt_vectype (stmt_info) : NULL_TREE;
int stmt_cost = rs6000_builtin_vectorization_cost (kind, vectype,
misalign);
if (where == vect_body && stmt_info && stmt_in_inner_loop_p (stmt_info))
count *= 50;  
retval = (unsigned) (count * stmt_cost);
cost_data->cost[where] += retval;
if ((kind == vec_to_scalar || kind == vec_perm
|| kind == vec_promote_demote || kind == vec_construct
|| kind == scalar_to_vec)
|| (where == vect_body && kind == vector_stmt))
rs6000_vect_nonmem = true;
}
return retval;
}
static void
rs6000_finish_cost (void *data, unsigned *prologue_cost,
unsigned *body_cost, unsigned *epilogue_cost)
{
rs6000_cost_data *cost_data = (rs6000_cost_data*) data;
if (cost_data->loop_info)
rs6000_density_test (cost_data);
if (cost_data->loop_info)
{
loop_vec_info vec_info = loop_vec_info_for_loop (cost_data->loop_info);
if (!rs6000_vect_nonmem
&& LOOP_VINFO_VECT_FACTOR (vec_info) == 2
&& LOOP_REQUIRES_VERSIONING (vec_info))
cost_data->cost[vect_body] += 10000;
}
*prologue_cost = cost_data->cost[vect_prologue];
*body_cost     = cost_data->cost[vect_body];
*epilogue_cost = cost_data->cost[vect_epilogue];
}
static void
rs6000_destroy_cost_data (void *data)
{
free (data);
}
static tree
rs6000_builtin_vectorized_libmass (combined_fn fn, tree type_out,
tree type_in)
{
char name[32];
const char *suffix = NULL;
tree fntype, new_fndecl, bdecl = NULL_TREE;
int n_args = 1;
const char *bname;
machine_mode el_mode, in_mode;
int n, in_n;
if (!flag_unsafe_math_optimizations || !TARGET_VSX)
return NULL_TREE;
el_mode = TYPE_MODE (TREE_TYPE (type_out));
n = TYPE_VECTOR_SUBPARTS (type_out);
in_mode = TYPE_MODE (TREE_TYPE (type_in));
in_n = TYPE_VECTOR_SUBPARTS (type_in);
if (el_mode != in_mode
|| n != in_n)
return NULL_TREE;
switch (fn)
{
CASE_CFN_ATAN2:
CASE_CFN_HYPOT:
CASE_CFN_POW:
n_args = 2;
gcc_fallthrough ();
CASE_CFN_ACOS:
CASE_CFN_ACOSH:
CASE_CFN_ASIN:
CASE_CFN_ASINH:
CASE_CFN_ATAN:
CASE_CFN_ATANH:
CASE_CFN_CBRT:
CASE_CFN_COS:
CASE_CFN_COSH:
CASE_CFN_ERF:
CASE_CFN_ERFC:
CASE_CFN_EXP2:
CASE_CFN_EXP:
CASE_CFN_EXPM1:
CASE_CFN_LGAMMA:
CASE_CFN_LOG10:
CASE_CFN_LOG1P:
CASE_CFN_LOG2:
CASE_CFN_LOG:
CASE_CFN_SIN:
CASE_CFN_SINH:
CASE_CFN_SQRT:
CASE_CFN_TAN:
CASE_CFN_TANH:
if (el_mode == DFmode && n == 2)
{
bdecl = mathfn_built_in (double_type_node, fn);
suffix = "d2";				
}
else if (el_mode == SFmode && n == 4)
{
bdecl = mathfn_built_in (float_type_node, fn);
suffix = "4";					
}
else
return NULL_TREE;
if (!bdecl)
return NULL_TREE;
break;
default:
return NULL_TREE;
}
gcc_assert (suffix != NULL);
bname = IDENTIFIER_POINTER (DECL_NAME (bdecl));
if (!bname)
return NULL_TREE;
strcpy (name, bname + sizeof ("__builtin_") - 1);
strcat (name, suffix);
if (n_args == 1)
fntype = build_function_type_list (type_out, type_in, NULL);
else if (n_args == 2)
fntype = build_function_type_list (type_out, type_in, type_in, NULL);
else
gcc_unreachable ();
new_fndecl = build_decl (BUILTINS_LOCATION,
FUNCTION_DECL, get_identifier (name), fntype);
TREE_PUBLIC (new_fndecl) = 1;
DECL_EXTERNAL (new_fndecl) = 1;
DECL_IS_NOVOPS (new_fndecl) = 1;
TREE_READONLY (new_fndecl) = 1;
return new_fndecl;
}
static tree
rs6000_builtin_vectorized_function (unsigned int fn, tree type_out,
tree type_in)
{
machine_mode in_mode, out_mode;
int in_n, out_n;
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin_vectorized_function (%s, %s, %s)\n",
combined_fn_name (combined_fn (fn)),
GET_MODE_NAME (TYPE_MODE (type_out)),
GET_MODE_NAME (TYPE_MODE (type_in)));
if (TREE_CODE (type_out) != VECTOR_TYPE
|| TREE_CODE (type_in) != VECTOR_TYPE)
return NULL_TREE;
out_mode = TYPE_MODE (TREE_TYPE (type_out));
out_n = TYPE_VECTOR_SUBPARTS (type_out);
in_mode = TYPE_MODE (TREE_TYPE (type_in));
in_n = TYPE_VECTOR_SUBPARTS (type_in);
switch (fn)
{
CASE_CFN_COPYSIGN:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_CPSGNDP];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_CPSGNSP];
if (VECTOR_UNIT_ALTIVEC_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_COPYSIGN_V4SF];
break;
CASE_CFN_CEIL:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVRDPIP];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVRSPIP];
if (VECTOR_UNIT_ALTIVEC_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VRFIP];
break;
CASE_CFN_FLOOR:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVRDPIM];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVRSPIM];
if (VECTOR_UNIT_ALTIVEC_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VRFIM];
break;
CASE_CFN_FMA:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVMADDDP];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVMADDSP];
if (VECTOR_UNIT_ALTIVEC_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VMADDFP];
break;
CASE_CFN_TRUNC:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVRDPIZ];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVRSPIZ];
if (VECTOR_UNIT_ALTIVEC_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VRFIZ];
break;
CASE_CFN_NEARBYINT:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& flag_unsafe_math_optimizations
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVRDPI];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& flag_unsafe_math_optimizations
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVRSPI];
break;
CASE_CFN_RINT:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& !flag_trapping_math
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_XVRDPIC];
if (VECTOR_UNIT_VSX_P (V4SFmode)
&& !flag_trapping_math
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[VSX_BUILTIN_XVRSPIC];
break;
default:
break;
}
if (rs6000_veclib_handler)
return rs6000_veclib_handler (combined_fn (fn), type_out, type_in);
return NULL_TREE;
}
static tree
rs6000_builtin_md_vectorized_function (tree fndecl, tree type_out,
tree type_in)
{
machine_mode in_mode, out_mode;
int in_n, out_n;
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin_md_vectorized_function (%s, %s, %s)\n",
IDENTIFIER_POINTER (DECL_NAME (fndecl)),
GET_MODE_NAME (TYPE_MODE (type_out)),
GET_MODE_NAME (TYPE_MODE (type_in)));
if (TREE_CODE (type_out) != VECTOR_TYPE
|| TREE_CODE (type_in) != VECTOR_TYPE)
return NULL_TREE;
out_mode = TYPE_MODE (TREE_TYPE (type_out));
out_n = TYPE_VECTOR_SUBPARTS (type_out);
in_mode = TYPE_MODE (TREE_TYPE (type_in));
in_n = TYPE_VECTOR_SUBPARTS (type_in);
enum rs6000_builtins fn
= (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
switch (fn)
{
case RS6000_BUILTIN_RSQRTF:
if (VECTOR_UNIT_ALTIVEC_OR_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VRSQRTFP];
break;
case RS6000_BUILTIN_RSQRT:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_RSQRT_2DF];
break;
case RS6000_BUILTIN_RECIPF:
if (VECTOR_UNIT_ALTIVEC_OR_VSX_P (V4SFmode)
&& out_mode == SFmode && out_n == 4
&& in_mode == SFmode && in_n == 4)
return rs6000_builtin_decls[ALTIVEC_BUILTIN_VRECIPFP];
break;
case RS6000_BUILTIN_RECIP:
if (VECTOR_UNIT_VSX_P (V2DFmode)
&& out_mode == DFmode && out_n == 2
&& in_mode == DFmode && in_n == 2)
return rs6000_builtin_decls[VSX_BUILTIN_RECIP_V2DF];
break;
default:
break;
}
return NULL_TREE;
}

static const char *rs6000_default_cpu;
static void
rs6000_file_start (void)
{
char buffer[80];
const char *start = buffer;
FILE *file = asm_out_file;
rs6000_default_cpu = TARGET_CPU_DEFAULT;
default_file_start ();
if (flag_verbose_asm)
{
sprintf (buffer, "\n%s rs6000/powerpc options:", ASM_COMMENT_START);
if (rs6000_default_cpu != 0 && rs6000_default_cpu[0] != '\0')
{
fprintf (file, "%s --with-cpu=%s", start, rs6000_default_cpu);
start = "";
}
if (global_options_set.x_rs6000_cpu_index)
{
fprintf (file, "%s -mcpu=%s", start,
processor_target_table[rs6000_cpu_index].name);
start = "";
}
if (global_options_set.x_rs6000_tune_index)
{
fprintf (file, "%s -mtune=%s", start,
processor_target_table[rs6000_tune_index].name);
start = "";
}
if (PPC405_ERRATUM77)
{
fprintf (file, "%s PPC405CR_ERRATUM77", start);
start = "";
}
#ifdef USING_ELFOS_H
switch (rs6000_sdata)
{
case SDATA_NONE: fprintf (file, "%s -msdata=none", start); start = ""; break;
case SDATA_DATA: fprintf (file, "%s -msdata=data", start); start = ""; break;
case SDATA_SYSV: fprintf (file, "%s -msdata=sysv", start); start = ""; break;
case SDATA_EABI: fprintf (file, "%s -msdata=eabi", start); start = ""; break;
}
if (rs6000_sdata && g_switch_value)
{
fprintf (file, "%s -G %d", start,
g_switch_value);
start = "";
}
#endif
if (*start == '\0')
putc ('\n', file);
}
#ifdef USING_ELFOS_H
if (!(rs6000_default_cpu && rs6000_default_cpu[0])
&& !global_options_set.x_rs6000_cpu_index)
{
fputs ("\t.machine ", asm_out_file);
if ((rs6000_isa_flags & OPTION_MASK_MODULO) != 0)
fputs ("power9\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_DIRECT_MOVE) != 0)
fputs ("power8\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_POPCNTD) != 0)
fputs ("power7\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_CMPB) != 0)
fputs ("power6\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_POPCNTB) != 0)
fputs ("power5\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_MFCRF) != 0)
fputs ("power4\n", asm_out_file);
else if ((rs6000_isa_flags & OPTION_MASK_POWERPC64) != 0)
fputs ("ppc64\n", asm_out_file);
else
fputs ("ppc\n", asm_out_file);
}
#endif
if (DEFAULT_ABI == ABI_ELFv2)
fprintf (file, "\t.abiversion 2\n");
}

int
direct_return (void)
{
if (reload_completed)
{
rs6000_stack_t *info = rs6000_stack_info ();
if (info->first_gp_reg_save == 32
&& info->first_fp_reg_save == 64
&& info->first_altivec_reg_save == LAST_ALTIVEC_REGNO + 1
&& ! info->lr_save_p
&& ! info->cr_save_p
&& info->vrsave_size == 0
&& ! info->push_p)
return 1;
}
return 0;
}
int
num_insns_constant_wide (HOST_WIDE_INT value)
{
if (((unsigned HOST_WIDE_INT) value + 0x8000) < 0x10000)
return 1;
else if ((value & 0xffff) == 0
&& (value >> 31 == -1 || value >> 31 == 0))
return 1;
else if (TARGET_POWERPC64)
{
HOST_WIDE_INT low  = ((value & 0xffffffff) ^ 0x80000000) - 0x80000000;
HOST_WIDE_INT high = value >> 31;
if (high == 0 || high == -1)
return 2;
high >>= 1;
if (low == 0)
return num_insns_constant_wide (high) + 1;
else if (high == 0)
return num_insns_constant_wide (low) + 1;
else
return (num_insns_constant_wide (high)
+ num_insns_constant_wide (low) + 1);
}
else
return 2;
}
int
num_insns_constant (rtx op, machine_mode mode)
{
HOST_WIDE_INT low, high;
switch (GET_CODE (op))
{
case CONST_INT:
if ((INTVAL (op) >> 31) != 0 && (INTVAL (op) >> 31) != -1
&& rs6000_is_valid_and_mask (op, mode))
return 2;
else
return num_insns_constant_wide (INTVAL (op));
case CONST_WIDE_INT:
{
int i;
int ins = CONST_WIDE_INT_NUNITS (op) - 1;
for (i = 0; i < CONST_WIDE_INT_NUNITS (op); i++)
ins += num_insns_constant_wide (CONST_WIDE_INT_ELT (op, i));
return ins;
}
case CONST_DOUBLE:
if (mode == SFmode || mode == SDmode)
{
long l;
if (DECIMAL_FLOAT_MODE_P (mode))
REAL_VALUE_TO_TARGET_DECIMAL32
(*CONST_DOUBLE_REAL_VALUE (op), l);
else
REAL_VALUE_TO_TARGET_SINGLE (*CONST_DOUBLE_REAL_VALUE (op), l);
return num_insns_constant_wide ((HOST_WIDE_INT) l);
}
long l[2];
if (DECIMAL_FLOAT_MODE_P (mode))
REAL_VALUE_TO_TARGET_DECIMAL64 (*CONST_DOUBLE_REAL_VALUE (op), l);
else
REAL_VALUE_TO_TARGET_DOUBLE (*CONST_DOUBLE_REAL_VALUE (op), l);
high = l[WORDS_BIG_ENDIAN == 0];
low  = l[WORDS_BIG_ENDIAN != 0];
if (TARGET_32BIT)
return (num_insns_constant_wide (low)
+ num_insns_constant_wide (high));
else
{
if ((high == 0 && low >= 0)
|| (high == -1 && low < 0))
return num_insns_constant_wide (low);
else if (rs6000_is_valid_and_mask (op, mode))
return 2;
else if (low == 0)
return num_insns_constant_wide (high) + 1;
else
return (num_insns_constant_wide (high)
+ num_insns_constant_wide (low) + 1);
}
default:
gcc_unreachable ();
}
}
HOST_WIDE_INT
const_vector_elt_as_int (rtx op, unsigned int elt)
{
rtx tmp;
gcc_assert (GET_MODE (op) != V2DImode
&& GET_MODE (op) != V2DFmode);
tmp = CONST_VECTOR_ELT (op, elt);
if (GET_MODE (op) == V4SFmode
|| GET_MODE (op) == V2SFmode)
tmp = gen_lowpart (SImode, tmp);
return INTVAL (tmp);
}
static bool
vspltis_constant (rtx op, unsigned step, unsigned copies)
{
machine_mode mode = GET_MODE (op);
machine_mode inner = GET_MODE_INNER (mode);
unsigned i;
unsigned nunits;
unsigned bitsize;
unsigned mask;
HOST_WIDE_INT val;
HOST_WIDE_INT splat_val;
HOST_WIDE_INT msb_val;
if (mode == V2DImode || mode == V2DFmode || mode == V1TImode)
return false;
nunits = GET_MODE_NUNITS (mode);
bitsize = GET_MODE_BITSIZE (inner);
mask = GET_MODE_MASK (inner);
val = const_vector_elt_as_int (op, BYTES_BIG_ENDIAN ? nunits - 1 : 0);
splat_val = val;
msb_val = val >= 0 ? 0 : -1;
for (i = 2; i <= copies; i *= 2)
{
HOST_WIDE_INT small_val;
bitsize /= 2;
small_val = splat_val >> bitsize;
mask >>= bitsize;
if (splat_val != ((HOST_WIDE_INT)
((unsigned HOST_WIDE_INT) small_val << bitsize)
| (small_val & mask)))
return false;
splat_val = small_val;
}
if (EASY_VECTOR_15 (splat_val))
;
else if (EASY_VECTOR_15_ADD_SELF (splat_val)
&& (splat_val >= 0 || (step == 1 && copies == 1)))
;
else if (EASY_VECTOR_MSB (splat_val, inner))
;
else
return false;
for (i = 1; i < nunits; ++i)
{
HOST_WIDE_INT desired_val;
unsigned elt = BYTES_BIG_ENDIAN ? nunits - 1 - i : i;
if ((i & (step - 1)) == 0)
desired_val = val;
else
desired_val = msb_val;
if (desired_val != const_vector_elt_as_int (op, elt))
return false;
}
return true;
}
int
vspltis_shifted (rtx op)
{
machine_mode mode = GET_MODE (op);
machine_mode inner = GET_MODE_INNER (mode);
unsigned i, j;
unsigned nunits;
unsigned mask;
HOST_WIDE_INT val;
if (mode != V16QImode && mode != V8HImode && mode != V4SImode)
return false;
if (!can_create_pseudo_p ())
return false;
nunits = GET_MODE_NUNITS (mode);
mask = GET_MODE_MASK (inner);
val = const_vector_elt_as_int (op, BYTES_BIG_ENDIAN ? 0 : nunits - 1);
if (EASY_VECTOR_15 (val))
;
else if (EASY_VECTOR_MSB (val, inner))
;
else
return 0;
for (i = 1; i < nunits; ++i)
{
unsigned elt = BYTES_BIG_ENDIAN ? i : nunits - 1 - i;
HOST_WIDE_INT elt_val = const_vector_elt_as_int (op, elt);
if (val != elt_val)
{
if (elt_val == 0)
{
for (j = i+1; j < nunits; ++j)
{
unsigned elt2 = BYTES_BIG_ENDIAN ? j : nunits - 1 - j;
if (const_vector_elt_as_int (op, elt2) != 0)
return 0;
}
return (nunits - i) * GET_MODE_SIZE (inner);
}
else if ((elt_val & mask) == mask)
{
for (j = i+1; j < nunits; ++j)
{
unsigned elt2 = BYTES_BIG_ENDIAN ? j : nunits - 1 - j;
if ((const_vector_elt_as_int (op, elt2) & mask) != mask)
return 0;
}
return -((nunits - i) * GET_MODE_SIZE (inner));
}
else
return 0;
}
}
return 0;
}
bool
easy_altivec_constant (rtx op, machine_mode mode)
{
unsigned step, copies;
if (mode == VOIDmode)
mode = GET_MODE (op);
else if (mode != GET_MODE (op))
return false;
if (mode == V2DFmode)
return zero_constant (op, mode);
else if (mode == V2DImode)
{
if (GET_CODE (CONST_VECTOR_ELT (op, 0)) != CONST_INT
|| GET_CODE (CONST_VECTOR_ELT (op, 1)) != CONST_INT)
return false;
if (zero_constant (op, mode))
return true;
if (INTVAL (CONST_VECTOR_ELT (op, 0)) == -1
&& INTVAL (CONST_VECTOR_ELT (op, 1)) == -1)
return true;
return false;
}
else if (mode == V1TImode)
return false;
step = GET_MODE_NUNITS (mode) / 4;
copies = 1;
if (vspltis_constant (op, step, copies))
return true;
if (step == 1)
copies <<= 1;
else
step >>= 1;
if (vspltis_constant (op, step, copies))
return true;
if (step == 1)
copies <<= 1;
else
step >>= 1;
if (vspltis_constant (op, step, copies))
return true;
if (vspltis_shifted (op) != 0)
return true;
return false;
}
rtx
gen_easy_altivec_constant (rtx op)
{
machine_mode mode = GET_MODE (op);
int nunits = GET_MODE_NUNITS (mode);
rtx val = CONST_VECTOR_ELT (op, BYTES_BIG_ENDIAN ? nunits - 1 : 0);
unsigned step = nunits / 4;
unsigned copies = 1;
if (vspltis_constant (op, step, copies))
return gen_rtx_VEC_DUPLICATE (V4SImode, gen_lowpart (SImode, val));
if (step == 1)
copies <<= 1;
else
step >>= 1;
if (vspltis_constant (op, step, copies))
return gen_rtx_VEC_DUPLICATE (V8HImode, gen_lowpart (HImode, val));
if (step == 1)
copies <<= 1;
else
step >>= 1;
if (vspltis_constant (op, step, copies))
return gen_rtx_VEC_DUPLICATE (V16QImode, gen_lowpart (QImode, val));
gcc_unreachable ();
}
bool
xxspltib_constant_p (rtx op,
machine_mode mode,
int *num_insns_ptr,
int *constant_ptr)
{
size_t nunits = GET_MODE_NUNITS (mode);
size_t i;
HOST_WIDE_INT value;
rtx element;
*num_insns_ptr = -1;
*constant_ptr = 256;
if (!TARGET_P9_VECTOR)
return false;
if (mode == VOIDmode)
mode = GET_MODE (op);
else if (mode != GET_MODE (op) && GET_MODE (op) != VOIDmode)
return false;
if (GET_CODE (op) == VEC_DUPLICATE)
{
if (mode != V16QImode && mode != V8HImode && mode != V4SImode
&& mode != V2DImode)
return false;
element = XEXP (op, 0);
if (!CONST_INT_P (element))
return false;
value = INTVAL (element);
if (!IN_RANGE (value, -128, 127))
return false;
}
else if (GET_CODE (op) == CONST_VECTOR)
{
if (mode != V16QImode && mode != V8HImode && mode != V4SImode
&& mode != V2DImode)
return false;
element = CONST_VECTOR_ELT (op, 0);
if (!CONST_INT_P (element))
return false;
value = INTVAL (element);
if (!IN_RANGE (value, -128, 127))
return false;
for (i = 1; i < nunits; i++)
{
element = CONST_VECTOR_ELT (op, i);
if (!CONST_INT_P (element))
return false;
if (value != INTVAL (element))
return false;
}
}
else if (CONST_INT_P (op))
{
if (!SCALAR_INT_MODE_P (mode))
return false;
value = INTVAL (op);
if (!IN_RANGE (value, -128, 127))
return false;
if (!IN_RANGE (value, -1, 0))
{
if (!(reg_addr[mode].addr_mask[RELOAD_REG_VMX] & RELOAD_REG_VALID))
return false;
if (EASY_VECTOR_15 (value))
return false;
}
}
else
return false;
if ((mode == V4SImode || mode == V8HImode) && !IN_RANGE (value, -1, 0)
&& EASY_VECTOR_15 (value))
return false;
if (mode == V16QImode)
*num_insns_ptr = 1;
else if (IN_RANGE (value, -1, 0))
*num_insns_ptr = 1;
else
*num_insns_ptr = 2;
*constant_ptr = (int) value;
return true;
}
const char *
output_vec_const_move (rtx *operands)
{
int shift;
machine_mode mode;
rtx dest, vec;
dest = operands[0];
vec = operands[1];
mode = GET_MODE (dest);
if (TARGET_VSX)
{
bool dest_vmx_p = ALTIVEC_REGNO_P (REGNO (dest));
int xxspltib_value = 256;
int num_insns = -1;
if (zero_constant (vec, mode))
{
if (TARGET_P9_VECTOR)
return "xxspltib %x0,0";
else if (dest_vmx_p)
return "vspltisw %0,0";
else
return "xxlxor %x0,%x0,%x0";
}
if (all_ones_constant (vec, mode))
{
if (TARGET_P9_VECTOR)
return "xxspltib %x0,255";
else if (dest_vmx_p)
return "vspltisw %0,-1";
else if (TARGET_P8_VECTOR)
return "xxlorc %x0,%x0,%x0";
else
gcc_unreachable ();
}
if (TARGET_P9_VECTOR
&& xxspltib_constant_p (vec, mode, &num_insns, &xxspltib_value))
{
if (num_insns == 1)
{
operands[2] = GEN_INT (xxspltib_value & 0xff);
return "xxspltib %x0,%2";
}
return "#";
}
}
if (TARGET_ALTIVEC)
{
rtx splat_vec;
gcc_assert (ALTIVEC_REGNO_P (REGNO (dest)));
if (zero_constant (vec, mode))
return "vspltisw %0,0";
if (all_ones_constant (vec, mode))
return "vspltisw %0,-1";
shift = vspltis_shifted (vec);
if (shift != 0)
return "#";
splat_vec = gen_easy_altivec_constant (vec);
gcc_assert (GET_CODE (splat_vec) == VEC_DUPLICATE);
operands[1] = XEXP (splat_vec, 0);
if (!EASY_VECTOR_15 (INTVAL (operands[1])))
return "#";
switch (GET_MODE (splat_vec))
{
case E_V4SImode:
return "vspltisw %0,%1";
case E_V8HImode:
return "vspltish %0,%1";
case E_V16QImode:
return "vspltisb %0,%1";
default:
gcc_unreachable ();
}
}
gcc_unreachable ();
}
void
paired_expand_vector_init (rtx target, rtx vals)
{
machine_mode mode = GET_MODE (target);
int n_elts = GET_MODE_NUNITS (mode);
int n_var = 0;
rtx x, new_rtx, tmp, constant_op, op1, op2;
int i;
for (i = 0; i < n_elts; ++i)
{
x = XVECEXP (vals, 0, i);
if (!(CONST_SCALAR_INT_P (x) || CONST_DOUBLE_P (x) || CONST_FIXED_P (x)))
++n_var;
}
if (n_var == 0)
{
emit_move_insn (target, gen_rtx_CONST_VECTOR (mode, XVEC (vals, 0)));
return;
}
if (n_var == 2)
{
new_rtx = gen_rtx_VEC_CONCAT (V2SFmode, XVECEXP (vals, 0, 0),
XVECEXP (vals, 0, 1));
emit_move_insn (target, new_rtx);
return;
}
op1 = XVECEXP (vals, 0, 0);
op2 = XVECEXP (vals, 0, 1);
constant_op = (CONSTANT_P (op1)) ? op1 : op2;
tmp = gen_reg_rtx (GET_MODE (constant_op));
emit_move_insn (tmp, constant_op);
if (CONSTANT_P (op1))
new_rtx = gen_rtx_VEC_CONCAT (V2SFmode, tmp, op2);
else
new_rtx = gen_rtx_VEC_CONCAT (V2SFmode, op1, tmp);
emit_move_insn (target, new_rtx);
}
void
paired_expand_vector_move (rtx operands[])
{
rtx op0 = operands[0], op1 = operands[1];
emit_move_insn (op0, op1);
}
static void
paired_emit_vector_compare (enum rtx_code rcode,
rtx dest, rtx op0, rtx op1,
rtx cc_op0, rtx cc_op1)
{
rtx tmp = gen_reg_rtx (V2SFmode);
rtx tmp1, max, min;
gcc_assert (TARGET_PAIRED_FLOAT);
gcc_assert (GET_MODE (op0) == GET_MODE (op1));
switch (rcode)
{
case LT:
case LTU:
paired_emit_vector_compare (GE, dest, op1, op0, cc_op0, cc_op1);
return;
case GE:
case GEU:
emit_insn (gen_subv2sf3 (tmp, cc_op0, cc_op1));
emit_insn (gen_selv2sf4 (dest, tmp, op0, op1, CONST0_RTX (SFmode)));
return;
case LE:
case LEU:
paired_emit_vector_compare (GE, dest, op0, op1, cc_op1, cc_op0);
return;
case GT:
paired_emit_vector_compare (LE, dest, op1, op0, cc_op0, cc_op1);
return;
case EQ:
tmp1 = gen_reg_rtx (V2SFmode);
max = gen_reg_rtx (V2SFmode);
min = gen_reg_rtx (V2SFmode);
gen_reg_rtx (V2SFmode);
emit_insn (gen_subv2sf3 (tmp, cc_op0, cc_op1));
emit_insn (gen_selv2sf4
(max, tmp, cc_op0, cc_op1, CONST0_RTX (SFmode)));
emit_insn (gen_subv2sf3 (tmp, cc_op1, cc_op0));
emit_insn (gen_selv2sf4
(min, tmp, cc_op0, cc_op1, CONST0_RTX (SFmode)));
emit_insn (gen_subv2sf3 (tmp1, min, max));
emit_insn (gen_selv2sf4 (dest, tmp1, op0, op1, CONST0_RTX (SFmode)));
return;
case NE:
paired_emit_vector_compare (EQ, dest, op1, op0, cc_op0, cc_op1);
return;
case UNLE:
paired_emit_vector_compare (LE, dest, op1, op0, cc_op0, cc_op1);
return;
case UNLT:
paired_emit_vector_compare (LT, dest, op1, op0, cc_op0, cc_op1);
return;
case UNGE:
paired_emit_vector_compare (GE, dest, op1, op0, cc_op0, cc_op1);
return;
case UNGT:
paired_emit_vector_compare (GT, dest, op1, op0, cc_op0, cc_op1);
return;
default:
gcc_unreachable ();
}
return;
}
int
paired_emit_vector_cond_expr (rtx dest, rtx op1, rtx op2,
rtx cond, rtx cc_op0, rtx cc_op1)
{
enum rtx_code rcode = GET_CODE (cond);
if (!TARGET_PAIRED_FLOAT)
return 0;
paired_emit_vector_compare (rcode, dest, op1, op2, cc_op0, cc_op1);
return 1;
}
void
rs6000_expand_vector_init (rtx target, rtx vals)
{
machine_mode mode = GET_MODE (target);
machine_mode inner_mode = GET_MODE_INNER (mode);
int n_elts = GET_MODE_NUNITS (mode);
int n_var = 0, one_var = -1;
bool all_same = true, all_const_zero = true;
rtx x, mem;
int i;
for (i = 0; i < n_elts; ++i)
{
x = XVECEXP (vals, 0, i);
if (!(CONST_SCALAR_INT_P (x) || CONST_DOUBLE_P (x) || CONST_FIXED_P (x)))
++n_var, one_var = i;
else if (x != CONST0_RTX (inner_mode))
all_const_zero = false;
if (i > 0 && !rtx_equal_p (x, XVECEXP (vals, 0, 0)))
all_same = false;
}
if (n_var == 0)
{
rtx const_vec = gen_rtx_CONST_VECTOR (mode, XVEC (vals, 0));
bool int_vector_p = (GET_MODE_CLASS (mode) == MODE_VECTOR_INT);
if ((int_vector_p || TARGET_VSX) && all_const_zero)
{
emit_move_insn (target, CONST0_RTX (mode));
return;
}
else if (int_vector_p && easy_vector_constant (const_vec, mode))
{
emit_insn (gen_rtx_SET (target, const_vec));
return;
}
else
{
emit_move_insn (target, const_vec);
return;
}
}
if (VECTOR_MEM_VSX_P (mode) && (mode == V2DFmode || mode == V2DImode))
{
rtx op[2];
size_t i;
size_t num_elements = all_same ? 1 : 2;
for (i = 0; i < num_elements; i++)
{
op[i] = XVECEXP (vals, 0, i);
if (GET_MODE (op[i]) != inner_mode)
{
rtx tmp = gen_reg_rtx (inner_mode);
convert_move (tmp, op[i], 0);
op[i] = tmp;
}
else if (MEM_P (op[i]))
{
if (!all_same)
op[i] = force_reg (inner_mode, op[i]);
}
else if (!REG_P (op[i]))
op[i] = force_reg (inner_mode, op[i]);
}
if (all_same)
{
if (mode == V2DFmode)
emit_insn (gen_vsx_splat_v2df (target, op[0]));
else
emit_insn (gen_vsx_splat_v2di (target, op[0]));
}
else
{
if (mode == V2DFmode)
emit_insn (gen_vsx_concat_v2df (target, op[0], op[1]));
else
emit_insn (gen_vsx_concat_v2di (target, op[0], op[1]));
}
return;
}
if (mode == V4SImode  && VECTOR_MEM_VSX_P (V4SImode)
&& TARGET_DIRECT_MOVE_64BIT)
{
if (all_same)
{
rtx element0 = XVECEXP (vals, 0, 0);
if (MEM_P (element0))
element0 = rs6000_address_for_fpconvert (element0);
else
element0 = force_reg (SImode, element0);
if (TARGET_P9_VECTOR)
emit_insn (gen_vsx_splat_v4si (target, element0));
else
{
rtx tmp = gen_reg_rtx (DImode);
emit_insn (gen_zero_extendsidi2 (tmp, element0));
emit_insn (gen_vsx_splat_v4si_di (target, tmp));
}
return;
}
else
{
rtx elements[4];
size_t i;
for (i = 0; i < 4; i++)
{
elements[i] = XVECEXP (vals, 0, i);
if (!CONST_INT_P (elements[i]) && !REG_P (elements[i]))
elements[i] = copy_to_mode_reg (SImode, elements[i]);
}
emit_insn (gen_vsx_init_v4si (target, elements[0], elements[1],
elements[2], elements[3]));
return;
}
}
if (mode == V4SFmode && VECTOR_MEM_VSX_P (V4SFmode))
{
if (all_same)
{
rtx element0 = XVECEXP (vals, 0, 0);
if (TARGET_P9_VECTOR)
{
if (MEM_P (element0))
element0 = rs6000_address_for_fpconvert (element0);
emit_insn (gen_vsx_splat_v4sf (target, element0));
}
else
{
rtx freg = gen_reg_rtx (V4SFmode);
rtx sreg = force_reg (SFmode, element0);
rtx cvt  = (TARGET_XSCVDPSPN
? gen_vsx_xscvdpspn_scalar (freg, sreg)
: gen_vsx_xscvdpsp_scalar (freg, sreg));
emit_insn (cvt);
emit_insn (gen_vsx_xxspltw_v4sf_direct (target, freg,
const0_rtx));
}
}
else
{
rtx dbl_even = gen_reg_rtx (V2DFmode);
rtx dbl_odd  = gen_reg_rtx (V2DFmode);
rtx flt_even = gen_reg_rtx (V4SFmode);
rtx flt_odd  = gen_reg_rtx (V4SFmode);
rtx op0 = force_reg (SFmode, XVECEXP (vals, 0, 0));
rtx op1 = force_reg (SFmode, XVECEXP (vals, 0, 1));
rtx op2 = force_reg (SFmode, XVECEXP (vals, 0, 2));
rtx op3 = force_reg (SFmode, XVECEXP (vals, 0, 3));
if (TARGET_P8_VECTOR)
{
emit_insn (gen_vsx_concat_v2sf (dbl_even, op0, op2));
emit_insn (gen_vsx_concat_v2sf (dbl_odd, op1, op3));
emit_insn (gen_vsx_xvcvdpsp (flt_even, dbl_even));
emit_insn (gen_vsx_xvcvdpsp (flt_odd, dbl_odd));
if (BYTES_BIG_ENDIAN)
emit_insn (gen_p8_vmrgew_v4sf_direct (target, flt_even, flt_odd));
else
emit_insn (gen_p8_vmrgew_v4sf_direct (target, flt_odd, flt_even));
}
else
{
emit_insn (gen_vsx_concat_v2sf (dbl_even, op0, op1));
emit_insn (gen_vsx_concat_v2sf (dbl_odd, op2, op3));
emit_insn (gen_vsx_xvcvdpsp (flt_even, dbl_even));
emit_insn (gen_vsx_xvcvdpsp (flt_odd, dbl_odd));
rs6000_expand_extract_even (target, flt_even, flt_odd);
}
}
return;
}
if (all_same && TARGET_DIRECT_MOVE_64BIT
&& (mode == V16QImode || mode == V8HImode))
{
rtx op0 = XVECEXP (vals, 0, 0);
rtx di_tmp = gen_reg_rtx (DImode);
if (!REG_P (op0))
op0 = force_reg (GET_MODE_INNER (mode), op0);
if (mode == V16QImode)
{
emit_insn (gen_zero_extendqidi2 (di_tmp, op0));
emit_insn (gen_vsx_vspltb_di (target, di_tmp));
return;
}
if (mode == V8HImode)
{
emit_insn (gen_zero_extendhidi2 (di_tmp, op0));
emit_insn (gen_vsx_vsplth_di (target, di_tmp));
return;
}
}
if (all_same && GET_MODE_SIZE (inner_mode) <= 4)
{
mem = assign_stack_temp (mode, GET_MODE_SIZE (inner_mode));
emit_move_insn (adjust_address_nv (mem, inner_mode, 0),
XVECEXP (vals, 0, 0));
x = gen_rtx_UNSPEC (VOIDmode,
gen_rtvec (1, const0_rtx), UNSPEC_LVE);
emit_insn (gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (2,
gen_rtx_SET (target, mem),
x)));
x = gen_rtx_VEC_SELECT (inner_mode, target,
gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (1, const0_rtx)));
emit_insn (gen_rtx_SET (target, gen_rtx_VEC_DUPLICATE (mode, x)));
return;
}
if (n_var == 1)
{
rtx copy = copy_rtx (vals);
XVECEXP (copy, 0, one_var) = XVECEXP (vals, 0, (one_var + 1) % n_elts);
rs6000_expand_vector_init (target, copy);
rs6000_expand_vector_set (target, XVECEXP (vals, 0, one_var), one_var);
return;
}
mem = assign_stack_temp (mode, GET_MODE_SIZE (mode));
for (i = 0; i < n_elts; i++)
emit_move_insn (adjust_address_nv (mem, inner_mode,
i * GET_MODE_SIZE (inner_mode)),
XVECEXP (vals, 0, i));
emit_move_insn (target, mem);
}
void
rs6000_expand_vector_set (rtx target, rtx val, int elt)
{
machine_mode mode = GET_MODE (target);
machine_mode inner_mode = GET_MODE_INNER (mode);
rtx reg = gen_reg_rtx (mode);
rtx mask, mem, x;
int width = GET_MODE_SIZE (inner_mode);
int i;
val = force_reg (GET_MODE (val), val);
if (VECTOR_MEM_VSX_P (mode))
{
rtx insn = NULL_RTX;
rtx elt_rtx = GEN_INT (elt);
if (mode == V2DFmode)
insn = gen_vsx_set_v2df (target, target, val, elt_rtx);
else if (mode == V2DImode)
insn = gen_vsx_set_v2di (target, target, val, elt_rtx);
else if (TARGET_P9_VECTOR && TARGET_POWERPC64)
{
if (mode == V4SImode)
insn = gen_vsx_set_v4si_p9 (target, target, val, elt_rtx);
else if (mode == V8HImode)
insn = gen_vsx_set_v8hi_p9 (target, target, val, elt_rtx);
else if (mode == V16QImode)
insn = gen_vsx_set_v16qi_p9 (target, target, val, elt_rtx);
else if (mode == V4SFmode)
insn = gen_vsx_set_v4sf_p9 (target, target, val, elt_rtx);
}
if (insn)
{
emit_insn (insn);
return;
}
}
if (GET_MODE_SIZE (mode) == GET_MODE_SIZE (inner_mode) && elt == 0)
{
emit_move_insn (target, gen_lowpart (mode, val));
return;
}
mem = assign_stack_temp (mode, GET_MODE_SIZE (inner_mode));
emit_move_insn (adjust_address_nv (mem, inner_mode, 0), val);
x = gen_rtx_UNSPEC (VOIDmode,
gen_rtvec (1, const0_rtx), UNSPEC_LVE);
emit_insn (gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (2,
gen_rtx_SET (reg, mem),
x)));
mask = gen_rtx_PARALLEL (V16QImode, rtvec_alloc (16));
for (i = 0; i < 16; ++i)
XVECEXP (mask, 0, i) = GEN_INT (i);
for (i = 0; i < width; ++i)
XVECEXP (mask, 0, elt*width + i)
= GEN_INT (i + 0x10);
x = gen_rtx_CONST_VECTOR (V16QImode, XVEC (mask, 0));
if (BYTES_BIG_ENDIAN)
x = gen_rtx_UNSPEC (mode,
gen_rtvec (3, target, reg,
force_reg (V16QImode, x)),
UNSPEC_VPERM);
else
{
if (TARGET_P9_VECTOR)
x = gen_rtx_UNSPEC (mode,
gen_rtvec (3, reg, target,
force_reg (V16QImode, x)),
UNSPEC_VPERMR);
else
{
rtx notx = gen_rtx_NOT (V16QImode, force_reg (V16QImode, x));
rtx iorx = (TARGET_P8_VECTOR
? gen_rtx_IOR (V16QImode, notx, notx)
: gen_rtx_AND (V16QImode, notx, notx));
rtx tmp = gen_reg_rtx (V16QImode);
emit_insn (gen_rtx_SET (tmp, iorx));
x = gen_rtx_UNSPEC (mode, gen_rtvec (3, reg, target, tmp),
UNSPEC_VPERM);
}
}
emit_insn (gen_rtx_SET (target, x));
}
void
rs6000_expand_vector_extract (rtx target, rtx vec, rtx elt)
{
machine_mode mode = GET_MODE (vec);
machine_mode inner_mode = GET_MODE_INNER (mode);
rtx mem;
if (VECTOR_MEM_VSX_P (mode) && CONST_INT_P (elt))
{
switch (mode)
{
default:
break;
case E_V1TImode:
gcc_assert (INTVAL (elt) == 0 && inner_mode == TImode);
emit_move_insn (target, gen_lowpart (TImode, vec));
break;
case E_V2DFmode:
emit_insn (gen_vsx_extract_v2df (target, vec, elt));
return;
case E_V2DImode:
emit_insn (gen_vsx_extract_v2di (target, vec, elt));
return;
case E_V4SFmode:
emit_insn (gen_vsx_extract_v4sf (target, vec, elt));
return;
case E_V16QImode:
if (TARGET_DIRECT_MOVE_64BIT)
{
emit_insn (gen_vsx_extract_v16qi (target, vec, elt));
return;
}
else
break;
case E_V8HImode:
if (TARGET_DIRECT_MOVE_64BIT)
{
emit_insn (gen_vsx_extract_v8hi (target, vec, elt));
return;
}
else
break;
case E_V4SImode:
if (TARGET_DIRECT_MOVE_64BIT)
{
emit_insn (gen_vsx_extract_v4si (target, vec, elt));
return;
}
break;
}
}
else if (VECTOR_MEM_VSX_P (mode) && !CONST_INT_P (elt)
&& TARGET_DIRECT_MOVE_64BIT)
{
if (GET_MODE (elt) != DImode)
{
rtx tmp = gen_reg_rtx (DImode);
convert_move (tmp, elt, 0);
elt = tmp;
}
else if (!REG_P (elt))
elt = force_reg (DImode, elt);
switch (mode)
{
case E_V2DFmode:
emit_insn (gen_vsx_extract_v2df_var (target, vec, elt));
return;
case E_V2DImode:
emit_insn (gen_vsx_extract_v2di_var (target, vec, elt));
return;
case E_V4SFmode:
emit_insn (gen_vsx_extract_v4sf_var (target, vec, elt));
return;
case E_V4SImode:
emit_insn (gen_vsx_extract_v4si_var (target, vec, elt));
return;
case E_V8HImode:
emit_insn (gen_vsx_extract_v8hi_var (target, vec, elt));
return;
case E_V16QImode:
emit_insn (gen_vsx_extract_v16qi_var (target, vec, elt));
return;
default:
gcc_unreachable ();
}
}
gcc_assert (CONST_INT_P (elt));
mem = assign_stack_temp (mode, GET_MODE_SIZE (mode));
emit_move_insn (mem, vec);
mem = adjust_address_nv (mem, inner_mode,
INTVAL (elt) * GET_MODE_SIZE (inner_mode));
emit_move_insn (target, adjust_address_nv (mem, inner_mode, 0));
}
static inline int
regno_or_subregno (rtx op)
{
if (REG_P (op))
return REGNO (op);
else if (SUBREG_P (op))
return subreg_regno (op);
else
gcc_unreachable ();
}
rtx
rs6000_adjust_vec_address (rtx scalar_reg,
rtx mem,
rtx element,
rtx base_tmp,
machine_mode scalar_mode)
{
unsigned scalar_size = GET_MODE_SIZE (scalar_mode);
rtx addr = XEXP (mem, 0);
rtx element_offset;
rtx new_addr;
bool valid_addr_p;
gcc_assert (GET_RTX_CLASS (GET_CODE (addr)) != RTX_AUTOINC);
if (CONST_INT_P (element))
element_offset = GEN_INT (INTVAL (element) * scalar_size);
else
{
int byte_shift = exact_log2 (scalar_size);
gcc_assert (byte_shift >= 0);
if (byte_shift == 0)
element_offset = element;
else
{
if (TARGET_POWERPC64)
emit_insn (gen_ashldi3 (base_tmp, element, GEN_INT (byte_shift)));
else
emit_insn (gen_ashlsi3 (base_tmp, element, GEN_INT (byte_shift)));
element_offset = base_tmp;
}
}
if (element_offset == const0_rtx)
new_addr = addr;
else if (REG_P (addr) || SUBREG_P (addr))
new_addr = gen_rtx_PLUS (Pmode, addr, element_offset);
else if (GET_CODE (addr) == PLUS)
{
rtx op0 = XEXP (addr, 0);
rtx op1 = XEXP (addr, 1);
rtx insn;
gcc_assert (REG_P (op0) || SUBREG_P (op0));
if (CONST_INT_P (op1) && CONST_INT_P (element_offset))
{
HOST_WIDE_INT offset = INTVAL (op1) + INTVAL (element_offset);
rtx offset_rtx = GEN_INT (offset);
if (IN_RANGE (offset, -32768, 32767)
&& (scalar_size < 8 || (offset & 0x3) == 0))
new_addr = gen_rtx_PLUS (Pmode, op0, offset_rtx);
else
{
emit_move_insn (base_tmp, offset_rtx);
new_addr = gen_rtx_PLUS (Pmode, op0, base_tmp);
}
}
else
{
bool op1_reg_p = (REG_P (op1) || SUBREG_P (op1));
bool ele_reg_p = (REG_P (element_offset) || SUBREG_P (element_offset));
if (op1_reg_p
&& (ele_reg_p || reg_or_subregno (op1) != FIRST_GPR_REGNO))
{
insn = gen_add3_insn (base_tmp, op1, element_offset);
gcc_assert (insn != NULL_RTX);
emit_insn (insn);
}
else if (ele_reg_p
&& reg_or_subregno (element_offset) != FIRST_GPR_REGNO)
{
insn = gen_add3_insn (base_tmp, element_offset, op1);
gcc_assert (insn != NULL_RTX);
emit_insn (insn);
}
else
{
emit_move_insn (base_tmp, op1);
emit_insn (gen_add2_insn (base_tmp, element_offset));
}
new_addr = gen_rtx_PLUS (Pmode, op0, base_tmp);
}
}
else
{
emit_move_insn (base_tmp, addr);
new_addr = gen_rtx_PLUS (Pmode, base_tmp, element_offset);
}
if (GET_CODE (new_addr) == PLUS)
{
rtx op1 = XEXP (new_addr, 1);
addr_mask_type addr_mask;
int scalar_regno = regno_or_subregno (scalar_reg);
gcc_assert (scalar_regno < FIRST_PSEUDO_REGISTER);
if (INT_REGNO_P (scalar_regno))
addr_mask = reg_addr[scalar_mode].addr_mask[RELOAD_REG_GPR];
else if (FP_REGNO_P (scalar_regno))
addr_mask = reg_addr[scalar_mode].addr_mask[RELOAD_REG_FPR];
else if (ALTIVEC_REGNO_P (scalar_regno))
addr_mask = reg_addr[scalar_mode].addr_mask[RELOAD_REG_VMX];
else
gcc_unreachable ();
if (REG_P (op1) || SUBREG_P (op1))
valid_addr_p = (addr_mask & RELOAD_REG_INDEXED) != 0;
else
valid_addr_p = (addr_mask & RELOAD_REG_OFFSET) != 0;
}
else if (REG_P (new_addr) || SUBREG_P (new_addr))
valid_addr_p = true;
else
valid_addr_p = false;
if (!valid_addr_p)
{
emit_move_insn (base_tmp, new_addr);
new_addr = base_tmp;
}
return change_address (mem, scalar_mode, new_addr);
}
void
rs6000_split_vec_extract_var (rtx dest, rtx src, rtx element, rtx tmp_gpr,
rtx tmp_altivec)
{
machine_mode mode = GET_MODE (src);
machine_mode scalar_mode = GET_MODE (dest);
unsigned scalar_size = GET_MODE_SIZE (scalar_mode);
int byte_shift = exact_log2 (scalar_size);
gcc_assert (byte_shift >= 0);
if (MEM_P (src))
{
gcc_assert (REG_P (tmp_gpr));
emit_move_insn (dest, rs6000_adjust_vec_address (dest, src, element,
tmp_gpr, scalar_mode));
return;
}
else if (REG_P (src) || SUBREG_P (src))
{
int bit_shift = byte_shift + 3;
rtx element2;
int dest_regno = regno_or_subregno (dest);
int src_regno = regno_or_subregno (src);
int element_regno = regno_or_subregno (element);
gcc_assert (REG_P (tmp_gpr));
if (TARGET_P9_VECTOR
&& (mode == V16QImode || mode == V8HImode || mode == V4SImode)
&& INT_REGNO_P (dest_regno)
&& ALTIVEC_REGNO_P (src_regno)
&& INT_REGNO_P (element_regno))
{
rtx dest_si = gen_rtx_REG (SImode, dest_regno);
rtx element_si = gen_rtx_REG (SImode, element_regno);
if (mode == V16QImode)
emit_insn (VECTOR_ELT_ORDER_BIG
? gen_vextublx (dest_si, element_si, src)
: gen_vextubrx (dest_si, element_si, src));
else if (mode == V8HImode)
{
rtx tmp_gpr_si = gen_rtx_REG (SImode, REGNO (tmp_gpr));
emit_insn (gen_ashlsi3 (tmp_gpr_si, element_si, const1_rtx));
emit_insn (VECTOR_ELT_ORDER_BIG
? gen_vextuhlx (dest_si, tmp_gpr_si, src)
: gen_vextuhrx (dest_si, tmp_gpr_si, src));
}
else
{
rtx tmp_gpr_si = gen_rtx_REG (SImode, REGNO (tmp_gpr));
emit_insn (gen_ashlsi3 (tmp_gpr_si, element_si, const2_rtx));
emit_insn (VECTOR_ELT_ORDER_BIG
? gen_vextuwlx (dest_si, tmp_gpr_si, src)
: gen_vextuwrx (dest_si, tmp_gpr_si, src));
}
return;
}
gcc_assert (REG_P (tmp_altivec));
if (scalar_size == 8)
{
if (!VECTOR_ELT_ORDER_BIG)
{
emit_insn (gen_xordi3 (tmp_gpr, element, const1_rtx));
element2 = tmp_gpr;
}
else
element2 = element;
emit_insn (gen_rtx_SET (tmp_gpr,
gen_rtx_AND (DImode,
gen_rtx_ASHIFT (DImode,
element2,
GEN_INT (6)),
GEN_INT (64))));
}
else
{
if (!VECTOR_ELT_ORDER_BIG)
{
rtx num_ele_m1 = GEN_INT (GET_MODE_NUNITS (mode) - 1);
emit_insn (gen_anddi3 (tmp_gpr, element, num_ele_m1));
emit_insn (gen_subdi3 (tmp_gpr, num_ele_m1, tmp_gpr));
element2 = tmp_gpr;
}
else
element2 = element;
emit_insn (gen_ashldi3 (tmp_gpr, element2, GEN_INT (bit_shift)));
}
if (TARGET_P9_VECTOR)
emit_insn (gen_vsx_splat_v2di (tmp_altivec, tmp_gpr));
else if (can_create_pseudo_p ())
emit_insn (gen_vsx_concat_v2di (tmp_altivec, tmp_gpr, tmp_gpr));
else
{
rtx tmp_di = gen_rtx_REG (DImode, REGNO (tmp_altivec));
emit_move_insn (tmp_di, tmp_gpr);
emit_insn (gen_vsx_concat_v2di (tmp_altivec, tmp_di, tmp_di));
}
switch (mode)
{
case E_V2DFmode:
emit_insn (gen_vsx_vslo_v2df (dest, src, tmp_altivec));
return;
case E_V2DImode:
emit_insn (gen_vsx_vslo_v2di (dest, src, tmp_altivec));
return;
case E_V4SFmode:
{
rtx tmp_altivec_di = gen_rtx_REG (DImode, REGNO (tmp_altivec));
rtx tmp_altivec_v4sf = gen_rtx_REG (V4SFmode, REGNO (tmp_altivec));
rtx src_v2di = gen_rtx_REG (V2DImode, REGNO (src));
emit_insn (gen_vsx_vslo_v2di (tmp_altivec_di, src_v2di,
tmp_altivec));
emit_insn (gen_vsx_xscvspdp_scalar2 (dest, tmp_altivec_v4sf));
return;
}
case E_V4SImode:
case E_V8HImode:
case E_V16QImode:
{
rtx tmp_altivec_di = gen_rtx_REG (DImode, REGNO (tmp_altivec));
rtx src_v2di = gen_rtx_REG (V2DImode, REGNO (src));
rtx tmp_gpr_di = gen_rtx_REG (DImode, REGNO (dest));
emit_insn (gen_vsx_vslo_v2di (tmp_altivec_di, src_v2di,
tmp_altivec));
emit_move_insn (tmp_gpr_di, tmp_altivec_di);
emit_insn (gen_ashrdi3 (tmp_gpr_di, tmp_gpr_di,
GEN_INT (64 - (8 * scalar_size))));
return;
}
default:
gcc_unreachable ();
}
return;
}
else
gcc_unreachable ();
}
static void
rs6000_split_v4si_init_di_reg (rtx dest, rtx si1, rtx si2, rtx tmp)
{
const unsigned HOST_WIDE_INT mask_32bit = HOST_WIDE_INT_C (0xffffffff);
if (CONST_INT_P (si1) && CONST_INT_P (si2))
{
unsigned HOST_WIDE_INT const1 = (UINTVAL (si1) & mask_32bit) << 32;
unsigned HOST_WIDE_INT const2 = UINTVAL (si2) & mask_32bit;
emit_move_insn (dest, GEN_INT (const1 | const2));
return;
}
if (CONST_INT_P (si1))
emit_move_insn (dest, GEN_INT ((UINTVAL (si1) & mask_32bit) << 32));
else
{
rtx si1_di = gen_rtx_REG (DImode, regno_or_subregno (si1));
rtx shift_rtx = gen_rtx_ASHIFT (DImode, si1_di, GEN_INT (32));
rtx mask_rtx = GEN_INT (mask_32bit << 32);
rtx and_rtx = gen_rtx_AND (DImode, shift_rtx, mask_rtx);
gcc_assert (!reg_overlap_mentioned_p (dest, si1));
emit_insn (gen_rtx_SET (dest, and_rtx));
}
gcc_assert (!reg_overlap_mentioned_p (dest, tmp));
if (CONST_INT_P (si2))
emit_move_insn (tmp, GEN_INT (UINTVAL (si2) & mask_32bit));
else
emit_insn (gen_zero_extendsidi2 (tmp, si2));
emit_insn (gen_iordi3 (dest, dest, tmp));
return;
}
void
rs6000_split_v4si_init (rtx operands[])
{
rtx dest = operands[0];
if (REG_P (dest) || SUBREG_P (dest))
{
int d_regno = regno_or_subregno (dest);
rtx scalar1 = operands[1];
rtx scalar2 = operands[2];
rtx scalar3 = operands[3];
rtx scalar4 = operands[4];
rtx tmp1 = operands[5];
rtx tmp2 = operands[6];
if (BYTES_BIG_ENDIAN)
{
rtx di_lo = gen_rtx_REG (DImode, d_regno);
rtx di_hi = gen_rtx_REG (DImode, d_regno + 1);
rs6000_split_v4si_init_di_reg (di_lo, scalar1, scalar2, tmp1);
rs6000_split_v4si_init_di_reg (di_hi, scalar3, scalar4, tmp2);
}
else
{
rtx di_lo = gen_rtx_REG (DImode, d_regno + 1);
rtx di_hi = gen_rtx_REG (DImode, d_regno);
gcc_assert (!VECTOR_ELT_ORDER_BIG);
rs6000_split_v4si_init_di_reg (di_lo, scalar4, scalar3, tmp1);
rs6000_split_v4si_init_di_reg (di_hi, scalar2, scalar1, tmp2);
}
return;
}
else
gcc_unreachable ();
}
unsigned int
rs6000_data_alignment (tree type, unsigned int align, enum data_align how)
{
if (how != align_opt)
{
if (TREE_CODE (type) == VECTOR_TYPE)
{
if (TARGET_PAIRED_FLOAT && PAIRED_VECTOR_MODE (TYPE_MODE (type)))
{
if (align < 64)
align = 64;
}
else if (align < 128)
align = 128;
}
}
if (how != align_abi)
{
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_MODE (TREE_TYPE (type)) == QImode)
{
if (align < BITS_PER_WORD)
align = BITS_PER_WORD;
}
}
return align;
}
static bool
rs6000_slow_unaligned_access (machine_mode mode, unsigned int align)
{
return (STRICT_ALIGNMENT
|| (!TARGET_EFFICIENT_UNALIGNED_VSX
&& ((SCALAR_FLOAT_MODE_NOT_VECTOR_P (mode) && align < 32)
|| ((VECTOR_MODE_P (mode) || FLOAT128_VECTOR_P (mode))
&& (int) align < VECTOR_ALIGN (mode)))));
}
bool
rs6000_special_adjust_field_align_p (tree type, unsigned int computed)
{
if (TARGET_ALTIVEC && TREE_CODE (type) == VECTOR_TYPE)
{
if (computed != 128)
{
static bool warned;
if (!warned && warn_psabi)
{
warned = true;
inform (input_location,
"the layout of aggregates containing vectors with"
" %d-byte alignment has changed in GCC 5",
computed / BITS_PER_UNIT);
}
}
return false;
}
return false;
}
unsigned int
rs6000_special_round_type_align (tree type, unsigned int computed,
unsigned int specified)
{
unsigned int align = MAX (computed, specified);
tree field = TYPE_FIELDS (type);
while (field != NULL && TREE_CODE (field) != FIELD_DECL)
field = DECL_CHAIN (field);
if (field != NULL && field != type)
{
type = TREE_TYPE (field);
while (TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
if (type != error_mark_node && TYPE_MODE (type) == DFmode)
align = MAX (align, 64);
}
return align;
}
unsigned int
darwin_rs6000_special_round_type_align (tree type, unsigned int computed,
unsigned int specified)
{
unsigned int align = MAX (computed, specified);
if (TYPE_PACKED (type))
return align;
do {
tree field = TYPE_FIELDS (type);
while (field != NULL && TREE_CODE (field) != FIELD_DECL)
field = DECL_CHAIN (field);
if (! field)
break;
if (DECL_PACKED (field))
return align;
type = TREE_TYPE (field);
while (TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
} while (AGGREGATE_TYPE_P (type));
if (! AGGREGATE_TYPE_P (type) && type != error_mark_node)
align = MAX (align, TYPE_ALIGN (type));
return align;
}
int
small_data_operand (rtx op ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED)
{
#if TARGET_ELF
rtx sym_ref;
if (rs6000_sdata == SDATA_NONE || rs6000_sdata == SDATA_DATA)
return 0;
if (DEFAULT_ABI != ABI_V4)
return 0;
if (GET_CODE (op) == SYMBOL_REF)
sym_ref = op;
else if (GET_CODE (op) != CONST
|| GET_CODE (XEXP (op, 0)) != PLUS
|| GET_CODE (XEXP (XEXP (op, 0), 0)) != SYMBOL_REF
|| GET_CODE (XEXP (XEXP (op, 0), 1)) != CONST_INT)
return 0;
else
{
rtx sum = XEXP (op, 0);
HOST_WIDE_INT summand;
summand = INTVAL (XEXP (sum, 1));
if (summand < 0 || summand > g_switch_value)
return 0;
sym_ref = XEXP (sum, 0);
}
return SYMBOL_REF_SMALL_P (sym_ref);
#else
return 0;
#endif
}
bool
gpr_or_gpr_p (rtx op0, rtx op1)
{
return ((REG_P (op0) && INT_REGNO_P (REGNO (op0)))
|| (REG_P (op1) && INT_REGNO_P (REGNO (op1))));
}
bool
direct_move_p (rtx op0, rtx op1)
{
int regno0, regno1;
if (!REG_P (op0) || !REG_P (op1))
return false;
if (!TARGET_DIRECT_MOVE && !TARGET_MFPGPR)
return false;
regno0 = REGNO (op0);
regno1 = REGNO (op1);
if (regno0 >= FIRST_PSEUDO_REGISTER || regno1 >= FIRST_PSEUDO_REGISTER)
return false;
if (INT_REGNO_P (regno0))
return (TARGET_DIRECT_MOVE) ? VSX_REGNO_P (regno1) : FP_REGNO_P (regno1);
else if (INT_REGNO_P (regno1))
{
if (TARGET_MFPGPR && FP_REGNO_P (regno0))
return true;
else if (TARGET_DIRECT_MOVE && VSX_REGNO_P (regno0))
return true;
}
return false;
}
static inline bool
quad_address_offset_p (HOST_WIDE_INT offset)
{
return (IN_RANGE (offset, -32768, 32767) && ((offset) & 0xf) == 0);
}
bool
quad_address_p (rtx addr, machine_mode mode, bool strict)
{
rtx op0, op1;
if (GET_MODE_SIZE (mode) != 16)
return false;
if (legitimate_indirect_address_p (addr, strict))
return true;
if (VECTOR_MODE_P (mode) && !mode_supports_vsx_dform_quad (mode))
return false;
if (GET_CODE (addr) != PLUS)
return false;
op0 = XEXP (addr, 0);
if (!REG_P (op0) || !INT_REG_OK_FOR_BASE_P (op0, strict))
return false;
op1 = XEXP (addr, 1);
if (!CONST_INT_P (op1))
return false;
return quad_address_offset_p (INTVAL (op1));
}
bool
quad_load_store_p (rtx op0, rtx op1)
{
bool ret;
if (!TARGET_QUAD_MEMORY)
ret = false;
else if (REG_P (op0) && MEM_P (op1))
ret = (quad_int_reg_operand (op0, GET_MODE (op0))
&& quad_memory_operand (op1, GET_MODE (op1))
&& !reg_overlap_mentioned_p (op0, op1));
else if (MEM_P (op0) && REG_P (op1))
ret = (quad_memory_operand (op0, GET_MODE (op0))
&& quad_int_reg_operand (op1, GET_MODE (op1)));
else
ret = false;
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\n========== quad_load_store, return %s\n",
ret ? "true" : "false");
debug_rtx (gen_rtx_SET (op0, op1));
}
return ret;
}
static rtx
address_offset (rtx op)
{
if (GET_CODE (op) == PRE_INC
|| GET_CODE (op) == PRE_DEC)
op = XEXP (op, 0);
else if (GET_CODE (op) == PRE_MODIFY
|| GET_CODE (op) == LO_SUM)
op = XEXP (op, 1);
if (GET_CODE (op) == CONST)
op = XEXP (op, 0);
if (GET_CODE (op) == PLUS)
op = XEXP (op, 1);
if (CONST_INT_P (op))
return op;
return NULL_RTX;
}
bool
mem_operand_gpr (rtx op, machine_mode mode)
{
unsigned HOST_WIDE_INT offset;
int extra;
rtx addr = XEXP (op, 0);
if (TARGET_UPDATE
&& (GET_CODE (addr) == PRE_INC || GET_CODE (addr) == PRE_DEC)
&& mode_supports_pre_incdec_p (mode)
&& legitimate_indirect_address_p (XEXP (addr, 0), false))
return true;
if (!rs6000_offsettable_memref_p (op, mode, false))
return false;
op = address_offset (addr);
if (op == NULL_RTX)
return true;
offset = INTVAL (op);
if (TARGET_POWERPC64 && (offset & 3) != 0)
return false;
extra = GET_MODE_SIZE (mode) - UNITS_PER_WORD;
if (extra < 0)
extra = 0;
if (GET_CODE (addr) == LO_SUM)
offset = ((offset & 0xffff) ^ 0x8000) - 0x8000;
return offset + 0x8000 < 0x10000u - extra;
}
bool
mem_operand_ds_form (rtx op, machine_mode mode)
{
unsigned HOST_WIDE_INT offset;
int extra;
rtx addr = XEXP (op, 0);
if (!offsettable_address_p (false, mode, addr))
return false;
op = address_offset (addr);
if (op == NULL_RTX)
return true;
offset = INTVAL (op);
if ((offset & 3) != 0)
return false;
extra = GET_MODE_SIZE (mode) - UNITS_PER_WORD;
if (extra < 0)
extra = 0;
if (GET_CODE (addr) == LO_SUM)
offset = ((offset & 0xffff) ^ 0x8000) - 0x8000;
return offset + 0x8000 < 0x10000u - extra;
}

static bool
reg_offset_addressing_ok_p (machine_mode mode)
{
switch (mode)
{
case E_V16QImode:
case E_V8HImode:
case E_V4SFmode:
case E_V4SImode:
case E_V2DFmode:
case E_V2DImode:
case E_V1TImode:
case E_TImode:
case E_TFmode:
case E_KFmode:
if (VECTOR_MEM_ALTIVEC_OR_VSX_P (mode))
return mode_supports_vsx_dform_quad (mode);
break;
case E_V2SImode:
case E_V2SFmode:
if (TARGET_PAIRED_FLOAT)
return false;
break;
case E_SDmode:
if (TARGET_NO_SDMODE_STACK)
return false;
break;
default:
break;
}
return true;
}
static bool
virtual_stack_registers_memory_p (rtx op)
{
int regnum;
if (GET_CODE (op) == REG)
regnum = REGNO (op);
else if (GET_CODE (op) == PLUS
&& GET_CODE (XEXP (op, 0)) == REG
&& GET_CODE (XEXP (op, 1)) == CONST_INT)
regnum = REGNO (XEXP (op, 0));
else
return false;
return (regnum >= FIRST_VIRTUAL_REGISTER
&& regnum <= LAST_VIRTUAL_POINTER_REGISTER);
}
#ifndef POWERPC64_TOC_POINTER_ALIGNMENT
#define POWERPC64_TOC_POINTER_ALIGNMENT 8
#endif
static bool
offsettable_ok_by_alignment (rtx op, HOST_WIDE_INT offset,
machine_mode mode)
{
tree decl;
unsigned HOST_WIDE_INT dsize, dalign, lsb, mask;
if (GET_CODE (op) != SYMBOL_REF)
return false;
if (mode_supports_vsx_dform_quad (mode))
return false;
dsize = GET_MODE_SIZE (mode);
decl = SYMBOL_REF_DECL (op);
if (!decl)
{
if (dsize == 0)
return false;
dalign = BITS_PER_UNIT;
if (SYMBOL_REF_HAS_BLOCK_INFO_P (op)
&& SYMBOL_REF_ANCHOR_P (op)
&& SYMBOL_REF_BLOCK (op) != NULL)
{
struct object_block *block = SYMBOL_REF_BLOCK (op);
dalign = block->alignment;
offset += SYMBOL_REF_BLOCK_OFFSET (op);
}
else if (CONSTANT_POOL_ADDRESS_P (op))
{
machine_mode cmode = get_pool_mode (op);
dalign = GET_MODE_ALIGNMENT (cmode);
}
}
else if (DECL_P (decl))
{
dalign = DECL_ALIGN (decl);
if (dsize == 0)
{
if (!DECL_SIZE_UNIT (decl))
return false;
if (!tree_fits_uhwi_p (DECL_SIZE_UNIT (decl)))
return false;
dsize = tree_to_uhwi (DECL_SIZE_UNIT (decl));
if (dsize > 32768)
return false;
dalign /= BITS_PER_UNIT;
if (dalign > POWERPC64_TOC_POINTER_ALIGNMENT)
dalign = POWERPC64_TOC_POINTER_ALIGNMENT;
return dalign >= dsize;
}
}
else
gcc_unreachable ();
dalign /= BITS_PER_UNIT;
if (dalign > POWERPC64_TOC_POINTER_ALIGNMENT)
dalign = POWERPC64_TOC_POINTER_ALIGNMENT;
mask = dalign - 1;
lsb = offset & -offset;
mask &= lsb - 1;
dalign = mask + 1;
return dalign >= dsize;
}
static bool
constant_pool_expr_p (rtx op)
{
rtx base, offset;
split_const (op, &base, &offset);
return (GET_CODE (base) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (base)
&& ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (get_pool_constant (base), Pmode));
}
static const_rtx tocrel_base_oac, tocrel_offset_oac;
bool
toc_relative_expr_p (const_rtx op, bool strict, const_rtx *tocrel_base_ret,
const_rtx *tocrel_offset_ret)
{
if (!TARGET_TOC)
return false;
if (TARGET_CMODEL != CMODEL_SMALL)
{
if (strict
&& !(GET_CODE (op) == LO_SUM
&& REG_P (XEXP (op, 0))
&& INT_REG_OK_FOR_BASE_P (XEXP (op, 0), strict)))
return false;
if (GET_CODE (op) == LO_SUM)
op = XEXP (op, 1);
}
const_rtx tocrel_base = op;
const_rtx tocrel_offset = const0_rtx;
if (GET_CODE (op) == PLUS && add_cint_operand (XEXP (op, 1), GET_MODE (op)))
{
tocrel_base = XEXP (op, 0);
tocrel_offset = XEXP (op, 1);
}
if (tocrel_base_ret)
*tocrel_base_ret = tocrel_base;
if (tocrel_offset_ret)
*tocrel_offset_ret = tocrel_offset;
return (GET_CODE (tocrel_base) == UNSPEC
&& XINT (tocrel_base, 1) == UNSPEC_TOCREL
&& REG_P (XVECEXP (tocrel_base, 0, 1))
&& REGNO (XVECEXP (tocrel_base, 0, 1)) == TOC_REGISTER);
}
bool
legitimate_constant_pool_address_p (const_rtx x, machine_mode mode,
bool strict)
{
const_rtx tocrel_base, tocrel_offset;
return (toc_relative_expr_p (x, strict, &tocrel_base, &tocrel_offset)
&& (TARGET_CMODEL != CMODEL_MEDIUM
|| constant_pool_expr_p (XVECEXP (tocrel_base, 0, 0))
|| mode == QImode
|| offsettable_ok_by_alignment (XVECEXP (tocrel_base, 0, 0),
INTVAL (tocrel_offset), mode)));
}
static bool
legitimate_small_data_p (machine_mode mode, rtx x)
{
return (DEFAULT_ABI == ABI_V4
&& !flag_pic && !TARGET_TOC
&& (GET_CODE (x) == SYMBOL_REF || GET_CODE (x) == CONST)
&& small_data_operand (x, mode));
}
bool
rs6000_legitimate_offset_address_p (machine_mode mode, rtx x,
bool strict, bool worst_case)
{
unsigned HOST_WIDE_INT offset;
unsigned int extra;
if (GET_CODE (x) != PLUS)
return false;
if (!REG_P (XEXP (x, 0)))
return false;
if (!INT_REG_OK_FOR_BASE_P (XEXP (x, 0), strict))
return false;
if (mode_supports_vsx_dform_quad (mode))
return quad_address_p (x, mode, strict);
if (!reg_offset_addressing_ok_p (mode))
return virtual_stack_registers_memory_p (x);
if (legitimate_constant_pool_address_p (x, mode, strict || lra_in_progress))
return true;
if (GET_CODE (XEXP (x, 1)) != CONST_INT)
return false;
offset = INTVAL (XEXP (x, 1));
extra = 0;
switch (mode)
{
case E_V2SImode:
case E_V2SFmode:
return false;
case E_DFmode:
case E_DDmode:
case E_DImode:
if (VECTOR_MEM_VSX_P (mode))
return false;
if (!worst_case)
break;
if (!TARGET_POWERPC64)
extra = 4;
else if (offset & 3)
return false;
break;
case E_TFmode:
case E_IFmode:
case E_KFmode:
case E_TDmode:
case E_TImode:
case E_PTImode:
extra = 8;
if (!worst_case)
break;
if (!TARGET_POWERPC64)
extra = 12;
else if (offset & 3)
return false;
break;
default:
break;
}
offset += 0x8000;
return offset < 0x10000 - extra;
}
bool
legitimate_indexed_address_p (rtx x, int strict)
{
rtx op0, op1;
if (GET_CODE (x) != PLUS)
return false;
op0 = XEXP (x, 0);
op1 = XEXP (x, 1);
return (REG_P (op0) && REG_P (op1)
&& ((INT_REG_OK_FOR_BASE_P (op0, strict)
&& INT_REG_OK_FOR_INDEX_P (op1, strict))
|| (INT_REG_OK_FOR_BASE_P (op1, strict)
&& INT_REG_OK_FOR_INDEX_P (op0, strict))));
}
bool
avoiding_indexed_address_p (machine_mode mode)
{
return (TARGET_AVOID_XFORM && VECTOR_MEM_NONE_P (mode));
}
bool
legitimate_indirect_address_p (rtx x, int strict)
{
return GET_CODE (x) == REG && INT_REG_OK_FOR_BASE_P (x, strict);
}
bool
macho_lo_sum_memory_operand (rtx x, machine_mode mode)
{
if (!TARGET_MACHO || !flag_pic
|| mode != SImode || GET_CODE (x) != MEM)
return false;
x = XEXP (x, 0);
if (GET_CODE (x) != LO_SUM)
return false;
if (GET_CODE (XEXP (x, 0)) != REG)
return false;
if (!INT_REG_OK_FOR_BASE_P (XEXP (x, 0), 0))
return false;
x = XEXP (x, 1);
return CONSTANT_P (x);
}
static bool
legitimate_lo_sum_address_p (machine_mode mode, rtx x, int strict)
{
if (GET_CODE (x) != LO_SUM)
return false;
if (GET_CODE (XEXP (x, 0)) != REG)
return false;
if (!INT_REG_OK_FOR_BASE_P (XEXP (x, 0), strict))
return false;
if (mode_supports_vsx_dform_quad (mode))
return false;
x = XEXP (x, 1);
if (TARGET_ELF || TARGET_MACHO)
{
bool large_toc_ok;
if (DEFAULT_ABI == ABI_V4 && flag_pic)
return false;
large_toc_ok = (lra_in_progress && TARGET_CMODEL != CMODEL_SMALL
&& small_toc_ref (x, VOIDmode));
if (TARGET_TOC && ! large_toc_ok)
return false;
if (GET_MODE_NUNITS (mode) != 1)
return false;
if (GET_MODE_SIZE (mode) > UNITS_PER_WORD
&& !(
TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT
&& (mode == DFmode || mode == DDmode)))
return false;
return CONSTANT_P (x) || large_toc_ok;
}
return false;
}
static rtx
rs6000_legitimize_address (rtx x, rtx oldx ATTRIBUTE_UNUSED,
machine_mode mode)
{
unsigned int extra;
if (!reg_offset_addressing_ok_p (mode)
|| mode_supports_vsx_dform_quad (mode))
{
if (virtual_stack_registers_memory_p (x))
return x;
if (GET_CODE (x) == PLUS && XEXP (x, 1) == const0_rtx)
return force_reg (Pmode, XEXP (x, 0));
else if (GET_CODE (x) == PLUS
&& (mode != TImode || !TARGET_VSX))
return gen_rtx_PLUS (Pmode,
force_reg (Pmode, XEXP (x, 0)),
force_reg (Pmode, XEXP (x, 1)));
else
return force_reg (Pmode, x);
}
if (GET_CODE (x) == SYMBOL_REF)
{
enum tls_model model = SYMBOL_REF_TLS_MODEL (x);
if (model != 0)
return rs6000_legitimize_tls_address (x, model);
}
extra = 0;
switch (mode)
{
case E_TFmode:
case E_TDmode:
case E_TImode:
case E_PTImode:
case E_IFmode:
case E_KFmode:
extra = 8;
break;
default:
break;
}
if (GET_CODE (x) == PLUS
&& GET_CODE (XEXP (x, 0)) == REG
&& GET_CODE (XEXP (x, 1)) == CONST_INT
&& ((unsigned HOST_WIDE_INT) (INTVAL (XEXP (x, 1)) + 0x8000)
>= 0x10000 - extra)
&& !PAIRED_VECTOR_MODE (mode))
{
HOST_WIDE_INT high_int, low_int;
rtx sum;
low_int = ((INTVAL (XEXP (x, 1)) & 0xffff) ^ 0x8000) - 0x8000;
if (low_int >= 0x8000 - extra)
low_int = 0;
high_int = INTVAL (XEXP (x, 1)) - low_int;
sum = force_operand (gen_rtx_PLUS (Pmode, XEXP (x, 0),
GEN_INT (high_int)), 0);
return plus_constant (Pmode, sum, low_int);
}
else if (GET_CODE (x) == PLUS
&& GET_CODE (XEXP (x, 0)) == REG
&& GET_CODE (XEXP (x, 1)) != CONST_INT
&& GET_MODE_NUNITS (mode) == 1
&& (GET_MODE_SIZE (mode) <= UNITS_PER_WORD
|| (
(TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
&& (mode == DFmode || mode == DDmode)))
&& !avoiding_indexed_address_p (mode))
{
return gen_rtx_PLUS (Pmode, XEXP (x, 0),
force_reg (Pmode, force_operand (XEXP (x, 1), 0)));
}
else if (PAIRED_VECTOR_MODE (mode))
{
if (mode == DImode)
return x;
if (GET_CODE (x) == PLUS)
{
rtx op1 = XEXP (x, 0);
rtx op2 = XEXP (x, 1);
rtx y;
op1 = force_reg (Pmode, op1);
op2 = force_reg (Pmode, op2);
y = gen_rtx_PLUS (Pmode, op1, op2);
if ((GET_MODE_SIZE (mode) > 8 || mode == DDmode) && REG_P (op2))
return force_reg (Pmode, y);
else
return y;
}
return force_reg (Pmode, x);
}
else if ((TARGET_ELF
#if TARGET_MACHO
|| !MACHO_DYNAMIC_NO_PIC_P
#endif
)
&& TARGET_32BIT
&& TARGET_NO_TOC
&& ! flag_pic
&& GET_CODE (x) != CONST_INT
&& GET_CODE (x) != CONST_WIDE_INT
&& GET_CODE (x) != CONST_DOUBLE
&& CONSTANT_P (x)
&& GET_MODE_NUNITS (mode) == 1
&& (GET_MODE_SIZE (mode) <= UNITS_PER_WORD
|| (
(TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
&& (mode == DFmode || mode == DDmode))))
{
rtx reg = gen_reg_rtx (Pmode);
if (TARGET_ELF)
emit_insn (gen_elf_high (reg, x));
else
emit_insn (gen_macho_high (reg, x));
return gen_rtx_LO_SUM (Pmode, reg, x);
}
else if (TARGET_TOC
&& GET_CODE (x) == SYMBOL_REF
&& constant_pool_expr_p (x)
&& ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (get_pool_constant (x), Pmode))
return create_TOC_reference (x, NULL_RTX);
else
return x;
}
static rtx
rs6000_debug_legitimize_address (rtx x, rtx oldx, machine_mode mode)
{
rtx ret;
rtx_insn *insns;
start_sequence ();
ret = rs6000_legitimize_address (x, oldx, mode);
insns = get_insns ();
end_sequence ();
if (ret != x)
{
fprintf (stderr,
"\nrs6000_legitimize_address: mode %s, old code %s, "
"new code %s, modified\n",
GET_MODE_NAME (mode), GET_RTX_NAME (GET_CODE (x)),
GET_RTX_NAME (GET_CODE (ret)));
fprintf (stderr, "Original address:\n");
debug_rtx (x);
fprintf (stderr, "oldx:\n");
debug_rtx (oldx);
fprintf (stderr, "New address:\n");
debug_rtx (ret);
if (insns)
{
fprintf (stderr, "Insns added:\n");
debug_rtx_list (insns, 20);
}
}
else
{
fprintf (stderr,
"\nrs6000_legitimize_address: mode %s, code %s, no change:\n",
GET_MODE_NAME (mode), GET_RTX_NAME (GET_CODE (x)));
debug_rtx (x);
}
if (insns)
emit_insn (insns);
return ret;
}
static void rs6000_output_dwarf_dtprel (FILE *, int, rtx) ATTRIBUTE_UNUSED;
static void
rs6000_output_dwarf_dtprel (FILE *file, int size, rtx x)
{
switch (size)
{
case 4:
fputs ("\t.long\t", file);
break;
case 8:
fputs (DOUBLE_INT_ASM_OP, file);
break;
default:
gcc_unreachable ();
}
output_addr_const (file, x);
if (TARGET_ELF)
fputs ("@dtprel+0x8000", file);
else if (TARGET_XCOFF && GET_CODE (x) == SYMBOL_REF)
{
switch (SYMBOL_REF_TLS_MODEL (x))
{
case 0:
break;
case TLS_MODEL_LOCAL_EXEC:
fputs ("@le", file);
break;
case TLS_MODEL_INITIAL_EXEC:
fputs ("@ie", file);
break;
case TLS_MODEL_GLOBAL_DYNAMIC:
case TLS_MODEL_LOCAL_DYNAMIC:
fputs ("@m", file);
break;
default:
gcc_unreachable ();
}
}
}
static bool
rs6000_real_tls_symbol_ref_p (rtx x)
{
return (GET_CODE (x) == SYMBOL_REF
&& SYMBOL_REF_TLS_MODEL (x) >= TLS_MODEL_REAL);
}
static rtx
rs6000_delegitimize_address (rtx orig_x)
{
rtx x, y, offset;
orig_x = delegitimize_mem_from_attrs (orig_x);
x = orig_x;
if (MEM_P (x))
x = XEXP (x, 0);
y = x;
if (TARGET_CMODEL != CMODEL_SMALL
&& GET_CODE (y) == LO_SUM)
y = XEXP (y, 1);
offset = NULL_RTX;
if (GET_CODE (y) == PLUS
&& GET_MODE (y) == Pmode
&& CONST_INT_P (XEXP (y, 1)))
{
offset = XEXP (y, 1);
y = XEXP (y, 0);
}
if (GET_CODE (y) == UNSPEC
&& XINT (y, 1) == UNSPEC_TOCREL)
{
y = XVECEXP (y, 0, 0);
#ifdef HAVE_AS_TLS
if (TARGET_XCOFF
&& GET_CODE (y) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (y)
&& rs6000_real_tls_symbol_ref_p (get_pool_constant (y)))
return orig_x;
#endif
if (offset != NULL_RTX)
y = gen_rtx_PLUS (Pmode, y, offset);
if (!MEM_P (orig_x))
return y;
else
return replace_equiv_address_nv (orig_x, y);
}
if (TARGET_MACHO
&& GET_CODE (orig_x) == LO_SUM
&& GET_CODE (XEXP (orig_x, 1)) == CONST)
{
y = XEXP (XEXP (orig_x, 1), 0);
if (GET_CODE (y) == UNSPEC
&& XINT (y, 1) == UNSPEC_MACHOPIC_OFFSET)
return XVECEXP (y, 0, 0);
}
return orig_x;
}
static bool
rs6000_const_not_ok_for_debug_p (rtx x)
{
if (GET_CODE (x) == UNSPEC)
return true;
if (GET_CODE (x) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x))
{
rtx c = get_pool_constant (x);
machine_mode cmode = get_pool_mode (x);
if (ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (c, cmode))
return true;
}
return false;
}
static bool
rs6000_legitimate_combined_insn (rtx_insn *insn)
{
int icode = INSN_CODE (insn);
if (icode != CODE_FOR_nothing
&& (icode == CODE_FOR_bdz_si
|| icode == CODE_FOR_bdz_di
|| icode == CODE_FOR_bdnz_si
|| icode == CODE_FOR_bdnz_di
|| icode == CODE_FOR_bdztf_si
|| icode == CODE_FOR_bdztf_di
|| icode == CODE_FOR_bdnztf_si
|| icode == CODE_FOR_bdnztf_di))
return false;
return true;
}
static GTY(()) rtx rs6000_tls_symbol;
static rtx
rs6000_tls_get_addr (void)
{
if (!rs6000_tls_symbol)
rs6000_tls_symbol = init_one_libfunc ("__tls_get_addr");
return rs6000_tls_symbol;
}
static GTY(()) rtx rs6000_got_symbol;
static rtx
rs6000_got_sym (void)
{
if (!rs6000_got_symbol)
{
rs6000_got_symbol = gen_rtx_SYMBOL_REF (Pmode, "_GLOBAL_OFFSET_TABLE_");
SYMBOL_REF_FLAGS (rs6000_got_symbol) |= SYMBOL_FLAG_LOCAL;
SYMBOL_REF_FLAGS (rs6000_got_symbol) |= SYMBOL_FLAG_EXTERNAL;
}
return rs6000_got_symbol;
}
static rtx
rs6000_legitimize_tls_address_aix (rtx addr, enum tls_model model)
{
rtx sym, mem, tocref, tlsreg, tmpreg, dest, tlsaddr;
const char *name;
char *tlsname;
name = XSTR (addr, 0);
if (name[strlen (name) - 1] != ']'
&& (TREE_PUBLIC (SYMBOL_REF_DECL (addr))
|| bss_initializer_p (SYMBOL_REF_DECL (addr))))
{
tlsname = XALLOCAVEC (char, strlen (name) + 4);
strcpy (tlsname, name);
strcat (tlsname,
bss_initializer_p (SYMBOL_REF_DECL (addr)) ? "[UL]" : "[TL]");
tlsaddr = copy_rtx (addr);
XSTR (tlsaddr, 0) = ggc_strdup (tlsname);
}
else
tlsaddr = addr;
sym = force_const_mem (GET_MODE (tlsaddr), tlsaddr);
if (constant_pool_expr_p (XEXP (sym, 0))
&& ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (get_pool_constant (XEXP (sym, 0)), Pmode))
{
tocref = create_TOC_reference (XEXP (sym, 0), NULL_RTX);
mem = gen_const_mem (Pmode, tocref);
set_mem_alias_set (mem, get_TOC_alias_set ());
}
else
return sym;
if (model == TLS_MODEL_GLOBAL_DYNAMIC
|| model == TLS_MODEL_LOCAL_DYNAMIC)
{
name = XSTR (XVECEXP (XEXP (mem, 0), 0, 0), 0);
tlsname = XALLOCAVEC (char, strlen (name) + 1);
strcpy (tlsname, "*LCM");
strcat (tlsname, name + 3);
rtx modaddr = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (tlsname));
SYMBOL_REF_FLAGS (modaddr) |= SYMBOL_FLAG_LOCAL;
tocref = create_TOC_reference (modaddr, NULL_RTX);
rtx modmem = gen_const_mem (Pmode, tocref);
set_mem_alias_set (modmem, get_TOC_alias_set ());
rtx modreg = gen_reg_rtx (Pmode);
emit_insn (gen_rtx_SET (modreg, modmem));
tmpreg = gen_reg_rtx (Pmode);
emit_insn (gen_rtx_SET (tmpreg, mem));
dest = gen_reg_rtx (Pmode);
if (TARGET_32BIT)
emit_insn (gen_tls_get_addrsi (dest, modreg, tmpreg));
else
emit_insn (gen_tls_get_addrdi (dest, modreg, tmpreg));
return dest;
}
else if (TARGET_32BIT)
{
tlsreg = gen_reg_rtx (SImode);
emit_insn (gen_tls_get_tpointer (tlsreg));
}
else
tlsreg = gen_rtx_REG (DImode, 13);
tmpreg = gen_reg_rtx (Pmode);
emit_insn (gen_rtx_SET (tmpreg, mem));
set_unique_reg_note (get_last_insn (), REG_EQUAL,
gen_rtx_MINUS (Pmode, addr, tlsreg));
dest = force_reg (Pmode, gen_rtx_PLUS (Pmode, tmpreg, tlsreg));
return dest;
}
static rtx
rs6000_legitimize_tls_address (rtx addr, enum tls_model model)
{
rtx dest, insn;
if (TARGET_XCOFF)
return rs6000_legitimize_tls_address_aix (addr, model);
dest = gen_reg_rtx (Pmode);
if (model == TLS_MODEL_LOCAL_EXEC && rs6000_tls_size == 16)
{
rtx tlsreg;
if (TARGET_64BIT)
{
tlsreg = gen_rtx_REG (Pmode, 13);
insn = gen_tls_tprel_64 (dest, tlsreg, addr);
}
else
{
tlsreg = gen_rtx_REG (Pmode, 2);
insn = gen_tls_tprel_32 (dest, tlsreg, addr);
}
emit_insn (insn);
}
else if (model == TLS_MODEL_LOCAL_EXEC && rs6000_tls_size == 32)
{
rtx tlsreg, tmp;
tmp = gen_reg_rtx (Pmode);
if (TARGET_64BIT)
{
tlsreg = gen_rtx_REG (Pmode, 13);
insn = gen_tls_tprel_ha_64 (tmp, tlsreg, addr);
}
else
{
tlsreg = gen_rtx_REG (Pmode, 2);
insn = gen_tls_tprel_ha_32 (tmp, tlsreg, addr);
}
emit_insn (insn);
if (TARGET_64BIT)
insn = gen_tls_tprel_lo_64 (dest, tmp, addr);
else
insn = gen_tls_tprel_lo_32 (dest, tmp, addr);
emit_insn (insn);
}
else
{
rtx r3, got, tga, tmp1, tmp2, call_insn;
if (TARGET_64BIT)
got = gen_rtx_REG (Pmode, 2);
else
{
if (flag_pic == 1)
got = gen_rtx_REG (Pmode, RS6000_PIC_OFFSET_TABLE_REGNUM);
else
{
rtx gsym = rs6000_got_sym ();
got = gen_reg_rtx (Pmode);
if (flag_pic == 0)
rs6000_emit_move (got, gsym, Pmode);
else
{
rtx mem, lab;
tmp1 = gen_reg_rtx (Pmode);
tmp2 = gen_reg_rtx (Pmode);
mem = gen_const_mem (Pmode, tmp1);
lab = gen_label_rtx ();
emit_insn (gen_load_toc_v4_PIC_1b (gsym, lab));
emit_move_insn (tmp1, gen_rtx_REG (Pmode, LR_REGNO));
if (TARGET_LINK_STACK)
emit_insn (gen_addsi3 (tmp1, tmp1, GEN_INT (4)));
emit_move_insn (tmp2, mem);
rtx_insn *last = emit_insn (gen_addsi3 (got, tmp1, tmp2));
set_unique_reg_note (last, REG_EQUAL, gsym);
}
}
}
if (model == TLS_MODEL_GLOBAL_DYNAMIC)
{
tga = rs6000_tls_get_addr ();
emit_library_call_value (tga, dest, LCT_CONST, Pmode,
const0_rtx, Pmode);
r3 = gen_rtx_REG (Pmode, 3);
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
{
if (TARGET_64BIT)
insn = gen_tls_gd_aix64 (r3, got, addr, tga, const0_rtx);
else
insn = gen_tls_gd_aix32 (r3, got, addr, tga, const0_rtx);
}
else if (DEFAULT_ABI == ABI_V4)
insn = gen_tls_gd_sysvsi (r3, got, addr, tga, const0_rtx);
else
gcc_unreachable ();
call_insn = last_call_insn ();
PATTERN (call_insn) = insn;
if (DEFAULT_ABI == ABI_V4 && TARGET_SECURE_PLT && flag_pic)
use_reg (&CALL_INSN_FUNCTION_USAGE (call_insn),
pic_offset_table_rtx);
}
else if (model == TLS_MODEL_LOCAL_DYNAMIC)
{
tga = rs6000_tls_get_addr ();
tmp1 = gen_reg_rtx (Pmode);
emit_library_call_value (tga, tmp1, LCT_CONST, Pmode,
const0_rtx, Pmode);
r3 = gen_rtx_REG (Pmode, 3);
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
{
if (TARGET_64BIT)
insn = gen_tls_ld_aix64 (r3, got, tga, const0_rtx);
else
insn = gen_tls_ld_aix32 (r3, got, tga, const0_rtx);
}
else if (DEFAULT_ABI == ABI_V4)
insn = gen_tls_ld_sysvsi (r3, got, tga, const0_rtx);
else
gcc_unreachable ();
call_insn = last_call_insn ();
PATTERN (call_insn) = insn;
if (DEFAULT_ABI == ABI_V4 && TARGET_SECURE_PLT && flag_pic)
use_reg (&CALL_INSN_FUNCTION_USAGE (call_insn),
pic_offset_table_rtx);
if (rs6000_tls_size == 16)
{
if (TARGET_64BIT)
insn = gen_tls_dtprel_64 (dest, tmp1, addr);
else
insn = gen_tls_dtprel_32 (dest, tmp1, addr);
}
else if (rs6000_tls_size == 32)
{
tmp2 = gen_reg_rtx (Pmode);
if (TARGET_64BIT)
insn = gen_tls_dtprel_ha_64 (tmp2, tmp1, addr);
else
insn = gen_tls_dtprel_ha_32 (tmp2, tmp1, addr);
emit_insn (insn);
if (TARGET_64BIT)
insn = gen_tls_dtprel_lo_64 (dest, tmp2, addr);
else
insn = gen_tls_dtprel_lo_32 (dest, tmp2, addr);
}
else
{
tmp2 = gen_reg_rtx (Pmode);
if (TARGET_64BIT)
insn = gen_tls_got_dtprel_64 (tmp2, got, addr);
else
insn = gen_tls_got_dtprel_32 (tmp2, got, addr);
emit_insn (insn);
insn = gen_rtx_SET (dest, gen_rtx_PLUS (Pmode, tmp2, tmp1));
}
emit_insn (insn);
}
else
{
tmp2 = gen_reg_rtx (Pmode);
if (TARGET_64BIT)
insn = gen_tls_got_tprel_64 (tmp2, got, addr);
else
insn = gen_tls_got_tprel_32 (tmp2, got, addr);
emit_insn (insn);
if (TARGET_64BIT)
insn = gen_tls_tls_64 (dest, tmp2, addr);
else
insn = gen_tls_tls_32 (dest, tmp2, addr);
emit_insn (insn);
}
}
return dest;
}
static tree
rs6000_init_stack_protect_guard (void)
{
if (rs6000_stack_protector_guard == SSP_GLOBAL)
return default_stack_protect_guard ();
return NULL_TREE;
}
static bool
rs6000_cannot_force_const_mem (machine_mode mode ATTRIBUTE_UNUSED, rtx x)
{
if (GET_CODE (x) == HIGH
&& GET_CODE (XEXP (x, 0)) == UNSPEC)
return true;
if (GET_CODE (x) == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF
&& SYMBOL_REF_TLS_MODEL (XEXP (XEXP (x, 0), 0)) != 0)
return true;
return TARGET_ELF && tls_referenced_p (x);
}
static bool
use_toc_relative_ref (rtx sym, machine_mode mode)
{
return ((constant_pool_expr_p (sym)
&& ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (get_pool_constant (sym),
get_pool_mode (sym)))
|| (TARGET_CMODEL == CMODEL_MEDIUM
&& SYMBOL_REF_LOCAL_P (sym)
&& GET_MODE_SIZE (mode) <= POWERPC64_TOC_POINTER_ALIGNMENT));
}
static rtx
rs6000_legitimize_reload_address (rtx x, machine_mode mode,
int opnum, int type,
int ind_levels ATTRIBUTE_UNUSED, int *win)
{
bool reg_offset_p = reg_offset_addressing_ok_p (mode);
bool quad_offset_p = mode_supports_vsx_dform_quad (mode);
if (reg_offset_p
&& opnum == 1
&& ((mode == DFmode && recog_data.operand_mode[0] == V2DFmode)
|| (mode == DImode && recog_data.operand_mode[0] == V2DImode)
|| (mode == SFmode && recog_data.operand_mode[0] == V4SFmode
&& TARGET_P9_VECTOR)
|| (mode == SImode && recog_data.operand_mode[0] == V4SImode
&& TARGET_P9_VECTOR)))
reg_offset_p = false;
if (GET_CODE (x) == PLUS
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == REG
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == CONST_INT
&& GET_CODE (XEXP (x, 1)) == CONST_INT)
{
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #1:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, GET_MODE (x), VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
if (GET_CODE (x) == LO_SUM
&& GET_CODE (XEXP (x, 0)) == HIGH)
{
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #2:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, Pmode, VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
#if TARGET_MACHO
if (DEFAULT_ABI == ABI_DARWIN && flag_pic
&& GET_CODE (x) == LO_SUM
&& GET_CODE (XEXP (x, 0)) == PLUS
&& XEXP (XEXP (x, 0), 0) == pic_offset_table_rtx
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == HIGH
&& XEXP (XEXP (XEXP (x, 0), 1), 0) == XEXP (x, 1)
&& machopic_operand_p (XEXP (x, 1)))
{
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, Pmode, VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
#endif
if (TARGET_CMODEL != CMODEL_SMALL
&& reg_offset_p
&& !quad_offset_p
&& small_toc_ref (x, VOIDmode))
{
rtx hi = gen_rtx_HIGH (Pmode, copy_rtx (x));
x = gen_rtx_LO_SUM (Pmode, hi, x);
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #3:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, Pmode, VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
if (GET_CODE (x) == PLUS
&& REG_P (XEXP (x, 0))
&& REGNO (XEXP (x, 0)) < FIRST_PSEUDO_REGISTER
&& INT_REG_OK_FOR_BASE_P (XEXP (x, 0), 1)
&& CONST_INT_P (XEXP (x, 1))
&& reg_offset_p
&& !PAIRED_VECTOR_MODE (mode)
&& (quad_offset_p || !VECTOR_MODE_P (mode) || VECTOR_MEM_NONE_P (mode)))
{
HOST_WIDE_INT val = INTVAL (XEXP (x, 1));
HOST_WIDE_INT low = ((val & 0xffff) ^ 0x8000) - 0x8000;
HOST_WIDE_INT high
= (((val - low) & 0xffffffff) ^ 0x80000000) - 0x80000000;
if (high + low != val
|| (quad_offset_p && (low & 0xf)))
{
*win = 0;
return x;
}
x = gen_rtx_PLUS (GET_MODE (x),
gen_rtx_PLUS (GET_MODE (x), XEXP (x, 0),
GEN_INT (high)),
GEN_INT (low));
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #4:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, GET_MODE (x), VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
if (GET_CODE (x) == SYMBOL_REF
&& reg_offset_p
&& !quad_offset_p
&& (!VECTOR_MODE_P (mode) || VECTOR_MEM_NONE_P (mode))
&& !PAIRED_VECTOR_MODE (mode)
#if TARGET_MACHO
&& DEFAULT_ABI == ABI_DARWIN
&& (flag_pic || MACHO_DYNAMIC_NO_PIC_P)
&& machopic_symbol_defined_p (x)
#else
&& DEFAULT_ABI == ABI_V4
&& !flag_pic
#endif
&& !reg_addr[mode].scalar_in_vmx_p
&& mode != TFmode
&& mode != TDmode
&& mode != IFmode
&& mode != KFmode
&& (mode != TImode || !TARGET_VSX)
&& mode != PTImode
&& (mode != DImode || TARGET_POWERPC64)
&& ((mode != DFmode && mode != DDmode) || TARGET_POWERPC64
|| (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)))
{
#if TARGET_MACHO
if (flag_pic)
{
rtx offset = machopic_gen_offset (x);
x = gen_rtx_LO_SUM (GET_MODE (x),
gen_rtx_PLUS (Pmode, pic_offset_table_rtx,
gen_rtx_HIGH (Pmode, offset)), offset);
}
else
#endif
x = gen_rtx_LO_SUM (GET_MODE (x),
gen_rtx_HIGH (Pmode, x), x);
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #5:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, Pmode, VOIDmode, 0, 0,
opnum, (enum reload_type) type);
*win = 1;
return x;
}
if (VECTOR_MEM_ALTIVEC_P (mode)
&& GET_CODE (x) == AND
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == REG
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == CONST_INT
&& GET_CODE (XEXP (x, 1)) == CONST_INT
&& INTVAL (XEXP (x, 1)) == -16)
{
x = XEXP (x, 0);
*win = 1;
return x;
}
if (TARGET_TOC
&& reg_offset_p
&& !quad_offset_p
&& GET_CODE (x) == SYMBOL_REF
&& use_toc_relative_ref (x, mode))
{
x = create_TOC_reference (x, NULL_RTX);
if (TARGET_CMODEL != CMODEL_SMALL)
{
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nlegitimize_reload_address push_reload #6:\n");
debug_rtx (x);
}
push_reload (XEXP (x, 0), NULL_RTX, &XEXP (x, 0), NULL,
BASE_REG_CLASS, Pmode, VOIDmode, 0, 0,
opnum, (enum reload_type) type);
}
*win = 1;
return x;
}
*win = 0;
return x;
}
static rtx
rs6000_debug_legitimize_reload_address (rtx x, machine_mode mode,
int opnum, int type,
int ind_levels, int *win)
{
rtx ret = rs6000_legitimize_reload_address (x, mode, opnum, type,
ind_levels, win);
fprintf (stderr,
"\nrs6000_legitimize_reload_address: mode = %s, opnum = %d, "
"type = %d, ind_levels = %d, win = %d, original addr:\n",
GET_MODE_NAME (mode), opnum, type, ind_levels, *win);
debug_rtx (x);
if (x == ret)
fprintf (stderr, "Same address returned\n");
else if (!ret)
fprintf (stderr, "NULL returned\n");
else
{
fprintf (stderr, "New address:\n");
debug_rtx (ret);
}
return ret;
}
static bool
rs6000_legitimate_address_p (machine_mode mode, rtx x, bool reg_ok_strict)
{
bool reg_offset_p = reg_offset_addressing_ok_p (mode);
bool quad_offset_p = mode_supports_vsx_dform_quad (mode);
if (VECTOR_MEM_ALTIVEC_P (mode)
&& GET_CODE (x) == AND
&& GET_CODE (XEXP (x, 1)) == CONST_INT
&& INTVAL (XEXP (x, 1)) == -16)
x = XEXP (x, 0);
if (TARGET_ELF && RS6000_SYMBOL_REF_TLS_P (x))
return 0;
if (legitimate_indirect_address_p (x, reg_ok_strict))
return 1;
if (TARGET_UPDATE
&& (GET_CODE (x) == PRE_INC || GET_CODE (x) == PRE_DEC)
&& mode_supports_pre_incdec_p (mode)
&& legitimate_indirect_address_p (XEXP (x, 0), reg_ok_strict))
return 1;
if (quad_offset_p)
{
if (quad_address_p (x, mode, reg_ok_strict))
return 1;
}
else if (virtual_stack_registers_memory_p (x))
return 1;
else if (reg_offset_p)
{
if (legitimate_small_data_p (mode, x))
return 1;
if (legitimate_constant_pool_address_p (x, mode,
reg_ok_strict || lra_in_progress))
return 1;
if (reg_addr[mode].fused_toc && GET_CODE (x) == UNSPEC
&& XINT (x, 1) == UNSPEC_FUSION_ADDIS)
return 1;
}
if (mode == TImode && TARGET_VSX)
return 0;
if (! reg_ok_strict
&& reg_offset_p
&& GET_CODE (x) == PLUS
&& GET_CODE (XEXP (x, 0)) == REG
&& (XEXP (x, 0) == virtual_stack_vars_rtx
|| XEXP (x, 0) == arg_pointer_rtx)
&& GET_CODE (XEXP (x, 1)) == CONST_INT)
return 1;
if (rs6000_legitimate_offset_address_p (mode, x, reg_ok_strict, false))
return 1;
if (!FLOAT128_2REG_P (mode)
&& ((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
|| TARGET_POWERPC64
|| (mode != DFmode && mode != DDmode))
&& (TARGET_POWERPC64 || mode != DImode)
&& (mode != TImode || VECTOR_MEM_VSX_P (TImode))
&& mode != PTImode
&& !avoiding_indexed_address_p (mode)
&& legitimate_indexed_address_p (x, reg_ok_strict))
return 1;
if (TARGET_UPDATE && GET_CODE (x) == PRE_MODIFY
&& mode_supports_pre_modify_p (mode)
&& legitimate_indirect_address_p (XEXP (x, 0), reg_ok_strict)
&& (rs6000_legitimate_offset_address_p (mode, XEXP (x, 1),
reg_ok_strict, false)
|| (!avoiding_indexed_address_p (mode)
&& legitimate_indexed_address_p (XEXP (x, 1), reg_ok_strict)))
&& rtx_equal_p (XEXP (XEXP (x, 1), 0), XEXP (x, 0)))
return 1;
if (reg_offset_p && !quad_offset_p
&& legitimate_lo_sum_address_p (mode, x, reg_ok_strict))
return 1;
return 0;
}
static bool
rs6000_debug_legitimate_address_p (machine_mode mode, rtx x,
bool reg_ok_strict)
{
bool ret = rs6000_legitimate_address_p (mode, x, reg_ok_strict);
fprintf (stderr,
"\nrs6000_legitimate_address_p: return = %s, mode = %s, "
"strict = %d, reload = %s, code = %s\n",
ret ? "true" : "false",
GET_MODE_NAME (mode),
reg_ok_strict,
(reload_completed ? "after" : "before"),
GET_RTX_NAME (GET_CODE (x)));
debug_rtx (x);
return ret;
}
static bool
rs6000_mode_dependent_address_p (const_rtx addr,
addr_space_t as ATTRIBUTE_UNUSED)
{
return rs6000_mode_dependent_address_ptr (addr);
}
static bool
rs6000_mode_dependent_address (const_rtx addr)
{
switch (GET_CODE (addr))
{
case PLUS:
if (XEXP (addr, 0) != virtual_stack_vars_rtx
&& XEXP (addr, 0) != arg_pointer_rtx
&& GET_CODE (XEXP (addr, 1)) == CONST_INT)
{
unsigned HOST_WIDE_INT val = INTVAL (XEXP (addr, 1));
return val + 0x8000 >= 0x10000 - (TARGET_POWERPC64 ? 8 : 12);
}
break;
case LO_SUM:
return !legitimate_constant_pool_address_p (addr, QImode, false);
case PRE_MODIFY:
return TARGET_UPDATE;
case AND:
return true;
default:
break;
}
return false;
}
static bool
rs6000_debug_mode_dependent_address (const_rtx addr)
{
bool ret = rs6000_mode_dependent_address (addr);
fprintf (stderr, "\nrs6000_mode_dependent_address: ret = %s\n",
ret ? "true" : "false");
debug_rtx (addr);
return ret;
}
rtx
rs6000_find_base_term (rtx op)
{
rtx base;
base = op;
if (GET_CODE (base) == CONST)
base = XEXP (base, 0);
if (GET_CODE (base) == PLUS)
base = XEXP (base, 0);
if (GET_CODE (base) == UNSPEC)
switch (XINT (base, 1))
{
case UNSPEC_TOCREL:
case UNSPEC_MACHOPIC_OFFSET:
return XVECEXP (base, 0, 0);
}
return op;
}
static bool
rs6000_offsettable_memref_p (rtx op, machine_mode reg_mode, bool strict)
{
bool worst_case;
if (!MEM_P (op))
return false;
if (offsettable_address_p (strict, GET_MODE (op), XEXP (op, 0)))
return true;
worst_case = ((TARGET_POWERPC64 && GET_MODE_CLASS (reg_mode) == MODE_INT)
|| GET_MODE_SIZE (reg_mode) == 4);
return rs6000_legitimate_offset_address_p (GET_MODE (op), XEXP (op, 0),
strict, worst_case);
}
static int
rs6000_reassociation_width (unsigned int opc ATTRIBUTE_UNUSED,
machine_mode mode)
{
switch (rs6000_tune)
{
case PROCESSOR_POWER8:
case PROCESSOR_POWER9:
if (DECIMAL_FLOAT_MODE_P (mode))
return 1;
if (VECTOR_MODE_P (mode))
return 4;
if (INTEGRAL_MODE_P (mode)) 
return 1;
if (FLOAT_MODE_P (mode))
return 4;
break;
default:
break;
}
return 1;
}
static void
rs6000_conditional_register_usage (void)
{
int i;
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_conditional_register_usage called\n");
fixed_regs[64] = 1;
if (TARGET_64BIT)
fixed_regs[13] = call_used_regs[13]
= call_really_used_regs[13] = 1;
if (TARGET_SOFT_FLOAT)
for (i = 32; i < 64; i++)
fixed_regs[i] = call_used_regs[i]
= call_really_used_regs[i] = 1;
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
call_really_used_regs[2] = 0;
if (DEFAULT_ABI == ABI_V4 && flag_pic == 2)
fixed_regs[RS6000_PIC_OFFSET_TABLE_REGNUM] = 1;
if (DEFAULT_ABI == ABI_V4 && flag_pic == 1)
fixed_regs[RS6000_PIC_OFFSET_TABLE_REGNUM]
= call_used_regs[RS6000_PIC_OFFSET_TABLE_REGNUM]
= call_really_used_regs[RS6000_PIC_OFFSET_TABLE_REGNUM] = 1;
if (DEFAULT_ABI == ABI_DARWIN && flag_pic)
fixed_regs[RS6000_PIC_OFFSET_TABLE_REGNUM]
= call_used_regs[RS6000_PIC_OFFSET_TABLE_REGNUM]
= call_really_used_regs[RS6000_PIC_OFFSET_TABLE_REGNUM] = 1;
if (TARGET_TOC && TARGET_MINIMAL_TOC)
fixed_regs[RS6000_PIC_OFFSET_TABLE_REGNUM]
= call_used_regs[RS6000_PIC_OFFSET_TABLE_REGNUM] = 1;
if (!TARGET_ALTIVEC && !TARGET_VSX)
{
for (i = FIRST_ALTIVEC_REGNO; i <= LAST_ALTIVEC_REGNO; ++i)
fixed_regs[i] = call_used_regs[i] = call_really_used_regs[i] = 1;
call_really_used_regs[VRSAVE_REGNO] = 1;
}
if (TARGET_ALTIVEC || TARGET_VSX)
global_regs[VSCR_REGNO] = 1;
if (TARGET_ALTIVEC_ABI)
{
for (i = FIRST_ALTIVEC_REGNO; i < FIRST_ALTIVEC_REGNO + 20; ++i)
call_used_regs[i] = call_really_used_regs[i] = 1;
if (TARGET_XCOFF)
for (i = FIRST_ALTIVEC_REGNO + 20; i < FIRST_ALTIVEC_REGNO + 32; ++i)
fixed_regs[i] = call_used_regs[i] = call_really_used_regs[i] = 1;
}
}

bool
rs6000_emit_set_const (rtx dest, rtx source)
{
machine_mode mode = GET_MODE (dest);
rtx temp, set;
rtx_insn *insn;
HOST_WIDE_INT c;
gcc_checking_assert (CONST_INT_P (source));
c = INTVAL (source);
switch (mode)
{
case E_QImode:
case E_HImode:
emit_insn (gen_rtx_SET (dest, source));
return true;
case E_SImode:
temp = !can_create_pseudo_p () ? dest : gen_reg_rtx (SImode);
emit_insn (gen_rtx_SET (copy_rtx (temp),
GEN_INT (c & ~(HOST_WIDE_INT) 0xffff)));
emit_insn (gen_rtx_SET (dest,
gen_rtx_IOR (SImode, copy_rtx (temp),
GEN_INT (c & 0xffff))));
break;
case E_DImode:
if (!TARGET_POWERPC64)
{
rtx hi, lo;
hi = operand_subword_force (copy_rtx (dest), WORDS_BIG_ENDIAN == 0,
DImode);
lo = operand_subword_force (dest, WORDS_BIG_ENDIAN != 0,
DImode);
emit_move_insn (hi, GEN_INT (c >> 32));
c = ((c & 0xffffffff) ^ 0x80000000) - 0x80000000;
emit_move_insn (lo, GEN_INT (c));
}
else
rs6000_emit_set_long_const (dest, c);
break;
default:
gcc_unreachable ();
}
insn = get_last_insn ();
set = single_set (insn);
if (! CONSTANT_P (SET_SRC (set)))
set_unique_reg_note (insn, REG_EQUAL, GEN_INT (c));
return true;
}
static void
rs6000_emit_set_long_const (rtx dest, HOST_WIDE_INT c)
{
rtx temp;
HOST_WIDE_INT ud1, ud2, ud3, ud4;
ud1 = c & 0xffff;
c = c >> 16;
ud2 = c & 0xffff;
c = c >> 16;
ud3 = c & 0xffff;
c = c >> 16;
ud4 = c & 0xffff;
if ((ud4 == 0xffff && ud3 == 0xffff && ud2 == 0xffff && (ud1 & 0x8000))
|| (ud4 == 0 && ud3 == 0 && ud2 == 0 && ! (ud1 & 0x8000)))
emit_move_insn (dest, GEN_INT ((ud1 ^ 0x8000) - 0x8000));
else if ((ud4 == 0xffff && ud3 == 0xffff && (ud2 & 0x8000))
|| (ud4 == 0 && ud3 == 0 && ! (ud2 & 0x8000)))
{
temp = !can_create_pseudo_p () ? dest : gen_reg_rtx (DImode);
emit_move_insn (ud1 != 0 ? copy_rtx (temp) : dest,
GEN_INT (((ud2 << 16) ^ 0x80000000) - 0x80000000));
if (ud1 != 0)
emit_move_insn (dest,
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud1)));
}
else if (ud3 == 0 && ud4 == 0)
{
temp = !can_create_pseudo_p () ? dest : gen_reg_rtx (DImode);
gcc_assert (ud2 & 0x8000);
emit_move_insn (copy_rtx (temp),
GEN_INT (((ud2 << 16) ^ 0x80000000) - 0x80000000));
if (ud1 != 0)
emit_move_insn (copy_rtx (temp),
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud1)));
emit_move_insn (dest,
gen_rtx_ZERO_EXTEND (DImode,
gen_lowpart (SImode,
copy_rtx (temp))));
}
else if ((ud4 == 0xffff && (ud3 & 0x8000))
|| (ud4 == 0 && ! (ud3 & 0x8000)))
{
temp = !can_create_pseudo_p () ? dest : gen_reg_rtx (DImode);
emit_move_insn (copy_rtx (temp),
GEN_INT (((ud3 << 16) ^ 0x80000000) - 0x80000000));
if (ud2 != 0)
emit_move_insn (copy_rtx (temp),
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud2)));
emit_move_insn (ud1 != 0 ? copy_rtx (temp) : dest,
gen_rtx_ASHIFT (DImode, copy_rtx (temp),
GEN_INT (16)));
if (ud1 != 0)
emit_move_insn (dest,
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud1)));
}
else
{
temp = !can_create_pseudo_p () ? dest : gen_reg_rtx (DImode);
emit_move_insn (copy_rtx (temp),
GEN_INT (((ud4 << 16) ^ 0x80000000) - 0x80000000));
if (ud3 != 0)
emit_move_insn (copy_rtx (temp),
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud3)));
emit_move_insn (ud2 != 0 || ud1 != 0 ? copy_rtx (temp) : dest,
gen_rtx_ASHIFT (DImode, copy_rtx (temp),
GEN_INT (32)));
if (ud2 != 0)
emit_move_insn (ud1 != 0 ? copy_rtx (temp) : dest,
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud2 << 16)));
if (ud1 != 0)
emit_move_insn (dest,
gen_rtx_IOR (DImode, copy_rtx (temp),
GEN_INT (ud1)));
}
}
static void
rs6000_eliminate_indexed_memrefs (rtx operands[2])
{
if (GET_CODE (operands[0]) == MEM
&& GET_CODE (XEXP (operands[0], 0)) != REG
&& ! legitimate_constant_pool_address_p (XEXP (operands[0], 0),
GET_MODE (operands[0]), false))
operands[0]
= replace_equiv_address (operands[0],
copy_addr_to_reg (XEXP (operands[0], 0)));
if (GET_CODE (operands[1]) == MEM
&& GET_CODE (XEXP (operands[1], 0)) != REG
&& ! legitimate_constant_pool_address_p (XEXP (operands[1], 0),
GET_MODE (operands[1]), false))
operands[1]
= replace_equiv_address (operands[1],
copy_addr_to_reg (XEXP (operands[1], 0)));
}
static rtvec
rs6000_const_vec (machine_mode mode)
{
int i, subparts;
rtvec v;
switch (mode)
{
case E_V1TImode:
subparts = 1;
break;
case E_V2DFmode:
case E_V2DImode:
subparts = 2;
break;
case E_V4SFmode:
case E_V4SImode:
subparts = 4;
break;
case E_V8HImode:
subparts = 8;
break;
case E_V16QImode:
subparts = 16;
break;
default:
gcc_unreachable();
}
v = rtvec_alloc (subparts);
for (i = 0; i < subparts / 2; ++i)
RTVEC_ELT (v, i) = gen_rtx_CONST_INT (DImode, i + subparts / 2);
for (i = subparts / 2; i < subparts; ++i)
RTVEC_ELT (v, i) = gen_rtx_CONST_INT (DImode, i - subparts / 2);
return v;
}
void
rs6000_emit_le_vsx_permute (rtx dest, rtx source, machine_mode mode)
{
if (FLOAT128_VECTOR_P (mode))
{
dest = gen_lowpart (V1TImode, dest);
source = gen_lowpart (V1TImode, source);
mode = V1TImode;
}
if (mode == TImode || mode == V1TImode)
emit_insn (gen_rtx_SET (dest, gen_rtx_ROTATE (mode, source,
GEN_INT (64))));
else
{
rtx par = gen_rtx_PARALLEL (VOIDmode, rs6000_const_vec (mode));
emit_insn (gen_rtx_SET (dest, gen_rtx_VEC_SELECT (mode, source, par)));
}
}
void
rs6000_emit_le_vsx_load (rtx dest, rtx source, machine_mode mode)
{
if (mode == TImode || mode == V1TImode)
{
mode = V2DImode;
dest = gen_lowpart (V2DImode, dest);
source = adjust_address (source, V2DImode, 0);
}
rtx tmp = can_create_pseudo_p () ? gen_reg_rtx_and_attrs (dest) : dest;
rs6000_emit_le_vsx_permute (tmp, source, mode);
rs6000_emit_le_vsx_permute (dest, tmp, mode);
}
void
rs6000_emit_le_vsx_store (rtx dest, rtx source, machine_mode mode)
{
gcc_assert (!lra_in_progress && !reload_completed);
if (mode == TImode || mode == V1TImode)
{
mode = V2DImode;
dest = adjust_address (dest, V2DImode, 0);
source = gen_lowpart (V2DImode, source);
}
rtx tmp = can_create_pseudo_p () ? gen_reg_rtx_and_attrs (source) : source;
rs6000_emit_le_vsx_permute (tmp, source, mode);
rs6000_emit_le_vsx_permute (dest, tmp, mode);
}
void
rs6000_emit_le_vsx_move (rtx dest, rtx source, machine_mode mode)
{
gcc_assert (!BYTES_BIG_ENDIAN
&& VECTOR_MEM_VSX_P (mode)
&& !TARGET_P9_VECTOR
&& !gpr_or_gpr_p (dest, source)
&& (MEM_P (source) ^ MEM_P (dest)));
if (MEM_P (source))
{
gcc_assert (REG_P (dest) || GET_CODE (dest) == SUBREG);
rs6000_emit_le_vsx_load (dest, source, mode);
}
else
{
if (!REG_P (source))
source = force_reg (mode, source);
rs6000_emit_le_vsx_store (dest, source, mode);
}
}
bool
valid_sf_si_move (rtx dest, rtx src, machine_mode mode)
{
if (TARGET_ALLOW_SF_SUBREG)
return true;
if (mode != SFmode && GET_MODE_CLASS (mode) != MODE_INT)
return true;
if (!SUBREG_P (src) || !sf_subreg_operand (src, mode))
return true;
if (SUBREG_P (dest))
{
rtx dest_subreg = SUBREG_REG (dest);
rtx src_subreg = SUBREG_REG (src);
return GET_MODE (dest_subreg) == GET_MODE (src_subreg);
}
return false;
}
static bool
rs6000_emit_move_si_sf_subreg (rtx dest, rtx source, machine_mode mode)
{
if (TARGET_DIRECT_MOVE_64BIT && !lra_in_progress && !reload_completed
&& (!SUBREG_P (dest) || !sf_subreg_operand (dest, mode))
&& SUBREG_P (source) && sf_subreg_operand (source, mode))
{
rtx inner_source = SUBREG_REG (source);
machine_mode inner_mode = GET_MODE (inner_source);
if (mode == SImode && inner_mode == SFmode)
{
emit_insn (gen_movsi_from_sf (dest, inner_source));
return true;
}
if (mode == SFmode && inner_mode == SImode)
{
emit_insn (gen_movsf_from_si (dest, inner_source));
return true;
}
}
return false;
}
void
rs6000_emit_move (rtx dest, rtx source, machine_mode mode)
{
rtx operands[2];
operands[0] = dest;
operands[1] = source;
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr,
"\nrs6000_emit_move: mode = %s, lra_in_progress = %d, "
"reload_completed = %d, can_create_pseudos = %d.\ndest:\n",
GET_MODE_NAME (mode),
lra_in_progress,
reload_completed,
can_create_pseudo_p ());
debug_rtx (dest);
fprintf (stderr, "source:\n");
debug_rtx (source);
}
if (CONST_WIDE_INT_P (operands[1])
&& GET_MODE_BITSIZE (mode) <= HOST_BITS_PER_WIDE_INT)
{
gcc_unreachable ();
}
#ifdef HAVE_AS_GNU_ATTRIBUTE
if (rs6000_gnu_attr && (HAVE_LD_PPC_GNU_ATTR_LONG_DOUBLE || TARGET_64BIT))
{
if (TARGET_LONG_DOUBLE_128 && (mode == TFmode || mode == TCmode))
rs6000_passes_float = rs6000_passes_long_double = true;
else if (!TARGET_LONG_DOUBLE_128 && (mode == DFmode || mode == DCmode))
rs6000_passes_float = rs6000_passes_long_double = true;
}
#endif
if ((mode == SImode || mode == SFmode) && SUBREG_P (source)
&& rs6000_emit_move_si_sf_subreg (dest, source, mode))
return;
if (GET_CODE (operands[0]) == MEM
&& GET_CODE (operands[1]) == MEM
&& mode == DImode
&& (rs6000_slow_unaligned_access (DImode, MEM_ALIGN (operands[0]))
|| rs6000_slow_unaligned_access (DImode, MEM_ALIGN (operands[1])))
&& ! (rs6000_slow_unaligned_access (SImode,
(MEM_ALIGN (operands[0]) > 32
? 32 : MEM_ALIGN (operands[0])))
|| rs6000_slow_unaligned_access (SImode,
(MEM_ALIGN (operands[1]) > 32
? 32 : MEM_ALIGN (operands[1]))))
&& ! MEM_VOLATILE_P (operands [0])
&& ! MEM_VOLATILE_P (operands [1]))
{
emit_move_insn (adjust_address (operands[0], SImode, 0),
adjust_address (operands[1], SImode, 0));
emit_move_insn (adjust_address (copy_rtx (operands[0]), SImode, 4),
adjust_address (copy_rtx (operands[1]), SImode, 4));
return;
}
if (can_create_pseudo_p () && GET_CODE (operands[0]) == MEM
&& !gpc_reg_operand (operands[1], mode))
operands[1] = force_reg (mode, operands[1]);
if (tls_referenced_p (operands[1]))
{
enum tls_model model;
rtx tmp = operands[1];
rtx addend = NULL;
if (GET_CODE (tmp) == CONST && GET_CODE (XEXP (tmp, 0)) == PLUS)
{
addend = XEXP (XEXP (tmp, 0), 1);
tmp = XEXP (XEXP (tmp, 0), 0);
}
gcc_assert (GET_CODE (tmp) == SYMBOL_REF);
model = SYMBOL_REF_TLS_MODEL (tmp);
gcc_assert (model != 0);
tmp = rs6000_legitimize_tls_address (tmp, model);
if (addend)
{
tmp = gen_rtx_PLUS (mode, tmp, addend);
tmp = force_operand (tmp, operands[0]);
}
operands[1] = tmp;
}
if (FLOAT128_IBM_P (mode) && !reg_addr[DFmode].scalar_in_vmx_p
&& GET_CODE (operands[1]) == CONST_DOUBLE)
{
rs6000_emit_move (simplify_gen_subreg (DFmode, operands[0], mode, 0),
simplify_gen_subreg (DFmode, operands[1], mode, 0),
DFmode);
rs6000_emit_move (simplify_gen_subreg (DFmode, operands[0], mode,
GET_MODE_SIZE (DFmode)),
simplify_gen_subreg (DFmode, operands[1], mode,
GET_MODE_SIZE (DFmode)),
DFmode);
return;
}
if (lra_in_progress && mode == DDmode
&& REG_P (operands[0]) && REGNO (operands[0]) >= FIRST_PSEUDO_REGISTER
&& reg_preferred_class (REGNO (operands[0])) == NO_REGS
&& GET_CODE (operands[1]) == SUBREG && REG_P (SUBREG_REG (operands[1]))
&& GET_MODE (SUBREG_REG (operands[1])) == SDmode)
{
enum reg_class cl;
int regno = REGNO (SUBREG_REG (operands[1]));
if (regno >= FIRST_PSEUDO_REGISTER)
{
cl = reg_preferred_class (regno);
regno = reg_renumber[regno];
if (regno < 0)
regno = cl == NO_REGS ? -1 : ira_class_hard_regs[cl][1];
}
if (regno >= 0 && ! FP_REGNO_P (regno))
{
mode = SDmode;
operands[0] = gen_lowpart_SUBREG (SDmode, operands[0]);
operands[1] = SUBREG_REG (operands[1]);
}
}
if (lra_in_progress
&& mode == SDmode
&& REG_P (operands[0]) && REGNO (operands[0]) >= FIRST_PSEUDO_REGISTER
&& reg_preferred_class (REGNO (operands[0])) == NO_REGS
&& (REG_P (operands[1])
|| (GET_CODE (operands[1]) == SUBREG
&& REG_P (SUBREG_REG (operands[1])))))
{
int regno = REGNO (GET_CODE (operands[1]) == SUBREG
? SUBREG_REG (operands[1]) : operands[1]);
enum reg_class cl;
if (regno >= FIRST_PSEUDO_REGISTER)
{
cl = reg_preferred_class (regno);
gcc_assert (cl != NO_REGS);
regno = reg_renumber[regno];
if (regno < 0)
regno = ira_class_hard_regs[cl][0];
}
if (FP_REGNO_P (regno))
{
if (GET_MODE (operands[0]) != DDmode)
operands[0] = gen_rtx_SUBREG (DDmode, operands[0], 0);
emit_insn (gen_movsd_store (operands[0], operands[1]));
}
else if (INT_REGNO_P (regno))
emit_insn (gen_movsd_hardfloat (operands[0], operands[1]));
else
gcc_unreachable();
return;
}
if (lra_in_progress && mode == DDmode
&& GET_CODE (operands[0]) == SUBREG && REG_P (SUBREG_REG (operands[0]))
&& GET_MODE (SUBREG_REG (operands[0])) == SDmode
&& REG_P (operands[1]) && REGNO (operands[1]) >= FIRST_PSEUDO_REGISTER
&& reg_preferred_class (REGNO (operands[1])) == NO_REGS)
{
enum reg_class cl;
int regno = REGNO (SUBREG_REG (operands[0]));
if (regno >= FIRST_PSEUDO_REGISTER)
{
cl = reg_preferred_class (regno);
regno = reg_renumber[regno];
if (regno < 0)
regno = cl == NO_REGS ? -1 : ira_class_hard_regs[cl][0];
}
if (regno >= 0 && ! FP_REGNO_P (regno))
{
mode = SDmode;
operands[0] = SUBREG_REG (operands[0]);
operands[1] = gen_lowpart_SUBREG (SDmode, operands[1]);
}
}
if (lra_in_progress
&& mode == SDmode
&& (REG_P (operands[0])
|| (GET_CODE (operands[0]) == SUBREG
&& REG_P (SUBREG_REG (operands[0]))))
&& REG_P (operands[1]) && REGNO (operands[1]) >= FIRST_PSEUDO_REGISTER
&& reg_preferred_class (REGNO (operands[1])) == NO_REGS)
{
int regno = REGNO (GET_CODE (operands[0]) == SUBREG
? SUBREG_REG (operands[0]) : operands[0]);
enum reg_class cl;
if (regno >= FIRST_PSEUDO_REGISTER)
{
cl = reg_preferred_class (regno);
gcc_assert (cl != NO_REGS);
regno = reg_renumber[regno];
if (regno < 0)
regno = ira_class_hard_regs[cl][0];
}
if (FP_REGNO_P (regno))
{
if (GET_MODE (operands[1]) != DDmode)
operands[1] = gen_rtx_SUBREG (DDmode, operands[1], 0);
emit_insn (gen_movsd_load (operands[0], operands[1]));
}
else if (INT_REGNO_P (regno))
emit_insn (gen_movsd_hardfloat (operands[0], operands[1]));
else
gcc_unreachable();
return;
}
switch (mode)
{
case E_HImode:
case E_QImode:
if (CONSTANT_P (operands[1])
&& GET_CODE (operands[1]) != CONST_INT)
operands[1] = force_const_mem (mode, operands[1]);
break;
case E_TFmode:
case E_TDmode:
case E_IFmode:
case E_KFmode:
if (FLOAT128_2REG_P (mode))
rs6000_eliminate_indexed_memrefs (operands);
case E_DFmode:
case E_DDmode:
case E_SFmode:
case E_SDmode:
if (CONSTANT_P (operands[1])
&& ! easy_fp_constant (operands[1], mode))
operands[1] = force_const_mem (mode, operands[1]);
break;
case E_V16QImode:
case E_V8HImode:
case E_V4SFmode:
case E_V4SImode:
case E_V2SFmode:
case E_V2SImode:
case E_V2DFmode:
case E_V2DImode:
case E_V1TImode:
if (CONSTANT_P (operands[1])
&& !easy_vector_constant (operands[1], mode))
operands[1] = force_const_mem (mode, operands[1]);
break;
case E_SImode:
case E_DImode:
if (TARGET_ELF
&& mode == Pmode
&& DEFAULT_ABI == ABI_V4
&& (GET_CODE (operands[1]) == SYMBOL_REF
|| GET_CODE (operands[1]) == CONST)
&& small_data_operand (operands[1], mode))
{
emit_insn (gen_rtx_SET (operands[0], operands[1]));
return;
}
if (DEFAULT_ABI == ABI_V4
&& mode == Pmode && mode == SImode
&& flag_pic == 1 && got_operand (operands[1], mode))
{
emit_insn (gen_movsi_got (operands[0], operands[1]));
return;
}
if ((TARGET_ELF || DEFAULT_ABI == ABI_DARWIN)
&& TARGET_NO_TOC
&& ! flag_pic
&& mode == Pmode
&& CONSTANT_P (operands[1])
&& GET_CODE (operands[1]) != HIGH
&& GET_CODE (operands[1]) != CONST_INT)
{
rtx target = (!can_create_pseudo_p ()
? operands[0]
: gen_reg_rtx (mode));
if (DEFAULT_ABI == ABI_AIX
&& GET_CODE (operands[1]) == SYMBOL_REF
&& XSTR (operands[1], 0)[0] == '.')
{
const char *name = XSTR (operands[1], 0);
rtx new_ref;
while (*name == '.')
name++;
new_ref = gen_rtx_SYMBOL_REF (Pmode, name);
CONSTANT_POOL_ADDRESS_P (new_ref)
= CONSTANT_POOL_ADDRESS_P (operands[1]);
SYMBOL_REF_FLAGS (new_ref) = SYMBOL_REF_FLAGS (operands[1]);
SYMBOL_REF_USED (new_ref) = SYMBOL_REF_USED (operands[1]);
SYMBOL_REF_DATA (new_ref) = SYMBOL_REF_DATA (operands[1]);
operands[1] = new_ref;
}
if (DEFAULT_ABI == ABI_DARWIN)
{
#if TARGET_MACHO
if (MACHO_DYNAMIC_NO_PIC_P)
{
operands[1] = rs6000_machopic_legitimize_pic_address (
operands[1], mode, operands[0]);
if (operands[0] != operands[1])
emit_insn (gen_rtx_SET (operands[0], operands[1]));
return;
}
#endif
emit_insn (gen_macho_high (target, operands[1]));
emit_insn (gen_macho_low (operands[0], target, operands[1]));
return;
}
emit_insn (gen_elf_high (target, operands[1]));
emit_insn (gen_elf_low (operands[0], target, operands[1]));
return;
}
if (TARGET_TOC
&& GET_CODE (operands[1]) == SYMBOL_REF
&& use_toc_relative_ref (operands[1], mode))
operands[1] = create_TOC_reference (operands[1], operands[0]);
else if (mode == Pmode
&& CONSTANT_P (operands[1])
&& GET_CODE (operands[1]) != HIGH
&& ((GET_CODE (operands[1]) != CONST_INT
&& ! easy_fp_constant (operands[1], mode))
|| (GET_CODE (operands[1]) == CONST_INT
&& (num_insns_constant (operands[1], mode)
> (TARGET_CMODEL != CMODEL_SMALL ? 3 : 2)))
|| (GET_CODE (operands[0]) == REG
&& FP_REGNO_P (REGNO (operands[0]))))
&& !toc_relative_expr_p (operands[1], false, NULL, NULL)
&& (TARGET_CMODEL == CMODEL_SMALL
|| can_create_pseudo_p ()
|| (REG_P (operands[0])
&& INT_REG_OK_FOR_BASE_P (operands[0], true))))
{
#if TARGET_MACHO
if (DEFAULT_ABI == ABI_DARWIN && MACHOPIC_INDIRECT)
{
operands[1] =
rs6000_machopic_legitimize_pic_address (operands[1], mode,
operands[0]);
if (operands[0] != operands[1])
emit_insn (gen_rtx_SET (operands[0], operands[1]));
return;
}
#endif
if (GET_CODE (operands[1]) == CONST
&& TARGET_NO_SUM_IN_TOC
&& GET_CODE (XEXP (operands[1], 0)) == PLUS
&& add_operand (XEXP (XEXP (operands[1], 0), 1), mode)
&& (GET_CODE (XEXP (XEXP (operands[1], 0), 0)) == LABEL_REF
|| GET_CODE (XEXP (XEXP (operands[1], 0), 0)) == SYMBOL_REF)
&& ! side_effects_p (operands[0]))
{
rtx sym =
force_const_mem (mode, XEXP (XEXP (operands[1], 0), 0));
rtx other = XEXP (XEXP (operands[1], 0), 1);
sym = force_reg (mode, sym);
emit_insn (gen_add3_insn (operands[0], sym, other));
return;
}
operands[1] = force_const_mem (mode, operands[1]);
if (TARGET_TOC
&& GET_CODE (XEXP (operands[1], 0)) == SYMBOL_REF
&& use_toc_relative_ref (XEXP (operands[1], 0), mode))
{
rtx tocref = create_TOC_reference (XEXP (operands[1], 0),
operands[0]);
operands[1] = gen_const_mem (mode, tocref);
set_mem_alias_set (operands[1], get_TOC_alias_set ());
}
}
break;
case E_TImode:
if (!VECTOR_MEM_VSX_P (TImode))
rs6000_eliminate_indexed_memrefs (operands);
break;
case E_PTImode:
rs6000_eliminate_indexed_memrefs (operands);
break;
default:
fatal_insn ("bad move", gen_rtx_SET (dest, source));
}
if (GET_CODE (operands[1]) == MEM)
operands[1] = validize_mem (operands[1]);
emit_insn (gen_rtx_SET (operands[0], operands[1]));
}

#define USE_FP_FOR_ARG_P(CUM,MODE)		\
(SCALAR_FLOAT_MODE_NOT_VECTOR_P (MODE)		\
&& (CUM)->fregno <= FP_ARG_MAX_REG		\
&& TARGET_HARD_FLOAT)
#define USE_ALTIVEC_FOR_ARG_P(CUM,MODE,NAMED)			\
(ALTIVEC_OR_VSX_VECTOR_MODE (MODE)				\
&& (CUM)->vregno <= ALTIVEC_ARG_MAX_REG			\
&& TARGET_ALTIVEC_ABI					\
&& (NAMED))
static int
rs6000_aggregate_candidate (const_tree type, machine_mode *modep)
{
machine_mode mode;
HOST_WIDE_INT size;
switch (TREE_CODE (type))
{
case REAL_TYPE:
mode = TYPE_MODE (type);
if (!SCALAR_FLOAT_MODE_P (mode))
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 1;
break;
case COMPLEX_TYPE:
mode = TYPE_MODE (TREE_TYPE (type));
if (!SCALAR_FLOAT_MODE_P (mode))
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 2;
break;
case VECTOR_TYPE:
if (!TARGET_ALTIVEC_ABI || !TARGET_ALTIVEC)
return -1;
size = int_size_in_bytes (type);
switch (size)
{
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
count = rs6000_aggregate_candidate (TREE_TYPE (type), modep);
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
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
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
sub_count = rs6000_aggregate_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count += sub_count;
}
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
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
sub_count = rs6000_aggregate_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count = count > sub_count ? count : sub_count;
}
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
return -1;
return count;
}
default:
break;
}
return -1;
}
static bool
rs6000_discover_homogeneous_aggregate (machine_mode mode, const_tree type,
machine_mode *elt_mode,
int *n_elts)
{
if (TARGET_HARD_FLOAT && DEFAULT_ABI == ABI_ELFv2 && type
&& AGGREGATE_TYPE_P (type))
{
machine_mode field_mode = VOIDmode;
int field_count = rs6000_aggregate_candidate (type, &field_mode);
if (field_count > 0)
{
int reg_size = ALTIVEC_OR_VSX_VECTOR_MODE (field_mode) ? 16 : 8;
int field_size = ROUND_UP (GET_MODE_SIZE (field_mode), reg_size);
if (field_count * field_size <= AGGR_ARG_NUM_REG * reg_size)
{
if (elt_mode)
*elt_mode = field_mode;
if (n_elts)
*n_elts = field_count;
return true;
}
}
}
if (elt_mode)
*elt_mode = mode;
if (n_elts)
*n_elts = 1;
return false;
}
static bool
rs6000_return_in_memory (const_tree type, const_tree fntype ATTRIBUTE_UNUSED)
{
if (TARGET_MACHO
&& rs6000_darwin64_abi
&& TREE_CODE (type) == RECORD_TYPE
&& int_size_in_bytes (type) > 0)
{
CUMULATIVE_ARGS valcum;
rtx valret;
valcum.words = 0;
valcum.fregno = FP_ARG_MIN_REG;
valcum.vregno = ALTIVEC_ARG_MIN_REG;
valret = rs6000_darwin64_record_arg (&valcum, type, true, true);
if (valret)
return false;
}
if (rs6000_discover_homogeneous_aggregate (TYPE_MODE (type), type,
NULL, NULL))
return false;
if (DEFAULT_ABI == ABI_ELFv2 && AGGREGATE_TYPE_P (type)
&& (unsigned HOST_WIDE_INT) int_size_in_bytes (type) <= 16)
return false;
if (AGGREGATE_TYPE_P (type)
&& (aix_struct_return
|| (unsigned HOST_WIDE_INT) int_size_in_bytes (type) > 8))
return true;
if (TARGET_32BIT && !TARGET_ALTIVEC_ABI
&& ALTIVEC_VECTOR_MODE (TYPE_MODE (type)))
return false;
if (TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) > (TARGET_ALTIVEC_ABI ? 16 : 8))
{
static bool warned_for_return_big_vectors = false;
if (!warned_for_return_big_vectors)
{
warning (OPT_Wpsabi, "GCC vector returned by reference: "
"non-standard ABI extension with no compatibility "
"guarantee");
warned_for_return_big_vectors = true;
}
return true;
}
if (DEFAULT_ABI == ABI_V4 && TARGET_IEEEQUAD
&& FLOAT128_IEEE_P (TYPE_MODE (type)))
return true;
return false;
}
static bool
rs6000_return_in_msb (const_tree valtype)
{
return (DEFAULT_ABI == ABI_ELFv2
&& BYTES_BIG_ENDIAN
&& AGGREGATE_TYPE_P (valtype)
&& (rs6000_function_arg_padding (TYPE_MODE (valtype), valtype)
== PAD_UPWARD));
}
#ifdef HAVE_AS_GNU_ATTRIBUTE
static bool
call_ABI_of_interest (tree fndecl)
{
if (rs6000_gnu_attr && symtab->state == EXPANSION)
{
struct cgraph_node *c_node;
if (fndecl == NULL_TREE)
return true;
if (DECL_EXTERNAL (fndecl))
return true;
c_node = cgraph_node::get (fndecl);
c_node = c_node->ultimate_alias_target ();
return !c_node->only_called_directly_p ();
}
return false;
}
#endif
void
init_cumulative_args (CUMULATIVE_ARGS *cum, tree fntype,
rtx libname ATTRIBUTE_UNUSED, int incoming,
int libcall, int n_named_args,
tree fndecl ATTRIBUTE_UNUSED,
machine_mode return_mode ATTRIBUTE_UNUSED)
{
static CUMULATIVE_ARGS zero_cumulative;
*cum = zero_cumulative;
cum->words = 0;
cum->fregno = FP_ARG_MIN_REG;
cum->vregno = ALTIVEC_ARG_MIN_REG;
cum->prototype = (fntype && prototype_p (fntype));
cum->call_cookie = ((DEFAULT_ABI == ABI_V4 && libcall)
? CALL_LIBCALL : CALL_NORMAL);
cum->sysv_gregno = GP_ARG_MIN_REG;
cum->stdarg = stdarg_p (fntype);
cum->libcall = libcall;
cum->nargs_prototype = 0;
if (incoming || cum->prototype)
cum->nargs_prototype = n_named_args;
if ((!fntype && rs6000_default_long_calls)
|| (fntype
&& lookup_attribute ("longcall", TYPE_ATTRIBUTES (fntype))
&& !lookup_attribute ("shortcall", TYPE_ATTRIBUTES (fntype))))
cum->call_cookie |= CALL_LONG;
if (TARGET_DEBUG_ARG)
{
fprintf (stderr, "\ninit_cumulative_args:");
if (fntype)
{
tree ret_type = TREE_TYPE (fntype);
fprintf (stderr, " ret code = %s,",
get_tree_code_name (TREE_CODE (ret_type)));
}
if (cum->call_cookie & CALL_LONG)
fprintf (stderr, " longcall,");
fprintf (stderr, " proto = %d, nargs = %d\n",
cum->prototype, cum->nargs_prototype);
}
#ifdef HAVE_AS_GNU_ATTRIBUTE
if (TARGET_ELF && (TARGET_64BIT || DEFAULT_ABI == ABI_V4))
{
cum->escapes = call_ABI_of_interest (fndecl);
if (cum->escapes)
{
tree return_type;
if (fntype)
{
return_type = TREE_TYPE (fntype);
return_mode = TYPE_MODE (return_type);
}
else
return_type = lang_hooks.types.type_for_mode (return_mode, 0);
if (return_type != NULL)
{
if (TREE_CODE (return_type) == RECORD_TYPE
&& TYPE_TRANSPARENT_AGGR (return_type))
{
return_type = TREE_TYPE (first_field (return_type));
return_mode = TYPE_MODE (return_type);
}
if (AGGREGATE_TYPE_P (return_type)
&& ((unsigned HOST_WIDE_INT) int_size_in_bytes (return_type)
<= 8))
rs6000_returns_struct = true;
}
if (SCALAR_FLOAT_MODE_P (return_mode))
{
rs6000_passes_float = true;
if ((HAVE_LD_PPC_GNU_ATTR_LONG_DOUBLE || TARGET_64BIT)
&& (FLOAT128_IBM_P (return_mode)
|| FLOAT128_IEEE_P (return_mode)
|| (return_type != NULL
&& (TYPE_MAIN_VARIANT (return_type)
== long_double_type_node))))
rs6000_passes_long_double = true;
if (FLOAT128_IEEE_P (return_mode))
rs6000_passes_ieee128 = true;
}
if (ALTIVEC_OR_VSX_VECTOR_MODE (return_mode)
|| PAIRED_VECTOR_MODE (return_mode))
rs6000_passes_vector = true;
}
}
#endif
if (fntype
&& !TARGET_ALTIVEC
&& TARGET_ALTIVEC_ABI
&& ALTIVEC_VECTOR_MODE (TYPE_MODE (TREE_TYPE (fntype))))
{
error ("cannot return value in vector register because"
" altivec instructions are disabled, use %qs"
" to enable them", "-maltivec");
}
}

static scalar_int_mode
rs6000_abi_word_mode (void)
{
return TARGET_32BIT ? SImode : DImode;
}
static char *
rs6000_offload_options (void)
{
if (TARGET_64BIT)
return xstrdup ("-foffload-abi=lp64");
else
return xstrdup ("-foffload-abi=ilp32");
}
static machine_mode
rs6000_promote_function_mode (const_tree type ATTRIBUTE_UNUSED,
machine_mode mode,
int *punsignedp ATTRIBUTE_UNUSED,
const_tree, int)
{
PROMOTE_MODE (mode, *punsignedp, type);
return mode;
}
static bool
rs6000_must_pass_in_stack (machine_mode mode, const_tree type)
{
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2 || TARGET_64BIT)
return must_pass_in_stack_var_size (mode, type);
else
return must_pass_in_stack_var_size_or_pad (mode, type);
}
static inline bool
is_complex_IBM_long_double (machine_mode mode)
{
return mode == ICmode || (mode == TCmode && FLOAT128_IBM_P (TCmode));
}
static bool
abi_v4_pass_in_fpr (machine_mode mode, bool named)
{
if (!TARGET_HARD_FLOAT)
return false;
if (TARGET_DOUBLE_FLOAT && mode == DFmode)
return true;
if (TARGET_SINGLE_FLOAT && mode == SFmode && named)
return true;
if (is_complex_IBM_long_double (mode))
return false;
if (FLOAT128_2REG_P (mode))
return true;
if (DECIMAL_FLOAT_MODE_P (mode))
return true;
return false;
}
static pad_direction
rs6000_function_arg_padding (machine_mode mode, const_tree type)
{
#ifndef AGGREGATE_PADDING_FIXED
#define AGGREGATE_PADDING_FIXED 0
#endif
#ifndef AGGREGATES_PAD_UPWARD_ALWAYS
#define AGGREGATES_PAD_UPWARD_ALWAYS 0
#endif
if (!AGGREGATE_PADDING_FIXED)
{
if (BYTES_BIG_ENDIAN)
{
HOST_WIDE_INT size = 0;
if (mode == BLKmode)
{
if (type && TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST)
size = int_size_in_bytes (type);
}
else
size = GET_MODE_SIZE (mode);
if (size == 1 || size == 2 || size == 4)
return PAD_DOWNWARD;
}
return PAD_UPWARD;
}
if (AGGREGATES_PAD_UPWARD_ALWAYS)
{
if (type != 0 && AGGREGATE_TYPE_P (type))
return PAD_UPWARD;
}
return default_function_arg_padding (mode, type);
}
static unsigned int
rs6000_function_arg_boundary (machine_mode mode, const_tree type)
{
machine_mode elt_mode;
int n_elts;
rs6000_discover_homogeneous_aggregate (mode, type, &elt_mode, &n_elts);
if (DEFAULT_ABI == ABI_V4
&& (GET_MODE_SIZE (mode) == 8
|| (TARGET_HARD_FLOAT
&& !is_complex_IBM_long_double (mode)
&& FLOAT128_2REG_P (mode))))
return 64;
else if (FLOAT128_VECTOR_P (mode))
return 128;
else if (PAIRED_VECTOR_MODE (mode)
|| (type && TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) >= 8
&& int_size_in_bytes (type) < 16))
return 64;
else if (ALTIVEC_OR_VSX_VECTOR_MODE (elt_mode)
|| (type && TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) >= 16))
return 128;
if (((DEFAULT_ABI == ABI_AIX && !rs6000_compat_align_parm)
|| DEFAULT_ABI == ABI_ELFv2)
&& type && TYPE_ALIGN (type) > 64)
{
bool aggregate_p = (AGGREGATE_TYPE_P (type)
&& !SCALAR_FLOAT_MODE_P (elt_mode));
if (aggregate_p != (mode == BLKmode))
{
static bool warned;
if (!warned && warn_psabi)
{
warned = true;
inform (input_location,
"the ABI of passing aggregates with %d-byte alignment"
" has changed in GCC 5",
(int) TYPE_ALIGN (type) / BITS_PER_UNIT);
}
}
if (aggregate_p)
return 128;
}
if (TARGET_MACHO && rs6000_darwin64_abi
&& mode == BLKmode
&& type && TYPE_ALIGN (type) > 64)
return 128;
return PARM_BOUNDARY;
}
static unsigned int
rs6000_parm_offset (void)
{
return (DEFAULT_ABI == ABI_V4 ? 2
: DEFAULT_ABI == ABI_ELFv2 ? 4
: 6);
}
static unsigned int
rs6000_parm_start (machine_mode mode, const_tree type,
unsigned int nwords)
{
unsigned int align;
align = rs6000_function_arg_boundary (mode, type) / PARM_BOUNDARY - 1;
return nwords + (-(rs6000_parm_offset () + nwords) & align);
}
static unsigned long
rs6000_arg_size (machine_mode mode, const_tree type)
{
unsigned long size;
if (mode != BLKmode)
size = GET_MODE_SIZE (mode);
else
size = int_size_in_bytes (type);
if (TARGET_32BIT)
return (size + 3) >> 2;
else
return (size + 7) >> 3;
}

static void
rs6000_darwin64_record_arg_advance_flush (CUMULATIVE_ARGS *cum,
HOST_WIDE_INT bitpos, int final)
{
unsigned int startbit, endbit;
int intregs, intoffset;
if (cum->floats_in_gpr == 1
&& (cum->intoffset % 64 == 0
|| (cum->intoffset == -1 && final)))
{
cum->words++;
cum->floats_in_gpr = 0;
}
if (cum->intoffset == -1)
return;
intoffset = cum->intoffset;
cum->intoffset = -1;
cum->floats_in_gpr = 0;
if (intoffset % BITS_PER_WORD != 0)
{
unsigned int bits = BITS_PER_WORD - intoffset % BITS_PER_WORD;
if (!int_mode_for_size (bits, 0).exists ())
{
intoffset = ROUND_DOWN (intoffset, BITS_PER_WORD);
}
}
startbit = ROUND_DOWN (intoffset, BITS_PER_WORD);
endbit = ROUND_UP (bitpos, BITS_PER_WORD);
intregs = (endbit - startbit) / BITS_PER_WORD;
cum->words += intregs;
if ((unsigned)cum->words < (endbit/BITS_PER_WORD))
{
int pad = (endbit/BITS_PER_WORD) - cum->words;
cum->words += pad;
}
}
static void
rs6000_darwin64_record_arg_advance_recurse (CUMULATIVE_ARGS *cum,
const_tree type,
HOST_WIDE_INT startbitpos)
{
tree f;
for (f = TYPE_FIELDS (type); f ; f = DECL_CHAIN (f))
if (TREE_CODE (f) == FIELD_DECL)
{
HOST_WIDE_INT bitpos = startbitpos;
tree ftype = TREE_TYPE (f);
machine_mode mode;
if (ftype == error_mark_node)
continue;
mode = TYPE_MODE (ftype);
if (DECL_SIZE (f) != 0
&& tree_fits_uhwi_p (bit_position (f)))
bitpos += int_bit_position (f);
if (TREE_CODE (ftype) == RECORD_TYPE)
rs6000_darwin64_record_arg_advance_recurse (cum, ftype, bitpos);
else if (USE_FP_FOR_ARG_P (cum, mode))
{
unsigned n_fpregs = (GET_MODE_SIZE (mode) + 7) >> 3;
rs6000_darwin64_record_arg_advance_flush (cum, bitpos, 0);
cum->fregno += n_fpregs;
if (mode == SFmode)
{
if (cum->floats_in_gpr == 1)
{
cum->words++;
cum->floats_in_gpr = 0;
}
else if (bitpos % 64 == 0)
{
cum->floats_in_gpr++;
}
else
{
}
}
else
cum->words += n_fpregs;
}
else if (USE_ALTIVEC_FOR_ARG_P (cum, mode, 1))
{
rs6000_darwin64_record_arg_advance_flush (cum, bitpos, 0);
cum->vregno++;
cum->words += 2;
}
else if (cum->intoffset == -1)
cum->intoffset = bitpos;
}
}
static int
rs6000_darwin64_struct_check_p (machine_mode mode, const_tree type)
{
return rs6000_darwin64_abi
&& ((mode == BLKmode 
&& TREE_CODE (type) == RECORD_TYPE 
&& int_size_in_bytes (type) > 0)
|| (type && TREE_CODE (type) == RECORD_TYPE 
&& int_size_in_bytes (type) == 8)) ? 1 : 0;
}
static void
rs6000_function_arg_advance_1 (CUMULATIVE_ARGS *cum, machine_mode mode,
const_tree type, bool named, int depth)
{
machine_mode elt_mode;
int n_elts;
rs6000_discover_homogeneous_aggregate (mode, type, &elt_mode, &n_elts);
if (depth == 0)
cum->nargs_prototype--;
#ifdef HAVE_AS_GNU_ATTRIBUTE
if (TARGET_ELF && (TARGET_64BIT || DEFAULT_ABI == ABI_V4)
&& cum->escapes)
{
if (SCALAR_FLOAT_MODE_P (mode))
{
rs6000_passes_float = true;
if ((HAVE_LD_PPC_GNU_ATTR_LONG_DOUBLE || TARGET_64BIT)
&& (FLOAT128_IBM_P (mode)
|| FLOAT128_IEEE_P (mode)
|| (type != NULL
&& TYPE_MAIN_VARIANT (type) == long_double_type_node)))
rs6000_passes_long_double = true;
if (FLOAT128_IEEE_P (mode))
rs6000_passes_ieee128 = true;
}
if ((named && ALTIVEC_OR_VSX_VECTOR_MODE (mode))
|| (PAIRED_VECTOR_MODE (mode)
&& !cum->stdarg
&& cum->sysv_gregno <= GP_ARG_MAX_REG))
rs6000_passes_vector = true;
}
#endif
if (TARGET_ALTIVEC_ABI
&& (ALTIVEC_OR_VSX_VECTOR_MODE (elt_mode)
|| (type && TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) == 16)))
{
bool stack = false;
if (USE_ALTIVEC_FOR_ARG_P (cum, elt_mode, named))
{
cum->vregno += n_elts;
if (!TARGET_ALTIVEC)
error ("cannot pass argument in vector register because"
" altivec instructions are disabled, use %qs"
" to enable them", "-maltivec");
if (((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& TARGET_64BIT)
|| (cum->stdarg && DEFAULT_ABI != ABI_V4))
stack = true;
}
else
stack = true;
if (stack)
{
int align;
if (TARGET_32BIT)
align = -(rs6000_parm_offset () + cum->words) & 3;
else
align = cum->words & 1;
cum->words += align + rs6000_arg_size (mode, type);
if (TARGET_DEBUG_ARG)
{
fprintf (stderr, "function_adv: words = %2d, align=%d, ",
cum->words, align);
fprintf (stderr, "nargs = %4d, proto = %d, mode = %4s\n",
cum->nargs_prototype, cum->prototype,
GET_MODE_NAME (mode));
}
}
}
else if (TARGET_MACHO && rs6000_darwin64_struct_check_p (mode, type))
{
int size = int_size_in_bytes (type);
if (TYPE_ALIGN (type) >= 2 * BITS_PER_WORD
&& (cum->words % 2) != 0)
cum->words++;
if (!named)
cum->words += (size + 7) / 8;
else
{
cum->intoffset = 0;
cum->floats_in_gpr = 0;
rs6000_darwin64_record_arg_advance_recurse (cum, type, 0);
rs6000_darwin64_record_arg_advance_flush (cum,
size * BITS_PER_UNIT, 1);
}
if (TARGET_DEBUG_ARG)
{
fprintf (stderr, "function_adv: words = %2d, align=%d, size=%d",
cum->words, TYPE_ALIGN (type), size);
fprintf (stderr, 
"nargs = %4d, proto = %d, mode = %4s (darwin64 abi)\n",
cum->nargs_prototype, cum->prototype,
GET_MODE_NAME (mode));
}
}
else if (DEFAULT_ABI == ABI_V4)
{
if (abi_v4_pass_in_fpr (mode, named))
{
if (mode == TDmode && (cum->fregno % 2) == 1)
cum->fregno++;
if (cum->fregno + (FLOAT128_2REG_P (mode) ? 1 : 0)
<= FP_ARG_V4_MAX_REG)
cum->fregno += (GET_MODE_SIZE (mode) + 7) >> 3;
else
{
cum->fregno = FP_ARG_V4_MAX_REG + 1;
if (mode == DFmode || FLOAT128_IBM_P (mode)
|| mode == DDmode || mode == TDmode)
cum->words += cum->words & 1;
cum->words += rs6000_arg_size (mode, type);
}
}
else
{
int n_words = rs6000_arg_size (mode, type);
int gregno = cum->sysv_gregno;
if (n_words == 2)
gregno += (1 - gregno) & 1;
if (gregno + n_words - 1 > GP_ARG_MAX_REG)
{
if (n_words == 2)
cum->words += cum->words & 1;
cum->words += n_words;
}
cum->sysv_gregno = gregno + n_words;
}
if (TARGET_DEBUG_ARG)
{
fprintf (stderr, "function_adv: words = %2d, fregno = %2d, ",
cum->words, cum->fregno);
fprintf (stderr, "gregno = %2d, nargs = %4d, proto = %d, ",
cum->sysv_gregno, cum->nargs_prototype, cum->prototype);
fprintf (stderr, "mode = %4s, named = %d\n",
GET_MODE_NAME (mode), named);
}
}
else
{
int n_words = rs6000_arg_size (mode, type);
int start_words = cum->words;
int align_words = rs6000_parm_start (mode, type, start_words);
cum->words = align_words + n_words;
if (SCALAR_FLOAT_MODE_P (elt_mode) && TARGET_HARD_FLOAT)
{
if (elt_mode == TDmode && (cum->fregno % 2) == 1)
cum->fregno++;
cum->fregno += n_elts * ((GET_MODE_SIZE (elt_mode) + 7) >> 3);
}
if (TARGET_DEBUG_ARG)
{
fprintf (stderr, "function_adv: words = %2d, fregno = %2d, ",
cum->words, cum->fregno);
fprintf (stderr, "nargs = %4d, proto = %d, mode = %4s, ",
cum->nargs_prototype, cum->prototype, GET_MODE_NAME (mode));
fprintf (stderr, "named = %d, align = %d, depth = %d\n",
named, align_words - start_words, depth);
}
}
}
static void
rs6000_function_arg_advance (cumulative_args_t cum, machine_mode mode,
const_tree type, bool named)
{
rs6000_function_arg_advance_1 (get_cumulative_args (cum), mode, type, named,
0);
}
static void
rs6000_darwin64_record_arg_flush (CUMULATIVE_ARGS *cum,
HOST_WIDE_INT bitpos, rtx rvec[], int *k)
{
machine_mode mode;
unsigned int regno;
unsigned int startbit, endbit;
int this_regno, intregs, intoffset;
rtx reg;
if (cum->intoffset == -1)
return;
intoffset = cum->intoffset;
cum->intoffset = -1;
if (intoffset % BITS_PER_WORD != 0)
{
unsigned int bits = BITS_PER_WORD - intoffset % BITS_PER_WORD;
if (!int_mode_for_size (bits, 0).exists (&mode))
{
intoffset = ROUND_DOWN (intoffset, BITS_PER_WORD);
mode = word_mode;
}
}
else
mode = word_mode;
startbit = ROUND_DOWN (intoffset, BITS_PER_WORD);
endbit = ROUND_UP (bitpos, BITS_PER_WORD);
intregs = (endbit - startbit) / BITS_PER_WORD;
this_regno = cum->words + intoffset / BITS_PER_WORD;
if (intregs > 0 && intregs > GP_ARG_NUM_REG - this_regno)
cum->use_stack = 1;
intregs = MIN (intregs, GP_ARG_NUM_REG - this_regno);
if (intregs <= 0)
return;
intoffset /= BITS_PER_UNIT;
do
{
regno = GP_ARG_MIN_REG + this_regno;
reg = gen_rtx_REG (mode, regno);
rvec[(*k)++] =
gen_rtx_EXPR_LIST (VOIDmode, reg, GEN_INT (intoffset));
this_regno += 1;
intoffset = (intoffset | (UNITS_PER_WORD-1)) + 1;
mode = word_mode;
intregs -= 1;
}
while (intregs > 0);
}
static void
rs6000_darwin64_record_arg_recurse (CUMULATIVE_ARGS *cum, const_tree type,
HOST_WIDE_INT startbitpos, rtx rvec[],
int *k)
{
tree f;
for (f = TYPE_FIELDS (type); f ; f = DECL_CHAIN (f))
if (TREE_CODE (f) == FIELD_DECL)
{
HOST_WIDE_INT bitpos = startbitpos;
tree ftype = TREE_TYPE (f);
machine_mode mode;
if (ftype == error_mark_node)
continue;
mode = TYPE_MODE (ftype);
if (DECL_SIZE (f) != 0
&& tree_fits_uhwi_p (bit_position (f)))
bitpos += int_bit_position (f);
if (TREE_CODE (ftype) == RECORD_TYPE)
rs6000_darwin64_record_arg_recurse (cum, ftype, bitpos, rvec, k);
else if (cum->named && USE_FP_FOR_ARG_P (cum, mode))
{
unsigned n_fpreg = (GET_MODE_SIZE (mode) + 7) >> 3;
#if 0
switch (mode)
{
case E_SCmode: mode = SFmode; break;
case E_DCmode: mode = DFmode; break;
case E_TCmode: mode = TFmode; break;
default: break;
}
#endif
rs6000_darwin64_record_arg_flush (cum, bitpos, rvec, k);
if (cum->fregno + n_fpreg > FP_ARG_MAX_REG + 1)
{
gcc_assert (cum->fregno == FP_ARG_MAX_REG
&& (mode == TFmode || mode == TDmode));
mode = DECIMAL_FLOAT_MODE_P (mode) ? DDmode : DFmode;
cum->use_stack=1;
}
rvec[(*k)++]
= gen_rtx_EXPR_LIST (VOIDmode,
gen_rtx_REG (mode, cum->fregno++),
GEN_INT (bitpos / BITS_PER_UNIT));
if (FLOAT128_2REG_P (mode))
cum->fregno++;
}
else if (cum->named && USE_ALTIVEC_FOR_ARG_P (cum, mode, 1))
{
rs6000_darwin64_record_arg_flush (cum, bitpos, rvec, k);
rvec[(*k)++]
= gen_rtx_EXPR_LIST (VOIDmode,
gen_rtx_REG (mode, cum->vregno++),
GEN_INT (bitpos / BITS_PER_UNIT));
}
else if (cum->intoffset == -1)
cum->intoffset = bitpos;
}
}
static rtx
rs6000_darwin64_record_arg (CUMULATIVE_ARGS *orig_cum, const_tree type,
bool named, bool retval)
{
rtx rvec[FIRST_PSEUDO_REGISTER];
int k = 1, kbase = 1;
HOST_WIDE_INT typesize = int_size_in_bytes (type);
CUMULATIVE_ARGS copy_cum = *orig_cum;
CUMULATIVE_ARGS *cum = &copy_cum;
if (!retval && TYPE_ALIGN (type) >= 2 * BITS_PER_WORD
&& (cum->words % 2) != 0)
cum->words++;
cum->intoffset = 0;
cum->use_stack = 0;
cum->named = named;
rs6000_darwin64_record_arg_recurse (cum, type,  0, rvec, &k);
rs6000_darwin64_record_arg_flush (cum, typesize * BITS_PER_UNIT, rvec, &k);
if (cum->use_stack)
{
if (retval)
return NULL_RTX;    
kbase = 0;
rvec[0] = gen_rtx_EXPR_LIST (VOIDmode, NULL_RTX, const0_rtx);
}
if (k > 1 || cum->use_stack)
return gen_rtx_PARALLEL (BLKmode, gen_rtvec_v (k - kbase, &rvec[kbase]));
else
return NULL_RTX;
}
static rtx
rs6000_mixed_function_arg (machine_mode mode, const_tree type,
int align_words)
{
int n_units;
int i, k;
rtx rvec[GP_ARG_NUM_REG + 1];
if (align_words >= GP_ARG_NUM_REG)
return NULL_RTX;
n_units = rs6000_arg_size (mode, type);
if (n_units == 0
|| (n_units == 1 && mode != BLKmode))
return gen_rtx_REG (mode, GP_ARG_MIN_REG + align_words);
k = 0;
if (align_words + n_units > GP_ARG_NUM_REG)
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, NULL_RTX, const0_rtx);
i = 0;
do
{
rtx r = gen_rtx_REG (SImode, GP_ARG_MIN_REG + align_words);
rtx off = GEN_INT (i++ * 4);
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, r, off);
}
while (++align_words < GP_ARG_NUM_REG && --n_units != 0);
return gen_rtx_PARALLEL (mode, gen_rtvec_v (k, rvec));
}
static int
rs6000_psave_function_arg (machine_mode mode, const_tree type,
int align_words, rtx *rvec)
{
int k = 0;
if (align_words < GP_ARG_NUM_REG)
{
int n_words = rs6000_arg_size (mode, type);
if (align_words + n_words > GP_ARG_NUM_REG
|| mode == BLKmode
|| (TARGET_32BIT && TARGET_POWERPC64))
{
machine_mode rmode = TARGET_32BIT ? SImode : DImode;
int i = 0;
if (align_words + n_words > GP_ARG_NUM_REG)
{
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, NULL_RTX, const0_rtx);
}
do
{
rtx r = gen_rtx_REG (rmode, GP_ARG_MIN_REG + align_words);
rtx off = GEN_INT (i++ * GET_MODE_SIZE (rmode));
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, r, off);
}
while (++align_words < GP_ARG_NUM_REG && --n_words != 0);
}
else
{
rtx r = gen_rtx_REG (mode, GP_ARG_MIN_REG + align_words);
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, r, const0_rtx);
}
}
else
{
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, NULL_RTX, const0_rtx);
}
return k;
}
static rtx
rs6000_finish_function_arg (machine_mode mode, rtx *rvec, int k)
{
gcc_assert (k >= 1);
if (k == 1)
{
if (XEXP (rvec[0], 0) == NULL_RTX)
return NULL_RTX;
if (GET_MODE (XEXP (rvec[0], 0)) == mode)
return XEXP (rvec[0], 0);
}
return gen_rtx_PARALLEL (mode, gen_rtvec_v (k, rvec));
}
static rtx
rs6000_function_arg (cumulative_args_t cum_v, machine_mode mode,
const_tree type, bool named)
{
CUMULATIVE_ARGS *cum = get_cumulative_args (cum_v);
enum rs6000_abi abi = DEFAULT_ABI;
machine_mode elt_mode;
int n_elts;
if (mode == VOIDmode)
{
if (abi == ABI_V4
&& (cum->call_cookie & CALL_LIBCALL) == 0
&& (cum->stdarg
|| (cum->nargs_prototype < 0
&& (cum->prototype || TARGET_NO_PROTOTYPE)))
&& TARGET_HARD_FLOAT)
return GEN_INT (cum->call_cookie
| ((cum->fregno == FP_ARG_MIN_REG)
? CALL_V4_SET_FP_ARGS
: CALL_V4_CLEAR_FP_ARGS));
return GEN_INT (cum->call_cookie & ~CALL_LIBCALL);
}
rs6000_discover_homogeneous_aggregate (mode, type, &elt_mode, &n_elts);
if (TARGET_MACHO && rs6000_darwin64_struct_check_p (mode, type))
{
rtx rslt = rs6000_darwin64_record_arg (cum, type, named, false);
if (rslt != NULL_RTX)
return rslt;
}
if (USE_ALTIVEC_FOR_ARG_P (cum, elt_mode, named))
{
rtx rvec[GP_ARG_NUM_REG + AGGR_ARG_NUM_REG + 1];
rtx r, off;
int i, k = 0;
if (TARGET_64BIT && !cum->prototype
&& (!cum->libcall || !FLOAT128_VECTOR_P (elt_mode)))
{
int align_words = ROUND_UP (cum->words, 2);
k = rs6000_psave_function_arg (mode, type, align_words, rvec);
}
for (i = 0; i < n_elts && cum->vregno + i <= ALTIVEC_ARG_MAX_REG; i++)
{
r = gen_rtx_REG (elt_mode, cum->vregno + i);
off = GEN_INT (i * GET_MODE_SIZE (elt_mode));
rvec[k++] =  gen_rtx_EXPR_LIST (VOIDmode, r, off);
}
return rs6000_finish_function_arg (mode, rvec, k);
}
else if (TARGET_ALTIVEC_ABI
&& (ALTIVEC_OR_VSX_VECTOR_MODE (mode)
|| (type && TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) == 16)))
{
if (named || abi == ABI_V4)
return NULL_RTX;
else
{
int align, align_words, n_words;
machine_mode part_mode;
if (TARGET_32BIT)
align = -(rs6000_parm_offset () + cum->words) & 3;
else
align = cum->words & 1;
align_words = cum->words + align;
if (align_words >= GP_ARG_NUM_REG)
return NULL_RTX;
if (TARGET_32BIT && TARGET_POWERPC64)
return rs6000_mixed_function_arg (mode, type, align_words);
part_mode = mode;
n_words = rs6000_arg_size (mode, type);
if (align_words + n_words > GP_ARG_NUM_REG)
part_mode = DImode;
return gen_rtx_REG (part_mode, GP_ARG_MIN_REG + align_words);
}
}
else if (abi == ABI_V4)
{
if (abi_v4_pass_in_fpr (mode, named))
{
if (mode == TDmode && (cum->fregno % 2) == 1)
cum->fregno++;
if (cum->fregno + (FLOAT128_2REG_P (mode) ? 1 : 0)
<= FP_ARG_V4_MAX_REG)
return gen_rtx_REG (mode, cum->fregno);
else
return NULL_RTX;
}
else
{
int n_words = rs6000_arg_size (mode, type);
int gregno = cum->sysv_gregno;
if (n_words == 2)
gregno += (1 - gregno) & 1;
if (gregno + n_words - 1 > GP_ARG_MAX_REG)
return NULL_RTX;
if (TARGET_32BIT && TARGET_POWERPC64)
return rs6000_mixed_function_arg (mode, type,
gregno - GP_ARG_MIN_REG);
return gen_rtx_REG (mode, gregno);
}
}
else
{
int align_words = rs6000_parm_start (mode, type, cum->words);
if (elt_mode == TDmode && (cum->fregno % 2) == 1)
cum->fregno++;
if (USE_FP_FOR_ARG_P (cum, elt_mode))
{
rtx rvec[GP_ARG_NUM_REG + AGGR_ARG_NUM_REG + 1];
rtx r, off;
int i, k = 0;
unsigned long n_fpreg = (GET_MODE_SIZE (elt_mode) + 7) >> 3;
int fpr_words;
if (type && (cum->nargs_prototype <= 0
|| ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& TARGET_XL_COMPAT
&& align_words >= GP_ARG_NUM_REG)))
k = rs6000_psave_function_arg (mode, type, align_words, rvec);
for (i = 0; i < n_elts
&& cum->fregno + i * n_fpreg <= FP_ARG_MAX_REG; i++)
{
machine_mode fmode = elt_mode;
if (cum->fregno + (i + 1) * n_fpreg > FP_ARG_MAX_REG + 1)
{
gcc_assert (FLOAT128_2REG_P (fmode));
fmode = DECIMAL_FLOAT_MODE_P (fmode) ? DDmode : DFmode;
}
r = gen_rtx_REG (fmode, cum->fregno + i * n_fpreg);
off = GEN_INT (i * GET_MODE_SIZE (elt_mode));
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, r, off);
}
fpr_words = (i * GET_MODE_SIZE (elt_mode)) / (TARGET_32BIT ? 4 : 8);
if (i < n_elts && align_words + fpr_words < GP_ARG_NUM_REG
&& cum->nargs_prototype > 0)
{
static bool warned;
machine_mode rmode = TARGET_32BIT ? SImode : DImode;
int n_words = rs6000_arg_size (mode, type);
align_words += fpr_words;
n_words -= fpr_words;
do
{
r = gen_rtx_REG (rmode, GP_ARG_MIN_REG + align_words);
off = GEN_INT (fpr_words++ * GET_MODE_SIZE (rmode));
rvec[k++] = gen_rtx_EXPR_LIST (VOIDmode, r, off);
}
while (++align_words < GP_ARG_NUM_REG && --n_words != 0);
if (!warned && warn_psabi)
{
warned = true;
inform (input_location,
"the ABI of passing homogeneous float aggregates"
" has changed in GCC 5");
}
}
return rs6000_finish_function_arg (mode, rvec, k);
}
else if (align_words < GP_ARG_NUM_REG)
{
if (TARGET_32BIT && TARGET_POWERPC64)
return rs6000_mixed_function_arg (mode, type, align_words);
return gen_rtx_REG (mode, GP_ARG_MIN_REG + align_words);
}
else
return NULL_RTX;
}
}

static int
rs6000_arg_partial_bytes (cumulative_args_t cum_v, machine_mode mode,
tree type, bool named)
{
CUMULATIVE_ARGS *cum = get_cumulative_args (cum_v);
bool passed_in_gprs = true;
int ret = 0;
int align_words;
machine_mode elt_mode;
int n_elts;
rs6000_discover_homogeneous_aggregate (mode, type, &elt_mode, &n_elts);
if (DEFAULT_ABI == ABI_V4)
return 0;
if (USE_ALTIVEC_FOR_ARG_P (cum, elt_mode, named))
{
if (TARGET_64BIT && !cum->prototype
&& (!cum->libcall || !FLOAT128_VECTOR_P (elt_mode)))
return 0;
passed_in_gprs = false;
if (cum->vregno + n_elts > ALTIVEC_ARG_MAX_REG + 1)
ret = (ALTIVEC_ARG_MAX_REG + 1 - cum->vregno) * 16;
}
if (TARGET_MACHO && rs6000_darwin64_struct_check_p (mode, type))
return 0;
align_words = rs6000_parm_start (mode, type, cum->words);
if (USE_FP_FOR_ARG_P (cum, elt_mode))
{
unsigned long n_fpreg = (GET_MODE_SIZE (elt_mode) + 7) >> 3;
if (type
&& (cum->nargs_prototype <= 0
|| ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& TARGET_XL_COMPAT
&& align_words >= GP_ARG_NUM_REG)))
return 0;
passed_in_gprs = false;
if (cum->fregno + n_elts * n_fpreg > FP_ARG_MAX_REG + 1)
{
int fpr = ((FP_ARG_MAX_REG + 1 - cum->fregno)
* MIN (8, GET_MODE_SIZE (elt_mode)));
int fpr_words = fpr / (TARGET_32BIT ? 4 : 8);
if (align_words + fpr_words < GP_ARG_NUM_REG)
passed_in_gprs = true;
else
ret = fpr;
}
}
if (passed_in_gprs
&& align_words < GP_ARG_NUM_REG
&& GP_ARG_NUM_REG < align_words + rs6000_arg_size (mode, type))
ret = (GP_ARG_NUM_REG - align_words) * (TARGET_32BIT ? 4 : 8);
if (ret != 0 && TARGET_DEBUG_ARG)
fprintf (stderr, "rs6000_arg_partial_bytes: %d\n", ret);
return ret;
}

static bool
rs6000_pass_by_reference (cumulative_args_t cum ATTRIBUTE_UNUSED,
machine_mode mode, const_tree type,
bool named ATTRIBUTE_UNUSED)
{
if (!type)
return 0;
if (DEFAULT_ABI == ABI_V4 && TARGET_IEEEQUAD
&& FLOAT128_IEEE_P (TYPE_MODE (type)))
{
if (TARGET_DEBUG_ARG)
fprintf (stderr, "function_arg_pass_by_reference: V4 IEEE 128-bit\n");
return 1;
}
if (DEFAULT_ABI == ABI_V4 && AGGREGATE_TYPE_P (type))
{
if (TARGET_DEBUG_ARG)
fprintf (stderr, "function_arg_pass_by_reference: V4 aggregate\n");
return 1;
}
if (int_size_in_bytes (type) < 0)
{
if (TARGET_DEBUG_ARG)
fprintf (stderr, "function_arg_pass_by_reference: variable size\n");
return 1;
}
if (TARGET_32BIT && !TARGET_ALTIVEC_ABI && ALTIVEC_VECTOR_MODE (mode))
{
if (TARGET_DEBUG_ARG)
fprintf (stderr, "function_arg_pass_by_reference: AltiVec\n");
return 1;
}
if (TREE_CODE (type) == VECTOR_TYPE
&& int_size_in_bytes (type) > (TARGET_ALTIVEC_ABI ? 16 : 8))
{
static bool warned_for_pass_big_vectors = false;
if (TARGET_DEBUG_ARG)
fprintf (stderr, "function_arg_pass_by_reference: synthetic vector\n");
if (!warned_for_pass_big_vectors)
{
warning (OPT_Wpsabi, "GCC vector passed by reference: "
"non-standard ABI extension with no compatibility "
"guarantee");
warned_for_pass_big_vectors = true;
}
return 1;
}
return 0;
}
static bool
rs6000_parm_needs_stack (cumulative_args_t args_so_far, tree type)
{
machine_mode mode;
int unsignedp;
rtx entry_parm;
if (type == NULL || type == error_mark_node)
return true;
if (TYPE_MODE (type) == VOIDmode)
return false;
if (TREE_CODE (type) == COMPLEX_TYPE)
return (rs6000_parm_needs_stack (args_so_far, TREE_TYPE (type))
|| rs6000_parm_needs_stack (args_so_far, TREE_TYPE (type)));
if ((TREE_CODE (type) == UNION_TYPE || TREE_CODE (type) == RECORD_TYPE)
&& TYPE_TRANSPARENT_AGGR (type))
type = TREE_TYPE (first_field (type));
if (pass_by_reference (get_cumulative_args (args_so_far),
TYPE_MODE (type), type, true))
type = build_pointer_type (type);
unsignedp = TYPE_UNSIGNED (type);
mode = promote_mode (type, TYPE_MODE (type), &unsignedp);
if (rs6000_must_pass_in_stack (mode, type))
return true;
entry_parm = rs6000_function_arg (args_so_far, mode, type, true);
if (entry_parm == NULL)
return true;
if (GET_CODE (entry_parm) == PARALLEL
&& XEXP (XVECEXP (entry_parm, 0, 0), 0) == NULL_RTX)
return true;
if (rs6000_arg_partial_bytes (args_so_far, mode, type, true) != 0)
return true;
rs6000_function_arg_advance (args_so_far, mode, type, true);
return false;
}
static bool
rs6000_function_parms_need_stack (tree fun, bool incoming)
{
tree fntype, result;
CUMULATIVE_ARGS args_so_far_v;
cumulative_args_t args_so_far;
if (!fun)
return false;
fntype = fun;
if (!TYPE_P (fun))
fntype = TREE_TYPE (fun);
if ((!incoming && !prototype_p (fntype)) || stdarg_p (fntype))
return true;
INIT_CUMULATIVE_INCOMING_ARGS (args_so_far_v, fntype, NULL_RTX);
args_so_far = pack_cumulative_args (&args_so_far_v);
if (incoming)
{
gcc_assert (DECL_P (fun));
result = DECL_RESULT (fun);
}
else
result = TREE_TYPE (fntype);
if (result && aggregate_value_p (result, fntype))
{
if (!TYPE_P (result))
result = TREE_TYPE (result);
result = build_pointer_type (result);
rs6000_parm_needs_stack (args_so_far, result);
}
if (incoming)
{
tree parm;
for (parm = DECL_ARGUMENTS (fun);
parm && parm != void_list_node;
parm = TREE_CHAIN (parm))
if (rs6000_parm_needs_stack (args_so_far, TREE_TYPE (parm)))
return true;
}
else
{
function_args_iterator args_iter;
tree arg_type;
FOREACH_FUNCTION_ARGS (fntype, arg_type, args_iter)
if (rs6000_parm_needs_stack (args_so_far, arg_type))
return true;
}
return false;
}
int
rs6000_reg_parm_stack_space (tree fun, bool incoming)
{
int reg_parm_stack_space;
switch (DEFAULT_ABI)
{
default:
reg_parm_stack_space = 0;
break;
case ABI_AIX:
case ABI_DARWIN:
reg_parm_stack_space = TARGET_64BIT ? 64 : 32;
break;
case ABI_ELFv2:
if (rs6000_function_parms_need_stack (fun, incoming))
reg_parm_stack_space = TARGET_64BIT ? 64 : 32;
else
reg_parm_stack_space = 0;
break;
}
return reg_parm_stack_space;
}
static void
rs6000_move_block_from_reg (int regno, rtx x, int nregs)
{
int i;
machine_mode reg_mode = TARGET_32BIT ? SImode : DImode;
if (nregs == 0)
return;
for (i = 0; i < nregs; i++)
{
rtx tem = adjust_address_nv (x, reg_mode, i * GET_MODE_SIZE (reg_mode));
if (reload_completed)
{
if (! strict_memory_address_p (reg_mode, XEXP (tem, 0)))
tem = NULL_RTX;
else
tem = simplify_gen_subreg (reg_mode, x, BLKmode,
i * GET_MODE_SIZE (reg_mode));
}
else
tem = replace_equiv_address (tem, XEXP (tem, 0));
gcc_assert (tem);
emit_move_insn (tem, gen_rtx_REG (reg_mode, regno + i));
}
}

static void
setup_incoming_varargs (cumulative_args_t cum, machine_mode mode,
tree type, int *pretend_size ATTRIBUTE_UNUSED,
int no_rtl)
{
CUMULATIVE_ARGS next_cum;
int reg_size = TARGET_32BIT ? 4 : 8;
rtx save_area = NULL_RTX, mem;
int first_reg_offset;
alias_set_type set;
next_cum = *get_cumulative_args (cum);
rs6000_function_arg_advance_1 (&next_cum, mode, type, true, 0);
if (DEFAULT_ABI == ABI_V4)
{
first_reg_offset = next_cum.sysv_gregno - GP_ARG_MIN_REG;
if (! no_rtl)
{
int gpr_reg_num = 0, gpr_size = 0, fpr_size = 0;
HOST_WIDE_INT offset = 0;
if (cfun->va_list_gpr_size && first_reg_offset < GP_ARG_NUM_REG)
gpr_reg_num = GP_ARG_NUM_REG - first_reg_offset;
if (TARGET_HARD_FLOAT
&& next_cum.fregno <= FP_ARG_V4_MAX_REG
&& cfun->va_list_fpr_size)
{
if (gpr_reg_num)
fpr_size = (next_cum.fregno - FP_ARG_MIN_REG)
* UNITS_PER_FP_WORD;
if (cfun->va_list_fpr_size
< FP_ARG_V4_MAX_REG + 1 - next_cum.fregno)
fpr_size += cfun->va_list_fpr_size * UNITS_PER_FP_WORD;
else
fpr_size += (FP_ARG_V4_MAX_REG + 1 - next_cum.fregno)
* UNITS_PER_FP_WORD;
}
if (gpr_reg_num)
{
offset = -((first_reg_offset * reg_size) & ~7);
if (!fpr_size && gpr_reg_num > cfun->va_list_gpr_size)
{
gpr_reg_num = cfun->va_list_gpr_size;
if (reg_size == 4 && (first_reg_offset & 1))
gpr_reg_num++;
}
gpr_size = (gpr_reg_num * reg_size + 7) & ~7;
}
else if (fpr_size)
offset = - (int) (next_cum.fregno - FP_ARG_MIN_REG)
* UNITS_PER_FP_WORD
- (int) (GP_ARG_NUM_REG * reg_size);
if (gpr_size + fpr_size)
{
rtx reg_save_area
= assign_stack_local (BLKmode, gpr_size + fpr_size, 64);
gcc_assert (GET_CODE (reg_save_area) == MEM);
reg_save_area = XEXP (reg_save_area, 0);
if (GET_CODE (reg_save_area) == PLUS)
{
gcc_assert (XEXP (reg_save_area, 0)
== virtual_stack_vars_rtx);
gcc_assert (GET_CODE (XEXP (reg_save_area, 1)) == CONST_INT);
offset += INTVAL (XEXP (reg_save_area, 1));
}
else
gcc_assert (reg_save_area == virtual_stack_vars_rtx);
}
cfun->machine->varargs_save_offset = offset;
save_area = plus_constant (Pmode, virtual_stack_vars_rtx, offset);
}
}
else
{
first_reg_offset = next_cum.words;
save_area = crtl->args.internal_arg_pointer;
if (targetm.calls.must_pass_in_stack (mode, type))
first_reg_offset += rs6000_arg_size (TYPE_MODE (type), type);
}
set = get_varargs_alias_set ();
if (! no_rtl && first_reg_offset < GP_ARG_NUM_REG
&& cfun->va_list_gpr_size)
{
int n_gpr, nregs = GP_ARG_NUM_REG - first_reg_offset;
if (va_list_gpr_counter_field)
n_gpr = cfun->va_list_gpr_size;
else
n_gpr = (cfun->va_list_gpr_size + reg_size - 1) / reg_size;
if (nregs > n_gpr)
nregs = n_gpr;
mem = gen_rtx_MEM (BLKmode,
plus_constant (Pmode, save_area,
first_reg_offset * reg_size));
MEM_NOTRAP_P (mem) = 1;
set_mem_alias_set (mem, set);
set_mem_align (mem, BITS_PER_WORD);
rs6000_move_block_from_reg (GP_ARG_MIN_REG + first_reg_offset, mem,
nregs);
}
if (DEFAULT_ABI == ABI_V4
&& TARGET_HARD_FLOAT
&& ! no_rtl
&& next_cum.fregno <= FP_ARG_V4_MAX_REG
&& cfun->va_list_fpr_size)
{
int fregno = next_cum.fregno, nregs;
rtx cr1 = gen_rtx_REG (CCmode, CR1_REGNO);
rtx lab = gen_label_rtx ();
int off = (GP_ARG_NUM_REG * reg_size) + ((fregno - FP_ARG_MIN_REG)
* UNITS_PER_FP_WORD);
emit_jump_insn
(gen_rtx_SET (pc_rtx,
gen_rtx_IF_THEN_ELSE (VOIDmode,
gen_rtx_NE (VOIDmode, cr1,
const0_rtx),
gen_rtx_LABEL_REF (VOIDmode, lab),
pc_rtx)));
for (nregs = 0;
fregno <= FP_ARG_V4_MAX_REG && nregs < cfun->va_list_fpr_size;
fregno++, off += UNITS_PER_FP_WORD, nregs++)
{
mem = gen_rtx_MEM ((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode, 
plus_constant (Pmode, save_area, off));
MEM_NOTRAP_P (mem) = 1;
set_mem_alias_set (mem, set);
set_mem_align (mem, GET_MODE_ALIGNMENT (
(TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode));
emit_move_insn (mem, gen_rtx_REG (
(TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode, fregno));
}
emit_label (lab);
}
}
static tree
rs6000_build_builtin_va_list (void)
{
tree f_gpr, f_fpr, f_res, f_ovf, f_sav, record, type_decl;
if (DEFAULT_ABI != ABI_V4)
return build_pointer_type (char_type_node);
record = (*lang_hooks.types.make_type) (RECORD_TYPE);
type_decl = build_decl (BUILTINS_LOCATION, TYPE_DECL,
get_identifier ("__va_list_tag"), record);
f_gpr = build_decl (BUILTINS_LOCATION, FIELD_DECL, get_identifier ("gpr"),
unsigned_char_type_node);
f_fpr = build_decl (BUILTINS_LOCATION, FIELD_DECL, get_identifier ("fpr"),
unsigned_char_type_node);
f_res = build_decl (BUILTINS_LOCATION, FIELD_DECL,
get_identifier ("reserved"), short_unsigned_type_node);
f_ovf = build_decl (BUILTINS_LOCATION, FIELD_DECL,
get_identifier ("overflow_arg_area"),
ptr_type_node);
f_sav = build_decl (BUILTINS_LOCATION, FIELD_DECL,
get_identifier ("reg_save_area"),
ptr_type_node);
va_list_gpr_counter_field = f_gpr;
va_list_fpr_counter_field = f_fpr;
DECL_FIELD_CONTEXT (f_gpr) = record;
DECL_FIELD_CONTEXT (f_fpr) = record;
DECL_FIELD_CONTEXT (f_res) = record;
DECL_FIELD_CONTEXT (f_ovf) = record;
DECL_FIELD_CONTEXT (f_sav) = record;
TYPE_STUB_DECL (record) = type_decl;
TYPE_NAME (record) = type_decl;
TYPE_FIELDS (record) = f_gpr;
DECL_CHAIN (f_gpr) = f_fpr;
DECL_CHAIN (f_fpr) = f_res;
DECL_CHAIN (f_res) = f_ovf;
DECL_CHAIN (f_ovf) = f_sav;
layout_type (record);
return build_array_type (record, build_index_type (size_zero_node));
}
static void
rs6000_va_start (tree valist, rtx nextarg)
{
HOST_WIDE_INT words, n_gpr, n_fpr;
tree f_gpr, f_fpr, f_res, f_ovf, f_sav;
tree gpr, fpr, ovf, sav, t;
if (DEFAULT_ABI != ABI_V4)
{
std_expand_builtin_va_start (valist, nextarg);
return;
}
f_gpr = TYPE_FIELDS (TREE_TYPE (va_list_type_node));
f_fpr = DECL_CHAIN (f_gpr);
f_res = DECL_CHAIN (f_fpr);
f_ovf = DECL_CHAIN (f_res);
f_sav = DECL_CHAIN (f_ovf);
valist = build_simple_mem_ref (valist);
gpr = build3 (COMPONENT_REF, TREE_TYPE (f_gpr), valist, f_gpr, NULL_TREE);
fpr = build3 (COMPONENT_REF, TREE_TYPE (f_fpr), unshare_expr (valist),
f_fpr, NULL_TREE);
ovf = build3 (COMPONENT_REF, TREE_TYPE (f_ovf), unshare_expr (valist),
f_ovf, NULL_TREE);
sav = build3 (COMPONENT_REF, TREE_TYPE (f_sav), unshare_expr (valist),
f_sav, NULL_TREE);
words = crtl->args.info.words;
n_gpr = MIN (crtl->args.info.sysv_gregno - GP_ARG_MIN_REG,
GP_ARG_NUM_REG);
n_fpr = MIN (crtl->args.info.fregno - FP_ARG_MIN_REG,
FP_ARG_NUM_REG);
if (TARGET_DEBUG_ARG)
fprintf (stderr, "va_start: words = " HOST_WIDE_INT_PRINT_DEC", n_gpr = "
HOST_WIDE_INT_PRINT_DEC", n_fpr = " HOST_WIDE_INT_PRINT_DEC"\n",
words, n_gpr, n_fpr);
if (cfun->va_list_gpr_size)
{
t = build2 (MODIFY_EXPR, TREE_TYPE (gpr), gpr,
build_int_cst (NULL_TREE, n_gpr));
TREE_SIDE_EFFECTS (t) = 1;
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
}
if (cfun->va_list_fpr_size)
{
t = build2 (MODIFY_EXPR, TREE_TYPE (fpr), fpr,
build_int_cst (NULL_TREE, n_fpr));
TREE_SIDE_EFFECTS (t) = 1;
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
#ifdef HAVE_AS_GNU_ATTRIBUTE
if (call_ABI_of_interest (cfun->decl))
rs6000_passes_float = true;
#endif
}
t = make_tree (TREE_TYPE (ovf), crtl->args.internal_arg_pointer);
if (words != 0)
t = fold_build_pointer_plus_hwi (t, words * MIN_UNITS_PER_WORD);
t = build2 (MODIFY_EXPR, TREE_TYPE (ovf), ovf, t);
TREE_SIDE_EFFECTS (t) = 1;
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
if (!cfun->va_list_gpr_size
&& !cfun->va_list_fpr_size
&& n_gpr < GP_ARG_NUM_REG
&& n_fpr < FP_ARG_V4_MAX_REG)
return;
t = make_tree (TREE_TYPE (sav), virtual_stack_vars_rtx);
if (cfun->machine->varargs_save_offset)
t = fold_build_pointer_plus_hwi (t, cfun->machine->varargs_save_offset);
t = build2 (MODIFY_EXPR, TREE_TYPE (sav), sav, t);
TREE_SIDE_EFFECTS (t) = 1;
expand_expr (t, const0_rtx, VOIDmode, EXPAND_NORMAL);
}
static tree
rs6000_gimplify_va_arg (tree valist, tree type, gimple_seq *pre_p,
gimple_seq *post_p)
{
tree f_gpr, f_fpr, f_res, f_ovf, f_sav;
tree gpr, fpr, ovf, sav, reg, t, u;
int size, rsize, n_reg, sav_ofs, sav_scale;
tree lab_false, lab_over, addr;
int align;
tree ptrtype = build_pointer_type_for_mode (type, ptr_mode, true);
int regalign = 0;
gimple *stmt;
if (pass_by_reference (NULL, TYPE_MODE (type), type, false))
{
t = rs6000_gimplify_va_arg (valist, ptrtype, pre_p, post_p);
return build_va_arg_indirect_ref (t);
}
if (((TARGET_MACHO
&& rs6000_darwin64_abi)
|| DEFAULT_ABI == ABI_ELFv2
|| (DEFAULT_ABI == ABI_AIX && !rs6000_compat_align_parm))
&& integer_zerop (TYPE_SIZE (type)))
{
unsigned HOST_WIDE_INT align, boundary;
tree valist_tmp = get_initialized_tmp_var (valist, pre_p, NULL);
align = PARM_BOUNDARY / BITS_PER_UNIT;
boundary = rs6000_function_arg_boundary (TYPE_MODE (type), type);
if (boundary > MAX_SUPPORTED_STACK_ALIGNMENT)
boundary = MAX_SUPPORTED_STACK_ALIGNMENT;
boundary /= BITS_PER_UNIT;
if (boundary > align)
{
tree t ;
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist_tmp,
fold_build_pointer_plus_hwi (valist_tmp, boundary - 1));
gimplify_and_add (t, pre_p);
t = fold_convert (sizetype, valist_tmp);
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist_tmp,
fold_convert (TREE_TYPE (valist),
fold_build2 (BIT_AND_EXPR, sizetype, t,
size_int (-boundary))));
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist, t);
gimplify_and_add (t, pre_p);
}
valist_tmp = fold_convert (build_pointer_type (type), valist_tmp);
return build_va_arg_indirect_ref (valist_tmp);
}
if (DEFAULT_ABI != ABI_V4)
{
if (targetm.calls.split_complex_arg && TREE_CODE (type) == COMPLEX_TYPE)
{
tree elem_type = TREE_TYPE (type);
machine_mode elem_mode = TYPE_MODE (elem_type);
int elem_size = GET_MODE_SIZE (elem_mode);
if (elem_size < UNITS_PER_WORD)
{
tree real_part, imag_part;
gimple_seq post = NULL;
real_part = rs6000_gimplify_va_arg (valist, elem_type, pre_p,
&post);
real_part = get_initialized_tmp_var (real_part, pre_p, &post);
gimple_seq_add_seq (pre_p, post);
imag_part = rs6000_gimplify_va_arg (valist, elem_type, pre_p,
post_p);
return build2 (COMPLEX_EXPR, type, real_part, imag_part);
}
}
return std_gimplify_va_arg_expr (valist, type, pre_p, post_p);
}
f_gpr = TYPE_FIELDS (TREE_TYPE (va_list_type_node));
f_fpr = DECL_CHAIN (f_gpr);
f_res = DECL_CHAIN (f_fpr);
f_ovf = DECL_CHAIN (f_res);
f_sav = DECL_CHAIN (f_ovf);
gpr = build3 (COMPONENT_REF, TREE_TYPE (f_gpr), valist, f_gpr, NULL_TREE);
fpr = build3 (COMPONENT_REF, TREE_TYPE (f_fpr), unshare_expr (valist),
f_fpr, NULL_TREE);
ovf = build3 (COMPONENT_REF, TREE_TYPE (f_ovf), unshare_expr (valist),
f_ovf, NULL_TREE);
sav = build3 (COMPONENT_REF, TREE_TYPE (f_sav), unshare_expr (valist),
f_sav, NULL_TREE);
size = int_size_in_bytes (type);
rsize = (size + 3) / 4;
int pad = 4 * rsize - size;
align = 1;
machine_mode mode = TYPE_MODE (type);
if (abi_v4_pass_in_fpr (mode, false))
{
reg = fpr;
n_reg = (size + 7) / 8;
sav_ofs = ((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT) ? 8 : 4) * 4;
sav_scale = ((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT) ? 8 : 4);
if (mode != SFmode && mode != SDmode)
align = 8;
}
else
{
reg = gpr;
n_reg = rsize;
sav_ofs = 0;
sav_scale = 4;
if (n_reg == 2)
align = 8;
}
lab_over = NULL;
addr = create_tmp_var (ptr_type_node, "addr");
if (TARGET_ALTIVEC_ABI && ALTIVEC_VECTOR_MODE (mode))
align = 16;
else
{
lab_false = create_artificial_label (input_location);
lab_over = create_artificial_label (input_location);
u = reg;
if (n_reg == 2 && reg == gpr)
{
regalign = 1;
u = build2 (BIT_AND_EXPR, TREE_TYPE (reg), unshare_expr (reg),
build_int_cst (TREE_TYPE (reg), n_reg - 1));
u = build2 (POSTINCREMENT_EXPR, TREE_TYPE (reg),
unshare_expr (reg), u);
}
else if (reg == fpr && mode == TDmode)
{
t = build2 (BIT_IOR_EXPR, TREE_TYPE (reg), unshare_expr (reg),
build_int_cst (TREE_TYPE (reg), 1));
u = build2 (MODIFY_EXPR, void_type_node, unshare_expr (reg), t);
}
t = fold_convert (TREE_TYPE (reg), size_int (8 - n_reg + 1));
t = build2 (GE_EXPR, boolean_type_node, u, t);
u = build1 (GOTO_EXPR, void_type_node, lab_false);
t = build3 (COND_EXPR, void_type_node, t, u, NULL_TREE);
gimplify_and_add (t, pre_p);
t = sav;
if (sav_ofs)
t = fold_build_pointer_plus_hwi (sav, sav_ofs);
u = build2 (POSTINCREMENT_EXPR, TREE_TYPE (reg), unshare_expr (reg),
build_int_cst (TREE_TYPE (reg), n_reg));
u = fold_convert (sizetype, u);
u = build2 (MULT_EXPR, sizetype, u, size_int (sav_scale));
t = fold_build_pointer_plus (t, u);
if (TARGET_32BIT && TARGET_HARD_FLOAT && mode == SDmode)
t = fold_build_pointer_plus_hwi (t, size);
if (BYTES_BIG_ENDIAN)
t = fold_build_pointer_plus_hwi (t, pad);
gimplify_assign (addr, t, pre_p);
gimple_seq_add_stmt (pre_p, gimple_build_goto (lab_over));
stmt = gimple_build_label (lab_false);
gimple_seq_add_stmt (pre_p, stmt);
if ((n_reg == 2 && !regalign) || n_reg > 2)
{
gimplify_assign (reg, build_int_cst (TREE_TYPE (reg), 8), pre_p);
}
}
t = ovf;
if (align != 1)
{
t = fold_build_pointer_plus_hwi (t, align - 1);
t = build2 (BIT_AND_EXPR, TREE_TYPE (t), t,
build_int_cst (TREE_TYPE (t), -align));
}
if (BYTES_BIG_ENDIAN)
t = fold_build_pointer_plus_hwi (t, pad);
gimplify_expr (&t, pre_p, NULL, is_gimple_val, fb_rvalue);
gimplify_assign (unshare_expr (addr), t, pre_p);
t = fold_build_pointer_plus_hwi (t, size);
gimplify_assign (unshare_expr (ovf), t, pre_p);
if (lab_over)
{
stmt = gimple_build_label (lab_over);
gimple_seq_add_stmt (pre_p, stmt);
}
if (STRICT_ALIGNMENT
&& (TYPE_ALIGN (type)
> (unsigned) BITS_PER_UNIT * (align < 4 ? 4 : align)))
{
tree tmp = create_tmp_var (type, "va_arg_tmp");
tree dest_addr = build_fold_addr_expr (tmp);
tree copy = build_call_expr (builtin_decl_implicit (BUILT_IN_MEMCPY),
3, dest_addr, addr, size_int (rsize * 4));
TREE_ADDRESSABLE (tmp) = 1;
gimplify_and_add (copy, pre_p);
addr = dest_addr;
}
addr = fold_convert (ptrtype, addr);
return build_va_arg_indirect_ref (addr);
}
static void
def_builtin (const char *name, tree type, enum rs6000_builtins code)
{
tree t;
unsigned classify = rs6000_builtin_info[(int)code].attr;
const char *attr_string = "";
gcc_assert (name != NULL);
gcc_assert (IN_RANGE ((int)code, 0, (int)RS6000_BUILTIN_COUNT));
if (rs6000_builtin_decls[(int)code])
fatal_error (input_location,
"internal error: builtin function %qs already processed",
name);
rs6000_builtin_decls[(int)code] = t =
add_builtin_function (name, type, (int)code, BUILT_IN_MD, NULL, NULL_TREE);
if ((classify & RS6000_BTC_CONST) != 0)
{
TREE_READONLY (t) = 1;
TREE_NOTHROW (t) = 1;
attr_string = ", const";
}
else if ((classify & RS6000_BTC_PURE) != 0)
{
DECL_PURE_P (t) = 1;
TREE_NOTHROW (t) = 1;
attr_string = ", pure";
}
else if ((classify & RS6000_BTC_FP) != 0)
{
TREE_NOTHROW (t) = 1;
if (flag_rounding_math)
{
DECL_PURE_P (t) = 1;
DECL_IS_NOVOPS (t) = 1;
attr_string = ", fp, pure";
}
else
{
TREE_READONLY (t) = 1;
attr_string = ", fp, const";
}
}
else if ((classify & RS6000_BTC_ATTR_MASK) != 0)
gcc_unreachable ();
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, code = %4d, %s%s\n",
(int)code, name, attr_string);
}
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_3arg[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_dst[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_2arg[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_altivec_preds[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_paired_preds[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_abs[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_1arg[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_0arg[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE) \
{ MASK, ICODE, NAME, ENUM },
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE)
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE)
static const struct builtin_description bdesc_htm[] =
{
#include "rs6000-builtin.def"
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
bool
rs6000_overloaded_builtin_p (enum rs6000_builtins fncode)
{
return (rs6000_builtin_info[(int)fncode].attr & RS6000_BTC_OVERLOADED) != 0;
}
const char *
rs6000_overloaded_builtin_name (enum rs6000_builtins fncode)
{
return rs6000_builtin_info[(int)fncode].name;
}
static rtx
rs6000_expand_zeroop_builtin (enum insn_code icode, rtx target)
{
rtx pat;
machine_mode tmode = insn_data[icode].operand[0].mode;
if (icode == CODE_FOR_nothing)
return 0;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
pat = GEN_FCN (icode) (target);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
rs6000_expand_mtfsf_builtin (enum insn_code icode, tree exp)
{
rtx pat;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
machine_mode mode0 = insn_data[icode].operand[0].mode;
machine_mode mode1 = insn_data[icode].operand[1].mode;
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (GET_CODE (op0) != CONST_INT
|| INTVAL (op0) > 255
|| INTVAL (op0) < 0)
{
error ("argument 1 must be an 8-bit field value");
return const0_rtx;
}
if (! (*insn_data[icode].operand[0].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (! (*insn_data[icode].operand[1].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
pat = GEN_FCN (icode) (op0, op1);
if (! pat)
return const0_rtx;
emit_insn (pat);
return NULL_RTX;
}
static rtx
rs6000_expand_unop_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat;
tree arg0 = CALL_EXPR_ARG (exp, 0);
rtx op0 = expand_normal (arg0);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = insn_data[icode].operand[1].mode;
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node)
return const0_rtx;
if (icode == CODE_FOR_altivec_vspltisb
|| icode == CODE_FOR_altivec_vspltish
|| icode == CODE_FOR_altivec_vspltisw)
{
if (GET_CODE (op0) != CONST_INT
|| INTVAL (op0) > 15
|| INTVAL (op0) < -16)
{
error ("argument 1 must be a 5-bit signed literal");
return CONST0_RTX (tmode);
}
}
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
pat = GEN_FCN (icode) (target, op0);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
altivec_expand_abs_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat, scratch1, scratch2;
tree arg0 = CALL_EXPR_ARG (exp, 0);
rtx op0 = expand_normal (arg0);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = insn_data[icode].operand[1].mode;
if (arg0 == error_mark_node)
return const0_rtx;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
scratch1 = gen_reg_rtx (mode0);
scratch2 = gen_reg_rtx (mode0);
pat = GEN_FCN (icode) (target, op0, scratch1, scratch2);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
rs6000_expand_binop_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = insn_data[icode].operand[1].mode;
machine_mode mode1 = insn_data[icode].operand[2].mode;
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (icode == CODE_FOR_altivec_vcfux
|| icode == CODE_FOR_altivec_vcfsx
|| icode == CODE_FOR_altivec_vctsxs
|| icode == CODE_FOR_altivec_vctuxs
|| icode == CODE_FOR_altivec_vspltb
|| icode == CODE_FOR_altivec_vsplth
|| icode == CODE_FOR_altivec_vspltw)
{
STRIP_NOPS (arg1);
if (TREE_CODE (arg1) != INTEGER_CST
|| TREE_INT_CST_LOW (arg1) & ~0x1f)
{
error ("argument 2 must be a 5-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_dfptstsfi_eq_dd
|| icode == CODE_FOR_dfptstsfi_lt_dd
|| icode == CODE_FOR_dfptstsfi_gt_dd
|| icode == CODE_FOR_dfptstsfi_unordered_dd
|| icode == CODE_FOR_dfptstsfi_eq_td
|| icode == CODE_FOR_dfptstsfi_lt_td
|| icode == CODE_FOR_dfptstsfi_gt_td
|| icode == CODE_FOR_dfptstsfi_unordered_td)
{
STRIP_NOPS (arg0);
if (TREE_CODE (arg0) != INTEGER_CST
|| !IN_RANGE (TREE_INT_CST_LOW (arg0), 0, 63))
{
error ("argument 1 must be a 6-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_xststdcqp_kf
|| icode == CODE_FOR_xststdcqp_tf
|| icode == CODE_FOR_xststdcdp
|| icode == CODE_FOR_xststdcsp
|| icode == CODE_FOR_xvtstdcdp
|| icode == CODE_FOR_xvtstdcsp)
{
STRIP_NOPS (arg1);
if (TREE_CODE (arg1) != INTEGER_CST
|| !IN_RANGE (TREE_INT_CST_LOW (arg1), 0, 127))
{
error ("argument 2 must be a 7-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_unpackv1ti
|| icode == CODE_FOR_unpackkf
|| icode == CODE_FOR_unpacktf
|| icode == CODE_FOR_unpackif
|| icode == CODE_FOR_unpacktd)
{
STRIP_NOPS (arg1);
if (TREE_CODE (arg1) != INTEGER_CST
|| !IN_RANGE (TREE_INT_CST_LOW (arg1), 0, 1))
{
error ("argument 2 must be a 1-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (! (*insn_data[icode].operand[2].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
pat = GEN_FCN (icode) (target, op0, op1);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
altivec_expand_predicate_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat, scratch;
tree cr6_form = CALL_EXPR_ARG (exp, 0);
tree arg0 = CALL_EXPR_ARG (exp, 1);
tree arg1 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
machine_mode tmode = SImode;
machine_mode mode0 = insn_data[icode].operand[1].mode;
machine_mode mode1 = insn_data[icode].operand[2].mode;
int cr6_form_int;
if (TREE_CODE (cr6_form) != INTEGER_CST)
{
error ("argument 1 of %qs must be a constant",
"__builtin_altivec_predicate");
return const0_rtx;
}
else
cr6_form_int = TREE_INT_CST_LOW (cr6_form);
gcc_assert (mode0 == mode1);
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (! (*insn_data[icode].operand[2].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
scratch = gen_reg_rtx (mode0);
pat = GEN_FCN (icode) (scratch, op0, op1);
if (! pat)
return 0;
emit_insn (pat);
switch (cr6_form_int)
{
case 0:
emit_insn (gen_cr6_test_for_zero (target));
break;
case 1:
emit_insn (gen_cr6_test_for_zero_reverse (target));
break;
case 2:
emit_insn (gen_cr6_test_for_lt (target));
break;
case 3:
emit_insn (gen_cr6_test_for_lt_reverse (target));
break;
default:
error ("argument 1 of %qs is out of range",
"__builtin_altivec_predicate");
break;
}
return target;
}
static rtx
paired_expand_lv_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat, addr;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = Pmode;
machine_mode mode1 = Pmode;
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
op1 = copy_to_mode_reg (mode1, op1);
if (op0 == const0_rtx)
{
addr = gen_rtx_MEM (tmode, op1);
}
else
{
op0 = copy_to_mode_reg (mode0, op0);
addr = gen_rtx_MEM (tmode, gen_rtx_PLUS (Pmode, op0, op1));
}
pat = GEN_FCN (icode) (target, addr);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
swap_selector_for_mode (machine_mode mode)
{
unsigned int swap2[16] = {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8};
unsigned int swap4[16] = {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
unsigned int swap8[16] = {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
unsigned int swap16[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
unsigned int *swaparray, i;
rtx perm[16];
switch (mode)
{
case E_V2DFmode:
case E_V2DImode:
swaparray = swap2;
break;
case E_V4SFmode:
case E_V4SImode:
swaparray = swap4;
break;
case E_V8HImode:
swaparray = swap8;
break;
case E_V16QImode:
swaparray = swap16;
break;
default:
gcc_unreachable ();
}
for (i = 0; i < 16; ++i)
perm[i] = GEN_INT (swaparray[i]);
return force_reg (V16QImode, gen_rtx_CONST_VECTOR (V16QImode, gen_rtvec_v (16, perm)));
}
rtx
swap_endian_selector_for_mode (machine_mode mode)
{
unsigned int swap1[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
unsigned int swap2[16] = {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8};
unsigned int swap4[16] = {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
unsigned int swap8[16] = {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
unsigned int *swaparray, i;
rtx perm[16];
switch (mode)
{
case E_V1TImode:
swaparray = swap1;
break;
case E_V2DFmode:
case E_V2DImode:
swaparray = swap2;
break;
case E_V4SFmode:
case E_V4SImode:
swaparray = swap4;
break;
case E_V8HImode:
swaparray = swap8;
break;
default:
gcc_unreachable ();
}
for (i = 0; i < 16; ++i)
perm[i] = GEN_INT (swaparray[i]);
return force_reg (V16QImode, gen_rtx_CONST_VECTOR (V16QImode,
gen_rtvec_v (16, perm)));
}
void
altivec_expand_lvx_be (rtx op0, rtx op1, machine_mode mode, unsigned unspec)
{
rtx tmp = gen_reg_rtx (mode);
rtx load = gen_rtx_SET (tmp, op1);
rtx lvx = gen_rtx_UNSPEC (mode, gen_rtvec (1, const0_rtx), unspec);
rtx par = gen_rtx_PARALLEL (mode, gen_rtvec (2, load, lvx));
rtx sel = swap_selector_for_mode (mode);
rtx vperm = gen_rtx_UNSPEC (mode, gen_rtvec (3, tmp, tmp, sel), UNSPEC_VPERM);
gcc_assert (REG_P (op0));
emit_insn (par);
emit_insn (gen_rtx_SET (op0, vperm));
}
void
altivec_expand_stvx_be (rtx op0, rtx op1, machine_mode mode, unsigned unspec)
{
rtx tmp = gen_reg_rtx (mode);
rtx store = gen_rtx_SET (op0, tmp);
rtx stvx = gen_rtx_UNSPEC (mode, gen_rtvec (1, const0_rtx), unspec);
rtx par = gen_rtx_PARALLEL (mode, gen_rtvec (2, store, stvx));
rtx sel = swap_selector_for_mode (mode);
rtx vperm;
gcc_assert (REG_P (op1));
vperm = gen_rtx_UNSPEC (mode, gen_rtvec (3, op1, op1, sel), UNSPEC_VPERM);
emit_insn (gen_rtx_SET (tmp, vperm));
emit_insn (par);
}
void
altivec_expand_stvex_be (rtx op0, rtx op1, machine_mode mode, unsigned unspec)
{
machine_mode inner_mode = GET_MODE_INNER (mode);
rtx tmp = gen_reg_rtx (mode);
rtx stvx = gen_rtx_UNSPEC (inner_mode, gen_rtvec (1, tmp), unspec);
rtx sel = swap_selector_for_mode (mode);
rtx vperm;
gcc_assert (REG_P (op1));
vperm = gen_rtx_UNSPEC (mode, gen_rtvec (3, op1, op1, sel), UNSPEC_VPERM);
emit_insn (gen_rtx_SET (tmp, vperm));
emit_insn (gen_rtx_SET (op0, stvx));
}
static rtx
altivec_expand_lv_builtin (enum insn_code icode, tree exp, rtx target, bool blk)
{
rtx pat, addr;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = Pmode;
machine_mode mode1 = Pmode;
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
op1 = copy_to_mode_reg (mode1, op1);
if (icode == CODE_FOR_altivec_lvx_v1ti
|| icode == CODE_FOR_altivec_lvx_v2df
|| icode == CODE_FOR_altivec_lvx_v2di
|| icode == CODE_FOR_altivec_lvx_v4sf
|| icode == CODE_FOR_altivec_lvx_v4si
|| icode == CODE_FOR_altivec_lvx_v8hi
|| icode == CODE_FOR_altivec_lvx_v16qi)
{
rtx rawaddr;
if (op0 == const0_rtx)
rawaddr = op1;
else
{
op0 = copy_to_mode_reg (mode0, op0);
rawaddr = gen_rtx_PLUS (Pmode, op1, op0);
}
addr = gen_rtx_AND (Pmode, rawaddr, gen_rtx_CONST_INT (Pmode, -16));
addr = gen_rtx_MEM (blk ? BLKmode : tmode, addr);
if (!BYTES_BIG_ENDIAN && VECTOR_ELT_ORDER_BIG)
{
rtx temp = gen_reg_rtx (tmode);
emit_insn (gen_rtx_SET (temp, addr));
rtx sel = swap_selector_for_mode (tmode);
rtx vperm = gen_rtx_UNSPEC (tmode, gen_rtvec (3, temp, temp, sel),
UNSPEC_VPERM);
emit_insn (gen_rtx_SET (target, vperm));
}
else
emit_insn (gen_rtx_SET (target, addr));
}
else
{
if (op0 == const0_rtx)
addr = gen_rtx_MEM (blk ? BLKmode : tmode, op1);
else
{
op0 = copy_to_mode_reg (mode0, op0);
addr = gen_rtx_MEM (blk ? BLKmode : tmode,
gen_rtx_PLUS (Pmode, op1, op0));
}
pat = GEN_FCN (icode) (target, addr);
if (! pat)
return 0;
emit_insn (pat);
}
return target;
}
static rtx
paired_expand_stv_builtin (enum insn_code icode, tree exp)
{
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
tree arg2 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
rtx op2 = expand_normal (arg2);
rtx pat, addr;
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode1 = Pmode;
machine_mode mode2 = Pmode;
if (arg0 == error_mark_node
|| arg1 == error_mark_node
|| arg2 == error_mark_node)
return const0_rtx;
if (! (*insn_data[icode].operand[1].predicate) (op0, tmode))
op0 = copy_to_mode_reg (tmode, op0);
op2 = copy_to_mode_reg (mode2, op2);
if (op1 == const0_rtx)
{
addr = gen_rtx_MEM (tmode, op2);
}
else
{
op1 = copy_to_mode_reg (mode1, op1);
addr = gen_rtx_MEM (tmode, gen_rtx_PLUS (Pmode, op1, op2));
}
pat = GEN_FCN (icode) (addr, op0);
if (pat)
emit_insn (pat);
return NULL_RTX;
}
static rtx
altivec_expand_stxvl_builtin (enum insn_code icode, tree exp)
{
rtx pat;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
tree arg2 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
rtx op2 = expand_normal (arg2);
machine_mode mode0 = insn_data[icode].operand[0].mode;
machine_mode mode1 = insn_data[icode].operand[1].mode;
machine_mode mode2 = insn_data[icode].operand[2].mode;
if (icode == CODE_FOR_nothing)
return NULL_RTX;
if (arg0 == error_mark_node
|| arg1 == error_mark_node
|| arg2 == error_mark_node)
return NULL_RTX;
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (! (*insn_data[icode].operand[2].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
if (! (*insn_data[icode].operand[3].predicate) (op2, mode2))
op2 = copy_to_mode_reg (mode2, op2);
pat = GEN_FCN (icode) (op0, op1, op2);
if (pat)
emit_insn (pat);
return NULL_RTX;
}
static rtx
altivec_expand_stv_builtin (enum insn_code icode, tree exp)
{
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
tree arg2 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
rtx op2 = expand_normal (arg2);
rtx pat, addr, rawaddr;
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode smode = insn_data[icode].operand[1].mode;
machine_mode mode1 = Pmode;
machine_mode mode2 = Pmode;
if (arg0 == error_mark_node
|| arg1 == error_mark_node
|| arg2 == error_mark_node)
return const0_rtx;
op2 = copy_to_mode_reg (mode2, op2);
if (icode == CODE_FOR_altivec_stvx_v2df
|| icode == CODE_FOR_altivec_stvx_v2di
|| icode == CODE_FOR_altivec_stvx_v4sf
|| icode == CODE_FOR_altivec_stvx_v4si
|| icode == CODE_FOR_altivec_stvx_v8hi
|| icode == CODE_FOR_altivec_stvx_v16qi)
{
if (op1 == const0_rtx)
rawaddr = op2;
else
{
op1 = copy_to_mode_reg (mode1, op1);
rawaddr = gen_rtx_PLUS (Pmode, op2, op1);
}
addr = gen_rtx_AND (Pmode, rawaddr, gen_rtx_CONST_INT (Pmode, -16));
addr = gen_rtx_MEM (tmode, addr);
op0 = copy_to_mode_reg (tmode, op0);
if (!BYTES_BIG_ENDIAN && VECTOR_ELT_ORDER_BIG)
{
rtx temp = gen_reg_rtx (tmode);
rtx sel = swap_selector_for_mode (tmode);
rtx vperm = gen_rtx_UNSPEC (tmode, gen_rtvec (3, op0, op0, sel),
UNSPEC_VPERM);
emit_insn (gen_rtx_SET (temp, vperm));
emit_insn (gen_rtx_SET (addr, temp));
}
else
emit_insn (gen_rtx_SET (addr, op0));
}
else
{
if (! (*insn_data[icode].operand[1].predicate) (op0, smode))
op0 = copy_to_mode_reg (smode, op0);
if (op1 == const0_rtx)
addr = gen_rtx_MEM (tmode, op2);
else
{
op1 = copy_to_mode_reg (mode1, op1);
addr = gen_rtx_MEM (tmode, gen_rtx_PLUS (Pmode, op2, op1));
}
pat = GEN_FCN (icode) (addr, op0);
if (pat)
emit_insn (pat);
}
return NULL_RTX;
}
static inline HOST_WIDE_INT
htm_spr_num (enum rs6000_builtins code)
{
if (code == HTM_BUILTIN_GET_TFHAR
|| code == HTM_BUILTIN_SET_TFHAR)
return TFHAR_SPR;
else if (code == HTM_BUILTIN_GET_TFIAR
|| code == HTM_BUILTIN_SET_TFIAR)
return TFIAR_SPR;
else if (code == HTM_BUILTIN_GET_TEXASR
|| code == HTM_BUILTIN_SET_TEXASR)
return TEXASR_SPR;
gcc_assert (code == HTM_BUILTIN_GET_TEXASRU
|| code == HTM_BUILTIN_SET_TEXASRU);
return TEXASRU_SPR;
}
static inline HOST_WIDE_INT
htm_spr_regno (enum rs6000_builtins code)
{
if (code == HTM_BUILTIN_GET_TFHAR
|| code == HTM_BUILTIN_SET_TFHAR)
return TFHAR_REGNO;
else if (code == HTM_BUILTIN_GET_TFIAR
|| code == HTM_BUILTIN_SET_TFIAR)
return TFIAR_REGNO;
gcc_assert (code == HTM_BUILTIN_GET_TEXASR
|| code == HTM_BUILTIN_SET_TEXASR
|| code == HTM_BUILTIN_GET_TEXASRU
|| code == HTM_BUILTIN_SET_TEXASRU);
return TEXASR_REGNO;
}
static inline enum insn_code
rs6000_htm_spr_icode (bool nonvoid)
{
if (nonvoid)
return (TARGET_POWERPC64) ? CODE_FOR_htm_mfspr_di : CODE_FOR_htm_mfspr_si;
else
return (TARGET_POWERPC64) ? CODE_FOR_htm_mtspr_di : CODE_FOR_htm_mtspr_si;
}
static rtx
htm_expand_builtin (tree exp, rtx target, bool * expandedp)
{
tree fndecl = TREE_OPERAND (CALL_EXPR_FN (exp), 0);
bool nonvoid = TREE_TYPE (TREE_TYPE (fndecl)) != void_type_node;
enum rs6000_builtins fcode = (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
const struct builtin_description *d;
size_t i;
*expandedp = true;
if (!TARGET_POWERPC64
&& (fcode == HTM_BUILTIN_TABORTDC
|| fcode == HTM_BUILTIN_TABORTDCI))
{
size_t uns_fcode = (size_t)fcode;
const char *name = rs6000_builtin_info[uns_fcode].name;
error ("builtin %qs is only valid in 64-bit mode", name);
return const0_rtx;
}
d = bdesc_htm;
for (i = 0; i < ARRAY_SIZE (bdesc_htm); i++, d++)
if (d->code == fcode)
{
rtx op[MAX_HTM_OPERANDS], pat;
int nopnds = 0;
tree arg;
call_expr_arg_iterator iter;
unsigned attr = rs6000_builtin_info[fcode].attr;
enum insn_code icode = d->icode;
const struct insn_operand_data *insn_op;
bool uses_spr = (attr & RS6000_BTC_SPR);
rtx cr = NULL_RTX;
if (uses_spr)
icode = rs6000_htm_spr_icode (nonvoid);
insn_op = &insn_data[icode].operand[0];
if (nonvoid)
{
machine_mode tmode = (uses_spr) ? insn_op->mode : E_SImode;
if (!target
|| GET_MODE (target) != tmode
|| (uses_spr && !(*insn_op->predicate) (target, tmode)))
target = gen_reg_rtx (tmode);
if (uses_spr)
op[nopnds++] = target;
}
FOR_EACH_CALL_EXPR_ARG (arg, iter, exp)
{
if (arg == error_mark_node || nopnds >= MAX_HTM_OPERANDS)
return const0_rtx;
insn_op = &insn_data[icode].operand[nopnds];
op[nopnds] = expand_normal (arg);
if (!(*insn_op->predicate) (op[nopnds], insn_op->mode))
{
if (!strcmp (insn_op->constraint, "n"))
{
int arg_num = (nonvoid) ? nopnds : nopnds + 1;
if (!CONST_INT_P (op[nopnds]))
error ("argument %d must be an unsigned literal", arg_num);
else
error ("argument %d is an unsigned literal that is "
"out of range", arg_num);
return const0_rtx;
}
op[nopnds] = copy_to_mode_reg (insn_op->mode, op[nopnds]);
}
nopnds++;
}
switch (fcode)
{
case HTM_BUILTIN_TENDALL:  
case HTM_BUILTIN_TRESUME:  
op[nopnds++] = GEN_INT (1);
if (flag_checking)
attr |= RS6000_BTC_UNARY;
break;
case HTM_BUILTIN_TSUSPEND: 
op[nopnds++] = GEN_INT (0);
if (flag_checking)
attr |= RS6000_BTC_UNARY;
break;
default:
break;
}
if (uses_spr)
{
machine_mode mode = (TARGET_POWERPC64) ? DImode : SImode;
op[nopnds++] = gen_rtx_CONST_INT (mode, htm_spr_num (fcode));
op[nopnds++] = gen_rtx_REG (mode, htm_spr_regno (fcode));
}
else if (attr & RS6000_BTC_CR)
{ cr = gen_reg_rtx (CCmode);
op[nopnds++] = cr;
}
if (flag_checking)
{
int expected_nopnds = 0;
if ((attr & RS6000_BTC_TYPE_MASK) == RS6000_BTC_UNARY)
expected_nopnds = 1;
else if ((attr & RS6000_BTC_TYPE_MASK) == RS6000_BTC_BINARY)
expected_nopnds = 2;
else if ((attr & RS6000_BTC_TYPE_MASK) == RS6000_BTC_TERNARY)
expected_nopnds = 3;
if (!(attr & RS6000_BTC_VOID))
expected_nopnds += 1;
if (uses_spr)
expected_nopnds += 2;
gcc_assert (nopnds == expected_nopnds
&& nopnds <= MAX_HTM_OPERANDS);
}
switch (nopnds)
{
case 1:
pat = GEN_FCN (icode) (op[0]);
break;
case 2:
pat = GEN_FCN (icode) (op[0], op[1]);
break;
case 3:
pat = GEN_FCN (icode) (op[0], op[1], op[2]);
break;
case 4:
pat = GEN_FCN (icode) (op[0], op[1], op[2], op[3]);
break;
default:
gcc_unreachable ();
}
if (!pat)
return NULL_RTX;
emit_insn (pat);
if (attr & RS6000_BTC_CR)
{
if (fcode == HTM_BUILTIN_TBEGIN)
{
rtx scratch = gen_reg_rtx (SImode);
emit_insn (gen_rtx_SET (scratch,
gen_rtx_EQ (SImode, cr,
const0_rtx)));
emit_insn (gen_rtx_SET (target,
gen_rtx_XOR (SImode, scratch,
GEN_INT (1))));
}
else
{
rtx scratch1 = gen_reg_rtx (SImode);
rtx scratch2 = gen_reg_rtx (SImode);
rtx subreg = simplify_gen_subreg (CCmode, scratch1, SImode, 0);
emit_insn (gen_movcc (subreg, cr));
emit_insn (gen_lshrsi3 (scratch2, scratch1, GEN_INT (28)));
emit_insn (gen_andsi3 (target, scratch2, GEN_INT (0xf)));
}
}
if (nonvoid)
return target;
return const0_rtx;
}
*expandedp = false;
return NULL_RTX;
}
static rtx
cpu_expand_builtin (enum rs6000_builtins fcode, tree exp ATTRIBUTE_UNUSED,
rtx target)
{
if (fcode == RS6000_BUILTIN_CPU_INIT)
return const0_rtx;
if (target == 0 || GET_MODE (target) != SImode)
target = gen_reg_rtx (SImode);
#ifdef TARGET_LIBC_PROVIDES_HWCAP_IN_TCB
tree arg = TREE_OPERAND (CALL_EXPR_ARG (exp, 0), 0);
if (TREE_CODE (arg) == ARRAY_REF
&& TREE_CODE (TREE_OPERAND (arg, 0)) == STRING_CST
&& TREE_CODE (TREE_OPERAND (arg, 1)) == INTEGER_CST
&& compare_tree_int (TREE_OPERAND (arg, 1), 0) == 0)
arg = TREE_OPERAND (arg, 0);
if (TREE_CODE (arg) != STRING_CST)
{
error ("builtin %qs only accepts a string argument",
rs6000_builtin_info[(size_t) fcode].name);
return const0_rtx;
}
if (fcode == RS6000_BUILTIN_CPU_IS)
{
const char *cpu = TREE_STRING_POINTER (arg);
rtx cpuid = NULL_RTX;
for (size_t i = 0; i < ARRAY_SIZE (cpu_is_info); i++)
if (strcmp (cpu, cpu_is_info[i].cpu) == 0)
{
cpuid = GEN_INT (cpu_is_info[i].cpuid + _DL_FIRST_PLATFORM);
break;
}
if (cpuid == NULL_RTX)
{
error ("cpu %qs is an invalid argument to builtin %qs",
cpu, rs6000_builtin_info[(size_t) fcode].name);
return const0_rtx;
}
rtx platform = gen_reg_rtx (SImode);
rtx tcbmem = gen_const_mem (SImode,
gen_rtx_PLUS (Pmode,
gen_rtx_REG (Pmode, TLS_REGNUM),
GEN_INT (TCB_PLATFORM_OFFSET)));
emit_move_insn (platform, tcbmem);
emit_insn (gen_eqsi3 (target, platform, cpuid));
}
else if (fcode == RS6000_BUILTIN_CPU_SUPPORTS)
{
const char *hwcap = TREE_STRING_POINTER (arg);
rtx mask = NULL_RTX;
int hwcap_offset;
for (size_t i = 0; i < ARRAY_SIZE (cpu_supports_info); i++)
if (strcmp (hwcap, cpu_supports_info[i].hwcap) == 0)
{
mask = GEN_INT (cpu_supports_info[i].mask);
hwcap_offset = TCB_HWCAP_OFFSET (cpu_supports_info[i].id);
break;
}
if (mask == NULL_RTX)
{
error ("%s %qs is an invalid argument to builtin %qs",
"hwcap", hwcap, rs6000_builtin_info[(size_t) fcode].name);
return const0_rtx;
}
rtx tcb_hwcap = gen_reg_rtx (SImode);
rtx tcbmem = gen_const_mem (SImode,
gen_rtx_PLUS (Pmode,
gen_rtx_REG (Pmode, TLS_REGNUM),
GEN_INT (hwcap_offset)));
emit_move_insn (tcb_hwcap, tcbmem);
rtx scratch1 = gen_reg_rtx (SImode);
emit_insn (gen_rtx_SET (scratch1, gen_rtx_AND (SImode, tcb_hwcap, mask)));
rtx scratch2 = gen_reg_rtx (SImode);
emit_insn (gen_eqsi3 (scratch2, scratch1, const0_rtx));
emit_insn (gen_rtx_SET (target, gen_rtx_XOR (SImode, scratch2, const1_rtx)));
}
else
gcc_unreachable ();
cpu_builtin_p = true;
#else
warning (0, "builtin %qs needs GLIBC (2.23 and newer) that exports hardware "
"capability bits", rs6000_builtin_info[(size_t) fcode].name);
emit_move_insn (target, GEN_INT (0));
#endif 
return target;
}
static rtx
rs6000_expand_ternop_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat;
tree arg0 = CALL_EXPR_ARG (exp, 0);
tree arg1 = CALL_EXPR_ARG (exp, 1);
tree arg2 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
rtx op2 = expand_normal (arg2);
machine_mode tmode = insn_data[icode].operand[0].mode;
machine_mode mode0 = insn_data[icode].operand[1].mode;
machine_mode mode1 = insn_data[icode].operand[2].mode;
machine_mode mode2 = insn_data[icode].operand[3].mode;
if (icode == CODE_FOR_nothing)
return 0;
if (arg0 == error_mark_node
|| arg1 == error_mark_node
|| arg2 == error_mark_node)
return const0_rtx;
if (icode == CODE_FOR_altivec_vsldoi_v4sf
|| icode == CODE_FOR_altivec_vsldoi_v2df
|| icode == CODE_FOR_altivec_vsldoi_v4si
|| icode == CODE_FOR_altivec_vsldoi_v8hi
|| icode == CODE_FOR_altivec_vsldoi_v16qi)
{
STRIP_NOPS (arg2);
if (TREE_CODE (arg2) != INTEGER_CST
|| TREE_INT_CST_LOW (arg2) & ~0xf)
{
error ("argument 3 must be a 4-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_vsx_xxpermdi_v2df
|| icode == CODE_FOR_vsx_xxpermdi_v2di
|| icode == CODE_FOR_vsx_xxpermdi_v2df_be
|| icode == CODE_FOR_vsx_xxpermdi_v2di_be
|| icode == CODE_FOR_vsx_xxpermdi_v1ti
|| icode == CODE_FOR_vsx_xxpermdi_v4sf
|| icode == CODE_FOR_vsx_xxpermdi_v4si
|| icode == CODE_FOR_vsx_xxpermdi_v8hi
|| icode == CODE_FOR_vsx_xxpermdi_v16qi
|| icode == CODE_FOR_vsx_xxsldwi_v16qi
|| icode == CODE_FOR_vsx_xxsldwi_v8hi
|| icode == CODE_FOR_vsx_xxsldwi_v4si
|| icode == CODE_FOR_vsx_xxsldwi_v4sf
|| icode == CODE_FOR_vsx_xxsldwi_v2di
|| icode == CODE_FOR_vsx_xxsldwi_v2df)
{
STRIP_NOPS (arg2);
if (TREE_CODE (arg2) != INTEGER_CST
|| TREE_INT_CST_LOW (arg2) & ~0x3)
{
error ("argument 3 must be a 2-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_vsx_set_v2df
|| icode == CODE_FOR_vsx_set_v2di
|| icode == CODE_FOR_bcdadd
|| icode == CODE_FOR_bcdadd_lt
|| icode == CODE_FOR_bcdadd_eq
|| icode == CODE_FOR_bcdadd_gt
|| icode == CODE_FOR_bcdsub
|| icode == CODE_FOR_bcdsub_lt
|| icode == CODE_FOR_bcdsub_eq
|| icode == CODE_FOR_bcdsub_gt)
{
STRIP_NOPS (arg2);
if (TREE_CODE (arg2) != INTEGER_CST
|| TREE_INT_CST_LOW (arg2) & ~0x1)
{
error ("argument 3 must be a 1-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_dfp_ddedpd_dd
|| icode == CODE_FOR_dfp_ddedpd_td)
{
STRIP_NOPS (arg0);
if (TREE_CODE (arg0) != INTEGER_CST
|| TREE_INT_CST_LOW (arg2) & ~0x3)
{
error ("argument 1 must be 0 or 2");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_dfp_denbcd_dd
|| icode == CODE_FOR_dfp_denbcd_td)
{
STRIP_NOPS (arg0);
if (TREE_CODE (arg0) != INTEGER_CST
|| TREE_INT_CST_LOW (arg0) & ~0x1)
{
error ("argument 1 must be a 1-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_dfp_dscli_dd
|| icode == CODE_FOR_dfp_dscli_td
|| icode == CODE_FOR_dfp_dscri_dd
|| icode == CODE_FOR_dfp_dscri_td)
{
STRIP_NOPS (arg1);
if (TREE_CODE (arg1) != INTEGER_CST
|| TREE_INT_CST_LOW (arg1) & ~0x3f)
{
error ("argument 2 must be a 6-bit unsigned literal");
return CONST0_RTX (tmode);
}
}
else if (icode == CODE_FOR_crypto_vshasigmaw
|| icode == CODE_FOR_crypto_vshasigmad)
{
STRIP_NOPS (arg1);
if (TREE_CODE (arg1) != INTEGER_CST || wi::geu_p (wi::to_wide (arg1), 2))
{
error ("argument 2 must be 0 or 1");
return CONST0_RTX (tmode);
}
STRIP_NOPS (arg2);
if (TREE_CODE (arg2) != INTEGER_CST
|| wi::geu_p (wi::to_wide (arg2), 16))
{
error ("argument 3 must be in the range 0..15");
return CONST0_RTX (tmode);
}
}
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
if (! (*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (! (*insn_data[icode].operand[2].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
if (! (*insn_data[icode].operand[3].predicate) (op2, mode2))
op2 = copy_to_mode_reg (mode2, op2);
if (TARGET_PAIRED_FLOAT && icode == CODE_FOR_selv2sf4)
pat = GEN_FCN (icode) (target, op0, op1, op2, CONST0_RTX (SFmode));
else 
pat = GEN_FCN (icode) (target, op0, op1, op2);
if (! pat)
return 0;
emit_insn (pat);
return target;
}
static rtx
altivec_expand_dst_builtin (tree exp, rtx target ATTRIBUTE_UNUSED,
bool *expandedp)
{
tree fndecl = TREE_OPERAND (CALL_EXPR_FN (exp), 0);
enum rs6000_builtins fcode = (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
tree arg0, arg1, arg2;
machine_mode mode0, mode1;
rtx pat, op0, op1, op2;
const struct builtin_description *d;
size_t i;
*expandedp = false;
d = bdesc_dst;
for (i = 0; i < ARRAY_SIZE (bdesc_dst); i++, d++)
if (d->code == fcode)
{
arg0 = CALL_EXPR_ARG (exp, 0);
arg1 = CALL_EXPR_ARG (exp, 1);
arg2 = CALL_EXPR_ARG (exp, 2);
op0 = expand_normal (arg0);
op1 = expand_normal (arg1);
op2 = expand_normal (arg2);
mode0 = insn_data[d->icode].operand[0].mode;
mode1 = insn_data[d->icode].operand[1].mode;
if (arg0 == error_mark_node
|| arg1 == error_mark_node
|| arg2 == error_mark_node)
return const0_rtx;
*expandedp = true;
STRIP_NOPS (arg2);
if (TREE_CODE (arg2) != INTEGER_CST
|| TREE_INT_CST_LOW (arg2) & ~0x3)
{
error ("argument to %qs must be a 2-bit unsigned literal", d->name);
return const0_rtx;
}
if (! (*insn_data[d->icode].operand[0].predicate) (op0, mode0))
op0 = copy_to_mode_reg (Pmode, op0);
if (! (*insn_data[d->icode].operand[1].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
pat = GEN_FCN (d->icode) (op0, op1, op2);
if (pat != 0)
emit_insn (pat);
return NULL_RTX;
}
return NULL_RTX;
}
static rtx
altivec_expand_vec_init_builtin (tree type, tree exp, rtx target)
{
machine_mode tmode = TYPE_MODE (type);
machine_mode inner_mode = GET_MODE_INNER (tmode);
int i, n_elt = GET_MODE_NUNITS (tmode);
gcc_assert (VECTOR_MODE_P (tmode));
gcc_assert (n_elt == call_expr_nargs (exp));
if (!target || !register_operand (target, tmode))
target = gen_reg_rtx (tmode);
if (n_elt == 1 && GET_MODE_SIZE (tmode) == GET_MODE_SIZE (inner_mode))
{
rtx x = expand_normal (CALL_EXPR_ARG (exp, 0));
emit_move_insn (target, gen_lowpart (tmode, x));
}
else
{
rtvec v = rtvec_alloc (n_elt);
for (i = 0; i < n_elt; ++i)
{
rtx x = expand_normal (CALL_EXPR_ARG (exp, i));
RTVEC_ELT (v, i) = gen_lowpart (inner_mode, x);
}
rs6000_expand_vector_init (target, gen_rtx_PARALLEL (tmode, v));
}
return target;
}
static int
get_element_number (tree vec_type, tree arg)
{
unsigned HOST_WIDE_INT elt, max = TYPE_VECTOR_SUBPARTS (vec_type) - 1;
if (!tree_fits_uhwi_p (arg)
|| (elt = tree_to_uhwi (arg), elt > max))
{
error ("selector must be an integer constant in the range 0..%wi", max);
return 0;
}
return elt;
}
static rtx
altivec_expand_vec_set_builtin (tree exp)
{
machine_mode tmode, mode1;
tree arg0, arg1, arg2;
int elt;
rtx op0, op1;
arg0 = CALL_EXPR_ARG (exp, 0);
arg1 = CALL_EXPR_ARG (exp, 1);
arg2 = CALL_EXPR_ARG (exp, 2);
tmode = TYPE_MODE (TREE_TYPE (arg0));
mode1 = TYPE_MODE (TREE_TYPE (TREE_TYPE (arg0)));
gcc_assert (VECTOR_MODE_P (tmode));
op0 = expand_expr (arg0, NULL_RTX, tmode, EXPAND_NORMAL);
op1 = expand_expr (arg1, NULL_RTX, mode1, EXPAND_NORMAL);
elt = get_element_number (TREE_TYPE (arg0), arg2);
if (GET_MODE (op1) != mode1 && GET_MODE (op1) != VOIDmode)
op1 = convert_modes (mode1, GET_MODE (op1), op1, true);
op0 = force_reg (tmode, op0);
op1 = force_reg (mode1, op1);
rs6000_expand_vector_set (op0, op1, elt);
return op0;
}
static rtx
altivec_expand_vec_ext_builtin (tree exp, rtx target)
{
machine_mode tmode, mode0;
tree arg0, arg1;
rtx op0;
rtx op1;
arg0 = CALL_EXPR_ARG (exp, 0);
arg1 = CALL_EXPR_ARG (exp, 1);
op0 = expand_normal (arg0);
op1 = expand_normal (arg1);
if (TREE_CODE (arg1) == INTEGER_CST)
(void) get_element_number (TREE_TYPE (arg0), arg1);
tmode = TYPE_MODE (TREE_TYPE (TREE_TYPE (arg0)));
mode0 = TYPE_MODE (TREE_TYPE (arg0));
gcc_assert (VECTOR_MODE_P (mode0));
op0 = force_reg (mode0, op0);
if (optimize || !target || !register_operand (target, tmode))
target = gen_reg_rtx (tmode);
rs6000_expand_vector_extract (target, op0, op1);
return target;
}
static rtx
altivec_expand_builtin (tree exp, rtx target, bool *expandedp)
{
const struct builtin_description *d;
size_t i;
enum insn_code icode;
tree fndecl = TREE_OPERAND (CALL_EXPR_FN (exp), 0);
tree arg0, arg1, arg2;
rtx op0, pat;
machine_mode tmode, mode0;
enum rs6000_builtins fcode
= (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
if (rs6000_overloaded_builtin_p (fcode))
{
*expandedp = true;
error ("unresolved overload for Altivec builtin %qF", fndecl);
return expand_call (exp, target, false);
}
target = altivec_expand_dst_builtin (exp, target, expandedp);
if (*expandedp)
return target;
*expandedp = true;
switch (fcode)
{
case ALTIVEC_BUILTIN_STVX_V2DF:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v2df, exp);
case ALTIVEC_BUILTIN_STVX_V2DI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v2di, exp);
case ALTIVEC_BUILTIN_STVX_V4SF:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v4sf, exp);
case ALTIVEC_BUILTIN_STVX:
case ALTIVEC_BUILTIN_STVX_V4SI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v4si, exp);
case ALTIVEC_BUILTIN_STVX_V8HI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v8hi, exp);
case ALTIVEC_BUILTIN_STVX_V16QI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvx_v16qi, exp);
case ALTIVEC_BUILTIN_STVEBX:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvebx, exp);
case ALTIVEC_BUILTIN_STVEHX:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvehx, exp);
case ALTIVEC_BUILTIN_STVEWX:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvewx, exp);
case ALTIVEC_BUILTIN_STVXL_V2DF:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v2df, exp);
case ALTIVEC_BUILTIN_STVXL_V2DI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v2di, exp);
case ALTIVEC_BUILTIN_STVXL_V4SF:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v4sf, exp);
case ALTIVEC_BUILTIN_STVXL:
case ALTIVEC_BUILTIN_STVXL_V4SI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v4si, exp);
case ALTIVEC_BUILTIN_STVXL_V8HI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v8hi, exp);
case ALTIVEC_BUILTIN_STVXL_V16QI:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvxl_v16qi, exp);
case ALTIVEC_BUILTIN_STVLX:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvlx, exp);
case ALTIVEC_BUILTIN_STVLXL:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvlxl, exp);
case ALTIVEC_BUILTIN_STVRX:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvrx, exp);
case ALTIVEC_BUILTIN_STVRXL:
return altivec_expand_stv_builtin (CODE_FOR_altivec_stvrxl, exp);
case P9V_BUILTIN_STXVL:
return altivec_expand_stxvl_builtin (CODE_FOR_stxvl, exp);
case P9V_BUILTIN_XST_LEN_R:
return altivec_expand_stxvl_builtin (CODE_FOR_xst_len_r, exp);
case VSX_BUILTIN_STXVD2X_V1TI:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v1ti, exp);
case VSX_BUILTIN_STXVD2X_V2DF:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v2df, exp);
case VSX_BUILTIN_STXVD2X_V2DI:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v2di, exp);
case VSX_BUILTIN_STXVW4X_V4SF:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v4sf, exp);
case VSX_BUILTIN_STXVW4X_V4SI:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v4si, exp);
case VSX_BUILTIN_STXVW4X_V8HI:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v8hi, exp);
case VSX_BUILTIN_STXVW4X_V16QI:
return altivec_expand_stv_builtin (CODE_FOR_vsx_store_v16qi, exp);
case VSX_BUILTIN_ST_ELEMREV_V1TI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v1ti
: CODE_FOR_vsx_st_elemrev_v1ti);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V2DF:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v2df
: CODE_FOR_vsx_st_elemrev_v2df);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V2DI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v2di
: CODE_FOR_vsx_st_elemrev_v2di);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V4SF:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v4sf
: CODE_FOR_vsx_st_elemrev_v4sf);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V4SI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v4si
: CODE_FOR_vsx_st_elemrev_v4si);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V8HI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v8hi
: CODE_FOR_vsx_st_elemrev_v8hi);
return altivec_expand_stv_builtin (code, exp);
}
case VSX_BUILTIN_ST_ELEMREV_V16QI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_store_v16qi
: CODE_FOR_vsx_st_elemrev_v16qi);
return altivec_expand_stv_builtin (code, exp);
}
case ALTIVEC_BUILTIN_MFVSCR:
icode = CODE_FOR_altivec_mfvscr;
tmode = insn_data[icode].operand[0].mode;
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
pat = GEN_FCN (icode) (target);
if (! pat)
return 0;
emit_insn (pat);
return target;
case ALTIVEC_BUILTIN_MTVSCR:
icode = CODE_FOR_altivec_mtvscr;
arg0 = CALL_EXPR_ARG (exp, 0);
op0 = expand_normal (arg0);
mode0 = insn_data[icode].operand[0].mode;
if (arg0 == error_mark_node)
return const0_rtx;
if (! (*insn_data[icode].operand[0].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
pat = GEN_FCN (icode) (op0);
if (pat)
emit_insn (pat);
return NULL_RTX;
case ALTIVEC_BUILTIN_DSSALL:
emit_insn (gen_altivec_dssall ());
return NULL_RTX;
case ALTIVEC_BUILTIN_DSS:
icode = CODE_FOR_altivec_dss;
arg0 = CALL_EXPR_ARG (exp, 0);
STRIP_NOPS (arg0);
op0 = expand_normal (arg0);
mode0 = insn_data[icode].operand[0].mode;
if (arg0 == error_mark_node)
return const0_rtx;
if (TREE_CODE (arg0) != INTEGER_CST
|| TREE_INT_CST_LOW (arg0) & ~0x3)
{
error ("argument to %qs must be a 2-bit unsigned literal", "dss");
return const0_rtx;
}
if (! (*insn_data[icode].operand[0].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
emit_insn (gen_altivec_dss (op0));
return NULL_RTX;
case ALTIVEC_BUILTIN_VEC_INIT_V4SI:
case ALTIVEC_BUILTIN_VEC_INIT_V8HI:
case ALTIVEC_BUILTIN_VEC_INIT_V16QI:
case ALTIVEC_BUILTIN_VEC_INIT_V4SF:
case VSX_BUILTIN_VEC_INIT_V2DF:
case VSX_BUILTIN_VEC_INIT_V2DI:
case VSX_BUILTIN_VEC_INIT_V1TI:
return altivec_expand_vec_init_builtin (TREE_TYPE (exp), exp, target);
case ALTIVEC_BUILTIN_VEC_SET_V4SI:
case ALTIVEC_BUILTIN_VEC_SET_V8HI:
case ALTIVEC_BUILTIN_VEC_SET_V16QI:
case ALTIVEC_BUILTIN_VEC_SET_V4SF:
case VSX_BUILTIN_VEC_SET_V2DF:
case VSX_BUILTIN_VEC_SET_V2DI:
case VSX_BUILTIN_VEC_SET_V1TI:
return altivec_expand_vec_set_builtin (exp);
case ALTIVEC_BUILTIN_VEC_EXT_V4SI:
case ALTIVEC_BUILTIN_VEC_EXT_V8HI:
case ALTIVEC_BUILTIN_VEC_EXT_V16QI:
case ALTIVEC_BUILTIN_VEC_EXT_V4SF:
case VSX_BUILTIN_VEC_EXT_V2DF:
case VSX_BUILTIN_VEC_EXT_V2DI:
case VSX_BUILTIN_VEC_EXT_V1TI:
return altivec_expand_vec_ext_builtin (exp, target);
case P9V_BUILTIN_VEC_EXTRACT4B:
arg1 = CALL_EXPR_ARG (exp, 1);
STRIP_NOPS (arg1);
if (arg1 == error_mark_node)
return expand_call (exp, target, false);
if (TREE_CODE (arg1) != INTEGER_CST || TREE_INT_CST_LOW (arg1) > 12)
{
error ("second argument to %qs must be 0..12", "vec_vextract4b");
return expand_call (exp, target, false);
}
break;
case P9V_BUILTIN_VEC_INSERT4B:
arg2 = CALL_EXPR_ARG (exp, 2);
STRIP_NOPS (arg2);
if (arg2 == error_mark_node)
return expand_call (exp, target, false);
if (TREE_CODE (arg2) != INTEGER_CST || TREE_INT_CST_LOW (arg2) > 12)
{
error ("third argument to %qs must be 0..12", "vec_vinsert4b");
return expand_call (exp, target, false);
}
break;
default:
break;
}
d = bdesc_abs;
for (i = 0; i < ARRAY_SIZE (bdesc_abs); i++, d++)
if (d->code == fcode)
return altivec_expand_abs_builtin (d->icode, exp, target);
d = bdesc_altivec_preds;
for (i = 0; i < ARRAY_SIZE (bdesc_altivec_preds); i++, d++)
if (d->code == fcode)
return altivec_expand_predicate_builtin (d->icode, exp, target);
switch (fcode)
{
case ALTIVEC_BUILTIN_LVSL:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvsl,
exp, target, false);
case ALTIVEC_BUILTIN_LVSR:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvsr,
exp, target, false);
case ALTIVEC_BUILTIN_LVEBX:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvebx,
exp, target, false);
case ALTIVEC_BUILTIN_LVEHX:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvehx,
exp, target, false);
case ALTIVEC_BUILTIN_LVEWX:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvewx,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL_V2DF:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v2df,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL_V2DI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v2di,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL_V4SF:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v4sf,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL:
case ALTIVEC_BUILTIN_LVXL_V4SI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v4si,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL_V8HI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v8hi,
exp, target, false);
case ALTIVEC_BUILTIN_LVXL_V16QI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvxl_v16qi,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V1TI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v1ti,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V2DF:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v2df,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V2DI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v2di,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V4SF:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v4sf,
exp, target, false);
case ALTIVEC_BUILTIN_LVX:
case ALTIVEC_BUILTIN_LVX_V4SI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v4si,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V8HI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v8hi,
exp, target, false);
case ALTIVEC_BUILTIN_LVX_V16QI:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvx_v16qi,
exp, target, false);
case ALTIVEC_BUILTIN_LVLX:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvlx,
exp, target, true);
case ALTIVEC_BUILTIN_LVLXL:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvlxl,
exp, target, true);
case ALTIVEC_BUILTIN_LVRX:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvrx,
exp, target, true);
case ALTIVEC_BUILTIN_LVRXL:
return altivec_expand_lv_builtin (CODE_FOR_altivec_lvrxl,
exp, target, true);
case VSX_BUILTIN_LXVD2X_V1TI:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v1ti,
exp, target, false);
case VSX_BUILTIN_LXVD2X_V2DF:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v2df,
exp, target, false);
case VSX_BUILTIN_LXVD2X_V2DI:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v2di,
exp, target, false);
case VSX_BUILTIN_LXVW4X_V4SF:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v4sf,
exp, target, false);
case VSX_BUILTIN_LXVW4X_V4SI:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v4si,
exp, target, false);
case VSX_BUILTIN_LXVW4X_V8HI:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v8hi,
exp, target, false);
case VSX_BUILTIN_LXVW4X_V16QI:
return altivec_expand_lv_builtin (CODE_FOR_vsx_load_v16qi,
exp, target, false);
case VSX_BUILTIN_LD_ELEMREV_V2DF:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v2df
: CODE_FOR_vsx_ld_elemrev_v2df);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V1TI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v1ti
: CODE_FOR_vsx_ld_elemrev_v1ti);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V2DI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v2di
: CODE_FOR_vsx_ld_elemrev_v2di);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V4SF:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v4sf
: CODE_FOR_vsx_ld_elemrev_v4sf);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V4SI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v4si
: CODE_FOR_vsx_ld_elemrev_v4si);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V8HI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v8hi
: CODE_FOR_vsx_ld_elemrev_v8hi);
return altivec_expand_lv_builtin (code, exp, target, false);
}
case VSX_BUILTIN_LD_ELEMREV_V16QI:
{
enum insn_code code = (BYTES_BIG_ENDIAN ? CODE_FOR_vsx_load_v16qi
: CODE_FOR_vsx_ld_elemrev_v16qi);
return altivec_expand_lv_builtin (code, exp, target, false);
}
break;
default:
break;
}
*expandedp = false;
return NULL_RTX;
}
static rtx
paired_expand_builtin (tree exp, rtx target, bool * expandedp)
{
tree fndecl = TREE_OPERAND (CALL_EXPR_FN (exp), 0);
enum rs6000_builtins fcode = (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
const struct builtin_description *d;
size_t i;
*expandedp = true;
switch (fcode)
{
case PAIRED_BUILTIN_STX:
return paired_expand_stv_builtin (CODE_FOR_paired_stx, exp);
case PAIRED_BUILTIN_LX:
return paired_expand_lv_builtin (CODE_FOR_paired_lx, exp, target);
default:
break;
}
d = bdesc_paired_preds;
for (i = 0; i < ARRAY_SIZE (bdesc_paired_preds); i++, d++)
if (d->code == fcode)
return paired_expand_predicate_builtin (d->icode, exp, target);
*expandedp = false;
return NULL_RTX;
}
static rtx
paired_expand_predicate_builtin (enum insn_code icode, tree exp, rtx target)
{
rtx pat, scratch, tmp;
tree form = CALL_EXPR_ARG (exp, 0);
tree arg0 = CALL_EXPR_ARG (exp, 1);
tree arg1 = CALL_EXPR_ARG (exp, 2);
rtx op0 = expand_normal (arg0);
rtx op1 = expand_normal (arg1);
machine_mode mode0 = insn_data[icode].operand[1].mode;
machine_mode mode1 = insn_data[icode].operand[2].mode;
int form_int;
enum rtx_code code;
if (TREE_CODE (form) != INTEGER_CST)
{
error ("argument 1 of %s must be a constant",
"__builtin_paired_predicate");
return const0_rtx;
}
else
form_int = TREE_INT_CST_LOW (form);
gcc_assert (mode0 == mode1);
if (arg0 == error_mark_node || arg1 == error_mark_node)
return const0_rtx;
if (target == 0
|| GET_MODE (target) != SImode
|| !(*insn_data[icode].operand[0].predicate) (target, SImode))
target = gen_reg_rtx (SImode);
if (!(*insn_data[icode].operand[1].predicate) (op0, mode0))
op0 = copy_to_mode_reg (mode0, op0);
if (!(*insn_data[icode].operand[2].predicate) (op1, mode1))
op1 = copy_to_mode_reg (mode1, op1);
scratch = gen_reg_rtx (CCFPmode);
pat = GEN_FCN (icode) (scratch, op0, op1);
if (!pat)
return const0_rtx;
emit_insn (pat);
switch (form_int)
{
case 0:
code = LT;
break;
case 1:
code = GT;
break;
case 2:
code = EQ;
break;
case 3:
emit_insn (gen_move_from_CR_ov_bit (target, scratch));
return target;
default:
error ("argument 1 of %qs is out of range",
"__builtin_paired_predicate");
return const0_rtx;
}
tmp = gen_rtx_fmt_ee (code, SImode, scratch, const0_rtx);
emit_move_insn (target, tmp);
return target;
}
bool
rs6000_builtin_is_supported_p (enum rs6000_builtins fncode)
{
HOST_WIDE_INT fnmask = rs6000_builtin_info[fncode].mask;
if ((fnmask & rs6000_builtin_mask) != fnmask)
return false;
else
return true;
}
static void
rs6000_invalid_builtin (enum rs6000_builtins fncode)
{
size_t uns_fncode = (size_t) fncode;
const char *name = rs6000_builtin_info[uns_fncode].name;
HOST_WIDE_INT fnmask = rs6000_builtin_info[uns_fncode].mask;
gcc_assert (name != NULL);
if ((fnmask & RS6000_BTM_CELL) != 0)
error ("builtin function %qs is only valid for the cell processor", name);
else if ((fnmask & RS6000_BTM_VSX) != 0)
error ("builtin function %qs requires the %qs option", name, "-mvsx");
else if ((fnmask & RS6000_BTM_HTM) != 0)
error ("builtin function %qs requires the %qs option", name, "-mhtm");
else if ((fnmask & RS6000_BTM_ALTIVEC) != 0)
error ("builtin function %qs requires the %qs option", name, "-maltivec");
else if ((fnmask & RS6000_BTM_PAIRED) != 0)
error ("builtin function %qs requires the %qs option", name, "-mpaired");
else if ((fnmask & (RS6000_BTM_DFP | RS6000_BTM_P8_VECTOR))
== (RS6000_BTM_DFP | RS6000_BTM_P8_VECTOR))
error ("builtin function %qs requires the %qs and %qs options",
name, "-mhard-dfp", "-mpower8-vector");
else if ((fnmask & RS6000_BTM_DFP) != 0)
error ("builtin function %qs requires the %qs option", name, "-mhard-dfp");
else if ((fnmask & RS6000_BTM_P8_VECTOR) != 0)
error ("builtin function %qs requires the %qs option", name,
"-mpower8-vector");
else if ((fnmask & (RS6000_BTM_P9_VECTOR | RS6000_BTM_64BIT))
== (RS6000_BTM_P9_VECTOR | RS6000_BTM_64BIT))
error ("builtin function %qs requires the %qs and %qs options",
name, "-mcpu=power9", "-m64");
else if ((fnmask & RS6000_BTM_P9_VECTOR) != 0)
error ("builtin function %qs requires the %qs option", name,
"-mcpu=power9");
else if ((fnmask & (RS6000_BTM_P9_MISC | RS6000_BTM_64BIT))
== (RS6000_BTM_P9_MISC | RS6000_BTM_64BIT))
error ("builtin function %qs requires the %qs and %qs options",
name, "-mcpu=power9", "-m64");
else if ((fnmask & RS6000_BTM_P9_MISC) == RS6000_BTM_P9_MISC)
error ("builtin function %qs requires the %qs option", name,
"-mcpu=power9");
else if ((fnmask & RS6000_BTM_LDBL128) == RS6000_BTM_LDBL128)
{
if (!TARGET_HARD_FLOAT)
error ("builtin function %qs requires the %qs option", name,
"-mhard-float");
else
error ("builtin function %qs requires the %qs option", name,
TARGET_IEEEQUAD ? "-mabi=ibmlongdouble" : "-mlong-double-128");
}
else if ((fnmask & RS6000_BTM_HARD_FLOAT) != 0)
error ("builtin function %qs requires the %qs option", name,
"-mhard-float");
else if ((fnmask & RS6000_BTM_FLOAT128_HW) != 0)
error ("builtin function %qs requires ISA 3.0 IEEE 128-bit floating point",
name);
else if ((fnmask & RS6000_BTM_FLOAT128) != 0)
error ("builtin function %qs requires the %qs option", name, "-mfloat128");
else if ((fnmask & (RS6000_BTM_POPCNTD | RS6000_BTM_POWERPC64))
== (RS6000_BTM_POPCNTD | RS6000_BTM_POWERPC64))
error ("builtin function %qs requires the %qs (or newer), and "
"%qs or %qs options",
name, "-mcpu=power7", "-m64", "-mpowerpc64");
else
error ("builtin function %qs is not supported with the current options",
name);
}
static tree
rs6000_fold_builtin (tree fndecl ATTRIBUTE_UNUSED,
int n_args ATTRIBUTE_UNUSED,
tree *args ATTRIBUTE_UNUSED,
bool ignore ATTRIBUTE_UNUSED)
{
#ifdef SUBTARGET_FOLD_BUILTIN
return SUBTARGET_FOLD_BUILTIN (fndecl, n_args, args, ignore);
#else
return NULL_TREE;
#endif
}
static bool
rs6000_builtin_valid_without_lhs (enum rs6000_builtins fn_code)
{
switch (fn_code)
{
case ALTIVEC_BUILTIN_STVX_V16QI:
case ALTIVEC_BUILTIN_STVX_V8HI:
case ALTIVEC_BUILTIN_STVX_V4SI:
case ALTIVEC_BUILTIN_STVX_V4SF:
case ALTIVEC_BUILTIN_STVX_V2DI:
case ALTIVEC_BUILTIN_STVX_V2DF:
return true;
default:
return false;
}
}
static tree
fold_build_vec_cmp (tree_code code, tree type,
tree arg0, tree arg1)
{
tree cmp_type = build_same_sized_truth_vector_type (type);
tree zero_vec = build_zero_cst (type);
tree minus_one_vec = build_minus_one_cst (type);
tree cmp = fold_build2 (code, cmp_type, arg0, arg1);
return fold_build3 (VEC_COND_EXPR, type, cmp, minus_one_vec, zero_vec);
}
static void
fold_compare_helper (gimple_stmt_iterator *gsi, tree_code code, gimple *stmt)
{
tree arg0 = gimple_call_arg (stmt, 0);
tree arg1 = gimple_call_arg (stmt, 1);
tree lhs = gimple_call_lhs (stmt);
tree cmp = fold_build_vec_cmp (code, TREE_TYPE (lhs), arg0, arg1);
gimple *g = gimple_build_assign (lhs, cmp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
}
static void
fold_mergehl_helper (gimple_stmt_iterator *gsi, gimple *stmt, int use_high)
{
tree arg0 = gimple_call_arg (stmt, 0);
tree arg1 = gimple_call_arg (stmt, 1);
tree lhs = gimple_call_lhs (stmt);
tree lhs_type = TREE_TYPE (lhs);
tree lhs_type_type = TREE_TYPE (lhs_type);
int n_elts = TYPE_VECTOR_SUBPARTS (lhs_type);
int midpoint = n_elts / 2;
int offset = 0;
if (use_high == 1)
offset = midpoint;
tree_vector_builder elts (lhs_type, VECTOR_CST_NELTS (arg0), 1);
for (int i = 0; i < midpoint; i++)
{
elts.safe_push (build_int_cst (lhs_type_type, offset + i));
elts.safe_push (build_int_cst (lhs_type_type, offset + n_elts + i));
}
tree permute = elts.build ();
gimple *g = gimple_build_assign (lhs, VEC_PERM_EXPR, arg0, arg1, permute);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
}
bool
rs6000_gimple_fold_builtin (gimple_stmt_iterator *gsi)
{
gimple *stmt = gsi_stmt (*gsi);
tree fndecl = gimple_call_fndecl (stmt);
gcc_checking_assert (fndecl && DECL_BUILT_IN_CLASS (fndecl) == BUILT_IN_MD);
enum rs6000_builtins fn_code
= (enum rs6000_builtins) DECL_FUNCTION_CODE (fndecl);
tree arg0, arg1, lhs, temp;
enum tree_code bcode;
gimple *g;
size_t uns_fncode = (size_t) fn_code;
enum insn_code icode = rs6000_builtin_info[uns_fncode].icode;
const char *fn_name1 = rs6000_builtin_info[uns_fncode].name;
const char *fn_name2 = (icode != CODE_FOR_nothing)
? get_insn_name ((int) icode)
: "nothing";
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_gimple_fold_builtin %d %s %s\n",
fn_code, fn_name1, fn_name2);
if (!rs6000_fold_gimple)
return false;
if (!gimple_call_lhs (stmt) && !rs6000_builtin_valid_without_lhs (fn_code))
return false;
HOST_WIDE_INT mask = rs6000_builtin_info[uns_fncode].mask;
bool func_valid_p = (rs6000_builtin_mask & mask) == mask;
if (!func_valid_p)
return false;
switch (fn_code)
{
case ALTIVEC_BUILTIN_VADDUBM:
case ALTIVEC_BUILTIN_VADDUHM:
case ALTIVEC_BUILTIN_VADDUWM:
case P8V_BUILTIN_VADDUDM:
case ALTIVEC_BUILTIN_VADDFP:
case VSX_BUILTIN_XVADDDP:
bcode = PLUS_EXPR;
do_binary:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
if (INTEGRAL_TYPE_P (TREE_TYPE (TREE_TYPE (lhs)))
&& !TYPE_OVERFLOW_WRAPS (TREE_TYPE (TREE_TYPE (lhs))))
{
gimple_seq stmts = NULL;
tree type = unsigned_type_for (TREE_TYPE (lhs));
tree uarg0 = gimple_build (&stmts, VIEW_CONVERT_EXPR,
type, arg0);
tree uarg1 = gimple_build (&stmts, VIEW_CONVERT_EXPR,
type, arg1);
tree res = gimple_build (&stmts, gimple_location (stmt), bcode,
type, uarg0, uarg1);
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
g = gimple_build_assign (lhs, VIEW_CONVERT_EXPR,
build1 (VIEW_CONVERT_EXPR,
TREE_TYPE (lhs), res));
gsi_replace (gsi, g, true);
return true;
}
g = gimple_build_assign (lhs, bcode, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VSUBUBM:
case ALTIVEC_BUILTIN_VSUBUHM:
case ALTIVEC_BUILTIN_VSUBUWM:
case P8V_BUILTIN_VSUBUDM:
case ALTIVEC_BUILTIN_VSUBFP:
case VSX_BUILTIN_XVSUBDP:
bcode = MINUS_EXPR;
goto do_binary;
case VSX_BUILTIN_XVMULSP:
case VSX_BUILTIN_XVMULDP:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, MULT_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VMULESB:
case ALTIVEC_BUILTIN_VMULESH:
case P8V_BUILTIN_VMULESW:
case ALTIVEC_BUILTIN_VMULEUB:
case ALTIVEC_BUILTIN_VMULEUH:
case P8V_BUILTIN_VMULEUW:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, VEC_WIDEN_MULT_EVEN_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VMULOSB:
case ALTIVEC_BUILTIN_VMULOSH:
case P8V_BUILTIN_VMULOSW:
case ALTIVEC_BUILTIN_VMULOUB:
case ALTIVEC_BUILTIN_VMULOUH:
case P8V_BUILTIN_VMULOUW:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, VEC_WIDEN_MULT_ODD_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case VSX_BUILTIN_DIV_V2DI:
case VSX_BUILTIN_UDIV_V2DI:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, TRUNC_DIV_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case VSX_BUILTIN_XVDIVSP:
case VSX_BUILTIN_XVDIVDP:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, RDIV_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VAND:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, BIT_AND_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VANDC:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
temp = create_tmp_reg_or_ssa_name (TREE_TYPE (arg1));
g = gimple_build_assign (temp, BIT_NOT_EXPR, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_insert_before (gsi, g, GSI_SAME_STMT);
g = gimple_build_assign (lhs, BIT_AND_EXPR, arg0, temp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case P8V_BUILTIN_VEC_NAND:
case P8V_BUILTIN_NAND_V16QI:
case P8V_BUILTIN_NAND_V8HI:
case P8V_BUILTIN_NAND_V4SI:
case P8V_BUILTIN_NAND_V4SF:
case P8V_BUILTIN_NAND_V2DF:
case P8V_BUILTIN_NAND_V2DI:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
temp = create_tmp_reg_or_ssa_name (TREE_TYPE (arg1));
g = gimple_build_assign (temp, BIT_AND_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_insert_before (gsi, g, GSI_SAME_STMT);
g = gimple_build_assign (lhs, BIT_NOT_EXPR, temp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VOR:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, BIT_IOR_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case P8V_BUILTIN_ORC_V16QI:
case P8V_BUILTIN_ORC_V8HI:
case P8V_BUILTIN_ORC_V4SI:
case P8V_BUILTIN_ORC_V4SF:
case P8V_BUILTIN_ORC_V2DF:
case P8V_BUILTIN_ORC_V2DI:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
temp = create_tmp_reg_or_ssa_name (TREE_TYPE (arg1));
g = gimple_build_assign (temp, BIT_NOT_EXPR, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_insert_before (gsi, g, GSI_SAME_STMT);
g = gimple_build_assign (lhs, BIT_IOR_EXPR, arg0, temp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VXOR:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, BIT_XOR_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VNOR:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
temp = create_tmp_reg_or_ssa_name (TREE_TYPE (arg1));
g = gimple_build_assign (temp, BIT_IOR_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_insert_before (gsi, g, GSI_SAME_STMT);
g = gimple_build_assign (lhs, BIT_NOT_EXPR, temp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_ABS_V16QI:
case ALTIVEC_BUILTIN_ABS_V8HI:
case ALTIVEC_BUILTIN_ABS_V4SI:
case ALTIVEC_BUILTIN_ABS_V4SF:
case P8V_BUILTIN_ABS_V2DI:
case VSX_BUILTIN_XVABSDP:
arg0 = gimple_call_arg (stmt, 0);
if (INTEGRAL_TYPE_P (TREE_TYPE (TREE_TYPE (arg0)))
&& !TYPE_OVERFLOW_WRAPS (TREE_TYPE (TREE_TYPE (arg0))))
return false;
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, ABS_EXPR, arg0);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case VSX_BUILTIN_XVMINDP:
case P8V_BUILTIN_VMINSD:
case P8V_BUILTIN_VMINUD:
case ALTIVEC_BUILTIN_VMINSB:
case ALTIVEC_BUILTIN_VMINSH:
case ALTIVEC_BUILTIN_VMINSW:
case ALTIVEC_BUILTIN_VMINUB:
case ALTIVEC_BUILTIN_VMINUH:
case ALTIVEC_BUILTIN_VMINUW:
case ALTIVEC_BUILTIN_VMINFP:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, MIN_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case VSX_BUILTIN_XVMAXDP:
case P8V_BUILTIN_VMAXSD:
case P8V_BUILTIN_VMAXUD:
case ALTIVEC_BUILTIN_VMAXSB:
case ALTIVEC_BUILTIN_VMAXSH:
case ALTIVEC_BUILTIN_VMAXSW:
case ALTIVEC_BUILTIN_VMAXUB:
case ALTIVEC_BUILTIN_VMAXUH:
case ALTIVEC_BUILTIN_VMAXUW:
case ALTIVEC_BUILTIN_VMAXFP:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, MAX_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case P8V_BUILTIN_EQV_V16QI:
case P8V_BUILTIN_EQV_V8HI:
case P8V_BUILTIN_EQV_V4SI:
case P8V_BUILTIN_EQV_V4SF:
case P8V_BUILTIN_EQV_V2DF:
case P8V_BUILTIN_EQV_V2DI:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
temp = create_tmp_reg_or_ssa_name (TREE_TYPE (arg1));
g = gimple_build_assign (temp, BIT_XOR_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_insert_before (gsi, g, GSI_SAME_STMT);
g = gimple_build_assign (lhs, BIT_NOT_EXPR, temp);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VRLB:
case ALTIVEC_BUILTIN_VRLH:
case ALTIVEC_BUILTIN_VRLW:
case P8V_BUILTIN_VRLD:
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
g = gimple_build_assign (lhs, LROTATE_EXPR, arg0, arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
case ALTIVEC_BUILTIN_VSRAB:
case ALTIVEC_BUILTIN_VSRAH:
case ALTIVEC_BUILTIN_VSRAW:
case P8V_BUILTIN_VSRAD:
{
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
tree arg1_type = TREE_TYPE (arg1);
tree unsigned_arg1_type = unsigned_type_for (TREE_TYPE (arg1));
tree unsigned_element_type = unsigned_type_for (TREE_TYPE (arg1_type));
location_t loc = gimple_location (stmt);
int n_elts = VECTOR_CST_NELTS (arg1);
tree element_size = build_int_cst (unsigned_element_type,
128 / n_elts);
tree_vector_builder elts (unsigned_arg1_type, n_elts, 1);
for (int i = 0; i < n_elts; i++)
elts.safe_push (element_size);
tree modulo_tree = elts.build ();
gimple_seq stmts = NULL;
tree unsigned_arg1 = gimple_build (&stmts, VIEW_CONVERT_EXPR,
unsigned_arg1_type, arg1);
tree new_arg1 = gimple_build (&stmts, loc, TRUNC_MOD_EXPR,
unsigned_arg1_type, unsigned_arg1,
modulo_tree);
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
g = gimple_build_assign (lhs, RSHIFT_EXPR, arg0, new_arg1);
gimple_set_location (g, loc);
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_VSLB:
case ALTIVEC_BUILTIN_VSLH:
case ALTIVEC_BUILTIN_VSLW:
case P8V_BUILTIN_VSLD:
{
location_t loc;
gimple_seq stmts = NULL;
arg0 = gimple_call_arg (stmt, 0);
tree arg0_type = TREE_TYPE (arg0);
if (INTEGRAL_TYPE_P (TREE_TYPE (arg0_type))
&& !TYPE_OVERFLOW_WRAPS (TREE_TYPE (arg0_type)))
return false;
arg1 = gimple_call_arg (stmt, 1);
tree arg1_type = TREE_TYPE (arg1);
tree unsigned_arg1_type = unsigned_type_for (TREE_TYPE (arg1));
tree unsigned_element_type = unsigned_type_for (TREE_TYPE (arg1_type));
loc = gimple_location (stmt);
lhs = gimple_call_lhs (stmt);
int n_elts = VECTOR_CST_NELTS (arg1);
int tree_size_in_bits = TREE_INT_CST_LOW (size_in_bytes (arg1_type))
* BITS_PER_UNIT;
tree element_size = build_int_cst (unsigned_element_type,
tree_size_in_bits / n_elts);
tree_vector_builder elts (unsigned_type_for (arg1_type), n_elts, 1);
for (int i = 0; i < n_elts; i++)
elts.safe_push (element_size);
tree modulo_tree = elts.build ();
tree unsigned_arg1 = gimple_build (&stmts, VIEW_CONVERT_EXPR,
unsigned_arg1_type, arg1);
tree new_arg1 = gimple_build (&stmts, loc, TRUNC_MOD_EXPR,
unsigned_arg1_type, unsigned_arg1,
modulo_tree);
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
g = gimple_build_assign (lhs, LSHIFT_EXPR, arg0, new_arg1);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_VSRB:
case ALTIVEC_BUILTIN_VSRH:
case ALTIVEC_BUILTIN_VSRW:
case P8V_BUILTIN_VSRD:
{
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
lhs = gimple_call_lhs (stmt);
tree arg1_type = TREE_TYPE (arg1);
tree unsigned_arg1_type = unsigned_type_for (TREE_TYPE (arg1));
tree unsigned_element_type = unsigned_type_for (TREE_TYPE (arg1_type));
location_t loc = gimple_location (stmt);
gimple_seq stmts = NULL;
tree arg0_unsigned
= gimple_build (&stmts, VIEW_CONVERT_EXPR,
unsigned_type_for (TREE_TYPE (arg0)), arg0);
int n_elts = VECTOR_CST_NELTS (arg1);
tree element_size = build_int_cst (unsigned_element_type,
128 / n_elts);
tree_vector_builder elts (unsigned_arg1_type, n_elts, 1);
for (int i = 0; i < n_elts; i++)
elts.safe_push (element_size);
tree modulo_tree = elts.build ();
tree unsigned_arg1 = gimple_build (&stmts, VIEW_CONVERT_EXPR,
unsigned_arg1_type, arg1);
tree new_arg1 = gimple_build (&stmts, loc, TRUNC_MOD_EXPR,
unsigned_arg1_type, unsigned_arg1,
modulo_tree);
tree res
= gimple_build (&stmts, RSHIFT_EXPR,
TREE_TYPE (arg0_unsigned), arg0_unsigned, new_arg1);
res = gimple_build (&stmts, VIEW_CONVERT_EXPR, TREE_TYPE (lhs), res);
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
update_call_from_tree (gsi, res);
return true;
}
case ALTIVEC_BUILTIN_LVX_V16QI:
case ALTIVEC_BUILTIN_LVX_V8HI:
case ALTIVEC_BUILTIN_LVX_V4SI:
case ALTIVEC_BUILTIN_LVX_V4SF:
case ALTIVEC_BUILTIN_LVX_V2DI:
case ALTIVEC_BUILTIN_LVX_V2DF:
case ALTIVEC_BUILTIN_LVX_V1TI:
{
arg0 = gimple_call_arg (stmt, 0);  
arg1 = gimple_call_arg (stmt, 1);  
if (VECTOR_ELT_ORDER_BIG && !BYTES_BIG_ENDIAN)
return false;
lhs = gimple_call_lhs (stmt);
location_t loc = gimple_location (stmt);
tree arg1_type = ptr_type_node;
tree lhs_type = TREE_TYPE (lhs);
gimple_seq stmts = NULL;
tree temp_offset = gimple_convert (&stmts, loc, sizetype, arg0);
tree temp_addr = gimple_build (&stmts, loc, POINTER_PLUS_EXPR,
arg1_type, arg1, temp_offset);
tree aligned_addr = gimple_build (&stmts, loc, BIT_AND_EXPR,
arg1_type, temp_addr,
build_int_cst (arg1_type, -16));
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
if (!is_gimple_mem_ref_addr (aligned_addr))
{
tree t = make_ssa_name (TREE_TYPE (aligned_addr));
gimple *g = gimple_build_assign (t, aligned_addr);
gsi_insert_before (gsi, g, GSI_SAME_STMT);
aligned_addr = t;
}
gimple *g
= gimple_build_assign (lhs, build2 (MEM_REF, lhs_type, aligned_addr,
build_int_cst (arg1_type, 0)));
gimple_set_location (g, loc);
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_STVX_V16QI:
case ALTIVEC_BUILTIN_STVX_V8HI:
case ALTIVEC_BUILTIN_STVX_V4SI:
case ALTIVEC_BUILTIN_STVX_V4SF:
case ALTIVEC_BUILTIN_STVX_V2DI:
case ALTIVEC_BUILTIN_STVX_V2DF:
{
if (VECTOR_ELT_ORDER_BIG && !BYTES_BIG_ENDIAN)
return false;
arg0 = gimple_call_arg (stmt, 0); 
arg1 = gimple_call_arg (stmt, 1); 
tree arg2 = gimple_call_arg (stmt, 2); 
location_t loc = gimple_location (stmt);
tree arg0_type = TREE_TYPE (arg0);
tree arg2_type = ptr_type_node;
gimple_seq stmts = NULL;
tree temp_offset = gimple_convert (&stmts, loc, sizetype, arg1);
tree temp_addr = gimple_build (&stmts, loc, POINTER_PLUS_EXPR,
arg2_type, arg2, temp_offset);
tree aligned_addr = gimple_build (&stmts, loc, BIT_AND_EXPR,
arg2_type, temp_addr,
build_int_cst (arg2_type, -16));
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
if (!is_gimple_mem_ref_addr (aligned_addr))
{
tree t = make_ssa_name (TREE_TYPE (aligned_addr));
gimple *g = gimple_build_assign (t, aligned_addr);
gsi_insert_before (gsi, g, GSI_SAME_STMT);
aligned_addr = t;
}
gimple *g
= gimple_build_assign (build2 (MEM_REF, arg0_type, aligned_addr,
build_int_cst (arg2_type, 0)), arg0);
gimple_set_location (g, loc);
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_VMADDFP:
case VSX_BUILTIN_XVMADDDP:
case ALTIVEC_BUILTIN_VMLADDUHM:
{
arg0 = gimple_call_arg (stmt, 0);
arg1 = gimple_call_arg (stmt, 1);
tree arg2 = gimple_call_arg (stmt, 2);
lhs = gimple_call_lhs (stmt);
gimple *g = gimple_build_assign (lhs, FMA_EXPR, arg0, arg1, arg2);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_VCMPEQUB:
case ALTIVEC_BUILTIN_VCMPEQUH:
case ALTIVEC_BUILTIN_VCMPEQUW:
case P8V_BUILTIN_VCMPEQUD:
fold_compare_helper (gsi, EQ_EXPR, stmt);
return true;
case P9V_BUILTIN_CMPNEB:
case P9V_BUILTIN_CMPNEH:
case P9V_BUILTIN_CMPNEW:
fold_compare_helper (gsi, NE_EXPR, stmt);
return true;
case VSX_BUILTIN_CMPGE_16QI:
case VSX_BUILTIN_CMPGE_U16QI:
case VSX_BUILTIN_CMPGE_8HI:
case VSX_BUILTIN_CMPGE_U8HI:
case VSX_BUILTIN_CMPGE_4SI:
case VSX_BUILTIN_CMPGE_U4SI:
case VSX_BUILTIN_CMPGE_2DI:
case VSX_BUILTIN_CMPGE_U2DI:
fold_compare_helper (gsi, GE_EXPR, stmt);
return true;
case ALTIVEC_BUILTIN_VCMPGTSB:
case ALTIVEC_BUILTIN_VCMPGTUB:
case ALTIVEC_BUILTIN_VCMPGTSH:
case ALTIVEC_BUILTIN_VCMPGTUH:
case ALTIVEC_BUILTIN_VCMPGTSW:
case ALTIVEC_BUILTIN_VCMPGTUW:
case P8V_BUILTIN_VCMPGTUD:
case P8V_BUILTIN_VCMPGTSD:
fold_compare_helper (gsi, GT_EXPR, stmt);
return true;
case VSX_BUILTIN_CMPLE_16QI:
case VSX_BUILTIN_CMPLE_U16QI:
case VSX_BUILTIN_CMPLE_8HI:
case VSX_BUILTIN_CMPLE_U8HI:
case VSX_BUILTIN_CMPLE_4SI:
case VSX_BUILTIN_CMPLE_U4SI:
case VSX_BUILTIN_CMPLE_2DI:
case VSX_BUILTIN_CMPLE_U2DI:
fold_compare_helper (gsi, LE_EXPR, stmt);
return true;
case ALTIVEC_BUILTIN_VSPLTISB:
case ALTIVEC_BUILTIN_VSPLTISH:
case ALTIVEC_BUILTIN_VSPLTISW:
{
int size;
if (fn_code == ALTIVEC_BUILTIN_VSPLTISB)
size = 8;
else if (fn_code == ALTIVEC_BUILTIN_VSPLTISH)
size = 16;
else
size = 32;
arg0 = gimple_call_arg (stmt, 0);
lhs = gimple_call_lhs (stmt);
if (TREE_CODE (arg0) != INTEGER_CST
|| !IN_RANGE (sext_hwi(TREE_INT_CST_LOW (arg0), size),
-16, 15))
return false;
gimple_seq stmts = NULL;
location_t loc = gimple_location (stmt);
tree splat_value = gimple_convert (&stmts, loc,
TREE_TYPE (TREE_TYPE (lhs)), arg0);
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
tree splat_tree = build_vector_from_val (TREE_TYPE (lhs), splat_value);
g = gimple_build_assign (lhs, splat_tree);
gimple_set_location (g, gimple_location (stmt));
gsi_replace (gsi, g, true);
return true;
}
case ALTIVEC_BUILTIN_VMRGLH:
case ALTIVEC_BUILTIN_VMRGLW:
case VSX_BUILTIN_XXMRGLW_4SI:
case ALTIVEC_BUILTIN_VMRGLB:
case VSX_BUILTIN_VEC_MERGEL_V2DI:
if (VECTOR_ELT_ORDER_BIG && !BYTES_BIG_ENDIAN)
return false;
fold_mergehl_helper (gsi, stmt, 1);
return true;
case ALTIVEC_BUILTIN_VMRGHH:
case ALTIVEC_BUILTIN_VMRGHW:
case VSX_BUILTIN_XXMRGHW_4SI:
case ALTIVEC_BUILTIN_VMRGHB:
case VSX_BUILTIN_VEC_MERGEH_V2DI:
if (VECTOR_ELT_ORDER_BIG && !BYTES_BIG_ENDIAN)
return false;
fold_mergehl_helper (gsi, stmt, 0);
return true;
default:
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "gimple builtin intrinsic not matched:%d %s %s\n",
fn_code, fn_name1, fn_name2);
break;
}
return false;
}
static rtx
rs6000_expand_builtin (tree exp, rtx target, rtx subtarget ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
int ignore ATTRIBUTE_UNUSED)
{
tree fndecl = TREE_OPERAND (CALL_EXPR_FN (exp), 0);
enum rs6000_builtins fcode
= (enum rs6000_builtins)DECL_FUNCTION_CODE (fndecl);
size_t uns_fcode = (size_t)fcode;
const struct builtin_description *d;
size_t i;
rtx ret;
bool success;
HOST_WIDE_INT mask = rs6000_builtin_info[uns_fcode].mask;
bool func_valid_p = ((rs6000_builtin_mask & mask) == mask);
enum insn_code icode = rs6000_builtin_info[uns_fcode].icode;
#ifdef HAVE_AS_POWER9
if (FLOAT128_IEEE_P (TFmode))
switch (icode)
{
default:
break;
case CODE_FOR_sqrtkf2_odd:	icode = CODE_FOR_sqrttf2_odd;	break;
case CODE_FOR_trunckfdf2_odd:	icode = CODE_FOR_trunctfdf2_odd; break;
case CODE_FOR_addkf3_odd:		icode = CODE_FOR_addtf3_odd;	break;
case CODE_FOR_subkf3_odd:		icode = CODE_FOR_subtf3_odd;	break;
case CODE_FOR_mulkf3_odd:		icode = CODE_FOR_multf3_odd;	break;
case CODE_FOR_divkf3_odd:		icode = CODE_FOR_divtf3_odd;	break;
case CODE_FOR_fmakf4_odd:		icode = CODE_FOR_fmatf4_odd;	break;
case CODE_FOR_xsxexpqp_kf:	icode = CODE_FOR_xsxexpqp_tf;	break;
case CODE_FOR_xsxsigqp_kf:	icode = CODE_FOR_xsxsigqp_tf;	break;
case CODE_FOR_xststdcnegqp_kf:	icode = CODE_FOR_xststdcnegqp_tf; break;
case CODE_FOR_xsiexpqp_kf:	icode = CODE_FOR_xsiexpqp_tf;	break;
case CODE_FOR_xsiexpqpf_kf:	icode = CODE_FOR_xsiexpqpf_tf;	break;
case CODE_FOR_xststdcqp_kf:	icode = CODE_FOR_xststdcqp_tf;	break;
}
#endif
if (TARGET_DEBUG_BUILTIN)
{
const char *name1 = rs6000_builtin_info[uns_fcode].name;
const char *name2 = (icode != CODE_FOR_nothing)
? get_insn_name ((int) icode)
: "nothing";
const char *name3;
switch (rs6000_builtin_info[uns_fcode].attr & RS6000_BTC_TYPE_MASK)
{
default:		   name3 = "unknown";	break;
case RS6000_BTC_SPECIAL:   name3 = "special";	break;
case RS6000_BTC_UNARY:	   name3 = "unary";	break;
case RS6000_BTC_BINARY:	   name3 = "binary";	break;
case RS6000_BTC_TERNARY:   name3 = "ternary";	break;
case RS6000_BTC_PREDICATE: name3 = "predicate";	break;
case RS6000_BTC_ABS:	   name3 = "abs";	break;
case RS6000_BTC_DST:	   name3 = "dst";	break;
}
fprintf (stderr,
"rs6000_expand_builtin, %s (%d), insn = %s (%d), type=%s%s\n",
(name1) ? name1 : "---", fcode,
(name2) ? name2 : "---", (int) icode,
name3,
func_valid_p ? "" : ", not valid");
}	     
if (!func_valid_p)
{
rs6000_invalid_builtin (fcode);
return expand_call (exp, target, ignore);
}
switch (fcode)
{
case RS6000_BUILTIN_RECIP:
return rs6000_expand_binop_builtin (CODE_FOR_recipdf3, exp, target);
case RS6000_BUILTIN_RECIPF:
return rs6000_expand_binop_builtin (CODE_FOR_recipsf3, exp, target);
case RS6000_BUILTIN_RSQRTF:
return rs6000_expand_unop_builtin (CODE_FOR_rsqrtsf2, exp, target);
case RS6000_BUILTIN_RSQRT:
return rs6000_expand_unop_builtin (CODE_FOR_rsqrtdf2, exp, target);
case POWER7_BUILTIN_BPERMD:
return rs6000_expand_binop_builtin (((TARGET_64BIT)
? CODE_FOR_bpermd_di
: CODE_FOR_bpermd_si), exp, target);
case RS6000_BUILTIN_GET_TB:
return rs6000_expand_zeroop_builtin (CODE_FOR_rs6000_get_timebase,
target);
case RS6000_BUILTIN_MFTB:
return rs6000_expand_zeroop_builtin (((TARGET_64BIT)
? CODE_FOR_rs6000_mftb_di
: CODE_FOR_rs6000_mftb_si),
target);
case RS6000_BUILTIN_MFFS:
return rs6000_expand_zeroop_builtin (CODE_FOR_rs6000_mffs, target);
case RS6000_BUILTIN_MTFSF:
return rs6000_expand_mtfsf_builtin (CODE_FOR_rs6000_mtfsf, exp);
case RS6000_BUILTIN_CPU_INIT:
case RS6000_BUILTIN_CPU_IS:
case RS6000_BUILTIN_CPU_SUPPORTS:
return cpu_expand_builtin (fcode, exp, target);
case MISC_BUILTIN_SPEC_BARRIER:
{
emit_insn (gen_rs6000_speculation_barrier ());
return NULL_RTX;
}
case ALTIVEC_BUILTIN_MASK_FOR_LOAD:
case ALTIVEC_BUILTIN_MASK_FOR_STORE:
{
int icode2 = (BYTES_BIG_ENDIAN ? (int) CODE_FOR_altivec_lvsr_direct
: (int) CODE_FOR_altivec_lvsl_direct);
machine_mode tmode = insn_data[icode2].operand[0].mode;
machine_mode mode = insn_data[icode2].operand[1].mode;
tree arg;
rtx op, addr, pat;
gcc_assert (TARGET_ALTIVEC);
arg = CALL_EXPR_ARG (exp, 0);
gcc_assert (POINTER_TYPE_P (TREE_TYPE (arg)));
op = expand_expr (arg, NULL_RTX, Pmode, EXPAND_NORMAL);
addr = memory_address (mode, op);
if (fcode == ALTIVEC_BUILTIN_MASK_FOR_STORE)
op = addr;
else
{
op = gen_reg_rtx (GET_MODE (addr));
emit_insn (gen_rtx_SET (op, gen_rtx_NEG (GET_MODE (addr), addr)));
}
op = gen_rtx_MEM (mode, op);
if (target == 0
|| GET_MODE (target) != tmode
|| ! (*insn_data[icode2].operand[0].predicate) (target, tmode))
target = gen_reg_rtx (tmode);
pat = GEN_FCN (icode2) (target, op);
if (!pat)
return 0;
emit_insn (pat);
return target;
}
case ALTIVEC_BUILTIN_VCFUX:
case ALTIVEC_BUILTIN_VCFSX:
case ALTIVEC_BUILTIN_VCTUXS:
case ALTIVEC_BUILTIN_VCTSXS:
if (call_expr_nargs (exp) == 1)
{
exp = build_call_nary (TREE_TYPE (exp), CALL_EXPR_FN (exp),
2, CALL_EXPR_ARG (exp, 0), integer_zero_node);
}
break;
case MISC_BUILTIN_PACK_IF:
if (TARGET_LONG_DOUBLE_128 && !TARGET_IEEEQUAD)
{
icode = CODE_FOR_packtf;
fcode = MISC_BUILTIN_PACK_TF;
uns_fcode = (size_t)fcode;
}
break;
case MISC_BUILTIN_UNPACK_IF:
if (TARGET_LONG_DOUBLE_128 && !TARGET_IEEEQUAD)
{
icode = CODE_FOR_unpacktf;
fcode = MISC_BUILTIN_UNPACK_TF;
uns_fcode = (size_t)fcode;
}
break;
default:
break;
}
if (TARGET_ALTIVEC)
{
ret = altivec_expand_builtin (exp, target, &success);
if (success)
return ret;
}
if (TARGET_PAIRED_FLOAT)
{
ret = paired_expand_builtin (exp, target, &success);
if (success)
return ret;
}  
if (TARGET_HTM)
{
ret = htm_expand_builtin (exp, target, &success);
if (success)
return ret;
}  
unsigned attr = rs6000_builtin_info[uns_fcode].attr & RS6000_BTC_TYPE_MASK;
gcc_assert (attr == RS6000_BTC_UNARY
|| attr == RS6000_BTC_BINARY
|| attr == RS6000_BTC_TERNARY
|| attr == RS6000_BTC_SPECIAL);
d = bdesc_1arg;
for (i = 0; i < ARRAY_SIZE (bdesc_1arg); i++, d++)
if (d->code == fcode)
return rs6000_expand_unop_builtin (icode, exp, target);
d = bdesc_2arg;
for (i = 0; i < ARRAY_SIZE (bdesc_2arg); i++, d++)
if (d->code == fcode)
return rs6000_expand_binop_builtin (icode, exp, target);
d = bdesc_3arg;
for (i = 0; i < ARRAY_SIZE  (bdesc_3arg); i++, d++)
if (d->code == fcode)
return rs6000_expand_ternop_builtin (icode, exp, target);
d = bdesc_0arg;
for (i = 0; i < ARRAY_SIZE (bdesc_0arg); i++, d++)
if (d->code == fcode)
return rs6000_expand_zeroop_builtin (icode, target);
gcc_unreachable ();
}
static tree
rs6000_vector_type (const char *name, tree elt_type, unsigned num_elts)
{
tree result = build_vector_type (elt_type, num_elts);
result = build_variant_type_copy (result);
add_builtin_type (name, result);
return result;
}
static void
rs6000_init_builtins (void)
{
tree tdecl;
tree ftype;
machine_mode mode;
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_init_builtins%s%s%s\n",
(TARGET_PAIRED_FLOAT) ? ", paired"	 : "",
(TARGET_ALTIVEC)	   ? ", altivec" : "",
(TARGET_VSX)	   ? ", vsx"	 : "");
V2SI_type_node = build_vector_type (intSI_type_node, 2);
V2SF_type_node = build_vector_type (float_type_node, 2);
V2DI_type_node = rs6000_vector_type (TARGET_POWERPC64 ? "__vector long"
: "__vector long long",
intDI_type_node, 2);
V2DF_type_node = rs6000_vector_type ("__vector double", double_type_node, 2);
V4SI_type_node = rs6000_vector_type ("__vector signed int",
intSI_type_node, 4);
V4SF_type_node = rs6000_vector_type ("__vector float", float_type_node, 4);
V8HI_type_node = rs6000_vector_type ("__vector signed short",
intHI_type_node, 8);
V16QI_type_node = rs6000_vector_type ("__vector signed char",
intQI_type_node, 16);
unsigned_V16QI_type_node = rs6000_vector_type ("__vector unsigned char",
unsigned_intQI_type_node, 16);
unsigned_V8HI_type_node = rs6000_vector_type ("__vector unsigned short",
unsigned_intHI_type_node, 8);
unsigned_V4SI_type_node = rs6000_vector_type ("__vector unsigned int",
unsigned_intSI_type_node, 4);
unsigned_V2DI_type_node = rs6000_vector_type (TARGET_POWERPC64
? "__vector unsigned long"
: "__vector unsigned long long",
unsigned_intDI_type_node, 2);
opaque_V2SF_type_node = build_opaque_vector_type (float_type_node, 2);
opaque_V2SI_type_node = build_opaque_vector_type (intSI_type_node, 2);
opaque_p_V2SI_type_node = build_pointer_type (opaque_V2SI_type_node);
opaque_V4SI_type_node = build_opaque_vector_type (intSI_type_node, 4);
const_str_type_node
= build_pointer_type (build_qualified_type (char_type_node,
TYPE_QUAL_CONST));
if (intTI_type_node)
{
V1TI_type_node = rs6000_vector_type ("__vector __int128",
intTI_type_node, 1);
unsigned_V1TI_type_node
= rs6000_vector_type ("__vector unsigned __int128",
unsigned_intTI_type_node, 1);
}
bool_char_type_node = build_distinct_type_copy (unsigned_intQI_type_node);
bool_short_type_node = build_distinct_type_copy (unsigned_intHI_type_node);
bool_int_type_node = build_distinct_type_copy (unsigned_intSI_type_node);
bool_long_long_type_node = build_distinct_type_copy (unsigned_intDI_type_node);
pixel_type_node = build_distinct_type_copy (unsigned_intHI_type_node);
long_integer_type_internal_node = long_integer_type_node;
long_unsigned_type_internal_node = long_unsigned_type_node;
long_long_integer_type_internal_node = long_long_integer_type_node;
long_long_unsigned_type_internal_node = long_long_unsigned_type_node;
intQI_type_internal_node = intQI_type_node;
uintQI_type_internal_node = unsigned_intQI_type_node;
intHI_type_internal_node = intHI_type_node;
uintHI_type_internal_node = unsigned_intHI_type_node;
intSI_type_internal_node = intSI_type_node;
uintSI_type_internal_node = unsigned_intSI_type_node;
intDI_type_internal_node = intDI_type_node;
uintDI_type_internal_node = unsigned_intDI_type_node;
intTI_type_internal_node = intTI_type_node;
uintTI_type_internal_node = unsigned_intTI_type_node;
float_type_internal_node = float_type_node;
double_type_internal_node = double_type_node;
long_double_type_internal_node = long_double_type_node;
dfloat64_type_internal_node = dfloat64_type_node;
dfloat128_type_internal_node = dfloat128_type_node;
void_type_internal_node = void_type_node;
if (TARGET_FLOAT128_TYPE)
{
if (!TARGET_IEEEQUAD && TARGET_LONG_DOUBLE_128)
ibm128_float_type_node = long_double_type_node;
else
{
ibm128_float_type_node = make_node (REAL_TYPE);
TYPE_PRECISION (ibm128_float_type_node) = 128;
SET_TYPE_MODE (ibm128_float_type_node, IFmode);
layout_type (ibm128_float_type_node);
}
lang_hooks.types.register_builtin_type (ibm128_float_type_node,
"__ibm128");
if (TARGET_IEEEQUAD && TARGET_LONG_DOUBLE_128)
ieee128_float_type_node = long_double_type_node;
else
ieee128_float_type_node = float128_type_node;
lang_hooks.types.register_builtin_type (ieee128_float_type_node,
"__ieee128");
}
else
ieee128_float_type_node = ibm128_float_type_node = long_double_type_node;
builtin_mode_to_type[QImode][0] = integer_type_node;
builtin_mode_to_type[HImode][0] = integer_type_node;
builtin_mode_to_type[SImode][0] = intSI_type_node;
builtin_mode_to_type[SImode][1] = unsigned_intSI_type_node;
builtin_mode_to_type[DImode][0] = intDI_type_node;
builtin_mode_to_type[DImode][1] = unsigned_intDI_type_node;
builtin_mode_to_type[TImode][0] = intTI_type_node;
builtin_mode_to_type[TImode][1] = unsigned_intTI_type_node;
builtin_mode_to_type[SFmode][0] = float_type_node;
builtin_mode_to_type[DFmode][0] = double_type_node;
builtin_mode_to_type[IFmode][0] = ibm128_float_type_node;
builtin_mode_to_type[KFmode][0] = ieee128_float_type_node;
builtin_mode_to_type[TFmode][0] = long_double_type_node;
builtin_mode_to_type[DDmode][0] = dfloat64_type_node;
builtin_mode_to_type[TDmode][0] = dfloat128_type_node;
builtin_mode_to_type[V1TImode][0] = V1TI_type_node;
builtin_mode_to_type[V1TImode][1] = unsigned_V1TI_type_node;
builtin_mode_to_type[V2SImode][0] = V2SI_type_node;
builtin_mode_to_type[V2SFmode][0] = V2SF_type_node;
builtin_mode_to_type[V2DImode][0] = V2DI_type_node;
builtin_mode_to_type[V2DImode][1] = unsigned_V2DI_type_node;
builtin_mode_to_type[V2DFmode][0] = V2DF_type_node;
builtin_mode_to_type[V4SImode][0] = V4SI_type_node;
builtin_mode_to_type[V4SImode][1] = unsigned_V4SI_type_node;
builtin_mode_to_type[V4SFmode][0] = V4SF_type_node;
builtin_mode_to_type[V8HImode][0] = V8HI_type_node;
builtin_mode_to_type[V8HImode][1] = unsigned_V8HI_type_node;
builtin_mode_to_type[V16QImode][0] = V16QI_type_node;
builtin_mode_to_type[V16QImode][1] = unsigned_V16QI_type_node;
tdecl = add_builtin_type ("__bool char", bool_char_type_node);
TYPE_NAME (bool_char_type_node) = tdecl;
tdecl = add_builtin_type ("__bool short", bool_short_type_node);
TYPE_NAME (bool_short_type_node) = tdecl;
tdecl = add_builtin_type ("__bool int", bool_int_type_node);
TYPE_NAME (bool_int_type_node) = tdecl;
tdecl = add_builtin_type ("__pixel", pixel_type_node);
TYPE_NAME (pixel_type_node) = tdecl;
bool_V16QI_type_node = rs6000_vector_type ("__vector __bool char",
bool_char_type_node, 16);
bool_V8HI_type_node = rs6000_vector_type ("__vector __bool short",
bool_short_type_node, 8);
bool_V4SI_type_node = rs6000_vector_type ("__vector __bool int",
bool_int_type_node, 4);
bool_V2DI_type_node = rs6000_vector_type (TARGET_POWERPC64
? "__vector __bool long"
: "__vector __bool long long",
bool_long_long_type_node, 2);
pixel_V8HI_type_node = rs6000_vector_type ("__vector __pixel",
pixel_type_node, 8);
if (TARGET_PAIRED_FLOAT)
paired_init_builtins ();
if (TARGET_EXTRA_BUILTINS)
altivec_init_builtins ();
if (TARGET_HTM)
htm_init_builtins ();
if (TARGET_EXTRA_BUILTINS || TARGET_PAIRED_FLOAT)
rs6000_common_init_builtins ();
ftype = builtin_function_type (DFmode, DFmode, DFmode, VOIDmode,
RS6000_BUILTIN_RECIP, "__builtin_recipdiv");
def_builtin ("__builtin_recipdiv", ftype, RS6000_BUILTIN_RECIP);
ftype = builtin_function_type (SFmode, SFmode, SFmode, VOIDmode,
RS6000_BUILTIN_RECIPF, "__builtin_recipdivf");
def_builtin ("__builtin_recipdivf", ftype, RS6000_BUILTIN_RECIPF);
ftype = builtin_function_type (DFmode, DFmode, VOIDmode, VOIDmode,
RS6000_BUILTIN_RSQRT, "__builtin_rsqrt");
def_builtin ("__builtin_rsqrt", ftype, RS6000_BUILTIN_RSQRT);
ftype = builtin_function_type (SFmode, SFmode, VOIDmode, VOIDmode,
RS6000_BUILTIN_RSQRTF, "__builtin_rsqrtf");
def_builtin ("__builtin_rsqrtf", ftype, RS6000_BUILTIN_RSQRTF);
mode = (TARGET_64BIT) ? DImode : SImode;
ftype = builtin_function_type (mode, mode, mode, VOIDmode,
POWER7_BUILTIN_BPERMD, "__builtin_bpermd");
def_builtin ("__builtin_bpermd", ftype, POWER7_BUILTIN_BPERMD);
ftype = build_function_type_list (unsigned_intDI_type_node,
NULL_TREE);
def_builtin ("__builtin_ppc_get_timebase", ftype, RS6000_BUILTIN_GET_TB);
if (TARGET_64BIT)
ftype = build_function_type_list (unsigned_intDI_type_node,
NULL_TREE);
else
ftype = build_function_type_list (unsigned_intSI_type_node,
NULL_TREE);
def_builtin ("__builtin_ppc_mftb", ftype, RS6000_BUILTIN_MFTB);
ftype = build_function_type_list (double_type_node, NULL_TREE);
def_builtin ("__builtin_mffs", ftype, RS6000_BUILTIN_MFFS);
ftype = build_function_type_list (void_type_node,
intSI_type_node, double_type_node,
NULL_TREE);
def_builtin ("__builtin_mtfsf", ftype, RS6000_BUILTIN_MTFSF);
ftype = build_function_type_list (void_type_node, NULL_TREE);
def_builtin ("__builtin_cpu_init", ftype, RS6000_BUILTIN_CPU_INIT);
def_builtin ("__builtin_ppc_speculation_barrier", ftype,
MISC_BUILTIN_SPEC_BARRIER);
ftype = build_function_type_list (bool_int_type_node, const_ptr_type_node,
NULL_TREE);
def_builtin ("__builtin_cpu_is", ftype, RS6000_BUILTIN_CPU_IS);
def_builtin ("__builtin_cpu_supports", ftype, RS6000_BUILTIN_CPU_SUPPORTS);
if (TARGET_XCOFF &&
(tdecl = builtin_decl_explicit (BUILT_IN_CLOG)) != NULL_TREE)
set_user_assembler_name (tdecl, "__clog");
#ifdef SUBTARGET_INIT_BUILTINS
SUBTARGET_INIT_BUILTINS;
#endif
}
static tree
rs6000_builtin_decl (unsigned code, bool initialize_p ATTRIBUTE_UNUSED)
{
HOST_WIDE_INT fnmask;
if (code >= RS6000_BUILTIN_COUNT)
return error_mark_node;
fnmask = rs6000_builtin_info[code].mask;
if ((fnmask & rs6000_builtin_mask) != fnmask)
{
rs6000_invalid_builtin ((enum rs6000_builtins)code);
return error_mark_node;
}
return rs6000_builtin_decls[code];
}
static void
paired_init_builtins (void)
{
const struct builtin_description *d;
size_t i;
HOST_WIDE_INT builtin_mask = rs6000_builtin_mask;
tree int_ftype_int_v2sf_v2sf
= build_function_type_list (integer_type_node,
integer_type_node,
V2SF_type_node,
V2SF_type_node,
NULL_TREE);
tree pcfloat_type_node =
build_pointer_type (build_qualified_type
(float_type_node, TYPE_QUAL_CONST));
tree v2sf_ftype_long_pcfloat = build_function_type_list (V2SF_type_node,
long_integer_type_node,
pcfloat_type_node,
NULL_TREE);
tree void_ftype_v2sf_long_pcfloat =
build_function_type_list (void_type_node,
V2SF_type_node,
long_integer_type_node,
pcfloat_type_node,
NULL_TREE);
def_builtin ("__builtin_paired_lx", v2sf_ftype_long_pcfloat,
PAIRED_BUILTIN_LX);
def_builtin ("__builtin_paired_stx", void_ftype_v2sf_long_pcfloat,
PAIRED_BUILTIN_STX);
d = bdesc_paired_preds;
for (i = 0; i < ARRAY_SIZE (bdesc_paired_preds); ++i, d++)
{
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "paired_init_builtins, skip predicate %s\n",
d->name);
continue;
}
gcc_assert (d->icode != CODE_FOR_nothing);
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "paired pred #%d, insn = %s [%d], mode = %s\n",
(int)i, get_insn_name (d->icode), (int)d->icode,
GET_MODE_NAME (insn_data[d->icode].operand[1].mode));
switch (insn_data[d->icode].operand[1].mode)
{
case E_V2SFmode:
type = int_ftype_int_v2sf_v2sf;
break;
default:
gcc_unreachable ();
}
def_builtin (d->name, type, d->code);
}
}
static void
altivec_init_builtins (void)
{
const struct builtin_description *d;
size_t i;
tree ftype;
tree decl;
HOST_WIDE_INT builtin_mask = rs6000_builtin_mask;
tree pvoid_type_node = build_pointer_type (void_type_node);
tree pcvoid_type_node
= build_pointer_type (build_qualified_type (void_type_node,
TYPE_QUAL_CONST));
tree int_ftype_opaque
= build_function_type_list (integer_type_node,
opaque_V4SI_type_node, NULL_TREE);
tree opaque_ftype_opaque
= build_function_type_list (integer_type_node, NULL_TREE);
tree opaque_ftype_opaque_int
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node, integer_type_node, NULL_TREE);
tree opaque_ftype_opaque_opaque_int
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node, opaque_V4SI_type_node,
integer_type_node, NULL_TREE);
tree opaque_ftype_opaque_opaque_opaque
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node, opaque_V4SI_type_node,
opaque_V4SI_type_node, NULL_TREE);
tree opaque_ftype_opaque_opaque
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node, opaque_V4SI_type_node,
NULL_TREE);
tree int_ftype_int_opaque_opaque
= build_function_type_list (integer_type_node,
integer_type_node, opaque_V4SI_type_node,
opaque_V4SI_type_node, NULL_TREE);
tree int_ftype_int_v4si_v4si
= build_function_type_list (integer_type_node,
integer_type_node, V4SI_type_node,
V4SI_type_node, NULL_TREE);
tree int_ftype_int_v2di_v2di
= build_function_type_list (integer_type_node,
integer_type_node, V2DI_type_node,
V2DI_type_node, NULL_TREE);
tree void_ftype_v4si
= build_function_type_list (void_type_node, V4SI_type_node, NULL_TREE);
tree v8hi_ftype_void
= build_function_type_list (V8HI_type_node, NULL_TREE);
tree void_ftype_void
= build_function_type_list (void_type_node, NULL_TREE);
tree void_ftype_int
= build_function_type_list (void_type_node, integer_type_node, NULL_TREE);
tree opaque_ftype_long_pcvoid
= build_function_type_list (opaque_V4SI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v16qi_ftype_long_pcvoid
= build_function_type_list (V16QI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v8hi_ftype_long_pcvoid
= build_function_type_list (V8HI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v4si_ftype_long_pcvoid
= build_function_type_list (V4SI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v4sf_ftype_long_pcvoid
= build_function_type_list (V4SF_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v2df_ftype_long_pcvoid
= build_function_type_list (V2DF_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v2di_ftype_long_pcvoid
= build_function_type_list (V2DI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree v1ti_ftype_long_pcvoid
= build_function_type_list (V1TI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree void_ftype_opaque_long_pvoid
= build_function_type_list (void_type_node,
opaque_V4SI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v4si_long_pvoid
= build_function_type_list (void_type_node,
V4SI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v16qi_long_pvoid
= build_function_type_list (void_type_node,
V16QI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v16qi_pvoid_long
= build_function_type_list (void_type_node,
V16QI_type_node, pvoid_type_node,
long_integer_type_node, NULL_TREE);
tree void_ftype_v8hi_long_pvoid
= build_function_type_list (void_type_node,
V8HI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v4sf_long_pvoid
= build_function_type_list (void_type_node,
V4SF_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v2df_long_pvoid
= build_function_type_list (void_type_node,
V2DF_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v1ti_long_pvoid
= build_function_type_list (void_type_node,
V1TI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree void_ftype_v2di_long_pvoid
= build_function_type_list (void_type_node,
V2DI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
tree int_ftype_int_v8hi_v8hi
= build_function_type_list (integer_type_node,
integer_type_node, V8HI_type_node,
V8HI_type_node, NULL_TREE);
tree int_ftype_int_v16qi_v16qi
= build_function_type_list (integer_type_node,
integer_type_node, V16QI_type_node,
V16QI_type_node, NULL_TREE);
tree int_ftype_int_v4sf_v4sf
= build_function_type_list (integer_type_node,
integer_type_node, V4SF_type_node,
V4SF_type_node, NULL_TREE);
tree int_ftype_int_v2df_v2df
= build_function_type_list (integer_type_node,
integer_type_node, V2DF_type_node,
V2DF_type_node, NULL_TREE);
tree v2di_ftype_v2di
= build_function_type_list (V2DI_type_node, V2DI_type_node, NULL_TREE);
tree v4si_ftype_v4si
= build_function_type_list (V4SI_type_node, V4SI_type_node, NULL_TREE);
tree v8hi_ftype_v8hi
= build_function_type_list (V8HI_type_node, V8HI_type_node, NULL_TREE);
tree v16qi_ftype_v16qi
= build_function_type_list (V16QI_type_node, V16QI_type_node, NULL_TREE);
tree v4sf_ftype_v4sf
= build_function_type_list (V4SF_type_node, V4SF_type_node, NULL_TREE);
tree v2df_ftype_v2df
= build_function_type_list (V2DF_type_node, V2DF_type_node, NULL_TREE);
tree void_ftype_pcvoid_int_int
= build_function_type_list (void_type_node,
pcvoid_type_node, integer_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_altivec_mtvscr", void_ftype_v4si, ALTIVEC_BUILTIN_MTVSCR);
def_builtin ("__builtin_altivec_mfvscr", v8hi_ftype_void, ALTIVEC_BUILTIN_MFVSCR);
def_builtin ("__builtin_altivec_dssall", void_ftype_void, ALTIVEC_BUILTIN_DSSALL);
def_builtin ("__builtin_altivec_dss", void_ftype_int, ALTIVEC_BUILTIN_DSS);
def_builtin ("__builtin_altivec_lvsl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVSL);
def_builtin ("__builtin_altivec_lvsr", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVSR);
def_builtin ("__builtin_altivec_lvebx", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVEBX);
def_builtin ("__builtin_altivec_lvehx", v8hi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVEHX);
def_builtin ("__builtin_altivec_lvewx", v4si_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVEWX);
def_builtin ("__builtin_altivec_lvxl", v4si_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVXL);
def_builtin ("__builtin_altivec_lvxl_v2df", v2df_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V2DF);
def_builtin ("__builtin_altivec_lvxl_v2di", v2di_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V2DI);
def_builtin ("__builtin_altivec_lvxl_v4sf", v4sf_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V4SF);
def_builtin ("__builtin_altivec_lvxl_v4si", v4si_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V4SI);
def_builtin ("__builtin_altivec_lvxl_v8hi", v8hi_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V8HI);
def_builtin ("__builtin_altivec_lvxl_v16qi", v16qi_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVXL_V16QI);
def_builtin ("__builtin_altivec_lvx", v4si_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVX);
def_builtin ("__builtin_altivec_lvx_v1ti", v1ti_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V1TI);
def_builtin ("__builtin_altivec_lvx_v2df", v2df_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V2DF);
def_builtin ("__builtin_altivec_lvx_v2di", v2di_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V2DI);
def_builtin ("__builtin_altivec_lvx_v4sf", v4sf_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V4SF);
def_builtin ("__builtin_altivec_lvx_v4si", v4si_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V4SI);
def_builtin ("__builtin_altivec_lvx_v8hi", v8hi_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V8HI);
def_builtin ("__builtin_altivec_lvx_v16qi", v16qi_ftype_long_pcvoid,
ALTIVEC_BUILTIN_LVX_V16QI);
def_builtin ("__builtin_altivec_stvx", void_ftype_v4si_long_pvoid, ALTIVEC_BUILTIN_STVX);
def_builtin ("__builtin_altivec_stvx_v2df", void_ftype_v2df_long_pvoid,
ALTIVEC_BUILTIN_STVX_V2DF);
def_builtin ("__builtin_altivec_stvx_v2di", void_ftype_v2di_long_pvoid,
ALTIVEC_BUILTIN_STVX_V2DI);
def_builtin ("__builtin_altivec_stvx_v4sf", void_ftype_v4sf_long_pvoid,
ALTIVEC_BUILTIN_STVX_V4SF);
def_builtin ("__builtin_altivec_stvx_v4si", void_ftype_v4si_long_pvoid,
ALTIVEC_BUILTIN_STVX_V4SI);
def_builtin ("__builtin_altivec_stvx_v8hi", void_ftype_v8hi_long_pvoid,
ALTIVEC_BUILTIN_STVX_V8HI);
def_builtin ("__builtin_altivec_stvx_v16qi", void_ftype_v16qi_long_pvoid,
ALTIVEC_BUILTIN_STVX_V16QI);
def_builtin ("__builtin_altivec_stvewx", void_ftype_v4si_long_pvoid, ALTIVEC_BUILTIN_STVEWX);
def_builtin ("__builtin_altivec_stvxl", void_ftype_v4si_long_pvoid, ALTIVEC_BUILTIN_STVXL);
def_builtin ("__builtin_altivec_stvxl_v2df", void_ftype_v2df_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V2DF);
def_builtin ("__builtin_altivec_stvxl_v2di", void_ftype_v2di_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V2DI);
def_builtin ("__builtin_altivec_stvxl_v4sf", void_ftype_v4sf_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V4SF);
def_builtin ("__builtin_altivec_stvxl_v4si", void_ftype_v4si_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V4SI);
def_builtin ("__builtin_altivec_stvxl_v8hi", void_ftype_v8hi_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V8HI);
def_builtin ("__builtin_altivec_stvxl_v16qi", void_ftype_v16qi_long_pvoid,
ALTIVEC_BUILTIN_STVXL_V16QI);
def_builtin ("__builtin_altivec_stvebx", void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_STVEBX);
def_builtin ("__builtin_altivec_stvehx", void_ftype_v8hi_long_pvoid, ALTIVEC_BUILTIN_STVEHX);
def_builtin ("__builtin_vec_ld", opaque_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LD);
def_builtin ("__builtin_vec_lde", opaque_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LDE);
def_builtin ("__builtin_vec_ldl", opaque_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LDL);
def_builtin ("__builtin_vec_lvsl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVSL);
def_builtin ("__builtin_vec_lvsr", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVSR);
def_builtin ("__builtin_vec_lvebx", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVEBX);
def_builtin ("__builtin_vec_lvehx", v8hi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVEHX);
def_builtin ("__builtin_vec_lvewx", v4si_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVEWX);
def_builtin ("__builtin_vec_st", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_ST);
def_builtin ("__builtin_vec_ste", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_STE);
def_builtin ("__builtin_vec_stl", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_STL);
def_builtin ("__builtin_vec_stvewx", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_STVEWX);
def_builtin ("__builtin_vec_stvebx", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_STVEBX);
def_builtin ("__builtin_vec_stvehx", void_ftype_opaque_long_pvoid, ALTIVEC_BUILTIN_VEC_STVEHX);
def_builtin ("__builtin_vsx_lxvd2x_v2df", v2df_ftype_long_pcvoid,
VSX_BUILTIN_LXVD2X_V2DF);
def_builtin ("__builtin_vsx_lxvd2x_v2di", v2di_ftype_long_pcvoid,
VSX_BUILTIN_LXVD2X_V2DI);
def_builtin ("__builtin_vsx_lxvw4x_v4sf", v4sf_ftype_long_pcvoid,
VSX_BUILTIN_LXVW4X_V4SF);
def_builtin ("__builtin_vsx_lxvw4x_v4si", v4si_ftype_long_pcvoid,
VSX_BUILTIN_LXVW4X_V4SI);
def_builtin ("__builtin_vsx_lxvw4x_v8hi", v8hi_ftype_long_pcvoid,
VSX_BUILTIN_LXVW4X_V8HI);
def_builtin ("__builtin_vsx_lxvw4x_v16qi", v16qi_ftype_long_pcvoid,
VSX_BUILTIN_LXVW4X_V16QI);
def_builtin ("__builtin_vsx_stxvd2x_v2df", void_ftype_v2df_long_pvoid,
VSX_BUILTIN_STXVD2X_V2DF);
def_builtin ("__builtin_vsx_stxvd2x_v2di", void_ftype_v2di_long_pvoid,
VSX_BUILTIN_STXVD2X_V2DI);
def_builtin ("__builtin_vsx_stxvw4x_v4sf", void_ftype_v4sf_long_pvoid,
VSX_BUILTIN_STXVW4X_V4SF);
def_builtin ("__builtin_vsx_stxvw4x_v4si", void_ftype_v4si_long_pvoid,
VSX_BUILTIN_STXVW4X_V4SI);
def_builtin ("__builtin_vsx_stxvw4x_v8hi", void_ftype_v8hi_long_pvoid,
VSX_BUILTIN_STXVW4X_V8HI);
def_builtin ("__builtin_vsx_stxvw4x_v16qi", void_ftype_v16qi_long_pvoid,
VSX_BUILTIN_STXVW4X_V16QI);
def_builtin ("__builtin_vsx_ld_elemrev_v2df", v2df_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V2DF);
def_builtin ("__builtin_vsx_ld_elemrev_v2di", v2di_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V2DI);
def_builtin ("__builtin_vsx_ld_elemrev_v4sf", v4sf_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V4SF);
def_builtin ("__builtin_vsx_ld_elemrev_v4si", v4si_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V4SI);
def_builtin ("__builtin_vsx_ld_elemrev_v8hi", v8hi_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V8HI);
def_builtin ("__builtin_vsx_ld_elemrev_v16qi", v16qi_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V16QI);
def_builtin ("__builtin_vsx_st_elemrev_v2df", void_ftype_v2df_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V2DF);
def_builtin ("__builtin_vsx_st_elemrev_v1ti", void_ftype_v1ti_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V1TI);
def_builtin ("__builtin_vsx_st_elemrev_v2di", void_ftype_v2di_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V2DI);
def_builtin ("__builtin_vsx_st_elemrev_v4sf", void_ftype_v4sf_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V4SF);
def_builtin ("__builtin_vsx_st_elemrev_v4si", void_ftype_v4si_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V4SI);
def_builtin ("__builtin_vsx_st_elemrev_v8hi", void_ftype_v8hi_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V8HI);
def_builtin ("__builtin_vsx_st_elemrev_v16qi", void_ftype_v16qi_long_pvoid,
VSX_BUILTIN_ST_ELEMREV_V16QI);
def_builtin ("__builtin_vec_vsx_ld", opaque_ftype_long_pcvoid,
VSX_BUILTIN_VEC_LD);
def_builtin ("__builtin_vec_vsx_st", void_ftype_opaque_long_pvoid,
VSX_BUILTIN_VEC_ST);
def_builtin ("__builtin_vec_xl", opaque_ftype_long_pcvoid,
VSX_BUILTIN_VEC_XL);
def_builtin ("__builtin_vec_xl_be", opaque_ftype_long_pcvoid,
VSX_BUILTIN_VEC_XL_BE);
def_builtin ("__builtin_vec_xst", void_ftype_opaque_long_pvoid,
VSX_BUILTIN_VEC_XST);
def_builtin ("__builtin_vec_xst_be", void_ftype_opaque_long_pvoid,
VSX_BUILTIN_VEC_XST_BE);
def_builtin ("__builtin_vec_step", int_ftype_opaque, ALTIVEC_BUILTIN_VEC_STEP);
def_builtin ("__builtin_vec_splats", opaque_ftype_opaque, ALTIVEC_BUILTIN_VEC_SPLATS);
def_builtin ("__builtin_vec_promote", opaque_ftype_opaque, ALTIVEC_BUILTIN_VEC_PROMOTE);
def_builtin ("__builtin_vec_sld", opaque_ftype_opaque_opaque_int, ALTIVEC_BUILTIN_VEC_SLD);
def_builtin ("__builtin_vec_splat", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_SPLAT);
def_builtin ("__builtin_vec_extract", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_EXTRACT);
def_builtin ("__builtin_vec_insert", opaque_ftype_opaque_opaque_int, ALTIVEC_BUILTIN_VEC_INSERT);
def_builtin ("__builtin_vec_vspltw", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_VSPLTW);
def_builtin ("__builtin_vec_vsplth", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_VSPLTH);
def_builtin ("__builtin_vec_vspltb", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_VSPLTB);
def_builtin ("__builtin_vec_ctf", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_CTF);
def_builtin ("__builtin_vec_vcfsx", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_VCFSX);
def_builtin ("__builtin_vec_vcfux", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_VCFUX);
def_builtin ("__builtin_vec_cts", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_CTS);
def_builtin ("__builtin_vec_ctu", opaque_ftype_opaque_int, ALTIVEC_BUILTIN_VEC_CTU);
def_builtin ("__builtin_vec_adde", opaque_ftype_opaque_opaque_opaque,
ALTIVEC_BUILTIN_VEC_ADDE);
def_builtin ("__builtin_vec_addec", opaque_ftype_opaque_opaque_opaque,
ALTIVEC_BUILTIN_VEC_ADDEC);
def_builtin ("__builtin_vec_cmpne", opaque_ftype_opaque_opaque,
ALTIVEC_BUILTIN_VEC_CMPNE);
def_builtin ("__builtin_vec_mul", opaque_ftype_opaque_opaque,
ALTIVEC_BUILTIN_VEC_MUL);
def_builtin ("__builtin_vec_sube", opaque_ftype_opaque_opaque_opaque,
ALTIVEC_BUILTIN_VEC_SUBE);
def_builtin ("__builtin_vec_subec", opaque_ftype_opaque_opaque_opaque,
ALTIVEC_BUILTIN_VEC_SUBEC);
def_builtin ("__builtin_altivec_lvlx",  v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVLX);
def_builtin ("__builtin_altivec_lvlxl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVLXL);
def_builtin ("__builtin_altivec_lvrx",  v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVRX);
def_builtin ("__builtin_altivec_lvrxl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_LVRXL);
def_builtin ("__builtin_vec_lvlx",  v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVLX);
def_builtin ("__builtin_vec_lvlxl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVLXL);
def_builtin ("__builtin_vec_lvrx",  v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVRX);
def_builtin ("__builtin_vec_lvrxl", v16qi_ftype_long_pcvoid, ALTIVEC_BUILTIN_VEC_LVRXL);
def_builtin ("__builtin_altivec_stvlx",  void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_STVLX);
def_builtin ("__builtin_altivec_stvlxl", void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_STVLXL);
def_builtin ("__builtin_altivec_stvrx",  void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_STVRX);
def_builtin ("__builtin_altivec_stvrxl", void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_STVRXL);
def_builtin ("__builtin_vec_stvlx",  void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_VEC_STVLX);
def_builtin ("__builtin_vec_stvlxl", void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_VEC_STVLXL);
def_builtin ("__builtin_vec_stvrx",  void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_VEC_STVRX);
def_builtin ("__builtin_vec_stvrxl", void_ftype_v16qi_long_pvoid, ALTIVEC_BUILTIN_VEC_STVRXL);
if (TARGET_P9_VECTOR)
{
def_builtin ("__builtin_altivec_stxvl", void_ftype_v16qi_pvoid_long,
P9V_BUILTIN_STXVL);
def_builtin ("__builtin_xst_len_r", void_ftype_v16qi_pvoid_long,
P9V_BUILTIN_XST_LEN_R);
}
d = bdesc_dst;
for (i = 0; i < ARRAY_SIZE (bdesc_dst); i++, d++)
{
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "altivec_init_builtins, skip dst %s\n",
d->name);
continue;
}
def_builtin (d->name, void_ftype_pcvoid_int_int, d->code);
}
d = bdesc_altivec_preds;
for (i = 0; i < ARRAY_SIZE (bdesc_altivec_preds); i++, d++)
{
machine_mode mode1;
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "altivec_init_builtins, skip predicate %s\n",
d->name);
continue;
}
if (rs6000_overloaded_builtin_p (d->code))
mode1 = VOIDmode;
else
{
gcc_assert (d->icode != CODE_FOR_nothing);
mode1 = insn_data[d->icode].operand[1].mode;
}
switch (mode1)
{
case E_VOIDmode:
type = int_ftype_int_opaque_opaque;
break;
case E_V2DImode:
type = int_ftype_int_v2di_v2di;
break;
case E_V4SImode:
type = int_ftype_int_v4si_v4si;
break;
case E_V8HImode:
type = int_ftype_int_v8hi_v8hi;
break;
case E_V16QImode:
type = int_ftype_int_v16qi_v16qi;
break;
case E_V4SFmode:
type = int_ftype_int_v4sf_v4sf;
break;
case E_V2DFmode:
type = int_ftype_int_v2df_v2df;
break;
default:
gcc_unreachable ();
}
def_builtin (d->name, type, d->code);
}
d = bdesc_abs;
for (i = 0; i < ARRAY_SIZE (bdesc_abs); i++, d++)
{
machine_mode mode0;
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "altivec_init_builtins, skip abs %s\n",
d->name);
continue;
}
gcc_assert (d->icode != CODE_FOR_nothing);
mode0 = insn_data[d->icode].operand[0].mode;
switch (mode0)
{
case E_V2DImode:
type = v2di_ftype_v2di;
break;
case E_V4SImode:
type = v4si_ftype_v4si;
break;
case E_V8HImode:
type = v8hi_ftype_v8hi;
break;
case E_V16QImode:
type = v16qi_ftype_v16qi;
break;
case E_V4SFmode:
type = v4sf_ftype_v4sf;
break;
case E_V2DFmode:
type = v2df_ftype_v2df;
break;
default:
gcc_unreachable ();
}
def_builtin (d->name, type, d->code);
}
decl = add_builtin_function ("__builtin_altivec_mask_for_load",
v16qi_ftype_long_pcvoid,
ALTIVEC_BUILTIN_MASK_FOR_LOAD,
BUILT_IN_MD, NULL, NULL_TREE);
TREE_READONLY (decl) = 1;
altivec_builtin_mask_for_load = decl;
ftype = build_function_type_list (V4SI_type_node, integer_type_node,
integer_type_node, integer_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v4si", ftype, ALTIVEC_BUILTIN_VEC_INIT_V4SI);
ftype = build_function_type_list (V8HI_type_node, short_integer_type_node,
short_integer_type_node,
short_integer_type_node,
short_integer_type_node,
short_integer_type_node,
short_integer_type_node,
short_integer_type_node,
short_integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v8hi", ftype, ALTIVEC_BUILTIN_VEC_INIT_V8HI);
ftype = build_function_type_list (V16QI_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, char_type_node,
char_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v16qi", ftype,
ALTIVEC_BUILTIN_VEC_INIT_V16QI);
ftype = build_function_type_list (V4SF_type_node, float_type_node,
float_type_node, float_type_node,
float_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v4sf", ftype, ALTIVEC_BUILTIN_VEC_INIT_V4SF);
ftype = build_function_type_list (V2DF_type_node, double_type_node,
double_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v2df", ftype, VSX_BUILTIN_VEC_INIT_V2DF);
ftype = build_function_type_list (V2DI_type_node, intDI_type_node,
intDI_type_node, NULL_TREE);
def_builtin ("__builtin_vec_init_v2di", ftype, VSX_BUILTIN_VEC_INIT_V2DI);
ftype = build_function_type_list (V4SI_type_node, V4SI_type_node,
intSI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v4si", ftype, ALTIVEC_BUILTIN_VEC_SET_V4SI);
ftype = build_function_type_list (V8HI_type_node, V8HI_type_node,
intHI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v8hi", ftype, ALTIVEC_BUILTIN_VEC_SET_V8HI);
ftype = build_function_type_list (V16QI_type_node, V16QI_type_node,
intQI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v16qi", ftype, ALTIVEC_BUILTIN_VEC_SET_V16QI);
ftype = build_function_type_list (V4SF_type_node, V4SF_type_node,
float_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v4sf", ftype, ALTIVEC_BUILTIN_VEC_SET_V4SF);
ftype = build_function_type_list (V2DF_type_node, V2DF_type_node,
double_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v2df", ftype, VSX_BUILTIN_VEC_SET_V2DF);
ftype = build_function_type_list (V2DI_type_node, V2DI_type_node,
intDI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v2di", ftype, VSX_BUILTIN_VEC_SET_V2DI);
ftype = build_function_type_list (intSI_type_node, V4SI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v4si", ftype, ALTIVEC_BUILTIN_VEC_EXT_V4SI);
ftype = build_function_type_list (intHI_type_node, V8HI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v8hi", ftype, ALTIVEC_BUILTIN_VEC_EXT_V8HI);
ftype = build_function_type_list (intQI_type_node, V16QI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v16qi", ftype, ALTIVEC_BUILTIN_VEC_EXT_V16QI);
ftype = build_function_type_list (float_type_node, V4SF_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v4sf", ftype, ALTIVEC_BUILTIN_VEC_EXT_V4SF);
ftype = build_function_type_list (double_type_node, V2DF_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v2df", ftype, VSX_BUILTIN_VEC_EXT_V2DF);
ftype = build_function_type_list (intDI_type_node, V2DI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v2di", ftype, VSX_BUILTIN_VEC_EXT_V2DI);
if (V1TI_type_node)
{
tree v1ti_ftype_long_pcvoid
= build_function_type_list (V1TI_type_node,
long_integer_type_node, pcvoid_type_node,
NULL_TREE);
tree void_ftype_v1ti_long_pvoid
= build_function_type_list (void_type_node,
V1TI_type_node, long_integer_type_node,
pvoid_type_node, NULL_TREE);
def_builtin ("__builtin_vsx_ld_elemrev_v1ti", v1ti_ftype_long_pcvoid,
VSX_BUILTIN_LD_ELEMREV_V1TI);
def_builtin ("__builtin_vsx_lxvd2x_v1ti", v1ti_ftype_long_pcvoid,
VSX_BUILTIN_LXVD2X_V1TI);
def_builtin ("__builtin_vsx_stxvd2x_v1ti", void_ftype_v1ti_long_pvoid,
VSX_BUILTIN_STXVD2X_V1TI);
ftype = build_function_type_list (V1TI_type_node, intTI_type_node,
NULL_TREE, NULL_TREE);
def_builtin ("__builtin_vec_init_v1ti", ftype, VSX_BUILTIN_VEC_INIT_V1TI);
ftype = build_function_type_list (V1TI_type_node, V1TI_type_node,
intTI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_set_v1ti", ftype, VSX_BUILTIN_VEC_SET_V1TI);
ftype = build_function_type_list (intTI_type_node, V1TI_type_node,
integer_type_node, NULL_TREE);
def_builtin ("__builtin_vec_ext_v1ti", ftype, VSX_BUILTIN_VEC_EXT_V1TI);
}
}
static void
htm_init_builtins (void)
{
HOST_WIDE_INT builtin_mask = rs6000_builtin_mask;
const struct builtin_description *d;
size_t i;
d = bdesc_htm;
for (i = 0; i < ARRAY_SIZE (bdesc_htm); i++, d++)
{
tree op[MAX_HTM_OPERANDS], type;
HOST_WIDE_INT mask = d->mask;
unsigned attr = rs6000_builtin_info[d->code].attr;
bool void_func = (attr & RS6000_BTC_VOID);
int attr_args = (attr & RS6000_BTC_TYPE_MASK);
int nopnds = 0;
tree gpr_type_node;
tree rettype;
tree argtype;
if (TARGET_32BIT && TARGET_POWERPC64)
gpr_type_node = long_long_unsigned_type_node;
else
gpr_type_node = long_unsigned_type_node;
if (attr & RS6000_BTC_SPR)
{
rettype = gpr_type_node;
argtype = gpr_type_node;
}
else if (d->code == HTM_BUILTIN_TABORTDC
|| d->code == HTM_BUILTIN_TABORTDCI)
{
rettype = unsigned_type_node;
argtype = gpr_type_node;
}
else
{
rettype = unsigned_type_node;
argtype = unsigned_type_node;
}
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "htm_builtin, skip binary %s\n", d->name);
continue;
}
if (d->name == 0)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "htm_builtin, bdesc_htm[%ld] no name\n",
(long unsigned) i);
continue;
}
op[nopnds++] = (void_func) ? void_type_node : rettype;
if (attr_args == RS6000_BTC_UNARY)
op[nopnds++] = argtype;
else if (attr_args == RS6000_BTC_BINARY)
{
op[nopnds++] = argtype;
op[nopnds++] = argtype;
}
else if (attr_args == RS6000_BTC_TERNARY)
{
op[nopnds++] = argtype;
op[nopnds++] = argtype;
op[nopnds++] = argtype;
}
switch (nopnds)
{
case 1:
type = build_function_type_list (op[0], NULL_TREE);
break;
case 2:
type = build_function_type_list (op[0], op[1], NULL_TREE);
break;
case 3:
type = build_function_type_list (op[0], op[1], op[2], NULL_TREE);
break;
case 4:
type = build_function_type_list (op[0], op[1], op[2], op[3],
NULL_TREE);
break;
default:
gcc_unreachable ();
}
def_builtin (d->name, type, d->code);
}
}
hashval_t
builtin_hasher::hash (builtin_hash_struct *bh)
{
unsigned ret = 0;
int i;
for (i = 0; i < 4; i++)
{
ret = (ret * (unsigned)MAX_MACHINE_MODE) + ((unsigned)bh->mode[i]);
ret = (ret * 2) + bh->uns_p[i];
}
return ret;
}
bool
builtin_hasher::equal (builtin_hash_struct *p1, builtin_hash_struct *p2)
{
return ((p1->mode[0] == p2->mode[0])
&& (p1->mode[1] == p2->mode[1])
&& (p1->mode[2] == p2->mode[2])
&& (p1->mode[3] == p2->mode[3])
&& (p1->uns_p[0] == p2->uns_p[0])
&& (p1->uns_p[1] == p2->uns_p[1])
&& (p1->uns_p[2] == p2->uns_p[2])
&& (p1->uns_p[3] == p2->uns_p[3]));
}
static tree
builtin_function_type (machine_mode mode_ret, machine_mode mode_arg0,
machine_mode mode_arg1, machine_mode mode_arg2,
enum rs6000_builtins builtin, const char *name)
{
struct builtin_hash_struct h;
struct builtin_hash_struct *h2;
int num_args = 3;
int i;
tree ret_type = NULL_TREE;
tree arg_type[3] = { NULL_TREE, NULL_TREE, NULL_TREE };
if (builtin_hash_table == NULL)
builtin_hash_table = hash_table<builtin_hasher>::create_ggc (1500);
h.type = NULL_TREE;
h.mode[0] = mode_ret;
h.mode[1] = mode_arg0;
h.mode[2] = mode_arg1;
h.mode[3] = mode_arg2;
h.uns_p[0] = 0;
h.uns_p[1] = 0;
h.uns_p[2] = 0;
h.uns_p[3] = 0;
switch (builtin)
{
case CRYPTO_BUILTIN_VSBOX:
case P8V_BUILTIN_VGBBD:
case MISC_BUILTIN_CDTBCD:
case MISC_BUILTIN_CBCDTD:
h.uns_p[0] = 1;
h.uns_p[1] = 1;
break;
case ALTIVEC_BUILTIN_VMULEUB:
case ALTIVEC_BUILTIN_VMULEUH:
case P8V_BUILTIN_VMULEUW:
case ALTIVEC_BUILTIN_VMULOUB:
case ALTIVEC_BUILTIN_VMULOUH:
case P8V_BUILTIN_VMULOUW:
case CRYPTO_BUILTIN_VCIPHER:
case CRYPTO_BUILTIN_VCIPHERLAST:
case CRYPTO_BUILTIN_VNCIPHER:
case CRYPTO_BUILTIN_VNCIPHERLAST:
case CRYPTO_BUILTIN_VPMSUMB:
case CRYPTO_BUILTIN_VPMSUMH:
case CRYPTO_BUILTIN_VPMSUMW:
case CRYPTO_BUILTIN_VPMSUMD:
case CRYPTO_BUILTIN_VPMSUM:
case MISC_BUILTIN_ADDG6S:
case MISC_BUILTIN_DIVWEU:
case MISC_BUILTIN_DIVDEU:
case VSX_BUILTIN_UDIV_V2DI:
case ALTIVEC_BUILTIN_VMAXUB:
case ALTIVEC_BUILTIN_VMINUB:
case ALTIVEC_BUILTIN_VMAXUH:
case ALTIVEC_BUILTIN_VMINUH:
case ALTIVEC_BUILTIN_VMAXUW:
case ALTIVEC_BUILTIN_VMINUW:
case P8V_BUILTIN_VMAXUD:
case P8V_BUILTIN_VMINUD:
h.uns_p[0] = 1;
h.uns_p[1] = 1;
h.uns_p[2] = 1;
break;
case ALTIVEC_BUILTIN_VPERM_16QI_UNS:
case ALTIVEC_BUILTIN_VPERM_8HI_UNS:
case ALTIVEC_BUILTIN_VPERM_4SI_UNS:
case ALTIVEC_BUILTIN_VPERM_2DI_UNS:
case ALTIVEC_BUILTIN_VSEL_16QI_UNS:
case ALTIVEC_BUILTIN_VSEL_8HI_UNS:
case ALTIVEC_BUILTIN_VSEL_4SI_UNS:
case ALTIVEC_BUILTIN_VSEL_2DI_UNS:
case VSX_BUILTIN_VPERM_16QI_UNS:
case VSX_BUILTIN_VPERM_8HI_UNS:
case VSX_BUILTIN_VPERM_4SI_UNS:
case VSX_BUILTIN_VPERM_2DI_UNS:
case VSX_BUILTIN_XXSEL_16QI_UNS:
case VSX_BUILTIN_XXSEL_8HI_UNS:
case VSX_BUILTIN_XXSEL_4SI_UNS:
case VSX_BUILTIN_XXSEL_2DI_UNS:
case CRYPTO_BUILTIN_VPERMXOR:
case CRYPTO_BUILTIN_VPERMXOR_V2DI:
case CRYPTO_BUILTIN_VPERMXOR_V4SI:
case CRYPTO_BUILTIN_VPERMXOR_V8HI:
case CRYPTO_BUILTIN_VPERMXOR_V16QI:
case CRYPTO_BUILTIN_VSHASIGMAW:
case CRYPTO_BUILTIN_VSHASIGMAD:
case CRYPTO_BUILTIN_VSHASIGMA:
h.uns_p[0] = 1;
h.uns_p[1] = 1;
h.uns_p[2] = 1;
h.uns_p[3] = 1;
break;
case ALTIVEC_BUILTIN_VPERM_16QI:
case ALTIVEC_BUILTIN_VPERM_8HI:
case ALTIVEC_BUILTIN_VPERM_4SI:
case ALTIVEC_BUILTIN_VPERM_4SF:
case ALTIVEC_BUILTIN_VPERM_2DI:
case ALTIVEC_BUILTIN_VPERM_2DF:
case VSX_BUILTIN_VPERM_16QI:
case VSX_BUILTIN_VPERM_8HI:
case VSX_BUILTIN_VPERM_4SI:
case VSX_BUILTIN_VPERM_4SF:
case VSX_BUILTIN_VPERM_2DI:
case VSX_BUILTIN_VPERM_2DF:
h.uns_p[3] = 1;
break;
case VSX_BUILTIN_XVCVUXDSP:
case VSX_BUILTIN_XVCVUXDDP_UNS:
case ALTIVEC_BUILTIN_UNSFLOAT_V4SI_V4SF:
h.uns_p[1] = 1;
break;
case VSX_BUILTIN_XVCVDPUXDS_UNS:
case ALTIVEC_BUILTIN_FIXUNS_V4SF_V4SI:
case MISC_BUILTIN_UNPACK_TD:
case MISC_BUILTIN_UNPACK_V1TI:
h.uns_p[0] = 1;
break;
case ALTIVEC_BUILTIN_VCMPEQUB:
case ALTIVEC_BUILTIN_VCMPEQUH:
case ALTIVEC_BUILTIN_VCMPEQUW:
case P8V_BUILTIN_VCMPEQUD:
case VSX_BUILTIN_CMPGE_U16QI:
case VSX_BUILTIN_CMPGE_U8HI:
case VSX_BUILTIN_CMPGE_U4SI:
case VSX_BUILTIN_CMPGE_U2DI:
case ALTIVEC_BUILTIN_VCMPGTUB:
case ALTIVEC_BUILTIN_VCMPGTUH:
case ALTIVEC_BUILTIN_VCMPGTUW:
case P8V_BUILTIN_VCMPGTUD:
h.uns_p[1] = 1;
h.uns_p[2] = 1;
break;
case MISC_BUILTIN_PACK_TD:
case MISC_BUILTIN_PACK_V1TI:
h.uns_p[1] = 1;
h.uns_p[2] = 1;
break;
case ALTIVEC_BUILTIN_VSRB:
case ALTIVEC_BUILTIN_VSRH:
case ALTIVEC_BUILTIN_VSRW:
case P8V_BUILTIN_VSRD:
h.uns_p[2] = 1;
break;
default:
break;
}
while (num_args > 0 && h.mode[num_args] == VOIDmode)
num_args--;
ret_type = builtin_mode_to_type[h.mode[0]][h.uns_p[0]];
if (!ret_type && h.uns_p[0])
ret_type = builtin_mode_to_type[h.mode[0]][0];
if (!ret_type)
fatal_error (input_location,
"internal error: builtin function %qs had an unexpected "
"return type %qs", name, GET_MODE_NAME (h.mode[0]));
for (i = 0; i < (int) ARRAY_SIZE (arg_type); i++)
arg_type[i] = NULL_TREE;
for (i = 0; i < num_args; i++)
{
int m = (int) h.mode[i+1];
int uns_p = h.uns_p[i+1];
arg_type[i] = builtin_mode_to_type[m][uns_p];
if (!arg_type[i] && uns_p)
arg_type[i] = builtin_mode_to_type[m][0];
if (!arg_type[i])
fatal_error (input_location,
"internal error: builtin function %qs, argument %d "
"had unexpected argument type %qs", name, i,
GET_MODE_NAME (m));
}
builtin_hash_struct **found = builtin_hash_table->find_slot (&h, INSERT);
if (*found == NULL)
{
h2 = ggc_alloc<builtin_hash_struct> ();
*h2 = h;
*found = h2;
h2->type = build_function_type_list (ret_type, arg_type[0], arg_type[1],
arg_type[2], NULL_TREE);
}
return (*found)->type;
}
static void
rs6000_common_init_builtins (void)
{
const struct builtin_description *d;
size_t i;
tree opaque_ftype_opaque = NULL_TREE;
tree opaque_ftype_opaque_opaque = NULL_TREE;
tree opaque_ftype_opaque_opaque_opaque = NULL_TREE;
tree v2si_ftype = NULL_TREE;
tree v2si_ftype_qi = NULL_TREE;
tree v2si_ftype_v2si_qi = NULL_TREE;
tree v2si_ftype_int_qi = NULL_TREE;
HOST_WIDE_INT builtin_mask = rs6000_builtin_mask;
if (!TARGET_PAIRED_FLOAT)
{
builtin_mode_to_type[V2SImode][0] = opaque_V2SI_type_node;
builtin_mode_to_type[V2SFmode][0] = opaque_V2SF_type_node;
}
if (TARGET_EXTRA_BUILTINS)
builtin_mask |= RS6000_BTM_COMMON;
d = bdesc_3arg;
for (i = 0; i < ARRAY_SIZE (bdesc_3arg); i++, d++)
{
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip ternary %s\n", d->name);
continue;
}
if (rs6000_overloaded_builtin_p (d->code))
{
if (! (type = opaque_ftype_opaque_opaque_opaque))
type = opaque_ftype_opaque_opaque_opaque
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node,
opaque_V4SI_type_node,
opaque_V4SI_type_node,
NULL_TREE);
}
else
{
enum insn_code icode = d->icode;
if (d->name == 0)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, bdesc_3arg[%ld] no name\n",
(long unsigned)i);
continue;
}
if (icode == CODE_FOR_nothing)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip ternary %s (no code)\n",
d->name);
continue;
}
type = builtin_function_type (insn_data[icode].operand[0].mode,
insn_data[icode].operand[1].mode,
insn_data[icode].operand[2].mode,
insn_data[icode].operand[3].mode,
d->code, d->name);
}
def_builtin (d->name, type, d->code);
}
d = bdesc_2arg;
for (i = 0; i < ARRAY_SIZE (bdesc_2arg); i++, d++)
{
machine_mode mode0, mode1, mode2;
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip binary %s\n", d->name);
continue;
}
if (rs6000_overloaded_builtin_p (d->code))
{
if (! (type = opaque_ftype_opaque_opaque))
type = opaque_ftype_opaque_opaque
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node,
opaque_V4SI_type_node,
NULL_TREE);
}
else
{
enum insn_code icode = d->icode;
if (d->name == 0)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, bdesc_2arg[%ld] no name\n",
(long unsigned)i);
continue;
}
if (icode == CODE_FOR_nothing)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip binary %s (no code)\n",
d->name);
continue;
}
mode0 = insn_data[icode].operand[0].mode;
mode1 = insn_data[icode].operand[1].mode;
mode2 = insn_data[icode].operand[2].mode;
if (mode0 == V2SImode && mode1 == V2SImode && mode2 == QImode)
{
if (! (type = v2si_ftype_v2si_qi))
type = v2si_ftype_v2si_qi
= build_function_type_list (opaque_V2SI_type_node,
opaque_V2SI_type_node,
char_type_node,
NULL_TREE);
}
else if (mode0 == V2SImode && GET_MODE_CLASS (mode1) == MODE_INT
&& mode2 == QImode)
{
if (! (type = v2si_ftype_int_qi))
type = v2si_ftype_int_qi
= build_function_type_list (opaque_V2SI_type_node,
integer_type_node,
char_type_node,
NULL_TREE);
}
else
type = builtin_function_type (mode0, mode1, mode2, VOIDmode,
d->code, d->name);
}
def_builtin (d->name, type, d->code);
}
d = bdesc_1arg;
for (i = 0; i < ARRAY_SIZE (bdesc_1arg); i++, d++)
{
machine_mode mode0, mode1;
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip unary %s\n", d->name);
continue;
}
if (rs6000_overloaded_builtin_p (d->code))
{
if (! (type = opaque_ftype_opaque))
type = opaque_ftype_opaque
= build_function_type_list (opaque_V4SI_type_node,
opaque_V4SI_type_node,
NULL_TREE);
}
else
{
enum insn_code icode = d->icode;
if (d->name == 0)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, bdesc_1arg[%ld] no name\n",
(long unsigned)i);
continue;
}
if (icode == CODE_FOR_nothing)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip unary %s (no code)\n",
d->name);
continue;
}
mode0 = insn_data[icode].operand[0].mode;
mode1 = insn_data[icode].operand[1].mode;
if (mode0 == V2SImode && mode1 == QImode)
{
if (! (type = v2si_ftype_qi))
type = v2si_ftype_qi
= build_function_type_list (opaque_V2SI_type_node,
char_type_node,
NULL_TREE);
}
else
type = builtin_function_type (mode0, mode1, VOIDmode, VOIDmode,
d->code, d->name);
}
def_builtin (d->name, type, d->code);
}
d = bdesc_0arg;
for (i = 0; i < ARRAY_SIZE (bdesc_0arg); i++, d++)
{
machine_mode mode0;
tree type;
HOST_WIDE_INT mask = d->mask;
if ((mask & builtin_mask) != mask)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, skip no-argument %s\n", d->name);
continue;
}
if (rs6000_overloaded_builtin_p (d->code))
{
if (!opaque_ftype_opaque)
opaque_ftype_opaque
= build_function_type_list (opaque_V4SI_type_node, NULL_TREE);
type = opaque_ftype_opaque;
}
else
{
enum insn_code icode = d->icode;
if (d->name == 0)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "rs6000_builtin, bdesc_0arg[%lu] no name\n",
(long unsigned) i);
continue;
}
if (icode == CODE_FOR_nothing)
{
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr,
"rs6000_builtin, skip no-argument %s (no code)\n",
d->name);
continue;
}
mode0 = insn_data[icode].operand[0].mode;
if (mode0 == V2SImode)
{
if (! (type = v2si_ftype))
{
v2si_ftype
= build_function_type_list (opaque_V2SI_type_node, 
NULL_TREE);
type = v2si_ftype;
}
}
else
type = builtin_function_type (mode0, VOIDmode, VOIDmode, VOIDmode,
d->code, d->name);
}
def_builtin (d->name, type, d->code);
}
}
static void
init_float128_ibm (machine_mode mode)
{
if (!TARGET_XL_COMPAT)
{
set_optab_libfunc (add_optab, mode, "__gcc_qadd");
set_optab_libfunc (sub_optab, mode, "__gcc_qsub");
set_optab_libfunc (smul_optab, mode, "__gcc_qmul");
set_optab_libfunc (sdiv_optab, mode, "__gcc_qdiv");
if (!TARGET_HARD_FLOAT)
{
set_optab_libfunc (neg_optab, mode, "__gcc_qneg");
set_optab_libfunc (eq_optab, mode, "__gcc_qeq");
set_optab_libfunc (ne_optab, mode, "__gcc_qne");
set_optab_libfunc (gt_optab, mode, "__gcc_qgt");
set_optab_libfunc (ge_optab, mode, "__gcc_qge");
set_optab_libfunc (lt_optab, mode, "__gcc_qlt");
set_optab_libfunc (le_optab, mode, "__gcc_qle");
set_optab_libfunc (unord_optab, mode, "__gcc_qunord");
set_conv_libfunc (sext_optab, mode, SFmode, "__gcc_stoq");
set_conv_libfunc (sext_optab, mode, DFmode, "__gcc_dtoq");
set_conv_libfunc (trunc_optab, SFmode, mode, "__gcc_qtos");
set_conv_libfunc (trunc_optab, DFmode, mode, "__gcc_qtod");
set_conv_libfunc (sfix_optab, SImode, mode, "__gcc_qtoi");
set_conv_libfunc (ufix_optab, SImode, mode, "__gcc_qtou");
set_conv_libfunc (sfloat_optab, mode, SImode, "__gcc_itoq");
set_conv_libfunc (ufloat_optab, mode, SImode, "__gcc_utoq");
}
}
else
{
set_optab_libfunc (add_optab, mode, "_xlqadd");
set_optab_libfunc (sub_optab, mode, "_xlqsub");
set_optab_libfunc (smul_optab, mode, "_xlqmul");
set_optab_libfunc (sdiv_optab, mode, "_xlqdiv");
}
if (mode == IFmode)
{
set_conv_libfunc (sext_optab, mode, SDmode, "__dpd_extendsdtf2");
set_conv_libfunc (sext_optab, mode, DDmode, "__dpd_extendddtf2");
set_conv_libfunc (trunc_optab, mode, TDmode, "__dpd_trunctftd2");
set_conv_libfunc (trunc_optab, SDmode, mode, "__dpd_trunctfsd2");
set_conv_libfunc (trunc_optab, DDmode, mode, "__dpd_trunctfdd2");
set_conv_libfunc (sext_optab, TDmode, mode, "__dpd_extendtdtf2");
if (TARGET_POWERPC64)
{
set_conv_libfunc (sfix_optab, TImode, mode, "__fixtfti");
set_conv_libfunc (ufix_optab, TImode, mode, "__fixunstfti");
set_conv_libfunc (sfloat_optab, mode, TImode, "__floattitf");
set_conv_libfunc (ufloat_optab, mode, TImode, "__floatuntitf");
}
}
}
static void
create_complex_muldiv (const char *name, built_in_function fncode, tree fntype)
{
tree fndecl = add_builtin_function (name, fntype, fncode, BUILT_IN_NORMAL,
name, NULL_TREE);
set_builtin_decl (fncode, fndecl, true);
if (TARGET_DEBUG_BUILTIN)
fprintf (stderr, "create complex %s, fncode: %d\n", name, (int) fncode);
return;
}
static void
init_float128_ieee (machine_mode mode)
{
if (FLOAT128_VECTOR_P (mode))
{
static bool complex_muldiv_init_p = false;
if (mode == TFmode && TARGET_IEEEQUAD && !complex_muldiv_init_p)
{
complex_muldiv_init_p = true;
built_in_function fncode_mul =
(built_in_function) (BUILT_IN_COMPLEX_MUL_MIN + TCmode
- MIN_MODE_COMPLEX_FLOAT);
built_in_function fncode_div =
(built_in_function) (BUILT_IN_COMPLEX_DIV_MIN + TCmode
- MIN_MODE_COMPLEX_FLOAT);
tree fntype = build_function_type_list (complex_long_double_type_node,
long_double_type_node,
long_double_type_node,
long_double_type_node,
long_double_type_node,
NULL_TREE);
create_complex_muldiv ("__mulkc3", fncode_mul, fntype);
create_complex_muldiv ("__divkc3", fncode_div, fntype);
}
set_optab_libfunc (add_optab, mode, "__addkf3");
set_optab_libfunc (sub_optab, mode, "__subkf3");
set_optab_libfunc (neg_optab, mode, "__negkf2");
set_optab_libfunc (smul_optab, mode, "__mulkf3");
set_optab_libfunc (sdiv_optab, mode, "__divkf3");
set_optab_libfunc (sqrt_optab, mode, "__sqrtkf2");
set_optab_libfunc (abs_optab, mode, "__abskf2");
set_optab_libfunc (powi_optab, mode, "__powikf2");
set_optab_libfunc (eq_optab, mode, "__eqkf2");
set_optab_libfunc (ne_optab, mode, "__nekf2");
set_optab_libfunc (gt_optab, mode, "__gtkf2");
set_optab_libfunc (ge_optab, mode, "__gekf2");
set_optab_libfunc (lt_optab, mode, "__ltkf2");
set_optab_libfunc (le_optab, mode, "__lekf2");
set_optab_libfunc (unord_optab, mode, "__unordkf2");
set_conv_libfunc (sext_optab, mode, SFmode, "__extendsfkf2");
set_conv_libfunc (sext_optab, mode, DFmode, "__extenddfkf2");
set_conv_libfunc (trunc_optab, SFmode, mode, "__trunckfsf2");
set_conv_libfunc (trunc_optab, DFmode, mode, "__trunckfdf2");
set_conv_libfunc (sext_optab, mode, IFmode, "__trunctfkf2");
if (mode != TFmode && FLOAT128_IBM_P (TFmode))
set_conv_libfunc (sext_optab, mode, TFmode, "__trunctfkf2");
set_conv_libfunc (trunc_optab, IFmode, mode, "__extendkftf2");
if (mode != TFmode && FLOAT128_IBM_P (TFmode))
set_conv_libfunc (trunc_optab, TFmode, mode, "__extendkftf2");
set_conv_libfunc (sext_optab, mode, SDmode, "__dpd_extendsdkf2");
set_conv_libfunc (sext_optab, mode, DDmode, "__dpd_extendddkf2");
set_conv_libfunc (trunc_optab, mode, TDmode, "__dpd_trunckftd2");
set_conv_libfunc (trunc_optab, SDmode, mode, "__dpd_trunckfsd2");
set_conv_libfunc (trunc_optab, DDmode, mode, "__dpd_trunckfdd2");
set_conv_libfunc (sext_optab, TDmode, mode, "__dpd_extendtdkf2");
set_conv_libfunc (sfix_optab, SImode, mode, "__fixkfsi");
set_conv_libfunc (ufix_optab, SImode, mode, "__fixunskfsi");
set_conv_libfunc (sfix_optab, DImode, mode, "__fixkfdi");
set_conv_libfunc (ufix_optab, DImode, mode, "__fixunskfdi");
set_conv_libfunc (sfloat_optab, mode, SImode, "__floatsikf");
set_conv_libfunc (ufloat_optab, mode, SImode, "__floatunsikf");
set_conv_libfunc (sfloat_optab, mode, DImode, "__floatdikf");
set_conv_libfunc (ufloat_optab, mode, DImode, "__floatundikf");
if (TARGET_POWERPC64)
{
set_conv_libfunc (sfix_optab, TImode, mode, "__fixkfti");
set_conv_libfunc (ufix_optab, TImode, mode, "__fixunskfti");
set_conv_libfunc (sfloat_optab, mode, TImode, "__floattikf");
set_conv_libfunc (ufloat_optab, mode, TImode, "__floatuntikf");
}
}
else
{
set_optab_libfunc (add_optab, mode, "_q_add");
set_optab_libfunc (sub_optab, mode, "_q_sub");
set_optab_libfunc (neg_optab, mode, "_q_neg");
set_optab_libfunc (smul_optab, mode, "_q_mul");
set_optab_libfunc (sdiv_optab, mode, "_q_div");
if (TARGET_PPC_GPOPT)
set_optab_libfunc (sqrt_optab, mode, "_q_sqrt");
set_optab_libfunc (eq_optab, mode, "_q_feq");
set_optab_libfunc (ne_optab, mode, "_q_fne");
set_optab_libfunc (gt_optab, mode, "_q_fgt");
set_optab_libfunc (ge_optab, mode, "_q_fge");
set_optab_libfunc (lt_optab, mode, "_q_flt");
set_optab_libfunc (le_optab, mode, "_q_fle");
set_conv_libfunc (sext_optab, mode, SFmode, "_q_stoq");
set_conv_libfunc (sext_optab, mode, DFmode, "_q_dtoq");
set_conv_libfunc (trunc_optab, SFmode, mode, "_q_qtos");
set_conv_libfunc (trunc_optab, DFmode, mode, "_q_qtod");
set_conv_libfunc (sfix_optab, SImode, mode, "_q_qtoi");
set_conv_libfunc (ufix_optab, SImode, mode, "_q_qtou");
set_conv_libfunc (sfloat_optab, mode, SImode, "_q_itoq");
set_conv_libfunc (ufloat_optab, mode, SImode, "_q_utoq");
}
}
static void
rs6000_init_libfuncs (void)
{
if (TARGET_FLOAT128_TYPE)
{
init_float128_ibm (IFmode);
init_float128_ieee (KFmode);
}
if (TARGET_LONG_DOUBLE_128)
{
if (!TARGET_IEEEQUAD)
init_float128_ibm (TFmode);
else
init_float128_ieee (TFmode);
}
}
void
rs6000_emit_dot_insn (rtx dst, rtx src, int dot, rtx ccreg)
{
if (dot == 0)
{
emit_move_insn (dst, src);
return;
}
if (cc_reg_not_cr0_operand (ccreg, CCmode))
{
emit_move_insn (dst, src);
emit_move_insn (ccreg, gen_rtx_COMPARE (CCmode, dst, const0_rtx));
return;
}
rtx ccset = gen_rtx_SET (ccreg, gen_rtx_COMPARE (CCmode, src, const0_rtx));
if (dot == 1)
{
rtx clobber = gen_rtx_CLOBBER (VOIDmode, dst);
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, ccset, clobber)));
}
else
{
rtx set = gen_rtx_SET (dst, src);
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, ccset, set)));
}
}

void
validate_condition_mode (enum rtx_code code, machine_mode mode)
{
gcc_assert ((GET_RTX_CLASS (code) == RTX_COMPARE
|| GET_RTX_CLASS (code) == RTX_COMM_COMPARE)
&& GET_MODE_CLASS (mode) == MODE_CC);
gcc_assert ((code != GT && code != LT && code != GE && code != LE)
|| mode != CCUNSmode);
gcc_assert ((code != GTU && code != LTU && code != GEU && code != LEU)
|| mode == CCUNSmode);
gcc_assert (mode == CCFPmode
|| (code != ORDERED && code != UNORDERED
&& code != UNEQ && code != LTGT
&& code != UNGT && code != UNLT
&& code != UNGE && code != UNLE));
gcc_assert (mode != CCFPmode
|| flag_finite_math_only
|| (code != LE && code != GE
&& code != UNEQ && code != LTGT
&& code != UNGT && code != UNLT));
gcc_assert (mode != CCEQmode || code == EQ || code == NE);
}

bool
rs6000_is_valid_mask (rtx mask, int *b, int *e, machine_mode mode)
{
unsigned HOST_WIDE_INT val = INTVAL (mask);
unsigned HOST_WIDE_INT bit;
int nb, ne;
int n = GET_MODE_PRECISION (mode);
if (mode != DImode && mode != SImode)
return false;
if (INTVAL (mask) >= 0)
{
bit = val & -val;
ne = exact_log2 (bit);
nb = exact_log2 (val + bit);
}
else if (val + 1 == 0)
{
nb = n;
ne = 0;
}
else if (val & 1)
{
val = ~val;
bit = val & -val;
nb = exact_log2 (bit);
ne = exact_log2 (val + bit);
}
else
{
bit = val & -val;
ne = exact_log2 (bit);
if (val + bit == 0)
nb = n;
else
nb = 0;
}
nb--;
if (nb < 0 || ne < 0 || nb >= n || ne >= n)
return false;
if (b)
*b = nb;
if (e)
*e = ne;
return true;
}
bool
rs6000_is_valid_and_mask (rtx mask, machine_mode mode)
{
int nb, ne;
if (!rs6000_is_valid_mask (mask, &nb, &ne, mode))
return false;
if (mode == DImode)
return (ne == 0 || nb == 63 || (nb < 32 && ne <= nb));
if (mode == SImode)
return (nb < 32 && ne < 32);
return false;
}
const char *
rs6000_insn_for_and_mask (machine_mode mode, rtx *operands, bool dot)
{
int nb, ne;
if (!rs6000_is_valid_mask (operands[2], &nb, &ne, mode))
gcc_unreachable ();
if (mode == DImode && ne == 0)
{
operands[3] = GEN_INT (63 - nb);
if (dot)
return "rldicl. %0,%1,0,%3";
return "rldicl %0,%1,0,%3";
}
if (mode == DImode && nb == 63)
{
operands[3] = GEN_INT (63 - ne);
if (dot)
return "rldicr. %0,%1,0,%3";
return "rldicr %0,%1,0,%3";
}
if (nb < 32 && ne < 32)
{
operands[3] = GEN_INT (31 - nb);
operands[4] = GEN_INT (31 - ne);
if (dot)
return "rlwinm. %0,%1,0,%3,%4";
return "rlwinm %0,%1,0,%3,%4";
}
gcc_unreachable ();
}
bool
rs6000_is_valid_shift_mask (rtx mask, rtx shift, machine_mode mode)
{
int nb, ne;
if (!rs6000_is_valid_mask (mask, &nb, &ne, mode))
return false;
int n = GET_MODE_PRECISION (mode);
int sh = -1;
if (CONST_INT_P (XEXP (shift, 1)))
{
sh = INTVAL (XEXP (shift, 1));
if (sh < 0 || sh >= n)
return false;
}
rtx_code code = GET_CODE (shift);
if (sh == 0)
code = ROTATE;
if (code == ROTATE && sh >= 0 && nb >= ne && ne >= sh)
code = ASHIFT;
if (code == ROTATE && sh >= 0 && nb >= ne && nb < sh)
{
code = LSHIFTRT;
sh = n - sh;
}
if (mode == DImode && code == ROTATE)
return (nb == 63 || ne == 0 || ne == sh);
if (mode == SImode && code == ROTATE)
return (nb < 32 && ne < 32 && sh < 32);
if (ne > nb)
return false;
if (sh < 0)
return false;
if (code == ASHIFT && ne < sh)
return false;
if (nb < 32 && ne < 32 && sh < 32
&& !(code == LSHIFTRT && nb >= 32 - sh))
return true;
if (code == LSHIFTRT)
sh = 64 - sh;
if (nb == 63 || ne == 0 || ne == sh)
return !(code == LSHIFTRT && nb >= sh);
return false;
}
const char *
rs6000_insn_for_shift_mask (machine_mode mode, rtx *operands, bool dot)
{
int nb, ne;
if (!rs6000_is_valid_mask (operands[3], &nb, &ne, mode))
gcc_unreachable ();
if (mode == DImode && ne == 0)
{
if (GET_CODE (operands[4]) == LSHIFTRT && INTVAL (operands[2]))
operands[2] = GEN_INT (64 - INTVAL (operands[2]));
operands[3] = GEN_INT (63 - nb);
if (dot)
return "rld%I2cl. %0,%1,%2,%3";
return "rld%I2cl %0,%1,%2,%3";
}
if (mode == DImode && nb == 63)
{
operands[3] = GEN_INT (63 - ne);
if (dot)
return "rld%I2cr. %0,%1,%2,%3";
return "rld%I2cr %0,%1,%2,%3";
}
if (mode == DImode
&& GET_CODE (operands[4]) != LSHIFTRT
&& CONST_INT_P (operands[2])
&& ne == INTVAL (operands[2]))
{
operands[3] = GEN_INT (63 - nb);
if (dot)
return "rld%I2c. %0,%1,%2,%3";
return "rld%I2c %0,%1,%2,%3";
}
if (nb < 32 && ne < 32)
{
if (GET_CODE (operands[4]) == LSHIFTRT && INTVAL (operands[2]))
operands[2] = GEN_INT (32 - INTVAL (operands[2]));
operands[3] = GEN_INT (31 - nb);
operands[4] = GEN_INT (31 - ne);
if (dot)
return "rlw%I2nm. %0,%1,%h2,%3,%4";
return "rlw%I2nm %0,%1,%h2,%3,%4";
}
gcc_unreachable ();
}
bool
rs6000_is_valid_insert_mask (rtx mask, rtx shift, machine_mode mode)
{
int nb, ne;
if (!rs6000_is_valid_mask (mask, &nb, &ne, mode))
return false;
int n = GET_MODE_PRECISION (mode);
int sh = INTVAL (XEXP (shift, 1));
if (sh < 0 || sh >= n)
return false;
rtx_code code = GET_CODE (shift);
if (sh == 0)
code = ROTATE;
if (code == ROTATE && sh >= 0 && nb >= ne && ne >= sh)
code = ASHIFT;
if (code == ROTATE && sh >= 0 && nb >= ne && nb < sh)
{
code = LSHIFTRT;
sh = n - sh;
}
if (mode == DImode && code == ROTATE)
return (ne == sh);
if (mode == SImode && code == ROTATE)
return (nb < 32 && ne < 32 && sh < 32);
if (ne > nb)
return false;
if (code == ASHIFT && ne < sh)
return false;
if (nb < 32 && ne < 32 && sh < 32
&& !(code == LSHIFTRT && nb >= 32 - sh))
return true;
if (code == LSHIFTRT)
sh = 64 - sh;
if (ne == sh)
return !(code == LSHIFTRT && nb >= sh);
return false;
}
const char *
rs6000_insn_for_insert_mask (machine_mode mode, rtx *operands, bool dot)
{
int nb, ne;
if (!rs6000_is_valid_mask (operands[3], &nb, &ne, mode))
gcc_unreachable ();
if (TARGET_POWERPC64
&& (!dot || mode == DImode)
&& GET_CODE (operands[4]) != LSHIFTRT
&& ne == INTVAL (operands[2]))
{
operands[3] = GEN_INT (63 - nb);
if (dot)
return "rldimi. %0,%1,%2,%3";
return "rldimi %0,%1,%2,%3";
}
if (nb < 32 && ne < 32)
{
if (GET_CODE (operands[4]) == LSHIFTRT && INTVAL (operands[2]))
operands[2] = GEN_INT (32 - INTVAL (operands[2]));
operands[3] = GEN_INT (31 - nb);
operands[4] = GEN_INT (31 - ne);
if (dot)
return "rlwimi. %0,%1,%2,%3,%4";
return "rlwimi %0,%1,%2,%3,%4";
}
gcc_unreachable ();
}
bool
rs6000_is_valid_2insn_and (rtx c, machine_mode mode)
{
if (rs6000_is_valid_mask (c, NULL, NULL, mode))
return true;
unsigned HOST_WIDE_INT val = INTVAL (c);
unsigned HOST_WIDE_INT bit1 = val & -val;
unsigned HOST_WIDE_INT bit2 = (val + bit1) & ~val;
unsigned HOST_WIDE_INT val1 = (val + bit1) & val;
unsigned HOST_WIDE_INT bit3 = val1 & -val1;
return rs6000_is_valid_and_mask (GEN_INT (val + bit3 - bit2), mode);
}
void
rs6000_emit_2insn_and (machine_mode mode, rtx *operands, bool expand, int dot)
{
gcc_assert (!(expand && dot));
unsigned HOST_WIDE_INT val = INTVAL (operands[2]);
int nb, ne;
if (rs6000_is_valid_mask (operands[2], &nb, &ne, mode) && nb >= ne)
{
gcc_assert (mode == DImode);
int shift = 63 - nb;
if (expand)
{
rtx tmp1 = gen_reg_rtx (DImode);
rtx tmp2 = gen_reg_rtx (DImode);
emit_insn (gen_ashldi3 (tmp1, operands[1], GEN_INT (shift)));
emit_insn (gen_anddi3 (tmp2, tmp1, GEN_INT (val << shift)));
emit_insn (gen_lshrdi3 (operands[0], tmp2, GEN_INT (shift)));
}
else
{
rtx tmp = gen_rtx_ASHIFT (mode, operands[1], GEN_INT (shift));
tmp = gen_rtx_AND (mode, tmp, GEN_INT (val << shift));
emit_move_insn (operands[0], tmp);
tmp = gen_rtx_LSHIFTRT (mode, operands[0], GEN_INT (shift));
rs6000_emit_dot_insn (operands[0], tmp, dot, dot ? operands[3] : 0);
}
return;
}
unsigned HOST_WIDE_INT bit1 = val & -val;
unsigned HOST_WIDE_INT bit2 = (val + bit1) & ~val;
unsigned HOST_WIDE_INT val1 = (val + bit1) & val;
unsigned HOST_WIDE_INT bit3 = val1 & -val1;
unsigned HOST_WIDE_INT mask1 = -bit3 + bit2 - 1;
unsigned HOST_WIDE_INT mask2 = val + bit3 - bit2;
gcc_assert (rs6000_is_valid_and_mask (GEN_INT (mask2), mode));
if (rs6000_is_valid_and_mask (GEN_INT (mask1), mode))
{
gcc_assert (mode == SImode);
rtx reg = expand ? gen_reg_rtx (mode) : operands[0];
rtx tmp = gen_rtx_AND (mode, operands[1], GEN_INT (mask1));
emit_move_insn (reg, tmp);
tmp = gen_rtx_AND (mode, reg, GEN_INT (mask2));
rs6000_emit_dot_insn (operands[0], tmp, dot, dot ? operands[3] : 0);
return;
}
gcc_assert (mode == DImode);
if (mask2 <= 0xffffffff
&& rs6000_is_valid_and_mask (GEN_INT (mask1), SImode))
{
rtx reg = expand ? gen_reg_rtx (mode) : operands[0];
rtx tmp = gen_rtx_AND (SImode, gen_lowpart (SImode, operands[1]),
GEN_INT (mask1));
rtx reg_low = gen_lowpart (SImode, reg);
emit_move_insn (reg_low, tmp);
tmp = gen_rtx_AND (mode, reg, GEN_INT (mask2));
rs6000_emit_dot_insn (operands[0], tmp, dot, dot ? operands[3] : 0);
return;
}
int right = exact_log2 (bit3);
int left = 64 - right;
mask1 = (mask1 >> right) | ((bit2 - 1) << left);
if (expand)
{
rtx tmp1 = gen_reg_rtx (DImode);
rtx tmp2 = gen_reg_rtx (DImode);
rtx tmp3 = gen_reg_rtx (DImode);
emit_insn (gen_rotldi3 (tmp1, operands[1], GEN_INT (left)));
emit_insn (gen_anddi3 (tmp2, tmp1, GEN_INT (mask1)));
emit_insn (gen_rotldi3 (tmp3, tmp2, GEN_INT (right)));
emit_insn (gen_anddi3 (operands[0], tmp3, GEN_INT (mask2)));
}
else
{
rtx tmp = gen_rtx_ROTATE (mode, operands[1], GEN_INT (left));
tmp = gen_rtx_AND (mode, tmp, GEN_INT (mask1));
emit_move_insn (operands[0], tmp);
tmp = gen_rtx_ROTATE (mode, operands[0], GEN_INT (right));
tmp = gen_rtx_AND (mode, tmp, GEN_INT (mask2));
rs6000_emit_dot_insn (operands[0], tmp, dot, dot ? operands[3] : 0);
}
}

int
registers_ok_for_quad_peep (rtx reg1, rtx reg2)
{
if (GET_CODE (reg1) != REG || GET_CODE (reg2) != REG)
return 0;
if (!FP_REGNO_P (REGNO (reg1))
|| !FP_REGNO_P (REGNO (reg2)))
return 0;
return (REGNO (reg1) == REGNO (reg2) - 1);
}
int
mems_ok_for_quad_peep (rtx mem1, rtx mem2)
{
rtx addr1, addr2;
unsigned int reg1, reg2;
int offset1, offset2;
if (MEM_VOLATILE_P (mem1) || MEM_VOLATILE_P (mem2))
return 0;
addr1 = XEXP (mem1, 0);
addr2 = XEXP (mem2, 0);
if (GET_CODE (addr1) == PLUS)
{
if (GET_CODE (XEXP (addr1, 0)) != REG)
return 0;
else
{
reg1 = REGNO (XEXP (addr1, 0));
if (GET_CODE (XEXP (addr1, 1)) != CONST_INT)
return 0;
offset1 = INTVAL (XEXP (addr1, 1));
}
}
else if (GET_CODE (addr1) != REG)
return 0;
else
{
reg1 = REGNO (addr1);
offset1 = 0;
}
if (GET_CODE (addr2) == PLUS)
{
if (GET_CODE (XEXP (addr2, 0)) != REG)
return 0;
else
{
reg2 = REGNO (XEXP (addr2, 0));
if (GET_CODE (XEXP (addr2, 1)) != CONST_INT)
return 0;
offset2 = INTVAL (XEXP (addr2, 1));
}
}
else if (GET_CODE (addr2) != REG)
return 0;
else
{
reg2 = REGNO (addr2);
offset2 = 0;
}
if (reg1 != reg2)
return 0;
if (offset2 != offset1 + 8)
return 0;
return 1;
}

static machine_mode
rs6000_secondary_memory_needed_mode (machine_mode mode)
{
if (lra_in_progress && mode == SDmode)
return DDmode;
return mode;
}
static enum rs6000_reg_type
register_to_reg_type (rtx reg, bool *is_altivec)
{
HOST_WIDE_INT regno;
enum reg_class rclass;
if (GET_CODE (reg) == SUBREG)
reg = SUBREG_REG (reg);
if (!REG_P (reg))
return NO_REG_TYPE;
regno = REGNO (reg);
if (regno >= FIRST_PSEUDO_REGISTER)
{
if (!lra_in_progress && !reload_completed)
return PSEUDO_REG_TYPE;
regno = true_regnum (reg);
if (regno < 0 || regno >= FIRST_PSEUDO_REGISTER)
return PSEUDO_REG_TYPE;
}
gcc_assert (regno >= 0);
if (is_altivec && ALTIVEC_REGNO_P (regno))
*is_altivec = true;
rclass = rs6000_regno_regclass[regno];
return reg_class_to_reg_type[(int)rclass];
}
static inline int
rs6000_secondary_reload_toc_costs (addr_mask_type addr_mask)
{
int ret;
if (TARGET_CMODEL != CMODEL_SMALL)
ret = ((addr_mask & RELOAD_REG_OFFSET) == 0) ? 1 : 2;
else
ret = (TARGET_MINIMAL_TOC) ? 6 : 3;
return ret;
}
static int
rs6000_secondary_reload_memory (rtx addr,
enum reg_class rclass,
machine_mode mode)
{
int extra_cost = 0;
rtx reg, and_arg, plus_arg0, plus_arg1;
addr_mask_type addr_mask;
const char *type = NULL;
const char *fail_msg = NULL;
if (GPR_REG_CLASS_P (rclass))
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_GPR];
else if (rclass == FLOAT_REGS)
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_FPR];
else if (rclass == ALTIVEC_REGS)
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_VMX];
else if (rclass == VSX_REGS)
addr_mask = (reg_addr[mode].addr_mask[RELOAD_REG_VMX]
& ~RELOAD_REG_AND_M16);
else if (rclass == NO_REGS)
{
addr_mask = (reg_addr[mode].addr_mask[RELOAD_REG_ANY]
& ~RELOAD_REG_AND_M16);
if ((addr_mask & RELOAD_REG_MULTIPLE) != 0)
addr_mask &= ~(RELOAD_REG_INDEXED
| RELOAD_REG_PRE_INCDEC
| RELOAD_REG_PRE_MODIFY);
}
else
addr_mask = 0;
if ((addr_mask & RELOAD_REG_VALID) == 0)
{
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr,
"rs6000_secondary_reload_memory: mode = %s, class = %s, "
"not valid in class\n",
GET_MODE_NAME (mode), reg_class_names[rclass]);
debug_rtx (addr);
}
return -1;
}
switch (GET_CODE (addr))
{
case PRE_INC:
case PRE_DEC:
reg = XEXP (addr, 0);
if (!base_reg_operand (addr, GET_MODE (reg)))
{
fail_msg = "no base register #1";
extra_cost = -1;
}
else if ((addr_mask & RELOAD_REG_PRE_INCDEC) == 0)
{
extra_cost = 1;
type = "update";
}
break;
case PRE_MODIFY:
reg = XEXP (addr, 0);
plus_arg1 = XEXP (addr, 1);
if (!base_reg_operand (reg, GET_MODE (reg))
|| GET_CODE (plus_arg1) != PLUS
|| !rtx_equal_p (reg, XEXP (plus_arg1, 0)))
{
fail_msg = "bad PRE_MODIFY";
extra_cost = -1;
}
else if ((addr_mask & RELOAD_REG_PRE_MODIFY) == 0)
{
extra_cost = 1;
type = "update";
}
break;
case AND:
and_arg = XEXP (addr, 0);
if (GET_MODE_SIZE (mode) != 16
|| GET_CODE (XEXP (addr, 1)) != CONST_INT
|| INTVAL (XEXP (addr, 1)) != -16)
{
fail_msg = "bad Altivec AND #1";
extra_cost = -1;
}
if (rclass != ALTIVEC_REGS)
{
if (legitimate_indirect_address_p (and_arg, false))
extra_cost = 1;
else if (legitimate_indexed_address_p (and_arg, false))
extra_cost = 2;
else
{
fail_msg = "bad Altivec AND #2";
extra_cost = -1;
}
type = "and";
}
break;
case REG:
case SUBREG:
if (!legitimate_indirect_address_p (addr, false))
{
extra_cost = 1;
type = "move";
}
break;
case PLUS:
plus_arg0 = XEXP (addr, 0);
plus_arg1 = XEXP (addr, 1);
if (GET_CODE (plus_arg0) == PLUS && CONST_INT_P (plus_arg1))
{
if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
extra_cost = 1;
type = "offset";
}
}
else if (GET_CODE (plus_arg0) == PLUS && REG_P (plus_arg1))
{
if ((addr_mask & RELOAD_REG_INDEXED) == 0)
{
extra_cost = 1;
type = "indexed #2";
}
}
else if (!base_reg_operand (plus_arg0, GET_MODE (plus_arg0)))
{
fail_msg = "no base register #2";
extra_cost = -1;
}
else if (int_reg_operand (plus_arg1, GET_MODE (plus_arg1)))
{
if ((addr_mask & RELOAD_REG_INDEXED) == 0
|| !legitimate_indexed_address_p (addr, false))
{
extra_cost = 1;
type = "indexed";
}
}
else if ((addr_mask & RELOAD_REG_QUAD_OFFSET) != 0
&& CONST_INT_P (plus_arg1))
{
if (!quad_address_offset_p (INTVAL (plus_arg1)))
{
extra_cost = 1;
type = "vector d-form offset";
}
}
else if (rs6000_legitimate_offset_address_p (mode, addr, false, true))
{
if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
extra_cost = 1;
type = "offset #2";
}
}
else
{
fail_msg = "bad PLUS";
extra_cost = -1;
}
break;
case LO_SUM:
if ((addr_mask & RELOAD_REG_QUAD_OFFSET) != 0)
{
extra_cost = -1;
type = "vector d-form lo_sum";
}
else if (!legitimate_lo_sum_address_p (mode, addr, false))
{
fail_msg = "bad LO_SUM";
extra_cost = -1;
}
if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
extra_cost = 1;
type = "lo_sum";
}
break;
case CONST:
case SYMBOL_REF:
case LABEL_REF:
if ((addr_mask & RELOAD_REG_QUAD_OFFSET) != 0)
{
extra_cost = -1;
type = "vector d-form lo_sum #2";
}
else
{
type = "address";
extra_cost = rs6000_secondary_reload_toc_costs (addr_mask);
}
break;
case UNSPEC:
if (TARGET_CMODEL == CMODEL_SMALL || XINT (addr, 1) != UNSPEC_TOCREL)
{
fail_msg = "bad UNSPEC";
extra_cost = -1;
}
else if ((addr_mask & RELOAD_REG_QUAD_OFFSET) != 0)
{
extra_cost = -1;
type = "vector d-form lo_sum #3";
}
else if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
extra_cost = 1;
type = "toc reference";
}
break;
default:
{
fail_msg = "bad address";
extra_cost = -1;
}
}
if (TARGET_DEBUG_ADDR )
{
if (extra_cost < 0)
fprintf (stderr,
"rs6000_secondary_reload_memory error: mode = %s, "
"class = %s, addr_mask = '%s', %s\n",
GET_MODE_NAME (mode),
reg_class_names[rclass],
rs6000_debug_addr_mask (addr_mask, false),
(fail_msg != NULL) ? fail_msg : "<bad address>");
else
fprintf (stderr,
"rs6000_secondary_reload_memory: mode = %s, class = %s, "
"addr_mask = '%s', extra cost = %d, %s\n",
GET_MODE_NAME (mode),
reg_class_names[rclass],
rs6000_debug_addr_mask (addr_mask, false),
extra_cost,
(type) ? type : "<none>");
debug_rtx (addr);
}
return extra_cost;
}
static bool
rs6000_secondary_reload_simple_move (enum rs6000_reg_type to_type,
enum rs6000_reg_type from_type,
machine_mode mode)
{
int size = GET_MODE_SIZE (mode);
if (TARGET_DIRECT_MOVE
&& ((to_type == GPR_REG_TYPE && from_type == VSX_REG_TYPE)
|| (to_type == VSX_REG_TYPE && from_type == GPR_REG_TYPE)))
{
if (TARGET_POWERPC64)
{
if (size == 8)
return true;
if (size == 16 && TARGET_P9_VECTOR && mode != TDmode)
return true;
}
if (TARGET_P8_VECTOR)
{
if (mode == SImode)
return true;
if (TARGET_P9_VECTOR && (mode == HImode || mode == QImode))
return true;
}
if (mode == SDmode)
return true;
}
else if (TARGET_MFPGPR && TARGET_POWERPC64 && size == 8
&& ((to_type == GPR_REG_TYPE && from_type == FPR_REG_TYPE)
|| (to_type == FPR_REG_TYPE && from_type == GPR_REG_TYPE)))
return true;
else if ((size == 4 || (TARGET_POWERPC64 && size == 8))
&& ((to_type == GPR_REG_TYPE && from_type == SPR_REG_TYPE)
|| (to_type == SPR_REG_TYPE && from_type == GPR_REG_TYPE)))
return true;
return false;
}
static bool
rs6000_secondary_reload_direct_move (enum rs6000_reg_type to_type,
enum rs6000_reg_type from_type,
machine_mode mode,
secondary_reload_info *sri,
bool altivec_p)
{
bool ret = false;
enum insn_code icode = CODE_FOR_nothing;
int cost = 0;
int size = GET_MODE_SIZE (mode);
if (TARGET_POWERPC64 && size == 16)
{
if (to_type == VSX_REG_TYPE && from_type == GPR_REG_TYPE)
{
cost = 3;			
icode = reg_addr[mode].reload_vsx_gpr;
}
else if (to_type == GPR_REG_TYPE && from_type == VSX_REG_TYPE)
{
cost = 3;			
icode = reg_addr[mode].reload_gpr_vsx;
}
}
else if (TARGET_POWERPC64 && mode == SFmode)
{
if (to_type == GPR_REG_TYPE && from_type == VSX_REG_TYPE)
{
cost = 3;			
icode = reg_addr[mode].reload_gpr_vsx;
}
else if (to_type == VSX_REG_TYPE && from_type == GPR_REG_TYPE)
{
cost = 2;			
icode = reg_addr[mode].reload_vsx_gpr;
}
}
else if (!TARGET_POWERPC64 && size == 8)
{
if (to_type == VSX_REG_TYPE && from_type == GPR_REG_TYPE && !altivec_p)
{
cost = 3;			
icode = reg_addr[mode].reload_fpr_gpr;
}
}
if (icode != CODE_FOR_nothing)
{
ret = true;
if (sri)
{
sri->icode = icode;
sri->extra_cost = cost;
}
}
return ret;
}
static bool
rs6000_secondary_reload_move (enum rs6000_reg_type to_type,
enum rs6000_reg_type from_type,
machine_mode mode,
secondary_reload_info *sri,
bool altivec_p)
{
if (to_type == NO_REG_TYPE || from_type == NO_REG_TYPE)
return false;
if ((to_type == PSEUDO_REG_TYPE && from_type == PSEUDO_REG_TYPE)
|| (to_type == PSEUDO_REG_TYPE && IS_STD_REG_TYPE (from_type))
|| (from_type == PSEUDO_REG_TYPE && IS_STD_REG_TYPE (to_type)))
return true;
if (to_type == from_type && IS_STD_REG_TYPE (to_type))
return true;
if (rs6000_secondary_reload_simple_move (to_type, from_type, mode))
{
if (sri)
{
sri->icode = CODE_FOR_nothing;
sri->extra_cost = 0;
}
return true;
}
return rs6000_secondary_reload_direct_move (to_type, from_type, mode, sri,
altivec_p);
}
static reg_class_t
rs6000_secondary_reload (bool in_p,
rtx x,
reg_class_t rclass_i,
machine_mode mode,
secondary_reload_info *sri)
{
enum reg_class rclass = (enum reg_class) rclass_i;
reg_class_t ret = ALL_REGS;
enum insn_code icode;
bool default_p = false;
bool done_p = false;
bool memory_p = (MEM_P (x)
|| (!reload_completed && GET_CODE (x) == SUBREG
&& MEM_P (SUBREG_REG (x))));
sri->icode = CODE_FOR_nothing;
sri->t_icode = CODE_FOR_nothing;
sri->extra_cost = 0;
icode = ((in_p)
? reg_addr[mode].reload_load
: reg_addr[mode].reload_store);
if (REG_P (x) || register_operand (x, mode))
{
enum rs6000_reg_type to_type = reg_class_to_reg_type[(int)rclass];
bool altivec_p = (rclass == ALTIVEC_REGS);
enum rs6000_reg_type from_type = register_to_reg_type (x, &altivec_p);
if (!in_p)
std::swap (to_type, from_type);
if (rs6000_secondary_reload_move (to_type, from_type, mode, sri,
altivec_p))
{
icode = (enum insn_code)sri->icode;
default_p = false;
done_p = true;
ret = NO_REGS;
}
}
if (x == CONST0_RTX (mode) && VSX_REG_CLASS_P (rclass))
{
ret = NO_REGS;
default_p = false;
done_p = true;
}
if (!done_p && reg_addr[mode].scalar_in_vmx_p
&& !mode_supports_vmx_dform (mode)
&& (rclass == VSX_REGS || rclass == ALTIVEC_REGS)
&& (memory_p || (GET_CODE (x) == CONST_DOUBLE)))
{
ret = FLOAT_REGS;
default_p = false;
done_p = true;
}
if (!done_p && icode != CODE_FOR_nothing && memory_p)
{
int extra_cost = rs6000_secondary_reload_memory (XEXP (x, 0), rclass,
mode);
if (extra_cost >= 0)
{
done_p = true;
ret = NO_REGS;
if (extra_cost > 0)
{
sri->extra_cost = extra_cost;
sri->icode = icode;
}
}
}
if (!done_p && TARGET_POWERPC64
&& reg_class_to_reg_type[(int)rclass] == GPR_REG_TYPE
&& memory_p
&& GET_MODE_SIZE (GET_MODE (x)) >= UNITS_PER_WORD)
{
rtx addr = XEXP (x, 0);
rtx off = address_offset (addr);
if (off != NULL_RTX)
{
unsigned int extra = GET_MODE_SIZE (GET_MODE (x)) - UNITS_PER_WORD;
unsigned HOST_WIDE_INT offset = INTVAL (off);
if (GET_CODE (addr) == LO_SUM
? (1 
&& ((offset & 3) != 0
|| ((offset & 0xffff) ^ 0x8000) >= 0x10000 - extra))
: (offset + 0x8000 < 0x10000 - extra 
&& (offset & 3) != 0))
{
if (in_p)
sri->icode = ((TARGET_32BIT) ? CODE_FOR_reload_si_load
: CODE_FOR_reload_di_load);
else
sri->icode = ((TARGET_32BIT) ? CODE_FOR_reload_si_store
: CODE_FOR_reload_di_store);
sri->extra_cost = 2;
ret = NO_REGS;
done_p = true;
}
else
default_p = true;
}
else
default_p = true;
}
if (!done_p && !TARGET_POWERPC64
&& reg_class_to_reg_type[(int)rclass] == GPR_REG_TYPE
&& memory_p
&& GET_MODE_SIZE (GET_MODE (x)) > UNITS_PER_WORD)
{
rtx addr = XEXP (x, 0);
rtx off = address_offset (addr);
if (off != NULL_RTX)
{
unsigned int extra = GET_MODE_SIZE (GET_MODE (x)) - UNITS_PER_WORD;
unsigned HOST_WIDE_INT offset = INTVAL (off);
if (GET_CODE (addr) == LO_SUM
? ((offset & 0xffff) ^ 0x8000) >= 0x10000 - extra
: offset - (0x8000 - extra) < UNITS_PER_WORD)
{
if (in_p)
sri->icode = CODE_FOR_reload_si_load;
else
sri->icode = CODE_FOR_reload_si_store;
sri->extra_cost = 2;
ret = NO_REGS;
done_p = true;
}
else
default_p = true;
}
else
default_p = true;
}
if (!done_p)
default_p = true;
if (default_p)
ret = default_secondary_reload (in_p, x, rclass, mode, sri);
gcc_assert (ret != ALL_REGS);
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr,
"\nrs6000_secondary_reload, return %s, in_p = %s, rclass = %s, "
"mode = %s",
reg_class_names[ret],
in_p ? "true" : "false",
reg_class_names[rclass],
GET_MODE_NAME (mode));
if (reload_completed)
fputs (", after reload", stderr);
if (!done_p)
fputs (", done_p not set", stderr);
if (default_p)
fputs (", default secondary reload", stderr);
if (sri->icode != CODE_FOR_nothing)
fprintf (stderr, ", reload func = %s, extra cost = %d",
insn_data[sri->icode].name, sri->extra_cost);
else if (sri->extra_cost > 0)
fprintf (stderr, ", extra cost = %d", sri->extra_cost);
fputs ("\n", stderr);
debug_rtx (x);
}
return ret;
}
static void
rs6000_secondary_reload_trace (int line, rtx reg, rtx mem, rtx scratch,
bool store_p)
{
rtx set, clobber;
gcc_assert (reg != NULL_RTX && mem != NULL_RTX && scratch != NULL_RTX);
fprintf (stderr, "rs6000_secondary_reload_inner:%d, type = %s\n", line,
store_p ? "store" : "load");
if (store_p)
set = gen_rtx_SET (mem, reg);
else
set = gen_rtx_SET (reg, mem);
clobber = gen_rtx_CLOBBER (VOIDmode, scratch);
debug_rtx (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, set, clobber)));
}
static void rs6000_secondary_reload_fail (int, rtx, rtx, rtx, bool)
ATTRIBUTE_NORETURN;
static void
rs6000_secondary_reload_fail (int line, rtx reg, rtx mem, rtx scratch,
bool store_p)
{
rs6000_secondary_reload_trace (line, reg, mem, scratch, store_p);
gcc_unreachable ();
}
void
rs6000_secondary_reload_inner (rtx reg, rtx mem, rtx scratch, bool store_p)
{
int regno = true_regnum (reg);
machine_mode mode = GET_MODE (reg);
addr_mask_type addr_mask;
rtx addr;
rtx new_addr;
rtx op_reg, op0, op1;
rtx and_op;
rtx cc_clobber;
rtvec rv;
if (regno < 0 || regno >= FIRST_PSEUDO_REGISTER || !MEM_P (mem)
|| !base_reg_operand (scratch, GET_MODE (scratch)))
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
if (IN_RANGE (regno, FIRST_GPR_REGNO, LAST_GPR_REGNO))
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_GPR];
else if (IN_RANGE (regno, FIRST_FPR_REGNO, LAST_FPR_REGNO))
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_FPR];
else if (IN_RANGE (regno, FIRST_ALTIVEC_REGNO, LAST_ALTIVEC_REGNO))
addr_mask = reg_addr[mode].addr_mask[RELOAD_REG_VMX];
else
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
if ((addr_mask & RELOAD_REG_VALID) == 0)
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
if (TARGET_DEBUG_ADDR)
rs6000_secondary_reload_trace (__LINE__, reg, mem, scratch, store_p);
new_addr = addr = XEXP (mem, 0);
switch (GET_CODE (addr))
{
case PRE_INC:
case PRE_DEC:
op_reg = XEXP (addr, 0);
if (!base_reg_operand (op_reg, Pmode))
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
if ((addr_mask & RELOAD_REG_PRE_INCDEC) == 0)
{
emit_insn (gen_add2_insn (op_reg, GEN_INT (GET_MODE_SIZE (mode))));
new_addr = op_reg;
}
break;
case PRE_MODIFY:
op0 = XEXP (addr, 0);
op1 = XEXP (addr, 1);
if (!base_reg_operand (op0, Pmode)
|| GET_CODE (op1) != PLUS
|| !rtx_equal_p (op0, XEXP (op1, 0)))
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
if ((addr_mask & RELOAD_REG_PRE_MODIFY) == 0)
{
emit_insn (gen_rtx_SET (op0, op1));
new_addr = reg;
}
break;
case AND:
op0 = XEXP (addr, 0);
op1 = XEXP (addr, 1);
if ((addr_mask & RELOAD_REG_AND_M16) == 0)
{
if (REG_P (op0) || GET_CODE (op0) == SUBREG)
op_reg = op0;
else if (GET_CODE (op1) == PLUS)
{
emit_insn (gen_rtx_SET (scratch, op1));
op_reg = scratch;
}
else
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
and_op = gen_rtx_AND (GET_MODE (scratch), op_reg, op1);
cc_clobber = gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (CCmode));
rv = gen_rtvec (2, gen_rtx_SET (scratch, and_op), cc_clobber);
emit_insn (gen_rtx_PARALLEL (VOIDmode, rv));
new_addr = scratch;
}
break;
case REG:
case SUBREG:
if (!base_reg_operand (addr, GET_MODE (addr)))
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
break;
case PLUS:
op0 = XEXP (addr, 0);
op1 = XEXP (addr, 1);
if (!base_reg_operand (op0, Pmode))
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
else if (int_reg_operand (op1, Pmode))
{
if ((addr_mask & RELOAD_REG_INDEXED) == 0)
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
}
else if (mode_supports_vsx_dform_quad (mode) && CONST_INT_P (op1))
{
if (((addr_mask & RELOAD_REG_QUAD_OFFSET) == 0)
|| !quad_address_p (addr, mode, false))
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
}
else if (rs6000_legitimate_offset_address_p (mode, addr, false, true))
{
if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
}
else
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
break;
case LO_SUM:
op0 = XEXP (addr, 0);
op1 = XEXP (addr, 1);
if (!base_reg_operand (op0, Pmode))
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
else if (int_reg_operand (op1, Pmode))
{
if ((addr_mask & RELOAD_REG_INDEXED) == 0)
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
}
else if (mode_supports_vsx_dform_quad (mode))
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
else if (legitimate_lo_sum_address_p (mode, addr, false))
{
if ((addr_mask & RELOAD_REG_OFFSET) == 0)
{
emit_insn (gen_rtx_SET (scratch, addr));
new_addr = scratch;
}
}
else
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
break;
case SYMBOL_REF:
case CONST:
case LABEL_REF:
rs6000_emit_move (scratch, addr, Pmode);
new_addr = scratch;
break;
default:
rs6000_secondary_reload_fail (__LINE__, reg, mem, scratch, store_p);
}
if (addr != new_addr)
{
mem = replace_equiv_address_nv (mem, new_addr);
if (TARGET_DEBUG_ADDR)
fprintf (stderr, "\nrs6000_secondary_reload_inner, mem adjusted.\n");
}
if (store_p)
emit_insn (gen_rtx_SET (mem, reg));
else
emit_insn (gen_rtx_SET (reg, mem));
return;
}
void
rs6000_secondary_reload_gpr (rtx reg, rtx mem, rtx scratch, bool store_p)
{
int regno = true_regnum (reg);
enum reg_class rclass;
rtx addr;
rtx scratch_or_premodify = scratch;
if (TARGET_DEBUG_ADDR)
{
fprintf (stderr, "\nrs6000_secondary_reload_gpr, type = %s\n",
store_p ? "store" : "load");
fprintf (stderr, "reg:\n");
debug_rtx (reg);
fprintf (stderr, "mem:\n");
debug_rtx (mem);
fprintf (stderr, "scratch:\n");
debug_rtx (scratch);
}
gcc_assert (regno >= 0 && regno < FIRST_PSEUDO_REGISTER);
gcc_assert (GET_CODE (mem) == MEM);
rclass = REGNO_REG_CLASS (regno);
gcc_assert (rclass == GENERAL_REGS || rclass == BASE_REGS);
addr = XEXP (mem, 0);
if (GET_CODE (addr) == PRE_MODIFY)
{
gcc_assert (REG_P (XEXP (addr, 0))
&& GET_CODE (XEXP (addr, 1)) == PLUS
&& XEXP (XEXP (addr, 1), 0) == XEXP (addr, 0));
scratch_or_premodify = XEXP (addr, 0);
if (!HARD_REGISTER_P (scratch_or_premodify))
scratch_or_premodify = find_replacement (&XEXP (addr, 0));
addr = XEXP (addr, 1);
}
gcc_assert (GET_CODE (addr) == PLUS || GET_CODE (addr) == LO_SUM);
rs6000_emit_move (scratch_or_premodify, addr, Pmode);
mem = replace_equiv_address_nv (mem, scratch_or_premodify);
if (store_p)
emit_insn (gen_rtx_SET (mem, reg));
else
emit_insn (gen_rtx_SET (reg, mem));
return;
}
static enum reg_class
rs6000_preferred_reload_class (rtx x, enum reg_class rclass)
{
machine_mode mode = GET_MODE (x);
bool is_constant = CONSTANT_P (x);
if ((rclass == ALTIVEC_REGS || rclass == VSX_REGS)
&& (reg_addr[mode].addr_mask[RELOAD_REG_VMX] & RELOAD_REG_VALID) == 0)
return NO_REGS;
if ((rclass == FLOAT_REGS || rclass == VSX_REGS)
&& (reg_addr[mode].addr_mask[RELOAD_REG_FPR] & RELOAD_REG_VALID) == 0)
return NO_REGS;
if (TARGET_VSX && VSX_REG_CLASS_P (rclass) && GET_CODE (x) != PLUS)
{
if (is_constant)
{
if (x == CONST0_RTX (mode))
return rclass;
if (GET_CODE (x) == CONST_VECTOR && easy_vector_constant (x, mode))
return ALTIVEC_REGS;
if (CONST_INT_P (x))
{
HOST_WIDE_INT value = INTVAL (x);
if (value == -1)
{
if (TARGET_P8_VECTOR)
return rclass;
else if (rclass == ALTIVEC_REGS || rclass == VSX_REGS)
return ALTIVEC_REGS;
else
return NO_REGS;
}
if (IN_RANGE (value, -128, 127) && TARGET_P9_VECTOR
&& (rclass == ALTIVEC_REGS || rclass == VSX_REGS))
return ALTIVEC_REGS;
}
return NO_REGS;
}
if (mode_supports_vmx_dform (mode)
|| mode_supports_vsx_dform_quad (mode))
return rclass;
if (rclass == VSX_REGS
&& (mode == SFmode || GET_MODE_SIZE (mode) == 8))
return FLOAT_REGS;
if (VECTOR_UNIT_ALTIVEC_P (mode) || VECTOR_MEM_ALTIVEC_P (mode)
|| mode == V1TImode)
return ALTIVEC_REGS;
return rclass;
}
if (is_constant || GET_CODE (x) == PLUS)
{
if (reg_class_subset_p (GENERAL_REGS, rclass))
return GENERAL_REGS;
if (reg_class_subset_p (BASE_REGS, rclass))
return BASE_REGS;
return NO_REGS;
}
if (GET_MODE_CLASS (mode) == MODE_INT && rclass == NON_SPECIAL_REGS)
return GENERAL_REGS;
return rclass;
}
static enum reg_class
rs6000_debug_preferred_reload_class (rtx x, enum reg_class rclass)
{
enum reg_class ret = rs6000_preferred_reload_class (x, rclass);
fprintf (stderr,
"\nrs6000_preferred_reload_class, return %s, rclass = %s, "
"mode = %s, x:\n",
reg_class_names[ret], reg_class_names[rclass],
GET_MODE_NAME (GET_MODE (x)));
debug_rtx (x);
return ret;
}
static bool
rs6000_secondary_memory_needed (machine_mode mode,
reg_class_t from_class,
reg_class_t to_class)
{
enum rs6000_reg_type from_type, to_type;
bool altivec_p = ((from_class == ALTIVEC_REGS)
|| (to_class == ALTIVEC_REGS));
from_type = reg_class_to_reg_type[(int)from_class];
to_type = reg_class_to_reg_type[(int)to_class];
if (rs6000_secondary_reload_move (to_type, from_type, mode,
(secondary_reload_info *)0, altivec_p))
return false;
if (IS_FP_VECT_REG_TYPE (from_type) || IS_FP_VECT_REG_TYPE (to_type))
return true;
return false;
}
static bool
rs6000_debug_secondary_memory_needed (machine_mode mode,
reg_class_t from_class,
reg_class_t to_class)
{
bool ret = rs6000_secondary_memory_needed (mode, from_class, to_class);
fprintf (stderr,
"rs6000_secondary_memory_needed, return: %s, from_class = %s, "
"to_class = %s, mode = %s\n",
ret ? "true" : "false",
reg_class_names[from_class],
reg_class_names[to_class],
GET_MODE_NAME (mode));
return ret;
}
static enum reg_class
rs6000_secondary_reload_class (enum reg_class rclass, machine_mode mode,
rtx in)
{
int regno;
if (TARGET_ELF || (DEFAULT_ABI == ABI_DARWIN
#if TARGET_MACHO
&& MACHOPIC_INDIRECT
#endif
))
{
if (rclass != BASE_REGS
&& (GET_CODE (in) == SYMBOL_REF
|| GET_CODE (in) == HIGH
|| GET_CODE (in) == LABEL_REF
|| GET_CODE (in) == CONST))
return BASE_REGS;
}
if (GET_CODE (in) == REG)
{
regno = REGNO (in);
if (regno >= FIRST_PSEUDO_REGISTER)
{
regno = true_regnum (in);
if (regno >= FIRST_PSEUDO_REGISTER)
regno = -1;
}
}
else if (GET_CODE (in) == SUBREG)
{
regno = true_regnum (in);
if (regno >= FIRST_PSEUDO_REGISTER)
regno = -1;
}
else
regno = -1;
if (TARGET_VSX
&& GET_MODE_SIZE (mode) < 16
&& !mode_supports_vmx_dform (mode)
&& (((rclass == GENERAL_REGS || rclass == BASE_REGS)
&& (regno >= 0 && ALTIVEC_REGNO_P (regno)))
|| ((rclass == VSX_REGS || rclass == ALTIVEC_REGS)
&& (regno >= 0 && INT_REGNO_P (regno)))))
return FLOAT_REGS;
if (rclass == GENERAL_REGS || rclass == BASE_REGS
|| (regno >= 0 && INT_REGNO_P (regno)))
return NO_REGS;
if (rclass == VSX_REGS
&& (regno == -1 || VSX_REGNO_P (regno)))
return NO_REGS;
if ((regno == -1 || FP_REGNO_P (regno))
&& (rclass == FLOAT_REGS || rclass == NON_SPECIAL_REGS))
return (mode != SDmode || lra_in_progress) ? NO_REGS : GENERAL_REGS;
if ((regno == -1 || ALTIVEC_REGNO_P (regno))
&& rclass == ALTIVEC_REGS)
return NO_REGS;
if ((rclass == CR_REGS || rclass == CR0_REGS)
&& regno >= 0 && CR_REGNO_P (regno))
return NO_REGS;
return GENERAL_REGS;
}
static enum reg_class
rs6000_debug_secondary_reload_class (enum reg_class rclass,
machine_mode mode, rtx in)
{
enum reg_class ret = rs6000_secondary_reload_class (rclass, mode, in);
fprintf (stderr,
"\nrs6000_secondary_reload_class, return %s, rclass = %s, "
"mode = %s, input rtx:\n",
reg_class_names[ret], reg_class_names[rclass],
GET_MODE_NAME (mode));
debug_rtx (in);
return ret;
}
static bool
rs6000_can_change_mode_class (machine_mode from,
machine_mode to,
reg_class_t rclass)
{
unsigned from_size = GET_MODE_SIZE (from);
unsigned to_size = GET_MODE_SIZE (to);
if (from_size != to_size)
{
enum reg_class xclass = (TARGET_VSX) ? VSX_REGS : FLOAT_REGS;
if (reg_classes_intersect_p (xclass, rclass))
{
unsigned to_nregs = hard_regno_nregs (FIRST_FPR_REGNO, to);
unsigned from_nregs = hard_regno_nregs (FIRST_FPR_REGNO, from);
bool to_float128_vector_p = FLOAT128_VECTOR_P (to);
bool from_float128_vector_p = FLOAT128_VECTOR_P (from);
if (to_float128_vector_p && from_float128_vector_p)
return true;
else if (to_float128_vector_p || from_float128_vector_p)
return false;
if (!BYTES_BIG_ENDIAN && (to == TDmode || from == TDmode))
return false;
if (from_size < 8 || to_size < 8)
return false;
if (from_size == 8 && (8 * to_nregs) != to_size)
return false;
if (to_size == 8 && (8 * from_nregs) != from_size)
return false;
return true;
}
else
return true;
}
if (TARGET_VSX && VSX_REG_CLASS_P (rclass))
{
unsigned num_regs = (from_size + 15) / 16;
if (hard_regno_nregs (FIRST_FPR_REGNO, to) > num_regs
|| hard_regno_nregs (FIRST_FPR_REGNO, from) > num_regs)
return false;
return (from_size == 8 || from_size == 16);
}
if (TARGET_ALTIVEC && rclass == ALTIVEC_REGS
&& (ALTIVEC_VECTOR_MODE (from) + ALTIVEC_VECTOR_MODE (to)) == 1)
return false;
return true;
}
static bool
rs6000_debug_can_change_mode_class (machine_mode from,
machine_mode to,
reg_class_t rclass)
{
bool ret = rs6000_can_change_mode_class (from, to, rclass);
fprintf (stderr,
"rs6000_can_change_mode_class, return %s, from = %s, "
"to = %s, rclass = %s\n",
ret ? "true" : "false",
GET_MODE_NAME (from), GET_MODE_NAME (to),
reg_class_names[rclass]);
return ret;
}

const char *
rs6000_output_move_128bit (rtx operands[])
{
rtx dest = operands[0];
rtx src = operands[1];
machine_mode mode = GET_MODE (dest);
int dest_regno;
int src_regno;
bool dest_gpr_p, dest_fp_p, dest_vmx_p, dest_vsx_p;
bool src_gpr_p, src_fp_p, src_vmx_p, src_vsx_p;
if (REG_P (dest))
{
dest_regno = REGNO (dest);
dest_gpr_p = INT_REGNO_P (dest_regno);
dest_fp_p = FP_REGNO_P (dest_regno);
dest_vmx_p = ALTIVEC_REGNO_P (dest_regno);
dest_vsx_p = dest_fp_p | dest_vmx_p;
}
else
{
dest_regno = -1;
dest_gpr_p = dest_fp_p = dest_vmx_p = dest_vsx_p = false;
}
if (REG_P (src))
{
src_regno = REGNO (src);
src_gpr_p = INT_REGNO_P (src_regno);
src_fp_p = FP_REGNO_P (src_regno);
src_vmx_p = ALTIVEC_REGNO_P (src_regno);
src_vsx_p = src_fp_p | src_vmx_p;
}
else
{
src_regno = -1;
src_gpr_p = src_fp_p = src_vmx_p = src_vsx_p = false;
}
if (dest_regno >= 0 && src_regno >= 0)
{
if (dest_gpr_p)
{
if (src_gpr_p)
return "#";
if (TARGET_DIRECT_MOVE_128 && src_vsx_p)
return (WORDS_BIG_ENDIAN
? "mfvsrd %0,%x1\n\tmfvsrld %L0,%x1"
: "mfvsrd %L0,%x1\n\tmfvsrld %0,%x1");
else if (TARGET_VSX && TARGET_DIRECT_MOVE && src_vsx_p)
return "#";
}
else if (TARGET_VSX && dest_vsx_p)
{
if (src_vsx_p)
return "xxlor %x0,%x1,%x1";
else if (TARGET_DIRECT_MOVE_128 && src_gpr_p)
return (WORDS_BIG_ENDIAN
? "mtvsrdd %x0,%1,%L1"
: "mtvsrdd %x0,%L1,%1");
else if (TARGET_DIRECT_MOVE && src_gpr_p)
return "#";
}
else if (TARGET_ALTIVEC && dest_vmx_p && src_vmx_p)
return "vor %0,%1,%1";
else if (dest_fp_p && src_fp_p)
return "#";
}
else if (dest_regno >= 0 && MEM_P (src))
{
if (dest_gpr_p)
{
if (TARGET_QUAD_MEMORY && quad_load_store_p (dest, src))
return "lq %0,%1";
else
return "#";
}
else if (TARGET_ALTIVEC && dest_vmx_p
&& altivec_indexed_or_indirect_operand (src, mode))
return "lvx %0,%y1";
else if (TARGET_VSX && dest_vsx_p)
{
if (mode_supports_vsx_dform_quad (mode)
&& quad_address_p (XEXP (src, 0), mode, true))
return "lxv %x0,%1";
else if (TARGET_P9_VECTOR)
return "lxvx %x0,%y1";
else if (mode == V16QImode || mode == V8HImode || mode == V4SImode)
return "lxvw4x %x0,%y1";
else
return "lxvd2x %x0,%y1";
}
else if (TARGET_ALTIVEC && dest_vmx_p)
return "lvx %0,%y1";
else if (dest_fp_p)
return "#";
}
else if (src_regno >= 0 && MEM_P (dest))
{
if (src_gpr_p)
{
if (TARGET_QUAD_MEMORY && quad_load_store_p (dest, src))
return "stq %1,%0";
else
return "#";
}
else if (TARGET_ALTIVEC && src_vmx_p
&& altivec_indexed_or_indirect_operand (dest, mode))
return "stvx %1,%y0";
else if (TARGET_VSX && src_vsx_p)
{
if (mode_supports_vsx_dform_quad (mode)
&& quad_address_p (XEXP (dest, 0), mode, true))
return "stxv %x1,%0";
else if (TARGET_P9_VECTOR)
return "stxvx %x1,%y0";
else if (mode == V16QImode || mode == V8HImode || mode == V4SImode)
return "stxvw4x %x1,%y0";
else
return "stxvd2x %x1,%y0";
}
else if (TARGET_ALTIVEC && src_vmx_p)
return "stvx %1,%y0";
else if (src_fp_p)
return "#";
}
else if (dest_regno >= 0
&& (GET_CODE (src) == CONST_INT
|| GET_CODE (src) == CONST_WIDE_INT
|| GET_CODE (src) == CONST_DOUBLE
|| GET_CODE (src) == CONST_VECTOR))
{
if (dest_gpr_p)
return "#";
else if ((dest_vmx_p && TARGET_ALTIVEC)
|| (dest_vsx_p && TARGET_VSX))
return output_vec_const_move (operands);
}
fatal_insn ("Bad 128-bit move", gen_rtx_SET (dest, src));
}
bool
rs6000_move_128bit_ok_p (rtx operands[])
{
machine_mode mode = GET_MODE (operands[0]);
return (gpc_reg_operand (operands[0], mode)
|| gpc_reg_operand (operands[1], mode));
}
bool
rs6000_split_128bit_ok_p (rtx operands[])
{
if (!reload_completed)
return false;
if (!gpr_or_gpr_p (operands[0], operands[1]))
return false;
if (quad_load_store_p (operands[0], operands[1]))
return false;
return true;
}

int
ccr_bit (rtx op, int scc_p)
{
enum rtx_code code = GET_CODE (op);
machine_mode cc_mode;
int cc_regnum;
int base_bit;
rtx reg;
if (!COMPARISON_P (op))
return -1;
reg = XEXP (op, 0);
gcc_assert (GET_CODE (reg) == REG && CR_REGNO_P (REGNO (reg)));
cc_mode = GET_MODE (reg);
cc_regnum = REGNO (reg);
base_bit = 4 * (cc_regnum - CR0_REGNO);
validate_condition_mode (code, cc_mode);
gcc_assert (!scc_p
|| code == EQ || code == GT || code == LT || code == UNORDERED
|| code == GTU || code == LTU);
switch (code)
{
case NE:
return scc_p ? base_bit + 3 : base_bit + 2;
case EQ:
return base_bit + 2;
case GT:  case GTU:  case UNLE:
return base_bit + 1;
case LT:  case LTU:  case UNGE:
return base_bit;
case ORDERED:  case UNORDERED:
return base_bit + 3;
case GE:  case GEU:
return scc_p ? base_bit + 3 : base_bit;
case LE:  case LEU:
return scc_p ? base_bit + 3 : base_bit + 1;
default:
gcc_unreachable ();
}
}

rtx
rs6000_got_register (rtx value ATTRIBUTE_UNUSED)
{
if (!can_create_pseudo_p ()
&& !df_regs_ever_live_p (RS6000_PIC_OFFSET_TABLE_REGNUM))
df_set_regs_ever_live (RS6000_PIC_OFFSET_TABLE_REGNUM, true);
crtl->uses_pic_offset_table = 1;
return pic_offset_table_rtx;
}

static rs6000_stack_t stack_info;
static struct machine_function *
rs6000_init_machine_status (void)
{
stack_info.reload_completed = 0;
return ggc_cleared_alloc<machine_function> ();
}

#define INT_P(X) (GET_CODE (X) == CONST_INT && GET_MODE (X) == VOIDmode)
void
rs6000_output_function_entry (FILE *file, const char *fname)
{
if (fname[0] != '.')
{
switch (DEFAULT_ABI)
{
default:
gcc_unreachable ();
case ABI_AIX:
if (DOT_SYMBOLS)
putc ('.', file);
else
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "L.");
break;
case ABI_ELFv2:
case ABI_V4:
case ABI_DARWIN:
break;
}
}
RS6000_OUTPUT_BASENAME (file, fname);
}
#if TARGET_ELF
#define SMALL_DATA_RELOC ((rs6000_sdata == SDATA_EABI) ? "sda21" : "sdarel")
#define SMALL_DATA_REG ((rs6000_sdata == SDATA_EABI) ? 0 : 13)
#else
#define SMALL_DATA_RELOC "sda21"
#define SMALL_DATA_REG 0
#endif
void
print_operand (FILE *file, rtx x, int code)
{
int i;
unsigned HOST_WIDE_INT uval;
switch (code)
{
case 'D':
gcc_assert (REG_P (x));
i = 4 * (REGNO (x) - CR0_REGNO) + 1;
fprintf (file, "%d", i + 1);
return;
case 'e':
if (! INT_P (x))
{
output_operand_lossage ("invalid %%e value");
return;
}
uval = INTVAL (x);
if ((uval & 0xffff) == 0 && uval != 0)
putc ('s', file);
return;
case 'E':
if (GET_CODE (x) != REG || ! CR_REGNO_P (REGNO (x)))
output_operand_lossage ("invalid %%E value");
else
fprintf (file, "%d", 4 * (REGNO (x) - CR0_REGNO) + 2);
return;
case 'f':
if (GET_CODE (x) != REG || ! CR_REGNO_P (REGNO (x)))
output_operand_lossage ("invalid %%f value");
else
fprintf (file, "%d", 4 * (REGNO (x) - CR0_REGNO));
return;
case 'F':
if (GET_CODE (x) != REG || ! CR_REGNO_P (REGNO (x)))
output_operand_lossage ("invalid %%F value");
else
fprintf (file, "%d", 32 - 4 * (REGNO (x) - CR0_REGNO));
return;
case 'G':
if (GET_CODE (x) != CONST_INT)
output_operand_lossage ("invalid %%G value");
else if (INTVAL (x) >= 0)
putc ('z', file);
else
putc ('m', file);
return;
case 'h':
if (INT_P (x))
fprintf (file, HOST_WIDE_INT_PRINT_DEC, INTVAL (x) & 31);
else
print_operand (file, x, 0);
return;
case 'H':
if (INT_P (x))
fprintf (file, HOST_WIDE_INT_PRINT_DEC, INTVAL (x) & 63);
else
print_operand (file, x, 0);
return;
case 'I':
if (INT_P (x))
putc ('i', file);
return;
case 'j':
i = ccr_bit (x, 0);
if (i == -1)
output_operand_lossage ("invalid %%j code");
else
fprintf (file, "%d", i);
return;
case 'J':
i = ccr_bit (x, 1);
if (i == -1)
output_operand_lossage ("invalid %%J code");
else
fprintf (file, "%d", i == 31 ? 0 : i + 1);
return;
case 'k':
if (! INT_P (x))
output_operand_lossage ("invalid %%k value");
else
fprintf (file, HOST_WIDE_INT_PRINT_DEC, ~ INTVAL (x));
return;
case 'K':
if (GET_CODE (x) == CONST)
{
if (GET_CODE (XEXP (x, 0)) != PLUS
|| (GET_CODE (XEXP (XEXP (x, 0), 0)) != SYMBOL_REF
&& GET_CODE (XEXP (XEXP (x, 0), 0)) != LABEL_REF)
|| GET_CODE (XEXP (XEXP (x, 0), 1)) != CONST_INT)
output_operand_lossage ("invalid %%K value");
}
print_operand_address (file, x);
fputs ("@l", file);
return;
case 'L':
if (REG_P (x))
fputs (reg_names[REGNO (x) + 1], file);
else if (MEM_P (x))
{
machine_mode mode = GET_MODE (x);
if (GET_CODE (XEXP (x, 0)) == PRE_INC
|| GET_CODE (XEXP (x, 0)) == PRE_DEC)
output_address (mode, plus_constant (Pmode, XEXP (XEXP (x, 0), 0),
UNITS_PER_WORD));
else if (GET_CODE (XEXP (x, 0)) == PRE_MODIFY)
output_address (mode, plus_constant (Pmode, XEXP (XEXP (x, 0), 0),
UNITS_PER_WORD));
else
output_address (mode, XEXP (adjust_address_nv (x, SImode,
UNITS_PER_WORD),
0));
if (small_data_operand (x, GET_MODE (x)))
fprintf (file, "@%s(%s)", SMALL_DATA_RELOC,
reg_names[SMALL_DATA_REG]);
}
return;
case 'N': 
if (GET_CODE (x) != PARALLEL)
output_operand_lossage ("invalid %%N value");
else
fprintf (file, "%d", XVECLEN (x, 0) * 4);
return;
case 'O': 
if (GET_CODE (x) != PARALLEL)
output_operand_lossage ("invalid %%O value");
else
fprintf (file, "%d", (XVECLEN (x, 0) - 1) * 4);
return;
case 'p':
if (! INT_P (x)
|| INTVAL (x) < 0
|| (i = exact_log2 (INTVAL (x))) < 0)
output_operand_lossage ("invalid %%p value");
else
fprintf (file, "%d", i);
return;
case 'P':
if (GET_CODE (x) != MEM || GET_CODE (XEXP (x, 0)) != REG
|| REGNO (XEXP (x, 0)) >= 32)
output_operand_lossage ("invalid %%P value");
else
fputs (reg_names[REGNO (XEXP (x, 0))], file);
return;
case 'q':
{
const char *const *t = 0;
const char *s;
enum rtx_code code = GET_CODE (x);
static const char * const tbl[3][3] = {
{ "and", "andc", "nor" },
{ "or", "orc", "nand" },
{ "xor", "eqv", "xor" } };
if (code == AND)
t = tbl[0];
else if (code == IOR)
t = tbl[1];
else if (code == XOR)
t = tbl[2];
else
output_operand_lossage ("invalid %%q value");
if (GET_CODE (XEXP (x, 0)) != NOT)
s = t[0];
else
{
if (GET_CODE (XEXP (x, 1)) == NOT)
s = t[2];
else
s = t[1];
}
fputs (s, file);
}
return;
case 'Q':
if (! TARGET_MFCRF)
return;
fputc (',', file);
case 'R':
if (GET_CODE (x) != REG || ! CR_REGNO_P (REGNO (x)))
output_operand_lossage ("invalid %%R value");
else
fprintf (file, "%d", 128 >> (REGNO (x) - CR0_REGNO));
return;
case 's':
if (! INT_P (x))
output_operand_lossage ("invalid %%s value");
else
fprintf (file, HOST_WIDE_INT_PRINT_DEC, (32 - INTVAL (x)) & 31);
return;
case 't':
gcc_assert (REG_P (x) && GET_MODE (x) == CCmode);
i = 4 * (REGNO (x) - CR0_REGNO) + 3;
fprintf (file, "%d", i == 31 ? 0 : i + 1);
return;
case 'T':
if (GET_CODE (x) != REG || (REGNO (x) != LR_REGNO
&& REGNO (x) != CTR_REGNO))
output_operand_lossage ("invalid %%T value");
else if (REGNO (x) == LR_REGNO)
fputs ("lr", file);
else
fputs ("ctr", file);
return;
case 'u':
if (! INT_P (x))
{
output_operand_lossage ("invalid %%u value");
return;
}
uval = INTVAL (x);
if ((uval & 0xffff) == 0)
uval >>= 16;
fprintf (file, HOST_WIDE_INT_PRINT_HEX, uval & 0xffff);
return;
case 'v':
if (! INT_P (x))
output_operand_lossage ("invalid %%v value");
else
fprintf (file, HOST_WIDE_INT_PRINT_HEX,
(INTVAL (x) >> 16) & 0xffff);
return;
case 'U':
if (MEM_P (x)
&& (GET_CODE (XEXP (x, 0)) == PRE_INC
|| GET_CODE (XEXP (x, 0)) == PRE_DEC
|| GET_CODE (XEXP (x, 0)) == PRE_MODIFY))
putc ('u', file);
return;
case 'V':
switch (GET_CODE (x))
{
case EQ:
fputs ("eq", file);   
break;
case NE:
fputs ("ne", file);   
break;
case LT:
fputs ("lt", file);   
break;
case LE:
fputs ("le", file);   
break;
case GT:
fputs ("gt", file);   
break;
case GE:
fputs ("ge", file);   
break;
case LTU:
fputs ("llt", file);  
break;
case LEU:
fputs ("lle", file);  
break;
case GTU:
fputs ("lgt", file);  
break;
case GEU:
fputs ("lge", file);  
break;
default:
gcc_unreachable ();
}
break;
case 'w':
if (INT_P (x))
fprintf (file, HOST_WIDE_INT_PRINT_DEC,
((INTVAL (x) & 0xffff) ^ 0x8000) - 0x8000);
else
print_operand (file, x, 0);
return;
case 'x':
if (GET_CODE (x) != REG || !VSX_REGNO_P (REGNO (x)))
output_operand_lossage ("invalid %%x value");
else
{
int reg = REGNO (x);
int vsx_reg = (FP_REGNO_P (reg)
? reg - 32
: reg - FIRST_ALTIVEC_REGNO + 32);
#ifdef TARGET_REGNAMES      
if (TARGET_REGNAMES)
fprintf (file, "%%vs%d", vsx_reg);
else
#endif
fprintf (file, "%d", vsx_reg);
}
return;
case 'X':
if (MEM_P (x)
&& (legitimate_indexed_address_p (XEXP (x, 0), 0)
|| (GET_CODE (XEXP (x, 0)) == PRE_MODIFY
&& legitimate_indexed_address_p (XEXP (XEXP (x, 0), 1), 0))))
putc ('x', file);
return;
case 'Y':
if (REG_P (x))
fputs (reg_names[REGNO (x) + 2], file);
else if (MEM_P (x))
{
machine_mode mode = GET_MODE (x);
if (GET_CODE (XEXP (x, 0)) == PRE_INC
|| GET_CODE (XEXP (x, 0)) == PRE_DEC)
output_address (mode, plus_constant (Pmode,
XEXP (XEXP (x, 0), 0), 8));
else if (GET_CODE (XEXP (x, 0)) == PRE_MODIFY)
output_address (mode, plus_constant (Pmode,
XEXP (XEXP (x, 0), 0), 8));
else
output_address (mode, XEXP (adjust_address_nv (x, SImode, 8), 0));
if (small_data_operand (x, GET_MODE (x)))
fprintf (file, "@%s(%s)", SMALL_DATA_RELOC,
reg_names[SMALL_DATA_REG]);
}
return;
case 'z':
gcc_assert (GET_CODE (x) == SYMBOL_REF);
if (TARGET_MACHO)
{
const char *name = XSTR (x, 0);
#if TARGET_MACHO
if (darwin_emit_branch_islands
&& MACHOPIC_INDIRECT
&& machopic_classify_symbol (x) == MACHOPIC_UNDEFINED_FUNCTION)
name = machopic_indirection_name (x, true);
#endif
assemble_name (file, name);
}
else if (!DOT_SYMBOLS)
assemble_name (file, XSTR (x, 0));
else
rs6000_output_function_entry (file, XSTR (x, 0));
return;
case 'Z':
if (REG_P (x))
fputs (reg_names[REGNO (x) + 3], file);
else if (MEM_P (x))
{
machine_mode mode = GET_MODE (x);
if (GET_CODE (XEXP (x, 0)) == PRE_INC
|| GET_CODE (XEXP (x, 0)) == PRE_DEC)
output_address (mode, plus_constant (Pmode,
XEXP (XEXP (x, 0), 0), 12));
else if (GET_CODE (XEXP (x, 0)) == PRE_MODIFY)
output_address (mode, plus_constant (Pmode,
XEXP (XEXP (x, 0), 0), 12));
else
output_address (mode, XEXP (adjust_address_nv (x, SImode, 12), 0));
if (small_data_operand (x, GET_MODE (x)))
fprintf (file, "@%s(%s)", SMALL_DATA_RELOC,
reg_names[SMALL_DATA_REG]);
}
return;
case 'y':
{
rtx tmp;
gcc_assert (MEM_P (x));
tmp = XEXP (x, 0);
if (VECTOR_MEM_ALTIVEC_OR_VSX_P (GET_MODE (x))
&& GET_CODE (tmp) == AND
&& GET_CODE (XEXP (tmp, 1)) == CONST_INT
&& INTVAL (XEXP (tmp, 1)) == -16)
tmp = XEXP (tmp, 0);
else if (VECTOR_MEM_VSX_P (GET_MODE (x))
&& GET_CODE (tmp) == PRE_MODIFY)
tmp = XEXP (tmp, 1);
if (REG_P (tmp))
fprintf (file, "0,%s", reg_names[REGNO (tmp)]);
else
{
if (GET_CODE (tmp) != PLUS
|| !REG_P (XEXP (tmp, 0))
|| !REG_P (XEXP (tmp, 1)))
{
output_operand_lossage ("invalid %%y value, try using the 'Z' constraint");
break;
}
if (REGNO (XEXP (tmp, 0)) == 0)
fprintf (file, "%s,%s", reg_names[ REGNO (XEXP (tmp, 1)) ],
reg_names[ REGNO (XEXP (tmp, 0)) ]);
else
fprintf (file, "%s,%s", reg_names[ REGNO (XEXP (tmp, 0)) ],
reg_names[ REGNO (XEXP (tmp, 1)) ]);
}
break;
}
case 0:
if (REG_P (x))
fprintf (file, "%s", reg_names[REGNO (x)]);
else if (MEM_P (x))
{
if (GET_CODE (XEXP (x, 0)) == PRE_INC)
fprintf (file, "%d(%s)", GET_MODE_SIZE (GET_MODE (x)),
reg_names[REGNO (XEXP (XEXP (x, 0), 0))]);
else if (GET_CODE (XEXP (x, 0)) == PRE_DEC)
fprintf (file, "%d(%s)", - GET_MODE_SIZE (GET_MODE (x)),
reg_names[REGNO (XEXP (XEXP (x, 0), 0))]);
else if (GET_CODE (XEXP (x, 0)) == PRE_MODIFY)
output_address (GET_MODE (x), XEXP (XEXP (x, 0), 1));
else
output_address (GET_MODE (x), XEXP (x, 0));
}
else
{
if (toc_relative_expr_p (x, false, &tocrel_base_oac, &tocrel_offset_oac))
output_addr_const (file, CONST_CAST_RTX (tocrel_base_oac));
else
output_addr_const (file, x);
}
return;
case '&':
if (const char *name = get_some_local_dynamic_name ())
assemble_name (file, name);
else
output_operand_lossage ("'%%&' used without any "
"local dynamic TLS references");
return;
default:
output_operand_lossage ("invalid %%xn code");
}
}

void
print_operand_address (FILE *file, rtx x)
{
if (REG_P (x))
fprintf (file, "0(%s)", reg_names[ REGNO (x) ]);
else if (GET_CODE (x) == SYMBOL_REF || GET_CODE (x) == CONST
|| GET_CODE (x) == LABEL_REF)
{
output_addr_const (file, x);
if (small_data_operand (x, GET_MODE (x)))
fprintf (file, "@%s(%s)", SMALL_DATA_RELOC,
reg_names[SMALL_DATA_REG]);
else
gcc_assert (!TARGET_TOC);
}
else if (GET_CODE (x) == PLUS && REG_P (XEXP (x, 0))
&& REG_P (XEXP (x, 1)))
{
if (REGNO (XEXP (x, 0)) == 0)
fprintf (file, "%s,%s", reg_names[ REGNO (XEXP (x, 1)) ],
reg_names[ REGNO (XEXP (x, 0)) ]);
else
fprintf (file, "%s,%s", reg_names[ REGNO (XEXP (x, 0)) ],
reg_names[ REGNO (XEXP (x, 1)) ]);
}
else if (GET_CODE (x) == PLUS && REG_P (XEXP (x, 0))
&& GET_CODE (XEXP (x, 1)) == CONST_INT)
fprintf (file, HOST_WIDE_INT_PRINT_DEC "(%s)",
INTVAL (XEXP (x, 1)), reg_names[ REGNO (XEXP (x, 0)) ]);
#if TARGET_MACHO
else if (GET_CODE (x) == LO_SUM && REG_P (XEXP (x, 0))
&& CONSTANT_P (XEXP (x, 1)))
{
fprintf (file, "lo16(");
output_addr_const (file, XEXP (x, 1));
fprintf (file, ")(%s)", reg_names[ REGNO (XEXP (x, 0)) ]);
}
#endif
#if TARGET_ELF
else if (GET_CODE (x) == LO_SUM && REG_P (XEXP (x, 0))
&& CONSTANT_P (XEXP (x, 1)))
{
output_addr_const (file, XEXP (x, 1));
fprintf (file, "@l(%s)", reg_names[ REGNO (XEXP (x, 0)) ]);
}
#endif
else if (toc_relative_expr_p (x, false, &tocrel_base_oac, &tocrel_offset_oac))
{
output_addr_const (file, CONST_CAST_RTX (tocrel_base_oac));
if (GET_CODE (x) == LO_SUM)
fprintf (file, "@l(%s)", reg_names[REGNO (XEXP (x, 0))]);
else
fprintf (file, "(%s)", reg_names[REGNO (XVECEXP (tocrel_base_oac, 0, 1))]);
}
else
gcc_unreachable ();
}

static bool
rs6000_output_addr_const_extra (FILE *file, rtx x)
{
if (GET_CODE (x) == UNSPEC)
switch (XINT (x, 1))
{
case UNSPEC_TOCREL:
gcc_checking_assert (GET_CODE (XVECEXP (x, 0, 0)) == SYMBOL_REF
&& REG_P (XVECEXP (x, 0, 1))
&& REGNO (XVECEXP (x, 0, 1)) == TOC_REGISTER);
output_addr_const (file, XVECEXP (x, 0, 0));
if (x == tocrel_base_oac && tocrel_offset_oac != const0_rtx)
{
if (INTVAL (tocrel_offset_oac) >= 0)
fprintf (file, "+");
output_addr_const (file, CONST_CAST_RTX (tocrel_offset_oac));
}
if (!TARGET_AIX || (TARGET_ELF && TARGET_MINIMAL_TOC))
{
putc ('-', file);
assemble_name (file, toc_label_name);
need_toc_init = 1;
}
else if (TARGET_ELF)
fputs ("@toc", file);
return true;
#if TARGET_MACHO
case UNSPEC_MACHOPIC_OFFSET:
output_addr_const (file, XVECEXP (x, 0, 0));
putc ('-', file);
machopic_output_function_base_name (file);
return true;
#endif
}
return false;
}

static bool
rs6000_assemble_integer (rtx x, unsigned int size, int aligned_p)
{
#ifdef RELOCATABLE_NEEDS_FIXUP
if (RELOCATABLE_NEEDS_FIXUP && size == 4 && aligned_p)
{
static int recurse = 0;
if (DEFAULT_ABI == ABI_V4
&& (TARGET_RELOCATABLE || flag_pic > 1)
&& in_section != toc_section
&& !recurse
&& !CONST_SCALAR_INT_P (x)
&& CONSTANT_P (x))
{
char buf[256];
recurse = 1;
ASM_GENERATE_INTERNAL_LABEL (buf, "LCP", fixuplabelno);
fixuplabelno++;
ASM_OUTPUT_LABEL (asm_out_file, buf);
fprintf (asm_out_file, "\t.long\t(");
output_addr_const (asm_out_file, x);
fprintf (asm_out_file, ")@fixup\n");
fprintf (asm_out_file, "\t.section\t\".fixup\",\"aw\"\n");
ASM_OUTPUT_ALIGN (asm_out_file, 2);
fprintf (asm_out_file, "\t.long\t");
assemble_name (asm_out_file, buf);
fprintf (asm_out_file, "\n\t.previous\n");
recurse = 0;
return true;
}
else if (GET_CODE (x) == SYMBOL_REF
&& XSTR (x, 0)[0] == '.'
&& DEFAULT_ABI == ABI_AIX)
{
const char *name = XSTR (x, 0);
while (*name == '.')
name++;
fprintf (asm_out_file, "\t.long\t%s\n", name);
return true;
}
}
#endif 
return default_assemble_integer (x, size, aligned_p);
}
#if defined (HAVE_GAS_HIDDEN) && !TARGET_MACHO
static void
rs6000_assemble_visibility (tree decl, int vis)
{
if (TARGET_XCOFF)
return;
if (DEFAULT_ABI == ABI_AIX
&& DOT_SYMBOLS
&& TREE_CODE (decl) == FUNCTION_DECL)
{
static const char * const visibility_types[] = {
NULL, "protected", "hidden", "internal"
};
const char *name, *type;
name = ((* targetm.strip_name_encoding)
(IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl))));
type = visibility_types[vis];
fprintf (asm_out_file, "\t.%s\t%s\n", type, name);
fprintf (asm_out_file, "\t.%s\t.%s\n", type, name);
}
else
default_assemble_visibility (decl, vis);
}
#endif

enum rtx_code
rs6000_reverse_condition (machine_mode mode, enum rtx_code code)
{
if (mode == CCFPmode
&& (!flag_finite_math_only
|| code == UNLT || code == UNLE || code == UNGT || code == UNGE
|| code == UNEQ || code == LTGT))
return reverse_condition_maybe_unordered (code);
else
return reverse_condition (code);
}
static rtx
rs6000_generate_compare (rtx cmp, machine_mode mode)
{
machine_mode comp_mode;
rtx compare_result;
enum rtx_code code = GET_CODE (cmp);
rtx op0 = XEXP (cmp, 0);
rtx op1 = XEXP (cmp, 1);
if (!TARGET_FLOAT128_HW && FLOAT128_VECTOR_P (mode))
comp_mode = CCmode;
else if (FLOAT_MODE_P (mode))
comp_mode = CCFPmode;
else if (code == GTU || code == LTU
|| code == GEU || code == LEU)
comp_mode = CCUNSmode;
else if ((code == EQ || code == NE)
&& unsigned_reg_p (op0)
&& (unsigned_reg_p (op1)
|| (CONST_INT_P (op1) && INTVAL (op1) != 0)))
comp_mode = CCUNSmode;
else
comp_mode = CCmode;
if (comp_mode == CCUNSmode && GET_CODE (op1) == CONST_INT
&& INTVAL (op1) < 0)
{
op0 = copy_rtx_if_shared (op0);
op1 = force_reg (GET_MODE (op0), op1);
cmp = gen_rtx_fmt_ee (code, GET_MODE (cmp), op0, op1);
}
compare_result = gen_reg_rtx (comp_mode);
if (!TARGET_FLOAT128_HW && FLOAT128_VECTOR_P (mode))
{
rtx libfunc = NULL_RTX;
bool check_nan = false;
rtx dest;
switch (code)
{
case EQ:
case NE:
libfunc = optab_libfunc (eq_optab, mode);
break;
case GT:
case GE:
libfunc = optab_libfunc (ge_optab, mode);
break;
case LT:
case LE:
libfunc = optab_libfunc (le_optab, mode);
break;
case UNORDERED:
case ORDERED:
libfunc = optab_libfunc (unord_optab, mode);
code = (code == UNORDERED) ? NE : EQ;
break;
case UNGE:
case UNGT:
check_nan = true;
libfunc = optab_libfunc (ge_optab, mode);
code = (code == UNGE) ? GE : GT;
break;
case UNLE:
case UNLT:
check_nan = true;
libfunc = optab_libfunc (le_optab, mode);
code = (code == UNLE) ? LE : LT;
break;
case UNEQ:
case LTGT:
check_nan = true;
libfunc = optab_libfunc (eq_optab, mode);
code = (code = UNEQ) ? EQ : NE;
break;
default:
gcc_unreachable ();
}
gcc_assert (libfunc);
if (!check_nan)
dest = emit_library_call_value (libfunc, NULL_RTX, LCT_CONST,
SImode, op0, mode, op1, mode);
else
{
rtx ne_rtx, normal_dest, unord_dest;
rtx unord_func = optab_libfunc (unord_optab, mode);
rtx join_label = gen_label_rtx ();
rtx join_ref = gen_rtx_LABEL_REF (VOIDmode, join_label);
rtx unord_cmp = gen_reg_rtx (comp_mode);
gcc_assert (unord_func);
unord_dest = emit_library_call_value (unord_func, NULL_RTX, LCT_CONST,
SImode, op0, mode, op1, mode);
dest = gen_reg_rtx (SImode);
emit_move_insn (dest, const1_rtx);
emit_insn (gen_rtx_SET (unord_cmp,
gen_rtx_COMPARE (comp_mode, unord_dest,
const0_rtx)));
ne_rtx = gen_rtx_NE (comp_mode, unord_cmp, const0_rtx);
emit_jump_insn (gen_rtx_SET (pc_rtx,
gen_rtx_IF_THEN_ELSE (VOIDmode, ne_rtx,
join_ref,
pc_rtx)));
normal_dest = emit_library_call_value (libfunc, NULL_RTX, LCT_CONST,
SImode, op0, mode, op1, mode);
emit_insn (gen_cstoresi4 (dest,
gen_rtx_fmt_ee (code, SImode, normal_dest,
const0_rtx),
normal_dest, const0_rtx));
emit_label (join_label);
code = NE;
}
emit_insn (gen_rtx_SET (compare_result,
gen_rtx_COMPARE (comp_mode, dest, const0_rtx)));
}
else
{
if (comp_mode == CCFPmode && TARGET_XL_COMPAT
&& FLOAT128_IBM_P (GET_MODE (op0))
&& TARGET_HARD_FLOAT)
emit_insn (gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (10,
gen_rtx_SET (compare_result,
gen_rtx_COMPARE (comp_mode, op0, op1)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (DFmode)),
gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (Pmode)))));
else if (GET_CODE (op1) == UNSPEC
&& XINT (op1, 1) == UNSPEC_SP_TEST)
{
rtx op1b = XVECEXP (op1, 0, 0);
comp_mode = CCEQmode;
compare_result = gen_reg_rtx (CCEQmode);
if (TARGET_64BIT)
emit_insn (gen_stack_protect_testdi (compare_result, op0, op1b));
else
emit_insn (gen_stack_protect_testsi (compare_result, op0, op1b));
}
else
emit_insn (gen_rtx_SET (compare_result,
gen_rtx_COMPARE (comp_mode, op0, op1)));
}
if (FLOAT_MODE_P (mode)
&& (!FLOAT128_IEEE_P (mode) || TARGET_FLOAT128_HW)
&& !flag_finite_math_only
&& (code == LE || code == GE
|| code == UNEQ || code == LTGT
|| code == UNGT || code == UNLT))
{
enum rtx_code or1, or2;
rtx or1_rtx, or2_rtx, compare2_rtx;
rtx or_result = gen_reg_rtx (CCEQmode);
switch (code)
{
case LE: or1 = LT;  or2 = EQ;  break;
case GE: or1 = GT;  or2 = EQ;  break;
case UNEQ: or1 = UNORDERED;  or2 = EQ;  break;
case LTGT: or1 = LT;  or2 = GT;  break;
case UNGT: or1 = UNORDERED;  or2 = GT;  break;
case UNLT: or1 = UNORDERED;  or2 = LT;  break;
default:  gcc_unreachable ();
}
validate_condition_mode (or1, comp_mode);
validate_condition_mode (or2, comp_mode);
or1_rtx = gen_rtx_fmt_ee (or1, SImode, compare_result, const0_rtx);
or2_rtx = gen_rtx_fmt_ee (or2, SImode, compare_result, const0_rtx);
compare2_rtx = gen_rtx_COMPARE (CCEQmode,
gen_rtx_IOR (SImode, or1_rtx, or2_rtx),
const_true_rtx);
emit_insn (gen_rtx_SET (or_result, compare2_rtx));
compare_result = or_result;
code = EQ;
}
validate_condition_mode (code, GET_MODE (compare_result));
return gen_rtx_fmt_ee (code, VOIDmode, compare_result, const0_rtx);
}

static const char*
rs6000_invalid_binary_op (int op ATTRIBUTE_UNUSED,
const_tree type1,
const_tree type2)
{
machine_mode mode1 = TYPE_MODE (type1);
machine_mode mode2 = TYPE_MODE (type2);
if (COMPLEX_MODE_P (mode1))
mode1 = GET_MODE_INNER (mode1);
if (COMPLEX_MODE_P (mode2))
mode2 = GET_MODE_INNER (mode2);
if (mode1 == mode2)
return NULL;
if (!TARGET_FLOAT128_CVT)
{
if ((mode1 == KFmode && mode2 == IFmode)
|| (mode1 == IFmode && mode2 == KFmode))
return N_("__float128 and __ibm128 cannot be used in the same "
"expression");
if (TARGET_IEEEQUAD
&& ((mode1 == IFmode && mode2 == TFmode)
|| (mode1 == TFmode && mode2 == IFmode)))
return N_("__ibm128 and long double cannot be used in the same "
"expression");
if (!TARGET_IEEEQUAD
&& ((mode1 == KFmode && mode2 == TFmode)
|| (mode1 == TFmode && mode2 == KFmode)))
return N_("__float128 and long double cannot be used in the same "
"expression");
}
return NULL;
}

void
rs6000_expand_float128_convert (rtx dest, rtx src, bool unsigned_p)
{
machine_mode dest_mode = GET_MODE (dest);
machine_mode src_mode = GET_MODE (src);
convert_optab cvt = unknown_optab;
bool do_move = false;
rtx libfunc = NULL_RTX;
rtx dest2;
typedef rtx (*rtx_2func_t) (rtx, rtx);
rtx_2func_t hw_convert = (rtx_2func_t)0;
size_t kf_or_tf;
struct hw_conv_t {
rtx_2func_t	from_df;
rtx_2func_t from_sf;
rtx_2func_t from_si_sign;
rtx_2func_t from_si_uns;
rtx_2func_t from_di_sign;
rtx_2func_t from_di_uns;
rtx_2func_t to_df;
rtx_2func_t to_sf;
rtx_2func_t to_si_sign;
rtx_2func_t to_si_uns;
rtx_2func_t to_di_sign;
rtx_2func_t to_di_uns;
} hw_conversions[2] = {
{
gen_extenddfkf2_hw,		
gen_extendsfkf2_hw,		
gen_float_kfsi2_hw,		
gen_floatuns_kfsi2_hw,		
gen_float_kfdi2_hw,		
gen_floatuns_kfdi2_hw,		
gen_trunckfdf2_hw,		
gen_trunckfsf2_hw,		
gen_fix_kfsi2_hw,			
gen_fixuns_kfsi2_hw,		
gen_fix_kfdi2_hw,			
gen_fixuns_kfdi2_hw,		
},
{
gen_extenddftf2_hw,		
gen_extendsftf2_hw,		
gen_float_tfsi2_hw,		
gen_floatuns_tfsi2_hw,		
gen_float_tfdi2_hw,		
gen_floatuns_tfdi2_hw,		
gen_trunctfdf2_hw,		
gen_trunctfsf2_hw,		
gen_fix_tfsi2_hw,			
gen_fixuns_tfsi2_hw,		
gen_fix_tfdi2_hw,			
gen_fixuns_tfdi2_hw,		
},
};
if (dest_mode == src_mode)
gcc_unreachable ();
if (MEM_P (src))
src = force_reg (src_mode, src);
if (MEM_P (dest))
{
rtx tmp = gen_reg_rtx (dest_mode);
rs6000_expand_float128_convert (tmp, src, unsigned_p);
rs6000_emit_move (dest, tmp, dest_mode);
return;
}
if (FLOAT128_IEEE_P (dest_mode))
{
if (dest_mode == KFmode)
kf_or_tf = 0;
else if (dest_mode == TFmode)
kf_or_tf = 1;
else
gcc_unreachable ();
switch (src_mode)
{
case E_DFmode:
cvt = sext_optab;
hw_convert = hw_conversions[kf_or_tf].from_df;
break;
case E_SFmode:
cvt = sext_optab;
hw_convert = hw_conversions[kf_or_tf].from_sf;
break;
case E_KFmode:
case E_IFmode:
case E_TFmode:
if (FLOAT128_IBM_P (src_mode))
cvt = sext_optab;
else
do_move = true;
break;
case E_SImode:
if (unsigned_p)
{
cvt = ufloat_optab;
hw_convert = hw_conversions[kf_or_tf].from_si_uns;
}
else
{
cvt = sfloat_optab;
hw_convert = hw_conversions[kf_or_tf].from_si_sign;
}
break;
case E_DImode:
if (unsigned_p)
{
cvt = ufloat_optab;
hw_convert = hw_conversions[kf_or_tf].from_di_uns;
}
else
{
cvt = sfloat_optab;
hw_convert = hw_conversions[kf_or_tf].from_di_sign;
}
break;
default:
gcc_unreachable ();
}
}
else if (FLOAT128_IEEE_P (src_mode))
{
if (src_mode == KFmode)
kf_or_tf = 0;
else if (src_mode == TFmode)
kf_or_tf = 1;
else
gcc_unreachable ();
switch (dest_mode)
{
case E_DFmode:
cvt = trunc_optab;
hw_convert = hw_conversions[kf_or_tf].to_df;
break;
case E_SFmode:
cvt = trunc_optab;
hw_convert = hw_conversions[kf_or_tf].to_sf;
break;
case E_KFmode:
case E_IFmode:
case E_TFmode:
if (FLOAT128_IBM_P (dest_mode))
cvt = trunc_optab;
else
do_move = true;
break;
case E_SImode:
if (unsigned_p)
{
cvt = ufix_optab;
hw_convert = hw_conversions[kf_or_tf].to_si_uns;
}
else
{
cvt = sfix_optab;
hw_convert = hw_conversions[kf_or_tf].to_si_sign;
}
break;
case E_DImode:
if (unsigned_p)
{
cvt = ufix_optab;
hw_convert = hw_conversions[kf_or_tf].to_di_uns;
}
else
{
cvt = sfix_optab;
hw_convert = hw_conversions[kf_or_tf].to_di_sign;
}
break;
default:
gcc_unreachable ();
}
}
else if (FLOAT128_IBM_P (dest_mode) && FLOAT128_IBM_P (src_mode))
do_move = true;
else
gcc_unreachable ();
if (do_move)
emit_insn (gen_rtx_SET (dest, gen_rtx_FLOAT_EXTEND (dest_mode, src)));
else if (TARGET_FLOAT128_HW && hw_convert)
emit_insn ((hw_convert) (dest, src));
else if (cvt != unknown_optab)
{
libfunc = convert_optab_libfunc (cvt, dest_mode, src_mode);
gcc_assert (libfunc != NULL_RTX);
dest2 = emit_library_call_value (libfunc, dest, LCT_CONST, dest_mode,
src, src_mode);
gcc_assert (dest2 != NULL_RTX);
if (!rtx_equal_p (dest, dest2))
emit_move_insn (dest, dest2);
}
else
gcc_unreachable ();
return;
}

rtx
rs6000_emit_eqne (machine_mode mode, rtx op1, rtx op2, rtx scratch)
{
if (op2 == const0_rtx)
return op1;
if (GET_CODE (scratch) == SCRATCH)
scratch = gen_reg_rtx (mode);
if (logical_operand (op2, mode))
emit_insn (gen_rtx_SET (scratch, gen_rtx_XOR (mode, op1, op2)));
else
emit_insn (gen_rtx_SET (scratch,
gen_rtx_PLUS (mode, op1, negate_rtx (mode, op2))));
return scratch;
}
void
rs6000_emit_sCOND (machine_mode mode, rtx operands[])
{
rtx condition_rtx;
machine_mode op_mode;
enum rtx_code cond_code;
rtx result = operands[0];
condition_rtx = rs6000_generate_compare (operands[1], mode);
cond_code = GET_CODE (condition_rtx);
if (cond_code == NE
|| cond_code == GE || cond_code == LE
|| cond_code == GEU || cond_code == LEU
|| cond_code == ORDERED || cond_code == UNGE || cond_code == UNLE)
{
rtx not_result = gen_reg_rtx (CCEQmode);
rtx not_op, rev_cond_rtx;
machine_mode cc_mode;
cc_mode = GET_MODE (XEXP (condition_rtx, 0));
rev_cond_rtx = gen_rtx_fmt_ee (rs6000_reverse_condition (cc_mode, cond_code),
SImode, XEXP (condition_rtx, 0), const0_rtx);
not_op = gen_rtx_COMPARE (CCEQmode, rev_cond_rtx, const0_rtx);
emit_insn (gen_rtx_SET (not_result, not_op));
condition_rtx = gen_rtx_EQ (VOIDmode, not_result, const0_rtx);
}
op_mode = GET_MODE (XEXP (operands[1], 0));
if (op_mode == VOIDmode)
op_mode = GET_MODE (XEXP (operands[1], 1));
if (TARGET_POWERPC64 && (op_mode == DImode || FLOAT_MODE_P (mode)))
{
PUT_MODE (condition_rtx, DImode);
convert_move (result, condition_rtx, 0);
}
else
{
PUT_MODE (condition_rtx, SImode);
emit_insn (gen_rtx_SET (result, condition_rtx));
}
}
void
rs6000_emit_cbranch (machine_mode mode, rtx operands[])
{
rtx condition_rtx, loc_ref;
condition_rtx = rs6000_generate_compare (operands[0], mode);
loc_ref = gen_rtx_LABEL_REF (VOIDmode, operands[3]);
emit_jump_insn (gen_rtx_SET (pc_rtx,
gen_rtx_IF_THEN_ELSE (VOIDmode, condition_rtx,
loc_ref, pc_rtx)));
}
char *
output_cbranch (rtx op, const char *label, int reversed, rtx_insn *insn)
{
static char string[64];
enum rtx_code code = GET_CODE (op);
rtx cc_reg = XEXP (op, 0);
machine_mode mode = GET_MODE (cc_reg);
int cc_regno = REGNO (cc_reg) - CR0_REGNO;
int need_longbranch = label != NULL && get_attr_length (insn) == 8;
int really_reversed = reversed ^ need_longbranch;
char *s = string;
const char *ccode;
const char *pred;
rtx note;
validate_condition_mode (code, mode);
if (really_reversed)
{
if (mode == CCFPmode)
code = reverse_condition_maybe_unordered (code);
else
code = reverse_condition (code);
}
switch (code)
{
case NE: case LTGT:
ccode = "ne"; break;
case EQ: case UNEQ:
ccode = "eq"; break;
case GE: case GEU:
ccode = "ge"; break;
case GT: case GTU: case UNGT:
ccode = "gt"; break;
case LE: case LEU:
ccode = "le"; break;
case LT: case LTU: case UNLT:
ccode = "lt"; break;
case UNORDERED: ccode = "un"; break;
case ORDERED: ccode = "nu"; break;
case UNGE: ccode = "nl"; break;
case UNLE: ccode = "ng"; break;
default:
gcc_unreachable ();
}
pred = "";
note = find_reg_note (insn, REG_BR_PROB, NULL_RTX);
if (note != NULL_RTX)
{
int prob = profile_probability::from_reg_br_prob_note (XINT (note, 0))
.to_reg_br_prob_base () - REG_BR_PROB_BASE / 2;
if (rs6000_always_hint
|| (abs (prob) > REG_BR_PROB_BASE / 100 * 48
&& (profile_status_for_fn (cfun) != PROFILE_GUESSED)
&& br_prob_note_reliable_p (note)))
{
if (abs (prob) > REG_BR_PROB_BASE / 20
&& ((prob > 0) ^ need_longbranch))
pred = "+";
else
pred = "-";
}
}
if (label == NULL)
s += sprintf (s, "b%slr%s ", ccode, pred);
else
s += sprintf (s, "b%s%s ", ccode, pred);
if (reg_names[cc_regno + CR0_REGNO][0] == '%')
*s++ = '%';
s += sprintf (s, "%s", reg_names[cc_regno + CR0_REGNO]);
if (label != NULL)
{
if (need_longbranch)
s += sprintf (s, ",$+8\n\tb %s", label);
else
s += sprintf (s, ",%s", label);
}
return string;
}
static rtx
rs6000_emit_vector_compare_inner (enum rtx_code code, rtx op0, rtx op1)
{
rtx mask;
machine_mode mode = GET_MODE (op0);
switch (code)
{
default:
break;
case GE:
if (GET_MODE_CLASS (mode) == MODE_VECTOR_INT)
return NULL_RTX;
case EQ:
case GT:
case GTU:
case ORDERED:
case UNORDERED:
case UNEQ:
case LTGT:
mask = gen_reg_rtx (mode);
emit_insn (gen_rtx_SET (mask, gen_rtx_fmt_ee (code, mode, op0, op1)));
return mask;
}
return NULL_RTX;
}
static rtx
rs6000_emit_vector_compare (enum rtx_code rcode,
rtx op0, rtx op1,
machine_mode dmode)
{
rtx mask;
bool swap_operands = false;
bool try_again = false;
gcc_assert (VECTOR_UNIT_ALTIVEC_OR_VSX_P (dmode));
gcc_assert (GET_MODE (op0) == GET_MODE (op1));
mask = rs6000_emit_vector_compare_inner (rcode, op0, op1);
if (mask)
return mask;
switch (rcode)
{
case LT:
rcode = GT;
swap_operands = true;
try_again = true;
break;
case LTU:
rcode = GTU;
swap_operands = true;
try_again = true;
break;
case NE:
case UNLE:
case UNLT:
case UNGE:
case UNGT:
{
enum rtx_code rev_code;
enum insn_code nor_code;
rtx mask2;
rev_code = reverse_condition_maybe_unordered (rcode);
if (rev_code == UNKNOWN)
return NULL_RTX;
nor_code = optab_handler (one_cmpl_optab, dmode);
if (nor_code == CODE_FOR_nothing)
return NULL_RTX;
mask2 = rs6000_emit_vector_compare (rev_code, op0, op1, dmode);
if (!mask2)
return NULL_RTX;
mask = gen_reg_rtx (dmode);
emit_insn (GEN_FCN (nor_code) (mask, mask2));
return mask;
}
break;
case GE:
case GEU:
case LE:
case LEU:
{
rtx c_rtx, eq_rtx;
enum insn_code ior_code;
enum rtx_code new_code;
switch (rcode)
{
case  GE:
new_code = GT;
break;
case GEU:
new_code = GTU;
break;
case LE:
new_code = LT;
break;
case LEU:
new_code = LTU;
break;
default:
gcc_unreachable ();
}
ior_code = optab_handler (ior_optab, dmode);
if (ior_code == CODE_FOR_nothing)
return NULL_RTX;
c_rtx = rs6000_emit_vector_compare (new_code, op0, op1, dmode);
if (!c_rtx)
return NULL_RTX;
eq_rtx = rs6000_emit_vector_compare (EQ, op0, op1, dmode);
if (!eq_rtx)
return NULL_RTX;
mask = gen_reg_rtx (dmode);
emit_insn (GEN_FCN (ior_code) (mask, c_rtx, eq_rtx));
return mask;
}
break;
default:
return NULL_RTX;
}
if (try_again)
{
if (swap_operands)
std::swap (op0, op1);
mask = rs6000_emit_vector_compare_inner (rcode, op0, op1);
if (mask)
return mask;
}
return NULL_RTX;
}
int
rs6000_emit_vector_cond_expr (rtx dest, rtx op_true, rtx op_false,
rtx cond, rtx cc_op0, rtx cc_op1)
{
machine_mode dest_mode = GET_MODE (dest);
machine_mode mask_mode = GET_MODE (cc_op0);
enum rtx_code rcode = GET_CODE (cond);
machine_mode cc_mode = CCmode;
rtx mask;
rtx cond2;
bool invert_move = false;
if (VECTOR_UNIT_NONE_P (dest_mode))
return 0;
gcc_assert (GET_MODE_SIZE (dest_mode) == GET_MODE_SIZE (mask_mode)
&& GET_MODE_NUNITS (dest_mode) == GET_MODE_NUNITS (mask_mode));
switch (rcode)
{
case NE:
case UNLE:
case UNLT:
case UNGE:
case UNGT:
invert_move = true;
rcode = reverse_condition_maybe_unordered (rcode);
if (rcode == UNKNOWN)
return 0;
break;
case GE:
case LE:
if (GET_MODE_CLASS (mask_mode) == MODE_VECTOR_INT)
{
invert_move = true;
rcode = reverse_condition (rcode);
}
break;
case GTU:
case GEU:
case LTU:
case LEU:
cc_mode = CCUNSmode;
if (rcode == GEU || rcode == LEU)
{
invert_move = true;
rcode = reverse_condition (rcode);
}
break;
default:
break;
}
mask = rs6000_emit_vector_compare (rcode, cc_op0, cc_op1, mask_mode);
if (!mask)
return 0;
if (invert_move)
std::swap (op_true, op_false);
if (GET_MODE_CLASS (dest_mode) == MODE_VECTOR_INT
&& (GET_CODE (op_true) == CONST_VECTOR
|| GET_CODE (op_false) == CONST_VECTOR))
{
rtx constant_0 = CONST0_RTX (dest_mode);
rtx constant_m1 = CONSTM1_RTX (dest_mode);
if (op_true == constant_m1 && op_false == constant_0)
{
emit_move_insn (dest, mask);
return 1;
}
else if (op_true == constant_0 && op_false == constant_m1)
{
emit_insn (gen_rtx_SET (dest, gen_rtx_NOT (dest_mode, mask)));
return 1;
}
if (op_true == constant_m1)
op_true = mask;
if (op_false == constant_0)
op_false = mask;
}
if (!REG_P (op_true) && !SUBREG_P (op_true))
op_true = force_reg (dest_mode, op_true);
if (!REG_P (op_false) && !SUBREG_P (op_false))
op_false = force_reg (dest_mode, op_false);
cond2 = gen_rtx_fmt_ee (NE, cc_mode, gen_lowpart (dest_mode, mask),
CONST0_RTX (dest_mode));
emit_insn (gen_rtx_SET (dest,
gen_rtx_IF_THEN_ELSE (dest_mode,
cond2,
op_true,
op_false)));
return 1;
}
static int
rs6000_emit_p9_fp_minmax (rtx dest, rtx op, rtx true_cond, rtx false_cond)
{
enum rtx_code code = GET_CODE (op);
rtx op0 = XEXP (op, 0);
rtx op1 = XEXP (op, 1);
machine_mode compare_mode = GET_MODE (op0);
machine_mode result_mode = GET_MODE (dest);
bool max_p = false;
if (result_mode != compare_mode)
return 0;
if (code == GE || code == GT)
max_p = true;
else if (code == LE || code == LT)
max_p = false;
else
return 0;
if (rtx_equal_p (op0, true_cond) && rtx_equal_p (op1, false_cond))
;
else if (rtx_equal_p (op1, true_cond) && rtx_equal_p (op0, false_cond))
max_p = !max_p;
else
return 0;
rs6000_emit_minmax (dest, max_p ? SMAX : SMIN, op0, op1);
return 1;
}
static int
rs6000_emit_p9_fp_cmove (rtx dest, rtx op, rtx true_cond, rtx false_cond)
{
enum rtx_code code = GET_CODE (op);
rtx op0 = XEXP (op, 0);
rtx op1 = XEXP (op, 1);
machine_mode result_mode = GET_MODE (dest);
rtx compare_rtx;
rtx cmove_rtx;
rtx clobber_rtx;
if (!can_create_pseudo_p ())
return 0;
switch (code)
{
case EQ:
case GE:
case GT:
break;
case NE:
case LT:
case LE:
code = swap_condition (code);
std::swap (op0, op1);
break;
default:
return 0;
}
compare_rtx = gen_rtx_fmt_ee (code, CCFPmode, op0, op1);
cmove_rtx = gen_rtx_SET (dest,
gen_rtx_IF_THEN_ELSE (result_mode,
compare_rtx,
true_cond,
false_cond));
clobber_rtx = gen_rtx_CLOBBER (VOIDmode, gen_rtx_SCRATCH (V2DImode));
emit_insn (gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (2, cmove_rtx, clobber_rtx)));
return 1;
}
int
rs6000_emit_cmove (rtx dest, rtx op, rtx true_cond, rtx false_cond)
{
enum rtx_code code = GET_CODE (op);
rtx op0 = XEXP (op, 0);
rtx op1 = XEXP (op, 1);
machine_mode compare_mode = GET_MODE (op0);
machine_mode result_mode = GET_MODE (dest);
rtx temp;
bool is_against_zero;
if (GET_MODE (op1) != compare_mode
&& (!TARGET_ISEL || !short_cint_operand (op1, VOIDmode)))
return 0;
if (GET_MODE (true_cond) != result_mode)
return 0;
if (GET_MODE (false_cond) != result_mode)
return 0;
if (TARGET_P9_MINMAX
&& (compare_mode == SFmode || compare_mode == DFmode)
&& (result_mode == SFmode || result_mode == DFmode))
{
if (rs6000_emit_p9_fp_minmax (dest, op, true_cond, false_cond))
return 1;
if (rs6000_emit_p9_fp_cmove (dest, op, true_cond, false_cond))
return 1;
}
if (FLOAT_MODE_P (compare_mode) && !FLOAT_MODE_P (result_mode))
return 0;
if (!FLOAT_MODE_P (compare_mode))
{
if (TARGET_ISEL)
return rs6000_emit_int_cmove (dest, op, true_cond, false_cond);
return 0;
}
is_against_zero = op1 == CONST0_RTX (compare_mode);
if (SCALAR_FLOAT_MODE_P (compare_mode)
&& flag_trapping_math && ! is_against_zero)
return 0;
if (code == UNLT || code == UNGT || code == UNORDERED || code == NE
|| code == LTGT || code == LT || code == UNLE)
{
code = reverse_condition_maybe_unordered (code);
temp = true_cond;
true_cond = false_cond;
false_cond = temp;
}
if (code == UNEQ && HONOR_NANS (compare_mode))
return 0;
if (HONOR_INFINITIES (compare_mode)
&& code != GT && code != UNGE
&& (GET_CODE (op1) != CONST_DOUBLE
|| real_isinf (CONST_DOUBLE_REAL_VALUE (op1)))
&& ((! rtx_equal_p (op0, false_cond) && ! rtx_equal_p (op1, false_cond))
|| (! rtx_equal_p (op0, true_cond)
&& ! rtx_equal_p (op1, true_cond))))
return 0;
if (! is_against_zero)
{
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp, gen_rtx_MINUS (compare_mode, op0, op1)));
op0 = temp;
op1 = CONST0_RTX (compare_mode);
}
if (! HONOR_NANS (compare_mode))
switch (code)
{
case GT:
code = LE;
temp = true_cond;
true_cond = false_cond;
false_cond = temp;
break;
case UNGE:
code = GE;
break;
case UNEQ:
code = EQ;
break;
default:
break;
}
switch (code)
{
case GE:
break;
case LE:
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp, gen_rtx_NEG (compare_mode, op0)));
op0 = temp;
break;
case ORDERED:
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp, gen_rtx_ABS (compare_mode, op0)));
op0 = temp;
break;
case EQ:
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp,
gen_rtx_NEG (compare_mode,
gen_rtx_ABS (compare_mode, op0))));
op0 = temp;
break;
case UNGE:
temp = gen_reg_rtx (result_mode);
emit_insn (gen_rtx_SET (temp,
gen_rtx_IF_THEN_ELSE (result_mode,
gen_rtx_GE (VOIDmode,
op0, op1),
true_cond, false_cond)));
false_cond = true_cond;
true_cond = temp;
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp, gen_rtx_NEG (compare_mode, op0)));
op0 = temp;
break;
case GT:
temp = gen_reg_rtx (result_mode);
emit_insn (gen_rtx_SET (temp,
gen_rtx_IF_THEN_ELSE (result_mode,
gen_rtx_GE (VOIDmode,
op0, op1),
true_cond, false_cond)));
true_cond = false_cond;
false_cond = temp;
temp = gen_reg_rtx (compare_mode);
emit_insn (gen_rtx_SET (temp, gen_rtx_NEG (compare_mode, op0)));
op0 = temp;
break;
default:
gcc_unreachable ();
}
emit_insn (gen_rtx_SET (dest,
gen_rtx_IF_THEN_ELSE (result_mode,
gen_rtx_GE (VOIDmode,
op0, op1),
true_cond, false_cond)));
return 1;
}
int
rs6000_emit_int_cmove (rtx dest, rtx op, rtx true_cond, rtx false_cond)
{
rtx condition_rtx, cr;
machine_mode mode = GET_MODE (dest);
enum rtx_code cond_code;
rtx (*isel_func) (rtx, rtx, rtx, rtx, rtx);
bool signedp;
if (mode != SImode && (!TARGET_POWERPC64 || mode != DImode))
return 0;
condition_rtx = rs6000_generate_compare (op, mode);
cond_code = GET_CODE (condition_rtx);
cr = XEXP (condition_rtx, 0);
signedp = GET_MODE (cr) == CCmode;
isel_func = (mode == SImode
? (signedp ? gen_isel_signed_si : gen_isel_unsigned_si)
: (signedp ? gen_isel_signed_di : gen_isel_unsigned_di));
switch (cond_code)
{
case LT: case GT: case LTU: case GTU: case EQ:
break;
default:
{
std::swap (false_cond, true_cond);
PUT_CODE (condition_rtx, reverse_condition (cond_code));
}
break;
}
false_cond = force_reg (mode, false_cond);
if (true_cond != const0_rtx)
true_cond = force_reg (mode, true_cond);
emit_insn (isel_func (dest, condition_rtx, true_cond, false_cond, cr));
return 1;
}
void
rs6000_emit_minmax (rtx dest, enum rtx_code code, rtx op0, rtx op1)
{
machine_mode mode = GET_MODE (op0);
enum rtx_code c;
rtx target;
if ((code == SMAX || code == SMIN)
&& (VECTOR_UNIT_ALTIVEC_OR_VSX_P (mode)
|| (mode == SFmode && VECTOR_UNIT_VSX_P (DFmode))))
{
emit_insn (gen_rtx_SET (dest, gen_rtx_fmt_ee (code, mode, op0, op1)));
return;
}
if (code == SMAX || code == SMIN)
c = GE;
else
c = GEU;
if (code == SMAX || code == UMAX)
target = emit_conditional_move (dest, c, op0, op1, mode,
op0, op1, mode, 0);
else
target = emit_conditional_move (dest, c, op0, op1, mode,
op1, op0, mode, 0);
gcc_assert (target);
if (target != dest)
emit_move_insn (dest, target);
}
static void
emit_unlikely_jump (rtx cond, rtx label)
{
rtx x = gen_rtx_IF_THEN_ELSE (VOIDmode, cond, label, pc_rtx);
rtx_insn *insn = emit_jump_insn (gen_rtx_SET (pc_rtx, x));
add_reg_br_prob_note (insn, profile_probability::very_unlikely ());
}
static void
emit_load_locked (machine_mode mode, rtx reg, rtx mem)
{
rtx (*fn) (rtx, rtx) = NULL;
switch (mode)
{
case E_QImode:
fn = gen_load_lockedqi;
break;
case E_HImode:
fn = gen_load_lockedhi;
break;
case E_SImode:
if (GET_MODE (mem) == QImode)
fn = gen_load_lockedqi_si;
else if (GET_MODE (mem) == HImode)
fn = gen_load_lockedhi_si;
else
fn = gen_load_lockedsi;
break;
case E_DImode:
fn = gen_load_lockeddi;
break;
case E_TImode:
fn = gen_load_lockedti;
break;
default:
gcc_unreachable ();
}
emit_insn (fn (reg, mem));
}
static void
emit_store_conditional (machine_mode mode, rtx res, rtx mem, rtx val)
{
rtx (*fn) (rtx, rtx, rtx) = NULL;
switch (mode)
{
case E_QImode:
fn = gen_store_conditionalqi;
break;
case E_HImode:
fn = gen_store_conditionalhi;
break;
case E_SImode:
fn = gen_store_conditionalsi;
break;
case E_DImode:
fn = gen_store_conditionaldi;
break;
case E_TImode:
fn = gen_store_conditionalti;
break;
default:
gcc_unreachable ();
}
if (PPC405_ERRATUM77)
emit_insn (gen_hwsync ());
emit_insn (fn (res, mem, val));
}
static rtx
rs6000_pre_atomic_barrier (rtx mem, enum memmodel model)
{
rtx addr = XEXP (mem, 0);
if (!legitimate_indirect_address_p (addr, reload_completed)
&& !legitimate_indexed_address_p (addr, reload_completed))
{
addr = force_reg (Pmode, addr);
mem = replace_equiv_address_nv (mem, addr);
}
switch (model)
{
case MEMMODEL_RELAXED:
case MEMMODEL_CONSUME:
case MEMMODEL_ACQUIRE:
break;
case MEMMODEL_RELEASE:
case MEMMODEL_ACQ_REL:
emit_insn (gen_lwsync ());
break;
case MEMMODEL_SEQ_CST:
emit_insn (gen_hwsync ());
break;
default:
gcc_unreachable ();
}
return mem;
}
static void
rs6000_post_atomic_barrier (enum memmodel model)
{
switch (model)
{
case MEMMODEL_RELAXED:
case MEMMODEL_CONSUME:
case MEMMODEL_RELEASE:
break;
case MEMMODEL_ACQUIRE:
case MEMMODEL_ACQ_REL:
case MEMMODEL_SEQ_CST:
emit_insn (gen_isync ());
break;
default:
gcc_unreachable ();
}
}
static rtx
rs6000_adjust_atomic_subword (rtx orig_mem, rtx *pshift, rtx *pmask)
{
rtx addr, align, shift, mask, mem;
HOST_WIDE_INT shift_mask;
machine_mode mode = GET_MODE (orig_mem);
shift_mask = (mode == QImode ? 0x18 : 0x10);
addr = XEXP (orig_mem, 0);
addr = force_reg (GET_MODE (addr), addr);
align = expand_simple_binop (Pmode, AND, addr, GEN_INT (-4),
NULL_RTX, 1, OPTAB_LIB_WIDEN);
mem = gen_rtx_MEM (SImode, align);
MEM_VOLATILE_P (mem) = MEM_VOLATILE_P (orig_mem);
if (MEM_ALIAS_SET (orig_mem) == ALIAS_SET_MEMORY_BARRIER)
set_mem_alias_set (mem, ALIAS_SET_MEMORY_BARRIER);
shift = gen_reg_rtx (SImode);
addr = gen_lowpart (SImode, addr);
rtx tmp = gen_reg_rtx (SImode);
emit_insn (gen_ashlsi3 (tmp, addr, GEN_INT (3)));
emit_insn (gen_andsi3 (shift, tmp, GEN_INT (shift_mask)));
if (BYTES_BIG_ENDIAN)
shift = expand_simple_binop (SImode, XOR, shift, GEN_INT (shift_mask),
shift, 1, OPTAB_LIB_WIDEN);
*pshift = shift;
mask = expand_simple_binop (SImode, ASHIFT, GEN_INT (GET_MODE_MASK (mode)),
shift, NULL_RTX, 1, OPTAB_LIB_WIDEN);
*pmask = mask;
return mem;
}
static rtx
rs6000_mask_atomic_subword (rtx oldval, rtx newval, rtx mask)
{
rtx x;
x = gen_reg_rtx (SImode);
emit_insn (gen_rtx_SET (x, gen_rtx_AND (SImode,
gen_rtx_NOT (SImode, mask),
oldval)));
x = expand_simple_binop (SImode, IOR, newval, x, x, 1, OPTAB_LIB_WIDEN);
return x;
}
static void
rs6000_finish_atomic_subword (rtx narrow, rtx wide, rtx shift)
{
wide = expand_simple_binop (SImode, LSHIFTRT, wide, shift,
wide, 1, OPTAB_LIB_WIDEN);
emit_move_insn (narrow, gen_lowpart (GET_MODE (narrow), wide));
}
void
rs6000_expand_atomic_compare_and_swap (rtx operands[])
{
rtx boolval, retval, mem, oldval, newval, cond;
rtx label1, label2, x, mask, shift;
machine_mode mode, orig_mode;
enum memmodel mod_s, mod_f;
bool is_weak;
boolval = operands[0];
retval = operands[1];
mem = operands[2];
oldval = operands[3];
newval = operands[4];
is_weak = (INTVAL (operands[5]) != 0);
mod_s = memmodel_base (INTVAL (operands[6]));
mod_f = memmodel_base (INTVAL (operands[7]));
orig_mode = mode = GET_MODE (mem);
mask = shift = NULL_RTX;
if (mode == QImode || mode == HImode)
{
oldval = convert_modes (SImode, mode, oldval, 1);
if (!TARGET_SYNC_HI_QI)
{
mem = rs6000_adjust_atomic_subword (mem, &shift, &mask);
oldval = expand_simple_binop (SImode, ASHIFT, oldval, shift,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
newval = convert_modes (SImode, mode, newval, 1);
newval = expand_simple_binop (SImode, ASHIFT, newval, shift,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
}
retval = gen_reg_rtx (SImode);
mode = SImode;
}
else if (reg_overlap_mentioned_p (retval, oldval))
oldval = copy_to_reg (oldval);
if (mode != TImode && !reg_or_short_operand (oldval, mode))
oldval = copy_to_mode_reg (mode, oldval);
if (reg_overlap_mentioned_p (retval, newval))
newval = copy_to_reg (newval);
mem = rs6000_pre_atomic_barrier (mem, mod_s);
label1 = NULL_RTX;
if (!is_weak)
{
label1 = gen_rtx_LABEL_REF (VOIDmode, gen_label_rtx ());
emit_label (XEXP (label1, 0));
}
label2 = gen_rtx_LABEL_REF (VOIDmode, gen_label_rtx ());
emit_load_locked (mode, retval, mem);
x = retval;
if (mask)
x = expand_simple_binop (SImode, AND, retval, mask,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
cond = gen_reg_rtx (CCmode);
if (mode != TImode)
x = gen_rtx_COMPARE (CCmode, x, oldval);
else
{
rtx xor1_result = gen_reg_rtx (DImode);
rtx xor2_result = gen_reg_rtx (DImode);
rtx or_result = gen_reg_rtx (DImode);
rtx new_word0 = simplify_gen_subreg (DImode, x, TImode, 0);
rtx new_word1 = simplify_gen_subreg (DImode, x, TImode, 8);
rtx old_word0 = simplify_gen_subreg (DImode, oldval, TImode, 0);
rtx old_word1 = simplify_gen_subreg (DImode, oldval, TImode, 8);
emit_insn (gen_xordi3 (xor1_result, new_word0, old_word0));
emit_insn (gen_xordi3 (xor2_result, new_word1, old_word1));
emit_insn (gen_iordi3 (or_result, xor1_result, xor2_result));
x = gen_rtx_COMPARE (CCmode, or_result, const0_rtx);
}
emit_insn (gen_rtx_SET (cond, x));
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
emit_unlikely_jump (x, label2);
x = newval;
if (mask)
x = rs6000_mask_atomic_subword (retval, newval, mask);
emit_store_conditional (orig_mode, cond, mem, x);
if (!is_weak)
{
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
emit_unlikely_jump (x, label1);
}
if (!is_mm_relaxed (mod_f))
emit_label (XEXP (label2, 0));
rs6000_post_atomic_barrier (mod_s);
if (is_mm_relaxed (mod_f))
emit_label (XEXP (label2, 0));
if (shift)
rs6000_finish_atomic_subword (operands[1], retval, shift);
else if (mode != GET_MODE (operands[1]))
convert_move (operands[1], retval, 1);
x = gen_rtx_EQ (SImode, cond, const0_rtx);
emit_insn (gen_rtx_SET (boolval, x));
}
void
rs6000_expand_atomic_exchange (rtx operands[])
{
rtx retval, mem, val, cond;
machine_mode mode;
enum memmodel model;
rtx label, x, mask, shift;
retval = operands[0];
mem = operands[1];
val = operands[2];
model = memmodel_base (INTVAL (operands[3]));
mode = GET_MODE (mem);
mask = shift = NULL_RTX;
if (!TARGET_SYNC_HI_QI && (mode == QImode || mode == HImode))
{
mem = rs6000_adjust_atomic_subword (mem, &shift, &mask);
val = convert_modes (SImode, mode, val, 1);
val = expand_simple_binop (SImode, ASHIFT, val, shift,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
retval = gen_reg_rtx (SImode);
mode = SImode;
}
mem = rs6000_pre_atomic_barrier (mem, model);
label = gen_rtx_LABEL_REF (VOIDmode, gen_label_rtx ());
emit_label (XEXP (label, 0));
emit_load_locked (mode, retval, mem);
x = val;
if (mask)
x = rs6000_mask_atomic_subword (retval, val, mask);
cond = gen_reg_rtx (CCmode);
emit_store_conditional (mode, cond, mem, x);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
emit_unlikely_jump (x, label);
rs6000_post_atomic_barrier (model);
if (shift)
rs6000_finish_atomic_subword (operands[0], retval, shift);
}
void
rs6000_expand_atomic_op (enum rtx_code code, rtx mem, rtx val,
rtx orig_before, rtx orig_after, rtx model_rtx)
{
enum memmodel model = memmodel_base (INTVAL (model_rtx));
machine_mode mode = GET_MODE (mem);
machine_mode store_mode = mode;
rtx label, x, cond, mask, shift;
rtx before = orig_before, after = orig_after;
mask = shift = NULL_RTX;
if (mode == QImode || mode == HImode)
{
if (TARGET_SYNC_HI_QI)
{
val = convert_modes (SImode, mode, val, 1);
before = gen_reg_rtx (SImode);
if (after)
after = gen_reg_rtx (SImode);
mode = SImode;
}
else
{
mem = rs6000_adjust_atomic_subword (mem, &shift, &mask);
val = convert_modes (SImode, mode, val, 1);
val = expand_simple_binop (SImode, ASHIFT, val, shift,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
switch (code)
{
case IOR:
case XOR:
mask = NULL;
break;
case AND:
x = gen_rtx_NOT (SImode, mask);
x = gen_rtx_IOR (SImode, x, val);
emit_insn (gen_rtx_SET (val, x));
mask = NULL;
break;
case NOT:
case PLUS:
case MINUS:
break;
default:
gcc_unreachable ();
}
before = gen_reg_rtx (SImode);
if (after)
after = gen_reg_rtx (SImode);
store_mode = mode = SImode;
}
}
mem = rs6000_pre_atomic_barrier (mem, model);
label = gen_label_rtx ();
emit_label (label);
label = gen_rtx_LABEL_REF (VOIDmode, label);
if (before == NULL_RTX)
before = gen_reg_rtx (mode);
emit_load_locked (mode, before, mem);
if (code == NOT)
{
x = expand_simple_binop (mode, AND, before, val,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
after = expand_simple_unop (mode, NOT, x, after, 1);
}
else
{
after = expand_simple_binop (mode, code, before, val,
after, 1, OPTAB_LIB_WIDEN);
}
x = after;
if (mask)
{
x = expand_simple_binop (SImode, AND, after, mask,
NULL_RTX, 1, OPTAB_LIB_WIDEN);
x = rs6000_mask_atomic_subword (before, x, mask);
}
else if (store_mode != mode)
x = convert_modes (store_mode, mode, x, 1);
cond = gen_reg_rtx (CCmode);
emit_store_conditional (store_mode, cond, mem, x);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
emit_unlikely_jump (x, label);
rs6000_post_atomic_barrier (model);
if (shift)
{
if (orig_before)
rs6000_finish_atomic_subword (orig_before, before, shift);
if (orig_after)
rs6000_finish_atomic_subword (orig_after, after, shift);
}
else if (store_mode != mode)
{
if (orig_before)
convert_move (orig_before, before, 1);
if (orig_after)
convert_move (orig_after, after, 1);
}
else if (orig_after && after != orig_after)
emit_move_insn (orig_after, after);
}
void
rs6000_split_multireg_move (rtx dst, rtx src)
{
int reg;
machine_mode mode;
machine_mode reg_mode;
int reg_mode_size;
int nregs;
reg = REG_P (dst) ? REGNO (dst) : REGNO (src);
mode = GET_MODE (dst);
nregs = hard_regno_nregs (reg, mode);
if (FP_REGNO_P (reg))
reg_mode = DECIMAL_FLOAT_MODE_P (mode) ? DDmode : 
((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT) ? DFmode : SFmode);
else if (ALTIVEC_REGNO_P (reg))
reg_mode = V16QImode;
else
reg_mode = word_mode;
reg_mode_size = GET_MODE_SIZE (reg_mode);
gcc_assert (reg_mode_size * nregs == GET_MODE_SIZE (mode));
if (FP_REGNO_P (reg) && DECIMAL_FLOAT_MODE_P (mode) && !BYTES_BIG_ENDIAN)
{
rtx p_src, p_dst;
int i;
for (i = 0; i < nregs; i++)
{
if (REG_P (src) && FP_REGNO_P (REGNO (src)))
p_src = gen_rtx_REG (reg_mode, REGNO (src) + nregs - 1 - i);
else
p_src = simplify_gen_subreg (reg_mode, src, mode,
i * reg_mode_size);
if (REG_P (dst) && FP_REGNO_P (REGNO (dst)))
p_dst = gen_rtx_REG (reg_mode, REGNO (dst) + nregs - 1 - i);
else
p_dst = simplify_gen_subreg (reg_mode, dst, mode,
i * reg_mode_size);
emit_insn (gen_rtx_SET (p_dst, p_src));
}
return;
}
if (REG_P (src) && REG_P (dst) && (REGNO (src) < REGNO (dst)))
{
int i;
for (i = nregs - 1; i >= 0; i--)
emit_insn (gen_rtx_SET (simplify_gen_subreg (reg_mode, dst, mode,
i * reg_mode_size),
simplify_gen_subreg (reg_mode, src, mode,
i * reg_mode_size)));
}
else
{
int i;
int j = -1;
bool used_update = false;
rtx restore_basereg = NULL_RTX;
if (MEM_P (src) && INT_REGNO_P (reg))
{
rtx breg;
if (GET_CODE (XEXP (src, 0)) == PRE_INC
|| GET_CODE (XEXP (src, 0)) == PRE_DEC)
{
rtx delta_rtx;
breg = XEXP (XEXP (src, 0), 0);
delta_rtx = (GET_CODE (XEXP (src, 0)) == PRE_INC
? GEN_INT (GET_MODE_SIZE (GET_MODE (src)))
: GEN_INT (-GET_MODE_SIZE (GET_MODE (src))));
emit_insn (gen_add3_insn (breg, breg, delta_rtx));
src = replace_equiv_address (src, breg);
}
else if (! rs6000_offsettable_memref_p (src, reg_mode, true))
{
if (GET_CODE (XEXP (src, 0)) == PRE_MODIFY)
{
rtx basereg = XEXP (XEXP (src, 0), 0);
if (TARGET_UPDATE)
{
rtx ndst = simplify_gen_subreg (reg_mode, dst, mode, 0);
emit_insn (gen_rtx_SET (ndst,
gen_rtx_MEM (reg_mode,
XEXP (src, 0))));
used_update = true;
}
else
emit_insn (gen_rtx_SET (basereg,
XEXP (XEXP (src, 0), 1)));
src = replace_equiv_address (src, basereg);
}
else
{
rtx basereg = gen_rtx_REG (Pmode, reg);
emit_insn (gen_rtx_SET (basereg, XEXP (src, 0)));
src = replace_equiv_address (src, basereg);
}
}
breg = XEXP (src, 0);
if (GET_CODE (breg) == PLUS || GET_CODE (breg) == LO_SUM)
breg = XEXP (breg, 0);
if (REG_P (breg)
&& REGNO (breg) >= REGNO (dst)
&& REGNO (breg) < REGNO (dst) + nregs)
j = REGNO (breg) - REGNO (dst);
}
else if (MEM_P (dst) && INT_REGNO_P (reg))
{
rtx breg;
if (GET_CODE (XEXP (dst, 0)) == PRE_INC
|| GET_CODE (XEXP (dst, 0)) == PRE_DEC)
{
rtx delta_rtx;
breg = XEXP (XEXP (dst, 0), 0);
delta_rtx = (GET_CODE (XEXP (dst, 0)) == PRE_INC
? GEN_INT (GET_MODE_SIZE (GET_MODE (dst)))
: GEN_INT (-GET_MODE_SIZE (GET_MODE (dst))));
if (TARGET_UPDATE)
{
rtx nsrc = simplify_gen_subreg (reg_mode, src, mode, 0);
emit_insn (TARGET_32BIT
? (TARGET_POWERPC64
? gen_movdi_si_update (breg, breg, delta_rtx, nsrc)
: gen_movsi_update (breg, breg, delta_rtx, nsrc))
: gen_movdi_di_update (breg, breg, delta_rtx, nsrc));
used_update = true;
}
else
emit_insn (gen_add3_insn (breg, breg, delta_rtx));
dst = replace_equiv_address (dst, breg);
}
else if (!rs6000_offsettable_memref_p (dst, reg_mode, true)
&& GET_CODE (XEXP (dst, 0)) != LO_SUM)
{
if (GET_CODE (XEXP (dst, 0)) == PRE_MODIFY)
{
rtx basereg = XEXP (XEXP (dst, 0), 0);
if (TARGET_UPDATE)
{
rtx nsrc = simplify_gen_subreg (reg_mode, src, mode, 0);
emit_insn (gen_rtx_SET (gen_rtx_MEM (reg_mode,
XEXP (dst, 0)),
nsrc));
used_update = true;
}
else
emit_insn (gen_rtx_SET (basereg,
XEXP (XEXP (dst, 0), 1)));
dst = replace_equiv_address (dst, basereg);
}
else
{
rtx basereg = XEXP (XEXP (dst, 0), 0);
rtx offsetreg = XEXP (XEXP (dst, 0), 1);
gcc_assert (GET_CODE (XEXP (dst, 0)) == PLUS
&& REG_P (basereg)
&& REG_P (offsetreg)
&& REGNO (basereg) != REGNO (offsetreg));
if (REGNO (basereg) == 0)
{
rtx tmp = offsetreg;
offsetreg = basereg;
basereg = tmp;
}
emit_insn (gen_add3_insn (basereg, basereg, offsetreg));
restore_basereg = gen_sub3_insn (basereg, basereg, offsetreg);
dst = replace_equiv_address (dst, basereg);
}
}
else if (GET_CODE (XEXP (dst, 0)) != LO_SUM)
gcc_assert (rs6000_offsettable_memref_p (dst, reg_mode, true));
}
for (i = 0; i < nregs; i++)
{
++j;
if (j == nregs)
j = 0;
if (j == 0 && used_update)
continue;
emit_insn (gen_rtx_SET (simplify_gen_subreg (reg_mode, dst, mode,
j * reg_mode_size),
simplify_gen_subreg (reg_mode, src, mode,
j * reg_mode_size)));
}
if (restore_basereg != NULL_RTX)
emit_insn (restore_basereg);
}
}

static bool
save_reg_p (int reg)
{
if (reg == RS6000_PIC_OFFSET_TABLE_REGNUM && !TARGET_SINGLE_PIC_BASE)
{
if (crtl->calls_eh_return
&& ((DEFAULT_ABI == ABI_V4 && flag_pic)
|| (DEFAULT_ABI == ABI_DARWIN && flag_pic)
|| (TARGET_TOC && TARGET_MINIMAL_TOC)))
return true;
if (TARGET_TOC && TARGET_MINIMAL_TOC
&& !constant_pool_empty_p ())
return true;
if (DEFAULT_ABI == ABI_V4
&& (flag_pic == 1 || (flag_pic && TARGET_SECURE_PLT))
&& df_regs_ever_live_p (RS6000_PIC_OFFSET_TABLE_REGNUM))
return true;
if (DEFAULT_ABI == ABI_DARWIN
&& flag_pic && crtl->uses_pic_offset_table)
return true;
}
return !call_used_regs[reg] && df_regs_ever_live_p (reg);
}
int
first_reg_to_save (void)
{
int first_reg;
for (first_reg = 13; first_reg <= 31; first_reg++)
if (save_reg_p (first_reg))
break;
#if TARGET_MACHO
if (flag_pic
&& crtl->uses_pic_offset_table
&& first_reg > RS6000_PIC_OFFSET_TABLE_REGNUM)
return RS6000_PIC_OFFSET_TABLE_REGNUM;
#endif
return first_reg;
}
int
first_fp_reg_to_save (void)
{
int first_reg;
for (first_reg = 14 + 32; first_reg <= 63; first_reg++)
if (save_reg_p (first_reg))
break;
return first_reg;
}
static int
first_altivec_reg_to_save (void)
{
int i;
if (! TARGET_ALTIVEC_ABI)
return LAST_ALTIVEC_REGNO + 1;
if (DEFAULT_ABI == ABI_DARWIN && crtl->calls_eh_return
&& ! TARGET_ALTIVEC)
return FIRST_ALTIVEC_REGNO + 20;
for (i = FIRST_ALTIVEC_REGNO + 20; i <= LAST_ALTIVEC_REGNO; ++i)
if (save_reg_p (i))
break;
return i;
}
static unsigned int
compute_vrsave_mask (void)
{
unsigned int i, mask = 0;
if (DEFAULT_ABI == ABI_DARWIN && crtl->calls_eh_return
&& ! TARGET_ALTIVEC)
mask |= 0xFFF;
for (i = FIRST_ALTIVEC_REGNO; i <= LAST_ALTIVEC_REGNO; ++i)
if (df_regs_ever_live_p (i))
mask |= ALTIVEC_REG_BIT (i);
if (mask == 0)
return mask;
for (i = ALTIVEC_ARG_MIN_REG; i < (unsigned) crtl->args.info.vregno; i++)
mask &= ~ALTIVEC_REG_BIT (i);
{
bool yes = false;
diddle_return_value (is_altivec_return_reg, &yes);
if (yes)
mask &= ~ALTIVEC_REG_BIT (ALTIVEC_ARG_RETURN);
}
return mask;
}
static void
compute_save_world_info (rs6000_stack_t *info)
{
info->world_save_p = 1;
info->world_save_p
= (WORLD_SAVE_P (info)
&& DEFAULT_ABI == ABI_DARWIN
&& !cfun->has_nonlocal_label
&& info->first_fp_reg_save == FIRST_SAVED_FP_REGNO
&& info->first_gp_reg_save == FIRST_SAVED_GP_REGNO
&& info->first_altivec_reg_save == FIRST_SAVED_ALTIVEC_REGNO
&& info->cr_save_p);
if (WORLD_SAVE_P (info))
{
rtx_insn *insn;
for (insn = get_last_insn_anywhere (); insn; insn = PREV_INSN (insn))
if (CALL_P (insn) && SIBLING_CALL_P (insn))
{
info->world_save_p = 0;
break;
}
}
if (WORLD_SAVE_P (info))
{
info->vrsave_size  = 4;
info->lr_save_p = 1;
if (info->vrsave_mask == 0)
info->vrsave_mask = compute_vrsave_mask ();
gcc_assert (info->first_fp_reg_save >= FIRST_SAVED_FP_REGNO
&& (info->first_altivec_reg_save
>= FIRST_SAVED_ALTIVEC_REGNO));
}
return;
}
static void
is_altivec_return_reg (rtx reg, void *xyes)
{
bool *yes = (bool *) xyes;
if (REGNO (reg) == ALTIVEC_ARG_RETURN)
*yes = true;
}

static bool
fixed_reg_p (int reg)
{
if (reg == RS6000_PIC_OFFSET_TABLE_REGNUM
&& ((DEFAULT_ABI == ABI_V4 && flag_pic)
|| (DEFAULT_ABI == ABI_DARWIN && flag_pic)
|| (TARGET_TOC && TARGET_MINIMAL_TOC)))
return false;
return fixed_regs[reg];
}
enum {
SAVE_MULTIPLE = 0x1,
SAVE_INLINE_GPRS = 0x2,
SAVE_INLINE_FPRS = 0x4,
SAVE_NOINLINE_GPRS_SAVES_LR = 0x8,
SAVE_NOINLINE_FPRS_SAVES_LR = 0x10,
SAVE_INLINE_VRS = 0x20,
REST_MULTIPLE = 0x100,
REST_INLINE_GPRS = 0x200,
REST_INLINE_FPRS = 0x400,
REST_NOINLINE_FPRS_DOESNT_RESTORE_LR = 0x800,
REST_INLINE_VRS = 0x1000
};
static int
rs6000_savres_strategy (rs6000_stack_t *info,
bool using_static_chain_p)
{
int strategy = 0;
if (crtl->calls_eh_return
|| cfun->machine->ra_need_lr)
strategy |= (SAVE_INLINE_FPRS | REST_INLINE_FPRS
| SAVE_INLINE_GPRS | REST_INLINE_GPRS
| SAVE_INLINE_VRS | REST_INLINE_VRS);
if (info->first_gp_reg_save == 32)
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
if (info->first_fp_reg_save == 64
|| (TARGET_HARD_FLOAT && !TARGET_DOUBLE_FLOAT))
strategy |= SAVE_INLINE_FPRS | REST_INLINE_FPRS;
if (info->first_altivec_reg_save == LAST_ALTIVEC_REGNO + 1)
strategy |= SAVE_INLINE_VRS | REST_INLINE_VRS;
if (DEFAULT_ABI == ABI_V4 || TARGET_ELF)
{
if (!optimize_size)
{
strategy |= SAVE_INLINE_FPRS | REST_INLINE_FPRS;
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
strategy |= SAVE_INLINE_VRS | REST_INLINE_VRS;
}
else
{
if (info->first_fp_reg_save > 61)
strategy |= SAVE_INLINE_FPRS;
if (info->first_gp_reg_save > 29)
{
if (info->first_fp_reg_save == 64)
strategy |= SAVE_INLINE_GPRS;
else
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
}
if (info->first_altivec_reg_save == LAST_ALTIVEC_REGNO)
strategy |= SAVE_INLINE_VRS | REST_INLINE_VRS;
}
}
else if (DEFAULT_ABI == ABI_DARWIN)
{
if (info->first_fp_reg_save > 60)
strategy |= SAVE_INLINE_FPRS | REST_INLINE_FPRS;
if (info->first_gp_reg_save > 29)
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
strategy |= SAVE_INLINE_VRS | REST_INLINE_VRS;
}
else
{
gcc_checking_assert (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2);
if ((flag_shrink_wrap_separate && optimize_function_for_speed_p (cfun))
|| info->first_fp_reg_save > 61)
strategy |= SAVE_INLINE_FPRS | REST_INLINE_FPRS;
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
strategy |= SAVE_INLINE_VRS | REST_INLINE_VRS;
}
if (using_static_chain_p
&& (DEFAULT_ABI == ABI_V4 || DEFAULT_ABI == ABI_DARWIN))
strategy |= ((DEFAULT_ABI == ABI_DARWIN ? 0 : SAVE_INLINE_FPRS)
| SAVE_INLINE_GPRS
| SAVE_INLINE_VRS);
if (!(strategy & REST_INLINE_FPRS))
for (int i = info->first_fp_reg_save; i < 64; i++)
if (fixed_regs[i])
{
strategy |= REST_INLINE_FPRS;
break;
}
if ((strategy & SAVE_INLINE_FPRS)
&& !(strategy & REST_INLINE_FPRS))
for (int i = info->first_fp_reg_save; i < 64; i++)
if (!save_reg_p (i))
{
strategy |= REST_INLINE_FPRS;
break;
}
if (!(strategy & REST_INLINE_VRS))
for (int i = info->first_altivec_reg_save; i < LAST_ALTIVEC_REGNO + 1; i++)
if (fixed_regs[i])
{
strategy |= REST_INLINE_VRS;
break;
}
if ((strategy & SAVE_INLINE_VRS)
&& !(strategy & REST_INLINE_VRS))
for (int i = info->first_altivec_reg_save; i < LAST_ALTIVEC_REGNO + 1; i++)
if (!save_reg_p (i))
{
strategy |= REST_INLINE_VRS;
break;
}
bool lr_save_p = (info->lr_save_p
|| !(strategy & SAVE_INLINE_FPRS)
|| !(strategy & SAVE_INLINE_VRS)
|| !(strategy & REST_INLINE_FPRS)
|| !(strategy & REST_INLINE_VRS));
if (TARGET_MULTIPLE
&& !TARGET_POWERPC64
&& info->first_gp_reg_save < 31
&& !(flag_shrink_wrap
&& flag_shrink_wrap_separate
&& optimize_function_for_speed_p (cfun)))
{
int count = 0;
for (int i = info->first_gp_reg_save; i < 32; i++)
if (save_reg_p (i))
count++;
if (count <= 1)
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
else
{
strategy |= SAVE_INLINE_GPRS | SAVE_MULTIPLE;
if (info->first_fp_reg_save != 64 || !lr_save_p)
strategy |= REST_INLINE_GPRS | REST_MULTIPLE;
}
}
else if (!lr_save_p && info->first_gp_reg_save > 29)
strategy |= SAVE_INLINE_GPRS | REST_INLINE_GPRS;
if ((strategy & (REST_INLINE_GPRS | REST_MULTIPLE)) != REST_INLINE_GPRS)
for (int i = info->first_gp_reg_save; i < 32; i++)
if (fixed_reg_p (i))
{
strategy |= REST_INLINE_GPRS;
strategy &= ~REST_MULTIPLE;
break;
}
if ((strategy & (SAVE_INLINE_GPRS | SAVE_MULTIPLE)) == SAVE_INLINE_GPRS
&& (strategy & (REST_INLINE_GPRS | REST_MULTIPLE)) != REST_INLINE_GPRS)
for (int i = info->first_gp_reg_save; i < 32; i++)
if (!save_reg_p (i))
{
strategy |= REST_INLINE_GPRS;
strategy &= ~REST_MULTIPLE;
break;
}
if (TARGET_ELF && TARGET_64BIT)
{
if (!(strategy & SAVE_INLINE_FPRS))
strategy |= SAVE_NOINLINE_FPRS_SAVES_LR;
else if (!(strategy & SAVE_INLINE_GPRS)
&& info->first_fp_reg_save == 64)
strategy |= SAVE_NOINLINE_GPRS_SAVES_LR;
}
else if (TARGET_AIX && !(strategy & REST_INLINE_FPRS))
strategy |= REST_NOINLINE_FPRS_DOESNT_RESTORE_LR;
if (TARGET_MACHO && !(strategy & SAVE_INLINE_FPRS))
strategy |= SAVE_NOINLINE_FPRS_SAVES_LR;
return strategy;
}
#ifndef ABI_STACK_BOUNDARY
#define ABI_STACK_BOUNDARY STACK_BOUNDARY
#endif
static rs6000_stack_t *
rs6000_stack_info (void)
{
gcc_assert (!cfun->is_thunk);
rs6000_stack_t *info = &stack_info;
int reg_size = TARGET_32BIT ? 4 : 8;
int ehrd_size;
int ehcr_size;
int save_align;
int first_gp;
HOST_WIDE_INT non_fixed_size;
bool using_static_chain_p;
if (reload_completed && info->reload_completed)
return info;
memset (info, 0, sizeof (*info));
info->reload_completed = reload_completed;
info->abi = DEFAULT_ABI;
info->first_gp_reg_save = first_reg_to_save ();
if (((TARGET_TOC && TARGET_MINIMAL_TOC)
|| (flag_pic == 1 && DEFAULT_ABI == ABI_V4)
|| (flag_pic && DEFAULT_ABI == ABI_DARWIN))
&& crtl->uses_const_pool
&& info->first_gp_reg_save > RS6000_PIC_OFFSET_TABLE_REGNUM)
first_gp = RS6000_PIC_OFFSET_TABLE_REGNUM;
else
first_gp = info->first_gp_reg_save;
info->gp_size = reg_size * (32 - first_gp);
info->first_fp_reg_save = first_fp_reg_to_save ();
info->fp_size = 8 * (64 - info->first_fp_reg_save);
info->first_altivec_reg_save = first_altivec_reg_to_save ();
info->altivec_size = 16 * (LAST_ALTIVEC_REGNO + 1
- info->first_altivec_reg_save);
info->calls_p = (!crtl->is_leaf || cfun->machine->ra_needs_full_frame);
if (save_reg_p (CR2_REGNO)
|| save_reg_p (CR3_REGNO)
|| save_reg_p (CR4_REGNO))
{
info->cr_save_p = 1;
if (DEFAULT_ABI == ABI_V4)
info->cr_size = reg_size;
}
if (crtl->calls_eh_return)
{
unsigned int i;
for (i = 0; EH_RETURN_DATA_REGNO (i) != INVALID_REGNUM; ++i)
continue;
ehrd_size = i * UNITS_PER_WORD;
}
else
ehrd_size = 0;
if (DEFAULT_ABI == ABI_ELFv2 && crtl->calls_eh_return)
{
ehcr_size = 3 * reg_size;
info->cr_save_p = 0;
}
else
ehcr_size = 0;
info->reg_size     = reg_size;
info->fixed_size   = RS6000_SAVE_AREA;
info->vars_size    = RS6000_ALIGN (get_frame_size (), 8);
if (cfun->calls_alloca)
info->parm_size  =
RS6000_ALIGN (crtl->outgoing_args_size + info->fixed_size,
STACK_BOUNDARY / BITS_PER_UNIT) - info->fixed_size;
else
info->parm_size  = RS6000_ALIGN (crtl->outgoing_args_size,
TARGET_ALTIVEC ? 16 : 8);
if (FRAME_GROWS_DOWNWARD)
info->vars_size
+= RS6000_ALIGN (info->fixed_size + info->vars_size + info->parm_size,
ABI_STACK_BOUNDARY / BITS_PER_UNIT)
- (info->fixed_size + info->vars_size + info->parm_size);
if (TARGET_ALTIVEC_ABI)
info->vrsave_mask = compute_vrsave_mask ();
if (TARGET_ALTIVEC_VRSAVE && info->vrsave_mask)
info->vrsave_size = 4;
compute_save_world_info (info);
switch (DEFAULT_ABI)
{
case ABI_NONE:
default:
gcc_unreachable ();
case ABI_AIX:
case ABI_ELFv2:
case ABI_DARWIN:
info->fp_save_offset = -info->fp_size;
info->gp_save_offset = info->fp_save_offset - info->gp_size;
if (TARGET_ALTIVEC_ABI)
{
info->vrsave_save_offset = info->gp_save_offset - info->vrsave_size;
if (info->altivec_size != 0)
info->altivec_padding_size = info->vrsave_save_offset & 0xF;
info->altivec_save_offset = info->vrsave_save_offset
- info->altivec_padding_size
- info->altivec_size;
gcc_assert (info->altivec_size == 0
|| info->altivec_save_offset % 16 == 0);
info->ehrd_offset = info->altivec_save_offset - ehrd_size;
}
else
info->ehrd_offset = info->gp_save_offset - ehrd_size;
info->ehcr_offset = info->ehrd_offset - ehcr_size;
info->cr_save_offset = reg_size; 
info->lr_save_offset = 2*reg_size;
break;
case ABI_V4:
info->fp_save_offset = -info->fp_size;
info->gp_save_offset = info->fp_save_offset - info->gp_size;
info->cr_save_offset = info->gp_save_offset - info->cr_size;
if (TARGET_ALTIVEC_ABI)
{
info->vrsave_save_offset = info->cr_save_offset - info->vrsave_size;
if (info->altivec_size != 0)
info->altivec_padding_size = 16 - (-info->vrsave_save_offset % 16);
info->altivec_save_offset = info->vrsave_save_offset
- info->altivec_padding_size
- info->altivec_size;
info->ehrd_offset = info->altivec_save_offset;
}
else
info->ehrd_offset = info->cr_save_offset;
info->ehrd_offset -= ehrd_size;
info->lr_save_offset = reg_size;
}
save_align = (TARGET_ALTIVEC_ABI || DEFAULT_ABI == ABI_DARWIN) ? 16 : 8;
info->save_size = RS6000_ALIGN (info->fp_size
+ info->gp_size
+ info->altivec_size
+ info->altivec_padding_size
+ ehrd_size
+ ehcr_size
+ info->cr_size
+ info->vrsave_size,
save_align);
non_fixed_size = info->vars_size + info->parm_size + info->save_size;
info->total_size = RS6000_ALIGN (non_fixed_size + info->fixed_size,
ABI_STACK_BOUNDARY / BITS_PER_UNIT);
if (info->calls_p
|| ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& crtl->profile
&& !TARGET_PROFILE_KERNEL)
|| (DEFAULT_ABI == ABI_V4 && cfun->calls_alloca)
#ifdef TARGET_RELOCATABLE
|| (DEFAULT_ABI == ABI_V4
&& (TARGET_RELOCATABLE || flag_pic > 1)
&& !constant_pool_empty_p ())
#endif
|| rs6000_ra_ever_killed ())
info->lr_save_p = 1;
using_static_chain_p = (cfun->static_chain_decl != NULL_TREE
&& df_regs_ever_live_p (STATIC_CHAIN_REGNUM)
&& call_used_regs[STATIC_CHAIN_REGNUM]);
info->savres_strategy = rs6000_savres_strategy (info, using_static_chain_p);
if (!(info->savres_strategy & SAVE_INLINE_GPRS)
|| !(info->savres_strategy & SAVE_INLINE_FPRS)
|| !(info->savres_strategy & SAVE_INLINE_VRS)
|| !(info->savres_strategy & REST_INLINE_GPRS)
|| !(info->savres_strategy & REST_INLINE_FPRS)
|| !(info->savres_strategy & REST_INLINE_VRS))
info->lr_save_p = 1;
if (info->lr_save_p)
df_set_regs_ever_live (LR_REGNO, true);
if (info->calls_p)
info->push_p = 1;
else if (DEFAULT_ABI == ABI_V4)
info->push_p = non_fixed_size != 0;
else if (frame_pointer_needed)
info->push_p = 1;
else if (TARGET_XCOFF && write_symbols != NO_DEBUG)
info->push_p = 1;
else
info->push_p = non_fixed_size > (TARGET_32BIT ? 220 : 288);
return info;
}
static void
debug_stack_info (rs6000_stack_t *info)
{
const char *abi_string;
if (! info)
info = rs6000_stack_info ();
fprintf (stderr, "\nStack information for function %s:\n",
((current_function_decl && DECL_NAME (current_function_decl))
? IDENTIFIER_POINTER (DECL_NAME (current_function_decl))
: "<unknown>"));
switch (info->abi)
{
default:		 abi_string = "Unknown";	break;
case ABI_NONE:	 abi_string = "NONE";		break;
case ABI_AIX:	 abi_string = "AIX";		break;
case ABI_ELFv2:	 abi_string = "ELFv2";		break;
case ABI_DARWIN:	 abi_string = "Darwin";		break;
case ABI_V4:	 abi_string = "V.4";		break;
}
fprintf (stderr, "\tABI                 = %5s\n", abi_string);
if (TARGET_ALTIVEC_ABI)
fprintf (stderr, "\tALTIVEC ABI extensions enabled.\n");
if (info->first_gp_reg_save != 32)
fprintf (stderr, "\tfirst_gp_reg_save   = %5d\n", info->first_gp_reg_save);
if (info->first_fp_reg_save != 64)
fprintf (stderr, "\tfirst_fp_reg_save   = %5d\n", info->first_fp_reg_save);
if (info->first_altivec_reg_save <= LAST_ALTIVEC_REGNO)
fprintf (stderr, "\tfirst_altivec_reg_save = %5d\n",
info->first_altivec_reg_save);
if (info->lr_save_p)
fprintf (stderr, "\tlr_save_p           = %5d\n", info->lr_save_p);
if (info->cr_save_p)
fprintf (stderr, "\tcr_save_p           = %5d\n", info->cr_save_p);
if (info->vrsave_mask)
fprintf (stderr, "\tvrsave_mask         = 0x%x\n", info->vrsave_mask);
if (info->push_p)
fprintf (stderr, "\tpush_p              = %5d\n", info->push_p);
if (info->calls_p)
fprintf (stderr, "\tcalls_p             = %5d\n", info->calls_p);
if (info->gp_size)
fprintf (stderr, "\tgp_save_offset      = %5d\n", info->gp_save_offset);
if (info->fp_size)
fprintf (stderr, "\tfp_save_offset      = %5d\n", info->fp_save_offset);
if (info->altivec_size)
fprintf (stderr, "\taltivec_save_offset = %5d\n",
info->altivec_save_offset);
if (info->vrsave_size)
fprintf (stderr, "\tvrsave_save_offset  = %5d\n",
info->vrsave_save_offset);
if (info->lr_save_p)
fprintf (stderr, "\tlr_save_offset      = %5d\n", info->lr_save_offset);
if (info->cr_save_p)
fprintf (stderr, "\tcr_save_offset      = %5d\n", info->cr_save_offset);
if (info->varargs_save_offset)
fprintf (stderr, "\tvarargs_save_offset = %5d\n", info->varargs_save_offset);
if (info->total_size)
fprintf (stderr, "\ttotal_size          = " HOST_WIDE_INT_PRINT_DEC"\n",
info->total_size);
if (info->vars_size)
fprintf (stderr, "\tvars_size           = " HOST_WIDE_INT_PRINT_DEC"\n",
info->vars_size);
if (info->parm_size)
fprintf (stderr, "\tparm_size           = %5d\n", info->parm_size);
if (info->fixed_size)
fprintf (stderr, "\tfixed_size          = %5d\n", info->fixed_size);
if (info->gp_size)
fprintf (stderr, "\tgp_size             = %5d\n", info->gp_size);
if (info->fp_size)
fprintf (stderr, "\tfp_size             = %5d\n", info->fp_size);
if (info->altivec_size)
fprintf (stderr, "\taltivec_size        = %5d\n", info->altivec_size);
if (info->vrsave_size)
fprintf (stderr, "\tvrsave_size         = %5d\n", info->vrsave_size);
if (info->altivec_padding_size)
fprintf (stderr, "\taltivec_padding_size= %5d\n",
info->altivec_padding_size);
if (info->cr_size)
fprintf (stderr, "\tcr_size             = %5d\n", info->cr_size);
if (info->save_size)
fprintf (stderr, "\tsave_size           = %5d\n", info->save_size);
if (info->reg_size != 4)
fprintf (stderr, "\treg_size            = %5d\n", info->reg_size);
fprintf (stderr, "\tsave-strategy       =  %04x\n", info->savres_strategy);
fprintf (stderr, "\n");
}
rtx
rs6000_return_addr (int count, rtx frame)
{
if (count != 0
|| ((DEFAULT_ABI == ABI_V4 || DEFAULT_ABI == ABI_DARWIN) && flag_pic))
{
cfun->machine->ra_needs_full_frame = 1;
if (count == 0)
frame = stack_pointer_rtx;
rtx prev_frame_addr = memory_address (Pmode, frame);
rtx prev_frame = copy_to_reg (gen_rtx_MEM (Pmode, prev_frame_addr));
rtx lr_save_off = plus_constant (Pmode,
prev_frame, RETURN_ADDRESS_OFFSET);
rtx lr_save_addr = memory_address (Pmode, lr_save_off);
return gen_rtx_MEM (Pmode, lr_save_addr);
}
cfun->machine->ra_need_lr = 1;
return get_hard_reg_initial_val (Pmode, LR_REGNO);
}
static bool
rs6000_function_ok_for_sibcall (tree decl, tree exp)
{
tree fntype;
if (CALL_EXPR_STATIC_CHAIN (exp))
return false;
if (decl)
fntype = TREE_TYPE (decl);
else
fntype = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (exp)));
if (TARGET_ALTIVEC_ABI
&& TARGET_ALTIVEC_VRSAVE
&& !(decl && decl == current_function_decl))
{
function_args_iterator args_iter;
tree type;
int nvreg = 0;
FOREACH_FUNCTION_ARGS(fntype, type, args_iter)
if (TREE_CODE (type) == VECTOR_TYPE
&& ALTIVEC_OR_VSX_VECTOR_MODE (TYPE_MODE (type)))
nvreg++;
FOREACH_FUNCTION_ARGS(TREE_TYPE (current_function_decl), type, args_iter)
if (TREE_CODE (type) == VECTOR_TYPE
&& ALTIVEC_OR_VSX_VECTOR_MODE (TYPE_MODE (type)))
nvreg--;
if (nvreg > 0)
return false;
}
if (DEFAULT_ABI == ABI_DARWIN
|| ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& decl
&& !DECL_EXTERNAL (decl)
&& !DECL_WEAK (decl)
&& (*targetm.binds_local_p) (decl))
|| (DEFAULT_ABI == ABI_V4
&& (!TARGET_SECURE_PLT
|| !flag_pic
|| (decl
&& (*targetm.binds_local_p) (decl)))))
{
tree attr_list = TYPE_ATTRIBUTES (fntype);
if (!lookup_attribute ("longcall", attr_list)
|| lookup_attribute ("shortcall", attr_list))
return true;
}
return false;
}
static int
rs6000_ra_ever_killed (void)
{
rtx_insn *top;
rtx reg;
rtx_insn *insn;
if (cfun->is_thunk)
return 0;
if (cfun->machine->lr_save_state)
return cfun->machine->lr_save_state - 1;
push_topmost_sequence ();
top = get_insns ();
pop_topmost_sequence ();
reg = gen_rtx_REG (Pmode, LR_REGNO);
for (insn = NEXT_INSN (top); insn != NULL_RTX; insn = NEXT_INSN (insn))
{
if (INSN_P (insn))
{
if (CALL_P (insn))
{
if (!SIBLING_CALL_P (insn))
return 1;
}
else if (find_regno_note (insn, REG_INC, LR_REGNO))
return 1;
else if (set_of (reg, insn) != NULL_RTX
&& !prologue_epilogue_contains (insn))
return 1;
}
}
return 0;
}

void
rs6000_emit_load_toc_table (int fromprolog)
{
rtx dest;
dest = gen_rtx_REG (Pmode, RS6000_PIC_OFFSET_TABLE_REGNUM);
if (TARGET_ELF && TARGET_SECURE_PLT && DEFAULT_ABI == ABI_V4 && flag_pic)
{
char buf[30];
rtx lab, tmp1, tmp2, got;
lab = gen_label_rtx ();
ASM_GENERATE_INTERNAL_LABEL (buf, "L", CODE_LABEL_NUMBER (lab));
lab = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (buf));
if (flag_pic == 2)
{
got = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (toc_label_name));
need_toc_init = 1;
}
else
got = rs6000_got_sym ();
tmp1 = tmp2 = dest;
if (!fromprolog)
{
tmp1 = gen_reg_rtx (Pmode);
tmp2 = gen_reg_rtx (Pmode);
}
emit_insn (gen_load_toc_v4_PIC_1 (lab));
emit_move_insn (tmp1, gen_rtx_REG (Pmode, LR_REGNO));
emit_insn (gen_load_toc_v4_PIC_3b (tmp2, tmp1, got, lab));
emit_insn (gen_load_toc_v4_PIC_3c (dest, tmp2, got, lab));
}
else if (TARGET_ELF && DEFAULT_ABI == ABI_V4 && flag_pic == 1)
{
emit_insn (gen_load_toc_v4_pic_si ());
emit_move_insn (dest, gen_rtx_REG (Pmode, LR_REGNO));
}
else if (TARGET_ELF && DEFAULT_ABI == ABI_V4 && flag_pic == 2)
{
char buf[30];
rtx temp0 = (fromprolog
? gen_rtx_REG (Pmode, 0)
: gen_reg_rtx (Pmode));
if (fromprolog)
{
rtx symF, symL;
ASM_GENERATE_INTERNAL_LABEL (buf, "LCF", rs6000_pic_labelno);
symF = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (buf));
ASM_GENERATE_INTERNAL_LABEL (buf, "LCL", rs6000_pic_labelno);
symL = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (buf));
emit_insn (gen_load_toc_v4_PIC_1 (symF));
emit_move_insn (dest, gen_rtx_REG (Pmode, LR_REGNO));
emit_insn (gen_load_toc_v4_PIC_2 (temp0, dest, symL, symF));
}
else
{
rtx tocsym, lab;
tocsym = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (toc_label_name));
need_toc_init = 1;
lab = gen_label_rtx ();
emit_insn (gen_load_toc_v4_PIC_1b (tocsym, lab));
emit_move_insn (dest, gen_rtx_REG (Pmode, LR_REGNO));
if (TARGET_LINK_STACK)
emit_insn (gen_addsi3 (dest, dest, GEN_INT (4)));
emit_move_insn (temp0, gen_rtx_MEM (Pmode, dest));
}
emit_insn (gen_addsi3 (dest, temp0, dest));
}
else if (TARGET_ELF && !TARGET_AIX && flag_pic == 0 && TARGET_MINIMAL_TOC)
{
rtx realsym = gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (toc_label_name));
need_toc_init = 1;
emit_insn (gen_elf_high (dest, realsym));
emit_insn (gen_elf_low (dest, dest, realsym));
}
else
{
gcc_assert (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2);
if (TARGET_32BIT)
emit_insn (gen_load_toc_aix_si (dest));
else
emit_insn (gen_load_toc_aix_di (dest));
}
}
void
rs6000_emit_eh_reg_restore (rtx source, rtx scratch)
{
rs6000_stack_t *info = rs6000_stack_info ();
rtx operands[2];
operands[0] = source;
operands[1] = scratch;
if (info->lr_save_p)
{
rtx frame_rtx = stack_pointer_rtx;
HOST_WIDE_INT sp_offset = 0;
rtx tmp;
if (frame_pointer_needed
|| cfun->calls_alloca
|| info->total_size > 32767)
{
tmp = gen_frame_mem (Pmode, frame_rtx);
emit_move_insn (operands[1], tmp);
frame_rtx = operands[1];
}
else if (info->push_p)
sp_offset = info->total_size;
tmp = plus_constant (Pmode, frame_rtx,
info->lr_save_offset + sp_offset);
tmp = gen_frame_mem (Pmode, tmp);
emit_move_insn (tmp, operands[0]);
}
else
emit_move_insn (gen_rtx_REG (Pmode, LR_REGNO), operands[0]);
cfun->machine->lr_save_state = info->lr_save_p + 1;
}
static GTY(()) alias_set_type set = -1;
alias_set_type
get_TOC_alias_set (void)
{
if (set == -1)
set = new_alias_set ();
return set;
}
#if TARGET_ELF
static int
uses_TOC (void)
{
rtx_insn *insn;
int ret = 1;
for (insn = get_insns (); insn; insn = NEXT_INSN (insn))
{
if (INSN_P (insn))
{
rtx pat = PATTERN (insn);
int i;
if (GET_CODE (pat) == PARALLEL)
for (i = 0; i < XVECLEN (pat, 0); i++)
{
rtx sub = XVECEXP (pat, 0, i);
if (GET_CODE (sub) == USE)
{
sub = XEXP (sub, 0);
if (GET_CODE (sub) == UNSPEC
&& XINT (sub, 1) == UNSPEC_TOC)
return ret;
}
}
}
else if (crtl->has_bb_partition
&& NOTE_P (insn)
&& NOTE_KIND (insn) == NOTE_INSN_SWITCH_TEXT_SECTIONS)
ret = 2;
}
return 0;
}
#endif
rtx
create_TOC_reference (rtx symbol, rtx largetoc_reg)
{
rtx tocrel, tocreg, hi;
if (TARGET_DEBUG_ADDR)
{
if (GET_CODE (symbol) == SYMBOL_REF)
fprintf (stderr, "\ncreate_TOC_reference, (symbol_ref %s)\n",
XSTR (symbol, 0));
else
{
fprintf (stderr, "\ncreate_TOC_reference, code %s:\n",
GET_RTX_NAME (GET_CODE (symbol)));
debug_rtx (symbol);
}
}
if (!can_create_pseudo_p ())
df_set_regs_ever_live (TOC_REGISTER, true);
tocreg = gen_rtx_REG (Pmode, TOC_REGISTER);
tocrel = gen_rtx_UNSPEC (Pmode, gen_rtvec (2, symbol, tocreg), UNSPEC_TOCREL);
if (TARGET_CMODEL == CMODEL_SMALL || can_create_pseudo_p ())
return tocrel;
hi = gen_rtx_HIGH (Pmode, copy_rtx (tocrel));
if (largetoc_reg != NULL)
{
emit_move_insn (largetoc_reg, hi);
hi = largetoc_reg;
}
return gen_rtx_LO_SUM (Pmode, hi, tocrel);
}
void
rs6000_aix_asm_output_dwarf_table_ref (char * frame_table_label)
{
fprintf (asm_out_file, "\t.ref %s\n",
(* targetm.strip_name_encoding) (frame_table_label));
}

static void
rs6000_emit_stack_tie (rtx fp, bool hard_frame_needed)
{
rtvec p;
int i;
rtx regs[3];
i = 0;
regs[i++] = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
if (hard_frame_needed)
regs[i++] = gen_rtx_REG (Pmode, HARD_FRAME_POINTER_REGNUM);
if (!(REGNO (fp) == STACK_POINTER_REGNUM
|| (hard_frame_needed
&& REGNO (fp) == HARD_FRAME_POINTER_REGNUM)))
regs[i++] = fp;
p = rtvec_alloc (i);
while (--i >= 0)
{
rtx mem = gen_frame_mem (BLKmode, regs[i]);
RTVEC_ELT (p, i) = gen_rtx_SET (mem, const0_rtx);
}
emit_insn (gen_stack_tie (gen_rtx_PARALLEL (VOIDmode, p)));
}
static rtx_insn *
rs6000_emit_allocate_stack_1 (HOST_WIDE_INT size_int, rtx orig_sp)
{
rtx_insn *insn;
rtx size_rtx = GEN_INT (-size_int);
if (size_int > 32767)
{
rtx tmp_reg = gen_rtx_REG (Pmode, 0);
if (get_last_insn () == NULL_RTX)
emit_note (NOTE_INSN_DELETED);
insn = emit_move_insn (tmp_reg, size_rtx);
try_split (PATTERN (insn), insn, 0);
size_rtx = tmp_reg;
}
if (Pmode == SImode)
insn = emit_insn (gen_movsi_update_stack (stack_pointer_rtx,
stack_pointer_rtx,
size_rtx,
orig_sp));
else
insn = emit_insn (gen_movdi_di_update_stack (stack_pointer_rtx,
stack_pointer_rtx,
size_rtx,
orig_sp));
rtx par = PATTERN (insn);
gcc_assert (GET_CODE (par) == PARALLEL);
rtx set = XVECEXP (par, 0, 0);
gcc_assert (GET_CODE (set) == SET);
rtx mem = SET_DEST (set);
gcc_assert (MEM_P (mem));
MEM_NOTRAP_P (mem) = 1;
set_mem_alias_set (mem, get_frame_alias_set ());
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR,
gen_rtx_SET (stack_pointer_rtx,
gen_rtx_PLUS (Pmode,
stack_pointer_rtx,
GEN_INT (-size_int))));
if (flag_stack_clash_protection)
{
add_reg_note (insn, REG_STACK_CHECK, const0_rtx);
emit_insn (gen_blockage ());
}
return insn;
}
static HOST_WIDE_INT
get_stack_clash_protection_probe_interval (void)
{
return (HOST_WIDE_INT_1U
<< PARAM_VALUE (PARAM_STACK_CLASH_PROTECTION_PROBE_INTERVAL));
}
static HOST_WIDE_INT
get_stack_clash_protection_guard_size (void)
{
return (HOST_WIDE_INT_1U
<< PARAM_VALUE (PARAM_STACK_CLASH_PROTECTION_GUARD_SIZE));
}
static rtx_insn *
rs6000_emit_probe_stack_range_stack_clash (HOST_WIDE_INT orig_size,
rtx copy_reg)
{
rtx orig_sp = copy_reg;
HOST_WIDE_INT probe_interval = get_stack_clash_protection_probe_interval ();
HOST_WIDE_INT rounded_size = ROUND_DOWN (orig_size, probe_interval);
if (rounded_size != orig_size
|| rounded_size > probe_interval
|| copy_reg)
{
if (!copy_reg)
orig_sp = gen_rtx_REG (Pmode, 0);
emit_move_insn (orig_sp, stack_pointer_rtx);
}
rtx_insn *retval = NULL;
if (rounded_size == probe_interval)
{
retval = rs6000_emit_allocate_stack_1 (probe_interval, stack_pointer_rtx);
dump_stack_clash_frame_info (PROBE_INLINE, rounded_size != orig_size);
}
else if (rounded_size <= 8 * probe_interval)
{
for (int i = 0; i < rounded_size; i += probe_interval)
{
rtx_insn *insn
= rs6000_emit_allocate_stack_1 (probe_interval, orig_sp);
if (i == 0)
retval = insn;
}
dump_stack_clash_frame_info (PROBE_INLINE, rounded_size != orig_size);
}
else
{
rtx end_addr
= copy_reg ? gen_rtx_REG (Pmode, 0) : gen_rtx_REG (Pmode, 12);
rtx rs = GEN_INT (-rounded_size);
rtx_insn *insn;
if (add_operand (rs, Pmode))
insn = emit_insn (gen_add3_insn (end_addr, stack_pointer_rtx, rs));
else
{
emit_move_insn (end_addr, GEN_INT (-rounded_size));
insn = emit_insn (gen_add3_insn (end_addr, end_addr,
stack_pointer_rtx));
add_reg_note (insn, REG_FRAME_RELATED_EXPR,
gen_rtx_SET (end_addr,
gen_rtx_PLUS (Pmode, stack_pointer_rtx,
rs)));
}
RTX_FRAME_RELATED_P (insn) = 1;
if (TARGET_64BIT)
retval = emit_insn (gen_probe_stack_rangedi (stack_pointer_rtx,
stack_pointer_rtx, orig_sp,
end_addr));
else
retval = emit_insn (gen_probe_stack_rangesi (stack_pointer_rtx,
stack_pointer_rtx, orig_sp,
end_addr));
RTX_FRAME_RELATED_P (retval) = 1;
add_reg_note (retval, REG_FRAME_RELATED_EXPR,
gen_rtx_SET (stack_pointer_rtx, end_addr));
emit_insn (gen_blockage ());
dump_stack_clash_frame_info (PROBE_LOOP, rounded_size != orig_size);
}
if (orig_size != rounded_size)
{
HOST_WIDE_INT residual = orig_size - rounded_size;
rtx_insn *insn = rs6000_emit_allocate_stack_1 (residual, orig_sp);
if (!retval)
retval = insn;
}
return retval;
}
static rtx_insn *
rs6000_emit_allocate_stack (HOST_WIDE_INT size, rtx copy_reg, int copy_off)
{
rtx_insn *insn;
rtx stack_reg = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
rtx tmp_reg = gen_rtx_REG (Pmode, 0);
rtx todec = gen_int_mode (-size, Pmode);
if (INTVAL (todec) != -size)
{
warning (0, "stack frame too large");
emit_insn (gen_trap ());
return 0;
}
if (crtl->limit_stack)
{
if (REG_P (stack_limit_rtx)
&& REGNO (stack_limit_rtx) > 1
&& REGNO (stack_limit_rtx) <= 31)
{
rtx_insn *insn
= gen_add3_insn (tmp_reg, stack_limit_rtx, GEN_INT (size));
gcc_assert (insn);
emit_insn (insn);
emit_insn (gen_cond_trap (LTU, stack_reg, tmp_reg, const0_rtx));
}
else if (GET_CODE (stack_limit_rtx) == SYMBOL_REF
&& TARGET_32BIT
&& DEFAULT_ABI == ABI_V4
&& !flag_pic)
{
rtx toload = gen_rtx_CONST (VOIDmode,
gen_rtx_PLUS (Pmode,
stack_limit_rtx,
GEN_INT (size)));
emit_insn (gen_elf_high (tmp_reg, toload));
emit_insn (gen_elf_low (tmp_reg, tmp_reg, toload));
emit_insn (gen_cond_trap (LTU, stack_reg, tmp_reg,
const0_rtx));
}
else
warning (0, "stack limit expression is not supported");
}
if (flag_stack_clash_protection)
{
if (size < get_stack_clash_protection_guard_size ())
dump_stack_clash_frame_info (NO_PROBE_SMALL_FRAME, true);
else
{
rtx_insn *insn = rs6000_emit_probe_stack_range_stack_clash (size,
copy_reg);
if (copy_reg && copy_off)
emit_insn (gen_add3_insn (copy_reg, copy_reg, GEN_INT (copy_off)));
return insn;
}
}
if (copy_reg)
{
if (copy_off != 0)
emit_insn (gen_add3_insn (copy_reg, stack_reg, GEN_INT (copy_off)));
else
emit_move_insn (copy_reg, stack_reg);
}
insn = rs6000_emit_allocate_stack_1 (size, stack_reg);
return insn;
}
#define PROBE_INTERVAL (1 << STACK_CHECK_PROBE_INTERVAL_EXP)
#if PROBE_INTERVAL > 32768
#error Cannot use indexed addressing mode for stack probing
#endif
static void
rs6000_emit_probe_stack_range (HOST_WIDE_INT first, HOST_WIDE_INT size)
{
if (first + size <= 32768)
{
HOST_WIDE_INT i;
for (i = PROBE_INTERVAL; i < size; i += PROBE_INTERVAL)
emit_stack_probe (plus_constant (Pmode, stack_pointer_rtx,
-(first + i)));
emit_stack_probe (plus_constant (Pmode, stack_pointer_rtx,
-(first + size)));
}
else
{
HOST_WIDE_INT rounded_size;
rtx r12 = gen_rtx_REG (Pmode, 12);
rtx r0 = gen_rtx_REG (Pmode, 0);
gcc_assert (first <= 32768);
rounded_size = ROUND_DOWN (size, PROBE_INTERVAL);
emit_insn (gen_rtx_SET (r12, plus_constant (Pmode, stack_pointer_rtx,
-first)));
if (rounded_size > 32768)
{
emit_move_insn (r0, GEN_INT (-rounded_size));
emit_insn (gen_rtx_SET (r0, gen_rtx_PLUS (Pmode, r12, r0)));
}
else
emit_insn (gen_rtx_SET (r0, plus_constant (Pmode, r12,
-rounded_size)));
if (TARGET_64BIT)
emit_insn (gen_probe_stack_rangedi (r12, r12, stack_pointer_rtx, r0));
else
emit_insn (gen_probe_stack_rangesi (r12, r12, stack_pointer_rtx, r0));
if (size != rounded_size)
emit_stack_probe (plus_constant (Pmode, r12, rounded_size - size));
}
}
static const char *
output_probe_stack_range_1 (rtx reg1, rtx reg2)
{
static int labelno = 0;
char loop_lab[32];
rtx xops[2];
ASM_GENERATE_INTERNAL_LABEL (loop_lab, "LPSRL", labelno++);
ASM_OUTPUT_INTERNAL_LABEL (asm_out_file, loop_lab);
xops[0] = reg1;
xops[1] = GEN_INT (-PROBE_INTERVAL);
output_asm_insn ("addi %0,%0,%1", xops);
xops[1] = gen_rtx_REG (Pmode, 0);
output_asm_insn ("stw %1,0(%0)", xops);
xops[1] = reg2;
if (TARGET_64BIT)
output_asm_insn ("cmpd 0,%0,%1", xops);
else
output_asm_insn ("cmpw 0,%0,%1", xops);
fputs ("\tbne 0,", asm_out_file);
assemble_name_raw (asm_out_file, loop_lab);
fputc ('\n', asm_out_file);
return "";
}
static bool
interesting_frame_related_regno (unsigned int regno)
{
if (regno == 0)
return true;
if (regno == CR2_REGNO)
return true;
return save_reg_p (regno);
}
static const char *
output_probe_stack_range_stack_clash (rtx reg1, rtx reg2, rtx reg3)
{
static int labelno = 0;
char loop_lab[32];
rtx xops[3];
HOST_WIDE_INT probe_interval = get_stack_clash_protection_probe_interval ();
ASM_GENERATE_INTERNAL_LABEL (loop_lab, "LPSRL", labelno++);
ASM_OUTPUT_INTERNAL_LABEL (asm_out_file, loop_lab);
xops[0] = reg1;
xops[1] = reg2;
xops[2] = GEN_INT (-probe_interval);
if (TARGET_64BIT)
output_asm_insn ("stdu %1,%2(%0)", xops);
else
output_asm_insn ("stwu %1,%2(%0)", xops);
xops[0] = reg1;
xops[1] = reg3;
if (TARGET_64BIT)
output_asm_insn ("cmpd 0,%0,%1", xops);
else
output_asm_insn ("cmpw 0,%0,%1", xops);
fputs ("\tbne 0,", asm_out_file);
assemble_name_raw (asm_out_file, loop_lab);
fputc ('\n', asm_out_file);
return "";
}
const char *
output_probe_stack_range (rtx reg1, rtx reg2, rtx reg3)
{
if (flag_stack_clash_protection)
return output_probe_stack_range_stack_clash (reg1, reg2, reg3);
else
return output_probe_stack_range_1 (reg1, reg3);
}
static rtx_insn *
rs6000_frame_related (rtx_insn *insn, rtx reg, HOST_WIDE_INT val,
rtx reg2, rtx repl2)
{
rtx repl;
if (REGNO (reg) == STACK_POINTER_REGNUM)
{
gcc_checking_assert (val == 0);
repl = NULL_RTX;
}
else
repl = gen_rtx_PLUS (Pmode, gen_rtx_REG (Pmode, STACK_POINTER_REGNUM),
GEN_INT (val));
rtx pat = PATTERN (insn);
if (!repl && !reg2)
{
if (GET_CODE (pat) == PARALLEL)
for (int i = 0; i < XVECLEN (pat, 0); i++)
if (GET_CODE (XVECEXP (pat, 0, i)) == SET)
{
rtx set = XVECEXP (pat, 0, i);
if (!REG_P (SET_SRC (set))
|| interesting_frame_related_regno (REGNO (SET_SRC (set))))
RTX_FRAME_RELATED_P (set) = 1;
}
RTX_FRAME_RELATED_P (insn) = 1;
return insn;
}
set_used_flags (pat);
if (GET_CODE (pat) == SET)
{
if (repl)
pat = simplify_replace_rtx (pat, reg, repl);
if (reg2)
pat = simplify_replace_rtx (pat, reg2, repl2);
}
else if (GET_CODE (pat) == PARALLEL)
{
pat = shallow_copy_rtx (pat);
XVEC (pat, 0) = shallow_copy_rtvec (XVEC (pat, 0));
for (int i = 0; i < XVECLEN (pat, 0); i++)
if (GET_CODE (XVECEXP (pat, 0, i)) == SET)
{
rtx set = XVECEXP (pat, 0, i);
if (repl)
set = simplify_replace_rtx (set, reg, repl);
if (reg2)
set = simplify_replace_rtx (set, reg2, repl2);
XVECEXP (pat, 0, i) = set;
if (!REG_P (SET_SRC (set))
|| interesting_frame_related_regno (REGNO (SET_SRC (set))))
RTX_FRAME_RELATED_P (set) = 1;
}
}
else
gcc_unreachable ();
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, copy_rtx_if_shared (pat));
return insn;
}
static rtx
generate_set_vrsave (rtx reg, rs6000_stack_t *info, int epiloguep)
{
int nclobs, i;
rtx insn, clobs[TOTAL_ALTIVEC_REGS + 1];
rtx vrsave = gen_rtx_REG (SImode, VRSAVE_REGNO);
clobs[0]
= gen_rtx_SET (vrsave,
gen_rtx_UNSPEC_VOLATILE (SImode,
gen_rtvec (2, reg, vrsave),
UNSPECV_SET_VRSAVE));
nclobs = 1;
for (i = FIRST_ALTIVEC_REGNO; i <= LAST_ALTIVEC_REGNO; ++i)
if (info->vrsave_mask & ALTIVEC_REG_BIT (i))
{
if (!epiloguep || call_used_regs [i])
clobs[nclobs++] = gen_rtx_CLOBBER (VOIDmode,
gen_rtx_REG (V4SImode, i));
else
{
rtx reg = gen_rtx_REG (V4SImode, i);
clobs[nclobs++]
= gen_rtx_SET (reg,
gen_rtx_UNSPEC (V4SImode,
gen_rtvec (1, reg), 27));
}
}
insn = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (nclobs));
for (i = 0; i < nclobs; ++i)
XVECEXP (insn, 0, i) = clobs[i];
return insn;
}
static rtx
gen_frame_set (rtx reg, rtx frame_reg, int offset, bool store)
{
rtx addr, mem;
addr = gen_rtx_PLUS (Pmode, frame_reg, GEN_INT (offset));
mem = gen_frame_mem (GET_MODE (reg), addr);
return gen_rtx_SET (store ? mem : reg, store ? reg : mem);
}
static rtx
gen_frame_load (rtx reg, rtx frame_reg, int offset)
{
return gen_frame_set (reg, frame_reg, offset, false);
}
static rtx
gen_frame_store (rtx reg, rtx frame_reg, int offset)
{
return gen_frame_set (reg, frame_reg, offset, true);
}
static rtx_insn *
emit_frame_save (rtx frame_reg, machine_mode mode,
unsigned int regno, int offset, HOST_WIDE_INT frame_reg_to_sp)
{
rtx reg;
gcc_checking_assert (!(TARGET_ALTIVEC_ABI && ALTIVEC_VECTOR_MODE (mode))
|| (TARGET_VSX && ALTIVEC_OR_VSX_VECTOR_MODE (mode)));
reg = gen_rtx_REG (mode, regno);
rtx_insn *insn = emit_insn (gen_frame_store (reg, frame_reg, offset));
return rs6000_frame_related (insn, frame_reg, frame_reg_to_sp,
NULL_RTX, NULL_RTX);
}
static rtx
gen_frame_mem_offset (machine_mode mode, rtx reg, int offset)
{
return gen_frame_mem (mode, gen_rtx_PLUS (Pmode, reg, GEN_INT (offset)));
}
#ifndef TARGET_FIX_AND_CONTINUE
#define TARGET_FIX_AND_CONTINUE 0
#endif
#define FIRST_SAVRES_REGISTER FIRST_SAVED_GP_REGNO
#define LAST_SAVRES_REGISTER 31
#define N_SAVRES_REGISTERS (LAST_SAVRES_REGISTER - FIRST_SAVRES_REGISTER + 1)
enum {
SAVRES_LR = 0x1,
SAVRES_SAVE = 0x2,
SAVRES_REG = 0x0c,
SAVRES_GPR = 0,
SAVRES_FPR = 4,
SAVRES_VR  = 8
};
static GTY(()) rtx savres_routine_syms[N_SAVRES_REGISTERS][12];
static char savres_routine_name[30];
static char *
rs6000_savres_routine_name (int regno, int sel)
{
const char *prefix = "";
const char *suffix = "";
if (DEFAULT_ABI == ABI_V4)
{
if (TARGET_64BIT)
goto aix_names;
if ((sel & SAVRES_REG) == SAVRES_GPR)
prefix = (sel & SAVRES_SAVE) ? "_savegpr_" : "_restgpr_";
else if ((sel & SAVRES_REG) == SAVRES_FPR)
prefix = (sel & SAVRES_SAVE) ? "_savefpr_" : "_restfpr_";
else if ((sel & SAVRES_REG) == SAVRES_VR)
prefix = (sel & SAVRES_SAVE) ? "_savevr_" : "_restvr_";
else
abort ();
if ((sel & SAVRES_LR))
suffix = "_x";
}
else if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
{
#if !defined (POWERPC_LINUX) && !defined (POWERPC_FREEBSD)
gcc_assert (!TARGET_AIX || (sel & SAVRES_REG) != SAVRES_GPR);
#endif
aix_names:
if ((sel & SAVRES_REG) == SAVRES_GPR)
prefix = ((sel & SAVRES_SAVE)
? ((sel & SAVRES_LR) ? "_savegpr0_" : "_savegpr1_")
: ((sel & SAVRES_LR) ? "_restgpr0_" : "_restgpr1_"));
else if ((sel & SAVRES_REG) == SAVRES_FPR)
{
#if defined (POWERPC_LINUX) || defined (POWERPC_FREEBSD)
if ((sel & SAVRES_LR))
prefix = ((sel & SAVRES_SAVE) ? "_savefpr_" : "_restfpr_");
else
#endif
{
prefix = (sel & SAVRES_SAVE) ? SAVE_FP_PREFIX : RESTORE_FP_PREFIX;
suffix = (sel & SAVRES_SAVE) ? SAVE_FP_SUFFIX : RESTORE_FP_SUFFIX;
}
}
else if ((sel & SAVRES_REG) == SAVRES_VR)
prefix = (sel & SAVRES_SAVE) ? "_savevr_" : "_restvr_";
else
abort ();
}
if (DEFAULT_ABI == ABI_DARWIN)
{
prefix = (sel & SAVRES_SAVE) ? "save" : "rest" ;
if ((sel & SAVRES_REG) == SAVRES_GPR)
sprintf (savres_routine_name, "*%sGPR%s%s%.0d ; %s r%d-r31", prefix,
((sel & SAVRES_LR) ? "x" : ""), (regno == 13 ? "" : "+"),
(regno - 13) * 4, prefix, regno);
else if ((sel & SAVRES_REG) == SAVRES_FPR)
sprintf (savres_routine_name, "*%sFP%s%.0d ; %s f%d-f31", prefix,
(regno == 14 ? "" : "+"), (regno - 14) * 4, prefix, regno);
else if ((sel & SAVRES_REG) == SAVRES_VR)
sprintf (savres_routine_name, "*%sVEC%s%.0d ; %s v%d-v31", prefix,
(regno == 20 ? "" : "+"), (regno - 20) * 8, prefix, regno);
else
abort ();
}
else
sprintf (savres_routine_name, "%s%d%s", prefix, regno, suffix);
return savres_routine_name;
}
static rtx
rs6000_savres_routine_sym (rs6000_stack_t *info, int sel)
{
int regno = ((sel & SAVRES_REG) == SAVRES_GPR
? info->first_gp_reg_save
: (sel & SAVRES_REG) == SAVRES_FPR
? info->first_fp_reg_save - 32
: (sel & SAVRES_REG) == SAVRES_VR
? info->first_altivec_reg_save - FIRST_ALTIVEC_REGNO
: -1);
rtx sym;
int select = sel;
gcc_assert (FIRST_SAVRES_REGISTER <= regno
&& regno <= LAST_SAVRES_REGISTER
&& select >= 0 && select <= 12);
sym = savres_routine_syms[regno-FIRST_SAVRES_REGISTER][select];
if (sym == NULL)
{
char *name;
name = rs6000_savres_routine_name (regno, sel);
sym = savres_routine_syms[regno-FIRST_SAVRES_REGISTER][select]
= gen_rtx_SYMBOL_REF (Pmode, ggc_strdup (name));
SYMBOL_REF_FLAGS (sym) |= SYMBOL_FLAG_FUNCTION;
}
return sym;
}
static rtx
rs6000_emit_stack_reset (rtx frame_reg_rtx, HOST_WIDE_INT frame_off,
unsigned updt_regno)
{
if (frame_off == 0 && REGNO (frame_reg_rtx) == updt_regno)
return NULL_RTX;
rtx updt_reg_rtx = gen_rtx_REG (Pmode, updt_regno);
if (DEFAULT_ABI == ABI_V4)
return emit_insn (gen_stack_restore_tie (updt_reg_rtx, frame_reg_rtx,
GEN_INT (frame_off)));
if (frame_off != 0)
return emit_insn (gen_add3_insn (updt_reg_rtx,
frame_reg_rtx, GEN_INT (frame_off)));
else
return emit_move_insn (updt_reg_rtx, frame_reg_rtx);
return NULL_RTX;
}
static inline unsigned
ptr_regno_for_savres (int sel)
{
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
return (sel & SAVRES_REG) == SAVRES_FPR || (sel & SAVRES_LR) ? 1 : 12;
return DEFAULT_ABI == ABI_DARWIN && (sel & SAVRES_REG) == SAVRES_FPR ? 1 : 11;
}
static rtx_insn *
rs6000_emit_savres_rtx (rs6000_stack_t *info,
rtx frame_reg_rtx, int save_area_offset, int lr_offset,
machine_mode reg_mode, int sel)
{
int i;
int offset, start_reg, end_reg, n_regs, use_reg;
int reg_size = GET_MODE_SIZE (reg_mode);
rtx sym;
rtvec p;
rtx par;
rtx_insn *insn;
offset = 0;
start_reg = ((sel & SAVRES_REG) == SAVRES_GPR
? info->first_gp_reg_save
: (sel & SAVRES_REG) == SAVRES_FPR
? info->first_fp_reg_save
: (sel & SAVRES_REG) == SAVRES_VR
? info->first_altivec_reg_save
: -1);
end_reg = ((sel & SAVRES_REG) == SAVRES_GPR
? 32
: (sel & SAVRES_REG) == SAVRES_FPR
? 64
: (sel & SAVRES_REG) == SAVRES_VR
? LAST_ALTIVEC_REGNO + 1
: -1);
n_regs = end_reg - start_reg;
p = rtvec_alloc (3 + ((sel & SAVRES_LR) ? 1 : 0)
+ ((sel & SAVRES_REG) == SAVRES_VR ? 1 : 0)
+ n_regs);
if (!(sel & SAVRES_SAVE) && (sel & SAVRES_LR))
RTVEC_ELT (p, offset++) = ret_rtx;
RTVEC_ELT (p, offset++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, LR_REGNO));
sym = rs6000_savres_routine_sym (info, sel);
RTVEC_ELT (p, offset++) = gen_rtx_USE (VOIDmode, sym);
use_reg = ptr_regno_for_savres (sel);
if ((sel & SAVRES_REG) == SAVRES_VR)
{
RTVEC_ELT (p, offset++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, use_reg));
RTVEC_ELT (p, offset++)
= gen_rtx_USE (VOIDmode, gen_rtx_REG (Pmode, 0));
}
else
RTVEC_ELT (p, offset++)
= gen_rtx_USE (VOIDmode, gen_rtx_REG (Pmode, use_reg));
for (i = 0; i < end_reg - start_reg; i++)
RTVEC_ELT (p, i + offset)
= gen_frame_set (gen_rtx_REG (reg_mode, start_reg + i),
frame_reg_rtx, save_area_offset + reg_size * i,
(sel & SAVRES_SAVE) != 0);
if ((sel & SAVRES_SAVE) && (sel & SAVRES_LR))
RTVEC_ELT (p, i + offset)
= gen_frame_store (gen_rtx_REG (Pmode, 0), frame_reg_rtx, lr_offset);
par = gen_rtx_PARALLEL (VOIDmode, p);
if (!(sel & SAVRES_SAVE) && (sel & SAVRES_LR))
{
insn = emit_jump_insn (par);
JUMP_LABEL (insn) = ret_rtx;
}
else
insn = emit_insn (par);
return insn;
}
static void
rs6000_emit_prologue_move_from_cr (rtx reg)
{
if (DEFAULT_ABI == ABI_ELFv2 && TARGET_MFCRF)
{
int i, cr_reg[8], count = 0;
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
cr_reg[count++] = i;
if (count == 1)
{
rtvec p = rtvec_alloc (1);
rtvec r = rtvec_alloc (2);
RTVEC_ELT (r, 0) = gen_rtx_REG (CCmode, CR0_REGNO + cr_reg[0]);
RTVEC_ELT (r, 1) = GEN_INT (1 << (7 - cr_reg[0]));
RTVEC_ELT (p, 0)
= gen_rtx_SET (reg,
gen_rtx_UNSPEC (SImode, r, UNSPEC_MOVESI_FROM_CR));
emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
return;
}
}
emit_insn (gen_prologue_movesi_from_cr (reg));
}
static bool
split_stack_arg_pointer_used_p (void)
{
if (cfun->machine->split_stack_arg_pointer != NULL_RTX
&& (!REG_P (cfun->machine->split_stack_arg_pointer)
|| (REGNO (cfun->machine->split_stack_arg_pointer)
< FIRST_PSEUDO_REGISTER)))
return true;
rtx_insn *insn;
basic_block bb = ENTRY_BLOCK_PTR_FOR_FN (cfun)->next_bb;
FOR_BB_INSNS (bb, insn)
if (NONDEBUG_INSN_P (insn))
{
if (CALL_P (insn))
return false;
df_ref use;
FOR_EACH_INSN_USE (use, insn)
{
rtx x = DF_REF_REG (use);
if (REG_P (x) && REGNO (x) == 12)
return true;
}
df_ref def;
FOR_EACH_INSN_DEF (def, insn)
{
rtx x = DF_REF_REG (def);
if (REG_P (x) && REGNO (x) == 12)
return false;
}
}
return bitmap_bit_p (DF_LR_OUT (bb), 12);
}
static bool
rs6000_global_entry_point_needed_p (void)
{
if (DEFAULT_ABI != ABI_ELFv2)
return false;
if (TARGET_SINGLE_PIC_BASE)
return false;
if (cfun->is_thunk)
return true;
return cfun->machine->r2_setup_needed;
}
static sbitmap
rs6000_get_separate_components (void)
{
rs6000_stack_t *info = rs6000_stack_info ();
if (WORLD_SAVE_P (info))
return NULL;
gcc_assert (!(info->savres_strategy & SAVE_MULTIPLE)
&& !(info->savres_strategy & REST_MULTIPLE));
cfun->machine->n_components = 64;
sbitmap components = sbitmap_alloc (cfun->machine->n_components);
bitmap_clear (components);
int reg_size = TARGET_32BIT ? 4 : 8;
int fp_reg_size = 8;
if ((info->savres_strategy & SAVE_INLINE_GPRS)
&& (info->savres_strategy & REST_INLINE_GPRS))
{
int offset = info->gp_save_offset;
if (info->push_p)
offset += info->total_size;
for (unsigned regno = info->first_gp_reg_save; regno < 32; regno++)
{
if (IN_RANGE (offset, -0x8000, 0x7fff)
&& save_reg_p (regno))
bitmap_set_bit (components, regno);
offset += reg_size;
}
}
if (frame_pointer_needed)
bitmap_clear_bit (components, HARD_FRAME_POINTER_REGNUM);
if ((TARGET_TOC && TARGET_MINIMAL_TOC)
|| (flag_pic == 1 && DEFAULT_ABI == ABI_V4)
|| (flag_pic && DEFAULT_ABI == ABI_DARWIN))
bitmap_clear_bit (components, RS6000_PIC_OFFSET_TABLE_REGNUM);
if ((info->savres_strategy & SAVE_INLINE_FPRS)
&& (info->savres_strategy & REST_INLINE_FPRS))
{
int offset = info->fp_save_offset;
if (info->push_p)
offset += info->total_size;
for (unsigned regno = info->first_fp_reg_save; regno < 64; regno++)
{
if (IN_RANGE (offset, -0x8000, 0x7fff) && save_reg_p (regno))
bitmap_set_bit (components, regno);
offset += fp_reg_size;
}
}
if (info->lr_save_p
&& !(flag_pic && (DEFAULT_ABI == ABI_V4 || DEFAULT_ABI == ABI_DARWIN))
&& (info->savres_strategy & SAVE_INLINE_GPRS)
&& (info->savres_strategy & REST_INLINE_GPRS)
&& (info->savres_strategy & SAVE_INLINE_FPRS)
&& (info->savres_strategy & REST_INLINE_FPRS)
&& (info->savres_strategy & SAVE_INLINE_VRS)
&& (info->savres_strategy & REST_INLINE_VRS))
{
int offset = info->lr_save_offset;
if (info->push_p)
offset += info->total_size;
if (IN_RANGE (offset, -0x8000, 0x7fff))
bitmap_set_bit (components, 0);
}
if (cfun->machine->save_toc_in_prologue)
bitmap_set_bit (components, 2);
return components;
}
static sbitmap
rs6000_components_for_bb (basic_block bb)
{
rs6000_stack_t *info = rs6000_stack_info ();
bitmap in = DF_LIVE_IN (bb);
bitmap gen = &DF_LIVE_BB_INFO (bb)->gen;
bitmap kill = &DF_LIVE_BB_INFO (bb)->kill;
sbitmap components = sbitmap_alloc (cfun->machine->n_components);
bitmap_clear (components);
for (unsigned regno = info->first_gp_reg_save; regno < 32; regno++)
if (bitmap_bit_p (in, regno)
|| bitmap_bit_p (gen, regno)
|| bitmap_bit_p (kill, regno))
bitmap_set_bit (components, regno);
for (unsigned regno = info->first_fp_reg_save; regno < 64; regno++)
if (bitmap_bit_p (in, regno)
|| bitmap_bit_p (gen, regno)
|| bitmap_bit_p (kill, regno))
bitmap_set_bit (components, regno);
if (bitmap_bit_p (in, LR_REGNO)
|| bitmap_bit_p (gen, LR_REGNO)
|| bitmap_bit_p (kill, LR_REGNO))
bitmap_set_bit (components, 0);
if (bitmap_bit_p (in, TOC_REGNUM)
|| bitmap_bit_p (gen, TOC_REGNUM)
|| bitmap_bit_p (kill, TOC_REGNUM))
bitmap_set_bit (components, 2);
return components;
}
static void
rs6000_disqualify_components (sbitmap components, edge e,
sbitmap edge_components, bool )
{
if (bitmap_bit_p (edge_components, 0)
&& bitmap_bit_p (DF_LIVE_IN (e->dest), 0))
{
if (dump_file)
fprintf (dump_file, "Disqualifying LR because GPR0 is live "
"on entry to bb %d\n", e->dest->index);
bitmap_clear_bit (components, 0);
}
}
static void
rs6000_emit_prologue_components (sbitmap components)
{
rs6000_stack_t *info = rs6000_stack_info ();
rtx ptr_reg = gen_rtx_REG (Pmode, frame_pointer_needed
? HARD_FRAME_POINTER_REGNUM
: STACK_POINTER_REGNUM);
machine_mode reg_mode = Pmode;
int reg_size = TARGET_32BIT ? 4 : 8;
machine_mode fp_reg_mode = (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode;
int fp_reg_size = 8;
if (bitmap_bit_p (components, 0))
{
rtx reg = gen_rtx_REG (reg_mode, 0);
rtx_insn *insn = emit_move_insn (reg, gen_rtx_REG (reg_mode, LR_REGNO));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_CFA_REGISTER, NULL);
int offset = info->lr_save_offset;
if (info->push_p)
offset += info->total_size;
insn = emit_insn (gen_frame_store (reg, ptr_reg, offset));
RTX_FRAME_RELATED_P (insn) = 1;
rtx lr = gen_rtx_REG (reg_mode, LR_REGNO);
rtx mem = copy_rtx (SET_DEST (single_set (insn)));
add_reg_note (insn, REG_CFA_OFFSET, gen_rtx_SET (mem, lr));
}
if (bitmap_bit_p (components, 2))
{
rtx reg = gen_rtx_REG (reg_mode, TOC_REGNUM);
rtx sp_reg = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
emit_insn (gen_frame_store (reg, sp_reg, RS6000_TOC_SAVE_SLOT));
}
int offset = info->gp_save_offset;
if (info->push_p)
offset += info->total_size;
for (int i = info->first_gp_reg_save; i < 32; i++)
{
if (bitmap_bit_p (components, i))
{
rtx reg = gen_rtx_REG (reg_mode, i);
rtx_insn *insn = emit_insn (gen_frame_store (reg, ptr_reg, offset));
RTX_FRAME_RELATED_P (insn) = 1;
rtx set = copy_rtx (single_set (insn));
add_reg_note (insn, REG_CFA_OFFSET, set);
}
offset += reg_size;
}
offset = info->fp_save_offset;
if (info->push_p)
offset += info->total_size;
for (int i = info->first_fp_reg_save; i < 64; i++)
{
if (bitmap_bit_p (components, i))
{
rtx reg = gen_rtx_REG (fp_reg_mode, i);
rtx_insn *insn = emit_insn (gen_frame_store (reg, ptr_reg, offset));
RTX_FRAME_RELATED_P (insn) = 1;
rtx set = copy_rtx (single_set (insn));
add_reg_note (insn, REG_CFA_OFFSET, set);
}
offset += fp_reg_size;
}
}
static void
rs6000_emit_epilogue_components (sbitmap components)
{
rs6000_stack_t *info = rs6000_stack_info ();
rtx ptr_reg = gen_rtx_REG (Pmode, frame_pointer_needed
? HARD_FRAME_POINTER_REGNUM
: STACK_POINTER_REGNUM);
machine_mode reg_mode = Pmode;
int reg_size = TARGET_32BIT ? 4 : 8;
machine_mode fp_reg_mode = (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode;
int fp_reg_size = 8;
int offset = info->fp_save_offset;
if (info->push_p)
offset += info->total_size;
for (int i = info->first_fp_reg_save; i < 64; i++)
{
if (bitmap_bit_p (components, i))
{
rtx reg = gen_rtx_REG (fp_reg_mode, i);
rtx_insn *insn = emit_insn (gen_frame_load (reg, ptr_reg, offset));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_CFA_RESTORE, reg);
}
offset += fp_reg_size;
}
offset = info->gp_save_offset;
if (info->push_p)
offset += info->total_size;
for (int i = info->first_gp_reg_save; i < 32; i++)
{
if (bitmap_bit_p (components, i))
{
rtx reg = gen_rtx_REG (reg_mode, i);
rtx_insn *insn = emit_insn (gen_frame_load (reg, ptr_reg, offset));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_CFA_RESTORE, reg);
}
offset += reg_size;
}
if (bitmap_bit_p (components, 0))
{
int offset = info->lr_save_offset;
if (info->push_p)
offset += info->total_size;
rtx reg = gen_rtx_REG (reg_mode, 0);
rtx_insn *insn = emit_insn (gen_frame_load (reg, ptr_reg, offset));
rtx lr = gen_rtx_REG (Pmode, LR_REGNO);
insn = emit_move_insn (lr, reg);
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_CFA_RESTORE, lr);
}
}
static void
rs6000_set_handled_components (sbitmap components)
{
rs6000_stack_t *info = rs6000_stack_info ();
for (int i = info->first_gp_reg_save; i < 32; i++)
if (bitmap_bit_p (components, i))
cfun->machine->gpr_is_wrapped_separately[i] = true;
for (int i = info->first_fp_reg_save; i < 64; i++)
if (bitmap_bit_p (components, i))
cfun->machine->fpr_is_wrapped_separately[i - 32] = true;
if (bitmap_bit_p (components, 0))
cfun->machine->lr_is_wrapped_separately = true;
if (bitmap_bit_p (components, 2))
cfun->machine->toc_is_wrapped_separately = true;
}
static void
emit_vrsave_prologue (rs6000_stack_t *info, int save_regno,
HOST_WIDE_INT frame_off, rtx frame_reg_rtx)
{
rtx reg = gen_rtx_REG (SImode, save_regno);
rtx vrsave = gen_rtx_REG (SImode, VRSAVE_REGNO);
if (TARGET_MACHO)
emit_insn (gen_get_vrsave_internal (reg));
else
emit_insn (gen_rtx_SET (reg, vrsave));
int offset = info->vrsave_save_offset + frame_off;
emit_insn (gen_frame_store (reg, frame_reg_rtx, offset));
emit_insn (gen_iorsi3 (reg, reg, GEN_INT (info->vrsave_mask)));
emit_insn (generate_set_vrsave (reg, info, 0));
}
static void
emit_split_stack_prologue (rs6000_stack_t *info, rtx_insn *sp_adjust,
HOST_WIDE_INT frame_off, rtx frame_reg_rtx)
{
cfun->machine->split_stack_argp_used = true;
if (sp_adjust)
{
rtx r12 = gen_rtx_REG (Pmode, 12);
rtx sp_reg_rtx = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
rtx set_r12 = gen_rtx_SET (r12, sp_reg_rtx);
emit_insn_before (set_r12, sp_adjust);
}
else if (frame_off != 0 || REGNO (frame_reg_rtx) != 12)
{
rtx r12 = gen_rtx_REG (Pmode, 12);
if (frame_off == 0)
emit_move_insn (r12, frame_reg_rtx);
else
emit_insn (gen_add3_insn (r12, frame_reg_rtx, GEN_INT (frame_off)));
}
if (info->push_p)
{
rtx r12 = gen_rtx_REG (Pmode, 12);
rtx r29 = gen_rtx_REG (Pmode, 29);
rtx cr7 = gen_rtx_REG (CCUNSmode, CR7_REGNO);
rtx not_more = gen_label_rtx ();
rtx jump;
jump = gen_rtx_IF_THEN_ELSE (VOIDmode,
gen_rtx_GEU (VOIDmode, cr7, const0_rtx),
gen_rtx_LABEL_REF (VOIDmode, not_more),
pc_rtx);
jump = emit_jump_insn (gen_rtx_SET (pc_rtx, jump));
JUMP_LABEL (jump) = not_more;
LABEL_NUSES (not_more) += 1;
emit_move_insn (r12, r29);
emit_label (not_more);
}
}
void
rs6000_emit_prologue (void)
{
rs6000_stack_t *info = rs6000_stack_info ();
machine_mode reg_mode = Pmode;
int reg_size = TARGET_32BIT ? 4 : 8;
machine_mode fp_reg_mode = (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode;
int fp_reg_size = 8;
rtx sp_reg_rtx = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
rtx frame_reg_rtx = sp_reg_rtx;
unsigned int cr_save_regno;
rtx cr_save_rtx = NULL_RTX;
rtx_insn *insn;
int strategy;
int using_static_chain_p = (cfun->static_chain_decl != NULL_TREE
&& df_regs_ever_live_p (STATIC_CHAIN_REGNUM)
&& call_used_regs[STATIC_CHAIN_REGNUM]);
int using_split_stack = (flag_split_stack
&& (lookup_attribute ("no_split_stack",
DECL_ATTRIBUTES (cfun->decl))
== NULL));
HOST_WIDE_INT frame_off = 0;
HOST_WIDE_INT sp_off = 0;
rtx_insn *sp_adjust = 0;
#if CHECKING_P
int reg_inuse = using_static_chain_p ? 1 << 11 : 0;
#define START_USE(R) do \
{						\
gcc_assert ((reg_inuse & (1 << (R))) == 0);	\
reg_inuse |= 1 << (R);			\
} while (0)
#define END_USE(R) do \
{						\
gcc_assert ((reg_inuse & (1 << (R))) != 0);	\
reg_inuse &= ~(1 << (R));			\
} while (0)
#define NOT_INUSE(R) do \
{						\
gcc_assert ((reg_inuse & (1 << (R))) == 0);	\
} while (0)
#else
#define START_USE(R) do {} while (0)
#define END_USE(R) do {} while (0)
#define NOT_INUSE(R) do {} while (0)
#endif
if (DEFAULT_ABI == ABI_ELFv2
&& !TARGET_SINGLE_PIC_BASE)
{
cfun->machine->r2_setup_needed = df_regs_ever_live_p (TOC_REGNUM);
if (TARGET_TOC && TARGET_MINIMAL_TOC
&& !constant_pool_empty_p ())
cfun->machine->r2_setup_needed = true;
}
if (flag_stack_usage_info)
current_function_static_stack_size = info->total_size;
if (flag_stack_check == STATIC_BUILTIN_STACK_CHECK)
{
HOST_WIDE_INT size = info->total_size;
if (crtl->is_leaf && !cfun->calls_alloca)
{
if (size > PROBE_INTERVAL && size > get_stack_check_protect ())
rs6000_emit_probe_stack_range (get_stack_check_protect (),
size - get_stack_check_protect ());
}
else if (size > 0)
rs6000_emit_probe_stack_range (get_stack_check_protect (), size);
}
if (TARGET_FIX_AND_CONTINUE)
{
emit_insn (gen_nop ());
emit_insn (gen_nop ());
emit_insn (gen_nop ());
emit_insn (gen_nop ());
emit_insn (gen_nop ());
}
if (WORLD_SAVE_P (info))
{
int i, j, sz;
rtx treg;
rtvec p;
rtx reg0;
reg0 = gen_rtx_REG (Pmode, 0);
if (info->lr_save_p)
{
insn = emit_move_insn (reg0,
gen_rtx_REG (Pmode, LR_REGNO));
RTX_FRAME_RELATED_P (insn) = 1;
}
gcc_assert (info->gp_save_offset == -220
&& info->fp_save_offset == -144
&& info->lr_save_offset == 8
&& info->cr_save_offset == 4
&& info->push_p
&& info->lr_save_p
&& (!crtl->calls_eh_return
|| info->ehrd_offset == -432)
&& info->vrsave_save_offset == -224
&& info->altivec_save_offset == -416);
treg = gen_rtx_REG (SImode, 11);
emit_move_insn (treg, GEN_INT (-info->total_size));
sz = 5;
sz += 32 - info->first_gp_reg_save;
sz += 64 - info->first_fp_reg_save;
sz += LAST_ALTIVEC_REGNO - info->first_altivec_reg_save + 1;
p = rtvec_alloc (sz);
j = 0;
RTVEC_ELT (p, j++) = gen_rtx_CLOBBER (VOIDmode,
gen_rtx_REG (SImode,
LR_REGNO));
RTVEC_ELT (p, j++) = gen_rtx_USE (VOIDmode,
gen_rtx_SYMBOL_REF (Pmode,
"*save_world"));
for (i = 0; i < 64 - info->first_fp_reg_save; i++)
RTVEC_ELT (p, j++)
= gen_frame_store (gen_rtx_REG (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT
? DFmode : SFmode,
info->first_fp_reg_save + i),
frame_reg_rtx,
info->fp_save_offset + frame_off + 8 * i);
for (i = 0; info->first_altivec_reg_save + i <= LAST_ALTIVEC_REGNO; i++)
RTVEC_ELT (p, j++)
= gen_frame_store (gen_rtx_REG (V4SImode,
info->first_altivec_reg_save + i),
frame_reg_rtx,
info->altivec_save_offset + frame_off + 16 * i);
for (i = 0; i < 32 - info->first_gp_reg_save; i++)
RTVEC_ELT (p, j++)
= gen_frame_store (gen_rtx_REG (reg_mode, info->first_gp_reg_save + i),
frame_reg_rtx,
info->gp_save_offset + frame_off + reg_size * i);
RTVEC_ELT (p, j++)
= gen_frame_store (gen_rtx_REG (SImode, CR2_REGNO),
frame_reg_rtx, info->cr_save_offset + frame_off);
if (info->lr_save_p)
RTVEC_ELT (p, j++)
= gen_frame_store (reg0,
frame_reg_rtx, info->lr_save_offset + frame_off);
{
rtx newval = gen_rtx_PLUS (Pmode, sp_reg_rtx, treg);
RTVEC_ELT (p, j++) = gen_rtx_SET (sp_reg_rtx, newval);
}
insn = emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
rs6000_frame_related (insn, frame_reg_rtx, sp_off - frame_off,
treg, GEN_INT (-info->total_size));
sp_off = frame_off = info->total_size;
}
strategy = info->savres_strategy;
if (! WORLD_SAVE_P (info)
&& info->push_p
&& (DEFAULT_ABI == ABI_V4
|| crtl->calls_eh_return))
{
bool need_r11 = (!(strategy & SAVE_INLINE_FPRS)
|| !(strategy & SAVE_INLINE_GPRS)
|| !(strategy & SAVE_INLINE_VRS));
int ptr_regno = -1;
rtx ptr_reg = NULL_RTX;
int ptr_off = 0;
if (info->total_size < 32767)
frame_off = info->total_size;
else if (need_r11)
ptr_regno = 11;
else if (info->cr_save_p
|| info->lr_save_p
|| info->first_fp_reg_save < 64
|| info->first_gp_reg_save < 32
|| info->altivec_size != 0
|| info->vrsave_size != 0
|| crtl->calls_eh_return)
ptr_regno = 12;
else
{
frame_off = info->total_size;
}
if (ptr_regno != -1)
{
START_USE (ptr_regno);
ptr_reg = gen_rtx_REG (Pmode, ptr_regno);
frame_reg_rtx = ptr_reg;
if (!(strategy & SAVE_INLINE_FPRS) && info->fp_size != 0)
gcc_checking_assert (info->fp_save_offset + info->fp_size == 0);
else if (!(strategy & SAVE_INLINE_GPRS) && info->first_gp_reg_save < 32)
ptr_off = info->gp_save_offset + info->gp_size;
else if (!(strategy & SAVE_INLINE_VRS) && info->altivec_size != 0)
ptr_off = info->altivec_save_offset + info->altivec_size;
frame_off = -ptr_off;
}
sp_adjust = rs6000_emit_allocate_stack (info->total_size,
ptr_reg, ptr_off);
if (REGNO (frame_reg_rtx) == 12)
sp_adjust = 0;
sp_off = info->total_size;
if (frame_reg_rtx != sp_reg_rtx)
rs6000_emit_stack_tie (frame_reg_rtx, false);
}
if (!WORLD_SAVE_P (info) && info->lr_save_p
&& !cfun->machine->lr_is_wrapped_separately)
{
rtx addr, reg, mem;
reg = gen_rtx_REG (Pmode, 0);
START_USE (0);
insn = emit_move_insn (reg, gen_rtx_REG (Pmode, LR_REGNO));
RTX_FRAME_RELATED_P (insn) = 1;
if (!(strategy & (SAVE_NOINLINE_GPRS_SAVES_LR
| SAVE_NOINLINE_FPRS_SAVES_LR)))
{
addr = gen_rtx_PLUS (Pmode, frame_reg_rtx,
GEN_INT (info->lr_save_offset + frame_off));
mem = gen_rtx_MEM (Pmode, addr);
insn = emit_move_insn (mem, reg);
rs6000_frame_related (insn, frame_reg_rtx, sp_off - frame_off,
NULL_RTX, NULL_RTX);
END_USE (0);
}
}
cr_save_regno = ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& !(strategy & (SAVE_INLINE_GPRS
| SAVE_NOINLINE_GPRS_SAVES_LR))
? 11 : 12);
if (!WORLD_SAVE_P (info)
&& info->cr_save_p
&& REGNO (frame_reg_rtx) != cr_save_regno
&& !(using_static_chain_p && cr_save_regno == 11)
&& !(using_split_stack && cr_save_regno == 12 && sp_adjust))
{
cr_save_rtx = gen_rtx_REG (SImode, cr_save_regno);
START_USE (cr_save_regno);
rs6000_emit_prologue_move_from_cr (cr_save_rtx);
}
if (!WORLD_SAVE_P (info) && (strategy & SAVE_INLINE_FPRS))
{
int offset = info->fp_save_offset + frame_off;
for (int i = info->first_fp_reg_save; i < 64; i++)
{
if (save_reg_p (i)
&& !cfun->machine->fpr_is_wrapped_separately[i - 32])
emit_frame_save (frame_reg_rtx, fp_reg_mode, i, offset,
sp_off - frame_off);
offset += fp_reg_size;
}
}
else if (!WORLD_SAVE_P (info) && info->first_fp_reg_save != 64)
{
bool lr = (strategy & SAVE_NOINLINE_FPRS_SAVES_LR) != 0;
int sel = SAVRES_SAVE | SAVRES_FPR | (lr ? SAVRES_LR : 0);
unsigned ptr_regno = ptr_regno_for_savres (sel);
rtx ptr_reg = frame_reg_rtx;
if (REGNO (frame_reg_rtx) == ptr_regno)
gcc_checking_assert (frame_off == 0);
else
{
ptr_reg = gen_rtx_REG (Pmode, ptr_regno);
NOT_INUSE (ptr_regno);
emit_insn (gen_add3_insn (ptr_reg,
frame_reg_rtx, GEN_INT (frame_off)));
}
insn = rs6000_emit_savres_rtx (info, ptr_reg,
info->fp_save_offset,
info->lr_save_offset,
DFmode, sel);
rs6000_frame_related (insn, ptr_reg, sp_off,
NULL_RTX, NULL_RTX);
if (lr)
END_USE (0);
}
if (!WORLD_SAVE_P (info) && !(strategy & SAVE_INLINE_GPRS))
{
bool lr = (strategy & SAVE_NOINLINE_GPRS_SAVES_LR) != 0;
int sel = SAVRES_SAVE | SAVRES_GPR | (lr ? SAVRES_LR : 0);
unsigned ptr_regno = ptr_regno_for_savres (sel);
rtx ptr_reg = frame_reg_rtx;
bool ptr_set_up = REGNO (ptr_reg) == ptr_regno;
int end_save = info->gp_save_offset + info->gp_size;
int ptr_off;
if (ptr_regno == 12)
sp_adjust = 0;
if (!ptr_set_up)
ptr_reg = gen_rtx_REG (Pmode, ptr_regno);
if (end_save + frame_off != 0)
{
rtx offset = GEN_INT (end_save + frame_off);
if (ptr_set_up)
frame_off = -end_save;
else
NOT_INUSE (ptr_regno);
emit_insn (gen_add3_insn (ptr_reg, frame_reg_rtx, offset));
}
else if (!ptr_set_up)
{
NOT_INUSE (ptr_regno);
emit_move_insn (ptr_reg, frame_reg_rtx);
}
ptr_off = -end_save;
insn = rs6000_emit_savres_rtx (info, ptr_reg,
info->gp_save_offset + ptr_off,
info->lr_save_offset + ptr_off,
reg_mode, sel);
rs6000_frame_related (insn, ptr_reg, sp_off - ptr_off,
NULL_RTX, NULL_RTX);
if (lr)
END_USE (0);
}
else if (!WORLD_SAVE_P (info) && (strategy & SAVE_MULTIPLE))
{
rtvec p;
int i;
p = rtvec_alloc (32 - info->first_gp_reg_save);
for (i = 0; i < 32 - info->first_gp_reg_save; i++)
RTVEC_ELT (p, i)
= gen_frame_store (gen_rtx_REG (reg_mode, info->first_gp_reg_save + i),
frame_reg_rtx,
info->gp_save_offset + frame_off + reg_size * i);
insn = emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
rs6000_frame_related (insn, frame_reg_rtx, sp_off - frame_off,
NULL_RTX, NULL_RTX);
}
else if (!WORLD_SAVE_P (info))
{
int offset = info->gp_save_offset + frame_off;
for (int i = info->first_gp_reg_save; i < 32; i++)
{
if (save_reg_p (i)
&& !cfun->machine->gpr_is_wrapped_separately[i])
emit_frame_save (frame_reg_rtx, reg_mode, i, offset,
sp_off - frame_off);
offset += reg_size;
}
}
if (crtl->calls_eh_return)
{
unsigned int i;
rtvec p;
for (i = 0; ; ++i)
{
unsigned int regno = EH_RETURN_DATA_REGNO (i);
if (regno == INVALID_REGNUM)
break;
}
p = rtvec_alloc (i);
for (i = 0; ; ++i)
{
unsigned int regno = EH_RETURN_DATA_REGNO (i);
if (regno == INVALID_REGNUM)
break;
rtx set
= gen_frame_store (gen_rtx_REG (reg_mode, regno),
sp_reg_rtx,
info->ehrd_offset + sp_off + reg_size * (int) i);
RTVEC_ELT (p, i) = set;
RTX_FRAME_RELATED_P (set) = 1;
}
insn = emit_insn (gen_blockage ());
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, gen_rtx_PARALLEL (VOIDmode, p));
}
if (TARGET_AIX && crtl->calls_eh_return)
{
rtx tmp_reg, tmp_reg_si, hi, lo, compare_result, toc_save_done, jump;
rtx join_insn, note;
rtx_insn *save_insn;
long toc_restore_insn;
tmp_reg = gen_rtx_REG (Pmode, 11);
tmp_reg_si = gen_rtx_REG (SImode, 11);
if (using_static_chain_p)
{
START_USE (0);
emit_move_insn (gen_rtx_REG (Pmode, 0), tmp_reg);
}
else
START_USE (11);
emit_move_insn (tmp_reg, gen_rtx_REG (Pmode, LR_REGNO));
emit_move_insn (tmp_reg_si, gen_rtx_MEM (SImode, tmp_reg));
toc_restore_insn = ((TARGET_32BIT ? 0x80410000 : 0xE8410000)
+ RS6000_TOC_SAVE_SLOT);
hi = gen_int_mode (toc_restore_insn & ~0xffff, SImode);
emit_insn (gen_xorsi3 (tmp_reg_si, tmp_reg_si, hi));
compare_result = gen_rtx_REG (CCUNSmode, CR0_REGNO);
validate_condition_mode (EQ, CCUNSmode);
lo = gen_int_mode (toc_restore_insn & 0xffff, SImode);
emit_insn (gen_rtx_SET (compare_result,
gen_rtx_COMPARE (CCUNSmode, tmp_reg_si, lo)));
toc_save_done = gen_label_rtx ();
jump = gen_rtx_IF_THEN_ELSE (VOIDmode,
gen_rtx_EQ (VOIDmode, compare_result,
const0_rtx),
gen_rtx_LABEL_REF (VOIDmode, toc_save_done),
pc_rtx);
jump = emit_jump_insn (gen_rtx_SET (pc_rtx, jump));
JUMP_LABEL (jump) = toc_save_done;
LABEL_NUSES (toc_save_done) += 1;
save_insn = emit_frame_save (frame_reg_rtx, reg_mode,
TOC_REGNUM, frame_off + RS6000_TOC_SAVE_SLOT,
sp_off - frame_off);
emit_label (toc_save_done);
note = find_reg_note (save_insn, REG_FRAME_RELATED_EXPR, NULL);
if (note)
remove_note (save_insn, note);
else
note = alloc_reg_note (REG_FRAME_RELATED_EXPR,
copy_rtx (PATTERN (save_insn)), NULL_RTX);
RTX_FRAME_RELATED_P (save_insn) = 0;
join_insn = emit_insn (gen_blockage ());
REG_NOTES (join_insn) = note;
RTX_FRAME_RELATED_P (join_insn) = 1;
if (using_static_chain_p)
{
emit_move_insn (tmp_reg, gen_rtx_REG (Pmode, 0));
END_USE (0);
}
else
END_USE (11);
}
if (!WORLD_SAVE_P (info) && info->cr_save_p)
{
rtx addr = gen_rtx_PLUS (Pmode, frame_reg_rtx,
GEN_INT (info->cr_save_offset + frame_off));
rtx mem = gen_frame_mem (SImode, addr);
if (cr_save_rtx == NULL_RTX)
{
START_USE (0);
cr_save_rtx = gen_rtx_REG (SImode, 0);
rs6000_emit_prologue_move_from_cr (cr_save_rtx);
}
rtx crsave_v[9];
int n_crsave = 0;
int i;
crsave_v[n_crsave++] = gen_rtx_SET (mem, cr_save_rtx);
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
crsave_v[n_crsave++]
= gen_rtx_USE (VOIDmode, gen_rtx_REG (CCmode, CR0_REGNO + i));
insn = emit_insn (gen_rtx_PARALLEL (VOIDmode,
gen_rtvec_v (n_crsave, crsave_v)));
END_USE (REGNO (cr_save_rtx));
RTX_FRAME_RELATED_P (insn) = 1;
addr = gen_rtx_PLUS (Pmode, gen_rtx_REG (Pmode, STACK_POINTER_REGNUM),
GEN_INT (info->cr_save_offset + sp_off));
mem = gen_frame_mem (SImode, addr);
if (DEFAULT_ABI == ABI_ELFv2)
{
rtx crframe[8];
int n_crframe = 0;
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
{
crframe[n_crframe]
= gen_rtx_SET (mem, gen_rtx_REG (SImode, CR0_REGNO + i));
RTX_FRAME_RELATED_P (crframe[n_crframe]) = 1;
n_crframe++;
}
add_reg_note (insn, REG_FRAME_RELATED_EXPR,
gen_rtx_PARALLEL (VOIDmode,
gen_rtvec_v (n_crframe, crframe)));
}
else
{
rtx set = gen_rtx_SET (mem, gen_rtx_REG (SImode, CR2_REGNO));
add_reg_note (insn, REG_FRAME_RELATED_EXPR, set);
}
}
if (DEFAULT_ABI == ABI_ELFv2 && crtl->calls_eh_return)
{
int i, cr_off = info->ehcr_offset;
rtx crsave;
crsave = gen_rtx_REG (SImode, 0);
emit_insn (gen_prologue_movesi_from_cr (crsave));
for (i = 0; i < 8; i++)
if (!call_used_regs[CR0_REGNO + i])
{
rtvec p = rtvec_alloc (2);
RTVEC_ELT (p, 0)
= gen_frame_store (crsave, frame_reg_rtx, cr_off + frame_off);
RTVEC_ELT (p, 1)
= gen_rtx_USE (VOIDmode, gen_rtx_REG (CCmode, CR0_REGNO + i));
insn = emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR,
gen_frame_store (gen_rtx_REG (SImode, CR0_REGNO + i),
sp_reg_rtx, cr_off + sp_off));
cr_off += reg_size;
}
}
if (flag_stack_clash_protection
&& dump_file
&& !info->push_p)
dump_stack_clash_frame_info (NO_PROBE_NO_FRAME, false);
if (!WORLD_SAVE_P (info) && info->push_p
&& !(DEFAULT_ABI == ABI_V4 || crtl->calls_eh_return))
{
rtx ptr_reg = NULL;
int ptr_off = 0;
if ((strategy & SAVE_INLINE_VRS) == 0
|| (info->altivec_size != 0
&& (info->altivec_save_offset + info->altivec_size - 16
+ info->total_size - frame_off) > 32767)
|| (info->vrsave_size != 0
&& (info->vrsave_save_offset
+ info->total_size - frame_off) > 32767))
{
int sel = SAVRES_SAVE | SAVRES_VR;
unsigned ptr_regno = ptr_regno_for_savres (sel);
if (using_static_chain_p
&& ptr_regno == STATIC_CHAIN_REGNUM)
ptr_regno = 12;
if (REGNO (frame_reg_rtx) != ptr_regno)
START_USE (ptr_regno);
ptr_reg = gen_rtx_REG (Pmode, ptr_regno);
frame_reg_rtx = ptr_reg;
ptr_off = info->altivec_save_offset + info->altivec_size;
frame_off = -ptr_off;
}
else if (REGNO (frame_reg_rtx) == 1)
frame_off = info->total_size;
sp_adjust = rs6000_emit_allocate_stack (info->total_size,
ptr_reg, ptr_off);
if (REGNO (frame_reg_rtx) == 12)
sp_adjust = 0;
sp_off = info->total_size;
if (frame_reg_rtx != sp_reg_rtx)
rs6000_emit_stack_tie (frame_reg_rtx, false);
}
if (frame_pointer_needed)
{
insn = emit_move_insn (gen_rtx_REG (Pmode, HARD_FRAME_POINTER_REGNUM),
sp_reg_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
if (!WORLD_SAVE_P (info)
&& info->altivec_size != 0 && (strategy & SAVE_INLINE_VRS) == 0)
{
int end_save = info->altivec_save_offset + info->altivec_size;
int ptr_off;
rtx ptr_reg = gen_rtx_REG (Pmode, 0);
int scratch_regno = ptr_regno_for_savres (SAVRES_SAVE | SAVRES_VR);
rtx scratch_reg = gen_rtx_REG (Pmode, scratch_regno);
gcc_checking_assert (scratch_regno == 11 || scratch_regno == 12);
NOT_INUSE (0);
if (scratch_regno == 12)
sp_adjust = 0;
if (end_save + frame_off != 0)
{
rtx offset = GEN_INT (end_save + frame_off);
emit_insn (gen_add3_insn (ptr_reg, frame_reg_rtx, offset));
}
else
emit_move_insn (ptr_reg, frame_reg_rtx);
ptr_off = -end_save;
insn = rs6000_emit_savres_rtx (info, scratch_reg,
info->altivec_save_offset + ptr_off,
0, V4SImode, SAVRES_SAVE | SAVRES_VR);
rs6000_frame_related (insn, scratch_reg, sp_off - ptr_off,
NULL_RTX, NULL_RTX);
if (REGNO (frame_reg_rtx) == REGNO (scratch_reg))
{
emit_move_insn (frame_reg_rtx, ptr_reg);
frame_off = ptr_off;
}
}
else if (!WORLD_SAVE_P (info)
&& info->altivec_size != 0)
{
int i;
for (i = info->first_altivec_reg_save; i <= LAST_ALTIVEC_REGNO; ++i)
if (info->vrsave_mask & ALTIVEC_REG_BIT (i))
{
rtx areg, savereg, mem;
HOST_WIDE_INT offset;
offset = (info->altivec_save_offset + frame_off
+ 16 * (i - info->first_altivec_reg_save));
savereg = gen_rtx_REG (V4SImode, i);
if (TARGET_P9_VECTOR && quad_address_offset_p (offset))
{
mem = gen_frame_mem (V4SImode,
gen_rtx_PLUS (Pmode, frame_reg_rtx,
GEN_INT (offset)));
insn = emit_insn (gen_rtx_SET (mem, savereg));
areg = NULL_RTX;
}
else
{
NOT_INUSE (0);
areg = gen_rtx_REG (Pmode, 0);
emit_move_insn (areg, GEN_INT (offset));
mem = gen_frame_mem (V4SImode,
gen_rtx_PLUS (Pmode, frame_reg_rtx, areg));
insn = emit_insn (gen_altivec_stvx_v4si_internal (mem, savereg));
}
rs6000_frame_related (insn, frame_reg_rtx, sp_off - frame_off,
areg, GEN_INT (offset));
}
}
if (!WORLD_SAVE_P (info) && info->vrsave_size != 0)
{
int save_regno = 12;
if ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& !using_static_chain_p)
save_regno = 11;
else if (using_split_stack || REGNO (frame_reg_rtx) == 12)
{
save_regno = 11;
if (using_static_chain_p)
save_regno = 0;
}
NOT_INUSE (save_regno);
emit_vrsave_prologue (info, save_regno, frame_off, frame_reg_rtx);
}
if (!TARGET_SINGLE_PIC_BASE
&& ((TARGET_TOC && TARGET_MINIMAL_TOC
&& !constant_pool_empty_p ())
|| (DEFAULT_ABI == ABI_V4
&& (flag_pic == 1 || (flag_pic && TARGET_SECURE_PLT))
&& df_regs_ever_live_p (RS6000_PIC_OFFSET_TABLE_REGNUM))))
{
int save_LR_around_toc_setup = (TARGET_ELF
&& DEFAULT_ABI == ABI_V4
&& flag_pic
&& ! info->lr_save_p
&& EDGE_COUNT (EXIT_BLOCK_PTR_FOR_FN (cfun)->preds) > 0);
if (save_LR_around_toc_setup)
{
rtx lr = gen_rtx_REG (Pmode, LR_REGNO);
rtx tmp = gen_rtx_REG (Pmode, 12);
sp_adjust = 0;
insn = emit_move_insn (tmp, lr);
RTX_FRAME_RELATED_P (insn) = 1;
rs6000_emit_load_toc_table (TRUE);
insn = emit_move_insn (lr, tmp);
add_reg_note (insn, REG_CFA_RESTORE, lr);
RTX_FRAME_RELATED_P (insn) = 1;
}
else
rs6000_emit_load_toc_table (TRUE);
}
#if TARGET_MACHO
if (!TARGET_SINGLE_PIC_BASE
&& DEFAULT_ABI == ABI_DARWIN
&& flag_pic && crtl->uses_pic_offset_table)
{
rtx lr = gen_rtx_REG (Pmode, LR_REGNO);
rtx src = gen_rtx_SYMBOL_REF (Pmode, MACHOPIC_FUNCTION_BASE_NAME);
if (!info->lr_save_p)
emit_move_insn (gen_rtx_REG (Pmode, 0), lr);
emit_insn (gen_load_macho_picbase (src));
emit_move_insn (gen_rtx_REG (Pmode,
RS6000_PIC_OFFSET_TABLE_REGNUM),
lr);
if (!info->lr_save_p)
emit_move_insn (lr, gen_rtx_REG (Pmode, 0));
}
#endif
if (rs6000_save_toc_in_prologue_p ()
&& !cfun->machine->toc_is_wrapped_separately)
{
rtx reg = gen_rtx_REG (reg_mode, TOC_REGNUM);
emit_insn (gen_frame_store (reg, sp_reg_rtx, RS6000_TOC_SAVE_SLOT));
}
if (using_split_stack && split_stack_arg_pointer_used_p ())
emit_split_stack_prologue (info, sp_adjust, frame_off, frame_reg_rtx);
}
static void
rs6000_output_savres_externs (FILE *file)
{
rs6000_stack_t *info = rs6000_stack_info ();
if (TARGET_DEBUG_STACK)
debug_stack_info (info);
if (info->first_fp_reg_save < 64
&& !TARGET_MACHO
&& !TARGET_ELF)
{
char *name;
int regno = info->first_fp_reg_save - 32;
if ((info->savres_strategy & SAVE_INLINE_FPRS) == 0)
{
bool lr = (info->savres_strategy & SAVE_NOINLINE_FPRS_SAVES_LR) != 0;
int sel = SAVRES_SAVE | SAVRES_FPR | (lr ? SAVRES_LR : 0);
name = rs6000_savres_routine_name (regno, sel);
fprintf (file, "\t.extern %s\n", name);
}
if ((info->savres_strategy & REST_INLINE_FPRS) == 0)
{
bool lr = (info->savres_strategy
& REST_NOINLINE_FPRS_DOESNT_RESTORE_LR) == 0;
int sel = SAVRES_FPR | (lr ? SAVRES_LR : 0);
name = rs6000_savres_routine_name (regno, sel);
fprintf (file, "\t.extern %s\n", name);
}
}
}
static void
rs6000_output_function_prologue (FILE *file)
{
if (!cfun->is_thunk)
rs6000_output_savres_externs (file);
if (rs6000_global_entry_point_needed_p ())
{
const char *name = XSTR (XEXP (DECL_RTL (current_function_decl), 0), 0);
(*targetm.asm_out.internal_label) (file, "LCF", rs6000_pic_labelno);
if (TARGET_CMODEL != CMODEL_LARGE)
{
char buf[256];
ASM_GENERATE_INTERNAL_LABEL (buf, "LCF", rs6000_pic_labelno);
fprintf (file, "0:\taddis 2,12,.TOC.-");
assemble_name (file, buf);
fprintf (file, "@ha\n");
fprintf (file, "\taddi 2,2,.TOC.-");
assemble_name (file, buf);
fprintf (file, "@l\n");
}
else
{
char buf[256];
#ifdef HAVE_AS_ENTRY_MARKERS
fprintf (file, "\t.reloc .,R_PPC64_ENTRY\n");
#endif
fprintf (file, "\tld 2,");
ASM_GENERATE_INTERNAL_LABEL (buf, "LCL", rs6000_pic_labelno);
assemble_name (file, buf);
fprintf (file, "-");
ASM_GENERATE_INTERNAL_LABEL (buf, "LCF", rs6000_pic_labelno);
assemble_name (file, buf);
fprintf (file, "(12)\n");
fprintf (file, "\tadd 2,2,12\n");
}
fputs ("\t.localentry\t", file);
assemble_name (file, name);
fputs (",.-", file);
assemble_name (file, name);
fputs ("\n", file);
}
if (TARGET_PROFILE_KERNEL && crtl->profile)
{
gcc_assert (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2);
gcc_assert (!TARGET_32BIT);
asm_fprintf (file, "\tmflr %s\n", reg_names[0]);
if (DEFAULT_ABI != ABI_ELFv2
&& cfun->static_chain_decl != NULL)
{
asm_fprintf (file, "\tstd %s,24(%s)\n",
reg_names[STATIC_CHAIN_REGNUM], reg_names[1]);
fprintf (file, "\tbl %s\n", RS6000_MCOUNT);
asm_fprintf (file, "\tld %s,24(%s)\n",
reg_names[STATIC_CHAIN_REGNUM], reg_names[1]);
}
else
fprintf (file, "\tbl %s\n", RS6000_MCOUNT);
}
rs6000_pic_labelno++;
}
static bool
rs6000_keep_leaf_when_profiled ()
{
return TARGET_PROFILE_KERNEL;
}
#define ALWAYS_RESTORE_ALTIVEC_BEFORE_POP 0
static rtx
load_cr_save (int regno, rtx frame_reg_rtx, int offset, bool exit_func)
{
rtx mem = gen_frame_mem_offset (SImode, frame_reg_rtx, offset);
rtx reg = gen_rtx_REG (SImode, regno);
rtx_insn *insn = emit_move_insn (reg, mem);
if (!exit_func && DEFAULT_ABI == ABI_V4)
{
rtx cr = gen_rtx_REG (SImode, CR2_REGNO);
rtx set = gen_rtx_SET (reg, cr);
add_reg_note (insn, REG_CFA_REGISTER, set);
RTX_FRAME_RELATED_P (insn) = 1;
}
return reg;
}
static void
restore_saved_cr (rtx reg, int using_mfcr_multiple, bool exit_func)
{
int count = 0;
int i;
if (using_mfcr_multiple)
{
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
count++;
gcc_assert (count);
}
if (using_mfcr_multiple && count > 1)
{
rtx_insn *insn;
rtvec p;
int ndx;
p = rtvec_alloc (count);
ndx = 0;
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
{
rtvec r = rtvec_alloc (2);
RTVEC_ELT (r, 0) = reg;
RTVEC_ELT (r, 1) = GEN_INT (1 << (7-i));
RTVEC_ELT (p, ndx) =
gen_rtx_SET (gen_rtx_REG (CCmode, CR0_REGNO + i),
gen_rtx_UNSPEC (CCmode, r, UNSPEC_MOVESI_TO_CR));
ndx++;
}
insn = emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
gcc_assert (ndx == count);
if (!exit_func && DEFAULT_ABI == ABI_ELFv2 && flag_shrink_wrap)
{
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
add_reg_note (insn, REG_CFA_RESTORE,
gen_rtx_REG (SImode, CR0_REGNO + i));
RTX_FRAME_RELATED_P (insn) = 1;
}
}
else
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
{
rtx insn = emit_insn (gen_movsi_to_cr_one
(gen_rtx_REG (CCmode, CR0_REGNO + i), reg));
if (!exit_func && DEFAULT_ABI == ABI_ELFv2 && flag_shrink_wrap)
{
add_reg_note (insn, REG_CFA_RESTORE,
gen_rtx_REG (SImode, CR0_REGNO + i));
RTX_FRAME_RELATED_P (insn) = 1;
}
}
if (!exit_func && DEFAULT_ABI != ABI_ELFv2
&& (DEFAULT_ABI == ABI_V4 || flag_shrink_wrap))
{
rtx_insn *insn = get_last_insn ();
rtx cr = gen_rtx_REG (SImode, CR2_REGNO);
add_reg_note (insn, REG_CFA_RESTORE, cr);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
static void
load_lr_save (int regno, rtx frame_reg_rtx, int offset)
{
rtx mem = gen_frame_mem_offset (Pmode, frame_reg_rtx, offset);
rtx reg = gen_rtx_REG (Pmode, regno);
emit_move_insn (reg, mem);
}
static void
restore_saved_lr (int regno, bool exit_func)
{
rtx reg = gen_rtx_REG (Pmode, regno);
rtx lr = gen_rtx_REG (Pmode, LR_REGNO);
rtx_insn *insn = emit_move_insn (lr, reg);
if (!exit_func && flag_shrink_wrap)
{
add_reg_note (insn, REG_CFA_RESTORE, lr);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
static rtx
add_crlr_cfa_restore (const rs6000_stack_t *info, rtx cfa_restores)
{
if (DEFAULT_ABI == ABI_ELFv2)
{
int i;
for (i = 0; i < 8; i++)
if (save_reg_p (CR0_REGNO + i))
{
rtx cr = gen_rtx_REG (SImode, CR0_REGNO + i);
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, cr,
cfa_restores);
}
}
else if (info->cr_save_p)
cfa_restores = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, CR2_REGNO),
cfa_restores);
if (info->lr_save_p)
cfa_restores = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (Pmode, LR_REGNO),
cfa_restores);
return cfa_restores;
}
static inline bool
offset_below_red_zone_p (HOST_WIDE_INT offset)
{
return offset < (DEFAULT_ABI == ABI_V4
? 0
: TARGET_32BIT ? -220 : -288);
}
static void
emit_cfa_restores (rtx cfa_restores)
{
rtx_insn *insn = get_last_insn ();
rtx *loc = &REG_NOTES (insn);
while (*loc)
loc = &XEXP (*loc, 1);
*loc = cfa_restores;
RTX_FRAME_RELATED_P (insn) = 1;
}
void
rs6000_emit_epilogue (int sibcall)
{
rs6000_stack_t *info;
int restoring_GPRs_inline;
int restoring_FPRs_inline;
int using_load_multiple;
int using_mtcr_multiple;
int use_backchain_to_restore_sp;
int restore_lr;
int strategy;
HOST_WIDE_INT frame_off = 0;
rtx sp_reg_rtx = gen_rtx_REG (Pmode, 1);
rtx frame_reg_rtx = sp_reg_rtx;
rtx cfa_restores = NULL_RTX;
rtx insn;
rtx cr_save_reg = NULL_RTX;
machine_mode reg_mode = Pmode;
int reg_size = TARGET_32BIT ? 4 : 8;
machine_mode fp_reg_mode = (TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT)
? DFmode : SFmode;
int fp_reg_size = 8;
int i;
bool exit_func;
unsigned ptr_regno;
info = rs6000_stack_info ();
strategy = info->savres_strategy;
using_load_multiple = strategy & REST_MULTIPLE;
restoring_FPRs_inline = sibcall || (strategy & REST_INLINE_FPRS);
restoring_GPRs_inline = sibcall || (strategy & REST_INLINE_GPRS);
using_mtcr_multiple = (rs6000_tune == PROCESSOR_PPC601
|| rs6000_tune == PROCESSOR_PPC603
|| rs6000_tune == PROCESSOR_PPC750
|| optimize_size);
use_backchain_to_restore_sp = (info->total_size + (info->lr_save_p
? info->lr_save_offset
: 0) > 32767
|| (cfun->calls_alloca
&& !frame_pointer_needed));
restore_lr = (info->lr_save_p
&& (restoring_FPRs_inline
|| (strategy & REST_NOINLINE_FPRS_DOESNT_RESTORE_LR))
&& (restoring_GPRs_inline
|| info->first_fp_reg_save < 64)
&& !cfun->machine->lr_is_wrapped_separately);
if (WORLD_SAVE_P (info))
{
int i, j;
char rname[30];
const char *alloc_rname;
rtvec p;
p = rtvec_alloc (9
+ 32 - info->first_gp_reg_save
+ LAST_ALTIVEC_REGNO + 1 - info->first_altivec_reg_save
+ 63 + 1 - info->first_fp_reg_save);
strcpy (rname, ((crtl->calls_eh_return) ?
"*eh_rest_world_r10" : "*rest_world"));
alloc_rname = ggc_strdup (rname);
j = 0;
RTVEC_ELT (p, j++) = ret_rtx;
RTVEC_ELT (p, j++)
= gen_rtx_USE (VOIDmode, gen_rtx_SYMBOL_REF (Pmode, alloc_rname));
RTVEC_ELT (p, j++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, 11));
{
rtx reg = gen_rtx_REG (SImode, CR2_REGNO);
RTVEC_ELT (p, j++)
= gen_frame_load (reg, frame_reg_rtx, info->cr_save_offset);
if (flag_shrink_wrap)
{
cfa_restores = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (Pmode, LR_REGNO),
cfa_restores);
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
}
for (i = 0; i < 32 - info->first_gp_reg_save; i++)
{
rtx reg = gen_rtx_REG (reg_mode, info->first_gp_reg_save + i);
RTVEC_ELT (p, j++)
= gen_frame_load (reg,
frame_reg_rtx, info->gp_save_offset + reg_size * i);
if (flag_shrink_wrap
&& save_reg_p (info->first_gp_reg_save + i))
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
for (i = 0; info->first_altivec_reg_save + i <= LAST_ALTIVEC_REGNO; i++)
{
rtx reg = gen_rtx_REG (V4SImode, info->first_altivec_reg_save + i);
RTVEC_ELT (p, j++)
= gen_frame_load (reg,
frame_reg_rtx, info->altivec_save_offset + 16 * i);
if (flag_shrink_wrap
&& save_reg_p (info->first_altivec_reg_save + i))
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
for (i = 0; info->first_fp_reg_save + i <= 63; i++)
{
rtx reg = gen_rtx_REG ((TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT
? DFmode : SFmode),
info->first_fp_reg_save + i);
RTVEC_ELT (p, j++)
= gen_frame_load (reg, frame_reg_rtx, info->fp_save_offset + 8 * i);
if (flag_shrink_wrap
&& save_reg_p (info->first_fp_reg_save + i))
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
RTVEC_ELT (p, j++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, 0));
RTVEC_ELT (p, j++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (SImode, 12));
RTVEC_ELT (p, j++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (SImode, 7));
RTVEC_ELT (p, j++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (SImode, 8));
RTVEC_ELT (p, j++)
= gen_rtx_USE (VOIDmode, gen_rtx_REG (SImode, 10));
insn = emit_jump_insn (gen_rtx_PARALLEL (VOIDmode, p));
if (flag_shrink_wrap)
{
REG_NOTES (insn) = cfa_restores;
add_reg_note (insn, REG_CFA_DEF_CFA, sp_reg_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
return;
}
if (info->push_p)
frame_off = info->total_size;
if (info->altivec_size != 0
&& (ALWAYS_RESTORE_ALTIVEC_BEFORE_POP
|| (DEFAULT_ABI != ABI_V4
&& offset_below_red_zone_p (info->altivec_save_offset))))
{
int i;
int scratch_regno = ptr_regno_for_savres (SAVRES_VR);
gcc_checking_assert (scratch_regno == 11 || scratch_regno == 12);
if (use_backchain_to_restore_sp)
{
int frame_regno = 11;
if ((strategy & REST_INLINE_VRS) == 0)
{
frame_regno = 11 + 12 - scratch_regno;
}
frame_reg_rtx = gen_rtx_REG (Pmode, frame_regno);
emit_move_insn (frame_reg_rtx,
gen_rtx_MEM (Pmode, sp_reg_rtx));
frame_off = 0;
}
else if (frame_pointer_needed)
frame_reg_rtx = hard_frame_pointer_rtx;
if ((strategy & REST_INLINE_VRS) == 0)
{
int end_save = info->altivec_save_offset + info->altivec_size;
int ptr_off;
rtx ptr_reg = gen_rtx_REG (Pmode, 0);
rtx scratch_reg = gen_rtx_REG (Pmode, scratch_regno);
if (end_save + frame_off != 0)
{
rtx offset = GEN_INT (end_save + frame_off);
emit_insn (gen_add3_insn (ptr_reg, frame_reg_rtx, offset));
}
else
emit_move_insn (ptr_reg, frame_reg_rtx);
ptr_off = -end_save;
insn = rs6000_emit_savres_rtx (info, scratch_reg,
info->altivec_save_offset + ptr_off,
0, V4SImode, SAVRES_VR);
}
else
{
for (i = info->first_altivec_reg_save; i <= LAST_ALTIVEC_REGNO; ++i)
if (info->vrsave_mask & ALTIVEC_REG_BIT (i))
{
rtx addr, areg, mem, insn;
rtx reg = gen_rtx_REG (V4SImode, i);
HOST_WIDE_INT offset
= (info->altivec_save_offset + frame_off
+ 16 * (i - info->first_altivec_reg_save));
if (TARGET_P9_VECTOR && quad_address_offset_p (offset))
{
mem = gen_frame_mem (V4SImode,
gen_rtx_PLUS (Pmode, frame_reg_rtx,
GEN_INT (offset)));
insn = gen_rtx_SET (reg, mem);
}
else
{
areg = gen_rtx_REG (Pmode, 0);
emit_move_insn (areg, GEN_INT (offset));
addr = gen_rtx_PLUS (Pmode, frame_reg_rtx, areg);
mem = gen_frame_mem (V4SImode, addr);
insn = gen_altivec_lvx_v4si_internal (reg, mem);
}
(void) emit_insn (insn);
}
}
for (i = info->first_altivec_reg_save; i <= LAST_ALTIVEC_REGNO; ++i)
if (((strategy & REST_INLINE_VRS) == 0
|| (info->vrsave_mask & ALTIVEC_REG_BIT (i)) != 0)
&& (flag_shrink_wrap
|| (offset_below_red_zone_p
(info->altivec_save_offset
+ 16 * (i - info->first_altivec_reg_save))))
&& save_reg_p (i))
{
rtx reg = gen_rtx_REG (V4SImode, i);
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
}
if (info->vrsave_size != 0
&& (ALWAYS_RESTORE_ALTIVEC_BEFORE_POP
|| (DEFAULT_ABI != ABI_V4
&& offset_below_red_zone_p (info->vrsave_save_offset))))
{
rtx reg;
if (frame_reg_rtx == sp_reg_rtx)
{
if (use_backchain_to_restore_sp)
{
frame_reg_rtx = gen_rtx_REG (Pmode, 11);
emit_move_insn (frame_reg_rtx,
gen_rtx_MEM (Pmode, sp_reg_rtx));
frame_off = 0;
}
else if (frame_pointer_needed)
frame_reg_rtx = hard_frame_pointer_rtx;
}
reg = gen_rtx_REG (SImode, 12);
emit_insn (gen_frame_load (reg, frame_reg_rtx,
info->vrsave_save_offset + frame_off));
emit_insn (generate_set_vrsave (reg, info, 1));
}
insn = NULL_RTX;
if (use_backchain_to_restore_sp)
{
if (frame_reg_rtx == sp_reg_rtx)
{
if (DEFAULT_ABI == ABI_V4)
frame_reg_rtx = gen_rtx_REG (Pmode, 11);
insn = emit_move_insn (frame_reg_rtx,
gen_rtx_MEM (Pmode, sp_reg_rtx));
frame_off = 0;
}
else if (ALWAYS_RESTORE_ALTIVEC_BEFORE_POP
&& DEFAULT_ABI == ABI_V4)
;
else
{
insn = emit_move_insn (sp_reg_rtx, frame_reg_rtx);
frame_reg_rtx = sp_reg_rtx;
}
}
else if (frame_pointer_needed)
{
frame_reg_rtx = sp_reg_rtx;
if (DEFAULT_ABI == ABI_V4)
frame_reg_rtx = gen_rtx_REG (Pmode, 11);
else if (cfun->calls_alloca
|| offset_below_red_zone_p (-info->total_size))
rs6000_emit_stack_tie (frame_reg_rtx, true);
insn = emit_insn (gen_add3_insn (frame_reg_rtx, hard_frame_pointer_rtx,
GEN_INT (info->total_size)));
frame_off = 0;
}
else if (info->push_p
&& DEFAULT_ABI != ABI_V4
&& !crtl->calls_eh_return)
{
if (cfun->calls_alloca
|| offset_below_red_zone_p (-info->total_size))
rs6000_emit_stack_tie (frame_reg_rtx, false);
insn = emit_insn (gen_add3_insn (sp_reg_rtx, sp_reg_rtx,
GEN_INT (info->total_size)));
frame_off = 0;
}
if (insn && frame_reg_rtx == sp_reg_rtx)
{
if (cfa_restores)
{
REG_NOTES (insn) = cfa_restores;
cfa_restores = NULL_RTX;
}
add_reg_note (insn, REG_CFA_DEF_CFA, sp_reg_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
if (!ALWAYS_RESTORE_ALTIVEC_BEFORE_POP
&& info->altivec_size != 0
&& (DEFAULT_ABI == ABI_V4
|| !offset_below_red_zone_p (info->altivec_save_offset)))
{
int i;
if ((strategy & REST_INLINE_VRS) == 0)
{
int end_save = info->altivec_save_offset + info->altivec_size;
int ptr_off;
rtx ptr_reg = gen_rtx_REG (Pmode, 0);
int scratch_regno = ptr_regno_for_savres (SAVRES_VR);
rtx scratch_reg = gen_rtx_REG (Pmode, scratch_regno);
if (end_save + frame_off != 0)
{
rtx offset = GEN_INT (end_save + frame_off);
emit_insn (gen_add3_insn (ptr_reg, frame_reg_rtx, offset));
}
else
emit_move_insn (ptr_reg, frame_reg_rtx);
ptr_off = -end_save;
insn = rs6000_emit_savres_rtx (info, scratch_reg,
info->altivec_save_offset + ptr_off,
0, V4SImode, SAVRES_VR);
if (REGNO (frame_reg_rtx) == REGNO (scratch_reg))
{
unsigned newptr_regno = 1;
if (!restoring_GPRs_inline)
{
bool lr = info->gp_save_offset + info->gp_size == 0;
int sel = SAVRES_GPR | (lr ? SAVRES_LR : 0);
newptr_regno = ptr_regno_for_savres (sel);
end_save = info->gp_save_offset + info->gp_size;
}
else if (!restoring_FPRs_inline)
{
bool lr = !(strategy & REST_NOINLINE_FPRS_DOESNT_RESTORE_LR);
int sel = SAVRES_FPR | (lr ? SAVRES_LR : 0);
newptr_regno = ptr_regno_for_savres (sel);
end_save = info->fp_save_offset + info->fp_size;
}
if (newptr_regno != 1 && REGNO (frame_reg_rtx) != newptr_regno)
frame_reg_rtx = gen_rtx_REG (Pmode, newptr_regno);
if (end_save + ptr_off != 0)
{
rtx offset = GEN_INT (end_save + ptr_off);
frame_off = -end_save;
if (TARGET_32BIT)
emit_insn (gen_addsi3_carry (frame_reg_rtx,
ptr_reg, offset));
else
emit_insn (gen_adddi3_carry (frame_reg_rtx,
ptr_reg, offset));
}
else
{
frame_off = ptr_off;
emit_move_insn (frame_reg_rtx, ptr_reg);
}
}
}
else
{
for (i = info->first_altivec_reg_save; i <= LAST_ALTIVEC_REGNO; ++i)
if (info->vrsave_mask & ALTIVEC_REG_BIT (i))
{
rtx addr, areg, mem, insn;
rtx reg = gen_rtx_REG (V4SImode, i);
HOST_WIDE_INT offset
= (info->altivec_save_offset + frame_off
+ 16 * (i - info->first_altivec_reg_save));
if (TARGET_P9_VECTOR && quad_address_offset_p (offset))
{
mem = gen_frame_mem (V4SImode,
gen_rtx_PLUS (Pmode, frame_reg_rtx,
GEN_INT (offset)));
insn = gen_rtx_SET (reg, mem);
}
else
{
areg = gen_rtx_REG (Pmode, 0);
emit_move_insn (areg, GEN_INT (offset));
addr = gen_rtx_PLUS (Pmode, frame_reg_rtx, areg);
mem = gen_frame_mem (V4SImode, addr);
insn = gen_altivec_lvx_v4si_internal (reg, mem);
}
(void) emit_insn (insn);
}
}
for (i = info->first_altivec_reg_save; i <= LAST_ALTIVEC_REGNO; ++i)
if (((strategy & REST_INLINE_VRS) == 0
|| (info->vrsave_mask & ALTIVEC_REG_BIT (i)) != 0)
&& (DEFAULT_ABI == ABI_V4 || flag_shrink_wrap)
&& save_reg_p (i))
{
rtx reg = gen_rtx_REG (V4SImode, i);
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
}
if (!ALWAYS_RESTORE_ALTIVEC_BEFORE_POP
&& info->vrsave_size != 0
&& (DEFAULT_ABI == ABI_V4
|| !offset_below_red_zone_p (info->vrsave_save_offset)))
{
rtx reg;
reg = gen_rtx_REG (SImode, 12);
emit_insn (gen_frame_load (reg, frame_reg_rtx,
info->vrsave_save_offset + frame_off));
emit_insn (generate_set_vrsave (reg, info, 1));
}
exit_func = (!restoring_FPRs_inline
|| (!restoring_GPRs_inline
&& info->first_fp_reg_save == 64));
if (DEFAULT_ABI == ABI_ELFv2 && crtl->calls_eh_return)
{
int i, cr_off = info->ehcr_offset;
for (i = 0; i < 8; i++)
if (!call_used_regs[CR0_REGNO + i])
{
rtx reg = gen_rtx_REG (SImode, 0);
emit_insn (gen_frame_load (reg, frame_reg_rtx,
cr_off + frame_off));
insn = emit_insn (gen_movsi_to_cr_one
(gen_rtx_REG (CCmode, CR0_REGNO + i), reg));
if (!exit_func && flag_shrink_wrap)
{
add_reg_note (insn, REG_CFA_RESTORE,
gen_rtx_REG (SImode, CR0_REGNO + i));
RTX_FRAME_RELATED_P (insn) = 1;
}
cr_off += reg_size;
}
}
if (restore_lr && restoring_GPRs_inline)
load_lr_save (0, frame_reg_rtx, info->lr_save_offset + frame_off);
if (info->cr_save_p)
{
unsigned cr_save_regno = 12;
if (!restoring_GPRs_inline)
{
bool lr = info->gp_save_offset + info->gp_size == 0;
int sel = SAVRES_GPR | (lr ? SAVRES_LR : 0);
int gpr_ptr_regno = ptr_regno_for_savres (sel);
if (gpr_ptr_regno == 12)
cr_save_regno = 11;
gcc_checking_assert (REGNO (frame_reg_rtx) != cr_save_regno);
}
else if (REGNO (frame_reg_rtx) == 12)
cr_save_regno = 11;
cr_save_reg = load_cr_save (cr_save_regno, frame_reg_rtx,
info->cr_save_offset + frame_off,
exit_func);
}
if (restore_lr && restoring_GPRs_inline)
restore_saved_lr (0, exit_func);
if (crtl->calls_eh_return)
{
unsigned int i, regno;
if (TARGET_AIX)
{
rtx reg = gen_rtx_REG (reg_mode, 2);
emit_insn (gen_frame_load (reg, frame_reg_rtx,
frame_off + RS6000_TOC_SAVE_SLOT));
}
for (i = 0; ; ++i)
{
rtx mem;
regno = EH_RETURN_DATA_REGNO (i);
if (regno == INVALID_REGNUM)
break;
mem = gen_frame_mem_offset (reg_mode, frame_reg_rtx,
info->ehrd_offset + frame_off
+ reg_size * (int) i);
emit_move_insn (gen_rtx_REG (reg_mode, regno), mem);
}
}
if (!restoring_GPRs_inline)
{
rtx ptr_reg;
int end_save = info->gp_save_offset + info->gp_size;
bool can_use_exit = end_save == 0;
int sel = SAVRES_GPR | (can_use_exit ? SAVRES_LR : 0);
int ptr_off;
ptr_regno = ptr_regno_for_savres (sel);
ptr_reg = gen_rtx_REG (Pmode, ptr_regno);
if (can_use_exit)
rs6000_emit_stack_reset (frame_reg_rtx, frame_off, ptr_regno);
else if (end_save + frame_off != 0)
emit_insn (gen_add3_insn (ptr_reg, frame_reg_rtx,
GEN_INT (end_save + frame_off)));
else if (REGNO (frame_reg_rtx) != ptr_regno)
emit_move_insn (ptr_reg, frame_reg_rtx);
if (REGNO (frame_reg_rtx) == ptr_regno)
frame_off = -end_save;
if (can_use_exit && info->cr_save_p)
restore_saved_cr (cr_save_reg, using_mtcr_multiple, true);
ptr_off = -end_save;
rs6000_emit_savres_rtx (info, ptr_reg,
info->gp_save_offset + ptr_off,
info->lr_save_offset + ptr_off,
reg_mode, sel);
}
else if (using_load_multiple)
{
rtvec p;
p = rtvec_alloc (32 - info->first_gp_reg_save);
for (i = 0; i < 32 - info->first_gp_reg_save; i++)
RTVEC_ELT (p, i)
= gen_frame_load (gen_rtx_REG (reg_mode, info->first_gp_reg_save + i),
frame_reg_rtx,
info->gp_save_offset + frame_off + reg_size * i);
emit_insn (gen_rtx_PARALLEL (VOIDmode, p));
}
else
{
int offset = info->gp_save_offset + frame_off;
for (i = info->first_gp_reg_save; i < 32; i++)
{
if (save_reg_p (i)
&& !cfun->machine->gpr_is_wrapped_separately[i])
{
rtx reg = gen_rtx_REG (reg_mode, i);
emit_insn (gen_frame_load (reg, frame_reg_rtx, offset));
}
offset += reg_size;
}
}
if (DEFAULT_ABI == ABI_V4 || flag_shrink_wrap)
{
if (frame_pointer_needed)
{
insn = get_last_insn ();
add_reg_note (insn, REG_CFA_DEF_CFA,
plus_constant (Pmode, frame_reg_rtx, frame_off));
RTX_FRAME_RELATED_P (insn) = 1;
}
if (flag_shrink_wrap
&& !restoring_GPRs_inline
&& info->first_fp_reg_save == 64)
cfa_restores = add_crlr_cfa_restore (info, cfa_restores);
for (i = info->first_gp_reg_save; i < 32; i++)
if (save_reg_p (i)
&& !cfun->machine->gpr_is_wrapped_separately[i])
{
rtx reg = gen_rtx_REG (reg_mode, i);
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
}
if (!restoring_GPRs_inline
&& info->first_fp_reg_save == 64)
{
if (cfa_restores)
emit_cfa_restores (cfa_restores);
return;
}
if (restore_lr && !restoring_GPRs_inline)
{
load_lr_save (0, frame_reg_rtx, info->lr_save_offset + frame_off);
restore_saved_lr (0, exit_func);
}
if (restoring_FPRs_inline)
{
int offset = info->fp_save_offset + frame_off;
for (i = info->first_fp_reg_save; i < 64; i++)
{
if (save_reg_p (i)
&& !cfun->machine->fpr_is_wrapped_separately[i - 32])
{
rtx reg = gen_rtx_REG (fp_reg_mode, i);
emit_insn (gen_frame_load (reg, frame_reg_rtx, offset));
if (DEFAULT_ABI == ABI_V4 || flag_shrink_wrap)
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg,
cfa_restores);
}
offset += fp_reg_size;
}
}
if (info->cr_save_p)
restore_saved_cr (cr_save_reg, using_mtcr_multiple, exit_func);
ptr_regno = 1;
if (!restoring_FPRs_inline)
{
bool lr = (strategy & REST_NOINLINE_FPRS_DOESNT_RESTORE_LR) == 0;
int sel = SAVRES_FPR | (lr ? SAVRES_LR : 0);
ptr_regno = ptr_regno_for_savres (sel);
}
insn = rs6000_emit_stack_reset (frame_reg_rtx, frame_off, ptr_regno);
if (REGNO (frame_reg_rtx) == ptr_regno)
frame_off = 0;
if (insn && restoring_FPRs_inline)
{
if (cfa_restores)
{
REG_NOTES (insn) = cfa_restores;
cfa_restores = NULL_RTX;
}
add_reg_note (insn, REG_CFA_DEF_CFA, sp_reg_rtx);
RTX_FRAME_RELATED_P (insn) = 1;
}
if (crtl->calls_eh_return)
{
rtx sa = EH_RETURN_STACKADJ_RTX;
emit_insn (gen_add3_insn (sp_reg_rtx, sp_reg_rtx, sa));
}
if (!sibcall && restoring_FPRs_inline)
{
if (cfa_restores)
{
emit_insn (gen_blockage ());
emit_cfa_restores (cfa_restores);
cfa_restores = NULL_RTX;
}
emit_jump_insn (targetm.gen_simple_return ());
}
if (!sibcall && !restoring_FPRs_inline)
{
bool lr = (strategy & REST_NOINLINE_FPRS_DOESNT_RESTORE_LR) == 0;
rtvec p = rtvec_alloc (3 + !!lr + 64 - info->first_fp_reg_save);
int elt = 0;
RTVEC_ELT (p, elt++) = ret_rtx;
if (lr)
RTVEC_ELT (p, elt++)
= gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, LR_REGNO));
int i;
int reg;
rtx sym;
if (flag_shrink_wrap)
cfa_restores = add_crlr_cfa_restore (info, cfa_restores);
sym = rs6000_savres_routine_sym (info, SAVRES_FPR | (lr ? SAVRES_LR : 0));
RTVEC_ELT (p, elt++) = gen_rtx_USE (VOIDmode, sym);
reg = (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)? 1 : 11;
RTVEC_ELT (p, elt++) = gen_rtx_USE (VOIDmode, gen_rtx_REG (Pmode, reg));
for (i = 0; i < 64 - info->first_fp_reg_save; i++)
{
rtx reg = gen_rtx_REG (DFmode, info->first_fp_reg_save + i);
RTVEC_ELT (p, elt++)
= gen_frame_load (reg, sp_reg_rtx, info->fp_save_offset + 8 * i);
if (flag_shrink_wrap
&& save_reg_p (info->first_fp_reg_save + i))
cfa_restores = alloc_reg_note (REG_CFA_RESTORE, reg, cfa_restores);
}
emit_jump_insn (gen_rtx_PARALLEL (VOIDmode, p));
}
if (cfa_restores)
{
if (sibcall)
emit_insn (gen_blockage ());
emit_cfa_restores (cfa_restores);
}
}
static void
rs6000_output_function_epilogue (FILE *file)
{
#if TARGET_MACHO
macho_branch_islands ();
{
rtx_insn *insn = get_last_insn ();
rtx_insn *deleted_debug_label = NULL;
while (insn
&& NOTE_P (insn)
&& NOTE_KIND (insn) != NOTE_INSN_DELETED_LABEL)
{
if (NOTE_P (insn) && NOTE_KIND (insn) == NOTE_INSN_DELETED_DEBUG_LABEL)
deleted_debug_label = insn;
insn = PREV_INSN (insn);
}
if (insn && BARRIER_P (insn))
insn = PREV_INSN (insn);
if (insn)
{
if (LABEL_P (insn)
|| (NOTE_P (insn)
&& NOTE_KIND (insn) == NOTE_INSN_DELETED_LABEL))
fputs ("\tnop\n", file);
else
{
while (insn && ! INSN_P (insn))
insn = PREV_INSN (insn);
if (insn == NULL)
fputs ("\ttrap\n", file);
}
}
else if (deleted_debug_label)
for (insn = deleted_debug_label; insn; insn = NEXT_INSN (insn))
if (NOTE_KIND (insn) == NOTE_INSN_DELETED_DEBUG_LABEL)
CODE_LABEL_NUMBER (insn) = -1;
}
#endif
if ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& ! flag_inhibit_size_directive
&& rs6000_traceback != traceback_none && !cfun->is_thunk)
{
const char *fname = NULL;
const char *language_string = lang_hooks.name;
int fixed_parms = 0, float_parms = 0, parm_info = 0;
int i;
int optional_tbtab;
rs6000_stack_t *info = rs6000_stack_info ();
if (rs6000_traceback == traceback_full)
optional_tbtab = 1;
else if (rs6000_traceback == traceback_part)
optional_tbtab = 0;
else
optional_tbtab = !optimize_size && !TARGET_ELF;
if (optional_tbtab)
{
fname = XSTR (XEXP (DECL_RTL (current_function_decl), 0), 0);
while (*fname == '.')	
fname++;
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LT");
ASM_OUTPUT_LABEL (file, fname);
}
fputs ("\t.long 0\n", file);
fputs ("\t.byte 0,", file);
if (lang_GNU_C ()
|| ! strcmp (language_string, "GNU GIMPLE")
|| ! strcmp (language_string, "GNU Go")
|| ! strcmp (language_string, "libgccjit"))
i = 0;
else if (! strcmp (language_string, "GNU F77")
|| lang_GNU_Fortran ())
i = 1;
else if (! strcmp (language_string, "GNU Pascal"))
i = 2;
else if (! strcmp (language_string, "GNU Ada"))
i = 3;
else if (lang_GNU_CXX ()
|| ! strcmp (language_string, "GNU Objective-C++"))
i = 9;
else if (! strcmp (language_string, "GNU Java"))
i = 13;
else if (! strcmp (language_string, "GNU Objective-C"))
i = 14;
else
gcc_unreachable ();
fprintf (file, "%d,", i);
fprintf (file, "%d,",
(optional_tbtab << 5) | ((info->first_fp_reg_save != 64) << 1));
fprintf (file, "%d,",
((optional_tbtab << 6)
| ((optional_tbtab & frame_pointer_needed) << 5)
| (info->cr_save_p << 1)
| (info->lr_save_p)));
fprintf (file, "%d,",
(info->push_p << 7) | (64 - info->first_fp_reg_save));
fprintf (file, "%d,", (32 - first_reg_to_save ()));
if (optional_tbtab)
{
tree decl;
int next_parm_info_bit = 31;
for (decl = DECL_ARGUMENTS (current_function_decl);
decl; decl = DECL_CHAIN (decl))
{
rtx parameter = DECL_INCOMING_RTL (decl);
machine_mode mode = GET_MODE (parameter);
if (GET_CODE (parameter) == REG)
{
if (SCALAR_FLOAT_MODE_P (mode))
{
int bits;
float_parms++;
switch (mode)
{
case E_SFmode:
case E_SDmode:
bits = 0x2;
break;
case E_DFmode:
case E_DDmode:
case E_TFmode:
case E_TDmode:
case E_IFmode:
case E_KFmode:
bits = 0x3;
break;
default:
gcc_unreachable ();
}
if (next_parm_info_bit > 0)
parm_info |= (bits << (next_parm_info_bit - 1));
next_parm_info_bit -= 2;
}
else
{
fixed_parms += ((GET_MODE_SIZE (mode)
+ (UNITS_PER_WORD - 1))
/ UNITS_PER_WORD);
next_parm_info_bit -= 1;
}
}
}
}
fprintf (file, "%d,", fixed_parms);
fprintf (file, "%d\n", ((float_parms << 1) | (! optimize)));
if (optional_tbtab)
{
if (fixed_parms || float_parms)
fprintf (file, "\t.long %d\n", parm_info);
fputs ("\t.long ", file);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LT");
RS6000_OUTPUT_BASENAME (file, fname);
putc ('-', file);
rs6000_output_function_entry (file, fname);
putc ('\n', file);
if (*fname == '*')
++fname;
fprintf (file, "\t.short %d\n", (int) strlen (fname));
assemble_string (fname, strlen (fname));
if (frame_pointer_needed)
fputs ("\t.byte 31\n", file);
fputs ("\t.align 2\n", file);
}
}
if (need_toc_init)
{
need_toc_init = 0;
if (!toc_initialized)
{
switch_to_section (toc_section);
switch_to_section (current_function_section ());
}
}
}
static GTY(()) rtx morestack_ref;
static rtx
gen_add3_const (rtx rt, rtx ra, long c)
{
if (TARGET_64BIT)
return gen_adddi3 (rt, ra, GEN_INT (c));
else
return gen_addsi3 (rt, ra, GEN_INT (c));
}
void
rs6000_expand_split_stack_prologue (void)
{
rs6000_stack_t *info = rs6000_stack_info ();
unsigned HOST_WIDE_INT allocate;
long alloc_hi, alloc_lo;
rtx r0, r1, r12, lr, ok_label, compare, jump, call_fusage;
rtx_insn *insn;
gcc_assert (flag_split_stack && reload_completed);
if (!info->push_p)
return;
if (global_regs[29])
{
error ("%qs uses register r29", "-fsplit-stack");
inform (DECL_SOURCE_LOCATION (global_regs_decl[29]),
"conflicts with %qD", global_regs_decl[29]);
}
allocate = info->total_size;
if (allocate > (unsigned HOST_WIDE_INT) 1 << 31)
{
sorry ("Stack frame larger than 2G is not supported for -fsplit-stack");
return;
}
if (morestack_ref == NULL_RTX)
{
morestack_ref = gen_rtx_SYMBOL_REF (Pmode, "__morestack");
SYMBOL_REF_FLAGS (morestack_ref) |= (SYMBOL_FLAG_LOCAL
| SYMBOL_FLAG_FUNCTION);
}
r0 = gen_rtx_REG (Pmode, 0);
r1 = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
r12 = gen_rtx_REG (Pmode, 12);
emit_insn (gen_load_split_stack_limit (r0));
alloc_hi = (-allocate + 0x8000) & ~0xffffL;
alloc_lo = -allocate - alloc_hi;
if (alloc_hi != 0)
{
emit_insn (gen_add3_const (r12, r1, alloc_hi));
if (alloc_lo != 0)
emit_insn (gen_add3_const (r12, r12, alloc_lo));
else
emit_insn (gen_nop ());
}
else
{
emit_insn (gen_add3_const (r12, r1, alloc_lo));
emit_insn (gen_nop ());
}
compare = gen_rtx_REG (CCUNSmode, CR7_REGNO);
emit_insn (gen_rtx_SET (compare, gen_rtx_COMPARE (CCUNSmode, r12, r0)));
ok_label = gen_label_rtx ();
jump = gen_rtx_IF_THEN_ELSE (VOIDmode,
gen_rtx_GEU (VOIDmode, compare, const0_rtx),
gen_rtx_LABEL_REF (VOIDmode, ok_label),
pc_rtx);
insn = emit_jump_insn (gen_rtx_SET (pc_rtx, jump));
JUMP_LABEL (insn) = ok_label;
add_reg_br_prob_note (insn, profile_probability::very_likely ());
lr = gen_rtx_REG (Pmode, LR_REGNO);
insn = emit_move_insn (r0, lr);
RTX_FRAME_RELATED_P (insn) = 1;
insn = emit_insn (gen_frame_store (r0, r1, info->lr_save_offset));
RTX_FRAME_RELATED_P (insn) = 1;
insn = emit_call_insn (gen_call (gen_rtx_MEM (SImode, morestack_ref),
const0_rtx, const0_rtx));
call_fusage = NULL_RTX;
use_reg (&call_fusage, r12);
use_reg (&call_fusage, r0);
add_function_usage_to (insn, call_fusage);
make_reg_eh_region_note_nothrow_nononlocal (insn);
emit_insn (gen_frame_load (r0, r1, info->lr_save_offset));
insn = emit_move_insn (lr, r0);
add_reg_note (insn, REG_CFA_RESTORE, lr);
RTX_FRAME_RELATED_P (insn) = 1;
emit_insn (gen_split_stack_return ());
emit_label (ok_label);
LABEL_NUSES (ok_label) = 1;
}
static rtx
rs6000_internal_arg_pointer (void)
{
if (flag_split_stack
&& (lookup_attribute ("no_split_stack", DECL_ATTRIBUTES (cfun->decl))
== NULL))
{
if (cfun->machine->split_stack_arg_pointer == NULL_RTX)
{
rtx pat;
cfun->machine->split_stack_arg_pointer = gen_reg_rtx (Pmode);
REG_POINTER (cfun->machine->split_stack_arg_pointer) = 1;
pat = gen_rtx_SET (cfun->machine->split_stack_arg_pointer,
gen_rtx_REG (Pmode, 12));
push_topmost_sequence ();
emit_insn_after (pat, get_insns ());
pop_topmost_sequence ();
}
rtx ret = plus_constant (Pmode, cfun->machine->split_stack_arg_pointer,
FIRST_PARM_OFFSET (current_function_decl));
return copy_to_reg (ret);
}
return virtual_incoming_args_rtx;
}
static void
rs6000_live_on_entry (bitmap regs)
{
if (flag_split_stack)
bitmap_set_bit (regs, 12);
}
void
rs6000_split_stack_space_check (rtx size, rtx label)
{
rtx sp = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
rtx limit = gen_reg_rtx (Pmode);
rtx requested = gen_reg_rtx (Pmode);
rtx cmp = gen_reg_rtx (CCUNSmode);
rtx jump;
emit_insn (gen_load_split_stack_limit (limit));
if (CONST_INT_P (size))
emit_insn (gen_add3_insn (requested, sp, GEN_INT (-INTVAL (size))));
else
{
size = force_reg (Pmode, size);
emit_move_insn (requested, gen_rtx_MINUS (Pmode, sp, size));
}
emit_insn (gen_rtx_SET (cmp, gen_rtx_COMPARE (CCUNSmode, requested, limit)));
jump = gen_rtx_IF_THEN_ELSE (VOIDmode,
gen_rtx_GEU (VOIDmode, cmp, const0_rtx),
gen_rtx_LABEL_REF (VOIDmode, label),
pc_rtx);
jump = emit_jump_insn (gen_rtx_SET (pc_rtx, jump));
JUMP_LABEL (jump) = label;
}

static void
rs6000_output_mi_thunk (FILE *file, tree thunk_fndecl ATTRIBUTE_UNUSED,
HOST_WIDE_INT delta, HOST_WIDE_INT vcall_offset,
tree function)
{
rtx this_rtx, funexp;
rtx_insn *insn;
reload_completed = 1;
epilogue_completed = 1;
emit_note (NOTE_INSN_PROLOGUE_END);
if (aggregate_value_p (TREE_TYPE (TREE_TYPE (function)), function))
this_rtx = gen_rtx_REG (Pmode, 4);
else
this_rtx = gen_rtx_REG (Pmode, 3);
if (delta)
emit_insn (gen_add3_insn (this_rtx, this_rtx, GEN_INT (delta)));
if (vcall_offset)
{
rtx vcall_offset_rtx = GEN_INT (vcall_offset);
rtx tmp = gen_rtx_REG (Pmode, 12);
emit_move_insn (tmp, gen_rtx_MEM (Pmode, this_rtx));
if (((unsigned HOST_WIDE_INT) vcall_offset) + 0x8000 >= 0x10000)
{
emit_insn (gen_add3_insn (tmp, tmp, vcall_offset_rtx));
emit_move_insn (tmp, gen_rtx_MEM (Pmode, tmp));
}
else
{
rtx loc = gen_rtx_PLUS (Pmode, tmp, vcall_offset_rtx);
emit_move_insn (tmp, gen_rtx_MEM (Pmode, loc));
}
emit_insn (gen_add3_insn (this_rtx, this_rtx, tmp));
}
if (!TREE_USED (function))
{
assemble_external (function);
TREE_USED (function) = 1;
}
funexp = XEXP (DECL_RTL (function), 0);
funexp = gen_rtx_MEM (FUNCTION_MODE, funexp);
#if TARGET_MACHO
if (MACHOPIC_INDIRECT)
funexp = machopic_indirect_call_target (funexp);
#endif
insn = emit_call_insn (
gen_rtx_PARALLEL (VOIDmode,
gen_rtvec (3,
gen_rtx_CALL (VOIDmode,
funexp, const0_rtx),
gen_rtx_USE (VOIDmode, const0_rtx),
simple_return_rtx)));
SIBLING_CALL_P (insn) = 1;
emit_barrier ();
insn = get_insns ();
shorten_branches (insn);
final_start_function (insn, file, 1);
final (insn, file, 1);
final_end_function ();
reload_completed = 0;
epilogue_completed = 0;
}

static unsigned
rs6000_hash_constant (rtx k)
{
enum rtx_code code = GET_CODE (k);
machine_mode mode = GET_MODE (k);
unsigned result = (code << 3) ^ mode;
const char *format;
int flen, fidx;
format = GET_RTX_FORMAT (code);
flen = strlen (format);
fidx = 0;
switch (code)
{
case LABEL_REF:
return result * 1231 + (unsigned) INSN_UID (XEXP (k, 0));
case CONST_WIDE_INT:
{
int i;
flen = CONST_WIDE_INT_NUNITS (k);
for (i = 0; i < flen; i++)
result = result * 613 + CONST_WIDE_INT_ELT (k, i);
return result;
}
case CONST_DOUBLE:
if (mode != VOIDmode)
return real_hash (CONST_DOUBLE_REAL_VALUE (k)) * result;
flen = 2;
break;
case CODE_LABEL:
fidx = 3;
break;
default:
break;
}
for (; fidx < flen; fidx++)
switch (format[fidx])
{
case 's':
{
unsigned i, len;
const char *str = XSTR (k, fidx);
len = strlen (str);
result = result * 613 + len;
for (i = 0; i < len; i++)
result = result * 613 + (unsigned) str[i];
break;
}
case 'u':
case 'e':
result = result * 1231 + rs6000_hash_constant (XEXP (k, fidx));
break;
case 'i':
case 'n':
result = result * 613 + (unsigned) XINT (k, fidx);
break;
case 'w':
if (sizeof (unsigned) >= sizeof (HOST_WIDE_INT))
result = result * 613 + (unsigned) XWINT (k, fidx);
else
{
size_t i;
for (i = 0; i < sizeof (HOST_WIDE_INT) / sizeof (unsigned); i++)
result = result * 613 + (unsigned) (XWINT (k, fidx)
>> CHAR_BIT * i);
}
break;
case '0':
break;
default:
gcc_unreachable ();
}
return result;
}
hashval_t
toc_hasher::hash (toc_hash_struct *thc)
{
return rs6000_hash_constant (thc->key) ^ thc->key_mode;
}
bool
toc_hasher::equal (toc_hash_struct *h1, toc_hash_struct *h2)
{
rtx r1 = h1->key;
rtx r2 = h2->key;
if (h1->key_mode != h2->key_mode)
return 0;
return rtx_equal_p (r1, r2);
}
#define VTABLE_NAME_P(NAME)				\
(strncmp ("_vt.", name, strlen ("_vt.")) == 0		\
|| strncmp ("_ZTV", name, strlen ("_ZTV")) == 0	\
|| strncmp ("_ZTT", name, strlen ("_ZTT")) == 0	\
|| strncmp ("_ZTI", name, strlen ("_ZTI")) == 0	\
|| strncmp ("_ZTC", name, strlen ("_ZTC")) == 0)
#ifdef NO_DOLLAR_IN_LABEL
const char *
rs6000_xcoff_strip_dollar (const char *name)
{
char *strip, *p;
const char *q;
size_t len;
q = (const char *) strchr (name, '$');
if (q == 0 || q == name)
return name;
len = strlen (name);
strip = XALLOCAVEC (char, len + 1);
strcpy (strip, name);
p = strip + (q - name);
while (p)
{
*p = '_';
p = strchr (p + 1, '$');
}
return ggc_alloc_string (strip, len);
}
#endif
void
rs6000_output_symbol_ref (FILE *file, rtx x)
{
const char *name = XSTR (x, 0);
if (VTABLE_NAME_P (name))
{
RS6000_OUTPUT_BASENAME (file, name);
}
else
assemble_name (file, name);
}
void
output_toc (FILE *file, rtx x, int labelno, machine_mode mode)
{
char buf[256];
const char *name = buf;
rtx base = x;
HOST_WIDE_INT offset = 0;
gcc_assert (!TARGET_NO_TOC);
if (TARGET_TOC && GET_CODE (x) != LABEL_REF)
{
struct toc_hash_struct *h;
if (toc_hash_table == NULL)
toc_hash_table = hash_table<toc_hasher>::create_ggc (1021);
h = ggc_alloc<toc_hash_struct> ();
h->key = x;
h->key_mode = mode;
h->labelno = labelno;
toc_hash_struct **found = toc_hash_table->find_slot (h, INSERT);
if (*found == NULL)
*found = h;
else  
{
fputs ("\t.set ", file);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LC");
fprintf (file, "%d,", labelno);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LC");
fprintf (file, "%d\n", ((*found)->labelno));
#ifdef HAVE_AS_TLS
if (TARGET_XCOFF && GET_CODE (x) == SYMBOL_REF
&& (SYMBOL_REF_TLS_MODEL (x) == TLS_MODEL_GLOBAL_DYNAMIC
|| SYMBOL_REF_TLS_MODEL (x) == TLS_MODEL_LOCAL_DYNAMIC))
{
fputs ("\t.set ", file);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LCM");
fprintf (file, "%d,", labelno);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (file, "LCM");
fprintf (file, "%d\n", ((*found)->labelno));
}
#endif
return;
}
}
if ((CONST_DOUBLE_P (x) || CONST_WIDE_INT_P (x))
&& STRICT_ALIGNMENT
&& GET_MODE_BITSIZE (mode) >= 64
&& ! (TARGET_NO_FP_IN_TOC && ! TARGET_MINIMAL_TOC)) {
ASM_OUTPUT_ALIGN (file, 3);
}
(*targetm.asm_out.internal_label) (file, "LC", labelno);
if (GET_CODE (x) == CONST_DOUBLE &&
(GET_MODE (x) == TFmode || GET_MODE (x) == TDmode
|| GET_MODE (x) == IFmode || GET_MODE (x) == KFmode))
{
long k[4];
if (DECIMAL_FLOAT_MODE_P (GET_MODE (x)))
REAL_VALUE_TO_TARGET_DECIMAL128 (*CONST_DOUBLE_REAL_VALUE (x), k);
else
REAL_VALUE_TO_TARGET_LONG_DOUBLE (*CONST_DOUBLE_REAL_VALUE (x), k);
if (TARGET_64BIT)
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs (DOUBLE_INT_ASM_OP, file);
else
fprintf (file, "\t.tc FT_%lx_%lx_%lx_%lx[TC],",
k[0] & 0xffffffff, k[1] & 0xffffffff,
k[2] & 0xffffffff, k[3] & 0xffffffff);
fprintf (file, "0x%lx%08lx,0x%lx%08lx\n",
k[WORDS_BIG_ENDIAN ? 0 : 1] & 0xffffffff,
k[WORDS_BIG_ENDIAN ? 1 : 0] & 0xffffffff,
k[WORDS_BIG_ENDIAN ? 2 : 3] & 0xffffffff,
k[WORDS_BIG_ENDIAN ? 3 : 2] & 0xffffffff);
return;
}
else
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs ("\t.long ", file);
else
fprintf (file, "\t.tc FT_%lx_%lx_%lx_%lx[TC],",
k[0] & 0xffffffff, k[1] & 0xffffffff,
k[2] & 0xffffffff, k[3] & 0xffffffff);
fprintf (file, "0x%lx,0x%lx,0x%lx,0x%lx\n",
k[0] & 0xffffffff, k[1] & 0xffffffff,
k[2] & 0xffffffff, k[3] & 0xffffffff);
return;
}
}
else if (GET_CODE (x) == CONST_DOUBLE &&
(GET_MODE (x) == DFmode || GET_MODE (x) == DDmode))
{
long k[2];
if (DECIMAL_FLOAT_MODE_P (GET_MODE (x)))
REAL_VALUE_TO_TARGET_DECIMAL64 (*CONST_DOUBLE_REAL_VALUE (x), k);
else
REAL_VALUE_TO_TARGET_DOUBLE (*CONST_DOUBLE_REAL_VALUE (x), k);
if (TARGET_64BIT)
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs (DOUBLE_INT_ASM_OP, file);
else
fprintf (file, "\t.tc FD_%lx_%lx[TC],",
k[0] & 0xffffffff, k[1] & 0xffffffff);
fprintf (file, "0x%lx%08lx\n",
k[WORDS_BIG_ENDIAN ? 0 : 1] & 0xffffffff,
k[WORDS_BIG_ENDIAN ? 1 : 0] & 0xffffffff);
return;
}
else
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs ("\t.long ", file);
else
fprintf (file, "\t.tc FD_%lx_%lx[TC],",
k[0] & 0xffffffff, k[1] & 0xffffffff);
fprintf (file, "0x%lx,0x%lx\n",
k[0] & 0xffffffff, k[1] & 0xffffffff);
return;
}
}
else if (GET_CODE (x) == CONST_DOUBLE &&
(GET_MODE (x) == SFmode || GET_MODE (x) == SDmode))
{
long l;
if (DECIMAL_FLOAT_MODE_P (GET_MODE (x)))
REAL_VALUE_TO_TARGET_DECIMAL32 (*CONST_DOUBLE_REAL_VALUE (x), l);
else
REAL_VALUE_TO_TARGET_SINGLE (*CONST_DOUBLE_REAL_VALUE (x), l);
if (TARGET_64BIT)
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs (DOUBLE_INT_ASM_OP, file);
else
fprintf (file, "\t.tc FS_%lx[TC],", l & 0xffffffff);
if (WORDS_BIG_ENDIAN)
fprintf (file, "0x%lx00000000\n", l & 0xffffffff);
else
fprintf (file, "0x%lx\n", l & 0xffffffff);
return;
}
else
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs ("\t.long ", file);
else
fprintf (file, "\t.tc FS_%lx[TC],", l & 0xffffffff);
fprintf (file, "0x%lx\n", l & 0xffffffff);
return;
}
}
else if (GET_MODE (x) == VOIDmode && GET_CODE (x) == CONST_INT)
{
unsigned HOST_WIDE_INT low;
HOST_WIDE_INT high;
low = INTVAL (x) & 0xffffffff;
high = (HOST_WIDE_INT) INTVAL (x) >> 32;
gcc_assert (!TARGET_64BIT || POINTER_SIZE >= GET_MODE_BITSIZE (mode));
if (WORDS_BIG_ENDIAN && POINTER_SIZE > GET_MODE_BITSIZE (mode))
{
low |= high << 32;
low <<= POINTER_SIZE - GET_MODE_BITSIZE (mode);
high = (HOST_WIDE_INT) low >> 32;
low &= 0xffffffff;
}
if (TARGET_64BIT)
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs (DOUBLE_INT_ASM_OP, file);
else
fprintf (file, "\t.tc ID_%lx_%lx[TC],",
(long) high & 0xffffffff, (long) low & 0xffffffff);
fprintf (file, "0x%lx%08lx\n",
(long) high & 0xffffffff, (long) low & 0xffffffff);
return;
}
else
{
if (POINTER_SIZE < GET_MODE_BITSIZE (mode))
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs ("\t.long ", file);
else
fprintf (file, "\t.tc ID_%lx_%lx[TC],",
(long) high & 0xffffffff, (long) low & 0xffffffff);
fprintf (file, "0x%lx,0x%lx\n",
(long) high & 0xffffffff, (long) low & 0xffffffff);
}
else
{
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs ("\t.long ", file);
else
fprintf (file, "\t.tc IS_%lx[TC],", (long) low & 0xffffffff);
fprintf (file, "0x%lx\n", (long) low & 0xffffffff);
}
return;
}
}
if (GET_CODE (x) == CONST)
{
gcc_assert (GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == CONST_INT);
base = XEXP (XEXP (x, 0), 0);
offset = INTVAL (XEXP (XEXP (x, 0), 1));
}
switch (GET_CODE (base))
{
case SYMBOL_REF:
name = XSTR (base, 0);
break;
case LABEL_REF:
ASM_GENERATE_INTERNAL_LABEL (buf, "L",
CODE_LABEL_NUMBER (XEXP (base, 0)));
break;
case CODE_LABEL:
ASM_GENERATE_INTERNAL_LABEL (buf, "L", CODE_LABEL_NUMBER (base));
break;
default:
gcc_unreachable ();
}
if (TARGET_ELF || TARGET_MINIMAL_TOC)
fputs (TARGET_32BIT ? "\t.long " : DOUBLE_INT_ASM_OP, file);
else
{
fputs ("\t.tc ", file);
RS6000_OUTPUT_BASENAME (file, name);
if (offset < 0)
fprintf (file, ".N" HOST_WIDE_INT_PRINT_UNSIGNED, - offset);
else if (offset)
fprintf (file, ".P" HOST_WIDE_INT_PRINT_UNSIGNED, offset);
fputs (TARGET_XCOFF && TARGET_CMODEL != CMODEL_SMALL
? "[TE]," : "[TC],", file);
}
if (VTABLE_NAME_P (name))
{
RS6000_OUTPUT_BASENAME (file, name);
if (offset < 0)
fprintf (file, HOST_WIDE_INT_PRINT_DEC, offset);
else if (offset > 0)
fprintf (file, "+" HOST_WIDE_INT_PRINT_DEC, offset);
}
else
output_addr_const (file, x);
#if HAVE_AS_TLS
if (TARGET_XCOFF && GET_CODE (base) == SYMBOL_REF)
{
switch (SYMBOL_REF_TLS_MODEL (base))
{
case 0:
break;
case TLS_MODEL_LOCAL_EXEC:
fputs ("@le", file);
break;
case TLS_MODEL_INITIAL_EXEC:
fputs ("@ie", file);
break;
case TLS_MODEL_GLOBAL_DYNAMIC:
case TLS_MODEL_LOCAL_DYNAMIC:
putc ('\n', file);
(*targetm.asm_out.internal_label) (file, "LCM", labelno);
fputs ("\t.tc .", file);
RS6000_OUTPUT_BASENAME (file, name);
fputs ("[TC],", file);
output_addr_const (file, x);
fputs ("@m", file);
break;
default:
gcc_unreachable ();
}
}
#endif
putc ('\n', file);
}

void
output_ascii (FILE *file, const char *p, int n)
{
char c;
int i, count_string;
const char *for_string = "\t.byte \"";
const char *for_decimal = "\t.byte ";
const char *to_close = NULL;
count_string = 0;
for (i = 0; i < n; i++)
{
c = *p++;
if (c >= ' ' && c < 0177)
{
if (for_string)
fputs (for_string, file);
putc (c, file);
if (c == '"')
{
putc (c, file);
++count_string;
}
for_string = NULL;
for_decimal = "\"\n\t.byte ";
to_close = "\"\n";
++count_string;
if (count_string >= 512)
{
fputs (to_close, file);
for_string = "\t.byte \"";
for_decimal = "\t.byte ";
to_close = NULL;
count_string = 0;
}
}
else
{
if (for_decimal)
fputs (for_decimal, file);
fprintf (file, "%d", c);
for_string = "\n\t.byte \"";
for_decimal = ", ";
to_close = "\n";
count_string = 0;
}
}
if (to_close)
fputs (to_close, file);
}

void
rs6000_gen_section_name (char **buf, const char *filename,
const char *section_desc)
{
const char *q, *after_last_slash, *last_period = 0;
char *p;
int len;
after_last_slash = filename;
for (q = filename; *q; q++)
{
if (*q == '/')
after_last_slash = q + 1;
else if (*q == '.')
last_period = q;
}
len = strlen (after_last_slash) + strlen (section_desc) + 2;
*buf = (char *) xmalloc (len);
p = *buf;
*p++ = '_';
for (q = after_last_slash; *q; q++)
{
if (q == last_period)
{
strcpy (p, section_desc);
p += strlen (section_desc);
break;
}
else if (ISALNUM (*q))
*p++ = *q;
}
if (last_period == 0)
strcpy (p, section_desc);
else
*p = '\0';
}

void
output_profile_hook (int labelno ATTRIBUTE_UNUSED)
{
if (TARGET_PROFILE_KERNEL)
return;
if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
{
#ifndef NO_PROFILE_COUNTERS
# define NO_PROFILE_COUNTERS 0
#endif
if (NO_PROFILE_COUNTERS)
emit_library_call (init_one_libfunc (RS6000_MCOUNT),
LCT_NORMAL, VOIDmode);
else
{
char buf[30];
const char *label_name;
rtx fun;
ASM_GENERATE_INTERNAL_LABEL (buf, "LP", labelno);
label_name = ggc_strdup ((*targetm.strip_name_encoding) (buf));
fun = gen_rtx_SYMBOL_REF (Pmode, label_name);
emit_library_call (init_one_libfunc (RS6000_MCOUNT),
LCT_NORMAL, VOIDmode, fun, Pmode);
}
}
else if (DEFAULT_ABI == ABI_DARWIN)
{
const char *mcount_name = RS6000_MCOUNT;
int caller_addr_regno = LR_REGNO;
crtl->uses_pic_offset_table = 1;
#if TARGET_MACHO
if (MACHOPIC_INDIRECT
&& crtl->uses_pic_offset_table)
caller_addr_regno = 0;
#endif
emit_library_call (gen_rtx_SYMBOL_REF (Pmode, mcount_name),
LCT_NORMAL, VOIDmode,
gen_rtx_REG (Pmode, caller_addr_regno), Pmode);
}
}
void
output_function_profiler (FILE *file, int labelno)
{
char buf[100];
switch (DEFAULT_ABI)
{
default:
gcc_unreachable ();
case ABI_V4:
if (!TARGET_32BIT)
{
warning (0, "no profiling of 64-bit code for this ABI");
return;
}
ASM_GENERATE_INTERNAL_LABEL (buf, "LP", labelno);
fprintf (file, "\tmflr %s\n", reg_names[0]);
if (NO_PROFILE_COUNTERS)
{
asm_fprintf (file, "\tstw %s,4(%s)\n",
reg_names[0], reg_names[1]);
}
else if (TARGET_SECURE_PLT && flag_pic)
{
if (TARGET_LINK_STACK)
{
char name[32];
get_ppc476_thunk_name (name);
asm_fprintf (file, "\tbl %s\n", name);
}
else
asm_fprintf (file, "\tbcl 20,31,1f\n1:\n");
asm_fprintf (file, "\tstw %s,4(%s)\n",
reg_names[0], reg_names[1]);
asm_fprintf (file, "\tmflr %s\n", reg_names[12]);
asm_fprintf (file, "\taddis %s,%s,",
reg_names[12], reg_names[12]);
assemble_name (file, buf);
asm_fprintf (file, "-1b@ha\n\tla %s,", reg_names[0]);
assemble_name (file, buf);
asm_fprintf (file, "-1b@l(%s)\n", reg_names[12]);
}
else if (flag_pic == 1)
{
fputs ("\tbl _GLOBAL_OFFSET_TABLE_@local-4\n", file);
asm_fprintf (file, "\tstw %s,4(%s)\n",
reg_names[0], reg_names[1]);
asm_fprintf (file, "\tmflr %s\n", reg_names[12]);
asm_fprintf (file, "\tlwz %s,", reg_names[0]);
assemble_name (file, buf);
asm_fprintf (file, "@got(%s)\n", reg_names[12]);
}
else if (flag_pic > 1)
{
asm_fprintf (file, "\tstw %s,4(%s)\n",
reg_names[0], reg_names[1]);
if (TARGET_LINK_STACK)
{
char name[32];
get_ppc476_thunk_name (name);
asm_fprintf (file, "\tbl %s\n\tb 1f\n\t.long ", name);
assemble_name (file, buf);
fputs ("-.\n1:", file);
asm_fprintf (file, "\tmflr %s\n", reg_names[11]);
asm_fprintf (file, "\taddi %s,%s,4\n",
reg_names[11], reg_names[11]);
}
else
{
fputs ("\tbcl 20,31,1f\n\t.long ", file);
assemble_name (file, buf);
fputs ("-.\n1:", file);
asm_fprintf (file, "\tmflr %s\n", reg_names[11]);
}
asm_fprintf (file, "\tlwz %s,0(%s)\n",
reg_names[0], reg_names[11]);
asm_fprintf (file, "\tadd %s,%s,%s\n",
reg_names[0], reg_names[0], reg_names[11]);
}
else
{
asm_fprintf (file, "\tlis %s,", reg_names[12]);
assemble_name (file, buf);
fputs ("@ha\n", file);
asm_fprintf (file, "\tstw %s,4(%s)\n",
reg_names[0], reg_names[1]);
asm_fprintf (file, "\tla %s,", reg_names[0]);
assemble_name (file, buf);
asm_fprintf (file, "@l(%s)\n", reg_names[12]);
}
fprintf (file, "\tbl %s%s\n",
RS6000_MCOUNT, flag_pic ? "@plt" : "");
break;
case ABI_AIX:
case ABI_ELFv2:
case ABI_DARWIN:
break;
}
}

static rtx_insn *last_scheduled_insn;
static int load_store_pendulum;
static int divide_cnt;
static int vec_pairing;
static int
rs6000_variable_issue_1 (rtx_insn *insn, int more)
{
last_scheduled_insn = insn;
if (GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
{
cached_can_issue_more = more;
return cached_can_issue_more;
}
if (insn_terminates_group_p (insn, current_group))
{
cached_can_issue_more = 0;
return cached_can_issue_more;
}
if (recog_memoized (insn) < 0)
return more;
if (rs6000_sched_groups)
{
if (is_microcoded_insn (insn))
cached_can_issue_more = 0;
else if (is_cracked_insn (insn))
cached_can_issue_more = more > 2 ? more - 2 : 0;
else
cached_can_issue_more = more - 1;
return cached_can_issue_more;
}
if (rs6000_tune == PROCESSOR_CELL && is_nonpipeline_insn (insn))
return 0;
cached_can_issue_more = more - 1;
return cached_can_issue_more;
}
static int
rs6000_variable_issue (FILE *stream, int verbose, rtx_insn *insn, int more)
{
int r = rs6000_variable_issue_1 (insn, more);
if (verbose)
fprintf (stream, "
return r;
}
static int
rs6000_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep_insn, int cost,
unsigned int)
{
enum attr_type attr_type;
if (recog_memoized (insn) < 0 || recog_memoized (dep_insn) < 0)
return cost;
switch (dep_type)
{
case REG_DEP_TRUE:
{
if ((rs6000_sched_groups || rs6000_tune == PROCESSOR_POWER9)
&& GET_CODE (PATTERN (insn)) == SET
&& GET_CODE (PATTERN (dep_insn)) == SET
&& GET_CODE (XEXP (PATTERN (insn), 1)) == MEM
&& GET_CODE (XEXP (PATTERN (dep_insn), 0)) == MEM
&& (GET_MODE_SIZE (GET_MODE (XEXP (PATTERN (insn), 1)))
> GET_MODE_SIZE (GET_MODE (XEXP (PATTERN (dep_insn), 0)))))
return cost + 14;
attr_type = get_attr_type (insn);
switch (attr_type)
{
case TYPE_JMPREG:
return 4;
case TYPE_BRANCH:
if ((rs6000_tune == PROCESSOR_PPC603
|| rs6000_tune == PROCESSOR_PPC604
|| rs6000_tune == PROCESSOR_PPC604e
|| rs6000_tune == PROCESSOR_PPC620
|| rs6000_tune == PROCESSOR_PPC630
|| rs6000_tune == PROCESSOR_PPC750
|| rs6000_tune == PROCESSOR_PPC7400
|| rs6000_tune == PROCESSOR_PPC7450
|| rs6000_tune == PROCESSOR_PPCE5500
|| rs6000_tune == PROCESSOR_PPCE6500
|| rs6000_tune == PROCESSOR_POWER4
|| rs6000_tune == PROCESSOR_POWER5
|| rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8
|| rs6000_tune == PROCESSOR_POWER9
|| rs6000_tune == PROCESSOR_CELL)
&& recog_memoized (dep_insn)
&& (INSN_CODE (dep_insn) >= 0))
switch (get_attr_type (dep_insn))
{
case TYPE_CMP:
case TYPE_FPCOMPARE:
case TYPE_CR_LOGICAL:
return cost + 2;
case TYPE_EXTS:
case TYPE_MUL:
if (get_attr_dot (dep_insn) == DOT_YES)
return cost + 2;
else
break;
case TYPE_SHIFT:
if (get_attr_dot (dep_insn) == DOT_YES
&& get_attr_var_shift (dep_insn) == VAR_SHIFT_NO)
return cost + 2;
else
break;
default:
break;
}
break;
case TYPE_STORE:
case TYPE_FPSTORE:
if ((rs6000_tune == PROCESSOR_POWER6)
&& recog_memoized (dep_insn)
&& (INSN_CODE (dep_insn) >= 0))
{
if (GET_CODE (PATTERN (insn)) != SET)
return cost;
switch (get_attr_type (dep_insn))
{
case TYPE_LOAD:
case TYPE_CNTLZ:
{
if (! rs6000_store_data_bypass_p (dep_insn, insn))
return get_attr_sign_extend (dep_insn)
== SIGN_EXTEND_YES ? 6 : 4;
break;
}
case TYPE_SHIFT:
{
if (! rs6000_store_data_bypass_p (dep_insn, insn))
return get_attr_var_shift (dep_insn) == VAR_SHIFT_YES ?
6 : 3;
break;
}
case TYPE_INTEGER:
case TYPE_ADD:
case TYPE_LOGICAL:
case TYPE_EXTS:
case TYPE_INSERT:
{
if (! rs6000_store_data_bypass_p (dep_insn, insn))
return 3;
break;
}
case TYPE_STORE:
case TYPE_FPLOAD:
case TYPE_FPSTORE:
{
if (get_attr_update (dep_insn) == UPDATE_YES
&& ! rs6000_store_data_bypass_p (dep_insn, insn))
return 3;
break;
}
case TYPE_MUL:
{
if (! rs6000_store_data_bypass_p (dep_insn, insn))
return 17;
break;
}
case TYPE_DIV:
{
if (! rs6000_store_data_bypass_p (dep_insn, insn))
return get_attr_size (dep_insn) == SIZE_32 ? 45 : 57;
break;
}
default:
break;
}
}
break;
case TYPE_LOAD:
if ((rs6000_tune == PROCESSOR_POWER6)
&& recog_memoized (dep_insn)
&& (INSN_CODE (dep_insn) >= 0))
{
switch (get_attr_type (dep_insn))
{
case TYPE_LOAD:
case TYPE_CNTLZ:
{
if (set_to_load_agen (dep_insn, insn))
return get_attr_sign_extend (dep_insn)
== SIGN_EXTEND_YES ? 6 : 4;
break;
}
case TYPE_SHIFT:
{
if (set_to_load_agen (dep_insn, insn))
return get_attr_var_shift (dep_insn) == VAR_SHIFT_YES ?
6 : 3;
break;
}
case TYPE_INTEGER:
case TYPE_ADD:
case TYPE_LOGICAL:
case TYPE_EXTS:
case TYPE_INSERT:
{
if (set_to_load_agen (dep_insn, insn))
return 3;
break;
}
case TYPE_STORE:
case TYPE_FPLOAD:
case TYPE_FPSTORE:
{
if (get_attr_update (dep_insn) == UPDATE_YES
&& set_to_load_agen (dep_insn, insn))
return 3;
break;
}
case TYPE_MUL:
{
if (set_to_load_agen (dep_insn, insn))
return 17;
break;
}
case TYPE_DIV:
{
if (set_to_load_agen (dep_insn, insn))
return get_attr_size (dep_insn) == SIZE_32 ? 45 : 57;
break;
}
default:
break;
}
}
break;
case TYPE_FPLOAD:
if ((rs6000_tune == PROCESSOR_POWER6)
&& get_attr_update (insn) == UPDATE_NO
&& recog_memoized (dep_insn)
&& (INSN_CODE (dep_insn) >= 0)
&& (get_attr_type (dep_insn) == TYPE_MFFGPR))
return 2;
default:
break;
}
}
break;
case REG_DEP_OUTPUT:
if ((rs6000_tune == PROCESSOR_POWER6)
&& recog_memoized (dep_insn)
&& (INSN_CODE (dep_insn) >= 0))
{
attr_type = get_attr_type (insn);
switch (attr_type)
{
case TYPE_FP:
case TYPE_FPSIMPLE:
if (get_attr_type (dep_insn) == TYPE_FP
|| get_attr_type (dep_insn) == TYPE_FPSIMPLE)
return 1;
break;
case TYPE_FPLOAD:
if (get_attr_update (insn) == UPDATE_NO
&& get_attr_type (dep_insn) == TYPE_MFFGPR)
return 2;
break;
default:
break;
}
}
case REG_DEP_ANTI:
return 0;
default:
gcc_unreachable ();
}
return cost;
}
static int
rs6000_debug_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep_insn,
int cost, unsigned int dw)
{
int ret = rs6000_adjust_cost (insn, dep_type, dep_insn, cost, dw);
if (ret != cost)
{
const char *dep;
switch (dep_type)
{
default:	     dep = "unknown depencency"; break;
case REG_DEP_TRUE:   dep = "data dependency";	 break;
case REG_DEP_OUTPUT: dep = "output dependency";  break;
case REG_DEP_ANTI:   dep = "anti depencency";	 break;
}
fprintf (stderr,
"\nrs6000_adjust_cost, final cost = %d, orig cost = %d, "
"%s, insn:\n", ret, cost, dep);
debug_rtx (insn);
}
return ret;
}
static bool
is_microcoded_insn (rtx_insn *insn)
{
if (!insn || !NONDEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
if (rs6000_tune == PROCESSOR_CELL)
return get_attr_cell_micro (insn) == CELL_MICRO_ALWAYS;
if (rs6000_sched_groups
&& (rs6000_tune == PROCESSOR_POWER4 || rs6000_tune == PROCESSOR_POWER5))
{
enum attr_type type = get_attr_type (insn);
if ((type == TYPE_LOAD
&& get_attr_update (insn) == UPDATE_YES
&& get_attr_sign_extend (insn) == SIGN_EXTEND_YES)
|| ((type == TYPE_LOAD || type == TYPE_STORE)
&& get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_YES)
|| type == TYPE_MFCR)
return true;
}
return false;
}
static bool
is_cracked_insn (rtx_insn *insn)
{
if (!insn || !NONDEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
if (rs6000_sched_groups
&& (rs6000_tune == PROCESSOR_POWER4 || rs6000_tune == PROCESSOR_POWER5))
{
enum attr_type type = get_attr_type (insn);
if ((type == TYPE_LOAD
&& get_attr_sign_extend (insn) == SIGN_EXTEND_YES
&& get_attr_update (insn) == UPDATE_NO)
|| (type == TYPE_LOAD
&& get_attr_sign_extend (insn) == SIGN_EXTEND_NO
&& get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_NO)
|| (type == TYPE_STORE
&& get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_NO)
|| ((type == TYPE_FPLOAD || type == TYPE_FPSTORE)
&& get_attr_update (insn) == UPDATE_YES)
|| (type == TYPE_CR_LOGICAL
&& get_attr_cr_logical_3op (insn) == CR_LOGICAL_3OP_YES)
|| (type == TYPE_EXTS
&& get_attr_dot (insn) == DOT_YES)
|| (type == TYPE_SHIFT
&& get_attr_dot (insn) == DOT_YES
&& get_attr_var_shift (insn) == VAR_SHIFT_NO)
|| (type == TYPE_MUL
&& get_attr_dot (insn) == DOT_YES)
|| type == TYPE_DIV
|| (type == TYPE_INSERT
&& get_attr_size (insn) == SIZE_32))
return true;
}
return false;
}
static bool
is_branch_slot_insn (rtx_insn *insn)
{
if (!insn || !NONDEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
if (rs6000_sched_groups)
{
enum attr_type type = get_attr_type (insn);
if (type == TYPE_BRANCH || type == TYPE_JMPREG)
return true;
return false;
}
return false;
}
static bool
set_to_load_agen (rtx_insn *out_insn, rtx_insn *in_insn)
{
rtx out_set, in_set;
out_set = single_set (out_insn);
if (out_set)
{
in_set = single_set (in_insn);
if (in_set)
return reg_mentioned_p (SET_DEST (out_set), SET_SRC (in_set));
}
return false;
}
static bool
get_memref_parts (rtx mem, rtx *base, HOST_WIDE_INT *offset,
HOST_WIDE_INT *size)
{
rtx addr_rtx;
if MEM_SIZE_KNOWN_P (mem)
*size = MEM_SIZE (mem);
else
return false;
addr_rtx = (XEXP (mem, 0));
if (GET_CODE (addr_rtx) == PRE_MODIFY)
addr_rtx = XEXP (addr_rtx, 1);
*offset = 0;
while (GET_CODE (addr_rtx) == PLUS
&& CONST_INT_P (XEXP (addr_rtx, 1)))
{
*offset += INTVAL (XEXP (addr_rtx, 1));
addr_rtx = XEXP (addr_rtx, 0);
}
if (!REG_P (addr_rtx))
return false;
*base = addr_rtx;
return true;
}
static bool
adjacent_mem_locations (rtx mem1, rtx mem2)
{
rtx reg1, reg2;
HOST_WIDE_INT off1, size1, off2, size2;
if (get_memref_parts (mem1, &reg1, &off1, &size1)
&& get_memref_parts (mem2, &reg2, &off2, &size2))
return ((REGNO (reg1) == REGNO (reg2))
&& ((off1 + size1 == off2)
|| (off2 + size2 == off1)));
return false;
}
static bool
mem_locations_overlap (rtx mem1, rtx mem2)
{
rtx reg1, reg2;
HOST_WIDE_INT off1, size1, off2, size2;
if (get_memref_parts (mem1, &reg1, &off1, &size1)
&& get_memref_parts (mem2, &reg2, &off2, &size2))
return ((REGNO (reg1) == REGNO (reg2))
&& (((off1 <= off2) && (off1 + size1 > off2))
|| ((off2 <= off1) && (off2 + size2 > off1))));
return false;
}
static int
rs6000_adjust_priority (rtx_insn *insn ATTRIBUTE_UNUSED, int priority)
{
rtx load_mem, str_mem;
#if 0
if (! INSN_P (insn))
return priority;
if (GET_CODE (PATTERN (insn)) == USE)
return priority;
switch (rs6000_tune) {
case PROCESSOR_PPC750:
switch (get_attr_type (insn))
{
default:
break;
case TYPE_MUL:
case TYPE_DIV:
fprintf (stderr, "priority was %#x (%d) before adjustment\n",
priority, priority);
if (priority >= 0 && priority < 0x01000000)
priority >>= 3;
break;
}
}
#endif
if (insn_must_be_first_in_group (insn)
&& reload_completed
&& current_sched_info->sched_max_insns_priority
&& rs6000_sched_restricted_insns_priority)
{
if (rs6000_sched_restricted_insns_priority == 1)
return current_sched_info->sched_max_insns_priority;
else if (rs6000_sched_restricted_insns_priority == 2)
return (priority + 1);
}
if (rs6000_tune == PROCESSOR_POWER6
&& ((load_store_pendulum == -2 && is_load_insn (insn, &load_mem))
|| (load_store_pendulum == 2 && is_store_insn (insn, &str_mem))))
return current_sched_info->sched_max_insns_priority;
return priority;
}
static bool
is_nonpipeline_insn (rtx_insn *insn)
{
enum attr_type type;
if (!insn || !NONDEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
type = get_attr_type (insn);
if (type == TYPE_MUL
|| type == TYPE_DIV
|| type == TYPE_SDIV
|| type == TYPE_DDIV
|| type == TYPE_SSQRT
|| type == TYPE_DSQRT
|| type == TYPE_MFCR
|| type == TYPE_MFCRF
|| type == TYPE_MFJMPR)
{
return true;
}
return false;
}
static int
rs6000_issue_rate (void)
{
if (!reload_completed && !flag_sched_pressure)
return 1;
switch (rs6000_tune) {
case PROCESSOR_RS64A:
case PROCESSOR_PPC601: 
case PROCESSOR_PPC7450:
return 3;
case PROCESSOR_PPC440:
case PROCESSOR_PPC603:
case PROCESSOR_PPC750:
case PROCESSOR_PPC7400:
case PROCESSOR_PPC8540:
case PROCESSOR_PPC8548:
case PROCESSOR_CELL:
case PROCESSOR_PPCE300C2:
case PROCESSOR_PPCE300C3:
case PROCESSOR_PPCE500MC:
case PROCESSOR_PPCE500MC64:
case PROCESSOR_PPCE5500:
case PROCESSOR_PPCE6500:
case PROCESSOR_TITAN:
return 2;
case PROCESSOR_PPC476:
case PROCESSOR_PPC604:
case PROCESSOR_PPC604e:
case PROCESSOR_PPC620:
case PROCESSOR_PPC630:
return 4;
case PROCESSOR_POWER4:
case PROCESSOR_POWER5:
case PROCESSOR_POWER6:
case PROCESSOR_POWER7:
return 5;
case PROCESSOR_POWER8:
return 7;
case PROCESSOR_POWER9:
return 6;
default:
return 1;
}
}
static int
rs6000_use_sched_lookahead (void)
{
switch (rs6000_tune)
{
case PROCESSOR_PPC8540:
case PROCESSOR_PPC8548:
return 4;
case PROCESSOR_CELL:
return (reload_completed ? 8 : 0);
default:
return 0;
}
}
static int
rs6000_use_sched_lookahead_guard (rtx_insn *insn, int ready_index)
{
if (ready_index == 0)
return 0;
if (rs6000_tune != PROCESSOR_CELL)
return 0;
gcc_assert (insn != NULL_RTX && INSN_P (insn));
if (!reload_completed
|| is_nonpipeline_insn (insn)
|| is_microcoded_insn (insn))
return 1;
return 0;
}
static bool
find_mem_ref (rtx pat, rtx *mem_ref)
{
const char * fmt;
int i, j;
if (tie_operand (pat, VOIDmode))
return false;
if (GET_CODE (pat) == MEM)
{
*mem_ref = pat;
return true;
}
fmt = GET_RTX_FORMAT (GET_CODE (pat));
for (i = GET_RTX_LENGTH (GET_CODE (pat)) - 1; i >= 0; i--)
{
if (fmt[i] == 'e')
{
if (find_mem_ref (XEXP (pat, i), mem_ref))
return true;
}
else if (fmt[i] == 'E')
for (j = XVECLEN (pat, i) - 1; j >= 0; j--)
{
if (find_mem_ref (XVECEXP (pat, i, j), mem_ref))
return true;
}
}
return false;
}
static bool
is_load_insn1 (rtx pat, rtx *load_mem)
{
if (!pat || pat == NULL_RTX)
return false;
if (GET_CODE (pat) == SET)
return find_mem_ref (SET_SRC (pat), load_mem);
if (GET_CODE (pat) == PARALLEL)
{
int i;
for (i = 0; i < XVECLEN (pat, 0); i++)
if (is_load_insn1 (XVECEXP (pat, 0, i), load_mem))
return true;
}
return false;
}
static bool
is_load_insn (rtx insn, rtx *load_mem)
{
if (!insn || !INSN_P (insn))
return false;
if (CALL_P (insn))
return false;
return is_load_insn1 (PATTERN (insn), load_mem);
}
static bool
is_store_insn1 (rtx pat, rtx *str_mem)
{
if (!pat || pat == NULL_RTX)
return false;
if (GET_CODE (pat) == SET)
return find_mem_ref (SET_DEST (pat), str_mem);
if (GET_CODE (pat) == PARALLEL)
{
int i;
for (i = 0; i < XVECLEN (pat, 0); i++)
if (is_store_insn1 (XVECEXP (pat, 0, i), str_mem))
return true;
}
return false;
}
static bool
is_store_insn (rtx insn, rtx *str_mem)
{
if (!insn || !INSN_P (insn))
return false;
return is_store_insn1 (PATTERN (insn), str_mem);
}
static bool
is_power9_pairable_vec_type (enum attr_type type)
{
switch (type)
{
case TYPE_VECSIMPLE:
case TYPE_VECCOMPLEX:
case TYPE_VECDIV:
case TYPE_VECCMP:
case TYPE_VECPERM:
case TYPE_VECFLOAT:
case TYPE_VECFDIV:
case TYPE_VECDOUBLE:
return true;
default:
break;
}
return false;
}
static bool
rs6000_is_costly_dependence (dep_t dep, int cost, int distance)
{
rtx insn;
rtx next;
rtx load_mem, str_mem;
if (rs6000_sched_costly_dep == no_dep_costly)
return false;
if (rs6000_sched_costly_dep == all_deps_costly)
return true;
insn = DEP_PRO (dep);
next = DEP_CON (dep);
if (rs6000_sched_costly_dep == store_to_load_dep_costly
&& is_load_insn (next, &load_mem)
&& is_store_insn (insn, &str_mem))
return true;
if (rs6000_sched_costly_dep == true_store_to_load_dep_costly
&& is_load_insn (next, &load_mem)
&& is_store_insn (insn, &str_mem)
&& DEP_TYPE (dep) == REG_DEP_TRUE
&& mem_locations_overlap(str_mem, load_mem))
return true;
if (rs6000_sched_costly_dep <= max_dep_latency
&& ((cost - distance) >= (int)rs6000_sched_costly_dep))
return true;
return false;
}
static rtx_insn *
get_next_active_insn (rtx_insn *insn, rtx_insn *tail)
{
if (insn == NULL_RTX || insn == tail)
return NULL;
while (1)
{
insn = NEXT_INSN (insn);
if (insn == NULL_RTX || insn == tail)
return NULL;
if (CALL_P (insn)
|| JUMP_P (insn) || JUMP_TABLE_DATA_P (insn)
|| (NONJUMP_INSN_P (insn)
&& GET_CODE (PATTERN (insn)) != USE
&& GET_CODE (PATTERN (insn)) != CLOBBER
&& INSN_CODE (insn) != CODE_FOR_stack_tie))
break;
}
return insn;
}
static int
power9_sched_reorder2 (rtx_insn **ready, int lastpos)
{
int pos;
int i;
rtx_insn *tmp;
enum attr_type type, type2;
type = get_attr_type (last_scheduled_insn);
if (type == TYPE_DIV && divide_cnt == 0)
{
divide_cnt = 1;
pos = lastpos;
while (pos >= 0)
{
if (recog_memoized (ready[pos]) >= 0
&& get_attr_type (ready[pos]) == TYPE_DIV)
{
tmp = ready[pos];
for (i = pos; i < lastpos; i++)
ready[i] = ready[i + 1];
ready[lastpos] = tmp;
break;
}
pos--;
}
}
else
{
divide_cnt = 0;
if (type == TYPE_VECLOAD)
{
if (vec_pairing == 0)
{
int vecload_pos = -1;
pos = lastpos;
while (pos >= 0)
{
if (recog_memoized (ready[pos]) >= 0)
{
type2 = get_attr_type (ready[pos]);
if (is_power9_pairable_vec_type (type2))
{
tmp = ready[pos];
for (i = pos; i < lastpos; i++)
ready[i] = ready[i + 1];
ready[lastpos] = tmp;
vec_pairing = 1;
return cached_can_issue_more;
}
else if (type2 == TYPE_VECLOAD && vecload_pos == -1)
vecload_pos = pos;
}
pos--;
}
if (vecload_pos >= 0)
{
tmp = ready[vecload_pos];
for (i = vecload_pos; i < lastpos; i++)
ready[i] = ready[i + 1];
ready[lastpos] = tmp;
vec_pairing = 1;
return cached_can_issue_more;
}
}
}
else if (is_power9_pairable_vec_type (type))
{
if (vec_pairing == 0)
{
int vec_pos = -1;
pos = lastpos;
while (pos >= 0)
{
if (recog_memoized (ready[pos]) >= 0)
{
type2 = get_attr_type (ready[pos]);
if (type2 == TYPE_VECLOAD)
{
tmp = ready[pos];
for (i = pos; i < lastpos; i++)
ready[i] = ready[i + 1];
ready[lastpos] = tmp;
vec_pairing = 1;
return cached_can_issue_more;
}
else if (is_power9_pairable_vec_type (type2)
&& vec_pos == -1)
vec_pos = pos;
}
pos--;
}
if (vec_pos >= 0)
{
tmp = ready[vec_pos];
for (i = vec_pos; i < lastpos; i++)
ready[i] = ready[i + 1];
ready[lastpos] = tmp;
vec_pairing = 1;
return cached_can_issue_more;
}
}
}
vec_pairing = 0;
}
return cached_can_issue_more;
}
static int
rs6000_sched_reorder (FILE *dump ATTRIBUTE_UNUSED, int sched_verbose,
rtx_insn **ready ATTRIBUTE_UNUSED,
int *pn_ready ATTRIBUTE_UNUSED,
int clock_var ATTRIBUTE_UNUSED)
{
int n_ready = *pn_ready;
if (sched_verbose)
fprintf (dump, "
if (rs6000_tune == PROCESSOR_CELL && n_ready > 1)
{
if (is_nonpipeline_insn (ready[n_ready - 1])
&& (recog_memoized (ready[n_ready - 2]) > 0))
std::swap (ready[n_ready - 1], ready[n_ready - 2]);
}
if (rs6000_tune == PROCESSOR_POWER6)
load_store_pendulum = 0;
return rs6000_issue_rate ();
}
static int
rs6000_sched_reorder2 (FILE *dump, int sched_verbose, rtx_insn **ready,
int *pn_ready, int clock_var ATTRIBUTE_UNUSED)
{
if (sched_verbose)
fprintf (dump, "
if (rs6000_tune == PROCESSOR_POWER6 && last_scheduled_insn)
{
int pos;
int i;
rtx_insn *tmp;
rtx load_mem, str_mem;
if (is_store_insn (last_scheduled_insn, &str_mem))
load_store_pendulum--;
else if (is_load_insn (last_scheduled_insn, &load_mem))
load_store_pendulum++;
else
return cached_can_issue_more;
if ((load_store_pendulum == 0) || (*pn_ready <= 1))
return cached_can_issue_more;
if (load_store_pendulum == 1)
{
pos = *pn_ready-1;
while (pos >= 0)
{
if (is_load_insn (ready[pos], &load_mem))
{
tmp = ready[pos];
for (i=pos; i<*pn_ready-1; i++)
ready[i] = ready[i + 1];
ready[*pn_ready-1] = tmp;
if (!sel_sched_p () && INSN_PRIORITY_KNOWN (tmp))
INSN_PRIORITY (tmp)++;
break;
}
pos--;
}
}
else if (load_store_pendulum == -2)
{
pos = *pn_ready-1;
while (pos >= 0)
{
if (is_load_insn (ready[pos], &load_mem)
&& !sel_sched_p ()
&& INSN_PRIORITY_KNOWN (ready[pos]))
{
INSN_PRIORITY (ready[pos])++;
load_store_pendulum--;
break;
}
pos--;
}
}
else if (load_store_pendulum == -1)
{
int first_store_pos = -1;
pos = *pn_ready-1;
while (pos >= 0)
{
if (is_store_insn (ready[pos], &str_mem))
{
rtx str_mem2;
if (first_store_pos == -1)
first_store_pos = pos;
if (is_store_insn (last_scheduled_insn, &str_mem2)
&& adjacent_mem_locations (str_mem, str_mem2))
{
tmp = ready[pos];
for (i=pos; i<*pn_ready-1; i++)
ready[i] = ready[i + 1];
ready[*pn_ready-1] = tmp;
if (!sel_sched_p () && INSN_PRIORITY_KNOWN (tmp))
INSN_PRIORITY (tmp)++;
first_store_pos = -1;
break;
};
}
pos--;
}
if (first_store_pos >= 0)
{
tmp = ready[first_store_pos];
for (i=first_store_pos; i<*pn_ready-1; i++)
ready[i] = ready[i + 1];
ready[*pn_ready-1] = tmp;
if (!sel_sched_p () && INSN_PRIORITY_KNOWN (tmp))
INSN_PRIORITY (tmp)++;
}
}
else if (load_store_pendulum == 2)
{
pos = *pn_ready-1;
while (pos >= 0)
{
if (is_store_insn (ready[pos], &str_mem)
&& !sel_sched_p ()
&& INSN_PRIORITY_KNOWN (ready[pos]))
{
INSN_PRIORITY (ready[pos])++;
load_store_pendulum++;
break;
}
pos--;
}
}
}
if (rs6000_tune == PROCESSOR_POWER9 && last_scheduled_insn
&& recog_memoized (last_scheduled_insn) >= 0)
return power9_sched_reorder2 (ready, *pn_ready - 1);
return cached_can_issue_more;
}
static bool
insn_terminates_group_p (rtx_insn *insn, enum group_termination which_group)
{
bool first, last;
if (! insn)
return false;
first = insn_must_be_first_in_group (insn);
last = insn_must_be_last_in_group (insn);
if (first && last)
return true;
if (which_group == current_group)
return last;
else if (which_group == previous_group)
return first;
return false;
}
static bool
insn_must_be_first_in_group (rtx_insn *insn)
{
enum attr_type type;
if (!insn
|| NOTE_P (insn)
|| DEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
switch (rs6000_tune)
{
case PROCESSOR_POWER5:
if (is_cracked_insn (insn))
return true;
case PROCESSOR_POWER4:
if (is_microcoded_insn (insn))
return true;
if (!rs6000_sched_groups)
return false;
type = get_attr_type (insn);
switch (type)
{
case TYPE_MFCR:
case TYPE_MFCRF:
case TYPE_MTCR:
case TYPE_CR_LOGICAL:
case TYPE_MTJMPR:
case TYPE_MFJMPR:
case TYPE_DIV:
case TYPE_LOAD_L:
case TYPE_STORE_C:
case TYPE_ISYNC:
case TYPE_SYNC:
return true;
default:
break;
}
break;
case PROCESSOR_POWER6:
type = get_attr_type (insn);
switch (type)
{
case TYPE_EXTS:
case TYPE_CNTLZ:
case TYPE_TRAP:
case TYPE_MUL:
case TYPE_INSERT:
case TYPE_FPCOMPARE:
case TYPE_MFCR:
case TYPE_MTCR:
case TYPE_MFJMPR:
case TYPE_MTJMPR:
case TYPE_ISYNC:
case TYPE_SYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
return true;
case TYPE_SHIFT:
if (get_attr_dot (insn) == DOT_NO
|| get_attr_var_shift (insn) == VAR_SHIFT_NO)
return true;
else
break;
case TYPE_DIV:
if (get_attr_size (insn) == SIZE_32)
return true;
else
break;
case TYPE_LOAD:
case TYPE_STORE:
case TYPE_FPLOAD:
case TYPE_FPSTORE:
if (get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
default:
break;
}
break;
case PROCESSOR_POWER7:
type = get_attr_type (insn);
switch (type)
{
case TYPE_CR_LOGICAL:
case TYPE_MFCR:
case TYPE_MFCRF:
case TYPE_MTCR:
case TYPE_DIV:
case TYPE_ISYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
case TYPE_MFJMPR:
case TYPE_MTJMPR:
return true;
case TYPE_MUL:
case TYPE_SHIFT:
case TYPE_EXTS:
if (get_attr_dot (insn) == DOT_YES)
return true;
else
break;
case TYPE_LOAD:
if (get_attr_sign_extend (insn) == SIGN_EXTEND_YES
|| get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
case TYPE_STORE:
case TYPE_FPLOAD:
case TYPE_FPSTORE:
if (get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
default:
break;
}
break;
case PROCESSOR_POWER8:
type = get_attr_type (insn);
switch (type)
{
case TYPE_CR_LOGICAL:
case TYPE_MFCR:
case TYPE_MFCRF:
case TYPE_MTCR:
case TYPE_SYNC:
case TYPE_ISYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
case TYPE_VECSTORE:
case TYPE_MFJMPR:
case TYPE_MTJMPR:
return true;
case TYPE_SHIFT:
case TYPE_EXTS:
case TYPE_MUL:
if (get_attr_dot (insn) == DOT_YES)
return true;
else
break;
case TYPE_LOAD:
if (get_attr_sign_extend (insn) == SIGN_EXTEND_YES
|| get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
case TYPE_STORE:
if (get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_YES)
return true;
else
break;
default:
break;
}
break;
default:
break;
}
return false;
}
static bool
insn_must_be_last_in_group (rtx_insn *insn)
{
enum attr_type type;
if (!insn
|| NOTE_P (insn)
|| DEBUG_INSN_P (insn)
|| GET_CODE (PATTERN (insn)) == USE
|| GET_CODE (PATTERN (insn)) == CLOBBER)
return false;
switch (rs6000_tune) {
case PROCESSOR_POWER4:
case PROCESSOR_POWER5:
if (is_microcoded_insn (insn))
return true;
if (is_branch_slot_insn (insn))
return true;
break;
case PROCESSOR_POWER6:
type = get_attr_type (insn);
switch (type)
{
case TYPE_EXTS:
case TYPE_CNTLZ:
case TYPE_TRAP:
case TYPE_MUL:
case TYPE_FPCOMPARE:
case TYPE_MFCR:
case TYPE_MTCR:
case TYPE_MFJMPR:
case TYPE_MTJMPR:
case TYPE_ISYNC:
case TYPE_SYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
return true;
case TYPE_SHIFT:
if (get_attr_dot (insn) == DOT_NO
|| get_attr_var_shift (insn) == VAR_SHIFT_NO)
return true;
else
break;
case TYPE_DIV:
if (get_attr_size (insn) == SIZE_32)
return true;
else
break;
default:
break;
}
break;
case PROCESSOR_POWER7:
type = get_attr_type (insn);
switch (type)
{
case TYPE_ISYNC:
case TYPE_SYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
return true;
case TYPE_LOAD:
if (get_attr_sign_extend (insn) == SIGN_EXTEND_YES
&& get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
case TYPE_STORE:
if (get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_YES)
return true;
else
break;
default:
break;
}
break;
case PROCESSOR_POWER8:
type = get_attr_type (insn);
switch (type)
{
case TYPE_MFCR:
case TYPE_MTCR:
case TYPE_ISYNC:
case TYPE_SYNC:
case TYPE_LOAD_L:
case TYPE_STORE_C:
return true;
case TYPE_LOAD:
if (get_attr_sign_extend (insn) == SIGN_EXTEND_YES
&& get_attr_update (insn) == UPDATE_YES)
return true;
else
break;
case TYPE_STORE:
if (get_attr_update (insn) == UPDATE_YES
&& get_attr_indexed (insn) == INDEXED_YES)
return true;
else
break;
default:
break;
}
break;
default:
break;
}
return false;
}
static bool
is_costly_group (rtx *group_insns, rtx next_insn)
{
int i;
int issue_rate = rs6000_issue_rate ();
for (i = 0; i < issue_rate; i++)
{
sd_iterator_def sd_it;
dep_t dep;
rtx insn = group_insns[i];
if (!insn)
continue;
FOR_EACH_DEP (insn, SD_LIST_RES_FORW, sd_it, dep)
{
rtx next = DEP_CON (dep);
if (next == next_insn
&& rs6000_is_costly_dependence (dep, dep_cost (dep), 0))
return true;
}
}
return false;
}
static int
force_new_group (int sched_verbose, FILE *dump, rtx *group_insns,
rtx_insn *next_insn, bool *group_end, int can_issue_more,
int *group_count)
{
rtx nop;
bool force;
int issue_rate = rs6000_issue_rate ();
bool end = *group_end;
int i;
if (next_insn == NULL_RTX || DEBUG_INSN_P (next_insn))
return can_issue_more;
if (rs6000_sched_insert_nops > sched_finish_regroup_exact)
return can_issue_more;
force = is_costly_group (group_insns, next_insn);
if (!force)
return can_issue_more;
if (sched_verbose > 6)
fprintf (dump,"force: group count = %d, can_issue_more = %d\n",
*group_count ,can_issue_more);
if (rs6000_sched_insert_nops == sched_finish_regroup_exact)
{
if (*group_end)
can_issue_more = 0;
if (can_issue_more && !is_branch_slot_insn (next_insn))
can_issue_more--;
if (rs6000_tune == PROCESSOR_POWER6 || rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8)
{
nop = gen_group_ending_nop ();
emit_insn_before (nop, next_insn);
can_issue_more = 0;
}
else
while (can_issue_more > 0)
{
nop = gen_nop ();
emit_insn_before (nop, next_insn);
can_issue_more--;
}
*group_end = true;
return 0;
}
if (rs6000_sched_insert_nops < sched_finish_regroup_exact)
{
int n_nops = rs6000_sched_insert_nops;
if (can_issue_more == 0)
can_issue_more = issue_rate;
can_issue_more--;
if (can_issue_more == 0)
{
can_issue_more = issue_rate - 1;
(*group_count)++;
end = true;
for (i = 0; i < issue_rate; i++)
{
group_insns[i] = 0;
}
}
while (n_nops > 0)
{
nop = gen_nop ();
emit_insn_before (nop, next_insn);
if (can_issue_more == issue_rate - 1) 
end = false;
can_issue_more--;
if (can_issue_more == 0)
{
can_issue_more = issue_rate - 1;
(*group_count)++;
end = true;
for (i = 0; i < issue_rate; i++)
{
group_insns[i] = 0;
}
}
n_nops--;
}
can_issue_more++;
*group_end
= (end
|| (can_issue_more == 1 && !is_branch_slot_insn (next_insn))
|| (can_issue_more <= 2 && is_cracked_insn (next_insn))
|| (can_issue_more < issue_rate &&
insn_terminates_group_p (next_insn, previous_group)));
if (*group_end && end)
(*group_count)--;
if (sched_verbose > 6)
fprintf (dump, "done force: group count = %d, can_issue_more = %d\n",
*group_count, can_issue_more);
return can_issue_more;
}
return can_issue_more;
}
static int
redefine_groups (FILE *dump, int sched_verbose, rtx_insn *prev_head_insn,
rtx_insn *tail)
{
rtx_insn *insn, *next_insn;
int issue_rate;
int can_issue_more;
int slot, i;
bool group_end;
int group_count = 0;
rtx *group_insns;
issue_rate = rs6000_issue_rate ();
group_insns = XALLOCAVEC (rtx, issue_rate);
for (i = 0; i < issue_rate; i++)
{
group_insns[i] = 0;
}
can_issue_more = issue_rate;
slot = 0;
insn = get_next_active_insn (prev_head_insn, tail);
group_end = false;
while (insn != NULL_RTX)
{
slot = (issue_rate - can_issue_more);
group_insns[slot] = insn;
can_issue_more =
rs6000_variable_issue (dump, sched_verbose, insn, can_issue_more);
if (insn_terminates_group_p (insn, current_group))
can_issue_more = 0;
next_insn = get_next_active_insn (insn, tail);
if (next_insn == NULL_RTX)
return group_count + 1;
group_end
= (can_issue_more == 0
|| (can_issue_more == 1 && !is_branch_slot_insn (next_insn))
|| (can_issue_more <= 2 && is_cracked_insn (next_insn))
|| (can_issue_more < issue_rate &&
insn_terminates_group_p (next_insn, previous_group)));
can_issue_more = force_new_group (sched_verbose, dump, group_insns,
next_insn, &group_end, can_issue_more,
&group_count);
if (group_end)
{
group_count++;
can_issue_more = 0;
for (i = 0; i < issue_rate; i++)
{
group_insns[i] = 0;
}
}
if (GET_MODE (next_insn) == TImode && can_issue_more)
PUT_MODE (next_insn, VOIDmode);
else if (!can_issue_more && GET_MODE (next_insn) != TImode)
PUT_MODE (next_insn, TImode);
insn = next_insn;
if (can_issue_more == 0)
can_issue_more = issue_rate;
} 
return group_count;
}
static int
pad_groups (FILE *dump, int sched_verbose, rtx_insn *prev_head_insn,
rtx_insn *tail)
{
rtx_insn *insn, *next_insn;
rtx nop;
int issue_rate;
int can_issue_more;
int group_end;
int group_count = 0;
issue_rate = rs6000_issue_rate ();
can_issue_more = issue_rate;
insn = get_next_active_insn (prev_head_insn, tail);
next_insn = get_next_active_insn (insn, tail);
while (insn != NULL_RTX)
{
can_issue_more =
rs6000_variable_issue (dump, sched_verbose, insn, can_issue_more);
group_end = (next_insn == NULL_RTX || GET_MODE (next_insn) == TImode);
if (next_insn == NULL_RTX)
break;
if (group_end)
{
if (can_issue_more
&& (rs6000_sched_insert_nops == sched_finish_pad_groups)
&& !insn_terminates_group_p (insn, current_group)
&& !insn_terminates_group_p (next_insn, previous_group))
{
if (!is_branch_slot_insn (next_insn))
can_issue_more--;
while (can_issue_more)
{
nop = gen_nop ();
emit_insn_before (nop, next_insn);
can_issue_more--;
}
}
can_issue_more = issue_rate;
group_count++;
}
insn = next_insn;
next_insn = get_next_active_insn (insn, tail);
}
return group_count;
}
static void
rs6000_sched_init (FILE *dump ATTRIBUTE_UNUSED,
int sched_verbose ATTRIBUTE_UNUSED,
int max_ready ATTRIBUTE_UNUSED)
{
last_scheduled_insn = NULL;
load_store_pendulum = 0;
divide_cnt = 0;
vec_pairing = 0;
}
static void
rs6000_sched_finish (FILE *dump, int sched_verbose)
{
int n_groups;
if (sched_verbose)
fprintf (dump, "=== Finishing schedule.\n");
if (reload_completed && rs6000_sched_groups)
{
if (sel_sched_p ())
return;
if (rs6000_sched_insert_nops == sched_finish_none)
return;
if (rs6000_sched_insert_nops == sched_finish_pad_groups)
n_groups = pad_groups (dump, sched_verbose,
current_sched_info->prev_head,
current_sched_info->next_tail);
else
n_groups = redefine_groups (dump, sched_verbose,
current_sched_info->prev_head,
current_sched_info->next_tail);
if (sched_verbose >= 6)
{
fprintf (dump, "ngroups = %d\n", n_groups);
print_rtl (dump, current_sched_info->prev_head);
fprintf (dump, "Done finish_sched\n");
}
}
}
struct rs6000_sched_context
{
short cached_can_issue_more;
rtx_insn *last_scheduled_insn;
int load_store_pendulum;
int divide_cnt;
int vec_pairing;
};
typedef struct rs6000_sched_context rs6000_sched_context_def;
typedef rs6000_sched_context_def *rs6000_sched_context_t;
static void *
rs6000_alloc_sched_context (void)
{
return xmalloc (sizeof (rs6000_sched_context_def));
}
static void
rs6000_init_sched_context (void *_sc, bool clean_p)
{
rs6000_sched_context_t sc = (rs6000_sched_context_t) _sc;
if (clean_p)
{
sc->cached_can_issue_more = 0;
sc->last_scheduled_insn = NULL;
sc->load_store_pendulum = 0;
sc->divide_cnt = 0;
sc->vec_pairing = 0;
}
else
{
sc->cached_can_issue_more = cached_can_issue_more;
sc->last_scheduled_insn = last_scheduled_insn;
sc->load_store_pendulum = load_store_pendulum;
sc->divide_cnt = divide_cnt;
sc->vec_pairing = vec_pairing;
}
}
static void
rs6000_set_sched_context (void *_sc)
{
rs6000_sched_context_t sc = (rs6000_sched_context_t) _sc;
gcc_assert (sc != NULL);
cached_can_issue_more = sc->cached_can_issue_more;
last_scheduled_insn = sc->last_scheduled_insn;
load_store_pendulum = sc->load_store_pendulum;
divide_cnt = sc->divide_cnt;
vec_pairing = sc->vec_pairing;
}
static void
rs6000_free_sched_context (void *_sc)
{
gcc_assert (_sc != NULL);
free (_sc);
}
static bool
rs6000_sched_can_speculate_insn (rtx_insn *insn)
{
switch (get_attr_type (insn))
{
case TYPE_DIV:
case TYPE_SDIV:
case TYPE_DDIV:
case TYPE_VECDIV:
case TYPE_SSQRT:
case TYPE_DSQRT:
return false;
default:
return true;
}
}

int
rs6000_trampoline_size (void)
{
int ret = 0;
switch (DEFAULT_ABI)
{
default:
gcc_unreachable ();
case ABI_AIX:
ret = (TARGET_32BIT) ? 12 : 24;
break;
case ABI_ELFv2:
gcc_assert (!TARGET_32BIT);
ret = 32;
break;
case ABI_DARWIN:
case ABI_V4:
ret = (TARGET_32BIT) ? 40 : 48;
break;
}
return ret;
}
static void
rs6000_trampoline_init (rtx m_tramp, tree fndecl, rtx cxt)
{
int regsize = (TARGET_32BIT) ? 4 : 8;
rtx fnaddr = XEXP (DECL_RTL (fndecl), 0);
rtx ctx_reg = force_reg (Pmode, cxt);
rtx addr = force_reg (Pmode, XEXP (m_tramp, 0));
switch (DEFAULT_ABI)
{
default:
gcc_unreachable ();
case ABI_AIX:
{
rtx fnmem, fn_reg, toc_reg;
if (!TARGET_POINTERS_TO_NESTED_FUNCTIONS)
error ("you cannot take the address of a nested function if you use "
"the %qs option", "-mno-pointers-to-nested-functions");
fnmem = gen_const_mem (Pmode, force_reg (Pmode, fnaddr));
fn_reg = gen_reg_rtx (Pmode);
toc_reg = gen_reg_rtx (Pmode);
# define MEM_PLUS(MEM, OFFSET) adjust_address (MEM, Pmode, OFFSET)
m_tramp = replace_equiv_address (m_tramp, addr);
emit_move_insn (fn_reg, MEM_PLUS (fnmem, 0));
emit_move_insn (toc_reg, MEM_PLUS (fnmem, regsize));
emit_move_insn (MEM_PLUS (m_tramp, 0), fn_reg);
emit_move_insn (MEM_PLUS (m_tramp, regsize), toc_reg);
emit_move_insn (MEM_PLUS (m_tramp, 2*regsize), ctx_reg);
# undef MEM_PLUS
}
break;
case ABI_ELFv2:
case ABI_DARWIN:
case ABI_V4:
emit_library_call (gen_rtx_SYMBOL_REF (Pmode, "__trampoline_setup"),
LCT_NORMAL, VOIDmode,
addr, Pmode,
GEN_INT (rs6000_trampoline_size ()), SImode,
fnaddr, Pmode,
ctx_reg, Pmode);
break;
}
}

static bool
rs6000_attribute_takes_identifier_p (const_tree attr_id)
{
return is_attribute_p ("altivec", attr_id);
}
static tree
rs6000_handle_altivec_attribute (tree *node,
tree name ATTRIBUTE_UNUSED,
tree args,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
tree type = *node, result = NULL_TREE;
machine_mode mode;
int unsigned_p;
char altivec_type
= ((args && TREE_CODE (args) == TREE_LIST && TREE_VALUE (args)
&& TREE_CODE (TREE_VALUE (args)) == IDENTIFIER_NODE)
? *IDENTIFIER_POINTER (TREE_VALUE (args))
: '?');
while (POINTER_TYPE_P (type)
|| TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE
|| TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
mode = TYPE_MODE (type);
if (type == long_double_type_node)
error ("use of %<long double%> in AltiVec types is invalid");
else if (type == boolean_type_node)
error ("use of boolean types in AltiVec types is invalid");
else if (TREE_CODE (type) == COMPLEX_TYPE)
error ("use of %<complex%> in AltiVec types is invalid");
else if (DECIMAL_FLOAT_MODE_P (mode))
error ("use of decimal floating point types in AltiVec types is invalid");
else if (!TARGET_VSX)
{
if (type == long_unsigned_type_node || type == long_integer_type_node)
{
if (TARGET_64BIT)
error ("use of %<long%> in AltiVec types is invalid for "
"64-bit code without %qs", "-mvsx");
else if (rs6000_warn_altivec_long)
warning (0, "use of %<long%> in AltiVec types is deprecated; "
"use %<int%>");
}
else if (type == long_long_unsigned_type_node
|| type == long_long_integer_type_node)
error ("use of %<long long%> in AltiVec types is invalid without %qs",
"-mvsx");
else if (type == double_type_node)
error ("use of %<double%> in AltiVec types is invalid without %qs",
"-mvsx");
}
switch (altivec_type)
{
case 'v':
unsigned_p = TYPE_UNSIGNED (type);
switch (mode)
{
case E_TImode:
result = (unsigned_p ? unsigned_V1TI_type_node : V1TI_type_node);
break;
case E_DImode:
result = (unsigned_p ? unsigned_V2DI_type_node : V2DI_type_node);
break;
case E_SImode:
result = (unsigned_p ? unsigned_V4SI_type_node : V4SI_type_node);
break;
case E_HImode:
result = (unsigned_p ? unsigned_V8HI_type_node : V8HI_type_node);
break;
case E_QImode:
result = (unsigned_p ? unsigned_V16QI_type_node : V16QI_type_node);
break;
case E_SFmode: result = V4SF_type_node; break;
case E_DFmode: result = V2DF_type_node; break;
case E_V4SImode: case E_V8HImode: case E_V16QImode: case E_V4SFmode:
case E_V2DImode: case E_V2DFmode:
result = type;
default: break;
}
break;
case 'b':
switch (mode)
{
case E_DImode: case E_V2DImode: result = bool_V2DI_type_node; break;
case E_SImode: case E_V4SImode: result = bool_V4SI_type_node; break;
case E_HImode: case E_V8HImode: result = bool_V8HI_type_node; break;
case E_QImode: case E_V16QImode: result = bool_V16QI_type_node;
default: break;
}
break;
case 'p':
switch (mode)
{
case E_V8HImode: result = pixel_V8HI_type_node;
default: break;
}
default: break;
}
if (result && result != type && TYPE_QUALS (type))
result = build_qualified_type (result, TYPE_QUALS (type));
*no_add_attrs = true;  
if (result)
*node = lang_hooks.types.reconstruct_complex_type (*node, result);
return NULL_TREE;
}
static const char *
rs6000_mangle_type (const_tree type)
{
type = TYPE_MAIN_VARIANT (type);
if (TREE_CODE (type) != VOID_TYPE && TREE_CODE (type) != BOOLEAN_TYPE
&& TREE_CODE (type) != INTEGER_TYPE && TREE_CODE (type) != REAL_TYPE)
return NULL;
if (type == bool_char_type_node) return "U6__boolc";
if (type == bool_short_type_node) return "U6__bools";
if (type == pixel_type_node) return "u7__pixel";
if (type == bool_int_type_node) return "U6__booli";
if (type == bool_long_long_type_node) return "U6__boolx";
if (SCALAR_FLOAT_TYPE_P (type) && FLOAT128_IBM_P (TYPE_MODE (type)))
return "g";
if (SCALAR_FLOAT_TYPE_P (type) && FLOAT128_IEEE_P (TYPE_MODE (type)))
return ieee128_mangling_gcc_8_1 ? "U10__float128" : "u9__ieee128";
return NULL;
}
static tree
rs6000_handle_longcall_attribute (tree *node, tree name,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
if (TREE_CODE (*node) != FUNCTION_TYPE
&& TREE_CODE (*node) != FIELD_DECL
&& TREE_CODE (*node) != TYPE_DECL)
{
warning (OPT_Wattributes, "%qE attribute only applies to functions",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static void
rs6000_set_default_type_attributes (tree type)
{
if (rs6000_default_long_calls
&& (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE))
TYPE_ATTRIBUTES (type) = tree_cons (get_identifier ("longcall"),
NULL_TREE,
TYPE_ATTRIBUTES (type));
#if TARGET_MACHO
darwin_set_default_type_attributes (type);
#endif
}
rtx
rs6000_longcall_ref (rtx call_ref)
{
const char *call_name;
tree node;
if (GET_CODE (call_ref) != SYMBOL_REF)
return call_ref;
call_name = XSTR (call_ref, 0);
if (*call_name == '.')
{
while (*call_name == '.')
call_name++;
node = get_identifier (call_name);
call_ref = gen_rtx_SYMBOL_REF (VOIDmode, IDENTIFIER_POINTER (node));
}
return force_reg (Pmode, call_ref);
}

#ifndef TARGET_USE_MS_BITFIELD_LAYOUT
#define TARGET_USE_MS_BITFIELD_LAYOUT 0
#endif
static tree
rs6000_handle_struct_attribute (tree *node, tree name,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED, bool *no_add_attrs)
{
tree *type = NULL;
if (DECL_P (*node))
{
if (TREE_CODE (*node) == TYPE_DECL)
type = &TREE_TYPE (*node);
}
else
type = node;
if (!(type && (TREE_CODE (*type) == RECORD_TYPE
|| TREE_CODE (*type) == UNION_TYPE)))
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
else if ((is_attribute_p ("ms_struct", name)
&& lookup_attribute ("gcc_struct", TYPE_ATTRIBUTES (*type)))
|| ((is_attribute_p ("gcc_struct", name)
&& lookup_attribute ("ms_struct", TYPE_ATTRIBUTES (*type)))))
{
warning (OPT_Wattributes, "%qE incompatible attribute ignored",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static bool
rs6000_ms_bitfield_layout_p (const_tree record_type)
{
return (TARGET_USE_MS_BITFIELD_LAYOUT &&
!lookup_attribute ("gcc_struct", TYPE_ATTRIBUTES (record_type)))
|| lookup_attribute ("ms_struct", TYPE_ATTRIBUTES (record_type));
}

#ifdef USING_ELFOS_H
static void
rs6000_elf_output_toc_section_asm_op (const void *data ATTRIBUTE_UNUSED)
{
if ((DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
&& TARGET_MINIMAL_TOC)
{
if (!toc_initialized)
{
fprintf (asm_out_file, "%s\n", TOC_SECTION_ASM_OP);
ASM_OUTPUT_ALIGN (asm_out_file, TARGET_64BIT ? 3 : 2);
(*targetm.asm_out.internal_label) (asm_out_file, "LCTOC", 0);
fprintf (asm_out_file, "\t.tc ");
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (asm_out_file, "LCTOC1[TC],");
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (asm_out_file, "LCTOC1");
fprintf (asm_out_file, "\n");
fprintf (asm_out_file, "%s\n", MINIMAL_TOC_SECTION_ASM_OP);
ASM_OUTPUT_ALIGN (asm_out_file, TARGET_64BIT ? 3 : 2);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (asm_out_file, "LCTOC1");
fprintf (asm_out_file, " = .+32768\n");
toc_initialized = 1;
}
else
fprintf (asm_out_file, "%s\n", MINIMAL_TOC_SECTION_ASM_OP);
}
else if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
{
fprintf (asm_out_file, "%s\n", TOC_SECTION_ASM_OP);
if (!toc_initialized)
{
ASM_OUTPUT_ALIGN (asm_out_file, TARGET_64BIT ? 3 : 2);
toc_initialized = 1;
}
}
else
{
fprintf (asm_out_file, "%s\n", MINIMAL_TOC_SECTION_ASM_OP);
if (!toc_initialized)
{
ASM_OUTPUT_ALIGN (asm_out_file, TARGET_64BIT ? 3 : 2);
ASM_OUTPUT_INTERNAL_LABEL_PREFIX (asm_out_file, "LCTOC1");
fprintf (asm_out_file, " = .+32768\n");
toc_initialized = 1;
}
}
}
static void
rs6000_elf_asm_init_sections (void)
{
toc_section
= get_unnamed_section (0, rs6000_elf_output_toc_section_asm_op, NULL);
sdata2_section
= get_unnamed_section (SECTION_WRITE, output_section_asm_op,
SDATA2_SECTION_ASM_OP);
}
static section *
rs6000_elf_select_rtx_section (machine_mode mode, rtx x,
unsigned HOST_WIDE_INT align)
{
if (ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (x, mode))
return toc_section;
else
return default_elf_select_rtx_section (mode, x, align);
}

static void rs6000_elf_encode_section_info (tree, rtx, int) ATTRIBUTE_UNUSED;
static void
rs6000_elf_encode_section_info (tree decl, rtx rtl, int first)
{
default_encode_section_info (decl, rtl, first);
if (first
&& TREE_CODE (decl) == FUNCTION_DECL
&& !TARGET_AIX
&& DEFAULT_ABI == ABI_AIX)
{
rtx sym_ref = XEXP (rtl, 0);
size_t len = strlen (XSTR (sym_ref, 0));
char *str = XALLOCAVEC (char, len + 2);
str[0] = '.';
memcpy (str + 1, XSTR (sym_ref, 0), len + 1);
XSTR (sym_ref, 0) = ggc_alloc_string (str, len + 1);
}
}
static inline bool
compare_section_name (const char *section, const char *templ)
{
int len;
len = strlen (templ);
return (strncmp (section, templ, len) == 0
&& (section[len] == 0 || section[len] == '.'));
}
bool
rs6000_elf_in_small_data_p (const_tree decl)
{
if (rs6000_sdata == SDATA_NONE)
return false;
if (TREE_CODE (decl) == STRING_CST)
return false;
if (TREE_CODE (decl) == FUNCTION_DECL)
return false;
if (TREE_CODE (decl) == VAR_DECL && DECL_SECTION_NAME (decl))
{
const char *section = DECL_SECTION_NAME (decl);
if (compare_section_name (section, ".sdata")
|| compare_section_name (section, ".sdata2")
|| compare_section_name (section, ".gnu.linkonce.s")
|| compare_section_name (section, ".sbss")
|| compare_section_name (section, ".sbss2")
|| compare_section_name (section, ".gnu.linkonce.sb")
|| strcmp (section, ".PPC.EMB.sdata0") == 0
|| strcmp (section, ".PPC.EMB.sbss0") == 0)
return true;
}
else
{
if (TREE_READONLY (decl) && rs6000_sdata != SDATA_EABI
&& !rs6000_readonly_in_sdata)
return false;
HOST_WIDE_INT size = int_size_in_bytes (TREE_TYPE (decl));
if (size > 0
&& size <= g_switch_value
&& (rs6000_sdata != SDATA_DATA || TREE_PUBLIC (decl)))
return true;
}
return false;
}
#endif 

static bool
rs6000_use_blocks_for_constant_p (machine_mode mode, const_rtx x)
{
return !ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (x, mode);
}
static bool
rs6000_use_blocks_for_decl_p (const_tree decl)
{
return !DECL_THREAD_LOCAL_P (decl);
}

rtx
find_addr_reg (rtx addr)
{
while (GET_CODE (addr) == PLUS)
{
if (GET_CODE (XEXP (addr, 0)) == REG
&& REGNO (XEXP (addr, 0)) != 0)
addr = XEXP (addr, 0);
else if (GET_CODE (XEXP (addr, 1)) == REG
&& REGNO (XEXP (addr, 1)) != 0)
addr = XEXP (addr, 1);
else if (CONSTANT_P (XEXP (addr, 0)))
addr = XEXP (addr, 1);
else if (CONSTANT_P (XEXP (addr, 1)))
addr = XEXP (addr, 0);
else
gcc_unreachable ();
}
gcc_assert (GET_CODE (addr) == REG && REGNO (addr) != 0);
return addr;
}
void
rs6000_fatal_bad_address (rtx op)
{
fatal_insn ("bad address", op);
}
#if TARGET_MACHO
typedef struct branch_island_d {
tree function_name;
tree label_name;
int line_number;
} branch_island;
static vec<branch_island, va_gc> *branch_islands;
static void
add_compiler_branch_island (tree label_name, tree function_name,
int line_number)
{
branch_island bi = {function_name, label_name, line_number};
vec_safe_push (branch_islands, bi);
}
static void
macho_branch_islands (void)
{
char tmp_buf[512];
while (!vec_safe_is_empty (branch_islands))
{
branch_island *bi = &branch_islands->last ();
const char *label = IDENTIFIER_POINTER (bi->label_name);
const char *name = IDENTIFIER_POINTER (bi->function_name);
char name_buf[512];
if (name[0] == '*' || name[0] == '&')
strcpy (name_buf, name+1);
else
{
name_buf[0] = '_';
strcpy (name_buf+1, name);
}
strcpy (tmp_buf, "\n");
strcat (tmp_buf, label);
#if defined (DBX_DEBUGGING_INFO) || defined (XCOFF_DEBUGGING_INFO)
if (write_symbols == DBX_DEBUG || write_symbols == XCOFF_DEBUG)
dbxout_stabd (N_SLINE, bi->line_number);
#endif 
if (flag_pic)
{
if (TARGET_LINK_STACK)
{
char name[32];
get_ppc476_thunk_name (name);
strcat (tmp_buf, ":\n\tmflr r0\n\tbl ");
strcat (tmp_buf, name);
strcat (tmp_buf, "\n");
strcat (tmp_buf, label);
strcat (tmp_buf, "_pic:\n\tmflr r11\n");
}
else
{
strcat (tmp_buf, ":\n\tmflr r0\n\tbcl 20,31,");
strcat (tmp_buf, label);
strcat (tmp_buf, "_pic\n");
strcat (tmp_buf, label);
strcat (tmp_buf, "_pic:\n\tmflr r11\n");
}
strcat (tmp_buf, "\taddis r11,r11,ha16(");
strcat (tmp_buf, name_buf);
strcat (tmp_buf, " - ");
strcat (tmp_buf, label);
strcat (tmp_buf, "_pic)\n");
strcat (tmp_buf, "\tmtlr r0\n");
strcat (tmp_buf, "\taddi r12,r11,lo16(");
strcat (tmp_buf, name_buf);
strcat (tmp_buf, " - ");
strcat (tmp_buf, label);
strcat (tmp_buf, "_pic)\n");
strcat (tmp_buf, "\tmtctr r12\n\tbctr\n");
}
else
{
strcat (tmp_buf, ":\nlis r12,hi16(");
strcat (tmp_buf, name_buf);
strcat (tmp_buf, ")\n\tori r12,r12,lo16(");
strcat (tmp_buf, name_buf);
strcat (tmp_buf, ")\n\tmtctr r12\n\tbctr");
}
output_asm_insn (tmp_buf, 0);
#if defined (DBX_DEBUGGING_INFO) || defined (XCOFF_DEBUGGING_INFO)
if (write_symbols == DBX_DEBUG || write_symbols == XCOFF_DEBUG)
dbxout_stabd (N_SLINE, bi->line_number);
#endif 
branch_islands->pop ();
}
}
static int
no_previous_def (tree function_name)
{
branch_island *bi;
unsigned ix;
FOR_EACH_VEC_SAFE_ELT (branch_islands, ix, bi)
if (function_name == bi->function_name)
return 0;
return 1;
}
static tree
get_prev_label (tree function_name)
{
branch_island *bi;
unsigned ix;
FOR_EACH_VEC_SAFE_ELT (branch_islands, ix, bi)
if (function_name == bi->function_name)
return bi->label_name;
return NULL_TREE;
}
char *
output_call (rtx_insn *insn, rtx *operands, int dest_operand_number,
int cookie_operand_number)
{
static char buf[256];
if (darwin_emit_branch_islands
&& GET_CODE (operands[dest_operand_number]) == SYMBOL_REF
&& (INTVAL (operands[cookie_operand_number]) & CALL_LONG))
{
tree labelname;
tree funname = get_identifier (XSTR (operands[dest_operand_number], 0));
if (no_previous_def (funname))
{
rtx label_rtx = gen_label_rtx ();
char *label_buf, temp_buf[256];
ASM_GENERATE_INTERNAL_LABEL (temp_buf, "L",
CODE_LABEL_NUMBER (label_rtx));
label_buf = temp_buf[0] == '*' ? temp_buf + 1 : temp_buf;
labelname = get_identifier (label_buf);
add_compiler_branch_island (labelname, funname, insn_line (insn));
}
else
labelname = get_prev_label (funname);
sprintf (buf, "jbsr %%z%d,%.246s",
dest_operand_number, IDENTIFIER_POINTER (labelname));
}
else
sprintf (buf, "bl %%z%d", dest_operand_number);
return buf;
}
void
machopic_output_stub (FILE *file, const char *symb, const char *stub)
{
unsigned int length;
char *symbol_name, *lazy_ptr_name;
char *local_label_0;
static int label = 0;
symb = (*targetm.strip_name_encoding) (symb);
length = strlen (symb);
symbol_name = XALLOCAVEC (char, length + 32);
GEN_SYMBOL_NAME_FOR_SYMBOL (symbol_name, symb, length);
lazy_ptr_name = XALLOCAVEC (char, length + 32);
GEN_LAZY_PTR_NAME_FOR_SYMBOL (lazy_ptr_name, symb, length);
if (flag_pic == 2)
switch_to_section (darwin_sections[machopic_picsymbol_stub1_section]);
else
switch_to_section (darwin_sections[machopic_symbol_stub1_section]);
if (flag_pic == 2)
{
fprintf (file, "\t.align 5\n");
fprintf (file, "%s:\n", stub);
fprintf (file, "\t.indirect_symbol %s\n", symbol_name);
label++;
local_label_0 = XALLOCAVEC (char, sizeof ("\"L00000000000$spb\""));
sprintf (local_label_0, "\"L%011d$spb\"", label);
fprintf (file, "\tmflr r0\n");
if (TARGET_LINK_STACK)
{
char name[32];
get_ppc476_thunk_name (name);
fprintf (file, "\tbl %s\n", name);
fprintf (file, "%s:\n\tmflr r11\n", local_label_0);
}
else
{
fprintf (file, "\tbcl 20,31,%s\n", local_label_0);
fprintf (file, "%s:\n\tmflr r11\n", local_label_0);
}
fprintf (file, "\taddis r11,r11,ha16(%s-%s)\n",
lazy_ptr_name, local_label_0);
fprintf (file, "\tmtlr r0\n");
fprintf (file, "\t%s r12,lo16(%s-%s)(r11)\n",
(TARGET_64BIT ? "ldu" : "lwzu"),
lazy_ptr_name, local_label_0);
fprintf (file, "\tmtctr r12\n");
fprintf (file, "\tbctr\n");
}
else
{
fprintf (file, "\t.align 4\n");
fprintf (file, "%s:\n", stub);
fprintf (file, "\t.indirect_symbol %s\n", symbol_name);
fprintf (file, "\tlis r11,ha16(%s)\n", lazy_ptr_name);
fprintf (file, "\t%s r12,lo16(%s)(r11)\n",
(TARGET_64BIT ? "ldu" : "lwzu"),
lazy_ptr_name);
fprintf (file, "\tmtctr r12\n");
fprintf (file, "\tbctr\n");
}
switch_to_section (darwin_sections[machopic_lazy_symbol_ptr_section]);
fprintf (file, "%s:\n", lazy_ptr_name);
fprintf (file, "\t.indirect_symbol %s\n", symbol_name);
fprintf (file, "%sdyld_stub_binding_helper\n",
(TARGET_64BIT ? DOUBLE_INT_ASM_OP : "\t.long\t"));
}
#define SMALL_INT(X) ((UINTVAL (X) + 0x8000) < 0x10000)
rtx
rs6000_machopic_legitimize_pic_address (rtx orig, machine_mode mode,
rtx reg)
{
rtx base, offset;
if (reg == NULL && !reload_completed)
reg = gen_reg_rtx (Pmode);
if (GET_CODE (orig) == CONST)
{
rtx reg_temp;
if (GET_CODE (XEXP (orig, 0)) == PLUS
&& XEXP (XEXP (orig, 0), 0) == pic_offset_table_rtx)
return orig;
gcc_assert (GET_CODE (XEXP (orig, 0)) == PLUS);
reg_temp = !can_create_pseudo_p () ? reg : gen_reg_rtx (Pmode);
base = rs6000_machopic_legitimize_pic_address (XEXP (XEXP (orig, 0), 0),
Pmode, reg_temp);
offset =
rs6000_machopic_legitimize_pic_address (XEXP (XEXP (orig, 0), 1),
Pmode, reg);
if (GET_CODE (offset) == CONST_INT)
{
if (SMALL_INT (offset))
return plus_constant (Pmode, base, INTVAL (offset));
else if (!reload_completed)
offset = force_reg (Pmode, offset);
else
{
rtx mem = force_const_mem (Pmode, orig);
return machopic_legitimize_pic_address (mem, Pmode, reg);
}
}
return gen_rtx_PLUS (Pmode, base, offset);
}
return machopic_legitimize_pic_address (orig, mode, reg);
}
static void
rs6000_darwin_file_start (void)
{
static const struct
{
const char *arg;
const char *name;
HOST_WIDE_INT if_set;
} mapping[] = {
{ "ppc64", "ppc64", MASK_64BIT },
{ "970", "ppc970", MASK_PPC_GPOPT | MASK_MFCRF | MASK_POWERPC64 },
{ "power4", "ppc970", 0 },
{ "G5", "ppc970", 0 },
{ "7450", "ppc7450", 0 },
{ "7400", "ppc7400", MASK_ALTIVEC },
{ "G4", "ppc7400", 0 },
{ "750", "ppc750", 0 },
{ "740", "ppc750", 0 },
{ "G3", "ppc750", 0 },
{ "604e", "ppc604e", 0 },
{ "604", "ppc604", 0 },
{ "603e", "ppc603", 0 },
{ "603", "ppc603", 0 },
{ "601", "ppc601", 0 },
{ NULL, "ppc", 0 } };
const char *cpu_id = "";
size_t i;
rs6000_file_start ();
darwin_file_start ();
if (rs6000_default_cpu != 0 && rs6000_default_cpu[0] != '\0')
cpu_id = rs6000_default_cpu;
if (global_options_set.x_rs6000_cpu_index)
cpu_id = processor_target_table[rs6000_cpu_index].name;
i = 0;
while (mapping[i].arg != NULL
&& strcmp (mapping[i].arg, cpu_id) != 0
&& (mapping[i].if_set & rs6000_isa_flags) == 0)
i++;
fprintf (asm_out_file, "\t.machine %s\n", mapping[i].name);
}
#endif 
#if TARGET_ELF
static int
rs6000_elf_reloc_rw_mask (void)
{
if (flag_pic)
return 3;
else if (DEFAULT_ABI == ABI_AIX || DEFAULT_ABI == ABI_ELFv2)
return 2;
else
return 0;
}
static void rs6000_elf_asm_out_constructor (rtx, int) ATTRIBUTE_UNUSED;
static void
rs6000_elf_asm_out_constructor (rtx symbol, int priority)
{
const char *section = ".ctors";
char buf[18];
if (priority != DEFAULT_INIT_PRIORITY)
{
sprintf (buf, ".ctors.%.5u",
MAX_INIT_PRIORITY - priority);
section = buf;
}
switch_to_section (get_section (section, SECTION_WRITE, NULL));
assemble_align (POINTER_SIZE);
if (DEFAULT_ABI == ABI_V4
&& (TARGET_RELOCATABLE || flag_pic > 1))
{
fputs ("\t.long (", asm_out_file);
output_addr_const (asm_out_file, symbol);
fputs (")@fixup\n", asm_out_file);
}
else
assemble_integer (symbol, POINTER_SIZE / BITS_PER_UNIT, POINTER_SIZE, 1);
}
static void rs6000_elf_asm_out_destructor (rtx, int) ATTRIBUTE_UNUSED;
static void
rs6000_elf_asm_out_destructor (rtx symbol, int priority)
{
const char *section = ".dtors";
char buf[18];
if (priority != DEFAULT_INIT_PRIORITY)
{
sprintf (buf, ".dtors.%.5u",
MAX_INIT_PRIORITY - priority);
section = buf;
}
switch_to_section (get_section (section, SECTION_WRITE, NULL));
assemble_align (POINTER_SIZE);
if (DEFAULT_ABI == ABI_V4
&& (TARGET_RELOCATABLE || flag_pic > 1))
{
fputs ("\t.long (", asm_out_file);
output_addr_const (asm_out_file, symbol);
fputs (")@fixup\n", asm_out_file);
}
else
assemble_integer (symbol, POINTER_SIZE / BITS_PER_UNIT, POINTER_SIZE, 1);
}
void
rs6000_elf_declare_function_name (FILE *file, const char *name, tree decl)
{
if (TARGET_64BIT && DEFAULT_ABI != ABI_ELFv2)
{
fputs ("\t.section\t\".opd\",\"aw\"\n\t.align 3\n", file);
ASM_OUTPUT_LABEL (file, name);
fputs (DOUBLE_INT_ASM_OP, file);
rs6000_output_function_entry (file, name);
fputs (",.TOC.@tocbase,0\n\t.previous\n", file);
if (DOT_SYMBOLS)
{
fputs ("\t.size\t", file);
assemble_name (file, name);
fputs (",24\n\t.type\t.", file);
assemble_name (file, name);
fputs (",@function\n", file);
if (TREE_PUBLIC (decl) && ! DECL_WEAK (decl))
{
fputs ("\t.globl\t.", file);
assemble_name (file, name);
putc ('\n', file);
}
}
else
ASM_OUTPUT_TYPE_DIRECTIVE (file, name, "function");
ASM_DECLARE_RESULT (file, DECL_RESULT (decl));
rs6000_output_function_entry (file, name);
fputs (":\n", file);
return;
}
int uses_toc;
if (DEFAULT_ABI == ABI_V4
&& (TARGET_RELOCATABLE || flag_pic > 1)
&& !TARGET_SECURE_PLT
&& (!constant_pool_empty_p () || crtl->profile)
&& (uses_toc = uses_TOC ()))
{
char buf[256];
if (uses_toc == 2)
switch_to_other_text_partition ();
(*targetm.asm_out.internal_label) (file, "LCL", rs6000_pic_labelno);
fprintf (file, "\t.long ");
assemble_name (file, toc_label_name);
need_toc_init = 1;
putc ('-', file);
ASM_GENERATE_INTERNAL_LABEL (buf, "LCF", rs6000_pic_labelno);
assemble_name (file, buf);
putc ('\n', file);
if (uses_toc == 2)
switch_to_other_text_partition ();
}
ASM_OUTPUT_TYPE_DIRECTIVE (file, name, "function");
ASM_DECLARE_RESULT (file, DECL_RESULT (decl));
if (TARGET_CMODEL == CMODEL_LARGE && rs6000_global_entry_point_needed_p ())
{
char buf[256];
(*targetm.asm_out.internal_label) (file, "LCL", rs6000_pic_labelno);
fprintf (file, "\t.quad .TOC.-");
ASM_GENERATE_INTERNAL_LABEL (buf, "LCF", rs6000_pic_labelno);
assemble_name (file, buf);
putc ('\n', file);
}
if (DEFAULT_ABI == ABI_AIX)
{
const char *desc_name, *orig_name;
orig_name = (*targetm.strip_name_encoding) (name);
desc_name = orig_name;
while (*desc_name == '.')
desc_name++;
if (TREE_PUBLIC (decl))
fprintf (file, "\t.globl %s\n", desc_name);
fprintf (file, "%s\n", MINIMAL_TOC_SECTION_ASM_OP);
fprintf (file, "%s:\n", desc_name);
fprintf (file, "\t.long %s\n", orig_name);
fputs ("\t.long _GLOBAL_OFFSET_TABLE_\n", file);
fputs ("\t.long 0\n", file);
fprintf (file, "\t.previous\n");
}
ASM_OUTPUT_LABEL (file, name);
}
static void rs6000_elf_file_end (void) ATTRIBUTE_UNUSED;
static void
rs6000_elf_file_end (void)
{
#ifdef HAVE_AS_GNU_ATTRIBUTE
if ((TARGET_64BIT || DEFAULT_ABI == ABI_V4)
&& rs6000_passes_float)
{
int fp;
if (TARGET_DF_FPR)
fp = 1;
else if (TARGET_SF_FPR)
fp = 3;
else
fp = 2;
if (rs6000_passes_long_double)
{
if (!TARGET_LONG_DOUBLE_128)
fp |= 2 * 4;
else if (TARGET_IEEEQUAD)
fp |= 3 * 4;
else
fp |= 1 * 4;
}
fprintf (asm_out_file, "\t.gnu_attribute 4, %d\n", fp);
}
if (TARGET_32BIT && DEFAULT_ABI == ABI_V4)
{
if (rs6000_passes_vector)
fprintf (asm_out_file, "\t.gnu_attribute 8, %d\n",
(TARGET_ALTIVEC_ABI ? 2 : 1));
if (rs6000_returns_struct)
fprintf (asm_out_file, "\t.gnu_attribute 12, %d\n",
aix_struct_return ? 2 : 1);
}
#endif
#if defined (POWERPC_LINUX) || defined (POWERPC_FREEBSD)
if (TARGET_32BIT || DEFAULT_ABI == ABI_ELFv2)
file_end_indicate_exec_stack ();
#endif
if (flag_split_stack)
file_end_indicate_split_stack ();
if (cpu_builtin_p)
{
switch_to_section (data_section);
fprintf (asm_out_file, "\t.align %u\n", TARGET_32BIT ? 2 : 3);
fprintf (asm_out_file, "\t%s %s\n",
TARGET_32BIT ? ".long" : ".quad", tcb_verification_symbol);
}
}
#endif
#if TARGET_XCOFF
#ifndef HAVE_XCOFF_DWARF_EXTRAS
#define HAVE_XCOFF_DWARF_EXTRAS 0
#endif
static enum unwind_info_type
rs6000_xcoff_debug_unwind_info (void)
{
return UI_NONE;
}
static void
rs6000_xcoff_asm_output_anchor (rtx symbol)
{
char buffer[100];
sprintf (buffer, "$ + " HOST_WIDE_INT_PRINT_DEC,
SYMBOL_REF_BLOCK_OFFSET (symbol));
fprintf (asm_out_file, "%s", SET_ASM_OP);
RS6000_OUTPUT_BASENAME (asm_out_file, XSTR (symbol, 0));
fprintf (asm_out_file, ",");
RS6000_OUTPUT_BASENAME (asm_out_file, buffer);
fprintf (asm_out_file, "\n");
}
static void
rs6000_xcoff_asm_globalize_label (FILE *stream, const char *name)
{
fputs (GLOBAL_ASM_OP, stream);
RS6000_OUTPUT_BASENAME (stream, name);
putc ('\n', stream);
}
static void
rs6000_xcoff_output_readonly_section_asm_op (const void *directive)
{
fprintf (asm_out_file, "\t.csect %s[RO],%s\n",
*(const char *const *) directive,
XCOFF_CSECT_DEFAULT_ALIGNMENT_STR);
}
static void
rs6000_xcoff_output_readwrite_section_asm_op (const void *directive)
{
fprintf (asm_out_file, "\t.csect %s[RW],%s\n",
*(const char *const *) directive,
XCOFF_CSECT_DEFAULT_ALIGNMENT_STR);
}
static void
rs6000_xcoff_output_tls_section_asm_op (const void *directive)
{
fprintf (asm_out_file, "\t.csect %s[TL],%s\n",
*(const char *const *) directive,
XCOFF_CSECT_DEFAULT_ALIGNMENT_STR);
}
static void
rs6000_xcoff_output_toc_section_asm_op (const void *data ATTRIBUTE_UNUSED)
{
if (TARGET_MINIMAL_TOC)
{
if (!toc_initialized)
{
fputs ("\t.toc\nLCTOC..1:\n", asm_out_file);
fputs ("\t.tc toc_table[TC],toc_table[RW]\n", asm_out_file);
toc_initialized = 1;
}
fprintf (asm_out_file, "\t.csect toc_table[RW]%s\n",
(TARGET_32BIT ? "" : ",3"));
}
else
fputs ("\t.toc\n", asm_out_file);
}
static void
rs6000_xcoff_asm_init_sections (void)
{
read_only_data_section
= get_unnamed_section (0, rs6000_xcoff_output_readonly_section_asm_op,
&xcoff_read_only_section_name);
private_data_section
= get_unnamed_section (SECTION_WRITE,
rs6000_xcoff_output_readwrite_section_asm_op,
&xcoff_private_data_section_name);
tls_data_section
= get_unnamed_section (SECTION_TLS,
rs6000_xcoff_output_tls_section_asm_op,
&xcoff_tls_data_section_name);
tls_private_data_section
= get_unnamed_section (SECTION_TLS,
rs6000_xcoff_output_tls_section_asm_op,
&xcoff_private_data_section_name);
read_only_private_data_section
= get_unnamed_section (0, rs6000_xcoff_output_readonly_section_asm_op,
&xcoff_private_data_section_name);
toc_section
= get_unnamed_section (0, rs6000_xcoff_output_toc_section_asm_op, NULL);
readonly_data_section = read_only_data_section;
}
static int
rs6000_xcoff_reloc_rw_mask (void)
{
return 3;
}
static void
rs6000_xcoff_asm_named_section (const char *name, unsigned int flags,
tree decl ATTRIBUTE_UNUSED)
{
int smclass;
static const char * const suffix[5] = { "PR", "RO", "RW", "TL", "XO" };
if (flags & SECTION_EXCLUDE)
smclass = 4;
else if (flags & SECTION_DEBUG)
{
fprintf (asm_out_file, "\t.dwsect %s\n", name);
return;
}
else if (flags & SECTION_CODE)
smclass = 0;
else if (flags & SECTION_TLS)
smclass = 3;
else if (flags & SECTION_WRITE)
smclass = 2;
else
smclass = 1;
fprintf (asm_out_file, "\t.csect %s%s[%s],%u\n",
(flags & SECTION_CODE) ? "." : "",
name, suffix[smclass], flags & SECTION_ENTSIZE);
}
#define IN_NAMED_SECTION(DECL) \
((TREE_CODE (DECL) == FUNCTION_DECL || TREE_CODE (DECL) == VAR_DECL) \
&& DECL_SECTION_NAME (DECL) != NULL)
static section *
rs6000_xcoff_select_section (tree decl, int reloc,
unsigned HOST_WIDE_INT align)
{
if (align > BIGGEST_ALIGNMENT)
{
resolve_unique_section (decl, reloc, true);
if (IN_NAMED_SECTION (decl))
return get_named_section (decl, NULL, reloc);
}
if (decl_readonly_section (decl, reloc))
{
if (TREE_PUBLIC (decl))
return read_only_data_section;
else
return read_only_private_data_section;
}
else
{
#if HAVE_AS_TLS
if (TREE_CODE (decl) == VAR_DECL && DECL_THREAD_LOCAL_P (decl))
{
if (TREE_PUBLIC (decl))
return tls_data_section;
else if (bss_initializer_p (decl))
{
DECL_COMMON (decl) = 1;
return tls_comm_section;
}
else
return tls_private_data_section;
}
else
#endif
if (TREE_PUBLIC (decl))
return data_section;
else
return private_data_section;
}
}
static void
rs6000_xcoff_unique_section (tree decl, int reloc ATTRIBUTE_UNUSED)
{
const char *name;
if (!TREE_PUBLIC (decl)
|| DECL_COMMON (decl)
|| (DECL_INITIAL (decl) == NULL_TREE
&& DECL_ALIGN (decl) <= BIGGEST_ALIGNMENT)
|| DECL_INITIAL (decl) == error_mark_node
|| (flag_zero_initialized_in_bss
&& initializer_zerop (DECL_INITIAL (decl))))
return;
name = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));
name = (*targetm.strip_name_encoding) (name);
set_decl_section_name (decl, name);
}
static section *
rs6000_xcoff_select_rtx_section (machine_mode mode, rtx x,
unsigned HOST_WIDE_INT align ATTRIBUTE_UNUSED)
{
if (ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (x, mode))
return toc_section;
else
return read_only_private_data_section;
}
static const char *
rs6000_xcoff_strip_name_encoding (const char *name)
{
size_t len;
if (*name == '*')
name++;
len = strlen (name);
if (name[len - 1] == ']')
return ggc_alloc_string (name, len - 4);
else
return name;
}
static unsigned int
rs6000_xcoff_section_type_flags (tree decl, const char *name, int reloc)
{
unsigned int align;
unsigned int flags = default_section_type_flags (decl, name, reloc);
if ((flags & SECTION_CODE) != 0 || !decl || !DECL_P (decl))
align = MIN_UNITS_PER_WORD;
else
align = MAX ((DECL_ALIGN (decl) / BITS_PER_UNIT),
int_size_in_bytes (TREE_TYPE (decl)) > MIN_UNITS_PER_WORD
? UNITS_PER_FP_WORD : MIN_UNITS_PER_WORD);
return flags | (exact_log2 (align) & SECTION_ENTSIZE);
}
static void
rs6000_xcoff_file_start (void)
{
rs6000_gen_section_name (&xcoff_bss_section_name,
main_input_filename, ".bss_");
rs6000_gen_section_name (&xcoff_private_data_section_name,
main_input_filename, ".rw_");
rs6000_gen_section_name (&xcoff_read_only_section_name,
main_input_filename, ".ro_");
rs6000_gen_section_name (&xcoff_tls_data_section_name,
main_input_filename, ".tls_");
rs6000_gen_section_name (&xcoff_tbss_section_name,
main_input_filename, ".tbss_[UL]");
fputs ("\t.file\t", asm_out_file);
output_quoted_string (asm_out_file, main_input_filename);
fputc ('\n', asm_out_file);
if (write_symbols != NO_DEBUG)
switch_to_section (private_data_section);
switch_to_section (toc_section);
switch_to_section (text_section);
if (profile_flag)
fprintf (asm_out_file, "\t.extern %s\n", RS6000_MCOUNT);
rs6000_file_start ();
}
static void
rs6000_xcoff_file_end (void)
{
switch_to_section (text_section);
fputs ("_section_.text:\n", asm_out_file);
switch_to_section (data_section);
fputs (TARGET_32BIT
? "\t.long _section_.text\n" : "\t.llong _section_.text\n",
asm_out_file);
}
struct declare_alias_data
{
FILE *file;
bool function_descriptor;
};
static bool
rs6000_declare_alias (struct symtab_node *n, void *d)
{
struct declare_alias_data *data = (struct declare_alias_data *)d;
if (!n->alias || n->weakref)
return false;
if (lookup_attribute ("ifunc", DECL_ATTRIBUTES (n->decl)))
return false;
TREE_ASM_WRITTEN (n->decl) = true;
const char *name = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (n->decl));
char *buffer = (char *) alloca (strlen (name) + 2);
char *p;
int dollar_inside = 0;
strcpy (buffer, name);
p = strchr (buffer, '$');
while (p) {
*p = '_';
dollar_inside++;
p = strchr (p + 1, '$');
}
if (TREE_PUBLIC (n->decl))
{
if (!RS6000_WEAK || !DECL_WEAK (n->decl))
{
if (dollar_inside) {
if (data->function_descriptor)
fprintf(data->file, "\t.rename .%s,\".%s\"\n", buffer, name);
fprintf(data->file, "\t.rename %s,\"%s\"\n", buffer, name);
}
if (data->function_descriptor)
{
fputs ("\t.globl .", data->file);
RS6000_OUTPUT_BASENAME (data->file, buffer);
putc ('\n', data->file);
}
fputs ("\t.globl ", data->file);
RS6000_OUTPUT_BASENAME (data->file, buffer);
putc ('\n', data->file);
}
#ifdef ASM_WEAKEN_DECL
else if (DECL_WEAK (n->decl) && !data->function_descriptor)
ASM_WEAKEN_DECL (data->file, n->decl, name, NULL);
#endif
}
else
{
if (dollar_inside)
{
if (data->function_descriptor)
fprintf(data->file, "\t.rename .%s,\".%s\"\n", buffer, name);
fprintf(data->file, "\t.rename %s,\"%s\"\n", buffer, name);
}
if (data->function_descriptor)
{
fputs ("\t.lglobl .", data->file);
RS6000_OUTPUT_BASENAME (data->file, buffer);
putc ('\n', data->file);
}
fputs ("\t.lglobl ", data->file);
RS6000_OUTPUT_BASENAME (data->file, buffer);
putc ('\n', data->file);
}
if (data->function_descriptor)
fputs (".", data->file);
RS6000_OUTPUT_BASENAME (data->file, buffer);
fputs (":\n", data->file);
return false;
}
#ifdef HAVE_GAS_HIDDEN
static const char *
rs6000_xcoff_visibility (tree decl)
{
static const char * const visibility_types[] = {
"", ",protected", ",hidden", ",internal"
};
enum symbol_visibility vis = DECL_VISIBILITY (decl);
if (TREE_CODE (decl) == FUNCTION_DECL
&& cgraph_node::get (decl)
&& cgraph_node::get (decl)->instrumentation_clone
&& cgraph_node::get (decl)->instrumented_version)
vis = DECL_VISIBILITY (cgraph_node::get (decl)->instrumented_version->decl);
return visibility_types[vis];
}
#endif
void
rs6000_xcoff_declare_function_name (FILE *file, const char *name, tree decl)
{
char *buffer = (char *) alloca (strlen (name) + 1);
char *p;
int dollar_inside = 0;
struct declare_alias_data data = {file, false};
strcpy (buffer, name);
p = strchr (buffer, '$');
while (p) {
*p = '_';
dollar_inside++;
p = strchr (p + 1, '$');
}
if (TREE_PUBLIC (decl))
{
if (!RS6000_WEAK || !DECL_WEAK (decl))
{
if (dollar_inside) {
fprintf(file, "\t.rename .%s,\".%s\"\n", buffer, name);
fprintf(file, "\t.rename %s,\"%s\"\n", buffer, name);
}
fputs ("\t.globl .", file);
RS6000_OUTPUT_BASENAME (file, buffer);
#ifdef HAVE_GAS_HIDDEN
fputs (rs6000_xcoff_visibility (decl), file);
#endif
putc ('\n', file);
}
}
else
{
if (dollar_inside) {
fprintf(file, "\t.rename .%s,\".%s\"\n", buffer, name);
fprintf(file, "\t.rename %s,\"%s\"\n", buffer, name);
}
fputs ("\t.lglobl .", file);
RS6000_OUTPUT_BASENAME (file, buffer);
putc ('\n', file);
}
fputs ("\t.csect ", file);
RS6000_OUTPUT_BASENAME (file, buffer);
fputs (TARGET_32BIT ? "[DS]\n" : "[DS],3\n", file);
RS6000_OUTPUT_BASENAME (file, buffer);
fputs (":\n", file);
symtab_node::get (decl)->call_for_symbol_and_aliases (rs6000_declare_alias,
&data, true);
fputs (TARGET_32BIT ? "\t.long ." : "\t.llong .", file);
RS6000_OUTPUT_BASENAME (file, buffer);
fputs (", TOC[tc0], 0\n", file);
in_section = NULL;
switch_to_section (function_section (decl));
putc ('.', file);
RS6000_OUTPUT_BASENAME (file, buffer);
fputs (":\n", file);
data.function_descriptor = true;
symtab_node::get (decl)->call_for_symbol_and_aliases (rs6000_declare_alias,
&data, true);
if (!DECL_IGNORED_P (decl))
{
if (write_symbols == DBX_DEBUG || write_symbols == XCOFF_DEBUG)
xcoffout_declare_function (file, decl, buffer);
else if (write_symbols == DWARF2_DEBUG)
{
name = (*targetm.strip_name_encoding) (name);
fprintf (file, "\t.function .%s,.%s,2,0\n", name, name);
}
}
return;
}
void
rs6000_xcoff_asm_globalize_decl_name (FILE *stream, tree decl)
{
const char *name = XSTR (XEXP (DECL_RTL (decl), 0), 0);
fputs (GLOBAL_ASM_OP, stream);
RS6000_OUTPUT_BASENAME (stream, name);
#ifdef HAVE_GAS_HIDDEN
fputs (rs6000_xcoff_visibility (decl), stream);
#endif
putc ('\n', stream);
}
void
rs6000_xcoff_asm_output_aligned_decl_common (FILE *stream,
tree decl ATTRIBUTE_UNUSED,
const char *name,
unsigned HOST_WIDE_INT size,
unsigned HOST_WIDE_INT align)
{
unsigned HOST_WIDE_INT align2 = 2;
if (align > 32)
align2 = floor_log2 (align / BITS_PER_UNIT);
else if (size > 4)
align2 = 3;
fputs (COMMON_ASM_OP, stream);
RS6000_OUTPUT_BASENAME (stream, name);
fprintf (stream,
"," HOST_WIDE_INT_PRINT_UNSIGNED "," HOST_WIDE_INT_PRINT_UNSIGNED,
size, align2);
#ifdef HAVE_GAS_HIDDEN
if (decl != NULL)
fputs (rs6000_xcoff_visibility (decl), stream);
#endif
putc ('\n', stream);
}
void
rs6000_xcoff_declare_object_name (FILE *file, const char *name, tree decl)
{
struct declare_alias_data data = {file, false};
RS6000_OUTPUT_BASENAME (file, name);
fputs (":\n", file);
symtab_node::get_create (decl)->call_for_symbol_and_aliases (rs6000_declare_alias,
&data, true);
}
void
rs6000_asm_output_dwarf_pcrel (FILE *file, int size, const char *label)
{
fputs (integer_asm_op (size, FALSE), file);
assemble_name (file, label);
fputs ("-$", file);
}
void
rs6000_asm_output_dwarf_datarel (FILE *file, int size, const char *label)
{
fputs (integer_asm_op (size, FALSE), file);
assemble_name (file, label);
fputs("-__gcc_unwind_dbase", file);
}
#ifdef HAVE_AS_TLS
static void
rs6000_xcoff_encode_section_info (tree decl, rtx rtl, int first)
{
rtx symbol;
int flags;
const char *symname;
default_encode_section_info (decl, rtl, first);
if (!MEM_P (rtl))
return;
symbol = XEXP (rtl, 0);
if (GET_CODE (symbol) != SYMBOL_REF)
return;
flags = SYMBOL_REF_FLAGS (symbol);
if (TREE_CODE (decl) == VAR_DECL && DECL_THREAD_LOCAL_P (decl))
flags &= ~SYMBOL_FLAG_HAS_BLOCK_INFO;
SYMBOL_REF_FLAGS (symbol) = flags;
symname = XSTR (symbol, 0);
if (decl 
&& DECL_P (decl) && DECL_EXTERNAL (decl) && TREE_PUBLIC (decl)
&& ((TREE_CODE (decl) == VAR_DECL && !DECL_THREAD_LOCAL_P (decl))
|| TREE_CODE (decl) == FUNCTION_DECL)
&& symname[strlen (symname) - 1] != ']')
{
char *newname = (char *) alloca (strlen (symname) + 5);
strcpy (newname, symname);
strcat (newname, (TREE_CODE (decl) == FUNCTION_DECL
? "[DS]" : "[UA]"));
XSTR (symbol, 0) = ggc_strdup (newname);
}
}
#endif 
#endif 
void
rs6000_asm_weaken_decl (FILE *stream, tree decl,
const char *name, const char *val)
{
fputs ("\t.weak\t", stream);
RS6000_OUTPUT_BASENAME (stream, name);
if (decl && TREE_CODE (decl) == FUNCTION_DECL
&& DEFAULT_ABI == ABI_AIX && DOT_SYMBOLS)
{
if (TARGET_XCOFF)						
fputs ("[DS]", stream);
#if TARGET_XCOFF && HAVE_GAS_HIDDEN
if (TARGET_XCOFF)
fputs (rs6000_xcoff_visibility (decl), stream);
#endif
fputs ("\n\t.weak\t.", stream);
RS6000_OUTPUT_BASENAME (stream, name);
}
#if TARGET_XCOFF && HAVE_GAS_HIDDEN
if (TARGET_XCOFF)
fputs (rs6000_xcoff_visibility (decl), stream);
#endif
fputc ('\n', stream);
if (val)
{
#ifdef ASM_OUTPUT_DEF
ASM_OUTPUT_DEF (stream, name, val);
#endif
if (decl && TREE_CODE (decl) == FUNCTION_DECL
&& DEFAULT_ABI == ABI_AIX && DOT_SYMBOLS)
{
fputs ("\t.set\t.", stream);
RS6000_OUTPUT_BASENAME (stream, name);
fputs (",.", stream);
RS6000_OUTPUT_BASENAME (stream, val);
fputc ('\n', stream);
}
}
}
static bool
rs6000_cannot_copy_insn_p (rtx_insn *insn)
{
return recog_memoized (insn) >= 0
&& get_attr_cannot_copy (insn);
}
static bool
rs6000_rtx_costs (rtx x, machine_mode mode, int outer_code,
int opno ATTRIBUTE_UNUSED, int *total, bool speed)
{
int code = GET_CODE (x);
switch (code)
{
case CONST_INT:
if (((outer_code == SET
|| outer_code == PLUS
|| outer_code == MINUS)
&& (satisfies_constraint_I (x)
|| satisfies_constraint_L (x)))
|| (outer_code == AND
&& (satisfies_constraint_K (x)
|| (mode == SImode
? satisfies_constraint_L (x)
: satisfies_constraint_J (x))))
|| ((outer_code == IOR || outer_code == XOR)
&& (satisfies_constraint_K (x)
|| (mode == SImode
? satisfies_constraint_L (x)
: satisfies_constraint_J (x))))
|| outer_code == ASHIFT
|| outer_code == ASHIFTRT
|| outer_code == LSHIFTRT
|| outer_code == ROTATE
|| outer_code == ROTATERT
|| outer_code == ZERO_EXTRACT
|| (outer_code == MULT
&& satisfies_constraint_I (x))
|| ((outer_code == DIV || outer_code == UDIV
|| outer_code == MOD || outer_code == UMOD)
&& exact_log2 (INTVAL (x)) >= 0)
|| (outer_code == COMPARE
&& (satisfies_constraint_I (x)
|| satisfies_constraint_K (x)))
|| ((outer_code == EQ || outer_code == NE)
&& (satisfies_constraint_I (x)
|| satisfies_constraint_K (x)
|| (mode == SImode
? satisfies_constraint_L (x)
: satisfies_constraint_J (x))))
|| (outer_code == GTU
&& satisfies_constraint_I (x))
|| (outer_code == LTU
&& satisfies_constraint_P (x)))
{
*total = 0;
return true;
}
else if ((outer_code == PLUS
&& reg_or_add_cint_operand (x, VOIDmode))
|| (outer_code == MINUS
&& reg_or_sub_cint_operand (x, VOIDmode))
|| ((outer_code == SET
|| outer_code == IOR
|| outer_code == XOR)
&& (INTVAL (x)
& ~ (unsigned HOST_WIDE_INT) 0xffffffff) == 0))
{
*total = COSTS_N_INSNS (1);
return true;
}
case CONST_DOUBLE:
case CONST_WIDE_INT:
case CONST:
case HIGH:
case SYMBOL_REF:
*total = !speed ? COSTS_N_INSNS (1) + 1 : COSTS_N_INSNS (2);
return true;
case MEM:
*total = !speed ? COSTS_N_INSNS (1) + 1 : COSTS_N_INSNS (2);
if (rs6000_slow_unaligned_access (mode, MEM_ALIGN (x)))
*total += COSTS_N_INSNS (100);
return true;
case LABEL_REF:
*total = 0;
return true;
case PLUS:
case MINUS:
if (FLOAT_MODE_P (mode))
*total = rs6000_cost->fp;
else
*total = COSTS_N_INSNS (1);
return false;
case MULT:
if (GET_CODE (XEXP (x, 1)) == CONST_INT
&& satisfies_constraint_I (XEXP (x, 1)))
{
if (INTVAL (XEXP (x, 1)) >= -256
&& INTVAL (XEXP (x, 1)) <= 255)
*total = rs6000_cost->mulsi_const9;
else
*total = rs6000_cost->mulsi_const;
}
else if (mode == SFmode)
*total = rs6000_cost->fp;
else if (FLOAT_MODE_P (mode))
*total = rs6000_cost->dmul;
else if (mode == DImode)
*total = rs6000_cost->muldi;
else
*total = rs6000_cost->mulsi;
return false;
case FMA:
if (mode == SFmode)
*total = rs6000_cost->fp;
else
*total = rs6000_cost->dmul;
break;
case DIV:
case MOD:
if (FLOAT_MODE_P (mode))
{
*total = mode == DFmode ? rs6000_cost->ddiv
: rs6000_cost->sdiv;
return false;
}
case UDIV:
case UMOD:
if (GET_CODE (XEXP (x, 1)) == CONST_INT
&& exact_log2 (INTVAL (XEXP (x, 1))) >= 0)
{
if (code == DIV || code == MOD)
*total = COSTS_N_INSNS (2);
else
*total = COSTS_N_INSNS (1);
}
else
{
if (GET_MODE (XEXP (x, 1)) == DImode)
*total = rs6000_cost->divdi;
else
*total = rs6000_cost->divsi;
}
if (!TARGET_MODULO && (code == MOD || code == UMOD))
*total += COSTS_N_INSNS (2);
return false;
case CTZ:
*total = COSTS_N_INSNS (TARGET_CTZ ? 1 : 4);
return false;
case FFS:
*total = COSTS_N_INSNS (4);
return false;
case POPCOUNT:
*total = COSTS_N_INSNS (TARGET_POPCNTD ? 1 : 6);
return false;
case PARITY:
*total = COSTS_N_INSNS (TARGET_CMPB ? 2 : 6);
return false;
case NOT:
if (outer_code == AND || outer_code == IOR || outer_code == XOR)
*total = 0;
else
*total = COSTS_N_INSNS (1);
return false;
case AND:
if (CONST_INT_P (XEXP (x, 1)))
{
rtx left = XEXP (x, 0);
rtx_code left_code = GET_CODE (left);
if ((left_code == ROTATE
|| left_code == ASHIFT
|| left_code == LSHIFTRT)
&& rs6000_is_valid_shift_mask (XEXP (x, 1), left, mode))
{
*total = rtx_cost (XEXP (left, 0), mode, left_code, 0, speed);
if (!CONST_INT_P (XEXP (left, 1)))
*total += rtx_cost (XEXP (left, 1), SImode, left_code, 1, speed);
*total += COSTS_N_INSNS (1);
return true;
}
HOST_WIDE_INT val = INTVAL (XEXP (x, 1));
if (rs6000_is_valid_and_mask (XEXP (x, 1), mode)
|| (val & 0xffff) == val
|| (val & 0xffff0000) == val
|| ((val & 0xffff) == 0 && mode == SImode))
{
*total = rtx_cost (left, mode, AND, 0, speed);
*total += COSTS_N_INSNS (1);
return true;
}
if (rs6000_is_valid_2insn_and (XEXP (x, 1), mode))
{
*total = rtx_cost (left, mode, AND, 0, speed);
*total += COSTS_N_INSNS (2);
return true;
}
}
*total = COSTS_N_INSNS (1);
return false;
case IOR:
*total = COSTS_N_INSNS (1);
return true;
case CLZ:
case XOR:
case ZERO_EXTRACT:
*total = COSTS_N_INSNS (1);
return false;
case ASHIFT:
if (TARGET_EXTSWSLI && mode == DImode
&& GET_CODE (XEXP (x, 0)) == SIGN_EXTEND
&& GET_MODE (XEXP (XEXP (x, 0), 0)) == SImode)
{
*total = 0;
return false;
}
case ASHIFTRT:
case LSHIFTRT:
case ROTATE:
case ROTATERT:
if (outer_code == TRUNCATE
&& GET_CODE (XEXP (x, 0)) == MULT)
{
if (mode == DImode)
*total = rs6000_cost->muldi;
else
*total = rs6000_cost->mulsi;
return true;
}
else if (outer_code == AND)
*total = 0;
else
*total = COSTS_N_INSNS (1);
return false;
case SIGN_EXTEND:
case ZERO_EXTEND:
if (GET_CODE (XEXP (x, 0)) == MEM)
*total = 0;
else
*total = COSTS_N_INSNS (1);
return false;
case COMPARE:
case NEG:
case ABS:
if (!FLOAT_MODE_P (mode))
{
*total = COSTS_N_INSNS (1);
return false;
}
case FLOAT:
case UNSIGNED_FLOAT:
case FIX:
case UNSIGNED_FIX:
case FLOAT_TRUNCATE:
*total = rs6000_cost->fp;
return false;
case FLOAT_EXTEND:
if (mode == DFmode)
*total = rs6000_cost->sfdf_convert;
else
*total = rs6000_cost->fp;
return false;
case UNSPEC:
switch (XINT (x, 1))
{
case UNSPEC_FRSP:
*total = rs6000_cost->fp;
return true;
default:
break;
}
break;
case CALL:
case IF_THEN_ELSE:
if (!speed)
{
*total = COSTS_N_INSNS (1);
return true;
}
else if (FLOAT_MODE_P (mode) && TARGET_PPC_GFXOPT && TARGET_HARD_FLOAT)
{
*total = rs6000_cost->fp;
return false;
}
break;
case NE:
case EQ:
case GTU:
case LTU:
if (mode == Pmode
&& (outer_code == NEG || outer_code == PLUS))
{
*total = COSTS_N_INSNS (1);
return true;
}
case GT:
case LT:
case UNORDERED:
if (outer_code == SET)
{
if (XEXP (x, 1) == const0_rtx)
{
*total = COSTS_N_INSNS (2);
return true;
}
else
{
*total = COSTS_N_INSNS (3);
return false;
}
}
if (outer_code == COMPARE)
{
*total = 0;
return true;
}
break;
default:
break;
}
return false;
}
static bool
rs6000_debug_rtx_costs (rtx x, machine_mode mode, int outer_code,
int opno, int *total, bool speed)
{
bool ret = rs6000_rtx_costs (x, mode, outer_code, opno, total, speed);
fprintf (stderr,
"\nrs6000_rtx_costs, return = %s, mode = %s, outer_code = %s, "
"opno = %d, total = %d, speed = %s, x:\n",
ret ? "complete" : "scan inner",
GET_MODE_NAME (mode),
GET_RTX_NAME (outer_code),
opno,
*total,
speed ? "true" : "false");
debug_rtx (x);
return ret;
}
static int
rs6000_insn_cost (rtx_insn *insn, bool speed)
{
if (recog_memoized (insn) < 0)
return 0;
if (!speed)
return get_attr_length (insn);
int cost = get_attr_cost (insn);
if (cost > 0)
return cost;
int n = get_attr_length (insn) / 4;
enum attr_type type = get_attr_type (insn);
switch (type)
{
case TYPE_LOAD:
case TYPE_FPLOAD:
case TYPE_VECLOAD:
cost = COSTS_N_INSNS (n + 1);
break;
case TYPE_MUL:
switch (get_attr_size (insn))
{
case SIZE_8:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->mulsi_const9;
break;
case SIZE_16:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->mulsi_const;
break;
case SIZE_32:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->mulsi;
break;
case SIZE_64:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->muldi;
break;
default:
gcc_unreachable ();
}
break;
case TYPE_DIV:
switch (get_attr_size (insn))
{
case SIZE_32:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->divsi;
break;
case SIZE_64:
cost = COSTS_N_INSNS (n - 1) + rs6000_cost->divdi;
break;
default:
gcc_unreachable ();
}
break;
case TYPE_FP:
cost = n * rs6000_cost->fp;
break;
case TYPE_DMUL:
cost = n * rs6000_cost->dmul;
break;
case TYPE_SDIV:
cost = n * rs6000_cost->sdiv;
break;
case TYPE_DDIV:
cost = n * rs6000_cost->ddiv;
break;
case TYPE_SYNC:
case TYPE_LOAD_L:
case TYPE_MFCR:
case TYPE_MFCRF:
cost = COSTS_N_INSNS (n + 2);
break;
default:
cost = COSTS_N_INSNS (n);
}
return cost;
}
static int
rs6000_debug_address_cost (rtx x, machine_mode mode,
addr_space_t as, bool speed)
{
int ret = TARGET_ADDRESS_COST (x, mode, as, speed);
fprintf (stderr, "\nrs6000_address_cost, return = %d, speed = %s, x:\n",
ret, speed ? "true" : "false");
debug_rtx (x);
return ret;
}
static int
rs6000_register_move_cost (machine_mode mode,
reg_class_t from, reg_class_t to)
{
int ret;
if (TARGET_DEBUG_COST)
dbg_cost_ctrl++;
if (reg_classes_intersect_p (to, GENERAL_REGS)
|| reg_classes_intersect_p (from, GENERAL_REGS))
{
reg_class_t rclass = from;
if (! reg_classes_intersect_p (to, GENERAL_REGS))
rclass = to;
if (rclass == FLOAT_REGS || rclass == ALTIVEC_REGS || rclass == VSX_REGS)
ret = (rs6000_memory_move_cost (mode, rclass, false)
+ rs6000_memory_move_cost (mode, GENERAL_REGS, false));
else if (rclass == CR_REGS)
ret = 4;
else if ((rs6000_tune == PROCESSOR_POWER6
|| rs6000_tune == PROCESSOR_POWER7
|| rs6000_tune == PROCESSOR_POWER8
|| rs6000_tune == PROCESSOR_POWER9)
&& reg_classes_intersect_p (rclass, LINK_OR_CTR_REGS))
ret = 6 * hard_regno_nregs (0, mode);
else
ret = 2 * hard_regno_nregs (0, mode);
}
else if (VECTOR_MEM_VSX_P (mode)
&& reg_classes_intersect_p (to, VSX_REGS)
&& reg_classes_intersect_p (from, VSX_REGS))
ret = 2 * hard_regno_nregs (FIRST_FPR_REGNO, mode);
else if (reg_classes_intersect_p (to, from))
ret = (FLOAT128_2REG_P (mode)) ? 4 : 2;
else
ret = (rs6000_register_move_cost (mode, GENERAL_REGS, to)
+ rs6000_register_move_cost (mode, from, GENERAL_REGS));
if (TARGET_DEBUG_COST)
{
if (dbg_cost_ctrl == 1)
fprintf (stderr,
"rs6000_register_move_cost:, ret=%d, mode=%s, from=%s, to=%s\n",
ret, GET_MODE_NAME (mode), reg_class_names[from],
reg_class_names[to]);
dbg_cost_ctrl--;
}
return ret;
}
static int
rs6000_memory_move_cost (machine_mode mode, reg_class_t rclass,
bool in ATTRIBUTE_UNUSED)
{
int ret;
if (TARGET_DEBUG_COST)
dbg_cost_ctrl++;
if (reg_classes_intersect_p (rclass, GENERAL_REGS))
ret = 4 * hard_regno_nregs (0, mode);
else if ((reg_classes_intersect_p (rclass, FLOAT_REGS)
|| reg_classes_intersect_p (rclass, VSX_REGS)))
ret = 4 * hard_regno_nregs (32, mode);
else if (reg_classes_intersect_p (rclass, ALTIVEC_REGS))
ret = 4 * hard_regno_nregs (FIRST_ALTIVEC_REGNO, mode);
else
ret = 4 + rs6000_register_move_cost (mode, rclass, GENERAL_REGS);
if (TARGET_DEBUG_COST)
{
if (dbg_cost_ctrl == 1)
fprintf (stderr,
"rs6000_memory_move_cost: ret=%d, mode=%s, rclass=%s, in=%d\n",
ret, GET_MODE_NAME (mode), reg_class_names[rclass], in);
dbg_cost_ctrl--;
}
return ret;
}
static tree
rs6000_builtin_reciprocal (tree fndecl)
{
switch (DECL_FUNCTION_CODE (fndecl))
{
case VSX_BUILTIN_XVSQRTDP:
if (!RS6000_RECIP_AUTO_RSQRTE_P (V2DFmode))
return NULL_TREE;
return rs6000_builtin_decls[VSX_BUILTIN_RSQRT_2DF];
case VSX_BUILTIN_XVSQRTSP:
if (!RS6000_RECIP_AUTO_RSQRTE_P (V4SFmode))
return NULL_TREE;
return rs6000_builtin_decls[VSX_BUILTIN_RSQRT_4SF];
default:
return NULL_TREE;
}
}
static rtx
rs6000_load_constant_and_splat (machine_mode mode, REAL_VALUE_TYPE dconst)
{
rtx reg;
if (mode == SFmode || mode == DFmode)
{
rtx d = const_double_from_real_value (dconst, mode);
reg = force_reg (mode, d);
}
else if (mode == V4SFmode)
{
rtx d = const_double_from_real_value (dconst, SFmode);
rtvec v = gen_rtvec (4, d, d, d, d);
reg = gen_reg_rtx (mode);
rs6000_expand_vector_init (reg, gen_rtx_PARALLEL (mode, v));
}
else if (mode == V2DFmode)
{
rtx d = const_double_from_real_value (dconst, DFmode);
rtvec v = gen_rtvec (2, d, d);
reg = gen_reg_rtx (mode);
rs6000_expand_vector_init (reg, gen_rtx_PARALLEL (mode, v));
}
else
gcc_unreachable ();
return reg;
}
static void
rs6000_emit_madd (rtx target, rtx m1, rtx m2, rtx a)
{
machine_mode mode = GET_MODE (target);
rtx dst;
dst = expand_ternary_op (mode, fma_optab, m1, m2, a, target, 0);
gcc_assert (dst != NULL);
if (dst != target)
emit_move_insn (target, dst);
}
static void
rs6000_emit_nmsub (rtx dst, rtx m1, rtx m2, rtx a)
{
machine_mode mode = GET_MODE (dst);
rtx r;
gcc_assert (optab_handler (fma_optab, mode) != CODE_FOR_nothing);
r = gen_rtx_NEG (mode, a);
r = gen_rtx_FMA (mode, m1, m2, r);
r = gen_rtx_NEG (mode, r);
emit_insn (gen_rtx_SET (dst, r));
}
void
rs6000_emit_swdiv (rtx dst, rtx n, rtx d, bool note_p)
{
machine_mode mode = GET_MODE (dst);
rtx one, x0, e0, x1, xprev, eprev, xnext, enext, u, v;
int i;
int passes = (TARGET_RECIP_PRECISION) ? 1 : 3;
if (mode == DFmode || mode == V2DFmode)
passes++;
enum insn_code code = optab_handler (smul_optab, mode);
insn_gen_fn gen_mul = GEN_FCN (code);
gcc_assert (code != CODE_FOR_nothing);
one = rs6000_load_constant_and_splat (mode, dconst1);
x0 = gen_reg_rtx (mode);
emit_insn (gen_rtx_SET (x0, gen_rtx_UNSPEC (mode, gen_rtvec (1, d),
UNSPEC_FRES)));
if (passes > 1) {
e0 = gen_reg_rtx (mode);
rs6000_emit_nmsub (e0, d, x0, one);
x1 = gen_reg_rtx (mode);
rs6000_emit_madd (x1, e0, x0, x0);
for (i = 0, xprev = x1, eprev = e0; i < passes - 2;
++i, xprev = xnext, eprev = enext) {
enext = gen_reg_rtx (mode);
emit_insn (gen_mul (enext, eprev, eprev));
xnext = gen_reg_rtx (mode);
rs6000_emit_madd (xnext, enext, xprev, xprev);
}
} else
xprev = x0;
u = gen_reg_rtx (mode);
emit_insn (gen_mul (u, n, xprev));
v = gen_reg_rtx (mode);
rs6000_emit_nmsub (v, d, u, n);
rs6000_emit_madd (dst, v, xprev, u);
if (note_p)
add_reg_note (get_last_insn (), REG_EQUAL, gen_rtx_DIV (mode, n, d));
}
void
rs6000_emit_swsqrt (rtx dst, rtx src, bool recip)
{
machine_mode mode = GET_MODE (src);
rtx e = gen_reg_rtx (mode);
rtx g = gen_reg_rtx (mode);
rtx h = gen_reg_rtx (mode);
int passes = (TARGET_RECIP_PRECISION) ? 1 : 3;
if (mode == DFmode || mode == V2DFmode)
passes++;
int i;
rtx mhalf;
enum insn_code code = optab_handler (smul_optab, mode);
insn_gen_fn gen_mul = GEN_FCN (code);
gcc_assert (code != CODE_FOR_nothing);
mhalf = rs6000_load_constant_and_splat (mode, dconsthalf);
emit_insn (gen_rtx_SET (e, gen_rtx_UNSPEC (mode, gen_rtvec (1, src),
UNSPEC_RSQRT)));
if (!recip)
{
rtx zero = force_reg (mode, CONST0_RTX (mode));
if (mode == SFmode)
{
rtx target = emit_conditional_move (e, GT, src, zero, mode,
e, zero, mode, 0);
if (target != e)
emit_move_insn (e, target);
}
else
{
rtx cond = gen_rtx_GT (VOIDmode, e, zero);
rs6000_emit_vector_cond_expr (e, e, zero, cond, src, zero);
}
}
emit_insn (gen_mul (g, e, src));
emit_insn (gen_mul (h, e, mhalf));
if (recip)
{
if (passes == 1)
{
rtx t = gen_reg_rtx (mode);
rs6000_emit_nmsub (t, g, h, mhalf);
rs6000_emit_madd (dst, e, t, e);
}
else
{
for (i = 0; i < passes; i++)
{
rtx t1 = gen_reg_rtx (mode);
rtx g1 = gen_reg_rtx (mode);
rtx h1 = gen_reg_rtx (mode);
rs6000_emit_nmsub (t1, g, h, mhalf);
rs6000_emit_madd (g1, g, t1, g);
rs6000_emit_madd (h1, h, t1, h);
g = g1;
h = h1;
}
emit_insn (gen_add3_insn (dst, h, h));
}
}
else
{
rtx t = gen_reg_rtx (mode);
rs6000_emit_nmsub (t, g, h, mhalf);
rs6000_emit_madd (dst, g, t, g);
}
return;
}
void
rs6000_emit_popcount (rtx dst, rtx src)
{
machine_mode mode = GET_MODE (dst);
rtx tmp1, tmp2;
if (TARGET_POPCNTD)
{
if (mode == SImode)
emit_insn (gen_popcntdsi2 (dst, src));
else
emit_insn (gen_popcntddi2 (dst, src));
return;
}
tmp1 = gen_reg_rtx (mode);
if (mode == SImode)
{
emit_insn (gen_popcntbsi2 (tmp1, src));
tmp2 = expand_mult (SImode, tmp1, GEN_INT (0x01010101),
NULL_RTX, 0);
tmp2 = force_reg (SImode, tmp2);
emit_insn (gen_lshrsi3 (dst, tmp2, GEN_INT (24)));
}
else
{
emit_insn (gen_popcntbdi2 (tmp1, src));
tmp2 = expand_mult (DImode, tmp1,
GEN_INT ((HOST_WIDE_INT)
0x01010101 << 32 | 0x01010101),
NULL_RTX, 0);
tmp2 = force_reg (DImode, tmp2);
emit_insn (gen_lshrdi3 (dst, tmp2, GEN_INT (56)));
}
}
void
rs6000_emit_parity (rtx dst, rtx src)
{
machine_mode mode = GET_MODE (dst);
rtx tmp;
tmp = gen_reg_rtx (mode);
if (TARGET_CMPB)
{
if (mode == SImode)
{
emit_insn (gen_popcntbsi2 (tmp, src));
emit_insn (gen_paritysi2_cmpb (dst, tmp));
}
else
{
emit_insn (gen_popcntbdi2 (tmp, src));
emit_insn (gen_paritydi2_cmpb (dst, tmp));
}
return;
}
if (mode == SImode)
{
if (rs6000_cost->mulsi_const >= COSTS_N_INSNS (3))
{
rtx tmp1, tmp2, tmp3, tmp4;
tmp1 = gen_reg_rtx (SImode);
emit_insn (gen_popcntbsi2 (tmp1, src));
tmp2 = gen_reg_rtx (SImode);
emit_insn (gen_lshrsi3 (tmp2, tmp1, GEN_INT (16)));
tmp3 = gen_reg_rtx (SImode);
emit_insn (gen_xorsi3 (tmp3, tmp1, tmp2));
tmp4 = gen_reg_rtx (SImode);
emit_insn (gen_lshrsi3 (tmp4, tmp3, GEN_INT (8)));
emit_insn (gen_xorsi3 (tmp, tmp3, tmp4));
}
else
rs6000_emit_popcount (tmp, src);
emit_insn (gen_andsi3 (dst, tmp, const1_rtx));
}
else
{
if (rs6000_cost->muldi >= COSTS_N_INSNS (5))
{
rtx tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
tmp1 = gen_reg_rtx (DImode);
emit_insn (gen_popcntbdi2 (tmp1, src));
tmp2 = gen_reg_rtx (DImode);
emit_insn (gen_lshrdi3 (tmp2, tmp1, GEN_INT (32)));
tmp3 = gen_reg_rtx (DImode);
emit_insn (gen_xordi3 (tmp3, tmp1, tmp2));
tmp4 = gen_reg_rtx (DImode);
emit_insn (gen_lshrdi3 (tmp4, tmp3, GEN_INT (16)));
tmp5 = gen_reg_rtx (DImode);
emit_insn (gen_xordi3 (tmp5, tmp3, tmp4));
tmp6 = gen_reg_rtx (DImode);
emit_insn (gen_lshrdi3 (tmp6, tmp5, GEN_INT (8)));
emit_insn (gen_xordi3 (tmp, tmp5, tmp6));
}
else
rs6000_emit_popcount (tmp, src);
emit_insn (gen_anddi3 (dst, tmp, const1_rtx));
}
}
static void
altivec_expand_vec_perm_const_le (rtx target, rtx op0, rtx op1,
const vec_perm_indices &sel)
{
unsigned int i;
rtx perm[16];
rtx constv, unspec;
for (i = 0; i < 16; ++i)
{
unsigned int elt = 31 - (sel[i] & 31);
perm[i] = GEN_INT (elt);
}
if (!REG_P (op0))
op0 = force_reg (V16QImode, op0);
if (!REG_P (op1))
op1 = force_reg (V16QImode, op1);
constv = gen_rtx_CONST_VECTOR (V16QImode, gen_rtvec_v (16, perm));
constv = force_reg (V16QImode, constv);
unspec = gen_rtx_UNSPEC (V16QImode, gen_rtvec (3, op1, op0, constv),
UNSPEC_VPERM);
if (!REG_P (target))
{
rtx tmp = gen_reg_rtx (V16QImode);
emit_move_insn (tmp, unspec);
unspec = tmp;
}
emit_move_insn (target, unspec);
}
void
altivec_expand_vec_perm_le (rtx operands[4])
{
rtx notx, iorx, unspec;
rtx target = operands[0];
rtx op0 = operands[1];
rtx op1 = operands[2];
rtx sel = operands[3];
rtx tmp = target;
rtx norreg = gen_reg_rtx (V16QImode);
machine_mode mode = GET_MODE (target);
if (!REG_P (op0))
op0 = force_reg (mode, op0);
if (!REG_P (op1))
op1 = force_reg (mode, op1);
if (!REG_P (sel))
sel = force_reg (V16QImode, sel);
if (!REG_P (target))
tmp = gen_reg_rtx (mode);
if (TARGET_P9_VECTOR)
{
unspec = gen_rtx_UNSPEC (mode, gen_rtvec (3, op1, op0, sel),
UNSPEC_VPERMR);
}
else
{
notx = gen_rtx_NOT (V16QImode, sel);
iorx = (TARGET_P8_VECTOR
? gen_rtx_IOR (V16QImode, notx, notx)
: gen_rtx_AND (V16QImode, notx, notx));
emit_insn (gen_rtx_SET (norreg, iorx));
unspec = gen_rtx_UNSPEC (mode, gen_rtvec (3, op1, op0, norreg),
UNSPEC_VPERM);
}
if (!REG_P (target))
{
emit_move_insn (tmp, unspec);
unspec = tmp;
}
emit_move_insn (target, unspec);
}
static bool
altivec_expand_vec_perm_const (rtx target, rtx op0, rtx op1,
const vec_perm_indices &sel)
{
struct altivec_perm_insn {
HOST_WIDE_INT mask;
enum insn_code impl;
unsigned char perm[16];
};
static const struct altivec_perm_insn patterns[] = {
{ OPTION_MASK_ALTIVEC, CODE_FOR_altivec_vpkuhum_direct,
{  1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 } },
{ OPTION_MASK_ALTIVEC, CODE_FOR_altivec_vpkuwum_direct,
{  2,  3,  6,  7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31 } },
{ OPTION_MASK_ALTIVEC, 
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrghb_direct
: CODE_FOR_altivec_vmrglb_direct),
{  0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23 } },
{ OPTION_MASK_ALTIVEC,
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrghh_direct
: CODE_FOR_altivec_vmrglh_direct),
{  0,  1, 16, 17,  2,  3, 18, 19,  4,  5, 20, 21,  6,  7, 22, 23 } },
{ OPTION_MASK_ALTIVEC,
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrghw_direct
: CODE_FOR_altivec_vmrglw_direct),
{  0,  1,  2,  3, 16, 17, 18, 19,  4,  5,  6,  7, 20, 21, 22, 23 } },
{ OPTION_MASK_ALTIVEC,
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrglb_direct
: CODE_FOR_altivec_vmrghb_direct),
{  8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 } },
{ OPTION_MASK_ALTIVEC,
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrglh_direct
: CODE_FOR_altivec_vmrghh_direct),
{  8,  9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31 } },
{ OPTION_MASK_ALTIVEC,
(BYTES_BIG_ENDIAN ? CODE_FOR_altivec_vmrglw_direct
: CODE_FOR_altivec_vmrghw_direct),
{  8,  9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31 } },
{ OPTION_MASK_P8_VECTOR,
(BYTES_BIG_ENDIAN ? CODE_FOR_p8_vmrgew_v4sf_direct
: CODE_FOR_p8_vmrgow_v4sf_direct),
{  0,  1,  2,  3, 16, 17, 18, 19,  8,  9, 10, 11, 24, 25, 26, 27 } },
{ OPTION_MASK_P8_VECTOR,
(BYTES_BIG_ENDIAN ? CODE_FOR_p8_vmrgow_v4sf_direct
: CODE_FOR_p8_vmrgew_v4sf_direct),
{  4,  5,  6,  7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 } }
};
unsigned int i, j, elt, which;
unsigned char perm[16];
rtx x;
bool one_vec;
for (i = which = 0; i < 16; ++i)
{
elt = sel[i] & 31;
which |= (elt < 16 ? 1 : 2);
perm[i] = elt;
}
switch (which)
{
default:
gcc_unreachable ();
case 3:
one_vec = false;
if (!rtx_equal_p (op0, op1))
break;
case 2:
for (i = 0; i < 16; ++i)
perm[i] &= 15;
op0 = op1;
one_vec = true;
break;
case 1:
op1 = op0;
one_vec = true;
break;
}
if (one_vec)
{
elt = perm[0];
for (i = 0; i < 16; ++i)
if (perm[i] != elt)
break;
if (i == 16)
{
if (!BYTES_BIG_ENDIAN)
elt = 15 - elt;
emit_insn (gen_altivec_vspltb_direct (target, op0, GEN_INT (elt)));
return true;
}
if (elt % 2 == 0)
{
for (i = 0; i < 16; i += 2)
if (perm[i] != elt || perm[i + 1] != elt + 1)
break;
if (i == 16)
{
int field = BYTES_BIG_ENDIAN ? elt / 2 : 7 - elt / 2;
x = gen_reg_rtx (V8HImode);
emit_insn (gen_altivec_vsplth_direct (x, gen_lowpart (V8HImode, op0),
GEN_INT (field)));
emit_move_insn (target, gen_lowpart (V16QImode, x));
return true;
}
}
if (elt % 4 == 0)
{
for (i = 0; i < 16; i += 4)
if (perm[i] != elt
|| perm[i + 1] != elt + 1
|| perm[i + 2] != elt + 2
|| perm[i + 3] != elt + 3)
break;
if (i == 16)
{
int field = BYTES_BIG_ENDIAN ? elt / 4 : 3 - elt / 4;
x = gen_reg_rtx (V4SImode);
emit_insn (gen_altivec_vspltw_direct (x, gen_lowpart (V4SImode, op0),
GEN_INT (field)));
emit_move_insn (target, gen_lowpart (V16QImode, x));
return true;
}
}
}
for (j = 0; j < ARRAY_SIZE (patterns); ++j)
{
bool swapped;
if ((patterns[j].mask & rs6000_isa_flags) == 0)
continue;
elt = patterns[j].perm[0];
if (perm[0] == elt)
swapped = false;
else if (perm[0] == elt + 16)
swapped = true;
else
continue;
for (i = 1; i < 16; ++i)
{
elt = patterns[j].perm[i];
if (swapped)
elt = (elt >= 16 ? elt - 16 : elt + 16);
else if (one_vec && elt >= 16)
elt -= 16;
if (perm[i] != elt)
break;
}
if (i == 16)
{
enum insn_code icode = patterns[j].impl;
machine_mode omode = insn_data[icode].operand[0].mode;
machine_mode imode = insn_data[icode].operand[1].mode;
if (!BYTES_BIG_ENDIAN
&& icode == CODE_FOR_altivec_vpkuwum_direct
&& ((GET_CODE (op0) == REG
&& GET_MODE (op0) != V4SImode)
|| (GET_CODE (op0) == SUBREG
&& GET_MODE (XEXP (op0, 0)) != V4SImode)))
continue;
if (!BYTES_BIG_ENDIAN
&& icode == CODE_FOR_altivec_vpkuhum_direct
&& ((GET_CODE (op0) == REG
&& GET_MODE (op0) != V8HImode)
|| (GET_CODE (op0) == SUBREG
&& GET_MODE (XEXP (op0, 0)) != V8HImode)))
continue;
if (swapped ^ !BYTES_BIG_ENDIAN)
std::swap (op0, op1);
if (imode != V16QImode)
{
op0 = gen_lowpart (imode, op0);
op1 = gen_lowpart (imode, op1);
}
if (omode == V16QImode)
x = target;
else
x = gen_reg_rtx (omode);
emit_insn (GEN_FCN (icode) (x, op0, op1));
if (omode != V16QImode)
emit_move_insn (target, gen_lowpart (V16QImode, x));
return true;
}
}
if (!BYTES_BIG_ENDIAN)
{
altivec_expand_vec_perm_const_le (target, op0, op1, sel);
return true;
}
return false;
}
static bool
rs6000_expand_vec_perm_const_1 (rtx target, rtx op0, rtx op1,
unsigned char perm0, unsigned char perm1)
{
rtx x;
if ((perm0 & 2) == (perm1 & 2))
{
if (perm0 & 2)
op0 = op1;
else
op1 = op0;
}
if (rtx_equal_p (op0, op1))
{
perm0 = perm0 & 1;
perm1 = (perm1 & 1) + 2;
}
else if (perm0 & 2)
{
if (perm1 & 2)
return false;
perm0 -= 2;
perm1 += 2;
std::swap (op0, op1);
}
else if ((perm1 & 2) == 0)
return false;
if (target != NULL)
{
machine_mode vmode, dmode;
rtvec v;
vmode = GET_MODE (target);
gcc_assert (GET_MODE_NUNITS (vmode) == 2);
dmode = mode_for_vector (GET_MODE_INNER (vmode), 4).require ();
x = gen_rtx_VEC_CONCAT (dmode, op0, op1);
v = gen_rtvec (2, GEN_INT (perm0), GEN_INT (perm1));
x = gen_rtx_VEC_SELECT (vmode, x, gen_rtx_PARALLEL (VOIDmode, v));
emit_insn (gen_rtx_SET (target, x));
}
return true;
}
static bool
rs6000_vectorize_vec_perm_const (machine_mode vmode, rtx target, rtx op0,
rtx op1, const vec_perm_indices &sel)
{
bool testing_p = !target;
if (TARGET_ALTIVEC && testing_p)
return true;
if ((vmode == V2SFmode && TARGET_PAIRED_FLOAT)
|| ((vmode == V2DFmode || vmode == V2DImode)
&& VECTOR_MEM_VSX_P (vmode)))
{
if (testing_p)
{
op0 = gen_raw_REG (vmode, LAST_VIRTUAL_REGISTER + 1);
op1 = gen_raw_REG (vmode, LAST_VIRTUAL_REGISTER + 2);
}
if (rs6000_expand_vec_perm_const_1 (target, op0, op1, sel[0], sel[1]))
return true;
}
if (TARGET_ALTIVEC)
{
if (vmode != V16QImode)
return false;
if (altivec_expand_vec_perm_const (target, op0, op1, sel))
return true;
}
return false;
}
static void
rs6000_do_expand_vec_perm (rtx target, rtx op0, rtx op1,
machine_mode vmode, const vec_perm_builder &perm)
{
rtx x = expand_vec_perm_const (vmode, op0, op1, perm, BLKmode, target);
if (x != target)
emit_move_insn (target, x);
}
void
rs6000_expand_extract_even (rtx target, rtx op0, rtx op1)
{
machine_mode vmode = GET_MODE (target);
unsigned i, nelt = GET_MODE_NUNITS (vmode);
vec_perm_builder perm (nelt, nelt, 1);
for (i = 0; i < nelt; i++)
perm.quick_push (i * 2);
rs6000_do_expand_vec_perm (target, op0, op1, vmode, perm);
}
void
rs6000_expand_interleave (rtx target, rtx op0, rtx op1, bool highp)
{
machine_mode vmode = GET_MODE (target);
unsigned i, high, nelt = GET_MODE_NUNITS (vmode);
vec_perm_builder perm (nelt, nelt, 1);
high = (highp ? 0 : nelt / 2);
for (i = 0; i < nelt / 2; i++)
{
perm.quick_push (i + high);
perm.quick_push (i + nelt + high);
}
rs6000_do_expand_vec_perm (target, op0, op1, vmode, perm);
}
void
rs6000_scale_v2df (rtx tgt, rtx src, int scale)
{
HOST_WIDE_INT hwi_scale (scale);
REAL_VALUE_TYPE r_pow;
rtvec v = rtvec_alloc (2);
rtx elt;
rtx scale_vec = gen_reg_rtx (V2DFmode);
(void)real_powi (&r_pow, DFmode, &dconst2, hwi_scale);
elt = const_double_from_real_value (r_pow, DFmode);
RTVEC_ELT (v, 0) = elt;
RTVEC_ELT (v, 1) = elt;
rs6000_expand_vector_init (scale_vec, gen_rtx_PARALLEL (V2DFmode, v));
emit_insn (gen_mulv2df3 (tgt, src, scale_vec));
}
static rtx
rs6000_complex_function_value (machine_mode mode)
{
unsigned int regno;
rtx r1, r2;
machine_mode inner = GET_MODE_INNER (mode);
unsigned int inner_bytes = GET_MODE_UNIT_SIZE (mode);
if (TARGET_FLOAT128_TYPE
&& (mode == KCmode
|| (mode == TCmode && TARGET_IEEEQUAD)))
regno = ALTIVEC_ARG_RETURN;
else if (FLOAT_MODE_P (mode) && TARGET_HARD_FLOAT)
regno = FP_ARG_RETURN;
else
{
regno = GP_ARG_RETURN;
if (TARGET_32BIT && inner_bytes >= 4)
return gen_rtx_REG (mode, regno);
}
if (inner_bytes >= 8)
return gen_rtx_REG (mode, regno);
r1 = gen_rtx_EXPR_LIST (inner, gen_rtx_REG (inner, regno),
const0_rtx);
r2 = gen_rtx_EXPR_LIST (inner, gen_rtx_REG (inner, regno + 1),
GEN_INT (inner_bytes));
return gen_rtx_PARALLEL (mode, gen_rtvec (2, r1, r2));
}
static rtx
rs6000_parallel_return (machine_mode mode,
int n_elts, machine_mode elt_mode,
unsigned int regno, unsigned int reg_stride)
{
rtx par = gen_rtx_PARALLEL (mode, rtvec_alloc (n_elts));
int i;
for (i = 0; i < n_elts; i++)
{
rtx r = gen_rtx_REG (elt_mode, regno);
rtx off = GEN_INT (i * GET_MODE_SIZE (elt_mode));
XVECEXP (par, 0, i) = gen_rtx_EXPR_LIST (VOIDmode, r, off);
regno += reg_stride;
}
return par;
}
static rtx
rs6000_function_value (const_tree valtype,
const_tree fn_decl_or_type ATTRIBUTE_UNUSED,
bool outgoing ATTRIBUTE_UNUSED)
{
machine_mode mode;
unsigned int regno;
machine_mode elt_mode;
int n_elts;
if (TARGET_MACHO 
&& rs6000_darwin64_struct_check_p (TYPE_MODE (valtype), valtype))
{
CUMULATIVE_ARGS valcum;
rtx valret;
valcum.words = 0;
valcum.fregno = FP_ARG_MIN_REG;
valcum.vregno = ALTIVEC_ARG_MIN_REG;
valret = rs6000_darwin64_record_arg (&valcum, valtype, true,  true);
if (valret)
return valret;
}
mode = TYPE_MODE (valtype);
if (rs6000_discover_homogeneous_aggregate (mode, valtype, &elt_mode, &n_elts))
{
int first_reg, n_regs;
if (SCALAR_FLOAT_MODE_NOT_VECTOR_P (elt_mode))
{
first_reg = (elt_mode == TDmode) ? FP_ARG_RETURN + 1 : FP_ARG_RETURN;
n_regs = (GET_MODE_SIZE (elt_mode) + 7) >> 3;
}
else
{
first_reg = ALTIVEC_ARG_RETURN;
n_regs = 1;
}
return rs6000_parallel_return (mode, n_elts, elt_mode, first_reg, n_regs);
}
if (TARGET_32BIT && TARGET_POWERPC64)
switch (mode)
{
default:
break;
case E_DImode:
case E_SCmode:
case E_DCmode:
case E_TCmode:
int count = GET_MODE_SIZE (mode) / 4;
return rs6000_parallel_return (mode, count, SImode, GP_ARG_RETURN, 1);
}
if ((INTEGRAL_TYPE_P (valtype)
&& GET_MODE_BITSIZE (mode) < (TARGET_32BIT ? 32 : 64))
|| POINTER_TYPE_P (valtype))
mode = TARGET_32BIT ? SImode : DImode;
if (DECIMAL_FLOAT_MODE_P (mode) && TARGET_HARD_FLOAT)
regno = (mode == TDmode) ? FP_ARG_RETURN + 1 : FP_ARG_RETURN;
else if (SCALAR_FLOAT_TYPE_P (valtype) && TARGET_HARD_FLOAT
&& !FLOAT128_VECTOR_P (mode)
&& ((TARGET_SINGLE_FLOAT && (mode == SFmode)) || TARGET_DOUBLE_FLOAT))
regno = FP_ARG_RETURN;
else if (TREE_CODE (valtype) == COMPLEX_TYPE
&& targetm.calls.split_complex_arg)
return rs6000_complex_function_value (mode);
else if ((TREE_CODE (valtype) == VECTOR_TYPE || FLOAT128_VECTOR_P (mode))
&& TARGET_ALTIVEC && TARGET_ALTIVEC_ABI
&& ALTIVEC_OR_VSX_VECTOR_MODE (mode))
regno = ALTIVEC_ARG_RETURN;
else
regno = GP_ARG_RETURN;
return gen_rtx_REG (mode, regno);
}
rtx
rs6000_libcall_value (machine_mode mode)
{
unsigned int regno;
if (TARGET_32BIT && TARGET_POWERPC64 && mode == DImode)
return rs6000_parallel_return (mode, 2, SImode, GP_ARG_RETURN, 1);
if (DECIMAL_FLOAT_MODE_P (mode) && TARGET_HARD_FLOAT)
regno = (mode == TDmode) ? FP_ARG_RETURN + 1 : FP_ARG_RETURN;
else if (SCALAR_FLOAT_MODE_NOT_VECTOR_P (mode)
&& TARGET_HARD_FLOAT
&& ((TARGET_SINGLE_FLOAT && mode == SFmode) || TARGET_DOUBLE_FLOAT))
regno = FP_ARG_RETURN;
else if (ALTIVEC_OR_VSX_VECTOR_MODE (mode)
&& TARGET_ALTIVEC && TARGET_ALTIVEC_ABI)
regno = ALTIVEC_ARG_RETURN;
else if (COMPLEX_MODE_P (mode) && targetm.calls.split_complex_arg)
return rs6000_complex_function_value (mode);
else
regno = GP_ARG_RETURN;
return gen_rtx_REG (mode, regno);
}
static int
rs6000_compute_pressure_classes (enum reg_class *pressure_classes)
{
int n;
n = 0;
pressure_classes[n++] = GENERAL_REGS;
if (TARGET_VSX)
pressure_classes[n++] = VSX_REGS;
else
{
if (TARGET_ALTIVEC)
pressure_classes[n++] = ALTIVEC_REGS;
if (TARGET_HARD_FLOAT)
pressure_classes[n++] = FLOAT_REGS;
}
pressure_classes[n++] = CR_REGS;
pressure_classes[n++] = SPECIAL_REGS;
return n;
}
static bool
rs6000_can_eliminate (const int from, const int to)
{
return (from == ARG_POINTER_REGNUM && to == STACK_POINTER_REGNUM
? ! frame_pointer_needed
: from == RS6000_PIC_OFFSET_TABLE_REGNUM
? ! TARGET_MINIMAL_TOC || TARGET_NO_TOC
|| constant_pool_empty_p ()
: true);
}
HOST_WIDE_INT
rs6000_initial_elimination_offset (int from, int to)
{
rs6000_stack_t *info = rs6000_stack_info ();
HOST_WIDE_INT offset;
if (from == HARD_FRAME_POINTER_REGNUM && to == STACK_POINTER_REGNUM)
offset = info->push_p ? 0 : -info->total_size;
else if (from == FRAME_POINTER_REGNUM && to == STACK_POINTER_REGNUM)
{
offset = info->push_p ? 0 : -info->total_size;
if (FRAME_GROWS_DOWNWARD)
offset += info->fixed_size + info->vars_size + info->parm_size;
}
else if (from == FRAME_POINTER_REGNUM && to == HARD_FRAME_POINTER_REGNUM)
offset = FRAME_GROWS_DOWNWARD
? info->fixed_size + info->vars_size + info->parm_size
: 0;
else if (from == ARG_POINTER_REGNUM && to == HARD_FRAME_POINTER_REGNUM)
offset = info->total_size;
else if (from == ARG_POINTER_REGNUM && to == STACK_POINTER_REGNUM)
offset = info->push_p ? info->total_size : 0;
else if (from == RS6000_PIC_OFFSET_TABLE_REGNUM)
offset = 0;
else
gcc_unreachable ();
return offset;
}
static void
rs6000_init_dwarf_reg_sizes_extra (tree address)
{
if (TARGET_MACHO && ! TARGET_ALTIVEC)
{
int i;
machine_mode mode = TYPE_MODE (char_type_node);
rtx addr = expand_expr (address, NULL_RTX, VOIDmode, EXPAND_NORMAL);
rtx mem = gen_rtx_MEM (BLKmode, addr);
rtx value = gen_int_mode (16, mode);
for (i = FIRST_ALTIVEC_REGNO; i < LAST_ALTIVEC_REGNO+1; i++)
{
int column = DWARF_REG_TO_UNWIND_COLUMN
(DWARF2_FRAME_REG_OUT (DWARF_FRAME_REGNUM (i), true));
HOST_WIDE_INT offset = column * GET_MODE_SIZE (mode);
emit_move_insn (adjust_address (mem, mode, offset), value);
}
}
}
unsigned int
rs6000_dbx_register_number (unsigned int regno, unsigned int format)
{
if ((format == 0 && write_symbols != DWARF2_DEBUG) || format == 2)
return regno;
#ifdef RS6000_USE_DWARF_NUMBERING
if (regno <= 63)
return regno;
if (regno == LR_REGNO)
return 108;
if (regno == CTR_REGNO)
return 109;
if (format == 1 && regno == CR2_REGNO)
return 64;
if (CR_REGNO_P (regno))
return regno - CR0_REGNO + 86;
if (regno == CA_REGNO)
return 101;  
if (ALTIVEC_REGNO_P (regno))
return regno - FIRST_ALTIVEC_REGNO + 1124;
if (regno == VRSAVE_REGNO)
return 356;
if (regno == VSCR_REGNO)
return 67;
#endif
return regno;
}
static scalar_int_mode
rs6000_eh_return_filter_mode (void)
{
return TARGET_32BIT ? SImode : word_mode;
}
static bool
rs6000_scalar_mode_supported_p (scalar_mode mode)
{
if (TARGET_32BIT && mode == TImode)
return false;
if (DECIMAL_FLOAT_MODE_P (mode))
return default_decimal_float_supported_p ();
else if (TARGET_FLOAT128_TYPE && (mode == KFmode || mode == IFmode))
return true;
else
return default_scalar_mode_supported_p (mode);
}
static bool
rs6000_vector_mode_supported_p (machine_mode mode)
{
if (TARGET_PAIRED_FLOAT && PAIRED_VECTOR_MODE (mode))
return true;
else if (VECTOR_MEM_ALTIVEC_OR_VSX_P (mode) && !FLOAT128_IEEE_P (mode))
return true;
else
return false;
}
static opt_scalar_float_mode
rs6000_floatn_mode (int n, bool extended)
{
if (extended)
{
switch (n)
{
case 32:
return DFmode;
case 64:
if (TARGET_FLOAT128_TYPE)
return (FLOAT128_IEEE_P (TFmode)) ? TFmode : KFmode;
else
return opt_scalar_float_mode ();
case 128:
return opt_scalar_float_mode ();
default:
gcc_unreachable ();
}
}
else
{
switch (n)
{
case 32:
return SFmode;
case 64:
return DFmode;
case 128:
if (TARGET_FLOAT128_TYPE)
return (FLOAT128_IEEE_P (TFmode)) ? TFmode : KFmode;
else
return opt_scalar_float_mode ();
default:
return opt_scalar_float_mode ();
}
}
}
static machine_mode
rs6000_c_mode_for_suffix (char suffix)
{
if (TARGET_FLOAT128_TYPE)
{
if (suffix == 'q' || suffix == 'Q')
return (FLOAT128_IEEE_P (TFmode)) ? TFmode : KFmode;
}
return VOIDmode;
}
static const char *
invalid_arg_for_unprototyped_fn (const_tree typelist, const_tree funcdecl, const_tree val)
{
return (!rs6000_darwin64_abi
&& typelist == 0
&& TREE_CODE (TREE_TYPE (val)) == VECTOR_TYPE
&& (funcdecl == NULL_TREE
|| (TREE_CODE (funcdecl) == FUNCTION_DECL
&& DECL_BUILT_IN_CLASS (funcdecl) != BUILT_IN_MD)))
? N_("AltiVec argument passed to unprototyped function")
: NULL;
}
static tree ATTRIBUTE_UNUSED
rs6000_stack_protect_fail (void)
{
return (DEFAULT_ABI == ABI_V4 && TARGET_SECURE_PLT && flag_pic)
? default_hidden_stack_protect_fail ()
: default_external_stack_protect_fail ();
}
#if TARGET_ELF
static unsigned HOST_WIDE_INT
rs6000_asan_shadow_offset (void)
{
return (unsigned HOST_WIDE_INT) 1 << (TARGET_64BIT ? 41 : 29);
}
#endif

struct rs6000_opt_mask {
const char *name;		
HOST_WIDE_INT mask;		
bool invert;			
bool valid_target;		
};
static struct rs6000_opt_mask const rs6000_opt_masks[] =
{
{ "altivec",			OPTION_MASK_ALTIVEC,		false, true  },
{ "cmpb",			OPTION_MASK_CMPB,		false, true  },
{ "crypto",			OPTION_MASK_CRYPTO,		false, true  },
{ "direct-move",		OPTION_MASK_DIRECT_MOVE,	false, true  },
{ "dlmzb",			OPTION_MASK_DLMZB,		false, true  },
{ "efficient-unaligned-vsx",	OPTION_MASK_EFFICIENT_UNALIGNED_VSX,
false, true  },
{ "float128",			OPTION_MASK_FLOAT128_KEYWORD,	false, true  },
{ "float128-hardware",	OPTION_MASK_FLOAT128_HW,	false, true  },
{ "fprnd",			OPTION_MASK_FPRND,		false, true  },
{ "hard-dfp",			OPTION_MASK_DFP,		false, true  },
{ "htm",			OPTION_MASK_HTM,		false, true  },
{ "isel",			OPTION_MASK_ISEL,		false, true  },
{ "mfcrf",			OPTION_MASK_MFCRF,		false, true  },
{ "mfpgpr",			OPTION_MASK_MFPGPR,		false, true  },
{ "modulo",			OPTION_MASK_MODULO,		false, true  },
{ "mulhw",			OPTION_MASK_MULHW,		false, true  },
{ "multiple",			OPTION_MASK_MULTIPLE,		false, true  },
{ "popcntb",			OPTION_MASK_POPCNTB,		false, true  },
{ "popcntd",			OPTION_MASK_POPCNTD,		false, true  },
{ "power8-fusion",		OPTION_MASK_P8_FUSION,		false, true  },
{ "power8-fusion-sign",	OPTION_MASK_P8_FUSION_SIGN,	false, true  },
{ "power8-vector",		OPTION_MASK_P8_VECTOR,		false, true  },
{ "power9-fusion",		OPTION_MASK_P9_FUSION,		false, true  },
{ "power9-minmax",		OPTION_MASK_P9_MINMAX,		false, true  },
{ "power9-misc",		OPTION_MASK_P9_MISC,		false, true  },
{ "power9-vector",		OPTION_MASK_P9_VECTOR,		false, true  },
{ "powerpc-gfxopt",		OPTION_MASK_PPC_GFXOPT,		false, true  },
{ "powerpc-gpopt",		OPTION_MASK_PPC_GPOPT,		false, true  },
{ "quad-memory",		OPTION_MASK_QUAD_MEMORY,	false, true  },
{ "quad-memory-atomic",	OPTION_MASK_QUAD_MEMORY_ATOMIC,	false, true  },
{ "recip-precision",		OPTION_MASK_RECIP_PRECISION,	false, true  },
{ "save-toc-indirect",	OPTION_MASK_SAVE_TOC_INDIRECT,	false, true  },
{ "string",			0,				false, true  },
{ "toc-fusion",		OPTION_MASK_TOC_FUSION,		false, true  },
{ "update",			OPTION_MASK_NO_UPDATE,		true , true  },
{ "vsx",			OPTION_MASK_VSX,		false, true  },
#ifdef OPTION_MASK_64BIT
#if TARGET_AIX_OS
{ "aix64",			OPTION_MASK_64BIT,		false, false },
{ "aix32",			OPTION_MASK_64BIT,		true,  false },
#else
{ "64",			OPTION_MASK_64BIT,		false, false },
{ "32",			OPTION_MASK_64BIT,		true,  false },
#endif
#endif
#ifdef OPTION_MASK_EABI
{ "eabi",			OPTION_MASK_EABI,		false, false },
#endif
#ifdef OPTION_MASK_LITTLE_ENDIAN
{ "little",			OPTION_MASK_LITTLE_ENDIAN,	false, false },
{ "big",			OPTION_MASK_LITTLE_ENDIAN,	true,  false },
#endif
#ifdef OPTION_MASK_RELOCATABLE
{ "relocatable",		OPTION_MASK_RELOCATABLE,	false, false },
#endif
#ifdef OPTION_MASK_STRICT_ALIGN
{ "strict-align",		OPTION_MASK_STRICT_ALIGN,	false, false },
#endif
{ "soft-float",		OPTION_MASK_SOFT_FLOAT,		false, false },
{ "string",			0,				false, false },
};
static struct rs6000_opt_mask const rs6000_builtin_mask_names[] =
{
{ "altivec",		 RS6000_BTM_ALTIVEC,	false, false },
{ "vsx",		 RS6000_BTM_VSX,	false, false },
{ "paired",		 RS6000_BTM_PAIRED,	false, false },
{ "fre",		 RS6000_BTM_FRE,	false, false },
{ "fres",		 RS6000_BTM_FRES,	false, false },
{ "frsqrte",		 RS6000_BTM_FRSQRTE,	false, false },
{ "frsqrtes",		 RS6000_BTM_FRSQRTES,	false, false },
{ "popcntd",		 RS6000_BTM_POPCNTD,	false, false },
{ "cell",		 RS6000_BTM_CELL,	false, false },
{ "power8-vector",	 RS6000_BTM_P8_VECTOR,	false, false },
{ "power9-vector",	 RS6000_BTM_P9_VECTOR,	false, false },
{ "power9-misc",	 RS6000_BTM_P9_MISC,	false, false },
{ "crypto",		 RS6000_BTM_CRYPTO,	false, false },
{ "htm",		 RS6000_BTM_HTM,	false, false },
{ "hard-dfp",		 RS6000_BTM_DFP,	false, false },
{ "hard-float",	 RS6000_BTM_HARD_FLOAT,	false, false },
{ "long-double-128",	 RS6000_BTM_LDBL128,	false, false },
{ "powerpc64",	 RS6000_BTM_POWERPC64,  false, false },
{ "float128",		 RS6000_BTM_FLOAT128,   false, false },
{ "float128-hw",	 RS6000_BTM_FLOAT128_HW,false, false },
};
struct rs6000_opt_var {
const char *name;		
size_t global_offset;		
size_t target_offset;		
};
static struct rs6000_opt_var const rs6000_opt_vars[] =
{
{ "friz",
offsetof (struct gcc_options, x_TARGET_FRIZ),
offsetof (struct cl_target_option, x_TARGET_FRIZ), },
{ "avoid-indexed-addresses",
offsetof (struct gcc_options, x_TARGET_AVOID_XFORM),
offsetof (struct cl_target_option, x_TARGET_AVOID_XFORM) },
{ "paired",
offsetof (struct gcc_options, x_rs6000_paired_float),
offsetof (struct cl_target_option, x_rs6000_paired_float), },
{ "longcall",
offsetof (struct gcc_options, x_rs6000_default_long_calls),
offsetof (struct cl_target_option, x_rs6000_default_long_calls), },
{ "optimize-swaps",
offsetof (struct gcc_options, x_rs6000_optimize_swaps),
offsetof (struct cl_target_option, x_rs6000_optimize_swaps), },
{ "allow-movmisalign",
offsetof (struct gcc_options, x_TARGET_ALLOW_MOVMISALIGN),
offsetof (struct cl_target_option, x_TARGET_ALLOW_MOVMISALIGN), },
{ "sched-groups",
offsetof (struct gcc_options, x_TARGET_SCHED_GROUPS),
offsetof (struct cl_target_option, x_TARGET_SCHED_GROUPS), },
{ "always-hint",
offsetof (struct gcc_options, x_TARGET_ALWAYS_HINT),
offsetof (struct cl_target_option, x_TARGET_ALWAYS_HINT), },
{ "align-branch-targets",
offsetof (struct gcc_options, x_TARGET_ALIGN_BRANCH_TARGETS),
offsetof (struct cl_target_option, x_TARGET_ALIGN_BRANCH_TARGETS), },
{ "tls-markers",
offsetof (struct gcc_options, x_tls_markers),
offsetof (struct cl_target_option, x_tls_markers), },
{ "sched-prolog",
offsetof (struct gcc_options, x_TARGET_SCHED_PROLOG),
offsetof (struct cl_target_option, x_TARGET_SCHED_PROLOG), },
{ "sched-epilog",
offsetof (struct gcc_options, x_TARGET_SCHED_PROLOG),
offsetof (struct cl_target_option, x_TARGET_SCHED_PROLOG), },
{ "speculate-indirect-jumps",
offsetof (struct gcc_options, x_rs6000_speculate_indirect_jumps),
offsetof (struct cl_target_option, x_rs6000_speculate_indirect_jumps), },
};
static bool
rs6000_inner_target_options (tree args, bool attr_p)
{
bool ret = true;
if (args == NULL_TREE)
;
else if (TREE_CODE (args) == STRING_CST)
{
char *p = ASTRDUP (TREE_STRING_POINTER (args));
char *q;
while ((q = strtok (p, ",")) != NULL)
{
bool error_p = false;
bool not_valid_p = false;
const char *cpu_opt = NULL;
p = NULL;
if (strncmp (q, "cpu=", 4) == 0)
{
int cpu_index = rs6000_cpu_name_lookup (q+4);
if (cpu_index >= 0)
rs6000_cpu_index = cpu_index;
else
{
error_p = true;
cpu_opt = q+4;
}
}
else if (strncmp (q, "tune=", 5) == 0)
{
int tune_index = rs6000_cpu_name_lookup (q+5);
if (tune_index >= 0)
rs6000_tune_index = tune_index;
else
{
error_p = true;
cpu_opt = q+5;
}
}
else
{
size_t i;
bool invert = false;
char *r = q;
error_p = true;
if (strncmp (r, "no-", 3) == 0)
{
invert = true;
r += 3;
}
for (i = 0; i < ARRAY_SIZE (rs6000_opt_masks); i++)
if (strcmp (r, rs6000_opt_masks[i].name) == 0)
{
HOST_WIDE_INT mask = rs6000_opt_masks[i].mask;
if (!rs6000_opt_masks[i].valid_target)
not_valid_p = true;
else
{
error_p = false;
rs6000_isa_flags_explicit |= mask;
if (!invert)
{
if (mask == OPTION_MASK_VSX)
{
mask |= OPTION_MASK_ALTIVEC;
TARGET_AVOID_XFORM = 0;
}
}
if (rs6000_opt_masks[i].invert)
invert = !invert;
if (invert)
rs6000_isa_flags &= ~mask;
else
rs6000_isa_flags |= mask;
}
break;
}
if (error_p && !not_valid_p)
{
for (i = 0; i < ARRAY_SIZE (rs6000_opt_vars); i++)
if (strcmp (r, rs6000_opt_vars[i].name) == 0)
{
size_t j = rs6000_opt_vars[i].global_offset;
*((int *) ((char *)&global_options + j)) = !invert;
error_p = false;
not_valid_p = false;
break;
}
}
}
if (error_p)
{
const char *eprefix, *esuffix;
ret = false;
if (attr_p)
{
eprefix = "__attribute__((__target__(";
esuffix = ")))";
}
else
{
eprefix = "#pragma GCC target ";
esuffix = "";
}
if (cpu_opt)
error ("invalid cpu %qs for %s%qs%s", cpu_opt, eprefix,
q, esuffix);
else if (not_valid_p)
error ("%s%qs%s is not allowed", eprefix, q, esuffix);
else
error ("%s%qs%s is invalid", eprefix, q, esuffix);
}
}
}
else if (TREE_CODE (args) == TREE_LIST)
{
do
{
tree value = TREE_VALUE (args);
if (value)
{
bool ret2 = rs6000_inner_target_options (value, attr_p);
if (!ret2)
ret = false;
}
args = TREE_CHAIN (args);
}
while (args != NULL_TREE);
}
else
{
error ("attribute %<target%> argument not a string");
return false;
}
return ret;
}
static void
rs6000_debug_target_options (tree args, const char *prefix)
{
if (args == NULL_TREE)
fprintf (stderr, "%s<NULL>", prefix);
else if (TREE_CODE (args) == STRING_CST)
{
char *p = ASTRDUP (TREE_STRING_POINTER (args));
char *q;
while ((q = strtok (p, ",")) != NULL)
{
p = NULL;
fprintf (stderr, "%s\"%s\"", prefix, q);
prefix = ", ";
}
}
else if (TREE_CODE (args) == TREE_LIST)
{
do
{
tree value = TREE_VALUE (args);
if (value)
{
rs6000_debug_target_options (value, prefix);
prefix = ", ";
}
args = TREE_CHAIN (args);
}
while (args != NULL_TREE);
}
else
gcc_unreachable ();
return;
}

static bool
rs6000_valid_attribute_p (tree fndecl,
tree ARG_UNUSED (name),
tree args,
int flags)
{
struct cl_target_option cur_target;
bool ret;
tree old_optimize;
tree new_target, new_optimize;
tree func_optimize;
gcc_assert ((fndecl != NULL_TREE) && (args != NULL_TREE));
if (TARGET_DEBUG_TARGET)
{
tree tname = DECL_NAME (fndecl);
fprintf (stderr, "\n==================== rs6000_valid_attribute_p:\n");
if (tname)
fprintf (stderr, "function: %.*s\n",
(int) IDENTIFIER_LENGTH (tname),
IDENTIFIER_POINTER (tname));
else
fprintf (stderr, "function: unknown\n");
fprintf (stderr, "args:");
rs6000_debug_target_options (args, " ");
fprintf (stderr, "\n");
if (flags)
fprintf (stderr, "flags: 0x%x\n", flags);
fprintf (stderr, "--------------------\n");
}
if (TREE_VALUE (args)
&& TREE_CODE (TREE_VALUE (args)) == STRING_CST
&& TREE_CHAIN (args) == NULL_TREE
&& strcmp (TREE_STRING_POINTER (TREE_VALUE (args)), "default") == 0)
return true;
old_optimize = build_optimization_node (&global_options);
func_optimize = DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl);
if (func_optimize && func_optimize != old_optimize)
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (func_optimize));
cl_target_option_save (&cur_target, &global_options);
rs6000_cpu_index = rs6000_tune_index = -1;
ret = rs6000_inner_target_options (args, true);
if (ret)
{
ret = rs6000_option_override_internal (false);
new_target = build_target_option_node (&global_options);
}
else
new_target = NULL;
new_optimize = build_optimization_node (&global_options);
if (!new_target)
ret = false;
else if (fndecl)
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

bool
rs6000_pragma_target_parse (tree args, tree pop_target)
{
tree prev_tree = build_target_option_node (&global_options);
tree cur_tree;
struct cl_target_option *prev_opt, *cur_opt;
HOST_WIDE_INT prev_flags, cur_flags, diff_flags;
HOST_WIDE_INT prev_bumask, cur_bumask, diff_bumask;
if (TARGET_DEBUG_TARGET)
{
fprintf (stderr, "\n==================== rs6000_pragma_target_parse\n");
fprintf (stderr, "args:");
rs6000_debug_target_options (args, " ");
fprintf (stderr, "\n");
if (pop_target)
{
fprintf (stderr, "pop_target:\n");
debug_tree (pop_target);
}
else
fprintf (stderr, "pop_target: <NULL>\n");
fprintf (stderr, "--------------------\n");
}
if (! args)
{
cur_tree = ((pop_target)
? pop_target
: target_option_default_node);
cl_target_option_restore (&global_options,
TREE_TARGET_OPTION (cur_tree));
}
else
{
rs6000_cpu_index = rs6000_tune_index = -1;
if (!rs6000_inner_target_options (args, false)
|| !rs6000_option_override_internal (false)
|| (cur_tree = build_target_option_node (&global_options))
== NULL_TREE)
{
if (TARGET_DEBUG_BUILTIN || TARGET_DEBUG_TARGET)
fprintf (stderr, "invalid pragma\n");
return false;
}
}
target_option_current_node = cur_tree;
rs6000_activate_target_options (target_option_current_node);
if (rs6000_target_modify_macros_ptr)
{
prev_opt    = TREE_TARGET_OPTION (prev_tree);
prev_bumask = prev_opt->x_rs6000_builtin_mask;
prev_flags  = prev_opt->x_rs6000_isa_flags;
cur_opt     = TREE_TARGET_OPTION (cur_tree);
cur_flags   = cur_opt->x_rs6000_isa_flags;
cur_bumask  = cur_opt->x_rs6000_builtin_mask;
diff_bumask = (prev_bumask ^ cur_bumask);
diff_flags  = (prev_flags ^ cur_flags);
if ((diff_flags != 0) || (diff_bumask != 0))
{
rs6000_target_modify_macros_ptr (false,
prev_flags & diff_flags,
prev_bumask & diff_bumask);
rs6000_target_modify_macros_ptr (true,
cur_flags & diff_flags,
cur_bumask & diff_bumask);
}
}
return true;
}

static GTY(()) tree rs6000_previous_fndecl;
void
rs6000_activate_target_options (tree new_tree)
{
cl_target_option_restore (&global_options, TREE_TARGET_OPTION (new_tree));
if (TREE_TARGET_GLOBALS (new_tree))
restore_target_globals (TREE_TARGET_GLOBALS (new_tree));
else if (new_tree == target_option_default_node)
restore_target_globals (&default_target_globals);
else
TREE_TARGET_GLOBALS (new_tree) = save_target_globals_default_opts ();
rs6000_previous_fndecl = NULL_TREE;
}
static void
rs6000_set_current_function (tree fndecl)
{
if (TARGET_DEBUG_TARGET)
{
fprintf (stderr, "\n==================== rs6000_set_current_function");
if (fndecl)
fprintf (stderr, ", fndecl %s (%p)",
(DECL_NAME (fndecl)
? IDENTIFIER_POINTER (DECL_NAME (fndecl))
: "<unknown>"), (void *)fndecl);
if (rs6000_previous_fndecl)
fprintf (stderr, ", prev_fndecl (%p)", (void *)rs6000_previous_fndecl);
fprintf (stderr, "\n");
}
if (fndecl == rs6000_previous_fndecl)
return;
tree old_tree;
if (rs6000_previous_fndecl == NULL_TREE)
old_tree = target_option_current_node;
else if (DECL_FUNCTION_SPECIFIC_TARGET (rs6000_previous_fndecl))
old_tree = DECL_FUNCTION_SPECIFIC_TARGET (rs6000_previous_fndecl);
else
old_tree = target_option_default_node;
tree new_tree;
if (fndecl == NULL_TREE)
{
if (old_tree != target_option_current_node)
new_tree = target_option_current_node;
else
new_tree = NULL_TREE;
}
else
{
new_tree = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
if (new_tree == NULL_TREE)
new_tree = target_option_default_node;
}
if (TARGET_DEBUG_TARGET)
{
if (new_tree)
{
fprintf (stderr, "\nnew fndecl target specific options:\n");
debug_tree (new_tree);
}
if (old_tree)
{
fprintf (stderr, "\nold fndecl target specific options:\n");
debug_tree (old_tree);
}
if (old_tree != NULL_TREE || new_tree != NULL_TREE)
fprintf (stderr, "--------------------\n");
}
if (new_tree && old_tree != new_tree)
rs6000_activate_target_options (new_tree);
if (fndecl)
rs6000_previous_fndecl = fndecl;
}

static void
rs6000_function_specific_save (struct cl_target_option *ptr,
struct gcc_options *opts)
{
ptr->x_rs6000_isa_flags = opts->x_rs6000_isa_flags;
ptr->x_rs6000_isa_flags_explicit = opts->x_rs6000_isa_flags_explicit;
}
static void
rs6000_function_specific_restore (struct gcc_options *opts,
struct cl_target_option *ptr)
{
opts->x_rs6000_isa_flags = ptr->x_rs6000_isa_flags;
opts->x_rs6000_isa_flags_explicit = ptr->x_rs6000_isa_flags_explicit;
(void) rs6000_option_override_internal (false);
}
static void
rs6000_function_specific_print (FILE *file, int indent,
struct cl_target_option *ptr)
{
rs6000_print_isa_options (file, indent, "Isa options set",
ptr->x_rs6000_isa_flags);
rs6000_print_isa_options (file, indent, "Isa options explicit",
ptr->x_rs6000_isa_flags_explicit);
}
static void
rs6000_print_options_internal (FILE *file,
int indent,
const char *string,
HOST_WIDE_INT flags,
const char *prefix,
const struct rs6000_opt_mask *opts,
size_t num_elements)
{
size_t i;
size_t start_column = 0;
size_t cur_column;
size_t max_column = 120;
size_t prefix_len = strlen (prefix);
size_t comma_len = 0;
const char *comma = "";
if (indent)
start_column += fprintf (file, "%*s", indent, "");
if (!flags)
{
fprintf (stderr, DEBUG_FMT_S, string, "<none>");
return;
}
start_column += fprintf (stderr, DEBUG_FMT_WX, string, flags);
cur_column = start_column;
for (i = 0; i < num_elements; i++)
{
bool invert = opts[i].invert;
const char *name = opts[i].name;
const char *no_str = "";
HOST_WIDE_INT mask = opts[i].mask;
size_t len = comma_len + prefix_len + strlen (name);
if (!invert)
{
if ((flags & mask) == 0)
{
no_str = "no-";
len += sizeof ("no-") - 1;
}
flags &= ~mask;
}
else
{
if ((flags & mask) != 0)
{
no_str = "no-";
len += sizeof ("no-") - 1;
}
flags |= mask;
}
cur_column += len;
if (cur_column > max_column)
{
fprintf (stderr, ", \\\n%*s", (int)start_column, "");
cur_column = start_column + len;
comma = "";
}
fprintf (file, "%s%s%s%s", comma, prefix, no_str, name);
comma = ", ";
comma_len = sizeof (", ") - 1;
}
fputs ("\n", file);
}
static void
rs6000_print_isa_options (FILE *file, int indent, const char *string,
HOST_WIDE_INT flags)
{
rs6000_print_options_internal (file, indent, string, flags, "-m",
&rs6000_opt_masks[0],
ARRAY_SIZE (rs6000_opt_masks));
}
static void
rs6000_print_builtin_options (FILE *file, int indent, const char *string,
HOST_WIDE_INT flags)
{
rs6000_print_options_internal (file, indent, string, flags, "",
&rs6000_builtin_mask_names[0],
ARRAY_SIZE (rs6000_builtin_mask_names));
}
static HOST_WIDE_INT
rs6000_disable_incompatible_switches (void)
{
HOST_WIDE_INT ignore_masks = rs6000_isa_flags_explicit;
size_t i, j;
static const struct {
const HOST_WIDE_INT no_flag;	
const HOST_WIDE_INT dep_flags;	
const char *const name;		
} flags[] = {
{ OPTION_MASK_P9_VECTOR,	OTHER_P9_VECTOR_MASKS,	"power9-vector"	},
{ OPTION_MASK_P8_VECTOR,	OTHER_P8_VECTOR_MASKS,	"power8-vector"	},
{ OPTION_MASK_VSX,		OTHER_VSX_VECTOR_MASKS,	"vsx"		},
};
for (i = 0; i < ARRAY_SIZE (flags); i++)
{
HOST_WIDE_INT no_flag = flags[i].no_flag;
if ((rs6000_isa_flags & no_flag) == 0
&& (rs6000_isa_flags_explicit & no_flag) != 0)
{
HOST_WIDE_INT dep_flags = flags[i].dep_flags;
HOST_WIDE_INT set_flags = (rs6000_isa_flags_explicit
& rs6000_isa_flags
& dep_flags);
if (set_flags)
{
for (j = 0; j < ARRAY_SIZE (rs6000_opt_masks); j++)
if ((set_flags & rs6000_opt_masks[j].mask) != 0)
{
set_flags &= ~rs6000_opt_masks[j].mask;
error ("%<-mno-%s%> turns off %<-m%s%>",
flags[i].name,
rs6000_opt_masks[j].name);
}
gcc_assert (!set_flags);
}
rs6000_isa_flags &= ~dep_flags;
ignore_masks |= no_flag | dep_flags;
}
}
return ignore_masks;
}

static const char *
get_decl_name (tree fn)
{
tree name;
if (!fn)
return "<null>";
name = DECL_NAME (fn);
if (!name)
return "<no-name>";
return IDENTIFIER_POINTER (name);
}
static int
rs6000_clone_priority (tree fndecl)
{
tree fn_opts = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
HOST_WIDE_INT isa_masks;
int ret = CLONE_DEFAULT;
tree attrs = lookup_attribute ("target", DECL_ATTRIBUTES (fndecl));
const char *attrs_str = NULL;
attrs = TREE_VALUE (TREE_VALUE (attrs));
attrs_str = TREE_STRING_POINTER (attrs);
if (strcmp (attrs_str, "default") != 0)
{
if (fn_opts == NULL_TREE)
fn_opts = target_option_default_node;
if (!fn_opts || !TREE_TARGET_OPTION (fn_opts))
isa_masks = rs6000_isa_flags;
else
isa_masks = TREE_TARGET_OPTION (fn_opts)->x_rs6000_isa_flags;
for (ret = CLONE_MAX - 1; ret != 0; ret--)
if ((rs6000_clone_map[ret].isa_mask & isa_masks) != 0)
break;
}
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_get_function_version_priority (%s) => %d\n",
get_decl_name (fndecl), ret);
return ret;
}
static int
rs6000_compare_version_priority (tree decl1, tree decl2)
{
int priority1 = rs6000_clone_priority (decl1);
int priority2 = rs6000_clone_priority (decl2);
int ret = priority1 - priority2;
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_compare_version_priority (%s, %s) => %d\n",
get_decl_name (decl1), get_decl_name (decl2), ret);
return ret;
}
static tree
rs6000_get_function_versions_dispatcher (void *decl)
{
tree fn = (tree) decl;
struct cgraph_node *node = NULL;
struct cgraph_node *default_node = NULL;
struct cgraph_function_version_info *node_v = NULL;
struct cgraph_function_version_info *first_v = NULL;
tree dispatch_decl = NULL;
struct cgraph_function_version_info *default_version_info = NULL;
gcc_assert (fn != NULL && DECL_FUNCTION_VERSIONED (fn));
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_get_function_versions_dispatcher (%s)\n",
get_decl_name (fn));
node = cgraph_node::get (fn);
gcc_assert (node != NULL);
node_v = node->function_version ();
gcc_assert (node_v != NULL);
if (node_v->dispatcher_resolver != NULL)
return node_v->dispatcher_resolver;
first_v = node_v;
while (first_v->prev != NULL)
first_v = first_v->prev;
default_version_info = first_v;
while (default_version_info != NULL)
{
const tree decl2 = default_version_info->this_node->decl;
if (is_function_default_version (decl2))
break;
default_version_info = default_version_info->next;
}
if (default_version_info == NULL)
return NULL;
if (first_v != default_version_info)
{
default_version_info->prev->next = default_version_info->next;
if (default_version_info->next)
default_version_info->next->prev = default_version_info->prev;
first_v->prev = default_version_info;
default_version_info->next = first_v;
default_version_info->prev = NULL;
}
default_node = default_version_info->this_node;
#ifndef TARGET_LIBC_PROVIDES_HWCAP_IN_TCB
error_at (DECL_SOURCE_LOCATION (default_node->decl),
"target_clones attribute needs GLIBC (2.23 and newer) that "
"exports hardware capability bits");
#else
if (targetm.has_ifunc_p ())
{
struct cgraph_function_version_info *it_v = NULL;
struct cgraph_node *dispatcher_node = NULL;
struct cgraph_function_version_info *dispatcher_version_info = NULL;
dispatch_decl = make_dispatcher_decl (default_node->decl);
dispatcher_node = cgraph_node::get_create (dispatch_decl);
gcc_assert (dispatcher_node != NULL);
dispatcher_node->dispatcher_function = 1;
dispatcher_version_info
= dispatcher_node->insert_new_function_version ();
dispatcher_version_info->next = default_version_info;
dispatcher_node->definition = 1;
it_v = default_version_info;
while (it_v != NULL)
{
it_v->dispatcher_resolver = dispatch_decl;
it_v = it_v->next;
}
}
else
{
error_at (DECL_SOURCE_LOCATION (default_node->decl),
"multiversioning needs ifunc which is not supported "
"on this target");
}
#endif
return dispatch_decl;
}
static tree
make_resolver_func (const tree default_decl,
const tree dispatch_decl,
basic_block *empty_bb)
{
tree decl_name = clone_function_name (default_decl, "resolver");
const char *resolver_name = IDENTIFIER_POINTER (decl_name);
tree type = build_function_type_list (ptr_type_node, NULL_TREE);
tree decl = build_fn_decl (resolver_name, type);
SET_DECL_ASSEMBLER_NAME (decl, decl_name);
DECL_NAME (decl) = decl_name;
TREE_USED (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
DECL_IGNORED_P (decl) = 0;
TREE_PUBLIC (decl) = 0;
DECL_UNINLINABLE (decl) = 1;
DECL_EXTERNAL (decl) = 0;
DECL_EXTERNAL (dispatch_decl) = 0;
DECL_CONTEXT (decl) = NULL_TREE;
DECL_INITIAL (decl) = make_node (BLOCK);
DECL_STATIC_CONSTRUCTOR (decl) = 0;
tree t = build_decl (UNKNOWN_LOCATION, RESULT_DECL, NULL_TREE, ptr_type_node);
DECL_ARTIFICIAL (t) = 1;
DECL_IGNORED_P (t) = 1;
DECL_RESULT (decl) = t;
gimplify_function_tree (decl);
push_cfun (DECL_STRUCT_FUNCTION (decl));
*empty_bb = init_lowered_empty_function (decl, false,
profile_count::uninitialized ());
cgraph_node::add_new_function (decl, true);
symtab->call_cgraph_insertion_hooks (cgraph_node::get_create (decl));
pop_cfun ();
DECL_ATTRIBUTES (dispatch_decl)
= make_attribute ("ifunc", resolver_name, DECL_ATTRIBUTES (dispatch_decl));
cgraph_node::create_same_body_alias (dispatch_decl, decl);
return decl;
}
static basic_block
add_condition_to_bb (tree function_decl, tree version_decl,
int clone_isa, basic_block new_bb)
{
push_cfun (DECL_STRUCT_FUNCTION (function_decl));
gcc_assert (new_bb != NULL);
gimple_seq gseq = bb_seq (new_bb);
tree convert_expr = build1 (CONVERT_EXPR, ptr_type_node,
build_fold_addr_expr (version_decl));
tree result_var = create_tmp_var (ptr_type_node);
gimple *convert_stmt = gimple_build_assign (result_var, convert_expr);
gimple *return_stmt = gimple_build_return (result_var);
if (clone_isa == CLONE_DEFAULT)
{
gimple_seq_add_stmt (&gseq, convert_stmt);
gimple_seq_add_stmt (&gseq, return_stmt);
set_bb_seq (new_bb, gseq);
gimple_set_bb (convert_stmt, new_bb);
gimple_set_bb (return_stmt, new_bb);
pop_cfun ();
return new_bb;
}
tree bool_zero = build_int_cst (bool_int_type_node, 0);
tree cond_var = create_tmp_var (bool_int_type_node);
tree predicate_decl = rs6000_builtin_decls [(int) RS6000_BUILTIN_CPU_SUPPORTS];
const char *arg_str = rs6000_clone_map[clone_isa].name;
tree predicate_arg = build_string_literal (strlen (arg_str) + 1, arg_str);
gimple *call_cond_stmt = gimple_build_call (predicate_decl, 1, predicate_arg);
gimple_call_set_lhs (call_cond_stmt, cond_var);
gimple_set_block (call_cond_stmt, DECL_INITIAL (function_decl));
gimple_set_bb (call_cond_stmt, new_bb);
gimple_seq_add_stmt (&gseq, call_cond_stmt);
gimple *if_else_stmt = gimple_build_cond (NE_EXPR, cond_var, bool_zero,
NULL_TREE, NULL_TREE);
gimple_set_block (if_else_stmt, DECL_INITIAL (function_decl));
gimple_set_bb (if_else_stmt, new_bb);
gimple_seq_add_stmt (&gseq, if_else_stmt);
gimple_seq_add_stmt (&gseq, convert_stmt);
gimple_seq_add_stmt (&gseq, return_stmt);
set_bb_seq (new_bb, gseq);
basic_block bb1 = new_bb;
edge e12 = split_block (bb1, if_else_stmt);
basic_block bb2 = e12->dest;
e12->flags &= ~EDGE_FALLTHRU;
e12->flags |= EDGE_TRUE_VALUE;
edge e23 = split_block (bb2, return_stmt);
gimple_set_bb (convert_stmt, bb2);
gimple_set_bb (return_stmt, bb2);
basic_block bb3 = e23->dest;
make_edge (bb1, bb3, EDGE_FALSE_VALUE);
remove_edge (e23);
make_edge (bb2, EXIT_BLOCK_PTR_FOR_FN (cfun), 0);
pop_cfun ();
return bb3;
}
static int
dispatch_function_versions (tree dispatch_decl,
void *fndecls_p,
basic_block *empty_bb)
{
int ix;
tree ele;
vec<tree> *fndecls;
tree clones[CLONE_MAX];
if (TARGET_DEBUG_TARGET)
fputs ("dispatch_function_versions, top\n", stderr);
gcc_assert (dispatch_decl != NULL
&& fndecls_p != NULL
&& empty_bb != NULL);
fndecls = static_cast<vec<tree> *> (fndecls_p);
gcc_assert (fndecls->length () >= 2);
memset ((void *) clones, '\0', sizeof (clones));
clones[CLONE_DEFAULT] = (*fndecls)[0];
for (ix = 1; fndecls->iterate (ix, &ele); ++ix)
{
int priority = rs6000_clone_priority (ele);
if (!clones[priority])
clones[priority] = ele;
}
for (ix = CLONE_MAX - 1; ix >= 0; ix--)
if (clones[ix])
{
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "dispatch_function_versions, clone %d, %s\n",
ix, get_decl_name (clones[ix]));
*empty_bb = add_condition_to_bb (dispatch_decl, clones[ix], ix,
*empty_bb);
}
return 0;
}
static tree
rs6000_generate_version_dispatcher_body (void *node_p)
{
tree resolver;
basic_block empty_bb;
struct cgraph_node *node = (cgraph_node *) node_p;
struct cgraph_function_version_info *ninfo = node->function_version ();
if (ninfo->dispatcher_resolver)
return ninfo->dispatcher_resolver;
node->definition = false;
ninfo->dispatcher_resolver = resolver
= make_resolver_func (ninfo->next->this_node->decl, node->decl, &empty_bb);
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_get_function_versions_dispatcher, %s\n",
get_decl_name (resolver));
push_cfun (DECL_STRUCT_FUNCTION (resolver));
auto_vec<tree, 2> fn_ver_vec;
for (struct cgraph_function_version_info *vinfo = ninfo->next;
vinfo;
vinfo = vinfo->next)
{
struct cgraph_node *version = vinfo->this_node;
if (DECL_VINDEX (version->decl))
sorry ("Virtual function multiversioning not supported");
fn_ver_vec.safe_push (version->decl);
}
dispatch_function_versions (resolver, &fn_ver_vec, &empty_bb);
cgraph_edge::rebuild_edges ();
pop_cfun ();
return resolver;
}

static bool
rs6000_can_inline_p (tree caller, tree callee)
{
bool ret = false;
tree caller_tree = DECL_FUNCTION_SPECIFIC_TARGET (caller);
tree callee_tree = DECL_FUNCTION_SPECIFIC_TARGET (callee);
if (!callee_tree)
ret = true;
else if (!caller_tree)
ret = false;
else
{
struct cl_target_option *caller_opts = TREE_TARGET_OPTION (caller_tree);
struct cl_target_option *callee_opts = TREE_TARGET_OPTION (callee_tree);
if ((caller_opts->x_rs6000_isa_flags & callee_opts->x_rs6000_isa_flags)
== callee_opts->x_rs6000_isa_flags)
ret = true;
}
if (TARGET_DEBUG_TARGET)
fprintf (stderr, "rs6000_can_inline_p:, caller %s, callee %s, %s inline\n",
get_decl_name (caller), get_decl_name (callee),
(ret ? "can" : "cannot"));
return ret;
}

rtx
rs6000_allocate_stack_temp (machine_mode mode,
bool offsettable_p,
bool reg_reg_p)
{
rtx stack = assign_stack_temp (mode, GET_MODE_SIZE (mode));
rtx addr = XEXP (stack, 0);
int strict_p = reload_completed;
if (!legitimate_indirect_address_p (addr, strict_p))
{
if (offsettable_p
&& !rs6000_legitimate_offset_address_p (mode, addr, strict_p, true))
stack = replace_equiv_address (stack, copy_addr_to_reg (addr));
else if (reg_reg_p && !legitimate_indexed_address_p (addr, strict_p))
stack = replace_equiv_address (stack, copy_addr_to_reg (addr));
}
return stack;
}
rtx
rs6000_address_for_fpconvert (rtx x)
{
rtx addr;
gcc_assert (MEM_P (x));
addr = XEXP (x, 0);
if (can_create_pseudo_p ()
&& ! legitimate_indirect_address_p (addr, reload_completed)
&& ! legitimate_indexed_address_p (addr, reload_completed))
{
if (GET_CODE (addr) == PRE_INC || GET_CODE (addr) == PRE_DEC)
{
rtx reg = XEXP (addr, 0);
HOST_WIDE_INT size = GET_MODE_SIZE (GET_MODE (x));
rtx size_rtx = GEN_INT ((GET_CODE (addr) == PRE_DEC) ? -size : size);
gcc_assert (REG_P (reg));
emit_insn (gen_add3_insn (reg, reg, size_rtx));
addr = reg;
}
else if (GET_CODE (addr) == PRE_MODIFY)
{
rtx reg = XEXP (addr, 0);
rtx expr = XEXP (addr, 1);
gcc_assert (REG_P (reg));
gcc_assert (GET_CODE (expr) == PLUS);
emit_insn (gen_add3_insn (reg, XEXP (expr, 0), XEXP (expr, 1)));
addr = reg;
}
x = replace_equiv_address (x, copy_addr_to_reg (addr));
}
return x;
}
static bool
rs6000_legitimate_constant_p (machine_mode mode, rtx x)
{
if (TARGET_ELF && tls_referenced_p (x))
return false;
return ((GET_CODE (x) != CONST_DOUBLE && GET_CODE (x) != CONST_VECTOR)
|| GET_MODE (x) == VOIDmode
|| (TARGET_POWERPC64 && mode == DImode)
|| easy_fp_constant (x, mode)
|| easy_vector_constant (x, mode));
}

static bool
chain_already_loaded (rtx_insn *last)
{
for (; last != NULL; last = PREV_INSN (last))
{
if (NONJUMP_INSN_P (last))
{
rtx patt = PATTERN (last);
if (GET_CODE (patt) == SET)
{
rtx lhs = XEXP (patt, 0);
if (REG_P (lhs) && REGNO (lhs) == STATIC_CHAIN_REGNUM)
return true;
}
}
}
return false;
}
void
rs6000_call_aix (rtx value, rtx func_desc, rtx flag, rtx cookie)
{
const bool direct_call_p
= GET_CODE (func_desc) == SYMBOL_REF && SYMBOL_REF_FUNCTION_P (func_desc);
rtx toc_reg = gen_rtx_REG (Pmode, TOC_REGNUM);
rtx toc_load = NULL_RTX;
rtx toc_restore = NULL_RTX;
rtx func_addr;
rtx abi_reg = NULL_RTX;
rtx call[4];
int n_call;
rtx insn;
if (INTVAL (cookie) & CALL_LONG)
func_desc = rs6000_longcall_ref (func_desc);
if (GET_CODE (func_desc) != SYMBOL_REF
|| (DEFAULT_ABI == ABI_AIX && !SYMBOL_REF_FUNCTION_P (func_desc)))
{
rtx stack_ptr = gen_rtx_REG (Pmode, STACK_POINTER_REGNUM);
rtx stack_toc_offset = GEN_INT (RS6000_TOC_SAVE_SLOT);
rtx stack_toc_mem = gen_frame_mem (Pmode,
gen_rtx_PLUS (Pmode, stack_ptr,
stack_toc_offset));
rtx stack_toc_unspec = gen_rtx_UNSPEC (Pmode,
gen_rtvec (1, stack_toc_offset),
UNSPEC_TOCSLOT);
toc_restore = gen_rtx_SET (toc_reg, stack_toc_unspec);
if (TARGET_SAVE_TOC_INDIRECT && !cfun->calls_alloca)
cfun->machine->save_toc_in_prologue = true;
else
{
MEM_VOLATILE_P (stack_toc_mem) = 1;
emit_move_insn (stack_toc_mem, toc_reg);
}
if (DEFAULT_ABI == ABI_ELFv2)
{
func_addr = gen_rtx_REG (Pmode, 12);
emit_move_insn (func_addr, func_desc);
abi_reg = func_addr;
}
else
{
func_desc = force_reg (Pmode, func_desc);
func_addr = gen_reg_rtx (Pmode);
emit_move_insn (func_addr, gen_rtx_MEM (Pmode, func_desc));
rtx func_toc_offset = GEN_INT (GET_MODE_SIZE (Pmode));
rtx func_toc_mem = gen_rtx_MEM (Pmode,
gen_rtx_PLUS (Pmode, func_desc,
func_toc_offset));
toc_load = gen_rtx_USE (VOIDmode, func_toc_mem);
if (!direct_call_p
&& TARGET_POINTERS_TO_NESTED_FUNCTIONS
&& !chain_already_loaded (get_current_sequence ()->next->last))
{
rtx sc_reg = gen_rtx_REG (Pmode, STATIC_CHAIN_REGNUM);
rtx func_sc_offset = GEN_INT (2 * GET_MODE_SIZE (Pmode));
rtx func_sc_mem = gen_rtx_MEM (Pmode,
gen_rtx_PLUS (Pmode, func_desc,
func_sc_offset));
emit_move_insn (sc_reg, func_sc_mem);
abi_reg = sc_reg;
}
}
}
else
{
abi_reg = toc_reg;
func_addr = func_desc;
}
call[0] = gen_rtx_CALL (VOIDmode, gen_rtx_MEM (SImode, func_addr), flag);
if (value != NULL_RTX)
call[0] = gen_rtx_SET (value, call[0]);
n_call = 1;
if (toc_load)
call[n_call++] = toc_load;
if (toc_restore)
call[n_call++] = toc_restore;
call[n_call++] = gen_rtx_CLOBBER (VOIDmode, gen_rtx_REG (Pmode, LR_REGNO));
insn = gen_rtx_PARALLEL (VOIDmode, gen_rtvec_v (n_call, call));
insn = emit_call_insn (insn);
if (abi_reg)
use_reg (&CALL_INSN_FUNCTION_USAGE (insn), abi_reg);
}
void
rs6000_sibcall_aix (rtx value, rtx func_desc, rtx flag, rtx cookie)
{
rtx call[2];
rtx insn;
gcc_assert (INTVAL (cookie) == 0);
call[0] = gen_rtx_CALL (VOIDmode, gen_rtx_MEM (SImode, func_desc), flag);
if (value != NULL_RTX)
call[0] = gen_rtx_SET (value, call[0]);
call[1] = simple_return_rtx;
insn = gen_rtx_PARALLEL (VOIDmode, gen_rtvec_v (2, call));
insn = emit_call_insn (insn);
use_reg (&CALL_INSN_FUNCTION_USAGE (insn), gen_rtx_REG (Pmode, TOC_REGNUM));
}
static bool
rs6000_save_toc_in_prologue_p (void)
{
return (cfun && cfun->machine && cfun->machine->save_toc_in_prologue);
}
#ifdef HAVE_GAS_HIDDEN
# define USE_HIDDEN_LINKONCE 1
#else
# define USE_HIDDEN_LINKONCE 0
#endif
void
get_ppc476_thunk_name (char name[32])
{
gcc_assert (TARGET_LINK_STACK);
if (USE_HIDDEN_LINKONCE)
sprintf (name, "__ppc476.get_thunk");
else
ASM_GENERATE_INTERNAL_LABEL (name, "LPPC476_", 0);
}
static void rs6000_code_end (void) ATTRIBUTE_UNUSED;
static void
rs6000_code_end (void)
{
char name[32];
tree decl;
if (!TARGET_LINK_STACK)
return;
get_ppc476_thunk_name (name);
decl = build_decl (BUILTINS_LOCATION, FUNCTION_DECL, get_identifier (name),
build_function_type_list (void_type_node, NULL_TREE));
DECL_RESULT (decl) = build_decl (BUILTINS_LOCATION, RESULT_DECL,
NULL_TREE, void_type_node);
TREE_PUBLIC (decl) = 1;
TREE_STATIC (decl) = 1;
#if RS6000_WEAK
if (USE_HIDDEN_LINKONCE && !TARGET_XCOFF)
{
cgraph_node::create (decl)->set_comdat_group (DECL_ASSEMBLER_NAME (decl));
targetm.asm_out.unique_section (decl, 0);
switch_to_section (get_named_section (decl, NULL, 0));
DECL_WEAK (decl) = 1;
ASM_WEAKEN_DECL (asm_out_file, decl, name, 0);
targetm.asm_out.globalize_label (asm_out_file, name);
targetm.asm_out.assemble_visibility (decl, VISIBILITY_HIDDEN);
ASM_DECLARE_FUNCTION_NAME (asm_out_file, name, decl);
}
else
#endif
{
switch_to_section (text_section);
ASM_OUTPUT_LABEL (asm_out_file, name);
}
DECL_INITIAL (decl) = make_node (BLOCK);
current_function_decl = decl;
allocate_struct_function (decl, false);
init_function_start (decl);
first_function_block_is_cold = false;
final_start_function (emit_barrier (), asm_out_file, 1);
fputs ("\tblr\n", asm_out_file);
final_end_function ();
init_insn_lengths ();
free_after_compilation (cfun);
set_cfun (NULL);
current_function_decl = NULL;
}
static void
rs6000_set_up_by_prologue (struct hard_reg_set_container *set)
{
if (!TARGET_SINGLE_PIC_BASE
&& TARGET_TOC
&& TARGET_MINIMAL_TOC
&& !constant_pool_empty_p ())
add_to_hard_reg_set (&set->set, Pmode, RS6000_PIC_OFFSET_TABLE_REGNUM);
if (cfun->machine->split_stack_argp_used)
add_to_hard_reg_set (&set->set, Pmode, 12);
if (TARGET_TOC)
remove_from_hard_reg_set (&set->set, Pmode, TOC_REGNUM);
}

static void
rs6000_split_logical_inner (rtx dest,
rtx op1,
rtx op2,
enum rtx_code code,
machine_mode mode,
bool complement_final_p,
bool complement_op1_p,
bool complement_op2_p)
{
rtx bool_rtx;
if (op2 && GET_CODE (op2) == CONST_INT
&& (mode == SImode || (mode == DImode && TARGET_POWERPC64))
&& !complement_final_p && !complement_op1_p && !complement_op2_p)
{
HOST_WIDE_INT mask = GET_MODE_MASK (mode);
HOST_WIDE_INT value = INTVAL (op2) & mask;
if (code == AND)
{
if (value == 0)
{
emit_insn (gen_rtx_SET (dest, const0_rtx));
return;
}
else if (value == mask)
{
if (!rtx_equal_p (dest, op1))
emit_insn (gen_rtx_SET (dest, op1));
return;
}
}
else if (code == IOR || code == XOR)
{
if (value == 0)
{
if (!rtx_equal_p (dest, op1))
emit_insn (gen_rtx_SET (dest, op1));
return;
}
}
}
if (code == AND && mode == SImode
&& !complement_final_p && !complement_op1_p && !complement_op2_p)
{
emit_insn (gen_andsi3 (dest, op1, op2));
return;
}
if (complement_op1_p)
op1 = gen_rtx_NOT (mode, op1);
if (complement_op2_p)
op2 = gen_rtx_NOT (mode, op2);
if (!complement_op1_p && complement_op2_p)
std::swap (op1, op2);
bool_rtx = ((code == NOT)
? gen_rtx_NOT (mode, op1)
: gen_rtx_fmt_ee (code, mode, op1, op2));
if (complement_final_p)
bool_rtx = gen_rtx_NOT (mode, bool_rtx);
emit_insn (gen_rtx_SET (dest, bool_rtx));
}
static void
rs6000_split_logical_di (rtx operands[3],
enum rtx_code code,
bool complement_final_p,
bool complement_op1_p,
bool complement_op2_p)
{
const HOST_WIDE_INT lower_32bits = HOST_WIDE_INT_C(0xffffffff);
const HOST_WIDE_INT upper_32bits = ~ lower_32bits;
const HOST_WIDE_INT sign_bit = HOST_WIDE_INT_C(0x80000000);
enum hi_lo { hi = 0, lo = 1 };
rtx op0_hi_lo[2], op1_hi_lo[2], op2_hi_lo[2];
size_t i;
op0_hi_lo[hi] = gen_highpart (SImode, operands[0]);
op1_hi_lo[hi] = gen_highpart (SImode, operands[1]);
op0_hi_lo[lo] = gen_lowpart (SImode, operands[0]);
op1_hi_lo[lo] = gen_lowpart (SImode, operands[1]);
if (code == NOT)
op2_hi_lo[hi] = op2_hi_lo[lo] = NULL_RTX;
else
{
if (GET_CODE (operands[2]) != CONST_INT)
{
op2_hi_lo[hi] = gen_highpart_mode (SImode, DImode, operands[2]);
op2_hi_lo[lo] = gen_lowpart (SImode, operands[2]);
}
else
{
HOST_WIDE_INT value = INTVAL (operands[2]);
HOST_WIDE_INT value_hi_lo[2];
gcc_assert (!complement_final_p);
gcc_assert (!complement_op1_p);
gcc_assert (!complement_op2_p);
value_hi_lo[hi] = value >> 32;
value_hi_lo[lo] = value & lower_32bits;
for (i = 0; i < 2; i++)
{
HOST_WIDE_INT sub_value = value_hi_lo[i];
if (sub_value & sign_bit)
sub_value |= upper_32bits;
op2_hi_lo[i] = GEN_INT (sub_value);
if (code == AND && sub_value != -1 && sub_value != 0
&& !and_operand (op2_hi_lo[i], SImode))
op2_hi_lo[i] = force_reg (SImode, op2_hi_lo[i]);
}
}
}
for (i = 0; i < 2; i++)
{
if ((code == IOR || code == XOR)
&& GET_CODE (op2_hi_lo[i]) == CONST_INT
&& !complement_final_p
&& !complement_op1_p
&& !complement_op2_p
&& !logical_const_operand (op2_hi_lo[i], SImode))
{
HOST_WIDE_INT value = INTVAL (op2_hi_lo[i]);
HOST_WIDE_INT hi_16bits = value & HOST_WIDE_INT_C(0xffff0000);
HOST_WIDE_INT lo_16bits = value & HOST_WIDE_INT_C(0x0000ffff);
rtx tmp = gen_reg_rtx (SImode);
if ((hi_16bits & sign_bit) != 0)
hi_16bits |= upper_32bits;
rs6000_split_logical_inner (tmp, op1_hi_lo[i], GEN_INT (hi_16bits),
code, SImode, false, false, false);
rs6000_split_logical_inner (op0_hi_lo[i], tmp, GEN_INT (lo_16bits),
code, SImode, false, false, false);
}
else
rs6000_split_logical_inner (op0_hi_lo[i], op1_hi_lo[i], op2_hi_lo[i],
code, SImode, complement_final_p,
complement_op1_p, complement_op2_p);
}
return;
}
void
rs6000_split_logical (rtx operands[3],
enum rtx_code code,
bool complement_final_p,
bool complement_op1_p,
bool complement_op2_p)
{
machine_mode mode = GET_MODE (operands[0]);
machine_mode sub_mode;
rtx op0, op1, op2;
int sub_size, regno0, regno1, nregs, i;
if (mode == DImode && !TARGET_POWERPC64)
{
rs6000_split_logical_di (operands, code, complement_final_p,
complement_op1_p, complement_op2_p);
return;
}
op0 = operands[0];
op1 = operands[1];
op2 = (code == NOT) ? NULL_RTX : operands[2];
sub_mode = (TARGET_POWERPC64) ? DImode : SImode;
sub_size = GET_MODE_SIZE (sub_mode);
regno0 = REGNO (op0);
regno1 = REGNO (op1);
gcc_assert (reload_completed);
gcc_assert (IN_RANGE (regno0, FIRST_GPR_REGNO, LAST_GPR_REGNO));
gcc_assert (IN_RANGE (regno1, FIRST_GPR_REGNO, LAST_GPR_REGNO));
nregs = rs6000_hard_regno_nregs[(int)mode][regno0];
gcc_assert (nregs > 1);
if (op2 && REG_P (op2))
gcc_assert (IN_RANGE (REGNO (op2), FIRST_GPR_REGNO, LAST_GPR_REGNO));
for (i = 0; i < nregs; i++)
{
int offset = i * sub_size;
rtx sub_op0 = simplify_subreg (sub_mode, op0, mode, offset);
rtx sub_op1 = simplify_subreg (sub_mode, op1, mode, offset);
rtx sub_op2 = ((code == NOT)
? NULL_RTX
: simplify_subreg (sub_mode, op2, mode, offset));
rs6000_split_logical_inner (sub_op0, sub_op1, sub_op2, code, sub_mode,
complement_final_p, complement_op1_p,
complement_op2_p);
}
return;
}

bool
fusion_gpr_load_p (rtx addis_reg,	
rtx addis_value,	
rtx target,		
rtx mem)		
{
rtx addr;
rtx base_reg;
if (!base_reg_operand (addis_reg, GET_MODE (addis_reg)))
return false;
if (!base_reg_operand (target, GET_MODE (target)))
return false;
if (!fusion_gpr_addis (addis_value, GET_MODE (addis_value)))
return false;
if (GET_CODE (mem) == ZERO_EXTEND
|| (GET_CODE (mem) == SIGN_EXTEND && TARGET_P8_FUSION_SIGN))
mem = XEXP (mem, 0);
if (!MEM_P (mem))
return false;
if (!fusion_gpr_mem_load (mem, GET_MODE (mem)))
return false;
addr = XEXP (mem, 0);			
if (GET_CODE (addr) != PLUS && GET_CODE (addr) != LO_SUM)
return false;
if (REGNO (addis_reg) != REGNO (target))
{
if (reg_mentioned_p (target, mem))
return false;
if (!peep2_reg_dead_p (2, addis_reg))
return false;
if (REG_P (target) && REGNO (target) == STACK_POINTER_REGNUM)
return false;
}
base_reg = XEXP (addr, 0);
return REGNO (addis_reg) == REGNO (base_reg);
}
void
expand_fusion_gpr_load (rtx *operands)
{
rtx addis_value = operands[1];
rtx target = operands[2];
rtx orig_mem = operands[3];
rtx  new_addr, new_mem, orig_addr, offset;
enum rtx_code plus_or_lo_sum;
machine_mode target_mode = GET_MODE (target);
machine_mode extend_mode = target_mode;
machine_mode ptr_mode = Pmode;
enum rtx_code extend = UNKNOWN;
if (GET_CODE (orig_mem) == ZERO_EXTEND
|| (TARGET_P8_FUSION_SIGN && GET_CODE (orig_mem) == SIGN_EXTEND))
{
extend = GET_CODE (orig_mem);
orig_mem = XEXP (orig_mem, 0);
target_mode = GET_MODE (orig_mem);
}
gcc_assert (MEM_P (orig_mem));
orig_addr = XEXP (orig_mem, 0);
plus_or_lo_sum = GET_CODE (orig_addr);
gcc_assert (plus_or_lo_sum == PLUS || plus_or_lo_sum == LO_SUM);
offset = XEXP (orig_addr, 1);
new_addr = gen_rtx_fmt_ee (plus_or_lo_sum, ptr_mode, addis_value, offset);
new_mem = replace_equiv_address_nv (orig_mem, new_addr, false);
if (extend != UNKNOWN)
new_mem = gen_rtx_fmt_e (ZERO_EXTEND, extend_mode, new_mem);
new_mem = gen_rtx_UNSPEC (extend_mode, gen_rtvec (1, new_mem),
UNSPEC_FUSION_GPR);
emit_insn (gen_rtx_SET (target, new_mem));
if (extend == SIGN_EXTEND)
{
int sub_off = ((BYTES_BIG_ENDIAN)
? GET_MODE_SIZE (extend_mode) - GET_MODE_SIZE (target_mode)
: 0);
rtx sign_reg
= simplify_subreg (target_mode, target, extend_mode, sub_off);
emit_insn (gen_rtx_SET (target,
gen_rtx_SIGN_EXTEND (extend_mode, sign_reg)));
}
return;
}
void
emit_fusion_addis (rtx target, rtx addis_value)
{
rtx fuse_ops[10];
const char *addis_str = NULL;
fuse_ops[0] = target;
if (satisfies_constraint_L (addis_value))
{
fuse_ops[1] = addis_value;
addis_str = "lis %0,%v1";
}
else if (GET_CODE (addis_value) == PLUS)
{
rtx op0 = XEXP (addis_value, 0);
rtx op1 = XEXP (addis_value, 1);
if (REG_P (op0) && CONST_INT_P (op1)
&& satisfies_constraint_L (op1))
{
fuse_ops[1] = op0;
fuse_ops[2] = op1;
addis_str = "addis %0,%1,%v2";
}
}
else if (GET_CODE (addis_value) == HIGH)
{
rtx value = XEXP (addis_value, 0);
if (GET_CODE (value) == UNSPEC && XINT (value, 1) == UNSPEC_TOCREL)
{
fuse_ops[1] = XVECEXP (value, 0, 0);		
fuse_ops[2] = XVECEXP (value, 0, 1);		
if (TARGET_ELF)
addis_str = "addis %0,%2,%1@toc@ha";
else if (TARGET_XCOFF)
addis_str = "addis %0,%1@u(%2)";
else
gcc_unreachable ();
}
else if (GET_CODE (value) == PLUS)
{
rtx op0 = XEXP (value, 0);
rtx op1 = XEXP (value, 1);
if (GET_CODE (op0) == UNSPEC
&& XINT (op0, 1) == UNSPEC_TOCREL
&& CONST_INT_P (op1))
{
fuse_ops[1] = XVECEXP (op0, 0, 0);	
fuse_ops[2] = XVECEXP (op0, 0, 1);	
fuse_ops[3] = op1;
if (TARGET_ELF)
addis_str = "addis %0,%2,%1+%3@toc@ha";
else if (TARGET_XCOFF)
addis_str = "addis %0,%1+%3@u(%2)";
else
gcc_unreachable ();
}
}
else if (satisfies_constraint_L (value))
{
fuse_ops[1] = value;
addis_str = "lis %0,%v1";
}
else if (TARGET_ELF && !TARGET_POWERPC64 && CONSTANT_P (value))
{
fuse_ops[1] = value;
addis_str = "lis %0,%1@ha";
}
}
if (!addis_str)
fatal_insn ("Could not generate addis value for fusion", addis_value);
output_asm_insn (addis_str, fuse_ops);
}
void
emit_fusion_load_store (rtx load_store_reg, rtx addis_reg, rtx offset,
const char *insn_str)
{
rtx fuse_ops[10];
char insn_template[80];
fuse_ops[0] = load_store_reg;
fuse_ops[1] = addis_reg;
if (CONST_INT_P (offset) && satisfies_constraint_I (offset))
{
sprintf (insn_template, "%s %%0,%%2(%%1)", insn_str);
fuse_ops[2] = offset;
output_asm_insn (insn_template, fuse_ops);
}
else if (GET_CODE (offset) == UNSPEC
&& XINT (offset, 1) == UNSPEC_TOCREL)
{
if (TARGET_ELF)
sprintf (insn_template, "%s %%0,%%2@toc@l(%%1)", insn_str);
else if (TARGET_XCOFF)
sprintf (insn_template, "%s %%0,%%2@l(%%1)", insn_str);
else
gcc_unreachable ();
fuse_ops[2] = XVECEXP (offset, 0, 0);
output_asm_insn (insn_template, fuse_ops);
}
else if (GET_CODE (offset) == PLUS
&& GET_CODE (XEXP (offset, 0)) == UNSPEC
&& XINT (XEXP (offset, 0), 1) == UNSPEC_TOCREL
&& CONST_INT_P (XEXP (offset, 1)))
{
rtx tocrel_unspec = XEXP (offset, 0);
if (TARGET_ELF)
sprintf (insn_template, "%s %%0,%%2+%%3@toc@l(%%1)", insn_str);
else if (TARGET_XCOFF)
sprintf (insn_template, "%s %%0,%%2+%%3@l(%%1)", insn_str);
else
gcc_unreachable ();
fuse_ops[2] = XVECEXP (tocrel_unspec, 0, 0);
fuse_ops[3] = XEXP (offset, 1);
output_asm_insn (insn_template, fuse_ops);
}
else if (TARGET_ELF && !TARGET_POWERPC64 && CONSTANT_P (offset))
{
sprintf (insn_template, "%s %%0,%%2@l(%%1)", insn_str);
fuse_ops[2] = offset;
output_asm_insn (insn_template, fuse_ops);
}
else
fatal_insn ("Unable to generate load/store offset for fusion", offset);
return;
}
rtx
fusion_wrap_memory_address (rtx old_mem)
{
rtx old_addr = XEXP (old_mem, 0);
rtvec v = gen_rtvec (1, old_addr);
rtx new_addr = gen_rtx_UNSPEC (Pmode, v, UNSPEC_FUSION_ADDIS);
return replace_equiv_address_nv (old_mem, new_addr, false);
}
static void
fusion_split_address (rtx addr, rtx *p_hi, rtx *p_lo)
{
rtx hi, lo;
if (GET_CODE (addr) == UNSPEC && XINT (addr, 1) == UNSPEC_FUSION_ADDIS)
{
lo = XVECEXP (addr, 0, 0);
hi = gen_rtx_HIGH (Pmode, lo);
}
else if (GET_CODE (addr) == PLUS || GET_CODE (addr) == LO_SUM)
{
hi = XEXP (addr, 0);
lo = XEXP (addr, 1);
}
else
gcc_unreachable ();
*p_hi = hi;
*p_lo = lo;
}
const char *
emit_fusion_gpr_load (rtx target, rtx mem)
{
rtx addis_value;
rtx addr;
rtx load_offset;
const char *load_str = NULL;
machine_mode mode;
if (GET_CODE (mem) == ZERO_EXTEND)
mem = XEXP (mem, 0);
gcc_assert (REG_P (target) && MEM_P (mem));
addr = XEXP (mem, 0);
fusion_split_address (addr, &addis_value, &load_offset);
mode = GET_MODE (mem);
switch (mode)
{
case E_QImode:
load_str = "lbz";
break;
case E_HImode:
load_str = "lhz";
break;
case E_SImode:
case E_SFmode:
load_str = "lwz";
break;
case E_DImode:
case E_DFmode:
gcc_assert (TARGET_POWERPC64);
load_str = "ld";
break;
default:
fatal_insn ("Bad GPR fusion", gen_rtx_SET (target, mem));
}
emit_fusion_addis (target, addis_value);
emit_fusion_load_store (target, target, load_offset, load_str);
return "";
}

bool
fusion_p9_p (rtx addis_reg,		
rtx addis_value,		
rtx dest,			
rtx src)			
{
rtx addr, mem, offset;
machine_mode mode = GET_MODE (src);
if (!base_reg_operand (addis_reg, GET_MODE (addis_reg)))
return false;
if (!fusion_gpr_addis (addis_value, GET_MODE (addis_value)))
return false;
if (GET_CODE (src) == FLOAT_EXTEND || GET_CODE (src) == ZERO_EXTEND)
src = XEXP (src, 0);
if (fpr_reg_operand (src, mode) || int_reg_operand (src, mode))
{
if (!MEM_P (dest))
return false;
mem = dest;
}
else if (MEM_P (src))
{
if (!fpr_reg_operand (dest, mode) && !int_reg_operand (dest, mode))
return false;
mem = src;
}
else
return false;
addr = XEXP (mem, 0);			
if (GET_CODE (addr) == PLUS)
{
if (!rtx_equal_p (addis_reg, XEXP (addr, 0)))
return false;
return satisfies_constraint_I (XEXP (addr, 1));
}
else if (GET_CODE (addr) == LO_SUM)
{
if (!rtx_equal_p (addis_reg, XEXP (addr, 0)))
return false;
offset = XEXP (addr, 1);
if (TARGET_XCOFF || (TARGET_ELF && TARGET_POWERPC64))
return small_toc_ref (offset, GET_MODE (offset));
else if (TARGET_ELF && !TARGET_POWERPC64)
return CONSTANT_P (offset);
}
return false;
}
void
expand_fusion_p9_load (rtx *operands)
{
rtx tmp_reg = operands[0];
rtx addis_value = operands[1];
rtx target = operands[2];
rtx orig_mem = operands[3];
rtx  new_addr, new_mem, orig_addr, offset, set, clobber, insn;
enum rtx_code plus_or_lo_sum;
machine_mode target_mode = GET_MODE (target);
machine_mode extend_mode = target_mode;
machine_mode ptr_mode = Pmode;
enum rtx_code extend = UNKNOWN;
if (GET_CODE (orig_mem) == FLOAT_EXTEND || GET_CODE (orig_mem) == ZERO_EXTEND)
{
extend = GET_CODE (orig_mem);
orig_mem = XEXP (orig_mem, 0);
target_mode = GET_MODE (orig_mem);
}
gcc_assert (MEM_P (orig_mem));
orig_addr = XEXP (orig_mem, 0);
plus_or_lo_sum = GET_CODE (orig_addr);
gcc_assert (plus_or_lo_sum == PLUS || plus_or_lo_sum == LO_SUM);
offset = XEXP (orig_addr, 1);
new_addr = gen_rtx_fmt_ee (plus_or_lo_sum, ptr_mode, addis_value, offset);
new_mem = replace_equiv_address_nv (orig_mem, new_addr, false);
if (extend != UNKNOWN)
new_mem = gen_rtx_fmt_e (extend, extend_mode, new_mem);
new_mem = gen_rtx_UNSPEC (extend_mode, gen_rtvec (1, new_mem),
UNSPEC_FUSION_P9);
set = gen_rtx_SET (target, new_mem);
clobber = gen_rtx_CLOBBER (VOIDmode, tmp_reg);
insn = gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, set, clobber));
emit_insn (insn);
return;
}
void
expand_fusion_p9_store (rtx *operands)
{
rtx tmp_reg = operands[0];
rtx addis_value = operands[1];
rtx orig_mem = operands[2];
rtx src = operands[3];
rtx  new_addr, new_mem, orig_addr, offset, set, clobber, insn, new_src;
enum rtx_code plus_or_lo_sum;
machine_mode target_mode = GET_MODE (orig_mem);
machine_mode ptr_mode = Pmode;
gcc_assert (MEM_P (orig_mem));
orig_addr = XEXP (orig_mem, 0);
plus_or_lo_sum = GET_CODE (orig_addr);
gcc_assert (plus_or_lo_sum == PLUS || plus_or_lo_sum == LO_SUM);
offset = XEXP (orig_addr, 1);
new_addr = gen_rtx_fmt_ee (plus_or_lo_sum, ptr_mode, addis_value, offset);
new_mem = replace_equiv_address_nv (orig_mem, new_addr, false);
new_src = gen_rtx_UNSPEC (target_mode, gen_rtvec (1, src),
UNSPEC_FUSION_P9);
set = gen_rtx_SET (new_mem, new_src);
clobber = gen_rtx_CLOBBER (VOIDmode, tmp_reg);
insn = gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, set, clobber));
emit_insn (insn);
return;
}
const char *
emit_fusion_p9_load (rtx reg, rtx mem, rtx tmp_reg)
{
machine_mode mode = GET_MODE (reg);
rtx hi;
rtx lo;
rtx addr;
const char *load_string;
int r;
if (GET_CODE (mem) == FLOAT_EXTEND || GET_CODE (mem) == ZERO_EXTEND)
{
mem = XEXP (mem, 0);
mode = GET_MODE (mem);
}
if (GET_CODE (reg) == SUBREG)
{
gcc_assert (SUBREG_BYTE (reg) == 0);
reg = SUBREG_REG (reg);
}
if (!REG_P (reg))
fatal_insn ("emit_fusion_p9_load, bad reg #1", reg);
r = REGNO (reg);
if (FP_REGNO_P (r))
{
if (mode == SFmode)
load_string = "lfs";
else if (mode == DFmode || mode == DImode)
load_string = "lfd";
else
gcc_unreachable ();
}
else if (ALTIVEC_REGNO_P (r) && TARGET_P9_VECTOR)
{
if (mode == SFmode)
load_string = "lxssp";
else if (mode == DFmode || mode == DImode)
load_string = "lxsd";
else
gcc_unreachable ();
}
else if (INT_REGNO_P (r))
{
switch (mode)
{
case E_QImode:
load_string = "lbz";
break;
case E_HImode:
load_string = "lhz";
break;
case E_SImode:
case E_SFmode:
load_string = "lwz";
break;
case E_DImode:
case E_DFmode:
if (!TARGET_POWERPC64)
gcc_unreachable ();
load_string = "ld";
break;
default:
gcc_unreachable ();
}
}
else
fatal_insn ("emit_fusion_p9_load, bad reg #2", reg);
if (!MEM_P (mem))
fatal_insn ("emit_fusion_p9_load not MEM", mem);
addr = XEXP (mem, 0);
fusion_split_address (addr, &hi, &lo);
emit_fusion_addis (tmp_reg, hi);
emit_fusion_load_store (reg, tmp_reg, lo, load_string);
return "";
}
const char *
emit_fusion_p9_store (rtx mem, rtx reg, rtx tmp_reg)
{
machine_mode mode = GET_MODE (reg);
rtx hi;
rtx lo;
rtx addr;
const char *store_string;
int r;
if (GET_CODE (reg) == SUBREG)
{
gcc_assert (SUBREG_BYTE (reg) == 0);
reg = SUBREG_REG (reg);
}
if (!REG_P (reg))
fatal_insn ("emit_fusion_p9_store, bad reg #1", reg);
r = REGNO (reg);
if (FP_REGNO_P (r))
{
if (mode == SFmode)
store_string = "stfs";
else if (mode == DFmode)
store_string = "stfd";
else
gcc_unreachable ();
}
else if (ALTIVEC_REGNO_P (r) && TARGET_P9_VECTOR)
{
if (mode == SFmode)
store_string = "stxssp";
else if (mode == DFmode || mode == DImode)
store_string = "stxsd";
else
gcc_unreachable ();
}
else if (INT_REGNO_P (r))
{
switch (mode)
{
case E_QImode:
store_string = "stb";
break;
case E_HImode:
store_string = "sth";
break;
case E_SImode:
case E_SFmode:
store_string = "stw";
break;
case E_DImode:
case E_DFmode:
if (!TARGET_POWERPC64)
gcc_unreachable ();
store_string = "std";
break;
default:
gcc_unreachable ();
}
}
else
fatal_insn ("emit_fusion_p9_store, bad reg #2", reg);
if (!MEM_P (mem))
fatal_insn ("emit_fusion_p9_store not MEM", mem);
addr = XEXP (mem, 0);
fusion_split_address (addr, &hi, &lo);
emit_fusion_addis (tmp_reg, hi);
emit_fusion_load_store (reg, tmp_reg, lo, store_string);
return "";
}
#ifdef RS6000_GLIBC_ATOMIC_FENV
static tree atomic_hold_decl, atomic_clear_decl, atomic_update_decl;
#endif
static void
rs6000_atomic_assign_expand_fenv (tree *hold, tree *clear, tree *update)
{
if (!TARGET_HARD_FLOAT)
{
#ifdef RS6000_GLIBC_ATOMIC_FENV
if (atomic_hold_decl == NULL_TREE)
{
atomic_hold_decl
= build_decl (BUILTINS_LOCATION, FUNCTION_DECL,
get_identifier ("__atomic_feholdexcept"),
build_function_type_list (void_type_node,
double_ptr_type_node,
NULL_TREE));
TREE_PUBLIC (atomic_hold_decl) = 1;
DECL_EXTERNAL (atomic_hold_decl) = 1;
}
if (atomic_clear_decl == NULL_TREE)
{
atomic_clear_decl
= build_decl (BUILTINS_LOCATION, FUNCTION_DECL,
get_identifier ("__atomic_feclearexcept"),
build_function_type_list (void_type_node,
NULL_TREE));
TREE_PUBLIC (atomic_clear_decl) = 1;
DECL_EXTERNAL (atomic_clear_decl) = 1;
}
tree const_double = build_qualified_type (double_type_node,
TYPE_QUAL_CONST);
tree const_double_ptr = build_pointer_type (const_double);
if (atomic_update_decl == NULL_TREE)
{
atomic_update_decl
= build_decl (BUILTINS_LOCATION, FUNCTION_DECL,
get_identifier ("__atomic_feupdateenv"),
build_function_type_list (void_type_node,
const_double_ptr,
NULL_TREE));
TREE_PUBLIC (atomic_update_decl) = 1;
DECL_EXTERNAL (atomic_update_decl) = 1;
}
tree fenv_var = create_tmp_var_raw (double_type_node);
TREE_ADDRESSABLE (fenv_var) = 1;
tree fenv_addr = build1 (ADDR_EXPR, double_ptr_type_node, fenv_var);
*hold = build_call_expr (atomic_hold_decl, 1, fenv_addr);
*clear = build_call_expr (atomic_clear_decl, 0);
*update = build_call_expr (atomic_update_decl, 1,
fold_convert (const_double_ptr, fenv_addr));
#endif
return;
}
tree mffs = rs6000_builtin_decls[RS6000_BUILTIN_MFFS];
tree mtfsf = rs6000_builtin_decls[RS6000_BUILTIN_MTFSF];
tree call_mffs = build_call_expr (mffs, 0);
const unsigned HOST_WIDE_INT hold_exception_mask =
HOST_WIDE_INT_C (0xffffffff00000007);
tree fenv_var = create_tmp_var_raw (double_type_node);
tree hold_mffs = build2 (MODIFY_EXPR, void_type_node, fenv_var, call_mffs);
tree fenv_llu = build1 (VIEW_CONVERT_EXPR, uint64_type_node, fenv_var);
tree fenv_llu_and = build2 (BIT_AND_EXPR, uint64_type_node, fenv_llu,
build_int_cst (uint64_type_node,
hold_exception_mask));
tree fenv_hold_mtfsf = build1 (VIEW_CONVERT_EXPR, double_type_node,
fenv_llu_and);
tree hold_mtfsf = build_call_expr (mtfsf, 2,
build_int_cst (unsigned_type_node, 0xff),
fenv_hold_mtfsf);
*hold = build2 (COMPOUND_EXPR, void_type_node, hold_mffs, hold_mtfsf);
const unsigned HOST_WIDE_INT clear_exception_mask =
HOST_WIDE_INT_C (0xffffffff00000000);
tree fenv_clear = create_tmp_var_raw (double_type_node);
tree clear_mffs = build2 (MODIFY_EXPR, void_type_node, fenv_clear, call_mffs);
tree fenv_clean_llu = build1 (VIEW_CONVERT_EXPR, uint64_type_node, fenv_clear);
tree fenv_clear_llu_and = build2 (BIT_AND_EXPR, uint64_type_node,
fenv_clean_llu,
build_int_cst (uint64_type_node,
clear_exception_mask));
tree fenv_clear_mtfsf = build1 (VIEW_CONVERT_EXPR, double_type_node,
fenv_clear_llu_and);
tree clear_mtfsf = build_call_expr (mtfsf, 2,
build_int_cst (unsigned_type_node, 0xff),
fenv_clear_mtfsf);
*clear = build2 (COMPOUND_EXPR, void_type_node, clear_mffs, clear_mtfsf);
const unsigned HOST_WIDE_INT update_exception_mask =
HOST_WIDE_INT_C (0xffffffff1fffff00);
const unsigned HOST_WIDE_INT new_exception_mask =
HOST_WIDE_INT_C (0x1ff80fff);
tree old_fenv = create_tmp_var_raw (double_type_node);
tree update_mffs = build2 (MODIFY_EXPR, void_type_node, old_fenv, call_mffs);
tree old_llu = build1 (VIEW_CONVERT_EXPR, uint64_type_node, old_fenv);
tree old_llu_and = build2 (BIT_AND_EXPR, uint64_type_node, old_llu,
build_int_cst (uint64_type_node,
update_exception_mask));
tree new_llu_and = build2 (BIT_AND_EXPR, uint64_type_node, fenv_llu,
build_int_cst (uint64_type_node,
new_exception_mask));
tree new_llu_mask = build2 (BIT_IOR_EXPR, uint64_type_node,
old_llu_and, new_llu_and);
tree fenv_update_mtfsf = build1 (VIEW_CONVERT_EXPR, double_type_node,
new_llu_mask);
tree update_mtfsf = build_call_expr (mtfsf, 2,
build_int_cst (unsigned_type_node, 0xff),
fenv_update_mtfsf);
*update = build2 (COMPOUND_EXPR, void_type_node, update_mffs, update_mtfsf);
}
void
rs6000_generate_float2_double_code (rtx dst, rtx src1, rtx src2)
{
rtx rtx_tmp0, rtx_tmp1, rtx_tmp2, rtx_tmp3;
rtx_tmp0 = gen_reg_rtx (V2DFmode);
rtx_tmp1 = gen_reg_rtx (V2DFmode);
if (VECTOR_ELT_ORDER_BIG)
{
emit_insn (gen_vsx_xxpermdi_v2df_be (rtx_tmp0, src1, src2,
GEN_INT (0)));
emit_insn (gen_vsx_xxpermdi_v2df_be (rtx_tmp1, src1, src2,
GEN_INT (3)));
}
else
{
emit_insn (gen_vsx_xxpermdi_v2df (rtx_tmp0, src1, src2, GEN_INT (3)));
emit_insn (gen_vsx_xxpermdi_v2df (rtx_tmp1, src1, src2, GEN_INT (0)));
}
rtx_tmp2 = gen_reg_rtx (V4SFmode);
rtx_tmp3 = gen_reg_rtx (V4SFmode);
emit_insn (gen_vsx_xvcdpsp (rtx_tmp2, rtx_tmp0));
emit_insn (gen_vsx_xvcdpsp (rtx_tmp3, rtx_tmp1));
if (VECTOR_ELT_ORDER_BIG)
emit_insn (gen_p8_vmrgew_v4sf (dst, rtx_tmp2, rtx_tmp3));
else
emit_insn (gen_p8_vmrgew_v4sf (dst, rtx_tmp3, rtx_tmp2));
}
void
rs6000_generate_float2_code (bool signed_convert, rtx dst, rtx src1, rtx src2)
{
rtx rtx_tmp0, rtx_tmp1, rtx_tmp2, rtx_tmp3;
rtx_tmp0 = gen_reg_rtx (V2DImode);
rtx_tmp1 = gen_reg_rtx (V2DImode);
if (VECTOR_ELT_ORDER_BIG)
{
emit_insn (gen_vsx_xxpermdi_v2di_be (rtx_tmp0, src1, src2, GEN_INT (0)));
emit_insn (gen_vsx_xxpermdi_v2di_be (rtx_tmp1, src1, src2, GEN_INT (3)));
}
else
{
emit_insn (gen_vsx_xxpermdi_v2di (rtx_tmp0, src1, src2, GEN_INT (3)));
emit_insn (gen_vsx_xxpermdi_v2di (rtx_tmp1, src1, src2, GEN_INT (0)));
}
rtx_tmp2 = gen_reg_rtx (V4SFmode);
rtx_tmp3 = gen_reg_rtx (V4SFmode);
if (signed_convert)
{
emit_insn (gen_vsx_xvcvsxdsp (rtx_tmp2, rtx_tmp0));
emit_insn (gen_vsx_xvcvsxdsp (rtx_tmp3, rtx_tmp1));
}
else
{
emit_insn (gen_vsx_xvcvuxdsp (rtx_tmp2, rtx_tmp0));
emit_insn (gen_vsx_xvcvuxdsp (rtx_tmp3, rtx_tmp1));
}
if (VECTOR_ELT_ORDER_BIG)
emit_insn (gen_p8_vmrgew_v4sf (dst, rtx_tmp2, rtx_tmp3));
else
emit_insn (gen_p8_vmrgew_v4sf (dst, rtx_tmp3, rtx_tmp2));
}
void
rs6000_generate_vsigned2_code (bool signed_convert, rtx dst, rtx src1,
rtx src2)
{
rtx rtx_tmp0, rtx_tmp1, rtx_tmp2, rtx_tmp3;
rtx_tmp0 = gen_reg_rtx (V2DFmode);
rtx_tmp1 = gen_reg_rtx (V2DFmode);
emit_insn (gen_vsx_xxpermdi_v2df (rtx_tmp0, src1, src2, GEN_INT (0)));
emit_insn (gen_vsx_xxpermdi_v2df (rtx_tmp1, src1, src2, GEN_INT (3)));
rtx_tmp2 = gen_reg_rtx (V4SImode);
rtx_tmp3 = gen_reg_rtx (V4SImode);
if (signed_convert)
{
emit_insn (gen_vsx_xvcvdpsxws (rtx_tmp2, rtx_tmp0));
emit_insn (gen_vsx_xvcvdpsxws (rtx_tmp3, rtx_tmp1));
}
else
{
emit_insn (gen_vsx_xvcvdpuxws (rtx_tmp2, rtx_tmp0));
emit_insn (gen_vsx_xvcvdpuxws (rtx_tmp3, rtx_tmp1));
}
emit_insn (gen_p8_vmrgew_v4si (dst, rtx_tmp2, rtx_tmp3));
}
static bool
rs6000_optab_supported_p (int op, machine_mode mode1, machine_mode,
optimization_type opt_type)
{
switch (op)
{
case rsqrt_optab:
return (opt_type == OPTIMIZE_FOR_SPEED
&& RS6000_RECIP_AUTO_RSQRTE_P (mode1));
default:
return true;
}
}
static HOST_WIDE_INT
rs6000_constant_alignment (const_tree exp, HOST_WIDE_INT align)
{
if (TREE_CODE (exp) == STRING_CST
&& (STRICT_ALIGNMENT || !optimize_size))
return MAX (align, BITS_PER_WORD);
return align;
}
static HOST_WIDE_INT
rs6000_starting_frame_offset (void)
{
if (FRAME_GROWS_DOWNWARD)
return 0;
return RS6000_STARTING_FRAME_OFFSET;
}

#if TARGET_ELF && RS6000_WEAK
static void
rs6000_globalize_decl_name (FILE * stream, tree decl)
{
const char *name = XSTR (XEXP (DECL_RTL (decl), 0), 0);
targetm.asm_out.globalize_label (stream, name);
if (rs6000_passes_ieee128 && name[0] == '_' && name[1] == 'Z')
{
tree save_asm_name = DECL_ASSEMBLER_NAME (decl);
const char *old_name;
ieee128_mangling_gcc_8_1 = true;
lang_hooks.set_decl_assembler_name (decl);
old_name = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));
SET_DECL_ASSEMBLER_NAME (decl, save_asm_name);
ieee128_mangling_gcc_8_1 = false;
if (strcmp (name, old_name) != 0)
{
fprintf (stream, "\t.weak %s\n", old_name);
fprintf (stream, "\t.set %s,%s\n", old_name, name);
}
}
}
#endif

struct gcc_target targetm = TARGET_INITIALIZER;
#include "gt-rs6000.h"
