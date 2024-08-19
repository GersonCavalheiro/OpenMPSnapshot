#ifndef GCC_AARCH64_H
#define GCC_AARCH64_H
#define TARGET_CPU_CPP_BUILTINS()	\
aarch64_cpu_cpp_builtins (pfile)

#define REGISTER_TARGET_PRAGMAS() aarch64_register_pragmas ()
#define PROMOTE_MODE(MODE, UNSIGNEDP, TYPE)	\
if (GET_MODE_CLASS (MODE) == MODE_INT		\
&& GET_MODE_SIZE (MODE) < 4)		\
{						\
if (MODE == QImode || MODE == HImode)	\
{					\
MODE = SImode;			\
}					\
}
#define BITS_BIG_ENDIAN 0
#define BYTES_BIG_ENDIAN (TARGET_BIG_END != 0)
#define WORDS_BIG_ENDIAN (BYTES_BIG_ENDIAN)
#define TARGET_SIMD (!TARGET_GENERAL_REGS_ONLY && AARCH64_ISA_SIMD)
#define TARGET_FLOAT (!TARGET_GENERAL_REGS_ONLY && AARCH64_ISA_FP)
#define UNITS_PER_WORD		8
#define UNITS_PER_VREG		16
#define PARM_BOUNDARY		64
#define STACK_BOUNDARY		128
#define FUNCTION_BOUNDARY	32
#define EMPTY_FIELD_BOUNDARY	32
#define BIGGEST_ALIGNMENT	128
#define SHORT_TYPE_SIZE		16
#define INT_TYPE_SIZE		32
#define LONG_TYPE_SIZE		(TARGET_ILP32 ? 32 : 64)
#define POINTER_SIZE		(TARGET_ILP32 ? 32 : 64)
#define LONG_LONG_TYPE_SIZE	64
#define FLOAT_TYPE_SIZE		32
#define DOUBLE_TYPE_SIZE	64
#define LONG_DOUBLE_TYPE_SIZE	128
#define TARGET_PTRMEMFUNC_VBIT_LOCATION ptrmemfunc_vbit_in_delta
#define AARCH64_EXPAND_ALIGNMENT(COND, EXP, ALIGN)			\
(((COND) && ((ALIGN) < BITS_PER_WORD)					\
&& (TREE_CODE (EXP) == ARRAY_TYPE					\
|| TREE_CODE (EXP) == UNION_TYPE				\
|| TREE_CODE (EXP) == RECORD_TYPE)) ? BITS_PER_WORD : (ALIGN))
#define DATA_ALIGNMENT(EXP, ALIGN)			\
AARCH64_EXPAND_ALIGNMENT (!optimize_size, EXP, ALIGN)
#define LOCAL_ALIGNMENT(EXP, ALIGN)				\
AARCH64_EXPAND_ALIGNMENT (!flag_conserve_stack, EXP, ALIGN)
#define STRUCTURE_SIZE_BOUNDARY		8
#define MALLOC_ABI_ALIGNMENT  128
#define WCHAR_TYPE "unsigned int"
#define WCHAR_TYPE_SIZE			32
#define SIZE_TYPE	"long unsigned int"
#define PTRDIFF_TYPE	"long int"
#define PCC_BITFIELD_TYPE_MATTERS	1
extern unsigned aarch64_architecture_version;
#define AARCH64_FL_SIMD       (1 << 0)	
#define AARCH64_FL_FP         (1 << 1)	
#define AARCH64_FL_CRYPTO     (1 << 2)	
#define AARCH64_FL_CRC        (1 << 3)	
#define AARCH64_FL_LSE	      (1 << 4)  
#define AARCH64_FL_RDMA       (1 << 5)  
#define AARCH64_FL_V8_1       (1 << 6)  
#define AARCH64_FL_V8_2       (1 << 8)  
#define AARCH64_FL_F16	      (1 << 9)  
#define AARCH64_FL_SVE        (1 << 10) 
#define AARCH64_FL_V8_3       (1 << 11)  
#define AARCH64_FL_RCPC       (1 << 12)  
#define AARCH64_FL_DOTPROD    (1 << 13)  
#define AARCH64_FL_AES	      (1 << 14)  
#define AARCH64_FL_SHA2	      (1 << 15)  
#define AARCH64_FL_V8_4	      (1 << 16)  
#define AARCH64_FL_SM4	      (1 << 17)  
#define AARCH64_FL_SHA3	      (1 << 18)  
#define AARCH64_FL_F16FML     (1 << 19)  
#define AARCH64_FL_FPSIMD     (AARCH64_FL_FP | AARCH64_FL_SIMD)
#define AARCH64_FL_FPQ16      (AARCH64_FL_FP & ~AARCH64_FL_SIMD)
#define AARCH64_FL_FOR_ARCH8       (AARCH64_FL_FPSIMD)
#define AARCH64_FL_FOR_ARCH8_1			       \
(AARCH64_FL_FOR_ARCH8 | AARCH64_FL_LSE | AARCH64_FL_CRC \
| AARCH64_FL_RDMA | AARCH64_FL_V8_1)
#define AARCH64_FL_FOR_ARCH8_2			\
(AARCH64_FL_FOR_ARCH8_1 | AARCH64_FL_V8_2)
#define AARCH64_FL_FOR_ARCH8_3			\
(AARCH64_FL_FOR_ARCH8_2 | AARCH64_FL_V8_3)
#define AARCH64_FL_FOR_ARCH8_4			\
(AARCH64_FL_FOR_ARCH8_3 | AARCH64_FL_V8_4 | AARCH64_FL_F16FML \
| AARCH64_FL_DOTPROD)
#define AARCH64_ISA_CRC            (aarch64_isa_flags & AARCH64_FL_CRC)
#define AARCH64_ISA_CRYPTO         (aarch64_isa_flags & AARCH64_FL_CRYPTO)
#define AARCH64_ISA_FP             (aarch64_isa_flags & AARCH64_FL_FP)
#define AARCH64_ISA_SIMD           (aarch64_isa_flags & AARCH64_FL_SIMD)
#define AARCH64_ISA_LSE		   (aarch64_isa_flags & AARCH64_FL_LSE)
#define AARCH64_ISA_RDMA	   (aarch64_isa_flags & AARCH64_FL_RDMA)
#define AARCH64_ISA_V8_2	   (aarch64_isa_flags & AARCH64_FL_V8_2)
#define AARCH64_ISA_F16		   (aarch64_isa_flags & AARCH64_FL_F16)
#define AARCH64_ISA_SVE            (aarch64_isa_flags & AARCH64_FL_SVE)
#define AARCH64_ISA_V8_3	   (aarch64_isa_flags & AARCH64_FL_V8_3)
#define AARCH64_ISA_DOTPROD	   (aarch64_isa_flags & AARCH64_FL_DOTPROD)
#define AARCH64_ISA_AES	           (aarch64_isa_flags & AARCH64_FL_AES)
#define AARCH64_ISA_SHA2	   (aarch64_isa_flags & AARCH64_FL_SHA2)
#define AARCH64_ISA_V8_4	   (aarch64_isa_flags & AARCH64_FL_V8_4)
#define AARCH64_ISA_SM4	           (aarch64_isa_flags & AARCH64_FL_SM4)
#define AARCH64_ISA_SHA3	   (aarch64_isa_flags & AARCH64_FL_SHA3)
#define AARCH64_ISA_F16FML	   (aarch64_isa_flags & AARCH64_FL_F16FML)
#define TARGET_CRYPTO (TARGET_SIMD && AARCH64_ISA_CRYPTO)
#define TARGET_SHA2 ((TARGET_SIMD && AARCH64_ISA_SHA2) || TARGET_CRYPTO)
#define TARGET_SHA3 (TARGET_SIMD && AARCH64_ISA_SHA3)
#define TARGET_AES ((TARGET_SIMD && AARCH64_ISA_AES) || TARGET_CRYPTO)
#define TARGET_SM4 (TARGET_SIMD && AARCH64_ISA_SM4)
#define TARGET_F16FML (TARGET_SIMD && AARCH64_ISA_F16FML && TARGET_FP_F16INST)
#define TARGET_CRC32 (AARCH64_ISA_CRC)
#define TARGET_LSE (AARCH64_ISA_LSE)
#define TARGET_FP_F16INST (TARGET_FLOAT && AARCH64_ISA_F16)
#define TARGET_SIMD_F16INST (TARGET_SIMD && AARCH64_ISA_F16)
#define TARGET_DOTPROD (TARGET_SIMD && AARCH64_ISA_DOTPROD)
#define TARGET_SVE (AARCH64_ISA_SVE)
#define TARGET_ARMV8_3	(AARCH64_ISA_V8_3)
#ifndef TARGET_FIX_ERR_A53_835769_DEFAULT
#define TARGET_FIX_ERR_A53_835769_DEFAULT 0
#else
#undef TARGET_FIX_ERR_A53_835769_DEFAULT
#define TARGET_FIX_ERR_A53_835769_DEFAULT 1
#endif
#define TARGET_FIX_ERR_A53_835769	\
((aarch64_fix_a53_err835769 == 2)	\
? TARGET_FIX_ERR_A53_835769_DEFAULT : aarch64_fix_a53_err835769)
#ifndef TARGET_FIX_ERR_A53_843419_DEFAULT
#define TARGET_FIX_ERR_A53_843419_DEFAULT 0
#else
#undef TARGET_FIX_ERR_A53_843419_DEFAULT
#define TARGET_FIX_ERR_A53_843419_DEFAULT 1
#endif
#define TARGET_FIX_ERR_A53_843419	\
((aarch64_fix_a53_err843419 == 2)	\
? TARGET_FIX_ERR_A53_843419_DEFAULT : aarch64_fix_a53_err843419)
#define TARGET_SIMD_RDMA (TARGET_SIMD && AARCH64_ISA_RDMA)
#define FIXED_REGISTERS					\
{							\
0, 0, 0, 0,   0, 0, 0, 0,			\
0, 0, 0, 0,   0, 0, 0, 0,			\
0, 0, 0, 0,   0, 0, 0, 0,			\
0, 0, 0, 0,   0, 1, 0, 1,		\
0, 0, 0, 0,   0, 0, 0, 0,              \
0, 0, 0, 0,   0, 0, 0, 0,   		\
0, 0, 0, 0,   0, 0, 0, 0,            \
0, 0, 0, 0,   0, 0, 0, 0,            \
1, 1, 1, 1,				\
0, 0, 0, 0,   0, 0, 0, 0,              \
0, 0, 0, 0,   0, 0, 0, 0,             \
}
#define CALL_USED_REGISTERS				\
{							\
1, 1, 1, 1,   1, 1, 1, 1,			\
1, 1, 1, 1,   1, 1, 1, 1,			\
1, 1, 1, 0,   0, 0, 0, 0,			\
0, 0, 0, 0,   0, 1, 1, 1,		\
1, 1, 1, 1,   1, 1, 1, 1,			\
0, 0, 0, 0,   0, 0, 0, 0,			\
1, 1, 1, 1,   1, 1, 1, 1,            \
1, 1, 1, 1,   1, 1, 1, 1,            \
1, 1, 1, 1,				\
1, 1, 1, 1,   1, 1, 1, 1,			\
1, 1, 1, 1,   1, 1, 1, 1,			\
}
#define REGISTER_NAMES						\
{								\
"x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",	\
"x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",	\
"x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",	\
"x24", "x25", "x26", "x27", "x28", "x29", "x30", "sp",	\
"v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",	\
"v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",	\
"v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",	\
"v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",	\
"sfp", "ap",  "cc",  "vg",					\
"p0",  "p1",  "p2",  "p3",  "p4",  "p5",  "p6",  "p7",	\
"p8",  "p9",  "p10", "p11", "p12", "p13", "p14", "p15",	\
}
#define R_ALIASES(N) {"r" # N, R0_REGNUM + (N)}, \
{"w" # N, R0_REGNUM + (N)}
#define V_ALIASES(N) {"q" # N, V0_REGNUM + (N)}, \
{"d" # N, V0_REGNUM + (N)}, \
{"s" # N, V0_REGNUM + (N)}, \
{"h" # N, V0_REGNUM + (N)}, \
{"b" # N, V0_REGNUM + (N)}, \
{"z" # N, V0_REGNUM + (N)}
#define ADDITIONAL_REGISTER_NAMES \
{ R_ALIASES(0),  R_ALIASES(1),  R_ALIASES(2),  R_ALIASES(3),  \
R_ALIASES(4),  R_ALIASES(5),  R_ALIASES(6),  R_ALIASES(7),  \
R_ALIASES(8),  R_ALIASES(9),  R_ALIASES(10), R_ALIASES(11), \
R_ALIASES(12), R_ALIASES(13), R_ALIASES(14), R_ALIASES(15), \
R_ALIASES(16), R_ALIASES(17), R_ALIASES(18), R_ALIASES(19), \
R_ALIASES(20), R_ALIASES(21), R_ALIASES(22), R_ALIASES(23), \
R_ALIASES(24), R_ALIASES(25), R_ALIASES(26), R_ALIASES(27), \
R_ALIASES(28), R_ALIASES(29), R_ALIASES(30), {"wsp", R0_REGNUM + 31}, \
V_ALIASES(0),  V_ALIASES(1),  V_ALIASES(2),  V_ALIASES(3),  \
V_ALIASES(4),  V_ALIASES(5),  V_ALIASES(6),  V_ALIASES(7),  \
V_ALIASES(8),  V_ALIASES(9),  V_ALIASES(10), V_ALIASES(11), \
V_ALIASES(12), V_ALIASES(13), V_ALIASES(14), V_ALIASES(15), \
V_ALIASES(16), V_ALIASES(17), V_ALIASES(18), V_ALIASES(19), \
V_ALIASES(20), V_ALIASES(21), V_ALIASES(22), V_ALIASES(23), \
V_ALIASES(24), V_ALIASES(25), V_ALIASES(26), V_ALIASES(27), \
V_ALIASES(28), V_ALIASES(29), V_ALIASES(30), V_ALIASES(31)  \
}
#define EPILOGUE_USES(REGNO) \
(epilogue_completed && (REGNO) == LR_REGNUM)
#define EXIT_IGNORE_STACK	(cfun->calls_alloca)
#define STATIC_CHAIN_REGNUM		R18_REGNUM
#define HARD_FRAME_POINTER_REGNUM	R29_REGNUM
#define FRAME_POINTER_REGNUM		SFP_REGNUM
#define STACK_POINTER_REGNUM		SP_REGNUM
#define ARG_POINTER_REGNUM		AP_REGNUM
#define FIRST_PSEUDO_REGISTER		(P15_REGNUM + 1)
#define NUM_ARG_REGS			8
#define NUM_FP_ARG_REGS			8
#define HA_MAX_NUM_FLDS		4
#define AARCH64_DWARF_R0        0
#define AARCH64_DWARF_NUMBER_R 31
#define AARCH64_DWARF_SP       31
#define AARCH64_DWARF_VG       46
#define AARCH64_DWARF_P0       48
#define AARCH64_DWARF_V0       64
#define AARCH64_DWARF_NUMBER_V 32
#define DWARF_ALT_FRAME_RETURN_COLUMN   \
(AARCH64_DWARF_V0 + AARCH64_DWARF_NUMBER_V)
#define DWARF_FRAME_REGISTERS           (DWARF_ALT_FRAME_RETURN_COLUMN + 1)
#define DBX_REGISTER_NUMBER(REGNO)	aarch64_dbx_register_number (REGNO)
#undef DWARF_FRAME_REGNUM
#define DWARF_FRAME_REGNUM(REGNO)	DBX_REGISTER_NUMBER (REGNO)
#define DWARF_FRAME_RETURN_COLUMN	DWARF_FRAME_REGNUM (LR_REGNUM)
#define DWARF2_UNWIND_INFO 1
#define EH_RETURN_DATA_REGNO(N) \
((N) < 4 ? ((unsigned int) R0_REGNUM + (N)) : INVALID_REGNUM)
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE, GLOBAL) \
aarch64_asm_preferred_eh_data_format ((CODE), (GLOBAL))
#define ASM_DECLARE_FUNCTION_NAME(STR, NAME, DECL)	\
aarch64_declare_function_name (STR, NAME, DECL)
#define EH_RETURN_STACKADJ_RTX	gen_rtx_REG (Pmode, R4_REGNUM)
#define EH_RETURN_HANDLER_RTX  aarch64_eh_return_handler_rtx ()
#undef DONT_USE_BUILTIN_SETJMP
#define DONT_USE_BUILTIN_SETJMP 1
#define AARCH64_STRUCT_VALUE_REGNUM R8_REGNUM
#define GP_REGNUM_P(REGNO)						\
(((unsigned) (REGNO - R0_REGNUM)) <= (R30_REGNUM - R0_REGNUM))
#define FP_REGNUM_P(REGNO)			\
(((unsigned) (REGNO - V0_REGNUM)) <= (V31_REGNUM - V0_REGNUM))
#define FP_LO_REGNUM_P(REGNO)            \
(((unsigned) (REGNO - V0_REGNUM)) <= (V15_REGNUM - V0_REGNUM))
#define PR_REGNUM_P(REGNO)\
(((unsigned) (REGNO - P0_REGNUM)) <= (P15_REGNUM - P0_REGNUM))
#define PR_LO_REGNUM_P(REGNO)\
(((unsigned) (REGNO - P0_REGNUM)) <= (P7_REGNUM - P0_REGNUM))

enum reg_class
{
NO_REGS,
TAILCALL_ADDR_REGS,
GENERAL_REGS,
STACK_REG,
POINTER_REGS,
FP_LO_REGS,
FP_REGS,
POINTER_AND_FP_REGS,
PR_LO_REGS,
PR_HI_REGS,
PR_REGS,
ALL_REGS,
LIM_REG_CLASSES		
};
#define N_REG_CLASSES	((int) LIM_REG_CLASSES)
#define REG_CLASS_NAMES				\
{						\
"NO_REGS",					\
"TAILCALL_ADDR_REGS",				\
"GENERAL_REGS",				\
"STACK_REG",					\
"POINTER_REGS",				\
"FP_LO_REGS",					\
"FP_REGS",					\
"POINTER_AND_FP_REGS",			\
"PR_LO_REGS",					\
"PR_HI_REGS",					\
"PR_REGS",					\
"ALL_REGS"					\
}
#define REG_CLASS_CONTENTS						\
{									\
{ 0x00000000, 0x00000000, 0x00000000 },			\
{ 0x0004ffff, 0x00000000, 0x00000000 },	\
{ 0x7fffffff, 0x00000000, 0x00000003 },		\
{ 0x80000000, 0x00000000, 0x00000000 },			\
{ 0xffffffff, 0x00000000, 0x00000003 },		\
{ 0x00000000, 0x0000ffff, 0x00000000 },       	\
{ 0x00000000, 0xffffffff, 0x00000000 },       		\
{ 0xffffffff, 0xffffffff, 0x00000003 },	\
{ 0x00000000, 0x00000000, 0x00000ff0 },		\
{ 0x00000000, 0x00000000, 0x000ff000 },		\
{ 0x00000000, 0x00000000, 0x000ffff0 },			\
{ 0xffffffff, 0xffffffff, 0x000fffff }			\
}
#define REGNO_REG_CLASS(REGNO)	aarch64_regno_regclass (REGNO)
#define INDEX_REG_CLASS	GENERAL_REGS
#define BASE_REG_CLASS  POINTER_REGS
#define ELIMINABLE_REGS							\
{									\
{ ARG_POINTER_REGNUM,		STACK_POINTER_REGNUM		},	\
{ ARG_POINTER_REGNUM,		HARD_FRAME_POINTER_REGNUM	},	\
{ FRAME_POINTER_REGNUM,	STACK_POINTER_REGNUM		},	\
{ FRAME_POINTER_REGNUM,	HARD_FRAME_POINTER_REGNUM	},	\
}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET) \
(OFFSET) = aarch64_initial_elimination_offset (FROM, TO)
#include "config/aarch64/aarch64-opts.h"
enum target_cpus
{
#define AARCH64_CORE(NAME, INTERNAL_IDENT, SCHED, ARCH, FLAGS, COSTS, IMP, PART, VARIANT) \
TARGET_CPU_##INTERNAL_IDENT,
#include "aarch64-cores.def"
TARGET_CPU_generic
};
#ifndef TARGET_CPU_DEFAULT
#define TARGET_CPU_DEFAULT \
(TARGET_CPU_generic | (AARCH64_CPU_DEFAULT_FLAGS << 6))
#endif
#define ADJUST_INSN_LENGTH(insn, length)	\
do						\
{						\
if (aarch64_madd_needs_nop (insn))	\
length += 4;				\
} while (0)
#define FINAL_PRESCAN_INSN(INSN, OPVEC, NOPERANDS)	\
aarch64_final_prescan_insn (INSN);			\
extern enum aarch64_processor aarch64_tune;
#define INIT_EXPANDERS aarch64_init_expanders ()

#define STACK_GROWS_DOWNWARD	1
#define FRAME_GROWS_DOWNWARD	1
#define ACCUMULATE_OUTGOING_ARGS	1
#define FIRST_PARM_OFFSET(FNDECL) 0
#define LIBCALL_VALUE(MODE)  \
gen_rtx_REG (MODE, FLOAT_MODE_P (MODE) ? V0_REGNUM : R0_REGNUM)
#define DEFAULT_PCC_STRUCT_RETURN 0
#ifdef HAVE_POLY_INT_H
struct GTY (()) aarch64_frame
{
HOST_WIDE_INT reg_offset[FIRST_PSEUDO_REGISTER];
HOST_WIDE_INT saved_varargs_size;
HOST_WIDE_INT saved_regs_size;
poly_int64 locals_offset;
poly_int64 hard_fp_offset;
poly_int64 frame_size;
poly_int64 initial_adjust;
HOST_WIDE_INT callee_adjust;
poly_int64 callee_offset;
poly_int64 final_adjust;
bool emit_frame_chain;
unsigned wb_candidate1;
unsigned wb_candidate2;
bool laid_out;
};
typedef struct GTY (()) machine_function
{
struct aarch64_frame frame;
bool reg_is_wrapped_separately[LAST_SAVED_REGNUM];
} machine_function;
#endif
enum aarch64_abi_type
{
AARCH64_ABI_LP64 = 0,
AARCH64_ABI_ILP32 = 1
};
#ifndef AARCH64_ABI_DEFAULT
#define AARCH64_ABI_DEFAULT AARCH64_ABI_LP64
#endif
#define TARGET_ILP32	(aarch64_abi & AARCH64_ABI_ILP32)
enum arm_pcs
{
ARM_PCS_AAPCS64,		
ARM_PCS_UNKNOWN
};
#ifdef GENERATOR_FILE
#define MACHMODE int
#else
#include "insn-modes.h"
#define MACHMODE machine_mode
#endif
#ifndef USED_FOR_TARGET
typedef struct
{
enum arm_pcs pcs_variant;
int aapcs_arg_processed;	
int aapcs_ncrn;		
int aapcs_nextncrn;		
int aapcs_nvrn;		
int aapcs_nextnvrn;		
rtx aapcs_reg;		
MACHMODE aapcs_vfp_rmode;
int aapcs_stack_words;	
int aapcs_stack_size;		
} CUMULATIVE_ARGS;
#endif
#define BLOCK_REG_PADDING(MODE, TYPE, FIRST) \
(aarch64_pad_reg_upward (MODE, TYPE, FIRST) ? PAD_UPWARD : PAD_DOWNWARD)
#define PAD_VARARGS_DOWN	0
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS) \
aarch64_init_cumulative_args (&(CUM), FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS)
#define FUNCTION_ARG_REGNO_P(REGNO) \
aarch64_function_arg_regno_p(REGNO)

#define HAVE_POST_INCREMENT	1
#define HAVE_PRE_INCREMENT	1
#define HAVE_POST_DECREMENT	1
#define HAVE_PRE_DECREMENT	1
#define HAVE_POST_MODIFY_DISP	1
#define HAVE_PRE_MODIFY_DISP	1
#define MAX_REGS_PER_ADDRESS	2
#define CONSTANT_ADDRESS_P(X)		aarch64_constant_address_p(X)
#define REGNO_OK_FOR_BASE_P(REGNO)	\
aarch64_regno_ok_for_base_p (REGNO, true)
#define REGNO_OK_FOR_INDEX_P(REGNO) \
aarch64_regno_ok_for_index_p (REGNO, true)
#define LEGITIMATE_PIC_OPERAND_P(X) \
aarch64_legitimate_pic_operand_p (X)
#define CASE_VECTOR_MODE Pmode
#define DEFAULT_SIGNED_CHAR 0
#define MAX_FIXED_MODE_SIZE GET_MODE_BITSIZE (TImode)
#define MOVE_MAX (UNITS_PER_WORD * 2)
#define AARCH64_CALL_RATIO 8
#define MOVE_RATIO(speed) \
(!STRICT_ALIGNMENT ? 2 : (((speed) ? 15 : AARCH64_CALL_RATIO) / 2))
#define CLEAR_RATIO(speed) \
((speed) ? 15 : AARCH64_CALL_RATIO)
#define SET_RATIO(speed) \
((speed) ? 15 : AARCH64_CALL_RATIO - 2)
#define USE_LOAD_POST_INCREMENT(MODE)   0
#define USE_LOAD_POST_DECREMENT(MODE)   0
#define USE_LOAD_PRE_INCREMENT(MODE)    0
#define USE_LOAD_PRE_DECREMENT(MODE)    0
#define USE_STORE_POST_INCREMENT(MODE)  0
#define USE_STORE_POST_DECREMENT(MODE)  0
#define USE_STORE_PRE_INCREMENT(MODE)   0
#define USE_STORE_PRE_DECREMENT(MODE)   0
#define WORD_REGISTER_OPERATIONS 0
#define LOAD_EXTEND_OP(MODE) ZERO_EXTEND
#define STRICT_ALIGNMENT		TARGET_STRICT_ALIGN
#define SLOW_BYTE_ACCESS		0
#define NO_FUNCTION_CSE	1
#define Pmode		DImode
#define POINTERS_EXTEND_UNSIGNED 1
#define FUNCTION_MODE	Pmode
#define SELECT_CC_MODE(OP, X, Y)	aarch64_select_cc_mode (OP, X, Y)
#define REVERSIBLE_CC_MODE(MODE) 1
#define REVERSE_CONDITION(CODE, MODE)		\
(((MODE) == CCFPmode || (MODE) == CCFPEmode)	\
? reverse_condition_maybe_unordered (CODE)	\
: reverse_condition (CODE))
#define CLZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) \
((VALUE) = GET_MODE_UNIT_BITSIZE (MODE), 2)
#define CTZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) \
((VALUE) = GET_MODE_UNIT_BITSIZE (MODE), 2)
#define INCOMING_RETURN_ADDR_RTX gen_rtx_REG (Pmode, LR_REGNUM)
#define RETURN_ADDR_RTX aarch64_return_addr
#define TRAMPOLINE_SIZE	(TARGET_ILP32 ? 24 : 32)
#define TRAMPOLINE_ALIGNMENT 64
#define TRAMPOLINE_SECTION text_section
#define BRANCH_COST(SPEED_P, PREDICTABLE_P) \
(aarch64_branch_cost (SPEED_P, PREDICTABLE_P))

#define CASE_VECTOR_PC_RELATIVE	1
#define CASE_VECTOR_SHORTEN_MODE(min, max, body)	\
((min < -0x1fff0 || max > 0x1fff0) ? SImode		\
: (min < -0x1f0 || max > 0x1f0) ? HImode		\
: QImode)
#define ADDR_VEC_ALIGN(JUMPTABLE) 0
#define MCOUNT_NAME "_mcount"
#define NO_PROFILE_COUNTERS 1
#define PROFILE_HOOK(LABEL)						\
{									\
rtx fun, lr;							\
lr = get_hard_reg_initial_val (Pmode, LR_REGNUM);			\
fun = gen_rtx_SYMBOL_REF (Pmode, MCOUNT_NAME);			\
emit_library_call (fun, LCT_NORMAL, VOIDmode, lr, Pmode);		\
}
#define FUNCTION_PROFILER(STREAM, LABELNO) do { } while (0)
#undef ASM_APP_ON
#undef ASM_APP_OFF
#define ASM_APP_ON	"\t" ASM_COMMENT_START " Start of user assembly\n"
#define ASM_APP_OFF	"\t" ASM_COMMENT_START " End of user assembly\n"
#define CONSTANT_POOL_BEFORE_FUNCTION 0
#define CLEAR_INSN_CACHE(beg, end)				\
extern void  __aarch64_sync_cache_range (void *, void *);	\
__aarch64_sync_cache_range (beg, end)
#define SHIFT_COUNT_TRUNCATED (!TARGET_SIMD)
#define HARD_REGNO_CALLER_SAVE_MODE(REGNO, NREGS, MODE) \
aarch64_hard_regno_caller_save_mode ((REGNO), (NREGS), (MODE))
#undef SWITCHABLE_TARGET
#define SWITCHABLE_TARGET 1
#define TARGET_TLS_DESC (aarch64_tls_dialect == TLS_DESCRIPTORS)
extern enum aarch64_code_model aarch64_cmodel;
#define HAS_LONG_COND_BRANCH				\
(aarch64_cmodel == AARCH64_CMODEL_TINY		\
|| aarch64_cmodel == AARCH64_CMODEL_TINY_PIC)
#define HAS_LONG_UNCOND_BRANCH				\
(aarch64_cmodel == AARCH64_CMODEL_TINY		\
|| aarch64_cmodel == AARCH64_CMODEL_TINY_PIC)
#define TARGET_SUPPORTS_WIDE_INT 1
#define AARCH64_VALID_SIMD_DREG_MODE(MODE) \
((MODE) == V2SImode || (MODE) == V4HImode || (MODE) == V8QImode \
|| (MODE) == V2SFmode || (MODE) == V4HFmode || (MODE) == DImode \
|| (MODE) == DFmode)
#define AARCH64_VALID_SIMD_QREG_MODE(MODE) \
((MODE) == V4SImode || (MODE) == V8HImode || (MODE) == V16QImode \
|| (MODE) == V4SFmode || (MODE) == V8HFmode || (MODE) == V2DImode \
|| (MODE) == V2DFmode)
#define ENDIAN_LANE_N(NUNITS, N) \
(BYTES_BIG_ENDIAN ? NUNITS - 1 - N : N)
#define OPTION_DEFAULT_SPECS				\
{"arch", "%{!march=*:%{!mcpu=*:-march=%(VALUE)}}" },	\
{"cpu",  "%{!march=*:%{!mcpu=*:-mcpu=%(VALUE)}}" },
#define MCPU_TO_MARCH_SPEC \
" %{mcpu=*:-march=%:rewrite_mcpu(%{mcpu=*:%*})}"
extern const char *aarch64_rewrite_mcpu (int argc, const char **argv);
#define MCPU_TO_MARCH_SPEC_FUNCTIONS \
{ "rewrite_mcpu", aarch64_rewrite_mcpu },
#if defined(__aarch64__)
extern const char *host_detect_local_cpu (int argc, const char **argv);
#define HAVE_LOCAL_CPU_DETECT
# define EXTRA_SPEC_FUNCTIONS						\
{ "local_cpu_detect", host_detect_local_cpu },			\
MCPU_TO_MARCH_SPEC_FUNCTIONS
# define MCPU_MTUNE_NATIVE_SPECS					\
" %{march=native:%<march=native %:local_cpu_detect(arch)}"		\
" %{mcpu=native:%<mcpu=native %:local_cpu_detect(cpu)}"		\
" %{mtune=native:%<mtune=native %:local_cpu_detect(tune)}"
#else
# define MCPU_MTUNE_NATIVE_SPECS ""
# define EXTRA_SPEC_FUNCTIONS MCPU_TO_MARCH_SPEC_FUNCTIONS
#endif
#define ASM_CPU_SPEC \
MCPU_TO_MARCH_SPEC
#define EXTRA_SPECS						\
{ "asm_cpu_spec",		ASM_CPU_SPEC }
#define ASM_OUTPUT_POOL_EPILOGUE  aarch64_asm_output_pool_epilogue
extern tree aarch64_fp16_type_node;
extern tree aarch64_fp16_ptr_type_node;
#define LIBGCC2_UNWIND_ATTRIBUTE \
__attribute__((optimize ("no-omit-frame-pointer")))
#ifndef USED_FOR_TARGET
extern poly_uint16 aarch64_sve_vg;
#define BITS_PER_SVE_VECTOR (poly_uint16 (aarch64_sve_vg * 64))
#define BYTES_PER_SVE_VECTOR (poly_uint16 (aarch64_sve_vg * 8))
#define BYTES_PER_SVE_PRED aarch64_sve_vg
#define SVE_BYTE_MODE VNx16QImode
#define MAX_COMPILE_TIME_VEC_BYTES (256 * 4)
#endif
#define REGMODE_NATURAL_SIZE(MODE) aarch64_regmode_natural_size (MODE)
#endif 
