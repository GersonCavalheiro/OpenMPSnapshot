#ifndef _S390_H
#define _S390_H
enum processor_flags
{
PF_IEEE_FLOAT = 1,
PF_ZARCH = 2,
PF_LONG_DISPLACEMENT = 4,
PF_EXTIMM = 8,
PF_DFP = 16,
PF_Z10 = 32,
PF_Z196 = 64,
PF_ZEC12 = 128,
PF_TX = 256,
PF_Z13 = 512,
PF_VX = 1024,
PF_ARCH12 = 2048,
PF_VXE = 4096
};
#define s390_tune_attr ((enum attr_cpu)(s390_tune > PROCESSOR_2964_Z13 ? PROCESSOR_2964_Z13 : s390_tune ))
#define TARGET_CPU_IEEE_FLOAT \
(s390_arch_flags & PF_IEEE_FLOAT)
#define TARGET_CPU_IEEE_FLOAT_P(opts) \
(opts->x_s390_arch_flags & PF_IEEE_FLOAT)
#define TARGET_CPU_ZARCH \
(s390_arch_flags & PF_ZARCH)
#define TARGET_CPU_ZARCH_P(opts) \
(opts->x_s390_arch_flags & PF_ZARCH)
#define TARGET_CPU_LONG_DISPLACEMENT \
(s390_arch_flags & PF_LONG_DISPLACEMENT)
#define TARGET_CPU_LONG_DISPLACEMENT_P(opts) \
(opts->x_s390_arch_flags & PF_LONG_DISPLACEMENT)
#define TARGET_CPU_EXTIMM \
(s390_arch_flags & PF_EXTIMM)
#define TARGET_CPU_EXTIMM_P(opts) \
(opts->x_s390_arch_flags & PF_EXTIMM)
#define TARGET_CPU_DFP \
(s390_arch_flags & PF_DFP)
#define TARGET_CPU_DFP_P(opts) \
(opts->x_s390_arch_flags & PF_DFP)
#define TARGET_CPU_Z10 \
(s390_arch_flags & PF_Z10)
#define TARGET_CPU_Z10_P(opts) \
(opts->x_s390_arch_flags & PF_Z10)
#define TARGET_CPU_Z196 \
(s390_arch_flags & PF_Z196)
#define TARGET_CPU_Z196_P(opts) \
(opts->x_s390_arch_flags & PF_Z196)
#define TARGET_CPU_ZEC12 \
(s390_arch_flags & PF_ZEC12)
#define TARGET_CPU_ZEC12_P(opts) \
(opts->x_s390_arch_flags & PF_ZEC12)
#define TARGET_CPU_HTM \
(s390_arch_flags & PF_TX)
#define TARGET_CPU_HTM_P(opts) \
(opts->x_s390_arch_flags & PF_TX)
#define TARGET_CPU_Z13 \
(s390_arch_flags & PF_Z13)
#define TARGET_CPU_Z13_P(opts) \
(opts->x_s390_arch_flags & PF_Z13)
#define TARGET_CPU_VX \
(s390_arch_flags & PF_VX)
#define TARGET_CPU_VX_P(opts) \
(opts->x_s390_arch_flags & PF_VX)
#define TARGET_CPU_ARCH12 \
(s390_arch_flags & PF_ARCH12)
#define TARGET_CPU_ARCH12_P(opts) \
(opts->x_s390_arch_flags & PF_ARCH12)
#define TARGET_CPU_VXE \
(s390_arch_flags & PF_VXE)
#define TARGET_CPU_VXE_P(opts) \
(opts->x_s390_arch_flags & PF_VXE)
#define TARGET_HARD_FLOAT_P(opts) (!TARGET_SOFT_FLOAT_P(opts))
#define TARGET_LONG_DISPLACEMENT \
(TARGET_ZARCH && TARGET_CPU_LONG_DISPLACEMENT)
#define TARGET_LONG_DISPLACEMENT_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) \
&& TARGET_CPU_LONG_DISPLACEMENT_P (opts))
#define TARGET_EXTIMM \
(TARGET_ZARCH && TARGET_CPU_EXTIMM)
#define TARGET_EXTIMM_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_EXTIMM_P (opts))
#define TARGET_DFP \
(TARGET_ZARCH && TARGET_CPU_DFP && TARGET_HARD_FLOAT)
#define TARGET_DFP_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_DFP_P (opts) \
&& TARGET_HARD_FLOAT_P (opts->x_target_flags))
#define TARGET_Z10 \
(TARGET_ZARCH && TARGET_CPU_Z10)
#define TARGET_Z10_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_Z10_P (opts))
#define TARGET_Z196 \
(TARGET_ZARCH && TARGET_CPU_Z196)
#define TARGET_Z196_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_Z196_P (opts))
#define TARGET_ZEC12 \
(TARGET_ZARCH && TARGET_CPU_ZEC12)
#define TARGET_ZEC12_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_ZEC12_P (opts))
#define TARGET_HTM (TARGET_OPT_HTM)
#define TARGET_HTM_P(opts) (TARGET_OPT_HTM_P (opts->x_target_flags))
#define TARGET_Z13 \
(TARGET_ZARCH && TARGET_CPU_Z13)
#define TARGET_Z13_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_Z13_P (opts))
#define TARGET_VX \
(TARGET_ZARCH && TARGET_CPU_VX && TARGET_OPT_VX && TARGET_HARD_FLOAT)
#define TARGET_VX_P(opts) \
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_VX_P (opts) \
&& TARGET_OPT_VX_P (opts->x_target_flags) \
&& TARGET_HARD_FLOAT_P (opts->x_target_flags))
#define TARGET_ARCH12 (TARGET_ZARCH && TARGET_CPU_ARCH12)
#define TARGET_ARCH12_P(opts)						\
(TARGET_ZARCH_P (opts->x_target_flags) && TARGET_CPU_ARCH12_P (opts))
#define TARGET_VXE				\
(TARGET_VX && TARGET_CPU_VXE)
#define TARGET_VXE_P(opts)						\
(TARGET_VX_P (opts) && TARGET_CPU_VXE_P (opts))
#ifdef HAVE_AS_MACHINE_MACHINEMODE
#define S390_USE_TARGET_ATTRIBUTE 1
#else
#define S390_USE_TARGET_ATTRIBUTE 0
#endif
#ifdef HAVE_AS_ARCHITECTURE_MODIFIERS
#define S390_USE_ARCHITECTURE_MODIFIERS 1
#else
#define S390_USE_ARCHITECTURE_MODIFIERS 0
#endif
#if S390_USE_TARGET_ATTRIBUTE
#define SWITCHABLE_TARGET 1
#endif
#define TARGET_SUPPORTS_WIDE_INT 1
#define TARGET_VX_ABI TARGET_VX
#define TARGET_AVOID_CMP_AND_BRANCH (s390_tune == PROCESSOR_2817_Z196)
#ifndef TARGET_TPF_PROFILING
#define TARGET_TPF_PROFILING 0
#endif
#define TARGET_TPF 0
#define TARGET_CPU_CPP_BUILTINS() s390_cpu_cpp_builtins (pfile)
#ifdef DEFAULT_TARGET_64BIT
#define TARGET_DEFAULT     (MASK_64BIT | MASK_ZARCH | MASK_HARD_DFP	\
| MASK_OPT_HTM | MASK_OPT_VX)
#else
#define TARGET_DEFAULT             0
#endif
#define OPTION_DEFAULT_SPECS 					\
{ "mode", "%{!mesa:%{!mzarch:-m%(VALUE)}}" },			\
{ "arch", "%{!march=*:-march=%(VALUE)}" },			\
{ "tune", "%{!mtune=*:%{!march=*:-mtune=%(VALUE)}}" }
#ifdef __s390__
extern const char *s390_host_detect_local_cpu (int argc, const char **argv);
# define EXTRA_SPEC_FUNCTIONS \
{ "local_cpu_detect", s390_host_detect_local_cpu },
#define MARCH_MTUNE_NATIVE_SPECS				\
"%{mtune=native:%<mtune=native %:local_cpu_detect(tune)} "	\
"%{march=native:%<march=native"				\
" %:local_cpu_detect(arch %{mesa|mzarch:mesa_mzarch})}"
#else
# define MARCH_MTUNE_NATIVE_SPECS ""
#endif
#ifdef DEFAULT_TARGET_64BIT
#define S390_TARGET_BITS_STRING "64"
#else
#define S390_TARGET_BITS_STRING "31"
#endif
#define DRIVER_SELF_SPECS					\
MARCH_MTUNE_NATIVE_SPECS,					\
"%{!m31:%{!m64:-m" S390_TARGET_BITS_STRING "}}",		\
"%{!mesa:%{!mzarch:%{m31:-mesa}%{m64:-mzarch}}}",		\
"%{!march=*:-march=z900}"
#define S390_TDC_POSITIVE_ZERO                     (1 << 11)
#define S390_TDC_NEGATIVE_ZERO                     (1 << 10)
#define S390_TDC_POSITIVE_NORMALIZED_BFP_NUMBER    (1 << 9)
#define S390_TDC_NEGATIVE_NORMALIZED_BFP_NUMBER    (1 << 8)
#define S390_TDC_POSITIVE_DENORMALIZED_BFP_NUMBER  (1 << 7)
#define S390_TDC_NEGATIVE_DENORMALIZED_BFP_NUMBER  (1 << 6)
#define S390_TDC_POSITIVE_INFINITY                 (1 << 5)
#define S390_TDC_NEGATIVE_INFINITY                 (1 << 4)
#define S390_TDC_POSITIVE_QUIET_NAN                (1 << 3)
#define S390_TDC_NEGATIVE_QUIET_NAN                (1 << 2)
#define S390_TDC_POSITIVE_SIGNALING_NAN            (1 << 1)
#define S390_TDC_NEGATIVE_SIGNALING_NAN            (1 << 0)
#define S390_TDC_POSITIVE_DENORMALIZED_DFP_NUMBER (1 << 9)
#define S390_TDC_NEGATIVE_DENORMALIZED_DFP_NUMBER (1 << 8)
#define S390_TDC_POSITIVE_NORMALIZED_DFP_NUMBER   (1 << 7)
#define S390_TDC_NEGATIVE_NORMALIZED_DFP_NUMBER   (1 << 6)
#define S390_TDC_SIGNBIT_SET (S390_TDC_NEGATIVE_ZERO \
| S390_TDC_NEGATIVE_NORMALIZED_BFP_NUMBER \
| S390_TDC_NEGATIVE_DENORMALIZED_BFP_NUMBER\
| S390_TDC_NEGATIVE_INFINITY \
| S390_TDC_NEGATIVE_QUIET_NAN \
| S390_TDC_NEGATIVE_SIGNALING_NAN )
#define S390_TDC_INFINITY (S390_TDC_POSITIVE_INFINITY \
| S390_TDC_NEGATIVE_INFINITY )
#define BITS_BIG_ENDIAN 1
#define BYTES_BIG_ENDIAN 1
#define WORDS_BIG_ENDIAN 1
#define STACK_SIZE_MODE (Pmode)
#ifndef IN_LIBGCC2
#define UNITS_PER_WORD (TARGET_ZARCH ? 8 : 4)
#define UNITS_PER_LONG (TARGET_64BIT ? 8 : 4)
#define MIN_UNITS_PER_WORD 4
#define MAX_BITS_PER_WORD 64
#else
#ifdef __s390x__
#define UNITS_PER_WORD 8
#else
#define UNITS_PER_WORD 4
#endif
#endif
#define POINTER_SIZE (TARGET_64BIT ? 64 : 32)
#define PARM_BOUNDARY (TARGET_64BIT ? 64 : 32)
#define STACK_BOUNDARY 64
#define FUNCTION_BOUNDARY 64
#define BIGGEST_ALIGNMENT 64
#define EMPTY_FIELD_BOUNDARY 32
#define DATA_ABI_ALIGNMENT(TYPE, ALIGN) (ALIGN) < 16 ? 16 : (ALIGN)
#define STRICT_ALIGNMENT 0
#define STACK_SAVEAREA_MODE(LEVEL)					\
((LEVEL) == SAVE_FUNCTION ? VOIDmode					\
: (LEVEL) == SAVE_NONLOCAL ? (TARGET_64BIT ? OImode : TImode) : Pmode)
#define SHORT_TYPE_SIZE 16
#define INT_TYPE_SIZE 32
#define LONG_TYPE_SIZE (TARGET_64BIT ? 64 : 32)
#define LONG_LONG_TYPE_SIZE 64
#define FLOAT_TYPE_SIZE 32
#define DOUBLE_TYPE_SIZE 64
#define LONG_DOUBLE_TYPE_SIZE (TARGET_LONG_DOUBLE_128 ? 128 : 64)
#define WIDEST_HARDWARE_FP_SIZE 64
#define DEFAULT_SIGNED_CHAR 0
#define FIRST_PSEUDO_REGISTER 54
#define GENERAL_REGNO_P(N)	((int)(N) >= 0 && (N) < 16)
#define ADDR_REGNO_P(N)		((N) >= 1 && (N) < 16)
#define FP_REGNO_P(N)		((N) >= 16 && (N) < 32)
#define CC_REGNO_P(N)		((N) == 33)
#define FRAME_REGNO_P(N)	((N) == 32 || (N) == 34 || (N) == 35)
#define ACCESS_REGNO_P(N)	((N) == 36 || (N) == 37)
#define VECTOR_NOFP_REGNO_P(N)  ((N) >= 38 && (N) <= 53)
#define VECTOR_REGNO_P(N)       (FP_REGNO_P (N) || VECTOR_NOFP_REGNO_P (N))
#define GENERAL_REG_P(X)	(REG_P (X) && GENERAL_REGNO_P (REGNO (X)))
#define ADDR_REG_P(X)		(REG_P (X) && ADDR_REGNO_P (REGNO (X)))
#define FP_REG_P(X)		(REG_P (X) && FP_REGNO_P (REGNO (X)))
#define CC_REG_P(X)		(REG_P (X) && CC_REGNO_P (REGNO (X)))
#define FRAME_REG_P(X)		(REG_P (X) && FRAME_REGNO_P (REGNO (X)))
#define ACCESS_REG_P(X)		(REG_P (X) && ACCESS_REGNO_P (REGNO (X)))
#define VECTOR_NOFP_REG_P(X)    (REG_P (X) && VECTOR_NOFP_REGNO_P (REGNO (X)))
#define VECTOR_REG_P(X)         (REG_P (X) && VECTOR_REGNO_P (REGNO (X)))
#define FIXED_REGISTERS				\
{ 0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 1, 1, 1,					\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
1, 1, 1, 1,					\
1, 1,						\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0 }
#define CALL_USED_REGISTERS			\
{ 1, 1, 1, 1, 					\
1, 1, 0, 0, 					\
0, 0, 0, 0, 					\
0, 1, 1, 1,					\
1, 1, 1, 1, 					\
1, 1, 1, 1, 					\
1, 1, 1, 1, 					\
1, 1, 1, 1, 					\
1, 1, 1, 1,					\
1, 1,					        \
1, 1, 1, 1, 					\
1, 1, 1, 1,					\
1, 1, 1, 1, 					\
1, 1, 1, 1 }
#define CALL_REALLY_USED_REGISTERS		\
{ 1, 1, 1, 1, 				\
1, 1, 0, 0, 					\
0, 0, 0, 0, 					\
0, 0, 0, 0,					\
1, 1, 1, 1, 		\
1, 1, 1, 1, 					\
1, 1, 1, 1, 					\
1, 1, 1, 1, 					\
1, 1, 1, 1,		\
0, 0,			        \
1, 1, 1, 1, 		\
1, 1, 1, 1,					\
1, 1, 1, 1, 		\
1, 1, 1, 1 }
#define REG_ALLOC_ORDER							\
{  1, 2, 3, 4, 5, 0, 12, 11, 10, 9, 8, 7, 6, 14, 13,			\
16, 17, 18, 19, 20, 21, 22, 23,					\
24, 25, 26, 27, 28, 29, 30, 31,					\
38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 	\
15, 32, 33, 34, 35, 36, 37 }
#define HARD_REGNO_RENAME_OK(FROM, TO)          \
s390_hard_regno_rename_ok ((FROM), (TO))
#define CLASS_MAX_NREGS(CLASS, MODE)   					\
s390_class_max_nregs ((CLASS), (MODE))
#define REVERSIBLE_CC_MODE(MODE)				\
((MODE) == CCVIALLmode || (MODE) == CCVIANYmode		\
|| (MODE) == CCVFALLmode || (MODE) == CCVFANYmode)
#define REVERSE_CONDITION(CODE, MODE) s390_reverse_condition (MODE, CODE)
enum reg_class
{
NO_REGS, CC_REGS, ADDR_REGS, GENERAL_REGS, ACCESS_REGS,
ADDR_CC_REGS, GENERAL_CC_REGS,
FP_REGS, ADDR_FP_REGS, GENERAL_FP_REGS,
VEC_REGS, ADDR_VEC_REGS, GENERAL_VEC_REGS,
ALL_REGS, LIM_REG_CLASSES
};
#define N_REG_CLASSES (int) LIM_REG_CLASSES
#define REG_CLASS_NAMES							\
{ "NO_REGS", "CC_REGS", "ADDR_REGS", "GENERAL_REGS", "ACCESS_REGS",	\
"ADDR_CC_REGS", "GENERAL_CC_REGS",					\
"FP_REGS", "ADDR_FP_REGS", "GENERAL_FP_REGS",				\
"VEC_REGS", "ADDR_VEC_REGS", "GENERAL_VEC_REGS",			\
"ALL_REGS" }
#define REG_CLASS_CONTENTS				\
{							\
{ 0x00000000, 0x00000000 },			\
{ 0x00000000, 0x00000002 },			\
{ 0x0000fffe, 0x0000000d },			\
{ 0x0000ffff, 0x0000000d },		\
{ 0x00000000, 0x00000030 },		\
{ 0x0000fffe, 0x0000000f },		\
{ 0x0000ffff, 0x0000000f },		\
{ 0xffff0000, 0x00000000 },			\
{ 0xfffffffe, 0x0000000d },		\
{ 0xffffffff, 0x0000000d },		\
{ 0xffff0000, 0x003fffc0 },			\
{ 0xfffffffe, 0x003fffcd },		\
{ 0xffffffff, 0x003fffcd },		\
{ 0xffffffff, 0x003fffff },			\
}
#define IRA_HARD_REGNO_ADD_COST_MULTIPLIER(regno)	\
((regno) != BASE_REGNUM ? 0.0 : 0.5)
extern const enum reg_class regclass_map[FIRST_PSEUDO_REGISTER];
#define REGNO_REG_CLASS(REGNO) (regclass_map[REGNO])
#define INDEX_REG_CLASS ADDR_REGS
#define BASE_REG_CLASS ADDR_REGS
#define REGNO_OK_FOR_INDEX_P(REGNO)					\
(((REGNO) < FIRST_PSEUDO_REGISTER 					\
&& REGNO_REG_CLASS ((REGNO)) == ADDR_REGS) 			\
|| ADDR_REGNO_P (reg_renumber[REGNO]))
#define REGNO_OK_FOR_BASE_P(REGNO) REGNO_OK_FOR_INDEX_P (REGNO)
#define STACK_GROWS_DOWNWARD 1
#define FRAME_GROWS_DOWNWARD 1
#define STACK_POINTER_OFFSET (TARGET_64BIT ? 160 : 96)
#define STACK_DYNAMIC_OFFSET(FUNDECL) \
(STACK_POINTER_OFFSET + crtl->outgoing_args_size)
#define FIRST_PARM_OFFSET(FNDECL) 0
#define INITIAL_FRAME_ADDRESS_RTX                                             \
(plus_constant (Pmode, arg_pointer_rtx, -STACK_POINTER_OFFSET))
#define DYNAMIC_CHAIN_ADDRESS(FRAME)                                          \
(TARGET_PACKED_STACK ?                                                      \
plus_constant (Pmode, (FRAME),					      \
STACK_POINTER_OFFSET - UNITS_PER_LONG) : (FRAME))
#define FRAME_ADDR_RTX(FRAME)			\
DYNAMIC_CHAIN_ADDRESS ((FRAME))
#define RETURN_ADDR_RTX(COUNT, FRAME)					      \
s390_return_addr_rtx ((COUNT), DYNAMIC_CHAIN_ADDRESS ((FRAME)))
#define MASK_RETURN_ADDR (TARGET_64BIT ? constm1_rtx : GEN_INT (0x7fffffff))
#define INCOMING_RETURN_ADDR_RTX  gen_rtx_REG (Pmode, RETURN_REGNUM)
#define INCOMING_FRAME_SP_OFFSET STACK_POINTER_OFFSET
#define DWARF_FRAME_RETURN_COLUMN  14
#define EH_RETURN_DATA_REGNO(N) ((N) < 4 ? (N) + 6 : INVALID_REGNUM)
#define EH_RETURN_HANDLER_RTX gen_rtx_MEM (Pmode, return_address_pointer_rtx)
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE, GLOBAL)			    \
(flag_pic								    \
? ((GLOBAL) ? DW_EH_PE_indirect : 0) | DW_EH_PE_pcrel | DW_EH_PE_sdata4 \
: DW_EH_PE_absptr)
#define DWARF_CIE_DATA_ALIGNMENT (-UNITS_PER_LONG)
#define DWARF2_ASM_LINE_DEBUG_INFO 1
#define DBX_REGISTER_NUMBER(regno)				\
(((regno) >= 38 && (regno) <= 53) ? (regno) + 30 : (regno))
#define STACK_POINTER_REGNUM 15
#define FRAME_POINTER_REGNUM 34
#define HARD_FRAME_POINTER_REGNUM 11
#define ARG_POINTER_REGNUM 32
#define RETURN_ADDRESS_POINTER_REGNUM 35
#define STATIC_CHAIN_REGNUM 0
#define DWARF_FRAME_REGISTERS 34
#define ELIMINABLE_REGS						\
{{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM },		\
{ FRAME_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM },		\
{ ARG_POINTER_REGNUM, STACK_POINTER_REGNUM },			\
{ ARG_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM },		\
{ RETURN_ADDRESS_POINTER_REGNUM, STACK_POINTER_REGNUM },	\
{ RETURN_ADDRESS_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM },	\
{ BASE_REGNUM, BASE_REGNUM }}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET) \
(OFFSET) = s390_initial_elimination_offset ((FROM), (TO))
#define ACCUMULATE_OUTGOING_ARGS 1
typedef struct s390_arg_structure
{
int gprs;			
int fprs;			
int vrs;                      
}
CUMULATIVE_ARGS;
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, NN, N_NAMED_ARGS) \
((CUM).gprs=0, (CUM).fprs=0, (CUM).vrs=0)
#define FIRST_VEC_ARG_REGNO 46
#define LAST_VEC_ARG_REGNO 53
#define FUNCTION_ARG_REGNO_P(N)						\
(((N) >=2 && (N) < 7) || (N) == 16 || (N) == 17			\
|| (TARGET_64BIT && ((N) == 18 || (N) == 19))			\
|| (TARGET_VX && ((N) >= FIRST_VEC_ARG_REGNO && (N) <= LAST_VEC_ARG_REGNO)))
#define FUNCTION_VALUE_REGNO_P(N)		\
((N) == 2 || (N) == 16			\
|| (TARGET_VX && (N) == FIRST_VEC_ARG_REGNO))
#define EXIT_IGNORE_STACK       1
#define FUNCTION_PROFILER(FILE, LABELNO) 			\
s390_function_profiler ((FILE), ((LABELNO)))
#define PROFILE_BEFORE_PROLOGUE 1
#define TRAMPOLINE_SIZE		(TARGET_64BIT ? 32 : 16)
#define TRAMPOLINE_ALIGNMENT	BITS_PER_WORD
#define CONSTANT_ADDRESS_P(X) 0
#define MAX_REGS_PER_ADDRESS 2
#define TARGET_MEM_CONSTRAINT 'e'
#define LEGITIMIZE_RELOAD_ADDRESS(AD, MODE, OPNUM, TYPE, IND, WIN)	\
do {									\
rtx new_rtx = legitimize_reload_address ((AD), (MODE),		\
(OPNUM), (int)(TYPE));	\
if (new_rtx)							\
{									\
(AD) = new_rtx;							\
goto WIN;							\
}									\
} while (0)
#define SYMBOLIC_CONST(X)						\
(GET_CODE (X) == SYMBOL_REF						\
|| GET_CODE (X) == LABEL_REF						\
|| (GET_CODE (X) == CONST && symbolic_reference_mentioned_p (X)))
#define TLS_SYMBOLIC_CONST(X)						\
((GET_CODE (X) == SYMBOL_REF && tls_symbolic_operand (X))		\
|| (GET_CODE (X) == CONST && tls_symbolic_reference_mentioned_p (X)))
#define SELECT_CC_MODE(OP, X, Y) s390_select_ccmode ((OP), (X), (Y))
#define BRANCH_COST(speed_p, predictable_p) s390_branch_cost
#define SLOW_BYTE_ACCESS 1
#define MAX_FIXED_MODE_SIZE GET_MODE_BITSIZE (TARGET_64BIT ? TImode : DImode)
#define MOVE_MAX (TARGET_ZARCH ? 16 : 8)
#define MOVE_MAX_PIECES (TARGET_ZARCH ? 8 : 4)
#define MAX_MOVE_MAX 16
#define NO_FUNCTION_CSE 1
#define MOVE_RATIO(speed) (TARGET_64BIT? 2 : 4)
#define TEXT_SECTION_ASM_OP ".text"
#define DATA_SECTION_ASM_OP ".data"
#define BSS_SECTION_ASM_OP ".bss"
#ifndef __s390x__
#define CRT_CALL_STATIC_FUNCTION(SECTION_OP, FUNC) \
asm (SECTION_OP "\n\
bras\t%r2,1f\n\
0:	.long\t" USER_LABEL_PREFIX #FUNC " - 0b\n\
1:	l\t%r3,0(%r2)\n\
bas\t%r14,0(%r3,%r2)\n\
.previous");
#endif
#define PIC_OFFSET_TABLE_REGNUM (flag_pic ? 12 : INVALID_REGNUM)
#define LEGITIMATE_PIC_OPERAND_P(X)  legitimate_pic_operand_p (X)
#ifndef TARGET_DEFAULT_PIC_DATA_IS_TEXT_RELATIVE
#define TARGET_DEFAULT_PIC_DATA_IS_TEXT_RELATIVE 1
#endif
#define ASM_COMMENT_START "#"
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN)		\
asm_output_aligned_bss ((FILE), (DECL), (NAME), (SIZE), (ALIGN))
#define GLOBAL_ASM_OP ".globl "
#define ASM_OUTPUT_ALIGN(FILE, LOG) \
if ((LOG)) fprintf ((FILE), "\t.align\t%d\n", 1 << (LOG))
#define ASM_OUTPUT_SKIP(FILE, SIZE) \
fprintf ((FILE), "\t.set\t.,.+" HOST_WIDE_INT_PRINT_UNSIGNED"\n", (SIZE))
#define LOCAL_LABEL_PREFIX "."
#define LABEL_ALIGN(LABEL) \
s390_label_align ((LABEL))
#define REGISTER_NAMES							\
{ "%r0",  "%r1",  "%r2",  "%r3",  "%r4",  "%r5",  "%r6",  "%r7",	\
"%r8",  "%r9",  "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",	\
"%f0",  "%f2",  "%f4",  "%f6",  "%f1",  "%f3",  "%f5",  "%f7",	\
"%f8",  "%f10", "%f12", "%f14", "%f9",  "%f11", "%f13", "%f15",	\
"%ap",  "%cc",  "%fp",  "%rp",  "%a0",  "%a1",			\
"%v16", "%v18", "%v20", "%v22", "%v17", "%v19", "%v21", "%v23",	\
"%v24", "%v26", "%v28", "%v30", "%v25", "%v27", "%v29", "%v31"	\
}
#define ADDITIONAL_REGISTER_NAMES					\
{ { "v0", 16 }, { "v2",  17 }, { "v4",  18 }, { "v6",  19 },		\
{ "v1", 20 }, { "v3",  21 }, { "v5",  22 }, { "v7",  23 },          \
{ "v8", 24 }, { "v10", 25 }, { "v12", 26 }, { "v14", 27 },          \
{ "v9", 28 }, { "v11", 29 }, { "v13", 30 }, { "v15", 31 } };
#define PRINT_OPERAND(FILE, X, CODE) print_operand ((FILE), (X), (CODE))
#define PRINT_OPERAND_ADDRESS(FILE, ADDR) print_operand_address ((FILE), (ADDR))
#define ASM_OUTPUT_ADDR_VEC_ELT(FILE, VALUE)				\
do {									\
char buf[32];								\
fputs (integer_asm_op (UNITS_PER_LONG, TRUE), (FILE));		\
ASM_GENERATE_INTERNAL_LABEL (buf, "L", (VALUE));			\
assemble_name ((FILE), buf);						\
fputc ('\n', (FILE));							\
} while (0)
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL)		\
do {									\
char buf[32];								\
fputs (integer_asm_op (UNITS_PER_LONG, TRUE), (FILE));		\
ASM_GENERATE_INTERNAL_LABEL (buf, "L", (VALUE));			\
assemble_name ((FILE), buf);						\
fputc ('-', (FILE));							\
ASM_GENERATE_INTERNAL_LABEL (buf, "L", (REL));			\
assemble_name ((FILE), buf);						\
fputc ('\n', (FILE));							\
} while (0)
#define EPILOGUE_USES(REGNO) ((REGNO) == RETURN_REGNUM)
#undef ASM_OUTPUT_FUNCTION_LABEL
#define ASM_OUTPUT_FUNCTION_LABEL(FILE, NAME, DECL)		\
s390_asm_output_function_label ((FILE), (NAME), (DECL))
#if S390_USE_TARGET_ATTRIBUTE
#undef ASM_OUTPUT_FUNCTION_PREFIX
#define ASM_OUTPUT_FUNCTION_PREFIX s390_asm_output_function_prefix
#undef ASM_DECLARE_FUNCTION_SIZE
#define ASM_DECLARE_FUNCTION_SIZE s390_asm_declare_function_size
#endif
#define CASE_VECTOR_MODE (TARGET_64BIT ? DImode : SImode)
#define Pmode (TARGET_64BIT ? DImode : SImode)
#define POINTERS_EXTEND_UNSIGNED -1
#define FUNCTION_MODE QImode
#define CLZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) ((VALUE) = 64, 1)
#define SYMBOL_FLAG_ALIGN_SHIFT	  SYMBOL_FLAG_MACH_DEP_SHIFT
#define SYMBOL_FLAG_ALIGN_MASK    \
((SYMBOL_FLAG_MACH_DEP << 0) | (SYMBOL_FLAG_MACH_DEP << 1))
#define SYMBOL_FLAG_SET_ALIGN(X, A) \
(SYMBOL_REF_FLAGS (X) = (SYMBOL_REF_FLAGS (X) & ~SYMBOL_FLAG_ALIGN_MASK) \
| (A << SYMBOL_FLAG_ALIGN_SHIFT))
#define SYMBOL_FLAG_GET_ALIGN(X) \
((SYMBOL_REF_FLAGS (X) & SYMBOL_FLAG_ALIGN_MASK) >> SYMBOL_FLAG_ALIGN_SHIFT)
#define SYMBOL_FLAG_SET_NOTALIGN2(X) SYMBOL_FLAG_SET_ALIGN((X), 1)
#define SYMBOL_FLAG_SET_NOTALIGN4(X) SYMBOL_FLAG_SET_ALIGN((X), 2)
#define SYMBOL_FLAG_SET_NOTALIGN8(X) SYMBOL_FLAG_SET_ALIGN((X), 3)
#define SYMBOL_FLAG_NOTALIGN2_P(X) (SYMBOL_FLAG_GET_ALIGN(X) == 1)
#define SYMBOL_FLAG_NOTALIGN4_P(X) (SYMBOL_FLAG_GET_ALIGN(X) == 2	\
|| SYMBOL_FLAG_GET_ALIGN(X) == 1)
#define SYMBOL_FLAG_NOTALIGN8_P(X) (SYMBOL_FLAG_GET_ALIGN(X) == 3	\
|| SYMBOL_FLAG_GET_ALIGN(X) == 2	\
|| SYMBOL_FLAG_GET_ALIGN(X) == 1)
#define SHORT_DISP_IN_RANGE(d) ((d) >= 0 && (d) <= 4095)
#define DISP_IN_RANGE(d)				\
(TARGET_LONG_DISPLACEMENT				\
? ((d) >= -524288 && (d) <= 524287)			\
: SHORT_DISP_IN_RANGE(d))
#define READ_CAN_USE_WRITE_PREFETCH 1
extern const int processor_flags_table[];
#define VECTOR_STORE_FLAG_VALUE(MODE) CONSTM1_RTX (GET_MODE_INNER (MODE))
#define REGISTER_TARGET_PRAGMAS()		\
do {						\
s390_register_target_pragmas ();		\
} while (0)
#ifndef USED_FOR_TARGET
struct GTY (()) s390_frame_layout
{
HOST_WIDE_INT gprs_offset;
HOST_WIDE_INT f0_offset;
HOST_WIDE_INT f4_offset;
HOST_WIDE_INT f8_offset;
HOST_WIDE_INT backchain_offset;
int first_save_gpr_slot;
int last_save_gpr_slot;
#define SAVE_SLOT_NONE   0
#define SAVE_SLOT_STACK -1
signed char gpr_save_slots[16];
int first_save_gpr;
int first_restore_gpr;
int last_save_gpr;
int last_restore_gpr;
unsigned int fpr_bitmap;
int high_fprs;
bool save_return_addr_p;
HOST_WIDE_INT frame_size;
};
struct GTY(()) machine_function
{
struct s390_frame_layout frame_layout;
rtx base_reg;
bool split_branches_pending_p;
bool has_landing_pad_p;
bool tbegin_p;
rtx split_stack_varargs_pointer;
enum indirect_branch indirect_branch_jump;
enum indirect_branch indirect_branch_call;
enum indirect_branch function_return_mem;
enum indirect_branch function_return_reg;
};
#endif
#define TARGET_INDIRECT_BRANCH_NOBP_RET_OPTION				\
(cfun->machine->function_return_reg != indirect_branch_keep		\
|| cfun->machine->function_return_mem != indirect_branch_keep)
#define TARGET_INDIRECT_BRANCH_NOBP_RET					\
((cfun->machine->function_return_reg != indirect_branch_keep		\
&& !s390_return_addr_from_memory ())				\
|| (cfun->machine->function_return_mem != indirect_branch_keep	\
&& s390_return_addr_from_memory ()))
#define TARGET_INDIRECT_BRANCH_NOBP_JUMP				\
(cfun->machine->indirect_branch_jump != indirect_branch_keep)
#define TARGET_INDIRECT_BRANCH_NOBP_JUMP_THUNK				\
(cfun->machine->indirect_branch_jump == indirect_branch_thunk		\
|| cfun->machine->indirect_branch_jump == indirect_branch_thunk_extern)
#define TARGET_INDIRECT_BRANCH_NOBP_JUMP_INLINE_THUNK			\
(cfun->machine->indirect_branch_jump == indirect_branch_thunk_inline)
#define TARGET_INDIRECT_BRANCH_NOBP_CALL			\
(cfun->machine->indirect_branch_call != indirect_branch_keep)
#ifndef TARGET_DEFAULT_INDIRECT_BRANCH_TABLE
#define TARGET_DEFAULT_INDIRECT_BRANCH_TABLE 0
#endif
#define TARGET_INDIRECT_BRANCH_THUNK_NAME_EXRL "__s390_indirect_jump_r%d"
#define TARGET_INDIRECT_BRANCH_THUNK_NAME_EX   "__s390_indirect_jump_r%duse_r%d"
#define TARGET_INDIRECT_BRANCH_TABLE s390_indirect_branch_table
#endif 
