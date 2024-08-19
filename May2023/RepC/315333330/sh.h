#ifndef GCC_SH_H
#define GCC_SH_H
#include "config/vxworks-dummy.h"
extern int code_for_indirect_jump_scratch;
#define TARGET_CPU_CPP_BUILTINS() sh_cpu_cpp_builtins (pfile)
#ifndef SUBTARGET_FRAME_POINTER_REQUIRED
#define SUBTARGET_FRAME_POINTER_REQUIRED 0
#endif

#define TARGET_ELF 0
#define TARGET_SH2E (TARGET_SH2 && TARGET_SH_E)
#define TARGET_SH2A TARGET_HARD_SH2A
#define TARGET_SH2A_SINGLE (TARGET_SH2A && TARGET_SH2E)
#define TARGET_SH2A_DOUBLE (TARGET_HARD_SH2A_DOUBLE && TARGET_SH2A)
#define TARGET_SH3E (TARGET_SH3 && TARGET_SH_E)
#define TARGET_SUPERSCALAR (TARGET_HARD_SH4 || TARGET_SH2A)
#define TARGET_FPU_DOUBLE (TARGET_SH4 || TARGET_SH2A_DOUBLE)
#define TARGET_FPU_ANY (TARGET_SH2E || TARGET_FPU_DOUBLE)
#define TARGET_SH4A_FP (TARGET_SH4A && TARGET_FPU_ANY)
#define TARGET_VARARGS_PRETEND_ARGS(FUN_DECL) \
(! TARGET_SH2E \
&& ! (TARGET_HITACHI || sh_attr_renesas_p (FUN_DECL)))
#ifndef TARGET_CPU_DEFAULT
#define TARGET_CPU_DEFAULT SELECT_SH1
#define SUPPORT_SH1 1
#define SUPPORT_SH2E 1
#define SUPPORT_SH4 1
#define SUPPORT_SH4_SINGLE 1
#define SUPPORT_SH2A 1
#define SUPPORT_SH2A_SINGLE 1
#endif
#define TARGET_DIVIDE_CALL_DIV1 (sh_div_strategy == SH_DIV_CALL_DIV1)
#define TARGET_DIVIDE_CALL_FP (sh_div_strategy == SH_DIV_CALL_FP)
#define TARGET_DIVIDE_CALL_TABLE (sh_div_strategy == SH_DIV_CALL_TABLE)
#define SELECT_SH1		 (MASK_SH1)
#define SELECT_SH2		 (MASK_SH2 | SELECT_SH1)
#define SELECT_SH2E		 (MASK_SH_E | MASK_SH2 | MASK_SH1 \
| MASK_FPU_SINGLE)
#define SELECT_SH2A		 (MASK_SH_E | MASK_HARD_SH2A \
| MASK_HARD_SH2A_DOUBLE \
| MASK_SH2 | MASK_SH1)
#define SELECT_SH2A_NOFPU	 (MASK_HARD_SH2A | MASK_SH2 | MASK_SH1)
#define SELECT_SH2A_SINGLE_ONLY  (MASK_SH_E | MASK_HARD_SH2A | MASK_SH2 \
| MASK_SH1 | MASK_FPU_SINGLE \
| MASK_FPU_SINGLE_ONLY)
#define SELECT_SH2A_SINGLE	 (MASK_SH_E | MASK_HARD_SH2A \
| MASK_FPU_SINGLE | MASK_HARD_SH2A_DOUBLE \
| MASK_SH2 | MASK_SH1)
#define SELECT_SH3		 (MASK_SH3 | SELECT_SH2)
#define SELECT_SH3E		 (MASK_SH_E | MASK_FPU_SINGLE | SELECT_SH3)
#define SELECT_SH4_NOFPU	 (MASK_HARD_SH4 | SELECT_SH3)
#define SELECT_SH4_SINGLE_ONLY	 (MASK_HARD_SH4 | SELECT_SH3E \
| MASK_FPU_SINGLE_ONLY)
#define SELECT_SH4		 (MASK_SH4 | MASK_SH_E | MASK_HARD_SH4 \
| SELECT_SH3)
#define SELECT_SH4_SINGLE	 (MASK_FPU_SINGLE | SELECT_SH4)
#define SELECT_SH4A_NOFPU	 (MASK_SH4A | SELECT_SH4_NOFPU)
#define SELECT_SH4A_SINGLE_ONLY  (MASK_SH4A | SELECT_SH4_SINGLE_ONLY)
#define SELECT_SH4A		 (MASK_SH4A | SELECT_SH4)
#define SELECT_SH4A_SINGLE	 (MASK_SH4A | SELECT_SH4_SINGLE)
#if SUPPORT_SH1
#define SUPPORT_SH2 1
#endif
#if SUPPORT_SH2
#define SUPPORT_SH3 1
#define SUPPORT_SH2A_NOFPU 1
#endif
#if SUPPORT_SH3
#define SUPPORT_SH4_NOFPU 1
#endif
#if SUPPORT_SH4_NOFPU
#define SUPPORT_SH4A_NOFPU 1
#define SUPPORT_SH4AL 1
#endif
#if SUPPORT_SH2E
#define SUPPORT_SH3E 1
#define SUPPORT_SH2A_SINGLE_ONLY 1
#endif
#if SUPPORT_SH3E
#define SUPPORT_SH4_SINGLE_ONLY 1
#endif
#if SUPPORT_SH4_SINGLE_ONLY
#define SUPPORT_SH4A_SINGLE_ONLY 1
#endif
#if SUPPORT_SH4
#define SUPPORT_SH4A 1
#endif
#if SUPPORT_SH4_SINGLE
#define SUPPORT_SH4A_SINGLE 1
#endif
#define MASK_ARCH (MASK_SH1 | MASK_SH2 | MASK_SH3 | MASK_SH_E | MASK_SH4 \
| MASK_HARD_SH2A | MASK_HARD_SH2A_DOUBLE | MASK_SH4A \
| MASK_HARD_SH4 | MASK_FPU_SINGLE \
| MASK_FPU_SINGLE_ONLY)
#ifndef TARGET_ENDIAN_DEFAULT
#define TARGET_ENDIAN_DEFAULT 0
#endif
#ifndef TARGET_OPT_DEFAULT
#define TARGET_OPT_DEFAULT  0
#endif
#define TARGET_DEFAULT \
(TARGET_CPU_DEFAULT | TARGET_ENDIAN_DEFAULT | TARGET_OPT_DEFAULT)
#ifndef SH_MULTILIB_CPU_DEFAULT
#define SH_MULTILIB_CPU_DEFAULT "m1"
#endif
#if TARGET_ENDIAN_DEFAULT
#define MULTILIB_DEFAULTS { "ml", SH_MULTILIB_CPU_DEFAULT }
#else
#define MULTILIB_DEFAULTS { "mb", SH_MULTILIB_CPU_DEFAULT }
#endif
#define CPP_SPEC " %(subtarget_cpp_spec) "
#ifndef SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC ""
#endif
#ifndef SUBTARGET_EXTRA_SPECS
#define SUBTARGET_EXTRA_SPECS
#endif
#define EXTRA_SPECS						\
{ "subtarget_cpp_spec", SUBTARGET_CPP_SPEC },			\
{ "link_emul_prefix", LINK_EMUL_PREFIX },			\
{ "link_default_cpu_emul", LINK_DEFAULT_CPU_EMUL },		\
{ "subtarget_link_emul_suffix", SUBTARGET_LINK_EMUL_SUFFIX },	\
{ "subtarget_link_spec", SUBTARGET_LINK_SPEC },		\
{ "subtarget_asm_endian_spec", SUBTARGET_ASM_ENDIAN_SPEC },	\
{ "subtarget_asm_relax_spec", SUBTARGET_ASM_RELAX_SPEC },	\
{ "subtarget_asm_isa_spec", SUBTARGET_ASM_ISA_SPEC },		\
{ "subtarget_asm_spec", SUBTARGET_ASM_SPEC },			\
SUBTARGET_EXTRA_SPECS
#if TARGET_CPU_DEFAULT & MASK_HARD_SH4
#define SUBTARGET_ASM_RELAX_SPEC "%{!m1:%{!m2:%{!m3*:-isa=sh4-up}}}"
#else
#define SUBTARGET_ASM_RELAX_SPEC "%{m4*:-isa=sh4-up}"
#endif
#define SH_ASM_SPEC \
"%(subtarget_asm_endian_spec) %{mrelax:-relax %(subtarget_asm_relax_spec)} \
%(subtarget_asm_isa_spec) %(subtarget_asm_spec) \
%{m1:--isa=sh} \
%{m2:--isa=sh2} \
%{m2e:--isa=sh2e} \
%{m3:--isa=sh3} \
%{m3e:--isa=sh3e} \
%{m4:--isa=sh4a} \
%{m4-single:--isa=sh4a} \
%{m4-single-only:--isa=sh4a} \
%{m4-nofpu:--isa=sh4a-nofpu} \
%{m4a:--isa=sh4a} \
%{m4a-single:--isa=sh4a} \
%{m4a-single-only:--isa=sh4a} \
%{m4a-nofpu:--isa=sh4a-nofpu} \
%{m2a:--isa=sh2a} \
%{m2a-single:--isa=sh2a} \
%{m2a-single-only:--isa=sh2a} \
%{m2a-nofpu:--isa=sh2a-nofpu} \
%{m4al:-dsp}"
#define ASM_SPEC SH_ASM_SPEC
#ifndef SUBTARGET_ASM_ENDIAN_SPEC
#if TARGET_ENDIAN_DEFAULT == MASK_LITTLE_ENDIAN
#define SUBTARGET_ASM_ENDIAN_SPEC "%{mb:-big} %{!mb:-little}"
#else
#define SUBTARGET_ASM_ENDIAN_SPEC "%{ml:-little} %{!ml:-big}"
#endif
#endif
#if STRICT_NOFPU == 1
#if TARGET_CPU_DEFAULT & MASK_HARD_SH4 && !(TARGET_CPU_DEFAULT & MASK_SH_E)
#define SUBTARGET_ASM_ISA_SPEC "%{!m1:%{!m2:%{!m3*:%{m4-nofpu|!m4*:-isa=sh4-nofpu}}}}"
#else
#define SUBTARGET_ASM_ISA_SPEC \
"%{m4-nofpu:-isa=sh4-nofpu} " ASM_ISA_DEFAULT_SPEC
#endif
#else 
#define SUBTARGET_ASM_ISA_SPEC ASM_ISA_DEFAULT_SPEC
#endif
#ifndef SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC "%{mfdpic:--fdpic}"
#endif
#if TARGET_ENDIAN_DEFAULT == MASK_LITTLE_ENDIAN
#define LINK_EMUL_PREFIX "sh%{!mb:l}"
#else
#define LINK_EMUL_PREFIX "sh%{ml:l}"
#endif
#define LINK_DEFAULT_CPU_EMUL ""
#define ASM_ISA_DEFAULT_SPEC ""
#define SUBTARGET_LINK_EMUL_SUFFIX "%{mfdpic:_fd}"
#define SUBTARGET_LINK_SPEC ""
#define LINK_SPEC SH_LINK_SPEC
#define SH_LINK_SPEC "\
-m %(link_emul_prefix)\
%{!m1:%{!m2:%{!m3*:%{!m4*:%(link_default_cpu_emul)}}}}\
%(subtarget_link_emul_suffix) \
%{mrelax:-relax} %(subtarget_link_spec)"
#ifndef SH_DIV_STR_FOR_SIZE
#define SH_DIV_STR_FOR_SIZE "call"
#endif
#if TARGET_ENDIAN_DEFAULT == MASK_BIG_ENDIAN
#define IS_LITTLE_ENDIAN_OPTION "%{ml:"
#else
#define IS_LITTLE_ENDIAN_OPTION "%{!mb:"
#endif
#if TARGET_CPU_DEFAULT & MASK_HARD_SH2A
#define UNSUPPORTED_SH2A IS_LITTLE_ENDIAN_OPTION \
"%{m2a*|!m1:%{!m2*:%{!m3*:%{!m4*:%eSH2a does not support little-endian}}}}}"
#else
#define UNSUPPORTED_SH2A IS_LITTLE_ENDIAN_OPTION \
"%{m2a*:%eSH2a does not support little-endian}}"
#endif
#ifdef FDPIC_DEFAULT
#define FDPIC_SELF_SPECS "%{!mno-fdpic:-mfdpic}"
#else
#define FDPIC_SELF_SPECS
#endif
#undef DRIVER_SELF_SPECS
#define DRIVER_SELF_SPECS UNSUPPORTED_SH2A SUBTARGET_DRIVER_SELF_SPECS \
FDPIC_SELF_SPECS
#undef SUBTARGET_DRIVER_SELF_SPECS
#define SUBTARGET_DRIVER_SELF_SPECS
#define ASSEMBLER_DIALECT assembler_dialect
extern int assembler_dialect;
enum sh_divide_strategy_e {
SH_DIV_CALL_DIV1, 
SH_DIV_CALL_FP,     
SH_DIV_CALL_TABLE,  
SH_DIV_INTRINSIC
};
extern enum sh_divide_strategy_e sh_div_strategy;
#ifndef SH_DIV_STRATEGY_DEFAULT
#define SH_DIV_STRATEGY_DEFAULT SH_DIV_CALL_DIV1
#endif
#ifdef __cplusplus
struct sh_atomic_model
{
enum enum_type
{
none = 0,
soft_gusa,
hard_llcs,
soft_tcb,
soft_imask,
num_models
};
bool strict;
enum_type type;
const char* name;
const char* cdef_name;
int tcb_gbr_offset;
};
extern const sh_atomic_model& selected_atomic_model (void);
#define TARGET_ATOMIC_ANY \
(selected_atomic_model ().type != sh_atomic_model::none)
#define TARGET_ATOMIC_STRICT \
(selected_atomic_model ().strict)
#define TARGET_ATOMIC_SOFT_GUSA \
(selected_atomic_model ().type == sh_atomic_model::soft_gusa)
#define TARGET_ATOMIC_HARD_LLCS \
(selected_atomic_model ().type == sh_atomic_model::hard_llcs)
#define TARGET_ATOMIC_SOFT_TCB \
(selected_atomic_model ().type == sh_atomic_model::soft_tcb)
#define TARGET_ATOMIC_SOFT_TCB_GBR_OFFSET_RTX \
GEN_INT (selected_atomic_model ().tcb_gbr_offset)
#define TARGET_ATOMIC_SOFT_IMASK \
(selected_atomic_model ().type == sh_atomic_model::soft_imask)
#endif 
#define SUBTARGET_OVERRIDE_OPTIONS (void) 0

#define TARGET_BIG_ENDIAN (!TARGET_LITTLE_ENDIAN)
#define SH_REG_MSW_OFFSET (TARGET_LITTLE_ENDIAN ? 1 : 0)
#define SH_REG_LSW_OFFSET (TARGET_LITTLE_ENDIAN ? 0 : 1)
#define BITS_BIG_ENDIAN  0
#define BYTES_BIG_ENDIAN TARGET_BIG_ENDIAN
#define WORDS_BIG_ENDIAN TARGET_BIG_ENDIAN
#define MAX_BITS_PER_WORD 64
#define INT_TYPE_SIZE 32
#define LONG_TYPE_SIZE (32)
#define LONG_LONG_TYPE_SIZE 64
#define LONG_DOUBLE_TYPE_SIZE 64
#define UNITS_PER_WORD	(4)
#define MIN_UNITS_PER_WORD 4
#define DWARF_CIE_DATA_ALIGNMENT -4
#define POINTER_SIZE  (32)
#define PARM_BOUNDARY  	(32)
#define STACK_BOUNDARY  BIGGEST_ALIGNMENT
#define CACHE_LOG (TARGET_HARD_SH4 ? 5 : TARGET_SH2 ? 4 : 2)
#define FUNCTION_BOUNDARY (16)
#define EMPTY_FIELD_BOUNDARY  32
#define BIGGEST_ALIGNMENT  (TARGET_ALIGN_DOUBLE ? 64 : 32)
#define FASTEST_ALIGNMENT (32)
#define LOCAL_ALIGNMENT(TYPE, ALIGN) \
((GET_MODE_CLASS (TYPE_MODE (TYPE)) == MODE_COMPLEX_INT \
|| GET_MODE_CLASS (TYPE_MODE (TYPE)) == MODE_COMPLEX_FLOAT) \
? (unsigned) MIN (BIGGEST_ALIGNMENT, \
GET_MODE_BITSIZE (as_a <fixed_size_mode> \
(TYPE_MODE (TYPE)))) \
: (unsigned) DATA_ALIGNMENT(TYPE, ALIGN))
#define DATA_ALIGNMENT(TYPE, ALIGN)		\
(TREE_CODE (TYPE) == ARRAY_TYPE		\
&& TYPE_MODE (TREE_TYPE (TYPE)) == QImode	\
&& (ALIGN) < FASTEST_ALIGNMENT ? FASTEST_ALIGNMENT : (ALIGN))
#define STRUCTURE_SIZE_BOUNDARY (TARGET_PADSTRUCT ? 32 : 8)
#define STRICT_ALIGNMENT 1
#define LABEL_ALIGN_AFTER_BARRIER(LABEL_AFTER_BARRIER) \
barrier_align (LABEL_AFTER_BARRIER)
#define LOOP_ALIGN(A_LABEL) sh_loop_align (A_LABEL)
#define LABEL_ALIGN(A_LABEL) \
(									\
(PREV_INSN (A_LABEL)							\
&& NONJUMP_INSN_P (PREV_INSN (A_LABEL))				\
&& GET_CODE (PATTERN (PREV_INSN (A_LABEL))) == UNSPEC_VOLATILE	\
&& XINT (PATTERN (PREV_INSN (A_LABEL)), 1) == UNSPECV_ALIGN)		\
\
? INTVAL (XVECEXP (PATTERN (PREV_INSN (A_LABEL)), 0, 0))		\
: 0)
#define ADDR_VEC_ALIGN(ADDR_VEC) 2
#define INSN_LENGTH_ALIGNMENT(A_INSN)		\
(NONJUMP_INSN_P (A_INSN)			\
? 1						\
: JUMP_P (A_INSN) || CALL_P (A_INSN)		\
? 1						\
: CACHE_LOG)

#define MAX_REGISTER_NAME_LENGTH 6
extern char sh_register_names[][MAX_REGISTER_NAME_LENGTH + 1];
#define SH_REGISTER_NAMES_INITIALIZER					\
{									\
"r0",   "r1",   "r2",   "r3",   "r4",   "r5",   "r6",   "r7", 	\
"r8",   "r9",   "r10",  "r11",  "r12",  "r13",  "r14",  "r15",	\
"r16",  "r17",  "r18",  "r19",  "r20",  "r21",  "r22",  "r23",	\
"r24",  "r25",  "r26",  "r27",  "r28",  "r29",  "r30",  "r31",	\
"r32",  "r33",  "r34",  "r35",  "r36",  "r37",  "r38",  "r39", 	\
"r40",  "r41",  "r42",  "r43",  "r44",  "r45",  "r46",  "r47",	\
"r48",  "r49",  "r50",  "r51",  "r52",  "r53",  "r54",  "r55",	\
"r56",  "r57",  "r58",  "r59",  "r60",  "r61",  "r62",  "r63",	\
"fr0",  "fr1",  "fr2",  "fr3",  "fr4",  "fr5",  "fr6",  "fr7", 	\
"fr8",  "fr9",  "fr10", "fr11", "fr12", "fr13", "fr14", "fr15",	\
"fr16", "fr17", "fr18", "fr19", "fr20", "fr21", "fr22", "fr23",	\
"fr24", "fr25", "fr26", "fr27", "fr28", "fr29", "fr30", "fr31",	\
"fr32", "fr33", "fr34", "fr35", "fr36", "fr37", "fr38", "fr39", 	\
"fr40", "fr41", "fr42", "fr43", "fr44", "fr45", "fr46", "fr47",	\
"fr48", "fr49", "fr50", "fr51", "fr52", "fr53", "fr54", "fr55",	\
"fr56", "fr57", "fr58", "fr59", "fr60", "fr61", "fr62", "fr63",	\
"tr0",  "tr1",  "tr2",  "tr3",  "tr4",  "tr5",  "tr6",  "tr7", 	\
"xd0",  "xd2",  "xd4",  "xd6",  "xd8",  "xd10", "xd12", "xd14",	\
"gbr",  "ap",	  "pr",   "t",    "mach", "macl", "fpul", "fpscr",	\
"rap",  "sfp", "fpscr0", "fpscr1"					\
}
#define REGNAMES_ARR_INDEX_1(index) \
(sh_register_names[index])
#define REGNAMES_ARR_INDEX_2(index) \
REGNAMES_ARR_INDEX_1 ((index)), REGNAMES_ARR_INDEX_1 ((index)+1)
#define REGNAMES_ARR_INDEX_4(index) \
REGNAMES_ARR_INDEX_2 ((index)), REGNAMES_ARR_INDEX_2 ((index)+2)
#define REGNAMES_ARR_INDEX_8(index) \
REGNAMES_ARR_INDEX_4 ((index)), REGNAMES_ARR_INDEX_4 ((index)+4)
#define REGNAMES_ARR_INDEX_16(index) \
REGNAMES_ARR_INDEX_8 ((index)), REGNAMES_ARR_INDEX_8 ((index)+8)
#define REGNAMES_ARR_INDEX_32(index) \
REGNAMES_ARR_INDEX_16 ((index)), REGNAMES_ARR_INDEX_16 ((index)+16)
#define REGNAMES_ARR_INDEX_64(index) \
REGNAMES_ARR_INDEX_32 ((index)), REGNAMES_ARR_INDEX_32 ((index)+32)
#define REGISTER_NAMES \
{ \
REGNAMES_ARR_INDEX_64 (0), \
REGNAMES_ARR_INDEX_64 (64), \
REGNAMES_ARR_INDEX_8 (128), \
REGNAMES_ARR_INDEX_8 (136), \
REGNAMES_ARR_INDEX_8 (144), \
REGNAMES_ARR_INDEX_4 (152) \
}
#define ADDREGNAMES_SIZE 32
#define MAX_ADDITIONAL_REGISTER_NAME_LENGTH 4
extern char sh_additional_register_names[ADDREGNAMES_SIZE] \
[MAX_ADDITIONAL_REGISTER_NAME_LENGTH + 1];
#define SH_ADDITIONAL_REGISTER_NAMES_INITIALIZER			\
{									\
"dr0",  "dr2",  "dr4",  "dr6",  "dr8",  "dr10", "dr12", "dr14",	\
"dr16", "dr18", "dr20", "dr22", "dr24", "dr26", "dr28", "dr30",	\
"dr32", "dr34", "dr36", "dr38", "dr40", "dr42", "dr44", "dr46",	\
"dr48", "dr50", "dr52", "dr54", "dr56", "dr58", "dr60", "dr62"	\
}
#define ADDREGNAMES_REGNO(index) \
((index < 32) ? (FIRST_FP_REG + (index) * 2) \
: (-1))
#define ADDREGNAMES_ARR_INDEX_1(index) \
{ (sh_additional_register_names[index]), ADDREGNAMES_REGNO (index) }
#define ADDREGNAMES_ARR_INDEX_2(index) \
ADDREGNAMES_ARR_INDEX_1 ((index)), ADDREGNAMES_ARR_INDEX_1 ((index)+1)
#define ADDREGNAMES_ARR_INDEX_4(index) \
ADDREGNAMES_ARR_INDEX_2 ((index)), ADDREGNAMES_ARR_INDEX_2 ((index)+2)
#define ADDREGNAMES_ARR_INDEX_8(index) \
ADDREGNAMES_ARR_INDEX_4 ((index)), ADDREGNAMES_ARR_INDEX_4 ((index)+4)
#define ADDREGNAMES_ARR_INDEX_16(index) \
ADDREGNAMES_ARR_INDEX_8 ((index)), ADDREGNAMES_ARR_INDEX_8 ((index)+8)
#define ADDREGNAMES_ARR_INDEX_32(index) \
ADDREGNAMES_ARR_INDEX_16 ((index)), ADDREGNAMES_ARR_INDEX_16 ((index)+16)
#define ADDITIONAL_REGISTER_NAMES \
{					\
ADDREGNAMES_ARR_INDEX_32 (0)		\
}
#define FIRST_GENERAL_REG R0_REG
#define LAST_GENERAL_REG (FIRST_GENERAL_REG + (15))
#define FIRST_FP_REG DR0_REG
#define LAST_FP_REG  (FIRST_FP_REG + (TARGET_SH2E ? 15 : -1))
#define FIRST_XD_REG XD0_REG
#define LAST_XD_REG  (FIRST_XD_REG + ((TARGET_SH4 && TARGET_FMOVD) ? 7 : -1))
#define FIRST_BANKED_REG R0_REG
#define LAST_BANKED_REG R7_REG
#define BANKED_REGISTER_P(REGNO) \
IN_RANGE ((REGNO), \
(unsigned HOST_WIDE_INT) FIRST_BANKED_REG, \
(unsigned HOST_WIDE_INT) LAST_BANKED_REG)
#define GENERAL_REGISTER_P(REGNO) \
IN_RANGE ((REGNO), \
(unsigned HOST_WIDE_INT) FIRST_GENERAL_REG, \
(unsigned HOST_WIDE_INT) LAST_GENERAL_REG)
#define GENERAL_OR_AP_REGISTER_P(REGNO) \
(GENERAL_REGISTER_P (REGNO) || ((REGNO) == AP_REG) \
|| ((REGNO) == FRAME_POINTER_REGNUM))
#define FP_REGISTER_P(REGNO) \
((int) (REGNO) >= FIRST_FP_REG && (int) (REGNO) <= LAST_FP_REG)
#define XD_REGISTER_P(REGNO) \
((int) (REGNO) >= FIRST_XD_REG && (int) (REGNO) <= LAST_XD_REG)
#define FP_OR_XD_REGISTER_P(REGNO) \
(FP_REGISTER_P (REGNO) || XD_REGISTER_P (REGNO))
#define FP_ANY_REGISTER_P(REGNO) \
(FP_REGISTER_P (REGNO) || XD_REGISTER_P (REGNO) || (REGNO) == FPUL_REG)
#define SPECIAL_REGISTER_P(REGNO) \
((REGNO) == GBR_REG || (REGNO) == T_REG \
|| (REGNO) == MACH_REG || (REGNO) == MACL_REG \
|| (REGNO) == FPSCR_MODES_REG || (REGNO) == FPSCR_STAT_REG)
#define VALID_REGISTER_P(REGNO) \
(GENERAL_REGISTER_P (REGNO) || FP_REGISTER_P (REGNO) \
|| XD_REGISTER_P (REGNO) \
|| (REGNO) == AP_REG || (REGNO) == RAP_REG \
|| (REGNO) == FRAME_POINTER_REGNUM \
|| ((SPECIAL_REGISTER_P (REGNO) || (REGNO) == PR_REG)) \
|| (TARGET_SH2E && (REGNO) == FPUL_REG))
#define REGISTER_NATURAL_MODE(REGNO) \
(FP_REGISTER_P (REGNO) ? E_SFmode \
: XD_REGISTER_P (REGNO) ? E_DFmode : E_SImode)
#define FIRST_PSEUDO_REGISTER 156
#define DWARF_FRAME_REGISTERS (153)
#define FIXED_REGISTERS							\
{									\
\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      1,		\
\
1,      0,      0,      0,      0,      0,      0,      0,		\
\
\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      1,		\
\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
\
0,      0,      0,      0,      0,      0,      0,      0,		\
\
0,      0,      0,      0,      0,      0,      0,      0,		\
\
1,      1,      1,      1,      1,      1,      0,      1,		\
\
1,      1,      1,      1,						\
}
#define CALL_USED_REGISTERS						\
{									\
\
1,      1,      1,      1,      1,      1,      1,      1,		\
\
0,      0,      0,      0,      0,      0,      0,      1,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      1,      1,      1,      1,		\
\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
\
1,      1,      1,      1,      1,      0,      0,      0,		\
\
1,      1,      1,      1,      1,      1,      0,      0,		\
\
1,      1,      1,      1,      1,      1,      1,      1,		\
\
1,      1,      1,      1,						\
}
#define CALL_REALLY_USED_REGISTERS 					\
{									\
\
1,      1,      1,      1,      1,      1,      1,      1,		\
\
0,      0,      0,      0,      0,      0,      0,      1,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      1,      1,      1,      1,		\
\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      1,      1,      1,      1,		\
1,      1,      1,      1,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
0,      0,      0,      0,      0,      0,      0,      0,		\
\
1,      1,      1,      1,      1,      0,      0,      0,		\
\
1,      1,      1,      1,      1,      1,      0,      0,		\
\
0,      1,      1,      1,      1,      1,      1,      1,		\
\
1,      1,      0,      0,						\
}
#define HARD_REGNO_CALLER_SAVE_MODE(REGNO, NREGS, MODE)	\
sh_hard_regno_caller_save_mode ((REGNO), (NREGS), (MODE))
#define HARD_REGNO_RENAME_OK(OLD_REG, NEW_REG) \
sh_hard_regno_rename_ok (OLD_REG, NEW_REG)
#define STACK_POINTER_REGNUM	SP_REG
#define HARD_FRAME_POINTER_REGNUM	FP_REG
#define FRAME_POINTER_REGNUM	153
#define RETURN_ADDRESS_POINTER_REGNUM RAP_REG
#define PIC_OFFSET_TABLE_REGNUM	(flag_pic ? PIC_REG : INVALID_REGNUM)
#define PIC_OFFSET_TABLE_REG_CALL_CLOBBERED TARGET_FDPIC
#define GOT_SYMBOL_NAME "*_GLOBAL_OFFSET_TABLE_"
#define ELIMINABLE_REGS						\
{{ HARD_FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM},		\
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM},			\
{ FRAME_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},		\
{ RETURN_ADDRESS_POINTER_REGNUM, STACK_POINTER_REGNUM},	\
{ RETURN_ADDRESS_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},	\
{ ARG_POINTER_REGNUM, STACK_POINTER_REGNUM},			\
{ ARG_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET) \
OFFSET = initial_elimination_offset ((FROM), (TO))
#define ARG_POINTER_REGNUM	AP_REG
#define STATIC_CHAIN_REGNUM	(3)
#define DEFAULT_PCC_STRUCT_RETURN 0

enum reg_class
{
NO_REGS,
R0_REGS,
PR_REGS,
T_REGS,
MAC_REGS,
FPUL_REGS,
SIBCALL_REGS,
NON_SP_REGS,
GENERAL_REGS,
FP0_REGS,
FP_REGS,
DF_REGS,
FPSCR_REGS,
GENERAL_FP_REGS,
GENERAL_DF_REGS,
TARGET_REGS,
ALL_REGS,
LIM_REG_CLASSES
};
#define N_REG_CLASSES  (int) LIM_REG_CLASSES
#define REG_CLASS_NAMES	\
{			\
"NO_REGS",		\
"R0_REGS",		\
"PR_REGS",		\
"T_REGS",		\
"MAC_REGS",		\
"FPUL_REGS",		\
"SIBCALL_REGS",	\
"NON_SP_REGS",	\
"GENERAL_REGS",	\
"FP0_REGS",		\
"FP_REGS",		\
"DF_REGS",		\
"FPSCR_REGS",		\
"GENERAL_FP_REGS",	\
"GENERAL_DF_REGS",	\
"TARGET_REGS",	\
"ALL_REGS",		\
}
#define REG_CLASS_CONTENTS						\
{									\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	\
\
{ 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00040000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00080000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00300000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00400000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	\
\
{ 0xffff7fff, 0xffffffff, 0x00000000, 0x00000000, 0x03020000 },	\
\
{ 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x03020000 },	\
\
{ 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0xffffffff, 0xffffffff, 0x0000ff00 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00800000 },	\
\
{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x03020000 },	\
\
{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x0302ff00 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x000000ff },	\
\
{ 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x0fffffff },	\
}
extern enum reg_class regno_reg_class[FIRST_PSEUDO_REGISTER];
#define REGNO_REG_CLASS(REGNO) regno_reg_class[(REGNO)]
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P \
sh_small_register_classes_for_mode_p
#define REG_ALLOC_ORDER \
{ \
65, 66, 67, 68, 69, 70, 71, 64, \
72, 73, 74, 75, 80, 81, 82, 83, \
84, 85, 86, 87, 88, 89, 90, 91, \
92, 93, 94, 95, 96, 97, 98, 99, \
\
76, 77, 78, 79,100,101,102,103, \
104,105,106,107,108,109,110,111, \
112,113,114,115,116,117,118,119, \
120,121,122,123,124,125,126,127, \
136,137,138,139,140,141,142,143, \
151, \
\
1,  2,  3,  7,  6,  5,  4,  0, \
8,  9, 17, 19, 20, 21, 22, 23, \
36, 37, 38, 39, 40, 41, 42, 43, \
60, 61, 62, \
\
10, 11, 12, 13, 14, 18, \
\
28, 29, 30, 31, 32, 33, 34, 35, \
44, 45, 46, 47, 48, 49, 50, 51, \
52, 53, 54, 55, 56, 57, 58, 59, \
150, \
\
15, 16, 24, 25, 26, 27, 63,144, \
145,146,147,148,149,152,153,154,155  }
#define INDEX_REG_CLASS R0_REGS
#define BASE_REG_CLASS GENERAL_REGS

#define CONST_OK_FOR_I08(VALUE) (((HOST_WIDE_INT)(VALUE))>= -128 \
&& ((HOST_WIDE_INT)(VALUE)) <= 127)
#define CONST_OK_FOR_K08(VALUE) (((HOST_WIDE_INT)(VALUE))>= 0 \
&& ((HOST_WIDE_INT)(VALUE)) <= 255)
#define ZERO_EXTRACT_ANDMASK(EXTRACT_SZ_RTX, EXTRACT_POS_RTX)\
(((1 << INTVAL (EXTRACT_SZ_RTX)) - 1) << INTVAL (EXTRACT_POS_RTX))
#define CLASS_MAX_NREGS(CLASS, MODE) \
((GET_MODE_SIZE (MODE) + UNITS_PER_WORD - 1) / UNITS_PER_WORD)

#define NPARM_REGS(MODE) \
(TARGET_FPU_ANY && (MODE) == SFmode \
? 8 \
: TARGET_FPU_DOUBLE \
&& (GET_MODE_CLASS (MODE) == MODE_FLOAT \
|| GET_MODE_CLASS (MODE) == MODE_COMPLEX_FLOAT) \
? 8 \
: 4)
#define FIRST_PARM_REG (FIRST_GENERAL_REG + 4)
#define FIRST_RET_REG  (FIRST_GENERAL_REG + 0)
#define FIRST_FP_PARM_REG (FIRST_FP_REG + 4)
#define FIRST_FP_RET_REG FIRST_FP_REG
#define STACK_GROWS_DOWNWARD 1
#define FRAME_GROWS_DOWNWARD 1
#if 0
#define PUSH_ROUNDING(NPUSHED)  (((NPUSHED) + 3) & ~3)
#endif
#define FIRST_PARM_OFFSET(FNDECL)  0
#define CALL_POPS_ARGS(CUM) (0)
#define BASE_RETURN_VALUE_REG(MODE) \
((TARGET_FPU_ANY && ((MODE) == SFmode))		\
? FIRST_FP_RET_REG					\
: TARGET_FPU_ANY && (MODE) == SCmode			\
? FIRST_FP_RET_REG					\
: (TARGET_FPU_DOUBLE					\
&& ((MODE) == DFmode || (MODE) == SFmode		\
|| (MODE) == DCmode || (MODE) == SCmode ))	\
? FIRST_FP_RET_REG					\
: FIRST_RET_REG)
#define BASE_ARG_REG(MODE) \
((TARGET_SH2E && ((MODE) == SFmode))			\
? FIRST_FP_PARM_REG					\
: TARGET_FPU_DOUBLE					\
&& (GET_MODE_CLASS (MODE) == MODE_FLOAT		\
|| GET_MODE_CLASS (MODE) == MODE_COMPLEX_FLOAT)\
? FIRST_FP_PARM_REG					\
: FIRST_PARM_REG)
#define FUNCTION_ARG_REGNO_P(REGNO) \
(((unsigned) (REGNO) >= (unsigned) FIRST_PARM_REG			\
&& (unsigned) (REGNO) < (unsigned) (FIRST_PARM_REG + NPARM_REGS (SImode)))\
|| (TARGET_FPU_ANY							\
&& (unsigned) (REGNO) >= (unsigned) FIRST_FP_PARM_REG		\
&& (unsigned) (REGNO) < (unsigned) (FIRST_FP_PARM_REG		\
+ NPARM_REGS (SFmode))))

#ifdef __cplusplus
enum sh_arg_class { SH_ARG_INT = 0, SH_ARG_FLOAT = 1 };
struct sh_args
{
int arg_count[2];
bool force_mem;
bool prototype_p;
int free_single_fp_reg;
bool outgoing;
bool renesas_abi;
};
typedef sh_args CUMULATIVE_ARGS;
extern bool current_function_interrupt;
#endif 
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS) \
sh_init_cumulative_args (& (CUM), (FNTYPE), (LIBNAME), (FNDECL),\
(N_NAMED_ARGS), VOIDmode)
#define INIT_CUMULATIVE_LIBCALL_ARGS(CUM, MODE, LIBNAME) \
sh_init_cumulative_args (& (CUM), NULL_TREE, (LIBNAME), NULL_TREE, 0, (MODE))
#define FUNCTION_ARG_SCmode_WART 1
#define SH_MIN_ALIGN_FOR_CALLEE_COPY (8 * BITS_PER_UNIT)
#define FUNCTION_PROFILER(STREAM,LABELNO)			\
{								\
fprintf((STREAM), "\t.align\t2\n");				\
fprintf((STREAM), "\ttrapa\t#33\n");				\
fprintf((STREAM), "\t.align\t2\n");				\
asm_fprintf((STREAM), "\t.long\t%LLP%d\n", (LABELNO));	\
}
#define PROFILE_BEFORE_PROLOGUE
#define EXIT_IGNORE_STACK 1
#define TRAMPOLINE_SIZE (TARGET_FDPIC ? 32 : 16)
#define TRAMPOLINE_ALIGNMENT \
((CACHE_LOG < 3 \
|| (optimize_size && ! (TARGET_HARD_SH4))) ? 32 \
: 64)
#define RETURN_ADDR_RTX(COUNT, FRAME)	\
(((COUNT) == 0) ? sh_get_pr_initial_val () : NULL_RTX)
#define INCOMING_RETURN_ADDR_RTX gen_rtx_REG (Pmode, PR_REG)

#define HAVE_POST_INCREMENT  TARGET_SH1
#define HAVE_PRE_DECREMENT   TARGET_SH1
#define USE_LOAD_POST_INCREMENT(mode) TARGET_SH1
#define USE_LOAD_PRE_DECREMENT(mode) TARGET_SH2A
#define USE_STORE_POST_INCREMENT(mode) TARGET_SH2A
#define USE_STORE_PRE_DECREMENT(mode) TARGET_SH1
#define CLEAR_RATIO(speed) ((speed) ? 15 : 3)
#define REGNO_OK_FOR_BASE_P(REGNO) \
(GENERAL_OR_AP_REGISTER_P (REGNO) \
|| GENERAL_OR_AP_REGISTER_P (reg_renumber[(REGNO)]))
#define REGNO_OK_FOR_INDEX_P(REGNO) \
((REGNO) == R0_REG || (unsigned) reg_renumber[(REGNO)] == R0_REG)
#define SH_OFFSETS_MUST_BE_WITHIN_SECTIONS_P TARGET_FDPIC
#define MAX_REGS_PER_ADDRESS 2
#define CONSTANT_ADDRESS_P(X)	(GET_CODE (X) == LABEL_REF)
#define REG_OK_FOR_BASE_P(X, STRICT)			\
(GENERAL_OR_AP_REGISTER_P (REGNO (X))			\
|| (!STRICT && REGNO (X) >= FIRST_PSEUDO_REGISTER))
#define REG_OK_FOR_INDEX_P(X, STRICT)			\
((REGNO (X) == R0_REG)				\
|| (!STRICT && REGNO (X) >= FIRST_PSEUDO_REGISTER))
#define SUBREG_OK_FOR_INDEX_P(X, OFFSET, STRICT)	\
((REGNO (X) == R0_REG && OFFSET == 0)			\
|| (!STRICT && REGNO (X) >= FIRST_PSEUDO_REGISTER))
#define IS_PC_RELATIVE_LOAD_ADDR_P(OP)					\
((GET_CODE ((OP)) == LABEL_REF)					\
|| (GET_CODE ((OP)) == CONST						\
&& GET_CODE (XEXP ((OP), 0)) == PLUS				\
&& GET_CODE (XEXP (XEXP ((OP), 0), 0)) == LABEL_REF		\
&& CONST_INT_P (XEXP (XEXP ((OP), 0), 1))))
#define IS_NON_EXPLICIT_CONSTANT_P(OP)					\
(CONSTANT_P (OP)							\
&& !CONST_INT_P (OP)							\
&& GET_CODE (OP) != CONST_DOUBLE					\
&& (!flag_pic							\
|| (LEGITIMATE_PIC_OPERAND_P (OP)				\
&& !PIC_ADDR_P (OP)						\
&& GET_CODE (OP) != LABEL_REF)))
#define GOT_ENTRY_P(OP) \
(GET_CODE (OP) == CONST && GET_CODE (XEXP ((OP), 0)) == UNSPEC \
&& XINT (XEXP ((OP), 0), 1) == UNSPEC_GOT)
#define GOTPLT_ENTRY_P(OP) \
(GET_CODE (OP) == CONST && GET_CODE (XEXP ((OP), 0)) == UNSPEC \
&& XINT (XEXP ((OP), 0), 1) == UNSPEC_GOTPLT)
#define UNSPEC_GOTOFF_P(OP) \
(GET_CODE (OP) == UNSPEC && XINT ((OP), 1) == UNSPEC_GOTOFF)
#define GOTOFF_P(OP) \
(GET_CODE (OP) == CONST \
&& (UNSPEC_GOTOFF_P (XEXP ((OP), 0)) \
|| (GET_CODE (XEXP ((OP), 0)) == PLUS \
&& UNSPEC_GOTOFF_P (XEXP (XEXP ((OP), 0), 0)) \
&& CONST_INT_P (XEXP (XEXP ((OP), 0), 1)))))
#define PIC_ADDR_P(OP) \
(GET_CODE (OP) == CONST && GET_CODE (XEXP ((OP), 0)) == UNSPEC \
&& XINT (XEXP ((OP), 0), 1) == UNSPEC_PIC)
#define PCREL_SYMOFF_P(OP) \
(GET_CODE (OP) == CONST \
&& GET_CODE (XEXP ((OP), 0)) == UNSPEC \
&& XINT (XEXP ((OP), 0), 1) == UNSPEC_PCREL_SYMOFF)
#define NON_PIC_REFERENCE_P(OP) \
(GET_CODE (OP) == LABEL_REF || GET_CODE (OP) == SYMBOL_REF \
|| (GET_CODE (OP) == CONST \
&& (GET_CODE (XEXP ((OP), 0)) == LABEL_REF \
|| GET_CODE (XEXP ((OP), 0)) == SYMBOL_REF)) \
|| (GET_CODE (OP) == CONST && GET_CODE (XEXP ((OP), 0)) == PLUS \
&& (GET_CODE (XEXP (XEXP ((OP), 0), 0)) == SYMBOL_REF \
|| GET_CODE (XEXP (XEXP ((OP), 0), 0)) == LABEL_REF) \
&& CONST_INT_P (XEXP (XEXP ((OP), 0), 1))))
#define PIC_REFERENCE_P(OP) \
(GOT_ENTRY_P (OP) || GOTPLT_ENTRY_P (OP) \
|| GOTOFF_P (OP) || PIC_ADDR_P (OP))
#define MAYBE_BASE_REGISTER_RTX_P(X, STRICT)			\
((REG_P (X) && REG_OK_FOR_BASE_P (X, STRICT))	\
|| (GET_CODE (X) == SUBREG					\
&& REG_P (SUBREG_REG (X))			\
&& REG_OK_FOR_BASE_P (SUBREG_REG (X), STRICT)))
#define MAYBE_INDEX_REGISTER_RTX_P(X, STRICT)				\
((REG_P (X) && REG_OK_FOR_INDEX_P (X, STRICT))	\
|| (GET_CODE (X) == SUBREG					\
&& REG_P (SUBREG_REG (X))		\
&& SUBREG_OK_FOR_INDEX_P (SUBREG_REG (X), SUBREG_BYTE (X), STRICT)))
#ifdef REG_OK_STRICT
#define BASE_REGISTER_RTX_P(X) MAYBE_BASE_REGISTER_RTX_P(X, true)
#define INDEX_REGISTER_RTX_P(X) MAYBE_INDEX_REGISTER_RTX_P(X, true)
#else
#define BASE_REGISTER_RTX_P(X) MAYBE_BASE_REGISTER_RTX_P(X, false)
#define INDEX_REGISTER_RTX_P(X) MAYBE_INDEX_REGISTER_RTX_P(X, false)
#endif

#define LEGITIMIZE_RELOAD_ADDRESS(X,MODE,OPNUM,TYPE,IND_LEVELS,WIN)	\
do {									\
if (sh_legitimize_reload_address (&(X), (MODE), (OPNUM), (TYPE)))	\
goto WIN;								\
} while (0)

#define CASE_VECTOR_MODE ((! optimize || TARGET_BIGTABLE) ? SImode : HImode)
#define CASE_VECTOR_SHORTEN_MODE(MIN_OFFSET, MAX_OFFSET, BODY) \
((MIN_OFFSET) >= 0 && (MAX_OFFSET) <= 127 \
? (ADDR_DIFF_VEC_FLAGS (BODY).offset_unsigned = 0, QImode) \
: (MIN_OFFSET) >= 0 && (MAX_OFFSET) <= 255 \
? (ADDR_DIFF_VEC_FLAGS (BODY).offset_unsigned = 1, QImode) \
: (MIN_OFFSET) >= -32768 && (MAX_OFFSET) <= 32767 ? HImode \
: SImode)
#define CASE_VECTOR_PC_RELATIVE 1
#define FLOAT_TYPE_SIZE 32
#define DOUBLE_TYPE_SIZE (TARGET_FPU_SINGLE_ONLY ? 32 : 64)
#define DEFAULT_SIGNED_CHAR  1
#define SIZE_TYPE ("unsigned int")
#undef  PTRDIFF_TYPE
#define PTRDIFF_TYPE ("int")
#define WCHAR_TYPE "short unsigned int"
#define WCHAR_TYPE_SIZE 16
#define SH_ELF_WCHAR_TYPE "long int"
#define MOVE_MAX (4)
#define MAX_MOVE_MAX 8
#define MOVE_MAX_PIECES (TARGET_SH4 ? 8 : 4)
#define WORD_REGISTER_OPERATIONS 1
#define LOAD_EXTEND_OP(MODE) ((MODE) != SImode ? SIGN_EXTEND : UNKNOWN)
#define SHORT_IMMEDIATES_SIGN_EXTEND 1
#define SLOW_BYTE_ACCESS 1
#define TARGET_DYNSHIFT (TARGET_SH3 || TARGET_SH2A)
#define SH_DYNAMIC_SHIFT_COST (TARGET_DYNSHIFT ? 1 : 20)
#define SHIFT_COUNT_TRUNCATED (0)
#define Pmode  (SImode)
#define FUNCTION_MODE  Pmode
#define INSN_SETS_ARE_DELAYED(X) 		\
((NONJUMP_INSN_P (X)				\
&& GET_CODE (PATTERN (X)) != SEQUENCE	\
&& GET_CODE (PATTERN (X)) != USE		\
&& GET_CODE (PATTERN (X)) != CLOBBER	\
&& get_attr_is_sfunc (X)))
#define INSN_REFERENCES_ARE_DELAYED(X) 		\
((NONJUMP_INSN_P (X)				\
&& GET_CODE (PATTERN (X)) != SEQUENCE	\
&& GET_CODE (PATTERN (X)) != USE		\
&& GET_CODE (PATTERN (X)) != CLOBBER	\
&& get_attr_is_sfunc (X)))

#define LEGITIMATE_PIC_OPERAND_P(X)				\
((! nonpic_symbol_mentioned_p (X)			\
&& (GET_CODE (X) != SYMBOL_REF			\
|| ! CONSTANT_POOL_ADDRESS_P (X)			\
|| ! nonpic_symbol_mentioned_p (get_pool_constant (X)))))
#define SYMBOLIC_CONST_P(X)	\
((GET_CODE (X) == SYMBOL_REF || GET_CODE (X) == LABEL_REF)	\
&& nonpic_symbol_mentioned_p (X))

#define REGCLASS_HAS_GENERAL_REG(CLASS) \
((CLASS) == GENERAL_REGS || (CLASS) == R0_REGS || (CLASS) == NON_SP_REGS \
|| ((CLASS) == SIBCALL_REGS))
#define REGCLASS_HAS_FP_REG(CLASS) \
((CLASS) == FP0_REGS || (CLASS) == FP_REGS \
|| (CLASS) == DF_REGS)
#define BRANCH_COST(speed_p, predictable_p) sh_branch_cost

#define ASM_COMMENT_START "!"
#define ASM_APP_ON  		""
#define ASM_APP_OFF  		""
#define FILE_ASM_OP 		"\t.file\n"
#define SET_ASM_OP		"\t.set\t"
#define TEXT_SECTION_ASM_OP	"\t.text"
#define DATA_SECTION_ASM_OP	"\t.data"
#if defined CRT_BEGIN || defined CRT_END
#undef TEXT_SECTION_ASM_OP
#define TEXT_SECTION_ASM_OP "\t.text"
#endif
#ifndef BSS_SECTION_ASM_OP
#define BSS_SECTION_ASM_OP	"\t.section\t.bss"
#endif
#ifndef ASM_OUTPUT_ALIGNED_BSS
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN) \
asm_output_aligned_bss (FILE, DECL, NAME, SIZE, ALIGN)
#endif
#define JUMP_TABLES_IN_TEXT_SECTION 1
#undef DO_GLOBAL_CTORS_BODY
#define DO_GLOBAL_CTORS_BODY			\
{						\
typedef void (*pfunc) (void);			\
extern pfunc __ctors[];			\
extern pfunc __ctors_end[];			\
pfunc *p;					\
for (p = __ctors_end; p > __ctors; )		\
{						\
(*--p)();					\
}						\
}
#undef DO_GLOBAL_DTORS_BODY
#define DO_GLOBAL_DTORS_BODY			\
{						\
typedef void (*pfunc) (void);			\
extern pfunc __dtors[];			\
extern pfunc __dtors_end[];			\
pfunc *p;					\
for (p = __dtors; p < __dtors_end; p++)	\
{						\
(*p)();					\
}						\
}
#define ASM_OUTPUT_REG_PUSH(file, v) \
{							\
fprintf ((file), "\tmov.l\tr%d,@-r15\n", (v));	\
}
#define ASM_OUTPUT_REG_POP(file, v) \
{							\
fprintf ((file), "\tmov.l\t@r15+,r%d\n", (v));	\
}
#define DBX_REGISTER_NUMBER(REGNO) SH_DBX_REGISTER_NUMBER (REGNO)
#define SH_DBX_REGISTER_NUMBER(REGNO) \
(IN_RANGE ((REGNO), \
(unsigned HOST_WIDE_INT) FIRST_GENERAL_REG, \
FIRST_GENERAL_REG + 15U) \
? ((unsigned) (REGNO) - FIRST_GENERAL_REG) \
: ((int) (REGNO) >= FIRST_FP_REG \
&& ((int) (REGNO) \
<= (FIRST_FP_REG + (TARGET_SH2E ? 15 : -1)))) \
? ((unsigned) (REGNO) - FIRST_FP_REG + 25) \
: XD_REGISTER_P (REGNO) \
? ((unsigned) (REGNO) - FIRST_XD_REG + 87) \
: (REGNO) == PR_REG \
? (17) \
: (REGNO) == GBR_REG \
? (18) \
: (REGNO) == MACH_REG \
? (20) \
: (REGNO) == MACL_REG \
? (21) \
: (REGNO) == T_REG \
? (22) \
: (REGNO) == FPUL_REG \
? (23) \
: (REGNO) == FPSCR_REG \
? (24) \
: (unsigned) -1)
#define ASM_OUTPUT_ALIGN(FILE,LOG)	\
if ((LOG) != 0)			\
fprintf ((FILE), "\t.align %d\n", (LOG))
#define GLOBAL_ASM_OP "\t.global\t"
#define ASM_OUTPUT_ADDR_DIFF_ELT(STREAM,BODY,VALUE,REL)			\
switch (GET_MODE (BODY))						\
{									\
case E_SImode:							\
asm_fprintf ((STREAM), "\t.long\t%LL%d-%LL%d\n", (VALUE),(REL));	\
break;								\
case E_HImode:							\
asm_fprintf ((STREAM), "\t.word\t%LL%d-%LL%d\n", (VALUE),(REL));	\
break;								\
case E_QImode:							\
asm_fprintf ((STREAM), "\t.byte\t%LL%d-%LL%d\n", (VALUE),(REL));	\
break;								\
default:								\
break;								\
}
#define ASM_OUTPUT_ADDR_VEC_ELT(STREAM,VALUE) \
do {									\
if (! optimize || TARGET_BIGTABLE)					\
asm_fprintf ((STREAM), "\t.long\t%LL%d\n", (VALUE)); 		\
else								\
asm_fprintf ((STREAM), "\t.word\t%LL%d\n", (VALUE));		\
} while (0)

#define FINAL_PRESCAN_INSN(INSN, OPVEC, NOPERANDS) \
final_prescan_insn ((INSN), (OPVEC), (NOPERANDS))
enum processor_type {
PROCESSOR_SH1,
PROCESSOR_SH2,
PROCESSOR_SH2E,
PROCESSOR_SH2A,
PROCESSOR_SH3,
PROCESSOR_SH3E,
PROCESSOR_SH4,
PROCESSOR_SH4A
};
#define sh_cpu_attr ((enum attr_cpu)sh_cpu)
extern enum processor_type sh_cpu;
enum mdep_reorg_phase_e
{
SH_BEFORE_MDEP_REORG,
SH_INSERT_USES_LABELS,
SH_SHORTEN_BRANCHES0,
SH_FIXUP_PCLOAD,
SH_SHORTEN_BRANCHES1,
SH_AFTER_MDEP_REORG
};
extern enum mdep_reorg_phase_e mdep_reorg_phase;
#define REGISTER_TARGET_PRAGMAS() do {					\
c_register_pragma (0, "interrupt", sh_pr_interrupt);			\
c_register_pragma (0, "trapa", sh_pr_trapa);				\
c_register_pragma (0, "nosave_low_regs", sh_pr_nosave_low_regs);	\
} while (0)
extern tree sh_deferred_function_attributes;
extern tree *sh_deferred_function_attributes_tail;

#define ADJUST_INSN_LENGTH(X, LENGTH)				\
(LENGTH) += sh_insn_length_adjustment (X);

#define PROMOTE_MODE(MODE, UNSIGNEDP, TYPE) \
if (GET_MODE_CLASS (MODE) == MODE_INT			\
&& GET_MODE_SIZE (MODE) < 4)\
(UNSIGNEDP) = ((MODE) == SImode ? 0 : (UNSIGNEDP)),	(MODE) = SImode;
#define MAX_FIXED_MODE_SIZE (64)
#define ACCUMULATE_OUTGOING_ARGS TARGET_ACCUMULATE_OUTGOING_ARGS
#define NUM_MODES_FOR_MODE_SWITCHING { FP_MODE_NONE }
#define OPTIMIZE_MODE_SWITCHING(ENTITY) (TARGET_FPU_DOUBLE)
#define ACTUAL_NORMAL_MODE(ENTITY) \
(TARGET_FPU_SINGLE ? FP_MODE_SINGLE : FP_MODE_DOUBLE)
#define NORMAL_MODE(ENTITY) \
(sh_cfun_interrupt_handler_p () \
? (TARGET_FMOVD ? FP_MODE_DOUBLE : FP_MODE_NONE) \
: ACTUAL_NORMAL_MODE (ENTITY))
#define EPILOGUE_USES(REGNO) (TARGET_FPU_ANY && REGNO == FPSCR_REG)
#define DWARF_FRAME_RETURN_COLUMN (DWARF_FRAME_REGNUM (PR_REG))
#define EH_RETURN_DATA_REGNO(N)	((N) < 4 ? (N) + 4U : INVALID_REGNUM)
#define EH_RETURN_STACKADJ_REGNO STATIC_CHAIN_REGNUM
#define EH_RETURN_STACKADJ_RTX	gen_rtx_REG (Pmode, EH_RETURN_STACKADJ_REGNO)
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE, GLOBAL) \
((TARGET_FDPIC \
? ((GLOBAL) ? DW_EH_PE_indirect | DW_EH_PE_datarel : DW_EH_PE_pcrel) \
: ((flag_pic && (GLOBAL) ? DW_EH_PE_indirect : 0) \
| (flag_pic ? DW_EH_PE_pcrel : DW_EH_PE_absptr))) \
| ((CODE) ? 0 : DW_EH_PE_sdata4))
#define ASM_MAYBE_OUTPUT_ENCODED_ADDR_RTX(FILE, ENCODING, SIZE, ADDR, DONE) \
do { \
if (((ENCODING) & 0xf) != DW_EH_PE_sdata4 \
&& ((ENCODING) & 0xf) != DW_EH_PE_sdata8) \
{ \
gcc_assert (GET_CODE (ADDR) == SYMBOL_REF); \
SYMBOL_REF_FLAGS (ADDR) |= SYMBOL_FLAG_FUNCTION; \
if (0) goto DONE; \
} \
if (TARGET_FDPIC \
&& ((ENCODING) & 0xf0) == (DW_EH_PE_indirect | DW_EH_PE_datarel)) \
{ \
fputs ("\t.ualong ", FILE); \
output_addr_const (FILE, ADDR); \
if (GET_CODE (ADDR) == SYMBOL_REF && SYMBOL_REF_FUNCTION_P (ADDR)) \
fputs ("@GOTFUNCDESC", FILE); \
else \
fputs ("@GOT", FILE); \
goto DONE; \
} \
} while (0)
#if (defined CRT_BEGIN || defined CRT_END)
#define CRT_CALL_STATIC_FUNCTION(SECTION_OP, FUNC) \
asm (SECTION_OP "\n\
mov.l	1f,r1\n\
mova	2f,r0\n\
braf	r1\n\
lds	r0,pr\n\
0:	.p2align 2\n\
1:	.long	" USER_LABEL_PREFIX #FUNC " - 0b\n\
2:\n" TEXT_SECTION_ASM_OP);
#endif 
#endif 
