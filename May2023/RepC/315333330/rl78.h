
#define RL78_MUL_NONE	(rl78_mul_type == MUL_NONE)
#define RL78_MUL_G13	(rl78_mul_type == MUL_G13)
#define RL78_MUL_G14	(rl78_mul_type == MUL_G14)
#define TARGET_G10	(rl78_cpu_type == CPU_G10)
#define TARGET_G13	(rl78_cpu_type == CPU_G13)
#define TARGET_G14	(rl78_cpu_type == CPU_G14)
#define TARGET_CPU_CPP_BUILTINS()               \
do                                            \
{                                           \
builtin_define ("__RL78__"); 		\
builtin_assert ("cpu=RL78"); 		\
\
if (RL78_MUL_NONE)			\
builtin_define ("__RL78_MUL_NONE__"); 	\
else if (RL78_MUL_G13)			\
builtin_define ("__RL78_MUL_G13__"); 	\
else if (RL78_MUL_G14)			\
builtin_define ("__RL78_MUL_G14__"); 	\
\
if (TARGET_G10)				\
builtin_define ("__RL78_G10__"); 	\
else if (TARGET_G13)			\
builtin_define ("__RL78_G13__"); 	\
else if (TARGET_G14)			\
builtin_define ("__RL78_G14__"); 	\
}                                           \
while (0)
#undef  STARTFILE_SPEC
#define STARTFILE_SPEC "%{pg:gcrt0.o%s}%{!pg:crt0.o%s} crtbegin.o%s"
#undef  ENDFILE_SPEC
#define ENDFILE_SPEC "crtend.o%s crtn.o%s"
#undef  ASM_SPEC
#define ASM_SPEC "\
%{mrelax:-relax} \
%{mg10:--mg10} \
%{mg13:--mg13} \
%{mg14:--mg14} \
%{mrl78:--mg14} \
%{mcpu=g10:--mg10} \
%{mcpu=g13:--mg13} \
%{mcpu=g14:--mg14} \
%{mcpu=rl78:--mg14} \
"
#undef  LINK_SPEC
#define LINK_SPEC "\
%{mrelax:-relax} \
%{!r:--gc-sections} \
"
#undef  LIB_SPEC
#define LIB_SPEC "					\
--start-group						\
-lc							\
-lsim							\
%{fprofile-arcs|fprofile-generate|coverage:-lgcov} 	\
--end-group					   	\
%{!T*: %{msim:%Trl78-sim.ld}%{!msim:%Trl78.ld}}		\
"

#define BITS_BIG_ENDIAN 		0
#define BYTES_BIG_ENDIAN 		0
#define WORDS_BIG_ENDIAN 		0
#ifdef IN_LIBGCC2
#define	UNITS_PER_WORD			4
#ifndef LIBGCC2_UNITS_PER_WORD
#define LIBGCC2_UNITS_PER_WORD 		4
#endif
#else
#define	UNITS_PER_WORD 			1
#endif
#define SHORT_TYPE_SIZE			16
#define INT_TYPE_SIZE			16
#define LONG_TYPE_SIZE			32
#define LONG_LONG_TYPE_SIZE		64
#define FLOAT_TYPE_SIZE 		32
#define DOUBLE_TYPE_SIZE 		32 
#define LONG_DOUBLE_TYPE_SIZE		64 
#define DEFAULT_SIGNED_CHAR		0
#define STRICT_ALIGNMENT 		1
#define FUNCTION_BOUNDARY 		8
#define BIGGEST_ALIGNMENT 		16
#define STACK_BOUNDARY 			16
#define PARM_BOUNDARY 			16
#define STACK_GROWS_DOWNWARD		1
#define FRAME_GROWS_DOWNWARD		1
#define FIRST_PARM_OFFSET(FNDECL) 	0
#define MAX_REGS_PER_ADDRESS 		1
#define Pmode 				HImode
#define POINTER_SIZE			16
#undef  SIZE_TYPE
#define SIZE_TYPE			"unsigned int"
#undef  PTRDIFF_TYPE
#define PTRDIFF_TYPE			"int"
#undef  WCHAR_TYPE
#define WCHAR_TYPE			"long int"
#undef  WCHAR_TYPE_SIZE
#define WCHAR_TYPE_SIZE			BITS_PER_WORD
#define POINTERS_EXTEND_UNSIGNED	1
#define FUNCTION_MODE 			HImode
#define CASE_VECTOR_MODE		Pmode
#define WORD_REGISTER_OPERATIONS	1
#define HAS_LONG_COND_BRANCH		0
#define HAS_LONG_UNCOND_BRANCH		0
#define MOVE_MAX 			2
#define ADDR_SPACE_NEAR			1
#define ADDR_SPACE_FAR			2
#define HAVE_PRE_DECCREMENT		0
#define HAVE_POST_INCREMENT		0
#define MOVE_RATIO(SPEED) 		((SPEED) ? 24 : 16)
#define SLOW_BYTE_ACCESS		0
#define STORE_FLAG_VALUE		1
#define LOAD_EXTEND_OP(MODE)		ZERO_EXTEND

#define REGISTER_NAMES						\
{								\
"x",   "a",   "c",   "b",   "e",   "d",   "l",   "h", 	\
"r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",	\
"r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",	\
"r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",	\
"sp",  "ap",  "psw", "es",  "cs"				\
}
#define ADDITIONAL_REGISTER_NAMES	\
{					\
{ "ax", 0 }, \
{ "bc", 2 }, \
{ "de", 4 }, \
{ "hl", 6 }, \
{ "rp0", 0 }, \
{ "rp1", 2 }, \
{ "rp2", 4 }, \
{ "rp3", 6 }, \
{ "r0", 0 }, \
{ "r1", 1 }, \
{ "r2", 2 }, \
{ "r3", 3 }, \
{ "r4", 4 }, \
{ "r5", 5 }, \
{ "r6", 6 }, \
{ "r7", 7 }, \
}
enum reg_class
{
NO_REGS,			
XREG,
AREG,
AXREG,
CREG,
BREG,
BCREG,
EREG,
DREG,
DEREG,
LREG,
HREG,
HLREG,
IDX_REGS,
QI_REGS,
SPREG,
R8W_REGS,
R10W_REGS,
INT_REGS,
V_REGS,			
GR_REGS,			
PSWREG,
ALL_REGS,			
LIM_REG_CLASSES		
};
#define REG_CLASS_NAMES					\
{							\
"NO_REGS",						\
"XREG",						\
"AREG",						\
"AXREG",						\
"CREG",						\
"BREG",						\
"BCREG",						\
"EREG",						\
"DREG",						\
"DEREG",						\
"LREG",						\
"HREG",						\
"HLREG",						\
"IDX_REGS",						\
"QI_REGS",						\
"SPREG",						\
"R8W_REGS",						\
"R10W_REGS",						\
"INT_REGS",						\
"V_REGS",						\
"GR_REGS",						\
"PSWREG",						\
"ALL_REGS"						\
}
#define REG_CLASS_CONTENTS				\
{							\
{ 0x00000000, 0x00000000 },			\
{ 0x00000001, 0x00000000 }, \
{ 0x00000002, 0x00000000 }, \
{ 0x00000003, 0x00000000 }, \
{ 0x00000004, 0x00000000 }, \
{ 0x00000008, 0x00000000 }, \
{ 0x0000000c, 0x00000000 }, \
{ 0x00000010, 0x00000000 }, \
{ 0x00000020, 0x00000000 }, \
{ 0x00000030, 0x00000000 }, \
{ 0x00000040, 0x00000000 }, \
{ 0x00000080, 0x00000000 }, \
{ 0x000000c0, 0x00000000 }, \
{ 0x0000000c, 0x00000000 },		\
{ 0x000000ff, 0x00000000 },		\
{ 0x00000000, 0x00000001 }, 				\
{ 0x00000300, 0x00000000 }, 			\
{ 0x00000c00, 0x00000000 }, 			\
{ 0xff000000, 0x00000000 }, 			\
{ 0xff7fff00, 0x00000000 },		\
{ 0xff7fff00, 0x00000002 },		\
{ 0x04000000, 0x00000004 },		\
{ 0xff7fffff, 0x0000001f }			\
}
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P hook_bool_mode_true
#define N_REG_CLASSES			(int) LIM_REG_CLASSES
#define CLASS_MAX_NREGS(CLASS, MODE)    ((GET_MODE_SIZE (MODE) \
+ UNITS_PER_WORD - 1) \
/ UNITS_PER_WORD)
#define GENERAL_REGS			GR_REGS
#define BASE_REG_CLASS  		V_REGS
#define INDEX_REG_CLASS			V_REGS
#define FIRST_PSEUDO_REGISTER 		37
#define REGNO_REG_CLASS(REGNO)          ((REGNO) < FIRST_PSEUDO_REGISTER \
? GR_REGS : NO_REGS)
#define FRAME_POINTER_REGNUM 		22
#define STACK_POINTER_REGNUM 	        32
#define ARG_POINTER_REGNUM 		33
#define CC_REGNUM                       34
#define FUNC_RETURN_REGNUM              8
#define STATIC_CHAIN_REGNUM 		14
#define TRAMPOLINE_SIZE			4
#define TRAMPOLINE_ALIGNMENT		16
#define ELIMINABLE_REGS					\
{{ ARG_POINTER_REGNUM,   STACK_POINTER_REGNUM },	\
{ ARG_POINTER_REGNUM,   FRAME_POINTER_REGNUM },	\
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM }}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)	\
(OFFSET) = rl78_initial_elimination_offset ((FROM), (TO))
#define FUNCTION_ARG_REGNO_P(N)	  	0
#define FUNCTION_VALUE_REGNO_P(N) 	((N) == 8)
#define DEFAULT_PCC_STRUCT_RETURN	0
#define FIXED_REGISTERS					\
{							\
1,1,1,1, 1,1,1,1,					\
0,0,0,0, 0,0,0,0,					\
0,0,0,0, 0,0,1,1,					\
1,1,1,1, 1,1,1,1,					\
0, 1, 0, 1, 1						\
}
#define CALL_USED_REGISTERS				\
{							\
1,1,1,1, 1,1,1,1,					\
1,1,1,1, 1,1,1,1,					\
0,0,0,0, 0,0,1,1,					\
1,1,1,1, 1,1,1,1,					\
0, 1, 1, 1, 1						\
}
#define LIBCALL_VALUE(MODE)				\
gen_rtx_REG ((MODE),					\
FUNC_RETURN_REGNUM)
#define REG_ALLOC_ORDER					\
{ 8, 9, 10, 11, 12, 13, 14, 15,			\
16, 17, 18, 19, 20, 21, 22, 23,			\
0, 1, 6, 7, 2, 3, 4, 5,				\
24, 25, 26, 27, 28, 29, 30, 31,			\
32, 33, 34						\
}
#define REGNO_IN_RANGE(REGNO, MIN, MAX)			\
(IN_RANGE ((REGNO), (MIN), (MAX)) 			\
|| (reg_renumber != NULL				\
&& reg_renumber[(REGNO)] >= (MIN)		\
&& reg_renumber[(REGNO)] <= (MAX)))
#ifdef REG_OK_STRICT
#define REGNO_OK_FOR_BASE_P(regno)      REGNO_IN_RANGE (regno, 16, 31)
#else
#define REGNO_OK_FOR_BASE_P(regno)	1
#endif
#define REGNO_OK_FOR_INDEX_P(regno)	REGNO_OK_FOR_BASE_P (regno)
#define REGNO_MODE_CODE_OK_FOR_BASE_P(regno, mode, address_space, outer_code, index_code) \
rl78_regno_mode_code_ok_for_base_p (regno, mode, address_space, outer_code, index_code)
#define MODE_CODE_BASE_REG_CLASS(mode, address_space, outer_code, index_code) \
rl78_mode_code_base_reg_class (mode, address_space, outer_code, index_code)
#define RETURN_ADDR_RTX(COUNT, FRAMEADDR)				\
((COUNT) == 0								\
? gen_rtx_MEM (Pmode, gen_rtx_PLUS (HImode, arg_pointer_rtx, GEN_INT (-4))) \
: NULL_RTX)
#define INCOMING_RETURN_ADDR_RTX	gen_rtx_MEM (Pmode, stack_pointer_rtx)
#define ACCUMULATE_OUTGOING_ARGS	1
typedef unsigned int CUMULATIVE_ARGS;
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, INDIRECT, N_NAMED_ARGS) \
(CUM) = 0

#define NO_PROFILE_COUNTERS     1
#define PROFILE_BEFORE_PROLOGUE 1
#define FUNCTION_PROFILER(FILE, LABELNO)	\
fprintf (FILE, "\tbsr\t__mcount\n");

#define TEXT_SECTION_ASM_OP ".text"
#define DATA_SECTION_ASM_OP ".data"
#define BSS_SECTION_ASM_OP ".bss"
#define CTORS_SECTION_ASM_OP ".section \".ctors\",\"a\""
#define DTORS_SECTION_ASM_OP ".section \".dtors\",\"a\""
#define ASM_COMMENT_START	" ;"
#define ASM_APP_ON		""
#define ASM_APP_OFF 		""
#define LOCAL_LABEL_PREFIX	".L"
#undef  USER_LABEL_PREFIX
#define USER_LABEL_PREFIX	"_"
#define GLOBAL_ASM_OP 		"\t.global\t"
#define ASM_OUTPUT_ADDR_VEC_ELT(FILE, VALUE) \
fprintf (FILE, "\t.long .L%d\n", VALUE)
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL) \
fprintf (FILE, "\t.long .L%d - 1b\n", VALUE)
#define ASM_OUTPUT_SYMBOL_REF(FILE, SYM) rl78_output_symbol_ref ((FILE), (SYM))
#define ASM_OUTPUT_LABELREF(FILE, SYM) rl78_output_labelref ((FILE), (SYM))
#define ASM_OUTPUT_ALIGNED_DECL_COMMON(STREAM, DECL, NAME, SIZE, ALIGNMENT) \
rl78_output_aligned_common (STREAM, DECL, NAME, SIZE, ALIGNMENT, 1)
#define ASM_OUTPUT_ALIGNED_DECL_LOCAL(STREAM, DECL, NAME, SIZE, ALIGNMENT) \
rl78_output_aligned_common (STREAM, DECL, NAME, SIZE, ALIGNMENT, 0)
#define ASM_OUTPUT_ALIGN(STREAM, LOG)		\
do						\
{						\
if ((LOG) == 0)				\
break;					\
fprintf (STREAM, "\t.balign %d\n", 1 << (LOG));	\
}						\
while (0)
#define JUMP_TABLES_IN_TEXT_SECTION	(flag_pic)

#define RL78_REG_P(rtl) (REG_P (rtl) || GET_CODE (rtl) == SUBREG)
#define SET_P(rtl)    (GET_CODE (rtl) == SET)

#undef  PREFERRED_DEBUGGING_TYPE
#define PREFERRED_DEBUGGING_TYPE DWARF2_DEBUG
#undef	DWARF2_ADDR_SIZE
#define	DWARF2_ADDR_SIZE			4
#define DWARF2_ASM_LINE_DEBUG_INFO		1
#define EXIT_IGNORE_STACK			0
#define INCOMING_FRAME_SP_OFFSET		4

#define BRANCH_COST(SPEED,PREDICT)       1
#define REGISTER_MOVE_COST(MODE,FROM,TO) 2
#define EH_RETURN_DATA_REGNO(N) (N < 2 ? (8+(N)*2) : INVALID_REGNUM)
#define EH_RETURN_STACKADJ_RTX gen_rtx_REG (HImode, 20)
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE,GLOBAL) DW_EH_PE_udata4
#define DWARF2_UNWIND_INFO 0
#define REGISTER_TARGET_PRAGMAS() rl78_register_pragmas()
