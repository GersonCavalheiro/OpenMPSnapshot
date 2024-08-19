
#ifndef IN_LIBGCC2
extern bool msp430x;
#endif
#define TARGET_CPU_CPP_BUILTINS()               \
do                                            \
{                                           \
builtin_define ("NO_TRAMPOLINES");        \
builtin_define ("__MSP430__"); 		\
builtin_define (msp430_mcu_name ());	\
if (msp430x)				\
{					\
builtin_define ("__MSP430X__");	\
builtin_assert ("cpu=MSP430X");	\
if (TARGET_LARGE)			\
builtin_define ("__MSP430X_LARGE__");	\
}					\
else					\
builtin_assert ("cpu=MSP430"); 		\
}                                           \
while (0)
#undef  STARTFILE_SPEC
#define STARTFILE_SPEC "%{pg:gcrt0.o%s}%{!pg:%{minrt:crt0-minrt.o%s}%{!minrt:crt0.o%s}} %{!minrt:crtbegin.o%s}"
#undef  ENDFILE_SPEC
#define ENDFILE_SPEC "%{!minrt:crtend.o%s} %{minrt:crtn-minrt.o%s}%{!minrt:crtn.o%s} -lgcc"
#define ASM_SPEC "-mP "  \
"%{mcpu=*:-mcpu=%*}%{!mcpu=*:%{mmcu=*:-mmcu=%*}} "  \
"%{mrelax=-mQ} "  \
"%{mlarge:-ml} "  \
"%{!msim:-md} %{msim:%{mlarge:-md}} "  \
"%{msilicon-errata=*:-msilicon-errata=%*} "  \
"%{msilicon-errata-warn=*:-msilicon-errata-warn=%*} "  \
"%{ffunction-sections:-gdwarf-sections} "  \
"%{mdata-region=*:-mdata-region=%*} " 
#define LINK_SPEC "%{mrelax:--relax} %{mlarge:%{!r:%{!g:--gc-sections}}} " \
"%{mcode-region=*:--code-region=%*} %{mdata-region=*:--data-region=%*}"
extern const char * msp430_select_hwmult_lib (int, const char **);
# define EXTRA_SPEC_FUNCTIONS				\
{ "msp430_hwmult_lib", msp430_select_hwmult_lib },
#undef  LIB_SPEC
#define LIB_SPEC "					\
--start-group						\
%{mhwmult=auto:%{mmcu=*:%:msp430_hwmult_lib(mcu %{mmcu=*:%*});:%:msp430_hwmult_lib(default)}; \
mhwmult=*:%:msp430_hwmult_lib(hwmult %{mhwmult=*:%*}); \
mmcu=*:%:msp430_hwmult_lib(mcu %{mmcu=*:%*});		\
:%:msp430_hwmult_lib(default)}			\
-lc							\
-lgcc							\
-lcrt							\
%{msim:-lsim}						\
%{!msim:-lnosys}					\
--end-group					   	\
%{!T*:%{!msim:%{mmcu=*:--script=%*.ld}}}		\
%{!T*:%{msim:%{mlarge:%Tmsp430xl-sim.ld}%{!mlarge:%Tmsp430-sim.ld}}} \
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
#define	UNITS_PER_WORD 			2
#endif
#define SHORT_TYPE_SIZE			16
#define INT_TYPE_SIZE			16
#define LONG_TYPE_SIZE			32
#define LONG_LONG_TYPE_SIZE		64
#define FLOAT_TYPE_SIZE 		32
#define DOUBLE_TYPE_SIZE 		64
#define LONG_DOUBLE_TYPE_SIZE		64 
#define DEFAULT_SIGNED_CHAR		0
#define STRICT_ALIGNMENT 		1
#define FUNCTION_BOUNDARY 		16
#define BIGGEST_ALIGNMENT 		16
#define STACK_BOUNDARY 			16
#define PARM_BOUNDARY 			8
#define PCC_BITFIELD_TYPE_MATTERS	1
#define STACK_GROWS_DOWNWARD		1
#define FRAME_GROWS_DOWNWARD		1
#define FIRST_PARM_OFFSET(FNDECL) 	0
#define MAX_REGS_PER_ADDRESS 		1
#define Pmode 				(TARGET_LARGE ? PSImode : HImode)
#define POINTER_SIZE			(TARGET_LARGE ? 20 : 16)
#define PTR_SIZE			(TARGET_LARGE ? 4 : 2)
#define	POINTERS_EXTEND_UNSIGNED	1
#define ADDR_SPACE_NEAR	1
#define ADDR_SPACE_FAR	2
#define REGISTER_TARGET_PRAGMAS() msp430_register_pragmas()
#if 1 
#define PROMOTE_MODE(MODE, UNSIGNEDP, TYPE)	\
if (GET_MODE_CLASS (MODE) == MODE_INT		\
&& GET_MODE_SIZE (MODE) < 2)      	\
(MODE) = HImode;
#endif

#undef  SIZE_TYPE
#define SIZE_TYPE			(TARGET_LARGE ? "__int20 unsigned" : "unsigned int")
#undef  PTRDIFF_TYPE
#define PTRDIFF_TYPE			(TARGET_LARGE ? "__int20" : "int")
#undef  WCHAR_TYPE
#define WCHAR_TYPE			"long int"
#undef  WCHAR_TYPE_SIZE
#define WCHAR_TYPE_SIZE			BITS_PER_WORD
#define FUNCTION_MODE 			HImode
#define CASE_VECTOR_MODE		Pmode
#define HAS_LONG_COND_BRANCH		0
#define HAS_LONG_UNCOND_BRANCH		0
#define LOAD_EXTEND_OP(M)		ZERO_EXTEND
#define WORD_REGISTER_OPERATIONS	1
#define MOVE_MAX 			8
#define INCOMING_RETURN_ADDR_RTX \
msp430_incoming_return_addr_rtx ()
#define RETURN_ADDR_RTX(COUNT, FA)		\
msp430_return_addr_rtx (COUNT)
#define SLOW_BYTE_ACCESS		0

#define REGISTER_NAMES						\
{								\
"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7",		\
"R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",	\
"argptr"							\
}
enum reg_class
{
NO_REGS,
R12_REGS,
R13_REGS,
GEN_REGS,
ALL_REGS,
LIM_REG_CLASSES
};
#define REG_CLASS_NAMES \
{			\
"NO_REGS",		\
"R12_REGS",		\
"R13_REGS",		\
"GEN_REGS",		\
"ALL_REGS"		\
}
#define REG_CLASS_CONTENTS \
{			   \
0x00000000,		   \
0x00001000,		   \
0x00002000,		   \
0x0000fff2,		   \
0x0001ffff		   \
}
#define GENERAL_REGS			GEN_REGS
#define BASE_REG_CLASS  		GEN_REGS
#define INDEX_REG_CLASS			GEN_REGS
#define N_REG_CLASSES			(int) LIM_REG_CLASSES
#define PC_REGNUM 		        0
#define STACK_POINTER_REGNUM 	        1
#define CC_REGNUM                       2
#define FRAME_POINTER_REGNUM 		4 
#define ARG_POINTER_REGNUM 		16
#define STATIC_CHAIN_REGNUM 		5 
#define FIRST_PSEUDO_REGISTER 		17
#define REGNO_REG_CLASS(REGNO)          ((REGNO) < 17 \
? GEN_REGS : NO_REGS)
#define TRAMPOLINE_SIZE			4 
#define TRAMPOLINE_ALIGNMENT		16 
#define ELIMINABLE_REGS					\
{{ ARG_POINTER_REGNUM,   STACK_POINTER_REGNUM },	\
{ ARG_POINTER_REGNUM,   FRAME_POINTER_REGNUM },	\
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM }}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)	\
(OFFSET) = msp430_initial_elimination_offset ((FROM), (TO))
#define FUNCTION_ARG_REGNO_P(N)	  	((N) >= 8 && (N) < ARG_POINTER_REGNUM)
#define DEFAULT_PCC_STRUCT_RETURN	0
#define FIXED_REGISTERS					\
{							\
1,0,1,1, 0,0,0,0,					\
0,0,0,0, 0,0,0,0,					\
1,							\
}
#define CALL_USED_REGISTERS				\
{							\
1,0,1,1, 0,0,0,0,					\
0,0,0,1, 1,1,1,1,					\
1,						\
}
#define REG_ALLOC_ORDER					\
{ 12, 13, 14, 15, 10, 9, 8, 7, 6, 5, 4, 11, 0, 1, 2, 3, 16 }
#define REGNO_OK_FOR_BASE_P(regno)	1
#define REGNO_OK_FOR_INDEX_P(regno)	1

typedef struct
{
char reg_used[4];
#define CA_FIRST_REG 12
char can_split;
char start_reg;
char reg_count;
char mem_count;
char special_p;
} CUMULATIVE_ARGS;
#define INIT_CUMULATIVE_ARGS(CA, FNTYPE, LIBNAME, INDIRECT, N_NAMED_ARGS) \
msp430_init_cumulative_args (&CA, FNTYPE, LIBNAME, INDIRECT, N_NAMED_ARGS)

#define NO_PROFILE_COUNTERS     1
#define PROFILE_BEFORE_PROLOGUE 1
#define FUNCTION_PROFILER(FILE, LABELNO)	\
fprintf (FILE, "\tcall\t__mcount\n");

#define EH_RETURN_DATA_REGNO(N) \
(((N) < 3) ? ((N) + 12) : INVALID_REGNUM)
#define EH_RETURN_HANDLER_RTX \
gen_rtx_MEM(Pmode, gen_rtx_PLUS (Pmode, gen_rtx_REG(Pmode, SP_REGNO), gen_rtx_REG (Pmode, 15)))
#define EH_RETURN_STACKADJ_RTX gen_rtx_REG (Pmode, 15)
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE,GLOBAL) DW_EH_PE_udata4



#define TEXT_SECTION_ASM_OP ".text"
#define DATA_SECTION_ASM_OP ".data"
#define BSS_SECTION_ASM_OP   "\t.section .bss"
#define ASM_COMMENT_START	" ;"
#define ASM_APP_ON		""
#define ASM_APP_OFF 		""
#define LOCAL_LABEL_PREFIX	".L"
#undef  USER_LABEL_PREFIX
#define USER_LABEL_PREFIX	""
#define GLOBAL_ASM_OP 		"\t.global\t"
#define ASM_OUTPUT_LABELREF(FILE, SYM) msp430_output_labelref ((FILE), (SYM))
#define ASM_OUTPUT_ADDR_VEC_ELT(FILE, VALUE) \
fprintf (FILE, "\t.long .L%d\n", VALUE)
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL) \
fprintf (FILE, "\t.long .L%d - 1b\n", VALUE)
#define ASM_OUTPUT_ALIGN(STREAM, LOG)		\
do						\
{						\
if ((LOG) == 0)				\
break;					\
fprintf (STREAM, "\t.balign %d\n", 1 << (LOG));	\
}						\
while (0)
#define JUMP_TABLES_IN_TEXT_SECTION	1

#undef	DWARF2_ADDR_SIZE
#define	DWARF2_ADDR_SIZE			4
#define INCOMING_FRAME_SP_OFFSET		(TARGET_LARGE ? 4 : 2)
#undef  PREFERRED_DEBUGGING_TYPE
#define PREFERRED_DEBUGGING_TYPE DWARF2_DEBUG
#define DWARF2_ASM_LINE_DEBUG_INFO		1
#define HARD_REGNO_CALLER_SAVE_MODE(REGNO,NREGS,MODE) \
((TARGET_LARGE && ((NREGS) <= 2)) ? PSImode : choose_hard_reg_mode ((REGNO), (NREGS), false))
#define ACCUMULATE_OUTGOING_ARGS 1
#undef  ASM_DECLARE_FUNCTION_NAME
#define ASM_DECLARE_FUNCTION_NAME(FILE, NAME, DECL) \
msp430_start_function ((FILE), (NAME), (DECL))
#define TARGET_HAS_NO_HW_DIVIDE (! TARGET_HWMULT)
#undef  USE_SELECT_SECTION_FOR_FUNCTIONS
#define USE_SELECT_SECTION_FOR_FUNCTIONS 1
#define ASM_OUTPUT_ALIGNED_DECL_COMMON(FILE, DECL, NAME, SIZE, ALIGN)	\
msp430_output_aligned_decl_common ((FILE), (DECL), (NAME), (SIZE), (ALIGN))
