#ifndef GCC_M32C_H
#define GCC_M32C_H
#undef  STARTFILE_SPEC
#define STARTFILE_SPEC "crt0.o%s crtbegin.o%s"
#undef  ENDFILE_SPEC
#define ENDFILE_SPEC "crtend.o%s crtn.o%s"
#undef  LINK_SPEC
#define LINK_SPEC "%{h*} %{v:-V} \
%{static:-Bstatic} %{shared:-shared} %{symbolic:-Bsymbolic}"
#undef  ASM_SPEC
#define ASM_SPEC "\
%{mcpu=r8c:--m16c} \
%{mcpu=m16c:--m16c} \
%{mcpu=m32cm:--m32c} \
%{mcpu=m32c:--m32c} "
#undef  LIB_SPEC
#define LIB_SPEC "-( -lc %{msim:-lsim}%{!msim:-lnosys} -) \
%{msim:%{!T*: %{mcpu=m32cm:%Tsim24.ld}%{mcpu=m32c:%Tsim24.ld} \
%{!mcpu=m32cm:%{!mcpu=m32c:%Tsim16.ld}}}} \
%{!T*:%{!msim: %{mcpu=m16c:%Tm16c.ld} \
%{mcpu=m32cm:%Tm32cm.ld} \
%{mcpu=m32c:%Tm32c.ld} \
%{!mcpu=m16c:%{!mcpu=m32cm:%{!mcpu=m32c:%Tr8c.ld}}}}} \
"
#define TARGET_CPU_CPP_BUILTINS() \
{ \
builtin_assert ("cpu=m32c"); \
builtin_assert ("machine=m32c"); \
builtin_define ("__m32c__=1"); \
if (TARGET_R8C) \
builtin_define ("__r8c_cpu__=1"); \
if (TARGET_M16C) \
builtin_define ("__m16c_cpu__=1"); \
if (TARGET_M32CM) \
builtin_define ("__m32cm_cpu__=1"); \
if (TARGET_M32C) \
builtin_define ("__m32c_cpu__=1"); \
}
extern int ok_to_change_target_memregs;
#define TARGET_R8C	(target_cpu == 'r')
#define TARGET_M16C	(target_cpu == '6')
#define TARGET_M32CM	(target_cpu == 'm')
#define TARGET_M32C	(target_cpu == '3')
#define TARGET_A16	(TARGET_R8C || TARGET_M16C)
#define TARGET_A24	(TARGET_M32CM || TARGET_M32C)
typedef struct GTY (()) machine_function
{
rtx eh_stack_adjust;
int is_interrupt;
int is_leaf;
int intr_pushm;
char intr_pushmem[16];
int use_rts;
}
machine_function;
#define INIT_EXPANDERS m32c_init_expanders ()
#define BITS_BIG_ENDIAN 0
#define BYTES_BIG_ENDIAN 0
#define WORDS_BIG_ENDIAN 0
#define UNITS_PER_WORD 2
#define POINTER_SIZE (TARGET_A16 ? 16 : 32)
#define POINTERS_EXTEND_UNSIGNED 1
#ifndef LIBGCC2_UNITS_PER_WORD
#define LIBGCC2_UNITS_PER_WORD 4
#endif
#define PARM_BOUNDARY (TARGET_A16 ? 8 : 16)
#define STACK_BOUNDARY (TARGET_A16 ? 8 : 16)
#define FUNCTION_BOUNDARY 8
#define BIGGEST_ALIGNMENT 8
#undef  PCC_BITFIELD_TYPE_MATTERS
#define PCC_BITFIELD_TYPE_MATTERS 0
#define STRICT_ALIGNMENT 0
#define SLOW_BYTE_ACCESS 1
#define INT_TYPE_SIZE 16
#define SHORT_TYPE_SIZE 16
#define LONG_TYPE_SIZE 32
#define LONG_LONG_TYPE_SIZE 64
#define FLOAT_TYPE_SIZE 32
#define DOUBLE_TYPE_SIZE 64
#define LONG_DOUBLE_TYPE_SIZE 64
#define DEFAULT_SIGNED_CHAR 1
#undef PTRDIFF_TYPE
#define PTRDIFF_TYPE (TARGET_A16 ? "int" : "long int")
#undef UINTPTR_TYPE
#define UINTPTR_TYPE (TARGET_A16 ? "unsigned int" : "long unsigned int")
#undef  SIZE_TYPE
#define SIZE_TYPE "unsigned int"
#undef  WCHAR_TYPE
#define WCHAR_TYPE "long int"
#undef  WCHAR_TYPE_SIZE
#define WCHAR_TYPE_SIZE 32
#define FIRST_PSEUDO_REGISTER   20
#define FIXED_REGISTERS     { 0, 0, 0, 0, \
0, 0, 1, 0, \
1, 1, 0, 1, \
0, 0, 0, 0, 0, 0, 0, 0 }
#define CALL_USED_REGISTERS { 1, 1, 1, 1, \
1, 1, 1, 0, \
1, 1, 1, 1, \
1, 1, 1, 1, 1, 1, 1, 1 }
#ifndef PC_REGNO
#define PC_REGNO 9
#endif
#define PC_REGNUM PC_REGNO
#define REG_ALLOC_ORDER { \
0, 1, 2, 3, 4, 5,  \
12, 13, 14, 15, 16, 17, 18, 19, 	\
6, 7, 8, 9, 10, 11  }
#define AVOID_CCMODE_COPIES
#define REG_CLASS_CONTENTS \
{ { 0x00000000 }, \
{ 0x00000100 }, \
{ 0x00000080 }, \
{ 0x00000040 }, \
{ 0x000001c0 }, \
{ 0x00000001 }, \
{ 0x00000004 }, \
{ 0x00000002 }, \
{ 0x00000008 }, \
{ 0x00000003 }, \
{ 0x0000000c }, \
{ 0x00000005 }, \
{ 0x0000000a }, \
{ 0x0000000f }, \
{ 0x00000010 }, \
{ 0x00000020 }, \
{ 0x00000030 }, \
{ 0x000000f0 }, \
{ 0x000001f0 }, \
{ 0x00000033 },  \
{ 0x0000003f }, \
{ 0x0000007f }, \
{ 0x00000400 }, \
{ 0x000001ff }, \
{ 0x000ff000 }, \
{ 0x000ff003 }, \
{ 0x000ff005 }, \
{ 0x000ff00c }, \
{ 0x000ff00f }, \
{ 0x000ff03f }, \
{ 0x000ff0ff }, \
{ 0x000ff5ff }, \
}
#define QI_REGS HL_REGS
#define HI_REGS RA_REGS
#define SI_REGS R03_REGS
#define DI_REGS R03_REGS
enum reg_class
{
NO_REGS,
SP_REGS,
FB_REGS,
SB_REGS,
CR_REGS,
R0_REGS,
R1_REGS,
R2_REGS,
R3_REGS,
R02_REGS,
R13_REGS,
HL_REGS,
R23_REGS,
R03_REGS,
A0_REGS,
A1_REGS,
A_REGS,
AD_REGS,
PS_REGS,
R02A_REGS,
RA_REGS,
GENERAL_REGS,
FLG_REGS,
HC_REGS,
MEM_REGS,
R02_A_MEM_REGS,
A_HL_MEM_REGS,
R1_R3_A_MEM_REGS,
R03_MEM_REGS,
A_HI_MEM_REGS,
A_AD_CR_MEM_SI_REGS,
ALL_REGS,
LIM_REG_CLASSES
};
#define N_REG_CLASSES LIM_REG_CLASSES
#define REG_CLASS_NAMES {\
"NO_REGS", \
"SP_REGS", \
"FB_REGS", \
"SB_REGS", \
"CR_REGS", \
"R0_REGS", \
"R1_REGS", \
"R2_REGS", \
"R3_REGS", \
"R02_REGS", \
"R13_REGS", \
"HL_REGS", \
"R23_REGS", \
"R03_REGS", \
"A0_REGS", \
"A1_REGS", \
"A_REGS", \
"AD_REGS", \
"PS_REGS", \
"R02A_REGS", \
"RA_REGS", \
"GENERAL_REGS", \
"FLG_REGS", \
"HC_REGS", \
"MEM_REGS", \
"R02_A_MEM_REGS", \
"A_HL_MEM_REGS", \
"R1_R3_A_MEM_REGS", \
"R03_MEM_REGS", \
"A_HI_MEM_REGS", \
"A_AD_CR_MEM_SI_REGS", \
"ALL_REGS", \
}
#define REGNO_REG_CLASS(R) m32c_regno_reg_class (R)
#define BASE_REG_CLASS A_REGS
#define INDEX_REG_CLASS NO_REGS
#define REGNO_OK_FOR_BASE_P(NUM) m32c_regno_ok_for_base_p (NUM)
#define REGNO_OK_FOR_INDEX_P(NUM) 0
#define LIMIT_RELOAD_CLASS(MODE,CLASS) \
(enum reg_class) m32c_limit_reload_class (MODE, CLASS)
#define SECONDARY_RELOAD_CLASS(CLASS,MODE,X) \
(enum reg_class) m32c_secondary_reload_class (CLASS, MODE, X)
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P hook_bool_mode_true
#define STACK_GROWS_DOWNWARD 1
#define STACK_PUSH_CODE PRE_DEC
#define FRAME_GROWS_DOWNWARD 1
#define FIRST_PARM_OFFSET(F) 0
#define RETURN_ADDR_RTX(COUNT,FA) m32c_return_addr_rtx (COUNT)
#define INCOMING_RETURN_ADDR_RTX m32c_incoming_return_addr_rtx()
#define INCOMING_FRAME_SP_OFFSET (TARGET_A24 ? 4 : 3)
#define EH_RETURN_DATA_REGNO(N) m32c_eh_return_data_regno (N)
#define EH_RETURN_STACKADJ_RTX m32c_eh_return_stackadj_rtx ()
#ifndef FP_REGNO
#define FP_REGNO 7
#endif
#ifndef SP_REGNO
#define SP_REGNO 8
#endif
#define AP_REGNO 11
#define STACK_POINTER_REGNUM	SP_REGNO
#define FRAME_POINTER_REGNUM	FP_REGNO
#define ARG_POINTER_REGNUM	AP_REGNO
#define STATIC_CHAIN_REGNUM A0_REGNO
#define DWARF_FRAME_REGISTERS 20
#define DWARF_FRAME_REGNUM(N) m32c_dwarf_frame_regnum (N)
#define DBX_REGISTER_NUMBER(N) m32c_dwarf_frame_regnum (N)
#undef ASM_PREFERRED_EH_DATA_FORMAT
#define ASM_PREFERRED_EH_DATA_FORMAT(CODE,GLOBAL) \
(TARGET_A16 ? DW_EH_PE_udata2 : DW_EH_PE_udata4)
#define ELIMINABLE_REGS \
{{AP_REGNO, SP_REGNO}, \
{AP_REGNO, FB_REGNO}, \
{FB_REGNO, SP_REGNO}}
#define INITIAL_ELIMINATION_OFFSET(FROM,TO,VAR) \
(VAR) = m32c_initial_elimination_offset(FROM,TO)
#define PUSH_ARGS 1
#define PUSH_ROUNDING(N) m32c_push_rounding (N)
#define CALL_POPS_ARGS(C) 0
typedef struct m32c_cumulative_args
{
int force_mem;
int parm_num;
} m32c_cumulative_args;
#define CUMULATIVE_ARGS m32c_cumulative_args
#define INIT_CUMULATIVE_ARGS(CA,FNTYPE,LIBNAME,FNDECL,N_NAMED_ARGS) \
m32c_init_cumulative_args (&(CA),FNTYPE,LIBNAME,FNDECL,N_NAMED_ARGS)
#define FUNCTION_ARG_REGNO_P(r) m32c_function_arg_regno_p (r)
#define DEFAULT_PCC_STRUCT_RETURN 1
#define EXIT_IGNORE_STACK 0
#define EPILOGUE_USES(REGNO) m32c_epilogue_uses(REGNO)
#define EH_USES(REGNO) 0	
#define FUNCTION_PROFILER(FILE,LABELNO)
#define TRAMPOLINE_SIZE m32c_trampoline_size ()
#define TRAMPOLINE_ALIGNMENT m32c_trampoline_alignment ()
#define HAVE_PRE_DECREMENT 1
#define HAVE_POST_INCREMENT 1
#define MAX_REGS_PER_ADDRESS 1
#ifdef REG_OK_STRICT
#define REG_OK_STRICT_V 1
#else
#define REG_OK_STRICT_V 0
#endif
#define REG_OK_FOR_BASE_P(X) m32c_reg_ok_for_base_p (X, REG_OK_STRICT_V)
#define REG_OK_FOR_INDEX_P(X) 0
#define LEGITIMIZE_RELOAD_ADDRESS(X,MODE,OPNUM,TYPE,IND_LEVELS,WIN) \
if (m32c_legitimize_reload_address(&(X),MODE,OPNUM,TYPE,IND_LEVELS)) \
goto WIN;
#define ADDR_SPACE_FAR	1
#define REVERSIBLE_CC_MODE(MODE) 1
#define TEXT_SECTION_ASM_OP ".text"
#define DATA_SECTION_ASM_OP ".data"
#define BSS_SECTION_ASM_OP ".bss"
#define CTOR_LIST_BEGIN
#define CTOR_LIST_END
#define DTOR_LIST_BEGIN
#define DTOR_LIST_END
#define CTORS_SECTION_ASM_OP "\t.section\t.init_array,\"aw\",%init_array"
#define DTORS_SECTION_ASM_OP "\t.section\t.fini_array,\"aw\",%fini_array"
#define INIT_ARRAY_SECTION_ASM_OP "\t.section\t.init_array,\"aw\",%init_array"
#define FINI_ARRAY_SECTION_ASM_OP "\t.section\t.fini_array,\"aw\",%fini_array"
#define ASM_COMMENT_START ";"
#define ASM_APP_ON ""
#define ASM_APP_OFF ""
#define GLOBAL_ASM_OP "\t.global\t"
#define REGISTER_NAMES {	\
"r0", "r2", "r1", "r3", \
"a0", "a1", "sb", "fb", "sp", \
"pc", "flg", "argp", \
"mem0",  "mem2",  "mem4",  "mem6",  "mem8",  "mem10",  "mem12",  "mem14", \
}
#define ADDITIONAL_REGISTER_NAMES { \
{"r0l", 0}, \
{"r1l", 2}, \
{"r0r2", 0}, \
{"r1r3", 2}, \
{"a0a1", 4}, \
{"r0r2r1r3", 0} }
#undef USER_LABEL_PREFIX
#define USER_LABEL_PREFIX "_"
#define ASM_OUTPUT_REG_PUSH(S,R) m32c_output_reg_push (S, R)
#define ASM_OUTPUT_REG_POP(S,R) m32c_output_reg_pop (S, R)
#define ASM_OUTPUT_ALIGNED_DECL_COMMON(STREAM, DECL, NAME, SIZE, ALIGNMENT) \
m32c_output_aligned_common (STREAM, DECL, NAME, SIZE, ALIGNMENT, 1)
#define ASM_OUTPUT_ALIGNED_DECL_LOCAL(STREAM, DECL, NAME, SIZE, ALIGNMENT) \
m32c_output_aligned_common (STREAM, DECL, NAME, SIZE, ALIGNMENT, 0)
#define ASM_OUTPUT_ADDR_VEC_ELT(S,V) \
fprintf (S, "\t.word L%d\n", V)
#define DWARF_CIE_DATA_ALIGNMENT -1
#define ASM_OUTPUT_ALIGN(STREAM,POWER) \
fprintf (STREAM, "\t.p2align\t%d\n", POWER);
#define DWARF2_ADDR_SIZE	4
#define HAS_LONG_COND_BRANCH false
#define HAS_LONG_UNCOND_BRANCH true
#define CASE_VECTOR_MODE SImode
#define LOAD_EXTEND_OP(MEM) ZERO_EXTEND
#define MOVE_MAX 4
#define STORE_FLAG_VALUE 1
#define Pmode (TARGET_A16 ? HImode : PSImode)
#define FUNCTION_MODE QImode
#define REGISTER_TARGET_PRAGMAS() m32c_register_pragmas()
#endif
