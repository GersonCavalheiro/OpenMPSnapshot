#ifndef GCC_H8300_H
#define GCC_H8300_H
#if 0 
enum attr_cpu { CPU_H8300, CPU_H8300H };
#endif
extern int cpu_type;
extern const char *h8_push_op, *h8_pop_op, *h8_mov_op;
extern const char * const *h8_reg_names;
#define TARGET_CPU_CPP_BUILTINS()			\
do							\
{							\
if (TARGET_H8300SX)				\
{						\
builtin_define ("__H8300SX__");		\
if (TARGET_NORMAL_MODE)			\
{						\
builtin_define ("__NORMAL_MODE__");	\
}						\
}						\
else if (TARGET_H8300S)				\
{						\
builtin_define ("__H8300S__");		\
builtin_assert ("cpu=h8300s");		\
builtin_assert ("machine=h8300s");		\
if (TARGET_NORMAL_MODE)			\
{						\
builtin_define ("__NORMAL_MODE__");	\
}						\
}						\
else if (TARGET_H8300H)				\
{						\
builtin_define ("__H8300H__");		\
builtin_assert ("cpu=h8300h");		\
builtin_assert ("machine=h8300h");		\
if (TARGET_NORMAL_MODE)			\
{						\
builtin_define ("__NORMAL_MODE__");	\
}						\
}						\
else						\
{						\
builtin_define ("__H8300__");			\
builtin_assert ("cpu=h8300");			\
builtin_assert ("machine=h8300");		\
}						\
}							\
while (0)
#define LINK_SPEC "%{mh:%{mn:-m h8300hn}} %{mh:%{!mn:-m h8300h}} %{ms:%{mn:-m h8300sn}} %{ms:%{!mn:-m h8300s}}"
#define LIB_SPEC "%{mrelax:-relax} %{g:-lg} %{!p:%{!pg:-lc}}%{p:-lc_p}%{pg:-lc_p}"
#define TARGET_H8300	(! TARGET_H8300H && ! TARGET_H8300S)
#define TARGET_H8300S	(TARGET_H8300S_1 || TARGET_H8300SX)
#define TARGET_H8300SXMUL TARGET_H8300SX
#ifdef IN_LIBGCC2
#undef TARGET_H8300H
#undef TARGET_H8300S
#undef TARGET_NORMAL_MODE
#ifdef __H8300H__
#define TARGET_H8300H	1
#else
#define TARGET_H8300H	0
#endif
#ifdef __H8300S__
#define TARGET_H8300S	1
#else
#define TARGET_H8300S	0
#endif
#ifdef __NORMAL_MODE__
#define TARGET_NORMAL_MODE 1
#else
#define TARGET_NORMAL_MODE 0
#endif
#endif 
#ifndef TARGET_DEFAULT
#define TARGET_DEFAULT (MASK_QUICKCALL)
#endif
#define DWARF2_DEBUGGING_INFO        1
#define INCOMING_RETURN_ADDR_RTX   gen_rtx_MEM (Pmode, gen_rtx_REG (Pmode, STACK_POINTER_REGNUM))
#define INCOMING_FRAME_SP_OFFSET   (POINTER_SIZE / 8)
#define DWARF_CIE_DATA_ALIGNMENT	2
#define NO_FUNCTION_CSE 1

#define BITS_BIG_ENDIAN 0
#define BYTES_BIG_ENDIAN 1
#define WORDS_BIG_ENDIAN 1
#define MAX_BITS_PER_WORD	32
#define UNITS_PER_WORD		(TARGET_H8300H || TARGET_H8300S ? 4 : 2)
#define MIN_UNITS_PER_WORD	2
#define SHORT_TYPE_SIZE	16
#define INT_TYPE_SIZE		(TARGET_INT32 ? 32 : 16)
#define LONG_TYPE_SIZE		32
#define LONG_LONG_TYPE_SIZE	64
#define FLOAT_TYPE_SIZE	32
#define DOUBLE_TYPE_SIZE	32
#define LONG_DOUBLE_TYPE_SIZE	DOUBLE_TYPE_SIZE
#define MAX_FIXED_MODE_SIZE	32
#define PARM_BOUNDARY (TARGET_H8300H || TARGET_H8300S ? 32 : 16)
#define FUNCTION_BOUNDARY 16
#define EMPTY_FIELD_BOUNDARY 16
#define BIGGEST_ALIGNMENT \
(((TARGET_H8300H || TARGET_H8300S) && ! TARGET_ALIGN_300) ? 32 : 16)
#define STACK_BOUNDARY (TARGET_H8300 ? 16 : 32)
#define STRICT_ALIGNMENT 1

#define FIRST_PSEUDO_REGISTER 12
#define FIXED_REGISTERS				\
\
{ 0, 0, 0, 0, 0, 0, 0, 1,  0, 1,  1, 1 }
#define CALL_USED_REGISTERS			\
\
{ 1, 1, 1, 1, 0, 0, 0, 1,  1, 1,  1, 1 }
#define REG_ALLOC_ORDER				\
\
{ 2, 3, 0, 1, 4, 5, 6, 8,  7, 9, 10, 11 }
#define HARD_REGNO_RENAME_OK(OLD_REG, NEW_REG)		\
h8300_hard_regno_rename_ok (OLD_REG, NEW_REG)
#define STACK_POINTER_REGNUM SP_REG
#define HARD_FRAME_POINTER_REGNUM HFP_REG
#define FRAME_POINTER_REGNUM FP_REG
#define ARG_POINTER_REGNUM AP_REG
#define STATIC_CHAIN_REGNUM SC_REG
#define RETURN_ADDRESS_POINTER_REGNUM RAP_REG
#define RETURN_ADDR_RTX(COUNT, FRAME) h8300_return_addr_rtx ((COUNT), (FRAME))

enum reg_class {
NO_REGS, COUNTER_REGS, SOURCE_REGS, DESTINATION_REGS,
GENERAL_REGS, MAC_REGS, ALL_REGS, LIM_REG_CLASSES
};
#define N_REG_CLASSES ((int) LIM_REG_CLASSES)
#define REG_CLASS_NAMES \
{ "NO_REGS", "COUNTER_REGS", "SOURCE_REGS", "DESTINATION_REGS", \
"GENERAL_REGS", "MAC_REGS", "ALL_REGS", "LIM_REGS" }
#define REG_CLASS_CONTENTS			\
{      {0},			\
{0x010},			\
{0x020},			\
{0x040},			\
{0xeff},			\
{0x100},				\
{0xfff},			\
}
#define REGNO_REG_CLASS(REGNO)				\
((REGNO) == MAC_REG ? MAC_REGS			\
: (REGNO) == COUNTER_REG ? COUNTER_REGS		\
: (REGNO) == SOURCE_REG ? SOURCE_REGS		\
: (REGNO) == DESTINATION_REG ? DESTINATION_REGS	\
: GENERAL_REGS)
#define INDEX_REG_CLASS (TARGET_H8300SX ? GENERAL_REGS : NO_REGS)
#define BASE_REG_CLASS  GENERAL_REGS
#define STACK_GROWS_DOWNWARD 1
#define FRAME_GROWS_DOWNWARD 1
#define PUSH_ROUNDING(BYTES) h8300_push_rounding (BYTES)
#define FIRST_PARM_OFFSET(FNDECL) 0
#define ELIMINABLE_REGS						\
{{ ARG_POINTER_REGNUM, STACK_POINTER_REGNUM},			\
{ ARG_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},		\
{ RETURN_ADDRESS_POINTER_REGNUM, STACK_POINTER_REGNUM},	\
{ RETURN_ADDRESS_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},	\
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM},			\
{ FRAME_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM}}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)		\
((OFFSET) = h8300_initial_elimination_offset ((FROM), (TO)))
#define FUNCTION_ARG_REGNO_P(N) (TARGET_QUICKCALL ? N < 3 : 0)
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P hook_bool_mode_true

#define CUMULATIVE_ARGS struct cum_arg
struct cum_arg
{
int nbytes;
rtx libcall;
};
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, INDIRECT, N_NAMED_ARGS) \
((CUM).nbytes = 0, (CUM).libcall = LIBNAME)
#define FUNCTION_PROFILER(FILE, LABELNO)  \
fprintf (FILE, "\t%s\t#LP%d,%s\n\tjsr @mcount\n", \
h8_mov_op, (LABELNO), h8_reg_names[0]);
#define EXIT_IGNORE_STACK 0
#define TRAMPOLINE_SIZE ((Pmode == HImode) ? 8 : 12)

#define HAVE_POST_INCREMENT 1
#define HAVE_PRE_DECREMENT 1
#define HAVE_POST_DECREMENT TARGET_H8300SX
#define HAVE_PRE_INCREMENT TARGET_H8300SX
#define REGNO_OK_FOR_INDEX_P(regno) 0
#define REGNO_OK_FOR_BASE_P(regno)				\
(((regno) < FIRST_PSEUDO_REGISTER && regno != MAC_REG)	\
|| reg_renumber[regno] >= 0)

#define MAX_REGS_PER_ADDRESS 1
#define CONSTANT_ADDRESS_P(X)					\
(GET_CODE (X) == LABEL_REF || GET_CODE (X) == SYMBOL_REF	\
|| (GET_CODE (X) == CONST_INT				\
\
&& INTVAL (X) > (TARGET_H8300 ? -0x10000 : -0x1000000)	\
&& INTVAL (X) < (TARGET_H8300 ? 0x10000 : 0x1000000))	\
|| (GET_CODE (X) == HIGH || GET_CODE (X) == CONST))
#define REG_OK_FOR_INDEX_NONSTRICT_P(X) 0
#define REG_OK_FOR_BASE_NONSTRICT_P(X)				\
(REGNO (X) >= FIRST_PSEUDO_REGISTER || REGNO (X) != MAC_REG)
#define REG_OK_FOR_INDEX_STRICT_P(X) REGNO_OK_FOR_INDEX_P (REGNO (X))
#define REG_OK_FOR_BASE_STRICT_P(X)  REGNO_OK_FOR_BASE_P (REGNO (X))
#ifndef REG_OK_STRICT
#define REG_OK_FOR_INDEX_P(X) REG_OK_FOR_INDEX_NONSTRICT_P (X)
#define REG_OK_FOR_BASE_P(X)  REG_OK_FOR_BASE_NONSTRICT_P (X)
#else
#define REG_OK_FOR_INDEX_P(X) REG_OK_FOR_INDEX_STRICT_P (X)
#define REG_OK_FOR_BASE_P(X)  REG_OK_FOR_BASE_STRICT_P (X)
#endif

#define CASE_VECTOR_MODE Pmode
#define DEFAULT_SIGNED_CHAR 0
#define MOVE_MAX	(TARGET_H8300H || TARGET_H8300S ? 4 : 2)
#define MAX_MOVE_MAX	4
#define SLOW_BYTE_ACCESS TARGET_SLOWBYTE
#define Pmode								      \
((TARGET_H8300H || TARGET_H8300S) && !TARGET_NORMAL_MODE ? SImode : HImode)
#define SIZE_TYPE								\
(TARGET_H8300 || TARGET_NORMAL_MODE ? TARGET_INT32 ? "short unsigned int" : "unsigned int" : "long unsigned int")
#define PTRDIFF_TYPE						\
(TARGET_H8300 || TARGET_NORMAL_MODE ? TARGET_INT32 ? "short int" : "int" : "long int")
#define POINTER_SIZE							\
((TARGET_H8300H || TARGET_H8300S) && !TARGET_NORMAL_MODE ? 32 : 16)
#define WCHAR_TYPE "short unsigned int"
#define WCHAR_TYPE_SIZE 16
#define FUNCTION_MODE QImode
#define DELAY_SLOT_LENGTH(JUMP) \
(NEXT_INSN (PREV_INSN (JUMP)) == JUMP ? 0 : 2)
#define BRANCH_COST(speed_p, predictable_p) 0
#define NOTICE_UPDATE_CC(EXP, INSN) notice_update_cc (EXP, INSN)
#define CC_OVERFLOW_UNUSABLE 01000
#define CC_NO_CARRY CC_NO_OVERFLOW

#define ASM_APP_ON "; #APP\n"
#define ASM_APP_OFF "; #NO_APP\n"
#define FILE_ASM_OP "\t.file\n"
#define ASM_WORD_OP							\
(TARGET_H8300 || TARGET_NORMAL_MODE ? "\t.word\t" : "\t.long\t")
#define TEXT_SECTION_ASM_OP "\t.section .text"
#define DATA_SECTION_ASM_OP "\t.section .data"
#define BSS_SECTION_ASM_OP "\t.section .bss"
#undef DO_GLOBAL_CTORS_BODY
#define DO_GLOBAL_CTORS_BODY			\
{						\
extern func_ptr __ctors[];			\
extern func_ptr __ctors_end[];		\
func_ptr *p;					\
for (p = __ctors_end; p > __ctors; )		\
{						\
(*--p)();					\
}						\
}
#undef DO_GLOBAL_DTORS_BODY
#define DO_GLOBAL_DTORS_BODY			\
{						\
extern func_ptr __dtors[];			\
extern func_ptr __dtors_end[];		\
func_ptr *p;					\
for (p = __dtors; p < __dtors_end; p++)	\
{						\
(*p)();					\
}						\
}
#define REGISTER_NAMES \
{ "r0", "r1", "r2", "r3", "r4", "r5", "r6", "sp", "mac", "ap", "rap", "fp" }
#define ADDITIONAL_REGISTER_NAMES \
{ {"er0", 0}, {"er1", 1}, {"er2", 2}, {"er3", 3}, {"er4", 4}, \
{"er5", 5}, {"er6", 6}, {"er7", 7}, {"r7", 7} }
#define GLOBAL_ASM_OP "\t.global "
#define ASM_DECLARE_FUNCTION_NAME(FILE, NAME, DECL) \
ASM_OUTPUT_LABEL (FILE, NAME)
#define USER_LABEL_PREFIX "_"
#define ASM_GENERATE_INTERNAL_LABEL(LABEL, PREFIX, NUM)	\
sprintf (LABEL, "*.%s%lu", PREFIX, (unsigned long)(NUM))
#define ASM_OUTPUT_REG_PUSH(FILE, REGNO) \
fprintf (FILE, "\t%s\t%s\n", h8_push_op, h8_reg_names[REGNO])
#define ASM_OUTPUT_REG_POP(FILE, REGNO) \
fprintf (FILE, "\t%s\t%s\n", h8_pop_op, h8_reg_names[REGNO])
#define ASM_OUTPUT_ADDR_VEC_ELT(FILE, VALUE) \
fprintf (FILE, "%s.L%d\n", ASM_WORD_OP, VALUE)
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL) \
fprintf (FILE, "%s.L%d-.L%d\n", ASM_WORD_OP, VALUE, REL)
#define ASM_OUTPUT_ALIGN(FILE, LOG)		\
if ((LOG) != 0)				\
fprintf (FILE, "\t.align %d\n", (LOG))
#define ASM_OUTPUT_SKIP(FILE, SIZE) \
fprintf (FILE, "\t.space %d\n", (int)(SIZE))
#define ASM_OUTPUT_COMMON(FILE, NAME, SIZE, ROUNDED)	\
( fputs ("\t.comm ", (FILE)),				\
assemble_name ((FILE), (NAME)),			\
fprintf ((FILE), ",%lu\n", (unsigned long)(SIZE)))
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN) \
asm_output_aligned_bss (FILE, DECL, NAME, SIZE, ALIGN)
#define ASM_OUTPUT_LOCAL(FILE, NAME, SIZE, ROUNDED)	\
( fputs ("\t.lcomm ", (FILE)),				\
assemble_name ((FILE), (NAME)),			\
fprintf ((FILE), ",%d\n", (int)(SIZE)))
#define ASM_PN_FORMAT "%s___%lu"
#define REGISTER_TARGET_PRAGMAS()				\
do								\
{								\
c_register_pragma (0, "saveall", h8300_pr_saveall);	\
c_register_pragma (0, "interrupt", h8300_pr_interrupt);	\
}								\
while (0)
#define FINAL_PRESCAN_INSN(insn, operand, nop)	\
final_prescan_insn (insn, operand, nop)
extern int h8300_move_ratio;
#define MOVE_RATIO(speed) h8300_move_ratio
#define SYMBOL_FLAG_FUNCVEC_FUNCTION	(SYMBOL_FLAG_MACH_DEP << 0)
#define SYMBOL_FLAG_EIGHTBIT_DATA	(SYMBOL_FLAG_MACH_DEP << 1)
#define SYMBOL_FLAG_TINY_DATA		(SYMBOL_FLAG_MACH_DEP << 2)
#endif 
