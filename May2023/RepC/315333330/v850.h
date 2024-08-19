#ifndef GCC_V850_H
#define GCC_V850_H
extern GTY(()) rtx v850_compare_op0;
extern GTY(()) rtx v850_compare_op1;
#undef LIB_SPEC
#define LIB_SPEC "%{!shared:%{!symbolic:--start-group -lc -lgcc --end-group}}"
#undef ENDFILE_SPEC
#undef LINK_SPEC
#undef STARTFILE_SPEC
#undef ASM_SPEC
#define TARGET_CPU_generic 	1
#define TARGET_CPU_v850e   	2
#define TARGET_CPU_v850e1	3
#define TARGET_CPU_v850e2	4
#define TARGET_CPU_v850e2v3	5
#define TARGET_CPU_v850e3v5	6
#ifndef TARGET_CPU_DEFAULT
#define TARGET_CPU_DEFAULT	TARGET_CPU_generic
#endif
#define MASK_DEFAULT            MASK_V850
#define SUBTARGET_ASM_SPEC 	"%{!mv*:-mv850}"
#define SUBTARGET_CPP_SPEC 	"%{!mv*:-D__v850__}"
#if TARGET_CPU_DEFAULT == TARGET_CPU_v850e
#undef  MASK_DEFAULT
#define MASK_DEFAULT            MASK_V850E
#undef  SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC 	"%{!mv*:-mv850e}"
#undef  SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC 	"%{!mv*:-D__v850e__}"
#endif
#if TARGET_CPU_DEFAULT == TARGET_CPU_v850e1
#undef  MASK_DEFAULT
#define MASK_DEFAULT            MASK_V850E          
#undef  SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC	"%{!mv*:-mv850e1}"
#undef  SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC	"%{!mv*:-D__v850e1__} %{mv850e1:-D__v850e1__}"
#endif
#if TARGET_CPU_DEFAULT == TARGET_CPU_v850e2
#undef  MASK_DEFAULT
#define MASK_DEFAULT            MASK_V850E2	
#undef  SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC 	"%{!mv*:-mv850e2}"
#undef  SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC 	"%{!mv*:-D__v850e2__} %{mv850e2:-D__v850e2__}"
#endif
#if TARGET_CPU_DEFAULT == TARGET_CPU_v850e2v3
#undef  MASK_DEFAULT
#define MASK_DEFAULT            MASK_V850E2V3
#undef  SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC	"%{!mv*:-mv850e2v3}"
#undef  SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC	"%{!mv*:-D__v850e2v3__} %{mv850e2v3:-D__v850e2v3__}"
#endif
#if TARGET_CPU_DEFAULT == TARGET_CPU_v850e3v5
#undef  MASK_DEFAULT
#define MASK_DEFAULT            MASK_V850E3V5
#undef  SUBTARGET_ASM_SPEC
#define SUBTARGET_ASM_SPEC	"%{!mv*:-mv850e3v5}"
#undef  SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC	"%{!mv*:-D__v850e3v5__} %{mv850e3v5:-D__v850e3v5__}"
#undef  TARGET_VERSION
#define TARGET_VERSION		fprintf (stderr, " (Renesas V850E3V5)");
#endif
#define TARGET_V850E3V5_UP ((TARGET_V850E3V5))     
#define TARGET_V850E2V3_UP ((TARGET_V850E2V3) || TARGET_V850E3V5_UP)
#define TARGET_V850E2_UP   ((TARGET_V850E2)   || TARGET_V850E2V3_UP)
#define TARGET_V850E_UP    ((TARGET_V850E)    || TARGET_V850E2_UP)
#define TARGET_ALL         ((TARGET_V850)     || TARGET_V850E_UP)
#define ASM_SPEC "%{m850es:-mv850e1}%{!mv850es:%{mv*:-mv%*}} \
%{mrelax:-mrelax} \
%{m8byte-align:-m8byte-align} \
%{msoft-float:-msoft-float} \
%{mhard-float:-mhard-float} \
%{mgcc-abi:-mgcc-abi}"
#define LINK_SPEC "%{mgcc-abi:-m v850}"
#define CPP_SPEC "\
%{mv850e3v5:-D__v850e3v5__} \
%{mv850e2v3:-D__v850e2v3__} \
%{mv850e2:-D__v850e2__} \
%{mv850es:-D__v850e1__} \
%{mv850e1:-D__v850e1__} \
%{mv850e:-D__v850e__} \
%{mv850:-D__v850__} \
%(subtarget_cpp_spec) \
%{mep:-D__EP__}"
#define EXTRA_SPECS \
{ "subtarget_asm_spec", SUBTARGET_ASM_SPEC }, \
{ "subtarget_cpp_spec", SUBTARGET_CPP_SPEC } 
#define TARGET_USE_FPU  (TARGET_V850E2V3_UP && ! TARGET_SOFT_FLOAT)
#define TARGET_CPU_CPP_BUILTINS()		\
do						\
{						\
builtin_define( "__v851__" );		\
builtin_define( "__v850" );		\
builtin_define( "__v850__" );		\
builtin_assert( "machine=v850" );		\
builtin_assert( "cpu=v850" );		\
if (TARGET_EP)				\
builtin_define ("__EP__");		\
if (TARGET_GCC_ABI)			\
builtin_define ("__V850_GCC_ABI__");	\
else					\
builtin_define ("__V850_RH850_ABI__");	\
if (! TARGET_DISABLE_CALLT)		\
builtin_define ("__V850_CALLT__");	\
if (TARGET_8BYTE_ALIGN)			\
builtin_define ("__V850_8BYTE_ALIGN__");\
builtin_define (TARGET_USE_FPU ?		\
"__FPU_OK__" : "__NO_FPU__");\
}						\
while(0)
#define MASK_CPU (MASK_V850 | MASK_V850E | MASK_V850E1 | MASK_V850E2 | MASK_V850E2V3 | MASK_V850E3V5)

#define BITS_BIG_ENDIAN 0
#define BYTES_BIG_ENDIAN 0
#define WORDS_BIG_ENDIAN 0
#define UNITS_PER_WORD		4
#define PROMOTE_MODE(MODE,UNSIGNEDP,TYPE)  \
if (GET_MODE_CLASS (MODE) == MODE_INT \
&& GET_MODE_SIZE (MODE) < 4)      \
{ (MODE) = SImode; }
#define PARM_BOUNDARY		32
#define STACK_BOUNDARY 		BIGGEST_ALIGNMENT
#define FUNCTION_BOUNDARY 	(((! TARGET_GCC_ABI) || optimize_size) ? 16 : 32)
#define BIGGEST_ALIGNMENT	(TARGET_8BYTE_ALIGN ? 64 : 32)
#define EMPTY_FIELD_BOUNDARY 32
#define BIGGEST_FIELD_ALIGNMENT BIGGEST_ALIGNMENT
#define STRICT_ALIGNMENT  (!TARGET_NO_STRICT_ALIGN)
#define DEFAULT_SIGNED_CHAR 1
#undef  SIZE_TYPE
#define SIZE_TYPE "unsigned int"
#undef  PTRDIFF_TYPE
#define PTRDIFF_TYPE "int"
#undef  WCHAR_TYPE
#define WCHAR_TYPE "long int"
#undef  WCHAR_TYPE_SIZE
#define WCHAR_TYPE_SIZE BITS_PER_WORD

#define FIRST_PSEUDO_REGISTER 36
#define FIXED_REGISTERS \
{ 1, 1, 1, 1, 1, 1, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 1, 0, \
1, 1,	\
1, 1}
#define CALL_USED_REGISTERS \
{ 1, 1, 1, 1, 1, 1, 1, 1, \
1, 1, 1, 1, 1, 1, 1, 1, \
1, 1, 1, 1, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 1, 1, \
1, 1,	\
1, 1}
#define REG_ALLOC_ORDER							\
{									\
10, 11,						\
12, 13, 14, 15, 16, 17, 18, 19,			\
6,  7,  8,  9, 31,				\
29, 28, 27, 26, 25, 24, 23, 22,			\
21, 20,  2,								\
0,  1,  3,  4,  5, 30, 32, 33,                 \
34, 35								\
}

enum reg_class
{
NO_REGS, EVEN_REGS, GENERAL_REGS, ALL_REGS, LIM_REG_CLASSES
};
#define N_REG_CLASSES (int) LIM_REG_CLASSES
#define REG_CLASS_NAMES \
{ "NO_REGS", "EVEN_REGS", "GENERAL_REGS", "ALL_REGS", "LIM_REGS" }
#define REG_CLASS_CONTENTS                     \
{                                              \
{ 0x00000000,0x0 },        \
{ 0x55555554,0x0 },           \
{ 0xfffffffe,0x0 },        \
{ 0xffffffff,0x0 },       \
}
#define REGNO_REG_CLASS(REGNO)  ((REGNO == CC_REGNUM || REGNO == FCC_REGNUM) ? NO_REGS : GENERAL_REGS)
#define INDEX_REG_CLASS NO_REGS
#define BASE_REG_CLASS  GENERAL_REGS
#define REGNO_OK_FOR_BASE_P(regno)             \
(((regno) < FIRST_PSEUDO_REGISTER            \
&& (regno) != CC_REGNUM                    \
&& (regno) != FCC_REGNUM)                  \
|| reg_renumber[regno] >= 0)
#define REGNO_OK_FOR_INDEX_P(regno) 0
#define CONST_OK_FOR_I(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_I)
#define CONST_OK_FOR_J(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_J)
#define CONST_OK_FOR_K(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_K)
#define CONST_OK_FOR_L(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_L)
#define CONST_OK_FOR_M(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_M)
#define CONST_OK_FOR_N(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_N)
#define CONST_OK_FOR_O(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_O)
#define CONST_OK_FOR_W(VALUE) \
insn_const_int_ok_for_constraint (VALUE, CONSTRAINT_W)

#define STACK_GROWS_DOWNWARD 1
#define FRAME_GROWS_DOWNWARD 1
#define FIRST_PARM_OFFSET(FNDECL) 0
#define STACK_POINTER_REGNUM SP_REGNUM
#define FRAME_POINTER_REGNUM 34
#define LINK_POINTER_REGNUM LP_REGNUM
#undef  HARD_FRAME_POINTER_REGNUM 
#define HARD_FRAME_POINTER_REGNUM 29
#define ARG_POINTER_REGNUM 35
#define STATIC_CHAIN_REGNUM 20
#define ELIMINABLE_REGS							\
{{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM },			\
{ FRAME_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM },			\
{ ARG_POINTER_REGNUM,	 STACK_POINTER_REGNUM },			\
{ ARG_POINTER_REGNUM,   HARD_FRAME_POINTER_REGNUM }}			\
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)			\
{									\
if ((FROM) == FRAME_POINTER_REGNUM)					\
(OFFSET) = get_frame_size () + crtl->outgoing_args_size;	\
else if ((FROM) == ARG_POINTER_REGNUM)				\
(OFFSET) = compute_frame_size (get_frame_size (), (long *)0);	\
else									\
gcc_unreachable ();							\
}
#define ACCUMULATE_OUTGOING_ARGS 1
#define RETURN_ADDR_RTX(COUNT, FP) v850_return_addr (COUNT)

#define CUMULATIVE_ARGS struct cum_arg
struct cum_arg { int nbytes; };
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, INDIRECT, N_NAMED_ARGS) \
do { (CUM).nbytes = 0; } while (0)
#define REG_PARM_STACK_SPACE(DECL) 0
#define FUNCTION_ARG_REGNO_P(N) (N >= 6 && N <= 9)
#define DEFAULT_PCC_STRUCT_RETURN 0
#define EXIT_IGNORE_STACK 1
#define EPILOGUE_USES(REGNO) \
(reload_completed && (REGNO) == LINK_POINTER_REGNUM)
#define FUNCTION_PROFILER(FILE, LABELNO) ;
#define TRAMPOLINE_SIZE 24

#define CONSTANT_ADDRESS_P(X) constraint_satisfied_p (X, CONSTRAINT_K)
#define MAX_REGS_PER_ADDRESS 1

#define SELECT_CC_MODE(OP, X, Y)       v850_select_cc_mode (OP, X, Y)
#define CC_OVERFLOW_UNUSABLE 0x200
#define CC_NO_CARRY CC_NO_OVERFLOW
#define NOTICE_UPDATE_CC(EXP, INSN) notice_update_cc(EXP, INSN)
#define SLOW_BYTE_ACCESS 1
#define MOVE_RATIO(speed) 6
#define NO_FUNCTION_CSE 1
typedef enum 
{
DATA_AREA_NORMAL,
DATA_AREA_SDA,
DATA_AREA_TDA,
DATA_AREA_ZDA
} v850_data_area;
#define TEXT_SECTION_ASM_OP  "\t.section .text"
#define DATA_SECTION_ASM_OP  "\t.section .data"
#define BSS_SECTION_ASM_OP   "\t.section .bss"
#define SDATA_SECTION_ASM_OP "\t.section .sdata,\"aw\""
#define SBSS_SECTION_ASM_OP  "\t.section .sbss,\"aw\""
#define SCOMMON_ASM_OP 	       "\t.scomm\t"
#define ZCOMMON_ASM_OP 	       "\t.zcomm\t"
#define TCOMMON_ASM_OP 	       "\t.tcomm\t"
#define ASM_COMMENT_START "#"
#define ASM_APP_ON "#APP\n"
#define ASM_APP_OFF "#NO_APP\n"
#undef  USER_LABEL_PREFIX
#define USER_LABEL_PREFIX "_"
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN) \
asm_output_aligned_bss ((FILE), (DECL), (NAME), (SIZE), (ALIGN))
#undef  ASM_OUTPUT_ALIGNED_BSS 
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN) \
v850_output_aligned_bss (FILE, DECL, NAME, SIZE, ALIGN)
#undef  ASM_OUTPUT_ALIGNED_COMMON
#undef  ASM_OUTPUT_COMMON
#define ASM_OUTPUT_ALIGNED_DECL_COMMON(FILE, DECL, NAME, SIZE, ALIGN) \
v850_output_common (FILE, DECL, NAME, SIZE, ALIGN)
#undef  ASM_OUTPUT_ALIGNED_LOCAL
#undef  ASM_OUTPUT_LOCAL
#define ASM_OUTPUT_ALIGNED_DECL_LOCAL(FILE, DECL, NAME, SIZE, ALIGN) \
v850_output_local (FILE, DECL, NAME, SIZE, ALIGN)
#define GLOBAL_ASM_OP "\t.global "
#define ASM_PN_FORMAT "%s___%lu"
#define ASM_OUTPUT_DEF(FILE,NAME1,NAME2) \
do { assemble_name(FILE, NAME1); 	 \
fputs(" = ", FILE);		 \
assemble_name(FILE, NAME2);	 \
fputc('\n', FILE); } while (0)
#define REGISTER_NAMES                                         \
{  "r0",  "r1",  "r2",  "sp",  "gp",  "r5",  "r6" , "r7",      \
"r8",  "r9", "r10", "r11", "r12", "r13", "r14", "r15",      \
"r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",      \
"r24", "r25", "r26", "r27", "r28", "r29",  "ep", "r31",      \
"psw", "fcc",      \
".fp", ".ap"}
#define ADDITIONAL_REGISTER_NAMES              \
{ { "zero",    ZERO_REGNUM },                  \
{ "hp",      2 },                            \
{ "r3",      3 },                            \
{ "r4",      4 },                            \
{ "tp",      5 },                            \
{ "fp",      29 },                           \
{ "r30",     30 },                           \
{ "lp",      LP_REGNUM} }
#define ASM_OUTPUT_ADDR_VEC_ELT(FILE, VALUE) \
fprintf (FILE, "\t%s .L%d\n",					\
(TARGET_BIG_SWITCH ? ".long" : ".short"), VALUE)
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL) 		\
fprintf (FILE, "\t%s %s.L%d-.L%d%s\n",				\
(TARGET_BIG_SWITCH ? ".long" : ".short"),			\
(0 && ! TARGET_BIG_SWITCH && (TARGET_V850E_UP) ? "(" : ""),             \
VALUE, REL,							\
(0 && ! TARGET_BIG_SWITCH && (TARGET_V850E_UP) ? ")>>1" : ""))
#define ASM_OUTPUT_ALIGN(FILE, LOG)	\
if ((LOG) != 0)			\
fprintf (FILE, "\t.align %d\n", (LOG))
#define DEFAULT_GDB_EXTENSIONS 1
#undef  PREFERRED_DEBUGGING_TYPE
#define PREFERRED_DEBUGGING_TYPE   DWARF2_DEBUG
#define DWARF2_FRAME_INFO          1
#define DWARF2_UNWIND_INFO         0
#define INCOMING_RETURN_ADDR_RTX   gen_rtx_REG (Pmode, LINK_POINTER_REGNUM)
#define DWARF_FRAME_RETURN_COLUMN  DWARF_FRAME_REGNUM (LINK_POINTER_REGNUM)
#ifndef ASM_GENERATE_INTERNAL_LABEL
#define ASM_GENERATE_INTERNAL_LABEL(STRING, PREFIX, NUM)  \
sprintf (STRING, "*.%s%u", PREFIX, (unsigned int)(NUM))
#endif
#define CASE_VECTOR_MODE (TARGET_BIG_SWITCH ? SImode : HImode)
#define CASE_VECTOR_PC_RELATIVE 1
#define JUMP_TABLES_IN_TEXT_SECTION (!TARGET_JUMP_TABLES_IN_DATA_SECTION)
#undef ASM_OUTPUT_BEFORE_CASE_LABEL
#define ASM_OUTPUT_BEFORE_CASE_LABEL(FILE,PREFIX,NUM,TABLE) \
ASM_OUTPUT_ALIGN ((FILE), (TARGET_BIG_SWITCH ? 2 : 1))
#define WORD_REGISTER_OPERATIONS 1
#define LOAD_EXTEND_OP(MODE) SIGN_EXTEND
#define MOVE_MAX	4
#define SHIFT_COUNT_TRUNCATED 1
#define Pmode SImode
#define FUNCTION_MODE QImode
#define REGISTER_TARGET_PRAGMAS() do {				\
c_register_pragma ("ghs", "interrupt", ghs_pragma_interrupt);	\
c_register_pragma ("ghs", "section",   ghs_pragma_section);	\
c_register_pragma ("ghs", "starttda",  ghs_pragma_starttda);	\
c_register_pragma ("ghs", "startsda",  ghs_pragma_startsda);	\
c_register_pragma ("ghs", "startzda",  ghs_pragma_startzda);	\
c_register_pragma ("ghs", "endtda",    ghs_pragma_endtda);	\
c_register_pragma ("ghs", "endsda",    ghs_pragma_endsda);	\
c_register_pragma ("ghs", "endzda",    ghs_pragma_endzda);	\
} while (0)
enum GHS_section_kind
{ 
GHS_SECTION_KIND_DEFAULT,
GHS_SECTION_KIND_TEXT,
GHS_SECTION_KIND_DATA, 
GHS_SECTION_KIND_RODATA,
GHS_SECTION_KIND_BSS,
GHS_SECTION_KIND_SDATA,
GHS_SECTION_KIND_ROSDATA,
GHS_SECTION_KIND_TDATA,
GHS_SECTION_KIND_ZDATA,
GHS_SECTION_KIND_ROZDATA,
COUNT_OF_GHS_SECTION_KINDS  
};
typedef struct data_area_stack_element
{
struct data_area_stack_element * prev;
v850_data_area                   data_area; 
} data_area_stack_element;
extern data_area_stack_element * data_area_stack;
extern const char * GHS_default_section_names [(int) COUNT_OF_GHS_SECTION_KINDS];
extern const char * GHS_current_section_names [(int) COUNT_OF_GHS_SECTION_KINDS];
#define FILE_ASM_OP "\t.file\n"
#define EP_REGNUM 30	
#define SYMBOL_FLAG_ZDA		(SYMBOL_FLAG_MACH_DEP << 0)
#define SYMBOL_FLAG_TDA		(SYMBOL_FLAG_MACH_DEP << 1)
#define SYMBOL_FLAG_SDA		(SYMBOL_FLAG_MACH_DEP << 2)
#define SYMBOL_REF_ZDA_P(X)	((SYMBOL_REF_FLAGS (X) & SYMBOL_FLAG_ZDA) != 0)
#define SYMBOL_REF_TDA_P(X)	((SYMBOL_REF_FLAGS (X) & SYMBOL_FLAG_TDA) != 0)
#define SYMBOL_REF_SDA_P(X)	((SYMBOL_REF_FLAGS (X) & SYMBOL_FLAG_SDA) != 0)
#define TARGET_ASM_INIT_SECTIONS v850_asm_init_sections
#define NO_IMPLICIT_EXTERN_C
#define ADJUST_INSN_LENGTH(INSN, LENGTH) \
((LENGTH) = v850_adjust_insn_length ((INSN), (LENGTH)))
#endif 
