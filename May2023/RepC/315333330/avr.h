typedef struct
{
unsigned char id;
int memory_class;
int pointer_size;
const char *name;
int segment;
const char *section_name;
} avr_addrspace_t;
extern const avr_addrspace_t avr_addrspace[];
enum
{
ADDR_SPACE_RAM, 
ADDR_SPACE_FLASH,
ADDR_SPACE_FLASH1,
ADDR_SPACE_FLASH2,
ADDR_SPACE_FLASH3,
ADDR_SPACE_FLASH4,
ADDR_SPACE_FLASH5,
ADDR_SPACE_MEMX,
ADDR_SPACE_COUNT
};
#define TARGET_CPU_CPP_BUILTINS()	avr_cpu_cpp_builtins (pfile)
#define AVR_SHORT_CALLS (TARGET_SHORT_CALLS                             \
&& avr_arch == &avr_arch_types[ARCH_AVRXMEGA3])
#define AVR_HAVE_JMP_CALL (avr_arch->have_jmp_call && ! AVR_SHORT_CALLS)
#define AVR_HAVE_MUL (avr_arch->have_mul)
#define AVR_HAVE_MOVW (avr_arch->have_movw_lpmx)
#define AVR_HAVE_LPM (!AVR_TINY)
#define AVR_HAVE_LPMX (avr_arch->have_movw_lpmx)
#define AVR_HAVE_ELPM (avr_arch->have_elpm)
#define AVR_HAVE_ELPMX (avr_arch->have_elpmx)
#define AVR_HAVE_RAMPD (avr_arch->have_rampd)
#define AVR_HAVE_RAMPX (avr_arch->have_rampd)
#define AVR_HAVE_RAMPY (avr_arch->have_rampd)
#define AVR_HAVE_RAMPZ (avr_arch->have_elpm             \
|| avr_arch->have_rampd)
#define AVR_HAVE_EIJMP_EICALL (avr_arch->have_eijmp_eicall)
#define AVR_HAVE_8BIT_SP                        \
(TARGET_TINY_STACK || avr_sp8)
#define AVR_HAVE_SPH (!avr_sp8)
#define AVR_2_BYTE_PC (!AVR_HAVE_EIJMP_EICALL)
#define AVR_3_BYTE_PC (AVR_HAVE_EIJMP_EICALL)
#define AVR_XMEGA (avr_arch->xmega_p)
#define AVR_TINY  (avr_arch->tiny_p)
#define BITS_BIG_ENDIAN 0
#define BYTES_BIG_ENDIAN 0
#define WORDS_BIG_ENDIAN 0
#ifdef IN_LIBGCC2
#define UNITS_PER_WORD 4
#else
#define UNITS_PER_WORD 1
#endif
#define POINTER_SIZE 16
#define MAX_FIXED_MODE_SIZE 32
#define PARM_BOUNDARY 8
#define FUNCTION_BOUNDARY 8
#define EMPTY_FIELD_BOUNDARY 8
#define BIGGEST_ALIGNMENT 8
#define TARGET_VTABLE_ENTRY_ALIGN 8
#define STRICT_ALIGNMENT 0
#define INT_TYPE_SIZE (TARGET_INT8 ? 8 : 16)
#define SHORT_TYPE_SIZE (INT_TYPE_SIZE == 8 ? INT_TYPE_SIZE : 16)
#define LONG_TYPE_SIZE (INT_TYPE_SIZE == 8 ? 16 : 32)
#define LONG_LONG_TYPE_SIZE (INT_TYPE_SIZE == 8 ? 32 : 64)
#define FLOAT_TYPE_SIZE 32
#define DOUBLE_TYPE_SIZE 32
#define LONG_DOUBLE_TYPE_SIZE 32
#define LONG_LONG_ACCUM_TYPE_SIZE 64
#define DEFAULT_SIGNED_CHAR 1
#define SIZE_TYPE (INT_TYPE_SIZE == 8 ? "long unsigned int" : "unsigned int")
#define PTRDIFF_TYPE (INT_TYPE_SIZE == 8 ? "long int" :"int")
#define WCHAR_TYPE_SIZE 16
#define FIRST_PSEUDO_REGISTER 36
#define GENERAL_REGNO_P(N)	IN_RANGE (N, 2, 31)
#define GENERAL_REG_P(X)	(REG_P (X) && GENERAL_REGNO_P (REGNO (X)))
#define FIXED_REGISTERS {\
1,1,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
0,0,\
1,1,\
1,1   }
#define CALL_USED_REGISTERS {			\
1,1,				\
0,0,				\
0,0,				\
0,0,				\
0,0,				\
0,0,				\
0,0,				\
0,0,				\
0,0,				\
1,1,				\
1,1,				\
1,1,				\
1,1,				\
1,1,				\
0,0,				\
1,1,				\
1,1,				\
1,1   }
#define REG_ALLOC_ORDER {			\
24,25,					\
18,19,					\
20,21,					\
22,23,					\
30,31,					\
26,27,					\
28,29,					\
17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,	\
0,1,					\
32,33,34,35					\
}
#define ADJUST_REG_ALLOC_ORDER avr_adjust_reg_alloc_order()
enum reg_class {
NO_REGS,
R0_REG,			
POINTER_X_REGS,		
POINTER_Y_REGS,		
POINTER_Z_REGS,		
STACK_REG,			
BASE_POINTER_REGS,		
POINTER_REGS,			
ADDW_REGS,			
SIMPLE_LD_REGS,		
LD_REGS,			
NO_LD_REGS,			
GENERAL_REGS,			
ALL_REGS, LIM_REG_CLASSES
};
#define N_REG_CLASSES (int)LIM_REG_CLASSES
#define REG_CLASS_NAMES {					\
"NO_REGS",					\
"R0_REG",	                        \
"POINTER_X_REGS", 		\
"POINTER_Y_REGS", 		\
"POINTER_Z_REGS", 		\
"STACK_REG",				\
"BASE_POINTER_REGS",			\
"POINTER_REGS", 		\
"ADDW_REGS",				\
"SIMPLE_LD_REGS",             \
"LD_REGS",				\
"NO_LD_REGS",                  \
"GENERAL_REGS", 		\
"ALL_REGS" }
#define REG_CLASS_CONTENTS {						\
{0x00000000,0x00000000},					\
{0x00000001,0x00000000},	                            \
{3u << REG_X,0x00000000},     		\
{3u << REG_Y,0x00000000},     		\
{3u << REG_Z,0x00000000},     		\
{0x00000000,0x00000003},				\
{(3u << REG_Y) | (3u << REG_Z),					\
0x00000000},			\
{(3u << REG_X) | (3u << REG_Y) | (3u << REG_Z),			\
0x00000000},				\
{(3u << REG_X) | (3u << REG_Y) | (3u << REG_Z) | (3u << REG_W),	\
0x00000000},				\
{0x00ff0000,0x00000000},	          \
{(3u << REG_X)|(3u << REG_Y)|(3u << REG_Z)|(3u << REG_W)|(0xffu << 16),\
0x00000000},				\
{0x0000ffff,0x00000000},	              \
{0xffffffff,0x00000000},			\
{0xffffffff,0x00000003}					\
}
#define REGNO_REG_CLASS(R) avr_regno_reg_class(R)
#define MODE_CODE_BASE_REG_CLASS(mode, as, outer_code, index_code)   \
avr_mode_code_base_reg_class (mode, as, outer_code, index_code)
#define INDEX_REG_CLASS NO_REGS
#define REGNO_MODE_CODE_OK_FOR_BASE_P(num, mode, as, outer_code, index_code) \
avr_regno_mode_code_ok_for_base_p (num, mode, as, outer_code, index_code)
#define REGNO_OK_FOR_INDEX_P(NUM) 0
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P hook_bool_mode_true
#define STACK_PUSH_CODE POST_DEC
#define STACK_GROWS_DOWNWARD 1
#define STACK_POINTER_OFFSET 1
#define FIRST_PARM_OFFSET(FUNDECL) 0
#define STACK_BOUNDARY 8
#define STACK_POINTER_REGNUM 32
#define FRAME_POINTER_REGNUM REG_Y
#define ARG_POINTER_REGNUM 34
#define STATIC_CHAIN_REGNUM ((AVR_TINY) ? 18 :2)
#define ELIMINABLE_REGS {					\
{ ARG_POINTER_REGNUM, STACK_POINTER_REGNUM },               \
{ ARG_POINTER_REGNUM, FRAME_POINTER_REGNUM },               \
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM },             \
{ FRAME_POINTER_REGNUM + 1, STACK_POINTER_REGNUM + 1 } }
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)			\
OFFSET = avr_initial_elimination_offset (FROM, TO)
#define RETURN_ADDR_RTX(count, tem) avr_return_addr_rtx (count, tem)
typedef struct avr_args
{
int nregs;
int regno;
} CUMULATIVE_ARGS;
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS) \
avr_init_cumulative_args (&(CUM), FNTYPE, LIBNAME, FNDECL)
#define FUNCTION_ARG_REGNO_P(r) avr_function_arg_regno_p(r)
#define DEFAULT_PCC_STRUCT_RETURN 0
#define EPILOGUE_USES(REGNO) avr_epilogue_uses(REGNO)
#define HAVE_POST_INCREMENT 1
#define HAVE_PRE_DECREMENT 1
#define MAX_REGS_PER_ADDRESS 1
#define LEGITIMIZE_RELOAD_ADDRESS(X,MODE,OPNUM,TYPE,IND_L,WIN)          \
do {                                                                  \
rtx new_x = avr_legitimize_reload_address (&(X), MODE, OPNUM, TYPE, \
ADDR_TYPE (TYPE),        \
IND_L, make_memloc);     \
if (new_x)                                                          \
{                                                                 \
X = new_x;                                                      \
goto WIN;                                                       \
}                                                                 \
} while (0)
#define BRANCH_COST(speed_p, predictable_p)     \
(avr_branch_cost + (reload_completed ? 4 : 0))
#define SLOW_BYTE_ACCESS 0
#define NO_FUNCTION_CSE 1
#define REGISTER_TARGET_PRAGMAS()                                       \
do {                                                                  \
avr_register_target_pragmas();                                      \
} while (0)
#define TEXT_SECTION_ASM_OP "\t.text"
#define DATA_SECTION_ASM_OP "\t.data"
#define BSS_SECTION_ASM_OP "\t.section .bss"
#undef CTORS_SECTION_ASM_OP
#define CTORS_SECTION_ASM_OP "\t.section .ctors,\"a\",@progbits"
#undef DTORS_SECTION_ASM_OP
#define DTORS_SECTION_ASM_OP "\t.section .dtors,\"a\",@progbits"
#define TARGET_ASM_CONSTRUCTOR avr_asm_out_ctor
#define TARGET_ASM_DESTRUCTOR avr_asm_out_dtor
#define SUPPORTS_INIT_PRIORITY 0
#define JUMP_TABLES_IN_TEXT_SECTION 1
#define ASM_COMMENT_START " ; "
#define ASM_APP_ON "\n"
#define ASM_APP_OFF "\n"
#define IS_ASM_LOGICAL_LINE_SEPARATOR(C, STR) ((C) == '\n' || ((C) == '$'))
#define ASM_OUTPUT_ALIGNED_DECL_COMMON(STREAM, DECL, NAME, SIZE, ALIGN) \
avr_asm_output_aligned_decl_common (STREAM, DECL, NAME, SIZE, ALIGN, false)
#define ASM_OUTPUT_ALIGNED_BSS(FILE, DECL, NAME, SIZE, ALIGN) \
avr_asm_asm_output_aligned_bss (FILE, DECL, NAME, SIZE, ALIGN, \
asm_output_aligned_bss)
#define ASM_OUTPUT_ALIGNED_DECL_LOCAL(STREAM, DECL, NAME, SIZE, ALIGN)  \
avr_asm_output_aligned_decl_common (STREAM, DECL, NAME, SIZE, ALIGN, true)
#define GLOBAL_ASM_OP ".global\t"
#define SUPPORTS_WEAK 1
#define HAS_INIT_SECTION 1
#define REGISTER_NAMES {				\
"r0","r1","r2","r3","r4","r5","r6","r7",		\
"r8","r9","r10","r11","r12","r13","r14","r15",	\
"r16","r17","r18","r19","r20","r21","r22","r23",	\
"r24","r25","r26","r27","r28","r29","r30","r31",	\
"__SP_L__","__SP_H__","argL","argH"}
#define FINAL_PRESCAN_INSN(insn, operand, nop)  \
avr_final_prescan_insn (insn, operand,nop)
#define ASM_OUTPUT_REG_PUSH(STREAM, REGNO)	\
{						\
gcc_assert (REGNO < 32);			\
fprintf (STREAM, "\tpush\tr%d", REGNO);	\
}
#define ASM_OUTPUT_REG_POP(STREAM, REGNO)	\
{						\
gcc_assert (REGNO < 32);			\
fprintf (STREAM, "\tpop\tr%d", REGNO);	\
}
#define ASM_OUTPUT_ADDR_VEC(TLABEL, TDATA)      \
avr_output_addr_vec (TLABEL, TDATA)
#define ASM_OUTPUT_ALIGN(STREAM, POWER)                 \
do {                                                  \
if ((POWER) > 0)                                    \
fprintf (STREAM, "\t.p2align\t%d\n", POWER);      \
} while (0)
#define CASE_VECTOR_MODE HImode
#undef WORD_REGISTER_OPERATIONS
#define MOVE_MAX 1
#define MOVE_MAX_PIECES 2
#define MOVE_RATIO(speed) ((speed) ? 3 : 2)
#define Pmode HImode
#define FUNCTION_MODE HImode
#define DOLLARS_IN_IDENTIFIERS 0
#define TRAMPOLINE_SIZE 4
#define NOTICE_UPDATE_CC(EXP, INSN) avr_notice_update_cc (EXP, INSN)
#define CC_OVERFLOW_UNUSABLE 01000
#define CC_NO_CARRY CC_NO_OVERFLOW
#define FUNCTION_PROFILER(FILE, LABELNO)  \
fprintf (FILE, "", (LABELNO))
#define ADJUST_INSN_LENGTH(INSN, LENGTH)                \
(LENGTH = avr_adjust_insn_length (INSN, LENGTH))
extern const char *avr_devicespecs_file (int, const char**);
#define EXTRA_SPEC_FUNCTIONS                                   \
{ "device-specs-file", avr_devicespecs_file },
#undef  DRIVER_SELF_SPECS
#define DRIVER_SELF_SPECS                       \
" %:device-specs-file(device-specs%s %{mmcu=*:%*})"
#define LIBSTDCXX "gcc"
#define MULTILIB_DEFAULTS { "mmcu=" AVR_MMCU_DEFAULT }
#define TEST_HARD_REG_CLASS(CLASS, REGNO) \
TEST_HARD_REG_BIT (reg_class_contents[ (int) (CLASS)], REGNO)
#define CR_TAB "\n\t"
#define DWARF2_ADDR_SIZE 4
#define INCOMING_RETURN_ADDR_RTX   avr_incoming_return_addr_rtx ()
#define INCOMING_FRAME_SP_OFFSET   (AVR_3_BYTE_PC ? 3 : 2)
#define ARG_POINTER_CFA_OFFSET(FNDECL)  -1
#define HARD_REGNO_RENAME_OK(OLD_REG, NEW_REG) \
avr_hard_regno_rename_ok (OLD_REG, NEW_REG)
struct GTY(()) machine_function
{
int is_naked;
int is_interrupt;
int is_signal;
int is_OS_task;
int is_OS_main;
int stack_usage;
int sibcall_fails;
int attributes_checked_p;
int is_no_gccisr;
struct
{
int yes;
int maybe;
int regno;
} gasisr;
int use_L__stack_usage;
};
#define PUSH_ROUNDING(X)	(X)
extern int avr_accumulate_outgoing_args (void);
#define ACCUMULATE_OUTGOING_ARGS avr_accumulate_outgoing_args()
#define INIT_EXPANDERS avr_init_expanders()
#define SYMBOL_FLAG_IO_LOW	(SYMBOL_FLAG_MACH_DEP << 4)
#define SYMBOL_FLAG_IO		(SYMBOL_FLAG_MACH_DEP << 5)
#define SYMBOL_FLAG_ADDRESS	(SYMBOL_FLAG_MACH_DEP << 6)
