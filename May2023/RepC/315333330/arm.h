#ifndef GCC_ARM_H
#define GCC_ARM_H
#ifdef GENERATOR_FILE
#define MACHMODE int
#else
#include "insn-modes.h"
#define MACHMODE machine_mode
#endif
#include "config/vxworks-dummy.h"
extern char arm_arch_name[];
#define TARGET_CPU_CPP_BUILTINS() arm_cpu_cpp_builtins (pfile)
#include "config/arm/arm-opts.h"
extern enum processor_type arm_tune;
typedef enum arm_cond_code
{
ARM_EQ = 0, ARM_NE, ARM_CS, ARM_CC, ARM_MI, ARM_PL, ARM_VS, ARM_VC,
ARM_HI, ARM_LS, ARM_GE, ARM_LT, ARM_GT, ARM_LE, ARM_AL, ARM_NV
}
arm_cc;
extern arm_cc arm_current_cc;
#define ARM_INVERSE_CONDITION_CODE(X)  ((arm_cc) (((int)X) ^ 1))
#undef MAX_CONDITIONAL_EXECUTE
#define MAX_CONDITIONAL_EXECUTE arm_max_conditional_execute ()
extern int arm_target_label;
extern int arm_ccfsm_state;
extern GTY(()) rtx arm_target_insn;
extern void (*arm_lang_output_object_attributes_hook)(void);
extern tree arm_fp16_type_node;

#undef  CPP_SPEC
#define CPP_SPEC "%(subtarget_cpp_spec)					\
%{mfloat-abi=soft:%{mfloat-abi=hard:					\
%e-mfloat-abi=soft and -mfloat-abi=hard may not be used together}} \
%{mbig-endian:%{mlittle-endian:						\
%e-mbig-endian and -mlittle-endian may not be used together}}"
#ifndef CC1_SPEC
#define CC1_SPEC ""
#endif
#define EXTRA_SPECS						\
{ "subtarget_cpp_spec",	SUBTARGET_CPP_SPEC },           \
{ "asm_cpu_spec",		ASM_CPU_SPEC },			\
SUBTARGET_EXTRA_SPECS
#ifndef SUBTARGET_EXTRA_SPECS
#define SUBTARGET_EXTRA_SPECS
#endif
#ifndef SUBTARGET_CPP_SPEC
#define SUBTARGET_CPP_SPEC      ""
#endif

#define TARGET_ARM_P(flags)    (!TARGET_THUMB_P (flags))
#define TARGET_THUMB1_P(flags) (TARGET_THUMB_P (flags) && !arm_arch_thumb2)
#define TARGET_THUMB2_P(flags) (TARGET_THUMB_P (flags) && arm_arch_thumb2)
#define TARGET_32BIT_P(flags)  (TARGET_ARM_P (flags) || TARGET_THUMB2_P (flags))
#define TARGET_HARD_FLOAT	(arm_float_abi != ARM_FLOAT_ABI_SOFT	\
&& bitmap_bit_p (arm_active_target.isa, \
isa_bit_vfpv2))
#define TARGET_SOFT_FLOAT	(!TARGET_HARD_FLOAT)
#define TARGET_MAYBE_HARD_FLOAT (arm_float_abi != ARM_FLOAT_ABI_SOFT)
#define TARGET_HARD_FLOAT_ABI		(arm_float_abi == ARM_FLOAT_ABI_HARD)
#define TARGET_IWMMXT			(arm_arch_iwmmxt)
#define TARGET_IWMMXT2			(arm_arch_iwmmxt2)
#define TARGET_REALLY_IWMMXT		(TARGET_IWMMXT && TARGET_32BIT)
#define TARGET_REALLY_IWMMXT2		(TARGET_IWMMXT2 && TARGET_32BIT)
#define TARGET_IWMMXT_ABI (TARGET_32BIT && arm_abi == ARM_ABI_IWMMXT)
#define TARGET_ARM                      (! TARGET_THUMB)
#define TARGET_EITHER			1 
#define TARGET_BACKTRACE	        (crtl->is_leaf \
? TARGET_TPCS_LEAF_FRAME \
: TARGET_TPCS_FRAME)
#define TARGET_AAPCS_BASED \
(arm_abi != ARM_ABI_APCS && arm_abi != ARM_ABI_ATPCS)
#define TARGET_HARD_TP			(target_thread_pointer == TP_CP15)
#define TARGET_SOFT_TP			(target_thread_pointer == TP_SOFT)
#define TARGET_GNU2_TLS			(target_tls_dialect == TLS_GNU2)
#define TARGET_THUMB1			(TARGET_THUMB && !arm_arch_thumb2)
#define TARGET_32BIT			(TARGET_ARM || arm_arch_thumb2)
#define TARGET_THUMB2			(TARGET_THUMB && arm_arch_thumb2)
#define TARGET_THUMB1_ONLY		(TARGET_THUMB1 && !arm_arch_notm)
#define TARGET_LDRD			(arm_arch5e && ARM_DOUBLEWORD_ALIGN \
&& !TARGET_THUMB1)
#define TARGET_CRC32			(arm_arch_crc)
#define TARGET_VFPD32 (bitmap_bit_p (arm_active_target.isa, isa_bit_fp_d32))
#define TARGET_VFP3 (bitmap_bit_p (arm_active_target.isa, isa_bit_vfpv3))
#define TARGET_VFP5 (bitmap_bit_p (arm_active_target.isa, isa_bit_fpv5))
#define TARGET_VFP_SINGLE (!TARGET_VFP_DOUBLE)
#define TARGET_VFP_DOUBLE (bitmap_bit_p (arm_active_target.isa, isa_bit_fp_dbl))
#define TARGET_NEON_FP16					\
(bitmap_bit_p (arm_active_target.isa, isa_bit_neon)		\
&& bitmap_bit_p (arm_active_target.isa, isa_bit_fp16conv))
#define TARGET_FP16 (bitmap_bit_p (arm_active_target.isa, isa_bit_fp16conv))
#define TARGET_FP16_TO_DOUBLE						\
(TARGET_HARD_FLOAT && (TARGET_FP16 && TARGET_VFP5))
#define TARGET_FMA (bitmap_bit_p (arm_active_target.isa, isa_bit_vfpv4))
#define TARGET_CRYPTO (bitmap_bit_p (arm_active_target.isa, isa_bit_crypto))
#define TARGET_NEON							\
(TARGET_32BIT && TARGET_HARD_FLOAT					\
&& bitmap_bit_p (arm_active_target.isa, isa_bit_neon))
#define TARGET_NEON_RDMA (TARGET_NEON && arm_arch8_1)
#define TARGET_DOTPROD (TARGET_NEON					\
&& bitmap_bit_p (arm_active_target.isa,		\
isa_bit_dotprod)		\
&& arm_arch8_2)
#define TARGET_VFP_FP16INST \
(TARGET_32BIT && TARGET_HARD_FLOAT && TARGET_VFP5 && arm_fp16_inst)
#define TARGET_FP16FML (TARGET_NEON					\
&& bitmap_bit_p (arm_active_target.isa,	\
isa_bit_fp16fml)		\
&& arm_arch8_2)
#define TARGET_NEON_FP16INST (TARGET_VFP_FP16INST && TARGET_NEON_RDMA)
#define TARGET_ARM_QBIT \
(TARGET_32BIT && arm_arch5e && (arm_arch_notm || arm_arch7))
#define TARGET_ARM_SAT \
(TARGET_32BIT && arm_arch6 && (arm_arch_notm || arm_arch7))
#define TARGET_DSP_MULTIPLY \
(TARGET_32BIT && arm_arch5e && (arm_arch_notm || arm_arch7em))
#define TARGET_INT_SIMD \
(TARGET_32BIT && arm_arch6 && (arm_arch_notm || arm_arch7em))
#define TARGET_USE_MOVT \
(TARGET_HAVE_MOVT \
&& (arm_disable_literal_pool \
|| (!optimize_size && !current_tune->prefer_constant_pool)))
#define TARGET_HAVE_DMB		(arm_arch6m || arm_arch7)
#define TARGET_HAVE_DMB_MCR	(arm_arch6 && ! TARGET_HAVE_DMB \
&& ! TARGET_THUMB1)
#define TARGET_HAVE_MEMORY_BARRIER (TARGET_HAVE_DMB || TARGET_HAVE_DMB_MCR)
#define TARGET_HAVE_LDREX        ((arm_arch6 && TARGET_ARM)	\
|| arm_arch7			\
|| (arm_arch8 && !arm_arch_notm))
#define TARGET_HAVE_LPAE (arm_arch_lpae)
#define TARGET_HAVE_LDREXBH ((arm_arch6k && TARGET_ARM)		\
|| arm_arch7			\
|| (arm_arch8 && !arm_arch_notm))
#define TARGET_HAVE_LDREXD (((arm_arch6k && TARGET_ARM) \
|| arm_arch7) && arm_arch_notm)
#define TARGET_HAVE_LDACQ	(TARGET_ARM_ARCH >= 8)
#define TARGET_HAVE_LDACQEXD	(TARGET_ARM_ARCH >= 8	\
&& TARGET_32BIT	\
&& arm_arch_notm)
#define TARGET_HAVE_MOVT	(arm_arch_thumb2 || arm_arch8)
#define TARGET_HAVE_CBZ		(arm_arch_thumb2 || arm_arch8)
#define TARGET_IDIV	((TARGET_ARM && arm_arch_arm_hwdiv)	\
|| (TARGET_THUMB && arm_arch_thumb_hwdiv))
#define TARGET_NO_VOLATILE_CE		(arm_arch_no_volatile_ce)
#define TARGET_PREFER_NEON_64BITS (prefer_neon_for_64bits)
#define DONT_EARLY_SPLIT_CONSTANT(i, op) \
((optimize >= 2) \
&& can_create_pseudo_p () \
&& !const_ok_for_op (i, op))
#ifndef TARGET_BPABI
#define TARGET_BPABI false
#endif
#define NEON_ENDIAN_LANE_N(mode, n)  \
(BYTES_BIG_ENDIAN ? GET_MODE_NUNITS (mode) - 1 - n : n)
#define OPTION_DEFAULT_SPECS \
{"arch", "%{!march=*:%{!mcpu=*:-march=%(VALUE)}}" }, \
{"cpu", "%{!march=*:%{!mcpu=*:-mcpu=%(VALUE)}}" }, \
{"tune", "%{!mcpu=*:%{!mtune=*:-mtune=%(VALUE)}}" }, \
{"float", "%{!mfloat-abi=*:-mfloat-abi=%(VALUE)}" }, \
{"fpu", "%{!mfpu=*:-mfpu=%(VALUE)}"}, \
{"abi", "%{!mabi=*:-mabi=%(VALUE)}"}, \
{"mode", "%{!marm:%{!mthumb:-m%(VALUE)}}"}, \
{"tls", "%{!mtls-dialect=*:-mtls-dialect=%(VALUE)}"},
extern const struct arm_fpu_desc
{
const char *name;
enum isa_feature isa_bits[isa_num_bits];
} all_fpus[];
extern int arm_fpu_attr;
#ifndef TARGET_DEFAULT_FLOAT_ABI
#define TARGET_DEFAULT_FLOAT_ABI ARM_FLOAT_ABI_SOFT
#endif
#ifndef ARM_DEFAULT_ABI
#define ARM_DEFAULT_ABI ARM_ABI_APCS
#endif
#ifndef ARM_DEFAULT_SHORT_ENUMS
#define ARM_DEFAULT_SHORT_ENUMS \
(TARGET_AAPCS_BASED && arm_abi != ARM_ABI_AAPCS_LINUX)
#endif
enum base_architecture
{
BASE_ARCH_0 = 0,
BASE_ARCH_2 = 2,
BASE_ARCH_3 = 3,
BASE_ARCH_3M = 3,
BASE_ARCH_4 = 4,
BASE_ARCH_4T = 4,
BASE_ARCH_5 = 5,
BASE_ARCH_5E = 5,
BASE_ARCH_5T = 5,
BASE_ARCH_5TE = 5,
BASE_ARCH_5TEJ = 5,
BASE_ARCH_6 = 6,
BASE_ARCH_6J = 6,
BASE_ARCH_6KZ = 6,
BASE_ARCH_6K = 6,
BASE_ARCH_6T2 = 6,
BASE_ARCH_6M = 6,
BASE_ARCH_6Z = 6,
BASE_ARCH_7 = 7,
BASE_ARCH_7A = 7,
BASE_ARCH_7R = 7,
BASE_ARCH_7M = 7,
BASE_ARCH_7EM = 7,
BASE_ARCH_8A = 8,
BASE_ARCH_8M_BASE = 8,
BASE_ARCH_8M_MAIN = 8,
BASE_ARCH_8R = 8
};
extern enum base_architecture arm_base_arch;
extern int arm_arch3m;
extern int arm_arch4;
extern int arm_arch4t;
extern int arm_arch5;
extern int arm_arch5e;
extern int arm_arch6;
extern int arm_arch6k;
extern int arm_arch6m;
extern int arm_arch7;
extern int arm_arch_notm;
extern int arm_arch7em;
extern int arm_arch8;
extern int arm_arch8_1;
extern int arm_arch8_2;
extern int arm_fp16_inst;
extern int arm_ld_sched;
extern int arm_tune_strongarm;
extern int arm_arch_iwmmxt;
extern int arm_arch_iwmmxt2;
extern int arm_arch_xscale;
extern int arm_tune_xscale;
extern int arm_tune_wbuf;
extern int arm_tune_cortex_a9;
extern int arm_cpp_interwork;
extern int arm_arch_thumb1;
extern int arm_arch_thumb2;
extern int arm_arch_arm_hwdiv;
extern int arm_arch_thumb_hwdiv;
extern int arm_arch_no_volatile_ce;
extern int prefer_neon_for_64bits;
#ifndef USED_FOR_TARGET
extern bool arm_disable_literal_pool;
#endif
extern int arm_arch_crc;
extern int arm_arch_cmse;
#ifndef TARGET_DEFAULT
#define TARGET_DEFAULT  (MASK_APCS_FRAME)
#endif
#ifndef NEED_GOT_RELOC
#define NEED_GOT_RELOC	0
#endif
#ifndef NEED_PLT_RELOC
#define NEED_PLT_RELOC	0
#endif
#ifndef TARGET_DEFAULT_PIC_DATA_IS_TEXT_RELATIVE
#define TARGET_DEFAULT_PIC_DATA_IS_TEXT_RELATIVE 1
#endif
#ifndef GOT_PCREL
#define GOT_PCREL   1
#endif

#define PROMOTE_MODE(MODE, UNSIGNEDP, TYPE)	\
if (GET_MODE_CLASS (MODE) == MODE_INT		\
&& GET_MODE_SIZE (MODE) < 4)      	\
{						\
(MODE) = SImode;				\
}
#define BITS_BIG_ENDIAN  0
#define BYTES_BIG_ENDIAN  (TARGET_BIG_END != 0)
#define WORDS_BIG_ENDIAN  (BYTES_BIG_ENDIAN)
#define UNITS_PER_WORD	4
#define ARM_DOUBLEWORD_ALIGN	TARGET_AAPCS_BASED
#define DOUBLEWORD_ALIGNMENT 64
#define PARM_BOUNDARY  	32
#define STACK_BOUNDARY  (ARM_DOUBLEWORD_ALIGN ? DOUBLEWORD_ALIGNMENT : 32)
#define PREFERRED_STACK_BOUNDARY \
(arm_abi == ARM_ABI_ATPCS ? 64 : STACK_BOUNDARY)
#define FUNCTION_BOUNDARY_P(flags)  (TARGET_THUMB_P (flags) ? 16 : 32)
#define FUNCTION_BOUNDARY           (FUNCTION_BOUNDARY_P (target_flags))
#define TARGET_PTRMEMFUNC_VBIT_LOCATION ptrmemfunc_vbit_in_delta
#define EMPTY_FIELD_BOUNDARY  32
#define BIGGEST_ALIGNMENT (ARM_DOUBLEWORD_ALIGN ? DOUBLEWORD_ALIGNMENT : 32)
#define MALLOC_ABI_ALIGNMENT  BIGGEST_ALIGNMENT
#ifdef IN_TARGET_LIBS
#define BIGGEST_FIELD_ALIGNMENT 64
#endif
#define ARM_EXPAND_ALIGNMENT(COND, EXP, ALIGN)				\
(((COND) && ((ALIGN) < BITS_PER_WORD)					\
&& (TREE_CODE (EXP) == ARRAY_TYPE					\
|| TREE_CODE (EXP) == UNION_TYPE				\
|| TREE_CODE (EXP) == RECORD_TYPE)) ? BITS_PER_WORD : (ALIGN))
#define DATA_ALIGNMENT(EXP, ALIGN)			\
ARM_EXPAND_ALIGNMENT(!optimize_size, EXP, ALIGN)
#define LOCAL_ALIGNMENT(EXP, ALIGN)				\
ARM_EXPAND_ALIGNMENT(!flag_conserve_stack, EXP, ALIGN)
#define STRUCTURE_SIZE_BOUNDARY arm_structure_size_boundary
#ifndef DEFAULT_STRUCTURE_SIZE_BOUNDARY
#define DEFAULT_STRUCTURE_SIZE_BOUNDARY 32
#endif
#define STRICT_ALIGNMENT 1
#ifndef WCHAR_TYPE
#define WCHAR_TYPE (TARGET_AAPCS_BASED ? "unsigned int" : "int")
#define WCHAR_TYPE_SIZE BITS_PER_WORD
#endif
#define SHORT_FRACT_TYPE_SIZE 8
#define FRACT_TYPE_SIZE 16
#define LONG_FRACT_TYPE_SIZE 32
#define LONG_LONG_FRACT_TYPE_SIZE 64
#define SHORT_ACCUM_TYPE_SIZE 16
#define ACCUM_TYPE_SIZE 32
#define LONG_ACCUM_TYPE_SIZE 64
#define LONG_LONG_ACCUM_TYPE_SIZE 64
#define MAX_FIXED_MODE_SIZE 64
#ifndef SIZE_TYPE
#define SIZE_TYPE (TARGET_AAPCS_BASED ? "unsigned int" : "long unsigned int")
#endif
#ifndef PTRDIFF_TYPE
#define PTRDIFF_TYPE (TARGET_AAPCS_BASED ? "int" : "long int")
#endif
#ifndef PCC_BITFIELD_TYPE_MATTERS
#define PCC_BITFIELD_TYPE_MATTERS TARGET_AAPCS_BASED
#endif
#ifndef MAX_SYNC_LIBFUNC_SIZE
#define MAX_SYNC_LIBFUNC_SIZE (2 * UNITS_PER_WORD)
#endif

#define FIXED_REGISTERS 	\
{				\
\
0,0,0,0,0,0,0,0,		\
0,0,0,0,0,1,0,1,		\
\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,			\
\
1,1,1,1			\
}
#define CALL_USED_REGISTERS	\
{				\
\
1,1,1,1,0,0,0,0,		\
0,0,0,0,1,1,1,1,		\
\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
\
1,1,1,1,1,1,1,1,		\
1,1,1,1,1,1,1,1,		\
1,1,1,1,			\
\
1,1,1,1			\
}
#ifndef SUBTARGET_CONDITIONAL_REGISTER_USAGE
#define SUBTARGET_CONDITIONAL_REGISTER_USAGE
#endif
#define ASM_FPRINTF_EXTENSIONS(FILE, ARGS, P)		\
case '@':						\
fputs (ASM_COMMENT_START, FILE);			\
break;						\
\
case 'r':						\
fputs (REGISTER_PREFIX, FILE);			\
fputs (reg_names [va_arg (ARGS, int)], FILE);	\
break;
#define ROUND_UP_WORD(X) (((X) + 3) & ~3)
#define ARM_NUM_INTS(X) (((X) + UNITS_PER_WORD - 1) / UNITS_PER_WORD)
#define ARM_NUM_REGS(MODE)				\
ARM_NUM_INTS (GET_MODE_SIZE (MODE))
#define ARM_NUM_REGS2(MODE, TYPE)                   \
ARM_NUM_INTS ((MODE) == BLKmode ? 		\
int_size_in_bytes (TYPE) : GET_MODE_SIZE (MODE))
#define NUM_ARG_REGS		4
#define NUM_VFP_ARG_REGS	16
#define ARG_REGISTER(N) 	(N - 1)
#define LAST_ARG_REGNUM 	ARG_REGISTER (NUM_ARG_REGS)
#define FIRST_LO_REGNUM  	0
#define LAST_LO_REGNUM  	7
#define FIRST_HI_REGNUM		8
#define LAST_HI_REGNUM		11
#ifndef ARM_UNWIND_INFO
#define ARM_UNWIND_INFO  0
#endif
#define EH_RETURN_DATA_REGNO(N) (((N) < 2) ? N : INVALID_REGNUM)
#define ARM_EH_STACKADJ_REGNUM	2
#define EH_RETURN_STACKADJ_RTX	gen_rtx_REG (SImode, ARM_EH_STACKADJ_REGNUM)
#ifndef ARM_TARGET2_DWARF_FORMAT
#define ARM_TARGET2_DWARF_FORMAT DW_EH_PE_pcrel
#endif
#define ASM_PREFERRED_EH_DATA_FORMAT(code, data) \
(((code) == 0 && (data) == 1 && ARM_UNWIND_INFO) ? ARM_TARGET2_DWARF_FORMAT \
: DW_EH_PE_absptr)
#define STATIC_CHAIN_REGNUM	12
#define ARM_HARD_FRAME_POINTER_REGNUM	11
#define THUMB_HARD_FRAME_POINTER_REGNUM	 7
#define HARD_FRAME_POINTER_REGNUM		\
(TARGET_ARM					\
? ARM_HARD_FRAME_POINTER_REGNUM		\
: THUMB_HARD_FRAME_POINTER_REGNUM)
#define HARD_FRAME_POINTER_IS_FRAME_POINTER 0
#define HARD_FRAME_POINTER_IS_ARG_POINTER 0
#define FP_REGNUM	                HARD_FRAME_POINTER_REGNUM
#define STACK_POINTER_REGNUM	SP_REGNUM
#define FIRST_IWMMXT_REGNUM	(LAST_HI_VFP_REGNUM + 1)
#define LAST_IWMMXT_REGNUM	(FIRST_IWMMXT_REGNUM + 15)
#define FIRST_IWMMXT_GR_REGNUM	(LAST_IWMMXT_REGNUM + 1)
#define LAST_IWMMXT_GR_REGNUM	(FIRST_IWMMXT_GR_REGNUM + 3)
#define IS_IWMMXT_REGNUM(REGNUM) \
(((REGNUM) >= FIRST_IWMMXT_REGNUM) && ((REGNUM) <= LAST_IWMMXT_REGNUM))
#define IS_IWMMXT_GR_REGNUM(REGNUM) \
(((REGNUM) >= FIRST_IWMMXT_GR_REGNUM) && ((REGNUM) <= LAST_IWMMXT_GR_REGNUM))
#define FRAME_POINTER_REGNUM	102
#define ARG_POINTER_REGNUM	103
#define FIRST_VFP_REGNUM	16
#define D7_VFP_REGNUM		(FIRST_VFP_REGNUM + 15)
#define LAST_VFP_REGNUM	\
(TARGET_VFPD32 ? LAST_HI_VFP_REGNUM : LAST_LO_VFP_REGNUM)
#define IS_VFP_REGNUM(REGNUM) \
(((REGNUM) >= FIRST_VFP_REGNUM) && ((REGNUM) <= LAST_VFP_REGNUM))
#define LAST_LO_VFP_REGNUM	(FIRST_VFP_REGNUM + 31)
#define FIRST_HI_VFP_REGNUM	(LAST_LO_VFP_REGNUM + 1)
#define LAST_HI_VFP_REGNUM	(FIRST_HI_VFP_REGNUM + 31)
#define VFP_REGNO_OK_FOR_SINGLE(REGNUM) \
((REGNUM) <= LAST_LO_VFP_REGNUM)
#define VFP_REGNO_OK_FOR_DOUBLE(REGNUM) \
((((REGNUM) - FIRST_VFP_REGNUM) & 1) == 0)
#define NEON_REGNO_OK_FOR_QUAD(REGNUM) \
((((REGNUM) - FIRST_VFP_REGNUM) & 3) == 0)
#define NEON_REGNO_OK_FOR_NREGS(REGNUM, N) \
((((REGNUM) - FIRST_VFP_REGNUM) & 3) == 0 \
&& (LAST_VFP_REGNUM - (REGNUM) >= 2 * (N) - 1))
#define FIRST_PSEUDO_REGISTER   104
#define DBX_REGISTER_NUMBER(REGNO) arm_dbx_register_number (REGNO)
#ifndef SUBTARGET_FRAME_POINTER_REQUIRED
#define SUBTARGET_FRAME_POINTER_REQUIRED 0
#endif
#define VALID_IWMMXT_REG_MODE(MODE) \
(arm_vector_mode_supported_p (MODE) || (MODE) == DImode)
#define VALID_NEON_DREG_MODE(MODE) \
((MODE) == V2SImode || (MODE) == V4HImode || (MODE) == V8QImode \
|| (MODE) == V4HFmode || (MODE) == V2SFmode || (MODE) == DImode)
#define VALID_NEON_QREG_MODE(MODE) \
((MODE) == V4SImode || (MODE) == V8HImode || (MODE) == V16QImode \
|| (MODE) == V8HFmode || (MODE) == V4SFmode || (MODE) == V2DImode)
#define VALID_NEON_STRUCT_MODE(MODE) \
((MODE) == TImode || (MODE) == EImode || (MODE) == OImode \
|| (MODE) == CImode || (MODE) == XImode)
extern int arm_regs_in_sequence[];
#define VREG(X)  (FIRST_VFP_REGNUM + (X))
#define WREG(X)  (FIRST_IWMMXT_REGNUM + (X))
#define WGREG(X) (FIRST_IWMMXT_GR_REGNUM + (X))
#define REG_ALLOC_ORDER				\
{						\
\
3,  2,  1,  0,  12, 14,  4,  5,		\
6,  7,  8,  9,  10, 11,			\
\
VREG(32), VREG(33), VREG(34), VREG(35),	\
VREG(36), VREG(37), VREG(38), VREG(39),	\
VREG(40), VREG(41), VREG(42), VREG(43),	\
VREG(44), VREG(45), VREG(46), VREG(47),	\
VREG(48), VREG(49), VREG(50), VREG(51),	\
VREG(52), VREG(53), VREG(54), VREG(55),	\
VREG(56), VREG(57), VREG(58), VREG(59),	\
VREG(60), VREG(61), VREG(62), VREG(63),	\
\
VREG(15), VREG(14), VREG(13), VREG(12),	\
VREG(11), VREG(10), VREG(9),  VREG(8),	\
VREG(7),  VREG(6),  VREG(5),  VREG(4),	\
VREG(3),  VREG(2),  VREG(1),  VREG(0),	\
\
VREG(16), VREG(17), VREG(18), VREG(19),	\
VREG(20), VREG(21), VREG(22), VREG(23),	\
VREG(24), VREG(25), VREG(26), VREG(27),	\
VREG(28), VREG(29), VREG(30), VREG(31),	\
\
WREG(0),  WREG(1),  WREG(2),  WREG(3),	\
WREG(4),  WREG(5),  WREG(6),  WREG(7),	\
WREG(8),  WREG(9),  WREG(10), WREG(11),	\
WREG(12), WREG(13), WREG(14), WREG(15),	\
WGREG(0), WGREG(1), WGREG(2), WGREG(3),	\
\
CC_REGNUM, VFPCC_REGNUM,			\
FRAME_POINTER_REGNUM, ARG_POINTER_REGNUM,	\
SP_REGNUM, PC_REGNUM 				\
}
#define ADJUST_REG_ALLOC_ORDER arm_order_regs_for_local_alloc ()
#define HONOR_REG_ALLOC_ORDER 1
#define HARD_REGNO_RENAME_OK(SRC, DST)					\
(! IS_INTERRUPT (cfun->machine->func_type) ||			\
df_regs_ever_live_p (DST))

enum reg_class
{
NO_REGS,
LO_REGS,
STACK_REG,
BASE_REGS,
HI_REGS,
CALLER_SAVE_REGS,
GENERAL_REGS,
CORE_REGS,
VFP_D0_D7_REGS,
VFP_LO_REGS,
VFP_HI_REGS,
VFP_REGS,
IWMMXT_REGS,
IWMMXT_GR_REGS,
CC_REG,
VFPCC_REG,
SFP_REG,
AFP_REG,
ALL_REGS,
LIM_REG_CLASSES
};
#define N_REG_CLASSES  (int) LIM_REG_CLASSES
#define REG_CLASS_NAMES  \
{			\
"NO_REGS",		\
"LO_REGS",		\
"STACK_REG",		\
"BASE_REGS",		\
"HI_REGS",		\
"CALLER_SAVE_REGS",	\
"GENERAL_REGS",	\
"CORE_REGS",		\
"VFP_D0_D7_REGS",	\
"VFP_LO_REGS",	\
"VFP_HI_REGS",	\
"VFP_REGS",		\
"IWMMXT_REGS",	\
"IWMMXT_GR_REGS",	\
"CC_REG",		\
"VFPCC_REG",		\
"SFP_REG",		\
"AFP_REG",		\
"ALL_REGS"		\
}
#define REG_CLASS_CONTENTS						\
{									\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0x000000FF, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0x00002000, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0x000020FF, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0x00005F00, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0x0000100F, 0x00000000, 0x00000000, 0x00000000 },  \
{ 0x00005FFF, 0x00000000, 0x00000000, 0x00000000 },  \
{ 0x00007FFF, 0x00000000, 0x00000000, 0x00000000 }, 	\
{ 0xFFFF0000, 0x00000000, 0x00000000, 0x00000000 },  \
{ 0xFFFF0000, 0x0000FFFF, 0x00000000, 0x00000000 },  \
{ 0x00000000, 0xFFFF0000, 0x0000FFFF, 0x00000000 },  \
{ 0xFFFF0000, 0xFFFFFFFF, 0x0000FFFF, 0x00000000 }, 	\
{ 0x00000000, 0x00000000, 0xFFFF0000, 0x00000000 }, 	\
{ 0x00000000, 0x00000000, 0x00000000, 0x0000000F },  \
{ 0x00000000, 0x00000000, 0x00000000, 0x00000010 }, 	\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000020 }, 	\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000040 }, 	\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000080 }, 	\
{ 0xFFFF7FFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000000F }  	\
}
#define IS_VFP_CLASS(X) \
((X) == VFP_D0_D7_REGS || (X) == VFP_LO_REGS \
|| (X) == VFP_HI_REGS || (X) == VFP_REGS)
#define REGNO_REG_CLASS(REGNO)  arm_regno_class (REGNO)
#define INDEX_REG_CLASS  (TARGET_THUMB1 ? LO_REGS : GENERAL_REGS)
#define BASE_REG_CLASS   (TARGET_THUMB1 ? LO_REGS : CORE_REGS)
#define MODE_BASE_REG_CLASS(MODE)				\
(TARGET_32BIT ? CORE_REGS					\
: GET_MODE_SIZE (MODE) >= 4 ? BASE_REGS			\
: LO_REGS)
#define MODE_BASE_REG_REG_CLASS(MODE) BASE_REG_CLASS
#define TARGET_SMALL_REGISTER_CLASSES_FOR_MODE_P \
arm_small_register_classes_for_mode_p 
#define THUMB_SECONDARY_INPUT_RELOAD_CLASS(CLASS, MODE, X)		\
(lra_in_progress ? NO_REGS						\
: ((CLASS) != LO_REGS && (CLASS) != BASE_REGS			\
? ((true_regnum (X) == -1 ? LO_REGS				\
: (true_regnum (X) + hard_regno_nregs (0, MODE) > 8) ? LO_REGS	\
: NO_REGS)) 							\
: NO_REGS))
#define THUMB_SECONDARY_OUTPUT_RELOAD_CLASS(CLASS, MODE, X)		\
(lra_in_progress ? NO_REGS						\
: (CLASS) != LO_REGS && (CLASS) != BASE_REGS				\
? ((true_regnum (X) == -1 ? LO_REGS				\
: (true_regnum (X) + hard_regno_nregs (0, MODE) > 8) ? LO_REGS	\
: NO_REGS)) 							\
: NO_REGS)
#define SECONDARY_OUTPUT_RELOAD_CLASS(CLASS, MODE, X)		\
\
((TARGET_HARD_FLOAT && IS_VFP_CLASS (CLASS))			\
? coproc_secondary_reload_class (MODE, X, FALSE)		\
: (TARGET_IWMMXT && (CLASS) == IWMMXT_REGS)			\
? coproc_secondary_reload_class (MODE, X, TRUE)		\
: TARGET_32BIT						\
? (((MODE) == HImode && ! arm_arch4 && true_regnum (X) == -1) \
? GENERAL_REGS : NO_REGS)					\
: THUMB_SECONDARY_OUTPUT_RELOAD_CLASS (CLASS, MODE, X))
#define SECONDARY_INPUT_RELOAD_CLASS(CLASS, MODE, X)		\
\
((TARGET_HARD_FLOAT && IS_VFP_CLASS (CLASS))			\
? coproc_secondary_reload_class (MODE, X, FALSE) :		\
(TARGET_IWMMXT && (CLASS) == IWMMXT_REGS) ?			\
coproc_secondary_reload_class (MODE, X, TRUE) :		\
(TARGET_32BIT ?						\
(((CLASS) == IWMMXT_REGS || (CLASS) == IWMMXT_GR_REGS)	\
&& CONSTANT_P (X))						\
? GENERAL_REGS :						\
(((MODE) == HImode && ! arm_arch4				\
&& (MEM_P (X)					\
|| ((REG_P (X) || GET_CODE (X) == SUBREG)	\
&& true_regnum (X) == -1)))			\
? GENERAL_REGS : NO_REGS)					\
: THUMB_SECONDARY_INPUT_RELOAD_CLASS (CLASS, MODE, X)))
#define CLASS_MAX_NREGS(CLASS, MODE)  \
(ARM_NUM_REGS (MODE))

#define STACK_GROWS_DOWNWARD  1
#define FRAME_GROWS_DOWNWARD 1
#define CALLER_INTERWORKING_SLOT_SIZE			\
(TARGET_CALLER_INTERWORKING				\
&& maybe_ne (crtl->outgoing_args_size, 0)		\
? UNITS_PER_WORD : 0)
#define ACCUMULATE_OUTGOING_ARGS 1
#define FIRST_PARM_OFFSET(FNDECL)  (TARGET_ARM ? 4 : 0)
#define APPLY_RESULT_SIZE arm_apply_result_size()
#define DEFAULT_PCC_STRUCT_RETURN 0
#define ARM_FT_UNKNOWN		 0 
#define ARM_FT_NORMAL		 1 
#define ARM_FT_INTERWORKED	 2 
#define ARM_FT_ISR		 4 
#define ARM_FT_FIQ		 5 
#define ARM_FT_EXCEPTION	 6 
#define ARM_FT_TYPE_MASK	((1 << 3) - 1)
#define ARM_FT_INTERRUPT	(1 << 2) 
#define ARM_FT_NAKED		(1 << 3) 
#define ARM_FT_VOLATILE		(1 << 4) 
#define ARM_FT_NESTED		(1 << 5) 
#define ARM_FT_STACKALIGN	(1 << 6) 
#define ARM_FT_CMSE_ENTRY	(1 << 7) 
#define ARM_FUNC_TYPE(t)	(t & ARM_FT_TYPE_MASK)
#define IS_INTERRUPT(t)		(t & ARM_FT_INTERRUPT)
#define IS_VOLATILE(t)     	(t & ARM_FT_VOLATILE)
#define IS_NAKED(t)        	(t & ARM_FT_NAKED)
#define IS_NESTED(t)       	(t & ARM_FT_NESTED)
#define IS_STACKALIGN(t)       	(t & ARM_FT_STACKALIGN)
#define IS_CMSE_ENTRY(t)	(t & ARM_FT_CMSE_ENTRY)
typedef struct GTY(()) arm_stack_offsets
{
int saved_args;	
int frame;		
int saved_regs;
int soft_frame;	
int locals_base;	
int outgoing_args;	
unsigned int saved_regs_mask;
}
arm_stack_offsets;
#if !defined(GENERATOR_FILE) && !defined (USED_FOR_TARGET)
typedef struct GTY(()) machine_function
{
rtx eh_epilogue_sp_ofs;
int far_jump_used;
int arg_pointer_live;
int lr_save_eliminated;
arm_stack_offsets stack_offsets;
unsigned long func_type;
int uses_anonymous_args;
int sibcall_blocked;
rtx pic_reg;
rtx call_via[14];
int return_used_this_function;
rtx thumb1_cc_insn;
rtx thumb1_cc_op0;
rtx thumb1_cc_op1;
machine_mode thumb1_cc_mode;
int after_arm_reorg;
int static_chain_stack_bytes;
}
machine_function;
#endif
extern GTY(()) rtx thumb_call_via_label[14];
#define ARM_NUM_COPROC_SLOTS 1
enum arm_pcs
{
ARM_PCS_AAPCS,	
ARM_PCS_AAPCS_VFP,	
ARM_PCS_AAPCS_IWMMXT, 
ARM_PCS_AAPCS_LOCAL,	
ARM_PCS_ATPCS,	
ARM_PCS_APCS,		
ARM_PCS_UNKNOWN
};
extern enum arm_pcs arm_pcs_default;
#if !defined (USED_FOR_TARGET)
typedef struct
{
int nregs;
int iwmmxt_nregs;
int named_count;
int nargs;
enum arm_pcs pcs_variant;
int aapcs_arg_processed;  
int aapcs_cprc_slot;      
int aapcs_ncrn;
int aapcs_next_ncrn;
rtx aapcs_reg;	    
int aapcs_partial;	    
int aapcs_cprc_failed[ARM_NUM_COPROC_SLOTS];
int can_split;	    
unsigned aapcs_vfp_regs_free;
unsigned aapcs_vfp_reg_alloc;
int aapcs_vfp_rcount;
MACHMODE aapcs_vfp_rmode;
} CUMULATIVE_ARGS;
#endif
#define BLOCK_REG_PADDING(MODE, TYPE, FIRST) \
(arm_pad_reg_upward (MODE, TYPE, FIRST) ? PAD_UPWARD : PAD_DOWNWARD)
#define PAD_VARARGS_DOWN \
((TARGET_AAPCS_BASED) ? 0 : BYTES_BIG_ENDIAN)
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS) \
arm_init_cumulative_args (&(CUM), (FNTYPE), (LIBNAME), (FNDECL))
#define FUNCTION_ARG_REGNO_P(REGNO)					\
(IN_RANGE ((REGNO), 0, 3)						\
|| (TARGET_AAPCS_BASED && TARGET_HARD_FLOAT				\
&& IN_RANGE ((REGNO), FIRST_VFP_REGNUM, FIRST_VFP_REGNUM + 15))	\
|| (TARGET_IWMMXT_ABI						\
&& IN_RANGE ((REGNO), FIRST_IWMMXT_REGNUM, FIRST_IWMMXT_REGNUM + 9)))

#ifndef ARM_MCOUNT_NAME
#define ARM_MCOUNT_NAME "*mcount"
#endif
#ifndef ARM_FUNCTION_PROFILER
#define ARM_FUNCTION_PROFILER(STREAM, LABELNO)  	\
{							\
char temp[20];					\
rtx sym;						\
\
asm_fprintf (STREAM, "\tmov\t%r, %r\n\tbl\t",		\
IP_REGNUM, LR_REGNUM);			\
assemble_name (STREAM, ARM_MCOUNT_NAME);		\
fputc ('\n', STREAM);					\
ASM_GENERATE_INTERNAL_LABEL (temp, "LP", LABELNO);	\
sym = gen_rtx_SYMBOL_REF (Pmode, temp);		\
assemble_aligned_integer (UNITS_PER_WORD, sym);	\
}
#endif
#ifdef THUMB_FUNCTION_PROFILER
#define FUNCTION_PROFILER(STREAM, LABELNO)		\
if (TARGET_ARM)					\
ARM_FUNCTION_PROFILER (STREAM, LABELNO)		\
else							\
THUMB_FUNCTION_PROFILER (STREAM, LABELNO)
#else
#define FUNCTION_PROFILER(STREAM, LABELNO)		\
ARM_FUNCTION_PROFILER (STREAM, LABELNO)
#endif
#define EXIT_IGNORE_STACK 1
#define EPILOGUE_USES(REGNO) (epilogue_completed && (REGNO) == LR_REGNUM)
#define USE_RETURN_INSN(ISCOND)				\
(TARGET_32BIT ? use_return_insn (ISCOND, NULL) : 0)
#define ELIMINABLE_REGS						\
{{ ARG_POINTER_REGNUM,        STACK_POINTER_REGNUM            },\
{ ARG_POINTER_REGNUM,        FRAME_POINTER_REGNUM            },\
{ ARG_POINTER_REGNUM,        ARM_HARD_FRAME_POINTER_REGNUM   },\
{ ARG_POINTER_REGNUM,        THUMB_HARD_FRAME_POINTER_REGNUM },\
{ FRAME_POINTER_REGNUM,      STACK_POINTER_REGNUM            },\
{ FRAME_POINTER_REGNUM,      ARM_HARD_FRAME_POINTER_REGNUM   },\
{ FRAME_POINTER_REGNUM,      THUMB_HARD_FRAME_POINTER_REGNUM }}
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET)			\
if (TARGET_ARM)							\
(OFFSET) = arm_compute_initial_elimination_offset (FROM, TO);	\
else									\
(OFFSET) = thumb_compute_initial_elimination_offset (FROM, TO)
#define DEBUGGER_ARG_OFFSET(value, addr) value ? value : arm_debugger_arg_offset (value, addr)
#define INIT_EXPANDERS  arm_init_expanders ()
#define TRAMPOLINE_SIZE  (TARGET_32BIT ? 16 : 20)
#define TRAMPOLINE_ALIGNMENT  32

#define HAVE_POST_INCREMENT   1
#define HAVE_PRE_INCREMENT    TARGET_32BIT
#define HAVE_POST_DECREMENT   TARGET_32BIT
#define HAVE_PRE_DECREMENT    TARGET_32BIT
#define HAVE_PRE_MODIFY_DISP  TARGET_32BIT
#define HAVE_POST_MODIFY_DISP TARGET_32BIT
#define HAVE_PRE_MODIFY_REG   TARGET_32BIT
#define HAVE_POST_MODIFY_REG  TARGET_32BIT
enum arm_auto_incmodes
{
ARM_POST_INC,
ARM_PRE_INC,
ARM_POST_DEC,
ARM_PRE_DEC
};
#define ARM_AUTOINC_VALID_FOR_MODE_P(mode, code) \
(TARGET_32BIT && arm_autoinc_modes_ok_p (mode, code))
#define USE_LOAD_POST_INCREMENT(mode) \
ARM_AUTOINC_VALID_FOR_MODE_P(mode, ARM_POST_INC)
#define USE_LOAD_PRE_INCREMENT(mode)  \
ARM_AUTOINC_VALID_FOR_MODE_P(mode, ARM_PRE_INC)
#define USE_LOAD_POST_DECREMENT(mode) \
ARM_AUTOINC_VALID_FOR_MODE_P(mode, ARM_POST_DEC)
#define USE_LOAD_PRE_DECREMENT(mode)  \
ARM_AUTOINC_VALID_FOR_MODE_P(mode, ARM_PRE_DEC)
#define USE_STORE_PRE_DECREMENT(mode) USE_LOAD_PRE_DECREMENT(mode)
#define USE_STORE_PRE_INCREMENT(mode) USE_LOAD_PRE_INCREMENT(mode)
#define USE_STORE_POST_DECREMENT(mode) USE_LOAD_POST_DECREMENT(mode)
#define USE_STORE_POST_INCREMENT(mode) USE_LOAD_POST_INCREMENT(mode)
#define TEST_REGNO(R, TEST, VALUE) \
((R TEST VALUE)	\
|| (reg_renumber && ((unsigned) reg_renumber[R] TEST VALUE)))
#define ARM_REGNO_OK_FOR_BASE_P(REGNO)			\
(TEST_REGNO (REGNO, <, PC_REGNUM)			\
|| TEST_REGNO (REGNO, ==, FRAME_POINTER_REGNUM)	\
|| TEST_REGNO (REGNO, ==, ARG_POINTER_REGNUM))
#define THUMB1_REGNO_MODE_OK_FOR_BASE_P(REGNO, MODE)		\
(TEST_REGNO (REGNO, <=, LAST_LO_REGNUM)			\
|| (GET_MODE_SIZE (MODE) >= 4				\
&& TEST_REGNO (REGNO, ==, STACK_POINTER_REGNUM)))
#define REGNO_MODE_OK_FOR_BASE_P(REGNO, MODE)		\
(TARGET_THUMB1					\
? THUMB1_REGNO_MODE_OK_FOR_BASE_P (REGNO, MODE)	\
: ARM_REGNO_OK_FOR_BASE_P (REGNO))
#define REGNO_MODE_OK_FOR_REG_BASE_P(X, MODE)	\
REGNO_MODE_OK_FOR_BASE_P (X, QImode)
#define REGNO_OK_FOR_INDEX_P(REGNO)	\
(REGNO_MODE_OK_FOR_BASE_P (REGNO, QImode) \
&& !TEST_REGNO (REGNO, ==, STACK_POINTER_REGNUM))
#define MAX_REGS_PER_ADDRESS 2
#define CONSTANT_ADDRESS_P(X)  			\
(GET_CODE (X) == SYMBOL_REF 			\
&& (CONSTANT_POOL_ADDRESS_P (X)		\
|| (TARGET_ARM && optimize > 0 && SYMBOL_REF_FLAG (X))))
#define ARM_OFFSETS_MUST_BE_WITHIN_SECTIONS_P 0
#ifndef TARGET_DEFAULT_WORD_RELOCATIONS
#define TARGET_DEFAULT_WORD_RELOCATIONS 0
#endif
#ifndef SUBTARGET_NAME_ENCODING_LENGTHS
#define SUBTARGET_NAME_ENCODING_LENGTHS
#endif
#define ARM_NAME_ENCODING_LENGTHS		\
case '*':  return 1;				\
SUBTARGET_NAME_ENCODING_LENGTHS
#undef  ASM_OUTPUT_LABELREF
#define ASM_OUTPUT_LABELREF(FILE, NAME)		\
arm_asm_output_labelref (FILE, NAME)
#define ASM_OUTPUT_OPCODE(STREAM, PTR)	\
if (TARGET_THUMB2)			\
thumb2_asm_output_opcode (STREAM);
#ifndef ARM_EABI_CTORS_SECTION_OP
#define ARM_EABI_CTORS_SECTION_OP \
"\t.section\t.init_array,\"aw\",%init_array"
#endif
#ifndef ARM_EABI_DTORS_SECTION_OP
#define ARM_EABI_DTORS_SECTION_OP \
"\t.section\t.fini_array,\"aw\",%fini_array"
#endif
#define ARM_CTORS_SECTION_OP \
"\t.section\t.ctors,\"aw\",%progbits"
#define ARM_DTORS_SECTION_OP \
"\t.section\t.dtors,\"aw\",%progbits"
#undef CTORS_SECTION_ASM_OP
#undef DTORS_SECTION_ASM_OP
#ifndef IN_LIBGCC2
# define CTORS_SECTION_ASM_OP \
(TARGET_AAPCS_BASED ? ARM_EABI_CTORS_SECTION_OP : ARM_CTORS_SECTION_OP)
# define DTORS_SECTION_ASM_OP \
(TARGET_AAPCS_BASED ? ARM_EABI_DTORS_SECTION_OP : ARM_DTORS_SECTION_OP)
#else 
# ifdef __ARM_EABI__
#   define CTOR_LIST_BEGIN asm (ARM_EABI_CTORS_SECTION_OP)
#   define CTOR_LIST_END 
#   define DTOR_LIST_BEGIN asm (ARM_EABI_DTORS_SECTION_OP)
#   define DTOR_LIST_END 
# else 
#   define CTORS_SECTION_ASM_OP ARM_CTORS_SECTION_OP
#   define DTORS_SECTION_ASM_OP ARM_DTORS_SECTION_OP
# endif 
#endif 
#ifndef TARGET_ARM_DYNAMIC_VAGUE_LINKAGE_P
#define TARGET_ARM_DYNAMIC_VAGUE_LINKAGE_P true
#endif
#define ARM_OUTPUT_FN_UNWIND(F, PROLOGUE) arm_output_fn_unwind (F, PROLOGUE)
#ifndef REG_OK_STRICT
#define ARM_REG_OK_FOR_BASE_P(X)		\
(REGNO (X) <= LAST_ARM_REGNUM			\
|| REGNO (X) >= FIRST_PSEUDO_REGISTER	\
|| REGNO (X) == FRAME_POINTER_REGNUM		\
|| REGNO (X) == ARG_POINTER_REGNUM)
#define ARM_REG_OK_FOR_INDEX_P(X)		\
((REGNO (X) <= LAST_ARM_REGNUM		\
&& REGNO (X) != STACK_POINTER_REGNUM)	\
|| REGNO (X) >= FIRST_PSEUDO_REGISTER	\
|| REGNO (X) == FRAME_POINTER_REGNUM		\
|| REGNO (X) == ARG_POINTER_REGNUM)
#define THUMB1_REG_MODE_OK_FOR_BASE_P(X, MODE)	\
(REGNO (X) <= LAST_LO_REGNUM			\
|| REGNO (X) >= FIRST_PSEUDO_REGISTER	\
|| (GET_MODE_SIZE (MODE) >= 4		\
&& (REGNO (X) == STACK_POINTER_REGNUM	\
|| (X) == hard_frame_pointer_rtx	\
|| (X) == arg_pointer_rtx)))
#define REG_STRICT_P 0
#else 
#define ARM_REG_OK_FOR_BASE_P(X) 		\
ARM_REGNO_OK_FOR_BASE_P (REGNO (X))
#define ARM_REG_OK_FOR_INDEX_P(X) 		\
ARM_REGNO_OK_FOR_INDEX_P (REGNO (X))
#define THUMB1_REG_MODE_OK_FOR_BASE_P(X, MODE)	\
THUMB1_REGNO_MODE_OK_FOR_BASE_P (REGNO (X), MODE)
#define REG_STRICT_P 1
#endif 
#define REG_MODE_OK_FOR_BASE_P(X, MODE)		\
(TARGET_THUMB1				\
? THUMB1_REG_MODE_OK_FOR_BASE_P (X, MODE)	\
: ARM_REG_OK_FOR_BASE_P (X))
#define THUMB1_REG_OK_FOR_INDEX_P(X) \
THUMB1_REG_MODE_OK_FOR_BASE_P (X, QImode)
#define REG_OK_FOR_INDEX_P(X)			\
(TARGET_THUMB1				\
? THUMB1_REG_OK_FOR_INDEX_P (X)		\
: ARM_REG_OK_FOR_INDEX_P (X))
#define REG_MODE_OK_FOR_REG_BASE_P(X, MODE)	\
REG_OK_FOR_INDEX_P (X)

#define ARM_BASE_REGISTER_RTX_P(X)  \
(REG_P (X) && ARM_REG_OK_FOR_BASE_P (X))
#define ARM_INDEX_REGISTER_RTX_P(X)  \
(REG_P (X) && ARM_REG_OK_FOR_INDEX_P (X))

#define CASE_VECTOR_MODE Pmode
#define CASE_VECTOR_PC_RELATIVE (TARGET_THUMB2				\
|| (TARGET_THUMB1			\
&& (optimize_size || flag_pic)))
#define CASE_VECTOR_SHORTEN_MODE(min, max, body)			\
(TARGET_THUMB1							\
? (min >= 0 && max < 512						\
? (ADDR_DIFF_VEC_FLAGS (body).offset_unsigned = 1, QImode)	\
: min >= -256 && max < 256					\
? (ADDR_DIFF_VEC_FLAGS (body).offset_unsigned = 0, QImode)	\
: min >= 0 && max < 8192						\
? (ADDR_DIFF_VEC_FLAGS (body).offset_unsigned = 1, HImode)	\
: min >= -4096 && max < 4096					\
? (ADDR_DIFF_VEC_FLAGS (body).offset_unsigned = 0, HImode)	\
: SImode)								\
: ((min < 0 || max >= 0x20000 || !TARGET_THUMB2) ? SImode		\
: (max >= 0x200) ? HImode						\
: QImode))
#ifndef DEFAULT_SIGNED_CHAR
#define DEFAULT_SIGNED_CHAR  0
#endif
#define MOVE_MAX 4
#undef  MOVE_RATIO
#define MOVE_RATIO(speed) (arm_tune_xscale ? 4 : 2)
#define WORD_REGISTER_OPERATIONS 1
#define LOAD_EXTEND_OP(MODE)						\
(TARGET_THUMB ? ZERO_EXTEND :						\
((arm_arch4 || (MODE) == QImode) ? ZERO_EXTEND			\
: ((BYTES_BIG_ENDIAN && (MODE) == HImode) ? SIGN_EXTEND : UNKNOWN)))
#define SLOW_BYTE_ACCESS 0
#define NO_FUNCTION_CSE 1
#define Pmode  SImode
#define FUNCTION_MODE  Pmode
#define ARM_FRAME_RTX(X)					\
(   (X) == frame_pointer_rtx || (X) == stack_pointer_rtx	\
|| (X) == arg_pointer_rtx)
#define BRANCH_COST(speed_p, predictable_p)			\
((arm_branch_cost != -1) ? arm_branch_cost :			\
(current_tune->branch_cost (speed_p, predictable_p)))
#define LOGICAL_OP_NON_SHORT_CIRCUIT					\
((optimize_size)							\
? (TARGET_THUMB ? false : true)					\
: TARGET_THUMB ? static_cast<bool> (current_tune->logical_op_non_short_circuit_thumb) \
: static_cast<bool> (current_tune->logical_op_non_short_circuit_arm))

extern unsigned arm_pic_register;
#define PIC_OFFSET_TABLE_REGNUM arm_pic_register
#define LEGITIMATE_PIC_OPERAND_P(X)					\
(!(symbol_mentioned_p (X)					\
|| label_mentioned_p (X)					\
|| (GET_CODE (X) == SYMBOL_REF				\
&& CONSTANT_POOL_ADDRESS_P (X)				\
&& (symbol_mentioned_p (get_pool_constant (X))		\
|| label_mentioned_p (get_pool_constant (X)))))	\
|| tls_mentioned_p (X))
extern int making_const_table;

#define REGISTER_TARGET_PRAGMAS() do {					\
c_register_pragma (0, "long_calls", arm_pr_long_calls);		\
c_register_pragma (0, "no_long_calls", arm_pr_no_long_calls);		\
c_register_pragma (0, "long_calls_off", arm_pr_long_calls_off);	\
arm_lang_object_attributes_init();					\
arm_register_target_pragmas();                                       \
} while (0)
#define SELECT_CC_MODE(OP, X, Y)  arm_select_cc_mode (OP, X, Y)
#define REVERSIBLE_CC_MODE(MODE) 1
#define REVERSE_CONDITION(CODE,MODE) \
(((MODE) == CCFPmode || (MODE) == CCFPEmode) \
? reverse_condition_maybe_unordered (code) \
: reverse_condition (code))
#define CLZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) \
((VALUE) = GET_MODE_UNIT_BITSIZE (MODE), 2)
#define CTZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) \
((VALUE) = GET_MODE_UNIT_BITSIZE (MODE), 2)

#define CC_STATUS_INIT \
do { cfun->machine->thumb1_cc_insn = NULL_RTX; } while (0)
#undef ASM_APP_ON
#define ASM_APP_ON (inline_asm_unified ? "\t.syntax unified\n" : \
"\t.syntax divided\n")
#undef  ASM_APP_OFF
#define ASM_APP_OFF (TARGET_ARM ? "\t.arm\n\t.syntax unified\n" : \
"\t.thumb\n\t.syntax unified\n")
#define ASM_OUTPUT_REG_PUSH(STREAM, REGNO)		\
do							\
{							\
if (TARGET_THUMB1					\
&& (REGNO) == STATIC_CHAIN_REGNUM)	\
{						\
asm_fprintf (STREAM, "\tpush\t{r7}\n");	\
asm_fprintf (STREAM, "\tmov\tr7, %r\n", REGNO);\
asm_fprintf (STREAM, "\tpush\t{r7}\n");	\
}						\
else						\
asm_fprintf (STREAM, "\tpush {%r}\n", REGNO);	\
} while (0)
#define ASM_OUTPUT_REG_POP(STREAM, REGNO)		\
do							\
{							\
if (TARGET_THUMB1					\
&& (REGNO) == STATIC_CHAIN_REGNUM)		\
{						\
asm_fprintf (STREAM, "\tpop\t{r7}\n");	\
asm_fprintf (STREAM, "\tmov\t%r, r7\n", REGNO);\
asm_fprintf (STREAM, "\tpop\t{r7}\n");	\
}						\
else						\
asm_fprintf (STREAM, "\tpop {%r}\n", REGNO);	\
} while (0)
#define ADDR_VEC_ALIGN(JUMPTABLE)	\
((TARGET_THUMB && GET_MODE (PATTERN (JUMPTABLE)) == SImode) ? 2 : 0)
#undef ASM_OUTPUT_BEFORE_CASE_LABEL
#define ASM_OUTPUT_BEFORE_CASE_LABEL(FILE, PREFIX, NUM, TABLE) 
#define LABEL_ALIGN_AFTER_BARRIER(LABEL)                \
(GET_CODE (PATTERN (prev_active_insn (LABEL))) == ADDR_DIFF_VEC \
? 1 : 0)
#define ARM_DECLARE_FUNCTION_NAME(STREAM, NAME, DECL) 	\
arm_declare_function_name ((STREAM), (NAME), (DECL));
#define ASM_OUTPUT_DEF_FROM_DECLS(FILE, DECL1, DECL2)		\
do						   		\
{								\
const char *const LABEL1 = XSTR (XEXP (DECL_RTL (decl), 0), 0); \
const char *const LABEL2 = IDENTIFIER_POINTER (DECL2);	\
\
if (TARGET_THUMB && TREE_CODE (DECL1) == FUNCTION_DECL)	\
{							\
fprintf (FILE, "\t.thumb_set ");			\
assemble_name (FILE, LABEL1);			   	\
fprintf (FILE, ",");			   		\
assemble_name (FILE, LABEL2);		   		\
fprintf (FILE, "\n");					\
}							\
else							\
ASM_OUTPUT_DEF (FILE, LABEL1, LABEL2);			\
}								\
while (0)
#ifdef HAVE_GAS_MAX_SKIP_P2ALIGN
#define ASM_OUTPUT_MAX_SKIP_ALIGN(FILE, LOG, MAX_SKIP)		\
if ((LOG) != 0)						\
{								\
if ((MAX_SKIP) == 0)					\
fprintf ((FILE), "\t.p2align %d\n", (int) (LOG));	\
else							\
fprintf ((FILE), "\t.p2align %d,,%d\n",			\
(int) (LOG), (int) (MAX_SKIP));		\
}
#endif

#define ADJUST_INSN_LENGTH(insn, length) \
if (TARGET_THUMB2 && GET_CODE (PATTERN (insn)) == COND_EXEC) \
length += 2;
#define FINAL_PRESCAN_INSN(INSN, OPVEC, NOPERANDS)	\
if (TARGET_ARM && optimize)				\
arm_final_prescan_insn (INSN);			\
else if (TARGET_THUMB2)				\
thumb2_final_prescan_insn (INSN);			\
else if (TARGET_THUMB1)				\
thumb1_final_prescan_insn (INSN)
#define ARM_SIGN_EXTEND(x)  ((HOST_WIDE_INT)			\
(HOST_BITS_PER_WIDE_INT <= 32 ? (unsigned HOST_WIDE_INT) (x)	\
: ((((unsigned HOST_WIDE_INT)(x)) & (unsigned HOST_WIDE_INT) 0xffffffff) |\
((((unsigned HOST_WIDE_INT)(x)) & (unsigned HOST_WIDE_INT) 0x80000000) \
? ((~ (unsigned HOST_WIDE_INT) 0)			\
& ~ (unsigned HOST_WIDE_INT) 0xffffffff)		\
: 0))))
#define RETURN_ADDR_RTX(COUNT, FRAME) \
arm_return_addr (COUNT, FRAME)
#define RETURN_ADDR_MASK26 (0x03fffffc)
#define INCOMING_RETURN_ADDR_RTX	gen_rtx_REG (Pmode, LR_REGNUM)
#define DWARF_FRAME_RETURN_COLUMN	DWARF_FRAME_REGNUM (LR_REGNUM)
#define MASK_RETURN_ADDR \
\
((arm_arch4 || TARGET_THUMB)						\
? (gen_int_mode ((unsigned long)0xffffffff, Pmode))			\
: arm_gen_return_addr_mask ())

#ifndef NEED_INDICATE_EXEC_STACK
#define NEED_INDICATE_EXEC_STACK	0
#endif
#define TARGET_ARM_ARCH	\
(arm_base_arch)	\
#define TARGET_ARM_ARCH_ISA_THUMB		\
(arm_arch_thumb2 ? 2 : (arm_arch_thumb1 ? 1 : 0))
#define TARGET_ARM_ARCH_PROFILE				\
(arm_active_target.profile)
#define TARGET_ARM_FEATURE_LDREX				\
((TARGET_HAVE_LDREX ? 4 : 0)					\
| (TARGET_HAVE_LDREXBH ? 3 : 0)				\
| (TARGET_HAVE_LDREXD ? 8 : 0))
#define TARGET_ARM_FP			\
(!TARGET_SOFT_FLOAT ? (TARGET_VFP_SINGLE ? 4		\
: (TARGET_VFP_DOUBLE ? (TARGET_FP16 ? 14 : 12) : 0)) \
: 0)
#define TARGET_NEON_FP				 \
(TARGET_NEON ? (TARGET_ARM_FP & (0xff ^ 0x08)) \
: 0)
#define FPUTYPE_AUTO "auto"
#define MAX_LDM_STM_OPS 4
extern const char *arm_rewrite_mcpu (int argc, const char **argv);
extern const char *arm_rewrite_march (int argc, const char **argv);
extern const char *arm_asm_auto_mfpu (int argc, const char **argv);
#define ASM_CPU_SPEC_FUNCTIONS			\
{ "rewrite_mcpu", arm_rewrite_mcpu },	\
{ "rewrite_march", arm_rewrite_march },	\
{ "asm_auto_mfpu", arm_asm_auto_mfpu },
#define ASM_CPU_SPEC							\
" %{mfpu=auto:%<mfpu=auto %:asm_auto_mfpu(%{march=*: arch %*})}"	\
" %{mcpu=generic-*:-march=%:rewrite_march(%{mcpu=generic-*:%*});"	\
"   march=*:-march=%:rewrite_march(%{march=*:%*});"			\
"   mcpu=*:-mcpu=%:rewrite_mcpu(%{mcpu=*:%*})"			\
" }"
extern const char *arm_target_thumb_only (int argc, const char **argv);
#define TARGET_MODE_SPEC_FUNCTIONS			\
{ "target_mode_check", arm_target_thumb_only },
#if defined(__arm__)
extern const char *host_detect_local_cpu (int argc, const char **argv);
#define HAVE_LOCAL_CPU_DETECT
# define MCPU_MTUNE_NATIVE_FUNCTIONS			\
{ "local_cpu_detect", host_detect_local_cpu },
# define MCPU_MTUNE_NATIVE_SPECS				\
" %{march=native:%<march=native %:local_cpu_detect(arch)}"	\
" %{mcpu=native:%<mcpu=native %:local_cpu_detect(cpu)}"	\
" %{mtune=native:%<mtune=native %:local_cpu_detect(tune)}"
#else
# define MCPU_MTUNE_NATIVE_FUNCTIONS
# define MCPU_MTUNE_NATIVE_SPECS ""
#endif
const char *arm_canon_arch_option (int argc, const char **argv);
#define CANON_ARCH_SPEC_FUNCTION		\
{ "canon_arch", arm_canon_arch_option },
const char *arm_be8_option (int argc, const char **argv);
#define BE8_SPEC_FUNCTION			\
{ "be8_linkopt", arm_be8_option },
# define EXTRA_SPEC_FUNCTIONS			\
MCPU_MTUNE_NATIVE_FUNCTIONS			\
ASM_CPU_SPEC_FUNCTIONS			\
CANON_ARCH_SPEC_FUNCTION			\
TARGET_MODE_SPEC_FUNCTIONS			\
BE8_SPEC_FUNCTION
#define TARGET_MODE_SPECS						\
" %{!marm:%{!mthumb:%:target_mode_check(%{march=*:arch %*;mcpu=*:cpu %*;:})}}"
#define ARCH_CANONICAL_SPECS				\
" -march=%:canon_arch(%{mcpu=*: cpu %*} "		\
"                     %{march=*: arch %*} "		\
"                     %{mfpu=*: fpu %*} "		\
"                     %{mfloat-abi=*: abi %*}"	\
"                     %<march=*) "
#define DRIVER_SELF_SPECS			\
MCPU_MTUNE_NATIVE_SPECS,			\
TARGET_MODE_SPECS,				\
ARCH_CANONICAL_SPECS
#define TARGET_SUPPORTS_WIDE_INT 1
#define SWITCHABLE_TARGET 1
#define SECTION_ARM_PURECODE SECTION_MACH_DEP
#endif 
