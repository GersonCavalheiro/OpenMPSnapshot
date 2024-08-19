#ifndef RS6000_OPTS_H
#include "config/powerpcspe/powerpcspe-opts.h"
#endif
#define OBJECT_XCOFF 1
#define OBJECT_ELF 2
#define OBJECT_PEF 3
#define OBJECT_MACHO 4
#define TARGET_ELF (TARGET_OBJECT_FORMAT == OBJECT_ELF)
#define TARGET_XCOFF (TARGET_OBJECT_FORMAT == OBJECT_XCOFF)
#define TARGET_MACOS (TARGET_OBJECT_FORMAT == OBJECT_PEF)
#define TARGET_MACHO (TARGET_OBJECT_FORMAT == OBJECT_MACHO)
#ifndef TARGET_AIX
#define TARGET_AIX 0
#endif
#ifndef TARGET_AIX_OS
#define TARGET_AIX_OS 0
#endif
#define DOT_SYMBOLS 1
#ifndef TARGET_CPU_DEFAULT
#define TARGET_CPU_DEFAULT ((char *)0)
#endif
#ifdef CONFIG_PPC405CR
#define PPC405_ERRATUM77 (rs6000_cpu == PROCESSOR_PPC405)
#else
#define PPC405_ERRATUM77 0
#endif
#ifndef TARGET_PAIRED_FLOAT
#define TARGET_PAIRED_FLOAT 0
#endif
#ifdef HAVE_AS_POPCNTB
#define ASM_CPU_POWER5_SPEC "-mpower5"
#else
#define ASM_CPU_POWER5_SPEC "-mpower4"
#endif
#ifdef HAVE_AS_DFP
#define ASM_CPU_POWER6_SPEC "-mpower6 -maltivec"
#else
#define ASM_CPU_POWER6_SPEC "-mpower4 -maltivec"
#endif
#ifdef HAVE_AS_POPCNTD
#define ASM_CPU_POWER7_SPEC "-mpower7"
#else
#define ASM_CPU_POWER7_SPEC "-mpower4 -maltivec"
#endif
#ifdef HAVE_AS_POWER8
#define ASM_CPU_POWER8_SPEC "-mpower8"
#else
#define ASM_CPU_POWER8_SPEC ASM_CPU_POWER7_SPEC
#endif
#ifdef HAVE_AS_POWER9
#define ASM_CPU_POWER9_SPEC "-mpower9"
#else
#define ASM_CPU_POWER9_SPEC ASM_CPU_POWER8_SPEC
#endif
#ifdef HAVE_AS_DCI
#define ASM_CPU_476_SPEC "-m476"
#else
#define ASM_CPU_476_SPEC "-mpower4"
#endif
#define ASM_CPU_SPEC \
"%{!mcpu*: \
%{mpowerpc64*: -mppc64} \
%{!mpowerpc64*: %(asm_default)}} \
%{mcpu=native: %(asm_cpu_native)} \
%{mcpu=cell: -mcell} \
%{mcpu=power3: -mppc64} \
%{mcpu=power4: -mpower4} \
%{mcpu=power5: %(asm_cpu_power5)} \
%{mcpu=power5+: %(asm_cpu_power5)} \
%{mcpu=power6: %(asm_cpu_power6) -maltivec} \
%{mcpu=power6x: %(asm_cpu_power6) -maltivec} \
%{mcpu=power7: %(asm_cpu_power7)} \
%{mcpu=power8: %(asm_cpu_power8)} \
%{mcpu=power9: %(asm_cpu_power9)} \
%{mcpu=a2: -ma2} \
%{mcpu=powerpc: -mppc} \
%{mcpu=powerpc64le: %(asm_cpu_power8)} \
%{mcpu=rs64a: -mppc64} \
%{mcpu=401: -mppc} \
%{mcpu=403: -m403} \
%{mcpu=405: -m405} \
%{mcpu=405fp: -m405} \
%{mcpu=440: -m440} \
%{mcpu=440fp: -m440} \
%{mcpu=464: -m440} \
%{mcpu=464fp: -m440} \
%{mcpu=476: %(asm_cpu_476)} \
%{mcpu=476fp: %(asm_cpu_476)} \
%{mcpu=505: -mppc} \
%{mcpu=601: -m601} \
%{mcpu=602: -mppc} \
%{mcpu=603: -mppc} \
%{mcpu=603e: -mppc} \
%{mcpu=ec603e: -mppc} \
%{mcpu=604: -mppc} \
%{mcpu=604e: -mppc} \
%{mcpu=620: -mppc64} \
%{mcpu=630: -mppc64} \
%{mcpu=740: -mppc} \
%{mcpu=750: -mppc} \
%{mcpu=G3: -mppc} \
%{mcpu=7400: -mppc -maltivec} \
%{mcpu=7450: -mppc -maltivec} \
%{mcpu=G4: -mppc -maltivec} \
%{mcpu=801: -mppc} \
%{mcpu=821: -mppc} \
%{mcpu=823: -mppc} \
%{mcpu=860: -mppc} \
%{mcpu=970: -mpower4 -maltivec} \
%{mcpu=G5: -mpower4 -maltivec} \
%{mcpu=8540: -me500} \
%{mcpu=8548: -me500} \
%{mcpu=e300c2: -me300} \
%{mcpu=e300c3: -me300} \
%{mcpu=e500mc: -me500mc} \
%{mcpu=e500mc64: -me500mc64} \
%{mcpu=e5500: -me5500} \
%{mcpu=e6500: -me6500} \
%{maltivec: -maltivec} \
%{mvsx: -mvsx %{!maltivec: -maltivec} %{!mcpu*: %(asm_cpu_power7)}} \
%{mpower8-vector|mcrypto|mdirect-move|mhtm: %{!mcpu*: %(asm_cpu_power8)}} \
-many"
#define CPP_DEFAULT_SPEC ""
#define ASM_DEFAULT_SPEC ""
#define SUBTARGET_EXTRA_SPECS
#define EXTRA_SPECS							\
{ "cpp_default",		CPP_DEFAULT_SPEC },			\
{ "asm_cpu",			ASM_CPU_SPEC },				\
{ "asm_cpu_native",		ASM_CPU_NATIVE_SPEC },			\
{ "asm_default",		ASM_DEFAULT_SPEC },			\
{ "cc1_cpu",			CC1_CPU_SPEC },				\
{ "asm_cpu_power5",		ASM_CPU_POWER5_SPEC },			\
{ "asm_cpu_power6",		ASM_CPU_POWER6_SPEC },			\
{ "asm_cpu_power7",		ASM_CPU_POWER7_SPEC },			\
{ "asm_cpu_power8",		ASM_CPU_POWER8_SPEC },			\
{ "asm_cpu_power9",		ASM_CPU_POWER9_SPEC },			\
{ "asm_cpu_476",		ASM_CPU_476_SPEC },			\
SUBTARGET_EXTRA_SPECS
#if defined(__powerpc__) || defined(__POWERPC__) || defined(_AIX)
extern const char *host_detect_local_cpu (int argc, const char **argv);
#define EXTRA_SPEC_FUNCTIONS \
{ "local_cpu_detect", host_detect_local_cpu },
#define HAVE_LOCAL_CPU_DETECT
#define ASM_CPU_NATIVE_SPEC "%:local_cpu_detect(asm)"
#else
#define ASM_CPU_NATIVE_SPEC "%(asm_default)"
#endif
#ifndef CC1_CPU_SPEC
#ifdef HAVE_LOCAL_CPU_DETECT
#define CC1_CPU_SPEC \
"%{mcpu=native:%<mcpu=native %:local_cpu_detect(cpu)} \
%{mtune=native:%<mtune=native %:local_cpu_detect(tune)}"
#else
#define CC1_CPU_SPEC ""
#endif
#endif
#ifndef HAVE_AS_MFCRF
#undef  TARGET_MFCRF
#define TARGET_MFCRF 0
#endif
#ifndef HAVE_AS_POPCNTB
#undef  TARGET_POPCNTB
#define TARGET_POPCNTB 0
#endif
#ifndef HAVE_AS_FPRND
#undef  TARGET_FPRND
#define TARGET_FPRND 0
#endif
#ifndef HAVE_AS_CMPB
#undef  TARGET_CMPB
#define TARGET_CMPB 0
#endif
#ifndef HAVE_AS_MFPGPR
#undef  TARGET_MFPGPR
#define TARGET_MFPGPR 0
#endif
#ifndef HAVE_AS_DFP
#undef  TARGET_DFP
#define TARGET_DFP 0
#endif
#ifndef HAVE_AS_POPCNTD
#undef  TARGET_POPCNTD
#define TARGET_POPCNTD 0
#endif
#ifndef HAVE_AS_POWER8
#undef  TARGET_DIRECT_MOVE
#undef  TARGET_CRYPTO
#undef  TARGET_HTM
#undef  TARGET_P8_VECTOR
#define TARGET_DIRECT_MOVE 0
#define TARGET_CRYPTO 0
#define TARGET_HTM 0
#define TARGET_P8_VECTOR 0
#endif
#ifndef HAVE_AS_POWER9
#undef  TARGET_FLOAT128_HW
#undef  TARGET_MODULO
#undef  TARGET_P9_VECTOR
#undef  TARGET_P9_MINMAX
#undef  TARGET_P9_DFORM_SCALAR
#undef  TARGET_P9_DFORM_VECTOR
#undef  TARGET_P9_MISC
#define TARGET_FLOAT128_HW 0
#define TARGET_MODULO 0
#define TARGET_P9_VECTOR 0
#define TARGET_P9_MINMAX 0
#define TARGET_P9_DFORM_SCALAR 0
#define TARGET_P9_DFORM_VECTOR 0
#define TARGET_P9_MISC 0
#endif
#ifdef HAVE_AS_LWSYNC
#define TARGET_LWSYNC_INSTRUCTION 1
#else
#define TARGET_LWSYNC_INSTRUCTION 0
#endif
#ifndef HAVE_AS_TLS_MARKERS
#undef  TARGET_TLS_MARKERS
#define TARGET_TLS_MARKERS 0
#else
#define TARGET_TLS_MARKERS tls_markers
#endif
#ifndef TARGET_SECURE_PLT
#define TARGET_SECURE_PLT 0
#endif
#ifndef TARGET_CMODEL
#define TARGET_CMODEL CMODEL_SMALL
#endif
#define TARGET_32BIT		(! TARGET_64BIT)
#ifndef HAVE_AS_TLS
#define HAVE_AS_TLS 0
#endif
#ifndef TARGET_LINK_STACK
#define TARGET_LINK_STACK 0
#endif
#ifndef SET_TARGET_LINK_STACK
#define SET_TARGET_LINK_STACK(X) do { } while (0)
#endif
#ifndef TARGET_FLOAT128_ENABLE_TYPE
#define TARGET_FLOAT128_ENABLE_TYPE 0
#endif
#define RS6000_SYMBOL_REF_TLS_P(RTX) \
(GET_CODE (RTX) == SYMBOL_REF && SYMBOL_REF_TLS_MODEL (RTX) != 0)
#ifdef IN_LIBGCC2
#if defined (__64BIT__) || defined (__powerpc64__) || defined (__ppc64__)
#undef TARGET_POWERPC64
#define TARGET_POWERPC64	1
#else
#undef TARGET_POWERPC64
#define TARGET_POWERPC64	0
#endif
#else
#endif
#define TARGET_DEFAULT (MASK_MULTIPLE | MASK_STRING)
#define TARGET_SINGLE_FLOAT 1
#define TARGET_DOUBLE_FLOAT 1
#define TARGET_SINGLE_FPU   0
#define TARGET_SIMPLE_FPU   0
#define TARGET_XILINX_FPU   0
#define rs6000_cpu_attr ((enum attr_cpu)rs6000_cpu)
#define PROCESSOR_COMMON    PROCESSOR_PPC601
#define PROCESSOR_POWERPC   PROCESSOR_PPC604
#define PROCESSOR_POWERPC64 PROCESSOR_RS64A
#define PROCESSOR_DEFAULT   PROCESSOR_PPC603
#define PROCESSOR_DEFAULT64 PROCESSOR_RS64A
#define ASSEMBLER_DIALECT 1
#define MASK_DEBUG_STACK	0x01	
#define	MASK_DEBUG_ARG		0x02	
#define MASK_DEBUG_REG		0x04	
#define MASK_DEBUG_ADDR		0x08	
#define MASK_DEBUG_COST		0x10	
#define MASK_DEBUG_TARGET	0x20	
#define MASK_DEBUG_BUILTIN	0x40	
#define MASK_DEBUG_ALL		(MASK_DEBUG_STACK \
| MASK_DEBUG_ARG \
| MASK_DEBUG_REG \
| MASK_DEBUG_ADDR \
| MASK_DEBUG_COST \
| MASK_DEBUG_TARGET \
| MASK_DEBUG_BUILTIN)
#define	TARGET_DEBUG_STACK	(rs6000_debug & MASK_DEBUG_STACK)
#define	TARGET_DEBUG_ARG	(rs6000_debug & MASK_DEBUG_ARG)
#define TARGET_DEBUG_REG	(rs6000_debug & MASK_DEBUG_REG)
#define TARGET_DEBUG_ADDR	(rs6000_debug & MASK_DEBUG_ADDR)
#define TARGET_DEBUG_COST	(rs6000_debug & MASK_DEBUG_COST)
#define TARGET_DEBUG_TARGET	(rs6000_debug & MASK_DEBUG_TARGET)
#define TARGET_DEBUG_BUILTIN	(rs6000_debug & MASK_DEBUG_BUILTIN)
#define FLOAT128_IEEE_P(MODE)						\
((TARGET_IEEEQUAD && ((MODE) == TFmode || (MODE) == TCmode))		\
|| ((MODE) == KFmode) || ((MODE) == KCmode))
#define FLOAT128_IBM_P(MODE)						\
((!TARGET_IEEEQUAD && ((MODE) == TFmode || (MODE) == TCmode))		\
|| (TARGET_HARD_FLOAT && TARGET_FPRS					\
&& ((MODE) == IFmode || (MODE) == ICmode)))
#define FLOAT128_VECTOR_P(MODE) (TARGET_FLOAT128_TYPE && FLOAT128_IEEE_P (MODE))
#define FLOAT128_2REG_P(MODE)						\
(FLOAT128_IBM_P (MODE)						\
|| ((MODE) == TDmode)						\
|| (!TARGET_FLOAT128_TYPE && FLOAT128_IEEE_P (MODE)))
#define SCALAR_FLOAT_MODE_NOT_VECTOR_P(MODE)				\
(SCALAR_FLOAT_MODE_P (MODE) && !FLOAT128_VECTOR_P (MODE))
extern enum rs6000_vector rs6000_vector_unit[];
#define VECTOR_UNIT_NONE_P(MODE)			\
(rs6000_vector_unit[(MODE)] == VECTOR_NONE)
#define VECTOR_UNIT_VSX_P(MODE)				\
(rs6000_vector_unit[(MODE)] == VECTOR_VSX)
#define VECTOR_UNIT_P8_VECTOR_P(MODE)			\
(rs6000_vector_unit[(MODE)] == VECTOR_P8_VECTOR)
#define VECTOR_UNIT_ALTIVEC_P(MODE)			\
(rs6000_vector_unit[(MODE)] == VECTOR_ALTIVEC)
#define VECTOR_UNIT_VSX_OR_P8_VECTOR_P(MODE)		\
(IN_RANGE ((int)rs6000_vector_unit[(MODE)],		\
(int)VECTOR_VSX,				\
(int)VECTOR_P8_VECTOR))
#define VECTOR_UNIT_ALTIVEC_OR_VSX_P(MODE)		\
(IN_RANGE ((int)rs6000_vector_unit[(MODE)],		\
(int)VECTOR_ALTIVEC,			\
(int)VECTOR_P8_VECTOR))
extern enum rs6000_vector rs6000_vector_mem[];
#define VECTOR_MEM_NONE_P(MODE)				\
(rs6000_vector_mem[(MODE)] == VECTOR_NONE)
#define VECTOR_MEM_VSX_P(MODE)				\
(rs6000_vector_mem[(MODE)] == VECTOR_VSX)
#define VECTOR_MEM_P8_VECTOR_P(MODE)			\
(rs6000_vector_mem[(MODE)] == VECTOR_VSX)
#define VECTOR_MEM_ALTIVEC_P(MODE)			\
(rs6000_vector_mem[(MODE)] == VECTOR_ALTIVEC)
#define VECTOR_MEM_VSX_OR_P8_VECTOR_P(MODE)		\
(IN_RANGE ((int)rs6000_vector_mem[(MODE)],		\
(int)VECTOR_VSX,				\
(int)VECTOR_P8_VECTOR))
#define VECTOR_MEM_ALTIVEC_OR_VSX_P(MODE)		\
(IN_RANGE ((int)rs6000_vector_mem[(MODE)],		\
(int)VECTOR_ALTIVEC,			\
(int)VECTOR_P8_VECTOR))
extern int rs6000_vector_align[];
#define VECTOR_ALIGN(MODE)						\
((rs6000_vector_align[(MODE)] != 0)					\
? rs6000_vector_align[(MODE)]					\
: (int)GET_MODE_BITSIZE ((MODE)))
#define VECTOR_ELT_ORDER_BIG                                  \
(BYTES_BIG_ENDIAN || (rs6000_altivec_element_order == 2))
#define VECTOR_ELEMENT_SCALAR_64BIT	((BYTES_BIG_ENDIAN) ? 0 : 1)
#define VECTOR_ELEMENT_MFVSRLD_64BIT	((BYTES_BIG_ENDIAN) ? 1 : 0)
#ifndef IN_TARGET_LIBS
#define MASK_ALIGN_POWER   0x00000000
#define MASK_ALIGN_NATURAL 0x00000001
#define TARGET_ALIGN_NATURAL (rs6000_alignment_flags & MASK_ALIGN_NATURAL)
#else
#define TARGET_ALIGN_NATURAL 0
#endif
#define TARGET_LONG_DOUBLE_128 (rs6000_long_double_type_size == 128)
#define TARGET_IEEEQUAD rs6000_ieeequad
#define TARGET_ALTIVEC_ABI rs6000_altivec_abi
#define TARGET_LDBRX (TARGET_POPCNTD || rs6000_cpu == PROCESSOR_CELL)
#define TARGET_SPE_ABI 0
#define TARGET_SPE 0
#define TARGET_ISEL64 (TARGET_ISEL && TARGET_POWERPC64)
#define TARGET_FPRS 1
#define TARGET_E500_SINGLE 0
#define TARGET_E500_DOUBLE 0
#define CHECK_E500_OPTIONS do { } while (0)
#define TARGET_FCFID	(TARGET_POWERPC64				\
|| TARGET_PPC_GPOPT		\
|| TARGET_POPCNTB			\
|| TARGET_CMPB				\
|| TARGET_POPCNTD			\
|| TARGET_XILINX_FPU)
#define TARGET_FCTIDZ	TARGET_FCFID
#define TARGET_STFIWX	TARGET_PPC_GFXOPT
#define TARGET_LFIWAX	TARGET_CMPB
#define TARGET_LFIWZX	TARGET_POPCNTD
#define TARGET_FCFIDS	TARGET_POPCNTD
#define TARGET_FCFIDU	TARGET_POPCNTD
#define TARGET_FCFIDUS	TARGET_POPCNTD
#define TARGET_FCTIDUZ	TARGET_POPCNTD
#define TARGET_FCTIWUZ	TARGET_POPCNTD
#define TARGET_CTZ	TARGET_MODULO
#define TARGET_EXTSWSLI	(TARGET_MODULO && TARGET_POWERPC64)
#define TARGET_MADDLD	(TARGET_MODULO && TARGET_POWERPC64)
#define TARGET_XSCVDPSPN	(TARGET_DIRECT_MOVE || TARGET_P8_VECTOR)
#define TARGET_XSCVSPDPN	(TARGET_DIRECT_MOVE || TARGET_P8_VECTOR)
#define TARGET_VADDUQM		(TARGET_P8_VECTOR && TARGET_POWERPC64)
#define TARGET_DIRECT_MOVE_128	(TARGET_P9_VECTOR && TARGET_DIRECT_MOVE \
&& TARGET_POWERPC64)
#define TARGET_VEXTRACTUB	(TARGET_P9_VECTOR && TARGET_DIRECT_MOVE \
&& TARGET_UPPER_REGS_DI && TARGET_POWERPC64)
#define TARGET_NO_SF_SUBREG	TARGET_DIRECT_MOVE_64BIT
#define TARGET_ALLOW_SF_SUBREG	(!TARGET_DIRECT_MOVE_64BIT)
#define TARGET_EFFICIENT_OVERLAPPING_UNALIGNED TARGET_EFFICIENT_UNALIGNED_VSX
#define TARGET_SYNC_HI_QI	(TARGET_QUAD_MEMORY			\
|| TARGET_QUAD_MEMORY_ATOMIC		\
|| TARGET_DIRECT_MOVE)
#define TARGET_SYNC_TI		TARGET_QUAD_MEMORY_ATOMIC
#define TARGET_NO_SDMODE_STACK	(TARGET_LFIWZX && TARGET_STFIWX && TARGET_DFP)
#define TARGET_MINMAX_SF	(TARGET_SF_FPR && TARGET_PPC_GFXOPT	\
&& (TARGET_P9_MINMAX || !flag_trapping_math))
#define TARGET_MINMAX_DF	(TARGET_DF_FPR && TARGET_PPC_GFXOPT	\
&& (TARGET_P9_MINMAX || !flag_trapping_math))
#define MASK_ALTIVEC			OPTION_MASK_ALTIVEC
#define MASK_CMPB			OPTION_MASK_CMPB
#define MASK_CRYPTO			OPTION_MASK_CRYPTO
#define MASK_DFP			OPTION_MASK_DFP
#define MASK_DIRECT_MOVE		OPTION_MASK_DIRECT_MOVE
#define MASK_DLMZB			OPTION_MASK_DLMZB
#define MASK_EABI			OPTION_MASK_EABI
#define MASK_FLOAT128_TYPE		OPTION_MASK_FLOAT128_TYPE
#define MASK_FPRND			OPTION_MASK_FPRND
#define MASK_P8_FUSION			OPTION_MASK_P8_FUSION
#define MASK_HARD_FLOAT			OPTION_MASK_HARD_FLOAT
#define MASK_HTM			OPTION_MASK_HTM
#define MASK_ISEL			OPTION_MASK_ISEL
#define MASK_MFCRF			OPTION_MASK_MFCRF
#define MASK_MFPGPR			OPTION_MASK_MFPGPR
#define MASK_MULHW			OPTION_MASK_MULHW
#define MASK_MULTIPLE			OPTION_MASK_MULTIPLE
#define MASK_NO_UPDATE			OPTION_MASK_NO_UPDATE
#define MASK_P8_VECTOR			OPTION_MASK_P8_VECTOR
#define MASK_P9_VECTOR			OPTION_MASK_P9_VECTOR
#define MASK_P9_MISC			OPTION_MASK_P9_MISC
#define MASK_POPCNTB			OPTION_MASK_POPCNTB
#define MASK_POPCNTD			OPTION_MASK_POPCNTD
#define MASK_PPC_GFXOPT			OPTION_MASK_PPC_GFXOPT
#define MASK_PPC_GPOPT			OPTION_MASK_PPC_GPOPT
#define MASK_RECIP_PRECISION		OPTION_MASK_RECIP_PRECISION
#define MASK_SOFT_FLOAT			OPTION_MASK_SOFT_FLOAT
#define MASK_STRICT_ALIGN		OPTION_MASK_STRICT_ALIGN
#define MASK_STRING			OPTION_MASK_STRING
#define MASK_UPDATE			OPTION_MASK_UPDATE
#define MASK_VSX			OPTION_MASK_VSX
#define MASK_VSX_TIMODE			OPTION_MASK_VSX_TIMODE
#ifndef IN_LIBGCC2
#define MASK_POWERPC64			OPTION_MASK_POWERPC64
#endif
#ifdef TARGET_64BIT
#define MASK_64BIT			OPTION_MASK_64BIT
#endif
#ifdef TARGET_LITTLE_ENDIAN
#define MASK_LITTLE_ENDIAN		OPTION_MASK_LITTLE_ENDIAN
#endif
#ifdef TARGET_REGNAMES
#define MASK_REGNAMES			OPTION_MASK_REGNAMES
#endif
#ifdef TARGET_PROTOTYPE
#define MASK_PROTOTYPE			OPTION_MASK_PROTOTYPE
#endif
#ifdef TARGET_MODULO
#define RS6000_BTM_MODULO		OPTION_MASK_MODULO
#endif
#define TARGET_EXTRA_BUILTINS	(!TARGET_SPE && !TARGET_PAIRED_FLOAT	 \
&& ((TARGET_POWERPC64			 \
|| TARGET_PPC_GPOPT  \
|| TARGET_POPCNTB	   \
|| TARGET_CMPB	   \
|| TARGET_POPCNTD	   \
|| TARGET_ALTIVEC			 \
|| TARGET_VSX			 \
|| TARGET_HARD_FLOAT)))
#define TARGET_NO_LWSYNC (rs6000_cpu == PROCESSOR_PPC8540 \
|| rs6000_cpu == PROCESSOR_PPC8548)
#define TARGET_SF_SPE	(TARGET_HARD_FLOAT && TARGET_SINGLE_FLOAT	\
&& !TARGET_FPRS)
#define TARGET_DF_SPE	(TARGET_HARD_FLOAT && TARGET_DOUBLE_FLOAT	\
&& !TARGET_FPRS && TARGET_E500_DOUBLE)
#define TARGET_SF_FPR	(TARGET_HARD_FLOAT && TARGET_FPRS		\
&& TARGET_SINGLE_FLOAT)
#define TARGET_DF_FPR	(TARGET_HARD_FLOAT && TARGET_FPRS		\
&& TARGET_DOUBLE_FLOAT)
#define TARGET_SF_INSN	(TARGET_SF_FPR || TARGET_SF_SPE)
#define TARGET_DF_INSN	(TARGET_DF_FPR || TARGET_DF_SPE)
#define TARGET_FRES	(TARGET_HARD_FLOAT && TARGET_PPC_GFXOPT \
&& TARGET_FPRS && TARGET_SINGLE_FLOAT)
#define TARGET_FRE	(TARGET_HARD_FLOAT && TARGET_FPRS \
&& TARGET_DOUBLE_FLOAT \
&& (TARGET_POPCNTB || VECTOR_UNIT_VSX_P (DFmode)))
#define TARGET_FRSQRTES	(TARGET_HARD_FLOAT && TARGET_POPCNTB \
&& TARGET_PPC_GFXOPT && TARGET_FPRS \
&& TARGET_SINGLE_FLOAT)
#define TARGET_FRSQRTE	(TARGET_HARD_FLOAT && TARGET_FPRS \
&& TARGET_DOUBLE_FLOAT \
&& (TARGET_PPC_GFXOPT || VECTOR_UNIT_VSX_P (DFmode)))
#define TARGET_TOC_FUSION_INT	(TARGET_P8_FUSION			\
&& TARGET_TOC_FUSION			\
&& (TARGET_CMODEL != CMODEL_SMALL)	\
&& TARGET_POWERPC64)
#define TARGET_TOC_FUSION_FP	(TARGET_P9_FUSION			\
&& TARGET_TOC_FUSION			\
&& (TARGET_CMODEL != CMODEL_SMALL)	\
&& TARGET_POWERPC64			\
&& TARGET_HARD_FLOAT			\
&& TARGET_FPRS				\
&& TARGET_SINGLE_FLOAT			\
&& TARGET_DOUBLE_FLOAT)
#define TARGET_DIRECT_MOVE_64BIT	(TARGET_DIRECT_MOVE		\
&& TARGET_P8_VECTOR		\
&& TARGET_POWERPC64		\
&& TARGET_UPPER_REGS_DI	\
&& (rs6000_altivec_element_order != 2))
#define RS6000_RECIP_MASK_HAVE_RE	0x1	
#define RS6000_RECIP_MASK_AUTO_RE	0x2	
#define RS6000_RECIP_MASK_HAVE_RSQRTE	0x4	
#define RS6000_RECIP_MASK_AUTO_RSQRTE	0x8	
extern unsigned char rs6000_recip_bits[];
#define RS6000_RECIP_HAVE_RE_P(MODE) \
(rs6000_recip_bits[(int)(MODE)] & RS6000_RECIP_MASK_HAVE_RE)
#define RS6000_RECIP_AUTO_RE_P(MODE) \
(rs6000_recip_bits[(int)(MODE)] & RS6000_RECIP_MASK_AUTO_RE)
#define RS6000_RECIP_HAVE_RSQRTE_P(MODE) \
(rs6000_recip_bits[(int)(MODE)] & RS6000_RECIP_MASK_HAVE_RSQRTE)
#define RS6000_RECIP_AUTO_RSQRTE_P(MODE) \
(rs6000_recip_bits[(int)(MODE)] & RS6000_RECIP_MASK_AUTO_RSQRTE)
#define OPTION_TARGET_CPU_DEFAULT TARGET_CPU_DEFAULT
#define REGISTER_TARGET_PRAGMAS() do {				\
c_register_pragma (0, "longcall", rs6000_pragma_longcall);	\
targetm.target_option.pragma_parse = rs6000_pragma_target_parse; \
targetm.resolve_overloaded_builtin = altivec_resolve_overloaded_builtin; \
rs6000_target_modify_macros_ptr = rs6000_target_modify_macros; \
} while (0)
#define TARGET_CPU_CPP_BUILTINS() \
rs6000_cpu_cpp_builtins (pfile)
#define RS6000_CPU_CPP_ENDIAN_BUILTINS()	\
do						\
{						\
if (BYTES_BIG_ENDIAN)			\
{					\
builtin_define ("__BIG_ENDIAN__");	\
builtin_define ("_BIG_ENDIAN");	\
builtin_assert ("machine=bigendian");	\
}					\
else					\
{					\
builtin_define ("__LITTLE_ENDIAN__");	\
builtin_define ("_LITTLE_ENDIAN");	\
builtin_assert ("machine=littleendian"); \
}					\
}						\
while (0)

#define PROMOTE_MODE(MODE,UNSIGNEDP,TYPE)	\
if (GET_MODE_CLASS (MODE) == MODE_INT		\
&& GET_MODE_SIZE (MODE) < (TARGET_32BIT ? 4 : 8)) \
(MODE) = TARGET_32BIT ? SImode : DImode;
#define BITS_BIG_ENDIAN 1
#define BYTES_BIG_ENDIAN 1
#define WORDS_BIG_ENDIAN 1
#define LONG_DOUBLE_LARGE_FIRST 1
#define MAX_BITS_PER_WORD 64
#define UNITS_PER_WORD (! TARGET_POWERPC64 ? 4 : 8)
#ifdef IN_LIBGCC2
#define MIN_UNITS_PER_WORD UNITS_PER_WORD
#else
#define MIN_UNITS_PER_WORD 4
#endif
#define UNITS_PER_FP_WORD 8
#define UNITS_PER_ALTIVEC_WORD 16
#define UNITS_PER_VSX_WORD 16
#define UNITS_PER_SPE_WORD 8
#define UNITS_PER_PAIRED_WORD 8
#define PTRDIFF_TYPE "int"
#define SIZE_TYPE "long unsigned int"
#define WCHAR_TYPE "short unsigned int"
#define WCHAR_TYPE_SIZE 16
#define SHORT_TYPE_SIZE 16
#define INT_TYPE_SIZE 32
#define LONG_TYPE_SIZE (TARGET_32BIT ? 32 : 64)
#define LONG_LONG_TYPE_SIZE 64
#define FLOAT_TYPE_SIZE 32
#define DOUBLE_TYPE_SIZE 64
#define LONG_DOUBLE_TYPE_SIZE rs6000_long_double_type_size
#define WIDEST_HARDWARE_FP_SIZE 64
extern unsigned rs6000_pointer_size;
#define POINTER_SIZE rs6000_pointer_size
#define PARM_BOUNDARY (TARGET_32BIT ? 32 : 64)
#define STACK_BOUNDARY	\
((TARGET_32BIT && !TARGET_ALTIVEC && !TARGET_ALTIVEC_ABI && !TARGET_VSX) \
? 64 : 128)
#define FUNCTION_BOUNDARY 32
#define BIGGEST_ALIGNMENT 128
#define EMPTY_FIELD_BOUNDARY 32
#define STRUCTURE_SIZE_BOUNDARY 8
#define PCC_BITFIELD_TYPE_MATTERS 1
enum data_align { align_abi, align_opt, align_both };
#define LOCAL_ALIGNMENT(TYPE, ALIGN)				\
rs6000_data_alignment (TYPE, ALIGN, align_both)
#define DATA_ALIGNMENT(TYPE, ALIGN) \
rs6000_data_alignment (TYPE, ALIGN, align_opt)
#define DATA_ABI_ALIGNMENT(TYPE, ALIGN) \
rs6000_data_alignment (TYPE, ALIGN, align_abi)
#define STRICT_ALIGNMENT 0

#define FIRST_PSEUDO_REGISTER 149
#define PRE_GCC3_DWARF_FRAME_REGISTERS 77
#define SPE_HIGH_REGNO_P(N) \
((N) >= FIRST_SPE_HIGH_REGNO && (N) <= LAST_SPE_HIGH_REGNO)
#define DWARF_FRAME_REGISTERS (FIRST_PSEUDO_REGISTER - 4)
#define DWARF_REG_TO_UNWIND_COLUMN(r) \
((r) >= 1200 ? ((r) - 1200 + (DWARF_FRAME_REGISTERS - 32)) : (r))
#define DBX_REGISTER_NUMBER(REGNO) rs6000_dbx_register_number ((REGNO), 0)
#define DWARF_FRAME_REGNUM(REGNO) (REGNO)
#define DWARF2_FRAME_REG_OUT(REGNO, FOR_EH) \
rs6000_dbx_register_number ((REGNO), (FOR_EH)? 2 : 1)
#define FIXED_REGISTERS  \
{0, 1, FIXED_R2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FIXED_R13, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,	   \
\
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
1, 1						   \
, 1, 1, 1, 1, 1, 1,				   \
\
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  \
}
#define CALL_USED_REGISTERS  \
{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, FIXED_R13, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,	   \
\
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
1, 1						   \
, 1, 1, 1, 1, 1, 1,				   \
\
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  \
}
#define CALL_REALLY_USED_REGISTERS  \
{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, FIXED_R13, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,	   \
\
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0						   \
, 0, 0, 0, 0, 0, 0,				   \
\
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  \
}
#define TOTAL_ALTIVEC_REGS	(LAST_ALTIVEC_REGNO - FIRST_ALTIVEC_REGNO + 1)
#define FIRST_SAVED_ALTIVEC_REGNO (FIRST_ALTIVEC_REGNO+20)
#define FIRST_SAVED_FP_REGNO	  (14+32)
#define FIRST_SAVED_GP_REGNO	  (FIXED_R13 ? 14 : 13)
#if FIXED_R2 == 1
#define MAYBE_R2_AVAILABLE
#define MAYBE_R2_FIXED 2,
#else
#define MAYBE_R2_AVAILABLE 2,
#define MAYBE_R2_FIXED
#endif
#if FIXED_R13 == 1
#define EARLY_R12 12,
#define LATE_R12
#else
#define EARLY_R12
#define LATE_R12 12,
#endif
#define REG_ALLOC_ORDER						\
{32,								\
\
\
44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 45,		\
33,								\
63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51,		\
50, 49, 48, 47, 46,						\
75, 73, 74, 69, 68, 72, 71, 70,				\
MAYBE_R2_AVAILABLE						\
9, 10, 8, 7, 6, 5, 4,					\
3, EARLY_R12 11, 0,						\
31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,		\
18, 17, 16, 15, 14, 13, LATE_R12				\
66, 65,							\
1, MAYBE_R2_FIXED 67, 76,					\
\
77, 78,							\
90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,			\
79,								\
96, 95, 94, 93, 92, 91,					\
108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,	\
109, 110,							\
111, 112, 113, 114, 115, 116,				\
117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,  \
129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,  \
141, 142, 143, 144, 145, 146, 147, 148			\
}
#define FP_REGNO_P(N) ((N) >= 32 && (N) <= 63)
#define CR_REGNO_P(N) ((N) >= CR0_REGNO && (N) <= CR7_REGNO)
#define CR_REGNO_NOT_CR0_P(N) ((N) >= CR1_REGNO && (N) <= CR7_REGNO)
#define INT_REGNO_P(N) \
((N) <= 31 || (N) == ARG_POINTER_REGNUM || (N) == FRAME_POINTER_REGNUM)
#define SPE_SIMD_REGNO_P(N) ((N) <= 31)
#define PAIRED_SIMD_REGNO_P(N) ((N) >= 32 && (N) <= 63)
#define CA_REGNO_P(N) ((N) == CA_REGNO)
#define ALTIVEC_REGNO_P(N) ((N) >= FIRST_ALTIVEC_REGNO && (N) <= LAST_ALTIVEC_REGNO)
#define VSX_REGNO_P(N) (FP_REGNO_P (N) || ALTIVEC_REGNO_P (N))
#define VFLOAT_REGNO_P(N) \
(ALTIVEC_REGNO_P (N) || (TARGET_VSX && FP_REGNO_P (N)))
#define VINT_REGNO_P(N) ALTIVEC_REGNO_P (N)
#define VLOGICAL_REGNO_P(N)						\
(INT_REGNO_P (N) || ALTIVEC_REGNO_P (N)				\
|| (TARGET_VSX && FP_REGNO_P (N)))					\
#define HARD_REGNO_CALLER_SAVE_MODE(REGNO, NREGS, MODE)			\
((NREGS) <= rs6000_hard_regno_nregs[MODE][REGNO]			\
? (MODE)								\
: TARGET_VSX								\
&& ((MODE) == VOIDmode || ALTIVEC_OR_VSX_VECTOR_MODE (MODE))	\
&& FP_REGNO_P (REGNO)						\
? V2DFmode								\
: TARGET_E500_DOUBLE && (MODE) == SImode				\
? SImode								\
: TARGET_E500_DOUBLE && ((MODE) == VOIDmode || (MODE) == DFmode)	\
? DFmode								\
: !TARGET_E500_DOUBLE && FLOAT128_IBM_P (MODE) && FP_REGNO_P (REGNO)	\
? DFmode								\
: !TARGET_E500_DOUBLE && (MODE) == TDmode && FP_REGNO_P (REGNO)	\
? DImode								\
: choose_hard_reg_mode ((REGNO), (NREGS), false))
#define VSX_VECTOR_MODE(MODE)		\
((MODE) == V4SFmode		\
|| (MODE) == V2DFmode)	\
#define ALTIVEC_VECTOR_MODE(MODE)					\
((MODE) == V16QImode							\
|| (MODE) == V8HImode						\
|| (MODE) == V4SFmode						\
|| (MODE) == V4SImode						\
|| FLOAT128_VECTOR_P (MODE))
#define ALTIVEC_OR_VSX_VECTOR_MODE(MODE)				\
(ALTIVEC_VECTOR_MODE (MODE) || VSX_VECTOR_MODE (MODE)			\
|| (MODE) == V2DImode || (MODE) == V1TImode)
#define SPE_VECTOR_MODE(MODE)		\
((MODE) == V4HImode          	\
|| (MODE) == V2SFmode          \
|| (MODE) == V1DImode          \
|| (MODE) == V2SImode)
#define PAIRED_VECTOR_MODE(MODE)        \
((MODE) == V2SFmode)            
#define HARD_REGNO_RENAME_OK(SRC, DST) \
(! ALTIVEC_REGNO_P (DST) || df_regs_ever_live_p (DST))
#define BRANCH_COST(speed_p, predictable_p) 3
#define LOGICAL_OP_NON_SHORT_CIRCUIT 0
#define FIXED_SCRATCH 0
#define STACK_POINTER_REGNUM 1
#define HARD_FRAME_POINTER_REGNUM 31
#define FRAME_POINTER_REGNUM 113
#define ARG_POINTER_REGNUM 67
#define STATIC_CHAIN_REGNUM 11
#define TLS_REGNUM ((TARGET_64BIT) ? 13 : 2)

enum reg_class
{
NO_REGS,
BASE_REGS,
GENERAL_REGS,
FLOAT_REGS,
ALTIVEC_REGS,
VSX_REGS,
VRSAVE_REGS,
VSCR_REGS,
SPE_ACC_REGS,
SPEFSCR_REGS,
SPR_REGS,
NON_SPECIAL_REGS,
LINK_REGS,
CTR_REGS,
LINK_OR_CTR_REGS,
SPECIAL_REGS,
SPEC_OR_GEN_REGS,
CR0_REGS,
CR_REGS,
NON_FLOAT_REGS,
CA_REGS,
SPE_HIGH_REGS,
ALL_REGS,
LIM_REG_CLASSES
};
#define N_REG_CLASSES (int) LIM_REG_CLASSES
#define REG_CLASS_NAMES							\
{									\
"NO_REGS",								\
"BASE_REGS",								\
"GENERAL_REGS",							\
"FLOAT_REGS",								\
"ALTIVEC_REGS",							\
"VSX_REGS",								\
"VRSAVE_REGS",							\
"VSCR_REGS",								\
"SPE_ACC_REGS",                                                       \
"SPEFSCR_REGS",                                                       \
"SPR_REGS",								\
"NON_SPECIAL_REGS",							\
"LINK_REGS",								\
"CTR_REGS",								\
"LINK_OR_CTR_REGS",							\
"SPECIAL_REGS",							\
"SPEC_OR_GEN_REGS",							\
"CR0_REGS",								\
"CR_REGS",								\
"NON_FLOAT_REGS",							\
"CA_REGS",								\
"SPE_HIGH_REGS",							\
"ALL_REGS"								\
}
#define REG_CLASS_CONTENTS						\
{									\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 },	\
\
{ 0xfffffffe, 0x00000000, 0x00000008, 0x00020000, 0x00000000 },	\
\
{ 0xffffffff, 0x00000000, 0x00000008, 0x00020000, 0x00000000 },	\
\
{ 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0xffffe000, 0x00001fff, 0x00000000 },	\
\
{ 0x00000000, 0xffffffff, 0xffffe000, 0x00001fff, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00002000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00004000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00008000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00010000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0x00040000, 0x00000000 },	\
\
{ 0xffffffff, 0xffffffff, 0x00000008, 0x00020000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000002, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000004, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000006, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000006, 0x00002000, 0x00000000 },	\
\
{ 0xffffffff, 0x00000000, 0x0000000e, 0x00022000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000010, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000ff0, 0x00000000, 0x00000000 },	\
\
{ 0xffffffff, 0x00000000, 0x00000ffe, 0x00020000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00001000, 0x00000000, 0x00000000 },	\
\
{ 0x00000000, 0x00000000, 0x00000000, 0xffe00000, 0x001fffff },	\
\
{ 0xffffffff, 0xffffffff, 0xfffffffe, 0xffe7ffff, 0x001fffff }	\
}
extern enum reg_class rs6000_regno_regclass[FIRST_PSEUDO_REGISTER];
#define REGNO_REG_CLASS(REGNO) 						\
(gcc_checking_assert (IN_RANGE ((REGNO), 0, FIRST_PSEUDO_REGISTER-1)),\
rs6000_regno_regclass[(REGNO)])
enum r6000_reg_class_enum {
RS6000_CONSTRAINT_d,		
RS6000_CONSTRAINT_f,		
RS6000_CONSTRAINT_v,		
RS6000_CONSTRAINT_wa,		
RS6000_CONSTRAINT_wb,		
RS6000_CONSTRAINT_wd,		
RS6000_CONSTRAINT_we,		
RS6000_CONSTRAINT_wf,		
RS6000_CONSTRAINT_wg,		
RS6000_CONSTRAINT_wh,		
RS6000_CONSTRAINT_wi,		
RS6000_CONSTRAINT_wj,		
RS6000_CONSTRAINT_wk,		
RS6000_CONSTRAINT_wl,		
RS6000_CONSTRAINT_wm,		
RS6000_CONSTRAINT_wo,		
RS6000_CONSTRAINT_wp,		
RS6000_CONSTRAINT_wq,		
RS6000_CONSTRAINT_wr,		
RS6000_CONSTRAINT_ws,		
RS6000_CONSTRAINT_wt,		
RS6000_CONSTRAINT_wu,		
RS6000_CONSTRAINT_wv,		
RS6000_CONSTRAINT_ww,		
RS6000_CONSTRAINT_wx,		
RS6000_CONSTRAINT_wy,		
RS6000_CONSTRAINT_wz,		
RS6000_CONSTRAINT_wA,		
RS6000_CONSTRAINT_wH,		
RS6000_CONSTRAINT_wI,		
RS6000_CONSTRAINT_wJ,		
RS6000_CONSTRAINT_wK,		
RS6000_CONSTRAINT_MAX
};
extern enum reg_class rs6000_constraints[RS6000_CONSTRAINT_MAX];
#define INDEX_REG_CLASS GENERAL_REGS
#define BASE_REG_CLASS BASE_REGS
#define VSX_REG_CLASS_P(CLASS)			\
((CLASS) == VSX_REGS || (CLASS) == FLOAT_REGS || (CLASS) == ALTIVEC_REGS)
#define GPR_REG_CLASS_P(CLASS) ((CLASS) == GENERAL_REGS || (CLASS) == BASE_REGS)
#define PREFERRED_RELOAD_CLASS(X,CLASS)			\
rs6000_preferred_reload_class_ptr (X, CLASS)
#define SECONDARY_RELOAD_CLASS(CLASS,MODE,IN) \
rs6000_secondary_reload_class_ptr (CLASS, MODE, IN)
#define SECONDARY_MEMORY_NEEDED_RTX(MODE) \
rs6000_secondary_memory_needed_rtx (MODE)
#define CLASS_MAX_NREGS(CLASS, MODE) rs6000_class_max_nregs[(MODE)][(CLASS)]
#define STACK_GROWS_DOWNWARD 1
#define DWARF_CIE_DATA_ALIGNMENT (-((int) (TARGET_32BIT ? 4 : 8)))
#define FRAME_GROWS_DOWNWARD (flag_stack_protect != 0			\
|| (flag_sanitize & SANITIZE_ADDRESS) != 0)
#define RS6000_SAVE_AREA \
((DEFAULT_ABI == ABI_V4 ? 8 : DEFAULT_ABI == ABI_ELFv2 ? 16 : 24)	\
<< (TARGET_64BIT ? 1 : 0))
#define RS6000_TOC_SAVE_SLOT \
((DEFAULT_ABI == ABI_ELFv2 ? 12 : 20) << (TARGET_64BIT ? 1 : 0))
#define RS6000_ALIGN(n,a) ROUND_UP ((n), (a))
#define RS6000_STARTING_FRAME_OFFSET					\
(cfun->calls_alloca							\
? (RS6000_ALIGN (crtl->outgoing_args_size + RS6000_SAVE_AREA,	\
(TARGET_ALTIVEC || TARGET_VSX) ? 16 : 8 ))		\
: (RS6000_ALIGN (crtl->outgoing_args_size,				\
(TARGET_ALTIVEC || TARGET_VSX) ? 16 : 8)		\
+ RS6000_SAVE_AREA))
#define STACK_DYNAMIC_OFFSET(FUNDECL)					\
RS6000_ALIGN (crtl->outgoing_args_size.to_constant ()			\
+ STACK_POINTER_OFFSET,					\
(TARGET_ALTIVEC || TARGET_VSX) ? 16 : 8)
#define FIRST_PARM_OFFSET(FNDECL) RS6000_SAVE_AREA
#define ARG_POINTER_CFA_OFFSET(FNDECL) 0
#define REG_PARM_STACK_SPACE(FNDECL) \
rs6000_reg_parm_stack_space ((FNDECL), false)
#define INCOMING_REG_PARM_STACK_SPACE(FNDECL) \
rs6000_reg_parm_stack_space ((FNDECL), true)
#define OUTGOING_REG_PARM_STACK_SPACE(FNTYPE) 1
#define STACK_POINTER_OFFSET RS6000_SAVE_AREA
#define ACCUMULATE_OUTGOING_ARGS 1
#define LIBCALL_VALUE(MODE) rs6000_libcall_value ((MODE))
#define DRAFT_V4_STRUCT_RET 0
#define DEFAULT_PCC_STRUCT_RETURN 0
#define STACK_SAVEAREA_MODE(LEVEL)	\
(LEVEL == SAVE_FUNCTION ? VOIDmode	\
: LEVEL == SAVE_NONLOCAL ? (TARGET_32BIT ? DImode : PTImode) : Pmode)
#define GP_ARG_MIN_REG 3
#define GP_ARG_MAX_REG 10
#define GP_ARG_NUM_REG (GP_ARG_MAX_REG - GP_ARG_MIN_REG + 1)
#define FP_ARG_MIN_REG 33
#define	FP_ARG_AIX_MAX_REG 45
#define	FP_ARG_V4_MAX_REG  40
#define	FP_ARG_MAX_REG (DEFAULT_ABI == ABI_V4				\
? FP_ARG_V4_MAX_REG : FP_ARG_AIX_MAX_REG)
#define FP_ARG_NUM_REG (FP_ARG_MAX_REG - FP_ARG_MIN_REG + 1)
#define ALTIVEC_ARG_MIN_REG (FIRST_ALTIVEC_REGNO + 2)
#define ALTIVEC_ARG_MAX_REG (ALTIVEC_ARG_MIN_REG + 11)
#define ALTIVEC_ARG_NUM_REG (ALTIVEC_ARG_MAX_REG - ALTIVEC_ARG_MIN_REG + 1)
#define AGGR_ARG_NUM_REG 8
#define GP_ARG_RETURN GP_ARG_MIN_REG
#define FP_ARG_RETURN FP_ARG_MIN_REG
#define ALTIVEC_ARG_RETURN (FIRST_ALTIVEC_REGNO + 2)
#define FP_ARG_MAX_RETURN (DEFAULT_ABI != ABI_ELFv2 ? FP_ARG_RETURN	\
: (FP_ARG_RETURN + AGGR_ARG_NUM_REG - 1))
#define ALTIVEC_ARG_MAX_RETURN (DEFAULT_ABI != ABI_ELFv2		\
? (ALTIVEC_ARG_RETURN			\
+ (TARGET_FLOAT128_TYPE ? 1 : 0))	\
: (ALTIVEC_ARG_RETURN + AGGR_ARG_NUM_REG - 1))
#define CALL_NORMAL		0x00000000	
#define CALL_V4_CLEAR_FP_ARGS	0x00000002	
#define CALL_V4_SET_FP_ARGS	0x00000004	
#define CALL_LONG		0x00000008	
#define CALL_LIBCALL		0x00000010	
#define WORLD_SAVE_P(INFO) 0
#define FUNCTION_VALUE_REGNO_P(N)					\
((N) == GP_ARG_RETURN							\
|| (IN_RANGE ((N), FP_ARG_RETURN, FP_ARG_MAX_RETURN)			\
&& TARGET_HARD_FLOAT && TARGET_FPRS)				\
|| (IN_RANGE ((N), ALTIVEC_ARG_RETURN, ALTIVEC_ARG_MAX_RETURN)	\
&& TARGET_ALTIVEC && TARGET_ALTIVEC_ABI))
#define FUNCTION_ARG_REGNO_P(N)						\
(IN_RANGE ((N), GP_ARG_MIN_REG, GP_ARG_MAX_REG)			\
|| (IN_RANGE ((N), ALTIVEC_ARG_MIN_REG, ALTIVEC_ARG_MAX_REG)		\
&& TARGET_ALTIVEC && TARGET_ALTIVEC_ABI)				\
|| (IN_RANGE ((N), FP_ARG_MIN_REG, FP_ARG_MAX_REG)			\
&& TARGET_HARD_FLOAT && TARGET_FPRS))

typedef struct rs6000_args
{
int words;			
int fregno;			
int vregno;			
int nargs_prototype;		
int prototype;		
int stdarg;			
int call_cookie;		
int sysv_gregno;		
int intoffset;		
int use_stack;		
int floats_in_gpr;		
int named;			
int escapes;			
int libcall;			
} CUMULATIVE_ARGS;
#define INIT_CUMULATIVE_ARGS(CUM, FNTYPE, LIBNAME, FNDECL, N_NAMED_ARGS) \
init_cumulative_args (&CUM, FNTYPE, LIBNAME, FALSE, FALSE, \
N_NAMED_ARGS, FNDECL, VOIDmode)
#define INIT_CUMULATIVE_INCOMING_ARGS(CUM, FNTYPE, LIBNAME) \
init_cumulative_args (&CUM, FNTYPE, LIBNAME, TRUE, FALSE, \
1000, current_function_decl, VOIDmode)
#define INIT_CUMULATIVE_LIBCALL_ARGS(CUM, MODE, LIBNAME) \
init_cumulative_args (&CUM, NULL_TREE, LIBNAME, FALSE, TRUE, \
0, NULL_TREE, MODE)
#define PAD_VARARGS_DOWN \
(targetm.calls.function_arg_padding (TYPE_MODE (type), type) == PAD_DOWNWARD)
#define FUNCTION_PROFILER(FILE, LABELNO)	\
output_function_profiler ((FILE), (LABELNO));
#define EXIT_IGNORE_STACK	1
#define	EPILOGUE_USES(REGNO)					\
((reload_completed && (REGNO) == LR_REGNO)			\
|| (TARGET_ALTIVEC && (REGNO) == VRSAVE_REGNO)		\
|| (crtl->calls_eh_return					\
&& TARGET_AIX						\
&& (REGNO) == 2))

#define TRAMPOLINE_SIZE rs6000_trampoline_size ()

#define RETURN_ADDRESS_OFFSET \
((DEFAULT_ABI == ABI_V4 ? 4 : 8) << (TARGET_64BIT ? 1 : 0))
#define RETURN_ADDR_RTX(COUNT, FRAME)                 \
(rs6000_return_addr (COUNT, FRAME))

#define ELIMINABLE_REGS					\
{{ HARD_FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM},	\
{ FRAME_POINTER_REGNUM, STACK_POINTER_REGNUM},		\
{ FRAME_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},	\
{ ARG_POINTER_REGNUM, STACK_POINTER_REGNUM},		\
{ ARG_POINTER_REGNUM, HARD_FRAME_POINTER_REGNUM},	\
{ RS6000_PIC_OFFSET_TABLE_REGNUM, RS6000_PIC_OFFSET_TABLE_REGNUM } }
#define INITIAL_ELIMINATION_OFFSET(FROM, TO, OFFSET) \
((OFFSET) = rs6000_initial_elimination_offset(FROM, TO))

#define HAVE_PRE_DECREMENT 1
#define HAVE_PRE_INCREMENT 1
#define HAVE_PRE_MODIFY_DISP 1
#define HAVE_PRE_MODIFY_REG 1
#define REGNO_OK_FOR_INDEX_P(REGNO)				\
((REGNO) < FIRST_PSEUDO_REGISTER				\
? (REGNO) <= 31 || (REGNO) == 67				\
|| (REGNO) == FRAME_POINTER_REGNUM				\
: (reg_renumber[REGNO] >= 0					\
&& (reg_renumber[REGNO] <= 31 || reg_renumber[REGNO] == 67	\
|| reg_renumber[REGNO] == FRAME_POINTER_REGNUM)))
#define REGNO_OK_FOR_BASE_P(REGNO)				\
((REGNO) < FIRST_PSEUDO_REGISTER				\
? ((REGNO) > 0 && (REGNO) <= 31) || (REGNO) == 67		\
|| (REGNO) == FRAME_POINTER_REGNUM				\
: (reg_renumber[REGNO] > 0					\
&& (reg_renumber[REGNO] <= 31 || reg_renumber[REGNO] == 67	\
|| reg_renumber[REGNO] == FRAME_POINTER_REGNUM)))
#define INT_REG_OK_FOR_INDEX_P(X, STRICT)			\
((!(STRICT) && REGNO (X) >= FIRST_PSEUDO_REGISTER)		\
|| REGNO_OK_FOR_INDEX_P (REGNO (X)))
#define INT_REG_OK_FOR_BASE_P(X, STRICT)			\
((!(STRICT) && REGNO (X) >= FIRST_PSEUDO_REGISTER)		\
|| REGNO_OK_FOR_BASE_P (REGNO (X)))

#define MAX_REGS_PER_ADDRESS 2
#define CONSTANT_ADDRESS_P(X)   \
(GET_CODE (X) == LABEL_REF || GET_CODE (X) == SYMBOL_REF		\
|| GET_CODE (X) == CONST_INT || GET_CODE (X) == CONST		\
|| GET_CODE (X) == HIGH)
#define EASY_VECTOR_15(n) ((n) >= -16 && (n) <= 15)
#define EASY_VECTOR_15_ADD_SELF(n) (!EASY_VECTOR_15((n))	\
&& EASY_VECTOR_15((n) >> 1) \
&& ((n) & 1) == 0)
#define EASY_VECTOR_MSB(n,mode)						\
((((unsigned HOST_WIDE_INT) (n)) & GET_MODE_MASK (mode)) ==		\
((((unsigned HOST_WIDE_INT)GET_MODE_MASK (mode)) + 1) >> 1))

#define LEGITIMIZE_RELOAD_ADDRESS(X,MODE,OPNUM,TYPE,IND_LEVELS,WIN)	     \
do {									     \
int win;								     \
(X) = rs6000_legitimize_reload_address_ptr ((X), (MODE), (OPNUM),	     \
(int)(TYPE), (IND_LEVELS), &win);		     \
if ( win )								     \
goto WIN;								     \
} while (0)
#define FIND_BASE_TERM rs6000_find_base_term

#define RS6000_PIC_OFFSET_TABLE_REGNUM 30
#define PIC_OFFSET_TABLE_REGNUM \
(TARGET_TOC ? TOC_REGISTER			\
: flag_pic ? RS6000_PIC_OFFSET_TABLE_REGNUM	\
: INVALID_REGNUM)
#define TOC_REGISTER (TARGET_MINIMAL_TOC ? RS6000_PIC_OFFSET_TABLE_REGNUM : 2)

#define FINAL_PRESCAN_INSN(INSN,OPERANDS,NOPERANDS) \
rs6000_final_prescan_insn (INSN, OPERANDS, NOPERANDS)
#define CASE_VECTOR_MODE SImode
#define CASE_VECTOR_PC_RELATIVE 1
#define DEFAULT_SIGNED_CHAR 0
#define MAX_FIXED_MODE_SIZE GET_MODE_BITSIZE (TARGET_POWERPC64 ? TImode : DImode)
#define MOVE_MAX (! TARGET_POWERPC64 ? 4 : 8)
#define MAX_MOVE_MAX 8
#define SLOW_BYTE_ACCESS 1
#define LOAD_EXTEND_OP(MODE) ZERO_EXTEND
#define SHORT_IMMEDIATES_SIGN_EXTEND 1

#define CLZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE) \
((VALUE) = GET_MODE_BITSIZE (MODE), 2)
#define CTZ_DEFINED_VALUE_AT_ZERO(MODE, VALUE)				\
(TARGET_CTZ || TARGET_POPCNTD						\
? ((VALUE) = GET_MODE_BITSIZE (MODE), 2)				\
: ((VALUE) = -1, 2))
extern scalar_int_mode rs6000_pmode;
#define Pmode rs6000_pmode
#define STACK_SIZE_MODE (TARGET_32BIT ? SImode : DImode)
#define FUNCTION_MODE SImode
#define NO_FUNCTION_CSE 1
#define SHIFT_COUNT_TRUNCATED 0
#define SELECT_CC_MODE(OP,X,Y) \
(SCALAR_FLOAT_MODE_P (GET_MODE (X)) ? CCFPmode	\
: (OP) == GTU || (OP) == LTU || (OP) == GEU || (OP) == LEU ? CCUNSmode \
: (((OP) == EQ || (OP) == NE) && COMPARISON_P (X)			  \
? CCEQmode : CCmode))
#define REVERSIBLE_CC_MODE(MODE) 1
#define REVERSE_CONDITION(CODE, MODE) rs6000_reverse_condition (MODE, CODE)

#define ASM_COMMENT_START " #"
extern int toc_initialized;
#define ASM_OUTPUT_SPECIAL_POOL_ENTRY(FILE, X, MODE, ALIGN, LABELNO, WIN) \
{ if (ASM_OUTPUT_SPECIAL_POOL_ENTRY_P (X, MODE))			  \
{									  \
output_toc (FILE, X, LABELNO, MODE);				  \
goto WIN;								  \
}									  \
}
#ifdef HAVE_GAS_WEAK
#define RS6000_WEAK 1
#else
#define RS6000_WEAK 0
#endif
#if RS6000_WEAK
#define        ASM_WEAKEN_DECL(FILE, DECL, NAME, VAL) \
rs6000_asm_weaken_decl ((FILE), (DECL), (NAME), (VAL))
#endif
#if HAVE_GAS_WEAKREF
#define ASM_OUTPUT_WEAKREF(FILE, DECL, NAME, VALUE)			\
do									\
{									\
fputs ("\t.weakref\t", (FILE));					\
RS6000_OUTPUT_BASENAME ((FILE), (NAME)); 				\
fputs (", ", (FILE));						\
RS6000_OUTPUT_BASENAME ((FILE), (VALUE));				\
if ((DECL) && TREE_CODE (DECL) == FUNCTION_DECL			\
&& DEFAULT_ABI == ABI_AIX && DOT_SYMBOLS)			\
{								\
fputs ("\n\t.weakref\t.", (FILE));				\
RS6000_OUTPUT_BASENAME ((FILE), (NAME)); 			\
fputs (", .", (FILE));					\
RS6000_OUTPUT_BASENAME ((FILE), (VALUE));			\
}								\
fputc ('\n', (FILE));						\
} while (0)
#endif
#undef	ASM_OUTPUT_DEF_FROM_DECLS
#define	ASM_OUTPUT_DEF_FROM_DECLS(FILE, DECL, TARGET)			\
do									\
{									\
const char *alias = XSTR (XEXP (DECL_RTL (DECL), 0), 0);		\
const char *name = IDENTIFIER_POINTER (TARGET);			\
if (TREE_CODE (DECL) == FUNCTION_DECL				\
&& DEFAULT_ABI == ABI_AIX && DOT_SYMBOLS)			\
{								\
if (TREE_PUBLIC (DECL))					\
{								\
if (!RS6000_WEAK || !DECL_WEAK (DECL))			\
{							\
fputs ("\t.globl\t.", FILE);				\
RS6000_OUTPUT_BASENAME (FILE, alias);			\
putc ('\n', FILE);					\
}							\
}								\
else if (TARGET_XCOFF)					\
{								\
if (!RS6000_WEAK || !DECL_WEAK (DECL))			\
{							\
fputs ("\t.lglobl\t.", FILE);				\
RS6000_OUTPUT_BASENAME (FILE, alias);			\
putc ('\n', FILE);					\
fputs ("\t.lglobl\t", FILE);				\
RS6000_OUTPUT_BASENAME (FILE, alias);			\
putc ('\n', FILE);					\
}							\
}								\
fputs ("\t.set\t.", FILE);					\
RS6000_OUTPUT_BASENAME (FILE, alias);				\
fputs (",.", FILE);						\
RS6000_OUTPUT_BASENAME (FILE, name);				\
fputc ('\n', FILE);						\
}								\
ASM_OUTPUT_DEF (FILE, alias, name);				\
}									\
while (0)
#define TARGET_ASM_FILE_START rs6000_file_start
#define ASM_APP_ON ""
#define ASM_APP_OFF ""
extern char rs6000_reg_names[][8];	
#define REGISTER_NAMES							\
{									\
&rs6000_reg_names[ 0][0],					\
&rs6000_reg_names[ 1][0],					\
&rs6000_reg_names[ 2][0],     				\
&rs6000_reg_names[ 3][0],					\
&rs6000_reg_names[ 4][0],					\
&rs6000_reg_names[ 5][0],					\
&rs6000_reg_names[ 6][0],					\
&rs6000_reg_names[ 7][0],					\
&rs6000_reg_names[ 8][0],					\
&rs6000_reg_names[ 9][0],					\
&rs6000_reg_names[10][0],					\
&rs6000_reg_names[11][0],					\
&rs6000_reg_names[12][0],					\
&rs6000_reg_names[13][0],					\
&rs6000_reg_names[14][0],					\
&rs6000_reg_names[15][0],					\
&rs6000_reg_names[16][0],					\
&rs6000_reg_names[17][0],					\
&rs6000_reg_names[18][0],					\
&rs6000_reg_names[19][0],					\
&rs6000_reg_names[20][0],					\
&rs6000_reg_names[21][0],					\
&rs6000_reg_names[22][0],					\
&rs6000_reg_names[23][0],					\
&rs6000_reg_names[24][0],					\
&rs6000_reg_names[25][0],					\
&rs6000_reg_names[26][0],					\
&rs6000_reg_names[27][0],					\
&rs6000_reg_names[28][0],					\
&rs6000_reg_names[29][0],					\
&rs6000_reg_names[30][0],					\
&rs6000_reg_names[31][0],					\
\
&rs6000_reg_names[32][0],     				\
&rs6000_reg_names[33][0],					\
&rs6000_reg_names[34][0],					\
&rs6000_reg_names[35][0],					\
&rs6000_reg_names[36][0],					\
&rs6000_reg_names[37][0],					\
&rs6000_reg_names[38][0],					\
&rs6000_reg_names[39][0],					\
&rs6000_reg_names[40][0],					\
&rs6000_reg_names[41][0],					\
&rs6000_reg_names[42][0],					\
&rs6000_reg_names[43][0],					\
&rs6000_reg_names[44][0],					\
&rs6000_reg_names[45][0],					\
&rs6000_reg_names[46][0],					\
&rs6000_reg_names[47][0],					\
&rs6000_reg_names[48][0],					\
&rs6000_reg_names[49][0],					\
&rs6000_reg_names[50][0],					\
&rs6000_reg_names[51][0],					\
&rs6000_reg_names[52][0],					\
&rs6000_reg_names[53][0],					\
&rs6000_reg_names[54][0],					\
&rs6000_reg_names[55][0],					\
&rs6000_reg_names[56][0],					\
&rs6000_reg_names[57][0],					\
&rs6000_reg_names[58][0],					\
&rs6000_reg_names[59][0],					\
&rs6000_reg_names[60][0],					\
&rs6000_reg_names[61][0],					\
&rs6000_reg_names[62][0],					\
&rs6000_reg_names[63][0],					\
\
&rs6000_reg_names[64][0],     				\
&rs6000_reg_names[65][0],					\
&rs6000_reg_names[66][0],					\
&rs6000_reg_names[67][0],					\
\
&rs6000_reg_names[68][0],					\
&rs6000_reg_names[69][0],					\
&rs6000_reg_names[70][0],					\
&rs6000_reg_names[71][0],					\
&rs6000_reg_names[72][0],					\
&rs6000_reg_names[73][0],					\
&rs6000_reg_names[74][0],					\
&rs6000_reg_names[75][0],					\
\
&rs6000_reg_names[76][0],					\
\
&rs6000_reg_names[77][0],					\
&rs6000_reg_names[78][0],					\
&rs6000_reg_names[79][0],					\
&rs6000_reg_names[80][0],					\
&rs6000_reg_names[81][0],					\
&rs6000_reg_names[82][0],					\
&rs6000_reg_names[83][0],					\
&rs6000_reg_names[84][0],					\
&rs6000_reg_names[85][0],					\
&rs6000_reg_names[86][0],					\
&rs6000_reg_names[87][0],					\
&rs6000_reg_names[88][0],					\
&rs6000_reg_names[89][0],					\
&rs6000_reg_names[90][0],					\
&rs6000_reg_names[91][0],					\
&rs6000_reg_names[92][0],					\
&rs6000_reg_names[93][0],					\
&rs6000_reg_names[94][0],					\
&rs6000_reg_names[95][0],					\
&rs6000_reg_names[96][0],					\
&rs6000_reg_names[97][0],					\
&rs6000_reg_names[98][0],					\
&rs6000_reg_names[99][0],					\
&rs6000_reg_names[100][0],					\
&rs6000_reg_names[101][0],					\
&rs6000_reg_names[102][0],					\
&rs6000_reg_names[103][0],					\
&rs6000_reg_names[104][0],					\
&rs6000_reg_names[105][0],					\
&rs6000_reg_names[106][0],					\
&rs6000_reg_names[107][0],					\
&rs6000_reg_names[108][0],					\
&rs6000_reg_names[109][0],					\
&rs6000_reg_names[110][0],					\
&rs6000_reg_names[111][0],					\
&rs6000_reg_names[112][0],					\
&rs6000_reg_names[113][0],					\
&rs6000_reg_names[114][0],					\
&rs6000_reg_names[115][0],					\
&rs6000_reg_names[116][0],					\
\
&rs6000_reg_names[117][0],					\
&rs6000_reg_names[118][0],					\
&rs6000_reg_names[119][0],					\
&rs6000_reg_names[120][0],					\
&rs6000_reg_names[121][0],					\
&rs6000_reg_names[122][0],					\
&rs6000_reg_names[123][0],					\
&rs6000_reg_names[124][0],					\
&rs6000_reg_names[125][0],					\
&rs6000_reg_names[126][0],					\
&rs6000_reg_names[127][0],				\
&rs6000_reg_names[128][0],				\
&rs6000_reg_names[129][0],				\
&rs6000_reg_names[130][0],				\
&rs6000_reg_names[131][0],				\
&rs6000_reg_names[132][0],				\
&rs6000_reg_names[133][0],				\
&rs6000_reg_names[134][0],				\
&rs6000_reg_names[135][0],				\
&rs6000_reg_names[136][0],				\
&rs6000_reg_names[137][0],				\
&rs6000_reg_names[138][0],				\
&rs6000_reg_names[139][0],				\
&rs6000_reg_names[140][0],				\
&rs6000_reg_names[141][0],				\
&rs6000_reg_names[142][0],				\
&rs6000_reg_names[143][0],				\
&rs6000_reg_names[144][0],				\
&rs6000_reg_names[145][0],				\
&rs6000_reg_names[146][0],				\
&rs6000_reg_names[147][0],				\
&rs6000_reg_names[148][0],				\
}
#define ADDITIONAL_REGISTER_NAMES \
{{"r0",    0}, {"r1",    1}, {"r2",    2}, {"r3",    3},	\
{"r4",    4}, {"r5",    5}, {"r6",    6}, {"r7",    7},	\
{"r8",    8}, {"r9",    9}, {"r10",  10}, {"r11",  11},	\
{"r12",  12}, {"r13",  13}, {"r14",  14}, {"r15",  15},	\
{"r16",  16}, {"r17",  17}, {"r18",  18}, {"r19",  19},	\
{"r20",  20}, {"r21",  21}, {"r22",  22}, {"r23",  23},	\
{"r24",  24}, {"r25",  25}, {"r26",  26}, {"r27",  27},	\
{"r28",  28}, {"r29",  29}, {"r30",  30}, {"r31",  31},	\
{"fr0",  32}, {"fr1",  33}, {"fr2",  34}, {"fr3",  35},	\
{"fr4",  36}, {"fr5",  37}, {"fr6",  38}, {"fr7",  39},	\
{"fr8",  40}, {"fr9",  41}, {"fr10", 42}, {"fr11", 43},	\
{"fr12", 44}, {"fr13", 45}, {"fr14", 46}, {"fr15", 47},	\
{"fr16", 48}, {"fr17", 49}, {"fr18", 50}, {"fr19", 51},	\
{"fr20", 52}, {"fr21", 53}, {"fr22", 54}, {"fr23", 55},	\
{"fr24", 56}, {"fr25", 57}, {"fr26", 58}, {"fr27", 59},	\
{"fr28", 60}, {"fr29", 61}, {"fr30", 62}, {"fr31", 63},	\
{"v0",   77}, {"v1",   78}, {"v2",   79}, {"v3",   80},       \
{"v4",   81}, {"v5",   82}, {"v6",   83}, {"v7",   84},       \
{"v8",   85}, {"v9",   86}, {"v10",  87}, {"v11",  88},       \
{"v12",  89}, {"v13",  90}, {"v14",  91}, {"v15",  92},       \
{"v16",  93}, {"v17",  94}, {"v18",  95}, {"v19",  96},       \
{"v20",  97}, {"v21",  98}, {"v22",  99}, {"v23",  100},	\
{"v24",  101},{"v25",  102},{"v26",  103},{"v27",  104},      \
{"v28",  105},{"v29",  106},{"v30",  107},{"v31",  108},      \
{"vrsave", 109}, {"vscr", 110},				\
{"spe_acc", 111}, {"spefscr", 112},				\
\
{"cr0",  68}, {"cr1",  69}, {"cr2",  70}, {"cr3",  71},	\
{"cr4",  72}, {"cr5",  73}, {"cr6",  74}, {"cr7",  75},	\
{"cc",   68}, {"sp",    1}, {"toc",   2},			\
\
{"xer",  76},							\
\
{"vs0",  32}, {"vs1",  33}, {"vs2",  34}, {"vs3",  35},	\
{"vs4",  36}, {"vs5",  37}, {"vs6",  38}, {"vs7",  39},	\
{"vs8",  40}, {"vs9",  41}, {"vs10", 42}, {"vs11", 43},	\
{"vs12", 44}, {"vs13", 45}, {"vs14", 46}, {"vs15", 47},	\
{"vs16", 48}, {"vs17", 49}, {"vs18", 50}, {"vs19", 51},	\
{"vs20", 52}, {"vs21", 53}, {"vs22", 54}, {"vs23", 55},	\
{"vs24", 56}, {"vs25", 57}, {"vs26", 58}, {"vs27", 59},	\
{"vs28", 60}, {"vs29", 61}, {"vs30", 62}, {"vs31", 63},	\
{"vs32", 77}, {"vs33", 78}, {"vs34", 79}, {"vs35", 80},       \
{"vs36", 81}, {"vs37", 82}, {"vs38", 83}, {"vs39", 84},       \
{"vs40", 85}, {"vs41", 86}, {"vs42", 87}, {"vs43", 88},       \
{"vs44", 89}, {"vs45", 90}, {"vs46", 91}, {"vs47", 92},       \
{"vs48", 93}, {"vs49", 94}, {"vs50", 95}, {"vs51", 96},       \
{"vs52", 97}, {"vs53", 98}, {"vs54", 99}, {"vs55", 100},	\
{"vs56", 101},{"vs57", 102},{"vs58", 103},{"vs59", 104},      \
{"vs60", 105},{"vs61", 106},{"vs62", 107},{"vs63", 108},	\
\
{"tfhar",  114}, {"tfiar",  115}, {"texasr",  116},		\
\
{"rh0",  117}, {"rh1",  118}, {"rh2",  119}, {"rh3",  120},	\
{"rh4",  121}, {"rh5",  122}, {"rh6",  123}, {"rh7",  124},	\
{"rh8",  125}, {"rh9",  126}, {"rh10", 127}, {"rh11", 128},	\
{"rh12", 129}, {"rh13", 130}, {"rh14", 131}, {"rh15", 132},	\
{"rh16", 133}, {"rh17", 134}, {"rh18", 135}, {"rh19", 136},	\
{"rh20", 137}, {"rh21", 138}, {"rh22", 139}, {"rh23", 140},	\
{"rh24", 141}, {"rh25", 142}, {"rh26", 143}, {"rh27", 144},	\
{"rh28", 145}, {"rh29", 146}, {"rh30", 147}, {"rh31", 148},	\
}
#define ASM_OUTPUT_ADDR_DIFF_ELT(FILE, BODY, VALUE, REL) \
do { char buf[100];					\
fputs ("\t.long ", FILE);			\
ASM_GENERATE_INTERNAL_LABEL (buf, "L", VALUE);	\
assemble_name (FILE, buf);			\
putc ('-', FILE);				\
ASM_GENERATE_INTERNAL_LABEL (buf, "L", REL);	\
assemble_name (FILE, buf);			\
putc ('\n', FILE);				\
} while (0)
#define ASM_OUTPUT_ALIGN(FILE,LOG)	\
if ((LOG) != 0)			\
fprintf (FILE, "\t.align %d\n", (LOG))
#define LOOP_ALIGN(LABEL)  rs6000_loop_align(LABEL)
#define MALLOC_ABI_ALIGNMENT (64)
#define INCOMING_RETURN_ADDR_RTX   gen_rtx_REG (Pmode, LR_REGNO)
#define DWARF_FRAME_RETURN_COLUMN  DWARF_FRAME_REGNUM (LR_REGNO)
#define EH_RETURN_DATA_REGNO(N) ((N) < 4 ? (N) + 3 : INVALID_REGNUM)
#define EH_RETURN_STACKADJ_RTX  gen_rtx_REG (Pmode, 10)
#define PRINT_OPERAND(FILE, X, CODE)  print_operand (FILE, X, CODE)
#define PRINT_OPERAND_PUNCT_VALID_P(CODE)  ((CODE) == '&')
#define PRINT_OPERAND_ADDRESS(FILE, ADDR) print_operand_address (FILE, ADDR)
#define SWITCHABLE_TARGET 1
extern int frame_pointer_needed;
#define RS6000_BTC_SPECIAL	0x00000000	
#define RS6000_BTC_UNARY	0x00000001	
#define RS6000_BTC_BINARY	0x00000002	
#define RS6000_BTC_TERNARY	0x00000003	
#define RS6000_BTC_PREDICATE	0x00000004	
#define RS6000_BTC_ABS		0x00000005	
#define RS6000_BTC_EVSEL	0x00000006	
#define RS6000_BTC_DST		0x00000007	
#define RS6000_BTC_TYPE_MASK	0x0000000f	
#define RS6000_BTC_MISC		0x00000000	
#define RS6000_BTC_CONST	0x00000100	
#define RS6000_BTC_PURE		0x00000200	
#define RS6000_BTC_FP		0x00000400	
#define RS6000_BTC_ATTR_MASK	0x00000700	
#define RS6000_BTC_SPR		0x01000000	
#define RS6000_BTC_VOID		0x02000000	
#define RS6000_BTC_CR		0x04000000	
#define RS6000_BTC_OVERLOADED	0x08000000	
#define RS6000_BTC_MISC_MASK	0x1f000000	
#define RS6000_BTC_MEM		RS6000_BTC_MISC	
#define RS6000_BTC_SAT		RS6000_BTC_MISC	
#define RS6000_BTM_ALWAYS	0		
#define RS6000_BTM_ALTIVEC	MASK_ALTIVEC	
#define RS6000_BTM_CMPB		MASK_CMPB	
#define RS6000_BTM_VSX		MASK_VSX	
#define RS6000_BTM_P8_VECTOR	MASK_P8_VECTOR	
#define RS6000_BTM_P9_VECTOR	MASK_P9_VECTOR	
#define RS6000_BTM_P9_MISC	MASK_P9_MISC	
#define RS6000_BTM_CRYPTO	MASK_CRYPTO	
#define RS6000_BTM_HTM		MASK_HTM	
#define RS6000_BTM_SPE		MASK_STRING	
#define RS6000_BTM_PAIRED	MASK_MULHW	
#define RS6000_BTM_FRE		MASK_POPCNTB	
#define RS6000_BTM_FRES		MASK_PPC_GFXOPT	
#define RS6000_BTM_FRSQRTE	MASK_PPC_GFXOPT	
#define RS6000_BTM_FRSQRTES	MASK_POPCNTB	
#define RS6000_BTM_POPCNTD	MASK_POPCNTD	
#define RS6000_BTM_CELL		MASK_FPRND	
#define RS6000_BTM_DFP		MASK_DFP	
#define RS6000_BTM_HARD_FLOAT	MASK_SOFT_FLOAT	
#define RS6000_BTM_LDBL128	MASK_MULTIPLE	
#define RS6000_BTM_64BIT	MASK_64BIT	
#define RS6000_BTM_FLOAT128	MASK_FLOAT128_TYPE 
#define RS6000_BTM_COMMON	(RS6000_BTM_ALTIVEC			\
| RS6000_BTM_VSX			\
| RS6000_BTM_P8_VECTOR			\
| RS6000_BTM_P9_VECTOR			\
| RS6000_BTM_P9_MISC			\
| RS6000_BTM_MODULO                    \
| RS6000_BTM_CRYPTO			\
| RS6000_BTM_FRE			\
| RS6000_BTM_FRES			\
| RS6000_BTM_FRSQRTE			\
| RS6000_BTM_FRSQRTES			\
| RS6000_BTM_HTM			\
| RS6000_BTM_POPCNTD			\
| RS6000_BTM_CELL			\
| RS6000_BTM_DFP			\
| RS6000_BTM_HARD_FLOAT		\
| RS6000_BTM_LDBL128			\
| RS6000_BTM_FLOAT128)
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_E
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_S
#undef RS6000_BUILTIN_X
#define RS6000_BUILTIN_0(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_1(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_2(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_3(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_A(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_D(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_E(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_H(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_P(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_Q(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_S(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
#define RS6000_BUILTIN_X(ENUM, NAME, MASK, ATTR, ICODE) ENUM,
enum rs6000_builtins
{
#include "powerpcspe-builtin.def"
RS6000_BUILTIN_COUNT
};
#undef RS6000_BUILTIN_0
#undef RS6000_BUILTIN_1
#undef RS6000_BUILTIN_2
#undef RS6000_BUILTIN_3
#undef RS6000_BUILTIN_A
#undef RS6000_BUILTIN_D
#undef RS6000_BUILTIN_E
#undef RS6000_BUILTIN_H
#undef RS6000_BUILTIN_P
#undef RS6000_BUILTIN_Q
#undef RS6000_BUILTIN_S
#undef RS6000_BUILTIN_X
enum rs6000_builtin_type_index
{
RS6000_BTI_NOT_OPAQUE,
RS6000_BTI_opaque_V2SI,
RS6000_BTI_opaque_V2SF,
RS6000_BTI_opaque_p_V2SI,
RS6000_BTI_opaque_V4SI,
RS6000_BTI_V16QI,
RS6000_BTI_V1TI,
RS6000_BTI_V2SI,
RS6000_BTI_V2SF,
RS6000_BTI_V2DI,
RS6000_BTI_V2DF,
RS6000_BTI_V4HI,
RS6000_BTI_V4SI,
RS6000_BTI_V4SF,
RS6000_BTI_V8HI,
RS6000_BTI_unsigned_V16QI,
RS6000_BTI_unsigned_V1TI,
RS6000_BTI_unsigned_V8HI,
RS6000_BTI_unsigned_V4SI,
RS6000_BTI_unsigned_V2DI,
RS6000_BTI_bool_char,          
RS6000_BTI_bool_short,         
RS6000_BTI_bool_int,           
RS6000_BTI_bool_long,		 
RS6000_BTI_pixel,              
RS6000_BTI_bool_V16QI,         
RS6000_BTI_bool_V8HI,          
RS6000_BTI_bool_V4SI,          
RS6000_BTI_bool_V2DI,          
RS6000_BTI_pixel_V8HI,         
RS6000_BTI_long,	         
RS6000_BTI_unsigned_long,      
RS6000_BTI_long_long,	         
RS6000_BTI_unsigned_long_long, 
RS6000_BTI_INTQI,	         
RS6000_BTI_UINTQI,		 
RS6000_BTI_INTHI,	         
RS6000_BTI_UINTHI,		 
RS6000_BTI_INTSI,		 
RS6000_BTI_UINTSI,		 
RS6000_BTI_INTDI,		 
RS6000_BTI_UINTDI,		 
RS6000_BTI_INTTI,		 
RS6000_BTI_UINTTI,		 
RS6000_BTI_float,	         
RS6000_BTI_double,	         
RS6000_BTI_long_double,        
RS6000_BTI_dfloat64,		 
RS6000_BTI_dfloat128,		 
RS6000_BTI_void,	         
RS6000_BTI_ieee128_float,	 
RS6000_BTI_ibm128_float,	 
RS6000_BTI_const_str,		 
RS6000_BTI_MAX
};
#define opaque_V2SI_type_node         (rs6000_builtin_types[RS6000_BTI_opaque_V2SI])
#define opaque_V2SF_type_node         (rs6000_builtin_types[RS6000_BTI_opaque_V2SF])
#define opaque_p_V2SI_type_node       (rs6000_builtin_types[RS6000_BTI_opaque_p_V2SI])
#define opaque_V4SI_type_node         (rs6000_builtin_types[RS6000_BTI_opaque_V4SI])
#define V16QI_type_node               (rs6000_builtin_types[RS6000_BTI_V16QI])
#define V1TI_type_node                (rs6000_builtin_types[RS6000_BTI_V1TI])
#define V2DI_type_node                (rs6000_builtin_types[RS6000_BTI_V2DI])
#define V2DF_type_node                (rs6000_builtin_types[RS6000_BTI_V2DF])
#define V2SI_type_node                (rs6000_builtin_types[RS6000_BTI_V2SI])
#define V2SF_type_node                (rs6000_builtin_types[RS6000_BTI_V2SF])
#define V4HI_type_node                (rs6000_builtin_types[RS6000_BTI_V4HI])
#define V4SI_type_node                (rs6000_builtin_types[RS6000_BTI_V4SI])
#define V4SF_type_node                (rs6000_builtin_types[RS6000_BTI_V4SF])
#define V8HI_type_node                (rs6000_builtin_types[RS6000_BTI_V8HI])
#define unsigned_V16QI_type_node      (rs6000_builtin_types[RS6000_BTI_unsigned_V16QI])
#define unsigned_V1TI_type_node       (rs6000_builtin_types[RS6000_BTI_unsigned_V1TI])
#define unsigned_V8HI_type_node       (rs6000_builtin_types[RS6000_BTI_unsigned_V8HI])
#define unsigned_V4SI_type_node       (rs6000_builtin_types[RS6000_BTI_unsigned_V4SI])
#define unsigned_V2DI_type_node       (rs6000_builtin_types[RS6000_BTI_unsigned_V2DI])
#define bool_char_type_node           (rs6000_builtin_types[RS6000_BTI_bool_char])
#define bool_short_type_node          (rs6000_builtin_types[RS6000_BTI_bool_short])
#define bool_int_type_node            (rs6000_builtin_types[RS6000_BTI_bool_int])
#define bool_long_type_node           (rs6000_builtin_types[RS6000_BTI_bool_long])
#define pixel_type_node               (rs6000_builtin_types[RS6000_BTI_pixel])
#define bool_V16QI_type_node	      (rs6000_builtin_types[RS6000_BTI_bool_V16QI])
#define bool_V8HI_type_node	      (rs6000_builtin_types[RS6000_BTI_bool_V8HI])
#define bool_V4SI_type_node	      (rs6000_builtin_types[RS6000_BTI_bool_V4SI])
#define bool_V2DI_type_node	      (rs6000_builtin_types[RS6000_BTI_bool_V2DI])
#define pixel_V8HI_type_node	      (rs6000_builtin_types[RS6000_BTI_pixel_V8HI])
#define long_long_integer_type_internal_node  (rs6000_builtin_types[RS6000_BTI_long_long])
#define long_long_unsigned_type_internal_node (rs6000_builtin_types[RS6000_BTI_unsigned_long_long])
#define long_integer_type_internal_node  (rs6000_builtin_types[RS6000_BTI_long])
#define long_unsigned_type_internal_node (rs6000_builtin_types[RS6000_BTI_unsigned_long])
#define intQI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_INTQI])
#define uintQI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_UINTQI])
#define intHI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_INTHI])
#define uintHI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_UINTHI])
#define intSI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_INTSI])
#define uintSI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_UINTSI])
#define intDI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_INTDI])
#define uintDI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_UINTDI])
#define intTI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_INTTI])
#define uintTI_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_UINTTI])
#define float_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_float])
#define double_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_double])
#define long_double_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_long_double])
#define dfloat64_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_dfloat64])
#define dfloat128_type_internal_node	 (rs6000_builtin_types[RS6000_BTI_dfloat128])
#define void_type_internal_node		 (rs6000_builtin_types[RS6000_BTI_void])
#define ieee128_float_type_node		 (rs6000_builtin_types[RS6000_BTI_ieee128_float])
#define ibm128_float_type_node		 (rs6000_builtin_types[RS6000_BTI_ibm128_float])
#define const_str_type_node		 (rs6000_builtin_types[RS6000_BTI_const_str])
extern GTY(()) tree rs6000_builtin_types[RS6000_BTI_MAX];
extern GTY(()) tree rs6000_builtin_decls[RS6000_BUILTIN_COUNT];
#define TARGET_SUPPORTS_WIDE_INT 1
#if (GCC_VERSION >= 3000)
#pragma GCC poison TARGET_FLOAT128 OPTION_MASK_FLOAT128 MASK_FLOAT128
#endif
