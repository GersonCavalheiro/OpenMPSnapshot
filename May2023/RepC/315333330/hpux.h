#undef  TARGET_HPUX
#define TARGET_HPUX 1
#undef WCHAR_TYPE
#define WCHAR_TYPE "unsigned int"
#undef WCHAR_TYPE_SIZE
#define WCHAR_TYPE_SIZE 32
#define TARGET_OS_CPP_BUILTINS()			\
do {							\
builtin_assert("system=hpux");			\
builtin_assert("system=posix");			\
builtin_assert("system=unix");			\
builtin_define_std("hpux");			\
builtin_define_std("unix");			\
builtin_define("__IA64__");			\
builtin_define("_LONGLONG");			\
builtin_define("_INCLUDE_LONGLONG");		\
builtin_define("__STDC_EXT__");			\
builtin_define("_UINT128_T");			\
if (c_dialect_cxx () || !flag_iso)		\
{						\
builtin_define("_HPUX_SOURCE");		\
builtin_define("__STDCPP__");		\
builtin_define("_INCLUDE__STDC_A1_SOURCE");	\
}						\
if (TARGET_ILP32)				\
builtin_define("_ILP32");			\
} while (0)
#undef CPP_SPEC
#define CPP_SPEC \
"%{mt|pthread:-D_REENTRANT -D_THREAD_SAFE -D_POSIX_C_SOURCE=199506L}"
#undef  ASM_EXTRA_SPEC
#define ASM_EXTRA_SPEC "%{milp32:-milp32} %{mlp64:-mlp64}"
#ifndef USE_GAS
#define AS_NEEDS_DASH_FOR_PIPED_INPUT
#endif
#ifndef CROSS_DIRECTORY_STRUCTURE
#undef MD_EXEC_PREFIX
#define MD_EXEC_PREFIX "/usr/ccs/bin/"
#undef MD_STARTFILE_PREFIX
#define MD_STARTFILE_PREFIX "/usr/ccs/lib/"
#endif
#undef ENDFILE_SPEC
#undef STARTFILE_SPEC
#define STARTFILE_SPEC "%{!shared:%{static:crt0%O%s} \
%{mlp64:/usr/lib/hpux64/unix98%O%s} \
%{!mlp64:/usr/lib/hpux32/unix98%O%s}}"
#undef LINK_SPEC
#define LINK_SPEC \
"-z +Accept TypeMismatch \
%{shared:-b} \
%{!shared: \
-u main \
%{static:-noshared}}"
#undef  LIB_SPEC
#define LIB_SPEC \
"%{!shared: \
%{mt|pthread:%{fopenacc|fopenmp|%:gt(%{ftree-parallelize-loops=*:%*} 1):-lrt} -lpthread} \
%{p:%{!mlp64:-L/usr/lib/hpux32/libp} \
%{mlp64:-L/usr/lib/hpux64/libp} -lprof} \
%{pg:%{!mlp64:-L/usr/lib/hpux32/libp} \
%{mlp64:-L/usr/lib/hpux64/libp} -lgprof} \
%{!symbolic:-lc}}"
#define MULTILIB_DEFAULTS { "milp32" }
#define POINTERS_EXTEND_UNSIGNED -1
#define JMP_BUF_SIZE  (8 * 76)
#undef TARGET_DEFAULT
#define TARGET_DEFAULT \
(MASK_DWARF2_ASM | MASK_BIG_ENDIAN | MASK_ILP32)
#undef ASM_OUTPUT_EXTERNAL_LIBCALL
#define ASM_OUTPUT_EXTERNAL_LIBCALL(FILE, FUN)			\
do {								\
(*targetm.asm_out.globalize_label) (FILE, XSTR (FUN, 0));	\
ASM_OUTPUT_TYPE_DIRECTIVE (FILE, XSTR (FUN, 0), "function");	\
} while (0)
#undef PAD_VARARGS_DOWN
#define PAD_VARARGS_DOWN (!AGGREGATE_TYPE_P (type))
#define REGISTER_TARGET_PRAGMAS() \
c_register_pragma (0, "builtin", ia64_hpux_handle_builtin_pragma)
#undef TARGET_HPUX_LD
#define TARGET_HPUX_LD	1
#define GTHREAD_USE_WEAK 0
#undef CTORS_SECTION_ASM_OP
#define CTORS_SECTION_ASM_OP  "\t.section\t.init_array,\t\"aw\",\"init_array\""
#undef DTORS_SECTION_ASM_OP
#define DTORS_SECTION_ASM_OP  "\t.section\t.fini_array,\t\"aw\",\"fini_array\""
#define SUPPORTS_INIT_PRIORITY 0
#undef READONLY_DATA_SECTION_ASM_OP
#define READONLY_DATA_SECTION_ASM_OP "\t.section\t.rodata,\t\"a\",\t\"progbits\""
#undef DATA_SECTION_ASM_OP
#define DATA_SECTION_ASM_OP "\t.section\t.data,\t\"aw\",\t\"progbits\""
#undef SDATA_SECTION_ASM_OP
#define SDATA_SECTION_ASM_OP "\t.section\t.sdata,\t\"asw\",\t\"progbits\""
#undef BSS_SECTION_ASM_OP
#define BSS_SECTION_ASM_OP "\t.section\t.bss,\t\"aw\",\t\"nobits\""
#undef SBSS_SECTION_ASM_OP
#define SBSS_SECTION_ASM_OP "\t.section\t.sbss,\t\"asw\",\t\"nobits\""
#undef TEXT_SECTION_ASM_OP
#define TEXT_SECTION_ASM_OP "\t.section\t.text,\t\"ax\",\t\"progbits\""
#undef  TARGET_ASM_RELOC_RW_MASK
#define TARGET_ASM_RELOC_RW_MASK  ia64_hpux_reloc_rw_mask
#undef TARGET_LIBC_HAS_FUNCTION
#define TARGET_LIBC_HAS_FUNCTION default_libc_has_function
#undef TARGET_INIT_LIBFUNCS
#define TARGET_INIT_LIBFUNCS ia64_hpux_init_libfuncs
#define FLOAT_LIB_COMPARE_RETURNS_BOOL(MODE, COMPARISON) ((MODE) == TFmode)
#define NO_IMPLICIT_EXTERN_C
#undef FUNCTION_PROFILER
#define FUNCTION_PROFILER(FILE, LABELNO) do { } while (0)
#undef PROFILE_HOOK
#define PROFILE_HOOK(LABEL) ia64_profile_hook (LABEL)
#undef  PROFILE_BEFORE_PROLOGUE
#undef NO_PROFILE_COUNTERS
#define NO_PROFILE_COUNTERS 0
#define TARGET_ASM_FUNCTION_SECTION ia64_hpux_function_section
#define TARGET_POSIX_IO
#define STACK_CHECK_STATIC_BUILTIN 1
#define STACK_CHECK_PROTECT (24 * 1024)
