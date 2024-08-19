#define TARGET_OBJECT_SUFFIX ".obj"
#define TARGET_EXECUTABLE_SUFFIX ".exe"
#define TARGET_OS_CPP_BUILTINS()					 \
do {									 \
builtin_define_std ("vms");						 \
builtin_define_std ("VMS");						 \
builtin_assert ("system=vms");					 \
SUBTARGET_OS_CPP_BUILTINS();					 \
builtin_define ("__int64=long long");				 \
if (flag_vms_pointer_size == VMS_POINTER_SIZE_32)			 \
builtin_define ("__INITIAL_POINTER_SIZE=32");			 \
else if (flag_vms_pointer_size == VMS_POINTER_SIZE_64)		 \
builtin_define ("__INITIAL_POINTER_SIZE=64");			 \
if (POINTER_SIZE == 64)						 \
builtin_define ("__LONG_POINTERS=1");				 \
builtin_define_with_int_value ("__CRTL_VER", vms_c_get_crtl_ver ()); \
builtin_define_with_int_value ("__VMS_VER", vms_c_get_vms_ver ());   \
} while (0)
extern void vms_c_register_includes (const char *, const char *, int);
#define TARGET_EXTRA_INCLUDES vms_c_register_includes
#define REGISTER_TARGET_PRAGMAS() vms_c_register_pragma ()
#define DOLLARS_IN_IDENTIFIERS 2
#undef TARGET_ABI_OPEN_VMS
#define TARGET_ABI_OPEN_VMS 1
#undef LONG_TYPE_SIZE
#define LONG_TYPE_SIZE 32
#define ADA_LONG_TYPE_SIZE 64
#undef POINTER_SIZE
#define POINTER_SIZE (flag_vms_pointer_size == VMS_POINTER_SIZE_NONE ? 32 : 64)
#define POINTERS_EXTEND_UNSIGNED 0
#undef SIZE_TYPE
#define SIZE_TYPE  "unsigned int"
#undef PTRDIFF_TYPE
#define PTRDIFF_TYPE (flag_vms_pointer_size == VMS_POINTER_SIZE_NONE ? \
"int" : "long long int")
#define SIZETYPE (flag_vms_pointer_size == VMS_POINTER_SIZE_NONE ? \
"unsigned int" : "long long unsigned int")
#define C_COMMON_OVERRIDE_OPTIONS vms_c_common_override_options ()
#define TARGET_ASM_FUNCTION_SECTION vms_function_section
#define DWARF2_ADDR_SIZE 8
#define MATH_LIBRARY ""
#define VMS_DEBUG_MAIN_POINTER "TRANSFER$BREAK$GO"
#undef TARGET_LIBC_HAS_FUNCTION
#define TARGET_LIBC_HAS_FUNCTION no_c99_libc_has_function
