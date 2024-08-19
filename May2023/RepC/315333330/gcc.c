#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "multilib.h" 
#include "tm.h"
#include "xregex.h"
#include "obstack.h"
#include "intl.h"
#include "prefix.h"
#include "gcc.h"
#include "diagnostic.h"
#include "flags.h"
#include "opts.h"
#include "params.h"
#include "filenames.h"
#include "spellcheck.h"

class env_manager
{
public:
void init (bool can_restore, bool debug);
const char *get (const char *name);
void xput (const char *string);
void restore ();
private:
bool m_can_restore;
bool m_debug;
struct kv
{
char *m_key;
char *m_value;
};
vec<kv> m_keys;
};
static env_manager env;
void
env_manager::init (bool can_restore, bool debug)
{
m_can_restore = can_restore;
m_debug = debug;
}
const char *
env_manager::get (const char *name)
{
const char *result = ::getenv (name);
if (m_debug)
fprintf (stderr, "env_manager::getenv (%s) -> %s\n", name, result);
return result;
}
void
env_manager::xput (const char *string)
{
if (m_debug)
fprintf (stderr, "env_manager::xput (%s)\n", string);
if (verbose_flag)
fnotice (stderr, "%s\n", string);
if (m_can_restore)
{
char *equals = strchr (const_cast <char *> (string), '=');
gcc_assert (equals);
struct kv kv;
kv.m_key = xstrndup (string, equals - string);
const char *cur_value = ::getenv (kv.m_key);
if (m_debug)
fprintf (stderr, "saving old value: %s\n",cur_value);
kv.m_value = cur_value ? xstrdup (cur_value) : NULL;
m_keys.safe_push (kv);
}
::putenv (CONST_CAST (char *, string));
}
void
env_manager::restore ()
{
unsigned int i;
struct kv *item;
gcc_assert (m_can_restore);
FOR_EACH_VEC_ELT_REVERSE (m_keys, i, item)
{
if (m_debug)
printf ("restoring saved key: %s value: %s\n", item->m_key, item->m_value);
if (item->m_value)
::setenv (item->m_key, item->m_value, 1);
else
::unsetenv (item->m_key);
free (item->m_key);
free (item->m_value);
}
m_keys.truncate (0);
}
#if (GCC_VERSION >= 3000)
#pragma GCC poison getenv putenv
#endif

#ifdef TARGET_EXECUTABLE_SUFFIX
#define HAVE_TARGET_EXECUTABLE_SUFFIX
#else
#define TARGET_EXECUTABLE_SUFFIX ""
#endif
#ifdef HOST_EXECUTABLE_SUFFIX
#define HAVE_HOST_EXECUTABLE_SUFFIX
#else
#define HOST_EXECUTABLE_SUFFIX ""
#endif
#ifdef TARGET_OBJECT_SUFFIX
#define HAVE_TARGET_OBJECT_SUFFIX
#else
#define TARGET_OBJECT_SUFFIX ".o"
#endif
static const char dir_separator_str[] = { DIR_SEPARATOR, 0 };
#ifndef LIBRARY_PATH_ENV
#define LIBRARY_PATH_ENV "LIBRARY_PATH"
#endif
#define MIN_FATAL_STATUS 1
int is_cpp_driver;
static bool at_file_supplied;
#include "configargs.h"
static int print_help_list;
static int print_version;
static int verbose_only_flag;
static int print_subprocess_help;
static const char *use_ld;
FILE *report_times_to_file = NULL;
#ifdef TARGET_SYSTEM_ROOT
#define DEFAULT_TARGET_SYSTEM_ROOT (TARGET_SYSTEM_ROOT)
#else
#define DEFAULT_TARGET_SYSTEM_ROOT (0)
#endif
static const char *target_system_root = DEFAULT_TARGET_SYSTEM_ROOT;
static int target_system_root_changed;
static const char *target_sysroot_suffix = 0;
static const char *target_sysroot_hdrs_suffix = 0;
static enum save_temps {
SAVE_TEMPS_NONE,		
SAVE_TEMPS_CWD,		
SAVE_TEMPS_OBJ		
} save_temps_flag;
static char *save_temps_prefix = 0;
static size_t save_temps_length = 0;
static const char *compiler_version;
static const char *const spec_version = DEFAULT_TARGET_VERSION;
static const char *spec_machine = DEFAULT_TARGET_MACHINE;
static const char *spec_host_machine = DEFAULT_REAL_TARGET_MACHINE;
static char *offload_targets = NULL;
#ifdef CROSS_DIRECTORY_STRUCTURE
static const char *cross_compile = "1";
#else
static const char *cross_compile = "0";
#endif
static int greatest_status = 1;
static struct obstack obstack;
static struct obstack collect_obstack;
struct path_prefix;
struct prefix_list;
static void init_spec (void);
static void store_arg (const char *, int, int);
static void insert_wrapper (const char *);
static char *load_specs (const char *);
static void read_specs (const char *, bool, bool);
static void set_spec (const char *, const char *, bool);
static struct compiler *lookup_compiler (const char *, size_t, const char *);
static char *build_search_list (const struct path_prefix *, const char *,
bool, bool);
static void xputenv (const char *);
static void putenv_from_prefixes (const struct path_prefix *, const char *,
bool);
static int access_check (const char *, int);
static char *find_a_file (const struct path_prefix *, const char *, int, bool);
static void add_prefix (struct path_prefix *, const char *, const char *,
int, int, int);
static void add_sysrooted_prefix (struct path_prefix *, const char *,
const char *, int, int, int);
static char *skip_whitespace (char *);
static void delete_if_ordinary (const char *);
static void delete_temp_files (void);
static void delete_failure_queue (void);
static void clear_failure_queue (void);
static int check_live_switch (int, int);
static const char *handle_braces (const char *);
static inline bool input_suffix_matches (const char *, const char *);
static inline bool switch_matches (const char *, const char *, int);
static inline void mark_matching_switches (const char *, const char *, int);
static inline void process_marked_switches (void);
static const char *process_brace_body (const char *, const char *, const char *, int, int);
static const struct spec_function *lookup_spec_function (const char *);
static const char *eval_spec_function (const char *, const char *);
static const char *handle_spec_function (const char *, bool *);
static char *save_string (const char *, int);
static void set_collect_gcc_options (void);
static int do_spec_1 (const char *, int, const char *);
static int do_spec_2 (const char *);
static void do_option_spec (const char *, const char *);
static void do_self_spec (const char *);
static const char *find_file (const char *);
static int is_directory (const char *, bool);
static const char *validate_switches (const char *, bool);
static void validate_all_switches (void);
static inline void validate_switches_from_spec (const char *, bool);
static void give_switch (int, int);
static int default_arg (const char *, int);
static void set_multilib_dir (void);
static void print_multilib_info (void);
static void perror_with_name (const char *);
static void display_help (void);
static void add_preprocessor_option (const char *, int);
static void add_assembler_option (const char *, int);
static void add_linker_option (const char *, int);
static void process_command (unsigned int, struct cl_decoded_option *);
static int execute (void);
static void alloc_args (void);
static void clear_args (void);
static void fatal_signal (int);
#if defined(ENABLE_SHARED_LIBGCC) && !defined(REAL_LIBGCC_SPEC)
static void init_gcc_specs (struct obstack *, const char *, const char *,
const char *);
#endif
#if defined(HAVE_TARGET_OBJECT_SUFFIX) || defined(HAVE_TARGET_EXECUTABLE_SUFFIX)
static const char *convert_filename (const char *, int, int);
#endif
static void try_generate_repro (const char **argv);
static const char *getenv_spec_function (int, const char **);
static const char *if_exists_spec_function (int, const char **);
static const char *if_exists_else_spec_function (int, const char **);
static const char *sanitize_spec_function (int, const char **);
static const char *replace_outfile_spec_function (int, const char **);
static const char *remove_outfile_spec_function (int, const char **);
static const char *version_compare_spec_function (int, const char **);
static const char *include_spec_function (int, const char **);
static const char *find_file_spec_function (int, const char **);
static const char *find_plugindir_spec_function (int, const char **);
static const char *print_asm_header_spec_function (int, const char **);
static const char *compare_debug_dump_opt_spec_function (int, const char **);
static const char *compare_debug_self_opt_spec_function (int, const char **);
static const char *compare_debug_auxbase_opt_spec_function (int, const char **);
static const char *pass_through_libs_spec_func (int, const char **);
static const char *replace_extension_spec_func (int, const char **);
static const char *greater_than_spec_func (int, const char **);
static const char *debug_level_greater_than_spec_func (int, const char **);
static char *convert_white_space (char *);


#ifndef ASM_SPEC
#define ASM_SPEC ""
#endif
#ifndef ASM_FINAL_SPEC
#define ASM_FINAL_SPEC \
"%{gsplit-dwarf: \n\
objcopy --extract-dwo \
%{c:%{o*:%*}%{!o*:%b%O}}%{!c:%U%O} \
%{c:%{o*:%:replace-extension(%{o*:%*} .dwo)}%{!o*:%b.dwo}}%{!c:%b.dwo} \n\
objcopy --strip-dwo \
%{c:%{o*:%*}%{!o*:%b%O}}%{!c:%U%O} \
}"
#endif
#ifndef CPP_SPEC
#define CPP_SPEC ""
#endif
#ifndef CC1_SPEC
#define CC1_SPEC ""
#endif
#ifndef CC1PLUS_SPEC
#define CC1PLUS_SPEC ""
#endif
#ifndef LINK_SPEC
#define LINK_SPEC ""
#endif
#ifndef LIB_SPEC
#define LIB_SPEC "%{!shared:%{g*:-lg} %{!p:%{!pg:-lc}}%{p:-lc_p}%{pg:-lc_p}}"
#endif
#ifdef HAVE_GOLD_NON_DEFAULT_SPLIT_STACK
#define STACK_SPLIT_SPEC " %{fsplit-stack: -fuse-ld=gold --wrap=pthread_create}"
#else
#define STACK_SPLIT_SPEC " %{fsplit-stack: --wrap=pthread_create}"
#endif
#ifndef LIBASAN_SPEC
#define STATIC_LIBASAN_LIBS \
" %{static-libasan|static:%:include(libsanitizer.spec)%(link_libasan)}"
#ifdef LIBASAN_EARLY_SPEC
#define LIBASAN_SPEC STATIC_LIBASAN_LIBS
#elif defined(HAVE_LD_STATIC_DYNAMIC)
#define LIBASAN_SPEC "%{static-libasan:" LD_STATIC_OPTION \
"} -lasan %{static-libasan:" LD_DYNAMIC_OPTION "}" \
STATIC_LIBASAN_LIBS
#else
#define LIBASAN_SPEC "-lasan" STATIC_LIBASAN_LIBS
#endif
#endif
#ifndef LIBASAN_EARLY_SPEC
#define LIBASAN_EARLY_SPEC ""
#endif
#ifndef LIBTSAN_SPEC
#define STATIC_LIBTSAN_LIBS \
" %{static-libtsan|static:%:include(libsanitizer.spec)%(link_libtsan)}"
#ifdef LIBTSAN_EARLY_SPEC
#define LIBTSAN_SPEC STATIC_LIBTSAN_LIBS
#elif defined(HAVE_LD_STATIC_DYNAMIC)
#define LIBTSAN_SPEC "%{static-libtsan:" LD_STATIC_OPTION \
"} -ltsan %{static-libtsan:" LD_DYNAMIC_OPTION "}" \
STATIC_LIBTSAN_LIBS
#else
#define LIBTSAN_SPEC "-ltsan" STATIC_LIBTSAN_LIBS
#endif
#endif
#ifndef LIBTSAN_EARLY_SPEC
#define LIBTSAN_EARLY_SPEC ""
#endif
#ifndef LIBLSAN_SPEC
#define STATIC_LIBLSAN_LIBS \
" %{static-liblsan|static:%:include(libsanitizer.spec)%(link_liblsan)}"
#ifdef LIBLSAN_EARLY_SPEC
#define LIBLSAN_SPEC STATIC_LIBLSAN_LIBS
#elif defined(HAVE_LD_STATIC_DYNAMIC)
#define LIBLSAN_SPEC "%{static-liblsan:" LD_STATIC_OPTION \
"} -llsan %{static-liblsan:" LD_DYNAMIC_OPTION "}" \
STATIC_LIBLSAN_LIBS
#else
#define LIBLSAN_SPEC "-llsan" STATIC_LIBLSAN_LIBS
#endif
#endif
#ifndef LIBLSAN_EARLY_SPEC
#define LIBLSAN_EARLY_SPEC ""
#endif
#ifndef LIBUBSAN_SPEC
#define STATIC_LIBUBSAN_LIBS \
" %{static-libubsan|static:%:include(libsanitizer.spec)%(link_libubsan)}"
#ifdef HAVE_LD_STATIC_DYNAMIC
#define LIBUBSAN_SPEC "%{static-libubsan:" LD_STATIC_OPTION \
"} -lubsan %{static-libubsan:" LD_DYNAMIC_OPTION "}" \
STATIC_LIBUBSAN_LIBS
#else
#define LIBUBSAN_SPEC "-lubsan" STATIC_LIBUBSAN_LIBS
#endif
#endif
#if HAVE_LD_COMPRESS_DEBUG == 0
#define LINK_COMPRESS_DEBUG_SPEC \
" %{gz*:%e-gz is not supported in this configuration} "
#elif HAVE_LD_COMPRESS_DEBUG == 1
#define LINK_COMPRESS_DEBUG_SPEC \
" %{gz*:%e-gz is not supported in this configuration} "
#elif HAVE_LD_COMPRESS_DEBUG == 2
#define LINK_COMPRESS_DEBUG_SPEC \
" %{gz|gz=zlib-gnu:" LD_COMPRESS_DEBUG_OPTION "=zlib}" \
" %{gz=none:"        LD_COMPRESS_DEBUG_OPTION "=none}" \
" %{gz=zlib:%e-gz=zlib is not supported in this configuration} "
#elif HAVE_LD_COMPRESS_DEBUG == 3
#define LINK_COMPRESS_DEBUG_SPEC \
" %{gz|gz=zlib:"  LD_COMPRESS_DEBUG_OPTION "=zlib}" \
" %{gz=none:"	  LD_COMPRESS_DEBUG_OPTION "=none}" \
" %{gz=zlib-gnu:" LD_COMPRESS_DEBUG_OPTION "=zlib-gnu} "
#else
#error Unknown value for HAVE_LD_COMPRESS_DEBUG.
#endif
#ifndef LIBGCC_SPEC
#if defined(REAL_LIBGCC_SPEC)
#define LIBGCC_SPEC REAL_LIBGCC_SPEC
#elif defined(LINK_LIBGCC_SPECIAL_1)
#define LIBGCC_SPEC "libgcc.a%s"
#else
#define LIBGCC_SPEC "-lgcc"
#endif
#endif
#ifndef STARTFILE_SPEC
#define STARTFILE_SPEC  \
"%{!shared:%{pg:gcrt0%O%s}%{!pg:%{p:mcrt0%O%s}%{!p:crt0%O%s}}}"
#endif
#ifndef ENDFILE_SPEC
#define ENDFILE_SPEC ""
#endif
#ifndef LINKER_NAME
#define LINKER_NAME "collect2"
#endif
#ifdef HAVE_AS_DEBUG_PREFIX_MAP
#define ASM_MAP " %{fdebug-prefix-map=*:--debug-prefix-map %*}"
#else
#define ASM_MAP ""
#endif
#if HAVE_LD_COMPRESS_DEBUG < 2
#define ASM_COMPRESS_DEBUG_SPEC \
" %{gz*:%e-gz is not supported in this configuration} "
#else 
#if HAVE_AS_COMPRESS_DEBUG == 0
#define ASM_COMPRESS_DEBUG_SPEC \
" %{gz*:} "
#elif HAVE_AS_COMPRESS_DEBUG == 1
#define ASM_COMPRESS_DEBUG_SPEC \
" %{gz|gz=zlib-gnu:" AS_COMPRESS_DEBUG_OPTION "}" \
" %{gz=none:"        AS_NO_COMPRESS_DEBUG_OPTION "}" \
" %{gz=zlib:%e-gz=zlib is not supported in this configuration} "
#elif HAVE_AS_COMPRESS_DEBUG == 2
#define ASM_COMPRESS_DEBUG_SPEC \
" %{gz|gz=zlib:"  AS_COMPRESS_DEBUG_OPTION "=zlib}" \
" %{gz=none:"	  AS_COMPRESS_DEBUG_OPTION "=none}" \
" %{gz=zlib-gnu:" AS_COMPRESS_DEBUG_OPTION "=zlib-gnu} "
#else
#error Unknown value for HAVE_AS_COMPRESS_DEBUG.
#endif
#endif 
#ifndef ASM_DEBUG_SPEC
# if defined(DBX_DEBUGGING_INFO) && defined(DWARF2_DEBUGGING_INFO) \
&& defined(HAVE_AS_GDWARF2_DEBUG_FLAG) && defined(HAVE_AS_GSTABS_DEBUG_FLAG)
#  define ASM_DEBUG_SPEC						\
(PREFERRED_DEBUGGING_TYPE == DBX_DEBUG				\
? "%{%:debug-level-gt(0):"					\
"%{gdwarf*:--gdwarf2}%{!gdwarf*:%{g*:--gstabs}}}" ASM_MAP	\
: "%{%:debug-level-gt(0):"					\
"%{gstabs*:--gstabs}%{!gstabs*:%{g*:--gdwarf2}}}" ASM_MAP)
# else
#  if defined(DBX_DEBUGGING_INFO) && defined(HAVE_AS_GSTABS_DEBUG_FLAG)
#   define ASM_DEBUG_SPEC "%{g*:%{%:debug-level-gt(0):--gstabs}}" ASM_MAP
#  endif
#  if defined(DWARF2_DEBUGGING_INFO) && defined(HAVE_AS_GDWARF2_DEBUG_FLAG)
#   define ASM_DEBUG_SPEC "%{g*:%{%:debug-level-gt(0):--gdwarf2}}" ASM_MAP
#  endif
# endif
#endif
#ifndef ASM_DEBUG_SPEC
# define ASM_DEBUG_SPEC ""
#endif
#ifndef LINK_GCC_C_SEQUENCE_SPEC
#define LINK_GCC_C_SEQUENCE_SPEC "%G %L %G"
#endif
#ifndef LINK_SSP_SPEC
#ifdef TARGET_LIBC_PROVIDES_SSP
#define LINK_SSP_SPEC "%{fstack-protector|fstack-protector-all" \
"|fstack-protector-strong|fstack-protector-explicit:}"
#else
#define LINK_SSP_SPEC "%{fstack-protector|fstack-protector-all" \
"|fstack-protector-strong|fstack-protector-explicit" \
":-lssp_nonshared -lssp}"
#endif
#endif
#ifdef ENABLE_DEFAULT_PIE
#define PIE_SPEC		"!no-pie"
#define NO_FPIE1_SPEC		"fno-pie"
#define FPIE1_SPEC		NO_FPIE1_SPEC ":;"
#define NO_FPIE2_SPEC		"fno-PIE"
#define FPIE2_SPEC		NO_FPIE2_SPEC ":;"
#define NO_FPIE_SPEC		NO_FPIE1_SPEC "|" NO_FPIE2_SPEC
#define FPIE_SPEC		NO_FPIE_SPEC ":;"
#define NO_FPIC1_SPEC		"fno-pic"
#define FPIC1_SPEC		NO_FPIC1_SPEC ":;"
#define NO_FPIC2_SPEC		"fno-PIC"
#define FPIC2_SPEC		NO_FPIC2_SPEC ":;"
#define NO_FPIC_SPEC		NO_FPIC1_SPEC "|" NO_FPIC2_SPEC
#define FPIC_SPEC		NO_FPIC_SPEC ":;"
#define NO_FPIE1_AND_FPIC1_SPEC	NO_FPIE1_SPEC "|" NO_FPIC1_SPEC
#define FPIE1_OR_FPIC1_SPEC	NO_FPIE1_AND_FPIC1_SPEC ":;"
#define NO_FPIE2_AND_FPIC2_SPEC	NO_FPIE2_SPEC "|" NO_FPIC2_SPEC
#define FPIE2_OR_FPIC2_SPEC	NO_FPIE2_AND_FPIC2_SPEC ":;"
#define NO_FPIE_AND_FPIC_SPEC	NO_FPIE_SPEC "|" NO_FPIC_SPEC
#define FPIE_OR_FPIC_SPEC	NO_FPIE_AND_FPIC_SPEC ":;"
#else
#define PIE_SPEC		"pie"
#define FPIE1_SPEC		"fpie"
#define NO_FPIE1_SPEC		FPIE1_SPEC ":;"
#define FPIE2_SPEC		"fPIE"
#define NO_FPIE2_SPEC		FPIE2_SPEC ":;"
#define FPIE_SPEC		FPIE1_SPEC "|" FPIE2_SPEC
#define NO_FPIE_SPEC		FPIE_SPEC ":;"
#define FPIC1_SPEC		"fpic"
#define NO_FPIC1_SPEC		FPIC1_SPEC ":;"
#define FPIC2_SPEC		"fPIC"
#define NO_FPIC2_SPEC		FPIC2_SPEC ":;"
#define FPIC_SPEC		FPIC1_SPEC "|" FPIC2_SPEC
#define NO_FPIC_SPEC		FPIC_SPEC ":;"
#define FPIE1_OR_FPIC1_SPEC	FPIE1_SPEC "|" FPIC1_SPEC
#define NO_FPIE1_AND_FPIC1_SPEC	FPIE1_OR_FPIC1_SPEC ":;"
#define FPIE2_OR_FPIC2_SPEC	FPIE2_SPEC "|" FPIC2_SPEC
#define NO_FPIE2_AND_FPIC2_SPEC	FPIE1_OR_FPIC2_SPEC ":;"
#define FPIE_OR_FPIC_SPEC	FPIE_SPEC "|" FPIC_SPEC
#define NO_FPIE_AND_FPIC_SPEC	FPIE_OR_FPIC_SPEC ":;"
#endif
#ifndef LINK_PIE_SPEC
#ifdef HAVE_LD_PIE
#ifndef LD_PIE_SPEC
#define LD_PIE_SPEC "-pie"
#endif
#else
#define LD_PIE_SPEC ""
#endif
#define LINK_PIE_SPEC "%{static|shared|r:;" PIE_SPEC ":" LD_PIE_SPEC "} "
#endif
#ifndef LINK_BUILDID_SPEC
# if defined(HAVE_LD_BUILDID) && defined(ENABLE_LD_BUILDID)
#  define LINK_BUILDID_SPEC "%{!r:--build-id} "
# endif
#endif
#if HAVE_LTO_PLUGIN > 0
#if HAVE_LTO_PLUGIN == 2
#define PLUGIN_COND "!fno-use-linker-plugin:%{!fno-lto"
#define PLUGIN_COND_CLOSE "}"
#else
#define PLUGIN_COND "fuse-linker-plugin"
#define PLUGIN_COND_CLOSE ""
#endif
#define LINK_PLUGIN_SPEC \
"%{" PLUGIN_COND": \
-plugin %(linker_plugin_file) \
-plugin-opt=%(lto_wrapper) \
-plugin-opt=-fresolution=%u.res \
%{!nostdlib:%{!nodefaultlibs:%:pass-through-libs(%(link_gcc_c_sequence))}} \
}" PLUGIN_COND_CLOSE
#else
#define LINK_PLUGIN_SPEC "%{fuse-linker-plugin:\
%e-fuse-linker-plugin is not supported in this configuration}"
#endif
#ifndef SANITIZER_EARLY_SPEC
#define SANITIZER_EARLY_SPEC "\
%{!nostdlib:%{!nodefaultlibs:%{%:sanitize(address):" LIBASAN_EARLY_SPEC "} \
%{%:sanitize(thread):" LIBTSAN_EARLY_SPEC "} \
%{%:sanitize(leak):" LIBLSAN_EARLY_SPEC "}}}"
#endif
#ifndef SANITIZER_SPEC
#define SANITIZER_SPEC "\
%{!nostdlib:%{!nodefaultlibs:%{%:sanitize(address):" LIBASAN_SPEC "\
%{static:%ecannot specify -static with -fsanitize=address}}\
%{%:sanitize(thread):" LIBTSAN_SPEC "\
%{static:%ecannot specify -static with -fsanitize=thread}}\
%{%:sanitize(undefined):" LIBUBSAN_SPEC "}\
%{%:sanitize(leak):" LIBLSAN_SPEC "}}}"
#endif
#ifndef POST_LINK_SPEC
#define POST_LINK_SPEC ""
#endif
#ifndef VTABLE_VERIFICATION_SPEC
#if ENABLE_VTABLE_VERIFY
#define VTABLE_VERIFICATION_SPEC "\
%{!nostdlib:%{fvtable-verify=std: -lvtv -u_vtable_map_vars_start -u_vtable_map_vars_end}\
%{fvtable-verify=preinit: -lvtv -u_vtable_map_vars_start -u_vtable_map_vars_end}}"
#else
#define VTABLE_VERIFICATION_SPEC "\
%{fvtable-verify=none:} \
%{fvtable-verify=std: \
%e-fvtable-verify=std is not supported in this configuration} \
%{fvtable-verify=preinit: \
%e-fvtable-verify=preinit is not supported in this configuration}"
#endif
#endif
#ifndef CHKP_SPEC
#define CHKP_SPEC ""
#endif
#ifndef LINK_COMMAND_SPEC
#define LINK_COMMAND_SPEC "\
%{!fsyntax-only:%{!c:%{!M:%{!MM:%{!E:%{!S:\
%(linker) " \
LINK_PLUGIN_SPEC \
"%{flto|flto=*:%<fcompare-debug*} \
%{flto} %{fno-lto} %{flto=*} %l " LINK_PIE_SPEC \
"%{fuse-ld=*:-fuse-ld=%*} " LINK_COMPRESS_DEBUG_SPEC \
"%X %{o*} %{e*} %{N} %{n} %{r}\
%{s} %{t} %{u*} %{z} %{Z} %{!nostdlib:%{!nostartfiles:%S}} \
%{static|no-pie|static-pie:} %{L*} %(mfwrap) %(link_libgcc) " \
VTABLE_VERIFICATION_SPEC " " SANITIZER_EARLY_SPEC " %o " CHKP_SPEC " \
%{fopenacc|fopenmp|%:gt(%{ftree-parallelize-loops=*:%*} 1):\
%:include(libgomp.spec)%(link_gomp)}\
%{fgnu-tm:%:include(libitm.spec)%(link_itm)}\
%(mflib) " STACK_SPLIT_SPEC "\
%{fprofile-arcs|fprofile-generate*|coverage:-lgcov} " SANITIZER_SPEC " \
%{!nostdlib:%{!nodefaultlibs:%(link_ssp) %(link_gcc_c_sequence)}}\
%{!nostdlib:%{!nostartfiles:%E}} %{T*}  \n%(post_link) }}}}}}"
#endif
#ifndef LINK_LIBGCC_SPEC
# define LINK_LIBGCC_SPEC "%D"
#endif
#ifndef STARTFILE_PREFIX_SPEC
# define STARTFILE_PREFIX_SPEC ""
#endif
#ifndef SYSROOT_SPEC
# define SYSROOT_SPEC "--sysroot=%R"
#endif
#ifndef SYSROOT_SUFFIX_SPEC
# define SYSROOT_SUFFIX_SPEC ""
#endif
#ifndef SYSROOT_HEADERS_SUFFIX_SPEC
# define SYSROOT_HEADERS_SUFFIX_SPEC ""
#endif
static const char *asm_debug = ASM_DEBUG_SPEC;
static const char *cpp_spec = CPP_SPEC;
static const char *cc1_spec = CC1_SPEC;
static const char *cc1plus_spec = CC1PLUS_SPEC;
static const char *link_gcc_c_sequence_spec = LINK_GCC_C_SEQUENCE_SPEC;
static const char *link_ssp_spec = LINK_SSP_SPEC;
static const char *asm_spec = ASM_SPEC;
static const char *asm_final_spec = ASM_FINAL_SPEC;
static const char *link_spec = LINK_SPEC;
static const char *lib_spec = LIB_SPEC;
static const char *link_gomp_spec = "";
static const char *libgcc_spec = LIBGCC_SPEC;
static const char *endfile_spec = ENDFILE_SPEC;
static const char *startfile_spec = STARTFILE_SPEC;
static const char *linker_name_spec = LINKER_NAME;
static const char *linker_plugin_file_spec = "";
static const char *lto_wrapper_spec = "";
static const char *lto_gcc_spec = "";
static const char *post_link_spec = POST_LINK_SPEC;
static const char *link_command_spec = LINK_COMMAND_SPEC;
static const char *link_libgcc_spec = LINK_LIBGCC_SPEC;
static const char *startfile_prefix_spec = STARTFILE_PREFIX_SPEC;
static const char *sysroot_spec = SYSROOT_SPEC;
static const char *sysroot_suffix_spec = SYSROOT_SUFFIX_SPEC;
static const char *sysroot_hdrs_suffix_spec = SYSROOT_HEADERS_SUFFIX_SPEC;
static const char *self_spec = "";
static const char *trad_capable_cpp =
"cc1 -E %{traditional|traditional-cpp:-traditional-cpp}";
static const char *cpp_unique_options =
"%{!Q:-quiet} %{nostdinc*} %{C} %{CC} %{v} %{I*&F*} %{P} %I\
%{MD:-MD %{!o:%b.d}%{o*:%.d%*}}\
%{MMD:-MMD %{!o:%b.d}%{o*:%.d%*}}\
%{M} %{MM} %{MF*} %{MG} %{MP} %{MQ*} %{MT*}\
%{!E:%{!M:%{!MM:%{!MT:%{!MQ:%{MD|MMD:%{o*:-MQ %*}}}}}}}\
%{remap} %{g3|ggdb3|gstabs3|gxcoff3|gvms3:-dD}\
%{!iplugindir*:%{fplugin*:%:find-plugindir()}}\
%{H} %C %{D*&U*&A*} %{i*} %Z %i\
%{E|M|MM:%W{o*}}";
static const char *cpp_options =
"%(cpp_unique_options) %1 %{m*} %{std*&ansi&trigraphs} %{W*&pedantic*} %{w}\
%{f*} %{g*:%{%:debug-level-gt(0):%{g*}\
%{!fno-working-directory:-fworking-directory}}} %{O*}\
%{undef} %{save-temps*:-fpch-preprocess}";
static const char *cpp_debug_options = "%{d*}";
static const char *cc1_options =
"%{pg:%{fomit-frame-pointer:%e-pg and -fomit-frame-pointer are incompatible}}\
%{!iplugindir*:%{fplugin*:%:find-plugindir()}}\
%1 %{!Q:-quiet} %{!dumpbase:-dumpbase %B} %{d*} %{m*} %{aux-info*}\
%{fcompare-debug-second:%:compare-debug-auxbase-opt(%b)} \
%{!fcompare-debug-second:%{c|S:%{o*:-auxbase-strip %*}%{!o*:-auxbase %b}}}%{!c:%{!S:-auxbase %b}} \
%{g*} %{O*} %{W*&pedantic*} %{w} %{std*&ansi&trigraphs}\
%{v:-version} %{pg:-p} %{p} %{f*} %{undef}\
%{Qn:-fno-ident} %{Qy:} %{-help:--help}\
%{-target-help:--target-help}\
%{-version:--version}\
%{-help=*:--help=%*}\
%{!fsyntax-only:%{S:%W{o*}%{!o*:-o %b.s}}}\
%{fsyntax-only:-o %j} %{-param*}\
%{coverage:-fprofile-arcs -ftest-coverage}\
%{fprofile-arcs|fprofile-generate*|coverage:\
%{!fprofile-update=single:\
%{pthread:-fprofile-update=prefer-atomic}}}";
static const char *asm_options =
"%{-target-help:%:print-asm-header()} "
#if HAVE_GNU_AS
"%{v} %{w:-W} %{I*} "
#endif
ASM_COMPRESS_DEBUG_SPEC
"%a %Y %{c:%W{o*}%{!o*:-o %w%b%O}}%{!c:-o %d%w%u%O}";
static const char *invoke_as =
#ifdef AS_NEEDS_DASH_FOR_PIPED_INPUT
"%{!fwpa*:\
%{fcompare-debug=*|fdump-final-insns=*:%:compare-debug-dump-opt()}\
%{!S:-o %|.s |\n as %(asm_options) %|.s %A }\
}";
#else
"%{!fwpa*:\
%{fcompare-debug=*|fdump-final-insns=*:%:compare-debug-dump-opt()}\
%{!S:-o %|.s |\n as %(asm_options) %m.s %A }\
}";
#endif
static struct obstack multilib_obstack;
static const char *multilib_select;
static const char *multilib_matches;
static const char *multilib_defaults;
static const char *multilib_exclusions;
static const char *multilib_reuse;
#ifndef MULTILIB_DEFAULTS
#define MULTILIB_DEFAULTS { "" }
#endif
static const char *const multilib_defaults_raw[] = MULTILIB_DEFAULTS;
#ifndef DRIVER_SELF_SPECS
#define DRIVER_SELF_SPECS ""
#endif
#ifndef GOMP_SELF_SPECS
#define GOMP_SELF_SPECS \
"%{fopenacc|fopenmp|%:gt(%{ftree-parallelize-loops=*:%*} 1): " \
"-pthread}"
#endif
#ifndef GTM_SELF_SPECS
#define GTM_SELF_SPECS "%{fgnu-tm: -pthread}"
#endif
static const char *const driver_self_specs[] = {
"%{fdump-final-insns:-fdump-final-insns=.} %<fdump-final-insns",
DRIVER_SELF_SPECS, CONFIGURE_SPECS, GOMP_SELF_SPECS, GTM_SELF_SPECS
};
#ifndef OPTION_DEFAULT_SPECS
#define OPTION_DEFAULT_SPECS { "", "" }
#endif
struct default_spec
{
const char *name;
const char *spec;
};
static const struct default_spec
option_default_specs[] = { OPTION_DEFAULT_SPECS };
struct user_specs
{
struct user_specs *next;
const char *filename;
};
static struct user_specs *user_specs_head, *user_specs_tail;

struct compiler
{
const char *suffix;		
const char *spec;		
const char *cpp_spec;         
int combinable;               
int needs_preprocessing;       
};
static struct compiler *compilers;
static int n_compilers;
static const struct compiler default_compilers[] =
{
{".m",  "#Objective-C", 0, 0, 0}, {".mi",  "#Objective-C", 0, 0, 0},
{".mm", "#Objective-C++", 0, 0, 0}, {".M", "#Objective-C++", 0, 0, 0},
{".mii", "#Objective-C++", 0, 0, 0},
{".cc", "#C++", 0, 0, 0}, {".cxx", "#C++", 0, 0, 0},
{".cpp", "#C++", 0, 0, 0}, {".cp", "#C++", 0, 0, 0},
{".c++", "#C++", 0, 0, 0}, {".C", "#C++", 0, 0, 0},
{".CPP", "#C++", 0, 0, 0}, {".ii", "#C++", 0, 0, 0},
{".ads", "#Ada", 0, 0, 0}, {".adb", "#Ada", 0, 0, 0},
{".f", "#Fortran", 0, 0, 0}, {".F", "#Fortran", 0, 0, 0},
{".for", "#Fortran", 0, 0, 0}, {".FOR", "#Fortran", 0, 0, 0},
{".ftn", "#Fortran", 0, 0, 0}, {".FTN", "#Fortran", 0, 0, 0},
{".fpp", "#Fortran", 0, 0, 0}, {".FPP", "#Fortran", 0, 0, 0},
{".f90", "#Fortran", 0, 0, 0}, {".F90", "#Fortran", 0, 0, 0},
{".f95", "#Fortran", 0, 0, 0}, {".F95", "#Fortran", 0, 0, 0},
{".f03", "#Fortran", 0, 0, 0}, {".F03", "#Fortran", 0, 0, 0},
{".f08", "#Fortran", 0, 0, 0}, {".F08", "#Fortran", 0, 0, 0},
{".r", "#Ratfor", 0, 0, 0},
{".go", "#Go", 0, 1, 0},
{".c", "@c", 0, 0, 1},
{"@c",
"%{E|M|MM:%(trad_capable_cpp) %(cpp_options) %(cpp_debug_options)}\
%{!E:%{!M:%{!MM:\
%{traditional:\
%eGNU C no longer supports -traditional without -E}\
%{save-temps*|traditional-cpp|no-integrated-cpp:%(trad_capable_cpp) \
%(cpp_options) -o %{save-temps*:%b.i} %{!save-temps*:%g.i} \n\
cc1 -fpreprocessed %{save-temps*:%b.i} %{!save-temps*:%g.i} \
%(cc1_options)}\
%{!save-temps*:%{!traditional-cpp:%{!no-integrated-cpp:\
cc1 %(cpp_unique_options) %(cc1_options)}}}\
%{!fsyntax-only:%(invoke_as)}}}}", 0, 0, 1},
{"-",
"%{!E:%e-E or -x required when input is from standard input}\
%(trad_capable_cpp) %(cpp_options) %(cpp_debug_options)", 0, 0, 0},
{".h", "@c-header", 0, 0, 0},
{"@c-header",
"%{E|M|MM:%(trad_capable_cpp) %(cpp_options) %(cpp_debug_options)}\
%{!E:%{!M:%{!MM:\
%{save-temps*|traditional-cpp|no-integrated-cpp:%(trad_capable_cpp) \
%(cpp_options) -o %{save-temps*:%b.i} %{!save-temps*:%g.i} \n\
cc1 -fpreprocessed %{save-temps*:%b.i} %{!save-temps*:%g.i} \
%(cc1_options)\
%{!fsyntax-only:%{!S:-o %g.s} \
%{!fdump-ada-spec*:%{!o*:--output-pch=%i.gch}\
%W{o*:--output-pch=%*}}%V}}\
%{!save-temps*:%{!traditional-cpp:%{!no-integrated-cpp:\
cc1 %(cpp_unique_options) %(cc1_options)\
%{!fsyntax-only:%{!S:-o %g.s} \
%{!fdump-ada-spec*:%{!o*:--output-pch=%i.gch}\
%W{o*:--output-pch=%*}}%V}}}}}}}", 0, 0, 0},
{".i", "@cpp-output", 0, 0, 0},
{"@cpp-output",
"%{!M:%{!MM:%{!E:cc1 -fpreprocessed %i %(cc1_options) %{!fsyntax-only:%(invoke_as)}}}}", 0, 0, 0},
{".s", "@assembler", 0, 0, 0},
{"@assembler",
"%{!M:%{!MM:%{!E:%{!S:as %(asm_debug) %(asm_options) %i %A }}}}", 0, 0, 0},
{".sx", "@assembler-with-cpp", 0, 0, 0},
{".S", "@assembler-with-cpp", 0, 0, 0},
{"@assembler-with-cpp",
#ifdef AS_NEEDS_DASH_FOR_PIPED_INPUT
"%(trad_capable_cpp) -lang-asm %(cpp_options) -fno-directives-only\
%{E|M|MM:%(cpp_debug_options)}\
%{!M:%{!MM:%{!E:%{!S:-o %|.s |\n\
as %(asm_debug) %(asm_options) %|.s %A }}}}"
#else
"%(trad_capable_cpp) -lang-asm %(cpp_options) -fno-directives-only\
%{E|M|MM:%(cpp_debug_options)}\
%{!M:%{!MM:%{!E:%{!S:-o %|.s |\n\
as %(asm_debug) %(asm_options) %m.s %A }}}}"
#endif
, 0, 0, 0},
#include "specs.h"
{0, 0, 0, 0, 0}
};
static const int n_default_compilers = ARRAY_SIZE (default_compilers) - 1;
typedef char *char_p; 
static vec<char_p> linker_options;
static vec<char_p> assembler_options;
static vec<char_p> preprocessor_options;

static char *
skip_whitespace (char *p)
{
while (1)
{
if (p[0] == '\n' && p[1] == '\n' && p[2] == '\n')
return p + 1;
else if (*p == '\n' || *p == ' ' || *p == '\t')
p++;
else if (*p == '#')
{
while (*p != '\n')
p++;
p++;
}
else
break;
}
return p;
}
struct prefix_list
{
const char *prefix;	      
struct prefix_list *next;   
int require_machine_suffix; 
int priority;		      
int os_multilib;	      
};
struct path_prefix
{
struct prefix_list *plist;  
int max_len;                
const char *name;           
};
static struct path_prefix exec_prefixes = { 0, 0, "exec" };
static struct path_prefix startfile_prefixes = { 0, 0, "startfile" };
static struct path_prefix include_prefixes = { 0, 0, "include" };
static const char *machine_suffix = 0;
static const char *just_machine_suffix = 0;
static const char *gcc_exec_prefix;
static const char *gcc_libexec_prefix;
#ifndef STANDARD_STARTFILE_PREFIX_1
#define STANDARD_STARTFILE_PREFIX_1 "/lib/"
#endif
#ifndef STANDARD_STARTFILE_PREFIX_2
#define STANDARD_STARTFILE_PREFIX_2 "/usr/lib/"
#endif
#ifdef CROSS_DIRECTORY_STRUCTURE  
#undef MD_EXEC_PREFIX
#undef MD_STARTFILE_PREFIX
#undef MD_STARTFILE_PREFIX_1
#endif
#ifndef MD_EXEC_PREFIX
#define MD_EXEC_PREFIX ""
#endif
#ifndef MD_STARTFILE_PREFIX
#define MD_STARTFILE_PREFIX ""
#endif
#ifndef MD_STARTFILE_PREFIX_1
#define MD_STARTFILE_PREFIX_1 ""
#endif
static const char *const standard_exec_prefix = STANDARD_EXEC_PREFIX;
static const char *const standard_libexec_prefix = STANDARD_LIBEXEC_PREFIX;
static const char *const standard_bindir_prefix = STANDARD_BINDIR_PREFIX;
static const char *const standard_startfile_prefix = STANDARD_STARTFILE_PREFIX;
static const char *md_exec_prefix = MD_EXEC_PREFIX;
static const char *md_startfile_prefix = MD_STARTFILE_PREFIX;
static const char *md_startfile_prefix_1 = MD_STARTFILE_PREFIX_1;
static const char *const standard_startfile_prefix_1
= STANDARD_STARTFILE_PREFIX_1;
static const char *const standard_startfile_prefix_2
= STANDARD_STARTFILE_PREFIX_2;
static const char *const tooldir_base_prefix = TOOLDIR_BASE_PREFIX;
static const char *const accel_dir_suffix = ACCEL_DIR_SUFFIX;
static const char *multilib_dir;
static const char *multilib_os_dir;
static const char *multiarch_dir;

struct spec_list
{
const char *name;		
const char *ptr;		
const char **ptr_spec;	
struct spec_list *next;	
int name_len;			
bool user_p;			
bool alloc_p;			
const char *default_ptr;	
};
#define INIT_STATIC_SPEC(NAME,PTR) \
{ NAME, NULL, PTR, (struct spec_list *) 0, sizeof (NAME) - 1, false, false, \
*PTR }
static struct spec_list static_specs[] =
{
INIT_STATIC_SPEC ("asm",			&asm_spec),
INIT_STATIC_SPEC ("asm_debug",		&asm_debug),
INIT_STATIC_SPEC ("asm_final",		&asm_final_spec),
INIT_STATIC_SPEC ("asm_options",		&asm_options),
INIT_STATIC_SPEC ("invoke_as",		&invoke_as),
INIT_STATIC_SPEC ("cpp",			&cpp_spec),
INIT_STATIC_SPEC ("cpp_options",		&cpp_options),
INIT_STATIC_SPEC ("cpp_debug_options",	&cpp_debug_options),
INIT_STATIC_SPEC ("cpp_unique_options",	&cpp_unique_options),
INIT_STATIC_SPEC ("trad_capable_cpp",		&trad_capable_cpp),
INIT_STATIC_SPEC ("cc1",			&cc1_spec),
INIT_STATIC_SPEC ("cc1_options",		&cc1_options),
INIT_STATIC_SPEC ("cc1plus",			&cc1plus_spec),
INIT_STATIC_SPEC ("link_gcc_c_sequence",	&link_gcc_c_sequence_spec),
INIT_STATIC_SPEC ("link_ssp",			&link_ssp_spec),
INIT_STATIC_SPEC ("endfile",			&endfile_spec),
INIT_STATIC_SPEC ("link",			&link_spec),
INIT_STATIC_SPEC ("lib",			&lib_spec),
INIT_STATIC_SPEC ("link_gomp",		&link_gomp_spec),
INIT_STATIC_SPEC ("libgcc",			&libgcc_spec),
INIT_STATIC_SPEC ("startfile",		&startfile_spec),
INIT_STATIC_SPEC ("cross_compile",		&cross_compile),
INIT_STATIC_SPEC ("version",			&compiler_version),
INIT_STATIC_SPEC ("multilib",			&multilib_select),
INIT_STATIC_SPEC ("multilib_defaults",	&multilib_defaults),
INIT_STATIC_SPEC ("multilib_extra",		&multilib_extra),
INIT_STATIC_SPEC ("multilib_matches",		&multilib_matches),
INIT_STATIC_SPEC ("multilib_exclusions",	&multilib_exclusions),
INIT_STATIC_SPEC ("multilib_options",		&multilib_options),
INIT_STATIC_SPEC ("multilib_reuse",		&multilib_reuse),
INIT_STATIC_SPEC ("linker",			&linker_name_spec),
INIT_STATIC_SPEC ("linker_plugin_file",	&linker_plugin_file_spec),
INIT_STATIC_SPEC ("lto_wrapper",		&lto_wrapper_spec),
INIT_STATIC_SPEC ("lto_gcc",			&lto_gcc_spec),
INIT_STATIC_SPEC ("post_link",		&post_link_spec),
INIT_STATIC_SPEC ("link_libgcc",		&link_libgcc_spec),
INIT_STATIC_SPEC ("md_exec_prefix",		&md_exec_prefix),
INIT_STATIC_SPEC ("md_startfile_prefix",	&md_startfile_prefix),
INIT_STATIC_SPEC ("md_startfile_prefix_1",	&md_startfile_prefix_1),
INIT_STATIC_SPEC ("startfile_prefix_spec",	&startfile_prefix_spec),
INIT_STATIC_SPEC ("sysroot_spec",             &sysroot_spec),
INIT_STATIC_SPEC ("sysroot_suffix_spec",	&sysroot_suffix_spec),
INIT_STATIC_SPEC ("sysroot_hdrs_suffix_spec",	&sysroot_hdrs_suffix_spec),
INIT_STATIC_SPEC ("self_spec",		&self_spec),
};
#ifdef EXTRA_SPECS		
struct spec_list_1
{
const char *const name;
const char *const ptr;
};
static const struct spec_list_1 extra_specs_1[] = { EXTRA_SPECS };
static struct spec_list *extra_specs = (struct spec_list *) 0;
#endif
static struct spec_list *specs = (struct spec_list *) 0;

static const struct spec_function static_spec_functions[] =
{
{ "getenv",                   getenv_spec_function },
{ "if-exists",		if_exists_spec_function },
{ "if-exists-else",		if_exists_else_spec_function },
{ "sanitize",			sanitize_spec_function },
{ "replace-outfile",		replace_outfile_spec_function },
{ "remove-outfile",		remove_outfile_spec_function },
{ "version-compare",		version_compare_spec_function },
{ "include",			include_spec_function },
{ "find-file",		find_file_spec_function },
{ "find-plugindir",		find_plugindir_spec_function },
{ "print-asm-header",		print_asm_header_spec_function },
{ "compare-debug-dump-opt",	compare_debug_dump_opt_spec_function },
{ "compare-debug-self-opt",	compare_debug_self_opt_spec_function },
{ "compare-debug-auxbase-opt", compare_debug_auxbase_opt_spec_function },
{ "pass-through-libs",	pass_through_libs_spec_func },
{ "replace-extension",	replace_extension_spec_func },
{ "gt",			greater_than_spec_func },
{ "debug-level-gt",		debug_level_greater_than_spec_func },
#ifdef EXTRA_SPEC_FUNCTIONS
EXTRA_SPEC_FUNCTIONS
#endif
{ 0, 0 }
};
static int processing_spec_function;

#if defined(ENABLE_SHARED_LIBGCC) && !defined(REAL_LIBGCC_SPEC)
#ifndef USE_LD_AS_NEEDED
#define USE_LD_AS_NEEDED 0
#endif
static void
init_gcc_specs (struct obstack *obstack, const char *shared_name,
const char *static_name, const char *eh_name)
{
char *buf;
#if USE_LD_AS_NEEDED
buf = concat ("%{static|static-libgcc|static-pie:", static_name, " ", eh_name, "}"
"%{!static:%{!static-libgcc:%{!static-pie:"
"%{!shared-libgcc:",
static_name, " " LD_AS_NEEDED_OPTION " ",
shared_name, " " LD_NO_AS_NEEDED_OPTION
"}"
"%{shared-libgcc:",
shared_name, "%{!shared: ", static_name, "}"
"}}"
#else
buf = concat ("%{static|static-libgcc:", static_name, " ", eh_name, "}"
"%{!static:%{!static-libgcc:"
"%{!shared:"
"%{!shared-libgcc:", static_name, " ", eh_name, "}"
"%{shared-libgcc:", shared_name, " ", static_name, "}"
"}"
#ifdef LINK_EH_SPEC
"%{shared:"
"%{shared-libgcc:", shared_name, "}"
"%{!shared-libgcc:", static_name, "}"
"}"
#else
"%{shared:", shared_name, "}"
#endif
#endif
"}}", NULL);
obstack_grow (obstack, buf, strlen (buf));
free (buf);
}
#endif 
static void
init_spec (void)
{
struct spec_list *next = (struct spec_list *) 0;
struct spec_list *sl   = (struct spec_list *) 0;
int i;
if (specs)
return;			
if (verbose_flag)
fnotice (stderr, "Using built-in specs.\n");
#ifdef EXTRA_SPECS
extra_specs = XCNEWVEC (struct spec_list, ARRAY_SIZE (extra_specs_1));
for (i = ARRAY_SIZE (extra_specs_1) - 1; i >= 0; i--)
{
sl = &extra_specs[i];
sl->name = extra_specs_1[i].name;
sl->ptr = extra_specs_1[i].ptr;
sl->next = next;
sl->name_len = strlen (sl->name);
sl->ptr_spec = &sl->ptr;
gcc_assert (sl->ptr_spec != NULL);
sl->default_ptr = sl->ptr;
next = sl;
}
#endif
for (i = ARRAY_SIZE (static_specs) - 1; i >= 0; i--)
{
sl = &static_specs[i];
sl->next = next;
next = sl;
}
#if defined(ENABLE_SHARED_LIBGCC) && !defined(REAL_LIBGCC_SPEC)
{
const char *p = libgcc_spec;
int in_sep = 1;
while (*p)
{
if (in_sep && *p == '-' && strncmp (p, "-lgcc", 5) == 0)
{
init_gcc_specs (&obstack,
"-lgcc_s"
#ifdef USE_LIBUNWIND_EXCEPTIONS
" -lunwind"
#endif
,
"-lgcc",
"-lgcc_eh"
#ifdef USE_LIBUNWIND_EXCEPTIONS
# ifdef HAVE_LD_STATIC_DYNAMIC
" %{!static:%{!static-pie:" LD_STATIC_OPTION "}} -lunwind"
" %{!static:%{!static-pie:" LD_DYNAMIC_OPTION "}}"
# else
" -lunwind"
# endif
#endif
);
p += 5;
in_sep = 0;
}
else if (in_sep && *p == 'l' && strncmp (p, "libgcc.a%s", 10) == 0)
{
init_gcc_specs (&obstack,
"-lgcc_s",
"libgcc.a%s",
"libgcc_eh.a%s"
#ifdef USE_LIBUNWIND_EXCEPTIONS
" -lunwind"
#endif
);
p += 10;
in_sep = 0;
}
else
{
obstack_1grow (&obstack, *p);
in_sep = (*p == ' ');
p += 1;
}
}
obstack_1grow (&obstack, '\0');
libgcc_spec = XOBFINISH (&obstack, const char *);
}
#endif
#ifdef USE_AS_TRADITIONAL_FORMAT
{
static const char tf[] = "--traditional-format ";
obstack_grow (&obstack, tf, sizeof (tf) - 1);
obstack_grow0 (&obstack, asm_spec, strlen (asm_spec));
asm_spec = XOBFINISH (&obstack, const char *);
}
#endif
#if defined LINK_EH_SPEC || defined LINK_BUILDID_SPEC || \
defined LINKER_HASH_STYLE
# ifdef LINK_BUILDID_SPEC
obstack_grow (&obstack, LINK_BUILDID_SPEC, sizeof (LINK_BUILDID_SPEC) - 1);
# endif
# ifdef LINK_EH_SPEC
obstack_grow (&obstack, LINK_EH_SPEC, sizeof (LINK_EH_SPEC) - 1);
# endif
# ifdef LINKER_HASH_STYLE
{
static const char hash_style[] = "--hash-style=";
obstack_grow (&obstack, hash_style, sizeof (hash_style) - 1);
obstack_grow (&obstack, LINKER_HASH_STYLE, sizeof (LINKER_HASH_STYLE) - 1);
obstack_1grow (&obstack, ' ');
}
# endif
obstack_grow0 (&obstack, link_spec, strlen (link_spec));
link_spec = XOBFINISH (&obstack, const char *);
#endif
specs = sl;
}

static void
set_spec (const char *name, const char *spec, bool user_p)
{
struct spec_list *sl;
const char *old_spec;
int name_len = strlen (name);
int i;
if (!specs)
{
struct spec_list *next = (struct spec_list *) 0;
for (i = ARRAY_SIZE (static_specs) - 1; i >= 0; i--)
{
sl = &static_specs[i];
sl->next = next;
next = sl;
}
specs = sl;
}
for (sl = specs; sl; sl = sl->next)
if (name_len == sl->name_len && !strcmp (sl->name, name))
break;
if (!sl)
{
sl = XNEW (struct spec_list);
sl->name = xstrdup (name);
sl->name_len = name_len;
sl->ptr_spec = &sl->ptr;
sl->alloc_p = 0;
*(sl->ptr_spec) = "";
sl->next = specs;
sl->default_ptr = NULL;
specs = sl;
}
old_spec = *(sl->ptr_spec);
*(sl->ptr_spec) = ((spec[0] == '+' && ISSPACE ((unsigned char)spec[1]))
? concat (old_spec, spec + 1, NULL)
: xstrdup (spec));
#ifdef DEBUG_SPECS
if (verbose_flag)
fnotice (stderr, "Setting spec %s to '%s'\n\n", name, *(sl->ptr_spec));
#endif
if (old_spec && sl->alloc_p)
free (CONST_CAST (char *, old_spec));
sl->user_p = user_p;
sl->alloc_p = true;
}

typedef const char *const_char_p; 
static vec<const_char_p> argbuf;
static int have_c = 0;
static int have_o = 0;
static int have_E = 0;
static const char *output_file = 0;
static struct temp_name {
const char *suffix;	
int length;		
int unique;		
const char *filename;	
int filename_length;	
struct temp_name *next;
} *temp_names;
static int execution_count;
static int signal_count;

static void
alloc_args (void)
{
argbuf.create (10);
}
static void
clear_args (void)
{
argbuf.truncate (0);
}
static void
store_arg (const char *arg, int delete_always, int delete_failure)
{
argbuf.safe_push (arg);
if (delete_always || delete_failure)
{
const char *p;
if (arg[0] == '-'
&& (p = strrchr (arg, '=')))
arg = p + 1;
record_temp_file (arg, delete_always, delete_failure);
}
}

static char *
load_specs (const char *filename)
{
int desc;
int readlen;
struct stat statbuf;
char *buffer;
char *buffer_p;
char *specs;
char *specs_p;
if (verbose_flag)
fnotice (stderr, "Reading specs from %s\n", filename);
desc = open (filename, O_RDONLY, 0);
if (desc < 0)
pfatal_with_name (filename);
if (stat (filename, &statbuf) < 0)
pfatal_with_name (filename);
buffer = XNEWVEC (char, statbuf.st_size + 1);
readlen = read (desc, buffer, (unsigned) statbuf.st_size);
if (readlen < 0)
pfatal_with_name (filename);
buffer[readlen] = 0;
close (desc);
specs = XNEWVEC (char, readlen + 1);
specs_p = specs;
for (buffer_p = buffer; buffer_p && *buffer_p; buffer_p++)
{
int skip = 0;
char c = *buffer_p;
if (c == '\r')
{
if (buffer_p > buffer && *(buffer_p - 1) == '\n')	
skip = 1;
else if (*(buffer_p + 1) == '\n')			
skip = 1;
else							
c = '\n';
}
if (! skip)
*specs_p++ = c;
}
*specs_p = '\0';
free (buffer);
return (specs);
}
static void
read_specs (const char *filename, bool main_p, bool user_p)
{
char *buffer;
char *p;
buffer = load_specs (filename);
p = buffer;
while (1)
{
char *suffix;
char *spec;
char *in, *out, *p1, *p2, *p3;
p = skip_whitespace (p);
if (*p == 0)
break;
if (*p == '%' && !main_p)
{
p1 = p;
while (*p && *p != '\n')
p++;
p++;
if (!strncmp (p1, "%include", sizeof ("%include") - 1)
&& (p1[sizeof "%include" - 1] == ' '
|| p1[sizeof "%include" - 1] == '\t'))
{
char *new_filename;
p1 += sizeof ("%include");
while (*p1 == ' ' || *p1 == '\t')
p1++;
if (*p1++ != '<' || p[-2] != '>')
fatal_error (input_location,
"specs %%include syntax malformed after "
"%ld characters",
(long) (p1 - buffer + 1));
p[-2] = '\0';
new_filename = find_a_file (&startfile_prefixes, p1, R_OK, true);
read_specs (new_filename ? new_filename : p1, false, user_p);
continue;
}
else if (!strncmp (p1, "%include_noerr", sizeof "%include_noerr" - 1)
&& (p1[sizeof "%include_noerr" - 1] == ' '
|| p1[sizeof "%include_noerr" - 1] == '\t'))
{
char *new_filename;
p1 += sizeof "%include_noerr";
while (*p1 == ' ' || *p1 == '\t')
p1++;
if (*p1++ != '<' || p[-2] != '>')
fatal_error (input_location,
"specs %%include syntax malformed after "
"%ld characters",
(long) (p1 - buffer + 1));
p[-2] = '\0';
new_filename = find_a_file (&startfile_prefixes, p1, R_OK, true);
if (new_filename)
read_specs (new_filename, false, user_p);
else if (verbose_flag)
fnotice (stderr, "could not find specs file %s\n", p1);
continue;
}
else if (!strncmp (p1, "%rename", sizeof "%rename" - 1)
&& (p1[sizeof "%rename" - 1] == ' '
|| p1[sizeof "%rename" - 1] == '\t'))
{
int name_len;
struct spec_list *sl;
struct spec_list *newsl;
p1 += sizeof "%rename";
while (*p1 == ' ' || *p1 == '\t')
p1++;
if (! ISALPHA ((unsigned char) *p1))
fatal_error (input_location,
"specs %%rename syntax malformed after "
"%ld characters",
(long) (p1 - buffer));
p2 = p1;
while (*p2 && !ISSPACE ((unsigned char) *p2))
p2++;
if (*p2 != ' ' && *p2 != '\t')
fatal_error (input_location,
"specs %%rename syntax malformed after "
"%ld characters",
(long) (p2 - buffer));
name_len = p2 - p1;
*p2++ = '\0';
while (*p2 == ' ' || *p2 == '\t')
p2++;
if (! ISALPHA ((unsigned char) *p2))
fatal_error (input_location,
"specs %%rename syntax malformed after "
"%ld characters",
(long) (p2 - buffer));
p3 = p2;
while (*p3 && !ISSPACE ((unsigned char) *p3))
p3++;
if (p3 != p - 1)
fatal_error (input_location,
"specs %%rename syntax malformed after "
"%ld characters",
(long) (p3 - buffer));
*p3 = '\0';
for (sl = specs; sl; sl = sl->next)
if (name_len == sl->name_len && !strcmp (sl->name, p1))
break;
if (!sl)
fatal_error (input_location,
"specs %s spec was not found to be renamed", p1);
if (strcmp (p1, p2) == 0)
continue;
for (newsl = specs; newsl; newsl = newsl->next)
if (strcmp (newsl->name, p2) == 0)
fatal_error (input_location,
"%s: attempt to rename spec %qs to "
"already defined spec %qs",
filename, p1, p2);
if (verbose_flag)
{
fnotice (stderr, "rename spec %s to %s\n", p1, p2);
#ifdef DEBUG_SPECS
fnotice (stderr, "spec is '%s'\n\n", *(sl->ptr_spec));
#endif
}
set_spec (p2, *(sl->ptr_spec), user_p);
if (sl->alloc_p)
free (CONST_CAST (char *, *(sl->ptr_spec)));
*(sl->ptr_spec) = "";
sl->alloc_p = 0;
continue;
}
else
fatal_error (input_location,
"specs unknown %% command after %ld characters",
(long) (p1 - buffer));
}
p1 = p;
while (*p1 && *p1 != ':' && *p1 != '\n')
p1++;
if (*p1 != ':')
fatal_error (input_location,
"specs file malformed after %ld characters",
(long) (p1 - buffer));
p2 = p1;
while (p2 > buffer && (p2[-1] == ' ' || p2[-1] == '\t'))
p2--;
suffix = save_string (p, p2 - p);
p = skip_whitespace (p1 + 1);
if (p[1] == 0)
fatal_error (input_location,
"specs file malformed after %ld characters",
(long) (p - buffer));
p1 = p;
while (*p1 && !(*p1 == '\n' && (p1[1] == '\n' || p1[1] == '\0')))
p1++;
spec = save_string (p, p1 - p);
p = p1;
in = spec;
out = spec;
while (*in != 0)
{
if (in[0] == '\\' && in[1] == '\n')
in += 2;
else if (in[0] == '#')
while (*in && *in != '\n')
in++;
else
*out++ = *in++;
}
*out = 0;
if (suffix[0] == '*')
{
if (! strcmp (suffix, "*link_command"))
link_command_spec = spec;
else
{
set_spec (suffix + 1, spec, user_p);
free (spec);
}
}
else
{
compilers
= XRESIZEVEC (struct compiler, compilers, n_compilers + 2);
compilers[n_compilers].suffix = suffix;
compilers[n_compilers].spec = spec;
n_compilers++;
memset (&compilers[n_compilers], 0, sizeof compilers[n_compilers]);
}
if (*suffix == 0)
link_command_spec = spec;
}
if (link_command_spec == 0)
fatal_error (input_location, "spec file has no spec for linking");
XDELETEVEC (buffer);
}

static const char *temp_filename;
static int temp_filename_length;
struct temp_file
{
const char *name;
struct temp_file *next;
};
static struct temp_file *always_delete_queue;
static struct temp_file *failure_delete_queue;
void
record_temp_file (const char *filename, int always_delete, int fail_delete)
{
char *const name = xstrdup (filename);
if (always_delete)
{
struct temp_file *temp;
for (temp = always_delete_queue; temp; temp = temp->next)
if (! filename_cmp (name, temp->name))
{
free (name);
goto already1;
}
temp = XNEW (struct temp_file);
temp->next = always_delete_queue;
temp->name = name;
always_delete_queue = temp;
already1:;
}
if (fail_delete)
{
struct temp_file *temp;
for (temp = failure_delete_queue; temp; temp = temp->next)
if (! filename_cmp (name, temp->name))
{
free (name);
goto already2;
}
temp = XNEW (struct temp_file);
temp->next = failure_delete_queue;
temp->name = name;
failure_delete_queue = temp;
already2:;
}
}
#ifndef DELETE_IF_ORDINARY
#define DELETE_IF_ORDINARY(NAME,ST,VERBOSE_FLAG)        \
do                                                      \
{                                                     \
if (stat (NAME, &ST) >= 0 && S_ISREG (ST.st_mode))  \
if (unlink (NAME) < 0)                            \
if (VERBOSE_FLAG)                               \
perror_with_name (NAME);                      \
} while (0)
#endif
static void
delete_if_ordinary (const char *name)
{
struct stat st;
#ifdef DEBUG
int i, c;
printf ("Delete %s? (y or n) ", name);
fflush (stdout);
i = getchar ();
if (i != '\n')
while ((c = getchar ()) != '\n' && c != EOF)
;
if (i == 'y' || i == 'Y')
#endif 
DELETE_IF_ORDINARY (name, st, verbose_flag);
}
static void
delete_temp_files (void)
{
struct temp_file *temp;
for (temp = always_delete_queue; temp; temp = temp->next)
delete_if_ordinary (temp->name);
always_delete_queue = 0;
}
static void
delete_failure_queue (void)
{
struct temp_file *temp;
for (temp = failure_delete_queue; temp; temp = temp->next)
delete_if_ordinary (temp->name);
}
static void
clear_failure_queue (void)
{
failure_delete_queue = 0;
}

static void *
for_each_path (const struct path_prefix *paths,
bool do_multi,
size_t extra_space,
void *(*callback) (char *, void *),
void *callback_info)
{
struct prefix_list *pl;
const char *multi_dir = NULL;
const char *multi_os_dir = NULL;
const char *multiarch_suffix = NULL;
const char *multi_suffix;
const char *just_multi_suffix;
char *path = NULL;
void *ret = NULL;
bool skip_multi_dir = false;
bool skip_multi_os_dir = false;
multi_suffix = machine_suffix;
just_multi_suffix = just_machine_suffix;
if (do_multi && multilib_dir && strcmp (multilib_dir, ".") != 0)
{
multi_dir = concat (multilib_dir, dir_separator_str, NULL);
multi_suffix = concat (multi_suffix, multi_dir, NULL);
just_multi_suffix = concat (just_multi_suffix, multi_dir, NULL);
}
if (do_multi && multilib_os_dir && strcmp (multilib_os_dir, ".") != 0)
multi_os_dir = concat (multilib_os_dir, dir_separator_str, NULL);
if (multiarch_dir)
multiarch_suffix = concat (multiarch_dir, dir_separator_str, NULL);
while (1)
{
size_t multi_dir_len = 0;
size_t multi_os_dir_len = 0;
size_t multiarch_len = 0;
size_t suffix_len;
size_t just_suffix_len;
size_t len;
if (multi_dir)
multi_dir_len = strlen (multi_dir);
if (multi_os_dir)
multi_os_dir_len = strlen (multi_os_dir);
if (multiarch_suffix)
multiarch_len = strlen (multiarch_suffix);
suffix_len = strlen (multi_suffix);
just_suffix_len = strlen (just_multi_suffix);
if (path == NULL)
{
len = paths->max_len + extra_space + 1;
len += MAX (MAX (suffix_len, multi_os_dir_len), multiarch_len);
path = XNEWVEC (char, len);
}
for (pl = paths->plist; pl != 0; pl = pl->next)
{
len = strlen (pl->prefix);
memcpy (path, pl->prefix, len);
if (!skip_multi_dir)
{
memcpy (path + len, multi_suffix, suffix_len + 1);
ret = callback (path, callback_info);
if (ret)
break;
}
if (!skip_multi_dir
&& pl->require_machine_suffix == 2)
{
memcpy (path + len, just_multi_suffix, just_suffix_len + 1);
ret = callback (path, callback_info);
if (ret)
break;
}
if (!skip_multi_dir
&& !pl->require_machine_suffix && multiarch_dir)
{
memcpy (path + len, multiarch_suffix, multiarch_len + 1);
ret = callback (path, callback_info);
if (ret)
break;
}
if (!pl->require_machine_suffix
&& !(pl->os_multilib ? skip_multi_os_dir : skip_multi_dir))
{
const char *this_multi;
size_t this_multi_len;
if (pl->os_multilib)
{
this_multi = multi_os_dir;
this_multi_len = multi_os_dir_len;
}
else
{
this_multi = multi_dir;
this_multi_len = multi_dir_len;
}
if (this_multi_len)
memcpy (path + len, this_multi, this_multi_len + 1);
else
path[len] = '\0';
ret = callback (path, callback_info);
if (ret)
break;
}
}
if (pl)
break;
if (multi_dir == NULL && multi_os_dir == NULL)
break;
if (multi_dir)
{
free (CONST_CAST (char *, multi_dir));
multi_dir = NULL;
free (CONST_CAST (char *, multi_suffix));
multi_suffix = machine_suffix;
free (CONST_CAST (char *, just_multi_suffix));
just_multi_suffix = just_machine_suffix;
}
else
skip_multi_dir = true;
if (multi_os_dir)
{
free (CONST_CAST (char *, multi_os_dir));
multi_os_dir = NULL;
}
else
skip_multi_os_dir = true;
}
if (multi_dir)
{
free (CONST_CAST (char *, multi_dir));
free (CONST_CAST (char *, multi_suffix));
free (CONST_CAST (char *, just_multi_suffix));
}
if (multi_os_dir)
free (CONST_CAST (char *, multi_os_dir));
if (ret != path)
free (path);
return ret;
}
struct add_to_obstack_info {
struct obstack *ob;
bool check_dir;
bool first_time;
};
static void *
add_to_obstack (char *path, void *data)
{
struct add_to_obstack_info *info = (struct add_to_obstack_info *) data;
if (info->check_dir && !is_directory (path, false))
return NULL;
if (!info->first_time)
obstack_1grow (info->ob, PATH_SEPARATOR);
obstack_grow (info->ob, path, strlen (path));
info->first_time = false;
return NULL;
}
static void
xputenv (const char *string)
{
env.xput (string);
}
static char *
build_search_list (const struct path_prefix *paths, const char *prefix,
bool check_dir, bool do_multi)
{
struct add_to_obstack_info info;
info.ob = &collect_obstack;
info.check_dir = check_dir;
info.first_time = true;
obstack_grow (&collect_obstack, prefix, strlen (prefix));
obstack_1grow (&collect_obstack, '=');
for_each_path (paths, do_multi, 0, add_to_obstack, &info);
obstack_1grow (&collect_obstack, '\0');
return XOBFINISH (&collect_obstack, char *);
}
static void
putenv_from_prefixes (const struct path_prefix *paths, const char *env_var,
bool do_multi)
{
xputenv (build_search_list (paths, env_var, true, do_multi));
}

static int
access_check (const char *name, int mode)
{
if (mode == X_OK)
{
struct stat st;
if (stat (name, &st) < 0
|| S_ISDIR (st.st_mode))
return -1;
}
return access (name, mode);
}
struct file_at_path_info {
const char *name;
const char *suffix;
int name_len;
int suffix_len;
int mode;
};
static void *
file_at_path (char *path, void *data)
{
struct file_at_path_info *info = (struct file_at_path_info *) data;
size_t len = strlen (path);
memcpy (path + len, info->name, info->name_len);
len += info->name_len;
if (info->suffix_len)
{
memcpy (path + len, info->suffix, info->suffix_len + 1);
if (access_check (path, info->mode) == 0)
return path;
}
path[len] = '\0';
if (access_check (path, info->mode) == 0)
return path;
return NULL;
}
static char *
find_a_file (const struct path_prefix *pprefix, const char *name, int mode,
bool do_multi)
{
struct file_at_path_info info;
#ifdef DEFAULT_ASSEMBLER
if (! strcmp (name, "as") && access (DEFAULT_ASSEMBLER, mode) == 0)
return xstrdup (DEFAULT_ASSEMBLER);
#endif
#ifdef DEFAULT_LINKER
if (! strcmp (name, "ld") && access (DEFAULT_LINKER, mode) == 0)
return xstrdup (DEFAULT_LINKER);
#endif
if (IS_ABSOLUTE_PATH (name))
{
if (access (name, mode) == 0)
return xstrdup (name);
return NULL;
}
info.name = name;
info.suffix = (mode & X_OK) != 0 ? HOST_EXECUTABLE_SUFFIX : "";
info.name_len = strlen (info.name);
info.suffix_len = strlen (info.suffix);
info.mode = mode;
return (char*) for_each_path (pprefix, do_multi,
info.name_len + info.suffix_len,
file_at_path, &info);
}
enum path_prefix_priority
{
PREFIX_PRIORITY_B_OPT,
PREFIX_PRIORITY_LAST
};
static void
add_prefix (struct path_prefix *pprefix, const char *prefix,
const char *component,  int priority,
int require_machine_suffix, int os_multilib)
{
struct prefix_list *pl, **prev;
int len;
for (prev = &pprefix->plist;
(*prev) != NULL && (*prev)->priority <= priority;
prev = &(*prev)->next)
;
prefix = update_path (prefix, component);
len = strlen (prefix);
if (len > pprefix->max_len)
pprefix->max_len = len;
pl = XNEW (struct prefix_list);
pl->prefix = prefix;
pl->require_machine_suffix = require_machine_suffix;
pl->priority = priority;
pl->os_multilib = os_multilib;
pl->next = (*prev);
(*prev) = pl;
}
static void
add_sysrooted_prefix (struct path_prefix *pprefix, const char *prefix,
const char *component,
int priority,
int require_machine_suffix, int os_multilib)
{
if (!IS_ABSOLUTE_PATH (prefix))
fatal_error (input_location, "system path %qs is not absolute", prefix);
if (target_system_root)
{
char *sysroot_no_trailing_dir_separator = xstrdup (target_system_root);
size_t sysroot_len = strlen (target_system_root);
if (sysroot_len > 0
&& target_system_root[sysroot_len - 1] == DIR_SEPARATOR)
sysroot_no_trailing_dir_separator[sysroot_len - 1] = '\0';
if (target_sysroot_suffix)
prefix = concat (sysroot_no_trailing_dir_separator,
target_sysroot_suffix, prefix, NULL);
else
prefix = concat (sysroot_no_trailing_dir_separator, prefix, NULL);
free (sysroot_no_trailing_dir_separator);
component = "GCC";
}
add_prefix (pprefix, prefix, component, priority,
require_machine_suffix, os_multilib);
}

static int
execute (void)
{
int i;
int n_commands;		
char *string;
struct pex_obj *pex;
struct command
{
const char *prog;		
const char **argv;		
};
const char *arg;
struct command *commands;	
gcc_assert (!processing_spec_function);
if (wrapper_string)
{
string = find_a_file (&exec_prefixes,
argbuf[0], X_OK, false);
if (string)
argbuf[0] = string;
insert_wrapper (wrapper_string);
}
for (n_commands = 1, i = 0; argbuf.iterate (i, &arg); i++)
if (strcmp (arg, "|") == 0)
n_commands++;
commands = (struct command *) alloca (n_commands * sizeof (struct command));
argbuf.safe_push (0);
commands[0].prog = argbuf[0]; 
commands[0].argv = argbuf.address ();
if (!wrapper_string)
{
string = find_a_file (&exec_prefixes, commands[0].prog, X_OK, false);
commands[0].argv[0] = (string) ? string : commands[0].argv[0];
}
for (n_commands = 1, i = 0; argbuf.iterate (i, &arg); i++)
if (arg && strcmp (arg, "|") == 0)
{				
#if defined (__MSDOS__) || defined (OS2) || defined (VMS)
fatal_error (input_location, "-pipe not supported");
#endif
argbuf[i] = 0; 
commands[n_commands].prog = argbuf[i + 1];
commands[n_commands].argv
= &(argbuf.address ())[i + 1];
string = find_a_file (&exec_prefixes, commands[n_commands].prog,
X_OK, false);
if (string)
commands[n_commands].argv[0] = string;
n_commands++;
}
if (verbose_flag)
{
if (print_help_list)
fputc ('\n', stderr);
for (i = 0; i < n_commands; i++)
{
const char *const *j;
if (verbose_only_flag)
{
for (j = commands[i].argv; *j; j++)
{
const char *p;
for (p = *j; *p; ++p)
if (!ISALNUM ((unsigned char) *p)
&& *p != '_' && *p != '/' && *p != '-' && *p != '.')
break;
if (*p || !*j)
{
fprintf (stderr, " \"");
for (p = *j; *p; ++p)
{
if (*p == '"' || *p == '\\' || *p == '$')
fputc ('\\', stderr);
fputc (*p, stderr);
}
fputc ('"', stderr);
}
else if (!**j)
fprintf (stderr, " \"\"");
else
fprintf (stderr, " %s", *j);
}
}
else
for (j = commands[i].argv; *j; j++)
if (!**j)
fprintf (stderr, " \"\"");
else
fprintf (stderr, " %s", *j);
if (i + 1 != n_commands)
fprintf (stderr, " |");
fprintf (stderr, "\n");
}
fflush (stderr);
if (verbose_only_flag != 0)
{
execution_count++;
return 0;
}
#ifdef DEBUG
fnotice (stderr, "\nGo ahead? (y or n) ");
fflush (stderr);
i = getchar ();
if (i != '\n')
while (getchar () != '\n')
;
if (i != 'y' && i != 'Y')
return 0;
#endif 
}
#ifdef ENABLE_VALGRIND_CHECKING
for (i = 0; i < n_commands; i++)
{
const char **argv;
int argc;
int j;
for (argc = 0; commands[i].argv[argc] != NULL; argc++)
;
argv = XALLOCAVEC (const char *, argc + 3);
argv[0] = VALGRIND_PATH;
argv[1] = "-q";
for (j = 2; j < argc + 2; j++)
argv[j] = commands[i].argv[j - 2];
argv[j] = NULL;
commands[i].argv = argv;
commands[i].prog = argv[0];
}
#endif
pex = pex_init (PEX_USE_PIPES | ((report_times || report_times_to_file)
? PEX_RECORD_TIMES : 0),
progname, temp_filename);
if (pex == NULL)
fatal_error (input_location, "pex_init failed: %m");
for (i = 0; i < n_commands; i++)
{
const char *errmsg;
int err;
const char *string = commands[i].argv[0];
errmsg = pex_run (pex,
((i + 1 == n_commands ? PEX_LAST : 0)
| (string == commands[i].prog ? PEX_SEARCH : 0)),
string, CONST_CAST (char **, commands[i].argv),
NULL, NULL, &err);
if (errmsg != NULL)
{
if (err == 0)
fatal_error (input_location, errmsg);
else
{
errno = err;
pfatal_with_name (errmsg);
}
}
if (i && string != commands[i].prog)
free (CONST_CAST (char *, string));
}
execution_count++;
{
int *statuses;
struct pex_time *times = NULL;
int ret_code = 0;
statuses = (int *) alloca (n_commands * sizeof (int));
if (!pex_get_status (pex, n_commands, statuses))
fatal_error (input_location, "failed to get exit status: %m");
if (report_times || report_times_to_file)
{
times = (struct pex_time *) alloca (n_commands * sizeof (struct pex_time));
if (!pex_get_times (pex, n_commands, times))
fatal_error (input_location, "failed to get process times: %m");
}
pex_free (pex);
for (i = 0; i < n_commands; ++i)
{
int status = statuses[i];
if (WIFSIGNALED (status))
switch (WTERMSIG (status))
{
case SIGINT:
case SIGTERM:
#ifdef SIGQUIT
case SIGQUIT:
#endif
#ifdef SIGKILL
case SIGKILL:
#endif
fatal_error (input_location,
"%s signal terminated program %s",
strsignal (WTERMSIG (status)),
commands[i].prog);
break;
#ifdef SIGPIPE
case SIGPIPE:
if (signal_count || greatest_status >= MIN_FATAL_STATUS)
{
signal_count++;
ret_code = -1;
break;
}
#endif
default:
internal_error_no_backtrace ("%s signal terminated program %s",
strsignal (WTERMSIG (status)),
commands[i].prog);
}
else if (WIFEXITED (status)
&& WEXITSTATUS (status) >= MIN_FATAL_STATUS)
{
const char *p;
if (flag_report_bug
&& WEXITSTATUS (status) == ICE_EXIT_CODE
&& i == 0
&& (p = strrchr (commands[0].argv[0], DIR_SEPARATOR))
&& ! strncmp (p + 1, "cc1", 3))
try_generate_repro (commands[0].argv);
if (WEXITSTATUS (status) > greatest_status)
greatest_status = WEXITSTATUS (status);
ret_code = -1;
}
if (report_times || report_times_to_file)
{
struct pex_time *pt = &times[i];
double ut, st;
ut = ((double) pt->user_seconds
+ (double) pt->user_microseconds / 1.0e6);
st = ((double) pt->system_seconds
+ (double) pt->system_microseconds / 1.0e6);
if (ut + st != 0)
{
if (report_times)
fnotice (stderr, "# %s %.2f %.2f\n",
commands[i].prog, ut, st);
if (report_times_to_file)
{
int c = 0;
const char *const *j;
fprintf (report_times_to_file, "%g %g", ut, st);
for (j = &commands[i].prog; *j; j = &commands[i].argv[++c])
{
const char *p;
for (p = *j; *p; ++p)
if (*p == '"' || *p == '\\' || *p == '$'
|| ISSPACE (*p))
break;
if (*p)
{
fprintf (report_times_to_file, " \"");
for (p = *j; *p; ++p)
{
if (*p == '"' || *p == '\\' || *p == '$')
fputc ('\\', report_times_to_file);
fputc (*p, report_times_to_file);
}
fputc ('"', report_times_to_file);
}
else
fprintf (report_times_to_file, " %s", *j);
}
fputc ('\n', report_times_to_file);
}
}
}
}
if (commands[0].argv[0] != commands[0].prog)
free (CONST_CAST (char *, commands[0].argv[0]));
return ret_code;
}
}

#define SWITCH_LIVE    			(1 << 0)
#define SWITCH_FALSE   			(1 << 1)
#define SWITCH_IGNORE			(1 << 2)
#define SWITCH_IGNORE_PERMANENTLY	(1 << 3)
#define SWITCH_KEEP_FOR_GCC		(1 << 4)
struct switchstr
{
const char *part1;
const char **args;
unsigned int live_cond;
bool known;
bool validated;
bool ordering;
};
static struct switchstr *switches;
static int n_switches;
static int n_switches_alloc;
int compare_debug;
int compare_debug_second;
const char *compare_debug_opt;
static struct switchstr *switches_debug_check[2];
static int n_switches_debug_check[2];
static int n_switches_alloc_debug_check[2];
static char *debug_check_temp_file[2];
struct infile
{
const char *name;
const char *language;
struct compiler *incompiler;
bool compiled;
bool preprocessed;
};
static struct infile *infiles;
int n_infiles;
static int n_infiles_alloc;
static bool spec_undefvar_allowed;
static bool combine_inputs;
static int added_libraries;
const char **outfiles;

#if defined(HAVE_TARGET_OBJECT_SUFFIX) || defined(HAVE_TARGET_EXECUTABLE_SUFFIX)
static const char *
convert_filename (const char *name, int do_exe ATTRIBUTE_UNUSED,
int do_obj ATTRIBUTE_UNUSED)
{
#if defined(HAVE_TARGET_EXECUTABLE_SUFFIX)
int i;
#endif
int len;
if (name == NULL)
return NULL;
len = strlen (name);
#ifdef HAVE_TARGET_OBJECT_SUFFIX
if (do_obj && len > 2
&& name[len - 2] == '.'
&& name[len - 1] == 'o')
{
obstack_grow (&obstack, name, len - 2);
obstack_grow0 (&obstack, TARGET_OBJECT_SUFFIX, strlen (TARGET_OBJECT_SUFFIX));
name = XOBFINISH (&obstack, const char *);
}
#endif
#if defined(HAVE_TARGET_EXECUTABLE_SUFFIX)
if (! do_exe || TARGET_EXECUTABLE_SUFFIX[0] == 0 || (len == 2 && name[0] == '-'))
return name;
for (i = len - 1; i >= 0; i--)
if (IS_DIR_SEPARATOR (name[i]))
break;
for (i++; i < len; i++)
if (name[i] == '.')
return name;
obstack_grow (&obstack, name, len);
obstack_grow0 (&obstack, TARGET_EXECUTABLE_SUFFIX,
strlen (TARGET_EXECUTABLE_SUFFIX));
name = XOBFINISH (&obstack, const char *);
#endif
return name;
}
#endif

static void
display_help (void)
{
printf (_("Usage: %s [options] file...\n"), progname);
fputs (_("Options:\n"), stdout);
fputs (_("  -pass-exit-codes         Exit with highest error code from a phase.\n"), stdout);
fputs (_("  --help                   Display this information.\n"), stdout);
fputs (_("  --target-help            Display target specific command line options.\n"), stdout);
fputs (_("  --help={common|optimizers|params|target|warnings|[^]{joined|separate|undocumented}}[,...].\n"), stdout);
fputs (_("                           Display specific types of command line options.\n"), stdout);
if (! verbose_flag)
fputs (_("  (Use '-v --help' to display command line options of sub-processes).\n"), stdout);
fputs (_("  --version                Display compiler version information.\n"), stdout);
fputs (_("  -dumpspecs               Display all of the built in spec strings.\n"), stdout);
fputs (_("  -dumpversion             Display the version of the compiler.\n"), stdout);
fputs (_("  -dumpmachine             Display the compiler's target processor.\n"), stdout);
fputs (_("  -print-search-dirs       Display the directories in the compiler's search path.\n"), stdout);
fputs (_("  -print-libgcc-file-name  Display the name of the compiler's companion library.\n"), stdout);
fputs (_("  -print-file-name=<lib>   Display the full path to library <lib>.\n"), stdout);
fputs (_("  -print-prog-name=<prog>  Display the full path to compiler component <prog>.\n"), stdout);
fputs (_("\
-print-multiarch         Display the target's normalized GNU triplet, used as\n\
a component in the library path.\n"), stdout);
fputs (_("  -print-multi-directory   Display the root directory for versions of libgcc.\n"), stdout);
fputs (_("\
-print-multi-lib         Display the mapping between command line options and\n\
multiple library search directories.\n"), stdout);
fputs (_("  -print-multi-os-directory Display the relative path to OS libraries.\n"), stdout);
fputs (_("  -print-sysroot           Display the target libraries directory.\n"), stdout);
fputs (_("  -print-sysroot-headers-suffix Display the sysroot suffix used to find headers.\n"), stdout);
fputs (_("  -Wa,<options>            Pass comma-separated <options> on to the assembler.\n"), stdout);
fputs (_("  -Wp,<options>            Pass comma-separated <options> on to the preprocessor.\n"), stdout);
fputs (_("  -Wl,<options>            Pass comma-separated <options> on to the linker.\n"), stdout);
fputs (_("  -Xassembler <arg>        Pass <arg> on to the assembler.\n"), stdout);
fputs (_("  -Xpreprocessor <arg>     Pass <arg> on to the preprocessor.\n"), stdout);
fputs (_("  -Xlinker <arg>           Pass <arg> on to the linker.\n"), stdout);
fputs (_("  -save-temps              Do not delete intermediate files.\n"), stdout);
fputs (_("  -save-temps=<arg>        Do not delete intermediate files.\n"), stdout);
fputs (_("\
-no-canonical-prefixes   Do not canonicalize paths when building relative\n\
prefixes to other gcc components.\n"), stdout);
fputs (_("  -pipe                    Use pipes rather than intermediate files.\n"), stdout);
fputs (_("  -time                    Time the execution of each subprocess.\n"), stdout);
fputs (_("  -specs=<file>            Override built-in specs with the contents of <file>.\n"), stdout);
fputs (_("  -std=<standard>          Assume that the input sources are for <standard>.\n"), stdout);
fputs (_("\
--sysroot=<directory>    Use <directory> as the root directory for headers\n\
and libraries.\n"), stdout);
fputs (_("  -B <directory>           Add <directory> to the compiler's search paths.\n"), stdout);
fputs (_("  -v                       Display the programs invoked by the compiler.\n"), stdout);
fputs (_("  -###                     Like -v but options quoted and commands not executed.\n"), stdout);
fputs (_("  -E                       Preprocess only; do not compile, assemble or link.\n"), stdout);
fputs (_("  -S                       Compile only; do not assemble or link.\n"), stdout);
fputs (_("  -c                       Compile and assemble, but do not link.\n"), stdout);
fputs (_("  -o <file>                Place the output into <file>.\n"), stdout);
fputs (_("  -pie                     Create a dynamically linked position independent\n\
executable.\n"), stdout);
fputs (_("  -shared                  Create a shared library.\n"), stdout);
fputs (_("\
-x <language>            Specify the language of the following input files.\n\
Permissible languages include: c c++ assembler none\n\
'none' means revert to the default behavior of\n\
guessing the language based on the file's extension.\n\
"), stdout);
printf (_("\
\nOptions starting with -g, -f, -m, -O, -W, or --param are automatically\n\
passed on to the various sub-processes invoked by %s.  In order to pass\n\
other options on to these processes the -W<letter> options must be used.\n\
"), progname);
}
static void
add_preprocessor_option (const char *option, int len)
{
preprocessor_options.safe_push (save_string (option, len));
}
static void
add_assembler_option (const char *option, int len)
{
assembler_options.safe_push (save_string (option, len));
}
static void
add_linker_option (const char *option, int len)
{
linker_options.safe_push (save_string (option, len));
}

static void
alloc_infile (void)
{
if (n_infiles_alloc == 0)
{
n_infiles_alloc = 16;
infiles = XNEWVEC (struct infile, n_infiles_alloc);
}
else if (n_infiles_alloc == n_infiles)
{
n_infiles_alloc *= 2;
infiles = XRESIZEVEC (struct infile, infiles, n_infiles_alloc);
}
}
static void
add_infile (const char *name, const char *language)
{
alloc_infile ();
infiles[n_infiles].name = name;
infiles[n_infiles++].language = language;
}
static void
alloc_switch (void)
{
if (n_switches_alloc == 0)
{
n_switches_alloc = 16;
switches = XNEWVEC (struct switchstr, n_switches_alloc);
}
else if (n_switches_alloc == n_switches)
{
n_switches_alloc *= 2;
switches = XRESIZEVEC (struct switchstr, switches, n_switches_alloc);
}
}
static void
save_switch (const char *opt, size_t n_args, const char *const *args,
bool validated, bool known)
{
alloc_switch ();
switches[n_switches].part1 = opt + 1;
if (n_args == 0)
switches[n_switches].args = 0;
else
{
switches[n_switches].args = XNEWVEC (const char *, n_args + 1);
memcpy (switches[n_switches].args, args, n_args * sizeof (const char *));
switches[n_switches].args[n_args] = NULL;
}
switches[n_switches].live_cond = 0;
switches[n_switches].validated = validated;
switches[n_switches].known = known;
switches[n_switches].ordering = 0;
n_switches++;
}
static void
set_source_date_epoch_envvar ()
{
char source_date_epoch[21];
time_t tt;
errno = 0;
tt = time (NULL);
if (tt < (time_t) 0 || errno != 0)
tt = (time_t) 0;
snprintf (source_date_epoch, 21, "%llu", (unsigned long long) tt);
setenv ("SOURCE_DATE_EPOCH", source_date_epoch, 0);
}
static bool
driver_unknown_option_callback (const struct cl_decoded_option *decoded)
{
const char *opt = decoded->arg;
if (opt[1] == 'W' && opt[2] == 'n' && opt[3] == 'o' && opt[4] == '-'
&& !(decoded->errors & CL_ERR_NEGATIVE))
{
save_switch (decoded->canonical_option[0],
decoded->canonical_option_num_elements - 1,
&decoded->canonical_option[1], false, true);
return false;
}
if (decoded->opt_index == OPT_SPECIAL_unknown)
{
save_switch (decoded->canonical_option[0],
decoded->canonical_option_num_elements - 1,
&decoded->canonical_option[1], false, false);
return false;
}
else
return true;
}
static void
driver_wrong_lang_callback (const struct cl_decoded_option *decoded,
unsigned int lang_mask ATTRIBUTE_UNUSED)
{
const struct cl_option *option = &cl_options[decoded->opt_index];
if (option->cl_reject_driver)
error ("unrecognized command line option %qs",
decoded->orig_option_with_args_text);
else
save_switch (decoded->canonical_option[0],
decoded->canonical_option_num_elements - 1,
&decoded->canonical_option[1], false, true);
}
static const char *spec_lang = 0;
static int last_language_n_infiles;
static void
handle_foffload_option (const char *arg)
{
const char *c, *cur, *n, *next, *end;
char *target;
if (arg[0] == '-')
return;
end = strchr (arg, '=');
if (end == NULL)
end = strchr (arg, '\0');
cur = arg;
while (cur < end)
{
next = strchr (cur, ',');
if (next == NULL)
next = end;
next = (next > end) ? end : next;
target = XNEWVEC (char, next - cur + 1);
memcpy (target, cur, next - cur);
target[next - cur] = '\0';
if (strcmp (target, "disable") == 0)
{
free (offload_targets);
offload_targets = xstrdup ("");
break;
}
c = OFFLOAD_TARGETS;
while (c)
{
n = strchr (c, ',');
if (n == NULL)
n = strchr (c, '\0');
if (next - cur == n - c && strncmp (target, c, n - c) == 0)
break;
c = *n ? n + 1 : NULL;
}
if (!c)
fatal_error (input_location,
"GCC is not configured to support %s as offload target",
target);
if (!offload_targets)
{
offload_targets = target;
target = NULL;
}
else
{
c = offload_targets;
do
{
n = strchr (c, ':');
if (n == NULL)
n = strchr (c, '\0');
if (next - cur == n - c && strncmp (c, target, n - c) == 0)
break;
c = n + 1;
}
while (*n);
if (c > n)
{
size_t offload_targets_len = strlen (offload_targets);
offload_targets
= XRESIZEVEC (char, offload_targets,
offload_targets_len + 1 + next - cur + 1);
offload_targets[offload_targets_len++] = ':';
memcpy (offload_targets + offload_targets_len, target, next - cur + 1);
}
}
cur = next + 1;
XDELETEVEC (target);
}
}
static bool
driver_handle_option (struct gcc_options *opts,
struct gcc_options *opts_set,
const struct cl_decoded_option *decoded,
unsigned int lang_mask ATTRIBUTE_UNUSED, int kind,
location_t loc,
const struct cl_option_handlers *handlers ATTRIBUTE_UNUSED,
diagnostic_context *dc,
void (*) (void))
{
size_t opt_index = decoded->opt_index;
const char *arg = decoded->arg;
const char *compare_debug_replacement_opt;
int value = decoded->value;
bool validated = false;
bool do_save = true;
gcc_assert (opts == &global_options);
gcc_assert (opts_set == &global_options_set);
gcc_assert (kind == DK_UNSPECIFIED);
gcc_assert (loc == UNKNOWN_LOCATION);
gcc_assert (dc == global_dc);
switch (opt_index)
{
case OPT_dumpspecs:
{
struct spec_list *sl;
init_spec ();
for (sl = specs; sl; sl = sl->next)
printf ("*%s:\n%s\n\n", sl->name, *(sl->ptr_spec));
if (link_command_spec)
printf ("*link_command:\n%s\n\n", link_command_spec);
exit (0);
}
case OPT_dumpversion:
printf ("%s\n", spec_version);
exit (0);
case OPT_dumpmachine:
printf ("%s\n", spec_machine);
exit (0);
case OPT_dumpfullversion:
printf ("%s\n", BASEVER);
exit (0);
case OPT__version:
print_version = 1;
if (is_cpp_driver)
add_preprocessor_option ("--version", strlen ("--version"));
add_assembler_option ("--version", strlen ("--version"));
add_linker_option ("--version", strlen ("--version"));
break;
case OPT__help:
print_help_list = 1;
if (is_cpp_driver)
add_preprocessor_option ("--help", 6);
add_assembler_option ("--help", 6);
add_linker_option ("--help", 6);
break;
case OPT__help_:
print_subprocess_help = 2;
break;
case OPT__target_help:
print_subprocess_help = 1;
if (is_cpp_driver)
add_preprocessor_option ("--target-help", 13);
add_assembler_option ("--target-help", 13);
add_linker_option ("--target-help", 13);
break;
case OPT__no_sysroot_suffix:
case OPT_pass_exit_codes:
case OPT_print_search_dirs:
case OPT_print_file_name_:
case OPT_print_prog_name_:
case OPT_print_multi_lib:
case OPT_print_multi_directory:
case OPT_print_sysroot:
case OPT_print_multi_os_directory:
case OPT_print_multiarch:
case OPT_print_sysroot_headers_suffix:
case OPT_time:
case OPT_wrapper:
do_save = false;
break;
case OPT_print_libgcc_file_name:
print_file_name = "libgcc.a";
do_save = false;
break;
case OPT_fuse_ld_bfd:
use_ld = ".bfd";
break;
case OPT_fuse_ld_gold:
use_ld = ".gold";
break;
case OPT_fcompare_debug_second:
compare_debug_second = 1;
break;
case OPT_fcompare_debug:
switch (value)
{
case 0:
compare_debug_replacement_opt = "-fcompare-debug=";
arg = "";
goto compare_debug_with_arg;
case 1:
compare_debug_replacement_opt = "-fcompare-debug=-gtoggle";
arg = "-gtoggle";
goto compare_debug_with_arg;
default:
gcc_unreachable ();
}
break;
case OPT_fcompare_debug_:
compare_debug_replacement_opt = decoded->canonical_option[0];
compare_debug_with_arg:
gcc_assert (decoded->canonical_option_num_elements == 1);
gcc_assert (arg != NULL);
if (*arg)
compare_debug = 1;
else
compare_debug = -1;
if (compare_debug < 0)
compare_debug_opt = NULL;
else
compare_debug_opt = arg;
save_switch (compare_debug_replacement_opt, 0, NULL, validated, true);
set_source_date_epoch_envvar ();
return true;
case OPT_fdiagnostics_color_:
diagnostic_color_init (dc, value);
break;
case OPT_Wa_:
{
int prev, j;
prev = 0;
for (j = 0; arg[j]; j++)
if (arg[j] == ',')
{
add_assembler_option (arg + prev, j - prev);
prev = j + 1;
}
add_assembler_option (arg + prev, j - prev);
}
do_save = false;
break;
case OPT_Wp_:
{
int prev, j;
prev = 0;
for (j = 0; arg[j]; j++)
if (arg[j] == ',')
{
add_preprocessor_option (arg + prev, j - prev);
prev = j + 1;
}
add_preprocessor_option (arg + prev, j - prev);
}
do_save = false;
break;
case OPT_Wl_:
{
int prev, j;
prev = 0;
for (j = 0; arg[j]; j++)
if (arg[j] == ',')
{
add_infile (save_string (arg + prev, j - prev), "*");
prev = j + 1;
}
add_infile (arg + prev, "*");
}
do_save = false;
break;
case OPT_Xlinker:
add_infile (arg, "*");
do_save = false;
break;
case OPT_Xpreprocessor:
add_preprocessor_option (arg, strlen (arg));
do_save = false;
break;
case OPT_Xassembler:
add_assembler_option (arg, strlen (arg));
do_save = false;
break;
case OPT_l:
add_infile (concat ("-l", arg, NULL), "*");
do_save = false;
break;
case OPT_L:
save_switch (concat ("-L", arg, NULL), 0, NULL, validated, true);
return true;
case OPT_F:
save_switch (concat ("-F", arg, NULL), 0, NULL, validated, true);
return true;
case OPT_save_temps:
save_temps_flag = SAVE_TEMPS_CWD;
validated = true;
break;
case OPT_save_temps_:
if (strcmp (arg, "cwd") == 0)
save_temps_flag = SAVE_TEMPS_CWD;
else if (strcmp (arg, "obj") == 0
|| strcmp (arg, "object") == 0)
save_temps_flag = SAVE_TEMPS_OBJ;
else
fatal_error (input_location, "%qs is an unknown -save-temps option",
decoded->orig_option_with_args_text);
break;
case OPT_no_canonical_prefixes:
do_save = false;
break;
case OPT_pipe:
validated = true;
break;
case OPT_specs_:
{
struct user_specs *user = XNEW (struct user_specs);
user->next = (struct user_specs *) 0;
user->filename = arg;
if (user_specs_tail)
user_specs_tail->next = user;
else
user_specs_head = user;
user_specs_tail = user;
}
validated = true;
break;
case OPT__sysroot_:
target_system_root = arg;
target_system_root_changed = 1;
do_save = false;
break;
case OPT_time_:
if (report_times_to_file)
fclose (report_times_to_file);
report_times_to_file = fopen (arg, "a");
do_save = false;
break;
case OPT____:
verbose_only_flag++;
verbose_flag = 1;
do_save = false;
break;
case OPT_B:
{
size_t len = strlen (arg);
if (!IS_DIR_SEPARATOR (arg[len - 1])
&& is_directory (arg, false))
{
char *tmp = XNEWVEC (char, len + 2);
strcpy (tmp, arg);
tmp[len] = DIR_SEPARATOR;
tmp[++len] = 0;
arg = tmp;
}
add_prefix (&exec_prefixes, arg, NULL,
PREFIX_PRIORITY_B_OPT, 0, 0);
add_prefix (&startfile_prefixes, arg, NULL,
PREFIX_PRIORITY_B_OPT, 0, 0);
add_prefix (&include_prefixes, arg, NULL,
PREFIX_PRIORITY_B_OPT, 0, 0);
}
validated = true;
break;
case OPT_E:
have_E = true;
break;
case OPT_x:
spec_lang = arg;
if (!strcmp (spec_lang, "none"))
spec_lang = 0;
else
last_language_n_infiles = n_infiles;
do_save = false;
break;
case OPT_o:
have_o = 1;
#if defined(HAVE_TARGET_EXECUTABLE_SUFFIX) || defined(HAVE_TARGET_OBJECT_SUFFIX)
arg = convert_filename (arg, ! have_c, 0);
#endif
output_file = arg;
save_temps_prefix = xstrdup (arg);
save_switch ("-o", 1, &arg, validated, true);
return true;
#ifdef ENABLE_DEFAULT_PIE
case OPT_pie:
#endif
case OPT_static_libgcc:
case OPT_shared_libgcc:
case OPT_static_libgfortran:
case OPT_static_libstdc__:
validated = true;
break;
case OPT_fwpa:
flag_wpa = "";
break;
case OPT_foffload_:
handle_foffload_option (arg);
break;
default:
break;
}
if (do_save)
save_switch (decoded->canonical_option[0],
decoded->canonical_option_num_elements - 1,
&decoded->canonical_option[1], validated, true);
return true;
}
static void
set_option_handlers (struct cl_option_handlers *handlers)
{
handlers->unknown_option_callback = driver_unknown_option_callback;
handlers->wrong_lang_callback = driver_wrong_lang_callback;
handlers->num_handlers = 3;
handlers->handlers[0].handler = driver_handle_option;
handlers->handlers[0].mask = CL_DRIVER;
handlers->handlers[1].handler = common_handle_option;
handlers->handlers[1].mask = CL_COMMON;
handlers->handlers[2].handler = target_handle_option;
handlers->handlers[2].mask = CL_TARGET;
}
static void
process_command (unsigned int decoded_options_count,
struct cl_decoded_option *decoded_options)
{
const char *temp;
char *temp1;
char *tooldir_prefix, *tooldir_prefix2;
char *(*get_relative_prefix) (const char *, const char *,
const char *) = NULL;
struct cl_option_handlers handlers;
unsigned int j;
gcc_exec_prefix = env.get ("GCC_EXEC_PREFIX");
n_switches = 0;
n_infiles = 0;
added_libraries = 0;
compiler_version = temp1 = xstrdup (version_string);
for (; *temp1; ++temp1)
{
if (*temp1 == ' ')
{
*temp1 = '\0';
break;
}
}
for (j = 1; j < decoded_options_count; j++)
{
if (decoded_options[j].opt_index == OPT_no_canonical_prefixes)
{
get_relative_prefix = make_relative_prefix_ignore_links;
break;
}
}
if (! get_relative_prefix)
get_relative_prefix = make_relative_prefix;
gcc_libexec_prefix = standard_libexec_prefix;
#ifndef VMS
if (!gcc_exec_prefix)
{
gcc_exec_prefix = get_relative_prefix (decoded_options[0].arg,
standard_bindir_prefix,
standard_exec_prefix);
gcc_libexec_prefix = get_relative_prefix (decoded_options[0].arg,
standard_bindir_prefix,
standard_libexec_prefix);
if (gcc_exec_prefix)
xputenv (concat ("GCC_EXEC_PREFIX=", gcc_exec_prefix, NULL));
}
else
{
char *tmp_prefix = concat (gcc_exec_prefix, "gcc", NULL);
gcc_libexec_prefix = get_relative_prefix (tmp_prefix,
standard_exec_prefix,
standard_libexec_prefix);
if (!gcc_libexec_prefix)
gcc_libexec_prefix = standard_libexec_prefix;
free (tmp_prefix);
}
#else
#endif
lang_specific_driver (&decoded_options, &decoded_options_count,
&added_libraries);
if (gcc_exec_prefix)
{
int len = strlen (gcc_exec_prefix);
if (len > (int) sizeof ("/lib/gcc/") - 1
&& (IS_DIR_SEPARATOR (gcc_exec_prefix[len-1])))
{
temp = gcc_exec_prefix + len - sizeof ("/lib/gcc/") + 1;
if (IS_DIR_SEPARATOR (*temp)
&& filename_ncmp (temp + 1, "lib", 3) == 0
&& IS_DIR_SEPARATOR (temp[4])
&& filename_ncmp (temp + 5, "gcc", 3) == 0)
len -= sizeof ("/lib/gcc/") - 1;
}
set_std_prefix (gcc_exec_prefix, len);
add_prefix (&exec_prefixes, gcc_libexec_prefix, "GCC",
PREFIX_PRIORITY_LAST, 0, 0);
add_prefix (&startfile_prefixes, gcc_exec_prefix, "GCC",
PREFIX_PRIORITY_LAST, 0, 0);
}
temp = env.get ("COMPILER_PATH");
if (temp)
{
const char *startp, *endp;
char *nstore = (char *) alloca (strlen (temp) + 3);
startp = endp = temp;
while (1)
{
if (*endp == PATH_SEPARATOR || *endp == 0)
{
strncpy (nstore, startp, endp - startp);
if (endp == startp)
strcpy (nstore, concat (".", dir_separator_str, NULL));
else if (!IS_DIR_SEPARATOR (endp[-1]))
{
nstore[endp - startp] = DIR_SEPARATOR;
nstore[endp - startp + 1] = 0;
}
else
nstore[endp - startp] = 0;
add_prefix (&exec_prefixes, nstore, 0,
PREFIX_PRIORITY_LAST, 0, 0);
add_prefix (&include_prefixes, nstore, 0,
PREFIX_PRIORITY_LAST, 0, 0);
if (*endp == 0)
break;
endp = startp = endp + 1;
}
else
endp++;
}
}
temp = env.get (LIBRARY_PATH_ENV);
if (temp && *cross_compile == '0')
{
const char *startp, *endp;
char *nstore = (char *) alloca (strlen (temp) + 3);
startp = endp = temp;
while (1)
{
if (*endp == PATH_SEPARATOR || *endp == 0)
{
strncpy (nstore, startp, endp - startp);
if (endp == startp)
strcpy (nstore, concat (".", dir_separator_str, NULL));
else if (!IS_DIR_SEPARATOR (endp[-1]))
{
nstore[endp - startp] = DIR_SEPARATOR;
nstore[endp - startp + 1] = 0;
}
else
nstore[endp - startp] = 0;
add_prefix (&startfile_prefixes, nstore, NULL,
PREFIX_PRIORITY_LAST, 0, 1);
if (*endp == 0)
break;
endp = startp = endp + 1;
}
else
endp++;
}
}
temp = env.get ("LPATH");
if (temp && *cross_compile == '0')
{
const char *startp, *endp;
char *nstore = (char *) alloca (strlen (temp) + 3);
startp = endp = temp;
while (1)
{
if (*endp == PATH_SEPARATOR || *endp == 0)
{
strncpy (nstore, startp, endp - startp);
if (endp == startp)
strcpy (nstore, concat (".", dir_separator_str, NULL));
else if (!IS_DIR_SEPARATOR (endp[-1]))
{
nstore[endp - startp] = DIR_SEPARATOR;
nstore[endp - startp + 1] = 0;
}
else
nstore[endp - startp] = 0;
add_prefix (&startfile_prefixes, nstore, NULL,
PREFIX_PRIORITY_LAST, 0, 1);
if (*endp == 0)
break;
endp = startp = endp + 1;
}
else
endp++;
}
}
last_language_n_infiles = -1;
set_option_handlers (&handlers);
for (j = 1; j < decoded_options_count; j++)
{
switch (decoded_options[j].opt_index)
{
case OPT_S:
case OPT_c:
case OPT_E:
have_c = 1;
break;
}
if (have_c)
break;
}
for (j = 1; j < decoded_options_count; j++)
{
if (decoded_options[j].opt_index == OPT_SPECIAL_input_file)
{
const char *arg = decoded_options[j].arg;
const char *p = strrchr (arg, '@');
char *fname;
long offset;
int consumed;
#ifdef HAVE_TARGET_OBJECT_SUFFIX
arg = convert_filename (arg, 0, access (arg, F_OK));
#endif
if (p
&& p != arg
&& sscanf (p, "@%li%n", &offset, &consumed) >= 1
&& strlen (p) == (unsigned int)consumed)
{
fname = (char *)xmalloc (p - arg + 1);
memcpy (fname, arg, p - arg);
fname[p - arg] = '\0';
if (strcmp (fname, "-") == 0 || access (fname, F_OK) < 0)
{
free (fname);
fname = xstrdup (arg);
}
}
else
fname = xstrdup (arg);
if (strcmp (fname, "-") != 0 && access (fname, F_OK) < 0)
{
if (fname[0] == '@' && access (fname + 1, F_OK) < 0)
perror_with_name (fname + 1);
else
perror_with_name (fname);
}
else
add_infile (arg, spec_lang);
free (fname);
continue;
}
read_cmdline_option (&global_options, &global_options_set,
decoded_options + j, UNKNOWN_LOCATION,
CL_DRIVER, &handlers, global_dc);
}
if (ENABLE_OFFLOADING && offload_targets == NULL)
handle_foffload_option (OFFLOAD_TARGETS);
if (output_file
&& strcmp (output_file, "-") != 0
&& strcmp (output_file, HOST_BIT_BUCKET) != 0)
{
int i;
for (i = 0; i < n_infiles; i++)
if ((!infiles[i].language || infiles[i].language[0] != '*')
&& canonical_filename_eq (infiles[i].name, output_file))
fatal_error (input_location,
"input file %qs is the same as output file",
output_file);
}
if (output_file != NULL && output_file[0] == '\0')
fatal_error (input_location, "output filename may not be empty");
if (save_temps_flag == SAVE_TEMPS_OBJ && save_temps_prefix != NULL)
{
save_temps_length = strlen (save_temps_prefix);
temp = strrchr (lbasename (save_temps_prefix), '.');
if (temp)
{
save_temps_length -= strlen (temp);
save_temps_prefix[save_temps_length] = '\0';
}
}
else if (save_temps_prefix != NULL)
{
free (save_temps_prefix);
save_temps_prefix = NULL;
}
if (save_temps_flag && use_pipes)
{
if (save_temps_flag)
warning (0, "-pipe ignored because -save-temps specified");
use_pipes = 0;
}
if (!compare_debug)
{
const char *gcd = env.get ("GCC_COMPARE_DEBUG");
if (gcd && gcd[0] == '-')
{
compare_debug = 2;
compare_debug_opt = gcd;
}
else if (gcd && *gcd && strcmp (gcd, "0"))
{
compare_debug = 3;
compare_debug_opt = "-gtoggle";
}
}
else if (compare_debug < 0)
{
compare_debug = 0;
gcc_assert (!compare_debug_opt);
}
if (!gcc_exec_prefix)
{
#ifndef OS2
add_prefix (&exec_prefixes, standard_libexec_prefix, "GCC",
PREFIX_PRIORITY_LAST, 1, 0);
add_prefix (&exec_prefixes, standard_libexec_prefix, "BINUTILS",
PREFIX_PRIORITY_LAST, 2, 0);
add_prefix (&exec_prefixes, standard_exec_prefix, "BINUTILS",
PREFIX_PRIORITY_LAST, 2, 0);
#endif
add_prefix (&startfile_prefixes, standard_exec_prefix, "BINUTILS",
PREFIX_PRIORITY_LAST, 1, 0);
}
gcc_assert (!IS_ABSOLUTE_PATH (tooldir_base_prefix));
tooldir_prefix2 = concat (tooldir_base_prefix, spec_machine,
dir_separator_str, NULL);
tooldir_prefix
= concat (gcc_exec_prefix ? gcc_exec_prefix : standard_exec_prefix,
spec_host_machine, dir_separator_str, spec_version,
accel_dir_suffix, dir_separator_str, tooldir_prefix2, NULL);
free (tooldir_prefix2);
add_prefix (&exec_prefixes,
concat (tooldir_prefix, "bin", dir_separator_str, NULL),
"BINUTILS", PREFIX_PRIORITY_LAST, 0, 0);
add_prefix (&startfile_prefixes,
concat (tooldir_prefix, "lib", dir_separator_str, NULL),
"BINUTILS", PREFIX_PRIORITY_LAST, 0, 1);
free (tooldir_prefix);
#if defined(TARGET_SYSTEM_ROOT_RELOCATABLE) && !defined(VMS)
if (target_system_root && !target_system_root_changed && gcc_exec_prefix)
{
char *tmp_prefix = get_relative_prefix (decoded_options[0].arg,
standard_bindir_prefix,
target_system_root);
if (tmp_prefix && access_check (tmp_prefix, F_OK) == 0)
{
target_system_root = tmp_prefix;
target_system_root_changed = 1;
}
}
#endif
if (n_infiles == last_language_n_infiles && spec_lang != 0)
warning (0, "%<-x %s%> after last input file has no effect", spec_lang);
if (compare_debug == 2 || compare_debug == 3)
{
const char *opt = concat ("-fcompare-debug=", compare_debug_opt, NULL);
save_switch (opt, 0, NULL, false, true);
compare_debug = 1;
}
if (print_subprocess_help || print_help_list || print_version)
{
n_infiles = 0;
add_infile ("help-dummy", "c");
}
unsigned help_version_count = 0;
if (print_version)
help_version_count++;
if (print_help_list)
help_version_count++;
spec_undefvar_allowed =
((verbose_flag && decoded_options_count == 2)
|| help_version_count == decoded_options_count - 1);
alloc_switch ();
switches[n_switches].part1 = 0;
alloc_infile ();
infiles[n_infiles].name = 0;
}
static void
set_collect_gcc_options (void)
{
int i;
int first_time;
obstack_grow (&collect_obstack, "COLLECT_GCC_OPTIONS=",
sizeof ("COLLECT_GCC_OPTIONS=") - 1);
first_time = TRUE;
for (i = 0; (int) i < n_switches; i++)
{
const char *const *args;
const char *p, *q;
if (!first_time)
obstack_grow (&collect_obstack, " ", 1);
first_time = FALSE;
if ((switches[i].live_cond
& (SWITCH_IGNORE | SWITCH_KEEP_FOR_GCC))
== SWITCH_IGNORE)
continue;
obstack_grow (&collect_obstack, "'-", 2);
q = switches[i].part1;
while ((p = strchr (q, '\'')))
{
obstack_grow (&collect_obstack, q, p - q);
obstack_grow (&collect_obstack, "'\\''", 4);
q = ++p;
}
obstack_grow (&collect_obstack, q, strlen (q));
obstack_grow (&collect_obstack, "'", 1);
for (args = switches[i].args; args && *args; args++)
{
obstack_grow (&collect_obstack, " '", 2);
q = *args;
while ((p = strchr (q, '\'')))
{
obstack_grow (&collect_obstack, q, p - q);
obstack_grow (&collect_obstack, "'\\''", 4);
q = ++p;
}
obstack_grow (&collect_obstack, q, strlen (q));
obstack_grow (&collect_obstack, "'", 1);
}
}
obstack_grow (&collect_obstack, "\0", 1);
xputenv (XOBFINISH (&collect_obstack, char *));
}

static const char *gcc_input_filename;
static int input_file_number;
size_t input_filename_length;
static int basename_length;
static int suffixed_basename_length;
static const char *input_basename;
static const char *input_suffix;
#ifndef HOST_LACKS_INODE_NUMBERS
static struct stat input_stat;
#endif
static int input_stat_set;
static struct compiler *input_file_compiler;
static int arg_going;
static int delete_this_arg;
static int this_is_output_file;
static int this_is_library_file;
static int this_is_linker_script;
static int input_from_pipe;
static const char *suffix_subst;
static void
end_going_arg (void)
{
if (arg_going)
{
const char *string;
obstack_1grow (&obstack, 0);
string = XOBFINISH (&obstack, const char *);
if (this_is_library_file)
string = find_file (string);
if (this_is_linker_script)
{
char * full_script_path = find_a_file (&startfile_prefixes, string, R_OK, true);
if (full_script_path == NULL)
{
error ("unable to locate default linker script %qs in the library search paths", string);
return;
}
store_arg ("--script", false, false);
string = full_script_path;
}
store_arg (string, delete_this_arg, this_is_output_file);
if (this_is_output_file)
outfiles[input_file_number] = string;
arg_going = 0;
}
}
static void
insert_wrapper (const char *wrapper)
{
int n = 0;
int i;
char *buf = xstrdup (wrapper);
char *p = buf;
unsigned int old_length = argbuf.length ();
do
{
n++;
while (*p == ',')
p++;
}
while ((p = strchr (p, ',')) != NULL);
argbuf.safe_grow (old_length + n);
memmove (argbuf.address () + n,
argbuf.address (),
old_length * sizeof (const_char_p));
i = 0;
p = buf;
do
{
while (*p == ',')
{
*p = 0;
p++;
}
argbuf[i] = p;
i++;
}
while ((p = strchr (p, ',')) != NULL);
gcc_assert (i == n);
}
int
do_spec (const char *spec)
{
int value;
value = do_spec_2 (spec);
if (value == 0)
{
if (argbuf.length () > 0
&& !strcmp (argbuf.last (), "|"))
argbuf.pop ();
set_collect_gcc_options ();
if (argbuf.length () > 0)
value = execute ();
}
return value;
}
static int
do_spec_2 (const char *spec)
{
int result;
clear_args ();
arg_going = 0;
delete_this_arg = 0;
this_is_output_file = 0;
this_is_library_file = 0;
this_is_linker_script = 0;
input_from_pipe = 0;
suffix_subst = NULL;
result = do_spec_1 (spec, 0, NULL);
end_going_arg ();
return result;
}
static void
do_option_spec (const char *name, const char *spec)
{
unsigned int i, value_count, value_len;
const char *p, *q, *value;
char *tmp_spec, *tmp_spec_p;
if (configure_default_options[0].name == NULL)
return;
for (i = 0; i < ARRAY_SIZE (configure_default_options); i++)
if (strcmp (configure_default_options[i].name, name) == 0)
break;
if (i == ARRAY_SIZE (configure_default_options))
return;
value = configure_default_options[i].value;
value_len = strlen (value);
value_count = 0;
p = spec;
while ((p = strstr (p, "%(VALUE)")) != NULL)
{
p ++;
value_count ++;
}
tmp_spec = (char *) alloca (strlen (spec) + 1
+ value_count * (value_len - strlen ("%(VALUE)")));
tmp_spec_p = tmp_spec;
q = spec;
while ((p = strstr (q, "%(VALUE)")) != NULL)
{
memcpy (tmp_spec_p, q, p - q);
tmp_spec_p = tmp_spec_p + (p - q);
memcpy (tmp_spec_p, value, value_len);
tmp_spec_p += value_len;
q = p + strlen ("%(VALUE)");
}
strcpy (tmp_spec_p, q);
do_self_spec (tmp_spec);
}
static void
do_self_spec (const char *spec)
{
int i;
do_spec_2 (spec);
do_spec_1 (" ", 0, NULL);
for (i = 0; i < n_switches; i++)
if ((switches[i].live_cond & SWITCH_IGNORE))
switches[i].live_cond |= SWITCH_IGNORE_PERMANENTLY;
if (argbuf.length () > 0)
{
const char **argbuf_copy;
struct cl_decoded_option *decoded_options;
struct cl_option_handlers handlers;
unsigned int decoded_options_count;
unsigned int j;
argbuf_copy = XNEWVEC (const char *,
argbuf.length () + 1);
argbuf_copy[0] = "";
memcpy (argbuf_copy + 1, argbuf.address (),
argbuf.length () * sizeof (const char *));
decode_cmdline_options_to_array (argbuf.length () + 1,
argbuf_copy,
CL_DRIVER, &decoded_options,
&decoded_options_count);
free (argbuf_copy);
set_option_handlers (&handlers);
for (j = 1; j < decoded_options_count; j++)
{
switch (decoded_options[j].opt_index)
{
case OPT_SPECIAL_input_file:
if (strcmp (decoded_options[j].arg, "-") != 0)
fatal_error (input_location,
"switch %qs does not start with %<-%>",
decoded_options[j].arg);
else
fatal_error (input_location,
"spec-generated switch is just %<-%>");
break;
case OPT_fcompare_debug_second:
case OPT_fcompare_debug:
case OPT_fcompare_debug_:
case OPT_o:
save_switch (decoded_options[j].canonical_option[0],
(decoded_options[j].canonical_option_num_elements
- 1),
&decoded_options[j].canonical_option[1], false, true);
break;
default:
read_cmdline_option (&global_options, &global_options_set,
decoded_options + j, UNKNOWN_LOCATION,
CL_DRIVER, &handlers, global_dc);
break;
}
}
free (decoded_options);
alloc_switch ();
switches[n_switches].part1 = 0;
}
}
struct spec_path_info {
const char *option;
const char *append;
size_t append_len;
bool omit_relative;
bool separate_options;
};
static void *
spec_path (char *path, void *data)
{
struct spec_path_info *info = (struct spec_path_info *) data;
size_t len = 0;
char save = 0;
if (info->omit_relative && !IS_ABSOLUTE_PATH (path))
return NULL;
if (info->append_len != 0)
{
len = strlen (path);
memcpy (path + len, info->append, info->append_len + 1);
}
if (!is_directory (path, true))
return NULL;
do_spec_1 (info->option, 1, NULL);
if (info->separate_options)
do_spec_1 (" ", 0, NULL);
if (info->append_len == 0)
{
len = strlen (path);
save = path[len - 1];
if (IS_DIR_SEPARATOR (path[len - 1]))
path[len - 1] = '\0';
}
do_spec_1 (path, 1, NULL);
do_spec_1 (" ", 0, NULL);
if (info->append_len == 0)
path[len - 1] = save;
return NULL;
}
static void
create_at_file (char **argv)
{
char *temp_file = make_temp_file ("");
char *at_argument = concat ("@", temp_file, NULL);
FILE *f = fopen (temp_file, "w");
int status;
if (f == NULL)
fatal_error (input_location, "could not open temporary response file %s",
temp_file);
status = writeargv (argv, f);
if (status)
fatal_error (input_location,
"could not write to temporary response file %s",
temp_file);
status = fclose (f);
if (EOF == status)
fatal_error (input_location, "could not close temporary response file %s",
temp_file);
store_arg (at_argument, 0, 0);
record_temp_file (temp_file, !save_temps_flag, !save_temps_flag);
}
static bool
compile_input_file_p (struct infile *infile)
{
if ((!infile->language) || (infile->language[0] != '*'))
if (infile->incompiler == input_file_compiler)
return true;
return false;
}
static void
do_specs_vec (vec<char_p> vec)
{
unsigned ix;
char *opt;
FOR_EACH_VEC_ELT (vec, ix, opt)
{
do_spec_1 (opt, 1, NULL);
do_spec_1 (" ", 0, NULL);
}
}
static int
do_spec_1 (const char *spec, int inswitch, const char *soft_matched_part)
{
const char *p = spec;
int c;
int i;
int value;
if (inswitch && !*p)
arg_going = 1;
while ((c = *p++))
switch (inswitch ? 'a' : c)
{
case '\n':
end_going_arg ();
if (argbuf.length () > 0
&& !strcmp (argbuf.last (), "|"))
{
if (use_pipes)
{
input_from_pipe = 1;
break;
}
else
argbuf.pop ();
}
set_collect_gcc_options ();
if (argbuf.length () > 0)
{
value = execute ();
if (value)
return value;
}
clear_args ();
arg_going = 0;
delete_this_arg = 0;
this_is_output_file = 0;
this_is_library_file = 0;
this_is_linker_script = 0;
input_from_pipe = 0;
break;
case '|':
end_going_arg ();
obstack_1grow (&obstack, c);
arg_going = 1;
break;
case '\t':
case ' ':
end_going_arg ();
delete_this_arg = 0;
this_is_output_file = 0;
this_is_library_file = 0;
this_is_linker_script = 0;
break;
case '%':
switch (c = *p++)
{
case 0:
fatal_error (input_location, "spec %qs invalid", spec);
case 'b':
if (save_temps_length)
obstack_grow (&obstack, save_temps_prefix, save_temps_length);
else
obstack_grow (&obstack, input_basename, basename_length);
if (compare_debug < 0)
obstack_grow (&obstack, ".gk", 3);
arg_going = 1;
break;
case 'B':
if (save_temps_length)
obstack_grow (&obstack, save_temps_prefix, save_temps_length);
else
obstack_grow (&obstack, input_basename, suffixed_basename_length);
if (compare_debug < 0)
obstack_grow (&obstack, ".gk", 3);
arg_going = 1;
break;
case 'd':
delete_this_arg = 2;
break;
case 'D':
{
struct spec_path_info info;
info.option = "-L";
info.append_len = 0;
#ifdef RELATIVE_PREFIX_NOT_LINKDIR
info.omit_relative = true;
#else
info.omit_relative = false;
#endif
info.separate_options = false;
for_each_path (&startfile_prefixes, true, 0, spec_path, &info);
}
break;
case 'e':
{
const char *q = p;
char *buf;
while (*p != 0 && *p != '\n')
p++;
buf = (char *) alloca (p - q + 1);
strncpy (buf, q, p - q);
buf[p - q] = 0;
error ("%s", _(buf));
return -1;
}
break;
case 'n':
{
const char *q = p;
char *buf;
while (*p != 0 && *p != '\n')
p++;
buf = (char *) alloca (p - q + 1);
strncpy (buf, q, p - q);
buf[p - q] = 0;
inform (UNKNOWN_LOCATION, "%s", _(buf));
if (*p)
p++;
}
break;
case 'j':
{
struct stat st;
if ((!save_temps_flag)
&& (stat (HOST_BIT_BUCKET, &st) == 0) && (!S_ISDIR (st.st_mode))
&& (access (HOST_BIT_BUCKET, W_OK) == 0))
{
obstack_grow (&obstack, HOST_BIT_BUCKET,
strlen (HOST_BIT_BUCKET));
delete_this_arg = 0;
arg_going = 1;
break;
}
}
goto create_temp_file;
case '|':
if (use_pipes)
{
obstack_1grow (&obstack, '-');
delete_this_arg = 0;
arg_going = 1;
while (*p == '.' || ISALNUM ((unsigned char) *p))
p++;
if (p[0] == '%' && p[1] == 'O')
p += 2;
break;
}
goto create_temp_file;
case 'm':
if (use_pipes)
{
while (*p == '.' || ISALNUM ((unsigned char) *p))
p++;
if (p[0] == '%' && p[1] == 'O')
p += 2;
break;
}
goto create_temp_file;
case 'g':
case 'u':
case 'U':
create_temp_file:
{
struct temp_name *t;
int suffix_length;
const char *suffix = p;
char *saved_suffix = NULL;
while (*p == '.' || ISALNUM ((unsigned char) *p))
p++;
suffix_length = p - suffix;
if (p[0] == '%' && p[1] == 'O')
{
p += 2;
if (*p == '.' || ISALNUM ((unsigned char) *p))
fatal_error (input_location,
"spec %qs has invalid %<%%0%c%>", spec, *p);
if (suffix_length == 0)
suffix = TARGET_OBJECT_SUFFIX;
else
{
saved_suffix
= XNEWVEC (char, suffix_length
+ strlen (TARGET_OBJECT_SUFFIX) + 1);
strncpy (saved_suffix, suffix, suffix_length);
strcpy (saved_suffix + suffix_length,
TARGET_OBJECT_SUFFIX);
}
suffix_length += strlen (TARGET_OBJECT_SUFFIX);
}
if (compare_debug < 0)
{
suffix = concat (".gk", suffix, NULL);
suffix_length += 3;
}
if (save_temps_length)
{
char *tmp;
temp_filename_length
= save_temps_length + suffix_length + 1;
tmp = (char *) alloca (temp_filename_length);
memcpy (tmp, save_temps_prefix, save_temps_length);
memcpy (tmp + save_temps_length, suffix, suffix_length);
tmp[save_temps_length + suffix_length] = '\0';
temp_filename = save_string (tmp, save_temps_length
+ suffix_length);
obstack_grow (&obstack, temp_filename,
temp_filename_length);
arg_going = 1;
delete_this_arg = 0;
break;
}
if (save_temps_flag)
{
char *tmp;
temp_filename_length = basename_length + suffix_length + 1;
tmp = (char *) alloca (temp_filename_length);
memcpy (tmp, input_basename, basename_length);
memcpy (tmp + basename_length, suffix, suffix_length);
tmp[basename_length + suffix_length] = '\0';
temp_filename = tmp;
if (filename_cmp (temp_filename, gcc_input_filename) != 0)
{
#ifndef HOST_LACKS_INODE_NUMBERS
struct stat st_temp;
if (input_stat_set == 0)
{
input_stat_set = stat (gcc_input_filename,
&input_stat);
if (input_stat_set >= 0)
input_stat_set = 1;
}
if (input_stat_set != 1
|| stat (temp_filename, &st_temp) < 0
|| input_stat.st_dev != st_temp.st_dev
|| input_stat.st_ino != st_temp.st_ino)
#else
char* input_realname = lrealpath (gcc_input_filename);
char* temp_realname = lrealpath (temp_filename);
bool files_differ = filename_cmp (input_realname, temp_realname);
free (input_realname);
free (temp_realname);
if (files_differ)
#endif
{
temp_filename
= save_string (temp_filename,
temp_filename_length - 1);
obstack_grow (&obstack, temp_filename,
temp_filename_length);
arg_going = 1;
delete_this_arg = 0;
break;
}
}
}
for (t = temp_names; t; t = t->next)
if (t->length == suffix_length
&& strncmp (t->suffix, suffix, suffix_length) == 0
&& t->unique == (c == 'u' || c == 'U' || c == 'j'))
break;
if (t == 0 || c == 'u' || c == 'j')
{
if (t == 0)
{
t = XNEW (struct temp_name);
t->next = temp_names;
temp_names = t;
}
t->length = suffix_length;
if (saved_suffix)
{
t->suffix = saved_suffix;
saved_suffix = NULL;
}
else
t->suffix = save_string (suffix, suffix_length);
t->unique = (c == 'u' || c == 'U' || c == 'j');
temp_filename = make_temp_file (t->suffix);
temp_filename_length = strlen (temp_filename);
t->filename = temp_filename;
t->filename_length = temp_filename_length;
}
free (saved_suffix);
obstack_grow (&obstack, t->filename, t->filename_length);
delete_this_arg = 1;
}
arg_going = 1;
break;
case 'i':
if (combine_inputs)
{
if (at_file_supplied)
{
char **argv;
int n_files = 0;
int j;
for (i = 0; i < n_infiles; i++)
if (compile_input_file_p (&infiles[i]))
n_files++;
argv = (char **) alloca (sizeof (char *) * (n_files + 1));
for (i = 0, j = 0; i < n_infiles; i++)
if (compile_input_file_p (&infiles[i]))
{
argv[j] = CONST_CAST (char *, infiles[i].name);
infiles[i].compiled = true;
j++;
}
argv[j] = NULL;
create_at_file (argv);
}
else
for (i = 0; (int) i < n_infiles; i++)
if (compile_input_file_p (&infiles[i]))
{
store_arg (infiles[i].name, 0, 0);
infiles[i].compiled = true;
}
}
else
{
obstack_grow (&obstack, gcc_input_filename,
input_filename_length);
arg_going = 1;
}
break;
case 'I':
{
struct spec_path_info info;
if (multilib_dir)
{
do_spec_1 ("-imultilib", 1, NULL);
do_spec_1 (" ", 0, NULL);
do_spec_1 (multilib_dir, 1, NULL);
do_spec_1 (" ", 0, NULL);
}
if (multiarch_dir)
{
do_spec_1 ("-imultiarch", 1, NULL);
do_spec_1 (" ", 0, NULL);
do_spec_1 (multiarch_dir, 1, NULL);
do_spec_1 (" ", 0, NULL);
}
if (gcc_exec_prefix)
{
do_spec_1 ("-iprefix", 1, NULL);
do_spec_1 (" ", 0, NULL);
do_spec_1 (gcc_exec_prefix, 1, NULL);
do_spec_1 (" ", 0, NULL);
}
if (target_system_root_changed ||
(target_system_root && target_sysroot_hdrs_suffix))
{
do_spec_1 ("-isysroot", 1, NULL);
do_spec_1 (" ", 0, NULL);
do_spec_1 (target_system_root, 1, NULL);
if (target_sysroot_hdrs_suffix)
do_spec_1 (target_sysroot_hdrs_suffix, 1, NULL);
do_spec_1 (" ", 0, NULL);
}
info.option = "-isystem";
info.append = "include";
info.append_len = strlen (info.append);
info.omit_relative = false;
info.separate_options = true;
for_each_path (&include_prefixes, false, info.append_len,
spec_path, &info);
info.append = "include-fixed";
if (*sysroot_hdrs_suffix_spec)
info.append = concat (info.append, dir_separator_str,
multilib_dir, NULL);
info.append_len = strlen (info.append);
for_each_path (&include_prefixes, false, info.append_len,
spec_path, &info);
}
break;
case 'o':
{
int max = n_infiles;
max += lang_specific_extra_outfiles;
if (HAVE_GNU_LD && at_file_supplied)
{
char **argv;
int n_files, j;
for (n_files = 0, i = 0; i < max; i++)
n_files += outfiles[i] != NULL;
argv = (char **) alloca (sizeof (char *) * (n_files + 1));
for (i = 0, j = 0; i < max; i++)
if (outfiles[i])
{
argv[j] = CONST_CAST (char *, outfiles[i]);
j++;
}
argv[j] = NULL;
create_at_file (argv);
}
else
for (i = 0; i < max; i++)
if (outfiles[i])
store_arg (outfiles[i], 0, 0);
break;
}
case 'O':
obstack_grow (&obstack, TARGET_OBJECT_SUFFIX, strlen (TARGET_OBJECT_SUFFIX));
arg_going = 1;
break;
case 's':
this_is_library_file = 1;
break;
case 'T':
this_is_linker_script = 1;
break;
case 'V':
outfiles[input_file_number] = NULL;
break;
case 'w':
this_is_output_file = 1;
break;
case 'W':
{
unsigned int cur_index = argbuf.length ();
if (*p != '{')
fatal_error (input_location,
"spec %qs has invalid %<%%W%c%>", spec, *p);
p = handle_braces (p + 1);
if (p == 0)
return -1;
end_going_arg ();
if (argbuf.length () != cur_index)
record_temp_file (argbuf.last (), 0, 1);
break;
}
case 'x':
{
const char *p1 = p;
char *string;
char *opt;
unsigned ix;
if (*p != '{')
fatal_error (input_location,
"spec %qs has invalid %<%%x%c%>", spec, *p);
while (*p++ != '}')
;
string = save_string (p1 + 1, p - p1 - 2);
FOR_EACH_VEC_ELT (linker_options, ix, opt)
if (! strcmp (string, opt))
{
free (string);
return 0;
}
add_linker_option (string, strlen (string));
free (string);
}
break;
case 'X':
do_specs_vec (linker_options);
break;
case 'Y':
do_specs_vec (assembler_options);
break;
case 'Z':
do_specs_vec (preprocessor_options);
break;
case '1':
value = do_spec_1 (cc1_spec, 0, NULL);
if (value != 0)
return value;
break;
case '2':
value = do_spec_1 (cc1plus_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'a':
value = do_spec_1 (asm_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'A':
value = do_spec_1 (asm_final_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'C':
{
const char *const spec
= (input_file_compiler->cpp_spec
? input_file_compiler->cpp_spec
: cpp_spec);
value = do_spec_1 (spec, 0, NULL);
if (value != 0)
return value;
}
break;
case 'E':
value = do_spec_1 (endfile_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'l':
value = do_spec_1 (link_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'L':
value = do_spec_1 (lib_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'M':
if (multilib_os_dir == NULL)
obstack_1grow (&obstack, '.');
else
obstack_grow (&obstack, multilib_os_dir,
strlen (multilib_os_dir));
break;
case 'G':
value = do_spec_1 (libgcc_spec, 0, NULL);
if (value != 0)
return value;
break;
case 'R':
if (target_system_root)
{
obstack_grow (&obstack, target_system_root,
strlen (target_system_root));
if (target_sysroot_suffix)
obstack_grow (&obstack, target_sysroot_suffix,
strlen (target_sysroot_suffix));
}
break;
case 'S':
value = do_spec_1 (startfile_spec, 0, NULL);
if (value != 0)
return value;
break;
case '{':
p = handle_braces (p);
if (p == 0)
return -1;
break;
case ':':
p = handle_spec_function (p, NULL);
if (p == 0)
return -1;
break;
case '%':
obstack_1grow (&obstack, '%');
break;
case '.':
{
unsigned len = 0;
while (p[len] && p[len] != ' ' && p[len] != '%')
len++;
suffix_subst = save_string (p - 1, len + 1);
p += len;
}
break;
case '<':
case '>':
{
unsigned len = 0;
int have_wildcard = 0;
int i;
int switch_option;
if (c == '>')
switch_option = SWITCH_IGNORE | SWITCH_KEEP_FOR_GCC;
else
switch_option = SWITCH_IGNORE;
while (p[len] && p[len] != ' ' && p[len] != '\t')
len++;
if (p[len-1] == '*')
have_wildcard = 1;
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, p, len - have_wildcard)
&& (have_wildcard || switches[i].part1[len] == '\0'))
{
switches[i].live_cond |= switch_option;
if (switches[i].known)
switches[i].validated = true;
}
p += len;
}
break;
case '*':
if (soft_matched_part)
{
if (soft_matched_part[0])
do_spec_1 (soft_matched_part, 1, NULL);
if (*p == 0 || *p == '}')
do_spec_1 (" ", 0, NULL);
}
else
error ("spec failure: %<%%*%> has not been initialized by pattern match");
break;
case '(':
{
const char *name = p;
struct spec_list *sl;
int len;
while (*p && *p != ')')
p++;
for (len = p - name, sl = specs; sl; sl = sl->next)
if (sl->name_len == len && !strncmp (sl->name, name, len))
{
name = *(sl->ptr_spec);
#ifdef DEBUG_SPECS
fnotice (stderr, "Processing spec (%s), which is '%s'\n",
sl->name, name);
#endif
break;
}
if (sl)
{
value = do_spec_1 (name, 0, NULL);
if (value != 0)
return value;
}
if (*p)
p++;
}
break;
default:
error ("spec failure: unrecognized spec option %qc", c);
break;
}
break;
case '\\':
c = *p++;
default:
obstack_1grow (&obstack, c);
arg_going = 1;
}
if (processing_spec_function)
end_going_arg ();
return 0;
}
static const struct spec_function *
lookup_spec_function (const char *name)
{
const struct spec_function *sf;
for (sf = static_spec_functions; sf->name != NULL; sf++)
if (strcmp (sf->name, name) == 0)
return sf;
return NULL;
}
static const char *
eval_spec_function (const char *func, const char *args)
{
const struct spec_function *sf;
const char *funcval;
vec<const_char_p> save_argbuf;
int save_arg_going;
int save_delete_this_arg;
int save_this_is_output_file;
int save_this_is_library_file;
int save_input_from_pipe;
int save_this_is_linker_script;
const char *save_suffix_subst;
int save_growing_size;
void *save_growing_value = NULL;
sf = lookup_spec_function (func);
if (sf == NULL)
fatal_error (input_location, "unknown spec function %qs", func);
save_argbuf = argbuf;
save_arg_going = arg_going;
save_delete_this_arg = delete_this_arg;
save_this_is_output_file = this_is_output_file;
save_this_is_library_file = this_is_library_file;
save_this_is_linker_script = this_is_linker_script;
save_input_from_pipe = input_from_pipe;
save_suffix_subst = suffix_subst;
save_growing_size = obstack_object_size (&obstack);
if (save_growing_size > 0)
save_growing_value = obstack_finish (&obstack);
alloc_args ();
if (do_spec_2 (args) < 0)
fatal_error (input_location, "error in args to spec function %qs", func);
funcval = (*sf->func) (argbuf.length (),
argbuf.address ());
argbuf.release ();
argbuf = save_argbuf;
arg_going = save_arg_going;
delete_this_arg = save_delete_this_arg;
this_is_output_file = save_this_is_output_file;
this_is_library_file = save_this_is_library_file;
this_is_linker_script = save_this_is_linker_script;
input_from_pipe = save_input_from_pipe;
suffix_subst = save_suffix_subst;
if (save_growing_size > 0)
obstack_grow (&obstack, save_growing_value, save_growing_size);
return funcval;
}
static const char *
handle_spec_function (const char *p, bool *retval_nonnull)
{
char *func, *args;
const char *endp, *funcval;
int count;
processing_spec_function++;
for (endp = p; *endp != '\0'; endp++)
{
if (*endp == '(')		
break;
if (!ISALNUM (*endp) && !(*endp == '-' || *endp == '_'))
fatal_error (input_location, "malformed spec function name");
}
if (*endp != '(')		
fatal_error (input_location, "no arguments for spec function");
func = save_string (p, endp - p);
p = ++endp;
for (count = 0; *endp != '\0'; endp++)
{
if (*endp == ')')
{
if (count == 0)
break;
count--;
}
else if (*endp == '(')	
count++;
}
if (*endp != ')')
fatal_error (input_location, "malformed spec function arguments");
args = save_string (p, endp - p);
p = ++endp;
funcval = eval_spec_function (func, args);
if (funcval != NULL && do_spec_1 (funcval, 0, NULL) < 0)
p = NULL;
if (retval_nonnull)
*retval_nonnull = funcval != NULL;
free (func);
free (args);
processing_spec_function--;
return p;
}
static inline bool
input_suffix_matches (const char *atom, const char *end_atom)
{
return (input_suffix
&& !strncmp (input_suffix, atom, end_atom - atom)
&& input_suffix[end_atom - atom] == '\0');
}
static bool
input_spec_matches (const char *atom, const char *end_atom)
{
return (input_file_compiler
&& input_file_compiler->suffix
&& input_file_compiler->suffix[0] != '\0'
&& !strncmp (input_file_compiler->suffix + 1, atom,
end_atom - atom)
&& input_file_compiler->suffix[end_atom - atom + 1] == '\0');
}
static bool
switch_matches (const char *atom, const char *end_atom, int starred)
{
int i;
int len = end_atom - atom;
int plen = starred ? len : -1;
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, atom, len)
&& (starred || switches[i].part1[len] == '\0')
&& check_live_switch (i, plen))
return true;
else if (switches[i].args != 0)
{
if ((*switches[i].part1 == 'D' || *switches[i].part1 == 'U')
&& *switches[i].part1 == atom[0])
{
if (!strncmp (switches[i].args[0], &atom[1], len - 1)
&& (starred || (switches[i].part1[1] == '\0'
&& switches[i].args[0][len - 1] == '\0'))
&& check_live_switch (i, (starred ? 1 : -1)))
return true;
}
}
return false;
}
static inline void
mark_matching_switches (const char *atom, const char *end_atom, int starred)
{
int i;
int len = end_atom - atom;
int plen = starred ? len : -1;
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, atom, len)
&& (starred || switches[i].part1[len] == '\0')
&& check_live_switch (i, plen))
switches[i].ordering = 1;
}
static inline void
process_marked_switches (void)
{
int i;
for (i = 0; i < n_switches; i++)
if (switches[i].ordering == 1)
{
switches[i].ordering = 0;
give_switch (i, 0);
}
}
static const char *
handle_braces (const char *p)
{
const char *atom, *end_atom;
const char *d_atom = NULL, *d_end_atom = NULL;
char *esc_buf = NULL, *d_esc_buf = NULL;
int esc;
const char *orig = p;
bool a_is_suffix;
bool a_is_spectype;
bool a_is_starred;
bool a_is_negated;
bool a_matched;
bool a_must_be_last = false;
bool ordered_set    = false;
bool disjunct_set   = false;
bool disj_matched   = false;
bool disj_starred   = true;
bool n_way_choice   = false;
bool n_way_matched  = false;
#define SKIP_WHITE() do { while (*p == ' ' || *p == '\t') p++; } while (0)
do
{
if (a_must_be_last)
goto invalid;
a_matched = false;
a_is_suffix = false;
a_is_starred = false;
a_is_negated = false;
a_is_spectype = false;
SKIP_WHITE ();
if (*p == '!')
p++, a_is_negated = true;
SKIP_WHITE ();
if (*p == '%' && p[1] == ':')
{
atom = NULL;
end_atom = NULL;
p = handle_spec_function (p + 2, &a_matched);
}
else
{
if (*p == '.')
p++, a_is_suffix = true;
else if (*p == ',')
p++, a_is_spectype = true;
atom = p;
esc = 0;
while (ISIDNUM (*p) || *p == '-' || *p == '+' || *p == '='
|| *p == ',' || *p == '.' || *p == '@' || *p == '\\')
{
if (*p == '\\')
{
p++;
if (!*p)
fatal_error (input_location,
"braced spec %qs ends in escape", orig);
esc++;
}
p++;
}
end_atom = p;
if (esc)
{
const char *ap;
char *ep;
if (esc_buf && esc_buf != d_esc_buf)
free (esc_buf);
esc_buf = NULL;
ep = esc_buf = (char *) xmalloc (end_atom - atom - esc + 1);
for (ap = atom; ap != end_atom; ap++, ep++)
{
if (*ap == '\\')
ap++;
*ep = *ap;
}
*ep = '\0';
atom = esc_buf;
end_atom = ep;
}
if (*p == '*')
p++, a_is_starred = 1;
}
SKIP_WHITE ();
switch (*p)
{
case '&': case '}':
ordered_set = true;
if (disjunct_set || n_way_choice || a_is_negated || a_is_suffix
|| a_is_spectype || atom == end_atom)
goto invalid;
mark_matching_switches (atom, end_atom, a_is_starred);
if (*p == '}')
process_marked_switches ();
break;
case '|': case ':':
disjunct_set = true;
if (ordered_set)
goto invalid;
if (atom && atom == end_atom)
{
if (!n_way_choice || disj_matched || *p == '|'
|| a_is_negated || a_is_suffix || a_is_spectype
|| a_is_starred)
goto invalid;
a_must_be_last = true;
disj_matched = !n_way_matched;
disj_starred = false;
}
else
{
if ((a_is_suffix || a_is_spectype) && a_is_starred)
goto invalid;
if (!a_is_starred)
disj_starred = false;
if (!disj_matched && !n_way_matched)
{
if (atom == NULL)
;
else if (a_is_suffix)
a_matched = input_suffix_matches (atom, end_atom);
else if (a_is_spectype)
a_matched = input_spec_matches (atom, end_atom);
else
a_matched = switch_matches (atom, end_atom, a_is_starred);
if (a_matched != a_is_negated)
{
disj_matched = true;
d_atom = atom;
d_end_atom = end_atom;
d_esc_buf = esc_buf;
}
}
}
if (*p == ':')
{
p = process_brace_body (p + 1, d_atom, d_end_atom, disj_starred,
disj_matched && !n_way_matched);
if (p == 0)
goto done;
if (*p == ';')
{
n_way_choice = true;
n_way_matched |= disj_matched;
disj_matched = false;
disj_starred = true;
d_atom = d_end_atom = NULL;
}
}
break;
default:
goto invalid;
}
}
while (*p++ != '}');
done:
if (d_esc_buf && d_esc_buf != esc_buf)
free (d_esc_buf);
if (esc_buf)
free (esc_buf);
return p;
invalid:
fatal_error (input_location, "braced spec %qs is invalid at %qc", orig, *p);
#undef SKIP_WHITE
}
static const char *
process_brace_body (const char *p, const char *atom, const char *end_atom,
int starred, int matched)
{
const char *body, *end_body;
unsigned int nesting_level;
bool have_subst     = false;
body = p;
nesting_level = 1;
for (;;)
{
if (*p == '{')
nesting_level++;
else if (*p == '}')
{
if (!--nesting_level)
break;
}
else if (*p == ';' && nesting_level == 1)
break;
else if (*p == '%' && p[1] == '*' && nesting_level == 1)
have_subst = true;
else if (*p == '\0')
goto invalid;
p++;
}
end_body = p;
while (end_body[-1] == ' ' || end_body[-1] == '\t')
end_body--;
if (have_subst && !starred)
goto invalid;
if (matched)
{
char *string = save_string (body, end_body - body);
if (!have_subst)
{
if (do_spec_1 (string, 0, NULL) < 0)
{
free (string);
return 0;
}
}
else
{
unsigned int hard_match_len = end_atom - atom;
int i;
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, atom, hard_match_len)
&& check_live_switch (i, hard_match_len))
{
if (do_spec_1 (string, 0,
&switches[i].part1[hard_match_len]) < 0)
{
free (string);
return 0;
}
give_switch (i, 1);
suffix_subst = NULL;
}
}
free (string);
}
return p;
invalid:
fatal_error (input_location, "braced spec body %qs is invalid", body);
}

static int
check_live_switch (int switchnum, int prefix_length)
{
const char *name = switches[switchnum].part1;
int i;
if (switches[switchnum].live_cond != 0)
return ((switches[switchnum].live_cond & SWITCH_LIVE) != 0
&& (switches[switchnum].live_cond & SWITCH_FALSE) == 0
&& (switches[switchnum].live_cond & SWITCH_IGNORE_PERMANENTLY)
== 0);
if (prefix_length >= 0 && prefix_length <= 1)
return 1;
switch (*name)
{
case 'O':
for (i = switchnum + 1; i < n_switches; i++)
if (switches[i].part1[0] == 'O')
{
switches[switchnum].validated = true;
switches[switchnum].live_cond = SWITCH_FALSE;
return 0;
}
break;
case 'W':  case 'f':  case 'm': case 'g':
if (! strncmp (name + 1, "no-", 3))
{
for (i = switchnum + 1; i < n_switches; i++)
if (switches[i].part1[0] == name[0]
&& ! strcmp (&switches[i].part1[1], &name[4]))
{
if (switches[switchnum].known)
switches[switchnum].validated = true;
switches[switchnum].live_cond = SWITCH_FALSE;
return 0;
}
}
else
{
for (i = switchnum + 1; i < n_switches; i++)
if (switches[i].part1[0] == name[0]
&& switches[i].part1[1] == 'n'
&& switches[i].part1[2] == 'o'
&& switches[i].part1[3] == '-'
&& !strcmp (&switches[i].part1[4], &name[1]))
{
if (switches[switchnum].known)
switches[switchnum].validated = true;
switches[switchnum].live_cond = SWITCH_FALSE;
return 0;
}
}
break;
}
switches[switchnum].live_cond |= SWITCH_LIVE;
return 1;
}

static void
give_switch (int switchnum, int omit_first_word)
{
if ((switches[switchnum].live_cond & SWITCH_IGNORE) != 0)
return;
if (!omit_first_word)
{
do_spec_1 ("-", 0, NULL);
do_spec_1 (switches[switchnum].part1, 1, NULL);
}
if (switches[switchnum].args != 0)
{
const char **p;
for (p = switches[switchnum].args; *p; p++)
{
const char *arg = *p;
do_spec_1 (" ", 0, NULL);
if (suffix_subst)
{
unsigned length = strlen (arg);
int dot = 0;
while (length-- && !IS_DIR_SEPARATOR (arg[length]))
if (arg[length] == '.')
{
(CONST_CAST (char *, arg))[length] = 0;
dot = 1;
break;
}
do_spec_1 (arg, 1, NULL);
if (dot)
(CONST_CAST (char *, arg))[length] = '.';
do_spec_1 (suffix_subst, 1, NULL);
}
else
do_spec_1 (arg, 1, NULL);
}
}
do_spec_1 (" ", 0, NULL);
switches[switchnum].validated = true;
}

static void
print_configuration (FILE *file)
{
int n;
const char *thrmod;
fnotice (file, "Target: %s\n", spec_machine);
fnotice (file, "Configured with: %s\n", configuration_arguments);
#ifdef THREAD_MODEL_SPEC
obstack_init (&obstack);
do_spec_1 (THREAD_MODEL_SPEC, 0, thread_model);
obstack_1grow (&obstack, '\0');
thrmod = XOBFINISH (&obstack, const char *);
#else
thrmod = thread_model;
#endif
fnotice (file, "Thread model: %s\n", thrmod);
for (n = 0; version_string[n]; n++)
if (version_string[n] == ' ')
break;
if (! strncmp (version_string, compiler_version, n)
&& compiler_version[n] == 0)
fnotice (file, "gcc version %s %s\n", version_string,
pkgversion_string);
else
fnotice (file, "gcc driver version %s %sexecuting gcc version %s\n",
version_string, pkgversion_string, compiler_version);
}
#define RETRY_ICE_ATTEMPTS 3
static bool
files_equal_p (char *file1, char *file2)
{
struct stat st1, st2;
off_t n, len;
int fd1, fd2;
const int bufsize = 8192;
char *buf = XNEWVEC (char, bufsize);
fd1 = open (file1, O_RDONLY);
fd2 = open (file2, O_RDONLY);
if (fd1 < 0 || fd2 < 0)
goto error;
if (fstat (fd1, &st1) < 0 || fstat (fd2, &st2) < 0)
goto error;
if (st1.st_size != st2.st_size)
goto error;
for (n = st1.st_size; n; n -= len)
{
len = n;
if ((int) len > bufsize / 2)
len = bufsize / 2;
if (read (fd1, buf, len) != (int) len
|| read (fd2, buf + bufsize / 2, len) != (int) len)
{
goto error;
}
if (memcmp (buf, buf + bufsize / 2, len) != 0)
goto error;
}
free (buf);
close (fd1);
close (fd2);
return 1;
error:
free (buf);
close (fd1);
close (fd2);
return 0;
}
static bool
check_repro (char **temp_stdout_files, char **temp_stderr_files)
{
int i;
for (i = 0; i < RETRY_ICE_ATTEMPTS - 2; ++i)
{
if (!files_equal_p (temp_stdout_files[i], temp_stdout_files[i + 1])
|| !files_equal_p (temp_stderr_files[i], temp_stderr_files[i + 1]))
{
fnotice (stderr, "The bug is not reproducible, so it is"
" likely a hardware or OS problem.\n");
break;
}
}
return i == RETRY_ICE_ATTEMPTS - 2;
}
enum attempt_status {
ATTEMPT_STATUS_FAIL_TO_RUN,
ATTEMPT_STATUS_SUCCESS,
ATTEMPT_STATUS_ICE
};
static enum attempt_status
run_attempt (const char **new_argv, const char *out_temp,
const char *err_temp, int emit_system_info, int append)
{
if (emit_system_info)
{
FILE *file_out = fopen (err_temp, "a");
print_configuration (file_out);
fputs ("\n", file_out);
fclose (file_out);
}
int exit_status;
const char *errmsg;
struct pex_obj *pex;
int err;
int pex_flags = PEX_USE_PIPES | PEX_LAST;
enum attempt_status status = ATTEMPT_STATUS_FAIL_TO_RUN;
if (append)
pex_flags |= PEX_STDOUT_APPEND | PEX_STDERR_APPEND;
pex = pex_init (PEX_USE_PIPES, new_argv[0], NULL);
if (!pex)
fatal_error (input_location, "pex_init failed: %m");
errmsg = pex_run (pex, pex_flags, new_argv[0],
CONST_CAST2 (char *const *, const char **, &new_argv[1]), out_temp,
err_temp, &err);
if (errmsg != NULL)
{
if (err == 0)
fatal_error (input_location, errmsg);
else
{
errno = err;
pfatal_with_name (errmsg);
}
}
if (!pex_get_status (pex, 1, &exit_status))
goto out;
switch (WEXITSTATUS (exit_status))
{
case ICE_EXIT_CODE:
status = ATTEMPT_STATUS_ICE;
break;
case SUCCESS_EXIT_CODE:
status = ATTEMPT_STATUS_SUCCESS;
break;
default:
;
}
out:
pex_free (pex);
return status;
}
static void
insert_comments (const char *file_in, const char *file_out)
{
FILE *in = fopen (file_in, "rb");
FILE *out = fopen (file_out, "wb");
char line[256];
bool add_comment = true;
while (fgets (line, sizeof (line), in))
{
if (add_comment)
fputs ("
fputs (line, out);
add_comment = strchr (line, '\n') != NULL;
}
fclose (in);
fclose (out);
}
static void
do_report_bug (const char **new_argv, const int nargs,
char **out_file, char **err_file)
{
int i, status;
int fd = open (*out_file, O_RDWR | O_APPEND);
if (fd < 0)
return;
write (fd, "\n
for (i = 0; i < nargs; i++)
{
write (fd, " ", 1);
write (fd, new_argv[i], strlen (new_argv[i]));
}
write (fd, "\n\n", 2);
close (fd);
new_argv[nargs] = "-E";
new_argv[nargs + 1] = NULL;
status = run_attempt (new_argv, *out_file, *err_file, 0, 1);
if (status == ATTEMPT_STATUS_SUCCESS)
{
fnotice (stderr, "Preprocessed source stored into %s file,"
" please attach this to your bugreport.\n", *out_file);
free (*out_file);
*out_file = NULL;
}
}
static void
try_generate_repro (const char **argv)
{
int i, nargs, out_arg = -1, quiet = 0, attempt;
const char **new_argv;
char *temp_files[RETRY_ICE_ATTEMPTS * 2];
char **temp_stdout_files = &temp_files[0];
char **temp_stderr_files = &temp_files[RETRY_ICE_ATTEMPTS];
if (gcc_input_filename == NULL || ! strcmp (gcc_input_filename, "-"))
return;
for (nargs = 0; argv[nargs] != NULL; ++nargs)
if (! strcmp (argv[nargs], "-E"))
return;
else if (argv[nargs][0] == '-' && argv[nargs][1] == 'o')
{
if (out_arg == -1)
out_arg = nargs;
else
return;
}
else if (! strcmp (argv[nargs], "-quiet"))
quiet = 1;
else if (! strcmp (argv[nargs], "-ftime-report"))
return;
if (out_arg == -1 || !quiet)
return;
memset (temp_files, '\0', sizeof (temp_files));
new_argv = XALLOCAVEC (const char *, nargs + 4);
memcpy (new_argv, argv, (nargs + 1) * sizeof (const char *));
new_argv[nargs++] = "-frandom-seed=0";
new_argv[nargs++] = "-fdump-noaddr";
new_argv[nargs] = NULL;
if (new_argv[out_arg][2] == '\0')
new_argv[out_arg + 1] = "-";
else
new_argv[out_arg] = "-o-";
int status;
for (attempt = 0; attempt < RETRY_ICE_ATTEMPTS; ++attempt)
{
int emit_system_info = 0;
int append = 0;
temp_stdout_files[attempt] = make_temp_file (".out");
temp_stderr_files[attempt] = make_temp_file (".err");
if (attempt == RETRY_ICE_ATTEMPTS - 1)
{
append = 1;
emit_system_info = 1;
}
status = run_attempt (new_argv, temp_stdout_files[attempt],
temp_stderr_files[attempt], emit_system_info,
append);
if (status != ATTEMPT_STATUS_ICE)
{
fnotice (stderr, "The bug is not reproducible, so it is"
" likely a hardware or OS problem.\n");
goto out;
}
}
if (!check_repro (temp_stdout_files, temp_stderr_files))
goto out;
{
char **stderr_commented = &temp_stdout_files[RETRY_ICE_ATTEMPTS - 1];
insert_comments (temp_stderr_files[RETRY_ICE_ATTEMPTS - 1],
*stderr_commented);
char **err = &temp_stderr_files[RETRY_ICE_ATTEMPTS - 1];
do_report_bug (new_argv, nargs, stderr_commented, err);
}
out:
for (i = 0; i < RETRY_ICE_ATTEMPTS * 2; i++)
if (temp_files[i])
{
unlink (temp_stdout_files[i]);
free (temp_stdout_files[i]);
}
}
static const char *
find_file (const char *name)
{
char *newname = find_a_file (&startfile_prefixes, name, R_OK, true);
return newname ? newname : name;
}
static int
is_directory (const char *path1, bool linker)
{
int len1;
char *path;
char *cp;
struct stat st;
len1 = strlen (path1);
path = (char *) alloca (3 + len1);
memcpy (path, path1, len1);
cp = path + len1;
if (!IS_DIR_SEPARATOR (cp[-1]))
*cp++ = DIR_SEPARATOR;
*cp++ = '.';
*cp = '\0';
if (linker
&& IS_DIR_SEPARATOR (path[0])
&& ((cp - path == 6
&& filename_ncmp (path + 1, "lib", 3) == 0)
|| (cp - path == 10
&& filename_ncmp (path + 1, "usr", 3) == 0
&& IS_DIR_SEPARATOR (path[4])
&& filename_ncmp (path + 5, "lib", 3) == 0)))
return 0;
return (stat (path, &st) >= 0 && S_ISDIR (st.st_mode));
}
void
set_input (const char *filename)
{
const char *p;
gcc_input_filename = filename;
input_filename_length = strlen (gcc_input_filename);
input_basename = lbasename (gcc_input_filename);
basename_length = strlen (input_basename);
suffixed_basename_length = basename_length;
p = input_basename + basename_length;
while (p != input_basename && *p != '.')
--p;
if (*p == '.' && p != input_basename)
{
basename_length = p - input_basename;
input_suffix = p + 1;
}
else
input_suffix = "";
input_stat_set = 0;
}

static void
fatal_signal (int signum)
{
signal (signum, SIG_DFL);
delete_failure_queue ();
delete_temp_files ();
kill (getpid (), signum);
}
static int
compare_files (char *cmpfile[])
{
int ret = 0;
FILE *temp[2] = { NULL, NULL };
int i;
#if HAVE_MMAP_FILE
{
size_t length[2];
void *map[2] = { NULL, NULL };
for (i = 0; i < 2; i++)
{
struct stat st;
if (stat (cmpfile[i], &st) < 0 || !S_ISREG (st.st_mode))
{
error ("%s: could not determine length of compare-debug file %s",
gcc_input_filename, cmpfile[i]);
ret = 1;
break;
}
length[i] = st.st_size;
}
if (!ret && length[0] != length[1])
{
error ("%s: -fcompare-debug failure (length)", gcc_input_filename);
ret = 1;
}
if (!ret)
for (i = 0; i < 2; i++)
{
int fd = open (cmpfile[i], O_RDONLY);
if (fd < 0)
{
error ("%s: could not open compare-debug file %s",
gcc_input_filename, cmpfile[i]);
ret = 1;
break;
}
map[i] = mmap (NULL, length[i], PROT_READ, MAP_PRIVATE, fd, 0);
close (fd);
if (map[i] == (void *) MAP_FAILED)
{
ret = -1;
break;
}
}
if (!ret)
{
if (memcmp (map[0], map[1], length[0]) != 0)
{
error ("%s: -fcompare-debug failure", gcc_input_filename);
ret = 1;
}
}
for (i = 0; i < 2; i++)
if (map[i])
munmap ((caddr_t) map[i], length[i]);
if (ret >= 0)
return ret;
ret = 0;
}
#endif
for (i = 0; i < 2; i++)
{
temp[i] = fopen (cmpfile[i], "r");
if (!temp[i])
{
error ("%s: could not open compare-debug file %s",
gcc_input_filename, cmpfile[i]);
ret = 1;
break;
}
}
if (!ret && temp[0] && temp[1])
for (;;)
{
int c0, c1;
c0 = fgetc (temp[0]);
c1 = fgetc (temp[1]);
if (c0 != c1)
{
error ("%s: -fcompare-debug failure",
gcc_input_filename);
ret = 1;
break;
}
if (c0 == EOF)
break;
}
for (i = 1; i >= 0; i--)
{
if (temp[i])
fclose (temp[i]);
}
return ret;
}
driver::driver (bool can_finalize, bool debug) :
explicit_link_files (NULL),
decoded_options (NULL),
m_option_suggestions (NULL)
{
env.init (can_finalize, debug);
}
driver::~driver ()
{
XDELETEVEC (explicit_link_files);
XDELETEVEC (decoded_options);
if (m_option_suggestions)
{
int i;
char *str;
FOR_EACH_VEC_ELT (*m_option_suggestions, i, str)
free (str);
delete m_option_suggestions;
}
}
int
driver::main (int argc, char **argv)
{
bool early_exit;
set_progname (argv[0]);
expand_at_files (&argc, &argv);
decode_argv (argc, const_cast <const char **> (argv));
global_initializations ();
build_multilib_strings ();
set_up_specs ();
putenv_COLLECT_GCC (argv[0]);
maybe_putenv_COLLECT_LTO_WRAPPER ();
maybe_putenv_OFFLOAD_TARGETS ();
handle_unrecognized_options ();
if (!maybe_print_and_exit ())
return 0;
early_exit = prepare_infiles ();
if (early_exit)
return get_exit_code ();
do_spec_on_infiles ();
maybe_run_linker (argv[0]);
final_actions ();
return get_exit_code ();
}
void
driver::set_progname (const char *argv0) const
{
const char *p = argv0 + strlen (argv0);
while (p != argv0 && !IS_DIR_SEPARATOR (p[-1]))
--p;
progname = p;
xmalloc_set_program_name (progname);
}
void
driver::expand_at_files (int *argc, char ***argv) const
{
char **old_argv = *argv;
expandargv (argc, argv);
if (*argv != old_argv)
at_file_supplied = true;
}
void
driver::decode_argv (int argc, const char **argv)
{
global_init_params ();
finish_params ();
init_opts_obstack ();
init_options_struct (&global_options, &global_options_set);
decode_cmdline_options_to_array (argc, argv,
CL_DRIVER,
&decoded_options, &decoded_options_count);
}
void
driver::global_initializations ()
{
unlock_std_streams ();
gcc_init_libintl ();
diagnostic_initialize (global_dc, 0);
diagnostic_color_init (global_dc);
#ifdef GCC_DRIVER_HOST_INITIALIZATION
GCC_DRIVER_HOST_INITIALIZATION;
#endif
if (atexit (delete_temp_files) != 0)
fatal_error (input_location, "atexit failed");
if (signal (SIGINT, SIG_IGN) != SIG_IGN)
signal (SIGINT, fatal_signal);
#ifdef SIGHUP
if (signal (SIGHUP, SIG_IGN) != SIG_IGN)
signal (SIGHUP, fatal_signal);
#endif
if (signal (SIGTERM, SIG_IGN) != SIG_IGN)
signal (SIGTERM, fatal_signal);
#ifdef SIGPIPE
if (signal (SIGPIPE, SIG_IGN) != SIG_IGN)
signal (SIGPIPE, fatal_signal);
#endif
#ifdef SIGCHLD
signal (SIGCHLD, SIG_DFL);
#endif
stack_limit_increase (64 * 1024 * 1024);
alloc_args ();
obstack_init (&obstack);
}
void
driver::build_multilib_strings () const
{
{
const char *p;
const char *const *q = multilib_raw;
int need_space;
obstack_init (&multilib_obstack);
while ((p = *q++) != (char *) 0)
obstack_grow (&multilib_obstack, p, strlen (p));
obstack_1grow (&multilib_obstack, 0);
multilib_select = XOBFINISH (&multilib_obstack, const char *);
q = multilib_matches_raw;
while ((p = *q++) != (char *) 0)
obstack_grow (&multilib_obstack, p, strlen (p));
obstack_1grow (&multilib_obstack, 0);
multilib_matches = XOBFINISH (&multilib_obstack, const char *);
q = multilib_exclusions_raw;
while ((p = *q++) != (char *) 0)
obstack_grow (&multilib_obstack, p, strlen (p));
obstack_1grow (&multilib_obstack, 0);
multilib_exclusions = XOBFINISH (&multilib_obstack, const char *);
q = multilib_reuse_raw;
while ((p = *q++) != (char *) 0)
obstack_grow (&multilib_obstack, p, strlen (p));
obstack_1grow (&multilib_obstack, 0);
multilib_reuse = XOBFINISH (&multilib_obstack, const char *);
need_space = FALSE;
for (size_t i = 0; i < ARRAY_SIZE (multilib_defaults_raw); i++)
{
if (need_space)
obstack_1grow (&multilib_obstack, ' ');
obstack_grow (&multilib_obstack,
multilib_defaults_raw[i],
strlen (multilib_defaults_raw[i]));
need_space = TRUE;
}
obstack_1grow (&multilib_obstack, 0);
multilib_defaults = XOBFINISH (&multilib_obstack, const char *);
}
}
void
driver::set_up_specs () const
{
const char *spec_machine_suffix;
char *specs_file;
size_t i;
#ifdef INIT_ENVIRONMENT
xputenv (INIT_ENVIRONMENT);
#endif
process_command (decoded_options_count, decoded_options);
compilers = XNEWVAR (struct compiler, sizeof default_compilers);
memcpy (compilers, default_compilers, sizeof default_compilers);
n_compilers = n_default_compilers;
machine_suffix = concat (spec_host_machine, dir_separator_str, spec_version,
accel_dir_suffix, dir_separator_str, NULL);
just_machine_suffix = concat (spec_machine, dir_separator_str, NULL);
specs_file = find_a_file (&startfile_prefixes, "specs", R_OK, true);
if (specs_file != 0 && strcmp (specs_file, "specs"))
read_specs (specs_file, true, false);
else
init_spec ();
#ifdef ACCEL_COMPILER
spec_machine_suffix = machine_suffix;
#else
spec_machine_suffix = just_machine_suffix;
#endif
specs_file = (char *) alloca (strlen (standard_exec_prefix)
+ strlen (spec_machine_suffix) + sizeof ("specs"));
strcpy (specs_file, standard_exec_prefix);
strcat (specs_file, spec_machine_suffix);
strcat (specs_file, "specs");
if (access (specs_file, R_OK) == 0)
read_specs (specs_file, true, false);
for (i = 0; i < ARRAY_SIZE (option_default_specs); i++)
do_option_spec (option_default_specs[i].name,
option_default_specs[i].spec);
for (i = 0; i < ARRAY_SIZE (driver_self_specs); i++)
do_self_spec (driver_self_specs[i]);
if (*cross_compile == '0')
{
if (*md_exec_prefix)
{
add_prefix (&exec_prefixes, md_exec_prefix, "GCC",
PREFIX_PRIORITY_LAST, 0, 0);
}
}
if (*sysroot_suffix_spec != 0
&& !no_sysroot_suffix
&& do_spec_2 (sysroot_suffix_spec) == 0)
{
if (argbuf.length () > 1)
error ("spec failure: more than one arg to SYSROOT_SUFFIX_SPEC");
else if (argbuf.length () == 1)
target_sysroot_suffix = xstrdup (argbuf.last ());
}
#ifdef HAVE_LD_SYSROOT
if (target_system_root)
{
obstack_grow (&obstack, "%(sysroot_spec) ", strlen ("%(sysroot_spec) "));
obstack_grow0 (&obstack, link_spec, strlen (link_spec));
set_spec ("link", XOBFINISH (&obstack, const char *), false);
}
#endif
if (*sysroot_hdrs_suffix_spec != 0
&& !no_sysroot_suffix
&& do_spec_2 (sysroot_hdrs_suffix_spec) == 0)
{
if (argbuf.length () > 1)
error ("spec failure: more than one arg to SYSROOT_HEADERS_SUFFIX_SPEC");
else if (argbuf.length () == 1)
target_sysroot_hdrs_suffix = xstrdup (argbuf.last ());
}
if (*startfile_prefix_spec != 0
&& do_spec_2 (startfile_prefix_spec) == 0
&& do_spec_1 (" ", 0, NULL) == 0)
{
const char *arg;
int ndx;
FOR_EACH_VEC_ELT (argbuf, ndx, arg)
add_sysrooted_prefix (&startfile_prefixes, arg, "BINUTILS",
PREFIX_PRIORITY_LAST, 0, 1);
}
else if (*cross_compile == '0' || target_system_root)
{
if (*md_startfile_prefix)
add_sysrooted_prefix (&startfile_prefixes, md_startfile_prefix,
"GCC", PREFIX_PRIORITY_LAST, 0, 1);
if (*md_startfile_prefix_1)
add_sysrooted_prefix (&startfile_prefixes, md_startfile_prefix_1,
"GCC", PREFIX_PRIORITY_LAST, 0, 1);
if (IS_ABSOLUTE_PATH (standard_startfile_prefix))
add_sysrooted_prefix (&startfile_prefixes,
standard_startfile_prefix, "BINUTILS",
PREFIX_PRIORITY_LAST, 0, 1);
else if (*cross_compile == '0')
{
add_prefix (&startfile_prefixes,
concat (gcc_exec_prefix
? gcc_exec_prefix : standard_exec_prefix,
machine_suffix,
standard_startfile_prefix, NULL),
NULL, PREFIX_PRIORITY_LAST, 0, 1);
}
if (*standard_startfile_prefix_1)
add_sysrooted_prefix (&startfile_prefixes,
standard_startfile_prefix_1, "BINUTILS",
PREFIX_PRIORITY_LAST, 0, 1);
if (*standard_startfile_prefix_2)
add_sysrooted_prefix (&startfile_prefixes,
standard_startfile_prefix_2, "BINUTILS",
PREFIX_PRIORITY_LAST, 0, 1);
}
for (struct user_specs *uptr = user_specs_head; uptr; uptr = uptr->next)
{
char *filename = find_a_file (&startfile_prefixes, uptr->filename,
R_OK, true);
read_specs (filename ? filename : uptr->filename, false, true);
}
{
struct spec_list *sl;
for (sl = specs; sl; sl = sl->next)
if (sl->name_len == sizeof "self_spec" - 1
&& !strcmp (sl->name, "self_spec"))
do_self_spec (*sl->ptr_spec);
}
if (compare_debug)
{
enum save_temps save;
if (!compare_debug_second)
{
n_switches_debug_check[1] = n_switches;
n_switches_alloc_debug_check[1] = n_switches_alloc;
switches_debug_check[1] = XDUPVEC (struct switchstr, switches,
n_switches_alloc);
do_self_spec ("%:compare-debug-self-opt()");
n_switches_debug_check[0] = n_switches;
n_switches_alloc_debug_check[0] = n_switches_alloc;
switches_debug_check[0] = switches;
n_switches = n_switches_debug_check[1];
n_switches_alloc = n_switches_alloc_debug_check[1];
switches = switches_debug_check[1];
}
save = save_temps_flag;
save_temps_flag = SAVE_TEMPS_NONE;
compare_debug = -compare_debug;
do_self_spec ("%:compare-debug-self-opt()");
save_temps_flag = save;
if (!compare_debug_second)
{
n_switches_debug_check[1] = n_switches;
n_switches_alloc_debug_check[1] = n_switches_alloc;
switches_debug_check[1] = switches;
compare_debug = -compare_debug;
n_switches = n_switches_debug_check[0];
n_switches_alloc = n_switches_debug_check[0];
switches = switches_debug_check[0];
}
}
if (gcc_exec_prefix)
gcc_exec_prefix = concat (gcc_exec_prefix, spec_host_machine,
dir_separator_str, spec_version,
accel_dir_suffix, dir_separator_str, NULL);
validate_all_switches ();
set_multilib_dir ();
}
void
driver::putenv_COLLECT_GCC (const char *argv0) const
{
obstack_init (&collect_obstack);
obstack_grow (&collect_obstack, "COLLECT_GCC=", sizeof ("COLLECT_GCC=") - 1);
obstack_grow (&collect_obstack, argv0, strlen (argv0) + 1);
xputenv (XOBFINISH (&collect_obstack, char *));
}
void
driver::maybe_putenv_COLLECT_LTO_WRAPPER () const
{
char *lto_wrapper_file;
if (have_c)
lto_wrapper_file = NULL;
else
lto_wrapper_file = find_a_file (&exec_prefixes, "lto-wrapper",
X_OK, false);
if (lto_wrapper_file)
{
lto_wrapper_file = convert_white_space (lto_wrapper_file);
lto_wrapper_spec = lto_wrapper_file;
obstack_init (&collect_obstack);
obstack_grow (&collect_obstack, "COLLECT_LTO_WRAPPER=",
sizeof ("COLLECT_LTO_WRAPPER=") - 1);
obstack_grow (&collect_obstack, lto_wrapper_spec,
strlen (lto_wrapper_spec) + 1);
xputenv (XOBFINISH (&collect_obstack, char *));
}
}
void
driver::maybe_putenv_OFFLOAD_TARGETS () const
{
if (offload_targets && offload_targets[0] != '\0')
{
obstack_grow (&collect_obstack, "OFFLOAD_TARGET_NAMES=",
sizeof ("OFFLOAD_TARGET_NAMES=") - 1);
obstack_grow (&collect_obstack, offload_targets,
strlen (offload_targets) + 1);
xputenv (XOBFINISH (&collect_obstack, char *));
}
free (offload_targets);
offload_targets = NULL;
}
void
driver::build_option_suggestions (void)
{
gcc_assert (m_option_suggestions == NULL);
m_option_suggestions = new auto_vec <char *> ();
for (unsigned int i = 0; i < cl_options_count; i++)
{
const struct cl_option *option = &cl_options[i];
const char *opt_text = option->opt_text;
switch (i)
{
default:
if (option->var_type == CLVC_ENUM)
{
const struct cl_enum *e = &cl_enums[option->var_enum];
for (unsigned j = 0; e->values[j].arg != NULL; j++)
{
char *with_arg = concat (opt_text, e->values[j].arg, NULL);
add_misspelling_candidates (m_option_suggestions, option,
with_arg);
free (with_arg);
}
}
else
add_misspelling_candidates (m_option_suggestions, option,
opt_text);
break;
case OPT_fsanitize_:
case OPT_fsanitize_recover_:
{
for (int j = 0; sanitizer_opts[j].name != NULL; ++j)
{
struct cl_option optb;
if (sanitizer_opts[j].flag == ~0U && i == OPT_fsanitize_)
{
optb = *option;
optb.opt_text = opt_text = "-fno-sanitize=";
optb.cl_reject_negative = true;
option = &optb;
}
char *with_arg = concat (opt_text,
sanitizer_opts[j].name,
NULL);
add_misspelling_candidates (m_option_suggestions, option,
with_arg);
free (with_arg);
}
}
break;
}
}
}
const char *
driver::suggest_option (const char *bad_opt)
{
if (!m_option_suggestions)
build_option_suggestions ();
gcc_assert (m_option_suggestions);
return find_closest_string
(bad_opt,
(auto_vec <const char *> *) m_option_suggestions);
}
void
driver::handle_unrecognized_options ()
{
for (size_t i = 0; (int) i < n_switches; i++)
if (! switches[i].validated)
{
const char *hint = suggest_option (switches[i].part1);
if (hint)
error ("unrecognized command line option %<-%s%>;"
" did you mean %<-%s%>?",
switches[i].part1, hint);
else
error ("unrecognized command line option %<-%s%>",
switches[i].part1);
}
}
int
driver::maybe_print_and_exit () const
{
if (print_search_dirs)
{
printf (_("install: %s%s\n"),
gcc_exec_prefix ? gcc_exec_prefix : standard_exec_prefix,
gcc_exec_prefix ? "" : machine_suffix);
printf (_("programs: %s\n"),
build_search_list (&exec_prefixes, "", false, false));
printf (_("libraries: %s\n"),
build_search_list (&startfile_prefixes, "", false, true));
return (0);
}
if (print_file_name)
{
printf ("%s\n", find_file (print_file_name));
return (0);
}
if (print_prog_name)
{
if (use_ld != NULL && ! strcmp (print_prog_name, "ld"))
{
#ifdef DEFAULT_LINKER
char *ld;
# ifdef HAVE_HOST_EXECUTABLE_SUFFIX
int len = (sizeof (DEFAULT_LINKER)
- sizeof (HOST_EXECUTABLE_SUFFIX));
ld = NULL;
if (len > 0)
{
char *default_linker = xstrdup (DEFAULT_LINKER);
if (! strcmp (&default_linker[len], HOST_EXECUTABLE_SUFFIX))
{
default_linker[len] = '\0';
ld = concat (default_linker, use_ld,
HOST_EXECUTABLE_SUFFIX, NULL);
}
}
if (ld == NULL)
# endif
ld = concat (DEFAULT_LINKER, use_ld, NULL);
if (access (ld, X_OK) == 0)
{
printf ("%s\n", ld);
return (0);
}
#endif
print_prog_name = concat (print_prog_name, use_ld, NULL);
}
char *newname = find_a_file (&exec_prefixes, print_prog_name, X_OK, 0);
printf ("%s\n", (newname ? newname : print_prog_name));
return (0);
}
if (print_multi_lib)
{
print_multilib_info ();
return (0);
}
if (print_multi_directory)
{
if (multilib_dir == NULL)
printf (".\n");
else
printf ("%s\n", multilib_dir);
return (0);
}
if (print_multiarch)
{
if (multiarch_dir == NULL)
printf ("\n");
else
printf ("%s\n", multiarch_dir);
return (0);
}
if (print_sysroot)
{
if (target_system_root)
{
if (target_sysroot_suffix)
printf ("%s%s\n", target_system_root, target_sysroot_suffix);
else
printf ("%s\n", target_system_root);
}
return (0);
}
if (print_multi_os_directory)
{
if (multilib_os_dir == NULL)
printf (".\n");
else
printf ("%s\n", multilib_os_dir);
return (0);
}
if (print_sysroot_headers_suffix)
{
if (*sysroot_hdrs_suffix_spec)
{
printf("%s\n", (target_sysroot_hdrs_suffix
? target_sysroot_hdrs_suffix
: ""));
return (0);
}
else
fatal_error (input_location,
"not configured with sysroot headers suffix");
}
if (print_help_list)
{
display_help ();
if (! verbose_flag)
{
printf (_("\nFor bug reporting instructions, please see:\n"));
printf ("%s.\n", bug_report_url);
return (0);
}
fputc ('\n', stdout);
fflush (stdout);
}
if (print_version)
{
printf (_("%s %s%s\n"), progname, pkgversion_string,
version_string);
printf ("Copyright %s 2018 Free Software Foundation, Inc.\n",
_("(C)"));
fputs (_("This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n"),
stdout);
if (! verbose_flag)
return 0;
fputc ('\n', stdout);
fflush (stdout);
}
if (verbose_flag)
{
print_configuration (stderr);
if (n_infiles == 0)
return (0);
}
return 1;
}
bool
driver::prepare_infiles ()
{
size_t i;
int lang_n_infiles = 0;
if (n_infiles == added_libraries)
fatal_error (input_location, "no input files");
if (seen_error ())
return true;
i = n_infiles;
i += lang_specific_extra_outfiles;
outfiles = XCNEWVEC (const char *, i);
explicit_link_files = XCNEWVEC (char, n_infiles);
combine_inputs = have_o || flag_wpa;
for (i = 0; (int) i < n_infiles; i++)
{
const char *name = infiles[i].name;
struct compiler *compiler = lookup_compiler (name,
strlen (name),
infiles[i].language);
if (compiler && !(compiler->combinable))
combine_inputs = false;
if (lang_n_infiles > 0 && compiler != input_file_compiler
&& infiles[i].language && infiles[i].language[0] != '*')
infiles[i].incompiler = compiler;
else if (compiler)
{
lang_n_infiles++;
input_file_compiler = compiler;
infiles[i].incompiler = compiler;
}
else
{
explicit_link_files[i] = 1;
infiles[i].incompiler = NULL;
}
infiles[i].compiled = false;
infiles[i].preprocessed = false;
}
if (!combine_inputs && have_c && have_o && lang_n_infiles > 1)
fatal_error (input_location,
"cannot specify -o with -c, -S or -E with multiple files");
return false;
}
void
driver::do_spec_on_infiles () const
{
size_t i;
for (i = 0; (int) i < n_infiles; i++)
{
int this_file_error = 0;
input_file_number = i;
set_input (infiles[i].name);
if (infiles[i].compiled)
continue;
outfiles[i] = gcc_input_filename;
input_file_compiler
= lookup_compiler (infiles[i].name, input_filename_length,
infiles[i].language);
if (input_file_compiler)
{
if (input_file_compiler->spec[0] == '#')
{
error ("%s: %s compiler not installed on this system",
gcc_input_filename, &input_file_compiler->spec[1]);
this_file_error = 1;
}
else
{
int value;
if (compare_debug)
{
free (debug_check_temp_file[0]);
debug_check_temp_file[0] = NULL;
free (debug_check_temp_file[1]);
debug_check_temp_file[1] = NULL;
}
value = do_spec (input_file_compiler->spec);
infiles[i].compiled = true;
if (value < 0)
this_file_error = 1;
else if (compare_debug && debug_check_temp_file[0])
{
if (verbose_flag)
inform (UNKNOWN_LOCATION,
"recompiling with -fcompare-debug");
compare_debug = -compare_debug;
n_switches = n_switches_debug_check[1];
n_switches_alloc = n_switches_alloc_debug_check[1];
switches = switches_debug_check[1];
value = do_spec (input_file_compiler->spec);
compare_debug = -compare_debug;
n_switches = n_switches_debug_check[0];
n_switches_alloc = n_switches_alloc_debug_check[0];
switches = switches_debug_check[0];
if (value < 0)
{
error ("during -fcompare-debug recompilation");
this_file_error = 1;
}
gcc_assert (debug_check_temp_file[1]
&& filename_cmp (debug_check_temp_file[0],
debug_check_temp_file[1]));
if (verbose_flag)
inform (UNKNOWN_LOCATION, "comparing final insns dumps");
if (compare_files (debug_check_temp_file))
this_file_error = 1;
}
if (compare_debug)
{
free (debug_check_temp_file[0]);
debug_check_temp_file[0] = NULL;
free (debug_check_temp_file[1]);
debug_check_temp_file[1] = NULL;
}
}
}
else
explicit_link_files[i] = 1;
if (this_file_error)
{
delete_failure_queue ();
errorcount++;
}
clear_failure_queue ();
}
if (n_infiles > 0)
{
int i;
for (i = 0; i < n_infiles ; i++)
if (infiles[i].incompiler
|| (infiles[i].language && infiles[i].language[0] != '*'))
{
set_input (infiles[i].name);
break;
}
}
if (!seen_error ())
{
input_file_number = n_infiles;
if (lang_specific_pre_link ())
errorcount++;
}
}
void
driver::maybe_run_linker (const char *argv0) const
{
size_t i;
int linker_was_run = 0;
int num_linker_inputs;
num_linker_inputs = 0;
for (i = 0; (int) i < n_infiles; i++)
if (explicit_link_files[i] || outfiles[i] != NULL)
num_linker_inputs++;
if (num_linker_inputs > 0 && !seen_error () && print_subprocess_help < 2)
{
int tmp = execution_count;
if (! have_c)
{
#if HAVE_LTO_PLUGIN > 0
#if HAVE_LTO_PLUGIN == 2
const char *fno_use_linker_plugin = "fno-use-linker-plugin";
#else
const char *fuse_linker_plugin = "fuse-linker-plugin";
#endif
#endif
if (! strcmp (linker_name_spec, "collect2"))
{
char *s = find_a_file (&exec_prefixes, "collect2", X_OK, false);
if (s == NULL)
linker_name_spec = "ld";
}
#if HAVE_LTO_PLUGIN > 0
#if HAVE_LTO_PLUGIN == 2
if (!switch_matches (fno_use_linker_plugin,
fno_use_linker_plugin
+ strlen (fno_use_linker_plugin), 0))
#else
if (switch_matches (fuse_linker_plugin,
fuse_linker_plugin
+ strlen (fuse_linker_plugin), 0))
#endif
{
char *temp_spec = find_a_file (&exec_prefixes,
LTOPLUGINSONAME, R_OK,
false);
if (!temp_spec)
fatal_error (input_location,
"-fuse-linker-plugin, but %s not found",
LTOPLUGINSONAME);
linker_plugin_file_spec = convert_white_space (temp_spec);
}
#endif
lto_gcc_spec = argv0;
}
putenv_from_prefixes (&exec_prefixes, "COMPILER_PATH", false);
putenv_from_prefixes (&startfile_prefixes, LIBRARY_PATH_ENV, true);
if (print_subprocess_help == 1)
{
printf (_("\nLinker options\n==============\n\n"));
printf (_("Use \"-Wl,OPTION\" to pass \"OPTION\""
" to the linker.\n\n"));
fflush (stdout);
}
int value = do_spec (link_command_spec);
if (value < 0)
errorcount = 1;
linker_was_run = (tmp != execution_count);
}
if (! linker_was_run && !seen_error ())
for (i = 0; (int) i < n_infiles; i++)
if (explicit_link_files[i]
&& !(infiles[i].language && infiles[i].language[0] == '*'))
warning (0, "%s: linker input file unused because linking not done",
outfiles[i]);
}
void
driver::final_actions () const
{
if (seen_error ())
delete_failure_queue ();
delete_temp_files ();
if (print_help_list)
{
printf (("\nFor bug reporting instructions, please see:\n"));
printf ("%s\n", bug_report_url);
}
}
int
driver::get_exit_code () const
{
return (signal_count != 0 ? 2
: seen_error () ? (pass_exit_codes ? greatest_status : 1)
: 0);
}
static struct compiler *
lookup_compiler (const char *name, size_t length, const char *language)
{
struct compiler *cp;
if (language != 0 && language[0] == '*')
return 0;
if (language != 0)
{
for (cp = compilers + n_compilers - 1; cp >= compilers; cp--)
if (cp->suffix[0] == '@' && !strcmp (cp->suffix + 1, language))
{
if (name != NULL && strcmp (name, "-") == 0
&& (strcmp (cp->suffix, "@c-header") == 0
|| strcmp (cp->suffix, "@c++-header") == 0)
&& !have_E)
fatal_error (input_location,
"cannot use %<-%> as input filename for a "
"precompiled header");
return cp;
}
error ("language %s not recognized", language);
return 0;
}
for (cp = compilers + n_compilers - 1; cp >= compilers; cp--)
{
if (
(!strcmp (cp->suffix, "-") && !strcmp (name, "-"))
|| (strlen (cp->suffix) < length
&& !strcmp (cp->suffix,
name + length - strlen (cp->suffix))
))
break;
}
#if defined (OS2) ||defined (HAVE_DOS_BASED_FILE_SYSTEM)
if (cp < compilers)
for (cp = compilers + n_compilers - 1; cp >= compilers; cp--)
{
if (
(!strcmp (cp->suffix, "-") && !strcmp (name, "-"))
|| (strlen (cp->suffix) < length
&& ((!strcmp (cp->suffix,
name + length - strlen (cp->suffix))
|| !strpbrk (cp->suffix, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
&& !strcasecmp (cp->suffix,
name + length - strlen (cp->suffix)))
))
break;
}
#endif
if (cp >= compilers)
{
if (cp->spec[0] != '@')
return cp;
return lookup_compiler (NULL, 0, cp->spec + 1);
}
return 0;
}

static char *
save_string (const char *s, int len)
{
char *result = XNEWVEC (char, len + 1);
gcc_checking_assert (strlen (s) >= (unsigned int) len);
memcpy (result, s, len);
result[len] = 0;
return result;
}
void
pfatal_with_name (const char *name)
{
perror_with_name (name);
delete_temp_files ();
exit (1);
}
static void
perror_with_name (const char *name)
{
error ("%s: %m", name);
}

static inline void
validate_switches_from_spec (const char *spec, bool user)
{
const char *p = spec;
char c;
while ((c = *p++))
if (c == '%' && (*p == '{' || *p == '<' || (*p == 'W' && *++p == '{')))
p = validate_switches (p + 1, user);
}
static void
validate_all_switches (void)
{
struct compiler *comp;
struct spec_list *spec;
for (comp = compilers; comp->spec; comp++)
validate_switches_from_spec (comp->spec, false);
for (spec = specs; spec; spec = spec->next)
validate_switches_from_spec (*spec->ptr_spec, spec->user_p);
validate_switches_from_spec (link_command_spec, false);
}
static const char *
validate_switches (const char *start, bool user_spec)
{
const char *p = start;
const char *atom;
size_t len;
int i;
bool suffix = false;
bool starred = false;
#define SKIP_WHITE() do { while (*p == ' ' || *p == '\t') p++; } while (0)
next_member:
SKIP_WHITE ();
if (*p == '!')
p++;
SKIP_WHITE ();
if (*p == '.' || *p == ',')
suffix = true, p++;
atom = p;
while (ISIDNUM (*p) || *p == '-' || *p == '+' || *p == '='
|| *p == ',' || *p == '.' || *p == '@')
p++;
len = p - atom;
if (*p == '*')
starred = true, p++;
SKIP_WHITE ();
if (!suffix)
{
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, atom, len)
&& (starred || switches[i].part1[len] == '\0')
&& (switches[i].known || user_spec))
switches[i].validated = true;
}
if (*p) p++;
if (*p && (p[-1] == '|' || p[-1] == '&'))
goto next_member;
if (*p && p[-1] == ':')
{
while (*p && *p != ';' && *p != '}')
{
if (*p == '%')
{
p++;
if (*p == '{' || *p == '<')
p = validate_switches (p+1, user_spec);
else if (p[0] == 'W' && p[1] == '{')
p = validate_switches (p+2, user_spec);
}
else
p++;
}
if (*p) p++;
if (*p && p[-1] == ';')
goto next_member;
}
return p;
#undef SKIP_WHITE
}

struct mdswitchstr
{
const char *str;
int len;
};
static struct mdswitchstr *mdswitches;
static int n_mdswitches;
class used_arg_t
{
public:
int operator () (const char *p, int len);
void finalize ();
private:
struct mswitchstr
{
const char *str;
const char *replace;
int len;
int rep_len;
};
mswitchstr *mswitches;
int n_mswitches;
};
used_arg_t used_arg;
int
used_arg_t::operator () (const char *p, int len)
{
int i, j;
if (!mswitches)
{
struct mswitchstr *matches;
const char *q;
int cnt = 0;
for (q = multilib_matches; *q != '\0'; q++)
if (*q == ';')
cnt++;
matches
= (struct mswitchstr *) alloca ((sizeof (struct mswitchstr)) * cnt);
i = 0;
q = multilib_matches;
while (*q != '\0')
{
matches[i].str = q;
while (*q != ' ')
{
if (*q == '\0')
{
invalid_matches:
fatal_error (input_location, "multilib spec %qs is invalid",
multilib_matches);
}
q++;
}
matches[i].len = q - matches[i].str;
matches[i].replace = ++q;
while (*q != ';' && *q != '\0')
{
if (*q == ' ')
goto invalid_matches;
q++;
}
matches[i].rep_len = q - matches[i].replace;
i++;
if (*q == ';')
q++;
}
mswitches
= XNEWVEC (struct mswitchstr, n_mdswitches + (n_switches ? n_switches : 1));
for (i = 0; i < n_switches; i++)
if ((switches[i].live_cond & SWITCH_IGNORE) == 0)
{
int xlen = strlen (switches[i].part1);
for (j = 0; j < cnt; j++)
if (xlen == matches[j].len
&& ! strncmp (switches[i].part1, matches[j].str, xlen))
{
mswitches[n_mswitches].str = matches[j].replace;
mswitches[n_mswitches].len = matches[j].rep_len;
mswitches[n_mswitches].replace = (char *) 0;
mswitches[n_mswitches].rep_len = 0;
n_mswitches++;
break;
}
}
for (i = 0; i < n_mdswitches; i++)
{
const char *r;
for (q = multilib_options; *q != '\0'; *q && q++)
{
while (*q == ' ')
q++;
r = q;
while (strncmp (q, mdswitches[i].str, mdswitches[i].len) != 0
|| strchr (" /", q[mdswitches[i].len]) == NULL)
{
while (*q != ' ' && *q != '/' && *q != '\0')
q++;
if (*q != '/')
break;
q++;
}
if (*q != ' ' && *q != '\0')
{
while (*r != ' ' && *r != '\0')
{
q = r;
while (*q != ' ' && *q != '/' && *q != '\0')
q++;
if (used_arg (r, q - r))
break;
if (*q != '/')
{
mswitches[n_mswitches].str = mdswitches[i].str;
mswitches[n_mswitches].len = mdswitches[i].len;
mswitches[n_mswitches].replace = (char *) 0;
mswitches[n_mswitches].rep_len = 0;
n_mswitches++;
break;
}
r = q + 1;
}
break;
}
}
}
}
for (i = 0; i < n_mswitches; i++)
if (len == mswitches[i].len && ! strncmp (p, mswitches[i].str, len))
return 1;
return 0;
}
void used_arg_t::finalize ()
{
XDELETEVEC (mswitches);
mswitches = NULL;
n_mswitches = 0;
}
static int
default_arg (const char *p, int len)
{
int i;
for (i = 0; i < n_mdswitches; i++)
if (len == mdswitches[i].len && ! strncmp (p, mdswitches[i].str, len))
return 1;
return 0;
}
static void
set_multilib_dir (void)
{
const char *p;
unsigned int this_path_len;
const char *this_path, *this_arg;
const char *start, *end;
int not_arg;
int ok, ndfltok, first;
n_mdswitches = 0;
start = multilib_defaults;
while (*start == ' ' || *start == '\t')
start++;
while (*start != '\0')
{
n_mdswitches++;
while (*start != ' ' && *start != '\t' && *start != '\0')
start++;
while (*start == ' ' || *start == '\t')
start++;
}
if (n_mdswitches)
{
int i = 0;
mdswitches = XNEWVEC (struct mdswitchstr, n_mdswitches);
for (start = multilib_defaults; *start != '\0'; start = end + 1)
{
while (*start == ' ' || *start == '\t')
start++;
if (*start == '\0')
break;
for (end = start + 1;
*end != ' ' && *end != '\t' && *end != '\0'; end++)
;
obstack_grow (&multilib_obstack, start, end - start);
obstack_1grow (&multilib_obstack, 0);
mdswitches[i].str = XOBFINISH (&multilib_obstack, const char *);
mdswitches[i++].len = end - start;
if (*end == '\0')
break;
}
}
p = multilib_exclusions;
while (*p != '\0')
{
if (*p == '\n')
{
++p;
continue;
}
ok = 1;
while (*p != ';')
{
if (*p == '\0')
{
invalid_exclusions:
fatal_error (input_location, "multilib exclusions %qs is invalid",
multilib_exclusions);
}
if (! ok)
{
++p;
continue;
}
this_arg = p;
while (*p != ' ' && *p != ';')
{
if (*p == '\0')
goto invalid_exclusions;
++p;
}
if (*this_arg != '!')
not_arg = 0;
else
{
not_arg = 1;
++this_arg;
}
ok = used_arg (this_arg, p - this_arg);
if (not_arg)
ok = ! ok;
if (*p == ' ')
++p;
}
if (ok)
return;
++p;
}
first = 1;
p = multilib_select;
if (strlen (multilib_reuse) > 0)
p = concat (p, multilib_reuse, NULL);
while (*p != '\0')
{
if (*p == '\n')
{
++p;
continue;
}
this_path = p;
while (*p != ' ')
{
if (*p == '\0')
{
invalid_select:
fatal_error (input_location, "multilib select %qs %qs is invalid",
multilib_select, multilib_reuse);
}
++p;
}
this_path_len = p - this_path;
ok = 1;
ndfltok = 1;
++p;
while (*p != ';')
{
if (*p == '\0')
goto invalid_select;
if (! ok)
{
++p;
continue;
}
this_arg = p;
while (*p != ' ' && *p != ';')
{
if (*p == '\0')
goto invalid_select;
++p;
}
if (*this_arg != '!')
not_arg = 0;
else
{
not_arg = 1;
++this_arg;
}
ok = used_arg (this_arg, p - this_arg);
if (not_arg)
ok = ! ok;
if (! ok)
ndfltok = 0;
if (default_arg (this_arg, p - this_arg))
ok = 1;
if (*p == ' ')
++p;
}
if (ok && first)
{
if (this_path_len != 1
|| this_path[0] != '.')
{
char *new_multilib_dir = XNEWVEC (char, this_path_len + 1);
char *q;
strncpy (new_multilib_dir, this_path, this_path_len);
new_multilib_dir[this_path_len] = '\0';
q = strchr (new_multilib_dir, ':');
if (q != NULL)
*q = '\0';
multilib_dir = new_multilib_dir;
}
first = 0;
}
if (ndfltok)
{
const char *q = this_path, *end = this_path + this_path_len;
while (q < end && *q != ':')
q++;
if (q < end)
{
const char *q2 = q + 1, *ml_end = end;
char *new_multilib_os_dir;
while (q2 < end && *q2 != ':')
q2++;
if (*q2 == ':')
ml_end = q2;
if (ml_end - q == 1)
multilib_os_dir = xstrdup (".");
else
{
new_multilib_os_dir = XNEWVEC (char, ml_end - q);
memcpy (new_multilib_os_dir, q + 1, ml_end - q - 1);
new_multilib_os_dir[ml_end - q - 1] = '\0';
multilib_os_dir = new_multilib_os_dir;
}
if (q2 < end && *q2 == ':')
{
char *new_multiarch_dir = XNEWVEC (char, end - q2);
memcpy (new_multiarch_dir, q2 + 1, end - q2 - 1);
new_multiarch_dir[end - q2 - 1] = '\0';
multiarch_dir = new_multiarch_dir;
}
break;
}
}
++p;
}
if (multilib_dir == NULL && multilib_os_dir != NULL
&& strcmp (multilib_os_dir, ".") == 0)
{
free (CONST_CAST (char *, multilib_os_dir));
multilib_os_dir = NULL;
}
else if (multilib_dir != NULL && multilib_os_dir == NULL)
multilib_os_dir = multilib_dir;
}
static void
print_multilib_info (void)
{
const char *p = multilib_select;
const char *last_path = 0, *this_path;
int skip;
unsigned int last_path_len = 0;
while (*p != '\0')
{
skip = 0;
if (*p == '\n')
{
++p;
continue;
}
this_path = p;
while (*p != ' ')
{
if (*p == '\0')
{
invalid_select:
fatal_error (input_location,
"multilib select %qs is invalid", multilib_select);
}
++p;
}
if (this_path[0] == '.' && this_path[1] == ':' && this_path[2] != ':')
skip = 1;
{
const char *e = multilib_exclusions;
const char *this_arg;
while (*e != '\0')
{
int m = 1;
if (*e == '\n')
{
++e;
continue;
}
while (*e != ';')
{
const char *q;
int mp = 0;
if (*e == '\0')
{
invalid_exclusion:
fatal_error (input_location,
"multilib exclusion %qs is invalid",
multilib_exclusions);
}
if (! m)
{
++e;
continue;
}
this_arg = e;
while (*e != ' ' && *e != ';')
{
if (*e == '\0')
goto invalid_exclusion;
++e;
}
q = p + 1;
while (*q != ';')
{
const char *arg;
int len = e - this_arg;
if (*q == '\0')
goto invalid_select;
arg = q;
while (*q != ' ' && *q != ';')
{
if (*q == '\0')
goto invalid_select;
++q;
}
if (! strncmp (arg, this_arg,
(len < q - arg) ? q - arg : len)
|| default_arg (this_arg, e - this_arg))
{
mp = 1;
break;
}
if (*q == ' ')
++q;
}
if (! mp)
m = 0;
if (*e == ' ')
++e;
}
if (m)
{
skip = 1;
break;
}
if (*e != '\0')
++e;
}
}
if (! skip)
{
skip = (last_path != 0
&& (unsigned int) (p - this_path) == last_path_len
&& ! filename_ncmp (last_path, this_path, last_path_len));
last_path = this_path;
last_path_len = p - this_path;
}
if (! skip)
{
const char *q;
q = p + 1;
while (*q != ';')
{
const char *arg;
if (*q == '\0')
goto invalid_select;
if (*q == '!')
arg = NULL;
else
arg = q;
while (*q != ' ' && *q != ';')
{
if (*q == '\0')
goto invalid_select;
++q;
}
if (arg != NULL
&& default_arg (arg, q - arg))
{
skip = 1;
break;
}
if (*q == ' ')
++q;
}
}
if (! skip)
{
const char *p1;
for (p1 = last_path; p1 < p && *p1 != ':'; p1++)
putchar (*p1);
putchar (';');
}
++p;
while (*p != ';')
{
int use_arg;
if (*p == '\0')
goto invalid_select;
if (skip)
{
++p;
continue;
}
use_arg = *p != '!';
if (use_arg)
putchar ('@');
while (*p != ' ' && *p != ';')
{
if (*p == '\0')
goto invalid_select;
if (use_arg)
putchar (*p);
++p;
}
if (*p == ' ')
++p;
}
if (! skip)
{
if (multilib_extra && *multilib_extra)
{
int print_at = TRUE;
const char *q;
for (q = multilib_extra; *q != '\0'; q++)
{
if (*q == ' ')
print_at = TRUE;
else
{
if (print_at)
putchar ('@');
putchar (*q);
print_at = FALSE;
}
}
}
putchar ('\n');
}
++p;
}
}

static const char *
getenv_spec_function (int argc, const char **argv)
{
const char *value;
const char *varname;
char *result;
char *ptr;
size_t len;
if (argc != 2)
return NULL;
varname = argv[0];
value = env.get (varname);
if (!value && spec_undefvar_allowed)
value = varname;
if (!value)
fatal_error (input_location,
"environment variable %qs not defined", varname);
len = strlen (value) * 2 + strlen (argv[1]) + 1;
result = XNEWVAR (char, len);
for (ptr = result; *value; ptr += 2)
{
ptr[0] = '\\';
ptr[1] = *value++;
}
strcpy (ptr, argv[1]);
return result;
}
static const char *
if_exists_spec_function (int argc, const char **argv)
{
if (argc == 1 && IS_ABSOLUTE_PATH (argv[0]) && ! access (argv[0], R_OK))
return argv[0];
return NULL;
}
static const char *
if_exists_else_spec_function (int argc, const char **argv)
{
if (argc != 2)
return NULL;
if (IS_ABSOLUTE_PATH (argv[0]) && ! access (argv[0], R_OK))
return argv[0];
return argv[1];
}
static const char *
sanitize_spec_function (int argc, const char **argv)
{
if (argc != 1)
return NULL;
if (strcmp (argv[0], "address") == 0)
return (flag_sanitize & SANITIZE_USER_ADDRESS) ? "" : NULL;
if (strcmp (argv[0], "kernel-address") == 0)
return (flag_sanitize & SANITIZE_KERNEL_ADDRESS) ? "" : NULL;
if (strcmp (argv[0], "thread") == 0)
return (flag_sanitize & SANITIZE_THREAD) ? "" : NULL;
if (strcmp (argv[0], "undefined") == 0)
return ((flag_sanitize
& (SANITIZE_UNDEFINED | SANITIZE_UNDEFINED_NONDEFAULT))
&& !flag_sanitize_undefined_trap_on_error) ? "" : NULL;
if (strcmp (argv[0], "leak") == 0)
return ((flag_sanitize
& (SANITIZE_ADDRESS | SANITIZE_LEAK | SANITIZE_THREAD))
== SANITIZE_LEAK) ? "" : NULL;
return NULL;
}
static const char *
replace_outfile_spec_function (int argc, const char **argv)
{
int i;
if (argc != 2)
abort ();
for (i = 0; i < n_infiles; i++)
{
if (outfiles[i] && !filename_cmp (outfiles[i], argv[0]))
outfiles[i] = xstrdup (argv[1]);
}
return NULL;
}
static const char *
remove_outfile_spec_function (int argc, const char **argv)
{
int i;
if (argc != 1)
abort ();
for (i = 0; i < n_infiles; i++)
{
if (outfiles[i] && !filename_cmp (outfiles[i], argv[0]))
outfiles[i] = NULL;
}
return NULL;
}
static int
compare_version_strings (const char *v1, const char *v2)
{
int rresult;
regex_t r;
if (regcomp (&r, "^([1-9][0-9]*|0)(\\.([1-9][0-9]*|0))*$",
REG_EXTENDED | REG_NOSUB) != 0)
abort ();
rresult = regexec (&r, v1, 0, NULL, 0);
if (rresult == REG_NOMATCH)
fatal_error (input_location, "invalid version number %qs", v1);
else if (rresult != 0)
abort ();
rresult = regexec (&r, v2, 0, NULL, 0);
if (rresult == REG_NOMATCH)
fatal_error (input_location, "invalid version number %qs", v2);
else if (rresult != 0)
abort ();
return strverscmp (v1, v2);
}
static const char *
version_compare_spec_function (int argc, const char **argv)
{
int comp1, comp2;
size_t switch_len;
const char *switch_value = NULL;
int nargs = 1, i;
bool result;
if (argc < 3)
fatal_error (input_location, "too few arguments to %%:version-compare");
if (argv[0][0] == '\0')
abort ();
if ((argv[0][1] == '<' || argv[0][1] == '>') && argv[0][0] != '!')
nargs = 2;
if (argc != nargs + 3)
fatal_error (input_location, "too many arguments to %%:version-compare");
switch_len = strlen (argv[nargs + 1]);
for (i = 0; i < n_switches; i++)
if (!strncmp (switches[i].part1, argv[nargs + 1], switch_len)
&& check_live_switch (i, switch_len))
switch_value = switches[i].part1 + switch_len;
if (switch_value == NULL)
comp1 = comp2 = -1;
else
{
comp1 = compare_version_strings (switch_value, argv[1]);
if (nargs == 2)
comp2 = compare_version_strings (switch_value, argv[2]);
else
comp2 = -1;  
}
switch (argv[0][0] << 8 | argv[0][1])
{
case '>' << 8 | '=':
result = comp1 >= 0;
break;
case '!' << 8 | '<':
result = comp1 >= 0 || switch_value == NULL;
break;
case '<' << 8:
result = comp1 < 0;
break;
case '!' << 8 | '>':
result = comp1 < 0 || switch_value == NULL;
break;
case '>' << 8 | '<':
result = comp1 >= 0 && comp2 < 0;
break;
case '<' << 8 | '>':
result = comp1 < 0 || comp2 >= 0;
break;
default:
fatal_error (input_location,
"unknown operator %qs in %%:version-compare", argv[0]);
}
if (! result)
return NULL;
return argv[nargs + 2];
}
static const char *
include_spec_function (int argc, const char **argv)
{
char *file;
if (argc != 1)
abort ();
file = find_a_file (&startfile_prefixes, argv[0], R_OK, true);
read_specs (file ? file : argv[0], false, false);
return NULL;
}
static const char *
find_file_spec_function (int argc, const char **argv)
{
const char *file;
if (argc != 1)
abort ();
file = find_file (argv[0]);
return file;
}
static const char *
find_plugindir_spec_function (int argc, const char **argv ATTRIBUTE_UNUSED)
{
const char *option;
if (argc != 0)
abort ();
option = concat ("-iplugindir=", find_file ("plugin"), NULL);
return option;
}
static const char *
print_asm_header_spec_function (int arg ATTRIBUTE_UNUSED,
const char **argv ATTRIBUTE_UNUSED)
{
printf (_("Assembler options\n=================\n\n"));
printf (_("Use \"-Wa,OPTION\" to pass \"OPTION\" to the assembler.\n\n"));
fflush (stdout);
return NULL;
}
static unsigned HOST_WIDE_INT
get_random_number (void)
{
unsigned HOST_WIDE_INT ret = 0;
int fd; 
fd = open ("/dev/urandom", O_RDONLY); 
if (fd >= 0)
{
read (fd, &ret, sizeof (HOST_WIDE_INT));
close (fd);
if (ret)
return ret;
}
#ifdef HAVE_GETTIMEOFDAY
{
struct timeval tv;
gettimeofday (&tv, NULL);
ret = tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#else
{
time_t now = time (NULL);
if (now != (time_t)-1)
ret = (unsigned) now;
}
#endif
return ret ^ getpid ();
}
static const char *
compare_debug_dump_opt_spec_function (int arg,
const char **argv ATTRIBUTE_UNUSED)
{
char *ret;
char *name;
int which;
static char random_seed[HOST_BITS_PER_WIDE_INT / 4 + 3];
if (arg != 0)
fatal_error (input_location,
"too many arguments to %%:compare-debug-dump-opt");
do_spec_2 ("%{fdump-final-insns=*:%*}");
do_spec_1 (" ", 0, NULL);
if (argbuf.length () > 0
&& strcmp (argv[argbuf.length () - 1], "."))
{
if (!compare_debug)
return NULL;
name = xstrdup (argv[argbuf.length () - 1]);
ret = NULL;
}
else
{
const char *ext = NULL;
if (argbuf.length () > 0)
{
do_spec_2 ("%{o*:%*}%{!o:%{!S:%b%O}%{S:%b.s}}");
ext = ".gkd";
}
else if (!compare_debug)
return NULL;
else
do_spec_2 ("%g.gkd");
do_spec_1 (" ", 0, NULL);
gcc_assert (argbuf.length () > 0);
name = concat (argbuf.last (), ext, NULL);
ret = concat ("-fdump-final-insns=", name, NULL);
}
which = compare_debug < 0;
debug_check_temp_file[which] = name;
if (!which)
{
unsigned HOST_WIDE_INT value = get_random_number ();
sprintf (random_seed, HOST_WIDE_INT_PRINT_HEX, value);
}
if (*random_seed)
{
char *tmp = ret;
ret = concat ("%{!frandom-seed=*:-frandom-seed=", random_seed, "} ",
ret, NULL);
free (tmp);
}
if (which)
*random_seed = 0;
return ret;
}
static const char *debug_auxbase_opt;
static const char *
compare_debug_self_opt_spec_function (int arg,
const char **argv ATTRIBUTE_UNUSED)
{
if (arg != 0)
fatal_error (input_location,
"too many arguments to %%:compare-debug-self-opt");
if (compare_debug >= 0)
return NULL;
do_spec_2 ("%{c|S:%{o*:%*}}");
do_spec_1 (" ", 0, NULL);
if (argbuf.length () > 0)
debug_auxbase_opt = concat ("-auxbase-strip ",
argbuf.last (),
NULL);
else
debug_auxbase_opt = NULL;
return concat ("\
%<o %<MD %<MMD %<MF* %<MG %<MP %<MQ* %<MT* \
%<fdump-final-insns=* -w -S -o %j \
%{!fcompare-debug-second:-fcompare-debug-second} \
", compare_debug_opt, NULL);
}
static const char *
compare_debug_auxbase_opt_spec_function (int arg,
const char **argv)
{
char *name;
int len;
if (arg == 0)
fatal_error (input_location,
"too few arguments to %%:compare-debug-auxbase-opt");
if (arg != 1)
fatal_error (input_location,
"too many arguments to %%:compare-debug-auxbase-opt");
if (compare_debug >= 0)
return NULL;
len = strlen (argv[0]);
if (len < 3 || strcmp (argv[0] + len - 3, ".gk") != 0)
fatal_error (input_location, "argument to %%:compare-debug-auxbase-opt "
"does not end in .gk");
if (debug_auxbase_opt)
return debug_auxbase_opt;
#define OPT "-auxbase "
len -= 3;
name = (char*) xmalloc (sizeof (OPT) + len);
memcpy (name, OPT, sizeof (OPT) - 1);
memcpy (name + sizeof (OPT) - 1, argv[0], len);
name[sizeof (OPT) - 1 + len] = '\0';
#undef OPT
return name;
}
const char *
pass_through_libs_spec_func (int argc, const char **argv)
{
char *prepended = xstrdup (" ");
int n;
for (n = 0; n < argc; n++)
{
char *old = prepended;
if (argv[n][0] == '-' && argv[n][1] == 'l')
{
const char *lopt = argv[n] + 2;
if (!*lopt && ++n >= argc)
break;
else if (!*lopt)
lopt = argv[n];
prepended = concat (prepended, "-plugin-opt=-pass-through=-l",
lopt, " ", NULL);
}
else if (!strcmp (".a", argv[n] + strlen (argv[n]) - 2))
{
prepended = concat (prepended, "-plugin-opt=-pass-through=",
argv[n], " ", NULL);
}
if (prepended != old)
free (old);
}
return prepended;
}
const char *
replace_extension_spec_func (int argc, const char **argv)
{
char *name;
char *p;
char *result;
int i;
if (argc != 2)
fatal_error (input_location, "too few arguments to %%:replace-extension");
name = xstrdup (argv[0]);
for (i = strlen (name) - 1; i >= 0; i--)
if (IS_DIR_SEPARATOR (name[i]))
break;
p = strrchr (name + i + 1, '.');
if (p != NULL)
*p = '\0';
result = concat (name, argv[1], NULL);
free (name);
return result;
}
static const char *
greater_than_spec_func (int argc, const char **argv)
{
char *converted;
if (argc == 1)
return NULL;
gcc_assert (argc >= 2);
long arg = strtol (argv[argc - 2], &converted, 10);
gcc_assert (converted != argv[argc - 2]);
long lim = strtol (argv[argc - 1], &converted, 10);
gcc_assert (converted != argv[argc - 1]);
if (arg > lim)
return "";
return NULL;
}
static const char *
debug_level_greater_than_spec_func (int argc, const char **argv)
{
char *converted;
if (argc != 1)
fatal_error (input_location,
"wrong number of arguments to %%:debug-level-gt");
long arg = strtol (argv[0], &converted, 10);
gcc_assert (converted != argv[0]);
if (debug_info_level > arg)
return "";
return NULL;
}
static char *
convert_white_space (char *orig)
{
int len, number_of_space = 0;
for (len = 0; orig[len]; len++)
if (orig[len] == ' ' || orig[len] == '\t') number_of_space++;
if (number_of_space)
{
char *new_spec = (char *) xmalloc (len + number_of_space + 1);
int j, k;
for (j = 0, k = 0; j <= len; j++, k++)
{
if (orig[j] == ' ' || orig[j] == '\t')
new_spec[k++] = '\\';
new_spec[k] = orig[j];
}
free (orig);
return new_spec;
}
else
return orig;
}
static void
path_prefix_reset (path_prefix *prefix)
{
struct prefix_list *iter, *next;
iter = prefix->plist;
while (iter)
{
next = iter->next;
free (const_cast <char *> (iter->prefix));
XDELETE (iter);
iter = next;
}
prefix->plist = 0;
prefix->max_len = 0;
}
void
driver::finalize ()
{
env.restore ();
params_c_finalize ();
diagnostic_finish (global_dc);
is_cpp_driver = 0;
at_file_supplied = 0;
print_help_list = 0;
print_version = 0;
verbose_only_flag = 0;
print_subprocess_help = 0;
use_ld = NULL;
report_times_to_file = NULL;
target_system_root = DEFAULT_TARGET_SYSTEM_ROOT;
target_system_root_changed = 0;
target_sysroot_suffix = 0;
target_sysroot_hdrs_suffix = 0;
save_temps_flag = SAVE_TEMPS_NONE;
save_temps_prefix = 0;
save_temps_length = 0;
spec_machine = DEFAULT_TARGET_MACHINE;
greatest_status = 1;
finalize_options_struct (&global_options);
finalize_options_struct (&global_options_set);
obstack_free (&obstack, NULL);
obstack_free (&opts_obstack, NULL); 
obstack_free (&collect_obstack, NULL);
link_command_spec = LINK_COMMAND_SPEC;
obstack_free (&multilib_obstack, NULL);
user_specs_head = NULL;
user_specs_tail = NULL;
for (int i = n_default_compilers; i < n_compilers; i++)
{
free (const_cast <char *> (compilers[i].suffix));
free (const_cast <char *> (compilers[i].spec));
}
XDELETEVEC (compilers);
compilers = NULL;
n_compilers = 0;
linker_options.truncate (0);
assembler_options.truncate (0);
preprocessor_options.truncate (0);
path_prefix_reset (&exec_prefixes);
path_prefix_reset (&startfile_prefixes);
path_prefix_reset (&include_prefixes);
machine_suffix = 0;
just_machine_suffix = 0;
gcc_exec_prefix = 0;
gcc_libexec_prefix = 0;
md_exec_prefix = MD_EXEC_PREFIX;
md_startfile_prefix = MD_STARTFILE_PREFIX;
md_startfile_prefix_1 = MD_STARTFILE_PREFIX_1;
multilib_dir = 0;
multilib_os_dir = 0;
multiarch_dir = 0;
if (specs)
{
while (specs != static_specs)
{
spec_list *next = specs->next;
free (const_cast <char *> (specs->name));
XDELETE (specs);
specs = next;
}
specs = 0;
}
for (unsigned i = 0; i < ARRAY_SIZE (static_specs); i++)
{
spec_list *sl = &static_specs[i];
if (sl->alloc_p)
{
if (0)
free (const_cast <char *> (*(sl->ptr_spec)));
sl->alloc_p = false;
}
*(sl->ptr_spec) = sl->default_ptr;
}
#ifdef EXTRA_SPECS
extra_specs = NULL;
#endif
processing_spec_function = 0;
argbuf.truncate (0);
have_c = 0;
have_o = 0;
temp_names = NULL;
execution_count = 0;
signal_count = 0;
temp_filename = NULL;
temp_filename_length = 0;
always_delete_queue = NULL;
failure_delete_queue = NULL;
XDELETEVEC (switches);
switches = NULL;
n_switches = 0;
n_switches_alloc = 0;
compare_debug = 0;
compare_debug_second = 0;
compare_debug_opt = NULL;
for (int i = 0; i < 2; i++)
{
switches_debug_check[i] = NULL;
n_switches_debug_check[i] = 0;
n_switches_alloc_debug_check[i] = 0;
debug_check_temp_file[i] = NULL;
}
XDELETEVEC (infiles);
infiles = NULL;
n_infiles = 0;
n_infiles_alloc = 0;
combine_inputs = false;
added_libraries = 0;
XDELETEVEC (outfiles);
outfiles = NULL;
spec_lang = 0;
last_language_n_infiles = 0;
gcc_input_filename = NULL;
input_file_number = 0;
input_filename_length = 0;
basename_length = 0;
suffixed_basename_length = 0;
input_basename = NULL;
input_suffix = NULL;
input_stat_set = 0;
input_file_compiler = NULL;
arg_going = 0;
delete_this_arg = 0;
this_is_output_file = 0;
this_is_library_file = 0;
this_is_linker_script = 0;
input_from_pipe = 0;
suffix_subst = NULL;
mdswitches = NULL;
n_mdswitches = 0;
debug_auxbase_opt = NULL;
used_arg.finalize ();
}
void
driver_get_configure_time_options (void (*cb) (const char *option,
void *user_data),
void *user_data)
{
size_t i;
obstack_init (&obstack);
init_opts_obstack ();
n_switches = 0;
for (i = 0; i < ARRAY_SIZE (option_default_specs); i++)
do_option_spec (option_default_specs[i].name,
option_default_specs[i].spec);
for (i = 0; (int) i < n_switches; i++)
{
gcc_assert (switches[i].part1);
(*cb) (switches[i].part1, user_data);
}
obstack_free (&opts_obstack, NULL);
obstack_free (&obstack, NULL);
n_switches = 0;
}
