#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-family/c-target.h"
#include "c-family/c-target-def.h"
#include "memmodel.h"
#include "tm_p.h"
#include "cgraph.h"
#include "incpath.h"
#include "c-family/c-pragma.h"
#include "c-family/c-format.h"
#include "cppdefault.h"
#include "prefix.h"
#include "../../libcpp/internal.h"
#define BAD(gmsgid) do { warning (OPT_Wpragmas, gmsgid); return; } while (0)
#define BAD2(msgid, arg) do { warning (OPT_Wpragmas, msgid, arg); return; } while (0)
static bool using_frameworks = false;
static const char *find_subframework_header (cpp_reader *pfile, const char *header,
cpp_dir **dirp);
typedef struct align_stack
{
int alignment;
struct align_stack * prev;
} align_stack;
static struct align_stack * field_align_stack = NULL;
static void
push_field_alignment (int bit_alignment)
{
align_stack *entry = XNEW (align_stack);
entry->alignment = maximum_field_alignment;
entry->prev = field_align_stack;
field_align_stack = entry;
maximum_field_alignment = bit_alignment;
}
static void
pop_field_alignment (void)
{
if (field_align_stack)
{
align_stack *entry = field_align_stack;
maximum_field_alignment = entry->alignment;
field_align_stack = entry->prev;
free (entry);
}
else
error ("too many #pragma options align=reset");
}
void
darwin_pragma_ignore (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
}
void
darwin_pragma_options (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
const char *arg;
tree t, x;
if (pragma_lex (&t) != CPP_NAME)
BAD ("malformed '#pragma options', ignoring");
arg = IDENTIFIER_POINTER (t);
if (strcmp (arg, "align"))
BAD ("malformed '#pragma options', ignoring");
if (pragma_lex (&t) != CPP_EQ)
BAD ("malformed '#pragma options', ignoring");
if (pragma_lex (&t) != CPP_NAME)
BAD ("malformed '#pragma options', ignoring");
if (pragma_lex (&x) != CPP_EOF)
warning (OPT_Wpragmas, "junk at end of '#pragma options'");
arg = IDENTIFIER_POINTER (t);
if (!strcmp (arg, "mac68k"))
push_field_alignment (16);
else if (!strcmp (arg, "power"))
push_field_alignment (0);
else if (!strcmp (arg, "reset"))
pop_field_alignment ();
else
BAD ("malformed '#pragma options align={mac68k|power|reset}', ignoring");
}
void
darwin_pragma_unused (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
tree decl, x;
int tok;
if (pragma_lex (&x) != CPP_OPEN_PAREN)
BAD ("missing '(' after '#pragma unused', ignoring");
while (1)
{
tok = pragma_lex (&decl);
if (tok == CPP_NAME && decl)
{
tree local = lookup_name (decl);
if (local && (TREE_CODE (local) == PARM_DECL
|| TREE_CODE (local) == VAR_DECL))
{
TREE_USED (local) = 1;
DECL_READ_P (local) = 1;
}
tok = pragma_lex (&x);
if (tok != CPP_COMMA)
break;
}
}
if (tok != CPP_CLOSE_PAREN)
BAD ("missing ')' after '#pragma unused', ignoring");
if (pragma_lex (&x) != CPP_EOF)
BAD ("junk at end of '#pragma unused'");
}
void
darwin_pragma_ms_struct (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
const char *arg;
tree t;
if (pragma_lex (&t) != CPP_NAME)
BAD ("malformed '#pragma ms_struct', ignoring");
arg = IDENTIFIER_POINTER (t);
if (!strcmp (arg, "on"))
darwin_ms_struct = true;
else if (!strcmp (arg, "off") || !strcmp (arg, "reset"))
darwin_ms_struct = false;
else
BAD ("malformed '#pragma ms_struct {on|off|reset}', ignoring");
if (pragma_lex (&t) != CPP_EOF)
BAD ("junk at end of '#pragma ms_struct'");
}
static struct frameworks_in_use {
size_t len;
const char *name;
cpp_dir* dir;
} *frameworks_in_use;
static int num_frameworks = 0;
static int max_frameworks = 0;
static void
add_framework (const char *name, size_t len, cpp_dir *dir)
{
char *dir_name;
int i;
for (i = 0; i < num_frameworks; ++i)
{
if (len == frameworks_in_use[i].len
&& strncmp (name, frameworks_in_use[i].name, len) == 0)
{
return;
}
}
if (i >= max_frameworks)
{
max_frameworks = i*2;
max_frameworks += i == 0;
frameworks_in_use = XRESIZEVEC (struct frameworks_in_use,
frameworks_in_use, max_frameworks);
}
dir_name = XNEWVEC (char, len + 1);
memcpy (dir_name, name, len);
dir_name[len] = '\0';
frameworks_in_use[num_frameworks].name = dir_name;
frameworks_in_use[num_frameworks].len = len;
frameworks_in_use[num_frameworks].dir = dir;
++num_frameworks;
}
static struct cpp_dir*
find_framework (const char *name, size_t len)
{
int i;
for (i = 0; i < num_frameworks; ++i)
{
if (len == frameworks_in_use[i].len
&& strncmp (name, frameworks_in_use[i].name, len) == 0)
{
return frameworks_in_use[i].dir;
}
}
return 0;
}
struct framework_header {const char * dirName; int dirNameLen; };
static struct framework_header framework_header_dirs[] = {
{ "Headers", 7 },
{ "PrivateHeaders", 14 },
{ NULL, 0 }
};
static char *
framework_construct_pathname (const char *fname, cpp_dir *dir)
{
const char *buf;
size_t fname_len, frname_len;
cpp_dir *fast_dir;
char *frname;
struct stat st;
int i;
buf = strchr (fname, '/');
if (buf)
fname_len = buf - fname;
else
return 0;
fast_dir = find_framework (fname, fname_len);
if (fast_dir && dir != fast_dir)
return 0;
frname = XNEWVEC (char, strlen (fname) + dir->len + 2
+ strlen(".framework/") + strlen("PrivateHeaders"));
memcpy (&frname[0], dir->name, dir->len);
frname_len = dir->len;
if (frname_len && frname[frname_len-1] != '/')
frname[frname_len++] = '/';
memcpy (&frname[frname_len], fname, fname_len);
frname_len += fname_len;
memcpy (&frname[frname_len], ".framework/", strlen (".framework/"));
frname_len += strlen (".framework/");
if (fast_dir == 0)
{
frname[frname_len-1] = 0;
if (stat (frname, &st) == 0)
{
add_framework (fname, fname_len, dir);
}
else
{
free (frname);
return 0;
}
frname[frname_len-1] = '/';
}
for (i = 0; framework_header_dirs[i].dirName; i++)
{
memcpy (&frname[frname_len],
framework_header_dirs[i].dirName,
framework_header_dirs[i].dirNameLen);
strcpy (&frname[frname_len + framework_header_dirs[i].dirNameLen],
&fname[fname_len]);
if (stat (frname, &st) == 0)
return frname;
}
free (frname);
return 0;
}
static const char*
find_subframework_file (const char *fname, const char *pname)
{
char *sfrname;
const char *dot_framework = ".framework/";
const char *bufptr;
int sfrname_len, i, fname_len;
struct cpp_dir *fast_dir;
static struct cpp_dir subframe_dir;
struct stat st;
bufptr = strchr (fname, '/');
if (bufptr == 0)
return 0;
fname_len = bufptr - fname;
fast_dir = find_framework (fname, fname_len);
bufptr = strstr (pname, dot_framework);
if (!bufptr)
return 0;
sfrname = XNEWVEC (char, strlen (pname) + strlen (fname) + 2 +
strlen ("Frameworks/") + strlen (".framework/")
+ strlen ("PrivateHeaders"));
bufptr += strlen (dot_framework);
sfrname_len = bufptr - pname;
memcpy (&sfrname[0], pname, sfrname_len);
memcpy (&sfrname[sfrname_len], "Frameworks/", strlen ("Frameworks/"));
sfrname_len += strlen("Frameworks/");
memcpy (&sfrname[sfrname_len], fname, fname_len);
sfrname_len += fname_len;
memcpy (&sfrname[sfrname_len], ".framework/", strlen (".framework/"));
sfrname_len += strlen (".framework/");
for (i = 0; framework_header_dirs[i].dirName; i++)
{
memcpy (&sfrname[sfrname_len],
framework_header_dirs[i].dirName,
framework_header_dirs[i].dirNameLen);
strcpy (&sfrname[sfrname_len + framework_header_dirs[i].dirNameLen],
&fname[fname_len]);
if (stat (sfrname, &st) == 0)
{
if (fast_dir != &subframe_dir)
{
if (fast_dir)
warning (0, "subframework include %s conflicts with framework include",
fname);
else
add_framework (fname, fname_len, &subframe_dir);
}
return sfrname;
}
}
free (sfrname);
return 0;
}
static void
add_system_framework_path (char *path)
{
int cxx_aware = 1;
cpp_dir *p;
p = XNEW (cpp_dir);
p->next = NULL;
p->name = path;
p->sysp = 1 + !cxx_aware;
p->construct = framework_construct_pathname;
using_frameworks = 1;
add_cpp_dir_path (p, INC_SYSTEM);
}
void
add_framework_path (char *path)
{
cpp_dir *p;
p = XNEW (cpp_dir);
p->next = NULL;
p->name = path;
p->sysp = 0;
p->construct = framework_construct_pathname;
using_frameworks = 1;
add_cpp_dir_path (p, INC_BRACKET);
}
static const char *framework_defaults [] =
{
"/System/Library/Frameworks",
"/Library/Frameworks",
};
void
darwin_register_objc_includes (const char *sysroot, const char *iprefix,
int stdinc)
{
const char *fname;
size_t len;
if (!stdinc)
return;
fname = GCC_INCLUDE_DIR "-gnu-runtime";
if (c_dialect_objc () && !flag_next_runtime)
{
char *str;
if (iprefix && (len = cpp_GCC_INCLUDE_DIR_len) != 0 && !sysroot
&& !strncmp (fname, cpp_GCC_INCLUDE_DIR, len))
{
str = concat (iprefix, fname + len, NULL);
add_path (str, INC_SYSTEM, false, false);
}
if (sysroot)
str = concat (sysroot, fname, NULL);
else
str = update_path (fname, "");
add_path (str, INC_SYSTEM, false, false);
}
}
void
darwin_register_frameworks (const char *sysroot,
const char *iprefix ATTRIBUTE_UNUSED, int stdinc)
{
if (stdinc)
{
size_t i;
for (i=0; i<sizeof (framework_defaults)/sizeof(const char *); ++i)
{
char *str;
if (sysroot)
str = concat (sysroot, xstrdup (framework_defaults [i]), NULL);
else
str = xstrdup (framework_defaults[i]);
add_system_framework_path (str);
}
}
if (using_frameworks)
cpp_get_callbacks (parse_in)->missing_header = find_subframework_header;
}
static const char*
find_subframework_header (cpp_reader *pfile, const char *header, cpp_dir **dirp)
{
const char *fname = header;
struct cpp_buffer *b;
const char *n;
for (b = cpp_get_buffer (pfile);
b && cpp_get_file (b) && cpp_get_path (cpp_get_file (b));
b = cpp_get_prev (b))
{
n = find_subframework_file (fname, cpp_get_path (cpp_get_file (b)));
if (n)
{
*dirp = cpp_get_dir (cpp_get_file (b));
return n;
}
}
return 0;
}
enum version_components { MAJOR, MINOR, TINY };
static const unsigned long *
parse_version (const char *version_str)
{
size_t version_len;
char *end;
static unsigned long version_array[3];
version_len = strlen (version_str);
if (version_len < 1)
return NULL;
if (strspn (version_str, "0123456789.") != version_len)
return NULL;
if (!ISDIGIT (version_str[0]) || !ISDIGIT (version_str[version_len - 1]))
return NULL;
version_array[MAJOR] = strtoul (version_str, &end, 10);
version_str = end + ((*end == '.') ? 1 : 0);
if (*version_str == '.')
return NULL;
version_array[MINOR] = strtoul (version_str, &end, 10);
version_str = end + ((*end == '.') ? 1 : 0);
version_array[TINY] = strtoul (version_str, &end, 10);
if (*end != '\0')
return NULL;
return version_array;
}
static const char *
version_as_legacy_macro (const unsigned long *version)
{
unsigned long major, minor, tiny;
static char result[5];
major = version[MAJOR];
minor = version[MINOR];
tiny = version[TINY];
if (major > 99 || minor > 99 || tiny > 99)
return NULL;
minor = ((minor > 9) ? 9 : minor);
tiny = ((tiny > 9) ? 9 : tiny);
if (sprintf (result, "%lu%lu%lu", major, minor, tiny) != 4)
return NULL;
return result;
}
static const char *
version_as_modern_macro (const unsigned long *version)
{
unsigned long major, minor, tiny;
static char result[7];
major = version[MAJOR];
minor = version[MINOR];
tiny = version[TINY];
if (major > 99 || minor > 99 || tiny > 99)
return NULL;
if (sprintf (result, "%02lu%02lu%02lu", major, minor, tiny) != 6)
return NULL;
return result;
}
static const char *
macosx_version_as_macro (void)
{
const unsigned long *version_array;
const char *version_macro;
version_array = parse_version (darwin_macosx_version_min);
if (!version_array)
goto fail;
if (version_array[MAJOR] != 10)
goto fail;
if (version_array[MINOR] < 10)
version_macro = version_as_legacy_macro (version_array);
else
version_macro = version_as_modern_macro (version_array);
if (!version_macro)
goto fail;
return version_macro;
fail:
error ("unknown value %qs of -mmacosx-version-min",
darwin_macosx_version_min);
return "1000";
}
#define builtin_define(TXT) cpp_define (pfile, TXT)
void
darwin_cpp_builtins (cpp_reader *pfile)
{
builtin_define ("__MACH__");
builtin_define ("__APPLE__");
builtin_define_with_value ("__APPLE_CC__", "1", false);
if (darwin_constant_cfstrings)
builtin_define ("__CONSTANT_CFSTRINGS__");
builtin_define_with_value ("__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__",
macosx_version_as_macro(), false);
if (flag_objc_gc)
{
builtin_define ("__strong=__attribute__((objc_gc(strong)))");
builtin_define ("__weak=__attribute__((objc_gc(weak)))");
builtin_define ("__OBJC_GC__");
}
else
{
builtin_define ("__strong=");
builtin_define ("__weak=");
}
if (CPP_OPTION (pfile, objc) && flag_objc_abi == 2)
builtin_define ("__OBJC2__");
}
static bool
handle_c_option (size_t code,
const char *arg,
int value ATTRIBUTE_UNUSED)
{
switch (code)
{
default:
return false;
case OPT_iframework:
add_system_framework_path (xstrdup (arg));
break;
case OPT_fapple_kext:
;
}
return true;
}
static tree
darwin_objc_construct_string (tree str)
{
if (!darwin_constant_cfstrings)
{
darwin_enter_string_into_cfstring_table (str);
return NULL_TREE;
}
return darwin_build_constant_cfstring (str);
}
static bool
darwin_cfstring_ref_p (const_tree strp)
{
tree tn;
if (!strp || TREE_CODE (strp) != POINTER_TYPE)
return false;
tn = TYPE_NAME (strp);
if (tn) 
tn = DECL_NAME (tn);
return (tn 
&& IDENTIFIER_POINTER (tn)
&& !strncmp (IDENTIFIER_POINTER (tn), "CFStringRef", 8));
}
static void
darwin_check_cfstring_format_arg (tree ARG_UNUSED (format_arg), 
tree ARG_UNUSED (args_list))
{
}
EXPORTED_CONST format_kind_info darwin_additional_format_types[] = {
{ "CFString",   NULL,  NULL, NULL, NULL, 
NULL, NULL, 
FMT_FLAG_ARG_CONVERT|FMT_FLAG_PARSE_ARG_CONVERT_EXTERNAL, 0, 0, 0, 0, 0, 0,
NULL, NULL
}
};
static void
darwin_objc_declare_unresolved_class_reference (const char *name)
{
const char *lazy_reference = ".lazy_reference\t";
const char *hard_reference = ".reference\t";
const char *reference = MACHOPIC_INDIRECT ? lazy_reference : hard_reference;
size_t len = strlen (reference) + strlen(name) + 2;
char *buf = (char *) alloca (len);
gcc_checking_assert (!strncmp (name, ".objc_class_name_", 17));
snprintf (buf, len, "%s%s", reference, name);
symtab->finalize_toplevel_asm (build_string (strlen (buf), buf));
}
static void
darwin_objc_declare_class_definition (const char *name)
{
const char *xname = targetm.strip_name_encoding (name);
size_t len = strlen (xname) + 7 + 5;
char *buf = (char *) alloca (len);
gcc_checking_assert (!strncmp (name, ".objc_class_name_", 17)
|| !strncmp (name, "*.objc_category_name_", 21));
snprintf (buf, len, ".globl\t%s", xname);
symtab->finalize_toplevel_asm (build_string (strlen (buf), buf));
snprintf (buf, len, "%s = 0", xname);
symtab->finalize_toplevel_asm (build_string (strlen (buf), buf));
}
#undef  TARGET_HANDLE_C_OPTION
#define TARGET_HANDLE_C_OPTION handle_c_option
#undef  TARGET_OBJC_CONSTRUCT_STRING_OBJECT
#define TARGET_OBJC_CONSTRUCT_STRING_OBJECT darwin_objc_construct_string
#undef  TARGET_OBJC_DECLARE_UNRESOLVED_CLASS_REFERENCE
#define TARGET_OBJC_DECLARE_UNRESOLVED_CLASS_REFERENCE \
darwin_objc_declare_unresolved_class_reference
#undef  TARGET_OBJC_DECLARE_CLASS_DEFINITION
#define TARGET_OBJC_DECLARE_CLASS_DEFINITION \
darwin_objc_declare_class_definition
#undef  TARGET_STRING_OBJECT_REF_TYPE_P
#define TARGET_STRING_OBJECT_REF_TYPE_P darwin_cfstring_ref_p
#undef TARGET_CHECK_STRING_OBJECT_FORMAT_ARG
#define TARGET_CHECK_STRING_OBJECT_FORMAT_ARG darwin_check_cfstring_format_arg
struct gcc_targetcm targetcm = TARGETCM_INITIALIZER;
