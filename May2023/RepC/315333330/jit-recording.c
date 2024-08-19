#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "pretty-print.h"
#include "toplev.h"
#include <pthread.h>
#include "jit-builtins.h"
#include "jit-recording.h"
#include "jit-playback.h"
namespace gcc {
namespace jit {
dump::dump (recording::context &ctxt,
const char *filename,
bool update_locations)
: m_ctxt (ctxt),
m_filename (filename),
m_update_locations (update_locations),
m_line (0),
m_column (0)
{
m_file = fopen (filename, "w");
if (!m_file)
ctxt.add_error (NULL,
"error opening dump file %s for writing: %s",
filename,
xstrerror (errno));
}
dump::~dump ()
{
if (m_file)
{
int err = fclose (m_file);
if (err)
m_ctxt.add_error (NULL,
"error closing dump file %s: %s",
m_filename,
xstrerror (errno));
}
}
void
dump::write (const char *fmt, ...)
{
int len;
va_list ap;
char *buf;
if (!m_file)
return;
va_start (ap, fmt);
len = vasprintf (&buf, fmt, ap);
va_end (ap);
if (buf == NULL || len < 0)
{
m_ctxt.add_error (NULL, "malloc failure writing to dumpfile %s",
m_filename);
return;
}
if (fwrite (buf, strlen (buf), 1, m_file) != 1)
m_ctxt.add_error (NULL, "error writing to dump file %s",
m_filename);
fflush (m_file);
for (const char *ptr = buf; *ptr; ptr++)
{
if ('\n' == *ptr)
{
m_line++;
m_column = 0;
}
else
m_column++;
}
free (buf);
}
recording::location *
dump::make_location () const
{
return m_ctxt.new_location (m_filename, m_line, m_column,
false);
}
class allocator
{
public:
~allocator ();
char *
xstrdup_printf (const char *, ...)
ATTRIBUTE_RETURNS_NONNULL
GNU_PRINTF(2, 3);
char *
xstrdup_printf_va (const char *, va_list ap)
ATTRIBUTE_RETURNS_NONNULL
GNU_PRINTF(2, 0);
private:
auto_vec <void *> m_buffers;
};
allocator::~allocator ()
{
unsigned i;
void *buffer;
FOR_EACH_VEC_ELT (m_buffers, i, buffer)
free (buffer);
}
char *
allocator::xstrdup_printf (const char *fmt, ...)
{
char *result;
va_list ap;
va_start (ap, fmt);
result = xstrdup_printf_va (fmt, ap);
va_end (ap);
return result;
}
char *
allocator::xstrdup_printf_va (const char *fmt, va_list ap)
{
char *result = xvasprintf (fmt, ap);
m_buffers.safe_push (result);
return result;
}
class reproducer : public dump
{
public:
reproducer (recording::context &ctxt,
const char *filename);
void
write_params (const vec <recording::context *> &contexts);
void
write_args (const vec <recording::context *> &contexts);
const char *
make_identifier (recording::memento *m, const char *prefix);
const char *
make_tmp_identifier (const char *prefix, recording::memento *m);
const char *
get_identifier (recording::context *ctxt);
const char *
get_identifier (recording::memento *m);
const char *
get_identifier_as_rvalue (recording::rvalue *m);
const char *
get_identifier_as_lvalue (recording::lvalue *m);
const char *
get_identifier_as_type (recording::type *m);
char *
xstrdup_printf (const char *, ...)
ATTRIBUTE_RETURNS_NONNULL
GNU_PRINTF(2, 3);
private:
const char * ensure_identifier_is_unique (const char *candidate, void *ptr);
private:
hash_map<recording::memento *, const char *> m_map_memento_to_identifier;
struct hash_traits : public string_hash
{
static void remove (const char *) {}
};
hash_set<const char *, hash_traits> m_set_identifiers;
allocator m_allocator;
};
reproducer::reproducer (recording::context &ctxt,
const char *filename) :
dump (ctxt, filename, 0),
m_map_memento_to_identifier (),
m_set_identifiers (),
m_allocator ()
{
}
void
reproducer::write_params (const vec <recording::context *> &contexts)
{
unsigned i;
recording::context *ctxt;
FOR_EACH_VEC_ELT (contexts, i, ctxt)
{
write ("gcc_jit_context *%s",
get_identifier (ctxt));
if (i < contexts.length () - 1)
write (",\n"
"             ");
}
}
void
reproducer::write_args (const vec <recording::context *> &contexts)
{
unsigned i;
recording::context *ctxt;
FOR_EACH_VEC_ELT (contexts, i, ctxt)
{
write ("%s",
get_identifier (ctxt));
if (i < contexts.length () - 1)
write (",\n"
"               ");
}
}
static void
convert_to_identifier (char *str)
{
for (char *p = str; *p; p++)
if (!ISALNUM (*p))
*p = '_';
}
const char *
reproducer::ensure_identifier_is_unique (const char *candidate, void *ptr)
{
if (m_set_identifiers.contains (candidate))
candidate = m_allocator.xstrdup_printf ("%s_%p", candidate, ptr);
gcc_assert (!m_set_identifiers.contains (candidate));
m_set_identifiers.add (candidate);
return candidate;
}
const char *
reproducer::make_identifier (recording::memento *m, const char *prefix)
{
const char *result;
if (strlen (m->get_debug_string ()) < 100)
{
char *buf = m_allocator.xstrdup_printf ("%s_%s",
prefix,
m->get_debug_string ());
convert_to_identifier (buf);
result = buf;
}
else
result = m_allocator.xstrdup_printf ("%s_%p",
prefix, (void *) m);
result = ensure_identifier_is_unique (result, m);
m_map_memento_to_identifier.put (m, result);
return result;
}
const char *
reproducer::make_tmp_identifier (const char *prefix, recording::memento *m)
{
return m_allocator.xstrdup_printf ("%s_%s",
prefix, get_identifier (m));
}
const char *
reproducer::get_identifier (recording::context *ctxt)
{
return m_allocator.xstrdup_printf ("ctxt_%p",
(void *)ctxt);
}
const char *
reproducer::get_identifier (recording::memento *m)
{
if (!m)
return "NULL";
if (recording::location *loc = m->dyn_cast_location ())
if (!loc->created_by_user ())
return "NULL";
const char **slot = m_map_memento_to_identifier.get (m);
if (!slot)
{
get_context ().add_error (NULL,
"unable to find identifier for %p: %s",
(void *)m,
m->get_debug_string ());
gcc_unreachable ();
}
return *slot;
}
const char *
reproducer::get_identifier_as_rvalue (recording::rvalue *m)
{
return m->access_as_rvalue (*this);
}
const char *
reproducer::get_identifier_as_lvalue (recording::lvalue *m)
{
return m->access_as_lvalue (*this);
}
const char *
reproducer::get_identifier_as_type (recording::type *m)
{
return m->access_as_type (*this);
}
char *
reproducer::xstrdup_printf (const char *fmt, ...)
{
char *result;
va_list ap;
va_start (ap, fmt);
result = m_allocator.xstrdup_printf_va (fmt, ap);
va_end (ap);
return result;
}
class comma_separated_string
{
public:
comma_separated_string (const auto_vec<recording::rvalue *> &rvalues,
enum recording::precedence prec);
~comma_separated_string ();
const char *as_char_ptr () const { return m_buf; }
private:
char *m_buf;
};
comma_separated_string::comma_separated_string
(const auto_vec<recording::rvalue *> &rvalues,
enum recording::precedence prec)
: m_buf (NULL)
{
size_t sz = 1; 
for (unsigned i = 0; i< rvalues.length (); i++)
{
sz += strlen (rvalues[i]->get_debug_string_parens (prec));
sz += 2; 
}
m_buf = new char[sz];
size_t len = 0;
for (unsigned i = 0; i< rvalues.length (); i++)
{
strcpy (m_buf + len, rvalues[i]->get_debug_string_parens (prec));
len += strlen (rvalues[i]->get_debug_string_parens (prec));
if (i + 1 < rvalues.length ())
{
strcpy (m_buf + len, ", ");
len += 2;
}
}
m_buf[len] = '\0';
}
comma_separated_string::~comma_separated_string ()
{
delete[] m_buf;
}
playback::location *
recording::playback_location (replayer *r, recording::location *loc)
{
if (loc)
return loc->playback_location (r);
else
return NULL;
}
const char *
recording::playback_string (recording::string *str)
{
if (str)
return str->c_str ();
else
return NULL;
}
playback::block *
recording::playback_block (recording::block *b)
{
if (b)
return b->playback_block ();
else
return NULL;
}
recording::context::context (context *parent_ctxt)
: log_user (NULL),
m_parent_ctxt (parent_ctxt),
m_toplevel_ctxt (m_parent_ctxt ? m_parent_ctxt->m_toplevel_ctxt : this),
m_timer (NULL),
m_error_count (0),
m_first_error_str (NULL),
m_owns_first_error_str (false),
m_last_error_str (NULL),
m_owns_last_error_str (false),
m_mementos (),
m_compound_types (),
m_globals (),
m_functions (),
m_FILE_type (NULL),
m_builtins_manager(NULL)
{
if (parent_ctxt)
{
for (unsigned i = 0;
i < sizeof (m_str_options) / sizeof (m_str_options[0]);
i++)
{
const char *parent_opt = parent_ctxt->m_str_options[i];
m_str_options[i] = parent_opt ? xstrdup (parent_opt) : NULL;
}
memcpy (m_int_options,
parent_ctxt->m_int_options,
sizeof (m_int_options));
memcpy (m_bool_options,
parent_ctxt->m_bool_options,
sizeof (m_bool_options));
memcpy (m_inner_bool_options,
parent_ctxt->m_inner_bool_options,
sizeof (m_inner_bool_options));
set_logger (parent_ctxt->get_logger ());
}
else
{
memset (m_str_options, 0, sizeof (m_str_options));
memset (m_int_options, 0, sizeof (m_int_options));
memset (m_bool_options, 0, sizeof (m_bool_options));
memset (m_inner_bool_options, 0, sizeof (m_inner_bool_options));
}
memset (m_basic_types, 0, sizeof (m_basic_types));
}
recording::context::~context ()
{
JIT_LOG_SCOPE (get_logger ());
int i;
memento *m;
FOR_EACH_VEC_ELT (m_mementos, i, m)
{
delete m;
}
for (i = 0; i < GCC_JIT_NUM_STR_OPTIONS; ++i)
free (m_str_options[i]);
char *optname;
FOR_EACH_VEC_ELT (m_command_line_options, i, optname)
free (optname);
if (m_builtins_manager)
delete m_builtins_manager;
if (m_owns_first_error_str)
free (m_first_error_str);
if (m_owns_last_error_str)
if (m_last_error_str != m_first_error_str)
free (m_last_error_str);
}
void
recording::context::record (memento *m)
{
gcc_assert (m);
m_mementos.safe_push (m);
}
void
recording::context::replay_into (replayer *r)
{
JIT_LOG_SCOPE (get_logger ());
int i;
memento *m;
if (m_parent_ctxt)
m_parent_ctxt->replay_into (r);
if (r->errors_occurred ())
return;
FOR_EACH_VEC_ELT (m_mementos, i, m)
{
if (0)
printf ("context %p replaying (%p): %s\n",
(void *)this, (void *)m, m->get_debug_string ());
m->replay_into (r);
if (r->errors_occurred ())
return;
}
}
void
recording::context::disassociate_from_playback ()
{
JIT_LOG_SCOPE (get_logger ());
int i;
memento *m;
if (m_parent_ctxt)
m_parent_ctxt->disassociate_from_playback ();
FOR_EACH_VEC_ELT (m_mementos, i, m)
{
m->set_playback_obj (NULL);
}
}
recording::string *
recording::context::new_string (const char *text)
{
if (!text)
return NULL;
recording::string *result = new string (this, text);
record (result);
return result;
}
recording::location *
recording::context::new_location (const char *filename,
int line,
int column,
bool created_by_user)
{
recording::location *result =
new recording::location (this,
new_string (filename),
line, column,
created_by_user);
record (result);
return result;
}
recording::type *
recording::context::get_type (enum gcc_jit_types kind)
{
if (!m_basic_types[kind])
{
if (m_parent_ctxt)
m_basic_types[kind] = m_parent_ctxt->get_type (kind);
else
{
recording::type *result = new memento_of_get_type (this, kind);
record (result);
m_basic_types[kind] = result;
}
}
return m_basic_types[kind];
}
recording::type *
recording::context::get_int_type (int num_bytes, int is_signed)
{
const int num_bits = num_bytes * 8;
if (num_bits == INT_TYPE_SIZE)
return get_type (is_signed
? GCC_JIT_TYPE_INT
: GCC_JIT_TYPE_UNSIGNED_INT);
if (num_bits == CHAR_TYPE_SIZE)
return get_type (is_signed
? GCC_JIT_TYPE_SIGNED_CHAR
: GCC_JIT_TYPE_UNSIGNED_CHAR);
if (num_bits == SHORT_TYPE_SIZE)
return get_type (is_signed
? GCC_JIT_TYPE_SHORT
: GCC_JIT_TYPE_UNSIGNED_SHORT);
if (num_bits == LONG_TYPE_SIZE)
return get_type (is_signed
? GCC_JIT_TYPE_LONG
: GCC_JIT_TYPE_UNSIGNED_LONG);
if (num_bits == LONG_LONG_TYPE_SIZE)
return get_type (is_signed
? GCC_JIT_TYPE_LONG_LONG
: GCC_JIT_TYPE_UNSIGNED_LONG_LONG);
gcc_unreachable ();
}
recording::type *
recording::context::new_array_type (recording::location *loc,
recording::type *element_type,
int num_elements)
{
if (struct_ *s = element_type->dyn_cast_struct ())
if (!s->get_fields ())
{
add_error (NULL,
"cannot create an array of type %s"
" until the fields have been set",
s->get_name ()->c_str ());
return NULL;
}
recording::type *result =
new recording::array_type (this, loc, element_type, num_elements);
record (result);
return result;
}
recording::field *
recording::context::new_field (recording::location *loc,
recording::type *type,
const char *name)
{
recording::field *result =
new recording::field (this, loc, type, new_string (name));
record (result);
return result;
}
recording::struct_ *
recording::context::new_struct_type (recording::location *loc,
const char *name)
{
recording::struct_ *result = new struct_ (this, loc, new_string (name));
record (result);
m_compound_types.safe_push (result);
return result;
}
recording::union_ *
recording::context::new_union_type (recording::location *loc,
const char *name)
{
recording::union_ *result = new union_ (this, loc, new_string (name));
record (result);
m_compound_types.safe_push (result);
return result;
}
recording::function_type *
recording::context::new_function_type (recording::type *return_type,
int num_params,
recording::type **param_types,
int is_variadic)
{
recording::function_type *fn_type
= new function_type (this,
return_type,
num_params,
param_types,
is_variadic);
record (fn_type);
return fn_type;
}
recording::type *
recording::context::new_function_ptr_type (recording::location *, 
recording::type *return_type,
int num_params,
recording::type **param_types,
int is_variadic)
{
recording::function_type *fn_type
= new_function_type (return_type,
num_params,
param_types,
is_variadic);
return fn_type->get_pointer ();
}
recording::param *
recording::context::new_param (recording::location *loc,
recording::type *type,
const char *name)
{
recording::param *result = new recording::param (this, loc, type, new_string (name));
record (result);
return result;
}
recording::function *
recording::context::new_function (recording::location *loc,
enum gcc_jit_function_kind kind,
recording::type *return_type,
const char *name,
int num_params,
recording::param **params,
int is_variadic,
enum built_in_function builtin_id)
{
recording::function *result =
new recording::function (this,
loc, kind, return_type,
new_string (name),
num_params, params, is_variadic,
builtin_id);
record (result);
m_functions.safe_push (result);
return result;
}
builtins_manager *
recording::context::get_builtins_manager ()
{
if (m_parent_ctxt)
return m_parent_ctxt->get_builtins_manager ();
if (!m_builtins_manager)
m_builtins_manager = new builtins_manager (this);
return m_builtins_manager;
}
recording::function *
recording::context::get_builtin_function (const char *name)
{
builtins_manager *bm = get_builtins_manager ();
return bm->get_builtin_function (name);
}
recording::lvalue *
recording::context::new_global (recording::location *loc,
enum gcc_jit_global_kind kind,
recording::type *type,
const char *name)
{
recording::global *result =
new recording::global (this, loc, kind, type, new_string (name));
record (result);
m_globals.safe_push (result);
return result;
}
recording::rvalue *
recording::context::new_string_literal (const char *value)
{
recording::rvalue *result =
new memento_of_new_string_literal (this, NULL, new_string (value));
record (result);
return result;
}
recording::rvalue *
recording::context::new_rvalue_from_vector (location *loc,
vector_type *type,
rvalue **elements)
{
recording::rvalue *result
= new memento_of_new_rvalue_from_vector (this, loc, type, elements);
record (result);
return result;
}
recording::rvalue *
recording::context::new_unary_op (recording::location *loc,
enum gcc_jit_unary_op op,
recording::type *result_type,
recording::rvalue *a)
{
recording::rvalue *result =
new unary_op (this, loc, op, result_type, a);
record (result);
return result;
}
recording::rvalue *
recording::context::new_binary_op (recording::location *loc,
enum gcc_jit_binary_op op,
recording::type *result_type,
recording::rvalue *a,
recording::rvalue *b)
{
recording::rvalue *result =
new binary_op (this, loc, op, result_type, a, b);
record (result);
return result;
}
recording::rvalue *
recording::context::new_comparison (recording::location *loc,
enum gcc_jit_comparison op,
recording::rvalue *a,
recording::rvalue *b)
{
recording::rvalue *result = new comparison (this, loc, op, a, b);
record (result);
return result;
}
recording::rvalue *
recording::context::new_cast (recording::location *loc,
recording::rvalue *expr,
recording::type *type_)
{
recording::rvalue *result = new cast (this, loc, expr, type_);
record (result);
return result;
}
recording::rvalue *
recording::context::new_call (recording::location *loc,
function *func,
int numargs , recording::rvalue **args)
{
recording::rvalue *result = new call (this, loc, func, numargs, args);
record (result);
return result;
}
recording::rvalue *
recording::context::new_call_through_ptr (recording::location *loc,
recording::rvalue *fn_ptr,
int numargs,
recording::rvalue **args)
{
recording::rvalue *result = new call_through_ptr (this, loc, fn_ptr, numargs, args);
record (result);
return result;
}
recording::lvalue *
recording::context::new_array_access (recording::location *loc,
recording::rvalue *ptr,
recording::rvalue *index)
{
recording::lvalue *result = new array_access (this, loc, ptr, index);
record (result);
return result;
}
recording::case_ *
recording::context::new_case (recording::rvalue *min_value,
recording::rvalue *max_value,
recording::block *block)
{
recording::case_ *result = new case_ (this, min_value, max_value, block);
record (result);
return result;
}
void
recording::context::set_str_option (enum gcc_jit_str_option opt,
const char *value)
{
if (opt < 0 || opt >= GCC_JIT_NUM_STR_OPTIONS)
{
add_error (NULL,
"unrecognized (enum gcc_jit_str_option) value: %i", opt);
return;
}
free (m_str_options[opt]);
m_str_options[opt] = value ? xstrdup (value) : NULL;
log_str_option (opt);
}
void
recording::context::set_int_option (enum gcc_jit_int_option opt,
int value)
{
if (opt < 0 || opt >= GCC_JIT_NUM_INT_OPTIONS)
{
add_error (NULL,
"unrecognized (enum gcc_jit_int_option) value: %i", opt);
return;
}
m_int_options[opt] = value;
log_int_option (opt);
}
void
recording::context::set_bool_option (enum gcc_jit_bool_option opt,
int value)
{
if (opt < 0 || opt >= GCC_JIT_NUM_BOOL_OPTIONS)
{
add_error (NULL,
"unrecognized (enum gcc_jit_bool_option) value: %i", opt);
return;
}
m_bool_options[opt] = value ? true : false;
log_bool_option (opt);
}
void
recording::context::set_inner_bool_option (enum inner_bool_option inner_opt,
int value)
{
gcc_assert (inner_opt >= 0 && inner_opt < NUM_INNER_BOOL_OPTIONS);
m_inner_bool_options[inner_opt] = value ? true : false;
log_inner_bool_option (inner_opt);
}
void
recording::context::add_command_line_option (const char *optname)
{
m_command_line_options.safe_push (xstrdup (optname));
}
void
recording::context::append_command_line_options (vec <char *> *argvec)
{
if (m_parent_ctxt)
m_parent_ctxt->append_command_line_options (argvec);
int i;
char *optname;
FOR_EACH_VEC_ELT (m_command_line_options, i, optname)
argvec->safe_push (xstrdup (optname));
}
void
recording::context::enable_dump (const char *dumpname,
char **out_ptr)
{
requested_dump d;
gcc_assert (dumpname);
gcc_assert (out_ptr);
d.m_dumpname = dumpname;
d.m_out_ptr = out_ptr;
*out_ptr = NULL;
m_requested_dumps.safe_push (d);
}
result *
recording::context::compile ()
{
JIT_LOG_SCOPE (get_logger ());
log_all_options ();
validate ();
if (errors_occurred ())
return NULL;
::gcc::jit::playback::compile_to_memory replayer (this);
replayer.compile ();
return replayer.get_result_obj ();
}
void
recording::context::compile_to_file (enum gcc_jit_output_kind output_kind,
const char *output_path)
{
JIT_LOG_SCOPE (get_logger ());
log_all_options ();
validate ();
if (errors_occurred ())
return;
::gcc::jit::playback::compile_to_file replayer (this,
output_kind,
output_path);
replayer.compile ();
}
void
recording::context::add_error (location *loc, const char *fmt, ...)
{
va_list ap;
va_start (ap, fmt);
add_error_va (loc, fmt, ap);
va_end (ap);
}
void
recording::context::add_error_va (location *loc, const char *fmt, va_list ap)
{
int len;
char *malloced_msg;
const char *errmsg;
bool has_ownership;
JIT_LOG_SCOPE (get_logger ());
len = vasprintf (&malloced_msg, fmt, ap);
if (malloced_msg == NULL || len < 0)
{
errmsg = "out of memory generating error message";
has_ownership = false;
}
else
{
errmsg = malloced_msg;
has_ownership = true;
}
if (get_logger ())
get_logger ()->log ("error %i: %s", m_error_count, errmsg);
const char *ctxt_progname =
get_str_option (GCC_JIT_STR_OPTION_PROGNAME);
if (!ctxt_progname)
ctxt_progname = "libgccjit.so";
if (loc)
fprintf (stderr, "%s: %s: error: %s\n",
ctxt_progname,
loc->get_debug_string (),
errmsg);
else
fprintf (stderr, "%s: error: %s\n",
ctxt_progname,
errmsg);
if (!m_error_count)
{
m_first_error_str = const_cast <char *> (errmsg);
m_owns_first_error_str = has_ownership;
}
if (m_owns_last_error_str)
if (m_last_error_str != m_first_error_str)
free (m_last_error_str);
m_last_error_str = const_cast <char *> (errmsg);
m_owns_last_error_str = has_ownership;
m_error_count++;
}
const char *
recording::context::get_first_error () const
{
return m_first_error_str;
}
const char *
recording::context::get_last_error () const
{
return m_last_error_str;
}
recording::type *
recording::context::get_opaque_FILE_type ()
{
if (!m_FILE_type)
m_FILE_type = new_struct_type (NULL, "FILE");
return m_FILE_type;
}
void
recording::context::dump_to_file (const char *path, bool update_locations)
{
int i;
dump d (*this, path, update_locations);
compound_type *st;
FOR_EACH_VEC_ELT (m_compound_types, i, st)
{
d.write ("%s;\n\n", st->get_debug_string ());
}
FOR_EACH_VEC_ELT (m_compound_types, i, st)
if (st->get_fields ())
{
st->get_fields ()->write_to_dump (d);
d.write ("\n");
}
global *g;
FOR_EACH_VEC_ELT (m_globals, i, g)
{
g->write_to_dump (d);
}
if (!m_globals.is_empty ())
d.write ("\n");
function *fn;
FOR_EACH_VEC_ELT (m_functions, i, fn)
{
fn->write_to_dump (d);
}
}
static const char * const
str_option_reproducer_strings[GCC_JIT_NUM_STR_OPTIONS] = {
"GCC_JIT_STR_OPTION_PROGNAME"
};
static const char * const
int_option_reproducer_strings[GCC_JIT_NUM_INT_OPTIONS] = {
"GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL"
};
static const char * const
bool_option_reproducer_strings[GCC_JIT_NUM_BOOL_OPTIONS] = {
"GCC_JIT_BOOL_OPTION_DEBUGINFO",
"GCC_JIT_BOOL_OPTION_DUMP_INITIAL_TREE",
"GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE",
"GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE",
"GCC_JIT_BOOL_OPTION_DUMP_SUMMARY",
"GCC_JIT_BOOL_OPTION_DUMP_EVERYTHING",
"GCC_JIT_BOOL_OPTION_SELFCHECK_GC",
"GCC_JIT_BOOL_OPTION_KEEP_INTERMEDIATES"
};
static const char * const
inner_bool_option_reproducer_strings[NUM_INNER_BOOL_OPTIONS] = {
"gcc_jit_context_set_bool_allow_unreachable_blocks",
"gcc_jit_context_set_bool_use_external_driver"
};
void
recording::context::log_all_options () const
{
int opt_idx;
if (!get_logger ())
return;
for (opt_idx = 0; opt_idx < GCC_JIT_NUM_STR_OPTIONS; opt_idx++)
log_str_option ((enum gcc_jit_str_option)opt_idx);
for (opt_idx = 0; opt_idx < GCC_JIT_NUM_INT_OPTIONS; opt_idx++)
log_int_option ((enum gcc_jit_int_option)opt_idx);
for (opt_idx = 0; opt_idx < GCC_JIT_NUM_BOOL_OPTIONS; opt_idx++)
log_bool_option ((enum gcc_jit_bool_option)opt_idx);
for (opt_idx = 0; opt_idx < NUM_INNER_BOOL_OPTIONS; opt_idx++)
log_inner_bool_option ((enum inner_bool_option)opt_idx);
}
void
recording::context::log_str_option (enum gcc_jit_str_option opt) const
{
gcc_assert (opt < GCC_JIT_NUM_STR_OPTIONS);
if (get_logger ())
{
if (m_str_options[opt])
log ("%s: \"%s\"",
str_option_reproducer_strings[opt],
m_str_options[opt]);
else
log ("%s: NULL",
str_option_reproducer_strings[opt]);
}
}
void
recording::context::log_int_option (enum gcc_jit_int_option opt) const
{
gcc_assert (opt < GCC_JIT_NUM_INT_OPTIONS);
if (get_logger ())
log ("%s: %i",
int_option_reproducer_strings[opt],
m_int_options[opt]);
}
void
recording::context::log_bool_option (enum gcc_jit_bool_option opt) const
{
gcc_assert (opt < GCC_JIT_NUM_BOOL_OPTIONS);
if (get_logger ())
log ("%s: %s",
bool_option_reproducer_strings[opt],
m_bool_options[opt] ? "true" : "false");
}
void
recording::context::log_inner_bool_option (enum inner_bool_option opt) const
{
gcc_assert (opt < NUM_INNER_BOOL_OPTIONS);
if (get_logger ())
log ("%s: %s",
inner_bool_option_reproducer_strings[opt],
m_inner_bool_options[opt] ? "true" : "false");
}
void
recording::context::dump_reproducer_to_file (const char *path)
{
JIT_LOG_SCOPE (get_logger ());
reproducer r (*this, path);
auto_vec <context *> ascending_contexts;
for (context *ctxt = this; ctxt; ctxt = ctxt->m_parent_ctxt)
ascending_contexts.safe_push (ctxt);
unsigned num_ctxts = ascending_contexts.length ();
auto_vec <context *> contexts (num_ctxts);
for (unsigned i = 0; i < num_ctxts; i++)
contexts.safe_push (ascending_contexts[num_ctxts - (i + 1)]);
gcc_assert (contexts[0]);
gcc_assert (contexts[0]->m_toplevel_ctxt == contexts[0]);
gcc_assert (contexts[contexts.length () - 1] == this);
gcc_assert (contexts[contexts.length () - 1]->m_toplevel_ctxt
== contexts[0]);
r.write ("\n");
r.write ("#include <libgccjit.h>\n\n");
r.write ("#pragma GCC diagnostic ignored \"-Wunused-variable\"\n\n");
r.write ("static void\nset_options (");
r.write_params (contexts);
r.write (");\n\n");
r.write ("static void\ncreate_code (");
r.write_params (contexts);
r.write (");\n\n");
r.write ("int\nmain (int argc, const char **argv)\n");
r.write ("{\n");
for (unsigned i = 0; i < num_ctxts; i++)
r.write ("  gcc_jit_context *%s;\n",
r.get_identifier (contexts[i]));
r.write ("  gcc_jit_result *result;\n"
"\n");
r.write ("  %s = gcc_jit_context_acquire ();\n",
r.get_identifier (contexts[0]));
for (unsigned i = 1; i < num_ctxts; i++)
r.write ("  %s = gcc_jit_context_new_child_context (%s);\n",
r.get_identifier (contexts[i]),
r.get_identifier (contexts[i - 1]));
r.write ("  set_options (");
r.write_args (contexts);
r.write (");\n");
r.write ("  create_code (");
r.write_args (contexts);
r.write (");\n");
r.write ("  result = gcc_jit_context_compile (%s);\n",
r.get_identifier (this));
for (unsigned i = num_ctxts; i > 0; i--)
r.write ("  gcc_jit_context_release (%s);\n",
r.get_identifier (contexts[i - 1]));
r.write ("  gcc_jit_result_release (result);\n"
"  return 0;\n"
"}\n\n");
for (unsigned ctxt_idx = 0; ctxt_idx < num_ctxts; ctxt_idx++)
{
if (m_requested_dumps.length ())
{
r.write ("\n",
r.get_identifier (contexts[ctxt_idx]));
for (unsigned i = 0; i < m_requested_dumps.length (); i++)
r.write ("static char *dump_%p;\n",
(void *)&m_requested_dumps[i]);
r.write ("\n");
}
}
r.write ("static void\nset_options (");
r.write_params (contexts);
r.write (")\n{\n");
for (unsigned ctxt_idx = 0; ctxt_idx < num_ctxts; ctxt_idx++)
{
if (ctxt_idx > 0)
r.write ("\n");
r.write ("  \n",
r.get_identifier (contexts[ctxt_idx]));
r.write ("  \n");
for (int opt_idx = 0; opt_idx < GCC_JIT_NUM_STR_OPTIONS; opt_idx++)
{
r.write ("  gcc_jit_context_set_str_option (%s,\n"
"                                  %s,\n",
r.get_identifier (contexts[ctxt_idx]),
str_option_reproducer_strings[opt_idx]);
if (m_str_options[opt_idx])
r.write ("                                  \"%s\");\n",
m_str_options[opt_idx]);
else
r.write ("                                  NULL);\n");
}
r.write ("  \n");
for (int opt_idx = 0; opt_idx < GCC_JIT_NUM_INT_OPTIONS; opt_idx++)
r.write ("  gcc_jit_context_set_int_option (%s,\n"
"                                  %s,\n"
"                                  %i);\n",
r.get_identifier (contexts[ctxt_idx]),
int_option_reproducer_strings[opt_idx],
m_int_options[opt_idx]);
r.write ("  \n");
for (int opt_idx = 0; opt_idx < GCC_JIT_NUM_BOOL_OPTIONS; opt_idx++)
r.write ("  gcc_jit_context_set_bool_option (%s,\n"
"                                  %s,\n"
"                                  %i);\n",
r.get_identifier (contexts[ctxt_idx]),
bool_option_reproducer_strings[opt_idx],
m_bool_options[opt_idx]);
for (int opt_idx = 0; opt_idx < NUM_INNER_BOOL_OPTIONS; opt_idx++)
r.write ("  %s (%s, %i);\n",
inner_bool_option_reproducer_strings[opt_idx],
r.get_identifier (contexts[ctxt_idx]),
m_inner_bool_options[opt_idx]);
if (!m_command_line_options.is_empty ())
{
int i;
char *optname;
r.write ("  \n");
FOR_EACH_VEC_ELT (m_command_line_options, i, optname)
r.write ("  gcc_jit_context_add_command_line_option (%s, \"%s\");\n",
r.get_identifier (contexts[ctxt_idx]),
optname);
}
if (m_requested_dumps.length ())
{
r.write ("  \n");
for (unsigned i = 0; i < m_requested_dumps.length (); i++)
{
r.write ("  gcc_jit_context_enable_dump (%s,\n"
"                               \"%s\",\n"
"                               &dump_%p);\n",
r.get_identifier (contexts[ctxt_idx]),
m_requested_dumps[i].m_dumpname,
(void *)&m_requested_dumps[i]);
}
}
}
r.write ("}\n\n");
r.write ("static void\ncreate_code (");
r.write_params (contexts);
r.write (")\n"
"{\n");
for (unsigned ctxt_idx = 0; ctxt_idx < num_ctxts; ctxt_idx++)
{
memento *m;
int i;
if (ctxt_idx > 0)
r.write ("\n\n");
r.write ("  \n",
r.get_identifier (contexts[ctxt_idx]));
FOR_EACH_VEC_ELT (contexts[ctxt_idx]->m_mementos, i, m)
m->write_reproducer (r);
}
r.write ("}\n");
}
void
recording::context::get_all_requested_dumps (vec <recording::requested_dump> *out)
{
if (m_parent_ctxt)
m_parent_ctxt->get_all_requested_dumps (out);
out->reserve (m_requested_dumps.length ());
out->splice (m_requested_dumps);
}
void
recording::context::validate ()
{
JIT_LOG_SCOPE (get_logger ());
if (m_parent_ctxt)
m_parent_ctxt->validate ();
int i;
function *fn;
FOR_EACH_VEC_ELT (m_functions, i, fn)
fn->validate ();
}
const char *
recording::memento::get_debug_string ()
{
if (!m_debug_string)
m_debug_string = make_debug_string ();
return m_debug_string->c_str ();
}
void
recording::memento::write_to_dump (dump &d)
{
d.write("  %s\n", get_debug_string ());
}
recording::string::string (context *ctxt, const char *text)
: memento (ctxt)
{
m_len = strlen (text);
m_buffer = new char[m_len + 1];
strcpy (m_buffer, text);
}
recording::string::~string ()
{
delete[] m_buffer;
}
recording::string *
recording::string::from_printf (context *ctxt, const char *fmt, ...)
{
int len;
va_list ap;
char *buf;
recording::string *result;
va_start (ap, fmt);
len = vasprintf (&buf, fmt, ap);
va_end (ap);
if (buf == NULL || len < 0)
{
ctxt->add_error (NULL, "malloc failure");
return NULL;
}
result = ctxt->new_string (buf);
free (buf);
return result;
}
recording::string *
recording::string::make_debug_string ()
{
if (m_buffer[0] == '"')
return this;
size_t sz = (1 
+ (m_len * 2) 
+ 1 
+ 1); 
char *tmp = new char[sz];
size_t len = 0;
#define APPEND(CH)  do { gcc_assert (len < sz); tmp[len++] = (CH); } while (0)
APPEND('"'); 
for (size_t i = 0; i < m_len ; i++)
{
char ch = m_buffer[i];
if (ch == '\t' || ch == '\n' || ch == '\\' || ch == '"')
APPEND('\\');
APPEND(ch);
}
APPEND('"'); 
#undef APPEND
tmp[len] = '\0'; 
string *result = m_ctxt->new_string (tmp);
delete[] tmp;
return result;
}
void
recording::string::write_reproducer (reproducer &)
{
}
void
recording::location::replay_into (replayer *r)
{
m_playback_obj = r->new_location (this,
m_filename->c_str (),
m_line,
m_column);
}
recording::string *
recording::location::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s:%i:%i",
m_filename->c_str (), m_line, m_column);
}
void
recording::location::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "loc");
r.write ("  gcc_jit_location *%s =\n"
"    gcc_jit_context_new_location (%s, \n"
"    %s, \n"
"    %i, \n"
"    %i);\n",
id,
r.get_identifier (get_context ()),
m_filename->get_debug_string (),
m_line, m_column);
}
recording::type *
recording::type::get_pointer ()
{
if (!m_pointer_to_this_type)
{
m_pointer_to_this_type = new memento_of_get_pointer (this);
m_ctxt->record (m_pointer_to_this_type);
}
return m_pointer_to_this_type;
}
recording::type *
recording::type::get_const ()
{
recording::type *result = new memento_of_get_const (this);
m_ctxt->record (result);
return result;
}
recording::type *
recording::type::get_volatile ()
{
recording::type *result = new memento_of_get_volatile (this);
m_ctxt->record (result);
return result;
}
recording::type *
recording::type::get_aligned (size_t alignment_in_bytes)
{
recording::type *result
= new memento_of_get_aligned (this, alignment_in_bytes);
m_ctxt->record (result);
return result;
}
recording::type *
recording::type::get_vector (size_t num_units)
{
recording::type *result
= new vector_type (this, num_units);
m_ctxt->record (result);
return result;
}
const char *
recording::type::access_as_type (reproducer &r)
{
return r.get_identifier (this);
}
recording::type *
recording::memento_of_get_type::dereference ()
{
switch (m_kind)
{
default: gcc_unreachable ();
case GCC_JIT_TYPE_VOID:
return NULL;
case GCC_JIT_TYPE_VOID_PTR:
return m_ctxt->get_type (GCC_JIT_TYPE_VOID);
case GCC_JIT_TYPE_BOOL:
case GCC_JIT_TYPE_CHAR:
case GCC_JIT_TYPE_SIGNED_CHAR:
case GCC_JIT_TYPE_UNSIGNED_CHAR:
case GCC_JIT_TYPE_SHORT:
case GCC_JIT_TYPE_UNSIGNED_SHORT:
case GCC_JIT_TYPE_INT:
case GCC_JIT_TYPE_UNSIGNED_INT:
case GCC_JIT_TYPE_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG:
case GCC_JIT_TYPE_LONG_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
case GCC_JIT_TYPE_FLOAT:
case GCC_JIT_TYPE_DOUBLE:
case GCC_JIT_TYPE_LONG_DOUBLE:
case GCC_JIT_TYPE_COMPLEX_FLOAT:
case GCC_JIT_TYPE_COMPLEX_DOUBLE:
case GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE:
return NULL;
case GCC_JIT_TYPE_CONST_CHAR_PTR:
return m_ctxt->get_type (GCC_JIT_TYPE_CHAR)->get_const ();
case GCC_JIT_TYPE_SIZE_T:
return NULL;
case GCC_JIT_TYPE_FILE_PTR:
return m_ctxt->get_opaque_FILE_type ();
}
}
bool
recording::memento_of_get_type::is_int () const
{
switch (m_kind)
{
default: gcc_unreachable ();
case GCC_JIT_TYPE_VOID:
return false;
case GCC_JIT_TYPE_VOID_PTR:
return false;
case GCC_JIT_TYPE_BOOL:
return false;
case GCC_JIT_TYPE_CHAR:
case GCC_JIT_TYPE_SIGNED_CHAR:
case GCC_JIT_TYPE_UNSIGNED_CHAR:
case GCC_JIT_TYPE_SHORT:
case GCC_JIT_TYPE_UNSIGNED_SHORT:
case GCC_JIT_TYPE_INT:
case GCC_JIT_TYPE_UNSIGNED_INT:
case GCC_JIT_TYPE_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG:
case GCC_JIT_TYPE_LONG_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
return true;
case GCC_JIT_TYPE_FLOAT:
case GCC_JIT_TYPE_DOUBLE:
case GCC_JIT_TYPE_LONG_DOUBLE:
return false;
case GCC_JIT_TYPE_CONST_CHAR_PTR:
return false;
case GCC_JIT_TYPE_SIZE_T:
return true;
case GCC_JIT_TYPE_FILE_PTR:
return false;
case GCC_JIT_TYPE_COMPLEX_FLOAT:
case GCC_JIT_TYPE_COMPLEX_DOUBLE:
case GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE:
return false;
}
}
bool
recording::memento_of_get_type::is_float () const
{
switch (m_kind)
{
default: gcc_unreachable ();
case GCC_JIT_TYPE_VOID:
return false;
case GCC_JIT_TYPE_VOID_PTR:
return false;
case GCC_JIT_TYPE_BOOL:
return false;
case GCC_JIT_TYPE_CHAR:
case GCC_JIT_TYPE_SIGNED_CHAR:
case GCC_JIT_TYPE_UNSIGNED_CHAR:
case GCC_JIT_TYPE_SHORT:
case GCC_JIT_TYPE_UNSIGNED_SHORT:
case GCC_JIT_TYPE_INT:
case GCC_JIT_TYPE_UNSIGNED_INT:
case GCC_JIT_TYPE_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG:
case GCC_JIT_TYPE_LONG_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
return false;
case GCC_JIT_TYPE_FLOAT:
case GCC_JIT_TYPE_DOUBLE:
case GCC_JIT_TYPE_LONG_DOUBLE:
return true;
case GCC_JIT_TYPE_CONST_CHAR_PTR:
return false;
case GCC_JIT_TYPE_SIZE_T:
return false;
case GCC_JIT_TYPE_FILE_PTR:
return false;
case GCC_JIT_TYPE_COMPLEX_FLOAT:
case GCC_JIT_TYPE_COMPLEX_DOUBLE:
case GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE:
return true;
}
}
bool
recording::memento_of_get_type::is_bool () const
{
switch (m_kind)
{
default: gcc_unreachable ();
case GCC_JIT_TYPE_VOID:
return false;
case GCC_JIT_TYPE_VOID_PTR:
return false;
case GCC_JIT_TYPE_BOOL:
return true;
case GCC_JIT_TYPE_CHAR:
case GCC_JIT_TYPE_SIGNED_CHAR:
case GCC_JIT_TYPE_UNSIGNED_CHAR:
case GCC_JIT_TYPE_SHORT:
case GCC_JIT_TYPE_UNSIGNED_SHORT:
case GCC_JIT_TYPE_INT:
case GCC_JIT_TYPE_UNSIGNED_INT:
case GCC_JIT_TYPE_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG:
case GCC_JIT_TYPE_LONG_LONG:
case GCC_JIT_TYPE_UNSIGNED_LONG_LONG:
return false;
case GCC_JIT_TYPE_FLOAT:
case GCC_JIT_TYPE_DOUBLE:
case GCC_JIT_TYPE_LONG_DOUBLE:
return false;
case GCC_JIT_TYPE_CONST_CHAR_PTR:
return false;
case GCC_JIT_TYPE_SIZE_T:
return false;
case GCC_JIT_TYPE_FILE_PTR:
return false;
case GCC_JIT_TYPE_COMPLEX_FLOAT:
case GCC_JIT_TYPE_COMPLEX_DOUBLE:
case GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE:
return false;
}
}
void
recording::memento_of_get_type::replay_into (replayer *r)
{
set_playback_obj (r->get_type (m_kind));
}
static const char * const get_type_strings[] = {
"void",    
"void *",  
"bool",  
"char",           
"signed char",    
"unsigned char",  
"short",           
"unsigned short",  
"int",           
"unsigned int",  
"long",           
"unsigned long",  
"long long",           
"unsigned long long",  
"float",        
"double",       
"long double",  
"const char *",  
"size_t",  
"FILE *",  
"complex float", 
"complex double", 
"complex long double"  
};
recording::string *
recording::memento_of_get_type::make_debug_string ()
{
return m_ctxt->new_string (get_type_strings[m_kind]);
}
static const char * const get_type_enum_strings[] = {
"GCC_JIT_TYPE_VOID",
"GCC_JIT_TYPE_VOID_PTR",
"GCC_JIT_TYPE_BOOL",
"GCC_JIT_TYPE_CHAR",
"GCC_JIT_TYPE_SIGNED_CHAR",
"GCC_JIT_TYPE_UNSIGNED_CHAR",
"GCC_JIT_TYPE_SHORT",
"GCC_JIT_TYPE_UNSIGNED_SHORT",
"GCC_JIT_TYPE_INT",
"GCC_JIT_TYPE_UNSIGNED_INT",
"GCC_JIT_TYPE_LONG",
"GCC_JIT_TYPE_UNSIGNED_LONG",
"GCC_JIT_TYPE_LONG_LONG",
"GCC_JIT_TYPE_UNSIGNED_LONG_LONG",
"GCC_JIT_TYPE_FLOAT",
"GCC_JIT_TYPE_DOUBLE",
"GCC_JIT_TYPE_LONG_DOUBLE",
"GCC_JIT_TYPE_CONST_CHAR_PTR",
"GCC_JIT_TYPE_SIZE_T",
"GCC_JIT_TYPE_FILE_PTR",
"GCC_JIT_TYPE_COMPLEX_FLOAT",
"GCC_JIT_TYPE_COMPLEX_DOUBLE",
"GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE"
};
void
recording::memento_of_get_type::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s = gcc_jit_context_get_type (%s, %s);\n",
id,
r.get_identifier (get_context ()),
get_type_enum_strings[m_kind]);
}
bool
recording::memento_of_get_pointer::accepts_writes_from (type *rtype)
{
type *rtype_points_to = rtype->is_pointer ();
if (!rtype_points_to)
return false;
return m_other_type->unqualified ()
->accepts_writes_from (rtype_points_to);
}
void
recording::memento_of_get_pointer::replay_into (replayer *)
{
set_playback_obj (m_other_type->playback_type ()->get_pointer ());
}
recording::string *
recording::memento_of_get_pointer::make_debug_string ()
{
if (function_type *fn_type = m_other_type->dyn_cast_function_type ())
return fn_type->make_debug_string_with_ptr ();
return string::from_printf (m_ctxt,
"%s *", m_other_type->get_debug_string ());
}
void
recording::memento_of_get_pointer::write_reproducer (reproducer &r)
{
if (function_type *fn_type = m_other_type->dyn_cast_function_type ())
{
fn_type->write_deferred_reproducer (r, this);
return;
}
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_type_get_pointer (%s);\n",
id,
r.get_identifier_as_type (m_other_type));
}
void
recording::memento_of_get_const::replay_into (replayer *)
{
set_playback_obj (m_other_type->playback_type ()->get_const ());
}
recording::string *
recording::memento_of_get_const::make_debug_string ()
{
return string::from_printf (m_ctxt,
"const %s", m_other_type->get_debug_string ());
}
void
recording::memento_of_get_const::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_type_get_const (%s);\n",
id,
r.get_identifier_as_type (m_other_type));
}
void
recording::memento_of_get_volatile::replay_into (replayer *)
{
set_playback_obj (m_other_type->playback_type ()->get_volatile ());
}
recording::string *
recording::memento_of_get_volatile::make_debug_string ()
{
return string::from_printf (m_ctxt,
"volatile %s", m_other_type->get_debug_string ());
}
void
recording::memento_of_get_volatile::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_type_get_volatile (%s);\n",
id,
r.get_identifier_as_type (m_other_type));
}
void
recording::memento_of_get_aligned::replay_into (replayer *)
{
set_playback_obj
(m_other_type->playback_type ()->get_aligned (m_alignment_in_bytes));
}
recording::string *
recording::memento_of_get_aligned::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s  __attribute__((aligned(%zi)))",
m_other_type->get_debug_string (),
m_alignment_in_bytes);
}
void
recording::memento_of_get_aligned::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_type_get_aligned (%s, %zi);\n",
id,
r.get_identifier_as_type (m_other_type),
m_alignment_in_bytes);
}
void
recording::vector_type::replay_into (replayer *)
{
set_playback_obj
(m_other_type->playback_type ()->get_vector (m_num_units));
}
recording::string *
recording::vector_type::make_debug_string ()
{
return string::from_printf
(m_ctxt,
"%s  __attribute__((vector_size(sizeof (%s) * %zi)))",
m_other_type->get_debug_string (),
m_other_type->get_debug_string (),
m_num_units);
}
void
recording::vector_type::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_type_get_vector (%s, %zi);\n",
id,
r.get_identifier_as_type (m_other_type),
m_num_units);
}
recording::type *
recording::array_type::dereference ()
{
return m_element_type;
}
void
recording::array_type::replay_into (replayer *r)
{
set_playback_obj (r->new_array_type (playback_location (r, m_loc),
m_element_type->playback_type (),
m_num_elements));
}
recording::string *
recording::array_type::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s[%d]",
m_element_type->get_debug_string (),
m_num_elements);
}
void
recording::array_type::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "array_type");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_context_new_array_type (%s,\n"
"                                    %s, \n"
"                                    %s, \n"
"                                    %i); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_type (m_element_type),
m_num_elements);
}
recording::function_type::function_type (context *ctxt,
type *return_type,
int num_params,
type **param_types,
int is_variadic)
: type (ctxt),
m_return_type (return_type),
m_param_types (),
m_is_variadic (is_variadic)
{
for (int i = 0; i< num_params; i++)
m_param_types.safe_push (param_types[i]);
}
recording::type *
recording::function_type::dereference ()
{
return NULL;
}
bool
recording::function_type::is_same_type_as (type *other)
{
gcc_assert (other);
function_type *other_fn_type = other->dyn_cast_function_type ();
if (!other_fn_type)
return false;
if (!m_return_type->is_same_type_as (other_fn_type->m_return_type))
return false;
if (m_param_types.length () != other_fn_type->m_param_types.length ())
return false;
unsigned i;
type *param_type;
FOR_EACH_VEC_ELT (m_param_types, i, param_type)
if (!param_type->is_same_type_as (other_fn_type->m_param_types[i]))
return false;
if (m_is_variadic != other_fn_type->m_is_variadic)
return false;
return true;
}
void
recording::function_type::replay_into (replayer *r)
{
auto_vec <playback::type *> param_types;
int i;
recording::type *type;
param_types.create (m_param_types.length ());
FOR_EACH_VEC_ELT (m_param_types, i, type)
param_types.safe_push (type->playback_type ());
set_playback_obj (r->new_function_type (m_return_type->playback_type (),
&param_types,
m_is_variadic));
}
recording::string *
recording::function_type::make_debug_string_with_ptr ()
{
return make_debug_string_with ("(*) ");
}
recording::string *
recording::function_type::make_debug_string ()
{
return make_debug_string_with ("");
}
recording::string *
recording::function_type::make_debug_string_with (const char *insert)
{
size_t sz = 1; 
for (unsigned i = 0; i< m_param_types.length (); i++)
{
sz += strlen (m_param_types[i]->get_debug_string ());
sz += 2; 
}
if (m_is_variadic)
sz += 5; 
char *argbuf = new char[sz];
size_t len = 0;
for (unsigned i = 0; i< m_param_types.length (); i++)
{
strcpy (argbuf + len, m_param_types[i]->get_debug_string ());
len += strlen (m_param_types[i]->get_debug_string ());
if (i + 1 < m_param_types.length ())
{
strcpy (argbuf + len, ", ");
len += 2;
}
}
if (m_is_variadic)
{
if (m_param_types.length ())
{
strcpy (argbuf + len, ", ");
len += 2;
}
strcpy (argbuf + len, "...");
len += 3;
}
argbuf[len] = '\0';
string *result = string::from_printf (m_ctxt,
"%s %s(%s)",
m_return_type->get_debug_string (),
insert,
argbuf);
delete[] argbuf;
return result;
}
void
recording::function_type::write_reproducer (reproducer &)
{
}
void
recording::function_type::write_deferred_reproducer (reproducer &r,
memento *ptr_type)
{
gcc_assert (ptr_type);
r.make_identifier (this, "function_type");
const char *ptr_id = r.make_identifier (ptr_type, "ptr_to");
const char *param_types_id = r.make_tmp_identifier ("params_for", this);
r.write ("  gcc_jit_type *%s[%i] = {\n",
param_types_id,
m_param_types.length ());
int i;
type *param_type;
FOR_EACH_VEC_ELT (m_param_types, i, param_type)
r.write ("    %s,\n", r.get_identifier_as_type (param_type));
r.write ("  };\n");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_context_new_function_ptr_type (%s, \n"
"                                           %s, \n"
"                                           %s, \n"
"                                           %i, \n"
"                                           %s, \n"
"                                           %i); \n",
ptr_id,
r.get_identifier (get_context ()),
"NULL", 
r.get_identifier_as_type (m_return_type),
m_param_types.length (),
param_types_id,
m_is_variadic);
}
void
recording::field::replay_into (replayer *r)
{
set_playback_obj (r->new_field (playback_location (r, m_loc),
m_type->playback_type (),
playback_string (m_name)));
}
void
recording::field::write_to_dump (dump &d)
{
d.write ("  %s %s;\n",
m_type->get_debug_string (),
m_name->c_str ());
}
recording::string *
recording::field::make_debug_string ()
{
return m_name;
}
void
recording::field::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "field");
r.write("  gcc_jit_field *%s =\n"
"    gcc_jit_context_new_field (%s,\n"
"                               %s, \n"
"                               %s, \n"
"                               %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_type (m_type),
m_name->get_debug_string ());
}
recording::compound_type::compound_type (context *ctxt,
location *loc,
string *name)
: type (ctxt),
m_loc (loc),
m_name (name),
m_fields (NULL)
{
}
void
recording::compound_type::set_fields (location *loc,
int num_fields,
field **field_array)
{
m_loc = loc;
gcc_assert (m_fields == NULL);
m_fields = new fields (this, num_fields, field_array);
m_ctxt->record (m_fields);
}
recording::type *
recording::compound_type::dereference ()
{
return NULL; 
}
recording::struct_::struct_ (context *ctxt,
location *loc,
string *name)
: compound_type (ctxt, loc, name)
{
}
void
recording::struct_::replay_into (replayer *r)
{
set_playback_obj (
r->new_compound_type (playback_location (r, get_loc ()),
get_name ()->c_str (),
true ));
}
const char *
recording::struct_::access_as_type (reproducer &r)
{
return r.xstrdup_printf ("gcc_jit_struct_as_type (%s)",
r.get_identifier (this));
}
recording::string *
recording::struct_::make_debug_string ()
{
return string::from_printf (m_ctxt,
"struct %s", get_name ()->c_str ());
}
void
recording::struct_::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "struct");
r.write ("  gcc_jit_struct *%s =\n"
"    gcc_jit_context_new_opaque_struct (%s,\n"
"                                       %s, \n"
"                                       %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (get_loc ()),
get_name ()->get_debug_string ());
}
recording::union_::union_ (context *ctxt,
location *loc,
string *name)
: compound_type (ctxt, loc, name)
{
}
void
recording::union_::replay_into (replayer *r)
{
set_playback_obj (
r->new_compound_type (playback_location (r, get_loc ()),
get_name ()->c_str (),
false ));
}
recording::string *
recording::union_::make_debug_string ()
{
return string::from_printf (m_ctxt,
"union %s", get_name ()->c_str ());
}
void
recording::union_::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "union");
const char *fields_id = r.make_tmp_identifier ("fields_for", this);
r.write ("  gcc_jit_field *%s[%i] = {\n",
fields_id,
get_fields ()->length ());
for (int i = 0; i < get_fields ()->length (); i++)
r.write ("    %s,\n", r.get_identifier (get_fields ()->get_field (i)));
r.write ("  };\n");
r.write ("  gcc_jit_type *%s =\n"
"    gcc_jit_context_new_union_type (%s,\n"
"                                    %s, \n"
"                                    %s, \n"
"                                    %i, \n"
"                                    %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (get_loc ()),
get_name ()->get_debug_string (),
get_fields ()->length (),
fields_id);
}
recording::fields::fields (compound_type *struct_or_union,
int num_fields,
field **fields)
: memento (struct_or_union->m_ctxt),
m_struct_or_union (struct_or_union),
m_fields ()
{
for (int i = 0; i < num_fields; i++)
{
gcc_assert (fields[i]->get_container () == NULL);
fields[i]->set_container (m_struct_or_union);
m_fields.safe_push (fields[i]);
}
}
void
recording::fields::replay_into (replayer *)
{
auto_vec<playback::field *> playback_fields;
playback_fields.create (m_fields.length ());
for (unsigned i = 0; i < m_fields.length (); i++)
playback_fields.safe_push (m_fields[i]->playback_field ());
m_struct_or_union->playback_compound_type ()->set_fields (&playback_fields);
}
void
recording::fields::write_to_dump (dump &d)
{
int i;
field *f;
d.write ("%s\n{\n", m_struct_or_union->get_debug_string ());
FOR_EACH_VEC_ELT (m_fields, i, f)
f->write_to_dump (d);
d.write ("};\n");
}
void
recording::fields::write_reproducer (reproducer &r)
{
if (m_struct_or_union)
if (m_struct_or_union->dyn_cast_struct () == NULL)
return;
const char *fields_id = r.make_identifier (this, "fields");
r.write ("  gcc_jit_field *%s[%i] = {\n",
fields_id,
m_fields.length ());
int i;
field *field;
FOR_EACH_VEC_ELT (m_fields, i, field)
r.write ("    %s,\n", r.get_identifier (field));
r.write ("  };\n");
r.write ("  gcc_jit_struct_set_fields (%s, \n"
"                             %s, \n"
"                             %i, \n"
"                             %s); \n",
r.get_identifier (m_struct_or_union),
r.get_identifier ((memento *)NULL),
m_fields.length (),
fields_id);
}
recording::string *
recording::fields::make_debug_string ()
{
return string::from_printf (m_ctxt,
"fields");
}
recording::rvalue *
recording::rvalue::access_field (recording::location *loc,
field *field)
{
recording::rvalue *result =
new access_field_rvalue (m_ctxt, loc, this, field);
m_ctxt->record (result);
return result;
}
recording::lvalue *
recording::rvalue::dereference_field (recording::location *loc,
field *field)
{
recording::lvalue *result =
new dereference_field_rvalue (m_ctxt, loc, this, field);
m_ctxt->record (result);
return result;
}
recording::lvalue *
recording::rvalue::dereference (recording::location *loc)
{
recording::lvalue *result =
new dereference_rvalue (m_ctxt, loc, this);
m_ctxt->record (result);
return result;
}
class rvalue_usage_validator : public recording::rvalue_visitor
{
public:
rvalue_usage_validator (const char *api_funcname,
recording::context *ctxt,
recording::statement *stmt);
void
visit (recording::rvalue *rvalue) FINAL OVERRIDE;
private:
const char *m_api_funcname;
recording::context *m_ctxt;
recording::statement *m_stmt;
};
rvalue_usage_validator::rvalue_usage_validator (const char *api_funcname,
recording::context *ctxt,
recording::statement *stmt)
: m_api_funcname (api_funcname),
m_ctxt (ctxt),
m_stmt (stmt)
{
}
void
rvalue_usage_validator::visit (recording::rvalue *rvalue)
{
gcc_assert (m_stmt->get_block ());
recording::function *stmt_scope = m_stmt->get_block ()->get_function ();
if (rvalue->get_scope ())
{
if (rvalue->get_scope () != stmt_scope)
m_ctxt->add_error
(rvalue->get_loc (),
"%s:"
" rvalue %s (type: %s)"
" has scope limited to function %s"
" but was used within function %s"
" (in statement: %s)",
m_api_funcname,
rvalue->get_debug_string (),
rvalue->get_type ()->get_debug_string (),
rvalue->get_scope ()->get_debug_string (),
stmt_scope->get_debug_string (),
m_stmt->get_debug_string ());
}
else
{
if (rvalue->dyn_cast_param ())
m_ctxt->add_error
(rvalue->get_loc (),
"%s:"
" param %s (type: %s)"
" was used within function %s"
" (in statement: %s)"
" but is not associated with any function",
m_api_funcname,
rvalue->get_debug_string (),
rvalue->get_type ()->get_debug_string (),
stmt_scope->get_debug_string (),
m_stmt->get_debug_string ());
}
}
void
recording::rvalue::verify_valid_within_stmt (const char *api_funcname, statement *s)
{
rvalue_usage_validator v (api_funcname,
s->get_context (),
s);
v.visit (this);
visit_children (&v);
}
void
recording::rvalue::set_scope (function *scope)
{
gcc_assert (scope);
gcc_assert (m_scope == NULL);
m_scope = scope;
}
const char *
recording::rvalue::access_as_rvalue (reproducer &r)
{
return r.get_identifier (this);
}
const char *
recording::rvalue::get_debug_string_parens (enum precedence outer_prec)
{
enum precedence this_prec = get_precedence ();
if (this_prec <= outer_prec)
return get_debug_string();
if (!m_parenthesized_string)
{
const char *debug_string = get_debug_string ();
m_parenthesized_string = string::from_printf (get_context (),
"(%s)",
debug_string);
}
gcc_assert (m_parenthesized_string);
return m_parenthesized_string->c_str ();
}
recording::lvalue *
recording::lvalue::access_field (recording::location *loc,
field *field)
{
recording::lvalue *result =
new access_field_of_lvalue (m_ctxt, loc, this, field);
m_ctxt->record (result);
return result;
}
const char *
recording::lvalue::access_as_rvalue (reproducer &r)
{
return r.xstrdup_printf ("gcc_jit_lvalue_as_rvalue (%s)",
r.get_identifier (this));
}
const char *
recording::lvalue::access_as_lvalue (reproducer &r)
{
return r.get_identifier (this);
}
recording::rvalue *
recording::lvalue::get_address (recording::location *loc)
{
recording::rvalue *result =
new get_address_of_lvalue (m_ctxt, loc, this);
m_ctxt->record (result);
return result;
}
void
recording::param::replay_into (replayer *r)
{
set_playback_obj (r->new_param (playback_location (r, m_loc),
m_type->playback_type (),
m_name->c_str ()));
}
const char *
recording::param::access_as_rvalue (reproducer &r)
{
return r.xstrdup_printf ("gcc_jit_param_as_rvalue (%s)",
r.get_identifier (this));
}
const char *
recording::param::access_as_lvalue (reproducer &r)
{
return r.xstrdup_printf ("gcc_jit_param_as_lvalue (%s)",
r.get_identifier (this));
}
void
recording::param::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "param");
r.write ("  gcc_jit_param *%s =\n"
"    gcc_jit_context_new_param (%s,\n"
"                               %s, \n"
"                               %s, \n"
"                               %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_type (m_type),
m_name->get_debug_string ());
}
recording::function::function (context *ctxt,
recording::location *loc,
enum gcc_jit_function_kind kind,
type *return_type,
recording::string *name,
int num_params,
recording::param **params,
int is_variadic,
enum built_in_function builtin_id)
: memento (ctxt),
m_loc (loc),
m_kind (kind),
m_return_type (return_type),
m_name (name),
m_params (),
m_is_variadic (is_variadic),
m_builtin_id (builtin_id),
m_locals (),
m_blocks (),
m_fn_ptr_type (NULL)
{
for (int i = 0; i< num_params; i++)
{
param *param = params[i];
gcc_assert (param);
if (param->get_scope ())
{
gcc_assert (this == param->get_scope ());
ctxt->add_error
(loc,
"gcc_jit_context_new_function:"
" parameter %s (type: %s)"
" is used more than once when creating function %s",
param->get_debug_string (),
param->get_type ()->get_debug_string (),
name->c_str ());
}
else
{
param->set_scope (this);
}
m_params.safe_push (param);
}
}
void
recording::function::replay_into (replayer *r)
{
auto_vec <playback::param *> params;
int i;
recording::param *param;
params.create (m_params.length ());
FOR_EACH_VEC_ELT (m_params, i, param)
params.safe_push (param->playback_param ());
set_playback_obj (r->new_function (playback_location (r, m_loc),
m_kind,
m_return_type->playback_type (),
m_name->c_str (),
&params,
m_is_variadic,
m_builtin_id));
}
recording::lvalue *
recording::function::new_local (recording::location *loc,
type *type,
const char *name)
{
local *result = new local (this, loc, type, new_string (name));
m_ctxt->record (result);
m_locals.safe_push (result);
return result;
}
recording::block*
recording::function::new_block (const char *name)
{
gcc_assert (m_kind != GCC_JIT_FUNCTION_IMPORTED);
recording::block *result =
new recording::block (this, m_blocks.length (), new_string (name));
m_ctxt->record (result);
m_blocks.safe_push (result);
return result;
}
void
recording::function::write_to_dump (dump &d)
{
switch (m_kind)
{
default: gcc_unreachable ();
case GCC_JIT_FUNCTION_EXPORTED:
case GCC_JIT_FUNCTION_IMPORTED:
d.write ("extern ");
break;
case GCC_JIT_FUNCTION_INTERNAL:
d.write ("static ");
break;
case GCC_JIT_FUNCTION_ALWAYS_INLINE:
d.write ("static inline ");
break;
}
d.write ("%s\n", m_return_type->get_debug_string ());
if (d.update_locations ())
m_loc = d.make_location ();
d.write ("%s (", get_debug_string ());
int i;
recording::param *param;
FOR_EACH_VEC_ELT (m_params, i, param)
{
if (i > 0)
d.write (", ");
d.write ("%s %s",
param->get_type ()->get_debug_string (),
param->get_debug_string ());
}
d.write (")");
if (m_kind == GCC_JIT_FUNCTION_IMPORTED)
{
d.write ("; \n\n");
}
else
{
int i;
local *var = NULL;
block *b;
d.write ("\n{\n");
FOR_EACH_VEC_ELT (m_locals, i, var)
var->write_to_dump (d);
if (m_locals.length ())
d.write ("\n");
FOR_EACH_VEC_ELT (m_blocks, i, b)
{
if (i > 0)
d.write ("\n");
b->write_to_dump (d);
}
d.write ("}\n\n");
}
}
void
recording::function::validate ()
{
if (m_kind != GCC_JIT_FUNCTION_IMPORTED
&& m_return_type != m_ctxt->get_type (GCC_JIT_TYPE_VOID))
if (m_blocks.length () == 0)
m_ctxt->add_error (m_loc,
"function %s returns non-void (type: %s)"
" but has no blocks",
get_debug_string (),
m_return_type->get_debug_string ());
int num_invalid_blocks = 0;
{
int i;
block *b;
FOR_EACH_VEC_ELT (m_blocks, i, b)
if (!b->validate ())
num_invalid_blocks++;
}
if (!m_ctxt->get_inner_bool_option
(INNER_BOOL_OPTION_ALLOW_UNREACHABLE_BLOCKS)
&& m_blocks.length () > 0 && num_invalid_blocks == 0)
{
auto_vec<block *> worklist (m_blocks.length ());
worklist.safe_push (m_blocks[0]);
while (worklist.length () > 0)
{
block *b = worklist.pop ();
b->m_is_reachable = true;
vec <block *> successors = b->get_successor_blocks ();
int i;
block *succ;
FOR_EACH_VEC_ELT (successors, i, succ)
if (!succ->m_is_reachable)
worklist.safe_push (succ);
successors.release ();
}
{
int i;
block *b;
FOR_EACH_VEC_ELT (m_blocks, i, b)
if (!b->m_is_reachable)
m_ctxt->add_error (b->get_loc (),
"unreachable block: %s",
b->get_debug_string ());
}
}
}
void
recording::function::dump_to_dot (const char *path)
{
FILE *fp  = fopen (path, "w");
if (!fp)
return;
pretty_printer the_pp;
the_pp.buffer->stream = fp;
pretty_printer *pp = &the_pp;
pp_printf (pp,
"digraph %s {\n", get_debug_string ());
{
int i;
block *b;
FOR_EACH_VEC_ELT (m_blocks, i, b)
b->dump_to_dot (pp);
}
{
int i;
block *b;
FOR_EACH_VEC_ELT (m_blocks, i, b)
b->dump_edges_to_dot (pp);
}
pp_printf (pp, "}\n");
pp_flush (pp);
fclose (fp);
}
recording::rvalue *
recording::function::get_address (recording::location *loc)
{
if (!m_fn_ptr_type)
{
auto_vec <recording::type *> param_types (m_params.length ());
unsigned i;
recording::param *param;
FOR_EACH_VEC_ELT (m_params, i, param)
param_types.safe_push (param->get_type ());
recording::function_type *fn_type
= m_ctxt->new_function_type (m_return_type,
m_params.length (),
param_types.address (),
m_is_variadic);
m_fn_ptr_type = fn_type->get_pointer ();
}
gcc_assert (m_fn_ptr_type);
rvalue *result = new function_pointer (get_context (), loc, this, m_fn_ptr_type);
m_ctxt->record (result);
return result;
}
recording::string *
recording::function::make_debug_string ()
{
return m_name;
}
static const char * const names_of_function_kinds[] = {
"GCC_JIT_FUNCTION_EXPORTED",
"GCC_JIT_FUNCTION_INTERNAL",
"GCC_JIT_FUNCTION_IMPORTED",
"GCC_JIT_FUNCTION_ALWAYS_INLINE"
};
void
recording::function::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "func");
if (m_builtin_id)
{
r.write ("  gcc_jit_function *%s =\n"
"    gcc_jit_context_get_builtin_function (%s,\n"
"                                          %s);\n",
id,
r.get_identifier (get_context ()),
m_name->get_debug_string ());
return;
}
const char *params_id = r.make_tmp_identifier ("params_for", this);
r.write ("  gcc_jit_param *%s[%i] = {\n",
params_id,
m_params.length ());
int i;
param *param;
FOR_EACH_VEC_ELT (m_params, i, param)
r.write ("    %s,\n", r.get_identifier (param));
r.write ("  };\n");
r.write ("  gcc_jit_function *%s =\n"
"    gcc_jit_context_new_function (%s, \n"
"                                  %s, \n"
"                                  %s, \n"
"                                  %s, \n"
"                                  %s, \n"
"                                  %i, \n"
"                                  %s, \n"
"                                  %i); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
names_of_function_kinds[m_kind],
r.get_identifier_as_type (m_return_type),
m_name->get_debug_string (),
m_params.length (),
params_id,
m_is_variadic);
}
recording::statement *
recording::block::add_eval (recording::location *loc,
recording::rvalue *rvalue)
{
statement *result = new eval (this, loc, rvalue);
m_ctxt->record (result);
m_statements.safe_push (result);
return result;
}
recording::statement *
recording::block::add_assignment (recording::location *loc,
recording::lvalue *lvalue,
recording::rvalue *rvalue)
{
statement *result = new assignment (this, loc, lvalue, rvalue);
m_ctxt->record (result);
m_statements.safe_push (result);
return result;
}
recording::statement *
recording::block::add_assignment_op (recording::location *loc,
recording::lvalue *lvalue,
enum gcc_jit_binary_op op,
recording::rvalue *rvalue)
{
statement *result = new assignment_op (this, loc, lvalue, op, rvalue);
m_ctxt->record (result);
m_statements.safe_push (result);
return result;
}
recording::statement *
recording::block::add_comment (recording::location *loc,
const char *text)
{
statement *result = new comment (this, loc, new_string (text));
m_ctxt->record (result);
m_statements.safe_push (result);
return result;
}
recording::statement *
recording::block::end_with_conditional (recording::location *loc,
recording::rvalue *boolval,
recording::block *on_true,
recording::block *on_false)
{
statement *result = new conditional (this, loc, boolval, on_true, on_false);
m_ctxt->record (result);
m_statements.safe_push (result);
m_has_been_terminated = true;
return result;
}
recording::statement *
recording::block::end_with_jump (recording::location *loc,
recording::block *target)
{
statement *result = new jump (this, loc, target);
m_ctxt->record (result);
m_statements.safe_push (result);
m_has_been_terminated = true;
return result;
}
recording::statement *
recording::block::end_with_return (recording::location *loc,
recording::rvalue *rvalue)
{
statement *result = new return_ (this, loc, rvalue);
m_ctxt->record (result);
m_statements.safe_push (result);
m_has_been_terminated = true;
return result;
}
recording::statement *
recording::block::end_with_switch (recording::location *loc,
recording::rvalue *expr,
recording::block *default_block,
int num_cases,
recording::case_ **cases)
{
statement *result = new switch_ (this, loc,
expr,
default_block,
num_cases,
cases);
m_ctxt->record (result);
m_statements.safe_push (result);
m_has_been_terminated = true;
return result;
}
void
recording::block::write_to_dump (dump &d)
{
d.write ("%s:\n", get_debug_string ());
int i;
statement *s;
FOR_EACH_VEC_ELT (m_statements, i, s)
s->write_to_dump (d);
}
bool
recording::block::validate ()
{
if (!has_been_terminated ())
{
statement *stmt = get_last_statement ();
location *loc = stmt ? stmt->get_loc () : NULL;
m_func->get_context ()->add_error (loc,
"unterminated block in %s: %s",
m_func->get_debug_string (),
get_debug_string ());
return false;
}
return true;
}
recording::location *
recording::block::get_loc () const
{
recording::statement *stmt = get_first_statement ();
if (stmt)
return stmt->get_loc ();
else
return NULL;
}
recording::statement *
recording::block::get_first_statement () const
{
if (m_statements.length ())
return m_statements[0];
else
return NULL;
}
recording::statement *
recording::block::get_last_statement () const
{
if (m_statements.length ())
return m_statements[m_statements.length () - 1];
else
return NULL;
}
vec <recording::block *>
recording::block::get_successor_blocks () const
{
gcc_assert (m_has_been_terminated);
statement *last_statement = get_last_statement ();
gcc_assert (last_statement);
return last_statement->get_successor_blocks ();
}
void
recording::block::replay_into (replayer *)
{
set_playback_obj (m_func->playback_function ()
->new_block (playback_string (m_name)));
}
recording::string *
recording::block::make_debug_string ()
{
if (m_name)
return m_name;
else
return string::from_printf (m_ctxt,
"<UNNAMED BLOCK %p>",
(void *)this);
}
void
recording::block::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "block");
r.write ("  gcc_jit_block *%s =\n"
"    gcc_jit_function_new_block (%s, %s);\n",
id,
r.get_identifier (m_func),
m_name ? m_name->get_debug_string () : "NULL");
}
void
recording::block::dump_to_dot (pretty_printer *pp)
{
pp_printf (pp,
("\tblock_%d "
"[shape=record,style=filled,fillcolor=white,label=\"{"),
m_index);
pp_write_text_to_stream (pp);
if (m_name)
{
pp_string (pp, m_name->c_str ());
pp_string (pp, ":");
pp_newline (pp);
pp_write_text_as_dot_label_to_stream (pp, true );
}
int i;
statement *s;
FOR_EACH_VEC_ELT (m_statements, i, s)
{
pp_string (pp, s->get_debug_string ());
pp_newline (pp);
pp_write_text_as_dot_label_to_stream (pp, true );
}
pp_printf (pp,
"}\"];\n\n");
pp_flush (pp);
}
void
recording::block::dump_edges_to_dot (pretty_printer *pp)
{
vec <block *> successors = get_successor_blocks ();
int i;
block *succ;
FOR_EACH_VEC_ELT (successors, i, succ)
pp_printf (pp,
"\tblock_%d:s -> block_%d:n;\n",
m_index, succ->m_index);
successors.release ();
}
void
recording::global::replay_into (replayer *r)
{
set_playback_obj (r->new_global (playback_location (r, m_loc),
m_kind,
m_type->playback_type (),
playback_string (m_name)));
}
void
recording::global::write_to_dump (dump &d)
{
if (d.update_locations ())
m_loc = d.make_location ();
switch (m_kind)
{
default:
gcc_unreachable ();
case GCC_JIT_GLOBAL_EXPORTED:
break;
case GCC_JIT_GLOBAL_INTERNAL:
d.write ("static ");
break;
case GCC_JIT_GLOBAL_IMPORTED:
d.write ("extern ");
break;
}
d.write ("%s %s;\n",
m_type->get_debug_string (),
get_debug_string ());
}
static const char * const global_kind_reproducer_strings[] = {
"GCC_JIT_GLOBAL_EXPORTED",
"GCC_JIT_GLOBAL_INTERNAL",
"GCC_JIT_GLOBAL_IMPORTED"
};
void
recording::global::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "block");
r.write ("  gcc_jit_lvalue *%s =\n"
"    gcc_jit_context_new_global (%s, \n"
"                                %s, \n"
"                                %s, \n"
"                                %s, \n"
"                                %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
global_kind_reproducer_strings[m_kind],
r.get_identifier_as_type (get_type ()),
m_name->get_debug_string ());
}
template class recording::memento_of_new_rvalue_from_const <int>;
template class recording::memento_of_new_rvalue_from_const <long>;
template class recording::memento_of_new_rvalue_from_const <double>;
template class recording::memento_of_new_rvalue_from_const <void *>;
template <typename HOST_TYPE>
void
recording::
memento_of_new_rvalue_from_const <HOST_TYPE>::replay_into (replayer *r)
{
set_playback_obj
(r->new_rvalue_from_const <HOST_TYPE> (m_type->playback_type (),
m_value));
}
namespace recording
{
template <>
string *
memento_of_new_rvalue_from_const <int>::make_debug_string ()
{
return string::from_printf (m_ctxt,
"(%s)%i",
m_type->get_debug_string (),
m_value);
}
template <>
bool
memento_of_new_rvalue_from_const <int>::get_wide_int (wide_int *out) const
{
*out = wi::shwi (m_value, sizeof (m_value) * 8);
return true;
}
template <>
void
memento_of_new_rvalue_from_const <int>::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_int (%s, \n"
"                                         %s, \n"
"                                         %i); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type),
m_value);
}
template <>
string *
memento_of_new_rvalue_from_const <long>::make_debug_string ()
{
return string::from_printf (m_ctxt,
"(%s)%li",
m_type->get_debug_string (),
m_value);
}
template <>
bool
memento_of_new_rvalue_from_const <long>::get_wide_int (wide_int *out) const
{
*out = wi::shwi (m_value, sizeof (m_value) * 8);
return true;
}
template <>
void
recording::memento_of_new_rvalue_from_const <long>::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
if (m_value == LONG_MIN)
{
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_long (%s, \n"
"                                          %s, \n"
"                                          %ldL - 1); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type),
m_value + 1);
return;
}
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_long (%s, \n"
"                                          %s, \n"
"                                          %ldL); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type),
m_value);
}
template <>
string *
memento_of_new_rvalue_from_const <double>::make_debug_string ()
{
return string::from_printf (m_ctxt,
"(%s)%f",
m_type->get_debug_string (),
m_value);
}
template <>
bool
memento_of_new_rvalue_from_const <double>::get_wide_int (wide_int *) const
{
return false;
}
template <>
void
recording::memento_of_new_rvalue_from_const <double>::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_double (%s, \n"
"                                            %s, \n"
"                                            %f); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type),
m_value);
}
template <>
string *
memento_of_new_rvalue_from_const <void *>::make_debug_string ()
{
if (m_value != NULL)
return string::from_printf (m_ctxt,
"(%s)%p",
m_type->get_debug_string (), m_value);
else
return string::from_printf (m_ctxt,
"(%s)NULL",
m_type->get_debug_string ());
}
template <>
bool
memento_of_new_rvalue_from_const <void *>::get_wide_int (wide_int *) const
{
return false;
}
template <>
void
memento_of_new_rvalue_from_const <void *>::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
if (m_value)
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_ptr (%s, \n"
"                                         %s, \n"
"                                         (void *)%p); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type),
m_value);
else
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_null (%s, \n"
"                          %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier_as_type (m_type));
}
} 
void
recording::memento_of_new_string_literal::replay_into (replayer *r)
{
set_playback_obj (r->new_string_literal (m_value->c_str ()));
}
recording::string *
recording::memento_of_new_string_literal::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s",
m_value->get_debug_string ());
}
void
recording::memento_of_new_string_literal::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_string_literal (%s, \n"
"                                        %s); \n",
id,
r.get_identifier (get_context ()),
m_value->get_debug_string ());
}
recording::memento_of_new_rvalue_from_vector::
memento_of_new_rvalue_from_vector (context *ctxt,
location *loc,
vector_type *type,
rvalue **elements)
: rvalue (ctxt, loc, type),
m_vector_type (type),
m_elements ()
{
for (unsigned i = 0; i < type->get_num_units (); i++)
m_elements.safe_push (elements[i]);
}
void
recording::memento_of_new_rvalue_from_vector::replay_into (replayer *r)
{
auto_vec<playback::rvalue *> playback_elements;
playback_elements.create (m_elements.length ());
for (unsigned i = 0; i< m_elements.length (); i++)
playback_elements.safe_push (m_elements[i]->playback_rvalue ());
set_playback_obj (r->new_rvalue_from_vector (playback_location (r, m_loc),
m_type->playback_type (),
playback_elements));
}
void
recording::memento_of_new_rvalue_from_vector::visit_children (rvalue_visitor *v)
{
for (unsigned i = 0; i< m_elements.length (); i++)
v->visit (m_elements[i]);
}
recording::string *
recording::memento_of_new_rvalue_from_vector::make_debug_string ()
{
comma_separated_string elements (m_elements, get_precedence ());
string *result = string::from_printf (m_ctxt,
"{%s}",
elements.as_char_ptr ());
return result;
}
void
recording::memento_of_new_rvalue_from_vector::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "vector");
const char *elements_id = r.make_tmp_identifier ("elements_for_", this);
r.write ("  gcc_jit_rvalue *%s[%i] = {\n",
elements_id,
m_elements.length ());
for (unsigned i = 0; i< m_elements.length (); i++)
r.write ("    %s,\n", r.get_identifier_as_rvalue (m_elements[i]));
r.write ("  };\n");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_rvalue_from_vector (%s, \n"
"                                            %s, \n"
"                                            %s, \n"
"                                            %i,  \n"
"                                            %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier (m_vector_type),
m_elements.length (),
elements_id);
}
void
recording::unary_op::replay_into (replayer *r)
{
set_playback_obj (r->new_unary_op (playback_location (r, m_loc),
m_op,
get_type ()->playback_type (),
m_a->playback_rvalue ()));
}
void
recording::unary_op::visit_children (rvalue_visitor *v)
{
v->visit (m_a);
}
static const char * const unary_op_strings[] = {
"-", 
"~", 
"!", 
"abs ", 
};
recording::string *
recording::unary_op::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s(%s)",
unary_op_strings[m_op],
m_a->get_debug_string ());
}
static const char * const unary_op_reproducer_strings[] = {
"GCC_JIT_UNARY_OP_MINUS",
"GCC_JIT_UNARY_OP_BITWISE_NEGATE",
"GCC_JIT_UNARY_OP_LOGICAL_NEGATE",
"GCC_JIT_UNARY_OP_ABS"
};
void
recording::unary_op::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_unary_op (%s,\n"
"                                  %s, \n"
"                                  %s, \n"
"                                  %s, \n"
"                                  %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
unary_op_reproducer_strings[m_op],
r.get_identifier_as_type (get_type ()),
r.get_identifier_as_rvalue (m_a));
}
void
recording::binary_op::replay_into (replayer *r)
{
set_playback_obj (r->new_binary_op (playback_location (r, m_loc),
m_op,
get_type ()->playback_type (),
m_a->playback_rvalue (),
m_b->playback_rvalue ()));
}
void
recording::binary_op::visit_children (rvalue_visitor *v)
{
v->visit (m_a);
v->visit (m_b);
}
static const char * const binary_op_strings[] = {
"+", 
"-", 
"*", 
"/", 
"%", 
"&", 
"^", 
"|", 
"&&", 
"||", 
"<<", 
">>", 
};
recording::string *
recording::binary_op::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s %s %s",
m_a->get_debug_string_parens (prec),
binary_op_strings[m_op],
m_b->get_debug_string_parens (prec));
}
static const char * const binary_op_reproducer_strings[] = {
"GCC_JIT_BINARY_OP_PLUS",
"GCC_JIT_BINARY_OP_MINUS",
"GCC_JIT_BINARY_OP_MULT",
"GCC_JIT_BINARY_OP_DIVIDE",
"GCC_JIT_BINARY_OP_MODULO",
"GCC_JIT_BINARY_OP_BITWISE_AND",
"GCC_JIT_BINARY_OP_BITWISE_XOR",
"GCC_JIT_BINARY_OP_BITWISE_OR",
"GCC_JIT_BINARY_OP_LOGICAL_AND",
"GCC_JIT_BINARY_OP_LOGICAL_OR",
"GCC_JIT_BINARY_OP_LSHIFT",
"GCC_JIT_BINARY_OP_RSHIFT"
};
void
recording::binary_op::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_binary_op (%s,\n"
"                                   %s, \n"
"                                   %s, \n"
"                                   %s, \n"
"                                   %s, \n"
"                                   %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
binary_op_reproducer_strings[m_op],
r.get_identifier_as_type (get_type ()),
r.get_identifier_as_rvalue (m_a),
r.get_identifier_as_rvalue (m_b));
}
namespace recording {
static const enum precedence binary_op_precedence[] = {
PRECEDENCE_ADDITIVE, 
PRECEDENCE_ADDITIVE, 
PRECEDENCE_MULTIPLICATIVE, 
PRECEDENCE_MULTIPLICATIVE, 
PRECEDENCE_MULTIPLICATIVE, 
PRECEDENCE_BITWISE_AND, 
PRECEDENCE_BITWISE_XOR, 
PRECEDENCE_BITWISE_IOR, 
PRECEDENCE_LOGICAL_AND, 
PRECEDENCE_LOGICAL_OR, 
PRECEDENCE_SHIFT, 
PRECEDENCE_SHIFT, 
};
} 
enum recording::precedence
recording::binary_op::get_precedence () const
{
return binary_op_precedence[m_op];
}
static const char * const comparison_strings[] =
{
"==", 
"!=", 
"<",  
"<=", 
">",  
">=", 
};
recording::string *
recording::comparison::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s %s %s",
m_a->get_debug_string_parens (prec),
comparison_strings[m_op],
m_b->get_debug_string_parens (prec));
}
static const char * const comparison_reproducer_strings[] =
{
"GCC_JIT_COMPARISON_EQ",
"GCC_JIT_COMPARISON_NE",
"GCC_JIT_COMPARISON_LT",
"GCC_JIT_COMPARISON_LE",
"GCC_JIT_COMPARISON_GT",
"GCC_JIT_COMPARISON_GE"
};
void
recording::comparison::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_comparison (%s,\n"
"                                    %s, \n"
"                                    %s, \n"
"                                    %s, \n"
"                                    %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
comparison_reproducer_strings[m_op],
r.get_identifier_as_rvalue (m_a),
r.get_identifier_as_rvalue (m_b));
}
void
recording::comparison::replay_into (replayer *r)
{
set_playback_obj (r->new_comparison (playback_location (r, m_loc),
m_op,
m_a->playback_rvalue (),
m_b->playback_rvalue ()));
}
void
recording::comparison::visit_children (rvalue_visitor *v)
{
v->visit (m_a);
v->visit (m_b);
}
namespace recording {
static const enum precedence comparison_precedence[] =
{
PRECEDENCE_EQUALITY, 
PRECEDENCE_EQUALITY, 
PRECEDENCE_RELATIONAL,  
PRECEDENCE_RELATIONAL, 
PRECEDENCE_RELATIONAL,  
PRECEDENCE_RELATIONAL, 
};
} 
enum recording::precedence
recording::comparison::get_precedence () const
{
return comparison_precedence[m_op];
}
void
recording::cast::replay_into (replayer *r)
{
set_playback_obj (r->new_cast (playback_location (r, m_loc),
m_rvalue->playback_rvalue (),
get_type ()->playback_type ()));
}
void
recording::cast::visit_children (rvalue_visitor *v)
{
v->visit (m_rvalue);
}
recording::string *
recording::cast::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"(%s)%s",
get_type ()->get_debug_string (),
m_rvalue->get_debug_string_parens (prec));
}
void
recording::cast::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_cast (%s,\n"
"                              %s, \n"
"                              %s, \n"
"                              %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_rvalue (m_rvalue),
r.get_identifier_as_type (get_type ()));
}
recording::base_call::base_call (context *ctxt,
location *loc,
type *type_,
int numargs,
rvalue **args)
: rvalue (ctxt, loc, type_),
m_args (),
m_require_tail_call (0)
{
for (int i = 0; i< numargs; i++)
m_args.safe_push (args[i]);
}
void
recording::base_call::write_reproducer_tail_call (reproducer &r,
const char *id)
{
if (m_require_tail_call)
{
r.write ("  gcc_jit_rvalue_set_bool_require_tail_call (%s,  \n"
"                                             %i); \n",
id,
1);
}
}
recording::call::call (recording::context *ctxt,
recording::location *loc,
recording::function *func,
int numargs,
rvalue **args)
: base_call (ctxt, loc, func->get_return_type (), numargs, args),
m_func (func)
{
}
void
recording::call::replay_into (replayer *r)
{
auto_vec<playback::rvalue *> playback_args;
playback_args.create (m_args.length ());
for (unsigned i = 0; i< m_args.length (); i++)
playback_args.safe_push (m_args[i]->playback_rvalue ());
set_playback_obj (r->new_call (playback_location (r, m_loc),
m_func->playback_function (),
&playback_args,
m_require_tail_call));
}
void
recording::call::visit_children (rvalue_visitor *v)
{
for (unsigned i = 0; i< m_args.length (); i++)
v->visit (m_args[i]);
}
recording::string *
recording::call::make_debug_string ()
{
comma_separated_string args (m_args, get_precedence ());
string *result = string::from_printf (m_ctxt,
"%s (%s)",
m_func->get_debug_string (),
args.as_char_ptr ());
return result;
}
void
recording::call::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "call");
const char *args_id = r.make_tmp_identifier ("args_for_", this);
r.write ("  gcc_jit_rvalue *%s[%i] = {\n",
args_id,
m_args.length ());
for (unsigned i = 0; i< m_args.length (); i++)
r.write ("    %s,\n", r.get_identifier_as_rvalue (m_args[i]));
r.write ("  };\n");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_call (%s, \n"
"                              %s, \n"
"                              %s, \n"
"                              %i,  \n"
"                              %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier (m_func),
m_args.length (),
args_id);
write_reproducer_tail_call (r, id);
}
recording::call_through_ptr::call_through_ptr (recording::context *ctxt,
recording::location *loc,
recording::rvalue *fn_ptr,
int numargs,
rvalue **args)
: base_call (ctxt, loc,
fn_ptr->get_type ()->dereference ()
->as_a_function_type ()->get_return_type (),
numargs, args),
m_fn_ptr (fn_ptr)
{
}
void
recording::call_through_ptr::replay_into (replayer *r)
{
auto_vec<playback::rvalue *> playback_args;
playback_args.create (m_args.length ());
for (unsigned i = 0; i< m_args.length (); i++)
playback_args.safe_push (m_args[i]->playback_rvalue ());
set_playback_obj (r->new_call_through_ptr (playback_location (r, m_loc),
m_fn_ptr->playback_rvalue (),
&playback_args,
m_require_tail_call));
}
void
recording::call_through_ptr::visit_children (rvalue_visitor *v)
{
v->visit (m_fn_ptr);
for (unsigned i = 0; i< m_args.length (); i++)
v->visit (m_args[i]);
}
recording::string *
recording::call_through_ptr::make_debug_string ()
{
enum precedence prec = get_precedence ();
size_t sz = 1; 
for (unsigned i = 0; i< m_args.length (); i++)
{
sz += strlen (m_args[i]->get_debug_string_parens (prec));
sz += 2; 
}
char *argbuf = new char[sz];
size_t len = 0;
for (unsigned i = 0; i< m_args.length (); i++)
{
strcpy (argbuf + len, m_args[i]->get_debug_string_parens (prec));
len += strlen (m_args[i]->get_debug_string_parens (prec));
if (i + 1 < m_args.length ())
{
strcpy (argbuf + len, ", ");
len += 2;
}
}
argbuf[len] = '\0';
string *result = string::from_printf (m_ctxt,
"%s (%s)",
m_fn_ptr->get_debug_string_parens (prec),
argbuf);
delete[] argbuf;
return result;
}
void
recording::call_through_ptr::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "call");
const char *args_id = r.make_tmp_identifier ("args_for_", this);
r.write ("  gcc_jit_rvalue *%s[%i] = {\n",
args_id,
m_args.length ());
for (unsigned i = 0; i< m_args.length (); i++)
r.write ("    %s,\n", r.get_identifier_as_rvalue (m_args[i]));
r.write ("  };\n");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_context_new_call_through_ptr (%s, \n"
"                              %s, \n"
"                              %s, \n"
"                              %i,  \n"
"                              %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_rvalue (m_fn_ptr),
m_args.length (),
args_id);
write_reproducer_tail_call (r, id);
}
void
recording::array_access::replay_into (replayer *r)
{
set_playback_obj (
r->new_array_access (playback_location (r, m_loc),
m_ptr->playback_rvalue (),
m_index->playback_rvalue ()));
}
void
recording::array_access::visit_children (rvalue_visitor *v)
{
v->visit (m_ptr);
v->visit (m_index);
}
recording::string *
recording::array_access::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s[%s]",
m_ptr->get_debug_string_parens (prec),
m_index->get_debug_string_parens (prec));
}
void
recording::array_access::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "lvalue");
r.write ("  gcc_jit_lvalue *%s = \n"
"    gcc_jit_context_new_array_access (%s, \n"
"                                      %s, \n"
"                                      %s, \n"
"                                      %s); \n",
id,
r.get_identifier (get_context ()),
r.get_identifier (m_loc),
r.get_identifier_as_rvalue (m_ptr),
r.get_identifier_as_rvalue (m_index));
}
void
recording::access_field_of_lvalue::replay_into (replayer *r)
{
set_playback_obj (
m_lvalue->playback_lvalue ()
->access_field (playback_location (r, m_loc),
m_field->playback_field ()));
}
void
recording::access_field_of_lvalue::visit_children (rvalue_visitor *v)
{
v->visit (m_lvalue);
}
recording::string *
recording::access_field_of_lvalue::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s.%s",
m_lvalue->get_debug_string_parens (prec),
m_field->get_debug_string ());
}
void
recording::access_field_of_lvalue::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "lvalue");
r.write ("  gcc_jit_lvalue *%s = \n"
"    gcc_jit_lvalue_access_field (%s, \n"
"                                 %s, \n"
"                                 %s);\n",
id,
r.get_identifier_as_lvalue (m_lvalue),
r.get_identifier (m_loc),
r.get_identifier (m_field));
}
void
recording::access_field_rvalue::replay_into (replayer *r)
{
set_playback_obj (
m_rvalue->playback_rvalue ()
->access_field (playback_location (r, m_loc),
m_field->playback_field ()));
}
void
recording::access_field_rvalue::visit_children (rvalue_visitor *v)
{
v->visit (m_rvalue);
}
recording::string *
recording::access_field_rvalue::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s.%s",
m_rvalue->get_debug_string_parens (prec),
m_field->get_debug_string ());
}
void
recording::access_field_rvalue::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "rvalue");
r.write ("  gcc_jit_rvalue *%s = \n"
"    gcc_jit_rvalue_access_field (%s, \n"
"                                 %s, \n"
"                                 %s);\n",
id,
r.get_identifier_as_rvalue (m_rvalue),
r.get_identifier (m_loc),
r.get_identifier (m_field));
}
void
recording::dereference_field_rvalue::replay_into (replayer *r)
{
set_playback_obj (
m_rvalue->playback_rvalue ()->
dereference_field (playback_location (r, m_loc),
m_field->playback_field ()));
}
void
recording::dereference_field_rvalue::visit_children (rvalue_visitor *v)
{
v->visit (m_rvalue);
}
recording::string *
recording::dereference_field_rvalue::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"%s->%s",
m_rvalue->get_debug_string_parens (prec),
m_field->get_debug_string ());
}
void
recording::dereference_field_rvalue::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "lvalue");
r.write ("  gcc_jit_lvalue *%s=\n"
"    gcc_jit_rvalue_dereference_field (%s, \n"
"                                      %s, \n"
"                                      %s); \n",
id,
r.get_identifier_as_rvalue (m_rvalue),
r.get_identifier (m_loc),
r.get_identifier (m_field));
}
void
recording::dereference_rvalue::replay_into (replayer *r)
{
set_playback_obj (
m_rvalue->playback_rvalue ()->
dereference (playback_location (r, m_loc)));
}
void
recording::dereference_rvalue::visit_children (rvalue_visitor *v)
{
v->visit (m_rvalue);
}
recording::string *
recording::dereference_rvalue::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"*%s",
m_rvalue->get_debug_string_parens (prec));
}
void
recording::dereference_rvalue::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "dereference");
r.write ("  gcc_jit_lvalue *%s =\n"
"    gcc_jit_rvalue_dereference (%s, \n"
"                                %s); \n",
id,
r.get_identifier_as_rvalue (m_rvalue),
r.get_identifier (m_loc));
}
void
recording::get_address_of_lvalue::replay_into (replayer *r)
{
set_playback_obj (
m_lvalue->playback_lvalue ()->
get_address (playback_location (r, m_loc)));
}
void
recording::get_address_of_lvalue::visit_children (rvalue_visitor *v)
{
v->visit (m_lvalue);
}
recording::string *
recording::get_address_of_lvalue::make_debug_string ()
{
enum precedence prec = get_precedence ();
return string::from_printf (m_ctxt,
"&%s",
m_lvalue->get_debug_string_parens (prec));
}
void
recording::get_address_of_lvalue::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "address_of");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_lvalue_get_address (%s, \n"
"                                %s); \n",
id,
r.get_identifier_as_lvalue (m_lvalue),
r.get_identifier (m_loc));
}
void
recording::function_pointer::replay_into (replayer *r)
{
set_playback_obj (
m_fn->playback_function ()->
get_address (playback_location (r, m_loc)));
}
void
recording::function_pointer::visit_children (rvalue_visitor *)
{
}
recording::string *
recording::function_pointer::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s",
m_fn->get_debug_string ());
}
void
recording::function_pointer::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "address_of");
r.write ("  gcc_jit_rvalue *%s =\n"
"    gcc_jit_function_get_address (%s, \n"
"                                  %s); \n",
id,
r.get_identifier (m_fn),
r.get_identifier (m_loc));
}
void
recording::local::replay_into (replayer *r)
{
set_playback_obj (
m_func->playback_function ()
->new_local (playback_location (r, m_loc),
m_type->playback_type (),
playback_string (m_name)));
}
void
recording::local::write_to_dump (dump &d)
{
if (d.update_locations ())
m_loc = d.make_location ();
d.write("  %s %s;\n",
m_type->get_debug_string (),
get_debug_string ());
}
void
recording::local::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "local");
r.write ("  gcc_jit_lvalue *%s =\n"
"    gcc_jit_function_new_local (%s, \n"
"                                %s, \n"
"                                %s, \n"
"                                %s); \n",
id,
r.get_identifier (m_func),
r.get_identifier (m_loc),
r.get_identifier_as_type (m_type),
m_name->get_debug_string ());
}
vec <recording::block *>
recording::statement::get_successor_blocks () const
{
gcc_unreachable ();
vec <block *> result;
result.create (0);
return result;
}
void
recording::statement::write_to_dump (dump &d)
{
memento::write_to_dump (d);
if (d.update_locations ())
m_loc = d.make_location ();
}
void
recording::eval::replay_into (replayer *r)
{
playback_block (get_block ())
->add_eval (playback_location (r),
m_rvalue->playback_rvalue ());
}
recording::string *
recording::eval::make_debug_string ()
{
return string::from_printf (m_ctxt,
"(void)%s;",
m_rvalue->get_debug_string ());
}
void
recording::eval::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_add_eval (%s, \n"
"                          %s, \n"
"                          %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_rvalue (m_rvalue));
}
void
recording::assignment::replay_into (replayer *r)
{
playback_block (get_block ())
->add_assignment (playback_location (r),
m_lvalue->playback_lvalue (),
m_rvalue->playback_rvalue ());
}
recording::string *
recording::assignment::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s = %s;",
m_lvalue->get_debug_string (),
m_rvalue->get_debug_string ());
}
void
recording::assignment::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_add_assignment (%s, \n"
"                                %s, \n"
"                                %s, \n"
"                                %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_lvalue (m_lvalue),
r.get_identifier_as_rvalue (m_rvalue));
}
void
recording::assignment_op::replay_into (replayer *r)
{
playback::type *result_type =
m_lvalue->playback_lvalue ()->get_type ();
playback::rvalue *binary_op =
r->new_binary_op (playback_location (r),
m_op,
result_type,
m_lvalue->playback_rvalue (),
m_rvalue->playback_rvalue ());
playback_block (get_block ())
->add_assignment (playback_location (r),
m_lvalue->playback_lvalue (),
binary_op);
}
recording::string *
recording::assignment_op::make_debug_string ()
{
return string::from_printf (m_ctxt,
"%s %s= %s;",
m_lvalue->get_debug_string (),
binary_op_strings[m_op],
m_rvalue->get_debug_string ());
}
void
recording::assignment_op::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_add_assignment_op (%s, \n"
"                                   %s, \n"
"                                   %s, \n"
"                                   %s, \n"
"                                   %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_lvalue (m_lvalue),
binary_op_reproducer_strings[m_op],
r.get_identifier_as_rvalue (m_rvalue));
}
void
recording::comment::replay_into (replayer *r)
{
playback_block (get_block ())
->add_comment (playback_location (r),
m_text->c_str ());
}
recording::string *
recording::comment::make_debug_string ()
{
return string::from_printf (m_ctxt,
"",
m_text->c_str ());
}
void
recording::comment::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_add_comment (%s, \n"
"                             %s, \n"
"                             %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
m_text->get_debug_string ());
}
void
recording::conditional::replay_into (replayer *r)
{
playback_block (get_block ())
->add_conditional (playback_location (r),
m_boolval->playback_rvalue (),
playback_block (m_on_true),
playback_block (m_on_false));
}
vec <recording::block *>
recording::conditional::get_successor_blocks () const
{
vec <block *> result;
result.create (2);
result.quick_push (m_on_true);
result.quick_push (m_on_false);
return result;
}
recording::string *
recording::conditional::make_debug_string ()
{
if (m_on_false)
return string::from_printf (m_ctxt,
"if (%s) goto %s; else goto %s;",
m_boolval->get_debug_string (),
m_on_true->get_debug_string (),
m_on_false->get_debug_string ());
else
return string::from_printf (m_ctxt,
"if (%s) goto %s;",
m_boolval->get_debug_string (),
m_on_true->get_debug_string ());
}
void
recording::conditional::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_end_with_conditional (%s, \n"
"                                      %s, \n"
"                                      %s, \n"
"                                      %s, \n"
"                                      %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_rvalue (m_boolval),
r.get_identifier (m_on_true),
r.get_identifier (m_on_false));
}
void
recording::jump::replay_into (replayer *r)
{
playback_block (get_block ())
->add_jump (playback_location (r),
m_target->playback_block ());
}
vec <recording::block *>
recording::jump::get_successor_blocks () const
{
vec <block *> result;
result.create (1);
result.quick_push (m_target);
return result;
}
recording::string *
recording::jump::make_debug_string ()
{
return string::from_printf (m_ctxt,
"goto %s;",
m_target->get_debug_string ());
}
void
recording::jump::write_reproducer (reproducer &r)
{
r.write ("  gcc_jit_block_end_with_jump (%s, \n"
"                               %s, \n"
"                               %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier (m_target));
}
void
recording::return_::replay_into (replayer *r)
{
playback_block (get_block ())
->add_return (playback_location (r),
m_rvalue ? m_rvalue->playback_rvalue () : NULL);
}
vec <recording::block *>
recording::return_::get_successor_blocks () const
{
vec <block *> result;
result.create (0);
return result;
}
recording::string *
recording::return_::make_debug_string ()
{
if (m_rvalue)
return string::from_printf (m_ctxt,
"return %s;",
m_rvalue->get_debug_string ());
else
return string::from_printf (m_ctxt,
"return;");
}
void
recording::return_::write_reproducer (reproducer &r)
{
if (m_rvalue)
r.write ("  gcc_jit_block_end_with_return (%s, \n"
"                                 %s, \n"
"                                 %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_rvalue (m_rvalue));
else
r.write ("  gcc_jit_block_end_with_void_return (%s, \n"
"                                      %s); \n",
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()));
}
void
recording::case_::write_reproducer (reproducer &r)
{
const char *id = r.make_identifier (this, "case");
const char *fmt =
"  gcc_jit_case *%s = \n"
"    gcc_jit_context_new_case (%s, \n"
"                              %s, \n"
"                              %s, \n"
"                              %s); \n";
r.write (fmt,
id,
r.get_identifier (get_context ()),
r.get_identifier_as_rvalue (m_min_value),
r.get_identifier_as_rvalue (m_max_value),
r.get_identifier (m_dest_block));
}
recording::string *
recording::case_::make_debug_string ()
{
return string::from_printf (get_context (),
"case %s ... %s: goto %s;",
m_min_value->get_debug_string (),
m_max_value->get_debug_string (),
m_dest_block->get_debug_string ());
}
recording::switch_::switch_ (block *b,
location *loc,
rvalue *expr,
block *default_block,
int num_cases,
case_ **cases)
: statement (b, loc),
m_expr (expr),
m_default_block (default_block)
{
m_cases.reserve_exact (num_cases);
for (int i = 0; i< num_cases; i++)
m_cases.quick_push (cases[i]);
}
void
recording::switch_::replay_into (replayer *r)
{
auto_vec <playback::case_> pcases;
int i;
recording::case_ *rcase;
pcases.reserve_exact (m_cases.length ());
FOR_EACH_VEC_ELT (m_cases, i, rcase)
{
playback::case_ pcase (rcase->get_min_value ()->playback_rvalue (),
rcase->get_max_value ()->playback_rvalue (),
rcase->get_dest_block ()->playback_block ());
pcases.safe_push (pcase);
}
playback_block (get_block ())
->add_switch (playback_location (r),
m_expr->playback_rvalue (),
m_default_block->playback_block (),
&pcases);
}
vec <recording::block *>
recording::switch_::get_successor_blocks () const
{
vec <block *> result;
result.create (m_cases.length () + 1);
result.quick_push (m_default_block);
int i;
case_ *c;
FOR_EACH_VEC_ELT (m_cases, i, c)
result.quick_push (c->get_dest_block ());
return result;
}
recording::string *
recording::switch_::make_debug_string ()
{
auto_vec <char> cases_str;
int i;
case_ *c;
FOR_EACH_VEC_ELT (m_cases, i, c)
{
size_t len = strlen (c->get_debug_string ());
unsigned idx = cases_str.length ();
cases_str.safe_grow (idx + 1 + len);
cases_str[idx] = ' ';
memcpy (&(cases_str[idx + 1]),
c->get_debug_string (),
len);
}
cases_str.safe_push ('\0');
return string::from_printf (m_ctxt,
"switch (%s) {default: goto %s;%s}",
m_expr->get_debug_string (),
m_default_block->get_debug_string (),
&cases_str[0]);
}
void
recording::switch_::write_reproducer (reproducer &r)
{
r.make_identifier (this, "switch");
int i;
case_ *c;
const char *cases_id =
r.make_tmp_identifier ("cases_for", this);
r.write ("  gcc_jit_case *%s[%i] = {\n",
cases_id,
m_cases.length ());
FOR_EACH_VEC_ELT (m_cases, i, c)
r.write ("    %s,\n", r.get_identifier (c));
r.write ("  };\n");
const char *fmt =
"  gcc_jit_block_end_with_switch (%s, \n"
"                                 %s, \n"
"                                 %s, \n"
"                                 %s, \n"
"                                 %i, \n"
"                                 %s); \n";
r.write (fmt,
r.get_identifier (get_block ()),
r.get_identifier (get_loc ()),
r.get_identifier_as_rvalue (m_expr),
r.get_identifier (m_default_block),
m_cases.length (),
cases_id);
}
} 
} 
