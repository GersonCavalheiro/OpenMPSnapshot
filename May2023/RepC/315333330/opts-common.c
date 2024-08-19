#include "config.h"
#include "system.h"
#include "intl.h"
#include "coretypes.h"
#include "opts.h"
#include "options.h"
#include "diagnostic.h"
#include "spellcheck.h"
static void prune_options (struct cl_decoded_option **, unsigned int *);
static bool
remapping_prefix_p (const struct cl_option *opt)
{
return opt->flags & CL_UNDOCUMENTED
&& opt->flags & CL_JOINED
&& !(opt->flags & (CL_DRIVER | CL_TARGET | CL_COMMON | CL_LANG_ALL));
}
size_t
find_opt (const char *input, unsigned int lang_mask)
{
size_t mn, mn_orig, mx, md, opt_len;
size_t match_wrong_lang;
int comp;
mn = 0;
mx = cl_options_count;
while (mx - mn > 1)
{
md = (mn + mx) / 2;
opt_len = cl_options[md].opt_len;
comp = strncmp (input, cl_options[md].opt_text + 1, opt_len);
if (comp < 0)
mx = md;
else
mn = md;
}
mn_orig = mn;
match_wrong_lang = OPT_SPECIAL_unknown;
do
{
const struct cl_option *opt = &cl_options[mn];
if (!strncmp (input, opt->opt_text + 1, opt->opt_len)
&& (input[opt->opt_len] == '\0' || (opt->flags & CL_JOINED)))
{
if (opt->flags & lang_mask)
return mn;
if (remapping_prefix_p (opt))
return OPT_SPECIAL_unknown;
if (match_wrong_lang == OPT_SPECIAL_unknown)
match_wrong_lang = mn;
}
mn = opt->back_chain;
}
while (mn != cl_options_count);
if (match_wrong_lang == OPT_SPECIAL_unknown && input[0] == '-')
{
size_t mnc = mn_orig + 1;
size_t cmp_len = strlen (input);
while (mnc < cl_options_count
&& strncmp (input, cl_options[mnc].opt_text + 1, cmp_len) == 0)
{
if (mnc == mn_orig + 1
&& !(cl_options[mnc].flags & CL_JOINED))
match_wrong_lang = mnc;
else if (mnc == mn_orig + 2
&& match_wrong_lang == mn_orig + 1
&& (cl_options[mnc].flags & CL_JOINED)
&& (cl_options[mnc].opt_len
== cl_options[mn_orig + 1].opt_len + 1)
&& strncmp (cl_options[mnc].opt_text + 1,
cl_options[mn_orig + 1].opt_text + 1,
cl_options[mn_orig + 1].opt_len) == 0)
; 
else
return OPT_SPECIAL_unknown;
mnc++;
}
}
return match_wrong_lang;
}
int
integral_argument (const char *arg)
{
const char *p = arg;
while (*p && ISDIGIT (*p))
p++;
if (*p == '\0')
return atoi (arg);
if (arg[0] == '0' && (arg[1] == 'x' || arg[1] == 'X'))
{
p = arg + 2;
while (*p && ISXDIGIT (*p))
p++;
if (p != arg + 2 && *p == '\0')
return strtol (arg, NULL, 16);
}
return -1;
}
static bool
option_ok_for_language (const struct cl_option *option,
unsigned int lang_mask)
{
if (!(option->flags & lang_mask))
return false;
else if ((option->flags & CL_TARGET)
&& (option->flags & (CL_LANG_ALL | CL_DRIVER))
&& !(option->flags & (lang_mask & ~CL_COMMON & ~CL_TARGET)))
return false;
return true;
}
static bool
enum_arg_ok_for_language (const struct cl_enum_arg *enum_arg,
unsigned int lang_mask)
{
return (lang_mask & CL_DRIVER) || !(enum_arg->flags & CL_ENUM_DRIVER_ONLY);
}
static bool
enum_arg_to_value (const struct cl_enum_arg *enum_args,
const char *arg, int *value, unsigned int lang_mask)
{
unsigned int i;
for (i = 0; enum_args[i].arg != NULL; i++)
if (strcmp (arg, enum_args[i].arg) == 0
&& enum_arg_ok_for_language (&enum_args[i], lang_mask))
{
*value = enum_args[i].value;
return true;
}
return false;
}
bool
opt_enum_arg_to_value (size_t opt_index, const char *arg, int *value,
unsigned int lang_mask)
{
const struct cl_option *option = &cl_options[opt_index];
gcc_assert (option->var_type == CLVC_ENUM);
return enum_arg_to_value (cl_enums[option->var_enum].values, arg,
value, lang_mask);
}
bool
enum_value_to_arg (const struct cl_enum_arg *enum_args,
const char **argp, int value, unsigned int lang_mask)
{
unsigned int i;
for (i = 0; enum_args[i].arg != NULL; i++)
if (enum_args[i].value == value
&& (enum_args[i].flags & CL_ENUM_CANONICAL)
&& enum_arg_ok_for_language (&enum_args[i], lang_mask))
{
*argp = enum_args[i].arg;
return true;
}
for (i = 0; enum_args[i].arg != NULL; i++)
if (enum_args[i].value == value
&& enum_arg_ok_for_language (&enum_args[i], lang_mask))
{
*argp = enum_args[i].arg;
return false;
}
*argp = NULL;
return false;
}
static void
generate_canonical_option (size_t opt_index, const char *arg, int value,
struct cl_decoded_option *decoded)
{
const struct cl_option *option = &cl_options[opt_index];
const char *opt_text = option->opt_text;
if (value == 0
&& !option->cl_reject_negative
&& (opt_text[1] == 'W' || opt_text[1] == 'f'
|| opt_text[1] == 'g' || opt_text[1] == 'm'))
{
char *t = XOBNEWVEC (&opts_obstack, char, option->opt_len + 5);
t[0] = '-';
t[1] = opt_text[1];
t[2] = 'n';
t[3] = 'o';
t[4] = '-';
memcpy (t + 5, opt_text + 2, option->opt_len);
opt_text = t;
}
decoded->canonical_option[2] = NULL;
decoded->canonical_option[3] = NULL;
if (arg)
{
if ((option->flags & CL_SEPARATE)
&& !option->cl_separate_alias)
{
decoded->canonical_option[0] = opt_text;
decoded->canonical_option[1] = arg;
decoded->canonical_option_num_elements = 2;
}
else
{
gcc_assert (option->flags & CL_JOINED);
decoded->canonical_option[0] = opts_concat (opt_text, arg, NULL);
decoded->canonical_option[1] = NULL;
decoded->canonical_option_num_elements = 1;
}
}
else
{
decoded->canonical_option[0] = opt_text;
decoded->canonical_option[1] = NULL;
decoded->canonical_option_num_elements = 1;
}
}
struct option_map
{
const char *opt0;
const char *opt1;
const char *new_prefix;
bool another_char_needed;
bool negated;
};
static const struct option_map option_map[] =
{
{ "-Wno-", NULL, "-W", false, true },
{ "-fno-", NULL, "-f", false, true },
{ "-gno-", NULL, "-g", false, true },
{ "-mno-", NULL, "-m", false, true },
{ "--debug=", NULL, "-g", false, false },
{ "--machine-", NULL, "-m", true, false },
{ "--machine-no-", NULL, "-m", false, true },
{ "--machine=", NULL, "-m", false, false },
{ "--machine=no-", NULL, "-m", false, true },
{ "--machine", "", "-m", false, false },
{ "--machine", "no-", "-m", false, true },
{ "--optimize=", NULL, "-O", false, false },
{ "--std=", NULL, "-std=", false, false },
{ "--std", "", "-std=", false, false },
{ "--warn-", NULL, "-W", true, false },
{ "--warn-no-", NULL, "-W", false, true },
{ "--", NULL, "-f", true, false },
{ "--no-", NULL, "-f", false, true }
};
void
add_misspelling_candidates (auto_vec<char *> *candidates,
const struct cl_option *option,
const char *opt_text)
{
gcc_assert (candidates);
gcc_assert (option);
gcc_assert (opt_text);
if (remapping_prefix_p (option))
return;
candidates->safe_push (xstrdup (opt_text + 1));
for (unsigned i = 0; i < ARRAY_SIZE (option_map); i++)
{
const char *opt0 = option_map[i].opt0;
const char *new_prefix = option_map[i].new_prefix;
size_t new_prefix_len = strlen (new_prefix);
if (option->cl_reject_negative && option_map[i].negated)
continue;
if (strncmp (opt_text, new_prefix, new_prefix_len) == 0)
{
char *alternative = concat (opt0 + 1, opt_text + new_prefix_len,
NULL);
candidates->safe_push (alternative);
}
}
}
static unsigned int
decode_cmdline_option (const char **argv, unsigned int lang_mask,
struct cl_decoded_option *decoded)
{
size_t opt_index;
const char *arg = 0;
int value = 1;
unsigned int result = 1, i, extra_args, separate_args = 0;
int adjust_len = 0;
size_t total_len;
char *p;
const struct cl_option *option;
int errors = 0;
const char *warn_message = NULL;
bool separate_arg_flag;
bool joined_arg_flag;
bool have_separate_arg = false;
extra_args = 0;
opt_index = find_opt (argv[0] + 1, lang_mask);
i = 0;
while (opt_index == OPT_SPECIAL_unknown
&& i < ARRAY_SIZE (option_map))
{
const char *opt0 = option_map[i].opt0;
const char *opt1 = option_map[i].opt1;
const char *new_prefix = option_map[i].new_prefix;
bool another_char_needed = option_map[i].another_char_needed;
size_t opt0_len = strlen (opt0);
size_t opt1_len = (opt1 == NULL ? 0 : strlen (opt1));
size_t optn_len = (opt1 == NULL ? opt0_len : opt1_len);
size_t new_prefix_len = strlen (new_prefix);
extra_args = (opt1 == NULL ? 0 : 1);
value = !option_map[i].negated;
if (strncmp (argv[0], opt0, opt0_len) == 0
&& (opt1 == NULL
|| (argv[1] != NULL && strncmp (argv[1], opt1, opt1_len) == 0))
&& (!another_char_needed
|| argv[extra_args][optn_len] != 0))
{
size_t arglen = strlen (argv[extra_args]);
char *dup;
adjust_len = (int) optn_len - (int) new_prefix_len;
dup = XNEWVEC (char, arglen + 1 - adjust_len);
memcpy (dup, new_prefix, new_prefix_len);
memcpy (dup + new_prefix_len, argv[extra_args] + optn_len,
arglen - optn_len + 1);
opt_index = find_opt (dup + 1, lang_mask);
free (dup);
}
i++;
}
if (opt_index == OPT_SPECIAL_unknown)
{
arg = argv[0];
extra_args = 0;
value = 1;
goto done;
}
option = &cl_options[opt_index];
if (!value && option->cl_reject_negative)
{
opt_index = OPT_SPECIAL_unknown;
errors |= CL_ERR_NEGATIVE;
arg = argv[0];
goto done;
}
result = extra_args + 1;
warn_message = option->warn_message;
if (option->cl_disabled)
errors |= CL_ERR_DISABLED;
separate_arg_flag = ((option->flags & CL_SEPARATE)
&& !(option->cl_no_driver_arg
&& (lang_mask & CL_DRIVER)));
separate_args = (separate_arg_flag
? option->cl_separate_nargs + 1
: 0);
joined_arg_flag = (option->flags & CL_JOINED) != 0;
if (joined_arg_flag)
{
arg = argv[extra_args] + cl_options[opt_index].opt_len + 1 + adjust_len;
if (*arg == '\0' && !option->cl_missing_ok)
{
if (separate_arg_flag)
{
arg = argv[extra_args + 1];
result = extra_args + 2;
if (arg == NULL)
result = extra_args + 1;
else
have_separate_arg = true;
}
else
arg = NULL;
}
}
else if (separate_arg_flag)
{
arg = argv[extra_args + 1];
for (i = 0; i < separate_args; i++)
if (argv[extra_args + 1 + i] == NULL)
{
errors |= CL_ERR_MISSING_ARG;
break;
}
result = extra_args + 1 + i;
if (arg != NULL)
have_separate_arg = true;
}
if (arg == NULL && (separate_arg_flag || joined_arg_flag))
errors |= CL_ERR_MISSING_ARG;
if (option->alias_target != N_OPTS
&& (!option->cl_separate_alias || have_separate_arg))
{
size_t new_opt_index = option->alias_target;
if (new_opt_index == OPT_SPECIAL_ignore)
{
gcc_assert (option->alias_arg == NULL);
gcc_assert (option->neg_alias_arg == NULL);
opt_index = new_opt_index;
arg = NULL;
value = 1;
}
else
{
const struct cl_option *new_option = &cl_options[new_opt_index];
gcc_assert (new_option->alias_target == N_OPTS
|| new_option->cl_separate_alias);
if (option->neg_alias_arg)
{
gcc_assert (option->alias_arg != NULL);
gcc_assert (arg == NULL);
gcc_assert (!option->cl_negative_alias);
if (value)
arg = option->alias_arg;
else
arg = option->neg_alias_arg;
value = 1;
}
else if (option->alias_arg)
{
gcc_assert (value == 1);
gcc_assert (arg == NULL);
gcc_assert (!option->cl_negative_alias);
arg = option->alias_arg;
}
if (option->cl_negative_alias)
value = !value;
opt_index = new_opt_index;
option = new_option;
if (value == 0)
gcc_assert (!option->cl_reject_negative);
separate_arg_flag = ((option->flags & CL_SEPARATE)
&& !(option->cl_no_driver_arg
&& (lang_mask & CL_DRIVER)));
joined_arg_flag = (option->flags & CL_JOINED) != 0;
if (separate_args > 1 || option->cl_separate_nargs)
gcc_assert (separate_args
== (unsigned int) option->cl_separate_nargs + 1);
if (!(errors & CL_ERR_MISSING_ARG))
{
if (separate_arg_flag || joined_arg_flag)
{
if (option->cl_missing_ok && arg == NULL)
arg = "";
gcc_assert (arg != NULL);
}
else
gcc_assert (arg == NULL);
}
if (option->warn_message)
{
gcc_assert (warn_message == NULL);
warn_message = option->warn_message;
}
if (option->cl_disabled)
errors |= CL_ERR_DISABLED;
}
}
if (!option_ok_for_language (option, lang_mask))
errors |= CL_ERR_WRONG_LANG;
if (arg && option->cl_tolower)
{
size_t j;
size_t len = strlen (arg);
char *arg_lower = XOBNEWVEC (&opts_obstack, char, len + 1);
for (j = 0; j < len; j++)
arg_lower[j] = TOLOWER ((unsigned char) arg[j]);
arg_lower[len] = 0;
arg = arg_lower;
}
if (arg && option->cl_uinteger)
{
value = integral_argument (arg);
if (value == -1)
errors |= CL_ERR_UINT_ARG;
if (option->range_max != -1
&& (value < option->range_min || value > option->range_max))
errors |= CL_ERR_INT_RANGE_ARG;
}
if (arg && (option->var_type == CLVC_ENUM))
{
const struct cl_enum *e = &cl_enums[option->var_enum];
gcc_assert (value == 1);
if (enum_arg_to_value (e->values, arg, &value, lang_mask))
{
const char *carg = NULL;
if (enum_value_to_arg (e->values, &carg, value, lang_mask))
arg = carg;
gcc_assert (carg != NULL);
}
else
errors |= CL_ERR_ENUM_ARG;
}
done:
decoded->opt_index = opt_index;
decoded->arg = arg;
decoded->value = value;
decoded->errors = errors;
decoded->warn_message = warn_message;
if (opt_index == OPT_SPECIAL_unknown)
gcc_assert (result == 1);
gcc_assert (result >= 1 && result <= ARRAY_SIZE (decoded->canonical_option));
decoded->canonical_option_num_elements = result;
total_len = 0;
for (i = 0; i < ARRAY_SIZE (decoded->canonical_option); i++)
{
if (i < result)
{
size_t len;
if (opt_index == OPT_SPECIAL_unknown)
decoded->canonical_option[i] = argv[i];
else
decoded->canonical_option[i] = NULL;
len = strlen (argv[i]);
total_len += (len != 0 ? len : 2) + 1;
}
else
decoded->canonical_option[i] = NULL;
}
if (opt_index != OPT_SPECIAL_unknown && opt_index != OPT_SPECIAL_ignore)
{
generate_canonical_option (opt_index, arg, value, decoded);
if (separate_args > 1)
{
for (i = 0; i < separate_args; i++)
{
if (argv[extra_args + 1 + i] == NULL)
break;
else
decoded->canonical_option[1 + i] = argv[extra_args + 1 + i];
}
gcc_assert (result == 1 + i);
decoded->canonical_option_num_elements = result;
}
}
decoded->orig_option_with_args_text
= p = XOBNEWVEC (&opts_obstack, char, total_len);
for (i = 0; i < result; i++)
{
size_t len = strlen (argv[i]);
if (len == 0)
{
*p++ = '"';
*p++ = '"';
}
else
memcpy (p, argv[i], len);
p += len;
if (i == result - 1)
*p++ = 0;
else
*p++ = ' ';
}
return result;
}
struct obstack opts_obstack;
char *
opts_concat (const char *first, ...)
{
char *newstr, *end;
size_t length = 0;
const char *arg;
va_list ap;
va_start (ap, first);
for (arg = first; arg; arg = va_arg (ap, const char *))
length += strlen (arg);
newstr = XOBNEWVEC (&opts_obstack, char, length + 1);
va_end (ap);
va_start (ap, first);
for (arg = first, end = newstr; arg; arg = va_arg (ap, const char *))
{
length = strlen (arg);
memcpy (end, arg, length);
end += length;
}
*end = '\0';
va_end (ap);
return newstr;
}
void
decode_cmdline_options_to_array (unsigned int argc, const char **argv, 
unsigned int lang_mask,
struct cl_decoded_option **decoded_options,
unsigned int *decoded_options_count)
{
unsigned int n, i;
struct cl_decoded_option *opt_array;
unsigned int num_decoded_options;
opt_array = XNEWVEC (struct cl_decoded_option, argc);
opt_array[0].opt_index = OPT_SPECIAL_program_name;
opt_array[0].warn_message = NULL;
opt_array[0].arg = argv[0];
opt_array[0].orig_option_with_args_text = argv[0];
opt_array[0].canonical_option_num_elements = 1;
opt_array[0].canonical_option[0] = argv[0];
opt_array[0].canonical_option[1] = NULL;
opt_array[0].canonical_option[2] = NULL;
opt_array[0].canonical_option[3] = NULL;
opt_array[0].value = 1;
opt_array[0].errors = 0;
num_decoded_options = 1;
for (i = 1; i < argc; i += n)
{
const char *opt = argv[i];
if (opt[0] != '-' || opt[1] == '\0')
{
generate_option_input_file (opt, &opt_array[num_decoded_options]);
num_decoded_options++;
n = 1;
continue;
}
n = decode_cmdline_option (argv + i, lang_mask,
&opt_array[num_decoded_options]);
num_decoded_options++;
}
*decoded_options = opt_array;
*decoded_options_count = num_decoded_options;
prune_options (decoded_options, decoded_options_count);
}
static bool
cancel_option (int opt_idx, int next_opt_idx, int orig_next_opt_idx)
{
if (cl_options [next_opt_idx].neg_index == opt_idx)
return true;
if (cl_options [next_opt_idx].neg_index != orig_next_opt_idx)
return cancel_option (opt_idx, cl_options [next_opt_idx].neg_index,
orig_next_opt_idx);
return false;
}
static void
prune_options (struct cl_decoded_option **decoded_options,
unsigned int *decoded_options_count)
{
unsigned int old_decoded_options_count = *decoded_options_count;
struct cl_decoded_option *old_decoded_options = *decoded_options;
unsigned int new_decoded_options_count;
struct cl_decoded_option *new_decoded_options
= XNEWVEC (struct cl_decoded_option, old_decoded_options_count);
unsigned int i;
const struct cl_option *option;
unsigned int fdiagnostics_color_idx = 0;
new_decoded_options_count = 0;
for (i = 0; i < old_decoded_options_count; i++)
{
unsigned int j, opt_idx, next_opt_idx;
if (old_decoded_options[i].errors & ~CL_ERR_WRONG_LANG)
goto keep;
opt_idx = old_decoded_options[i].opt_index;
switch (opt_idx)
{
case OPT_SPECIAL_unknown:
case OPT_SPECIAL_ignore:
case OPT_SPECIAL_program_name:
case OPT_SPECIAL_input_file:
goto keep;
case OPT_fdiagnostics_color_:
fdiagnostics_color_idx = i;
continue;
default:
gcc_assert (opt_idx < cl_options_count);
option = &cl_options[opt_idx];
if (option->neg_index < 0)
goto keep;
if ((option->flags & CL_JOINED))
goto keep;
for (j = i + 1; j < old_decoded_options_count; j++)
{
if (old_decoded_options[j].errors & ~CL_ERR_WRONG_LANG)
continue;
next_opt_idx = old_decoded_options[j].opt_index;
if (next_opt_idx >= cl_options_count)
continue;
if (cl_options[next_opt_idx].neg_index < 0)
continue;
if ((cl_options[next_opt_idx].flags & CL_JOINED))
continue;
if (cancel_option (opt_idx, next_opt_idx, next_opt_idx))
break;
}
if (j == old_decoded_options_count)
{
keep:
new_decoded_options[new_decoded_options_count]
= old_decoded_options[i];
new_decoded_options_count++;
}
break;
}
}
if (fdiagnostics_color_idx >= 1)
{
memmove (new_decoded_options + 2, new_decoded_options + 1,
sizeof (struct cl_decoded_option) 
* (new_decoded_options_count - 1));
new_decoded_options[1] = old_decoded_options[fdiagnostics_color_idx];
new_decoded_options_count++;
}
free (old_decoded_options);
new_decoded_options = XRESIZEVEC (struct cl_decoded_option,
new_decoded_options,
new_decoded_options_count);
*decoded_options = new_decoded_options;
*decoded_options_count = new_decoded_options_count;
}
static bool
handle_option (struct gcc_options *opts,
struct gcc_options *opts_set,
const struct cl_decoded_option *decoded,
unsigned int lang_mask, int kind, location_t loc,
const struct cl_option_handlers *handlers,
bool generated_p, diagnostic_context *dc)
{
size_t opt_index = decoded->opt_index;
const char *arg = decoded->arg;
int value = decoded->value;
const struct cl_option *option = &cl_options[opt_index];
void *flag_var = option_flag_var (opt_index, opts);
size_t i;
if (flag_var)
set_option (opts, (generated_p ? NULL : opts_set),
opt_index, value, arg, kind, loc, dc);
for (i = 0; i < handlers->num_handlers; i++)
if (option->flags & handlers->handlers[i].mask)
{
if (!handlers->handlers[i].handler (opts, opts_set, decoded,
lang_mask, kind, loc,
handlers, dc,
handlers->target_option_override_hook))
return false;
}
return true;
}
bool
handle_generated_option (struct gcc_options *opts,
struct gcc_options *opts_set,
size_t opt_index, const char *arg, int value,
unsigned int lang_mask, int kind, location_t loc,
const struct cl_option_handlers *handlers,
bool generated_p, diagnostic_context *dc)
{
struct cl_decoded_option decoded;
generate_option (opt_index, arg, value, lang_mask, &decoded);
return handle_option (opts, opts_set, &decoded, lang_mask, kind, loc,
handlers, generated_p, dc);
}
void
generate_option (size_t opt_index, const char *arg, int value,
unsigned int lang_mask, struct cl_decoded_option *decoded)
{
const struct cl_option *option = &cl_options[opt_index];
decoded->opt_index = opt_index;
decoded->warn_message = NULL;
decoded->arg = arg;
decoded->value = value;
decoded->errors = (option_ok_for_language (option, lang_mask)
? 0
: CL_ERR_WRONG_LANG);
generate_canonical_option (opt_index, arg, value, decoded);
switch (decoded->canonical_option_num_elements)
{
case 1:
decoded->orig_option_with_args_text = decoded->canonical_option[0];
break;
case 2:
decoded->orig_option_with_args_text
= opts_concat (decoded->canonical_option[0], " ",
decoded->canonical_option[1], NULL);
break;
default:
gcc_unreachable ();
}
}
void
generate_option_input_file (const char *file,
struct cl_decoded_option *decoded)
{
decoded->opt_index = OPT_SPECIAL_input_file;
decoded->warn_message = NULL;
decoded->arg = file;
decoded->orig_option_with_args_text = file;
decoded->canonical_option_num_elements = 1;
decoded->canonical_option[0] = file;
decoded->canonical_option[1] = NULL;
decoded->canonical_option[2] = NULL;
decoded->canonical_option[3] = NULL;
decoded->value = 1;
decoded->errors = 0;
}
const char *
candidates_list_and_hint (const char *arg, char *&str,
const auto_vec <const char *> &candidates)
{
size_t len = 0;
int i;
const char *candidate;
char *p;
FOR_EACH_VEC_ELT (candidates, i, candidate)
len += strlen (candidate) + 1;
str = p = XNEWVEC (char, len);
FOR_EACH_VEC_ELT (candidates, i, candidate)
{
len = strlen (candidate);
memcpy (p, candidate, len);
p[len] = ' ';
p += len + 1;
}
p[-1] = '\0';
return find_closest_string (arg, &candidates);
}
static bool
cmdline_handle_error (location_t loc, const struct cl_option *option,
const char *opt, const char *arg, int errors,
unsigned int lang_mask)
{
if (errors & CL_ERR_DISABLED)
{
error_at (loc, "command line option %qs"
" is not supported by this configuration", opt);
return true;
}
if (errors & CL_ERR_MISSING_ARG)
{
if (option->missing_argument_error)
error_at (loc, option->missing_argument_error, opt);
else
error_at (loc, "missing argument to %qs", opt);
return true;
}
if (errors & CL_ERR_UINT_ARG)
{
error_at (loc, "argument to %qs should be a non-negative integer",
option->opt_text);
return true;
}
if (errors & CL_ERR_INT_RANGE_ARG)
{
error_at (loc, "argument to %qs is not between %d and %d",
option->opt_text, option->range_min, option->range_max);
return true;
}
if (errors & CL_ERR_ENUM_ARG)
{
const struct cl_enum *e = &cl_enums[option->var_enum];
unsigned int i;
char *s;
if (e->unknown_error)
error_at (loc, e->unknown_error, arg);
else
error_at (loc, "unrecognized argument in option %qs", opt);
auto_vec <const char *> candidates;
for (i = 0; e->values[i].arg != NULL; i++)
{
if (!enum_arg_ok_for_language (&e->values[i], lang_mask))
continue;
candidates.safe_push (e->values[i].arg);
}
const char *hint = candidates_list_and_hint (arg, s, candidates);
if (hint)
inform (loc, "valid arguments to %qs are: %s; did you mean %qs?",
option->opt_text, s, hint);
else
inform (loc, "valid arguments to %qs are: %s", option->opt_text, s);
XDELETEVEC (s);
return true;
}
return false;
}
void
read_cmdline_option (struct gcc_options *opts,
struct gcc_options *opts_set,
struct cl_decoded_option *decoded,
location_t loc,
unsigned int lang_mask,
const struct cl_option_handlers *handlers,
diagnostic_context *dc)
{
const struct cl_option *option;
const char *opt = decoded->orig_option_with_args_text;
if (decoded->warn_message)
warning_at (loc, 0, decoded->warn_message, opt);
if (decoded->opt_index == OPT_SPECIAL_unknown)
{
if (handlers->unknown_option_callback (decoded))
error_at (loc, "unrecognized command line option %qs", decoded->arg);
return;
}
if (decoded->opt_index == OPT_SPECIAL_ignore)
return;
option = &cl_options[decoded->opt_index];
if (decoded->errors
&& cmdline_handle_error (loc, option, opt, decoded->arg,
decoded->errors, lang_mask))
return;
if (decoded->errors & CL_ERR_WRONG_LANG)
{
handlers->wrong_lang_callback (decoded, lang_mask);
return;
}
gcc_assert (!decoded->errors);
if (!handle_option (opts, opts_set, decoded, lang_mask, DK_UNSPECIFIED,
loc, handlers, false, dc))
error_at (loc, "unrecognized command line option %qs", opt);
}
void
set_option (struct gcc_options *opts, struct gcc_options *opts_set,
int opt_index, int value, const char *arg, int kind,
location_t loc, diagnostic_context *dc)
{
const struct cl_option *option = &cl_options[opt_index];
void *flag_var = option_flag_var (opt_index, opts);
void *set_flag_var = NULL;
if (!flag_var)
return;
if ((diagnostic_t) kind != DK_UNSPECIFIED && dc != NULL)
diagnostic_classify_diagnostic (dc, opt_index, (diagnostic_t) kind, loc);
if (opts_set != NULL)
set_flag_var = option_flag_var (opt_index, opts_set);
switch (option->var_type)
{
case CLVC_BOOLEAN:
*(int *) flag_var = value;
if (set_flag_var)
*(int *) set_flag_var = 1;
break;
case CLVC_EQUAL:
if (option->cl_host_wide_int) 
*(HOST_WIDE_INT *) flag_var = (value
? option->var_value
: !option->var_value);
else
*(int *) flag_var = (value
? option->var_value
: !option->var_value);
if (set_flag_var)
*(int *) set_flag_var = 1;
break;
case CLVC_BIT_CLEAR:
case CLVC_BIT_SET:
if ((value != 0) == (option->var_type == CLVC_BIT_SET))
{
if (option->cl_host_wide_int) 
*(HOST_WIDE_INT *) flag_var |= option->var_value;
else 
*(int *) flag_var |= option->var_value;
}
else
{
if (option->cl_host_wide_int) 
*(HOST_WIDE_INT *) flag_var &= ~option->var_value;
else 
*(int *) flag_var &= ~option->var_value;
}
if (set_flag_var)
{
if (option->cl_host_wide_int) 
*(HOST_WIDE_INT *) set_flag_var |= option->var_value;
else
*(int *) set_flag_var |= option->var_value;
}
break;
case CLVC_STRING:
*(const char **) flag_var = arg;
if (set_flag_var)
*(const char **) set_flag_var = "";
break;
case CLVC_ENUM:
{
const struct cl_enum *e = &cl_enums[option->var_enum];
e->set (flag_var, value);
if (set_flag_var)
e->set (set_flag_var, 1);
}
break;
case CLVC_DEFER:
{
vec<cl_deferred_option> *v
= (vec<cl_deferred_option> *) *(void **) flag_var;
cl_deferred_option p = {opt_index, arg, value};
if (!v)
v = XCNEW (vec<cl_deferred_option>);
v->safe_push (p);
*(void **) flag_var = v;
if (set_flag_var)
*(void **) set_flag_var = v;
}
break;
}
}
void *
option_flag_var (int opt_index, struct gcc_options *opts)
{
const struct cl_option *option = &cl_options[opt_index];
if (option->flag_var_offset == (unsigned short) -1)
return NULL;
return (void *)(((char *) opts) + option->flag_var_offset);
}
int
option_enabled (int opt_idx, void *opts)
{
const struct cl_option *option = &(cl_options[opt_idx]);
struct gcc_options *optsg = (struct gcc_options *) opts;
void *flag_var = option_flag_var (opt_idx, optsg);
if (flag_var)
switch (option->var_type)
{
case CLVC_BOOLEAN:
return *(int *) flag_var != 0;
case CLVC_EQUAL:
if (option->cl_host_wide_int) 
return *(HOST_WIDE_INT *) flag_var == option->var_value;
else
return *(int *) flag_var == option->var_value;
case CLVC_BIT_CLEAR:
if (option->cl_host_wide_int) 
return (*(HOST_WIDE_INT *) flag_var & option->var_value) == 0;
else
return (*(int *) flag_var & option->var_value) == 0;
case CLVC_BIT_SET:
if (option->cl_host_wide_int) 
return (*(HOST_WIDE_INT *) flag_var & option->var_value) != 0;
else 
return (*(int *) flag_var & option->var_value) != 0;
case CLVC_STRING:
case CLVC_ENUM:
case CLVC_DEFER:
break;
}
return -1;
}
bool
get_option_state (struct gcc_options *opts, int option,
struct cl_option_state *state)
{
void *flag_var = option_flag_var (option, opts);
if (flag_var == 0)
return false;
switch (cl_options[option].var_type)
{
case CLVC_BOOLEAN:
case CLVC_EQUAL:
state->data = flag_var;
state->size = (cl_options[option].cl_host_wide_int
? sizeof (HOST_WIDE_INT)
: sizeof (int));
break;
case CLVC_BIT_CLEAR:
case CLVC_BIT_SET:
state->ch = option_enabled (option, opts);
state->data = &state->ch;
state->size = 1;
break;
case CLVC_STRING:
state->data = *(const char **) flag_var;
if (state->data == 0)
state->data = "";
state->size = strlen ((const char *) state->data) + 1;
break;
case CLVC_ENUM:
state->data = flag_var;
state->size = cl_enums[cl_options[option].var_enum].var_size;
break;
case CLVC_DEFER:
return false;
}
return true;
}
void
control_warning_option (unsigned int opt_index, int kind, const char *arg,
bool imply, location_t loc, unsigned int lang_mask,
const struct cl_option_handlers *handlers,
struct gcc_options *opts,
struct gcc_options *opts_set,
diagnostic_context *dc)
{
if (cl_options[opt_index].alias_target != N_OPTS)
{
gcc_assert (!cl_options[opt_index].cl_separate_alias
&& !cl_options[opt_index].cl_negative_alias);
if (cl_options[opt_index].alias_arg)
arg = cl_options[opt_index].alias_arg;
opt_index = cl_options[opt_index].alias_target;
}
if (opt_index == OPT_SPECIAL_ignore)
return;
if (dc)
diagnostic_classify_diagnostic (dc, opt_index, (diagnostic_t) kind, loc);
if (imply)
{
const struct cl_option *option = &cl_options[opt_index];
if (option->var_type == CLVC_BOOLEAN || option->var_type == CLVC_ENUM)
{
int value = 1;
if (arg && *arg == '\0' && !option->cl_missing_ok)
arg = NULL;
if ((option->flags & CL_JOINED) && arg == NULL)
{
cmdline_handle_error (loc, option, option->opt_text, arg,
CL_ERR_MISSING_ARG, lang_mask);
return;
}
if (arg && option->cl_uinteger)
{
value = integral_argument (arg);
if (value == -1)
{
cmdline_handle_error (loc, option, option->opt_text, arg,
CL_ERR_UINT_ARG, lang_mask);
return;
}
}
if (arg && option->var_type == CLVC_ENUM)
{
const struct cl_enum *e = &cl_enums[option->var_enum];
if (enum_arg_to_value (e->values, arg, &value, lang_mask))
{
const char *carg = NULL;
if (enum_value_to_arg (e->values, &carg, value, lang_mask))
arg = carg;
gcc_assert (carg != NULL);
}
else
{
cmdline_handle_error (loc, option, option->opt_text, arg,
CL_ERR_ENUM_ARG, lang_mask);
return;
}
}
handle_generated_option (opts, opts_set,
opt_index, arg, value, lang_mask,
kind, loc, handlers, false, dc);
}
}
}
