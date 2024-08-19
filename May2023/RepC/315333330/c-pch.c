#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-common.h"
#include "timevar.h"
#include "flags.h"
#include "debug.h"
#include "c-pragma.h"
#include "langhooks.h"
#include "hosthooks.h"
static const struct c_pch_matching
{
int *flag_var;
const char *flag_name;
} pch_matching[] = {
{ &flag_exceptions, "-fexceptions" },
};
enum {
MATCH_SIZE = ARRAY_SIZE (pch_matching)
};
static const char no_checksum[16] = { 0 };
struct c_pch_validity
{
unsigned char debug_info_type;
signed char match[MATCH_SIZE];
void (*pch_init) (void);
size_t target_data_length;
};
#define IDENT_LENGTH 8
static FILE *pch_outfile;
static const char *get_ident (void);
static const char *
get_ident (void)
{
static char result[IDENT_LENGTH];
static const char templ[] = "gpch.014";
static const char c_language_chars[] = "Co+O";
memcpy (result, templ, IDENT_LENGTH);
result[4] = c_language_chars[c_language];
return result;
}
static bool pch_ready_to_save_cpp_state = false;
void
pch_init (void)
{
FILE *f;
struct c_pch_validity v;
void *target_validity;
static const char partial_pch[] = "gpcWrite";
if (!pch_file)
return;
f = fopen (pch_file, "w+b");
if (f == NULL)
fatal_error (input_location, "can%'t create precompiled header %s: %m",
pch_file);
pch_outfile = f;
gcc_assert (memcmp (executable_checksum, no_checksum, 16) != 0);
memset (&v, '\0', sizeof (v));
v.debug_info_type = write_symbols;
{
size_t i;
for (i = 0; i < MATCH_SIZE; i++)
{
v.match[i] = *pch_matching[i].flag_var;
gcc_assert (v.match[i] == *pch_matching[i].flag_var);
}
}
v.pch_init = &pch_init;
target_validity = targetm.get_pch_validity (&v.target_data_length);
if (fwrite (partial_pch, IDENT_LENGTH, 1, f) != 1
|| fwrite (executable_checksum, 16, 1, f) != 1
|| fwrite (&v, sizeof (v), 1, f) != 1
|| fwrite (target_validity, v.target_data_length, 1, f) != 1)
fatal_error (input_location, "can%'t write to %s: %m", pch_file);
(*debug_hooks->handle_pch) (0);
if (pch_ready_to_save_cpp_state)
pch_cpp_save_state ();
XDELETE (target_validity);
}
static bool pch_cpp_state_saved = false;
void
pch_cpp_save_state (void)
{
if (!pch_cpp_state_saved)
{
if (pch_outfile)
{
cpp_save_state (parse_in, pch_outfile);
pch_cpp_state_saved = true;
}
else
pch_ready_to_save_cpp_state = true;
}
}
void
c_common_write_pch (void)
{
timevar_push (TV_PCH_SAVE);
targetm.prepare_pch_save ();
(*debug_hooks->handle_pch) (1);
prepare_target_option_nodes_for_pch ();
cpp_write_pch_deps (parse_in, pch_outfile);
gt_pch_save (pch_outfile);
timevar_push (TV_PCH_CPP_SAVE);
cpp_write_pch_state (parse_in, pch_outfile);
timevar_pop (TV_PCH_CPP_SAVE);
if (fseek (pch_outfile, 0, SEEK_SET) != 0
|| fwrite (get_ident (), IDENT_LENGTH, 1, pch_outfile) != 1)
fatal_error (input_location, "can%'t write %s: %m", pch_file);
fclose (pch_outfile);
timevar_pop (TV_PCH_SAVE);
}
int
c_common_valid_pch (cpp_reader *pfile, const char *name, int fd)
{
int sizeread;
int result;
char ident[IDENT_LENGTH + 16];
const char *pch_ident;
struct c_pch_validity v;
gcc_assert (memcmp (executable_checksum, no_checksum, 16) != 0);
sizeread = read (fd, ident, IDENT_LENGTH + 16);
if (sizeread == -1)
fatal_error (input_location, "can%'t read %s: %m", name);
else if (sizeread != IDENT_LENGTH + 16)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING, "%s: too short to be a PCH file",
name);
return 2;
}
pch_ident = get_ident();
if (memcmp (ident, pch_ident, IDENT_LENGTH) != 0)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
{
if (memcmp (ident, pch_ident, 5) == 0)
cpp_error (pfile, CPP_DL_WARNING,
"%s: not compatible with this GCC version", name);
else if (memcmp (ident, pch_ident, 4) == 0)
cpp_error (pfile, CPP_DL_WARNING, "%s: not for %s", name,
lang_hooks.name);
else
cpp_error (pfile, CPP_DL_WARNING, "%s: not a PCH file", name);
}
return 2;
}
if (memcmp (ident + IDENT_LENGTH, executable_checksum, 16) != 0)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING,
"%s: created by a different GCC executable", name);
return 2;
}
if (read (fd, &v, sizeof (v)) != sizeof (v))
fatal_error (input_location, "can%'t read %s: %m", name);
if (v.debug_info_type != write_symbols
&& write_symbols != NO_DEBUG)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING,
"%s: created with -g%s, but used with -g%s", name,
debug_type_names[v.debug_info_type],
debug_type_names[write_symbols]);
return 2;
}
{
size_t i;
for (i = 0; i < MATCH_SIZE; i++)
if (*pch_matching[i].flag_var != v.match[i])
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING,
"%s: settings for %s do not match", name,
pch_matching[i].flag_name);
return 2;
}
}
if (v.pch_init != &pch_init)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING,
"%s: had text segment at different address", name);
return 2;
}
{
void *this_file_data = xmalloc (v.target_data_length);
const char *msg;
if ((size_t) read (fd, this_file_data, v.target_data_length)
!= v.target_data_length)
fatal_error (input_location, "can%'t read %s: %m", name);
msg = targetm.pch_valid_p (this_file_data, v.target_data_length);
free (this_file_data);
if (msg != NULL)
{
if (cpp_get_options (pfile)->warn_invalid_pch)
cpp_error (pfile, CPP_DL_WARNING, "%s: %s", name, msg);
return 2;
}
}
result = cpp_valid_state (pfile, name, fd);
if (result == -1)
return 2;
else
return result == 0;
}
void (*lang_post_pch_load) (void);
void
c_common_read_pch (cpp_reader *pfile, const char *name,
int fd, const char *orig_name ATTRIBUTE_UNUSED)
{
FILE *f;
struct save_macro_data *smd;
expanded_location saved_loc;
bool saved_trace_includes;
timevar_push (TV_PCH_RESTORE);
f = fdopen (fd, "rb");
if (f == NULL)
{
cpp_errno (pfile, CPP_DL_ERROR, "calling fdopen");
close (fd);
goto end;
}
cpp_get_callbacks (parse_in)->valid_pch = NULL;
saved_loc = expand_location (line_table->highest_line);
saved_trace_includes = line_table->trace_includes;
timevar_push (TV_PCH_CPP_RESTORE);
cpp_prepare_state (pfile, &smd);
timevar_pop (TV_PCH_CPP_RESTORE);
gt_pch_restore (f);
cpp_set_line_map (pfile, line_table);
rebuild_location_adhoc_htab (line_table);
timevar_push (TV_PCH_CPP_RESTORE);
if (cpp_read_state (pfile, name, f, smd) != 0)
{
fclose (f);
timevar_pop (TV_PCH_CPP_RESTORE);
goto end;
}
timevar_pop (TV_PCH_CPP_RESTORE);
fclose (f);
line_table->trace_includes = saved_trace_includes;
linemap_add (line_table, LC_ENTER, 0, saved_loc.file, saved_loc.line);
if (lang_post_pch_load)
(*lang_post_pch_load) ();
end:
timevar_pop (TV_PCH_RESTORE);
}
void
c_common_no_more_pch (void)
{
if (cpp_get_callbacks (parse_in)->valid_pch)
{
cpp_get_callbacks (parse_in)->valid_pch = NULL;
host_hooks.gt_pch_use_address (NULL, 0, -1, 0);
}
}
void
c_common_pch_pragma (cpp_reader *pfile, const char *name)
{
int fd;
if (!cpp_get_options (pfile)->preprocessed)
{
error ("pch_preprocess pragma should only be used with -fpreprocessed");
inform (input_location, "use #include instead");
return;
}
fd = open (name, O_RDONLY | O_BINARY, 0666);
if (fd == -1)
fatal_error (input_location, "%s: couldn%'t open PCH file: %m", name);
if (c_common_valid_pch (pfile, name, fd) != 1)
{
if (!cpp_get_options (pfile)->warn_invalid_pch)
inform (input_location, "use -Winvalid-pch for more information");
fatal_error (input_location, "%s: PCH file was invalid", name);
}
c_common_read_pch (pfile, name, fd, name);
close (fd);
}
