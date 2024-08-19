#define IN_TARGET_CODE 1
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "c-family/c-common.h"
#include "c-family/c-pragma.h"
#include "m32c-protos.h"
static void
m32c_pragma_memregs (cpp_reader * reader ATTRIBUTE_UNUSED)
{
tree val;
enum cpp_ttype type;
HOST_WIDE_INT i;
type = pragma_lex (&val);
if (type == CPP_NUMBER)
{
if (tree_fits_uhwi_p (val))
{
i = tree_to_uhwi (val);
type = pragma_lex (&val);
if (type != CPP_EOF)
warning (0, "junk at end of #pragma GCC memregs [0..16]");
if (i >= 0 && i <= 16)
{
if (!ok_to_change_target_memregs)
{
warning (0,
"#pragma GCC memregs must precede any function decls");
return;
}
target_memregs = i;
m32c_conditional_register_usage ();
}
else
{
warning (0, "#pragma GCC memregs takes a number [0..16]");
}
return;
}
}
error ("#pragma GCC memregs takes a number [0..16]");
}
static void
m32c_pragma_address (cpp_reader * reader ATTRIBUTE_UNUSED)
{
tree var, addr;
enum cpp_ttype type;
type = pragma_lex (&var);
if (type == CPP_NAME)
{
type = pragma_lex (&addr);
if (type == CPP_NUMBER)
{
if (var != error_mark_node)
{
unsigned uaddr = tree_to_uhwi (addr);
m32c_note_pragma_address (IDENTIFIER_POINTER (var), uaddr);
}
type = pragma_lex (&var);
if (type != CPP_EOF)
{
error ("junk at end of #pragma ADDRESS");
}
return;
}
}
error ("malformed #pragma ADDRESS variable address");
}
void
m32c_register_pragmas (void)
{
c_register_pragma ("GCC", "memregs", m32c_pragma_memregs);
c_register_pragma (NULL, "ADDRESS", m32c_pragma_address);
c_register_pragma (NULL, "address", m32c_pragma_address);
if (TARGET_A16)
c_register_addr_space ("__far", ADDR_SPACE_FAR);
else
c_register_addr_space ("__far", ADDR_SPACE_GENERIC);
}
