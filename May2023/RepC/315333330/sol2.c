#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "attribs.h"
#include "diagnostic-core.h"
#include "varasm.h"
#include "output.h"
tree solaris_pending_aligns, solaris_pending_inits, solaris_pending_finis;
void
solaris_insert_attributes (tree decl, tree *attributes)
{
tree *x, next;
if (solaris_pending_aligns != NULL && TREE_CODE (decl) == VAR_DECL)
for (x = &solaris_pending_aligns; *x; x = &TREE_CHAIN (*x))
{
tree name = TREE_PURPOSE (*x);
tree value = TREE_VALUE (*x);
if (DECL_NAME (decl) == name)
{
if (lookup_attribute ("aligned", DECL_ATTRIBUTES (decl))
|| lookup_attribute ("aligned", *attributes))
warning (0, "ignoring %<#pragma align%> for explicitly "
"aligned %q+D", decl);
else
*attributes = tree_cons (get_identifier ("aligned"), value,
*attributes);
next = TREE_CHAIN (*x);
ggc_free (*x);
*x = next;
break;
}
}
if (solaris_pending_inits != NULL && TREE_CODE (decl) == FUNCTION_DECL)
for (x = &solaris_pending_inits; *x; x = &TREE_CHAIN (*x))
{
tree name = TREE_PURPOSE (*x);
if (DECL_NAME (decl) == name)
{
*attributes = tree_cons (get_identifier ("init"), NULL,
*attributes);
TREE_USED (decl) = 1;
DECL_PRESERVE_P (decl) = 1;
next = TREE_CHAIN (*x);
ggc_free (*x);
*x = next;
break;
}
}
if (solaris_pending_finis != NULL && TREE_CODE (decl) == FUNCTION_DECL)
for (x = &solaris_pending_finis; *x; x = &TREE_CHAIN (*x))
{
tree name = TREE_PURPOSE (*x);
if (DECL_NAME (decl) == name)
{
*attributes = tree_cons (get_identifier ("fini"), NULL,
*attributes);
TREE_USED (decl) = 1;
DECL_PRESERVE_P (decl) = 1;
next = TREE_CHAIN (*x);
ggc_free (*x);
*x = next;
break;
}
}
}
void
solaris_output_init_fini (FILE *file, tree decl)
{
if (lookup_attribute ("init", DECL_ATTRIBUTES (decl)))
{
fprintf (file, "\t.pushsection\t" SECTION_NAME_FORMAT "\n", ".init");
ASM_OUTPUT_CALL (file, decl);
fprintf (file, "\t.popsection\n");
}
if (lookup_attribute ("fini", DECL_ATTRIBUTES (decl)))
{
fprintf (file, "\t.pushsection\t" SECTION_NAME_FORMAT "\n", ".fini");
ASM_OUTPUT_CALL (file, decl);
fprintf (file, "\t.popsection\n");
}
}
void
solaris_assemble_visibility (tree decl, int vis ATTRIBUTE_UNUSED)
{
#ifdef HAVE_GAS_HIDDEN
static const char * const visibility_types[] = {
NULL, "symbolic", "hidden", "hidden"
};
const char *name, *type;
tree id = DECL_ASSEMBLER_NAME (decl);
while (IDENTIFIER_TRANSPARENT_ALIAS (id))
id = TREE_CHAIN (id);
name = IDENTIFIER_POINTER (id);
type = visibility_types[vis];
fprintf (asm_out_file, "\t.%s\t", type);
assemble_name (asm_out_file, name);
fprintf (asm_out_file, "\n");
#else
if (!DECL_ARTIFICIAL (decl))
warning (OPT_Wattributes, "visibility attribute not supported "
"in this configuration; ignored");
#endif
}
typedef struct comdat_entry
{
const char *name;
unsigned int flags;
tree decl;
const char *sig;
} comdat_entry;
struct comdat_entry_hasher : nofree_ptr_hash <comdat_entry>
{
static inline hashval_t hash (const comdat_entry *);
static inline bool equal (const comdat_entry *, const comdat_entry *);
static inline void remove (comdat_entry *);
};
inline hashval_t
comdat_entry_hasher::hash (const comdat_entry *entry)
{
return htab_hash_string (entry->sig);
}
inline bool
comdat_entry_hasher::equal (const comdat_entry *entry1,
const comdat_entry *entry2)
{
return strcmp (entry1->sig, entry2->sig) == 0;
}
static hash_table<comdat_entry_hasher> *solaris_comdat_htab;
void
solaris_elf_asm_comdat_section (const char *name, unsigned int flags, tree decl)
{
const char *signature;
char *section;
comdat_entry entry, **slot;
if (TREE_CODE (decl) == IDENTIFIER_NODE)
signature = IDENTIFIER_POINTER (decl);
else
signature = IDENTIFIER_POINTER (DECL_COMDAT_GROUP (decl));
section = concat (name, "%", signature, NULL);
targetm.asm_out.named_section (section, flags & ~SECTION_LINKONCE, decl);
fprintf (asm_out_file, "\t.group\t%s," SECTION_NAME_FORMAT ",#comdat\n",
signature, section);
if (!solaris_comdat_htab)
solaris_comdat_htab = new hash_table<comdat_entry_hasher> (37);
entry.sig = signature;
slot = solaris_comdat_htab->find_slot (&entry, INSERT);
if (*slot == NULL)
{
*slot = XCNEW (comdat_entry);
(*slot)->name = section;
(*slot)->flags = flags & ~SECTION_LINKONCE;
(*slot)->decl = decl;
(*slot)->sig = signature;
}
}
int
solaris_define_comdat_signature (comdat_entry **slot,
void *aux ATTRIBUTE_UNUSED)
{
comdat_entry *entry = *slot;
tree decl = entry->decl;
if (TREE_CODE (decl) != IDENTIFIER_NODE)
decl = DECL_COMDAT_GROUP (decl);
if (!TREE_SYMBOL_REFERENCED (decl))
{
switch_to_section (get_section (entry->name, entry->flags, entry->decl));
ASM_OUTPUT_LABEL (asm_out_file, entry->sig);
}
return 1;
}
void
solaris_file_end (void)
{
if (!solaris_comdat_htab)
return;
solaris_comdat_htab->traverse <void *, solaris_define_comdat_signature>
(NULL);
}
void
solaris_override_options (void)
{
if (!HAVE_LD_EH_FRAME_CIEV3 && !global_options_set.x_dwarf_version)
dwarf_version = 2;
}
