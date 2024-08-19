extern void vms_c_register_pragma (void);
extern void vms_c_common_override_options (void);
extern int vms_c_get_crtl_ver (void);
extern int vms_c_get_vms_ver (void);
void vms_patch_builtins (void);
#ifdef TREE_CODE
extern section *vms_function_section (tree decl ATTRIBUTE_UNUSED,
enum node_frequency freq ATTRIBUTE_UNUSED,
bool startup ATTRIBUTE_UNUSED,
bool exit ATTRIBUTE_UNUSED);
extern void vms_start_function (const char *fname);
#endif 
