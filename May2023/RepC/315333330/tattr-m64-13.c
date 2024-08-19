void fn_default_start (void) { }
__attribute__ ((target ("backchain")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("no-backchain")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("no-backchain,backchain")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("backchain,no-backchain")))
void fn_att_1_0 (void) { }
#pragma GCC target ("backchain")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("no-backchain")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("no-backchain")
#pragma GCC target ("backchain")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("backchain")
#pragma GCC target ("no-backchain")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("backchain")
__attribute__ ((target ("backchain")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("backchain")
__attribute__ ((target ("backchain")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("backchain")
__attribute__ ((target ("no-backchain")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("backchain")
__attribute__ ((target ("no-backchain")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }
