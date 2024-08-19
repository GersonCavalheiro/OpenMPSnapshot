void fn_default_start (void) { }
__attribute__ ((target ("tune=z10")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("tune=z13")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("tune=z13,tune=z10")))
void fn_att_1_0 (void) { }
__attribute__ ((target ("tune=z10,tune=z13")))
void fn_att_0_1 (void) { }
#pragma GCC target ("tune=z10")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("tune=z13")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("tune=z13")
#pragma GCC target ("tune=z10")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z10")
#pragma GCC target ("tune=z13")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z10")
__attribute__ ((target ("tune=z10")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z10")
__attribute__ ((target ("tune=z10")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z10")
__attribute__ ((target ("tune=z13")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z10")
__attribute__ ((target ("tune=z13")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }
