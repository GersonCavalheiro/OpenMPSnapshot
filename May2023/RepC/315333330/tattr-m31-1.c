void fn_default_start (void) { }
__attribute__ ((target ("arch=z13")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("arch=z10")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("arch=z10,arch=z13")))
void fn_att_0_1 (void) { }
__attribute__ ((target ("arch=z13,arch=z10")))
void fn_att_1_0 (void) { }
#pragma GCC target ("arch=z13")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("arch=z10")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("arch=z10")
#pragma GCC target ("arch=z13")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z13")
#pragma GCC target ("arch=z10")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z13")
__attribute__ ((target ("arch=z13")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z13")
__attribute__ ((target ("arch=z13")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z13")
__attribute__ ((target ("arch=z10")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z13")
__attribute__ ((target ("arch=z10")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }
