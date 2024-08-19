void fn_default_start (void) { }
__attribute__ ((target ("stack-guard=0")))
void fn_att_0 (void) { }
void fn_att_0_default (void) { }
__attribute__ ((target ("stack-guard=16")))
void fn_att_1 (void) { }
void fn_att_1_default (void) { }
__attribute__ ((target ("stack-guard=16,stack-guard=0")))
void fn_att_1_0 (void) { }
__attribute__ ((target ("stack-guard=0,stack-guard=16")))
void fn_att_0_1 (void) { }
#pragma GCC target ("stack-guard=0")
void fn_pragma_0 (void) { }
#pragma GCC reset_options
void fn_pragma_0_default (void) { }
#pragma GCC target ("stack-guard=16")
void fn_pragma_1 (void) { }
#pragma GCC reset_options
void fn_pragma_1_default (void) { }
#pragma GCC target ("stack-guard=16")
#pragma GCC target ("stack-guard=0")
void fn_pragma_1_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("stack-guard=0")
#pragma GCC target ("stack-guard=16")
void fn_pragma_0_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("stack-guard=0")
__attribute__ ((target ("stack-guard=0")))
void fn_pragma_0_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("stack-guard=0")
__attribute__ ((target ("stack-guard=0")))
void fn_pragma_1_att_0 (void) { }
#pragma GCC reset_options
#pragma GCC target ("stack-guard=0")
__attribute__ ((target ("stack-guard=16")))
void fn_pragma_0_att_1 (void) { }
#pragma GCC reset_options
#pragma GCC target ("stack-guard=0")
__attribute__ ((target ("stack-guard=16")))
void fn_pragma_1_att_1 (void) { }
#pragma GCC reset_options
void fn_default_end (void) { }
