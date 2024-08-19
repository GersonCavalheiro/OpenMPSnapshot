void fn_p0_1 (void) { }
__attribute__ ((target("arch=zEC12")))
void fn_p0_2 (void) { }
__attribute__ ((target("tune=z196")))
void fn_p0_3 (void) { }
__attribute__ ((target("arch=zEC12,tune=z196")))
void fn_p0_4 (void) { }
__attribute__ ((target("tune=z196,arch=zEC12")))
void fn_p0_5 (void) { }
#pragma GCC target ("arch=z9-ec")
void fn_pa_1 (void) { }
__attribute__ ((target("arch=zEC12")))
void fn_pa_2 (void) { }
__attribute__ ((target("tune=z196")))
void fn_pa_3 (void) { }
__attribute__ ((target("arch=zEC12,tune=z196")))
void fn_pa_4 (void) { }
__attribute__ ((target("tune=z196,arch=zEC12")))
void fn_pa_5 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z9-109")
void fn_pt_1 (void) { }
__attribute__ ((target("arch=zEC12")))
void fn_pt_2 (void) { }
__attribute__ ((target("tune=z196")))
void fn_pt_3 (void) { }
__attribute__ ((target("arch=zEC12,tune=z196")))
void fn_pt_4 (void) { }
__attribute__ ((target("tune=z196,arch=zEC12")))
void fn_pt_5 (void) { }
#pragma GCC reset_options
#pragma GCC target ("arch=z9-ec,tune=z9-109")
void fn_pat_1 (void) { }
__attribute__ ((target("arch=zEC12")))
void fn_pat_2 (void) { }
__attribute__ ((target("tune=z196")))
void fn_pat_3 (void) { }
__attribute__ ((target("arch=zEC12,tune=z196")))
void fn_pat_4 (void) { }
__attribute__ ((target("tune=z196,arch=zEC12")))
void fn_pat_5 (void) { }
#pragma GCC reset_options
#pragma GCC target ("tune=z9-109,arch=z9-ec")
void fn_pta_1 (void) { }
__attribute__ ((target("arch=zEC12")))
void fn_pta_2 (void) { }
__attribute__ ((target("tune=z196")))
void fn_pta_3 (void) { }
__attribute__ ((target("arch=zEC12,tune=z196")))
void fn_pta_4 (void) { }
__attribute__ ((target("tune=z196,arch=zEC12")))
void fn_pta_5 (void) { }
#pragma GCC reset_options
