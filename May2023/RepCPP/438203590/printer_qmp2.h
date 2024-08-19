#ifndef LIBQQC_PRINTER_QMP2_H
#define LIBQQC_PRINTER_QMP2_H

#if LIBQQC_WITH_OPENMP
#include <omp.h>
#endif

#if LIBQQC_WITH_MPI
#include <mpi.h>
#endif

#include <string>

#include "../ttimer.h"
#include "../../vaults/vault_qmp2.h"

using namespace std;

namespace libqqc {


class Printer_qmp2 {
private:
size_t mwidth = 80; 
string mmethod_name = "Q-MP(2)"; 
string mauthors = "Benjamin Thomitzni"; 

string mlibqqc_vers = "v0.1"; 
string mloader_vers = "v0.1"; 
string mvault_vers = "v0.1"; 
string mdoqmp2_vers = "v0.1"; 

bool mb_openmp = false; 
size_t mnthreads = 0; 

bool mb_mpi = false; 
size_t mnprocs = 0; 

size_t mnao = 0; 
size_t mnmo = 0; 
size_t mnocc = 0; 
size_t mnvirt = 0; 
size_t m3Dnpts = 0; 
size_t m1Dnpts = 0; 
int mprnt_lvl = 0; 

Ttimer &mtimings; 

public: 
Printer_qmp2 (Vault_qmp2 &vault, Ttimer &timings) 
: mnao(vault.get_mnao()), mnmo(vault.get_mnmo()), 
mnocc(vault.get_mnocc()), mnvirt(vault.get_mnvirt()),
m3Dnpts(vault.get_m3Dgrid().get_mnpts()),
m1Dnpts(vault.get_m1Dgrid().get_mnpts()),
mprnt_lvl(vault.get_mprnt_lvl()),
mtimings(timings) {
#if LIBQQC_WITH_OPENMP
mb_openmp = true;
#pragma omp parallel
{
#pragma omp single
mnthreads = omp_get_num_threads();
}
#endif

#if LIBQQC_WITH_MPI
mb_mpi = true;
int max_id; 
MPI_Comm_size(MPI_COMM_WORLD, &max_id);
mnprocs = max_id;
#endif
};
int member_fn(int param);
void print_openmp(ostringstream &out){
out << make_line("* OpenMP enabled", 
'l') << endl
<< make_line("    -- threads:   " + to_string(mnthreads),
'l')
<< endl; 
};
void print_mpi(ostringstream &out){
out << make_line("* MPI enabled", 
'l') << endl
<< make_line("    -- processes: " + to_string(mnprocs), 'l')
<< endl;
};
void print_method_qmp2(ostringstream &out){
out << make_line("Calculation Details", 'm') << endl
<< make_full_line('-') << endl
<< make_line("* Method:             " + mmethod_name, 'l') 
<< endl
<< make_full_line('-') << endl
<< make_line("* grid points (3D):   " + to_string(m3Dnpts)
, 'l') << endl
<< make_line("* grid points (1D):   " + to_string(m1Dnpts)
, 'l') << endl
<< make_full_line('-') << endl
<< make_line("* atomic orbitals:    " + to_string(mnao), 
'l') << endl
<< make_line("* molecular orbitals: " + to_string(mnmo), 
'l') << endl
<< make_line("    -- occupied:      " + to_string(mnocc)
, 'l') << endl
<< make_line("    -- virtual:       " + to_string(mnvirt)
, 'l') << endl
<< make_full_line('-') << endl;
};
string make_line(string in, char align){
size_t length_in = in.length();
string out = "";

size_t space = ((length_in < (mwidth - 2)) ? 
((mwidth - 2) - length_in) : 0); 
string left = "";
string right = "";

if (align == 'l'){
left = ((space > 0 ) ? " " : "" );
space--;
for (size_t i = 0; i < space; i++){
right += " ";
}
} else if (align == 'r'){
left = ((space > 0 ) ? " " : "" );
space--;
for (size_t i = 0; i < space; i++){
left += " ";
}

} else {
for (size_t i = space; i > 1; i -= 2){
left += " ";
right += " ";
}
left += ((space % 2) == 1) ? " " : "";
}

out = "|" + left + in + right + "|";
return out;
};
string make_full_line(char c){
string out = "";
for (int i = mwidth - 2; i > 0; i--){
out += c;
}

out = "|" + out + "|";
return out;
};
string make_hdr_ftr(bool type){
string out = "";

for (int i = mwidth - 2; i > 0; i--){
out += "_";
}

out = ((type) ? " " : "|") + out + ((type) ? " " : "|");

return out;

};

void print_final(ostringstream &out){
string before = out.str();
out.str("");

out << make_hdr_ftr(1) << endl
<< make_full_line(' ') << endl
<< make_line("** Quadrature Calculation through libqqc **", 
'm') << endl
<< make_full_line(' ') << endl
<< make_full_line('+') << endl
<< make_line("* Author(s): " + mauthors, 'l') << endl
<< make_full_line('+') << endl;

out << make_line("Programm Details", 'm') << endl
<< make_full_line('-') 
<< endl
<< make_line("* library vers.      " + mlibqqc_vers, 'l') 
<< endl
<< make_line("    -- Loader vers.  " + mloader_vers, 'l') 
<< endl
<< make_line("    -- Vault vers.   " + mvault_vers, 'l') 
<< endl
<< make_line("    -- Do_qmp2 vers. " + mdoqmp2_vers, 'l') 
<< endl;

if (mb_openmp || mb_mpi) out << make_full_line('-') << endl;
if (mb_openmp) print_openmp(out);
if (mb_mpi) print_mpi(out);
out << make_full_line('+') << endl;

print_method_qmp2(out);

out << make_line("Results", 'm') << endl
<< make_full_line('-') << endl
<< make_line(before, 'l') << endl;

if (mprnt_lvl != 0) {
out << make_full_line('+') << endl
<< make_line("Calculation Timings", 'm') << endl
<< make_full_line('-') << endl;

stringstream ss;
ss << mtimings.print_all_clocks();

for (string line; getline(ss, line, '\n');)
out << make_line(line, 'l') << endl;

out << make_full_line('-') << endl
<< make_line("if no timings are printed,"
" choose higher print level", 'm') 
<< endl;
}

out << make_hdr_ftr(0) << endl;
};
}; 

} 

#endif 
