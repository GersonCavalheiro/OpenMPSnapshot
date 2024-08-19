#pragma once

#include <cstdio>     
#include <cstdint>    
#include <cassert>    
#include <cstring>    
#include <cmath>      
#include <algorithm>  
#include <vector>     
#include <cstdlib>    

#include "status.hxx" 
#include "simple_timer.hxx" 

#ifdef  HAS_RAPIDXML
#include "rapidxml/rapidxml.hpp" 
#include "rapidxml/rapidxml_utils.hpp" 

#include "xml_reading.hxx" 
#endif 

#include "unit_system.hxx" 
#include "print_tools.hxx" 
#include "control.hxx" 



namespace pawxml_import {

struct pawxmlstate_t {
std::vector<double> tsp[3];
double e;
float f, rc;
int8_t n, l;
}; 

struct pawxml_t {
double Z, core, valence;
double ae_energy[4];
double core_energy_kinetic;
double radial_grid_a;
int    n; 
double shape_function_rc;
std::vector<pawxmlstate_t> states;
std::vector<double> func[6]; 
std::vector<double> dkin; 
char xc[16];
char Sy[4];
status_t parse_status;
}; 

char const radial_state_quantities[3][20] = {"ae_partial_wave", "pseudo_partial_wave", "projector_function"};

char const radial_quantities[6][40] = {"ae_core_density", "pseudo_core_density",
"ae_core_kinetic_energy_density", "pseudo_core_kinetic_energy_density", 
"zero_potential", "pseudo_valence_density"};

inline pawxml_t parse_pawxml(char const *filename, int const echo=0) {
#ifndef HAS_RAPIDXML
warn("Unable to test GPAW loading when compiled without -D HAS_RAPIDXML", 0);
return pawxml_t();
#else  

pawxml_t p;

if (echo > 9) std::printf("# %s file=%s\n", __func__, filename);
rapidxml::xml_document<> doc;
try {
rapidxml::file<> infile(filename);
try {
doc.parse<0>(infile.data());
} catch (...) {
warn("failed to parse \"%s\"", filename);
p.parse_status = 2; return p; 
} 
} catch (...) {
warn("failed to open \"%s\"", filename);
p.parse_status = 1; return p; 
} 

auto const paw_setup = doc.first_node("paw_setup");
if (!paw_setup) return p; 

auto const atom = xml_reading::find_child(paw_setup, "atom", echo);
if (atom) {
auto const symbol  = xml_reading::find_attribute(atom, "symbol", "?_", echo);
auto const Z       = xml_reading::find_attribute(atom, "Z",      "-9", echo);
auto const core    = xml_reading::find_attribute(atom, "core",    "0", echo);
auto const valence = xml_reading::find_attribute(atom, "valence", "0", echo);
if (echo > 5) std::printf("# %s:  <atom symbol=\"%s\" Z=\"%s\" core=\"%s\" valence=\"%s\"/>\n",
filename, symbol, Z, core, valence);
p.Z       = std::atof(Z);
p.core    = std::atof(core);
p.valence = std::atof(valence);
std::snprintf(p.Sy, 3, "%s", symbol);
} else warn("no <atom> found in xml-file '%s'", filename);

auto const xc_functional = xml_reading::find_child(paw_setup, "xc_functional", echo);
if (xc_functional) {
auto const type = xml_reading::find_attribute(xc_functional, "type", "?type");
auto const name = xml_reading::find_attribute(xc_functional, "name", "?name");
if (echo > 5) std::printf("# %s:  <xc_functional type=\"%s\" name=\"%s\"/>\n", filename, type, name);
std::snprintf(p.xc, 15, "%s", name); 
} else warn("no <xc_functional> found in xml-file '%s'", filename);

auto const generator = xml_reading::find_child(paw_setup, "generator", echo);
if (generator) {
auto const type = xml_reading::find_attribute(generator, "type", "?type");
auto const name = xml_reading::find_attribute(generator, "name", "?name");
if (echo > 5) std::printf("# %s:  <generator type=\"%s\" name=\"%s\"> ... </generator>\n",
filename, type, name);
} else warn("no <generator> found in xml-file '%s'", filename);

auto const ae_energy = xml_reading::find_child(paw_setup, "ae_energy", echo);
if (ae_energy) {
auto const kinetic       = xml_reading::find_attribute(ae_energy, "kinetic",       "0");
auto const xc            = xml_reading::find_attribute(ae_energy, "xc",            "0");
auto const electrostatic = xml_reading::find_attribute(ae_energy, "electrostatic", "0");
auto const total         = xml_reading::find_attribute(ae_energy, "total",         "0");
if (echo > 5) std::printf("# %s:  <ae_energy kinetic=\"%s\" xc=\"%s\" electrostatic=\"%s\" total=\"%s\"/>\n",
filename, kinetic, xc, electrostatic, total);
p.ae_energy[0] = std::atof(kinetic);
p.ae_energy[1] = std::atof(xc);
p.ae_energy[2] = std::atof(electrostatic);
p.ae_energy[3] = std::atof(total);
} else warn("no <ae_energy> found in xml-file '%s'", filename);

auto const core_energy = xml_reading::find_child(paw_setup, "core_energy", echo);
if (core_energy) {
auto const kinetic = xml_reading::find_attribute(core_energy, "kinetic",       "0");
if (echo > 5) std::printf("# %s:  <core_energy kinetic=\"%s\"/>\n", filename, kinetic);
p.core_energy_kinetic = std::atof(kinetic); 
} else warn("no <core_energy> found in xml-file '%s'", filename);

p.n = 0;
auto const radial_grid = xml_reading::find_child(paw_setup, "radial_grid", echo);
if (radial_grid) {
auto const eq     = xml_reading::find_attribute(radial_grid, "eq", "?eq");
auto const a      = xml_reading::find_attribute(radial_grid, "a", ".4");
auto const n      = xml_reading::find_attribute(radial_grid, "n", "0");
auto const istart = xml_reading::find_attribute(radial_grid, "istart", "0");
auto const iend   = xml_reading::find_attribute(radial_grid, "iend", "-1");
auto const id     = xml_reading::find_attribute(radial_grid, "id", "?");
if (echo > 5) std::printf("# %s:  <radial_grid eq=\"%s\" a=\"%s\" n=\"%s\" istart=\"%s\" iend=\"%s\" id=\"%s\"/>\n",
filename, eq, a, n, istart, iend, id);
p.n = std::atoi(n); 
p.radial_grid_a = std::atof(a); 
if (0 != std::strcmp(eq, "r=a*i/(n-i)")) error("%s: assume a radial expontential grid", filename);
if (0 != std::atoi(istart))              error("%s: assume a radial grid starting at 0", filename);
if (p.n != std::atoi(iend) + 1)          error("%s: assume a radial grid starting from 0 to n-1", filename);
} else warn("no <radial_grid> found in xml-file '%s'", filename);

int istate{0};
p.states.resize(0);
int nwarn[] = {0, 0, 0};
auto const valence_states = xml_reading::find_child(paw_setup, "valence_states", echo);
if (valence_states) {
if (echo > 5) std::printf("# %s:  <valence_states>\n", filename);
for (auto state = valence_states->first_node(); state; state = state->next_sibling()) {
auto const n  = xml_reading::find_attribute(state, "n",  "0");
auto const l  = xml_reading::find_attribute(state, "l", "-1");
auto const f  = xml_reading::find_attribute(state, "f",  "0");
auto const rc = xml_reading::find_attribute(state, "rc", "0");
auto const e  = xml_reading::find_attribute(state, "e",  "0");
auto const id = xml_reading::find_attribute(state, "id", "?");
if (echo > 5) std::printf("# %s:    <state n=\"%s\" l=\"%s\" f=\"%s\" rc=\"%s\" e=\"%s\" id=\"%s\"/>\n",
filename, n, l, f, rc, e, id);

pawxmlstate_t s;
s.n  = std::atoi(n);
s.l  = std::atoi(l);
s.f  = std::atof(f);
s.rc = std::atof(rc);
s.e  = std::atof(e);

for (int iq = 0; iq < 3; ++iq) {
auto const q_name = radial_state_quantities[iq];
int radial_data_found{0};
for (auto child = paw_setup->first_node(); child; child = child->next_sibling()) {
if (0 == std::strcmp(q_name, child->name())) {
auto const state_id = xml_reading::find_attribute(child, "state", "?state");
if (0 == std::strcmp(state_id, id)) {
auto const grid = xml_reading::find_attribute(child, "grid", "?grid");
auto const vals = xml_reading::read_sequence<double>(child->value(), echo, p.n);
if (echo > 8) std::printf("# %s:  <%s state=\"%s\" grid=\"%s\"> ...(%ld numbers)... </%s>\n",
filename, q_name, state_id, grid, vals.size(), q_name);
nwarn[iq] += (vals.size() != p.n);
s.tsp[iq] = vals;
++radial_data_found;
} 
} 
} 
if (1 != radial_data_found) error("%s: radial state quantities in pawxml files must be defined exactly once!", filename);
} 
p.states.push_back(s); 
++istate;

} 
if (echo > 5) std::printf("# %s:  </valence_states>\n", filename);
} else warn("no <valence_states> found in xml-file '%s'", filename);
int const nstates = istate;
assert(nstates == p.states.size());

for (int iq = 0; iq < 3; ++iq) {
if (nwarn[iq]) {
warn("%s: %d %s deviate from the expected number of %d grid points",
filename, nwarn[iq], radial_state_quantities[iq], p.n);
} 
} 

auto const shape_function = xml_reading::find_child(paw_setup, "shape_function", echo);
if (shape_function) {
auto const type = xml_reading::find_attribute(shape_function, "type", "?type");
auto const rc   = xml_reading::find_attribute(shape_function, "rc", "0");
if (echo > 5) std::printf("# %s:  <shape_function type=\"%s\" rc=\"%s\"/>\n", filename, type, rc);
if (std::strcmp(type, "gauss")) error("%s: assume a shape_function type gauss", filename);
p.shape_function_rc = std::atof(rc); 
} else warn("no <shape_function> found in xml-file '%s'", filename);

for (int iq = 0; iq < 5; ++iq) {
auto const q_name = radial_quantities[iq];
auto const q_node = xml_reading::find_child(paw_setup, q_name, echo);
if (q_node) {
auto const grid = xml_reading::find_attribute(q_node, "grid", "?grid");
auto const vals = xml_reading::read_sequence<double>(q_node->value(), echo, p.n);
if (echo > 7) std::printf("# %s:  <%s grid=\"%s\"> ...(%ld numbers)... </%s>\n",
filename, q_name, grid, vals.size(), q_name);
if (vals.size() != p.n) warn("%s: %s has %ld but expected %d grid points",
filename, q_name, vals.size(), p.n);
p.func[iq] = vals; 
} else warn("no <%s> found in xml-file '%s'", q_name, filename);
} 

auto const kinetic_energy_differences = xml_reading::find_child(paw_setup, "kinetic_energy_differences", echo);
if (kinetic_energy_differences) {
auto const vals = xml_reading::read_sequence<double>(kinetic_energy_differences->value(), echo, nstates*nstates);
if (echo > 7) std::printf("# %s:  <kinetic_energy_differences> ...(%ld numbers)... </kinetic_energy_differences>\n", filename, vals.size());
if (vals.size() != nstates*nstates) warn("%s: expected %d^2 numbers in kinetic_energy_differences matrix but found %ld", filename, nstates, vals.size());
p.dkin = vals; 
} else warn("no <kinetic_energy_differences> found in xml-file '%s'", filename);


p.parse_status = 0;
return p;
#endif 
} 



#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_loading(int const echo=0) {
SimpleTimer timer(__FILE__, __LINE__, __func__, echo);

char const *filename = control::get("pawxml_import.test.filename", "C.LDA");
if (echo > 7) std::printf("# %s file=%s\n", __func__, filename);
auto const p = parse_pawxml(filename, echo);

int const repeat_file = control::get("pawxml_import.test.repeat", 0.);
if (repeat_file) {
char outfile[992]; std::snprintf(outfile, 991, "%s.repeat.xml", filename);
auto const f = std::fopen(outfile, "w");
std::fprintf(f, "<?xml version=\"1.0\"?>\n"
"<paw_setup version=\"0.6\">\n"
"\t<!-- Element setup for the Projector Augmented Wave method -->\n"
"\t<!-- Units: Hartree and Bohr radii.                        -->\n");
std::fprintf(f, "\t<atom symbol=\"%s\" Z=\"%g\" core=\"%g\" valence=\"%g\"/>\n", p.Sy, p.Z, p.core, p.valence);
std::fprintf(f, "\t<xc_functional name=\"%s\"/>\n", p.xc);
std::fprintf(f, "\t<ae_energy kinetic=\"%.6f\" xc=\"%.6f\" electrostatic=\"%.6f\" total=\"%.6f\"/>\n",
p.ae_energy[0], p.ae_energy[1], p.ae_energy[2], p.ae_energy[3]);
std::fprintf(f, "\t<core_energy kinetic=\"%.6f\"/>\n", p.core_energy_kinetic);
std::fprintf(f, "\t<valence_states>\n");
char const ellchar[8] = "spdfghi";
for (auto & s : p.states) {
char id[8]; std::snprintf(id, 8, "%s-%d%c", p.Sy, s.n, ellchar[s.l]);
std::fprintf(f, "\t\t<state n=\"%d\" l=\"%d\" f=\"%g\" rc=\"%g\" e=\"%g\" id=\"%s\"/>\n",
s.n,     s.l,     s.f,     s.rc,     s.e,       id);
} 
std::fprintf(f, "\t</valence_states>\n");
std::fprintf(f, "\t<radial_grid eq=\"r=a*i/(n-i)\" a=\"%g\" n=\"%d\" istart=\"0\" iend=\"%d\" id=\"g1\"/>\n",
p.radial_grid_a,     p.n,                     p.n - 1);
std::fprintf(f, "\t<shape_function type=\"gauss\" rc=\"%.12e\"/>\n", p.shape_function_rc);

for (int iq = 0; iq < 5; ++iq) {
std::fprintf(f, "\t<%s grid=\"g1\">\n", radial_quantities[iq]);
for (auto val : p.func[iq]) {
std::fprintf(f, " %g", val);
} 
std::fprintf(f, "\n\t</%s>\n", radial_quantities[iq]);
} 

for (auto const & s : p.states) {
char id[8]; std::snprintf(id, 7, "%s-%d%c", p.Sy, s.n, ellchar[s.l]);
for (int iq = 0; iq < 3; ++iq) {
std::fprintf(f, "\t<%s state=\"%s\" grid=\"g1\">\n", radial_state_quantities[iq], id);
for (auto val : s.tsp[iq]) {
std::fprintf(f, " %g", val);
} 
std::fprintf(f, "\n\t</%s>\n", radial_state_quantities[iq]);
} 
} 

{
std::fprintf(f, "\t<kinetic_energy_differences>");
int const n = p.states.size();
for (int i = 0; i < n; ++i) {
for (int j = 0; j < n; ++j) {
std::fprintf(f, "%s%g", j?" ":"\n\t\t", p.dkin[i*n + j]);
} 
} 
}
std::fprintf(f, "\n\t</kinetic_energy_differences>\n");
std::fprintf(f, "</paw_setup>\n");
std::fclose(f);
} 

return p.parse_status;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_loading(echo);
return stat;
} 

#endif 

} 
