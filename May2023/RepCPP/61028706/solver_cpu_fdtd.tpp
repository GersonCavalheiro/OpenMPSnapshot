

#ifndef MBSOLVE_SOLVER_CPU_FDTD_TPP
#define MBSOLVE_SOLVER_CPU_FDTD_TPP

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_NO_MALLOC
#define EIGEN_STRONG_INLINE inline

#include <iostream>
#include <omp.h>
#include <mbsolve/solver-cpu/internal/common_cpu.hpp>
#include <mbsolve/solver-cpu/solver_cpu_fdtd.hpp>

namespace mbsolve {

template<unsigned int num_lvl, template<unsigned int> class density_algo>
solver_cpu_fdtd<num_lvl, density_algo>::solver_cpu_fdtd(
std::shared_ptr<const device> dev,
std::shared_ptr<scenario> scen)
: solver(
"cpu-fdtd-" + std::to_string(num_lvl) + "lvl-" +
density_algo<num_lvl>::name(),
dev,
scen)
{






unsigned int P = omp_get_max_threads();
std::cout << "Number of threads: " << P << std::endl;

if (dev->get_regions().size() == 0) {
throw std::invalid_argument("No regions in device!");
}


init_fdtd_simulation(dev, scen, 0.5);


m_dx_inv = 1.0 / scen->get_gridpoint_size();


std::vector<sim_constants_fdtd> m_sim_consts_fdtd;
std::map<std::string, unsigned int> id_to_idx;
unsigned int j = 0;
for (const auto& mat_id : dev->get_used_materials()) {
auto mat = material::get_from_library(mat_id);


m_sim_consts_fdtd.push_back(get_fdtd_constants(dev, scen, mat));


m_sim_consts_qm.push_back(density_algo<num_lvl>::get_qm_constants(
mat->get_qm(), scen->get_timestep_size()));


id_to_idx[mat->get_id()] = j;
j++;
}


typedef typename density_algo<num_lvl>::density density_t;
m_d = new density_t[scen->get_num_gridpoints()];

m_h = (real*) mb_aligned_alloc(
sizeof(real) * (scen->get_num_gridpoints() + 1));
m_e = (real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());
m_p = (real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());

m_fac_a =
(real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());
m_fac_b =
(real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());
m_fac_c =
(real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());
m_gamma =
(real*) mb_aligned_alloc(sizeof(real) * scen->get_num_gridpoints());

m_mat_indices = (unsigned int*) mb_aligned_alloc(
sizeof(unsigned int) * scen->get_num_gridpoints());


#pragma omp parallel for schedule(static)
for (int i = 0; i < scen->get_num_gridpoints(); i++) {

int idx = -1;
bool has_qm = false;
real x = i * scen->get_gridpoint_size();
for (const auto& reg : dev->get_regions()) {
if ((x >= reg->get_x_start()) && (x <= reg->get_x_end())) {
idx = id_to_idx[reg->get_material()->get_id()];
has_qm = (reg->get_material()->get_qm()) ? true : false;
break;
}
}

if ((idx < 0) || (idx >= dev->get_used_materials().size())) {
std::cout << "At index " << i << std::endl;
throw std::invalid_argument("region not found");
}


m_fac_a[i] = m_sim_consts_fdtd[idx].fac_a;
m_fac_b[i] = m_sim_consts_fdtd[idx].fac_b;
m_fac_c[i] = m_sim_consts_fdtd[idx].fac_c;
m_gamma[i] = m_sim_consts_fdtd[idx].gamma;
m_mat_indices[i] = idx;


if (has_qm) {
auto ic_dm = scen->get_ic_density();
m_d[i] = density_algo<num_lvl>::get_density(ic_dm->initialize(x));
} else {
m_d[i] = density_algo<num_lvl>::get_density();
}
auto ic_e = scen->get_ic_electric();
auto ic_h = scen->get_ic_magnetic();
m_e[i] = ic_e->initialize(x);
m_h[i] = ic_h->initialize(x);
if (i == scen->get_num_gridpoints() - 1) {
m_h[i + 1] = 0.0;
}
m_p[i] = 0.0;
}


unsigned int scratch_size = 0;
for (const auto& rec : scen->get_records()) {

copy_list_entry entry(rec, scen, scratch_size);


m_results.push_back(entry.get_result());


scratch_size += entry.get_size();


if (rec->is_complex()) {
scratch_size += entry.get_size();
}




m_copy_list.push_back(entry);
}


m_result_scratch = (real*) mb_aligned_alloc(sizeof(real) * scratch_size);


m_source_data =
new real[scen->get_num_timesteps() * scen->get_sources().size()];
unsigned int base_idx = 0;
for (const auto& src : scen->get_sources()) {
sim_source s;
s.type = src->get_type();
s.x_idx = src->get_position() / scen->get_gridpoint_size();
s.data_base_idx = base_idx;
m_sim_sources.push_back(s);


for (unsigned int j = 0; j < scen->get_num_timesteps(); j++) {
m_source_data[base_idx + j] =
src->get_value(j * scen->get_timestep_size());
}

base_idx += scen->get_num_timesteps();
}
}

template<unsigned int num_lvl, template<unsigned int> class density_algo>
solver_cpu_fdtd<num_lvl, density_algo>::~solver_cpu_fdtd()
{
mb_aligned_free(m_fac_a);
mb_aligned_free(m_fac_b);
mb_aligned_free(m_fac_c);
mb_aligned_free(m_gamma);

mb_aligned_free(m_h);
mb_aligned_free(m_e);
mb_aligned_free(m_p);

mb_aligned_free(m_mat_indices);
mb_aligned_free(m_result_scratch);
delete[] m_source_data;
delete[] m_d;
}

template<unsigned int num_lvl, template<unsigned int> class density_algo>
void
solver_cpu_fdtd<num_lvl, density_algo>::run() const
{
#pragma omp parallel
{

for (int n = 0; n < m_scenario->get_num_timesteps(); n++) {


#if USE_OMP_SIMD
#pragma omp for simd schedule(static)
#else
#pragma omp for schedule(static)
#endif
for (int i = 0; i < m_scenario->get_num_gridpoints(); i++) {
m_e[i] = m_fac_a[i] * m_e[i] +
m_fac_b[i] *
(-m_gamma[i] * m_p[i] +
(m_h[i + 1] - m_h[i]) * m_dx_inv);
}



for (const auto& src : m_sim_sources) {

if (src.type == source::type::hard_source) {
m_e[src.x_idx] = m_source_data[src.data_base_idx + n];
} else if (src.type == source::type::soft_source) {
m_e[src.x_idx] += m_source_data[src.data_base_idx + n];
} else {
}
}


#if USE_OMP_SIMD
#pragma omp for simd nowait schedule(static)
#else
#pragma omp for nowait schedule(static)
#endif
for (int i = 1; i < m_scenario->get_num_gridpoints(); i++) {
m_h[i] += m_fac_c[i] * (m_e[i] - m_e[i - 1]);
}




m_h[0] = 0;
m_h[m_scenario->get_num_gridpoints()] = 0;


#pragma omp for schedule(static)
for (int i = 0; i < m_scenario->get_num_gridpoints(); i++) {
unsigned int mat_idx = m_mat_indices[i];


density_algo<num_lvl>::update(
m_sim_consts_qm[mat_idx], m_d[i], m_e[i], &m_p[i]);
}


for (const auto& cle : m_copy_list) {
if (cle.hasto_record(n)) {
uint64_t pos = cle.get_position();
uint64_t cols = cle.get_cols();
record::type t = cle.get_type();
uint64_t o_r = cle.get_offset_scratch_real(n, 0);
unsigned int cidx = cle.get_col_idx();
unsigned int ridx = cle.get_row_idx();

#pragma omp for schedule(static)
for (int i = pos; i < pos + cols; i++) {
if (t == record::type::electric) {
m_result_scratch[o_r + i - pos] = m_e[i];
} else if (t == record::type::polar_dt) {
m_result_scratch[o_r + i - pos] = m_p[i];
} else if (t == record::type::magnetic) {
m_result_scratch[o_r + i - pos] = m_h[i];
} else if (t == record::type::inversion) {
m_result_scratch[o_r + i - pos] =
density_algo<num_lvl>::calc_inversion(m_d[i]);
} else if (t == record::type::density) {

if (ridx == cidx) {
m_result_scratch[o_r + i - pos] =
density_algo<num_lvl>::calc_population(
m_d[i], ridx);
} else {



}
} else {

}
}

}
}
}
}


for (const auto& cle : m_copy_list) {
real* dr = m_result_scratch + cle.get_offset_scratch_real(0, 0);
std::copy(dr, dr + cle.get_size(), cle.get_result_real(0, 0));
if (cle.is_complex()) {
real* di = m_result_scratch + cle.get_offset_scratch_imag(0, 0);
std::copy(di, di + cle.get_size(), cle.get_result_imag(0, 0));
}
}
}
}

#endif
