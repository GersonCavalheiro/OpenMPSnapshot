#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef USE_STDCOMPLEX
#include <complex>
#endif
#ifdef PARALLEL_OPENMP
#include <omp.h>
#endif

#include "mandelbrot.hh"
#include "grid.hh"
#include "utils.hh"


#ifdef PARALLEL_MPI
#define WORK_TAG    2
#define FEEBACK_TAG 1
#define END_TAG     0
#endif


MandelbrotSet::MandelbrotSet(int nx, int ny, 
dfloat x_min, dfloat x_max, 
dfloat y_min, dfloat y_max,
int n_iter, int n_rows)
: m_global_nx(nx), m_global_ny(ny), 
m_global_xmin(x_min), m_global_xmax(x_max), 
m_global_ymin(y_min), m_global_ymax(y_max),
m_max_iter(n_iter), m_n_rows(n_rows), 
m_mandel_set(nx, ny) {

m_mod_z2_th = 4.0;
m_value_inside  = 0.0;   
m_value_outside = 100.0; 

m_dx = (m_global_xmax - m_global_xmin) / (dfloat)(m_global_nx - 1);
m_dy = (m_global_ymax - m_global_ymin) / (dfloat)(m_global_ny - 1);

#ifdef USE_DISTANCE_ESTIM
m_dist2_th = 1.0e-6;
#endif
}



#ifdef PARALLEL_MPI

void MandelbrotSet::initialize(MPI_Comm comm) {
m_communicator = comm;

MPI_Comm_rank(comm, &m_prank);
MPI_Comm_size(comm, &m_psize);

m_pdumper = std::unique_ptr<DumperBinary>(
new DumperBinary(m_mandel_set.storage(), comm));
m_ptimer  = std::unique_ptr<Timer>(new TimerMPI(comm));

#if defined(MPI_SIMPLE)
init_mpi_simple();
#elif defined(MPI_MASTER_WORKERS)
init_mpi_writers(0); 
#else
#error "MACRO 'MPI_' UNDEFINED"
#endif

init_dumper_colors();
}

#else

void MandelbrotSet::initialize() {
m_pdumper = std::unique_ptr<DumperASCII>(
new DumperASCII(m_mandel_set.storage()));
m_ptimer  = std::unique_ptr<Timer>(new TimerSTD);

m_local_nx = m_global_nx;
m_local_ny = m_global_ny;
m_local_offset_x = m_local_offset_y = 0;

init_dumper_colors();
}

#endif







void MandelbrotSet::run() {


#if defined(PARALLEL_MPI) && defined(MPI_MASTER_WORKERS)
m_ptimer->start_chrono();


if (m_prank == 0)      compute_master(); 
else if (m_prank != 0) compute_worker(); 

m_ptimer->stop_chrono();
#ifdef OUTPUT_TIMINGS
if (m_prank == 0) cout_timing(m_ptimer->get_timing());
#endif


#elif defined(PARALLEL_MPI) && defined(MPI_SIMPLE)
m_ptimer->start_chrono();


compute_set();

#ifdef OUTPUT_IMAGE
m_pdumper->dump(m_global_nx, m_max_iter);
#endif

m_ptimer->stop_chrono();
#ifdef OUTPUT_TIMINGS
if (m_prank == 0) cout_timing(m_ptimer->get_timing());
#endif


#else
m_ptimer->start_chrono();


compute_set();

#ifdef OUTPUT_IMAGE
m_pdumper->dump(m_global_nx, m_max_iter);
#endif

m_ptimer->stop_chrono();
#ifdef OUTPUT_TIMINGS
cout_timing(m_ptimer->get_timing());
#endif
#endif
}




void MandelbrotSet::compute_set() {


#ifdef PARALLEL_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif

for (int ix = 0; ix < m_local_nx; ix++) {

for (int iy = 0; iy < m_local_ny; iy++) {

compute_pix(ix, iy); 

}
}
}

void MandelbrotSet::compute_pix(int ix, int iy) {
dfloat cx, cy;
dfloat z0x, z0y;


cx = m_global_xmin + (dfloat)(ix + m_local_offset_x) * m_dx;
cy = m_global_ymin + (dfloat)(iy + m_local_offset_y) * m_dy;


z0x = z0y = 0.0;

Grid & mset = m_mandel_set.storage();

mset(ix, iy) = solve_recursive(cx, cy, z0x, z0y);
}

dfloat MandelbrotSet::solve_recursive(dfloat cx, dfloat cy, 
dfloat z0x, dfloat z0y) {
int iter;
dfloat zx, zy;
dfloat mod_z2;
bool diverge;
#ifdef USE_STDCOMPLEX
std::complex<dfloat> c;
std::complex<dfloat> z, z2;
std::complex<dfloat> z_new;
#else
dfloat zx2, zy2;
dfloat zx_new, zy_new;
#endif
#ifdef USE_DISTANCE_ESTIM
dfloat dzx, dzy, dzx_new, dzy_new;
dfloat mod_dz2;
dfloat dist2;
#endif
dfloat pix_value;


#ifdef USE_STDCOMPLEX
c = {cx, cy};
z = {z0x, z0y};
#else
zx = z0x;
zy = z0y;
#endif
mod_z2 = 0.0;
diverge = false;
#ifdef USE_DISTANCE_ESTIM
dzx = dzy = dzx_new = dzy_new = 0.0;
mod_dz2 = 0.0;
dist2 = 0.0;
#endif


for (iter = 1; iter <= m_max_iter; iter++) {


#ifdef USE_STDCOMPLEX
z_new = z*z + c;

#else
zx2 = zx*zx;
zy2 = zy*zy;
zx_new = zx2 - zy2 + cx;
#ifdef SQUARE_TRICK 
zy_new = pow(zx * zy, 2) - zx2 - zy2;
#else
zy_new = zx * zy;
zy_new += zy_new; 
#endif
zy_new += cy;
#endif

#ifdef USE_DISTANCE_ESTIM

#ifdef USE_STDCOMPLEX
zx = z.real();
zy = z.imag();
#endif
dzx_new = zx*dzx - zy*dzy;
dzx_new += dzx_new;
dzx_new += 1.0;
dzy_new = zx*dzy + zy*dzx;
dzy_new += dzy_new;
#endif


#ifdef USE_STDCOMPLEX
z = z_new;
#else
zx = zx_new;
zy = zy_new;
#endif

#ifdef USE_DISTANCE_ESTIM
dzx = dzx_new;
dzy = dzy_new;
#endif

#ifdef USE_STDCOMPLEX
mod_z2 = std::norm(z);
#else
mod_z2 = zx2 + zy2;
#endif


if (mod_z2 > m_mod_z2_th) {

diverge = true;

#ifdef USE_DISTANCE_ESTIM
mod_dz2 = dzx*dzx + dzy*dzy;
dist2   = pow(log(mod_z2), 2) * mod_z2 / mod_dz2;
#endif
break;
}

} 



#ifdef USE_DISTANCE_ESTIM
if ((!diverge) || (dist2 < m_dist2_th)) {
pix_value = m_value_inside;
}
#else
if (!diverge) {
pix_value = m_value_inside;
}
#endif

else {
#ifdef COLOR_ITERATIONS
pix_value = (dfloat) iter;

#elif defined(PARALLEL_MPI) && defined(COLOR_PRANK)
pix_value = m_prank;

#else
pix_value = m_value_outside;
#endif
}

return pix_value;
}


std::vector<int> MandelbrotSet::get_row_def(int row_idx, int nx, int ny, 
int n_rows) {
std::vector<int> sizes(4);

int row_nx = nx / n_rows + (row_idx < nx % n_rows ? 1 : 0);
int row_ny = ny;
int row_offset_x = (nx / n_rows) * row_idx + 
(row_idx < nx % n_rows ? row_idx : nx % n_rows);
int row_offset_y = 0;

sizes[0] = row_nx;
sizes[1] = row_ny;
sizes[2] = row_offset_x;
sizes[3] = row_offset_y;
return sizes;
}


void MandelbrotSet::init_dumper_colors() {
m_pdumper->set_min(m_value_inside);

#ifdef COLOR_ITERATIONS
m_pdumper->set_max(m_max_iter);

#elif defined(PARALLEL_MPI) && defined(COLOR_PRANK)
m_pdumper->set_max(m_psize-1); 

#else
m_pdumper->set_max(m_value_outside);
#endif
}

void MandelbrotSet::cout_timing(double timing) const {

int n_procs   = 1; 
int n_threads = 1; 
int n_rows    = 1; 

#ifdef PARALLEL_MPI
n_procs = m_psize;
#if defined(MPI_MASTER_WORKERS)
n_rows = m_n_rows;
#elif define(MPI_SIMPLE)
n_rows = m_psize;
#endif
#endif

#ifdef PARALLEL_OPENMP
n_threads = omp_get_max_threads();
#endif

std::cout << m_global_nx << " " << m_global_ny << " " 
<< m_max_iter << " " << n_rows << " "
<< n_procs << " " << n_threads << " " 
<< timing << std::endl;
}



#ifdef PARALLEL_MPI

void MandelbrotSet::init_mpi_simple() {
m_n_rows = m_psize;
std::vector<int> locals = get_row_def(m_prank, 
m_global_nx, m_global_ny, m_n_rows);

m_local_nx = locals[0];
m_local_ny = locals[1];
m_local_offset_x = locals[2];
m_local_offset_y = locals[3];

#ifdef VERBOSE
std::cerr << m_prank << " " 
<< m_global_nx << " " << m_global_ny << " " 
<< m_local_nx << " " << m_local_ny << " " 
<< m_local_offset_x << " " << m_local_offset_y << std::endl;
#endif

m_mandel_set.resize(m_local_nx, m_local_ny);
}

void MandelbrotSet::init_mpi_writers(int prank_nonwriter) {
int color = m_prank; 
MPI_Comm_split(m_communicator, color, m_prank, &m_MW_communicator);

if (m_prank != prank_nonwriter) {
m_pdumper->set_mpi_communicator(m_MW_communicator);
}
}

void MandelbrotSet::compute_master() {
MPI_Status status;

int n_workers = m_psize - 1;

std::vector<int> buf_locals(4);

if (n_workers > m_n_rows)
throw std::invalid_argument("Too much workers processors !");

#ifdef VERBOSE
std::cerr << "> " << n_workers << " workers" << std::endl;
#endif

int w_prank; 
int n_busy  = 0; 
int row_idx = 0; 


for (w_prank = 1; w_prank <= n_workers; w_prank++) {
buf_locals = get_row_def(row_idx, m_global_nx, m_global_ny, m_n_rows);

MPI_Send(&buf_locals[0], 4, MPI_INT, w_prank, WORK_TAG, m_communicator);
n_busy++;
row_idx++;
}


int w_feeback;
for (;;) {

MPI_Recv(&w_feeback, 1, MPI_INT, 
MPI_ANY_SOURCE, FEEBACK_TAG, m_communicator, &status);
w_prank = status.MPI_SOURCE;


if (row_idx < m_n_rows) {

buf_locals = get_row_def(row_idx, m_global_nx, m_global_ny, m_n_rows);
MPI_Send(&buf_locals[0], 4, MPI_INT, w_prank, WORK_TAG, m_communicator);

row_idx++;
}


else {
MPI_Send(&buf_locals[0], 4, MPI_INT, w_prank, END_TAG, m_communicator);
n_busy--;
}

if (n_busy == 0) {
break; 
}

} 
}

void MandelbrotSet::compute_worker() {
int w_feeback;
MPI_Status status;

std::vector<int> buf_locals(4);


for (;;) {
MPI_Recv(&buf_locals[0], 4, MPI_INT, 
0, MPI_ANY_TAG, m_communicator, &status);

if (status.MPI_TAG == END_TAG) {
break;
}

m_local_nx       = buf_locals[0];
m_local_ny       = buf_locals[1];
m_local_offset_x = buf_locals[2];
m_local_offset_y = buf_locals[3];

#ifdef VERBOSE
std::cerr << m_prank << " " 
<< m_global_nx << " " << m_global_ny << " "
<< m_local_nx << " " << m_local_ny << " " 
<< m_local_offset_x << " " << m_local_offset_y << std::endl;
#endif

m_mandel_set.resize(m_local_nx, m_local_ny);


compute_set();

#ifdef OUTPUT_IMAGE
m_pdumper->dump_manual(m_global_nx, m_max_iter, 
m_local_offset_x, m_global_nx);
#endif


MPI_Send(&w_feeback, 1, MPI_INT, 0, FEEBACK_TAG, m_communicator);

} 

}

#endif 

