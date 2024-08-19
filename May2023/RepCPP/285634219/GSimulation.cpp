
#include "GSimulation.hpp"


GSimulation::GSimulation() {
std::cout << "==============================="
<< "\n";
std::cout << " Initialize Gravity Simulation"
<< "\n";
set_npart(16000);
set_nsteps(10);
set_tstep(0.1);
set_sfreq(1);
}


void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }


void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }


void GSimulation::InitPos() {
std::mt19937 gen(42);
std::uniform_real_distribution<RealType> unif_d(0, 1.0);

for (int i = 0; i < get_npart(); ++i) {
particles_[i].pos[0] = unif_d(gen);
particles_[i].pos[1] = unif_d(gen);
particles_[i].pos[2] = unif_d(gen);
}
}


void GSimulation::InitVel() {
std::mt19937 gen(42);
std::uniform_real_distribution<RealType> unif_d(-1.0, 1.0);

for (int i = 0; i < get_npart(); ++i) {
particles_[i].vel[0] = unif_d(gen) * 1.0e-3f;
particles_[i].vel[1] = unif_d(gen) * 1.0e-3f;
particles_[i].vel[2] = unif_d(gen) * 1.0e-3f;
}
}


void GSimulation::InitAcc() {
for (int i = 0; i < get_npart(); ++i) {
particles_[i].acc[0] = 0.f;
particles_[i].acc[1] = 0.f;
particles_[i].acc[2] = 0.f;
}
}


void GSimulation::InitMass() {
RealType n = static_cast<RealType>(get_npart());
std::mt19937 gen(42);
std::uniform_real_distribution<RealType> unif_d(0.0, 1.0);

for (int i = 0; i < get_npart(); ++i) {
particles_[i].mass = n * unif_d(gen);
}
}


void GSimulation::Start() {
RealType dt = get_tstep();
int n = get_npart();
std::vector<RealType> energy(n, 0.f);
particles_.resize(n);

InitPos();
InitVel();
InitAcc();
InitMass();

#ifdef DEBUG
PrintHeader();
#endif

total_time_ = 0.;

constexpr float kSofteningSquared = 1e-3f;
constexpr float kG = 6.67259e-11f;
double gflops = 1e-9 * ((11. + 18.) * n * n + n * 19.);
int nf = 0;
double av = 0.0, dev = 0.0;

Particle *p = particles_.data();
RealType *e = energy.data();

TimeInterval t0;
int nsteps = get_nsteps();

#pragma omp target data map (to: p[0:n]) map(alloc: e[0:n])
{
for (int s = 1; s <= nsteps; ++s) {
TimeInterval ts0;

#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < n; i++) {
auto pi = p[i];
RealType acc0 = pi.acc[0];
RealType acc1 = pi.acc[1];
RealType acc2 = pi.acc[2];

for (int j = 0; j < n; j++) {
RealType dx, dy, dz;
RealType distance_sqr = 0.0f;
RealType distance_inv = 0.0f;

auto pj = p[j];
dx = pj.pos[0] - pi.pos[0];  
dy = pj.pos[1] - pi.pos[1];  
dz = pj.pos[2] - pi.pos[2];  

distance_sqr =
dx * dx + dy * dy + dz * dz + kSofteningSquared;  
distance_inv = 1.0f / sqrtf(distance_sqr);       

acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;  
acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;  
acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;  
}
pi.acc[0] = acc0;
pi.acc[1] = acc1;
pi.acc[2] = acc2;
p[i] = pi;
}

#pragma omp target teams distribute parallel for thread_limit(256)
for (int i = 0; i < n; i++) {
auto pi = p[i];
pi.vel[0] += pi.acc[0] * dt;  
pi.vel[1] += pi.acc[1] * dt;  
pi.vel[2] += pi.acc[2] * dt;  

pi.pos[0] += pi.vel[0] * dt;  
pi.pos[1] += pi.vel[1] * dt;  
pi.pos[2] += pi.vel[2] * dt;  

pi.acc[0] = 0.f;
pi.acc[1] = 0.f;
pi.acc[2] = 0.f;

e[i] = pi.mass *
(pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
pi.vel[2] * pi.vel[2]);  

p[i] = pi;
}
#pragma omp target 
for (int i = 1; i < n; i++) e[0] += e[i];

double elapsed_seconds = ts0.Elapsed();

#pragma omp target update from (e[0:1])
kenergy_ = 0.5 * e[0];
e[0] = 0;
if ((s % get_sfreq()) == 0) {
nf += 1;
#ifdef DEBUG
std::cout << " " << std::left << std::setw(8) << s << std::left
<< std::setprecision(5) << std::setw(8) << s * get_tstep()
<< std::left << std::setprecision(5) << std::setw(12)
<< kenergy_ << std::left << std::setprecision(5)
<< std::setw(12) << elapsed_seconds << std::left
<< std::setprecision(5) << std::setw(12)
<< gflops * get_sfreq() / elapsed_seconds << "\n";
#endif
if (nf > 2) {
av += gflops * get_sfreq() / elapsed_seconds;
dev += gflops * get_sfreq() * gflops * get_sfreq() /
(elapsed_seconds * elapsed_seconds);
}
}
}  
}
total_time_ = t0.Elapsed();
total_flops_ = gflops * get_nsteps();
av /= (double)(nf - 2);
dev = sqrt(dev / (double)(nf - 2) - av * av);

std::cout << "\n";
std::cout << "# Total Energy        : " << kenergy_ << "\n";
std::cout << "# Total Time (s)      : " << total_time_ << "\n";
std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
std::cout << "===============================" << "\n";
}

#ifdef DEBUG

void GSimulation::PrintHeader() {
std::cout << " nPart = " << get_npart() << "; "
<< "nSteps = " << get_nsteps() << "; "
<< "dt = " << get_tstep() << "\n";

std::cout << "------------------------------------------------"
<< "\n";
std::cout << " " << std::left << std::setw(8) << "s" << std::left
<< std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
<< std::left << std::setw(12) << "time (s)" << std::left
<< std::setw(12) << "GFLOPS"
<< "\n";
std::cout << "------------------------------------------------"
<< "\n";
}
#endif
