#pragma once 
#include <string>
#include "Types.h"
#include "ParamsVersion.h"

struct ExperimentParams {
rr_uint version_major{ SPH2D_PARAMS_VERSION_MAJOR };
rr_uint version_minor{ SPH2D_PARAMS_VERSION_MINOR };
rr_uint dim{ 2 };

rr_uint maxn{ 1 << 20 }; 
rr_uint max_neighbours{ 64 };
rr_uint max_cells{ max_neighbours * maxn }; 

rr_float x_maxgeom;
rr_float x_mingeom;
rr_float y_maxgeom;
rr_float y_mingeom;

rr_uint x_fluid_particles;
rr_uint y_fluid_particles;
rr_float x_fluid_min;
rr_float y_fluid_min;
rr_float x_fluid_max;
rr_float y_fluid_max;

rr_float x_boundary_min;
rr_float y_boundary_min;
rr_float x_boundary_max;
rr_float y_boundary_max;

rr_uint nfluid;
rr_uint nvirt;
rr_uint ntotal;
rr_uint fluid_particles_per_d;

rr_float wave_length;
rr_float depth;
rr_float freq;
rr_float piston_amp;
rr_float wave_amp;
rr_float wave_number;
rr_float beach_x;

rr_uint left_wall_start;
rr_uint left_wall_end;
rr_float generator_time_wait;

rr_float dt;
rr_float simulation_time;

rr_uint local_threads;

rr_float eos_csqr_k{ 1.f };

rr_uint pa_sph{ 2 };

rr_uint density_skf{ 1 };
rr_uint int_force_skf{ 1 }; 
rr_uint artificial_viscosity_skf{ 1 };
rr_uint average_velocity_skf{ 1 };
rr_float cell_scale_k{ 2.f }; 

rr_uint nwm{ 0 };
rr_uint boundary_layers_num = 1;

rr_uint sbt{ 1 };

rr_float hsml;

rr_float delta;
rr_float boundary_delta;


bool summation_density{ true };
bool nor_density{ summation_density && false };

bool average_velocity{ true }; 
rr_float average_velocity_epsilon{ 0.3f };

bool visc{ true };
rr_float water_dynamic_visc = 1.e-3f;
rr_float artificial_shear_visc = 1.f;
rr_float artificial_bulk_visc = 0.f;

enum {
TYPE_BOUNDARY = -2,
TYPE_NON_EXISTENT = 0,
TYPE_WATER = 2,
};

rr_float mass = 1000 * delta * delta; 


bool enable_check_consistency{ true };
bool inf_stop{ enable_check_consistency && true };

rr_uint starttimestep = 0;
rr_uint maxtimestep; 
rr_uint normal_check_step; 
rr_uint save_step; 
rr_uint dump_step;
rr_uint print_time_est_step; 

static constexpr rr_float pi{ 3.14159265358979323846f };
static constexpr rr_float g{ 9.81f };

std::string experiment_name;
std::string format_line;

void makeHeader(const std::string& path);
void makeJson(const std::string& path);
void load(const std::string& path);
};

inline ExperimentParams params;
