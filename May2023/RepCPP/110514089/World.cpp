
#include "World.h"
#include "DNA.h"
#include "Organism.h"
#if WITH_GRAPHICS_CONTEXT
#include "GraphicDisplay.h"
#endif

World::World(int width, int height, uint32_t seed) {
width_ = width;
height_ = height;
time_ = 0;

grid_cell_ = new GridCell*[width*height];

global_gen_.seed(seed);

std::uniform_int_distribution<uint32_t> dis(0,UINT32_MAX);

for (int i = 0; i < width; i++) {
for (int j=0; j < height; j++) {
grid_cell_[i*width+j] = new GridCell(this,i,j,dis(global_gen_));
}
}

statfile_best_.open("stats_best.txt");
statfile_mean_.open("stats_mean.txt");
}

void World::random_population() {
float fitness = 0.0;
Organism* org = nullptr;
DNA* dna = nullptr;
printf("Searching for a viable organism ");

long i = 0;
while (fitness <= 0.0) {
delete org;
dna = new DNA(grid_cell_[0]);
org = new Organism(new DNA(dna));
org->gridcell_ = grid_cell_[0];
org->init_organism();
org->build_regulation_network();
for (int t = 0; t < Common::Number_Degradation_Step; t++)
org->compute_protein_concentration();
org->compute_fitness();
fitness = org->fitness_;
if (org->dying_or_not()) {
fitness = 0;
}
printf(".");
delete dna;
i++;
}

min_fitness_ = org->fitness_;
max_fitness_ = org->fitness_;

printf("Found !\nFilling the grid\n");
for (i = 0; i < width_; i++) {
for (int j=0; j < height_; j++) {
grid_cell_[i*width_+j]->organism_ = new Organism(new DNA(org->dna_));
grid_cell_[i*width_+j]->organism_->init_organism();
grid_cell_[i*width_+j]->organism_->gridcell_ = grid_cell_[i*width_+j];
}
}

delete org;
}

void World::init_environment() {
float env[Common::Metabolic_Error_Precision];
std::uniform_real_distribution<float> dis(0,1);
for (int i = 0; i < Common::Metabolic_Error_Precision; i++)
env[i] = dis(global_gen_);

for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {
for (int k = 0; k < Common::Metabolic_Error_Precision; k++) {
grid_cell_[i * width_ + j]->environment_target[k] = env[k];
}
}
}

}

void World::run_evolution() {
#if WITH_GRAPHICS_CONTEXT
GraphicDisplay* display = new GraphicDisplay(this);
#endif
while (time_ < Common::Number_Evolution_Step) {
evolution_step();
int living_one = 0;
for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {
if (grid_cell_[i * width_ + j]->organism_ != nullptr) {
living_one++;
}
}
}

#if WITH_GRAPHICS_CONTEXT
display->display();
#endif

stats();
if (time_%100 == 0) {
printf(
"Evolution at step %d -- Number of Organism %d  (Dead: %d -- Mutant: %d)-- Min Fitness: %f -- Max Fitness: %f\n",
time_, living_one, death_, new_mutant_, min_fitness_, max_fitness_);
statfile_best_.flush();
statfile_mean_.flush();
}
time_++;
}
}

void World::evolution_step() {
death_=0;
new_mutant_=0;

min_fitness_ = 1;
max_fitness_ = 0;

Organism* best;
#pragma omp parallel for collapse(2)
for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {
if (grid_cell_[i * width_ + j]->organism_ != nullptr) {
grid_cell_[i * width_ + j]->organism_->activate_pump();
grid_cell_[i * width_ + j]->organism_->build_regulation_network();

for (int t = 0; t < Common::Number_Degradation_Step; t++)
grid_cell_[i * width_ +
j]->organism_->compute_protein_concentration();

if (grid_cell_[i * width_ + j]->organism_->dying_or_not()) {
delete grid_cell_[i * width_ + j]->organism_;
grid_cell_[i * width_ + j]->organism_ = nullptr;
death_++;
}
}
}
}

for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {
if (grid_cell_[i * width_ + j]->organism_ != nullptr) {

grid_cell_[i * width_ + j]->organism_->compute_fitness();

max_fitness_ = grid_cell_[i * width_ + j]->organism_->fitness_ >
max_fitness_ ? grid_cell_[i * width_ +
j]->organism_->fitness_
: max_fitness_;
min_fitness_ = grid_cell_[i * width_ + j]->organism_->fitness_ <
min_fitness_ ? grid_cell_[i * width_ +
j]->organism_->fitness_
: min_fitness_;
}
}
}

for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {

if (grid_cell_[i * width_ + j]->organism_ == nullptr) {
Organism* org_n = nullptr;

for (int x = i - Common::Duplicate_Neighbors_Offset;
x <= i + Common::Duplicate_Neighbors_Offset; x++) {
for (int y = j - Common::Duplicate_Neighbors_Offset;
y <= j + Common::Duplicate_Neighbors_Offset; y++) {
if (x >= 0 && x < width_)
if (y >= 0 && y < height_) {
if (grid_cell_[x * width_ + y]->organism_ != nullptr) {
if (org_n != nullptr)
org_n = grid_cell_[x * width_ + y]->organism_->fitness_ <
org_n->fitness_ ? grid_cell_[x * width_ +
y]->organism_
: org_n;
else
org_n = grid_cell_[x * width_ + y]->organism_;
}
}
}
}

if (org_n != nullptr) {
new_mutant_++;
org_n->dupli_success_++;
grid_cell_[i * width_ + j]->organism_ = new Organism(new DNA(org_n->dna_));
grid_cell_[i * width_ + j]->organism_->gridcell_ = grid_cell_[
i * width_ + j];
grid_cell_[i * width_ + j]->organism_->mutate();
grid_cell_[i * width_ + j]->organism_->init_organism();
}
}
}
}


for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {
grid_cell_[i * width_ + j]->diffuse_protein();
grid_cell_[i * width_ + j]->degrade_protein();
}
}
}

void World::test_mutate() {

float fitness = 0.0;
Organism* org = nullptr;
DNA* dna = nullptr;
printf("Searching for a viable organism ");

long i = 0;
while (fitness <= 0.0) {
delete org;
dna = new DNA(grid_cell_[0]);
org = new Organism(new DNA(dna));
org->gridcell_ = grid_cell_[0];
org->init_organism();
org->build_regulation_network();
for (int t = 0; t < Common::Number_Degradation_Step; t++)
org->compute_protein_concentration();
org->compute_fitness();
fitness = org->fitness_;
printf(".");
delete dna;
i++;
}

min_fitness_ = org->fitness_;
max_fitness_ = org->fitness_;
death_ = 0;

int better = 0;
int worse = 0;
int equal = 0;

for (i = 0; i < 10000;i++) {
if (i%1000==0) printf("%li\n",i);

Organism* org_new = new Organism(new DNA(org->dna_));
org_new->gridcell_ = grid_cell_[0];
org_new->mutate();
org_new->init_organism();
org_new->activate_pump();
org_new->build_regulation_network();

for (int t = 0; t < Common::Number_Degradation_Step; t++)
org_new->compute_protein_concentration();

if (org_new->dying_or_not()) {
death_++;
}

org_new->compute_fitness();

if (org->fitness_ == org_new->fitness_)
equal++;
else if (org->fitness_ > org_new->fitness_)
worse++;
else if (org->fitness_ < org_new->fitness_)
better++;

delete org_new;
}

printf("Death %d -- Worse %d -- Better %d -- Equal %d\n",death_,worse,better,equal);
}

void World::stats() {
Organism* best;
float best_fitness = 1000;

float avg_fitness = 0;
float avg_meta_error = 0;
float avg_dna_size = 0;
float avg_protein = 0;
float avg_protein_fitness = 0;
float avg_protein_pure_TF = 0;
float avg_protein_poison = 0;
float avg_protein_anti_poison = 0;
float avg_pump = 0;
float avg_move = 0;
float avg_nb_rna = 0;
float avg_network_size = 0;
float avg_life_duration = 0;
float avg_move_success = 0;
float avg_dupli_sucess = 0;
float nb_indiv=0;

for (int i = 0; i < width_; i++) {
for (int j = 0; j < height_; j++) {

if (grid_cell_[i * width_ + j]->organism_ != nullptr) {
if (grid_cell_[i * width_ + j]->organism_->fitness_< best_fitness) {
best = grid_cell_[i * width_ + j]->organism_;
best_fitness = grid_cell_[i * width_ + j]->organism_->fitness_;
}

nb_indiv++;

avg_fitness+=grid_cell_[i * width_ + j]->organism_->fitness_;
avg_meta_error+=grid_cell_[i * width_ + j]->organism_->sum_metabolic_error;
avg_dna_size+=grid_cell_[i * width_ + j]->organism_->dna_->bp_list_.size();
avg_protein+=grid_cell_[i * width_ + j]->organism_->protein_list_map_.size();
avg_protein_fitness+=grid_cell_[i * width_ + j]->organism_->protein_fitness_list_.size();
avg_protein_pure_TF+=grid_cell_[i * width_ + j]->organism_->protein_TF_list_.size();
avg_protein_poison+=grid_cell_[i * width_ + j]->organism_->protein_poison_list_.size();
avg_protein_anti_poison+=grid_cell_[i * width_ + j]->organism_->protein_antipoison_list_.size();
avg_pump+=grid_cell_[i * width_ + j]->organism_->pump_list_.size();
avg_move+=grid_cell_[i * width_ + j]->organism_->move_list_.size();
avg_nb_rna+=grid_cell_[i * width_ + j]->organism_->rna_list_.size();
avg_network_size+=grid_cell_[i * width_ + j]->organism_->rna_influence_.size();
avg_life_duration+=grid_cell_[i * width_ + j]->organism_->life_duration_;
avg_move_success+=grid_cell_[i * width_ + j]->organism_->move_success_;
avg_dupli_sucess+=grid_cell_[i * width_ + j]->organism_->dupli_success_;
}
}
}

avg_fitness/=nb_indiv;
avg_meta_error/=nb_indiv;
avg_dna_size/=nb_indiv;
avg_protein/=nb_indiv;
avg_protein_fitness/=nb_indiv;
avg_protein_pure_TF/=nb_indiv;
avg_protein_poison/=nb_indiv;
avg_protein_anti_poison/=nb_indiv;
avg_pump/=nb_indiv;
avg_move/=nb_indiv;
avg_nb_rna/=nb_indiv;
avg_network_size/=nb_indiv;
avg_life_duration/=nb_indiv;
avg_move_success/=nb_indiv;
avg_dupli_sucess/=nb_indiv;

statfile_mean_<<time_<<","<<avg_fitness<<","<<
avg_meta_error<<","<<
avg_dna_size<<","<<
avg_protein<<","<<
avg_protein_fitness<<","<<
avg_protein_pure_TF<<","<<
avg_protein_poison<<","<<
avg_protein_anti_poison<<","<<
avg_pump<<","<<
avg_move<<","<<
avg_nb_rna<<","<<
avg_network_size<<","<<
avg_life_duration<<","<<
avg_move_success<<","<<
avg_dupli_sucess<<std::endl;

statfile_best_<<time_<<","<<best->fitness_<<","<<
best->sum_metabolic_error<<","<<
best->dna_->bp_list_.size()<<","<<
best->protein_list_map_.size()<<","<<
best->protein_fitness_list_.size()<<","<<
best->protein_TF_list_.size()<<","<<
best->protein_poison_list_.size()<<","<<
best->protein_antipoison_list_.size()<<","<<
best->pump_list_.size()<<","<<
best->move_list_.size()<<","<<
best->rna_list_.size()<<","<<
best->rna_influence_.size()<<","<<
best->life_duration_<<","<<
best->move_success_<<","<<
best->dupli_success_<<std::endl;
}




