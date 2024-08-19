#include "barnes_hut.h"
int N = 2500;
int time_steps = 100;
const double L = 1, W = 1, dt = 1e-3, alpha = 0.25, V = 50, epsilon = 1e-1, grav = 0.04; 
double *x, *y, *u, *v, *force_x, *force_y, *mass;
struct node_t *root;
int read_case(char *filename)
{
int i, n;
FILE *arq = fopen(filename, "r");    
if(arq == NULL)
{
printf("Error: The file %s could not be opened.\n", filename);
return 1;
}
n = fscanf(arq, "%d", &N);
if (n != 1)
{
printf("Error: The file %s could not be read for number of particles.\n", filename);
fclose(arq);
return 1;
}
x = (double *)malloc(N * sizeof(double));
y = (double *)malloc(N * sizeof(double));
u = (double *)malloc(N * sizeof(double));
v = (double *)malloc(N * sizeof(double));
force_x = (double *)calloc(N, sizeof(double));
force_y = (double *)calloc(N, sizeof(double));
mass = (double *)malloc(N * sizeof(double));
if (x == NULL || y == NULL || u == NULL || v == NULL || force_x == NULL || force_y == NULL || mass == NULL)
{
printf("Error: Some malloc won't work.\n");
fclose(arq);
return 1;
}
for (i = 0; i < N; i++)
{
n = fscanf(arq, "%lf %lf %lf %lf %lf", &mass[i], &x[i], &y[i], &u[i], &v[i]);
if (n != 5)
{
printf("Error: Some reading won't work at line %d (%d).\n", i + 1, n);
fclose(arq);
return 1;
}
}
fscanf(arq, "%d", &time_steps);
if (filename)
{
fclose(arq);
}
return 0;
}
void free_case()
{
free(x);
free(y);
free(u);
free(v);
free(force_x);
free(force_y);
free(mass);
}
void print_statistics(clock_t s, clock_t e, float ut, float vt, float xc, float xy)
{
#ifdef DEBUG
printf("%f\n", (double)(e - s) / CLOCKS_PER_SEC);
printf("%d\n", N);
printf("%d\n", time_steps);
#endif
printf("%.5f %.5f\n", ut, vt);
printf("%.5f %.5f\n", xc, xy);
}
void time_step(void)
{
root = malloc(sizeof(struct node_t));
set_node(root);
root->min_x = 0;
root->max_x = 1;
root->min_y = 0;
root->max_y = 1;
for (int i = 0; i < N; i++)
{
put_particle_in_tree(i, root);
}
calculate_mass(root);
calculate_center_of_mass_x(root);
calculate_center_of_mass_y(root);
update_forces();
#pragma omp paralle for
for (int i = 0; i < N; i++)
{
double ax = force_x[i] / mass[i];
double ay = force_y[i] / mass[i];
u[i] += ax * dt;
v[i] += ay * dt;
x[i] += u[i] * dt;
y[i] += v[i] * dt;
bounce(&x[i], &y[i], &u[i], &v[i]);
}
free_node(root);
free(root);
}
void bounce(double *x, double *y, double *u, double *v)
{
double W = 1.0f, H = 1.0f;
if (*x > W)
{
*x = 2 * W - *x;
*u = -*u;
}
if (*x < 0)
{
*x = -*x;
*u = -*u;
}
if (*y > H)
{
*y = 2 * H - *y;
*v = -*v;
}
if (*y < 0)
{
*y = -*y;
*v = -*v;
}
}
void put_particle_in_tree(int new_particle, struct node_t *node)
{
if (!node->has_particle)
{
node->particle = new_particle;
node->has_particle = 1;
}
else if (!node->has_children)
{
node->children = malloc(4 * sizeof(struct node_t));
for (int i = 0; i < 4; i++)
{
set_node(&node->children[i]);
}
node->children[0].min_x = node->min_x;
node->children[0].max_x = (node->min_x + node->max_x) / 2;
node->children[0].min_y = node->min_y;
node->children[0].max_y = (node->min_y + node->max_y) / 2;
node->children[1].min_x = (node->min_x + node->max_x) / 2;
node->children[1].max_x = node->max_x;
node->children[1].min_y = node->min_y;
node->children[1].max_y = (node->min_y + node->max_y) / 2;
node->children[2].min_x = node->min_x;
node->children[2].max_x = (node->min_x + node->max_x) / 2;
node->children[2].min_y = (node->min_y + node->max_y) / 2;
node->children[2].max_y = node->max_y;
node->children[3].min_x = (node->min_x + node->max_x) / 2;
node->children[3].max_x = node->max_x;
node->children[3].min_y = (node->min_y + node->max_y) / 2;
node->children[3].max_y = node->max_y;
place_particle(node->particle, node);
place_particle(new_particle, node);
node->has_children = 1;
}
else
{
place_particle(new_particle, node);
}
}
void place_particle(int particle, struct node_t *node)
{
if (x[particle] <= (node->min_x + node->max_x) / 2 && y[particle] <= (node->min_y + node->max_y) / 2)
{
put_particle_in_tree(particle, &node->children[0]);
}
else if (x[particle] > (node->min_x + node->max_x) / 2 && y[particle] < (node->min_y + node->max_y) / 2)
{
put_particle_in_tree(particle, &node->children[1]);
}
else if (x[particle] < (node->min_x + node->max_x) / 2 && y[particle] > (node->min_y + node->max_y) / 2)
{
put_particle_in_tree(particle, &node->children[2]);
}
else
{
put_particle_in_tree(particle, &node->children[3]);
}
}
void set_node(struct node_t *node)
{
node->has_particle = 0;
node->has_children = 0;
}
void free_node(struct node_t *node)
{
if (node->has_children)
{
free_node(&node->children[0]);
free_node(&node->children[1]);
free_node(&node->children[2]);
free_node(&node->children[3]);
free(node->children);
}
}
double calculate_mass(struct node_t *node)
{
if (!node->has_particle)
{
node->total_mass = 0;
}
else if (!node->has_children)
{
node->total_mass = mass[node->particle];
}
else
{
node->total_mass = 0;
for (int i = 0; i < 4; i++)
{
node->total_mass += calculate_mass(&node->children[i]);
}
}
return node->total_mass;
}
double calculate_center_of_mass_x(struct node_t *node)
{
if (!node->has_children)
{
node->c_x = x[node->particle];
}
else
{
node->c_x = 0;
double m_tot = 0;
for (int i = 0; i < 4; i++)
{
if (node->children[i].has_particle)
{
node->c_x += node->children[i].total_mass * calculate_center_of_mass_x(&node->children[i]);
m_tot += node->children[i].total_mass;
}
}
node->c_x /= m_tot;
}
return node->c_x;
}
double calculate_center_of_mass_y(struct node_t *node)
{
if (!node->has_children)
{
node->c_y = y[node->particle];
}
else
{
node->c_y = 0;
double m_tot = 0;
for (int i = 0; i < 4; i++)
{
if (node->children[i].has_particle)
{
node->c_y += node->children[i].total_mass * calculate_center_of_mass_y(&node->children[i]);
m_tot += node->children[i].total_mass;
}
}
node->c_y /= m_tot;
}
return node->c_y;
}
void update_forces()
{
#pragma omp parallel for
for (int i = 0; i < N; i++)
{
force_x[i] = 0;
force_y[i] = 0;
update_forces_help(i, root);
}
}
void update_forces_help(int particle, struct node_t *node)
{
if (!node->has_children && node->has_particle && node->particle != particle)
{
double r = sqrt((x[particle] - node->c_x) * (x[particle] - node->c_x) + (y[particle] - node->c_y) * (y[particle] - node->c_y));
calculate_force(particle, node, r);
}
else if (node->has_children)
{
double r = sqrt((x[particle] - node->c_x) * (x[particle] - node->c_x) + (y[particle] - node->c_y) * (y[particle] - node->c_y));
double theta = (node->max_x - node->min_x) / r;
if (theta < 0.5)
{
calculate_force(particle, node, r);
}
else
{
update_forces_help(particle, &node->children[0]);
update_forces_help(particle, &node->children[1]);
update_forces_help(particle, &node->children[2]);
update_forces_help(particle, &node->children[3]);
}
}
}
void calculate_force(int particle, struct node_t *node, double r)
{
double temp = -grav * mass[particle] * node->total_mass / ((r + epsilon) * (r + epsilon) * (r + epsilon));
force_x[particle] += (x[particle] - node->c_x) * temp;
force_y[particle] += (y[particle] - node->c_y) * temp;
}
int main(int argc, char *argv[])
{
if (argc > 1)
{
int threads = atoi(argv[1]);
if(threads != 2 && threads != 4 && threads != 8 && threads != 16)
{
printf("Error: The argument with the number of threads is invalid. Must be 2, 4, 8, or 16.\n");
return 1;
}
else
{
omp_set_num_threads(threads);
}
}
else
{
printf("Error: The argument with the number of threads is missing.\n");
return 1;
}
if (argc > 2)
{
char *filename = argv[2];
if(read_case(filename) == 1)
{
return 1;
}
}
else
{
printf("Error: The argument with the name of input file is missing.\n");
return 1;
}
long start = clock();
for (int i = 0; i < time_steps; i++)
{
time_step();
}
long stop = clock();
double vu = 0;
double vv = 0;
double sumx = 0;
double sumy = 0;
double total_mass = 0;
#pragma omp parallel for reduction(+ : vu, vv, sumx, sumy, total_mass)
for (int i = 0; i < N; i++)
{
sumx += mass[i] * x[i];
sumy += mass[i] * y[i];
vu += u[i];
vv += v[i];
total_mass += mass[i];
}
double cx = sumx / total_mass;
double cy = sumy / total_mass;
print_statistics(start, stop, vu, vv, cx, cy);
free_case();
return 0;
}