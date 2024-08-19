typedef struct caCmBordersS
{
uint32_t left_x;
uint32_t right_x;
uint32_t left_y;
uint32_t right_y;
uint32_t left_z;
uint32_t right_z;
} caCmBorders;
typedef struct caCmIpBordersS
{
uint32_t left_x;
uint32_t right_x;
uint32_t left_y;
uint32_t right_y;
uint32_t left_z;
uint32_t right_z;
} caCmIpBorders;
typedef struct caCmCompartmentS
{
uint32_t m_num; 
uint32_t m_x;
uint32_t m_y;
uint32_t m_z;
uint32_t m_thdnum; 
caCmBorders m_brd1;
caCmBorders m_brd2;
caCmBorders m_brd4;
caCmIpBorders m_ip_brd1;
caCmIpBorders m_ip_brd2;
caCmIpBorders m_ip_brd4;
} caCmCompartment;
const int X_COMPARTMENT_NUM = 4;
const int Y_COMPARTMENT_NUM = 4;
const int Z_COMPARTMENT_NUM = 1;
caCmCompartment *m_compartments[X_COMPARTMENT_NUM][Y_COMPARTMENT_NUM][Z_COMPARTMENT_NUM];
void prepare_ph(caCmCompartment *c, int32_t step);
void iteration_ph(caCmCompartment *c, int32_t step);
void complete_ph(caCmCompartment *c, int32_t step);
void compartmentsComputation (void)
{
int32_t x, y, z;
int32_t step;
int32_t done_state;
#pragma analysis_check assert range(step:1:6:0)
for (step=1; step <= 5; step++) 
{
#pragma analysis_check assert range(step:1:5:0; x:0:4:0)
for (x = 0; x < X_COMPARTMENT_NUM; x++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:4:0)
for (y = 0; y < Y_COMPARTMENT_NUM; y++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:3:0; z:0:1:0)
for (z = 0; z < Z_COMPARTMENT_NUM; z++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(inout:m_compartments[x][y][z])
prepare_ph(m_compartments[x][y][z], step);
}
}
}
#pragma analysis_check assert range(step:1:5:0; done_state:1:13:0)
for (done_state=1; done_state <= 12; done_state++) {
#pragma analysis_check assert range(step:1:5:0; done_state:1:12:0; x:0:4:0)
#pragma analysis_check assert range(x:0:4:0)
for (x = 0; x < X_COMPARTMENT_NUM; x++) {
#pragma analysis_check assert range(step:1:5:0; done_state:1:12:0; x:0:3:0; y:0:4:0)
for (y = 0; y < Y_COMPARTMENT_NUM; y++) {
#pragma analysis_check assert range(step:1:5:0; done_state:1:12:0; x:0:3:0; y:0:3:0; z:0:1:0)
for (z = 0; z < Z_COMPARTMENT_NUM; z++) {
if (x == 0 && y == 0 && z == 0) {                           
#pragma analysis_check assert range(x:0:0:0; y:0:0:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in:m_compartments[x+1][y][z]) depend(in:m_compartments[x][y+1][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
}
else if (x == 0 && z == 0 && (y == Y_COMPARTMENT_NUM-1)) {  
#pragma analysis_check assert range(x:0:0:0; y:3:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x+1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
}
else if (x == 0 && z == 0 && (y != Y_COMPARTMENT_NUM-1)) {  
#pragma analysis_check assert range(x:0:0:0; y:0:2:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x][y+1][z]) depend(in: m_compartments[x+1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else if (y == 0 && z == 0 && (x != X_COMPARTMENT_NUM-1)) {  
#pragma analysis_check assert range(x:0:2:0; y:0:0:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x-1][y][z]) depend(in: m_compartments[x][y+1][z]) depend(in: m_compartments[x+1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else if (y == 0 && z == 0 && (x == X_COMPARTMENT_NUM-1)) {  
#pragma analysis_check assert range(x:3:3:0; y:0:0:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x-1][y][z]) depend(in: m_compartments[x][y+1][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else if ((x == X_COMPARTMENT_NUM-1)  && (y != Y_COMPARTMENT_NUM-1) && z == 0) {  
#pragma analysis_check assert range(x:3:3:0; y:0:2:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x][y+1][z]) depend(in: m_compartments[x-1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else if ((x == X_COMPARTMENT_NUM-1)  && (y == Y_COMPARTMENT_NUM-1) && z == 0) {   
#pragma analysis_check assert range(x:3:3:0; y:3:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x-1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else if ((x != X_COMPARTMENT_NUM-1)  && (y == Y_COMPARTMENT_NUM-1) && z == 0) {  
#pragma analysis_check assert range(x:0:2:0; y:3:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x+1][y][z]) depend(in: m_compartments[x-1][y][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
} 
else {                          
#pragma analysis_check assert range(x:0:3:0; y:0:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(in: m_compartments[x-1][y][z]) depend(in: m_compartments[x+1][y][z]) depend(in: m_compartments[x][y-1][z]) depend(in: m_compartments[x][y+1][z]) depend(inout:m_compartments[x][y][z])
iteration_ph(m_compartments[x][y][z], step);
}    
}
}
}
}
#pragma analysis_check assert range(step:1:5:0; x:0:4:0)
for(x = 0; x < X_COMPARTMENT_NUM; x++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:4:0)
for(y = 0; y < Y_COMPARTMENT_NUM; y++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:3:0; z:0:1:0)
for(z = 0; z < Z_COMPARTMENT_NUM; z++) {
#pragma analysis_check assert range(step:1:5:0; x:0:3:0; y:0:3:0; z:0:0:0)
#pragma omp task firstprivate(step) depend(inout:m_compartments[x][y][z])
complete_ph(m_compartments[x][y][z], step);
}
}
}
#pragma omp taskwait
}
}
