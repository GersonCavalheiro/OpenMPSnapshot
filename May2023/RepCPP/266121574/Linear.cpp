class Linear{
public:
Linear(int input_size, int output_size){

std::random_device device{};
std::normal_distribution<float> distribution{0, 2};

std::mt19937 generator{device()};
this->input_neurons = input_size;
this->output_neurons = output_size;

this->weights = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
this->bias = (float*)malloc(this->output_neurons * sizeof(float));
#pragma omp parallel for
for(int i = 0; i < this->input_neurons; i++){
#pragma omp simd
for(int j = 0; j<this->output_neurons; j++){
*(this->weights + i*this->output_neurons + j) = distribution(generator);
}
}

#pragma omp parallel for
for(int i = 0; i<this->output_neurons; i++){
*(this->bias + i) = distribution(generator); 
}
}


float* forward(float* input, int batch_size){
this->current_input = input;
float* output = (float*)malloc(batch_size * this->output_neurons * sizeof(float));

for(int i = 0; i< batch_size; i++){
float* per_batch_input = input + this->input_neurons*i;

#pragma omp parallel for
{
for(int j=0; j<output_neurons; j++){
float result = 0.0f;
#pragma omp simd
for(int k=0; k< this->input_neurons; k++){
result += *(per_batch_input + k) * *(this->weights + k*this->output_neurons + j);
}
*(output + i*this->output_neurons + j) = result + *(this->bias+j);
}
}
}
return output;
}


void backward(float* dldY, int batch_size){

this->dldX = (float*)malloc(batch_size * this->input_neurons * sizeof(float));
this->dldW = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
float* w_t = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
float* x_t = (float*)malloc(batch_size * this->input_neurons * sizeof(float));

#pragma omp parallel for
for(int i =0; i<batch_size; i++){
#pragma omp simd
for(int j=0; j<this->input_neurons; j++){
*(x_t + j*batch_size + i) = *(this->current_input + i*batch_size + j);
}
}
#pragma omp parallel for
for (int i = 0; i<this->input_neurons; i++){
#pragma omp simd
for (int j=0; j<this->output_neurons; j++){
*(w_t + j*input_neurons + i) = *(this->weights + i*this->output_neurons + j);
}
}
for(int i =0; i<batch_size; i++){
float* per_batch_gradient = dldY + i*this->output_neurons;

#pragma omp parallel for
for(int j=0; j<this->input_neurons; j++){
float result = 0.0f;
#pragma omp simd
for (int k=0; k<this->output_neurons; k++){
result += *(per_batch_gradient + k) * *(w_t + k*this->input_neurons + j);
}
*(this->dldW + i*this->input_neurons + j) = result;
}
}

#pragma omp parallel for
for(int i=0; i<this->input_neurons; i++){
for(int j=0; j<this->output_neurons; j++){
float result = 0.0f;
for(int k=0; k<batch_size; k++){
result +=  *(x_t + i*batch_size + k) * *(dldY + k*this->output_neurons +j);
}
*(this->dldX + i*this->input_neurons + j) = result;
}
}
this->grad[0] = this->dldW;
this->grad[1] = this->dldX;
}



private:
int input_neurons;
int output_neurons;
float* weights;
float* bias;
float* current_input;
float* grad[2]; 
float* dldX, *dldW; 
};
