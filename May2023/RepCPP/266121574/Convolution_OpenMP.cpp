
class Convolution2D{
public:
Convolution2D(int in_channels, int out_channels, int kernel_size, int stride, int padding)
{
std::random_device device{};
std::normal_distribution<float> distribution{0, 2};
std::mt19937 generator{device()};

this->in_channels = in_channels;
this->out_channels = out_channels;
this->padding = padding;
this->kernel_size = kernel_size;
this->stride = stride;

for(int j=0; j<this->out_channels; j++){
this->Kernels.push_back((float*)malloc(this->in_channels * kernel_size * kernel_size * sizeof(float)));
}

#pragma omp parallel for
for(int i =0; i<in_channels; i++){
float* kernel_plane = this->Kernels.at(i) + i*kernel_size*kernel_size;

for(int j=0; j<kernel_size; j++){
#pragma omp simd
for(int k=0; k<kernel_size; k++){
*(kernel_plane + j*kernel_size + k) = distribution(generator);
}
}
}

this->bias = (float*)malloc(this->out_channels * sizeof(float));

#pragma omp parallel for
for(int j=0; j<this->out_channels; j++){
*(this->bias + j) = distribution(generator);
}

}
float* forward(float* input, int batch, int width, int height){

omp_set_num_threads(12);

int dx = this->kernel_size/2;
int dy = this->kernel_size/2; 

this->current_input = input;
float* output = (float*)calloc(batch * this->out_channels * width * height, sizeof(float));

for(int i=0; i<batch; i++){
auto image_per_batch = input + i*this->in_channels*width*height;
#pragma omp collapse(2) parallel for
for (int j = 0; j<this->out_channels; j++){
float* kernel = this->Kernels.at(j);

for(int k = 0; k<this->in_channels; k++){
float* kernel_slice = kernel + k* this->kernel_size * this->kernel_size;
float* image_slice = image_per_batch + k*width*height;

for(int w = this->kernel_size; w < width - this->kernel_size; w++){
for(int h = this->kernel_size; h < height - this->kernel_size; h++){
float pixel_value = 0.0f;

for(int l = 0; l<this->kernel_size; l++){
#pragma omp simd
for(int m = 0; m<this->kernel_size; m++){
int x = h - dx + l;
int y = i - dy + k;
pixel_value += *(input + x*width + y) * *(kernel + l*this->kernel_size + m);
}
}
*(output + i*this->out_channels * width * height + j*width*height + w*width + h) += pixel_value;
}
}
}
}
}

return output;
}
private:
int in_channels;
int out_channels;
int kernel_size;
int stride;
int padding;
float* bias;
float* current_input;
std::vector<float*> Kernels;
};
