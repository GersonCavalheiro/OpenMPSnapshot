#include "ConvertRepresentation.hpp"


void WriteSlices(string xray_dir, vector<string>& files, string results_dir, int max_number_threads){

string row_dir = results_dir + "images/";

string command = "mkdir " + row_dir;
system(command.c_str());

vector<Mat> zslices(files.size());

int number_threads = max_number_threads;

omp_set_num_threads(number_threads);

#pragma omp parallel
{
int thread_id = omp_get_thread_num();

Mat frame;
string filename;
for (int i = 0, in  = files.size(); i < in; i++){

if (i % number_threads == thread_id){
#pragma omp critical (write_text)
{
cout << "read " << i << " of " << files.size() <<  endl;
}
filename = xray_dir + files[i];
frame = imread(filename.c_str(), cv::IMREAD_GRAYSCALE);

frame.copyTo(zslices[i]);
}
}
}

int rows = zslices[0].rows;
int cols = zslices[0].cols;
int number_zslices = files.size();
int number_row_major_slices = rows;
int number_col_major_slices = cols;
vector<Mat> row_major_slices(number_row_major_slices);
vector<Mat> col_major_slices(number_col_major_slices);


int rm_rows = number_zslices;
int rm_cols = cols;

int cm_rows = rows;
int cm_cols = number_zslices;



cout << "zslices, number of " << number_zslices << endl;
cout << "rows, cols " << rows << ", " << cols << endl;
cout << "number row_major slices " << number_row_major_slices << endl;
cout << "rm rows, cols " << rm_rows << ", " << rm_cols << endl;
cout << "cm rows, cols " << cm_rows << ", " << cm_cols << endl;

#pragma omp parallel
{
int thread_id = omp_get_thread_num();
Mat onechannel_image;
string filename;
onechannel_image.create(rm_rows, rm_cols, CV_8U);

for (int i = 0, in  = number_row_major_slices; i < in; i++){
if (i % number_threads == thread_id){
onechannel_image.copyTo(row_major_slices[i]);
}
}
}

cout << "Done creating row major slices." << endl;

#pragma omp parallel
{
int thread_id = omp_get_thread_num();
string filename;

for (int i = 0, in  = number_zslices; i < in; i++){
if (i % number_threads == thread_id){

for (int r = 0; r < rows; r++ ){
for (int c = 0; c < cols; c++){
row_major_slices[r].at<uchar>(i,c) = zslices[i].at<uchar>(r, c);
}
}
}
}
}

cout << "Done copying over row-major slices " << endl;

#pragma omp parallel
{
int thread_id = omp_get_thread_num();
string filename;

for (int i = 0, in  = number_row_major_slices; i < in; i++){
if (i % number_threads == thread_id){

filename  = row_dir + "image" +  ToString<int>(i) + ".png";
cout << "filename " << filename << endl;
cv::imwrite(filename.c_str(), row_major_slices[i]);

}
}
}

{
vector<Mat> temp;
temp.swap(row_major_slices);
}

cout << "Done writing" << endl;

}

