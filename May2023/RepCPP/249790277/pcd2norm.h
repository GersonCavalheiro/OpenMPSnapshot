#include "ply2pcd.h"

#include <pcl/PCLPointCloud2.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

void computeNorm(const pcl::PCLPointCloud2::ConstPtr &cloud, pcl::PCLPointCloud2 &output, int k, double radius){
PointCloud<PointXYZ>::Ptr xyz (new PointCloud<PointXYZ>);
fromPCLPointCloud2 (*cloud, *xyz);

TicToc tt;
tt.tic ();

PointCloud<Normal> normals;


NormalEstimation<PointXYZ, Normal> ne;
ne.setInputCloud (xyz);
ne.setSearchMethod (search::KdTree<PointXYZ>::Ptr (new search::KdTree<PointXYZ>));
ne.setKSearch (k);
ne.setRadiusSearch (radius);
ne.compute (normals);

print_highlight ("Computed normals in "); print_value ("%g", tt.toc ()); print_info (" ms for "); print_value ("%d", normals.width * normals.height); print_info (" points.\n");

pcl::PCLPointCloud2 output_normals;
toPCLPointCloud2 (normals, output_normals);
concatenateFields (*cloud, output_normals, output);
}

int batchProcess (const vector<string> &pcd_files, const string &output_dir, int k, double radius){
#if _OPENMP
#pragma omp parallel for
#endif
for (int i = 0; i < int (pcd_files.size ()); ++i){
Eigen::Vector4f translation;
Eigen::Quaternionf rotation;
pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
if (!loadCloudPCD (pcd_files[i], *cloud, translation, rotation)) 
continue;

pcl::PCLPointCloud2 output;
computeNorm(cloud, output, k, radius);

string filename = pcd_files[i];
boost::trim (filename);
vector<string> st;
boost::split (st, filename, boost::is_any_of ("/\\"), boost::token_compress_on);

stringstream ss;
ss << output_dir << "/" << st.at (st.size () - 1);
saveCloudPCD(ss.str (), output, translation, rotation, false);
}
return (0);
}

bool normalEstimation(const string &pcdFilename, const string &normFilename, int k, double radius, bool batch_mode){
print_info ("Estimate surface normals using NormalEstimation.\n");

if (!batch_mode){
print_info ("Estimating normals with a k/radius/smoothing size of: "); 
print_value ("%d / %f / %f\n", k, radius, radius); 

Eigen::Vector4f translation;
Eigen::Quaternionf rotation;
pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2);
if (!loadCloudPCD(pcdFilename, *cloud, translation, rotation)) 
return (false);
pcl::PCLPointCloud2 output;
computeNorm(cloud, output, k, radius);
saveCloudPCD(normFilename, output, translation, rotation, false);
return true;
}
else  {
if (pcdFilename != "" && boost::filesystem::exists (pcdFilename)){
vector<string> pcd_files;
boost::filesystem::directory_iterator end_itr;
for (boost::filesystem::directory_iterator itr (pcdFilename); itr != end_itr; ++itr){
if (!is_directory (itr->status ()) && boost::algorithm::to_upper_copy (boost::filesystem::extension (itr->path ())) == ".pcd" ){
pcd_files.push_back (itr->path ().string ());
PCL_INFO ("[Batch processing mode] Added %s for processing.\n", itr->path ().string ().c_str ());
}
}
batchProcess (pcd_files, normFilename, k, radius);
return true;
} else{
PCL_ERROR ("Batch processing mode enabled, but invalid input directory (%s) given!\n", pcdFilename.c_str ());
return (false);
}
}
}
