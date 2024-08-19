#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <thread>
#include <map>
#include <cassert>
#include <cstring>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <functional>

#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include "donne.h"

using std::string;
using namespace std::chrono;
using namespace std;
using namespace cv;

Donne::Download::Download(){}

Donne::Download::~Download(){}

auto Donne::Download::cv_write(char *ptr__, size_t size, size_t nmemb, void *userdata__) -> size_t
{
std::vector<uchar> *stream__ = (vector<uchar>*)userdata__;
size_t count {size * nmemb};
stream__ -> insert(stream__ -> end(), ptr__, ptr__ + count);
return count;
}

auto Donne::Download::cv_download(const char *imurl__) -> cv::Mat
{
std::vector<uchar> stream;
curlhandle__ = curl_easy_init();
curl_easy_setopt(curlhandle__, CURLOPT_URL, imurl__);
curl_easy_setopt(curlhandle__, CURLOPT_WRITEFUNCTION, cv_write);
curl_easy_setopt(curlhandle__, CURLOPT_WRITEDATA, &stream);
curl_easy_setopt(curlhandle__, CURLOPT_TIMEOUT, 10);
CURLcode res = curl_easy_perform(curlhandle__);
curl_easy_cleanup(curlhandle__);
if(stream.size() < 1){cv::Mat failed; return failed;}
return cv::imdecode(stream, -1);
}

auto Donne::Download::cv_conduit(std::string &url) -> int
{
count++;
const std::string count_s {std::to_string(count)};
cv::Mat image;
image = cv_download(&url[0]);
if(image.empty()){return 0;};
if((d_1 > 0 || d_2 > 0) || (s_1 > 0 || s_2 > 0))
{
cv::Mat resized;
cv::resize(image, resized, cvSize(d_1, d_2), s_1, s_2);
cv::imwrite(count_s + ".jpg", resized);
}
cv::imwrite(count_s + ".jpg", image);
return 1;
}

auto Donne::Download::format_url(std::vector<std::string> &X) -> void
{
std::transform(X.begin(), X.end(), std::back_inserter(urls), [](std::string &url){return url;});
}

auto Donne::Download::exe_download() -> void
{
assert(!urls.empty()); 
#pragma omp parallel for if(urls.size() >= threshold)
for(int i = 0; i < urls.size(); i++){this -> cv_conduit(urls[i]);}
}

Donne::SQLDatabase::SQLDatabase(){};

Donne::SQLDatabase::SQLDatabase(const std::string &sql_ip, const std::string &sql_db, const std::string &sql_user, const std::string &sql_pass)
{
try {
dri__ = get_driver_instance();
con__ = dri__ -> connect(sql_ip, sql_user, sql_pass);
con__ -> setSchema(sql_db);
} catch(sql::SQLException &e)
{
std::cout<<e.what()<<std::endl;
std::cout<<e.getErrorCode()<<std::endl;
std::cout<<e.getSQLState()<<std::endl;
}
}

Donne::SQLDatabase::~SQLDatabase()
{
if(con__) {delete con__;};
if(res__) {delete res__;};
if(stmt__) {delete stmt__;};
if(prestmt__) {delete prestmt__;};
}

auto Donne::SQLDatabase::select(const std::string &sql_table, const std::string &sql_field, const int &rowstart, const int &rowend) -> int
{
if(con__)
{
try{
prestmt__ = con__ -> prepareStatement("SELECT url FROM image_description LIMIT ?, ?");
prestmt__ -> setInt(1, rowstart);
prestmt__ -> setInt(2, rowend);
res__ = prestmt__ -> executeQuery();
while(res__ -> next())
{
url_store.push_back(res__ -> getString(1));
}
return 1;
} catch(sql::SQLException &e)
{
std::cout<<e.what()<<std::endl;
std::cout<<e.getErrorCode()<<std::endl;
std::cout<<e.getSQLState()<<std::endl;
}
}
}

Donne::Donne(){};

Donne::Donne(const std::string &ip, const std::string &db, const std::string &user, const std::string &pass)
: dwnl__{new Download()}, sqld__{new SQLDatabase(ip, db, user, pass)} {};
Donne::~Donne()
{
if(dwnl__) {delete dwnl__;};
};

auto Donne::select_query(const std::string &table, const std::string &field, const int &rowstart, const int &rowend) -> int
{
sqld__ -> select(table, field, rowstart, rowend);
return 1;
}

auto Donne::dimensions(int d1, int d2, int s1, int s2) -> int
{
dwnl__ -> d_1 = d1;
dwnl__ -> d_2 = d2;
dwnl__ -> s_1 = s1;
dwnl__ -> s_2 = s2;
return 1;
}

auto Donne::download() -> int
{
assert(!sqld__ -> url_store.empty());
dwnl__ -> format_url(sqld__ -> url_store);
dwnl__ -> exe_download();
}

auto Donne::override_threshold(int t) -> void
{
dwnl__ -> threshold = t;
}

auto Donne::override_corethread(int t) -> void
{
omp_set_num_threads(t);
}
