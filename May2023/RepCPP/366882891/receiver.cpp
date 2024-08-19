#include <memory>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iterator>
#include <openssl/sha.h>
#include <random>
#include <stdexcept>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <algorithm> 
#include <chrono>
#include <thread>

#define PAYLOAD_SIZE 1400       
#define PACKET_HEADER_SIZE 24   

#pragma pack(push, 1)
struct Packet
{
uint64_t id;                
uint64_t size;              
uint32_t part_id;           
uint32_t part_count;        
uint8_t data[PAYLOAD_SIZE]; 
};
#pragma pack(pop)

struct config{
std::string address;
int port;
std::string folder;
};

void recive_to_udp(config& conf) {

Packet packet;    
uint32_t packet_size = PAYLOAD_SIZE + PACKET_HEADER_SIZE;
std::vector<uint8_t> sha512sum(SHA512_DIGEST_LENGTH);
std::vector<uint8_t> sha512sum_check(SHA512_DIGEST_LENGTH);
std::vector<uint8_t> image_data;
std::vector<uint8_t> buff(PAYLOAD_SIZE);

int udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
sockaddr_in to_addr = {0, 0, 0, 0};
to_addr.sin_family = AF_INET;
to_addr.sin_addr.s_addr = inet_addr(conf.address.c_str());
to_addr.sin_port = htons(conf.port);

int result = 1;
socklen_t socklen = sizeof(result);
setsockopt(udp_socket, SOL_SOCKET, SO_REUSEADDR, &result, socklen);

if(bind(udp_socket, (struct sockaddr*)&to_addr, sizeof(to_addr)) < 0) {
throw 2;
}
recvfrom(udp_socket, &packet, packet_size, 0, NULL, NULL);

image_data.resize(packet.size);

for(int i = 0; i < SHA512_DIGEST_LENGTH; ++i) sha512sum[i] = packet.data[i];

std::cout << "Reciving file with id: " << packet.id << std::endl;
std::cout << "Size: " << packet.size  << std::endl;
std::cout << "Sha512: " << std::hex;
for(int i = 0; i < SHA512_DIGEST_LENGTH; ++i) std::cout << (int)sha512sum[i];
std::cout << std::dec << std::endl;

for(uint32_t i = 0; i < packet.part_count - 1; i++) {
recvfrom(udp_socket, &packet, packet_size, 0, NULL, NULL);

if(packet.part_id != packet.part_count - 1) {
for(uint j = 0; j < PAYLOAD_SIZE; ++j) buff[j] = packet.data[j];
std::copy(buff.begin(), buff.end(), image_data.begin() + (packet.part_id-1) * PAYLOAD_SIZE); 
}
else {
for(uint j = 0; j < packet.size - (packet.part_count - 2) * PAYLOAD_SIZE; ++j) buff[j] = packet.data[j];
std::copy(buff.begin(), buff.begin() + packet.size -  (packet.part_id-1) * PAYLOAD_SIZE, image_data.begin() + (packet.part_id-1) * PAYLOAD_SIZE);
}
}

SHA512(image_data.data(), image_data.size(), sha512sum_check.data());                                                

std::cout << "Sha512_check: ";
std::cout << std::hex;
for(uint i=0; i < sha512sum_check.size(); i++) std::cout << (int)sha512sum_check[i];
std::cout << std::dec << std::endl;

if(sha512sum != sha512sum_check) {
shutdown(udp_socket, SHUT_RDWR);
throw 1;
}
std::cout << "Transmission completed succesfully!" << std::endl;

const char * dir = conf.folder.c_str();
struct stat s;
if(stat(dir, &s)){
mkdir(dir, 0777);
}
std::string filename = conf.folder + "/" + std::to_string(packet.id) + ".jpg";
std::ofstream ostrm(filename, std::ios::binary);   
std::copy(image_data.begin(), image_data.end(), std::ostreambuf_iterator<char>(ostrm));
std::cout << "File saved in " << filename << std::endl << std::endl;

shutdown(udp_socket, SHUT_RDWR);
}

void json_read(config& conf){
std::string str[7];
std::ifstream param("config.json", std::ifstream::binary);
for(int i = 0; i < 7; ++i) {
param >> str[i];
}

conf.address = str[2].substr(1, str[2].size() - 3);
conf.port = std::stoi(str[4]);
conf.folder = str[6].substr(1, str[6].size() - 2);
}


int main(){

std::cout << "Program start" << std::endl; 

config conf;

json_read(conf);
std::cout << "Address: " << conf.address << " Port: " << conf.port << " Folder: " << conf.folder << std::endl << std::endl;

while(1) {
try {
recive_to_udp(conf);
}
catch (int x) {
switch (x) {
case 1:
std::cout << "Error! File has been corrupted!" << std::endl << std::endl;
break;
case 2:
std::cout << "Error! Can't bind!" << std::endl << std::endl;  
return 1;  
default:
return 1;                
}
}
}
return 0;
}