#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <iterator>
#include <openssl/sha.h>
#include <random>

#include <sys/types.h>
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



void send_to_udp(std::vector<uint8_t> image_data, std::vector<uint8_t> sha512sum, std::string address, int port)
{
std::random_device rd;
std::mt19937_64 rng(rd());
uint64_t id = std::uniform_int_distribution<uint64_t>()(rng);

uint32_t part_count = (image_data.size() + PAYLOAD_SIZE - 1) / PAYLOAD_SIZE + 1;
std::vector<uint32_t> partindex(part_count - 1);

for(uint32_t i=0; i<partindex.size(); i++) partindex[i] = i;
std::cout << "id = " << id << " part_count = " << part_count << std::endl;
std::shuffle(partindex.begin(), partindex.end(), rng);                      

int udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
sockaddr_in to_addr = {0, 0, 0, 0};
to_addr.sin_family = AF_INET;
to_addr.sin_addr.s_addr = inet_addr(address.c_str());
to_addr.sin_port = htons(port);


Packet packet;
packet.id = id;
packet.size = image_data.size();
packet.part_count = part_count;
packet.part_id = 0;

uint32_t packet_size = PACKET_HEADER_SIZE + sha512sum.size();
std::copy(sha512sum.begin(), sha512sum.end(), packet.data);

sendto(udp_socket, &packet, packet_size, 0, (sockaddr*)(&to_addr), sizeof(to_addr));
for(uint32_t i=0; i<part_count-1; i++)
{
uint32_t idx = partindex[i];
uint32_t packet_start = idx * PAYLOAD_SIZE;
uint32_t packet_end = packet_start + PAYLOAD_SIZE;
if (packet_end > image_data.size()) packet_end = image_data.size();
if (packet_start >= packet_end) throw std::exception();

packet.part_id = idx + 1;
packet_size = PACKET_HEADER_SIZE + packet_end - packet_start;

std::copy(image_data.begin() + packet_start, image_data.begin() + packet_end, packet.data);
sendto(udp_socket, &packet, packet_size, 0, (sockaddr*)(&to_addr), sizeof(to_addr));
std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
}

int main(int argc, char** argv)
{
if (argc != 4)
{
std::cout << argc << std::endl <<
"Syntax: sender <filename> <ip> <port>" << std::endl << 
"Example: sender image.jpg 127.0.0.1 30000" << std::endl;
exit(1);
}
std::string filename = argv[1];
std::string address = argv[2];
int port = std::stoi(argv[3]);                                                                                  

std::ifstream image(filename, std::ios::binary);                                                                
std::vector<uint8_t> image_data((std::istreambuf_iterator<char>(image)), std::istreambuf_iterator<char>());     
std::vector<uint8_t> sha512sum(SHA512_DIGEST_LENGTH);
SHA512(image_data.data(), image_data.size(), sha512sum.data());                                                 

std::cout << "Sending file " << filename << " length = " << image_data.size() << " sha512sum = ";
std::cout << std::hex;
for(uint i=0; i < sha512sum.size(); i++) std::cout << (int)sha512sum[i];
std::cout << std::dec;
std::cout << std::endl << " to address " << address << " port " << port << " "<<std::endl;

send_to_udp(image_data, sha512sum, address, port);
return 0;
}