

#pragma once

#include <array>   
#include <cstdint> 
#include <cstdio>  
#include <string>  

namespace rawspeed::md5 {

using md5_state = std::array<uint32_t, 4>;

static constexpr const md5_state md5_init = {
{UINT32_C(0x67452301), UINT32_C(0xEFCDAB89), UINT32_C(0x98BADCFE),
UINT32_C(0x10325476)}};

void md5_hash(const uint8_t* message, size_t len, md5_state* hash);

std::string hash_to_string(const md5_state& hash);

std::string md5_hash(const uint8_t* message, size_t len);

} 
