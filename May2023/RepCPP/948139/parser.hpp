#pragma once
#ifndef NBT_GZ_PARSE_HPP_
#define NBT_GZ_PARSE_HPP_

#include <filesystem>
#include <nbt/nbt.hpp>
#include <nbt/stream.hpp>
#include <stack>
#include <translator.hpp>

#define MAXELEMENTSIZE 65025

#define LIST (context.size() && context.top().second < tag_type::tag_long_array)

namespace fs = std::filesystem;

namespace nbt {

static bool format_check(io::ByteStreamReader &b) {
uint8_t buffer[MAXELEMENTSIZE];
uint16_t name_length = 0;
bool error = false;

b.read(1, buffer, &error);
if (error || !buffer[0] || buffer[0] > 13) {
logger::trace("NBT format check error: Invalid type read");
return false;
}

b.read(2, buffer, &error);
if (error) {
logger::trace("NBT format check error: Invalid name size read");
return false;
}

name_length = translate<uint16_t>(buffer);
b.read(name_length, buffer, &error);
if (error) {
logger::trace("NBT format check error: Invalid name read");
return false;
}

for (uint16_t i = 0; i < name_length; i++) {
if (buffer[i] < 0x21 || buffer[i] > 0x7e) {
logger::trace("NBT format check error: Invalid character read: {:02x}",
buffer[i]);
return false;
}
}

return true;
}

static bool matryoshka(io::ByteStreamReader &b, NBT &destination) {
bool error = false;

uint8_t buffer[MAXELEMENTSIZE];

NBT current;
tag_type current_type = tag_type::tag_end, list_type;

std::string current_name;
uint32_t elements, list_elements, name_size;

std::stack<NBT> opened_elements = {};

std::stack<std::pair<uint32_t, tag_type>> context = {};

do {
current_name = "";

if (!LIST) {
b.read(1, buffer, &error);
current_type = tag_type(buffer[0]);

} else {
current_type = context.top().second;
}

if (!LIST && current_type != tag_type::tag_end) {
b.read(2, buffer, &error);
name_size = translate<uint16_t>(buffer);

if (name_size) {
b.read(name_size, buffer, &error);

current_name = std::string((char *)buffer, name_size);
}
}

if (current_type == tag_type::tag_end ||
(LIST && context.top().first == 0)) {
current = std::move(opened_elements.top());
opened_elements.pop();

context.pop();

current_type = tag_type::tag_end;
}

if (current_type == tag_type::tag_compound) {
opened_elements.push(NBT(NBT::tag_compound_t(), current_name));

context.push({0, tag_type(0xff)});

continue;
}

if (current_type == tag_type::tag_list) {
b.read(1, buffer, &error);
list_type = nbt::tag_type(buffer[0]);

b.read(4, buffer, &error);
list_elements = translate<uint32_t>(buffer);

opened_elements.push(NBT(NBT::tag_list_t(), current_name));

context.push({list_elements, list_type});

continue;
}

switch (current_type) {
case tag_type::tag_list:
case tag_type::tag_compound:
case tag_type::tag_end:
break;

case tag_type::tag_byte: {
b.read(1, buffer, &error);
uint8_t byte = buffer[0];

current = NBT(byte, current_name);
break;
}

case tag_type::tag_short: {
b.read(2, buffer, &error);

current = NBT(translate<NBT::tag_short_t>(buffer), current_name);
break;
}

case tag_type::tag_int: {
b.read(4, buffer, &error);

current = NBT(translate<NBT::tag_int_t>(buffer), current_name);
break;
}

case tag_type::tag_long: {
b.read(8, buffer, &error);

current = NBT(translate<NBT::tag_long_t>(buffer), current_name);
break;
}

case tag_type::tag_float: {
b.read(4, buffer, &error);

current = NBT(translate<NBT::tag_float_t>(buffer), current_name);
break;
}

case tag_type::tag_double: {
b.read(8, buffer, &error);

current = NBT(translate<NBT::tag_double_t>(buffer), current_name);
break;
}

case tag_type::tag_byte_array: {
b.read(4, buffer, &error);
elements = translate<uint32_t>(buffer);

std::vector<int8_t> bytes(elements);

for (uint32_t i = 0; i < elements; i++) {
b.read(1, buffer, &error);
bytes[i] = buffer[0];
}

current = NBT(bytes, current_name);
break;
}

case tag_type::tag_int_array: {
b.read(4, buffer, &error);
elements = translate<uint32_t>(buffer);

std::vector<int32_t> ints(elements);

for (uint32_t i = 0; i < elements; i++) {
b.read(4, buffer, &error);
ints[i] = translate<NBT::tag_int_array_t::value_type>(buffer);
}

current = NBT(ints, current_name);
break;
}

case tag_type::tag_long_array: {
b.read(4, buffer, &error);
elements = translate<uint32_t>(buffer);

std::vector<int64_t> longs(elements);
for (uint32_t i = 0; i < elements; i++) {
b.read(8, buffer, &error);
longs[i] = translate<NBT::tag_long_array_t::value_type>(buffer);
}

current = NBT(longs, current_name);
break;
}

case tag_type::tag_string: {
b.read(2, buffer, &error);
uint16_t string_size = translate<uint16_t>(buffer);

b.read(string_size, buffer, &error);
std::string content((char *)buffer, string_size);

current = NBT(std::move(content), current_name);
break;
}
}

if (!LIST) {
if (opened_elements.size())
opened_elements.top()[current.get_name()] = std::move(current);

} else {
NBT::tag_list_t *array = opened_elements.top().get<NBT::tag_list_t *>();
array->insert(array->end(), std::move(current));

context.top().first = std::max(uint32_t(0), context.top().first - 1);
}
} while (!error && opened_elements.size());

destination = std::move(current);
return !error;
}

template <typename Bool_Type = bool,
typename std::enable_if<std::is_same<Bool_Type, bool>::value,
int>::type = 0>
static bool assert_NBT(const fs::path &file) {
gzFile f;
bool status = false;

if ((f = gzopen(file.string().c_str(), "rb"))) {
io::ByteStreamReader gz(f);
status = format_check(gz);

gzclose(f);
} else {
logger::error("Error opening file '{}': {}", file.string(),
strerror(errno));
}

return status;
}

template <typename Bool_Type = bool,
typename std::enable_if<std::is_same<Bool_Type, bool>::value,
int>::type = 0>
static bool assert_NBT(uint8_t *buffer, size_t size) {
bool status = false;

io::ByteStreamReader mem(buffer, size);
status = format_check(mem);

return status;
}

template <
typename NBT_Type = NBT,
typename std::enable_if<std::is_same<NBT_Type, NBT>::value, int>::type = 0>
static bool parse(const fs::path &file, NBT &container) {
gzFile f;
bool status = false;

if ((f = gzopen(file.string().c_str(), "rb"))) {
io::ByteStreamReader gz(f);
status = matryoshka(gz, container);
if (!status)
logger::error("Error reading file {}", file.string());

gzclose(f);
} else {
logger::error("Error opening file '{}': {}", file.string(),
strerror(errno));
}

return status;
}

template <
typename NBT_Type = NBT,
typename std::enable_if<std::is_same<NBT_Type, NBT>::value, int>::type = 0>
static bool parse(uint8_t *buffer, size_t size, NBT &container) {
bool status = false;

io::ByteStreamReader mem(buffer, size);
status = matryoshka(mem, container);
if (!status)
logger::error("Error reading NBT data");

return status;
}
} 

#endif
