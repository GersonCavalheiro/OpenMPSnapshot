#include <algorithm>
#include "rle.h"

std::shared_ptr<EncodedBlock> RLE(std::vector<std::shared_ptr<PixelYcbcr>> block, int block_size) {

std::shared_ptr<EncodedBlock> result(new EncodedBlock());

std::shared_ptr<EncodedBlockColor> result_y = buildTable(block, COLOR_Y, block_size);
encodeValues(block, result_y, COLOR_Y);
result->y = result_y;

std::shared_ptr<EncodedBlockColor> result_cr = buildTable(block, COLOR_CR, block_size);
encodeValues(block, result_cr, COLOR_CR);
result->cr = result_cr;

std::shared_ptr<EncodedBlockColor> result_cb = buildTable(block, COLOR_CB, block_size);
encodeValues(block, result_cb, COLOR_CB);
result->cb = result_cb;

return result;
}

std::vector<std::shared_ptr<PixelYcbcr>> decodeRLE( std::shared_ptr<EncodedBlock> encoded, int block_size) {

std::vector<std::shared_ptr<PixelYcbcr>> result(block_size * block_size);
for (unsigned int i = 0; i < result.size(); i++) {
result[i] = std::make_shared<PixelYcbcr>();
}

unsigned int y_idx = 0;
std::shared_ptr<EncodedBlockColor> y_channel = encoded->y;
std::shared_ptr<std::vector<RleTuple>> tups = y_channel->encoded;

result[y_idx]->y = y_channel->dc_val;
y_idx++;

for (RleTuple tup : *tups) {
std::shared_ptr<std::map<char, double>> decode_table = y_channel->decode_table;
std::map<char, double> decode_table_ptr = *decode_table;
char encoded = tup.encoded;
char freq = tup.count;
double decoded_val = decode_table_ptr[encoded];
for (char c = 0; c < freq; c++) {
result[y_idx]->y = decoded_val;
y_idx++;
}
}

unsigned int cr_idx = 0;
std::shared_ptr<EncodedBlockColor> cr_channel = encoded->cr;
tups = cr_channel->encoded;

result[cr_idx]->cr = cr_channel->dc_val;
cr_idx++;

for (RleTuple tup : *tups) {
std::shared_ptr<std::map<char, double>> decode_table = cr_channel->decode_table;
std::map<char, double> decode_table_ptr = *decode_table;
char encoded = tup.encoded;
char freq = tup.count;
double decoded_val = decode_table_ptr[encoded];
for (char c = 0; c < freq; c++) {
result[cr_idx]->cr = decoded_val;
cr_idx++;
}
}

unsigned int cb_idx = 0;
std::shared_ptr<EncodedBlockColor> cb_channel = encoded->cb;
tups = cb_channel->encoded;

result[cb_idx]->cb = cb_channel->dc_val;
cb_idx++;

for (RleTuple tup : *tups) {
std::shared_ptr<std::map<char, double>> decode_table = cb_channel->decode_table;
std::map<char, double> decode_table_ptr = *decode_table;
char encoded = tup.encoded;
char freq = tup.count;
double decoded_val = decode_table_ptr[encoded];
for (char c = 0; c < freq; c++) {
result[cb_idx]->cb = decoded_val;
cb_idx++;
}
}

return result;
}

std::vector<double> extractChannel(std::vector<std::shared_ptr<PixelYcbcr>> block, int chan) {
std::vector<double> block_vals(block.size());

#pragma omp parallel for
for (unsigned int i = 0; i < block.size(); i++) {
double val;
if (chan == COLOR_Y) {
val = block[i]->y;
} else if (chan == COLOR_CR) {
val = block[i]->cr;
} else { 
val = block[i]->cb;
}

if (val == -0.0) {
val = 0.0;
}
block_vals[i] = val;
}

return block_vals;
}

std::shared_ptr<EncodedBlockColor> buildTable( std::vector<std::shared_ptr<PixelYcbcr>> block, int chan, int block_size) {

std::shared_ptr<EncodedBlockColor> result = std::make_shared<EncodedBlockColor>();
result->encoded = std::make_shared<std::vector<RleTuple>>();
result->decode_table = std::make_shared<std::map<char,double>>();
result->encode_table = std::make_shared<std::map<double,char>>();

std::vector<double> block_vals = extractChannel(block, chan);

std::map<double, char> freq;
#pragma omp parallel for
for (unsigned int i = 1; i < block_vals.size(); i++) {
if (freq.count(block_vals[i])) {
freq[block_vals[i]] += 1;
} else {
freq[block_vals[i]] = 1;
}
}

std::map<char, std::vector<double>> encoded;
char num_chars = 0; 
std::vector<char> key_freqs;
for (auto& kv : freq) {
double key_value = kv.first;
char key_freq = kv.second;
encoded[key_freq].push_back(key_value);
key_freqs.push_back(key_freq);
num_chars++;
}

std::sort(key_freqs.begin(), key_freqs.end(), std::greater<char>());

char curr_encoding = 0;
for (std::map<char, std::vector<double>>::iterator iter = encoded.begin(); iter != encoded.end(); ++iter) {
std::vector<double> vals_to_encode = iter->second;
for (double val_to_encode : vals_to_encode) {
(*result->encode_table)[val_to_encode] = curr_encoding;
(*result->decode_table)[curr_encoding] = val_to_encode;
curr_encoding++;
}
}

return result;
}


void encodeValues(std::vector<std::shared_ptr<PixelYcbcr>> block, std::shared_ptr<EncodedBlockColor> color, int chan) {

std::vector<double> chan_vals = extractChannel(block, chan);

int n = chan_vals.size();
std::shared_ptr<std::vector<RleTuple>> encoded_ptr = color->encoded;

color->dc_val = chan_vals[0];

int curr_idx = 1;
int curr_run = 1;
double curr_val = chan_vals[curr_idx];

while (curr_idx < n - 1) {
curr_idx++;
double val = chan_vals[curr_idx];
if (val == curr_val) {
curr_run++;
} else {
RleTuple rleTuple;
rleTuple.encoded = (*color->encode_table)[curr_val];
rleTuple.count = curr_run;
(*encoded_ptr).push_back(rleTuple);
curr_run = 1;
curr_val = chan_vals[curr_idx];
}
}

RleTuple rleTuple;
rleTuple.encoded = (*color->encode_table)[chan_vals[n-1]];
if (chan_vals[n-1] != chan_vals[n-2]) {
rleTuple.count = 1;
} else { 
rleTuple.count = curr_run;
}
(*color->encoded).push_back(rleTuple);
}

void writeToBuffer(
std::shared_ptr<EncodedBlockNoPtr> encodedBlockBuffer,
std::vector<std::shared_ptr<EncodedBlock>> encodedBlocks,
int idx, int chan) {

std::shared_ptr<EncodedBlock> encodedBlock = encodedBlocks[idx];

(encodedBlockBuffer.get())[idx].y.dc_val = encodedBlock->y->dc_val;

if (chan == COLOR_Y) {
unsigned int sz = (*encodedBlock->y->encoded.get()).size();
for (unsigned int j = 0; j < sz; j++) {
(encodedBlockBuffer.get())[idx].y.encoded[j].encoded = (*encodedBlock->y->encoded.get())[j].encoded;
(encodedBlockBuffer.get())[idx].y.encoded[j].count = (*encodedBlock->y->encoded.get())[j].count;
}
(encodedBlockBuffer.get())[idx].y.encoded_len = sz;
int kv_idx = 0;
for (std::map<double, char>::iterator iter = (
*encodedBlock->y->encode_table).begin();
iter != (*encodedBlock->y->encode_table).end(); ++iter) {
double double_val = iter->first;
char char_val = iter->second;
(encodedBlockBuffer.get())[idx].y.char_vals[kv_idx] = char_val;
(encodedBlockBuffer.get())[idx].y.double_vals[kv_idx] = double_val;
kv_idx++;
}
(encodedBlockBuffer.get())[idx].y.table_size = kv_idx;
} else if (chan == COLOR_CR) {
unsigned int sz = (*encodedBlock->cr->encoded.get()).size();
for (unsigned int j = 0; j < sz; j++) {
(encodedBlockBuffer.get())[idx].cr.encoded[j].encoded = (*encodedBlock->cr->encoded.get())[j].encoded;
(encodedBlockBuffer.get())[idx].cr.encoded[j].count = (*encodedBlock->cr->encoded.get())[j].count;
}
(encodedBlockBuffer.get())[idx].cr.encoded_len = sz;
int kv_idx = 0;
for (std::map<double, char>::iterator iter = (
*encodedBlock->cr->encode_table).begin();
iter != (*encodedBlock->cr->encode_table).end(); ++iter) {
double double_val = iter->first;
char char_val = iter->second;
(encodedBlockBuffer.get())[idx].cr.char_vals[kv_idx] = char_val;
(encodedBlockBuffer.get())[idx].cr.double_vals[kv_idx] = double_val;
kv_idx++;
}
(encodedBlockBuffer.get())[idx].cr.table_size = kv_idx;
} else { 
unsigned int sz = (*encodedBlock->cb->encoded.get()).size();
for (unsigned int j = 0; j < sz; j++) {
(encodedBlockBuffer.get())[idx].cb.encoded[j].encoded = (*encodedBlock->cb->encoded.get())[j].encoded;
(encodedBlockBuffer.get())[idx].cb.encoded[j].count = (*encodedBlock->cb->encoded.get())[j].count;
}
(encodedBlockBuffer.get())[idx].cb.encoded_len = sz;
int kv_idx = 0;
for (std::map<double, char>::iterator iter = (
*encodedBlock->cb->encode_table).begin();
iter != (*encodedBlock->cb->encode_table).end(); ++iter) {
double double_val = iter->first;
char char_val = iter->second;
(encodedBlockBuffer.get())[idx].cb.char_vals[kv_idx] = char_val;
(encodedBlockBuffer.get())[idx].cb.double_vals[kv_idx] = double_val;
kv_idx++;
}
(encodedBlockBuffer.get())[idx].cb.table_size = kv_idx;
}
}

std::vector<std::shared_ptr<EncodedBlock>> convertBufferToEncodedBlocks(
std::shared_ptr<EncodedBlockNoPtr> encodedBlocksBuffer, int numEncodedBlocks) {

std::vector<std::shared_ptr<EncodedBlock>> result;

for (int i = 0; i < numEncodedBlocks; i++) {
EncodedBlockNoPtr block = (encodedBlocksBuffer.get())[i];
std::shared_ptr<EncodedBlock> resultBlock = std::make_shared<EncodedBlock>();
resultBlock->y = std::make_shared<EncodedBlockColor>();
resultBlock->cr = std::make_shared<EncodedBlockColor>();
resultBlock->cb = std::make_shared<EncodedBlockColor>();
resultBlock->y->encoded = std::make_shared<std::vector<RleTuple>>();
resultBlock->cr->encoded = std::make_shared<std::vector<RleTuple>>();
resultBlock->cb->encoded = std::make_shared<std::vector<RleTuple>>();
resultBlock->y->decode_table = std::make_shared<std::map<char, double>>();
resultBlock->cr->decode_table = std::make_shared<std::map<char, double>>();
resultBlock->cb->decode_table = std::make_shared<std::map<char, double>>();
resultBlock->y->encode_table = std::make_shared<std::map<double,char>>();
resultBlock->cr->encode_table = std::make_shared<std::map<double,char>>();
resultBlock->cb->encode_table = std::make_shared<std::map<double,char>>();
resultBlock->y->dc_val = block.y.dc_val;
resultBlock->cr->dc_val = block.cr.dc_val;
resultBlock->cb->dc_val = block.cb.dc_val;
for (int j = 0; j < block.y.encoded_len; j++) {
RleTuple rleTuple;
rleTuple.encoded = block.y.encoded[j].encoded;
rleTuple.count = block.y.encoded[j].count;
(*resultBlock->y->encoded.get()).push_back(rleTuple);
}
for (int j = 0; j < block.cb.encoded_len; j++) {
RleTuple rleTuple;
rleTuple.encoded = block.cb.encoded[j].encoded;
rleTuple.count = block.cb.encoded[j].count;
(*resultBlock->cb->encoded.get()).push_back(rleTuple);
}
for (int j = 0; j < block.cr.encoded_len; j++) {
RleTuple rleTuple;
rleTuple.encoded = block.cr.encoded[j].encoded;
rleTuple.count = block.cr.encoded[j].count;
(*resultBlock->cr->encoded.get()).push_back(rleTuple);
}
char char_val;
double double_val;
for (int j = 0; j < block.y.table_size; j++) {
char_val = block.y.char_vals[j];
double_val = block.y.double_vals[j];
(*resultBlock->y->encode_table.get())[double_val] = char_val;
(*resultBlock->y->decode_table.get())[char_val] = double_val;
}
for (int j = 0; j < block.cb.table_size; j++) {
char_val = block.cb.char_vals[j];
double_val = block.cb.double_vals[j];
(*resultBlock->cb->encode_table.get())[double_val] = char_val;
(*resultBlock->cb->decode_table.get())[char_val] = double_val;
}
for (int j = 0; j < block.cr.table_size; j++) {
char_val = block.cr.char_vals[j];
double_val = block.cr.double_vals[j];
(*resultBlock->cr->encode_table.get())[double_val] = char_val;
(*resultBlock->cr->decode_table.get())[char_val] = double_val;
}
result.push_back(resultBlock);
}

return result;
}
