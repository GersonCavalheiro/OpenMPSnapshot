



#include <omp.h>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <boost/algorithm/string.hpp>
#include "wsd_v2.hpp"

#include <emmintrin.h>
#include <immintrin.h>
#include "nmmintrin.h"

using json = nlohmann::json;
using namespace std;


string remove_punctuation(string str) {
string result;
std::remove_copy_if(str.begin(), str.end(),            
std::back_inserter(result), 
::ispunct);

return result;
}


int hash_string(string str) {

std::transform(str.begin(), str.end(), str.begin(), ::tolower);
std::hash<std::string> hasher;
return hasher(str);
}


int compute_overlap(string sense, set<int> context) {

int overlap = 0;
set<int> sense_tokens = tokenize_string(sense);

vector<int> vector_sense(sense_tokens.begin(), sense_tokens.end());
vector<int> vector_context(context.begin(), context.end());

auto const sense_len = vector_sense.size();
auto const context_len = vector_context.size();


while (vector_sense.size() % 4 != 0)
vector_sense.push_back(1);

vector_context.push_back(2);
vector_context.push_back(2);
vector_context.push_back(2);
std::reverse(vector_context.begin(), vector_context.end());
vector_context.push_back(2);
vector_context.push_back(2);
vector_context.push_back(2);

for (int i = 0; i < sense_len; i += 4) {
__m128i simd_sense = _mm_loadu_si128((__m128i const*) &vector_sense[i]);
for (int j = 0; j < context_len - 3; j++) {
__m128i simd_context = _mm_loadu_si128((__m128i const*) &vector_context[i]);
__m128i equality_results = _mm_cmpeq_epi32(simd_sense, simd_context);
equality_results = _mm_hadd_epi32(equality_results, equality_results);
equality_results = _mm_hadd_epi32(equality_results, equality_results);
overlap += _mm_extract_epi32(equality_results, 0) * -1;
}
}


return overlap;
}

vector<string> get_all_senses(string word) {

string dictionary_name = "/Users/ahmedsiddiqui/Workspace/UVic/Winter_2021/CSC485C/wsd-485c/final_dictionary/";
dictionary_name += word[0];
if (word[1] != '\0')
dictionary_name += word[1];
dictionary_name += ".json";

std::ifstream i(dictionary_name);
json j;
i >> j;

return j[word];
}

set<int> get_word_set(string word, string sentence) {
set<int> words = tokenize_string(sentence);
words.erase(hash_string(word));
return words;
}

set<int> tokenize_string(string sentence) {
stringstream stream(sentence);
set<int> words;
string tmp;
while (getline(stream, tmp, ' ')) {
words.insert(hash_string(remove_punctuation(tmp)));
}

return words;
}


string simplified_wsd(string word, string sentence) {
string best_sense;
int max_overlap = 0;

set<int> context = get_word_set(word, sentence); 
vector<string> all_senses = get_all_senses(word); 

vector<int> overlaps(all_senses.size());

auto const start = chrono::steady_clock::now();

#pragma omp parallel for
for (int i = 0; i < all_senses.size(); i++)
overlaps[i] = compute_overlap(all_senses[i], context);

for (int i = 0; i < all_senses.size(); i++){
int overlap = overlaps[i];
if (overlap > max_overlap) {
max_overlap = overlap;
best_sense = all_senses[i];

}
}

auto const end = chrono::steady_clock::now();

cout << "Time to run compute overlap was: " << chrono::duration <double, milli> (end - start).count() << " ms" << endl;

return best_sense;
}

int main(int argc, char ** argv)
{



if( argc >= 2 )
{
omp_set_num_threads(atoi(argv[ 1 ]));
}

cout << simplified_wsd("set", "It was a great day of tennis. Game, set, match");



return 0;
}
