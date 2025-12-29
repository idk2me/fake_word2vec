#pragma once
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace w2v {

std::vector<std::string> tokenize(const std::string &content);

std::vector<std::pair<int, int>>
get_pairs_for_index(int index, const std::vector<int> &token_ids,
                    int window_size);

float sigmoid(float x);

float dot(const float *u, const float *v, int D);

void train_positive(const int center_id, const int context_id,
                    std::vector<float> &W_in, std::vector<float> &W_out, int D,
                    float lr);

void train_negative(const int center_id, const int neg_id,
                    std::vector<float> &W_in, std::vector<float> &W_out, int D,
                    float lr);

int sample_negative(int vocab_size, std::mt19937 &rng);

float l2_norm(int word_id, const std::vector<float> &W, int D);

float cosine(int a, int b, const std::vector<float> &W, int D);

int get_id(const std::string &word,
           const std::unordered_map<std::string, size_t> &token_id_map);

void print_cosine(const std::string &a, const std::string &b,
                  const std::unordered_map<std::string, size_t> &token_id_map,
                  const std::vector<float> &W, int D);

} // namespace w2v
