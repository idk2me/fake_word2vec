#include "w2v/util.h"
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace w2v {

std::vector<std::string> tokenize(const std::string &content) {
  std::vector<std::string> tokens;
  std::string current_token = "";

  for (size_t i = 0; i < content.size(); i++) {
    unsigned char curr = static_cast<unsigned char>(content[i]);
    if (std::isalpha(curr)) {
      current_token.push_back(static_cast<char>(std::tolower(curr)));
    } else {
      if (!current_token.empty()) {
        tokens.push_back(current_token);
        current_token.clear();
      }
    }
  }

  if (!current_token.empty()) {
    tokens.push_back(current_token);
  }

  return tokens;
}

std::vector<std::pair<int, int>>
get_pairs_for_index(const int index, const std::vector<int> &token_ids,
                    int window_size) {
  std::vector<std::pair<int, int>> pairs;

  for (int offset = -window_size; offset <= window_size; offset++) {
    if (offset == 0)
      continue;

    int context_index = index + offset;

    if (context_index >= 0 &&
        context_index < static_cast<int>(token_ids.size())) {
      pairs.emplace_back(token_ids[index], token_ids[context_index]);
    }
  }

  return pairs;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float dot(const float *u, const float *v, int D) {
  float sum = 0.0f;
  for (int d = 0; d < D; d++) {
    sum += u[d] * v[d];
  }

  return sum;
}

void train_positive(const int center_id, const int context_id,
                    std::vector<float> &W_in, std::vector<float> &W_out, int D,
                    float lr) {
  float *u = &W_in[center_id * D];
  float *v = &W_out[context_id * D];

  float score = dot(u, v, D);
  float pred = sigmoid(score);
  float err = 1 - pred;

  for (int d = 0; d < D; d++) {
    float u_old = u[d];
    float v_old = v[d];

    u[d] += lr * err * v_old;
    v[d] += lr * err * u_old;
  }
}

void train_negative(int center_id, int neg_id, std::vector<float> &W_in,
                    std::vector<float> &W_out, int D, float lr) {
  float *u = &W_in[center_id * D];
  float *v = &W_out[neg_id * D];

  float pred = sigmoid(dot(u, v, D));
  float err = -pred;

  for (int d = 0; d < D; ++d) {
    float u_old = u[d];
    float v_old = v[d];
    u[d] += lr * err * v_old;
    v[d] += lr * err * u_old;
  }
}

float l2_norm(int word_id, const std::vector<float> &W, int D) {
  float sum = 0.0f;
  const float *u = &W[word_id * D];
  for (int d = 0; d < D; ++d)
    sum += u[d] * u[d];
  return std::sqrt(sum);
}

float cosine(int a, int b, const std::vector<float> &W, int D) {
  const float *u = &W[a * D];
  const float *v = &W[b * D];

  float dotv = 0, nu = 0, nv = 0;
  for (int d = 0; d < D; d++) {
    dotv += u[d] * v[d];
    nu += u[d] * u[d];
    nv += v[d] * v[d];
  }
  return dotv / (std::sqrt(nu) * std::sqrt(nv));
}

int get_id(const std::string &word,
           const std::unordered_map<std::string, size_t> &token_id_map) {
  const auto it = token_id_map.find(word);
  if (it == token_id_map.end()) {
    std::cerr << "Word not in vocab: " << word << std::endl;
    return -1;
  }
  return static_cast<int>(it->second);
}

void print_cosine(const std::string &a, const std::string &b,
                  const std::unordered_map<std::string, size_t> &token_id_map,
                  const std::vector<float> &W, int D) {
  const int ia = get_id(a, token_id_map);
  const int ib = get_id(b, token_id_map);
  if (ia == -1 || ib == -1)
    return;
  std::cout << "Cosine similarity between " << a << " and " << b << ": "
            << cosine(ia, ib, W, D) << std::endl;
}

int sample_negative(int vocab_size, std::mt19937 &rng) {
  std::uniform_int_distribution<int> dist(0, vocab_size - 1);
  return dist(rng);
}

} // namespace w2v
