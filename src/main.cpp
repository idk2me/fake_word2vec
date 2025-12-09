#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> tokenize(const std::string &content);

std::vector<std::pair<int, int>>
get_pairs_for_index(const int index, const std::vector<int> &token_ids,
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

int main() {
  std::string path = "data/corpus.txt";
  std::ifstream istrm(path);

  std::string content((std::istreambuf_iterator<char>(istrm)),
                      std::istreambuf_iterator<char>());

  if (!istrm.is_open()) {
    std::cerr << "Failed to open file: " << path << std::endl;
    return 1;
  } else {
    std::cout << "File opened successfully: " << path
              << "\nsize: " << content.size() << std::endl;
  }

  std::string current_token = "";
  std::unordered_map<std::string, size_t> token_freq = {};

  for (const std::string &line : tokenize(content)) {
    token_freq[line]++;
  }

  std::unordered_map<std::string, size_t> token_id_map = {};
  size_t id = 0;

  constexpr size_t MINCOUNT = 5;

  for (const auto &kv : token_freq) {
    if (kv.second < MINCOUNT) {
      continue;
    }
    token_id_map[kv.first] = id;
    id++;
  }

  std::vector<int> token_ids;
  for (const std::string &line : tokenize(content)) {
    const auto it = token_id_map.find(line);
    if (it != token_id_map.end()) {
      token_ids.push_back(it->second);
    }
  }

  constexpr int WINDOW_SIZE = 2;
  constexpr int D = 50;
  constexpr float LR = 0.005f;

  size_t V = token_id_map.size();

  std::vector<size_t> effective_frequency(V, 0);
  std::vector<float> discard_prob(V);

  const float t = 1e-5f;

  for (int id : token_ids) {
    effective_frequency[id]++;
  }

  const float total_tokens = static_cast<float>(token_ids.size());

  for (size_t i = 0; i < V; i++) {
    float f = static_cast<float>(effective_frequency[i]) / total_tokens;
    discard_prob[i] = std::max(0.0f, 1.0f - std::sqrt(t / f));
  }

  std::uniform_real_distribution<float> uni(0.0f, 1.0f);

  std::vector<float> W_in(V * D);
  std::vector<float> W_out(V * D);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f / D, 0.5f / D);

  for (float &w : W_in) {
    w = dist(rng);
  }

  for (float &w : W_out) {
    w = dist(rng);
  }

  size_t updates = 0;

  constexpr int NEG_K = 40;
  static int neg_calls = 0;
  static int epochs = 5;

  for (int epoch = 0; epoch < epochs; epoch++) {
    std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;

    for (size_t i = 0; i < token_ids.size(); i++) {
      int center = token_ids[i];

      if (uni(rng) < discard_prob[center]) {
        continue;
      }

      auto pairs = get_pairs_for_index(i, token_ids, WINDOW_SIZE);
      for (const auto &pair : pairs) {
        int context = pair.second;

        if (uni(rng) < discard_prob[context]) {
          continue;
        }

        train_positive(pair.first, context, W_in, W_out, D, LR);

        for (int k = 0; k < NEG_K; k++) {
          int neg = sample_negative(static_cast<int>(V), rng);
          if (neg == context)
            continue;
          train_negative(pair.first, neg, W_in, W_out, D, LR);
          neg_calls++;
        }
      }
    }
  }

  std::cout << "Training completed." << std::endl;
  std::cout << "NEG Updates: " << neg_calls << "\n";
  const int the_id = get_id("the", token_id_map);
  if (the_id != -1) {
    std::cout << "L2 norm of word the: " << l2_norm(the_id, W_in, D)
              << std::endl;
  }

  print_cosine("man", "woman", token_id_map, W_in, D);
  print_cosine("the", "and", token_id_map, W_in, D);
  print_cosine("whale", "whales", token_id_map, W_in, D);
  print_cosine("ship", "boat", token_id_map, W_in, D);
  print_cosine("ahab", "starbuck", token_id_map, W_in, D);
  print_cosine("queequeg", "stubb", token_id_map, W_in, D);
  print_cosine("sea", "ocean", token_id_map, W_in, D);
  print_cosine("captain", "crew", token_id_map, W_in, D);
  print_cosine("man", "sea", token_id_map, W_in, D);

  return 0;
}

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
