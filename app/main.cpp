#include "w2v/util.h"
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

  for (const std::string &line : w2v::tokenize(content)) {
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
  for (const std::string &line : w2v::tokenize(content)) {
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

      auto pairs = w2v::get_pairs_for_index(i, token_ids, WINDOW_SIZE);
      for (const auto &pair : pairs) {
        int context = pair.second;

        if (uni(rng) < discard_prob[context]) {
          continue;
        }

        w2v::train_positive(pair.first, context, W_in, W_out, D, LR);

        for (int k = 0; k < NEG_K; k++) {
          int neg = w2v::sample_negative(static_cast<int>(V), rng);
          if (neg == context)
            continue;
          w2v::train_negative(pair.first, neg, W_in, W_out, D, LR);
          neg_calls++;
        }
      }
    }
  }

  std::cout << "Training completed." << std::endl;
  std::cout << "NEG Updates: " << neg_calls << "\n";
  const int the_id = w2v::get_id("the", token_id_map);
  if (the_id != -1) {
    std::cout << "L2 norm of word the: " << w2v::l2_norm(the_id, W_in, D)
              << std::endl;
  }

  w2v::print_cosine("man", "woman", token_id_map, W_in, D);
  w2v::print_cosine("the", "and", token_id_map, W_in, D);
  w2v::print_cosine("whale", "whales", token_id_map, W_in, D);
  w2v::print_cosine("ship", "boat", token_id_map, W_in, D);
  w2v::print_cosine("ahab", "starbuck", token_id_map, W_in, D);
  w2v::print_cosine("queequeg", "stubb", token_id_map, W_in, D);
  w2v::print_cosine("sea", "ocean", token_id_map, W_in, D);
  w2v::print_cosine("captain", "crew", token_id_map, W_in, D);
  w2v::print_cosine("man", "sea", token_id_map, W_in, D);

  return 0;
}
