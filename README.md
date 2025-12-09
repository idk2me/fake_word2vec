# Mini Word2Vec

Learning about embeddings. Wanted to implement a simple version of Word2Vec from scratch in C++.

## Overview
- Trains a tiny skip-gram with negative sampling (SGNS) model in C++ on `data/corpus.txt`. You can inspect the data [here](https://www.gutenberg.org/cache/epub/2701/pg2701.txt).
- Pipeline: tokenize to lowercase alpha words → drop words with `MINCOUNT` → subsample frequent tokens → build vocab → train `W_in` and `W_out` with SGD.
- Defaults (see `src/main.cpp`): window size 2, embedding dim 50, learning rate 0.005, negatives per positive 40.
- Logs basic metrics to stdout: total negative updates, L2 norm for "the", and cosine similarities for several in-vocab pairs.
- Results are pretty bad due to the tiny corpus and minimal tuning, but that wasn't the goal of this project.

## Build Instructions
2. you can run the build script:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```
3. you can manually build it using cmake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
4. run the executable:
   ```bash
   # in the build directory
   ./embdeddings
   ```

## Results
```text
L2 norm of word the: 0.124984
Cosine similarity between man and woman: 0.271158
Cosine similarity between the and and: 0.806912
Cosine similarity between whale and whales: 0.104451
Cosine similarity between ship and boat: 0.346876
Cosine similarity between ahab and starbuck: -0.210074
Cosine similarity between queequeg and stubb: 0.197628
Cosine similarity between sea and ocean: 0.0623986
Cosine similarity between captain and crew: 0.367167
Cosine similarity between man and sea: 0.135398
```
