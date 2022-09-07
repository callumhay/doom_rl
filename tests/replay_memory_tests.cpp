#define BOOST_TEST_MODULE DoomRL Test Suite

#include <boost/test/included/unit_test.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <torch/torch.h>
#include <iostream>
#include <ostream>
#include <istream>

#define private public
#define protected public
#include "../app/ReplayMemory.hpp"

BOOST_AUTO_TEST_SUITE(replay_memory_test_suite)

constexpr int MEM_SIZE = 10;

BOOST_AUTO_TEST_CASE(fill_memory) {

  auto replayMemory = std::make_unique<ReplayMemory>(MEM_SIZE);
  replayMemory->initState(torch::full({1}, 99));
  for (int i = 0; i < MEM_SIZE+1; i++) {
    replayMemory->cache(torch::full({1}, i), i, 0.1*i, i == MEM_SIZE ? true : false);
    BOOST_TEST(replayMemory->tdDiffs[i%MEM_SIZE] == ReplayMemory::UNASSIGNED_TD_DIFF);
  }
  // First item should be overwritten for all parts of the replay memory
  BOOST_TEST(replayMemory->states[0][0].item<int>() == 9);
  BOOST_TEST(replayMemory->actions[0][0].item<int>() == MEM_SIZE);
  BOOST_TEST(replayMemory->rewards[0][0].item<double>() == 0.1*MEM_SIZE);
  BOOST_TEST(replayMemory->dones[0][0].item<int>() == 1);

  replayMemory->initState(torch::full({1}, 88));
  BOOST_TEST(replayMemory->states[1][0].item<int>() == 88);

  // All items should have unassigned TD diffs
  BOOST_TEST(replayMemory->priorityMap.size() == 1);
  BOOST_TEST(replayMemory->priorityMap[ReplayMemory::UNASSIGNED_TD_DIFF].size() == MEM_SIZE);
}

BOOST_AUTO_TEST_CASE(td_diff_priority_updates) {
  constexpr int BATCH_SIZE = 3;
  constexpr int SEQ_LEN    = 2;

  double tdDiff = 0;
  auto replayMemory = std::make_unique<ReplayMemory>(MEM_SIZE);
  replayMemory->initState(torch::full({1}, 0));
  for (int i = 0; i < MEM_SIZE; i++) {
    replayMemory->cache(torch::full({1}, i+1), i, 0.1*i, false);
    if (i+1 >= BATCH_SIZE+SEQ_LEN-1) {
      auto [indices, seqs] = replayMemory->sequenceRecall(BATCH_SIZE, SEQ_LEN);
      BOOST_TEST(indices.size() == BATCH_SIZE);

      auto [stateSeqBatch, nextStateBatch, actionBatch, rewardBatch, doneBatch] = seqs;
      BOOST_TEST(stateSeqBatch.sizes() == torch::IntArrayRef({BATCH_SIZE, SEQ_LEN, 1}));
      BOOST_TEST(nextStateBatch.sizes() == torch::IntArrayRef({BATCH_SIZE, 1}));
      BOOST_TEST(actionBatch.sizes() == torch::IntArrayRef({BATCH_SIZE, 1}));
      BOOST_TEST(rewardBatch.sizes() == torch::IntArrayRef({BATCH_SIZE, 1}));
      BOOST_TEST(doneBatch.sizes() == torch::IntArrayRef({BATCH_SIZE, 1}));

      // Make a fake TD diff for the batch
      auto tdDiffs = torch::zeros({BATCH_SIZE, 1});
      for (auto j = 0; j < BATCH_SIZE; j++) { tdDiffs[j][0] = tdDiff++; }
      replayMemory->updateIndexTDDiffs(indices, tdDiffs);
      for (auto j = tdDiff-1; j >= tdDiff-BATCH_SIZE; j--) { 
        BOOST_TEST(replayMemory->priorityMap[j].size() == 1);
      }
    }

    auto sum = 0;
    for (const auto& [key,value] : replayMemory->priorityMap) { sum += value.size(); }
    BOOST_TEST(sum == i+1);
  }
}

BOOST_AUTO_TEST_CASE(td_diff_priority_wraparound_with_done) {
  auto replayMemory = std::make_unique<ReplayMemory>(MEM_SIZE);
  replayMemory->initState(torch::full({1}, -100));
  for (int i = 0; i < MEM_SIZE-1; i++) {
    replayMemory->cache(torch::full({1}, i), i, 0.1*i, i==5); // 'done' state at i==5
    if (i==5) { replayMemory->initState(torch::full({1}, -200)); }
  }
  // States: [-100, 0, 1, 2, 3, 4, -200, 6, 7, 8]
  BOOST_TEST(torch::equal(replayMemory->states[0], torch::full({1},-100)));
  BOOST_TEST(torch::equal(replayMemory->states[6], torch::full({1},-200)));

  BOOST_TEST(torch::equal(replayMemory->dones[4], torch::full({1}, 0.0)));
  BOOST_TEST(torch::equal(replayMemory->dones[5], torch::full({1}, 1.0)));
  BOOST_TEST(torch::equal(replayMemory->dones[6], torch::full({1}, 0.0)));

  replayMemory->cache(torch::full({1}, 99), 9, 99.9, false); // Wraparound state
  BOOST_TEST(torch::equal(replayMemory->states[0], torch::full({1},99)));
  BOOST_TEST(torch::equal(replayMemory->actions[MEM_SIZE-1], torch::full({1},9)));
  BOOST_TEST(torch::equal(replayMemory->rewards[MEM_SIZE-1], torch::full({1},99.9)));
  
  replayMemory->cache(torch::full({1}, 100), 100, 100.1, false); // Wraparound all else
  BOOST_TEST(torch::equal(replayMemory->states[1], torch::full({1},100)));
  BOOST_TEST(torch::equal(replayMemory->actions[0], torch::full({1},100)));
  BOOST_TEST(torch::equal(replayMemory->rewards[0], torch::full({1},100.1)));

  BOOST_TEST(replayMemory->currStateIdx == 2);
  BOOST_TEST(replayMemory->currNonStateIdx == 1);
  
  // Update the replayMemory with TD diff values until everything is assigned a value
  // Each time we check to make sure the sequence never goes across the current non-state index
  auto tdDiff = 0;
  constexpr auto BATCH_SIZE = 4;
  constexpr auto SEQ_LEN    = 2;
  // NOTE: There will always be 1 unassigned value left because the sequence length is 2 and
  // the 0th index cannot be used since it would overlap with the currNonStateIdx
  while (replayMemory->priorityMap[ReplayMemory::UNASSIGNED_TD_DIFF].size() != 1) {
    auto [indices, seqs] = replayMemory->sequenceRecall(BATCH_SIZE, SEQ_LEN);
    auto tdDiffs = torch::zeros({BATCH_SIZE, 1});
    for (auto j = 0; j < BATCH_SIZE; j++) { tdDiffs[j][0] = tdDiff++; }
    replayMemory->updateIndexTDDiffs(indices, tdDiffs);
    
    for (auto i = 1; i < BATCH_SIZE-1; i++) {
      BOOST_TEST(indices[i] != replayMemory->currNonStateIdx-1);
    }
  }

}

BOOST_AUTO_TEST_CASE(serialization) {
  auto rmWrite = std::make_unique<ReplayMemory>(MEM_SIZE);
  rmWrite->initState(torch::full({1}, 0));
  for (int i = 0; i < MEM_SIZE; i++) {
    rmWrite->cache(torch::full({1}, i+1), i, 0.1*i, false);
  }
  rmWrite->cache(torch::full({1}, 99), 99, 99.9, true);

  constexpr char filename[] = "replay_mem_test.txt";
  {
    // Write to file
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa(ofs);
    oa << *rmWrite;
  }
  
  auto rmRead = std::make_unique<ReplayMemory>(2*MEM_SIZE);
  {
    // Read from file
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> *rmRead;
  }

  // Compare written vs. read
  BOOST_TEST(rmWrite->capacity == rmRead->capacity);
  BOOST_TEST(rmWrite->currStateIdx == rmRead->currStateIdx);
  BOOST_TEST(rmWrite->currNonStateIdx == rmRead->currNonStateIdx);
  BOOST_TEST(std::equal(rmWrite->priorityMap.cbegin(), rmWrite->priorityMap.cend(), rmRead->priorityMap.cbegin()));
  BOOST_TEST(std::equal(rmWrite->tdDiffs.cbegin(), rmWrite->tdDiffs.cend(), rmRead->tdDiffs.cbegin()));

  auto testTensorVec = [](const auto& vec1, const auto& vec2) {
    BOOST_TEST(vec1.size() == vec2.size());
    for (auto i = 0; i < vec1.size(); i++) {
      BOOST_TEST(torch::equal(vec1[i], vec2[i]));
    }
    BOOST_TEST(vec1.capacity() == MEM_SIZE);
    BOOST_TEST(vec2.capacity() == MEM_SIZE);
  };

  testTensorVec(rmWrite->states, rmRead->states);
  testTensorVec(rmWrite->actions, rmRead->actions);
  testTensorVec(rmWrite->rewards, rmRead->rewards);
  testTensorVec(rmWrite->dones, rmRead->dones);
}

BOOST_AUTO_TEST_SUITE_END()