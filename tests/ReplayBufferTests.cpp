/*************************************************************************
ReplayBufferTests - Capacity, sampling and stored transitions
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//- Implementation of the module <ReplayBufferTests> (file ReplayBufferTests.cpp) -

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <random>
#include <set>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"
#include "src/DQN/ReplayBuffer.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
Transition makeTransition(int index)
// Algorithm : Builds a transition whose indexed fields make sampling corruption and aliasing easy to detect.
{
    Transition transition{};
    transition.action = static_cast<Action>(index % 4);
    transition.reward = float(index);
    transition.done = index % 2 == 0;
    transition.state.extraFeatures = {float(index), float(index + 1)};
    transition.nextState.extraFeatures = {float(index + 2)};
    transition.state.legalActions.set(index % 4);
    transition.nextState.legalActions.set();

    return transition;
}  //----- end of makeTransition


void testCapacity()
// Algorithm : Tests replay bounds by repeatedly overfilling buffers of several small capacities.
{
    for (int capacity : {1, 2, 7, 32})
    {
        ReplayBuffer buffer(capacity, mt19937(42));

        for (int index = 0; index < capacity*4; index++)
        {
            buffer.push(makeTransition(index));
            check(buffer.size() > 0 && buffer.size() <= capacity, "Replay capacity exceeded");
        }
    }
}  //----- end of testCapacity


void testBatchSizes()
// Algorithm : Tests empty and exact-size sampling while checking that reading batches does not remove transitions.
{
    ReplayBuffer buffer(32, mt19937(42));
    check(buffer.createBatch(0).empty(), "Zero-size batch must be empty");
    check(buffer.createBatch(4).empty(), "Empty replay must return no samples");

    for (int index = 0; index < 16; index++)
    {
        buffer.push(makeTransition(index));
    }

    for (int size : {1, 2, 3, 5, 8, 16})
    {
        check(buffer.createBatch(size).size() == static_cast<size_t>(size), "Requested batch size was not filled");
        check(buffer.size() == 16, "Sampling removed transitions");
    }
}  //----- end of testBatchSizes


void testTransitionIntegrity()
// Algorithm : Tests stored transition ownership by mutating the original and checking all sampled fields.
{
    ReplayBuffer buffer(16, mt19937(42));
    auto transition = makeTransition(5);
    buffer.push(transition);
    transition.state.extraFeatures[0] = -100.f;
    auto sample = buffer.createBatch(1).at(0);

    check(sample.action == Action::Up && sample.reward == 5.f && !sample.done, "Transition scalars changed");
    check(sample.state.extraFeatures == vector<float>{5.f, 6.f}, "Stored state aliases caller memory");
    check(sample.nextState.extraFeatures == vector<float>{7.f}, "Next-state features changed");
    check(sample.state.legalActions.count() == 1 && sample.state.legalActions[1], "Legal mask changed");
}  //----- end of testTransitionIntegrity


void testSamplingReproducibility()
// Algorithm : Tests seeded sampling equality, valid membership and diversity across repeated batches.
{
    ReplayBuffer first(32, mt19937(42));
    ReplayBuffer second(32, mt19937(42));
    set<int> observed;

    for (int index = 0; index < 20; index++)
    {
        first.push(makeTransition(index));
        second.push(makeTransition(index));
    }

    for (int batch = 0; batch < 16; batch++)
    {
        auto left = first.createBatch(5);
        auto right = second.createBatch(5);
        check(left.size() == right.size(), "Seeded batch sizes differ");

        for (size_t index = 0; index < left.size(); index++)
        {
            check(left[index].reward == right[index].reward, "Seeded sampling differs");
            check(left[index].reward >= 0.f && left[index].reward < 20.f, "Sample was never inserted");
            observed.insert(static_cast<int>(left[index].reward));
        }
    }

    check(observed.size() > 5, "Sampling repeatedly returns the same fixed subset");
}  //----- end of testSamplingReproducibility


vector<TestCase> getTests()
// Algorithm : Registers this component’s named test functions for independent selection and execution.
{
    return {
        {"capacity", testCapacity},
        {"batch_sizes", testBatchSizes},
        {"transition_integrity", testTransitionIntegrity},
        {"sampling_reproducibility", testSamplingReproducibility}
    };
}  //----- end of getTests
