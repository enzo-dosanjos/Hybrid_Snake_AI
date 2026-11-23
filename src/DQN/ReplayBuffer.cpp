/*************************************************************************
ReplayBuffer - Stores and samples Deep Q-Learning transitions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <ReplayBuffer> (file ReplayBuffer.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <algorithm>
#include <iterator>
#include <stdexcept>

using namespace std;

//------------------------------------------------------- Personal Include
#include "ReplayBuffer.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
ReplayBuffer::ReplayBuffer(int p_capacity, const mt19937 &p_gen)
    : capacity(p_capacity), gen(p_gen)
// Algorithm : Copy the generator and reject a capacity that cannot hold a transition.
{
    if (capacity <= 0)
    {
        throw invalid_argument("Replay capacity must be positive");
    }
}  //----- end of ReplayBuffer


void ReplayBuffer::push(const Transition &transition)
// Algorithm : Append an owned transition before removing the oldest entry when full.
{
    buffer.push_back(transition);

    if (buffer.size() > static_cast<size_t>(capacity))
    {
        buffer.pop_front();
    }
}  //----- end of push


vector<Transition> ReplayBuffer::createBatch(int batchSize)
// Algorithm : Sample distinct entries using the stored generator and shuffle the resulting batch.
{
    if (batchSize <= 0 || buffer.empty())
    {
        return {};
    }

    int sampleCount = min(batchSize, size());
    vector<Transition> batch;
    batch.reserve(sampleCount);

    // ponytail: std::sample scans the buffer; sample distinct indices if this dominates training time.
    sample(buffer.begin(), buffer.end(), back_inserter(batch), sampleCount, gen);
    shuffle(batch.begin(), batch.end(), gen);
    return batch;
}  //----- end of createBatch
