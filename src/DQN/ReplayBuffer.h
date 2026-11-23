/*************************************************************************
ReplayBuffer - Stores and samples Deep Q-Learning transitions
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//--------- Interface of the module <ReplayBuffer> (file ReplayBuffer.h) ---------
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

//------------------------------------------------------------------------
// Role of the <ReplayBuffer> module
// Owns a bounded history of transitions and a seeded batch sampler.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <deque>
#include <random>
#include <vector>

//-------------------------------------------------------- Used interfaces
#include "../Structs.h"

//----------------------------------------------------------------- PUBLIC
class ReplayBuffer {
//--------------------------------------------------------- Public Methods
    public:
        void push(const Transition &transition);
        // Contract : Copy the transition and evict the oldest entry when full.

        std::vector<Transition> createBatch(int batchSize);
        // Contract : Sample uniformly without replacement, up to the number stored.
        // A nonpositive request returns an empty batch.

//--------------------------------------------------------- SETTERS/GETTERS
        std::deque<Transition> getTransitions() const
        // Algorithm : Return an independent copy in insertion order.
        {
            return buffer;
        }  //----- end of getTransitions


        int size() const
        // Algorithm : Return the number of stored transitions.
        {
            return static_cast<int>(buffer.size());
        }  //----- end of size


        void clearReplayBuffer()
        // Algorithm : Erase all stored experiences while retaining the sampling generator.
        {
            buffer.clear();
        }  //----- end of clearReplayBuffer

//---------------------------------------------- Constructors - destructor
        ReplayBuffer(int p_capacity, const std::mt19937 &p_gen);
        // Contract : Require a positive capacity and copy the sampling generator.

//---------------------------------------------------------------- PRIVATE
    private:
        friend class ModelPersistence;

        std::deque<Transition> buffer;
        int capacity;
        std::mt19937 gen;
};

#endif //REPLAY_BUFFER_H
