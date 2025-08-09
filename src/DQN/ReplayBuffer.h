/*************************************************************************
ReplayBuffer - A replay buffer for the Deep Q-Learning algorithm
                             -------------------
    copyright            : (C) 2025 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <ReplayBuffer> (file ReplayBuffer.h) ----
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

//------------------------------------------------------------------------
// Role of the <ReplayBuffer> module
// The ReplayBuffer module implements z replay buffer for the Deep
// Q-Learning algorithm. The replay buffer stores the transitions of the
// agent and allows to sample a batch of transitions for the training.
//------------------------------------------------------------------------

//-------------------------------------------------------- Used interfaces
#include "Structs.h"

//---------------------------------------------------------------- INCLUDE
#include <random>
#include <sstream>


//------------------------------------------------------------------ Types
struct Transition {
    std::vector<float> boardState;        // State of the board (1st input of the DQN)
    std::vector<float> extraState;        // State of the extras (2nd input of the DQN)

    std::string action;                   // choosen action
    float reward;                         // reward given

    std::vector<float> nextBoardState;    // next state of the board
    std::vector<float> nextExtraState;    // next state of the extras

    bool done;                            // indicate if the episode has ended
    float tdError;                        // temporal difference error
};


//----------------------------------------------------------------- PUBLIC
class ReplayBuffer
{
//--------------------------------------------------------- Public Methods
    public:
        void push(
            const Transition &transition
        );
        // Usage :
        //
        // Contract :
        //

        Transition createTransition(
            const std::vector<float> &prevBoardState,
            const std::vector<float> &boardState,
            const std::vector<float> &prevExtraState,
            const std::vector<float> &extraState,
            const std::string &action,
            float reward,
            bool done
        );
        // Usage :
        //
        // Contract :
        //

        std::vector<Transition> createBatch(
            int batchSize
        );
        // Usage :
        //
        // Contract :
        //

//--------------------------------------------------------- SETTERS/GETTERS
        std::deque<Transition> getTransitions()
        {
            return buffer;
        }
        // Usage :
        //
        // Contract :
        //

        int size()
        {
            return buffer.size();
        }
        // Usage :
        //
        // Contract :
        //

        void clearReplayBuffer()
            {
                buffer.clear();
            }
        // Usage :
        //
        // Contract :
        //

//---------------------------------------------- Constructors - destructor
        explicit ReplayBuffer(
            int p_capacity,
            const std::mt19937 &p_gen
        ) : capacity (p_capacity),
            gen(p_gen)  // Initialize with a random seed
        // Usage :
        //
        // Contract :
        //
        {
            #ifdef MAP
                    cout << "Calling the constructor of ReplayBuffer" << endl;
            #endif
        }

        ~ReplayBuffer ()
        // Usage :
        //
        // Contract :
        //
        {
            #ifdef MAP
                    cout << "Calling the destructor of ReplayBuffer" << endl;
            #endif
        }


    protected:
        std::vector<Transition> sample(
            int start,
            int end,
            int batchSize
        );
        // Usage :
        //
        // Contract :
        //


        void mergeSamples(
            std::vector<Transition>& top,
            std::vector<Transition>& rest,
            std::vector<Transition>& output
        );
        // Usage :
        //
        // Contract :
        //

        void entropyInjection(
            std::vector<Transition>& batch,
            float entropyWeight
        );
        // Usage :
        //
        // Contract :
        //

        void removeLowPriority(
            int numToRemove,
            float priorityThreshold = 0.2
        );
        // Usage :
        //
        // Contract :
        //


//---------------------------------------------------------------- PRIVATE
    private:
        std::deque<Transition> buffer;
        int capacity;  // should be in [100000, 1000000] to keep only transitions that can be useful

        // random number generator for sampling
        std::mt19937 gen;
};



#endif //REPLAY_BUFFER_H
