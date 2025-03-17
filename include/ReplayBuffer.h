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

    void deserialize(const string& data) {
        stringstream ss(data);

        // Vérifier la taille minimale
        if(data.size() < sizeof(reward) + sizeof(done) + sizeof(tdError)) {
            throw runtime_error("Invalid transition data");
        }


        // Deserialize primitive types
        ss.read(reinterpret_cast<char*>(&reward), sizeof(reward));
        ss.read(reinterpret_cast<char*>(&done), sizeof(done));
        ss.read(reinterpret_cast<char*>(&tdError), sizeof(tdError));

        // Deserialize vectors
        auto deserialize_vector = [&](vector<float>& vec) {
            size_t size;
            ss.read(reinterpret_cast<char*>(&size), sizeof(size));
            vec.resize(size);
            ss.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
        };

        deserialize_vector(boardState);
        deserialize_vector(extraState);
        deserialize_vector(nextBoardState);
        deserialize_vector(nextExtraState);

        // Deserialize string
        size_t action_size;
        ss.read(reinterpret_cast<char*>(&action_size), sizeof(action_size));
        vector<char> buffer(action_size + 1); // +1 pour le terminateur nul
        ss.read(buffer.data(), action_size);
        buffer[action_size] = '\0'; // Assurance du terminateur
        action = string(buffer.data());

        // Vérifier la cohérence de la taille de l'action
        if(action_size > 1000) { // Taille max réaliste
            throw runtime_error("Invalid action size");
        }
    }

    string serialize() const {
        stringstream ss;

        // Serialize primitive types
        ss.write(reinterpret_cast<const char*>(&reward), sizeof(reward));
        ss.write(reinterpret_cast<const char*>(&done), sizeof(done));
        ss.write(reinterpret_cast<const char*>(&tdError), sizeof(tdError));

        // Serialize vectors
        auto serialize_vector = [&](const vector<float>& vec) {
            size_t size = vec.size();
            ss.write(reinterpret_cast<const char*>(&size), sizeof(size));
            ss.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        };

        serialize_vector(boardState);
        serialize_vector(extraState);
        serialize_vector(nextBoardState);
        serialize_vector(nextExtraState);

        // Serialize string
        size_t action_size = action.size();
        ss.write(reinterpret_cast<const char*>(&action_size), sizeof(action_size));
        ss.write(action.c_str(), action_size);

        return ss.str();
    }
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
            const std::vector<float> &currentBoardState,
            const std::vector<float> &currentExtraState,
            const std::vector<float> &nextBoardState,
            const std::vector<float> &nextExtraState,
            const std::string &bestMove,
            int turn,
            const PlayersData &Players,
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
            int cap
        ) : capacity (cap),
            gen(std::random_device()())  // Initialize with a random seed
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
            int count
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
            float noiseRatio
        );
        // Usage :
        //
        // Contract :
        //

        void removeLowPriority(
            int nbToRemove,
            float percentileThreshold = 0.2
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
