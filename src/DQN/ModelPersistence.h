/*************************************************************************
ModelPersistence - Saves and restores DQN training checkpoints
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//---- Interface of the module <ModelPersistence> (file ModelPersistence.h) ----
#ifndef MODEL_PERSISTENCE_H
#define MODEL_PERSISTENCE_H

//------------------------------------------------------------------------
// Role of the <ModelPersistence> module
// Persist the state required to resume training in a compatible learner.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <iosfwd>
#include <string>

//-------------------------------------------------------- Used interfaces
#include "DQN.h"

//----------------------------------------------------------------- PUBLIC
class ModelPersistence {
//--------------------------------------------------------- Public Methods
    public:
        void saveDQN(const DQN &dqn, const std::string &filename, int trainingStep) const;
        // Contract : Replace the checkpoint only after successfully writing its complete contents.

        int loadDQN(DQN &dqn, const std::string &filename) const;
        // Contract : Restore into a compatible learner and return the saved training step.
        // The starting epsilon and seeds may differ. The saved exploration state takes precedence.
        // Throw on invalid or incompatible input and leave the learner unchanged.

//---------------------------------------------------------------- PRIVATE
    private:
        static void writeState(std::ostream &file, const DQN &dqn,
                               const EncodedState &state, bool allowEmpty);
        static EncodedState readState(std::istream &file, const DQN &dqn,
                                      ActionMask mask, bool allowEmpty);
};

#endif //MODEL_PERSISTENCE_H
