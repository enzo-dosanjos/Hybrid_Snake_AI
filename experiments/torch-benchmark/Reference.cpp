/*************************************************************************
Reference - Measures production DQN updates on a fixed replay batch
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//--------- Implementation of the module <Reference> (file Reference.cpp) ---------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <chrono>
#include <iostream>

using namespace std;

//------------------------------------------------------- Personal Include
#include "src/ApplicationModel.h"
#include "src/DQN/ModelPersistence.h"

//----------------------------------------------------------------- PUBLIC
int main()
// Algorithm : Restore the fixture, save one reference update and time subsequent full-batch updates.
{
    DQN learner(makeApplicationModelConfig(2), mt19937(17));
    ModelPersistence persistence;
    int step = persistence.loadDQN(learner, "batch.model");
    learner.train(32, 1e-5f);
    persistence.saveDQN(learner, "after.model", step + 1);

    for (int i = 0; i < 10; i++)
    {
        learner.train(32, 1e-5f);
    }

    for (int i = 0; i < 50; i++)
    {
        auto start = chrono::steady_clock::now();
        learner.train(32, 1e-5f);
        cout << "ms " << chrono::duration<double, milli>(chrono::steady_clock::now() - start).count() << '\n';
    }
}  //----- end of main
