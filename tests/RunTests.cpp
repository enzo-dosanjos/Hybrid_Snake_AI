/*************************************************************************
RunTests - Builds and runs the selected component suites
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//----- Implementation of the module <RunTests> (file RunTests.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cerrno>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int runCommand(const vector<string> &arguments)
// Algorithm : Runs a build or suite directly in a child process and returns its exit status or failure.
{
    vector<char*> command;

    for (const string &argument : arguments)
    {
        // execvp requires mutable pointer types but does not modify these argument strings.
        command.push_back(const_cast<char*>(argument.c_str()));
    }

    command.push_back(nullptr);
    cout.flush();
    pid_t child = fork();

    if (child == 0)
    {
        alarm(120);
        // Execute the argument vector directly, without shell parsing or interpolation.
        execvp(command[0], command.data());
        cerr << "Cannot execute " << arguments[0] << endl;
        _exit(127);
    }

    if (child < 0)
    {
        return 1;
    }

    int status;
    pid_t result;

    // Retry interrupted waits; signal termination still counts as command failure.
    do
    {
        result = waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);

    return result > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}  //----- end of runCommand


// todo: remove after refactoring
vector<string> checkpointSuites(const string &checkpoint)
// Algorithm : Selects temporary refactoring groups without introducing checkpoint dependencies into component tests.
{
    // Temporary refactoring groups; component suites remain independent.
    if (checkpoint == "1")
    {
        return {"game-engine", "input-handler", "game-loop", "observation", "state-analyzer"};
    }

    if (checkpoint == "2")
    {
        return {"game-engine", "input-handler", "game-loop", "observation", "state-analyzer", "minimax"};
    }

    if (checkpoint == "3")
    {
        return {"cnn", "fcnn", "replay-buffer", "dqn", "persistence"};
    }

    throw invalid_argument("Checkpoint must be 1, 2 or 3");
}  //----- end of checkpointSuites


int main(int argc, char **argv)
// Algorithm : Builds and runs selected component suites while reporting compilation failures separately from test failures.
{
    vector<string> suites = {
        "game-engine", "input-handler", "game-loop", "observation", "state-analyzer", "minimax",
        "cnn", "fcnn", "replay-buffer", "dqn", "persistence"
    };
    vector<string> selected = suites;

    if (argc == 2 && string(argv[1]) == "--list")
    {
        for (const string &suite : suites)
        {
            cout << suite << endl;
        }

        return 0;
    }

    try
    {
        // todo: remove after refactoring
        if (argc == 3 && string(argv[1]) == "--checkpoint")
        {
            selected = checkpointSuites(argv[2]);
        }
        else if (argc == 2 && string(argv[1]) != "all")
        {
            bool found = false;

            for (const string &suite : suites)
            {
                found = found || suite == argv[1];
            }

            if (!found)
            {
                throw invalid_argument("Unknown component: " + string(argv[1]));
            }

            selected = {argv[1]};
        }
        else if (argc > 2)
        {
            throw invalid_argument("Use: RunTests [component | all | --list | --checkpoint N]");
        }
    }
    catch (const exception &error)
    {
        cerr << error.what() << endl;
        return 2;
    }

    int passed = 0;
    int failed = 0;
    int blocked = 0;

    for (const string &suite : selected)
    {
        string executable = "build/tests/" + suite;
        cout << "\n" << suite << endl;

        if (runCommand({"make", "--no-print-directory", "-s", executable}) != 0)
        {
            cerr << "BUILD FAILED " << suite << endl;
            blocked++;
            continue;
        }

        if (runCommand({"./" + executable}) == 0)
        {
            passed++;
        }
        else
        {
            failed++;
        }
    }

    cout << "\nSuites: " << passed << " passed, " << failed << " failed, "
         << blocked << " blocked by compilation" << endl;

    return failed != 0 || blocked != 0;
}  //----- end of main
