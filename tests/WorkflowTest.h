/*************************************************************************
WorkflowTest - Runs and captures workflow executables in integration tests
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//------- Interface of the module <WorkflowTest> (file WorkflowTest.h) ----
#ifndef WORKFLOWTEST_H
#define WORKFLOWTEST_H

//------------------------------------------------------------------------
// Role of the <WorkflowTest> module
// Launch workflow executables from isolated test directories.
//------------------------------------------------------------------------

//--------------------------------------------------------- System Include
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

//-------------------------------------------------------- Used interfaces
#include "Test.h"

//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
inline std::string readFile(const std::string &filename)
// Algorithm : Read a complete test output or checkpoint for comparison.
{
    std::ifstream file(filename, std::ios::binary);
    check(bool(file), "Cannot read " + filename);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}  //----- end of readFile


inline std::pair<int, std::string> runExecutable(const std::vector<std::string> &arguments,
                                                const std::string &input = "")
// Algorithm : Run a real entry point with redirected input and capture its exit code and output.
{
    std::ofstream("stdin.txt") << input;
    std::vector<char *> argv;
    for (const auto &argument : arguments)
    {
        argv.push_back(const_cast<char *>(argument.c_str()));
    }

    argv.push_back(nullptr);
    pid_t child = fork();
    check(child >= 0, "Cannot start workflow executable");

    if (child == 0)
    {
        alarm(10);
        if (!std::freopen("stdin.txt", "r", stdin) || !std::freopen("stdout.txt", "w", stdout) ||
            !std::freopen("stderr.txt", "w", stderr))
        {
            _exit(126);
        }

        execv(argv[0], argv.data());
        _exit(127);
    }

    int status;
    pid_t result;
    do
    {
        result = waitpid(child, &status, 0);
    } while (result < 0 && errno == EINTR);

    check(result == child && WIFEXITED(status) && WEXITSTATUS(status) < 126,
          "Workflow executable could not run or timed out");
    return {WEXITSTATUS(status), readFile("stdout.txt")};
}  //----- end of runExecutable

#endif //WORKFLOWTEST_H
