/*************************************************************************
TestMain - Runs component tests independently
                             -------------------
    copyright            : (C) 2026 by Enzo DOS ANJOS
*************************************************************************/

//----- Implementation of the module <TestMain> (file TestMain.cpp) ------

//---------------------------------------------------------------- INCLUDE
//--------------------------------------------------------- System Include
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std;

//------------------------------------------------------- Personal Include
#include "Test.h"


//----------------------------------------------------------------- PUBLIC
//--------------------------------------------------------- Public Methods
int main(int argc, char **argv)
// Algorithm : Runs named component tests in isolated child processes and reports assertions, crashes and timeouts.
{
    if (argc > 2)
    {
        cerr << "Use: suite [test_name]" << endl;
        return 2;
    }

    vector<TestCase> tests = getTests();
    int passed = 0;
    int failed = 0;

    for (const auto &test : tests)
    {
        if (argc == 2 && test.name != argv[1])
        {
            continue;
        }

        string pattern = (filesystem::temp_directory_path() / "snake-test-XXXXXX").string();
        // mkdtemp replaces XXXXXX in a writable, null-terminated buffer.
        vector<char> directory(pattern.begin(), pattern.end());
        directory.push_back('\0');

        if (mkdtemp(directory.data()) == nullptr)
        {
            cerr << "Cannot create test directory" << endl;
            return 1;
        }

        // Flush before fork so the child cannot inherit pending output from the parent.
        cout.flush();
        pid_t child = fork();

        if (child == 0)
        {
            // Bound each test’s runtime and suppress core files when an assertion or crash is expected.
            rlimit limit{0, 0};
            setrlimit(RLIMIT_CORE, &limit);
            alarm(20);

            try
            {
                filesystem::current_path(directory.data());
                test.run();
                _exit(0);
            }
            catch (const exception &error)
            {
                cerr << "    " << error.what() << endl;
                _exit(1);
            }
        }

        int status = 0;
        pid_t result;

        // A signal may interrupt waiting without the child having finished.
        do
        {
            result = child < 0 ? -1 : waitpid(child, &status, 0);
        } while (result < 0 && errno == EINTR);

        filesystem::remove_all(directory.data());

        if (result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0)
        {
            cout << "PASS " << test.name << endl;
            passed++;
        }
        else
        {
            cout << "FAIL " << test.name;

            if (result > 0 && WIFSIGNALED(status))
            {
                cout << " (signal " << WTERMSIG(status) << ")";
            }

            cout << endl;
            failed++;
        }
    }

    cout << passed << " passed, " << failed << " failed" << endl;
    return failed != 0 || passed == 0;
}  //----- end of main
