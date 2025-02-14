#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <utility>
#include <deque>

using namespace std;


int main() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;  // grow 1 block every M rounds
    cin >> N >> P;  // nb of player and player's turn

    cout << "> Received grid size: " << W << "x" << H << endl;
    cout << "> Growth rate: every " << M << " turns" << endl;
    cout << "> Number of players: " << N << " (I am player " << P << ")" << endl;

    typedef pair<int, int> coord;

    unordered_map<int, deque<coord>> Players;
    for (int i = 1; i <= N; i++) {
        cin >> X >> Y;
        Players[i].push_back(make_pair(X, Y));
        cout << "> Player " << (i) << " starts at (" << X << "," << Y << ")" << endl;
    }

    map<string, coord> directions = { {"left", make_pair(-1, 0)}, {"right", make_pair(1, 0)}, 
                                               {"up", make_pair(0, 1)}, {"down", make_pair(0, -1)} };

    // directionsCycle to cycle through
    const string directionsCycle[] = {"right", "down", "left", "up"};
    int dir_index = 0;

    // Immediate move if we're first player
    if(P == 1) {
        cout << directionsCycle[dir_index++] << endl;
        dir_index %= 4;
    }

    // Game loop
    string move;
    string trash;
    int moveLen;

    int turn = 0;
    int snakeSize = 1;
    
    while(true) {        
        // Read opponent moves
        cin >> trash >> moveLen >> move;
        if(!move.empty()) {
            cout << "> Received opponent move: " << move << ", " << move << endl;
        }

        if (P == 1) {            
            if (turn%(M) == 0) {
                snakeSize++;
            } else {
                Players[2].pop_front();
            }
            coord currentHead = Players[0][Players[0].size()-1];
            Players[2].push_back(make_pair(currentHead.first + directions[move].first, currentHead.second + directions[move].second));
            turn++;

            cout << "> Snake Size: " << snakeSize << endl;
            cout << "> turn: " << turn << endl;

            for (int i = 0; i < snakeSize; i++) {
                 cout << "> Snake part " << i << ": " << Players[2][i].first << ", " << Players[2][i].second << endl;
            }
        }

        // Send our move
        cout << "> Choosing direction: " << directions[directionsCycle[dir_index]].first << ", " << directions[directionsCycle[dir_index]].second << endl;
        cout << directionsCycle[dir_index++] << endl;
        dir_index %= 4;

        if (P == 2) {
            if (turn%(M) == 0) {
                snakeSize++;
            }
            turn++;
            
            cout << "> Snake Size: " << snakeSize << endl;
            cout << "> turn: " << turn << endl;
        }

        // Flush output buffer
        cout.flush();
    }

    return 0;
}