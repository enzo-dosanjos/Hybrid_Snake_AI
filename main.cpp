#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;


int main() {
    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;  // grow 1 block every M rounds
    cin >> N >> P;  // nb of player and player's turn

    cout << "> Received grid size: " << W << "x" << H << endl;
    cout << "> Growth rate: every " << M << " turns" << endl;
    cout << "> Number of players: " << N << " (I am player " << P << ")" << endl;

    unordered_map<int, vector<pair<int, int>>> Players;
    for (int i = 0; i < N; i++) {
        cin >> X >> Y;
        Players[i].push_back(make_pair(X, Y));
        cout << "> Player " << (i+1) << " starts at (" << X << "," << Y << ")" << endl;
    }

    // Directions to cycle through
    const string directions[] = {"right", "down", "left", "up"};
    int dir_index = 0;

    // Immediate move if we're first player
    if(P == 1) {
        cout << directions[dir_index++] << endl;
        dir_index %= 4;
    }

    // Game loop
    string move;
    int turn = 0;
    while(true) {        
        // Read opponent moves
        getline(cin, move);
        if(!move.empty()) {
            cout << "> Received opponent move: " << move << endl;
        }

        // Send our move
        cout << "> Choosing direction: " << directions[dir_index] << endl;
        cout << directions[dir_index++] << endl;
        dir_index %= 4;
        
        // Flush output buffer
        cout.flush();
    }

    return 0;
}