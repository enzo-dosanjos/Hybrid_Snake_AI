#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;


// Debug file stream
ofstream debugFile("debug.log");  // Will create/overwrite debug.log


int fallHeight(const vector<vector<int>>& board, int x) {
    int y = board[x].size();
    while (y > 0 && board[x][y - 1] == 0) {
        y--;
    }
    return y;
}

int strategy(const vector<vector<int>>& board, int H) {
    int x = 0;
    while (x < board.size() && board[x][H - 1] != 0) {
        x++;
    }
    return x;
}

int main() {
    if (!debugFile) {
        cerr << "Error opening debug file!" << endl;
        return 1;
    }

    int W, H, M, N, P, X, Y;
    cin >> W >> H;  // size of the board
    cin >> M;  // grow M blocks per round
    cin >> N >> P;  // nb of player and player's turn

    debugFile << "Initialized with W=" << W << " H=" << H << " M=" << M << " N=" << N << " P=" << P << endl;

    unordered_map<int, vector<pair<int, int>>> Players;
    for (int i = 0; i < N; i++) {
        cin >> X >> Y;
        Players[i].push_back(make_pair(X, Y));
        debugFile << "Player " << i << " starts at (" << X << "," << Y << ")" << endl;
    }

    vector<vector<int>> board(W, vector<int>(H, 0));
    int player = 0;

    while (true) {
        player = (player % N) + 1;

        int x;

        if (player == P) {
            x = strategy(board, H);
            cout << x << endl;
        } else {
            cin >> x;
        }

        if (x >= 0 && x < W) {
            int y = fallHeight(board, x);
            board[x][y] = player;
        }
    }

    return 0;
}