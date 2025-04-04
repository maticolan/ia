#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <queue>

using namespace std;


struct Tablero {
    vector<vector<int>> matriz;
    int dimension;

    Tablero(int N) : dimension(N) {
        matriz = vector<vector<int>>(N, vector<int>(N, 0));
    }

    void mostrarTablero() {
        for (const auto& fila : matriz) {
            for (int valor : fila) {
                if (valor == 0) {
                    cout << "- ";
                }
                else if (valor == 1) {
                    cout << "X ";
                }
                else if (valor == 2) {
                    cout << "O ";
                }
            }
            cout << endl;
        }
    }

    bool estaVacio(int fila, int columna) {
        return matriz[fila][columna] == 0;
    }

    void realizarMovimiento(int fila, int columna, int jugador) {
        matriz[fila][columna] = jugador;
    }

    bool tableroCompleto() {
        for (const auto& fila : matriz) {
            for (int valor : fila) {
                if (valor == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    bool hayGanador(int jugador) {
        // Verificar filas y columnas
        for (int i = 0; i < dimension; ++i) {
            bool ganadorFila = true;
            bool ganadorColumna = true;
            for (int j = 0; j < dimension; ++j) {
                if (matriz[i][j] != jugador) {
                    ganadorFila = false;
                }
                if (matriz[j][i] != jugador) {
                    ganadorColumna = false;
                }
            }
            if (ganadorFila || ganadorColumna) {
                return true;
            }
        }

        // Verificar diagonales
        bool ganadorDiagonal1 = true;
        bool ganadorDiagonal2 = true;
        for (int i = 0; i < dimension; ++i) {
            if (matriz[i][i] != jugador) {
                ganadorDiagonal1 = false;
            }
            if (matriz[i][dimension - i - 1] != jugador) {
                ganadorDiagonal2 = false;
            }
        }
        if (ganadorDiagonal1 || ganadorDiagonal2) {
            return true;
        }

        return false;
    }
};

struct TreeMinMax {
    vector<vector<int>> tab;
    int v;
    int winner = 0;
    vector<TreeMinMax*> children;
};

vector<vector<vector<int>>> PossMov(vector<vector<int>> tab, int player) {
    int n_v = tab.size();
    vector<vector<vector<int>>> possibleMoves;
    for (int i = 0; i < n_v; i++) {
        for (int j = 0; j < n_v; j++) {
            if (tab[i][j] == 0) {  // If the cell is empty
                vector<vector<int>> newtab = tab;
                newtab[i][j] = player;
                possibleMoves.push_back(newtab);
            }
        }
    }
    return possibleMoves;
}

int calPos(TreeMinMax& TreeMinMax, int player) {
    int possibilities = 0;
    int tabSize = TreeMinMax.tab.size();
    // Rows
    for (int i = 0; i < tabSize; i++) {
        bool possible = true;
        for (int j = 0; j < tabSize; j++) {
            if (TreeMinMax.tab[i][j] != player && TreeMinMax.tab[i][j] != 0) {
                possible = false;
                break;
            }
        }
        if (possible)
            possibilities++;
    }
    // Columns
    for (int i = 0; i < tabSize; i++) {
        bool possible = true;
        for (int j = 0; j < tabSize; j++) {
            if (TreeMinMax.tab[j][i] != player && TreeMinMax.tab[j][i] != 0) {
                possible = false;
                break;
            }
        }
        if (possible)
            possibilities++;
    }
    // Diagonals
    bool possible = true;
    for (int i = 0; i < tabSize; i++) {
        if (TreeMinMax.tab[i][i] != player && TreeMinMax.tab[i][i] != 0) {
            possible = false;
            break;
        }
    }
    if (possible)
        possibilities++;
    possible = true;
    for (int i = 0; i < tabSize; i++) {
        if (TreeMinMax.tab[i][tabSize - i - 1] != player && TreeMinMax.tab[i][tabSize - i - 1] != 0) {
            possible = false;
            break;
        }
    }
    if (possible)
        possibilities++;

    return possibilities;
}

TreeMinMax* createTreeMM(vector<vector<int>> tab, int player, int depth) {
    TreeMinMax* treeNode = new TreeMinMax(); // Rename the variable to avoid conflict
    treeNode->tab = tab;
    treeNode->v = calPos(*treeNode, 1) - calPos(*treeNode, 2);
    if (treeNode->winner != 0) {
        return treeNode;
    }
    if (depth > 0) {
        vector<vector<vector<int>>> possibleMoves = PossMov(tab, player);
        for (const auto& move : possibleMoves) {
            TreeMinMax* child = createTreeMM(move, player == 1 ? 2 : 1, depth - 1);
            treeNode->children.push_back(child); // Corrected variable name here as well
        }
    }
    return treeNode;
}


void deleteTree(TreeMinMax* TreeNode) {
    if (TreeNode == nullptr) {
        return;
    }
    for (TreeMinMax* child : TreeNode->children) {
        deleteTree(child);
    }
    delete TreeNode;
}

int minMax(TreeMinMax* treeNode, int depth, bool isMaximizingPlayer) {
    if (depth == 0 || treeNode->children.empty()) {
        return treeNode->v;
    }
    if (isMaximizingPlayer) {
        int maxEval = INT_MIN;
        for (TreeMinMax* child : treeNode->children) {
            int eval = minMax(child, depth - 1, false);
            maxEval = max(maxEval, eval);
        }
        return maxEval;
    }
    else {
        int minEval = INT_MAX;
        for (TreeMinMax* child : treeNode->children) {
            int eval = minMax(child, depth - 1, true);
            minEval = min(minEval, eval);
        }
        return minEval;
    }
}


pair<vector<vector<int>>, int> bestMove(TreeMinMax* TreeNode, int depth) {
    vector<vector<int>> besttab;
    int bestv = INT_MIN;  // Inicializar con el valor mínimo posible
    for (TreeMinMax* child : TreeNode->children) {
        int eval = minMax(child, depth, false);
        if (eval > bestv) {
            bestv = eval;
            besttab = child->tab;
        }
    }
    return make_pair(besttab, bestv);
}


int main() {
    int d, p;
    cout << "DAME LA DIMENSION DE JUEGO: ";
    cin >> d;
    cout << endl;
    cout << "DAME LA PROFUNDIDAD DE ARBOL: ";
    cin >> p;
    cout << endl;

    int dimension = d;
    int profundidad = p;

    Tablero tablero(dimension);

    string ini;
    cout << "DIME QUIEN INICIA EL JUEGO (HUMANO O COMPUTADOR): ";
    cin >> ini;
    cout << endl;

    int jugadorHumano = 1; // Representa 'X'
    int jugadorMaquina = 2; // Representa 'O'
    bool turnoHumano = (ini == "HUMANO");

    while (true) {
        cout << "Tablero actual:" << endl;
        tablero.mostrarTablero();

        if (turnoHumano) {
            // Turno del jugador humano
            int fila, columna;
            cout << "Ingrese la fila y columna para su movimiento (ejemplo: 1 2): ";
            cin >> fila >> columna;
            fila--;
            columna--;

            if (fila < 0 || fila >= dimension || columna < 0 || columna >= dimension || !tablero.estaVacio(fila, columna)) {
                cout << "Movimiento inválido. Inténtelo de nuevo." << endl;
                continue;
            }

            tablero.realizarMovimiento(fila, columna, jugadorHumano);
            cout << "Jugador humano ha hecho su movimiento." << endl;
        }
        else {
            // Turno de la computadora
            TreeMinMax* raiz = createTreeMM(tablero.matriz, jugadorMaquina, profundidad);
            pair<vector<vector<int>>, int> mejorMovimiento = bestMove(raiz, profundidad);
            cout << "Valor del mejor movimiento de la computadora: " << mejorMovimiento.second << endl;
            cout << "Movimiento de la computadora:" << endl;
            for (const auto& fila : mejorMovimiento.first) {
                for (int valor : fila) {
                    if (valor == 0) {
                        cout << "- ";
                    }
                    else if (valor == 1) {
                        cout << "X ";
                    }
                    else if (valor == 2) {
                        cout << "O ";
                    }
                }
                cout << endl;
            }

            /* queue<TreeMinMax> q;
            q.push(raiz);
            while (!q.empty()) {
                TreeMinMax* current = q.front();
                q.pop();
                cout << "Valor del nodo: " << current->v << endl;
                for (TreeMinMax* child : current->children) {
                    q.push(child);
                }
            }*/
            tablero.matriz = mejorMovimiento.first;
            cout << "La máquina ha realizado su movimiento." << endl;
            deleteTree(raiz);
        }

        // Verificar el estado del juego después del movimiento
        if (tablero.hayGanador(jugadorHumano)) {
            cout << "¡Felicidades! ¡Has ganado!" << endl;
            break;
        }
        else if (tablero.hayGanador(jugadorMaquina)) {
            cout << "¡La máquina ha ganado!" << endl;
            break;
        }
        else if (tablero.tableroCompleto()) {
            cout << "¡Empate! El tablero está lleno." << endl;
            break;
        }

        turnoHumano = !turnoHumano; // Cambiar al siguiente turno
    }

    // Limpiar el árbol
    return 0;
}