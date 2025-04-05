// ==================== LIBRERÍAS ====================
#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <climits>
#include <algorithm>
#include <map>

using namespace std;

// ==================== DEFINICIONES ====================
const int GRAFO_WIDTH = 400;  // 40 * STEP
const int GRAFO_HEIGHT = 200; // 20 * STEP

const int STEP = 10;                  // Distancia entre nodos lógicos
const float PERCENT_DELETED = 0.2f;
const int INF = INT_MAX;
const int WINDOW_WIDTH = 600;         // Tamaño de la ventana
const int WINDOW_HEIGHT = 600;

const float SCALE_X = (float)WINDOW_WIDTH / GRAFO_WIDTH;
const float SCALE_Y = (float)WINDOW_HEIGHT / GRAFO_HEIGHT;

// ==================== ESTRUCTURAS ====================
struct Punto {
    int x, y;
    bool deleted = false;
};

struct Nodo {
    int x, y;
    int costo;
    bool operator<(const Nodo& o) const {
        return costo > o.costo;
    }
};

struct Grafo {
    vector<vector<Punto>> puntos;
    vector<vector<bool>> visitados;

    Grafo(int width, int height) {
        int cols = width / STEP;
        int rows = height / STEP;
        puntos.resize(cols, vector<Punto>(rows));
        visitados.resize(cols, vector<bool>(rows, false));
    }
};

// ==================== VARIABLES GLOBALES ====================
Grafo grafo(GRAFO_WIDTH, GRAFO_HEIGHT);
int startX = 0, startY = 0;
int targetX = (GRAFO_WIDTH / STEP) - 1, targetY = (GRAFO_HEIGHT / STEP) - 1;
int dx[] = {1, -1, 0, 0, -1, 1, -1, 1};
int dy[] = {0, 0, 1, -1, 1, 1, -1, -1};
vector<pair<int, int>> caminosExplorados;
vector<pair<int, int>> caminoMasCorto;

// ==================== FUNCIONES UTILITARIAS ====================
void drawBigPoint(int x, int y, float r, float g, float b) {
    glColor3f(r, g, b);
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glVertex2i(x * SCALE_X, y * SCALE_Y);
    glEnd();
}

void drawLineBetween(Punto& a, Punto& b) {
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    glVertex2i(a.x * SCALE_X, a.y * SCALE_Y);
    glVertex2i(b.x * SCALE_X, b.y * SCALE_Y);
    glEnd();
}

void marcarCamino(vector<pair<int, int>>& camino, float r, float g, float b) {
    for (auto& [x, y] : camino) {
        drawBigPoint(grafo.puntos[x][y].x, grafo.puntos[x][y].y, r, g, b);
    }
}

vector<pair<int, int>> reconstruirCamino(map<pair<int, int>, pair<int, int>>& padre, int tx, int ty) {
    vector<pair<int, int>> camino;
    pair<int, int> actual = {tx, ty};
    while (padre.count(actual)) {
        camino.push_back(actual);
        actual = padre[actual];
    }
    camino.push_back(actual);
    reverse(camino.begin(), camino.end());
    return camino;
}

bool esValido(Grafo& g, int x, int y) {
    return x >= 0 && y >= 0 && x < g.puntos.size() && y < g.puntos[0].size() && !g.puntos[x][y].deleted && !g.visitados[x][y];
}

void resetVisitados(Grafo& g) {
    for (auto& fila : g.visitados) fill(fila.begin(), fila.end(), false);
}

// ==================== FUNCIONES DE BÚSQUEDA ====================
void AEstrella(Grafo& g, int sx, int sy, int tx, int ty) {
    resetVisitados(g);
    priority_queue<Nodo> pq;
    map<pair<int, int>, pair<int, int>> padre;
    vector<vector<int>> costo(GRAFO_WIDTH / STEP, vector<int>(GRAFO_HEIGHT / STEP, INF));

    pq.push({sx, sy, 0});
    costo[sx][sy] = 0;
    vector<pair<int, int>> explorado;

    while (!pq.empty()) {
        Nodo actual = pq.top(); pq.pop();
        int x = actual.x, y = actual.y;
        if (g.visitados[x][y]) continue;
        g.visitados[x][y] = true;
        if (x == tx && y == ty) break;

        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (esValido(g, nx, ny)) {
                int nuevoCosto = costo[x][y] + 1;
                if (nuevoCosto < costo[nx][ny]) {
                    costo[nx][ny] = nuevoCosto;
                    int prioridad = nuevoCosto + abs(nx - tx) + abs(ny - ty);
                    pq.push({nx, ny, prioridad});
                    padre[{nx, ny}] = {x, y};
                    explorado.push_back({nx, ny});
                }
            }
        }
    }

    caminosExplorados = explorado;
    caminoMasCorto = reconstruirCamino(padre, tx, ty);
}

void BFS(Grafo& g, int sx, int sy, int tx, int ty) {
    resetVisitados(g);
    queue<pair<int, int>> q;
    map<pair<int, int>, pair<int, int>> padre;
    vector<pair<int, int>> explorado;

    q.push({sx, sy});
    g.visitados[sx][sy] = true;

    while (!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        if (x == tx && y == ty) break;

        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (esValido(g, nx, ny)) {
                g.visitados[nx][ny] = true;
                padre[{nx, ny}] = {x, y};
                q.push({nx, ny});
                explorado.push_back({nx, ny});
            }
        }
    }

    caminosExplorados = explorado;
    caminoMasCorto = reconstruirCamino(padre, tx, ty);
}

void DFS(Grafo& g, int sx, int sy, int tx, int ty) {
    resetVisitados(g);
    stack<pair<int, int>> s;
    map<pair<int, int>, pair<int, int>> padre;
    vector<pair<int, int>> explorado;

    s.push({sx, sy});

    while (!s.empty()) {
        auto [x, y] = s.top(); s.pop();
        if (g.visitados[x][y]) continue;
        g.visitados[x][y] = true;
        explorado.push_back({x, y});
        if (x == tx && y == ty) break;

        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (esValido(g, nx, ny)) {
                s.push({nx, ny});
                padre[{nx, ny}] = {x, y};
            }
        }
    }

    caminosExplorados = explorado;
    caminoMasCorto = reconstruirCamino(padre, tx, ty);
}

void HillClimbing(Grafo& g, int sx, int sy, int tx, int ty) {
    resetVisitados(g);
    map<pair<int, int>, pair<int, int>> padre;
    vector<pair<int, int>> explorado;

    int x = sx, y = sy;
    g.visitados[x][y] = true;

    while (x != tx || y != ty) {
        int mejorHeuristica = INF;
        int mejorX = -1, mejorY = -1;

        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if (esValido(g, nx, ny)) {
                int heuristica = abs(nx - tx) + abs(ny - ty);
                if (heuristica < mejorHeuristica) {
                    mejorHeuristica = heuristica;
                    mejorX = nx;
                    mejorY = ny;
                }
            }
        }

        if (mejorX == -1) break;
        padre[{mejorX, mejorY}] = {x, y};
        x = mejorX;
        y = mejorY;
        g.visitados[x][y] = true;
        explorado.push_back({x, y});
    }

    caminosExplorados = explorado;
    caminoMasCorto = reconstruirCamino(padre, x, y);
}

// ==================== GRAFICADO ====================
void drawGraph(Grafo& g) {
    int gridSize =400 / STEP;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            Punto& p = g.puntos[i][j];
            if (!p.deleted) {
                drawBigPoint(p.x, p.y, 0.0f, 1.0f, 0.0f);
                for (int k = 0; k < 8; k++) {
                    int ni = i + dx[k], nj = j + dy[k];
                    if (ni >= 0 && nj >= 0 && ni < gridSize && nj < gridSize && !g.puntos[ni][nj].deleted) {
                        drawLineBetween(p, g.puntos[ni][nj]);
                    }
                }
            }
        }
    }

    marcarCamino(caminosExplorados, 0.7f, 0.7f, 0.7f);
    marcarCamino(caminoMasCorto, 0.0f, 0.0f, 1.0f);
    drawBigPoint(g.puntos[startX][startY].x, grafo.puntos[startX][startY].y, 1.0f, 1.0f, 0.0f);
    drawBigPoint(g.puntos[targetX][targetY].x, grafo.puntos[targetX][targetY].y, 1.0f, 0.0f, 1.0f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    drawGraph(grafo);
    glutSwapBuffers();
}

void keyboard(unsigned char key, int, int) {
    switch (key) {
        case 27: exit(0); break;
        case '1': AEstrella(grafo, startX, startY, targetX, targetY); break;
        case '2': BFS(grafo, startX, startY, targetX, targetY); break;
        case '3': DFS(grafo, startX, startY, targetX, targetY); break;
        case '4': HillClimbing(grafo, startX, startY, targetX, targetY); break;
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
    if (state != GLUT_DOWN) return;
    int gx = x / (WINDOW_WIDTH / (GRAFO_WIDTH / STEP));
    int gy = (WINDOW_HEIGHT - y) / (WINDOW_HEIGHT / (GRAFO_HEIGHT / STEP));

    if (gx >= 0 && gy >= 0 && gx < grafo.puntos.size() && gy < grafo.puntos[0].size() && !grafo.puntos[gx][gy].deleted) {
        if (button == GLUT_LEFT_BUTTON) {
            startX = gx;
            startY = gy;
        } else if (button == GLUT_RIGHT_BUTTON) {
            targetX = gx;
            targetY = gy;
        }
        AEstrella(grafo, startX, startY, targetX, targetY);
        glutPostRedisplay();
    }
}

void initGraph(Grafo& g) {
    srand(time(NULL));
   for (int i = 0; i < GRAFO_WIDTH; i += STEP){
    for (int j = 0; j < GRAFO_HEIGHT; j += STEP){
 
        int xi = i / STEP, yj = j / STEP;
            g.puntos[xi][yj].x = i;
            g.puntos[xi][yj].y = j;
            g.puntos[xi][yj].deleted = ((rand() / (float)RAND_MAX) < PERCENT_DELETED);
        }
    }
}

// ==================== MAIN ====================
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Visualizador de Algoritmos de Búsqueda");

    glewInit();
    initGraph(grafo);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMainLoop();

    return 0;
}
