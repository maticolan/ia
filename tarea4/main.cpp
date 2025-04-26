#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>

using namespace std;

struct Nodo {
	float x, y;
	char etiqueta;
};

vector<Nodo> nodos;
vector<int> mejorRecorrido;
float mejorDistancia = 1e9;
bool corriendoGenetico = false;
int generacionActual = 0;
int maxGeneraciones = 100;
mt19937 rng(random_device{}());
vector<vector<int>> poblacion;
ofstream archivoSalida("datos_evolucion.txt");
	

void inicializarPoblacion() {
	poblacion.clear();
	vector<int> base(nodos.size());
	for (int i = 0; i < nodos.size(); i++) base[i] = i;
	for (int i = 0; i < 20; i++) { // Tamaño población
		shuffle(base.begin(), base.end(), rng);
		poblacion.push_back(base);
	}
}

float calcularDistancia(const vector<int>& recorrido) {
	float dist = 0;
	for (int i = 0; i < recorrido.size(); i++) {
		Nodo a = nodos[recorrido[i]];
		Nodo b = nodos[recorrido[(i + 1) % recorrido.size()]];
		dist += hypot(b.x - a.x, b.y - a.y);
	}
	return dist;
}

vector<int> pmx(const vector<int>& p1, const vector<int>& p2) {
	int n = p1.size();
	vector<int> hijo(n, -1);
	int a = rng() % n, b = rng() % n;
	if (a > b) swap(a, b);
	for (int i = a; i <= b; i++) hijo[i] = p1[i];
	
	for (int i = a; i <= b; i++) {
		if (find(hijo.begin(), hijo.end(), p2[i]) == hijo.end()) {
			int pos = i;
			while (hijo[pos] != -1) {
				pos = find(p2.begin(), p2.end(), p1[pos]) - p2.begin();
			}
			hijo[pos] = p2[i];
		}
	}
	for (int i = 0; i < n; i++) {
		if (hijo[i] == -1) hijo[i] = p2[i];
	}
	return hijo;
}

void avanzarGeneracion() {
	if (corriendoGenetico) {
		vector<vector<int>> nuevaGeneracion;
		nuevaGeneracion.push_back(mejorRecorrido);
		
		while (nuevaGeneracion.size() < poblacion.size()) {
			int idx1 = rng() % poblacion.size();
			int idx2 = rng() % poblacion.size();
			vector<int> hijo = pmx(poblacion[idx1], poblacion[idx2]);
			if (rng() % 100 < 10) {
				int a = rng() % hijo.size();
				int b = rng() % hijo.size();
				swap(hijo[a], hijo[b]);
			}
			nuevaGeneracion.push_back(hijo);
		}
		
		poblacion = nuevaGeneracion;
		
		for (const auto& recorrido : poblacion) {
			float dist = calcularDistancia(recorrido);
			if (dist < mejorDistancia) {
				mejorDistancia = dist;
				mejorRecorrido = recorrido;
			}
		}
		float distanciaTotal = 0;
		for (const auto& recorrido : poblacion) {
			distanciaTotal += calcularDistancia(recorrido);
		}
		float promedioDistancia = distanciaTotal / poblacion.size();
		
		// Escribir en el archivo
		if (archivoSalida.is_open()) {
			archivoSalida << generacionActual << " " 
				<< mejorDistancia << " " 
				<< promedioDistancia << endl;
		}
		
		cout << "Generación " << generacionActual << ": Mejor distancia = " << mejorDistancia << endl;
		
		generacionActual++;
		if (generacionActual >= maxGeneraciones) {
			corriendoGenetico = false;
			cout << "Evolución terminada." << endl;
		}
		
		glutPostRedisplay();
	}
}

void timer(int = 0) {
	avanzarGeneracion();
	glutTimerFunc(30, timer, 0);
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(10);
	
	glColor3f(1, 0.5, 0.5);
	glBegin(GL_POINTS);
	for (auto& nodo : nodos) {
		glVertex2f(nodo.x, nodo.y);
	}
	glEnd();
	
	// Dibujar etiquetas
	for (auto& nodo : nodos) {
		glRasterPos2f(nodo.x + 0.01, nodo.y + 0.01);
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, nodo.etiqueta);
	}
	
	// Dibujar recorrido si existe
	if (!mejorRecorrido.empty()) {
		glColor3f(0, 0, 0);
		glBegin(GL_LINE_LOOP);
		for (auto idx : mejorRecorrido) {
			glVertex2f(nodos[idx].x, nodos[idx].y);
		}
		glEnd();
	}
	
	glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float xf = (float)x / glutGet(GLUT_WINDOW_WIDTH);
		float yf = 1.0f - (float)y / glutGet(GLUT_WINDOW_HEIGHT);
		char etiqueta = 'A' + nodos.size();
		nodos.push_back({xf, yf, etiqueta});
		mejorRecorrido.clear();
		glutPostRedisplay();
	}
}

void key(unsigned char k, int, int) {
	switch (k) {
	case 'g':
		if (!corriendoGenetico && !nodos.empty()) {
			cout << "Iniciando algoritmo genético..." << endl;
			inicializarPoblacion();
			mejorDistancia = calcularDistancia(poblacion[0]);
			mejorRecorrido = poblacion[0];
			generacionActual = 0;
			corriendoGenetico = true;
		}
		break;
	case 'r':
		nodos.clear();
		mejorRecorrido.clear();
		corriendoGenetico = false;
		glutPostRedisplay();
		break;
	case 27: // Escape
		exit(0);
		break;
	}
}

void init() {
	glClearColor(1, 1, 1, 1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 1, 0, 1);
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(800, 600);
	glutCreateWindow("TSP Genético");
	init();
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutKeyboardFunc(key);
	glutTimerFunc(30, timer, 0);
	atexit([]() { if (archivoSalida.is_open()) archivoSalida.close(); });	
	glutMainLoop();
}

