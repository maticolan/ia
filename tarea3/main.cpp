#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <GL/glut.h>
using namespace std;

struct nodo;
vector<nodo*> nodos;
nodo* nodoSeleccionado = nullptr;
vector<pair<nodo*, nodo*>> aristas;
const float radioNodo = 0.05f;
int modoColoracion = 1; // 1: restrictiva, 2: restringida

struct nodo {
	char letra;
	float x, y;
	string color;
	vector<nodo*> conexiones;
	
	nodo(char l, float _x, float _y) : letra(l), x(_x), y(_y) {}
	
	int grado() const { return conexiones.size(); }
};

void limpiarColores() {
	for(nodo* n : nodos) {
		n->color = "";
	}
}

void conectarNodos(nodo* a, nodo* b) {
	a->conexiones.push_back(b);
	b->conexiones.push_back(a);
}

// Algoritmo restrictivo (greedy por grado)
void coloracionRestrictiva() {
	limpiarColores();
	vector<nodo*> nodosOrdenados = nodos;
	
	sort(nodosOrdenados.begin(), nodosOrdenados.end(), [](nodo* a, nodo* b) {
		return a->grado() > b->grado();
	});
	
	vector<string> colores = {"rojo", "verde", "azul"};
	
	for(nodo* n : nodosOrdenados) {
		vector<string> coloresProhibidos;
		for(nodo* vecino : n->conexiones) {
			if(!vecino->color.empty()) {
				coloresProhibidos.push_back(vecino->color);
			}
		}
		
		for(string color : colores) {
			if(find(coloresProhibidos.begin(), coloresProhibidos.end(), color) == coloresProhibidos.end()) {
				n->color = color;
				break;
			}
		}
	}
}

// Algoritmo restringido (heurística MRV)
void coloracionRestringida() {
	limpiarColores();
	vector<nodo*> nodosOrdenados = nodos;
	
	sort(nodosOrdenados.begin(), nodosOrdenados.end(), [](nodo* a, nodo* b) {
		return a->conexiones.size() < b->conexiones.size();
	});
	
	vector<string> colores = {"rojo", "verde", "azul"};
	
	for(nodo* n : nodosOrdenados) {
		vector<string> coloresDisponibles = colores;
		
		for(nodo* vecino : n->conexiones) {
			auto it = find(coloresDisponibles.begin(), coloresDisponibles.end(), vecino->color);
			if(it != coloresDisponibles.end()) {
				coloresDisponibles.erase(it);
			}
		}
		
		if(!coloresDisponibles.empty()) {
			n->color = coloresDisponibles[0];
		} else {
			// Backtracking simple
			n->color = "rojo";
			for(nodo* vecino : n->conexiones) {
				if(vecino->color == "rojo") {
					n->color = "verde";
					break;
				}
			}
			if(n->color == "rojo") return;
			
			for(nodo* vecino : n->conexiones) {
				if(vecino->color == "verde") {
					n->color = "azul";
					break;
				}
			}
		}
	}
}

void dibujarCirculo(float cx, float cy, float r, int num_segments, float color[3]) {
	glColor3fv(color);
	glBegin(GL_TRIANGLE_FAN);
	for(int i = 0; i < num_segments; i++) {
		float theta = 2.0f * 3.1415926f * float(i) / float(num_segments);
		float x = r * cosf(theta);
		float y = r * sinf(theta);
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

void dibujarTexto(const char* texto, float x, float y) {
	glColor3f(0, 0, 0);
	glRasterPos2f(x, y);
	for(const char* c = texto; *c != '\0'; c++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
	}
}

void display() {
	glClearColor(0.8, 0.8, 0.8, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	
	// Dibujar aristas
	glColor3f(0, 0, 0);
	glLineWidth(2.0);
	for(auto& arista : aristas) {
		glBegin(GL_LINES);
		glVertex2f(arista.first->x, arista.first->y);
		glVertex2f(arista.second->x, arista.second->y);
		glEnd();
	}
	
	// Dibujar nodos
	for(nodo* n : nodos) {
		float color[3] = {0.5, 0.5, 0.5}; // Color gris por defecto
		if(n->color == "rojo") color[0] = 1, color[1] = 0, color[2] = 0;
		else if(n->color == "verde") color[0] = 0, color[1] = 1, color[2] = 0;
		else if(n->color == "azul") color[0] = 0, color[1] = 0, color[2] = 1;
		
		dibujarCirculo(n->x, n->y, radioNodo, 50, color);
		
		float borde[3] = {0, 0, 0};
		dibujarCirculo(n->x, n->y, radioNodo, 50, borde);
		dibujarCirculo(n->x, n->y, radioNodo * 0.9, 50, color);
		
		char letra[2] = {n->letra, '\0'};
		dibujarTexto(letra, n->x - 0.015, n->y - 0.02);
	}
	
	glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {
	if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		int winWidth = glutGet(GLUT_WINDOW_WIDTH);
		int winHeight = glutGet(GLUT_WINDOW_HEIGHT);
		float mx = (x / (float)winWidth) * 2 - 1;
		float my = 1 - (y / (float)winHeight) * 2;
		
		nodo* clicado = nullptr;
		for(nodo* n : nodos) {
			float dx = n->x - mx;
			float dy = n->y - my;
			if(sqrt(dx*dx + dy*dy) < radioNodo) {
				clicado = n;
				break;
			}
		}
		
		if(clicado) {
			if(nodoSeleccionado) {
				if(nodoSeleccionado != clicado) {
					conectarNodos(nodoSeleccionado, clicado);
					aristas.push_back({nodoSeleccionado, clicado});
					
					if(modoColoracion == 1) coloracionRestrictiva();
					else coloracionRestringida();
				}
				nodoSeleccionado = nullptr;
			} else {
				nodoSeleccionado = clicado;
			}
		} else {
			nodos.push_back(new nodo('A' + nodos.size(), mx, my));
			if(modoColoracion == 1) coloracionRestrictiva();
			else coloracionRestringida();
		}
		
		glutPostRedisplay();
	}
}

void teclado(unsigned char key, int x, int y) {
	switch(key) {
	case '1':
		modoColoracion = 1;
		coloracionRestrictiva();
		break;
	case '2':
		modoColoracion = 2;
		coloracionRestringida();
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(800, 800);
	glutCreateWindow("Grafo");
	
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutKeyboardFunc(teclado);
	
	glutMainLoop();
	return 0;
}
