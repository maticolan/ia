#include <GL/gl.h> 
#include <GL/glu.h> 
#include <GL/freeglut.h>
#include <vector>
#include <stack>
#include <cmath>
#include <ctime>
#include <queue>
#include <algorithm>
#include <thread>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <unistd.h> 

using namespace std;

pair<int, int> mouseCoords;
bool clicked = false;
int numPoints = 100; // Valor por defecto
int nearestNodes = 5; // Valor por defecto
bool removido = false;

struct node {
	int x, y;
	vector<node*> edges;
	float gCost; // Coste desde el nodo inicial
	float hCost; // Coste estimado hasta el nodo final
	node* parent; // Nodo anterior en la ruta más corta

	node() : gCost(0), hCost(0), parent(nullptr) {}

	void setCoord(int x, int y) {
		this->x = x;
		this->y = y;
	}

	void addEdge(node* n) {
		if (n == this) return;
		for (auto it : edges) {
			if (it == n) return;
		}
		edges.push_back(n);
		n->edges.push_back(this);
	}

	float fCost() const { return gCost + hCost; }
};

struct nodeD {
	float distance;
	node* coord;
	nodeD* next;
	nodeD(float d, node* c = 0, nodeD* n = 0) {
		distance = d;
		next = n;
		coord = c;
	}
};

struct sortedLinkedList {
	nodeD* head;
	int size;
	sortedLinkedList(int s) {
		size = s;
		head = new nodeD(numeric_limits<float>::max());
		nodeD* n = head;
		while (s - 1) {
			n->next = new nodeD(numeric_limits<float>::max());
			n = n->next;
			s--;
		}
	}
	void insert(float distance, node* coordinate) {
		if (distance < head->distance) {
			nodeD* n = head;
			while (n->next && n->next->distance > distance) {
				n = n->next;
			}
			n->next = new nodeD(distance, coordinate, n->next);
			nodeD* tmp = head;
			head = head->next;
			delete tmp;
		}
	}

	~sortedLinkedList() {
		while (head) {
			nodeD* tmp = head;
			head = head->next;
			delete tmp;
		}
	}
};

vector<node> points;
vector<node*> lonelyNodes; // Lista de nodos sin conexiones
bool animating = false;

void init() {
	glClearColor(0.0, 0.0, 0.0, 1.0); // Fondo negro
	glColor3f(0.5, 0.5, 0.5); // Color plomo para los puntos
	glPointSize(5.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, 1000.0, 0.0, 1000.0); // Configura el sistema de coordenadas
}

vector<node> genRandCoords(int n, int minX, int maxX, int minY, int maxY) {
	int size = (maxX - minX + 1) * (maxY - minY + 1);
	vector<node> vec(size);
	int x = minX, y = minY;
	int i = 0;
	while (i < size) {
		if (x < maxX) {
			vec[i].setCoord(x, y);
			x++;
		}
		else {
			vec[i].setCoord(x, y);
			y++;
			x = minX;
		}
		i++;
	}
	srand(time(NULL));
	int last = size - 1;
	while (n) {
		int randIndex = rand() % size;
		swap(vec[randIndex], vec[last]);
		last--;
		n--;
	}
	return vector<node>(vec.begin() + (last + 1), vec.end());
}

float squaredDistance(node a, node b) {
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

bool isPointInNode(int x, int y, node& n, int threshold = 5) {
	return abs(n.x - x) <= threshold && abs(n.y - y) <= threshold;
}

node* findNodeAtClick(int x, int y, vector<node>& coordinates, int threshold = 5) {
	for (auto& node : coordinates) {
		if (isPointInNode(x, y, node, threshold)) {
			return &node;
		}
	}
	return nullptr;
}

void drawLineBetweenNodes(node* a, node* b) {
	glBegin(GL_LINES);
	glVertex2i(a->x, a->y);
	glVertex2i(b->x, b->y);
	glEnd();
	glutSwapBuffers();
}
void drawHighlightedNode(node* n, float r, float g, float b) {
	glColor3f(r, g, b);
	glBegin(GL_POINTS);
	glVertex2i(n->x, n->y);
	glEnd();
	glutSwapBuffers();
}

void linkClosestPoints(node* startNode, vector<node>& coordinates, int n) {
	sortedLinkedList closest(n);
	for (auto& node : coordinates) {
		if (&node != startNode) {
			float dist = squaredDistance(*startNode, node);
			closest.insert(dist, &node);
		}
	}

	nodeD* it = closest.head;
	while (it) {
		startNode->addEdge(it->coord);
		drawLineBetweenNodes(startNode, it->coord);
		it = it->next;
	}
}

pair<int, int> getMouseCoordinates(int x, int y) {
	int viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	int realY = viewport[3] - y;
	return { x, realY };
}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		mouseCoords = getMouseCoordinates(x, y);
		clicked = true;
	}
}

pair<int, int> waitForClick(vector<node>& coordinates) {
	clicked = false;
	while (true) {
		glutMainLoopEvent();
		if (clicked) {
			node* clickedNode = findNodeAtClick(mouseCoords.first, mouseCoords.second, coordinates);
			if (clickedNode) {
				return { clickedNode->x, clickedNode->y };
			}
		}
	}
}

void findLonelyNodes(vector<node>& coordinates) {
	for (auto& p : coordinates) {
		if (p.edges.empty()) {
			lonelyNodes.push_back(&p);
		}
	}
}

void connectLonelyNodes(vector<node*>& lonelyNodes, vector<node>& coordinates, int n) {
	for (auto& ln : lonelyNodes) {
		linkClosestPoints(ln, coordinates, n);
	}
}


// Función para dibujar la arista entre dos nodos de color
void drawColoredLineBetweenNodes(node* a, node* b, float r, float g, float k) {
	glColor3f(r, g, k);
	glBegin(GL_LINES);
	glVertex2i(a->x, a->y);
	glVertex2i(b->x, b->y);
	glEnd();
	glutSwapBuffers();
}

void recDFS(node*& end, vector<node*>& route, bool& done, unordered_set<node*>& visited);

vector<node*> DFS(node* start, node* end) {
	vector<node*> route;
	unordered_set<node*> visited;
	visited.emplace(start);
	route.push_back(start);
	bool done = route.back() == end;
	recDFS(end, route, done, visited);
	return route;
}

void recDFS(node*& end, vector<node*>& route, bool& done, unordered_set<node*>& visited) {
	if (done) return;
	if (route.back() == end) {
		done = true;
		return;
	}
	for (int j = 0; j < route.back()->edges.size(); j++) {
		if (visited.emplace(route.back()->edges[j]).second) {
			route.push_back(route.back()->edges[j]);

			drawColoredLineBetweenNodes(route[route.size() - 2], route.back(), 0.0f, 0.0f, 1.0f);

			recDFS(end, route, done, visited);
			if (done) return;
		}
	}
	route.pop_back();
}

vector<node*> BFS(node* start, node* end) {
	unordered_set<node*> visited;
	queue<pair<node*, std::stack<node*>>> route;
	visited.emplace(start);
	route.push(pair<node*, stack<node*>>(start, stack<node*>()));
	route.front().second.push(start);

	while (route.front().first != end) {
		for (int i = 0; i < route.front().first->edges.size(); i++) {
			if (visited.emplace(route.front().first->edges[i]).second) {
				stack<node*> s(route.front().second);
				s.push(route.front().first->edges[i]);
				route.push(pair<node*, stack<node*>>(route.front().first->edges[i], s));

				drawColoredLineBetweenNodes(route.front().first, route.front().first->edges[i], 0.0f, 0.0f, 1.0f);
			}
		}
		route.pop();
	}

	node* aux = route.front().first;
	int s = route.front().second.size();
	vector<node*> ans(s);
	while (s--) {
		ans[s] = route.front().second.top();
		route.front().second.pop();
	}
	return ans;
}

vector<node*> HillClimbing(node* start, node* end) {
	unordered_set<node*> visited;
	vector<node*> route;
	node* current = start;
	route.push_back(current);
	visited.emplace(current);

	while (current != end) {
		vector<pair<node*, float>> neighbors;

		for (node* neighbor : current->edges) {
			if (visited.find(neighbor) == visited.end()) {
				float dist = squaredDistance(*neighbor, *end);
				neighbors.push_back({ neighbor, dist });
			}
		}

		if (neighbors.empty()) {
			while (!route.empty() && neighbors.empty()) {
				route.pop_back(); // Retrocede
				if (!route.empty()) {
					current = route.back(); // Vuelve al nodo anterior
					for (node* neighbor : current->edges) {
						if (visited.find(neighbor) == visited.end()) {
							float dist = squaredDistance(*neighbor, *end);
							neighbors.push_back({ neighbor, dist });
						}
					}
					sort(neighbors.begin(), neighbors.end(), [](const pair<node*, float>& a, const pair<node*, float>& b) {
						return a.second < b.second;
						});
				}
			}
			if (route.empty()) {
				break; // No se encontró una ruta
			}
		}

		sort(neighbors.begin(), neighbors.end(), [](const pair<node*, float>& a, const pair<node*, float>& b) {
			return a.second < b.second;
			});

		if (!neighbors.empty()) {
			current = neighbors.front().first;
			route.push_back(current);
			visited.emplace(current);
			drawColoredLineBetweenNodes(route[route.size() - 2], current, 0.0f, 0.0f, 1.0f);
			usleep(80000);
		}
	}

	if (current == end) {
		return route;
	}
	return {}; // Retornar vacío si no se encontró una ruta
}

float heuristic(node* a, node* b) {
	return sqrt(squaredDistance(*a, *b));
}

vector<node*> AStar(node* start, node* end) {
	auto compare = [](node* lhs, node* rhs) { return lhs->fCost() > rhs->fCost(); };
	priority_queue<node*, vector<node*>, decltype(compare)> openSet(compare);
	unordered_set<node*> closedSet;
	unordered_map<node*, float> gCosts;
	unordered_map<node*, node*> parents;

	openSet.push(start);
	gCosts[start] = 0;
	parents[start] = nullptr;

	while (!openSet.empty()) {
		node* current = openSet.top();
		openSet.pop();

		if (closedSet.find(current) != closedSet.end()) continue;

		if (current == end) {
			vector<node*> path;
			while (current) {
				path.push_back(current);
				current = parents[current];
			}
			reverse(path.begin(), path.end());
			return path;
		}

		closedSet.insert(current);

		for (node* neighbor : current->edges) {
			float tentativeGCost = gCosts[current] + sqrt(squaredDistance(*current, *neighbor));

			if (gCosts.find(neighbor) == gCosts.end() || tentativeGCost < gCosts[neighbor]) {
				gCosts[neighbor] = tentativeGCost;
				neighbor->gCost = tentativeGCost; // Si gCost es un atributo del nodo
				neighbor->hCost = heuristic(neighbor, end);
				parents[neighbor] = current;

				// Solo añadimos a la priority_queue si el costo mejora
				openSet.push(neighbor);

				// Visualización condicional

				drawColoredLineBetweenNodes(current, neighbor, 0.0f, 0.0f, 1.0f);
				usleep(5000);

			}
		}
	}

	return vector<node*>(); // Si no se encuentra ruta
}



std::vector<node*> Dijkstra(node* start, node* end) {
	auto compare = [](node* lhs, node* rhs) { return lhs->gCost > rhs->gCost; };
	priority_queue<node*, vector<node*>, decltype(compare)> openSet(compare);
	unordered_map<node*, float> gCosts;
	unordered_map<node*, node*> parents;
	unordered_set<node*> closedSet;

	start->gCost = 0;
	openSet.push(start);
	gCosts[start] = 0;
	parents[start] = nullptr;

	while (!openSet.empty()) {
		node* current = openSet.top();
		openSet.pop();

		if (current == end) {
			vector<node*> path;
			while (current) {
				path.push_back(current);
				current = parents[current];
			}
			reverse(path.begin(), path.end());
			return path;
		}

		closedSet.insert(current);

		for (node* neighbor : current->edges) {
			if (closedSet.find(neighbor) != closedSet.end()) continue;

			float tentativeGCost = gCosts[current] + sqrt(squaredDistance(*current, *neighbor));
			if (gCosts.find(neighbor) == gCosts.end() || tentativeGCost < gCosts[neighbor]) {
				gCosts[neighbor] = tentativeGCost;
				neighbor->gCost = tentativeGCost;
				parents[neighbor] = current;

				openSet.push(neighbor);

				drawColoredLineBetweenNodes(current, neighbor, 0.0f, 0.0f, 1.0f);
				usleep(50000);
			}
		}
	}

	return vector<node*>(); // Si no se encuentra ruta
}


void chooseAndRunSearch() {
	int option;
	cout << "Elige el tipo de búsqueda: (1) DFS, (2) BFS, (3) Hill Climbing, (4) A* , (5) Dijkstra";
	cin >> option;

	cout << "Selecciona el nodo de inicio con el mouse...\n";

	pair<int, int> coords1 = waitForClick(points);
	node* startNode = findNodeAtClick(coords1.first, coords1.second, points);
	cout << "NODO INICIAL -> (" << startNode->x << "," << startNode->y << ")" << endl;
	drawHighlightedNode(startNode, 1.0f, 0.0f, 0.0f);

	cout << "Selecciona el nodo de fin con el mouse...\n";
	pair<int, int> coords2 = waitForClick(points);
	node* endNode = findNodeAtClick(coords2.first, coords2.second, points);
	cout << "NODO FINAL -> (" << endNode->x << "," << endNode->y << ")" << endl;
	drawHighlightedNode(endNode, 1.0f, 0.0f, 0.0f);

	vector<node*> resultRoute;
	if (option == 1) {
		// Ejecutar DFS
		resultRoute = DFS(startNode, endNode);
	}
	else if (option == 2) {
		// Ejecutar BFS
		resultRoute = BFS(startNode, endNode);
	}
	else if (option == 3) {
		// Ejecutar Hill Climbing
		resultRoute = HillClimbing(startNode, endNode);
		if (resultRoute.empty()) {
			cout << "Hill Climbing no encontró un camino exitoso.\n";
			return;
		}
	}
	else if (option == 4) {
		resultRoute = AStar(startNode, endNode);
	}

	for (int i = 0; i < resultRoute.size() - 1; i++) {
		drawColoredLineBetweenNodes(resultRoute[i], resultRoute[i + 1], 1.0f, 0.0f, 0.0f);
	}

	cout << "Búsqueda completada y camino resaltado en rojo.\n";
}


void eliminarNodosPorcentaje(vector<node>& puntos, int porcentaje) {
	int cantidadAEliminar = (puntos.size() * porcentaje) / 100;
	srand(time(NULL));
	for (int i = 0; i < cantidadAEliminar; i++) {
		int indice = rand() % puntos.size();
		puntos.erase(puntos.begin() + indice);
	}
	cout << "Se eliminaron el " << porcentaje << "% de los nodos.\n";
}


void requestAndDeleteNodes() {
	// Prompt for percentage of nodes to delete
	float porcentaje;
	cout << "Ingrese el porcentaje de nodos a eliminar: ";
	cin >> porcentaje;

	// Show the current graph (with no deletions)
	cout << "Mostrando los nodos antes de eliminar...\n";
	glutPostRedisplay();
	usleep(1000000); // Wait for a second to view the original graph

	// Ask for confirmation
	char confirm;
	cout << "¿Desea eliminar el " << porcentaje << "% de los nodos? (s/n): ";
	cin >> confirm;

	if (confirm == 's' || confirm == 'S') {
		eliminarNodosPorcentaje(points, porcentaje);

		// Display the graph after deletion
		cout << "Mostrando los nodos después de eliminar...\n";
		glutPostRedisplay();
	}
	else {
		cout << "No se eliminaron nodos.\n";
	}
}



void display() {
	glClear(GL_COLOR_BUFFER_BIT);

	glColor3f(0.5, 0.5, 0.5);
	glBegin(GL_POINTS);
	for (auto& p : points) {
		glVertex2i(p.x, p.y);
	}
	glEnd();
	glutSwapBuffers(); // Mostrar puntos en pantalla

	if (!removido) {
		int porcentaje;
		cout << "¿Qué porcentaje de nodos deseas eliminar? (ejemplo: 30): ";
		cin >> porcentaje;

		// Eliminar los nodos según el porcentaje ingresado
		eliminarNodosPorcentaje(points, porcentaje);

		// Cambiar la bandera para que no se repita
		removido = true;

		// Ahora mostrar el mapa con los nodos eliminados, por ejemplo:
		cout << "Pantalla con nodos eliminados...\n";
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(0.5, 0.5, 0.5);
		glBegin(GL_POINTS);
		for (auto& p : points) {
			glVertex2i(p.x, p.y);
		}
		glEnd();
		glutSwapBuffers(); // Mostrar puntos en pantalla
	}

	pair<int, int> coords = waitForClick(points);

	node* clickedNode = findNodeAtClick(coords.first, coords.second, points);
	if (clickedNode) {
		unordered_set<node*> visitados;
		queue<node*> colita;
		colita.push(clickedNode);
		visitados.insert(clickedNode);

		while (!colita.empty()) {
			node* nodoactual = colita.front();
			colita.pop();

			// Conectar con los n más cercanos si no se han visitado
			linkClosestPoints(nodoactual, points, nearestNodes);

			for (node* vecino : nodoactual->edges) {
				if (visitados.find(vecino) == visitados.end()) {
					colita.push(vecino);
					visitados.insert(vecino);
				}
			}
		}
		cout << "Se han conectado todos los nodos con sus " << nearestNodes << " más cercanos." << endl;

		// Buscar nodos solitarios
		findLonelyNodes(points);

		if (!lonelyNodes.empty()) {
			cout << "Existen " << lonelyNodes.size() << " nodos solitarios. ¿Desea conectarlos con sus " << nearestNodes << " más cercanos? (s/n): ";
			char respuesta;
			cin >> respuesta;

			if (respuesta == 's' || respuesta == 'S') {
				connectLonelyNodes(lonelyNodes, points, nearestNodes);
				cout << "Se han conectado los nodos solitarios con sus " << nearestNodes << " más cercanos." << endl;
			}
			else {
				cout << "Los nodos solitarios permanecen sin conexión." << endl;
			}
		}
		else {
			cout << "No hay nodos solitarios." << endl;
		}
	}
	chooseAndRunSearch();
}


int main(int argc, char** argv) {
	// Pedir al usuario el número de puntos
	cout << "Ingrese el número de puntos (valor predeterminado 100): ";
	cin >> numPoints;
	points = genRandCoords(numPoints, 0, 1000, 0, 1000);

	// Pedir al usuario el número de nodos más cercanos para conectar
	cout << "Ingrese el número de nodos más cercanos para conectar (valor predeterminado 5): ";
	cin >> nearestNodes;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Conexion de puntos cercanos");
	init();
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMainLoop();
	return 0;
}