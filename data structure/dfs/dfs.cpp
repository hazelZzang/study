#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph{
private:
	int V;
	vector <int> *adj;
	vector <bool> visited;
	vector <int> tpSort;
public:
	Graph(int vertex)
	:V(vertex){
		adj = new vector<int>[V];
		visited = vector<bool>(V, 0);
	}
	void addEdge(int from, int to) {
		adj[from].push_back(to);
	}
	void dfs(int now) {
		visited[now] = 1;

		//주변 정점을 순회한다.
		for (int i = 0; i < adj[now].size(); i++) {
			int next = adj[now][i];

			//방문한 적이 없는 경우에는 재귀로 탐색
			if (!visited[next]) {
				dfs(next);
				//위상정렬
				tpSort.push_back(next);
			}
		}
	}
	//모든 정점에서 시작한다.
	//이어지지 않은 그래프가 있을 수 있기 때문
	void dfsAll() {
		for (int i = 0; i < V; i++) {
			if (!visited[i]) {
				dfs(i);
			}
		}
	}
	void printTopologicalSort() {
		for (int i = 0; i < tpSort.size(); i++) {
			cout << tpSort[i] << " ";
		}
		cout << endl;
	}
	~Graph() {
		delete[] adj;
	}
};

int main() {
	Graph g(5);
	g.addEdge(0, 1);
	g.addEdge(1, 3);
	g.addEdge(1, 2);
	g.addEdge(2, 4);
	g.dfsAll();
	g.printTopologicalSort();
}