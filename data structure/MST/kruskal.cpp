#include <iostream>
#include <queue>
#include <algorithm>
#define MAX 7
using namespace std;

class Disjoint {
	int *parent, *rnk;
	int vertex;
	Disjoint(int V) {
		vertex = V;
		parent = new int[vertex + 1];
		rnk = new int[vertex + 1];

		for (int i = 1; i <= vertex; i++) {
			//초기 부모는 나 자신
			parent[i] = i;
			rnk[i] = 0;
		}
	}
	
	int find(int u) {
		//재귀로 최상부모를 구한다
		if (u != parent[u])
			parent[u] = find(parent[u]);

		return parent[u];
	}

	//두 tree를 합친다.
	void merge(int u, int v) {
		u = find(u), v = find(v);
		if (rnk[u] > rnk[v])
			parent[v] = u;
		else
			parent[u] = v;

		if (rnk[u] == rnk[v])
			rnk[v]++;
	}

	friend class Graph;
};

class Graph {
private:
	int V, E;
	vector <pair<int, pair<int, int>> > vec;
public:
	Graph(int vertex)
	:V(vertex), E(0){
	}

	//가중치 그래프행렬을 출력한다.
	void printGraph() {
		int **graph = new int*[V + 1];
		for (int i = 0; i <= V; i++) {
			graph[i] = new int[V + 1];
		}
		for (int i = 1; i <= V; i++) {
			for (int j = 1; j <= V + 1; j++) {
				graph[i][j] = 100;
			}
		}
	
		for (vector<pair<int, pair<int, int>>>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter) {
			pair<int, pair<int, int>> result = *iter;
			graph[result.second.first][result.second.second] = result.first;
		}

		for (int i = 1; i <= V; i++) {
			for (int j = 1; j <= V; j++) {
				cout << graph[i][j] << ' ';
			}
			cout << endl;
		}

	}
	//그래프 edge 추가
	void addEdge(int w, int u, int v) {
		vec.push_back({w,{u,v}});
		E++;
	}

	//가장 최소의 가중치를 지닌 edge를 골라 트리를 제작한다.
	//cycle 확인을 해야한다
	void getKruskal(){
		queue <pair<int,int>> q;
		Disjoint dj(V);

		//가중치에 따라서 정렬한다.
		sort(vec.begin(), vec.end());

		//spanning tree의 edge는 vertex - 1개이다.
		while (q.size() < V - 1) {
			for (int i = 0; i < E; i++) {

				int from = vec[i].second.first;
				int to = vec[i].second.second;

				int f_parent = dj.find(from);
				int t_parent = dj.find(to);

				//cycle check
				if (f_parent != t_parent) {
					q.push({ from, to });
					cout << from << "-" << to << endl;

					dj.merge(f_parent, t_parent);
				}
			}
		}
	}
	  
};

int main() {
	Graph g(6);
	g.addEdge(6, 1, 2);
	g.addEdge(1, 1, 3);
	g.addEdge(5, 1, 4);
	g.addEdge(6, 2, 1);
	g.addEdge(4, 2, 3);
	g.addEdge(3, 2, 5);
	g.addEdge(1, 3, 1);
	g.addEdge(4, 3, 2);
	g.addEdge(5, 3, 4);
	g.addEdge(6, 3, 5);
	g.addEdge(5, 3, 6);
	g.addEdge(5, 4, 1);
	g.addEdge(5, 4, 3);
	g.addEdge(2, 4, 6);
	g.addEdge(3, 5, 2);
	g.addEdge(6, 5, 3);
	g.addEdge(6, 5, 6);
	g.addEdge(5, 6, 3);
	g.addEdge(2, 6, 4);
	g.addEdge(6, 6, 5);



	g.getKruskal();
	g.printGraph();
}
