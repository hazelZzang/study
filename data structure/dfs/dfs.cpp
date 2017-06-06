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

		//�ֺ� ������ ��ȸ�Ѵ�.
		for (int i = 0; i < adj[now].size(); i++) {
			int next = adj[now][i];

			//�湮�� ���� ���� ��쿡�� ��ͷ� Ž��
			if (!visited[next]) {
				dfs(next);
				//��������
				tpSort.push_back(next);
			}
		}
	}
	//��� �������� �����Ѵ�.
	//�̾����� ���� �׷����� ���� �� �ֱ� ����
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