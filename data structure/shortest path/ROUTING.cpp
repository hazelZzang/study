#include <iostream>
#include <vector>
#include <queue>
#include <limits>
using namespace std;
class DJK {
private:
	int V;
	vector < pair<long double, int> > *adj;
	vector <long double> dst;
public:
	DJK(int vertex) {
		V = vertex;
		adj = new vector < pair <long double, int> >[V];

		for (int i = 0; i < V; i++) {
			dst.push_back(numeric_limits<long double>::max());
		}
	}
	void addEdge(int v1, int v2, long double w) {
		adj[v1].push_back({ w, v2 });
		adj[v2].push_back({ w, v1 });
	}
	long double getShortestPath() {
		priority_queue <pair <long double, int> > pq;
		pq.push({ -1.0,0 });
		dst[0] = 1.0;

		while (!pq.empty()) {
			int now = pq.top().second;
			long double now_w = -pq.top().first;
			pq.pop();

			if (dst[now] < now_w) continue;

			for (int i = 0; i < adj[now].size(); i++) {
				int next = adj[now][i].second;
				long double next_w = now_w * adj[now][i].first;

				if (dst[next] > next_w) {
					dst[next] = next_w;
					pq.push({ -next_w , next });
				}
			}

		}
		return dst[V - 1];
	}
};
int main() {
	int testCasesN;
	scanf_s("%d", &testCasesN);
	while (testCasesN--) {
		int v, e;
		scanf_s("%d %d", &v, &e);
		DJK d(v);
		while (e--) {
			int v1, v2;
			long double weight;
			scanf_s("%d %d %Lf\n", &v1, &v2, &weight);
			d.addEdge(v1, v2, weight);
		}
		printf("%.10Lf\n", d.getShortestPath());
	}
}