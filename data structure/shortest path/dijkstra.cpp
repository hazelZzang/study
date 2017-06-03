/*
20143051 김혜지

Homework 5. Shortest Path

Algorithm :
시작점에서 방문하지 않은 정점 중 가중치가 가장 작은 곳을 방문하며 각 정점의 최단 길이를 찾아낸다.
*/

#include <iostream>
#include <vector>
#include <queue>
#define MAX 100;
using namespace std;

class DIJK {
private:
	int V;
	vector <pair <int, int> > *adj;
	vector <int> dst;

public:
	DIJK(vector <vector<int> > cMat) {
		V = cMat.size();
		adj = new vector< pair< int, int> > [V];

		//인접 리스트로 구현한다.
		for(int i = 0; i < cMat.size(); i++){
			for (int j = 0; j < cMat[i].size(); j++) {
				if(cMat[i][j] != 100)
					adj[i].push_back({ cMat[i][j], j });
			}
		}

		for (int i = 0; i < V; i++) {
			int num = MAX;
			dst.push_back(num);
		}
	}
	void shortestPath() {
		priority_queue <pair<int, int> > state;

		//초기 지점 입력
		state.push({ 0,0 });
		dst[0] = 0;
		while (!state.empty()) {
			//시작점에서 가장 가까운 vertex를 선택한다.
			int now = state.top().second;
			//우선 순위 큐는 큰 순서로 정렬하기 때문에, 부호를 붙여주어야 한다.
			int now_dst = -state.top().first; 
			state.pop();

			//중복되는 경우
			if (dst[now] < now_dst) continue;
			
			//vertex 인근 edge를 모두 검사하며 가중치를 갱신한다.
			for (int i = 0; i < adj[now].size(); i++) {
				int next = adj[now][i].second;
				int next_dst = adj[now][i].first + now_dst;

				//now를 거쳐갈 경우가 거리가 더 작아 갱신하는 경우
				if (dst[next] > next_dst) {
					dst[next] = next_dst;
					//우선 순위 큐에 추가해준다.
					state.push({ -next_dst, next });
				}
			}
			cout << now << ":";
			printDst();
		}

	}

	void printDst() {
		for (int i = 0; i < dst.size(); i++) {
			cout << dst[i] <<' ';
		}
		cout << endl;
	}
};

int main() {
	vector <vector<int> > cMat(7, vector<int>(7, 0));
	cMat = { 
	{100,2,4,5,100,100,100},
	{100,100,100,2,7,100,100},
	{100,100,100,1,100,4,100},
	{100,2,1,100,4,3,100},
	{100,7,100,4,100,1,5},
	{100,100,4,3,1,100,7},
	{100,100,100,100,7,5,100} };

	DIJK d(cMat);
	d.shortestPath();
}