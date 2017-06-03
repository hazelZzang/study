/*
20143051 ������

Homework 5. Shortest Path

Algorithm :
���������� �湮���� ���� ���� �� ����ġ�� ���� ���� ���� �湮�ϸ� �� ������ �ִ� ���̸� ã�Ƴ���.
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

		//���� ����Ʈ�� �����Ѵ�.
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

		//�ʱ� ���� �Է�
		state.push({ 0,0 });
		dst[0] = 0;
		while (!state.empty()) {
			//���������� ���� ����� vertex�� �����Ѵ�.
			int now = state.top().second;
			//�켱 ���� ť�� ū ������ �����ϱ� ������, ��ȣ�� �ٿ��־�� �Ѵ�.
			int now_dst = -state.top().first; 
			state.pop();

			//�ߺ��Ǵ� ���
			if (dst[now] < now_dst) continue;
			
			//vertex �α� edge�� ��� �˻��ϸ� ����ġ�� �����Ѵ�.
			for (int i = 0; i < adj[now].size(); i++) {
				int next = adj[now][i].second;
				int next_dst = adj[now][i].first + now_dst;

				//now�� ���İ� ��찡 �Ÿ��� �� �۾� �����ϴ� ���
				if (dst[next] > next_dst) {
					dst[next] = next_dst;
					//�켱 ���� ť�� �߰����ش�.
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