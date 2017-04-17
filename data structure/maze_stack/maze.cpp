#include "maze.h"


int ROW[8] = { -1,-1,0,1,1,1,0,-1 };
int COL[8] = { 0,1,1,1,0,-1,-1,-1 };


Node::Node()
		:row(0), col(0), dir(0) {
	}

	//Node 값 추가
Node::Node(int r, int c, int d)
		:row(r), col(c), dir(d) {
	}


Stack::Stack(int d) {
		top = -1;
		len = d;
		arr.resize(d);
	}
	void Stack::push(Node v) {
		if (!isFull()) {
			arr[++top] = v;
		}
	}
	Node Stack::pop() {
		Node temp;
		if (isEmpty()) {
			return temp;
		}
		return arr[top--];
	}
	bool Stack::isEmpty() {
		if (top < 0) return true;
		else return false;
	}
	bool Stack::isFull() {
		if (len - 1 <= top) return true;
		else return false;
	}
	void Stack::print() {
		if (isEmpty()) return;

		for (int i = 0; i <= top; i++) {
			cout << arr[i].row << ' ' << arr[i].col << endl;
		}
	}

	//변수 초기화
	Maze::Maze(vector <vector <int>> m)
		:path(100) {
		found = 0;
		maze = m;
		//미로를 초기화 한다.
		for (int i = 0; i < m.size(); i++) {
			vector<int> temp;
			temp.resize(m[0].size());
			fill(temp.begin(), temp.end(), 0);
			mark.push_back(temp);
		}
		//시작 지점 체크
		mark[0][0] = 1;

		//초기 path Node 의 상태
		Node temp(0, 0, E);
		path.push(temp);

	}

	//미로의 출구를 확인한다.
	void Maze::find() {
		while (!path.isEmpty() && !found) {
			Node temp;
			temp = path.pop();
			int row = temp.row, col = temp.col, dir = temp.dir;

			//주변을 다 확인하기 전까지 반복
			while (dir < 8 && !found) {

				int nextRow = row + ROW[dir];
				int nextCol = col + COL[dir];

				if (nextRow < 0 || nextCol < 0 || MAX <= nextRow || MAX <= nextRow) {
					dir++;
				}
				//나아갈 수 있는 길 
				else if (!maze[nextRow][nextCol] && !mark[nextRow][nextCol]) {

					//exit
					if (nextRow == 5 && nextCol == 5) {
						found = 1;
					}

					mark[nextRow][nextCol] = 1;
					temp.row = row, temp.col = col, temp.dir = dir + 1;
					path.push(temp);
					row = nextRow, col = nextCol, dir = N;
				}
				else dir++;
			}
		}
	}
	//경로를 출력한다.
	void Maze::printPath() {
		path.print();
	}
	//미로를 출력한다.
	void Maze::printMaze() {
		for (int i = 0; i < maze.size(); i++) {
			for (int j = 0; j < maze[0].size(); j++) {
				cout << maze[i][j] << ' ';
			}
			cout << endl;
		}
	}

	//방문한 길을 출력한다.
	void Maze::printMark() {
		for (int i = 0; i < mark.size(); i++) {
			for (int j = 0; j < mark[0].size(); j++) {
				cout << mark[i][j] << ' ';
			}
			cout << endl;
		}
	}
