//트리 생성 및 순회, 계산 결과 출력
// inorder, postorder, preorder

#include<iostream>
#include <string>
#include <cmath>
using namespace std;

//연산자 우선순위
char prec[5][2] = { '^',3, '*',2, '/',2,'+',1,'-',1 };


//Tree의 Node
class Node {
private:
	//계산 데이터
	char data;
	//연산자 우선 순위 저장
	int prio;
	Node *left;
	Node *right;

	friend class Tree;
	
	Node(char d) {
		data = d;
		prio = 4;
		left = NULL;
		right = NULL;
	}
};


class Tree {
private:
	Node *root;
public:
	Tree() { root = 0; }

	//input : 계산식
	// 트리를 생성한다.
	void buildTree(string input) {
		int count = 0;
		while (input[count] != NULL) {
			Node *n = new Node(input[count++]);
			int i = 0;
			for (; i < 5; i++) {
				if (n->data == prec[i][0]) {
					n->prio = prec[i][1];
					break;
				}
			}
			if (i == 5) {
				operandNode(n);
			}
			else {
				operatorNode(n);
			}
		}
	}

	//세가지 방법으로 트리를 출력한다.
	void print() {
		cout << "inorder : ";
		inorderTree(root);
		cout << "postorder :";
		postorderTree(root);
		cout << "preorder : ";
		preorderTree(root);
		cout << endl;
	}
	//8+9-2*3
	Node* inorderTree(Node *n) {
		if (n != NULL) {
			inorderTree(n->left);
			cout << n->data << ' ';
			inorderTree(n->right);
		}
		return n;
	}
	//8 9 + 2 3 * -
	Node* postorderTree(Node *n) {
		if (n != NULL) {
			postorderTree(n->left);
			postorderTree(n->right);
			cout << n->data << ' ';
		}
		return n;
	}
	//- + 8 9 * 2 3
	Node* preorderTree(Node *n) {
		if (n != NULL) {
			cout << n->data << ' ';
			preorderTree(n->left);
			preorderTree(n->right);
		}
		return n;
	}

	//data가 operand일 때
	//root가 없는 경우는 root로 지정,
	//root가 존재하는 경우는, 가장 오른쪽에 저장한다.
	void operandNode(Node *n) {
		if (root == NULL) {
			root = n;
			return;
		}
		Node *p = root;
		while (p->right != NULL) {
			p = p->right;
		}
		p->right = n;
	}

	//data가 operator일 때,
	//root와 data의 연산자 우선 순위를 비교한다.
	//root가 우선순위가 높은 경우
	//	data Node의 왼쪽으로 지정하고, data Node가 root가 된다.
	//그렇지 않은 경우
	//	root의 right를 data Node의 왼쪽으로 지정하고, root의 오른쪽에 data Node가 위치한다.
	//왼쪽 자식 노드 -> 오른쪽 자식 노드 -> 부모 노드 순서
	void operatorNode(Node *n) {
		if (root->prio >= n->prio) {
			n->left = root;
			root = n;
		}
		else {
			n->left = root->right;
			root->right = n;
		}
	}
	int eval() {
		return evalTree(root);
	}
	//리프 노드까지 도달하기 위해 재귀함수를 사용한다.
	//왼쪽 자식 노드와 오른쪽 자식 노드를 계산한다.
	int evalTree(Node *p) {
		int value = 0; 
		if (p != NULL) {
			if (isdigit(p->data)) value = p->data - '0';
			else {
				int left = evalTree(p->left);
				int right = evalTree(p->right);
				switch (p->data) {
				case'+':
					value = left + right;
					break;
				case'-':
					value = left - right;
					break;
				case'*':
					value = left * right;
					break;
				case'/':
					value = left / right;
					break;
				case'^':
					value = pow(left,right);
					break;
				}
			}
		}
		else {
			cout << "empty tree";
		}
		return value; 
	}
	~Tree(){}
};

int main() {
	string input;
	Tree t;
	cin >> input;
	t.buildTree(input);
	t.print();
	cout << "Output :" << t.eval();
	
}