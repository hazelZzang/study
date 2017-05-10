//Ʈ�� ���� �� ��ȸ, ��� ��� ���
// inorder, postorder, preorder

#include<iostream>
#include <string>
#include <cmath>
using namespace std;

//������ �켱����
char prec[5][2] = { '^',3, '*',2, '/',2,'+',1,'-',1 };


//Tree�� Node
class Node {
private:
	//��� ������
	char data;
	//������ �켱 ���� ����
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

	//input : ����
	// Ʈ���� �����Ѵ�.
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

	//������ ������� Ʈ���� ����Ѵ�.
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

	//data�� operand�� ��
	//root�� ���� ���� root�� ����,
	//root�� �����ϴ� ����, ���� �����ʿ� �����Ѵ�.
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

	//data�� operator�� ��,
	//root�� data�� ������ �켱 ������ ���Ѵ�.
	//root�� �켱������ ���� ���
	//	data Node�� �������� �����ϰ�, data Node�� root�� �ȴ�.
	//�׷��� ���� ���
	//	root�� right�� data Node�� �������� �����ϰ�, root�� �����ʿ� data Node�� ��ġ�Ѵ�.
	//���� �ڽ� ��� -> ������ �ڽ� ��� -> �θ� ��� ����
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
	//���� ������ �����ϱ� ���� ����Լ��� ����Ѵ�.
	//���� �ڽ� ���� ������ �ڽ� ��带 ����Ѵ�.
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