/*
lab 13. hash table
ADT �Լ� : 1) findkey, 2) insertKey 3) deleteKey 4) prinTable
Linear Probing
*/

#include <iostream>
#include <vector>
using namespace std;

class HASH {
private:
	int maxSize;
	vector <pair<int, bool>> hTable;
public:
	HASH(int max)
	:maxSize(max){
		// data, ������ ����
		hTable = vector <pair<int, bool>>(max, { 0,0 });
	}
	bool findKey(int key) {
		int index = hashing(key);
		if (hTable[index] == pair<int, bool>(key, 1))
			return true;

		return false;		
	}

	int findKeyLinear(int key) {
		int index = hashing(key);

		for (int i = 0; i < maxSize; i++) {
			if (hTable[index] == pair<int, bool> (key, 1))
				return index;
			index = (index + 1) % maxSize;
		}
		return -1;
	}

	bool isEmpty() {
		for (int i = 0; i < hTable.size(); i++) {
			if (hTable[i].second == false)
				return true;
		}
		return false;
	}

	bool insertKey(int key) {
		if (!isEmpty()) return false;
		
		int index = hashing(key);
		int findIndex = findKeyLinear(key);

		//�̹� key�� �����ϴ� ���
		if (findIndex != -1) {
			return false;
		}
		else {
			//�ش� �ڸ��� �ٸ� ���� �����ϸ�
			//���� ĭ�� �����Ѵ�.
			for (int i = 0; i < maxSize; i++) {
				if (hTable[index].second == false) {
					hTable[index] = { key, true };
					return true;
				}
				index = (index + 1) % maxSize;
			}
			return false;
		}
	}
	bool deleteKey(int key) {
		int findIndex = findKeyLinear(key);

		if (findIndex != -1) {
			hTable[findIndex].second = false;
			return true;
		}
		return false;
	}

	int hashing(int key) {
		return key%maxSize;
	}

	void printTable() {
		for (int i = 0; i < hTable.size(); i++) {
			if(hTable[i].second)
				cout << hTable[i].first << " ";
			else cout << " _ ";
		}
		cout << endl;
	}
};

int main() {
	HASH h(7);
	while (1) {
		cout << "1. insert 2.find 3.delete" << endl;
		int num, data;
		cin >> num;
		switch (num) {
		case 1:
			cout << "data :";
			cin >> data;
			if (!h.insertKey(data)) {
				cout << "Couldn't insert" << endl;
			}
			break;
		case 2:
			cout << "data :";
			cin >> data;
			if(h.findKeyLinear(data) != -1) cout << "FOUND " <<data<<endl;
			else cout << "Not found" << endl;
			break;
		case 3:
			cout << "data :";
			cin >> data;
			if (!h.deleteKey(data)){
				cout << "Not found" << endl;
			}
			break;
		}
		h.printTable();
	}
}