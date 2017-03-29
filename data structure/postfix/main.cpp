#include "stack.h"
#include <iostream>
#include <string>
using namespace std;

int token(char t) {
	switch (t) {

	case ')':
		return 3;
		break;

	case '*': case '/': case '%':
		return 2;
		break;

	case '+': case '-':
		return 1;
		break;

	case '(':
		return 0;
		break;
	}
	return 0;
}


string postfix(Stack& s) {
	Stack temp;
	string result;

	int count = 0;
	while (!s.isEmpty()) {
		char input = s.top();

		if (isdigit(input)) { // operand
			result += input;
		}
		else if (input == '(') {	//left paren
			temp.push(input);
		}
		else if (input == ')') {	//right paren
			while (temp.top() != '(') {
				result += temp.pop();
			}
			temp.pop(); //pop '('
		}
		else {	//operator
			if (token(input) > token(temp.top())) {	//input priority is bigger
				temp.push(input);
			}
			else {
				while ((!temp.isEmpty()) && token(input) <= token(temp.top())) {	//stack top priority is bigger
					result += temp.pop();
				}
				temp.push(input);
			}
		}
		s.pop();
	}
	while (!temp.isEmpty()) {	//left in stack
		result += temp.pop();
	}
	return result;
}

char eval(Stack& s) {
	char op1, op2;
	int n;
	int top = -1;
	char token;
	Stack temp;
	while (token = s.pop()) {
		if (isdigit(token)) {
			temp.push(token);
		}
		else {
			op2 = temp.pop();
			op1 = temp.pop();
			switch (token) {
			case '+':
				temp.push(op1 + op2);
				break;
			case '-':
				temp.push(op1 - op2);
				break;
			case '*':
				temp.push(op1 * op2);
				break;
			case '/':
				temp.push(op1 / op2);
				break;
			}
		}
	}
	return temp.pop();
}

int main() {
	string test = "3+4-(9+1)";
	string pf;
	Stack i(20), p(20);
	i.pushAll(test);

	//infix to postfix
	pf = postfix(i);
	cout << pf << endl;

	p.pushAll(pf);
	
	//value
	cout << int(eval(p))<<endl;
	
	return 0;
}
