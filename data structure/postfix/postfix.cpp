#define __POSTFIX_CPP__
#ifdef __POSTFIX_CPP__

#include "stackFile.cpp"


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


void postfix(Stack& s) {
	Stack temp;

	while (!s.isEmpty()) {
		char input = s.top();

		if (isdigit(input)) { // operand
			cout << input;
		}
		else if (input == '(') {	//left paren
			temp.push(input);
		}
		else if (input == ')') {	//right paren
			while (temp.top() != '(') {
				cout << temp.pop();
			}
			temp.pop(); //pop '('
		}
		else {	//operator
			if (token(input) > token(temp.top())) {	//input priority is bigger
				temp.push(input);
			}
			else {
				while ((!temp.isEmpty()) && token(input) <= token(temp.top())) {	//stack top priority is bigger
					cout << temp.pop();
				}
				temp.push(input);
			}
		}
		s.pop();
	}
	while (!temp.isEmpty()) {	//left in stack
		cout << temp.pop();
	}
}


int main(){
	Stack s(20);
	char * test = new char[20];
	cin >> test;
	s.pushAll(test);
	postfix(s);
	delete[] test;
}
 
#endif